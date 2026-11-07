//! Host-request deferral evidence at engine boundaries (§9.2).

use std::sync::atomic::{AtomicU64, Ordering};

use crate::vm::EXIT_REQUESTED;

/// Work that must finish and release its engine locks before returning a yield.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[repr(u8)]
pub enum CriticalSectionKind {
    /// Decode and lift a single block, including fetch-policy callbacks.
    Lift,
    /// Synchronous backend materialization, including compiler lock waits and
    /// interpreter fallback. Background compiler threads are not vCPU work.
    Compile,
    /// Publish a cache entry or link, including the cache's publication locks.
    Publish,
    /// Submit background tier-up work or drain its completed results. This does
    /// not mean the vCPU waited for a background compile to finish.
    TierUpWait,
    /// Probe or lift a multi-block region, including fetch-policy callbacks.
    RegionBuild,
}

const ACTIVE_SHIFT: u32 = 3;
const KINDS_SHIFT: u32 = 8;
const COUNT_SHIFT: u32 = 16;
const KIND_MASK: u64 = 0x1f;
const ACTIVE_MASK: u64 = KIND_MASK << ACTIVE_SHIFT;

/// Owned evidence for a host-requested yield. Requests coalesce; the count is
/// completed engine sections that deferred a pending request, not requester calls.
/// A section entered with a request already pending also counts. Nested sections
/// count separately, and the kind set is their union. Duration is not measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InterruptionEvidence {
    kinds: u8,
    count: u64,
    path: crate::FaultPath,
    chain_link: bool,
    rip: u64,
}

impl Default for InterruptionEvidence {
    fn default() -> Self {
        Self {
            kinds: 0,
            count: 0,
            path: crate::FaultPath::Unknown,
            chain_link: false,
            rip: 0,
        }
    }
}

impl InterruptionEvidence {
    /// Boundary where the exit request was checked and taken. A helper is
    /// reported only for a check inside that helper (REP string iterations),
    /// never for a helper that has already returned. Dispatcher checks before
    /// selecting an execution unit report `FaultPath::Unknown`. Compiled
    /// boundaries use the function entry as their code-map key; a site is
    /// supplied only if that key has an exact code-map site.
    pub const fn path(&self) -> crate::FaultPath {
        self.path
    }

    /// Whether the checked boundary followed a resolved direct chain link,
    /// including a self-link. Internal region edges, helper iteration checks,
    /// and dispatcher resolution of an empty link do not count as chain links.
    pub const fn chain_link(&self) -> bool {
        self.chain_link
    }

    /// Architectural resume point, equal to the vCPU's RIP at this exit.
    pub const fn rip(&self) -> u64 {
        self.rip
    }

    pub(crate) fn at(mut self, path: crate::FaultPath, chain_link: bool, rip: u64) -> Self {
        self.path = path;
        self.chain_link = chain_link;
        self.rip = rip;
        self
    }

    pub const fn deferred(&self) -> bool {
        self.count != 0
    }

    pub const fn contains(&self, kind: CriticalSectionKind) -> bool {
        self.kinds & (1 << kind as u8) != 0
    }

    /// Number of completed sections that deferred the consumed request.
    /// Saturates at 2^48 - 1 until the request is consumed.
    pub const fn deferral_count(&self) -> u64 {
        self.count
    }
}

/// Only the owning vCPU writes `sections`; requesters and other notification
/// producers touch only the separate dispatch word. A section ends at its dispatch
/// load: a request ordered after that load belongs to the next boundary/section.
/// Cache hits and guest execution do not access the section word.
pub(crate) struct EngineSection<'a> {
    dispatch: &'a AtomicU64,
    sections: &'a AtomicU64,
    kind: CriticalSectionKind,
}

impl<'a> EngineSection<'a> {
    #[inline]
    pub(crate) fn enter(
        dispatch: &'a AtomicU64,
        sections: &'a AtomicU64,
        kind: CriticalSectionKind,
    ) -> Self {
        let previous = sections.load(Ordering::Relaxed);
        debug_assert_eq!(previous & (1 << (ACTIVE_SHIFT + kind as u32)), 0);
        sections.store(
            previous | (1 << (ACTIVE_SHIFT + kind as u32)),
            Ordering::Relaxed,
        );
        Self {
            dispatch,
            sections,
            kind,
        }
    }
}

impl Drop for EngineSection<'_> {
    #[inline]
    fn drop(&mut self) {
        let mut next =
            self.sections.load(Ordering::Relaxed) & !(1 << (ACTIVE_SHIFT + self.kind as u32));
        if self.dispatch.load(Ordering::Relaxed) & EXIT_REQUESTED != 0 {
            next |= 1 << (KINDS_SHIFT + self.kind as u32);
            if next >> COUNT_SHIFT != u64::MAX >> COUNT_SHIFT {
                next += 1 << COUNT_SHIFT;
            }
        }
        self.sections.store(next, Ordering::Relaxed);
    }
}

pub(crate) fn consume(dispatch: &AtomicU64, sections: &AtomicU64) -> InterruptionEvidence {
    // Acquire requester writes without clearing SMC or backend notifications.
    dispatch.fetch_and(!EXIT_REQUESTED, Ordering::Acquire);
    let previous = sections.load(Ordering::Relaxed);
    debug_assert_eq!(previous & ACTIVE_MASK, 0, "yield inside an engine section");
    sections.store(0, Ordering::Relaxed);
    InterruptionEvidence {
        kinds: ((previous >> KINDS_SHIFT) & KIND_MASK) as u8,
        count: previous >> COUNT_SHIFT,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KINDS: [CriticalSectionKind; 5] = [
        CriticalSectionKind::Lift,
        CriticalSectionKind::Compile,
        CriticalSectionKind::Publish,
        CriticalSectionKind::TierUpWait,
        CriticalSectionKind::RegionBuild,
    ];

    #[test]
    fn interruption_sections_record_each_kind_and_coalesce_requests() {
        for kind in KINDS {
            let word = AtomicU64::new(0);
            let sections = AtomicU64::new(0);
            let guard = EngineSection::enter(&word, &sections, kind);
            word.fetch_or(EXIT_REQUESTED, Ordering::Release);
            word.fetch_or(EXIT_REQUESTED, Ordering::Release);
            drop(guard);
            let evidence = consume(&word, &sections);
            assert!(evidence.deferred());
            assert!(evidence.contains(kind));
            assert_eq!(evidence.deferral_count(), 1);
            assert_eq!(word.load(Ordering::Relaxed), 0);
            assert_eq!(consume(&word, &sections), InterruptionEvidence::default());
        }
    }

    #[test]
    fn interruption_nested_sections_preserve_notifications_and_count() {
        let word = AtomicU64::new(0);
        let sections = AtomicU64::new(0);
        let outer = EngineSection::enter(&word, &sections, CriticalSectionKind::RegionBuild);
        let inner = EngineSection::enter(&word, &sections, CriticalSectionKind::Lift);
        word.fetch_or(EXIT_REQUESTED, Ordering::Release);
        drop(inner);
        // Notifications arriving while a section is active must survive its
        // plain store, and then survive request consumption too.
        word.fetch_or(1 | crate::vm::CODE_SPACE_EXHAUSTED, Ordering::Release);
        drop(outer);
        let evidence = consume(&word, &sections);
        assert!(evidence.contains(CriticalSectionKind::RegionBuild));
        assert!(evidence.contains(CriticalSectionKind::Lift));
        assert_eq!(evidence.deferral_count(), 2);
        assert_eq!(
            word.load(Ordering::Relaxed),
            1 | crate::vm::CODE_SPACE_EXHAUSTED
        );
        word.fetch_or(EXIT_REQUESTED, Ordering::Release);
        assert!(!consume(&word, &sections).deferred());
    }

    #[test]
    fn interruption_outside_sections_and_unrequested_work_are_immediate() {
        let word = AtomicU64::new(0);
        let sections = AtomicU64::new(0);
        for kind in KINDS {
            drop(EngineSection::enter(&word, &sections, kind));
        }
        word.fetch_or(EXIT_REQUESTED, Ordering::Release);
        assert_eq!(consume(&word, &sections), InterruptionEvidence::default());
    }

    #[test]
    fn interruption_unwinding_clears_active_state_and_saturates_count() {
        let word = AtomicU64::new(0);
        let sections = AtomicU64::new(u64::MAX << COUNT_SHIFT);
        let unwind = std::panic::catch_unwind(|| {
            let _section = EngineSection::enter(&word, &sections, CriticalSectionKind::Compile);
            word.fetch_or(EXIT_REQUESTED, Ordering::Release);
            panic!("unwind fixture");
        });
        assert!(unwind.is_err());
        let evidence = consume(&word, &sections);
        assert_eq!(evidence.deferral_count(), u64::MAX >> COUNT_SHIFT);
        assert!(evidence.contains(CriticalSectionKind::Compile));
        assert_eq!(word.load(Ordering::Relaxed), 0);
    }
}
