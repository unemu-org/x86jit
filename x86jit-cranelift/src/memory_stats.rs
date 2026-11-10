//! Allocation diagnostics for the pinned cranelift-jit 0.135.8 (§8.2).

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default)]
pub(crate) struct CodeCounters {
    units: AtomicU64,
    emitted: AtomicU64,
    reserved: AtomicU64,
}

impl CodeCounters {
    pub(crate) fn snapshot(&self) -> x86jit_core::BackendMemoryStats {
        let mut stats = x86jit_core::BackendMemoryStats::default();
        stats.compiled_units_live = self.units.load(Ordering::Relaxed);
        stats.compiled_units_total = stats.compiled_units_live;
        stats.code_bytes_emitted = self.emitted.load(Ordering::Relaxed);
        stats.code_bytes_reserved = self.reserved.load(Ordering::Relaxed);
        stats
    }
}

/// Mirrors cranelift-jit 0.135.8's default SystemMemoryProvider:
/// memory/system.rs::Memory::{allocate, finish_current, set_readable_and_executable}.
/// Each new allocation is a page-rounded MmapMut::map_anon mapping. Aligned
/// definitions share only the current mapping's tail; rollover and finalize
/// abandon that tail, with no reuse of earlier mappings. The optional arena
/// provider is not used. This backend declares no data, disables PIC/hotswap,
/// and emits Abs8 relocations (no Arm64Call veneer reservations), so executable
/// allocation requests equal function lengths, including constant pools.
#[derive(Default)]
pub(crate) struct PendingAllocations {
    len: usize,
    position: usize,
    reserved: u64,
    emitted: u64,
    units: u64,
}

impl PendingAllocations {
    pub(crate) fn define(&mut self, size: usize, align: usize) {
        self.position = self.position.div_ceil(align) * align;
        if size > self.len.saturating_sub(self.position) {
            let page = region::page::size();
            self.len = size.div_ceil(page) * page;
            self.position = 0;
            self.reserved += self.len as u64;
        }
        self.position += size;
        self.emitted += size as u64;
        self.units += 1;
    }

    pub(crate) fn finalize(&mut self, counters: &CodeCounters, success: bool) {
        counters
            .reserved
            .fetch_add(self.reserved, Ordering::Relaxed);
        if success {
            counters.emitted.fetch_add(self.emitted, Ordering::Relaxed);
            counters.units.fetch_add(self.units, Ordering::Relaxed);
        }
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cranelift_jit::{BranchProtection, JITMemoryKind, JITMemoryProvider, SystemMemoryProvider};

    #[test]
    fn tails_alignment_and_finalize_boundaries() {
        let counters = CodeCounters::default();
        let mut pending = PendingAllocations::default();
        let page = region::page::size();
        pending.define(40, 16);
        pending.define(40, 16);
        assert_eq!(counters.snapshot().code_bytes_reserved, 0);
        pending.finalize(&counters, true);
        assert_eq!(counters.snapshot().code_bytes_reserved, page as u64);
        assert_eq!(counters.snapshot().code_bytes_emitted, 80);
        pending.define(page + 1, 16);
        pending.finalize(&counters, true);
        assert_eq!(counters.snapshot().code_bytes_reserved, 3 * page as u64);
        assert_eq!(counters.snapshot().compiled_units_live, 3);
        pending.define(1, 16);
        pending.finalize(&counters, false);
        assert_eq!(counters.snapshot().code_bytes_reserved, 4 * page as u64);
        assert_eq!(counters.snapshot().compiled_units_total, 3);
    }

    #[test]
    fn reservations_match_default_provider_mappings() {
        let mut provider = SystemMemoryProvider::new();
        let counters = CodeCounters::default();
        let mut pending = PendingAllocations::default();
        let page = region::page::size();
        let mut pages = std::collections::BTreeSet::new();
        let mut emitted = 0;
        // Alignment forces rollover despite enough unaligned space. Later
        // batches must not reuse either abandoned or finalized page tails.
        for batch in [
            vec![(page - 32, 1), (17, 64), (40, 16)],
            vec![(page + 1, 16), (40, 16)],
            vec![(1, 16)],
        ] {
            for (size, align) in batch {
                let ptr = provider
                    .allocate(size, align as u64, JITMemoryKind::Executable)
                    .unwrap();
                assert_eq!(ptr.addr() % align, 0);
                pages.extend(ptr.addr() / page..=(ptr.addr() + size - 1) / page);
                pending.define(size, align);
                emitted += size as u64;
            }
            provider.finalize(BranchProtection::None).unwrap();
            pending.finalize(&counters, true);
            assert_eq!(
                counters.snapshot().code_bytes_reserved,
                (pages.len() * page) as u64
            );
            assert_eq!(counters.snapshot().code_bytes_emitted, emitted);
        }
        assert_eq!(pages.len(), 5);
        // SAFETY: no allocated function pointers escape or are executed.
        unsafe { provider.free_memory() };
    }
}
