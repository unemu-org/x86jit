//! Translation cache keyed by [`BlockKey`] — guest address, decode mode and SIMD FP mode (§9.1, §17.4).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use crate::ir::IrBlock;
use crate::lift::CpuMode;

/// Translation-cache key (§17.4): the same guest bytes decode and lift differently per
/// decode mode (mode × CS.D), and numerical operations depend on MXCSR.RC/FTZ/DAZ.
/// All three form the translation identity. A future protected
/// mode adds a `CpuMode` variant, not a new key shape; segment bases stay out of the key
/// (they are runtime `CpuState`, per the FS/GS pattern), so a segment reload never
/// invalidates a translation.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct BlockKey {
    pub guest_addr: u64,
    pub mode: CpuMode,
    /// RC | FTZ | DAZ assumed by this translation (SDM Vol 1 §10.2.3).
    pub mxcsr_mode: u32,
}

impl std::hash::Hash for BlockKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.guest_addr.hash(state);
        // Keep the two-word hash stream: the decode discriminator occupies the
        // low byte, leaving the upper bits for the SIMD specialization.
        ((u64::from(self.mxcsr_mode) << 8) | self.mode as u64).hash(state);
    }
}

impl BlockKey {
    pub fn new(guest_addr: u64, mode: CpuMode) -> Self {
        Self {
            guest_addr,
            mode,
            mxcsr_mode: 0,
        }
    }
    pub fn with_mxcsr(guest_addr: u64, mode: CpuMode, mxcsr: u32) -> Self {
        Self {
            guest_addr,
            mode,
            mxcsr_mode: mxcsr & crate::interp::softfloat::MODE_MASK,
        }
    }
}

/// A compiled entry address and its allocation generation.
///
/// Manually `Send + Sync`: compiled code is read-only and executable from any
/// thread. This identity does not own storage (§9.1). Execution must be protected
/// by the backend's code-lifetime domain before pointer resolution
/// through the final call and cold fault handling.
#[derive(Copy, Clone)]
pub struct CompiledPtr(pub *const u8, u64);

impl CompiledPtr {
    /// An entry owned by a backend which retains code indefinitely. Reclaiming
    /// backends must instead return the entry from `CodeLifetime::register`'s unit.
    pub const fn new(address: *const u8) -> Self {
        Self(address, 0)
    }

    pub const fn generation(self) -> u64 {
        self.1
    }

    pub(crate) const fn with_generation(address: *const u8, generation: u64) -> Self {
        Self(address, generation)
    }
}

// SAFETY: see the type-level comment.
unsafe impl Send for CompiledPtr {}
unsafe impl Sync for CompiledPtr {}

/// A materialized block. The backend produces it; the dispatcher executes it
/// uniformly by matching on the variant (§8).
#[derive(Clone)]
pub enum CachedBlock {
    Interpreted(Arc<IrBlock>),
    Compiled { entry: CompiledPtr },
}

// A sparse radix directory avoids hashing on a byte-coverage miss. Its height
// grows with the highest page number, bounded by 13 nibbles for a u64 address.
#[derive(Default)]
struct CoverageIndex {
    root: Option<CoverageNode>,
    height: u32,
    bytes: u64,
    #[cfg(test)]
    uncovered_ranges: usize,
}

enum CoverageNode {
    Branch(Box<[Option<CoverageNode>; 16]>),
    Page(Box<PageCoverage>),
}

struct PageCoverage {
    bits: [u64; 64],
    // Only overlapping spans need counts beyond the first reference. Disjoint
    // translations on the same page retain the bitmap's compact representation.
    extra: Option<Box<ExtraCoverage>>,
}

struct ExtraCoverage {
    // Two exact extra-reference counts per byte; overflow stores the excess
    // above 15 and is always consumed before the packed count is decremented.
    counts: [u8; 2048],
    overflow: Vec<(u16, usize)>,
    shared_bytes: u16,
}

impl ExtraCoverage {
    fn bytes(&self) -> u64 {
        (std::mem::size_of::<Self>()
            + self.overflow.capacity() * std::mem::size_of::<(u16, usize)>()) as u64
    }

    fn increment(&mut self, byte: usize) {
        let shift = (byte % 2) * 4;
        let count = (self.counts[byte / 2] >> shift) & 15;
        if count == 15 {
            if let Some((_, excess)) = self
                .overflow
                .iter_mut()
                .find(|(b, _)| usize::from(*b) == byte)
            {
                *excess = excess.checked_add(1).expect("coverage count exhausted");
            } else {
                self.overflow.push((byte as u16, 1));
            }
        } else {
            self.shared_bytes += u16::from(count == 0);
            self.counts[byte / 2] += 1 << shift;
        }
    }

    fn decrement(&mut self, byte: usize) -> bool {
        let shift = (byte % 2) * 4;
        let count = (self.counts[byte / 2] >> shift) & 15;
        if count == 0 {
            return false;
        }
        if count == 15 {
            if let Some(i) = self
                .overflow
                .iter()
                .position(|(b, _)| usize::from(*b) == byte)
            {
                self.overflow[i].1 -= 1;
                if self.overflow[i].1 == 0 {
                    self.overflow.swap_remove(i);
                    if self.overflow.is_empty() {
                        self.overflow = Vec::new();
                    }
                }
                return true;
            }
        }
        self.counts[byte / 2] -= 1 << shift;
        self.shared_bytes -= u16::from(count == 1);
        true
    }
}

impl CoverageIndex {
    fn page(&self, page: u64) -> Option<&[u64; 64]> {
        if page >> (self.height * 4) != 0 {
            return None;
        }
        let mut node = self.root.as_ref()?;
        for level in (0..self.height).rev() {
            let CoverageNode::Branch(children) = node else {
                unreachable!()
            };
            node = children[((page >> (level * 4)) & 15) as usize].as_ref()?;
        }
        let CoverageNode::Page(bits) = node else {
            unreachable!()
        };
        Some(&bits.bits)
    }

    fn set_page(&mut self, page: u64, bits: [u64; 64]) {
        while page >> (self.height * 4) != 0 {
            if let Some(root) = self.root.take() {
                let mut children = Box::new(std::array::from_fn(|_| None));
                children[0] = Some(root);
                self.bytes += std::mem::size_of_val(&*children) as u64;
                self.root = Some(CoverageNode::Branch(children));
            }
            self.height += 1;
        }
        fn set(
            node: &mut Option<CoverageNode>,
            height: u32,
            page: u64,
            bits: [u64; 64],
            bytes: &mut u64,
        ) {
            if height == 0 {
                if node.is_none() {
                    *bytes += std::mem::size_of::<PageCoverage>() as u64;
                }
                if let Some(CoverageNode::Page(current)) = node {
                    current.bits = bits;
                } else {
                    *node = Some(CoverageNode::Page(Box::new(PageCoverage {
                        bits,
                        extra: None,
                    })));
                }
                return;
            }
            if node.is_none() {
                let children = Box::new(std::array::from_fn(|_| None));
                *bytes += std::mem::size_of_val(&*children) as u64;
                *node = Some(CoverageNode::Branch(children));
            }
            let Some(CoverageNode::Branch(children)) = node else {
                unreachable!()
            };
            set(
                &mut children[((page >> ((height - 1) * 4)) & 15) as usize],
                height - 1,
                page,
                bits,
                bytes,
            );
        }
        set(&mut self.root, self.height, page, bits, &mut self.bytes);
    }

    fn page_mut(&mut self, page: u64) -> &mut PageCoverage {
        let mut node = self.root.as_mut().unwrap();
        for level in (0..self.height).rev() {
            let CoverageNode::Branch(children) = node else {
                unreachable!()
            };
            node = children[((page >> (level * 4)) & 15) as usize]
                .as_mut()
                .unwrap();
        }
        let CoverageNode::Page(coverage) = node else {
            unreachable!()
        };
        coverage
    }

    fn cover(&mut self, page: u64, lo: usize, hi: usize) {
        if self.page(page).is_none() {
            let mut bits = [0; 64];
            cover_page(&mut bits, lo, hi);
            self.set_page(page, bits);
            return;
        }
        let coverage = self.page_mut(page);
        let before = coverage.extra.as_ref().map_or(0, |extra| extra.bytes());
        for word in lo / 64..=(hi - 1) / 64 {
            let mask = page_word_mask(word, lo, hi);
            let mut shared = coverage.bits[word] & mask;
            if shared != 0 {
                let extra = coverage.extra.get_or_insert_with(|| {
                    Box::new(ExtraCoverage {
                        counts: [0; 2048],
                        overflow: Vec::new(),
                        shared_bytes: 0,
                    })
                });
                while shared != 0 {
                    extra.increment(word * 64 + shared.trailing_zeros() as usize);
                    shared &= shared - 1;
                }
            }
            coverage.bits[word] |= mask;
        }
        let after = coverage.extra.as_ref().map_or(0, |extra| extra.bytes());
        self.bytes = self.bytes - before + after;
    }

    fn uncover(&mut self, page: u64, lo: usize, hi: usize) {
        #[cfg(test)]
        {
            self.uncovered_ranges += 1;
        }
        let coverage = self.page_mut(page);
        let before = coverage.extra.as_ref().map_or(0, |extra| extra.bytes());
        for word in lo / 64..=(hi - 1) / 64 {
            let mask = page_word_mask(word, lo, hi);
            let mut keep = 0;
            if let Some(extra) = coverage.extra.as_mut() {
                let mut shared = mask;
                while shared != 0 {
                    let bit = shared.trailing_zeros() as usize;
                    if extra.decrement(word * 64 + bit) {
                        keep |= 1 << bit;
                    }
                    shared &= shared - 1;
                }
            }
            coverage.bits[word] = (coverage.bits[word] & !mask) | keep;
        }
        if coverage
            .extra
            .as_ref()
            .is_some_and(|extra| extra.shared_bytes == 0)
        {
            coverage.extra = None;
        }
        let after = coverage.extra.as_ref().map_or(0, |extra| extra.bytes());
        self.bytes = self.bytes - before + after;
    }

    fn remove_page(&mut self, page: u64) {
        fn remove(node: &mut Option<CoverageNode>, height: u32, page: u64, bytes: &mut u64) {
            let Some(current) = node else { return };
            match current {
                CoverageNode::Page(coverage) => {
                    *bytes -= std::mem::size_of::<PageCoverage>() as u64;
                    *bytes -= coverage.extra.as_ref().map_or(0, |extra| extra.bytes());
                }
                CoverageNode::Branch(children) => {
                    remove(
                        &mut children[((page >> ((height - 1) * 4)) & 15) as usize],
                        height - 1,
                        page,
                        bytes,
                    );
                    if children.iter().any(Option::is_some) {
                        return;
                    }
                    *bytes -= std::mem::size_of_val(&**children) as u64;
                }
            }
            *node = None;
        }
        if page >> (self.height * 4) == 0 {
            remove(&mut self.root, self.height, page, &mut self.bytes);
        }
        if self.root.is_none() {
            self.height = 0;
        }
    }
}

fn page_word_mask(word: usize, lo: usize, hi: usize) -> u64 {
    (u64::MAX << (lo.saturating_sub(word * 64))) & (u64::MAX >> (64 - (hi - word * 64).min(64)))
}

fn cover_page(bits: &mut [u64; 64], lo: usize, hi: usize) {
    if lo < hi {
        for (word, value) in bits
            .iter_mut()
            .enumerate()
            .take((hi - 1) / 64 + 1)
            .skip(lo / 64)
        {
            *value |= page_word_mask(word, lo, hi);
        }
    }
}

// Bounds fit in 13 bits each, including the exclusive page end (4096).
// A second span sets the fallback bit: its gaps are checked against units.
struct PageSpans(u32);

impl PageSpans {
    fn new(lo: usize, hi: usize) -> Self {
        Self(lo as u32 | ((hi as u32) << 13))
    }

    fn extend(&mut self, lo: usize, hi: usize) {
        *self = Self(Self::new(lo.min(self.lo()), hi.max(self.hi())).0 | (1 << 26));
    }

    fn lo(&self) -> usize {
        (self.0 & 8191) as usize
    }
    fn hi(&self) -> usize {
        ((self.0 >> 13) & 8191) as usize
    }
}

// HashMap's public capacity excludes its empty buckets. Include those buckets
// and control bytes (including the trailing group), but not allocator metadata.
// Deletion can lower reported capacity without freeing buckets: retain the
// largest allocation estimate until the owning table is dropped.
fn table_bytes<T>(capacity: usize) -> u64 {
    if capacity == 0 {
        return 0;
    }
    let buckets = (capacity + 1).next_power_of_two();
    (buckets * (std::mem::size_of::<T>() + 1) + 16) as u64
}

#[derive(Default)]
struct PageMembership {
    entries: HashMap<BlockKey, PageSpans>,
    bytes: u64,
}

#[derive(Default)]
struct SourceIndex {
    coverage: CoverageIndex,
    allocation_bytes: u64,
    units_bytes: u64,
    pages_bytes: u64,
    units: HashMap<BlockKey, Vec<(u64, u32)>>,
    pages: HashMap<u64, PageMembership>,
}

// One lock protects both span indices and pending sources. Lock order is
// spans -> map -> hotness, with atomic page-tag publication (§10, §17.4).
#[derive(Default)]
struct SpanIndex {
    cached: SourceIndex,
    pending_sources: SourceIndex,
    pending: HashMap<BlockKey, PendingTierUp>,
    pending_bytes: u64,
    suppression_bytes: u64,
    next_request: u64,
    interpreter_only: HashSet<BlockKey>,
}

impl SpanIndex {
    fn insert(&mut self, key: BlockKey, spans: Vec<(u64, u32)>) {
        self.interpreter_only.remove(&key);
        self.cached.insert(key, spans);
    }

    fn coverage_bytes(&self) -> u64 {
        self.cached.bytes()
            + self.pending_sources.bytes()
            + self.pending_bytes
            + self.suppression_bytes
    }
}

struct PendingTierUp {
    queued_at: Option<crate::time_stats::MonotonicTime>,
    id: u64,
    cancelled: bool,
    generation: u64,
}

fn span_pages(start: u64, len: u32) -> std::ops::RangeInclusive<u64> {
    (start >> crate::memory::CODE_PAGE_BITS)
        ..=(start.saturating_add(u64::from(len).saturating_sub(1)) >> crate::memory::CODE_PAGE_BITS)
}

impl SourceIndex {
    fn bytes(&self) -> u64 {
        self.coverage.bytes + self.allocation_bytes + self.units_bytes + self.pages_bytes
    }

    fn insert(&mut self, key: BlockKey, spans: Vec<(u64, u32)>) {
        self.remove(&key);
        for &(start, len) in &spans {
            if len != 0 {
                for page in span_pages(start, len) {
                    let base = page << crate::memory::CODE_PAGE_BITS;
                    let lo = start.saturating_sub(base) as usize;
                    let hi = start
                        .saturating_add(u64::from(len))
                        .saturating_sub(base)
                        .min(4096) as usize;
                    let membership = self.pages.entry(page).or_default();
                    let before = membership.bytes;
                    membership
                        .entries
                        .entry(key)
                        .and_modify(|spans| spans.extend(lo, hi))
                        .or_insert_with(|| PageSpans::new(lo, hi));
                    membership.bytes = before.max(table_bytes::<(BlockKey, PageSpans)>(
                        membership.entries.capacity(),
                    ));
                    self.allocation_bytes += membership.bytes - before;
                    self.coverage.cover(page, lo, hi);
                }
            }
        }
        self.allocation_bytes += (spans.capacity() * std::mem::size_of::<(u64, u32)>()) as u64;
        self.units.insert(key, spans);
        self.units_bytes = self
            .units_bytes
            .max(table_bytes::<(BlockKey, Vec<(u64, u32)>)>(
                self.units.capacity(),
            ));
        self.pages_bytes = self
            .pages_bytes
            .max(table_bytes::<(u64, PageMembership)>(self.pages.capacity()));
    }

    fn remove(&mut self, key: &BlockKey) -> Option<Vec<(u64, u32)>> {
        let removed = self.units.remove(key);
        if let Some(spans) = &removed {
            self.allocation_bytes -= (spans.capacity() * std::mem::size_of::<(u64, u32)>()) as u64;
            // Remove every reference before dropping membership: repeated and
            // overlapping spans of one unit must each decrement their counts.
            for &(start, len) in spans {
                if len != 0 {
                    for page in span_pages(start, len) {
                        let base = page << crate::memory::CODE_PAGE_BITS;
                        let lo = start.saturating_sub(base) as usize;
                        let hi = start
                            .saturating_add(u64::from(len))
                            .saturating_sub(base)
                            .min(4096) as usize;
                        self.coverage.uncover(page, lo, hi);
                    }
                }
            }
            for &(start, len) in spans {
                if len != 0 {
                    for page in span_pages(start, len) {
                        if let Some(membership) = self.pages.get_mut(&page) {
                            membership.entries.remove(key);
                            if membership.entries.is_empty() {
                                self.allocation_bytes -= membership.bytes;
                                self.pages.remove(&page);
                                self.coverage.remove_page(page);
                            }
                        }
                    }
                }
            }
            if self.units.is_empty() {
                self.units = HashMap::new();
                self.pages = HashMap::new();
                self.units_bytes = 0;
                self.pages_bytes = 0;
            }
        }
        removed
    }

    // Large sparse flushes use the page index instead of walking address holes.
    // Ordinary writes touch one or two pages, and do no hashing or allocation.
    fn coverage_overlap(&self, lo: u64, hi: u64) -> (bool, bool) {
        if lo >= hi {
            return (false, false);
        }
        let first = lo >> crate::memory::CODE_PAGE_BITS;
        let last = (hi - 1) >> crate::memory::CODE_PAGE_BITS;
        if last - first > 1 {
            return (true, true);
        }
        let mut on_page = false;
        for page in first..=last {
            if let Some(bits) = self.coverage.page(page) {
                on_page = true;
                let base = page << crate::memory::CODE_PAGE_BITS;
                let start = lo.saturating_sub(base) as usize;
                let end = hi.saturating_sub(base).min(4096) as usize;
                for (word, value) in bits
                    .iter()
                    .enumerate()
                    .take((end - 1) / 64 + 1)
                    .skip(start / 64)
                {
                    if value & page_word_mask(word, start, end) != 0 {
                        return (true, true);
                    }
                }
            }
        }
        (false, on_page)
    }

    fn has_units(&self, lo: u64, hi: u64) -> bool {
        if lo >= hi {
            return false;
        }
        let first = lo >> crate::memory::CODE_PAGE_BITS;
        let last = (hi - 1) >> crate::memory::CODE_PAGE_BITS;
        if last - first < self.pages.len() as u64 {
            (first..=last).any(|page| self.pages.contains_key(&page))
        } else {
            self.pages
                .keys()
                .any(|page| first <= *page && *page <= last)
        }
    }

    fn victims(&self, lo: u64, hi: u64) -> Vec<BlockKey> {
        let mut keys = Vec::new();
        if lo >= hi {
            return keys;
        }
        let first = lo >> crate::memory::CODE_PAGE_BITS;
        let last = (hi - 1) >> crate::memory::CODE_PAGE_BITS;
        let mut seen = HashSet::new();
        let mut collect = |page: u64, entries: &HashMap<BlockKey, PageSpans>| {
            let base = page << crate::memory::CODE_PAGE_BITS;
            let start = lo.saturating_sub(base);
            let end = hi.saturating_sub(base).min(4096);
            for (key, spans) in entries {
                if spans.lo() as u64 >= end || start >= spans.hi() as u64 {
                    continue;
                }
                if spans.0 & (1 << 26) != 0
                    && !self.units[key].iter().any(|&(start, len)| {
                        start < hi && lo < start.saturating_add(u64::from(len)) && len != 0
                    })
                {
                    continue;
                }
                if first == last {
                    keys.push(*key);
                    continue;
                }
                if seen.insert(*key) {
                    keys.push(*key);
                }
            }
        };
        // Whole-cache/unmap requests can cover a sparse 64-bit address space.
        if last - first < self.pages.len() as u64 {
            for page in first..=last {
                if let Some(entries) = self.pages.get(&page) {
                    collect(page, &entries.entries);
                }
            }
        } else {
            for (&page, entries) in &self.pages {
                if first <= page && page <= last {
                    collect(page, &entries.entries);
                }
            }
        }
        keys
    }
}

/// Shared translation cache. Cloned-out on `get` so no lock guard is held
/// across block execution (which may mutate memory -> SMC invalidation) (§9.2).
pub struct TranslationCache {
    memory_counts: [AtomicU64; 4],
    coverage_bytes: AtomicU64,
    invalidation_calls: AtomicU64,
    invalidation_blocks_dropped: AtomicU64,
    tier_bg_submitted: AtomicU64,
    tier_bg_completed: AtomicU64,
    tier_bg_duplicates_skipped: AtomicU64,
    tier_bg_rejected_generation: AtomicU64,
    tier_bg_rejected_compile_failed: AtomicU64,
    tier_bg_failed_descriptor_limit: AtomicU64,
    tier_bg_failed_code_space: AtomicU64,
    tier_bg_failed_other_backend: AtomicU64,
    tier_bg_interpreter_only: AtomicU64,
    tier_bg_interpreter_only_skipped: AtomicU64,
    tier_bg_superseded: AtomicU64,

    // §17.4: keyed by BlockKey { guest_addr, mode } — the same bytes lift differently
    // per decode mode, so translations never alias across modes.
    map: RwLock<HashMap<BlockKey, CachedBlock>>,
    // Guest byte spans of each cached unit, keyed by its `BlockKey`. A single
    // block has one `(start, len)`; a superblock (M5-T3) has one per sub-block,
    // possibly non-contiguous. A write overlapping ANY span drops the whole unit
    // (§10) — SMC is address-scoped, so `invalidate_overlapping` matches spans by
    // address across every mode. Kept in lockstep with `map`.
    spans: RwLock<SpanIndex>,
    // Dispatcher stats (§12 M3): a miss means the block had to be lifted. `Relaxed`
    // — these are counters, not synchronization.
    hits: AtomicU64,
    misses: AtomicU64,
    // Block-chaining "fires" counter (§12 M5, testing.md §8.2): a chained transfer
    // took the direct link-slot path instead of a cache lookup.
    chained: AtomicU64,
    // Superblocks compiled as multi-block regions (§12 M5-T3, testing.md §8.2).
    regions: AtomicU64,
    // Invalidation epoch (fast dispatch R1): bumped on every
    // `invalidate_overlapping` that drops a unit. Vcpu-local predictor state
    // (fast-resolve array, return ring) snapshots this in the dispatcher outer loop
    // and flushes when it changes — the coherence channel for state that cannot be
    // reached by the backend's `invalidate_links` (esp. cross-thread invalidation).
    lifetime: RwLock<Arc<crate::code_lifetime::CodeLifetime>>,
    previous_lifetimes: Mutex<Vec<Arc<crate::code_lifetime::CodeLifetime>>>,
    // Every invalidation cancels source reads still in flight, even if no cached
    // span overlaps. Keep this separate so data writes preserve dispatch predictors.
    publication_generation: AtomicU64,
    // Policy/configuration changes also advance the dispatch epoch without
    // victims; source publication is always canceled independently.
    pub(crate) invalidate_pending_fetches: bool,
    // IBTC "fires" counter: descriptors published (§12 M5-style stat).
    ibtc_filled: AtomicU64,
    // Per-block execution counts for hotness-gated tier-up (FD tiering): a block
    // starts interpreted and is JIT-compiled only after it runs `tier_up_after`
    // times. Keyed by entry address; dropped alongside the block on invalidation.
    hotness: RwLock<HashMap<BlockKey, AtomicU32>>,
    // Cached region-candidacy decision per hot entry pc: `true` = this pc is
    // a multi-block loop that should tier up to a region at T2, `false` = tier the
    // single block at T1. Decided once (one `lift_region`) when a block first crosses
    // T1, so the dispatcher doesn't re-lift every dispatch while a loop warms toward T2.
    // Perf-only (both tiers are correct); a stale entry after SMC just re-decides on the
    // fresh block. Keyed by entry address.
    region_decision: RwLock<HashMap<BlockKey, bool>>,
    // Background tier-up "fires" counters: a completion published into the cache, or
    // rejected (epoch moved / block gone) at publish time.
    tier_bg_published: AtomicU64,
    tier_up_latency_ns: AtomicU64,
    tier_up_latency_count: AtomicU64,
    tier_bg_rejected: AtomicU64,
}

impl TranslationCache {
    #[cfg(test)]
    pub(crate) fn poison_for_child_test(&self) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _map = self.map.write().unwrap();
            let _spans = self.spans.write().unwrap();
            panic!("poison parent translation cache");
        }));
        assert!(result.is_err());
        assert!(self.map.is_poisoned() && self.spans.is_poisoned());
    }

    fn account_replace(
        &self,
        old: Option<&CachedBlock>,
        old_spans: usize,
        new: Option<&CachedBlock>,
        new_spans: usize,
    ) -> bool {
        if !self.code_lifetime().replace(old, new) {
            return false;
        }
        self.account_memory_replace(old, old_spans, new, new_spans);
        true
    }

    fn account_memory_replace(
        &self,
        old: Option<&CachedBlock>,
        old_spans: usize,
        new: Option<&CachedBlock>,
        new_spans: usize,
    ) {
        fn values(block: Option<&CachedBlock>, spans: usize) -> [u64; 4] {
            match block {
                Some(CachedBlock::Interpreted(ir)) => [
                    1,
                    0,
                    0,
                    (std::mem::size_of::<IrBlock>()
                        + 2 * std::mem::size_of::<usize>()
                        + ir.ops.capacity() * std::mem::size_of::<crate::IrOp>())
                        as u64,
                ],
                Some(CachedBlock::Compiled { .. }) if spans > 1 => [0, 0, 1, 0],
                Some(CachedBlock::Compiled { .. }) => [0, 1, 0, 0],
                None => [0; 4],
            }
        }
        for ((counter, old), new) in self
            .memory_counts
            .iter()
            .zip(values(old, old_spans))
            .zip(values(new, new_spans))
        {
            if new > old {
                counter.fetch_add(new - old, Ordering::Relaxed);
            } else if old > new {
                counter.fetch_sub(old - new, Ordering::Relaxed);
            }
        }
    }

    /// Allocation diagnostics without walking or locking the cache.
    pub fn memory_stats(&self) -> crate::CacheMemoryStats {
        let [interpreted_entries, compiled_entries, region_entries, interpreted_ir_bytes] =
            std::array::from_fn(|i| self.memory_counts[i].load(Ordering::Relaxed));
        crate::CacheMemoryStats {
            interpreted_entries,
            compiled_entries,
            region_entries,
            interpreted_ir_bytes,
            coverage_index_bytes: self.coverage_bytes.load(Ordering::Relaxed),
            invalidation_calls: self.invalidation_calls.load(Ordering::Relaxed),
            invalidation_blocks_dropped: self.invalidation_blocks_dropped.load(Ordering::Relaxed),
            tier_bg_submitted: self.tier_bg_submitted.load(Ordering::Relaxed),
            tier_bg_completed: self.tier_bg_completed.load(Ordering::Relaxed),
            tier_bg_duplicates_skipped: self.tier_bg_duplicates_skipped.load(Ordering::Relaxed),
            tier_bg_rejected_generation: self.tier_bg_rejected_generation.load(Ordering::Relaxed),
            tier_bg_rejected_compile_failed: self
                .tier_bg_rejected_compile_failed
                .load(Ordering::Relaxed),
            tier_bg_failed_descriptor_limit: self
                .tier_bg_failed_descriptor_limit
                .load(Ordering::Relaxed),
            tier_bg_failed_code_space: self.tier_bg_failed_code_space.load(Ordering::Relaxed),
            tier_bg_failed_other_backend: self.tier_bg_failed_other_backend.load(Ordering::Relaxed),
            tier_bg_interpreter_only: self.tier_bg_interpreter_only.load(Ordering::Relaxed),
            tier_bg_interpreter_only_skipped: self
                .tier_bg_interpreter_only_skipped
                .load(Ordering::Relaxed),
            tier_bg_superseded: self.tier_bg_superseded.load(Ordering::Relaxed),
            tier_bg_published: self.tier_bg_published(),
            tier_bg_rejected: self.tier_bg_rejected(),
        }
    }

    pub fn new() -> Self {
        Self {
            memory_counts: std::array::from_fn(|_| AtomicU64::new(0)),
            coverage_bytes: AtomicU64::new(0),
            invalidation_calls: AtomicU64::new(0),
            invalidation_blocks_dropped: AtomicU64::new(0),
            tier_bg_submitted: AtomicU64::new(0),
            tier_bg_completed: AtomicU64::new(0),
            tier_bg_duplicates_skipped: AtomicU64::new(0),
            tier_bg_rejected_generation: AtomicU64::new(0),
            tier_bg_rejected_compile_failed: AtomicU64::new(0),
            tier_bg_failed_descriptor_limit: AtomicU64::new(0),
            tier_bg_failed_code_space: AtomicU64::new(0),
            tier_bg_failed_other_backend: AtomicU64::new(0),
            tier_bg_interpreter_only: AtomicU64::new(0),
            tier_bg_interpreter_only_skipped: AtomicU64::new(0),
            tier_bg_superseded: AtomicU64::new(0),

            map: RwLock::new(HashMap::new()),
            spans: RwLock::new(SpanIndex::default()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            chained: AtomicU64::new(0),
            regions: AtomicU64::new(0),
            lifetime: RwLock::default(),
            previous_lifetimes: Mutex::default(),
            publication_generation: AtomicU64::new(0),
            invalidate_pending_fetches: false,
            ibtc_filled: AtomicU64::new(0),
            hotness: RwLock::new(HashMap::new()),
            region_decision: RwLock::new(HashMap::new()),
            tier_bg_published: AtomicU64::new(0),
            tier_up_latency_ns: AtomicU64::new(0),
            tier_up_latency_count: AtomicU64::new(0),
            tier_bg_rejected: AtomicU64::new(0),
        }
    }

    /// Increment a block's execution count and return the new value (FD tiering).
    /// Called on each dispatch of an interpreted block; when it reaches the Vm's
    /// `tier_up_after` the block is JIT-compiled. The count is an `AtomicU32`, so once
    /// a block's entry exists the bump takes only a **read** lock — concurrent vcpus
    /// running the same pre-hot block no longer serialize on a write lock. Only the
    /// first sight of a block takes the write lock to insert the counter.
    pub fn bump_hotness(&self, key: BlockKey) -> u32 {
        if let Some(c) = self.hotness.read().unwrap().get(&key) {
            return c.fetch_add(1, Ordering::Relaxed) + 1;
        }
        self.hotness
            .write()
            .unwrap()
            .entry(key)
            .or_insert_with(|| AtomicU32::new(0))
            .fetch_add(1, Ordering::Relaxed)
            + 1
    }

    /// The cached region-candidacy decision for `key`, or `None` if this block hasn't
    /// been decided yet. Read on the hot path while a loop warms; a `read` lock.
    pub fn region_decision(&self, key: BlockKey) -> Option<bool> {
        self.region_decision.read().unwrap().get(&key).copied()
    }

    /// Record `key`'s region-candidacy decision — done once, when the block first
    /// crosses T1, so the dispatcher never re-lifts to re-decide.
    pub fn set_region_decision(&self, key: BlockKey, candidate: bool) {
        self.region_decision.write().unwrap().insert(key, candidate);
    }

    /// Replace a cached block's materialization (interpreted → compiled), retaining
    /// its span and dropping its hotness counter. `since_generation` must be read
    /// from [`Self::publication_generation`] before fetching source bytes or IR.
    /// Returns `false` without changes if an invalidation has intervened; the
    /// caller must re-lift rather than resurrect a stale translation. Tracked
    /// entries must also name a live generation in this cache's lifetime domain.
    #[must_use]
    pub fn upgrade(
        &self,
        key: BlockKey,
        block: CachedBlock,
        span: (u64, u32),
        since_generation: u64,
    ) -> bool {
        // Lock order matches `invalidate_overlapping` (spans → map → hotness) so the
        // two can't deadlock. The spans lock keeps the generation check and
        // publication atomic with respect to invalidation.
        let mut spans = self.spans.write().unwrap();
        let mut map = self.map.write().unwrap();
        if self.publication_generation.load(Ordering::Acquire) != since_generation {
            self.code_lifetime().discard(&block);
            return false;
        }
        if !self.account_replace(
            map.get(&key),
            spans.cached.units.get(&key).map_or(0, Vec::len),
            Some(&block),
            1,
        ) {
            return false;
        }
        spans.insert(key, vec![span]);
        self.coverage_bytes
            .store(spans.coverage_bytes(), Ordering::Relaxed);
        map.insert(key, block);
        self.hotness.write().unwrap().remove(&key);
        true
    }

    /// Publish a background-compiled unit with one source span per sub-block.
    /// Uses [`Self::upgrade`]'s generation check and hotness removal, re-tagging
    /// every span's pages via `on_mark` under the spans lock. Returns `false`
    /// without changes if invalidation advanced `since_generation`.
    #[must_use]
    pub fn upgrade_region(
        &self,
        key: BlockKey,
        block: CachedBlock,
        spans: Vec<(u64, u32)>,
        since_generation: u64,
        on_mark: impl FnOnce(&[(u64, u32)]),
    ) -> bool {
        let mut sp = self.spans.write().unwrap();
        let mut mp = self.map.write().unwrap();
        if self.publication_generation.load(Ordering::Acquire) != since_generation {
            self.code_lifetime().discard(&block);
            return false;
        }
        on_mark(&spans);
        if !self.account_replace(
            mp.get(&key),
            sp.cached.units.get(&key).map_or(0, Vec::len),
            Some(&block),
            spans.len(),
        ) {
            return false;
        }
        mp.insert(key, block);
        sp.insert(key, spans);
        self.coverage_bytes
            .store(sp.coverage_bytes(), Ordering::Relaxed);
        self.hotness.write().unwrap().remove(&key);
        true
    }

    /// Allocate an immutable IBTC descriptor `{target, entry}` (R4) and return its
    /// stable heap address for the compiled code to load. The descriptor is never
    /// mutated (a new target gets a new descriptor). Execution grace after slot
    /// clearing protects every reader before reclamation. The pointer publication
    /// into the slot must be a **`Release`** store (see
    /// the `RET_IBTC_MISS` site in `vm.rs`): the two field writes above happen-before
    /// the pointer becomes visible, pairing with the reader's address dependency
    /// (release/consume). A `Relaxed` publish would let a weakly-ordered host expose
    /// the pointer before the `{target, entry}` fields — a torn payload, not just a
    /// torn scalar.
    pub fn alloc_ibtc_descriptor(&self, target: u64, entry: CompiledPtr) -> u64 {
        let addr = self.code_lifetime().alloc_ibtc_descriptor(target, entry);
        self.ibtc_filled.fetch_add(1, Ordering::Relaxed);
        addr
    }

    /// IBTC descriptors published (the R4 "fires" counter).
    pub fn ibtc_filled(&self) -> u64 {
        self.ibtc_filled.load(Ordering::Relaxed)
    }

    /// Publish something — a link slot, an IBTC descriptor — only if the invalidation
    /// epoch has not moved since it was resolved (R1). Returns `false` without
    /// publishing if it has.
    ///
    /// The window this closes: a vcpu resolves an address to a compiled entry, and
    /// between the resolve returning and the slot store another vcpu drops that
    /// translation and clears every slot. The store then republishes the pointer INTO A
    /// SLOT THAT WAS JUST CLEARED, and the resolving side jumps to it. Keeping the code
    /// bytes allocated stops that being a use-after-free; it does not stop it being a
    /// translation of code the guest has since rewritten.
    ///
    /// Checking the epoch before the store would not be enough — the invalidation could
    /// land in between. The publish happens while HOLDING the locks `invalidate_overlapping`
    /// takes, in the same order (spans → map), so the two serialize. That is the shape
    /// [`upgrade`](Self::upgrade) already uses for the same reason.
    ///
    /// The lifetime registry serializes the store with retirement in other caches
    /// sharing this backend. The closure must not reenter the engine.
    ///
    /// Cost is a write-lock pair and registry mutex per slot fill: a direct
    /// edge fills once and is then followed with no dispatcher involvement at all, and
    /// IBTC refills stop at `IBTC_MEGAMORPHIC_CAP`.
    #[must_use]
    pub fn publish_if_current(&self, since_epoch: u64, publish: impl FnOnce()) -> bool {
        let _spans = self.spans.write().unwrap();
        let _map = self.map.write().unwrap();
        self.code_lifetime()
            .publish_if_current(since_epoch, publish)
    }

    /// Validate the target generation under the retirement lock before exposing
    /// any link or descriptor. A current epoch alone cannot validate a private
    /// prediction obtained before that epoch.
    #[must_use]
    pub fn publish_entry_if_current(
        &self,
        since_epoch: u64,
        entry: CompiledPtr,
        publish: impl FnOnce(),
    ) -> bool {
        let _spans = self.spans.write().unwrap();
        let _map = self.map.write().unwrap();
        self.code_lifetime()
            .publish_entry_if_current(since_epoch, entry, publish)
    }

    /// Execution and predictor lifetime shared with the compiling backend.
    pub fn code_lifetime(&self) -> Arc<crate::code_lifetime::CodeLifetime> {
        self.lifetime.read().unwrap().clone()
    }

    /// Rebind a closed cache at run entry, serialized with source publication.
    pub(crate) fn reconcile_code_lifetime(
        &self,
        next: Arc<crate::code_lifetime::CodeLifetime>,
        clear_page: impl Fn(u64),
    ) -> bool {
        let mut spans = self.spans.write().unwrap();
        let mut map = self.map.write().unwrap();
        let mut lifetime = self.lifetime.write().unwrap();
        if lifetime.is_open() || !next.is_open() || Arc::ptr_eq(&lifetime, &next) {
            return false;
        }
        lifetime.remove_publications(map.values());
        for (key, block) in map.iter() {
            self.account_memory_replace(
                Some(block),
                spans.cached.units.get(key).map_or(0, Vec::len),
                None,
                0,
            );
        }
        map.clear();
        for page in spans
            .cached
            .pages
            .keys()
            .chain(spans.pending_sources.pages.keys())
        {
            clear_page(*page);
        }
        let next_request = spans.next_request;
        *spans = SpanIndex {
            next_request,
            ..Default::default()
        };
        self.coverage_bytes.store(0, Ordering::Relaxed);
        self.hotness.write().unwrap().clear();
        self.region_decision.write().unwrap().clear();
        self.publication_generation.fetch_add(1, Ordering::Release);
        // Closed domains can still own deferred teardown workers. Keep them
        // reachable so VM shutdown joins every replaced backend's workers.
        self.previous_lifetimes
            .lock()
            .unwrap()
            .push(std::mem::replace(&mut lifetime, next));
        true
    }

    pub(crate) fn join_reclamation_workers(&self) -> Result<(), crate::ShutdownError> {
        let mut result = self.code_lifetime().join_reclamation_workers();
        for lifetime in self.previous_lifetimes.lock().unwrap().iter() {
            if let Err(error) = lifetime.join_reclamation_workers() {
                result = Err(error);
            }
        }
        result
    }

    pub(crate) fn with_code_lifetime(
        lifetime: Option<Arc<crate::code_lifetime::CodeLifetime>>,
    ) -> Self {
        let mut cache = Self::new();
        *cache.lifetime.get_mut().unwrap() = lifetime.unwrap_or_default();
        cache
    }

    /// Current invalidation epoch (R1). Monotonic; bumped whenever
    /// `invalidate_overlapping` drops at least one unit, or invalidates an in-flight
    /// fetch under an installed fetch policy. A vcpu snapshots this and flushes its
    /// local predictor caches when the value changes.
    pub fn epoch(&self) -> u64 {
        self.code_lifetime().epoch()
    }

    /// Source-publication generation. Snapshot before fetching source bytes or
    /// cached IR; insertion and upgrades check it under the spans lock. Every
    /// invalidation advances it, including writes with no overlapping cached unit.
    /// This does not flush vCPU dispatch predictors (see [`Self::epoch`]).
    pub fn publication_generation(&self) -> u64 {
        self.publication_generation.load(Ordering::Acquire)
    }

    /// Look up a block, recording a hit (found) or miss (must lift).
    #[inline(always)]
    pub fn get(&self, key: BlockKey) -> Option<CachedBlock> {
        let found = self.get_for_dispatch(key);
        if found.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        found
    }

    // Keep lookup temporaries out of the compiled chaining loop's register set.
    #[inline(never)]
    pub(crate) fn get_for_dispatch(&self, key: BlockKey) -> Option<CachedBlock> {
        self.map
            .read()
            .unwrap()
            .get(&key)
            .filter(|block| {
                !matches!(block, CachedBlock::Compiled { .. }) || self.code_lifetime().is_open()
            })
            .cloned()
    }

    pub(crate) fn add_dispatch_counts(&self, hits: u64, misses: u64) {
        if hits != 0 {
            self.hits.fetch_add(hits, Ordering::Relaxed);
        }
        if misses != 0 {
            self.misses.fetch_add(misses, Ordering::Relaxed);
        }
    }

    pub(crate) fn tier_up_time_stats(&self) -> (u64, u64) {
        (
            self.tier_up_latency_ns.load(Ordering::Relaxed),
            self.tier_up_latency_count.load(Ordering::Relaxed),
        )
    }

    /// Blocks served from the cache (no re-lift). Dispatcher activity is
    /// published at run return; direct `get` calls are counted immediately.
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Blocks that had to be lifted (distinct addresses, plus any re-lift after
    /// invalidation). Dispatcher activity is published at run return; direct
    /// `get` calls are counted immediately.
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Add `n` chained (link-slot) block transfers, once per `Vcpu::run` rather than
    /// once per transfer.
    ///
    /// Chaining is the engine's hottest event — a real game measured ~1,000,000 of
    /// them per frame against ~5,000 indirect-branch misses — so a shared
    /// `fetch_add` here is a locked RMW on a line every vcpu writes, on the hottest
    /// path there is. That is the same contention `Vcpu::fast_hits` already avoids by
    /// counting privately; this counter simply had not been given the same treatment.
    /// The vcpu accumulates in a plain `u64` and flushes once at the end of a run.
    pub fn add_chained(&self, n: u64) {
        if n != 0 {
            self.chained.fetch_add(n, Ordering::Relaxed);
        }
    }

    /// Chained transfers taken (the block-chaining "fires" counter, §12 M5).
    pub fn chained(&self) -> u64 {
        self.chained.load(Ordering::Relaxed)
    }

    /// Record a superblock (multi-block region) compilation.
    pub fn record_region(&self) {
        self.regions.fetch_add(1, Ordering::Relaxed);
    }

    /// Superblocks compiled as regions (M5-T3 "fires" counter).
    pub fn regions(&self) -> u64 {
        self.regions.load(Ordering::Relaxed)
    }

    pub(crate) fn tier_up_interpreter_only(&self, key: BlockKey) -> bool {
        let blocked = self.spans.read().unwrap().interpreter_only.contains(&key);
        if blocked {
            self.tier_bg_interpreter_only_skipped
                .fetch_add(1, Ordering::Relaxed);
        }
        blocked
    }

    /// Claim a background slot. Invalidation cancels its source snapshot but
    /// retains ownership until completion, preventing duplicate queued work.
    pub fn try_begin_tier_up(&self, key: BlockKey) -> bool {
        let mut spans = self.spans.write().unwrap();
        if spans.interpreter_only.contains(&key) {
            self.tier_bg_interpreter_only_skipped
                .fetch_add(1, Ordering::Relaxed);
            return false;
        }
        if spans.pending.contains_key(&key) {
            self.tier_bg_duplicates_skipped
                .fetch_add(1, Ordering::Relaxed);
            return false;
        }
        spans.next_request = spans
            .next_request
            .checked_add(1)
            .expect("tier-up request IDs exhausted");
        let id = spans.next_request;
        spans.pending.insert(
            key,
            PendingTierUp {
                queued_at: None,
                id,
                cancelled: false,
                generation: self.publication_generation(),
            },
        );
        spans.pending_bytes = spans
            .pending_bytes
            .max(table_bytes::<(BlockKey, PendingTierUp)>(
                spans.pending.capacity(),
            ));
        self.coverage_bytes
            .store(spans.coverage_bytes(), Ordering::Relaxed);
        true
    }

    /// Reconcile only retired request sources. Publication keeps its existing
    /// indexed insertion path; speculative lift tags have separate counted owners.
    pub(crate) fn release_source_tags(&self, mem: &crate::Memory, source: &[(u64, u32)]) {
        let spans = self.spans.read().unwrap();
        for &(start, len) in source {
            for page in span_pages(start, len) {
                if !spans.cached.pages.contains_key(&page)
                    && !spans.pending_sources.pages.contains_key(&page)
                {
                    mem.release_code_registration(page);
                }
            }
        }
    }

    pub(crate) fn abandon_tier_up(&self, key: BlockKey, mem: &crate::Memory) {
        let source = {
            let mut spans = self.spans.write().unwrap();
            spans.pending.remove(&key);
            if spans.pending.is_empty() {
                spans.pending = HashMap::new();
                spans.pending_bytes = 0;
            }
            let source = spans.pending_sources.remove(&key);
            self.coverage_bytes
                .store(spans.coverage_bytes(), Ordering::Relaxed);
            source
        };
        if let Some(source) = source {
            self.release_source_tags(mem, &source);
        }
    }

    /// Release an unsubmitted request's slot, or explicitly abandon ownership.
    pub fn end_tier_up(&self, key: BlockKey) {
        let mut spans = self.spans.write().unwrap();
        spans.pending.remove(&key);
        if spans.pending.is_empty() {
            spans.pending = HashMap::new();
            spans.pending_bytes = 0;
        }
        spans.pending_sources.remove(&key);
        self.coverage_bytes
            .store(spans.coverage_bytes(), Ordering::Relaxed);
    }

    /// Requests still awaiting a completion, including canceled source snapshots.
    pub fn tier_pending_len(&self) -> usize {
        self.spans.read().unwrap().pending.len()
    }

    // Source reads precede registration, so this short window still uses the
    // global generation. Once registered, only overlapping writes cancel a request.
    pub(crate) fn prepare_tier_up(
        &self,
        key: BlockKey,
        source_spans: Vec<(u64, u32)>,
        generation: u64,
        on_mark: impl FnOnce(&[(u64, u32)]),
    ) -> Option<u64> {
        let mut spans = self.spans.write().unwrap();
        let pending = spans.pending.get_mut(&key)?;
        // Until sources are registered, any intervening write cancels the
        // request through its claim generation without mutating the pending map.
        if pending.cancelled
            || pending.generation != generation
            || self.publication_generation() != generation
        {
            return None;
        }
        on_mark(&source_spans);
        pending.queued_at =
            crate::time_stats::enabled().then(crate::time_stats::MonotonicTime::now);
        let id = pending.id;
        spans.pending_sources.insert(key, source_spans);
        self.coverage_bytes
            .store(spans.coverage_bytes(), Ordering::Relaxed);
        Some(id)
    }

    pub(crate) fn publish_tier_up(
        &self,
        key: BlockKey,
        block: CachedBlock,
        compile_failure: Option<crate::CompileFailureReason>,
        request_id: u64,
        on_mark: impl FnOnce(&[(u64, u32)]),
    ) -> bool {
        let mut spans = self.spans.write().unwrap();
        if spans.pending.get(&key).is_none_or(|p| p.id != request_id) {
            self.record_tier_bg_rejected();
            self.tier_bg_superseded.fetch_add(1, Ordering::Relaxed);
            self.code_lifetime().discard(&block);
            return false;
        }
        let pending = spans.pending.remove(&key).unwrap();
        if spans.pending.is_empty() {
            spans.pending = HashMap::new();
            spans.pending_bytes = 0;
        }
        let source_spans = spans.pending_sources.remove(&key);
        self.coverage_bytes
            .store(spans.coverage_bytes(), Ordering::Relaxed);
        if pending.cancelled
            || (source_spans.is_none() && pending.generation != self.publication_generation())
        {
            self.record_tier_bg_rejected_generation();
            self.code_lifetime().discard(&block);
            return false;
        }
        let mut map = self.map.write().unwrap();
        if !matches!(map.get(&key), Some(CachedBlock::Interpreted(_))) {
            self.record_tier_bg_rejected();
            self.tier_bg_superseded.fetch_add(1, Ordering::Relaxed);
            self.code_lifetime().discard(&block);
            return false;
        }
        let source_spans = source_spans.expect("submitted tier-up has source spans");
        if matches!(block, CachedBlock::Interpreted(_)) {
            self.record_tier_bg_compile_failed();
            let reason = compile_failure.unwrap_or(crate::CompileFailureReason::OtherBackendError);
            let counter = match reason {
                crate::CompileFailureReason::DescriptorLimit => {
                    &self.tier_bg_failed_descriptor_limit
                }
                crate::CompileFailureReason::CodeSpace => &self.tier_bg_failed_code_space,
                crate::CompileFailureReason::OtherBackendError
                | crate::CompileFailureReason::Unclassified => &self.tier_bg_failed_other_backend,
            };
            counter.fetch_add(1, Ordering::Relaxed);
            if compile_failure.is_some_and(crate::CompileFailureReason::is_deterministic) {
                // Index all failed region sources, including uncached successors:
                // a write to any of them must release the suppression marker.
                on_mark(&source_spans);
                spans.insert(key, source_spans);
                self.coverage_bytes
                    .store(spans.coverage_bytes(), Ordering::Relaxed);
                spans.interpreter_only.insert(key);
                spans.suppression_bytes = spans
                    .suppression_bytes
                    .max(table_bytes::<BlockKey>(spans.interpreter_only.capacity()));
                self.coverage_bytes
                    .store(spans.coverage_bytes(), Ordering::Relaxed);
                self.tier_bg_interpreter_only
                    .fetch_add(1, Ordering::Relaxed);
            }
            self.code_lifetime().discard(&block);
            return false;
        }
        on_mark(&source_spans);
        if !self.account_replace(
            map.get(&key),
            spans.cached.units.get(&key).map_or(0, Vec::len),
            Some(&block),
            source_spans.len(),
        ) {
            self.record_tier_bg_rejected();
            return false;
        }
        map.insert(key, block);
        spans.insert(key, source_spans);
        self.coverage_bytes
            .store(spans.coverage_bytes(), Ordering::Relaxed);
        self.hotness.write().unwrap().remove(&key);
        self.record_tier_bg_published();
        if let Some(start) = pending.queued_at {
            let ns = start.elapsed_ns();
            self.tier_up_latency_ns
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                    Some(old.saturating_add(ns))
                })
                .ok();
            self.tier_up_latency_count
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                    Some(old.saturating_add(1))
                })
                .ok();
        }
        true
    }

    pub(crate) fn record_tier_bg_submitted(&self) {
        self.tier_bg_submitted.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_tier_bg_completed(&self) {
        self.tier_bg_completed.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_tier_bg_rejected_generation(&self) {
        self.record_tier_bg_rejected();
        self.tier_bg_rejected_generation
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_tier_bg_compile_failed(&self) {
        self.record_tier_bg_rejected();
        self.tier_bg_rejected_compile_failed
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record a background tier-up completion published into the cache.
    pub fn record_tier_bg_published(&self) {
        self.tier_bg_published.fetch_add(1, Ordering::Relaxed);
    }

    /// Background tier-up completions published (interp → compiled swap landed).
    pub fn tier_bg_published(&self) -> u64 {
        self.tier_bg_published.load(Ordering::Relaxed)
    }

    /// Record a background completion discarded at publication.
    pub fn record_tier_bg_rejected(&self) {
        self.tier_bg_rejected.fetch_add(1, Ordering::Relaxed);
    }

    /// Background tier-up completions dropped at publish time.
    pub fn tier_bg_rejected(&self) -> u64 {
        self.tier_bg_rejected.load(Ordering::Relaxed)
    }

    /// Cache a materialized unit keyed by `pc`, covering the given guest byte
    /// `spans` (one for a single block, several for a superblock). `on_mark` tags
    /// the covered code pages (§10) and runs **under the spans lock**, so it is
    /// serialized against `invalidate_overlapping`'s page-tag clear — a concurrent
    /// SMC drop can't wipe the tag of a block being inserted here.
    pub fn insert(
        &self,
        key: BlockKey,
        block: CachedBlock,
        spans: Vec<(u64, u32)>,
        on_mark: impl FnOnce(&[(u64, u32)]),
    ) {
        self.insert_checked(key, block, spans, |sp| {
            on_mark(sp);
            true
        });
    }

    pub(crate) fn insert_checked(
        &self,
        key: BlockKey,
        block: CachedBlock,
        spans: Vec<(u64, u32)>,
        on_mark: impl FnOnce(&[(u64, u32)]) -> bool,
    ) -> bool {
        let mut sp = self.spans.write().unwrap();
        let mut mp = self.map.write().unwrap();
        if !on_mark(&spans) {
            self.code_lifetime().discard(&block);
            return false;
        }
        if !self.account_replace(
            mp.get(&key),
            sp.cached.units.get(&key).map_or(0, Vec::len),
            Some(&block),
            spans.len(),
        ) {
            return false;
        }
        mp.insert(key, block);
        sp.insert(key, spans);
        self.coverage_bytes
            .store(sp.coverage_bytes(), Ordering::Relaxed);
        true
    }

    /// SMC invalidation (§10): drop every cached unit *any* of whose guest spans
    /// overlaps `[lo, hi)` and report their entry addresses. Only units indexed on
    /// the touched pages are checked, across every decode/FP mode.
    ///
    /// `on_clear_page` runs under the spans lock, only when no unit remains on
    /// any page touched by `[lo, hi)`. Thus byte-precise writes preserve live page
    /// tags, and a concurrent insert cannot have its mark wiped by a late clear.
    pub fn invalidate_overlapping(
        &self,
        lo: u64,
        hi: u64,
        on_clear_page: impl FnOnce(),
    ) -> Vec<BlockKey> {
        self.invalidate_range::<false>(lo, hi, |empty| {
            if empty {
                on_clear_page();
            }
        })
    }

    pub(crate) fn invalidate_child_written(
        &self,
        lo: u64,
        hi: u64,
        on_clear_page: impl FnOnce(),
    ) -> Vec<BlockKey> {
        self.invalidate_range::<true>(lo, hi, |empty| {
            if empty {
                on_clear_page();
            }
        })
    }

    // Drained writes serialize with source publication, including victim-free writes.
    pub(crate) fn invalidate_written(
        &self,
        lo: u64,
        hi: u64,
        on_write: impl FnOnce(bool),
    ) -> Vec<BlockKey> {
        self.invalidate_range::<false>(lo, hi, on_write)
    }

    fn invalidate_range<const CHILD: bool>(
        &self,
        lo: u64,
        hi: u64,
        on_write: impl FnOnce(bool),
    ) -> Vec<BlockKey> {
        self.invalidation_calls.fetch_add(1, Ordering::Relaxed);
        {
            let spans = self.spans.read().unwrap();
            let (overlap, on_page) = spans.cached.coverage_overlap(lo, hi);
            let (pending_overlap, pending_on_page) = spans.pending_sources.coverage_overlap(lo, hi);
            if !overlap && !pending_overlap {
                // §10: publication requires the exclusive spans lock. If it won
                // first, this reader sees its coverage bits. Otherwise this write
                // advances the generation before releasing the read lock, so the
                // publisher rejects source bytes read before the store. Concurrent
                // victim-free writers may share this lock; fetch_add preserves all
                // generation advances. Temporary lift tags precede source reads and
                // survive this clear even though they have no coverage bits yet.
                // Registration/publication drains queued tagged-store notifications
                // before checking the pre-read generation under the exclusive lock;
                // a miss drained by another vCPU still bumps that generation here.
                // Thus an unregistered lift cannot publish its pre-store snapshot.
                // After registration, coverage overlap cancels the request instead.
                // Page-tag clears also remain under the lock.
                on_write(!on_page && !pending_on_page);
                self.publication_generation.fetch_add(1, Ordering::Release);
                if CHILD || self.invalidate_pending_fetches {
                    self.code_lifetime().flush_predictions();
                }
                return Vec::new();
            }
        }
        // Recompute after upgrading the lock: publication may have intervened.
        let mut spans = self.spans.write().unwrap();
        let victims = spans.cached.victims(lo, hi);
        let mut map = (!victims.is_empty()).then(|| self.map.write().unwrap());
        if let Some(map) = map.as_mut() {
            self.invalidation_blocks_dropped
                .fetch_add(victims.len() as u64, Ordering::Relaxed);
            let mut hotness = self.hotness.write().unwrap();
            self.code_lifetime()
                .remove_publications(victims.iter().filter_map(|key| map.get(key)));
            for entry in &victims {
                self.account_memory_replace(
                    map.get(entry),
                    spans.cached.units.get(entry).map_or(0, Vec::len),
                    None,
                    0,
                );
                spans.interpreter_only.remove(entry);
                if spans.interpreter_only.is_empty() {
                    spans.interpreter_only = HashSet::new();
                    spans.suppression_bytes = 0;
                }
                spans.cached.remove(entry);
                map.remove(entry);
                hotness.remove(entry);
            }
            self.coverage_bytes
                .store(spans.coverage_bytes(), Ordering::Relaxed);
        }
        if lo < hi {
            for key in spans.pending_sources.victims(lo, hi) {
                spans.pending.get_mut(&key).unwrap().cancelled = true;
                spans.pending_sources.remove(&key);
            }
        }
        self.coverage_bytes
            .store(spans.coverage_bytes(), Ordering::Relaxed);
        on_write(!spans.pending_sources.has_units(lo, hi) && !spans.cached.has_units(lo, hi));
        // A source span may have been read but not cached yet. Reject its
        // publication even when all currently cached spans survive this write.
        self.publication_generation.fetch_add(1, Ordering::Release);
        if CHILD || !victims.is_empty() || self.invalidate_pending_fetches {
            self.code_lifetime().flush_predictions();
        }
        victims
    }
}

impl Drop for TranslationCache {
    fn drop(&mut self) {
        self.code_lifetime().remove_publications(
            self.map
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .values(),
        );
    }
}

impl Default for TranslationCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compiled() -> CachedBlock {
        CachedBlock::Compiled {
            entry: CompiledPtr::new(std::ptr::null()),
        }
    }

    /// A Long64 key at `addr` — the shape most cache tests key by.
    fn k(addr: u64) -> BlockKey {
        BlockKey::new(addr, CpuMode::Long64)
    }

    #[test]
    fn overlap_counts_are_exact_past_narrow_overflow_and_release_early() {
        let mut coverage = CoverageIndex::default();
        coverage.cover(0, 63, 66);
        let single = coverage.bytes;
        for _ in 1..600 {
            coverage.cover(0, 63, 66);
        }
        let extra = coverage.page_mut(0).extra.as_ref().unwrap();
        assert_eq!(extra.overflow.len(), 3);
        assert_eq!(
            coverage.bytes,
            single
                + std::mem::size_of::<ExtraCoverage>() as u64
                + 4 * std::mem::size_of::<(u16, usize)>() as u64
        );
        for owners in (1..600).rev() {
            coverage.uncover(0, 63, 66);
            assert_eq!(coverage.page(0).unwrap()[0], 1 << 63);
            assert_eq!(coverage.page(0).unwrap()[1], 3);
            if owners == 16 {
                let extra = coverage.page_mut(0).extra.as_ref().unwrap();
                assert_eq!(extra.overflow.capacity(), 0);
                assert_eq!(
                    coverage.bytes,
                    single + std::mem::size_of::<ExtraCoverage>() as u64
                );
            }
            if owners == 1 {
                assert_eq!(coverage.bytes, single);
                assert!(coverage.page_mut(0).extra.is_none());
            }
        }
        coverage.uncover(0, 63, 66);
        assert!(coverage.page(0).unwrap().iter().all(|&bits| bits == 0));
        coverage.remove_page(0);
        assert_eq!(coverage.bytes, 0);
    }

    #[test]
    fn index_memory_counts_unregistered_and_cancelled_pending_slots() {
        let c = TranslationCache::new();
        assert!(c.try_begin_tier_up(k(0x1000)));
        let claim = c.memory_stats().coverage_index_bytes;
        assert!(claim > 0);
        assert!(c
            .prepare_tier_up(
                k(0x1000),
                vec![(0x1000, 4)],
                c.publication_generation(),
                |_| {}
            )
            .is_some());
        assert!(c.memory_stats().coverage_index_bytes > claim);
        c.invalidate_overlapping(0x1000, 0x1001, || {});
        assert_eq!(c.memory_stats().coverage_index_bytes, claim);
        c.end_tier_up(k(0x1000));
        assert_eq!(c.memory_stats().coverage_index_bytes, 0);
    }

    #[test]
    fn compact_membership_checks_gaps_and_repeated_spans() {
        let mut index = SourceIndex::default();
        index.insert(k(0), vec![(10, 4), (100, 4), (10, 4)]);
        assert!(index.victims(50, 51).is_empty());
        assert_eq!(index.victims(103, 104), vec![k(0)]);
        index.remove(&k(0));
        assert_eq!(index.bytes(), 0);
        assert_eq!(std::mem::size_of::<(BlockKey, PageSpans)>(), 24);
    }

    #[test]
    fn index_memory_reclaims_full_membership_tables_after_removal() {
        let mut index = SourceIndex::default();
        for i in 0..1792 {
            index.insert(k(i), vec![(0, 1)]);
        }
        for i in 0..1792 {
            index.remove(&k(i));
        }
        assert_eq!(index.bytes(), 0);
    }

    #[test]
    fn index_memory_counts_retained_pending_capacity_after_deletion() {
        let c = TranslationCache::new();
        for i in 0..1792 {
            assert!(c.try_begin_tier_up(k(i)));
        }
        let allocated = c.memory_stats().coverage_index_bytes;
        for i in 0..1791 {
            c.end_tier_up(k(i));
            assert_eq!(c.memory_stats().coverage_index_bytes, allocated);
        }
        c.end_tier_up(k(1791));
        assert_eq!(c.memory_stats().coverage_index_bytes, 0);
    }

    #[test]
    fn realistic_cache_index_memory_report() {
        let c = TranslationCache::new();
        for i in 0..31_000u64 {
            let page = (i / 32 + 1) * 4096;
            let slot = i % 32;
            let spans = if slot < 16 {
                vec![(page + slot * 128, 64)]
            } else {
                (0..4)
                    .map(|j| {
                        (
                            page + ((slot - 16 + j) % 16) * 128 + u64::from(j == 0) * 4,
                            if j == 0 { 60 } else { 64 },
                        )
                    })
                    .collect()
            };
            c.insert(
                k(page + (slot % 16) * 128 + u64::from(slot >= 16) * 4),
                compiled(),
                spans,
                |_| {},
            );
        }
        let bytes = c.memory_stats().coverage_index_bytes;
        println!("cache_index,keys=31000,keys_per_page=32,bytes={bytes}");
        // Same workload measured on main 73ed5d6, including membership tables
        // and source-vector capacity with the same table allocation estimate.
        const MAIN_INDEX_BYTES: u64 = 5_112_784;
        assert!(bytes <= 2 * MAIN_INDEX_BYTES, "index bytes: {bytes}");
        let spans = c.spans.read().unwrap();
        let index = &spans.cached;
        let allocations = index
            .units
            .values()
            .map(|spans| spans.capacity() * std::mem::size_of::<(u64, u32)>())
            .sum::<usize>() as u64
            + index
                .pages
                .values()
                .map(|membership| membership.bytes)
                .sum::<u64>();
        assert_eq!(index.allocation_bytes, allocations);
    }

    #[test]
    fn disjoint_writes_share_the_span_read_lock_with_registered_sources() {
        use std::sync::mpsc;
        use std::time::Duration;
        for pending in [false, true] {
            let c = Arc::new(TranslationCache::new());
            if pending {
                pending_region(&c);
            } else {
                c.insert(k(0x1000), compiled(), vec![(0x1000, 4)], |_| {});
            }
            let generation = c.publication_generation();
            let reader = c.spans.read().unwrap();
            let (tx, rx) = mpsc::channel();
            let writer = {
                let c = c.clone();
                std::thread::spawn(move || {
                    let result = c.invalidate_written(0x3800, 0x3801, |empty| {
                        assert_eq!(empty, !pending);
                    });
                    tx.send(result).unwrap();
                })
            };
            let result = rx.recv_timeout(Duration::from_secs(5));
            drop(reader);
            writer.join().unwrap();
            assert!(result
                .expect("disjoint write must finish with a shared reader")
                .is_empty());
            assert_eq!(c.publication_generation(), generation + 1);
        }
    }

    #[test]
    fn removing_sources_never_rebuilds_unrelated_page_survivors() {
        for pending in [false, true] {
            let c = TranslationCache::new();
            for i in 0..1000 {
                let key = k(0x4000 + i);
                if pending {
                    assert!(c.try_begin_tier_up(key));
                    assert!(c
                        .prepare_tier_up(
                            key,
                            vec![(0x3000 + i, 1)],
                            c.publication_generation(),
                            |_| {}
                        )
                        .is_some());
                } else {
                    c.insert(key, compiled(), vec![(0x3000 + i, 1)], |_| {});
                }
            }
            let victim = k(0x1000);
            // Repeated and overlapping spans within one unit count separately.
            let source = vec![(0x1000, 4), (0x3000, 66), (0x3040, 4), (0x3000, 66)];
            if pending {
                assert!(c.try_begin_tier_up(victim));
                assert!(c
                    .prepare_tier_up(victim, source, c.publication_generation(), |_| {})
                    .is_some());
            } else {
                c.insert(victim, compiled(), source, |_| {});
            }
            let victims = c.invalidate_overlapping(0x1000, 0x1001, || {});
            assert_eq!(victims, if pending { vec![] } else { vec![victim] });
            let spans = c.spans.read().unwrap();
            let index = if pending {
                &spans.pending_sources
            } else {
                &spans.cached
            };
            assert_eq!(index.coverage.uncovered_ranges, 4);
            assert_eq!(index.units.len(), 1000);
            assert_eq!(index.victims(0x3041, 0x3042), vec![k(0x4041)]);
            assert!(!index.coverage_overlap(0x33e8, 0x33e9).0);
        }
    }

    #[test]
    fn many_page_victims_are_emitted_once_per_unit() {
        let mut index = SourceIndex::default();
        let spans: Vec<_> = (1..=2048).rev().map(|page| (page * 4096 + 63, 2)).collect();
        index.insert(k(0x1000), spans.clone());
        index.insert(k(0x2000), spans);
        let mut victims = index.victims(0, u64::MAX);
        victims.sort_by_key(|key| key.guest_addr);
        assert_eq!(victims, vec![k(0x1000), k(0x2000)]);
        assert_eq!(index.victims(0x1040, 0x1041).len(), 2);
        assert!(index.victims(0x1041, 0x2000).is_empty());
        index.remove(&k(0x1000));
        assert_eq!(index.victims(0, u64::MAX), vec![k(0x2000)]);
        index.remove(&k(0x2000));
        assert_eq!(index.coverage.bytes, 0);
    }

    #[test]
    fn byte_coverage_preserves_shared_tails_and_reclaims_sparse_storage() {
        let c = TranslationCache::new();
        let entries = [
            (k(0x1000), vec![(0x103f, 66), (0x1ffe, 4)]),
            (k(0x1100), vec![(0x1040, 2), (0x2000, 4)]),
            (k(u64::MAX - 8), vec![(u64::MAX - 8, 4)]),
        ];
        for (key, spans) in &entries {
            c.insert(*key, compiled(), spans.clone(), |_| {});
        }
        assert!(c.memory_stats().coverage_index_bytes >= 3 * 512);
        for addr in 0x1000..0x3000 {
            let expected = entries
                .iter()
                .flat_map(|(_, spans)| spans)
                .any(|&(start, len)| start <= addr && addr - start < u64::from(len));
            assert_eq!(
                c.spans
                    .read()
                    .unwrap()
                    .cached
                    .coverage_overlap(addr, addr + 1)
                    .0,
                expected,
                "coverage at {addr:#x}"
            );
        }
        assert_eq!(
            c.invalidate_overlapping(0x103f, 0x1040, || {}),
            vec![entries[0].0]
        );
        // The removed block shared these bytes with a different unit.
        assert_eq!(
            c.invalidate_overlapping(0x1041, 0x1042, || {}),
            vec![entries[1].0]
        );
        assert!(c.invalidate_overlapping(0x2000, 0x2001, || {}).is_empty());
        assert_eq!(
            c.invalidate_overlapping(u64::MAX - 5, u64::MAX - 4, || {}),
            vec![entries[2].0]
        );
        assert_eq!(c.memory_stats().coverage_index_bytes, 0);
        c.insert(k(0), compiled(), vec![(0, 1)], |_| {});
        assert_eq!(
            c.memory_stats().coverage_index_bytes,
            std::mem::size_of::<PageCoverage>() as u64
                + table_bytes::<(BlockKey, Vec<(u64, u32)>)>(3)
                + table_bytes::<(u64, PageMembership)>(3)
                + table_bytes::<(BlockKey, PageSpans)>(3)
                + std::mem::size_of::<(u64, u32)>() as u64
        );
        assert_eq!(c.invalidate_overlapping(0, 1, || {}), vec![k(0)]);
        assert_eq!(c.memory_stats().coverage_index_bytes, 0);
    }

    #[test]
    fn concurrent_rejected_writes_both_advance_publication_generation() {
        let c = TranslationCache::new();
        c.insert(k(0x1000), compiled(), vec![(0x1000, 4)], |_| {});
        let generation = c.publication_generation();
        let rendezvous = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            for lo in [0x1800, 0x1801] {
                let rendezvous = &rendezvous;
                let c = &c;
                scope.spawn(move || {
                    assert!(c
                        .invalidate_written(lo, lo + 1, |empty| {
                            assert!(!empty);
                            rendezvous.wait();
                        })
                        .is_empty());
                });
            }
        });
        assert_eq!(c.publication_generation(), generation + 2);
        assert!(!c.upgrade(k(0x1000), compiled(), (0x1000, 4), generation));
    }

    #[test]
    fn disjoint_write_cancels_unregistered_request_without_exclusive_lock() {
        let c = TranslationCache::new();
        assert!(c.try_begin_tier_up(k(0x1000)));
        let reader = c.spans.read().unwrap();
        assert!(c.invalidate_overlapping(0x1800, 0x1801, || {}).is_empty());
        drop(reader);
        // Re-reading the generation must not revive a claim canceled during its
        // source-read window. Ownership still lasts until end_tier_up.
        assert!(c
            .prepare_tier_up(
                k(0x1000),
                vec![(0x1000, 4)],
                c.publication_generation(),
                |_| {}
            )
            .is_none());
        assert!(!c.try_begin_tier_up(k(0x1000)));
        c.end_tier_up(k(0x1000));
    }

    #[test]
    fn pending_coverage_is_reclaimed_on_end_and_cancellation() {
        let c = TranslationCache::new();
        pending_region(&c);
        let with_pending = c.memory_stats().coverage_index_bytes;
        c.end_tier_up(k(0x1000));
        assert!(c.memory_stats().coverage_index_bytes < with_pending);
        let cached_only = c.memory_stats().coverage_index_bytes;
        let id = pending_region(&c);
        c.invalidate_overlapping(0x3001, 0x3002, || {});
        assert!(c.memory_stats().coverage_index_bytes > cached_only);
        assert!(!c.publish_tier_up(k(0x1000), compiled(), None, id, |_| {}));
        c.invalidate_overlapping(0x1000, 0x1004, || {});
        assert_eq!(c.memory_stats().coverage_index_bytes, 0);
    }

    #[test]
    fn abandoning_request_releases_index_and_tags_but_preserves_lift_owner() {
        let mem = crate::Memory::new(crate::MemoryModel::Flat { size: 0x6000 });
        let cache = TranslationCache::new();
        let lift = crate::memory::LiftSource::new(&mem);
        lift.watch(0x3000, 4);
        assert!(cache.try_begin_tier_up(k(0x1000)));
        cache
            .prepare_tier_up(
                k(0x1000),
                vec![(0x3000, 4)],
                cache.publication_generation(),
                |_| mem.mark_code(0x3000, 4),
            )
            .unwrap();
        cache.abandon_tier_up(k(0x1000), &mem);
        assert_eq!(cache.tier_pending_len(), 0);
        assert_eq!(cache.memory_stats().coverage_index_bytes, 0);
        assert!(mem.is_code_page(3));
        drop(lift);
        assert!(!mem.is_code_page(3));
    }

    #[test]
    fn retiring_request_tags_preserves_other_owners() {
        let mem = crate::Memory::new(crate::MemoryModel::Flat { size: 0x6000 });
        let cache = TranslationCache::new();
        let source = vec![(0x3000, 4), (0x4000, 4)];
        for key in [k(0x1000), k(0x2000)] {
            assert!(cache.try_begin_tier_up(key));
            cache
                .prepare_tier_up(key, source.clone(), cache.publication_generation(), |sp| {
                    for &(start, len) in sp {
                        mem.mark_code(start, len);
                    }
                })
                .unwrap();
        }
        cache.end_tier_up(k(0x1000));
        cache.release_source_tags(&mem, &source);
        assert!(mem.is_code_page(3)); // the other registered request
        cache.insert(k(0x4000), compiled(), vec![(0x4000, 4)], |_| {
            mem.mark_code(0x4000, 4)
        });
        cache.end_tier_up(k(0x2000));
        cache.release_source_tags(&mem, &source);
        assert!(!mem.is_code_page(3));
        assert!(mem.is_code_page(4)); // a cached owner on the same page
    }

    #[test]
    fn byte_overlap_keeps_other_modes_and_crossing_page_tags() {
        let c = TranslationCache::new();
        let crossing = k(0x1ffe);
        let modes = [CpuMode::Long64, CpuMode::Compat32, CpuMode::Real16];
        c.insert(crossing, compiled(), vec![(0x1ffe, 4)], |_| {});
        for mode in modes {
            for mxcsr in [0, 0x2000] {
                let key = BlockKey::with_mxcsr(0x2100, mode, mxcsr);
                c.insert(key, compiled(), vec![(0x2100, 8)], |_| {});
            }
        }
        let epoch = c.epoch();
        assert!(c
            .invalidate_overlapping(0x2f00, 0x2f01, || panic!("live page"))
            .is_empty());
        assert_eq!(c.epoch(), epoch);
        assert_eq!(
            c.invalidate_overlapping(0x2000, 0x2001, || panic!("live page")),
            vec![crossing]
        );
        let spans = c.spans.read().unwrap();
        assert!(!spans.cached.pages.contains_key(&1));
        assert_eq!(spans.cached.pages[&2].entries.len(), 6);
        drop(spans);
        let victims = c.invalidate_overlapping(0x2107, 0x2108, || {});
        assert_eq!(victims.len(), 6);
        assert!(c.spans.read().unwrap().cached.pages.is_empty());
    }

    #[test]
    fn replacement_and_region_upgrade_update_every_page() {
        let c = TranslationCache::new();
        let key = k(0x1000);
        c.insert(key, compiled(), vec![(0x1000, 4), (0x3ffe, 4)], |_| {});
        c.insert(key, compiled(), vec![(0x5000, 2)], |_| {});
        assert!(c.invalidate_overlapping(0x1000, 0x5000, || {}).is_empty());
        assert!(c.upgrade(key, compiled(), (0x6000, 2), c.publication_generation()));
        assert!(c.invalidate_overlapping(0x5000, 0x6000, || {}).is_empty());
        assert!(c.upgrade_region(
            key,
            compiled(),
            vec![(0x7fff, 2), (0xa000, 4)],
            c.publication_generation(),
            |_| {}
        ));
        assert!(c.invalidate_overlapping(0x6000, 0x7000, || {}).is_empty());
        assert_eq!(c.invalidate_overlapping(0x8000, 0x8001, || {}), vec![key]);
        assert!(c.spans.read().unwrap().cached.pages.is_empty());
        c.insert(key, compiled(), vec![(0x1000, 4), (0xffff_0000, 4)], |_| {});
        assert_eq!(c.invalidate_overlapping(0, u64::MAX, || {}), vec![key]);
        assert!(c.spans.read().unwrap().cached.pages.is_empty());
    }

    /// No race: a tier-up whose generation still matches commits, and the block
    /// stays invalidatable (its span is present).
    #[test]
    fn tier_up_commits_and_keeps_span() {
        let c = TranslationCache::new();
        c.insert(k(0x1000), compiled(), vec![(0x1000, 4)], |_| {});
        let e = c.publication_generation();
        assert!(c.upgrade(k(0x1000), compiled(), (0x1000, 4), e));
        // A later write to the block's page must still find it via its span.
        assert_eq!(
            c.invalidate_overlapping(0x1000, 0x1004, || {}),
            vec![k(0x1000)]
        );
    }

    /// An `invalidate_overlapping` drops the unit between the tier-up's generation
    /// snapshot and its `upgrade`. `upgrade` must reject
    /// the stale compile rather than resurrect a spanless — permanently
    /// uninvalidatable — block.
    #[test]
    fn tier_up_rejected_when_invalidated_mid_upgrade() {
        let c = TranslationCache::new();
        c.insert(k(0x1000), compiled(), vec![(0x1000, 4)], |_| {});
        let e = c.publication_generation(); // snapshot BEFORE the racing invalidation

        // Concurrent SMC drop: removes the unit and bumps the epoch.
        assert_eq!(
            c.invalidate_overlapping(0x1000, 0x1004, || {}),
            vec![k(0x1000)]
        );
        assert!(c.get(k(0x1000)).is_none(), "invalidation dropped the block");

        // The stale tier-up now tries to commit with the pre-drop generation.
        assert!(
            !c.upgrade(k(0x1000), compiled(), (0x1000, 4), e),
            "upgrade must reject a compile the SMC drop raced past"
        );
        assert!(
            c.get(k(0x1000)).is_none(),
            "must not resurrect the block (a spanless entry would be permanent)"
        );
    }

    /// The background tier-up in-flight set. A pc is claimable
    /// once; a second claim while pending is rejected (no double-submit); `end_tier_up`
    /// releases it and is idempotent.
    #[test]
    fn tier_pending_set_transitions() {
        let c = TranslationCache::new();

        assert!(c.try_begin_tier_up(k(0x1000)), "first claim owns the slot");
        assert!(
            !c.try_begin_tier_up(k(0x1000)),
            "double-begin rejected while pending"
        );

        c.end_tier_up(k(0x1000));
        assert!(c.try_begin_tier_up(k(0x1000)), "claimable again after end");

        // Idempotent: extra ends are harmless, and a distinct pc is independent.
        c.end_tier_up(k(0x1000));
        c.end_tier_up(k(0x1000));
        assert!(
            c.try_begin_tier_up(k(0x1000)),
            "still claimable after double-end"
        );
        assert!(
            c.try_begin_tier_up(k(0x2000)),
            "a different pc has its own slot"
        );
        c.end_tier_up(k(0x1000));
        c.end_tier_up(k(0x2000));
    }

    /// A canceled request retains its slot until completion, so re-lifting cannot
    /// enqueue duplicate work while the old compile is still running.
    #[test]
    fn invalidate_retains_pending_marker_until_completion() {
        let c = TranslationCache::new();
        c.insert(k(0x1000), compiled(), vec![(0x1000, 4)], |_| {});
        assert!(c.try_begin_tier_up(k(0x1000)), "claim the in-flight slot");

        assert_eq!(
            c.invalidate_overlapping(0x1000, 0x1004, || {}),
            vec![k(0x1000)]
        );

        assert!(
            !c.try_begin_tier_up(k(0x1000)),
            "canceled request still owns the slot"
        );
        c.end_tier_up(k(0x1000));
        assert!(c.try_begin_tier_up(k(0x1000)));
    }

    /// The background tier-up "fires" counters start at zero and count monotonically.
    #[test]
    fn tier_bg_counters_count() {
        let c = TranslationCache::new();
        assert_eq!((c.tier_bg_published(), c.tier_bg_rejected()), (0, 0));
        c.record_tier_bg_published();
        c.record_tier_bg_published();
        c.record_tier_bg_rejected();
        assert_eq!((c.tier_bg_published(), c.tier_bg_rejected()), (2, 1));
    }

    /// §17.4: the block key carries the decode mode, so the same guest address holds
    /// **distinct** translations under Long64 vs Compat32 — they never alias.
    #[test]
    fn same_addr_distinct_entry_per_mode() {
        let c = TranslationCache::new();
        let long = BlockKey::new(0x1000, CpuMode::Long64);
        let compat = BlockKey::new(0x1000, CpuMode::Compat32);
        assert_ne!(long, compat);

        c.insert(long, compiled(), vec![(0x1000, 4)], |_| {});
        assert!(c.get(long).is_some(), "the Long64 translation is present");
        assert!(
            c.get(compat).is_none(),
            "the same address in Compat32 is a distinct, absent key"
        );

        c.insert(compat, compiled(), vec![(0x1000, 4)], |_| {});
        assert!(
            c.get(long).is_some() && c.get(compat).is_some(),
            "both modes coexist at one address"
        );

        // SMC to the page is address-scoped: it drops both mode's translations.
        let victims = c.invalidate_overlapping(0x1000, 0x1004, || {});
        assert_eq!(victims.len(), 2, "SMC drops every mode at the address");
        assert!(c.get(long).is_none() && c.get(compat).is_none());
    }

    /// `insert` tags the page and `invalidate_overlapping` clears it, both
    /// through their callbacks under the spans lock — so an insert's mark and an SMC
    /// drop's clear can't interleave and wipe a live block's tag.
    #[test]
    fn insert_marks_and_invalidate_clears_page_tag() {
        use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
        let c = TranslationCache::new();
        let tag = AtomicBool::new(false);
        c.insert(k(0x1000), compiled(), vec![(0x1000, 4)], |_| {
            tag.store(true, Relaxed)
        });
        assert!(tag.load(Relaxed), "insert tagged the page");
        let v = c.invalidate_overlapping(0x1000, 0x2000, || tag.store(false, Relaxed));
        assert_eq!(v, vec![k(0x1000)]);
        assert!(!tag.load(Relaxed), "invalidate cleared the page tag");
    }

    /// An entry resolved before an invalidation must NOT be published after it (R1).
    ///
    /// This is the deterministic half of the link/IBTC race. The real interleaving —
    /// vcpu A between `resolve` and its slot store while vcpu B drops the translation —
    /// cannot be forced from a test without a hook in the dispatcher that the core does
    /// not have and should not grow for this. What CAN be pinned exactly is the rule the
    /// dispatcher hands the decision to, and that is what this does: publish with a
    /// stale epoch and it must be refused, with the closure never run.
    #[test]
    fn a_publish_from_a_stale_epoch_is_refused() {
        use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
        let c = TranslationCache::new();
        c.insert(k(0x1000), compiled(), vec![(0x1000, 4)], |_| {});

        // What a vcpu snapshots before calling `resolve`.
        let since = c.epoch();

        // ...and what another vcpu does while it resolves. Only a real drop bumps the
        // epoch, so this also checks that the guard keys off invalidation rather than
        // off any cache activity.
        let v = c.invalidate_overlapping(0x1000, 0x2000, || {});
        assert_eq!(v, vec![k(0x1000)], "the drop must produce a victim");
        assert_ne!(c.epoch(), since, "a real drop moves the epoch");

        let published = AtomicBool::new(false);
        assert!(
            !c.publish_if_current(since, || published.store(true, Relaxed)),
            "a stale publish must be refused"
        );
        assert!(
            !published.load(Relaxed),
            "refused means the slot store never ran — reporting false after publishing \
             would leave the stale pointer in the slot, which is the whole defect"
        );

        // The current epoch still publishes, so the guard is a gate and not a wall.
        assert!(c.publish_if_current(c.epoch(), || published.store(true, Relaxed)));
        assert!(published.load(Relaxed));
    }

    /// A write to a page with no translations must not move the epoch, or every data
    /// store would make every in-flight link publish spuriously fail.
    #[test]
    fn a_write_with_no_victims_leaves_the_epoch_alone() {
        let c = TranslationCache::new();
        c.insert(k(0x1000), compiled(), vec![(0x1000, 4)], |_| {});
        let since = c.epoch();
        let v = c.invalidate_overlapping(0x9000, 0xa000, || {});
        assert!(v.is_empty());
        assert_eq!(c.epoch(), since);
        assert!(c.publish_if_current(since, || {}));
    }

    fn pending_region(c: &TranslationCache) -> u64 {
        let ir = Arc::new(IrBlock {
            legacy_context: None,
            mode: CpuMode::Long64,
            guest_start: 0x1000,
            guest_len: 4,
            icount: 1,
            temp_count: 0,
            ops: Vec::new(),
        });
        c.insert(
            k(0x1000),
            CachedBlock::Interpreted(ir),
            vec![(0x1000, 4)],
            |_| {},
        );
        let generation = c.publication_generation();
        assert!(c.try_begin_tier_up(k(0x1000)));
        c.prepare_tier_up(
            k(0x1000),
            vec![(0x1000, 4), (0x3000, 4)],
            generation,
            |_| {},
        )
        .unwrap()
    }

    #[test]
    fn deterministic_failure_stays_suppressed_until_any_source_is_invalidated() {
        for reason in [
            crate::CompileFailureReason::DescriptorLimit,
            crate::CompileFailureReason::OtherBackendError,
        ] {
            for (lo, hi) in [(0x1001, 0x1002), (0x3001, 0x3002)] {
                let c = TranslationCache::new();
                let id = pending_region(&c);
                let interpreted = c.get(k(0x1000)).unwrap();
                assert!(!c.publish_tier_up(
                    k(0x1000),
                    interpreted,
                    Some(reason),
                    id,
                    |spans| assert_eq!(spans.len(), 2)
                ));
                assert_eq!(c.tier_pending_len(), 0);
                for _ in 0..32 {
                    assert!(!c.try_begin_tier_up(k(0x1000)));
                }
                let stats = c.memory_stats();
                assert_eq!(stats.tier_bg_interpreter_only, 1);
                assert_eq!(stats.tier_bg_interpreter_only_skipped, 32);
                assert_eq!(stats.tier_bg_rejected_compile_failed, 1);
                assert_eq!(
                    stats.tier_bg_failed_descriptor_limit + stats.tier_bg_failed_other_backend,
                    1
                );
                assert!(c.invalidate_overlapping(0x3800, 0x3801, || {}).is_empty());
                assert!(!c.try_begin_tier_up(k(0x1000)));
                assert_eq!(c.invalidate_overlapping(lo, hi, || {}), vec![k(0x1000)]);
                assert!(c.try_begin_tier_up(k(0x1000)));
            }
        }
    }

    #[test]
    fn stale_failure_cannot_suppress_new_bytes() {
        let c = TranslationCache::new();
        let id = pending_region(&c);
        c.invalidate_overlapping(0x3000, 0x3001, || {});
        let interpreted = c.get(k(0x1000)).unwrap();
        assert!(!c.publish_tier_up(
            k(0x1000),
            interpreted,
            Some(crate::CompileFailureReason::DescriptorLimit),
            id,
            |_| panic!("stale failure")
        ));
        assert!(c.try_begin_tier_up(k(0x1000)));
        assert_eq!(c.memory_stats().tier_bg_interpreter_only, 0);
        assert_eq!(c.memory_stats().tier_bg_rejected_generation, 1);
    }

    #[test]
    fn superseded_failure_cannot_suppress_a_newer_pending_request() {
        let c = TranslationCache::new();
        let old = pending_region(&c);
        let interpreted = c.get(k(0x1000)).unwrap();
        c.end_tier_up(k(0x1000));
        let new = pending_region(&c);
        assert!(!c.publish_tier_up(
            k(0x1000),
            interpreted,
            Some(crate::CompileFailureReason::DescriptorLimit),
            old,
            |_| panic!("superseded failure")
        ));
        assert_eq!(c.tier_pending_len(), 1);
        assert_eq!(c.memory_stats().tier_bg_interpreter_only, 0);
        assert_eq!(c.memory_stats().tier_bg_superseded, 1);
        assert!(c.publish_tier_up(k(0x1000), compiled(), None, new, |_| {}));
    }

    #[test]
    fn unclassified_failure_is_counted_without_permanent_suppression() {
        let c = TranslationCache::new();
        let id = pending_region(&c);
        let interpreted = c.get(k(0x1000)).unwrap();
        assert!(!c.publish_tier_up(k(0x1000), interpreted, None, id, |_| {}));
        assert_eq!(c.memory_stats().tier_bg_failed_other_backend, 1);
        assert_eq!(c.memory_stats().tier_bg_interpreter_only, 0);
        assert!(c.try_begin_tier_up(k(0x1000)));
        c.end_tier_up(k(0x1000));
        let id = pending_region(&c);
        let interpreted = c.get(k(0x1000)).unwrap();
        assert!(!c.publish_tier_up(
            k(0x1000),
            interpreted,
            Some(crate::CompileFailureReason::Unclassified),
            id,
            |_| {}
        ));
        assert_eq!(c.memory_stats().tier_bg_failed_other_backend, 2);
        assert_eq!(c.memory_stats().tier_bg_interpreter_only, 0);
        assert!(c.try_begin_tier_up(k(0x1000)));
    }

    #[test]
    fn pending_sources_cancel_only_on_overlap() {
        for (lo, hi, publish) in [
            (0x1800, 0x1801, true),
            (0x3800, 0x3801, true),
            (0x1003, 0x1004, false),
            (0x3001, 0x3002, false),
        ] {
            let c = TranslationCache::new();
            let id = pending_region(&c);
            c.invalidate_overlapping(lo, hi, || {});
            assert_eq!(c.tier_pending_len(), 1);
            assert!(!c.try_begin_tier_up(k(0x1000)));
            assert_eq!(
                c.publish_tier_up(k(0x1000), compiled(), None, id, |_| {}),
                publish
            );
            assert_eq!(c.tier_pending_len(), 0);
            assert_eq!(
                c.memory_stats().tier_bg_rejected_generation,
                u64::from(!publish)
            );
        }
    }

    #[test]
    fn pending_uncached_successor_preserves_page_tracking() {
        let c = TranslationCache::new();
        let id = pending_region(&c);
        c.invalidate_overlapping(0x3800, 0x3801, || {
            panic!("pending successor needs its page tag")
        });
        c.invalidate_child_written(0x3000, 0x4000, || {});
        assert!(!c.publish_tier_up(k(0x1000), compiled(), None, id, |_| panic!("canceled")));
        assert_eq!(c.memory_stats().tier_bg_rejected_generation, 1);
    }

    #[test]
    fn old_completion_does_not_release_new_request() {
        let c = TranslationCache::new();
        let old = pending_region(&c);
        c.end_tier_up(k(0x1000));
        let new = pending_region(&c);
        assert_ne!(old, new);
        assert!(!c.publish_tier_up(k(0x1000), compiled(), None, old, |_| panic!("superseded")));
        assert_eq!(c.tier_pending_len(), 1);
        assert!(c.publish_tier_up(k(0x1000), compiled(), None, new, |_| {}));
        assert_eq!(c.memory_stats().tier_bg_superseded, 1);
        assert_eq!(c.tier_pending_len(), 0);
    }

    #[test]
    fn invalidation_before_slot_claim_prevents_registration_by_generation() {
        let c = TranslationCache::new();
        let generation = c.publication_generation();
        let source_spans = vec![(0x1000, 4), (0x3000, 4)];
        let epoch = c.epoch();

        // No pending slot exists for invalidation to cancel during the source read.
        c.invalidate_overlapping(0x3000, 0x3001, || {});
        assert_eq!(c.epoch(), epoch);
        assert_ne!(c.publication_generation(), generation);
        assert!(c.try_begin_tier_up(k(0x1000)));
        assert!(!c.spans.read().unwrap().pending[&k(0x1000)].cancelled);
        assert!(c
            .prepare_tier_up(k(0x1000), source_spans, generation, |_| {})
            .is_none());
        c.end_tier_up(k(0x1000));
        assert_eq!(c.tier_pending_len(), 0);
    }

    #[test]
    fn invalidation_during_source_read_prevents_registration() {
        let c = TranslationCache::new();
        let generation = c.publication_generation();
        assert!(c.try_begin_tier_up(k(0x1000)));
        c.invalidate_overlapping(0x3000, 0x3001, || {});
        assert!(c
            .prepare_tier_up(
                k(0x1000),
                vec![(0x1000, 4), (0x3000, 4)],
                generation,
                |_| panic!("stale source")
            )
            .is_none());
        c.end_tier_up(k(0x1000));
        assert_eq!(c.tier_pending_len(), 0);
    }
}
