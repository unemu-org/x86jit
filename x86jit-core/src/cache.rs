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

/// A raw pointer into the JIT code arena.
///
/// Manually `Send + Sync`: compiled code is read-only and executable from any
/// thread; its lifetime is owned by the JIT arena, which lives as long as the
/// `Vm` (§9.1). Declaring this in M4 (not M7) avoids reworking signatures once
/// threads arrive — the M7 trap.
#[derive(Copy, Clone)]
pub struct CompiledPtr(pub *const u8);

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

// One lock protects both span indices and pending sources. Lock order is
// spans -> map -> hotness, with atomic page-tag publication (§10, §17.4).
#[derive(Default)]
struct SpanIndex {
    units: HashMap<BlockKey, Vec<(u64, u32)>>,
    pages: HashMap<u64, HashSet<BlockKey>>,
    pending: HashMap<BlockKey, PendingTierUp>,
    next_request: u64,
}

struct PendingTierUp {
    id: u64,
    spans: Option<Vec<(u64, u32)>>,
    cancelled: bool,
}

fn span_pages(start: u64, len: u32) -> std::ops::RangeInclusive<u64> {
    (start >> crate::memory::CODE_PAGE_BITS)
        ..=(start.saturating_add(u64::from(len).saturating_sub(1)) >> crate::memory::CODE_PAGE_BITS)
}

impl SpanIndex {
    fn insert(&mut self, key: BlockKey, spans: Vec<(u64, u32)>) {
        self.remove(&key);
        for &(start, len) in &spans {
            if len != 0 {
                for page in span_pages(start, len) {
                    self.pages.entry(page).or_default().insert(key);
                }
            }
        }
        self.units.insert(key, spans);
    }

    fn remove(&mut self, key: &BlockKey) {
        if let Some(spans) = self.units.remove(key) {
            for (start, len) in spans {
                if len != 0 {
                    for page in span_pages(start, len) {
                        if let Some(keys) = self.pages.get_mut(&page) {
                            keys.remove(key);
                            if keys.is_empty() {
                                self.pages.remove(&page);
                            }
                        }
                    }
                }
            }
        }
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

    fn candidates(&self, lo: u64, hi: u64) -> HashSet<BlockKey> {
        let mut keys = HashSet::new();
        if lo >= hi {
            return keys;
        }
        let first = lo >> crate::memory::CODE_PAGE_BITS;
        let last = (hi - 1) >> crate::memory::CODE_PAGE_BITS;
        // Whole-cache/unmap requests can cover a sparse 64-bit address space.
        if last - first < self.pages.len() as u64 {
            for page in first..=last {
                if let Some(entries) = self.pages.get(&page) {
                    keys.extend(entries);
                }
            }
        } else {
            for (&page, entries) in &self.pages {
                if first <= page && page <= last {
                    keys.extend(entries);
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
    invalidation_calls: AtomicU64,
    invalidation_blocks_dropped: AtomicU64,
    tier_bg_submitted: AtomicU64,
    tier_bg_completed: AtomicU64,
    tier_bg_duplicates_skipped: AtomicU64,
    tier_bg_rejected_generation: AtomicU64,
    tier_bg_rejected_compile_failed: AtomicU64,
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
    epoch: AtomicU64,
    // Every invalidation cancels source reads still in flight, even if no cached
    // span overlaps. Keep this separate so data writes preserve dispatch predictors.
    publication_generation: AtomicU64,
    // Policy/configuration changes also advance the dispatch epoch without
    // victims; source publication is always canceled independently.
    pub(crate) invalidate_pending_fetches: bool,
    // IBTC descriptors (R4): immutable `[target, entry]` pairs published by pointer
    // into a per-site slot. Owned here (the dispatcher, in the core crate, allocates
    // them on a miss) and never freed before `Vm` drop, so a torn/late read of a
    // slot pointer can never dangle. The `Box` keeps each descriptor's address
    // stable across pushes. Only appended to; growth on a megamorphic site is capped
    // by the dispatcher, not here.
    #[allow(clippy::vec_box)]
    ibtc_descriptors: Mutex<Vec<Box<[u64; 2]>>>,
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
            invalidation_calls: self.invalidation_calls.load(Ordering::Relaxed),
            invalidation_blocks_dropped: self.invalidation_blocks_dropped.load(Ordering::Relaxed),
            tier_bg_submitted: self.tier_bg_submitted.load(Ordering::Relaxed),
            tier_bg_completed: self.tier_bg_completed.load(Ordering::Relaxed),
            tier_bg_duplicates_skipped: self.tier_bg_duplicates_skipped.load(Ordering::Relaxed),
            tier_bg_rejected_generation: self.tier_bg_rejected_generation.load(Ordering::Relaxed),
            tier_bg_rejected_compile_failed: self
                .tier_bg_rejected_compile_failed
                .load(Ordering::Relaxed),
            tier_bg_superseded: self.tier_bg_superseded.load(Ordering::Relaxed),
            tier_bg_published: self.tier_bg_published(),
            tier_bg_rejected: self.tier_bg_rejected(),
        }
    }

    pub fn new() -> Self {
        Self {
            memory_counts: std::array::from_fn(|_| AtomicU64::new(0)),
            invalidation_calls: AtomicU64::new(0),
            invalidation_blocks_dropped: AtomicU64::new(0),
            tier_bg_submitted: AtomicU64::new(0),
            tier_bg_completed: AtomicU64::new(0),
            tier_bg_duplicates_skipped: AtomicU64::new(0),
            tier_bg_rejected_generation: AtomicU64::new(0),
            tier_bg_rejected_compile_failed: AtomicU64::new(0),
            tier_bg_superseded: AtomicU64::new(0),

            map: RwLock::new(HashMap::new()),
            spans: RwLock::new(SpanIndex::default()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            chained: AtomicU64::new(0),
            regions: AtomicU64::new(0),
            epoch: AtomicU64::new(0),
            publication_generation: AtomicU64::new(0),
            invalidate_pending_fetches: false,
            ibtc_descriptors: Mutex::new(Vec::new()),
            ibtc_filled: AtomicU64::new(0),
            hotness: RwLock::new(HashMap::new()),
            region_decision: RwLock::new(HashMap::new()),
            tier_bg_published: AtomicU64::new(0),
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
    /// caller must re-lift rather than resurrect a stale translation.
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
            return false;
        }
        self.account_replace(
            map.get(&key),
            spans.units.get(&key).map_or(0, Vec::len),
            Some(&block),
            1,
        );
        spans.insert(key, vec![span]);
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
            return false;
        }
        on_mark(&spans);
        self.account_replace(
            mp.get(&key),
            sp.units.get(&key).map_or(0, Vec::len),
            Some(&block),
            spans.len(),
        );
        mp.insert(key, block);
        sp.insert(key, spans);
        self.hotness.write().unwrap().remove(&key);
        true
    }

    /// Allocate an immutable IBTC descriptor `{target, entry}` (R4) and return its
    /// stable heap address for the compiled code to load. The descriptor is never
    /// mutated (a new target gets a new descriptor) and never freed before `Vm`
    /// drop. The pointer publication into the slot must be a **`Release`** store (see
    /// the `RET_IBTC_MISS` site in `vm.rs`): the two field writes above happen-before
    /// the pointer becomes visible, pairing with the reader's address dependency
    /// (release/consume). A `Relaxed` publish would let a weakly-ordered host expose
    /// the pointer before the `{target, entry}` fields — a torn payload, not just a
    /// torn scalar.
    pub fn alloc_ibtc_descriptor(&self, target: u64, entry: CompiledPtr) -> u64 {
        let b = Box::new([target, entry.0 as u64]);
        let addr = &*b as *const [u64; 2] as u64;
        self.ibtc_descriptors.lock().unwrap().push(b);
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
    /// Cost is one uncontended write-lock pair per SLOT FILL, not per transfer: a direct
    /// edge fills once and is then followed with no dispatcher involvement at all, and
    /// IBTC refills stop at `IBTC_MEGAMORPHIC_CAP`.
    #[must_use]
    pub fn publish_if_current(&self, since_epoch: u64, publish: impl FnOnce()) -> bool {
        let _spans = self.spans.write().unwrap();
        let _map = self.map.write().unwrap();
        if self.epoch.load(Ordering::Acquire) != since_epoch {
            return false;
        }
        publish();
        true
    }

    /// Current invalidation epoch (R1). Monotonic; bumped whenever
    /// `invalidate_overlapping` drops at least one unit, or invalidates an in-flight
    /// fetch under an installed fetch policy. A vcpu snapshots this and flushes its
    /// local predictor caches when the value changes.
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
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
        let found = self.map.read().unwrap().get(&key).cloned();
        if found.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        found
    }

    /// Blocks served from the cache (no re-lift).
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Blocks that had to be lifted (distinct addresses, plus any re-lift after
    /// invalidation).
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

    /// Claim a background slot. Invalidation cancels its source snapshot but
    /// retains ownership until completion, preventing duplicate queued work.
    pub fn try_begin_tier_up(&self, key: BlockKey) -> bool {
        let mut spans = self.spans.write().unwrap();
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
                id,
                spans: None,
                cancelled: false,
            },
        );
        true
    }

    /// Release an unsubmitted request's slot, or explicitly abandon ownership.
    pub fn end_tier_up(&self, key: BlockKey) {
        self.spans.write().unwrap().pending.remove(&key);
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
        if pending.cancelled || self.publication_generation() != generation {
            return None;
        }
        on_mark(&source_spans);
        pending.spans = Some(source_spans);
        Some(pending.id)
    }

    pub(crate) fn publish_tier_up(
        &self,
        key: BlockKey,
        block: CachedBlock,
        request_id: u64,
        on_mark: impl FnOnce(&[(u64, u32)]),
    ) -> bool {
        let mut spans = self.spans.write().unwrap();
        if spans.pending.get(&key).is_none_or(|p| p.id != request_id) {
            self.record_tier_bg_rejected();
            self.tier_bg_superseded.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let pending = spans.pending.remove(&key).unwrap();
        if pending.cancelled {
            self.record_tier_bg_rejected_generation();
            return false;
        }
        if matches!(block, CachedBlock::Interpreted(_)) {
            self.record_tier_bg_compile_failed();
            return false;
        }
        let mut map = self.map.write().unwrap();
        if !matches!(map.get(&key), Some(CachedBlock::Interpreted(_))) {
            self.record_tier_bg_rejected();
            self.tier_bg_superseded.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let source_spans = pending.spans.expect("submitted tier-up has source spans");
        on_mark(&source_spans);
        self.account_replace(
            map.get(&key),
            spans.units.get(&key).map_or(0, Vec::len),
            Some(&block),
            source_spans.len(),
        );
        map.insert(key, block);
        spans.insert(key, source_spans);
        self.hotness.write().unwrap().remove(&key);
        self.record_tier_bg_published();
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
            return false;
        }
        self.account_replace(
            mp.get(&key),
            sp.units.get(&key).map_or(0, Vec::len),
            Some(&block),
            spans.len(),
        );
        mp.insert(key, block);
        sp.insert(key, spans);
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
        let mut spans = self.spans.write().unwrap();
        // SMC is address-scoped: a write to `[lo, hi)` drops every overlapping unit
        // regardless of the mode it was lifted under (§17.4 — the key carries the mode,
        // but invalidation matches by span address across all modes).
        let victims: Vec<BlockKey> = spans
            .candidates(lo, hi)
            .into_iter()
            .filter(|key| {
                spans.units[key]
                    .iter()
                    .any(|(start, len)| *start < hi && lo < start.saturating_add(u64::from(*len)))
            })
            .collect();
        let mut map = (!victims.is_empty()).then(|| self.map.write().unwrap());
        if let Some(map) = map.as_mut() {
            self.invalidation_blocks_dropped
                .fetch_add(victims.len() as u64, Ordering::Relaxed);
            let mut hotness = self.hotness.write().unwrap();
            for entry in &victims {
                self.account_replace(
                    map.get(entry),
                    spans.units.get(entry).map_or(0, Vec::len),
                    None,
                    0,
                );
                spans.remove(entry);
                map.remove(entry);
                hotness.remove(entry);
            }
        }
        // No extra lock or source scan when there is no in-flight request.
        let mut pending_on_page = false;
        if lo < hi && !spans.pending.is_empty() {
            for pending in spans.pending.values_mut() {
                if pending.cancelled {
                    continue;
                }
                match &pending.spans {
                    Some(source_spans) => {
                        pending.cancelled |= source_spans.iter().any(|&(start, len)| {
                            start < hi && lo < start.saturating_add(u64::from(len))
                        });
                        // Live uncached successors still need write notifications.
                        // Canceled requests must allow fetch approvals to be cleared.
                        pending_on_page |= !pending.cancelled
                            && source_spans.iter().any(|&(start, len)| {
                                len != 0
                                    && (start >> crate::memory::CODE_PAGE_BITS)
                                        <= ((hi - 1) >> crate::memory::CODE_PAGE_BITS)
                                    && (lo >> crate::memory::CODE_PAGE_BITS)
                                        <= (start.saturating_add(u64::from(len) - 1)
                                            >> crate::memory::CODE_PAGE_BITS)
                            });
                    }
                    None => pending.cancelled = true,
                }
            }
        }
        on_write(!pending_on_page && !spans.has_units(lo, hi));
        // A source span may have been read but not cached yet. Reject its
        // publication even when all currently cached spans survive this write.
        self.publication_generation.fetch_add(1, Ordering::Release);
        if CHILD || !victims.is_empty() || self.invalidate_pending_fetches {
            self.epoch.fetch_add(1, Ordering::Release);
        }
        victims
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
            entry: CompiledPtr(std::ptr::null()),
        }
    }

    /// A Long64 key at `addr` — the shape most cache tests key by.
    fn k(addr: u64) -> BlockKey {
        BlockKey::new(addr, CpuMode::Long64)
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
        assert!(!spans.pages.contains_key(&1));
        assert_eq!(spans.pages[&2].len(), 6);
        drop(spans);
        let victims = c.invalidate_overlapping(0x2107, 0x2108, || {});
        assert_eq!(victims.len(), 6);
        assert!(c.spans.read().unwrap().pages.is_empty());
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
        assert!(c.spans.read().unwrap().pages.is_empty());
        c.insert(key, compiled(), vec![(0x1000, 4), (0xffff_0000, 4)], |_| {});
        assert_eq!(c.invalidate_overlapping(0, u64::MAX, || {}), vec![key]);
        assert!(c.spans.read().unwrap().pages.is_empty());
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
                c.publish_tier_up(k(0x1000), compiled(), id, |_| {}),
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
        assert!(!c.publish_tier_up(k(0x1000), compiled(), id, |_| panic!("canceled")));
        assert_eq!(c.memory_stats().tier_bg_rejected_generation, 1);
    }

    #[test]
    fn old_completion_does_not_release_new_request() {
        let c = TranslationCache::new();
        let old = pending_region(&c);
        c.end_tier_up(k(0x1000));
        let new = pending_region(&c);
        assert_ne!(old, new);
        assert!(!c.publish_tier_up(k(0x1000), compiled(), old, |_| panic!("superseded")));
        assert_eq!(c.tier_pending_len(), 1);
        assert!(c.publish_tier_up(k(0x1000), compiled(), new, |_| {}));
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
