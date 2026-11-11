//! Allocation diagnostics for embedders (§4.1, §9.1).

/// Backend-owned storage. Counters are independent relaxed snapshots, not a
/// transaction. Bytes exclude allocator bookkeeping and physical residency.
/// A backend that cannot account for storage returns the default (all zero).
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct BackendMemoryStats {
    /// Finalized units whose executable storage is retained. The current JIT
    /// never reclaims code, so invalidation does not decrease this count.
    pub compiled_units_live: u64,
    /// Units successfully finalized over the backend's lifetime, including
    /// completions rejected by cache publication and fault-map registration.
    pub compiled_units_total: u64,
    /// Sum of finalized unit lengths, including constant pools, excluding padding.
    pub code_bytes_emitted: u64,
    /// Logical code-region capacity, including padding and unused tails.
    /// Exact provider accounting, including pending/failed definitions. On macOS
    /// these are slices of code_process_reservation_bytes. Never an RSS estimate.
    pub code_bytes_reserved: u64,
    /// Logical code regions owned by this backend (128 MiB or larger).
    pub code_regions: u64,
    /// Distinct OS views covering those regions: two aliases per Unix region,
    /// one shared process-wide MAP_JIT view on macOS. Not physical allocations.
    pub code_mapping_views: u64,
    /// Process-wide macOS MAP_JIT virtual reservation (shared across backends;
    /// do not sum across VMs). Zero on other systems.
    pub code_process_reservation_bytes: u64,
    /// Page-rounded high-water extents of allocated code, counted once per
    /// backing object. Demand-paged allocator commitment, not physical residency.
    pub code_bytes_committed: u64,
    /// Code allocation extents, including alignment padding and veneer capacity.
    pub code_bytes_used: u64,
    /// Unused region tails. No retired extents are reusable before phase 3.
    pub code_bytes_free: u64,
    /// Largest currently unused contiguous region tail.
    pub code_largest_free_extent: u64,
    /// Provider allocation/finalization failures, including OS reservation errors.
    pub code_allocation_failures: u64,
    /// Allocation-handle and region-vector capacities, excluding allocator overhead.
    pub code_allocation_metadata_bytes: u64,
    /// Requests waiting in the background queue, excluding the active batch.
    pub background_queue_len: u64,
    /// Requests removed from the queue whose batch has not completed publication.
    pub background_in_flight: u64,
}

/// Current translation-cache publications and lifetime activity counters.
/// Independently sampled relaxed counters;
/// concurrent replacement can temporarily expose parts of two snapshots.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct CacheMemoryStats {
    /// Current interpreted entries (one per decode/FP-mode cache key).
    pub interpreted_entries: u64,
    /// Current compiled single-block entries.
    pub compiled_entries: u64,
    /// Current compiled entries with multiple source spans (regions).
    pub region_entries: u64,
    /// Approximate IR held by interpreted cache entries: IrBlock + Arc counters
    /// plus `ops.capacity() * size_of::<IrOp>()`. Counts shared IR per cache key;
    /// excludes queued IR, executing clones, and nested operation allocations.
    pub interpreted_ir_bytes: u64,
    /// Total source-index allocation estimate: cached and pending membership tables,
    /// source vectors, byte bitmaps, exact overlap counters, radix nodes, pending
    /// request and suppression tables. Includes table empty buckets/control bytes
    /// and retained capacity; excludes allocator metadata and transient victims.
    pub coverage_index_bytes: u64,
    /// Calls to cache invalidation, including victim-free writes and explicit flushes.
    pub invalidation_calls: u64,
    /// Cache entries removed by invalidation (regions count once).
    pub invalidation_blocks_dropped: u64,
    /// Background requests accepted by the backend (excludes Busy/Unsupported).
    pub tier_bg_submitted: u64,
    /// Submission attempts skipped because the key already has a pending request.
    pub tier_bg_duplicates_skipped: u64,
    /// Background completions received by the dispatcher, including compile failures.
    pub tier_bg_completed: u64,
    /// Background completions installed into the cache.
    pub tier_bg_published: u64,
    /// Background completions discarded; sum of the rejection reasons below.
    pub tier_bg_rejected: u64,
    /// Completions rejected because their source publication snapshot was invalidated.
    pub tier_bg_rejected_generation: u64,
    /// Completions returning interpreted IR because compilation/publication in the backend failed.
    pub tier_bg_rejected_compile_failed: u64,
    /// Compile-failed completions exceeding the guest-access descriptor capacity.
    pub tier_bg_failed_descriptor_limit: u64,
    /// Compile-failed completions due to code space, storage budget or allocation.
    pub tier_bg_failed_code_space: u64,
    /// Other backend failures, including failures with no classification.
    /// These three reason counters sum to `tier_bg_rejected_compile_failed`.
    pub tier_bg_failed_other_backend: u64,
    /// Units marked interpreter-only after a deterministic background failure.
    /// Lifetime count; invalidation permits the changed unit to tier up again.
    pub tier_bg_interpreter_only: u64,
    /// Tier-up attempts suppressed for interpreter-only units, including after
    /// switching from background to inline tier-up.
    pub tier_bg_interpreter_only_skipped: u64,
    /// Completions discarded because their request no longer owns the pending slot or a compiled entry superseded it.
    pub tier_bg_superseded: u64,
}

/// Estimated storage summed over live vCPUs, excluding allocator overhead.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct VcpuMemoryStats {
    /// Number of live vCPUs created by this VM.
    pub count: u64,
    /// Inline Vcpu plus return stack, fault record, dispatch word and accounting
    /// record allocations (including their Arc headers).
    /// Excludes CPU configuration/override trees, evidence payloads and embedder storage.
    pub state_bytes: u64,
    /// Heap bytes in the fixed fast-resolve arrays.
    pub fast_resolve_bytes: u64,
    /// Allocated interpreter temporary-buffer capacity in bytes.
    pub scratch_bytes: u64,
    /// IBTC refill map capacity times key/value sizes; excludes hash control
    /// bytes and spare buckets. Updated when a refill entry is created.
    pub refill_bytes: u64,
}

/// Cheap diagnostic snapshot; categories are not a complete accounting of RSS.
/// Standalone ChildArena storage belongs to the embedder and is excluded;
/// vCPUs own no child arenas. Compiler contexts, cache hash tables/other span indices,
/// link slots and allocator bookkeeping are also excluded.
/// No compiler lock or cache traversal is required. The provider registry, vCPU
/// registry, background queue and code-map writer locks are acquired briefly.
/// Code-map sampling also collects obsolete versions after reader quiescence.
/// Fields can reflect different instants.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct MemoryStats {
    /// Backend counters; the interpreter returns zero.
    pub backend: BackendMemoryStats,
    /// Current cache publications, distinct from retained backend allocations.
    pub cache: CacheMemoryStats,
    /// Per-vCPU state summed over live registered vCPUs.
    pub vcpus: VcpuMemoryStats,
    /// Process-wide retained code-map/fault metadata, including live/deferred AVL
    /// nodes, dense tables, shared recipes and instruction records. Includes other
    /// VMs and dropped backends. Decreases on reclamation/removal, but invalidation
    /// alone does not retire executable ranges. Excludes allocator overhead.
    pub codemap_bytes: u64,
    /// Guest backing extent, including any spare embedder-supplied backing bytes.
    pub guest_ram_reserved: u64,
    /// Owned RAM allocation bytes, or None for embedder-provided backing whose
    /// commitment is unknown. Allocation commitment is not resident memory.
    pub guest_ram_committed: Option<u64>,
    /// Allocated capacities of code/watch/dirty bitmaps. Lazily resident and
    /// separate from RAM; excludes sparse index and alias-tree allocations.
    pub memory_tracking_bytes: u64,
}
