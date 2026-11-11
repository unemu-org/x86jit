//! Per-VM execution diagnostics (§2, §9.2).
//!
//! Dispatch and exit counts are always enabled. Helper accounting and detailed
//! timing are opt-in with [`crate::Vm::set_time_accounting`]. The clock is CNTVCT_EL0 on AArch64,
//! converted using CNTFRQ_EL0, and `std::time::Instant` elsewhere. Durations
//! measure elapsed host time, including descheduling, not thread CPU time.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Calls with deterministic 1-in-64 timing (first call per run, then every 64th).
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct SampledTimeStats {
    /// Exact number of calls in the category's accounting scope. Helper calls
    /// are counted only while detailed accounting is enabled; mixed
    /// `fetch_translate` calls always. `fetch_hits` calls require accounting.
    pub calls: u64,
    /// Exact number of timed calls; zero when detailed timing was never enabled.
    pub samples: u64,
    /// Nanoseconds in timed calls only, measured at call entry/exit. Estimate
    /// total time as `sample_ns * calls / samples` only for a representative
    /// interval with timing enabled throughout. Sampling can alias periodic work.
    pub sample_ns: u64,
}

/// One category of returns to the embedder.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct ExitTimeStats {
    /// Exact return count, including returns while detailed timing is disabled.
    pub count: u64,
    /// Nanoseconds from return to the next accepted run entry on that vCPU,
    /// measured at transitions with detailed timing enabled at both boundaries.
    /// The current outstanding interval is excluded; snapshot reads do not
    /// advance it. Includes embedder work, scheduling and idle time.
    pub outside_ns: u64,
}

/// Returns grouped by their public exit reason.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct ExitsTimeStats {
    /// Syscall returns: exact count and transition-timed outside nanoseconds.
    pub syscall: ExitTimeStats,
    /// Unmapped-memory and exception returns: exact count and outside nanoseconds.
    pub fault: ExitTimeStats,
    /// Host exit requests: exact count and transition-timed outside nanoseconds.
    pub interrupted: ExitTimeStats,
    /// Port-I/O returns: exact count and transition-timed outside nanoseconds.
    pub port_io: ExitTimeStats,
    /// MMIO reads/writes: exact count and transition-timed outside nanoseconds.
    pub mmio: ExitTimeStats,
    /// HLT returns: exact count and transition-timed outside nanoseconds.
    pub hlt: ExitTimeStats,
    /// Budget/IRQ-ready stops: exact count and transition-timed outside nanoseconds.
    pub budget: ExitTimeStats,
    /// Other returns: exact count and transition-timed outside nanoseconds.
    pub other: ExitTimeStats,
}

/// Cumulative, saturating, monotonic per-VM diagnostics. No compiler lock is taken.
/// A short vCPU-registry lock sums independent relaxed snapshots. vCPU totals
/// survive vCPU destruction. A running vCPU publishes at run exit, so snapshots
/// can lag by an entire run. Fields are not transactional. Only `run` and
/// `run_steps` are tracked; standalone `step_one` and child execution are excluded.
/// Backend worker totals are independently sampled and can lag an active batch.
/// Forked VMs start fresh. Timing fields remain zero until explicitly enabled.
///
/// Interpreter and compiled segments are timed only at tier switches and run
/// boundaries, never at individual instruction or chained-block boundaries.
/// Dispatch/lifting/bookkeeping is charged to the most recently selected tier
/// (initially interpreter). Foreground materialization is excluded from those
/// segments. Helper and fetch/translate measurements are overlapping subsets,
/// not additive categories; fetch/translate samples exclude nested compilation.
/// The split fetch fields require detailed accounting. Miss callback, lift and
/// other nanoseconds partition every miss path from its first failed probe
/// (or resolve entry when an entry sample timestamp is available);
/// their sum requires no sampling extrapolation. Whole-resolve callback/lift
/// totals also include work during hit-path tier-up and overlap those buckets.
/// `fetch_from_*` and `fetch_private_*` are two independent partitions of enabled
/// resolves: each group sums to `fetch_hits.calls + fetch_misses`. Retries count
/// once. Private miss attribution describes slots, not previously seen RIPs;
/// indirect jumps/calls share an ABI reason, as do block/region boundaries.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct TimeStats {
    /// Nanoseconds in interpreter segments, measured at transitions when enabled.
    pub interpreter_ns: u64,
    /// Nanoseconds in compiled segments, including helpers, measured at transitions.
    pub compiled_ns: u64,
    /// Nanoseconds in foreground backend materialization calls, timed at call
    /// boundaries when enabled, including compiler-lock wait and failed calls.
    /// Backends instrument their materialization boundaries; the interpreter
    /// backend returns zero. Standalone calls outside a vCPU run are excluded.
    pub foreground_compile_ns: u64,
    /// Nanoseconds of background worker active batches, timed at batch boundaries
    /// when enabled. Includes definition, finalization, publication preparation,
    /// observers and lock wait; excludes waiting for queued work.
    pub background_compile_ns: u64,
    /// Compiled Rust helper/slow-replay calls while detailed accounting is enabled:
    /// exact counts and 1-in-64 timing. Disabled calls leave all helper fields unchanged.
    /// Inline memory, arithmetic, FP-control switches and generated fast paths
    /// are excluded. Nested helper calls count only the outer call.
    pub helpers: SampledTimeStats,
    /// Dispatcher fetch/translate resolution calls: exact counts, 1-in-64 timing.
    /// Includes cache access, lifting and publication; excludes compiler time.
    /// Fast-resolve hits and chained transfers do not enter this call path.
    pub fetch_translate: SampledTimeStats,
    /// Resolve calls with no shared-cache miss, counted exactly while accounting
    /// is enabled, with independent 1-in-64 timing (first hit per run, then every
    /// 64th hit). Nanoseconds cover the whole resolve, including tier-up work,
    /// excluding nested foreground compilation. Private fast-cache hits and
    /// chained transfers are excluded, as in `fetch_translate`.
    pub fetch_hits: SampledTimeStats,
    /// Exact resolve calls encountering at least one shared-cache miss while
    /// accounting is enabled. Retries and failed/unwound calls count once;
    /// `fetch_hits.calls + fetch_misses` counts all enabled resolves.
    pub fetch_misses: u64,
    /// Shared resolves at run entry (except the budget/request resumptions below).
    /// Counted only with time accounting enabled.
    pub fetch_from_entry: u64,
    /// Shared resolves at the first dispatch after an accounted budget return.
    /// Counted only with time accounting enabled.
    pub fetch_from_budget_resume: u64,
    /// Shared resolves at the first dispatch after an accounted host exit request.
    /// Counted only with time accounting enabled.
    pub fetch_from_exit_request_resume: u64,
    /// Shared resolves after an unchained compiled block/region boundary (RET_CONTINUE or RET_INTERRUPTED).
    /// Counted only with time accounting enabled.
    pub fetch_from_boundary: u64,
    /// Shared resolves after a compiled return prediction mismatch or underflow.
    /// Counted only with time accounting enabled.
    pub fetch_from_return_miss: u64,
    /// Shared resolves for an unfilled direct or return-continuation link slot (RET_LINK).
    /// Counted only with time accounting enabled.
    pub fetch_from_direct_link: u64,
    /// Shared resolves after a compiled indirect jump/call IBTC miss.
    /// Counted only with time accounting enabled.
    pub fetch_from_ibtc_miss: u64,
    /// Shared resolves after interpreted execution.
    /// Counted only with time accounting enabled.
    pub fetch_from_interpreter: u64,
    /// Shared resolves whose slot was empty since construction or explicit reset.
    /// Counted only with time accounting enabled.
    pub fetch_private_cold: u64,
    /// Shared resolves whose private-cache slot held a different RIP.
    /// Counted only with time accounting enabled.
    pub fetch_private_collision: u64,
    /// Shared resolves whose slot was still empty after an invalidation-epoch flush.
    /// Counted only with time accounting enabled.
    pub fetch_private_epoch: u64,
    /// Shared resolves whose slot was still empty after an MXCSR-mode flush.
    /// Counted only with time accounting enabled.
    pub fetch_private_mode: u64,
    /// Shared resolves without a private probe (unfilled link slots).
    /// Counted only with time accounting enabled.
    pub fetch_private_bypass: u64,
    /// Nanoseconds inside embedder fetch-policy callbacks during enabled resolves,
    /// timed on every outermost callback, including denial, retries and unwinding.
    /// Nested callbacks are included once in their outer callback. Includes
    /// callbacks during hit-path tier-up; excludes approval-cache hits, policy
    /// locks/waits and callbacks outside resolve. Overlaps sampled hit time.
    pub fetch_approval_ns: u64,
    /// Nanoseconds decoding/lifting blocks and forming regions during enabled
    /// resolves, timed on every lift, including failed lifts and hit-path tier-up.
    /// Includes instruction/region reads; excludes fetch-policy callbacks and
    /// approval bookkeeping (including locks/waits), compilation and publication.
    /// Overlaps sampled hit time.
    pub fetch_lift_ns: u64,
    /// Nanoseconds inside fetch-policy callbacks in the miss timing window
    /// described above, timed on every outermost callback. The miss-path subset
    /// of `fetch_approval_ns`; excludes approval-cache hits and policy locks/waits.
    pub fetch_miss_approval_ns: u64,
    /// Nanoseconds decoding/lifting and forming regions in the miss timing window
    /// described above, timed on every lift. The miss-path subset
    /// of `fetch_lift_ns`; excludes callbacks, approval bookkeeping (including
    /// locks/waits), compilation and publication.
    pub fetch_miss_lift_ns: u64,
    /// Nanoseconds on every enabled miss path from the first failed shared-cache
    /// probe (or resolve entry when an entry sample timestamp is available) to
    /// resolve return/unwind, excluding callback, lift and foreground compile time.
    /// Includes fetch permits, approval bookkeeping and policy locks/waits,
    /// publication, tier-up submission and retries; excludes background
    /// compilation. Never extrapolated from samples.
    pub fetch_miss_other_ns: u64,
    /// Embedder returns: exact counts and transition-timed outside nanoseconds.
    pub exits: ExitsTimeStats,
    /// Exact shared block-cache probes plus successful private fast-cache probes.
    /// Excludes chain hits and internal compiled-region edges.
    pub block_lookups: u64,
    /// Exact shared block-cache misses (private misses fall through to this cache).
    pub block_misses: u64,
    /// Exact compiled IBTC misses and return-prediction mismatches/underflows
    /// sent to the dispatcher. Filled indirect chains and cold link-slot resolves
    /// (including correctly predicted returns) are excluded.
    pub indirect_branch_resolves: u64,
    /// Exact RET_CHAIN transfers (filled direct links, IBTC and return predictions).
    /// Internal edges within one compiled region are excluded.
    pub chain_hits: u64,
    /// Nanoseconds from accepted tier-up submission to successful cache publication,
    /// summed at publication when timing was enabled at source registration,
    /// immediately before backend admission. Includes submission overhead, worker
    /// queue wait and delay before dispatcher drain; rejected jobs are excluded.
    pub tier_up_latency_ns: u64,
    /// Exact number of successful publications contributing to tier_up_latency_ns.
    /// Zero for submissions made while detailed timing was disabled.
    pub tier_up_latency_count: u64,
    /// Shared resolves after an internal MXCSR specialization restart, rather
    /// than a run entry/resume. Counted only with detailed accounting enabled.
    pub fetch_from_mode_restart: u64,
    /// Shard read acquisitions that failed their first `try_read`, counted exactly
    /// only with detailed accounting. First contended acquisition per run and
    /// every 64th thereafter time the blocking read; `sample_ns` includes elapsed
    /// wait and descheduling. Uncontended reads contribute to none of these fields.
    pub fetch_shard_read_contention: SampledTimeStats,
    /// Timed hit samples exceeding [`SLOW_HIT_THRESHOLD_NS`]. Subset of
    /// `fetch_hits.samples`; zero without detailed accounting. Subtract these
    /// samples and `fetch_hit_slow_ns` to compute the remaining sample mean.
    pub fetch_hit_slow_samples: u64,
    /// Total nanoseconds of the slow samples counted by `fetch_hit_slow_samples`.
    /// Subset of `fetch_hits.sample_ns`, with the same compilation exclusion.
    pub fetch_hit_slow_ns: u64,
    /// Nanoseconds draining/publishing completed background compiles within
    /// resolves that finish as hits, timed on every enabled drain (including
    /// empty drains). Includes backend completion polling and publication waits.
    /// Excluded from no other field: sampled hit windows include this work,
    /// whereas this total includes unsampled hits too. Never extrapolate it.
    pub fetch_hit_publication_ns: u64,
}

/// Backend worker timing, sampled without compiler locks. Wrappers should forward
/// [`crate::Backend::time_stats`] and [`crate::Backend::set_time_accounting`].
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct BackendTimeStats {
    /// Nanoseconds in completed active worker batches, timed at batch boundaries.
    pub background_compile_ns: u64,
}

#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct MonotonicTime(u64);
#[cfg(not(target_arch = "aarch64"))]
#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct MonotonicTime(std::time::Instant);

impl MonotonicTime {
    #[inline]
    /// Read the host monotonic counter.
    pub fn now() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            let count;
            // Arm ARM DDI0601, AArch64 system registers CNTVCT_EL0/CNTFRQ_EL0.
            // SAFETY: these read-only timer registers are available to user mode
            // on supported AArch64 hosts; ISB orders the counter read.
            unsafe {
                core::arch::asm!("isb", "mrs {count}, cntvct_el0", count = out(reg) count,
                    options(nostack, preserves_flags));
            }
            Self(count)
        }
        #[cfg(not(target_arch = "aarch64"))]
        Self(std::time::Instant::now())
    }

    /// Elapsed nanoseconds, truncated to the counter resolution.
    pub fn elapsed_ns(self) -> u64 {
        #[cfg(target_arch = "aarch64")]
        {
            let elapsed = Self::now().0.saturating_sub(self.0);
            static FREQUENCY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
            let frequency = *FREQUENCY.get_or_init(|| {
                let frequency;
                // SAFETY: read-only user-mode timer frequency; see now().
                unsafe {
                    core::arch::asm!("mrs {frequency}, cntfrq_el0", frequency = out(reg) frequency,
                        options(nomem, nostack, preserves_flags));
                }
                frequency
            });
            ((elapsed as u128 * 1_000_000_000) / u128::from(frequency)).min(u64::MAX as u128) as u64
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.0.elapsed().as_nanos().min(u64::MAX as u128) as u64
        }
    }
}

const INTERPRETER: usize = 0;
const COMPILED: usize = 1;
const COMPILER: usize = 2;
const HELPER: usize = 3;
const FETCH: usize = 6;
const LOOKUPS: usize = 9;
const MISSES: usize = 10;
const INDIRECT: usize = 11;
const CHAINS: usize = 12;
const EXITS: usize = 13;
const CATEGORIES: usize = 8;
const FETCH_HITS: usize = EXITS + 2 * CATEGORIES;
const FETCH_MISSES: usize = FETCH_HITS + 3;
const FETCH_APPROVAL: usize = FETCH_MISSES + 1;
const FETCH_LIFT: usize = FETCH_APPROVAL + 1;
const FETCH_MISS_APPROVAL: usize = FETCH_LIFT + 1;
const FETCH_MISS_LIFT: usize = FETCH_MISS_APPROVAL + 1;
const FETCH_MISS_OTHER: usize = FETCH_MISS_LIFT + 1;
const ENTRY: usize = FETCH_MISS_OTHER + 1;
const BUDGET_RESUME: usize = ENTRY + 1;
const REQUEST_RESUME: usize = BUDGET_RESUME + 1;
const BOUNDARY: usize = REQUEST_RESUME + 1;
const RETURN_MISS: usize = BOUNDARY + 1;
const DIRECT_LINK: usize = RETURN_MISS + 1;
const IBTC_MISS: usize = DIRECT_LINK + 1;
const INTERPRETER_EDGE: usize = IBTC_MISS + 1;
const PRIVATE_COLD: usize = INTERPRETER_EDGE + 1;
const PRIVATE_COLLISION: usize = PRIVATE_COLD + 1;
const PRIVATE_EPOCH: usize = PRIVATE_COLLISION + 1;
const PRIVATE_MODE: usize = PRIVATE_EPOCH + 1;
const PRIVATE_BYPASS: usize = PRIVATE_MODE + 1;
const MODE_RESTART: usize = PRIVATE_BYPASS + 1;
const SHARD_CONTENTION: usize = MODE_RESTART + 1;
const HIT_SLOW_SAMPLES: usize = SHARD_CONTENTION + 3;
const HIT_SLOW_NS: usize = HIT_SLOW_SAMPLES + 1;
const HIT_PUBLICATION: usize = HIT_SLOW_NS + 1;
const FIELDS: usize = HIT_PUBLICATION + 1;

/// Elapsed hit samples above 20 microseconds are also recorded as slow samples.
/// This fixed threshold separates long scheduling/lock/publication delays for
/// consumers computing a trimmed mean; it does not identify their cause.
pub const SLOW_HIT_THRESHOLD_NS: u64 = 20_000;

pub(crate) struct Record([AtomicU64; FIELDS]);
impl Default for Record {
    fn default() -> Self {
        Self(std::array::from_fn(|_| AtomicU64::new(0)))
    }
}

pub(crate) struct Registry {
    records: Vec<Arc<Record>>,
    retired: [u64; FIELDS],
}
impl Default for Registry {
    fn default() -> Self {
        Self {
            records: Vec::new(),
            retired: [0; FIELDS],
        }
    }
}
impl Registry {
    pub(crate) fn register(&mut self) -> Arc<Record> {
        // Reclaim retired records even when the embedder never reads snapshots.
        let _ = self.snapshot();
        let record = Arc::new(Record::default());
        self.records.push(record.clone());
        record
    }
    pub(crate) fn snapshot(&mut self) -> TimeStats {
        let mut sum = self.retired;
        self.records.retain(|r| {
            let retired = Arc::strong_count(r) == 1;
            for (i, counter) in r.0.iter().enumerate() {
                let value = counter.load(Ordering::Relaxed);
                sum[i] = sum[i].saturating_add(value);
                if retired {
                    self.retired[i] = self.retired[i].saturating_add(value);
                }
            }
            !retired
        });
        let sampled = |i| SampledTimeStats {
            calls: sum[i],
            samples: sum[i + 1],
            sample_ns: sum[i + 2],
        };
        let exit = |i: usize| ExitTimeStats {
            count: sum[EXITS + i * 2],
            outside_ns: sum[EXITS + i * 2 + 1],
        };
        TimeStats {
            interpreter_ns: sum[INTERPRETER],
            compiled_ns: sum[COMPILED],
            foreground_compile_ns: sum[COMPILER],
            helpers: sampled(HELPER),
            fetch_translate: sampled(FETCH),
            fetch_hits: sampled(FETCH_HITS),
            fetch_misses: sum[FETCH_MISSES],
            fetch_from_mode_restart: sum[MODE_RESTART],
            fetch_shard_read_contention: sampled(SHARD_CONTENTION),
            fetch_hit_slow_samples: sum[HIT_SLOW_SAMPLES],
            fetch_hit_slow_ns: sum[HIT_SLOW_NS],
            fetch_hit_publication_ns: sum[HIT_PUBLICATION],
            fetch_from_entry: sum[ENTRY],
            fetch_from_budget_resume: sum[BUDGET_RESUME],
            fetch_from_exit_request_resume: sum[REQUEST_RESUME],
            fetch_from_boundary: sum[BOUNDARY],
            fetch_from_return_miss: sum[RETURN_MISS],
            fetch_from_direct_link: sum[DIRECT_LINK],
            fetch_from_ibtc_miss: sum[IBTC_MISS],
            fetch_from_interpreter: sum[INTERPRETER_EDGE],
            fetch_private_cold: sum[PRIVATE_COLD],
            fetch_private_collision: sum[PRIVATE_COLLISION],
            fetch_private_epoch: sum[PRIVATE_EPOCH],
            fetch_private_mode: sum[PRIVATE_MODE],
            fetch_private_bypass: sum[PRIVATE_BYPASS],
            fetch_approval_ns: sum[FETCH_APPROVAL],
            fetch_lift_ns: sum[FETCH_LIFT],
            fetch_miss_approval_ns: sum[FETCH_MISS_APPROVAL],
            fetch_miss_lift_ns: sum[FETCH_MISS_LIFT],
            fetch_miss_other_ns: sum[FETCH_MISS_OTHER],
            block_lookups: sum[LOOKUPS],
            block_misses: sum[MISSES],
            indirect_branch_resolves: sum[INDIRECT],
            chain_hits: sum[CHAINS],
            exits: ExitsTimeStats {
                syscall: exit(0),
                fault: exit(1),
                interrupted: exit(2),
                port_io: exit(3),
                mmio: exit(4),
                hlt: exit(5),
                budget: exit(6),
                other: exit(7),
            },
            ..TimeStats::default()
        }
    }
}

// Disabled runs initialize and publish only the original always-on counters.
// Detailed storage is allocated/reset/scanned solely for enabled runs.
struct DetailedCounters {
    values: [u64; FIELDS - FETCH_HITS],
    publication_ns: u64,
}
struct Counters {
    basic: [u64; FETCH_HITS],
    detailed: Option<Box<DetailedCounters>>,
}
impl std::ops::Index<usize> for Counters {
    type Output = u64;
    fn index(&self, index: usize) -> &u64 {
        if index < FETCH_HITS {
            &self.basic[index]
        } else {
            self.detailed
                .as_ref()
                .map_or(&0, |v| &v.values[index - FETCH_HITS])
        }
    }
}
impl std::ops::IndexMut<usize> for Counters {
    fn index_mut(&mut self, index: usize) -> &mut u64 {
        if index < FETCH_HITS {
            &mut self.basic[index]
        } else {
            &mut self.detailed.as_mut().unwrap().values[index - FETCH_HITS]
        }
    }
}
struct Local {
    values: Counters,
    resume: SharedReason,
    active: bool,
    enabled: bool,
    segment: usize,
    since: Option<MonotonicTime>,
    helper_depth: u32,
    approval_depth: u32,
    policy_ns: u64,
    resolving: bool,
    hit_start: Option<MissTiming>,
    miss: Option<MissTiming>,
}
impl Local {
    const EMPTY: Self = Self {
        values: Counters {
            basic: [0; FETCH_HITS],
            detailed: None,
        },
        resume: SharedReason::Entry,
        active: false,
        enabled: false,
        segment: INTERPRETER,
        since: None,
        helper_depth: 0,
        approval_depth: 0,
        policy_ns: 0,
        resolving: false,
        hit_start: None,
        miss: None,
    };
}
impl Default for Local {
    fn default() -> Self {
        Self::EMPTY
    }
}
thread_local! { static LOCAL: RefCell<Local> = const { RefCell::new(Local::EMPTY) }; }

fn add(values: &mut Counters, i: usize, value: u64) {
    values[i] = values[i].saturating_add(value);
}
fn switch(local: &mut Local, segment: usize) {
    if local.enabled && local.segment != segment {
        if let Some(start) = local.since {
            add(&mut local.values, local.segment, start.elapsed_ns());
        }
        local.since = Some(MonotonicTime::now());
        local.segment = segment;
    }
}
pub(crate) fn interpreted() {
    LOCAL.with(|l| switch(&mut l.borrow_mut(), INTERPRETER));
}
pub(crate) fn compiled() {
    LOCAL.with(|l| switch(&mut l.borrow_mut(), COMPILED));
}
pub(crate) fn enabled() -> bool {
    LOCAL.with(|l| l.borrow().enabled)
}
/// Orthogonal partitions of enabled shared resolves: source edge and private probe.
#[derive(Clone, Copy)]
pub(crate) enum SharedReason {
    Entry = ENTRY as isize,
    BudgetResume = BUDGET_RESUME as isize,
    RequestResume = REQUEST_RESUME as isize,
    Boundary = BOUNDARY as isize,
    ReturnMiss = RETURN_MISS as isize,
    DirectLink = DIRECT_LINK as isize,
    IbtcMiss = IBTC_MISS as isize,
    Interpreter = INTERPRETER_EDGE as isize,
    ModeRestart = MODE_RESTART as isize,
}
#[derive(Clone, Copy)]
pub(crate) enum PrivateMiss {
    Cold = PRIVATE_COLD as isize,
    Collision = PRIVATE_COLLISION as isize,
    Epoch = PRIVATE_EPOCH as isize,
    Mode = PRIVATE_MODE as isize,
    Bypass = PRIVATE_BYPASS as isize,
}
/// Call only behind the VM accounting flag. No disabled-path TLS or counters.
pub(crate) fn shared_resolve(reason: SharedReason, private: PrivateMiss) {
    LOCAL.with(|local| {
        let mut l = local.borrow_mut();
        let reason = if matches!(reason, SharedReason::Entry) {
            l.resume
        } else {
            reason
        };
        add(&mut l.values, reason as usize, 1);
        add(&mut l.values, private as usize, 1);
    });
}

#[derive(Default)]
pub(crate) struct DispatchCounts {
    pub lookups: u64,
    pub misses: u64,
    pub fetch_calls: u64,
    pub indirect: u64,
}

pub(crate) type Outside = Option<(usize, MonotonicTime)>;
pub(crate) struct RunGuard {
    previous: Local,
    record: Arc<Record>,
}
impl RunGuard {
    pub(crate) fn enter(record: Arc<Record>, enabled: bool, outside: Outside) -> Self {
        let mut current = Local {
            active: true,
            enabled,
            since: enabled.then(MonotonicTime::now),
            ..Local::default()
        };
        if enabled {
            current.resume = match outside.map(|(kind, _)| kind) {
                Some(6) => SharedReason::BudgetResume,
                Some(2) => SharedReason::RequestResume,
                _ => SharedReason::Entry,
            };
            current.values.detailed = Some(Box::new(DetailedCounters {
                values: [0; FIELDS - FETCH_HITS],
                publication_ns: 0,
            }));
            if let Some((kind, start)) = outside {
                add(
                    &mut current.values,
                    EXITS + kind * 2 + 1,
                    start.elapsed_ns(),
                );
            }
        }
        let previous = LOCAL.with(|l| l.replace(current));
        Self { previous, record }
    }
    pub(crate) fn finish(&self, exit: Option<&crate::Exit>) -> Outside {
        let kind = match exit {
            Some(crate::Exit::Syscall) => 0,
            Some(crate::Exit::UnmappedMemory { .. } | crate::Exit::Exception { .. }) => 1,
            Some(crate::Exit::Interrupted) => 2,
            Some(crate::Exit::PortIo { .. }) => 3,
            Some(crate::Exit::MmioRead { .. } | crate::Exit::MmioWrite { .. }) => 4,
            Some(crate::Exit::Hlt) => 5,
            Some(crate::Exit::BudgetExhausted) | None => 6,
            _ => 7,
        };
        LOCAL.with(|l| {
            let mut l = l.borrow_mut();
            add(&mut l.values, EXITS + kind * 2, 1);
            l.enabled.then(|| (kind, MonotonicTime::now()))
        })
    }
}
pub(crate) fn dispatch_counts(chains: u64, counts: DispatchCounts) {
    LOCAL.with(|l| {
        let mut l = l.borrow_mut();
        add(&mut l.values, CHAINS, chains);
        add(&mut l.values, LOOKUPS, counts.lookups);
        add(&mut l.values, MISSES, counts.misses);
        add(&mut l.values, INDIRECT, counts.indirect);
        add(&mut l.values, FETCH, counts.fetch_calls);
    });
}

impl Drop for RunGuard {
    fn drop(&mut self) {
        let previous = std::mem::replace(&mut self.previous, Local::EMPTY);
        let mut current = LOCAL.with(|l| l.replace(previous));
        if let Some(start) = current.since {
            add(&mut current.values, current.segment, start.elapsed_ns());
        }
        for (counter, value) in self.record.0.iter().zip(
            current
                .values
                .basic
                .into_iter()
                .chain(current.values.detailed.into_iter().flat_map(|v| v.values)),
        ) {
            if value == 0 {
                continue;
            }
            let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                Some(old.saturating_add(value))
            });
        }
    }
}

/// Foreground backend materialization timer. Wrapping backends should leave
/// instrumentation to the compiler implementation; nested scopes are supported.
#[doc(hidden)]
pub struct ForegroundCompileTimer(usize);
impl ForegroundCompileTimer {
    /// Time compilation inside the current vCPU run, excluding it from execution.
    pub fn enter() -> Self {
        LOCAL.with(|l| {
            let mut l = l.borrow_mut();
            let previous = l.segment;
            switch(&mut l, COMPILER);
            Self(previous)
        })
    }
}
impl Drop for ForegroundCompileTimer {
    fn drop(&mut self) {
        LOCAL.with(|l| switch(&mut l.borrow_mut(), self.0));
    }
}

/// Run-scoped helper timing hook for backend Rust call boundaries. No allocation,
/// locks, TLS access or clock reads in the disabled `measure` path. Safe for
/// nested VM runs; do not use from a signal handler. Standalone backend execution
/// outside Vcpu is ignored.
#[doc(hidden)]
pub struct HelperTimer {
    _sample: SampleGuard,
}
impl HelperTimer {
    /// Count a compiled helper call and time one in every 64.
    /// Call only when the current run enables detailed accounting.
    #[inline]
    pub fn enter() -> Self {
        Self {
            _sample: SampleGuard::enter(HELPER, None),
        }
    }

    /// Execute a helper with one flag branch and no TLS access when disabled.
    #[inline]
    pub fn measure<T>(enabled: bool, call: impl FnOnce() -> T) -> T {
        if !enabled {
            return call();
        }
        let _timer = Self::enter();
        call()
    }
}
#[derive(Clone, Copy)]
struct MissTiming {
    start: MonotonicTime,
    compiler: u64,
    approval: u64,
    lift: u64,
}

pub(crate) struct ResolveTimer {
    _mixed: SampleGuard,
}
impl ResolveTimer {
    /// Called behind the VM's accounting flag; disabled resolves never touch TLS.
    pub(crate) fn enter(calls: u64) -> Self {
        let mixed = SampleGuard::fetch(calls);
        LOCAL.with(|local| {
            let mut l = local.borrow_mut();
            l.resolving = true;
            l.hit_start = (l.values[FETCH_HITS] % 64 == 0).then(|| MissTiming {
                start: mixed.start.unwrap_or_else(MonotonicTime::now),
                compiler: l.values[COMPILER],
                approval: l.values[FETCH_APPROVAL],
                lift: l.values[FETCH_LIFT],
            });
            l.miss = None;
            l.values.detailed.as_mut().unwrap().publication_ns = 0;
            Self { _mixed: mixed }
        })
    }

    pub(crate) fn miss() {
        LOCAL.with(|local| {
            let mut l = local.borrow_mut();
            if l.miss.is_none() {
                l.miss = Some(l.hit_start.take().unwrap_or_else(|| MissTiming {
                    start: MonotonicTime::now(),
                    compiler: l.values[COMPILER],
                    approval: l.values[FETCH_APPROVAL],
                    lift: l.values[FETCH_LIFT],
                }));
                add(&mut l.values, FETCH_MISSES, 1);
            }
        });
    }
}
impl Drop for ResolveTimer {
    fn drop(&mut self) {
        LOCAL.with(|local| {
            let mut l = local.borrow_mut();
            if let Some(miss) = l.miss.take() {
                let approval = l.values[FETCH_APPROVAL].saturating_sub(miss.approval);
                let lift = l.values[FETCH_LIFT].saturating_sub(miss.lift);
                let ns = miss
                    .start
                    .elapsed_ns()
                    .saturating_sub(l.values[COMPILER].saturating_sub(miss.compiler))
                    .saturating_sub(approval)
                    .saturating_sub(lift);
                add(&mut l.values, FETCH_MISS_APPROVAL, approval);
                add(&mut l.values, FETCH_MISS_LIFT, lift);
                add(&mut l.values, FETCH_MISS_OTHER, ns);
            } else {
                let publication_ns = l.values.detailed.as_ref().unwrap().publication_ns;
                add(&mut l.values, HIT_PUBLICATION, publication_ns);
                add(&mut l.values, FETCH_HITS, 1);
                if let Some(hit) = l.hit_start {
                    let ns = hit
                        .start
                        .elapsed_ns()
                        .saturating_sub(l.values[COMPILER].saturating_sub(hit.compiler));
                    add(&mut l.values, FETCH_HITS + 1, 1);
                    add(&mut l.values, FETCH_HITS + 2, ns);
                    if ns > SLOW_HIT_THRESHOLD_NS {
                        add(&mut l.values, HIT_SLOW_SAMPLES, 1);
                        add(&mut l.values, HIT_SLOW_NS, ns);
                    }
                }
            }
            l.resolving = false;
        });
    }
}

/// Internal shared-cache instrumentation, called only behind the accounting flag.
#[doc(hidden)]
pub struct ShardReadTimer(Option<MonotonicTime>);
impl ShardReadTimer {
    /// Count a first-try failure and sample its blocking acquisition.
    pub fn contended() -> Self {
        LOCAL.with(|local| {
            let mut l = local.borrow_mut();
            add(&mut l.values, SHARD_CONTENTION, 1);
            Self((l.values[SHARD_CONTENTION] % 64 == 1).then(MonotonicTime::now))
        })
    }
}
impl Drop for ShardReadTimer {
    fn drop(&mut self) {
        if let Some(start) = self.0 {
            let ns = start.elapsed_ns();
            LOCAL.with(|local| {
                let mut l = local.borrow_mut();
                add(&mut l.values, SHARD_CONTENTION + 1, 1);
                add(&mut l.values, SHARD_CONTENTION + 2, ns);
            });
        }
    }
}

pub(crate) struct PublicationTimer(MonotonicTime);
impl PublicationTimer {
    pub(crate) fn enter() -> Self {
        Self(MonotonicTime::now())
    }
}
impl Drop for PublicationTimer {
    fn drop(&mut self) {
        let ns = self.0.elapsed_ns();
        LOCAL.with(|local| {
            let mut l = local.borrow_mut();
            let detailed = l.values.detailed.as_mut().unwrap();
            detailed.publication_ns = detailed.publication_ns.saturating_add(ns);
        });
    }
}

pub(crate) struct FetchPhaseTimer {
    phase: FetchPhase,
    start: Option<MonotonicTime>,
    approval: u64,
    policy: u64,
}
#[derive(Clone, Copy)]
enum FetchPhase {
    Approval,
    Policy,
    Lift,
}
impl FetchPhaseTimer {
    pub(crate) fn approval() -> Option<Self> {
        Self::enter(FetchPhase::Approval)
    }
    pub(crate) fn policy() -> Option<Self> {
        Self::enter(FetchPhase::Policy)
    }
    pub(crate) fn lift() -> Option<Self> {
        Self::enter(FetchPhase::Lift)
    }
    fn enter(phase: FetchPhase) -> Option<Self> {
        LOCAL.with(|local| {
            let mut l = local.borrow_mut();
            if !l.resolving {
                return None;
            }
            let nested = l.approval_depth != 0;
            if matches!(phase, FetchPhase::Approval) {
                l.approval_depth += 1;
            } else if nested {
                // Work inside a callback already belongs to its outer timer.
                return None;
            }
            Some(Self {
                phase,
                start: (!nested).then(MonotonicTime::now),
                approval: l.values[FETCH_APPROVAL],
                policy: l.policy_ns,
            })
        })
    }
}
impl Drop for FetchPhaseTimer {
    fn drop(&mut self) {
        LOCAL.with(|local| {
            let mut l = local.borrow_mut();
            if matches!(self.phase, FetchPhase::Approval) {
                l.approval_depth -= 1;
            }
            let Some(start) = self.start else {
                return;
            };
            let ns = start.elapsed_ns();
            let approval = l.values[FETCH_APPROVAL].saturating_sub(self.approval);
            match self.phase {
                FetchPhase::Approval => add(&mut l.values, FETCH_APPROVAL, ns),
                FetchPhase::Policy => {
                    l.policy_ns = l.policy_ns.saturating_add(ns.saturating_sub(approval));
                }
                FetchPhase::Lift => {
                    let ns = ns
                        .saturating_sub(approval)
                        .saturating_sub(l.policy_ns.saturating_sub(self.policy));
                    add(&mut l.values, FETCH_LIFT, ns);
                }
            }
        });
    }
}

pub(crate) struct SampleGuard {
    field: usize,
    start: Option<MonotonicTime>,
    compiler_before: u64,
    nested: bool,
    helper: bool,
}
impl SampleGuard {
    pub(crate) fn fetch(calls: u64) -> Self {
        Self::enter(FETCH, Some(calls))
    }
    fn enter(field: usize, calls: Option<u64>) -> Self {
        LOCAL.with(|l| {
            let mut l = l.borrow_mut();
            let nested = field == HELPER && l.helper_depth != 0;
            if field == HELPER {
                l.helper_depth += 1;
            }
            let start = if l.active && !nested {
                let calls = calls.unwrap_or_else(|| {
                    add(&mut l.values, field, 1);
                    l.values[field]
                });
                (l.enabled && calls % 64 == 1).then(MonotonicTime::now)
            } else {
                None
            };
            Self {
                field,
                start,
                compiler_before: l.values[COMPILER],
                nested,
                helper: field == HELPER,
            }
        })
    }
}
impl Drop for SampleGuard {
    fn drop(&mut self) {
        if self.start.is_none() && !self.helper {
            return;
        }
        LOCAL.with(|l| {
            let mut l = l.borrow_mut();
            if self.helper {
                l.helper_depth -= 1;
            }
            if !self.nested {
                if let Some(start) = self.start {
                    let ns = start
                        .elapsed_ns()
                        .saturating_sub(l.values[COMPILER].saturating_sub(self.compiler_before));
                    add(&mut l.values, self.field + 1, 1);
                    add(&mut l.values, self.field + 2, ns);
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_run_has_no_detailed_counter_storage() {
        let mut registry = Registry::default();
        let run = RunGuard::enter(registry.register(), false, None);
        LOCAL.with(|local| assert!(local.borrow().values.detailed.is_none()));
        run.finish(Some(&crate::Exit::Hlt));
        drop(run);
        assert_eq!(registry.snapshot().exits.hlt.count, 1);
    }

    #[test]
    fn contention_sampling_is_first_then_every_64th() {
        let mut registry = Registry::default();
        let run = RunGuard::enter(registry.register(), true, None);
        for _ in 0..130 {
            drop(ShardReadTimer::contended());
        }
        drop(run);
        let stats = registry.snapshot();
        assert_eq!(stats.fetch_shard_read_contention.calls, 130);
        assert_eq!(stats.fetch_shard_read_contention.samples, 3);
    }

    #[test]
    fn disabled_helper_never_borrows_local_accounting() {
        LOCAL.with(|local| {
            let _borrow = local.borrow_mut();
            HelperTimer::measure(false, || ());
        });
    }

    #[test]
    fn disabled_fetch_callback_never_borrows_local_accounting() {
        let mut vm = crate::Vm::new(crate::VmConfig::flat(0x1000));
        vm.set_fetch_policy(Box::new(|_| crate::FetchVerdict::Allow));
        LOCAL.with(|local| {
            let _borrow = local.borrow_mut();
            assert_eq!(
                vm.mem.fetch_policy.as_ref().unwrap().approve(0, &vm.mem),
                crate::FetchVerdict::Allow
            );
        });
    }

    #[test]
    fn policy_wait_is_other_time() {
        use std::sync::Barrier;
        use std::time::Duration;

        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let mut vm = crate::Vm::new(crate::VmConfig::flat(0x1000));
        vm.set_time_accounting(true);
        vm.set_fetch_policy(Box::new({
            let entered = entered.clone();
            let release = release.clone();
            move |_| {
                entered.wait();
                release.wait();
                crate::FetchVerdict::Allow
            }
        }));
        let mut registry = Registry::default();
        let record = registry.register();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                assert_eq!(
                    vm.mem.fetch_policy.as_ref().unwrap().approve(0, &vm.mem),
                    crate::FetchVerdict::Allow
                );
            });
            entered.wait();
            let _run = RunGuard::enter(record, true, None);
            let _resolve = ResolveTimer::enter(1);
            ResolveTimer::miss();
            let _lift = FetchPhaseTimer::lift();
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(30));
                release.wait();
            });
            assert_eq!(
                vm.mem.fetch_policy.as_ref().unwrap().approve(0, &vm.mem),
                crate::FetchVerdict::Allow
            );
            // The only timed work in this lift scope is approval bookkeeping.
            LOCAL.with(|local| {
                let l = local.borrow();
                assert!(l.policy_ns >= 20_000_000);
                assert_eq!(l.values[FETCH_APPROVAL], 0);
            });
        });
        let stats = registry.snapshot();
        assert!(stats.fetch_miss_other_ns >= 20_000_000, "{stats:?}");
        assert!(
            stats.fetch_miss_lift_ns < stats.fetch_miss_other_ns,
            "{stats:?}"
        );
        assert!(
            stats.fetch_miss_approval_ns + stats.fetch_miss_lift_ns + stats.fetch_miss_other_ns
                <= stats.fetch_translate.sample_ns,
            "{stats:?}"
        );
    }
}
