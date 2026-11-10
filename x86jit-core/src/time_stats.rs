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
    /// are counted only while detailed accounting is enabled; fetch calls always.
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
            ((Self::now().0.saturating_sub(self.0) as u128 * 1_000_000_000) / u128::from(frequency))
                .min(u64::MAX as u128) as u64
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
const FIELDS: usize = EXITS + 2 * CATEGORIES;

#[derive(Default)]
pub(crate) struct Record([AtomicU64; FIELDS]);

#[derive(Default)]
pub(crate) struct Registry {
    records: Vec<Arc<Record>>,
    retired: [u64; FIELDS],
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

#[derive(Clone, Copy, Default)]
struct Local {
    values: [u64; FIELDS],
    active: bool,
    enabled: bool,
    segment: usize,
    since: Option<MonotonicTime>,
    helper_depth: u32,
}
thread_local! { static LOCAL: RefCell<Local> = const { RefCell::new(Local { values: [0; FIELDS], active: false, enabled: false, segment: INTERPRETER, since: None, helper_depth: 0 }) }; }

fn add(values: &mut [u64; FIELDS], i: usize, value: u64) {
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
        let mut current = LOCAL.with(|l| l.replace(self.previous));
        if let Some(start) = current.since {
            add(&mut current.values, current.segment, start.elapsed_ns());
        }
        for (counter, value) in self.record.0.iter().zip(current.values) {
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
    fn disabled_helper_never_borrows_local_accounting() {
        LOCAL.with(|local| {
            let _borrow = local.borrow_mut();
            HelperTimer::measure(false, || ());
        });
    }
}
