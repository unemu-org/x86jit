use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// Completed acquisitions of one compile-serialization lock.
///
/// All counters saturate at `u64::MAX`. Durations use monotonic wall time in
/// nanoseconds. Histogram bucket 0 counts zero; bucket `i` (1 through 30) counts
/// `[2^(i-1), 2^i)` ns; bucket 31 counts everything at or above `2^30` ns.
/// An empty entry has zero acquisitions, durations, and histogram counts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LockTelemetry {
    pub acquisitions: u64,
    pub total_hold_ns: u64,
    pub max_hold_ns: u64,
    pub total_wait_ns: u64,
    pub max_wait_ns: u64,
    pub hold_histogram: [u64; 32],
    pub wait_histogram: [u64; 32],
}

impl LockTelemetry {
    fn record(&mut self, hold_ns: u64, wait_ns: u64) {
        self.acquisitions = self.acquisitions.saturating_add(1);
        self.total_hold_ns = self.total_hold_ns.saturating_add(hold_ns);
        self.max_hold_ns = self.max_hold_ns.max(hold_ns);
        self.total_wait_ns = self.total_wait_ns.saturating_add(wait_ns);
        self.max_wait_ns = self.max_wait_ns.max(wait_ns);
        let hold = &mut self.hold_histogram[bucket(hold_ns)];
        *hold = hold.saturating_add(1);
        let wait = &mut self.wait_histogram[bucket(wait_ns)];
        *wait = wait.saturating_add(1);
    }
}

fn bucket(ns: u64) -> usize {
    (u64::BITS - ns.leading_zeros()).min(31) as usize
}

fn nanos(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

/// A coherent snapshot of completed lock holds, including errors and unwinds.
///
/// Compilation locks `compile_serial` first, then one module. Link invalidation
/// takes each module separately without `compile_serial` or lazy construction.
/// Module wait measures only the module mutex. The `compile_serial` hold includes
/// the nested module wait and hold. Do not add the entries to estimate elapsed time.
/// Observer callbacks run after both compilation guards have been released.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LockTelemetrySnapshot {
    /// Global serialization of foreground and background compilers.
    pub compile_serial: LockTelemetry,
    /// Default-mode module access, including lazy construction and invalidation.
    pub default_module: LockTelemetry,
    /// Non-default-mode module access, including lazy construction and invalidation.
    pub nondefault_module: LockTelemetry,
}

#[derive(Clone, Copy)]
pub(crate) enum LockKind {
    CompileSerial,
    DefaultModule,
    NondefaultModule,
}

#[derive(Default)]
pub(crate) struct Telemetry {
    enabled: AtomicBool,
    snapshot: Mutex<LockTelemetrySnapshot>,
    #[cfg(test)]
    waiting: [std::sync::atomic::AtomicUsize; 3],
}

impl Telemetry {
    pub(crate) fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    pub(crate) fn snapshot(&self) -> LockTelemetrySnapshot {
        self.snapshot
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub(crate) fn reset(&self) {
        *self.snapshot.lock().unwrap_or_else(|e| e.into_inner()) = LockTelemetrySnapshot::default();
    }

    pub(crate) fn lock<'a, T>(
        &'a self,
        mutex: &'a Mutex<T>,
        kind: LockKind,
        what: &str,
    ) -> Guard<'a, T> {
        let started = self.enabled.load(Ordering::Relaxed).then(Instant::now);
        #[cfg(test)]
        if started.is_some() {
            self.waiting[kind as usize].fetch_add(1, Ordering::Relaxed);
        }
        let guard = super::lock_recovering(mutex, what);
        #[cfg(test)]
        if started.is_some() {
            self.waiting[kind as usize].fetch_sub(1, Ordering::Relaxed);
        }
        let timing = started.map(|started| {
            let acquired = Instant::now();
            (
                self,
                kind,
                acquired,
                nanos(acquired.duration_since(started)),
            )
        });
        Guard {
            guard: Some(guard),
            timing,
        }
    }

    #[cfg(test)]
    pub(crate) fn waiting(&self, kind: LockKind) -> usize {
        self.waiting[kind as usize].load(Ordering::Relaxed)
    }

    fn record(&self, kind: LockKind, hold_ns: u64, wait_ns: u64) {
        let mut snapshot = self.snapshot.lock().unwrap_or_else(|e| e.into_inner());
        let entry = match kind {
            LockKind::CompileSerial => &mut snapshot.compile_serial,
            LockKind::DefaultModule => &mut snapshot.default_module,
            LockKind::NondefaultModule => &mut snapshot.nondefault_module,
        };
        entry.record(hold_ns, wait_ns);
    }
}

pub(crate) struct Guard<'a, T> {
    guard: Option<MutexGuard<'a, T>>,
    timing: Option<(&'a Telemetry, LockKind, Instant, u64)>,
}

impl<T> Deref for Guard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.guard.as_deref().expect("live lock guard")
    }
}

impl<T> DerefMut for Guard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.guard.as_deref_mut().expect("live lock guard")
    }
}

impl<T> Drop for Guard<'_, T> {
    fn drop(&mut self) {
        let sample = self.timing.map(|(telemetry, kind, acquired, wait)| {
            (telemetry, kind, nanos(acquired.elapsed()), wait)
        });
        // Release the measured lock before taking the telemetry mutex. Snapshot
        // and reset never acquire compile/module locks or invoke callbacks.
        drop(self.guard.take());
        if let Some((telemetry, kind, hold, wait)) = sample {
            telemetry.record(kind, hold, wait);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn lock_telemetry_poison_recovery_with_broken_stderr() {
        const CHILD: &str = "X86JIT_TEST_BROKEN_RECOVERY_STDERR";
        if std::env::var_os(CHILD).is_none() {
            // Stderr redirection and the recovery diagnostic's Once are process-wide.
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    concat!(
                        "lock_telemetry::tests::",
                        "lock_telemetry_poison_recovery_with_broken_stderr"
                    ),
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            print!("{stdout}");
            assert!(
                output.status.success() && stdout.contains("running 1 test"),
                "{output:?}"
            );
            return;
        }

        use std::io::Write;
        use std::os::fd::AsRawFd;
        use std::os::unix::net::UnixStream;

        let telemetry = Telemetry::default();
        telemetry.set_enabled(true);
        let mutex = Mutex::new(());
        assert!(std::panic::catch_unwind(|| {
            let _held = mutex.lock().unwrap();
            panic!("poison recovery probe");
        })
        .is_err());

        let (reader, writer) = UnixStream::pair().unwrap();
        drop(reader);
        // SAFETY: writer owns a live descriptor; dup2 redirects only this child's stderr.
        assert_eq!(unsafe { libc::dup2(writer.as_raw_fd(), 2) }, 2);
        assert!(std::io::stderr().write_all(b"broken pipe probe\n").is_err());
        let recovered = std::panic::catch_unwind(|| {
            drop(telemetry.lock(&mutex, LockKind::CompileSerial, "test"));
        });
        let acquisitions = telemetry.snapshot().compile_serial.acquisitions;
        let released = matches!(mutex.try_lock(), Err(std::sync::TryLockError::Poisoned(_)));
        println!(
            "recovery_unwound={}, acquisitions={acquisitions}, mutex_released={released}",
            recovered.is_err()
        );
        assert!(recovered.is_ok());
        assert_eq!(acquisitions, 1);
        assert!(released);
    }

    #[test]
    fn lock_telemetry_empty_snapshot_and_active_guard_switch_reset() {
        let telemetry = Telemetry::default();
        let mutex = Mutex::new(());
        let empty = telemetry.snapshot();
        for entry in [
            &empty.compile_serial,
            &empty.default_module,
            &empty.nondefault_module,
        ] {
            assert_eq!(entry.acquisitions, 0);
            assert_eq!(entry.total_hold_ns, 0);
            assert_eq!(entry.max_hold_ns, 0);
            assert_eq!(entry.total_wait_ns, 0);
            assert_eq!(entry.max_wait_ns, 0);
            assert_eq!(entry.hold_histogram, [0; 32]);
            assert_eq!(entry.wait_histogram, [0; 32]);
        }
        let disabled = telemetry.lock(&mutex, LockKind::CompileSerial, "test");
        telemetry.set_enabled(true);
        drop(disabled);
        assert_eq!(telemetry.snapshot(), empty);
        let enabled = telemetry.lock(&mutex, LockKind::CompileSerial, "test");
        assert_eq!(telemetry.snapshot(), empty);
        telemetry.reset();
        telemetry.set_enabled(false);
        drop(enabled);
        assert_eq!(telemetry.snapshot().compile_serial.acquisitions, 1);
        telemetry.reset();
        assert_eq!(telemetry.snapshot(), empty);
    }

    #[test]
    fn lock_telemetry_histogram_boundaries_and_top_saturation() {
        assert_eq!(bucket(0), 0);
        for power in 0..30 {
            assert_eq!(bucket(1 << power), power + 1);
            assert_eq!(bucket((1 << (power + 1)) - 1), power + 1);
        }
        for ns in [1 << 30, 1 << 31, 1 << 63, u64::MAX] {
            assert_eq!(bucket(ns), 31);
        }
        let mut entry = LockTelemetry::default();
        entry.record(0, u64::MAX);
        assert_eq!(entry.hold_histogram[0], 1);
        assert_eq!(entry.wait_histogram[31], 1);
    }

    #[test]
    fn lock_telemetry_totals_counts_and_duration_conversion_saturate() {
        let mut entry = LockTelemetry::default();
        entry.record(u64::MAX, u64::MAX);
        entry.acquisitions = u64::MAX;
        entry.hold_histogram[31] = u64::MAX;
        entry.wait_histogram[31] = u64::MAX;
        entry.record(u64::MAX, u64::MAX);
        assert_eq!(entry.acquisitions, u64::MAX);
        assert_eq!(entry.total_hold_ns, u64::MAX);
        assert_eq!(entry.total_wait_ns, u64::MAX);
        assert_eq!(entry.max_hold_ns, u64::MAX);
        assert_eq!(entry.max_wait_ns, u64::MAX);
        assert_eq!(entry.hold_histogram[31], u64::MAX);
        assert_eq!(entry.wait_histogram[31], u64::MAX);
        assert_eq!(nanos(Duration::MAX), u64::MAX);
    }
}
