//! Source-publication ordering (§10). Select before guest writers can start.
//!
//! SDM 325462-092US Vol 3A §11.2.3.4, Example 11-3 permits both
//! post-store loads to miss the other store. Degraded mode pairs full fences
//! on both sides; normal mode replaces writer fences with a process barrier.

use std::io;
use std::sync::atomic::{compiler_fence, fence, AtomicI32, AtomicU64, Ordering};

#[cfg(any(target_os = "linux", target_os = "freebsd", test))]
use std::sync::atomic::AtomicBool;

/// Source-publication protocol, fixed when guest memory is constructed (§10).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SourceBarrierMode {
    /// Process-wide barrier on newly tagged sources; compiler barriers on writers.
    #[default]
    Process,
    /// Paired writer/lifter fences and mandatory JIT write notification.
    Degraded,
}

pub(crate) struct ProcessBarrier {
    pub(crate) mode: SourceBarrierMode,
    failures: AtomicU64,
    last_errno: AtomicI32,
    pub(crate) uncached_executions: AtomicU64,
    pub(crate) calls: AtomicU64,
    pub(crate) ns: AtomicU64,
    pub(crate) tagged_pages: AtomicU64,
}

impl ProcessBarrier {
    pub(crate) fn new() -> Self {
        let mut barrier = Self {
            mode: SourceBarrierMode::Process,
            failures: AtomicU64::new(0),
            last_errno: AtomicI32::new(0),
            uncached_executions: AtomicU64::new(0),
            calls: AtomicU64::new(0),
            ns: AtomicU64::new(0),
            tagged_pages: AtomicU64::new(0),
        };
        #[cfg(any(test, feature = "test-hooks"))]
        let probe = if FORCE_UNAVAILABLE.with(|v| v.get()) {
            barrier.failed(&io::Error::from_raw_os_error(1));
            Err(io::Error::from_raw_os_error(1))
        } else if FORCE_SUCCESS.with(|v| v.get()) {
            Ok(())
        } else {
            os::probe(&barrier)
        };
        #[cfg(not(any(test, feature = "test-hooks")))]
        let probe = os::probe(&barrier);
        if probe.is_err() {
            barrier.mode = SourceBarrierMode::Degraded;
        }
        barrier
    }

    fn failed(&self, error: &io::Error) {
        self.failures.fetch_add(1, Ordering::Relaxed);
        self.last_errno
            .store(error.raw_os_error().unwrap_or(0), Ordering::Relaxed);
    }

    pub(crate) fn stats(&self) -> (SourceBarrierMode, u64, Option<i32>) {
        let errno = self.last_errno.load(Ordering::Relaxed);
        (
            self.mode,
            self.failures.load(Ordering::Relaxed),
            (errno != 0).then_some(errno),
        )
    }

    #[inline(always)]
    pub(crate) fn writer_fence(&self) {
        if self.mode == SourceBarrierMode::Degraded {
            fence(Ordering::SeqCst);
        } else {
            compiler_fence(Ordering::SeqCst);
        }
    }

    pub(crate) fn flush(&self) -> bool {
        fence(Ordering::SeqCst);
        if self.mode == SourceBarrierMode::Process {
            // A per-thread filter can reject a command after another thread's
            // successful probe. Keep the mode fixed and refuse publication.
            for attempt in 0..3 {
                let start = crate::time_stats::MonotonicTime::now();
                #[cfg(any(test, feature = "test-hooks"))]
                let result = if FORCE_FLUSH_FAILURE.with(|v| v.get()) {
                    Err(io::Error::from_raw_os_error(1))
                } else if FORCE_SUCCESS.with(|v| v.get()) {
                    Ok(())
                } else {
                    os::flush()
                };
                #[cfg(not(any(test, feature = "test-hooks")))]
                let result = os::flush();
                let ns = start.elapsed_ns();
                self.calls.fetch_add(1, Ordering::Relaxed);
                self.ns.fetch_add(ns, Ordering::Relaxed);
                match result {
                    Ok(()) => break,
                    Err(error) => {
                        self.failed(&error);
                        if attempt == 2 {
                            return false;
                        }
                    }
                }
                std::thread::yield_now();
            }
        }
        fence(Ordering::SeqCst);
        true
    }
}

// ENOSYS differs across these hosts; EINVAL and EPERM share their values.
#[cfg(any(target_os = "linux", target_os = "freebsd", test))]
fn unsupported(errno: Option<i32>) -> bool {
    #[cfg(target_os = "freebsd")]
    const ENOSYS: i32 = 78;
    #[cfg(not(target_os = "freebsd"))]
    const ENOSYS: i32 = 38;
    matches!(errno, Some(1 | 22)) || errno == Some(ENOSYS)
}

#[cfg(any(target_os = "linux", target_os = "freebsd", test))]
fn retry(stats: &ProcessBarrier, mut command: impl FnMut() -> io::Result<()>) -> io::Result<()> {
    for attempt in 0..3 {
        match command() {
            Ok(()) => return Ok(()),
            Err(error) => {
                stats.failed(&error);
                if unsupported(error.raw_os_error()) || attempt == 2 {
                    return Err(error);
                }
                std::thread::yield_now();
            }
        }
    }
    unreachable!()
}

#[cfg(any(target_os = "linux", target_os = "freebsd", test))]
fn register(
    registered: &AtomicBool,
    stats: &ProcessBarrier,
    command: impl FnMut() -> io::Result<()>,
) -> io::Result<()> {
    if !registered.load(Ordering::Acquire) {
        retry(stats, command)?;
        registered.store(true, Ordering::Release);
    }
    Ok(())
}

#[cfg(any(test, feature = "test-hooks"))]
thread_local! {
    static FORCE_UNAVAILABLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FORCE_FLUSH_FAILURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FORCE_SUCCESS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Model successful host commands for single-threaded protocol witnesses.
/// This does not issue a real process-wide barrier.
#[cfg(any(test, feature = "test-hooks"))]
#[doc(hidden)]
pub fn with_barrier_success<T>(run: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            FORCE_SUCCESS.with(|v| v.set(self.0));
        }
    }
    let _reset = Reset(FORCE_SUCCESS.with(|v| v.replace(true)));
    run()
}

/// Reject publication barriers after construction, without changing the VM's mode.
#[cfg(any(test, feature = "test-hooks"))]
#[doc(hidden)]
pub fn with_barrier_flush_failure<T>(run: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            FORCE_FLUSH_FAILURE.with(|v| v.set(self.0));
        }
    }
    let _reset = Reset(FORCE_FLUSH_FAILURE.with(|v| v.replace(true)));
    run()
}

/// Run a constructor with an unavailable process barrier. Only for witnesses and benchmarks.
#[cfg(any(test, feature = "test-hooks"))]
#[doc(hidden)]
pub fn with_barrier_unavailable<T>(construct: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            FORCE_UNAVAILABLE.with(|v| v.set(self.0));
        }
    }
    let _reset = Reset(FORCE_UNAVAILABLE.with(|v| v.replace(true)));
    construct()
}

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
mod os {
    use super::*;
    use std::sync::atomic::AtomicBool;

    const PRIVATE_EXPEDITED: i32 = 1 << 3;
    const REGISTER_PRIVATE_EXPEDITED: i32 = 1 << 4;

    // Linux membarrier(2), DESCRIPTION/Memory ordering:
    // https://man7.org/linux/man-pages/man2/membarrier.2.html
    // FreeBSD membarrier(2), DESCRIPTION (introduced in 14.1):
    // https://man.freebsd.org/cgi/man.cgi?query=membarrier&sektion=2
    // PRIVATE_EXPEDITED orders every running sibling; registration is process-wide.
    // Use the same one-time registration pattern as code_memory::linux_sync,
    // with the data-ordering command rather than instruction synchronization.
    fn command(cmd: i32) -> io::Result<()> {
        #[cfg(target_os = "linux")]
        unsafe extern "C" {
            fn syscall(number: std::ffi::c_long, ...) -> std::ffi::c_long;
        }
        #[cfg(target_os = "freebsd")]
        unsafe extern "C" {
            fn syscall(number: i32, ...) -> i32;
        }
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        const NUMBER: std::ffi::c_long = 324;
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        const NUMBER: std::ffi::c_long = 283;
        // FreeBSD sys/kern/syscalls.master: membarrier, syscall 584. Calling
        // syscall also permits running against a libc predating the new kernel API.
        #[cfg(target_os = "freebsd")]
        const NUMBER: i32 = 584;
        // SAFETY: membarrier accepts a command, zero flags and an unused CPU ID.
        if unsafe { syscall(NUMBER, cmd, 0u32, 0i32) } == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    static REGISTERED: AtomicBool = AtomicBool::new(false);

    pub(super) fn probe(stats: &ProcessBarrier) -> io::Result<()> {
        register(&REGISTERED, stats, || command(REGISTER_PRIVATE_EXPEDITED))?;
        retry(stats, || command(PRIVATE_EXPEDITED))
    }

    pub(super) fn flush() -> io::Result<()> {
        command(PRIVATE_EXPEDITED)
    }
}

#[cfg(target_os = "macos")]
mod os {
    use super::*;

    unsafe extern "C" {
        static mach_task_self_: u32;
        fn mach_thread_self() -> u32;
        fn mach_port_deallocate(task: u32, port: u32) -> i32;
    }

    pub(super) fn probe(stats: &ProcessBarrier) -> io::Result<()> {
        // Validate self-identification even though no siblings are enumerated.
        // XNU osfmk/mach/port.h: MACH_PORT_NULL = 0, MACH_PORT_DEAD = ~0.
        // SAFETY: these calls acquire/release only the calling thread's send right.
        let current = unsafe { mach_thread_self() };
        let errno = if current == 0 || current == u32::MAX {
            22 // EINVAL
        } else {
            // SAFETY: a non-null self right belongs to the current task.
            unsafe { mach_port_deallocate(mach_task_self_, current) };
            // XNU libsyscall/wrappers/thread_register_state.c calls thread_get_state;
            // mach/thread_act.defs permits stale state for an executing target.
            // There is no public process-memory-ordering guarantee. Use paired
            // store/lifter fences for this VM's whole lifetime (§10).
            45 // Darwin ENOTSUP (sys/errno.h).
        };
        let error = io::Error::from_raw_os_error(errno);
        stats.failed(&error);
        Err(error)
    }

    pub(super) fn flush() -> io::Result<()> {
        Err(io::Error::from_raw_os_error(45))
    }
}

#[cfg(not(any(target_os = "linux", target_os = "freebsd", target_os = "macos")))]
mod os {
    use super::*;

    pub(super) fn probe(stats: &ProcessBarrier) -> io::Result<()> {
        let error = io::Error::new(io::ErrorKind::Unsupported, "process barrier");
        stats.failed(&error);
        Err(error)
    }

    pub(super) fn flush() -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "process barrier",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counters() -> ProcessBarrier {
        ProcessBarrier {
            mode: SourceBarrierMode::Process,
            failures: AtomicU64::new(0),
            last_errno: AtomicI32::new(0),
            uncached_executions: AtomicU64::new(0),
            calls: AtomicU64::new(0),
            ns: AtomicU64::new(0),
            tagged_pages: AtomicU64::new(0),
        }
    }

    #[test]
    fn registration_retries_transient_errors_and_caches_only_success() {
        let stats = counters();
        let registered = AtomicBool::new(false);
        let mut attempts = 0;
        register(&registered, &stats, || {
            attempts += 1;
            match attempts {
                1 => Err(io::Error::from_raw_os_error(12)), // ENOMEM
                2 => Err(io::Error::from_raw_os_error(4)),  // EINTR
                _ => Ok(()),
            }
        })
        .unwrap();
        assert_eq!(attempts, 3);
        register(&registered, &stats, || {
            panic!("successful registration is cached")
        })
        .unwrap();
        assert_eq!(stats.stats(), (SourceBarrierMode::Process, 2, Some(4)));

        let registered = AtomicBool::new(false);
        assert!(
            register(&registered, &stats, || Err(io::Error::from_raw_os_error(
                12
            )))
            .is_err()
        );
        assert!(!registered.load(Ordering::Relaxed));
        register(&registered, &stats, || Ok(())).unwrap();
        assert!(registered.load(Ordering::Relaxed));
    }

    #[test]
    fn unsupported_errors_are_not_retried() {
        for errno in [1, 22, if cfg!(target_os = "freebsd") { 78 } else { 38 }] {
            let stats = counters();
            assert!(retry(&stats, || Err(io::Error::from_raw_os_error(errno))).is_err());
            assert_eq!(stats.stats().1, 1);
        }
    }

    #[test]
    fn unavailable_probe_selects_degraded_before_writers() {
        let barrier = with_barrier_unavailable(ProcessBarrier::new);
        assert_eq!(barrier.stats(), (SourceBarrierMode::Degraded, 1, Some(1)));
        barrier.writer_fence();
        barrier.flush();
        assert_eq!(barrier.calls.load(Ordering::Relaxed), 0);
        assert_eq!(barrier.ns.load(Ordering::Relaxed), 0);
        assert_eq!(barrier.tagged_pages.load(Ordering::Relaxed), 0);
        assert_eq!(
            barrier.stats().1,
            1,
            "degraded lifts never retry the host barrier"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_uses_paired_fences_without_suspending_threads() {
        let barrier = ProcessBarrier::new();
        assert_eq!(barrier.stats(), (SourceBarrierMode::Degraded, 1, Some(45)));
        std::thread::scope(|scope| {
            scope.spawn(|| {
                barrier.writer_fence();
            });
            barrier.flush();
        });
    }
}
