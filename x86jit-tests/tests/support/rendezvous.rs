//! Two-worker litmus rendezvous shared by the ordering tests.
//!
//! A coordinator thread runs trials one at a time: it resets guest memory,
//! releases both workers into trial `epoch`, and waits until both have
//! published a result. Waiters spin and yield only once per `YIELD_EVERY`:
//! on a host with more runnable threads than cores, every yield can hand the
//! core away for a whole time slice, and a trial needs all three threads.
//!
//! The run is paced by progress, not by a total deadline. Past `budget` the
//! sample ends early once `floor` trials have completed; before that it keeps
//! going however slow the host is, and fails only when one trial does not
//! complete within `STALL`.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// Longest wait for one trial before the run is declared stalled.
pub const STALL: Duration = Duration::from_secs(60);
/// How long a waiter spins between yields of its time slice.
const YIELD_EVERY: Duration = Duration::from_millis(1);

/// Wait until `ready()` holds, or until `give_up()` does; returns whether
/// `ready()` held.
fn wait(ready: impl Fn() -> bool, give_up: impl Fn() -> bool) -> bool {
    let mut spun = Instant::now();
    loop {
        for _ in 0..64 {
            if ready() {
                return true;
            }
            std::hint::spin_loop();
        }
        if spun.elapsed() >= YIELD_EVERY {
            if give_up() {
                return false;
            }
            thread::yield_now();
            spun = Instant::now();
        }
    }
}

#[derive(Default)]
pub struct Rendezvous {
    epoch: AtomicUsize,
    done: [AtomicUsize; 2],
    result: [AtomicU64; 2],
    stop: AtomicBool,
}

impl Rendezvous {
    /// Worker: wait for trial `epoch`; false once the coordinator stopped.
    pub fn wait_start(&self, epoch: usize) -> bool {
        wait(
            || self.epoch.load(Ordering::Acquire) == epoch,
            || self.stop.load(Ordering::Acquire),
        )
    }

    /// Worker `id`: publish the result of trial `epoch`.
    pub fn finish(&self, id: usize, epoch: usize, result: u64) {
        self.result[id].store(result, Ordering::Relaxed);
        self.done[id].store(epoch, Ordering::Release);
    }

    /// Coordinator: run up to `iters` trials. Each calls `reset` before the
    /// workers start and `tally` with both results after they finish. Past
    /// `budget`, the run ends early once `floor` trials completed. Returns
    /// the number of completed trials, or `Err` with it when no trial
    /// completed for `STALL`. Always stops the workers before returning.
    pub fn run(
        &self,
        iters: usize,
        floor: usize,
        budget: Duration,
        mut reset: impl FnMut(),
        mut tally: impl FnMut([u64; 2]),
    ) -> Result<usize, usize> {
        let start = Instant::now();
        let mut outcome = Ok(iters);
        for epoch in 1..=iters {
            let completed = epoch - 1;
            if completed >= floor && start.elapsed() >= budget {
                outcome = Ok(completed);
                break;
            }
            reset();
            self.epoch.store(epoch, Ordering::Release);
            let trial = Instant::now();
            let finished = wait(
                || self.done.iter().all(|d| d.load(Ordering::Acquire) == epoch),
                || trial.elapsed() >= STALL,
            );
            if !finished {
                outcome = Err(completed);
                break;
            }
            tally([
                self.result[0].load(Ordering::Relaxed),
                self.result[1].load(Ordering::Relaxed),
            ]);
        }
        self.stop.store(true, Ordering::Release);
        outcome
    }
}
