---
id: TASK-382
title: >-
  jit: quiesce, reset-after-fork and spawn hooks for the background compile
  worker
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - embedding
  - bg-tier
dependencies: []
priority: medium
ordinal: 418000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The background tier-up worker is a `std::thread`. Whichever vcpu submits first spawns it lazily (`ensure_worker`, x86jit-cranelift/src/lib.rs:2358-2371), and only `JitBackend`'s `Drop` joins it (lib.rs:2931-2950). No other part of its lifecycle is reachable through the API. That causes three problems for an embedder that runs a full native userspace with real threads and signals.

1. FORK. POSIX `fork()` copies only the calling thread. The child gets a copy of the address space "possibly including the states of mutexes and other resources" (POSIX.1-2017, fork(), RATIONALE). If the worker was compiling at that moment, the child inherits whichever of these locks it held:
   - the module mutex `Shared::inner`, held for the whole compile from lib.rs:2490;
   - the `queue` and `done` mutexes (lib.rs:2047, 2051);
   - the process-global codemap write lock (codemap.rs:65), taken by `codemap::register` at lib.rs:2712;
   - the perf-map writer lock (perfmap.rs:56), taken at lib.rs:2718.
   The child's next compile then deadlocks. Even when no lock is held, the child's state is wrong:
   - `queue.outstanding` counts a request that no thread will ever finish, so `TierUpHandle::wait_idle` (lib.rs:2787-2792) blocks forever.
   - `worker` still holds a `JoinHandle` for a thread that does not exist in the child. `ensure_worker` never spawns a new one, so tier-up silently stops, and `Drop` later joins a thread id that is not in this process.
   `Vm::fork_with_backend` (x86jit-core/src/vm.rs:388-410) does not help here. It builds a fresh `Vm` with a fresh backend over a deep copy of guest memory. That is the right primitive when the embedder models a guest fork as a second `Vm` in the same host process. It does not fit an embedder that forks the host process itself, where the child carries on with its own copy of everything, translations included.

2. SIGNAL MASK AND SCHEDULING. A new thread inherits the signal mask of the thread that created it (POSIX, pthread_create()), which here is the first vcpu thread that submitted. An embedder that routes host signals to guest threads usually leaves guest-relevant signals unblocked on vcpu threads. A process-directed signal can then land on the compile worker, which runs no guest code. The embedder also cannot set the worker's affinity, priority or name, or enter it in the thread registry it keeps for its own stop-the-world.

3. SPAWN FAILURE PANICS. `.expect("spawn tier-up worker")` (lib.rs:2368) panics whichever vcpu thread happened to submit when thread creation fails, for example at a thread-count limit. The alternative already exists: `TierUpSubmit::Unsupported`, the path a racing `Drop` takes at lib.rs:2875-2877, leaves the block interpreted, which is slow but correct.

Add:
- `quiesce()`. Bring the worker to a park point between requests, where it holds none of the locks above, and refuse new submissions until resumed. A compile in flight finishes rather than being interrupted. That bounds the wait to one compile, and Cranelift cannot be stopped safely partway.
- `resume()`, for the parent after the fork.
- `reset_after_fork()`, for the child:
  - forget the dead worker's handle without joining it;
  - rebuild the queue, condvars and done list, and zero `outstanding`;
  - drop queued requests, which re-submit on the next hotness trigger;
  - keep the module and every existing translation.
  The next tier-up spawns a new worker.
- A spawner hook. The embedder supplies the function that creates the worker thread, given the closure to run, so it controls the signal mask, affinity, scheduling class, name and registration. When the spawner fails, the tier-up is declined instead of panicking.

The three lifecycle calls are shaped to be registered by the embedder with `pthread_atfork` (prepare = quiesce, parent = resume, child = reset_after_fork). The library does not register atfork handlers itself, because the embedder decides whether engine state survives a fork at all.

Trade-off: quiesce waits for the compile in flight instead of abandoning it. That costs up to one compile of latency at fork time and keeps the worker simple. The process-global locks (codemap, perf map) are covered for free, provided the park point is outside `compile_with`.

Existing work: the deferred.md entry "fork / execve from a threaded process" is about the process model of an embedder (unemulinux returns `-EAGAIN` to a threaded guest's fork). That is the embedder's half, not this hook, and README:114 describes the same embedder-side gap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The API exposes quiesce, resume and reset_after_fork, documented as the prepare/parent/child triple for `pthread_atfork`
- [ ] #2 Test in x86jit-tests/tests/bg_tier.rs: with background tier-up active, call quiesce and then `libc::fork`. The child calls reset_after_fork, runs a hot loop until a block tiers up in the background, and exits 0. The parent resumes, keeps tiering up, and checks the child's exit status. The child runs under an alarm, so a deadlock fails the test instead of hanging it. The test passes on x86-64 and on an aarch64 host
- [ ] #3 A fork requested while the worker is inside a compile (held there by a test lever) makes quiesce wait for that compile. A negative control without quiesce deadlocks the child, and that is recorded on this task
- [ ] #4 An embedder-supplied spawner creates the worker. A test spawns it with all signals blocked and reads that mask from inside the worker
- [ ] #5 A failing spawner makes `tier_up_async` decline, the block stays interpreted, and the run completes with the correct result. No panic
- [ ] #6 `Drop` in a child that called reset_after_fork does not join the dead worker's handle
- [ ] #7 The dispatch path costs nothing extra: bench/baseline.json is unchanged within its noise band
- [ ] #8 spec.md (§11 or §17) documents the fork contract, and README:114 says what the engine now provides and what stays the embedder's job
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 cargo nextest run (--features unicorn) green, minus fuzz_robustness
- [ ] #2 cargo clippy --all-targets --all-features -- -D warnings clean
- [ ] #3 cargo fmt --check clean (nix-pinned rustfmt)
<!-- DOD:END -->
