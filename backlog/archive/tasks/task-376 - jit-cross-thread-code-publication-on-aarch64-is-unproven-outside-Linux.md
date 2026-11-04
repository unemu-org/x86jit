---
id: TASK-376
title: 'jit: cross-thread code publication on aarch64 is unproven outside Linux'
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - memory-order
  - aarch64
dependencies: []
priority: medium
ordinal: 412000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Translated code is regularly written on one core and executed on another:
- Background tier-up compiles on the `x86jit-tier` worker thread (`ensure_worker`, x86jit-cranelift/src/lib.rs:2358-2371; `worker_loop`, :2741). A vcpu thread's dispatcher then publishes the result into the cache and enters it (`drain_tier_up`, x86jit-core/src/vm.rs:1393-1399; decision-4). EVERY background compile crosses threads.
- A foreground compile finalizes on the vcpu thread itself (`finalize_definitions` / `get_finalized_function`, lib.rs:2709-2711). Any other vcpu that later reaches the block through the cache, a chain or link slot, or the IBTC executes code another core wrote.

THE DATA SIDE IS FINE, AND SHOULD SAY SO. The cache is lock-protected, and link slots are `AtomicU64` (lib.rs:2079-2091). Compiled code loads a slot and branches indirectly, so no instruction byte is patched after finalize. Publishing a slot is a data write. That guarantee is currently implicit; state it where the slots are declared.

THE INSTRUCTION SIDE ON AARCH64. Two steps are required. First, the writer cleans the data cache to the point of unification, invalidates the instruction cache for the range, and completes both with DSB. Second, the core that EXECUTES the code must go through a context synchronization event (ISB, or an exception entry or return) after that maintenance and before it runs the new instructions. Without it, the core may execute instructions it fetched earlier. See the Arm ARM (DDI 0487), "Concurrent modification and execution of instructions" and the instruction-cache maintenance requirements.

x86jit hands the writer side to cranelift-jit 0.115.1, which delegates to `wasmtime-jit-icache-coherence` 28.0.1 (Cargo.lock:745-747). aot-plan.md:57-58 already names that crate's `clear_cache` + `pipeline_flush_mt()` as required on ARM64. On Linux the cross-core part is expected to be a process-wide membarrier (SYNC_CORE). Not verified here:
- whether cranelift-jit calls `pipeline_flush_mt` on every finalize (the crate source was not opened for this task);
- what the crate does on macOS and on the BSDs;
- what happens when the syscall is missing (an older kernel) or filtered (a seccomp sandbox).

The aarch64 CI runner is Linux only (`.github/workflows/ci.yml`, matrix), so every green aarch64 run covers the Linux path and nothing else. `bg_tier.rs` exercises worker-to-vcpu publication with one vcpu and `wait_idle()` for determinism. Nothing stresses many vcpus entering freshly published code while the worker keeps writing.

WHY AN EMBEDDER NEEDS IT. An embedder that runs a multithreaded native userspace on an aarch64 host keeps the background worker busy for minutes while every guest thread enters new code. A stale fetch is a silent wrong-instruction execution, a crash in the host process, or a guest crash that no differential test can reproduce. It has to be a guarantee, not a property of the host OS the CI happened to use.

DESIGN OPTIONS.

(a) Reader-side ISB. A global publish generation is bumped (Release) after each finalize. Each vcpu remembers the generation it last synchronized at and issues an ISB on its way into compiled code when it sees a newer one. Cost: a load and a compare per dispatcher entry, plus an ISB only after a publish. The catch is that a vcpu does not pass the dispatcher on chain, link-slot and IBTC transfers. If vcpu A fills a slot with code published after vcpu B's last ISB, B reaches the new code without one. So (a) also needs one of:
- the check on those edges (a cost per edge);
- a rule that a slot is filled only with code whose generation every running vcpu has synchronized past (a grace-period scheme);
- the writer-side guarantee below.

(b) Writer-side, process-wide synchronization after each finalize. That is membarrier SYNC_CORE on Linux and the equivalent on each other supported OS, verified per OS. If the facility is unavailable, `JitBackend` construction says so instead of running without it.

The task picks one and records it in a decision, with the per-OS table.

On x86 hosts. SDM Vol 3A §11.1.3 describes a serializing instruction on the executing processor as part of the cross-modifying-code protocol. Whether x86jit's fresh-address publication needs it is recorded as a question in the decision. The x86 path is not changed by this task.

RELATION.
- decision-4 is the publication protocol (data side).
- task-323 (done) is the GUEST's cross-modifying-code protocol, a different property.
- task-353 A (watch/mark ordering on aarch64) is the neighbouring aarch64 ordering question.
- TASK-383 (an owned code arena) takes over cache maintenance from cranelift-jit and must carry whatever this task decides.
- TASK-382 (worker lifecycle) touches the same worker.
- TASK-361 runs the aarch64 suite on every push.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A decision records how a core that executes freshly published code is context-synchronized, per supported host OS, covering entry through the dispatcher AND through chain, link-slot and IBTC transfers
- [ ] #2 The slot declaration documents that link and chain slots are data and that no instruction byte is patched after finalize
- [ ] #3 A stress test in x86jit-tests: the background worker compiles a stream of distinct blocks while several vcpu threads enter freshly published code through the dispatcher and through chain edges; guest results are checked against the interpreter; it runs on the aarch64 runner
- [ ] #4 The test is run on at least one non-Linux aarch64 host, or the decision records that none was available and which hosts remain unproven
- [ ] #5 If the mechanism needs an OS facility, its absence is detected when the JitBackend is built and reported with a clear error, not ignored; the unavailable path is tested where it can be simulated
- [ ] #6 The cost is measured: the bench's background-tier and region modes on aarch64 are recorded on this task; x86 timings are within noise against bench/baseline.json
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
