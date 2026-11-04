---
id: TASK-390
title: 'core: an API-driven interpreter-vs-JIT whole-state lockstep mode'
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - tests
  - oracle
dependencies:
  - TASK-360
priority: low
ordinal: 426000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The existing lockstep (x86jit-core/src/lockstep.rs:1-20) is a forensic tracer for bugs where the interpreter disagrees with hardware:
- It is switched on by the environment (`X86JIT_LOCKSTEP=<path>`, plus window and cap variables; lockstep.rs:335-366).
- It runs in the interpreter only.
- It captures vector instructions, with ymm0-15 before and after (`NVEC`, lockstep.rs:32), for replay on a native x86 host (doc-24).
It cannot check the JIT, and replay needs an x86 host.

An aarch64 host has no native x86 oracle. Unicorn exists in the test crate, not inside a running embedder. For a real workload, the only differential check left is the engine against itself: run each unit both compiled and interpreted from the same pre-state, and compare. That is how bugs that exist only in the AArch64 lowering surface in the workload that triggers them, rather than in a hand-written snippet. Examples are the host NaN and signed-zero rules (TASK-381), ordering lowering (TASK-377) and flag materialisation (TASK-356).

Add a `Vm`/`Vcpu` mode, set through the API rather than the environment (TASK-360). At the dispatch of a compiled block or region (vm.rs:1175-1205):
1. Snapshot `CpuState`, which is `Clone` (state.rs:312-313).
2. Run the interpreter over the same IR from the snapshot. For a region, run it block by block until it reaches the RIP where the compiled region exited.
3. Run the compiled code.
4. Compare the following, and report the first divergence: block or region guest address, the field, both values, and the unit's IR:
   - GPRs, RIP and RFLAGS;
   - all vector state that exists (xmm/ymm/zmm, k registers);
   - x87 state (stack, control, status, tag);
   - MXCSR, once TASK-358 makes it state;
   - the `Exit`;
   - the memory written.
5. Then stop, or continue with the interpreter's result, as configured.

RFLAGS is compared exactly, with no undefined-flag waiver. Both tiers execute the same IR, and codegen promises that it computes flags exactly as the interpreter does (x86jit-cranelift/src/codegen/mod.rs:1-4). A waiver here would hide a real bug, unlike the hardware-oracle comparisons, where x86jit-tests/src/compare.rs:134-138 masks undefined flags for a good reason.

MEMORY. The interpreter accesses memory through `Memory`'s methods, so a journalling layer can record its writes and roll them back before the compiled run. The compiled run's writes are inline stores, and they are visible only where a store reports itself. The watch mechanism does exactly that for watched pages (`MemCtx.watch_count_ptr`, jit_abi.rs:102-109; `Memory::watch_range`, memory.rs:889). So lockstep can either watch the pages the unit may touch, or compare whole touched pages after both runs. Choose one. The trade-off is catching a stray compiled store to an unrelated page against the cost per block.

MULTI-THREAD LIMITS. The comparison means something only if no other thread writes the memory the unit reads between the two runs. Document it: run single-vcpu, or with the world stopped (the embedder's job). A result taken under concurrent writers is not evidence. MMIO and port-I/O accesses already defer to the interpreter, so a unit that reaches one is compared up to the deferral.

The same hook later checks an AOT unit (TASK-385) against a fresh JIT compile of the same range.

Cost: zero when off (one branch per dispatch, measured). When on, it is a debugging mode and may be slow.

Existing work:
- doc-24 is the interpreter-against-hardware tracer, a different oracle: it trusts the CPU, and this mode trusts the interpreter.
- task-346 covers blind spots of `jit_eq_interp` in the test harness. This mode complements it on real workloads.
- The comparison logic in x86jit-tests/src/compare.rs (the `Divergence` report) is in the test crate today. Move the shared part rather than writing a second comparator.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The `Vm`/`Vcpu` API enables lockstep with a report callback. Nothing reads the environment
- [ ] #2 A deliberately broken lowering (a test-only fault injection that corrupts one register, one flag, one vector lane or one stored byte) is caught at the first diverging block, with its guest address and the field. The negative control is recorded on this task
- [ ] #3 The comparison covers GPRs, RIP, RFLAGS (exact), vector and mask registers, x87 state, the exit kind and written memory, and MXCSR once TASK-358 lands
- [ ] #4 The jit.rs and superblock.rs suites run under lockstep with no divergence on x86-64 and on an aarch64 host, and the aarch64 run is recorded
- [ ] #5 Off, it costs nothing measurable against bench/baseline.json
- [ ] #6 The multi-thread limits are documented in testing.md or spec.md §13, and doc-24 points to the new mode
- [ ] #7 The env-gated tracer keeps working, behind TASK-360's feature
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
