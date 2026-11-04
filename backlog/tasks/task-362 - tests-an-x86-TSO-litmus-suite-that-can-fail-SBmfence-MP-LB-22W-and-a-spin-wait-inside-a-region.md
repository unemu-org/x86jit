---
id: TASK-362
title: >-
  tests: an x86-TSO litmus suite that can fail: SB+mfence, MP, LB, 2+2W and a
  spin-wait inside a region
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - memory-order
  - tests
dependencies:
  - TASK-355
priority: high
ordinal: 398000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`x86jit-tests/tests/tso.rs` is the only weak-memory test. It runs one shape, message passing (MP):
- on `JitBackend::new()` (tso.rs:109), which forms no superblocks (`new()` vs `with_superblocks`, x86jit-cranelift/src/lib.rs:2094-2108);
- never on the interpreter;
- with hard assertions for `AcqRel` and `FullTso` only. `Fast`'s count is printed, not asserted (tso.rs:154-162).

On an x86 host every tier compiles to identical code, so those assertions test the host CPU. status.md:165 records that no reorder ever showed on the virtualised ARM runner, which makes the test "a liveness/correctness check rather than a demonstration". The suite cannot fail for the reasons that matter. That is how the fence defect (TASK-355) stayed invisible through every green run.

WHAT IS MISSING.

1. Store buffering (SB). This is the one reordering x86-TSO allows (SDM Vol 3A §11.2.3.4). Without plain SB the suite never shows that it can observe a reordering at all. Without SB+mfence and SB+locked-op it cannot see TASK-355.
2. The other forbidden shapes:
   - LB: loads are not reordered with later stores (§11.2.3.3).
   - 2+2W: stores are not reordered with other stores (§11.2.3.2). It is checked on the final memory state after both threads finish.
   - IRIW: other processors see stores in a consistent order (§11.2.3.7). It needs four threads; add it if the harness generalises cheaply.

   Section numbers follow the SDM edition this repository already cites; older editions number the chapter 8.
3. The interpreter. It ignores the tier completely. A Store's `order` is discarded (interp/mod.rs:385-391) and ordinary accesses are Relaxed host atomics (memory.rs:1500-1543). With hotness-gated tier-up every block starts interpreted, so a tier promise that holds only for compiled code does not hold for the guest. The suite has to show what the interpreter does under each tier on aarch64, not assume it.
4. Regions and compiler reordering. Inside a region the guest GPRs and the fuel counter are SSA Variables (codegen/mod.rs:322-343). A Vm with tier-up builds its module at `OptLevel::Speed` (`OptLevel::for_tiering`, lib.rs:1944-1950). Take a spin-wait such as `1: mov rax,[flag]; test rax,rax; jz 1b`, with no store inside the loop: redundant-load elimination may merge the load. With `budget = None` the region's fuel is `u64::MAX` (vm.rs:1200-1201), so a merged load spins forever instead of falling back to the dispatcher and reloading. The spin must terminate in EVERY tier. x86 guarantees that a store eventually becomes visible, and so does the host, so a hang is a compiler defect, not a memory-model choice.
5. SB with a locked op, on aarch64. A guest locked op becomes a host atomic: Cranelift `atomic_rmw`/`atomic_cas` in the JIT (codegen/memory.rs:21-91), SeqCst `fetch_*`/`compare_exchange` in the interpreter (memory.rs:1412-1487). x86 orders every load and store around a locked instruction (§11.2.3.9). On aarch64, whether an acquire-release atomic (an LSE `...AL` op, or an exclusive-pair loop) also orders the plain accesses on either side of it the way x86 does is exactly what SB+locked tests. Do not assume the answer, for either backend.

THE MATRIX. Backends: interpreter, JIT single-block, and JIT with regions (`with_superblocks` and a tier-up policy, so the module is at `Speed`). Tiers: `Fast`, `AcqRel`, `FullTso`. Hosts: x86-64 and aarch64. Assert what each tier promises (spec.md §8.2.3; vm.rs:226-243):
- every tier, `Fast` included: SB+mfence and SB+locked never show the relaxed outcome, and the spin-wait terminates;
- `AcqRel` and `FullTso`: additionally, MP, LB and 2+2W never show the relaxed outcome;
- `Fast`: MP, LB and 2+2W counts are reported, not asserted;
- plain SB: the relaxed count is reported in every cell. A separate sanity check asserts it is non-zero on a host where it is reliably observable (the x86-host JIT leg is the obvious one), so a forbidden-outcome pass is known to mean something.

THE HARNESS. Generalise tso.rs's lockstep design (start barrier, per-cell arrays, threads marching through the cells so they keep meeting on the same wavefront, tso.rs:85-99) to N threads and per-shape guest code. Read the results back from guest memory. Choose iteration counts by measurement, with each cell finishing well inside the nextest timeout on the aarch64 runner.

FOR EMBEDDERS. An embedder that declares that its host runs a TSO memory model (TASK-377) has only one way to back the claim: run this suite on its target. Expose the suite as a library entry point, in `x86jit-tests` src or a small public module. It takes a Vm/backend factory and returns per-shape, per-tier counts, so an embedder can run it at start-up or in its own CI and refuse host-TSO mode if a forbidden outcome appears.

RELATION. Depends on TASK-355. Until that lands, SB+mfence is expected to fail:
- in the interpreter on every host;
- in the JIT on an x86 host in every tier;
- on aarch64 under `Fast` and `AcqRel`;
- and to pass only on aarch64 `FullTso` compiled code, which hides the defect.

Seeing that failure once, before the fix, is the evidence that the suite can fail. task-80 (done) wrote tso.rs. status.md:165 records the null result on the virtualised ARM runner. TASK-377 uses this suite as its check. TASK-361 makes the aarch64 half run on every push.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Litmus tests SB, SB+mfence, SB+locked (xchg and lock add), MP, LB and 2+2W (IRIW if cheap) exist in x86jit-tests. Each runs on the interpreter, the JIT single-block and the JIT with regions, under Fast, AcqRel and FullTso
- [ ] #2 Assertions follow the per-tier promises above. Fast's MP/LB/2+2W counts and every cell's plain-SB count are printed in one table per run
- [ ] #3 SB+mfence is shown FAILING before TASK-355 on an x86 host, with counts recorded in this task's notes, and passing after it; plain SB shows a non-zero relaxed count on that host
- [ ] #4 A spin-wait on a plain load inside a compiled loop region at OptLevel::Speed with budget None terminates when another thread releases it, on both hosts. A negative control (e.g. a Cranelift IR dump showing the load stays inside the loop, or a deliberately merged variant) shows the test would catch a merged load
- [ ] #5 The suite runs on the aarch64 runner; relaxed-outcome counts per cell are recorded in this task's notes, and the MP-under-Fast result (observed, or absent with the reason) is recorded
- [ ] #6 Interpreter cells that violate a tier's promise on aarch64 are kept as expected failures that name the task that owns the fix (TASK-377 for the tier, TASK-355 for fences), never deleted or weakened
- [ ] #7 The suite is callable outside the test binary through a documented entry point that returns per-shape counts; spec.md §8.2.3 points at it
- [ ] #8 Each cell's run time is measured and the whole suite fits the existing nextest budget on both runners
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
