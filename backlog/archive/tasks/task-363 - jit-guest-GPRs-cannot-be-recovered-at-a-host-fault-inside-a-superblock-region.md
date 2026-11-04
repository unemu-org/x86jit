---
id: TASK-363
title: >-
  jit: guest GPRs cannot be recovered at a host fault inside a superblock
  region
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - fault-precision
  - jit
dependencies:
  - TASK-357
  - TASK-356
priority: high
ordinal: 399000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Region mode keeps the 16 guest GPRs in Cranelift SSA Variables. They are loaded from CpuState once, in the entry block (codegen/mod.rs:322-335), and `write_gpr` only redefines the Variable without storing it (codegen/mod.rs:2782-2783). Values reach CpuState only through `flush_gprs` (codegen/mod.rs:2791-2805). That runs in `ret()` (codegen/mod.rs:3805-3811), which covers every exit including the bounds-check fault block, and before each helper call that reads CpuState (codegen/integer.rs:295, 320, 332; codegen/memory.rs:111; codegen/vector.rs:138 and others). No flush precedes an inline guest load or store. When such an access faults on a host guard page, the fault happens in the middle of compiled code and `ret()` never runs. CpuState then holds the GPRs from region entry or from the last helper call.

decision-5 records this ("GPRs exact in single-block mode, may be stale in region mode") and defers it until a guest signal frame needs it. decision-6 keeps single-block mode write-through partly to preserve that precision. An embedder that delivers real POSIX signals needs exact GPRs now. The handler receives a ucontext it may read and edit, and a managed runtime turns a fault on a null reference into a language exception by reading the base register and rewriting RIP and RSP. A stale register there means the wrong exception, or a corrupted resume, with no error anywhere. RFLAGS has the same requirement and is TASK-356; this task covers the GPRs.

Superblocks are opt-in (`JitBackend::with_superblocks`, lib.rs:2103-2108), so leaving them off is the zero-cost answer that already exists. The question is whether region performance can be kept together with exact state. Price each candidate on the bench:

(a) Per-trap-site register locations from Cranelift: for every trap site TASK-357 creates, where each guest GPR lives (a register or a spill slot). Cranelift's value-label ranges are debug info and explicitly best effort (cranelift-codegen 0.115.1, ir/dfg.rs: "Debug info is best-effort"). This option is acceptable only if completeness is proven for every site. The test below is that proof, and a site without full coverage must fall back to (b).
(b) Flush before every access that can fault: store the GPR Variables written since the last flush point in front of each guest access, the same pattern helper calls already use. This is exact by construction. Its cost is stores in the loop bodies that region mode exists to keep in registers.
(c) Declare the limit: a backend option, or a documented rule, that an embedder needing exact state at host faults does not enable regions, plus a debug assertion that the two are not combined.

Whatever is chosen is enforced by a test, not by a comment.

Until TASK-364 provides the landing pad, the test can catch the host fault with a fork-isolated test-only handler, the pattern the native oracle already uses (x86jit-tests/src/native.rs). The handler reads the per-site record and reconstructs the state without resuming.

RELATION. decision-5 records the gap as deferred, and decision-6 keeps single-block mode write-through. No existing task covers it. It needs the trap sites of TASK-357 and the exact RFLAGS of TASK-356. TASK-364 depends on it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test compiles regions over loops that contain guest loads, stores, locked RMWs and vector accesses. It forces a host fault at each guest access in turn (a guard page under that access's address, one at a time) and compares the recovered GPRs and RFLAGS with the interpreter's state at the same instruction. It fails at 96fd5c6 for at least one access
- [ ] #2 The test runs on x86-64 and on aarch64 (the aarch64 CI lane, real hardware)
- [ ] #3 The chosen strategy is measured against bench/baseline.json with superblocks on, next to the number with superblocks off, and both are recorded on this task
- [ ] #4 If (a) is chosen, a site whose register locations are incomplete is detected at compile time and falls back to flushing, and a test forces that path
- [ ] #5 decision-5's "Precision recorded" paragraph and decision-6's consequence about region staleness are amended with a date and the new precision
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
