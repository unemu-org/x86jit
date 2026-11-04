---
id: TASK-356
title: >-
  lift: dead-flag elision leaves RFLAGS stale at a fault inside a block and
  drops flags that lahf reads
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - fault-precision
  - flags
dependencies: []
priority: high
ordinal: 392000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
READ from the code at 96fd5c6, not yet executed: the first acceptance criterion is the probe.

`elide_dead_flags` (lift/mod.rs:867-897) narrows every ALU op's `FlagMask` to the flags still live, walking the block backwards from "all six live at the block end" (lift/mod.rs:878). Liveness comes only from `op_reads` (lift/mod.rs:774-784), which knows `Branch`/`GetCond` (a condition code) and `Adc`/`Sbb`/`Rcl`/`Rcr` (carry-in), and nothing else. Both tiers execute the narrowed IR: the interpreter's `apply` writes only the masked flags (called from interp/integer.rs:31-44, defined at interp/mod.rs:6624), and the JIT's `store_flags` does the same (codegen/mod.rs:2315). So `jit == interp` cannot see either defect below, and the differential corpus ends at `hlt`, where all flags are live.

A. RFLAGS AT A FAULT INSIDE A BLOCK. An op that can leave the block in the middle is not a flag reader to the walk: a guest load or store (UnmappedMemory, MMIO defer), a helper with a fault path (string, x87, fxsave), a divide (#DE), a waiting x87 op (#MF). In

    add rax, rbx      ; mask narrowed to nothing: the cmp overwrites all six
    mov rcx, [rdx]    ; faults
    cmp rcx, 0

the exit leaves RFLAGS as they were before the `add`. A fault must report the machine state as it was before the faulting instruction began, with every earlier instruction retired (SDM Vol 3A, chapter "Interrupt and Exception Handling", section "Exception Classifications": the processor restores the state prior to the beginning of execution of the faulting instruction; AMD APM Vol 2 chapter 8). The retired `add` is part of that state. An embedder that runs a native userspace turns this exit into a POSIX signal, and the handler gets a ucontext whose RFLAGS the program never produced. Handlers that resume are common: a managed runtime's null-check or safepoint handler, a user-space pager that maps the page and returns. They continue with wrong flags, silently. The elision is per lifted block, so single blocks and regions are affected alike.

B. FLAG-IMAGE READERS MISSING FROM `op_reads`. `IrOp::Lahf` is lifted in every mode (lift/mod.rs:2816-2824) and documented as reading the flags (ir.rs:2415-2418), but `op_reads` returns 0 for it, and the comment above `op_reads` still says no `lahf`/`sahf`/`pushf` exists. In `add; lahf; cmp` within one block, AH is built from flags older than the `add` (SDM Vol 2A, LAHF). Checking this turned up two more readers with the same omission, both Real16-only: `IrOp::PushfReal` (lift/mod.rs:2722-2725) pushes the FLAGS image, and `IrOp::Bcd` for `daa`/`das`/`aaa`/`aas` (lift/mod.rs:2755-2770) reads CF and AF (interp/integer.rs:718-744). So the textbook `add al, bl; daa`, followed in the same block by any instruction that writes all flags, adjusts AL with stale CF/AF. The comment at lift/mod.rs:755-756 ("only `daa`/`aaa`, which the lift doesn't cover") is stale for the same reason. `cmc` (`SetCf { value: None }`, lift/mod.rs:2746-2749) also reads CF, but its only output is CF, so it matters only through case A.

DIRECTION. Case B is a completeness fix. Add `Lahf`, `PushfReal`, the flag-reading `Bcd` kinds and `SetCf { value: None }` to `op_reads`, and make the match exhaustive, so a new reader (the long-mode `pushfq` TASK-370 adds) cannot be left out without a compile error.

Case A involves a cost trade-off. Decide it with numbers:
1. Treat every op that can exit (any guest access, any helper with a fault path, `Div`, the x87 waiting ops) as reading all six flags. This is simple and obviously right. But most ALU ops are followed by a memory access inside the same block, so the cost approaches `X86JIT_NO_FLAG_ELISION` (lift/mod.rs:859-865), which is the upper bound to price.
2. Keep the narrowing on the hot path and materialise the elided flags only on the exit path. The JIT's bounds-check fault block (codegen/mod.rs:2457-2498) is cold and dominated by the operands of the pending flag writer, so it can compute and store the flags there. The interpreter can ignore the narrowed mask and always compute the full flags, which costs it little and changes nothing observable except making mid-block exits right. A host fault on a guard page has no cold block, so that half needs per-site data from TASK-357 or the flush strategy TASK-363 chooses.

RELATION. TASK-305 (done) set the contract that a trapped instruction leaves precise state, but it covered destinations and memory, not RFLAGS; this extends it. TASK-354 (done) only moved the environment-variable read in the same function and does not overlap. TASK-363 and TASK-364 depend on this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test in x86jit-tests runs `add; <faulting load>; cmp`, `add; <faulting store>; cmp` and `add; div <zero>; cmp` on both backends. It asserts that RFLAGS at the exit equal the flags the `add` produced. It fails at 96fd5c6 and passes after the fix
- [ ] #2 The expected RFLAGS in #1 come from a real CPU, for all three cases. Today the native oracle snapshots the register file only for the terminating `hlt` and records only RIP for any other signal (x86jit-tests/src/native.rs:169-195). It must capture the full register file for a guest fault signal other than `hlt` too: at least SIGSEGV from a page fault on a designated address and SIGFPE from the #DE of `div <zero>` (SIGFPE is already among the installed handlers, native.rs:639-646)
- [ ] #3 `add; lahf; cmp` in Long64 and Compat32 matches the native oracle on both backends. In Real16, `add al, bl; daa; cmp` and `add ax, bx; pushf; cmp` give the AL and the pushed FLAGS word the SDM pseudocode defines. Each of these fails before the fix
- [ ] #4 `op_reads` is an exhaustive match over `IrOp` with no `_ =>` arm, and the stale comments at lift/mod.rs:755-756 and 774-777 say what is true
- [ ] #5 The cost of the case-A strategy is measured against bench/baseline.json, next to the `X86JIT_NO_FLAG_ELISION=1` upper bound, and both numbers are recorded on this task
- [ ] #6 A region (`JitBackend::with_superblocks`) whose fault is in a non-entry sub-block reports the same RFLAGS as the single-block JIT and the interpreter
- [ ] #7 spec.md §3.2 states that the flags at every exit, a mid-block fault included, are the architectural flags after the last retired instruction
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
