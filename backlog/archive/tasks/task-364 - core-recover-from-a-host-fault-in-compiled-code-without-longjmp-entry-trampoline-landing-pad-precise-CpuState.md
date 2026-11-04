---
id: TASK-364
title: >-
  core: recover from a host fault in compiled code without longjmp: entry
  trampoline, landing pad, precise CpuState
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - fault-precision
  - embedder-api
dependencies:
  - TASK-357
  - TASK-363
priority: high
ordinal: 400000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
decision-5 GP-2/GP-3 make a host fault in compiled code resumable like this: the embedder's SIGSEGV handler `siglongjmp`s back to a `guarded_run` wrapper that binds glibc's `__sigsetjmp`, and the faulting PC is looked up in `codemap` (codemap.rs:1-18). That contract has two problems, and no test can see either of them:

- Rust has no way to declare that a function returns twice. To the optimizer, a `sigsetjmp` called from Rust is an ordinary call that returns once, so values it keeps in registers across that call are not guaranteed on the second return.
- `siglongjmp` across Rust frames is defined only if every skipped frame is a plain old frame with nothing to drop (Rust RFC 2945, "C-unwind ABI"). The frames between the wrapper and compiled code are `Vcpu::run`/`run_inner` (vm.rs:996-1205), `call_block` (jit_abi.rs:356-362) and any Rust helper the compiled code called (string, x87, fxsave). The contract holds only while none of them, now or after any future edit, owns something with a destructor across the call, and nothing checks that. It is also glibc-only (decision-5, "Consequences").

The lookup it relies on is a linear scan over every compile ever made (codemap.rs:113-136, TASK-352 AC#3), with RIPs truncated to 32 bits (TASK-351).

`call_block` is a plain Rust call through the compiled function pointer (jit_abi.rs:356-362, called at vm.rs:1200-1205). Replace the longjmp path with what an OS kernel does for faults in its own code: rewrite the signal context so the handler returns into a landing pad.

1. ENTRY TRAMPOLINE, per host (x86-64 and aarch64, `global_asm!`, no new dependency). It saves the callee-saved registers, SP and FP into an entry record the classifier can find for this vcpu thread, then calls the compiled code. Normal returns pass back through it unchanged.
2. CLASSIFIER: a core function that is async-signal-safe (no lock, no allocation, no thread-local the core cannot prove safe in a handler). It takes the faulting host PC and the host register snapshot from the ucontext, as plain integers, and returns one of two answers. NOT OURS: the embedder chains to the previous disposition and the process crashes honestly, as GP-2 does now. OURS: the answer carries (i) the context rewrite — PC set to the trampoline's landing pad, SP, FP and the callee-saved registers restored from the entry record — and (ii) the precise guest state, written into the vcpu's CpuState through its MemCtx. That state is the RIP from the per-site record (TASK-357), the GPRs per TASK-363, RFLAGS per TASK-356, and the fault address, direction and width.
3. LANDING PAD: returns a fault code to the dispatcher the same way a compiled block's own `RET_UNMAPPED` does, so `run_inner` builds a resumable `Exit::UnmappedMemory` with nothing new on its side.

The signal handler itself stays in the embedder: `sigaction`, the alternate signal stack, and reading and writing the OS-specific ucontext. x86jit-core keeps exactly `{iced-x86}` (x86jit-tests/tests/boundary.rs). Because the core API takes and returns plain integers, it serves Linux, the BSDs and macOS alike.

The main design choice is where the entry record lives. The embedder already keeps a per-thread pointer to its vcpu and can pass it to the classifier, which spares the core from depending on thread-local storage inside a signal handler.

Cost: the inner loop calls `call_block` on every chain edge (vm.rs:1204-1214), so the trampoline runs once per compiled block executed, not once per `run`. Measure it.

Faults inside Rust helpers are out of scope here. They need the fixup-table copy routines of TASK-365, which reuses this classifier.

RELATION. The lookup needs TASK-351 (64-bit RIP) and TASK-352 AC#3 (no linear scan in the handler). This task supersedes decision-5's recovery mechanism; the guard pages themselves stay. TASK-372 (guest MXCSR modes in compiled code) needs the landing pad to restore the host FP control register (AC#7).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test in x86jit-tests installs a test-only SIGSEGV handler, fork-isolated like src/native.rs, and runs compiled code that faults on a guard page, in single-block and in region mode. Execution resumes through the landing pad to an `Exit::UnmappedMemory` whose addr, access and RIP match the interpreter's. The test then maps the page and runs on to a final state equal to the interpreter's
- [ ] #2 #1 passes on Linux x86-64 and on Linux aarch64 (the aarch64 CI lane, real hardware)
- [ ] #3 The classifier is documented and tested as async-signal-safe: no allocation, no lock, bounded time (no scan over all compiles), 64-bit guest RIPs
- [ ] #4 A fault at a PC outside every compiled range and every trampoline is reported as not ours, and a subprocess test pins that a genuine host bug still crashes with a core dump
- [ ] #5 The cost of the trampoline on chained workloads is measured against bench/baseline.json and recorded on this task
- [ ] #6 A dated decision supersedes decision-5's GP-2 longjmp recovery. It states the new embedder contract: what the handler passes in, and what it writes back into the ucontext. x86jit-core's dependency set stays exactly `{iced-x86}`
- [ ] #7 The landing pad restores the host floating-point control register (MXCSR on x86-64, FPCR on aarch64) to its pre-`run()` value before any Rust runs, because TASK-372 lets compiled code run in a non-default guest FP mode. If TASK-372 has landed, #1's test also faults inside a non-default-mode block and checks the register after the exit. If it has not, record on this task that TASK-372's AC#9 adds the case
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
