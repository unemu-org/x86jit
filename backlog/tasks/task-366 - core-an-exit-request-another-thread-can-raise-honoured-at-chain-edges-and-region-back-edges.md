---
id: TASK-366
title: >-
  core: an exit request another thread can raise, honoured at chain edges and
  region back-edges
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - embedder-api
  - threads
dependencies: []
priority: high
ordinal: 402000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A vcpu returns from `Vcpu::run` only through an exit of its own (syscall, fault, `hlt`, ...) or when the block budget passed to `run` is spent (vm.rs:996-1012). Inside compiled code, the inner loop follows chain, link and IBTC edges and checks only `has_dirty_code` there (vm.rs:1175-1205; memory.rs:826-832). A region keeps its fuel in a register (codegen/mod.rs:337-343, 4054-4078). With no budget the region gets `u64::MAX` fuel (vm.rs:1200), so a loop inside it never returns. Nothing another thread can do stops that vcpu. `Vcpu::inject_irq` (vm.rs:705-719) does not help: it is real-mode IVT delivery, takes `&mut self`, and is consumed only at a `run` boundary.

An embedder running a multi-threaded native userspace needs exactly that stop:
- to deliver an asynchronous POSIX signal (`kill`, `pthread_kill`, a timer) to a thread that is spinning in guest code;
- for a stop-the-world garbage collector that suspends every thread (managed runtimes signal each thread and wait for it to park);
- to quiesce every vcpu before `fork`, or for a debugger.

A budget is no substitute. A small budget costs throughput on every thread all the time, and the request comes from outside, at times the vcpu cannot predict.

DESIGN. Give each vcpu a cloneable `Send + Sync` handle (an `Arc` around an atomic word) with a `request_exit()` method. The vcpu then returns a new `Exit` variant (e.g. `Exit::Interrupted`) at the next instruction boundary it reaches, with precise state and nothing half-done, as `BudgetExhausted` leaves it (exit.rs:56-57). The request is checked:
- in the outer dispatch loop, and in the compiled inner loop at every chain, link and IBTC edge, next to the existing `has_dirty_code` load. Folding both into one word keeps the edge at one load;
- in compiled regions at the per-sub-block fuel gate: one load through a pointer carried in MemCtx (a new field appended at the end, since the layout is append-only, jit_abi.rs:60 and 209-224). A loop with no other exit is then checked on its back-edge;
- in the interpreter loop, once per block;
- in long-running helpers. A `rep movs`/`stos` with a huge count may be interrupted between iterations, with RCX/RSI/RDI updated and RIP left on the instruction (SDM Vol 2B, REP/REPE/REPZ/REPNE/REPNZ: a repeating string operation can be suspended by an exception or interrupt and resumed).

The latency bound is then one block, one region sub-block, or one bounded slice of a string operation. State it in the API docs. Adding an `Exit` variant breaks the public enum, which is acceptable before 1.0 (README "API stability").

RELATION. deferred.md "IRQ injection and icount virtual time" (task-144 closed into it) is about delivering interrupts into the guest through its own interrupt table. This task is an exit to the embedder and needs neither. TASK-327 touches the same inner loop. TASK-382 (quiesce the compile worker around `fork`) is the backend-side counterpart for fork.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One vcpu runs a multi-block loop with no budget while another thread calls `request_exit()`, in three shapes: a superblock region, chained single blocks, and the interpreter. `run` returns `Exit::Interrupted` within a bounded time, and RIP, the GPRs and RFLAGS equal the interpreter's state at that instruction boundary. At 96fd5c6 the test times out
- [ ] #2 Calling `run` again after `Exit::Interrupted` reaches the same final state as an uninterrupted run
- [ ] #3 A `rep stosb` over a large count is interrupted part-way, with consistent RCX/RDI and RIP on the instruction, and completes correctly on resume, on both tiers
- [ ] #4 The documented semantics cover a request raised while the vcpu is outside `run` (honoured at the start of the next `run`, never lost) and a request raised twice (one exit), and a test pins each
- [ ] #5 The cost is measured against bench/baseline.json on chained and region-heavy workloads and recorded on this task. The hot path gains at most one load per chain edge and one per region sub-block
- [ ] #6 #1 passes on the aarch64 lane, and the memory orderings used for the request word are stated in the code
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
