---
title: Allocation-free child step for shared-address-space children
date: '2026-11-09'
status: accepted
---

# Allocation-free child step for shared-address-space children

The frozen-tag SMC contract and prohibition on all shared engine atomics below
are amended by [decision-19](decision-19%20-%20Live-child-stores-use-an-atomic-SMC-bridge.md).

## Context

An embedder runs a guest that calls `rfork(RFPROC | RFMEM | ...)` (or `vfork`). The
child shares the parent's whole address space, including the engine's heap, translation
cache, code space and the parent's `Vm` and `Vcpu` objects, and runs guest code until it
calls exec or exits. Under RFMEM the parent's other threads keep running, so they keep
mutating the engine's `Memory` (the region list under its locks, the code-page and watch
bitmaps), the translation cache and the process-wide code map. The child must therefore
neither allocate nor take a lock nor touch any atomic or object the parent engine owns,
must not compile, and must not run destructors over parent-owned objects. It runs on a
host stack and TLS the embedder provides.

The interpreter's single-instruction path (`interp::step_one`) already decodes, lifts and
interprets one instruction with the same code every tier shares, but it allocates (the IR
op vector, boxed fault metadata), takes the code-page dirty-list lock on stores to code
pages, consults the fetch policy through a mutex, and reads `Memory`, which the parent's
other threads may be changing.

## Decision

Add a child execution mode, additive to the public API of `x86jit-core`:

1. **Snapshot in the parent.** Before forwarding the fork, the embedder calls
   `Vm::child_view(mode) -> ChildView`. The view is plain owned data: the guest base
   pointer and span, the `CpuMode`, a copy of the region table (kind, protection,
   routing), the fetch-policy verdicts the parent's policy gives for the regions it covers
   (so a fetch from a non-executable page faults in the child exactly as in the parent,
   without the child consulting the parent's policy), the guest CPU identity and feature
   policy, the `X87PointerModel` and MXCSR handling. It holds no reference into `Vm` or
   `Memory`. The snapshot marks which pages back translated code.
2. **Arena in the parent.** `ChildArena::new(capacity)` preallocates everything one step
   needs: an IR op buffer sized for one instruction, temporaries, a fault-record slot, the
   per-thread fault-recovery record, and a bounded written-page log. Nothing is allocated
   after creation. Several children may run at once, each with its own arena.
3. **The step.** `child::step(view: &ChildView, cpu: &mut CpuState, arena: &mut ChildArena)
   -> ChildStep` is a free function. It reuses the shared lift and interpret code,
   specialised by a const flag so that every allocation, lock, shared atomic, cache, code
   map, compiler, tier-up, telemetry and fetch-policy access is compiled out. It never
   compiles and never falls back to the engine. For every instruction it supports, the
   result equals `step_one`'s. `CpuState` carries the whole architectural state the child
   continues with, including state the child cannot operate on (FS/GS base, MXCSR, x87
   control, status, tag and registers, XCR0 and AVX state).
4. **Results.** `ChildStep` is one of: `Continue`; `Syscall` with the same evidence as
   `Vcpu::syscall_evidence()` (path `Interpreter`, the SYSCALL instruction's RIP and its
   length, so instruction RIP plus length equals the resume RIP); `Fault(ChildFault)`, an
   unboxed record with the decision-14 fields and an optional `RawHostFault`, converted to
   `FaultMetadata` later in the parent; `Refused(ChildRefusal)`, naming the instruction
   class (x87, FXSAVE/XSAVE, other helpers, MMIO or Trap region, store to a page that backs
   translated code, arena full, unsupported instruction); and the other plain exits. A
   refusal leaves `cpu` and guest memory unchanged.
5. **Self-modifying code.** A store to a page the snapshot marks as code is refused: logging
   it would leave a window in which the parent's other threads run stale translations of
   code the child already changed. Every other page the child writes goes into the arena's
   written-page log (a full log is a refusal). After the child execs or exits, the parent
   calls `Vm::invalidate_written_pages(&arena)`, which is safe against concurrent
   translation and execution, as the normal invalidation is, so that translations made
   meanwhile from pages the child changed are dropped.
6. **Host faults.** On an identity-mapped span a child access to an inaccessible page is a
   real host fault. The child mode reads and writes guest memory through dedicated fixup
   routines whose PC range is a static constant, and the embedder's child signal handler
   calls `child::classify(pc, regs, raw) -> bool`, which is async-signal-safe and
   reentrant per thread, uses no global other than that PC range and keeps its record in
   the arena.
7. **Scope.** The first scope is the integer, SSE and AVX subset the interpreter executes
   through inline IR, plus REP MOVS/STOS and, if cheap, the plain string instructions,
   because libc's memset and memcpy on a spawn path commonly use them. Other helpers (x87
   arithmetic, FXSAVE/XSAVE) are refused at first and added when the embedder's measured
   path to exec needs them.
8. **Proof.** Tests run the step under a test allocator that aborts on any allocation,
   poison the parent-owned `Memory`, cache and code-map objects to prove the step never
   touches them, and compare `child::step` with `step_one` over the supported corpus.

## Consequences

- No change to the normal execution paths; the mode costs nothing unless used.
- Embedders that fork must snapshot before the fork, give the child a stack, TLS, view,
  arena and copied `CpuState`, and invalidate the logged pages after the child is gone.
- The supported scope grows from refusals the embedder measures on real paths.
