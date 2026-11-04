---
id: TASK-387
title: >-
  core: a misaligned locked read-modify-write is not atomic, and the aarch64 JIT
  may fault on it
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - memory-order
  - atomics
dependencies:
  - TASK-368
priority: low
ordinal: 423000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
INTERPRETER. `Memory::atomic_rmw` (memory.rs:1338-1360) and `atomic_cas` (memory.rs:1362-1379) go through `atomic_rmw_raw` / `atomic_cas_raw`. These use a host atomic only when the host pointer is naturally aligned. Otherwise they do `plain_read`, apply the op, then `plain_write` (memory.rs:1440-1445 and 1471-1476). The doc comment accepts this: "the *value* is identical, only cross-thread atomicity is lost, which aligned guest atomics never hit" (memory.rs:1338-1343). task-73 called it rare. `from_backing` explains that host alignment tracks guest alignment only in practice (memory.rs:600-608).

JIT. `emit_atomic_rmw` (codegen/memory.rs:21-68) and `emit_atomic_cas` (codegen/memory.rs:70-91) pass the host address straight to Cranelift `atomic_rmw` / `atomic_cas` / `atomic_load`. They use `MemFlags::trusted()`, which declares the access aligned, and perform no alignment check. So the two backends diverge:
- On an x86 host the emitted locked instruction is atomic even when misaligned (a split lock), but slow. Hosts with split-lock detection enabled may log it, throttle the process, or kill it.
- On aarch64, exclusive-pair loops (LDXR/STXR) always need natural alignment. The LSE atomics need it too, except that FEAT_LSE2 lets an access through when it stays inside a 16-byte aligned granule. So a misaligned locked op in compiled code LIKELY takes a host alignment fault where the interpreter completes. That is a backend divergence, and until TASK-357 gives compiled accesses trap sites, it is a host fault the engine cannot attribute.

Not yet run on the aarch64 runner. Verifying it is the first step.

THE RULE. SDM Vol 3A §11.1.2.2 (Software Controlled Bus Locking): the integrity of a bus lock is not affected by the alignment of the memory field, and the LOCK semantics hold for as many bus cycles as the whole operand needs. (Section numbers follow the SDM edition this repository already cites; older editions number the chapter 8.) A misaligned `lock xadd`, `lock cmpxchg` or `xchg` with memory is atomic on hardware. Programs rarely do it on purpose. It does happen: counters inside packed structures, atomics on fields of serialized buffers, and code ported from x86-only codebases that never had to care.

WHY AN EMBEDDER NEEDS IT. An embedder that runs arbitrary native binaries does not choose the guest's alignment. A lost update is a heisenbug in guest code. A SIGBUS raised in the translator's own compiled code crashes the host process.

DESIGN DIRECTION.
- Aligned operands keep their current path. The JIT cannot know alignment at compile time, so a runtime test is needed: an `and` and a branch before each locked op, with the slow path in a cold block. A locked op already costs tens of cycles, so the test should be noise. Measure it.
- An operand inside one aligned 8-byte word: run a CAS loop on the containing word, merging the operand's bytes. This is atomic against EVERY other access to those bytes, plain stores included, because any concurrent write to the word fails the CAS.
- An operand inside one aligned 16-byte granule: a 16-byte CAS. That is `cmpxchg16b` on an x86 host and CASP or an LDXP/STXP loop on aarch64. It is the same host primitive TASK-368 needs for guest `cmpxchg16b`, hence the dependency. `x86jit-core` must keep its dependency set `{iced-x86}`, so the interpreter side uses `core::arch` or inline `asm!`, as TASK-368 will.
- An operand that crosses a 16-byte boundary: no host primitive covers it. Fall back to a striped lock keyed by address, taken only on this path. It is atomic against other misaligned locked ops on the same bytes, NOT against aligned atomics or plain stores to those bytes from other threads. Document exactly that.
- A page-crossing operand must be fault-checked on both pages before any byte is written (the task-305 fault-atomicity rule).
- Both backends share one policy. The JIT's cold path calls a helper in core that runs the same code as the interpreter.
- `lock adc` / `lock sbb` stay deferred (deferred.md:191).

RELATION.
- task-73 (done) recorded the fallback as acceptable. This task replaces it.
- TASK-368 supplies the 16-byte host CAS.
- TASK-357 (trap sites) turns any remaining host fault into an attributable one, but the goal here is that none remains.
- TASK-362's SB+locked test covers ordering, not atomicity, so it does not cover this.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 First, on the aarch64 runner: a JIT-compiled lock xadd and lock cmpxchg at byte offsets 1..7 inside a word and across a 16-byte boundary; what happens (completes, wrong value, or host fault) is recorded on this task
- [ ] #2 A contended-counter test (like threads.rs contended_counter_*) with the counter at a misaligned address inside an 8-byte word, inside a 16-byte granule, across a 16-byte boundary and across a page boundary gives the exact total on both backends, on the x86 and the aarch64 runner
- [ ] #3 Misaligned lock xadd, lock cmpxchg and xchg results and flags match the Unicorn oracle (single-threaded) on both backends
- [ ] #4 Aligned operands add no instructions in the interpreter, and at most an alignment test with a cold slow path in the JIT; the density test and the perf gate against bench/baseline.json stay within noise
- [ ] #5 The guarantee that remains for operands crossing a 16-byte boundary is documented at atomic_rmw/atomic_cas and in README Known gaps, and the old 'only cross-thread atomicity is lost' comment is gone
- [ ] #6 No guest locked op can raise a host alignment fault on any supported host
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
