---
id: TASK-361
title: >-
  tests: flag lift arms that emit no IR, and run the aarch64 suite on every
  push
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - tests
  - ci
dependencies: []
priority: high
ordinal: 397000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
No oracle can see a silent no-op. Take a lift arm that returns `Ok(false)` without pushing IR. It passes every jit == interp test, because both tiers do nothing in the same way. It also passes the coverage ratchet: the ratchet's ALLOWLIST doc comment waives "intentionally trivial (nops, fences, `endbr`, …)" ops (x86jit-tests/tests/coverage_ratchet.rs:176-184), so fences are waived by name. `Emms` is listed with "emms is a no-op in our model" (line 345), and `Wait`, `Mfence` and `Lfence` with no reason at all (lines 415, 420, 990).

READ at HEAD 96fd5c6. These arms emit no IR:
- lift/mod.rs:931-938: `Nop`, `Endbr64`/`Endbr32`, `Pause`, `Wait`, `Rdsspd`/`Rdsspq`, `Prefetchnta`/`t0`/`t1`/`t2`/`w`/`wt1`
- lift/mod.rs:1126: `Ldmxcsr`/`Vldmxcsr`
- lift/mod.rs:1146-1147: `Mfence`/`Lfence`/`Sfence`, commented "single-threaded ordering is already TSO"
- lift/mod.rs:1360-1362: `Emms`/`Femms`, commented "no tag word modeled"

And two arms write only constants:
- lift/mod.rs:1114-1125: `Stmxcsr`/`Vstmxcsr` store `0x1F80`
- lift/mod.rs:1065-1079: `Rdtsc` writes EAX=`0x12345678`, EDX=0

Some of these are correct, because the instruction really has no architectural effect here:
- A prefetch is a hint (Intel SDM Vol. 2B, PREFETCHh).
- PAUSE is a spin-wait hint (Vol. 2B, PAUSE).
- ENDBR32/ENDBR64 is a NOP when indirect-branch tracking is not enabled (Vol. 2A, ENDBR64).
- RDSSPD/RDSSPQ is a NOP when shadow stacks are not enabled (Vol. 2B, RDSSPD/RDSSPQ).

The rest are wrong:
- MFENCE and SFENCE (Vol. 2B) and LFENCE (Vol. 2A) order memory accesses as other processors see them. "Single-threaded" does not hold for a `Vm` with several vcpus, and on an aarch64 host under `MemConsistency::Fast` nothing else orders the accesses (TASK-355).
- EMMS sets every x87 tag to empty (Vol. 2A, EMMS), and FEMMS does the same (AMD APM Vol. 5, FEMMS). The tag state is modelled now: `CpuState::fpu_empty` (state.rs:377-385), with stack faults delivered (README:117). Dropping emms makes the next x87 push after MMX code see a full stack (TASK-371).
- WAIT/FWAIT checks for and handles pending unmasked x87 exceptions (Vol. 2D, WAIT/FWAIT). The engine now delivers `#MF` (x87.rs:674-706, README:117), so the comment at lift/mod.rs:935-936 ("no pending unmasked x87 exceptions modeled") is stale, and the no-op is wrong (TASK-373).
- LDMXCSR (Vol. 2A) and STMXCSR (Vol. 2B): MXCSR is not state (TASK-358). The ratchet's own reason is wrong in the opposite direction. coverage_ratchet.rs:284-288 says the register "is modelled as storage", and deferred.md:177 says the same, but `CpuState` has no MXCSR field and ldmxcsr throws its operand away.
- RDTSC (Vol. 2B) reads a counter that increments. A value that never moves breaks every guest that measures elapsed time or spins until it advances (TASK-369).

FIRST HALF: an effect audit, separate from the ratchet's coverage question. Add a test that lifts every in-scope encoding through the compat probe (`x86jit_tests::compat`, as `lifted_mnemonics` does at compat.rs:416-432) and sorts the IR between `InsnStart` markers into three classes: no op, constant writes only, or real work. Every mnemonic in the first two classes must be on one of two lists:
- an allow-list, where each entry carries the SDM or APM page that makes the instruction architecturally empty;
- a known-defect list, where each entry names the open task that fixes it. When that task closes, the entry must leave the list.
A new silent arm then fails the build instead of passing every oracle.

SECOND HALF: CI. `.github/workflows/ci.yml` runs only on `workflow_dispatch` (lines 19-20; the header at 3-5 says "Manual only", and status.md:163 repeats it). Its aarch64 leg (`ubuntu-24.04-arm`) is the only place where the AArch64 codegen and the TSO barriers execute. That leg has no real-CPU oracle: interp == JIT and interp == Unicorn carry it (ci.yml:11-14). Several items in this backlog only show up on aarch64: memory ordering (TASK-355, TASK-362, TASK-377), the host-FP NaN and signed-zero rules (TASK-381), and fault recovery (TASK-364). Run that leg on every push and pull request. At minimum it runs:
- the jit == interp suites (jit.rs, superblock.rs);
- the multi-thread and ordering tests (mt.rs, threads.rs, tso.rs, mt_atomic_store_coherence.rs, and the litmus suite of TASK-362 once it lands);
- the x87 and FP suites (x87_*.rs, differential.rs).
If the push leg has to be a subset to stay cheap, keep the full matrix on dispatch.

Existing work:
- task-236 asks for push-triggered CI to run the unemulinux ladder and changes the same `workflow_dispatch` trigger. Coordinate the trigger edit with it. Its scope is the cross-repository ladder; this task covers this repository's own aarch64 leg.
- task-346 covers a nearby blind spot: `jit_eq_interp` passes when the op does not lift. This task covers the case where the lift succeeds and does nothing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test in x86jit-tests sorts every lifted encoding's IR as empty, constant-only or effectful, and fails on any empty or constant-only mnemonic that is on neither list, naming it
- [ ] #2 Every allow-list entry cites the SDM or APM instruction page that makes it architecturally empty. Every known-defect entry names an open task id, and a guard fails when that task's status is Done
- [ ] #3 Negative control, recorded on this task: an injected arm that returns `Ok(false)` for an instruction with an architectural effect turns the test red
- [ ] #4 The first known-defect list holds exactly mfence/lfence/sfence, emms/femms, wait, ldmxcsr/vldmxcsr, stmxcsr/vstmxcsr and rdtsc, minus any whose task has already closed
- [ ] #5 The wrong justifications are corrected: coverage_ratchet.rs:176-184 (fences as "intentionally trivial") and 284-288 ("modelled as storage"), lift/mod.rs:935-936 (no pending #MF), 1146 ("single-threaded") and 1360-1361 ("no tag word"), and deferred.md:177
- [ ] #6 ci.yml runs the aarch64 leg on push and pull_request, covering at least the jit == interp, multi-thread/ordering and x87/FP suites. The ci.yml header and status.md:163 say what now runs when
- [ ] #7 The push leg's wall clock is measured and recorded. An injected failing test on the aarch64 leg turns the check red (verified once)
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
