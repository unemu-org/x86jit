---
id: TASK-369
title: >-
  core: rdtsc returns the constant 0x12345678 and rdtscp/rdrand do not lift; let
  the embedder supply them
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - isa-identity
  - embedder-api
dependencies:
  - TASK-359
priority: high
ordinal: 405000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
READ at 96fd5c6, not executed. `rdtsc` lifts to two constant register writes, EAX = 0x12345678 and EDX = 0, "to keep whole-program runs deterministic" (lift/mod.rs:1065-1079). `Rdtscp` is on the v1 missing list (backlog/docs/compat/isa-coverage.md:164). `rdrand` and `rdseed` have no lift arm, and since iced tags them RDRAND/RDSEED, for which `feature_gen` (x86jit-tests/src/compat.rs:49-74) has no arm, no list shows them. CPUID reports TSC (leaf-1 EDX bit 4, features.rs:222-234) but neither RDTSCP (0x8000_0001 EDX is SYSCALL and LM only, interp/mod.rs:5810) nor RDRAND, so a CPUID-dispatched guest does not reach the two missing instructions today. A guest that does not check, or an embedder whose CPUID table reports a real CPU (TASK-359), does.

WHY THE CONSTANT IS WRONG for a full native userspace: software reads the TSC for its time base, for spin back-off and for calibration loops that wait until the counter has advanced by N. A constant gives zero deltas: time stands still, back-off never grows, and a loop that waits for the counter to move never ends. The deterministic default is still worth keeping for tests and differential runs. Lockstep already treats rdtsc, rdtscp, rdrand and rdseed as nondeterministic and skips them (lockstep.rs:310-320).

ARCHITECTURE:
- RDTSC: EDX:EAX = TSC, upper halves of RAX and RDX cleared; not serializing (SDM Vol 2B RDTSC). RDTSCP: the same, plus ECX = IA32_TSC_AUX[31:0] with RCX[63:32] cleared; it waits until all earlier instructions have executed and earlier loads are globally visible; #UD if CPUID.80000001H:EDX.RDTSCP[27] = 0 (SDM Vol 2B RDTSCP).
- The TSC is monotonic, and with the invariant TSC (CPUID.80000007H:EDX[8]) it runs at a constant rate (SDM Vol 3B §20.17, §20.17.1). IA32_TSC_AUX is per logical processor; operating systems store a CPU number there, which is what rdtscp-based getcpu reads.
- Both raise #GP(0) at CPL > 0 when CR4.TSD = 1 (SDM Vol 2B RDTSC, Operation). An OS can set it; the privilege policy in TASK-388 owns that switch, not this task.
- RDRAND/RDSEED: CF = 1 and a value, or CF = 0 and the destination zeroed; OF, SF, ZF, AF and PF are cleared in both cases (SDM Vol 2B RDRAND, RDSEED). Software retries on CF = 0 (SDM Vol 1 §7.3.17). Enumerated by CPUID.01H:ECX.RDRAND[30] and CPUID.07H.00H:EBX.RDSEED[18].

DESIGN DIRECTION. An embedder-supplied source, per Vm, that knows which vcpu is asking. It returns the counter for rdtsc and rdtscp, the per-vcpu TSC_AUX for rdtscp's ECX, and a value plus success flag for rdrand and rdseed. Both tiers call one shared run function, the way cpuid does (interp/integer.rs:594-598; x86jit-cranelift/src/lib.rs:231-239). With no source installed the defaults reproduce today exactly: rdtsc's constant, TSC_AUX = 0, and for rdrand/rdseed a deterministic per-vcpu sequence with CF = 1. No determinism test moves. The source owns scale, offset and cross-vcpu monotonicity; x86jit promises only one call per executed instruction. When the embedder also serves CPUID leaves (TASK-359), invariant-TSC, 0x15 and 0x16 must agree with what the source does; that consistency is the embedder's, but the docs must say so.

TRADE-OFF. An exit per read (the return-based model of spec §5, like port I/O) needs no hook, but it leaves compiled code on every read, and back-off loops read the counter thousands of times. A hook costs one indirect call per read, but it is an embedder callback in the core, the question spec §14 (spec.md:1040) left open. Use whatever mechanism TASK-359's decision settles. If that decision picks a pure data table for CPUID, time still needs something callable, because the value changes on every read.

FLAGS. rdrand and rdseed write all six status flags. The new IR op must be a full flag writer in `elide_dead_flags` (lift/mod.rs:867-897): either it kills earlier producers, or it is treated as a barrier. Its flags must also follow whatever contract TASK-356 settles for exact flags at a fault.

ADVERTISING. RDTSCP (CPUID.80000001H:EDX[27]), RDRAND and RDSEED become visible only through the `Feature` bits TASK-359 adds and only once they lift; the default preset stays frozen (decision-9).

RELATION. No open task. deferred.md's "Threaded virtual clock" entry belongs to unemulinux and is unrelated: that is an OS clock, not the instruction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With a source installed, rdtsc and rdtscp return its counter on both backends and rdtscp's ECX returns the vcpu's TSC_AUX; a test with two vcpus reads two different TSC_AUX values and a counter that never decreases
- [ ] #2 With no source installed, every existing test and fixture is unchanged; a test pins the defaults for rdtsc (0x12345678), rdtscp and rdrand/rdseed
- [ ] #3 rdrand and rdseed lift for 16-, 32- and 64-bit destinations; a test drives a source that fails once and then succeeds, and checks CF, the zeroed destination on failure and OF/SF/ZF/AF/PF cleared, on both backends
- [ ] #4 A test proves the flag writes survive dead-flag elision without relying on interp == JIT: both tiers execute the same narrowed IR, so an elision error is identical in both (TASK-356). With a source that succeeds and one that fails, it asserts flag values directly on both backends: after rdrand, CF matches the outcome and OF/SF/ZF/AF/PF are clear; a flag read between an earlier producer and rdrand (e.g. `add; setc; rdrand; seto`) sees the producer's value. A unit test on the lifted IR asserts the FlagMask `elide_dead_flags` leaves on rdrand and on the earlier producer, against the live sets the chosen model (kill or barrier) implies
- [ ] #5 Rdtscp leaves the v1 missing list and the compat map counts rdrand/rdseed (RDRAND and RDSEED brought into feature_gen scope)
- [ ] #6 The cost of rdtsc in a tight loop is measured for the default and for an installed source (a bench row next to bench/baseline.json) and recorded on this task
- [ ] #7 Feature::Rdtscp, Rdrand and Rdseed project to CPUID.80000001H:EDX[27], CPUID.01H:ECX[30] and CPUID.07H.00H:EBX[18], with witness rows; none is in the default preset
- [ ] #8 spec.md and the rdtsc comment at lift/mod.rs:1065-1066 describe the source and its deterministic default
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
