---
id: TASK-388
title: >-
  lift: forms the guest CPU profile does not advertise, and privileged forms,
  must fault as they would on that CPU
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - isa-identity
  - fault-precision
dependencies:
  - TASK-367
  - TASK-359
priority: low
ordinal: 424000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
READ at 96fd5c6, not executed. The lifter decodes Long64 and Compat32 with `DecoderOptions::NONE` (lift/mod.rs:156-168) and lifts every form it has an arm for, whatever the vcpu advertises. `lift_block(mem, at, mode)` (lift/mod.rs:237) takes no feature set, and the translation-cache key is guest address plus CpuMode (cache.rs:10-20). `GuestCpuFeatures` only shapes what CPUID and xgetbv report (features.rs:1-11). So a guest on the default preset, which advertises neither AVX-512 nor BMI, still executes EVEX and BMI instructions where the CPU it was told about raises #UD. Each instruction page lists that fault: "#UD If CPUID.07H.00H:EBX.ADX[19] = 0" (SDM Vol 2A ADCX), for example; CMPXCHG16B lists its CPUID check among the 64-bit mode exceptions (SDM Vol 2A).

Privileged and policy-dependent forms in user mode take the wrong exit:
- cli, sti, clts, invd, invlpg, lldt/ltr/lmsw, wbinvd, sysexit/sysret and the rest of the system instructions are Unsupported, so they become Exit::UnknownInstruction (v1 missing list, backlog/docs/compat/isa-coverage.md:102-219). At CPL 3 they raise #GP(0) (SDM Vol 3A §6.9; Vol 2A CLI and STI: #GP(0) if CPL > IOPL).
- rdpmc raises #GP(0) at CPL 3 unless CR4.PCE = 1 (SDM Vol 2B RDPMC); rdtsc and rdtscp raise #GP(0) at CPL 3 when CR4.TSD = 1 (SDM Vol 2B RDTSC, RDTSCP).
- hlt lifts to Exit::Hlt (lift/mod.rs:2597-2600) and in/out to Exit::PortIo (lift/mod.rs:2611-2612). At CPL 3 hardware raises #GP(0) for hlt, and for in/out unless IOPL or the I/O permission bitmap allows the port. These stay the default: the unit-test corpus ends its snippets with hlt (spec §14, spec.md:1046, records that hlt is privileged and why the corpus uses it anyway).
UnknownInstruction is the right exit for a lift gap and the wrong one for an architectural fault. An embedder that turns faults into signals needs Exit::Exception with the vector, at the instruction.

WHY AN EMBEDDER NEEDS IT. Software probes features by executing an instruction under a SIGILL handler and checking whether the handler ran. Profilers read counters with rdpmc and fall back when it faults. Runtimes with their own code generators trust the fault, not CPUID. An engine that executes whatever it can lift answers every such probe "supported", and the program then mixes paths chosen by CPUID with paths chosen by probing, on a CPU that does not exist.

DESIGN DIRECTION:
1. A lift-time check. For each decoded instruction, map iced's `cpuid_features()` (the default `instr_info` feature, already enabled; no new dependency, boundary.rs stays green) to `Feature` bits and compare with the advertised set. A form that needs an unadvertised feature lifts to #UD at that instruction through the exception path TASK-367 builds for invalid opcodes: the instructions before it in the block execute, then the block ends with the fault. The iced-to-Feature mapping needs a table. Some iced features have no `Feature` bit because they are always on (the scalar baseline, CMOV, FXSR, ...). femms (3DNOW) must fault on every profile that does not report 3DNow!. The compat probe already maps iced features to generations in `code_gen` (x86jit-tests/src/compat.rs:76-94); the new table is a finer version of that mapping and should replace it rather than live beside it.
2. A user-mode privilege policy on the Vm: which privileged forms raise #GP(0) instead of their current exit, plus the CR4.TSD, CR4.PCE and IOPL-equivalent switches. The default is today's behaviour.
3. Translations then depend on the feature set and the policy. Today the set is copied into each vcpu when it is created (vm.rs:575-578), and the cache key does not carry it. Either the set and the policy become Vm-wide and immutable once a vcpu exists, or they join the cache key, or changing them flushes the cache. The first is the smallest change, and it gives TASK-385 (AOT translation units) a well-defined input for its cache key.

COST. One table lookup per decoded instruction at lift time, nothing at run time. Measure lift throughput and the bench suite.

RELATION. Complements task-333 and task-334 (EVEX handlers ignore the write mask and the embedded broadcast): those fix AVX-512 when it is advertised; this faults it when it is not. Depends on TASK-367 (the #UD exit at an instruction inside a block) and TASK-359 (the advertised set, preset or whole-leaf table, as the single source of truth). Amends decision-9, which calls advertising past the lifter a caller risk and says nothing about the reverse.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With the default preset, an EVEX instruction, a BMI2 instruction and femms each exit with Exception { vector: 6 } at that instruction on both backends, after the earlier instructions of the block have executed; with v4 (v3 for BMI2) they run as today
- [ ] #2 The check is driven by the same advertised set CPUID reports, preset or TASK-359 table: a test advertises a set, reads CPUID, and asserts that every instruction whose feature CPUID reports as absent faults with #UD
- [ ] #3 With the privilege policy on, cli, sti, rdmsr, wrmsr, rdpmc (PCE clear), rdtsc (TSD set), hlt and in/out exit with Exception { vector: 13 } at the instruction on both backends; with it off every current exit is unchanged and the existing corpus passes untouched
- [ ] #4 Changing the feature set or the policy after a vcpu exists is either refused or invalidates every translation lifted under the old values; a test lifts a block, changes the set, and observes the new behaviour
- [ ] #5 Lift throughput (a lift micro-benchmark or the compat probe's run time) and the bench suite are measured before and after against bench/baseline.json, and the numbers are recorded on this task
- [ ] #6 code_gen in x86jit-tests/src/compat.rs uses the new iced-to-Feature table instead of a second mapping
- [ ] #7 decision-9 is amended (or superseded) to state the new rule, and README Known gaps (plus spec §7, wherever it describes what the lifter accepts) says what is enforced
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
