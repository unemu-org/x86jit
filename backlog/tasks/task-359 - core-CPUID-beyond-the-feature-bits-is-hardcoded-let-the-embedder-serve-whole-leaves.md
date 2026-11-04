---
id: TASK-359
title: >-
  core: CPUID beyond the feature bits is hardcoded; let the embedder serve whole
  leaves
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - isa-identity
  - embedder-api
dependencies: []
priority: high
ordinal: 395000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
READ at 96fd5c6, not executed. `cpuid_run` (x86jit-core/src/interp/mod.rs:5786-5817) is the only CPUID implementation: the interpreter calls it from `exec_cpuid` (interp/integer.rs:594-598) and the JIT from `cpuid_helper` (x86jit-cranelift/src/lib.rs:231-239). Only the feature bits come from `GuestCpuFeatures`. Everything else is a literal:

- leaf 0: maximum basic leaf 7 and "GenuineIntel" (mod.rs:5801);
- leaf 1: EAX = 0x000306C3 and EBX = 0 (mod.rs:5803). EBX = 0 reports a CLFLUSH line size of 0 bytes, no addressable logical-processor IDs and initial APIC ID 0 (SDM Vol 1 Table 21-8);
- leaf 7 ignores the subleaf in ECX (mod.rs:5806). Subleaf 0 reports MAX_SUBLEAF = 0, so subleaf 1 is invalid and must read as zeros; instead it returns subleaf 0's EBX/ECX, whose bit positions name different features in subleaves 1 and 2 (SDM Vol 1 Tables 21-17 to 21-27);
- maximum extended leaf 0x8000_0001 (mod.rs:5808): no brand string (0x8000_0002-4), no address sizes (0x8000_0008), no 0x8000_0007 (invariant TSC);
- 0x8000_0001 EDX is SYSCALL and LM only (mod.rs:5810), ECX is LAHF plus LZCNT (features.rs:255-258); no AMD-defined bit such as SSE4A (Fn8000_0001_ECX[6], AMD APM Vol 3 Appendix E);
- every other leaf returns zeros (mod.rs:5811): 0x4/0xB/0x1F (cache and topology), 0xD (XSAVE sizes), 0x15/0x16 (TSC and core frequency). Intel answers a leaf above the maximum with the data of the highest basic leaf (SDM Vol 1 §21.1.3), so the zeros do not even match the vendor string being reported.

`GuestCpuFeatures` is a u64 bitset over 28 `Feature` values (features.rs:13-72) projected into five registers. It cannot carry a family, a brand string, a cache descriptor or a topology. An embedder that runs a full native userspace has to present one specific CPU: the software checks vendor and family/model before taking tuned paths, sizes buffers from the cache leaves, sizes thread pools from the topology leaves, and derives the TSC frequency from leaf 0x15. Today it cannot.

DESIGN DIRECTION. Let the embedder supply whole leaves: a table keyed by (leaf, subleaf) returning EAX..EDX, consulted by `cpuid_run` first, with the `GuestCpuFeatures` projection kept as the answer when no table is installed, so decision-9's frozen default preset does not move. The table needs two things beyond a plain map:
1. each entry says whether ECX selects a subleaf, and what an out-of-range leaf or subleaf returns (the vendor's rule, not a hardcoded zero);
2. some fields belong to one logical processor (initial APIC ID in leaf 1 EBX[31:24], x2APIC ID in 0xB/0x1F EDX: SDM Vol 1 §21.1.4, "CPUID Domains"), and some follow runtime state (OSXSAVE and the leaf 0xD sizes track CR4/XCR0: SDM Vol 1 Table 21-1). A per-vcpu override of those fields, set when the vcpu is created, covers them without a callback.

Trade-off. A callback (leaf, subleaf, vcpu) → registers is more general and smaller to specify, but it would be the core's first embedder hook (spec §14, spec.md:1040, left "hooks alongside return-based" open), it is opaque to lockstep, to `fork_with_backend` and to an AOT cache key, and it cannot be compared or serialised. A data table can be. Recommendation: table plus per-vcpu overrides; record the choice in a decision that amends decision-9, whose "trigger to revisit" already names per-preset advertising.

FEATURE BITS. Add `Feature` bits for the lifts this backlog adds: SSE4A and ADX (TASK-380), RDRAND, RDSEED and RDTSCP (TASK-369). CX8 is not a `Feature` today; it is part of the always-on leaf-1 EDX baseline (features.rs:222-234) while cmpxchg8b does not lift (TASK-368). Make it a `Feature` so a profile can withhold it, and keep it in the default preset so the frozen default does not change.

ADVERTISE ⊆ LIFT is weaker than the module doc (features.rs:7-11) suggests:
- `advertised_simd_features` (x86jit-tests/src/compat.rs:578-610) reads leaf 1 of a default `CpuState` and looks at eight bits only: MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT. The v2/v3/v4 presets are never checked, nor are CX16, MOVBE, AES, PCLMUL, XSAVE, or any leaf-7 or extended bit.
- `feature_gen` (x86jit-tests/src/compat.rs:49-74) puts AES, SHA and the vendor extensions out of scope, and has no arm for CX8, XSAVE, ADX, SSE4A, RDRAND, RDSEED, TSC or X64. Instructions tagged with those are invisible to the compat map and to the check.
- Two live consequences. v2, v3 and v4 set `Cx16` (features.rs:104-119) while `Cmpxchg16b_m128` is on the v2 missing list (backlog/docs/compat/isa-coverage.md:223); TASK-368 lifts it. The default preset advertises XSAVE and OSXSAVE (features.rs:170-171) while xsave/xrstor/xsavec/xsaveopt have no lift arm (only fxsave/fxrstor, lift/mod.rs:1127-1145). What keeps software that sizes its save area from CPUID.(EAX=0DH,ECX=0):EBX off xsave today is that leaf 0xD lies above the reported maximum basic leaf and reads as zero. That zero is load-bearing: a table that reports real XSAVE sizes moves such code onto an instruction that traps. The comment at lift/mod.rs:1127-1129 ("when XSAVE isn't advertised") describes a state that no longer holds.

Generalise the check into a helper that takes any CPUID source (a preset or a table) and reports every advertised bit whose user-mode instructions do not all lift, using iced's per-`Code` `cpuid_features()` directly instead of the generation buckets. Forms that are privileged at CPL 3 (xsetbv, for example) are excluded. The compat test runs it for every preset; an embedder can run it against its own table.

Also stale: the witness test cites "SDM Vol 2A Table 3-10 (leaf 1 ECX), Table 3-11 (leaf 1 EDX), Table 3-8 (leaf 7)" (features.rs:286-292, and the message at features.rs:336). In the checked-in oracle revision (325462-092US) those numbers belong to the CMPPD/CMPPS tables; the CPUID leaf tables are SDM Vol 1 Tables 21-9, 21-10, 21-19 and 21-20.

RELATION. decision-9 and task-117 (done) introduced `GuestCpuFeatures`; this extends that model from feature bits to whole leaves. TASK-368, TASK-369 and TASK-380 add the instructions behind the new bits. TASK-388 (fault on unadvertised forms) needs the advertised set, preset or table, as its single source of truth.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An embedder can install whole CPUID leaves ((leaf, subleaf) → EAX..EDX); one test runs a table covering vendor, family/model/stepping, leaf 1 EBX, leaf 7 subleaves 0-2, 0xD, 0x8000_0002-4 and 0x8000_0008 through the interpreter and the JIT and asserts identical registers
- [ ] #2 With no table installed, every existing reply is unchanged (the_default_preset_projects_the_exact_frozen_cpuid_bits and the compat tests pass untouched), except that leaf 7 honours the subleaf: a test reads CPUID.(EAX=7,ECX=1) and gets zeros, and fails at 96fd5c6
- [ ] #3 The out-of-range rule (leaf above the maximum, invalid subleaf) is part of the table; the default projection's zeros are documented as a deviation from SDM Vol 1 §21.1.3
- [ ] #4 Per-logical-processor fields (initial APIC ID, x2APIC ID) can differ between vcpus of one Vm: a test creates two vcpus and reads two different IDs
- [ ] #5 Feature gains Sse4a, Adx, Rdrand, Rdseed, Rdtscp and Cx8, each with a witness row in cpuid_feature_bits_match_the_documented_positions citing SDM Vol 1 Tables 21-9/21-10/21-19 or AMD APM Vol 3 Appendix E; the default preset still reports CX8 and nothing else changes in it
- [ ] #6 An advertise ⊆ lift helper reports, for any preset or table, each advertised bit whose user-mode instructions do not all lift; the compat test runs it for default, v2, v3 and v4. At this HEAD it reports at least CX16 (v2 and up), XSAVE (default) and CX8; each is fixed or carried as a reasoned per-preset waiver
- [ ] #7 The choice between table, callback and exit is recorded as a decision that amends decision-9 and resolves the hooks question in spec §14
- [ ] #8 The stale comment at lift/mod.rs:1127-1129, the SDM table citations at features.rs:286-292 and 336, and the module doc at features.rs:7-11 state what is true
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
