---
id: TASK-380
title: 'lift: SSE4A (extrq, insertq, movntsd, movntss) and ADX (adcx, adox)'
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - isa-identity
dependencies:
  - TASK-359
  - TASK-356
priority: medium
ordinal: 416000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
READ at 96fd5c6, not executed. Neither extension has a lift arm: no `Extrq`, `Insertq`, `Movntsd`, `Movntss`, `Adcx` or `Adox` anywhere under x86jit-core/src/lift. `Feature` has no bit for either (features.rs:13-55). The compat map treats both as out of scope: `feature_gen` names "SSE4A/FMA4 AMD" among the out-of-scope extensions and has no arm for SSE4A or ADX (x86jit-tests/src/compat.rs:49-74), so no coverage figure shows them.

WHO NEEDS THEM:
- SSE4A. An embedder that presents an AMD CPU through TASK-359's CPUID table reports SSE4A (CPUID Fn8000_0001_ECX[6], AMD APM Vol 3 Appendix E). Code built for such a CPU uses extrq/insertq for bit-field work and movntsd/movntss for streaming scalar stores, and runs them unconditionally.
- ADX. Most current CPUs report it (CPUID.07H.00H:EBX.ADX[19]). Multi-precision arithmetic in crypto and bignum libraries dispatches on it to run two independent carry chains (mulx with adcx and adox).

ARCHITECTURE:
- EXTRQ xmm1, imm8, imm8 and EXTRQ xmm1, xmm2 extract a field of a given length at a given bit index from xmm1[63:0] into the low bits of xmm1, clearing the rest of the low quadword; the upper quadword is undefined. A length of 0 means 64. Index + length > 64 is undefined, and so is length 0 with a non-zero index. In the register form, xmm2[5:0] is the length and xmm2[13:8] the index; other bits are ignored (AMD APM Vol 4, EXTRQ).
- INSERTQ xmm1, xmm2, imm8, imm8 and INSERTQ xmm1, xmm2 insert the low bits of xmm2 into xmm1[63:0] at a given index, leaving the other bits of the low quadword alone; the upper quadword is undefined; the same length-0 and index + length rules apply. The register form takes its controls from the source's upper quadword, not its lower one: length in xmm2[69:64], index in xmm2[77:72] (AMD APM Vol 4, INSERTQ).
- MOVNTSD m64, xmm and MOVNTSS m32, xmm are scalar non-temporal stores (AMD APM Vol 4). Here they are ordinary sized stores, the way movnti is lowered (lift/mod.rs:950-958).
- ADCX r, r/m computes DEST + SRC + CF and writes CF only; ADOX computes DEST + SRC + OF and writes OF only; every other flag is unchanged; #UD with LOCK (SDM Vol 2A ADCX, ADOX).
The undefined SSE4A results must be pinned to one documented choice. SSE4A executes only on AMD hardware, so a native witness needs an AMD host. QEMU implements SSE4A, so Unicorn can be the witness if the CPU model it runs enables the extension (check that first; if it does not, the tests are written from the APM pseudo-code and the pinned choices).

FLAGS AND ELISION. ADCX fits the IR as it stands: an `Adc` whose `set_flags` mask is CF alone already reads CF (`op_reads`, lift/mod.rs:774-784) and kills only CF in `elide_dead_flags` (lift/mod.rs:867-897). ADOX does not: its carry-in is OF, and no IR consumer reads OF as a carry. It needs an op, or an `Adc` variant, whose `op_reads` entry reports OF and whose write mask is OF alone. Both must stay partial writers, so that in a chain `adcx; adox; adcx; adox` both carries stay live across each other. This depends on TASK-356, which settles flag liveness at a fault: a fault in the middle of the chain must leave CF and OF as the last completed instruction left them.

ADVERTISING. Add `Feature::Sse4a` and `Feature::Adx` (TASK-359 adds the bits) and project them to CPUID. The v-level presets follow the x86-64 psABI levels, which include neither SSE4A nor ADX, so adding either to a preset is a separate, recorded decision. Until then an embedder opts in with `with()` or a CPUID table. Bring both extensions into the compat map's scope so the advertise ⊆ lift check covers them.

RELATION. None.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 extrq and insertq (immediate and register forms), movntsd and movntss lift in Long64 and Compat32; each has a test whose one body runs on both backends
- [ ] #2 extrq/insertq tests cover length 0 (= 64), index + length = 64, and register-form controls with bits set outside the six-bit fields (extrq: xmm2[5:0] and [13:8]; insertq: xmm2[69:64] and [77:72]); the undefined cases are pinned to one documented choice and compared with Unicorn (and with a native AMD host when one is available)
- [ ] #3 adcx and adox lift for 32- and 64-bit operands with register and memory sources; a test interleaves two carry chains and checks the sums and that SF, ZF, AF, PF and the other carry flag are unchanged, on both backends, against the native oracle on x86-64
- [ ] #4 A fault on the memory source in the middle of an adcx/adox chain leaves CF and OF as the last completed instruction left them, on both backends
- [ ] #5 Feature::Sse4a projects to CPUID Fn8000_0001_ECX[6] and Feature::Adx to CPUID.07H.00H:EBX[19], each with a witness row; neither is in the default preset, and any preset change is recorded as a decision
- [ ] #6 feature_gen brings SSE4A and ADX into scope; both reach 100% in the compat map and the TASK-359 advertise ⊆ lift helper passes for a profile that advertises them
- [ ] #7 The coverage ratchet credits the six mnemonics through fuzz arms or reasoned allowlist entries
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
