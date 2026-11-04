---
id: TASK-371
title: >-
  lift: 59 of 60 MMX encodings are missing, and emms does not empty the x87 tag
  word
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - isa-identity
  - mmx
dependencies: []
priority: high
ordinal: 407000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
READ at 96fd5c6, not executed. Of MMX, only the MMX↔XMM bridge lifts: movq2dq and movdq2q (lift/mod.rs:1347-1359, task-152). The coverage map's Long64 MMX row is 1 lifted, 59 missing (backlog/docs/compat/isa-coverage.md:20; list at 40-100): movd/movq to and from MM, the packs, padd/psub with their saturating forms, pand/pandn/por/pxor, pcmpeq/pcmpgt, pmaddwd/pmulhw/pmullw, the shifts and the punpcks. The one lifted entry is emms, and it is wrong (below). The v1 missing list holds 24 more MMX-register encodings (isa-coverage.md:115-120 and 142-160: cvt*pi*, movntq, pavgb/w, pextrw/pinsrw, pmaxsw/ub, pminsw/ub, pmovmskb, pmulhuw, pmuludq, psadbw, pshufw, psubq) and the v2 list 16 SSSE3 forms on MM registers (isa-coverage.md:224-249: pabs*, palignr, phadd*/phsub*, pmaddubsw, pmulhrsw, pshufb, psign*). 99 MMX-register encodings in all.

Every preset advertises MMX (features.rs:96-102, CPUID.01H:EDX[23]). The waiver that allows it (x86jit-tests/compat/cpuid-waivers.ron:9) still says "Zero MMX lifted", stale since task-152. MMX is not dead in 64-bit code: hand-written memory and string routines in older libraries, codecs and context-save code use `pxor mm, mm`, `movq mm, [mem]` and the rest, and a guest that checks the MMX bit takes those paths.

EMMS. `Emms | Femms => Ok(false)` (lift/mod.rs:1360-1362), with the comment "with no tag word modeled ... this is a no-op". The tag word has been real state since task-324: `fpu_empty`, bit i set = physical register R(i) empty (state.rs:377-385), reset to 0xFF (state.rs:474-476). Today the no-op is invisible only because no MMX instruction marks the tags valid either: `exec_movdq2q` writes the mantissa and sets the exponent bytes to all ones, but leaves TOP and `fpu_empty` alone (interp/mod.rs:4797-4805). The two halves must land together. Making MMX instructions mark every tag valid without fixing emms turns the first x87 load after any MMX code into a stack overflow; fixing emms alone changes nothing observable.

ARCHITECTURE (SDM Vol 1 §9.5.1 and §9.6.2; Vol 3A §15.2, Table 15-2 "Effects of MMX Instructions on x87 FPU State"):
- every MMX instruction except EMMS sets TOP = 0 and every tag to valid (00B);
- a write to MMn sets bits 79:64 of the aliased physical register to all ones; a read leaves them alone;
- EMMS sets every tag to empty (11B) and TOP = 0;
- the rest of the status word, the control word and the pointers are unchanged;
- MMn aliases the low 64 bits of physical R(n), not ST(n): the existing aliasing (lift/mod.rs:2957-2961).
FEMMS is AMD 3DNow! (iced tags it 3DNOW) with EMMS semantics. On a profile that does not report 3DNow! it is #UD; TASK-388 makes that check.

DESIGN DIRECTION. Lift over the low 64 bits of `fpr[i]`. The JIT reaches MMX state only through the `mmx_bridge` helper today (x86jit-cranelift/src/codegen/vector.rs:2961-2977), because `fpr`, `fpu_top` and `fpu_empty` are not in `jit_abi::CpuOffsets`. A helper per MMX op is the quick path; inline lowering needs those offsets in the ABI. Most arithmetic can reuse the 64-bit lane paths of the SSE handlers: an MMX op is its SSE counterpart on one 64-bit lane. The x87 transition (TOP = 0, tags valid) is idempotent, so one per block is enough, emitted before the first MMX instruction's first state commit. It must come after that instruction's fault-capable load, so that a faulting `movq mm, [mem]` leaves the x87 state untouched (task-305's commit-last rule).

TESTING TRAPS. The coverage ratchet keys on mnemonic names (x86jit-tests/tests/coverage_ratchet.rs:1-17), and most MMX mnemonics (Paddb, Pxor, ...) are already credited by their SSE forms, so a new MMX form passes it with no MMX test at all. The MMX forms need their own fuzz generator over MM registers. Unicorn is a weak witness: task-152 recorded that it leaves the exponent bytes 0 on movdq2q. The native oracle is the authority; it already reads an FXSAVE image (deferred.md:178), which carries the MM registers and the abridged tag word.

RELATION. task-152 (done) lifted the bridge; task-324 (done) made the tag word real. Nothing covers the rest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Long64 and Compat32 MMX missing lists are empty, and the v1/v2 missing lists hold no MM-register form
- [ ] #2 After any MMX instruction TOP = 0 and every tag is valid; after a write to MMn bits 79:64 of R(n) are all ones; after emms (and femms where 3DNow! is advertised) every tag is empty. Tested on both backends through fnstenv and fxsave images
- [ ] #3 An x87 sequence after MMX code and emms runs without a stack fault on both backends; the same sequence without emms reports the stack overflow hardware reports; both compared against the native oracle
- [ ] #4 A faulting MMX memory load leaves fpr, TOP and the tags unchanged on both backends
- [ ] #5 A differential fuzz generator over MM registers and m64 operands runs under cargo xfuzz (register and --mem legs), compared with the native oracle on x86-64 (MM registers and tag word from the FXSAVE image) and with Unicorn with bits 79:64 masked; the coverage ratchet credits MMX forms separately from SSE forms
- [ ] #6 An MMX loop is added to the bench suite; helper versus inline lowering is measured against bench/baseline.json and the choice recorded on this task
- [ ] #7 The MMX waiver in cpuid-waivers.ron, README Known gaps ("MMX is minimal", README.md:102) and the stale comment at lift/mod.rs:1360-1361 state what is true
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
