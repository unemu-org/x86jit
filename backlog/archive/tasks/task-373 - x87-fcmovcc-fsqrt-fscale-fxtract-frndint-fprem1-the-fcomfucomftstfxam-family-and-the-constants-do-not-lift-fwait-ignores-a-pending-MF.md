---
id: TASK-373
title: >-
  x87: fcmovcc, fsqrt, fscale, fxtract, frndint, fprem1, the fcom/fucom/ftst/fxam
  family and the constants do not lift; fwait ignores a pending #MF
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - fp-fidelity
  - x87
dependencies: []
priority: high
ordinal: 409000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two defects in the x87 front end: a block of ordinary encodings that do not lift, and `fwait`, which lifts to nothing even though pending-#MF state now exists.

A. MISSING FORMS. READ at 96fd5c6: the complete x87 accept list is lift/mod.rs:1080-1085. Any other x87 mnemonic is `Unsupported`. `lift_insn`'s error then fails the whole block (lift/mod.rs:293-294) and surfaces as `Exit::UnknownInstruction` (vm.rs:1622-1626). How many instructions that one gap costs is TASK-367. The generated coverage map lists 52 Long64 x87 encodings as missing (backlog/docs/compat/isa-coverage.md:434-487). They fall into three groups.

MUST LIFT (41 encodings):
- `fcmovb/be/e/nb/nbe/ne/nu/u st0, st(i)` (8)
- `fcom` m32/m64/st(i) and its `DC D0+i` alias; `fcomp` m32/m64/st(i) and its `DC D8+i` / `DE D0+i` aliases; `fcompp` (10)
- `fucom`, `fucomp`, `fucompp` (3)
- `ftst`, `fxam` (2)
- `fsqrt`, `fscale`, `fxtract`, `frndint`, `fprem1` (5)
- `fist` m16/m32 (2)
- `fldpi`, `fldl2e`, `fldl2t`, `fldlg2`, `fldln2` (5)
- `fnop`, `ffree`, `ffreep`, `fincstp`, `fdecstp` (5)
- `fstpnce` (1), an alias encoding of `fstp st(i)`

Libc math routines and compiler-generated `long double` code use these. Today each one takes its whole block down with it.

DECIDE AND RECORD (4): the 14-byte `fldenv`/`fnstenv` (66h operand size), which are refused on purpose with a test (lift/mod.rs:570-576), and the 108-byte `fnsave`/`frstor`, whose refusal is reasoned at x87.rs:556-560 (register-image ordering). Either lift them or keep them refused, with the reason in deferred.md.

MAY STAY UNLIFTED, WITH THE REASON RECORDED (7): `fbld`, `fbstp`, the 94-byte `fnsave`/`frstor`, and `fneni`/`fndisi`/`fnsetpm`.

Architecture, all SDM Vol 2A instruction pages unless noted; AMD APM Vol 5 for `ffreep`:
- `fcmovcc` tests CF/ZF/PF and raises #IS on an empty register.
- `fcom*` raise IE on ANY NaN operand; `fucom*` raise IE only on an SNaN. Both report the result in C3/C2/C0.
- `ftst` compares with +0.0.
- `fxam` classifies, including the "empty" class (C3=1, C0=1), and puts the sign in C1. It has to read the tag state in `fpu_empty` (state.rs:377-385).
- `fsqrt` obeys precision control and RC.
- `frndint` rounds by RC.
- `fscale` truncates ST(1) toward zero.
- `fxtract` raises ZE on a zero operand.
- `fprem1` is the IEEE remainder (quotient rounded to nearest), with C2 marking a partial result and C0/C3/C1 carrying quotient bits.
- The load-constant forms round their internal constant by RC.

Implement all of them on the existing F80 model: the control-word handling of task-324, and the exception flags, condition codes, stack-fault (SF + C1) and deferred-#MF model of task-328. Witness them against hardware the way the existing x87 tests are witnessed: x87_exception_flags.rs runs every case on the host CPU through `native::run_native` (x86-64 Linux only, x87_exception_flags.rs:14-18), and the architecture-asserted half runs everywhere (x87_mf.rs).

`fist` m16/m32 share `fistp`'s store path, which task-339 says stores 0 instead of the integer indefinite. Do not add two more forms with that defect: land after task-339 or fix both together. `fprem1` shares its reduction loop shape with `fprem`, whose C2 defect is task-342.

B. FWAIT. lift/mod.rs:935-938 lifts `Wait` (9B) as a no-op "with no pending unmasked x87 exceptions modeled". That premise is out of date. task-328 added the model: `mf_pending_before` (x87.rs:705-707) checks ES (status bit 7) at the head of every waiting x87 op, and `waits_for_pending` (x87.rs:689-694) lists the non-waiting forms. SDM Vol 1 §8.6 and SDM Vol 2D WAIT/FWAIT: WAIT checks for pending unmasked x87 exceptions and raises #MF if there is one.

So after an unmasked `fdiv` by zero, a following `fwait` must trap AT the `fwait`. Today execution runs past it, and the #MF lands on the next waiting x87 instruction, or never if the code moves on to SSE. Compilers place `fwait` exactly to pin where an exception is reported.

The same section lists an approximation (x87.rs:684-692): the WAITING forms the lifter accepts (`finit`, `fclex`, `fstenv` at lift/mod.rs:1083-1084) are folded onto their no-wait twins, so they never check either. Once `fwait` checks, those forms should do the same check first, which closes that paragraph.

TRADE-OFF. The F80 software model is slower than host f64 arithmetic, but it is the only way to get 64-bit significands, precision control and C0-C3 right on every host, and it is what the already-lifted forms use. `fsqrt` needs a correctly rounded 64-bit-significand square root (a 128-bit integer root) rather than `f64::sqrt`.

RELATION. task-339..task-344 are defects in x87 forms that already lift. They are disjoint from this task, except for `fist` (task-339) and `fprem1`'s sibling `fprem` (task-342). task-328 (done) is the flags/#MF model this builds on. TASK-367 decides how much of a block one unsupported instruction costs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the 41 must-lift encodings lifts in Long64, and in Compat32 where the encoding exists. The coverage map is regenerated (`compat_map_is_current` green), and every x87 entry still listed as missing has a recorded reason
- [ ] #2 Every new op has a native-oracle case on x86-64 Linux that compares ST registers, the status word (C0-C3, the six flags, SF, ES) and the tag word. Operands cover normal, denormal, ±0, ±inf, QNaN, SNaN and an empty register. Interp == JIT from the same body on both backends. `fsqrt`, `frndint` and the constants are also covered under all four RC values and the three precision-control settings
- [ ] #3 `fcom*` raise IE on a QNaN operand and `fucom*` do not; both raise it on an SNaN. `ftst`/`fxam` set C3/C2/C0 (and C1 = sign for `fxam`) as hardware does, including the empty class. All witnessed against the native oracle
- [ ] #4 Stack faults: a push/pop form on a full or empty stack sets IE|SF with C1 per task-328's model, leaves the registers as hardware leaves them, and an unmasked one raises #MF at the next waiting instruction
- [ ] #5 `fwait` raises `Exit::Exception { vector: 16 }` with the saved RIP on the `fwait` when ES is set, on both backends, and stays a no-op otherwise. The waiting `finit`/`fclex`/`fstenv` perform the same check first. The comments at lift/mod.rs:935-936 and x87.rs:684-692 are rewritten
- [ ] #6 `fist` m16/m32 store the integer indefinite with IE on overflow and NaN, consistently with task-339
- [ ] #7 `fbld`/`fbstp`, the 94-byte `fnsave`/`frstor`, `fneni`/`fndisi`/`fnsetpm`, the 14-byte `fldenv`/`fnstenv` and the 108-byte `fnsave`/`frstor` are each either lifted or listed in deferred.md with the reason
- [ ] #8 README "Known gaps" names the x87 forms that remain unlifted
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
