---
id: TASK-374
title: >-
  fp: SSE float→integer conversions saturate instead of returning the integer
  indefinite
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - fp-fidelity
dependencies: []
priority: medium
ordinal: 410000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
READ at 96fd5c6: both tiers use saturating casts for every SSE/AVX float-to-integer conversion.
- The scalar `cvt(t)ss2si`/`cvt(t)sd2si` (SSE and VEX) and the AVX-512 unsigned `vcvt(t)ss2usi`/`vcvt(t)sd2usi` (lift/mod.rs:2157-2173) are executed by `exec_v_cvt_to_int` (interp/vector.rs:4389-4417). It ends in Rust `as` casts (4412-4415). Its comment says this "matches the JIT's `fcvt_to_{s,u}int_sat`" and that "the x86 integer-indefinite result on invalid operands is deferred" (4408-4410). The JIT emits exactly that (x86jit-cranelift/src/codegen/vector.rs:4287-4330; `fcvt_to_sint_sat`/`fcvt_to_uint_sat` at 4320/4322).
- The packed `cvt(t)ps2dq`/`cvt(t)pd2dq` in SSE, VEX.128 and VEX.256 (lift/mod.rs:2176-2192) go through `packed_cvt128` (interp/vector.rs:4463-4477, `r as i32 as u32` at 4476) and `exec_v_packed_cvt_wide256` (4538-4549). The JIT uses `fcvt_to_sint_sat` at codegen/vector.rs:3765, 4381, 4411, 4477 and 4479. The doc comment at 4360-4363 states the same deferral.

A Rust float-to-integer `as` cast saturates to MIN/MAX and turns NaN into 0 (Rust Reference, "Numeric cast"). Cranelift's `*_sat` conversions are defined the same way. So the two tiers agree with each other and disagree with hardware.

ARCHITECTURE.
- Signed forms (SDM Vol 2A CVTSS2SI, CVTTSS2SI, CVTSD2SI, CVTTSD2SI, CVTPS2DQ, CVTTPS2DQ, CVTPD2DQ, CVTTPD2DQ): when the converted result is out of range (NaN included), IE is raised and, if it is masked, the result is the integer indefinite: 80000000H for a doubleword and 80000000_00000000H for a quadword, per lane for the packed forms.
- Unsigned AVX-512 forms (SDM Vol 2C VCVTSS2USI, VCVTTSS2USI, VCVTSD2USI, VCVTTSD2USI): an unrepresentable result returns 2^w - 1, i.e. all ones.

IE is masked in the reset MXCSR, so this result is what every program that does not unmask IE observes, and that is nearly all of them. This task does NOT depend on MXCSR state.

Divergences that follow from the rules and the code above. They were derived, not executed; AC#1 turns them into witnesses:

    form                    input              hardware               engine today
    cvttsd2si r32           +2^31              0x80000000             0x7FFFFFFF
    cvttsd2si r32           +inf               0x80000000             0x7FFFFFFF
    cvttsd2si r32           NaN                0x80000000             0x00000000
    cvtsd2si  r32           2147483647.5 (RN)  0x80000000             0x7FFFFFFF
    cvttsd2si r64           +2^63              0x8000000000000000     0x7FFFFFFFFFFFFFFF
    cvttps2dq (each lane)   NaN                0x80000000             0x00000000
    vcvttsd2usi r32         -1.0               0xFFFFFFFF             0x00000000
    vcvttsd2usi r32         NaN                0xFFFFFFFF             0x00000000
    cvttsd2si r32           -inf, -2^31-1      0x80000000             0x80000000  (agree)

The signed negative-overflow case agrees only because saturating MIN happens to equal the indefinite. The `2147483647.5` row is the one a range check placed BEFORE rounding would miss: the value is in range and rounds out of it.

WHY AN EMBEDDER NEEDS IT. Libc and language runtimes rely on this result. Code that converts and then tests for the indefinite to detect overflow or NaN (`lrint`/`llrint`-style paths, JIT-compiled managed code that implements its own saturation, checked casts) takes the wrong branch when it gets MAX or 0 instead.

DESIGN DIRECTION. Use one stated rule per (width, signedness), applied after rounding: the result is valid iff `lo <= r < hi`, with the bounds as exact floats (-2^31/2^31, -2^63/2^63; for unsigned, 0 <= r < 2^32 or 2^64, where a rounded or truncated -0.0 counts as 0). A NaN fails both compares, so a single select between the saturated conversion and the indefinite covers every case. In the interpreter this is one shared helper used by all three sites. In the JIT it is the same compare/compare/and/select shape on every host. That follows the rationale at codegen/mod.rs:3617-3622: the same IR carries the same meaning on both hosts, and the x86 suite exercises the aarch64 path's IR. The alternative, a raw `cvttsd2si` on x86 hosts and a fix-up only on aarch64, would ship the aarch64 path untested by the x86 suite.

TRADE-OFF. Two compares, an `and` and a select per lane on every convert. That is cheap next to a helper call, but convert-heavy loops pay it. Measure it.

SCOPE LIMITS. The sticky IE flag and an unmasked-IE #XM are not modelled. TASK-372 records that decision in deferred.md. The rounding for the non-truncating forms becomes RC-dependent under TASK-372; the range check is placed after it, whichever task lands first. EVEX packed converts that are not lifted today (the `*2udq`/`*2qq` families) must reuse the same helper when they are added. The x87 `fist`/`fistp`/`fisttp` path is task-339.

RELATION. deferred.md's MXCSR entry says that "convert-to-int saturates, and the integer-indefinite result is deferred with it". This task removes that clause. task-339 is the x87 equivalent on a separate code path (x87.rs).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Native-oracle witnesses on x86-64 Linux for every lifted form: `cvt(t)ss2si` and `cvt(t)sd2si` with r32/r64 destinations in SSE and VEX encodings; `vcvt(t)ss2usi` and `vcvt(t)sd2usi` with r32/r64; `cvt(t)ps2dq` and `cvt(t)pd2dq` in SSE, VEX.128 and VEX.256. Inputs: ±2^31, ±2^63, 2^32, 2^64, ±inf, QNaN, SNaN, -1.0, -0.5, and a value that rounds out of range (2147483647.5). The test fails at 96fd5c6
- [ ] #2 The same cases pass on BOTH backends from one body (interp == JIT == hardware). On the aarch64 runner, interp == JIT == Unicorn, where `fcvt_to_sint_sat` lowers to a different host instruction
- [ ] #3 The interpreter has one shared range/indefinite helper used by `exec_v_cvt_to_int`, `packed_cvt128` and `exec_v_packed_cvt_wide256`, and the JIT emits one shape. The comments at interp/vector.rs:4408-4410 and 4463-4465 and codegen/vector.rs:4310-4313 and 4360-4363 are rewritten
- [ ] #4 Cost measured against bench/baseline.json (the simd workload) and the numbers recorded on this task
- [ ] #5 The differential fuzzer's operand pools reach out-of-range and NaN inputs for these forms (verified, or added), so a regression to saturation is found without these hand-written cases
- [ ] #6 deferred.md no longer says convert-to-int saturates, and README "Known gaps" makes no such claim
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
