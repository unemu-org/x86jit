---
id: TASK-381
title: >-
  jit: audit the SSE/AVX lowerings for ARM-host NaN, min/max and signed-zero
  rules
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - fp-fidelity
  - arm-host
dependencies: []
priority: medium
ordinal: 417000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On an x86 host, a bare Cranelift float op computes the x86 way, because it IS the x86 instruction. On aarch64 the same IR computes the ARM way. The rules differ in ways a guest can observe once it reads a float's bits as an integer, which libc `isnan`/`signbit` paths, NaN-boxing runtimes and hash-of-double code all do. The x86 suite cannot see these divergences, and the aarch64 lane is manual (.github/workflows/ci.yml:20, matrix at 33-37).

WHERE THE RULES DIFFER.
1. INVALID OPERATION WITH NO NaN OPERAND (0×inf, inf-inf, 0/0, the square root of a negative). x86 returns the QNaN floating-point indefinite, which has the sign bit set: 0xFFC00000, 0xFFF8000000000000 (SDM Vol 1 §4.8.3.7). ARM returns the default NaN, which has the sign bit clear: 0x7FC00000 (Arm ARM DDI 0487, FPDefaultNaN). MEASURED on an arm64 host with Rust host arithmetic and FPCR = 0: `0/0`, `0*inf`, `inf-inf` and `sqrt(-1)` all give 0x7fc00000.
2. SRC1 QUIET, SRC2 SIGNALLING. x86 returns SRC1 (SDM Vol 1 §4.8.3.5, Table 4-8). ARM returns quiet(SRC2). MEASURED on the same host: 0x7FC0AAAA + 0x7F80BBBB = 0x7fc0bbbb. This is the case `x86_nan_from_src1` exists for (x86jit-cranelift/src/codegen/mod.rs:3609-3623).
3. DEFAULT-NaN MODE. With FPCR.DN = 1, every propagated NaN becomes the default NaN. FPCR measured 0 on that host, but nothing in x86jit states the requirement or checks it, and an embedder or host runtime may set DN.
4. MIN/MAX. ARM FMIN/FMAX return a NaN if either operand is NaN and order -0 < +0. x86 MINPS/MAXPS return the SECOND operand on a NaN or when both are zeros, and forward an SNaN SRC2 unquieted (SDM Vol 2B MINPS, MAXPS).

READ at 96fd5c6: the lowerings fall into three classes.
- (a) x86 RULE SPELLED OUT IN IR.
  - `emit_fbin` add/sub/mul/div wrap the result in `x86_nan_from_src1` (codegen/mod.rs:3571-3588). That covers rule 2 only.
  - min/max lower to compare+select (3589-3605), which is host-independent and matches rule 4.
  - `dpps_native` routes products and sums through `sse_nan_result`/`sse_fadd` (codegen/vector.rs:545-608).
- (b) SHARED RUST HELPER called from compiled code, for example `fma_helper` (lib.rs:1619), `vhfloat_helper` (1340) and `cvtps2ph_helper` (1449). Rust compiled for aarch64 is itself subject to the ARM rules unless it spells the x86 rule out, so "shared helper" is NOT automatically safe.
  - `fma_elem` spells everything out, including the indefinite (interp/mod.rs:4484-4535).
  - `hfloat` (6030) does not: it uses `sse_binop_f32` (6050-6051).
- (c) BARE CRANELIFT OP. A grep at 96fd5c6 finds 33 call sites of float-producing ops: `fadd`, `fsub`, `fmul`, `fdiv`, `sqrt`, `fdemote`, `fpromote`, `fvdemote`, `fvpromote_low`, `nearest`, `floor`, `ceil`, `trunc`, `fcvt_from_sint` and `fcvt_from_uint`. Twelve are in codegen/mod.rs and 21 in codegen/vector.rs. There are also seven `fcvt_to_*int_sat` sites, whose integer results are TASK-374's.

TWO PREDICTED DEFECTS. Both follow from the code plus the measured host behaviour. The JIT was not executed on aarch64 for this task.
- (i) `sqrt`/`rsqrt` OF A NEGATIVE. The interpreter returns the indefinite explicitly (`apply_un_f32`/`apply_un_f64`, interp/mod.rs:6271-6321). The JIT emits a bare `sqrt` (codegen/mod.rs:3542) and `fdiv(1, sqrt(x))` (3547-3551). On aarch64, then, the JIT returns 0x7FC00000 where the interpreter returns 0xFFC00000, and interp != JIT.
- (ii) ADD/SUB/MUL/DIV INVALID OPERATION. The interpreter's `sse_binop_f32`/`f64` (interp/mod.rs:2815-2863) hand the no-NaN-operand case to host arithmetic, and `x86_nan_from_src1` only guards a NaN SRC1. On aarch64, BOTH tiers return 0x7FC00000. They AGREE with each other and are both wrong. `hsubps(inf, inf)` and every other `hfloat` form inherit this.

So "the interpreter is the oracle" does not hold on an ARM host. The interpreter is host-independent only where it spells the rule out. On the aarch64 runner the reference has to be either Unicorn, which models x86 FP in software, or bit patterns captured from hardware through the native oracle on x86-64 and pinned in the test. The CI comment says the aarch64 lane runs interp == Unicorn (ci.yml:12). Whether that corpus ever feeds an invalid operation was not checked here.

DESIGN DIRECTION. Inventory every float-producing lowering and every float-doing helper reachable from compiled code, and classify each as (a), (b) or (c). Then either fix or prove each one. The likely fix is a single `x86_nan_result(src1, src2, r)` that implements the WHOLE Table 4-8 rule plus the indefinite: quiet(SRC1) if SRC1 is a NaN, else quiet(SRC2) if SRC2 is a NaN, else the indefinite if r is a NaN, else r. A single-operand counterpart covers `sqrt`/`round`/`cvt`. The interpreter's `sse_binop_*` get the same third branch. With the whole rule emitted explicitly, the lowering no longer depends on FPCR.DN either: every NaN result is selected from the sources or from the constant.

Emit it on every host, as codegen/mod.rs:3617-3622 already argues for `x86_nan_from_src1`. Gating it to aarch64 would save a compare and a select per op on x86, but it would ship an aarch64 path that the local x86 loop never runs.

TRADE-OFF. One extra compare and select per float op on every host. Measure it on the simd workload. If the cost is material, the alternative is to prove per site that the case cannot arise, which is valid for min/max and the conversions, not for arithmetic.

RELATION. task-348 is one instance (`dppd_native`, codegen/vector.rs:653-679). Fold it in or land it first; its SRC1-quiet/SRC2-signalling test belongs in this suite. task-326 (done) was earlier AVX float work. TASK-372 makes the same lowerings mode-dependent, so whichever lands second rebases. TASK-374 owns the integer results of the float-to-integer conversions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An inventory lists every float-producing lowering site and every float-doing helper reachable from compiled code, with its class and verdict, in a doc or a module comment. A ratchet in x86jit-tests (in the style of coverage_ratchet.rs) fails when a new bare float op is added without a classification
- [ ] #2 For each rule (SRC1/SRC2 NaN priority, invalid operation -> indefinite, min/max with NaN and ±0, signed-zero sums, NaN through `cvtss2sd`/`cvtsd2ss`), one test body runs on BOTH backends with operands: QNaN and SNaN in each position, both operands NaN with mixed quietness, 0×inf, inf-inf, 0/0, sqrt of a negative, (-0,+0) and (+0,-0), and denormals
- [ ] #3 Expected values are pinned from hardware: captured through the native oracle on x86-64 Linux and committed as constants. The aarch64 run is therefore checked against x86 truth, not only interp == JIT. On x86-64 Linux the same tests also compare against native directly
- [ ] #4 The suite runs green on the aarch64 runner (.github/workflows/ci.yml:33-37). Predicted defects (i) and (ii) are confirmed or refuted there first, and the outcome is recorded on this task
- [ ] #5 The interpreter's `sse_binop_f32`/`f64` return the indefinite for an invalid operation without a NaN operand, so the interpreter is host-independent for every SSE/AVX float op, and this is documented where `sse_binop_f32` is
- [ ] #6 The FPCR.DN dependency is gone, because the explicit rule covers every NaN result; if any site keeps it, the requirement is stated and debug-asserted at JIT entry
- [ ] #7 Cost: the simd workload is measured against bench/baseline.json and the number recorded. Any regression beyond its band is accepted explicitly here, not silently
- [ ] #8 task-348's AC#2 test runs as part of this suite
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
