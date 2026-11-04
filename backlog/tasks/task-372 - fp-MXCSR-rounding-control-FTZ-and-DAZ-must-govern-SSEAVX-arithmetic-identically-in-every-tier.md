---
id: TASK-372
title: >-
  fp: MXCSR rounding control, FTZ and DAZ must govern SSE/AVX arithmetic
  identically in every tier
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - fp-fidelity
  - mxcsr
dependencies:
  - TASK-358
priority: high
ordinal: 408000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-358 makes MXCSR stored state. This task makes three of its control fields change results: rounding control RC (bits 14:13), flush-to-zero FTZ (bit 15) and denormals-are-zeros DAZ (bit 6). They must change results the SAME way in the interpreter, in the Rust helpers and in compiled code. Otherwise tier-up changes guest results: the same instruction gives one answer while interpreted and another once its block is compiled, depending on hotness and on when the background worker finishes. That is the worst kind of divergence to debug.

READ at 96fd5c6: the default mode is built into every tier.
- Interpreter arithmetic is host Rust f32/f64. `apply_f32`/`apply_f64` (interp/mod.rs:6337, 6364) call `sse_binop_f32`/`f64` (2815-2863). The horizontal forms (`hfloat`, 6030) and FMA (`fma_elem`, 4501) also use host arithmetic. LLVM compiles Rust float code assuming the default floating-point environment (LLVM LangRef, "Floating-Point Environment": round-to-nearest, status flags not observed). So running these paths with the host control register set to the guest mode is NOT sound, because LLVM may fold or move them on the default assumption.
- Explicit round-to-nearest-even is hardcoded in several places:
  - non-truncating converts: `exec_v_cvt_to_int` (interp/vector.rs:4403-4407), `packed_cvt128` (4474-4477), `exec_v_packed_cvt_wide256` (4546-4549);
  - the `round*` immediate's "use MXCSR" bit 2, treated as nearest: interp `vround` (interp/mod.rs:4157-4158) and JIT `emit_round` (x86jit-cranelift/src/codegen/mod.rs:3372);
  - `vcvtps2ph` imm8[2], resolved to nearest at lift time (lift/vector.rs:4668-4671; interp/mod.rs:2919-2921).
- Compiled code:
  - bare Cranelift `fadd`/`fsub`/`fmul`/`fdiv`/`sqrt` (codegen/mod.rs:3540-3588);
  - an explicit `nearest` before every non-truncating convert (codegen/vector.rs:3761, 4308, 4377, 4407, 4476);
  - conversions whose rounding is the host instruction's (`fdemote`, `fvdemote`, `fcvt_from_sint`/`uint`).

  The bare ops follow the host FP control register, and nothing sets it: `call_block` (jit_abi.rs:356-362) enters compiled code in whatever mode the host thread happens to be in.
- Rust helpers called from compiled code do float math: `fma_helper` -> `exec_fma` (x86jit-cranelift/src/lib.rs:1619), `vhfloat_helper` (1340), `cvtps2ph_helper` (1449) and `x87_helper` (145). The x87 Fast transcendentals use f64 libm (state.rs:362-365).
- Blocks that tier up are compiled at `opt_level=speed` (lib.rs:1931-1950), whose egraph pass does constant folding (lib.rs:1897-1898). Whether Cranelift 0.115 folds FLOAT arithmetic was not verified here. If it does, it folds at round-to-nearest.

ARCHITECTURE. RC governs every SSE/AVX floating-point result and the non-truncating conversions (SDM Vol 1 §10.2.3.2; SDM Vol 2A CVTSD2SI and its siblings: "rounded according to the MXCSR.RC bits"). FTZ: when underflow is masked and a result is tiny, the result is a zero with the sign of the true result (SDM Vol 1 §10.2.3.3). x86 detects tininess after rounding. DAZ: denormal source operands are treated as zeros with their sign (SDM Vol 1 §10.2.3.4). Some instructions have their own rules for these modes, for instance the F16C conversions; each one's SDM page is the authority, not a blanket rule.

WHY AN EMBEDDER NEEDS IT. A full native userspace sets these modes on purpose:
- interval and correctly-rounded libm code switches RC;
- managed and media runtimes set FTZ|DAZ at thread start and expect every later operation to honour it;
- libc `strtod` and printf-style conversions produce different digits when FTZ/DAZ are in force.

Today all of these silently compute in the default mode.

DESIGN DIRECTION.
1. The MODE is RC|FTZ|DAZ. The masks and sticky flags are not part of it, because no SIMD FP exception is raised (see 6). The default mode (RC=00, FTZ=0, DAZ=0) runs exactly today's code in every tier and must cost nothing extra.
2. Translations are SPECIALISED to the mode they assumed. The mode becomes part of the block's identity: either add it to `BlockKey` (cache.rs:17-20 already carries `mode: CpuMode` next to the address) or guard it at block entry. Consequences:
   - An `ldmxcsr`/`vldmxcsr`/`fxrstor` (and `xrstor` once lifted) that can change the mode ends the block and leaves through the dispatcher, never through a chain or IBTC edge, so a chain never crosses a mode change.
   - RC-dependent explicit rounding becomes a compile-time choice of nearest/floor/ceil/trunc: the non-truncating converts, `round*` with imm8[2]=1, and `vcvtps2ph` with imm8[2]=1.
   - Float constant folding (Cranelift's egraph, or any fold in the lifter) is allowed only in the default mode. Either show that Cranelift does not fold float arithmetic, or compile non-default blocks at `OptLevel::None`.
3. COMPILED CODE IN A NON-DEFAULT MODE runs with the host FP control register set to the guest mode, and the default is restored before any Rust runs.
   - On an x86 host the register is MXCSR itself.
   - On aarch64 it is FPCR. RMode maps from x86 RC with the two middle encodings SWAPPED: x86 01 = toward -inf and 10 = toward +inf; FPCR.RMode 01 = RP (+inf) and 10 = RM (-inf) (Arm ARM DDI 0487, FPCR).
   - FPCR.FZ without FEAT_AFP flushes inputs AND outputs and detects tininess BEFORE rounding. So FTZ-without-DAZ, DAZ-without-FTZ and x86's after-rounding tininess cannot be expressed by the control register alone. They need FEAT_AFP (FPCR.AH/FIZ) or a software fix-up in the emitted code.
   - Every helper call from a non-default block either runs the software path (4) or switches to the default around the call.
   - A host fault or signal taken inside compiled code must restore the default before Rust runs. x86jit has no in-repo recovery from such a fault today (decision-5 leaves `guarded_run` to the embedder); TASK-364 builds the landing pad. Whichever of the two tasks lands second adds the check (AC#9 here, AC#7 on TASK-364). Do not rely on the host OS resetting the FP control register when a signal handler starts.
4. INTERPRETER AND HELPERS apply the mode in software, and only when the mode is not the default (one predictable branch on the default path).
   - DAZ: map a denormal input to a signed zero before the operation.
   - FTZ: map a result that is tiny after rounding to a signed zero.
   - RC: directed rounding through error-free transformations (exact TwoSum; an FMA-based product, division or square-root residual) or a soft-float implementation.
5. Tests compare the tiers against each other, against the native oracle on x86-64 Linux, and against Unicorn on aarch64. Unicorn models x86 FP in software. Check its FTZ/tininess handling against the native oracle once before trusting it.
6. Sticky exception flags and unmasked SIMD exceptions (#XM) stay OUT of scope. Record that decision in deferred.md by narrowing the MXCSR entry to exactly that.

MAIN TRADE-OFF. Specialising translations keeps the default mode at zero cost and turns RC into a constant. The price: code run under two modes is translated twice, the cache key or an entry guard grows, and mode changes end blocks. The alternative, a mode-agnostic translation that reads MXCSR at run time, would add a load and a branch per float op for every guest, including the overwhelming majority that never leave the default mode.

RELATION. Depends on TASK-358 (storage, accessors, `fxrstor`/`ldmxcsr` semantics). It reopens the deferred.md MXCSR entry. task-326 (done) and task-348 are adjacent FP-fidelity work with no overlap. TASK-374 (integer indefinite) range-checks AFTER this task's rounding, whichever lands first. TASK-381 (ARM-host NaN audit) edits the same lowerings, so rebase whichever lands second. TASK-364's fault landing pad must restore the host FP mode; whichever of the two lands second adds that test (AC#9 here, AC#7 there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One test body runs on BOTH backends under each of the four RC values, for: add/sub/mul/div/sqrt (scalar and packed, f32 and f64), an inexact `cvtsi2ss` from a 64-bit integer, `cvtsd2ss`, `cvtsd2si`, `cvtps2dq`, `roundps` imm8=4 and `vcvtps2ph` imm8=4. Interp == JIT, and both == native oracle on x86-64 Linux. It fails at 96fd5c6
- [ ] #2 FTZ and DAZ, each alone and both together: a denormal input, a tiny result, and a result that is tiny BEFORE rounding but rounds to the smallest normal (the after-rounding tininess case). Interp == JIT == native
- [ ] #3 Tier-up does not change results: a loop that runs one float op in a non-default mode across the tier-up threshold, with the background tier enabled, produces identical bits on every iteration
- [ ] #4 A translation records the mode it assumed. A guest that changes the mode in the middle of a loop sees the new mode from the next instruction on, on both backends, and a test asserts that no chain or IBTC edge crosses the change
- [ ] #5 The guest mode never leaks into Rust: a float-doing helper (FMA, horizontal add, `vcvtps2ph`) called from a non-default block produces the guest-mode result, and the host FP control register equals its pre-`run()` value on every exit path through the dispatcher, every `Exit` variant included (`Exit::Exception` among them). The host-fault path is #9
- [ ] #6 The AC#1-#2 suite runs on the aarch64 runner (.github/workflows/ci.yml:33-37) with interp == JIT and both == Unicorn. The FTZ-only, DAZ-only and after-rounding-tininess cases pass on a host without FEAT_AFP
- [ ] #7 The default mode costs nothing measurable: the simd, hotloop and memcpy workloads are within their bands against bench/baseline.json. The cost of a non-default mode is measured too, and both numbers are recorded on this task
- [ ] #8 deferred.md's MXCSR entry is narrowed to sticky flags and unmasked SIMD exceptions, with the reason. README.md:117 and the code comments that state the default mode (interp/mod.rs:2919-2921, 4154-4155, 6387; lift/vector.rs:4668; ir.rs:1370, 2889) describe the new behaviour
- [ ] #9 If TASK-364's landing pad has landed, its guard-page test also faults inside a non-default-mode block, and the host FP control register equals its pre-`run()` value after the exit, on x86-64 and on aarch64. If it has not landed, record on this task that TASK-364's AC#7 adds the case
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
