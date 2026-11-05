---
id: doc-26
title: SIMD helper-fallback audit — workload-hotness ranking (task-170)
type: other
created_date: '2026-07-13 08:18'
---

# SIMD helper-fallback audit — workload-hotness ranking

**Task:** task-170 (guest-perf / doc-25 Tier-1). **Method:** two cross-verified
inventory passes over `x86jit-cranelift/src/codegen/{mod.rs,vector.rs}`,
`x86jit-core/src/lift/`, `x86jit-core/src/interp/`, and `backlog/docs/compat/coverage.json`.
Each SIMD op classified as **NATIVE** (`builder.ins().*` → host NEON/SSE) vs
**HELPER** (`call_helper` → per-instruction C-ABI call into the interpreter) vs
**MISSING** (not lifted at all → traps). Analysis only, no behavior change.

## Headline (this reshapes doc-25 Tier-1 / task-171)

1. **The SIMD-hot float core is ALREADY native.** Packed/scalar
   `add/sub/mul/div/min/max/sqrt/cmp` (ss/sd/ps/pd), *all* packed-integer arith,
   *all* bitwise incl. `vpternlog`, imm-count shifts, and the common
   shuffles/blends/broadcasts/`pshufb`/`palignr` lower to `builder.ins()` → real
   NEON on ARM. **task-171 as written ("native-lower vmulps/vaddps/… , expect
   2–10×") is largely a no-op — there is no float-arith lever left to pull.** The
   task-169 `simd` microbench confirms it: its region win was only 1.1× because the
   inner `mulps`/`addps` were already native. **Reset the 2–10× expectation.**

2. **The single highest-value SIMD gap is a correctness hole, not a lowering.**
   **Packed float↔int converts** — `cvtps2dq / cvtdq2ps / cvttps2dq / cvtps2pd /
   cvtpd2ps` (+ `cvtdq2pd / cvtpd2dq / cvttpd2dq`) — are **unimplemented and TRAP**.
   They sit in the **x86-64-*v1* (baseline SSE2) `missing` list** of coverage.json —
   i.e. baseline SSE2 coverage, used in SIMD code (per-vertex int↔float,
   colour/normal packing, fixed-point). A guest
   hitting one traps as an unknown instruction *today*. **This is both hot and
   broken → do it first.** Native-lowerable directly: vector `fcvt_to_sint_sat` /
   `fcvt_from_sint` / `fpromote` / `fdemote` (the scalar `cvtsi2ss` etc. already use
   these — `vector.rs:2914-3016`).

3. **The audited workload profile uses SSE4.2 and 128-bit AVX, excluding AVX2,
   FMA3 and AVX-512.** Remaining HELPER ops outside that profile include variable
   shifts (`vpsllvd`), cross-lane permutes (`vpermd/vpermt2`), EVEX-masked families
   and AVX-512 lane ops. FMA (`vfmadd*`) is a hot helper-backed operation for
   workloads that use it, but falls outside this audit's initial profile. Rank
   reachability against the workload's instruction profile below.

## What's already NATIVE (no work — reference)

`builder.ins()` lowering, maps to NEON. (`M` = mod.rs, `V` = vector.rs line refs.)

- **Float arith:** `add/sub/mul/div ss·sd·ps·pd` → `fadd/fsub/fmul/fdiv` (M:2867-2870);
  `min/max` → `fcmp`+`bitselect` (x86 NaN/equal semantics, not IEEE `fmin`) (M:2871-2887);
  `sqrt*` → `sqrt` (M:2857).
- **Float compare:** `cmpps/pd/ss/sd` → `fcmp`+bit-negate (V:2851-2912); `[u]comis*`
  → flags (V:2818).
- **Scalar converts:** `cvtsi2ss/sd`, `cvt[t]ss2si/sd2si`, `cvtss2sd/cvtsd2ss` →
  `fcvt_*`/`fpromote`/`fdemote` (V:2914-3016).
- **Bitwise:** `and/or/xor/andn ps·pd` + `pand/por/pxor/pandn` → `band/bor/bxor`
  (M:2650-2656); `vpternlogd/q` → 8-term SOP (M:2744).
- **Packed int arith:** `padd/psub/pcmpeq/pcmpgt/pmin·pmax(u/s)/pmull(w/d)/pmulhw/
  pmulhuw/pmuludq/pmuldq/padds·psubs(u)/pavg` → native (M:2766-2852); plus
  `pmovsx/zx`, `psign`, `ptest`, `popcnt`, `pmovmskb`, `pextr/pinsr`, `movd/movq`.
- **Imm-count shifts:** `psll/psrl/psra w·d·q`, `pslldq/psrldq` → `ishl/ushr/sshr`
  (M:2598, V:1960).
- **Shuffle/blend/broadcast:** `pshufd/pshuflw/pshufhw/shufps/shufpd/unpck*/vpblendw/d/
  blendvps·pd·pblendvb/pshufb/vpbroadcastb·w·d·q/vpermq/vpermd/vperm2i128/palignr/
  insertps/roundps·pd` → `shuffle/swizzle/splat/bitselect/…` (V:360-2748, M:2364-2713).
- **Opmask (k-regs), movemask, movhlps/lhps/hps/lps/ss/sd** — native.

## Ranked worklist — hot-first (currently HELPER or MISSING)

Legend — **SSE4.2/AVX-128 profile?** = included in the audited SSE4.2/AVX-128 workload profile. **Verdict:** how hard to
make native. Ordered by workload-hotness × reachability.

| # | Op(s) | Gen | SSE4.2/AVX-128 profile? | State | Workload hotness | Verdict | Notes |
|---|-------|-----|------|-------|---------------|---------|-------|
| 1 | **cvtps2dq / cvtdq2ps / cvttps2dq / cvtps2pd / cvtpd2ps** (+dq2pd/pd2dq/tpd2dq) | v1 (SSE2) | **yes** | **MISSING (traps)** | **very high** | **native, EASY** | vector `fcvt_to_sint_sat`/`fcvt_from_sint`/`fpromote`/`fdemote`; scalar forms already do this. **Hot AND broken → do first.** |
| 2 | **shift_reg**: `psll/psrl/psra {w,d,q} xmm, xmm` (scalar xmm count) | v1 (SSE2) | **yes** | HELPER (`shift_reg`, V:772) | medium | native, EASY–MED | broadcast lane-0 count → vector `ishl/ushr/sshr`; clamp over-shift like the imm path. |
| 3 | **dpps / dppd** | SSE4.1 | **yes** | HELPER (`dpps`, V:432/446) | medium (dot/lighting) | native, MEDIUM | `fmul` + imm src-mask + horizontal `faddv` + imm dst-mask. No single NEON op but a fixed sequence. |
| 4 | **pmaddwd** | v1 (SSE2) | **yes** | HELPER (`pmaddwd`, V:2458) | low–med (audio/codec) | native, HARD | i16×i16 → adjacent-pair i32 hadd; needs deinterleave (`swiden`+`iadd_pairwise`-style). |
| 5 | **packsswb / packssdw / packuswb / packusdw** (wide) | v1 (SSE2) | **yes** | HELPER (`vpack`, V:2446) | low–med (format) | native, HARD | interleaved saturating pack; 128-bit `packuswb` already native via clamp+shuffle — extend that. |
| 6 | **vfmadd/vfmsub/vfnmadd/vfnmsub {132,213,231}{ss,sd,ps,pd}** | FMA3 | **NO** | HELPER (`fma`/`fma_mem`, V:2078/2122) | high (outside the profile) | native, EASY | `ins().fma` (single-rounding). **Big lever for AVX2+ guests; irrelevant to SSE4.2/AVX-128 profile.** Do for the general track, not guest-perf. |
| 7 | `vpsllvd/vpsrlvd/vpsravd/…` (per-lane variable) | AVX2 | **NO** | HELPER (`var_shift`, V:741) | med (outside the profile) | native, EASY | vector `ushr/sshr/ishl` with per-lane counts. |
| 8 | `vpermd/vpermps/vpermq(var)/vpermt2*/vshufi32x4/vperm2f128-wide` | AVX2/512 | **NO** | HELPER (permute family) | med (outside the profile) | native, HARD | cross-lane gather; some (`vpermd`) already native, rest need spill/gather. |
| 9 | EVEX-masked packed/logic/shift/blend (`{k}{z}`) | AVX-512 | **NO** | HELPER (`vmasked_*`, `vp_blendm`, …) | n/a for profile | native, HARD | k-mask merge/zero over wide ALU. Deprioritize — outside the audited workload profile. |

### LEGIT fallbacks — keep as helpers (do not lower)

`aes*/sha*/pclmulqdq/gfni·gf2p8` (crypto), `pcmpestr[im]/pcmpistr[im]` (SSE4.2 string
state machine), `maskmovdqu`/`vmaskmov` (per-element fault suppression), `movq2dq/movdq2q`
(MMX↔XMM state bridge), `dpps`-if-deprioritized. No clean NEON equivalent; rare or
inherently sequential. Crypto could later use ARM crypto extensions, but that is a
separate, non-perf-critical track.

## Recommendations (feeds task-171 re-scope + new work)

1. **New task, top of guest-perf: implement packed float↔int converts** (worklist #1).
   It is a *v1/SSE2 coverage hole that traps*, not a perf tweak — higher impact than
   anything in task-171. Native-lowerable, bit-exact vs unicorn, add a microbench
   (int↔float per-lane) to task-169's suite.
2. **Re-scope task-171.** Drop the "native-lower vmulps/vaddps/…" framing (already
   native) and the 2–10× claim. Retarget it at the *actually* helper-backed
   SSE4.2/AVX-128-profile ops: worklist **#2 shift_reg, #3 dpps** (both SSE, both hit real
   SIMD code). Expect single-digit-% whole-program wins, not multiples — the hot
   float loops were never the helper cost.
3. **FMA (worklist #6) → the general (wider-ISA) perf track**, not guest-perf. Cheap
   (`ins().fma`) and high-value for AVX2+ guests, but the audited workload profile excludes
   it.
4. **AVX2/AVX-512 helper ops (#7–#9): leave for now.** Zero SSE4.2/AVX-128 profile reachability; only
   worth lowering if a workload using those extensions appears.

## Provenance

Native inventory: `codegen/mod.rs:512-1311` dispatch → `emit_v_*`/`emit_*` in
`vector.rs`; primitives `emit_fbin` (M:2867), `emit_packed_bin` (M:2766), `emit_vlogic`
(M:2650), `emit_pshufb` (M:2364), converts (V:2914-3016). Helper set:
`Helpers` struct `codegen/mod.rs:44-99`, wired `mod.rs:~3563-3597`, impls in
`x86jit-core/src/interp/`. MISSING converts confirmed: 0 hits in lift/interp/cranelift,
listed under `x86-64-v1/missing` in `backlog/docs/compat/coverage.json`.
