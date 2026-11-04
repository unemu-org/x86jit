---
id: TASK-379
title: >-
  lift: 93 memory-operand forms whose register form lifts, plus vlddqu and
  vpmaskmovd/q
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - isa-identity
  - memory-forms
dependencies: []
priority: medium
ordinal: 415000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
READ at 96fd5c6, not executed. The coverage map lists 93 Codes whose register form lifts and whose memory form does not (the `reg_only` column, backlog/docs/compat/isa-coverage.md:21-23; lists at 953-1053):
- v1, 16 (isa-coverage.md:953-970): packuswb, pmaddwd, pshufhw/pshuflw, psll/psrl/psra{w,d,q} with an xmm/m128 count, sqrtpd/ps/sd/ss;
- v2, 4 (972-977): mpsadbw, pblendw, pmaddubsw, ptest;
- v3, 73 (979-1053): vcvtps2ph to memory, vpabs*, the 256-bit vpack*, vpalignr, vpblendd, vperm2f128/vperm2i128, vpermd/vpermps, vpermilps/pd with a vector control, the ymm vpmovsx/zx forms, vpshufd/hw/lw, the AVX shifts with an m128 count, the variable shifts vpsllv/vpsrlv/vpsrav, vptest/vtestps/vtestpd.
The v3 missing list adds vlddqu and vpmaskmovd/q (isa-coverage.md:251-262).
An optimising compiler folds a load into the instruction whenever it can, so in real code these memory forms are at least as common as the register forms. Each one fails its block with UnknownInstruction where the register form would have run.

Two of them are nearly free. vlddqu belongs in the VEX move arm (lift/mod.rs:1486-1487) the way legacy lddqu already sits in the SSE move arm (lift/mod.rs:1198-1201): architecturally it is an unaligned load. vpmaskmovd/q have the semantics of vmaskmovps/pd with integer element names, and `lift_vmaskmov` (lift/vector.rs:898-934) already lowers those to `VVecMaskLoadMem` / `VVecMaskStoreMem` with masked-off lanes that never fault.

THE ALIASED vpblendw. Its memory form loads the operand into dst and blends dst against it, which is unsound when dst is also src1, so `vpblendw xmm0, xmm0, [mem], imm` is rejected on purpose (lift/vector.rs:2441-2461). The compat probe never makes dst and src1 the same register, so this form is in no list. `lift_vmpsadbw` guards the same trick. The fix needs either a scratch vector register, which the IR does not have (the comment at lift/vector.rs:2445 says so), or the shape the GFNI and PCLMULQDQ memory helpers use: the operand loaded first and passed by value (`exec_gfni_mem`, `exec_pclmul_mem`, interp/mod.rs:4770-4789). Pick one, and use it for every form in this task that would otherwise load through a destination register.

ARCHITECTURE. No new semantics. Each memory form computes what its register form computes, with the memory operand as the last source. Where the memory operand is narrower than the register, the load must be exactly that wide: the count operand of the AVX shifts is m128 even in the ymm form; vpmovsx/zx read 32, 64 or 128 bits; vcvtps2ph stores 64 or 128 bits (the instruction pages in SDM Vol 2B and 2C). A legacy-SSE m128 operand must be 16-byte aligned (#GP(0) otherwise); TASK-378 adds that check, and these forms must go through it once it exists. For vpmaskmov, faults occur only for elements whose mask bit is set; a masked-off element on an inaccessible page does not fault (SDM Vol 2C VPMASKMOV).

RULES TO KEEP. Commit-last (task-305): no architecturally visible write before every fault-capable read has succeeded. Faulting sub-address (task-347): both tiers report the same fault address for an access that crosses into an unmapped page.

RELATION. task-325 and task-320 (done) made these gaps visible (the reg_only column and the memory-operand fuzz leg) and closed with them triaged, but filed no task to lift them. task-333 and task-334 (EVEX write mask and broadcast) cover other forms; the 101 v4 reg_only entries (isa-coverage.md:1055) wait for them and are out of scope here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The v1, v2 and v3 reg_only lists are empty for Long64 and Compat32, vlddqu and vpmaskmovd/q leave the v3 missing list, and coverage.json and isa-coverage.md are regenerated
- [ ] #2 vpblendw and vmpsadbw with dst == src1 and a memory src2 lift and produce what the register form produces on the same values, on both backends
- [ ] #3 cargo xfuzz --mem covers every newly lifted memory form (fuzz menu and coverage ratchet updated) and runs clean for a recorded duration against the native oracle on x86-64
- [ ] #4 Each newly lifted form is replayed with an operand that straddles into an unmapped page: both tiers report the same fault address, and no register or memory changed
- [ ] #5 vpmaskmovd/q: a masked-off element on an unmapped page does not fault; a selected one does, with no partial store before the fault
- [ ] #6 Narrow memory operands (the m128 shift count in the ymm forms, vpmovsx/zx, vcvtps2ph m64) load or store exactly their width: a test places each operand flush against the end of a mapped page
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
