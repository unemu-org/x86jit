---
id: TASK-357
title: >-
  jit: guest memory accesses are MemFlags::trusted(), so a host fault in
  compiled code has no trap site
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - fault-precision
  - jit
dependencies: []
priority: high
ordinal: 393000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`gload`/`gstore` (codegen/mod.rs:2543-2571) and the locked-op lowerings (codegen/memory.rs:20-90: `atomic_load`, `atomic_rmw`, `atomic_cas`) give every guest access `MemFlags::trusted()`. In the pinned cranelift-codegen 0.115.1 that means `notrap` plus `aligned` (ir/memflags.rs, `trusted()`: "known to be aligned and non-trapping"). Both are promises to the compiler about guest addresses, and since decision-5 neither holds. An in-span hole is a PROT_NONE guard page, so the hardware fault is the mechanism. And x86 data accesses may be misaligned.

What follows, split into what is wrong today and what is latent at 96fd5c6:

1. WRONG TODAY: no access has a trap site. Cranelift records a trap site (code offset plus trap code, `MachBuffer::traps`) only for an access whose MemFlags carry a trap code, and `notrap` means none. Recovery has only the srcloc ranges registered in `codemap` (lib.rs:2697-2712, codemap.rs:33-44). They map a host PC to a guest instruction but not to the access within it that faulted, nor to its direction or width. decision-5 GP-2 reads the direction from the hardware fault frame. That frame is host- and OS-specific (the x86 page-fault error code, the aarch64 ESR) and is not exposed the same way on every OS. The width, and which part of a split vector access faulted, are not available at all. The srcloc RIP is also truncated to 32 bits (TASK-351).

2. LATENT: a dead guest load can be deleted. Cranelift's own predicate treats a `notrap` load as free of side effects: inst_predicates.rs, `is_load_with_defined_trapping`, whose comment says only loads WITHOUT `notrap` "can't [be treated] as side-effect-free even if the loaded value is unused". Guest loads with dead results are common: a null probe or safepoint poll (`test [reg], reg`, `cmp [reg], 0`) whose flags dead-flag elision has narrowed away, or, in region mode, a load into a GPR Variable that is redefined before use (codegen/mod.rs:2782-2783). Today such a load survives only because the 0.115.1 pipeline has no DCE pass (`Context::optimize` runs legalize, unreachable-code removal, constant-phi removal and the egraph) and lowering treats every load as side-effecting (`has_lowering_side_effect`). Nothing in x86jit pins this. A Cranelift upgrade that removes dead `notrap` loads would silently stop guard-page faults for exactly the probes managed runtimes rely on.

3. LATENT: alignment. On x64, Cranelift 0.115.1 lets a legacy-SSE instruction take a memory operand only when the access is marked `aligned` (isa/x64/inst/args.rs, `XmmMemAligned::new` checks `addr.aligned()`), because those encodings require 16-byte alignment. That path is taken when AVX is unavailable to the backend: an old host, or `HostTarget::Baseline`, which switches AVX off (lib.rs:2249-2266). Today guest 128-bit loads are typed `I128` (codegen/vector.rs:4-20) and lowered as two GPR loads, so no fold happens. The first change that loads a guest vector straight into an XMM register, which is the obvious speed-up, will turn a legal misaligned guest access into a host #GP inside compiled code that no recovery path classifies as a guest event.

DIRECTION. Replace `notrap` with a dedicated user trap code on every guest access (`MemFlags::new().with_trap_code(TrapCode::user(..))`, one code per direction if that helps), and set `aligned` only where the guest instruction architecturally requires that alignment AND a preceding check has already raised the guest #GP for a misaligned address. TASK-378 adds that check for the legacy-SSE forms. Until it lands, no guest access carries `aligned`, because a misaligned legacy-SSE operand would otherwise fault in the host (item 3). After each compile, collect the trap sites into a per-site record (host offset, 64-bit guest RIP, access kind, width). Register it next to the codemap entry so that TASK-364's classifier can look it up without a lock and without a linear scan (TASK-352 AC#3). Accesses to CpuState and MemCtx stay `trusted()`: that memory belongs to the engine.

Trade-off: a trapping load is a skeleton instruction to the egraph, as a `notrap` non-readonly load already is, so generated code should barely change. The cost is metadata size and compile time. Measure both rather than assume.

RELATION. TASK-351 (the codemap truncates a guest RIP to 32 bits) must land first or in the same change, because the per-site record needs full 64-bit RIPs. TASK-352 AC#3 removes the linear lookup. decision-5 is the guard-page mechanism this serves. TASK-363 and TASK-364 depend on this task, and TASK-387 owns the separate problem of misaligned locked accesses on aarch64.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No guest-memory access emitted by the JIT carries `notrap`: not gload, not gstore, not the atomic lowerings, not any inline vector access. A test over the compiled output of blocks covering each access shape asserts that every guest access has a trap site with the guest trap code. It fails at 96fd5c6
- [ ] #2 `aligned` appears on a guest access only where the guest instruction requires that alignment and the guest #GP for a misaligned address is checked first (TASK-378); before TASK-378 lands, on no guest access. A test runs misaligned `movups`/`movdqu` loads and stores and a misaligned VEX-encoded arithmetic memory operand (e.g. `vaddps xmm, xmm, [misaligned]`, which has no alignment requirement: SDM Vol 1 §14.9, "Memory Alignment"), on `HostTarget::Baseline` and on `Native`, on both tiers. It matches the interpreter, and the native oracle on x86-64. A misaligned legacy-SSE arithmetic operand (e.g. `addps xmm, [misaligned]`) takes no host fault and is compared with the interpreter only; on hardware it raises #GP(0), which TASK-378's AC#2 covers
- [ ] #3 Every compiled block and region produces a per-site record (host offset, 64-bit guest RIP, access kind, width). A test checks completeness both ways: every trap site Cranelift reports has a record, and every guest access in the IR has a trap site
- [ ] #4 A regression test pins that a guest load whose result is dead is still emitted with a trap site, in two shapes: a `test [reg], reg` whose flags are overwritten in the same block, and a region-mode load into a register that is redefined. The test says it guards against a Cranelift upgrade, because 0.115.1 happens to keep such loads
- [ ] #5 Code size, compile time and run time are measured against bench/baseline.json before and after, and recorded on this task
- [ ] #6 The new tests build in the aarch64 cross-check and pass on the aarch64 CI lane
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
