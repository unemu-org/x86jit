---
id: decision-6
title: >-
  Single-block guest registers stay write-through — SSA-Variable promotion
  measured as a regression
date: '2026-07-07 13:43'
status: accepted
---

**Deciders:** Mikołaj Mikołajczyk

## Context

task-105 proposed "full cross-block register allocation": carry the guest register
file as Cranelift SSA `Variable`s in **every** compiled unit (not just loop regions,
which already do this — §12 M5-T3e) and flush lazily, on the theory that the
single-block path's write-through (`store_cpu` on every guest register write) is a hot
round-trip worth eliminating.

Investigated empirically (2026-07-07). Isolated A/B via `x86jit-bench` `gate`, each
variant vs the same the then-current baseline, on the same host — where the shipped
write-through code measures ~0% (fib32 +0.3%, sha256 −1.1%).

## Decision

**Keep single-block guest registers write-through. Do not promote them to SSA
Variables.** The premise is largely false — `CpuState` register stores are L1-cheap and
the host CPU / Cranelift absorb them; the round-trip is not a bottleneck.

Two implementations were measured and reverted:

- **Variables in single blocks (the "full" approach) — REGRESSION.** fib32 +16%,
  sha256 +21%, sqlite/lua +20%+; the perf gate blocked. Cause: Cranelift carries the
  16 guest GPRs as `Variable`s live across the whole block, adding host-register
  pressure (only ~14 usable GPRs on x86-64) with **no reuse to amortize** it.
  Loops/regions amortize the same cost across iterations — which is exactly why region
  mode (M5-T3e) uses Variables and single blocks should not.
- **Write-back cache (defer the stores to a dirty-flush at exit, no Variables) —
  NEUTRAL.** fib32 −1…−3%, sha256 ±3%, all inside the noise bands; gate OK. The
  deferred stores save essentially nothing over write-through.

Neither yields a meaningful single-block speedup, and both would trade away the
single-block **guard-fault GPR precision** (the guard-page SIGSEGV path
`siglongjmp`s past the register flush, so Variable/write-back GPRs go stale in
`CpuState` — pinned by `guarded_single_block_fault_preserves_gpr_ordering`), which
task-31 (unemulinux) (a guest signal frame built from a JIT fault) will want.

## Consequences

- Single blocks keep the `gpr_cache` write-through path unchanged; region mode keeps
  its SSA-`Variable` carry (M5-T3e). Register-in-host-register optimization lives only
  where loop reuse pays for it.
- task-105 is closed **won't-do** with this decision as the record, so the idea isn't
  re-attempted blind.
- **Amended 2026-11-04:** guard-fault GPR precision is now exact in both modes
  (decision-5). Regions keep SSA Variables but synchronize dirty GPRs before each
  inline guest access that can fault, as well as before helpers and exits. Dirty
  tracking is a compile-time CFG analysis; registers clean on every incoming path
  require no stores. Both JIT modes materialize deferred flags before inline accesses;
  a memory RMW's result flags and recovery recipes commit only after its final store
  succeeds (SDM Vol 3A §5.6, access rights; §7.5, fault state). CFG dirty analysis
  removes repeated flag stores on
  paths where they are already materialized. Single-block GPRs remain write-through;
  flag store ordering changes in both modes, with the same RMW rule in the interpreter.
  `x86jit-tests/tests/region_fault_state.rs` contains
  `region_guard_fault_recovers_gprs_and_flags`, which pins the fault state;
  `write_fault_*` pins instruction-entry GPRs/RFLAGS at write-only faults;
  `region_fault_flush_only_dirty_gprs` and
  `wide_transfer_materializes_flags_once_in_both_modes` in
  `x86jit-cranelift/src/codegen/trap_tests.rs`
  pin elimination of clean-register and repeated flag stores.
  This adds stores to region loop bodies. Measured on an aarch64 host against the
  parent commit, three interleaved `x86jit-bench experiment` runs: `hotloop`
  region-bg 7.80 ms -> 7.93 ms (about +1.7%; regions stay 3.3x faster than
  single-block), the long `hotloop` 37.7 ms -> 38.4 ms, `memcpy` region-bg
  unchanged; single-block modes within noise on every workload.
- Real JIT run-side wins should be sought elsewhere — e.g. task-106 (guard pages
  eliminate the per-access bound check), or widening region formation (BGT-6, task-100)
  so more hot code runs where Variable carry already helps.

## Links

- task-105 (closed won't-do) · [[decision-5]] (guard-fault precision residual).
- `x86jit-cranelift/src/codegen.rs` (`gpr_cache` write-through; `gpr_vars` region carry).
- `x86jit-bench` perf gate; the then-current baseline.
