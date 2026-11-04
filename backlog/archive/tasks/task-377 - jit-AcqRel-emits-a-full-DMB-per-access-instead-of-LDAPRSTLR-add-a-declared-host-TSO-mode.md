---
id: TASK-377
title: >-
  jit: AcqRel emits a full DMB per access instead of LDAPR/STLR; add a declared
  host-TSO mode
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - memory-order
  - aarch64
dependencies:
  - TASK-362
  - TASK-357
priority: medium
ordinal: 413000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The spec's §8.2.3 table (spec.md:843) and the `MemConsistency` doc (vm.rs:237-239) describe `AcqRel` as STLR for ordinary stores and LDAPR for ordinary loads (FEAT_LRCPC, with LDAR as the fallback). What `gload`/`gstore` actually emit on aarch64 (codegen/mod.rs:2547-2571):
- after every load under `AcqRel` and `FullTso`, a plain LDR followed by Cranelift `fence`, which is DMB ISH;
- before every store under `AcqRel`, a DMB ISH, then a plain STR.

The codegen test pins exactly that: `tiers_emit_the_right_aarch64_barriers` (codegen/mod.rs:4453-4467) expects AcqRel = 3 DMB for two stores and one load. status.md:165 and task-80 call the leaner form "a future optimization". A full DMB ISH per access makes AcqRel cost about what FullTso costs, while it promises less.

THE LOWERING.
- Loads use LDAPR where the host has FEAT_LRCPC, LDAR otherwise. Stores use STLR. RCpc is the exact fit for x86-TSO. It allows the store→load reordering x86 itself performs (SDM Vol 3A §11.2.3.4) and forbids the others. LDAR+STLR (RCsc) also orders an STLR before a later LDAR, which x86 does not require and which costs more.
- Alignment. Without FEAT_LSE2, a misaligned LDAR/LDAPR/STLR takes an alignment fault. With FEAT_LSE2 an access that stays inside a 16-byte aligned granule may be performed unaligned, depending on a system-control bit the OS sets. Guest accesses are often misaligned, and their alignment is rarely known when the block is compiled. Options:
  - a runtime alignment test with a DMB fallback path, measured;
  - an optimistic acquire/release access that falls back on fault. The fallback needs the per-access trap sites TASK-357 adds, and that is one reason this task follows it. The other is that both edit the same `gload`/`gstore` lines.
- Sizes. Acquire/release forms exist for 8/16/32/64-bit accesses. 128- and 256-bit guest vector accesses (`movdqu`, `vmovdqu`) have none. x86 does not promise single-copy atomicity for them but does order them (§11.2.2), so keep a DMB around them.
- Cranelift. `atomic_load`/`atomic_store` lower to LDAR/STLR on aarch64 and assume an aligned address. Not verified here: whether cranelift 0.115 can emit LDAPR at all, and under which ISA flag. If it cannot, LDAR is the first step, and LDAPR needs an upstream Cranelift change or a lowering of x86jit's own. Record which one.
- `FullTso` emission stays as it is.

THE INTERPRETER IGNORES THE TIER. A Store's `order` is discarded (interp/mod.rs:385-391) and every ordinary access is a Relaxed host atomic (memory.rs:1500-1543). With hotness-gated tier-up a block runs interpreted before it is compiled, so the tier's promise does not hold for the guest. Under `AcqRel` the interpreter should use Acquire loads and Release stores; on an x86 host those are plain moves and cost nothing. Under `FullTso` it should add a SeqCst fence after each store. TASK-362's interpreter rows are the test.

A DECLARED HOST-TSO MODE. Some aarch64 hosts can run a process under a TSO memory model. It is an implementation-defined control set by the OS or hypervisor, and x86jit cannot detect it from user space. An embedder that knows it runs there should be able to say so: a `MemConsistency` variant or a `VmConfig` flag under which ordinary accesses compile as in `Fast` and explicit synchronization stays as it is (mfence from TASK-355, locked ops). Requirements:
- The docs say plainly that this is the embedder's assertion: a false claim gives `Fast`'s behaviour. TASK-362's suite, run on the target host, is how the embedder backs it.
- Hardware TSO does not stop COMPILER reordering. Compiled code is covered by TASK-362's spin-wait and MP tests. The interpreter's Relaxed accesses may legally be reordered by rustc, so its host-TSO path needs at least compiler-only ordering (`compiler_fence`, or Acquire/Release, which are free on an x86 host). Measure the choice.

WHY AN EMBEDDER NEEDS IT. A full native userspace with many threads runs C, C++ and managed runtimes whose ORDINARY accesses synchronize: flag publication, lock-free queues, reference-count handoffs. Those are MP and LB shapes. `Fast` is wrong for them. `AcqRel` at a full barrier per access is the price of `FullTso` for a weaker promise. And on a TSO-capable host, both pay for ordering the hardware already gives.

MEASURE on an aarch64 host, over the bench workloads: `Fast`, `AcqRel` with DMB (today), `AcqRel` with the acquire/release lowering (RCpc, or LDAR if that is what AC#1 records), `FullTso`, and the host-TSO mode. Record the table on this task. bench/baseline.json is an x86 host, where all tiers compile identically, so x86 numbers must not move.

RELATION.
- task-80 (done) introduced the tiers and the barrier-count test.
- TASK-355 adds the fence op that every tier, the new mode included, must keep.
- TASK-362 is the correctness check and must pass for `AcqRel` after this change.
- TASK-357 (trap sites) edits the same `gload`/`gstore` lines; sequence after it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On aarch64, AcqRel ordinary loads of 1/2/4/8 bytes lower to an acquire load and stores to STLR. The load is LDAPR on a FEAT_LRCPC host if Cranelift 0.115 can emit it, and LDAR otherwise; which one is emitted, and why (the Cranelift capability or ISA flag found, or the upstream change or own lowering LDAPR would need), is recorded on this task. tiers_emit_the_right_aarch64_barriers is rewritten to count LDAPR/LDAR/STLR and DMB per tier, not deleted
- [ ] #2 Misaligned and vector-width accesses under AcqRel take a defined path (DMB fallback or split) and never raise a host alignment fault; a test with misaligned guest loads and stores under AcqRel runs on the aarch64 runner, on both backends
- [ ] #3 FullTso emission is unchanged (codegen test)
- [ ] #4 The interpreter honours AcqRel and FullTso for ordinary accesses; TASK-362's interpreter rows for those tiers pass on the aarch64 runner
- [ ] #5 A declared host-TSO mode exists. It is documented as an embedder assertion that x86jit cannot verify, compiles ordinary accesses as Fast does, keeps explicit synchronization, and TASK-362's suite runs under it
- [ ] #6 TASK-362's suite passes under AcqRel on the aarch64 runner after the change
- [ ] #7 Per-workload aarch64 timings for Fast, AcqRel-DMB, AcqRel with the acquire/release lowering of #1 (LDAPR or LDAR), FullTso and host-TSO are recorded on this task; x86 timings are within noise against bench/baseline.json
- [ ] #8 spec.md §8.2.3, the MemConsistency doc (vm.rs:226-243), status.md:165 and glossary.md:40 describe what is emitted, including whether LDAPR or LDAR is used and why
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
