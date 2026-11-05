---
id: TASK-355
title: 'lift: mfence, lfence and sfence lift to nothing, against spec §8.2.3'
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - memory-order
dependencies: []
priority: high
ordinal: 391000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
READ, not yet measured: the fence arm emits no IR (lift/mod.rs:1146-1147):

    // Fences: single-threaded ordering is already TSO; no-op (§8.2.3).
    Mfence | Lfence | Sfence => Ok(false),

Everything around it promises the opposite:
- spec.md:850: "explicit fences (`mfence`) always translate to real atomics / full barriers (`DMB ISH`), in **every** tier including `Fast`".
- The `MemConsistency` doc (vm.rs:226-243, lines 228-229): "locked ops (`lock`, `xchg`) and `mfence` get real atomics/fences in every tier".
- The `MemOrder` doc (ir.rs:13-16) names "`mfence` lowering" as one of its explicit uses. No lift site produces anything but `MemOrder::None` (lift/mod.rs:1122, :3371), the interpreter discards the field (`IrOp::Store { addr, src, size, .. }`, interp/mod.rs:385-391), and the IR has no fence op.
- status.md:165 ("Locked ops/`mfence` keep their real atomics in every tier"), glossary.md:40, and README.md:90-92 ("x86-TSO memory-ordering is preserved on weak (ARM) hosts").

task-73's own description lists "`mfence` → `DMB ISH`" under Remaining, and status.md:94 still says "Remaining (T4/T5): TSO barrier tiers + `mfence`". The tiers landed in task-80. The fence never did, but the docs were written as if it had.

THE RULE. x86-TSO allows one reordering: a load can pass an earlier store to a different location. SDM Vol 3A §11.2.3.4 ("Loads May Be Reordered with Earlier Stores to Different Locations") is the store-buffering (SB) shape, where both threads can read 0. MFENCE (SDM Vol 2B, MFENCE) forbids it: every load and store before the fence is globally visible before any load or store after it. A locked instruction does the same (§11.2.3.9). Section numbers follow the SDM edition this repository already cites (§11.1.3 for cross-modifying code). Older editions number this chapter 8. AMD APM Vol 2 §7.2 states the same model. Because mfence guards the one reordering x86 hardware itself performs, losing it shows up on ANY host, x86 included.

WHERE THE ORDERING IS LOST TODAY, by reading the code:
- Interpreter, every host. Ordinary accesses are Relaxed host atomics (`atomic_load_le` / `atomic_store_le`, memory.rs:1500-1543), and nothing runs between the store and the load. On an x86 host the hardware store buffer reorders exactly the SB pair.
- JIT on an x86 host, every tier. `gload`/`gstore` emit a plain load or store unless `target_arch = "aarch64"` (codegen/mod.rs:2547-2571), and with no IR op there is nothing to lower to MFENCE.
- JIT on aarch64 under `Fast`. No barriers at all.
- JIT on aarch64 under `AcqRel`. The barrier goes before each store and after each load (codegen/mod.rs:2549-2551, 2562-2564), so the guest's `store x; mfence; load y` becomes `DMB; STR x; LDR y; DMB`. The store→load pair is still unordered.
- JIT on aarch64 under `FullTso`. `STR x; DMB; LDR y; DMB`: the trailing store barrier happens to supply the order. FullTso HIDES the defect, and only in compiled code. With hotness-gated tier-up every block runs interpreted first, and the interpreter is exposed.

WHO RELIES ON IT. GCC lowers a sequentially consistent atomic store to `mov` + `mfence`, and both GCC and Clang lower `atomic_thread_fence(memory_order_seq_cst)` and `__sync_synchronize()` to `mfence`. Dekker- and Peterson-style mutual exclusion, hazard-pointer reclamation and the GC safepoint handshakes of managed runtimes (managed runtime uses mfence there) all have the SB shape and depend on it. An embedder that runs a full native userspace with real threads reaches these paths in libc and in every runtime it hosts, on every host and in every tier.

DESIGN DIRECTION.
- Add an IR op, e.g. `IrOp::Fence { kind: FenceKind::{Full, Load, Store} }`. Either use `MemOrder::Release`/`Acquire` or delete them: today they are dead, and their doc claims a use that does not exist.
- `mfence` becomes `Full` in every tier, on both backends and on every host. In the interpreter that is `std::sync::atomic::fence(SeqCst)`. In the JIT it is Cranelift `fence`, which must come out as `DMB ISH` on aarch64 and as `MFENCE` (or a locked no-op RMW) on x86-64. Check the emitted bytes; do not assume either.
- `lfence` / `sfence`. x86 already orders load→load and store→store (SDM Vol 3A §11.2.2). The exceptions are non-temporal stores and fast-string stores, and SFENCE exists to order those (§11.2.5, SDM Vol 2B SFENCE). x86jit lifts `movnti` (lift/mod.rs:950-953) and `movntdq`/`movntps`/`movntpd` and their VEX forms (lift/mod.rs:1192-1201, 1486-1487) as ordinary stores, so all that is left for lfence/sfence to do is the load→load / store→store ordering the host tier may not give. The rule:
  - no-op where the tier already orders those accesses (an x86 host; aarch64 `AcqRel`/`FullTso`);
  - `DMB ISH` under `Fast` on aarch64. Cranelift IR has one `fence` opcode, a full barrier: on aarch64 it comes out as DMB ISH and on x86-64 as MFENCE. Cranelift has no load-only or store-only fence and no inline assembly. The repository already lives with this: spec.md:844 and vm.rs:240 document `FullTso` loads as "LDR + DMB ISHLD", but `gload` emits `fence()` (codegen/mod.rs:2547-2551), and `tiers_emit_the_right_aarch64_barriers` counts DMB ISH (codegen/mod.rs:4333, 4453-4467). DMB ISH is stronger than lfence/sfence need, and correct. DMB ISHLD / DMB ISHST would need a Cranelift change (a fence kind) or a helper call; that is a later optimisation, not part of this task;
  - in the interpreter, `fence(Acquire)` / `fence(Release)`, which cost no instruction on an x86 host.

  Write the rule at both non-temporal lift sites too. If non-temporal stores are ever lowered to host non-temporal stores, sfence has to become a real store barrier. LFENCE's other role, instruction-stream ordering such as `lfence; rdtsc` (SDM Vol 2A, LFENCE), needs nothing in an emulator that retires guest instructions in order.
- The fence must also be a COMPILER barrier. Cranelift must not move or merge guest loads and stores across it: confirm its alias analysis treats `fence` as clobbering memory, with a test, not by assumption. rustc must not reorder the interpreter's Relaxed accesses across it; a SeqCst fence is enough for that.
- Trade-off. Code without a fence must emit byte-identical code. A fence-heavy guest pays one real DMB/MFENCE per guest fence, which is what the guest asked for. Merging adjacent fences is a later optimisation, not part of this task.

STALE TEXT to correct in the same change:
- lift/mod.rs:1146 comment
- spec.md:850: keep the promise and make it true
- vm.rs:228-229
- ir.rs:13-16
- status.md:94 and :165
- glossary.md:40
- README.md:90-92: under the default `Fast` tier a weak host does NOT give TSO for ordinary accesses. Say so, and say which tier does.
- spec.md:844 and vm.rs:240: `FullTso` loads are followed by DMB ISH, not DMB ISHLD.

RELATION. task-73 (done) recorded this as remaining work. task-80 (done) added the tiers and `tiers_emit_the_right_aarch64_barriers` (codegen/mod.rs:4453), which counts DMBs for ordinary accesses and never had a fence to count. TASK-362 (the litmus suite) depends on this task and is the broader check. TASK-377 rewrites the AcqRel lowering and must keep the fence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The IR has an explicit fence op; mfence lifts to a full fence and lfence/sfence lift by the rule above, which is written at the lift site and at the non-temporal store lift sites
- [ ] #2 A deterministic codegen test: `mov [x],1; mfence; mov r,[y]` compiled for aarch64 has a DMB ISH between the store and the load under Fast, AcqRel and FullTso, and compiled for x86-64 contains MFENCE (0F AE F0) or a locked op between them; under Fast on aarch64, lfence and sfence each emit one DMB ISH between the accesses they order, and under AcqRel/FullTso on aarch64 and on x86-64 they add no barrier
- [ ] #3 The interpreter executes the fence op (SeqCst fence for mfence; Acquire/Release fence for lfence/sfence). A fence changes no architectural state, so a jit == interp comparison passes whether or not a tier drops it (TASK-361). The proof is structural: a lift test asserts that mfence, lfence and sfence each produce the fence op with the right kind, and a test shows that the interpreter's fence arm runs (a test-only counter, or equivalent). #2 (bytes) and #4 (SB litmus) are the behavioural evidence
- [ ] #4 A two-thread SB+mfence test in x86jit-tests runs on the interpreter, the JIT single-block and the JIT with regions, under every tier. It FAILS before the fix on an x86 host and passes after; relaxed-outcome counts before and after are recorded on this task. The same test runs on the aarch64 runner
- [ ] #5 A test shows that a guest load after the fence is not merged with or hoisted above an identical load before it (Cranelift alias analysis respects the fence)
- [ ] #6 Code without a fence is byte-identical: the density and barrier-count tests are unchanged, and the perf gate against bench/baseline.json is within noise
- [ ] #7 lift/mod.rs:1146, spec.md:844 and :850, vm.rs:228-229 and :240, ir.rs:13-16, status.md:94 and :165, glossary.md:40 and README.md:90-92 state what is true after the change
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
