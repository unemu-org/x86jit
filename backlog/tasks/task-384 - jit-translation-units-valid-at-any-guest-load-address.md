---
id: TASK-384
title: 'jit: translation units valid at any guest load address'
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - aot
dependencies:
  - TASK-383
priority: medium
ordinal: 420000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
doc-17 B1 plans to make host-side addresses (helpers, link slots) relocatable through a per-unit table, delivered in a new `MemCtx` field (aot-plan.md:61-66). It then accepts this residual: "guest RIP constants / return addresses stay in code, so a persisted unit is valid only for identical guest load addresses" (aot-plan.md:67-69). B1 is not implemented, and no other task implements it: TASK-383 is B0.2, which keeps the slots unchanged (aot-plan.md:59), and TASK-385 comes after this task. So this task does both: part 1 is B1, part 2 removes B1's residual.

PART 1, doc-17 B1: host addresses through a per-unit table. At HEAD every compile bakes host addresses into its code:
- Link slots. `alloc_slot` boxes a fresh `AtomicU64` on every compile and returns its heap address (x86jit-cranelift/src/lib.rs:2585-2590); "its heap address is baked into the code" (lib.rs:2079-2081). `chain_or_link`, `ibtc_or_miss` and `emit_ret_push` emit the slot address as an `iconst` (codegen/mod.rs:3887-3888, 3914-3915, 3969-3978).
- Helpers. Each `Helpers` entry is a signature, an absolute function address and a call-counter address; compiled code reaches the helper by `call_indirect` through the baked address (codegen/mod.rs:42-51).
So two compiles of the same IR differ in their bytes even at the same guest base, and a copied unit would point at another compile's slots. B1 replaces each of these immediates with a load from `table_base + i*8`. `table_base` comes from a new `MemCtx` field appended after `MEMCTX_CODE_RANGE_PTR` (= 120, x86jit-core/src/jit_abi.rs:224), as the append-only rule for `MemCtx` requires (jit_abi.rs:59-62). The dispatcher fills it, and the compiled-function signature does not change. The table layout (helper, slot, constant) is what TASK-385 persists.

PART 2, B1's residual: guest addresses. An embedder that translates a shared library once and maps it at a different base in each process cannot use a unit whose guest addresses are baked in. The base can differ because of address-space randomisation, or simply because of load order. READ at HEAD, the position dependence starts in the IR, before codegen:
- RIP-relative operands. The lifter returns iced's folded absolute address as `Val::Imm(disp)` (x86jit-core/src/lift/mod.rs:3144-3151). Architecturally the effective address is the address of the next instruction plus disp32 (Intel SDM Vol. 2A §2.2.1.6, "RIP-Relative Addressing"). Once folded, the IR cannot tell a guest address from any other immediate.
- `call`. `IrOp::Call` carries `return_addr: u64` as an absolute value (lift/mod.rs:2526-2531). CALL pushes the address of the next instruction (SDM Vol. 2A, CALL), and codegen bakes the same value into the shadow-return-ring prediction (`emit_ret_push`, x86jit-cranelift/src/codegen/mod.rs:3969-3986, `iconst(return_addr)`).
- Direct edges. RIP is stored as a constant before `chain_or_link` (codegen/mod.rs:3884-3906).
- Block identity. The translation cache and the IBTC descriptors key on absolute guest RIPs.
- Fault-site metadata. codemap's srcloc table stores guest RIPs (codemap.rs:33-44).
Also baked: `MemCtx.guest_base`, which the inlined RAM path folds in as a compile-time constant (x86jit-core/src/jit_abi.rs:92-97). That value belongs to the process's memory layout, not to the module, so it too must come from `MemCtx` and not from code.

DESIGN DIRECTION. Every guest address a unit produces is computed as module_base + offset, with module_base read from a per-unit slot. That slot is an entry of the part-1 table, reached through its new `MemCtx` field. The parts:
- IR. A relocatable lift tags guest-address immediates as module-relative instead of `Val::Imm`: RIP-relative operands, return addresses, branch targets. Codegen adds the base, loaded once per unit.
- Keys. Entry tables, chain targets and fault-site metadata are keyed by offset. The embedder supplies the base at load time.
- Edges that leave the module. A call into another module through PLT/GOT is already indirect at guest level and needs nothing. A direct branch whose target is outside the module needs a rule: end the unit there, or emit a typed cross-module relocation. Do not quietly bake an absolute constant.

Cranelift `is_pic` is a separate, host-side question. It stays false today (x86jit-cranelift/src/lib.rs:2223-2226), and doc-17's header decision is that relocatability comes from indirection, not PIC (aot-plan.md:16-27). Measure both, table-load indirection against `is_pic=true` with GOT-relative host references, for code size and run time. Then amend doc-17 with the numbers rather than re-arguing it.

Trade-off: a base-relative guest address costs one add, plus one load of the base per unit, hoisted. RIP-relative accesses are frequent in position-independent guest code. The alternative, keeping the base in a pinned host register, costs a register everywhere. Measure before choosing.

Dependency: TASK-383, because a function owned by `JITModule` cannot be copied to a new location or rebased. Part 1 needs no other task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 (B1) Helper addresses, helper call-counter addresses and every link, IBTC and return-continuation slot address are loads from a per-unit table, not immediates. The table base comes from a new `MemCtx` field appended after `MEMCTX_CODE_RANGE_PTR`, the dispatcher fills it, and the compiled-function signature is unchanged. A test asserts that no slot or helper address appears as an immediate in the emitted code, and the full differential corpus and the jit == interp suites stay green on x86-64 and on an aarch64 host
- [ ] #2 A unit compiled once runs correctly at two different guest bases in one process, with interp == JIT at both bases. It covers a RIP-relative load and store, a direct call/ret pair (both the guest return address and the shadow-ring prediction), a direct jump, an indirect jump through a table, and a fault inside the unit whose reported guest RIP is right at each base. Run on x86-64 and on an aarch64 host
- [ ] #3 No guest address of the unit's own module appears as an immediate in the emitted code. A test compiles the same unit for two bases and asserts that the code bytes are identical
- [ ] #4 A direct branch out of the module follows an explicit rule, and a test covers it
- [ ] #5 The cost of base-relative addressing, and of the part-1 table loads, is measured against bench/baseline.json on both host architectures. The `is_pic` comparison is measured too, and doc-17's header decision is amended with both numbers
- [ ] #6 The non-relocatable path (ordinary JIT) emits the same code as before, or its cost difference is measured and accepted
- [ ] #7 aot-plan.md B1 describes the table that was built, and its "residual bake" paragraph is replaced by what was implemented
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
