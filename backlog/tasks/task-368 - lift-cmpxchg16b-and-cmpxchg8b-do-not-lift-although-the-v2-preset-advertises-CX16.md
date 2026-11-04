---
id: TASK-368
title: >-
  lift: cmpxchg16b and cmpxchg8b do not lift although the v2 preset advertises
  CX16
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - isa-identity
  - atomics
dependencies: []
priority: high
ordinal: 404000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
READ at 96fd5c6, not executed. `0F C7 /1` has no lift arm. With REX.W it is `cmpxchg16b m128`, on the v2 missing list (backlog/docs/compat/isa-coverage.md:223). Without REX.W it is `cmpxchg8b m64`, which iced tags CX8; `feature_gen` (x86jit-tests/src/compat.rs:49-74) has no arm for CX8, so the compat map does not count it and no list shows it. Yet:
- v2, v3 and v4 set `Feature::Cx16` (features.rs:104-119), projected to CPUID.01H:ECX[13] (features.rs:210);
- every preset, the default included, reports CX8 in the always-on leaf-1 EDX baseline (features.rs:222-234, CPUID.01H:EDX[8]).

A CPUID-dispatched guest therefore takes the double-width path and traps on its first 16-byte compare-exchange: lock-free stacks and queues that pair a pointer with an ABA counter, C11/C++ atomics on 16-byte types, the C runtime's 128-bit atomics. The default preset does not advertise CX16, which is why the corpus stays green; any embedder that selects v2 or above, or presents a real CPU's CPUID (TASK-359), hits it on the first contended data structure.

WHAT EXISTS. `lift_cmpxchg` (lift/integer.rs:510-559) is the model: one `IrOp::AtomicCas` on a memory destination, flags from cmp(acc, old), the accumulator written with the old value. The interpreter's `Memory::atomic_cas` (memory.rs:1362-1379) goes to `atomic_cas_raw` (memory.rs:1457-1487), which handles 1/2/4/8 bytes with std atomics and falls back to a plain read-compare-write when misaligned (memory.rs:1471-1477). The JIT's `emit_atomic_cas` (x86jit-cranelift/src/codegen/memory.rs:70-92) emits Cranelift `atomic_cas` at the operand width. Neither has a 16-byte form:
- Rust stable has no AtomicU128 and the MSRV is 1.75 (Cargo.toml:16). x86jit-core may depend on iced-x86 only (x86jit-tests/tests/boundary.rs:40-60), so a portable-atomic crate is out. The interpreter needs inline assembly (`asm!`, stable): `lock cmpxchg16b` on an x86-64 host; CASP (FEAT_LSE) or an LDAXP/STLXP retry loop on aarch64.
- Cranelift 0.115.1 lowers an I128 `atomic_cas` to `cmpxchg16b` on x64 when the ISA flag `use_cmpxchg16b` is set (cranelift-codegen src/isa/x64/lower.isle, the `$I128` `atomic_cas` rule). x86jit builds its ISA from `cranelift_native::builder()` (x86jit-cranelift/src/lib.rs:2249), which enables `has_cmpxchg16b` when the host reports it (cranelift-native 0.115.1 src/lib.rs:52-53). The aarch64 backend accepts only I8..I64 (`valid_atomic_transaction`, cranelift-codegen src/isa/aarch64/lower/isle.rs:197-202) and has no CASP or LDXP/STXP lowering. So on aarch64 the JIT must call a helper (the same asm routine the interpreter uses) or build the loop by hand. The helper is the smaller change; measure it against an inline sequence before choosing.

SEMANTICS (SDM Vol 2A, CMPXCHG8B/CMPXCHG16B):
- compare RDX:RAX (EDX:EAX) with m128 (m64). Equal: ZF = 1 and RCX:RBX (ECX:EBX) is stored. Not equal: ZF = 0 and the memory value is loaded into RDX:RAX (EDX:EAX). CF, PF, AF, SF and OF are unaffected, unlike `cmpxchg`, which sets all six from the compare. The 32-bit register pair is written only on a mismatch, and in 64-bit mode that write clears bits 63:32 of RAX and RDX.
- "the destination operand receives a write cycle without regard to the result of the comparison": a failed compare writes the old value back. On hardware, a non-writable destination therefore faults even when the compare fails. x86jit keeps `Prot` advisory (TASK-330, done; README.md:115), so here the write-back is observable only through SMC and watch notification, which must fire on both outcomes, as `atomic_cas` already does (memory.rs:1377).
- cmpxchg16b raises #GP(0) when the operand is not 16-byte aligned (64-Bit Mode Exceptions). cmpxchg8b has no alignment requirement. A register destination is #UD.
- with LOCK it is atomic. Without LOCK it is not architecturally atomic, but executing it atomically anyway is allowed and matches `lift_cmpxchg`.

"Atomic" here means atomic against the other tier and against other vcpus, on both host architectures: one vcpu in the interpreter and one in compiled code must contend on the same 16 bytes correctly. That rules out a lock-based emulation unless every inline 16-byte access in compiled code takes the same lock.

cmpxchg8b in 64-bit mode is an 8-byte compare-exchange. It can reuse the existing size-8 `AtomicCas` with EDX:EAX and ECX:EBX combined into one value; the misaligned case goes through the plain fallback above, which TASK-387 deals with.

The #GP(0) depends on a register value, so it is a runtime exit. The JIT has the shape already: the divide helper's #DE path stores RIP and the vector and returns RET_EXCEPTION (x86jit-cranelift/src/codegen/mod.rs:2185-2209). TASK-378 (#GP for misaligned legacy-SSE operands) needs the same exit; whichever lands first builds it.

SIDE OBSERVATION, not in scope. `lift_cmpxchg` writes the accumulator unconditionally, "a no-op on success" (lift/integer.rs:553-558). For a 32-bit operand in 64-bit mode that clears RAX[63:32] on success too, while the SDM pseudo-code (Vol 2A CMPXCHG, Operation) writes the accumulator only on a mismatch. Check it against the native oracle; if hardware follows the pseudo-code, file it. Do not copy the pattern into cmpxchg8b.

RELATION. task-73 (done) is the pattern for lowering locked ops to host atomics in every MemConsistency tier. No task covers this one. TASK-359's advertise ⊆ lift helper reports CX16 and CX8 until it lands.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 cmpxchg16b and cmpxchg8b (with and without LOCK) lift; Cmpxchg16b_m128 leaves the v2 missing list and the compat map counts cmpxchg8b
- [ ] #2 One test body run on both backends covers: a match, a mismatch (RDX:RAX or EDX:EAX loaded, ZF clear, CF/PF/AF/SF/OF preserved), cmpxchg8b clearing RAX[63:32] and RDX[63:32] only on a mismatch, and the write-back on a mismatch (a watched page and a code page are notified on both outcomes; `Prot` stays advisory by TASK-330's decision, so there is no protection fault to check)
- [ ] #3 A cmpxchg16b on an operand that is not 16-byte aligned exits with Exception { vector: 13 } at the instruction on both backends, with no register or memory change; a misaligned cmpxchg8b succeeds; the register form is #UD
- [ ] #4 A contended test (at least two vcpu threads incrementing a 128-bit counter in a cmpxchg16b loop, one interpreted and one compiled) ends with the exact count on x86-64 and on a real aarch64 host (the aarch64 lane TASK-361 adds; until then a recorded run on aarch64 hardware, not only the cross-check)
- [ ] #5 Results match the native oracle on x86-64, and Unicorn as a second witness
- [ ] #6 x86jit-core still depends on iced-x86 only (boundary.rs green); the host-specific paths build under cargo check --workspace --target aarch64-unknown-linux-gnu --tests
- [ ] #7 The cost of the chosen aarch64 JIT path (helper or inline) is measured on a cmpxchg16b loop and recorded on this task
- [ ] #8 The coverage ratchet credits cmpxchg16b and cmpxchg8b through a fuzz arm or a reasoned allowlist entry
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
