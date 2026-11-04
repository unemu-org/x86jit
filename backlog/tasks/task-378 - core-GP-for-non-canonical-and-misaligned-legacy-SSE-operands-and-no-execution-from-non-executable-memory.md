---
id: TASK-378
title: >-
  core: #GP for non-canonical and misaligned legacy-SSE operands, and no
  execution from non-executable memory
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - fault-precision
  - isa-identity
dependencies:
  - TASK-365
priority: medium
ordinal: 414000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The engine does not raise three architectural faults. Read from the code at 96fd5c6.

1. NON-CANONICAL ADDRESSES. In 64-bit mode, a memory reference whose linear address is not canonical (bits 63:47 not all equal) raises #GP(0), or #SS(0) for a stack reference. It is not a page fault (SDM Vol 1 §3.3.7.1, "Canonical Addressing"; AMD APM Vol 2 chapter 5, canonical address form). The JIT's `checked_addr` compares only against the span (codegen/mod.rs:2424-2498), and the interpreter's `region_at` only against the region list (memory.rs:1256-1278). So `mov rax, [rbx]` with a non-canonical RBX returns `Exit::UnmappedMemory`. An OS reports the two differently: Linux delivers #GP as SIGSEGV with si_code SI_KERNEL and no address, and a page fault as SEGV_MAPERR with si_addr. An embedder cannot tell from the exit which one happened. The hot path does not need to change for this. Reject at configuration time any span that overlaps the non-canonical range. Then every non-canonical address already fails the span check, and the classification can live entirely in the cold fault block and in `region_at`'s miss path.

2. ALIGNMENT OF 16-BYTE OPERANDS. A legacy-SSE (non-VEX) instruction with a 128-bit memory operand raises #GP(0) when the operand is not 16-byte aligned. The exceptions are the forms SDM Vol 2A lists as not requiring explicit alignment (movups, movupd, movdqu, lddqu, the SSE4.2 string compares). The explicitly aligned moves (movaps, movapd, movdqa, movntps, movntpd, movntdq, movntdqa) fault in their VEX and EVEX forms too, at the vector width. Source: SDM Vol 2A chapter 2, the AVX/SSE exception specification, "Memory alignment", and its tables of instructions that do and do not require explicit alignment.

The lifter routes aligned and unaligned moves through one helper (lift/mod.rs:1191-1201: `Movdqa | Movdqu | Lddqu | Movaps | Movups | ...` all go to `lift_vmov`), and neither tier checks alignment anywhere. A misaligned `movaps` that crashes on hardware runs on here. The classic cause is a misaligned stack at a call into SSE code, so the same binary behaves differently from real hardware exactly where a bug should surface. The check belongs only in the forms that require it: a test of the low address bits with a cold fault block. AMD's optional misaligned-SSE mode relaxes the legacy-SSE rule; follow the configured guest CPU profile (TASK-359, TASK-388).

3. EXECUTE PERMISSION. The lifter fetches through `code_slice` (memory.rs:1295-1312; lift/mod.rs:237-241), which finds the region and ignores its `Prot`, so a jump into memory mapped `R` or `RW` executes it. On hardware with NX/XD that is a page fault with the instruction-fetch bit set in the error code (SDM Vol 3A, chapter "Paging", access rights and page-fault error code; AMD APM Vol 2 chapter 5, no-execute protection). An embedder needs this in two cases. A guest JIT writes code into RW memory and flips it to RX; executing before the flip must fault. And a wild function pointer into heap or stack should crash where the bug is, not execute data.

DIRECTION FOR 3. Honour the execute bit of registered regions when lifting, opt-in. Many existing fixtures map code `RW`, and `Prot` has been advisory by decision (TASK-330; deferred.md "Memory protection (Prot) is advisory"; the `Prot` doc comment, memory.rs:205-232). Also add an optional embedder fetch-permission callback, consulted when a page is first translated, for embedders that keep protections outside the region list. If the answer changes (an mprotect that removes execute), the page's translations must be invalidated; the SMC invalidation path is the mechanism (spec.md §10). The check runs at translation time, not on the execution hot path. Reads and writes stay advisory, as TASK-330 decided.

EXIT SHAPE.
- For 1 and 2: `Exit::Exception` with vector 13 (#GP) or 12 (#SS), RIP on the instruction, error code 0, nothing committed.
- For 3, the exit must carry both the instruction's RIP and the first non-executable byte, since an instruction can straddle into a non-executable page. It must also distinguish "not present" from "present but not executable". `Exit::UnmappedMemory { access: Execute }` does neither, and deferred.md's suggested `Exit::Exception { vector: 14 }` has no fault-address field. Decide the shape in this task.

DEPENDENCY. The execute check sits in the lifter's fetch, which TASK-365 turns into a fault-tolerant copy. Doing both in one place avoids a second fetch path.

RELATION. TASK-330 (done) kept `Prot` advisory for reads and writes; this task adds execute only. TASK-347 changes the same `checked_addr` fault block (the fault address at the top of the span). Either can land first, but the fault block should end up in one shape. TASK-367 owns #UD and the fetch-fault address for unmapped code.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A load, a store, and a push/pop through a non-canonical address raise vector 13, or 12 for the stack references, on both tiers. RIP is on the instruction and no state is committed. Each test fails at 96fd5c6
- [ ] #2 With a misaligned m128: movaps, movdqa, and a legacy-SSE packed arithmetic op raise vector 13 on both tiers. movups, movdqu, lddqu and the VEX form of the arithmetic op do not. vmovaps does
- [ ] #3 #1 and #2 are checked against a real CPU through the native oracle on x86-64. The oracle must first record the signal and si_code of a non-`hlt` fault; today it records only RIP (x86jit-tests/src/native.rs:169-195)
- [ ] #4 The hot-path cost is measured against bench/baseline.json and recorded on this task. It is zero for ordinary loads and stores, because the canonical check is in the cold fault path, and one test per access for the forms that require alignment
- [ ] #5 With execute enforcement on, on both tiers: a jump into an `RW` region gives the execute fault with the right RIP and faulting byte, a page that later gains execute runs, and a page that loses it stops running at the next dispatch. With enforcement off nothing changes, and prot_advisory.rs still passes
- [ ] #6 A test exercises the fetch-permission callback with a callback that denies one page
- [ ] #7 README "Known gaps", deferred.md "Memory protection (Prot) is advisory" and the `Prot` doc comment state the execute behaviour
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
