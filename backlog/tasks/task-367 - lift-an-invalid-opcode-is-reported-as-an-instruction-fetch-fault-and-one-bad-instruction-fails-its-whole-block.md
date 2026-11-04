---
id: TASK-367
title: >-
  lift: an invalid opcode is reported as an instruction-fetch fault, and one
  bad instruction fails its whole block
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - fault-precision
  - isa-identity
dependencies: []
priority: high
ordinal: 403000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two defects in how a block containing a bad instruction is reported. Both were read from the code at 96fd5c6, not run.

A. THE WRONG EXCEPTION. In Long64 and Compat32, an encoding that iced decodes as invalid ends `lift_block` with `LiftError::DecodeFault { addr: insn.ip() }` (lift/mod.rs:250-277; only Real16 delivers #UD, at 267-275). `lift_exit` turns every `DecodeFault` into `Exit::UnmappedMemory { access: Execute }` (vm.rs:1621-1632). On hardware an invalid opcode raises #UD (SDM Vol 3A, "Interrupt 6—Invalid Opcode Exception (#UD)"; AMD APM Vol 2 chapter 8), a fault with RIP on the instruction, which an OS reports as SIGILL. The engine reports a fetch page fault, which an OS reports as SIGSEGV with si_addr at the RIP. Software that probes for an instruction by executing it and catching SIGILL gets the wrong signal: libraries that do not trust CPUID, and JITs that test for an extension, both do this. `Exit::Exception` with vector 6 already exists for `ud2` (exit.rs:36-42).

The same mapping blurs real fetch faults:
- An instruction whose bytes run off the end of mapped code reports `insn.ip()`, not the first unmapped byte that the page fault is about.
- `code_slice` stops at the end of the containing region even when the next region is mapped and adjacent (memory.rs:1295-1312, no region coalescing). So an instruction that straddles two adjacent mapped regions decodes as truncated and is reported as a fetch fault at a mapped address. An embedder that maps the page and retries gets the same fault again.

B. ONE BAD INSTRUCTION FAILS THE WHOLE BLOCK. Any `LiftError` on any instruction of a block fails the whole block (lift/mod.rs:276 and 293-294). That covers `Unsupported` for a lifter gap as well as `DecodeFault`. `resolve` then returns the exit without running anything (vm.rs:1552-1596; the caller at vm.rs:1119-1127 returns at once). The valid instructions before the bad one never run, and the vcpu's RIP stays at the block start while the exit names a later address. So the embedder gets an exit whose `addr` disagrees with the vcpu's RIP, and instructions that look retired were not. Its only way forward is to single-step the prefix itself. `lift_region` inherits this: if the entry block fails, the region fails, and a successor block that fails is silently turned into an exit edge (lift/mod.rs:704-714), then fails again at dispatch.

The engine already handles this correctly in two places. The `NoMoreBytes` window-truncation path ends the block cleanly at the last complete instruction (lift/mod.rs:261-266). Real16 appends the #UD gate after the valid prefix (lift/mod.rs:267-275: "the prefix retires, then #UD faults on this instruction").

DIRECTION.
- When instruction k > 0 of a block fails to lift, drop the ops it already emitted (truncate back to its `InsnStart`) and end the block before it as a fall-through, so the next dispatch starts a block at the bad instruction.
- When the bad instruction is the first one, report it with the vcpu's RIP on that instruction:
  - `Exit::Exception { vector: 6 }` for an architecturally invalid encoding (iced `DecoderError::InvalidInstruction`);
  - `Exit::UnknownInstruction` for a lifter gap;
  - `Exit::UnmappedMemory { access: Execute }` naming the first unmapped byte, for a fetch that runs off mapped code.
- Fetch across adjacent mapped regions instead of stopping at a region boundary.
- `lift_one` (lift/mod.rs:316-372) follows the same rules.

RELATION. TASK-346 covers the test blind spot that hides lift gaps (`jit_eq_interp` passes when the op does not lift); nothing covers this fix. TASK-388 is the companion for encodings that are valid but must fault on the configured guest CPU. TASK-365 turns the fetch into a fault-tolerant copy, and the "first unreadable byte" rule here is the same one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On both tiers, in single-block and region mode: N valid instructions followed by an invalid encoding retire all N, then return `Exit::Exception { vector: 6 }` with the vcpu's RIP and `addr` on the invalid instruction. The test fails at 96fd5c6
- [ ] #2 The same shape with an instruction the lifter does not support retires N and returns `Exit::UnknownInstruction` with RIP on it
- [ ] #3 An instruction whose bytes cross into unmapped memory reports `Exit::UnmappedMemory { access: Execute }` naming the first unmapped byte, with RIP on the instruction. One that straddles two adjacent mapped regions executes
- [ ] #4 On x86-64, the native oracle confirms the class for #1 and #3 (SIGILL versus SIGSEGV). It must first record the signal and si_code of a non-`hlt` fault; today it records only RIP (x86jit-tests/src/native.rs:169-195)
- [ ] #5 spec.md §5.2 and the `Exit` doc comments say which exit each case produces and where RIP is
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
