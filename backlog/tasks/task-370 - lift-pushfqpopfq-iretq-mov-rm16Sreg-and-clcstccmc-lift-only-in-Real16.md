---
id: TASK-370
title: >-
  lift: pushfq/popfq, iretq, mov r/m16,Sreg and clc/stc/cmc lift only in
  Real16
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - isa-identity
  - flags
dependencies:
  - TASK-356
priority: high
ordinal: 406000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
READ at 96fd5c6, not executed. In Long64 and Compat32 none of these lift:
- `pushf`, `popf` and `iret` (iced's 16-bit mnemonics) have arms gated on `mode.wraps_16()` (lift/mod.rs:2712-2733), whose comment says Long64/Compat32 keep returning UnknownInstruction. The 64- and 32-bit mnemonics `Pushfq`, `Popfq`, `Iretq`, `Pushfd`, `Popfd` and `Iretd` have no arm at all. The compat map lists Pushfw, Popfw, Iretd and Iretw (backlog/docs/compat/isa-coverage.md:123-124, 157, 161) but cannot show Pushfq/Popfq/Iretq: iced tags them X64, which `feature_gen` does not map (x86jit-tests/src/compat.rs:49-74).
- `clc`, `stc` and `cmc` are Real16-gated (lift/mod.rs:2735-2749) and on the v1 missing list (isa-coverage.md:106, 111, 201).
- `mov r/m16, Sreg` is Real16-gated (lift/mod.rs:940-948). In 64-bit mode it falls through to the generic `Mov` arm, where `iced_to_reg` returns None for a non-GPR and the caller turns that into Unsupported (lift/mod.rs:3465-3477). The compat probe cannot template a segment-register operand (x86jit-tests/src/compat.rs:339-341), so these forms count as unencodable, not missing.
- The IR ops behind the Real16 arms (`PushfReal`, `PopfReal`, `IretReal`, `SetCf`) are interpreter-only: they sit in the JIT's `unreachable!` list (x86jit-cranelift/src/codegen/mod.rs:1761-1784) on the grounds that Real16 never reaches the JIT.

Ordinary 64-bit user code uses every one of them. clc/stc/cmc appear in hand-written multi-precision arithmetic. pushfq/popfq save and restore flags around code that clobbers them, and toggle EFLAGS.ID (bit 21) and AC (bit 18) to identify the processor (SDM Vol 1 §21.1.2). iretq is a same-privilege return in signal trampolines and user-mode context switches, because it restores RIP, RSP and RFLAGS in one instruction. A segment-register store reads the current selector, for example to save a complete context. An embedder that runs a full native userspace with signals and threads meets all of them.

ARCHITECTURE:
- PUSHF in 64-bit mode pushes 8 bytes of RFLAGS with VM (bit 16) and RF (bit 17) cleared in the image; with a 66 prefix it pushes 16 bits (SDM Vol 2B PUSHF/PUSHFD/PUSHFQ).
- POPF at CPL > 0: IOPL unchanged; IF changes only when CPL ≤ IOPL; RF is 0 afterwards; VIP, VIF and VM unchanged; AC and ID writable only with a 32- or 64-bit operand; reserved bits and RFLAGS[63:32] unaffected; insufficient privilege raises no exception, the privileged bits simply do not change (SDM Vol 2B POPF/POPFD/POPFQ and its Table 4-20; Vol 1 §3.4.3.3).
- IRET in 64-bit mode pops RIP, CS, RFLAGS, RSP and SS unconditionally (SDM Vol 3A §7.14.3; Vol 2A IRET/IRETD/IRETQ). In scope: a return to the same privilege and the same code-segment mode. Anything else must fault (#GP with the selector as error code). A Vm runs in one CpuMode (vm.rs:314-318), so an iretq whose CS would switch to compatibility mode cannot be honoured and must not silently continue.
- MOV r/m, Sreg stores the 16-bit selector: to a register it zero-extends on every Intel 64 processor, to memory it writes 16 bits whatever the operand size (SDM Vol 2B MOV). Loads into segment registers stay out of scope; they need descriptor checks.
- CLC, STC and CMC change CF only (SDM Vol 2A CLC, CMC; Vol 2B STC).

STATE THE ENGINE LACKS. `Flags` holds CF, the PF/AF sources, ZF, SF, OF, DF and an IF that only Real16 uses (state.rs:135-167). TF, IOPL, NT, AC and ID are not modelled, and `if_` took the last free pad byte (state.rs:150-158). An RFLAGS image needs somewhere to keep the system bits a guest can write and read back (AC and ID; TF stored but not acted on), so CpuState grows an appended field, as the real-mode selectors and `fpu_empty` did (state.rs:366-385). CpuState has CS/DS/ES/SS selectors for Real16 (state.rs:373-376) but none for FS and GS. The selector values a 64-bit or compat user process sees, and the user-mode RFLAGS policy (IF = 1, IOPL = 0, for example), come from embedder configuration: they are facts about the guest OS's descriptor tables and its privilege setup, not about x86. The JIT already assembles the arithmetic flags for syscall's R11 in `assemble_rflags` (x86jit-cranelift/src/codegen/control.rs:158-184); pushfq extends that with the system word.

FLAGS AND ELISION. `op_reads` (lift/mod.rs:774-784) lists Branch, GetCond, Adc, Sbb, Rcl and Rcr, and its comment says no pushf exists. pushfq reads all six status flags and must be listed; popfq writes all six; cmc reads CF; clc and stc write CF only. Without that, `add; pushfq; sub` pushes stale flags, because the sub makes the add's flags dead. `PushfReal` is not in `op_reads` either, so the Real16 path already carries the same hazard. TASK-356 owns exact flags at a fault and the lahf case; this task depends on it so that pushfq, cmc and lahf share one corrected liveness model.

TF. Storing TF without delivering the single-step #DB is a documented gap; single-step delivery is out of scope.

RELATION. No open task. Completed task-219 and task-220 discuss pushf observing AF and PF, the lazily kept flags that `assemble_rflags` derives.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 pushfq, popfq, iretq, the 66-prefixed pushf/popf, clc, stc, cmc and mov r/m16|r32|r64, Sreg lift in Long64, and their 32-bit forms in Compat32; each has a test whose one body runs on both backends
- [ ] #2 A pushfq after an ALU op whose flags a later op overwrites pushes the ALU op's flags on both backends, and a cmc between a flag producer and a faulting store leaves the complemented CF visible at the fault. Negative control, tied to the case-A strategy TASK-356 chose: in every case a unit test asserts that `op_reads` returns all six status flags for the pushfq op and CF for cmc. Under TASK-356's option 2 (narrowing kept on the hot path), both behavioural tests must also fail with pushfq and cmc removed from op_reads. Under option 1 (every op that can exit reads all six flags), the faulting store and pushfq's own stack store keep the producer's flags live whatever op_reads says for pushfq and cmc, so the unit test is the only control. Record which case applied
- [ ] #3 popfq at CPL 3 with an image that sets IOPL, IF, VM, RF and reserved bits changes only the bits SDM Vol 2B Table 4-20 allows; AC and ID round-trip through popfq/pushfq; a 66-prefixed popf cannot change AC or ID
- [ ] #4 iretq to the configured same-privilege CS/SS restores RIP, RSP and RFLAGS; an iretq to any other CS or SS exits with #GP and changes no state
- [ ] #5 pushfq/popfq round trips, clc/stc/cmc and the selector stores match the native oracle on x86-64 (with the embedder configured with the host process's own selectors and RFLAGS policy)
- [ ] #6 No unreachable! in the JIT is reachable from a Long64 or Compat32 block; the new ops have JIT lowerings
- [ ] #7 The compat map counts these forms (X64-tagged codes and segment-register operands in scope), and README's Known gaps says what the RFLAGS image models (TF stored, no single-step)
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
