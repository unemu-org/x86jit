---
id: decision-25
title: I286 empty coprocessor socket output policy
date: '2026-11-12'
status: accepted
---

## Context

An I80286 Machine with `FpuModel::None` and MSW.EM clear could execute FNINIT
in real mode, but returned `UnknownInstruction` for every ESC output. Software that
detects an installed extension by initializing it and examining its status or
control word cannot finish its probe. Decision-23 §4 makes the FPU a separate
axis; §5 keeps CPU operand checks and explicit exception delivery on Machine.

**Accepted by the maintainer on 2026-11-12.** This decision gives I286
empty-socket outputs a deterministic board default. Every unpopulated physical
board need not have identical electrical behavior. Decision-24 is separate work.

## Primary-source evidence

Intel **80286 and 80287 Programmer's Reference Manual**, order **210498-005**,
1987, `oracles/intel/i286-prm.pdf`. The following page images were checked
(PDF page numbers are one-based):

| Location | Printed / PDF page | Evidence |
|---|---|---|
| Part I Appendix A, "80286 System Initialization" | A-4 / 200 | The initialization listing explicitly assumes all ones when no 80287 is present. The printed mnemonic is `FSTSW AX`, followed by `OR AL,AL` and `JNZ set_mode`. |
| Part I §10.2.2, Table 10-1 | 10-4 / 182 | EM causes exception 7 on ESC; TS causes exception 7 on extension instructions; MP permits WAIT to cause exception 7 when TS is set. |
| Part I Appendix B, WAIT | B-113 / 321 | WAIT waits for inactive BUSY and checks for a numeric error. |
| Part II Ch. 1, "Hardware Interface" | 1-6 / 356; 1-8 / 358 | PEREQ, PEACK, BUSY and ERROR coordinate the processors. Reserved ports F8h, FAh and FCh carry automatic interface operations; F8h–FFh are reserved, independent of program IN/OUT and IOPL. |
| Part II Ch. 2, "Data Synchronization" | 2-49 / 439 | Even FNSTSW/FNSTCW memory transfers synchronize through an extension request on the Processor Extension Data Channel. FSTSW AX explicitly updates AX before the next CPU instruction. |
| Part II Ch. 3, "Processor Extension Data Channel" | 3-1 / 445 | The CPU's internal DMA-like channel performs all extension-memory transfers under CPU memory protection. Detailed protocol is referred to Ch. 6 of the Hardware Reference Manual. |
| Part II Ch. 3, "Recognizing the 80287 NPX", Fig. 3-1 | 3-2–3-3 / 446–447 | Floating-bus data is undefined. The probe pre-fills a memory word with 5A5Ah, executes FNSTSW, and tests for zero; a subsequent FNSTCW checks selected control bits. |
| Part II Appendix B, "Compatibility Between the 80287 NPX and the 8087", item 10 | B-2 / 490 | Intel's detection sequence stores status to memory without a pre-fill, then loads AX, tests AL and branches to the present case if zero. The board must pull up at least one low-byte data bit when no NPX drives the bus. Thus the memory status store carries the bus value; preserving a previously zero word would falsely detect an NPX. |
| Part II Appendix A, Table A-2; Ch. 1 Table 1-2 | A-2–A-4 / 482–484; 1-6 / 356 | Output encodings, 14/94-byte environment/save images and numeric operand widths used by the witnesses. |
| Part II Appendix A, Table A-2 (continued) | A-5 / 485 | `DF E0` encodes `FSTSW AX`. Waiting forms add the separate CPU WAIT instruction (Part II Appendix B p. B-2). |

Intel **80386 Programmer's Reference Manual**, order **230985-001**, 1986,
`oracles/intel/i386-prm.pdf`, Ch. 11 §§11.1.1–11.1.4, pp. **11-1–11-2**
(PDF 219–220; page images checked): ET selects the 287 or 387 protocol. With no
coprocessor, BUSY# and ERROR# should be tied inactive. ESC tests EM and TS;
WAIT tests MP and TS. These sections do **not** specify absent-extension output
data or memory writes. The 286 supplement's Fig. 3-1 includes the 386, but is
a detection procedure robust to missing output, not a definition of its value.

## Accepted policy

Only on I286 Machine with `FpuModel::None`, after the ordinary EM/TS and opcode
availability checks:

- `FNSTSW AX` supplies **FFFFh** to AX, preserving the rest of its GPR.
- `FNSTSW m16` and `FNSTCW m16` write **FFFFh** after the existing complete CPU
  segment and access checks. They use ordinary Machine memory transfers,
  including A20 routing, faults, MMIO handoffs and retry behavior. Part II
  Appendix B item 10 establishes the memory status transfer with an absent
  extension; Part I Appendix A supplies the all-high bus assumption. The same
  board default applies to the control-word transfer, which Fig. 3-1 probes
  alongside status. Only a high low-byte bit is required by the pullup rule;
  FFFFh is the chosen all-high board default, not a universal electrical value.
- Other memory-output ESC forms retire after the existing complete CPU segment
  and access checks, with **memory unchanged**: FST/FSTP, FIST/FISTP, FBSTP,
  FNSTENV and FNSAVE. This includes the two-byte integer stores. No source
  documents their empty-socket outcomes, so the engine fabricates no write.
  These stores can finish even if the valid segmented destination has no
  mapped backing memory.
- No instruction changes the x87 image, including FNINIT, popping stores and
  FNSAVE. Existing input-form behavior is retained.
- BUSY and ERROR are inactive. WAIT completes unless MP and TS are both set.
  ESC raises #NM (vector 7) when either EM or TS is set. The existing gate
  precedes operand checks; both `InGuest` and `Outward` use it.

WAIT and the following ESC retire separately in real and protected mode.
For example, with CR0=1, `9B DF E0` retires WAIT then stores FFFFh in AX;
with CR0=9, WAIT retires because MP is clear, then the ESC faults with #NM
at its own IP because TS is set. A pending exception in the inert x87 image
does not assert the absent ERROR input.

Preserving the other memory destinations remains an explicit policy choice.
Part II Ch. 2 p. 2-49 and Ch. 3 p. 3-1 describe extension-requested operand
transfers; they do not define the outcomes of those stores with no extension.
Their absence of an outcome cannot override Appendix B's concrete word-status
detection sequence, nor justify synthesizing writes for every other store.

Keep I386 output refusal: its cited PRM defines the absent-pin convention and
exception controls, but does not supply an empty output value under either ET
protocol. Keep Application Real16, Long64, Compat32, Present, `CpuState` and
`jit_abi` unchanged. No new configuration knob or bus-signal emulation is added.

## Rejected alternatives

**All memory unchanged (earlier proposal).** This inferred that an absent
extension could request no transfer, including word status/control stores.
It passes probes that first write a nonzero sentinel, but fails Intel's
Appendix B item 10 sequence when STAT initially holds zero: MOV AX,[STAT]
reads zero and JZ incorrectly selects the present case. The documented pullup
would have no effect if the store did not carry the bus value. The maintainer
rejected this proposal on 2026-11-12 in favor of the word-transfer rule above.

**FFh in every store.** Extending the high-bus word default to FST/FSTP,
FIST/FISTP, FBSTP, FNSTENV and FNSAVE would assume transfer requests and lengths
without a source documenting their absent-extension outcomes. The maintainer
retained memory preservation for those forms instead.

## Witnesses

`x86jit-tests/tests/machine_empty_fpu.rs` pins the chosen images and memory rule
by value. It covers FNINIT, Appendix B's status probe over zero, Fig. 3-1's
pre-filled status probe, the control-word comparison, every output family including
the Present engine's FBSTP gap, exact-end and crossing operands through DS and
SS in real and protected mode, retirement through all three run APIs, inactive
WAIT, word-store handoffs, and #NM precedence/delivery. Waiting AX outputs
retire in both regimes; with TS set and MP clear only WAIT retires, followed
by #NM at the ESC. I386 retains its explicit output refusal.
Existing Application Real16 witnesses and frozen IR snapshots remain intact.
