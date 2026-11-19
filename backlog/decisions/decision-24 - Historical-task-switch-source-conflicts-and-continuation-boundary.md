---
id: decision-24
title: Historical task-switch source conflicts and continuation boundary
date: '2026-11-12 07:07'
status: accepted
---

## Context

Decision-23 licenses model behaviour through the manufacturer's manual. The
pinned 80286 PRM (210498-005), 80386 PRM (230985-001), and SDM (325462-092US)
contain conflicting task-switch statements. The PRM page images cited below were
inspected, including both models' #TS descriptions and the LTR instruction.
The maintainer accepted this record; the ROM clarification and continuation
ownership contract below are incorporated into the accepted policy.

## Source-resolution rule

For a conflict inside one model's manual, use the statement supported by the SDM
only if the SDM's own relevant statements agree. If the SDM is silent or internally
inconsistent for that condition, prefer the model's more specific instruction or
exception text over a general summary. A later SDM difference alone does not
replace an unambiguous historical model rule. Record all counter-statements by
section and page. No emulator implementation supplies semantics.

The audit distinguishes the descriptor's access privilege from incoming CS/SS/data
privilege, and descriptor validation from state loaded after a switch. An apparent
agreement for one condition does not settle a different condition on the same page.

## Row-by-row audit

| Case | Selected rule and audit result | Conflicting statements and reason |
|---|---|---|
| Incoming NT on JMP | **Changed: load NT from the incoming TSS FLAGS image**, on I286 and I386 with either supported TSS format. 286 PRM App. B SWITCH_TASKS p. B-12 loads all machine registers; 386 PRM §7.5 step 5 p. 7-8 explicitly loads the flag register. SDM Vol 3A §10.3 step 12 p. 10-11 and §10.4 Table 10-2 p. 10-15 explicitly restore incoming NT on JMP and IRET; CALL/INT set incoming NT. | 286 PRM §8.4 Table 8-2 p. 8-7 and 386 PRM §7.6 Table 7-2 p. 7-10 say incoming NT is "Cleared" on JMP; SDM Vol 3A §10.4 prose p. 10-15 also says NT=0. The maintainer-approved resolution below selects the FLAGS-load procedures and explicit SDM step/table. Outgoing NT, busy-bit effects and back-link effects are unchanged. |
| Incoming TSS limit fault context | **Retained:** 286 checks after saving and loading TR, before loading registers/setting TS (App. B p. B-12). 386 checks before saving (§7.5 steps 2–3 p. 7-8). | 286 §8.3 pp. 8-5–6 gives an overview and separates the limit check from later selector validation; its table does not identify a different TR-load checkpoint. SDM Vol 3A §10.3 step 3 p. 10-10 describes the later old-context check. Use the detailed model procedure. |
| Minimum inclusive TSS limit | **Retained: 2Bh / 67h.** 286 App. B SWITCH_TASKS p. B-12 and §9.6.4 Table 9-5 p. 9-12 give 43; 386 §§7.2,9.8.10 pp. 7-4/9-14 give 103 for a 32-bit TSS. | 286 Table 8-1 p. 8-6 says greater than 43. SDM Vol 3A Table 10-1 p. 10-13 says 44/108; §10.6 p. 10-18 says 2Ch; §7.15 Table 7-6 p. 7-39 says 2Ch/67h; §10.2.2 p. 10-6 explicitly says 67h. The SDM is internally inconsistent and cannot break this tie. Use the specific model instruction/#TS text and the 44-/104-byte formats. |
| 286 segment-load checks | **Retained:** #TS for invalid selectors/type/privilege; SS before CS. App. B pp. B-12–13 and §9.6.4 Table 9-5 p. 9-12. | Table 8-1 p. 8-6 lists #GP/#SS for some of these failures. SDM Vol 3A Tables 7-6/10-1 pp. 7-39/10-14 consistently support #TS for these selector conditions. Presence still gives #SS for SS and #NP for code/data. |
| 386 LDT failure error code | **Changed: #TS(LDT selector)**. §9.8.10 Table 9-5 p. 9-14. | Table 7-1 p. 7-9 names incoming TSS. SDM Vol 3A Tables 7-6/10-1 pp. 7-39/10-14 both name the LDT, resolving the model conflict in favour of its exception description. |
| 386 invalid stack/data selectors | **Changed: #TS(selector)** for invalid selectors/types and privilege mismatches; retain #SS/#NP for not-present stack/data. §9.8.10 Table 9-5 p. 9-14 and §9.8.12 p. 9-16 identify the task-switch #TS exception. | Table 7-1 p. 7-9 uses #GP for selector/readability failures and #SS for SS DPL. SDM Vol 3A Tables 7-6/10-1 pp. 7-39/10-14 agree on #TS for invalid selectors/type/privilege. |
| Segment check ordering | **Model-specific:** 286 checks privilege before presence (p. B-13). 386 CS presence test 7 precedes privilege test 8; analogous presence tests 10/15 precede privilege tests 11–12/16 (Table 7-1 p. 7-9). | Table 9-5 specifies vectors and selector codes, not a replacement ordering. SDM Table 10-1 p. 10-13 explicitly identifies its ordering as P6-specific. It does not replace the historical ordering. Combined-failure witnesses pin the selected order. |
| 386 CALL task/gate DPL and target IP | **Changed to #GP**. §9.8.13 pp. 9-16–17 covers privilege and code limits; Appendix E Table 8-1 notes k/r p. E-15 explicitly names CALL. | §17 CALL p. 17-41 says #TS. SDM Vol 2A CALL p. 3-134 and Vol 3A §10.3 p. 10-10 agree on task/gate DPL #GP; CALL p. 3-134 and the exception list p. 3-135 agree on target-IP #GP. These conditions have no opposed SDM #TS statement. |
| 386 CALL/INT invalid or busy target TSS | **Retained: instruction-specific #TS** (§17 CALL/INT pp. 17-41/17-72). JMP uses #GP (§17 p. 17-86); IRET uses #TS (§17 p. 17-76). | PRM Table 7-1 p. 7-9 / §9.8.13 item 11 p. 9-17 give #GP for busy tasks. SDM Vol 3A Table 10-1 p. 10-13 and Vol 2A CALL/INT pp. 3-134/3-471 give #GP, but SDM §7.15 Table 7-6 p. 7-39 also lists #TS for null/out-of-table/local/nonavailable target TSSs, including lock attempts. The SDM cannot settle these conditions; keep the more specific model instruction branch. This does not change the separate DPL rule above. |
| IRET departing busy-bit clear | **Corrected: read the incoming image and finish its descriptor loads before clearing the departing B**, then check incoming IP. 286 IRET p. B-51 and SWITCH_TASKS pp. B-12–13; 386 IRET p. 17-76 and §7.5 step 5 p. 7-8. Separate return witnesses cover both 16-bit model formats and the 386 32-bit format. | The generic 286 SWITCH_TASKS p. B-12 clears old B before reading the incoming task when not nesting. The specific IRET branch p. B-51 instead places the abandoned-task clear after SWITCH_TASKS. SDM Vol 3A §10.3 steps 9–10 p. 10-11 consistently supports reading the incoming image before clearing old B, resolving this conflict in favour of the specific IRET branch. The SDM places descriptor qualification later (step 14 on the same page); the historical PRMs place it inside SWITCH_TASKS. Retain the historical instruction order for those loads, including a segment fault before the departing busy clear. |
| Direct 286 JMP nesting | **Retained: no nesting**, §8.4 Table 8-2 p. 8-7 and JMP description/task-gate procedure p. B-57. | Direct-TSS JMP pseudocode p. B-58 says with nesting. SDM Vol 3A §10.4 p. 10-15 consistently says JMP does not write the backlink or nest. The NT value disagreement is recorded separately above. |
| CR3 with paging disabled | **Retained: leave CR3 unchanged, do not read its TSS field**, 386 §7.1 p. 7-3's explicit conditional read rule. | §7.5 step 5 p. 7-8 lists PDBR among loaded registers without specifying PG=0; it does not explicitly require a read with paging disabled. SDM Vol 3A §10.3 step 9 p. 10-11 reads the field but does not load CR3. This later difference does not override the model's specific conditional rule. A witness leaves the CR3 field unmapped. |
| 286 LTR versus outgoing-TSS limit | **Retained LTR rule; corrected switch ordering.** LTR p. B-72 has no minimum-limit check. SWITCH_TASKS p. B-12 acquires the target's busy bit before checking the current cached limit >=29h. | No contradictory LTR limit condition is stated on p. B-72. The incoming TSS must separately satisfy 2Bh. The witness executes LTR with 28h, then checks #TS(new TSS), target busy, unchanged outgoing image and MSW. |

The 386 PRM §7.5 step 1 p. 7-8 reverses the DPL inequality in prose. Its §7.4
p. 7-5 and CALL/JMP instruction procedures, supported by SDM Vol 3A §10.3 step 2
p. 10-10, require descriptor DPL >= max(CPL,RPL).

The 386 INT procedure p. 17-72 describes a two-byte error push even for a 32-bit
task. Use the destination TSS's width, consistent with the 32-bit frame in §9.6.1
Fig. 9-5 p. 9-8, interrupt tasks §9.6.2 p. 9-10, and SDM Vol 3A §§7.12.2–7.13
pp. 7-18–19. Both TSS widths have error-push witnesses.

For a 16-bit TSS on I386, upper GPR halves are not preserved (SDM Vol 3A §10.6
p. 10-18); the engine zeroes them deterministically. Reserved upper selector words
in a 32-bit TSS are left untouched. These are engine storage policies rather than
claims about unspecified bus writes of every stepping.

## Incoming NT on JMP clarification

2026-11-12: The maintainer approved restoring incoming NT from the TSS on JMP
for I286 and I386, with both supported TSS formats. This supersedes the earlier
clear-on-JMP selection in the row above; this decision remains **accepted**.

A diagnostic program validated on 286 hardware CALLs task B from A through a
task gate, leaving B with NT=1 and a back link to A. B and C then JMP to each
other about fifty times. B saves FLAGS=4046h when departing and must restore
that image when resumed so that its final IRET returns to A through its back
link. Clearing incoming NT yields 0046h and a stack IRET over fill data. The
hardware-validated program falsifies the clear-on-JMP reading; it is an oracle,
not the authority for the selected semantics.

The licence is the incoming register/FLAGS load in 80286 PRM App. B
SWITCH_TASKS p. B-12 and 80386 PRM §7.5 step 5 p. 7-8, together with SDM
Vol 3A §10.3 step 12 p. 10-11 and §10.4 Table 10-2 p. 10-15's explicit
incoming-NT rule. Each cited page image, including the opposed historical tables
and SDM prose recorded in the row, was checked. This is a specific approved
resolution of the conflicting NT statements; the other source-resolution rows
stand. CALL and INT, including task gates and IDT task gates, still set incoming
NT=1. IRET still restores incoming NT as saved and clears NT only in the outgoing
saved FLAGS image. JMP preserves outgoing NT; its old/new busy-bit updates and
both back links are unchanged.

## Continuation and commit boundary

Preflight the A20-gated access bytes of every descriptor that will receive a CAS.
Device (`Trap`/MMIO) bytes produce `LegacyInstruction` before target acquisition
or TSS writes: separate MMIO acknowledgments cannot implement an atomic RMW
(the xj-srh.8 contract). ROM accepts the locked operation and discards its write,
including busy and accessed updates, LTR, and ordinary GDT/LDT segment loads.
Caches are installed normally; a later descriptor check reads the byte ROM
actually holds. SDM Vol 3A §3.4.5.1 p. 3-13 makes accessed-bit setting conditional
on descriptor memory supporting processor writes; non-writable descriptor storage
is a software concern. This restores the shared helper's earlier ROM behaviour
and supersedes the ROM-refusal interpretation introduced in round 4.

Incoming-task reads occur after busy acquisition and the outgoing save; physical
aliases and A20 aliases therefore observe the just-saved state. Keep progress
through save, incoming reads, descriptor loads and error-code pushes in private
per-vCPU execution state. Preserve progress if mappings change during a handoff
and a later device update must be refused. CAS failures use the observed byte
when retrying, preserving competing descriptor edits.

A task-switch continuation and its accepted event have one ownership rule: any
embedder change to `CpuState` between an exit and its resume abandons both, except
what that exit explicitly asked the embedder to supply. This includes all GPRs,
flags, SIMD/x87 state, system state (including CR2), configuration and retry state;
only the requested MMIO read value or write acknowledgment is permitted to differ.
Mapping repairs, IRQ injection and queued A20 requests are outside `CpuState`.
Compare the returned state after engine retry bookkeeping, without raw padding or
hash comparisons. Exhaustive field coverage must be extended when state is
appended. The edited architectural state stands, the abandoned instruction does
not retire, and an already-consumed event is never delivered again.

On a nonzero-budget entry, decide resume versus abandon before deriving an entry
context. Abandonment clears switch, event and instruction progress together,
discards their MMIO completion bookkeeping, and releases queued A20 changes even
if the edited state's unfinished CS/SS caches then cause an entry refusal. If the
accepted IRQ has not reached commit, abandonment removes that queued instance;
a fresh IRQ queued after the accepted one was consumed remains independent.
Event abandonment releases the same private boundaries as completion: STI and
MOV/POP SS inhibition, the delayed TF boundary and the pending accepted event.
Architectural TF in the edited CPU state remains unchanged. Event redirection and
nonresumable delivery exits use the same boundary cleanup; a retained delivery
continues to own its inhibition until completion or abandonment.
Zero-budget probes preserve progress and completion inputs. This rule belongs to
the task-switch continuation; it does not extend other Machine continuations.

Forking a VM or copying `CpuState` to another vCPU while any Machine mid-instruction continuation is pending is unsupported and tracked separately.

All engine refusals from unfinished switch progress retain the continuation,
independently of its current step. This includes incoming-image, LDT, every segment
and the return busy update after commit. Architectural exceptions abandon the
failed switch and use its reached task context. Error-push bus handoffs retain
progress. The completed T-bit refusal is generated after the continuation has
finished: the instruction has retired once, or the event has finished delivery.
Clearing the recorded T bit permits the next instruction without replaying a
completed switch.

Pending switch progress blocks new event acceptance even without an MMIO retry
ledger entry. An interrupted CALL completes and retires before a queued IRQ is
accepted. An already accepted event may finish its own switch continuation. At task commit,
its IRQ is consumed once and the retained event owns delivery through handoffs and
refusals; a subsequently queued IRQ, including the same vector, remains separate.
Completion finalizes event shadows/cache synchronization even when reporting the
T-bit refusal. Protected-mode architectural delivery faults also consume the
accepted IRQ; the existing real-mode Outward retry policy is unchanged.
A20 remains fixed through the continuation as decision-22 requires.

Before commit, faults restore the outgoing CPU image while retaining memory
side effects prescribed by the model (notably B set before the 286 short-current-
TSS fault). After TR installation, retain incoming state and completed cache
loads. A 286 short-incoming-TSS fault has changed TR and saved the outgoing state,
but has not loaded incoming registers or set TS. Set TS after that load; the 386
sets it at its earlier TR-load step (§7.5 p. 7-8). Unsupported incoming CPL/VM is
reported after acquisition/save/load, retaining the reached task state.

Task-switching software INT preserves incoming RF, like JMP/CALL/IRET (386
§12.3.1.1 p. 12-7), including both immediate completion and resumed completion.
The incoming T bit is recorded and refused after completing the switch, because
DR6.BT support is absent (§12.3.1.5 p. 12-8). The I/O-map base is read; bitmap use
is excluded by the supported CPL-0 and non-V86 scope. Paging remains refused.

Application lowering, frozen Real16 snapshots, CpuState's existing field offsets,
and jit_abi are unchanged. Revisit this policy when more specific model
errata or primary-source evidence becomes available, and the private continuation
representation when unified Machine lowering lands.

## Amendment — 2026-11-18: privilege transitions (xj-srh.9, first landing)

Accepted by the maintainer. The I80286 and I80386 Machine models run protected-mode code at
CPL 1–3. The design is in backlog/docs/design/privilege-transitions.md, and the rules are in
spec.md §17.10. The first landing covers:
- inward interrupts and exceptions with the stack switch from the TSS;
- outward RETF, RETF n and IRET;
- task switches into rings 1–3;
- the CPL and IOPL checks that become reachable at CPL > 0.

Call gates are the second landing. Until then they keep the typed refusal
`LegacyGateOrTaskTransfer`.

**Second landing (2026-11-18).** CALL and JMP through a call gate in the GDT or the LDT are
implemented on both generations:
- 286 gates on both models, and 386 gates on the 80386 only (#GP(selector) on the 80286);
- inward calls switch to the stack from the TSS and copy the gate's parameter count, words or
  dwords, following O1, O7 and O11;
- same-level calls and conforming targets;
- JMP through a gate goes to the same privilege level only.

`LegacyGateOrTaskTransfer` is no longer produced. A task gate or a TSS named by CALL or JMP is a
task switch, as before.

The source-resolution rule above settles these conflicts between the 80286 PRM, the 80386 PRM and
the SDM. Each counter-statement is recorded in spec.md:
- **O1. Stack room on an inter-level CALL** (second landing): #SS(0) on the 80286, #SS(new SS)
  on the 80386.
- **O2. Stack room on an inward interrupt:** #SS(new SS + EXT), and the room includes the error
  code, on both generations.
- **O3. A return SS that is not present:** #SS(SS selector) on both generations. The 386
  instruction pages' #NP is recorded.
- **O4. Return-SS selector, type and privilege failures:** #GP(SS selector). The 80286 Table
  7-4 "SF" is recorded.
- **O5. Two single-page outliers:**
  - a conforming return CS on an outward IRET needs DPL ≤ RPL;
  - a null new SS on an inward interrupt is #TS(EXT).
- **O6. Data-segment nulling:** the SDM rule. A register whose cache is data or non-conforming
  code with DPL below the new CPL becomes null.
- **O7. The call-gate target CS presence check** (second landing): kept, in the SDM order.
- **O8. Interrupt target CS checks:** the order that landed with xj-srh.6 is kept.
- **O9. RET n** also adjusts the outer stack.
- **O10. The CPL or IOPL check** comes before the memory operand. ASSUMED: no source orders
  them.
- **O11. Parameter copy** (second landing): every parameter is read before the first write.
- **O12. LOADALL with a CS DPL of 1–3** stays refused.
- **O13. Retained contexts:** the bound of 16 stays.
- **O14. The CPL and IOPL instruction checks** land here. xj-srh.10 keeps only the 80386 I/O
  permission bitmap.

The review round added four more points:
- **D-1.** On the 80286, a LOCK-prefixed instruction at CPL > IOPL raises #GP(0) (286 PRM
  p. B-68). The 80386 has no such rule.
- **D-2.** On the 80386, an I/O access at CPL > IOPL with a 286 TSS in TR raises #GP(0) (386 PRM
  p. 17-15). With a 386 TSS it is the typed refusal `LegacyIoPermissionBitmap` until the bitmap
  lands.
- **D-3.** REP INS/OUTS with a zero count makes no I/O test (SDM REP Operation).
- **D-4.** An IRET image with VM = 1 at CPL > 0 stays refused. The 386 PRM p. 17-74 and SDM
  p. 3-492 conflict here; spec.md records both.

A 16-bit outward pop zero-extends ESP, as the SDM pseudocode reads. The hardware's retention of
the upper bits is not modelled.

CpuState, jit_abi, the model identities and the compiled path are unchanged.

## Amendment — 2026-11-19: real-address stack frames that wrap SP (xj-srh.27)

Accepted by the maintainer (2026-11-19). On the I80286 and I80386 Machine models, a
real-address-mode stack frame wraps SP inside the stack segment. A transfer faults only when
one of its own operands leaves the segment: a word at offset FFFFh, or a dword that crosses
the top. The rule applies to every multi-slot frame:
- an interrupt frame (INT n, INT 3, INTO, a processor exception, an IRQ or NMI);
- PUSHA/PUSHAD and POPA/POPAD;
- the far CALL frame, and RETF and RETF n;
- IRET.

The governing clause is the last sentence of the source-resolution rule: "A later SDM
difference alone does not replace an unambiguous historical model rule." Both PRMs state the
rule on each instruction's own exceptions entry, and they agree. Every page image below was
checked.
- **INT/INTO:** shut down only at SP = 1, 3 or 5 (I286 PRM App. B p. B-50; 80386 PRM §17
  p. 17-73). Those are the SP values at which one frame word crosses offset FFFFh (80386 PRM
  §14.7 item 7 p. 14-6). The word's fault is delivered on the same SP, crosses again, and #DF
  then shuts down.
- **PUSHA/PUSHAD:** the same three shutdown values, and exception 13 at SP = 7, 9, 11, 13 or
  15 (I286 PRM p. B-88; 80386 PRM p. 17-134).
- **CALL:** "Interrupt 13 for a word operand at offset 0FFFFH" (I286 PRM p. B-26). The 80386
  faults "if any part of the operand would lie outside of the effective address space from 0
  to 0FFFFH" (p. 17-42). 80386 PRM Table 14-2 p. 14-8, interrupt 12: "push with SP=1 during
  PUSH, CALL, or INT".
- **IRET:** "Interrupt 13 if the stack is popped when it has offset 0FFFFH" (I286 PRM p. B-53).
  The 80386 faults "if any part of the operand being popped lies beyond address 0FFFFH"
  (p. 17-79).
- **POPA:** "Interrupt 13 for a word operand at offset 0FFFFH" (I286 PRM p. B-85). The 80386
  faults per operand (p. 17-129).
- **RET:** the 80386 faults per operand (p. 17-145). I286 PRM App. B RET p. B-95 says
  "Interrupt 13 if the stack pop wraps around from 0FFFFH to 0". That is read as a word popped
  at FFFFh (maintainer decision D3), consistent with B-53, B-85 and Table C-1 p. C-1 ("a word
  operation at offset 0FFFFH").

The resulting behaviour:
- An SP = 2 interrupt stores FLAGS at 0000h, CS at FFFEh and IP at FFFCh, and its IRET returns
  through the same wrap.
- PUSHA then POPA round-trips at every even SP.
- A far CALL at SP = 2 pushes CS at 0000h and IP at FFFEh, and RETF returns.
- On the 80386 with SS.B = 1 (big real), the pointer is ESP. An interrupt at ESP = 2 or 4 wraps
  through 2^32 (maintainer decision D4). The 80386 page names only "SP or ESP = 1, 3, or 5".
- A frame word at FFFFh faults with that word as its operand evidence (offset FFFFh, 2 bytes),
  not the whole frame. The vector is unchanged: #GP on the 80286, #SS on the 80386
  (I286 PRM App. B MOV p. B-74; 80386 PRM §14.6 Table 14-1 p. 14-5).
- PUSHAD at SP = 2 or 6 raises #SS: a dword crosses FFFFh (§14.7 item 7), a case the PUSHAD
  list does not name.

Counter-statements, recorded as decision-24 asks:
- **SDM Vol 2A INT, REAL-ADDRESS-MODE p. 3-470:** "IF stack not large enough for a 6-byte
  return information THEN #SS". The far CALL pseudocode in Vol 2A uses the same "stack not
  large enough" wording. The engine followed this before xj-srh.27.
- **SDM Vol 2A IRET p. 3-497:** "#SS If the top bytes of stack are not within stack limits".
  Its REAL-ADDRESS-MODE pseudocode on p. 3-491 pops slot by slot.
- **SDM Vol 3B §25.33.1 p. 25-34:** "If a stack operation wraps around the address limit,
  shutdown occurs." The same section raises an exception only for an operand that crosses
  offset FFFFh or 0. It also says that an aligned pop at the top "will wrap around to 0" with
  no limit violation.
- **SDM Vol 2B PUSHA pp. 4-526/527:** states the PRM rule (shutdown at 1, 3 or 5; #GP at
  7..15).
- **I286 PRM App. B #SS p. B-11:** a PUSHA or POPA that "attempts to wrap around the 64K
  boundary of a stack segment" faults. That text describes #SS, which the 80286 raises only in
  protected mode. Protected mode keeps its whole-frame check.

Scope:
- Protected mode keeps the whole-frame check.
- ENTER is unchanged.
- The 8086 family has no limit and no shutdown (8086 Family UM §2.3 p. 2-12) and is
  unchanged.
- The frozen Real16 alias keeps its whole-frame checks (maintainer decision D1; decision-23
  §6). Its PUSHA at SP = 2 still raises #13, and its IRET at FFFCh and far CALL at SP = 2 still
  fault.

Witnesses:
- x86jit-tests/tests/legacy/legacy_real_sp_wrap.rs;
- legacy_delivery_liveness.rs;
- legacy_u3_review.rs (`alias_pusha_keeps_the_whole_frame_check` and the moved pins).
