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
