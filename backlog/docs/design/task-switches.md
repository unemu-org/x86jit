---
title: Hardware task-switch implementation and witnesses
type: specification
---

# Hardware task switches on Machine

The legacy Machine interpreter executes protected-mode task transfers on I286
and I386. Application lowering and frozen Real16 IR are unchanged. Task switching
reuses `LegacyOp::Transfer`, `CheckTaskReturn`, and `Interrupt`; no new IR variant
or compiled ABI operation is introduced. `ir/verify/legacy_probes.rs` now explicitly
checks far-transfer and task-return coverage as well as all existing operations.

Sources below refer to the pinned Intel 80286/80287 PRM 210498-005, 80386 PRM
230985-001, and SDM 325462-092US. PRM page images, including instruction procedures,
were inspected. [Decision-24](../../decisions/decision-24%20-%20Historical-task-switch-source-conflicts-and-continuation-boundary.md)
records conflicting statements and the choices made for the historical models.

## Scope, implementation, authority, witness

Test names below are in `x86jit-tests/tests/legacy_task_switch.rs` unless another
module is named. The format matrix is I286/16-bit TSS, I386/16-bit TSS, and
I386/32-bit TSS. An interpreter-only backend fixture panics if compilation occurs.

| Scope | Change | Citation | Witness |
|---|---|---|---|
| Far JMP to TSS | Save outgoing dynamic state including NT, load target FLAGS including NT, clear old B/set new B; preserve both back links | 286 App. B SWITCH_TASKS p. B-12; 386 §7.5 step 5 p. 7-8; SDM Vol 3A §10.3 step 12 p. 10-11 / §10.4 Table 10-2 p. 10-15 | `jmp_to_tss_and_back_saves_state_and_sets_ts`, `incoming_nt_on_jmp_comes_from_the_tss`; nested JMP witnesses below |
| Far CALL and nested IRET | Keep caller busy, write back link, set NT; IRET requires busy target and clears departing B/NT in its saved flags | 286 §8.4 p. 8-7, IRET p. B-51; 386 §7.6 pp. 7-9–10, IRET p. 17-76 | `call_and_nested_iret_preserve_backlink_and_busy_chain`, `iret_backlink_must_be_busy`, `iret_rejects_null_local_wrong_type_absent_and_outside_backlinks` |
| Task gates in GDT/LDT | Validate gate access/presence, use its TSS selector, ignore target TSS DPL/RPL | 286 §8.5 pp. 8-8–9; 386 §7.4 pp. 7-5–6; CALL pp. B-25 / 17-41 | `call_task_gate_in_gdt_and_ldt`, `descriptor_privilege_presence_and_task_gate_target_checks` |
| Type, privilege, presence, limit and error codes | Model-specific #GP/#NP/#TS, selector error codes, old/new task placement | 286 SWITCH_TASKS pp. B-12–13, CALL/JMP/IRET pp. B-25/B-51/B-57–58; 386 §7.5 Table 7-1 pp. 7-8–9, §17 pp. 17-41/17-76/17-86 | `pre_switch_faults_keep_the_outgoing_task`, `short_tss_fault_placement_is_model_specific`, `target_ip_limit_fault_is_in_new_task_with_gp`, `task_segment_validation_vectors_codes_and_order_follow_each_prm` |
| 286 and 386 state images | Word/dword dynamic fields, all eight GPRs, flags/IP/selectors; static LDTR/CR3/privilege stacks/T/I/O-base fields remain unwritten | 286 §8.2 Fig. 8-1 pp. 8-2–3; 386 §7.1 Fig. 7-1 pp. 7-1–3 | `full_386_registers_and_static_tss_fields_survive`, `i286_rejects_386_tss_descriptors_and_old_tr_cache_owns_saves` |
| Busy updates and A20 | Share LTR's physical byte CAS for acquiring and clearing B, preserving other bits | 286 §8.4 p. 8-7; 386 §7.6.1 p. 7-10; SDM Vol 3A §11.1.2.1 p. 11-3; decision-22 | `a20_gates_tss_images_and_both_busy_updates`; core `interp::legacy::tasks::tests::task_acquire_cas_revalidates_competing_privilege_and_busy_edits`; existing LTR CAS witnesses |
| MSW/CR0.TS | Set TS after the 286 register load / at the 386 TR load; a short incoming 286 TSS leaves TS unchanged | 286 §8.3 p. 8-6; 386 §7.5 p. 7-8; SDM Vol 3A §10.3 p. 10-14 | `jmp_to_tss_and_back_saves_state_and_sets_ts`, `typed_refusals_for_v86_privilege_and_task_debug_trap` |
| TR, LDTR and segment caches | Install incoming TR and LDT, qualify/load all applicable segment caches, preserve cached old TR base for saving | 286 SWITCH_TASKS pp. B-12–13; 386 §§7.3,7.5 pp. 7-4/7-8–9; SDM Vol 3A §10.2.4 p. 10-7 | `incoming_ldtr_and_every_segment_cache_are_reloaded`, `i286_rejects_386_tss_descriptors_and_old_tr_cache_owns_saves` |
| Retained IR isolation | Existing complete context stamp includes TR/LDTR and all caches; successful transfers return to dispatch | decision-20; decision-18 §3 | core `vm::legacy::tests::task_register_and_ldtr_separate_retained_context_namespaces` |
| Fault after commit | Preserve incoming registers/selectors/loaded caches and deliver using new IP/stack context | 286 §9.6.4 p. 9-11, SWITCH_TASKS pp. B-12–13; 386 §7.5 pp. 7-8–9; SDM Vol 3A §10.3 pp. 10-13–14 | `segment_fault_stays_in_the_new_task_outward_and_in_guest`, `fault_in_interrupt_task_load_uses_new_context_without_old_double_fault_pair` |
| IDT task gates | Nest like CALL, save fault/interrupt resume IP, push only error code on incoming stack | 286 §9.4 p. 9-4, INT pp. B-49–50; 386 §9.6.2 pp. 9-9–10, INT p. 17-72; SDM Vol 3A §7.12.2 p. 7-18 | `exception_task_gate_pushes_error_on_new_stack_and_returns_by_link`, `software_and_external_idt_task_gates_nest` |
| 386 T bit | Record incoming T, complete switch, refuse pending task debug trap before executing new task | 386 §12.3.1.5 p. 12-8; SDM Vol 3A §10.2.1 p. 10-5 | `typed_refusals_for_v86_privilege_and_task_debug_trap` |
| CR3 and I/O-map base | With PG clear, retain CR3 and read only I/O base; refuse paging reloads | 386 §7.1 p. 7-3; I/O map §8.3.2 pp. 8-4–5 | `full_386_registers_and_static_tss_fields_survive`; core `task_paging_reload_is_refused_before_saves_or_busy_updates` |
| Outward faults | Return precise architectural exception and error code without guest delivery | decision-23 §5 | `pre_switch_faults_keep_the_outgoing_task`, `segment_fault_stays_in_the_new_task_outward_and_in_guest` |
| Resume handoffs | Retain progress through saves, descriptor loads and error push; never execute incoming instruction before completing switch | decisions 16, 22; task commit split above | `task_save_mmio_handoffs_preserve_the_acquired_task_and_old_state`, `post_commit_unmapped_segment_load_resumes_without_restarting_transfer`, `idt_task_gate_error_push_mmio_resumes_in_new_task_once` |
| Diagnostic-style end-to-end fixture | Three GDT TSSs plus task gate; LTR, CALL B, B changes AX/IRETs, A checks B's saved state, JMP C, C changes DX/JMPs A | 286 §§8.2–8.5 pp. 8-1–9; instruction citations above | `three_286_tasks_gate_call_register_checks_iret_then_two_jumps` |

## Deliberate boundaries

An incoming CS.RPL of 1-3 is the new CPL, and the segment loads check against it
(xj-srh.9; spec.md §17.10). Incoming VM uses `Virtual8086Unsupported`. That refusal
occurs after acquiring the task, saving the outgoing state and loading the incoming
image, so that aliasing and busy acquisition cannot use a stale input snapshot.
Paging is refused by the existing entry guard and by `LegacyTaskPaging` at the
switch seam; no paging reload is simulated. `LegacyTaskDebugTrap` persists until
the embedder clears the recorded T bit or replaces state, because DR6.BT is absent.
All these facilities are excluded from execution rather than approximated.

The I/O-map base is visible in the appended CPU field. CPL 0 never needs bitmap
permission checks, so an existing bitmap alone is not a reason to refuse a task.
Contexts that could require those checks are already excluded by the privilege
and VM refusals. Atomic descriptor updates against an MMIO descriptor retain
LTR's `LegacyInstruction` refusal: separate read/write acknowledgments cannot
implement the required CAS. Application paths do not consume the appended fields.

## Review corrections and reproductions

The round-2 end-to-end LDT/gate witness is retained. The following regressions
were run against the unfixed implementation before their fixes: nine integration
tests failed together, the extra CALL privilege/IP reproduction failed separately,
and the deterministic CAS/read-order core witness failed separately. The atomic
refusal reproduction was also run through its MMIO-read acknowledgment against
HEAD and failed with target B left busy (83h instead of 81h).

The format matrix is I286/16-bit, I386/16-bit, and I386/32-bit wherever applicable.

| Finding | Reproduction and correction | Witness |
|---|---|---|
| 1. Lost IRQ / missing CALL retirement | Incoming IF=1; LDTR's GDT page missing before any retry ledger entry. Block new events and IRQ-ready stops for every pending switch. Resume and retire CALL first, retaining the IRQ; deliver it next with saved IP at the first B instruction. Assert both retirement and execution counters. | `review_switch_continuation_retires_call_before_accepting_irq` |
| 2. Stale incoming image | Alias A/B TSS bases directly and through A20. Acquire B and save A before reading B. The core CAS hook changes B's AX at acquisition and the load must observe it. | `review_aliasing_tss_loads_the_state_just_saved_including_a20_aliases`; core `review_incoming_image_is_read_after_successful_busy_cas` |
| 3. Lost switch at old B-bit update | Put A's live access byte on a device page, B on RAM, including A20 aliases. Preflight all CAS bytes before any target acquisition/save. Return `LegacyInstruction` with unchanged TSSs and B still available. Preserve continuation if a preflighted mapping changes during a handoff. | `review_trapped_outgoing_busy_byte_refuses_before_any_task_write` |
| 4. RF cleared by software INT task gate | I386 32-bit incoming EFLAGS=00010002h. Preserve incoming RF on the direct INT completion as on its continuation. | `review_software_task_gate_preserves_incoming_rf_with_or_without_continuation` |
| 5. CS privilege checked before presence | Incoming code is DPL 3/P=0 at CPL 0. Keep the model ordering: #TS on I286; #NP before the DPL check on I386. | `review_code_presence_and_privilege_combined_failure_uses_model_order` |
| 6. 286 TS set too early | JMP to B with limit 2Ah. TR changes and #TS(B) occurs before register loading/TS setting; CR0 remains 1. | `review_short_incoming_i286_tss_does_not_set_ts` |
| 7. 286 outgoing cache checked too early | Execute LTR with limit 28h; JMP B must acquire B, then #TS(B), leaving outgoing state/MSW intact. LTR itself correctly has no limit check (PRM p. B-72); outgoing save needs 29h, incoming load needs 2Bh (p. B-12). | `review_short_current_i286_tss_is_checked_after_busy_acquisition` |
| 8. Missing conflicting source | Resolve 386 LDT failures as #TS(LDT), invalid SS/data as #TS(selector), using PRM §9.8.10 Table 9-5 p. 9-14 and the consistently agreeing SDM entries. Split CALL DPL/IP #GP from disputed target-TSS #TS cases. Keep minimum limits after auditing inconsistent SDM statements; the incoming-NT selection is superseded by the clarification below. | `review_i386_exception_table_resolves_ldt_stack_and_data_conflicts`, `review_i386_call_privilege_ip_and_disputed_busy_checks`; decision-24's full row audit |

`review_complete_i286_tss_alias_image_and_system_caches` compares **every one of
22 words in the 44-byte TSS**, both raw eight-byte task descriptors, all saved GPRs,
TR/LDTR caches, and the guest SMSW result. The retained round-2 witness compares
complete larger TSS images, all segment caches, descriptor access bytes, FLAGS/NT,
and guest MSW across CALL/IRET and two gate JMPs between different LDTs.

`i386_paging_off_does_not_read_cr3_from_the_incoming_tss` leaves the static CR3
field unmapped while the dynamic fields are mapped, pinning the retained 386 PRM
conditional-read rule without changing the existing CR3 value.

## Further continuation and descriptor-update review

Seven new integration witnesses were first run against WIP `57daf55`: **0 passed,
7 failed**. They reproduced all four findings and the shared LTR/ordinary-segment
ROM interpretation. The maintainer superseded that ROM interpretation in the
clarification below. The format matrix remains I286/16-bit, I386/16-bit and I386/32-bit
where applicable. Additional witnesses check the surrounding exit paths.

| Finding | Reproduction, correction and authority | Witness |
|---|---|---|
| 1. Committed accessed refusal lost | Put each incoming descriptor on a device page with A=0; acknowledge both descriptor reads; observe `LegacyInstruction`, then repair the mapping. The old implementation dropped progress and returned `UnusableCodeOrStack`. All unfinished switch refusals now retain progress, using one rule independent of the step. Image CPL/VM and LDT-facility refusals also resume after their input is corrected; every incoming segment is covered, including FS/GS for 32-bit TSSs. Architectural faults still abandon progress in the reached task context. 286 SWITCH_TASKS pp. B-12–13; 386 §7.5 pp. 7-8–9; decisions 16/23. | `review4_accessed_refusals_keep_each_incoming_segment_continuation`, `review4_image_refusals_keep_committed_state_until_input_is_corrected`, `review4_ldt_facility_refusal_keeps_committed_continuation` |
| 2. ROM update interpretation, superseded | The round-4 refusal against ROM was reverted by the maintainer's direction. A switch, LTR or segment load completes its locked update while ROM discards the write. The replacement witnesses cover both remapping during a save and ROM present from entry. Device-region refusals remain. SDM Vol 3A §3.4.5.1 p. 3-13; Vol 2A LTR p. 3-576; Vol 3A §11.1.2.1 p. 11-3. | `review5_outgoing_busy_clear_discards_rom_write_after_save_handoff`, `review5_rom_descriptor_updates_complete_ltr_and_segment_loads`, `review5_task_busy_and_accessed_updates_complete_on_rom_from_entry` |
| 3. Completed IRQ stayed queued | Deliver an IRQ through a task gate to a 386 TSS with IF=1/T=1, directly and through a missing-LDT-page handoff. Previously the completed debug refusal left the IRQ queued. Consume the accepted IRQ once at commit, retain its event through unfinished switch refusals, and finalize completed delivery even when reporting the T bit. A fresh queued instance of the same vector survives continuation completion. The T-bit refusal is produced after switch completion, so resumption cannot retire a completed CALL again. 386 §12.3.1.5 p. 12-8; interrupt tasks §9.6.2 pp. 9-9–10; decision-23 §5. | `review4_irq_task_debug_refusal_consumes_delivery_once`, `review4_irq_segment_refusal_retains_event_and_does_not_consume_a_new_irq`, `review4_irq_new_task_segment_fault_consumes_irq_outward_and_in_guest`, `review4_task_debug_completion_retires_instruction_only_once`, `review4_exception_task_debug_completion_finishes_error_push_once` |
| 4. IRET changed incoming IP before reading it | Alias the incoming IP/EIP with the departing descriptor's access byte. Previously it loaded 81h/89h; now it loads 83h/8Bh and only then makes the departed task available. Both historical IRET branches place the clear after SWITCH_TASKS, before the IP-limit check. The 286 generic procedure's early-clear conflict and the SDM's descriptor-load ordering difference are recorded in decision-24. 286 IRET p. B-51 / SWITCH_TASKS pp. B-12–13; 386 IRET p. 17-76 / §7.5 p. 7-8; SDM Vol 3A §10.3 steps 9–10,14 p. 10-11. | `review4_i286_iret_reads_incoming_image_before_departing_busy_clear`, `review4_i386_iret_reads_incoming_image_before_departing_busy_clear`, `review4_iret_busy_clear_follows_each_models_segment_loads` |

Post-commit coverage includes the incoming-image and LDT refusals, every segment
load, and the IRET busy update. Existing unmapped-segment and error-push MMIO
witnesses continue to cover their bus handoffs. The commit-limit and final IP
checks produce architectural exceptions, which deliberately end the switch
continuation. The completed T-bit boundary has separate retirement/event witnesses, including
an exception error push with and without an MMIO handoff; its unrelated masked
IRQ remains queued and its new stack pointer changes exactly once.
The round-2 complete-image fixture and all round-3 reproductions remain enabled.

## ROM clarification and continuation ownership

Six replacement/new integration witnesses were first run against WIP `070bc44`:
**0 passed, 6 failed**. The precommit accepted-IRQ abandonment witness also failed
separately before its correction. The maintainer accepted decision-24 and explicitly restored
ROM-discard semantics. Its other source-resolution rows are unchanged.

| Item | Reproduction, fix and witness |
|---|---|
| A. ROM semantics | Ordinary GDT/LDT DS, SS and far CS loads, LTR, and task busy/accessed updates wrongly returned `LegacyInstruction`. Remove the shared helper's ROM refusal and allow ROM through task CAS preflight; retain the `Trap`/MMIO refusal and preflight. The remapping witness now expects JMP/IRET completion with the outgoing busy byte unchanged, and the entry-ROM witness checks caches plus the next busy-descriptor fault. Witnesses: `review5_rom_descriptor_updates_complete_ltr_and_segment_loads`, `review5_outgoing_busy_clear_discards_rom_write_after_save_handoff`, `review5_task_busy_and_accessed_updates_complete_on_rom_from_entry`. Authority: SDM Vol 3A §3.4.5.1 p. 3-13 and the accepted decision-24 clarification. |
| B. One continuation/event owner | IRQ 30h commits B, then a device-backed DS update refuses after valid CS/SS loads. CR2 editing previously cancelled only the switch, replaying the consumed event; GPR editing previously cancelled neither. Snapshot all CPU-state fields and allow only the completion requested by the returned exit. Abandon switch, event and instruction progress together; preserve edited registers/caches without retiring the abandoned instruction. Witnesses: `review5_state_edits_abandon_task_and_consumed_irq_together` (all three formats, both delivery policies, CR2/GPR edits), `review5_gpr_edit_abandons_call_without_retiring_it`; core `task_owner_rejects_edits_outside_the_requested_completion` covers flags, SIMD/x87/configuration/retry fields and allowed versus unsolicited completion inputs. Precommit abandonment also removes the accepted queued IRQ, while a fresh IRQ after consumption survives: `review5_abandonment_discards_accepted_irq_before_task_commit`, `review5_abandonment_preserves_a_new_irq_after_the_old_one_was_consumed`. Authority: decision-24's continuation contract. |
| C. Cleanup before validation | Refuse an incoming CS update, repair its device mapping, edit CR2 and queue A20. Previously entry validation stranded the owner behind unusable CS. Decide abandonment first; release the event/IRQ block and pending A20 before reporting the edited state's entry refusal. The unfinished caches stand. Witness: `review5_abandonment_precedes_unusable_cs_validation_and_releases_a20` (all three formats, both delivery policies), including readiness and next-instruction execution after the embedder repairs caches. `review5_zero_budget_preserves_owner_until_the_next_resume` retains the zero-budget no-consumption contract. |
| D. Cross-vCPU continuation copying | Documentation only: decision-24 states that VM forks or CPU-state copying with any pending Machine mid-instruction continuation are unsupported and tracked separately. No fork/copy implementation or witness was added in this change, as requested. |

The ownership snapshot is sealed after engine-owned exit bookkeeping, so valid
MMIO completions continue to work through all existing save/read/segment/error-push
handoffs. It uses field comparisons, source/table identities and exhaustive CPU
and port-state destructuring; CPU layout and compiled ABI are unchanged. The old
IRQ-retention witness now uses a device page and explicit descriptor-read
acknowledgments, preserving its atomic-refusal coverage under the ROM clarification.

## Event inhibition on abandonment

Before correction, four integration witnesses failed against WIP `74055a0`;
a fifth delivery-refusal witness and the core boundary witness also failed
separately. The requested STI/#DB case was reproduced with both mapped and
unmapped incoming instruction bytes. In the latter case, three successive fetch
exits retained the IRQ and made no retirement progress.

Completion and abandonment now share one private event-boundary cleanup. It
releases STI inhibition, MOV/POP SS inhibition, the deferred port TF boundary and
the pending accepted event. Task-state cancellation continues to discard its
saved single-step state and instruction retry, preserving edited CPU state and
the existing IRQ-consumption rules. Architectural TF is unchanged. The main
witnesses check all three TSS formats: incoming CS/SS are usable, DS loading has
refused a device CAS, the embedder repairs the mapping and edits AX, then IRQ 30h
is accepted with saved IP=3000h before the incoming task executes or fetches.
Only the IRQ handler NOP retires; the abandoned task transfer is not replayed.

| Path | Witness |
|---|---|
| Task-gate event cancelled with mapped incoming code | `review6_sti_debug_task_abandonment_accepts_irq_before_first_instruction` |
| Task-gate event cancelled before an unmapped first fetch | `review6_sti_debug_task_abandonment_accepts_irq_before_unmapped_fetch` |
| Non-task event cancelled by CS:IP redirection during stack MMIO | `review6_redirected_debug_event_releases_sti_shadow_before_fetch` |
| Outward single-step exit | `review6_outward_debug_event_exit_releases_sti_shadow` |
| Nonresumable delivery refusal | `review6_nonresumable_debug_delivery_refusal_releases_sti_shadow` |
| STI, SS inhibition, delayed TF and pending event released together | core `vm::legacy::tests::redirected_event_discards_every_private_boundary_before_fetch` |

The abandonment audit also checked architectural delivery faults, shutdown and
execution reset. Nonresumable delivery faults release their event boundary before
returning or starting replacement exception delivery; resumable bus/task exits
retain it. Reset already clears these fields. Existing event, MMIO, A20, reset,
STI/SS and single-step regression suites remain enabled. Authority: decision-24's
accepted continuation contract; 286 PRM §9.7.1 p. 9-14, MOV/POP pp. B-73/B-83;
386 PRM §§9.2.4,12.3.1.4 pp. 9-3–4/12-8; SDM Vol 3A §§7.8.1,7.8.3 pp. 7-7–9.

## Incoming NT on JMP clarification (2026-11-12)

The maintainer approved loading NT from the incoming TSS FLAGS on JMP for I286
and I386 with either supported TSS format. CALL and INT (including task gates
and IDT task gates) set incoming NT=1; IRET restores incoming NT as saved.
JMP still saves outgoing NT unchanged, clears the outgoing busy bit, sets the
incoming busy bit and preserves both back links. IRET still clears departing
NT in the saved image. The implementation change is confined to incoming NT in
`interp/legacy/tasks.rs`'s `Stage::ReadImage`.

Authority, checked against each page image: 80286 PRM App. B SWITCH_TASKS
p. B-12; 80386 PRM §7.5 step 5 p. 7-8; SDM Vol 3A §10.3 step 12 p. 10-11
and §10.4 Table 10-2 p. 10-15. Decision-24 records the opposed clear-on-JMP
statements in 286 §8.4 Table 8-2 p. 8-7, 386 §7.6 Table 7-2 p. 7-10 and
SDM §10.4 prose p. 10-15. A generic diagnostic validated on 286 hardware
falsifies that reading by requiring B's saved FLAGS=4046h after repeated JMPs
between B and C, followed by IRET through B's CALL-created back link to A.
The manual load procedures and explicit SDM step/table license the rule.

The existing `incoming_nt_on_jmp_uses_the_historical_prm_rule` witness is renamed
`incoming_nt_on_jmp_comes_from_the_tss`. It now checks incoming NT=0/1 against
outgoing NT=0/1, direct TSS and task-gate JMPs, all three model/TSS combinations,
unchanged outgoing NT and back links, both busy bits and TR.

Three added witnesses cover CALL A→B through a task gate, four B→C→B JMP
round trips, then IRET B→A: `i286_nested_jmp_ping_pong_returns_through_backlink`,
`i386_286_tss_nested_jmp_ping_pong_returns_through_backlink`, and
`i386_386_tss_nested_jmp_ping_pong_returns_through_backlink`. After every switch
they compare all GPRs, IP, FLAGS including NT, segment selectors, TR cache,
LDTR selector, CR0, all three busy bits/back links, and complete TSS byte images.
The 32-bit format uses nonzero upper GPR halves. Each runs with A's saved NT=0
and NT=1, also pinning IRET's incoming-NT rule and outgoing-NT clear. B's stack
contains fill data so a stack IRET cannot imitate the required task return.

Before the engine change, the three ping-pong witnesses each failed on the first
JMP back into B: FLAGS was 0046h rather than 4046h. The updated incoming-NT
witness also failed separately, observing 0203h rather than 4203h.

The corrected `legacy_task_switch` suite passed **65 tests**; all other
`legacy_*` suites passed **332 tests**, for **397 passed** in total.
`legacy_upper_audit` has no enabled tests on the ARM host. Commands used at most
four build jobs, a 180-second timeout per test command and a separate
`CARGO_TARGET_DIR`.

`cargo clippy --offline --all-targets --all-features -- -D warnings`,
`cargo fmt --all -- --check` and `git diff --check` passed.

## Initial implementation validation

All commands use at most six Cargo jobs, a 180-second timeout per command,
one test binary at a time, and one libtest thread. Build artifacts used a
separate `CARGO_TARGET_DIR`. The test-only compiler wrapper
selected the installed linker for a core witness that launches rustc directly;
repository toolchain configuration was unchanged.

| Binary | Passed | Ignored |
|---|---:|---:|
| x86jit-core library | 402 | 2 |
| legacy_task_switch | 62 | 0 |
| legacy_task_registers | 6 | 0 |
| legacy_ldt | 25 | 0 |
| legacy_idt | 22 | 0 |
| legacy_i286 | 34 | 0 |
| legacy_protected | 36 | 0 |
| legacy_contract | 13 | 0 |
| legacy_a20 | 50 | 0 |
| legacy_real | 92 | 0 |
| legacy_real_exceptions | 8 | 0 |
| legacy_real_parity | 44 | 0 |
| vcpu_reset | 10 | 0 |
| boundary | 1 | 0 |
| **Selected suites total** | **805** | **2** |

The initial 25 task-switch integration witnesses were also run with the original
interpreter and dispatcher: **0 passed, 25 failed**. The current source was
restored afterward. The four new core witnesses cover CAS races, paging refusal,
interrupt-task error-push ordering, and context namespace isolation. Existing
refusal-only task tests now assert invalid-TSS faults. Legacy real-mode NT
refusal remains unchanged.

```sh
cargo test --offline -p x86jit-core --lib -- --test-threads=1
cargo test --offline -p x86jit-tests --test legacy_task_switch -- --test-threads=1
# Repeat the second command with each regression binary listed above.
cargo clippy --offline --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
git diff --check
```

Clippy, formatting and diff whitespace checks passed. This is a selected
regression run, not the full workspace test gate. `scripts/ladder.sh` returned
exit 2 because the sibling real-program checkout is absent; it did not run.
No commits, bead operations, git index/ref changes or pushes were performed.
