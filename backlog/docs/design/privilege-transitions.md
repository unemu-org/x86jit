---
title: Privilege transitions at CPL 1-3 and their witnesses
type: specification
---

# Privilege transitions on Machine (xj-srh.9, first landing)

The legacy Machine interpreter runs protected-mode code at CPL 0-3 on `I80286`
and `I80386`. The helpers live in `x86jit-core/src/interp/legacy/privilege.rs`;
the run-time CPL and IOPL test is `LegacyOp::Privileged`, emitted by
`lift/legacy.rs` in the protected regime only. Call gates stay refused
(`LegacyGateOrTaskTransfer`) until the second landing. spec.md §17.10 holds the
rules and the source rows that disagree, with the engine's choice for each.

Sources are the pinned Intel 80286/80287 PRM 210498-005 ("286 PRM"), 80386 PRM
230985-001 ("386 PRM") and SDM 325462-092US, cited by printed page. ASSUMED and
DERIVED mark engine choices.

## Fact, citation, witness

Test names are in `x86jit-tests/tests/legacy_privilege.rs` unless another module
is named. Every test runs on both generations.

| Fact | Citation | Witness |
|---|---|---|
| Inward delivery takes SS:(E)SP for the target ring from the TSS and pushes old SS, old (E)SP, FLAGS, CS, IP at the gate's width; IF clear for interrupt gates, TF and NT clear | 286 PRM App. B INT p. B-49, §8.2 Fig. 8-1 p. 8-2, §9.4 Fig. 9-4 pp. 9-4-5; 386 PRM §17 INT pp. 17-70-71, §7.1 Fig. 7-1 p. 7-2; SDM Vol 2A INT pp. 3-472-474 | `int_n_from_ring_3_enters_ring_0_on_the_tss_stack`, `ring_3_to_1_and_ring_2_to_0_use_the_target_rings_tss_stack`, `irq_and_nmi_from_ring_3_enter_ring_0` |
| Exceptions push their error code on the inner stack and ignore gate DPL; INT n, INT3 and INTO check it | 286 PRM §9.4 p. 9-4, p. B-49; 386 PRM p. 17-70; 386 PRM §12.3.1.1 p. 12-7 (RF) | `an_exception_from_ring_3_pushes_its_error_code_and_ignores_gate_dpl`, `software_interrupts_check_gate_dpl_and_bound_does_not` |
| Inward faults, in check order, with EXT for external events; room counts the error code (O2) against the real SS limit | SDM INT pp. 3-471-474; 286 PRM Table 9-1 p. 9-6 | `inward_delivery_faults_precede_every_change_and_carry_ext`, `a_failing_nmi_delivery_from_ring_3_carries_ext`; core `interp::legacy::privilege::tests::inward_room_counts_the_error_code_and_faults_with_the_new_ss_and_ext` |
| A failing delivery escalates to #DF, then shutdown | 286 PRM §9.6.2 p. 9-10; 386 PRM Tables 9-3/9-4 p. 9-13 | `a_failing_inward_delivery_escalates_to_double_fault_and_shutdown` |
| Outward RETF, RETF n and IRET: SS:SP from the frame, check order, #SS(0) bound of 8+n/10 or 16+n/20 bytes | 286 PRM pp. B-51-52, B-94-95, §7.5.2 pp. 7-20-22; 386 PRM pp. 17-76-78, 17-142-144, Table 6-3 p. 6-17; SDM RET pp. 4-571-574, IRET pp. 3-493-497 | `retf_to_an_outer_ring_reloads_ss_sp_and_nulls_data_segments`, `outward_return_faults_precede_every_change`, `outward_return_bounds_count_the_operand_size_and_n`, `iret_and_retf_to_conforming_code_with_dpl_below_rpl_enter_ring_3` |
| Data segments with a cached DPL below the new CPL are nulled; the rest, a null selector included, is kept (O6) | SDM RET p. 4-574, IRET p. 3-495, Vol 3A §6.8.6 p. 6-20 | `retf_to_an_outer_ring_reloads_ss_sp_and_nulls_data_segments`, `an_outward_return_to_ring_2_nulls_by_the_cached_dpl_only` |
| IRET loads IF and IOPL by the old CPL; NT, TF and the 386 RF restore; a TF trap after the next instruction enters ring 0 | SDM IRET p. 3-493; 286 PRM p. B-51, §9.7.1 p. 9-14 | `iret_to_an_outer_ring_loads_if_and_iopl_by_the_old_cpl`, `iret_to_an_outer_ring_restores_nt_tf_and_the_386_rf`, `a_tf_trap_after_an_outward_iret_is_delivered_to_ring_0` |
| POPF and IRET at CPL > 0 mask IOPL and IF without a fault | 286 PRM p. B-86, §10.1 p. 10-1; 386 PRM §8.3.1 p. 8-5, p. 17-130; SDM POPF pp. 4-407-408 | `popf_and_iret_at_each_cpl_keep_if_and_iopl_without_faulting` |
| CPL-0 and IOPL-sensitive instructions are #GP(0) before any operand or port exit (O10, ASSUMED order); I80386 I/O through a 386 TSS is refused, through a 286 TSS #GP(0) | 286 PRM App. B pp. B-30-B-106 (spec.md §17.10); 386 PRM §6.3.5.1 p. 6-17, §8.3 pp. 8-4-5, §17.2.2.5 p. 17-15 | `privileged_and_iopl_sensitive_instructions_check_before_any_effect`, `port_io_above_iopl_on_the_80386_faults_through_a_286_tss` |
| 80286 LOCK is IOPL-sensitive; the 386 has no such rule | 286 PRM p. B-68, §3.11.2 p. 3-29; 386 PRM pp. 17-100-101 | `lock_is_iopl_sensitive_on_the_80286_only` |
| REP INS and REP OUTS with a zero count make no I/O test | SDM Vol 2B REP p. 4-564 | `rep_ins_and_outs_with_a_zero_count_make_no_io_test` |
| Same-level far CALL, JMP and RETF at CPL 3; conforming targets keep CPL; MOV SS and task gates at CPL 3 | 286 PRM pp. B-24, B-49, B-57, B-75; 386 PRM pp. 17-39, 17-72, 17-85, 17-97; SDM CALL p. 3-131 | `same_level_far_transfers_and_segment_checks_at_cpl_3` |
| A task switch enters ring 3; its faults, before and after commit, are delivered inward | 286 PRM p. B-13; 386 PRM §7.5 p. 7-8, Table 7-1 p. 7-9 | `a_task_switch_enters_ring_3_and_its_fault_is_delivered_inward`; `legacy_task_switch.rs` `a_task_with_cs_rpl_3_runs_at_cpl_3_or_faults_on_its_segments` |
| Frames, TSS fields and descriptors on devices stop with nothing committed and resume exactly; a device descriptor needing its accessed bit is `LegacyInstruction` | decision-24; spec.md §5.2 | `inward_frames_tss_fields_and_descriptors_on_devices_hand_off_exactly`, `outward_frames_and_descriptors_on_devices_hand_off_exactly` |
| End to end: a ring-0 kernel IRETs to ring 3 and takes INT, IRQ and #GP back; an outward IRET through a 286 TSS | as above | `a_ring_0_kernel_irets_to_ring_3_and_takes_int_irq_and_gp_back_to_ring_0`, `an_outward_iret_from_cpl_0_to_cpl_3_through_a_286_tss_and_an_int_back` |
| Entry at CPL 1-3 runs | decision-21 amendment (xj-srh.9) | `an_embedder_state_at_cpl_3_runs`; `legacy_protected.rs` |
| Retention: ring round trips retain two contexts and relift nothing (O13) | decision-20 | core `vm::legacy::tests::ring_trips_retain_two_contexts_and_relift_nothing` |
