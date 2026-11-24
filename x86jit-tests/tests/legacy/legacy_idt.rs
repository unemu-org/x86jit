//! IDT delivery witnesses: I286 PRM §§9.4–9.6 pp. 9-3–10;
//! I386 PRM §§9.5–9.8 pp. 9-6–13; §17 INT pp. 17-70–72 (scan checked).
use x86jit_core::{
    CpuGeneration, ExceptionDelivery, Exit, LegacyCpuModel, Prot, RegionKind, StepOptions,
    StepStop, Vcpu, Vm, VmConfig,
};

fn setup(generation: CpuGeneration, code: &[u8]) -> (Vm, Vcpu) {
    setup_delivery(generation, code, ExceptionDelivery::Outward)
}
#[allow(deprecated)] // Compatibility API witness.
fn setup_delivery(
    generation: CpuGeneration,
    code: &[u8],
    delivery: ExceptionDelivery,
) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(0x40000));
    vm.set_legacy_cpu_model(LegacyCpuModel {
        generation,
        model_semantics_id: generation as u64,
    });
    vm.set_exception_delivery(delivery);
    vm.map(0, 0x40000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, code).unwrap();
    // Flat 16-bit code/data descriptors. I286 PRM §6.3 Fig. 6-3 p. 6-5.
    vm.write_bytes(0x2008, &[0xff, 0xff, 0, 0, 0, 0x9b, 0, 0])
        .unwrap();
    vm.write_bytes(0x2010, &[0xff, 0xff, 0, 0, 0, 0x93, 0, 0])
        .unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[4] = 0x8000;
    cpu.cpu.cs = 8;
    cpu.cpu.ss = 16;
    cpu.cpu.legacy.cr0 = 1;
    cpu.cpu.legacy.gdtr.base = 0x2000;
    cpu.cpu.legacy.gdtr.limit = 23;
    cpu.cpu.legacy.idtr.base = 0x3000;
    cpu.cpu.legacy.idtr.limit = 0x7ff;
    (vm, cpu)
}
fn gate(vm: &Vm, vector: u8, kind: u8, target: u32) {
    let low = (8u32 << 16) | (target & 0xffff);
    let high = (target & 0xffff0000) | (u32::from(kind) << 8);
    vm.write_bytes(0x3000 + u64::from(vector) * 8, &low.to_le_bytes())
        .unwrap();
    vm.write_bytes(0x3004 + u64::from(vector) * 8, &high.to_le_bytes())
        .unwrap();
}
fn word(vm: &Vm, address: u64, size: usize) -> u32 {
    let mut bytes = [0; 4];
    vm.read_bytes(address, &mut bytes[..size]).unwrap();
    u32::from_le_bytes(bytes)
}
fn steps(cpu: &mut Vcpu, vm: &Vm, count: u64) {
    let result = cpu.run_steps(vm, count, StepOptions::default()).unwrap();
    assert_eq!(result.steps(), count, "{:?}", result.stop());
    assert!(
        matches!(result.stop(), StepStop::Exit(Exit::BudgetExhausted)),
        "{:?}",
        result.stop()
    );
}
#[test]
fn software_interrupt_gate_frame_and_iret_both_models() {
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        for kind in [0x86, 0x87] {
            let (vm, mut cpu) = setup(generation, b"\xcd\x30\x43\xf4");
            gate(&vm, 0x30, kind, 0x1800);
            vm.write_bytes(0x1800, b"\xcf").unwrap();
            cpu.cpu.flags.if_ = true;
            cpu.cpu.rflags_system = 0x4100;
            steps(&mut cpu, &vm, 1);
            assert_eq!(
                (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
                (8, 0x1800, 0x7ffa)
            );
            assert_eq!(word(&vm, 0x7ffa, 2), 0x1002);
            assert_eq!(word(&vm, 0x7ffc, 2), 8);
            assert_eq!(word(&vm, 0x7ffe, 2) & 0x4300, 0x4300);
            assert_eq!(cpu.cpu.flags.if_, kind == 0x87);
            assert_eq!(cpu.cpu.rflags_system & 0x4100, 0);
            steps(&mut cpu, &vm, 1);
            assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[4]), (0x1002, 0x8000));
            assert_eq!(cpu.cpu.rflags_system & 0x4100, 0x4100);
        }
    }
}

#[test]
fn gate_width_is_independent_of_code_and_stack_defaults() {
    // I386 PRM §16.4.3 p. 16-7: IDT gate type selects frame width;
    // §17.1.1 p. 17-1: CS.D selects default operand/address size;
    // §17.1.3 pp. 17-1–2: SS.B selects stack addressing (scan checked).
    for gate32 in [false, true] {
        for code32 in [false, true] {
            for stack32 in [false, true] {
                for trap in [false, true] {
                    let (vm, mut cpu) = setup(CpuGeneration::I386, b"\xcd\x30");
                    let attr = if code32 { 0x40 } else { 0 };
                    vm.write_bytes(0x200e, &[attr, 0]).unwrap();
                    cpu.cpu.legacy.segments[0].attributes |= u16::from(code32) << 14;
                    cpu.cpu.legacy.segments[3].attributes |= u16::from(stack32) << 14;
                    cpu.cpu.gpr[4] = if stack32 { 0x8000 } else { 0x1234_8000 };
                    let old_sp = cpu.cpu.gpr[4];
                    gate(
                        &vm,
                        0x30,
                        0x86 | (u8::from(gate32) * 8) | u8::from(trap),
                        0x1800,
                    );
                    vm.write_bytes(
                        0x1800,
                        if gate32 == code32 {
                            b"\xcf"
                        } else {
                            b"\x66\xcf"
                        },
                    )
                    .unwrap();
                    cpu.cpu.flags.if_ = true;
                    steps(&mut cpu, &vm, 1);
                    let size = if gate32 { 4 } else { 2 };
                    assert_eq!(word(&vm, 0x8000 - 3 * size, size as usize), 0x1002);
                    assert_eq!(word(&vm, 0x8000 - 2 * size, size as usize), 8);
                    assert_eq!(cpu.cpu.flags.if_, trap);
                    steps(&mut cpu, &vm, 1);
                    assert_eq!(
                        (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
                        (8, 0x1002, old_sp)
                    );
                }
            }
        }
    }
}

fn outward(cpu: &mut Vcpu, vm: &Vm, vector: u8, code: Option<u32>) {
    let before = format!("{:?}", cpu.cpu);
    let result = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    assert_eq!(result.steps(), 0);
    let StepStop::Exit(exit @ Exit::Exception { vector: actual, .. }) = result.stop() else {
        panic!("{:?}", result.stop())
    };
    assert_eq!(*actual, vector);
    let expected = code.map_or(x86jit_core::fault::ErrorCode::NoCode, |value| {
        x86jit_core::fault::ErrorCode::bits(value, 0xffff)
    });
    assert_eq!(exit.fault_metadata().unwrap().error_code(), expected);
    assert_eq!(before, format!("{:?}", cpu.cpu));
}

#[test]
fn gate_limit_type_presence_and_target_checks_are_precise() {
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        for (kind, selector, target, limit, vector, code) in [
            (0x86, 8, 0x1800, 0x186, 13, 0x182), // missing last IDT byte
            (0x96, 8, 0x1800, 0x7ff, 13, 0x182), // S=1
            (0x06, 8, 0x1800, 0x7ff, 11, 0x182), // P=0
            (0x86, 0, 0x1800, 0x7ff, 13, 0),
            (0x86, 24, 0x1800, 0x7ff, 13, 24),
            (0x86, 16, 0x1800, 0x7ff, 13, 16), // data target
        ] {
            let (vm, mut cpu) = setup(generation, b"\xcd\x30");
            gate(&vm, 0x30, kind, target);
            vm.write_bytes(0x3182, &(selector as u16).to_le_bytes())
                .unwrap();
            cpu.cpu.legacy.idtr.limit = limit;
            outward(&mut cpu, &vm, vector, Some(code));
            assert_eq!(word(&vm, 0x7ffa, 2), 0);
        }
        let (vm, mut cpu) = setup(generation, b"\xcd\x30");
        gate(&vm, 0x30, 0x86, 0x1800);
        vm.write_bytes(0x200d, &[0x1b]).unwrap();
        outward(&mut cpu, &vm, 11, Some(8));
        vm.write_bytes(0x200d, &[0xfb]).unwrap();
        outward(&mut cpu, &vm, 13, Some(8));
        vm.write_bytes(0x200d, &[0x9b]).unwrap();
        vm.write_bytes(0x2008, &0x17ffu16.to_le_bytes()).unwrap();
        outward(&mut cpu, &vm, 13, Some(0));
    }
}

#[test]
fn gate_selector_rpl_is_ignored_and_accessed_bit_is_set() {
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let (vm, mut cpu) = setup(generation, b"\xcd\x30");
        gate(&vm, 0x30, 0x86, 0x1800);
        vm.write_bytes(0x3182, &11u16.to_le_bytes()).unwrap();
        vm.write_bytes(0x200d, &[0x9a]).unwrap();
        steps(&mut cpu, &vm, 1);
        assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (8, 0x1800));
        assert_eq!(word(&vm, 0x200d, 1), 0x9b);
    }
}

#[test]
fn i286_rejects_i386_gates_and_task_gates_validate_the_target_tss() {
    // 286 PRM App. B INT pp. B-49–50; 386 PRM §17 INT p. 17-72:
    // the SDM's conflicting Tables 7-6/10-1 cannot settle #GP versus #TS:
    // the gate target is a code descriptor, so no task switch takes place.
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let (vm, mut cpu) = setup(generation, b"\xcd\x30");
        gate(&vm, 0x30, 0x85, 0x1800);
        outward(
            &mut cpu,
            &vm,
            if generation == CpuGeneration::I286 {
                13
            } else {
                10
            },
            Some(8),
        );
        assert_eq!(cpu.cpu.gpr[4], 0x8000);
        if generation == CpuGeneration::I286 {
            for kind in [0x8e, 0x8f] {
                gate(&vm, 0x30, kind, 0x1800);
                outward(&mut cpu, &vm, 13, Some(0x182));
            }
        }
    }
}

#[test]
fn software_int3_into_and_int_exception_vectors_push_no_error_code() {
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        for (code, vector) in [(&b"\xcc"[..], 3), (&b"\xce"[..], 4), (&b"\xcd\x0d"[..], 13)] {
            let (vm, mut cpu) = setup(generation, code);
            gate(&vm, vector, 0x86, 0x1800);
            cpu.cpu.flags.of = true;
            steps(&mut cpu, &vm, 1);
            assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[4]), (0x1800, 0x7ffa));
            assert_eq!(word(&vm, 0x7ffa, 2), 0x1000 + code.len() as u32);
        }
        let (vm, mut cpu) = setup(generation, b"\xce\x43");
        steps(&mut cpu, &vm, 1);
        assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[4]), (0x1001, 0x8000));
    }
}

#[test]
fn gp_exception_is_outward_by_default_and_delivered_when_selected() {
    // I286 PRM Appendix B MOV pp. B-73–74; I386 PRM §17 MOV pp. 17-109–110.
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let (vm, mut cpu) = setup(generation, b"\x8e\xd8"); // MOV DS,AX, invalid selector
        cpu.cpu.gpr[0] = 24;
        outward(&mut cpu, &vm, 13, Some(24));
        let (vm, mut cpu) = setup_delivery(generation, b"\x8e\xd8", ExceptionDelivery::InGuest);
        cpu.cpu.gpr[0] = 24;
        gate(&vm, 13, 0x86, 0x1800);
        vm.write_bytes(0x1800, b"\xf4").unwrap();
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::Hlt)),
            "{:?}",
            run.stop()
        );
        assert_eq!((run.steps(), cpu.retired_instructions()), (1, 1));
        assert_eq!(word(&vm, 0x7ff8, 2), 24);
        assert_eq!(word(&vm, 0x7ffa, 2), 0x1000);
        assert_eq!(word(&vm, 0x7ffc, 2), 8);
    }
}

#[test]
fn benign_ud_and_contributory_de_have_different_delivery_escalation() {
    // I286 PRM §9.6.2 p. 9-10; I386 PRM §9.8.8 Tables 9-3/9-4 p. 9-13.
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        for (instruction, first, expected, error_code) in [
            (&b"\x0f\x0b"[..], 6, 11, 6 * 8 + 3),
            (&b"\xf6\xf3"[..], 0, 8, 0), // DIV BL, BL=0
        ] {
            let (vm, mut cpu) = setup_delivery(generation, instruction, ExceptionDelivery::InGuest);
            gate(&vm, first, 0x06, 0x1800); // not present
            gate(&vm, 11, 0x86, 0x1900);
            gate(&vm, 8, 0x86, 0x1a00);
            vm.write_bytes(0x1900, b"\xf4").unwrap();
            vm.write_bytes(0x1a00, b"\xf4").unwrap();
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(run.stop(), StepStop::Exit(Exit::Hlt)),
                "{:?}",
                run.stop()
            );
            assert_eq!(cpu.cpu.rip, if expected == 8 { 0x1a01 } else { 0x1901 });
            assert_eq!(word(&vm, 0x7ff8, 2), error_code);
            assert_eq!(word(&vm, 0x7ffa, 2), 0x1000);
        }
    }
}

#[test]
fn shutdown_is_latched_until_execution_reset() {
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let (vm, mut cpu) = setup_delivery(generation, b"\x8e\xd8", ExceptionDelivery::InGuest);
        cpu.cpu.gpr[0] = 24;
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(
            matches!(
                run.stop(),
                StepStop::Exit(Exit::Shutdown { addr: 0x1000, .. })
            ),
            "{:?}",
            run.stop()
        );
        assert_eq!(run.steps(), 0);
        vm.write_bytes(0x1000, b"\xf4").unwrap();
        assert!(matches!(cpu.run(&vm, None), Exit::Shutdown { .. }));
        cpu.reset_execution_state(x86jit_core::ResetOptions::default());
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    }
}

#[test]
fn protected_irq_uses_idt_and_iret_and_costs_no_instruction_unit() {
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let (vm, mut cpu) = setup(generation, b"\x43\xf4");
        gate(&vm, 0x30, 0x86, 0x1800);
        vm.write_bytes(0x1800, b"\xcf").unwrap();
        cpu.cpu.flags.if_ = true;
        assert!(cpu.irq_ready());
        cpu.inject_irq(0x30);
        steps(&mut cpu, &vm, 1); // IRQ plus IRET
        assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[4]), (0x1000, 0x8000));
        assert!(!cpu.has_pending_irq());
        assert_eq!(word(&vm, 0x7ffa, 2), 0x1000);
        assert_eq!(cpu.retired_instructions(), 1);
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.gpr[3], 1);
    }
}

#[test]
fn irq_delivery_fault_sets_ext_and_can_enter_np() {
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        for delivery in [ExceptionDelivery::Outward, ExceptionDelivery::InGuest] {
            let (vm, mut cpu) = setup_delivery(generation, b"\x90", delivery);
            gate(&vm, 0x30, 0x06, 0x1800);
            gate(&vm, 11, 0x86, 0x1900);
            vm.write_bytes(0x1900, b"\xf4").unwrap();
            cpu.cpu.flags.if_ = true;
            cpu.inject_irq(0x30);
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            if delivery == ExceptionDelivery::Outward {
                let StepStop::Exit(exit @ Exit::Exception { vector: 11, .. }) = run.stop() else {
                    panic!("{:?}", run.stop())
                };
                assert_eq!(
                    exit.fault_metadata().unwrap().error_code(),
                    x86jit_core::fault::ErrorCode::bits(0x183, 0xffff)
                );
                assert_eq!(run.steps(), 0);
            } else {
                assert!(matches!(run.stop(), StepStop::Exit(Exit::Hlt)));
                assert_eq!(word(&vm, 0x7ff8, 2), 0x183);
            }
            assert!(!cpu.has_pending_irq());
        }
    }
}

#[test]
fn protected_tf_delay_ss_inhibit_and_outward_db() {
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let (vm, mut cpu) =
            setup_delivery(generation, b"\x9d\x8e\xd0\x90", ExceptionDelivery::InGuest);
        gate(&vm, 1, 0x86, 0x1800);
        vm.write_bytes(0x1800, b"\xcf").unwrap();
        vm.write_bytes(0x8000, &0x102u16.to_le_bytes()).unwrap();
        cpu.cpu.gpr[0] = 16;
        steps(&mut cpu, &vm, 1); // POPF enables TF with one-instruction delay
        assert_eq!(cpu.cpu.rip, 0x1001);
        steps(&mut cpu, &vm, 1); // MOV SS suppresses this boundary
        assert_eq!(cpu.cpu.rip, 0x1003);
        steps(&mut cpu, &vm, 1); // following NOP traps
        assert_eq!(cpu.cpu.rip, 0x1800);
        assert_eq!(word(&vm, 0x7ffc, 2), 0x1004);
        steps(&mut cpu, &vm, 1); // IRET restoring TF does not trap
        assert_eq!(cpu.cpu.rip, 0x1004);
        let (vm, mut cpu) = setup(generation, b"\x90");
        cpu.cpu.rflags_system = 0x100;
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 1);
        assert!(
            matches!(
                run.stop(),
                StepStop::Exit(Exit::Exception { vector: 1, .. })
            ),
            "{:?}",
            run.stop()
        );
        assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[4]), (0x1001, 0x8000));
    }
}

#[test]
fn bound_fault_routes_to_br_at_the_faulting_ip_in_both_widths() {
    // I286 PRM Appendix B BOUND p. B-22; I386 PRM §17 BOUND p. 17-26.
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        for wide in [false, true] {
            if wide && generation == CpuGeneration::I286 {
                continue;
            }
            let code = if wide {
                &b"\x66\x62\x06\x00\x40"[..]
            } else {
                &b"\x62\x06\x00\x40"[..]
            };
            for (index, lower, upper, fails) in [
                (4i32, -3i32, 3i32, true),
                (-3, -3, 3, false),
                (-4, -3, 3, true),
                (70000, 65536, 100000, false),
            ] {
                if !wide && index == 70000 {
                    continue;
                }
                let (vm, mut cpu) = setup_delivery(generation, code, ExceptionDelivery::InGuest);
                gate(&vm, 5, 0x86, 0x1800);
                vm.write_bytes(0x1800, b"\xf4").unwrap();
                let size = if wide { 4 } else { 2 };
                vm.write_bytes(0x4000, &lower.to_le_bytes()[..size])
                    .unwrap();
                vm.write_bytes(0x4000 + size as u64, &upper.to_le_bytes()[..size])
                    .unwrap();
                cpu.cpu.gpr[0] = index as u64;
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                if fails {
                    assert!(
                        matches!(run.stop(), StepStop::Exit(Exit::Hlt)),
                        "{:?}",
                        run.stop()
                    );
                    assert_eq!(word(&vm, 0x7ffa, 2), 0x1000);
                } else {
                    assert!(
                        matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
                        "{:?}",
                        run.stop()
                    );
                    assert_eq!(cpu.cpu.gpr[4], 0x8000);
                }
            }
        }
    }
}

#[test]
fn protected_entry_lidt_int_handler_iret_and_continue() {
    // I286 PRM §10.3 p. 10-5, §9.4 pp. 9-4–6; I386 PRM §14.4 p. 14-4.
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let code = b"\xb8\x01\x00\x0f\x01\xf0\xea\x00\x12\x08\x00"; // LMSW; far JMP
        let (vm, mut cpu) = setup(generation, code);
        cpu.cpu.legacy.cr0 = 0;
        cpu.cpu.cs = 0;
        vm.write_bytes(0x1200, b"\x0f\x01\x1e\x00\x21\xcd\x30\x43\xf4")
            .unwrap();
        vm.write_bytes(0x2100, b"\xff\x07\x00\x30\x00\x00").unwrap();
        gate(&vm, 0x30, 0x86, 0x1800);
        vm.write_bytes(0x1800, b"\x41\xcf").unwrap();
        let run = cpu.run_steps(&vm, 12, StepOptions::default()).unwrap();
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::Hlt)),
            "{:?}",
            run.stop()
        );
        assert_eq!(
            (cpu.cpu.gpr[1], cpu.cpu.gpr[3], cpu.cpu.gpr[4]),
            (1, 1, 0x8000)
        );
        assert_eq!((run.steps(), cpu.retired_instructions()), (9, 9));
    }
}

fn trap_range(vm: &mut Vm, address: u64, size: u64) {
    let mut bytes = vec![0; 0x40000];
    vm.read_bytes(0, &mut bytes).unwrap();
    vm.unmap(0, 0x40000).unwrap();
    vm.map(0, address as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.map(address, size as usize, Prot::RW, RegionKind::Trap)
        .unwrap();
    vm.map(
        address + size,
        (0x40000 - address - size) as usize,
        Prot::RWX,
        RegionKind::Ram,
    )
    .unwrap();
    vm.write_bytes(0, &bytes[..address as usize]).unwrap();
    vm.write_bytes(address + size, &bytes[(address + size) as usize..])
        .unwrap();
}

#[test]
fn software_frame_retry_retains_overwritten_gate_and_consumed_instruction() {
    // Decision-16 §3, decision-22: consumed operands and the accepted A20 line
    // survive frame handoffs, even when earlier pushes overwrite the IDT/code.
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let (mut vm, mut cpu) = setup(generation, b"\xcd\x30");
        gate(&vm, 0x30, 0x86, 0x1800);
        trap_range(&mut vm, 0x317e, 2);
        cpu.cpu.gpr[4] = 0x3184;
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(
            matches!(
                run.stop(),
                StepStop::Exit(Exit::MmioWrite {
                    addr: 0x317e,
                    size: 2,
                    value: 0x1002,
                    ..
                })
            ),
            "{:?}",
            run.stop()
        );
        assert_eq!(
            (run.steps(), cpu.cpu.rip, cpu.cpu.gpr[4]),
            (0, 0x1000, 0x3184)
        );
        assert_eq!(word(&vm, 0x3180, 2), 8);
        vm.write_bytes(0x1000, b"\xf4\xf4").unwrap();
        cpu.set_a20_enabled(false);
        assert_eq!(cpu.pending_a20_enabled(), Some(false));
        cpu.complete_mmio_write();
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (cpu.cpu.rip, cpu.cpu.cs, cpu.cpu.gpr[4]),
            (0x1800, 8, 0x317e)
        );
        assert!(!cpu.a20_enabled());
        assert_eq!(cpu.retired_instructions(), 1);
    }
}

#[test]
fn accepted_exception_and_irq_delivery_resume_frame_before_handler_execution() {
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        for irq in [false, true] {
            let (mut vm, mut cpu) = setup_delivery(
                generation,
                if irq { b"\x90" } else { b"\x8e\xd8" },
                ExceptionDelivery::InGuest,
            );
            let vector = if irq { 0x30 } else { 13 };
            gate(&vm, vector, 0x86, 0x1800);
            vm.write_bytes(0x1800, b"\x43").unwrap();
            trap_range(&mut vm, if irq { 0x7ffa } else { 0x7ff8 }, 2);
            cpu.cpu.gpr[0] = 24;
            cpu.cpu.flags.if_ = true;
            if irq {
                cpu.inject_irq(vector);
            }
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(run.stop(), StepStop::Exit(Exit::MmioWrite { .. })),
                "{:?}",
                run.stop()
            );
            assert_eq!((run.steps(), cpu.retired_instructions()), (0, 0));
            assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[4]), (0x1000, 0x8000));
            assert!(!cpu.irq_ready());
            // Delivery does not re-execute/refetch the faulting instruction.
            vm.write_bytes(0x1000, b"\xf4\xf4").unwrap();
            cpu.complete_mmio_write();
            steps(&mut cpu, &vm, 1);
            assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[3]), (0x1801, 1));
            assert_eq!(cpu.retired_instructions(), 1);
            assert!(!cpu.has_pending_irq());
        }
    }
}

#[test]
fn idt_mmio_reads_resume_without_reaccepting_irq() {
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let (mut vm, mut cpu) = setup(generation, b"\x90");
        vm.write_bytes(0x1800, b"\xcf").unwrap();
        trap_range(&mut vm, 0x3180, 8);
        cpu.cpu.flags.if_ = true;
        cpu.inject_irq(0x30);
        for (address, answer) in [(0x3180, 0x0008_1800), (0x3184, 0x8600)] {
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 0);
            assert!(
                matches!(run.stop(),StepStop::Exit(Exit::MmioRead {addr,size:4,..}) if *addr==address),
                "{:?}",
                run.stop()
            );
            cpu.complete_mmio_read(answer);
        }
        steps(&mut cpu, &vm, 1);
        assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[4]), (0x1000, 0x8000));
        assert!(!cpu.has_pending_irq());
        assert_eq!(cpu.retired_instructions(), 1);
    }
}

#[test]
fn iretd_restores_fault_frame_rf_after_removing_error_code() {
    // I386 PRM §12.3.1.1 p. 12-7, §17 IRET pp. 17-76–78.
    let (vm, mut cpu) = setup_delivery(
        CpuGeneration::I386,
        b"\x8e\xd8\x90",
        ExceptionDelivery::InGuest,
    );
    cpu.cpu.gpr[0] = 24;
    gate(&vm, 13, 0x8e, 0x1800);
    // POP EAX (error code); POP EDX (fault EIP); ADD EDX,2; PUSH EDX; IRETD.
    vm.write_bytes(0x1800, b"\x66\x58\x66\x5a\x66\x83\xc2\x02\x66\x52\x66\xcf")
        .unwrap();
    steps(&mut cpu, &vm, 5);
    assert_eq!(cpu.cpu.gpr[0], 24);
    assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[4]), (0x1002, 0x8000));
    assert_eq!(cpu.cpu.rflags_system & 0x10000, 0x10000);
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.rflags_system & 0x10000, 0);
}

#[test]
fn stack_limit_failure_is_atomic_and_fetch_fault_is_delivered() {
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let (vm, mut cpu) = setup(generation, b"\xcd\x30");
        gate(&vm, 0x30, 0x86, 0x1800);
        cpu.cpu.gpr[4] = 4;
        outward(&mut cpu, &vm, 12, Some(0));
        let (vm, mut cpu) = setup_delivery(generation, b"\x90", ExceptionDelivery::InGuest);
        cpu.cpu.rip = 0x10000; // current cached CS limit is 0xffff
        gate(&vm, 13, 0x86, 0x1800);
        vm.write_bytes(0x1800, b"\xf4").unwrap();
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::Hlt)),
            "{:?}",
            run.stop()
        );
        assert_eq!(word(&vm, 0x7ff8, 2), 0);
        assert_eq!(word(&vm, 0x7ffa, 2), 0); // 16-bit gate truncates the return offset
    }
}

#[test]
fn coprocessor_unavailable_and_stack_fault_route_with_correct_frames() {
    // I286 PRM §10.2.2 Table 10-1 p. 10-4; I386 PRM §§11.1.2–11.1.4;
    // I286 PRM §9.4 Fig. 9-4 p. 9-5; I386 PRM §9.7 p. 9-10.
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        for (code, vector, error) in [
            (&b"\xd9\xe8"[..], 7, None),
            (&b"\x8e\xd0"[..], 12, Some(16)),
        ] {
            let (vm, mut cpu) = setup_delivery(generation, code, ExceptionDelivery::InGuest);
            cpu.cpu.legacy.cr0 |= 4; // EM
            cpu.cpu.gpr[0] = 16;
            vm.write_bytes(0x2015, &[0x13]).unwrap(); // new SS is not present
            gate(&vm, vector, 0x86, 0x1800);
            vm.write_bytes(0x1800, b"\xf4").unwrap();
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(run.stop(), StepStop::Exit(Exit::Hlt)),
                "{:?}",
                run.stop()
            );
            assert_eq!(word(&vm, 0x7ffa, 2), 0x1000);
            assert_eq!(
                cpu.cpu.gpr[4],
                if error.is_some() { 0x7ff8 } else { 0x7ffa }
            );
            if let Some(error) = error {
                assert_eq!(word(&vm, 0x7ff8, 2), error);
            }
        }
    }
}

#[test]
fn protected_sti_shadow_delays_irq_until_after_the_following_instruction() {
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let (vm, mut cpu) = setup(generation, b"\xfb\x43\xf4");
        gate(&vm, 0x30, 0x86, 0x1800);
        vm.write_bytes(0x1800, b"\xcf").unwrap();
        cpu.inject_irq(0x30);
        steps(&mut cpu, &vm, 1);
        assert!(!cpu.irq_ready());
        steps(&mut cpu, &vm, 1);
        assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[3]), (0x1002, 1));
        assert!(cpu.irq_ready());
        steps(&mut cpu, &vm, 1);
        assert_eq!(word(&vm, 0x7ffa, 2), 0x1002);
        assert_eq!(cpu.cpu.rip, 0x1002);
        assert!(!cpu.has_pending_irq());
    }
}

#[test]
fn db_trap_gate_precedes_irq_which_enters_before_first_debug_handler_instruction() {
    // I286 PRM §9.7.1 p. 9-14; I386 PRM §§9.3,12.3.1.4 pp. 9-4,12-8.
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let (vm, mut cpu) = setup_delivery(generation, b"\x90", ExceptionDelivery::InGuest);
        gate(&vm, 1, 0x87, 0x1800);
        gate(&vm, 0x30, 0x86, 0x1900);
        vm.write_bytes(0x1800, b"\x43").unwrap();
        vm.write_bytes(0x1900, b"\xf4").unwrap();
        cpu.cpu.rflags_system = 0x100;
        // STI sets IF while its shadow prevents acceptance at the first entry.
        vm.write_bytes(0x0fff, b"\xfb").unwrap();
        cpu.cpu.rip = 0x0fff;
        // Start with TF clear so STI does not itself trap.
        cpu.cpu.rflags_system = 0;
        steps(&mut cpu, &vm, 1);
        cpu.cpu.rflags_system = 0x100;
        cpu.inject_irq(0x30);
        let run = cpu.run_steps(&vm, 2, StepOptions::default()).unwrap();
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::Hlt)),
            "{:?}",
            run.stop()
        );
        assert_eq!(
            (cpu.cpu.rip, cpu.cpu.gpr[3], cpu.cpu.gpr[4]),
            (0x1901, 0, 0x7ff4)
        );
        assert_eq!(word(&vm, 0x7ffa, 2), 0x1001);
        assert_eq!(word(&vm, 0x7ff4, 2), 0x1800);
    }
}

// ---------------------------------------------------------------------------
// The NMI input of the I80286 and I80386 (xj-50w.19; spec.md §17.10):
// I286 PRM §5.2 pp. 5-4–6, §9.2 p. 9-3, §9.6 pp. 9-9–10, §11.6 p. 11-7,
// App. B pp. B-42, B-73, B-83; 80386 PRM §9.2 pp. 9-2–4, Table 9-3 p. 9-13,
// §12.3.1.4 p. 12-8, pp. 17-60, 17-109, 17-126; SDM Vol 3A §7.7.2 p. 7-6
// and §7.15 p. 7-37 (second witness).

use x86jit_core::{CpuModel, CpuState, UnsupportedSystemOperation};

const NMI_MODELS: [CpuModel; 2] = [CpuModel::I80286, CpuModel::I80386];
const HANDLER: u64 = 0x1800;

fn nmi_state(c: &CpuState) -> String {
    format!("{c:?}")
}
// A Machine at 0:1000 (real mode) or 8:1000 (protected, flat 16-bit
// descriptors and the IDT at 3000h, as `setup_delivery`), SP 8000h, with the
// NMI handler at HANDLER: IVT entry 2 or a 16-bit interrupt gate.
fn nmi_machine(
    model: CpuModel,
    protected: bool,
    code: &[u8],
    handler: &[u8],
    delivery: ExceptionDelivery,
) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(0x40000));
    vm.set_cpu_model(model);
    vm.set_exception_delivery(delivery);
    vm.map(0, 0x40000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, code).unwrap();
    vm.write_bytes(HANDLER, handler).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[4] = 0x8000;
    if protected {
        vm.write_bytes(0x2008, &[0xff, 0xff, 0, 0, 0, 0x9b, 0, 0])
            .unwrap();
        vm.write_bytes(0x2010, &[0xff, 0xff, 0, 0, 0, 0x93, 0, 0])
            .unwrap();
        cpu.cpu.cs = 8;
        cpu.cpu.ss = 16;
        cpu.cpu.legacy.cr0 = 1;
        cpu.cpu.legacy.gdtr.base = 0x2000;
        cpu.cpu.legacy.gdtr.limit = 23;
        cpu.cpu.legacy.idtr.base = 0x3000;
        cpu.cpu.legacy.idtr.limit = 0x7ff;
        gate(&vm, 2, 0x86, HANDLER as u32);
    } else {
        vm.mem.write(8, HANDLER, 4).unwrap();
    }
    (vm, cpu)
}
fn ivt(vm: &Vm, vector: u8, target: u64) {
    vm.mem.write(u64::from(vector) * 4, target, 4).unwrap();
}
fn stop(cpu: &mut Vcpu, vm: &Vm, budget: u64) -> (u64, StepStop) {
    let r = cpu.run_steps(vm, budget, StepOptions::default()).unwrap();
    (r.steps(), r.into_stop())
}
fn halted(cpu: &mut Vcpu, vm: &Vm, budget: u64) -> u64 {
    let (n, s) = stop(cpu, vm, budget);
    assert!(matches!(s, StepStop::Exit(Exit::Hlt)), "{s:?}");
    n
}

#[test]
fn nmi_delivery_is_external_vector_2_and_ignores_if() {
    // I286 PRM §9.2 p. 9-3 and Table 9-3 p. 9-9; 80386 PRM Table 9-1 p. 9-2:
    // vector 2, no acknowledge cycle; §9.4 p. 9-4: an external event, CPL
    // and gate DPL not checked. The entry is that of an external vector 2 in
    // both regimes (state and frame equal a queued IRQ 2 taken with IF = 1),
    // and IF does not mask it (I286 PRM §3.11.2 p. 3-28; 80386 PRM §4.1).
    for model in NMI_MODELS {
        for protected in [false, true] {
            let delivered = |nmi: bool| {
                let (vm, mut cpu) = nmi_machine(
                    model,
                    protected,
                    b"\x90",
                    b"\xf4",
                    ExceptionDelivery::InGuest,
                );
                cpu.cpu.flags.if_ = true;
                if nmi {
                    cpu.raise_nmi().unwrap();
                    assert!(!cpu.irq_ready(), "NMI has priority (I286 PRM p. 9-3)");
                } else {
                    cpu.inject_irq(2);
                }
                assert_eq!(halted(&mut cpu, &vm, 4), 1);
                let mut stack = [0; 6];
                vm.read_bytes(0x7ffa, &mut stack).unwrap();
                (nmi_state(&cpu.cpu), stack, cpu.retired_instructions())
            };
            assert_eq!(delivered(true), delivered(false), "{model:?} {protected}");
            let (vm, mut cpu) = nmi_machine(
                model,
                protected,
                b"\x90",
                b"\xf4",
                ExceptionDelivery::InGuest,
            );
            cpu.raise_nmi().unwrap();
            assert!(cpu.nmi_ready() && cpu.has_pending_nmi());
            assert_eq!(halted(&mut cpu, &vm, 4), 1);
            assert_eq!(word(&vm, 0x7ffa, 2), 0x1000);
            assert_eq!(word(&vm, 0x7ffe, 2) & 0x200, 0, "taken with IF = 0");
            assert!(!cpu.has_pending_nmi());
        }
    }
}

#[test]
fn a_queued_irq_waits_behind_the_nmi_interrupt_gate_and_follows_a_trap_gate() {
    // I286 PRM §5.2.2 p. 5-5 (real-mode entry clears IF), §9.6 p. 9-9 ("If
    // other interrupts remain enabled, they are processed before the first
    // instruction of the current interrupt handler"), §9.2 p. 9-3.
    for model in NMI_MODELS {
        // Real mode: IF cleared; the IRQ is taken after the handler's IRET.
        let (vm, mut cpu) = nmi_machine(
            model,
            false,
            b"\x90",
            b"\x90\xcf",
            ExceptionDelivery::InGuest,
        );
        ivt(&vm, 0x20, 0x1900);
        vm.write_bytes(0x1900, b"\xf4").unwrap();
        cpu.cpu.flags.if_ = true;
        cpu.inject_irq(0x20);
        cpu.raise_nmi().unwrap();
        step_to(&mut cpu, &vm, 1, (0, HANDLER + 1));
        assert!(cpu.has_pending_irq());
        assert_eq!(halted(&mut cpu, &vm, 4), 2); // IRET, then the IRQ's HLT
        assert_eq!((cpu.cpu.rip, word(&vm, 0x7ffa, 2)), (0x1901, 0x1000));
        assert!(!cpu.has_pending_irq());
        // Protected mode through an interrupt gate, which "enters with
        // interrupts disabled" (I286 PRM §9.4 p. 9-3): the IRQ waits until the
        // handler's IRET, and its frame then names the interrupted code.
        let (vm, mut cpu) = nmi_machine(
            model,
            true,
            b"\x90",
            b"\x90\xcf",
            ExceptionDelivery::InGuest,
        );
        gate(&vm, 0x30, 0x86, 0x1900);
        vm.write_bytes(0x1900, b"\xf4").unwrap();
        cpu.cpu.flags.if_ = true;
        cpu.inject_irq(0x30);
        cpu.raise_nmi().unwrap();
        step_to(&mut cpu, &vm, 1, (8, HANDLER + 1));
        assert!(cpu.has_pending_irq() && !cpu.cpu.flags.if_);
        assert_eq!(word(&vm, 0x7ffa, 2), 0x1000, "the NMI frame");
        assert_eq!(halted(&mut cpu, &vm, 4), 2); // IRET, then the IRQ's HLT
        assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (8, 0x1901));
        assert_eq!((word(&vm, 0x7ffa, 2), word(&vm, 0x7ffc, 2)), (0x1000, 8));
        assert_eq!(word(&vm, 0x7ffe, 2) & 0x200, 0x200, "IF restored by IRET");
        assert!(!cpu.has_pending_irq() && !cpu.has_pending_nmi());
        // Protected mode through a trap gate: NMI frame, then the IRQ frame,
        // before the NMI handler's first instruction.
        let (vm, mut cpu) = nmi_machine(model, true, b"\x90", b"\xf4", ExceptionDelivery::InGuest);
        gate(&vm, 2, 0x87, HANDLER as u32);
        gate(&vm, 0x30, 0x86, 0x1900);
        vm.write_bytes(0x1900, b"\xf4").unwrap();
        cpu.cpu.flags.if_ = true;
        cpu.inject_irq(0x30);
        cpu.raise_nmi().unwrap();
        assert_eq!(halted(&mut cpu, &vm, 4), 1);
        assert_eq!(cpu.cpu.rip, 0x1901);
        assert_eq!((word(&vm, 0x7ffa, 2), word(&vm, 0x7ffc, 2)), (0x1000, 8));
        assert_eq!((word(&vm, 0x7ff4, 2), word(&vm, 0x7ff6, 2)), (0x1800, 8));
        assert_eq!(word(&vm, 0x7ff8, 2) & 0x200, 0x200, "trap gate kept IF");
        assert!(!cpu.has_pending_irq() && !cpu.has_pending_nmi());
    }
}
fn step_to(cpu: &mut Vcpu, vm: &Vm, n: u64, at: (u16, u64)) {
    steps(cpu, vm, n);
    assert_eq!((cpu.cpu.cs, cpu.cpu.rip), at);
}

#[test]
fn the_step_frame_precedes_an_nmi_taken_before_the_step_handler() {
    // I286 PRM §9.7.1 p. 9-14, App. C item 12 p. C-2; 80386 PRM §12.3.1.4
    // p. 12-8, Table 9-2 p. 9-4: the single-step trap is delivered first and
    // the NMI is taken before the step handler's first instruction. MOV SS
    // holds the NMI (and the trap) one instruction, so both meet after NOP.
    for model in NMI_MODELS {
        let (vm, mut cpu) = nmi_machine(
            model,
            false,
            b"\x8e\xd0\x90\x90",
            b"\xf4",
            ExceptionDelivery::InGuest,
        );
        ivt(&vm, 1, 0x1900);
        vm.write_bytes(0x1900, b"\xf4").unwrap();
        cpu.cpu.rflags_system = 0x100;
        step_to(&mut cpu, &vm, 1, (0, 0x1002)); // MOV SS: no trap at its boundary
        cpu.raise_nmi().unwrap();
        assert!(!cpu.nmi_ready());
        assert_eq!(halted(&mut cpu, &vm, 4), 2); // NOP, then the NMI handler's HLT
        assert_eq!(cpu.cpu.rip, HANDLER + 1);
        assert_eq!(word(&vm, 0x7ffa, 2), 0x1003, "step frame: after NOP");
        assert_eq!(word(&vm, 0x7ffe, 2) & 0x100, 0x100);
        assert_eq!(word(&vm, 0x7ff4, 2), 0x1900, "NMI frame: the step handler");
        assert_eq!(word(&vm, 0x7ff8, 2) & 0x100, 0);
    }
}

#[test]
fn ss_loads_hold_nmi_at_each_boundary_and_lss_holds_nothing() {
    // I286 PRM App. B MOV p. B-73 ("inhibit all interrupts"), POP p. B-83
    // ("including NMI"); 80386 PRM §9.2.4 pp. 9-3–4, pp. 17-109, 17-126.
    // Each load holds the next boundary (spec.md §17.10); INT keeps
    // decision-16 §4. SDM Vol 3A §7.8.3 p. 7-9: LSS inhibits nothing.
    for model in NMI_MODELS {
        for code in [&b"\x8e\xd0\x90\x90"[..], b"\x17\x90\x90"] {
            let (vm, mut cpu) =
                nmi_machine(model, false, code, b"\xf4", ExceptionDelivery::InGuest);
            steps(&mut cpu, &vm, 1);
            cpu.raise_nmi().unwrap();
            assert!(!cpu.nmi_ready());
            steps(&mut cpu, &vm, 1);
            assert!(cpu.nmi_ready());
            let before = cpu.cpu.clone();
            assert_eq!(halted(&mut cpu, &vm, 4), 1);
            assert_eq!(word(&vm, before.gpr[4] - 6, 2), before.rip as u32);
        }
        // MOV SS,AX; MOV SS,BX; MOV SP,8000h; NOP.
        let (vm, mut cpu) = nmi_machine(
            model,
            false,
            b"\x8e\xd0\x8e\xd3\xbc\x00\x80\x90",
            b"\xf4",
            ExceptionDelivery::InGuest,
        );
        cpu.cpu.flags.if_ = true;
        steps(&mut cpu, &vm, 1);
        cpu.raise_nmi().unwrap();
        steps(&mut cpu, &vm, 1);
        assert!(!cpu.nmi_ready(), "the second load holds NMI again");
        assert!(cpu.irq_ready(), "INT is not held again (decision-16 §4)");
        steps(&mut cpu, &vm, 1);
        assert!(cpu.nmi_ready() && !cpu.irq_ready());
        assert_eq!(halted(&mut cpu, &vm, 4), 1);
        assert_eq!(word(&vm, 0x7ffa, 2), 0x1007);
        // MOV DS holds nothing on these models; LSS holds nothing (80386).
        let mut forms = vec![&b"\x8e\xd8\x90"[..]];
        if model == CpuModel::I80386 {
            forms.push(b"\x0f\xb2\x26\x00\x01\x90");
        }
        for code in forms {
            let (vm, mut cpu) =
                nmi_machine(model, false, code, b"\xf4", ExceptionDelivery::InGuest);
            vm.mem.write(0x100, 0x8000, 4).unwrap();
            steps(&mut cpu, &vm, 1);
            cpu.raise_nmi().unwrap();
            assert!(cpu.nmi_ready(), "{code:x?}");
        }
    }
}

#[test]
fn blocking_until_iret_remembers_one_edge_on_the_286_and_ignores_it_on_the_386() {
    // I286 PRM §9.2 p. 9-3 ("Only one NMI request can be remembered",
    // "serviced after the first IRET"), App. C item 14 p. C-3; 80386 PRM
    // §9.2.1 p. 9-3 ("ignores"), §14.7 item 13 p. 14-7.
    for model in NMI_MODELS {
        let remember = model == CpuModel::I80286;
        let (vm, mut cpu) = nmi_machine(
            model,
            false,
            b"\x90\x90",
            b"\x90\x90\xcf",
            ExceptionDelivery::InGuest,
        );
        cpu.raise_nmi().unwrap();
        step_to(&mut cpu, &vm, 1, (0, HANDLER + 1));
        assert!(!cpu.nmi_ready(), "blocked from acceptance");
        cpu.raise_nmi().unwrap();
        cpu.nmi_line().unwrap().raise();
        assert_eq!(cpu.has_pending_nmi(), remember);
        assert!(cpu.nmi_line().unwrap().is_raised());
        step_to(&mut cpu, &vm, 1, (0, HANDLER + 2));
        // The edge left the latch at that boundary: remembered or ignored.
        assert!(!cpu.nmi_line().unwrap().is_raised());
        assert_eq!(cpu.has_pending_nmi(), remember);
        cpu.raise_nmi().unwrap(); // a third edge: still one remembered
        step_to(&mut cpu, &vm, 1, (0, 0x1000)); // the IRET ends blocking
        assert!(cpu.nmi_ready());
        assert_eq!(cpu.has_pending_nmi(), remember);
        let before = cpu.cpu.clone();
        let retired = cpu.retired_instructions();
        steps(&mut cpu, &vm, 1);
        if remember {
            // Serviced after the first IRET, before the next instruction.
            assert_eq!((cpu.cpu.rip, word(&vm, 0x7ffa, 2)), (HANDLER + 1, 0x1000));
            assert_eq!(cpu.cpu.gpr[4], 0x7ffa);
            assert!(!cpu.has_pending_nmi(), "only one was remembered");
        } else {
            let mut expected = before.clone();
            expected.rip = 0x1001;
            assert_eq!(nmi_state(&cpu.cpu), nmi_state(&expected));
        }
        assert_eq!(cpu.retired_instructions(), retired + 1);
        // An edge after the IRET is delivered.
        if !remember {
            cpu.raise_nmi().unwrap();
            steps(&mut cpu, &vm, 1);
            assert_eq!((cpu.cpu.rip, word(&vm, 0x7ffa, 2)), (HANDLER + 1, 0x1001));
        }
    }
}

#[test]
fn a_nested_irq_handlers_iret_ends_286_blocking() {
    // I286 PRM §9.2 p. 9-3: the remembered NMI is serviced after the first
    // IRET, which here is the nested IRQ handler's.
    let (vm, mut cpu) = nmi_machine(
        CpuModel::I80286,
        false,
        b"\x90",
        b"\x90\x90\xf4",
        ExceptionDelivery::InGuest,
    );
    ivt(&vm, 0x20, 0x1900);
    vm.write_bytes(0x1900, b"\xcf").unwrap();
    cpu.raise_nmi().unwrap();
    step_to(&mut cpu, &vm, 1, (0, HANDLER + 1));
    cpu.raise_nmi().unwrap();
    cpu.cpu.flags.if_ = true;
    cpu.inject_irq(0x20);
    // IRQ entry (zero steps) and its IRET (one step) end blocking.
    steps(&mut cpu, &vm, 1);
    assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[4]), (HANDLER + 1, 0x7ffa));
    assert!(cpu.has_pending_nmi() && cpu.nmi_ready(), "latched again");
    // The remembered NMI nests in the NMI handler, before its next NOP.
    steps(&mut cpu, &vm, 1);
    assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[4]), (HANDLER + 1, 0x7ff4));
    assert_eq!(word(&vm, 0x7ff4, 2), HANDLER as u32 + 1);
    assert!(!cpu.has_pending_nmi() && !cpu.nmi_ready());
}

#[test]
fn a_handler_that_never_irets_stays_blocked() {
    // I286 PRM §9.2 p. 9-3; 80386 PRM §9.2.1 p. 9-3: blocking lasts until
    // the next IRET; a far jump out of the handler does not end it.
    for model in NMI_MODELS {
        let (vm, mut cpu) = nmi_machine(
            model,
            false,
            b"\x90\x90\x90\xeb\xfd",
            b"\xea\x00\x10\x00\x00",
            ExceptionDelivery::InGuest,
        );
        cpu.raise_nmi().unwrap();
        step_to(&mut cpu, &vm, 1, (0, 0x1000));
        cpu.raise_nmi().unwrap();
        for _ in 0..4 {
            steps(&mut cpu, &vm, 1000);
            assert!(!cpu.nmi_ready());
            assert_eq!(cpu.cpu.gpr[4], 0x7ffa, "no second NMI frame");
        }
        assert_eq!(cpu.has_pending_nmi(), model == CpuModel::I80286);
        // reset_execution_state ends blocking (ASSUMED) and drops the edge.
        cpu.reset_execution_state(x86jit_core::ResetOptions::default());
        assert!(cpu.nmi_ready() && !cpu.has_pending_nmi());
    }
}

#[test]
fn iret_through_a_device_unblocks_at_completion_and_a_faulting_iret_unblocks() {
    // spec.md §17.10: a bus handoff inside IRET does not end blocking; its
    // completion does. A faulting IRET ends it (SDM Vol 3A §7.7.2 p. 7-6
    // for the 80386; ASSUMED for the 80286).
    for model in NMI_MODELS {
        // MOV SP,4000h; IRET with the frame in a trapped page.
        let mut vm = Vm::new(VmConfig::flat(0x40000));
        vm.set_cpu_model(model);
        vm.set_exception_delivery(ExceptionDelivery::InGuest);
        vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x4000, 0x1000, Prot::RWX, RegionKind::Trap).unwrap();
        vm.map(0x5000, 0x3b000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, b"\x90\x90").unwrap();
        vm.write_bytes(HANDLER, b"\xbc\x00\x40\xcf").unwrap();
        vm.mem.write(8, HANDLER, 4).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x1000;
        cpu.cpu.gpr[4] = 0x8000;
        cpu.raise_nmi().unwrap();
        step_to(&mut cpu, &vm, 1, (0, HANDLER + 3));
        let line = cpu.nmi_line().unwrap();
        let remember = model == CpuModel::I80286;
        let mut answers = [0x1001u64, 0, 0x0002].into_iter();
        let mut exits = 0;
        loop {
            let (n, s) = stop(&mut cpu, &vm, 1);
            match s {
                StepStop::Exit(Exit::MmioRead { size, .. }) => {
                    assert_eq!((n, size), (0, 2));
                    if exits == 1 {
                        // The edge raised during the first handoff left the
                        // word at this call's first boundary as a blocked
                        // edge: remembered (80286) or ignored (80386).
                        assert!(!line.is_raised(), "still blocked during the handoff");
                        assert_eq!(cpu.has_pending_nmi(), remember);
                    }
                    cpu.complete_mmio_read(answers.next().unwrap());
                    if exits == 0 {
                        cpu.raise_nmi().unwrap();
                    }
                    exits += 1;
                }
                StepStop::Exit(Exit::BudgetExhausted) => break,
                other => panic!("{other:?}"),
            }
        }
        assert_eq!(exits, 3);
        assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0, 0x1001));
        assert!(cpu.nmi_ready(), "the completed IRET ended blocking");
        // I286 PRM §9.2 p. 9-3: the remembered edge is latched again,
        // "serviced after the first IRET"; the 80386 ignored it.
        assert_eq!(
            (cpu.has_pending_nmi(), line.is_raised()),
            (remember, remember)
        );

        // Protected: the handler's IRET pops CS 40h, beyond the GDT: #GP.
        let (vm, mut cpu) = nmi_machine(
            model,
            true,
            b"\x90",
            b"\x89\xe5\xc7\x46\x02\x40\x00\xcf",
            ExceptionDelivery::InGuest,
        );
        gate(&vm, 13, 0x86, 0x1900);
        vm.write_bytes(0x1900, b"\xf4").unwrap();
        cpu.raise_nmi().unwrap();
        steps(&mut cpu, &vm, 2);
        assert!(!cpu.nmi_ready());
        cpu.raise_nmi().unwrap();
        let (n, s) = stop(&mut cpu, &vm, 4);
        assert!(matches!(s, StepStop::Exit(Exit::Hlt)), "{s:?}");
        if model == CpuModel::I80286 {
            // The remembered edge is serviced before the #GP handler's first
            // instruction (its frame names 1900h, above the #GP frame with
            // its error code); that NMI handler's IRET faults again.
            assert_eq!((n, cpu.cpu.rip), (3, 0x1901));
            assert_eq!(word(&vm, 0x7fec, 2), 0x1900);
        } else {
            assert_eq!((n, cpu.cpu.rip), (1, 0x1901));
            assert!(cpu.nmi_ready(), "unblocked by the faulting IRET");
            assert!(!cpu.has_pending_nmi(), "the blocked edge was ignored");
        }
    }
}

#[test]
fn iretd_into_v86_ends_nmi_blocking() {
    // SDM Vol 2A IRET pp. 3-490–493: V86 entry ends NMI blocking.
    let (vm, mut cpu) = nmi_machine(
        CpuModel::I80386,
        true,
        b"\x90",
        b"\x90\x66\xcf",
        ExceptionDelivery::InGuest,
    );
    cpu.raise_nmi().unwrap();
    step_to(&mut cpu, &vm, 1, (8, HANDLER + 1));
    // Replace the frame with a 32-bit one whose EFLAGS sets VM.
    let sp = cpu.cpu.gpr[4];
    vm.mem.write(sp, 0x1000, 4).unwrap();
    vm.mem.write(sp + 4, 8, 4).unwrap();
    vm.mem.write(sp + 8, 0x2_0002, 4).unwrap();
    let (n, s) = stop(&mut cpu, &vm, 1);
    assert!(matches!(s, StepStop::Exit(Exit::BudgetExhausted)), "{s:?}");
    assert_eq!(n, 1);
    assert_eq!(cpu.cpu.rflags_system & 0x20000, 0x20000);
    assert_eq!(cpu.cpu.legacy.cpl, 3);
    assert!(cpu.nmi_ready());
}

#[test]
fn an_nmi_that_never_entered_the_guest_leaves_no_blocking() {
    // spec.md §17.10, "Undoing an acceptance". Real-address Outward: the
    // delivery fault is returned and the edge latched again; protected
    // Outward: the NMI is lost (386 PRM Table 9-6 note 3 p. 9-20). Either
    // way blocking is clear.
    for model in NMI_MODELS {
        let (vm, mut cpu) = nmi_machine(model, false, b"\x90", b"\xf4", ExceptionDelivery::Outward);
        cpu.cpu.legacy.idtr.limit = 7;
        cpu.raise_nmi().unwrap();
        let before = cpu.cpu.clone();
        let (n, s) = stop(&mut cpu, &vm, 1);
        let vector = if model == CpuModel::I80286 { 13 } else { 8 };
        assert!(
            matches!(s, StepStop::Exit(Exit::Exception { vector: v, .. }) if v == vector),
            "{s:?}"
        );
        assert_eq!(n, 0);
        assert_eq!(nmi_state(&cpu.cpu), nmi_state(&before));
        assert!(cpu.has_pending_nmi() && cpu.nmi_ready(), "latched again");
        cpu.cpu.legacy.idtr.limit = 0x3ff;
        assert_eq!(halted(&mut cpu, &vm, 4), 1);

        let (vm, mut cpu) = nmi_machine(model, true, b"\x90", b"\xf4", ExceptionDelivery::Outward);
        gate(&vm, 2, 0x06, HANDLER as u32);
        cpu.raise_nmi().unwrap();
        let (n, s) = stop(&mut cpu, &vm, 1);
        assert!(
            matches!(s, StepStop::Exit(Exit::Exception { vector: 11, .. })),
            "{s:?}"
        );
        assert_eq!(n, 0);
        assert!(!cpu.has_pending_nmi() && cpu.nmi_ready(), "lost, unblocked");

        // A released continuation: the IDT read hands off, then the host
        // moves CS:IP. The NMI is latched again and taken at the new IP.
        let mut vm = Vm::new(VmConfig::flat(0x40000));
        vm.set_cpu_model(model);
        vm.map(0, 0x1000, Prot::RWX, RegionKind::Trap).unwrap();
        vm.map(0x1000, 0x3f000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, b"\x90\x90").unwrap();
        vm.write_bytes(HANDLER, b"\xf4").unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x1000;
        cpu.cpu.gpr[4] = 0x8000;
        cpu.raise_nmi().unwrap();
        let (n, s) = stop(&mut cpu, &vm, 1);
        assert!(
            matches!(s, StepStop::Exit(Exit::MmioRead { addr: 8, .. })),
            "{s:?}"
        );
        assert_eq!(n, 0);
        assert!(!cpu.has_pending_nmi() && !cpu.nmi_ready(), "accepted");
        cpu.reset_execution_state(x86jit_core::ResetOptions::default());
        assert!(cpu.nmi_ready() && !cpu.has_pending_nmi(), "reset clears");
        cpu.raise_nmi().unwrap();
        let (_, s) = stop(&mut cpu, &vm, 1);
        assert!(matches!(s, StepStop::Exit(Exit::MmioRead { addr: 8, .. })));
        cpu.cpu.rip = 0x1001;
        let (_, s) = stop(&mut cpu, &vm, 1);
        assert!(
            matches!(s, StepStop::Exit(Exit::MmioRead { addr: 8, .. })),
            "{s:?}"
        );
        cpu.complete_mmio_read(HANDLER);
        let (_, s) = stop(&mut cpu, &vm, 1);
        assert!(
            matches!(s, StepStop::Exit(Exit::MmioRead { addr: 10, .. })),
            "{s:?}"
        );
        cpu.complete_mmio_read(0);
        assert_eq!(halted(&mut cpu, &vm, 4), 1);
        assert_eq!(word(&vm, 0x7ffa, 2), 0x1001, "taken at the new IP");
    }
}

#[test]
fn an_nmi_delivery_fault_is_serial_and_blocking_holds_to_its_iret() {
    // I286 PRM §9.6.2 p. 9-10 (no double fault for vector 2); 386 PRM Table
    // 9-3 p. 9-13 ("Benign"); SDM Vol 3A §7.7.2 p. 7-6: NMIs stay blocked
    // through the nested exception, whose IRET ends blocking.
    for model in NMI_MODELS {
        let (vm, mut cpu) = nmi_machine(model, true, b"\x90", b"\xf4", ExceptionDelivery::InGuest);
        gate(&vm, 2, 0x06, HANDLER as u32);
        gate(&vm, 8, 0x86, 0x1a00);
        gate(&vm, 11, 0x86, 0x1900);
        vm.write_bytes(0x1900, b"\x83\xc4\x02\xcf").unwrap(); // ADD SP,2; IRET
        cpu.raise_nmi().unwrap();
        step_to(&mut cpu, &vm, 1, (8, 0x1903));
        assert_eq!(word(&vm, 0x7ff8, 2), 0x13, "IDT entry 2, EXT = 1");
        assert_eq!(word(&vm, 0x7ffa, 2), 0x1000);
        assert!(!cpu.nmi_ready(), "blocked inside the #NP handler");
        steps(&mut cpu, &vm, 1); // IRET
        assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[4]), (0x1000, 0x8000));
        assert!(
            cpu.nmi_ready() && !cpu.has_pending_nmi(),
            "the NMI was lost"
        );
    }
}

#[test]
fn a_released_serial_exception_after_a_faulting_nmi_delivery_ends_blocking() {
    // spec.md §17.10: the NMI whose delivery faults is lost and its blocking
    // stays for the serial exception's handler (SDM Vol 3A §7.7.2 p. 7-6).
    // When that exception's delivery never enters the guest either (here the
    // host moves CS:IP while IDT entry 11 is read from a device), no handler
    // will IRET: blocking ends with it, and a later edge is delivered.
    for model in NMI_MODELS {
        let (mut vm, mut cpu) = nmi_machine(
            model,
            true,
            b"\x90\x90",
            b"\xf4",
            ExceptionDelivery::InGuest,
        );
        gate(&vm, 2, 0x06, HANDLER as u32); // not present: #NP(13h)
        trap_range(&mut vm, 0x3058, 8);
        cpu.raise_nmi().unwrap();
        let (n, s) = stop(&mut cpu, &vm, 1);
        assert!(
            matches!(s, StepStop::Exit(Exit::MmioRead { addr: 0x3058, .. })),
            "{s:?}"
        );
        assert_eq!(n, 0);
        assert!(!cpu.has_pending_nmi() && !cpu.nmi_ready());
        cpu.cpu.rip = 0x1001;
        steps(&mut cpu, &vm, 1);
        assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[4]), (0x1002, 0x8000));
        assert!(cpu.nmi_ready() && !cpu.has_pending_nmi(), "lost, unblocked");
        gate(&vm, 2, 0x86, HANDLER as u32);
        cpu.raise_nmi().unwrap();
        assert_eq!(halted(&mut cpu, &vm, 4), 1);
        assert_eq!((cpu.cpu.rip, word(&vm, 0x7ffa, 2)), (HANDLER + 1, 0x1002));
    }
}

#[test]
fn a_released_nmi_taken_out_of_shutdown_restores_the_shutdown() {
    // spec.md §17.10: an NMI accepted out of shutdown whose delivery is
    // released (the host moves CS:IP while IDT entry 2 is read from a device)
    // is latched again with the shutdown restored. The next acceptance is
    // again one out of shutdown, so its faulting delivery re-enters shutdown
    // (I286 PRM §9.6.2 p. 9-10) rather than delivering #NP serially through
    // the present gate 11.
    for model in NMI_MODELS {
        let (mut vm, mut cpu) = nmi_machine(
            model,
            true,
            b"\x8e\xd8",
            b"\xf4",
            ExceptionDelivery::InGuest,
        );
        gate(&vm, 11, 0x86, 0x1900);
        vm.write_bytes(0x1900, b"\xf4").unwrap();
        trap_range(&mut vm, 0x3010, 8);
        cpu.cpu.gpr[0] = 24; // MOV DS: #GP, then #DF, then shutdown
        let (n, s) = stop(&mut cpu, &vm, 1);
        assert!(
            matches!(s, StepStop::Exit(Exit::Shutdown { addr: 0x1000, .. })),
            "{s:?}"
        );
        assert_eq!(n, 0);
        cpu.raise_nmi().unwrap();
        let (_, s) = stop(&mut cpu, &vm, 1);
        assert!(
            matches!(s, StepStop::Exit(Exit::MmioRead { addr: 0x3010, .. })),
            "{s:?}"
        );
        assert!(!cpu.has_pending_nmi() && !cpu.nmi_ready(), "accepted");
        cpu.cpu.rip = 0x1002;
        let (_, s) = stop(&mut cpu, &vm, 1);
        assert!(
            matches!(s, StepStop::Exit(Exit::MmioRead { addr: 0x3010, .. })),
            "accepted again: {s:?}"
        );
        cpu.complete_mmio_read((8 << 16) | HANDLER);
        let (_, s) = stop(&mut cpu, &vm, 1);
        assert!(
            matches!(s, StepStop::Exit(Exit::MmioRead { addr: 0x3014, .. })),
            "{s:?}"
        );
        cpu.complete_mmio_read(0x0600); // not present
        let (n, s) = stop(&mut cpu, &vm, 4);
        assert!(
            matches!(s, StepStop::Exit(Exit::Shutdown { addr: 0x1002, .. })),
            "{s:?}"
        );
        assert_eq!((n, cpu.cpu.gpr[4]), (0, 0x8000));
        assert!(!cpu.nmi_ready(), "blocked: only reset ends it");
    }
}

#[test]
fn a_nonresumable_refusal_during_nmi_delivery_latches_the_edge_again() {
    // spec.md §17.10: an accepted NMI that never entered the guest leaves no
    // blocking behind. An engine refusal with no continuation (the 80386
    // gate's target code descriptor sets L: `LegacyEntryFacility`) puts the
    // edge back, as a queued IRQ stays queued there.
    let (vm, mut cpu) = nmi_machine(
        CpuModel::I80386,
        true,
        b"\x90",
        b"\xf4",
        ExceptionDelivery::InGuest,
    );
    cpu.cpu.legacy.gdtr.limit = 31;
    vm.write_bytes(0x2018, &[0xff, 0xff, 0, 0, 0, 0x9b, 0x20, 0])
        .unwrap();
    vm.mem.write(0x3010, (0x18 << 16) | HANDLER, 4).unwrap();
    cpu.raise_nmi().unwrap();
    let before = cpu.cpu.clone();
    let (n, s) = stop(&mut cpu, &vm, 1);
    assert!(
        matches!(
            s,
            StepStop::Exit(Exit::UnsupportedSystemContext {
                operation: UnsupportedSystemOperation::LegacyEntryFacility,
                ..
            })
        ),
        "{s:?}"
    );
    assert_eq!(n, 0);
    assert_eq!(nmi_state(&cpu.cpu), nmi_state(&before));
    assert!(
        cpu.has_pending_nmi() && cpu.nmi_ready(),
        "latched again, unblocked"
    );
    vm.write_bytes(0x201e, &[0]).unwrap();
    assert_eq!(halted(&mut cpu, &vm, 4), 1);
    assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0x18, HANDLER + 1));
    assert_eq!(word(&vm, 0x7ffa, 2), 0x1000);
    assert!(
        !cpu.has_pending_nmi() && !cpu.nmi_ready(),
        "blocked in the handler"
    );
}

#[test]
fn an_edge_raised_during_a_blocked_run_keeps_the_sti_shadow() {
    // spec.md §17.10: the walk and the dispatcher apply one NMI predicate, NMI
    // blocking included. The handler is entered with IF = 0 and never left:
    // STI; CLI; JMP $-4, so its only IF = 1 boundary is the one the STI
    // shadow holds for INTR (SDM Vol 3A §7.8.1, as for `irq_ready`). An edge
    // storm from another thread while it runs is remembered or ignored, never
    // delivered, and does not end that shadow: the queued IRQ is not taken.
    for model in NMI_MODELS {
        let (vm, mut cpu) = nmi_machine(
            model,
            false,
            b"\x90",
            b"\xfb\xfa\xeb\xfc",
            ExceptionDelivery::InGuest,
        );
        ivt(&vm, 0x20, 0x1900);
        vm.write_bytes(0x1900, b"\xf4").unwrap();
        cpu.raise_nmi().unwrap();
        step_to(&mut cpu, &vm, 1, (0, HANDLER + 1)); // STI
        cpu.inject_irq(0x20);
        let line = cpu.nmi_line().unwrap();
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let raiser = {
            let done = std::sync::Arc::clone(&done);
            std::thread::spawn(move || {
                let mut edges = 0u64;
                while !done.load(std::sync::atomic::Ordering::Relaxed) {
                    line.raise();
                    edges += 1;
                    std::hint::spin_loop();
                }
                edges
            })
        };
        let run = cpu.run_steps(&vm, 50_000, StepOptions::default());
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(raiser.join().unwrap() > 0);
        let run = run.unwrap();
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
            "{model:?}: {:?} at {:#x}",
            run.stop(),
            cpu.cpu.rip
        );
        assert_eq!(run.steps(), 50_000);
        assert!(cpu.has_pending_irq(), "never taken in the STI shadow");
        assert_eq!(cpu.cpu.gpr[4], 0x7ffa, "one NMI frame");
        assert!(!cpu.nmi_ready(), "still blocked");
    }
}

#[test]
fn hlt_rep_and_the_blocked_hlt() {
    // I286 PRM App. B HLT p. B-42, §3.11.2 p. 3-29; 80386 PRM p. 17-60: NMI
    // resumes a halt, saving the next IP. Inside the handler the 80286 keeps
    // the remembered edge until an IRET (ready stays false); the 80386 has
    // none. §3.7.2 p. 3-22, 80386 PRM §9.2 p. 9-2: between repetitions.
    for model in NMI_MODELS {
        let (vm, mut cpu) = nmi_machine(model, false, b"\xf4", b"\xf4", ExceptionDelivery::InGuest);
        assert_eq!(halted(&mut cpu, &vm, 4), 1);
        cpu.raise_nmi().unwrap();
        assert!(cpu.has_pending_nmi() && cpu.nmi_ready());
        assert_eq!(halted(&mut cpu, &vm, 4), 1);
        assert_eq!(word(&vm, 0x7ffa, 2), 0x1001);
        cpu.raise_nmi().unwrap();
        assert!(!cpu.nmi_ready());
        assert_eq!(cpu.has_pending_nmi(), model == CpuModel::I80286);

        let (vm, mut cpu) = nmi_machine(
            model,
            false,
            b"\xf3\xa4\x90",
            b"\xf4",
            ExceptionDelivery::InGuest,
        );
        cpu.cpu.gpr[1] = 3;
        cpu.cpu.gpr[6] = 0x3000;
        cpu.cpu.gpr[7] = 0x3100;
        steps(&mut cpu, &vm, 1);
        cpu.raise_nmi().unwrap();
        assert_eq!(halted(&mut cpu, &vm, 4), 1);
        assert_eq!((word(&vm, 0x7ffa, 2), cpu.cpu.gpr[1]), (0x1000, 2));
    }
}

#[test]
fn nmi_ends_shutdown_unless_its_delivery_faults() {
    // I286 PRM §9.6.2 p. 9-10, §11.6 p. 11-7: "Either NMI or RESET can force
    // the CPU out of shutdown ... if no errors occur while processing the NMI
    // interrupt; otherwise ... only via the RESET input"; SDM Vol 3A §7.15
    // p. 7-37 for the 80386. The saved CS:IP is the one held at shutdown
    // (ASSUMED); a delivery fault re-enters shutdown, still blocked.
    for model in NMI_MODELS {
        for gate_present in [true, false] {
            let (vm, mut cpu) = nmi_machine(
                model,
                true,
                b"\x8e\xd8",
                b"\xf4",
                ExceptionDelivery::InGuest,
            );
            if !gate_present {
                gate(&vm, 2, 0, 0);
            }
            cpu.cpu.gpr[0] = 24;
            let (n, s) = stop(&mut cpu, &vm, 1);
            assert!(matches!(
                s,
                StepStop::Exit(Exit::Shutdown { addr: 0x1000, .. })
            ));
            assert_eq!(n, 0);
            assert!(matches!(cpu.run(&vm, None), Exit::Shutdown { .. }));
            assert!(cpu.nmi_ready() && !cpu.irq_ready());
            cpu.raise_nmi().unwrap();
            // N = 0 and a host request leave the shutdown and the latch.
            assert!(matches!(
                stop(&mut cpu, &vm, 0).1,
                StepStop::Exit(Exit::BudgetExhausted)
            ));
            cpu.exit_request().request_exit();
            assert!(matches!(
                stop(&mut cpu, &vm, 4).1,
                StepStop::Exit(Exit::Interrupted)
            ));
            assert!(cpu.has_pending_nmi());
            let (n, s) = stop(&mut cpu, &vm, 4);
            if gate_present {
                assert!(matches!(s, StepStop::Exit(Exit::Hlt)), "{s:?}");
                assert_eq!(n, 1);
                assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (8, HANDLER + 1));
                assert_eq!((word(&vm, 0x7ffa, 2), word(&vm, 0x7ffc, 2)), (0x1000, 8));
                assert!(!cpu.nmi_ready(), "blocked in the handler");
            } else {
                assert!(matches!(s, StepStop::Exit(Exit::Shutdown { .. })), "{s:?}");
                assert_eq!(n, 0);
                assert!(!cpu.nmi_ready(), "blocked: only reset ends it");
                cpu.raise_nmi().unwrap();
                assert!(matches!(
                    stop(&mut cpu, &vm, 4).1,
                    StepStop::Exit(Exit::Shutdown { .. })
                ));
            }
        }
    }
}

#[test]
fn nmi_readiness_follows_completions_and_application_has_no_input() {
    // spec.md §17.10: nmi_ready is false with an IN or MMIO completion
    // outstanding; irq_ready is false exactly while an NMI is pending and
    // ready. Application vCPUs refuse the input with no change.
    for model in NMI_MODELS {
        let (vm, mut cpu) = nmi_machine(
            model,
            false,
            b"\xe4\x10\x90",
            b"\xf4",
            ExceptionDelivery::InGuest,
        );
        cpu.cpu.flags.if_ = true;
        assert!(cpu.nmi_ready() && cpu.irq_ready() && !cpu.has_pending_nmi());
        let (_, s) = stop(&mut cpu, &vm, 1);
        assert!(matches!(s, StepStop::Exit(Exit::PortIo { .. })));
        cpu.raise_nmi().unwrap();
        assert!(!cpu.nmi_ready() && !cpu.irq_ready());
        cpu.complete_port_in(0x5a);
        assert!(cpu.nmi_ready() && !cpu.irq_ready());
        assert_eq!(halted(&mut cpu, &vm, 4), 1);
    }
    for mode in [
        x86jit_core::CpuMode::Long64,
        x86jit_core::CpuMode::Compat32,
        x86jit_core::CpuMode::Real16,
    ] {
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        vm.set_cpu_mode(mode);
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, b"\x90\xf4").unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x1000;
        assert_eq!(cpu.raise_nmi(), Err(x86jit_core::NmiUnsupported));
        assert!(cpu.nmi_line().is_none());
        assert!(!cpu.has_pending_nmi() && !cpu.nmi_ready());
        // The dispatch word is unchanged: compiled code exits on any bit.
        assert!(matches!(cpu.run(&vm, Some(64)), Exit::Hlt), "{mode:?}");
        assert_eq!(cpu.cpu.rip, 0x1002);
    }
    let _ = UnsupportedSystemOperation::NecStepWithNmi;
}

#[test]
fn ins_completion_then_nmi_then_irq() {
    // spec.md §17.10 boundary order: the service steps (an answered INS
    // completes), then the NMI, then a queued IRQ; NMI first (I286 PRM
    // Table 5-1 p. 5-4; 80386 PRM Table 9-2 p. 9-4). The real-mode NMI entry
    // clears IF (I286 PRM §5.2.2 p. 5-5), so the IRQ stays queued.
    for model in NMI_MODELS {
        let (vm, mut cpu) = nmi_machine(
            model,
            false,
            b"\x6c\x90\xf4",
            b"\xf4",
            ExceptionDelivery::InGuest,
        );
        ivt(&vm, 0x20, 0x1900);
        vm.write_bytes(0x1900, b"\xf4").unwrap();
        cpu.cpu.gpr[7] = 0x3000;
        cpu.cpu.gpr[2] = 0x10;
        cpu.cpu.flags.if_ = true;
        let (_, s) = stop(&mut cpu, &vm, 1);
        assert!(matches!(s, StepStop::Exit(Exit::PortIo { .. })), "{s:?}");
        cpu.complete_port_in(0x5a);
        cpu.inject_irq(0x20);
        cpu.raise_nmi().unwrap();
        let _ = halted(&mut cpu, &vm, 8);
        assert_eq!(vm.mem.read(0x3000, 1).unwrap(), 0x5a);
        assert_eq!(cpu.cpu.rip, HANDLER + 1, "{model:?}");
        assert_eq!(cpu.cpu.gpr[4], 0x7ffa, "{model:?}: the NMI frame only");
        assert_eq!(word(&vm, 0x7ffa, 2), 0x1001, "{model:?}");
        assert!(cpu.has_pending_irq(), "{model:?}");
    }
}

#[test]
fn port_debug_step_trap_precedes_nmi() {
    // I286 PRM §9.7.1 p. 9-14; 80386 PRM §12.3.1.4 p. 12-8: the step frame
    // first, then the NMI before the step handler's first instruction. The
    // OUT hands off before its step trap; spec.md §17.10 makes that trap the
    // owned continuation, taken before the NMI.
    for model in NMI_MODELS {
        let (vm, mut cpu) = nmi_machine(
            model,
            false,
            b"\xe6\x10\x90\xf4",
            b"\xf4",
            ExceptionDelivery::InGuest,
        );
        ivt(&vm, 1, 0x1900);
        vm.write_bytes(0x1900, b"\xf4").unwrap();
        cpu.cpu.rflags_system = 0x100;
        let (_, s) = stop(&mut cpu, &vm, 1);
        assert!(matches!(s, StepStop::Exit(Exit::PortIo { .. })), "{s:?}");
        cpu.raise_nmi().unwrap();
        let _ = halted(&mut cpu, &vm, 8);
        assert_eq!(cpu.cpu.rip, HANDLER + 1, "{model:?}");
        assert_eq!(word(&vm, 0x7ffa, 2), 0x1002, "{model:?} step frame");
        assert_eq!(word(&vm, 0x7ff4, 2), 0x1900, "{model:?} NMI frame");
    }
}

#[test]
fn one_step_host_request_waits_for_an_answered_ins() {
    // spec.md §17.10 boundary order: the host request is taken only when no
    // completion is pending, so an answered INS completes first at every
    // budget, a one-step budget included.
    for model in NMI_MODELS {
        for budget in [1, 2] {
            let (vm, mut cpu) = nmi_machine(
                model,
                false,
                b"\x6c\x90\xf4",
                b"\xf4",
                ExceptionDelivery::InGuest,
            );
            cpu.cpu.gpr[7] = 0x3000;
            cpu.cpu.gpr[2] = 0x10;
            let (_, s) = stop(&mut cpu, &vm, 1);
            assert!(matches!(s, StepStop::Exit(Exit::PortIo { .. })), "{s:?}");
            cpu.complete_port_in(0x5a);
            cpu.exit_request().request_exit();
            let (n, s) = stop(&mut cpu, &vm, budget);
            assert_eq!(
                vm.mem.read(0x3000, 1).unwrap(),
                0x5a,
                "{model:?} budget {budget}: {n} {s:?}"
            );
            assert_eq!(cpu.cpu.gpr[7] & 0xffff, 0x3001, "{model:?} budget {budget}");
        }
    }
}

#[test]
fn nmi_into_a_32_bit_code_segment_rederives_the_context() {
    // 80386 PRM Table 9-1 p. 9-2: NMI through IDT entry 2, here a 386
    // interrupt gate (type 8Eh; I386 PRM §§9.5–9.7 pp. 9-6–10) to selector
    // 18h, a code segment with D = 1. I386 PRM §17.1.1 p. 17-1: CS.D selects
    // the default operand size, so the handler's MOV EAX, imm32 loads all
    // four bytes (spec.md §17.10: the context is derived again after any
    // delivery).
    let (vm, mut cpu) = nmi_machine(
        CpuModel::I80386,
        true,
        b"\x90\x90",
        b"\xb8\x78\x56\x34\x12\xf4",
        ExceptionDelivery::InGuest,
    );
    vm.write_bytes(0x2018, &[0xff, 0xff, 0, 0, 0, 0x9b, 0x40, 0])
        .unwrap();
    cpu.cpu.legacy.gdtr.limit = 31;
    let low = (0x18u32 << 16) | (HANDLER as u32 & 0xffff);
    let high = (HANDLER as u32 & 0xffff_0000) | (0x8eu32 << 8);
    vm.write_bytes(0x3010, &low.to_le_bytes()).unwrap();
    vm.write_bytes(0x3014, &high.to_le_bytes()).unwrap();
    cpu.raise_nmi().unwrap();
    let (n, s) = stop(&mut cpu, &vm, 8);
    assert!(matches!(s, StepStop::Exit(Exit::Hlt)), "{n} {s:?}");
    assert_eq!(cpu.cpu.cs, 0x18);
    assert_eq!(cpu.cpu.rip, HANDLER + 6);
    assert_eq!(cpu.cpu.gpr[0] & 0xffff_ffff, 0x1234_5678);
}
