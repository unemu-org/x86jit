//! IDT delivery witnesses: I286 PRM §§9.4–9.6 pp. 9-3–10;
//! I386 PRM §§9.5–9.8 pp. 9-6–13; §17 INT pp. 17-70–72 (scan checked).
use x86jit_core::{
    CpuGeneration, ExceptionDelivery, Exit, LegacyCpuModel, Prot, RegionKind, StepOptions,
    StepStop, Vcpu, Vm, VmConfig,
};

fn setup(generation: CpuGeneration, code: &[u8]) -> (Vm, Vcpu) {
    setup_delivery(generation, code, ExceptionDelivery::Outward)
}
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
fn i286_rejects_i386_gates_and_task_gates_stay_typed() {
    use x86jit_core::UnsupportedSystemOperation;
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let (vm, mut cpu) = setup(generation, b"\xcd\x30");
        gate(&vm, 0x30, 0x85, 0x1800);
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(
            matches!(
                run.stop(),
                StepStop::Exit(Exit::UnsupportedSystemContext {
                    operation: UnsupportedSystemOperation::LegacyGateOrTaskTransfer,
                    ..
                })
            ),
            "{:?}",
            run.stop()
        );
        assert_eq!((run.steps(), cpu.cpu.gpr[4]), (0, 0x8000));
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
