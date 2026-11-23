//! I80386 special-register witnesses. Sources use printed manual page numbers.
//! Machine instructions run through Direct and Checked interpreter walks; the
//! backend sentinel also pins that these instructions never reach codegen.
use x86jit_core::{
    Backend, CachedBlock, CpuModel, ExceptionDelivery, Exit, IrBlock, MemConsistency, Prot,
    RegionKind, SegmentId, StepOptions, StepStop, UnsupportedSystemOperation, Vcpu, Vm, VmConfig,
};

struct NoBackend;
impl Backend for NoBackend {
    fn materialize(
        &self,
        _: &IrBlock,
        _: MemConsistency,
        _: Option<(u64, u64)>,
        _: u64,
    ) -> CachedBlock {
        panic!("Machine execution reached application codegen");
    }
}
fn setup(model: CpuModel, code: &[u8], checked: bool) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x40000), Box::new(NoBackend));
    vm.set_cpu_model(model);
    vm.set_exception_delivery(ExceptionDelivery::Outward);
    vm.map(0, 0x40000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[4] = 0x8000;
    if checked {
        // Engine fixture: withhold Direct's ES proof without changing accesses.
        cpu.cpu.legacy.segments[SegmentId::Es as usize].attributes = 0x92;
    }
    (vm, cpu)
}
fn instruction(op: u8, register: u8, gpr: u8, mod_bits: u8, prefix: bool) -> Vec<u8> {
    let mut bytes = Vec::new();
    if prefix {
        bytes.push(0x66);
    }
    bytes.extend([0x0f, op, (mod_bits << 6) | (register << 3) | gpr]);
    bytes
}
fn steps(cpu: &mut Vcpu, vm: &Vm, count: u64) {
    let run = cpu.run_steps(vm, count, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), count, "{:?}", run.stop());
    assert!(matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)));
}
fn refused(cpu: &mut Vcpu, vm: &Vm, expected: UnsupportedSystemOperation) {
    let before = format!("{:?}", cpu.cpu);
    let retired = cpu.retired_instructions();
    let run = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0);
    assert!(
        matches!(run.stop(), StepStop::Exit(Exit::UnsupportedSystemContext { operation, .. }) if *operation == expected),
        "{:?}",
        run.stop()
    );
    assert_eq!(cpu.retired_instructions(), retired);
    assert_eq!(format!("{:?}", cpu.cpu), before);
}
fn fault(cpu: &mut Vcpu, vm: &Vm, vector: u8, code: Option<u32>) {
    let before = format!("{:?}", cpu.cpu);
    let run = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0);
    let StepStop::Exit(exit @ Exit::Exception { vector: actual, .. }) = run.stop() else {
        panic!("{:?}", run.stop());
    };
    assert_eq!(*actual, vector);
    assert_eq!(
        exit.fault_metadata().unwrap().error_code(),
        code.map_or(x86jit_core::fault::ErrorCode::NoCode, |c| {
            x86jit_core::fault::ErrorCode::bits(c, 0xffff)
        })
    );
    assert_eq!(format!("{:?}", cpu.cpu), before);
}

// 80386 PRM §17 MOV p. 17-111: real-mode 32-bit transfers, including 66h.
// SDM Vol 2B §4.3 MOV DR p. 4-35: ModR/M.mod is ignored and DR4/5
// alias DR6/7 for Intel386 compatibility. The PRM lists only mod=11 for
// special-register MOV (p. 17-111). ASSUMED: TR6/7 ignore mod as DR/CR do.
// All GPR encodings are exercised, including those otherwise requiring an EA.
fn round_trip(register: u8, test: bool, values: &[(u32, u32)]) {
    for checked in [false, true] {
        for prefix in [false, true] {
            for mod_bits in 0..4 {
                for gpr in 0..8 {
                    for &(written, expected) in values {
                        let mut code = instruction(
                            if test { 0x26 } else { 0x23 },
                            register,
                            gpr,
                            mod_bits,
                            prefix,
                        );
                        let dst = (gpr + 1) % 8;
                        code.extend(instruction(
                            if test { 0x24 } else { 0x21 },
                            register,
                            dst,
                            mod_bits,
                            prefix,
                        ));
                        let (vm, mut cpu) = setup(CpuModel::I80386, &code, checked);
                        cpu.cpu.gpr[gpr as usize] = 0xfeed_beef_0000_0000 | u64::from(written);
                        cpu.cpu.gpr[dst as usize] = u64::MAX;
                        steps(&mut cpu, &vm, 2);
                        assert_eq!(
                            cpu.cpu.gpr[dst as usize],
                            u64::from(expected),
                            "{code:02x?}"
                        );
                        assert_eq!(cpu.cpu.rip, 0x1000 + code.len() as u64);
                    }
                }
            }
        }
    }
}
macro_rules! address_register {
    ($name:ident, $register:expr) => {
        #[test]
        fn $name() {
            round_trip(
                $register,
                false,
                &[(0, 0), (0x89ab_cdef, 0x89ab_cdef), (u32::MAX, u32::MAX)],
            );
        }
    };
}
address_register!(dr0_round_trip, 0);
address_register!(dr1_round_trip, 1);
address_register!(dr2_round_trip, 2);
address_register!(dr3_round_trip, 3);

#[test]
fn dr6_status_and_reserved_image() {
    // PRM §12.2 Fig. 12-1 pp. 12-3–5: B0–3/BD/BS/BT fields.
    // ASSUMED: reserved bits freeze to the SDM Vol 3A §12.1 Table 12-1
    // p. 12-3 image. The PRM does not prescribe reserved-bit readback.
    round_trip(
        6,
        false,
        &[
            (0, 0xffff_0ff0),
            (u32::MAX, 0xffff_efff),
            (0xe005, 0xffff_eff5),
        ],
    );
}
#[test]
fn dr7_disabled_breakpoints_and_reserved_image() {
    // PRM §12.2.2 pp. 12-3–4: retain exactness, R/W and LEN fields.
    // ASSUMED: reserved bits freeze to the SDM Table 12-1 p. 12-3 image.
    round_trip(
        7,
        false,
        &[
            (0, 0x400),
            (0xffff_dc00, 0xffff_0400),
            (0xffff_0300, 0xffff_0700),
        ],
    );
}
#[test]
fn dr4_dr5_alias_status_and_control() {
    // SDM Vol 2B §4.3 MOV DR p. 4-35 explicitly specifies these aliases
    // as Intel386-compatible. Write through each name and read its partner.
    for checked in [false, true] {
        for prefix in [false, true] {
            for (write, read, written, expected) in [
                (4, 6, 0xe005, 0xffff_eff5),
                (6, 4, 0, 0xffff_0ff0),
                (5, 7, 0x300, 0x700),
                (7, 5, 0, 0x400),
            ] {
                let mut code = instruction(0x23, write, 0, 3, prefix);
                code.extend(instruction(0x21, read, 3, 3, prefix));
                let (vm, mut cpu) = setup(CpuModel::I80386, &code, checked);
                cpu.cpu.gpr[0] = written;
                steps(&mut cpu, &vm, 2);
                assert_eq!(cpu.cpu.gpr[3], expected);
            }
        }
    }
}
#[test]
fn tr7_data_round_trip() {
    // 80386 PRM §10.6.2 pp. 10-15–16: TR7 data register. ASSUMED:
    // retain all written bits; no TLB command or lookup has completed.
    round_trip(
        7,
        true,
        &[(0, 0), (0x89ab_cdef, 0x89ab_cdef), (u32::MAX, u32::MAX)],
    );
}
#[test]
fn tr6_initial_read() {
    // 80386 PRM §17 MOV p. 17-111: read TR6 as a 32-bit register.
    // ASSUMED: the engine's initial TR6 image is zero.
    for checked in [false, true] {
        for prefix in [false, true] {
            for mod_bits in 0..4 {
                for gpr in 0..8 {
                    let code = instruction(0x24, 6, gpr, mod_bits, prefix);
                    let (vm, mut cpu) = setup(CpuModel::I80386, &code, checked);
                    cpu.cpu.gpr[gpr as usize] = u64::MAX;
                    steps(&mut cpu, &vm, 1);
                    assert_eq!(cpu.cpu.gpr[gpr as usize], 0);
                }
            }
        }
    }
}
#[test]
fn tr6_both_commands_are_atomic_refusals() {
    // PRM §10.6.2 p. 10-14: C=0 writes, C=1 looks up; neither is a
    // passive write, even though MOV itself is defined in real mode.
    // ASSUMED: refuse both commands atomically with PG=0.
    for checked in [false, true] {
        for prefix in [false, true] {
            for mod_bits in 0..4 {
                for gpr in 0..8 {
                    for command in [0, 1, 0x1234_5ffe, 0x1234_5fff] {
                        let code = instruction(0x26, 6, gpr, mod_bits, prefix);
                        let (vm, mut cpu) = setup(CpuModel::I80386, &code, checked);
                        cpu.cpu.gpr[gpr as usize] = command;
                        refused(&mut cpu, &vm, UnsupportedSystemOperation::LegacyTlbTest);
                    }
                }
            }
        }
    }
}
#[test]
fn dr7_enabling_each_breakpoint_or_general_detect_is_atomic_refusal() {
    // PRM §12.2.2 pp. 12-3–4: low eight enable bits. SDM Vol 3B
    // §20.2.4 p. 20-5: GD. ASSUMED: refuse the write without retiring.
    for checked in [false, true] {
        for prefix in [false, true] {
            for register in [5, 7] {
                for bit in [0, 1, 2, 3, 4, 5, 6, 7, 13] {
                    let code = instruction(0x23, register, 0, 3, prefix);
                    let (vm, mut cpu) = setup(CpuModel::I80386, &code, checked);
                    cpu.cpu.gpr[0] = 1 << bit;
                    refused(
                        &mut cpu,
                        &vm,
                        UnsupportedSystemOperation::LegacyDebugBreakpoints,
                    );
                    // A retry with disabled breakpoints completes exactly once.
                    cpu.cpu.gpr[0] = 0x300;
                    steps(&mut cpu, &vm, 1);
                    assert_eq!(cpu.retired_instructions(), 1);
                }
            }
        }
    }
}
#[test]
fn single_step_handler_reads_sticky_dr6_bs_and_can_clear_it() {
    // 80386 PRM §12.2.3 pp. 12-4–5: BS marks single-step; all DR6
    // status bits remain set until software clears them. No address matches
    // the disabled DR0–DR3 conditions in this fixture.
    for checked in [false, true] {
        for initial in [0xffff_0ff0, 0xffff_aff5, 0xffff_efff] {
            let (vm, mut cpu) = setup(CpuModel::I80386, b"\x90\x90", checked);
            vm.write_bytes(4, &[0, 0x18, 0, 0]).unwrap();
            // Read EAX; NOP; read EBX; zero ECX; write DR6; read EDX; IRET.
            vm.write_bytes(
                0x1800,
                b"\x0f\x21\xf0\x90\x0f\x21\xf3\x66\xb9\x00\x00\x00\x00\x0f\x23\xf1\x0f\x21\xf2\xcf",
            )
            .unwrap();
            cpu.cpu.debug_test_registers[4] = initial;
            cpu.cpu.rflags_system = 0x100;
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.rip, 0x1800);
            assert_eq!(cpu.cpu.debug_test_registers[4], initial | 0x4000);
            steps(&mut cpu, &vm, 3);
            assert_eq!(cpu.cpu.gpr[0], u64::from(initial | 0x4000));
            assert_eq!(cpu.cpu.gpr[3], cpu.cpu.gpr[0]);
            steps(&mut cpu, &vm, 3);
            assert_eq!(cpu.cpu.gpr[2], 0xffff_0ff0);
            assert_eq!(cpu.cpu.debug_test_registers[4], 0xffff_0ff0);
            steps(&mut cpu, &vm, 2); // IRET restores TF; the next NOP traps.
            assert_eq!(cpu.cpu.rip, 0x1800);
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.gpr[0], 0xffff_4ff0);
        }
    }
}

#[test]
fn single_step_on_i286_leaves_debug_register_images_inert() {
    for checked in [false, true] {
        let (vm, mut cpu) = setup(CpuModel::I80286, b"\x90", checked);
        vm.write_bytes(4, &[0, 0x18, 0, 0]).unwrap();
        cpu.cpu.rflags_system = 0x100;
        cpu.cpu.debug_test_registers = [u32::MAX; 8];
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.rip, 0x1800);
        assert_eq!(cpu.cpu.debug_test_registers, [u32::MAX; 8]);
    }
}

#[test]
fn restored_dr7_enables_are_refused_before_cold_or_warm_execution() {
    // PRM §12.2.2 p. 12-4; SDM Vol 3B §20.2.4 p. 20-5.
    // Engine contract: restoring an image has the same typed refusal as
    // MOV DR7, including zero-budget probes, with no CpuState changes.
    for checked in [false, true] {
        for warm in [false, true] {
            for stepped in [false, true] {
                for budget in [0, 1] {
                    for bit in [0, 1, 2, 3, 4, 5, 6, 7, 13] {
                        let (vm, mut cpu) = setup(CpuModel::I80386, b"\x90\x90", checked);
                        if warm {
                            steps(&mut cpu, &vm, 1);
                        }
                        let mut restored = cpu.cpu.clone();
                        restored.debug_test_registers[5] = 0x400 | (1 << bit);
                        cpu.cpu = restored;
                        let before = format!("{:?}", cpu.cpu);
                        let retired = cpu.retired_instructions();
                        let exit = if stepped {
                            let run = cpu.run_steps(&vm, budget, StepOptions::default()).unwrap();
                            assert_eq!(run.steps(), 0);
                            let StepStop::Exit(exit) = run.stop() else {
                                panic!("{:?}", run.stop());
                            };
                            exit.clone()
                        } else {
                            cpu.run(&vm, Some(budget))
                        };
                        assert!(matches!(
                            exit,
                            Exit::UnsupportedSystemContext {
                                operation: UnsupportedSystemOperation::LegacyDebugBreakpoints,
                                ..
                            }
                        ));
                        assert_eq!(cpu.retired_instructions(), retired);
                        assert_eq!(format!("{:?}", cpu.cpu), before);
                        cpu.cpu.debug_test_registers[5] = 0x400;
                        steps(&mut cpu, &vm, 1);
                        assert_eq!(cpu.retired_instructions(), retired + 1);
                    }
                }
            }
        }
    }
}

#[test]
fn special_register_moves_remain_undefined_on_i286() {
    // DERIVED from 80286 PRM App. B MOV p. B-73 (byte/word forms only)
    // and #UD p. B-12: these absent two-byte opcodes raise #UD.
    for checked in [false, true] {
        for prefix in [false, true] {
            for op in [0x21, 0x23, 0x24, 0x26] {
                for register in 0..8 {
                    let code = instruction(op, register, 0, 3, prefix);
                    let (vm, mut cpu) = setup(CpuModel::I80286, &code, checked);
                    fault(&mut cpu, &vm, 6, None);
                }
            }
        }
    }
}
#[test]
fn protected_special_register_moves_have_the_mov_cr_cpl_rule() {
    // 80386 PRM §17 MOV p. 17-111 and §10.6.2 p. 10-13: #GP(0)
    // outside CPL0. The existing MOV CR privilege guard also guards DR/TR.
    for prefix in [false, true] {
        for op in [0x20, 0x22, 0x21, 0x23, 0x24, 0x26] {
            let registers: &[u8] = match op {
                0x20 | 0x22 => &[2],
                0x24 | 0x26 => &[6, 7],
                _ => &[0, 1, 2, 3, 4, 5, 6, 7],
            };
            for &register in registers {
                for ring in 0..4 {
                    let code = instruction(op, register, 0, 3, prefix);
                    let (vm, mut cpu) = setup(CpuModel::I80386, &code, true);
                    cpu.cpu.legacy.cr0 |= 1;
                    cpu.cpu.legacy.cpl = ring;
                    cpu.cpu.cs = u16::from(ring);
                    cpu.cpu.ss = u16::from(ring);
                    for segment in &mut cpu.cpu.legacy.segments {
                        segment.attributes |= u16::from(ring) << 5;
                    }
                    if ring != 0 {
                        // Also verify privilege precedes unsupported commands.
                        cpu.cpu.gpr[0] = 0xff;
                        fault(&mut cpu, &vm, 13, Some(0));
                    } else if op == 0x26 && register == 6 {
                        refused(&mut cpu, &vm, UnsupportedSystemOperation::LegacyTlbTest);
                    } else {
                        steps(&mut cpu, &vm, 1);
                    }
                }
            }
        }
    }
}
#[test]
fn real_iret_ignores_current_nt_and_loads_stacked_nt() {
    real_iret_nt(false);
}
#[test]
fn real_iretd_ignores_current_nt_and_loads_stacked_nt() {
    real_iret_nt(true);
}
fn real_iret_nt(wide: bool) {
    // 80386 PRM §17 IRET p. 17-74: PE=0 pops IP/EIP, CS, FLAGS/EFLAGS
    // before the protected-mode NT test. §2.3.4 Fig. 2-8 p. 2-11: NT bit14.
    for checked in [false, true] {
        for stacked_nt in [0, 0x4000u32] {
            let code: &[u8] = if wide { b"\x66\xcf" } else { b"\xcf" };
            let (vm, mut cpu) = setup(CpuModel::I80386, code, checked);
            cpu.cpu.rflags_system = 0x4000;
            let size = if wide { 4 } else { 2 };
            for (slot, value) in [0x2200u32, 0x100, 0x202 | stacked_nt]
                .into_iter()
                .enumerate()
            {
                vm.write_bytes(0x8000 + (slot * size) as u64, &value.to_le_bytes()[..size])
                    .unwrap();
            }
            steps(&mut cpu, &vm, 1);
            assert_eq!(
                (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
                (0x100, 0x2200, 0x8000 + (3 * size) as u64)
            );
            assert_eq!(cpu.cpu.rflags() & 0xffff, u64::from(0x202 | stacked_nt));
            assert_eq!(cpu.cpu.legacy.cr0 & 1, 0);
        }
    }
}
#[test]
fn i286_real_nt_boundary_is_unchanged() {
    // Scope witness: retain the existing 286 engine refusal. Its PRM App. B
    // IRET p. B-51 also describes a real-mode stack return, but changing that
    // model is outside this change. This is an engine boundary, not hardware.
    let (vm, mut cpu) = setup(CpuModel::I80286, b"\xcf", false);
    cpu.cpu.rflags_system = 0x4000;
    refused(&mut cpu, &vm, UnsupportedSystemOperation::LegacyTaskReturn);
}
