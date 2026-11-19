//! Real-mode comparison-fault compatibility witnesses (decision-23 §6, M4).
//! SDM Vol 2B §4.3 REPE/REPZ p. 4-565 and REPNE/REPNZ p. 4-567 require
//! original EFLAGS on a comparison fault. These expectations pin current legacy
//! behavior for a later real-mode change; the alias personality is frozen.
use x86jit_core::{
    CpuMode, CpuModel, CpuState, ExceptionDelivery, Exit, Prot, RegionKind, StepOptions, StepStop,
    Vm, VmConfig,
};

#[test]
fn repeated_comparison_fault_flags_match_frozen_real_behavior() {
    // SDM Vol 2A §3.3 CMP/CMPS and Vol 2B §4.3 SCAS: equal operands
    // produce PF|ZF; 1111h - 1010h produces no arithmetic status bits.
    // CMPS/SCAS real-address exceptions gate the full operand at FFFFh.
    // Decision-16 §3 supplies the resumable engine-map-fault contract.
    const STATUS: u16 = 0x8d5;
    let mut cases = 0;
    for (model, path) in [
        (Some(CpuModel::I80286), "Machine I286"),
        (Some(CpuModel::I80386), "Machine I386"),
        (None, "alias"),
    ] {
        for (address32, limit, fault_kind) in [
            (false, false, "unmapped/addr16"),
            (true, true, "limit/addr32"),
            (true, false, "unmapped/addr32"),
        ] {
            if model == Some(CpuModel::I80286) && address32 {
                continue;
            }
            for opcode in [0xa7, 0xaf] {
                for repe in [false, true] {
                    for budget in [1, 2, 8] {
                        let mut vm = Vm::new(VmConfig::flat(0x20000));
                        if let Some(model) = model {
                            vm.set_cpu_model(model);
                        } else {
                            vm.set_cpu_mode(CpuMode::Real16);
                        }
                        vm.set_exception_delivery(ExceptionDelivery::Outward);
                        if limit {
                            vm.map(0, 0x20000, Prot::RWX, RegionKind::Ram).unwrap();
                        } else {
                            vm.map(0, 0x9000, Prot::RWX, RegionKind::Ram).unwrap();
                            vm.map(0xa000, 0x16000, Prot::RWX, RegionKind::Ram).unwrap();
                        }
                        let mut code = Vec::new();
                        if address32 {
                            code.push(0x67);
                        }
                        code.extend([if repe { 0xf3 } else { 0xf2 }, opcode, 0xf4]);
                        vm.write_bytes(0x1000, &code).unwrap();
                        vm.write_bytes(0x34, b"\x00\x18\x00\x00").unwrap();
                        vm.write_bytes(0x1800, b"\xf4").unwrap();
                        let near = if limit { 0xfffc } else { 0x8ffc };
                        let mut cpu = vm.new_vcpu();
                        cpu.cpu = CpuState::legacy_real_entry([0; 6]);
                        cpu.cpu.rip = 0x1000;
                        cpu.cpu.gpr[4] = 0x8000;
                        cpu.cpu.gpr[0] = 0x1111;
                        cpu.cpu.gpr[1] = 3;
                        cpu.cpu.gpr[6] = if opcode == 0xa7 { near } else { 0x4000 };
                        cpu.cpu.gpr[7] = if opcode == 0xa7 { 0x5000 } else { near };
                        if opcode == 0xa7 {
                            vm.write_bytes(near, &[0x11; 6][..if limit { 6 } else { 4 }])
                                .unwrap();
                        }
                        vm.write_bytes(
                            cpu.cpu.gpr[7],
                            &[if repe { 0x11 } else { 0x10 }; 6]
                                [..if opcode == 0xaf && !limit { 4 } else { 6 }],
                        )
                        .unwrap();
                        cpu.cpu.flags.cf = true;
                        cpu.cpu.flags.of = true;
                        cpu.cpu.flags.zf = !repe;
                        cpu.cpu.flags.pf_src = 1;
                        cpu.cpu.flags.af_src = 0;
                        cpu.cpu.flags.sf = false;
                        cpu.cpu.flags.df = false;
                        cpu.cpu.flags.if_ = false;
                        cpu.cpu.rflags_system = 0;
                        let original_status = cpu.cpu.flags.to_flags16() & STATUS;
                        let mut steps = 0;
                        let mut stop = None;
                        for _ in 0..8 {
                            let run = cpu.run_steps(&vm, budget, StepOptions::default()).unwrap();
                            steps += run.steps();
                            stop = Some(match run.stop() {
                                StepStop::Exit(Exit::BudgetExhausted) => continue,
                                StepStop::Exit(Exit::UnmappedMemory { .. }) => "unmapped",
                                StepStop::Exit(Exit::Exception { vector: 13, .. }) => "gp",
                                StepStop::Exit(Exit::Hlt) if cpu.cpu.rip == 0x1801 => "delivered-gp",
                                StepStop::Exit(Exit::Hlt) => "completed",
                                other => panic!("{path}, bytes={code:02x?}, {fault_kind}, budget={budget}: {other:?}"),
                            });
                            break;
                        }
                        let stop = stop.expect("comparison did not stop in eight slices");
                        let complete = model.is_none() && limit;
                        let expected_stop = if complete {
                            "completed"
                        } else if limit {
                            "gp"
                        } else {
                            "unmapped"
                        };
                        assert_eq!(
                            stop, expected_stop,
                            "{path}, bytes={code:02x?}, {fault_kind}, budget={budget}"
                        );
                        let iterations = if complete { 3 } else { 2 };
                        assert_eq!(steps, if complete { 4 } else { 2 });
                        assert_eq!(cpu.retired_instructions(), if complete { 2 } else { 0 });
                        assert_eq!(cpu.cpu.gpr[1], 3 - iterations);
                        assert_eq!(
                            cpu.cpu.gpr[6],
                            if opcode == 0xa7 {
                                near + 2 * iterations
                            } else {
                                0x4000
                            }
                        );
                        assert_eq!(
                            cpu.cpu.gpr[7],
                            if opcode == 0xa7 {
                                0x5000 + 2 * iterations
                            } else {
                                near + 2 * iterations
                            }
                        );
                        assert_eq!(
                            cpu.cpu.rip,
                            if complete {
                                0x1000 + code.len() as u64
                            } else {
                                0x1000
                            }
                        );
                        let status = cpu.cpu.flags.to_flags16() & STATUS;
                        assert_eq!(status, if repe { 0x44 } else { 0 }, "{path}, bytes={code:02x?}, {fault_kind}, budget={budget}: status={status:#x}, original={original_status:#x}");
                        if !complete {
                            assert_ne!(status, original_status);
                        }
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 84);
}
