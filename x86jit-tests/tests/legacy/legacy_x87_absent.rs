//! Absent-socket output policy and obsolete x87 controls.
#[path = "../support/descriptor_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use super::legacy_i386_x87::{mem, setup_model};
use x86jit_core::{CpuModel, Exit, FpuModel, SegmentId, StepOptions, StepStop, Vcpu, Vm};

const DATA: u64 = 0x20400;
const MODELS: [CpuModel; 4] = [
    CpuModel::I80286,
    CpuModel::I80386,
    CpuModel::I80486Dx,
    CpuModel::I80486DxCpuid,
];

// Real, protected 16/32, V86, paged protected 32, paged V86.
fn setup(model: CpuModel, mode: u8, code: &[u8]) -> (Vm, Vcpu) {
    let wide = matches!(mode, 2 | 4);
    let real = matches!(mode, 0 | 3 | 5);
    let (vm, mut cpu) = setup_model(model, wide, real, code);
    if matches!(mode, 3 | 5) {
        cpu.cpu.legacy.cr0 |= 1;
        cpu.cpu.rflags_system |= 1 << 17;
        cpu.cpu.legacy.cpl = 3;
        cpu.cpu.es = 0x2000;
        cpu.cpu.fs = 0x2000;
        cpu.cpu.gs = 0x2000;
    }
    if mode >= 4 {
        vm.write_bytes(0x30000, &0x31007u32.to_le_bytes()).unwrap();
        for page in 0..1024u32 {
            vm.write_bytes(
                0x31000 + u64::from(page) * 4,
                &((page * 0x1000) | 7).to_le_bytes(),
            )
            .unwrap();
        }
        cpu.cpu.legacy.cr3 = 0x30000;
        cpu.cpu.legacy.cr0 |= 1 << 31;
    }
    (vm, cpu)
}

fn modes(model: CpuModel) -> std::ops::Range<u8> {
    0..if model == CpuModel::I80286 { 2 } else { 6 }
}

fn image(cpu: &Vcpu) -> String {
    format!(
        "{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}",
        cpu.cpu.fpr,
        cpu.cpu.fpu_top,
        cpu.cpu.fpu_empty,
        cpu.cpu.fpu_cw,
        cpu.cpu.fpu_sw,
        cpu.cpu.fpu_fip,
        cpu.cpu.fpu_fcs,
        cpu.cpu.fpu_fop,
        cpu.cpu.fpu_fdp,
        cpu.cpu.fpu_fds,
        cpu.cpu.external_numeric_error
    )
}

// ASSUMED, maintainer policy 2026-11-24: every absent-socket output word
// reads FFFFh. 286 PRM Part II Ch. 3 p. 3-2 explicitly calls the floating
// bus undefined; 386 PRM §11.1.2 pp. 11-1–2 defines no output value.
#[test]
fn absent_detection_probe_writes_ones_without_changing_the_fpu() {
    absent_no_wait_controls_retire_but_8086_present_controls_stay_refused();
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for mode in modes(model) {
            let wide = matches!(mode, 2 | 4);
            let mut code = vec![0xdb, 0xe3]; // FNINIT
            code.extend(mem(wide, 0xd9, 7, false, false)); // FNSTCW
            code.extend([0xdf, 0xe0]); // FNSTSW AX
            if wide {
                code.push(0x66);
            }
            code.extend([0x3d, 0xff, 0xff]); // CMP AX,FFFFh
            let (vm, mut cpu) = setup(model, mode, &code);
            cpu.set_fpu_model(FpuModel::None);
            cpu.cpu.gpr[0] = 0x1234_5678;
            cpu.cpu.fpu_cw = 0x1234;
            cpu.cpu.fpu_sw = 0x8084;
            let before = image(&cpu);
            fixture::steps(&mut cpu, &vm, 4);
            assert!(cpu.cpu.flags.zf);
            assert_eq!(cpu.cpu.gpr[0], 0x1234_ffff, "{model:?}/{mode}");
            let mut word = [0; 2];
            vm.read_bytes(DATA, &mut word).unwrap();
            assert_eq!(word, [0xff; 2], "{model:?}/{mode}");
            assert_eq!(image(&cpu), before);
        }
    }
}

#[test]
fn absent_control_stores_cover_every_byte_and_obey_em() {
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for mode in modes(model) {
            let wide = matches!(mode, 2 | 4);
            for p66 in [false, true] {
                if model == CpuModel::I80286 && p66 {
                    continue;
                }
                for (op, group, bytes) in [
                    (0xd9, 7, 2),
                    (0xdd, 7, 2),
                    (0xd9, 6, if wide != p66 { 28 } else { 14 }),
                    (0xdd, 6, if wide != p66 { 108 } else { 94 }),
                ] {
                    let code = mem(wide, op, group, p66, false);
                    for em in [false, true] {
                        let (vm, mut cpu) = setup(model, mode, &code);
                        cpu.set_fpu_model(FpuModel::None);
                        cpu.cpu.legacy.cr0 |= u32::from(em) << 2;
                        cpu.cpu.fpu_sw = 0x8084;
                        let before = image(&cpu);
                        vm.write_bytes(DATA - 1, &vec![0x5a; bytes + 2]).unwrap();
                        if em {
                            fixture::fault(&mut cpu, &vm, 7, None);
                        } else {
                            fixture::steps(&mut cpu, &vm, 1);
                        }
                        let mut data = vec![0; bytes + 2];
                        vm.read_bytes(DATA - 1, &mut data).unwrap();
                        assert_eq!(data[0], 0x5a);
                        assert_eq!(data[bytes + 1], 0x5a);
                        assert!(
                            data[1..=bytes]
                                .iter()
                                .all(|b| *b == if em { 0x5a } else { 0xff }),
                            "{model:?}/{mode}/{code:02x?}: {data:02x?}"
                        );
                        assert_eq!(image(&cpu), before);
                    }
                }
            }
            let (vm, mut cpu) = setup(model, mode, &[0xdf, 0xe0]);
            cpu.set_fpu_model(FpuModel::None);
            cpu.cpu.legacy.cr0 |= 4;
            let ax = cpu.cpu.gpr[0];
            fixture::fault(&mut cpu, &vm, 7, None);
            assert_eq!(cpu.cpu.gpr[0], ax);
        }
    }
}

// 287 PRM Part II Ch. 2 p. 2-16 / App. B item 2 p. B-1;
// SDM Vol 1 §8.3.13 p. 8-24, Vol 3B §25.18.9 p. 25-15;
// Intel486 PRM §17.7 p. 17-8 / §25.3 item 2 p. 25-11.
#[test]
fn obsolete_controls_are_model_specific_noops_with_wait_twins() {
    for model in MODELS {
        for mode in modes(model) {
            for opcode in [0xe0, 0xe1, 0xe4] {
                for wait in [false, true] {
                    let mut code = Vec::new();
                    if wait {
                        code.push(0x9b);
                    }
                    code.extend([0xdb, opcode]);
                    let (vm, mut cpu) = setup(model, mode, &code);
                    if !wait {
                        cpu.cpu.fpu_cw = 0x037b;
                        cpu.cpu.fpu_sw = 0x8084;
                    }
                    let before = image(&cpu);
                    let flags = cpu.cpu.flags;
                    if model == CpuModel::I80286 && opcode == 0xe4 {
                        // 287 mode latch is not implemented: retain its refusal.
                        if wait && mode == 0 {
                            fixture::steps(&mut cpu, &vm, 1);
                        }
                        let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                        assert_eq!(r.steps(), 0);
                        assert!(
                            matches!(
                                r.stop(),
                                StepStop::Exit(
                                    Exit::UnknownInstruction { .. }
                                        | Exit::UnsupportedSystemContext { .. }
                                )
                            ),
                            "{:?}",
                            r.stop()
                        );
                    } else if wait && model == CpuModel::I80286 && mode == 1 {
                        // Present protected 287 WAIT remains outside the scope.
                        let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                        assert_eq!(r.steps(), 0);
                        assert!(matches!(
                            r.stop(),
                            StepStop::Exit(Exit::UnsupportedSystemContext { .. })
                        ));
                    } else {
                        fixture::steps(&mut cpu, &vm, 1 + u64::from(wait));
                    }
                    assert_eq!(image(&cpu), before);
                    assert_eq!(cpu.cpu.flags, flags);
                    if !(model == CpuModel::I80286 && (opcode == 0xe4 || wait && mode == 1)) {
                        cpu.cpu.rip = 0x1000;
                        cpu.cpu.legacy.cr0 |= 4;
                        if wait {
                            fixture::steps(&mut cpu, &vm, 1);
                        }
                        fixture::fault(&mut cpu, &vm, 7, None);
                        assert_eq!(image(&cpu), before);
                    }
                }
            }
        }
    }
}

fn absent_no_wait_controls_retire_but_8086_present_controls_stay_refused() {
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for mode in modes(model) {
            for opcode in [0xe0, 0xe1, 0xe2, 0xe3, 0xe4] {
                let (vm, mut cpu) = setup(model, mode, &[0xdb, opcode]);
                cpu.set_fpu_model(FpuModel::None);
                cpu.cpu.fpu_sw = 0x8084;
                let before = image(&cpu);
                fixture::steps(&mut cpu, &vm, 1);
                assert_eq!(image(&cpu), before);
                cpu.cpu.rip = 0x1000;
                cpu.cpu.legacy.cr0 |= 4;
                fixture::fault(&mut cpu, &vm, 7, None);
            }
        }
    }
    for model in [CpuModel::I8086, CpuModel::I8088] {
        for opcode in [0xe0, 0xe1, 0xe4] {
            let (vm, mut cpu) = setup_model(model, false, true, &[0xdb, opcode]);
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(r.steps(), 0);
            assert!(matches!(
                r.stop(),
                StepStop::Exit(Exit::UnknownInstruction { .. })
            ));
        }
    }
}

#[test]
fn absent_image_stores_preflight_segments_pages_and_mapping() {
    absent_image_page_fault_leaves_the_first_page_untouched();
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for mode in modes(model) {
            let wide = matches!(mode, 2 | 4);
            for (op, group, bytes) in [
                (0xd9, 7, 2),
                (0xdd, 7, 2),
                (0xd9, 6, if wide { 28 } else { 14 }),
                (0xdd, 6, if wide { 108 } else { 94 }),
            ] {
                let code = mem(wide, op, group, false, false);
                for fault in 0..3 {
                    let mut code = code.clone();
                    if fault == 0 && matches!(mode, 3 | 5) {
                        let end = code.len();
                        code[end - 2..].copy_from_slice(&0xffffu16.to_le_bytes());
                    }
                    let (mut vm, mut cpu) = setup(model, mode, &code);
                    cpu.set_fpu_model(FpuModel::None);
                    vm.write_bytes(DATA, &vec![0x5a; bytes]).unwrap();
                    let before = image(&cpu);
                    match fault {
                        0 if !matches!(mode, 3 | 5) => {
                            cpu.cpu.legacy.segments[SegmentId::Ds as usize].limit = 0x400
                        }
                        0 => {}
                        1 if mode == 1 || mode == 2 || mode == 4 => {
                            cpu.cpu.legacy.segments[SegmentId::Ds as usize].attributes &= !2;
                        }
                        1 => continue,
                        _ => {
                            vm.unmap(0, 0x240000).unwrap();
                            vm.map(
                                0,
                                DATA as usize + bytes - 1,
                                x86jit_core::Prot::RWX,
                                x86jit_core::RegionKind::Ram,
                            )
                            .unwrap();
                            vm.write_bytes(DATA, &vec![0x5a; bytes - 1]).unwrap();
                        }
                    }
                    let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                    assert_eq!(r.steps(), 0, "{model:?}/{mode}/{fault}/{code:02x?}");
                    assert!(
                        matches!(
                            r.stop(),
                            StepStop::Exit(Exit::Exception { .. } | Exit::UnmappedMemory { .. })
                        ),
                        "{:?}",
                        r.stop()
                    );
                    let mut first = [0];
                    vm.read_bytes(DATA, &mut first).unwrap();
                    assert_eq!(first, [0x5a]);
                    assert_eq!(image(&cpu), before);
                }
            }
        }
    }
}

fn absent_image_page_fault_leaves_the_first_page_untouched() {
    for mode in [4, 5] {
        let wide = mode == 4;
        for (op, group, bytes) in [
            (0xd9, 7, 2),
            (0xdd, 7, 2),
            (0xd9, 6, if wide { 28 } else { 14 }),
            (0xdd, 6, if wide { 108 } else { 94 }),
        ] {
            let mut code = mem(wide, op, group, false, false);
            let displacement = if wide { 4 } else { 2 };
            let start = code.len() - displacement;
            code[start..].copy_from_slice(&0xfffu32.to_le_bytes()[..displacement]);
            let (vm, mut cpu) = setup(CpuModel::I80386, mode, &code);
            cpu.set_fpu_model(FpuModel::None);
            vm.write_bytes(0x20fff, &vec![0x5a; bytes]).unwrap();
            vm.write_bytes(0x31000 + 0x21 * 4, &0u32.to_le_bytes())
                .unwrap();
            let before = image(&cpu);
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(r.steps(), 0);
            let StepStop::Exit(exit @ Exit::Exception { vector: 14, .. }) = r.stop() else {
                panic!("{:?}", r.stop());
            };
            assert_eq!(
                exit.fault_metadata().unwrap().error_code(),
                x86jit_core::fault::ErrorCode::bits(if mode == 5 { 6 } else { 2 }, 7)
            );
            assert_eq!(cpu.cpu.legacy.cr2, 0x21000);
            let mut data = vec![0; bytes];
            vm.read_bytes(0x20fff, &mut data).unwrap();
            assert_eq!(data, vec![0x5a; bytes]);
            assert_eq!(image(&cpu), before);
            vm.write_bytes(0x31000 + 0x21 * 4, &0x21007u32.to_le_bytes())
                .unwrap();
            fixture::steps(&mut cpu, &vm, 1);
            vm.read_bytes(0x20fff, &mut data).unwrap();
            assert_eq!(data, vec![0xff; bytes]);
            assert_eq!(image(&cpu), before);
        }
    }
}
