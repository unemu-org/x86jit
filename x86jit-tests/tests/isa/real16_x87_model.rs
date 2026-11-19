//! SDM Vol 3A §2.5, Table 2-2; 80386 PRM Ch. 11 §§11.1.2–11.1.3,
//! pp. 11-1–11-2: coprocessor presence is independent of MSW.EM.
use x86jit_core::{
    CpuMode, CpuState, Exit, FpuModel, Prot, Reg, RegionKind, StepOptions, StepStop, Vcpu, Vm,
    VmConfig,
};

fn make_vm(code: &[u8]) -> Vm {
    let mut vm = Vm::new(VmConfig::flat(0x40000));
    vm.set_cpu_mode(CpuMode::Real16);
    vm.map(0, 0x40000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, code).unwrap();
    // Distinguish #NM, #GP and the ordinary successor.
    for (vector, ip) in [(7, 0x700u16), (13, 0xd00u16)] {
        vm.write_bytes(vector * 4, &ip.to_le_bytes()).unwrap();
        vm.write_bytes(u64::from(ip), &[0xf4]).unwrap();
    }
    vm
}

fn make_cpu(vm: &Vm) -> Vcpu {
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Cs, 0x100);
    cpu.set_reg(Reg::Rip, 0);
    cpu.set_reg(Reg::Rsp, 0x800);
    cpu
}

fn image(cpu: &CpuState) -> ([[u8; 10]; 8], [u64; 9]) {
    (
        cpu.fpr,
        [
            u64::from(cpu.fpu_top),
            u64::from(cpu.fpu_cw),
            u64::from(cpu.fpu_sw),
            u64::from(cpu.fpu_empty),
            cpu.fpu_fip,
            cpu.fpu_fdp,
            u64::from(cpu.fpu_fop),
            u64::from(cpu.fpu_fcs),
            u64::from(cpu.fpu_fds),
        ],
    )
}

fn dirty(cpu: &mut Vcpu) {
    for (i, reg) in cpu.cpu.fpr.iter_mut().enumerate() {
        reg[..8].copy_from_slice(&0x8000_0000_0000_0000u64.to_le_bytes());
        reg[8..].copy_from_slice(&(0x3fffu16 + i as u16).to_le_bytes());
    }
    cpu.cpu.fpu_top = 3;
    cpu.cpu.fpu_cw = 0x0360;
    cpu.cpu.fpu_sw = 0x8081;
    cpu.cpu.fpu_empty = 0;
    cpu.cpu.fpu_fip = 0x2345;
    cpu.cpu.fpu_fdp = 0x4567;
    cpu.cpu.fpu_fcs = 0x1234;
    cpu.cpu.fpu_fds = 0x2345;
    cpu.cpu.fpu_fop = 0x523;
}

#[test]
fn selection_is_per_vcpu_and_cached_ir_is_shared() {
    let vm = make_vm(&[0xdb, 0xe3, 0xf4]); // FNINIT; HLT
    let mut present = make_cpu(&vm);
    let mut absent = make_cpu(&vm);
    assert_eq!(present.fpu_model(), FpuModel::Present);
    assert!(!present.fpu_emulation());
    dirty(&mut present);
    dirty(&mut absent);
    absent.set_fpu_model(FpuModel::None);
    let before = image(&absent.cpu);
    assert!(matches!(present.run(&vm, Some(10)), Exit::Hlt));
    assert_eq!(present.fpu_cw(), 0x037f); // SDM Vol 2A §3.3 FINIT/FNINIT.
    assert!(matches!(absent.run(&vm, Some(10)), Exit::Hlt));
    assert_eq!(image(&absent.cpu), before);
    assert_eq!(absent.cpu.clone().fpu_model, FpuModel::None);
    absent.set_fpu_model(FpuModel::Present);
    absent.set_reg(Reg::Rip, 0);
    assert!(matches!(absent.run(&vm, Some(10)), Exit::Hlt));
    assert_eq!(absent.fpu_cw(), 0x037f);
}

const ESC: [&[u8]; 8] = [
    &[0xd8, 0xc0],             // FADD ST(0),ST(0)
    &[0xd9, 0xe8],             // FLD1
    &[0xda, 0x06, 0x00, 0x20], // FIADD m32
    &[0xdb, 0xe3],             // FNINIT
    &[0xdc, 0xc0],             // FADD ST(0),ST(0)
    &[0xdd, 0xc0],             // FFREE ST(0)
    &[0xde, 0xc1],             // FADDP
    &[0xdf, 0x06, 0x00, 0x20], // FILD m16
];

#[test]
fn absent_esc_and_wait_preserve_the_entire_fpu_image() {
    let mut code = ESC.concat();
    code.extend_from_slice(&[0x9b, 0xf4]);
    let vm = make_vm(&code);
    let mut cpu = make_cpu(&vm);
    dirty(&mut cpu); // Stale ES must not produce #MF with no coprocessor.
    cpu.set_fpu_model(FpuModel::None);
    let before = image(&cpu.cpu);
    let flags = cpu.flags().to_rflags();
    let result = cpu.run_steps(&vm, 20, StepOptions::default()).unwrap();
    assert!(matches!(result.stop(), StepStop::Exit(Exit::Hlt)));
    assert_eq!(result.steps(), 10);
    assert_eq!(image(&cpu.cpu), before);
    assert_eq!(cpu.flags().to_rflags(), flags);
}

#[test]
fn em_vectors_all_esc_groups_without_retiring_or_modifying_the_fpu() {
    for esc in ESC {
        let vm = make_vm(esc);
        for model in [FpuModel::Present, FpuModel::None] {
            let mut cpu = make_cpu(&vm);
            dirty(&mut cpu);
            cpu.set_fpu_model(model);
            cpu.set_fpu_emulation(true);
            let before = image(&cpu.cpu);
            let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(result.stop(), StepStop::Exit(Exit::Hlt)),
                "{esc:02x?}"
            );
            assert_eq!(result.steps(), 1); // Only the handler's HLT retired.
            assert_eq!(cpu.reg(Reg::Rip), 0x701);
            assert_eq!(cpu.reg(Reg::Cs), 0);
            assert_eq!(vm.mem.read(0x7fa, 2).unwrap(), 0); // Faulting IP.
            assert_eq!(vm.mem.read(0x7fc, 2).unwrap(), 0x100);
            assert_eq!(image(&cpu.cpu), before);
        }
    }
}

#[test]
fn em_precedes_operand_limit_and_wait_ignores_em() {
    let vm = make_vm(&[0xd9, 0x3e, 0xff, 0xff]); // FNSTCW [ffff]
    let mut cpu = make_cpu(&vm);
    cpu.set_fpu_model(FpuModel::None);
    cpu.set_fpu_emulation(true);
    assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
    assert_eq!(cpu.reg(Reg::Rip), 0x701);
    let vm = make_vm(&[0x9b, 0xf4]);
    let mut cpu = make_cpu(&vm);
    dirty(&mut cpu);
    cpu.set_fpu_model(FpuModel::None);
    cpu.set_fpu_emulation(true);
    let before = image(&cpu.cpu);
    assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
    assert_eq!(cpu.reg(Reg::Rip), 2);
    assert_eq!(image(&cpu.cpu), before);
}

#[test]
fn em_also_gates_esc_forms_the_engine_does_not_lift() {
    // 80386 PRM Ch. 11 §11.1.3, p. 11-2: the CPU tests EM before
    // handing ESC to a coprocessor. The engine does not lift FNSAVE m108.
    for code in [
        &[0x66, 0xdd, 0x36, 0x00, 0x20][..],
        &[0xdf, 0x26, 0x00, 0x20],
    ] {
        let vm = make_vm(code);
        for model in [FpuModel::Present, FpuModel::None] {
            for path in 0..3 {
                let mut cpu = make_cpu(&vm);
                dirty(&mut cpu);
                cpu.set_fpu_model(model);
                cpu.set_fpu_emulation(true);
                let before = image(&cpu.cpu);
                match path {
                    0 => assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt)),
                    1 => {
                        let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                        assert!(matches!(result.stop(), StepStop::Exit(Exit::Hlt)));
                        assert_eq!(result.steps(), 1);
                    }
                    _ => {
                        assert!(matches!(
                            cpu.step_instruction(&vm),
                            x86jit_core::StepResult::Continue
                        ));
                        assert_eq!(cpu.retired_instructions(), 0);
                    }
                }
                assert_eq!(cpu.reg(Reg::Rip), if path == 2 { 0x700 } else { 0x701 });
                assert_eq!(image(&cpu.cpu), before);
            }
        }
    }
}

#[test]
fn empty_socket_outputs_exit_without_inventing_memory_or_ax_values() {
    for code in [
        &[0xd9, 0x3e, 0x00, 0x20][..],   // FNSTCW
        &[0xdd, 0x3e, 0x00, 0x20],       // FNSTSW memory
        &[0xdf, 0xe0],                   // FNSTSW AX
        &[0xd9, 0x36, 0x00, 0x20],       // FNSTENV
        &[0xdd, 0x36, 0x00, 0x20],       // FNSAVE
        &[0xdf, 0x36, 0x00, 0x20],       // FBSTP (Present lifter gap)
        &[0x66, 0xdd, 0x36, 0x00, 0x20], // FNSAVE m108 (Present lifter gap)
        &[0xd9, 0x3e, 0xff, 0xff],       // FNSTCW across the limit
        &[0xdd, 0x36, 0xff, 0xff],       // FNSAVE across the limit
        &[0xdf, 0x36, 0xff, 0xff],       // FBSTP across the limit
    ] {
        let vm = make_vm(code);
        let operand = if code.ends_with(&[0xff, 0xff]) {
            0xffff
        } else {
            0x2000
        };
        vm.write_bytes(operand, &[0xa5; 108]).unwrap();
        let mut cpu = make_cpu(&vm);
        dirty(&mut cpu);
        cpu.set_reg(Reg::Rax, 0x1234_5678);
        cpu.set_fpu_model(FpuModel::None);
        let before = image(&cpu.cpu);
        let exit = cpu.run(&vm, Some(10));
        assert!(
            matches!(exit, Exit::UnknownInstruction { addr: 0, bytes, len, .. }
            if &bytes[..usize::from(len)] == code),
            "{exit:?}"
        );
        assert_eq!(cpu.reg(Reg::Rax), 0x1234_5678);
        assert_eq!(cpu.reg(Reg::Rip), 0);
        assert_eq!(image(&cpu.cpu), before);
        let mut out = [0; 108];
        vm.read_bytes(operand, &mut out).unwrap();
        assert_eq!(out, [0xa5; 108]);
    }
}

#[test]
fn present_real_environment_uses_fourteen_bytes_and_segmented_addressing() {
    // SDM Vol 1 §8.1.10, Fig. 8-12: packed 20-bit FIP/FDP and FOP.
    let vm = make_vm(&[0xd9, 0x36, 0x00, 0x20, 0xf4]); // FNSTENV DS:2000
    let mut cpu = make_cpu(&vm);
    dirty(&mut cpu);
    cpu.cpu.fpu_sw = 0x0100;
    cpu.set_reg(Reg::Ds, 0x1000);
    vm.write_bytes(0x12000, &[0xa5; 16]).unwrap();
    assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
    let mut out = [0; 16];
    vm.read_bytes(0x12000, &mut out).unwrap();
    let words = [0x0360u16, 0x1900, 0, 0x4685, 0x1523, 0x79b7, 0x2000];
    let expected: Vec<u8> = words.into_iter().flat_map(u16::to_le_bytes).collect();
    assert_eq!(&out[..14], expected);
    assert_eq!(&out[14..], &[0xa5; 2]);
    assert_eq!(cpu.fpu_cw(), 0x037f);
}

#[test]
fn real_save_restore_preserves_logical_stack_and_real_pointers() {
    // SDM Vol 1 §8.1.10; Vol 2A §3.3 FSAVE/FNSAVE and FRSTOR, Operation.
    let vm = make_vm(&[0xdd, 0x36, 0x00, 0x20, 0xf4, 0xdd, 0x26, 0x00, 0x20, 0xf4]);
    let mut cpu = make_cpu(&vm);
    dirty(&mut cpu);
    cpu.cpu.fpu_sw = 0x0100;
    cpu.cpu.fpu_cw = 0x037f;
    let original = cpu.cpu.clone();
    vm.write_bytes(0x2000, &[0xa5; 96]).unwrap();
    assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
    assert_eq!(
        (
            cpu.cpu.fpu_cw,
            cpu.cpu.fpu_top,
            cpu.cpu.fpu_sw,
            cpu.cpu.fpu_empty
        ),
        (0x037f, 0, 0, 0xff)
    );
    assert_eq!(cpu.cpu.fpr, original.fpr);
    let mut out = [0; 96];
    vm.read_bytes(0x2000, &mut out).unwrap();
    assert_eq!(&out[94..], &[0xa5; 2]);
    for i in 0..8 {
        assert_eq!(&out[14 + i * 10..24 + i * 10], &original.fpr[(3 + i) & 7]);
    }
    cpu.cpu.fpr = [[0; 10]; 8];
    assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
    assert_eq!(cpu.cpu.fpr, original.fpr);
    assert_eq!(
        (cpu.cpu.fpu_top, cpu.cpu.fpu_sw, cpu.cpu.fpu_empty),
        (3, 0x0100, 0)
    );
    assert_eq!(
        (cpu.cpu.fpu_fip, cpu.cpu.fpu_fdp, cpu.cpu.fpu_fop),
        (0x14685, 0x279b7, 0x523)
    );
    assert_eq!((cpu.cpu.fpu_fcs, cpu.cpu.fpu_fds), (0, 0));
}

#[test]
fn fldenv_real_image_retains_registers_and_imports_packed_pointers() {
    // SDM Vol 1 §8.1.10, Fig. 8-12; Vol 2A §3.3 FLDENV, Operation.
    let vm = make_vm(&[0xd9, 0x26, 0x00, 0x20, 0xf4]);
    let words = [0x037fu16, 0x2900, 0x000c, 0x9876, 0xa523, 0x5432, 0xf000];
    let bytes: Vec<u8> = words.into_iter().flat_map(u16::to_le_bytes).collect();
    vm.write_bytes(0x2000, &bytes).unwrap();
    let mut cpu = make_cpu(&vm);
    dirty(&mut cpu);
    cpu.cpu.fpu_sw = 0;
    let registers = cpu.cpu.fpr;
    assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
    assert_eq!(cpu.cpu.fpr, registers);
    assert_eq!(
        (cpu.cpu.fpu_top, cpu.cpu.fpu_sw, cpu.cpu.fpu_empty),
        (5, 0x0100, 2)
    );
    assert_eq!(
        (cpu.cpu.fpu_fip, cpu.cpu.fpu_fdp, cpu.cpu.fpu_fop),
        (0xa9876, 0xf5432, 0x523)
    );
    assert_eq!((cpu.cpu.fpu_fcs, cpu.cpu.fpu_fds), (0, 0));
}

#[test]
fn x87_uses_wrapped_effective_offset_and_default_ss_for_bp() {
    // SDM Vol 1 §§3.3.7, 8.1.8; 80386 PRM Ch. 17, Table 17-2,
    // p. 17-5: BP+SI defaults to SS and 16-bit effective offsets wrap.
    let vm = make_vm(&[0xdd, 0x02, 0xf4]); // FLD qword [BP+SI]
    vm.write_bytes(0x2020, &1f64.to_le_bytes()).unwrap();
    let mut cpu = make_cpu(&vm);
    cpu.set_reg(Reg::Ss, 0x200);
    cpu.set_reg(Reg::Rbp, 0xfff0);
    cpu.set_reg(Reg::Rsi, 0x30);
    assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
    assert_eq!(&cpu.cpu.fpr[7], &[0, 0, 0, 0, 0, 0, 0, 0x80, 0xff, 0x3f]);
    assert_eq!((cpu.cpu.fpu_fip, cpu.cpu.fpu_fcs), (0, 0x100));
    assert_eq!((cpu.cpu.fpu_fdp, cpu.cpu.fpu_fds), (0x20, 0x200));
}

#[test]
fn wide_real_x87_operands_fault_before_any_fpu_commit() {
    // SDM Vol 2A §3.3 FLD/FLDENV/FRSTOR/FSAVE, Real-Address Mode Exceptions.
    for code in [
        &[0xd9, 0x26, 0xff, 0xff][..], // FLDENV
        &[0xdd, 0x26, 0xff, 0xff],     // FRSTOR
        &[0xdd, 0x36, 0xff, 0xff],     // FNSAVE
        &[0xdd, 0x06, 0xff, 0xff],     // FLD m64
    ] {
        let vm = make_vm(code);
        let mut cpu = make_cpu(&vm);
        dirty(&mut cpu);
        cpu.cpu.fpu_sw = 0;
        let before = image(&cpu.cpu);
        assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rip), 0xd01);
        assert_eq!(image(&cpu.cpu), before);
    }
}

#[test]
fn flat_modes_ignore_real16_coprocessor_policy() {
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        let mut vm = make_vm(&[0xdb, 0xe3, 0xf4]);
        vm.set_cpu_mode(mode);
        let mut cpu = vm.new_vcpu();
        dirty(&mut cpu);
        cpu.set_reg(Reg::Rip, 0x1000);
        cpu.set_fpu_model(FpuModel::None);
        cpu.set_fpu_emulation(true);
        assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
        assert_eq!(
            (cpu.fpu_cw(), cpu.cpu.fpu_sw, cpu.cpu.fpu_empty),
            (0x037f, 0, 0xff)
        );
    }
}

#[test]
fn absent_inputs_include_lifter_gaps_and_environment_loads() {
    // 80386 PRM §§11.1.2–11.1.3: ESC classification belongs to the CPU.
    // SDM Vol 2D §B.17, Table B-39; Vol 2A §3.3 FBLD: DF /4 reads m80bcd.
    for input in [
        &[0xdf, 0x26, 0x00, 0x20][..],   // FBLD (Present lifter gap)
        &[0xdf, 0xc0],                   // FFREEP (Present lifter gap)
        &[0xd9, 0x2e, 0x00, 0x20],       // FLDCW
        &[0xd9, 0x26, 0x00, 0x20],       // FLDENV m14
        &[0xdd, 0x26, 0x00, 0x20],       // FRSTOR m94
        &[0x66, 0xdd, 0x26, 0x00, 0x20], // FRSTOR m108 (Present lifter gap)
    ] {
        let mut code = input.to_vec();
        code.push(0xf4);
        let vm = make_vm(&code);
        vm.write_bytes(0x2000, &[0xa5; 108]).unwrap();
        for path in 0..3 {
            let mut cpu = make_cpu(&vm);
            dirty(&mut cpu);
            cpu.set_fpu_model(FpuModel::None);
            let before = image(&cpu.cpu);
            let flags = cpu.flags().to_rflags();
            match path {
                0 => assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt), "{input:02x?}"),
                1 => {
                    let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                    assert_eq!(result.steps(), 1);
                    assert_eq!(cpu.reg(Reg::Rip), input.len() as u64);
                }
                _ => {
                    assert!(matches!(
                        cpu.step_instruction(&vm),
                        x86jit_core::StepResult::Continue
                    ));
                    assert_eq!(cpu.retired_instructions(), 1);
                    assert_eq!(cpu.reg(Reg::Rip), input.len() as u64);
                }
            }
            assert_eq!(image(&cpu.cpu), before, "{input:02x?}");
            assert_eq!(cpu.flags().to_rflags(), flags);
        }
    }
}

#[test]
fn absent_fbld_still_checks_the_complete_cpu_operand() {
    // SDM Vol 2A §3.3 FBLD, Real-Address Mode Exceptions: m80bcd is 10 bytes.
    for (offset, successor) in [(0xfff6u16, 5), (0xfff7, 0xd01)] {
        let [lo, hi] = offset.to_le_bytes();
        let vm = make_vm(&[0xdf, 0x26, lo, hi, 0xf4]);
        let mut cpu = make_cpu(&vm);
        dirty(&mut cpu);
        cpu.set_fpu_model(FpuModel::None);
        let before = image(&cpu.cpu);
        assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rip), successor);
        assert_eq!(image(&cpu.cpu), before);
    }
}

#[test]
fn em_precedes_ud_for_reserved_esc_encodings_on_every_dispatch_path() {
    // 80386 PRM §§11.1.2–11.1.3, pp. 11-1–11-2: 11011b identifies ESC
    // before the coprocessor decodes it. SDM Vol 3A §2.5, Table 2-2: EM => #NM.
    for code in [&[0xd9, 0xd1][..], &[0x26, 0xd9, 0xd1]] {
        let vm = make_vm(code);
        vm.write_bytes(6 * 4, &0x600u16.to_le_bytes()).unwrap();
        vm.write_bytes(0x600, &[0xf4]).unwrap();
        for model in [FpuModel::Present, FpuModel::None] {
            for path in 0..3 {
                let mut cpu = make_cpu(&vm);
                dirty(&mut cpu);
                cpu.set_fpu_model(model);
                cpu.set_fpu_emulation(true);
                let before = image(&cpu.cpu);
                match path {
                    0 => assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt)),
                    1 => {
                        let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                        assert_eq!(result.steps(), 1);
                        assert!(matches!(result.stop(), StepStop::Exit(Exit::Hlt)));
                    }
                    _ => {
                        assert!(matches!(
                            cpu.step_instruction(&vm),
                            x86jit_core::StepResult::Continue
                        ));
                        assert_eq!(cpu.retired_instructions(), 0);
                    }
                }
                assert_eq!(cpu.reg(Reg::Rip), if path == 2 { 0x700 } else { 0x701 });
                assert_eq!(vm.mem.read(0x7fa, 2).unwrap(), 0);
                assert_eq!(vm.mem.read(0x7fc, 2).unwrap(), 0x100);
                assert_eq!(image(&cpu.cpu), before);
            }
        }
        let mut cpu = make_cpu(&vm);
        assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rip), 0x601); // EM=0 retains Present's #UD.
    }
}

#[test]
fn absent_output_length_comes_from_decode_at_the_ip_wrap() {
    // SDM Vol 2D §B.17, Table B-39: DF E0 is exactly FNSTSW AX.
    for code in [&[0xdf, 0xe0][..], &[0x26, 0xdf, 0xe0]] {
        let vm = make_vm(&[0xf4]);
        let ip = 0x10000 - code.len() as u64;
        vm.write_bytes(0x1000 + ip, code).unwrap();
        vm.write_bytes(0x11000, &[0xa5; 15]).unwrap();
        for path in 0..3 {
            let mut cpu = make_cpu(&vm);
            dirty(&mut cpu);
            cpu.set_fpu_model(FpuModel::None);
            cpu.set_reg(Reg::Rip, ip);
            cpu.set_reg(Reg::Rax, 0x12345678);
            let before = image(&cpu.cpu);
            let exit = match path {
                0 => cpu.run(&vm, Some(10)),
                1 => {
                    let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                    assert_eq!(result.steps(), 0);
                    let StepStop::Exit(exit) = result.stop() else {
                        panic!("{result:?}")
                    };
                    exit.clone()
                }
                _ => {
                    let x86jit_core::StepResult::Exit(exit) = cpu.step_instruction(&vm) else {
                        panic!("expected output exit")
                    };
                    exit
                }
            };
            let Exit::UnknownInstruction {
                addr, bytes, len, ..
            } = exit
            else {
                panic!("{exit:?}")
            };
            assert_eq!(addr, ip);
            assert_eq!(usize::from(len), code.len());
            assert_eq!(&bytes[..usize::from(len)], code);
            assert!(bytes[usize::from(len)..].iter().all(|b| *b == 0));
            assert_eq!(cpu.reg(Reg::Rip), ip);
            assert_eq!(cpu.reg(Reg::Rax), 0x12345678);
            assert_eq!(image(&cpu.cpu), before);
        }
    }
}

#[test]
fn ss_wide_operand_keeps_the_scalar_real16_gp_policy() {
    // Deliberate divergence: 80386 PRM §14.7 item 7 (p. 14-6) and SDM
    // Vol 2A §3.3 FLD, Real-Address Mode Exceptions prescribe #SS for SS.
    // Real16 retains its established scalar #GP policy for ordinary BP operands.
    for code in [&[0x8b, 0x46, 0x00][..], &[0xdd, 0x46, 0x00]] {
        for model in [FpuModel::Present, FpuModel::None] {
            let vm = make_vm(code);
            vm.write_bytes(12 * 4, &0xc00u16.to_le_bytes()).unwrap();
            vm.write_bytes(0xc00, &[0xf4]).unwrap();
            let mut cpu = make_cpu(&vm);
            dirty(&mut cpu);
            cpu.cpu.fpu_sw = 0;
            cpu.set_fpu_model(model);
            cpu.set_reg(Reg::Ss, 0x200);
            cpu.set_reg(Reg::Rbp, if code[0] == 0x8b { 0xffff } else { 0xfffc });
            let before = image(&cpu.cpu);
            assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rip), 0xd01);
            assert_eq!(vm.mem.read(0x27fa, 2).unwrap(), 0);
            assert_eq!(image(&cpu.cpu), before);
        }
    }
}

#[test]
fn present_wait_ignores_em_but_still_reports_pending_x87_exceptions() {
    // SDM Vol 3A §2.5, Table 2-2; Vol 2D §6.1 WAIT/FWAIT: EM does not gate WAIT.
    let vm = make_vm(&[0x9b, 0xf4]); // WAIT and FWAIT share opcode 9B.
    vm.write_bytes(16 * 4, &0xe00u16.to_le_bytes()).unwrap();
    vm.write_bytes(0xe00, &[0xf4]).unwrap();
    for pending in [false, true] {
        let mut cpu = make_cpu(&vm);
        dirty(&mut cpu);
        if !pending {
            cpu.cpu.fpu_sw = 0;
        }
        cpu.set_fpu_emulation(true);
        let before = image(&cpu.cpu);
        let exit = cpu.run(&vm, Some(10));
        if pending {
            assert!(matches!(exit, Exit::Hlt), "{exit:?}");
            assert_eq!(cpu.reg(Reg::Rip), 0xe01);
            assert_eq!(vm.mem.read(0x7fa, 2).unwrap(), 0);
        } else {
            assert!(matches!(exit, Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rip), 2);
        }
        assert_eq!(image(&cpu.cpu), before);
    }
}

#[test]
fn present_wide_outputs_are_atomic_and_accept_the_exact_segment_end() {
    // SDM Vol 2A §3.3 FST/FSTP, FSTENV/FNSTENV, FSAVE/FNSAVE,
    // Real-Address Mode Exceptions; Vol 1 §8.1.10: 14/94-byte real images.
    for (opcode, modrm, width) in [(0xdd, 0x1e, 8u16), (0xd9, 0x36, 14), (0xdd, 0x36, 94)] {
        for overrun in [false, true] {
            let offset = 0u16.wrapping_sub(width) + u16::from(overrun);
            let [lo, hi] = offset.to_le_bytes();
            let vm = make_vm(&[opcode, modrm, lo, hi, 0xf4]);
            vm.write_bytes(u64::from(offset) - 1, &vec![0xa5; usize::from(width) + 2])
                .unwrap();
            let mut cpu = make_cpu(&vm);
            dirty(&mut cpu);
            cpu.cpu.fpu_sw = 0;
            let before = image(&cpu.cpu);
            assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rip), if overrun { 0xd01 } else { 5 });
            let mut out = vec![0; usize::from(width) + 2];
            vm.read_bytes(u64::from(offset) - 1, &mut out).unwrap();
            assert_eq!(out[0], 0xa5);
            assert_eq!(*out.last().unwrap(), 0xa5);
            if overrun {
                assert!(out.iter().all(|b| *b == 0xa5));
                assert_eq!(image(&cpu.cpu), before);
            } else {
                assert!(out[1..out.len() - 1].iter().any(|b| *b != 0xa5));
            }
        }
    }
}

#[test]
fn present_lifter_gaps_still_exit_before_operand_checks() {
    // SDM Vol 2A §3.3 FBLD: DF /4 reads a ten-byte packed-BCD operand.
    // Present has no BCD engine; even a crossing operand remains an embedder exit.
    for offset in [0x2000u16, 0xffff] {
        let [lo, hi] = offset.to_le_bytes();
        let code = [0xdf, 0x26, lo, hi];
        let vm = make_vm(&code);
        for path in 0..3 {
            let mut cpu = make_cpu(&vm);
            dirty(&mut cpu);
            let before = image(&cpu.cpu);
            let exit = match path {
                0 => cpu.run(&vm, Some(10)),
                1 => {
                    let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                    assert_eq!(result.steps(), 0);
                    let StepStop::Exit(exit) = result.stop() else {
                        panic!("{result:?}")
                    };
                    exit.clone()
                }
                _ => {
                    let x86jit_core::StepResult::Exit(exit) = cpu.step_instruction(&vm) else {
                        panic!("expected gap exit")
                    };
                    exit
                }
            };
            assert!(
                matches!(exit, Exit::UnknownInstruction { addr: 0, bytes, len: 4, .. } if bytes[..4] == code)
            );
            assert_eq!(cpu.reg(Reg::Rip), 0);
            assert_eq!(image(&cpu.cpu), before);
        }
    }
}

#[test]
fn present_wide_inputs_accept_the_exact_segment_end() {
    // SDM Vol 2A §3.3 FLD/FLDENV/FRSTOR, Real-Address Mode Exceptions;
    // Vol 1 §8.1.10: the 16-bit environment and save images are 14 and 94 bytes.
    for (opcode, modrm, width) in [
        (0xdd, 0x06, 8u16),
        (0xdb, 0x2e, 10),
        (0xd9, 0x26, 14),
        (0xdd, 0x26, 94),
    ] {
        let offset = 0u16.wrapping_sub(width);
        let [lo, hi] = offset.to_le_bytes();
        let vm = make_vm(&[opcode, modrm, lo, hi, 0xf4]);
        let mut cpu = make_cpu(&vm);
        assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rip), 5);
    }
}
