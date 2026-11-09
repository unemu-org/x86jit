//! SDM Vol 2C §5.2, VPGATHERDD/VPGATHERQD, VPGATHERDQ/VPGATHERQQ,
//! VGATHERDPS/VGATHERQPS and VGATHERDPD/VGATHERQPD: VSIB and restart semantics.
use x86jit_core::{
    Backend, CpuMode, Exit, InterpreterBackend, Prot, Reg, RegionCaps, RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

macro_rules! tiers {
    ($body:ident, $interp:ident, $jit:ident, $region:ident) => {
        #[test]
        fn $interp() {
            $body(0);
        }
        #[test]
        fn $jit() {
            $body(1);
        }
        #[test]
        fn $region() {
            $body(2);
        }
    };
}

const CODE: u64 = 0x10000;
const DATA: u64 = 0x20000;
const END: u64 = 0x30000;

fn instruction(float: bool, elem: u8, index: u8, wide: bool) -> Vec<u8> {
    vec![
        0xc4,
        0xe2,
        0x69 | if wide { 4 } else { 0 } | if elem == 8 { 0x80 } else { 0 },
        0x90 + u8::from(index == 8) + 2 * u8::from(float),
        0x04,
        0x88,
    ]
}

fn vm(tier: usize, code: &[u8], mode: CpuMode) -> Vm {
    vm_with_trap(tier, code, mode, false)
}

fn vm_with_trap(tier: usize, code: &[u8], mode: CpuMode, trap: bool) -> Vm {
    let backend: Box<dyn Backend> = match tier {
        0 => Box::new(InterpreterBackend),
        1 => Box::new(JitBackend::new()),
        2 => Box::new(JitBackend::with_superblocks(RegionCaps {
            max_blocks: 16,
            max_icount: 256,
        })),
        _ => unreachable!(),
    };
    let mut vm = Vm::with_backend(VmConfig::flat(END), backend);
    vm.set_cpu_mode(mode);
    vm.set_tier_up_after(None);
    if trap {
        vm.map(0, DATA as usize, Prot::RWX, RegionKind::Ram)
            .unwrap();
        vm.map(DATA, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    } else {
        vm.map(0, END as usize, Prot::RWX, RegionKind::Ram).unwrap();
    }
    let mut bytes = if tier == 2 {
        // A real back edge forces compilation of the gather's region.
        vec![0xeb, 0, 0xff, 0xc2, 0x83, 0xfa, 3, 0x72, 0xf9]
    } else {
        vec![]
    };
    bytes.extend_from_slice(code);
    bytes.push(0xf4);
    vm.write_bytes(CODE, &bytes).unwrap();
    vm
}

fn put(v: &mut [u128; 4], i: usize, size: u8, value: u64) {
    let byte = i * size as usize;
    let mask = (1u128 << (size * 8)) - 1;
    v[byte / 16] = (v[byte / 16] & !(mask << (8 * (byte % 16))))
        | ((value as u128 & mask) << (8 * (byte % 16)));
}

fn gathers_all_forms_all_tiers(tier: usize) {
    for float in [false, true] {
        for elem in [4, 8] {
            for index in [4, 8] {
                for wide in [false, true] {
                    let bytes = if wide { 32 } else { 16 };
                    let n = bytes / elem.max(index) as usize;
                    let vm = vm(
                        tier,
                        &instruction(float, elem, index, wide),
                        CpuMode::Long64,
                    );
                    let mut cpu = vm.new_vcpu();
                    cpu.set_reg(Reg::Rip, CODE);
                    cpu.set_reg(Reg::Rax, DATA + 32);
                    let mut indices = [0; 4];
                    let mut masks = [u128::MAX; 4];
                    let mut expected = [0; 4];
                    for i in 0..n {
                        let offset = i as i64 * 3 - 7;
                        put(&mut indices, i, index, offset as u64);
                        let active = i % 3 != 1;
                        put(
                            &mut masks,
                            i,
                            elem,
                            if active { 1 << (elem * 8 - 1) } else { 7 },
                        );
                        let value = 0x1234567811223300 + i as u64;
                        vm.write_bytes(
                            (DATA as i64 + 32 + offset * 4) as u64,
                            &value.to_le_bytes()[..elem as usize],
                        )
                        .unwrap();
                        put(
                            &mut expected,
                            i,
                            elem,
                            if active { value } else { u64::MAX },
                        );
                    }
                    cpu.cpu.set_vec(0, [u128::MAX; 4], 64);
                    cpu.cpu.set_vec(1, indices, 64);
                    cpu.cpu.set_vec(2, masks, 64);
                    assert!(
                        matches!(cpu.run(&vm, Some(40)), Exit::Hlt),
                        "tier={tier}, float={float}, elem={elem}, index={index}, wide={wide}"
                    );
                    assert_eq!(cpu.cpu.vec_lanes(0), expected);
                    assert_eq!(cpu.cpu.vec_lanes(2), [0; 4]);
                    assert_eq!(cpu.cpu.vec_lanes(1), indices);
                    if tier == 2 {
                        assert!(vm.cache.regions() > 0);
                    }
                }
            }
        }
    }
}

fn gathers_fault_progress_and_restart_all_tiers(tier: usize) {
    for elem in [4, 8] {
        for index in [4, 8] {
            let mut code = instruction(false, elem, index, true);
            code[5] = 0x08; // scale 1, index register 1
            let vm = vm(tier, &code, CpuMode::Long64);
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, CODE);
            cpu.set_reg(Reg::Rax, DATA);
            let n = 32 / elem.max(index) as usize;
            let mut indices = [0; 4];
            let mut masks = [0; 4];
            let mut expected = [0; 4];
            for i in 0..n {
                put(
                    &mut indices,
                    i,
                    index,
                    if i == 1 {
                        END - DATA - 2
                    } else {
                        i as u64 * 16
                    },
                );
                put(&mut masks, i, elem, 1 << (elem * 8 - 1));
                put(&mut expected, i, elem, 0x55);
            }
            vm.write_bytes(DATA, &123u64.to_le_bytes()).unwrap();
            cpu.cpu.set_vec(0, expected, 64);
            cpu.cpu.set_vec(1, indices, 64);
            cpu.cpu.set_vec(2, masks, 64);
            let exit = cpu.run(&vm, Some(40));
            assert!(
                matches!(exit, Exit::UnmappedMemory { addr: END, .. }),
                "{exit:?}"
            );
            put(&mut expected, 0, elem, 123);
            assert_eq!(cpu.cpu.vec_lanes(0), expected);
            put(&mut masks, 0, elem, 0);
            // The SDM Operation normalizes uncompleted active elements to all ones.
            for i in 1..n {
                put(&mut masks, i, elem, u64::MAX);
            }
            assert_eq!(cpu.cpu.vec_lanes(2), masks);
            let rip = cpu.reg(Reg::Rip);
            assert_eq!(rip, CODE + if tier == 2 { 9 } else { 0 });
            // Repair the index and change an already completed source: it must not reload.
            put(&mut indices, 1, index, 8);
            cpu.cpu.set_vec(1, indices, 64);
            vm.write_bytes(DATA, &999u64.to_le_bytes()).unwrap();
            vm.write_bytes(DATA + 8, &456u64.to_le_bytes()).unwrap();
            assert!(matches!(cpu.run(&vm, Some(40)), Exit::Hlt));
            put(&mut expected, 1, elem, 456);
            for i in 2..n {
                put(&mut expected, i, elem, 0);
            }
            assert_eq!(cpu.cpu.vec_lanes(0), expected);
            assert_eq!(cpu.cpu.vec_lanes(2), [0; 4]);
        }
    }
}

fn gathers_vsib_address_sizes_segments_and_scales(tier: usize) {
    // SDM Vol 2C §5.2, VPGATHERDD/VPGATHERQD: signed indices, scale and
    // address-size truncation; Vol 1 §3.3.7: FS/GS bases follow offset formation.

    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for addr32 in [false, true] {
            if mode == CpuMode::Compat32 && !addr32 {
                continue;
            }
            for no_base in [false, true] {
                for segment in [0, 0x64, 0x65] {
                    for scale in 0..4 {
                        let mut code = instruction(false, 4, 4, false);
                        code[4] = if no_base { 0x04 } else { 0x44 };
                        code[5] = (scale << 6) | 8 | if no_base { 5 } else { 0 };
                        if no_base {
                            code.extend_from_slice(&(DATA as i32 + 4).to_le_bytes());
                        } else {
                            code.push(4);
                        }
                        if addr32 && mode == CpuMode::Long64 {
                            code.insert(0, 0x67);
                        }
                        if segment != 0 {
                            code.insert(0, segment);
                        }
                        let vm = vm(tier, &code, mode);
                        let mut cpu = vm.new_vcpu();
                        cpu.set_reg(Reg::Rip, CODE);
                        let idx = if no_base {
                            (-4i32) as u32 as u64
                        } else if addr32 {
                            (DATA >> scale) + 4
                        } else {
                            4
                        };
                        let base = if addr32 { (1u64 << 32) - 8 } else { DATA - 8 };
                        cpu.set_reg(Reg::Rax, base);
                        let mut indices = [0; 4];
                        put(&mut indices, 0, 4, idx);
                        cpu.cpu.set_vec(1, indices, 64);
                        cpu.cpu.set_vec(2, [0x80000000, 0, 0, 0], 64);
                        let offset = if no_base {
                            DATA + 4 - (4 << scale)
                        } else {
                            DATA - 4 + (4 << scale)
                        };
                        let seg_base = if segment == 0 { 0 } else { 0x1000 };
                        cpu.set_reg(Reg::FsBase, seg_base);
                        cpu.set_reg(Reg::GsBase, seg_base);
                        vm.write_bytes(offset + seg_base, &0x12345678u32.to_le_bytes())
                            .unwrap();
                        assert!(matches!(cpu.run(&vm, Some(40)), Exit::Hlt), "tier={tier}, mode={mode:?}, addr32={addr32}, no_base={no_base}, segment={segment}, scale={scale}");
                        assert_eq!(cpu.xmm(0), 0x12345678);
                    }
                }
            }
        }
    }
}

fn gathers_mask_suppresses_invalid_addresses_and_loads_exact_width(tier: usize) {
    // SDM Vol 2C §5.2, all VEX gather Operation sections: inactive lanes
    // merge, do not fetch, and unused destination/mask bits become zero.

    for float in [false, true] {
        for elem in [4, 8] {
            for index in [4, 8] {
                for wide in [false, true] {
                    let vm = vm(
                        tier,
                        &instruction(float, elem, index, wide),
                        CpuMode::Long64,
                    );
                    let mut cpu = vm.new_vcpu();
                    cpu.set_reg(Reg::Rip, CODE);
                    cpu.set_reg(Reg::Rax, END - u64::from(elem));
                    let mut indices = [u128::MAX; 4];
                    put(&mut indices, 0, index, 0);
                    let mut masks = [0; 4];
                    put(&mut masks, 0, elem, u64::MAX);
                    // Every other address is invalid, but its sign-bit-clear mask suppresses it.
                    cpu.cpu.set_vec(1, indices, 64);
                    cpu.cpu.set_vec(2, masks, 64);
                    vm.write_bytes(
                        END - u64::from(elem),
                        &0xabcdefu64.to_le_bytes()[..elem as usize],
                    )
                    .unwrap();
                    assert!(matches!(cpu.run(&vm, Some(40)), Exit::Hlt));
                    assert_eq!(cpu.cpu.vec_lanes(0), [0xabcdef, 0, 0, 0]);
                    assert_eq!(cpu.cpu.vec_lanes(2), [0; 4]);
                }
            }
        }
    }
    let vm = vm(tier, &instruction(false, 4, 4, true), CpuMode::Long64);
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rax, 0x8000000000000000);
    cpu.cpu.set_vec(0, [u128::MAX; 4], 64);
    cpu.cpu.set_vec(2, [0x123; 4], 64);
    assert!(matches!(cpu.run(&vm, Some(40)), Exit::Hlt));
    assert_eq!(cpu.cpu.vec_lanes(0), [u128::MAX, u128::MAX, 0, 0]);
    assert_eq!(cpu.cpu.vec_lanes(2), [0; 4]);
}

fn gathers_overlapping_registers_raise_ud_without_commit(tier: usize) {
    // SDM Vol 2C §5.2, VPGATHERDD/VPGATHERQD: any pair of destination,
    // index and mask registers overlapping raises #UD.

    for (dest, index, mask) in [(0, 0, 2), (0, 1, 0), (0, 1, 1)] {
        let mut code = instruction(false, 4, 4, true);
        code[2] = 0x05 | ((!mask & 15) << 3);
        code[4] = 4 | (dest << 3);
        code[5] = 0x80 | (index << 3);
        let vm = vm(tier, &code, CpuMode::Long64);
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, CODE);
        for i in 0..3 {
            cpu.cpu.set_vec(i, [u128::MAX; 4], 64);
        }
        assert!(matches!(
            cpu.run(&vm, Some(40)),
            Exit::Exception { vector: 6, .. }
        ));
        for i in 0..3 {
            assert_eq!(cpu.cpu.vec_lanes(i), [u128::MAX; 4]);
        }
        assert_eq!(cpu.reg(Reg::Rip), CODE + if tier == 2 { 9 } else { 0 });
    }
}

fn oracle_input(
    float: bool,
    elem: u8,
    index: u8,
    wide: bool,
    fault: bool,
) -> x86jit_tests::oracle::VectorInput {
    use x86jit_tests::vector::{CpuSnapshot, MemChunk, MemKind, RunSpec};
    let mut code = instruction(float, elem, index, wide);
    code[5] = 0x08; // scale 1
    code.push(0xf4);
    let n = (if wide { 32 } else { 16 }) / usize::from(elem.max(index));
    let mut indices = [0; 4];
    let mut masks = [0; 4];
    for i in 0..n {
        let offset = if fault && i == 1 {
            END - DATA - 4 - u64::from(elem) + 1
        } else {
            (i as u64 * 16).wrapping_sub(4)
        };
        put(&mut indices, i, index, offset);
        put(
            &mut masks,
            i,
            elem,
            if !fault || i < 2 { u64::MAX } else { 0 },
        );
    }
    let mut cpu = CpuSnapshot {
        rip: CODE,
        ..CpuSnapshot::default()
    };
    cpu.gpr[0] = DATA + 4;
    cpu.xmm[0] = 0x8877665544332211;
    cpu.ymm_hi[0] = if fault { 0 } else { u128::MAX };
    cpu.xmm[1] = indices[0];
    cpu.ymm_hi[1] = indices[1];
    cpu.xmm[2] = masks[0];
    cpu.ymm_hi[2] = masks[1];
    x86jit_tests::oracle::VectorInput {
        cpu_init: cpu,
        mem_init: vec![
            MemChunk {
                addr: CODE,
                bytes: code,
                kind: MemKind::Ram,
            },
            MemChunk {
                addr: DATA,
                bytes: (0..END - DATA).map(|i| i as u8).collect(),
                kind: MemKind::Ram,
            },
        ],
        entry: CODE,
        run: RunSpec::UntilExit,
    }
}

#[test]
fn gathers_jit_equals_interp_all_forms_and_faults() {
    use x86jit_tests::{compare::compare, oracle::run_with_backend};
    for float in [false, true] {
        for elem in [4, 8] {
            for index in [4, 8] {
                for wide in [false, true] {
                    for fault in [false, true] {
                        let input = oracle_input(float, elem, index, wide, fault);
                        let interp = run_with_backend(&input, Box::new(InterpreterBackend));
                        let jit = run_with_backend(&input, Box::new(JitBackend::new()));
                        if fault {
                            assert!(matches!(
                                interp.exit,
                                x86jit_tests::vector::ExitKind::UnmappedMemory { addr: END, .. }
                            ));
                        } else {
                            assert_eq!(interp.exit, x86jit_tests::vector::ExitKind::Hlt);
                        }
                        assert!(
                            compare(&interp, &jit, &[]).is_none(),
                            "{:?}",
                            compare(&interp, &jit, &[])
                        );
                    }
                }
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn gathers_native_all_forms_and_faults() {
    use x86jit_tests::{
        compare::compare,
        native::{run_native, run_native_fault, NativeFault},
        oracle::run_with_backend,
    };
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    for float in [false, true] {
        for elem in [4, 8] {
            for index in [4, 8] {
                for wide in [false, true] {
                    for fault in [false, true] {
                        let input = oracle_input(float, elem, index, wide, fault);
                        let native = if fault {
                            run_native_fault(&input, NativeFault::PageFault { addr: END })
                        } else {
                            run_native(&input)
                        }
                        .expect("AVX2 gather native fixture");
                        let interp = run_with_backend(&input, Box::new(InterpreterBackend));
                        assert!(compare(&native, &interp, &[]).is_none(), "float={float}, elem={elem}, index={index}, wide={wide}, fault={fault}: {:?}", compare(&native, &interp, &[]));
                    }
                }
            }
        }
    }
}

#[cfg(feature = "unicorn")]
#[test]
fn gathers_unicorn_probe_and_differential_when_supported() {
    use unicorn_engine::{
        uc_error,
        unicorn_const::{Arch, Mode, Prot as UProt},
        RegisterX86, Unicorn,
    };
    use x86jit_tests::oracle::run_with_backend;
    for float in [false, true] {
        for elem in [4, 8] {
            for index in [4, 8] {
                for wide in [false, true] {
                    let mut uc = Unicorn::new(Arch::X86, Mode::MODE_64).unwrap();
                    uc.mem_map(CODE, 4096, UProt::ALL).unwrap();
                    let code = instruction(float, elem, index, wide);
                    uc.mem_write(CODE, &code).unwrap();
                    // A zero mask needs no mapped data. Distinguish unsupported AVX2
                    // from other errors before handing the fixture to the oracle.
                    match uc.emu_start(CODE, CODE + code.len() as u64, 0, 1) {
                        Err(uc_error::INSN_INVALID) => {
                            eprintln!("Unicorn gather unsupported: {:02x?}, INSN_INVALID", code);
                            continue;
                        }
                        Err(e) => panic!("Unicorn gather probe: {e:?}"),
                        Ok(()) => {}
                    }
                    let input = oracle_input(float, elem, index, wide, false);
                    uc.mem_map(DATA, END - DATA, UProt::ALL).unwrap();
                    uc.mem_write(DATA, &input.mem_init[1].bytes).unwrap();
                    uc.mem_write(CODE, &input.mem_init[0].bytes).unwrap();
                    uc.reg_write(RegisterX86::RAX, input.cpu_init.gpr[0])
                        .unwrap();
                    let regs = [RegisterX86::YMM0, RegisterX86::YMM1, RegisterX86::YMM2];
                    for (i, reg) in regs.iter().enumerate() {
                        let mut value = input.cpu_init.xmm[i].to_le_bytes().to_vec();
                        value.extend_from_slice(&input.cpu_init.ymm_hi[i].to_le_bytes());
                        uc.reg_write_long(*reg, &value).unwrap();
                    }
                    let flags = uc.reg_read(RegisterX86::EFLAGS).unwrap();
                    uc.emu_start(CODE, CODE + code.len() as u64, 0, 1).unwrap();
                    assert_eq!(uc.reg_read(RegisterX86::EFLAGS).unwrap(), flags);
                    let interp = run_with_backend(&input, Box::new(InterpreterBackend));
                    assert_eq!(interp.exit, x86jit_tests::vector::ExitKind::Hlt);
                    for (i, reg) in regs.iter().enumerate() {
                        let value = uc.reg_read_long(*reg).unwrap();
                        assert_eq!(
                            u128::from_le_bytes(value[..16].try_into().unwrap()),
                            interp.cpu.xmm[i]
                        );
                        assert_eq!(
                            u128::from_le_bytes(value[16..32].try_into().unwrap()),
                            interp.cpu.ymm_hi[i]
                        );
                    }
                }
            }
        }
    }
}

tiers!(
    gathers_all_forms_all_tiers,
    gathers_all_forms_all_tiers_interp,
    gathers_all_forms_all_tiers_jit,
    gathers_all_forms_all_tiers_region
);

tiers!(
    gathers_fault_progress_and_restart_all_tiers,
    gathers_fault_progress_and_restart_all_tiers_interp,
    gathers_fault_progress_and_restart_all_tiers_jit,
    gathers_fault_progress_and_restart_all_tiers_region
);

tiers!(
    gathers_vsib_address_sizes_segments_and_scales,
    gathers_vsib_address_sizes_segments_and_scales_interp,
    gathers_vsib_address_sizes_segments_and_scales_jit,
    gathers_vsib_address_sizes_segments_and_scales_region
);

tiers!(
    gathers_mask_suppresses_invalid_addresses_and_loads_exact_width,
    gathers_mask_suppresses_invalid_addresses_and_loads_exact_width_interp,
    gathers_mask_suppresses_invalid_addresses_and_loads_exact_width_jit,
    gathers_mask_suppresses_invalid_addresses_and_loads_exact_width_region
);

tiers!(
    gathers_overlapping_registers_raise_ud_without_commit,
    gathers_overlapping_registers_raise_ud_without_commit_interp,
    gathers_overlapping_registers_raise_ud_without_commit_jit,
    gathers_overlapping_registers_raise_ud_without_commit_region
);

fn gathers_fault_order_and_first_element(tier: usize) {
    // SDM Vol 2C §5.2, gather Description: every selected lower element is
    // complete at a delivered fault; higher elements may remain uncompleted.
    // The engine chooses ascending order, including on the first active element.
    for float in [false, true] {
        for elem in [4, 8] {
            for index in [4, 8] {
                for wide in [false, true] {
                    let n = (if wide { 32 } else { 16 }) / usize::from(elem.max(index));
                    for first_fault in 0..n {
                        let mut code = instruction(float, elem, index, wide);
                        code[5] = 0x08;
                        let vm = vm(tier, &code, CpuMode::Long64);
                        let mut cpu = vm.new_vcpu();
                        cpu.set_reg(Reg::Rip, CODE);
                        cpu.set_reg(Reg::Rax, DATA);
                        let mut indices = [0; 4];
                        let mut masks = [0; 4];
                        let mut expected = [0; 4];
                        for i in 0..n {
                            // Distinct unavailable addresses also witness fault priority.
                            let offset = if i < first_fault {
                                i as u64 * 16
                            } else {
                                END - DATA - 1 + (i - first_fault) as u64 * 32
                            };
                            put(&mut indices, i, index, offset);
                            put(&mut masks, i, elem, u64::MAX);
                            put(
                                &mut expected,
                                i,
                                elem,
                                if i < first_fault { i as u64 + 1 } else { 0x99 },
                            );
                            vm.write_bytes(DATA + i as u64 * 16, &(i as u64 + 1).to_le_bytes())
                                .unwrap();
                        }
                        // Seed each destination element identically at either width.
                        let mut initial = [0; 4];
                        for i in 0..n {
                            put(&mut initial, i, elem, 0x99);
                        }
                        cpu.cpu.set_vec(0, initial, 64);
                        cpu.cpu.set_vec(1, indices, 64);
                        cpu.cpu.set_vec(2, masks, 64);
                        let exit = cpu.run(&vm, Some(40));
                        let Exit::UnmappedMemory {
                            addr,
                            metadata: Some(metadata),
                            ..
                        } = exit
                        else {
                            panic!("{exit:?}");
                        };
                        assert_eq!(addr, END);
                        assert_eq!(
                            metadata.operand_base(),
                            x86jit_core::Knowledge::Known(END - 1)
                        );
                        assert_eq!(
                            metadata.operand_size(),
                            x86jit_core::Knowledge::Known(u16::from(elem))
                        );
                        assert_eq!(
                            metadata.transfer_size(),
                            x86jit_core::Knowledge::Known(u16::from(elem))
                        );
                        assert_eq!(cpu.cpu.vec_lanes(0), expected);
                        for i in 0..first_fault {
                            put(&mut masks, i, elem, 0);
                        }
                        assert_eq!(cpu.cpu.vec_lanes(2), masks);
                        if tier == 2 {
                            assert!(vm.cache.regions() > 0);
                        }
                    }
                }
            }
        }
    }
}

tiers!(
    gathers_fault_order_and_first_element,
    gathers_fault_order_interp,
    gathers_fault_order_jit,
    gathers_fault_order_region
);

#[test]
fn gathers_high_registers_and_noncanonical_fault_all_tiers() {
    // SDM Vol 2C §5.2, VPGATHERDD/VPGATHERQD: VSIB selects the extended
    // vector register, and a noncanonical selected element faults after lower loads.
    for tier in 0..3 {
        let mut code = instruction(false, 8, 8, true);
        code[1] = 0x22; // R/X extensions, base RAX
        code[2] = 0x95; // mask YMM13, W=1, L=1
        code[4] = 0x3c; // destination YMM15
        code[5] = 0x30; // index YMM14, scale 1
        let vm = vm(tier, &code, CpuMode::Long64);
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, CODE);
        cpu.set_reg(Reg::Rax, DATA);
        let mut indices = [0; 4];
        put(&mut indices, 1, 8, 0x8000_0000_0000_0000 - DATA);
        cpu.cpu.set_vec(14, indices, 64);
        cpu.cpu.set_vec(13, [u128::MAX, 0, 0, 0], 64);
        vm.write_bytes(DATA, &123u64.to_le_bytes()).unwrap();
        let exit = cpu.run(&vm, Some(40));
        assert!(
            matches!(exit, Exit::Exception { vector: 13, .. }),
            "{exit:?}"
        );
        assert_eq!(cpu.cpu.vec_lanes(15), [123, 0, 0, 0]);
        assert_eq!(cpu.cpu.vec_lanes(13), [u128::from(u64::MAX) << 64, 0, 0, 0]);
    }
}

#[test]
fn gathers_compat_probe_measures_all_forms() {
    use iced_x86::{Decoder, DecoderOptions};
    use x86jit_tests::compat::{probe_code_in, Probe};
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for float in [false, true] {
            for elem in [4, 8] {
                for index in [4, 8] {
                    for wide in [false, true] {
                        let bytes = instruction(float, elem, index, wide);
                        let code = Decoder::new(
                            if mode == CpuMode::Long64 { 64 } else { 32 },
                            &bytes,
                            DecoderOptions::NONE,
                        )
                        .decode()
                        .code();
                        assert_eq!(
                            probe_code_in(code, mode),
                            Some(Probe::Lifted),
                            "{code:?}, {mode:?}"
                        );
                    }
                }
            }
        }
    }
}

fn gathers_route_vsib_elements_to_mmio(tier: usize) {
    for float in [false, true] {
        for elem in [4, 8] {
            for index in [4, 8] {
                for wide in [false, true] {
                    for first_fault in [0, 1] {
                        let mut code = instruction(float, elem, index, wide);
                        code[5] = 0x08; // index register 1, scale 1
                        let vm = vm_with_trap(tier, &code, CpuMode::Long64, true);
                        let mut cpu = vm.new_vcpu();
                        cpu.set_reg(Reg::Rip, CODE);
                        cpu.set_reg(Reg::Rax, DATA - 0x1000);
                        vm.write_bytes(DATA - 0x1000, &123u64.to_le_bytes())
                            .unwrap();
                        let mut indices = [0; 4];
                        put(&mut indices, first_fault, index, 0x1000);
                        let mut masks = [0; 4];
                        for i in 0..=first_fault {
                            put(&mut masks, i, elem, 1 << (elem * 8 - 1));
                        }
                        cpu.cpu.set_vec(1, indices, 64);
                        cpu.cpu.set_vec(2, masks, 64);
                        let exit = cpu.run(&vm, Some(40));
                        assert!(
                            matches!(exit, Exit::MmioRead { addr: DATA, size, .. } if size == elem),
                            "tier={tier}, float={float}, elem={elem}, index={index}, wide={wide}: {exit:?}"
                        );
                        assert_eq!(cpu.reg(Reg::Rip), CODE + if tier == 2 { 9 } else { 0 });
                        let mut expected = [0; 4];
                        if first_fault == 1 {
                            put(&mut expected, 0, elem, 123);
                        }
                        assert_eq!(cpu.cpu.vec_lanes(0), expected);
                        masks = [0; 4];
                        put(&mut masks, first_fault, elem, u64::MAX);
                        assert_eq!(cpu.cpu.vec_lanes(2), masks);
                        // SDM Vol 2C §5.2, gather Operation: completed elements have
                        // cleared masks and must not be reloaded on restart.
                        vm.write_bytes(DATA - 0x1000, &999u64.to_le_bytes())
                            .unwrap();
                        cpu.complete_mmio_read(456);
                        assert!(matches!(cpu.run(&vm, Some(40)), Exit::Hlt));
                        put(&mut expected, first_fault, elem, 456);
                        assert_eq!(cpu.cpu.vec_lanes(0), expected);
                        assert_eq!(cpu.cpu.vec_lanes(2), [0; 4]);
                        if tier == 2 {
                            assert!(vm.cache.regions() > 0);
                        }
                    }
                }
            }
        }
    }
}

tiers!(
    gathers_route_vsib_elements_to_mmio,
    gathers_route_vsib_elements_to_mmio_interp,
    gathers_route_vsib_elements_to_mmio_jit,
    gathers_route_vsib_elements_to_mmio_region
);

fn gathers_partial_register_progress_metadata(tier: usize) {
    for float in [false, true] {
        for elem in [4, 8] {
            for index in [4, 8] {
                for wide in [false, true] {
                    let mut code = instruction(float, elem, index, wide);
                    code[5] = 0x08;
                    let vm = vm(tier, &code, CpuMode::Long64);
                    vm.write_bytes(DATA, &123u64.to_le_bytes()).unwrap();
                    let mut cpu = vm.new_vcpu();
                    cpu.set_reg(Reg::Rip, CODE);
                    cpu.set_reg(Reg::Rax, DATA);
                    let mut indices = [0; 4];
                    put(&mut indices, 1, index, END - DATA);
                    let mut masks = [0; 4];
                    for i in 0..2 {
                        put(&mut masks, i, elem, 1 << (elem * 8 - 1));
                    }
                    cpu.cpu.set_vec(1, indices, 64);
                    cpu.cpu.set_vec(2, masks, 64);
                    let exit = cpu.run(&vm, Some(40));
                    assert!(
                        matches!(exit, Exit::UnmappedMemory { addr: END, .. }),
                        "{exit:?}"
                    );
                    // SDM Vol 2C §5.2, gather Description: lower elements complete
                    // before delivery of a higher element's fault.
                    assert_eq!(cpu.cpu.vec_lanes(0), [123, 0, 0, 0]);
                    masks = [0; 4];
                    put(&mut masks, 1, elem, u64::MAX);
                    assert_eq!(cpu.cpu.vec_lanes(2), masks);
                    let progress = exit.fault_metadata().unwrap().progress();
                    assert_eq!(progress.registers(), x86jit_core::RegisterProgress::Unknown);
                    assert_eq!(
                        progress.had_partial_progress(),
                        x86jit_core::Knowledge::Unknown
                    );
                    if tier == 2 {
                        assert!(vm.cache.regions() > 0);
                    }
                }
            }
        }
    }
}

tiers!(
    gathers_partial_register_progress_metadata,
    gathers_partial_register_progress_metadata_interp,
    gathers_partial_register_progress_metadata_jit,
    gathers_partial_register_progress_metadata_region
);
