//! SDM Vol 2A §3.2 CLC/CMC, Vol 2B §4.3 STC, PUSHF/POPF, MOV,
//! and Vol 2A §3.2 IRET: user-mode flag images and selector witnesses.
use x86jit_core::{
    Backend, CpuMode, Exit, InterpreterBackend, Prot, Reg, RegionCaps, RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

fn machine(body: &[u8], mode: CpuMode, tier: usize) -> Vm {
    let backend: Box<dyn Backend> = match tier {
        0 => Box::new(InterpreterBackend),
        1 => Box::new(JitBackend::new()),
        _ => Box::new(JitBackend::with_superblocks(RegionCaps {
            max_blocks: 16,
            max_icount: 256,
        })),
    };
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
    vm.set_cpu_mode(mode);
    vm.map(0, 0x8000, Prot::RWX, RegionKind::Ram).unwrap();
    // The forward edge followed by a backedge gives the region tier a real region.
    let mut code = vec![0xeb, 1, 0x90, 0x75, 0xfb];
    code.extend_from_slice(body);
    code.extend_from_slice(&[0x75, (-(body.len() as i8 + 7)) as u8, 0xf4]);
    vm.write_bytes(0x1000, &code).unwrap();
    vm
}

#[test]
fn carry_and_flag_stack_all_tiers() {
    // STC; CMC; CLC; ADD eax,1; PUSHF; SUB eax,eax; POP ebx/rbx.
    // SDM Vol 2B PUSHF: the pushed flags observe ADD before SUB overwrites them.
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for tier in 0..3 {
            let vm = machine(
                &[0xf9, 0xf5, 0xf8, 0x83, 0xc0, 1, 0x9c, 0x29, 0xc0, 0x5b],
                mode,
                tier,
            );
            let mut cpu = vm.new_vcpu();
            cpu.cpu.flags.zf = true;
            cpu.set_reg(Reg::Rip, 0x1000);
            cpu.set_reg(Reg::Rsp, 0x7000);
            cpu.set_reg(Reg::Rax, 0xffff_ffff);
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rbx) & 0x8d5, 0x55);
            assert_eq!(cpu.reg(Reg::Rsp), 0x7000);
            if tier == 2 {
                assert!(vm.cache.regions() > 0);
            }
        }
    }
}

#[test]
fn selector_store_widths_all_tiers() {
    // SDM Vol 2B MOV: r32 zero extends; r16 preserves the rest; memory is 16 bits.
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for tier in 0..3 {
            let vm = machine(
                &[0x8c, 0xc8, 0x66, 0x8c, 0xcb, 0x8c, 0x0f, 0x39, 0xc0],
                mode,
                tier,
            );
            vm.write_bytes(0x6000, &[0xaa; 8]).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu.flags.zf = true;
            cpu.cpu.cs = 0x33;
            cpu.set_reg(Reg::Rip, 0x1000);
            cpu.set_reg(Reg::Rax, u64::MAX);
            cpu.set_reg(Reg::Rbx, u64::MAX);
            cpu.set_reg(Reg::Rdi, 0x6000);
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rax), 0x33);
            assert_eq!(cpu.reg(Reg::Rbx), 0xffff_ffff_ffff_0033);
            let mut data = [0; 8];
            vm.read_bytes(0x6000, &mut data).unwrap();
            assert_eq!(data, [0x33, 0, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa]);
            if tier == 2 {
                assert!(vm.cache.regions() > 0);
            }
        }
    }
}

#[test]
fn popf_privilege_masks_and_operand_sizes_all_tiers() {
    // SDM Vol 2B §4.3 POPF, Table 4-20: CPL=3 preserves IOPL/VM/VIF/VIP,
    // clears RF, and changes IF only for old IOPL=3. Word POPF preserves AC/ID.
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for tier in 0..3 {
            for word in [false, true] {
                for iopl in [0, 3] {
                    for image in [0u64, u64::MAX, 0x24_0101] {
                        let mut body = if word { vec![0x66, 0x9d] } else { vec![0x9d] };
                        body.extend([0x9c, 0x58, 0x39, 0xc9]);
                        let vm = machine(&body, mode, tier);
                        vm.write_bytes(0x7000, &image.to_le_bytes()).unwrap();
                        let mut cpu = vm.new_vcpu();
                        cpu.cpu.flags.zf = true;
                        cpu.set_reg(Reg::Rip, 0x1000);
                        cpu.set_reg(Reg::Rsp, 0x7000);
                        cpu.cpu.flags.if_ = true;
                        // Include reserved bits to witness that a restore leaves them untouched.
                        let old = 0xa500_0000_001b_0000 | (iopl << 12) | (1 << 18);
                        cpu.cpu.rflags_system = old;
                        let exit = cpu.run(&vm, None);
                        assert!(
                            matches!(exit, Exit::Hlt),
                            "{mode:?}/{tier}/{word}: {exit:?}"
                        );
                        let changed = 0x4100 | if word { 0 } else { 0x24_0000 };
                        let expected_system = (old & !(changed | (1 << 16))) | (image & changed);
                        assert_eq!(cpu.cpu.rflags_system, expected_system);
                        let expected_if = iopl != 3 || image & (1 << 9) != 0;
                        assert_eq!(cpu.cpu.flags.if_, expected_if);
                        let expected = (image & 0xcd5)
                            | 2
                            | ((expected_if as u64) << 9)
                            | (expected_system & 0x3c_7100);
                        assert_eq!(cpu.reg(Reg::Rax), expected);
                        let slot = if word {
                            2
                        } else if mode == CpuMode::Long64 {
                            8
                        } else {
                            4
                        };
                        assert_eq!(cpu.reg(Reg::Rsp), 0x7000 + slot);
                        if tier == 2 {
                            assert!(vm.cache.regions() > 0);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn carry_changes_only_cf_all_tiers() {
    // SDM Vol 2A §3.2 CLC/CMC; Vol 2B §4.3 STC, Flags Affected.
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for tier in 0..3 {
            for (opcode, old_cf, cf) in [
                (0xf8, true, false),
                (0xf9, false, true),
                (0xf5, false, true),
                (0xf5, true, false),
            ] {
                let vm = machine(&[opcode, 0x9c, 0x58, 0x39, 0xc9], mode, tier);
                let mut cpu = vm.new_vcpu();
                cpu.cpu.flags.zf = true;
                cpu.set_reg(Reg::Rip, 0x1000);
                cpu.set_reg(Reg::Rsp, 0x7000);
                cpu.cpu.flags.set_flags16(0xed4 | old_cf as u16);
                cpu.cpu.rflags_system = 0x24_0100;
                assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
                assert_eq!(cpu.reg(Reg::Rax), 0x24_0fd6 | cf as u64);
            }
        }
    }
}

#[test]
fn cmc_before_fault_preserves_exact_flags_all_tiers() {
    // SDM Vol 2A §3.2 ADD/CMC: carry is complemented, other arithmetic bits survive.
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for tier in 0..3 {
            for (eax, expected) in [(u32::MAX, 0x54), (0, 1)] {
                let vm = machine(&[0x83, 0xc0, 1, 0xf5, 0x89, 0x07, 0x29, 0xc0], mode, tier);
                let mut cpu = vm.new_vcpu();
                cpu.cpu.flags.zf = true;
                cpu.set_reg(Reg::Rip, 0x1000);
                cpu.set_reg(Reg::Rax, eax as u64);
                cpu.set_reg(Reg::Rdi, 0x10000);
                let exit = cpu.run(&vm, None);
                assert!(
                    matches!(exit, Exit::UnmappedMemory { addr: 0x10000, .. }),
                    "{mode:?}/{tier}: {exit:?}"
                );
                assert_eq!(cpu.reg(Reg::Rip), 0x1009);
                assert_eq!(cpu.flags().to_rflags() & 0x8d5, expected);
                if tier == 2 {
                    assert!(vm.cache.regions() > 0);
                }
            }
        }
    }
}

#[test]
fn selector_reads_and_descriptor_load_exits_all_tiers() {
    use x86jit_core::lift::{lift_block, FetchAddr};
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for tier in 0..3 {
            for segment in 0..6 {
                // SDM Vol 2B §4.3 MOV: all six visible selectors are readable.
                let vm = machine(&[0x8c, 0xc0 | segment << 3, 0x39, 0xc9], mode, tier);
                let mut cpu = vm.new_vcpu();
                cpu.cpu.flags.zf = true;
                cpu.set_reg(Reg::Rip, 0x1000);
                cpu.cpu.es = 0x10;
                cpu.cpu.cs = 0x33;
                cpu.cpu.ss = 0x2b;
                cpu.cpu.ds = 0x20;
                cpu.cpu.fs = 0x43;
                cpu.cpu.gs = 0x53;
                assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
                assert_eq!(
                    cpu.reg(Reg::Rax),
                    [0x10, 0x33, 0x2b, 0x20, 0x43, 0x53][segment as usize]
                );
            }
            let mut forms = vec![
                vec![0x8e, 0xd8],
                vec![0x8e, 0x1f],
                vec![0x0f, 0xa1],
                vec![0x0f, 0xa9],
            ];
            if mode == CpuMode::Compat32 {
                forms.extend([vec![0x07], vec![0x17], vec![0x1f]]);
            }
            for form in forms {
                let vm = machine(&form, mode, tier);
                assert!(lift_block(&vm.mem, FetchAddr::flat(0x1005), mode).is_ok());
                let mut cpu = vm.new_vcpu();
                cpu.cpu.flags.zf = true;
                cpu.set_reg(Reg::Rip, 0x1000);
                cpu.set_reg(Reg::Rsp, 0x8000); // no stack/source read before delegation
                cpu.set_reg(Reg::Rdi, 0x8000);
                cpu.cpu.fs_base = 0x1234;
                cpu.cpu.gs_base = 0x5678;
                let before = cpu.cpu.clone();
                match cpu.run(&vm, None) {
                    Exit::UnknownInstruction {
                        addr, bytes, len, ..
                    } => {
                        assert_eq!(addr, 0x1005);
                        assert_eq!(len as usize, form.len());
                        assert_eq!(&bytes[..len as usize], form);
                    }
                    exit => panic!("{mode:?}/{tier}: {exit:?}"),
                }
                assert_eq!(cpu.cpu.gpr, before.gpr);
                assert_eq!(cpu.cpu.flags, before.flags);
                assert_eq!(cpu.cpu.rflags_system, before.rflags_system);
                assert_eq!(cpu.cpu.fs_base, before.fs_base);
                assert_eq!(cpu.cpu.gs_base, before.gs_base);
                assert_eq!(
                    [cpu.cpu.cs, cpu.cpu.ss, cpu.cpu.ds, cpu.cpu.es, cpu.cpu.fs, cpu.cpu.gs],
                    [before.cs, before.ss, before.ds, before.es, before.fs, before.gs]
                );
            }
        }
    }
}

#[test]
fn iret_same_selectors_and_rejected_frames_all_tiers() {
    // SDM Vol 2A §3.2 IRET, RETURN-TO-SAME-PRIVILEGE-LEVEL; Vol 3A §7.14.3.
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for tier in 0..3 {
            for word in [false, true] {
                for rejected in [0, 1, 2, 3, 4] {
                    if mode == CpuMode::Compat32 && matches!(rejected, 2 | 3) {
                        continue;
                    }
                    let opcode = if mode == CpuMode::Long64 && !word {
                        vec![0x48, 0xcf]
                    } else if word {
                        vec![0x66, 0xcf]
                    } else {
                        vec![0xcf]
                    };
                    let vm = machine(&opcode, mode, tier);
                    vm.write_bytes(0x2000, &[0xf4]).unwrap();
                    let slot = if word {
                        2
                    } else if mode == CpuMode::Long64 {
                        8
                    } else {
                        4
                    };
                    let target = if rejected == 3 && !word {
                        0x0000_8000_0000_0000u64
                    } else {
                        0x2000
                    };
                    let frame = [
                        target,
                        if rejected == 1 { 0x23 } else { 0x33 },
                        0x25_0fd7,
                        0x6000,
                        if rejected == 2 { 0x3b } else { 0x2b },
                    ];
                    let count = if mode == CpuMode::Long64 { 5 } else { 3 };
                    for (i, value) in frame[..count].iter().enumerate() {
                        vm.write_bytes(0x7000 + (i * slot) as u64, &value.to_le_bytes()[..slot])
                            .unwrap();
                    }
                    let mut cpu = vm.new_vcpu();
                    cpu.cpu.flags.zf = true;
                    cpu.set_reg(Reg::Rip, 0x1000);
                    cpu.set_reg(Reg::Rsp, 0x7000);
                    cpu.cpu.cs = 0x33;
                    cpu.cpu.ss = 0x2b;
                    cpu.cpu.flags.if_ = true;
                    cpu.cpu.rflags_system = if rejected == 4 { 1 << 14 } else { 0 };
                    let before = cpu.cpu.clone();
                    let exit = cpu.run(&vm, None);
                    if rejected != 0 && !(rejected == 3 && word) {
                        assert!(
                            matches!(
                                exit,
                                Exit::Exception {
                                    addr: 0x1005,
                                    vector: 13,
                                    ..
                                }
                            ),
                            "{mode:?}/{tier}/{word}/{rejected}: {exit:?}"
                        );
                        assert_eq!(cpu.cpu.gpr, before.gpr);
                        assert_eq!(cpu.cpu.flags, before.flags);
                        assert_eq!(cpu.cpu.rflags_system, before.rflags_system);
                        assert_eq!((cpu.cpu.cs, cpu.cpu.ss), (before.cs, before.ss));
                    } else {
                        assert!(
                            matches!(exit, Exit::Hlt),
                            "{mode:?}/{tier}/{word}: {exit:?}"
                        );
                        assert_eq!(
                            cpu.reg(Reg::Rsp),
                            if count == 5 {
                                0x6000
                            } else {
                                0x7000 + (3 * slot) as u64
                            }
                        );
                        assert_eq!(
                            cpu.cpu.rflags() & 0x25_0fd7,
                            if word { 0xfd7 } else { 0x25_0fd7 }
                        );
                    }
                    if tier == 2 {
                        assert!(vm.cache.regions() > 0);
                    }
                }
            }
        }
    }
}

#[test]
fn flag_stack_faults_commit_nothing_all_tiers() {
    // SDM Vol 2B §4.3 PUSHF/POPF, Vol 2A §3.2 IRET, Exceptions:
    // a failed stack access exposes instruction-entry state.
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for tier in 0..3 {
            for form in [
                vec![0x9c],
                vec![0x66, 0x9c],
                vec![0x9d],
                vec![0x66, 0x9d],
                if mode == CpuMode::Long64 {
                    vec![0x48, 0xcf]
                } else {
                    vec![0xcf]
                },
            ] {
                let vm = machine(&form, mode, tier);
                let push = form.last() == Some(&0x9c);
                let slot = if form[0] == 0x66 {
                    2
                } else if mode == CpuMode::Long64 {
                    8
                } else {
                    4
                };
                let mut cpu = vm.new_vcpu();
                cpu.cpu.flags.zf = true;
                cpu.set_reg(Reg::Rip, 0x1000);
                cpu.set_reg(Reg::Rsp, 0x10000 + if push { slot } else { 0 });
                cpu.cpu.rflags_system = 0x24_0100;
                let before = cpu.cpu.clone();
                assert!(matches!(
                    cpu.run(&vm, None),
                    Exit::UnmappedMemory { addr: 0x10000, .. }
                ));
                assert_eq!(cpu.reg(Reg::Rip), 0x1005);
                assert_eq!(cpu.cpu.gpr, before.gpr);
                assert_eq!(cpu.cpu.flags, before.flags);
                assert_eq!(cpu.cpu.rflags_system, before.rflags_system);
                if tier == 2 {
                    assert!(vm.cache.regions() > 0);
                }
            }
            // All five IRETQ loads precede commits, including a late SS load fault.
            if mode == CpuMode::Long64 {
                let mut vm = machine(&[0x48, 0xcf], mode, tier);
                vm.map(0xf000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
                for (i, value) in [0x2000u64, 0x33, 0x24_0fd7, 0x6000].iter().enumerate() {
                    vm.write_bytes(0xffe0 + i as u64 * 8, &value.to_le_bytes())
                        .unwrap();
                }
                let mut cpu = vm.new_vcpu();
                cpu.cpu.flags.zf = true;
                cpu.cpu.cs = 0x33;
                cpu.cpu.ss = 0x2b;
                cpu.set_reg(Reg::Rip, 0x1000);
                cpu.set_reg(Reg::Rsp, 0xffe0);
                let before = cpu.cpu.clone();
                assert!(matches!(
                    cpu.run(&vm, None),
                    Exit::UnmappedMemory { addr: 0x10000, .. }
                ));
                assert_eq!(cpu.cpu.gpr, before.gpr);
                assert_eq!(cpu.cpu.flags, before.flags);
                assert_eq!(cpu.cpu.rflags_system, before.rflags_system);
                assert_eq!((cpu.cpu.cs, cpu.cpu.ss), (before.cs, before.ss));
            }
        }
    }
}

#[test]
fn word_pushf_and_wide_selector_all_tiers() {
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for tier in 0..3 {
            // SDM Vol 2B §4.3 PUSHF: 66H pushes two bytes in both modes.
            let mut body = vec![0x66, 0x9c, 0x66, 0x58];
            if mode == CpuMode::Long64 {
                body.extend([0x48, 0x8c, 0xcb]);
            }
            body.extend([0x39, 0xc9]);
            let vm = machine(&body, mode, tier);
            vm.write_bytes(0x6ff8, &[0xaa; 8]).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu.flags.set_flags16(0xed5);
            cpu.cpu.rflags_system = 0x24_0100;
            cpu.cpu.cs = 0x33;
            cpu.set_reg(Reg::Rip, 0x1000);
            cpu.set_reg(Reg::Rsp, 0x7000);
            cpu.set_reg(Reg::Rax, u64::MAX);
            cpu.set_reg(Reg::Rbx, u64::MAX);
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rax), 0xffff_ffff_ffff_0fd7);
            assert_eq!(cpu.reg(Reg::Rsp), 0x7000);
            if mode == CpuMode::Long64 {
                assert_eq!(cpu.reg(Reg::Rbx), 0x33);
            }
            let mut data = [0; 8];
            vm.read_bytes(0x6ff8, &mut data).unwrap();
            assert_eq!(data, [0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xd7, 0x0f]);
            if tier == 2 {
                assert!(vm.cache.regions() > 0);
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn flag_stack_native_witnesses() {
    use x86jit_tests::native::run_native;
    use x86jit_tests::oracle::VectorInput;
    use x86jit_tests::vector::{CpuSnapshot, ExitKind, MemChunk, MemKind, RunSpec, SnapFlags};

    let (cs, ss, ds, es, fs, gs): (u16, u16, u16, u16, u16, u16);
    let host_flags: u64;
    // SAFETY: read-only selector and RFLAGS observations; the stack is balanced.
    unsafe {
        core::arch::asm!("mov {cs:x}, cs", "mov {ss:x}, ss", "mov {ds:x}, ds",
            "mov {es:x}, es", "mov {fs:x}, fs", "mov {gs:x}, gs", "pushfq", "pop {flags}",
            cs = out(reg) cs, ss = out(reg) ss, ds = out(reg) ds, es = out(reg) es,
            fs = out(reg) fs, gs = out(reg) gs, flags = out(reg) host_flags,
            options(preserves_flags));
    }
    const CODE: u64 = 0x210000;
    const STACK: u64 = 0x220000;
    let mut cases = vec![
        // SDM Vol 2A §3.2 CLC/CMC, Vol 2B §4.3 STC/PUSHF: only CF changes.
        vec![0xf9, 0x9c, 0x58, 0xf5, 0x9c, 0x5b, 0xf8, 0x9c, 0x5a],
        // SDM Vol 2B §4.3 POPF Table 4-20: writable bits plus forbidden IOPL/IF.
        // AC is set only across aligned stack accesses, then cleared before exit.
        vec![0x56, 0x9d, 0x9c, 0x58, 0x6a, 0, 0x9d],
        vec![0x66, 0x56, 0x66, 0x9d, 0x66, 0x9c, 0x66, 0x58],
    ];
    for segment in 0..6 {
        // SDM Vol 2B §4.3 MOV: word, dword, qword and memory destinations.
        cases.push(vec![
            0x66,
            0x8c,
            0xc0 | segment << 3,
            0x8c,
            0xc3 | segment << 3,
            0x48,
            0x8c,
            0xc2 | segment << 3,
            0x8c,
            0x07 | segment << 3,
        ]);
    }
    // SDM Vol 2A §3.2 IRET; Vol 3A §7.14.3: same-CPL IRETQ restores SS:RSP.
    cases.push(vec![0x48, 0xcf]);
    for (case, body) in cases.iter().enumerate() {
        let mut code = vec![0xeb, 1, 0x90, 0x75, 0xfb];
        code.extend(body);
        // IRETQ's target skips the fallback terminator and reaches this address.
        let target = CODE + code.len() as u64;
        if case == cases.len() - 1 {
            code.extend([0x9c, 0x58]);
        }
        code.extend([0x39, 0xc9, 0xf4]);
        let mut stack = vec![0u8; 0x1000];
        if case == cases.len() - 1 {
            for (i, value) in [target, cs as u64, 0x200003, STACK + 0x800, ss as u64]
                .iter()
                .enumerate()
            {
                stack[0x800 + i * 8..0x808 + i * 8].copy_from_slice(&value.to_le_bytes());
            }
        }
        let mut initial = CpuSnapshot {
            flags: SnapFlags::from_rflags(0x42), // ZF makes the loop edge untaken.
            ..CpuSnapshot::default()
        };
        initial.gpr[0] = u64::MAX;
        initial.gpr[3] = u64::MAX;
        initial.gpr[4] = STACK + 0x800;
        initial.gpr[6] = 0x24_3003; // AC/ID/IOPL; keep TF and NT clear for native execution.
        initial.gpr[7] = STACK;
        let input = VectorInput {
            cpu_init: initial.clone(),
            mem_init: vec![
                MemChunk {
                    addr: CODE,
                    bytes: code.clone(),
                    kind: MemKind::Ram,
                },
                MemChunk {
                    addr: STACK,
                    bytes: stack.clone(),
                    kind: MemKind::Ram,
                },
            ],
            entry: CODE,
            run: RunSpec::UntilExit,
        };
        let native = run_native(&input).expect("native flag/selector witness must execute");
        assert!(matches!(native.exit, ExitKind::Hlt));
        for tier in 0..3 {
            let backend: Box<dyn Backend> = match tier {
                0 => Box::new(InterpreterBackend),
                1 => Box::new(JitBackend::new()),
                _ => Box::new(JitBackend::with_superblocks(RegionCaps {
                    max_blocks: 16,
                    max_icount: 256,
                })),
            };
            let mut vm = Vm::with_backend(VmConfig::flat(0x230000), backend);
            vm.map(CODE, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.map(STACK, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
            vm.write_bytes(CODE, &code).unwrap();
            vm.write_bytes(STACK, &stack).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu.gpr = initial.gpr;
            cpu.cpu.flags = initial.flags.into();
            cpu.cpu.flags.if_ = host_flags & (1 << 9) != 0;
            cpu.cpu.rflags_system = host_flags & 0x1a_3000;
            cpu.cpu.cs = cs;
            cpu.cpu.ss = ss;
            cpu.cpu.ds = ds;
            cpu.cpu.es = es;
            cpu.cpu.fs = fs;
            cpu.cpu.gs = gs;
            cpu.set_reg(Reg::Rip, CODE);
            assert!(
                matches!(cpu.run(&vm, None), Exit::Hlt),
                "case {case}, tier {tier}"
            );
            assert_eq!(cpu.cpu.gpr, native.cpu.gpr, "case {case}, tier {tier}");
            assert_eq!(SnapFlags::from(cpu.cpu.flags), native.cpu.flags);
            let mut actual = vec![0; 0x1000];
            vm.read_bytes(STACK, &mut actual).unwrap();
            assert_eq!(actual, native.mem[1].bytes, "case {case}, tier {tier}");
            if tier == 2 {
                assert!(vm.cache.regions() > 0);
            }
        }
    }
}

#[test]
fn syscall_saves_configured_rflags_all_tiers() {
    // SDM Vol 2B §4.3 SYSCALL, Operation; Vol 3B §20.3.1.1:
    // RF clears at instruction start, before R11 receives RFLAGS.
    let mut failures = Vec::new();
    for tier in 0..3 {
        for system in [0x24_0100, 0x25_0100] {
            let vm = machine(&[0x0f, 0x05], CpuMode::Long64, tier);
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, 0x1000);
            cpu.cpu.flags.set_flags16(0xed5);
            cpu.cpu.rflags_system = system;
            assert!(matches!(cpu.run(&vm, None), Exit::Syscall));
            if cpu.reg(Reg::R11) != 0x24_0fd7 {
                failures.push(format!(
                    "tier {tier}, system {system:#x}: R11={:#x}",
                    cpu.reg(Reg::R11)
                ));
            }
            if tier == 2 {
                assert!(vm.cache.regions() > 0);
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

#[test]
fn iretq_to_syscall_clears_saved_rf_all_tiers() {
    // SDM Vol 3B §20.3.1.1: IRETQ restores RF, then SYSCALL clears it
    // before saving RFLAGS (Vol 2B §4.3 SYSCALL, Operation).
    let mut failures = Vec::new();
    for tier in 0..3 {
        let vm = machine(&[0x48, 0xcf], CpuMode::Long64, tier);
        vm.write_bytes(0x2000, &[0x0f, 0x05]).unwrap();
        for (i, value) in [0x2000u64, 0x33, 0x1_0002, 0x6000, 0x2b].iter().enumerate() {
            vm.write_bytes(0x7000 + i as u64 * 8, &value.to_le_bytes())
                .unwrap();
        }
        let mut cpu = vm.new_vcpu();
        cpu.cpu.flags.zf = true;
        cpu.cpu.cs = 0x33;
        cpu.cpu.ss = 0x2b;
        cpu.set_reg(Reg::Rip, 0x1000);
        cpu.set_reg(Reg::Rsp, 0x7000);
        assert!(matches!(cpu.run(&vm, None), Exit::Syscall));
        if cpu.reg(Reg::R11) != 2 {
            failures.push(format!("tier {tier}: R11={:#x}", cpu.reg(Reg::R11)));
        }
        if tier == 2 {
            assert!(vm.cache.regions() > 0);
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

#[test]
fn iret_nt_precedes_inaccessible_stack_all_tiers() {
    // SDM Vol 2A §3.2 IRET, Operation, IA-32e-MODE: NT raises #GP(0)
    // before the first pop, including in compatibility mode.
    let mut failures = Vec::new();
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for tier in 0..3 {
            for word in [false, true] {
                let form = if word {
                    vec![0x66, 0xcf]
                } else if mode == CpuMode::Long64 {
                    vec![0x48, 0xcf]
                } else {
                    vec![0xcf]
                };
                let vm = machine(&form, mode, tier);
                let mut cpu = vm.new_vcpu();
                cpu.cpu.flags.zf = true;
                cpu.cpu.rflags_system = 1 << 14;
                cpu.set_reg(Reg::Rip, 0x1000);
                cpu.set_reg(Reg::Rsp, 0x10000);
                let before = cpu.cpu.clone();
                let exit = cpu.run(&vm, None);
                if !matches!(
                    exit,
                    Exit::Exception {
                        addr: 0x1005,
                        vector: 13,
                        ..
                    }
                ) {
                    failures.push(format!("{mode:?}/tier {tier}/word {word}: {exit:?}"));
                }
                assert_eq!(cpu.cpu.gpr, before.gpr);
                assert_eq!(cpu.cpu.flags, before.flags);
                assert_eq!(cpu.cpu.rflags_system, before.rflags_system);
                assert_eq!((cpu.cpu.cs, cpu.cpu.ss), (before.cs, before.ss));
                assert_eq!(cpu.reg(Reg::Rip), 0x1005);
                if tier == 2 {
                    assert!(vm.cache.regions() > 0);
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

#[test]
fn flag_stack_noncanonical_address_is_ss_fault_all_tiers() {
    // SDM Vol 1 §3.3.7.1; Vol 2B §4.3 PUSHF/POPF and Vol 2A §3.2 IRET,
    // 64-bit mode exceptions: a noncanonical stack reference raises #SS(0).
    for tier in 0..3 {
        for form in [
            vec![0x9c],
            vec![0x66, 0x9c],
            vec![0x9d],
            vec![0x66, 0x9d],
            vec![0x48, 0xcf],
        ] {
            let vm = machine(&form, CpuMode::Long64, tier);
            let mut cpu = vm.new_vcpu();
            cpu.cpu.flags.zf = true;
            cpu.set_reg(Reg::Rip, 0x1000);
            let delta = if form.last() == Some(&0x9c) {
                if form[0] == 0x66 {
                    2
                } else {
                    8
                }
            } else {
                0
            };
            cpu.set_reg(Reg::Rsp, 0x0000_8000_0000_0000 + delta);
            let before = cpu.cpu.clone();
            let exit = cpu.run(&vm, None);
            assert!(
                matches!(
                    exit,
                    Exit::Exception {
                        addr: 0x1005,
                        vector: 12,
                        ..
                    }
                ),
                "{tier}: {exit:?}"
            );
            assert_eq!(cpu.cpu.gpr, before.gpr);
            assert_eq!(cpu.cpu.flags, before.flags);
            assert_eq!(cpu.cpu.rflags_system, before.rflags_system);
        }
    }
}

#[test]
fn flag_stack_jit_and_region_match_interpreter() {
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        let mut forms = vec![
            vec![0xf8],
            vec![0xf9],
            vec![0xf5],
            vec![0x9c],
            vec![0x66, 0x9c],
            vec![0x9d],
            vec![0x66, 0x9d],
            vec![0x8c, 0xc8],
            vec![0x66, 0x8c, 0xc8],
            vec![0x8c, 0x0f],
            vec![0x8e, 0x1f],
            vec![0x0f, 0xa1],
            vec![0x66, 0xcf],
        ];
        forms.push(if mode == CpuMode::Long64 {
            vec![0x48, 0xcf]
        } else {
            vec![0xcf]
        });
        if mode == CpuMode::Long64 {
            forms.extend([vec![0x48, 0x8c, 0xc8], vec![0x48, 0x8c, 0x0f]]);
        }
        for form in forms {
            let mut reference = None;
            for tier in 0..3 {
                let vm = machine(&form, mode, tier);
                vm.write_bytes(0x2000, &[0xf4]).unwrap();
                if form.last() == Some(&0xcf) {
                    let slot = if form[0] == 0x66 {
                        2
                    } else if mode == CpuMode::Long64 {
                        8
                    } else {
                        4
                    };
                    let count = if mode == CpuMode::Long64 { 5 } else { 3 };
                    for (i, value) in [0x2000u64, 0x33, 0x24_0fd7, 0x6000, 0x2b][..count]
                        .iter()
                        .enumerate()
                    {
                        vm.write_bytes(0x7000 + (i * slot) as u64, &value.to_le_bytes()[..slot])
                            .unwrap();
                    }
                } else {
                    vm.write_bytes(0x7000, &0x24_0fd7u64.to_le_bytes()).unwrap();
                }
                let mut cpu = vm.new_vcpu();
                cpu.cpu.flags.set_flags16(0xed5);
                cpu.cpu.rflags_system = 0x20_0100;
                cpu.cpu.cs = 0x33;
                cpu.cpu.ss = 0x2b;
                cpu.set_reg(Reg::Rip, 0x1000);
                cpu.set_reg(Reg::Rsp, 0x7000);
                cpu.set_reg(Reg::Rax, u64::MAX);
                cpu.set_reg(Reg::Rdi, 0x6000);
                let exit = cpu.run(&vm, None);
                let delegates = form[0] == 0x8e || form == [0x0f, 0xa1];
                assert!(
                    if delegates {
                        matches!(exit, Exit::UnknownInstruction { .. })
                    } else {
                        matches!(exit, Exit::Hlt)
                    },
                    "{mode:?}/{form:02x?}: {exit:?}"
                );
                let mut memory = vec![0; 0x1200];
                vm.read_bytes(0x5f00, &mut memory).unwrap();
                let state = (
                    cpu.cpu.gpr,
                    cpu.cpu.rip,
                    cpu.cpu.flags,
                    cpu.cpu.rflags_system,
                    [
                        cpu.cpu.cs, cpu.cpu.ss, cpu.cpu.ds, cpu.cpu.es, cpu.cpu.fs, cpu.cpu.gs,
                    ],
                    cpu.cpu.fs_base,
                    cpu.cpu.gs_base,
                    format!("{exit:?}"),
                    memory,
                );
                if let Some(reference) = &reference {
                    assert_eq!(&state, reference, "{mode:?}/{form:02x?}/tier {tier}");
                } else {
                    reference = Some(state);
                }
                if tier == 2 {
                    assert!(vm.cache.regions() > 0);
                }
            }
        }
    }
}
