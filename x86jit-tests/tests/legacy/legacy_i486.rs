//! Intel486 PRM (1992) §4.1.1 p. 4-2, §4.1.3 pp. 4-6–7,
//! §10.1 pp. 10-1–3, §10.7.2 pp. 10-13–17, §26 instruction entries;
//! Embedded Intel486 DM 273021-001 App. D pp. D-1–3. Scan images checked.
#[path = "../support/descriptor_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use x86jit_core::{
    CpuModel, CpuState, Exit, FpuModel, Prot, RegionKind, SegmentCache, StepOptions, StepStop,
    UnsupportedSystemOperation as U, Vcpu, Vm, VmConfig,
};

const MODELS: [CpuModel; 2] = [CpuModel::I80486Dx, CpuModel::I80486DxCpuid];
#[derive(Clone, Copy, Debug)]
enum Mode {
    Real,
    P16,
    P32,
}
const MODES: [Mode; 3] = [Mode::Real, Mode::P16, Mode::P32];
fn setup(model: CpuModel, mode: Mode, code: &[u8]) -> (Vm, Vcpu) {
    let (vm, mut cpu) = fixture::setup(model, code);
    let protected = !matches!(mode, Mode::Real);
    let wide = matches!(mode, Mode::P32);
    cpu.cpu.legacy.cr0 = 0x10 | u32::from(protected);
    cpu.cpu.cs = if protected { 8 } else { 0 };
    let data = if protected { 16 } else { 0 };
    cpu.cpu.ds = data;
    cpu.cpu.ss = data;
    cpu.cpu.es = data;
    cpu.cpu.fs = data;
    cpu.cpu.gs = data;
    for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
        *cache = SegmentCache {
            base: 0,
            limit: if protected { u32::MAX } else { 0xffff },
            attributes: if i == 0 { 0x9b } else { 0x93 } | if wide { 0xc000 } else { 0 },
            ..Default::default()
        };
    }
    (vm, cpu)
}
fn instruction(mode: Mode, size: u8, opcode: &[u8]) -> Vec<u8> {
    let mut b = vec![];
    if size != 1 && (size == 4) != matches!(mode, Mode::P32) {
        b.push(0x66);
    }
    b.extend(opcode);
    b
}
fn memory(mode: Mode, size: u8, op: u8, lock: bool) -> Vec<u8> {
    let mut b = instruction(mode, size, &[0x0f, op]);
    if lock {
        b.insert(0, 0xf0);
    }
    if matches!(mode, Mode::P32) {
        b.push(0x0d);
        b.extend(0x6000u32.to_le_bytes());
    } else {
        b.push(0x0e);
        b.extend(0x6000u16.to_le_bytes());
    }
    b
}
fn refused(cpu: &mut Vcpu, vm: &Vm, expected: U) {
    let before = format!("{:?}", cpu.cpu);
    let run = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0);
    assert!(
        matches!(run.stop(), StepStop::Exit(Exit::UnsupportedSystemContext { operation, .. }) if *operation == expected),
        "{:?}",
        run.stop()
    );
    assert_eq!(format!("{:?}", cpu.cpu), before);
}
fn ring(cpu: &mut Vcpu, cpl: u8) {
    cpu.cpu.legacy.cpl = cpl;
    cpu.cpu.cs = 8 | u16::from(cpl);
    cpu.cpu.ss = 16 | u16::from(cpl);
    for cache in &mut cpu.cpu.legacy.segments {
        cache.attributes = (cache.attributes & !0x60) | u16::from(cpl) << 5;
    }
}

#[test]
fn reset_images_and_catalogue() {
    // PRM §10.1 Figs. 10-1/2, Table 10-1 pp. 10-1–3. ASSUMED stepping
    // 03h early DX, 0h model-1 CPUID DX (DM Table D-2 p. D-3).
    for (model, signature) in [(MODELS[0], 0x403), (MODELS[1], 0x410)] {
        let cpu = CpuState::i486_reset(model);
        assert_eq!(cpu.gpr[2], signature);
        assert_eq!(cpu.rip, 0xfff0);
        assert_eq!(cpu.rflags(), 2);
        assert_eq!(cpu.legacy.cr0, 0x6000_0010);
        assert_eq!(
            [cpu.cs, cpu.ds, cpu.es, cpu.ss, cpu.fs, cpu.gs],
            [0xf000, 0, 0, 0, 0, 0]
        );
        assert_eq!((cpu.legacy.idtr.base, cpu.legacy.idtr.limit), (0, 0x3ff));
        assert_eq!(cpu.legacy.reset_fetch, 1);
        assert_eq!(cpu.debug_test_registers[4..6], [0xffff_0ff0, 0]);
        assert_eq!(cpu.fpu_model, FpuModel::Present);
        for (i, cache) in cpu.legacy.segments.iter().enumerate() {
            assert_eq!(cache.base, if i == 0 { 0xffff_0000 } else { 0 });
            assert_eq!(cache.limit, 0xffff);
        }
        let (vm, cpu) = setup(model, Mode::Real, &[0x90]);
        assert_eq!(vm.fpu_model(), FpuModel::Present);
        assert_eq!(cpu.cpu.fpu_model, FpuModel::Present);
        assert_eq!(model.profile().fpu, &[FpuModel::Present]);
    }
    assert_ne!(
        MODELS[0].profile().semantics_id,
        MODELS[1].profile().semantics_id
    );
    assert_ne!(
        MODELS[0].profile().semantics_id,
        CpuModel::I80386.profile().semantics_id
    );
}

#[test]
fn ordinary_construction_reads_fixed_et_before_any_cr0_write() {
    // Intel486 PRM §4.1.3 p. 4-7; SDM Vol 3B §25.18.1 p. 25-8:
    // ET is hardwired to one. Execution reset preserves architectural state.
    for model in [CpuModel::I80386, MODELS[0], MODELS[1]] {
        let mut vm = Vm::new(VmConfig::flat(0x2000));
        // Model selection can replace a previous selection before construction.
        vm.set_cpu_model(MODELS[1]);
        vm.set_cpu_model(model);
        vm.map(0, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, &[0x0f, 0x20, 0xc0]).unwrap(); // MOV EAX,CR0
        let expected = if model == CpuModel::I80386 { 0 } else { 0x10 };
        let mut cpu = vm.new_vcpu();
        assert_eq!(cpu.cpu.legacy.cr0, expected);
        for reset in [false, true] {
            if reset {
                cpu.reset_execution_state(Default::default());
            }
            cpu.cpu.rip = 0x1000;
            fixture::steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.gpr[0], u64::from(expected));
            assert_eq!(cpu.cpu.legacy.cr0, expected);
        }
    }
}

#[test]
fn id_and_ac_popfd_pushfd_and_popf_preservation() {
    // PRM §4.1.1 p. 4-2 / §26 POPFD p. 26-235; DM §D.1 p. D-1.
    for model in [CpuModel::I80386, MODELS[0], MODELS[1]] {
        let expected = match model {
            CpuModel::I80386 => 0,
            CpuModel::I80486Dx => 1 << 18,
            _ => (1 << 18) | (1 << 21),
        };
        for mode in MODES {
            for cpl in [0, 3] {
                if matches!(mode, Mode::Real) && cpl != 0 {
                    continue;
                }
                let mut code = instruction(mode, 4, &[0x9d]);
                code.extend(instruction(mode, 4, &[0x9c]));
                // A word POPF must preserve the high EFLAGS bits.
                code.extend(instruction(mode, 2, &[0x9d]));
                let (vm, mut cpu) = setup(model, mode, &code);
                if cpl != 0 {
                    ring(&mut cpu, cpl);
                }
                vm.write_bytes(0x8000, &0x0024_0002u32.to_le_bytes())
                    .unwrap();
                fixture::steps(&mut cpu, &vm, 2);
                assert_eq!(
                    fixture::value(&vm, 0x8000, 4),
                    expected | 2,
                    "{model:?} {mode:?}"
                );
                assert_eq!(cpu.cpu.rflags() & 0x240000, expected);
                vm.write_bytes(0x8000, &2u16.to_le_bytes()).unwrap();
                fixture::steps(&mut cpu, &vm, 1);
                assert_eq!(cpu.cpu.rflags() & 0x240000, expected);
                cpu.cpu.rip = 0x1000;
                cpu.cpu.gpr[4] = 0x8000;
                vm.write_bytes(0x8000, &2u32.to_le_bytes()).unwrap();
                fixture::steps(&mut cpu, &vm, 2);
                assert_eq!(cpu.cpu.rflags() & 0x240000, 0);
            }
        }
    }
}

#[test]
fn cpuid_identity_leaves_ignore_subleaf_and_flags() {
    // DM §§D.1–D.4 pp. D-1–2: availability is independent of live ID.
    // ASSUMED: FPU-only feature mask, reserved bits zero (decision-23 amendment).
    for mode in MODES {
        for cpl in [0, 3] {
            if matches!(mode, Mode::Real) && cpl != 0 {
                continue;
            }
            for (leaf, values) in [
                (0, [1, 0x756e6547, 0x6c65746e, 0x49656e69]),
                (1, [0x410, 0, 0, 1]),
                (2, [0; 4]),
                (u32::MAX, [0; 4]),
            ] {
                let (vm, mut cpu) = setup(MODELS[1], mode, &[0x0f, 0xa2]);
                if cpl != 0 {
                    ring(&mut cpu, cpl);
                }
                cpu.cpu.gpr[0] = u64::from(leaf);
                cpu.cpu.gpr[1] = 0xdeadbeef;
                cpu.cpu.flags.cf = true;
                cpu.cpu.flags.of = true;
                let flags = cpu.cpu.rflags();
                fixture::steps(&mut cpu, &vm, 1);
                assert_eq!(
                    [
                        cpu.cpu.gpr[0],
                        cpu.cpu.gpr[3],
                        cpu.cpu.gpr[1],
                        cpu.cpu.gpr[2]
                    ],
                    values
                );
                assert_eq!(cpu.cpu.rflags(), flags);
            }
            for model in [CpuModel::I80386, MODELS[0]] {
                let (vm, mut cpu) = setup(model, mode, &[0x0f, 0xa2]);
                if cpl != 0 {
                    ring(&mut cpu, cpl);
                }
                fixture::fault(&mut cpu, &vm, 6, None);
            }
        }
    }
}

#[test]
fn bswap_and_undefined_word_form() {
    // PRM §26 BSWAP p. 26-35: 32-bit only, flags unchanged; word undefined.
    for model in MODELS {
        for mode in MODES {
            let (vm, mut cpu) = setup(model, mode, &instruction(mode, 4, &[0x0f, 0xcb]));
            cpu.cpu.gpr[3] = 0x12345678;
            cpu.cpu.flags.cf = true;
            cpu.cpu.flags.of = true;
            let flags = cpu.cpu.rflags();
            fixture::steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.gpr[3], 0x78563412);
            assert_eq!(cpu.cpu.rflags(), flags);
            let (vm, mut cpu) = setup(model, mode, &instruction(mode, 2, &[0x0f, 0xcb]));
            refused(&mut cpu, &vm, U::LegacyInstruction);
        }
    }
}

#[test]
fn xadd_widths_aliases_flags_and_locked_memory() {
    // PRM §26 XADD pp. 26-282–283 (LOCK); flags as ADD.
    for model in MODELS {
        for mode in MODES {
            for size in [1, 2, 4] {
                let op = if size == 1 { 0xc0 } else { 0xc1 };
                let mask = (1u64 << (size * 8)) - 1;
                for mem in [false, true] {
                    for lock in [false, true] {
                        if lock && !mem {
                            continue;
                        }
                        let code = if mem {
                            memory(mode, size, op, lock)
                        } else {
                            instruction(mode, size, &[0x0f, op, 0xcb])
                        };
                        let (vm, mut cpu) = setup(model, mode, &code);
                        cpu.cpu.gpr[1] = 1;
                        cpu.cpu.gpr[3] = mask;
                        vm.write_bytes(0x6000, &mask.to_le_bytes()[..usize::from(size)])
                            .unwrap();
                        fixture::steps(&mut cpu, &vm, 1);
                        assert_eq!(cpu.cpu.gpr[1], mask);
                        assert_eq!(
                            if mem {
                                fixture::value(&vm, 0x6000, usize::from(size))
                            } else {
                                cpu.cpu.gpr[3]
                            },
                            0
                        );
                        assert_eq!(cpu.cpu.rflags() & 0x8d5, 0x55);
                    }
                }
                let (vm, mut cpu) = setup(model, mode, &instruction(mode, size, &[0x0f, op, 0xc9]));
                cpu.cpu.gpr[1] = 7;
                fixture::steps(&mut cpu, &vm, 1);
                assert_eq!(cpu.cpu.gpr[1], 14);
            }
        }
    }
}

#[test]
fn cmpxchg_widths_success_failure_flags_and_lock() {
    // PRM §26 CMPXCHG pp. 26-62–63: compare accumulator, source on success;
    // on failure accumulator receives old destination; LOCK permits atomic use.
    for model in MODELS {
        for mode in MODES {
            for size in [1, 2, 4] {
                let op = if size == 1 { 0xb0 } else { 0xb1 };
                for mem in [false, true] {
                    for lock in [false, true] {
                        for equal in [false, true] {
                            if lock && !mem {
                                continue;
                            }
                            let code = if mem {
                                memory(mode, size, op, lock)
                            } else {
                                instruction(mode, size, &[0x0f, op, 0xcb])
                            };
                            let (vm, mut cpu) = setup(model, mode, &code);
                            cpu.cpu.gpr[0] = if equal { 7 } else { 6 };
                            cpu.cpu.gpr[1] = 9;
                            cpu.cpu.gpr[3] = 7;
                            vm.write_bytes(0x6000, &7u32.to_le_bytes()).unwrap();
                            fixture::steps(&mut cpu, &vm, 1);
                            assert_eq!(cpu.cpu.gpr[0], 7);
                            assert_eq!(
                                if mem {
                                    fixture::value(&vm, 0x6000, usize::from(size))
                                } else {
                                    cpu.cpu.gpr[3]
                                },
                                if equal { 9 } else { 7 }
                            );
                            assert_eq!(cpu.cpu.rflags() & 0x8d5, if equal { 0x44 } else { 0x95 });
                            assert_eq!(cpu.cpu.gpr[1], 9);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn cache_tlb_invalidation_privilege_and_register_ud() {
    // PRM §26 INVD/INVLPG/WBINVD pp. 26-171–172, 26-281.
    for model in MODELS {
        for mode in MODES {
            for code in [&[0x0f, 0x08][..], &[0x0f, 0x09], &[0x0f, 0x01, 0x38]] {
                let (vm, mut cpu) = setup(model, mode, code);
                cpu.cpu.flags.cf = true;
                let flags = cpu.cpu.rflags();
                fixture::steps(&mut cpu, &vm, 1);
                assert_eq!(cpu.cpu.rflags(), flags);
                if !matches!(mode, Mode::Real) {
                    for cpl in 1..=3 {
                        let (vm, mut cpu) = setup(model, mode, code);
                        ring(&mut cpu, cpl);
                        fixture::fault(&mut cpu, &vm, 13, Some(0));
                    }
                }
            }
            let (vm, mut cpu) = setup(model, mode, &[0x0f, 0x01, 0xff]);
            fixture::fault(&mut cpu, &vm, 6, None);
            let (vm, mut cpu) = setup(model, mode, &[0x0f, 0x08]);
            cpu.cpu.legacy.cr0 |= 0x80000001;
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 0);
            let StepStop::Exit(exit @ Exit::Exception { vector: 14, .. }) = run.stop() else {
                panic!("{run:?}");
            };
            assert_eq!(
                exit.fault_metadata().unwrap().error_code(),
                x86jit_core::ErrorCode::bits(0, 7)
            );
        }
    }
}

#[test]
fn control_bits_and_cache_test_refusals() {
    // PRM §4.1.3 pp. 4-6–7: CR0.CD/NW/AM/WP/NE/ET, CR3.PCD/PWT.
    for model in MODELS {
        for mode in MODES {
            let (vm, mut cpu) = setup(model, mode, &[0x0f, 0x22, 0xc0, 0x0f, 0x20, 0xc3]);
            let value = 0x60050020 | u32::from(!matches!(mode, Mode::Real));
            cpu.cpu.gpr[0] = u64::from(value);
            fixture::steps(&mut cpu, &vm, 2);
            assert_eq!(cpu.cpu.legacy.cr0, value | 0x10);
            assert_eq!(cpu.cpu.gpr[3], u64::from(value | 0x10));
            let (vm, mut cpu) = setup(model, mode, &[0x0f, 0x22, 0xd8, 0x0f, 0x20, 0xdb]);
            cpu.cpu.gpr[0] = 0x12345fff;
            fixture::steps(&mut cpu, &vm, 2);
            assert_eq!(cpu.cpu.gpr[3], 0x12345018);
            // PRM §12.2.1 Table 12-1 p. 12-3: invalid NW/CD combination.
            let (vm, mut cpu) = setup(model, mode, &[0x0f, 0x22, 0xc0]);
            cpu.cpu.gpr[0] = 0x2000_0010 | u64::from(!matches!(mode, Mode::Real));
            fixture::fault(
                &mut cpu,
                &vm,
                13,
                if matches!(mode, Mode::Real) {
                    None
                } else {
                    Some(0)
                },
            );
            // PRM §10.7.2 pp. 10-13–17: no cache test model (ASSUMED refusal).
            for tr in 3..=5 {
                for op in [0x24, 0x26] {
                    let (vm, mut cpu) = setup(model, mode, &[0x0f, op, 0xc0 | tr << 3]);
                    refused(&mut cpu, &vm, U::LegacyCacheTest);
                    if !matches!(mode, Mode::Real) {
                        let (vm, mut cpu) = setup(model, mode, &[0x0f, op, 0xc0 | tr << 3]);
                        ring(&mut cpu, 3);
                        fixture::fault(&mut cpu, &vm, 13, Some(0));
                    }
                }
            }
        }
    }
}

#[test]
fn cr0_cache_validity_precedes_paging_enablement() {
    // Intel486 PRM §12.2.1 Table 12-1 p. 12-3: NW=1/CD=0 is #GP(0),
    // including when PG and PE are set. Valid writes enable paging.
    for model in [CpuModel::I80386, MODELS[0], MODELS[1]] {
        for mode in MODES {
            for value in [0xa000_0011, 0x8000_0011, 0xe000_0011] {
                let (vm, mut cpu) = setup(model, mode, &[0x0f, 0x22, 0xc0]);
                cpu.cpu.gpr[0] = value;
                if model != CpuModel::I80386 && value == 0xa000_0011 {
                    fixture::fault(
                        &mut cpu,
                        &vm,
                        13,
                        if matches!(mode, Mode::Real) {
                            None
                        } else {
                            Some(0)
                        },
                    );
                } else {
                    fixture::steps(&mut cpu, &vm, 1);
                    assert_ne!(cpu.cpu.legacy.cr0 & 0x8000_0000, 0);
                }
            }
        }
    }
}

#[test]
fn alignment_regime_is_refused_before_execution() {
    // PRM §4.1.1 p. 4-2. ASSUMED conservative refusal of the whole regime.
    for model in MODELS {
        for mode in [Mode::P16, Mode::P32] {
            for (am, ac, cpl) in [
                (true, true, 3),
                (false, true, 3),
                (true, false, 3),
                (true, true, 0),
            ] {
                let (vm, mut cpu) = setup(model, mode, &[0x90]);
                ring(&mut cpu, cpl);
                if am {
                    cpu.cpu.legacy.cr0 |= 1 << 18;
                }
                if ac {
                    cpu.cpu.rflags_system |= 1 << 18;
                }
                if am && ac && cpl == 3 {
                    refused(&mut cpu, &vm, U::LegacyAlignmentCheck);
                } else {
                    fixture::steps(&mut cpu, &vm, 1);
                }
            }
        }
    }
}

#[test]
fn alignment_regime_entered_by_popfd_refuses_the_next_instruction() {
    // PRM §4.1.1 p. 4-2: the regime is AM, AC and CPL 3; POPFD at CPL 3
    // writes AC (§26 POPF/POPFD). Every instruction before the POPFD runs
    // its gate outside the regime; the NOP after it is refused, whole.
    for model in MODELS {
        for mode in [Mode::P16, Mode::P32] {
            let mut code = vec![0x40]; // INC (E)AX
            code.extend(instruction(mode, 4, &[0x9c])); // PUSHFD
            code.extend(instruction(mode, 4, &[0x58])); // POP EAX
            let mut or = vec![0x0d]; // OR EAX, AC
            or.extend((1u32 << 18).to_le_bytes());
            code.extend(instruction(mode, 4, &or));
            code.extend(instruction(mode, 4, &[0x50])); // PUSH EAX
            code.extend(instruction(mode, 4, &[0x9d])); // POPFD
            let nop = 0x1000 + code.len() as u64;
            code.push(0x90);
            let (vm, mut cpu) = setup(model, mode, &code);
            ring(&mut cpu, 3);
            cpu.cpu.legacy.cr0 |= 1 << 18;
            let run = cpu.run_steps(&vm, 16, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 6, "{mode:?}: {:?}", run.stop());
            assert!(
                matches!(
                    run.stop(),
                    StepStop::Exit(Exit::UnsupportedSystemContext {
                        operation: U::LegacyAlignmentCheck,
                        ..
                    })
                ),
                "{:?}",
                run.stop()
            );
            assert_eq!(cpu.cpu.rip, nop);
            assert_ne!(cpu.cpu.rflags_system & (1 << 18), 0);
            assert_ne!(cpu.cpu.gpr[0] & (1 << 18), 0);
        }
    }
}

#[test]
fn decode_boundaries_preserve_386_and_reject_later_encodings() {
    // PRM §3.11 p. 3-43, App. A pp. A-4–7; DM App. D pp. D-1–3.
    for mode in MODES {
        for code in [
            instruction(mode, 4, &[0x0f, 0xc8]),
            memory(mode, 4, 0xb1, true),
            memory(mode, 4, 0xc1, true),
            vec![0x0f, 0x08],
            vec![0x0f, 0x09],
            vec![0x0f, 0x01, 0x38],
        ] {
            let (vm, mut cpu) = setup(CpuModel::I80386, mode, &code);
            fixture::fault(&mut cpu, &vm, 6, None);
        }
        // Post-486: TSC, MSRs, CMPXCHG8B, CMOV, MMX, SSE, CR4 and SYSENTER.
        for model in MODELS {
            for code in [
                &[0x0f, 0x31][..],
                &[0x0f, 0x32],
                &[0x0f, 0x30],
                &[0x0f, 0xc7, 0x08],
                &[0x0f, 0x44, 0xc3],
                &[0x0f, 0x6f, 0xc0],
                &[0x0f, 0x10, 0xc0],
                &[0x0f, 0x20, 0xe0],
                &[0x0f, 0x34],
            ] {
                let (vm, mut cpu) = setup(model, mode, code);
                fixture::fault(&mut cpu, &vm, 6, None);
            }
        }
        for model in MODELS {
            for op in [0xb1, 0xc1] {
                let (vm, mut cpu) =
                    setup(model, mode, &instruction(mode, 4, &[0xf0, 0x0f, op, 0xcb]));
                fixture::fault(&mut cpu, &vm, 6, None);
            }
        }
    }
}

#[test]
fn cmpxchg_device_write_cycle_and_retry_commit() {
    // PRM §26 CMPXCHG p. 26-63: even a failed comparison writes DEST back.
    for model in MODELS {
        for mode in MODES {
            for equal in [false, true] {
                let code = memory(mode, 4, 0xb1, true);
                let (mut vm, mut cpu) = setup(model, mode, &code);
                vm.unmap(0, 0x240000).unwrap();
                vm.map(
                    0,
                    0x6000,
                    x86jit_core::Prot::RWX,
                    x86jit_core::RegionKind::Ram,
                )
                .unwrap();
                vm.map(
                    0x7000,
                    0x239000,
                    x86jit_core::Prot::RWX,
                    x86jit_core::RegionKind::Ram,
                )
                .unwrap();
                vm.write_bytes(0x1000, &code).unwrap();
                vm.map(
                    0x6000,
                    0x1000,
                    x86jit_core::Prot::RW,
                    x86jit_core::RegionKind::Trap,
                )
                .unwrap();
                cpu.cpu.gpr[0] = if equal { 7 } else { 6 };
                cpu.cpu.gpr[1] = 9;
                let before = (cpu.cpu.gpr, cpu.cpu.rflags());
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 0);
                assert!(
                    matches!(
                        run.stop(),
                        StepStop::Exit(Exit::MmioRead {
                            addr: 0x6000,
                            size: 4,
                            ..
                        })
                    ),
                    "{:?}",
                    run.stop()
                );
                assert_eq!((cpu.cpu.gpr, cpu.cpu.rflags()), before);
                cpu.complete_mmio_read(7);
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 0);
                let expected = if equal { 9 } else { 7 };
                assert!(
                    matches!(run.stop(), StepStop::Exit(Exit::MmioWrite { addr: 0x6000, size: 4, value, .. }) if *value == expected),
                    "{:?}",
                    run.stop()
                );
                assert_eq!((cpu.cpu.gpr, cpu.cpu.rflags()), before);
                cpu.complete_mmio_write();
                fixture::steps(&mut cpu, &vm, 1);
                assert_eq!(cpu.cpu.gpr[0], 7);
                assert_eq!(cpu.cpu.rflags() & 0x8d5, if equal { 0x44 } else { 0x95 });
            }
        }
    }
}

#[test]
fn locked_cmpxchg_and_xadd_classify_device_boundaries() {
    // Intel486 PRM §26 CMPXCHG pp. 26-62–63, XADD pp. 26-282–283:
    // failed compares still write; LOCK covers the entire update. The engine
    // hands two-device operands to the embedder (spec.md §5.2); one-device
    // splits use the shared RMW retry/commit policy.
    for model in MODELS {
        for mode in MODES {
            for (low, high) in [
                (RegionKind::Trap, RegionKind::Trap),
                (RegionKind::Ram, RegionKind::Trap),
                (RegionKind::Trap, RegionKind::Ram),
            ] {
                for (op, equal) in [(0xb1, false), (0xb1, true), (0xc1, true)] {
                    let mut code = memory(mode, 4, op, true);
                    let displacement = if matches!(mode, Mode::P32) { 4 } else { 2 };
                    let start = code.len() - displacement;
                    code[start..].copy_from_slice(&0x6ffeu32.to_le_bytes()[..displacement]);
                    let (mut vm, mut cpu) = setup(model, mode, &code);
                    vm.unmap(0, 0x240000).unwrap();
                    vm.map(0, 0x6000, Prot::RWX, RegionKind::Ram).unwrap();
                    vm.map(0x6000, 0x1000, Prot::RW, low).unwrap();
                    vm.map(0x7000, 0x1000, Prot::RW, high).unwrap();
                    vm.write_bytes(0x1000, &code).unwrap();
                    let old = 0x1122_3344u32;
                    let src = 0x5566_7788u32;
                    cpu.cpu.gpr[0] = u64::from(old - u32::from(!equal));
                    cpu.cpu.gpr[1] = u64::from(src);
                    for (addr, kind, shift) in [(0x6ffe, low, 0), (0x7000, high, 16)] {
                        if kind == RegionKind::Ram {
                            vm.write_bytes(addr, &((old >> shift) as u16).to_le_bytes())
                                .unwrap();
                        }
                    }
                    let before = (cpu.cpu.gpr, cpu.cpu.rip, cpu.cpu.rflags());
                    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                    assert_eq!(run.steps(), 0);
                    if low == RegionKind::Trap && high == RegionKind::Trap {
                        let StepStop::Exit(Exit::UnknownInstruction {
                            addr, bytes, len, ..
                        }) = run.stop()
                        else {
                            panic!("{:?}", run.stop());
                        };
                        assert_eq!(*addr, 0x1000);
                        assert_eq!(&bytes[..usize::from(*len)], code.as_slice());
                        assert_eq!((cpu.cpu.gpr, cpu.cpu.rip, cpu.cpu.rflags()), before);
                        continue;
                    }
                    let (device, ram, shift, ram_shift) = if low == RegionKind::Trap {
                        (0x6ffe, 0x7000, 0, 16)
                    } else {
                        (0x7000, 0x6ffe, 16, 0)
                    };
                    assert!(
                        matches!(run.stop(), StepStop::Exit(Exit::MmioRead {
                        addr, size: 2, total_size: 4, part, ..
                    }) if *addr == device && *part == shift / 16),
                        "{:?}",
                        run.stop()
                    );
                    assert_eq!((cpu.cpu.gpr, cpu.cpu.rip, cpu.cpu.rflags()), before);
                    cpu.complete_mmio_read(u64::from((old >> shift) & 0xffff));
                    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                    assert_eq!(run.steps(), 0);
                    let updated = if op == 0xc1 {
                        old.wrapping_add(src)
                    } else if equal {
                        src
                    } else {
                        old
                    };
                    assert!(
                        matches!(run.stop(), StepStop::Exit(Exit::MmioWrite {
                        addr, size: 2, value, total_size: 4, part, ..
                    }) if *addr == device && *value == u64::from((updated >> shift) & 0xffff)
                        && *part == shift / 16),
                        "{:?}",
                        run.stop()
                    );
                    assert_eq!((cpu.cpu.gpr, cpu.cpu.rip, cpu.cpu.rflags()), before);
                    assert_eq!(
                        fixture::value(&vm, ram, 2),
                        u64::from((old >> ram_shift) & 0xffff)
                    );
                    cpu.complete_mmio_write();
                    fixture::steps(&mut cpu, &vm, 1);
                    assert_eq!(
                        fixture::value(&vm, ram, 2),
                        u64::from((updated >> ram_shift) & 0xffff)
                    );
                    assert_eq!(cpu.cpu.gpr[0], u64::from(old));
                    assert_eq!(
                        cpu.cpu.gpr[1],
                        u64::from(if op == 0xc1 { old } else { src })
                    );
                    assert_eq!(
                        cpu.cpu.rflags() & 0x8d5,
                        if op == 0xc1 {
                            4
                        } else if equal {
                            0x44
                        } else {
                            0x95
                        }
                    );
                }
            }
        }
    }
}

#[test]
fn locked_xadd_and_cmpxchg_from_two_vcpus_lose_no_update() {
    // PRM §3.11.6–7 p. 3-47, §26 pp. 26-63, 26-283: LOCK holds
    // the complete update. Each successful run increments shared RAM once.
    use std::sync::{Arc, Barrier};
    for model in MODELS {
        for cas in [false, true] {
            let code: &[u8] = if cas {
                // MOV EAX,[6000]; LEA ECX,[EAX+1]; LOCK CMPXCHG [6000],ECX;
                // JNE to MOV; HLT.
                &[
                    0x8b, 0x05, 0, 0x60, 0, 0, 0x8d, 0x48, 1, 0xf0, 0x0f, 0xb1, 0x0d, 0, 0x60, 0,
                    0, 0x75, 0xed, 0xf4,
                ]
            } else {
                &[
                    0xb9, 1, 0, 0, 0, 0xf0, 0x0f, 0xc1, 0x0d, 0, 0x60, 0, 0, 0xf4,
                ]
            };
            let (vm, first) = setup(model, Mode::P32, code);
            let mut second = vm.new_vcpu();
            second.cpu = first.cpu.clone();
            let vm = Arc::new(vm);
            let barrier = Arc::new(Barrier::new(2));
            let threads: Vec<_> = [first, second]
                .into_iter()
                .map(|mut cpu| {
                    let vm = Arc::clone(&vm);
                    let barrier = Arc::clone(&barrier);
                    std::thread::spawn(move || {
                        barrier.wait();
                        for _ in 0..128 {
                            cpu.reset_execution_state(Default::default());
                            cpu.cpu.rip = 0x1000;
                            loop {
                                let run = cpu.run_steps(&vm, 64, StepOptions::default()).unwrap();
                                match run.stop() {
                                    StepStop::Exit(Exit::Hlt) => break,
                                    StepStop::Exit(Exit::BudgetExhausted) => {}
                                    stop => panic!("{stop:?}"),
                                }
                            }
                        }
                    })
                })
                .collect();
            for thread in threads {
                thread.join().unwrap();
            }
            assert_eq!(fixture::value(&vm, 0x6000, 4), 256);
        }
    }
}

#[test]
fn external_numeric_error_routing_is_not_offered_on_the_486_entries() {
    // On the 486, CR0.NE is the routing authority (fpu-error-routing.md, maintainer
    // decision 4); until that is implemented, both entries keep the internal route.
    for model in MODELS {
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        vm.set_cpu_model(model);
        assert_eq!(
            vm.set_numeric_error_route(x86jit_core::NumericErrorRoute::External),
            Err(x86jit_core::NumericErrorConfigError::UnsupportedModel)
        );
        let (_vm, mut cpu) = setup(model, Mode::Real, &[0xf4]);
        assert_eq!(
            cpu.set_numeric_error_route(x86jit_core::NumericErrorRoute::External),
            Err(x86jit_core::NumericErrorConfigError::UnsupportedModel)
        );
        assert_eq!(
            cpu.numeric_error_route(),
            x86jit_core::NumericErrorRoute::Internal
        );
    }
}
