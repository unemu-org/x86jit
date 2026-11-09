use x86jit_core::{Backend, Exit, InterpreterBackend, Prot, Reg, RegionKind, Vm, VmConfig};
use x86jit_cranelift::JitBackend;

fn backends() -> [Box<dyn Backend>; 2] {
    [Box::new(InterpreterBackend), Box::new(JitBackend::new())]
}

fn vm(backend: Box<dyn Backend>) -> Vm {
    let mut vm = Vm::with_backend(VmConfig::flat(0x2000), backend);
    vm.map(0x1000, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, &[0x0f, 0xa2, 0xf4]).unwrap(); // cpuid; hlt
    vm
}

fn default_invalid_leaf7_subleaf_is_zero(backend: Box<dyn Backend>) {
    // SDM Vol 1 §21.3, CPUID.07H and Table 21-17: inputs above MAX_SUBLEAF
    // return zero in all four registers.
    let vm = vm(backend);
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    cpu.set_reg(Reg::Rax, 7);
    cpu.set_reg(Reg::Rcx, 1);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    for reg in [Reg::Rax, Reg::Rbx, Reg::Rcx, Reg::Rdx] {
        assert_eq!(cpu.reg(reg), 0, "{reg:?}");
    }
}

#[test]
fn cpuid_default_invalid_leaf7_subleaf_is_zero_interp() {
    default_invalid_leaf7_subleaf_is_zero(Box::new(InterpreterBackend));
}

#[test]
fn cpuid_default_invalid_leaf7_subleaf_is_zero_jit() {
    default_invalid_leaf7_subleaf_is_zero(Box::new(JitBackend::new()));
}

use x86jit_core::{
    CpuidLeaf, CpuidOutOfRange, CpuidOverride, CpuidOverrides, CpuidRegisters as R, CpuidTable,
    GuestCpuFeatures, Vcpu,
};

fn read_cpuid(vm: &Vm, cpu: &mut Vcpu, leaf: u32, subleaf: u32) -> R {
    cpu.set_reg(Reg::Rip, 0x1000);
    // Dirty upper halves witness CPUID's 32-bit zero extension (SDM Vol 2A CPUID).
    cpu.set_reg(Reg::Rax, 0xffff_ffff_0000_0000 | u64::from(leaf));
    cpu.set_reg(Reg::Rcx, 0xffff_ffff_0000_0000 | u64::from(subleaf));
    cpu.set_reg(Reg::Rbx, u64::MAX);
    cpu.set_reg(Reg::Rdx, u64::MAX);
    assert!(matches!(cpu.run(vm, None), Exit::Hlt));
    let values = [Reg::Rax, Reg::Rbx, Reg::Rcx, Reg::Rdx].map(|r| cpu.reg(r));
    assert!(values.iter().all(|&v| v <= u64::from(u32::MAX)));
    R::new(
        values[0] as u32,
        values[1] as u32,
        values[2] as u32,
        values[3] as u32,
    )
}

fn table() -> (CpuidTable, Vec<(u32, u32, R)>) {
    // Synthetic profile; layouts: SDM Vol 1 Tables 21-4/6/8, 21-17–27,
    // 21-31/34/35/36/70, 21-89–93/98. These values exercise transport, not
    // instruction support or the consistency of a particular physical CPU model.
    let mut rows = vec![
        (0, 0, R::new(0x1f, 0x756e_6547, 0x6c65_746e, 0x4965_6e69)),
        (1, 0, R::new(0x0009_06e9, 0x0104_0800, 1 << 27, 1 << 8)),
        (7, 0, R::new(2, 1 << 18, 1 << 8, 0)),
        (7, 1, R::new(1, 0, 0, 0)),
        (7, 2, R::new(0, 0, 0, 1)),
        (0xb, 0, R::new(1, 2, 0x100, 0)),
        (0xd, 0, R::new(7, 832, 832, 0)),
        (0xd, 1, R::new(3, 832, 0, 0)),
        (0xd, 2, R::new(256, 576, 0, 0)),
        (0x1f, 0, R::new(1, 2, 0x100, 0)),
        (0x8000_0000, 0, R::new(0x8000_0008, 0, 0, 0)),
        (0x8000_0008, 0, R::new(0x3030, 0, 0, 0)),
    ];
    let mut brand = [0u8; 48];
    brand[..16].copy_from_slice(b"x86jit table CPU");
    for (i, bytes) in brand.chunks_exact(16).enumerate() {
        let words: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        rows.push((
            0x8000_0002 + i as u32,
            0,
            R::new(words[0], words[1], words[2], words[3]),
        ));
    }
    let mut table = CpuidTable::new(CpuidOutOfRange::HighestBasicLeaf);
    for &(leaf, subleaf, regs) in &rows {
        if [7, 0xb, 0xd, 0x1f].contains(&leaf) {
            let entry = table
                .leaves
                .entry(leaf)
                .or_insert_with(|| CpuidLeaf::Indexed {
                    subleaves: Default::default(),
                    invalid_subleaf: R::ZERO,
                });
            let CpuidLeaf::Indexed { subleaves, .. } = entry else {
                unreachable!()
            };
            subleaves.insert(subleaf, regs);
        } else {
            table.leaves.insert(leaf, CpuidLeaf::Constant(regs));
        }
    }
    (table, rows)
}

#[test]
fn cpuid_whole_leaves_match_on_interpreter_and_jit() {
    let (table, rows) = table();
    let mut outputs = Vec::new();
    for backend in backends() {
        let mut vm = vm(backend);
        vm.set_cpuid_table(Some(table.clone()));
        let mut cpu = vm.new_vcpu();
        let mut output = Vec::new();
        for &(leaf, subleaf, expected) in &rows {
            let actual = read_cpuid(&vm, &mut cpu, leaf, subleaf);
            assert_eq!(actual, expected, "{leaf:#x}:{subleaf}");
            output.push(actual);
        }
        // Leaf 1 ignores ECX (SDM Vol 1 §21.3, CPUID.01H preamble / Table 21-5).
        assert_eq!(read_cpuid(&vm, &mut cpu, 1, u32::MAX), rows[1].2);
        assert_eq!(read_cpuid(&vm, &mut cpu, 7, 3), R::ZERO);
        // No preset leakage into an unlisted leaf.
        assert_eq!(read_cpuid(&vm, &mut cpu, 0x8000_0001, 0), R::ZERO);
        outputs.push(output);
    }
    assert_eq!(outputs[0], outputs[1]);
}

#[test]
fn cpuid_table_out_of_range_and_invalid_subleaf_policies() {
    for backend in backends() {
        let mut vm = vm(backend);
        let (mut table, _) = table();
        let invalid = R::new(10, 20, 30, 40);
        let CpuidLeaf::Indexed {
            invalid_subleaf, ..
        } = table.leaves.get_mut(&7).unwrap()
        else {
            unreachable!()
        };
        *invalid_subleaf = invalid;
        table.missing_leaf = R::new(4, 3, 2, 1);
        // Entries outside architectural maxima do not bypass the range policy.
        table.leaves.insert(0x20, CpuidLeaf::Constant(invalid));
        table
            .leaves
            .insert(0x8000_0009, CpuidLeaf::Constant(invalid));
        // SDM Vol 1 §21.1.3 reserves this separate range for software emulation.
        table
            .leaves
            .insert(0x4000_0000, CpuidLeaf::Constant(invalid));
        vm.set_cpuid_table(Some(table.clone()));
        let mut cpu = vm.new_vcpu();
        assert_eq!(read_cpuid(&vm, &mut cpu, 7, 99), invalid);
        assert_eq!(read_cpuid(&vm, &mut cpu, 0x4000_0000, 123), invalid);
        assert_eq!(read_cpuid(&vm, &mut cpu, 5, 0), table.missing_leaf);
        // SDM Vol 1 §21.1.3 specifies the highest-basic-leaf redirect.
        // Retaining ECX is engine policy; Vol 2A CPUID Operation calls invalid
        // leaf outputs "Reserved".
        for leaf in [0x20, 0x8000_0009, u32::MAX] {
            assert_eq!(read_cpuid(&vm, &mut cpu, leaf, 0), R::new(1, 2, 0x100, 0));
            assert_eq!(read_cpuid(&vm, &mut cpu, leaf, 1), R::ZERO);
        }
        table.out_of_range = CpuidOutOfRange::Fixed(invalid);
        vm.set_cpuid_table(Some(table));
        let mut cpu = vm.new_vcpu();
        for (leaf, expected) in [
            (0x1f, R::new(1, 2, 0x100, 0)),
            (0x20, invalid),
            (0x8000_0008, R::new(0x3030, 0, 0, 0)),
            (0x8000_0009, invalid),
        ] {
            assert_eq!(read_cpuid(&vm, &mut cpu, leaf, 0), expected);
        }
    }
}

fn overrides(id: u32) -> CpuidOverrides {
    let mut over = CpuidOverrides {
        xcr0: Some(0x1_0000_0003),
        ..Default::default()
    };
    // SDM Vol 1 §21.1.4, Tables 21-8/31/70: logical processor IDs.
    over.leaves.insert(
        (1, 0),
        CpuidOverride {
            mask: R::new(0, 0xff00_0000, 1 << 27, 0),
            value: R::new(0, id << 24, 0, 0),
        },
    );
    for leaf in [0xb, 0x1f] {
        over.leaves.insert(
            (leaf, 0),
            CpuidOverride {
                mask: R::new(0, 0, 0, u32::MAX),
                value: R::new(0, 0, 0, id),
            },
        );
    }
    // SDM Vol 1 Table 21-1: OSXSAVE and enabled-state sizes are runtime mutable.
    for subleaf in [0, 1] {
        over.leaves.insert(
            (0xd, subleaf),
            CpuidOverride {
                mask: R::new(0, u32::MAX, 0, 0),
                value: R::new(0, 576, 0, 0),
            },
        );
    }
    over
}

#[test]
fn cpuid_vcpus_override_ids_and_runtime_fields_on_both_backends() {
    for backend in backends() {
        let mut vm = vm(backend);
        vm.set_cpuid_table(Some(table().0));
        vm.write_bytes(0x1010, &[0x0f, 0x01, 0xd0, 0xf4]).unwrap(); // xgetbv; hlt
        let mut first = vm.new_vcpu_with_cpuid_overrides(overrides(3));
        let mut second = vm.new_vcpu_with_cpuid_overrides(overrides(9));
        for (cpu, id) in [(&mut first, 3), (&mut second, 9)] {
            let one = read_cpuid(&vm, cpu, 1, 123);
            assert_eq!(one.ebx, (id << 24) | 0x0004_0800);
            assert_eq!(one.ecx & (1 << 27), 0);
            for leaf in [0xb, 0x1f] {
                assert_eq!(read_cpuid(&vm, cpu, leaf, 0).edx, id);
            }
            for subleaf in [0, 1] {
                assert_eq!(read_cpuid(&vm, cpu, 0xd, subleaf).ebx, 576);
            }
            cpu.set_reg(Reg::Rip, 0x1010);
            cpu.set_reg(Reg::Rcx, 0);
            assert!(matches!(
                cpu.run(&vm, None),
                Exit::Exception { vector: 6, .. }
            ));
            cpu.cpu
                .cpuid_overrides
                .leaves
                .get_mut(&(1, 0))
                .unwrap()
                .value
                .ecx |= 1 << 27;
            cpu.set_reg(Reg::Rip, 0x1010);
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rax), 3);
            assert_eq!(cpu.reg(Reg::Rdx), 1);
        }
        // Highest-basic-leaf fallback uses the same per-vCPU ID override.
        assert_eq!(read_cpuid(&vm, &mut first, 0x20, 0).edx, 3);
        assert_eq!(read_cpuid(&vm, &mut second, 0x20, 0).edx, 9);
    }
}

#[test]
fn cpuid_table_survives_vm_fork_and_state_clone_and_can_be_removed() {
    for backend in backends() {
        let mut vm = vm(backend);
        vm.set_guest_cpu_features(GuestCpuFeatures::v2());
        let mut original = vm.new_vcpu();
        let (table, _) = table();
        vm.set_cpuid_table(Some(table.clone()));
        assert_eq!(vm.cpuid_table(), Some(&table));
        assert!(original.cpu.cpuid_table.is_none());
        let mut cpu = vm.new_vcpu_with_cpuid_overrides(overrides(42));
        let saved = cpu.cpu.clone();
        let expected_one = R::new(0x0009_06e9, 0x2a04_0800, 0, 1 << 8);
        for _ in 0..2 {
            assert_eq!(
                read_cpuid(&vm, &mut original, 1, 123),
                GuestCpuFeatures::v2().cpuid(1, 0)
            );
            assert_eq!(read_cpuid(&vm, &mut cpu, 1, 123), expected_one);
            assert_eq!(read_cpuid(&vm, &mut original, 0, 0).eax, 7);
            assert_eq!(read_cpuid(&vm, &mut cpu, 0, 0).eax, 0x1f);
        }
        for child_backend in backends() {
            let child = vm.fork_with_backend(child_backend).unwrap();
            assert_eq!(child.cpuid_table(), Some(&table));
            let mut fresh = child.new_vcpu();
            let mut cloned = child.new_vcpu();
            cloned.cpu = saved.clone();
            for _ in 0..2 {
                assert_eq!(read_cpuid(&child, &mut cloned, 1, 123), expected_one);
                assert_eq!(read_cpuid(&child, &mut cloned, 0, 0).eax, 0x1f);
                assert_eq!(read_cpuid(&child, &mut fresh, 1, 123), table.query(1, 0));
                assert_eq!(read_cpuid(&child, &mut fresh, 0, 0).eax, 0x1f);
            }
        }
        vm.set_cpuid_table(None);
        let mut restored = vm.new_vcpu();
        // Interleave old and new descriptions through the same cached CPUID block.
        for _ in 0..2 {
            for legacy in [&mut original, &mut restored] {
                assert_eq!(
                    read_cpuid(&vm, legacy, 1, 123),
                    GuestCpuFeatures::v2().cpuid(1, 0)
                );
                assert_eq!(read_cpuid(&vm, legacy, 0, 0).eax, 7);
            }
            assert_eq!(read_cpuid(&vm, &mut cpu, 1, 123), expected_one);
            assert_eq!(read_cpuid(&vm, &mut cpu, 0, 0).eax, 0x1f);
        }
    }
}

#[test]
fn cpuid_default_projection_overrides_normalize_only_constant_leaves() {
    for backend in backends() {
        let vm = vm(backend);
        let mut first = vm.new_vcpu_with_cpuid_overrides(overrides(3));
        let mut second = vm.new_vcpu_with_cpuid_overrides(overrides(9));
        for cpu in [&mut first, &mut second] {
            cpu.cpu.cpuid_overrides.leaves.insert(
                (7, 0),
                CpuidOverride {
                    mask: R::new(0, 1 << 18, 0, 0),
                    value: R::new(0, 1 << 18, 0, 0),
                },
            );
        }
        for _ in 0..2 {
            for (cpu, id) in [(&mut first, 3), (&mut second, 9)] {
                let mut expected = GuestCpuFeatures::default().cpuid(1, 0);
                expected.ebx = id << 24;
                expected.ecx &= !(1 << 27);
                assert_eq!(read_cpuid(&vm, cpu, 1, 123), expected);
                let mut seven = GuestCpuFeatures::default().cpuid(7, 0);
                seven.ebx |= 1 << 18;
                assert_eq!(read_cpuid(&vm, cpu, 7, 0), seven);
                assert_eq!(read_cpuid(&vm, cpu, 7, 1), R::ZERO);
            }
        }
    }
}

#[test]
fn cpuid_new_feature_bits_are_individually_selectable() {
    use x86jit_core::Feature::*;
    // SDM Vol 1 Tables 21-9/10/19/90; AMD APM Vol 3 §E.4.2 (SSE4A).
    for (feature, leaf, field, bit) in [
        (Cx8, 1, 3, 8),
        (Sse4a, 0x8000_0001, 2, 6),
        (Adx, 7, 1, 19),
        (Rdrand, 1, 2, 30),
        (Rdseed, 7, 1, 18),
        (Rdtscp, 0x8000_0001, 3, 27),
    ] {
        let words = |r: R| [r.eax, r.ebx, r.ecx, r.edx];
        let empty = words(GuestCpuFeatures::empty().cpuid(leaf, 0));
        let enabled = words(GuestCpuFeatures::empty().with(feature).cpuid(leaf, 0));
        let mut expected = empty;
        expected[field] |= 1 << bit;
        assert_eq!(empty[field] & (1 << bit), 0, "{feature:?}");
        assert_eq!(enabled, expected, "{feature:?}");
    }
    for backend in backends() {
        let mut vm = vm(backend);
        vm.set_guest_cpu_features(GuestCpuFeatures::default().without(Cx8));
        let mut cpu = vm.new_vcpu();
        let mut expected = GuestCpuFeatures::default().cpuid(1, 0);
        expected.edx &= !(1 << 8);
        assert_eq!(read_cpuid(&vm, &mut cpu, 1, 0), expected);
    }
}

#[test]
fn cpuid_override_values_cannot_escape_their_masks() {
    for backend in backends() {
        let mut vm = vm(backend);
        let (table, _) = table();
        let base = table.query(1, 0);
        vm.set_cpuid_table(Some(table));
        let mask = R::new(0xff, 0xff00, 0xff0000, 0xff000000);
        let mut cpu = vm.new_vcpu();
        for value in [R::new(u32::MAX, u32::MAX, u32::MAX, u32::MAX), R::ZERO] {
            cpu.cpu
                .cpuid_overrides
                .leaves
                .insert((1, 0), CpuidOverride { mask, value });
            let expected = if value == R::ZERO {
                R::new(
                    base.eax & !mask.eax,
                    base.ebx & !mask.ebx,
                    base.ecx & !mask.ecx,
                    base.edx & !mask.edx,
                )
            } else {
                R::new(
                    base.eax | mask.eax,
                    base.ebx | mask.ebx,
                    base.ecx | mask.ecx,
                    base.edx | mask.edx,
                )
            };
            assert_eq!(read_cpuid(&vm, &mut cpu, 1, 123), expected);
        }
    }
}

fn read_xcr0(vm: &Vm, cpu: &mut Vcpu) -> u64 {
    cpu.set_reg(Reg::Rip, 0x1010);
    cpu.set_reg(Reg::Rcx, 0);
    assert!(matches!(cpu.run(vm, None), Exit::Hlt));
    cpu.reg(Reg::Rax) | (cpu.reg(Reg::Rdx) << 32)
}

#[test]
fn cpuid_runtime_updates_and_xcr0_are_per_vcpu_on_cached_code() {
    for backend in backends() {
        let mut vm = vm(backend);
        let mut table = table().0;
        table.leaves.insert(0x19, CpuidLeaf::Constant(R::ZERO));
        table
            .leaves
            .insert(0x8000_0001, CpuidLeaf::Constant(R::ZERO));
        vm.set_cpuid_table(Some(table));
        vm.write_bytes(0x1010, &[0x0f, 0x01, 0xd0, 0xf4]).unwrap();
        let mut first = vm.new_vcpu_with_cpuid_overrides(overrides(3));
        let mut second = vm.new_vcpu_with_cpuid_overrides(overrides(9));
        for round in 0..2 {
            for (cpu, xcr0, bytes, osxsave) in [
                (
                    &mut first,
                    if round == 0 { 3 } else { 7 },
                    if round == 0 { 576 } else { 832 },
                    round << 27,
                ),
                (
                    &mut second,
                    if round == 0 {
                        0x1_0000_0007
                    } else {
                        0x1_0000_0003
                    },
                    if round == 0 { 832 } else { 576 },
                    (1 - round) << 27,
                ),
            ] {
                cpu.cpu.cpuid_overrides.xcr0 = Some(xcr0);
                cpu.cpu
                    .cpuid_overrides
                    .leaves
                    .get_mut(&(1, 0))
                    .unwrap()
                    .value
                    .ecx = osxsave;
                for subleaf in [0, 1] {
                    cpu.cpu
                        .cpuid_overrides
                        .leaves
                        .get_mut(&(0xd, subleaf))
                        .unwrap()
                        .value
                        .ebx = bytes;
                }
                // Runtime mutable layouts: SDM Vol 1 §21.1.5, Table 21-1;
                // SYSCALL is EDX[11] per Table 21-90.
                for (leaf, mask, value) in [
                    (7, R::new(0, 0, 1 << 4, 0), R::new(0, 0, round << 4, 0)),
                    (0x19, R::new(0, 1, 0, 0), R::new(0, round, 0, 0)),
                    (
                        0x8000_0001,
                        R::new(0, 0, 0, 1 << 11),
                        R::new(0, 0, 0, round << 11),
                    ),
                ] {
                    cpu.cpu
                        .cpuid_overrides
                        .leaves
                        .insert((leaf, 0), CpuidOverride { mask, value });
                }
                if osxsave != 0 {
                    assert_eq!(read_xcr0(&vm, cpu), xcr0);
                } else {
                    cpu.set_reg(Reg::Rip, 0x1010);
                    cpu.set_reg(Reg::Rcx, 0);
                    assert!(matches!(
                        cpu.run(&vm, None),
                        Exit::Exception { vector: 6, .. }
                    ));
                }
                assert_eq!(read_cpuid(&vm, cpu, 1, 99).ecx & (1 << 27), osxsave);
                for subleaf in [0, 1] {
                    assert_eq!(read_cpuid(&vm, cpu, 0xd, subleaf).ebx, bytes);
                }
                assert_eq!(read_cpuid(&vm, cpu, 7, 0).ecx & (1 << 4), round << 4);
                assert_eq!(read_cpuid(&vm, cpu, 0x19, 99).ebx, round);
                assert_eq!(read_cpuid(&vm, cpu, 0x8000_0001, 99).edx, round << 11);
            }
        }
        first.cpu.cpuid_overrides.xcr0 = None;
        first
            .cpu
            .cpuid_overrides
            .leaves
            .get_mut(&(1, 0))
            .unwrap()
            .value
            .ecx |= 1 << 27;
        assert_eq!(
            read_xcr0(&vm, &mut first),
            GuestCpuFeatures::default().xcr0()
        );
    }
}

#[test]
fn cpuid_syscall_bit_is_a_frozen_deviation_outside_long_mode() {
    use x86jit_core::CpuMode;
    // SDM Vol 1 §21.3, Table 21-90: Intel enumerates EDX[11] as zero outside
    // 64-bit mode. The frozen engine projection sets it in every mode.
    for mode in [CpuMode::Long64, CpuMode::Compat32, CpuMode::Real16] {
        for backend in backends() {
            let mut vm = vm(backend);
            vm.set_cpu_mode(mode);
            let mut cpu = vm.new_vcpu();
            assert_eq!(
                read_cpuid(&vm, &mut cpu, 0x8000_0001, 99).edx & (1 << 11),
                1 << 11
            );
            cpu.cpu.cpuid_overrides.leaves.insert(
                (0x8000_0001, 0),
                CpuidOverride {
                    mask: R::new(0, 0, 0, 1 << 11),
                    value: R::ZERO,
                },
            );
            assert_eq!(
                read_cpuid(&vm, &mut cpu, 0x8000_0001, 99).edx & (1 << 11),
                0
            );
        }
    }
}

#[test]
fn cpuid_default_replies_include_extended_state_layout() {
    // Frozen engine policy; layouts are SDM Vol 1 Tables 21-4/6/8/9/10/17/19/90.
    // Out-of-range zeros intentionally deviate from SDM Vol 1 §21.1.3.
    let replies = [
        (0, 99, R::new(0xd, 0x756e_6547, 0x6c65_746e, 0x4965_6e69)),
        (1, 99, R::new(0x0003_06c3, 0, 0x1c80_0201, 0x0780_8111)),
        (7, 0, R::new(0, 0x20, 0, 0)),
        (7, 1, R::ZERO),
        (0x8000_0000, 99, R::new(0x8000_0001, 0, 0, 0)),
        (0x8000_0001, 99, R::new(0, 0, 1, 0x2000_0800)),
        (8, 0, R::ZERO),
        (0xd, 0, R::new(7, 832, 832, 0)),
        (0x8000_0002, 0, R::ZERO),
        (u32::MAX, 0, R::ZERO),
    ];
    for backend in backends() {
        let vm = vm(backend);
        let mut cpu = vm.new_vcpu();
        for (leaf, subleaf, expected) in replies {
            assert_eq!(
                read_cpuid(&vm, &mut cpu, leaf, subleaf),
                expected,
                "{leaf:#x}:{subleaf}"
            );
        }
    }
}
