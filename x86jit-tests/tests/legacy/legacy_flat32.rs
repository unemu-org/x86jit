//! Flat32 proof and exact interpreted blocks (decision-27 order 2).
#[path = "../support/descriptor_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::descriptor;
use x86jit_core::{
    derive_legacy_context, CpuModel, Exit, SegmentClass, StepOptions, StepStop, Vcpu, Vm,
};

fn setup(model: CpuModel, code: &[u8], checked: bool) -> (Vm, Vcpu) {
    setup_delivery(
        model,
        code,
        checked,
        x86jit_core::ExceptionDelivery::Outward,
    )
}
fn setup_delivery(
    model: CpuModel,
    code: &[u8],
    checked: bool,
    delivery: x86jit_core::ExceptionDelivery,
) -> (Vm, Vcpu) {
    let (vm, mut cpu) = fixture::setup_delivery(model, code, delivery);
    for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
        cache.base = 0;
        cache.limit = u32::MAX;
        cache.attributes = if i == 0 { 0x409b } else { 0x4093 };
    }
    cpu.cpu.cs = 8;
    cpu.cpu.ds = 16;
    cpu.cpu.es = 16;
    cpu.cpu.ss = 16;
    descriptor(&vm, 0x2008, 0, u32::MAX, 0xc09b);
    descriptor(&vm, 0x2010, 0, u32::MAX, 0xc093);
    if checked {
        // These programs never reach ES:FFFFFFFF. All their accesses have
        // identical rights/addresses, while this cache cannot prove Flat32.
        // A non-present hidden GS cache also keeps the state off Segmented:
        // P is checked on a descriptor load, not on an access (SDM Vol 3A
        // §3.4.5).
        cpu.cpu.legacy.segments[2].limit -= 1;
        cpu.cpu.legacy.segments[5].attributes &= !0x80;
    }
    cpu.set_a20_enabled(true);
    (vm, cpu)
}
fn class(cpu: &Vcpu, _model: CpuModel) -> SegmentClass {
    derive_legacy_context(cpu.legacy_cpu_model().unwrap(), &cpu.cpu)
        .unwrap()
        .segment_class
}
fn same(a: &Vcpu, b: &mut Vcpu, avm: &Vm, bvm: &Vm) {
    b.cpu.legacy.segments[2].limit = a.cpu.legacy.segments[2].limit;
    b.cpu.legacy.segments[5].attributes = a.cpu.legacy.segments[5].attributes;
    assert_eq!(format!("{:?}", a.cpu), format!("{:?}", b.cpu));
    assert_eq!(a.retired_instructions(), b.retired_instructions());
    let mut am = vec![0; 0x10000];
    let mut bm = vec![0; 0x10000];
    avm.read_bytes(0, &mut am).unwrap();
    bvm.read_bytes(0, &mut bm).unwrap();
    assert_eq!(am, bm);
}

#[test]
fn selection_requires_every_proof_dimension_but_allows_checked_fs_gs() {
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        let (_, mut cpu) = setup(model, b"\x90", false);
        assert_eq!(class(&cpu, model), SegmentClass::Flat32);
        let original = cpu.cpu.clone();
        for segment in 0..4 {
            for field in 0..7 {
                cpu.cpu = original.clone();
                let cache = &mut cpu.cpu.legacy.segments[segment];
                match field {
                    0 => cache.base = 1,
                    1 => cache.limit -= 1,
                    2 => cache.attributes &= !0x80,
                    3 => cache.attributes &= !0x10,
                    4 => cache.attributes ^= if segment == 0 { 8 } else { 4 },
                    5 => cache.attributes &= !2,
                    _ => cache.unusable = 1,
                }
                assert!(
                    !derive_legacy_context(cpu.legacy_cpu_model().unwrap(), &cpu.cpu)
                        .is_ok_and(|c| c.segment_class == SegmentClass::Flat32)
                );
            }
        }
        for field in 0..4 {
            cpu.cpu = original.clone();
            match field {
                0 => cpu.cpu.legacy.segments[0].attributes &= !0x4000,
                1 => cpu.cpu.legacy.segments[3].attributes &= !0x4000,
                2 => cpu.cpu.legacy.cpl = 3,
                _ => cpu.cpu.legacy.cr0 = 0,
            }
            assert_ne!(class(&cpu, model), SegmentClass::Flat32);
        }
        cpu.cpu = original;
        cpu.cpu.legacy.cr0 |= 1 << 31;
        let paged = cpu.cpu.clone();
        cpu.cpu.rflags_system |= 1 << 17;
        assert!(
            !derive_legacy_context(cpu.legacy_cpu_model().unwrap(), &cpu.cpu)
                .is_ok_and(|c| c.segment_class == SegmentClass::Flat32)
        );
        cpu.cpu = paged;
        cpu.cpu.legacy.segments[4].base = 0x4000;
        cpu.cpu.legacy.segments[5].unusable = 1;
        assert_eq!(class(&cpu, model), SegmentClass::Flat32);
    }
}

#[test]
fn budgets_and_fault_ordinals_match_checked_with_full_state() {
    let bodies: &[&[u8]] = &[
        b"\x40\x01\xc3\x31\xda", // register/flags
        b"\xa1\x00\x60\x00\x00\x40\xa3\x00\x60\x00\x00",
        b"\x50\x5b\x66\x50\x66\x59", // stack widths
        b"\x67\x8b\x07\x66\x01\xc3", // address/operand overrides
        b"\x64\xa1\x00\x00\x00\x00\x65\xa3\x04\x00\x00\x00",
        b"\xf3\xa4", // REP stays in the singleton ledger
    ];
    let faults: &[&[u8]] = &[
        b"\xf4",
        b"\x0f\x0b",
        b"\xf7\xf1",                 // HLT, #UD, #DE
        b"\xa1\xff\xff\xff\xff",     // #GP at the whole-operand edge
        b"\x36\xa1\xff\xff\xff\xff", // same offset through SS: #SS
        b"\x2e\xa3\x00\x60\x00\x00", // CS remains read-only
        b"\x65\xa1\x00\x01\x00\x00", // checked GS limit
    ];
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        for body in bodies {
            for fault in faults {
                for ordinal in 0..4 {
                    let mut code = body.to_vec();
                    code.extend(vec![0x40; ordinal]);
                    code.extend_from_slice(fault);
                    code.push(0xf4);
                    for budget in 0..10 {
                        let (avm, mut a) = setup(model, &code, false);
                        let (bvm, mut b) = setup(model, &code, true);
                        for cpu in [&mut a, &mut b] {
                            cpu.cpu.gpr[1] = if *body == b"\xf3\xa4" { 3 } else { 0 };
                            cpu.cpu.gpr[3] = 0x6000;
                            cpu.cpu.gpr[6] = 0x6000;
                            cpu.cpu.gpr[7] = 0x7000;
                            cpu.cpu.legacy.segments[4].base = 0x6000;
                            cpu.cpu.legacy.segments[5].base = 0x7000;
                            cpu.cpu.legacy.segments[5].limit = 0xff;
                        }
                        let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
                        let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
                        assert_eq!(
                            format!("{ar:?}"),
                            format!("{br:?}"),
                            "{code:02x?} budget={budget}"
                        );
                        assert!(!matches!(
                            ar.stop(),
                            StepStop::Exit(Exit::Exception { vector: 14, .. })
                        ));
                        same(&a, &mut b, &avm, &bvm);
                    }
                }
            }
        }
    }
}

#[test]
fn smc_stops_before_a_modified_successor_and_embedder_edits_reselect() {
    // Store HLT over the following INC in a cached multi-instruction block.
    let code = b"\x40\xc6\x05\x08\x10\x00\x00\xf4\x40\xf4";
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        let (avm, mut a) = setup(model, code, false);
        let (bvm, mut b) = setup(model, code, true);
        let ar = a.run_steps(&avm, 20, StepOptions::default()).unwrap();
        let br = b.run_steps(&bvm, 20, StepOptions::default()).unwrap();
        assert_eq!(ar.steps(), 3);
        assert!(matches!(ar.stop(), StepStop::Exit(Exit::Hlt)));
        assert_eq!(format!("{ar:?}"), format!("{br:?}"));
        same(&a, &mut b, &avm, &bvm);
        a.cpu.rip = 0x1000;
        a.cpu.legacy.segments[1].base = 0x100;
        assert_eq!(class(&a, model), SegmentClass::Segmented);
        avm.write_bytes(0x1000, b"\xa1\x00\x60\x00\x00\xf4")
            .unwrap();
        avm.write_bytes(0x6100, &42u32.to_le_bytes()).unwrap();
        assert!(matches!(a.run(&avm, Some(20)), Exit::Hlt));
        assert_eq!(a.cpu.gpr[0], 42);
    }
}

#[test]
fn failed_source_barrier_refetches_a_modified_successor() {
    let code = b"\x40\xc6\x05\x08\x10\x00\x00\xf4\x40\xf4";
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        let (avm, mut a) = x86jit_core::with_barrier_success(|| setup(model, code, false));
        let (bvm, mut b) = x86jit_core::with_barrier_success(|| setup(model, code, true));
        x86jit_core::with_barrier_flush_failure(|| {
            let ar = a.run_steps(&avm, 20, StepOptions::default()).unwrap();
            let br = b.run_steps(&bvm, 20, StepOptions::default()).unwrap();
            assert_eq!(ar.steps(), 3);
            assert!(matches!(ar.stop(), StepStop::Exit(Exit::Hlt)));
            assert_eq!(a.cpu.gpr[0], 1);
            assert_eq!(a.retired_instructions(), 3);
            assert_eq!(format!("{ar:?}"), format!("{br:?}"));
            same(&a, &mut b, &avm, &bvm);
        });
        let stats = avm.memory_stats();
        assert_eq!(
            stats.source_barrier_mode,
            x86jit_core::SourceBarrierMode::Process
        );
        assert!(stats.source_barrier_failures > 0);
        assert!(stats.source_barrier_uncached_executions > 0);
    }
}

#[test]
fn segment_load_cr0_and_interrupt_delivery_end_the_proof() {
    for code in [
        &b"\x40\x8e\xd8\xa1\x00\x60\x00\x00\xf4"[..],
        &b"\x40\x0f\x22\xc0\xf4"[..],
        &b"\x40\xcd\x20\xf4"[..],
    ] {
        let model = CpuModel::I80386;
        let (vm, mut cpu) = setup(model, code, false);
        if code[1] == 0x8e {
            descriptor(&vm, 0x2018, 0x100, u32::MAX, 0xc093);
            cpu.cpu.gpr[0] = 23; // INC produces selector 24
            fixture::steps(&mut cpu, &vm, 2);
            assert_eq!(class(&cpu, model), SegmentClass::Segmented);
            assert_eq!(cpu.cpu.legacy.segments[1].base, 0x100);
        } else if code[1] == 0x0f {
            cpu.cpu.gpr[0] = u32::MAX as u64; // INC wraps EAX to zero, clears PE
            fixture::steps(&mut cpu, &vm, 2);
            assert_eq!(class(&cpu, model), SegmentClass::Checked);
            assert_eq!(cpu.cpu.legacy.cr0 & 1, 0);
        } else {
            // 386 PRM §9.5: interrupt gate reloads CS from its descriptor.
            descriptor(&vm, 0x2018, 0, 0xffff, 0x9b);
            cpu.cpu.legacy.idtr.base = 0x2800;
            cpu.cpu.legacy.idtr.limit = 0x107;
            vm.write_bytes(0x2900, &[0, 0x18, 24, 0, 0, 0x8e, 0, 0])
                .unwrap();
            vm.write_bytes(0x1800, b"\x90").unwrap();
            // Delivery policy is selected on the VM, before vCPU creation.
            // Software INT uses the architectural gate even with Outward faults.
            fixture::steps(&mut cpu, &vm, 2);
            assert_eq!(cpu.cpu.rip, 0x1800);
            assert_eq!(class(&cpu, model), SegmentClass::Segmented);
        }
    }
}

#[test]
fn hardware_interrupt_and_debug_boundaries_reselect_before_successors() {
    let model = CpuModel::I80386;
    let (vm, mut cpu) = setup(model, b"\x40\x43\x42\xeb\xfb", false);
    descriptor(&vm, 0x2018, 0, 0xffff, 0x9b);
    cpu.cpu.legacy.idtr.base = 0x2800;
    cpu.cpu.legacy.idtr.limit = 0x107;
    vm.write_bytes(0x2900, &[0, 0x18, 24, 0, 0, 0x8e, 0, 0])
        .unwrap();
    vm.write_bytes(0x1800, b"\x90").unwrap();
    fixture::steps(&mut cpu, &vm, 2); // populate a block, stop in its middle
    cpu.cpu.flags.if_ = true;
    cpu.inject_irq(0x20);
    let before = format!("{:?}", cpu.cpu);
    let retired = cpu.retired_instructions();
    let zero = cpu.run_steps(&vm, 0, StepOptions::default()).unwrap();
    assert_eq!(zero.steps(), 0);
    assert_eq!(format!("{:?}", cpu.cpu), before);
    assert!(cpu.has_pending_irq());
    fixture::steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.retired_instructions(), retired + 1);
    assert_eq!(cpu.cpu.rip, 0x1801);
    assert_eq!(cpu.cpu.gpr[2], 0); // pre-interrupt successor never ran
    assert_eq!(class(&cpu, model), SegmentClass::Segmented);

    for budget in 0..5 {
        let (avm, mut a) = setup(model, b"\x40\x43\x42\xf4", false);
        let (bvm, mut b) = setup(model, b"\x40\x43\x42\xf4", true);
        a.cpu.rflags_system |= 0x100;
        b.cpu.rflags_system |= 0x100;
        let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
        let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
        assert_eq!(format!("{ar:?}"), format!("{br:?}"));
        same(&a, &mut b, &avm, &bvm);
    }
}

#[test]
fn page_boundaries_straddles_and_a20_source_edits_are_exact() {
    let model = CpuModel::I80386;
    for ip in [0x1ffe, 0x1fff] {
        for budget in 0..5 {
            let code = b"\x40\xb8\x78\x56\x34\x12\xf4";
            let (avm, mut a) = setup(model, b"", false);
            let (bvm, mut b) = setup(model, b"", true);
            avm.write_bytes(ip, code).unwrap();
            bvm.write_bytes(ip, code).unwrap();
            a.cpu.rip = ip;
            b.cpu.rip = ip;
            let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
            let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
            assert_eq!(format!("{ar:?}"), format!("{br:?}"));
            same(&a, &mut b, &avm, &bvm);
        }
    }
    let (vm, mut cpu) = setup(model, b"", false);
    cpu.set_a20_enabled(false);
    cpu.cpu.rip = 0x1fffff;
    vm.write_bytes(0xfffff, b"\xb8").unwrap();
    vm.write_bytes(0x200000, &[42, 0, 0, 0]).unwrap();
    fixture::steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.gpr[0], 42);
    cpu.cpu.rip = 0x1fffff;
    vm.write_bytes(0x200000, &[43]).unwrap();
    fixture::steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.gpr[0], 43);
}

#[test]
fn a_mid_block_mmio_retry_owns_only_its_faulting_instruction() {
    use x86jit_core::{Prot, RegionKind};
    let model = CpuModel::I80386;
    let code = b"\x40\xa1\x00\x60\x00\x00\x40\xf4";
    let make = |checked| {
        let (mut vm, cpu) = setup(model, code, checked);
        vm.unmap(0, 0x240000).unwrap();
        vm.map(0, 0x6000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x6000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        vm.map(0x7000, 0x19000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, code).unwrap();
        (vm, cpu)
    };
    let (avm, mut a) = make(false);
    let (bvm, mut b) = make(true);
    let ar = a.run_steps(&avm, 20, StepOptions::default()).unwrap();
    let br = b.run_steps(&bvm, 20, StepOptions::default()).unwrap();
    assert_eq!(ar.steps(), 1);
    assert!(matches!(ar.stop(), StepStop::Exit(Exit::MmioRead { .. })));
    assert_eq!(format!("{ar:?}"), format!("{br:?}"));
    assert_eq!(a.cpu.rip, 0x1001);
    a.complete_mmio_read(41);
    b.complete_mmio_read(41);
    let zero = a.run_steps(&avm, 0, StepOptions::default()).unwrap();
    assert_eq!(zero.steps(), 0);
    let ar = a.run_steps(&avm, 20, StepOptions::default()).unwrap();
    let br = b.run_steps(&bvm, 20, StepOptions::default()).unwrap();
    assert_eq!(ar.steps(), 3);
    assert_eq!(format!("{ar:?}"), format!("{br:?}"));
    assert_eq!(a.cpu.gpr[0], 42);
    assert_eq!(a.retired_instructions(), 4);
    b.cpu.legacy.segments[2].limit += 1;
    b.cpu.legacy.segments[5].attributes |= 0x80;
    assert_eq!(format!("{:?}", a.cpu), format!("{:?}", b.cpu));
    assert_eq!(a.retired_instructions(), b.retired_instructions());
}

#[test]
fn cached_flat32_blocks_revalidate_fetch_permits_and_region_mappings() {
    use x86jit_core::{FetchVerdict, Prot, RegionKind};
    for checked in [false, true] {
        let (mut vm, mut cpu) = setup(CpuModel::I80386, b"\x40\x43\xeb\xfc", checked);
        fixture::steps(&mut cpu, &vm, 3);
        cpu.cpu.rip = 0x1000;
        vm.set_fetch_policy(Box::new(|_| FetchVerdict::DenyProtection));
        let before = format!("{:?}", cpu.cpu);
        let retired = cpu.retired_instructions();
        let denied = cpu.run_steps(&vm, 3, StepOptions::default()).unwrap();
        assert_eq!(denied.steps(), 0);
        assert!(!matches!(
            denied.stop(),
            StepStop::Exit(Exit::BudgetExhausted)
        ));
        assert_eq!(format!("{:?}", cpu.cpu), before);
        assert_eq!(cpu.retired_instructions(), retired);
        vm.set_fetch_policy(Box::new(|_| FetchVerdict::Allow));
        fixture::steps(&mut cpu, &vm, 3);
        cpu.cpu.rip = 0x1000;
        vm.unmap(0, 0x240000).unwrap();
        vm.map(0, 0x240000, Prot::RWX, RegionKind::Trap).unwrap();
        let stopped = cpu.run_steps(&vm, 3, StepOptions::default()).unwrap();
        assert_eq!(stopped.steps(), 0);
        assert_eq!(cpu.retired_instructions(), retired + 3);
        assert!(!matches!(
            stopped.stop(),
            StepStop::Exit(Exit::BudgetExhausted)
        ));
    }
}

const PD: u64 = 0x1e000;
const PT: u64 = 0x1f000;
fn put(vm: &Vm, at: u64, value: u32) {
    vm.write_bytes(at, &value.to_le_bytes()).unwrap();
}
fn get(vm: &Vm, at: u64) -> u32 {
    let mut bytes = [0; 4];
    vm.read_bytes(at, &mut bytes).unwrap();
    u32::from_le_bytes(bytes)
}
fn enable_paging(vm: &Vm, cpu: &mut Vcpu) {
    // 80386 PRM §5.2 pp. 5-7–11: identity pages, separate RAM tables.
    put(vm, PD, PT as u32 | 7);
    for page in 0..32 {
        put(vm, PT + page * 4, (page * 4096) as u32 | 7);
    }
    cpu.cpu.legacy.cr3 = PD as u32;
    cpu.cpu.legacy.cr0 |= 1 << 31;
}
fn paged_setup(model: CpuModel, code: &[u8], checked: bool) -> (Vm, Vcpu) {
    let (vm, mut cpu) = setup(model, code, checked);
    enable_paging(&vm, &mut cpu);
    (vm, cpu)
}
fn paged_same(a: &Vcpu, b: &mut Vcpu, avm: &Vm, bvm: &Vm) {
    same(a, b, avm, bvm);
    let mut am = vec![0; 0x2000];
    let mut bm = vec![0; 0x2000];
    avm.read_bytes(PD, &mut am).unwrap();
    bvm.read_bytes(PD, &mut bm).unwrap();
    assert_eq!(am, bm, "page-table A/D effects");
}

#[test]
fn paged_warm_data_and_stack_transfers_equal_checked_at_every_budget() {
    // SDM Vol 2A ENTER/LEAVE; Vol 2B PUSH/POP/RET; 80386 PRM §5.2.4.3.
    let mut code = b"\xa1\x00\x60\x00\x00\xa3\x04\x60\x00\x00\x50\x5b\xe8".to_vec();
    let call = code.len();
    code.extend([0; 4]);
    code.extend_from_slice(b"\xc8\x08\x00\x00\xc9\x4f\x75");
    code.push((-(code.len() as i8 + 1)) as u8);
    code.extend_from_slice(b"\x0f\x0b");
    let sub = code.len();
    code[call..call + 4].copy_from_slice(&((sub - call - 4) as u32).to_le_bytes());
    code.extend_from_slice(b"\x40\xc3");
    for model in [
        CpuModel::I80386,
        CpuModel::I80486Dx,
        CpuModel::I80486DxCpuid,
    ] {
        for cpl in [0, 3] {
            for dirty in [false, true] {
                for budget in 0..36 {
                    let make = |checked| {
                        let (vm, mut cpu) = if cpl == 3 {
                            user_setup(model, &code, true, checked)
                        } else {
                            paged_setup(model, &code, checked)
                        };
                        cpu.cpu.gpr[4] = 0x7ff0;
                        cpu.cpu.gpr[5] = 0x7fe0;
                        cpu.cpu.gpr[7] = 3;
                        put(&vm, 0x6000, 41);
                        if dirty {
                            for page in [6, 7] {
                                put(&vm, PT + 4 * page, (page * 4096) as u32 | 0x67);
                            }
                        }
                        (vm, cpu)
                    };
                    let (avm, mut a) = make(false);
                    let (bvm, mut b) = make(true);
                    assert_eq!(class(&a, model), SegmentClass::Flat32);
                    assert_eq!(class(&b, model), SegmentClass::Checked);
                    let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
                    let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
                    assert_eq!(
                        format!("{ar:?}"),
                        format!("{br:?}"),
                        "{model:?} CPL{cpl} dirty={dirty} budget={budget}"
                    );
                    paged_same(&a, &mut b, &avm, &bvm);
                }
            }
        }
    }
}

#[test]
fn paged_warm_multi_operand_second_fault_and_locked_transfers_equal_checked() {
    // 80386 PRM §9.8.14: restartable #PF; SDM Vol 3A §11.1.2: LOCK.
    let cases: &[(&[u8], u64, bool)] = &[
        (
            b"\xa1\x00\x60\x00\x00\xff\x35\x00\x60\x00\x00",
            0x8004,
            true,
        ), // PUSH [mem]: stack faults second
        (
            b"\xa1\xf0\x7f\x00\x00\x8f\x05\x00\x60\x00\x00",
            0x7ff0,
            false,
        ), // POP [mem]: destination faults second
        (
            b"\xa1\x00\x60\x00\x00\xf0\x01\x05\x00\x60\x00\x00\x0f\x0b",
            0x7ff0,
            false,
        ),
        (
            b"\xa1\xfe\x6f\x00\x00\xf0\x01\x05\xfe\x6f\x00\x00\x0f\x0b",
            0x7ff0,
            false,
        ),
    ];
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        for cpl in [0, 3] {
            for (index, &(code, stack, stack_fault)) in cases.iter().enumerate() {
                let make = |checked| {
                    let (vm, mut cpu) = if cpl == 3 {
                        user_setup(model, code, true, checked)
                    } else {
                        paged_setup(model, code, checked)
                    };
                    cpu.cpu.gpr[4] = stack;
                    cpu.cpu.gpr[0] = 11;
                    put(&vm, 0x6000, 41);
                    put(&vm, 0x7ff0, 123);
                    if index == 3 {
                        put(&vm, PT + 7 * 4, 0x9007);
                        vm.write_bytes(0x6ffe, &[0x78, 0x56]).unwrap();
                        vm.write_bytes(0x9000, &[0x34, 0x12]).unwrap();
                    }
                    if index < 2 {
                        put(&vm, PT + 4 * if stack_fault { 8 } else { 6 }, 0);
                    }
                    (vm, cpu)
                };
                let (avm, mut a) = make(false);
                let (bvm, mut b) = make(true);
                assert_eq!(class(&b, model), SegmentClass::Checked);
                let ar = a.run_steps(&avm, 8, StepOptions::default()).unwrap();
                let br = b.run_steps(&bvm, 8, StepOptions::default()).unwrap();
                assert_eq!(format!("{ar:?}"), format!("{br:?}"));
                if index < 2 {
                    assert_eq!(ar.steps(), 1);
                    assert_eq!(a.cpu.gpr[4], stack);
                    assert_eq!(a.cpu.legacy.cr2, if stack_fault { 0x8000 } else { 0x6000 });
                    let StepStop::Exit(exit @ Exit::Exception { vector: 14, .. }) = ar.stop()
                    else {
                        panic!("{ar:?}");
                    };
                    assert_eq!(
                        exit.fault_metadata().unwrap().error_code(),
                        x86jit_core::fault::ErrorCode::bits(2 | if cpl == 3 { 4 } else { 0 }, 7)
                    );
                } else {
                    assert_eq!(ar.steps(), 2);
                    if index == 2 {
                        assert_eq!(get(&avm, 0x6000), 82);
                    } else {
                        let mut low = [0; 2];
                        let mut high = [0; 2];
                        avm.read_bytes(0x6ffe, &mut low).unwrap();
                        avm.read_bytes(0x9000, &mut high).unwrap();
                        assert_eq!(
                            u32::from_le_bytes([low[0], low[1], high[0], high[1]]),
                            0x2468acf0
                        );
                        assert_eq!(get(&avm, PT + 6 * 4) & 0x60, 0x60);
                        assert_eq!(get(&avm, PT + 7 * 4) & 0x60, 0x60);
                    }
                }
                paged_same(&a, &mut b, &avm, &bvm);
            }
        }
    }
}

#[test]
fn paged_dirty_code_alias_stores_refresh_current_and_following_blocks() {
    // Paging design §3.5: both linear aliases register the physical source;
    // an already-dirty TLB hit must still pass through the physical SMC barrier.
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        for cpl in [0, 3] {
            for following in [false, true] {
                let target = if following { 16 } else { 13 };
                let mut code = b"\x8b\x1d\x00\x90\x00\x00\xc6\x05".to_vec();
                // Six-byte load + seven-byte store: fallthrough is offset 13.
                code.extend((0x9000u32 + target).to_le_bytes());
                code.push(0x43); // Replace INC EAX with INC EBX.
                if following {
                    code.extend_from_slice(b"\xeb\x01\x90");
                }
                code.extend_from_slice(b"\x40\x0f\x0b");
                let make = |checked| {
                    let (vm, mut cpu) = if cpl == 3 {
                        user_setup(model, &code, true, checked)
                    } else {
                        paged_setup(model, &code, checked)
                    };
                    put(&vm, PT + 9 * 4, 0x1067);
                    // Precache the old following block before the alias store.
                    if following {
                        cpu.cpu.rip = 0x1000 + u64::from(target);
                        assert_eq!(
                            cpu.run_steps(&vm, 1, StepOptions::default())
                                .unwrap()
                                .steps(),
                            1
                        );
                        cpu.cpu.gpr[0] = 0;
                        cpu.cpu.rip = 0x1000;
                    }
                    (vm, cpu)
                };
                let (avm, mut a) = make(false);
                let (bvm, mut b) = make(true);
                assert_eq!(class(&b, model), SegmentClass::Checked);
                let ar = a.run_steps(&avm, 10, StepOptions::default()).unwrap();
                let br = b.run_steps(&bvm, 10, StepOptions::default()).unwrap();
                assert_eq!(format!("{ar:?}"), format!("{br:?}"));
                assert_eq!(a.cpu.gpr[0], 0);
                // EBX was loaded from the source before its successor changed.
                assert_eq!(
                    a.cpu.gpr[3],
                    u64::from(u32::from_le_bytes(code[..4].try_into().unwrap())) + 1
                );
                paged_same(&a, &mut b, &avm, &bvm);
            }
        }
    }
}

#[test]
fn arpl_conditional_segment_write_checks_equal_checked() {
    // SDM Vol 2A §3.3 ARPL pp. 3-75–76: only a raised RPL writes DEST;
    // a write through a non-writable segment raises #GP(0).
    // Vol 3A §§3.4.5.1, 6.4: CS is never writable; FS/GS retain cached rights.
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        for cpl in [0, 3] {
            for paged in [false, true] {
                for (prefix, segment, base, writable) in [
                    (0x2e, 0, 0, false), // CS
                    (0x3e, 1, 0, true),  // DS
                    (0x64, 4, 0, false), // read-only flat FS/GS
                    (0x65, 5, 0, false),
                    (0x64, 4, 0x1000, false), // read-only non-flat FS/GS
                    (0x65, 5, 0x1000, false),
                    (0x64, 4, 0x1000, true), // writable non-flat FS/GS
                    (0x65, 5, 0x1000, true),
                ] {
                    for raising in [false, true] {
                        let code = [prefix, 0x63, 0x05, 0, 0x60, 0, 0];
                        let make = |checked| {
                            let (vm, mut cpu) = if cpl == 3 {
                                user_setup(model, &code, paged, checked)
                            } else if paged {
                                paged_setup(model, &code, checked)
                            } else {
                                setup(model, &code, checked)
                            };
                            if segment >= 4 {
                                let cache = &mut cpu.cpu.legacy.segments[segment];
                                cache.base = base;
                                cache.limit = if base == 0 { u32::MAX } else { 0x6001 };
                                if !writable {
                                    cache.attributes &= !2;
                                }
                            }
                            cpu.cpu.gpr[0] = 3;
                            cpu.cpu.flags.cf = true;
                            cpu.cpu.flags.of = true;
                            cpu.cpu.flags.zf = !raising;
                            let dest = if raising { 0x1200u16 } else { 0x1203 };
                            vm.write_bytes(u64::from(base) + 0x6000, &dest.to_le_bytes())
                                .unwrap();
                            (vm, cpu)
                        };
                        let (avm, mut a) = make(false);
                        let (bvm, mut b) = make(true);
                        assert_eq!(class(&a, model), SegmentClass::Flat32);
                        assert_eq!(class(&b, model), SegmentClass::Checked);
                        let before = a.cpu.clone();
                        let ar = a.run_steps(&avm, 1, StepOptions::default()).unwrap();
                        let br = b.run_steps(&bvm, 1, StepOptions::default()).unwrap();
                        assert_eq!(
                            format!("{ar:?}"),
                            format!("{br:?}"),
                            "{model:?} CPL={cpl} paged={paged} prefix={prefix:02x} base={base:x} writable={writable} raising={raising}"
                        );
                        if raising && !writable {
                            let StepStop::Exit(exit @ Exit::Exception { vector: 13, .. }) =
                                ar.stop()
                            else {
                                panic!("{ar:?}");
                            };
                            assert_eq!(
                                exit.fault_metadata().unwrap().error_code(),
                                x86jit_core::fault::ErrorCode::bits(0, 0xffff)
                            );
                            assert_eq!(ar.steps(), 0);
                            assert_eq!(format!("{:?}", a.cpu), format!("{before:?}"));
                        } else {
                            assert!(matches!(ar.stop(), StepStop::Exit(Exit::BudgetExhausted)));
                            assert_eq!(ar.steps(), 1);
                            assert_eq!(a.cpu.flags.zf, raising);
                            let mut flags = before.flags;
                            flags.zf = raising;
                            assert_eq!(a.cpu.flags, flags);
                        }
                        assert_eq!(
                            fixture::value(&avm, u64::from(base) + 0x6000, 2),
                            if raising && !writable { 0x1200 } else { 0x1203 }
                        );
                        if paged {
                            let pte = get(&avm, PT + ((u64::from(base) + 0x6000) >> 12) * 4);
                            assert_eq!(pte & 0x40 != 0, raising && writable);
                        }
                        paged_same(&a, &mut b, &avm, &bvm);
                    }
                }
            }
        }
    }
}

#[test]
fn segment_write_checks_for_other_flat32_paths_equal_checked() {
    // SDM Vol 3A §6.4: write permission applies to explicit memory writes,
    // including read-modify-write, system-register and x87 destinations.
    let writes: &[&[u8]] = &[
        b"\x89\x05\x00\x60\x00\x00",     // MOV
        b"\x01\x05\x00\x60\x00\x00",     // ADD
        b"\xf0\x01\x05\x00\x60\x00\x00", // LOCK ADD
        b"\x87\x05\x00\x60\x00\x00",     // XCHG
        b"\x0f\xb1\x0d\x00\x60\x00\x00", // CMPXCHG
        b"\x0f\xab\x0d\x00\x60\x00\x00", // BTS (register index)
        b"\x0f\x94\x05\x00\x60\x00\x00", // SETZ
        b"\x8f\x05\x00\x60\x00\x00",     // POP
        b"\x0f\x01\x05\x00\x60\x00\x00", // SGDT
        b"\x0f\x01\x0d\x00\x60\x00\x00", // SIDT
        b"\x0f\x00\x05\x00\x60\x00\x00", // SLDT
        b"\x0f\x00\x0d\x00\x60\x00\x00", // STR
        b"\x0f\x01\x25\x00\x60\x00\x00", // SMSW
        b"\xd9\x3d\x00\x60\x00\x00",     // FNSTCW
        b"\xd9\x35\x00\x60\x00\x00",     // FNSTENV
        b"\xdd\x35\x00\x60\x00\x00",     // FNSAVE
    ];
    let model = CpuModel::I80486Dx;
    for cpl in [0, 3] {
        for paged in [false, true] {
            for prefix in [0x2e, 0x64, 0x65] {
                for write in writes {
                    let mut code = vec![prefix];
                    code.extend_from_slice(write);
                    let make = |checked| {
                        let (vm, mut cpu) = if cpl == 3 {
                            user_setup(model, &code, paged, checked)
                        } else if paged {
                            paged_setup(model, &code, checked)
                        } else {
                            setup(model, &code, checked)
                        };
                        for cache in &mut cpu.cpu.legacy.segments[4..] {
                            cache.base = 0x1000;
                            cache.limit = 0x60ff;
                            cache.attributes &= !2;
                        }
                        cpu.cpu.gpr[1] = 1;
                        cpu.cpu.gpr[4] = 0xa000;
                        (vm, cpu)
                    };
                    let (avm, mut a) = make(false);
                    let (bvm, mut b) = make(true);
                    assert_eq!(class(&a, model), SegmentClass::Flat32);
                    assert_eq!(class(&b, model), SegmentClass::Checked);
                    fixture::fault(&mut a, &avm, 13, Some(0));
                    fixture::fault(&mut b, &bvm, 13, Some(0));
                    paged_same(&a, &mut b, &avm, &bvm);
                }
            }
        }
    }
}

#[test]
fn paged_flat32_fault_ordinals_cross_page_atomicity_and_budgets_equal_checked() {
    // Paging design §3.3 (386 failed walks); Intel486 PRM §9.9.14 p. 9-21
    // (protection-fault A); 80386 PRM §9.8.14 (CR2, restart and error bits).
    for model in [
        CpuModel::I80386,
        CpuModel::I80486Dx,
        CpuModel::I80486DxCpuid,
    ] {
        for case in 0..4 {
            if case == 3 && model == CpuModel::I80386 {
                continue;
            }
            for ordinal in 0..4 {
                let mut code = vec![0x43; ordinal];
                let write = case != 0;
                let addr: u32 = if case == 2 { 0x6ffe } else { 0x7000 };
                code.push(if write { 0xa3 } else { 0xa1 });
                code.extend(addr.to_le_bytes());
                code.extend_from_slice(b"\x40\xf4");
                for budget in 0..8 {
                    let (avm, mut a) = paged_setup(model, &code, false);
                    let (bvm, mut b) = paged_setup(model, &code, true);
                    for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                        put(vm, PT + 7 * 4, if case == 3 { 0x7001 } else { 0 });
                        cpu.cpu.legacy.cr0 |= 1 << 16;
                        cpu.cpu.gpr[0] = 0x12345678;
                    }
                    let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
                    let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
                    assert_eq!(format!("{ar:?}"), format!("{br:?}"));
                    paged_same(&a, &mut b, &avm, &bvm);
                    assert_eq!(ar.steps(), budget.min(ordinal as u64));
                    assert_eq!(a.retired_instructions(), ar.steps());
                    assert_eq!(get(&avm, 0x6ffc), 0, "no partial operand store");
                    if budget > ordinal as u64 {
                        let StepStop::Exit(exit @ Exit::Exception { vector: 14, .. }) = ar.stop()
                        else {
                            panic!("expected #PF: {ar:?}")
                        };
                        assert_eq!(a.cpu.rip, 0x1000 + ordinal as u64);
                        assert_eq!(a.cpu.legacy.cr2, 0x7000);
                        assert_eq!(
                            exit.fault_metadata().unwrap().error_code(),
                            x86jit_core::fault::ErrorCode::bits(
                                u32::from(case == 3) | (u32::from(write) << 1),
                                7
                            )
                        );
                        assert_eq!(get(&avm, PT + 7 * 4), if case == 3 { 0x7021 } else { 0 });
                        assert_eq!(
                            get(&avm, PT + 6 * 4) & 0x40,
                            0,
                            "failed split write never sets D"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn paged_flat32_guest_pte_edit_and_invalidation_inside_a_loop_equal_checked() {
    // 80386 PRM §5.2.5; Intel486 PRM §5.3.5, INVLPG p. 26-172.
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        for invlpg in [false, true] {
            if invlpg && model == CpuModel::I80386 {
                continue;
            }
            // Warm data translation, edit its PTE through guest memory, then
            // invalidate inside the running loop before loading the new frame.
            let mut code = b"\x8b\x1d\x00\x60\x00\x00\xc7\x05".to_vec();
            code.extend((PT as u32 + 6 * 4).to_le_bytes());
            code.extend(0x9007u32.to_le_bytes());
            if invlpg {
                code.extend_from_slice(b"\x0f\x01\x3d\x00\x60\x00\x00");
            } else {
                code.extend_from_slice(b"\x0f\x22\xd8"); // MOV CR3,EAX
            }
            code.extend_from_slice(b"\x8b\x15\x00\x60\x00\x00\x4f");
            let displacement = -(code.len() as i8 + 2);
            code.extend([0x75, displacement as u8, 0xf4]);
            let (avm, mut a) = paged_setup(model, &code, false);
            let (bvm, mut b) = paged_setup(model, &code, true);
            for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                put(vm, 0x6000, 11);
                put(vm, 0x9000, 22);
                cpu.cpu.gpr[0] = PD;
                cpu.cpu.gpr[7] = 3;
            }
            let ar = a.run_steps(&avm, 100, StepOptions::default()).unwrap();
            let br = b.run_steps(&bvm, 100, StepOptions::default()).unwrap();
            assert_eq!(format!("{ar:?}"), format!("{br:?}"));
            assert!(matches!(ar.stop(), StepStop::Exit(Exit::Hlt)));
            assert_eq!(a.cpu.gpr[2], 22);
            assert_eq!(ar.steps(), 19);
            paged_same(&a, &mut b, &avm, &bvm);
        }
    }
}

#[test]
fn paged_flat32_alias_smc_and_device_retry_equal_checked() {
    use x86jit_core::{Prot, RegionKind};
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        // A linear alias of the current source frame overwrites its successor.
        let code = b"\x40\xc6\x05\x08\x90\x00\x00\xf4\x40\xf4";
        let (avm, mut a) = paged_setup(model, code, false);
        let (bvm, mut b) = paged_setup(model, code, true);
        for vm in [&avm, &bvm] {
            put(vm, PT + 9 * 4, 0x1007);
        }
        let ar = a.run_steps(&avm, 20, StepOptions::default()).unwrap();
        let br = b.run_steps(&bvm, 20, StepOptions::default()).unwrap();
        assert_eq!(format!("{ar:?}"), format!("{br:?}"));
        assert_eq!(ar.steps(), 3);
        assert_eq!(a.cpu.gpr[0], 1);
        paged_same(&a, &mut b, &avm, &bvm);

        let code = b"\x40\xa1\x00\x60\x00\x00\xa3\x00\x60\x00\x00\x40\xf4";
        let make = |checked| {
            let (mut vm, cpu) = paged_setup(model, code, checked);
            vm.unmap(0, 0x240000).unwrap();
            vm.map(0, 0x6000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.map(0x6000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
            vm.map(0x7000, 0x19000, Prot::RWX, RegionKind::Ram).unwrap();
            (vm, cpu)
        };
        let (avm, mut a) = make(false);
        let (bvm, mut b) = make(true);
        for stage in 0..3 {
            let ar = a.run_steps(&avm, 20, StepOptions::default()).unwrap();
            let br = b.run_steps(&bvm, 20, StepOptions::default()).unwrap();
            assert_eq!(format!("{ar:?}"), format!("{br:?}"));
            assert_eq!(a.retired_instructions(), b.retired_instructions());
            match stage {
                0 => {
                    assert_eq!(ar.steps(), 1);
                    assert!(matches!(ar.stop(), StepStop::Exit(Exit::MmioRead { .. })));
                    a.complete_mmio_read(41);
                    b.complete_mmio_read(41);
                }
                1 => {
                    assert_eq!(ar.steps(), 1);
                    assert!(matches!(
                        ar.stop(),
                        StepStop::Exit(Exit::MmioWrite { value: 41, .. })
                    ));
                    a.complete_mmio_write();
                    b.complete_mmio_write();
                }
                _ => {
                    assert_eq!(ar.steps(), 3);
                    assert!(matches!(ar.stop(), StepStop::Exit(Exit::Hlt)));
                    assert_eq!(a.cpu.gpr[0], 42);
                }
            }
            assert_eq!(
                a.run_steps(&avm, 0, StepOptions::default())
                    .unwrap()
                    .steps(),
                0
            );
        }
        b.cpu.legacy.segments[2].limit += 1;
        b.cpu.legacy.segments[5].attributes |= 0x80;
        assert_eq!(format!("{:?}", a.cpu), format!("{:?}", b.cpu));
    }
}

#[test]
fn paged_flat32_page_end_and_noncontiguous_straddles_equal_checked() {
    // Paging design §3.5: never speculate into an absent next page; only a
    // required instruction byte can cause fetch #PF (386 PRM §9.8.14).
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        for mapped in [false, true] {
            for ip in [0x1ffe, 0x1fff] {
                for code in [&b"\x40\xf4"[..], &b"\xb8\x78\x56\x34\x12\xf4"[..]] {
                    for budget in 0..5 {
                        let (avm, mut a) = paged_setup(model, b"", false);
                        let (bvm, mut b) = paged_setup(model, b"", true);
                        for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                            put(vm, PT + 2 * 4, if mapped { 0xb007 } else { 0 });
                            for (offset, byte) in code.iter().enumerate() {
                                let linear = ip + offset as u64;
                                let bus = if linear < 0x2000 {
                                    linear
                                } else {
                                    0xb000 + (linear & 4095)
                                };
                                vm.write_bytes(bus, &[*byte]).unwrap();
                            }
                            cpu.cpu.rip = ip;
                        }
                        let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
                        let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
                        assert_eq!(format!("{ar:?}"), format!("{br:?}"));
                        paged_same(&a, &mut b, &avm, &bvm);
                    }
                }
            }
        }
    }
}

#[test]
fn paged_flat32_read_fill_first_write_and_tlb_replacement_equal_checked() {
    // Paging design §3.2: a clean read fill cannot waive first-write D;
    // replacement must expire the source token before executing a successor.
    let code = b"\xa1\x00\x60\x00\x00\x40\xa3\x00\x60\x00\x00\xa1\x00\x00\x10\x00\x43\xf4";
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        for budget in 0..8 {
            let (avm, mut a) = paged_setup(model, code, false);
            let (bvm, mut b) = paged_setup(model, code, true);
            for vm in [&avm, &bvm] {
                // Linear page 100h folds onto code page 1's TLB slot.
                put(vm, PT + 0x100 * 4, 0xb007);
                put(vm, 0x6000, 41);
                put(vm, 0xb000, 123);
            }
            let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
            let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
            assert_eq!(format!("{ar:?}"), format!("{br:?}"));
            paged_same(&a, &mut b, &avm, &bvm);
            assert_eq!(
                get(&avm, PT + 6 * 4) & 0x60,
                if budget == 0 {
                    0
                } else if budget < 3 {
                    0x20
                } else {
                    0x60
                }
            );
            if budget >= 3 {
                assert_eq!(get(&avm, 0x6000), 42);
            }
            if budget >= 4 {
                assert_eq!(a.cpu.gpr[0], 123);
            }
        }
    }
}

#[test]
fn paged_checked_reference_and_flat32_keep_distinct_cached_blocks() {
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        let (vm, mut cpu) = paged_setup(model, b"\x40\x43\xf4", false);
        for checked in [false, true, false, true] {
            cpu.cpu.rip = 0x1000;
            cpu.cpu.gpr[0] = 0;
            cpu.cpu.gpr[3] = 0;
            if checked {
                cpu.interpret_paged_for_test(&vm).unwrap();
            } else {
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 1);
            }
            assert_eq!(cpu.cpu.rip, 0x1001);
            assert_eq!(cpu.cpu.gpr[0], 1);
            assert_eq!(cpu.cpu.gpr[3], 0);
        }
        let (_, blocks, lifts) = cpu.paging_cache_for_test();
        assert_eq!(blocks, 2);
        assert_eq!(lifts, 2);
    }
}

fn user_setup(model: CpuModel, code: &[u8], paged: bool, checked: bool) -> (Vm, Vcpu) {
    user_setup_delivery(
        model,
        code,
        paged,
        checked,
        x86jit_core::ExceptionDelivery::Outward,
    )
}
fn user_setup_delivery(
    model: CpuModel,
    code: &[u8],
    paged: bool,
    checked: bool,
    delivery: x86jit_core::ExceptionDelivery,
) -> (Vm, Vcpu) {
    let (vm, mut cpu) = setup_delivery(model, code, checked, delivery);
    cpu.cpu.legacy.cpl = 3;
    cpu.cpu.cs = 0x1b;
    cpu.cpu.ss = 0x23;
    cpu.cpu.ds = 0x23;
    cpu.cpu.es = 0x23;
    for cache in &mut cpu.cpu.legacy.segments {
        cache.attributes |= 0x60;
    }
    descriptor(&vm, 0x2018, 0, u32::MAX, 0xc0fb);
    descriptor(&vm, 0x2020, 0, u32::MAX, 0xc0f3);
    // SDM Vol 3A §§6.8.5, 6.12.1: inward delivery uses the TSS's SS0:ESP0.
    descriptor(&vm, 0x2008, 0, 0xffff, 0x409b);
    descriptor(&vm, 0x2010, 0, 0xffff, 0x4093);
    cpu.cpu.legacy.tr = x86jit_core::legacy::SystemSegment {
        selector: 0x28,
        cache: x86jit_core::SegmentCache {
            base: 0x3000,
            limit: 0x1ff,
            attributes: 0x8b,
            ..Default::default()
        },
        ..Default::default()
    };
    put(&vm, 0x3004, 0x9000);
    put(&vm, 0x3008, 0x10);
    vm.write_bytes(0x3066, &0x100u16.to_le_bytes()).unwrap();
    cpu.cpu.legacy.idtr.base = 0x2800;
    cpu.cpu.legacy.idtr.limit = 0x7ff;
    vm.write_bytes(0x2900, &[0, 0x40, 8, 0, 0, 0xee, 0, 0])
        .unwrap();
    if paged {
        enable_paging(&vm, &mut cpu);
        // System tables, handler and inner stack remain supervisor-only.
        for page in [2, 3, 4, 8] {
            put(&vm, PT + 4 * page, (page * 4096) as u32 | 3);
        }
    }
    assert_eq!(
        class(&cpu, model),
        if checked {
            SegmentClass::Checked
        } else {
            SegmentClass::Flat32
        }
    );
    (vm, cpu)
}

#[test]
fn selection_at_each_cpl_uses_cached_dpl_and_conforming_code_rules() {
    // SDM Vol 3A §§6.6–6.7, 6.8.1.1–6.8.1.2: loaded data, stack and CS rights.
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        for paged in [false, true] {
            for cpl in 1..=3 {
                let (_, mut cpu) = user_setup(model, b"\x90", paged, false);
                cpu.cpu.legacy.cpl = cpl;
                for cache in &mut cpu.cpu.legacy.segments[..4] {
                    cache.attributes = (cache.attributes & !0x60) | (u16::from(cpl) << 5);
                }
                let original = cpu.cpu.clone();
                for conforming in [false, true] {
                    for dpl in 0..=3u16 {
                        cpu.cpu = original.clone();
                        cpu.cpu.legacy.segments[0].attributes =
                            0x409b | (dpl << 5) | if conforming { 4 } else { 0 };
                        let allowed = if conforming {
                            dpl <= u16::from(cpl)
                        } else {
                            dpl == u16::from(cpl)
                        };
                        assert_eq!(class(&cpu, model) == SegmentClass::Flat32, allowed);
                    }
                }
                for segment in 1..4 {
                    for dpl in 0..=3u16 {
                        cpu.cpu = original.clone();
                        cpu.cpu.legacy.segments[segment].attributes = 0x4093 | (dpl << 5);
                        let allowed = if segment == 3 {
                            dpl == u16::from(cpl)
                        } else {
                            dpl >= u16::from(cpl)
                        };
                        assert_eq!(class(&cpu, model) == SegmentClass::Flat32, allowed);
                    }
                }
                cpu.cpu = original;
                // A changed GDT entry cannot revoke an already loaded cache.
                cpu.cpu.cs = 0;
                cpu.cpu.ds = 0;
                assert_eq!(class(&cpu, model), SegmentClass::Flat32);
            }
        }
    }
}

#[test]
fn cpl3_mid_block_budgets_faults_and_smc_equal_checked() {
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        for paged in [false, true] {
            for code in [
                &b"\x40\x43\xa3\x00\x60\x00\x00\x42\x0f\x0b"[..],
                &b"\x40\xc6\x05\x08\x10\x00\x00\x90\x40\x0f\x0b"[..],
                &b"\x40\x43\xa1\xff\xff\xff\xff\x42\x0f\x0b"[..],
            ] {
                for budget in 0..8 {
                    let (avm, mut a) = user_setup(model, code, paged, false);
                    let (bvm, mut b) = user_setup(model, code, paged, true);
                    let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
                    let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
                    assert_eq!(format!("{ar:?}"), format!("{br:?}"));
                    paged_same(&a, &mut b, &avm, &bvm);
                }
            }
        }
    }
}

#[test]
fn cpl3_privileged_and_iopl_instructions_equal_checked() {
    // SDM Vol 3A §6.9; Vol 2A CLI, STI, IN, INS; Vol 2B OUT, OUTS, POPF.
    let privileged: &[&[u8]] = &[
        b"\xf4",
        b"\x0f\x01\x15\x00\x60\x00\x00", // HLT, LGDT
        b"\x0f\x01\x1d\x00\x60\x00\x00", // LIDT
        b"\x0f\x01\xf0",
        b"\x0f\x06", // LMSW, CLTS
        b"\x0f\x00\xd0",
        b"\x0f\x00\xd8", // LLDT, LTR
        b"\x0f\x20\xc0",
        b"\x0f\x22\xc0", // MOV from/to CR0
        b"\x0f\x21\xc0",
        b"\x0f\x23\xc0", // MOV from/to DR0
        b"\x0f\x24\xf0",
        b"\x0f\x26\xf0", // MOV from/to TR6
    ];
    let sensitive: &[&[u8]] = &[
        b"\xfa",
        b"\xfb",
        b"\xe4\x20",
        b"\xe5\x20",
        b"\xe6\x20",
        b"\xe7\x20",
        b"\xec",
        b"\xed",
        b"\xee",
        b"\xef",
        b"\x66\xed",
        b"\x66\xef",
        b"\x6c",
        b"\x6d",
        b"\x6e",
        b"\x6f",
        b"\x66\x6d",
        b"\x66\x6f",
        b"\xf3\x6c",
        b"\xf3\x6f",
        b"\x9d",
        b"\x66\x9d",
    ];
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        for paged in [false, true] {
            for iopl in [0, 3] {
                for bitmap_denied in [false, true] {
                    let mut instructions = privileged.to_vec();
                    instructions.extend_from_slice(sensitive);
                    if model != CpuModel::I80386 {
                        instructions.extend_from_slice(&[
                            &b"\x0f\x08"[..],
                            &b"\x0f\x09"[..],
                            &b"\x0f\x01\x3e"[..],
                        ]);
                    }
                    for instruction in instructions {
                        let mut code = vec![0x43];
                        code.extend_from_slice(instruction);
                        code.extend_from_slice(b"\x42\x0f\x0b");
                        let (avm, mut a) = user_setup(model, &code, paged, false);
                        let (bvm, mut b) = user_setup(model, &code, paged, true);
                        for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                            cpu.cpu.rflags_system = iopl << 12;
                            cpu.cpu.flags.if_ = true;
                            cpu.cpu.gpr[1] = 1;
                            cpu.cpu.gpr[2] = 0x20;
                            cpu.cpu.gpr[6] = 0x6000;
                            cpu.cpu.gpr[7] = 0x7000;
                            cpu.cpu.gpr[4] = 0x7ff0;
                            put(vm, 0x7ff0, if iopl == 0 { 0x3002 } else { 2 }); // POPF tries to change IOPL and clear IF.
                            vm.write_bytes(0x3104, &[if bitmap_denied { 0xff } else { 0 }, 0])
                                .unwrap();
                        }
                        let ar = a.run_steps(&avm, 8, StepOptions::default()).unwrap();
                        let br = b.run_steps(&bvm, 8, StepOptions::default()).unwrap();
                        assert_eq!(
                            format!("{ar:?}"),
                            format!("{br:?}"),
                            "{instruction:02x?} IOPL={iopl} bitmap={bitmap_denied}"
                        );
                        if privileged.contains(&instruction) || instruction.starts_with(b"\x0f") {
                            assert_eq!(ar.steps(), 1);
                            let StepStop::Exit(exit @ Exit::Exception { vector: 13, .. }) =
                                ar.stop()
                            else {
                                panic!("{ar:?}")
                            };
                            assert_eq!(
                                exit.fault_metadata().unwrap().error_code(),
                                x86jit_core::fault::ErrorCode::bits(0, 0xffff)
                            );
                        }
                        if instruction == b"\x9d" || instruction == b"\x66\x9d" {
                            assert_eq!((a.cpu.rflags_system >> 12) & 3, iopl);
                            assert_eq!(a.cpu.flags.if_, iopl == 0);
                        }
                        paged_same(&a, &mut b, &avm, &bvm);
                    }
                }
            }
        }
    }
}

#[test]
fn cpl3_int_and_call_gate_return_reselect_before_successors() {
    // SDM Vol 3A §§6.8.5, 6.12.1; Vol 2A IRET and Vol 2B RET.
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        for paged in [false, true] {
            for call in [false, true] {
                let code = if call {
                    &b"\x40\x9a\x00\x00\x00\x00\x33\x00\x42\x0f\x0b"[..]
                } else {
                    &b"\x40\xcd\x20\x42\x0f\x0b"[..]
                };
                for budget in 0..8 {
                    let (avm, mut a) = user_setup(model, code, paged, false);
                    let (bvm, mut b) = user_setup(model, code, paged, true);
                    for vm in [&avm, &bvm] {
                        vm.write_bytes(0x2030, &[0, 0x40, 8, 0, 0, 0xec, 0, 0])
                            .unwrap();
                        vm.write_bytes(0x4000, if call { b"\x43\xcb" } else { b"\x43\xcf" })
                            .unwrap();
                    }
                    let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
                    let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
                    assert_eq!(format!("{ar:?}"), format!("{br:?}"));
                    assert_eq!(ar.steps(), budget.min(5));
                    assert_eq!(
                        a.cpu.legacy.cpl,
                        if (2..4).contains(&budget) { 0 } else { 3 }
                    );
                    assert_eq!(
                        class(&a, model),
                        if (2..4).contains(&budget) {
                            SegmentClass::Segmented
                        } else {
                            SegmentClass::Flat32
                        }
                    );
                    paged_same(&a, &mut b, &avm, &bvm);
                }
            }
        }
    }
}

#[test]
fn cpl3_supervisor_pages_fault_with_user_error_bits_and_model_ad() {
    // SDM Vol 3A §5.6.1; 80386 PRM §9.8.14; paging design §3.3.
    for model in [
        CpuModel::I80386,
        CpuModel::I80486Dx,
        CpuModel::I80486DxCpuid,
    ] {
        for directory in [false, true] {
            for access in 0..3 {
                let code = match access {
                    0 => &b"\x40\xa1\x00\x60\x00\x00\x42"[..],
                    1 => &b"\x40\xa3\x00\x60\x00\x00\x42"[..],
                    _ => &b"\x40\x42"[..],
                };
                let (avm, mut a) = user_setup(model, code, true, false);
                let (bvm, mut b) = user_setup(model, code, true, true);
                let page = if access == 2 || directory { 1 } else { 6 };
                for vm in [&avm, &bvm] {
                    if directory {
                        put(vm, PD, PT as u32 | 3);
                    } else {
                        put(vm, PT + page * 4, (page * 4096) as u32 | 3);
                    }
                }
                let ar = a.run_steps(&avm, 8, StepOptions::default()).unwrap();
                let br = b.run_steps(&bvm, 8, StepOptions::default()).unwrap();
                assert_eq!(format!("{ar:?}"), format!("{br:?}"));
                let StepStop::Exit(exit @ Exit::Exception { vector: 14, .. }) = ar.stop() else {
                    panic!("{ar:?}")
                };
                let write = access == 1 && !directory;
                assert_eq!(
                    exit.fault_metadata().unwrap().error_code(),
                    x86jit_core::fault::ErrorCode::bits(if write { 7 } else { 5 }, 7)
                );
                assert_eq!(a.cpu.legacy.cr2, if page == 1 { 0x1000 } else { 0x6000 });
                let ad = if model == CpuModel::I80386 { 0 } else { 0x20 };
                assert_eq!(get(&avm, PT + page * 4) & 0x60, ad);
                assert_eq!(
                    get(&avm, PD) & 0x60,
                    if directory || access == 2 { ad } else { 0x20 }
                );
                assert_eq!(ar.steps(), u64::from(!directory && access != 2));
                paged_same(&a, &mut b, &avm, &bvm);
            }
        }
    }
}

#[test]
fn cpl3_486_alignment_regime_keeps_checked_refusal() {
    for paged in [false, true] {
        for code in [&b"\x40\x43\x42"[..], &b"\xa1\x01\x60\x00\x00"[..]] {
            let (avm, mut a) = user_setup(CpuModel::I80486Dx, code, paged, false);
            let (bvm, mut b) = user_setup(CpuModel::I80486Dx, code, paged, true);
            for cpu in [&mut a, &mut b] {
                cpu.cpu.legacy.cr0 |= 1 << 18;
                cpu.cpu.rflags_system |= 1 << 18;
            }
            let ar = a.run_steps(&avm, 8, StepOptions::default()).unwrap();
            let br = b.run_steps(&bvm, 8, StepOptions::default()).unwrap();
            assert_eq!(format!("{ar:?}"), format!("{br:?}"));
            assert_eq!(ar.steps(), 0);
            assert!(matches!(
                ar.stop(),
                StepStop::Exit(Exit::UnsupportedSystemContext {
                    operation: x86jit_core::UnsupportedSystemOperation::LegacyAlignmentCheck,
                    ..
                })
            ));
            paged_same(&a, &mut b, &avm, &bvm);
        }
    }
}

#[test]
fn cpl3_block_lifted_before_am_is_not_reused_after_the_kernel_sets_it() {
    // SDM Vol 3A §7.15, Event 17—Alignment Check Exception (#AC), p. 7-53;
    // 486 PRM §4.1.1 p. 4-2: checking needs CR0.AM, EFLAGS.AC and CPL 3.
    // The kernel returns three times to the same CPL 3 block from the same
    // frame, each time after a MOV to CR0. CR0 starts with ET set, which the
    // 486 forces (486 PRM §4.1.3 p. 4-7), so the first two writes leave CR0
    // as it is: the block is lifted with AM clear, without its gate, and the
    // second entry reuses it. The third write sets only AM, and the return
    // sets AC: the block is lifted again and its misaligned load is refused,
    // whole, as every instruction of the regime is.
    // 1000: MOV EAX,[6001h]; 1006: INT 20h.
    let code = b"\x8b\x05\x01\x60\x00\x00\xcd\x20";
    let kernel = [
        &b"\x41"[..],                        // INC ECX
        b"\x0f\x20\xc0",                     // MOV EAX,CR0
        b"\x83\xf9\x03",                     // CMP ECX,3
        b"\x72\x0d",                         // JB the CR0 write
        b"\x0d\x00\x00\x04\x00",             // OR EAX,AM
        b"\x81\x4c\x24\x08\x00\x00\x04\x00", // OR dword [ESP+8],AC
        b"\x0f\x22\xc0",                     // MOV CR0,EAX
        b"\xc7\x04\x24\x00\x10\x00\x00",     // MOV dword [ESP],1000h
        b"\xcf",                             // IRETD
    ]
    .concat();
    for model in [CpuModel::I80486Dx, CpuModel::I80486DxCpuid] {
        for paged in [false, true] {
            for checked in [false, true] {
                let case = format!("{model:?} paged {paged} checked {checked}");
                let (vm, mut cpu) = user_setup(model, code, paged, checked);
                vm.write_bytes(0x4000, &kernel).unwrap();
                cpu.cpu.legacy.cr0 |= 0x10;
                cpu.cpu.gpr[1] = 0;
                let cr0 = cpu.cpu.legacy.cr0;
                assert_eq!(cr0 & (1 << 18), 0);
                let steps = |cpu: &mut Vcpu, budget| {
                    x86jit_core::Vcpu::take_machine_counters_for_test();
                    let run = cpu.run_steps(&vm, budget, StepOptions::default()).unwrap();
                    let [_, _, lifts, proofs, ..] =
                        x86jit_core::Vcpu::take_machine_counters_for_test();
                    (run, lifts, proofs)
                };
                // The first entry and the first kernel call (7 instructions).
                assert_eq!(steps(&mut cpu, 9).0.steps(), 9, "{case}");
                assert_eq!(cpu.cpu.legacy.cr0, cr0, "{case}");
                let (run, lifts, _) = steps(&mut cpu, 2);
                assert_eq!(run.steps(), 2, "{case}");
                // The IRET loads CS and SS caches with G set (C0FBh, C0F3h)
                // where the entry had it clear: an unpaged Flat32 identity
                // names neither G nor the selectors (xj-to9.56), so the
                // entry's proof serves the return; the paged and Checked
                // identities name the whole stamp and lift again.
                if paged || checked {
                    assert_ne!(lifts, 0, "{case}: the block's first lift after a return");
                } else {
                    assert_eq!(lifts, 0, "{case}: the entry's proof serves the return");
                }
                assert_eq!(steps(&mut cpu, 7).0.steps(), 7, "{case}");
                assert_eq!(cpu.cpu.legacy.cr0, cr0, "{case}");
                let (run, lifts, proofs) = steps(&mut cpu, 2);
                assert_eq!(run.steps(), 2, "{case}");
                assert_eq!((lifts, proofs != 0), (0, true), "{case}: reused unchanged");
                // The third call sets AM, and AC in the frame (9 instructions).
                assert_eq!(steps(&mut cpu, 9).0.steps(), 9, "{case}");
                assert_eq!(cpu.cpu.legacy.cr0, cr0 | 1 << 18, "{case}");
                let (run, lifts, _) = steps(&mut cpu, 4);
                assert_eq!(run.steps(), 0, "{case}");
                assert!(
                    matches!(
                        run.stop(),
                        StepStop::Exit(Exit::UnsupportedSystemContext {
                            operation:
                                x86jit_core::UnsupportedSystemOperation::LegacyAlignmentCheck,
                            ..
                        })
                    ),
                    "{case}: {:?}",
                    run.stop()
                );
                assert_ne!(lifts, 0, "{case}: lifted again once AM is set");
                assert_eq!(cpu.cpu.rip, 0x1000, "{case}");
                assert_eq!(cpu.cpu.legacy.cpl, 3, "{case}");
                assert_ne!(cpu.cpu.rflags_system & (1 << 18), 0, "{case}");
            }
        }
    }
}

#[test]
fn cpl3_task_gate_reselects_before_the_incoming_task() {
    // SDM Vol 3A §§8.2.1, 8.3, 8.6: a task gate loads the incoming
    // TSS's CS/CPL and CR3; the interrupted instruction has no successor.
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        for paged in [false, true] {
            for budget in 1..=4 {
                let (avm, mut a) = user_setup(model, b"\x42\xcd\x20\x42", paged, false);
                let (bvm, mut b) = user_setup(model, b"\x42\xcd\x20\x42", paged, true);
                for vm in [&avm, &bvm] {
                    descriptor(vm, 0x2028, 0x3000, 0x1ff, 0x8b);
                    descriptor(vm, 0x2038, 0x5000, 0x67, 0x89);
                    vm.write_bytes(0x2900, &[0, 0, 0x38, 0, 0, 0xe5, 0, 0])
                        .unwrap();
                    for (offset, value) in [
                        (0x1c, PD as u32),
                        (0x20, 0x4000),
                        (0x24, 2),
                        (0x28, 41),
                        (0x38, 0x9000),
                        (0x4c, 8),
                        (0x50, 16),
                    ] {
                        put(vm, 0x5000 + offset, value);
                    }
                    vm.write_bytes(0x4000, b"\x40\x0f\x0b").unwrap();
                }
                let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
                let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
                assert_eq!(format!("{ar:?}"), format!("{br:?}"));
                assert_eq!(ar.steps(), budget.min(3));
                assert_eq!(a.cpu.gpr[2], if budget == 1 { 1 } else { 0 });
                if budget >= 2 {
                    assert_eq!(a.cpu.legacy.cpl, 0);
                    assert_eq!(a.cpu.legacy.tr.selector, 0x38);
                    assert_eq!(class(&a, model), SegmentClass::Segmented);
                    assert_eq!(a.cpu.gpr[0], if budget == 2 { 41 } else { 42 });
                }
                // The incoming task has null ES in both executions.
                b.cpu.legacy.segments[2].limit = a.cpu.legacy.segments[2].limit;
                paged_same(&a, &mut b, &avm, &bvm);
            }
        }
    }
}

#[test]
fn cpl3_exception_delivery_reselects_before_the_ring0_handler() {
    // SDM Vol 3A §§6.12.1, 6.15: #DE ignores gate DPL and retires nothing.
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        for paged in [false, true] {
            let code = b"\x40\xf7\xf1\x42";
            let (avm, mut a) = user_setup_delivery(
                model,
                code,
                paged,
                false,
                x86jit_core::ExceptionDelivery::InGuest,
            );
            let (bvm, mut b) = user_setup_delivery(
                model,
                code,
                paged,
                true,
                x86jit_core::ExceptionDelivery::InGuest,
            );
            for vm in [&avm, &bvm] {
                vm.write_bytes(0x2800, &[0, 0x40, 8, 0, 0, 0x8e, 0, 0])
                    .unwrap();
                vm.write_bytes(0x4000, b"\x43\x90").unwrap();
            }
            let ar = a.run_steps(&avm, 2, StepOptions::default()).unwrap();
            let br = b.run_steps(&bvm, 2, StepOptions::default()).unwrap();
            assert_eq!(format!("{ar:?}"), format!("{br:?}"));
            assert_eq!(ar.steps(), 2);
            assert_eq!(a.cpu.legacy.cpl, 0);
            assert_eq!(a.cpu.rip, 0x4001);
            assert_eq!(a.cpu.gpr[2], 0);
            assert_eq!(a.cpu.gpr[3], 1);
            assert_eq!(class(&a, model), SegmentClass::Segmented);
            paged_same(&a, &mut b, &avm, &bvm);
        }
    }
}

#[test]
fn cpl1_and_cpl2_flat32_operands_keep_supervisor_paging_rights() {
    // SDM Vol 3A §5.6.1: CPL 0, 1 and 2 use supervisor page permissions.
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        for paged in [false, true] {
            for cpl in [1, 2] {
                let code = b"\x40\xa3\x00\x60\x00\x00\xa1\x00\x60\x00\x00\x0f\x0b";
                let (avm, mut a) = user_setup(model, code, paged, false);
                let (bvm, mut b) = user_setup(model, code, paged, true);
                for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                    cpu.cpu.legacy.cpl = cpl;
                    for cache in &mut cpu.cpu.legacy.segments {
                        cache.attributes = (cache.attributes & !0x60) | (u16::from(cpl) << 5);
                    }
                    if paged {
                        put(vm, PT + 6 * 4, 0x6003);
                    }
                }
                assert_eq!(class(&a, model), SegmentClass::Flat32);
                assert_eq!(class(&b, model), SegmentClass::Checked);
                let ar = a.run_steps(&avm, 8, StepOptions::default()).unwrap();
                let br = b.run_steps(&bvm, 8, StepOptions::default()).unwrap();
                assert_eq!(ar.steps(), 3);
                assert_eq!(a.cpu.gpr[0], 1);
                assert_eq!(format!("{ar:?}"), format!("{br:?}"));
                paged_same(&a, &mut b, &avm, &bvm);
            }
        }
    }
}

#[test]
fn reserved_bit21_preserves_flat32_selection() {
    for model in [
        CpuModel::I80386,
        CpuModel::I80486Dx,
        CpuModel::I80486DxCpuid,
    ] {
        let (vm, mut cpu) = setup(model, b"\x90", false);
        for cache in &mut cpu.cpu.legacy.segments {
            cache.attributes |= 0x2000;
        }
        assert_eq!(class(&cpu, model), SegmentClass::Flat32);
        fixture::steps(&mut cpu, &vm, 1);
    }
}

#[test]
fn paged_context_instructions_keep_translation_and_reuse_blocks() {
    // SDM Vol 2A CALL/IRET/MOV, Vol 2B POPF/RET; paging design §3.2.
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        for event in 0..4 {
            let code: &[u8] = match event {
                0 => b"\x8e\xd8\x8e\xc0",             // MOV DS,AX; MOV ES,AX
                1 => b"\x9a\x00\x11\x00\x00\x08\x00", // CALL FAR
                2 => b"\x50\x9d",                     // PUSH EAX; POPFD (change IOPL)
                3 => b"\xcf",                         // IRETD (change IOPL)
                _ => unreachable!(),
            };
            let (vm, mut cpu) = paged_setup(model, code, false);
            vm.write_bytes(0x1100, b"\xcb").unwrap();
            let mut warm = None;
            // Initial segment-cache descriptor encoding can normalize on reload.
            // Two full passes warm both the initial and normalized contexts.
            for round in 0..4 {
                cpu.cpu.rip = 0x1000;
                cpu.cpu.gpr[4] = 0x7000;
                cpu.cpu.gpr[0] = if event == 0 { 16 } else { 0x3002 };
                cpu.cpu.rflags_system = 0;
                if event == 3 {
                    put(&vm, 0x7000, 0x1100);
                    put(&vm, 0x7004, 8);
                    put(&vm, 0x7008, 0x3002);
                }
                let count = if event == 3 { 1 } else { 2 };
                fixture::steps(&mut cpu, &vm, count);
                if event >= 2 {
                    assert_eq!(cpu.cpu.rflags_system & 0x3000, 0x3000);
                }
                let proof = cpu.paging_cache_for_test();
                assert!(proof.1 >= 1);
                if round == 1 {
                    warm = Some(proof);
                } else if round > 1 {
                    assert_eq!(Some(proof), warm, "event {event}");
                }
            }
        }
    }
}

#[test]
fn paged_smc_expires_blocks_in_every_retained_context_without_flushing_tlb() {
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        let (vm, mut cpu) = paged_setup(model, b"\x40\xeb\xfd", false);
        for flags in [0, 0x3000] {
            cpu.cpu.rip = 0x1000;
            cpu.cpu.rflags_system = flags;
            fixture::steps(&mut cpu, &vm, 1);
        }
        let before = cpu.paging_cache_for_test();
        assert_eq!(before.1, 2);
        vm.write_bytes(0x1000, b"\x43").unwrap(); // INC EBX in both contexts.
        for flags in [0, 0x3000] {
            cpu.cpu.rip = 0x1000;
            cpu.cpu.rflags_system = flags;
            let eax = cpu.cpu.gpr[0];
            let ebx = cpu.cpu.gpr[3];
            fixture::steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.gpr[0], eax);
            assert_eq!(cpu.cpu.gpr[3], ebx + 1);
        }
        let after = cpu.paging_cache_for_test();
        assert_eq!(after.0, before.0);
        assert_eq!(after.1, 2);
        assert_eq!(after.2, before.2 + 2);
    }
}

#[test]
fn paged_privilege_gate_round_trips_reuse_prior_context_blocks() {
    // SDM Vol 3A §§6.8.5, 6.12.1; Vol 2A IRET and Vol 2B RET.
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        for call in [false, true] {
            let code: &[u8] = if call {
                b"\x40\x9a\x00\x00\x00\x00\x33\x00\x42\x0f\x0b"
            } else {
                b"\x40\xcd\x20\x42\x0f\x0b"
            };
            let (vm, mut cpu) = user_setup(model, code, true, false);
            vm.write_bytes(0x2030, &[0, 0x40, 8, 0, 0, 0xec, 0, 0])
                .unwrap();
            vm.write_bytes(0x4000, if call { b"\x43\xcb" } else { b"\x43\xcf" })
                .unwrap();
            let mut warm = None;
            for round in 0..4 {
                cpu.cpu.rip = 0x1000;
                fixture::steps(&mut cpu, &vm, 5);
                assert_eq!(cpu.cpu.legacy.cpl, 3);
                let proof = cpu.paging_cache_for_test();
                assert!(proof.1 > 0, "{proof:?}");
                if round == 1 {
                    warm = Some(proof);
                } else if round > 1 {
                    assert_eq!(Some(proof), warm, "call={call}");
                }
            }
        }
    }
}

/// One witness case of `a_reduced_identity_reuses_its_proofs_exactly`: the
/// code, its privilege, its step budget and the two states.
struct ReuseCase {
    name: &'static str,
    code: &'static [u8],
    user: bool,
    budget: u64,
    a: fn(&Vm, &mut Vcpu),
    b: fn(&Vm, &mut Vcpu),
}

/// A second, protected 486 GDT at 5800h: the first one's 16 entries with
/// selector 20h's limit at 1234h and selector 08h not present.
fn second_gdt(vm: &Vm, cpu: &mut Vcpu) {
    let mut table = [0; 0x80];
    vm.read_bytes(0x2000, &mut table).unwrap();
    vm.write_bytes(0x5800, &table).unwrap();
    descriptor(vm, 0x5820, 0, 0x1234, 0x40f3);
    descriptor(vm, 0x5808, 0, u32::MAX, 0xc01b);
    cpu.cpu.legacy.gdtr.base = 0x5800;
}

// xj-to9.56: an unpaged Flat32 identity leaves out the selectors, GDTR,
// IDTR, LDTR, TR, CR3, CR0 other than PE and AM, IOPL, NT and the flat
// caches' bits other than D/B. Every instruction that reads one of them
// reads it live: lifted under state A, the same proof runs under state B,
// which changes one such field, and must do exactly what a vCPU that never
// saw A does (a fresh lift). The cases: the IOPL checks of CLI, STI and
// POPF (SDM Vol 1 §3.4.3.3, §20.5.1); the I/O permission bit map of IN,
// OUT and INS, found through TR (Vol 1 §20.5.2); the task return of IRET
// under NT (Vol 3A §10.4); the IDT limit of INT (Vol 3A §7.10); MOV from a
// segment register, SLDT, STR, SIDT and SGDT (Vol 2B); LAR and LSL through
// the GDT (Vol 3A §6.10); a far JMP and LDS through the GDT; MOV from CR3
// and CR0.TS for an x87 instruction (Vol 3A §2.5); and loads, stores and
// stack operations through caches with other G and A bits. Each runs
// under Flat32 and Segmented.
#[test]
fn a_reduced_identity_reuses_its_proofs_exactly() {
    let cases = [
        ReuseCase {
            name: "CLI/STI, IOPL",
            code: b"\xfa\xfb\x90",
            user: true,
            budget: 3,
            a: |_, c| c.cpu.rflags_system |= 0x3000,
            b: |_, c| c.cpu.rflags_system &= !0x3000,
        },
        ReuseCase {
            name: "POPF, IOPL",
            // PUSHFD; POP EAX; XOR EAX,200h; PUSH EAX; POPFD; PUSHFD; POP EBX.
            code: b"\x9c\x58\x35\x00\x02\x00\x00\x50\x9d\x9c\x5b",
            user: true,
            budget: 7,
            a: |_, c| c.cpu.rflags_system |= 0x3000,
            b: |_, c| c.cpu.rflags_system &= !0x3000,
        },
        ReuseCase {
            name: "IN, TR",
            code: b"\xe4\x60",
            user: true,
            budget: 1,
            a: |vm, c| {
                // A TSS at 5000h whose bitmap denies port 60h.
                vm.write_bytes(0x5066, &0x100u16.to_le_bytes()).unwrap();
                vm.write_bytes(0x510c, &[0xff]).unwrap();
                c.cpu.legacy.tr.cache.base = 0x5000;
            },
            b: |_, _| {},
        },
        ReuseCase {
            name: "OUT, TR",
            code: b"\xe6\x60",
            user: true,
            budget: 1,
            a: |vm, c| {
                vm.write_bytes(0x5066, &0x100u16.to_le_bytes()).unwrap();
                vm.write_bytes(0x510c, &[0xff]).unwrap();
                c.cpu.legacy.tr.cache.base = 0x5000;
            },
            b: |_, _| {},
        },
        ReuseCase {
            name: "INSB, TR",
            code: b"\x6c",
            user: true,
            budget: 1,
            a: |vm, c| {
                vm.write_bytes(0x5066, &0x100u16.to_le_bytes()).unwrap();
                vm.write_bytes(0x510c, &[0xff]).unwrap();
                c.cpu.legacy.tr.cache.base = 0x5000;
                c.cpu.gpr[2] = 0x60;
                c.cpu.gpr[7] = 0x6000;
            },
            b: |_, c| {
                c.cpu.gpr[2] = 0x60;
                c.cpu.gpr[7] = 0x6000;
            },
        },
        ReuseCase {
            name: "IRETD, NT",
            // A frame at 7FF4h: EIP 1100h, CS 8, EFLAGS 2; 1100h: NOP.
            code: b"\xcf",
            user: false,
            budget: 1,
            a: |vm, c| {
                vm.write_bytes(0x1100, b"\x90").unwrap();
                put(vm, 0x7ff4, 0x1100);
                put(vm, 0x7ff8, 8);
                put(vm, 0x7ffc, 2);
                c.cpu.gpr[4] = 0x7ff4;
            },
            b: |_, c| {
                c.cpu.gpr[4] = 0x7ff4;
                c.cpu.rflags_system |= 0x4000;
            },
        },
        ReuseCase {
            name: "INT 20h, IDTR",
            code: b"\xcd\x20",
            user: true,
            budget: 1,
            a: |_, _| {},
            b: |_, c| c.cpu.legacy.idtr.limit = 0xff,
        },
        ReuseCase {
            name: "MOV r32,Sreg, selectors",
            // MOV EAX,DS; MOV EBX,ES; MOV ECX,SS; MOV EDX,CS.
            code: b"\x8c\xd8\x8c\xc3\x8c\xd1\x8c\xca",
            user: true,
            budget: 4,
            a: |_, _| {},
            b: |_, c| {
                c.cpu.ds = 0x2b;
                c.cpu.es = 0x33;
                c.cpu.ss = 0x3b;
                c.cpu.cs = 0x43;
            },
        },
        ReuseCase {
            name: "SLDT/STR/SIDT/SGDT, LDTR TR IDTR GDTR",
            // SLDT EAX; STR EBX; SIDT [6000h]; SGDT [6008h].
            code:
                b"\x0f\x00\xc0\x0f\x00\xcb\x0f\x01\x0d\x00\x60\x00\x00\x0f\x01\x05\x08\x60\x00\x00",
            user: true,
            budget: 4,
            a: |_, _| {},
            b: |_, c| {
                c.cpu.legacy.ldtr.selector = 0x38;
                c.cpu.legacy.tr.selector = 0x40;
                c.cpu.legacy.idtr.base = 0x2c00;
                c.cpu.legacy.gdtr.limit = 0x7f;
                c.cpu.legacy.gdtr.base = 0x5800;
            },
        },
        ReuseCase {
            name: "LAR/LSL, GDTR",
            // MOV EAX,23h; LAR EBX,AX; LSL ECX,AX.
            code: b"\xb8\x23\x00\x00\x00\x0f\x02\xd8\x0f\x03\xc8",
            user: true,
            budget: 3,
            a: |_, _| {},
            b: second_gdt,
        },
        ReuseCase {
            name: "JMP FAR, GDTR",
            // JMP FAR 0008h:1100h; 1100h: NOP.
            code: b"\xea\x00\x11\x00\x00\x08\x00",
            user: false,
            budget: 2,
            a: |vm, _| vm.write_bytes(0x1100, b"\x90").unwrap(),
            b: second_gdt,
        },
        ReuseCase {
            name: "LDS, GDTR",
            // LDS EAX,[6010h], the pointer 0020h:1234h.
            code: b"\xc5\x05\x10\x60\x00\x00",
            user: true,
            budget: 1,
            a: |vm, _| {
                put(vm, 0x6010, 0x1234);
                vm.write_bytes(0x6014, &0x23u16.to_le_bytes()).unwrap();
            },
            b: second_gdt,
        },
        ReuseCase {
            name: "MOV r32,CR3, CR3",
            code: b"\x0f\x20\xd8",
            user: false,
            budget: 1,
            a: |_, _| {},
            b: |_, c| c.cpu.legacy.cr3 = 0x7000,
        },
        ReuseCase {
            name: "FLD1, CR0.TS",
            code: b"\xd9\xe8\x90",
            user: false,
            budget: 2,
            a: |_, _| {},
            b: |_, c| c.cpu.legacy.cr0 |= 1 << 3,
        },
        ReuseCase {
            name: "loads and stores, G and A bits",
            // MOV EAX,[6000h]; MOV [6004h],EAX; PUSH EAX; POP EBX.
            code: b"\x8b\x05\x00\x60\x00\x00\x89\x05\x04\x60\x00\x00\x50\x5b",
            user: true,
            budget: 4,
            a: |vm, _| put(vm, 0x6000, 0x1234_5678),
            b: |_, c| {
                for cache in &mut c.cpu.legacy.segments[1..4] {
                    cache.attributes = (cache.attributes ^ 0x8000) & !1;
                }
            },
        },
    ];
    // Segmented (64 KiB data and stack limits) as well as Flat32: its
    // identity also names the caches, so only the G and A case lifts again.
    for segmented in [false, true] {
        let class_of = if segmented {
            SegmentClass::Segmented
        } else {
            SegmentClass::Flat32
        };
        for case in &cases {
            let name = format!("{} ({class_of:?})", case.name);
            let caches = case.name.contains("G and A");
            let (vm, mut warm) = if case.user {
                user_setup(CpuModel::I80486Dx, case.code, false, false)
            } else {
                setup(CpuModel::I80486Dx, case.code, false)
            };
            warm.cpu.legacy.cr0 |= 0x10;
            if segmented {
                for cache in &mut warm.cpu.legacy.segments[1..4] {
                    cache.limit = 0xffff;
                }
            }
            let start = warm.cpu.clone();
            (case.a)(&vm, &mut warm);
            assert_eq!(class(&warm, CpuModel::I80486Dx), class_of, "{name}");
            // State A lifts the proofs.
            warm.run_steps(&vm, case.budget, StepOptions::default())
                .unwrap();
            // State B: the same code and registers, one dropped field changed.
            let mut cold = vm.new_vcpu();
            cold.cpu = start;
            (case.b)(&vm, &mut cold);
            assert_eq!(class(&cold, CpuModel::I80486Dx), class_of, "{name}");
            let mut memory = vec![0; 0x10000];
            vm.read_bytes(0, &mut memory).unwrap();
            warm.cpu = cold.cpu.clone();
            Vcpu::take_machine_counters_for_test();
            let warm_run = warm.run_steps(&vm, case.budget, StepOptions::default());
            let [_, _, lifts, ..] = Vcpu::take_machine_counters_for_test();
            assert_eq!(
                lifts == 0,
                !(segmented && caches),
                "{name}: state B reuses state A's proofs ({lifts} lifts)"
            );
            let mut warm_memory = vec![0; 0x10000];
            vm.read_bytes(0, &mut warm_memory).unwrap();
            // A vCPU that never ran under state A, on the same memory.
            vm.write_bytes(0, &memory).unwrap();
            let cold_run = cold.run_steps(&vm, case.budget, StepOptions::default());
            let [_, _, lifts, ..] = Vcpu::take_machine_counters_for_test();
            assert_ne!(lifts, 0, "{name}: the fresh vCPU lifts");
            let mut cold_memory = vec![0; 0x10000];
            vm.read_bytes(0, &mut cold_memory).unwrap();
            assert_eq!(format!("{warm_run:?}"), format!("{cold_run:?}"), "{name}");
            assert_eq!(
                format!("{:?}", warm.cpu),
                format!("{:?}", cold.cpu),
                "{name}"
            );
            assert_eq!(warm_memory, cold_memory, "{name}");
        }
    }
}
