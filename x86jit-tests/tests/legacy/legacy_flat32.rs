//! Flat32 proof and exact interpreted blocks (decision-27 order 2).
#[path = "../support/descriptor_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::descriptor;
use x86jit_core::{
    derive_legacy_context, CpuModel, Exit, SegmentClass, StepOptions, StepStop, Vcpu, Vm,
};

fn setup(model: CpuModel, code: &[u8], checked: bool) -> (Vm, Vcpu) {
    let (vm, mut cpu) = fixture::setup(model, code);
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
        cpu.cpu.legacy.segments[2].limit -= 1;
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
                    4 => cache.attributes ^= 4,
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
        assert_eq!(class(&a, model), SegmentClass::Checked);
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
            assert_eq!(class(&cpu, model), SegmentClass::Checked);
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
            assert_eq!(class(&cpu, model), SegmentClass::Checked);
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
    assert_eq!(class(&cpu, model), SegmentClass::Checked);

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
    let code = b"\xa1\x00\x60\x00\x00\x40\xa3\x00\x60\x00\x00\xa1\x00\x10\x04\x00\x43\xf4";
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        for budget in 0..8 {
            let (avm, mut a) = paged_setup(model, code, false);
            let (bvm, mut b) = paged_setup(model, code, true);
            for vm in [&avm, &bvm] {
                // Linear page 65 collides with code page 1's TLB slot.
                put(vm, PT + 65 * 4, 0xb007);
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
