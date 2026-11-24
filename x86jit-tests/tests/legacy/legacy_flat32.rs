//! Flat32 proof and exact interpreted blocks (decision-27, PG=0 checkpoint).
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
        for field in 0..5 {
            cpu.cpu = original.clone();
            match field {
                0 => cpu.cpu.legacy.segments[0].attributes &= !0x4000,
                1 => cpu.cpu.legacy.segments[3].attributes &= !0x4000,
                2 => cpu.cpu.legacy.cpl = 3,
                3 => cpu.cpu.legacy.cr0 |= 1 << 31,
                _ => cpu.cpu.legacy.cr0 = 0,
            }
            assert_ne!(class(&cpu, model), SegmentClass::Flat32);
        }
        cpu.cpu = original;
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
