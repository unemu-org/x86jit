//! Direct/Checked boundary and shared translation witnesses (decision-23 §5).
use std::sync::Arc;
use x86jit_core::{
    derive_legacy_context, BlockKey, CachedBlock, CpuGeneration, CpuMode, CpuModel, CpuState,
    ExceptionDelivery, Exit, IrOp, LegacyCpuModel, Prot, RegionKind, SegmentClass, StepOptions,
    StepStop, Vcpu, Vm, VmConfig,
};

fn setup(model: CpuModel, code: &[u8]) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(0x10000));
    vm.set_cpu_model(model);
    vm.set_exception_delivery(ExceptionDelivery::Outward);
    vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x3000, 0x1000, Prot::R, RegionKind::Ram).unwrap();
    vm.map(0x8000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[4] = 0x8800;
    (vm, cpu)
}

fn block(vm: &Vm, ip: u64) -> Arc<x86jit_core::IrBlock> {
    let CachedBlock::Interpreted(ir) = vm
        .cache
        .get(BlockKey::machine_direct(ip, 0, true))
        .or_else(|| vm.cache.get(BlockKey::machine_direct(ip, 0, false)))
        .unwrap()
    else {
        panic!("Machine block must remain interpreted")
    };
    ir
}

#[test]
fn direct_proof_checks_all_six_caches_and_ignores_the_board_line() {
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let model = LegacyCpuModel {
            generation,
            model_semantics_id: 1,
        };
        let ordinary = CpuState::legacy_real_entry([0x100, 0x200, 0x300, 0x400, 0x500, 0x600]);
        for enabled in [0, 1] {
            let mut cpu = ordinary.clone();
            cpu.legacy.a20_enabled = enabled;
            assert_eq!(
                derive_legacy_context(model, &cpu).unwrap().segment_class,
                SegmentClass::Direct
            );
            for segment in 0..6 {
                for field in 0..4 {
                    let mut changed = cpu.clone();
                    let cache = &mut changed.legacy.segments[segment];
                    match field {
                        0 => cache.base ^= 0x10,
                        1 => cache.limit -= 1,
                        2 => cache.attributes ^= 1,
                        _ => cache.unusable = 1,
                    }
                    let context = derive_legacy_context(model, &changed);
                    if (segment == 0 || segment == 3) && field == 3 {
                        assert!(context.is_err());
                        continue;
                    }
                    assert_eq!(
                        context.unwrap().segment_class,
                        SegmentClass::Checked,
                        "{generation:?} segment {segment}, field {field}"
                    );
                }
            }
            for reset in [false, true] {
                let mut changed = cpu.clone();
                if reset {
                    changed.legacy.reset_fetch = 1;
                } else {
                    changed.legacy.cr0 = 1;
                }
                assert_eq!(
                    derive_legacy_context(model, &changed)
                        .unwrap()
                        .segment_class,
                    SegmentClass::Checked
                );
            }
        }
    }
}

#[test]
fn direct_blocks_are_multi_instruction_shared_and_keyed_by_a20() {
    for model in [CpuModel::I80286, CpuModel::I80386] {
        let (vm, mut cpu) = setup(model, b"\x43\xa1\x00\x80\x42\xf4");
        cpu.cpu.legacy.segments[1].limit = 0xfffe;
        let first = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(first.steps(), 1);
        assert!(vm
            .cache
            .get(BlockKey::new(0x1000, CpuMode::Real16))
            .is_none());
        cpu.cpu.legacy.segments[1].limit = 0xffff;
        cpu.cpu.rip = 0x1000;
        let first = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(first.steps(), 1);
        let ir = block(&vm, 0x1000);
        assert_eq!(ir.icount, 4);
        assert_eq!(
            ir.legacy_context.unwrap().segment_class,
            SegmentClass::Direct
        );
        assert!(ir.ops.iter().all(|op| !matches!(
            op,
            IrOp::Legacy(x86jit_core::legacy::LegacyOp::Address { .. })
        )));
        // The temp verifier exists only in debug builds of the core (`ir.rs`).
        #[cfg(debug_assertions)]
        assert_eq!(ir.verify_temp_defs(), Ok(()));
        for enabled in [false, true, false] {
            cpu.cpu.rip = 0x1000;
            cpu.set_a20_enabled(enabled);
            assert_eq!(
                cpu.run_steps(&vm, 1, StepOptions::default())
                    .unwrap()
                    .steps(),
                1
            );
            let key = BlockKey::machine_direct(0x1000, 0, enabled);
            assert!(vm.cache.get(key).is_some());
        }
        let mut second = vm.new_vcpu();
        second.cpu.rip = 0x1000;
        assert_eq!(
            second
                .run_steps(&vm, 1, StepOptions::default())
                .unwrap()
                .steps(),
            1
        );
        assert!(Arc::ptr_eq(&ir, &block(&vm, 0x1000)));
    }
}

#[test]
fn fault_after_a_direct_prefix_keeps_only_the_faulting_instruction_for_retry() {
    for model in [CpuModel::I80286, CpuModel::I80386] {
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        vm.set_cpu_model(model);
        vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        vm.write_bytes(0x1000, b"\x43\xa1\x00\x30\xf4").unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x1000;
        let run = cpu.run_steps(&vm, 3, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 1);
        assert!(matches!(run.stop(), StepStop::Exit(Exit::MmioRead { .. })));
        assert_eq!(
            (cpu.cpu.rip, cpu.cpu.gpr[3], cpu.retired_instructions()),
            (0x1001, 1, 1)
        );
        vm.write_bytes(0x1000, b"\x42").unwrap();
        cpu.complete_mmio_read(0x1234);
        let run = cpu.run_steps(&vm, 3, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 2);
        assert!(matches!(run.stop(), StepStop::Exit(Exit::Hlt)));
        assert_eq!(
            (
                cpu.cpu.gpr[0],
                cpu.cpu.gpr[3],
                cpu.cpu.gpr[2],
                cpu.retired_instructions()
            ),
            (0x1234, 1, 0, 3)
        );
    }
}

#[test]
fn direct_store_refetches_a_modified_successor_in_the_same_block() {
    for model in [CpuModel::I80286, CpuModel::I80386] {
        let (vm, mut cpu) = setup(model, b"\x42\xc6\x06\x07\x10\x2a\xb0\x00\xf4");
        let run = cpu.run_steps(&vm, 4, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 4);
        assert!(matches!(run.stop(), StepStop::Exit(Exit::Hlt)));
        assert_eq!(
            (cpu.cpu.gpr[0], cpu.cpu.gpr[2], cpu.retired_instructions()),
            (42, 1, 4)
        );
    }
}

#[test]
fn direct_scalar_store_fault_keeps_the_pre_instruction_flags() {
    // SDM Vol 3A §7.5: the faulting instruction does not commit its flags.
    for code in [&b"\x11\x06\x00\x30"[..], &b"\x0f\xba\x3e\x00\x30\x03"[..]] {
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        vm.set_cpu_model(CpuModel::I80386);
        vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        vm.write_bytes(0x1000, code).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x1000;
        cpu.cpu.flags.cf = true;
        cpu.cpu.flags.of = true;
        let before = cpu.cpu.flags;
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(matches!(run.stop(), StepStop::Exit(Exit::MmioRead { .. })));
        cpu.complete_mmio_read(0);
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(matches!(run.stop(), StepStop::Exit(Exit::MmioWrite { .. })));
        assert_eq!(cpu.cpu.flags, before);
        assert_eq!(cpu.cpu.rip, 0x1000);
        let ir = block(&vm, 0x1000);
        assert!(!ir
            .ops
            .iter()
            .any(|op| matches!(op, IrOp::Legacy(x86jit_core::legacy::LegacyOp::Checkpoint))));
    }
}

#[test]
fn restartable_stack_and_control_helpers_need_no_checkpoint() {
    for code in [
        &b"\xe2\xfe"[..],
        &b"\x50"[..],
        &b"\x58"[..],
        &b"\xe8\x00\x00"[..],
        &b"\xc3"[..],
    ] {
        let (vm, mut cpu) = setup(CpuModel::I80386, code);
        assert_eq!(
            cpu.run_steps(&vm, 1, StepOptions::default())
                .unwrap()
                .steps(),
            1
        );
        let ir = block(&vm, 0x1000);
        assert!(matches!(
            ir.ops.last(),
            Some(IrOp::Legacy(x86jit_core::legacy::LegacyOp::SharedWalk))
        ));
        assert!(!ir
            .ops
            .iter()
            .any(|op| matches!(op, IrOp::Legacy(x86jit_core::legacy::LegacyOp::Checkpoint))));
    }
}

#[test]
fn direct_unaligned_strings_fault_after_the_completed_prefix() {
    // SDM Vol 2B §4.3 MOVS, Real-Address Mode Exceptions: a word
    // beginning at FFFFh faults before changing either index or the count.
    for model in [CpuModel::I80286, CpuModel::I80386] {
        let mut vm = Vm::new(VmConfig::flat(0x20000));
        vm.set_cpu_model(model);
        vm.set_exception_delivery(ExceptionDelivery::Outward);
        vm.map(0, 0x20000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, b"\xf3\xa5\xf4").unwrap();
        vm.write_bytes(0xfffd, &[0x12, 0x34]).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x1000;
        cpu.cpu.gpr[1] = 2;
        cpu.cpu.gpr[6] = 0xfffd;
        cpu.cpu.gpr[7] = 0x8001;
        let run = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 1);
        assert!(matches!(
            run.stop(),
            StepStop::Exit(Exit::Exception { vector: 13, .. })
        ));
        assert_eq!(
            (cpu.cpu.rip, cpu.cpu.gpr[1], cpu.cpu.gpr[6], cpu.cpu.gpr[7]),
            (0x1000, 1, 0xffff, 0x8003)
        );
        assert_eq!(cpu.retired_instructions(), 0);
    }
}

#[test]
fn shared_strings_preserve_mmio_iteration_ownership() {
    // SDM Vol 2B §4.3 REP/MOVS: expose only completed iterations;
    // decision-16 §3 retains each device answer until its iteration commits.
    for model in [CpuModel::I80286, CpuModel::I80386] {
        let (mut vm, mut cpu) = setup(model, b"\xf3\xa5\xf4");
        vm.unmap(0x3000, 0x1000).unwrap();
        vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        cpu.cpu.gpr[1] = 2;
        cpu.cpu.gpr[6] = 0x3000;
        cpu.cpu.gpr[7] = 0x8000;
        let run = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(matches!(
            run.stop(),
            StepStop::Exit(Exit::MmioRead { addr: 0x3000, .. })
        ));
        assert!(block(&vm, 0x1000).ops.iter().any(|op| matches!(
            op,
            IrOp::Legacy(x86jit_core::legacy::LegacyOp::String { .. })
        )));
        cpu.complete_mmio_read(0xabcd);
        assert_eq!(
            cpu.run_steps(&vm, 1, StepOptions::default())
                .unwrap()
                .steps(),
            1
        );
        assert_eq!(
            (
                cpu.cpu.gpr[1],
                cpu.cpu.gpr[6],
                cpu.cpu.gpr[7],
                cpu.retired_instructions()
            ),
            (1, 0x3002, 0x8002, 0)
        );
        let run = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(matches!(
            run.stop(),
            StepStop::Exit(Exit::MmioRead { addr: 0x3002, .. })
        ));
        cpu.complete_mmio_read(0x55ee);
        let run = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 2);
        assert!(matches!(run.stop(), StepStop::Exit(Exit::Hlt)));
        assert_eq!(cpu.retired_instructions(), 2);
        let mut bytes = [0; 4];
        vm.read_bytes(0x8000, &mut bytes).unwrap();
        assert_eq!(bytes, [0xcd, 0xab, 0xee, 0x55]);
    }
}

#[test]
fn application_ram_selection_keeps_active_string_transfer_slots() {
    // Decision-16 §3: a completed device read belongs to the interrupted
    // iteration, including when mappings change before its write completes.
    let mut vm = Vm::new(VmConfig::flat(0x10000));
    vm.set_cpu_mode(CpuMode::Real16);
    vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x3000, 0x2000, Prot::RW, RegionKind::Trap).unwrap();
    vm.write_bytes(0x1000, b"\xf3\xa5\xf4").unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[1] = 2;
    cpu.cpu.gpr[6] = 0x3000;
    cpu.cpu.gpr[7] = 0x4000;
    let run = cpu.run_steps(&vm, 64, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0);
    assert!(matches!(run.stop(), StepStop::Exit(Exit::MmioRead { .. })));
    cpu.complete_mmio_read(0xabcd);
    let run = cpu.run_steps(&vm, 64, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0);
    assert!(matches!(
        run.stop(),
        StepStop::Exit(Exit::MmioWrite { value: 0xabcd, .. })
    ));
    vm.unmap(0x3000, 0x2000).unwrap();
    vm.map(0x3000, 0x2000, Prot::RW, RegionKind::Ram).unwrap();
    vm.write_bytes(0x3000, &[0x11, 0x22, 0x33, 0x44]).unwrap();
    cpu.complete_mmio_write();
    let run = cpu.run_steps(&vm, 64, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 3);
    assert!(matches!(run.stop(), StepStop::Exit(Exit::Hlt)));
    assert_eq!(cpu.retired_instructions(), 2);
    let mut data = [0; 4];
    vm.read_bytes(0x4000, &mut data).unwrap();
    assert_eq!(data, [0xcd, 0xab, 0x33, 0x44]);
}
