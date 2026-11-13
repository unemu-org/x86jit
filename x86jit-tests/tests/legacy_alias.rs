//! Decision-23 §6 / M4: the frozen alias runs with selector-derived caches.
use x86jit_core::{
    CpuMode, ExceptionDelivery, ExecutionProfile, Exit, Prot, RegionKind, SegmentId, StepOptions,
    StepStop, Vm, VmConfig,
};

fn vm(code: &[u8]) -> Vm {
    let mut vm = Vm::new(VmConfig::flat(0x40000));
    vm.set_cpu_mode(CpuMode::Real16);
    vm.map(0, 0x40000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, code).unwrap();
    vm
}

#[test]
fn profile_keeps_compatibility_identity() {
    let vm = vm(&[0x90]);
    assert_eq!(
        vm.execution_profile(),
        ExecutionProfile::Application(CpuMode::Real16)
    );
    let cpu = vm.new_vcpu();
    assert_eq!(
        cpu.execution_profile(),
        ExecutionProfile::Application(CpuMode::Real16)
    );
    assert!(vm.cpu_model().is_none());
    assert!(cpu.cpu_model().is_none());
}

#[test]
fn direct_selector_edits_derive_caches_and_execution_between_runs() {
    // SDM Vol 3A §§3.4.3, 12.9.2: cache base, limit and access image.
    // Decision-23 §6: each alias run takes visible selectors as its authority.
    let vm = vm(&[0xa1, 0x10, 0x00]); // mov ax,[0010h]
    vm.write_bytes(0x2000, &[0xa1, 0x10, 0x00]).unwrap();
    vm.write_bytes(0x3010, &[0x34, 0x12]).unwrap();
    vm.write_bytes(0x4010, &[0x78, 0x56]).unwrap();
    let mut cpu = vm.new_vcpu();
    for (cs, ds, value) in [(0x100, 0x300, 0x1234), (0x200, 0x400, 0x5678)] {
        cpu.cpu.cs = cs;
        cpu.cpu.ds = ds;
        cpu.cpu.es = 0x500;
        cpu.cpu.ss = 0x600;
        cpu.cpu.fs = 0x700;
        cpu.cpu.gs = 0x800;
        cpu.cpu.rip = 0;
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)));
        assert_eq!(run.steps(), 1);
        assert_eq!(cpu.cpu.gpr[0], value);
        for (i, selector) in [cs, ds, 0x500, 0x600, 0x700, 0x800].into_iter().enumerate() {
            let cache = cpu.cpu.legacy.segments[i];
            assert_eq!(cache.base, u32::from(selector) << 4);
            assert_eq!(cache.limit, 0xffff);
            assert_eq!(
                cache.attributes,
                if i == SegmentId::Cs as usize {
                    0x9b
                } else {
                    0x93
                }
            );
            assert_eq!((cache.unusable, cache.reserved), (0, 0));
        }
    }
}

#[test]
fn guest_segment_load_publishes_cache() {
    let vm = vm(&[0x8e, 0xd8]); // mov ds,ax
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[0] = 0x1234;
    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 1);
    assert_eq!(cpu.cpu.ds, 0x1234);
    assert_eq!(
        cpu.cpu.legacy.segments[SegmentId::Ds as usize].base,
        0x12340
    );
}

#[test]
fn outward_setting_is_ignored_and_exception_is_delivered_in_guest() {
    let mut vm = vm(&[0xf6, 0xf3]); // div bl, BL=0
    vm.set_exception_delivery(ExceptionDelivery::Outward);
    vm.write_bytes(0, &[0, 0x10, 0, 1]).unwrap();
    vm.write_bytes(0x2000, &[0xf4]).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[4] = 0x8000;
    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert!(matches!(run.stop(), StepStop::Exit(Exit::Hlt)));
    assert_eq!(
        (run.steps(), cpu.retired_instructions(), cpu.cpu.rip),
        (1, 1, 0x1001)
    );
    assert_eq!(cpu.cpu.cs, 0x100);
    assert_eq!(cpu.cpu.legacy.segments[SegmentId::Cs as usize].base, 0x1000);
    assert_eq!(cpu.cpu.gpr[4], 0x7ffa);
    let mut saved = [0; 2];
    vm.read_bytes(0x7ffa, &mut saved).unwrap();
    assert_eq!(u16::from_le_bytes(saved), 0x1000);
}

#[test]
fn single_step_does_not_reach_alias() {
    let vm = vm(&[0x90, 0xf4]);
    vm.write_bytes(4, &[0, 0x20, 0, 0]).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[4] = 0x8000;
    cpu.cpu.rflags_system |= 0x100;
    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert!(matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)));
    assert_eq!(
        (run.steps(), cpu.retired_instructions(), cpu.cpu.rip),
        (1, 1, 0x1001)
    );
    assert_eq!(cpu.cpu.gpr[4], 0x8000);
}
