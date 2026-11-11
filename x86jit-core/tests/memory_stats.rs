use x86jit_core::{Exit, Prot, Reg, RegionKind, Vm, VmConfig};

#[test]
fn interpreted_ir_and_live_vcpu_allocations() {
    let mut vm = Vm::new(VmConfig::flat(0x4000));
    vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, &[0xb8, 7, 0, 0, 0, 0x48, 0xff, 0xc0, 0xf4])
        .unwrap();
    let initial = vm.memory_stats();
    assert_eq!(initial.vcpus.count, 0);
    assert_eq!(initial.guest_ram_reserved, 0x4000);
    assert_eq!(initial.guest_ram_committed, Some(0x4000));
    assert!(initial.memory_tracking_bytes > 0);
    let mut cpu = vm.new_vcpu();
    let cpu2 = vm.new_vcpu();
    let created = vm.memory_stats();
    assert_eq!(created.vcpus.count, 2);
    assert!(created.vcpus.state_bytes > 0);
    assert!(created.vcpus.fast_resolve_bytes > 0);
    assert_eq!(created.vcpus.scratch_bytes, 0);
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let lifted = vm.memory_stats();
    assert_eq!(lifted.cache.interpreted_entries, 1);
    assert!(lifted.cache.interpreted_ir_bytes > 0);
    assert!(lifted.vcpus.scratch_bytes > 0);
    assert_eq!(lifted.backend.code_bytes_reserved, 0);
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(
        vm.memory_stats().cache.interpreted_ir_bytes,
        lifted.cache.interpreted_ir_bytes
    );
    vm.cache.invalidate_overlapping(0x1000, 0x1009, || {});
    assert_eq!(vm.memory_stats().cache.interpreted_entries, 0);
    assert_eq!(vm.memory_stats().cache.interpreted_ir_bytes, 0);
    drop(cpu2);
    assert_eq!(
        vm.memory_stats().vcpus.state_bytes,
        created.vcpus.state_bytes / 2
    );
    drop(cpu);
    let dropped = vm.memory_stats();
    assert_eq!(dropped.vcpus.count, 0);
    assert_eq!(dropped.vcpus.scratch_bytes, 0);
    assert_eq!(dropped.vcpus.fast_resolve_bytes, 0);
}

#[test]
fn cache_replacements_and_stale_upgrades_preserve_accounting() {
    use std::sync::Arc;
    use x86jit_core::{BlockKey, CachedBlock, CompiledPtr, CpuMode, IrBlock, TranslationCache};
    let cache = TranslationCache::new();
    let key = BlockKey::new(0x1000, CpuMode::Long64);
    let ir = Arc::new(IrBlock {
        legacy_context: None,
        mode: CpuMode::Long64,
        guest_start: 0x1000,
        ops: Vec::new(),
        temp_count: 0,
        guest_len: 1,
        icount: 1,
    });
    let interpreted = CachedBlock::Interpreted(ir);
    let compiled = CachedBlock::Compiled {
        entry: CompiledPtr::new(std::ptr::null()),
    };
    cache.insert(key, interpreted.clone(), vec![(0x1000, 1)], |_| {});
    let bytes = cache.memory_stats().interpreted_ir_bytes;
    cache.insert(key, interpreted, vec![(0x1000, 1)], |_| {});
    assert_eq!(cache.memory_stats().interpreted_entries, 1);
    assert_eq!(cache.memory_stats().interpreted_ir_bytes, bytes);
    let generation = cache.publication_generation();
    assert!(cache.upgrade_region(
        key,
        compiled.clone(),
        vec![(0x1000, 1), (0x2000, 1)],
        generation,
        |_| {}
    ));
    assert_eq!(cache.memory_stats().region_entries, 1);
    assert_eq!(cache.memory_stats().interpreted_ir_bytes, 0);
    assert!(cache.upgrade(key, compiled.clone(), (0x1000, 1), generation));
    assert_eq!(cache.memory_stats().region_entries, 0);
    assert_eq!(cache.memory_stats().compiled_entries, 1);
    cache.invalidate_overlapping(0x1000, 0x1001, || {});
    assert!(!cache.upgrade(key, compiled, (0x1000, 1), generation));
    assert_eq!(cache.memory_stats().compiled_entries, 0);
}

#[test]
fn invalidation_activity_counts_calls_and_victims() {
    let mut vm = Vm::new(VmConfig::flat(0x4000));
    vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, &[0xf4]).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let before = vm.memory_stats().cache;
    vm.cache.invalidate_overlapping(0x1800, 0x1801, || {});
    vm.cache.invalidate_overlapping(0x1000, 0x1001, || {});
    let after = vm.memory_stats().cache;
    assert_eq!(after.invalidation_calls - before.invalidation_calls, 2);
    assert_eq!(
        after.invalidation_blocks_dropped - before.invalidation_blocks_dropped,
        1
    );
    assert_eq!(after.tier_bg_submitted, 0);
    assert_eq!(after.tier_bg_completed, 0);
}
