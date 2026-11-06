use x86jit_core::lift::{lift_block, CpuMode, FetchAddr};
use x86jit_core::{Backend, CachedBlock, MemConsistency, Memory, MemoryModel, Prot, RegionKind};
use x86jit_cranelift::{JitBackend, LockTelemetry, LockTelemetrySnapshot};

fn block() -> x86jit_core::IrBlock {
    let mut mem = Memory::new(MemoryModel::Flat { size: 0x2000 });
    mem.map(0, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
    mem.write_bytes(0x1000, &[0xf4]).unwrap();
    lift_block(&mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap()
}

fn compile(jit: &JitBackend, mode: u32) {
    assert!(matches!(
        jit.materialize_with_mxcsr(&block(), MemConsistency::Fast, None, 0, mode),
        CachedBlock::Compiled { .. }
    ));
}

fn assert_records(stats: &LockTelemetry, count: u64) {
    assert_eq!(stats.acquisitions, count);
    assert!(stats.total_hold_ns >= stats.max_hold_ns);
    // No lower bound on the hold time: with a coarse monotonic clock (a 24 MHz
    // counter ticks every 41.7 ns) a short uncontended hold can measure 0 ns.
    // The contended tests in embedder_tests.rs force a measurable hold.
    assert!(stats.total_wait_ns >= stats.max_wait_ns);
    assert_eq!(stats.hold_histogram.iter().sum::<u64>(), count);
    assert_eq!(stats.wait_histogram.iter().sum::<u64>(), count);
}

#[test]
fn lock_telemetry_disabled_and_runtime_switch() {
    let jit = JitBackend::new();
    assert_eq!(jit.lock_telemetry(), LockTelemetrySnapshot::default());
    compile(&jit, 0);
    compile(&jit, 1);
    jit.invalidate_links();
    assert_eq!(jit.lock_telemetry(), LockTelemetrySnapshot::default());

    jit.set_lock_telemetry(true);
    compile(&jit, 0);
    let enabled = jit.lock_telemetry();
    assert_records(&enabled.compile_serial, 1);
    assert_records(&enabled.default_module, 1);
    assert_eq!(enabled.nondefault_module, LockTelemetry::default());
    jit.set_lock_telemetry(false);
    compile(&jit, 0);
    jit.invalidate_links();
    assert_eq!(jit.lock_telemetry(), enabled);
    jit.reset_lock_telemetry();
    assert_eq!(jit.lock_telemetry(), LockTelemetrySnapshot::default());
}

#[test]
fn lock_telemetry_counts_each_module_and_invalidation_without_building() {
    let jit = JitBackend::new();
    jit.set_lock_telemetry(true);
    // Invalidating an empty backend acquires both modules without building them.
    jit.invalidate_links();
    let empty = jit.lock_telemetry();
    assert_records(&empty.default_module, 1);
    assert_records(&empty.nondefault_module, 1);
    assert_eq!(empty.compile_serial, LockTelemetry::default());
    jit.reset_lock_telemetry();

    compile(&jit, 0);
    compile(&jit, 1);
    compile(&jit, 0);
    let compiled = jit.lock_telemetry();
    assert_records(&compiled.compile_serial, 3);
    assert_records(&compiled.default_module, 2);
    assert_records(&compiled.nondefault_module, 1);
    jit.invalidate_links();
    let invalidated = jit.lock_telemetry();
    assert_records(&invalidated.compile_serial, 3);
    assert_records(&invalidated.default_module, 3);
    assert_records(&invalidated.nondefault_module, 2);
}
