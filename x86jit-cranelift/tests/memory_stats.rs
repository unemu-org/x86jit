#![cfg(feature = "jit")]

use std::sync::{Arc, Condvar, Mutex};
use x86jit_core::{Exit, Prot, Reg, RegionCaps, RegionKind, Vm, VmConfig};
use x86jit_cranelift::JitBackend;

fn fixture(backend: JitBackend) -> Vm {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), Box::new(backend));
    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, &[0xb8, 7, 0, 0, 0, 0xf4]).unwrap();
    vm
}

#[test]
fn eager_units_stay_allocated_after_invalidation() {
    let vm = fixture(JitBackend::new());
    let before = vm.memory_stats();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let compiled = vm.memory_stats();
    assert_eq!(compiled.backend.compiled_units_live, 1);
    assert_eq!(compiled.backend.compiled_units_total, 1);
    assert!(compiled.backend.code_bytes_emitted > 0);
    #[cfg(target_os = "macos")]
    // SAFETY: pthread capability query has no preconditions.
    let dense = unsafe { libc::pthread_jit_write_protect_supported_np() != 0 };
    #[cfg(not(target_os = "macos"))]
    let dense = true;
    assert_eq!(
        compiled.backend.code_bytes_reserved,
        if dense { 128 << 20 } else { 0 }
    );
    assert_eq!(compiled.cache.compiled_entries, 1);
    assert_eq!(compiled.cache.interpreted_entries, 0);
    assert!(compiled.codemap_bytes > before.codemap_bytes);
    vm.cache.invalidate_overlapping(0x1000, 0x1006, || {});
    let invalidated = vm.memory_stats();
    assert_eq!(invalidated.cache.compiled_entries, 0);
    assert_eq!(
        invalidated.backend.code_bytes_reserved,
        compiled.backend.code_bytes_reserved
    );
    assert_eq!(invalidated.backend.compiled_units_live, 1);
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(vm.memory_stats().backend.compiled_units_total, 2);
}

#[test]
fn background_queue_batch_and_publication_are_distinct() {
    let jit = JitBackend::new();
    let handle = jit.tier_up_handle();
    let pause = handle.pause_compiler();
    // The observer runs after finalization, before the batch completion guards drop.
    let gate = Arc::new((Mutex::new((false, false)), Condvar::new()));
    let observed = gate.clone();
    jit.set_compiled_code_observer(move |_| {
        let (lock, cv) = &*observed;
        let mut state = lock.lock().unwrap();
        state.0 = true;
        cv.notify_all();
        while !state.1 {
            state = cv.wait(state).unwrap();
        }
    });
    let mut vm = fixture(jit);
    vm.set_tier_up_after(Some(32));
    vm.set_tier_up_background(true);
    let mut cpu = vm.new_vcpu();
    for _ in 0..33 {
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    }
    let queued = vm.memory_stats();
    assert_eq!(queued.cache.interpreted_entries, 1);
    assert!(queued.cache.interpreted_ir_bytes > 0);
    assert_eq!(queued.backend.background_queue_len, 1);
    assert_eq!(queued.backend.background_in_flight, 0);
    assert_eq!(queued.backend.compiled_units_total, 0);
    drop(pause);
    {
        let (lock, cv) = &*gate;
        let state = lock.lock().unwrap();
        let (mut state, timeout) = cv
            .wait_timeout_while(state, std::time::Duration::from_secs(20), |s| !s.0)
            .unwrap();
        let active = vm.memory_stats();
        // Always release the worker before an assertion can unwind.
        state.1 = true;
        cv.notify_all();
        assert!(!timeout.timed_out());
        assert_eq!(active.backend.background_queue_len, 0);
        assert_eq!(active.backend.background_in_flight, 1);
        assert_eq!(active.backend.compiled_units_total, 1);
    }
    handle.wait_idle();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let published = vm.memory_stats();
    assert_eq!(published.cache.interpreted_entries, 0);
    assert_eq!(published.cache.interpreted_ir_bytes, 0);
    assert_eq!(published.cache.compiled_entries, 1);
    assert_eq!(published.backend.background_in_flight, 0);
    vm.cache.invalidate_overlapping(0x1000, 0x1006, || {});
    assert_eq!(vm.memory_stats().cache.compiled_entries, 0);
    assert_eq!(vm.memory_stats().backend.compiled_units_total, 1);
}

#[test]
fn region_publications_have_their_own_category() {
    let vm = fixture(JitBackend::with_superblocks(RegionCaps {
        max_blocks: 4,
        max_icount: 32,
    }));
    vm.write_bytes(
        0x1000,
        &[0xb9, 2, 0, 0, 0, 0xeb, 0, 0xff, 0xc9, 0x75, 0xfc, 0xf4],
    )
    .unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let stats = vm.memory_stats();
    assert_eq!(stats.cache.region_entries, 1);
    assert_eq!(stats.cache.compiled_entries, 0);
    assert_eq!(stats.backend.compiled_units_total, 1);
    vm.cache.invalidate_overlapping(0x1007, 0x100b, || {});
    assert_eq!(vm.memory_stats().cache.region_entries, 0);
}
