#![cfg(feature = "jit")]

use std::sync::{Arc, Condvar, Mutex};
use x86jit_core::cache::BlockKey;
use x86jit_core::lift::{lift_block, CpuMode, FetchAddr};
use x86jit_core::{Exit, Prot, Reg, RegionCaps, RegionKind, Vm, VmConfig};
use x86jit_cranelift::JitBackend;

fn fixture(backend: JitBackend) -> Vm {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), Box::new(backend));
    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, &[0xb8, 7, 0, 0, 0, 0xf4]).unwrap();
    vm
}

#[test]
fn source_barrier_stats_batch_two_page_jit_lift() {
    x86jit_core::with_barrier_success(|| {
        let vm = fixture(JitBackend::new());
        vm.write_bytes(0x1ffd, &[0xb8, 7, 0, 0, 0, 0xf4]).unwrap();
        let initial = vm.memory_stats();
        assert_eq!(initial.source_barrier_calls, 0);
        assert_eq!(initial.source_barrier_ns, 0);
        assert_eq!(initial.source_barrier_tagged_pages, 0);
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, 0x1ffd);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        let stats = vm.memory_stats();
        assert_eq!(stats.cache.compiled_entries, 1);
        assert_eq!(stats.source_barrier_calls, 1);
        assert_eq!(stats.source_barrier_tagged_pages, 2);
        cpu.set_reg(Reg::Rip, 0x1ffd);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        let cached = vm.memory_stats();
        assert_eq!(cached.source_barrier_calls, 1);
        assert_eq!(cached.source_barrier_ns, stats.source_barrier_ns);
        assert_eq!(cached.source_barrier_tagged_pages, 2);
    });
}

#[test]
fn backend_replacement_rejects_cached_compiled_entries_and_private_predictions() {
    let mut vm = fixture(JitBackend::new());
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(vm.memory_stats().cache.compiled_entries, 1);
    vm.backend = Box::new(x86jit_core::InterpreterBackend);
    assert!(vm
        .cache
        .get(BlockKey::new(0x1000, CpuMode::Long64))
        .is_none());
    for mut vcpu in [cpu, vm.new_vcpu()] {
        vcpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(vcpu.run(&vm, None), Exit::Hlt));
        assert_eq!(vcpu.reg(Reg::Rax), 7);
        assert!(vcpu.retired_instructions() > 0);
    }
}

#[test]
fn uncached_execution_after_backend_replacement_never_registers_code() {
    for tiered in [false, true] {
        let mut vm = x86jit_core::with_barrier_success(|| fixture(JitBackend::new()));
        if tiered {
            vm.set_tier_up_after(Some(1));
        }
        let old = vm.cache.code_lifetime();
        let mut cpu = vm.new_vcpu();
        for value in 1..=2 {
            if value == 2 {
                vm.backend = Box::new(JitBackend::new());
                assert!(!old.is_open());
            }
            vm.write_bytes(0x1001, &[value]).unwrap();
            cpu.set_reg(Reg::Rip, 0x1000);
            x86jit_core::with_barrier_flush_failure(|| {
                assert!(matches!(cpu.run(&vm, Some(4)), Exit::Hlt));
            });
            assert_eq!(cpu.reg(Reg::Rax), u64::from(value));
            let current = vm.backend.code_lifetime().unwrap();
            assert!(Arc::ptr_eq(&vm.cache.code_lifetime(), &current));
            assert_eq!(current.unit_stats(), (0, 0, 0, 0));
            assert_eq!(old.unit_stats(), (0, 0, 0, 0));
            let stats = vm.memory_stats();
            assert_eq!(
                stats.source_barrier_mode,
                x86jit_core::SourceBarrierMode::Process
            );
            assert_eq!(stats.source_barrier_uncached_executions, u64::from(value));
            assert_eq!(stats.source_barrier_calls, u64::from(value) * 3);
            assert_eq!(stats.source_barrier_tagged_pages, u64::from(value));
            assert_eq!(stats.backend.compiled_units_total, 0);
            assert_eq!(stats.cache.interpreted_entries, 0);
            assert_eq!(stats.cache.compiled_entries, 0);
            assert_eq!(stats.cache.region_entries, 0);
        }
    }
}

#[test]
fn backend_replacement_preserves_source_barrier_mode() {
    x86jit_core::with_barrier_success(|| {
        for degraded in [false, true] {
            let create = || fixture(JitBackend::new());
            let mut vm = if degraded {
                x86jit_core::with_barrier_unavailable(create)
            } else {
                create()
            };
            let mode = vm.memory_stats().source_barrier_mode;
            assert_eq!(
                mode,
                if degraded {
                    x86jit_core::SourceBarrierMode::Degraded
                } else {
                    x86jit_core::SourceBarrierMode::Process
                },
            );
            let code_range = vm.mem.code_range_ptr();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, 0x1000);
            assert!(matches!(cpu.run(&vm, Some(4)), Exit::Hlt));
            vm.backend = Box::new(JitBackend::new());
            cpu.set_reg(Reg::Rip, 0x1000);
            assert!(matches!(cpu.run(&vm, Some(4)), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rax), 7);
            assert_eq!(vm.memory_stats().source_barrier_mode, mode);
            assert_eq!(vm.mem.code_range_ptr(), code_range);
            assert_eq!(vm.memory_stats().cache.compiled_entries, 1);
            assert!(Arc::ptr_eq(
                &vm.cache.code_lifetime(),
                &vm.backend.code_lifetime().unwrap(),
            ));
        }
    });
}

#[test]
fn shutdown_joins_deferred_worker_from_replaced_backend() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::Duration;
    let mut jit = JitBackend::new();
    let starts = AtomicUsize::new(0);
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let worker_release = release.clone();
    let (finished_tx, finished_rx) = mpsc::channel();
    jit.set_worker_spawner(move |task| {
        let deferred = starts.fetch_add(1, Ordering::SeqCst) == 1;
        let release = worker_release.clone();
        let finished = finished_tx.clone();
        std::thread::Builder::new().spawn(move || {
            task();
            if deferred {
                finished.send(()).unwrap();
                let (lock, cv) = &*release;
                let _released = cv
                    .wait_while(lock.lock().unwrap(), |ready| !*ready)
                    .unwrap();
            }
        })
    });
    let mut vm = fixture(jit);
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    drop(cpu);
    let domain = vm.cache.code_lifetime().clone();
    let mut participant = domain.participant();
    let held = participant.enter();
    vm.backend = Box::new(x86jit_core::InterpreterBackend);
    drop(held);
    finished_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(!domain.has_pending_reclamation());
    let (shutdown_tx, shutdown_rx) = mpsc::channel();
    let shutdown = std::thread::spawn(move || shutdown_tx.send(vm.shutdown()).unwrap());
    let premature = shutdown_rx.recv_timeout(Duration::from_millis(40));
    let (lock, cv) = &*release;
    *lock.lock().unwrap() = true;
    cv.notify_all();
    assert!(
        premature.is_err(),
        "shutdown returned before the old worker exited"
    );
    assert_eq!(
        shutdown_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap()
            .outstanding_handles,
        0
    );
    shutdown.join().unwrap();
}

#[test]
fn eager_units_stay_allocated_until_execution_grace_after_invalidation() {
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
    let mut participant = vm.cache.code_lifetime().participant();
    let held = participant.enter();
    vm.cache.invalidate_overlapping(0x1000, 0x1006, || {});
    let invalidated = vm.memory_stats();
    assert_eq!(invalidated.cache.compiled_entries, 0);
    assert_eq!(
        invalidated.backend.code_bytes_reserved,
        compiled.backend.code_bytes_reserved
    );
    assert_eq!(invalidated.backend.compiled_units_live, 1);
    let lifetime = vm.cache.code_lifetime();
    assert!(lifetime.live_units().is_empty());
    let retired = lifetime.retired_units();
    assert_eq!(retired.len(), 1);
    assert!(!lifetime.execution_complete(retired[0]));
    assert_eq!(invalidated.backend.code_units_live, 0);
    assert_eq!(invalidated.backend.code_units_retired, 1);
    assert_eq!(invalidated.backend.code_bytes_live, 0);
    assert_eq!(
        invalidated.backend.code_bytes_retired,
        compiled.backend.code_bytes_live
    );
    drop(held);
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(vm.memory_stats().backend.compiled_units_total, 2);
}

#[test]
fn compiled_tier_replacement_retires_the_previous_generation() {
    let vm = fixture(JitBackend::new());
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let lifetime = vm.cache.code_lifetime();
    let old = lifetime.live_units()[0];
    let mut participant = lifetime.participant();
    let held = participant.enter();
    let ir = lift_block(&vm.mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
    let replacement = vm
        .backend
        .materialize(&ir, vm.consistency, None, vm.mem.guest_base());
    assert!(vm.cache.upgrade(
        BlockKey::new(0x1000, CpuMode::Long64),
        replacement,
        (0x1000, ir.guest_len),
        vm.cache.publication_generation(),
    ));
    assert_eq!(lifetime.retired_units()[0].unit(), old);
    assert!(!lifetime.execution_complete(lifetime.retired_units()[0]));
    assert_ne!(lifetime.live_units()[0].generation, old.generation);
    drop(held);
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(cpu.reg(Reg::Rax), 7);
}

#[test]
fn materialized_identity_survives_cache_and_direct_entry_and_rejects_stale_clones() {
    let vm = fixture(JitBackend::new());
    let key = BlockKey::new(0x1000, CpuMode::Long64);
    let ir = lift_block(&vm.mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
    let block = vm
        .backend
        .materialize(&ir, vm.consistency, None, vm.mem.guest_base());
    let x86jit_core::CachedBlock::Compiled { entry } = block.clone() else {
        panic!("expected compiled entry")
    };
    let domain = vm.cache.code_lifetime();
    assert_ne!(entry.generation(), 0);
    assert_eq!(domain.live_units()[0].generation, entry.generation());
    let mut participant = domain.participant();
    let guard = participant.enter();
    assert!(guard.admits(entry));
    let mut cpu = x86jit_core::CpuState::new();
    // SAFETY: admission validated this generation under participation retained
    // through the compiled call and decoding, with exclusive CPU and live memory.
    let result = unsafe {
        x86jit_core::jit_abi::run_compiled(
            entry,
            &mut cpu,
            &vm.mem,
            CpuMode::Long64,
            vm.consistency,
        )
    };
    assert!(matches!(result, x86jit_core::StepResult::Exit(Exit::Hlt)));
    assert_eq!(cpu.gpr[0], 7);
    vm.cache
        .insert(key, block.clone(), vec![(0x1000, ir.guest_len)], |_| {});
    let x86jit_core::CachedBlock::Compiled { entry: cached } = vm.cache.get(key).unwrap() else {
        unreachable!()
    };
    assert_eq!(cached.generation(), entry.generation());
    vm.cache.invalidate_overlapping(0x1000, 0x1006, || {});
    assert!(!guard.admits(entry));
    assert!(!domain.execution_complete(domain.retired_units()[0]));
    assert!(!vm.cache.upgrade(
        key,
        block,
        (0x1000, ir.guest_len),
        vm.cache.publication_generation()
    ));
    drop(guard);
    let new_guard = participant.enter();
    assert!(!new_guard.admits(entry));
    let replacement = vm
        .backend
        .materialize(&ir, vm.consistency, None, vm.mem.guest_base());
    let x86jit_core::CachedBlock::Compiled { entry: new_entry } = replacement.clone() else {
        unreachable!()
    };
    assert_ne!(entry.generation(), new_entry.generation());
    assert!(new_guard.admits(new_entry));
    let raw = x86jit_core::CachedBlock::Compiled {
        entry: x86jit_core::CompiledPtr::new(new_entry.0),
    };
    assert!(!vm.cache.upgrade(
        key,
        raw,
        (0x1000, ir.guest_len),
        vm.cache.publication_generation()
    ));
    assert!(vm.cache.upgrade(
        key,
        replacement,
        (0x1000, ir.guest_len),
        vm.cache.publication_generation()
    ));
}

#[test]
fn backend_teardown_defers_reclamation_until_execution_grace() {
    let vm = fixture(JitBackend::new());
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let lifetime = vm.cache.code_lifetime().clone();
    let unit = lifetime.live_units()[0];
    let mut participant = lifetime.participant();
    let held = participant.enter();
    drop(cpu);
    drop(vm);
    let retired = lifetime.retired_units();
    assert_eq!(retired.len(), 1);
    assert_eq!(retired[0].unit(), unit);
    assert!(!lifetime.execution_complete(retired[0]));
    assert!(lifetime.live_units().is_empty());
    assert!(x86jit_core::codemap::contains(unit.host_start));
    drop(held);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !lifetime.retired_units().is_empty() {
        assert!(std::time::Instant::now() < deadline);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(!x86jit_core::codemap::contains(unit.host_start));
}

#[test]
fn vcpu_cannot_carry_predictions_into_another_execution_domain_with_equal_epoch() {
    let first = fixture(JitBackend::new());
    let second = fixture(JitBackend::new());
    second
        .write_bytes(0x1000, &[0xb8, 9, 0, 0, 0, 0xf4])
        .unwrap();
    let mut cpu = first.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&first, None), Exit::Hlt));
    assert_eq!(cpu.reg(Reg::Rax), 7);
    assert_eq!(first.cache.epoch(), second.cache.epoch());
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&second, None), Exit::Hlt));
    assert_eq!(cpu.reg(Reg::Rax), 9);
    assert_eq!(second.cache.code_lifetime().live_units().len(), 1);
}

#[derive(Debug, Default)]
struct HelperGate {
    state: Mutex<(bool, bool)>,
    cv: Condvar,
}

impl x86jit_core::CpuSource for HelperGate {
    fn read_tsc(&self, _: u64) -> x86jit_core::TscSample {
        let mut state = self.state.lock().unwrap();
        state.0 = true;
        self.cv.notify_all();
        let (_state, timeout) = self
            .cv
            .wait_timeout_while(state, std::time::Duration::from_secs(20), |s| !s.1)
            .unwrap();
        assert!(!timeout.timed_out(), "compiled helper was not released");
        x86jit_core::TscSample { counter: 0, aux: 0 }
    }

    fn read_random(&self, _: u64, _: x86jit_core::RandomKind, _: u8) -> Option<u64> {
        None
    }
}

#[test]
fn vcpu_inside_compiled_helper_pins_retired_generation_until_dispatch_returns() {
    let gate = Arc::new(HelperGate::default());
    let mut vm = fixture(JitBackend::new());
    // SDM Vol 2B, RDTSC: the injected source is called from the compiled helper.
    vm.write_bytes(0x1000, &[0x0f, 0x31, 0xf4]).unwrap();
    vm.set_cpu_source(Some(gate.clone()));
    let vm = Arc::new(vm);
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    let running = vm.clone();
    let thread = std::thread::spawn(move || cpu.run(&running, None));
    let state = gate.state.lock().unwrap();
    let (mut state, timeout) = gate
        .cv
        .wait_timeout_while(state, std::time::Duration::from_secs(20), |s| !s.0)
        .unwrap();
    let entered = !timeout.timed_out();
    let lifetime = vm.cache.code_lifetime();
    vm.cache.invalidate_overlapping(0x1000, 0x1003, || {});
    let retired = lifetime.retired_units();
    let pinned = retired
        .first()
        .is_some_and(|unit| !lifetime.execution_complete(*unit));
    // Release before asserting, so a failing witness cannot strand a worker.
    state.1 = true;
    gate.cv.notify_all();
    drop(state);
    let exit = thread.join().unwrap();
    assert!(entered);
    assert!(pinned, "a helper still has a return PC in the retired unit");
    assert!(matches!(exit, Exit::Hlt));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !lifetime.retired_units().is_empty() {
        assert!(std::time::Instant::now() < deadline);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(!x86jit_core::codemap::contains(
        retired[0].unit().host_start
    ));
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
#[test]
fn repeated_invalidation_recompilation_bounds_storage_and_metadata() {
    use std::time::{Duration, Instant};
    use x86jit_core::{cache::BlockKey, CpuMode, Exit, Prot, Reg, RegionKind, Vm, VmConfig};
    use x86jit_cranelift::JitBackend;
    let mut vm = Vm::with_backend(VmConfig::flat(0x4000), Box::new(JitBackend::new()));
    vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, &[0xb8, 7, 0, 0, 0, 0xf4]).unwrap();
    let mut cpu = vm.new_vcpu();
    let mut bound = None;
    for cycle in 0..16 {
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rax), 7);
        vm.cache.invalidate_overlapping(0x1000, 0x1006, || {});
        assert!(vm
            .cache
            .get(BlockKey::new(0x1000, CpuMode::Long64))
            .is_none());
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let stats = vm.memory_stats();
            if stats.backend.code_bytes_used == 0 {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "cycle {cycle}: retired executable storage remains allocated: {} bytes",
                stats.backend.code_bytes_used
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        let stats = vm.memory_stats();
        let retained = (
            stats.backend.code_bytes_reserved,
            stats.backend.code_bytes_committed,
            stats.backend.code_allocation_metadata_bytes,
            stats.codemap_bytes,
        );
        if cycle > 3 {
            assert_eq!(
                Some(retained),
                bound,
                "cycle {cycle}: metadata/storage grew"
            );
        }
        bound = Some(retained);
    }
}
