//! Publication witnesses for the Cranelift 0.135.8 background finalization batch.
use super::*;
use x86jit_core::jit_abi::run_compiled;
use x86jit_core::lift::{lift_block, CpuMode, FetchAddr};
use x86jit_core::{CpuState, Memory, MemoryModel, Prot, RegionKind};

fn fixture() -> (Memory, TierUpRequest) {
    let mut mem = Memory::new(MemoryModel::Flat { size: 0x4000 });
    mem.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
    mem.write_bytes(0x1000, &[0xb8, 7, 0, 0, 0, 0xf4]).unwrap();
    let ir = lift_block(&mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
    let req = TierUpRequest {
        pc: 0x1000,
        mxcsr_mode: 0,
        unit: TierUpUnit::Block(Arc::new(ir)),
        consistency: MemConsistency::Fast,
        mmio: None,
        guest_base: 0,
        spans: vec![(0x1000, 6)],
        epoch: 0,
        precise_split_stores: false,
    };
    (mem, req)
}

fn execute(block: CachedBlock, mem: &Memory) -> usize {
    let CachedBlock::Compiled { entry } = block else {
        panic!("expected a published entry")
    };
    let mut cpu = CpuState::new();
    // SAFETY: tests retain the backend and supply valid CPU and memory objects.
    let _ = unsafe { run_compiled(entry, &mut cpu, mem, CpuMode::Long64, MemConsistency::Fast) };
    assert_eq!(cpu.gpr[0], 7);
    assert!(
        (0..128).any(|offset| x86jit_core::codemap::lookup(entry.0 as usize + offset).is_some())
    );
    entry.0 as usize
}

fn collect_one(jit: &JitBackend) {
    let deadline = Instant::now() + std::time::Duration::from_secs(5);
    while !jit.shared.lifetime.retired_units().is_empty() {
        jit.collect_retired_metadata();
        assert!(Instant::now() < deadline, "metadata retirement stalled");
        std::thread::yield_now();
    }
}

#[test]
fn metadata_collection_waits_for_execution_and_preserves_live_ranges() {
    let (mem, req) = fixture();
    let TierUpUnit::Block(ir) = req.unit else {
        unreachable!()
    };
    let jit = JitBackend::new();
    let old = jit.materialize(&ir, MemConsistency::Fast, None, 0);
    let live = jit.materialize(&ir, MemConsistency::Fast, None, 0);
    let old_start = execute(old.clone(), &mem);
    let live_start = execute(live.clone(), &mem);
    let domain = jit.shared.lifetime.clone();
    let old_len = domain
        .live_units()
        .into_iter()
        .find(|u| u.host_start == old_start)
        .unwrap()
        .host_len;
    let mut participant = domain.participant();
    let guard = participant.enter();
    domain.discard(&old);
    assert_eq!(jit.collect_retired_metadata(), 0);
    assert!((0..128).any(|off| x86jit_core::codemap::lookup(old_start + off).is_some()));
    assert_eq!(jit.shared.jit(0).functions.len(), 2);
    drop(guard);
    collect_one(&jit);
    assert!(
        (0..old_len as usize).all(|off| x86jit_core::codemap::lookup(old_start + off).is_none())
    );
    assert!((0..128).any(|off| x86jit_core::codemap::lookup(live_start + off).is_some()));
    assert_eq!(jit.shared.jit(0).functions.len(), 1);
    assert!(domain.retired_units().is_empty());
    assert!(jit.shared.retired_metadata.lock().unwrap().is_empty());
    assert_eq!(jit.collect_retired_metadata(), 0);
    execute(live, &mem);
}

#[test]
fn repeated_metadata_retirement_recycles_declarations_in_both_modules_and_providers() {
    let (mem, req) = fixture();
    let TierUpUnit::Block(ir) = req.unit else {
        unreachable!()
    };
    let providers = [
        JitBackend::new(),
        JitBackend::new().with_system_memory_provider(),
        JitBackend::new().with_no_fd_memory_provider(CodeMemoryPolicy::WriteXorExecute),
    ];
    for jit in providers {
        let mut ids = [None, None];
        for cycle in 0..32 {
            for (mode, previous_id) in ids.iter_mut().enumerate() {
                let block = jit.materialize_with_mxcsr(
                    &ir,
                    MemConsistency::Fast,
                    None,
                    0,
                    if mode == 0 { 0 } else { 0x2000 },
                );
                let CachedBlock::Compiled { entry } = block.clone() else {
                    panic!("expected compiled entry")
                };
                let id = jit.shared.jit(mode as u32).functions[&entry.generation()].id;
                if let Some(previous) = previous_id {
                    assert_eq!(id, *previous, "declarations must be reused");
                }
                *previous_id = Some(id);
                let mut participant = jit.shared.lifetime.participant();
                let guard = participant.enter();
                assert!(guard.admits(entry));
                execute(block.clone(), &mem);
                drop(guard);
                jit.shared.lifetime.discard(&block);
                collect_one(&jit);
                let module = jit.shared.jit(mode as u32);
                assert!(module.functions.is_empty());
                assert!(module.pending_slots.is_empty());
                assert_eq!(module.reusable_functions.len(), 1);
                assert_eq!(module.module.declarations().get_functions().count(), 1);
                assert!(jit.shared.lifetime.retired_units().is_empty());
                assert!(jit.shared.retired_metadata.lock().unwrap().is_empty());
            }
            assert_eq!(jit.memory_stats().compiled_units_total, (cycle + 1) * 2);
        }
    }
}

#[test]
fn retired_unit_releases_stable_slots_after_grace() {
    let (mem, _) = fixture();
    mem.write_bytes(0x1000, &[0xe9, 0xfb, 0x0f, 0, 0]).unwrap();
    let ir = lift_block(&mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
    let jit = JitBackend::new();
    let block = jit.materialize(&ir, MemConsistency::Fast, None, 0);
    let CachedBlock::Compiled { entry } = block.clone() else {
        panic!("compiled branch")
    };
    let weak = {
        let module = jit.shared.jit(0);
        Arc::downgrade(&module.functions[&entry.generation()].slots[0])
    };
    jit.shared.lifetime.discard(&block);
    assert!(weak.upgrade().is_some());
    collect_one(&jit);
    assert!(weak.upgrade().is_none());
    let mut participant = jit.shared.lifetime.participant();
    assert!(!participant.enter().admits(entry));
}

#[test]
fn storage_reuse_waits_for_execution_and_rejects_stale_generations() {
    let (mem, req) = fixture();
    let TierUpUnit::Block(ir) = req.unit else {
        unreachable!()
    };
    let jit = JitBackend::new();
    // Keep a neighboring unit live, so alias providers retain the region too.
    let anchor = jit.materialize(&ir, MemConsistency::Fast, None, 0);
    let anchor_start = execute(anchor.clone(), &mem);
    let old = jit.materialize(&ir, MemConsistency::Fast, None, 0);
    let CachedBlock::Compiled { entry: stale } = old.clone() else {
        panic!("compiled unit")
    };
    let mut participant = jit.shared.lifetime.participant();
    let held = participant.enter();
    assert!(held.admits(stale));
    execute(old.clone(), &mem);
    jit.shared.lifetime.discard(&old);
    assert_eq!(jit.collect_retired_metadata(), 0);
    let during_grace = jit.materialize(&ir, MemConsistency::Fast, None, 0);
    let CachedBlock::Compiled { entry: other } = during_grace.clone() else {
        panic!("compiled unit")
    };
    assert_ne!(other.0, stale.0, "held execution pins its physical extent");
    drop(held);
    collect_one(&jit);
    let reused = jit.materialize(&ir, MemConsistency::Fast, None, 0);
    let CachedBlock::Compiled { entry: fresh } = reused.clone() else {
        panic!("compiled unit")
    };
    assert_eq!(
        fresh.0, stale.0,
        "the exact quiescent extent must be reused"
    );
    assert_ne!(fresh.generation(), stale.generation());
    let guard = participant.enter();
    assert!(
        !guard.admits(stale),
        "stale entry cannot resolve the new generation"
    );
    assert!(guard.admits(fresh));
    execute(reused.clone(), &mem);
    assert!((0..128).any(|off| x86jit_core::codemap::lookup(anchor_start + off).is_some()));
    execute(anchor.clone(), &mem);
    drop(guard);
    jit.shared.lifetime.discard(&reused);
    jit.shared.lifetime.discard(&during_grace);
    while !jit.shared.lifetime.retired_units().is_empty() {
        collect_one(&jit);
    }
    let baseline = jit.memory_stats();
    for _ in 0..128 {
        let block = jit.materialize(&ir, MemConsistency::Fast, None, 0);
        let CachedBlock::Compiled { entry } = block.clone() else {
            panic!("compiled unit")
        };
        assert_eq!(entry.0, stale.0);
        assert_ne!(entry.generation(), stale.generation());
        assert!(!participant.enter().admits(stale));
        execute(block.clone(), &mem);
        jit.shared.lifetime.discard(&block);
        collect_one(&jit);
        let stats = jit.memory_stats();
        assert_eq!(stats.code_bytes_used, baseline.code_bytes_used);
        assert_eq!(stats.code_bytes_committed, baseline.code_bytes_committed);
        assert_eq!(stats.code_bytes_reserved, baseline.code_bytes_reserved);
        assert_eq!(
            stats.code_allocation_metadata_bytes,
            baseline.code_allocation_metadata_bytes
        );
        assert_eq!(jit.shared.jit(0).functions.len(), 1);
    }
    jit.shared.lifetime.discard(&anchor);
    collect_one(&jit);
    assert_eq!(jit.memory_stats().code_bytes_committed, 0);
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    assert_eq!(jit.memory_stats().code_regions, 0);
}

#[test]
fn repeated_retirement_bounds_code_and_provider_records_in_both_modules() {
    let (mem, req) = fixture();
    let TierUpUnit::Block(ir) = req.unit else {
        unreachable!()
    };
    for jit in [
        JitBackend::new(),
        JitBackend::new().with_no_fd_memory_provider(CodeMemoryPolicy::WriteXorExecute),
    ] {
        let mut bounds = None;
        for cycle in 0..128 {
            for mode in [0, 0x2000] {
                let block = jit.materialize_with_mxcsr(&ir, MemConsistency::Fast, None, 0, mode);
                execute(block.clone(), &mem);
                jit.shared.lifetime.discard(&block);
                collect_one(&jit);
                let module = jit.shared.jit(u32::from(mode != 0));
                assert!(module.functions.is_empty());
                assert!(module.pending_slots.is_empty());
                assert_eq!(module.module.declarations().get_functions().count(), 1);
            }
            let stats = jit.memory_stats();
            assert_eq!(
                stats.code_bytes_used, 0,
                "cycle {cycle}: released storage retained"
            );
            assert_eq!(stats.code_bytes_committed, 0);
            let retained = (
                stats.code_bytes_reserved,
                stats.code_allocation_metadata_bytes,
            );
            if let Some(bounds) = bounds {
                assert_eq!(retained, bounds, "provider storage grew on cycle {cycle}");
            }
            bounds = Some(retained);
            assert!(jit.shared.lifetime.retired_units().is_empty());
            assert!(jit.shared.retired_metadata.lock().unwrap().is_empty());
        }
    }
}

#[test]
fn legacy_observer_keeps_backend_lifetime_ranges_after_metadata_collection() {
    let (_, req) = fixture();
    let TierUpUnit::Block(ir) = req.unit else {
        unreachable!()
    };
    let jit = JitBackend::new();
    // A panic disables notifications but cannot revoke the range lifetime.
    jit.set_compiled_code_observer(|_| panic!("disable legacy observer"));
    let block = jit.materialize(&ir, MemConsistency::Fast, None, 0);
    let CachedBlock::Compiled { entry } = block.clone() else {
        panic!("compiled unit")
    };
    let len = jit.shared.lifetime.live_units()[0].host_len as usize;
    // SAFETY: the backend retains this observed range through its lifetime.
    let bytes = unsafe { std::slice::from_raw_parts(entry.0, len) }.to_vec();
    let before = jit.memory_stats();
    jit.shared.lifetime.discard(&block);
    collect_one(&jit);
    assert_eq!(jit.memory_stats().code_bytes_used, before.code_bytes_used);
    assert_eq!(
        jit.memory_stats().code_bytes_committed,
        before.code_bytes_committed
    );
    assert!(jit.shared.jit(0).functions.is_empty());
    // SAFETY: collecting metadata cannot revoke the legacy observer's contract.
    assert_eq!(unsafe { std::slice::from_raw_parts(entry.0, len) }, bytes);
    let next = jit.materialize(&ir, MemConsistency::Fast, None, 0);
    let CachedBlock::Compiled { entry: next } = next else {
        panic!("compiled unit")
    };
    assert_ne!(next.0, entry.0);
}

#[test]
fn queued_small_units_share_pages_in_both_modules_and_publish_in_order() {
    let (mem, _) = fixture();
    let jit = JitBackend::with_opt_level(OptLevel::Speed);
    let handle = jit.tier_up_handle();
    let pause = handle.pause_compiler();
    for i in 0..TIER_QUEUE_CAP {
        let (_, mut req) = fixture();
        // Same PC with different generations must preserve each completion's identity.
        req.epoch = i as u64;
        req.mxcsr_mode = if i % 2 == 0 { 0 } else { 0x2000 };
        assert_eq!(jit.tier_up_async(req), TierUpSubmit::Queued);
    }
    assert!(jit.tier_up_finished().is_empty());
    drop(pause);
    handle.wait_idle();
    let done = jit.tier_up_finished();
    assert_eq!(done.len(), TIER_QUEUE_CAP);
    let mut pages = [
        std::collections::BTreeSet::new(),
        std::collections::BTreeSet::new(),
    ];
    for (i, finished) in done.into_iter().enumerate() {
        assert_eq!(finished.epoch, i as u64);
        let module = usize::from(finished.mxcsr_mode != 0);
        pages[module].insert(execute(finished.block, &mem) / region::page::size());
    }
    let stats = jit.memory_stats();
    assert_eq!(stats.compiled_units_total, TIER_QUEUE_CAP as u64);
    #[cfg(target_os = "macos")]
    // SAFETY: pthread capability query has no preconditions.
    let dense = unsafe { libc::pthread_jit_write_protect_supported_np() != 0 };
    #[cfg(not(target_os = "macos"))]
    let dense = true;
    if dense {
        assert_eq!(stats.code_regions, 2);
        assert_eq!(stats.code_bytes_reserved, 2 * (128 << 20));
        assert_eq!(
            stats.code_bytes_committed,
            (pages.iter().map(|pages| pages.len()).sum::<usize>() * region::page::size()) as u64
        );
    } else {
        assert_eq!(stats.code_regions, 0);
        assert_eq!(stats.code_bytes_reserved, 0);
        assert_eq!(stats.code_bytes_committed, 0);
    }
    for pages in pages {
        assert!(
            pages.len() <= TIER_QUEUE_CAP.div_ceil(TIER_FINALIZE_BATCH),
            "small queued units wasted pages: {pages:?}"
        );
    }
    assert!(jit.tier_up_finished().is_empty());
}

#[test]
fn foreground_executes_immediately_with_background_definitions_pending() {
    let (mem, req) = fixture();
    let jit = JitBackend::new();
    let observed = Arc::new(AtomicUsize::new(0));
    let events = observed.clone();
    jit.set_compiled_code_observer(move |_| {
        events.fetch_add(1, Ordering::Relaxed);
    });
    // Stop exactly between a worker definition and its batch finalization.
    let pending = jit.shared.define_request(&req, None).unwrap();
    assert_eq!(observed.load(Ordering::Relaxed), 0);
    assert!(jit.tier_up_finished().is_empty());
    let TierUpUnit::Block(ir) = &req.unit else {
        unreachable!()
    };
    execute(jit.materialize(ir, MemConsistency::Fast, None, 0), &mem);
    assert_eq!(observed.load(Ordering::Relaxed), 1);
    // Foreground finalization sealed both definitions; the worker can now register
    // its saved metadata and publish without another compile or a queue refill.
    let serial = jit.shared.compile_serial.lock().unwrap();
    let mut module = jit.shared.jit(0);
    let entry = Shared::finalized_entry(&mut module, &pending);
    drop(module);
    drop(serial);
    let (entry, event) = jit.shared.publish(entry, pending).unwrap();
    jit.shared.observe(event);
    execute(CachedBlock::Compiled { entry }, &mem);
    assert_eq!(observed.load(Ordering::Relaxed), 2);
}

#[test]
fn batch_resource_failures_never_publish_unfinalized_entries() {
    for stage in 1..=4 {
        let jit = JitBackend::new();
        jit.shared
            .fail_compile_stage
            .store(stage, Ordering::Relaxed);
        let handle = jit.tier_up_handle();
        let pause = handle.pause_compiler();
        for _ in 0..4 {
            assert_eq!(jit.tier_up_async(fixture().1), TierUpSubmit::Queued);
        }
        drop(pause);
        handle.wait_idle();
        let done = jit.tier_up_finished();
        assert_eq!(done.len(), 4);
        assert!(done
            .into_iter()
            .all(|fin| matches!(fin.block, CachedBlock::Interpreted(_))));
        assert!(jit.code_space_exhausted());
        assert_eq!(jit.shared.background.queue.lock().unwrap().outstanding, 0);
    }
}

#[test]
fn units_sharing_a_page_keep_distinct_fault_metadata() {
    let (mem, _) = fixture();
    mem.write_bytes(0x2000, &7u32.to_le_bytes()).unwrap();
    let jit = JitBackend::with_opt_level(OptLevel::Speed);
    let ranges = Arc::new(Mutex::new(Vec::new()));
    let observed = ranges.clone();
    jit.set_compiled_code_observer(move |code| observed.lock().unwrap().push(code));
    let handle = jit.tier_up_handle();
    let pause = handle.pause_compiler();
    for pc in [0x1000, 0x1010] {
        // Exercise an existing scalar-load lowering with an exact transfer site.
        mem.write_bytes(pc, &[0x8b, 0x04, 0x25, 0x00, 0x20, 0, 0, 0xf4])
            .unwrap();
        let ir = lift_block(&mem, FetchAddr::flat(pc), CpuMode::Long64).unwrap();
        let (_, mut req) = fixture();
        req.pc = pc;
        req.spans = vec![(pc, ir.guest_len)];
        req.unit = TierUpUnit::Block(Arc::new(ir));
        assert_eq!(jit.tier_up_async(req), TierUpSubmit::Queued);
    }
    drop(pause);
    handle.wait_idle();
    for finished in jit.tier_up_finished() {
        execute(finished.block, &mem);
    }
    let ranges = ranges.lock().unwrap();
    assert_eq!(ranges.len(), 2);
    assert_eq!(
        ranges[0].host_start / region::page::size(),
        ranges[1].host_start / region::page::size()
    );
    for code in ranges.iter() {
        let pc = (0..code.host_len)
            .map(|offset| code.host_start + offset as usize)
            .find(|&pc| x86jit_core::codemap::lookup_trap(pc).is_some())
            .expect("scalar transfer site");
        assert_eq!(
            x86jit_core::codemap::lookup_trap(pc).unwrap().guest_rip,
            code.guest_rip
        );
        assert!(x86jit_core::codemap::lookup_fault_info(pc, code.guest_rip).is_some());
        #[cfg(any(target_os = "linux", target_os = "macos", target_os = "freebsd"))]
        assert!(x86jit_core::codemap::lookup_recovery(pc).is_some());
    }
}

#[test]
fn waiting_foreground_compiler_precedes_the_queued_wave() {
    let (mem, req) = fixture();
    let jit = JitBackend::new();
    // Represent a foreground caller just before it takes compile_serial.
    let foreground = ForegroundCompile::new(&jit.shared.background);
    let handle = jit.tier_up_handle();
    for _ in 0..TIER_QUEUE_CAP {
        assert_eq!(jit.tier_up_async(fixture().1), TierUpSubmit::Queued);
    }
    let TierUpUnit::Block(ir) = req.unit else {
        unreachable!()
    };
    execute(jit.materialize(&ir, MemConsistency::Fast, None, 0), &mem);
    assert_eq!(jit.shared.compile_attempts.load(Ordering::Relaxed), 1);
    assert!(jit.tier_up_finished().is_empty());
    drop(foreground);
    handle.wait_idle();
    assert_eq!(jit.tier_up_finished().len(), TIER_QUEUE_CAP);
}

#[test]
fn foreground_observer_can_wait_for_a_background_completion() {
    let (mem, req) = fixture();
    let jit = Arc::new(JitBackend::new());
    let weak = Arc::downgrade(&jit);
    let once = AtomicBool::new(false);
    // The backend contains observer panics, so record the outcome and assert it
    // after materialize instead of asserting inside the observer.
    let outcome = Arc::new(std::sync::Mutex::new(None));
    let seen = Arc::clone(&outcome);
    jit.set_compiled_code_observer(move |_| {
        if !once.swap(true, Ordering::Relaxed) {
            let jit = weak.upgrade().unwrap();
            let submitted = jit.tier_up_async(fixture().1);
            jit.tier_up_handle().wait_idle();
            *seen.lock().unwrap() = Some((submitted, jit.tier_up_finished().len()));
        }
    });
    let TierUpUnit::Block(ir) = req.unit else {
        unreachable!()
    };
    execute(jit.materialize(&ir, MemConsistency::Fast, None, 0), &mem);
    assert_eq!(
        *outcome.lock().unwrap(),
        Some((TierUpSubmit::Queued, 1)),
        "observer must submit, wait and see exactly one completion"
    );
}

fn wait_reclaimed(shared: &Shared) {
    let deadline = Instant::now() + std::time::Duration::from_secs(5);
    while !shared.lifetime.retired_units().is_empty() {
        assert!(Instant::now() < deadline, "automatic reclamation stalled");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

#[test]
fn jit_replacement_adopts_the_new_domain_once_and_recompiles_each_block_once() {
    use x86jit_core::{Exit, Reg, Vm, VmConfig};
    let old = JitBackend::new();
    let old_shared = old.shared.clone();
    let mut vm = Vm::with_backend(VmConfig::flat(0x4000), Box::new(old));
    vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, &[0xb8, 7, 0, 0, 0, 0xe9, 0xf6, 0, 0, 0])
        .unwrap();
    vm.write_bytes(0x1100, &[0xff, 0xc0, 0xf4]).unwrap();
    let mut cpu = vm.new_vcpu();
    let mut second_cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, Some(2)), Exit::Hlt));
    assert_eq!(cpu.reg(Reg::Rax), 8);
    assert_eq!(old_shared.compile_attempts.load(Ordering::Relaxed), 2);
    let mut participant = old_shared.lifetime.participant();
    let held = participant.enter();
    let next = JitBackend::new();
    let next_shared = next.shared.clone();
    vm.backend = Box::new(next);
    assert!(!old_shared.lifetime.is_open());
    let generation = vm.cache.publication_generation();
    for run in 0..3 {
        let vcpu = if run == 1 { &mut second_cpu } else { &mut cpu };
        vcpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(vcpu.run(&vm, Some(2)), Exit::Hlt));
        assert_eq!(vcpu.reg(Reg::Rax), 8);
        assert!(Arc::ptr_eq(
            &vm.cache.code_lifetime(),
            &next_shared.lifetime
        ));
        assert_eq!(vm.cache.publication_generation(), generation + 1);
        assert_eq!(next_shared.compile_attempts.load(Ordering::Relaxed), 2);
        assert_eq!(vm.memory_stats().cache.compiled_entries, 2);
    }
    drop(held);
    wait_reclaimed(&old_shared);
    assert_eq!(old_shared.memory.snapshot().code_bytes_used, 0);
    assert!(old_shared.jit(0).functions.is_empty());
    drop(cpu);
    drop(second_cpu);
    drop(old_shared);
    drop(next_shared);
    vm.shutdown().unwrap();
}

#[test]
fn rejected_foreground_allocations_are_reclaimed_and_respect_budget_and_exit() {
    use x86jit_core::{CriticalSectionKind, Exit, ExitRequest, FaultPath, Reg, Vm, VmConfig};
    for interrupt in [false, true] {
        let jit = JitBackend::new();
        let shared = jit.shared.clone();
        let request = Arc::new(Mutex::new(None::<ExitRequest>));
        let on_compile = request.clone();
        jit.set_code_lifecycle_observer(move |event| {
            if interrupt && event.kind == CodeLifecycleKind::Publish {
                on_compile.lock().unwrap().as_ref().unwrap().request_exit();
            }
        });
        let mut vm = Vm::with_backend(VmConfig::flat(0x4000), Box::new(jit));
        vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, &[0x48, 0xff, 0xc0, 0xeb, 0xfb])
            .unwrap();
        // Keep a different open domain to force publication rejection even
        // though the current backend can allocate executable storage.
        vm.cache = x86jit_core::TranslationCache::new();
        let mut cpu = vm.new_vcpu();
        *request.lock().unwrap() = Some(cpu.exit_request());
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&vm, Some(0)), Exit::BudgetExhausted));
        assert_eq!(shared.compile_attempts.load(Ordering::Relaxed), 0);
        let exit = cpu.run(&vm, Some(1));
        if interrupt {
            assert!(matches!(exit, Exit::Interrupted));
            assert_eq!(cpu.reg(Reg::Rax), 1);
            assert_eq!(cpu.cpu.rip, 0x1000);
            let evidence = cpu.interruption_evidence().unwrap();
            assert!(evidence.deferred());
            assert!(evidence.contains(CriticalSectionKind::Compile));
            assert!(evidence.contains(CriticalSectionKind::Publish));
            assert_eq!(evidence.path(), FaultPath::Interpreter);
            assert_eq!(evidence.rip(), cpu.cpu.rip);
        } else {
            assert!(matches!(exit, Exit::BudgetExhausted));
            assert_eq!(cpu.reg(Reg::Rax), 1);
            assert!(cpu.interruption_evidence().is_none());
        }
        assert_eq!(shared.compile_attempts.load(Ordering::Relaxed), 1);
        assert!(shared.lifetime.live_units().is_empty());
        wait_reclaimed(&shared);
        assert_eq!(shared.memory.snapshot().code_bytes_used, 0);
        assert!(shared.jit(0).functions.is_empty());
        assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
        assert_eq!(cpu.reg(Reg::Rax), 2);
        assert!(cpu.interruption_evidence().is_none());
        assert_eq!(shared.compile_attempts.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn automatic_reclamation_covers_invalidation_replacement_and_rejected_completion() {
    use x86jit_core::{cache::BlockKey, Exit, Reg, Vm, VmConfig};
    for trigger in 0..3 {
        let jit = JitBackend::new();
        let shared = jit.shared.clone();
        let events = Arc::new(Mutex::new(Vec::new()));
        let observed = events.clone();
        jit.set_code_lifecycle_observer(move |event| observed.lock().unwrap().push(event));
        let mut vm = Vm::with_backend(VmConfig::flat(0x4000), Box::new(jit));
        vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, &[0xb8, 7, 0, 0, 0, 0xf4]).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        let old = shared.lifetime.live_units()[0];
        let mut participant = shared.lifetime.participant();
        let held = participant.enter();
        let key = BlockKey::new(0x1000, CpuMode::Long64);
        let target = match trigger {
            0 => {
                vm.unmap(0, 0x4000).unwrap();
                old
            }
            1 => {
                let ir = lift_block(&vm.mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
                let new = vm.backend.materialize(&ir, MemConsistency::Fast, None, 0);
                assert!(vm.cache.upgrade(
                    key,
                    new,
                    (0x1000, ir.guest_len),
                    vm.cache.publication_generation()
                ));
                old
            }
            _ => {
                let (_, request) = fixture();
                assert_eq!(vm.backend.tier_up_async(request), TierUpSubmit::Queued);
                let deadline = Instant::now() + std::time::Duration::from_secs(5);
                let unit = loop {
                    let done = shared.done.lock().unwrap();
                    if let Some(finished) = done.first() {
                        let CachedBlock::Compiled { entry } = &finished.block else {
                            panic!("compiled completion")
                        };
                        break shared.lifetime.unit_at(entry.0 as usize).unwrap();
                    }
                    drop(done);
                    assert!(Instant::now() < deadline);
                    std::thread::sleep(std::time::Duration::from_millis(1));
                };
                // A fresh dispatcher drains it. No pending cache request owns
                // this completion, so publication must reject it.
                vm.set_tier_up_background(true);
                let mut dispatcher = vm.new_vcpu();
                dispatcher.set_reg(Reg::Rip, 0x1000);
                assert!(matches!(dispatcher.run(&vm, None), Exit::Hlt));
                assert_eq!(vm.cache.memory_stats().tier_bg_superseded, 1);
                unit
            }
        };
        assert_eq!(vm.memory_stats().backend.code_units_retired, 1);
        assert_eq!(
            vm.memory_stats().backend.code_bytes_retired,
            u64::from(target.host_len)
        );
        assert_eq!(shared.lifetime.retired_units()[0].unit(), target);
        std::thread::sleep(std::time::Duration::from_millis(25));
        assert_eq!(
            shared.lifetime.retired_units().len(),
            1,
            "held execution pins extent"
        );
        drop(held);
        wait_reclaimed(&shared);
        for event in events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| event.unit == target)
        {
            assert_eq!(event.code.guest_rip, 0x1000);
            assert_eq!(event.code.kind, CompiledCodeKind::Block);
            assert_eq!(event.code.host_start, event.unit.host_start);
            assert_eq!(event.code.host_len, event.unit.host_len);
        }
        let phases: Vec<_> = events
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.unit == target)
            .map(|e| e.kind)
            .collect();
        assert_eq!(
            phases,
            [
                CodeLifecycleKind::Publish,
                CodeLifecycleKind::Retire,
                CodeLifecycleKind::Release
            ]
        );
        assert_eq!(vm.memory_stats().backend.code_units_retired, 0);
        assert!((0..target.host_len as usize)
            .all(|off| x86jit_core::codemap::lookup(target.host_start + off).is_none()));
        assert!(!shared.jit(0).functions.contains_key(&target.generation));
        if trigger == 0 {
            assert_eq!(shared.memory.snapshot().code_bytes_used, 0);
        }
    }
}

#[test]
fn deferred_teardown_owns_code_until_held_execution_releases() {
    let (_, req) = fixture();
    let TierUpUnit::Block(ir) = req.unit else {
        unreachable!()
    };
    let jit = JitBackend::new();
    let counters = jit.shared.memory.clone();
    let weak = Arc::downgrade(&jit.shared);
    let events = Arc::new(Mutex::new(Vec::new()));
    let observed = events.clone();
    jit.set_code_lifecycle_observer(move |event| observed.lock().unwrap().push(event));
    let block = jit.materialize(&ir, MemConsistency::Fast, None, 0);
    let CachedBlock::Compiled { entry } = block else {
        panic!("compiled unit")
    };
    let mut participant = jit.shared.lifetime.participant();
    let held = participant.enter();
    assert!(held.admits(entry));
    let bytes = counters.snapshot().code_bytes_used;
    drop(jit);
    std::thread::sleep(std::time::Duration::from_millis(25));
    assert!(
        weak.upgrade().is_some(),
        "deferred owner retains helpers and compiler records"
    );
    assert_eq!(counters.snapshot().code_bytes_used, bytes);
    assert!(!events
        .lock()
        .unwrap()
        .iter()
        .any(|e| e.kind == CodeLifecycleKind::Release));
    drop(held);
    let deadline = Instant::now() + std::time::Duration::from_secs(5);
    while weak.upgrade().is_some() || counters.snapshot().code_bytes_reserved != 0 {
        assert!(Instant::now() < deadline);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(counters.snapshot().code_bytes_reserved, 0);
    assert_eq!(counters.snapshot().code_bytes_used, 0);
    assert_eq!(
        events
            .lock()
            .unwrap()
            .iter()
            .map(|e| e.kind)
            .collect::<Vec<_>>(),
        [
            CodeLifecycleKind::Publish,
            CodeLifecycleKind::Retire,
            CodeLifecycleKind::Release
        ]
    );
}

#[test]
fn repeated_automatic_reclamation_bounds_all_metadata_including_ibtc() {
    let (mem, _) = fixture();
    mem.write_bytes(0x1000, &[0x48, 0x8b, 0x04, 0x25, 0, 0x30, 0, 0, 0xff, 0xe0])
        .unwrap();
    let ir = lift_block(&mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
    let jit = JitBackend::new();
    let mut bounds = None;
    for cycle in 0..64 {
        let block = jit.materialize(&ir, MemConsistency::Fast, None, 0);
        let CachedBlock::Compiled { entry } = block.clone() else {
            panic!("compiled unit")
        };
        let mut participant = jit.shared.lifetime.participant();
        let held = participant.enter();
        let slot = jit.shared.jit(0).functions[&entry.generation()].slots[0].clone();
        let descriptor = jit.shared.lifetime.alloc_ibtc_descriptor(0x1000, entry);
        slot.store(descriptor, Ordering::Release);
        jit.shared.lifetime.discard(&block);
        assert_eq!(slot.load(Ordering::SeqCst), 0);
        assert_eq!(jit.memory_stats().code_ibtc_descriptors, 1);
        // SAFETY: the held participant pins the immutable descriptor after unlinking.
        assert_eq!(
            unsafe { *(descriptor as *const [u64; 2]) },
            [0x1000, entry.0 as u64]
        );
        drop(held);
        wait_reclaimed(&jit.shared);
        let stats = jit.memory_stats();
        assert_eq!(stats.code_bytes_used, 0, "cycle {cycle}");
        assert_eq!(stats.code_ibtc_descriptors, 0);
        assert_eq!(stats.code_units_live + stats.code_units_retired, 0);
        let metadata = x86jit_core::codemap::memory_stats();
        assert_eq!(
            metadata.nodes
                + metadata.entries
                + metadata.sources
                + metadata.traps
                + metadata.recipes
                + metadata.instructions,
            0
        );
        let retained = (
            metadata.writer_storage,
            stats.code_allocation_metadata_bytes,
        );
        if cycle > 3 {
            assert_eq!(Some(retained), bounds, "cycle {cycle}");
        }
        bounds = Some(retained);
        let module = jit.shared.jit(0);
        assert!(module.functions.is_empty());
        assert!(module.pending_slots.is_empty());
        assert!(module.slots.is_empty());
        assert_eq!(module.module.declarations().get_functions().count(), 1);
        assert!(jit.shared.retirement_announced.lock().unwrap().is_empty());
        assert!(jit.shared.retired_metadata.lock().unwrap().is_empty());
    }
}

#[test]
fn failed_definitions_release_storage_and_scratch_slots() {
    for stage in 1..=4 {
        let (mem, _) = fixture();
        mem.write_bytes(0x1000, &[0xff, 0xe0]).unwrap();
        let ir = lift_block(&mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
        let jit = JitBackend::new();
        jit.shared
            .fail_compile_stage
            .store(stage, Ordering::Relaxed);
        assert!(matches!(
            jit.materialize(&ir, MemConsistency::Fast, None, 0),
            CachedBlock::Interpreted(_)
        ));
        let module = jit.shared.jit(0);
        assert!(module.slots.is_empty(), "stage {stage}");
        assert!(module.pending_slots.is_empty(), "stage {stage}");
        assert!(module.functions.is_empty(), "stage {stage}");
        assert_eq!(module.reusable_functions.len(), 1);
        assert_eq!(module.module.defined_function_count(), 0);
        assert_eq!(jit.memory_stats().code_bytes_used, 0, "stage {stage}");
    }
}

#[test]
fn release_observer_finishes_before_reuse_and_can_reenter_compilation() {
    let (_, req) = fixture();
    let TierUpUnit::Block(ir) = req.unit else {
        unreachable!()
    };
    let jit = JitBackend::new();
    let weak = Arc::downgrade(&jit.shared);
    let nested_ir = ir.clone();
    let called = Arc::new(AtomicBool::new(false));
    let observed = called.clone();
    jit.set_code_lifecycle_observer(move |event| {
        if event.kind == CodeLifecycleKind::Release && !observed.swap(true, Ordering::AcqRel) {
            let shared = weak.upgrade().unwrap();
            assert_eq!(
                shared.collect_retired_metadata(),
                0,
                "reentrant collection is nonblocking"
            );
            let entry = shared
                .compile(&nested_ir, MemConsistency::Fast, None, 0, 0, false)
                .unwrap();
            assert_ne!(
                entry.0 as usize, event.unit.host_start,
                "release callback still owns its range"
            );
            shared.lifetime.discard(&CachedBlock::Compiled { entry });
        }
    });
    let block = jit.materialize(&ir, MemConsistency::Fast, None, 0);
    jit.shared.lifetime.discard(&block);
    wait_reclaimed(&jit.shared);
    assert!(called.load(Ordering::Acquire));
    assert_eq!(jit.memory_stats().code_bytes_used, 0);
}

#[test]
fn abandoned_codegen_scratch_and_failed_background_definitions_are_released() {
    let jit = JitBackend::new();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        jit.shared
            .compile_with(CompiledCodeKind::Block, 0x1000, 0, |_, _, alloc_slot| {
                alloc_slot();
                panic!("abandoned translation");
            })
    }));
    assert!(panic.is_err());
    assert!(jit.shared.jit(0).slots.is_empty());
    assert!(jit.shared.jit(0).pending_slots.is_empty());
    drop(jit);
    for stage in 1..=4 {
        let jit = JitBackend::new();
        jit.shared
            .fail_compile_stage
            .store(stage, Ordering::Relaxed);
        let handle = jit.tier_up_handle();
        let (_, req) = fixture();
        assert_eq!(jit.tier_up_async(req), TierUpSubmit::Queued);
        handle.wait_idle();
        let module = jit.shared.jit(0);
        assert!(module.functions.is_empty(), "stage {stage}");
        assert!(module.pending_slots.is_empty(), "stage {stage}");
        assert!(module.slots.is_empty(), "stage {stage}");
        assert_eq!(module.module.defined_function_count(), 0);
        assert_eq!(jit.memory_stats().code_bytes_used, 0, "stage {stage}");
    }
}

#[test]
fn deferred_teardown_drains_more_than_one_maintenance_batch() {
    let (_, req) = fixture();
    let TierUpUnit::Block(ir) = req.unit else {
        unreachable!()
    };
    let jit = JitBackend::new();
    let counters = jit.shared.memory.clone();
    let weak = Arc::downgrade(&jit.shared);
    let released = Arc::new(AtomicUsize::new(0));
    let events = released.clone();
    jit.set_code_lifecycle_observer(move |event| {
        if event.kind == CodeLifecycleKind::Release {
            events.fetch_add(1, Ordering::Release);
        }
    });
    for _ in 0..129 {
        assert!(matches!(
            jit.materialize(&ir, MemConsistency::Fast, None, 0),
            CachedBlock::Compiled { .. }
        ));
    }
    drop(jit);
    let deadline = Instant::now() + std::time::Duration::from_secs(5);
    while weak.upgrade().is_some() || counters.snapshot().code_bytes_reserved != 0 {
        assert!(Instant::now() < deadline, "deferred batches did not drain");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(released.load(Ordering::Acquire), 129);
    assert_eq!(counters.snapshot().code_bytes_used, 0);
    assert_eq!(counters.snapshot().code_bytes_reserved, 0);
}
