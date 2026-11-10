//! Publication witnesses for the Cranelift 0.115 background finalization batch.
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
    assert_eq!(
        stats.code_bytes_reserved,
        (pages.iter().map(|pages| pages.len()).sum::<usize>() * region::page::size()) as u64
    );
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
    let module = jit.shared.jit(0);
    let entry = CompiledPtr(module.module.get_finalized_function(pending.id));
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
