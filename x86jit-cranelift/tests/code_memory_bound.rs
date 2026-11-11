//! Decision-17 phase-2 bound, run in its own nextest process on both CI ISAs.
#![cfg(all(feature = "jit", any(target_os = "macos", target_os = "linux")))]
#[path = "support/code_memory.rs"]
mod probe;
use std::sync::{Arc, Mutex};
use x86jit_core::lift::{lift_block, CpuMode, FetchAddr};
use x86jit_core::{
    Backend, CachedBlock, MemConsistency, Memory, MemoryModel, Prot, RegionKind, TierUpRequest,
    TierUpSubmit, TierUpUnit,
};
use x86jit_cranelift::{CodeMemoryPolicy, JitBackend, OptLevel};

fn check_no_fd_batched_bound(units: usize) {
    let mut mem = Memory::new(MemoryModel::Flat { size: 65536 });
    mem.map(0, 65536, Prot::RWX, RegionKind::Ram).unwrap();
    mem.write_bytes(0x1000, &[0xb8, 7, 0, 0, 0, 0xf4]).unwrap();
    let ir = lift_block(&mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
    let jit = JitBackend::with_opt_level(OptLevel::Speed)
        .with_no_fd_memory_provider(CodeMemoryPolicy::WriteXorExecute);
    let ranges = Arc::new(Mutex::new(Vec::with_capacity(units)));
    let observed = ranges.clone();
    jit.set_compiled_code_observer(move |code| {
        observed
            .lock()
            .unwrap()
            .push((code.host_start, code.host_len))
    });
    let handle = jit.tier_up_handle();
    // Deterministic publication batches witness the actual worker path and
    // account for at most one partial-page tail per batch, rather than per unit.
    const BATCH: usize = 64;
    for start in (0..units).step_by(BATCH) {
        let pause = handle.pause_compiler();
        for i in start..(start + BATCH).min(units) {
            let mut ir = ir.clone();
            ir.guest_start = 0x1000 + i as u64 * 16;
            assert_eq!(
                jit.tier_up_async(TierUpRequest {
                    pc: ir.guest_start,
                    mxcsr_mode: 0,
                    spans: vec![(ir.guest_start, ir.guest_len)],
                    unit: TierUpUnit::Block(Arc::new(ir)),
                    consistency: MemConsistency::Fast,
                    mmio: None,
                    guest_base: 0,
                    epoch: 0,
                    precise_split_stores: false,
                }),
                TierUpSubmit::Queued
            );
        }
        drop(pause);
        handle.wait_idle();
        let done = jit.tier_up_finished();
        assert_eq!(done.len(), (units - start).min(BATCH));
        assert!(done
            .iter()
            .all(|unit| matches!(unit.block, CachedBlock::Compiled { .. })));
    }
    let ranges = ranges.lock().unwrap();
    let sample = probe::snapshot(&format!("{units}-unit-no-fd-wx-bound"), &ranges);
    let stats = jit.memory_stats();
    let page = region::page::size() as u64;
    assert_eq!(stats.compiled_units_total, units as u64);
    assert!(sample.mappings <= stats.code_regions * 2 + 2, "{sample:?}");
    let padding = page * (units as u64).div_ceil(BATCH as u64);
    assert!(
        sample.resident <= stats.code_bytes_emitted + 16 * units as u64 + padding,
        "{sample:?} {stats:?}"
    );
    if let Some(pss) = sample.pss {
        assert!(pss <= stats.code_bytes_committed + page);
    }
    assert!(sample.rss <= (64 << 20) + 4096 * units as u64, "{sample:?}");
    assert_eq!(stats.code_regions, 1);
    assert_eq!(stats.code_mapping_views, 1);
    assert_eq!(stats.code_bytes_reserved, 128 << 20);
    assert_eq!(
        stats.code_bytes_used + stats.code_bytes_free,
        stats.code_bytes_reserved
    );
    assert_eq!(stats.code_allocation_failures, 0);
    assert!(stats.code_bytes_committed <= stats.code_bytes_emitted + 16 * units as u64 + padding);
}

#[test]
#[ignore = "100,000-unit storage bound; explicit command in backlog/docs/commands.md"]
fn hundred_thousand_units_have_bounded_mappings_and_residency() {
    check_no_fd_batched_bound(100_000);
}

#[test]
fn anonymous_batched_units_have_bounded_mappings_and_residency() {
    check_no_fd_batched_bound(512);
}

#[test]
fn anonymous_policies_keep_mapping_entries_bounded_and_execute() {
    use x86jit_core::{jit_abi::run_compiled, CpuState};
    let mut memory = Memory::new(MemoryModel::Flat { size: 65536 });
    memory.map(0, 65536, Prot::RWX, RegionKind::Ram).unwrap();
    memory
        .write_bytes(0x1000, &[0xb8, 7, 0, 0, 0, 0xf4])
        .unwrap();
    let mut ir = lift_block(&memory, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
    for policy in [CodeMemoryPolicy::WriteXorExecute]
        .into_iter()
        .chain((!cfg!(target_os = "macos")).then_some(CodeMemoryPolicy::ReadWriteExecute))
    {
        let jit = JitBackend::with_opt_level(OptLevel::Speed).with_no_fd_memory_provider(policy);
        let ranges = Arc::new(Mutex::new(Vec::new()));
        let observed = ranges.clone();
        jit.set_compiled_code_observer(move |code| {
            observed
                .lock()
                .unwrap()
                .push((code.host_start, code.host_len))
        });
        for i in 0..512 {
            ir.guest_start = 0x1000 + i * 16;
            let CachedBlock::Compiled { entry } =
                jit.materialize(&ir, MemConsistency::Fast, None, 0)
            else {
                panic!("expected anonymous compilation");
            };
            let mut cpu = CpuState::new();
            // SAFETY: finalized entries remain retained by this backend.
            unsafe {
                run_compiled(
                    entry,
                    &mut cpu,
                    &memory,
                    CpuMode::Long64,
                    MemConsistency::Fast,
                );
            }
            assert_eq!(cpu.gpr[0], 7);
        }
        let sample = probe::snapshot(
            &format!("512-unit-no-fd-{policy:?}"),
            &ranges.lock().unwrap(),
        );
        let stats = jit.memory_stats();
        assert_eq!(stats.code_mapping_views, stats.code_regions);
        assert!(sample.mappings <= stats.code_regions * 2 + 2, "{sample:?}");
        assert_eq!(stats.code_regions, 1);
    }
}

#[test]
fn relaxed_cross_thread_publication_executes_each_new_unit() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use x86jit_core::{cache::CompiledPtr, jit_abi::run_compiled, CpuState};
    let mut memory = Memory::new(MemoryModel::Flat { size: 65536 });
    memory.map(0, 65536, Prot::RWX, RegionKind::Ram).unwrap();
    let jit = JitBackend::with_opt_level(OptLevel::Speed)
        .with_no_fd_memory_provider(CodeMemoryPolicy::WriteXorExecute);
    let slot = AtomicUsize::new(0);
    let consumed = AtomicUsize::new(0);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    // A relaxed pointer handoff matches generated link-slot loads. The
    // provider/entry protocol must order cache maintenance and each reader ISB.
    std::thread::scope(|scope| {
        let memory = &memory;
        let slot = &slot;
        let consumed = &consumed;
        scope.spawn(move || {
            x86jit_core::jit_abi::synchronize_instruction_stream();
            let mut previous = 0;
            for expected in 1..=1000 {
                let pointer = loop {
                    let pointer = slot.load(Ordering::Relaxed);
                    if pointer != previous {
                        break pointer;
                    }
                    assert!(
                        std::time::Instant::now() < deadline,
                        "publication timed out"
                    );
                    std::thread::yield_now();
                };
                previous = pointer;
                let mut cpu = CpuState::new();
                // SAFETY: compiler retains every unit until both threads exit;
                // the pointer is published only after cache/metadata completion.
                unsafe {
                    run_compiled(
                        CompiledPtr::new(pointer as *const u8),
                        &mut cpu,
                        memory,
                        CpuMode::Long64,
                        MemConsistency::Fast,
                    );
                }
                assert_eq!(cpu.gpr[0], expected);
                consumed.store(expected as usize, Ordering::Release);
            }
        });
        for value in 1u32..=1000 {
            let mut bytes = vec![0xb8];
            bytes.extend_from_slice(&value.to_le_bytes());
            bytes.push(0xf4);
            memory.write_bytes(0x1000, &bytes).unwrap();
            let ir = lift_block(memory, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
            let CachedBlock::Compiled { entry } =
                jit.materialize(&ir, MemConsistency::Fast, None, 0)
            else {
                panic!("expected compilation");
            };
            slot.store(entry.0 as usize, Ordering::Relaxed);
            while consumed.load(Ordering::Acquire) != value as usize {
                assert!(std::time::Instant::now() < deadline, "execution timed out");
                std::thread::yield_now();
            }
        }
    });
}
