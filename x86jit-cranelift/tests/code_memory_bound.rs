//! Decision-17 phase-2 bound, run in its own nextest process on both CI ISAs.
#![cfg(all(feature = "jit", any(target_os = "macos", target_os = "linux")))]
#[path = "support/code_memory.rs"]
mod probe;
use std::sync::{Arc, Mutex};
use x86jit_core::lift::{lift_block, CpuMode, FetchAddr};
use x86jit_core::{Backend, CachedBlock, MemConsistency, Memory, MemoryModel, Prot, RegionKind};
use x86jit_cranelift::{JitBackend, OptLevel};

#[test]
fn hundred_thousand_units_have_bounded_mappings_and_residency() {
    #[cfg(target_os = "macos")]
    // SAFETY: pthread capability query has no preconditions. The default
    // provider on unsupported hosts makes no dense-allocation bound promise.
    if unsafe { libc::pthread_jit_write_protect_supported_np() } == 0 {
        eprintln!("dense memory bound requires per-thread JIT write protection");
        return;
    }
    let mut mem = Memory::new(MemoryModel::Flat { size: 65536 });
    mem.map(0, 65536, Prot::RWX, RegionKind::Ram).unwrap();
    mem.write_bytes(0x1000, &[0xb8, 7, 0, 0, 0, 0xf4]).unwrap();
    let mut ir = lift_block(&mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
    let jit = JitBackend::with_opt_level(OptLevel::Speed);
    let ranges = Arc::new(Mutex::new(Vec::with_capacity(100_000)));
    let observed = ranges.clone();
    jit.set_compiled_code_observer(move |code| {
        observed
            .lock()
            .unwrap()
            .push((code.host_start, code.host_len))
    });
    for i in 0..100_000 {
        ir.guest_start = 0x1000 + i * 16;
        assert!(matches!(
            jit.materialize(&ir, MemConsistency::Fast, None, 0),
            CachedBlock::Compiled { .. }
        ));
    }
    let ranges = ranges.lock().unwrap();
    let sample = probe::snapshot("100000-unit-bound", &ranges);
    let stats = jit.memory_stats();
    let page = region::page::size() as u64;
    assert_eq!(stats.compiled_units_total, 100_000);
    assert!(sample.mappings <= stats.code_regions * 2 + 2, "{sample:?}");
    let resident_views = if cfg!(target_os = "linux") { 2 } else { 1 };
    assert!(
        sample.resident <= resident_views * (stats.code_bytes_emitted + 16 * 100_000 + page),
        "{sample:?} {stats:?}"
    );
    if let Some(pss) = sample.pss {
        assert!(pss <= stats.code_bytes_committed + page);
    }
    assert!(sample.rss <= (64 << 20) + 4096 * 100_000, "{sample:?}");
    assert_eq!(stats.code_regions, 1);
    assert_eq!(stats.code_bytes_reserved, 128 << 20);
    assert_eq!(
        stats.code_bytes_used + stats.code_bytes_free,
        stats.code_bytes_reserved
    );
    assert_eq!(stats.code_allocation_failures, 0);
    assert!(stats.code_bytes_committed <= stats.code_bytes_emitted + 16 * 100_000 + page);
}

#[test]
fn relaxed_cross_thread_publication_executes_each_new_unit() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use x86jit_core::{cache::CompiledPtr, jit_abi::run_compiled, CpuState};
    let mut memory = Memory::new(MemoryModel::Flat { size: 65536 });
    memory.map(0, 65536, Prot::RWX, RegionKind::Ram).unwrap();
    let jit = JitBackend::with_opt_level(OptLevel::Speed);
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
                        CompiledPtr(pointer as *const u8),
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
