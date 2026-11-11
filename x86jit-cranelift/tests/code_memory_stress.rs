//! Bounded publication/reclamation campaigns (spec §9.1; decision-17).
#![cfg(all(
    feature = "jit",
    any(target_os = "macos", target_os = "linux", target_os = "freebsd")
))]

#[cfg(any(target_os = "macos", target_os = "linux"))]
#[path = "support/code_memory.rs"]
mod probe;
#[path = "support/stress.rs"]
mod stress;

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;
use x86jit_core::cache::CompiledPtr;
use x86jit_core::jit_abi::run_compiled;
use x86jit_core::lift::{lift_block, CpuMode, FetchAddr};
use x86jit_core::{
    BlockKey, CachedBlock, CpuState, Exit, MemConsistency, Prot, Reg, RegionKind, TierUpRequest,
    TierUpSubmit, TierUpUnit, Vm, VmConfig,
};
use x86jit_cranelift::{CodeLifecycleKind, CodeMemoryPolicy, JitBackend, OptLevel};

const PC: u64 = 0x2000;

fn program(value: u32) -> Vec<u8> {
    let mut bytes = vec![0xb8];
    bytes.extend(value.to_le_bytes());
    bytes.push(0xf4);
    bytes
}

fn entry(block: CachedBlock) -> CompiledPtr {
    let CachedBlock::Compiled { entry } = block else {
        panic!("compiled entry required")
    };
    entry
}

// Keep only outstanding phases and a fixed window of addresses, even in the soak.
#[derive(Default)]
struct Lifecycle {
    phases: HashMap<u64, u8>,
    addresses: VecDeque<usize>,
    published: usize,
    reused: usize,
}

impl Lifecycle {
    fn record(&mut self, event: x86jit_cranelift::CodeLifecycleEvent) {
        let expected = match event.kind {
            CodeLifecycleKind::Publish => 0,
            CodeLifecycleKind::Retire => 1,
            CodeLifecycleKind::Release => 2,
        };
        let phase = self.phases.entry(event.unit.generation).or_default();
        assert_eq!(*phase, expected);
        *phase += 1;
        if event.kind == CodeLifecycleKind::Release {
            self.phases.remove(&event.unit.generation);
        }
        assert!(
            self.phases.len() <= 16,
            "unbounded outstanding lifecycle phases"
        );
        if event.kind == CodeLifecycleKind::Publish {
            self.published += 1;
            self.reused += usize::from(self.addresses.contains(&event.unit.host_start));
            if self.addresses.len() == 16 {
                self.addresses.pop_front();
            }
            self.addresses.push_back(event.unit.host_start);
        }
    }
}

fn dense_supported() -> bool {
    #[cfg(target_os = "macos")]
    // SAFETY: capability query has no arguments or memory preconditions.
    if unsafe { libc::pthread_jit_write_protect_supported_np() } == 0 {
        eprintln!("SKIP dense-provider witness: per-thread JIT write protection unsupported; system-provider fallback applies");
        return false;
    }
    true
}

fn campaign(deadline: Instant, anonymous: bool, cycles: u32) {
    if !dense_supported() {
        return;
    }
    let jit = JitBackend::with_opt_level(OptLevel::Speed);
    let jit = if anonymous {
        jit.with_no_fd_memory_provider(CodeMemoryPolicy::WriteXorExecute)
    } else {
        jit
    };
    let events = Arc::new(Mutex::new(Lifecycle::default()));
    let observed = events.clone();
    jit.set_code_lifecycle_observer(move |event| observed.lock().unwrap().record(event));
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), Box::new(jit));
    vm.set_tier_up_after(None);
    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, &[0xf4]).unwrap();
    vm.write_bytes(PC, &program(1)).unwrap();
    let mut cpu = vm.new_vcpu();
    // A live neighbor forces dense reuse within a retained region. Anonymous
    // sealed batches instead release their entire reservation when wholly free.
    if !anonymous {
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&vm, Some(8)), Exit::Hlt));
    }
    cpu.set_reg(Reg::Rip, PC);
    assert!(matches!(cpu.run(&vm, Some(8)), Exit::Hlt));
    let domain = vm.cache.code_lifetime();
    let key = BlockKey::new(PC, CpuMode::Long64);
    let proceed = AtomicBool::new(false);
    let (tx, rx) = mpsc::channel::<Option<(CompiledPtr, u32)>>();
    let (ack, acknowledgments) = mpsc::channel();
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    let mut baseline_resident = None;
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    let mut baseline_mappings = None;
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    let mut quarters = [(0u64, 0u64, 0u64); 2];
    // Three workload threads plus the idle libtest driver: two vCPUs (one
    // compiles foreground replacements) and backend maintenance.
    std::thread::scope(|scope| {
        let vm = &vm;
        let proceed = &proceed;
        let domain = &domain;
        scope.spawn(move || {
            let mut participant = domain.participant();
            let mut dispatcher = vm.new_vcpu();
            while let Some((pointer, expected)) =
                rx.recv_timeout(stress::remaining(deadline)).unwrap()
            {
                let held = participant.enter();
                assert!(held.admits(pointer));
                ack.send(()).unwrap();
                // The driver releases this gate only after retiring this unit.
                stress::until(deadline, || proceed.load(Ordering::Acquire));
                assert!(domain
                    .retired_units()
                    .iter()
                    .any(|t| t.unit().host_start == pointer.0 as usize));
                let mut cpu = CpuState::new();
                // SAFETY: held pins the admitted allocation through execution
                // after retirement, including its metadata and provider storage.
                unsafe {
                    run_compiled(
                        pointer,
                        &mut cpu,
                        &vm.mem,
                        CpuMode::Long64,
                        MemConsistency::Fast,
                    );
                }
                assert_eq!(cpu.gpr[0], u64::from(expected));
                drop(held);
                // Reuse this vCPU's fast cache across every SMC epoch as well
                // as executing the admitted old generation through the ABI.
                dispatcher.set_reg(Reg::Rip, PC);
                assert!(matches!(dispatcher.run(vm, Some(8)), Exit::Hlt));
                assert_eq!(dispatcher.reg(Reg::Rax), u64::from(expected + 1));
                ack.send(()).unwrap();
            }
        });
        for cycle in 1..=cycles {
            stress::remaining(deadline);
            cpu.set_reg(Reg::Rip, PC);
            assert!(matches!(cpu.run(vm, Some(8)), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rax), u64::from(cycle));
            let stale = entry(vm.cache.get(key).unwrap());
            proceed.store(false, Ordering::Release);
            tx.send(Some((stale, cycle))).unwrap();
            acknowledgments
                .recv_timeout(stress::remaining(deadline))
                .unwrap();
            let generation = vm.cache.publication_generation();
            let ir = lift_block(&vm.mem, FetchAddr::flat(PC), CpuMode::Long64).unwrap();
            let replacement = vm.backend.materialize(&ir, MemConsistency::Fast, None, 0);
            assert!(matches!(replacement, CachedBlock::Compiled { .. }));
            assert!(vm.cache.upgrade(key, replacement, (PC, 6), generation));
            assert!(!domain.retired_units().is_empty());
            // The embedder write path queues SMC; the dispatcher drains it before
            // resolving its old fast entry. Its replacement returns fresh bytes.
            vm.write_bytes(PC, &program(cycle + 1)).unwrap();
            cpu.set_reg(Reg::Rip, PC);
            assert!(matches!(cpu.run(vm, Some(8)), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rax), u64::from(cycle + 1));
            proceed.store(true, Ordering::Release);
            acknowledgments
                .recv_timeout(stress::remaining(deadline))
                .unwrap();
            stress::until(deadline, || {
                vm.memory_stats().backend.code_units_retired == 0
            });
            assert!(!domain.participant().enter().admits(stale));
            let stats = vm.memory_stats().backend;
            assert_eq!(stats.code_allocation_failures, 0);
            assert!(stats.code_bytes_used < 1 << 20, "{stats:?}");
            assert!(stats.code_bytes_committed < 4 << 20, "{stats:?}");
            assert!(stats.code_regions <= 2, "{stats:?}");
            #[cfg(any(target_os = "macos", target_os = "linux"))]
            if cycle % (cycles / 16).max(1) == 0 && cycle >= cycles / 4 {
                let ranges: Vec<_> = domain
                    .live_units()
                    .iter()
                    .map(|u| (u.host_start, u.host_len))
                    .collect();
                let sample = probe::snapshot(&format!("churn-{anonymous}-{cycle}"), &ranges);
                assert!(
                    sample.mappings <= stats.code_mapping_views * 2 + 2,
                    "{sample:?}"
                );
                assert!(sample.resident < 8 << 20, "{sample:?}");
                if let Some(pss) = sample.pss {
                    assert!(pss <= stats.code_bytes_committed + 2 * region::page::size() as u64);
                }
                if let Some(rss) = baseline_resident {
                    assert!(sample.provider_resident <= rss + (32 << 20), "{sample:?}");
                }
                baseline_resident.get_or_insert(sample.provider_resident);
                if let Some(entries) = baseline_mappings {
                    assert!(sample.provider_mappings <= entries + 64, "{sample:?}");
                }
                baseline_mappings.get_or_insert(sample.provider_mappings);
                if cycle > cycles / 2 {
                    let quarter = usize::from(cycle > cycles * 3 / 4);
                    quarters[quarter].0 += sample.provider_resident;
                    quarters[quarter].1 += sample.provider_mappings;
                    quarters[quarter].2 += 1;
                }
            }
            if anonymous {
                // Immutable anonymous batches cannot recycle individual holes.
                // Quiesce the last unit so the complete reservation can release.
                vm.cache.invalidate_overlapping(PC, PC + 6, || {});
                stress::until(deadline, || {
                    vm.memory_stats().backend.code_units_retired == 0
                });
                assert_eq!(vm.memory_stats().backend.code_bytes_committed, 0);
                #[cfg(any(target_os = "linux", target_os = "freebsd"))]
                assert_eq!(vm.memory_stats().backend.code_regions, 0);
            }
        }
        tx.send(None).unwrap();
    });
    vm.cache.invalidate_overlapping(0, u64::MAX, || {});
    stress::until(deadline, || {
        vm.memory_stats().backend.code_units_retired == 0
    });
    let stats = vm.memory_stats().backend;
    assert_eq!(stats.code_units_live, 0);
    assert_eq!(stats.code_bytes_used, 0);
    assert_eq!(stats.code_bytes_committed, 0);
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    assert_eq!(stats.code_regions, 0);
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        let [(rss0, entries0, n0), (rss1, entries1, n1)] = quarters;
        let growth = (rss1 / n1).saturating_sub(rss0 / n0);
        assert!(
            growth <= 1 << 20,
            "late provider residency growth: {growth}"
        );
        assert!(
            entries1 / n1 <= entries0 / n0 + 4,
            "late mapping growth: {quarters:?}"
        );
        eprintln!(
            "MEASURED late-quarter provider residency growth={growth} bytes mappings={}/{}",
            entries0 / n0,
            entries1 / n1
        );
    }
    let events = events.lock().unwrap();
    assert!(events.phases.is_empty());
    assert!(events.published >= cycles as usize * 2);
    if !anonymous {
        assert!(
            events.reused > cycles as usize,
            "dense holes were never reused"
        );
    }
    eprintln!("MEASURED anonymous={anonymous} cycles={cycles} units={} reused={} executions_after_retirement={cycles}", events.published, events.reused);
}

#[test]
fn dense_multi_vcpu_smc_tiering_and_reuse() {
    stress::isolated("dense_multi_vcpu_smc_tiering_and_reuse", |deadline| {
        campaign(deadline, false, 96)
    });
}

#[test]
fn anonymous_multi_vcpu_smc_tiering_and_release() {
    stress::isolated("anonymous_multi_vcpu_smc_tiering_and_release", |deadline| {
        campaign(deadline, true, 96)
    });
}

#[test]
fn relaxed_republication_at_same_address_is_visible() {
    stress::isolated(
        "relaxed_republication_at_same_address_is_visible",
        |deadline| {
            use x86jit_core::{Backend, Memory, MemoryModel};
            let mut memory = Memory::new(MemoryModel::Flat { size: 65536 });
            memory.map(0, 65536, Prot::RWX, RegionKind::Ram).unwrap();
            memory.write_bytes(PC, &program(0)).unwrap();
            if !dense_supported() {
                return;
            }
            let jit = JitBackend::with_opt_level(OptLevel::Speed);
            let ir = lift_block(&memory, FetchAddr::flat(PC), CpuMode::Long64).unwrap();
            let anchor = jit.materialize(&ir, MemConsistency::Fast, None, 0);
            let domain = jit.code_lifetime().unwrap();
            memory.write_bytes(PC, &program(1)).unwrap();
            let first_ir = lift_block(&memory, FetchAddr::flat(PC), CpuMode::Long64).unwrap();
            let first = jit.materialize(&first_ir, MemConsistency::Fast, None, 0);
            let address = entry(first.clone()).0 as usize;
            let version = AtomicUsize::new(0);
            let consumed = AtomicUsize::new(0);
            std::thread::scope(|scope| {
                let version = &version;
                let consumed = &consumed;
                let memory = &memory;
                let reader = scope.spawn(move || {
                    for expected in 1..=128 {
                        // Avoid a yield/sleep after publication: a context switch
                        // could synchronize instructions and mask a missing entry ISB.
                        for poll in 0..50_000_000 {
                            if version.load(Ordering::Relaxed) == expected {
                                break;
                            }
                            if poll % 1024 == 0 {
                                stress::remaining(deadline);
                            }
                            std::hint::spin_loop();
                        }
                        assert_eq!(
                            version.load(Ordering::Relaxed),
                            expected,
                            "publication stalled"
                        );
                        let pointer = address;
                        let mut cpu = CpuState::new();
                        // SAFETY: compiler retains the allocation until consumed;
                        // the fixed address was handed off before thread start; the
                        // reuse notification is deliberately relaxed,
                        // matching generated link loads. Entry synchronizes the ISA.
                        unsafe {
                            run_compiled(
                                CompiledPtr::new(pointer as *const u8),
                                &mut cpu,
                                memory,
                                CpuMode::Long64,
                                MemConsistency::Fast,
                            );
                        }
                        assert_eq!(cpu.gpr[0], expected as u64);
                        consumed.store(expected, Ordering::Release);
                    }
                });
                for value in 1..=128 {
                    memory.write_bytes(PC, &program(value as u32)).unwrap();
                    let ir = lift_block(memory, FetchAddr::flat(PC), CpuMode::Long64).unwrap();
                    let block = if value == 1 {
                        first.clone()
                    } else {
                        jit.materialize(&ir, MemConsistency::Fast, None, 0)
                    };
                    assert_eq!(entry(block.clone()).0 as usize, address);
                    version.store(value, Ordering::Relaxed);
                    stress::until(deadline, || {
                        if consumed.load(Ordering::Acquire) == value {
                            true
                        } else {
                            assert!(
                                !reader.is_finished(),
                                "consumer stopped before acknowledgment"
                            );
                            false
                        }
                    });
                    domain.discard(&block);
                    stress::until(deadline, || jit.memory_stats().code_units_retired == 0);
                }
            });
            domain.discard(&anchor);
            stress::until(deadline, || jit.memory_stats().code_units_retired == 0);
        },
    );
}

#[test]
#[ignore = "2048-cycle, 120s soak; command in backlog/docs/commands.md"]
fn dense_reclamation_soak() {
    stress::isolated_for(
        "dense_reclamation_soak",
        std::time::Duration::from_secs(120),
        |deadline| campaign(deadline, false, 2048),
    );
}

#[test]
fn worker_creation_failures_never_publish() {
    stress::isolated("worker_creation_failures_never_publish", |deadline| {
        use x86jit_core::{Backend, Memory, MemoryModel};
        let mut memory = Memory::new(MemoryModel::Flat { size: 65536 });
        memory.map(0, 65536, Prot::RWX, RegionKind::Ram).unwrap();
        memory.write_bytes(PC, &program(7)).unwrap();
        let ir = Arc::new(lift_block(&memory, FetchAddr::flat(PC), CpuMode::Long64).unwrap());
        for cycle in 0..24 {
            stress::remaining(deadline);
            let mut jit = JitBackend::with_opt_level(OptLevel::Speed);
            let starts = Arc::new(AtomicUsize::new(0));
            let spawned = starts.clone();
            let fail_on = 1 + cycle % 2;
            jit.set_worker_spawner(move |task| {
                if spawned.fetch_add(1, Ordering::Relaxed) + 1 == fail_on {
                    Err(std::io::Error::from_raw_os_error(libc::EAGAIN))
                } else {
                    std::thread::Builder::new().spawn(task)
                }
            });
            let published = Arc::new(AtomicUsize::new(0));
            let observed = published.clone();
            jit.set_code_lifecycle_observer(move |_| {
                observed.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(
                jit.tier_up_async(TierUpRequest {
                    pc: PC,
                    mxcsr_mode: 0,
                    spans: vec![(PC, ir.guest_len)],
                    unit: TierUpUnit::Block(ir.clone()),
                    consistency: MemConsistency::Fast,
                    mmio: None,
                    guest_base: 0,
                    epoch: 0,
                    precise_split_stores: false,
                }),
                TierUpSubmit::Busy
            );
            assert!(jit.code_space_exhausted());
            assert!(jit.tier_up_finished().is_empty());
            assert_eq!(published.load(Ordering::Relaxed), 0);
            assert_eq!(jit.memory_stats().code_bytes_used, 0);
            assert_eq!(starts.load(Ordering::Relaxed), fail_on);
            assert!(matches!(
                jit.materialize(&ir, MemConsistency::Fast, None, 0),
                CachedBlock::Interpreted(_)
            ));
        }
    });
}

#[test]
fn background_completion_rejection_during_smc() {
    stress::isolated("background_completion_rejection_during_smc", |deadline| {
        let jit = JitBackend::with_opt_level(OptLevel::Speed);
        let mut vm = Vm::with_backend(VmConfig::flat(65536), Box::new(jit));
        vm.map(0, 65536, Prot::RWX, RegionKind::Ram).unwrap();
        vm.set_tier_up_after(None);
        vm.write_bytes(PC, &program(0)).unwrap();
        let mut cpu = vm.new_vcpu();
        let key = BlockKey::new(PC, CpuMode::Long64);
        for cycle in 0..64 {
            cpu.set_reg(Reg::Rip, PC);
            assert!(matches!(cpu.run(&vm, Some(8)), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rax), cycle);
            let generation = vm.cache.publication_generation();
            let ir = lift_block(&vm.mem, FetchAddr::flat(PC), CpuMode::Long64).unwrap();
            assert_eq!(
                vm.backend.tier_up_async(TierUpRequest {
                    pc: PC,
                    mxcsr_mode: 0,
                    spans: vec![(PC, ir.guest_len)],
                    unit: TierUpUnit::Block(Arc::new(ir)),
                    consistency: MemConsistency::Fast,
                    mmio: None,
                    guest_base: 0,
                    epoch: vm.cache.epoch(),
                    precise_split_stores: false,
                }),
                TierUpSubmit::Queued
            );
            vm.write_bytes(PC, &program(cycle as u32 + 1)).unwrap();
            // Advance the publication generation before admitting the old
            // completion, regardless of how quickly the worker finishes.
            vm.cache.invalidate_overlapping(PC, PC + 6, || {});
            let mut finished = Vec::new();
            stress::until(deadline, || {
                finished.extend(vm.backend.tier_up_finished());
                !finished.is_empty()
            });
            assert_eq!(finished.len(), 1);
            let block = finished.pop().unwrap().block;
            assert!(matches!(block, CachedBlock::Compiled { .. }));
            assert!(!vm.cache.upgrade(key, block, (PC, 6), generation));
            stress::until(deadline, || {
                vm.memory_stats().backend.code_units_retired == 0
            });
            assert_eq!(vm.memory_stats().backend.code_bytes_used, 0);
        }
    });
}
