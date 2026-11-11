//! Real signal recovery in repeatedly reused code (spec §8.2 and §9.1).
//! Instruction-entry fault state follows SDM Vol 3A §7.5.
#![cfg(all(
    feature = "jit",
    any(target_os = "macos", target_os = "linux", target_os = "freebsd")
))]

#[allow(dead_code)]
#[path = "../src/unwind_test_signal.rs"]
mod signal;
#[path = "support/stress.rs"]
mod stress;

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;
use x86jit_core::lift::{lift_block, CpuMode, FetchAddr};
use x86jit_core::{
    memory::HostRam, BlockKey, CachedBlock, Exit, Knowledge, MemConsistency, Memory, MemoryModel,
    Prot, Reg, RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::{CodeLifecycleKind, JitBackend, OptLevel};

static SAMPLE_PC: AtomicUsize = AtomicUsize::new(0);
static OUTER: AtomicUsize = AtomicUsize::new(0);
static INNER: AtomicUsize = AtomicUsize::new(0);
static SAMPLE_HITS: AtomicUsize = AtomicUsize::new(0);

extern "C" fn sample(signal: libc::c_int) {
    let pc = SAMPLE_PC.load(Ordering::Relaxed);
    if x86jit_core::codemap::lookup_recovery(pc).is_some() {
        SAMPLE_HITS.fetch_add(1, Ordering::Relaxed);
    }
    if signal == libc::SIGUSR1 {
        OUTER.fetch_add(1, Ordering::Relaxed);
        // SAFETY: SIGUSR2 has this adapter installed; it nests inside SIGUSR1.
        unsafe {
            libc::raise(libc::SIGUSR2);
        }
    } else {
        INNER.fetch_add(1, Ordering::Relaxed);
    }
}

fn campaign(deadline: Instant) {
    const CYCLES: usize = 48;
    let page = region::page::size();
    let len = 8 * page;
    let data = (6 * page) as u64;
    let jit = JitBackend::with_opt_level(OptLevel::Speed);
    let events = Arc::new(Mutex::new(Vec::new()));
    let observed = events.clone();
    jit.set_code_lifecycle_observer(move |event| observed.lock().unwrap().push(event));
    // SAFETY: the mapping, protection changes, signal actions and fault handle
    // belong to this isolated process. HostRam releases the complete mapping.
    let mut vm = unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        );
        assert_ne!(ptr, libc::MAP_FAILED);
        Vm::with_backend_host_ram(
            VmConfig::flat(len as u64),
            Box::new(jit),
            HostRam {
                ptr: ptr.cast(),
                len,
                guest_base: 0,
                protect: None,
                dtor: Box::new(|p, n| {
                    libc::munmap(p.cast(), n);
                }),
            },
        )
        .unwrap()
    };
    vm.map(0, len, Prot::RWX, RegionKind::Ram).unwrap();
    vm.set_tier_up_after(None);
    vm.write_bytes(0x100, &[0xf4]).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x100);
    assert!(matches!(cpu.run(&vm, Some(8)), Exit::Hlt));
    signal::HANDLE.store(cpu.fault_handle().as_usize(), Ordering::Relaxed);
    // SAFETY: adapters are confined to this subprocess; every field is initialized.
    unsafe {
        signal::install();
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = sample as *const () as usize;
        libc::sigemptyset(&mut action.sa_mask);
        assert_eq!(
            libc::sigaction(libc::SIGUSR1, &action, std::ptr::null_mut()),
            0
        );
        assert_eq!(
            libc::sigaction(libc::SIGUSR2, &action, std::ptr::null_mut()),
            0
        );
    }
    let (work, requests) = mpsc::channel();
    let (done, completions) = mpsc::channel();
    // Three workload threads plus the idle libtest driver: recovery vCPU,
    // foreground churn compiler and their shared backend maintenance thread.
    std::thread::scope(|scope| {
        let vm = &vm;
        scope.spawn(move || {
            let mut mem = Memory::new(MemoryModel::Flat { size: 65536 });
            mem.map(0, 65536, Prot::RWX, RegionKind::Ram).unwrap();
            mem.write_bytes(0x4000, &[0xf4]).unwrap();
            let ir = lift_block(&mem, FetchAddr::flat(0x4000), CpuMode::Long64).unwrap();
            let domain = vm.cache.code_lifetime();
            for _ in 0..CYCLES {
                requests.recv_timeout(stress::remaining(deadline)).unwrap();
                for _ in 0..4 {
                    let block = vm.backend.materialize(&ir, MemConsistency::Fast, None, 0);
                    assert!(matches!(block, CachedBlock::Compiled { .. }));
                    domain.discard(&block);
                    stress::until(deadline, || {
                        vm.memory_stats().backend.code_units_retired == 0
                    });
                }
                done.send(()).unwrap();
            }
        });
        for cycle in 0..CYCLES {
            stress::remaining(deadline);
            work.send(()).unwrap();
            // Changing guest RIP at the same host extent catches stale source,
            // trap and unwind images after range removal and generation reuse.
            let pc = 0x1000 + (cycle % 2) as u64 * 0x100;
            let mut code = vec![0xb8];
            code.extend((cycle as u32 + 1).to_le_bytes());
            code.extend([0x48, 0x8b, 0x1f, 0xf4]);
            vm.write_bytes(pc, &code).unwrap();
            cpu.set_reg(Reg::Rip, pc);
            cpu.set_reg(Reg::Rdi, data);
            cpu.set_reg(Reg::Rbx, 0xfeed);
            // SAFETY: page-aligned data page in the retained HostRam mapping.
            unsafe {
                assert_eq!(
                    libc::mprotect(
                        (vm.mem.host_base() as *mut u8).add(data as usize).cast(),
                        page,
                        libc::PROT_NONE
                    ),
                    0
                );
            }
            let faults = signal::FAULTS.load(Ordering::Relaxed);
            let exit = cpu.run(vm, Some(16));
            let metadata = exit
                .fault_metadata()
                .expect("host fault must recover through landing pad");
            assert_eq!(metadata.instruction_rip(), Knowledge::Known(pc + 5));
            assert_eq!(cpu.reg(Reg::Rip), pc + 5);
            assert_eq!(cpu.reg(Reg::Rax), cycle as u64 + 1);
            assert_eq!(cpu.reg(Reg::Rbx), 0xfeed);
            assert_eq!(signal::FAULTS.load(Ordering::Relaxed), faults + 1);
            let raw = metadata.raw_host_fault().unwrap();
            assert_eq!(
                raw.fault_address.or(raw.signal_address),
                Some(vm.mem.host_base() as u64 + data)
            );
            let CachedBlock::Compiled { entry } =
                vm.cache.get(BlockKey::new(pc, CpuMode::Long64)).unwrap()
            else {
                panic!("fault must execute compiled code")
            };
            let unit = vm.cache.code_lifetime().unit_at(entry.0 as usize).unwrap();
            let trap_pc = (unit.host_start..unit.host_start + unit.host_len as usize)
                .find(|host_pc| {
                    x86jit_core::codemap::lookup_trap(*host_pc)
                        .is_some_and(|site| site.guest_rip == pc + 5)
                })
                .unwrap();
            SAMPLE_PC.store(trap_pc, Ordering::Relaxed);
            // SAFETY: both sample handlers are installed in this process.
            unsafe {
                assert_eq!(libc::raise(libc::SIGUSR1), 0);
            }
            // SAFETY: restore only the owned data page, then retry the fault.
            unsafe {
                assert_eq!(
                    libc::mprotect(
                        (vm.mem.host_base() as *mut u8).add(data as usize).cast(),
                        page,
                        libc::PROT_READ | libc::PROT_WRITE
                    ),
                    0
                );
            }
            let resumed = cpu.run(vm, Some(16));
            assert!(matches!(resumed, Exit::Hlt));
            assert!(resumed.fault_metadata().is_none());
            vm.cache.invalidate_overlapping(0x1000, 0x2000, || {});
            stress::until(deadline, || {
                vm.memory_stats().backend.code_units_retired == 0
            });
            completions
                .recv_timeout(stress::remaining(deadline))
                .unwrap();
        }
    });
    assert_eq!(signal::FAULTS.load(Ordering::Relaxed), CYCLES);
    assert_eq!(OUTER.load(Ordering::Relaxed), CYCLES);
    assert_eq!(INNER.load(Ordering::Relaxed), CYCLES);
    assert_eq!(SAMPLE_HITS.load(Ordering::Relaxed), 2 * CYCLES);
    let events = events.lock().unwrap();
    let mut seen = HashSet::new();
    let mut reused = 0;
    for event in events.iter().filter(|event| {
        event.kind == CodeLifecycleKind::Publish && matches!(event.code.guest_rip, 0x1000 | 0x1100)
    }) {
        if !seen.insert(event.unit.host_start) {
            reused += 1;
        }
    }
    if vm.memory_stats().backend.code_provider_fallback == 0 {
        assert!(
            reused >= CYCLES - 4,
            "fault recovery did not exercise reused code: {reused}"
        );
    } else {
        eprintln!("SKIP dense-reuse assertion: system-provider fallback active");
    }
    eprintln!("MEASURED recovered_faults={CYCLES} nested_signal_pairs={CYCLES} reused_fault_units={reused}");
}

#[test]
fn signals_and_fault_recovery_in_reused_code() {
    stress::isolated("signals_and_fault_recovery_in_reused_code", campaign);
}

static READER_STAGE: AtomicUsize = AtomicUsize::new(0);
static NESTED_READERS: AtomicUsize = AtomicUsize::new(0);

extern "C" fn held_sample(signal: libc::c_int) {
    let pc = SAMPLE_PC.load(Ordering::Relaxed);
    let mut visited = false;
    let hit = x86jit_core::codemap::lookup_probed_for_test(pc, || {
        if visited {
            return;
        }
        visited = true;
        NESTED_READERS.fetch_add(1, Ordering::Relaxed);
        if signal == libc::SIGUSR1 {
            // SAFETY: the inner handler is installed; the outer reader remains held.
            unsafe { libc::raise(libc::SIGUSR2) };
        } else {
            READER_STAGE.store(1, Ordering::Release);
            while READER_STAGE.load(Ordering::Acquire) == 1 {
                std::hint::spin_loop();
            }
        }
    });
    if hit.is_some() {
        SAMPLE_HITS.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn nested_readers_delay_provider_release_of_retired_unit() {
    stress::isolated(
        "nested_readers_delay_provider_release_of_retired_unit",
        |deadline| {
            use x86jit_core::{Backend, MemoryModel};
            let mut jit = JitBackend::with_opt_level(OptLevel::Speed);
            struct StartMaintenance(Arc<AtomicUsize>);
            impl Drop for StartMaintenance {
                fn drop(&mut self) {
                    self.0.store(1, Ordering::Release);
                }
            }
            let maintenance = StartMaintenance(Arc::new(AtomicUsize::new(0)));
            let gate = maintenance.0.clone();
            // Park automatic maintenance so each explicit collection below runs
            // a complete pass, rather than racing the collector's reentry guard.
            jit.set_worker_spawner(move |task| {
                let gate = gate.clone();
                std::thread::Builder::new().spawn(move || {
                    while gate.load(Ordering::Acquire) == 0 {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                    task();
                })
            });
            let mut mem = Memory::new(MemoryModel::Flat { size: 65536 });
            mem.map(0, 65536, Prot::RWX, RegionKind::Ram).unwrap();
            mem.write_bytes(0x1000, &[0xb8, 7, 0, 0, 0, 0xf4]).unwrap();
            let ir = lift_block(&mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
            let released = Arc::new(AtomicUsize::new(0));
            let observed = released.clone();
            jit.set_code_lifecycle_observer(move |event| {
                if event.kind == CodeLifecycleKind::Release {
                    observed.fetch_add(1, Ordering::Release);
                }
            });
            let block = jit.materialize(&ir, MemConsistency::Fast, None, 0);
            let CachedBlock::Compiled { entry } = block else {
                panic!("compiled entry required")
            };
            let domain = jit.code_lifetime().unwrap();
            let unit = domain.unit_at(entry.0 as usize).unwrap();
            let sampled_pc = (unit.host_start..unit.host_start + unit.host_len as usize)
                .find(|pc| x86jit_core::codemap::lookup(*pc).is_some())
                .unwrap();
            SAMPLE_PC.store(sampled_pc, Ordering::Relaxed);
            // SAFETY: this isolated child owns the two signal dispositions.
            unsafe {
                let mut action: libc::sigaction = std::mem::zeroed();
                action.sa_sigaction = held_sample as *const () as usize;
                libc::sigemptyset(&mut action.sa_mask);
                for signal in [libc::SIGUSR1, libc::SIGUSR2] {
                    assert_eq!(libc::sigaction(signal, &action, std::ptr::null_mut()), 0);
                }
            }
            std::thread::scope(|scope| {
                let reader = scope.spawn(|| {
                    // SAFETY: both sample handlers are installed in this subprocess.
                    assert_eq!(unsafe { libc::raise(libc::SIGUSR1) }, 0);
                });
                struct ResumeReaders;
                impl Drop for ResumeReaders {
                    fn drop(&mut self) {
                        READER_STAGE.store(2, Ordering::Release);
                    }
                }
                let resume = ResumeReaders;
                stress::until(deadline, || READER_STAGE.load(Ordering::Acquire) == 1);
                assert_eq!(NESTED_READERS.load(Ordering::Relaxed), 2);
                domain.discard(&block);
                // Wait for actual range removal, then explicitly drive collection;
                // no execution participant can account for the retained allocation.
                stress::until(deadline, || {
                    jit.collect_retired_metadata();
                    !x86jit_core::codemap::contains(unit.host_start)
                });
                for _ in 0..32 {
                    jit.collect_retired_metadata();
                    assert_eq!(
                        released.load(Ordering::Acquire),
                        0,
                        "provider release preceded nested reader exit"
                    );
                    let stats = jit.memory_stats();
                    assert_eq!(stats.code_units_retired, 1);
                    assert!(stats.code_bytes_used > 0);
                }
                drop(resume);
                reader.join().unwrap();
            });
            stress::until(deadline, || {
                jit.collect_retired_metadata();
                jit.memory_stats().code_units_retired == 0
            });
            assert_eq!(released.load(Ordering::Acquire), 1);
            assert_eq!(jit.memory_stats().code_bytes_used, 0);
            assert_eq!(SAMPLE_HITS.load(Ordering::Relaxed), 2);
            drop(maintenance);
            eprintln!("MEASURED two nested readers retained provider storage through range removal and 32 collector passes");
        },
    );
}
