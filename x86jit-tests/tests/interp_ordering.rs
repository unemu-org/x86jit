//! Interpreter-only ordering witnesses for SDM Vol 3A §11.2.2.
use iced_x86::code_asm::*;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use x86jit_core::{Exit, MemConsistency, Prot, Reg, RegionKind, Vm, VmConfig};

const CODE: u64 = 0x1000;
const X: u64 = 0x8000;
const Y: u64 = 0x8100;

/// Fewest litmus trials a deadline-shortened run must still complete.
const MIN_TRIALS: usize = 200;

#[test]
fn standalone_compiled_ram_store_replays_inside_mmio_window() {
    use x86jit_core::jit_abi::{call_block, run_compiled, MemCtx, RET_MMIO_DEFER};
    use x86jit_core::lift::{lift_block, FetchAddr};
    use x86jit_core::{CachedBlock, CpuMode, StepResult};
    use x86jit_cranelift::JitBackend;

    for tier in [MemConsistency::AcqRel, MemConsistency::FullTso] {
        let mut config = VmConfig::flat(0x10000);
        config.consistency = tier;
        let mut vm = Vm::with_backend(config, Box::new(JitBackend::new()));
        vm.map(CODE, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
        vm.map(X, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
        for addr in [0x7000, 0x9000] {
            vm.map(addr, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        }
        // The generated window check defers this RAM store between Trap regions.
        vm.write_bytes(CODE, &[0x48, 0x89, 0x03, 0xf4]).unwrap();
        let ir = lift_block(&vm.mem, FetchAddr::flat(CODE), CpuMode::Long64).unwrap();
        let CachedBlock::Compiled { entry } = vm.backend.materialize(
            &ir,
            vm.consistency,
            vm.mem.trap_window(),
            vm.mem.guest_base(),
        ) else {
            panic!("expected a compiled block");
        };
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, CODE);
        cpu.set_reg(Reg::Rbx, X);
        cpu.set_reg(Reg::Rax, 0x42);
        let mut ctx = MemCtx::for_memory(&vm.mem);
        // SAFETY: vm owns the entry and its matching live memory/context.
        assert_eq!(
            unsafe { call_block(entry, &mut cpu.cpu, &mut ctx) },
            RET_MMIO_DEFER
        );
        assert_eq!(cpu.reg(Reg::Rip), CODE);
        assert_eq!(vm.mem.read(X, 8).unwrap(), 0);
        // SAFETY: the same live entry was compiled for this VM's ordering tier.
        let result = unsafe {
            run_compiled(
                entry,
                &mut cpu.cpu,
                &vm.mem,
                CpuMode::Long64,
                vm.consistency,
            )
        };
        assert!(matches!(result, StepResult::Continue));
        assert_eq!(cpu.reg(Reg::Rip), CODE + 3);
        assert_eq!(vm.mem.read(X, 8).unwrap(), 0x42);
    }
}

fn program(writer: bool, helper_write: bool, sb: bool) -> Vec<u8> {
    let mut a = CodeAssembler::new(64).unwrap();
    if sb {
        a.mov(qword_ptr(rbx), 1).unwrap();
        a.mov(rax, qword_ptr(rdx)).unwrap();
    } else if writer {
        if helper_write {
            a.rep().stosb().unwrap();
        } else {
            a.mov(qword_ptr(X), 1).unwrap();
        }
        a.mov(qword_ptr(Y), 1).unwrap();
    } else {
        a.mov(rax, qword_ptr(Y)).unwrap();
        a.mov(rcx, qword_ptr(X)).unwrap();
    }
    a.hlt().unwrap();
    a.assemble(CODE + if writer { 0 } else { 0x100 }).unwrap()
}

#[derive(Default)]
struct Rendezvous {
    epoch: AtomicUsize,
    done: [AtomicUsize; 2],
    result: [AtomicU64; 2],
    stop: AtomicBool,
}

fn wait_yield(spins: &mut u32) {
    *spins += 1;
    if *spins == 64 {
        thread::yield_now();
        *spins = 0;
    } else {
        std::hint::spin_loop();
    }
}

fn race(tier: MemConsistency, helper_write: bool, sb: bool) -> usize {
    const ITERS: usize = 20_000;
    let iters = ITERS;
    let mut config = VmConfig::flat(0x10000);
    config.consistency = tier;
    let mut vm = Vm::new(config);
    vm.map(0, 0x10000, Prot::RW, RegionKind::Ram).unwrap();
    for id in 0..2 {
        let code = program(id == 0, helper_write, sb);
        vm.write_bytes(CODE + id * 0x100, &code).unwrap();
    }
    // Materialize before the race: neither compilation nor a first code-page
    // registration should add incidental synchronization to the measured pair.
    for id in 0..2 {
        let mut warm = vm.new_vcpu();
        warm.set_reg(Reg::Rip, CODE + id * 0x100);
        warm.set_reg(Reg::Rcx, 1);
        warm.set_reg(Reg::Rax, 1);
        warm.set_reg(Reg::Rdi, X);
        warm.set_reg(Reg::Rbx, X);
        warm.set_reg(Reg::Rdx, Y);
        warm.set_reg(Reg::R8, 1);
        assert!(matches!(warm.run(&vm, Some(8)), Exit::Hlt));
    }
    let vm = Arc::new(vm);
    let sync = Arc::new(Rendezvous::default());
    let deadline = Instant::now() + Duration::from_secs(8);
    let handles: Vec<_> = (0..2)
        .map(|id| {
            let vm = vm.clone();
            let sync = sync.clone();
            thread::spawn(move || {
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rbx, if id == 0 { X } else { Y });
                cpu.set_reg(Reg::Rdx, if id == 0 { Y } else { X });
                for epoch in 1..=iters {
                    let mut spins = 0;
                    while sync.epoch.load(Ordering::Acquire) != epoch {
                        if sync.stop.load(Ordering::Relaxed) || Instant::now() >= deadline {
                            sync.stop.store(true, Ordering::Relaxed);
                            return;
                        }
                        wait_yield(&mut spins);
                    }
                    cpu.set_reg(Reg::Rip, CODE + id as u64 * 0x100);
                    cpu.set_reg(Reg::Rcx, 1);
                    cpu.set_reg(Reg::Rax, 1);
                    cpu.set_reg(Reg::Rdi, X);
                    cpu.set_reg(Reg::R8, 1);
                    assert!(matches!(cpu.run(&vm, Some(8)), Exit::Hlt));
                    let result = if !sb {
                        u64::from(cpu.reg(Reg::Rax) == 1 && cpu.reg(Reg::Rcx) == 0)
                    } else {
                        cpu.reg(Reg::Rax)
                    };
                    sync.result[id].store(result, Ordering::Relaxed);
                    sync.done[id].store(epoch, Ordering::Release);
                }
            })
        })
        .collect();
    let mut relaxed = 0;
    let mut completed = 0;
    let mut timed_out = false;
    for epoch in 1..=iters {
        vm.mem.write(X, 0, 8).unwrap();
        vm.mem.write(Y, 0, 8).unwrap();
        sync.epoch.store(epoch, Ordering::Release);
        let mut spins = 0;
        while sync.done.iter().any(|d| d.load(Ordering::Acquire) != epoch) {
            if Instant::now() >= deadline {
                timed_out = true;
                break;
            }
            wait_yield(&mut spins);
        }
        if timed_out {
            break;
        }
        completed += 1;
        relaxed += usize::from(if !sb {
            sync.result[1].load(Ordering::Relaxed) == 1
        } else {
            sync.result.iter().all(|r| r.load(Ordering::Relaxed) == 0)
        });
    }
    sync.stop.store(true, Ordering::Relaxed);
    for handle in handles {
        handle.join().unwrap();
    }
    // A loaded host (the full suite runs tests in parallel) can make the
    // rendezvous slow; the deadline then ends the sample early instead of
    // failing, as long as enough trials ran to be meaningful.
    assert!(
        !timed_out || completed >= MIN_TRIALS,
        "rendezvous timed out after {completed} trials: {tier:?}, helper_write={helper_write}, sb={sb}"
    );
    eprintln!(
        "{tier:?}, helper_write={helper_write}, sb={sb}: {relaxed}/{completed} forbidden outcomes"
    );
    relaxed
}

#[test]
fn acqrel_interpreter_message_passing() {
    // SDM Vol 3A §11.2.2: stores and loads retain their respective orders;
    // §11.2.4.1 item 3 also orders string stores with other stores.
    for helper_write in [false, true] {
        assert_eq!(race(MemConsistency::AcqRel, helper_write, false), 0);
    }
}

#[test]
fn fulltso_interpreter_message_passing() {
    for helper_write in [false, true] {
        assert_eq!(race(MemConsistency::FullTso, helper_write, false), 0);
    }
}

#[test]
fn child_and_interpreter_message_passing() {
    use x86jit_core::{child, ChildArena, ChildStep, CpuMode};
    // SDM Vol 3A §11.2.2, §11.2.4.1: publishing Y after X must not let
    // the other processor observe Y=1, X=0, including a string-store writer.
    const ITERS: usize = 2_000;
    for tier in [MemConsistency::AcqRel, MemConsistency::FullTso] {
        for helper_write in [false, true] {
            for child_writer in [false, true] {
                let mut config = VmConfig::flat(0x10000);
                config.consistency = tier;
                let mut vm = Vm::new(config);
                vm.map(0, 0x10000, Prot::RW, RegionKind::Ram).unwrap();
                for id in 0..2 {
                    vm.write_bytes(CODE + id * 0x100, &program(id == 0, helper_write, false))
                        .unwrap();
                    let mut warm = vm.new_vcpu();
                    warm.cpu.rip = CODE + id * 0x100;
                    warm.cpu.gpr[0] = 1;
                    warm.cpu.gpr[1] = 1;
                    warm.cpu.gpr[7] = X;
                    assert!(matches!(warm.run(&vm, Some(8)), Exit::Hlt));
                }
                let view = vm.child_view(CpuMode::Long64);
                let sync = Rendezvous::default();
                let deadline = Instant::now() + Duration::from_secs(8);
                thread::scope(|scope| {
                    let handles: Vec<_> = (0..2)
                        .map(|id| {
                            let vm = &vm;
                            let view = &view;
                            let sync = &sync;
                            scope.spawn(move || {
                                let mut cpu = vm.new_vcpu();
                                let mut arena = ChildArena::new(2);
                                for epoch in 1..=ITERS {
                                    let mut spins = 0;
                                    while sync.epoch.load(Ordering::Acquire) != epoch {
                                        if sync.stop.load(Ordering::Relaxed)
                                            || Instant::now() >= deadline
                                        {
                                            sync.stop.store(true, Ordering::Relaxed);
                                            return;
                                        }
                                        wait_yield(&mut spins);
                                    }
                                    cpu.cpu.rip = CODE + id as u64 * 0x100;
                                    cpu.cpu.gpr[0] = 1;
                                    cpu.cpu.gpr[1] = 1;
                                    cpu.cpu.gpr[7] = X;
                                    if (id == 0) == child_writer {
                                        for instruction in 0..3 {
                                            let result =
                                                child::step(view, &mut cpu.cpu, &mut arena);
                                            assert!(matches!(
                                                (instruction, result),
                                                (0 | 1, ChildStep::Continue) | (2, ChildStep::Hlt)
                                            ));
                                        }
                                    } else {
                                        assert!(matches!(cpu.run(vm, Some(8)), Exit::Hlt));
                                    }
                                    sync.result[id].store(
                                        u64::from(cpu.cpu.gpr[0] == 1 && cpu.cpu.gpr[1] == 0),
                                        Ordering::Relaxed,
                                    );
                                    sync.done[id].store(epoch, Ordering::Release);
                                }
                            })
                        })
                        .collect();
                    let mut forbidden = 0;
                    let mut completed = 0;
                    let mut timed_out = false;
                    for epoch in 1..=ITERS {
                        vm.mem.write(X, 0, 8).unwrap();
                        vm.mem.write(Y, 0, 8).unwrap();
                        sync.epoch.store(epoch, Ordering::Release);
                        let mut spins = 0;
                        while sync
                            .done
                            .iter()
                            .any(|done| done.load(Ordering::Acquire) != epoch)
                        {
                            if Instant::now() >= deadline {
                                timed_out = true;
                                break;
                            }
                            wait_yield(&mut spins);
                        }
                        if timed_out {
                            break;
                        }
                        completed += 1;
                        forbidden += sync.result[1].load(Ordering::Relaxed);
                    }
                    sync.stop.store(true, Ordering::Relaxed);
                    for handle in handles {
                        handle.join().unwrap();
                    }
                    assert!(
                        !timed_out || completed >= MIN_TRIALS,
                        "child MP rendezvous timed out after {completed} trials: {tier:?}, helper={helper_write}, child_writer={child_writer}"
                    );
                    eprintln!("{tier:?}, helper={helper_write}, child_writer={child_writer}: {forbidden}/{completed} forbidden outcomes");
                    assert_eq!(forbidden, 0);
                });
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn fulltso_interpreter_closes_store_buffering_window() {
    // SDM Vol 3A §11.2.3.4 permits SB (0,0) on x86. The ARM FullTso
    // implementation deliberately strengthens this with a post-store full fence,
    // matching JIT gstore; AcqRel and the unchanged x86 path may allow (0,0).
    assert_eq!(race(MemConsistency::FullTso, false, true), 0);
}
