//! Opt-in fetch approval, independent of host page size and advisory region Prot.
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use x86jit_core::{
    AccessKind, Backend, Exit, InterpreterBackend, Knowledge, MemoryEffects, Prot, Reg, RegionCaps,
    RegionKind, StepResult, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

use x86jit_core::FetchVerdict;

const PAGE: u64 = 0x1000;
fn vm(tier: usize) -> Vm {
    let backend: Box<dyn Backend> = match tier {
        0 => Box::new(InterpreterBackend),
        1 | 3 | 4 => Box::new(JitBackend::new()),
        _ => Box::new(JitBackend::with_superblocks(RegionCaps {
            max_blocks: 16,
            max_icount: 256,
        })),
    };
    let mut vm = Vm::with_backend(VmConfig::flat(0x5000), backend);
    vm.map(0, 0x5000, Prot::RWX, RegionKind::Ram).unwrap();
    if tier >= 3 {
        vm.set_tier_up_after(Some(1));
    }
    if tier >= 4 {
        vm.set_tier_up_background(true);
    }
    vm
}
fn refused(exit: &Exit, addr: u64, rip: u64, present: bool) {
    assert!(
        matches!(exit, Exit::UnmappedMemory { addr: a, access: AccessKind::Execute, .. } if *a == addr),
        "{exit:?}"
    );
    // SDM Vol 3A §5.7, Fig. 5-12: P distinguishes protection from absence.
    // I/D depends on paging controls; the independent fetch fact is always known.
    // SDM Vol 3A §7.15, Event 14: IP names the instruction; CR2 the denied byte.
    let m = exit.fault_metadata().unwrap();
    assert_eq!(m.instruction_rip(), Knowledge::Known(rip));
    assert_eq!(m.saved_rip(), rip);
    let projection = m.page_fault_projection().unwrap();
    assert!(projection.is_fetch());
    assert_eq!(projection.error_code().bit(0), Some(present));
    assert_eq!(projection.error_code().bit(4), None);
    assert_eq!(m.progress().ram(), MemoryEffects::None);
}
#[test]
fn fetch_policy_refusal_every_tier() {
    for tier in 0..6 {
        for verdict in [FetchVerdict::DenyProtection, FetchVerdict::DenyNotPresent] {
            let mut vm = vm(tier);
            vm.write_bytes(PAGE + 7, &[0xb8, 42, 0, 0, 0, 0xf4])
                .unwrap();
            vm.set_fetch_policy(Box::new(move |page| {
                assert_eq!(page, PAGE);
                verdict
            }));
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, PAGE + 7);
            cpu.set_reg(Reg::Rax, 99);
            let exit = cpu.run(&vm, Some(100));
            refused(
                &exit,
                PAGE + 7,
                PAGE + 7,
                verdict == FetchVerdict::DenyProtection,
            );
            assert_eq!(cpu.reg(Reg::Rip), PAGE + 7);
            assert_eq!(cpu.reg(Reg::Rax), 99);
        }
    }
}
#[test]
fn fetch_policy_revocation_evicts_disjoint_same_page_code() {
    for tier in 0..6 {
        let mut vm = vm(tier);
        let entry = PAGE + 0x100;
        vm.write_bytes(entry, &[0xb8, 42, 0, 0, 0, 0xf4]).unwrap();
        let allow = Arc::new(AtomicBool::new(true));
        let calls = Arc::new(AtomicUsize::new(0));
        vm.set_fetch_policy(Box::new({
            let allow = allow.clone();
            let calls = calls.clone();
            move |_| {
                calls.fetch_add(1, Ordering::Relaxed);
                if allow.load(Ordering::Acquire) {
                    FetchVerdict::Allow
                } else {
                    FetchVerdict::DenyProtection
                }
            }
        }));
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, entry);
        assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        allow.store(false, Ordering::Release);
        // The notification is disjoint from every cached instruction byte.
        vm.mem.note_write(PAGE, 1);
        cpu.set_reg(Reg::Rip, entry);
        cpu.set_reg(Reg::Rax, 99);
        refused(&cpu.run(&vm, Some(100)), entry, entry, true);
        assert_eq!(cpu.reg(Reg::Rax), 99);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }
}
#[test]
fn fetch_policy_crossing_retains_prefix_every_tier() {
    for tier in 0..6 {
        for crossing in [false, true] {
            let mut vm = vm(tier);
            let fault_rip = 2 * PAGE - if crossing { 2 } else { 0 };
            let start = fault_rip - 3;
            vm.write_bytes(start, &[0x48, 0xff, 0xc3, 0xb8, 42, 0, 0, 0, 0xf4])
                .unwrap();
            vm.set_fetch_policy(Box::new(|page| {
                if page == 2 * PAGE {
                    FetchVerdict::DenyNotPresent
                } else {
                    FetchVerdict::Allow
                }
            }));
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, start);
            cpu.set_reg(Reg::Rax, 99);
            let exit = cpu.run(&vm, Some(100));
            refused(&exit, 2 * PAGE, fault_rip, false);
            assert_eq!(cpu.reg(Reg::Rip), fault_rip);
            assert_eq!(cpu.reg(Reg::Rbx), 1);
            assert_eq!(cpu.reg(Reg::Rax), 99);
        }
    }
}
#[test]
fn fetch_policy_invalidation_forgets_approval_every_tier() {
    for tier in 0..6 {
        let mut vm = vm(tier);
        vm.write_bytes(PAGE, &[0xb8, 42, 0, 0, 0, 0xf4]).unwrap();
        let allow = Arc::new(AtomicBool::new(true));
        let calls = Arc::new(AtomicUsize::new(0));
        vm.set_fetch_policy(Box::new({
            let allow = allow.clone();
            let calls = calls.clone();
            move |_| {
                calls.fetch_add(1, Ordering::Relaxed);
                if allow.load(Ordering::Acquire) {
                    FetchVerdict::Allow
                } else {
                    FetchVerdict::DenyProtection
                }
            }
        }));
        let mut cpu = vm.new_vcpu();
        for _ in 0..6 {
            cpu.set_reg(Reg::Rip, PAGE);
            assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
        }
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        allow.store(false, Ordering::Release);
        vm.write_bytes(PAGE, &[0xb8]).unwrap();
        cpu.set_reg(Reg::Rip, PAGE);
        cpu.set_reg(Reg::Rax, 99);
        refused(&cpu.run(&vm, Some(100)), PAGE, PAGE, true);
        assert_eq!(cpu.reg(Reg::Rax), 99);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }
}
#[test]
fn fetch_policy_step_refuses_crossing() {
    let mut vm = vm(1);
    vm.write_bytes(2 * PAGE - 2, &[0xb8, 42, 0, 0, 0]).unwrap();
    vm.set_fetch_policy(Box::new(|page| {
        if page == 2 * PAGE {
            FetchVerdict::DenyProtection
        } else {
            FetchVerdict::Allow
        }
    }));
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 2 * PAGE - 2);
    cpu.set_reg(Reg::Rax, 99);
    let StepResult::Exit(exit) = cpu.step_instruction(&vm) else {
        panic!("instruction must refuse its second page")
    };
    refused(&exit, 2 * PAGE, 2 * PAGE - 2, true);
    assert_eq!(cpu.reg(Reg::Rax), 99);
}
#[test]
fn fetch_policy_absent_preserves_advisory_protection() {
    for tier in 0..3 {
        let vm = vm(tier);
        vm.write_bytes(PAGE, &[0xf4]).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, PAGE);
        assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
    }
}

fn invalidate(vm: &Vm, addr: u64) {
    vm.cache
        .invalidate_overlapping(addr, addr + PAGE, || vm.mem.clear_code_page(addr / PAGE));
    vm.backend.invalidate_links();
}

#[test]
fn fetch_policy_compiled_chain_and_region_invalidation() {
    for tier in [1, 2] {
        let mut vm = vm(tier);
        // Two blocks forming a loop, followed by a successor on another page.
        vm.write_bytes(
            PAGE,
            &[
                0xb9, 3, 0, 0, 0, // mov ecx,3
                0xeb, 0, // jmp body
                0x48, 0xff, 0xc3, // inc rbx
                0xff, 0xc9, // dec ecx
                0x75, 0xf9, // jnz body
                0xe9, 0xed, 0x0f, 0, 0, // jmp 2000h
            ],
        )
        .unwrap();
        vm.write_bytes(2 * PAGE, &[0xb8, 42, 0, 0, 0, 0xf4])
            .unwrap();
        let deny = Arc::new(AtomicBool::new(false));
        vm.set_fetch_policy(Box::new({
            let deny = deny.clone();
            move |page| {
                if page == 2 * PAGE && deny.load(Ordering::Acquire) {
                    FetchVerdict::DenyProtection
                } else {
                    FetchVerdict::Allow
                }
            }
        }));
        let mut cpu = vm.new_vcpu();
        for _ in 0..4 {
            cpu.set_reg(Reg::Rip, PAGE);
            assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
        }
        if tier == 2 {
            assert!(vm.cache.regions() > 0);
        } else {
            assert!(vm.cache.chained() > 0);
        }
        deny.store(true, Ordering::Release);
        // A second vCPU drains the protection-change notification.
        vm.mem.note_write(2 * PAGE, 1);
        vm.write_bytes(0, &[0xf4]).unwrap();
        let mut other = vm.new_vcpu();
        other.set_reg(Reg::Rip, 0);
        assert!(matches!(other.run(&vm, Some(100)), Exit::Hlt));
        cpu.set_reg(Reg::Rip, PAGE);
        cpu.set_reg(Reg::Rax, 99);
        cpu.set_reg(Reg::Rbx, 0);
        refused(&cpu.run(&vm, Some(100)), 2 * PAGE, 2 * PAGE, true);
        assert_eq!(cpu.reg(Reg::Rax), 99);
        assert_eq!(cpu.reg(Reg::Rbx), 3);
    }
}

#[test]
fn fetch_policy_denied_region_tail_keeps_complete_instructions() {
    let mut vm = vm(2);
    vm.write_bytes(
        PAGE,
        &[
            0xb9, 3, 0, 0, 0, 0xeb, 0, 0xff, 0xc9, 0x75, 0xfc, 0xe9, 0xeb, 0x0f, 0,
            0, // jmp 1ffbh
        ],
    )
    .unwrap();
    vm.write_bytes(2 * PAGE - 5, &[0x48, 0xff, 0xc3, 0xb8, 42, 0, 0, 0, 0xf4])
        .unwrap();
    vm.set_fetch_policy(Box::new(|page| {
        if page == 2 * PAGE {
            FetchVerdict::DenyProtection
        } else {
            FetchVerdict::Allow
        }
    }));
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, PAGE);
    cpu.set_reg(Reg::Rax, 99);
    refused(&cpu.run(&vm, Some(100)), 2 * PAGE, 2 * PAGE - 2, true);
    assert!(vm.cache.regions() > 0);
    assert_eq!(cpu.reg(Reg::Rbx), 1);
    assert_eq!(cpu.reg(Reg::Rax), 99);
}

#[test]
fn fetch_policy_unmap_same_page_interpreter() {
    unmap_same_page(0);
}

#[test]
fn fetch_policy_unmap_same_page_jit() {
    unmap_same_page(1);
}

#[test]
fn fetch_policy_unmap_same_page_region() {
    unmap_same_page(2);
}

fn unmap_same_page(tier: usize) {
    for chained in [false, true] {
        let mut vm = vm(tier);
        vm.unmap(0, 0x5000).unwrap();
        vm.map(0, PAGE as usize, Prot::RWX, RegionKind::Ram)
            .unwrap();
        vm.map(PAGE, 0x800, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(PAGE + 0x800, 0x800, Prot::RWX, RegionKind::Ram)
            .unwrap();
        vm.write_bytes(0, &[0xe9, 0xfb, 0x0f, 0, 0]).unwrap(); // jmp 1000h
        vm.write_bytes(
            PAGE,
            &[
                0xb9, 3, 0, 0, 0, // mov ecx,3
                0xeb, 0, // jmp body
                0xff, 0xc9, // dec ecx
                0x75, 0xfc, // jnz body
                0xb8, 42, 0, 0, 0, 0xf4,
            ],
        )
        .unwrap();
        let deny = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        vm.set_fetch_policy(Box::new({
            let deny = deny.clone();
            let calls = calls.clone();
            move |page| {
                if page == PAGE {
                    calls.fetch_add(1, Ordering::Relaxed);
                    if deny.load(Ordering::Acquire) {
                        return FetchVerdict::DenyProtection;
                    }
                }
                FetchVerdict::Allow
            }
        }));
        let start = if chained { 0 } else { PAGE };
        let mut cpu = vm.new_vcpu();
        for _ in 0..4 {
            cpu.set_reg(Reg::Rip, start);
            assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rax), 42);
        }
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        if tier == 2 {
            assert!(vm.cache.regions() > 0);
        } else if tier == 1 {
            assert!(vm.cache.chained() > 0);
        }
        deny.store(true, Ordering::Release);
        vm.unmap(PAGE + 0x800, 0x800).unwrap();
        cpu.set_reg(Reg::Rip, start);
        cpu.set_reg(Reg::Rax, 99);
        refused(&cpu.run(&vm, Some(100)), PAGE, PAGE, true);
        assert_eq!(cpu.reg(Reg::Rip), PAGE);
        assert_eq!(cpu.reg(Reg::Rax), 99);
        assert!(calls.load(Ordering::Relaxed) > 1);
    }
}

#[test]
fn fetch_policy_clear_unmap_replace_and_fork_forget_approvals() {
    for clear in 0..3 {
        let mut vm = vm(0);
        vm.write_bytes(PAGE, &[0x90, 0xf4]).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        vm.set_fetch_policy(Box::new({
            let calls = calls.clone();
            move |_| {
                calls.fetch_add(1, Ordering::Relaxed);
                FetchVerdict::Allow
            }
        }));
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, PAGE);
        assert!(matches!(cpu.step_instruction(&vm), StepResult::Continue));
        // Exercise approval reset without a cached block.
        match clear {
            0 => vm.mem.clear_code_page(1),
            1 => vm.mem.clear_all_code_pages(),
            _ => {
                vm.unmap(0, 0x5000).unwrap();
                vm.map(0, 0x5000, Prot::RWX, RegionKind::Ram).unwrap();
            }
        }
        cpu.set_reg(Reg::Rip, PAGE);
        assert!(matches!(cpu.step_instruction(&vm), StepResult::Continue));
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        let child = vm.fork_with_backend(Box::new(InterpreterBackend)).unwrap();
        let mut child_cpu = child.new_vcpu();
        child_cpu.set_reg(Reg::Rip, PAGE);
        assert!(matches!(child_cpu.run(&child, Some(100)), Exit::Hlt));
        assert_eq!(calls.load(Ordering::Relaxed), 3);
        cpu.set_reg(Reg::Rip, PAGE);
        assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
        vm.set_fetch_policy(Box::new(|_| FetchVerdict::DenyNotPresent));
        cpu.set_reg(Reg::Rip, PAGE);
        refused(&cpu.run(&vm, Some(100)), PAGE, PAGE, false);
    }
}

#[test]
fn fetch_policy_callback_invalidation_on_another_vcpu() {
    use std::{sync::mpsc, time::Duration};
    for tier in [0, 1, 2] {
        let mut vm = vm(tier);
        vm.write_bytes(0, &[0xf4]).unwrap();
        vm.write_bytes(PAGE, &[0xb8, 42, 0, 0, 0, 0xf4]).unwrap();
        let deny = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = std::sync::Mutex::new(release_rx);
        vm.set_fetch_policy(Box::new({
            let deny = deny.clone();
            let calls = calls.clone();
            move |page| {
                if page == PAGE {
                    if calls.fetch_add(1, Ordering::Relaxed) == 0 {
                        entered_tx.send(()).unwrap();
                        release_rx.lock().unwrap().recv().unwrap();
                        // Simulate an answer obtained before the protection change.
                        return FetchVerdict::Allow;
                    }
                    if deny.load(Ordering::Acquire) {
                        return FetchVerdict::DenyProtection;
                    }
                }
                FetchVerdict::Allow
            }
        }));
        let vm = Arc::new(vm);
        let worker = std::thread::spawn({
            let vm = vm.clone();
            move || {
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, PAGE);
                cpu.set_reg(Reg::Rax, 99);
                let exit = cpu.run(&vm, Some(100));
                (exit, cpu.reg(Reg::Rax))
            }
        });
        let entered = entered_rx.recv_timeout(Duration::from_secs(5));
        if entered.is_ok() {
            deny.store(true, Ordering::Release);
            // No cache span exists yet. The other vCPU processes invalidation
            // while the callback blocks; this also detects held engine locks.
            vm.mem.note_write(PAGE, 1);
            let mut other = vm.new_vcpu();
            other.set_reg(Reg::Rip, 0);
            assert!(matches!(other.run(&vm, Some(100)), Exit::Hlt));
        }
        let _ = release_tx.send(());
        let (exit, rax) = worker.join().unwrap();
        entered.expect("fetch callback must be reached");
        refused(&exit, PAGE, PAGE, true);
        assert_eq!(rax, 99);
        assert!(calls.load(Ordering::Relaxed) >= 2);
    }
}

#[test]
fn fetch_policy_background_discards_invalidated_work() {
    for regions in [false, true] {
        for completed in [false, true] {
            let jit = if regions {
                JitBackend::with_superblocks(RegionCaps {
                    max_blocks: 16,
                    max_icount: 256,
                })
            } else {
                JitBackend::new()
            };
            let handle = jit.tier_up_handle();
            let pause = handle.pause_compiler();
            let compiled = Arc::new(AtomicUsize::new(0));
            jit.set_compiled_code_observer({
                let compiled = compiled.clone();
                move |_| {
                    compiled.fetch_add(1, Ordering::Relaxed);
                }
            });
            let mut vm = Vm::with_backend(VmConfig::flat(0x4000), Box::new(jit));
            vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(
                PAGE,
                &[
                    0xb9, 3, 0, 0, 0, 0xeb, 0, 0xff, 0xc9, 0x75, 0xfc, 0xe9, 0xf0, 0x0f, 0,
                    0, // jmp 2000h
                ],
            )
            .unwrap();
            vm.write_bytes(2 * PAGE, &[0xb8, 42, 0, 0, 0, 0xf4])
                .unwrap();
            vm.write_bytes(0, &[0xf4]).unwrap();
            vm.set_tier_up_after(Some(1));
            vm.set_tier_up_background(true);
            let deny = Arc::new(AtomicBool::new(false));
            vm.set_fetch_policy(Box::new({
                let deny = deny.clone();
                move |page| {
                    if page == 2 * PAGE && deny.load(Ordering::Acquire) {
                        FetchVerdict::DenyProtection
                    } else {
                        FetchVerdict::Allow
                    }
                }
            }));
            let mut cpu = vm.new_vcpu();
            for _ in 0..2 {
                cpu.set_reg(Reg::Rip, PAGE);
                assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
            }
            assert!(vm.cache.tier_pending_len() > 0);
            let mut pause = Some(pause);
            if completed {
                drop(pause.take());
                handle.wait_idle();
                assert!(compiled.load(Ordering::Relaxed) > 0);
            }
            deny.store(true, Ordering::Release);
            // Invalidate all approved translations while the compiler is paused or
            // its finished results are awaiting publication, using another vCPU.
            vm.mem.note_write(PAGE, 1);
            vm.mem.note_write(2 * PAGE, 1);
            let mut other = vm.new_vcpu();
            other.set_reg(Reg::Rip, 0);
            assert!(matches!(other.run(&vm, Some(100)), Exit::Hlt));
            drop(pause.take());
            handle.wait_idle();
            if !completed {
                assert_eq!(
                    compiled.load(Ordering::Relaxed),
                    0,
                    "invalidated queued work must not compile"
                );
            }
            cpu.set_reg(Reg::Rip, PAGE);
            cpu.set_reg(Reg::Rax, 99);
            refused(&cpu.run(&vm, Some(100)), 2 * PAGE, 2 * PAGE, true);
            assert_eq!(cpu.reg(Reg::Rax), 99);
            assert!(vm.cache.tier_bg_rejected() > 0);
        }
    }
}

#[test]
fn fetch_policy_replay_rechecks_after_invalidation() {
    for tier in [0, 1] {
        let mut vm = vm(tier);
        vm.unmap(0, 0x5000).unwrap();
        vm.map(PAGE, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
        vm.map(3 * PAGE, 0x1000, Prot::RW, RegionKind::Trap)
            .unwrap();
        vm.write_bytes(PAGE, &[0x48, 0x8b, 0x03, 0xf4]).unwrap();
        let deny = Arc::new(AtomicBool::new(false));
        vm.set_fetch_policy(Box::new({
            let deny = deny.clone();
            move |_| {
                if deny.load(Ordering::Acquire) {
                    FetchVerdict::DenyProtection
                } else {
                    FetchVerdict::Allow
                }
            }
        }));
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, PAGE);
        cpu.set_reg(Reg::Rbx, 3 * PAGE);
        cpu.set_reg(Reg::Rax, 99);
        assert!(matches!(cpu.run(&vm, Some(100)), Exit::MmioRead { .. }));
        cpu.complete_mmio_read(42);
        deny.store(true, Ordering::Release);
        invalidate(&vm, PAGE);
        refused(&cpu.run(&vm, Some(100)), PAGE, PAGE, true);
        assert_eq!(cpu.reg(Reg::Rax), 99);
    }
}

#[test]
fn fetch_policy_step_write_notification_forgets_approval() {
    let mut vm = vm(0);
    vm.write_bytes(PAGE, &[0x90]).unwrap();
    let deny = Arc::new(AtomicBool::new(false));
    vm.set_fetch_policy(Box::new({
        let deny = deny.clone();
        move |_| {
            if deny.load(Ordering::Acquire) {
                FetchVerdict::DenyNotPresent
            } else {
                FetchVerdict::Allow
            }
        }
    }));
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, PAGE);
    assert!(matches!(cpu.step_instruction(&vm), StepResult::Continue));
    deny.store(true, Ordering::Release);
    vm.mem.note_write(PAGE, 1);
    cpu.set_reg(Reg::Rip, PAGE);
    let StepResult::Exit(exit) = cpu.step_instruction(&vm) else {
        panic!("step must observe invalidation")
    };
    refused(&exit, PAGE, PAGE, false);
}

#[test]
fn fetch_policy_background_region_denied_tail_keeps_prefix() {
    let jit = JitBackend::with_superblocks(RegionCaps {
        max_blocks: 16,
        max_icount: 256,
    });
    let handle = jit.tier_up_handle();
    let pause = handle.pause_compiler();
    let mut vm = Vm::with_backend(VmConfig::flat(0x4000), Box::new(jit));
    vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.set_tier_up_after(Some(1));
    vm.set_tier_up_background(true);
    vm.write_bytes(
        PAGE,
        &[
            0xb9, 3, 0, 0, 0, 0xeb, 0, 0xff, 0xc9, 0x75, 0xfc, 0xe9, 0xeb, 0x0f, 0, 0,
        ],
    )
    .unwrap();
    vm.write_bytes(2 * PAGE - 5, &[0x48, 0xff, 0xc3, 0xb8, 42, 0, 0, 0, 0xf4])
        .unwrap();
    vm.set_fetch_policy(Box::new(|page| {
        if page == 2 * PAGE {
            FetchVerdict::DenyNotPresent
        } else {
            FetchVerdict::Allow
        }
    }));
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, PAGE);
    cpu.set_reg(Reg::Rax, 99);
    refused(&cpu.run(&vm, Some(100)), 2 * PAGE, 2 * PAGE - 2, false);
    assert!(vm.cache.tier_pending_len() > 0);
    drop(pause);
    handle.wait_idle();
    cpu.set_reg(Reg::Rip, PAGE);
    cpu.set_reg(Reg::Rbx, 0);
    refused(&cpu.run(&vm, Some(100)), 2 * PAGE, 2 * PAGE - 2, false);
    assert_eq!(cpu.reg(Reg::Rbx), 1);
    assert_eq!(cpu.reg(Reg::Rax), 99);
    assert!(vm.cache.regions() > 0);
    assert!(vm.cache.tier_bg_published() > 0);
}

#[test]
fn fetch_policy_concurrent_approval_is_cached_once() {
    let mut vm = vm(0);
    vm.write_bytes(PAGE, &[0xf4]).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    vm.set_fetch_policy(Box::new({
        let calls = calls.clone();
        move |_| {
            calls.fetch_add(1, Ordering::Relaxed);
            FetchVerdict::Allow
        }
    }));
    let vm = Arc::new(vm);
    let start = Arc::new(std::sync::Barrier::new(3));
    let workers: Vec<_> = (0..2)
        .map(|_| {
            let vm = vm.clone();
            let start = start.clone();
            std::thread::spawn(move || {
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, PAGE);
                start.wait();
                assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
            })
        })
        .collect();
    start.wait();
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

/// Opt-in measurement: compile many distinct short blocks, timing lift separately
/// from codegen. Run once in release mode with --ignored --nocapture.
#[test]
#[ignore = "compile-heavy fetch-policy cost measurement"]
fn fetch_policy_lift_cost() {
    use std::{hint::black_box, time::Instant};
    use x86jit_core::{
        lift::{lift_block, FetchAddr},
        CpuMode,
    };
    const BLOCKS: u64 = 2048;
    const STRIDE: u64 = 256;
    fn sample(policy: bool) -> (u128, u128) {
        let span = PAGE + BLOCKS * STRIDE;
        let mut vm = Vm::with_backend(VmConfig::flat(span), Box::new(JitBackend::new()));
        vm.map(0, span as usize, Prot::RX, RegionKind::Ram).unwrap();
        for i in 0..BLOCKS {
            vm.write_bytes(
                PAGE + i * STRIDE,
                &[
                    0xb8, 42, 0, 0, 0, // mov eax,42
                    0x48, 0x83, 0xc0, 1, // add rax,1
                    0x48, 0x83, 0xe8, 1, // sub rax,1
                    0xf4,
                ],
            )
            .unwrap();
        }
        let calls = Arc::new(AtomicUsize::new(0));
        if policy {
            vm.set_fetch_policy(Box::new({
                let calls = calls.clone();
                move |_| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    FetchVerdict::Allow
                }
            }));
        }
        let mut lift_ns = 0;
        let started = Instant::now();
        for i in 0..BLOCKS {
            let lift_started = Instant::now();
            let ir =
                lift_block(&vm.mem, FetchAddr::flat(PAGE + i * STRIDE), CpuMode::Long64).unwrap();
            lift_ns += lift_started.elapsed().as_nanos();
            black_box(vm.backend.materialize(&ir, vm.consistency, None, 0));
        }
        let total_ns = started.elapsed().as_nanos();
        assert_eq!(
            calls.load(Ordering::Relaxed),
            if policy {
                (BLOCKS * STRIDE / PAGE) as usize
            } else {
                0
            }
        );
        (lift_ns, total_ns)
    }
    let mut off = Vec::new();
    let mut on = Vec::new();
    // Paired samples alternate order; the first pair warms both paths.
    for round in 0..6 {
        let (a, b) = if round % 2 == 0 {
            (sample(false), sample(true))
        } else {
            let b = sample(true);
            (sample(false), b)
        };
        if round > 0 {
            off.push(a);
            on.push(b);
        }
    }
    fn median(samples: &[(u128, u128)], total: bool) -> u128 {
        let mut values: Vec<_> = samples
            .iter()
            .map(|v| if total { v.1 } else { v.0 })
            .collect();
        values.sort_unstable();
        values[values.len() / 2]
    }
    let a = median(&off, false);
    let b = median(&on, false);
    println!(
        "blocks={BLOCKS}, pages={}, paired_samples=5",
        BLOCKS * STRIDE / PAGE
    );
    println!(
        "lift ns/block: no policy={:.1}, allow policy={:.1}, delta={:+.2}%",
        a as f64 / BLOCKS as f64,
        b as f64 / BLOCKS as f64,
        (b as f64 / a as f64 - 1.0) * 100.0
    );
    println!(
        "lift+compile ms: no policy={:.3}, allow policy={:.3}",
        median(&off, true) as f64 / 1e6,
        median(&on, true) as f64 / 1e6
    );
}

#[test]
fn fetch_policy_invalidation_before_first_publication_retries() {
    use std::{
        sync::{mpsc, Mutex},
        time::Duration,
    };
    use x86jit_core::{ir::IrBlock, CachedBlock, MemConsistency};
    struct PausedBackend {
        inner: Box<dyn Backend>,
        first: AtomicBool,
        entered: mpsc::Sender<()>,
        release: Mutex<mpsc::Receiver<()>>,
    }
    impl Backend for PausedBackend {
        fn code_lifetime(
            &self,
        ) -> Option<std::sync::Arc<x86jit_core::code_lifetime::CodeLifetime>> {
            self.inner.code_lifetime()
        }

        fn materialize(
            &self,
            ir: &IrBlock,
            consistency: MemConsistency,
            mmio: Option<(u64, u64)>,
            guest_base: u64,
        ) -> CachedBlock {
            if self.first.swap(false, Ordering::AcqRel) {
                self.entered.send(()).unwrap();
                self.release.lock().unwrap().recv().unwrap();
            }
            self.inner.materialize(ir, consistency, mmio, guest_base)
        }
    }
    for compiling in [false, true] {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let inner: Box<dyn Backend> = if compiling {
            Box::new(JitBackend::new())
        } else {
            Box::new(InterpreterBackend)
        };
        let mut vm = Vm::with_backend(
            VmConfig::flat(0x3000),
            Box::new(PausedBackend {
                inner,
                first: AtomicBool::new(true),
                entered: entered_tx,
                release: Mutex::new(release_rx),
            }),
        );
        vm.map(0, 0x3000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.set_time_accounting(true);
        vm.write_bytes(0, &[0xf4]).unwrap();
        vm.write_bytes(PAGE, &[0xb8, 42, 0, 0, 0, 0xf4]).unwrap();
        let deny = Arc::new(AtomicBool::new(false));
        vm.set_fetch_policy(Box::new({
            let deny = deny.clone();
            move |page| {
                if page == PAGE && deny.load(Ordering::Acquire) {
                    FetchVerdict::DenyProtection
                } else {
                    FetchVerdict::Allow
                }
            }
        }));
        let vm = Arc::new(vm);
        let worker = std::thread::spawn({
            let vm = vm.clone();
            move || {
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, PAGE);
                cpu.set_reg(Reg::Rax, 99);
                let exit = cpu.run(&vm, Some(100));
                (exit, cpu.reg(Reg::Rax))
            }
        });
        let entered = entered_rx.recv_timeout(Duration::from_secs(5));
        if entered.is_ok() {
            deny.store(true, Ordering::Release);
            vm.mem.note_write(PAGE, 1);
            let mut other = vm.new_vcpu();
            other.set_reg(Reg::Rip, 0);
            assert!(matches!(other.run(&vm, Some(100)), Exit::Hlt));
        }
        let _ = release_tx.send(());
        let (exit, rax) = worker.join().unwrap();
        entered.unwrap();
        refused(&exit, PAGE, PAGE, true);
        assert_eq!(rax, 99);
        let stats = vm.time_stats();
        assert_eq!(stats.fetch_translate.calls, 2);
        assert_eq!(stats.block_misses, 3);
        assert_eq!(stats.fetch_misses, 2);
        assert_eq!(stats.fetch_hits.calls, 0);
        assert!(vm
            .cache
            .get(x86jit_core::BlockKey::new(
                PAGE,
                x86jit_core::CpuMode::Long64
            ))
            .is_none());
    }
}

#[test]
fn fetch_policy_backend_without_cancellation_compiles_inline() {
    use x86jit_core::{ir::IrBlock, CachedBlock, MemConsistency, TierUpRequest, TierUpSubmit};
    struct LegacyBackend(JitBackend);
    impl Backend for LegacyBackend {
        fn code_lifetime(
            &self,
        ) -> Option<std::sync::Arc<x86jit_core::code_lifetime::CodeLifetime>> {
            self.0.code_lifetime()
        }

        fn materialize(
            &self,
            ir: &IrBlock,
            consistency: MemConsistency,
            mmio: Option<(u64, u64)>,
            guest_base: u64,
        ) -> CachedBlock {
            self.0.materialize(ir, consistency, mmio, guest_base)
        }
        fn tier_up_async(&self, _request: TierUpRequest) -> TierUpSubmit {
            panic!("a policy must not submit work without a cancellation witness")
        }
    }
    let mut vm = Vm::with_backend(
        VmConfig::flat(0x2000),
        Box::new(LegacyBackend(JitBackend::new())),
    );
    vm.map(PAGE, PAGE as usize, Prot::RX, RegionKind::Ram)
        .unwrap();
    vm.write_bytes(PAGE, &[0xb8, 42, 0, 0, 0, 0xf4]).unwrap();
    vm.set_tier_up_after(Some(1));
    vm.set_tier_up_background(true);
    vm.set_fetch_policy(Box::new(|_| FetchVerdict::Allow));
    let mut cpu = vm.new_vcpu();
    for _ in 0..3 {
        cpu.set_reg(Reg::Rip, PAGE);
        assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rax), 42);
    }
    assert!(matches!(
        vm.cache.get(x86jit_core::BlockKey::new(
            PAGE,
            x86jit_core::CpuMode::Long64
        )),
        Some(CachedBlock::Compiled { .. })
    ));
}
