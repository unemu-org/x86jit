//! Background tier-up: the dispatcher submits a hot block to the backend's compiler
//! thread and swaps it in when it lands, instead of compiling inline. Opt-in
//! (`Vm::set_tier_up_background`), so these tests drive it explicitly; the default-off
//! corpus is unaffected. Determinism comes from
//! `JitBackend::tier_up_handle().wait_idle()` — no sleeps.

use iced_x86::code_asm::*;
use x86jit_core::{
    Backend, Exit, InterpreterBackend, Prot, Reg, RegionCaps, RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const CODE: u64 = 0x1000;

fn vm_with(backend: Box<dyn Backend>, tier: u32, background: bool, prog: &[u8]) -> Vm {
    vm_with_flat(backend, tier, background, prog, 0x2000)
}

fn vm_with_flat(
    backend: Box<dyn Backend>,
    tier: u32,
    background: bool,
    prog: &[u8],
    flat: u64,
) -> Vm {
    let mut vm = Vm::with_backend(VmConfig::flat(flat), backend);
    vm.set_tier_up_after(Some(tier));
    vm.set_tier_up_background(background);
    vm.map(CODE, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
    vm.write_bytes(CODE, prog).unwrap();
    vm
}

/// `mov eax, 42 ; hlt` — one block, re-runnable by resetting RIP.
fn single_block() -> Vec<u8> {
    single_block_val(42)
}

/// `mov eax, <v> ; hlt` at `CODE`, six bytes — so overwriting one payload with
/// another keeps the block length identical (SMC in place).
fn single_block_val(v: i32) -> Vec<u8> {
    let mut a = CodeAssembler::new(64).unwrap();
    a.mov(eax, v).unwrap();
    a.hlt().unwrap();
    a.assemble(CODE).unwrap()
}

/// Heat the single block until it submits its background compile: pass 1 caches it
/// (no bump), passes 2-3 bump to 2, pass 4 bumps to 3 (== threshold) and submits.
fn heat_to_submit(vm: &Vm, cpu: &mut x86jit_core::Vcpu, expect: u64) {
    for _ in 0..4 {
        assert_eq!(one_pass(vm, cpu), expect);
    }
}

/// A counted accumulation loop: eax = sum(1..=n). Its body block runs `n` times, so it
/// tiers up and (under background mode) switches to compiled mid-run.
fn accumulate_loop(n: i32) -> Vec<u8> {
    let mut a = CodeAssembler::new(64).unwrap();
    let mut top = a.create_label();
    a.mov(eax, 0i32).unwrap();
    a.mov(ecx, n).unwrap();
    a.set_label(&mut top).unwrap();
    a.add(eax, ecx).unwrap();
    a.sub(ecx, 1i32).unwrap();
    a.jnz(top).unwrap();
    a.hlt().unwrap();
    a.assemble(CODE).unwrap()
}

/// Run to `hlt` and return RAX.
fn run_to_hlt(vm: &Vm) -> u64 {
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    assert!(matches!(cpu.run(vm, Some(50_000_000)), Exit::Hlt), "hlts");
    cpu.reg(Reg::Rax)
}

/// One pass through the single block (reset RIP, run to hlt), returning RAX.
fn one_pass(vm: &Vm, cpu: &mut x86jit_core::Vcpu) -> u64 {
    cpu.set_reg(Reg::Rip, CODE);
    assert!(matches!(cpu.run(vm, Some(1_000_000)), Exit::Hlt), "hlts");
    cpu.reg(Reg::Rax)
}

/// The deterministic tier-up recipe. With threshold 3 and background on, the block
/// stays interpreted (published == 0) through the submit, `wait_idle`
/// compiles it off-thread, and the next dispatch publishes it (published == 1) — with
/// RAX identical to the interpreter throughout, no sleeps or timing.
#[test]
fn deterministic_background_tier_up() {
    let prog = single_block();
    let jit = JitBackend::new();
    let handle = jit.tier_up_handle();
    let vm = vm_with(Box::new(jit), 3, true, &prog);
    let mut cpu = vm.new_vcpu();

    // Four passes: run1 caches (no bump), runs 2-3 bump to 2 (< 3), run4 bumps to 3 and
    // submits the compile in the background. Still interpreted, nothing published.
    for _ in 0..4 {
        assert_eq!(one_pass(&vm, &mut cpu), 42);
    }
    assert_eq!(
        vm.cache.tier_bg_published(),
        0,
        "submitted, not yet published"
    );

    handle.wait_idle(); // the worker finishes the compile — still not published
    assert_eq!(
        vm.cache.tier_bg_published(),
        0,
        "compiled, awaiting a drain"
    );

    // The next dispatch drains the completion and publishes it.
    assert_eq!(one_pass(&vm, &mut cpu), 42);
    assert_eq!(
        vm.cache.tier_bg_published(),
        1,
        "published on the next dispatch"
    );
    assert_eq!(vm.cache.tier_bg_rejected(), 0, "no stale rejections");

    // Final state matches the interpreter oracle.
    assert_eq!(
        run_to_hlt(&vm_with(Box::new(InterpreterBackend), 3, true, &prog)),
        42
    );
}

/// A real loop under background tier-up produces the interpreter's result and
/// actually publishes at least one background compile (the body block tiers up mid-run).
#[test]
fn real_loop_background_matches_interp_and_publishes() {
    let prog = accumulate_loop(100_000);
    let interp = run_to_hlt(&vm_with(Box::new(InterpreterBackend), 2, false, &prog));

    let jit = JitBackend::new();
    let handle = jit.tier_up_handle();
    let vm = vm_with(Box::new(jit), 2, true, &prog);
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    assert!(matches!(cpu.run(&vm, Some(50_000_000)), Exit::Hlt));
    handle.wait_idle(); // drain any straggler so the count is stable

    assert_eq!(
        cpu.reg(Reg::Rax),
        interp,
        "background result matches interp"
    );
    assert!(
        vm.cache.tier_bg_published() > 0,
        "the hot loop body tiered up in the background"
    );
}

/// With a region-forming backend AND background tier-up on, a hot loop
/// tiers up to a **background-compiled REGION** (a multi-block superblock), not just a
/// single block — off the vcpu, and only for the proven-hot loop (the eager inline
/// region path is gated off when bg is on). The result matches the interpreter.
#[test]
fn hot_loop_tiers_up_to_a_background_region() {
    const CAPS: RegionCaps = RegionCaps {
        max_blocks: 16,
        max_icount: 256,
    };
    let prog = accumulate_loop(100_000);
    let interp = run_to_hlt(&vm_with(Box::new(InterpreterBackend), 2, false, &prog));

    let jit = JitBackend::with_superblocks(CAPS);
    let handle = jit.tier_up_handle();
    let vm = vm_with(Box::new(jit), 2, true, &prog);
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    assert!(matches!(cpu.run(&vm, Some(50_000_000)), Exit::Hlt));
    handle.wait_idle(); // finish any straggler compile
    cpu.set_reg(Reg::Rip, CODE); // one more run drains + exercises the published region
    assert!(matches!(cpu.run(&vm, Some(50_000_000)), Exit::Hlt));

    assert_eq!(
        cpu.reg(Reg::Rax),
        interp,
        "background region matches interp"
    );
    assert!(
        vm.cache.tier_bg_published() > 0,
        "the hot loop tiered up in the background"
    );
    assert!(
        vm.cache.regions() > 0,
        "it formed a background REGION, not a single block"
    );
}

/// Build a region-forming VM with adaptive tiering: single-block tier at `t1`, region
/// tier at the higher backedge threshold `t2`.
fn vm_adaptive(backend: Box<dyn Backend>, t1: u32, t2: u32, prog: &[u8]) -> Vm {
    let mut vm = Vm::with_backend(VmConfig::flat(0x2000), backend);
    vm.set_tier_up_after(Some(t1));
    vm.set_tier_up_region_after(Some(t2));
    vm.set_tier_up_background(true);
    vm.map(CODE, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
    vm.write_bytes(CODE, prog).unwrap();
    vm
}

const ADAPT_CAPS: RegionCaps = RegionCaps {
    max_blocks: 16,
    max_icount: 256,
};

/// Adaptive per-block tiering self-selects the tier — a SHORT hot loop
/// (fewer iterations than the region threshold T2) never forms a region (it would be a
/// wasted heavy compile), a LONG hot loop crosses T2 and tiers up to one. No mode
/// switch: the same VM config picks the right tier from the loop's own execution count.
#[test]
fn adaptive_tier_forms_a_region_only_for_a_long_loop() {
    const T1: u32 = 2;
    const T2: u32 = 1_000;

    // Short loop: 100 iterations < T2 → stays interpreted, no region.
    let short = accumulate_loop(100);
    let jit = JitBackend::with_superblocks(ADAPT_CAPS);
    let handle = jit.tier_up_handle();
    let vm = vm_adaptive(Box::new(jit), T1, T2, &short);
    let interp_short = run_to_hlt(&vm_with(Box::new(InterpreterBackend), 2, false, &short));
    let got = run_to_hlt(&vm);
    handle.wait_idle();
    assert_eq!(got, interp_short, "short-loop result matches interp");
    assert_eq!(
        vm.cache.regions(),
        0,
        "a short loop must NOT form a region (below T2)"
    );

    // Long loop: 100k iterations ≫ T2 → crosses the region threshold, forms a region.
    let long = accumulate_loop(100_000);
    let jit = JitBackend::with_superblocks(ADAPT_CAPS);
    let handle = jit.tier_up_handle();
    let vm = vm_adaptive(Box::new(jit), T1, T2, &long);
    let interp_long = run_to_hlt(&vm_with(Box::new(InterpreterBackend), 2, false, &long));
    let got = run_to_hlt(&vm);
    handle.wait_idle();
    assert_eq!(got, interp_long, "long-loop result matches interp");
    assert!(
        vm.cache.regions() > 0,
        "a long loop must tier up to a region (crossed T2)"
    );
}

/// The interpreter backend with the background flag on returns `Unsupported` from
/// `tier_up_async`, so a hot block falls through to inline tier-up — behaving exactly
/// like the flag-off path (identical result, nothing published in the background).
#[test]
fn interp_backend_background_falls_back_to_inline() {
    let prog = accumulate_loop(1_000);
    let off = run_to_hlt(&vm_with(Box::new(InterpreterBackend), 2, false, &prog));
    let bg_vm = vm_with(Box::new(InterpreterBackend), 2, true, &prog);
    let on = run_to_hlt(&bg_vm);
    assert_eq!(on, off, "bg flag is a no-op on the interpreter backend");
    assert_eq!(
        bg_vm.cache.tier_bg_published(),
        0,
        "interp never publishes bg"
    );
}

// ---- Races between a background compile and invalidation ----

/// An SMC write to the hot block's page while its compile is pending. `handle_smc`
/// (which runs before the drain in the dispatch loop) drops the block and bumps the
/// epoch, so the drain's `upgrade` rejects the now-stale compile; the block re-lifts to
/// the new bytes, re-heats, and re-tiers cleanly.
#[test]
fn smc_while_pending_rejects_then_reheats() {
    let jit = JitBackend::new();
    let handle = jit.tier_up_handle();
    let vm = vm_with(Box::new(jit), 3, true, &single_block_val(42));
    let mut cpu = vm.new_vcpu();

    heat_to_submit(&vm, &mut cpu, 42);
    handle.wait_idle(); // compile now sits in `done`, undrained
    assert_eq!(vm.cache.tier_bg_published(), 0);

    // Overwrite the block (marks its page dirty; invalidation lands next dispatch).
    vm.write_bytes(CODE, &single_block_val(7)).unwrap();

    assert_eq!(one_pass(&vm, &mut cpu), 7); // handle_smc → reject stale → re-lift v2
    assert_eq!(vm.cache.tier_bg_rejected(), 1, "stale compile rejected");
    assert_eq!(vm.cache.tier_bg_published(), 0);

    // New block re-tiers and publishes.
    heat_to_submit(&vm, &mut cpu, 7);
    handle.wait_idle();
    assert_eq!(one_pass(&vm, &mut cpu), 7);
    assert_eq!(vm.cache.tier_bg_published(), 1, "new block published");
    assert_eq!(vm.cache.tier_pending_len(), 0, "no stuck in-flight marker");
}

/// Mapping a Trap region mid-flight flushes the whole cache and bumps the epoch
/// (a JIT bakes the mmio window as a constant, so every prior compile is stale). The
/// pending compile is rejected on drain; the block re-lifts with the new window.
#[test]
fn trap_map_midflight_rejects_stale() {
    let jit = JitBackend::new();
    let handle = jit.tier_up_handle();
    let mut vm = vm_with_flat(Box::new(jit), 3, true, &single_block_val(42), 0x4000);
    let mut cpu = vm.new_vcpu();

    heat_to_submit(&vm, &mut cpu, 42);
    handle.wait_idle();

    // A Trap map: full flush + epoch bump (vm.rs map()).
    vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();

    assert_eq!(one_pass(&vm, &mut cpu), 42);
    assert_eq!(vm.cache.tier_bg_rejected(), 1, "stale compile rejected");

    heat_to_submit(&vm, &mut cpu, 42);
    handle.wait_idle();
    assert_eq!(one_pass(&vm, &mut cpu), 42);
    assert_eq!(vm.cache.tier_bg_published(), 1);
    assert_eq!(vm.cache.tier_pending_len(), 0);
}

/// An unrelated block can be invalidated without discarding this request's work.
#[test]
fn unrelated_invalidation_preserves_pending_completion() {
    const DECOY: u64 = 0x3000;
    let jit = JitBackend::new();
    let handle = jit.tier_up_handle();
    let pause = handle.pause_compiler();
    let mut vm = vm_with_flat(Box::new(jit), 3, true, &single_block_val(42), 0x4000);
    vm.map(DECOY, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
    vm.write_bytes(DECOY, &[0xf4]).unwrap();
    let mut dcpu = vm.new_vcpu();
    dcpu.set_reg(Reg::Rip, DECOY);
    assert!(matches!(dcpu.run(&vm, Some(1_000)), Exit::Hlt));
    let mut cpu = vm.new_vcpu();
    heat_to_submit(&vm, &mut cpu, 42);
    vm.write_bytes(DECOY, &[0xf4]).unwrap();
    assert_eq!(one_pass(&vm, &mut cpu), 42);
    assert_eq!(vm.memory_stats().cache.tier_bg_submitted, 1);
    drop(pause);
    handle.wait_idle();
    assert_eq!(one_pass(&vm, &mut cpu), 42);
    let stats = vm.memory_stats().cache;
    assert_eq!(stats.tier_bg_completed, 1);
    assert_eq!(stats.tier_bg_published, 1);
    assert_eq!(stats.tier_bg_rejected, 0);
    assert_eq!(vm.cache.tier_pending_len(), 0);
}

/// Even overlapping SMC cannot enqueue a replacement until the old request drains.
#[test]
fn overlapping_write_cancels_without_duplicate_submission() {
    let jit = JitBackend::new();
    let handle = jit.tier_up_handle();
    let pause = handle.pause_compiler();
    let vm = vm_with(Box::new(jit), 3, true, &single_block_val(42));
    let mut cpu = vm.new_vcpu();
    heat_to_submit(&vm, &mut cpu, 42);
    vm.write_bytes(CODE, &single_block_val(7)).unwrap();
    for _ in 0..10 {
        assert_eq!(one_pass(&vm, &mut cpu), 7);
    }
    assert_eq!(vm.cache.tier_pending_len(), 1);
    assert_eq!(vm.memory_stats().cache.tier_bg_submitted, 1);
    assert!(vm.memory_stats().cache.tier_bg_duplicates_skipped > 0);
    drop(pause);
    handle.wait_idle();
    assert_eq!(one_pass(&vm, &mut cpu), 7);
    assert_eq!(vm.memory_stats().cache.tier_bg_rejected_generation, 1);
    handle.wait_idle();
    assert_eq!(one_pass(&vm, &mut cpu), 7);
    let stats = vm.memory_stats().cache;
    assert_eq!(stats.tier_bg_submitted, 2);
    assert_eq!(stats.tier_bg_completed, 2);
    assert_eq!(stats.tier_bg_published, 1);
    assert_eq!(stats.tier_bg_superseded, 0);
    assert_eq!(vm.cache.tier_pending_len(), 0);
}

// Shutdown consumes exclusive VM ownership. Stopping vCPU host threads, including
// waking one blocked in the embedder, is the embedder's part of the fork boundary.
const SHUTDOWN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

#[test]
fn shutdown_cancels_queued_compiles_and_expires_handles() {
    let jit = JitBackend::new();
    let handle = jit.tier_up_handle();
    let pause = handle.pause_compiler();
    let vm = vm_with(Box::new(jit), 3, true, &single_block());
    let mut cpu = vm.new_vcpu();
    heat_to_submit(&vm, &mut cpu, 42);
    drop(cpu);
    assert_eq!(
        vm.shutdown().unwrap(),
        x86jit_core::ShutdownReport {
            workers_joined: 1,
            jobs_cancelled: 1,
            outstanding_handles: 0,
        }
    );
    handle.wait_idle();
    drop(pause);
}

#[test]
fn shutdown_interpreter_has_no_workers() {
    let vm = vm_with(Box::new(InterpreterBackend), 3, true, &single_block());
    assert_eq!(run_to_hlt(&vm), 42);
    assert_eq!(
        vm.shutdown().unwrap(),
        x86jit_core::ShutdownReport::default()
    );
}

#[test]
fn worker_spawner_configures_signal_mask() {
    use std::sync::mpsc;
    let (tx, rx) = mpsc::channel();
    let mut jit = JitBackend::new();
    jit.set_worker_spawner(move |task| {
        let tx = tx.clone();
        std::thread::Builder::new()
            .name("x86jit-tier".into())
            .spawn(move || {
                // SAFETY: sigset_t is initialized by sigfillset before being passed
                // to pthread_sigmask; this changes only the compiler thread's mask.
                let blocked = unsafe {
                    let mut all = std::mem::zeroed();
                    assert_eq!(libc::sigfillset(&mut all), 0);
                    assert_eq!(
                        libc::pthread_sigmask(libc::SIG_SETMASK, &all, std::ptr::null_mut()),
                        0
                    );
                    let mut actual = std::mem::zeroed();
                    assert_eq!(
                        libc::pthread_sigmask(libc::SIG_SETMASK, std::ptr::null(), &mut actual),
                        0
                    );
                    [libc::SIGUSR1, libc::SIGTERM, libc::SIGINT, libc::SIGALRM]
                        .iter()
                        .all(|&signal| libc::sigismember(&actual, signal) == 1)
                };
                tx.send(blocked).unwrap();
                task();
            })
    });
    let handle = jit.tier_up_handle();
    let vm = vm_with(Box::new(jit), 3, true, &single_block());
    let mut cpu = vm.new_vcpu();
    heat_to_submit(&vm, &mut cpu, 42);
    assert!(rx.recv_timeout(SHUTDOWN_TIMEOUT).unwrap());
    handle.wait_idle();
    assert_eq!(one_pass(&vm, &mut cpu), 42);
    assert_eq!(vm.cache.tier_bg_published(), 1);
    drop(cpu);
    assert_eq!(vm.shutdown().unwrap().workers_joined, 1);
}

#[test]
fn worker_spawner_failure_stays_interpreted() {
    let mut jit = JitBackend::new();
    jit.set_worker_spawner(|_| Err(std::io::ErrorKind::WouldBlock.into()));
    let vm = vm_with(Box::new(jit), 3, true, &single_block());
    let mut cpu = vm.new_vcpu();
    for _ in 0..3 {
        assert_eq!(one_pass(&vm, &mut cpu), 42);
    }
    cpu.set_reg(Reg::Rip, CODE);
    assert!(matches!(
        cpu.run(&vm, Some(100)),
        Exit::CodeSpaceExhausted { .. }
    ));
    assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
    assert_eq!(cpu.reg(Reg::Rax), 42);
    assert_eq!(vm.cache.tier_pending_len(), 0);
    assert_eq!(vm.cache.tier_bg_published(), 0);
    assert!(matches!(
        vm.cache.get(x86jit_core::BlockKey::new(
            CODE,
            x86jit_core::CpuMode::Long64
        )),
        Some(x86jit_core::CachedBlock::Interpreted(_))
    ));
    drop(cpu);
    assert_eq!(
        vm.shutdown().unwrap(),
        x86jit_core::ShutdownReport::default()
    );
}

#[cfg(target_os = "linux")]
fn process_thread_count() -> usize {
    std::fs::read_dir("/proc/self/task").unwrap().count()
}

#[cfg(target_os = "macos")]
#[allow(deprecated)]
fn process_thread_count() -> usize {
    unsafe extern "C" {
        fn mach_port_deallocate(
            task: libc::mach_port_t,
            name: libc::mach_port_t,
        ) -> libc::kern_return_t;
    }
    // SAFETY: task_threads initializes the allocated array and count on success.
    // Release each returned send right and then the out-of-line array exactly once.
    unsafe {
        let task = libc::mach_task_self();
        let mut threads = std::ptr::null_mut();
        let mut count = 0;
        assert_eq!(libc::task_threads(task, &mut threads, &mut count), 0);
        for &port in std::slice::from_raw_parts(threads, count as usize) {
            assert_eq!(mach_port_deallocate(task, port), 0);
        }
        assert_eq!(
            libc::vm_deallocate(
                task,
                threads as libc::vm_address_t,
                count as usize * std::mem::size_of::<libc::thread_act_t>()
            ),
            0
        );
        count as usize
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn wait_for_thread_count(expected: usize) {
    let deadline = std::time::Instant::now() + SHUTDOWN_TIMEOUT;
    loop {
        let actual = process_thread_count();
        if actual == expected {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "thread count stayed at {actual}, expected {expected}"
        );
        std::thread::yield_now();
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn shutdown_fork_boundary(jit_enabled: bool) {
    use std::sync::atomic::Ordering;
    use std::sync::{mpsc, Arc};
    use std::time::{Duration, Instant};
    // A separate test process gives thread counts a stable baseline even under
    // cargo test's parallel harness, and excludes unrelated engines at fork.
    let name = if jit_enabled {
        "shutdown_fork_boundary_jit"
    } else {
        "shutdown_fork_boundary_interpreter"
    };
    if std::env::var_os("X86JIT_SHUTDOWN_FIXTURE").is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name, "--test-threads=1", "--nocapture"])
            .env("X86JIT_SHUTDOWN_FIXTURE", "1")
            .status()
            .unwrap();
        assert!(status.success(), "isolated shutdown fixture failed");
        return;
    }
    let baseline = process_thread_count();
    let jit = JitBackend::new();
    jit.set_lock_telemetry(true);
    let handle = jit.tier_up_handle();
    let pause = handle.pause_compiler();
    let mut vm = vm_with(
        if jit_enabled {
            Box::new(jit)
        } else {
            Box::new(InterpreterBackend)
        },
        3,
        true,
        &single_block(),
    );
    let mut warmup = vm.new_vcpu();
    heat_to_submit(&vm, &mut warmup, 42);
    drop(warmup);
    vm.set_tier_up_after(None);
    // Use the existing two-edge loop fixture from the deferral tests.
    vm.write_bytes(CODE + 0x100, &[0xeb, 0x00, 0xeb, 0xfc])
        .unwrap();
    let held = vm.test_support_hold_compile(Duration::from_millis(250));
    let vm = Arc::new(vm);
    let mut spinning = vm.new_vcpu();
    spinning.set_reg(Reg::Rip, CODE + 0x100);
    let request = spinning.exit_request();
    let blocked = vm.new_vcpu();
    let blocked_vm = vm.clone();
    let (wake_tx, wake_rx) = mpsc::channel();
    let (blocked_tx, blocked_rx) = mpsc::channel();
    let blocked_thread = std::thread::spawn(move || {
        blocked_tx.send(()).unwrap();
        wake_rx.recv_timeout(SHUTDOWN_TIMEOUT).unwrap();
        drop(blocked);
        drop(blocked_vm);
    });
    blocked_rx.recv_timeout(SHUTDOWN_TIMEOUT).unwrap();
    let running_vm = vm.clone();
    let (done_tx, done_rx) = mpsc::channel();
    let running_thread = std::thread::spawn(move || {
        let exit = spinning.run(&running_vm, None);
        done_tx.send((exit, spinning)).unwrap();
    });
    let deadline = Instant::now() + SHUTDOWN_TIMEOUT;
    while !held.load(Ordering::Acquire) {
        assert!(Instant::now() < deadline, "fresh compile was never entered");
        std::thread::yield_now();
    }
    assert_eq!(
        process_thread_count(),
        baseline + 2 + usize::from(jit_enabled)
    );
    request.request_exit();
    let (exit, mut spinning) = done_rx.recv_timeout(SHUTDOWN_TIMEOUT).unwrap();
    assert!(matches!(exit, Exit::Interrupted));
    let evidence = spinning.interruption_evidence().unwrap();
    assert!(evidence.contains(x86jit_core::CriticalSectionKind::Compile));
    assert!(evidence.deferred());
    running_thread.join().unwrap();
    // Resume the fresh translation and prove the vCPU really executes its loop
    // while the other vCPU remains blocked outside the engine.
    let running_vm = vm.clone();
    let (ready_tx, ready_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let running_thread = std::thread::spawn(move || {
        assert!(matches!(
            spinning.run(&running_vm, Some(64)),
            Exit::BudgetExhausted
        ));
        if jit_enabled {
            assert!(
                running_vm.cache.chained() > 0,
                "fresh compiled loop must execute"
            );
        }
        ready_tx.send(()).unwrap();
        let exit = spinning.run(&running_vm, None);
        done_tx.send((exit, spinning)).unwrap();
    });
    ready_rx.recv_timeout(SHUTDOWN_TIMEOUT).unwrap();
    request.request_exit();
    wake_tx.send(()).unwrap();
    let (exit, spinning) = done_rx.recv_timeout(SHUTDOWN_TIMEOUT).unwrap();
    assert!(matches!(exit, Exit::Interrupted));
    drop(spinning);
    running_thread.join().unwrap();
    blocked_thread.join().unwrap();
    let vm = Arc::try_unwrap(vm)
        .ok()
        .expect("all vCPU owners must be gone");
    assert_eq!(
        vm.shutdown().unwrap(),
        x86jit_core::ShutdownReport {
            workers_joined: usize::from(jit_enabled),
            jobs_cancelled: usize::from(jit_enabled),
            outstanding_handles: 0,
        }
    );
    handle.wait_idle();
    drop(pause);
    wait_for_thread_count(baseline);
    // Rebuilding in each process cannot reuse a compiler lock, queue or JoinHandle.
    // All engine workers have been joined before fork. The alarm bounds a stale-lock failure.
    // SAFETY: no engine code or compiler is active; the child terminates with _exit.
    let child = unsafe { libc::fork() };
    assert!(child >= 0);
    if child == 0 {
        // SAFETY: alarm bounds the child and _exit avoids the copied test harness.
        unsafe { libc::alarm(10) };
        let result = std::panic::catch_unwind(|| {
            assert_eq!(process_thread_count(), 1);
            rebuild_after_shutdown(jit_enabled);
            wait_for_thread_count(1);
        });
        // SAFETY: exit only this child, without running copied parent destructors.
        unsafe { libc::_exit(i32::from(result.is_err())) };
    }
    rebuild_after_shutdown(jit_enabled);
    let mut status = 0;
    // SAFETY: child is our live child PID and status is writable.
    assert_eq!(unsafe { libc::waitpid(child, &mut status, 0) }, child);
    assert!(libc::WIFEXITED(status));
    assert_eq!(libc::WEXITSTATUS(status), 0);
    assert!(
        evidence.deferred(),
        "owned deferral evidence survives teardown and rebuild"
    );
    wait_for_thread_count(baseline);
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn rebuild_after_shutdown(jit_enabled: bool) {
    let (backend, handle): (Box<dyn Backend>, _) = if jit_enabled {
        let jit = JitBackend::new();
        let handle = jit.tier_up_handle();
        (Box::new(jit), Some(handle))
    } else {
        (Box::new(InterpreterBackend), None)
    };
    let fresh = vm_with(backend, 1, true, &accumulate_loop(1000));
    assert_eq!(run_to_hlt(&fresh), 500500);
    if let Some(handle) = handle {
        handle.wait_idle();
        assert_eq!(run_to_hlt(&fresh), 500500);
        assert!(
            fresh.cache.tier_bg_published() > 0,
            "rebuilt worker must compile and publish"
        );
    }
    assert_eq!(
        fresh.shutdown().unwrap().workers_joined,
        usize::from(jit_enabled)
    );
}

#[test]
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn shutdown_fork_boundary_jit() {
    shutdown_fork_boundary(true);
}

#[test]
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn shutdown_fork_boundary_interpreter() {
    shutdown_fork_boundary(false);
}

#[test]
fn shutdown_and_drop_release_backend_before_cache_before_memory() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    struct ObserverLifetime {
        dropped: Arc<AtomicBool>,
        cached_ir: Arc<std::sync::Mutex<std::sync::Weak<x86jit_core::IrBlock>>>,
    }
    impl Drop for ObserverLifetime {
        fn drop(&mut self) {
            assert!(
                self.cached_ir.lock().unwrap().upgrade().is_some(),
                "translation cache released before backend"
            );
            self.dropped.store(true, Ordering::Release);
        }
    }
    for explicit in [false, true] {
        let backend_dropped = Arc::new(AtomicBool::new(false));
        let memory_dropped = Arc::new(AtomicBool::new(false));
        let cached_ir = Arc::new(std::sync::Mutex::new(std::sync::Weak::new()));
        let observer = ObserverLifetime {
            dropped: backend_dropped.clone(),
            cached_ir: cached_ir.clone(),
        };
        let jit = JitBackend::new();
        jit.set_compiled_code_observer(move |_| {
            std::hint::black_box(&observer);
        });
        let handle = jit.tier_up_handle();
        let pause = handle.pause_compiler();
        let bytes = vec![0; 0x2000].into_boxed_slice();
        let ptr = Box::into_raw(bytes) as *mut u8;
        let dropped = memory_dropped.clone();
        let memory_cached_ir = cached_ir.clone();
        let ram = x86jit_core::HostRam {
            ptr,
            len: 0x2000,
            guest_base: 0,
            dtor: Box::new(move |ptr, len| {
                // SAFETY: this is the sole owner of the Box allocation above,
                // returned once by Memory with its original pointer and length.
                unsafe { drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len))) };
                assert!(
                    backend_dropped.load(Ordering::Acquire),
                    "memory released before the backend"
                );
                assert!(
                    memory_cached_ir.lock().unwrap().upgrade().is_none(),
                    "memory released before translation cache"
                );
                dropped.store(true, Ordering::Release);
            }),
            protect: None,
        };
        let mut vm =
            Vm::with_backend_host_ram(VmConfig::reserved(0x2000), Box::new(jit), ram).unwrap();
        vm.set_tier_up_after(Some(3));
        vm.set_tier_up_background(true);
        vm.map(CODE, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
        vm.write_bytes(CODE, &single_block()).unwrap();
        let mut cpu = vm.new_vcpu();
        heat_to_submit(&vm, &mut cpu, 42);
        {
            let Some(x86jit_core::CachedBlock::Interpreted(ir)) = vm.cache.get(
                x86jit_core::BlockKey::new(CODE, x86jit_core::CpuMode::Long64),
            ) else {
                panic!("paused tier-up must remain interpreted")
            };
            *cached_ir.lock().unwrap() = Arc::downgrade(&ir);
        }
        drop(cpu);
        if explicit {
            assert_eq!(vm.shutdown().unwrap().workers_joined, 1);
        } else {
            drop(vm);
        }
        assert!(memory_dropped.load(Ordering::Acquire));
        handle.wait_idle();
        drop(pause);
    }
}

#[test]
fn shutdown_names_live_vcpus_on_both_backends() {
    for backend in [
        Box::new(InterpreterBackend) as Box<dyn Backend>,
        Box::new(JitBackend::new()),
    ] {
        let vm = vm_with(backend, 3, true, &single_block());
        let mut cpu = vm.new_vcpu();
        assert_eq!(one_pass(&vm, &mut cpu), 42);
        let error = vm.shutdown().unwrap_err();
        assert_eq!(
            error,
            x86jit_core::ShutdownError::LiveResources {
                resource: "vCPUs",
                report: x86jit_core::ShutdownReport {
                    outstanding_handles: 1,
                    ..Default::default()
                },
            }
        );
        assert!(error.to_string().contains("vCPUs"));
        drop(cpu);
    }
}

#[test]
fn shutdown_wait_idle_race_finishes_compiler_destruction() {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    struct Marker {
        dropped: Arc<AtomicBool>,
        owner: std::thread::ThreadId,
    }
    impl Drop for Marker {
        fn drop(&mut self) {
            if std::thread::current().id() != self.owner {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            self.dropped.store(true, Ordering::Release);
        }
    }
    let mut clean_but_retained = 0;
    let mut live_errors = 0;
    for _ in 0..1000 {
        let jit = JitBackend::new();
        let dropped = Arc::new(AtomicBool::new(false));
        let marker = Marker {
            dropped: dropped.clone(),
            owner: std::thread::current().id(),
        };
        jit.set_compiled_code_observer(move |_| {
            std::hint::black_box(&marker);
        });
        let handle = jit.tier_up_handle();
        let mut vm = vm_with(Box::new(jit), 3, false, &single_block());
        vm.set_tier_up_after(None);
        assert_eq!(run_to_hlt(&vm), 42);
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let started = Arc::new(AtomicBool::new(false));
        let worker_started = started.clone();
        let worker = std::thread::spawn(move || {
            worker_started.store(true, Ordering::Release);
            while !worker_stop.load(Ordering::Acquire) {
                handle.wait_idle();
            }
        });
        while !started.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        let result = vm.shutdown();
        if result.is_ok() && !dropped.load(Ordering::Acquire) {
            clean_but_retained += 1;
        }
        if result.is_err() {
            live_errors += 1;
        }
        stop.store(true, Ordering::Release);
        worker.join().unwrap();
        assert!(dropped.load(Ordering::Acquire));
    }
    println!("MEASURED public wait_idle with compiled module: clean shutdown retaining Shared/destructor={clean_but_retained}; transient live-resource errors={live_errors}; total=1000");
    assert_eq!(
        live_errors, 0,
        "diagnostic operations must not retain compiler resources"
    );
    assert_eq!(clean_but_retained, 0, "Ok(0 outstanding handles) returned before Shared destruction completed on diagnostic thread");
}

#[test]
fn victim_free_drain_rejects_background_region_with_uncached_successor() {
    use std::sync::Arc;
    use x86jit_core::{BlockKey, CachedBlock, CpuMode, TierUpUnit};
    const SUCCESSOR: u64 = CODE + 0x100;
    // Observe that this request really contains the uncached successor's old IR.
    struct ObservedRegionBackend {
        jit: JitBackend,
        submitted: Arc<std::sync::atomic::AtomicBool>,
    }
    impl Backend for ObservedRegionBackend {
        fn materialize(
            &self,
            ir: &x86jit_core::IrBlock,
            c: x86jit_core::MemConsistency,
            m: Option<(u64, u64)>,
            b: u64,
        ) -> CachedBlock {
            self.jit.materialize(ir, c, m, b)
        }
        fn region_caps(&self) -> Option<RegionCaps> {
            self.jit.region_caps()
        }
        fn tier_up_async(&self, req: x86jit_core::TierUpRequest) -> x86jit_core::TierUpSubmit {
            if req.pc == CODE {
                assert!(
                    matches!(&req.unit, TierUpUnit::Region(r) if r.blocks.iter().any(|b| b.guest_start == SUCCESSOR))
                );
                self.submitted
                    .store(true, std::sync::atomic::Ordering::Release);
            }
            self.jit.tier_up_async(req)
        }
        fn tier_up_finished(&self) -> Vec<x86jit_core::TierUpFinished> {
            self.jit.tier_up_finished()
        }
        fn invalidate_links(&self) {
            self.jit.invalidate_links();
        }
    }
    let jit = JitBackend::with_superblocks(RegionCaps {
        max_blocks: 8,
        max_icount: 128,
    });
    let handle = jit.tier_up_handle();
    let pause = handle.pause_compiler();
    let submitted = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut vm = Vm::with_backend(
        VmConfig::flat(0x5000),
        Box::new(ObservedRegionBackend {
            jit,
            submitted: submitted.clone(),
        }),
    );
    vm.set_tier_up_after(Some(1));
    vm.set_tier_up_background(true);
    vm.map(0, 0x5000, Prot::RWX, RegionKind::Ram).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    a.jmp(SUCCESSOR).unwrap();
    vm.write_bytes(CODE, &a.assemble(CODE).unwrap()).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    a.mov(eax, 1).unwrap();
    a.dec(ecx).unwrap();
    a.jnz(CODE).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(SUCCESSOR, &a.assemble(SUCCESSOR).unwrap())
        .unwrap();
    vm.write_bytes(0x4000, &[0xf4]).unwrap();
    let ir = x86jit_core::lift::lift_block(
        &vm.mem,
        x86jit_core::lift::FetchAddr::flat(CODE),
        CpuMode::Long64,
    )
    .unwrap();
    let span = (CODE, ir.guest_len);
    vm.cache.insert(
        BlockKey::new(CODE, CpuMode::Long64),
        CachedBlock::Interpreted(Arc::new(ir)),
        vec![span],
        |_| vm.mem.mark_code(span.0, span.1),
    );
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rcx, 2);
    assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
    assert!(submitted.load(std::sync::atomic::Ordering::Acquire));
    assert_eq!(vm.cache.tier_pending_len(), 1);
    assert!(vm
        .cache
        .get(BlockKey::new(SUCCESSOR, CpuMode::Long64))
        .is_none());
    let epoch = vm.cache.epoch();
    vm.write_bytes(SUCCESSOR + 1, &[2]).unwrap();
    let mut drain = vm.new_vcpu();
    drain.set_reg(Reg::Rip, 0x4000);
    assert!(matches!(drain.run(&vm, Some(10)), Exit::Hlt));
    assert!(!vm.mem.has_dirty_code());
    assert_eq!(vm.cache.epoch(), epoch, "the cached entry survives");
    drop(pause);
    handle.wait_idle();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rcx, 2);
    assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
    assert_eq!(
        cpu.reg(Reg::Rax),
        2,
        "stale successor IR must not be published"
    );
    assert!(vm.cache.tier_bg_rejected() > 0);
    handle.wait_idle();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rcx, 2);
    assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
    assert_eq!(cpu.reg(Reg::Rax), 2);
    assert!(
        vm.cache.tier_bg_published() > 0,
        "fresh region can still publish"
    );
    assert!(matches!(
        vm.cache.get(BlockKey::new(CODE, CpuMode::Long64)),
        Some(CachedBlock::Compiled { .. })
    ));
    assert_eq!(vm.cache.tier_pending_len(), 0);
}

fn disjoint_guest_store_preserves_completion(different_page: bool) {
    let jit = JitBackend::new();
    let handle = jit.tier_up_handle();
    let pause = handle.pause_compiler();
    let mut vm = Vm::with_backend(VmConfig::flat(0x5000), Box::new(jit));
    vm.set_tier_up_after(Some(32));
    vm.set_tier_up_background(true);
    vm.map(0, 0x5000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(CODE, &single_block()).unwrap();
    vm.write_bytes(0x3000, &[0xf4]).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    a.mov(byte_ptr(rdi), 42).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(0x4000, &a.assemble(0x4000).unwrap())
        .unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x3000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    for _ in 0..33 {
        assert_eq!(one_pass(&vm, &mut cpu), 42);
    }
    assert_eq!(vm.memory_stats().cache.tier_bg_submitted, 1);
    let before = vm.memory_stats().cache;
    cpu.set_reg(Reg::Rip, 0x4000);
    cpu.set_reg(Reg::Rdi, if different_page { 0x3800 } else { 0x1800 });
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(one_pass(&vm, &mut cpu), 42);
    let after = vm.memory_stats().cache;
    assert_eq!(after.invalidation_calls - before.invalidation_calls, 1);
    assert_eq!(
        after.invalidation_blocks_dropped,
        before.invalidation_blocks_dropped
    );
    assert_eq!(after.tier_bg_submitted, 1);
    assert!(after.tier_bg_duplicates_skipped > 0);
    drop(pause);
    handle.wait_idle();
    assert_eq!(one_pass(&vm, &mut cpu), 42);
    let stats = vm.memory_stats().cache;
    assert_eq!(stats.tier_bg_completed, 1);
    assert_eq!(stats.tier_bg_published, 1);
    assert_eq!(stats.tier_bg_rejected, 0);
    assert_eq!(vm.cache.tier_pending_len(), 0);
}

#[test]
fn disjoint_guest_store_on_same_code_page_preserves_completion() {
    disjoint_guest_store_preserves_completion(false);
}

#[test]
fn disjoint_guest_store_on_different_code_page_preserves_completion() {
    disjoint_guest_store_preserves_completion(true);
}

/// Deliberately bypass the shared lifter's descriptor cap to exercise the
/// backend's defensive refusal and the dispatcher's retry policy.
fn oversized_background_unit(region: bool) {
    use std::sync::Arc;
    use x86jit_core::{BlockKey, CachedBlock, CpuMode, IrBlock, IrRegion, TierUpUnit};
    struct OversizedBackend {
        jit: JitBackend,
        blocks: Vec<IrBlock>,
        region: bool,
    }
    impl Backend for OversizedBackend {
        fn materialize(
            &self,
            ir: &IrBlock,
            c: x86jit_core::MemConsistency,
            m: Option<(u64, u64)>,
            b: u64,
        ) -> CachedBlock {
            self.jit.materialize(ir, c, m, b)
        }
        fn tier_up_async(&self, mut req: x86jit_core::TierUpRequest) -> x86jit_core::TierUpSubmit {
            if self.region {
                req.unit = TierUpUnit::Region(Arc::new(IrRegion {
                    entry: CODE,
                    blocks: self.blocks.clone(),
                    has_loop: false,
                }));
                req.spans = self
                    .blocks
                    .iter()
                    .map(|b| (b.guest_start, b.guest_len))
                    .collect();
            }
            self.jit.tier_up_async(req)
        }
        fn tier_up_finished(&self) -> Vec<x86jit_core::TierUpFinished> {
            self.jit.tier_up_finished()
        }
    }
    let mut a = CodeAssembler::new(64).unwrap();
    for _ in 0..160 {
        a.mov(rax, qword_ptr(rdi)).unwrap();
    }
    let successor = CODE + 160 * 3 + 2;
    a.jmp(successor).unwrap();
    for _ in 0..160 {
        a.mov(rax, qword_ptr(rdi)).unwrap();
    }
    a.hlt().unwrap();
    let code = a.assemble(CODE).unwrap();
    let jit = JitBackend::new();
    let events = Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed = events.clone();
    jit.set_compile_failure_observer(move |event| observed.lock().unwrap().push(event));
    let handle = jit.tier_up_handle();
    let mut mem = x86jit_core::Memory::new(x86jit_core::MemoryModel::Flat { size: 0x4000 });
    mem.map(CODE, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
    mem.write_bytes(CODE, &code).unwrap();
    let blocks = if region {
        [CODE, successor]
            .into_iter()
            .map(|pc| {
                x86jit_core::lift::lift_block(
                    &mem,
                    x86jit_core::lift::FetchAddr::flat(pc),
                    CpuMode::Long64,
                )
                .unwrap()
            })
            .collect()
    } else {
        // A straight-line block with 320 distinct instruction owners.
        let mut code = [0x48, 0x8b, 0x07].repeat(320);
        code.push(0xf4);
        mem.write_bytes(CODE, &code).unwrap();
        let mut block = x86jit_core::lift::lift_one(
            &mem,
            x86jit_core::lift::FetchAddr::flat(CODE),
            CpuMode::Long64,
        )
        .unwrap();
        for i in 1..=320 {
            let next = x86jit_core::lift::lift_one(
                &mem,
                x86jit_core::lift::FetchAddr::flat(CODE + i * 3),
                CpuMode::Long64,
            )
            .unwrap();
            block.ops.extend(next.ops);
            block.guest_len += next.guest_len;
            block.icount += 1;
            block.temp_count = block.temp_count.max(next.temp_count);
        }
        vec![block]
    };
    let first = Arc::new(blocks[0].clone());
    let mut vm = vm_with_flat(
        Box::new(OversizedBackend {
            jit,
            blocks,
            region,
        }),
        1,
        true,
        &code,
        0x4000,
    );
    if !region {
        let mut code = [0x48, 0x8b, 0x07].repeat(320);
        code.push(0xf4);
        vm.write_bytes(CODE, &code).unwrap();
    }
    vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
    vm.write_bytes(0x3000, &42u64.to_le_bytes()).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rdi, 0x3000);
    let pass = |vm: &Vm, cpu: &mut x86jit_core::Vcpu| {
        cpu.set_reg(Reg::Rip, CODE);
        let exit = cpu.run(vm, Some(if region { 1 } else { 1_000_000 }));
        assert!(if region {
            matches!(exit, Exit::BudgetExhausted)
        } else {
            matches!(exit, Exit::Hlt)
        });
        cpu.reg(Reg::Rax)
    };
    // Drain setup writes before injecting synthetic cached IR.
    assert_eq!(pass(&vm, &mut cpu), 42);
    vm.cache.insert(
        BlockKey::new(CODE, CpuMode::Long64),
        CachedBlock::Interpreted(first.clone()),
        vec![(CODE, first.guest_len)],
        |_| vm.mem.mark_code(CODE, first.guest_len),
    );
    for _ in 0..3 {
        assert_eq!(pass(&vm, &mut cpu), 42);
        handle.wait_idle();
    }
    let stats = vm.memory_stats().cache;
    println!(
        "MEASURED oversized region={region}: submitted={}, failed={}, published={}",
        stats.tier_bg_submitted, stats.tier_bg_rejected_compile_failed, stats.tier_bg_published
    );
    assert_eq!(stats.tier_bg_rejected_compile_failed, 1);
    assert_eq!(stats.tier_bg_failed_descriptor_limit, 1);
    assert_eq!(stats.tier_bg_failed_code_space, 0);
    assert_eq!(stats.tier_bg_failed_other_backend, 0);
    assert_eq!(stats.tier_bg_interpreter_only, 1);
    assert!(stats.tier_bg_interpreter_only_skipped > 0);
    assert_eq!(stats.tier_bg_submitted, 1);
    assert_eq!(stats.tier_bg_published, 0);
    assert_eq!(vm.cache.tier_pending_len(), 0);
    let events = events.lock().unwrap();
    assert_eq!(events.len(), 1);
    let event = events[0];
    assert_eq!(event.guest_rip, CODE);
    assert_eq!(
        event.reason,
        x86jit_core::CompileFailureReason::DescriptorLimit
    );
    assert_eq!(
        event.kind,
        if region {
            x86jit_cranelift::CompiledCodeKind::Region
        } else {
            x86jit_cranelift::CompiledCodeKind::Block
        }
    );
    assert_eq!(event.blocks, if region { 2 } else { 1 });
    assert_eq!(event.instructions, if region { 322 } else { 321 });
    assert_eq!(event.guest_bytes, if region { 963 } else { 961 });
    assert!(event.ir_ops > 640);
    drop(events);
    vm.set_tier_up_background(false);
    for _ in 0..3 {
        assert_eq!(pass(&vm, &mut cpu), 42);
    }
    assert_eq!(vm.memory_stats().cache.tier_bg_submitted, 1);
    assert_eq!(vm.memory_stats().backend.compiled_units_total, 0);
    vm.set_tier_up_background(true);
    // A byte invalidation releases suppression and the ordinary lifter supplies
    // capped blocks again. The forced-region fixture remains intentionally oversized.
    vm.write_bytes(CODE, &[0x48]).unwrap();
    assert_eq!(pass(&vm, &mut cpu), 42);
    assert_eq!(pass(&vm, &mut cpu), 42);
    handle.wait_idle();
    assert!(vm.memory_stats().cache.tier_bg_submitted > stats.tier_bg_submitted);
}

#[test]
fn oversized_background_block_retry_policy() {
    oversized_background_unit(false);
}

#[test]
fn oversized_background_region_retry_policy() {
    oversized_background_unit(true);
}

#[test]
fn background_320_access_program_compiles_as_bounded_blocks() {
    let mut code = [0x48, 0x8b, 0x07].repeat(320);
    code.push(0xf4);
    let jit = JitBackend::new();
    let handle = jit.tier_up_handle();
    let mut vm = vm_with_flat(Box::new(jit), 32, true, &code, 0x4000);
    vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
    vm.write_bytes(0x3000, &42u64.to_le_bytes()).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rdi, 0x3000);
    for _ in 0..33 {
        assert_eq!(one_pass(&vm, &mut cpu), 42);
    }
    handle.wait_idle();
    assert_eq!(one_pass(&vm, &mut cpu), 42);
    let stats = vm.memory_stats().cache;
    println!(
        "MEASURED normal 320-access program: submitted={}, failed={}, published={}",
        stats.tier_bg_submitted, stats.tier_bg_rejected_compile_failed, stats.tier_bg_published
    );
    assert_eq!(stats.tier_bg_submitted, 2);
    assert_eq!(stats.tier_bg_published, 2);
    assert_eq!(stats.tier_bg_rejected_compile_failed, 0);
    assert_eq!(vm.cache.tier_pending_len(), 0);
}
