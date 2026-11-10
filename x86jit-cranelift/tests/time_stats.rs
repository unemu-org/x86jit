#![cfg(feature = "jit")]

use x86jit_core::{Exit, Prot, Reg, RegionKind, Vm, VmConfig};
use x86jit_cranelift::JitBackend;

fn vm(jit: JitBackend, code: &[u8]) -> Vm {
    let mut vm = Vm::with_backend(VmConfig::flat(0x4000), Box::new(jit));
    vm.set_time_accounting(true);
    vm.map(0, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x2000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    vm.write_bytes(0x1000, code).unwrap();
    vm
}

#[test]
fn eager_compiler_compiled_chains_and_sampled_helpers() {
    // mov ecx,130; fld1; fstp st(0); dec ecx; jnz -8; hlt
    let mut vm = vm(
        JitBackend::new(),
        &[
            0xb9, 130, 0, 0, 0, 0xd9, 0xe8, 0xdd, 0xd8, 0xff, 0xc9, 0x75, 0xf8, 0xf4,
        ],
    );
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let first = vm.time_stats();
    assert!(first.foreground_compile_ns > 0);
    assert!(first.compiled_ns > 0);
    assert_eq!(first.background_compile_ns, 0);
    assert_eq!(first.helpers.calls, 260);
    assert_eq!(first.helpers.samples, 5);
    assert!(first.helpers.sample_ns > 0);
    assert_eq!(first.chain_hits, vm.cache.chained());
    assert!(first.chain_hits >= 127);
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let second = vm.time_stats();
    assert!(second.compiled_ns >= first.compiled_ns);
    assert_eq!(second.foreground_compile_ns, first.foreground_compile_ns);
    assert_eq!(second.helpers.calls, 520);
    assert!(second.chain_hits >= first.chain_hits);
    vm.set_time_accounting(false);
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let disabled = vm.time_stats();
    assert_eq!(disabled.helpers.calls, second.helpers.calls);
    assert_eq!(disabled.helpers.samples, second.helpers.samples);
    assert_eq!(disabled.compiled_ns, second.compiled_ns);
}

#[test]
fn background_worker_and_enqueue_to_cache_publication() {
    let jit = JitBackend::new();
    let handle = jit.tier_up_handle();
    let mut vm = vm(jit, &[0xf4]);
    vm.set_tier_up_after(Some(1));
    vm.set_tier_up_background(true);
    let mut cpu = vm.new_vcpu();
    for _ in 0..2 {
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    }
    handle.wait_idle();
    let ready = vm.time_stats();
    assert!(ready.background_compile_ns > 0);
    assert_eq!(ready.foreground_compile_ns, 0);
    assert_eq!(ready.tier_up_latency_count, 0);
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let published = vm.time_stats();
    assert_eq!(published.tier_up_latency_count, 1);
    assert!(published.tier_up_latency_ns > 0);
    assert!(published.background_compile_ns >= ready.background_compile_ns);
    assert!(published.compiled_ns > 0);
}

#[test]
fn compiled_exits_and_slow_replay() {
    let vm = vm(
        JitBackend::new(),
        &[0x0f, 0x05, 0xe6, 0x80, 0x8b, 0x00, 0xf4],
    );
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Syscall));
    assert!(matches!(cpu.run(&vm, None), Exit::PortIo { .. }));
    cpu.set_reg(Reg::Rax, 0x2000);
    assert!(matches!(cpu.run(&vm, None), Exit::MmioRead { .. }));
    cpu.complete_mmio_read(9);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let stats = vm.time_stats();
    assert_eq!(stats.exits.syscall.count, 1);
    assert_eq!(stats.exits.port_io.count, 1);
    assert_eq!(stats.exits.mmio.count, 1);
    assert_eq!(stats.exits.hlt.count, 1);
    assert!(stats.helpers.calls >= 2);
}

#[test]
fn indirect_branch_misses_are_counted() {
    // mov eax,0x1007; jmp rax; hlt
    let vm = vm(JitBackend::new(), &[0xb8, 7, 0x10, 0, 0, 0xff, 0xe0, 0xf4]);
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(vm.time_stats().indirect_branch_resolves, 1);
}

#[test]
fn integer_loop_chains() {
    let vm = vm(
        JitBackend::new(),
        &[0xb9, 100, 0, 0, 0, 0xff, 0xc9, 0x75, 0xfc, 0xf4],
    );
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert!(vm.time_stats().chain_hits >= 97);
}

#[test]
fn snapshot_does_not_wait_for_compiler_or_running_vcpu() {
    use std::sync::{mpsc, Arc, Mutex};
    use std::time::Duration;
    let jit = JitBackend::new();
    let (started_tx, started_rx) = mpsc::sync_channel(1);
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let release_rx = Mutex::new(release_rx);
    jit.set_compiled_code_observer(move |_| {
        started_tx.send(()).unwrap();
        release_rx
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(20))
            .unwrap();
    });
    let vm = Arc::new(vm(jit, &[0xf4]));
    let running = vm.clone();
    let run = std::thread::spawn(move || {
        let mut cpu = running.new_vcpu();
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&running, None), Exit::Hlt));
    });
    started_rx.recv_timeout(Duration::from_secs(20)).unwrap();
    let snapshot_vm = vm.clone();
    let (snapshot_tx, snapshot_rx) = mpsc::sync_channel(1);
    let snapshot_thread =
        std::thread::spawn(move || snapshot_tx.send(snapshot_vm.time_stats()).unwrap());
    let snapshot = snapshot_rx.recv_timeout(Duration::from_secs(2));
    release_tx.send(()).unwrap();
    run.join().unwrap();
    snapshot_thread.join().unwrap();
    let snapshot = snapshot.expect("snapshot must not wait for the active compiler");
    assert_eq!(snapshot.compiled_ns, 0, "active vCPU has not published yet");
    let finished = vm.time_stats();
    assert!(finished.foreground_compile_ns > 0);
    assert_eq!(finished.exits.hlt.count, 1);
}

#[test]
fn rejected_background_work_does_not_contribute_latency() {
    let jit = JitBackend::new();
    let handle = jit.tier_up_handle();
    let pause = handle.pause_compiler();
    let mut vm = vm(jit, &[0xf4]);
    vm.set_tier_up_after(Some(1));
    vm.set_tier_up_background(true);
    let mut cpu = vm.new_vcpu();
    for _ in 0..2 {
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    }
    vm.write_bytes(0x1000, &[0xf4]).unwrap();
    drop(pause);
    handle.wait_idle();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert!(vm.memory_stats().cache.tier_bg_rejected > 0);
    assert_eq!(vm.time_stats().tier_up_latency_count, 0);
    assert_eq!(vm.time_stats().tier_up_latency_ns, 0);
}

#[test]
fn return_prediction_underflow_is_an_indirect_resolve() {
    let vm = vm(JitBackend::new(), &[0xc3, 0xf4]);
    vm.write_bytes(0x1800, &0x1001u64.to_le_bytes()).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    cpu.set_reg(Reg::Rsp, 0x1800);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(vm.time_stats().indirect_branch_resolves, 1);
}
