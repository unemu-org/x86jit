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
    assert!(first.fetch_from_direct_link > 0);
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
    assert_eq!(vm.time_stats().fetch_from_ibtc_miss, 1);
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
fn region_reads_on_a_resolve_hit_charge_policy_and_lift_time() {
    use std::time::Duration;
    use x86jit_core::{FetchVerdict, RegionCaps};
    let jit = JitBackend::with_superblocks(RegionCaps {
        max_blocks: 8,
        max_icount: 64,
    });
    let handle = jit.tier_up_handle();
    // jz 0; hlt. One-block budgets leave the cold successor unfetched until
    // region formation during tier-up of the cached conditional branch.
    let mut vm = vm(jit, &[0x0f, 0x84, 0xfa, 0xef, 0xff, 0xff, 0xf4]);
    vm.write_bytes(0, &[0xf4]).unwrap();
    vm.set_tier_up_after(Some(1));
    vm.set_tier_up_background(true);
    vm.set_fetch_policy(Box::new(|_| {
        std::thread::sleep(Duration::from_millis(5));
        FetchVerdict::Allow
    }));
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
    let cold = vm.time_stats();
    assert_eq!(cold.fetch_misses, 1);
    assert!(cold.fetch_miss_approval_ns >= 5_000_000);
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
    handle.wait_idle();
    let hot = vm.time_stats();
    assert_eq!(hot.fetch_misses, 1);
    assert_eq!(hot.fetch_hits.calls, 1);
    assert_eq!(hot.fetch_miss_approval_ns, cold.fetch_miss_approval_ns);
    assert_eq!(hot.fetch_miss_lift_ns, cold.fetch_miss_lift_ns);
    assert!(hot.fetch_approval_ns >= cold.fetch_approval_ns + 5_000_000);
    assert!(hot.fetch_lift_ns > cold.fetch_lift_ns);
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
    assert_eq!(vm.time_stats().fetch_from_return_miss, 1);
}

fn shared_partitions(s: x86jit_core::TimeStats) -> (u64, u64) {
    (
        s.fetch_from_entry
            + s.fetch_from_budget_resume
            + s.fetch_from_exit_request_resume
            + s.fetch_from_boundary
            + s.fetch_from_return_miss
            + s.fetch_from_direct_link
            + s.fetch_from_ibtc_miss
            + s.fetch_from_interpreter
            + s.fetch_from_mode_restart,
        s.fetch_private_cold
            + s.fetch_private_collision
            + s.fetch_private_epoch
            + s.fetch_private_mode
            + s.fetch_private_bypass,
    )
}

#[test]
fn shared_reasons_partition_resolves_and_survive_invalidation() {
    let mut vm = vm(JitBackend::new(), &[0xb8, 7, 0x10, 0, 0, 0xff, 0xe0, 0xf4]);
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let first = vm.time_stats();
    assert_eq!(first.fetch_from_entry, 1);
    assert_eq!(first.fetch_from_ibtc_miss, 1);
    assert_eq!(shared_partitions(first), (2, 2));
    vm.write_bytes(0x1007, &[0x90, 0xf4]).unwrap();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let invalidated = vm.time_stats();
    assert!(invalidated.fetch_private_epoch > 0);
    assert_eq!(
        shared_partitions(invalidated),
        (
            invalidated.fetch_translate.calls,
            invalidated.fetch_translate.calls
        )
    );
    vm.set_time_accounting(false);
    vm.write_bytes(0x1007, &[0xf4]).unwrap();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let disabled = vm.time_stats();
    assert_eq!(shared_partitions(disabled), shared_partitions(invalidated));
}

#[test]
fn budget_and_request_resumptions_are_attributed_only_on_shared_resolve() {
    let vm = vm(JitBackend::new(), &[0xf4, 0xf4]);
    let mut cpu = vm.new_vcpu();
    assert!(matches!(cpu.run(&vm, Some(0)), Exit::BudgetExhausted));
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(vm.time_stats().fetch_from_budget_resume, 1);
    cpu.exit_request().request_exit();
    assert!(matches!(cpu.run(&vm, None), Exit::Interrupted));
    cpu.set_reg(Reg::Rip, 0x1001);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(vm.time_stats().fetch_from_exit_request_resume, 1);
    let count = vm.time_stats().fetch_translate.calls;
    cpu.set_reg(Reg::Rip, 0x1001);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(
        vm.time_stats().fetch_translate.calls,
        count,
        "private hit has no shared reason"
    );
}

#[test]
fn private_conflict_and_mode_flush_are_distinct() {
    let vm = vm(JitBackend::new(), &[0xf4]);
    // Three RIPs in one set: the third must evict from the two-way cache.
    for pc in [0x2db, 0xfde] {
        vm.write_bytes(pc, &[0xf4]).unwrap();
    }
    let mut cpu = vm.new_vcpu();
    for pc in [0x1000, 0x2db, 0xfde, 0x1000] {
        cpu.set_reg(Reg::Rip, pc);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    }
    assert_eq!(vm.time_stats().fetch_private_cold, 2);
    assert_eq!(vm.time_stats().fetch_private_collision, 2);
    cpu.cpu.mxcsr ^= 1 << 13;
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(vm.time_stats().fetch_private_mode, 1);
    let s = vm.time_stats();
    assert_eq!(
        shared_partitions(s),
        (s.fetch_translate.calls, s.fetch_translate.calls)
    );
}

#[test]
fn accounting_enabled_after_a_disabled_epoch_flush_keeps_the_flush_reason() {
    let mut vm = vm(JitBackend::new(), &[0xf4, 0xf4]);
    vm.set_time_accounting(false);
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    vm.write_bytes(0x1000, &[0xf4]).unwrap();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(shared_partitions(vm.time_stats()), (0, 0));
    vm.set_time_accounting(true);
    cpu.set_reg(Reg::Rip, 0x1001);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let s = vm.time_stats();
    assert_eq!(s.fetch_private_epoch, 1);
    assert_eq!(
        shared_partitions(s),
        (
            s.fetch_hits.calls + s.fetch_misses,
            s.fetch_hits.calls + s.fetch_misses
        )
    );
}

#[test]
fn held_shard_counts_contention_only_while_enabled() {
    use std::sync::{Arc, Barrier};
    use std::time::Duration;
    use x86jit_core::{BlockKey, CpuMode};

    let mut vm = vm(JitBackend::new(), &[0xf4]);
    vm.set_time_accounting(false);
    let mut warm = vm.new_vcpu();
    warm.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(warm.run(&vm, None), Exit::Hlt));
    let key = BlockKey::new(0x1000, CpuMode::Long64);
    let block = vm.cache.get(key).unwrap();
    for enabled in [false, true] {
        vm.set_time_accounting(enabled);
        let held = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        std::thread::scope(|scope| {
            scope.spawn(|| {
                vm.cache.insert(key, block.clone(), vec![(0x1000, 1)], |_| {
                    held.wait();
                    release.wait();
                });
            });
            held.wait();
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(50));
                release.wait();
            });
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, 0x1000);
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        });
        let stats = vm.time_stats();
        let contention = stats.fetch_shard_read_contention;
        if enabled {
            assert_eq!(contention.calls, 1);
            assert_eq!(contention.samples, 1);
            assert!(contention.sample_ns >= 10_000_000, "{stats:?}");
            assert_eq!(stats.fetch_hits.samples, 1);
            assert_eq!(stats.fetch_hit_slow_samples, 1);
            assert_eq!(stats.fetch_hit_slow_ns, stats.fetch_hits.sample_ns);
            assert_eq!(stats.fetch_hit_publication_ns, 0);
        } else {
            assert_eq!(contention.calls, 0);
            assert_eq!(contention.samples, 0);
            assert_eq!(contention.sample_ns, 0);
            assert_eq!(stats.fetch_hit_slow_samples, 0);
            assert_eq!(stats.fetch_hit_slow_ns, 0);
        }
    }
}

#[test]
fn sleeping_hit_drain_is_slow_and_publication_overlaps_hit_time() {
    use std::time::Duration;
    use x86jit_core::{Backend, CachedBlock, IrBlock, MemConsistency, TierUpFinished};
    struct SleepingDrain(JitBackend);
    impl Backend for SleepingDrain {
        fn code_lifetime(
            &self,
        ) -> Option<std::sync::Arc<x86jit_core::code_lifetime::CodeLifetime>> {
            self.0.code_lifetime()
        }

        fn materialize(
            &self,
            ir: &IrBlock,
            order: MemConsistency,
            mmio: Option<(u64, u64)>,
            base: u64,
        ) -> CachedBlock {
            self.0.materialize(ir, order, mmio, base)
        }
        fn tier_up_finished(&self) -> Vec<TierUpFinished> {
            std::thread::sleep(Duration::from_millis(5));
            Vec::new()
        }
    }
    let mut vm = Vm::with_backend(
        VmConfig::flat(0x4000),
        Box::new(SleepingDrain(JitBackend::new())),
    );
    vm.map(0, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, &[0xf4]).unwrap();
    vm.set_tier_up_background(true);
    vm.set_time_accounting(true);
    let mut cold = vm.new_vcpu();
    cold.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cold.run(&vm, None), Exit::Hlt));
    let missed = vm.time_stats();
    assert_eq!(
        missed.fetch_hit_publication_ns, 0,
        "miss drains are excluded"
    );
    let mut hit = vm.new_vcpu();
    hit.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(hit.run(&vm, None), Exit::Hlt));
    let stats = vm.time_stats();
    assert_eq!(stats.fetch_hits.calls, 1);
    assert_eq!(stats.fetch_hits.samples, 1);
    assert_eq!(stats.fetch_hit_slow_samples, 1);
    assert_eq!(stats.fetch_hit_slow_ns, stats.fetch_hits.sample_ns);
    assert!(stats.fetch_hit_publication_ns >= 4_000_000, "{stats:?}");
    assert!(stats.fetch_hits.sample_ns >= stats.fetch_hit_publication_ns);
    assert_eq!(stats.fetch_shard_read_contention.calls, 0);
    vm.set_time_accounting(false);
    let mut disabled = vm.new_vcpu();
    disabled.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(disabled.run(&vm, None), Exit::Hlt));
    let after = vm.time_stats();
    assert_eq!(
        after.fetch_hit_publication_ns,
        stats.fetch_hit_publication_ns
    );
    assert_eq!(after.fetch_hit_slow_samples, stats.fetch_hit_slow_samples);
    assert_eq!(after.fetch_hit_slow_ns, stats.fetch_hit_slow_ns);
}

#[test]
fn internal_mxcsr_restart_has_its_own_origin() {
    // Load a different supported mode, then continue in a new specialization.
    let vm = vm(JitBackend::new(), &[0x0f, 0xae, 0x15, 0xf9, 7, 0, 0, 0xf4]);
    vm.write_bytes(0x1800, &0x3f80_u32.to_le_bytes()).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let stats = vm.time_stats();
    assert_eq!(stats.fetch_from_entry, 1);
    assert_eq!(stats.fetch_from_mode_restart, 1);
    assert_eq!(stats.fetch_from_budget_resume, 0);
    assert_eq!(shared_partitions(stats), (2, 2));
}
