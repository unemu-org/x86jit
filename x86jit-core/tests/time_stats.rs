use std::time::Duration;
use x86jit_core::{CpuMode, Exit, Prot, Reg, RegionKind, StepOptions, Vm, VmConfig};

fn make_vm(code: &[u8]) -> Vm {
    let mut vm = Vm::new(VmConfig::flat(0x4000));
    vm.set_time_accounting(true);
    vm.map(0, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x2000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    vm.write_bytes(0x1000, code).unwrap();
    vm
}

#[test]
fn resolve_hits_and_misses_are_separate_populations() {
    // The initial block executes the first decrement; the loop block executes
    // 129 more times, followed by HLT: 131 resolves, three cold blocks.
    let vm = make_vm(&[0xb9, 130, 0, 0, 0, 0xff, 0xc9, 0x75, 0xfc, 0xf4]);
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let stats = vm.time_stats();
    assert_eq!(stats.fetch_translate.calls, 131);
    assert_eq!(stats.fetch_translate.samples, 3);
    assert_eq!(stats.fetch_misses, 3);
    assert_eq!(stats.fetch_hits.calls, 128);
    assert_eq!(stats.fetch_hits.samples, 2);
    assert!(stats.fetch_hits.sample_ns > 0);
    assert!(stats.fetch_lift_ns > 0);
    assert_eq!(stats.fetch_miss_lift_ns, stats.fetch_lift_ns);
    assert!(stats.fetch_miss_other_ns > 0);
    assert_eq!(stats.fetch_approval_ns, 0);
}

#[test]
fn every_policy_callback_is_timed_including_unsampled_misses() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    // Jump from the second page to HLT on the first: two approvals in one run,
    // while the compatibility sampler times only the first resolve.
    let mut vm = make_vm(&[0xe9, 0xfb, 0xef, 0xff, 0xff]);
    vm.write_bytes(0, &[0xf4]).unwrap();
    let callbacks = Arc::new(AtomicU64::new(0));
    let observed = callbacks.clone();
    vm.set_fetch_policy(Box::new(move |_| {
        observed.fetch_add(1, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(5));
        x86jit_core::FetchVerdict::Allow
    }));
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let stats = vm.time_stats();
    assert_eq!(callbacks.load(Ordering::Relaxed), 2);
    assert_eq!(stats.fetch_misses, 2);
    assert_eq!(stats.fetch_hits.calls, 0);
    assert_eq!(stats.fetch_translate.samples, 1);
    assert!(stats.fetch_approval_ns >= 10_000_000);
    assert_eq!(stats.fetch_miss_approval_ns, stats.fetch_approval_ns);
    assert_eq!(stats.fetch_miss_lift_ns, stats.fetch_lift_ns);
    assert!(stats.fetch_lift_ns > 0);
}

#[test]
fn nested_child_view_callbacks_are_counted_once() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    let calls = Arc::new(AtomicU64::new(0));
    let observed = calls.clone();
    let mut child_source = make_vm(&[0xf4]);
    child_source.set_fetch_policy(Box::new(move |_| {
        observed.fetch_add(1, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(10));
        x86jit_core::FetchVerdict::Allow
    }));
    let mut vm = make_vm(&[0xf4]);
    vm.set_fetch_policy(Box::new(move |_| {
        let _view = child_source.child_view(CpuMode::Long64);
        x86jit_core::FetchVerdict::Allow
    }));
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let stats = vm.time_stats();
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(stats.fetch_translate.calls, 1);
    assert_eq!(stats.fetch_translate.samples, 1);
    assert_eq!(stats.fetch_misses, 1);
    assert!(stats.fetch_miss_approval_ns >= 20_000_000);
    assert!(stats.fetch_miss_lift_ns > 0);
    assert!(
        stats.fetch_miss_approval_ns + stats.fetch_miss_lift_ns + stats.fetch_miss_other_ns
            <= stats.fetch_translate.sample_ns,
        "{stats:?}"
    );
}

#[test]
fn disabled_resolve_accounting_leaves_split_fields_zero() {
    let mut vm = make_vm(&[0xb9, 130, 0, 0, 0, 0xff, 0xc9, 0x75, 0xfc, 0xf4]);
    vm.set_time_accounting(false);
    vm.set_fetch_policy(Box::new(|_| x86jit_core::FetchVerdict::Allow));
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let stats = vm.time_stats();
    assert_eq!(stats.fetch_translate.calls, 131);
    assert_eq!(stats.fetch_hits.calls, 0);
    assert_eq!(stats.fetch_hits.samples, 0);
    assert_eq!(stats.fetch_hits.sample_ns, 0);
    assert_eq!(stats.fetch_misses, 0);
    assert_eq!(stats.fetch_approval_ns, 0);
    assert_eq!(stats.fetch_lift_ns, 0);
    assert_eq!(stats.fetch_miss_approval_ns, 0);
    assert_eq!(stats.fetch_miss_lift_ns, 0);
    assert_eq!(stats.fetch_miss_other_ns, 0);
}

#[test]
fn child_view_policy_callbacks_outside_resolve_are_excluded() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    let mut vm = make_vm(&[0xf4]);
    let calls = Arc::new(AtomicU64::new(0));
    let observed = calls.clone();
    vm.set_fetch_policy(Box::new(move |_| {
        observed.fetch_add(1, Ordering::Relaxed);
        x86jit_core::FetchVerdict::Allow
    }));
    let _view = vm.child_view(CpuMode::Long64);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    let stats = vm.time_stats();
    assert_eq!(stats.fetch_approval_ns, 0);
    assert_eq!(stats.fetch_miss_approval_ns, 0);
    assert_eq!(stats.fetch_misses, 0);
    assert_eq!(stats.fetch_hits.calls, 0);
}

#[test]
fn empty_and_interpreter_totals_survive_vcpu_drop() {
    let vm = make_vm(&[0xb9, 100, 0, 0, 0, 0xff, 0xc9, 0x75, 0xfc, 0xf4]);
    let empty = vm.time_stats();
    assert_eq!(empty.interpreter_ns, 0);
    assert_eq!(empty.block_lookups, 0);
    assert_eq!(empty.exits.hlt.count, 0);
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let first = vm.time_stats();
    assert!(first.interpreter_ns > 0);
    assert_eq!(first.compiled_ns, 0);
    assert_eq!(first.foreground_compile_ns, 0);
    assert!(first.block_lookups >= 100);
    assert!(first.block_misses > 0);
    assert!(first.fetch_translate.calls >= 100);
    assert!(first.fetch_translate.samples > 0);
    assert_eq!(first.exits.hlt.count, 1);
    std::thread::sleep(Duration::from_millis(2));
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let second = vm.time_stats();
    assert!(second.interpreter_ns >= first.interpreter_ns);
    assert!(second.fetch_translate.sample_ns >= first.fetch_translate.sample_ns);
    assert!(second.exits.hlt.outside_ns >= 1_000_000);
    assert_eq!(second.exits.hlt.count, 2);
    drop(cpu);
    let dropped = vm.time_stats();
    assert_eq!(dropped.interpreter_ns, second.interpreter_ns);
    assert_eq!(vm.time_stats().interpreter_ns, second.interpreter_ns);
    let mut another = vm.new_vcpu();
    another.set_reg(Reg::Rip, 0x1000);
    another.run(&vm, None);
    assert_eq!(vm.time_stats().exits.hlt.count, 3);
}

#[test]
fn exits_by_kind_and_completed_outside_intervals() {
    let vm = make_vm(&[0x0f, 0x05, 0xe6, 0x80, 0x8b, 0x00, 0xf4]);
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Syscall));
    std::thread::sleep(Duration::from_millis(1));
    assert!(matches!(cpu.run(&vm, None), Exit::PortIo { .. }));
    cpu.set_reg(Reg::Rax, 0x2000);
    assert!(matches!(cpu.run(&vm, None), Exit::MmioRead { .. }));
    cpu.complete_mmio_read(42);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    cpu.set_reg(Reg::Rip, 0x3000);
    assert!(matches!(cpu.run(&vm, None), Exit::UnmappedMemory { .. }));
    cpu.exit_request().request_exit();
    assert!(matches!(cpu.run(&vm, None), Exit::Interrupted));
    let stats = vm.time_stats();
    assert_eq!(stats.exits.syscall.count, 1);
    assert!(stats.exits.syscall.outside_ns > 0);
    assert_eq!(stats.exits.port_io.count, 1);
    assert!(stats.exits.port_io.outside_ns > 0);
    assert_eq!(stats.exits.mmio.count, 1);
    assert!(stats.exits.mmio.outside_ns > 0);
    assert_eq!(stats.exits.hlt.count, 1);
    assert_eq!(stats.exits.fault.count, 1);
    assert_eq!(stats.exits.interrupted.count, 1);
    assert_eq!(stats.exits.interrupted.outside_ns, 0);
}

#[test]
fn steps_zero_budget_refusal_and_timing_toggle() {
    let mut vm = make_vm(&[0x90, 0xf4]);
    vm.set_cpu_mode(CpuMode::Real16);
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert_eq!(
        cpu.run_steps(&vm, 0, StepOptions::default())
            .unwrap()
            .steps(),
        0
    );
    assert_eq!(vm.time_stats().exits.budget.count, 1);
    assert_eq!(
        cpu.run_steps(&vm, 1, StepOptions::default())
            .unwrap()
            .steps(),
        1
    );
    let first = vm.time_stats();
    assert!(first.interpreter_ns > 0);
    assert_eq!(first.exits.budget.count, 2);
    vm.set_time_accounting(false);
    cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    let second = vm.time_stats();
    assert_eq!(first.interpreter_ns, second.interpreter_ns);
    assert_eq!(second.exits.hlt.count, 1);
    assert_eq!(
        second.exits.budget.outside_ns,
        first.exits.budget.outside_ns
    );
    let mut other = make_vm(&[0xf4]);
    other.set_time_accounting(false);
    let mut cpu = other.new_vcpu();
    assert!(cpu.run_steps(&other, 1, StepOptions::default()).is_err());
    assert_eq!(other.time_stats().exits.other.count, 0);
}

#[test]
fn vcpu_records_sum_across_host_threads() {
    let vm = make_vm(&[0xb9, 100, 0, 0, 0, 0xff, 0xc9, 0x75, 0xfc, 0xf4]);
    std::thread::scope(|scope| {
        for _ in 0..2 {
            let vm = &vm;
            scope.spawn(move || {
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, 0x1000);
                assert!(matches!(cpu.run(vm, None), Exit::Hlt));
            });
        }
    });
    let stats = vm.time_stats();
    assert_eq!(stats.exits.hlt.count, 2);
    assert_eq!(stats.block_lookups, vm.cache.hits() + vm.cache.misses());
    assert_eq!(stats.block_misses, vm.cache.misses());
    assert!(stats.interpreter_ns > 0);
}

fn fetch_panic(resume: bool, timing: bool, steps: bool) {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::atomic::{AtomicBool, Ordering};
    let mut vm = make_vm(&[0xf4]);
    vm.set_time_accounting(timing);
    if steps {
        vm.set_cpu_mode(CpuMode::Real16);
    }
    let first = AtomicBool::new(true);
    vm.set_fetch_policy(Box::new(move |_| {
        assert!(!first.swap(false, Ordering::Relaxed), "fetch policy panic");
        x86jit_core::FetchVerdict::Allow
    }));
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    let result = catch_unwind(AssertUnwindSafe(|| {
        if steps {
            cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        } else {
            cpu.run(&vm, None);
        }
    }));
    assert!(result.is_err());
    let first = vm.time_stats();
    assert_eq!(first.block_lookups, 1);
    assert_eq!(first.block_misses, 1);
    assert_eq!(first.fetch_translate.calls, 1);
    assert_eq!(first.fetch_translate.samples, u64::from(timing));
    assert_eq!(first.fetch_misses, u64::from(timing));
    assert_eq!(first.fetch_hits.calls, 0);
    assert_eq!(vm.cache.hits(), 0);
    assert_eq!(vm.cache.misses(), 1);
    assert_eq!(first.exits.hlt.count, 0);
    if resume {
        if steps {
            cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        } else {
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        }
    }
    drop(cpu);
    let totals = vm.time_stats();
    let calls = if resume { 2 } else { 1 };
    assert_eq!(totals.fetch_translate.calls, calls);
    assert_eq!(totals.fetch_misses, if timing { calls } else { 0 });
    assert_eq!(totals.block_lookups, calls);
    assert_eq!(totals.block_misses, calls);
    assert_eq!(vm.cache.misses(), calls);
    assert_eq!(totals.exits.hlt.count, u64::from(resume));
}

#[test]
fn unwound_fetch_totals_survive_vcpu_drop() {
    for timing in [false, true] {
        for steps in [false, true] {
            fetch_panic(false, timing, steps);
        }
    }
}

#[test]
fn unwound_fetch_resume_counts_each_call_once() {
    for timing in [false, true] {
        for steps in [false, true] {
            fetch_panic(true, timing, steps);
        }
    }
}
