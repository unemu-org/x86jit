//! Shared-VM thunk/return dispatch witness. Arguments: vcpus rounds thunks [account].
//! Compilation and two warm sweeps are outside the counters.
//! Example: dispatch_thunks 4 1000 1024 account
use iced_x86::code_asm::*;
use std::sync::Barrier;
use std::time::Instant;
use x86jit_core::{Exit, Prot, Reg, RegionKind, Vm, VmConfig};
use x86jit_cranelift::JitBackend;

#[path = "dispatch_support/usage.rs"]
mod usage;
use usage::retired;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let n: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(1);
    let rounds: u64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1000);
    let thunks: u64 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1024);
    let accounting = args.get(4).is_some_and(|s| s == "account");
    assert!((1..=1024).contains(&n));
    assert!((1..=4096).contains(&thunks));
    assert!((1..=100_000).contains(&rounds));
    let memory_size = 0x82000 + n as u64 * 0x2000;
    let jit = JitBackend::new();
    jit.enable_icount();
    let mut vm = Vm::with_backend(VmConfig::flat(memory_size), Box::new(jit));
    vm.set_time_accounting(accounting);
    vm.map(0, memory_size as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    let mut sweep = a.create_label();
    let mut call = a.create_label();
    a.set_label(&mut sweep).unwrap();
    a.mov(rsi, rbx).unwrap();
    a.mov(edi, thunks as u32).unwrap();
    a.set_label(&mut call).unwrap();
    a.call(qword_ptr(rsi)).unwrap();
    a.add(rsi, 8).unwrap();
    a.dec(edi).unwrap();
    a.jnz(call).unwrap();
    a.dec(ecx).unwrap();
    a.jnz(sweep).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
        .unwrap();
    // Each small import thunk jumps through a memory slot to a shared callee.
    for i in 0..thunks {
        let pc = 0x10000 + i * 32;
        let mut a = CodeAssembler::new(64).unwrap();
        a.jmp(qword_ptr(pc + 16)).unwrap();
        vm.write_bytes(pc, &a.assemble(pc).unwrap()).unwrap();
        vm.write_bytes(pc + 16, &0x2000u64.to_le_bytes()).unwrap();
        vm.write_bytes(0x40000 + i * 8, &pc.to_le_bytes()).unwrap();
    }
    vm.write_bytes(0x2000, &[0x48, 0xff, 0xc0, 0xc3]).unwrap(); // inc rax; ret
    let mut cpus: Vec<_> = (0..n)
        .map(|i| {
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rsp, 0x80000 + i as u64 * 0x2000);
            cpu.set_reg(Reg::Rbx, 0x40000);
            cpu.set_reg(Reg::Rip, 0x1000);
            cpu.set_reg(Reg::Rcx, 2);
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rax), 2 * thunks);
            cpu
        })
        .collect();
    let before = vm.time_stats();
    let ready = Barrier::new(n + 1);
    let done = Barrier::new(n + 1);
    let go = Barrier::new(n + 1);
    std::thread::scope(|scope| {
        for cpu in &mut cpus {
            let (vm, ready, done, go) = (&vm, &ready, &done, &go);
            scope.spawn(move || {
                let guest_before = cpu.executed_instructions();
                cpu.set_reg(Reg::Rax, 0);
                cpu.set_reg(Reg::Rip, 0x1000);
                cpu.set_reg(Reg::Rcx, rounds);
                ready.wait();
                go.wait();
                assert!(matches!(cpu.run(vm, None), Exit::Hlt));
                assert_eq!(cpu.reg(Reg::Rax), rounds * thunks);
                assert_eq!(
                    cpu.executed_instructions() - guest_before,
                    rounds * (7 * thunks + 4) + 1
                );
                done.wait();
            });
        }
        ready.wait();
        let instructions = retired();
        let start = Instant::now();
        go.wait();
        done.wait();
        let ns = start.elapsed().as_nanos();
        let instructions = retired() - instructions;
        let after = vm.time_stats();
        let guest = n as u64 * (rounds * (7 * thunks + 4) + 1);
        // One dispatch = one compiled-unit invocation, including chained transfers.
        let dispatches =
            after.block_lookups - before.block_lookups + after.chain_hits - before.chain_hits;
        let shared = after.fetch_translate.calls - before.fetch_translate.calls;
        println!("vcpus,rounds,thunks,account,ns,host_instructions,guest_instructions,dispatches,shared_resolves,host_per_dispatch,host_per_guest");
        println!("{n},{rounds},{thunks},{accounting},{ns},{instructions},{guest},{dispatches},{shared},{:.3},{:.3}", instructions as f64 / dispatches as f64, instructions as f64 / guest as f64);
        macro_rules! delta { ($($f:ident),*) => { $(println!("{},{}", stringify!($f), after.$f - before.$f);)* }; }
        delta!(
            fetch_misses,
            fetch_from_mode_restart,
            fetch_hit_slow_samples,
            fetch_hit_slow_ns,
            fetch_hit_publication_ns,
            fetch_from_entry,
            fetch_from_budget_resume,
            fetch_from_exit_request_resume,
            fetch_from_boundary,
            fetch_from_return_miss,
            fetch_from_direct_link,
            fetch_from_ibtc_miss,
            fetch_from_interpreter,
            fetch_private_cold,
            fetch_private_collision,
            fetch_private_epoch,
            fetch_private_mode,
            fetch_private_bypass,
            indirect_branch_resolves
        );
        println!(
            "shard_read_contended,{}",
            after.fetch_shard_read_contention.calls - before.fetch_shard_read_contention.calls
        );
        println!(
            "shard_read_wait_samples,{}",
            after.fetch_shard_read_contention.samples - before.fetch_shard_read_contention.samples
        );
        println!(
            "shard_read_wait_sample_ns,{}",
            after.fetch_shard_read_contention.sample_ns
                - before.fetch_shard_read_contention.sample_ns
        );
        println!(
            "hit_samples,{}",
            after.fetch_hits.samples - before.fetch_hits.samples
        );
        println!(
            "hit_sample_ns,{}",
            after.fetch_hits.sample_ns - before.fetch_hits.sample_ns
        );
        println!(
            "sampled_hit_ns,{:.3}",
            (after.fetch_hits.sample_ns - before.fetch_hits.sample_ns) as f64
                / (after.fetch_hits.samples - before.fetch_hits.samples).max(1) as f64
        );
    });
}
