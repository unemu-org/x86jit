//! Deterministic background tier-up churn witness. Run in release mode; arguments:
//! control|same|different, rounds (default 100). Each round pauses compilation while
//! a guest call/store loop runs, then finishes the worker and drains publication.
use iced_x86::code_asm::*;
use x86jit_core::{Exit, Prot, Reg, RegionKind, Vm, VmConfig};
use x86jit_cranelift::JitBackend;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let variant = args.get(1).map(String::as_str).unwrap_or("same");
    assert!(matches!(variant, "control" | "same" | "different"));
    let rounds: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(100);
    let jit = JitBackend::new();
    let handle = jit.tier_up_handle();
    let mut vm = Vm::with_backend(VmConfig::flat(0x9000), Box::new(jit));
    vm.set_tier_up_after(Some(32));
    vm.set_tier_up_background(true);
    vm.map(0, 0x9000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, &[0xb8, 42, 0, 0, 0, 0xc3]).unwrap();
    vm.write_bytes(0x2000, &[0xf4]).unwrap();
    vm.write_bytes(0x6000, &[0xf4]).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    let mut top = a.create_label();
    a.set_label(&mut top).unwrap();
    a.call(rbx).unwrap();
    if variant != "control" {
        a.mov(byte_ptr(rdi), 42).unwrap();
    }
    a.dec(rcx).unwrap();
    a.jnz(top).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(0x4000, &a.assemble(0x4000).unwrap())
        .unwrap();
    let mut cpu = vm.new_vcpu();
    // Tag a different page with real cached code, without heating that entry.
    cpu.set_reg(Reg::Rip, 0x2000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    cpu.set_reg(Reg::Rsp, 0x8000);
    cpu.set_reg(Reg::Rbx, 0x1000);
    cpu.set_reg(
        Reg::Rdi,
        if variant == "different" {
            0x2800
        } else {
            0x1800
        },
    );
    for _ in 0..rounds {
        let pause = handle.pause_compiler();
        cpu.set_reg(Reg::Rip, 0x4000);
        cpu.set_reg(Reg::Rcx, 128);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rax), 42);
        drop(pause);
        handle.wait_idle();
        cpu.set_reg(Reg::Rip, 0x6000);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    }
    handle.wait_idle();
    cpu.set_reg(Reg::Rip, 0x6000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let s = vm.memory_stats();
    let c = s.cache;
    println!("variant,rounds,units,entries,ratio,invalidations,dropped,submitted,completed,published,generation_rejected,compile_failed,superseded,duplicates_skipped");
    println!(
        "{variant},{rounds},{},{},{:.2},{},{},{},{},{},{},{},{},{}",
        s.backend.compiled_units_total,
        c.compiled_entries,
        s.backend.compiled_units_total as f64 / c.compiled_entries.max(1) as f64,
        c.invalidation_calls,
        c.invalidation_blocks_dropped,
        c.tier_bg_submitted,
        c.tier_bg_completed,
        c.tier_bg_published,
        c.tier_bg_rejected_generation,
        c.tier_bg_rejected_compile_failed,
        c.tier_bg_superseded,
        c.tier_bg_duplicates_skipped
    );
}
