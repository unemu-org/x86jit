//! Targeted MXCSR measurements. Build with --release; execute `default` in
//! interleaved reference/candidate rounds, then execute `modes` on the candidate.

#[allow(dead_code)]
#[path = "../src/workloads.rs"]
mod workloads;

use iced_x86::code_asm::*;
use std::hint::black_box;
use std::time::Instant;
use x86jit_core::{
    Backend, Exit, InterpreterBackend, Prot, Reg, RegionCaps, RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::{HostTarget, JitBackend, OptLevel};

fn backend(tier: usize) -> Box<dyn Backend> {
    match tier {
        0 => Box::new(InterpreterBackend),
        1 => Box::new(JitBackend::new()),
        _ => Box::new(JitBackend::with_superblocks(RegionCaps {
            max_blocks: 16,
            max_icount: 256,
        })),
    }
}

fn defaults() {
    for wl in workloads::all()
        .into_iter()
        .filter(|w| ["simd", "hotloop", "memcpy"].contains(&w.name))
    {
        for tier in 0..3 {
            let (warm, _) = (wl.guest)(backend(tier), workloads::TierCfg::EAGER);
            assert_eq!(warm, wl.expect);
            let start = Instant::now();
            let (out, counters) = (wl.guest)(backend(tier), workloads::TierCfg::EAGER);
            let ns = start.elapsed().as_nanos();
            assert_eq!(out, wl.expect);
            println!("default,{},{tier},{ns},{}", wl.name, counters.helper_calls);
        }
    }
}

fn loop_mode(tier: usize, mode: u32) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend(tier));
    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    let mut top = a.create_label();
    let mut tail = a.create_label();
    let mut done = a.create_label();
    a.mov(ecx, 100_000).unwrap();
    a.jmp(top).unwrap();
    a.set_label(&mut top).unwrap();
    for _ in 0..4 {
        a.addps(xmm0, xmm1).unwrap();
        a.mulps(xmm0, xmm2).unwrap();
    }
    a.dec(ecx).unwrap();
    a.jz(done).unwrap();
    a.set_label(&mut tail).unwrap();
    a.jmp(top).unwrap();
    a.set_label(&mut done).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
        .unwrap();
    let mut samples = Vec::new();
    for sample in 0..4 {
        let mut cpu = vm.new_vcpu_with_mxcsr(mode);
        cpu.set_reg(Reg::Rip, 0x1000);
        cpu.set_xmm(0, 0x3f800000_3f800000_3f800000_3f800000);
        cpu.set_xmm(1, 0x3b800000_3b800000_3b800000_3b800000);
        cpu.set_xmm(2, 0x3f7f0000_3f7f0000_3f7f0000_3f7f0000);
        let start = Instant::now();
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        let ns = start.elapsed().as_nanos();
        black_box(cpu.xmm(0));
        if sample != 0 {
            samples.push(ns);
        }
    }
    samples.sort_unstable();
    let helpers = vm
        .backend
        .helper_calls()
        .iter()
        .map(|(_, n)| n)
        .sum::<u64>();
    println!(
        "mode,{mode:04x},{tier},{},{helpers},{}",
        samples[1],
        vm.cache.regions()
    );
}

fn helper_mode(mode: u32, op: usize) {
    let mut vm = Vm::with_backend(
        VmConfig::flat(0x10000),
        Box::new(JitBackend::with_options(
            None,
            HostTarget::Baseline,
            OptLevel::None,
        )),
    );
    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    match op {
        0 => a.vfmadd132ps(xmm0, xmm2, xmm1).unwrap(),
        1 => a.haddps(xmm0, xmm1).unwrap(),
        2 => a.vcvtps2ph(xmm0, xmm1, 4).unwrap(),
        3 => a.vrcpps(xmm0, xmm1).unwrap(),
        4 => a.mulps(xmm0, xmm1).unwrap(),
        _ => unreachable!(),
    }
    a.hlt().unwrap();
    vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
        .unwrap();
    let mut cpu = vm.new_vcpu_with_mxcsr(mode);
    cpu.set_reg(Reg::Rip, 0x1000);
    cpu.set_xmm(0, 0x00800000_00800000_00800000_00800000);
    cpu.set_xmm(1, 0x3f7fffff_3f7fffff_3f7fffff_3f7fffff);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    println!("helper,{mode:04x},{op},{:?}", vm.backend.helper_calls());
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("default") {
        defaults();
    } else if std::env::args().nth(1).as_deref() == Some("helpers") {
        for op in 0..5 {
            helper_mode(0x9fc0, op);
        }
    } else {
        for mode in [0x1f80, 0x9fc0, 0x7f80] {
            for tier in 0..3 {
                loop_mode(tier, mode);
            }
        }
        for op in 0..5 {
            helper_mode(0x9fc0, op);
        }
    }
}
