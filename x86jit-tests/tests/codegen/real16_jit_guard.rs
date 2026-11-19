//! Real16 execution policy: a compiling backend must still use the interpreter.

use iced_x86::code_asm::*;
use x86jit_core::{
    Backend, CpuMode, Exit, InterpreterBackend, Prot, Reg, RegionCaps, RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const CODE: u64 = 0x1000;
const OUT: u64 = 0x8000;

fn vm_with(backend: Box<dyn Backend>, tier: Option<u32>, background: bool) -> Vm {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
    vm.set_cpu_mode(CpuMode::Real16);
    vm.set_tier_up_after(tier);
    vm.set_tier_up_background(background);
    vm.set_tier_up_region_after(Some(2));
    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();

    let mut a = CodeAssembler::new(16).unwrap();
    let mut top = a.create_label();
    let mut body = a.create_label();
    a.mov(ax, 0).unwrap();
    a.mov(cx, 8).unwrap();
    a.set_label(&mut top).unwrap();
    a.jmp(body).unwrap();
    a.set_label(&mut body).unwrap();
    a.add(ax, cx).unwrap();
    a.dec(cx).unwrap();
    a.stc().unwrap(); // SDM Vol 2B, STC, Operation: CF = 1 (Real16-only IR).
    a.jnz(top).unwrap();
    a.mov(word_ptr(OUT), ax).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(CODE, &a.assemble(0).unwrap()).unwrap();
    vm
}

fn run(vm: &Vm) -> ([u64; 12], u64, u64, u64) {
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Cs, 0x100);
    cpu.set_reg(Reg::Rip, 0);
    let exit = cpu.run(vm, Some(100));
    assert!(matches!(exit, Exit::Hlt), "unexpected exit: {exit:?}");
    assert!(cpu.flags().cf);
    (
        [
            Reg::Rax,
            Reg::Rcx,
            Reg::Rdx,
            Reg::Rbx,
            Reg::Rsp,
            Reg::Rbp,
            Reg::Rsi,
            Reg::Rdi,
            Reg::Cs,
            Reg::Ds,
            Reg::Es,
            Reg::Ss,
        ]
        .map(|r| cpu.reg(r)),
        cpu.reg(Reg::Rip),
        cpu.flags().to_rflags(),
        vm.mem.read(OUT, 2).unwrap(),
    )
}

fn check(superblocks: bool, tier: Option<u32>, background: bool) {
    let expected = run(&vm_with(Box::new(InterpreterBackend), tier, background));
    let jit = if superblocks {
        JitBackend::with_superblocks(RegionCaps {
            max_blocks: 16,
            max_icount: 256,
        })
    } else {
        JitBackend::new()
    };
    let handle = jit.tier_up_handle();
    let vm = vm_with(Box::new(jit), tier, background);
    // A forbidden submission must fail deterministically before its worker can panic.
    let pause = handle.pause_compiler();
    for _ in 0..2 {
        assert_eq!(run(&vm), expected);
        assert_eq!(vm.cache.tier_pending_len(), 0, "no background requests");
        assert_eq!(vm.backend.compile_ns(), 0, "no inline compilation");
        assert_eq!(vm.cache.regions(), 0, "no compiled regions");
    }
    drop(pause);
    handle.wait_idle();
    assert_eq!(vm.cache.tier_bg_published(), 0);
    assert_eq!(vm.cache.tier_bg_rejected(), 0);
}

#[test]
fn real16_jit_eager() {
    check(false, None, false);
}

#[test]
fn real16_jit_inline_tier_up() {
    check(false, Some(1), false);
}

#[test]
fn real16_jit_background_tier_up() {
    check(false, Some(1), true);
}

#[test]
fn real16_jit_superblocks_eager() {
    check(true, None, false);
}

#[test]
fn real16_jit_superblocks_inline_tier_up() {
    check(true, Some(1), false);
}

#[test]
fn real16_jit_superblocks_background_tier_up() {
    check(true, Some(1), true);
}
