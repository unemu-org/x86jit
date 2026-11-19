//! Recoverable code-storage exhaustion, without exhausting the host's mappings.

use iced_x86::code_asm::*;
use x86jit_core::{
    cache::BlockKey, Backend, CachedBlock, CpuMode, Exit, InterpreterBackend, Prot, Reg,
    RegionCaps, RegionKind, Vcpu, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const ENTRY: u64 = 0x1000;
const TARGET: u64 = 0x1100;

fn vm_with(backend: Box<dyn Backend>) -> Vm {
    let mut vm = Vm::with_backend(VmConfig::flat(0x3000), backend);
    vm.map(ENTRY, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
    vm
}

fn write_single(vm: &Vm, value: i32) {
    let mut a = CodeAssembler::new(64).unwrap();
    a.mov(eax, value).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(ENTRY, &a.assemble(ENTRY).unwrap()).unwrap();
}

fn assert_exhausted(cpu: &mut Vcpu, vm: &Vm, addr: u64) {
    assert!(
        matches!(cpu.run(vm, Some(100)), Exit::CodeSpaceExhausted { addr: at, .. } if at == addr)
    );
    assert_eq!(cpu.reg(Reg::Rip), addr);
    assert!(vm.backend.code_space_exhausted());
}

fn assert_hlt(cpu: &mut Vcpu, vm: &Vm, result: u64) {
    assert!(matches!(cpu.run(vm, Some(100)), Exit::Hlt));
    assert_eq!(cpu.reg(Reg::Rax), result);
}

#[test]
fn eager_exhaustion_preserves_state_and_falls_back_after_compiled_edge() {
    let jit = JitBackend::new();
    jit.set_compile_limit(1);
    let vm = vm_with(Box::new(jit));
    let mut a = CodeAssembler::new(64).unwrap();
    a.mov(eax, 7).unwrap();
    a.jmp(TARGET).unwrap();
    vm.write_bytes(ENTRY, &a.assemble(ENTRY).unwrap()).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    a.add(eax, 35).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(TARGET, &a.assemble(TARGET).unwrap())
        .unwrap();

    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, ENTRY);
    assert_exhausted(&mut cpu, &vm, TARGET);
    assert_eq!(
        cpu.reg(Reg::Rax),
        7,
        "failed compilation executes no guest code"
    );
    assert!(matches!(
        vm.cache.get(BlockKey::new(ENTRY, CpuMode::Long64)),
        Some(CachedBlock::Compiled { .. })
    ));
    assert!(matches!(
        vm.cache.get(BlockKey::new(TARGET, CpuMode::Long64)),
        Some(CachedBlock::Interpreted(_))
    ));
    assert_hlt(&mut cpu, &vm, 42);

    // Existing code remains usable; this vcpu is notified only once.
    for _ in 0..3 {
        cpu.set_reg(Reg::Rip, ENTRY);
        assert_hlt(&mut cpu, &vm, 42);
    }
    // Each vcpu gets its own notification, including one created after exhaustion.
    let mut other = vm.new_vcpu();
    other.set_reg(Reg::Rip, ENTRY);
    assert_exhausted(&mut other, &vm, ENTRY);
    assert_hlt(&mut other, &vm, 42);

    // Rewriting code after exhaustion still re-lifts it and interprets fresh bytes.
    write_single(&vm, 91);
    cpu.set_reg(Reg::Rip, ENTRY);
    assert_hlt(&mut cpu, &vm, 91);
}

#[test]
fn zero_compile_limit_matches_interpreter_after_resume() {
    for use_jit in [false, true] {
        let backend: Box<dyn Backend> = if use_jit {
            let jit = JitBackend::new();
            jit.set_compile_limit(0);
            Box::new(jit)
        } else {
            Box::new(InterpreterBackend)
        };
        let vm = vm_with(backend);
        write_single(&vm, 42);
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, ENTRY);
        if use_jit {
            assert_exhausted(&mut cpu, &vm, ENTRY);
            assert_eq!(cpu.reg(Reg::Rax), 0);
        }
        assert_hlt(&mut cpu, &vm, 42);
        cpu.set_reg(Reg::Rip, ENTRY);
        assert_hlt(&mut cpu, &vm, 42);
        assert_eq!(vm.backend.code_space_exhausted(), use_jit);
    }
}

#[test]
fn inline_tier_up_exhaustion_keeps_interpreted_block() {
    let jit = JitBackend::new();
    jit.set_compile_limit(0);
    let mut vm = vm_with(Box::new(jit));
    vm.set_tier_up_after(Some(1));
    write_single(&vm, 42);
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, ENTRY);
    assert_hlt(&mut cpu, &vm, 42);
    cpu.set_reg(Reg::Rip, ENTRY);
    assert_exhausted(&mut cpu, &vm, ENTRY);
    assert_hlt(&mut cpu, &vm, 42);
    cpu.set_reg(Reg::Rip, ENTRY);
    assert_hlt(&mut cpu, &vm, 42);
}

#[test]
fn background_exhaustion_notifies_and_drains_without_retry() {
    let jit = JitBackend::new();
    jit.set_compile_limit(0);
    let handle = jit.tier_up_handle();
    let pause = handle.pause_compiler();
    let mut vm = vm_with(Box::new(jit));
    vm.set_tier_up_after(Some(1));
    vm.set_tier_up_background(true);
    write_single(&vm, 42);
    let mut cpu = vm.new_vcpu();
    for _ in 0..2 {
        cpu.set_reg(Reg::Rip, ENTRY);
        assert_hlt(&mut cpu, &vm, 42);
    }
    drop(pause);
    handle.wait_idle();
    cpu.set_reg(Reg::Rip, ENTRY);
    assert_exhausted(&mut cpu, &vm, ENTRY);
    assert_hlt(&mut cpu, &vm, 42);
    assert_eq!(vm.cache.tier_bg_published(), 0);
    let key = BlockKey::new(ENTRY, CpuMode::Long64);
    assert!(
        vm.cache.try_begin_tier_up(key),
        "failed completion cleared its in-flight marker"
    );
    vm.cache.end_tier_up(key);
    for _ in 0..3 {
        cpu.set_reg(Reg::Rip, ENTRY);
        assert_hlt(&mut cpu, &vm, 42);
    }
    handle.wait_idle();
}

#[test]
fn region_exhaustion_interprets_entry_then_remaining_blocks() {
    let jit = JitBackend::with_superblocks(RegionCaps {
        max_blocks: 8,
        max_icount: 64,
    });
    jit.set_compile_limit(0);
    let vm = vm_with(Box::new(jit));
    let mut a = CodeAssembler::new(64).unwrap();
    let mut top = a.create_label();
    a.mov(eax, 0).unwrap();
    a.mov(ecx, 7).unwrap();
    a.set_label(&mut top).unwrap();
    a.add(eax, ecx).unwrap();
    a.sub(ecx, 1).unwrap();
    a.jnz(top).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(ENTRY, &a.assemble(ENTRY).unwrap()).unwrap();
    let region = x86jit_core::lift::lift_region(
        &vm.mem,
        ENTRY,
        vm.backend.region_caps().unwrap(),
        CpuMode::Long64,
    )
    .unwrap();
    assert!(
        region.has_loop && region.blocks.len() > 1,
        "exercise the region materializer"
    );
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, ENTRY);
    assert_exhausted(&mut cpu, &vm, ENTRY);
    assert_hlt(&mut cpu, &vm, 28);
    assert_eq!(vm.cache.regions(), 0);
    assert!(matches!(
        vm.cache.get(BlockKey::new(ENTRY, CpuMode::Long64)),
        Some(CachedBlock::Interpreted(_))
    ));
}
