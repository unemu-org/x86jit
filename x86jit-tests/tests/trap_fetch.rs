//! Trap regions never supply instruction bytes (spec.md §5.2, §7.3).

use iced_x86::code_asm::*;
use x86jit_core::lift::{lift_block, lift_one, FetchAddr, LiftError};
use x86jit_core::{
    AccessKind, Backend, CpuMode, Exit, InterpreterBackend, MemTrap, Prot, Reg, RegionKind, Vcpu,
    Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const CODE: u64 = 0x1000;
const TRAP: u64 = 0x2000;
const STACK: u64 = 0x4000;
const MARKER: u64 = 0x1234;

fn vm(backend: Box<dyn Backend>) -> Vm {
    let mut vm = Vm::with_backend(VmConfig::flat(STACK), backend);
    vm.map(CODE, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
    vm.map(TRAP, 0x1000, Prot::RWX, RegionKind::Trap).unwrap();
    vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
    vm
}

fn assemble(origin: u64, build: impl FnOnce(&mut CodeAssembler)) -> Vec<u8> {
    let mut a = CodeAssembler::new(64).unwrap();
    build(&mut a);
    a.assemble(origin).unwrap()
}

fn assert_fetch_exit(vm: &Vm, cpu: &mut Vcpu, addr: u64) {
    let exit = cpu.run(vm, Some(10));
    assert!(
        matches!(exit, Exit::UnmappedMemory { addr: a, access: AccessKind::Execute, .. } if a == addr),
        "expected execute fault at {addr:#x}, got {exit:?}"
    );
    assert_eq!(cpu.reg(Reg::Rip), addr);
}

macro_rules! both_backends {
    ($body:ident, $interp:ident, $jit:ident) => {
        #[test]
        fn $interp() {
            $body(Box::new(InterpreterBackend));
        }
        #[test]
        fn $jit() {
            $body(Box::new(JitBackend::new()));
        }
    };
}

#[test]
fn shared_fetch_rejects_trap_backing() {
    let vm = vm(Box::new(InterpreterBackend));
    vm.write_bytes(TRAP, &[0xf4]).unwrap();
    assert_eq!(vm.mem.code_slice(CODE, 0x2000).unwrap().len(), 0x1000);
    assert!(matches!(vm.mem.code_slice(TRAP, 15), Err(MemTrap::Mmio)));
    for lift in [lift_block, lift_one] {
        assert!(matches!(
            lift(&vm.mem, FetchAddr::flat(TRAP), CpuMode::Long64),
            Err(LiftError::DecodeFault { addr: TRAP })
        ));
    }
}

fn direct_entry(backend: Box<dyn Backend>) {
    let vm = vm(backend);
    // Check both zero-filled backing and a valid instruction loaded by the embedder.
    for bytes in [&[0, 0][..], &[0xf4][..]] {
        vm.write_bytes(TRAP + 0x80, bytes).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, TRAP + 0x80);
        cpu.set_reg(Reg::Rax, MARKER);
        assert_fetch_exit(&vm, &mut cpu, TRAP + 0x80);
        assert_eq!(cpu.reg(Reg::Rax), MARKER);
    }
}

both_backends!(direct_entry, direct_entry_interp, direct_entry_jit);

fn jump(backend: Box<dyn Backend>) {
    let vm = vm(backend);
    let code = assemble(CODE, |a| {
        a.mov(ebx, MARKER as i32).unwrap();
        a.jmp(TRAP).unwrap();
    });
    vm.write_bytes(CODE, &code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    assert_fetch_exit(&vm, &mut cpu, TRAP);
    assert_eq!(cpu.reg(Reg::Rbx), MARKER);
}

both_backends!(jump, jump_interp, jump_jit);

fn call(backend: Box<dyn Backend>) {
    let vm = vm(backend);
    let code = assemble(CODE, |a| {
        a.mov(ebx, MARKER as i32).unwrap();
        a.call(TRAP).unwrap();
    });
    vm.write_bytes(CODE, &code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rsp, STACK);
    assert_fetch_exit(&vm, &mut cpu, TRAP);
    assert_eq!(cpu.reg(Reg::Rbx), MARKER);
    // SDM Vol 1 §6.4.1, §6.2.5: near CALL pushes the return IP on the 64-bit stack.
    assert_eq!(cpu.reg(Reg::Rsp), STACK - 8);
    assert_eq!(vm.mem.read(STACK - 8, 8).unwrap(), CODE + code.len() as u64);
}

both_backends!(call, call_interp, call_jit);

fn fall_through(backend: Box<dyn Backend>) {
    let vm = vm(backend);
    let code = assemble(0, |a| {
        a.mov(ebx, MARKER as i32).unwrap();
        a.mov(qword_ptr(0x3000), rbx).unwrap();
    });
    let start = TRAP - code.len() as u64;
    vm.write_bytes(start, &code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, start);
    assert_fetch_exit(&vm, &mut cpu, TRAP);
    assert_eq!(cpu.reg(Reg::Rbx), MARKER);
    assert_eq!(vm.mem.read(0x3000, 8).unwrap(), MARKER);
}

both_backends!(fall_through, fall_through_interp, fall_through_jit);

fn straddling_instruction(backend: Box<dyn Backend>) {
    let vm = vm(backend);
    let prefix = assemble(0, |a| {
        a.mov(ebx, MARKER as i32).unwrap();
        a.mov(qword_ptr(0x3000), rbx).unwrap();
    });
    // MOV EAX, imm32 needs five bytes. Only its opcode and one immediate byte
    // are in RAM; fetching the remaining bytes from Trap backing is forbidden.
    let fault = TRAP - 2;
    let start = fault - prefix.len() as u64;
    vm.write_bytes(start, &prefix).unwrap();
    vm.write_bytes(fault, &[0xb8, 0x78]).unwrap();
    vm.write_bytes(TRAP, &[0x56, 0x34, 0x12, 0xf4]).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, start);
    cpu.set_reg(Reg::Rax, MARKER);
    assert_fetch_exit(&vm, &mut cpu, fault);
    assert_eq!(cpu.reg(Reg::Rbx), MARKER);
    assert_eq!(vm.mem.read(0x3000, 8).unwrap(), MARKER);
    // SDM Vol 3A §7.5: a fault preserves the state before the faulting instruction.
    assert_eq!(cpu.reg(Reg::Rax), MARKER);
    // Re-entry must report the same incomplete instruction without changing state.
    assert_fetch_exit(&vm, &mut cpu, fault);
    assert_eq!(cpu.reg(Reg::Rax), MARKER);
}

both_backends!(
    straddling_instruction,
    straddling_instruction_interp,
    straddling_instruction_jit
);
