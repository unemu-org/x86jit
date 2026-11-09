//! Interpreter retirement at completion, restart and interrupt boundaries (§17.6).

use x86jit_core::{
    CpuMode, Exit, InterpreterBackend, Prot, Reg, RegionKind, StepResult, Vm, VmConfig,
};

const FLAT: u64 = 0x4_0000;

fn inject_vm(cs: u16, ss: u16, code: &[u8]) -> (Vm, x86jit_core::Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(FLAT), Box::new(InterpreterBackend));
    vm.set_cpu_mode(CpuMode::Real16);
    vm.map(0, FLAT as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.write_bytes((cs as u64) << 4, code).unwrap();
    let mut vcpu = vm.new_vcpu();
    vcpu.set_reg(Reg::Cs, cs as u64);
    vcpu.set_reg(Reg::Ss, ss as u64);
    vcpu.set_reg(Reg::Rip, 0);
    vcpu.set_reg(Reg::Rsp, 0x0100);
    (vm, vcpu)
}

/// Seed IVT[vector] → HCS:0 and write handler bytes at HCS:0.
fn seed_handler(vm: &mut Vm, vector: u8, hcs: u16, handler: &[u8]) {
    vm.write_bytes((hcs as u64) << 4, handler).unwrap();
    let mut e = 0u16.to_le_bytes().to_vec();
    e.extend_from_slice(&hcs.to_le_bytes());
    vm.write_bytes(vector as u64 * 4, &e).unwrap();
}

fn retired_vm(mode: CpuMode, code: &[u8]) -> (Vm, x86jit_core::Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x4000), Box::new(InterpreterBackend));
    vm.set_cpu_mode(mode);
    vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, code).unwrap();
    let mut vcpu = vm.new_vcpu();
    vcpu.set_reg(Reg::Rip, 0x1000);
    (vm, vcpu)
}

fn retired_self_jump(mode: CpuMode) {
    // SDM Vol 2A §3.3, JMP: a relative target is based on the next instruction.
    let (vm, mut vcpu) = retired_vm(mode, &[0xeb, 0xfe]); // jmp $
    let mut total = 0;
    for n in [1, 7, 31] {
        assert!(matches!(vcpu.run(&vm, Some(n)), Exit::BudgetExhausted));
        total += n;
        assert_eq!(vcpu.reg(Reg::Rip), 0x1000);
        assert_eq!(vcpu.retired_instructions(), total, "{mode:?}, N={n}");
    }
}

#[test]
fn retired_self_jump_real16() {
    retired_self_jump(CpuMode::Real16);
}

#[test]
fn retired_self_jump_compat32() {
    retired_self_jump(CpuMode::Compat32);
}

#[test]
fn retired_self_jump_long64() {
    retired_self_jump(CpuMode::Long64);
}

#[test]
fn retired_conditional_self_loop_and_not_taken() {
    for mode in [CpuMode::Real16, CpuMode::Compat32, CpuMode::Long64] {
        // SDM Vol 2A §3.3, DEC/Jcc: DEC sets ZF; JNZ branches when ZF=0.
        let mut code = vec![];
        if mode != CpuMode::Real16 {
            code.push(0x66);
        }
        code.extend_from_slice(&[0xff, 0xc9, 0x75, 0xfe, 0xf4]); // dec cx; jnz $; hlt
        let branch = 0x1000 + code.len() as u64 - 3;
        let (vm, mut vcpu) = retired_vm(mode, &code);
        vcpu.set_reg(Reg::Rcx, 2);
        assert!(matches!(vcpu.run(&vm, Some(1)), Exit::BudgetExhausted));
        assert_eq!(vcpu.reg(Reg::Rip), branch);
        assert_eq!(vcpu.reg(Reg::Rcx), 1);
        assert_eq!(vcpu.retired_instructions(), 2, "{mode:?}: dec + taken jnz");
        assert!(matches!(vcpu.run(&vm, Some(3)), Exit::BudgetExhausted));
        assert_eq!(vcpu.reg(Reg::Rip), branch);
        assert_eq!(vcpu.retired_instructions(), 5, "{mode:?}: three taken jnz");

        // Re-enter the DEC with CX=1 so the same branch falls through.
        vcpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(vcpu.run(&vm, Some(1)), Exit::BudgetExhausted));
        assert_eq!(vcpu.reg(Reg::Rip), branch + 2);
        assert_eq!(vcpu.reg(Reg::Rcx), 0);
        assert_eq!(
            vcpu.retired_instructions(),
            7,
            "{mode:?}: dec + untaken jnz"
        );
        assert!(matches!(vcpu.run(&vm, Some(1)), Exit::Hlt));
        assert_eq!(vcpu.retired_instructions(), 8, "{mode:?}: hlt");
    }
}

#[test]
fn retired_faulting_load_counts_once_on_retry() {
    for mode in [CpuMode::Real16, CpuMode::Compat32, CpuMode::Long64] {
        // SDM Vol 3A §7.5: a fault saves the faulting instruction for restart.
        // NOP; MOV AL,[BX/EBX/RBX]; HLT, with a completed prefix before the fault.
        let code = if mode == CpuMode::Real16 {
            [0x90, 0x8a, 0x07, 0xf4]
        } else {
            [0x90, 0x8a, 0x03, 0xf4]
        };
        let (mut vm, mut vcpu) = retired_vm(mode, &code);
        vcpu.set_reg(Reg::Rbx, 0x2000);
        for _ in 0..2 {
            let exit = vcpu.run(&vm, Some(8));
            assert!(
                matches!(exit, Exit::UnmappedMemory { addr: 0x2000, .. }),
                "{mode:?}: {exit:?}"
            );
            assert_eq!(vcpu.reg(Reg::Rip), 0x1001);
            assert_eq!(vcpu.retired_instructions(), 1, "{mode:?}: only nop retired");
        }
        vm.map(0x2000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
        vm.write_bytes(0x2000, &[0x42]).unwrap();
        assert!(matches!(vcpu.run(&vm, Some(8)), Exit::Hlt));
        assert_eq!(vcpu.reg(Reg::Rax), 0x42);
        assert_eq!(vcpu.retired_instructions(), 3, "{mode:?}: nop + load + hlt");
    }
}

#[test]
fn retired_sti_self_jump_allows_pending_irq() {
    // SDM Vol 3A §7.8.1: with IF initially clear, STI blocks maskable interrupts
    // through the following instruction, even when that instruction jumps to itself.
    let (mut vm, mut vcpu) = inject_vm(0x100, 0x300, &[0xfb, 0xeb, 0xfe]);
    seed_handler(&mut vm, 0x40, 0x200, &[0xf4]);
    vcpu.inject_irq(0x40);

    assert!(matches!(vcpu.run(&vm, Some(1)), Exit::BudgetExhausted));
    assert_eq!(vcpu.reg(Reg::Cs), 0x100);
    assert_eq!(vcpu.reg(Reg::Rip), 1);
    assert!(vcpu.has_pending_irq());
    assert_eq!(vcpu.retired_instructions(), 2, "sti + jmp retired");

    assert!(matches!(vcpu.run(&vm, Some(1)), Exit::BudgetExhausted));
    assert_eq!(vcpu.reg(Reg::Cs), 0x200);
    assert_eq!(vcpu.reg(Reg::Rip), 0);
    assert!(!vcpu.has_pending_irq());
    assert_eq!(
        vcpu.retired_instructions(),
        2,
        "IRQ delivery does not retire an instruction"
    );
    let mut frame = [0; 6];
    vm.read_bytes(0x3000 + 0xfa, &mut frame).unwrap();
    assert_eq!(u16::from_le_bytes([frame[0], frame[1]]), 1, "return to jmp");
    assert_eq!(u16::from_le_bytes([frame[2], frame[3]]), 0x100);
    assert_ne!(u16::from_le_bytes([frame[4], frame[5]]) & (1 << 9), 0);
    assert!(!vcpu.flags().if_);
    assert!(matches!(vcpu.run(&vm, Some(1)), Exit::Hlt));
    assert_eq!(vcpu.retired_instructions(), 3, "handler hlt retired");
}

fn step_retired(vm: &Vm, cpu: &mut x86jit_core::Vcpu, total: u64) {
    assert!(matches!(cpu.step_instruction(vm), StepResult::Continue));
    assert_eq!(cpu.retired_instructions(), total);
}

// SDM Vol 3A §7.5: fault delivery leaves the instruction incomplete; IRET
// restarts it. The handler repairs the cause and returns to the same CS:IP.
fn fault_retry(code: &[u8], vector: u8, fix: &[u8], ax: u64, bx: u64) {
    for handler_ip in [0u16, 0x80] {
        let (vm, mut cpu) = inject_vm(0x100, 0x300, code);
        cpu.set_reg(Reg::Es, 0x100);
        cpu.set_reg(Reg::Rax, ax);
        cpu.set_reg(Reg::Rbx, bx);
        vm.write_bytes(0x4000, &[0, 0, 1, 0]).unwrap();
        let mut handler = fix.to_vec();
        handler.push(0xcf); // iret
        vm.write_bytes(0x2000 + u64::from(handler_ip), &handler)
            .unwrap();
        let mut entry = handler_ip.to_le_bytes().to_vec();
        entry.extend_from_slice(&0x200u16.to_le_bytes());
        vm.write_bytes(u64::from(vector) * 4, &entry).unwrap();

        step_retired(&vm, &mut cpu, 0);
        assert_eq!(cpu.reg(Reg::Cs), 0x200);
        assert_eq!(cpu.reg(Reg::Rip), u64::from(handler_ip));
        let mut saved_ip = [0; 2];
        vm.read_bytes(0x30fa, &mut saved_ip).unwrap();
        assert_eq!(saved_ip, [0, 0]);
        step_retired(&vm, &mut cpu, 1); // repair instruction
        step_retired(&vm, &mut cpu, 2); // iret
        assert_eq!(cpu.reg(Reg::Cs), 0x100);
        assert_eq!(cpu.reg(Reg::Rip), 0);
        step_retired(&vm, &mut cpu, 3); // successful retry, counted exactly once
        assert_eq!(cpu.reg(Reg::Rip), code.len() as u64 - 1);
        assert!(matches!(
            cpu.step_instruction(&vm),
            StepResult::Exit(Exit::Hlt)
        ));
        assert_eq!(cpu.retired_instructions(), 4);
    }
}

#[test]
fn retired_invalid_opcode_fault_retries_once() {
    // Invalid BOUND register encoding, then HLT. Handler: mov es:[0],00ebh,
    // replacing it with JMP +0 (one completed instruction on retry).
    fault_retry(
        &[0x62, 0xc0, 0xf4],
        6,
        &[0x26, 0xc7, 0x06, 0, 0, 0xeb, 0],
        0,
        0,
    );
}

#[test]
fn retired_ud2_fault_retries_once() {
    fault_retry(
        &[0x0f, 0x0b, 0xf4],
        6,
        &[0x26, 0xc7, 0x06, 0, 0, 0xeb, 0],
        0,
        0,
    );
}

#[test]
fn retired_instruction_length_fault_retries_once() {
    let mut code = vec![0x26; 10];
    code.extend_from_slice(&[0x90, 0xf4]);
    // Replace the overlong instruction with JMP +9, reaching the original HLT.
    fault_retry(&code, 13, &[0x26, 0xc7, 0x06, 0, 0, 0xeb, 9], 0, 0);
}

#[test]
fn retired_bound_fault_retries_once() {
    // BOUND AX,[BX], bounds [0,1], AX=2; handler XOR AX,AX.
    fault_retry(&[0x62, 0x07, 0xf4], 5, &[0x31, 0xc0], 2, 0x4000);
}

#[test]
fn retired_segment_limit_fault_retries_once() {
    // MOV AX,[BX] crosses the data segment limit; handler XOR BX,BX.
    fault_retry(&[0x8b, 0x07, 0xf4], 13, &[0x31, 0xdb], 0, 0xffff);
}

#[test]
fn retired_ss_relative_segment_limit_fault_retries_once() {
    fault_retry(&[0x36, 0x8b, 0x07, 0xf4], 13, &[0x31, 0xdb], 0, 0xffff);
}

#[test]
fn retired_fault_delivery_keeps_completed_prefix() {
    let (mut vm, mut cpu) = inject_vm(0x100, 0x300, &[0x90, 0x62, 0x07, 0xf4]);
    seed_handler(&mut vm, 5, 0x200, &[0x31, 0xc0, 0xcf]);
    cpu.set_reg(Reg::Rax, 2);
    cpu.set_reg(Reg::Rbx, 0x4000);
    vm.write_bytes(0x4000, &[0, 0, 1, 0]).unwrap();
    assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
    assert_eq!(cpu.reg(Reg::Cs), 0x200);
    assert_eq!(cpu.retired_instructions(), 1, "only NOP completed");
}

#[test]
fn retired_software_interrupts_and_iret() {
    // SDM Vol 3A §7.5: traps complete the instruction and save its next IP.
    // INT n remains a completed software interrupt even for a fault's vector.
    for (code, vector) in [
        (&[0xcd, 0x40][..], 0x40),
        (&[0xcd, 6][..], 6),
        (&[0xcd, 13][..], 13),
        (&[0xcc][..], 3),
        (&[0xce][..], 4),
    ] {
        let (mut vm, mut cpu) = inject_vm(0x100, 0x300, code);
        seed_handler(&mut vm, vector, 0x200, &[0xcf]);
        let mut flags = cpu.flags();
        flags.of = true;
        cpu.set_flags(flags);
        step_retired(&vm, &mut cpu, 1);
        assert_eq!(cpu.reg(Reg::Cs), 0x200);
        assert_eq!(cpu.reg(Reg::Rip), 0, "handler has same numeric IP as INT");
        step_retired(&vm, &mut cpu, 2);
        assert_eq!(cpu.reg(Reg::Cs), 0x100);
        assert_eq!(cpu.reg(Reg::Rip), code.len() as u64);
        assert_eq!(cpu.reg(Reg::Rsp), 0x100);
    }
}

#[test]
fn retired_iret_to_same_numeric_ip() {
    let (vm, mut cpu) = inject_vm(0x200, 0x300, &[0xcf]);
    vm.write_bytes(0x3100, &[0, 0, 0, 1, 2, 0]).unwrap();
    step_retired(&vm, &mut cpu, 1);
    assert_eq!(cpu.reg(Reg::Cs), 0x100);
    assert_eq!(cpu.reg(Reg::Rip), 0);
}

#[test]
fn retired_into_not_taken() {
    let (vm, mut cpu) = inject_vm(0x100, 0x300, &[0xce]);
    step_retired(&vm, &mut cpu, 1);
    assert_eq!(cpu.reg(Reg::Cs), 0x100);
    assert_eq!(cpu.reg(Reg::Rip), 1);
}

#[test]
fn retired_mode_change() {
    let (mut vm, mut cpu) = retired_vm(CpuMode::Long64, &[0x0f, 0xae, 0x13]); // ldmxcsr [rbx]
    vm.map(0x2000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
    vm.write_bytes(0x2000, &0x3f80u32.to_le_bytes()).unwrap();
    cpu.set_reg(Reg::Rbx, 0x2000);
    step_retired(&vm, &mut cpu, 1);
    assert_eq!(cpu.mxcsr(), 0x3f80);
    assert_eq!(cpu.reg(Reg::Rip), 0x1003);
}

#[test]
fn retired_syscall_and_advancing_trap() {
    for (code, vector) in [(&[0x0f, 0x05][..], None), (&[0xcc][..], Some(3))] {
        let (vm, mut cpu) = retired_vm(CpuMode::Long64, code);
        match (cpu.step_instruction(&vm), vector) {
            (StepResult::Exit(Exit::Syscall), None) => {}
            (StepResult::Exit(Exit::Exception { vector: actual, .. }), Some(expected)) => {
                assert_eq!(actual, expected);
            }
            _ => panic!("unexpected syscall/trap result"),
        }
        assert_eq!(cpu.reg(Reg::Rip), 0x1000 + code.len() as u64);
        assert_eq!(cpu.retired_instructions(), 1);
    }
}

#[test]
fn retired_port_io_completion() {
    let (vm, mut cpu) = retired_vm(CpuMode::Long64, &[0xe4, 0x80, 0xe6, 0x80, 0xf4]);
    assert!(matches!(
        cpu.step_instruction(&vm),
        StepResult::Exit(Exit::PortIo { .. })
    ));
    assert_eq!(cpu.retired_instructions(), 1);
    cpu.complete_port_in(0x42);
    assert_eq!(cpu.retired_instructions(), 1);
    assert_eq!(cpu.reg(Reg::Rax), 0x42);
    assert!(matches!(
        cpu.step_instruction(&vm),
        StepResult::Exit(Exit::PortIo { .. })
    ));
    assert_eq!(cpu.retired_instructions(), 2);
    assert!(matches!(
        cpu.step_instruction(&vm),
        StepResult::Exit(Exit::Hlt)
    ));
    assert_eq!(cpu.retired_instructions(), 3);
}

#[test]
fn retired_mmio_replay() {
    // Read then write the same MMIO byte; neither initial attempt retires.
    let (mut vm, mut cpu) = retired_vm(CpuMode::Long64, &[0x8a, 0x03, 0x88, 0x03, 0xf4]);
    vm.map(0x2000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    cpu.set_reg(Reg::Rbx, 0x2000);
    assert!(matches!(cpu.run(&vm, Some(1)), Exit::MmioRead { .. }));
    assert_eq!(cpu.retired_instructions(), 0);
    cpu.complete_mmio_read(0x42);
    assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
    assert_eq!(cpu.retired_instructions(), 1);
    assert!(matches!(
        cpu.run(&vm, Some(1)),
        Exit::MmioWrite { value: 0x42, .. }
    ));
    assert_eq!(cpu.retired_instructions(), 1);
    cpu.complete_mmio_write();
    assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
    assert_eq!(cpu.retired_instructions(), 2);
    assert!(matches!(cpu.run(&vm, Some(1)), Exit::Hlt));
    assert_eq!(cpu.retired_instructions(), 3);
}

#[test]
fn retired_interrupted_rep_completion() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use x86jit_core::interp::{step_one, RetireInfo};

    // SDM Vol 2B §4.3, REP, Description: an interrupted repeat resumes with
    // the remaining count and next element pointers, at the same instruction.
    let (mut vm, mut cpu) = retired_vm(CpuMode::Long64, &[0xf3, 0xaa]); // rep stosb
    vm.map(0x2000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
    cpu.set_reg(Reg::Rdi, 0x2000);
    cpu.set_reg(Reg::Rcx, 4096);
    cpu.set_reg(Reg::Rax, 0x42);
    let request = AtomicU64::new(u64::MAX);
    let mut info = RetireInfo {
        exit_request: Some(&request),
        ..Default::default()
    };
    let mut scratch = Vec::new();
    assert!(matches!(
        step_one(
            &vm.mem,
            &mut cpu.cpu,
            CpuMode::Long64,
            &mut scratch,
            &mut info
        ),
        StepResult::Exit(Exit::Interrupted)
    ));
    assert_eq!(info.retired, 0);
    assert_eq!(cpu.reg(Reg::Rip), 0x1000);
    assert_ne!(cpu.reg(Reg::Rcx), 0);
    request.store(0, Ordering::Relaxed);
    assert!(matches!(
        step_one(
            &vm.mem,
            &mut cpu.cpu,
            CpuMode::Long64,
            &mut scratch,
            &mut info
        ),
        StepResult::Continue
    ));
    assert_eq!(info.retired, 1);
    assert_eq!(cpu.reg(Reg::Rcx), 0);
    assert_eq!(cpu.reg(Reg::Rdi), 0x3000);
    assert_eq!(cpu.reg(Reg::Rip), 0x1002);
    assert_eq!(vm.mem.read(0x2fff, 1).unwrap(), 0x42);
}
