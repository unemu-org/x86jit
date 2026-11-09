//! Segment-boundary witnesses without an external execution oracle.
//! Fetch faults: SDM Vol 3A §7.15 (Event 13); Vol 3B §23.1.4, Table 23-1.
//! Sequential IP: Vol 1 §3.5; Vol 3B §23.1. Operand faults: Vol 3B §25.33.1.

use x86jit_core::{
    Backend, CpuMode, Exit, FetchVerdict, InterpreterBackend, PortDir, Prot, Reg, RegionKind,
    StepOptions, StepResult, StepStop, Vcpu, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const CS: u64 = 0x1000;
const HANDLER_CS: u64 = 0x3000;
const SS: u64 = 0x4000;

fn setup(ip: u64) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x100000), Box::new(InterpreterBackend));
    vm.set_cpu_mode(CpuMode::Real16);
    for page in (0..0xa0000).step_by(4096) {
        vm.map(page, 4096, Prot::RWX, RegionKind::Ram).unwrap();
    }
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Cs, CS);
    cpu.set_reg(Reg::Rip, ip);
    cpu.set_reg(Reg::Ss, SS);
    cpu.set_reg(Reg::Rsp, 0x100);
    (vm, cpu)
}

fn halt(vm: &Vm, cpu: &mut Vcpu, budget: Option<u64>, retired: u64) {
    if let Some(limit) = budget {
        let mut steps = 0;
        for _ in 0..16 {
            let result = cpu.run_steps(vm, limit, StepOptions::default()).unwrap();
            assert!(result.steps() <= limit);
            steps += result.steps();
            match result.stop() {
                StepStop::Exit(Exit::Hlt) => {
                    assert_eq!(steps, retired);
                    assert_eq!(cpu.retired_instructions(), retired);
                    return;
                }
                StepStop::Exit(Exit::BudgetExhausted) => assert_eq!(result.steps(), limit),
                other => panic!("unexpected stop: {other:?}"),
            }
        }
        panic!("did not halt");
    }
    assert!(matches!(cpu.run(vm, Some(16)), Exit::Hlt));
    assert_eq!(cpu.retired_instructions(), retired);
}

fn check_wrap(budget: Option<u64>) {
    let (vm, mut cpu) = setup(0xfffe);
    vm.write_bytes(0x1fffe, &[0x90, 0x90]).unwrap(); // nop; nop
    vm.write_bytes(0x10000, &[0x42, 0xf4]).unwrap(); // inc dx; hlt
    vm.write_bytes(0x20000, &[0x43, 0xf4]).unwrap(); // inc bx; hlt
    halt(&vm, &mut cpu, budget, 4);
    assert_eq!(cpu.reg(Reg::Rdx), 1);
    assert_eq!(cpu.reg(Reg::Rbx), 0);
    assert_eq!(cpu.reg(Reg::Rip), 2);
}

#[test]
fn wrap_budget_1() {
    check_wrap(Some(1));
}

#[test]
fn wrap_budget_7() {
    check_wrap(Some(7));
}

#[test]
fn wrap_budget_64() {
    check_wrap(Some(64));
}

#[test]
fn wrap_run() {
    check_wrap(None);
}

#[test]
fn exact_end_instruction_wraps_in_block_and_single_step() {
    for budget in [Some(1), Some(7), Some(64), None] {
        let (vm, mut cpu) = setup(0xfffd);
        vm.write_bytes(0x1fffd, &[0xb8, 0x34, 0x12]).unwrap(); // mov ax,1234
        vm.write_bytes(0x10000, &[0x42, 0xf4]).unwrap();
        vm.write_bytes(0x20000, &[0x43, 0xf4]).unwrap();
        halt(&vm, &mut cpu, budget, 3);
        assert_eq!(cpu.reg(Reg::Rax), 0x1234);
        assert_eq!(cpu.reg(Reg::Rdx), 1);
        assert_eq!(cpu.reg(Reg::Rbx), 0);
        assert_eq!(cpu.reg(Reg::Rip), 2);
    }
    let (vm, mut cpu) = setup(0xfffd);
    vm.write_bytes(0x1fffd, &[0xb8, 0x34, 0x12]).unwrap();
    assert!(matches!(cpu.step_instruction(&vm), StepResult::Continue));
    assert_eq!(cpu.reg(Reg::Rip), 0);
    assert_eq!(cpu.reg(Reg::Rax), 0x1234);
    assert_eq!(cpu.retired_instructions(), 1);
}

#[derive(Clone, Copy, Debug)]
enum Execution {
    Run,
    Steps(u64),
    Single,
}

const EXECUTIONS: [Execution; 5] = [
    Execution::Run,
    Execution::Steps(1),
    Execution::Steps(7),
    Execution::Steps(64),
    Execution::Single,
];

fn execute(vm: &Vm, cpu: &mut Vcpu, execution: Execution) -> Exit {
    for _ in 0..32 {
        let exit = match execution {
            Execution::Run => Some(cpu.run(vm, Some(16))),
            Execution::Steps(budget) => {
                let result = cpu.run_steps(vm, budget, StepOptions::default()).unwrap();
                assert!(result.steps() <= budget);
                match result.stop() {
                    StepStop::Exit(Exit::BudgetExhausted) => {
                        assert_eq!(result.steps(), budget);
                        None
                    }
                    StepStop::Exit(exit) => Some(exit.clone()),
                    other => panic!("unexpected stop: {other:?}"),
                }
            }
            Execution::Single => match cpu.step_instruction(vm) {
                StepResult::Continue => None,
                StepResult::Exit(exit) => Some(exit),
            },
        };
        assert!(
            cpu.reg(Reg::Rip) <= 0xffff,
            "{execution:?}: IP must fit in 16 bits"
        );
        if let Some(exit) = exit {
            return exit;
        }
    }
    panic!("did not exit: {execution:?}");
}

fn boundary_exit(opcode: u8) {
    // SDM Vol 3B §23.1: IP is the 16-bit instruction pointer. HLT/IN/OUT
    // complete before returning control to the embedder (Vol 2A §3.3 HLT/IN;
    // Vol 2B §4.3 OUT).
    for execution in EXECUTIONS {
        let (vm, mut cpu) = setup(0xffff);
        vm.write_bytes(0x1ffff, &[opcode]).unwrap();
        cpu.set_reg(Reg::Rdx, 0x1234);
        cpu.set_reg(Reg::Rax, 0x56);
        let exit = execute(&vm, &mut cpu, execution);
        match opcode {
            0xf4 => assert!(matches!(exit, Exit::Hlt)),
            0xee => assert!(matches!(
                exit,
                Exit::PortIo {
                    port: 0x1234,
                    size: 1,
                    dir: PortDir::Out,
                    value: 0x56,
                    ..
                }
            )),
            0xec => {
                assert!(matches!(
                    exit,
                    Exit::PortIo {
                        port: 0x1234,
                        size: 1,
                        dir: PortDir::In,
                        ..
                    }
                ));
                cpu.complete_port_in(0xab);
                assert_eq!(cpu.reg(Reg::Rax), 0xab);
            }
            _ => unreachable!(),
        }
        assert_eq!(cpu.reg(Reg::Rip), 0);
        assert_eq!(cpu.retired_instructions(), 1);
    }
}

#[test]
fn hlt_at_ffff_wraps() {
    boundary_exit(0xf4);
}

#[test]
fn out_at_ffff_wraps() {
    boundary_exit(0xee);
}

#[test]
fn in_at_ffff_wraps() {
    boundary_exit(0xec);
}

#[test]
fn prefixed_instruction_ending_at_ffff_wraps() {
    for execution in EXECUTIONS {
        let (vm, mut cpu) = setup(0xfffc);
        vm.write_bytes(0x1fffc, &[0x26, 0xb8, 0x34, 0x12]).unwrap();
        vm.write_bytes(0x10000, &[0xf4]).unwrap();
        assert!(matches!(execute(&vm, &mut cpu, execution), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rax), 0x1234);
        assert_eq!(cpu.reg(Reg::Rip), 1);
        assert_eq!(cpu.retired_instructions(), 2);
    }
}

#[test]
fn near_branch_targets_and_fallthrough_wrap() {
    // SDM Vol 2A §3.3 JMP/Jcc; Vol 2B §4.3 LOOP: 16-bit near targets.
    for (bytes, taken, cx) in [
        (&[0xe9, 2, 0][..], true, 0), // jmp rel16
        (&[0x74, 2][..], true, 0),    // jz, taken
        (&[0x75, 2][..], false, 0),   // jnz, not taken
        (&[0xe2, 2][..], true, 2),    // loop, taken
        (&[0xe2, 2][..], false, 1),   // loop, not taken
    ] {
        for execution in EXECUTIONS {
            let (vm, mut cpu) = setup(0x10000 - bytes.len() as u64);
            vm.write_bytes(0x20000 - bytes.len() as u64, bytes).unwrap();
            vm.write_bytes(0x10000, &[0xf4, 0x90, 0xf4]).unwrap();
            cpu.set_flags(x86jit_core::Flags {
                zf: true,
                ..Default::default()
            }); // ZF=1
            cpu.set_reg(Reg::Rcx, cx);
            assert!(matches!(execute(&vm, &mut cpu, execution), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rip), if taken { 3 } else { 1 });
            assert_eq!(
                cpu.reg(Reg::Rcx),
                if bytes[0] == 0xe2 { cx - 1 } else { cx }
            );
            assert_eq!(cpu.retired_instructions(), 2);
        }
    }
}

#[test]
fn near_call_at_fffd_pushes_wrapped_return_ip_and_ret_uses_it() {
    // SDM Vol 2A §3.3 CALL; Vol 2B §4.3 RET: near call saves the next IP.
    for execution in EXECUTIONS {
        let (vm, mut cpu) = setup(0xfffd);
        vm.write_bytes(0x1fffd, &[0xe8, 0x10, 0]).unwrap(); // call 0010
        vm.write_bytes(0x10010, &[0x42, 0xc3]).unwrap(); // inc dx; ret
        vm.write_bytes(0x10000, &[0xf4]).unwrap();
        assert!(matches!(execute(&vm, &mut cpu, execution), Exit::Hlt));
        assert_eq!(vm.mem.read((SS << 4) + 0xfe, 2).unwrap(), 0);
        assert_eq!(cpu.reg(Reg::Rsp), 0x100);
        assert_eq!(cpu.reg(Reg::Rdx), 1);
        assert_eq!(cpu.reg(Reg::Rip), 1);
        assert_eq!(cpu.retired_instructions(), 4);
    }
}

#[test]
fn string_completion_wraps() {
    // SDM Vol 2B §4.3 MOVS/REP: a completed string instruction advances IP;
    // an incomplete REP iteration retains the instruction's restart IP.
    for rep in [false, true] {
        for execution in EXECUTIONS {
            let (vm, mut cpu) = setup(if rep { 0xfffe } else { 0xffff });
            if rep {
                vm.write_bytes(0x1fffe, &[0xf3, 0xa4]).unwrap();
            } else {
                vm.write_bytes(0x1ffff, &[0xa4]).unwrap();
            }
            vm.write_bytes(0x10000, &[0xf4]).unwrap();
            vm.write_bytes(0x50020, &[0x12, 0x34, 0x56]).unwrap();
            cpu.set_reg(Reg::Ds, 0x5000);
            cpu.set_reg(Reg::Es, 0x6000);
            cpu.set_reg(Reg::Rsi, 0x20);
            cpu.set_reg(Reg::Rdi, 0x40);
            cpu.set_reg(Reg::Rcx, 3);
            assert!(matches!(execute(&vm, &mut cpu, execution), Exit::Hlt));
            let count = if rep { 3 } else { 1 };
            assert_eq!(
                vm.mem.read(0x60040, count).unwrap(),
                if rep { 0x563412 } else { 0x12 }
            );
            assert_eq!(cpu.reg(Reg::Rsi), 0x20 + u64::from(count));
            assert_eq!(cpu.reg(Reg::Rdi), 0x40 + u64::from(count));
            assert_eq!(cpu.reg(Reg::Rcx), if rep { 0 } else { 3 });
            assert_eq!(cpu.reg(Reg::Rip), 1);
            assert_eq!(cpu.retired_instructions(), 2);
        }
    }
}

#[test]
fn fifteen_prefixes_raise_gp_with_restart_ip() {
    // SDM Vol 3A §7.15, Event 13: instructions exceeding 15 bytes raise #GP.
    // Vol 3B §23.1.4, Table 23-1: real-mode delivery saves the start CS:IP.
    for ip in [0x100, 0xfff1] {
        for execution in EXECUTIONS {
            let (mut vm, mut cpu) = setup(ip);
            vm.write_bytes((CS << 4) + ip, &[0x26; 15]).unwrap();
            vm.write_bytes((CS << 4) + ip + 15, &[0x90]).unwrap();
            install_gp(&mut vm);
            assert!(matches!(execute(&vm, &mut cpu, execution), Exit::Hlt));
            check_gp(&vm, &cpu, ip);
            assert_eq!(cpu.retired_instructions(), 1);
        }
    }
}

#[test]
fn invalid_opcode_at_segment_end_remains_ud() {
    // SDM Vol 3A §7.15, Event 6: LEA requires a memory source (Vol 2A §3.3
    // LEA). A complete invalid encoding still raises #UD.
    for prefixes in [0, 13] {
        for execution in EXECUTIONS {
            let ip = 0x10000 - prefixes - 2;
            let (vm, mut cpu) = setup(ip);
            let mut bytes = vec![0x26; prefixes as usize];
            bytes.extend_from_slice(&[0x8d, 0xc0]); // lea ax,ax
            vm.write_bytes((CS << 4) + ip, &bytes).unwrap();
            vm.write_bytes(6 * 4, &[0, 0, 0, 0x30]).unwrap();
            vm.write_bytes(HANDLER_CS << 4, &[0xf4]).unwrap();
            assert!(matches!(execute(&vm, &mut cpu, execution), Exit::Hlt));
            check_gp(&vm, &cpu, ip);
            assert_eq!(cpu.retired_instructions(), 1);
        }
    }
}

#[test]
fn trap_at_ffff_saves_wrapped_successor() {
    // SDM Vol 3A §7.15, Events 1 and 3: INT1/INT3 save the following IP.
    // Vol 3B §23.1.4: real-mode delivery pushes its low 16 bits through the IVT.
    for (opcode, vector) in [(0xf1, 1), (0xcc, 3)] {
        for execution in EXECUTIONS {
            let (vm, mut cpu) = setup(0xffff);
            vm.write_bytes(0x1ffff, &[opcode]).unwrap();
            vm.write_bytes(vector * 4, &[0, 0, 0, 0x30]).unwrap();
            vm.write_bytes(HANDLER_CS << 4, &[0xf4]).unwrap();
            assert!(matches!(execute(&vm, &mut cpu, execution), Exit::Hlt));
            check_gp(&vm, &cpu, 0);
            assert_eq!(cpu.retired_instructions(), 2);
        }
    }
}

#[test]
fn mode_change_at_segment_end_wraps() {
    for execution in EXECUTIONS {
        let (vm, mut cpu) = setup(0xfffb);
        vm.write_bytes(0x1fffb, &[0x0f, 0xae, 0x16, 0, 0]).unwrap(); // ldmxcsr [0]
        vm.write_bytes(0x50000, &0x3f80u32.to_le_bytes()).unwrap();
        vm.write_bytes(0x10000, &[0xf4]).unwrap();
        cpu.set_reg(Reg::Ds, 0x5000);
        assert!(matches!(execute(&vm, &mut cpu, execution), Exit::Hlt));
        assert_eq!(cpu.mxcsr(), 0x3f80);
        assert_eq!(cpu.reg(Reg::Rip), 1);
        assert_eq!(cpu.retired_instructions(), 2);
    }
}

#[test]
fn fault_at_segment_end_preserves_instruction_length() {
    for execution in EXECUTIONS {
        let (mut vm, mut cpu) = setup(0xfffd);
        vm.write_bytes(0x1fffd, &[0xa1, 0, 0]).unwrap(); // mov ax,[0]
        vm.unmap(0x50000, 4096).unwrap();
        cpu.set_reg(Reg::Ds, 0x5000);
        let exit = execute(&vm, &mut cpu, execution);
        assert!(matches!(exit, Exit::UnmappedMemory { addr: 0x50000, .. }));
        let metadata = exit.fault_metadata().unwrap();
        assert_eq!(metadata.instruction_len(), x86jit_core::Knowledge::Known(3));
        assert_eq!(
            metadata.instruction_rip(),
            x86jit_core::Knowledge::Known(0xfffd)
        );
        assert_eq!(cpu.reg(Reg::Rip), 0xfffd);
        assert_eq!(cpu.retired_instructions(), 0);
    }
}

fn install_gp(vm: &mut Vm) {
    vm.write_bytes(13 * 4, &[0, 0, 0, 0x30]).unwrap();
    vm.write_bytes(HANDLER_CS << 4, &[0xf4]).unwrap();
}

fn check_gp(vm: &Vm, cpu: &Vcpu, fault_ip: u64) {
    // SDM Vol 3A §7.15, Event 13: saved IP identifies the faulting instruction.
    // Vol 3B §23.1.4: real-mode delivery uses the IVT and pushes no error code.
    assert_eq!(cpu.reg(Reg::Cs), HANDLER_CS);
    assert_eq!(cpu.reg(Reg::Rip), 1);
    assert_eq!(cpu.reg(Reg::Rsp), 0xfa);
    assert_eq!(vm.mem.read((SS << 4) + 0xfa, 2).unwrap(), fault_ip);
    assert_eq!(vm.mem.read((SS << 4) + 0xfc, 2).unwrap(), CS);
}

#[test]
fn straddling_instruction_faults_through_ivt() {
    // MOV AX,imm16 needs bytes beyond CS:FFFF. The CS limit violation raises
    // #GP before MOV executes (SDM Vol 3A §7.15, Event 13; Vol 3B §23.1.4,
    // Table 23-1), rather than fetching either linear or wrapped immediate bytes.
    for budget in [Some(1), Some(7), Some(64), None] {
        for mapped_tail in [true, false] {
            let (mut vm, mut cpu) = setup(0xfffe);
            vm.write_bytes(0x1fffe, &[0x90, 0xb8]).unwrap(); // nop; mov ax,imm16
            vm.write_bytes(0x20000, &[0x34, 0x12, 0xf4]).unwrap();
            vm.write_bytes(0x10000, &[0x78, 0x56, 0xf4]).unwrap();
            if !mapped_tail {
                vm.unmap(0x20000, 4096).unwrap();
            }
            install_gp(&mut vm);
            vm.set_fetch_policy(Box::new(|page| {
                assert_ne!(page, 0x20000, "fetch must stop at the CS limit");
                FetchVerdict::Allow
            }));
            cpu.set_reg(Reg::Rax, 0xbeef);
            halt(&vm, &mut cpu, budget, 2); // NOP and handler HLT; MOV does not retire.
            check_gp(&vm, &cpu, 0xffff);
            assert_eq!(cpu.reg(Reg::Rax), 0xbeef);
        }
    }
    let (mut vm, mut cpu) = setup(0xffff);
    vm.write_bytes(0x1ffff, &[0xb8]).unwrap();
    install_gp(&mut vm);
    assert!(matches!(cpu.step_instruction(&vm), StepResult::Continue));
    assert_eq!(cpu.reg(Reg::Cs), HANDLER_CS);
    assert_eq!(cpu.reg(Reg::Rip), 0);
    assert_eq!(cpu.retired_instructions(), 0);
    assert_eq!(vm.mem.read((SS << 4) + 0xfa, 2).unwrap(), 0xffff);
}

#[test]
fn operand_straddles_fault_before_read_or_write_for_every_budget() {
    // SDM Vol 3B §25.33.1: a DS operand crossing FFFFH raises #GP.
    for opcode in [0xa1, 0xa3] {
        // mov ax,[ffff]; mov [ffff],ax
        for budget in [Some(1), Some(7), Some(64), None] {
            let (mut vm, mut cpu) = setup(0);
            vm.write_bytes(0x10000, &[0x90, opcode, 0xff, 0xff, 0xf4])
                .unwrap();
            vm.write_bytes(0x50000, &[0x78]).unwrap();
            vm.write_bytes(0x5ffff, &[0x34]).unwrap();
            vm.write_bytes(0x60000, &[0x12]).unwrap();
            install_gp(&mut vm);
            cpu.set_reg(Reg::Ds, 0x5000);
            cpu.set_reg(Reg::Rax, 0xbeef);
            halt(&vm, &mut cpu, budget, 2);
            check_gp(&vm, &cpu, 1);
            assert_eq!(cpu.reg(Reg::Rax), 0xbeef);
            assert_eq!(vm.mem.read(0x5ffff, 1).unwrap(), 0x34);
            assert_eq!(vm.mem.read(0x60000, 1).unwrap(), 0x12);
            assert_eq!(vm.mem.read(0x50000, 1).unwrap(), 0x78);
        }
    }
}

#[test]
fn flat_modes_keep_linear_fetch_in_interpreter_and_jit() {
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for backend in [
            Box::new(InterpreterBackend) as Box<dyn Backend>,
            Box::new(JitBackend::new()),
        ] {
            let mut vm = Vm::with_backend(VmConfig::flat(0x40000), backend);
            vm.set_cpu_mode(mode);
            vm.map(0, 0x40000, Prot::RWX, RegionKind::Ram).unwrap();
            // nop; nop; mov ebx,1; hlt -- the instruction stream stays linear.
            vm.write_bytes(0x1fffe, &[0x90, 0x90, 0xbb, 1, 0, 0, 0, 0xf4])
                .unwrap();
            vm.write_bytes(0x10000, &[0xba, 1, 0, 0, 0, 0xf4]).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, 0x1fffe);
            assert!(matches!(cpu.run(&vm, Some(16)), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rbx), 1);
            assert_eq!(cpu.reg(Reg::Rdx), 0);
            assert_eq!(cpu.reg(Reg::Rip), 0x20006);
        }
    }
}
