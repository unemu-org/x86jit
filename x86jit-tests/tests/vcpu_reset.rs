//! In-place reset witnesses for engine state, independent of a CPU-model reset table.

use x86jit_core::{
    Backend, CpuMode, Exit, InterpreterBackend, PortDir, Prot, Reg, RegionKind, ResetOptions,
    StepOptions, StepStop, Vcpu, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

fn setup(code: &[u8], mode: CpuMode, jit: bool) -> (Vm, Vcpu) {
    let backend: Box<dyn Backend> = if jit {
        Box::new(JitBackend::new())
    } else {
        Box::new(InterpreterBackend)
    };
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
    vm.set_cpu_mode(mode);
    vm.map(0, 0x6000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x7000, 0x2000, Prot::RW, RegionKind::Trap).unwrap();
    vm.write_bytes(0x1000, code).unwrap();
    vm.write_bytes(0x2020, &[0xff, 0xc3, 0xf4]).unwrap(); // inc bx/ebx; hlt
    let mut cpu = vm.new_vcpu_with_source_id(41).unwrap();
    reset_entry(&mut cpu, mode);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt)); // warm the reset translation
    cpu.cpu.cs = if mode == CpuMode::Real16 { 0x100 } else { 0 };
    cpu.cpu.rip = if mode == CpuMode::Real16 { 0 } else { 0x1000 };
    cpu.cpu.ss = 0;
    cpu.cpu.gpr[Reg::Rsp.gpr_index().unwrap()] = 0x5000;
    cpu.cpu.gpr[Reg::Rbx.gpr_index().unwrap()] = 0x10;
    (vm, cpu)
}

fn reset_entry(cpu: &mut Vcpu, mode: CpuMode) {
    // Direct state writes ensure set_reg's retry abandonment cannot hide a stale latch.
    cpu.cpu.cs = if mode == CpuMode::Real16 { 0x200 } else { 0x23 };
    cpu.cpu.rip = if mode == CpuMode::Real16 {
        0x20
    } else {
        0x2020
    };
}

fn reset_and_resume(vm: &Vm, cpu: &mut Vcpu, mode: CpuMode) {
    let retired = cpu.retired_instructions();
    let executed = cpu.executed_instructions();
    let source_id = cpu.cpu.source_id;
    let next_id = vm.next_source_id();
    let misses = vm.cache.misses();
    let epoch = vm.cache.epoch();
    let registers = cpu.cpu.gpr;
    let old_cs_ip = (cpu.cpu.cs, cpu.cpu.rip);
    cpu.reset_execution_state(ResetOptions::default());
    assert_eq!(cpu.cpu.gpr, registers);
    assert_eq!((cpu.cpu.cs, cpu.cpu.rip), old_cs_ip);
    assert_eq!(cpu.retired_instructions(), retired);
    assert_eq!(cpu.executed_instructions(), executed);
    assert_eq!(cpu.cpu.source_id, source_id);
    assert_eq!(vm.next_source_id(), next_id);
    assert!(!cpu.has_pending_irq());
    assert!(cpu.interruption_evidence().is_none());
    assert!(cpu.syscall_evidence().is_none());
    assert!(cpu.cpu.pending_port_in.is_none());
    assert!(cpu.cpu.port_string_in.is_none());
    assert!(cpu.cpu.pending_mmio.is_none());
    assert!(!cpu.cpu.pending_mmio_write);
    assert!(!cpu.cpu.mmio_retry_active);
    assert_eq!(cpu.cpu.mmio_parts_len, 0);
    assert_eq!(cpu.cpu.mmio_parts_rip, 0);
    assert_eq!(cpu.cpu.mmio_retry_cs, 0);
    assert!(cpu.cpu.mmio_parts.iter().all(|p| *p == (0, 0)));
    cpu.cpu.flags.if_ = true;
    if mode == CpuMode::Real16 {
        assert!(cpu.irq_ready()); // neither interrupt shadow survived
    }
    reset_entry(cpu, mode);
    if mode == CpuMode::Real16 {
        let result = cpu.run_steps(vm, 10, StepOptions::default()).unwrap();
        assert_eq!(result.steps(), 2);
        assert!(matches!(result.stop(), StepStop::Exit(Exit::Hlt)));
        assert_eq!(cpu.retired_instructions(), retired + 2);
    } else {
        assert!(matches!(cpu.run(vm, None), Exit::Hlt));
    }
    assert_eq!(cpu.reg(Reg::Rbx), 0x11);
    assert_eq!(
        cpu.cpu.cs,
        if mode == CpuMode::Real16 { 0x200 } else { 0x23 }
    );
    assert_eq!(
        cpu.cpu.rip,
        if mode == CpuMode::Real16 {
            0x23
        } else {
            0x2023
        }
    );
    assert_eq!(vm.cache.epoch(), epoch);
    assert_eq!(vm.cache.misses(), misses); // unchanged code needs no re-lift or compile
}

#[test]
fn real16_discards_queued_irq_and_both_interrupt_shadows() {
    // SDM Vol 3A §§7.8.1, 7.8.3: STI and MOV SS inhibit the successor.
    for load in [&[0xfb][..], &[0x8e, 0xd0][..]] {
        let (vm, mut cpu) = setup(load, CpuMode::Real16, false);
        cpu.cpu.gpr[0] = 0;
        cpu.cpu.flags.if_ = load[0] != 0xfb;
        let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(r.steps(), 1);
        assert!(matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)));
        assert!(!cpu.irq_ready());
        cpu.inject_irq(0x40);
        assert!(cpu.has_pending_irq());
        reset_and_resume(&vm, &mut cpu, CpuMode::Real16);
    }
}

#[test]
fn pending_in_cannot_complete_into_the_reset_accumulator() {
    for mode in [CpuMode::Real16, CpuMode::Compat32, CpuMode::Long64] {
        for jit in [false, true] {
            let (vm, mut cpu) = setup(&[0xe4, 0x80], mode, jit);
            assert!(matches!(
                cpu.run(&vm, None),
                Exit::PortIo {
                    dir: PortDir::In,
                    ..
                }
            ));
            assert!(cpu.cpu.pending_port_in.is_some());
            reset_and_resume(&vm, &mut cpu, mode);
            cpu.cpu.gpr[0] = 0x1234;
            cpu.complete_port_in(0x55);
            assert_eq!(cpu.cpu.gpr[0], 0x1234);
        }
    }
}

#[test]
fn pending_ins_and_answered_faulting_ins_store_are_abandoned() {
    for answered in [false, true] {
        let (vm, mut cpu) = setup(&[0xf3, 0x6c], CpuMode::Real16, false);
        cpu.cpu.gpr[Reg::Rdi.gpr_index().unwrap()] = 0x7000;
        cpu.cpu.gpr[Reg::Rcx.gpr_index().unwrap()] = 2;
        assert!(matches!(
            cpu.run(&vm, None),
            Exit::PortIo {
                dir: PortDir::In,
                ..
            }
        ));
        assert!(cpu.cpu.port_string_in.is_some());
        if answered {
            cpu.complete_port_in(0x55);
            assert!(matches!(
                cpu.run(&vm, None),
                Exit::MmioWrite { addr: 0x7000, .. }
            ));
            assert_eq!(cpu.cpu.port_string_in.unwrap().value, Some(0x55));
            cpu.complete_mmio_write();
        }
        let indices = (cpu.reg(Reg::Rdi), cpu.reg(Reg::Rcx));
        reset_and_resume(&vm, &mut cpu, CpuMode::Real16);
        assert_eq!((cpu.reg(Reg::Rdi), cpu.reg(Reg::Rcx)), indices);
    }
}

#[test]
fn pending_mmio_reads_writes_and_completions_are_abandoned() {
    for write in [false, true] {
        for answered in [false, true] {
            let (vm, mut cpu) = setup(
                &[if write { 0xa2 } else { 0xa0 }, 0, 0x70],
                CpuMode::Real16,
                false,
            );
            let exit = cpu.run(&vm, None);
            assert!(if write {
                matches!(exit, Exit::MmioWrite { addr: 0x7000, .. })
            } else {
                matches!(exit, Exit::MmioRead { addr: 0x7000, .. })
            });
            assert!(cpu.cpu.mmio_retry_active);
            if answered {
                if write {
                    cpu.complete_mmio_write();
                } else {
                    cpu.complete_mmio_read(0x55);
                }
            }
            reset_and_resume(&vm, &mut cpu, CpuMode::Real16);
            // A fresh access must still trap: the old answer/ack cannot service it.
            cpu.cpu.cs = 0x100;
            cpu.cpu.rip = 0;
            assert!(matches!(
                cpu.run(&vm, None),
                Exit::MmioRead { .. } | Exit::MmioWrite { .. }
            ));
        }
    }
}

#[test]
fn partially_answered_mmio_instruction_loses_its_retry_journal_and_write_ack() {
    let (vm, mut cpu) = setup(&[0xa4], CpuMode::Real16, false); // movsb
    cpu.cpu.gpr[Reg::Rsi.gpr_index().unwrap()] = 0x7000;
    cpu.cpu.gpr[Reg::Rdi.gpr_index().unwrap()] = 0x8000;
    let result = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
    assert_eq!(result.steps(), 0);
    assert!(matches!(
        result.stop(),
        StepStop::Exit(Exit::MmioRead { addr: 0x7000, .. })
    ));
    cpu.complete_mmio_read(0x55);
    let result = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
    assert_eq!(result.steps(), 0);
    assert!(matches!(
        result.stop(),
        StepStop::Exit(Exit::MmioWrite {
            addr: 0x8000,
            value: 0x55,
            ..
        })
    ));
    assert!(cpu.cpu.mmio_parts_len > 0);
    cpu.complete_mmio_write();
    let indices = (cpu.reg(Reg::Rsi), cpu.reg(Reg::Rdi));
    reset_and_resume(&vm, &mut cpu, CpuMode::Real16);
    assert_eq!((cpu.reg(Reg::Rsi), cpu.reg(Reg::Rdi)), indices);
}

#[test]
fn compiled_indirect_branch_and_call_predictions_can_be_reset_without_recompiling() {
    // SDM Vol 2A §3.2 CALL and Vol 2B §4.3 JMP/RET: indirect targets come
    // from the operand; a near RET pops its target from the architectural stack.
    for tier in 0..3 {
        let backend: Box<dyn Backend> = match tier {
            0 => Box::new(InterpreterBackend),
            1 => Box::new(JitBackend::new()),
            _ => Box::new(JitBackend::with_superblocks(x86jit_core::RegionCaps {
                max_blocks: 8,
                max_icount: 128,
            })),
        };
        let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
        vm.map(0, 0x6000, Prot::RWX, RegionKind::Ram).unwrap();
        let mut code = vec![0x90; 0x33];
        code[..2].copy_from_slice(&[0xeb, 0x02]); // jmp 0x1004 (region edge)
        code[4..8].copy_from_slice(&[0xff, 0xc9, 0x75, 0xf8]); // dec ecx; jnz 0x1000
        code[8..10].copy_from_slice(&[0xff, 0xe0]); // jmp rax
        code[0x10..0x16].copy_from_slice(&[0xe8, 0x0b, 0, 0, 0, 0xf4]); // call 0x1020; hlt
        code[0x20..0x23].copy_from_slice(&[0x0f, 0x05, 0xc3]); // syscall; ret
        code[0x30..].copy_from_slice(&[0xff, 0xc3, 0xf4]); // inc ebx; hlt
        vm.write_bytes(0x1000, &code).unwrap();
        let mut cpu = vm.new_vcpu();
        // Warm both possible destinations before reset, without changing code later.
        for target in [0x1030, 0x1015] {
            cpu.set_reg(Reg::Rip, target);
            assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
        }
        vm.write_bytes(0x5000, &0x1030u64.to_le_bytes()).unwrap();
        cpu.set_reg(Reg::Rsp, 0x5000);
        cpu.set_reg(Reg::Rip, 0x1022);
        assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
        cpu.set_reg(Reg::Rcx, 2);
        cpu.set_reg(Reg::Rip, 0x1000);
        cpu.set_reg(Reg::Rax, 0x1010);
        cpu.set_reg(Reg::Rsp, 0x5000);
        assert!(matches!(cpu.run(&vm, Some(100)), Exit::Syscall));
        assert_eq!(cpu.reg(Reg::Rsp), 0x4ff8);
        if tier != 0 {
            assert!(vm.cache.ibtc_filled() > 0);
        }
        if tier == 2 {
            assert!(vm.cache.regions() > 0);
        }
        let misses = vm.cache.misses();
        let epoch = vm.cache.epoch();
        cpu.reset_execution_state(ResetOptions::default());
        cpu.set_reg(Reg::Rbx, 0);
        cpu.set_reg(Reg::Rax, 0x1030);
        cpu.set_reg(Reg::Rcx, 2);
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rip), 0x1033);
        assert_eq!(cpu.reg(Reg::Rbx), 1);
        // The outstanding CALL predicted 0x1015. RET must use the new stack value.
        vm.write_bytes(0x4ff8, &0x1030u64.to_le_bytes()).unwrap();
        cpu.set_reg(Reg::Rip, 0x1022);
        assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rip), 0x1033);
        assert_eq!(cpu.reg(Reg::Rbx), 2);
        assert_eq!(cpu.reg(Reg::Rsp), 0x5000);
        assert_eq!(vm.cache.epoch(), epoch);
        assert_eq!(vm.cache.misses(), misses);
    }
}

#[test]
fn syscall_interruption_and_pending_host_requests_are_cleared() {
    for jit in [false, true] {
        let (vm, mut cpu) = setup(&[0x0f, 0x05], CpuMode::Long64, jit);
        assert!(matches!(cpu.run(&vm, None), Exit::Syscall));
        assert!(cpu.syscall_evidence().is_some());
        reset_and_resume(&vm, &mut cpu, CpuMode::Long64);
        let request = cpu.exit_request();
        request.request_exit();
        assert!(matches!(cpu.run(&vm, None), Exit::Interrupted));
        assert!(cpu.interruption_evidence().is_some());
        request.request_exit(); // an unconsumed request must also be discarded
        cpu.cpu.gpr[Reg::Rbx.gpr_index().unwrap()] = 0x10;
        reset_and_resume(&vm, &mut cpu, CpuMode::Long64);
        request.request_exit(); // the existing handle still targets this vCPU
        assert!(matches!(cpu.run(&vm, None), Exit::Interrupted));
    }
}

#[test]
fn architectural_state_and_source_configuration_are_preserved_by_default() {
    for mode in [CpuMode::Real16, CpuMode::Compat32, CpuMode::Long64] {
        let (vm, mut cpu) = setup(&[], mode, false);
        cpu.cpu.gpr = [0x1234; 16];
        cpu.cpu.flags.set_flags16(0xcd7);
        cpu.cpu.rflags_system = 0x40000;
        cpu.cpu.xmm = [u128::MAX; 32];
        cpu.cpu.ymm_hi = [0x1234; 32];
        cpu.cpu.zmm_hi = [[0x5678; 2]; 32];
        cpu.cpu.kmask = [0x4321; 8];
        cpu.cpu.ds = 0x12;
        cpu.cpu.es = 0x34;
        cpu.cpu.ss = 0x56;
        cpu.cpu.fs = 0x78;
        cpu.cpu.gs = 0x9a;
        cpu.cpu.fs_base = 0x9876;
        cpu.cpu.gs_base = 0x5432;
        cpu.cpu.mxcsr = 0x3f80;
        cpu.cpu.random_state = 0xabcdef;
        dirty_x87(&mut cpu);
        let before = format!("{:?}", cpu.cpu);
        let next = vm.next_source_id();
        cpu.reset_execution_state(ResetOptions::default());
        assert_eq!(format!("{:?}", cpu.cpu), before);
        assert_eq!(vm.next_source_id(), next);
    }
}

fn dirty_x87(cpu: &mut Vcpu) {
    cpu.cpu.fpr = [[0x55; 10]; 8];
    cpu.cpu.fpu_cw = 0x1234;
    cpu.cpu.fpu_top = 5;
    cpu.cpu.fpu_sw = 0xffff;
    cpu.cpu.fpu_empty = 0;
    cpu.cpu.fpu_fip = 0x1234;
    cpu.cpu.fpu_fdp = 0x5678;
    cpu.cpu.fpu_fop = 0x321;
    cpu.cpu.fpu_fcs = 0x56;
    cpu.cpu.fpu_fds = 0x78;
}

#[test]
fn optional_x87_init_matches_fninit_without_changing_data_or_simd_state() {
    // SDM Vol 2A §3.3 FINIT/FNINIT, Description/Operation: environment defaults,
    // data registers unchanged and empty, XMM/MXCSR unaffected.
    for mode in [CpuMode::Real16, CpuMode::Compat32, CpuMode::Long64] {
        let (_, mut cpu) = setup(&[], mode, false);
        dirty_x87(&mut cpu);
        cpu.cpu.mxcsr = 0x3f80;
        cpu.cpu.xmm = [0x9876; 32];
        let mut expected = cpu.cpu.clone();
        expected.fpu_cw = 0x037f;
        expected.fpu_top = 0;
        expected.fpu_sw = 0;
        expected.fpu_empty = 0xff;
        expected.fpu_fip = 0;
        expected.fpu_fdp = 0;
        expected.fpu_fop = 0;
        expected.fpu_fcs = 0;
        expected.fpu_fds = 0;
        cpu.reset_execution_state(ResetOptions::default().with_x87_init(true));
        assert_eq!(format!("{:?}", cpu.cpu), format!("{expected:?}"));
    }
}
