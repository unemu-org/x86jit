//! Interrupt-shadow witnesses, independent of the optional Unicorn oracle.
//!
//! SDM Vol 3A §§7.8.1, 7.8.3 and Vol 2B §4.3 STI/MOV/POP: inhibition ends
//! after the following instruction or event delivery. Decision-16 §4 specifies
//! consecutive/mixed shadow policy and the first-completed-REP-iteration boundary.

use x86jit_core::{
    Backend, CpuMode, Exit, InterpreterBackend, Prot, Reg, RegionKind, StepResult, Vcpu, Vm,
    VmConfig,
};
use x86jit_cranelift::JitBackend;

const CS: u64 = 0x100;
const SS: u64 = 0x300;
const HANDLER_CS: u64 = 0x600;
const VECTOR: u8 = 0x40;

fn setup(code: &[u8]) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), Box::new(InterpreterBackend));
    vm.set_cpu_mode(CpuMode::Real16);
    for addr in (0..0x10000).step_by(4096) {
        vm.map(addr, 4096, Prot::RWX, RegionKind::Ram).unwrap();
    }
    for (page, bytes) in code.chunks(4096).enumerate() {
        vm.write_bytes((CS << 4) + (page * 4096) as u64, bytes)
            .unwrap();
    }
    vm.write_bytes(HANDLER_CS << 4, &[0xf4]).unwrap();
    vm.write_bytes(
        u64::from(VECTOR) * 4 + 2,
        &(HANDLER_CS as u16).to_le_bytes(),
    )
    .unwrap();
    vm.write_bytes(0x2100, &(SS as u16).to_le_bytes()).unwrap();
    vm.write_bytes(0x7000, &(SS as u16).to_le_bytes()).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Cs, CS);
    cpu.set_reg(Reg::Ss, 0x200);
    cpu.set_reg(Reg::Rsp, 0x100);
    cpu.set_reg(Reg::Rax, SS);
    let mut flags = cpu.flags();
    flags.if_ = true;
    cpu.set_flags(flags);
    (vm, cpu)
}

fn step(vm: &Vm, cpu: &mut Vcpu) {
    assert!(matches!(cpu.step_instruction(vm), StepResult::Continue));
}

fn delivered_at(vm: &Vm, cpu: &mut Vcpu, ip: u16) {
    assert!(matches!(cpu.run(vm, Some(4)), Exit::Hlt));
    assert!(!cpu.has_pending_irq());
    assert_eq!(cpu.reg(Reg::Cs), HANDLER_CS);
    let mut saved_ip = [0; 2];
    vm.read_bytes((SS << 4) + cpu.reg(Reg::Rsp), &mut saved_ip)
        .unwrap();
    assert_eq!(u16::from_le_bytes(saved_ip), ip);
}

fn check_load_shadow(load: &[u8], full_block: bool) {
    let mut code = if full_block {
        // The bounded fetch window ends exactly after the SS load.
        vec![0x90; 4096 - load.len()]
    } else {
        Vec::new()
    };
    code.extend_from_slice(load);
    let after_load = code.len() as u16;
    code.extend_from_slice(&[0xeb, 0, 0xf4]); // jmp next; hlt
    let (vm, mut cpu) = setup(&code);
    if full_block {
        assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
    } else {
        step(&vm, &mut cpu);
    }
    assert_eq!(cpu.reg(Reg::Rip), u64::from(after_load));
    assert_eq!(cpu.reg(Reg::Ss), SS);
    cpu.inject_irq(VECTOR);
    // The following JMP is one instruction and ends the next block. IRQ delivery
    // itself charges a dispatch boundary, so an early IRQ cannot pass this check.
    assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
    assert!(cpu.has_pending_irq());
    assert_eq!(cpu.reg(Reg::Cs), CS);
    assert_eq!(cpu.reg(Reg::Rip), u64::from(after_load + 2));
    delivered_at(&vm, &mut cpu, after_load + 2);
}

#[test]
fn mov_ss_register_step_boundary() {
    check_load_shadow(&[0x8e, 0xd0], false); // mov ss, ax
}

#[test]
fn mov_ss_memory_step_boundary() {
    check_load_shadow(&[0x8e, 0x16, 0, 0x70], false); // mov ss, [0x7000]
}

#[test]
fn pop_ss_step_boundary() {
    check_load_shadow(&[0x17], false);
}

#[test]
fn mov_ss_fetch_window_boundary() {
    check_load_shadow(&[0x8e, 0xd0], true);
}

#[test]
fn pop_ss_fetch_window_boundary() {
    check_load_shadow(&[0x17], true);
}

#[test]
fn mov_ss_mov_sp_switches_stack_before_irq() {
    // JMP ends the protected instruction's block so run() polls the queued IRQ.
    let (vm, mut cpu) = setup(&[0x8e, 0xd0, 0x89, 0xdc, 0xeb, 0, 0xf4]);
    cpu.set_reg(Reg::Rbx, 0x800);
    step(&vm, &mut cpu);
    cpu.inject_irq(VECTOR);
    assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
    assert!(cpu.has_pending_irq());
    assert_eq!(cpu.reg(Reg::Rsp), 0x800);
    delivered_at(&vm, &mut cpu, 6);
    assert_eq!(cpu.reg(Reg::Rsp), 0x800 - 6);
}

#[test]
fn following_instruction_in_same_block_consumes_ss_shadow() {
    for load in [&[0x8e, 0xd0][..], &[0x17][..]] {
        let mut code = load.to_vec();
        code.extend_from_slice(&[0x89, 0xdc, 0xeb, 0, 0xf4]); // mov sp,bx; jmp next; hlt
        let (vm, mut cpu) = setup(&code);
        cpu.set_reg(Reg::Rbx, 0x800);
        assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
        cpu.inject_irq(VECTOR);
        delivered_at(&vm, &mut cpu, code.len() as u16 - 1);
        assert_eq!(cpu.reg(Reg::Rsp), 0x800 - 6);
    }
}

#[test]
fn faulting_ss_load_does_not_arm_shadow() {
    for (code, fault_page) in [(&[0x8e, 0x16, 0, 0x70][..], 0x7000), (&[0x17][..], 0x2000)] {
        let (mut vm, mut cpu) = setup(code);
        vm.unmap(fault_page, 4096).unwrap();
        assert!(matches!(
            cpu.step_instruction(&vm),
            StepResult::Exit(Exit::UnmappedMemory { .. })
        ));
        assert_eq!(cpu.reg(Reg::Ss), 0x200);
        assert_eq!(cpu.reg(Reg::Rsp), 0x100);
        assert_eq!(cpu.reg(Reg::Rip), 0);
        // Supply a mapped interrupt stack after observing the fault.
        cpu.set_reg(Reg::Ss, SS);
        cpu.inject_irq(VECTOR);
        delivered_at(&vm, &mut cpu, 0);
    }
}

#[test]
fn sti_still_inhibits_the_following_instruction() {
    // SDM Vol 2B §4.3 STI: IF=0 on entry delays IRQ recognition by one instruction.
    let (vm, mut cpu) = setup(&[0xfb, 0xeb, 0, 0xf4]);
    cpu.set_reg(Reg::Ss, SS);
    let mut flags = cpu.flags();
    flags.if_ = false;
    cpu.set_flags(flags);
    step(&vm, &mut cpu);
    cpu.inject_irq(VECTOR);
    assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
    assert!(cpu.has_pending_irq());
    delivered_at(&vm, &mut cpu, 3);
}

#[test]
fn sti_then_mov_ss_inhibits_through_the_stack_switch() {
    // SDM Vol 3A §7.8.3: MOV SS inhibits the following boundary; STI is not
    // an SS load, so this is not the consecutive-SS-load qualification.
    let (vm, mut cpu) = setup(&[0xfb, 0x8e, 0xd0, 0xeb, 0, 0xf4]);
    let mut flags = cpu.flags();
    flags.if_ = false;
    cpu.set_flags(flags);
    step(&vm, &mut cpu);
    cpu.inject_irq(VECTOR);
    step(&vm, &mut cpu);
    assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
    assert!(cpu.has_pending_irq());
    delivered_at(&vm, &mut cpu, 5);
}

#[test]
fn consecutive_ss_loads_consume_without_rearming() {
    // SDM Vol 3A §7.8.3 guarantees only the first SS load's inhibition.
    // Decision-16 §4 chooses consumption without rearming for the second.
    for (first, second) in [
        (&[0x8e, 0xd0][..], &[0x8e, 0xd0][..]),
        (&[0x17][..], &[0x8e, 0xd0][..]),
        (&[0x8e, 0xd0][..], &[0x17][..]),
        (&[0x17][..], &[0x17][..]),
    ] {
        check_shadow_policy(first, second, true, false);
    }
}

fn check_unmapped_shadow(sti: bool) {
    // SDM Vol 3A §§7.8.1, 7.8.3: a host mapping request delivers no exception.
    for split in [false, true] {
        let producer: &[u8] = if sti { &[0xfb] } else { &[0x8e, 0xd0] };
        let mut code = producer.to_vec();
        // mov ax,[0x7000] / mov sp,[0x7000]; jmp next; hlt
        code.extend_from_slice(&[0x8b, if sti { 0x06 } else { 0x26 }, 0, 0x70]);
        code.extend_from_slice(&[0xeb, 0, 0xf4]);
        let (mut vm, mut cpu) = setup(&code);
        if sti {
            cpu.set_reg(Reg::Ss, SS);
            let mut flags = cpu.flags();
            flags.if_ = false;
            cpu.set_flags(flags);
        }
        vm.unmap(0x7000, 4096).unwrap();
        if split {
            step(&vm, &mut cpu);
        }
        assert!(matches!(cpu.run(&vm, Some(1)), Exit::UnmappedMemory { .. }));
        assert_eq!(cpu.reg(Reg::Rip), producer.len() as u64);
        assert_eq!(cpu.reg(Reg::Rsp), 0x100);
        assert_eq!(cpu.retired_instructions(), 1);
        // Repeated unsuccessful calls must preserve the same incoming state.
        assert!(matches!(
            cpu.step_instruction(&vm),
            StepResult::Exit(Exit::UnmappedMemory { .. })
        ));
        vm.map(0x7000, 4096, Prot::RW, RegionKind::Ram).unwrap();
        vm.write_bytes(0x7000, &0x800u16.to_le_bytes()).unwrap();
        cpu.inject_irq(VECTOR);
        assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
        assert!(cpu.has_pending_irq(), "sti={sti}, split={split}");
        assert_eq!(cpu.reg(Reg::Cs), CS);
        assert_eq!(cpu.reg(if sti { Reg::Rax } else { Reg::Rsp }), 0x800);
        delivered_at(&vm, &mut cpu, code.len() as u16 - 1);
        assert_eq!(cpu.reg(Reg::Rsp), if sti { 0x100 - 6 } else { 0x800 - 6 });
    }
}

#[test]
fn ss_shadow_survives_unmapped_protected_instruction() {
    check_unmapped_shadow(false);
}

#[test]
fn sti_shadow_survives_unmapped_protected_instruction() {
    check_unmapped_shadow(true);
}

#[test]
fn protected_instruction_fetch_failure_keeps_shadow() {
    for sti in [false, true] {
        let producer: &[u8] = if sti { &[0xfb] } else { &[0x8e, 0xd0] };
        let mut code = vec![0x90; 4096 - producer.len()];
        code.extend_from_slice(producer);
        let (mut vm, mut cpu) = setup(&code);
        cpu.set_reg(Reg::Ss, SS);
        cpu.set_reg(Reg::Rbx, 0x800);
        let mut flags = cpu.flags();
        flags.if_ = !sti;
        cpu.set_flags(flags);
        vm.unmap(0x2000, 4096).unwrap();
        assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
        assert!(matches!(cpu.run(&vm, Some(1)), Exit::UnmappedMemory { .. }));
        assert!(matches!(
            cpu.step_instruction(&vm),
            StepResult::Exit(Exit::UnmappedMemory { .. })
        ));
        vm.map(0x2000, 4096, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x2000, &[0x89, 0xdc, 0xeb, 0, 0xf4])
            .unwrap();
        cpu.inject_irq(VECTOR);
        assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
        assert!(cpu.has_pending_irq());
        assert_eq!(cpu.reg(Reg::Rsp), 0x800);
        delivered_at(&vm, &mut cpu, 4100);
    }
}

#[test]
fn sti_with_if_already_set_does_not_arm() {
    // SDM Vol 2B §4.3 STI: inhibition is conditional on entry IF=0.
    let (vm, mut cpu) = setup(&[0xfb, 0xeb, 0, 0xf4]);
    cpu.set_reg(Reg::Ss, SS);
    step(&vm, &mut cpu);
    cpu.inject_irq(VECTOR);
    delivered_at(&vm, &mut cpu, 1);
}

#[test]
fn sti_then_mov_ss_replaces_sti_with_ss() {
    check_shadow_policy(&[0xfb], &[0x8e, 0xd0], false, true);
}

#[test]
fn mov_ss_then_sti_with_clear_if_replaces_ss_with_sti() {
    check_shadow_policy(&[0x8e, 0xd0], &[0xfb], false, true);
}

#[test]
fn mov_ss_then_sti_with_set_if_leaves_no_shadow() {
    check_shadow_policy(&[0x8e, 0xd0], &[0xfb], true, false);
}

fn check_shadow_policy(first: &[u8], second: &[u8], initial_if: bool, blocked: bool) {
    // Decision-16 §4 resolves the combinations permitted by SDM Vol 3A
    // §§7.8.1, 7.8.3 and Vol 2B §4.3 STI/MOV/POP.
    for split in [false, true] {
        let mut code = if split {
            Vec::new()
        } else {
            vec![0x90; 4096 - first.len() - second.len()]
        };
        code.extend_from_slice(first);
        code.extend_from_slice(second);
        let after = code.len() as u16;
        code.extend_from_slice(&[0xeb, 0, 0xf4]);
        let (vm, mut cpu) = setup(&code);
        // Both stack slots support consecutive POP SS loads, before/after SS changes.
        for addr in [0x2100, 0x3100, 0x3102] {
            vm.write_bytes(addr, &(SS as u16).to_le_bytes()).unwrap();
        }
        let mut flags = cpu.flags();
        flags.if_ = initial_if;
        cpu.set_flags(flags);
        if split {
            step(&vm, &mut cpu);
            step(&vm, &mut cpu);
        } else {
            assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
        }
        assert_eq!(cpu.reg(Reg::Rip), u64::from(after));
        cpu.inject_irq(VECTOR);
        if blocked {
            assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
            assert!(cpu.has_pending_irq());
        }
        delivered_at(&vm, &mut cpu, after + if blocked { 2 } else { 0 });
    }
}

#[test]
fn mmio_request_preserves_sti_shadow_until_replay_completes() {
    use x86jit_core::interp::{step_one, RetireInfo};
    use x86jit_core::memory::{Memory, MemoryModel};
    use x86jit_core::state::CpuState;

    // Inspect the interpreter's continuation as well as the dispatcher's device
    // gate: that gate alone also inhibits IRQs during MMIO and can mask a lost latch.
    let mut mem = Memory::new(MemoryModel::Flat { size: 0x10000 });
    mem.map(0x1000, 4096, Prot::RWX, RegionKind::Ram).unwrap();
    mem.map(0x7000, 4096, Prot::RW, RegionKind::Trap).unwrap();
    mem.write_bytes(0x1000, &[0xfb, 0xa1, 0, 0x70]).unwrap();
    let mut cpu = CpuState::new();
    cpu.cs = CS as u16;
    cpu.flags.if_ = false;
    let mut scratch = Vec::new();
    let mut info = RetireInfo::default();
    assert!(matches!(
        step_one(&mem, &mut cpu, CpuMode::Real16, &mut scratch, &mut info),
        StepResult::Continue
    ));
    assert!(info.sti_shadow);
    assert!(matches!(
        step_one(&mem, &mut cpu, CpuMode::Real16, &mut scratch, &mut info),
        StepResult::Exit(Exit::MmioRead { .. })
    ));
    assert!(info.sti_shadow, "an MMIO request does not complete MOV");
    cpu.pending_mmio = Some(0x800);
    assert!(matches!(
        step_one(&mem, &mut cpu, CpuMode::Real16, &mut scratch, &mut info),
        StepResult::Continue
    ));
    assert!(!info.sti_shadow);
    assert_eq!(cpu.gpr[0], 0x800);
}

#[test]
fn mmio_protected_ss_load_consumes_incoming_ss_without_rearming() {
    let (mut vm, mut cpu) = setup(&[0x8e, 0xd0, 0x8e, 0x16, 0, 0x70, 0xeb, 0, 0xf4]);
    vm.unmap(0x7000, 4096).unwrap();
    vm.map(0x7000, 4096, Prot::RW, RegionKind::Trap).unwrap();
    assert!(matches!(cpu.run(&vm, Some(1)), Exit::MmioRead { .. }));
    cpu.inject_irq(VECTOR);
    cpu.complete_mmio_read(SS);
    assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
    assert!(cpu.has_pending_irq());
    assert_eq!(cpu.reg(Reg::Rip), 6);
    delivered_at(&vm, &mut cpu, 6);
}

#[test]
fn pending_port_in_keeps_shadow_until_result_completion() {
    use x86jit_core::interp::{step_one, RetireInfo};
    use x86jit_core::memory::{Memory, MemoryModel};
    use x86jit_core::state::CpuState;

    let mut mem = Memory::new(MemoryModel::Flat { size: 0x10000 });
    mem.map(0x1000, 4096, Prot::RWX, RegionKind::Ram).unwrap();
    mem.write_bytes(0x1000, &[0xfb, 0xe4, 0x80]).unwrap(); // sti; in al,0x80
    let mut cpu = CpuState::new();
    cpu.cs = CS as u16;
    cpu.flags.if_ = false;
    let mut scratch = Vec::new();
    let mut info = RetireInfo::default();
    assert!(matches!(
        step_one(&mem, &mut cpu, CpuMode::Real16, &mut scratch, &mut info),
        StepResult::Continue
    ));
    assert!(matches!(
        step_one(&mem, &mut cpu, CpuMode::Real16, &mut scratch, &mut info),
        StepResult::Exit(Exit::PortIo { .. })
    ));
    assert_eq!(info.retired, 1); // The handoff convention still counts IN once.
    assert!(info.sti_shadow);
    assert_eq!(cpu.pending_port_in, Some(1));

    for producer in [&[0xfb][..], &[0x8e, 0xd0][..]] {
        let mut code = producer.to_vec();
        code.extend_from_slice(&[0xe4, 0x80, 0xeb, 0, 0xf4]);
        let (vm, mut cpu) = setup(&code);
        cpu.set_reg(Reg::Ss, SS);
        let mut flags = cpu.flags();
        flags.if_ = producer != [0xfb];
        cpu.set_flags(flags);
        assert!(matches!(cpu.run(&vm, Some(1)), Exit::PortIo { .. }));
        assert_eq!(cpu.retired_instructions(), 2);
        cpu.inject_irq(VECTOR);
        cpu.complete_port_in(0x42);
        assert_eq!(cpu.reg(Reg::Rax) & 0xff, 0x42);
        assert_eq!(cpu.retired_instructions(), 2);
        // Completing IN must consume inhibition before the next dispatch.
        delivered_at(&vm, &mut cpu, producer.len() as u16 + 2);
    }
}

#[test]
fn rep_without_completed_iteration_keeps_shadow() {
    check_rep_shadow(false);
}

#[test]
fn rep_first_completed_iteration_consumes_shadow() {
    check_rep_shadow(true);
}

fn check_rep_shadow(progress: bool) {
    // Decision-16 §4; SDM Vol 3A §§7.6, 7.8.1, 7.8.3: the first committed
    // iteration ends inhibition; a host exit before that iteration does not.
    for sti in [false, true] {
        let producer: &[u8] = if sti { &[0xfb] } else { &[0x8e, 0xd0] };
        let mut code = producer.to_vec();
        code.extend_from_slice(&[0xf3, 0xa4, 0xeb, 0, 0xf4]); // rep movsb; jmp next; hlt
        let (mut vm, mut cpu) = setup(&code);
        cpu.set_reg(Reg::Ss, SS);
        cpu.set_reg(Reg::Rsi, if progress { 0x6fff } else { 0x7000 });
        cpu.set_reg(Reg::Rdi, 0x8000);
        cpu.set_reg(Reg::Rcx, 2);
        let mut flags = cpu.flags();
        flags.if_ = !sti;
        cpu.set_flags(flags);
        vm.write_bytes(0x6fff, &[0x42]).unwrap();
        vm.unmap(0x7000, 4096).unwrap();
        assert!(matches!(cpu.run(&vm, Some(1)), Exit::UnmappedMemory { .. }));
        assert_eq!(cpu.reg(Reg::Rip), producer.len() as u64);
        assert_eq!(cpu.reg(Reg::Rcx), if progress { 1 } else { 2 });
        assert_eq!(cpu.reg(Reg::Rdi), 0x8000 + u64::from(progress));
        assert_eq!(cpu.retired_instructions(), 1);
        vm.map(0x7000, 4096, Prot::RW, RegionKind::Ram).unwrap();
        cpu.inject_irq(VECTOR);
        if progress {
            delivered_at(&vm, &mut cpu, producer.len() as u16);
            assert_eq!(cpu.reg(Reg::Rcx), 1); // No second iteration before the IRQ.
        } else {
            assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
            assert!(cpu.has_pending_irq());
            assert_eq!(cpu.reg(Reg::Rcx), 0);
            delivered_at(&vm, &mut cpu, code.len() as u16 - 1);
        }
    }
}

#[test]
fn interrupted_rep_before_first_iteration_preserves_sti_shadow() {
    use std::sync::atomic::AtomicU64;
    use x86jit_core::interp::{step_one, RetireInfo};
    use x86jit_core::memory::{Memory, MemoryModel};
    use x86jit_core::state::CpuState;

    let mut mem = Memory::new(MemoryModel::Flat { size: 0x10000 });
    mem.map(0x1000, 4096, Prot::RWX, RegionKind::Ram).unwrap();
    mem.write_bytes(0x1000, &[0xfb, 0xf3, 0xa4]).unwrap();
    let mut cpu = CpuState::new();
    cpu.cs = CS as u16;
    cpu.flags.if_ = false;
    cpu.gpr[1] = 2;
    let mut scratch = Vec::new();
    let mut info = RetireInfo::default();
    assert!(matches!(
        step_one(&mem, &mut cpu, CpuMode::Real16, &mut scratch, &mut info),
        StepResult::Continue
    ));
    let request = AtomicU64::new(2); // The engine exit-request bit (retired_count witness).
    info.exit_request = Some(&request);
    assert!(matches!(
        step_one(&mem, &mut cpu, CpuMode::Real16, &mut scratch, &mut info),
        StepResult::Exit(Exit::Interrupted)
    ));
    assert_eq!(info.retired, 0);
    assert!(info.sti_shadow);
    assert_eq!(cpu.gpr[1], 2);
}

#[test]
fn delivered_guest_fault_ends_shadow() {
    check_delivered_guest_fault(false);
}

#[test]
fn stepped_guest_fault_delivery_ends_shadow() {
    check_delivered_guest_fault(true);
}

fn check_delivered_guest_fault(stepped: bool) {
    // SDM Vol 3A §§7.8.1, 7.8.3: actual exception delivery ends inhibition.
    // UD2 is delivered by the walk; DIV's #DE is delivered by Vcpu.
    for fault in [&[0x0f, 0x0b][..], &[0xf6, 0xf3][..]] {
        for sti in [false, true] {
            let mut code = if sti { vec![0xfb] } else { vec![0x8e, 0xd0] };
            code.extend_from_slice(fault);
            let (vm, mut cpu) = setup(&code);
            cpu.set_reg(Reg::Ss, SS);
            let mut flags = cpu.flags();
            flags.if_ = !sti;
            cpu.set_flags(flags);
            let vector = if fault[0] == 0x0f { 6 } else { 0 };
            vm.write_bytes(vector * 4 + 2, &0x500u16.to_le_bytes())
                .unwrap();
            vm.write_bytes(0x5000, &[0x90, 0xf4]).unwrap();
            step(&vm, &mut cpu);
            cpu.inject_irq(VECTOR);
            if stepped {
                step(&vm, &mut cpu);
            } else {
                assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
            }
            assert_eq!(cpu.reg(Reg::Cs), 0x500);
            assert_eq!(cpu.reg(Reg::Rip), 0);
            assert_eq!(cpu.retired_instructions(), 1);
            let mut flags = cpu.flags();
            flags.if_ = true;
            cpu.set_flags(flags);
            delivered_at(&vm, &mut cpu, 0);
            let mut saved_cs = [0; 2];
            vm.read_bytes((SS << 4) + cpu.reg(Reg::Rsp) + 2, &mut saved_cs)
                .unwrap();
            assert_eq!(u16::from_le_bytes(saved_cs), 0x500);
        }
    }
}

#[test]
fn other_segment_loads_do_not_inhibit_irqs() {
    let (vm, mut cpu) = setup(&[0x8e, 0xd8, 0xf4]); // mov ds, ax; hlt
    cpu.set_reg(Reg::Ss, SS);
    step(&vm, &mut cpu);
    cpu.inject_irq(VECTOR);
    delivered_at(&vm, &mut cpu, 2);
}

#[test]
fn descriptor_loads_remain_delegated_in_user_modes() {
    // SDM Vol 2B §4.3 MOV/POP, Operation: protected-mode SS loads require
    // descriptor validation, which the embedder owns. Refusal must not load SS
    // or advance RIP. Long64 POP SS is invalid and raises #UD instead.
    for mode in [CpuMode::Compat32, CpuMode::Long64] {
        for jit in [false, true] {
            for code in [&[0x8e, 0xd0][..], &[0x17][..]] {
                let backend: Box<dyn Backend> = if jit {
                    Box::new(JitBackend::new())
                } else {
                    Box::new(InterpreterBackend)
                };
                let mut vm = Vm::with_backend(VmConfig::flat(0x4000), backend);
                vm.set_cpu_mode(mode);
                vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
                vm.write_bytes(0x1000, code).unwrap();
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, 0x1000);
                cpu.set_reg(Reg::Ss, 0x23);
                cpu.set_reg(Reg::Rax, 0x2b);
                let exit = cpu.run(&vm, Some(4));
                if mode == CpuMode::Long64 && code == [0x17] {
                    assert!(matches!(exit, Exit::Exception { vector: 6, .. }));
                } else {
                    assert!(matches!(exit, Exit::UnknownInstruction { .. }));
                }
                assert_eq!(cpu.reg(Reg::Rip), 0x1000);
                assert_eq!(cpu.reg(Reg::Ss), 0x23);
            }
        }
    }
}
