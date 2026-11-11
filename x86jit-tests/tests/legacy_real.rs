//! Legacy I386 real-address witnesses. Expected values are derived from the
//! cited manuals; no emulator supplies instruction semantics.
use x86jit_core::{
    Backend, CachedBlock, CpuGeneration, CpuMode, CpuState, Exit, IrBlock, LegacyCpuModel,
    MemConsistency, Prot, RegionKind, SegmentId, StepOptions, StepRunError, StepStop,
    UnsupportedSystemOperation, Vcpu, Vm, VmConfig,
};

const MODEL: LegacyCpuModel = LegacyCpuModel {
    generation: CpuGeneration::I386,
    model_semantics_id: 0x386,
};
struct NoBackend;
impl Backend for NoBackend {
    fn materialize(
        &self,
        _: &IrBlock,
        _: MemConsistency,
        _: Option<(u64, u64)>,
        _: u64,
    ) -> CachedBlock {
        panic!("legacy execution reached application backend");
    }
}
fn setup(code: &[u8]) -> (Vm, Vcpu) {
    setup_model(code, CpuGeneration::I386)
}
fn setup_model(code: &[u8], generation: CpuGeneration) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x40000), Box::new(NoBackend));
    vm.map(0, 0x40000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.set_legacy_cpu_model(LegacyCpuModel {
        generation,
        ..MODEL
    });
    vm.write_bytes(0x1000, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[4] = 0x8000;
    (vm, cpu)
}
fn steps(cpu: &mut Vcpu, vm: &Vm, n: u64) {
    let run = cpu.run_steps(vm, n, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), n, "{:?}", run.stop());
    assert!(
        matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
        "{:?}",
        run.stop()
    );
}
fn value(vm: &Vm, a: u64, n: usize) -> u64 {
    let mut b = [0; 8];
    vm.read_bytes(a, &mut b[..n]).unwrap();
    u64::from_le_bytes(b)
}

#[test]
fn iretd_ignores_stacked_vm_in_real_mode() {
    // SDM Vol 2A §3.3, IRET/IRETD/IRETQ, Operation, REAL-ADDRESS-MODE,
    // p. 3-491: (tempEFLAGS AND 257FD5H) OR (EFLAGS AND 1A0000H)
    // preserves the current VM bit instead of loading the stacked VM bit.
    let (vm, mut cpu) = setup(b"\x66\xcf");
    for (i, word) in [0x1200u32, 0x100, 0x20002].into_iter().enumerate() {
        vm.write_bytes(0x8000 + i as u64 * 4, &word.to_le_bytes())
            .unwrap();
    }
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.legacy.cr0 & 1, 0);
    assert_eq!(
        (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
        (0x100, 0x1200, 0x800c)
    );
    assert_eq!(cpu.cpu.rflags_system & (1 << 17), 0);
    assert_eq!(cpu.cpu.flags.to_flags16(), 2);
    assert_eq!(value(&vm, 0x8008, 4), 0x20002);
}

fn exception(cpu: &mut Vcpu, vm: &Vm, vector: u8) {
    let run = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0);
    assert!(
        matches!(run.stop(),StepStop::Exit(Exit::Exception { vector:v,.. }) if *v==vector),
        "{:?}",
        run.stop()
    );
    let StepStop::Exit(exit) = run.stop() else {
        unreachable!()
    };
    let details = exit
        .fault_metadata()
        .expect("precise legacy fault evidence");
    assert_eq!(details.class(), x86jit_core::fault::FaultClass::Fault);
    assert_eq!(details.error_code(), x86jit_core::fault::ErrorCode::NoCode);
}

#[test]
fn real_word_at_ffff_faults_with_ss_for_stack_and_gp_for_data() {
    // 80386 PRM §14.6 Table 14-1 p. 14-5; SDM Vol 3B §23.1.4 Table 23-1:
    // real-mode SS overruns raise #SS; other segment overruns raise #GP.
    for (code, vector) in [(b"\x36\xa1\xff\xff", 12), (b"\x3e\xa1\xff\xff", 13)] {
        let (vm, mut cpu) = setup(code);
        vm.write_bytes(0xffff, b"\x34\x12").unwrap();
        cpu.cpu.gpr[0] = 0xabcd;
        let before = format!("{:?}", cpu.cpu);
        exception(&mut cpu, &vm, vector);
        assert_eq!(format!("{:?}", cpu.cpu), before);
    }
}

#[test]
fn operand_and_address_prefix_combinations_and_full_offsets() {
    // SDM Vol 3B §§23.1.1–23.1.2: operand and address prefixes are
    // independent; 32-bit offsets are checked in full against the real limit.
    for op32 in [false, true] {
        for addr32 in [false, true] {
            let mut code = Vec::new();
            if op32 {
                code.push(0x66);
            }
            if addr32 {
                code.push(0x67);
            }
            code.extend(if addr32 {
                &b"\x8b\x03"[..]
            } else {
                &b"\x8b\x07"[..]
            });
            let (vm, mut cpu) = setup(&code);
            cpu.cpu.gpr[3] = 0x2300;
            vm.write_bytes(0x2300, &[0x78, 0x56, 0x34, 0x12]).unwrap();
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.gpr[0], if op32 { 0x12345678 } else { 0x5678 });
            cpu.cpu.rip = 0x1000;
            cpu.cpu.gpr[3] = 0x12300;
            if addr32 {
                exception(&mut cpu, &vm, 13);
            } else {
                steps(&mut cpu, &vm, 1);
                assert_eq!(cpu.cpu.gpr[0], if op32 { 0x12345678 } else { 0x5678 });
            }
        }
    }
    let (vm, mut cpu) = setup(b"\x66\x67\x89\x03");
    cpu.cpu.gpr[3] = 0xfffe;
    cpu.cpu.gpr[0] = 0x12345678;
    vm.write_bytes(0xfffe, &[0xaa; 4]).unwrap();
    exception(&mut cpu, &vm, 13);
    assert_eq!(value(&vm, 0xfffe, 4), 0xaaaaaaaa);
    assert_eq!(cpu.cpu.rip, 0x1000);
}

#[test]
fn fs_gs_loads_overrides_and_far_pointer_loads() {
    // SDM Vol 3B §§23.1.1–23.1.3; Vol 2B §4.3 MOV; Vol 2A §3.3
    // LDS/LES/LFS/LGS/LSS: offset width follows operand size; selector is a word.
    let (vm, mut cpu) =
        setup(b"\xb8\x00\x02\x8e\xe0\xb8\x00\x03\x8e\xe8\x64\x66\x67\x8b\x03\x65\x66\x67\x89\x03");
    cpu.cpu.gpr[3] = 0x120;
    cpu.cpu.legacy.segments[4].unusable = 1;
    cpu.cpu.fs_base = 0xdead0000;
    cpu.cpu.gs_base = 0xbeef0000;
    vm.write_bytes(0x2120, &[0x78, 0x56, 0x34, 0x12]).unwrap();
    steps(&mut cpu, &vm, 6);
    assert_eq!((cpu.cpu.fs, cpu.cpu.gs), (0x200, 0x300));
    assert_eq!(value(&vm, 0x3120, 4), 0x12345678);
    assert_eq!(cpu.cpu.legacy.segments[4].base, 0x2000);
    assert_eq!(cpu.cpu.fs_base, 0xdead0000);
    for (opcode, segment) in [
        (&b"\x0f\xb4"[..], SegmentId::Fs),
        (&b"\x0f\xb5"[..], SegmentId::Gs),
        (&b"\x0f\xb2"[..], SegmentId::Ss),
        (&b"\xc4"[..], SegmentId::Es),
        (&b"\xc5"[..], SegmentId::Ds),
    ] {
        for wide in [false, true] {
            let mut code = Vec::new();
            if wide {
                code.push(0x66);
            }
            code.extend(opcode);
            code.extend(b"\x06\x00\x20");
            let (vm, mut cpu) = setup(&code);
            vm.write_bytes(
                0x2000,
                if wide {
                    &b"\x78\x56\x34\x12\x00\x04"[..]
                } else {
                    &b"\x78\x56\x00\x04"[..]
                },
            )
            .unwrap();
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.gpr[0], if wide { 0x12345678 } else { 0x5678 });
            assert_eq!(cpu.cpu.legacy.segments[segment as usize].base, 0x4000);
        }
    }
}

#[test]
fn pushfd_popfd_identification_masks_and_tf_storage() {
    // SDM Vol 1 §21.1.2; 80386 PRM §2.3.4 Fig. 2-8 p. 2-11,
    // §17 POPF p. 17-130 (page images checked): AC/ID/reserved bits zero,
    // IOPL/NT writable; RF/VM unaffected. §12.3.1.4 p. 12-8: setting
    // TF does not trap until after the following instruction.
    let (vm, mut cpu) = setup(b"\x66\x9d\x66\x9c");
    vm.write_bytes(0x8000, &0xfffffeffu32.to_le_bytes())
        .unwrap();
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.rflags_system, 0x7000);
    steps(&mut cpu, &vm, 1);
    assert_eq!(value(&vm, 0x8000, 4), 0x7ed7);
    assert_eq!(cpu.cpu.gpr[4], 0x8000);
    cpu.cpu.rip = 0x1000;
    vm.write_bytes(0x8000, &0x100u32.to_le_bytes()).unwrap();
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.gpr[4], 0x8004);
    assert_eq!(cpu.cpu.rflags_system, 0x100);
}

#[test]
fn pushad_popad_original_esp_and_independent_stack_width() {
    // SDM Vol 1 §6.2.3; Vol 3A §3.4.5; Vol 2B §4.3 PUSHA/POPA:
    // SS.B chooses pointer size; the operand size chooses each saved value.
    for big in [false, true] {
        let (vm, mut cpu) = setup(b"\x66\x60\x66\x61");
        cpu.cpu.gpr[..8].copy_from_slice(&[
            0x11112222, 0x33334444, 0x55556666, 0x77778888, 0x18000, 0x9999aaaa, 0xbbbbcccc,
            0xddddeeee,
        ]);
        if big {
            cpu.cpu.legacy.segments[3].attributes |= 1 << 14;
            cpu.cpu.legacy.segments[3].limit = 0x1ffff;
        }
        let before = cpu.cpu.gpr;
        steps(&mut cpu, &vm, 1);
        let top = if big { 0x17fe0 } else { 0x7fe0 };
        assert_eq!(value(&vm, top + 12, 4), 0x18000);
        assert_eq!(value(&vm, top, 4), 0xddddeeee);
        cpu.cpu.gpr[0] = 0;
        cpu.cpu.gpr[7] = 0;
        steps(&mut cpu, &vm, 1);
        assert_eq!(&cpu.cpu.gpr[..8], &before[..8]);
    }
}

#[test]
fn far_1632_call_jump_and_return_frames() {
    // SDM Vol 2A §3.3 CALL/JMP; Vol 2B §4.3 RET: a 32-bit far call
    // saves dword CS and EIP slots; RETF pops both, then its optional byte count.
    let (vm, mut cpu) = setup(b"\x66\x9a\x00\x20\x00\x00\x00\x01\xf4");
    vm.write_bytes(0x3000, b"\x66\xcb").unwrap();
    steps(&mut cpu, &vm, 1);
    assert_eq!(
        (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
        (0x100, 0x2000, 0x7ff8)
    );
    assert_eq!(value(&vm, 0x7ff8, 4), 0x1008);
    assert_eq!(value(&vm, 0x7ffc, 4), 0);
    steps(&mut cpu, &vm, 1);
    assert_eq!(
        (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
        (0, 0x1008, 0x8000)
    );
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    for (code, call) in [
        (&b"\x66\xea\x00\x20\x00\x00\x00\x01"[..], false),
        (&b"\x66\x67\xff\x2b"[..], false),
        (&b"\x66\x67\xff\x1b"[..], true),
    ] {
        let (vm, mut cpu) = setup(code);
        cpu.cpu.gpr[3] = 0x4000;
        vm.write_bytes(0x4000, b"\x00\x20\x00\x00\x00\x01").unwrap();
        steps(&mut cpu, &vm, 1);
        assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0x100, 0x2000));
        assert_eq!(cpu.cpu.gpr[4], if call { 0x7ff8 } else { 0x8000 });
        if call {
            assert_eq!(value(&vm, 0x7ff8, 4), 0x1004);
        }
    }
    let (vm, mut cpu) = setup(b"\x66\xea\x00\x00\x01\x00\x00\x01");
    exception(&mut cpu, &vm, 13);
    assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0, 0x1000));
}

#[test]
fn strings_step_by_iteration_with_effective_indices_and_limits() {
    // SDM Vol 2B §4.3 REP/MOVS: one effective-sized element per iteration;
    // decision-16 charges an iteration without prematurely retiring the REP.
    let (vm, mut cpu) = setup(b"\x66\x67\xf3\xa5\xf4");
    cpu.cpu.gpr[1] = 2;
    cpu.cpu.gpr[6] = 0xfffc;
    cpu.cpu.gpr[7] = 0x2000;
    vm.write_bytes(0xfffc, b"\x78\x56\x34\x12").unwrap();
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.gpr[6], 0x10000);
    assert_eq!(cpu.cpu.gpr[1], 1);
    assert_eq!(cpu.retired_instructions(), 0);
    assert_eq!(value(&vm, 0x2000, 4), 0x12345678);
    exception(&mut cpu, &vm, 13);
    assert_eq!(cpu.cpu.gpr[1], 1);
    assert_eq!(value(&vm, 0x2004, 4), 0);
}

#[test]
fn zero_steps_port_completion_and_sti_ss_shadows() {
    // Decision-16 §§1,4; SDM Vol 3A §§7.8.1,7.8.3: a queued IRQ waits
    // through STI and one completed instruction, then delivery costs zero steps.
    let (vm, mut cpu) = setup(b"\xfb\x66\x60\x90");
    vm.write_bytes(0x80, b"\x00\x18\x00\x00").unwrap();
    vm.write_bytes(0x1800, b"\xf4").unwrap();
    cpu.inject_irq(0x20);
    steps(&mut cpu, &vm, 0);
    steps(&mut cpu, &vm, 1);
    steps(&mut cpu, &vm, 1);
    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 1);
    assert!(matches!(run.stop(), StepStop::Exit(Exit::Hlt)));
    assert!(!cpu.has_pending_irq());
    assert!(!cpu.irq_ready());
    let (vm, mut cpu) = setup(b"\xe4\x20\xf4");
    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 1);
    assert!(matches!(run.stop(), StepStop::Exit(Exit::PortIo { .. })));
    assert!(matches!(
        cpu.run_steps(&vm, 1, StepOptions::default()),
        Err(StepRunError::PendingPortIn)
    ));
    steps(&mut cpu, &vm, 0);
    cpu.complete_port_in(0x42);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(cpu.cpu.gpr[0], 0x42);
}

#[test]
fn protected_entry_executes_with_retained_caches_and_legacy_never_materializes() {
    let (vm, mut cpu) = setup(b"\x66\xb8\x78\x56\x34\x12\xf4");
    cpu.cpu.legacy.cr0 = 1;
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(cpu.cpu.gpr[0], 0x12345678);
    let mut jit = Vm::with_backend(
        VmConfig::flat(0x4000),
        Box::new(x86jit_cranelift::JitBackend::new()),
    );
    jit.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
    jit.set_legacy_cpu_model(MODEL);
    jit.write_bytes(0x1000, b"\x66\x50\x66\x5b\xf4").unwrap();
    let mut cpu = jit.new_vcpu();
    cpu.cpu = CpuState::legacy_real_entry([0; 6]);
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[4] = 0x3000;
    cpu.cpu.gpr[0] = 0x12345678;
    assert!(matches!(cpu.run(&jit, None), Exit::Hlt));
    assert_eq!(cpu.cpu.gpr[3], 0x12345678);
}

#[test]
fn cache_guards_alias_decode_ip_retained_cache_and_smc() {
    // Decision-18 §3: decode IP and hidden caches participate in translation
    // identity. SDM Vol 3A §3.4.3: visible selector edits do not reload a cache.
    let (vm, mut cpu) = setup(b"\x66\xb8\x01\x00\x00\x00\xeb\xf8");
    steps(&mut cpu, &vm, 2);
    assert_eq!(cpu.cpu.rip, 0x1000);
    vm.write_bytes(0x1002, &2u32.to_le_bytes()).unwrap();
    steps(&mut cpu, &vm, 2);
    assert_eq!(cpu.cpu.gpr[0], 2);
    // The same physical bytes represent a different offset under another CS map.
    cpu.cpu.cs = 0x80;
    cpu.cpu.legacy.segments[0].base = 0x800;
    cpu.cpu.rip = 0x800;
    steps(&mut cpu, &vm, 2);
    assert_eq!(cpu.cpu.rip, 0x800);
    cpu.cpu.rip = 0x200;
    vm.write_bytes(0xa00, b"\x67\x8b\x03").unwrap();
    cpu.cpu.gpr[3] = 0x100;
    cpu.cpu.ds = 0x500;
    cpu.cpu.legacy.segments[1].base = 0x4000;
    vm.write_bytes(0x4100, b"\x34\x12").unwrap();
    vm.write_bytes(0x5100, b"\x78\x56").unwrap();
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.gpr[0] & 0xffff, 0x1234);
    cpu.cpu.rip = 0x200;
    cpu.cpu.legacy.segments[1].base = 0x5000;
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.gpr[0] & 0xffff, 0x5678);
    let mut other = vm.new_vcpu();
    other.cpu.rip = 0xa00;
    other.cpu.gpr[3] = 0x100;
    vm.write_bytes(0x100, b"\xbc\x9a").unwrap();
    steps(&mut other, &vm, 1);
    assert_eq!(other.cpu.gpr[0], 0x9abc);
}

#[test]
fn fetch_crossings_fault_at_exact_instruction_and_keep_completed_prefix() {
    // SDM Vol 3B §23.1.4 Table 23-1: executing beyond CS limit is #GP;
    // decode/fetch never turns a fault in the next instruction into an early exit.
    let (vm, mut cpu) = setup(b"\x66\x40");
    cpu.cpu.legacy.segments[0].limit = 0x1001;
    let run = cpu.run_steps(&vm, 2, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 1);
    assert_eq!(cpu.cpu.gpr[0], 1);
    assert!(matches!(
        run.stop(),
        StepStop::Exit(Exit::Exception {
            vector: 13,
            addr: 0x1002,
            ..
        })
    ));
    cpu.cpu.rip = 0xffff;
    cpu.cpu.legacy.segments[0].limit = 0xffff;
    vm.write_bytes(0xffff, b"\x66\x40").unwrap();
    exception(&mut cpu, &vm, 13);
    assert_eq!(cpu.cpu.gpr[0], 1);
}

#[test]
fn stack_limit_faults_preserve_registers_and_multiword_frame() {
    // SDM Vol 3B §23.1.4 Table 23-1: stack references use #SS. Engine
    // fault-before-commit contract preflights all destinations of a frame.
    let (vm, mut cpu) = setup(b"\x66\x60");
    cpu.cpu.gpr[4] = 0x8000;
    cpu.cpu.legacy.segments[3].limit = 0x7ff0;
    vm.write_bytes(0x7fe0, &[0xaa; 32]).unwrap();
    let before = cpu.cpu.gpr;
    exception(&mut cpu, &vm, 12);
    assert_eq!(cpu.cpu.gpr, before);
    assert_eq!(value(&vm, 0x7ffc, 4), 0xaaaaaaaa);
    let (vm, mut cpu) = setup(b"\x66\x67\x8b\x45\x00");
    cpu.cpu.gpr[5] = 0x10000;
    exception(&mut cpu, &vm, 12);
}

#[test]
fn mmio_far_pointer_retry_keeps_completed_reads_and_no_partial_cs() {
    // Decision-16 completion ownership and decision-18 §3: all target reads
    // precede the transfer; a retry does not repeat an answered device transfer.
    let (mut vm, mut cpu) = setup(b"\x66\xff\x2e\x00\x00");
    vm.unmap(0, 0x40000).unwrap();
    vm.map(0, 0x30000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, b"\x66\xff\x2e\x00\x00").unwrap();
    vm.map(0x30000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    cpu.cpu.legacy.segments[1].base = 0x30000;
    for (address, size, answer) in [(0x30000, 4, 0x2000), (0x30004, 2, 0x100)] {
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(
            matches!(run.stop(),StepStop::Exit(Exit::MmioRead { addr,size:s,.. }) if *addr==address && *s==size),
            "{:?}",
            run.stop()
        );
        assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0, 0x1000));
        cpu.complete_mmio_read(answer);
        let before = format!("{:?}", cpu.cpu);
        steps(&mut cpu, &vm, 0);
        assert_eq!(format!("{:?}", cpu.cpu), before);
    }
    steps(&mut cpu, &vm, 1);
    assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0x100, 0x2000));
    assert!(!cpu.cpu.mmio_retry_active);
}

#[test]
fn retained_code_and_stack_defaults_remain_independent_of_real_regime() {
    // Decision-18 §1; SDM Vol 3A §§3.4.5,12.9.2 and AMD64 APM
    // Vol 2 §§4.1,14.6: cached D/B attributes select independent sizes.
    let (vm, mut cpu) = setup(b"\xb8\x78\x56\x34\x12\x67\x8b\x07\x50\x5a");
    cpu.cpu.legacy.segments[0].attributes |= 1 << 14;
    cpu.cpu.legacy.segments[1].base = 0x4000;
    cpu.cpu.gpr[3] = 0x100;
    vm.write_bytes(0x4100, b"\x44\x33\x22\x11").unwrap();
    steps(&mut cpu, &vm, 4);
    assert_eq!(cpu.cpu.gpr[0], 0x11223344);
    assert_eq!(cpu.cpu.gpr[2], 0x11223344);
    assert_eq!(cpu.cpu.gpr[4], 0x8000);
}

#[test]
fn segment_stack_forms_model_identification_and_task_return_refusal() {
    // SDM Vol 2B §4.3 PUSH/POP; Vol 1 §21.1.2 (no 386 CPUID);
    // decision-18 §2; Vol 3A §10.3 (NT task-return scope).
    let (vm, mut cpu) = setup(b"\x66\x0f\xa0\x66\x0f\xa9\xcf");
    cpu.cpu.fs = 0x1234;
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.gpr[4], 0x7ffc);
    assert_eq!(value(&vm, 0x7ffc, 4), 0x1234);
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.gs, 0x1234);
    assert_eq!(cpu.cpu.gpr[4], 0x8000);
    cpu.cpu.rflags_system = 0x4000;
    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0);
    assert!(matches!(
        run.stop(),
        StepStop::Exit(Exit::UnsupportedSystemContext {
            operation: UnsupportedSystemOperation::LegacyTaskReturn,
            ..
        })
    ));
    let (vm, mut cpu) = setup(b"\x0f\xa2");
    exception(&mut cpu, &vm, 6);
    // SDM Vol 2B §4.3 NOP: the multi-byte form is unavailable on I386.
    let (vm, mut cpu) = setup(b"\x0f\x1f\x00");
    exception(&mut cpu, &vm, 6);
}

#[test]
fn far_pointer_whole_operand_limit_and_ret_cleanup() {
    // SDM Vol 3B §23.1.1; Vol 2A §3.3 CALL/JMP; Vol 2B §4.3 RET:
    // the whole m16:32 operand must fit; RETF immediate adds bytes after both slots.
    let (vm, mut cpu) = setup(b"\x66\x67\xff\x1b");
    cpu.cpu.gpr[3] = 0xfffc;
    vm.write_bytes(0xfffc, b"\x00\x20\x00\x00\x00\x01").unwrap();
    exception(&mut cpu, &vm, 13);
    assert_eq!((cpu.cpu.cs, cpu.cpu.gpr[4]), (0, 0x8000));
    let (vm, mut cpu) = setup(b"\x66\xca\x0c\x00");
    vm.write_bytes(0x8000, b"\x00\x20\x00\x00\x00\x01\xff\xff")
        .unwrap();
    steps(&mut cpu, &vm, 1);
    assert_eq!(
        (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
        (0x100, 0x2000, 0x8014)
    );
}

#[test]
fn sequential_eip_does_not_wrap_at_ffff_and_32bit_targets_keep_their_offset() {
    // 80386 PRM §14.7 item 8 p. 14-6: sequential execution past FFFF
    // faults. SDM Vol 2A §3.3 JMP/CALL: effective operand width controls
    // transfer truncation separately from the default decode width.
    let (vm, mut cpu) = setup(b"\x90");
    vm.write_bytes(0xffff, b"\x90").unwrap();
    cpu.cpu.rip = 0xffff;
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.rip, 0x10000);
    exception(&mut cpu, &vm, 13);
    let (vm, mut cpu) = setup(b"\x66\xea\x00\x20\x01\x00\x00\x01");
    cpu.cpu.legacy.segments[0].limit = 0x1ffff;
    vm.write_bytes(0x13000, b"\x90").unwrap();
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.rip, 0x12000);
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.rip, 0x12001);
}

#[test]
fn mmio_selector_answer_does_not_replace_a_ram_far_offset() {
    // Decision-16 §3: an answer belongs to its device transfer, including when
    // the SDM Vol 2A §3.3 JMP m16:32 operand spans RAM and MMIO.
    let (mut vm, mut cpu) = setup(b"\x66\xff\x2e\x00\x00");
    vm.unmap(0, 0x40000).unwrap();
    vm.map(0, 0x30000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x30000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    vm.write_bytes(0x1000, b"\x66\xff\x2e\x00\x00").unwrap();
    cpu.cpu.legacy.segments[1].base = 0x2fffc;
    vm.write_bytes(0x2fffc, &0x2345u32.to_le_bytes()).unwrap();
    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0);
    assert!(matches!(
        run.stop(),
        StepStop::Exit(Exit::MmioRead {
            addr: 0x30000,
            size: 2,
            ..
        })
    ));
    assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0, 0x1000));
    cpu.complete_mmio_read(0x100);
    steps(&mut cpu, &vm, 1);
    assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0x100, 0x2345));
    assert!(!cpu.cpu.mmio_retry_active);
    assert_eq!(cpu.cpu.pending_mmio, None);
}

#[test]
fn abandoned_mmio_answer_is_cleared_for_a_new_cs_or_rip_owner() {
    // Decision-16 §§1,3: zero budget preserves completion storage; positive
    // execution discards an abandoned CS:RIP owner's answer before any access.
    for change_cs in [false, true] {
        let (mut vm, mut cpu) = setup(b"\x8b\x07\x8b\x07");
        vm.unmap(0, 0x40000).unwrap();
        vm.map(0, 0x30000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x30000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        vm.write_bytes(0x1000, b"\x8b\x07\x8b\x07").unwrap();
        cpu.cpu.legacy.segments[1].base = 0x30000;
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(matches!(run.stop(), StepStop::Exit(Exit::MmioRead { .. })));
        cpu.complete_mmio_read(0xdead);
        if change_cs {
            cpu.cpu.cs = 0x100; // Hidden base is retained (decision-18 §2).
        } else {
            cpu.cpu.rip += 2;
        }
        cpu.cpu.legacy.segments[1].base = 0x2000;
        vm.write_bytes(0x2000, &0x1234u16.to_le_bytes()).unwrap();
        let before = format!("{:?}", cpu.cpu);
        steps(&mut cpu, &vm, 0);
        assert_eq!(format!("{:?}", cpu.cpu), before);
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.gpr[0], 0x1234);
        assert_eq!(cpu.cpu.pending_mmio, None);
        assert!(!cpu.cpu.mmio_retry_active);
    }
}

#[test]
fn lss_accepts_irq_at_its_next_boundary_while_mov_and_pop_ss_inhibit() {
    // SDM Vol 3A §7.8.3; decision-16 §4: only MOV/POP SS inhibit;
    // LSS atomically loads both parts and does not create that shadow.
    for (code, shadow) in [
        (&b"\x0f\xb2\x06\x00\x20\x40"[..], false),
        (&b"\x8e\xd0\x40"[..], true),
        (&b"\x17\x40"[..], true),
    ] {
        let (vm, mut cpu) = setup(code);
        vm.write_bytes(0x2000, b"\x34\x12\x00\x00").unwrap();
        vm.write_bytes(0x8000, b"\x00\x00").unwrap();
        vm.write_bytes(0x80, b"\x00\x18\x00\x00").unwrap();
        vm.write_bytes(0x1800, b"\xf4").unwrap();
        cpu.cpu.flags.if_ = true;
        steps(&mut cpu, &vm, 1);
        cpu.inject_irq(0x20);
        let ax = cpu.cpu.gpr[0];
        if shadow {
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.gpr[0], ax + 1);
        }
        let rip = cpu.cpu.rip;
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 1);
        assert!(matches!(run.stop(), StepStop::Exit(Exit::Hlt)));
        assert_eq!(cpu.cpu.rip, 0x1801);
        assert_eq!(value(&vm, cpu.cpu.gpr[4], 2), rip);
        assert_eq!(cpu.cpu.gpr[0], ax + u64::from(shadow));
        assert!(!cpu.has_pending_irq());
    }
}

#[test]
fn untaken_jcc_and_loop_complete_before_the_next_fetch_limit_fault() {
    // SDM Vol 2A §3.3 Jcc/LOOP: validate only a taken target. 80386 PRM
    // §14.7 item 8, p. 14-6 (page image checked): sequential overflow faults
    // on the subsequent fetch, after the instruction and decrement complete.
    for (code, zf, count, remaining) in [
        (&b"\x75\x00"[..], true, 2, 2),
        (&b"\x74\x00"[..], false, 2, 2),
        (&b"\x0f\x85\x00\x00"[..], true, 2, 2),
        (&b"\xe2\x00"[..], false, 1, 0),
        (&b"\xe1\x00"[..], false, 2, 1),
        (&b"\xe0\x00"[..], true, 2, 1),
        (&b"\xe3\x00"[..], false, 1, 1),
        (&b"\x67\xe2\x00"[..], false, 1, 0),
    ] {
        for budget in [1, 2] {
            let (vm, mut cpu) = setup(code);
            cpu.cpu.rip = 0x10000 - code.len() as u64;
            vm.write_bytes(cpu.cpu.rip, code).unwrap();
            cpu.cpu.flags.zf = zf;
            cpu.cpu.gpr[1] = count;
            let run = cpu.run_steps(&vm, budget, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1, "{code:02x?}: {:?}", run.stop());
            assert_eq!(cpu.retired_instructions(), 1);
            assert_eq!(cpu.cpu.rip, 0x10000);
            assert_eq!(cpu.cpu.gpr[1], remaining);
            if budget == 1 {
                assert!(matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)));
                exception(&mut cpu, &vm, 13);
            } else {
                assert!(matches!(
                    run.stop(),
                    StepStop::Exit(Exit::Exception {
                        vector: 13,
                        addr: 0x10000,
                        ..
                    })
                ));
            }
        }
    }
    // Taken targets still fault before committing the branch or decrement.
    for code in [&b"\x75\x00"[..], &b"\xe2\x00"[..]] {
        let (vm, mut cpu) = setup(code);
        cpu.cpu.legacy.segments[0].attributes |= 1 << 14;
        cpu.cpu.rip = 0xfffe;
        cpu.cpu.gpr[1] = 2;
        vm.write_bytes(0xfffe, code).unwrap();
        exception(&mut cpu, &vm, 13);
        assert_eq!(cpu.cpu.rip, 0xfffe);
        assert_eq!(cpu.cpu.gpr[1], 2);
        assert_eq!(cpu.retired_instructions(), 0);
    }
}

#[test]
fn pusha_and_pushad_special_real_stack_values_fault_before_any_push() {
    // SDM Vol 2B §4.3 PUSHA/PUSHAD; 80386 PRM §17, pp. 17-133–134
    // (page images checked): SP/ESP in {7,9,11,13,15} raises #GP.
    for code in [&b"\x60"[..], &b"\x66\x60"[..]] {
        for big in [false, true] {
            for pointer in [7, 9, 11, 13, 15] {
                let (vm, mut cpu) = setup(code);
                if big {
                    cpu.cpu.legacy.segments[3].attributes |= 1 << 14;
                }
                cpu.cpu.gpr[4] = pointer;
                vm.write_bytes(0, &[0xaa; 16]).unwrap();
                let before = format!("{:?}", cpu.cpu);
                exception(&mut cpu, &vm, 13);
                assert_eq!(format!("{:?}", cpu.cpu), before);
                assert_eq!(cpu.retired_instructions(), 0);
                assert_eq!(value(&vm, 0, 8), 0xaaaaaaaaaaaaaaaa);
                assert_eq!(value(&vm, 8, 8), 0xaaaaaaaaaaaaaaaa);
            }
        }
    }
}

#[test]
fn ordinary_completion_clears_rf_but_zero_budget_faults_and_mmio_do_not() {
    // SDM Vol 3B §20.3.1.1; 80386 PRM §12.3.1.1, p. 12-7 (page image
    // checked): ordinary instructions clear RF; incomplete retries preserve it.
    for code in [
        &b"\x90"[..],
        &b"\x40"[..],
        &b"\xeb\x00"[..],
        &b"\x66\x9d"[..],
    ] {
        let (vm, mut cpu) = setup(code);
        // SDM Vol 3B §20.3.1.1: POPFD does not restore RF from the stack;
        // ordinary execution clearing is separate from its flag-load mask.
        vm.write_bytes(0x8000, &0x17002u32.to_le_bytes()).unwrap();
        cpu.cpu.rflags_system = 0x17000;
        steps(&mut cpu, &vm, 0);
        assert_eq!(cpu.cpu.rflags_system, 0x17000);
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.rflags_system, 0x7000);
    }
    for code in [&b"\xf4"[..], &b"\xe6\x20"[..]] {
        let (vm, mut cpu) = setup(code);
        cpu.cpu.rflags_system = 0x10000;
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 1);
        assert!(matches!(
            run.stop(),
            StepStop::Exit(Exit::Hlt | Exit::PortIo { .. })
        ));
        assert_eq!(cpu.cpu.rflags_system, 0);
    }
    let (mut vm, mut cpu) = setup(b"\x8b\x07");
    cpu.cpu.rflags_system = 0x10000;
    cpu.cpu.gpr[3] = 0xffff;
    exception(&mut cpu, &vm, 13);
    assert_eq!(cpu.cpu.rflags_system, 0x10000);
    vm.unmap(0, 0x40000).unwrap();
    vm.map(0, 0x30000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x30000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    vm.write_bytes(0x1000, b"\x8b\x07").unwrap();
    cpu.cpu.gpr[3] = 0;
    cpu.cpu.legacy.segments[1].base = 0x30000;
    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0);
    assert!(matches!(run.stop(), StepStop::Exit(Exit::MmioRead { .. })));
    assert_eq!(cpu.cpu.rflags_system, 0x10000);
    cpu.complete_mmio_read(0x1234);
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.rflags_system, 0);
}

#[test]
fn deferred_system_register_accesses_have_effect_free_typed_refusals() {
    // Decision-18 §4: system-register operations owned by later phases use a
    // typed engine refusal. SDM Vol 2B §4.3 MOV to/from Control/Debug Registers;
    // 80386 PRM §17 MOV to/from Special Registers, pp. 17-111 (page image checked).
    for opcode in [0x21, 0x23, 0x24, 0x26] {
        let registers: &[u8] = match opcode {
            0x21 | 0x23 => &[0, 1, 2, 3, 6, 7],
            _ => &[6, 7],
        };
        for &register in registers {
            let (vm, mut cpu) = setup(&[0x0f, opcode, 0xc0 | (register << 3)]);
            cpu.cpu.gpr[0] = 1;
            cpu.cpu.rflags_system = 0x10000;
            let before = format!("{:?}", cpu.cpu);
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 0);
            assert!(
                matches!(
                    run.stop(),
                    StepStop::Exit(Exit::UnsupportedSystemContext {
                        addr: 0x1000,
                        operation: UnsupportedSystemOperation::LegacyInstruction,
                        ..
                    })
                ),
                "{opcode:02x}/{register}: {:?}",
                run.stop()
            );
            assert_eq!(format!("{:?}", cpu.cpu), before);
            assert_eq!(cpu.retired_instructions(), 0);
        }
    }
}

#[test]
fn descriptor_table_operand_images_and_flags_for_both_sizes() {
    // SDM 325462-092US Vol 2A §3.3 LGDT/LIDT; Vol 2B §4.3 SGDT/SIDT.
    // 80386 PRM §17 pp. 17-94–95, 17-154 (page images checked).
    // Narrow reading: the explicit SGDT/SIDT compatibility paragraphs prescribe
    // zero in byte 5 of the 16-bit store. PRM p. 17-94 says full base for both
    // stores and p. 17-154's description swaps widths; the compatibility rule
    // agrees with the primary SDM and governs this generation-specific image.
    // The 286's undefined/ones byte is not an I386 rule or activated generation.
    for default32 in [false, true] {
        for op32 in [false, true] {
            for idtr in [false, true] {
                let mut code = vec![];
                if default32 {
                    code.push(0x67);
                }
                if op32 != default32 {
                    code.push(0x66);
                }
                code.extend([0x0f, 0x01, if idtr { 0x1e } else { 0x16 }, 0, 0x20]);
                let (vm, mut cpu) = setup(&code);
                if default32 {
                    cpu.cpu.legacy.segments[0].attributes |= 1 << 14;
                }
                cpu.cpu.flags.set_flags16(0x7ed7);
                cpu.cpu.rflags_system = 0x7000;
                let flags = cpu.cpu.flags;
                vm.write_bytes(0x2000, &[0x45, 0x23, 0x78, 0x56, 0x34, 0x92])
                    .unwrap();
                steps(&mut cpu, &vm, 1);
                let table = if idtr {
                    cpu.cpu.legacy.idtr
                } else {
                    cpu.cpu.legacy.gdtr
                };
                assert_eq!(table.limit, 0x2345);
                assert_eq!(table.base, if op32 { 0x92345678 } else { 0x345678 });
                assert_eq!(cpu.cpu.flags, flags);
                assert_eq!(cpu.cpu.rflags_system, 0x7000);
                // Store a nonzero upper base even in the 16-bit case so this tests
                // the store image independently of the preceding load's truncation.
                if idtr {
                    cpu.cpu.legacy.idtr.base = 0x92345678;
                } else {
                    cpu.cpu.legacy.gdtr.base = 0x92345678;
                }
                code.clear();
                if default32 {
                    code.push(0x67);
                }
                if op32 != default32 {
                    code.push(0x66);
                }
                code.extend([0x0f, 0x01, if idtr { 0x0e } else { 0x06 }, 0, 0x20]);
                vm.write_bytes(0x1000, &code).unwrap();
                cpu.cpu.rip = 0x1000;
                vm.write_bytes(0x2000, &[0xaa; 8]).unwrap();
                steps(&mut cpu, &vm, 1);
                assert_eq!(
                    value(&vm, 0x2000, 8),
                    if op32 {
                        0xaaaa923456782345
                    } else {
                        0xaaaa003456782345
                    }
                );
                assert_eq!(cpu.cpu.flags, flags);
                assert_eq!(cpu.cpu.rflags_system, 0x7000);
            }
        }
    }
}

#[test]
fn table_operands_check_all_six_bytes_before_effects_and_retry_mmio() {
    // SDM Vol 2A §3.3 LGDT/LIDT; Vol 2B §4.3 SGDT/SIDT:
    // memory-only, six bytes, unchanged flags; segment-limit #GP/#SS.
    for prefix in [&b""[..], &b"\x66"[..]] {
        for extension in [0, 1, 2, 3] {
            for ss in [false, true] {
                let mut code = prefix.to_vec();
                if ss {
                    code.push(0x36);
                }
                code.extend([0x0f, 0x01, 0x06 | (extension << 3), 0xfb, 0xff]);
                let (vm, mut cpu) = setup(&code);
                vm.write_bytes(0xfffb, &[0xaa; 6]).unwrap();
                let before = format!("{:?}", cpu.cpu);
                exception(&mut cpu, &vm, if ss { 12 } else { 13 });
                assert_eq!(format!("{:?}", cpu.cpu), before);
                assert_eq!(value(&vm, 0xfffb, 6), 0xaaaaaaaaaaaa);
            }
        }
    }
    for extension in [0, 1, 2, 3] {
        let (mut vm, mut cpu) = setup(&[0x66, 0x0f, 0x01, 0x06 | (extension << 3), 0xfe, 0x2f]);
        vm.unmap(0, 0x40000).unwrap();
        vm.map(0, 0x3000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(
            0x1000,
            &[0x66, 0x0f, 0x01, 0x06 | (extension << 3), 0xfe, 0x2f],
        )
        .unwrap();
        vm.write_bytes(0x2ffe, &[0x78, 0x56]).unwrap();
        let before = cpu.cpu.legacy;
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(matches!(
            run.stop(),
            StepStop::Exit(Exit::UnmappedMemory { addr: 0x3000, .. })
        ));
        assert_eq!(cpu.cpu.legacy, before);
        assert_eq!(value(&vm, 0x2ffe, 2), 0x5678);
        vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        if extension >= 2 {
            assert!(matches!(
                run.stop(),
                StepStop::Exit(Exit::MmioRead {
                    addr: 0x3000,
                    size: 4,
                    ..
                })
            ));
            assert_eq!(cpu.cpu.legacy, before);
            cpu.complete_mmio_read(0x92345678);
            steps(&mut cpu, &vm, 1);
            let table = if extension == 3 {
                cpu.cpu.legacy.idtr
            } else {
                cpu.cpu.legacy.gdtr
            };
            assert_eq!((table.base, table.limit), (0x92345678, 0x5678));
        } else {
            assert!(matches!(
                run.stop(),
                StepStop::Exit(Exit::MmioWrite {
                    addr: 0x3000,
                    size: 4,
                    ..
                })
            ));
            cpu.complete_mmio_write();
            steps(&mut cpu, &vm, 1);
        }
    }
}

#[test]
fn control_registers_have_386_masks_fixed_width_and_real_privilege() {
    // SDM Vol 3A §2.5; Vol 2B §4.3 MOV CR (reserved CR0 bits ignored;
    // arithmetic flags undefined). 80386 PRM §4.1.3 Fig. 4-2 p. 4-3;
    // §17 p. 17-111 (page images checked): PG + low five CR0 bits, full
    // CR2, CR3[31:12], fixed 32-bit transfers. Deterministic undefined flags
    // retain their input; this is an engine policy, not an architectural value.
    for prefix in [&b""[..], &b"\x66"[..]] {
        for (register, source, expected) in [
            (0, 0x7ffffffe, 0x1e),
            (0, 0, 0),
            (2, 0x87654321, 0x87654321),
            (3, 0x12345fff, 0x12345000),
        ] {
            let mut code = prefix.to_vec();
            code.extend([0x0f, 0x22, 0xc0 | (register << 3)]);
            code.extend(prefix);
            code.extend([0x0f, 0x20, 0xc3 | (register << 3)]);
            let (vm, mut cpu) = setup(&code);
            // PRM §11.1.1 p. 11-1 (scan checked): MOV can also clear ET.
            if register == 0 && source == 0 {
                cpu.cpu.legacy.cr0 = 0x1e;
            }
            cpu.cpu.gpr[0] = 0xdead00000000 | source;
            cpu.cpu.gpr[3] = u64::MAX;
            cpu.cpu.flags.set_flags16(0x7ed7);
            cpu.cpu.rflags_system = 0x7000;
            // A real selector's low bits are not CPL; real mode executes at 0.
            cpu.cpu.cs = 3;
            let flags = cpu.cpu.flags;
            steps(&mut cpu, &vm, 2);
            assert_eq!(cpu.cpu.gpr[3], expected);
            assert_eq!(cpu.cpu.flags, flags);
            assert_eq!(cpu.cpu.rflags_system, 0x7000);
        }
    }
}

#[test]
fn msw_and_clts_use_only_their_documented_bits_and_preserve_flags() {
    // SDM Vol 2A §3.3 LMSW/CLTS; Vol 2B §4.3 SMSW. 80386 PRM §17
    // pp. 17-48, 17-99, 17-160 (page images checked). SMSW r32 upper
    // half is undefined; retaining it is our policy. Memory is always m16.
    for prefix in [&b""[..], &b"\x66"[..]] {
        for memory in [false, true] {
            let mut code = prefix.to_vec();
            code.extend(if memory {
                &b"\x0f\x01\x36\x00\x20"[..]
            } else {
                &b"\x0f\x01\xf0"[..]
            });
            code.extend(prefix);
            code.extend(b"\x0f\x01\xe3"); // SMSW (E)BX
            code.extend(prefix);
            code.extend(b"\x0f\x01\x26\x02\x20"); // SMSW m16
            code.extend(b"\x0f\x06");
            let (vm, mut cpu) = setup(&code);
            cpu.cpu.legacy.cr0 = 0x10;
            cpu.cpu.gpr[0] = 0xfffe;
            cpu.cpu.gpr[3] = 0xfeedbeef12345678;
            cpu.cpu.flags.set_flags16(0x7ed7);
            let flags = cpu.cpu.flags;
            vm.write_bytes(0x2000, &[0xfe, 0xff, 0xaa, 0xaa, 0xaa, 0xaa])
                .unwrap();
            steps(&mut cpu, &vm, 3);
            assert_eq!(cpu.cpu.legacy.cr0, 0x1e);
            assert_eq!(
                cpu.cpu.gpr[3],
                if prefix.is_empty() {
                    0xfeedbeef1234001e
                } else {
                    0x1234001e
                }
            );
            assert_eq!(value(&vm, 0x2002, 4), 0xaaaa001e);
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.legacy.cr0, 0x16);
            assert_eq!(cpu.cpu.flags, flags);
        }
    }
}

#[test]
fn enabling_pg_refuses_atomically_but_pg_without_pe_faults() {
    // SDM Vol 3A §2.5; decision-18 §4: legality precedes an engine refusal.
    for prefix in [&b""[..], &b"\x66"[..]] {
        for (code, value, gp) in [
            (&b"\x0f\x22\xc0"[..], 0x80000001, false),
            (&b"\x0f\x22\xc0"[..], 0x80000000, true),
        ] {
            let mut bytes = prefix.to_vec();
            bytes.extend(code);
            bytes.push(0x43);
            let (vm, mut cpu) = setup(&bytes);
            cpu.cpu.gpr[0] = value;
            cpu.cpu.rflags_system = 0x10000;
            vm.write_bytes(0x2000, &[0xff, 0xff]).unwrap();
            let before = format!("{:?}", cpu.cpu);
            let run = cpu.run_steps(&vm, 2, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 0);
            if gp {
                assert!(matches!(
                    run.stop(),
                    StepStop::Exit(Exit::Exception { vector: 13, .. })
                ));
            } else {
                assert!(matches!(
                    run.stop(),
                    StepStop::Exit(Exit::UnsupportedSystemContext {
                        operation: UnsupportedSystemOperation::LegacyExecution,
                        ..
                    })
                ));
            }
            assert_eq!(format!("{:?}", cpu.cpu), before);
            assert_eq!(cpu.retired_instructions(), 0);
        }
    }
}

#[test]
fn illegal_system_encodings_and_word_memory_limits_fault_without_effects() {
    // SDM Vol 2A §3.3 LGDT/LIDT/LMSW/CLTS; Vol 2B §4.3
    // SGDT/SIDT/SMSW/MOV CR: LOCK is #UD, table operands are memory-only.
    // 80386 PRM §4.1.3 p. 4-3: CR1 reserved, no CR4 on I386.
    for code in [
        &b"\xf0\x0f\x06"[..],
        &b"\xf0\x0f\x22\xc0"[..],
        &b"\xf0\x0f\x20\xc0"[..],
        &b"\xf0\x0f\x01\xe0"[..],
        &b"\xf0\x0f\x01\xf0"[..],
        &b"\xf0\x0f\x01\x16\x00\x20"[..],
        &b"\xf0\x0f\x01\x06\x00\x20"[..],
        &b"\x0f\x20\xc8"[..],
        &b"\x0f\x22\xc8"[..],
        &b"\x0f\x20\xe0"[..],
        &b"\x0f\x22\xe0"[..],
        &b"\x0f\x01\xcb"[..],
        &b"\x0f\x01\xdb"[..],
    ] {
        let (vm, mut cpu) = setup(code);
        let before = format!("{:?}", cpu.cpu);
        exception(&mut cpu, &vm, 6);
        assert_eq!(format!("{:?}", cpu.cpu), before);
    }
    for prefix in [&b""[..], &b"\x66"[..]] {
        for extension in 0..4 {
            for register in 0..8 {
                let mut code = prefix.to_vec();
                code.extend([0x0f, 0x01, 0xc0 | (extension << 3) | register]);
                let (vm, mut cpu) = setup(&code);
                exception(&mut cpu, &vm, 6);
            }
        }
    }
    for code in [
        &b"\x0f\x01\x26\xff\xff"[..],
        &b"\x66\x0f\x01\x36\xff\xff"[..],
    ] {
        let (vm, mut cpu) = setup(code);
        exception(&mut cpu, &vm, 13);
    }
}

#[test]
fn real_delivery_uses_idtr_base_limit_and_exact_six_byte_frame() {
    // SDM Vol 3B §23.1.4; Vol 2A §3.3 INT real-address operation:
    // four-byte IDTR-relative entry, inclusive limit, word FLAGS/CS/IP frame.
    // 80386 PRM §14.3 p.14-3, §14.6 Table 14-1 p.14-5, Table 14-2 p.14-8:
    // an entry beyond IDTR.limit faults with vector 8 on I386, diverging from
    // the current SDM Vol 2A §3.3 INT's #GP. Delivery faults remain outward.
    for (code, vector, saved) in [
        (&b"\xcd\x20"[..], 0x20, 0x1002),
        (&b"\x66\xcd\x20"[..], 0x20, 0x1003),
        (&b"\xcc"[..], 3, 0x1001),
        (&b"\xce"[..], 4, 0x1001),
        (&b"\x90"[..], 0x20, 0x1000),
    ] {
        for short_limit in [false, true] {
            let (vm, mut cpu) = setup(code);
            cpu.cpu.legacy.idtr.base = 0x2000;
            cpu.cpu.legacy.segments[1].base = 0x9000;
            cpu.cpu.legacy.idtr.limit = vector * 4 + if short_limit { 2 } else { 3 };
            cpu.cpu.flags.set_flags16(0x0ed7);
            cpu.cpu.rflags_system = 0x7000;
            vm.write_bytes(0x2000 + u64::from(vector) * 4, b"\x00\x02\x00\x01")
                .unwrap();
            vm.write_bytes(0x1200, b"\xf4").unwrap();
            vm.write_bytes(0x7ffa, &[0xaa; 6]).unwrap();
            let irq = code == b"\x90";
            if irq {
                cpu.inject_irq(vector as u8);
            }
            let before = format!("{:?}", cpu.cpu);
            steps(&mut cpu, &vm, 0);
            assert_eq!(format!("{:?}", cpu.cpu), before);
            if short_limit {
                exception(&mut cpu, &vm, 8);
                assert_eq!(format!("{:?}", cpu.cpu), before);
                assert_eq!(value(&vm, 0x7ffa, 6), 0xaaaaaaaaaaaa);
                assert_eq!(cpu.has_pending_irq(), irq);
            } else {
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 1);
                assert_eq!(cpu.retired_instructions(), 1);
                assert_eq!(
                    (cpu.cpu.cs, cpu.cpu.rip),
                    (0x100, if irq { 0x201 } else { 0x200 })
                );
                assert_eq!(cpu.cpu.legacy.segments[0].base, 0x1000);
                assert_eq!(cpu.cpu.gpr[4], 0x7ffa);
                assert_eq!(value(&vm, 0x7ffa, 6), (0x7ed7 << 32) | saved);
                assert!(!cpu.cpu.flags.if_);
                assert_eq!(cpu.cpu.rflags_system, 0x7000);
                assert!(!cpu.has_pending_irq());
            }
        }
    }
    let (vm, mut cpu) = setup(b"\xce\x90");
    steps(&mut cpu, &vm, 1); // INTO with OF=0 does not access even a short IDTR.
    assert_eq!(cpu.cpu.rip, 0x1001);
}

#[test]
fn accepted_irq_mmio_retry_finishes_delivery_before_the_interrupted_instruction() {
    // Decision-16 §§1,3; SDM Vol 3B §23.1.4. A device answer belongs to
    // delivery, and neither accepting nor retrying delivery consumes a step.
    let (mut vm, mut cpu) = setup(b"\x43");
    vm.unmap(0, 0x40000).unwrap();
    vm.map(0, 0x30000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x30000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    vm.write_bytes(0x1000, b"\x43").unwrap();
    vm.write_bytes(0x1800, b"\xf4").unwrap();
    cpu.cpu.legacy.idtr.base = 0x30000;
    cpu.cpu.flags.if_ = true;
    cpu.inject_irq(0x20);
    for (address, answer) in [(0x30080, 0x1800), (0x30082, 0)] {
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::MmioRead { addr, size: 2, .. }) if *addr == address)
        );
        assert_eq!(cpu.cpu.rip, 0x1000);
        assert_eq!(cpu.cpu.gpr[3], 0);
        assert_eq!(cpu.cpu.gpr[4], 0x8000);
        assert!(cpu.has_pending_irq());
        cpu.cpu.flags.if_ = false; // Accepted transfer delivery must still finish.
        cpu.complete_mmio_read(answer);
        steps(&mut cpu, &vm, 0);
    }
    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 1);
    assert!(matches!(run.stop(), StepStop::Exit(Exit::Hlt)));
    assert_eq!(cpu.cpu.gpr[3], 0);
    assert_eq!(value(&vm, 0x7ffa, 2), 0x1000);
    assert_eq!(value(&vm, 0x7ffe, 2), 2);
    assert!(!cpu.has_pending_irq());
}

#[test]
fn accepted_irq_unmapped_retry_finishes_delivery_with_if_cleared() {
    // Decision-16 §3: an unmapped transfer preserves accepted event delivery,
    // just like MMIO; changing IF cannot split it from its continuation.
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let (mut vm, mut cpu) = setup_model(b"\x43", generation);
        vm.unmap(0, 0x40000).unwrap();
        vm.map(0, 0x30000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, b"\x43").unwrap();
        vm.write_bytes(0x1800, b"\xf4").unwrap();
        cpu.cpu.legacy.idtr.base = 0x30000;
        cpu.cpu.flags.if_ = true;
        cpu.inject_irq(0x20);
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(matches!(
            run.stop(),
            StepStop::Exit(Exit::UnmappedMemory { addr: 0x30080, .. })
        ));
        cpu.cpu.flags.if_ = false;
        vm.map(0x30000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
        vm.write_bytes(0x30080, b"\x00\x18\x00\x00").unwrap();
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 1);
        assert!(matches!(run.stop(), StepStop::Exit(Exit::Hlt)));
        assert_eq!((cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[3]), (0, 0x1801, 0));
        assert_eq!(value(&vm, 0x7ffa, 2), 0x1000);
        assert_eq!(value(&vm, 0x7ffe, 2), 2);
        assert!(!cpu.has_pending_irq());
    }
}

#[test]
fn application_real16_delivery_ignores_all_legacy_table_and_control_storage() {
    let mut vm = Vm::with_backend(VmConfig::flat(0x40000), Box::new(NoBackend));
    vm.set_cpu_mode(CpuMode::Real16);
    vm.map(0, 0x40000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, b"\xcd\x20").unwrap();
    vm.write_bytes(0x80, b"\x00\x18\x00\x00").unwrap();
    vm.write_bytes(0x2080, b"\x00\x28\x00\x00").unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[4] = 0x8000;
    cpu.cpu.legacy.idtr.base = 0x2000;
    cpu.cpu.legacy.idtr.limit = 0;
    cpu.cpu.legacy.cr0 = u32::MAX;
    cpu.cpu.legacy.cpl = 255;
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.rip, 0x1800);
    assert_eq!(cpu.cpu.gpr[4], 0x7ffa);
    assert_eq!(value(&vm, 0x7ffa, 2), 0x1002);
}

#[test]
fn application_real16_ir_matches_frozen_parent_snapshots() {
    // Measured at parent 9958575 with the same Rust 1.95 toolchain. Real16
    // never materializes; compare its actual application IR, including metadata.
    let mut snapshot = String::new();
    for (name, code) in [
        ("alu", &b"\x01\xd8\x46\x31\xc2\x75\xfa"[..]),
        ("stack", &b"\x50\x5b\xf4"[..]),
        ("rep", &b"\xf3\xa5\xf4"[..]),
        ("fs", &b"\x64\x8b\x07\xf4"[..]),
        ("int", &b"\xcd\x20"[..]),
    ] {
        let mut vm = Vm::with_backend(VmConfig::flat(0x100000), Box::new(NoBackend));
        vm.set_cpu_mode(CpuMode::Real16);
        vm.map(0, 0x100000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, code).unwrap();
        let ir = x86jit_core::lift::lift_block(
            &vm.mem,
            x86jit_core::lift::FetchAddr {
                pa: 0x1000,
                ip: 0x1000,
            },
            CpuMode::Real16,
        )
        .unwrap();
        snapshot.push_str(&format!("{name}\n{ir:?}\n"));
    }
    assert_eq!(snapshot, include_str!("fixtures/real16_parent_ir.snap"));
}

mod single_step {
    use super::*;

    const MODELS: [CpuGeneration; 2] = [CpuGeneration::I286, CpuGeneration::I386];
    fn vector(vm: &Vm, number: u8, ip: u16, code: &[u8]) {
        vm.write_bytes(u64::from(number) * 4, &ip.to_le_bytes())
            .unwrap();
        vm.write_bytes(u64::from(number) * 4 + 2, &[0, 0]).unwrap();
        vm.write_bytes(u64::from(ip), code).unwrap();
    }
    fn traced(code: &[u8], generation: CpuGeneration) -> (Vm, Vcpu) {
        let (vm, cpu) = setup_model(code, generation);
        vector(&vm, 1, 0x1800, b"\xcf");
        (vm, cpu)
    }
    fn frame(vm: &Vm, cpu: &Vcpu, ip: u16, flags: u16) {
        assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0, 0x1800));
        let sp = cpu.cpu.gpr[4] & 0xffff;
        assert_eq!(value(vm, sp, 2), u64::from(ip));
        assert_eq!(value(vm, sp + 2, 2), 0);
        assert_eq!(value(vm, sp + 4, 2), u64::from(flags));
        assert_eq!(cpu.cpu.rflags_system & 0x100, 0);
        assert!(!cpu.cpu.flags.if_);
    }

    #[test]
    fn probe_logs_pop_ds_without_inhibition_and_is_deterministic() {
        // I286 PRM §9.7.1 p. 9-14, Appendix B POP p. B-83;
        // 80386 PRM §12.3.1.4 p. 12-8, §17 POP pp. 17-125–126:
        // POP DS has no SS inhibition. Both models trap immediately after it.
        let code = b"\x9c\x9c\x58\x0d\x00\x01\x50\x31\xc0\x9d\x90\x1e\x1f\x90\x9d\xf4";
        for generation in MODELS {
            let mut previous = None;
            for _ in 0..3 {
                let (vm, mut cpu) = traced(code, generation);
                // Preserve AX/BP; append the saved IP to ES:DI; IRET restores TF.
                vector(
                    &vm,
                    1,
                    0x1800,
                    b"\x55\x89\xe5\x50\x8b\x46\x02\xab\x58\x5d\xcf",
                );
                cpu.cpu.gpr[7] = 0x6000;
                let run = cpu.run_steps(&vm, 100, StepOptions::default()).unwrap();
                assert!(
                    matches!(run.stop(), StepStop::Exit(Exit::Hlt)),
                    "{:?}",
                    run.stop()
                );
                let ips: Vec<_> = (0..5).map(|i| value(&vm, 0x6000 + i * 2, 2)).collect();
                assert_eq!(ips, [0x100b, 0x100c, 0x100d, 0x100e, 0x100f]);
                // 13 probe instructions + five eight-instruction handlers.
                assert_eq!((run.steps(), cpu.retired_instructions()), (53, 53));
                assert_eq!(cpu.cpu.gpr[4], 0x8000);
                assert_eq!(cpu.cpu.rflags_system & 0x100, 0);
                let image = (format!("{:?}", cpu.cpu), ips, run.steps());
                if let Some(previous) = &previous {
                    assert_eq!(&image, previous);
                }
                previous = Some(image);
            }
        }
    }

    #[test]
    fn popf_and_iret_set_tf_with_one_instruction_delay() {
        // I286 PRM §2.3 FLAGS p. 2-15, §9.7.1 p. 9-14;
        // 80386 PRM §12.3.1.4 p. 12-8: TF at instruction entry controls trapping.
        for generation in MODELS {
            for (code, size, iret) in [
                (b"\x9d\x90".as_slice(), 2, false),
                (b"\xcf", 2, true),
                (b"\x66\x9d\x90", 4, false),
                (b"\x66\xcf", 4, true),
            ] {
                if size == 4 && generation == CpuGeneration::I286 {
                    continue;
                }
                let (vm, mut cpu) = traced(code, generation);
                if iret {
                    for (i, word) in [0x1200u32, 0, 0x102].into_iter().enumerate() {
                        vm.write_bytes(0x8000 + (i * size) as u64, &word.to_le_bytes()[..size])
                            .unwrap();
                    }
                    vm.write_bytes(0x1200, b"\x90").unwrap();
                } else {
                    vm.write_bytes(0x8000, &0x102u32.to_le_bytes()[..size])
                        .unwrap();
                }
                let next = if iret {
                    0x1200
                } else {
                    0x1000 + code.len() as u16 - 1
                };
                steps(&mut cpu, &vm, 1);
                assert_eq!(cpu.cpu.rip, u64::from(next));
                assert_eq!(cpu.cpu.rflags_system & 0x100, 0x100);
                steps(&mut cpu, &vm, 1);
                frame(&vm, &cpu, next + 1, 0x102);
                assert_eq!(cpu.retired_instructions(), 2);
            }
        }
    }

    #[test]
    fn rep_traps_each_iteration_but_retires_only_on_completion() {
        // I286 PRM Appendix B REP pp. B-92–93; 80386 PRM §17 REP
        // pp. 17-139–140: interrupt window per iteration, updated CX/SI/DI.
        // Together with §9.7.1 / §12.3.1.4 this gives one TF trap per iteration.
        for generation in MODELS {
            for count in [0, 3] {
                let (vm, mut cpu) = traced(b"\xf3\xa4", generation);
                cpu.cpu.rflags_system = 0x100;
                cpu.cpu.gpr[1] = count;
                cpu.cpu.gpr[6] = 0x4000;
                cpu.cpu.gpr[7] = 0x5000;
                vm.write_bytes(0x4000, b"abc").unwrap();
                for left in (0..count.max(1)).rev() {
                    steps(&mut cpu, &vm, 1);
                    let done = left == 0;
                    frame(&vm, &cpu, if done { 0x1002 } else { 0x1000 }, 0x102);
                    assert_eq!(cpu.cpu.gpr[1], left);
                    assert_eq!(
                        cpu.retired_instructions(),
                        count.max(1) - left - u64::from(!done)
                    );
                    if !done {
                        steps(&mut cpu, &vm, 1);
                    } // IRET is a retired instruction.
                }
                assert_eq!(cpu.cpu.gpr[6], 0x4000 + count);
                assert_eq!(cpu.cpu.gpr[7], 0x5000 + count);
                if count != 0 {
                    assert_eq!(value(&vm, 0x5000, 3), 0x636261);
                }
            }
        }
    }

    #[test]
    fn mov_pop_ss_suppress_only_the_load_boundary() {
        // I286 PRM Appendix B MOV p. B-73, POP p. B-83;
        // 80386 PRM §9.2.4 pp. 9-3–4: SS load, then successor, then #DB.
        for generation in MODELS {
            for code in [b"\x8e\xd0\x90".as_slice(), b"\x17\x90"] {
                let (vm, mut cpu) = traced(code, generation);
                cpu.cpu.rflags_system = 0x100;
                vm.write_bytes(0x8000, &[0, 0]).unwrap();
                steps(&mut cpu, &vm, 1);
                assert_eq!(cpu.cpu.rip, 0x1000 + code.len() as u64 - 1);
                assert_eq!(cpu.cpu.rflags_system, 0x100);
                steps(&mut cpu, &vm, 1);
                frame(&vm, &cpu, 0x1000 + code.len() as u16, 0x102);
                assert_eq!(cpu.retired_instructions(), 2);
            }
        }
    }

    #[test]
    fn data_segment_loads_and_lss_do_not_suppress_tf() {
        // I286 PRM Appendix B MOV p. B-73, POP p. B-83, LDS/LES pp. B-61–62;
        // 80386 PRM §17 MOV p. 17-109, POP p. 17-126, LxS pp. 17-96–97;
        // SDM Vol 3A §7.8.3 p. 7-9: LSS does not inhibit events.
        for generation in MODELS {
            let mut forms = vec![
                vec![0x8e, 0xd8],
                vec![0x8e, 0xc0],
                vec![0x1f],
                vec![0x07],
                vec![0xc5, 0x1e, 0, 0x40],
                vec![0xc4, 0x1e, 0, 0x40],
            ];
            if generation == CpuGeneration::I386 {
                forms.extend([
                    vec![0x8e, 0xe0],
                    vec![0x8e, 0xe8],
                    vec![0x0f, 0xa1],
                    vec![0x0f, 0xa9],
                    vec![0x0f, 0xb2, 0x1e, 0, 0x40],
                    vec![0x0f, 0xb4, 0x1e, 0, 0x40],
                    vec![0x0f, 0xb5, 0x1e, 0, 0x40],
                ]);
            }
            for code in forms {
                let (vm, mut cpu) = traced(&code, generation);
                cpu.cpu.rflags_system = 0x100;
                vm.write_bytes(0x8000, &[0, 0]).unwrap();
                vm.write_bytes(0x4000, &[0, 0, 0, 0]).unwrap();
                steps(&mut cpu, &vm, 1);
                frame(&vm, &cpu, 0x1000 + code.len() as u16, 0x102);
            }
        }
    }

    #[test]
    fn tf_precedes_irq_and_sti_does_not_mask_debug() {
        // I286 PRM Table 5-1 p. 5-4, §9.7.1 p. 9-14;
        // 80386 PRM Table 9-2 p. 9-4, §12.3.1.4 p. 12-8:
        // debug wins over INTR. Real delivery clears IF, leaving INTR pending.
        for generation in MODELS {
            let (vm, mut cpu) = traced(b"\xfb\x90", generation);
            vector(&vm, 0x20, 0x1900, b"\xf4");
            cpu.cpu.rflags_system = 0x100;
            cpu.inject_irq(0x20);
            steps(&mut cpu, &vm, 1);
            frame(&vm, &cpu, 0x1001, 0x302);
            assert!(cpu.has_pending_irq());
            steps(&mut cpu, &vm, 1); // IRET restores TF/IF, no immediate trap.
            assert_eq!(cpu.cpu.rip, 0x1001);
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(matches!(run.stop(), StepStop::Exit(Exit::Hlt)));
            assert_eq!(cpu.cpu.rip, 0x1901);
            assert!(!cpu.has_pending_irq());
            assert_eq!(value(&vm, 0x7ffa, 2), 0x1001);
            assert_eq!(value(&vm, 0x7ffe, 2), 0x302);
            assert_eq!(cpu.retired_instructions(), 3);
        }
    }

    #[test]
    fn faults_precede_tf_but_untaken_into_and_hlt_trap() {
        // I286 PRM Table 5-1 p. 5-4, §9.7.1 p. 9-14;
        // 80386 PRM Table 9-2 p. 9-4, §12.3.1.4 p. 12-8.
        for generation in MODELS {
            let (vm, mut cpu) = traced(b"\xf7\xf0", generation); // DIV AX, AX=0
            cpu.cpu.rflags_system = 0x100;
            exception(&mut cpu, &vm, 0);
            assert_eq!(cpu.retired_instructions(), 0);
            assert_eq!(cpu.cpu.gpr[4], 0x8000);
            assert_eq!(cpu.cpu.rflags_system, 0x100);
            vm.write_bytes(0x1000, b"\x90").unwrap();
            steps(&mut cpu, &vm, 1);
            frame(&vm, &cpu, 0x1001, 0x102);
            for code in [b"\xce", b"\xf4"] {
                let (vm, mut cpu) = traced(code, generation);
                cpu.cpu.rflags_system = 0x100;
                steps(&mut cpu, &vm, 1); // untaken INTO / completed HLT
                frame(&vm, &cpu, 0x1001, 0x102);
            }
        }
    }

    #[test]
    fn taken_int_int3_into_save_tf_and_stop_stepping_on_both_models() {
        // I286 PRM §9.7.1 p. 9-14, Appendix B INT/INTO p. B-48;
        // 80386 PRM §12.3.1.4 p. 12-8, §17 INT/INTO pp. 17-69, 17-73:
        // save TF=1 and the successor, clear live TF, no #DB on entry or
        // after the first handler instruction. IRET restores delayed stepping.
        for generation in MODELS {
            for (code, number) in [
                (b"\xcd\x20".as_slice(), 0x20),
                (b"\xcd\x03", 3),
                (b"\xcc", 3),
                (b"\xce", 4),
            ] {
                let mut program = code.to_vec();
                program.push(0x90);
                let (vm, mut cpu) = traced(&program, generation);
                vector(&vm, number, 0x1900, b"\x43\xcf"); // INC BX; IRET
                cpu.cpu.rflags_system = 0x100;
                cpu.cpu.flags.of = true;
                cpu.cpu.flags.if_ = true;
                let successor = 0x1000 + code.len() as u64;
                steps(&mut cpu, &vm, 1);
                assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0, 0x1900));
                assert_eq!(cpu.cpu.gpr[4], 0x7ffa);
                assert_eq!(value(&vm, 0x7ffa, 2), successor);
                assert_eq!(value(&vm, 0x7ffc, 2), 0);
                assert_eq!(value(&vm, 0x7ffe, 2), 0xb02); // OF, IF, TF
                assert_eq!(cpu.cpu.rflags_system & 0x100, 0);
                assert!(!cpu.cpu.flags.if_);
                assert_eq!(cpu.cpu.gpr[3], 0);
                steps(&mut cpu, &vm, 1);
                assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0, 0x1901));
                assert_eq!(cpu.cpu.gpr[3], 1);
                assert_eq!(cpu.cpu.gpr[4], 0x7ffa);
                assert_eq!(cpu.cpu.rflags_system & 0x100, 0);
                steps(&mut cpu, &vm, 1); // IRET restores TF without trapping.
                assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0, successor));
                assert_eq!(cpu.cpu.gpr[4], 0x8000);
                assert_eq!(cpu.cpu.rflags_system & 0x100, 0x100);
                assert!(cpu.cpu.flags.if_);
                steps(&mut cpu, &vm, 1);
                frame(&vm, &cpu, successor as u16 + 1, 0xb02);
                assert_eq!(cpu.retired_instructions(), 4);
            }
        }
    }

    #[test]
    fn irq_delivery_fault_rechecks_if_before_retry() {
        // I286 PRM §9.2; 80386 PRM §9.2.2 p. 9-3:
        // IF masks INTR. An architectural delivery fault abandons acceptance;
        // a still-pending IRQ must consult the embedder's updated IF.
        // Delivery faults: I286 PRM Appendix B MOV p. B-74; SDM Vol 3B
        // §23.1.4; 80386 PRM §14.6 Tables 14-1/14-2 pp. 14-5, 14-8.
        for generation in MODELS {
            for short_idtr in [false, true] {
                let (vm, mut cpu) = traced(b"\x90", generation);
                vector(&vm, 0x20, 0x1900, b"\xf4");
                cpu.cpu.flags.if_ = true;
                cpu.inject_irq(0x20);
                let fault = if short_idtr {
                    cpu.cpu.legacy.idtr.limit = 0x82;
                    if generation == CpuGeneration::I386 {
                        8
                    } else {
                        13
                    }
                } else {
                    cpu.cpu.legacy.segments[3].limit = 0x7ff0;
                    if generation == CpuGeneration::I386 {
                        12
                    } else {
                        13
                    }
                };
                exception(&mut cpu, &vm, fault);
                assert_eq!(
                    (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
                    (0, 0x1000, 0x8000)
                );
                assert!(cpu.has_pending_irq());
                cpu.cpu.flags.if_ = false;
                cpu.cpu.legacy.idtr.limit = 0x3ff;
                cpu.cpu.legacy.segments[3].limit = 0xffff;
                steps(&mut cpu, &vm, 1);
                assert_eq!(
                    (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
                    (0, 0x1001, 0x8000)
                );
                assert!(cpu.has_pending_irq());
                cpu.cpu.flags.if_ = true;
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert!(matches!(run.stop(), StepStop::Exit(Exit::Hlt)));
                assert_eq!(cpu.cpu.rip, 0x1901);
                assert_eq!(value(&vm, 0x7ffa, 2), 0x1001);
                assert!(!cpu.has_pending_irq());
            }
        }
    }

    #[test]
    fn port_handoffs_deliver_tf_after_completion() {
        // I286 PRM Appendix B REP p. B-93; 80386 PRM §17 REP p. 17-140:
        // an INS element includes its destination store before the TF window.
        // Port completion and vector delivery cost no instruction unit.
        for generation in MODELS {
            for ss in [false, true] {
                let code = if ss {
                    b"\x8e\xd0\xf3\x6c".as_slice()
                } else {
                    b"\xf3\x6c"
                };
                let (vm, mut cpu) = traced(code, generation);
                cpu.cpu.rflags_system = 0x100;
                cpu.cpu.gpr[1] = 2;
                cpu.cpu.gpr[7] = 0x4000;
                if ss {
                    steps(&mut cpu, &vm, 1);
                }
                let base = u64::from(ss);
                let ip = 0x1000 + if ss { 2 } else { 0 };
                for left in [1, 0] {
                    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                    assert_eq!(run.steps(), 1);
                    assert!(matches!(run.stop(), StepStop::Exit(Exit::PortIo { .. })));
                    assert_eq!(cpu.cpu.rflags_system, 0x100);
                    assert_eq!(cpu.cpu.gpr[1], left + 1);
                    cpu.complete_port_in(0x5a);
                    steps(&mut cpu, &vm, 0);
                    assert_eq!(cpu.cpu.gpr[1], left + 1);
                    steps(&mut cpu, &vm, 1); // store, #DB, then one handler IRET
                    assert_eq!(cpu.cpu.rip, if left == 0 { ip + 2 } else { ip });
                    assert_eq!(cpu.cpu.gpr[1], left);
                    assert_eq!(value(&vm, 0x7ffa, 2), cpu.cpu.rip);
                    assert_eq!(
                        cpu.retired_instructions(),
                        base + if left == 0 { 3 } else { 1 }
                    );
                }
                assert_eq!(value(&vm, 0x4000, 2), 0x5a5a);
            }
            for code in [b"\xe4\x20", b"\xe6\x20"] {
                let (vm, mut cpu) = traced(code, generation);
                cpu.cpu.rflags_system = 0x100;
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert!(matches!(run.stop(), StepStop::Exit(Exit::PortIo { .. })));
                cpu.complete_port_in(0x5a);
                steps(&mut cpu, &vm, 1); // deferred #DB, then handler IRET
                assert_eq!(cpu.cpu.rip, 0x1002);
                assert_eq!(value(&vm, 0x7ffa, 2), 0x1002);
                assert_eq!(cpu.retired_instructions(), 2);
            }
        }
    }

    #[test]
    fn debug_delivery_retry_keeps_retirement_and_priority() {
        // The same IVT/MMIO continuation as IRQ delivery (decision-16 §3).
        for generation in MODELS {
            let (mut vm, mut cpu) = traced(b"\x90", generation);
            vm.unmap(0, 0x40000).unwrap();
            vm.map(0x1000, 0x3f000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0x1000, b"\x90").unwrap();
            vm.write_bytes(0x1800, b"\xcf").unwrap();
            vm.map(0, 0x1000, Prot::RWX, RegionKind::Trap).unwrap();
            cpu.cpu.rflags_system = 0x100;
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1);
            assert!(matches!(
                run.stop(),
                StepStop::Exit(Exit::MmioRead { addr: 4, .. })
            ));
            assert_eq!(cpu.retired_instructions(), 1);
            for (answer, next) in [(0x1800, 6), (0, 0)] {
                cpu.complete_mmio_read(answer);
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                if next != 0 {
                    assert!(
                        matches!(run.stop(), StepStop::Exit(Exit::MmioRead { addr, .. }) if *addr == next)
                    );
                    assert_eq!(run.steps(), 0);
                } else {
                    // Delivery retires nothing; this unit executes the handler's IRET.
                    assert_eq!(run.steps(), 1);
                    assert_eq!(cpu.cpu.rip, 0x1001);
                    assert_eq!(cpu.retired_instructions(), 2);
                }
            }
        }
    }
}
