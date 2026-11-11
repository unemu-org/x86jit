//! Legacy I386 real-address witnesses. Expected values are derived from the
//! cited manuals; no emulator supplies instruction semantics.
use x86jit_core::{
    Backend, CachedBlock, CpuGeneration, CpuState, Exit, IrBlock, LegacyCpuModel, MemConsistency,
    Prot, RegionKind, SegmentId, StepOptions, StepRunError, StepStop, UnsupportedSystemOperation,
    Vcpu, Vm, VmConfig,
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
    let mut vm = Vm::with_backend(VmConfig::flat(0x40000), Box::new(NoBackend));
    vm.map(0, 0x40000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.set_legacy_cpu_model(MODEL);
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
fn pushfd_popfd_identification_masks_and_atomic_refusal() {
    // SDM Vol 1 §21.1.2; 80386 PRM §2.3.4 Fig. 2-8 p. 2-11,
    // §17 POPF p. 17-130 (page images checked): AC/ID/reserved bits zero,
    // IOPL/NT writable; RF/VM unaffected. TF requests are outside engine scope.
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
    let before = cpu.cpu.gpr;
    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0);
    assert!(matches!(
        run.stop(),
        StepStop::Exit(Exit::UnsupportedSystemContext { .. })
    ));
    assert_eq!(cpu.cpu.gpr, before);
    assert_eq!(cpu.cpu.rflags_system, 0x7000);
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
    // Decision-16 §§1,4; SDM Vol 3A §§7.8.1,7.8.3. Unavailable legacy
    // event delivery is a typed refusal at acceptance, retaining the vector.
    let (vm, mut cpu) = setup(b"\xfb\x66\x60\x90");
    cpu.inject_irq(0x20);
    steps(&mut cpu, &vm, 0);
    steps(&mut cpu, &vm, 1);
    steps(&mut cpu, &vm, 1);
    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0);
    assert!(matches!(
        run.stop(),
        StepStop::Exit(Exit::UnsupportedSystemContext {
            operation: UnsupportedSystemOperation::LegacyEventDelivery,
            ..
        })
    ));
    assert!(cpu.has_pending_irq());
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
fn protected_entry_stays_refused_and_legacy_never_materializes() {
    let (vm, mut cpu) = setup(b"\x66\xb8\x78\x56\x34\x12\xf4");
    cpu.cpu.legacy.cr0 = 1;
    let before = format!("{:?}", cpu.cpu);
    assert!(matches!(
        cpu.run(&vm, None),
        Exit::UnsupportedSystemContext { .. }
    ));
    assert_eq!(format!("{:?}", cpu.cpu), before);
    cpu.cpu.legacy.cr0 = 0;
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
        assert_eq!(run.steps(), 0);
        assert!(matches!(
            run.stop(),
            StepStop::Exit(Exit::UnsupportedSystemContext {
                operation: UnsupportedSystemOperation::LegacyEventDelivery,
                ..
            })
        ));
        assert_eq!(cpu.cpu.rip, rip);
        assert_eq!(cpu.cpu.gpr[0], ax + u64::from(shadow));
        assert!(cpu.has_pending_irq());
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
    for opcode in [0x20, 0x21, 0x22, 0x23, 0x24, 0x26] {
        let registers: &[u8] = match opcode {
            0x20 | 0x22 => &[0, 2, 3, 4],
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
