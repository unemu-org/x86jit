//! Intel 80286/80287 PRM 210498-005 (1987), cited page images checked.
//! Same-CPL legacy interpreter witnesses; no application backend may execute.
use x86jit_core::fault::ErrorCode;
use x86jit_core::{
    derive_legacy_context, Backend, CachedBlock, CpuGeneration, CpuState, Exit, IrBlock,
    LegacyContextError, LegacyCpuModel, LegacyRegime, MemConsistency, Prot, RegionKind,
    ResetOptions, StepOptions, StepStop, UnsupportedSystemOperation, Vcpu, Vm, VmConfig,
};

const MODEL: LegacyCpuModel = LegacyCpuModel {
    generation: CpuGeneration::I286,
    model_semantics_id: 0x286,
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
        panic!("legacy execution reached application backend")
    }
}
fn setup(code: &[u8], protected: bool) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x1000000), Box::new(NoBackend));
    vm.map(0, 0x1000000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.set_legacy_cpu_model(MODEL);
    vm.write_bytes(0x1000, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[4] = 0xabcd_8000;
    cpu.cpu.legacy.cr0 = u32::from(protected);
    cpu.cpu.legacy.gdtr.base = 0x2000;
    cpu.cpu.legacy.gdtr.limit = 31;
    descriptor(&vm, 8, 0, 0xffff, 0x9a);
    descriptor(&vm, 16, 0x123000, 0x1f, 0x92);
    descriptor(&vm, 24, 0x3000, 0xffff, 0x92);
    (vm, cpu)
}
fn descriptor(vm: &Vm, selector: u16, base: u32, limit: u16, access: u8) {
    // PRM §6.3 Fig. 6-3 p. 6-5: limit 0:1, base 2:4, access 5.
    let bytes = [
        limit as u8,
        (limit >> 8) as u8,
        base as u8,
        (base >> 8) as u8,
        (base >> 16) as u8,
        access,
        0,
        0,
    ];
    vm.write_bytes(0x2000 + u64::from(selector), &bytes)
        .unwrap();
}
fn bytes(vm: &Vm, addr: u64, len: usize) -> Vec<u8> {
    let mut result = vec![0; len];
    vm.read_bytes(addr, &mut result).unwrap();
    result
}
fn steps(cpu: &mut Vcpu, vm: &Vm, count: u64) {
    let run = cpu.run_steps(vm, count, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), count, "{:?}", run.stop());
    assert!(
        matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
        "{:?}",
        run.stop()
    );
}
fn fault(cpu: &mut Vcpu, vm: &Vm, vector: u8, code: Option<u32>) {
    let before = format!("{:?}", cpu.cpu);
    let run = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0);
    let StepStop::Exit(exit @ Exit::Exception { vector: actual, .. }) = run.stop() else {
        panic!("{:?}", run.stop());
    };
    assert_eq!(*actual, vector);
    let details = exit.fault_metadata().unwrap();
    assert_eq!(details.saved_rip(), cpu.cpu.rip);
    assert_eq!(
        details.error_code(),
        code.map_or(ErrorCode::NoCode, |c| ErrorCode::bits(c, 0xffff))
    );
    assert_eq!(format!("{:?}", cpu.cpu), before);
}

#[test]
fn real_lgdt_lmsw_far_jump_data_fault_reset_and_continue() {
    // PRM Appendix B LGDT p. B-65, LMSW p. B-67; §§7.4–7.5 pp. 7-12–15.
    let (vm, mut cpu) = setup(
        b"\x0f\x01\x16\x00\x21\xb8\x01\x00\x0f\x01\xf0\xea\x00\x12\x08\x00",
        false,
    );
    cpu.cpu.legacy.gdtr = Default::default();
    vm.write_bytes(0x2100, b"\x1f\x00\x00\x20\x00\xa5").unwrap();
    vm.write_bytes(0x1200, b"\xb8\x10\x00\x8e\xd8\xa1\x1e\x00\xa1\x1f\x00")
        .unwrap();
    vm.write_bytes(0x12301e, b"\xef\xbe").unwrap();
    steps(&mut cpu, &vm, 2);
    let caches = cpu.cpu.legacy.segments;
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.legacy.cr0 & 1, 1);
    assert_eq!(cpu.cpu.legacy.segments, caches);
    steps(&mut cpu, &vm, 4);
    assert_eq!((cpu.cpu.cs, cpu.cpu.ds, cpu.cpu.gpr[0]), (8, 16, 0xbeef));
    assert_eq!(cpu.cpu.legacy.segments[1].base, 0x123000);
    fault(&mut cpu, &vm, 13, Some(0));

    // PRM §10.4 p. 10-6: the embedder installs the reset register image.
    // reset_execution_state abandons handoffs; it supplies no hardware reset table.
    cpu.inject_irq(0x20);
    cpu.complete_port_in(0xffff);
    cpu.complete_mmio_read(0xffff);
    cpu.exit_request().request_exit();
    cpu.reset_execution_state(ResetOptions::default());
    assert!(!cpu.has_pending_irq());
    cpu.cpu = CpuState::legacy_real_entry([0xf000, 0, 0, 0, 0, 0]);
    cpu.cpu.rip = 0xfff0;
    cpu.cpu.legacy.cr0 = 0xfff0;
    cpu.cpu.legacy.segments[0].base = 0xff0000;
    cpu.cpu.legacy.reset_fetch = 1;
    vm.write_bytes(0xfffff0, b"\xea\x00\x15\x00\x00").unwrap();
    vm.write_bytes(0x1500, b"\x0f\x01\xe0\xbb\xfe\xca\xf4")
        .unwrap();
    steps(&mut cpu, &vm, 1);
    assert_eq!(
        (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.legacy.reset_fetch),
        (0, 0x1500, 0)
    );
    assert_eq!(
        derive_legacy_context(MODEL, &cpu.cpu).unwrap().regime,
        LegacyRegime::Real
    );
    assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
    assert_eq!((cpu.cpu.gpr[0], cpu.cpu.gpr[3]), (0xfff0, 0xcafe));
    assert_eq!(cpu.cpu.legacy.segments[1].limit, 0xffff);
    assert_eq!(cpu.cpu.legacy.segments[1].base, 0);
}

#[test]
fn msw_word_forms_sticky_pe_and_clts() {
    // PRM §10.2.2 Table 10-1 p. 10-4; Appendix B LMSW/SMSW pp. B-67, B-103.
    let (vm, mut cpu) = setup(
        b"\x0f\x01\xf0\x0f\x01\xe3\x0f\x01\x36\x00\x22\x0f\x01\x26\x02\x22\x0f\x06\x0f\x01\xe1",
        false,
    );
    cpu.cpu.gpr[0] = 0xffff;
    cpu.cpu.gpr[3] = 0x12340000;
    vm.write_bytes(0x2200, b"\x00\x00\xaa\xaa\xaa\xaa").unwrap();
    steps(&mut cpu, &vm, 2);
    assert_eq!(cpu.cpu.gpr[3], 0x1234ffff);
    assert_eq!(cpu.cpu.legacy.cr0 & 0xf, 0xf);
    steps(&mut cpu, &vm, 2);
    assert_eq!(cpu.cpu.legacy.cr0 & 0xf, 1);
    assert_eq!(bytes(&vm, 0x2202, 4), b"\xf1\xff\xaa\xaa");
    cpu.cpu.legacy.cr0 |= 8;
    steps(&mut cpu, &vm, 2);
    assert_eq!(cpu.cpu.gpr[1], 0xfff1);
    assert_eq!(cpu.cpu.legacy.cr0 & 0xf, 1);
}

#[test]
fn table_loads_ignore_sixth_byte_stores_write_ones_in_both_regimes() {
    // PRM §10.2 Fig. 10-3 p. 10-3; Appendix B LGDT p. B-65, SGDT p. B-101.
    // PRM leaves stored byte 5 undefined; SDM Vol 2B §4.3 SGDT/SIDT,
    // IA-32 Architecture Compatibility specifies ones for the 286.
    for protected in [false, true] {
        let (vm, mut cpu) = setup(
            b"\x0f\x01\x16\x00\x22\x0f\x01\x1e\x06\x22\x0f\x01\x06\x10\x22\x0f\x01\x0e\x18\x22",
            protected,
        );
        vm.write_bytes(0x2200, b"\x34\x12\x78\x56\xab\x12\x21\x43\x65\x87\xcd\x34")
            .unwrap();
        steps(&mut cpu, &vm, 4);
        assert_eq!(
            (cpu.cpu.legacy.gdtr.base, cpu.cpu.legacy.gdtr.limit),
            (0xab5678, 0x1234)
        );
        assert_eq!(
            (cpu.cpu.legacy.idtr.base, cpu.cpu.legacy.idtr.limit),
            (0xcd8765, 0x4321)
        );
        assert_eq!(bytes(&vm, 0x2210, 6), b"\x34\x12\x78\x56\xab\xff");
        assert_eq!(bytes(&vm, 0x2218, 6), b"\x21\x43\x65\x87\xcd\xff");
    }
}

#[test]
fn table_images_check_all_six_bytes_before_effects() {
    // PRM Appendix B LGDT/SGDT pp. B-65, B-101; §7.4.1 p. 7-12.
    for load in [false, true] {
        let (vm, mut cpu) = setup(
            if load {
                b"\x0f\x01\x16\x00\x22"
            } else {
                b"\x0f\x01\x06\x00\x22"
            },
            true,
        );
        vm.write_bytes(0x2200, &[0xa5; 6]).unwrap();
        cpu.cpu.legacy.segments[1].limit = 0x2204;
        fault(&mut cpu, &vm, 13, Some(0));
        assert_eq!(bytes(&vm, 0x2200, 6), [0xa5; 6]);
    }
}

#[test]
fn descriptors_have_no_386_flags_high_base_or_limit_and_cache_until_reload() {
    // PRM §6.3 Fig. 6-3 p. 6-5; §7.4 pp. 7-11–12. Engine policy:
    // reserved bytes do not supply 386 fields, even when nonzero.
    let (vm, mut cpu) = setup(b"\x8e\xd8\x8e\xc0\x8e\xd0\xea\x00\x12\x08\x00", true);
    for selector in [8, 16] {
        vm.write_bytes(0x2000 + selector + 6, b"\xff\xff").unwrap();
    }
    cpu.cpu.gpr[0] = 16;
    steps(&mut cpu, &vm, 3);
    for i in [1, 2, 3] {
        let cache = cpu.cpu.legacy.segments[i];
        assert_eq!(
            (cache.base, cache.limit, cache.attributes),
            (0x123000, 0x1f, 0x93)
        );
    }
    // PRM §7.4 Table 7-1 p. 7-11: segment loads set A in the descriptor table.
    assert_eq!(bytes(&vm, 0x2015, 1), [0x93]);
    steps(&mut cpu, &vm, 1);
    assert_eq!(
        (
            cpu.cpu.legacy.segments[0].base,
            cpu.cpu.legacy.segments[0].limit,
            cpu.cpu.legacy.segments[0].attributes
        ),
        (0, 0xffff, 0x9b)
    );
    assert_eq!(bytes(&vm, 0x200d, 1), [0x9b]);
    assert_eq!(bytes(&vm, 0x200e, 2), b"\xff\xff");
    vm.write_bytes(0x1200, b"\xa1\x00\x00\x8e\xd8").unwrap();
    vm.write_bytes(0x123000, b"\x34\x12").unwrap();
    descriptor(&vm, 16, 0x4000, 0x10, 0x92);
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.gpr[0], 0x1234);
    cpu.cpu.gpr[0] = 16;
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.legacy.segments[1].base, 0x4000);
}

#[test]
fn far_calls_and_returns_use_word_frames_and_preserve_upper_sp() {
    // PRM Appendix B CALL p. B-23, RET pp. B-94–95; §7.5 p. 7-15.
    for indirect in [false, true] {
        let code = if indirect {
            &b"\xff\x1e\x00\x22"[..]
        } else {
            &b"\x9a\x00\x12\x08\x00"[..]
        };
        let (vm, mut cpu) = setup(code, true);
        cpu.cpu.cs = 8;
        vm.write_bytes(0x2200, b"\x00\x12\x08\x00").unwrap();
        vm.write_bytes(0x1200, b"\xcb").unwrap();
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
            (8, 0x1200, 0xabcd7ffc)
        );
        assert_eq!(bytes(&vm, 0x7ffc, 4), [code.len() as u8, 0x10, 8, 0]);
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (cpu.cpu.rip, cpu.cpu.gpr[4]),
            (0x1000 + code.len() as u64, 0xabcd8000)
        );
    }
}

#[test]
fn iret_restores_protected_flags_and_refuses_a_later_task_return() {
    // PRM §10.1 p. 10-1; Appendix B IRET pp. B-51–52: six-byte frame,
    // IOPL and NT restored at CPL 0. Task execution remains an engine refusal.
    let (vm, mut cpu) = setup(b"\xcf", true);
    vm.write_bytes(0x8000, b"\x00\x12\x08\x00\x03\x72").unwrap();
    vm.write_bytes(0x1200, b"\x9c\xcf").unwrap();
    steps(&mut cpu, &vm, 1);
    assert_eq!(
        (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
        (8, 0x1200, 0xabcd8006)
    );
    assert_eq!(cpu.cpu.rflags_system, 0x7000);
    assert!(cpu.cpu.flags.cf && cpu.cpu.flags.if_);
    steps(&mut cpu, &vm, 1);
    assert_eq!(bytes(&vm, 0x8004, 2), b"\x03\x72");
    let before = format!("{:?}", cpu.cpu);
    assert!(matches!(
        cpu.run(&vm, Some(1)),
        Exit::UnsupportedSystemContext {
            operation: UnsupportedSystemOperation::LegacyTaskReturn,
            ..
        }
    ));
    assert_eq!(format!("{:?}", cpu.cpu), before);
}

#[test]
fn protected_segment_loads_report_selector_faults_without_committing() {
    // PRM §7.4.1 p. 7-12; Appendix B MOV pp. B-73–74:
    // type, presence and selector-table limits are checked on load.
    for (selector, access, vector, code) in
        [(16, 0x12, 11, 16), (16, 0x98, 13, 16), (32, 0x92, 13, 32)]
    {
        let (vm, mut cpu) = setup(b"\x8e\xd8", true);
        descriptor(&vm, 16, 0x123000, 0x1f, access);
        cpu.cpu.gpr[0] = selector;
        fault(&mut cpu, &vm, vector, Some(code));
    }
    let (vm, mut cpu) = setup(b"\x8e\xd0", true);
    cpu.cpu.gpr[0] = 0;
    fault(&mut cpu, &vm, 13, Some(0));
}

#[test]
fn protected_far_targets_and_stack_fault_before_committing() {
    // PRM §7.4.2 p. 7-13; Appendix B CALL/IRET/RET pp. B-23, B-52, B-95.
    for code in [
        &b"\xea\x00\x12\x08\x00"[..],
        &b"\x9a\x00\x12\x08\x00"[..],
        &b"\xcb"[..],
        &b"\xcf"[..],
    ] {
        let (vm, mut cpu) = setup(code, true);
        descriptor(&vm, 8, 0, 0x1100, 0x9a);
        vm.write_bytes(0x8000, b"\x00\x12\x08\x00\x02\x00").unwrap();
        let stack = bytes(&vm, 0x7ffc, 10);
        fault(&mut cpu, &vm, 13, Some(0));
        assert_eq!(bytes(&vm, 0x7ffc, 10), stack);
    }
    let (vm, mut cpu) = setup(b"\x50", true);
    cpu.cpu.legacy.segments[3].limit = 0x7ffe;
    fault(&mut cpu, &vm, 12, Some(0));
}

#[test]
fn later_instruction_encodings_and_prefixes_are_ud_in_both_regimes() {
    // PRM §6.2 p. 6-2, Appendix B instruction encodings;
    // §9.6.1 p. 9-10: #UD saves the instruction including prefixes.
    let codes: &[&[u8]] = &[
        b"\x66\xb8\x78\x56\x34\x12",
        b"\x67\x8b\x00",
        b"\x64\x8b\x00",
        b"\x65\x90",
        b"\x0f\x20\xc0",
        b"\x0f\x22\xc0",
        b"\x0f\x20\xd0",
        b"\x0f\x22\xd0",
        b"\x0f\x20\xd8",
        b"\x0f\x22\xd8",
        b"\x0f\x21\xc0",
        b"\x0f\x24\xf0",
        b"\x8e\xe0",
        b"\x8c\xe8",
        b"\x0f\xa0",
        b"\x0f\xa9",
        b"\x0f\xb2\x06\x00\x22",
        b"\x0f\x94\xc0",
        b"\x0f\xbc\xc0",
        b"\x0f\xb6\xc0",
        b"\x0f\xa4\xc0\x01",
        b"\x0f\x84\x00\x00",
        b"\x0f\xa2",
        b"\x0f\x01\xc0",
    ];
    for protected in [false, true] {
        for &code in codes {
            let (vm, mut cpu) = setup(code, protected);
            fault(&mut cpu, &vm, 6, None);
        }
    }
}

#[test]
fn unavailable_opcode_bytes_at_cs_limit_are_ud_before_another_fetch() {
    // PRM Appendix B #UD p. B-12: invalid first/second opcode bytes raise #UD;
    // §9.6.1 p. 9-10: the saved IP includes leading prefixes.
    let codes: &[&[u8]] = &[
        b"\x64",
        b"\x65",
        b"\x66",
        b"\x67",
        b"\x0f\x20",
        b"\x0f\x22",
        b"\x0f\x80",
        b"\x0f\xb6",
        b"\x0f\xff",
        b"\x0f\x01\x3e", // INVLPG with an unfetched disp16; absent on the 286.
        b"\x0f\x01\x2e", // Reserved /5 with an unfetched disp16.
    ];
    for protected in [false, true] {
        for leading_prefix in [false, true] {
            for &code in codes {
                let mut instruction = Vec::new();
                if leading_prefix {
                    instruction.push(0x2e);
                }
                instruction.extend_from_slice(code);
                let (vm, mut cpu) = setup(&instruction, protected);
                cpu.cpu.rip = 0x10000 - instruction.len() as u64;
                vm.write_bytes(cpu.cpu.rip, &instruction).unwrap();
                fault(&mut cpu, &vm, 6, None);
            }
        }
        // A valid 286 prefix or incomplete two-byte opcode still needs a byte.
        for code in [b"\x2e", b"\x0f"] {
            let (vm, mut cpu) = setup(code, protected);
            cpu.cpu.rip = 0xffff;
            vm.write_bytes(0xffff, code).unwrap();
            fault(&mut cpu, &vm, 13, protected.then_some(0));
        }
    }
}

#[test]
fn real_word_at_ffff_faults_with_gp_for_ss_and_ds() {
    // PRM Appendix B MOV p. B-74: real-mode word operands at FFFFh
    // raise interrupt 13, including SS overrides.
    for code in [b"\x36\xa1\xff\xff", b"\x3e\xa1\xff\xff"] {
        let (vm, mut cpu) = setup(code, false);
        vm.write_bytes(0xffff, b"\x34\x12").unwrap();
        cpu.cpu.gpr[0] = 0xabcd;
        fault(&mut cpu, &vm, 13, None);
    }
}

#[test]
fn retained_386_sizes_and_descriptor_extensions_are_refused_for_i286() {
    // PRM §6.2 p. 6-2; §6.3 Fig. 6-3 p. 6-5. These are invalid
    // embedder states, so refusal precedes guest execution rather than #UD.
    for (index, attr, error) in [
        (0, 0x4000, LegacyContextError::SizeRequires386),
        (3, 0x4000, LegacyContextError::SizeRequires386),
        (1, 0x8000, LegacyContextError::InvalidStorage),
    ] {
        let (vm, mut cpu) = setup(b"\x90", true);
        cpu.cpu.legacy.segments[index].attributes |= attr;
        assert_eq!(derive_legacy_context(MODEL, &cpu.cpu), Err(error));
        assert!(
            matches!(cpu.run(&vm, Some(1)), Exit::UnsupportedSystemContext { operation: UnsupportedSystemOperation::InvalidLegacyContext(actual), .. } if actual == error)
        );
    }
}

#[test]
fn i286_instruction_length_is_ten_bytes_in_both_regimes() {
    // PRM Appendix B #GP p. B-9: exceeding ten bytes raises #GP(0).
    for protected in [false, true] {
        let mut code = vec![0x2e; 9];
        code.push(0x90);
        let (vm, mut cpu) = setup(&code, protected);
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.rip, 0x100a);
        code.insert(0, 0x2e);
        vm.write_bytes(0x1000, &code).unwrap();
        cpu.cpu.rip = 0x1000;
        fault(&mut cpu, &vm, 13, protected.then_some(0));
    }
}

#[test]
fn real_coprocessor_execution_and_deferred_system_forms() {
    // I286/287 PRM Part II Ch. 2, Processor Control Instructions, p. 2-16: FNINIT and FSETPM exist.
    // Real execution shares the Application FPU engine. Protected execution
    // and FSETPM's coprocessor mode change remain deferred (decision-18 §4).
    for protected in [false, true] {
        for code in [b"\xdb\xe3", b"\xdb\xe4"] {
            let (vm, mut cpu) = setup(code, protected);
            let exit = cpu.run(&vm, Some(1));
            if protected {
                assert!(matches!(
                    exit,
                    Exit::UnsupportedSystemContext {
                        operation: UnsupportedSystemOperation::LegacyInstruction,
                        ..
                    }
                ));
            } else if code == b"\xdb\xe3" {
                assert!(matches!(exit, Exit::BudgetExhausted));
                assert_eq!(cpu.cpu.rip, 0x1002);
                assert_eq!((cpu.cpu.fpu_cw, cpu.cpu.fpu_empty), (0x37f, 0xff));
                continue;
            } else {
                assert!(matches!(exit, Exit::UnknownInstruction { .. }));
            }
            assert_eq!(cpu.cpu.rip, 0x1000);
        }
    }
}
