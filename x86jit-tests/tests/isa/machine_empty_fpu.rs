//! I286/287 PRM 210498-005, Part I Appendix A p. A-4; Part II Ch. 2
//! "Data Synchronization" p. 2-49, Ch. 3 pp. 3-1–3-3, Appendix B item 10
//! p. B-2, Appendix A p. A-5. Page images checked. Decision-25 supplies the
//! all-high word-transfer default and preserves larger memory destinations.
use x86jit_core::{
    fault::{ErrorCode, FaultClass},
    Backend, CachedBlock, CpuModel, CpuState, ExceptionDelivery, Exit, FpuModel, IrBlock,
    MemConsistency, Prot, RegionKind, SegmentId, StepOptions, StepResult, StepStop, Vcpu, Vm,
    VmConfig,
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
        panic!("Machine execution reached application backend")
    }
}

fn setup(model: CpuModel, code: &[u8]) -> (Vm, Vcpu) {
    setup_with_delivery(model, code, ExceptionDelivery::Outward)
}

fn setup_with_delivery(model: CpuModel, code: &[u8], delivery: ExceptionDelivery) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x20000), Box::new(NoBackend));
    vm.set_cpu_model(model);
    vm.set_exception_delivery(delivery);
    vm.set_fpu_model(FpuModel::None);
    for (base, size) in [(0, 0x2000), (0x2000, 0x1000), (0x3000, 0x1d000)] {
        vm.map(base, size, Prot::RWX, RegionKind::Ram).unwrap();
    }
    vm.write_bytes(0x1000, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu = CpuState::legacy_real_entry([0, 0x200, 0, 0x300, 0, 0]);
    cpu.cpu.fpu_model = FpuModel::None;
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[0] = 0x1234_5678_9abc_def0;
    cpu.cpu.gpr[4] = 0x8000;
    // A dirty, exception-pending image must remain inert with no coprocessor.
    cpu.cpu.fpr = [[0xa5; 10]; 8];
    cpu.cpu.fpu_top = 3;
    cpu.cpu.fpu_cw = 0x0360;
    cpu.cpu.fpu_sw = 0x8081;
    cpu.cpu.fpu_empty = 0x12;
    cpu.cpu.fpu_fip = 0x2345;
    cpu.cpu.fpu_fdp = 0x4567;
    cpu.cpu.fpu_fcs = 0x1234;
    cpu.cpu.fpu_fds = 0x2345;
    cpu.cpu.fpu_fop = 0x523;
    (vm, cpu)
}

fn image(cpu: &CpuState) -> ([[u8; 10]; 8], [u64; 9]) {
    (
        cpu.fpr,
        [
            u64::from(cpu.fpu_top),
            u64::from(cpu.fpu_cw),
            u64::from(cpu.fpu_sw),
            u64::from(cpu.fpu_empty),
            cpu.fpu_fip,
            cpu.fpu_fdp,
            u64::from(cpu.fpu_fop),
            u64::from(cpu.fpu_fcs),
            u64::from(cpu.fpu_fds),
        ],
    )
}

fn bytes(vm: &Vm, at: u64, len: usize) -> Vec<u8> {
    let mut out = vec![0; len];
    vm.read_bytes(at, &mut out).unwrap();
    out
}

fn retire(cpu: &mut Vcpu, vm: &Vm, len: u64, path: u8) {
    let ip = cpu.cpu.rip;
    let retired = cpu.retired_instructions();
    match path {
        0 => assert!(matches!(cpu.run(vm, Some(1)), Exit::BudgetExhausted)),
        1 => {
            let result = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
            assert_eq!(result.steps(), 1, "{result:?}");
            assert!(matches!(
                result.stop(),
                StepStop::Exit(Exit::BudgetExhausted)
            ));
        }
        _ => assert!(matches!(cpu.step_instruction(vm), StepResult::Continue)),
    }
    assert_eq!(cpu.cpu.rip, ip + len);
    assert_eq!(cpu.retired_instructions(), retired + 1);
}

#[test]
fn fninit_retires_without_initializing_an_absent_x87_image() {
    for path in 0..3 {
        let (vm, mut cpu) = setup(CpuModel::I80286, b"\xdb\xe3");
        let before = image(&cpu.cpu);
        retire(&mut cpu, &vm, 2, path);
        assert_eq!(image(&cpu.cpu), before);
    }
}

#[test]
fn fninit_fnstcw_probe_compares_the_bus_word_with_037f() {
    let (vm, mut cpu) = setup(
        CpuModel::I80286,
        b"\xdb\xe3\xd9\x3e\x00\x04\x81\x3e\x00\x04\x7f\x03\xf4",
    );
    vm.write_bytes(0x2400, &0x5a5au16.to_le_bytes()).unwrap();
    let before = image(&cpu.cpu);
    let result = cpu.run_steps(&vm, 3, StepOptions::default()).unwrap();
    assert_eq!(result.steps(), 3, "{result:?}");
    assert_eq!(cpu.cpu.rip, 0x100c);
    assert!(!cpu.cpu.flags.zf);
    assert_eq!(bytes(&vm, 0x2400, 2), [0xff, 0xff]);
    assert_eq!(image(&cpu.cpu), before);
}

#[test]
fn appendix_b_status_probe_over_zero_rejects_an_empty_socket() {
    // Part II Appendix B item 10 p. B-2: no pre-fill before the status store.
    // FNINIT; FNSTSW [STAT]; MOV AX,[STAT]; OR AL,AL; JZ present.
    let (vm, mut cpu) = setup(
        CpuModel::I80286,
        b"\xdb\xe3\xdd\x3e\x00\x04\xa1\x00\x04\x08\xc0\x74\x04\xbb\x00\x00\xf4\xbb\x01\x00\xf4",
    );
    vm.write_bytes(0x2400, &[0, 0]).unwrap();
    let before = image(&cpu.cpu);
    assert!(matches!(cpu.run(&vm, Some(20)), Exit::Hlt));
    assert_eq!(cpu.cpu.gpr[0], 0x1234_5678_9abc_ffff);
    assert_eq!(cpu.cpu.gpr[3] & 0xffff, 0); // Absent branch.
    assert_eq!(cpu.cpu.rip, 0x1011);
    assert_eq!(bytes(&vm, 0x2400, 2), [0xff, 0xff]);
    assert_eq!(image(&cpu.cpu), before);
}

#[test]
fn figure_3_1_prefilled_status_probe_rejects_an_empty_socket() {
    // Part II Fig. 3-1 p. 3-3: FNINIT, pre-fill 5A5Ah, FNSTSW, compare the
    // low byte with zero, then FNSTCW / AND AX,103Fh / CMP AX,003Fh if present.
    let (vm, mut cpu) = setup(
        CpuModel::I80286,
        b"\xdb\xe3\xbe\x00\x04\xc7\x04\x5a\x5a\xdd\x3c\x80\x3c\x00\x75\x10\xd9\x3c\x8b\x04\x25\x3f\x10\x3d\x3f\x00\x75\x04\xbb\x01\x00\xf4\xbb\x00\x00\xf4",
    );
    let before = image(&cpu.cpu);
    assert!(matches!(cpu.run(&vm, Some(20)), Exit::Hlt));
    assert_eq!(cpu.cpu.gpr[3] & 0xffff, 0);
    assert_eq!(cpu.cpu.rip, 0x1024);
    assert_eq!(bytes(&vm, 0x2400, 2), [0xff, 0xff]);
    assert_eq!(image(&cpu.cpu), before);
}

#[test]
fn fnstsw_ax_reads_all_ones_and_preserves_upper_register_bits() {
    // Part I Appendix A "80286 System Initialization", p. A-4 assumes all
    // ones, then FSTSW AX / OR AL,AL / JNZ to select the absent case.
    for code in [&b"\xdf\xe0"[..], &b"\x26\xdf\xe0"[..]] {
        for path in 0..3 {
            for protected in [false, true] {
                let (vm, mut cpu) = setup(CpuModel::I80286, code);
                cpu.cpu.legacy.cr0 = u32::from(protected);
                let before = image(&cpu.cpu);
                let flags = cpu.cpu.flags.to_rflags();
                retire(&mut cpu, &vm, code.len() as u64, path);
                assert_eq!(cpu.cpu.gpr[0], 0x1234_5678_9abc_ffff);
                assert_eq!(cpu.cpu.flags.to_rflags(), flags);
                assert_eq!(image(&cpu.cpu), before);
            }
        }
    }
    let (vm, mut cpu) = setup(CpuModel::I80286, b"\xdb\xe3\xdf\xe0\x08\xc0");
    let result = cpu.run_steps(&vm, 3, StepOptions::default()).unwrap();
    assert_eq!(result.steps(), 3, "{result:?}");
    assert!(!cpu.cpu.flags.zf); // OR AL,AL selects the absent branch.
}

// 287 output encodings and widths: Part II Appendix A, Table A-2 pp. A-2–A-4;
// Ch. 1 Table 1-2 p. 1-6. FISTTP remains outside the 286 ISA.
const OUTPUTS: [(u8, u8, u8); 15] = [
    (0xd9, 0x3e, 2),  // FNSTCW m16
    (0xdd, 0x3e, 2),  // FNSTSW m16
    (0xd9, 0x16, 4),  // FST m32
    (0xd9, 0x1e, 4),  // FSTP m32
    (0xdd, 0x16, 8),  // FST m64
    (0xdd, 0x1e, 8),  // FSTP m64
    (0xdb, 0x3e, 10), // FSTP m80
    (0xdf, 0x16, 2),  // FIST m16
    (0xdf, 0x1e, 2),  // FISTP m16
    (0xdb, 0x16, 4),  // FIST m32
    (0xdb, 0x1e, 4),  // FISTP m32
    (0xdf, 0x3e, 8),  // FISTP m64
    (0xdf, 0x36, 10), // FBSTP m80 (Present lifter gap)
    (0xd9, 0x36, 14), // FNSTENV m14
    (0xdd, 0x36, 94), // FNSAVE m94
];

fn transfers_word(opcode: u8, modrm: u8) -> bool {
    matches!((opcode, modrm), (0xd9 | 0xdd, 0x3e))
}

#[test]
fn memory_outputs_follow_the_word_transfer_and_larger_store_rules() {
    for (opcode, modrm, _) in OUTPUTS {
        for path in 0..3 {
            let (vm, mut cpu) = setup(CpuModel::I80286, &[opcode, modrm, 0, 4]);
            let mut sentinel: Vec<u8> = (0..96).map(|i| i ^ 0x5a).collect();
            vm.write_bytes(0x23ff, &sentinel).unwrap();
            let before = image(&cpu.cpu);
            let gpr = cpu.cpu.gpr;
            let flags = cpu.cpu.flags.to_rflags();
            retire(&mut cpu, &vm, 4, path);
            if transfers_word(opcode, modrm) {
                sentinel[1..3].fill(0xff);
            }
            assert_eq!(bytes(&vm, 0x23ff, 96), sentinel, "{opcode:02x}/{modrm:02x}");
            assert_eq!(cpu.cpu.gpr, gpr);
            assert_eq!(cpu.cpu.flags.to_rflags(), flags);
            assert_eq!(image(&cpu.cpu), before);
        }
    }
}

#[test]
fn memory_outputs_accept_the_exact_segment_end() {
    memory_output_boundary(false);
}

#[test]
fn memory_outputs_fault_on_a_crossing_operand_before_any_write() {
    memory_output_boundary(true);
}

fn memory_output_boundary(crossing: bool) {
    // Existing Machine x87 policy: I286 real mode #GP for DS and SS;
    // protected mode #GP(0) for DS and #SS(0) for SS (decision-21).
    for (opcode, modrm, width) in OUTPUTS {
        for protected in [false, true] {
            for segment in [SegmentId::Ds, SegmentId::Ss] {
                let offset = 0x10000u32 - u32::from(width) + u32::from(crossing);
                let [lo, hi] = (offset as u16).to_le_bytes();
                let prefix = if segment == SegmentId::Ss { 0x36 } else { 0x3e };
                let (vm, mut cpu) = setup(CpuModel::I80286, &[prefix, opcode, modrm, lo, hi]);
                cpu.cpu.legacy.cr0 = u32::from(protected);
                let at =
                    u64::from(cpu.cpu.legacy.segments[segment as usize].base) + u64::from(offset);
                vm.write_bytes(at - 1, &[0xa5; 96]).unwrap();
                let before = image(&cpu.cpu);
                let gpr = cpu.cpu.gpr;
                let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                if crossing {
                    let vector = if protected && segment == SegmentId::Ss {
                        12
                    } else {
                        13
                    };
                    assert_eq!(result.steps(), 0);
                    let StepStop::Exit(exit @ Exit::Exception { vector: actual, .. }) =
                        result.stop()
                    else {
                        panic!("{opcode:02x}/{modrm:02x}: {result:?}");
                    };
                    assert_eq!(*actual, vector);
                    let details = exit.fault_metadata().unwrap();
                    assert_eq!(details.saved_rip(), 0x1000);
                    assert_eq!(
                        details.error_code(),
                        if protected {
                            ErrorCode::bits(0, 0xffff)
                        } else {
                            ErrorCode::NoCode
                        }
                    );
                    assert_eq!(cpu.cpu.rip, 0x1000);
                } else {
                    assert_eq!(result.steps(), 1, "{opcode:02x}/{modrm:02x}: {result:?}");
                    assert_eq!(cpu.cpu.rip, 0x1005);
                }
                let mut expected = [0xa5; 96];
                if !crossing && transfers_word(opcode, modrm) {
                    expected[1..3].fill(0xff);
                }
                assert_eq!(bytes(&vm, at - 1, 96), expected);
                assert_eq!(cpu.cpu.gpr, gpr);
                assert_eq!(image(&cpu.cpu), before);
            }
        }
    }
}

#[test]
fn larger_outputs_issue_no_bus_transfer_to_unmapped_or_trapped_memory() {
    for (opcode, modrm, _) in OUTPUTS {
        if transfers_word(opcode, modrm) {
            continue;
        }
        for trap in [false, true] {
            let (mut vm, mut cpu) = setup(CpuModel::I80286, &[opcode, modrm, 0, 4]);
            vm.unmap(0x2000, 0x1000).unwrap();
            if trap {
                vm.map(0x2000, 0x1000, Prot::RWX, RegionKind::Trap).unwrap();
            }
            let before = image(&cpu.cpu);
            retire(&mut cpu, &vm, 4, 1);
            assert_eq!(image(&cpu.cpu), before);
        }
    }
}

#[test]
fn word_outputs_use_normal_memory_handoffs_before_retirement() {
    for opcode in [0xd9, 0xdd] {
        for protected in [false, true] {
            for trap in [false, true] {
                let (mut vm, mut cpu) = setup(CpuModel::I80286, &[opcode, 0x3e, 0, 4]);
                cpu.cpu.legacy.cr0 = u32::from(protected);
                vm.unmap(0x2000, 0x1000).unwrap();
                if trap {
                    vm.map(0x2000, 0x1000, Prot::RWX, RegionKind::Trap).unwrap();
                }
                let before = image(&cpu.cpu);
                let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(result.steps(), 0);
                if trap {
                    assert!(
                        matches!(
                            result.stop(),
                            StepStop::Exit(Exit::MmioWrite {
                                addr: 0x2400,
                                size: 2,
                                value: 0xffff,
                                ..
                            })
                        ),
                        "{result:?}"
                    );
                    cpu.complete_mmio_write();
                } else {
                    assert!(
                        matches!(
                            result.stop(),
                            StepStop::Exit(Exit::UnmappedMemory { addr: 0x2400, .. })
                        ),
                        "{result:?}"
                    );
                    vm.map(0x2000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
                }
                assert_eq!(cpu.cpu.rip, 0x1000);
                assert_eq!(image(&cpu.cpu), before);
                retire(&mut cpu, &vm, 4, 1);
                if !trap {
                    assert_eq!(bytes(&vm, 0x2400, 2), [0xff, 0xff]);
                }
                assert_eq!(image(&cpu.cpu), before);
            }
        }
    }
}

#[test]
fn wait_then_fnstsw_ax_retires_in_real_and_protected_mode() {
    for protected in [false, true] {
        for path in 0..3 {
            let (vm, mut cpu) = setup(CpuModel::I80286, b"\x9b\xdf\xe0");
            cpu.cpu.legacy.cr0 = u32::from(protected);
            let before = image(&cpu.cpu);
            let ax = cpu.cpu.gpr[0];
            retire(&mut cpu, &vm, 1, path);
            assert_eq!(cpu.cpu.gpr[0], ax);
            retire(&mut cpu, &vm, 2, path);
            assert_eq!(cpu.cpu.gpr[0], 0x1234_5678_9abc_ffff);
            assert_eq!(image(&cpu.cpu), before);
        }
    }
}

#[test]
fn wait_then_fnstsw_ax_with_ts_faults_at_the_esc_in_both_regimes() {
    for protected in [false, true] {
        let (vm, mut cpu) = setup(CpuModel::I80286, b"\x9b\xdf\xe0");
        cpu.cpu.legacy.cr0 = u32::from(protected) | 8; // TS=1, MP=0.
        let before = image(&cpu.cpu);
        let ax = cpu.cpu.gpr[0];
        let result = cpu.run_steps(&vm, 2, StepOptions::default()).unwrap();
        assert_eq!(result.steps(), 1, "{result:?}");
        assert!(
            matches!(
                result.stop(),
                StepStop::Exit(Exit::Exception {
                    addr: 0x1001,
                    vector: 7,
                    ..
                })
            ),
            "{result:?}"
        );
        assert_eq!(cpu.cpu.rip, 0x1001);
        assert_eq!(cpu.retired_instructions(), 1);
        assert_eq!(cpu.cpu.gpr[0], ax);
        assert_eq!(image(&cpu.cpu), before);
    }
}

#[test]
fn cached_protected_outputs_keep_present_coprocessor_execution_deferred() {
    for code in [
        &b"\x9b"[..],
        &b"\xdf\xe0"[..],
        &b"\xd9\x3e\x00\x04"[..],
        &b"\xdf\x36\x00\x04"[..],
    ] {
        let (vm, mut cpu) = setup(CpuModel::I80286, code);
        cpu.cpu.legacy.cr0 = 1;
        let before = image(&cpu.cpu);
        retire(&mut cpu, &vm, code.len() as u64, 1);
        cpu.cpu.rip = 0x1000;
        cpu.set_fpu_model(FpuModel::Present);
        let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(result.steps(), 0);
        assert!(matches!(
            result.stop(),
            StepStop::Exit(Exit::UnsupportedSystemContext {
                operation: x86jit_core::UnsupportedSystemOperation::LegacyInstruction,
                ..
            })
        ));
        assert_eq!(image(&cpu.cpu), before);
    }
}

#[test]
fn absent_outputs_still_obey_generation_checks_after_the_msw_gate() {
    // Decision-21: I286 rejects 386 size prefixes and later FISTTP encodings.
    for code in [&b"\x66\xd9\x3e\xff\xff"[..], &b"\xdf\x0e\xff\xff"[..]] {
        for msw in [0, 4, 8] {
            let (vm, mut cpu) = setup(CpuModel::I80286, code);
            cpu.cpu.legacy.cr0 = msw;
            let before = image(&cpu.cpu);
            let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(result.steps(), 0);
            // An unavailable first opcode byte (66h on I286) faults before
            // fetching an ESC; a decoded ESC reaches the MSW gate first.
            let expected = if msw == 0 || code[0] == 0x66 { 6 } else { 7 };
            assert!(
                matches!(result.stop(), StepStop::Exit(Exit::Exception { vector, .. }) if *vector == expected),
                "{result:?}"
            );
            assert_eq!(cpu.cpu.rip, 0x1000);
            assert_eq!(image(&cpu.cpu), before);
        }
    }
}

#[test]
fn wait_with_inactive_busy_and_error_ignores_em_and_ts_without_mp() {
    // I286 PRM §10.2.2 Table 10-1 p. 10-4, Appendix B WAIT p. B-113;
    // I386 PRM §§11.1.2–11.1.4 pp. 11-1–11-2 (page images checked).
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for msw in [0, 2, 4, 6, 8, 12] {
            let (vm, mut cpu) = setup(model, b"\x9b");
            cpu.cpu.legacy.cr0 = msw;
            let before = image(&cpu.cpu);
            retire(&mut cpu, &vm, 1, 1);
            assert_eq!(image(&cpu.cpu), before);
        }
    }
}

#[test]
fn nm_precedes_output_and_operand_checks_with_outward_or_in_guest_delivery() {
    // The same MSW/CR0 gate applies with no attached coprocessor. A crossing
    // FNSTCW operand distinguishes #NM-before-operands from a segment fault.
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for (code, msw) in [
            (&b"\xd9\x3e\xff\xff"[..], 4), // ESC EM
            (&b"\xd9\x3e\xff\xff"[..], 8), // ESC TS, MP=0
            (&b"\xdf\xe0"[..], 4),
            (&b"\xdf\xe0"[..], 8),
            (&b"\xdb\xe3"[..], 4),
            (&b"\xdb\xe3"[..], 8),
            (&b"\x9b"[..], 10), // WAIT MP && TS
            (&b"\x9b"[..], 14), // WAIT also ignores EM here
        ] {
            for delivery in [ExceptionDelivery::Outward, ExceptionDelivery::InGuest] {
                let (vm, mut cpu) = setup_with_delivery(model, code, delivery);
                vm.write_bytes(7 * 4, &[0, 0x70, 0, 0]).unwrap();
                vm.write_bytes(0x7000, b"\xf4").unwrap();
                vm.write_bytes(0x11fff, &[0xa5; 96]).unwrap();
                vm.write_bytes(0xaffe, &[0xa5; 4]).unwrap();
                cpu.cpu.legacy.cr0 = msw;
                let before = image(&cpu.cpu);
                let gpr = cpu.cpu.gpr;
                let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                if delivery == ExceptionDelivery::Outward {
                    assert_eq!(result.steps(), 0);
                    let StepStop::Exit(exit @ Exit::Exception { vector: 7, .. }) = result.stop()
                    else {
                        panic!("{model:?} {code:02x?} {msw:x}: {result:?}");
                    };
                    let details = exit.fault_metadata().unwrap();
                    assert_eq!(details.saved_rip(), 0x1000);
                    assert_eq!(details.class(), FaultClass::Fault);
                    assert_eq!(details.error_code(), ErrorCode::NoCode);
                    assert_eq!(cpu.cpu.rip, 0x1000);
                    assert_eq!(cpu.cpu.gpr, gpr);
                    assert_eq!(bytes(&vm, 0xaffe, 4), [0xa5; 4]);
                } else {
                    assert_eq!(result.steps(), 1, "{result:?}");
                    assert!(matches!(result.stop(), StepStop::Exit(Exit::Hlt)));
                    assert_eq!(cpu.cpu.rip, 0x7001);
                    assert_eq!(cpu.cpu.gpr[0], gpr[0]);
                    assert_eq!(cpu.cpu.gpr[4], 0x7ffa);
                    assert_eq!(bytes(&vm, 0xaffa, 4), [0, 0x10, 0, 0]);
                }
                assert_eq!(bytes(&vm, 0x11fff, 96), [0xa5; 96]);
                assert_eq!(image(&cpu.cpu), before);
            }
        }
    }
}

#[test]
fn i386_outputs_keep_the_existing_refusal_without_a_sourced_empty_value() {
    // I386 PRM §11.1.2 specifies inactive pins when absent, not output data.
    for code in [
        &b"\xdf\xe0"[..],
        &b"\xd9\x3e\x00\x04"[..],
        &b"\xdd\x3e\x00\x04"[..],
        &b"\xdd\x36\x00\x04"[..],
        &b"\xdf\x36\x00\x04"[..],
        &b"\xd9\x3e\xff\xff"[..],
    ] {
        let (vm, mut cpu) = setup(CpuModel::I80386, code);
        vm.write_bytes(0x2400, &[0xa5; 96]).unwrap();
        let before = image(&cpu.cpu);
        let gpr = cpu.cpu.gpr;
        let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(result.steps(), 0);
        assert!(
            matches!(result.stop(), StepStop::Exit(Exit::UnknownInstruction { addr: 0x1000, bytes, len, .. }) if &bytes[..usize::from(*len)] == code)
        );
        assert_eq!(cpu.cpu.rip, 0x1000);
        assert_eq!(cpu.cpu.gpr, gpr);
        assert_eq!(image(&cpu.cpu), before);
        assert_eq!(bytes(&vm, 0x2400, 96), [0xa5; 96]);
    }
}
