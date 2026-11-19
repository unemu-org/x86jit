//! Real-address exception witnesses (decision-18 legacy path).
//! I286 PRM §§5.2.1–5.2.3, Fig. 5-4, Table 5-2 pp. 5-3–7;
//! I386 PRM §§14.1, 14.3, 14.6, Table 14-1 pp. 14-1, 14-3–5
//! (page images checked): IVT entries, FLAGS/CS/IP, IF/TF clearing,
//! no error code, faulting IP for faults and next IP for INTO.
use x86jit_core::{
    fault::{ErrorCode, FaultClass},
    CpuGeneration, CpuModel, CpuState, ExceptionDelivery, Exit, Prot, RegionKind, SegmentId,
    StepOptions, StepStop, Vcpu, Vm, VmConfig,
};

const GENERATIONS: [CpuGeneration; 2] = [CpuGeneration::I286, CpuGeneration::I386];
const CS: u16 = 0x1234;
const SP: u64 = 0xabcd_8000;
const STACK_BASE: u64 = 0xffff0;
const FLAGS: u16 = 0x0fd7;
const SENTINEL: [u8; 10] = [0xa5; 10];

fn physical(linear: u64, a20: bool) -> u64 {
    // Decision-22; SDM Vol 3A §11.7.13.4: the board gate masks bit 20.
    if a20 {
        linear
    } else {
        linear & !(1 << 20)
    }
}

fn setup(
    generation: CpuGeneration,
    delivery: ExceptionDelivery,
    a20: bool,
    code: &[u8],
    ip: u16,
    vector: u8,
) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(0x200000));
    vm.set_cpu_model(match generation {
        CpuGeneration::I286 => CpuModel::I80286,
        _ => CpuModel::I80386,
    });
    vm.set_exception_delivery(delivery);
    vm.map(0, 0x200000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes((u64::from(CS) << 4) + u64::from(ip), code)
        .unwrap();
    // A wrong vector or A20 selection reaches a distinct, terminating handler.
    vm.write_bytes(0x3000, b"\xf4").unwrap();
    for base in [0x400, 0x100400] {
        for index in 0..32 {
            vm.write_bytes(base + index * 4, b"\x00\x30\x00\x00")
                .unwrap();
        }
    }
    let target = 0x40 + u16::from(vector) * 0x10;
    let entry = physical(0x100400 + u64::from(vector) * 4, a20);
    vm.write_bytes(entry, &target.to_le_bytes()).unwrap();
    vm.write_bytes(entry + 2, &0xffffu16.to_le_bytes()).unwrap();
    vm.write_bytes(physical(0xffff0 + u64::from(target), a20), b"\xf4")
        .unwrap();
    // Both physical aliases have sentinels around the exact six-byte frame.
    for enabled in [false, true] {
        vm.write_bytes(physical(STACK_BASE + 0x7ff8, enabled), &SENTINEL)
            .unwrap();
    }
    let mut cpu = vm.new_vcpu();
    cpu.cpu = CpuState::legacy_real_entry([CS, 0x200, 0, 0xffff, 0, 0]);
    cpu.cpu.rip = u64::from(ip);
    cpu.cpu.gpr[0] = 4; // Nonzero dividend; also outside BOUND's [-3, 3].
    cpu.cpu.gpr[4] = SP;
    cpu.cpu.flags.set_flags16(FLAGS);
    cpu.cpu.rflags_system = 0x100;
    cpu.cpu.legacy.idtr.base = 0x100400;
    // The expected entry ends exactly at the inclusive table limit.
    cpu.cpu.legacy.idtr.limit = u16::from(vector) * 4 + 3;
    cpu.set_a20_enabled(a20);
    vm.write_bytes(0x2400, b"\xfd\xff\x03\x00").unwrap();
    (vm, cpu)
}

fn bytes(vm: &Vm, address: u64) -> [u8; 10] {
    let mut result = [0; 10];
    vm.read_bytes(address, &mut result).unwrap();
    result
}

fn witness(
    generation: CpuGeneration,
    code: &[u8],
    ip: u16,
    vector: u8,
    trap: bool,
    limit: Option<(SegmentId, u32)>,
) {
    for a20 in [false, true] {
        for delivery in [ExceptionDelivery::InGuest, ExceptionDelivery::Outward] {
            let label = format!("{generation:?} {delivery:?} A20={a20}");
            let (vm, mut cpu) = setup(generation, delivery, a20, code, ip, vector);
            // Retained inclusive limits are authoritative (decision-18 §1;
            // SDM Vol 3A §§3.4.3, 12.9.2; AMD64 APM Vol 2 §4.1).
            if let Some((segment, value)) = limit {
                cpu.cpu.legacy.segments[segment as usize].limit = value;
            }
            let before = cpu.cpu.clone();
            let saved_ip = if trap { ip + code.len() as u16 } else { ip };
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            // INTO is a software interrupt and retains its existing guest-delivery
            // contract under Outward too (ExceptionDelivery's public contract).
            if delivery == ExceptionDelivery::Outward && !trap {
                let StepStop::Exit(exit @ Exit::Exception { vector: actual, .. }) = run.stop()
                else {
                    panic!("{label}: {:?}", run.stop());
                };
                assert_eq!(*actual, vector, "{label}");
                let metadata = exit.fault_metadata().unwrap();
                assert_eq!(metadata.saved_rip(), u64::from(ip), "{label}");
                assert_eq!(metadata.class(), FaultClass::Fault, "{label}");
                assert_eq!(metadata.error_code(), ErrorCode::NoCode, "{label}");
                assert_eq!((run.steps(), cpu.retired_instructions()), (0, 0), "{label}");
                assert_eq!(format!("{:?}", cpu.cpu), format!("{before:?}"), "{label}");
                for enabled in [false, true] {
                    assert_eq!(bytes(&vm, physical(STACK_BASE + 0x7ff8, enabled)), SENTINEL);
                }
                continue;
            }
            // Fault delivery costs no instruction; its HLT handler consumes the
            // unit. Taken INTO itself consumes the unit before handler execution.
            assert!(
                if trap {
                    matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted))
                } else {
                    matches!(run.stop(), StepStop::Exit(Exit::Hlt))
                },
                "{label}: {:?}",
                run.stop()
            );
            assert_eq!((run.steps(), cpu.retired_instructions()), (1, 1), "{label}");
            let target = 0x40 + u64::from(vector) * 0x10;
            assert_eq!(
                (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
                (0xffff, target + u64::from(!trap), SP - 6),
                "{label}"
            );
            assert_eq!(
                cpu.cpu.legacy.segments[SegmentId::Cs as usize].base,
                0xffff0
            );
            assert!(!cpu.cpu.flags.if_, "{label}");
            assert_eq!(cpu.cpu.rflags_system & 0x100, 0, "{label}");
            assert_eq!(cpu.cpu.gpr[0], before.gpr[0], "{label}");
            assert_eq!(cpu.cpu.gpr[2], before.gpr[2], "{label}");
            let mut expected = SENTINEL;
            expected[2..4].copy_from_slice(&saved_ip.to_le_bytes());
            expected[4..6].copy_from_slice(&CS.to_le_bytes());
            expected[6..8].copy_from_slice(&FLAGS.to_le_bytes());
            assert_eq!(
                bytes(&vm, physical(STACK_BASE + 0x7ff8, a20)),
                expected,
                "{label}: IP/CS/FLAGS only, no error code"
            );
            assert_eq!(bytes(&vm, physical(STACK_BASE + 0x7ff8, !a20)), SENTINEL);
            assert_eq!(cpu.a20_enabled(), a20);
        }
    }
}

#[test]
fn divide_by_zero_delivers_vector_zero_at_faulting_ip() {
    for generation in GENERATIONS {
        witness(generation, b"\x2e\xf6\xf3", 0x1000, 0, false, None); // DIV BL, BL=0
    }
}

#[test]
fn undefined_opcode_delivers_ud_at_faulting_ip() {
    // I286 PRM §5.2.3 p. 5-7; I386 PRM §14.6 Table 14-1 p. 14-5:
    // undefined opcodes fault; 0F FF is unassigned in both instruction sets.
    for generation in GENERATIONS {
        witness(generation, b"\x2e\x0f\xff", 0x1000, 6, false, None);
    }
}

#[test]
fn i286_rejects_386_movzx_through_vector_six() {
    // I286 PRM Appendix B #UD p. B-12;
    // I386 PRM §17 MOVZX p. 17-115: 0F B6 /r is a 386 instruction.
    witness(CpuGeneration::I286, b"\x0f\xb6\xc0", 0x1000, 6, false, None);
}

#[test]
fn instruction_crossing_ffff_delivers_gp_at_instruction_start() {
    // I286 PRM Appendix B #GP p. B-9: execution beyond the CS limit;
    // I386 PRM §14.6 Table 14-1 p. 14-5: execution past FFFFh is #GP.
    for generation in GENERATIONS {
        witness(generation, b"\xb8\x34\x12", 0xffff, 13, false, None);
    }
}

#[test]
fn word_at_ffff_uses_generation_specific_ss_and_ds_vectors() {
    // I286 PRM Appendix B MOV p. B-74: interrupt 13 even with SS.
    // I386 PRM §§14.1, 14.6 Table 14-1: SS overrun is 12, DS overrun 13.
    // Also witnessed outward in legacy_i286::real_word_at_ffff_faults_with_gp_for_ss_and_ds
    // and legacy_real::real_word_at_ffff_faults_with_ss_for_stack_and_gp_for_data.
    for generation in GENERATIONS {
        for ss in [false, true] {
            let vector = if ss && generation == CpuGeneration::I386 {
                12
            } else {
                13
            };
            // Keep the delivery stack valid; fault the explicit word at FFFFh.
            let code = if ss {
                b"\x36\xa1\xff\xff"
            } else {
                b"\x3e\xa1\xff\xff"
            };
            witness(generation, code, 0x1000, vector, false, None);
        }
    }
}

#[test]
fn bound_delivers_br_at_faulting_ip() {
    for generation in GENERATIONS {
        witness(generation, b"\x62\x06\x00\x04", 0x1000, 5, false, None);
    }
}

#[test]
fn into_delivers_overflow_at_next_ip_under_both_delivery_options() {
    for generation in GENERATIONS {
        for limit in [None, Some((SegmentId::Ss, 0x7fff))] {
            witness(generation, b"\x2e\xce", 0x1000, 4, true, limit);
        }
    }
}

#[test]
fn retained_cs_and_ds_limits_fault_before_ffff_and_deliver_through_ivt() {
    for generation in GENERATIONS {
        witness(
            generation,
            b"\xb8\x34\x12",
            0x1000,
            13,
            false,
            Some((SegmentId::Cs, 0x1001)),
        );
        witness(
            generation,
            b"\xa1\x00\x04",
            0x1000,
            13,
            false,
            Some((SegmentId::Ds, 0x400)),
        );
    }
}
