//! I80386/387 protected-mode witnesses. Printed manual pages are cited below.
#[path = "../support/descriptor_fixture.rs"]
#[allow(dead_code)]
mod fixture;

use x86jit_core::{
    CpuModel, Exit, FpuModel, SegmentCache, SegmentId, StepOptions, StepStop, Vcpu, Vm,
};

const DATA: u64 = 0x20000;
const THREE: [u8; 10] = [0, 0, 0, 0, 0, 0, 0, 0xc0, 0, 0x40];

fn setup(wide: bool, real: bool, code: &[u8]) -> (Vm, Vcpu) {
    setup_model(CpuModel::I80386, wide, real, code)
}

fn setup_model(model: CpuModel, wide: bool, real: bool, code: &[u8]) -> (Vm, Vcpu) {
    setup_model_delivery(
        model,
        wide,
        real,
        code,
        x86jit_core::ExceptionDelivery::Outward,
    )
}

fn setup_model_delivery(
    model: CpuModel,
    wide: bool,
    real: bool,
    code: &[u8],
    delivery: x86jit_core::ExceptionDelivery,
) -> (Vm, Vcpu) {
    let (vm, mut cpu) = fixture::setup_delivery(model, code, delivery);
    cpu.set_fpu_model(FpuModel::Present);
    cpu.cpu.legacy.cr0 = 0x10 | u32::from(!real);
    cpu.cpu.cs = if real { 0 } else { 8 };
    cpu.cpu.ds = if real { 0x2000 } else { 0x18 };
    cpu.cpu.ss = if real { 0x2000 } else { 0x20 };
    if !real {
        vm.write_bytes(0x11000, code).unwrap();
    }
    for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
        *cache = SegmentCache {
            base: if i == SegmentId::Cs as usize {
                if real {
                    0
                } else {
                    0x10000
                }
            } else {
                DATA as u32
            },
            limit: 0xffff,
            attributes: if i == SegmentId::Cs as usize {
                0x9b
            } else {
                0x93
            } | if wide { 0x4000 } else { 0 },
            unusable: 0,
            reserved: 0,
        };
    }
    (vm, cpu)
}

fn mem(wide: bool, op: u8, group: u8, p66: bool, p67: bool) -> Vec<u8> {
    let mut code = Vec::new();
    if p66 {
        code.push(0x66);
    }
    if p67 {
        code.push(0x67);
    }
    code.extend([op, (group << 3) | if wide != p67 { 5 } else { 6 }]);
    code.extend(if wide != p67 {
        0x400u32.to_le_bytes().to_vec()
    } else {
        0x400u16.to_le_bytes().to_vec()
    });
    code
}

fn bytes(vm: &Vm, address: u64, len: usize) -> Vec<u8> {
    let mut b = vec![0; len];
    vm.read_bytes(address, &mut b).unwrap();
    b
}

fn seed(cpu: &mut Vcpu) {
    cpu.cpu.fpu_top = 5;
    cpu.cpu.fpu_empty = !(1 << 5);
    cpu.cpu.fpr[5] = THREE;
    cpu.cpu.fpu_cw = 0x037f;
    cpu.cpu.fpu_sw = 0x4100;
    cpu.cpu.fpu_fip = 0x1234abcd;
    cpu.cpu.fpu_fcs = 0x38;
    cpu.cpu.fpu_fop = 0x345;
    cpu.cpu.fpu_fdp = 0x5678cdef;
    cpu.cpu.fpu_fds = 0x40;
}

fn fpu(cpu: &Vcpu) -> String {
    format!(
        "{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}",
        cpu.cpu.fpr,
        cpu.cpu.fpu_top,
        cpu.cpu.fpu_empty,
        cpu.cpu.fpu_cw,
        cpu.cpu.fpu_sw,
        cpu.cpu.fpu_fip,
        cpu.cpu.fpu_fcs,
        cpu.cpu.fpu_fop,
        cpu.cpu.fpu_fdp,
        cpu.cpu.fpu_fds,
        cpu.cpu.flags
    )
}

// SDM Vol 1 §8.2 p. 8-13: real and integer formats; §8.3.3
// pp. 8-15–16: loads and stores. These exact representations encode 3.
#[test]
fn every_memory_format_and_size_prefix_matches_real_mode() {
    let forms: Vec<(u8, u8, u8, u8, Vec<u8>)> = vec![
        (0xd9, 0, 0xd9, 3, 3f32.to_le_bytes().to_vec()),
        (0xdd, 0, 0xdd, 3, 3f64.to_le_bytes().to_vec()),
        (0xdb, 5, 0xdb, 7, THREE.to_vec()),
        (0xdf, 0, 0xdf, 3, 3i16.to_le_bytes().to_vec()),
        (0xdb, 0, 0xdb, 3, 3i32.to_le_bytes().to_vec()),
        (0xdf, 5, 0xdf, 7, 3i64.to_le_bytes().to_vec()),
    ];
    for wide in [false, true] {
        for p66 in [false, true] {
            for p67 in [false, true] {
                for (load, lg, store, sg, value) in &forms {
                    let mut code = mem(wide, *load, *lg, p66, p67);
                    code.extend(mem(wide, *store, *sg, p66, p67));
                    let mut results = Vec::new();
                    for real in [true, false] {
                        let (vm, mut cpu) = setup(wide, real, &code);
                        vm.write_bytes(DATA + 0x400, value).unwrap();
                        fixture::steps(&mut cpu, &vm, 1);
                        assert_eq!(cpu.cpu.fpr[cpu.cpu.fpu_top as usize], THREE);
                        assert_eq!(cpu.cpu.fpu_fip, 0x1000);
                        assert_eq!(cpu.cpu.fpu_fcs, cpu.cpu.cs);
                        assert_eq!(cpu.cpu.fpu_fdp, 0x400);
                        assert_eq!(cpu.cpu.fpu_fds, cpu.cpu.ds);
                        let esc = code.iter().position(|b| *b == *load).unwrap();
                        assert_eq!(
                            cpu.cpu.fpu_fop,
                            (u16::from(load & 7) << 8) | u16::from(code[esc + 1])
                        );
                        fixture::steps(&mut cpu, &vm, 1);
                        assert_eq!(bytes(&vm, DATA + 0x400, value.len()), *value);
                        results.push((
                            cpu.cpu.fpr,
                            cpu.cpu.fpu_top,
                            cpu.cpu.fpu_empty,
                            cpu.cpu.fpu_sw,
                        ));
                    }
                    assert_eq!(results[0], results[1]);
                }
            }
        }
    }
}

// SDM Vol 1 §§8.3.5–8.3.6 pp. 8-17–18: real/integer memory and
// stack arithmetic and comparisons.
#[test]
fn arithmetic_and_comparison_groups_match_real_mode() {
    for wide in [false, true] {
        for (op, value) in [
            (0xd8, 3f32.to_le_bytes().to_vec()),
            (0xdc, 3f64.to_le_bytes().to_vec()),
            (0xda, 3i32.to_le_bytes().to_vec()),
            (0xde, 3i16.to_le_bytes().to_vec()),
        ] {
            for group in 0..8 {
                let code = mem(wide, op, group, false, false);
                let mut results = Vec::new();
                for real in [true, false] {
                    let (vm, mut cpu) = setup(wide, real, &code);
                    seed(&mut cpu);
                    vm.write_bytes(DATA + 0x400, &value).unwrap();
                    fixture::steps(&mut cpu, &vm, 1);
                    if matches!(group, 2 | 3) {
                        assert_eq!(cpu.cpu.fpu_sw & 0x4500, 0x4000);
                    }
                    results.push((
                        cpu.cpu.fpr,
                        cpu.cpu.fpu_top,
                        cpu.cpu.fpu_empty,
                        cpu.cpu.fpu_sw,
                    ));
                }
                assert_eq!(results[0], results[1], "{op:x}/{group}");
            }
        }
        for code in [
            &[0xd8, 0xc0][..],
            &[0xdc, 0xc0],
            &[0xde, 0xc0],
            &[0xd8, 0xd0],
            &[0xdd, 0xe0],
            &[0xda, 0xe9],
            &[0xd9, 0xe8],
        ] {
            let mut results = Vec::new();
            for real in [true, false] {
                let (vm, mut cpu) = setup(wide, real, code);
                seed(&mut cpu);
                cpu.cpu.fpu_empty &= !(1 << 6);
                cpu.cpu.fpr[6] = THREE;
                fixture::steps(&mut cpu, &vm, 1);
                results.push((
                    cpu.cpu.fpr,
                    cpu.cpu.fpu_top,
                    cpu.cpu.fpu_empty,
                    cpu.cpu.fpu_sw,
                ));
            }
            assert_eq!(results[0], results[1]);
        }
    }
}

// SDM Vol 1 §8.1.10 Figs. 8-9/8-11 pp. 8-11–12; Intel486 PRM
// §15.1.6 Figs. 15-6/15-8 pp. 15-8–9 (second witness for the 387).
// ASSUMED: reserved words save as FFFFh, matching the existing engine policy.
fn environment(short: bool) -> Vec<u8> {
    if short {
        [0x0360u16, 0x6900, 0xf3ff, 0xabcd, 0x38, 0xcdef, 0x40]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect()
    } else {
        [
            0xffff0360u32,
            0xffff6900,
            0xfffff3ff,
            0x1234abcd,
            0x03450038,
            0x5678cdef,
            0xffff0040,
        ]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect()
    }
}

#[test]
fn protected_environment_and_full_state_images() {
    for wide in [false, true] {
        for p66 in [false, true] {
            for p67 in [false, true] {
                let short = wide == p66;
                for full in [false, true] {
                    for waiting in [false, true] {
                        let op = if full { 0xdd } else { 0xd9 };
                        let mut code = if waiting { vec![0x9b] } else { Vec::new() };
                        code.extend(mem(wide, op, 6, p66, p67));
                        code.extend(mem(wide, op, 4, p66, p67));
                        let (vm, mut cpu) = setup(wide, false, &code);
                        seed(&mut cpu);
                        cpu.cpu.fpu_cw = 0x0360;
                        let original = fpu(&cpu);
                        let mut expected = environment(short);
                        if full {
                            for i in 0..8 {
                                expected.extend(cpu.cpu.fpr[(5 + i) & 7]);
                            }
                        }
                        vm.write_bytes(DATA + 0x400, &vec![0xa5; expected.len() + 1])
                            .unwrap();
                        fixture::steps(&mut cpu, &vm, 1 + u64::from(waiting));
                        assert_eq!(cpu.cpu.fpu_cw, 0x037f);
                        assert_eq!(bytes(&vm, DATA + 0x400, expected.len()), expected);
                        assert_eq!(bytes(&vm, DATA + 0x400 + expected.len() as u64, 1), [0xa5]);
                        if full {
                            assert_eq!(cpu.cpu.fpu_empty, 0xff);
                        }
                        // ASSUMED: a 16-bit protected restore retains FOP; that
                        // layout supplies no opcode. Seed it to witness retention.
                        cpu.cpu.fpu_fop = 0x345;
                        fixture::steps(&mut cpu, &vm, 1);
                        if short {
                            assert_eq!(cpu.cpu.fpu_fip, 0xabcd);
                            assert_eq!(cpu.cpu.fpu_fdp, 0xcdef);
                            cpu.cpu.fpu_fip = 0x1234abcd;
                            cpu.cpu.fpu_fdp = 0x5678cdef;
                        }
                        assert_eq!(fpu(&cpu), original);
                    }
                }
            }
        }
    }
}

// SDM Vol 1 §§8.3.11–8.3.12 pp. 8-23–24: control operations; the waiting encodings
// are WAIT followed by the corresponding no-wait ESC (80386 PRM §17 p.17-171).
#[test]
fn control_words_status_ax_init_and_clear() {
    for wide in [false, true] {
        let mut code = mem(wide, 0xd9, 5, false, false); // FLDCW
        code.extend(mem(wide, 0xd9, 7, false, false)); // FNSTCW
        code.extend([0xdf, 0xe0]); // FNSTSW AX
        code.extend(mem(wide, 0xdd, 7, false, false)); // FNSTSW m16
        code.extend([0x9b, 0xdb, 0xe2, 0x9b, 0xdb, 0xe3]); // FCLEX; FINIT
        let (vm, mut cpu) = setup(wide, false, &code);
        seed(&mut cpu);
        vm.write_bytes(DATA + 0x400, &0x077fu16.to_le_bytes())
            .unwrap();
        fixture::steps(&mut cpu, &vm, 2);
        assert_eq!(cpu.cpu.fpu_cw, 0x077f);
        assert_eq!(bytes(&vm, DATA + 0x400, 2), 0x077fu16.to_le_bytes());
        cpu.cpu.gpr[0] = 0xabcd0000;
        fixture::steps(&mut cpu, &vm, 2);
        assert_eq!(cpu.cpu.gpr[0], 0xabcd6900);
        assert_eq!(bytes(&vm, DATA + 0x400, 2), 0x6900u16.to_le_bytes());
        cpu.cpu.fpu_sw |= 0x3f; // masked exceptions; WAIT completes
        fixture::steps(&mut cpu, &vm, 2);
        assert_eq!(cpu.cpu.fpu_sw & 0x80ff, 0);
        fixture::steps(&mut cpu, &vm, 2);
        assert_eq!(cpu.cpu.fpu_cw, 0x037f);
        assert_eq!(cpu.cpu.fpu_empty, 0xff);
        assert_eq!(cpu.cpu.fpu_top, 0);
        assert_eq!(cpu.cpu.fpu_sw, 0);
    }
}

// 80386 PRM §§6.3.1.1–6.3.1.2 pp. 6-4–6-6; SDM Vol 3A §§6.3–6.4:
// the whole operand needs access, with #SS(0) for SS and #GP(0) otherwise.
#[test]
fn segment_limits_and_write_permissions_are_atomic() {
    for wide in [false, true] {
        for (op, group) in [
            (0xdb, 5),
            (0xdb, 7),
            (0xd9, 4),
            (0xd9, 6),
            (0xdd, 4),
            (0xdd, 6),
            (0xdf, 7),
        ] {
            for stack in [false, true] {
                let mut code = if stack { vec![0x36] } else { Vec::new() };
                code.extend(mem(wide, op, group, false, false));
                let (vm, mut cpu) = setup(wide, false, &code);
                seed(&mut cpu);
                let segment = if stack { SegmentId::Ss } else { SegmentId::Ds };
                cpu.cpu.legacy.segments[segment as usize].limit = 0x401;
                let before = bytes(&vm, DATA + 0x400, 108);
                fixture::fault(&mut cpu, &vm, if stack { 12 } else { 13 }, Some(0));
                assert_eq!(bytes(&vm, DATA + 0x400, 108), before);
            }
        }
        let (vm, mut cpu) = setup(wide, false, &mem(wide, 0xdb, 7, false, false));
        seed(&mut cpu);
        cpu.cpu.legacy.segments[SegmentId::Ds as usize].attributes &= !2;
        fixture::fault(&mut cpu, &vm, 13, Some(0));
        let (vm, mut cpu) = setup(wide, false, &mem(wide, 0xd9, 0, false, false));
        cpu.cpu.legacy.segments[SegmentId::Ds as usize].unusable = 1;
        fixture::fault(&mut cpu, &vm, 13, Some(0));
    }
}

// 80386 PRM §§11.1.1–11.1.4 pp. 11-1–2: ET chooses the bus protocol,
// EM/TS gate ESC, MP && TS gates WAIT independently of EM and socket presence.
#[test]
fn nm_gate_and_absent_socket_share_real_mode_policy() {
    for wide in [false, true] {
        for real in [false, true] {
            for model in [FpuModel::Present, FpuModel::None] {
                for bits in 0..16u32 {
                    for wait in [false, true] {
                        let code = if wait {
                            vec![0x9b]
                        } else {
                            mem(wide, 0xd9, 0, false, false)
                        };
                        let (vm, mut cpu) = setup(wide, real, &code);
                        cpu.set_fpu_model(model);
                        cpu.cpu.legacy.cr0 = u32::from(!real) | bits << 1;
                        let msw = cpu.cpu.legacy.cr0;
                        let nm = if wait {
                            msw & 0xa == 0xa
                        } else {
                            msw & 0xc != 0
                        };
                        if nm {
                            // The CPU gate wins over even an invalid operand segment.
                            cpu.cpu.legacy.segments[SegmentId::Ds as usize].limit = 0;
                            fixture::fault(&mut cpu, &vm, 7, None);
                        } else {
                            let before = fpu(&cpu);
                            fixture::steps(&mut cpu, &vm, 1);
                            if model == FpuModel::None {
                                assert_eq!(fpu(&cpu), before);
                            }
                        }
                    }
                }
                let (vm, mut cpu) = setup(wide, real, &mem(wide, 0xd9, 3, false, false));
                cpu.set_fpu_model(FpuModel::None);
                let before = fpu(&cpu);
                let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert!(
                    matches!(r.stop(), StepStop::Exit(Exit::UnknownInstruction { .. })),
                    "{:?}",
                    r.stop()
                );
                assert_eq!(r.steps(), 0);
                assert_eq!(fpu(&cpu), before);
            }
        }
    }
}

// 80386 PRM §11.1.5.3 p. 11-4: an unmasked numeric error asserts ERROR#;
// a subsequent WAIT or waiting ESC raises exception 16 in either regime.
// ASSUMED engine boundary: model the direct ERROR# connection. IRQ13 routing
// is a board concern; Exit::Exception carries the CPU's vector 16 outward.
#[test]
fn error_pin_reports_vector_16_on_wait_and_waiting_esc() {
    for wide in [false, true] {
        for real in [false, true] {
            for next in [&[0x9b][..], &[0xd9, 0xe8]] {
                let mut code = vec![0xd8, 0xf1]; // FDIV ST0,ST1 = 3/0
                code.extend(next);
                let (vm, mut cpu) = setup(wide, real, &code);
                seed(&mut cpu);
                cpu.cpu.fpu_cw &= !4;
                cpu.cpu.fpu_sw = 0;
                cpu.cpu.fpu_empty &= !(1 << 6);
                cpu.cpu.fpr[6] = [0; 10];
                fixture::steps(&mut cpu, &vm, 1);
                assert_eq!(cpu.cpu.fpu_sw & 0x84, 0x84);
                fixture::fault(&mut cpu, &vm, 16, None);
            }
        }
    }
}

// SDM Vol 1 §8.1.3.3 p. 8-5; Intel486 PRM §15.1.2 pp. 15-2–3
// (387 second witness, §25.1 pp. 25-1–2): ES/B summarize unmasked flags.
// 80286/287 PRM Part II Ch. 1, "The NPX Status Word", p. 1-10:
// ES has the same rule; B is execution/error busy, so it is not asserted here.
const PENDING_CONTEXTS: [(CpuModel, bool, bool); 4] = [
    (CpuModel::I80386, false, false),
    (CpuModel::I80386, true, false),
    (CpuModel::I80386, false, true),
    (CpuModel::I80286, false, true),
];

fn assert_pending(cpu: &Vcpu, model: CpuModel, pending: bool) {
    let summary = if model == CpuModel::I80386 {
        0x8080
    } else {
        0x80
    };
    assert_eq!(cpu.cpu.fpu_sw & summary, if pending { summary } else { 0 });
}

fn seed_divide(cpu: &mut Vcpu, unmasked: bool) {
    seed(cpu);
    cpu.cpu.fpu_cw = if unmasked { 0x037b } else { 0x037f };
    cpu.cpu.fpu_sw = 0;
    cpu.cpu.fpu_empty &= !(1 << 6);
    cpu.cpu.fpr[6] = [0; 10];
}

// SDM Vol 2A §3.3 FLDCW p. 3-356; 80286/287 PRM Part II Ch. 2,
// "FLDCW", p. 2-16: unmasking a sticky exception activates pending error.
#[test]
fn x87_fldcw_unmasks_a_masked_divide_on_the_next_wait() {
    for (model, wide, real) in PENDING_CONTEXTS {
        for next in [&[0x9b][..], &[0xd9, 0xe8]] {
            let mut code = vec![0xd8, 0xf1]; // FDIV ST0,ST1 = 3/0
            code.extend(mem(wide, 0xd9, 5, false, false)); // FLDCW
            code.extend([0xdf, 0xe0]); // FNSTSW AX must not report the error
            code.extend(next);
            let (vm, mut cpu) = setup_model(model, wide, real, &code);
            seed_divide(&mut cpu, false);
            vm.write_bytes(DATA + 0x400, &0x037bu16.to_le_bytes())
                .unwrap();
            fixture::steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.fpu_sw & 0x8084, 4);
            fixture::steps(&mut cpu, &vm, 1);
            assert_pending(&cpu, model, true);
            assert_eq!(cpu.cpu.fpu_sw & 4, 4);
            fixture::steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.gpr[0] & 0x84, 0x84);
            fixture::fault(&mut cpu, &vm, 16, None);
        }
    }
}

// SDM Vol 2A §3.3 FSTENV/FNSTENV p. 3-390; 80286/287 PRM Part II
// Ch. 2, "FSTENV/FNSTENV", p. 2-20: save before masking all exceptions.
#[test]
fn x87_fnstenv_masks_an_unmasked_divide_before_the_next_wait() {
    for (model, wide, real) in PENDING_CONTEXTS {
        for next in [&[0x9b][..], &[0xd9, 0xe8]] {
            let mut code = vec![0xd8, 0xf1];
            code.extend(mem(wide, 0xd9, 6, false, false)); // FNSTENV
            code.extend(next);
            let (vm, mut cpu) = setup_model(model, wide, real, &code);
            seed_divide(&mut cpu, true);
            fixture::steps(&mut cpu, &vm, 1);
            assert_pending(&cpu, model, true);
            fixture::steps(&mut cpu, &vm, 1);
            let saved = bytes(&vm, DATA + 0x400, if wide { 28 } else { 14 });
            assert_eq!(u16::from_le_bytes(saved[..2].try_into().unwrap()), 0x037b);
            let sw = if wide { 4 } else { 2 };
            assert_eq!(
                u16::from_le_bytes(saved[sw..sw + 2].try_into().unwrap()) & 0x84,
                0x84
            );
            assert_eq!(cpu.cpu.fpu_cw & 0x3f, 0x3f);
            assert_eq!(cpu.cpu.fpu_sw & 4, 4);
            assert_pending(&cpu, model, false);
            fixture::steps(&mut cpu, &vm, 1);
        }
    }
}

// SDM Vol 2A §3.3 FLDENV p. 3-358, FRSTOR p. 3-373;
// 80286/287 PRM Part II Ch. 2, "FRSTOR"/"FLDENV", pp. 2-20/2-21:
// flags and masks in the loaded image determine the next waiting op's error.
#[test]
fn x87_environment_restores_recompute_pending_from_flags_and_masks() {
    for (model, wide, real) in PENDING_CONTEXTS {
        for p66 in [false, true] {
            if p66 && model == CpuModel::I80286 {
                continue;
            }
            let short = wide == p66;
            for full in [false, true] {
                for flag in [1u16, 2, 4, 8, 16, 32] {
                    for pending in [false, true] {
                        for next in [&[0x9b][..], &[0xd9, 0xe8]] {
                            let mut code = mem(wide, if full { 0xdd } else { 0xd9 }, 4, p66, false);
                            code.extend(next);
                            let (vm, mut cpu) = setup_model(model, wide, real, &code);
                            let mut image =
                                vec![0; if short { 14 } else { 28 } + if full { 80 } else { 0 }];
                            let cw = if pending { 0x037f & !flag } else { 0x037f };
                            image[..2].copy_from_slice(&cw.to_le_bytes());
                            let sw = if short { 2 } else { 4 };
                            // Deliberately stale summary in both directions.
                            let status = flag | if pending { 0 } else { 0x8080 };
                            image[sw..sw + 2].copy_from_slice(&status.to_le_bytes());
                            let tw = if short { 4 } else { 8 };
                            image[tw..tw + 2].copy_from_slice(&0xffffu16.to_le_bytes());
                            vm.write_bytes(DATA + 0x400, &image).unwrap();
                            fixture::steps(&mut cpu, &vm, 1);
                            assert_pending(&cpu, model, pending);
                            assert_eq!(cpu.cpu.fpu_sw & 0x3f, flag);
                            if pending {
                                fixture::fault(&mut cpu, &vm, 16, None);
                            } else {
                                fixture::steps(&mut cpu, &vm, 1);
                            }
                        }
                    }
                }
            }
        }
    }
}

// SDM Vol 2A §3.3 FCLEX/FNCLEX p. 3-320, FINIT/FNINIT p. 3-345,
// FSAVE/FNSAVE p. 3-375: clearing or reinitializing removes pending errors.
#[test]
fn x87_clear_init_and_save_remove_pending_errors() {
    for (model, wide, real) in PENDING_CONTEXTS {
        for control in [
            vec![0xdb, 0xe2],
            vec![0xdb, 0xe3],
            mem(wide, 0xdd, 6, false, false),
        ] {
            for waiting in [false, true] {
                let mut code = control.clone();
                if waiting {
                    code.insert(0, 0x9b);
                }
                code.extend([0x9b, 0xd9, 0xe8]);
                let (vm, mut cpu) = setup_model(model, wide, real, &code);
                seed_divide(&mut cpu, true);
                cpu.cpu.fpu_sw = 0x8084;
                if waiting {
                    fixture::fault(&mut cpu, &vm, 16, None);
                } else {
                    fixture::steps(&mut cpu, &vm, 1);
                    assert_pending(&cpu, model, false);
                    assert_eq!(cpu.cpu.fpu_sw & 0x3f, 0);
                    fixture::steps(&mut cpu, &vm, 2);
                }
            }
        }
    }
}

// External tests below witness engine/board policy from spec.md §17.9.
// CPU recognition: H1, 80286 PRM Part I Appendix B #MF p. B-9;
// H2, Part II Ch. 1 Fig. 1-4 p. 1-10; H3, 80386 PRM §§11.1.2/11.1.5.3.
// IRQ13 latching and F0h acknowledgement are embedder requirements.
use x86jit_core::{
    DeliveryKind, ExceptionDelivery, NumericErrorConfigError as ConfigError,
    NumericErrorEvent as Event, NumericErrorRoute as Route, ResetOptions,
};

fn external(model: CpuModel, checked: bool, code: &[u8]) -> (Vm, Vcpu) {
    let (vm, mut cpu) = setup_model(model, false, true, code);
    if checked {
        cpu.cpu.legacy.segments[SegmentId::Fs as usize].attributes = 0x92;
    }
    cpu.set_numeric_error_route(Route::External).unwrap();
    seed_divide(&mut cpu, true);
    cpu.enable_delivery_log(32).unwrap();
    (vm, cpu)
}

fn handoff(cpu: &mut Vcpu, vm: &Vm, budget: u64, steps: u64, ip: u64, event: Event) {
    let before = cpu.retired_instructions();
    let r = cpu.run_steps(vm, budget, StepOptions::default()).unwrap();
    assert_eq!(r.steps(), steps, "{:?}", r.stop());
    assert_eq!(cpu.retired_instructions() - before, steps);
    assert!(
        matches!(r.stop(), StepStop::Exit(Exit::ExternalNumericError { addr, event: e })
        if *addr == ip && *e == event),
        "{:?}",
        r.stop()
    );
    assert_eq!(cpu.cpu.rip, ip);
}

#[test]
fn external_producer_wait_ack_and_second_episode() {
    for model in fixture::MODELS {
        for checked in [false, true] {
            // Divide; WAIT; FNSTSW AX; FNCLEX; divide; WAIT.
            let (vm, mut cpu) = external(
                model,
                checked,
                &[0xd8, 0xf1, 0x9b, 0xdf, 0xe0, 0xdb, 0xe2, 0xd8, 0xf1, 0x9b],
            );
            assert!(!cpu.cpu.flags.if_);
            let destination = cpu.cpu.fpr[5];
            handoff(&mut cpu, &vm, 1, 1, 0x1002, Event::Asserted);
            assert!(cpu.numeric_error_pending());
            assert_eq!(cpu.cpu.fpu_sw & 0x84, 0x84);
            assert_eq!(cpu.cpu.fpr[5], destination); // pre-operation ZE
            let image = fpu(&cpu);
            let checkpoint = cpu.cpu.clone();
            for _ in 0..2 {
                handoff(&mut cpu, &vm, 4, 0, 0x1002, Event::Waiting);
                assert_eq!(fpu(&cpu), image);
                assert_eq!(format!("{:?}", cpu.cpu), format!("{checkpoint:?}"));
            }
            cpu.ack_external_numeric_error().unwrap();
            cpu.ack_external_numeric_error().unwrap();
            assert_eq!(fpu(&cpu), image);
            assert!(cpu.numeric_error_pending());
            // Acknowledgement does not manufacture another assertion with ES=1.
            fixture::steps(&mut cpu, &vm, 2);
            assert_eq!(cpu.cpu.gpr[0] & 0x84, 0x84);
            fixture::steps(&mut cpu, &vm, 1);
            assert!(!cpu.numeric_error_pending());
            handoff(&mut cpu, &vm, 99, 1, 0x1009, Event::Asserted);
            assert_eq!(cpu.delivery_log().unwrap().total(), 0);
            // A restored reported/held checkpoint must stall, not assert twice.
            cpu.cpu = checkpoint;
            handoff(&mut cpu, &vm, 1, 0, 0x1002, Event::Waiting);
        }
    }
}

#[test]
fn external_entry_zero_budget_checkpoint_and_reset() {
    for model in fixture::MODELS {
        for checked in [false, true] {
            let (vm, mut cpu) = external(model, checked, &[0x9b, 0xd9, 0xe8]);
            cpu.cpu.fpu_sw = 0x8084;
            let unreported = cpu.cpu.clone();
            let request = cpu.exit_request();
            request.request_exit();
            fixture::steps(&mut cpu, &vm, 0);
            assert!(request.is_requested());
            assert_eq!(format!("{:?}", cpu.cpu), format!("{unreported:?}"));
            let interrupted = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(interrupted.steps(), 0);
            assert!(matches!(
                interrupted.stop(),
                StepStop::Exit(Exit::Interrupted)
            ));
            assert_eq!(
                cpu.cpu.external_numeric_error,
                unreported.external_numeric_error
            );
            handoff(&mut cpu, &vm, 4, 0, 0x1000, Event::Asserted);
            let held = cpu.cpu.clone();
            cpu.reset_execution_state(ResetOptions::default());
            handoff(&mut cpu, &vm, 1, 0, 0x1000, Event::Waiting);
            cpu.reset_execution_state(ResetOptions::default().with_x87_init(true));
            assert!(!cpu.numeric_error_pending());
            handoff(&mut cpu, &vm, 1, 0, 0x1000, Event::Waiting);
            let cleared = cpu.cpu.clone();
            cpu.ack_external_numeric_error().unwrap();
            let acked = cpu.cpu.clone();
            fixture::steps(&mut cpu, &vm, 1);
            cpu.cpu = acked;
            fixture::steps(&mut cpu, &vm, 1);
            cpu.cpu = cleared;
            handoff(&mut cpu, &vm, 1, 0, 0x1000, Event::Waiting);
            cpu.cpu = held;
            cpu.ack_external_numeric_error().unwrap();
            let acked_pending = cpu.cpu.clone();
            fixture::steps(&mut cpu, &vm, 1);
            cpu.cpu = acked_pending;
            fixture::steps(&mut cpu, &vm, 1);
            cpu.cpu = unreported;
            handoff(&mut cpu, &vm, 1, 0, 0x1000, Event::Asserted);
            assert_eq!(cpu.delivery_log().unwrap().total(), 0);
        }
    }
}

#[test]
fn external_run_returns_before_an_unbounded_integer_loop() {
    for model in fixture::MODELS {
        let (vm, mut cpu) = external(model, false, &[0xd8, 0xf1, 0xeb, 0xfe]);
        let before = cpu.cpu.clone();
        // Fail promptly if notification is absent before probing unbounded run.
        assert!(matches!(
            cpu.run(&vm, Some(1)),
            Exit::ExternalNumericError {
                addr: 0x1002,
                event: Event::Asserted,
            }
        ));
        cpu.cpu = before;
        assert!(matches!(
            cpu.run(&vm, None),
            Exit::ExternalNumericError {
                addr: 0x1002,
                event: Event::Asserted,
            }
        ));
        assert_eq!(cpu.retired_instructions(), 2); // CpuState restore keeps accounting
        assert!(matches!(cpu.run(&vm, Some(8)), Exit::BudgetExhausted));
    }
}

#[test]
fn external_assertion_precedes_owned_single_step() {
    // Engine handoff order (spec.md §17.9); TF recognition: 80286 PRM
    // §9.7.1 p. 9-14; 80386 PRM §12.3.1.4 p. 12-8.
    for model in fixture::MODELS {
        for checked in [false, true] {
            for stepped in [false, true] {
                let (vm, mut cpu) = external(model, checked, &[0xd8, 0xf1, 0x9b]);
                real_handler(&vm, 1, &[0xf4]); // HLT in #DB handler
                cpu.cpu.rflags_system |= 0x100;
                if stepped {
                    handoff(&mut cpu, &vm, 1, 1, 0x1002, Event::Asserted);
                } else {
                    assert!(matches!(
                        cpu.run(&vm, Some(8)),
                        Exit::ExternalNumericError {
                            addr: 0x1002,
                            event: Event::Asserted
                        }
                    ));
                }
                assert_eq!(cpu.retired_instructions(), 1);
                assert_eq!(cpu.delivery_log().unwrap().total(), 0);
                let before = format!("{:?}", cpu.cpu);
                assert_eq!(
                    cpu.set_numeric_error_route(Route::Internal),
                    Err(ConfigError::ContinuationInFlight)
                );
                assert_eq!(format!("{:?}", cpu.cpu), before);
                fixture::steps(&mut cpu, &vm, 0);
                assert_eq!(cpu.delivery_log().unwrap().total(), 0);
                if stepped {
                    let result = cpu.run_steps(&vm, 8, StepOptions::default()).unwrap();
                    assert_eq!(result.steps(), 1); // only handler HLT retires
                    assert!(
                        matches!(result.stop(), StepStop::Exit(Exit::Hlt)),
                        "{:?}",
                        result.stop()
                    );
                } else {
                    assert!(matches!(cpu.run(&vm, Some(8)), Exit::Hlt));
                }
                assert_eq!(cpu.retired_instructions(), 2);
                assert_eq!(cpu.delivery_log().unwrap().total(), 1);
                let record = cpu.delivery_log().unwrap().iter().next().unwrap();
                assert_eq!(
                    (record.vector, record.kind, record.saved_ip),
                    (1, DeliveryKind::Trap, 0x1002)
                );
                cpu.set_numeric_error_route(Route::External).unwrap();
                cpu.cpu.rip = 0x1002;
                handoff(&mut cpu, &vm, 8, 0, 0x1002, Event::Waiting);
                assert_eq!(cpu.delivery_log().unwrap().total(), 1);
            }
        }
    }
}

#[test]
fn external_configuration_rejects_debug_held_after_out() {
    for model in fixture::MODELS {
        for checked in [false, true] {
            let (vm, mut cpu) = external(model, checked, &[0xe6, 0x80, 0x90]);
            real_handler(&vm, 1, &[0xf4]);
            cpu.cpu.rflags_system |= 0x100;
            assert!(matches!(cpu.run(&vm, Some(8)), Exit::PortIo { .. }));
            assert!(cpu.cpu.pending_port_in.is_none());
            assert!(!cpu.cpu.mmio_retry_active);
            let before = format!("{:?}", cpu.cpu);
            assert_eq!(
                cpu.set_numeric_error_route(Route::Internal),
                Err(ConfigError::ContinuationInFlight)
            );
            assert_eq!(
                cpu.ack_external_numeric_error(),
                Err(ConfigError::ContinuationInFlight)
            );
            assert_eq!(format!("{:?}", cpu.cpu), before);
            assert_eq!(cpu.numeric_error_route(), Route::External);
            assert!(matches!(cpu.run(&vm, Some(8)), Exit::Hlt));
            assert_eq!(cpu.delivery_log().unwrap().total(), 1);
            cpu.set_numeric_error_route(Route::Internal).unwrap();
        }
    }
}

#[test]
fn internal_service_does_not_reconcile_external_bookkeeping() {
    for model in [CpuModel::I8086, CpuModel::I80286, CpuModel::I80386] {
        for checked in [false, true] {
            let code: &[u8] = if model == CpuModel::I8086 {
                &[0x90, 0x90]
            } else {
                &[0x90, 0xdb, 0xe2] // FNCLEX also keeps reconciliation off Internal.
            };
            let (vm, mut cpu) = setup_model(model, false, true, code);
            if checked {
                cpu.cpu.legacy.segments[SegmentId::Fs as usize].attributes = 0x92;
            }
            assert_eq!(cpu.numeric_error_route(), Route::Internal);
            cpu.cpu.external_numeric_error.reported = true;
            cpu.cpu.external_numeric_error.held = true;
            let bookkeeping = cpu.cpu.external_numeric_error;
            // A blocked IRQ forces existing service work at both boundaries.
            cpu.inject_irq(0x70);
            fixture::steps(&mut cpu, &vm, 2);
            assert!(cpu.has_pending_irq());
            assert_eq!(cpu.cpu.external_numeric_error, bookkeeping);
        }
    }
}

#[test]
fn external_control_and_environment_producers_commit_before_notification() {
    for (model, wide, real) in PENDING_CONTEXTS {
        for full in [false, true] {
            let code = mem(wide, if full { 0xdd } else { 0xd9 }, 4, false, false);
            let (vm, mut cpu) = setup_model(model, wide, real, &code);
            cpu.set_numeric_error_route(Route::External).unwrap();
            let mut image = vec![0; if wide { 28 } else { 14 } + if full { 80 } else { 0 }];
            image[..2].copy_from_slice(&0x037bu16.to_le_bytes());
            let sw = if wide { 4 } else { 2 };
            image[sw..sw + 2].copy_from_slice(&4u16.to_le_bytes());
            vm.write_bytes(DATA + 0x400, &image).unwrap();
            handoff(
                &mut cpu,
                &vm,
                1,
                1,
                0x1000 + code.len() as u64,
                Event::Asserted,
            );
            assert_eq!(cpu.cpu.fpu_sw & 0x84, 0x84);
        }
        let mut code = vec![0xd8, 0xf1]; // masked divide then FLDCW unmasks ZE
        code.extend(mem(wide, 0xd9, 5, false, false));
        let (vm, mut cpu) = setup_model(model, wide, real, &code);
        cpu.set_numeric_error_route(Route::External).unwrap();
        seed_divide(&mut cpu, false);
        vm.write_bytes(DATA + 0x400, &0x037bu16.to_le_bytes())
            .unwrap();
        fixture::steps(&mut cpu, &vm, 1);
        assert!(!cpu.numeric_error_pending());
        handoff(
            &mut cpu,
            &vm,
            1,
            1,
            0x1000 + code.len() as u64,
            Event::Asserted,
        );
    }
}

#[test]
fn external_clear_does_not_acknowledge_and_rearms_within_one_run() {
    for (model, wide, real) in PENDING_CONTEXTS {
        for control in [
            vec![0xdb, 0xe2],
            vec![0xdb, 0xe3],
            mem(wide, 0xd9, 6, false, false),
            mem(wide, 0xdd, 6, false, false),
        ] {
            let mut code = vec![0xd8, 0xf1];
            code.extend(&control);
            code.push(0x9b);
            let (vm, mut cpu) = setup_model(model, wide, real, &code);
            cpu.set_numeric_error_route(Route::External).unwrap();
            seed_divide(&mut cpu, true);
            handoff(&mut cpu, &vm, 10, 1, 0x1002, Event::Asserted);
            handoff(
                &mut cpu,
                &vm,
                10,
                1,
                0x1000 + code.len() as u64 - 1,
                Event::Waiting,
            );
            assert!(!cpu.numeric_error_pending());
            assert!(!cpu.cpu.external_numeric_error.reported);
            assert!(cpu.cpu.external_numeric_error.held);
            cpu.ack_external_numeric_error().unwrap();
            fixture::steps(&mut cpu, &vm, 1);
        }
        // FNCLEX deasserts then FLDENV installs a fresh pending episode,
        // both within one positive run.
        let load = mem(wide, 0xd9, 4, false, false);
        let mut code = vec![0xdb, 0xe2];
        code.extend(&load);
        let (vm, mut cpu) = setup_model(model, wide, real, &code);
        cpu.set_numeric_error_route(Route::External).unwrap();
        seed_divide(&mut cpu, true);
        cpu.cpu.fpu_sw = 0x8084;
        handoff(&mut cpu, &vm, 1, 0, 0x1000, Event::Asserted);
        cpu.ack_external_numeric_error().unwrap();
        let mut image = vec![0; if wide { 28 } else { 14 }];
        image[..2].copy_from_slice(&0x037bu16.to_le_bytes());
        let sw = if wide { 4 } else { 2 };
        image[sw..sw + 2].copy_from_slice(&4u16.to_le_bytes());
        vm.write_bytes(DATA + 0x400, &image).unwrap();
        handoff(
            &mut cpu,
            &vm,
            2,
            2,
            0x1000 + code.len() as u64,
            Event::Asserted,
        );
    }
}

#[test]
fn external_failed_numeric_store_does_not_publish_or_commit_status() {
    for model in fixture::MODELS {
        for checked in [false, true] {
            // FIST word [0400], a masked invalid conversion whose store faults.
            let (mut vm, mut cpu) = external(model, checked, &[0xdf, 0x16, 0, 4]);
            cpu.cpu.fpu_cw = 0x037f;
            cpu.cpu.fpu_empty = 0xff;
            vm.unmap(0, 0x240000).unwrap();
            vm.map(
                0,
                DATA as usize,
                x86jit_core::Prot::RWX,
                x86jit_core::RegionKind::Ram,
            )
            .unwrap();
            let before = fpu(&cpu);
            let r = cpu.run_steps(&vm, 2, StepOptions::default()).unwrap();
            assert_eq!(r.steps(), 0);
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::UnmappedMemory { .. })),
                "{:?}",
                r.stop()
            );
            assert_eq!(fpu(&cpu), before);
            assert!(!cpu.cpu.external_numeric_error.reported);
            assert!(!cpu.cpu.external_numeric_error.held);
        }
    }
}

#[test]
fn external_runtime_selection_is_per_vcpu_and_preserves_cached_ir() {
    for model in fixture::MODELS {
        let (vm, mut a) = external(model, false, &[0xd8, 0xf1, 0x9b]);
        let mut b = vm.new_vcpu();
        b.cpu = a.cpu.clone();
        b.set_numeric_error_route(Route::Internal).unwrap();
        fixture::steps(&mut b, &vm, 1);
        fixture::fault(&mut b, &vm, 16, None);
        handoff(&mut a, &vm, 1, 1, 0x1002, Event::Asserted);
        b.set_numeric_error_route(Route::External).unwrap();
        handoff(&mut b, &vm, 1, 0, 0x1002, Event::Asserted);
        handoff(&mut a, &vm, 1, 0, 0x1002, Event::Waiting);
        a.set_numeric_error_route(Route::Internal).unwrap();
        fixture::fault(&mut a, &vm, 16, None);
        a.set_numeric_error_route(Route::External).unwrap();
        handoff(&mut a, &vm, 1, 0, 0x1002, Event::Waiting);
    }
}

#[test]
fn external_configuration_rejects_unsupported_and_owned_continuations() {
    let mut vm = Vm::new(x86jit_core::VmConfig::flat(0x10000));
    assert_eq!(
        vm.set_numeric_error_route(Route::External),
        Err(ConfigError::UnsupportedProfile)
    );
    let mut application = vm.new_vcpu();
    assert_eq!(
        application.set_numeric_error_route(Route::External),
        Err(ConfigError::UnsupportedProfile)
    );
    // A fresh VM remains configurable after the Application refusal.
    let mut vm = Vm::new(x86jit_core::VmConfig::flat(0x10000));
    vm.set_cpu_model(CpuModel::I8086);
    assert_eq!(
        vm.set_numeric_error_route(Route::External),
        Err(ConfigError::UnsupportedModel)
    );
    vm.set_cpu_model(CpuModel::I80286);
    vm.set_fpu_model(FpuModel::None);
    assert_eq!(
        vm.set_numeric_error_route(Route::External),
        Err(ConfigError::CoprocessorAbsent)
    );
    vm.set_fpu_model(FpuModel::Present);
    vm.set_numeric_error_route(Route::External).unwrap();
    let mut cpu = vm.new_vcpu();
    assert_eq!(cpu.numeric_error_route(), Route::External);
    assert_eq!(
        vm.set_numeric_error_route(Route::Internal),
        Err(ConfigError::ConfigurationLocked)
    );
    assert_eq!(cpu.set_ignne(true), Err(ConfigError::UnsupportedModel));
    cpu.cpu.pending_port_in = Some(1);
    let before = format!("{:?}", cpu.cpu);
    assert_eq!(
        cpu.set_numeric_error_route(Route::Internal),
        Err(ConfigError::ContinuationInFlight)
    );
    assert_eq!(
        cpu.ack_external_numeric_error(),
        Err(ConfigError::ContinuationInFlight)
    );
    assert_eq!(format!("{:?}", cpu.cpu), before);
    cpu.cpu.pending_port_in = None;
    cpu.cpu.task_debug_trap = 1;
    let before = format!("{:?}", cpu.cpu);
    assert_eq!(
        cpu.set_numeric_error_route(Route::Internal),
        Err(ConfigError::ContinuationInFlight)
    );
    assert_eq!(format!("{:?}", cpu.cpu), before);
    cpu.cpu.task_debug_trap = 0;
    cpu.set_numeric_error_route(Route::Internal).unwrap();
    assert_eq!(
        cpu.ack_external_numeric_error(),
        Err(ConfigError::ExternalRouteRequired)
    );
}

// A deliberately small AT-class board fixture. This is embedder policy, not
// evidence for a particular board's exact latch/re-arm circuit.
#[derive(Clone)]
struct NumericBoard {
    latched: bool,
    masked: bool,
    vector: u8,
}
impl NumericBoard {
    fn queue(&self, cpu: &mut Vcpu) {
        if self.latched && !self.masked && !cpu.has_pending_irq() {
            cpu.inject_irq(self.vector);
        }
    }
    fn acknowledge(&mut self, cpu: &mut Vcpu) {
        self.latched = false;
        cpu.ack_external_numeric_error().unwrap();
    }
}

fn real_handler(vm: &Vm, vector: u8, bytes: &[u8]) {
    let ip = 0x5000 + u16::from(vector) * 0x20;
    let mut entry = [0; 4];
    entry[..2].copy_from_slice(&ip.to_le_bytes());
    vm.write_bytes(u64::from(vector) * 4, &entry).unwrap();
    vm.write_bytes(u64::from(ip), bytes).unwrap();
}

#[test]
fn external_board_irq_handler_port_ack_and_delivery_evidence() {
    for model in fixture::MODELS {
        for checked in [false, true] {
            for clear_first in [false, true] {
                let delivery = if checked {
                    ExceptionDelivery::InGuest
                } else {
                    ExceptionDelivery::Outward
                };
                let (vm, mut cpu) =
                    setup_model_delivery(model, false, true, &[0xd8, 0xf1, 0x9b, 0x90], delivery);
                if checked {
                    cpu.cpu.legacy.segments[SegmentId::Fs as usize].attributes = 0x92;
                }
                cpu.set_numeric_error_route(Route::External).unwrap();
                seed_divide(&mut cpu, true);
                cpu.enable_delivery_log(32).unwrap();
                let mut handler = vec![0xdf, 0xe0]; // no-wait inspection
                if clear_first {
                    handler.extend([0xdb, 0xe2]);
                }
                handler.extend([0xe6, 0xf0]); // OUT F0h,AL
                if !clear_first {
                    handler.extend([0x9b, 0xdb, 0xe2]);
                }
                handler.push(0xcf);
                real_handler(&vm, 0x75, &handler);
                real_handler(&vm, 0x60, &[0xcf]);
                let mut board = NumericBoard {
                    latched: false,
                    masked: true,
                    vector: 0x75,
                };
                handoff(&mut cpu, &vm, 10, 1, 0x1002, Event::Asserted);
                board.latched = true;
                board.queue(&mut cpu);
                assert!(!cpu.has_pending_irq());
                board.masked = false;
                cpu.inject_irq(0x60);
                board.queue(&mut cpu); // never overwrite an unrelated vector
                cpu.cpu.flags.if_ = true;
                let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(r.steps(), 1); // unrelated IRET
                assert_eq!(cpu.cpu.rip, 0x1002);
                board.queue(&mut cpu);
                assert!(cpu.has_pending_irq());
                assert!(
                    cpu.irq_ready(),
                    "{model:?} checked={checked} flags={:?}",
                    cpu.cpu.flags
                );
                let saved_cpu = cpu.cpu.clone();
                let saved_board = board.clone();
                let r = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
                assert!(matches!(r.stop(), StepStop::Exit(Exit::PortIo {
                    port: 0xf0, dir: x86jit_core::PortDir::Out, ..
                })), "{model:?} checked={checked} clear_first={clear_first} {:?} pending={} ready={} deliveries={:?}", r.stop(), cpu.has_pending_irq(), cpu.irq_ready(), cpu.delivery_log_mut().unwrap().take());
                assert_eq!(r.steps(), if clear_first { 3 } else { 2 });
                assert_eq!(cpu.cpu.gpr[0] & 0x84, 0x84);
                let before = fpu(&cpu);
                board.acknowledge(&mut cpu);
                assert_eq!(fpu(&cpu), before);
                assert_eq!(cpu.numeric_error_pending(), !clear_first);
                fixture::steps(&mut cpu, &vm, if clear_first { 2 } else { 4 });
                assert_eq!(cpu.cpu.rip, 0x1003);
                let batch = cpu.delivery_log_mut().unwrap().take();
                assert_eq!(batch.records.len(), 2);
                assert_eq!(
                    (batch.records[0].vector, batch.records[0].kind),
                    (0x60, DeliveryKind::Irq)
                );
                assert_eq!(
                    (batch.records[1].vector, batch.records[1].kind),
                    (0x75, DeliveryKind::Irq)
                );
                // CPU restoration does not rewind the log or injected events.
                cpu.cpu = saved_cpu;
                board = saved_board;
                board.queue(&mut cpu);
                let r = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
                assert!(matches!(
                    r.stop(),
                    StepStop::Exit(Exit::PortIo { port: 0xf0, .. })
                ));
                assert_eq!(cpu.delivery_log().unwrap().total(), 3);
            }
        }
    }
}

#[test]
fn external_wait_keeps_host_nmi_irq_and_shadow_priorities() {
    for model in fixture::MODELS {
        for checked in [false, true] {
            for prefix in [&[0xfb][..], &[0x8e, 0xd0][..]] {
                // STI or MOV SS,AX
                let mut code = prefix.to_vec();
                code.push(0x9b);
                let (vm, mut cpu) = external(model, checked, &code);
                cpu.cpu.gpr[0] = cpu.cpu.ss as u64;
                cpu.cpu.fpu_sw = 0x8084;
                handoff(&mut cpu, &vm, 1, 0, 0x1000, Event::Asserted);
                fixture::steps(&mut cpu, &vm, 1); // arms the existing shadow
                real_handler(&vm, 0x75, &[0xdb, 0xe2, 0xe6, 0xf0, 0xcf]);
                cpu.cpu.flags.if_ = true;
                cpu.inject_irq(0x75);
                let wait_ip = 0x1000 + prefix.len() as u64;
                handoff(&mut cpu, &vm, 1, 0, wait_ip, Event::Waiting);
                assert!(cpu.has_pending_irq());
                let request = cpu.exit_request();
                request.request_exit();
                let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert!(matches!(r.stop(), StepStop::Exit(Exit::Interrupted)));
                assert_eq!(r.steps(), 0);
                // Clear execution shadows explicitly; reset preserves the hold.
                cpu.reset_execution_state(ResetOptions::default());
                cpu.cpu.flags.if_ = false;
                real_handler(&vm, 2, &[0xdf, 0xe0, 0xe6, 0xf0, 0xcf]);
                cpu.raise_nmi().unwrap();
                let r = cpu.run_steps(&vm, 4, StepOptions::default()).unwrap();
                assert!(
                    matches!(r.stop(), StepStop::Exit(Exit::PortIo { port: 0xf0, .. })),
                    "{:?}",
                    r.stop()
                );
                assert_eq!(r.steps(), 2);
                assert!(cpu.numeric_error_pending());
                assert_eq!(cpu.delivery_log().unwrap().total(), 1);
                let record = &cpu.delivery_log_mut().unwrap().take().records[0];
                assert_eq!((record.vector, record.kind), (2, DeliveryKind::Nmi));
            }
        }
    }
}

#[test]
fn internal_numeric_error_matrix_and_protected_286_refusals_remain() {
    for model in fixture::MODELS {
        for delivery in [ExceptionDelivery::Outward, ExceptionDelivery::InGuest] {
            for next in [&[0x9b][..], &[0xd9, 0xe8][..]] {
                let mut code = vec![0xd8, 0xf1];
                code.extend(next);
                let mut vm = Vm::new(x86jit_core::VmConfig::flat(0x240000));
                vm.map(
                    0,
                    0x240000,
                    x86jit_core::Prot::RWX,
                    x86jit_core::RegionKind::Ram,
                )
                .unwrap();
                vm.set_cpu_model(model);
                vm.set_exception_delivery(delivery);
                vm.write_bytes(0x1000, &code).unwrap();
                let mut cpu = vm.new_vcpu();
                cpu.cpu =
                    x86jit_core::CpuState::legacy_real_entry([0, 0x2000, 0x2000, 0x2000, 0, 0]);
                cpu.cpu.rip = 0x1000;
                cpu.cpu.gpr[4] = 0x8000;
                seed_divide(&mut cpu, true);
                cpu.enable_delivery_log(8).unwrap();
                assert_eq!(cpu.numeric_error_route(), Route::Internal);
                let dst = cpu.cpu.fpr[5];
                fixture::steps(&mut cpu, &vm, 1);
                assert_eq!(cpu.cpu.fpr[5], dst);
                assert_eq!(cpu.cpu.fpu_sw & 0x84, 0x84);
                if delivery == ExceptionDelivery::Outward {
                    fixture::fault(&mut cpu, &vm, 16, None);
                    assert_eq!(cpu.delivery_log().unwrap().total(), 0);
                } else {
                    real_handler(&vm, 16, &[0xe6, 0x80]);
                    let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                    assert_eq!(r.steps(), 1);
                    assert!(matches!(
                        r.stop(),
                        StepStop::Exit(Exit::PortIo { port: 0x80, .. })
                    ));
                    let batch = cpu.delivery_log_mut().unwrap().take();
                    assert_eq!(batch.records.len(), 1);
                    assert_eq!(
                        (batch.records[0].vector, batch.records[0].kind),
                        (16, DeliveryKind::Fault)
                    );
                    assert_eq!(fixture::value(&vm, 0x28000 - 6, 2), 0x1002);
                }
            }
        }
    }
    for route in [Route::Internal, Route::External] {
        for code in [&[0x9b][..], &[0xdf, 0xe0], &[0xdb, 0xe3]] {
            let (vm, mut cpu) = setup_model(CpuModel::I80286, false, false, code);
            cpu.set_numeric_error_route(route).unwrap();
            cpu.cpu.fpu_sw = 0x8084;
            if route == Route::External {
                handoff(&mut cpu, &vm, 1, 0, 0x1000, Event::Asserted);
            }
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(r.steps(), 0);
            assert!(
                matches!(
                    r.stop(),
                    StepStop::Exit(Exit::UnsupportedSystemContext {
                        operation: x86jit_core::UnsupportedSystemOperation::LegacyInstruction,
                        ..
                    })
                ),
                "{:?}",
                r.stop()
            );
            assert_eq!(cpu.cpu.rip, 0x1000);
        }
    }
}

#[test]
fn early_cpu_pending_wait_refuses_and_empty_socket_stays_inactive() {
    // H1-H3 do not extend the accepted 8087 scope; retain its explicit refusal.
    for model in [CpuModel::I8086, CpuModel::I8088] {
        for fpu in [FpuModel::Present, FpuModel::None] {
            let (vm, mut cpu) = setup_model(model, false, true, &[0x9b]);
            cpu.set_fpu_model(fpu);
            cpu.cpu.fpu_sw = 0x8084;
            cpu.enable_delivery_log(4).unwrap();
            let before = fpu_image(&cpu);
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            if fpu == FpuModel::Present {
                assert!(matches!(
                    r.stop(),
                    StepStop::Exit(Exit::UnknownInstruction { addr: 0x1000, .. })
                ));
                assert_eq!((r.steps(), cpu.cpu.rip), (0, 0x1000));
            } else {
                assert!(matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)));
                assert_eq!((r.steps(), cpu.cpu.rip), (1, 0x1001));
            }
            assert_eq!(fpu_image(&cpu), before);
            assert_eq!(cpu.delivery_log().unwrap().total(), 0);
            assert_eq!(
                cpu.set_numeric_error_route(Route::External),
                Err(ConfigError::UnsupportedModel)
            );
        }
    }
}

fn fpu_image(cpu: &Vcpu) -> (u16, u16, u32, u8, [[u8; 10]; 8]) {
    (
        cpu.cpu.fpu_cw,
        cpu.cpu.fpu_sw,
        cpu.cpu.fpu_top,
        cpu.cpu.fpu_empty,
        cpu.cpu.fpr,
    )
}

#[test]
fn external_explicit_wait_prefix_and_no_wait_inspection() {
    for (model, wide, real) in PENDING_CONTEXTS {
        for control in [vec![0xdf, 0xe0], mem(wide, 0xd9, 7, false, false)] {
            for prefix in [false, true] {
                let mut code = Vec::new();
                if prefix {
                    code.push(0x9b);
                }
                code.extend(&control);
                code.push(0x9b);
                let (vm, mut cpu) = setup_model(model, wide, real, &code);
                cpu.set_numeric_error_route(Route::External).unwrap();
                seed_divide(&mut cpu, true);
                cpu.cpu.fpu_sw = 0x8084;
                handoff(&mut cpu, &vm, 1, 0, 0x1000, Event::Asserted);
                let before = fpu(&cpu);
                handoff(
                    &mut cpu,
                    &vm,
                    4,
                    u64::from(!prefix),
                    if prefix {
                        0x1000
                    } else {
                        0x1000 + control.len() as u64
                    },
                    Event::Waiting,
                );
                assert_eq!(fpu(&cpu), before);
                if !prefix && control[0] == 0xdf {
                    assert_eq!(cpu.cpu.gpr[0] & 0x84, 0x84);
                }
                if !prefix && control[0] == 0xd9 {
                    assert_eq!(fixture::value(&vm, DATA + 0x400, 2), 0x037b);
                }
            }
        }
    }
}

#[test]
fn external_failed_environment_transfer_defers_notification_until_retry() {
    for model in fixture::MODELS {
        for checked in [false, true] {
            let (mut vm, mut cpu) = external(model, checked, &[0xd9, 0x26, 0, 4, 0x9b]);
            vm.unmap(0, 0x240000).unwrap();
            vm.map(
                0,
                (DATA + 0x408) as usize,
                x86jit_core::Prot::RWX,
                x86jit_core::RegionKind::Ram,
            )
            .unwrap();
            // The pending environment starts in RAM but its final bytes are
            // unmapped. Numeric transfers currently require RAM on this path.
            let mut env = [0; 14];
            env[..2].copy_from_slice(&0x037bu16.to_le_bytes());
            env[2..4].copy_from_slice(&4u16.to_le_bytes());
            vm.write_bytes(DATA + 0x400, &env[..8]).unwrap();
            let before = fpu(&cpu);
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(r.steps(), 0);
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::UnmappedMemory { .. })),
                "{:?}",
                r.stop()
            );
            assert_eq!(fpu(&cpu), before);
            assert!(!cpu.numeric_error_pending());
            assert!(!cpu.cpu.external_numeric_error.reported);
            assert!(!cpu.cpu.external_numeric_error.held);
            vm.map(
                DATA + 0x408,
                6,
                x86jit_core::Prot::RW,
                x86jit_core::RegionKind::Ram,
            )
            .unwrap();
            vm.write_bytes(DATA + 0x408, &env[8..]).unwrap();
            handoff(&mut cpu, &vm, 1, 1, 0x1004, Event::Asserted);
            assert!(cpu.numeric_error_pending());
            handoff(&mut cpu, &vm, 1, 0, 0x1004, Event::Waiting);
            cpu.ack_external_numeric_error().unwrap();
            fixture::steps(&mut cpu, &vm, 1);
        }
    }
}

#[test]
fn external_configuration_preserves_real_memory_and_port_continuations() {
    for model in fixture::MODELS {
        for checked in [false, true] {
            for port in [false, true] {
                let code: &[u8] = if port { &[0xe4, 0x80] } else { &[0xa1, 0, 4] };
                let (mut vm, mut cpu) = external(model, checked, code);
                if !port {
                    vm.unmap(0, 0x240000).unwrap();
                    vm.map(
                        0,
                        DATA as usize,
                        x86jit_core::Prot::RWX,
                        x86jit_core::RegionKind::Ram,
                    )
                    .unwrap();
                    vm.map(
                        DATA,
                        0x1000,
                        x86jit_core::Prot::RW,
                        x86jit_core::RegionKind::Trap,
                    )
                    .unwrap();
                }
                let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(r.steps(), u64::from(port));
                assert!(
                    matches!(
                        r.stop(),
                        StepStop::Exit(Exit::MmioRead { .. } | Exit::PortIo { .. })
                    ),
                    "{:?}",
                    r.stop()
                );
                assert_eq!(
                    cpu.set_numeric_error_route(Route::Internal),
                    Err(ConfigError::ContinuationInFlight)
                );
                assert_eq!(
                    cpu.ack_external_numeric_error(),
                    Err(ConfigError::ContinuationInFlight)
                );
                if port {
                    cpu.complete_port_in(0x42);
                } else {
                    cpu.complete_mmio_read(0x42);
                }
                if !port {
                    assert_eq!(
                        cpu.set_numeric_error_route(Route::Internal),
                        Err(ConfigError::ContinuationInFlight)
                    );
                    fixture::steps(&mut cpu, &vm, 1);
                }
                assert_eq!(cpu.cpu.gpr[0] & 0xff, 0x42);
                cpu.set_numeric_error_route(Route::Internal).unwrap();
            }
        }
    }
}
