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
    let (vm, mut cpu) = fixture::setup(model, code);
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
