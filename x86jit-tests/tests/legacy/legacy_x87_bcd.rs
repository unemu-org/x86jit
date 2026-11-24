//! SDM Vol 1 §§8.2–8.2.1 p. 8-13, §4.7 Table 4-5 p. 4-12;
//! Vol 2A §3.3 FBLD/FBSTP pp. 3-314–317. Second witnesses:
//! 80286/287 PRM Part II Ch. 2, Data Transfer Instructions, pp. 2-3–4, Table 1-14 p. 1-30;
//! Intel486 PRM §15.2.2.2 p. 15-13, FBLD/FBSTP pp. 26-77–79.
#[path = "../support/descriptor_fixture.rs"]
#[allow(dead_code)]
mod fixture;

use super::legacy_i386_x87::{mem, setup_model};
use x86jit_core::{
    CpuGeneration, CpuModel, Exit, Prot, RegionKind, SegmentId, StepOptions, StepStop, Vcpu, Vm,
};

const ADDRESS: u64 = 0x20400;
const MAX: i64 = 999_999_999_999_999_999;
const INDEFINITE: [u8; 10] = [0, 0, 0, 0, 0, 0, 0, 0xc0, 0xff, 0xff];

fn contexts() -> Vec<(CpuModel, bool, bool)> {
    let mut result = vec![];
    for model in [
        CpuModel::I80286,
        CpuModel::I80386,
        CpuModel::I80486Dx,
        CpuModel::I80486DxCpuid,
    ] {
        result.push((model, false, true));
        if model != CpuModel::I80286 {
            result.extend([(model, false, false), (model, true, false)]);
        }
    }
    result
}

fn packed(value: i64, negative_zero: bool) -> [u8; 10] {
    let mut result = [0; 10];
    let mut magnitude = value.unsigned_abs();
    for byte in &mut result[..9] {
        *byte = (magnitude % 10) as u8;
        magnitude /= 10;
        *byte |= ((magnitude % 10) as u8) << 4;
        magnitude /= 10;
    }
    result[9] = if value < 0 || negative_zero { 0x80 } else { 0 };
    result
}

// DERIVED: normalize the exact integer directly, independently of engine F80.
fn extended(value: i64, negative_zero: bool) -> [u8; 10] {
    let mut result = [0; 10];
    let magnitude = value.unsigned_abs();
    let mut exponent = 0;
    if magnitude != 0 {
        let shift = magnitude.leading_zeros();
        result[..8].copy_from_slice(&(magnitude << shift).to_le_bytes());
        exponent = 16383 + 63 - shift as u16;
    }
    if value < 0 || negative_zero {
        exponent |= 0x8000;
    }
    result[8..].copy_from_slice(&exponent.to_le_bytes());
    result
}

fn read(vm: &Vm) -> [u8; 10] {
    let mut result = [0; 10];
    vm.read_bytes(ADDRESS, &mut result).unwrap();
    result
}

fn seed(cpu: &mut Vcpu, value: [u8; 10]) {
    cpu.cpu.fpu_top = 0;
    cpu.cpu.fpu_empty = 0xfe;
    cpu.cpu.fpu_sw = 0x4700;
    cpu.cpu.fpu_cw = 0x037f;
    cpu.cpu.fpr[0] = value;
}

#[test]
fn early_models_keep_bcd_and_later_register_forms_refused() {
    // Engine availability policy: the October 1979 8086 UM §1.2 p. 1-4
    // describes future coprocessor hooks and has no 8087 instruction chapter.
    for model in [
        CpuModel::I8086,
        CpuModel::I8088,
        CpuModel::NecV20,
        CpuModel::NecV30,
    ] {
        for code in [
            mem(false, 0xdf, 4, false, false),
            mem(false, 0xdf, 6, false, false),
            vec![0xdf, 0xc0], // FFREEP
            vec![0xd9, 0xd8], // FSTPNCE
        ] {
            // setup_model explicitly selects FpuModel::Present.
            let (vm, mut cpu) = setup_model(model, false, true, &code);
            seed(&mut cpu, extended(7, false));
            vm.write_bytes(ADDRESS, &[0x55; 10]).unwrap();
            let before = cpu.cpu.clone();
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 0);
            assert!(
                matches!(
                    run.stop(),
                    StepStop::Exit(Exit::UnknownInstruction { addr: 0x1000, .. })
                ),
                "{model:?}/{code:02x?}: {:?}",
                run.stop()
            );
            assert_eq!(cpu.cpu.rip, before.rip);
            assert_eq!(cpu.cpu.fpr, before.fpr);
            assert_eq!(
                (cpu.cpu.fpu_top, cpu.cpu.fpu_empty, cpu.cpu.fpu_sw),
                (before.fpu_top, before.fpu_empty, before.fpu_sw)
            );
            assert_eq!(read(&vm), [0x55; 10]);
        }
    }
}

#[test]
fn fbld_exact_signed_values_and_pointers_on_every_machine() {
    for (model, wide, real) in contexts() {
        for (value, negative_zero) in [
            (0, false),
            (0, true),
            (MAX, false),
            (-MAX, false),
            (123_456_789_012_345_678, false),
        ] {
            let code = mem(wide, 0xdf, 4, false, false);
            let (vm, mut cpu) = setup_model(model, wide, real, &code);
            vm.write_bytes(ADDRESS, &packed(value, negative_zero))
                .unwrap();
            cpu.cpu.fpu_sw = 0x4700;
            // PC must not reduce an exact packed-decimal load.
            cpu.cpu.fpu_cw = 0x007f;
            fixture::steps(&mut cpu, &vm, 1);
            assert_eq!(
                cpu.cpu.fpr[cpu.cpu.fpu_top as usize],
                extended(value, negative_zero),
                "{model:?}/{wide}/{real}"
            );
            assert_eq!(cpu.cpu.fpu_top, 7);
            assert_eq!(cpu.cpu.fpu_sw & 0x027f, 0);
            if model.profile().isa.generation() != CpuGeneration::I8086 {
                assert_eq!(cpu.cpu.fpu_fip, 0x1000);
                assert_eq!(cpu.cpu.fpu_fdp, 0x400);
                assert_eq!(cpu.cpu.fpu_fds, cpu.cpu.ds);
                if model.profile().isa.generation() == CpuGeneration::I386 {
                    assert_eq!(cpu.cpu.fpu_fop, 0x700 | u16::from(code[1]));
                }
            }
        }
        // ASSUMED undefined-digit policy: A/F contribute 10/15, and bits
        // 78:72 are ignored (SDM Vol 1 §8.2 Fig. 8-13 p. 8-13).
        let (vm, mut cpu) = setup_model(model, wide, real, &mem(wide, 0xdf, 4, false, false));
        let mut invalid = [0; 10];
        invalid[0] = 0xfa;
        invalid[9] = 0x7f;
        vm.write_bytes(ADDRESS, &invalid).unwrap();
        fixture::steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.fpr[cpu.cpu.fpu_top as usize], extended(160, false));
        assert_eq!(cpu.cpu.fpu_sw & 0x7f, 0);
    }
}

#[test]
fn fbstp_rounding_precision_and_signed_zero_on_every_machine() {
    // SDM Vol 2A FBSTP Table 3-21 p. 3-316: retain sign when rounding to zero.
    // ASSUMED for 287: use the requested modern RC rule; the 287 PRM
    // Part II Ch. 2, Data Transfer Instructions, p. 2-4 instead describes adding .5 then chopping.
    for (model, wide, real) in contexts() {
        let (vm, mut cpu) = setup_model(model, wide, real, &mem(wide, 0xdf, 6, false, false));
        for (value, expected) in [
            (1.5f64, [2, 1, 2, 1]),
            (-1.5, [-2, -2, -1, -1]),
            (2.5, [2, 2, 3, 2]),
            (-0.25, [0, -1, 0, 0]),
            (0.25, [0, 0, 1, 0]),
        ] {
            for (rc, integer) in expected.into_iter().enumerate() {
                let raw = x86jit_core::f80::F80::from_f64(value.to_bits()).to_bytes();
                seed(&mut cpu, raw);
                cpu.cpu.fpu_cw |= (rc as u16) << 10;
                cpu.cpu.rip = 0x1000;
                fixture::steps(&mut cpu, &vm, 1);
                assert_eq!(
                    read(&vm),
                    packed(integer, value.is_sign_negative() && integer == 0),
                    "{model:?}/{rc}/{value}"
                );
                assert_eq!(cpu.cpu.fpu_sw & 0x3f, 0x20);
                assert_eq!(
                    cpu.cpu.fpu_sw & 0x200 != 0,
                    (integer.unsigned_abs() as f64) > value.abs()
                );
                assert_eq!(cpu.cpu.fpu_top, 1);
                assert_eq!(cpu.cpu.fpu_empty, 0xff);
            }
        }
        for negative in [false, true] {
            seed(&mut cpu, extended(0, negative));
            cpu.cpu.rip = 0x1000;
            fixture::steps(&mut cpu, &vm, 1);
            assert_eq!(read(&vm), packed(0, negative));
            assert_eq!(cpu.cpu.fpu_sw & 0x23f, 0);
        }
        // SDM Vol 1 §4.9.1.6 p. 4-24: unmasked #P retains the rounded
        // store; FBSTP also completes its pop (Vol 2A p. 3-316).
        seed(
            &mut cpu,
            x86jit_core::f80::F80::from_f64(1.5f64.to_bits()).to_bytes(),
        );
        cpu.cpu.fpu_cw &= !0x20;
        cpu.cpu.rip = 0x1000;
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 1, "{model:?}: {:?}", run.stop());
        assert_eq!(read(&vm), packed(2, false));
        assert_eq!(cpu.cpu.fpu_top, 1);
        assert_eq!(cpu.cpu.fpu_sw & 0x2bf, 0x2a0);
    }
}

#[test]
fn fbstp_extended_denormals_round_without_invalid_or_denormal_exceptions() {
    // Intel486 PRM §16.1.1 p. 16-3, §16.1.6 Table 16-9 p. 16-17:
    // extended denormals have exponent
    // and integer bit zero. §26 FBSTP p. 26-79: RC conversion and pop,
    // exceptions P/I/IS (no D); §16.2.6 p. 16-27: inexact sets PE,
    // C1 distinguishes rounding up the significand from chopping.
    for (model, wide, real) in contexts() {
        let (vm, mut cpu) = setup_model(model, wide, real, &mem(wide, 0xdf, 6, false, false));
        for significand in [1u64, 0x7fff_ffff_ffff_ffff] {
            for negative in [false, true] {
                let mut raw = [0; 10];
                raw[..8].copy_from_slice(&significand.to_le_bytes());
                raw[9] = u8::from(negative) << 7;
                for rc in 0..4 {
                    seed(&mut cpu, raw);
                    // Unmask I and D so an unexpected exception cannot hide
                    // behind the rounded result of a masked conversion.
                    cpu.cpu.fpu_cw = (cpu.cpu.fpu_cw & !3) | (rc << 10);
                    cpu.cpu.rip = 0x1000;
                    vm.write_bytes(ADDRESS, &[0x55; 10]).unwrap();
                    fixture::steps(&mut cpu, &vm, 1);
                    let up = rc == if negative { 1 } else { 2 };
                    let integer = if up {
                        if negative {
                            -1
                        } else {
                            1
                        }
                    } else {
                        0
                    };
                    assert_eq!(
                        read(&vm),
                        packed(integer, negative && integer == 0),
                        "{model:?}/{wide}/{real}/{significand:x}/{negative}/{rc}"
                    );
                    assert_eq!(cpu.cpu.fpu_sw & 0x2ff, 0x20 | if up { 0x200 } else { 0 });
                    assert_eq!(cpu.cpu.fpu_top, 1);
                    assert_eq!(cpu.cpu.fpu_empty, 0xff);
                    assert_eq!(cpu.cpu.fpr[0], raw);
                }
            }
        }
    }
}

#[test]
fn invalid_conversions_store_indefinite_or_suppress_unmasked_store() {
    for (model, wide, real) in contexts() {
        let (vm, mut cpu) = setup_model(model, wide, real, &mem(wide, 0xdf, 6, false, false));
        let mut special = [0; 10];
        special[7] = 0x80;
        special[8..].copy_from_slice(&0x7fffu16.to_le_bytes());
        let mut nan = special;
        nan[7] = 0xc0;
        let mut snan = special;
        snan[0] = 1;
        for value in [
            extended(MAX + 1, false),
            extended(-MAX - 1, false),
            extended(i64::MIN, false),
            special,
            nan,
            snan,
            // Intel486 PRM §16.1.7 Table 16-10 p. 16-18: unsupported
            // unnormal, with nonzero exponent and integer bit clear.
            [1; 10],
        ] {
            for masked in [true, false] {
                seed(&mut cpu, value);
                cpu.cpu.rip = 0x1000;
                if !masked {
                    cpu.cpu.fpu_cw &= !1;
                }
                vm.write_bytes(ADDRESS, &[0x55; 10]).unwrap();
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 1, "{model:?}: {:?}", run.stop());
                assert_eq!(cpu.cpu.fpu_sw & 0x23f, 1);
                assert_eq!(read(&vm), if masked { INDEFINITE } else { [0x55; 10] });
                assert_eq!(cpu.cpu.fpu_top, u32::from(masked));
                assert_eq!(cpu.cpu.fpu_empty, if masked { 0xff } else { 0xfe });
                assert_eq!(cpu.cpu.fpr[0], value);
            }
        }
        // DERIVED: MAX has exponent 59 and an extended ULP of 1/16;
        // adding eight significand units constructs MAX + 1/2 exactly.
        for negative in [false, true] {
            let mut raw = extended(MAX, negative);
            let significand = u64::from_le_bytes(raw[..8].try_into().unwrap()) + 8;
            raw[..8].copy_from_slice(&significand.to_le_bytes());
            for rc in 0..4 {
                seed(&mut cpu, raw);
                cpu.cpu.fpu_cw |= rc << 10;
                cpu.cpu.rip = 0x1000;
                fixture::steps(&mut cpu, &vm, 1);
                let invalid = rc == 0 || rc == if negative { 1 } else { 2 };
                assert_eq!(
                    read(&vm),
                    if invalid {
                        INDEFINITE
                    } else {
                        packed(if negative { -MAX } else { MAX }, false)
                    }
                );
                assert_eq!(cpu.cpu.fpu_sw & 0x23f, if invalid { 1 } else { 0x20 });
            }
        }
    }
}

#[test]
fn bcd_stack_faults_preserve_unmasked_stack_and_use_masked_indefinites() {
    for (model, wide, real) in contexts() {
        for load in [true, false] {
            for masked in [true, false] {
                let (vm, mut cpu) = setup_model(
                    model,
                    wide,
                    real,
                    &mem(wide, 0xdf, if load { 4 } else { 6 }, false, false),
                );
                seed(&mut cpu, extended(7, false));
                cpu.cpu.fpu_empty = if load { 0 } else { 0xff };
                if !masked {
                    cpu.cpu.fpu_cw &= !1;
                }
                vm.write_bytes(ADDRESS, &[0; 10]).unwrap();
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 1, "{model:?}: {:?}", run.stop());
                assert_eq!(cpu.cpu.fpu_sw & 0x27f, if load { 0x241 } else { 0x41 });
                assert_eq!(
                    cpu.cpu.fpu_top,
                    if masked {
                        if load {
                            7
                        } else {
                            1
                        }
                    } else {
                        0
                    }
                );
                if masked && load {
                    assert_eq!(cpu.cpu.fpr[7], [0, 0, 0, 0, 0, 0, 0, 0xc0, 0xff, 0xff]);
                } else if !load {
                    assert_eq!(read(&vm), if masked { INDEFINITE } else { [0; 10] });
                }
            }
        }
    }
}

#[test]
fn bcd_segment_limit_faults_leave_memory_stack_status_and_pointers_unchanged() {
    for (model, wide, real) in contexts()
        .into_iter()
        .filter(|(m, _, _)| m.profile().isa.generation() != CpuGeneration::I8086)
    {
        for group in [4, 6] {
            let mut code = mem(wide, 0xdf, group, false, false);
            if real {
                code[2..].copy_from_slice(&0xfff8u16.to_le_bytes());
            }
            let (vm, mut cpu) = setup_model(model, wide, real, &code);
            seed(
                &mut cpu,
                x86jit_core::f80::F80::from_f64(1.5f64.to_bits()).to_bytes(),
            );
            cpu.cpu.legacy.segments[SegmentId::Ds as usize].limit =
                if real { 0xffff } else { 0x408 };
            let address = if real { 0x2fff8 } else { ADDRESS };
            vm.write_bytes(address, &[0x55; 10]).unwrap();
            // 287 PRM Part I App. B #MP p. B-9: a split operand raises #9.
            // SDM Vol 2A FBLD/FBSTP pp. 3-314/317: otherwise #GP.
            fixture::fault(
                &mut cpu,
                &vm,
                if model == CpuModel::I80286 { 9 } else { 13 },
                if model != CpuModel::I80286 && !real {
                    Some(0)
                } else {
                    None
                },
            );
            let mut bytes = [0; 10];
            vm.read_bytes(address, &mut bytes).unwrap();
            assert_eq!(bytes, [0x55; 10]);
        }
    }
    // Fault on the last byte of a RAM destination, after conversion would
    // raise #P. All models must retain the first nine bytes and FPU state.
    for (model, wide, real) in contexts() {
        let code = mem(wide, 0xdf, 6, false, false);
        let (mut vm, mut cpu) = setup_model(model, wide, real, &code);
        vm.unmap(0, 0x240000).unwrap();
        vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x11000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(ADDRESS, 9, Prot::RW, RegionKind::Ram).unwrap();
        vm.write_bytes(if real { 0x1000 } else { 0x11000 }, &code)
            .unwrap();
        vm.write_bytes(ADDRESS, &[0x55; 9]).unwrap();
        seed(
            &mut cpu,
            x86jit_core::f80::F80::from_f64(1.5f64.to_bits()).to_bytes(),
        );
        let before = (
            cpu.cpu.fpr,
            cpu.cpu.fpu_sw,
            cpu.cpu.fpu_top,
            cpu.cpu.fpu_empty,
            cpu.cpu.fpu_fip,
            cpu.cpu.fpu_fdp,
            cpu.cpu.fpu_fop,
        );
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::UnmappedMemory { .. })),
            "{model:?}: {:?}",
            run.stop()
        );
        assert_eq!(
            (
                cpu.cpu.fpr,
                cpu.cpu.fpu_sw,
                cpu.cpu.fpu_top,
                cpu.cpu.fpu_empty,
                cpu.cpu.fpu_fip,
                cpu.cpu.fpu_fdp,
                cpu.cpu.fpu_fop
            ),
            before
        );
        let mut bytes = [0; 9];
        vm.read_bytes(ADDRESS, &mut bytes).unwrap();
        assert_eq!(bytes, [0x55; 9]);
    }
}

#[test]
fn packed_integer_roundtrips_many_values_in_every_machine_context() {
    for (model, wide, real) in contexts() {
        let mut code = mem(wide, 0xdf, 4, false, false);
        code.extend(mem(wide, 0xdf, 6, false, false));
        let (vm, mut cpu) = setup_model(model, wide, real, &code);
        let mut state = 0x1234_5678_9abc_def0u64;
        for index in 0..256 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let value = (state % (MAX as u64 + 1)) as i64 * if index % 2 == 0 { 1 } else { -1 };
            let input = packed(value, false);
            vm.write_bytes(ADDRESS, &input).unwrap();
            cpu.cpu.rip = 0x1000;
            cpu.cpu.fpu_empty = 0xff;
            cpu.cpu.fpu_sw = 0;
            fixture::steps(&mut cpu, &vm, 2);
            assert_eq!(read(&vm), input, "{model:?}/{value}");
            assert_eq!(cpu.cpu.fpu_sw & 0x27f, 0);
            assert_eq!(cpu.cpu.fpu_empty, 0xff);
        }
    }
}

#[test]
fn alternate_register_encodings_follow_the_model_manuals() {
    // 287 PRM Part II Appendix A Table A-2 pp. A-2–5 notes 1,4,6–9;
    // SDM Vol 2D §A.5.2 Tables A-10/18/22 pp. A-23/27/29 mark these
    // reserved; Vol 3B §25.18.9 p. 25-15 defines their 32-bit-FPU behavior.
    for (model, wide, real) in contexts()
        .into_iter()
        .filter(|(m, _, _)| m.profile().isa.generation() != CpuGeneration::I8086)
    {
        for index in 0..8 {
            for (op, base) in [
                (0xd9, 0xd8),
                (0xdd, 0xc8),
                (0xdf, 0xc0),
                (0xdf, 0xc8),
                (0xdf, 0xd0),
                (0xdf, 0xd8),
            ] {
                let (vm, mut cpu) = setup_model(model, wide, real, &[op, base + index]);
                seed(&mut cpu, extended(7, false));
                cpu.cpu.fpu_top = 5;
                cpu.cpu.fpu_empty = 0;
                cpu.cpu.fpr[5] = extended(7, false);
                let other = (5 + index as usize) & 7;
                if index != 0 {
                    cpu.cpu.fpr[other] = extended(3, false);
                }
                fixture::steps(&mut cpu, &vm, 1);
                if op == 0xdf && base == 0xc0 {
                    assert_eq!(cpu.cpu.fpu_top, 6);
                    assert_eq!(cpu.cpu.fpu_empty, (1 << 5) | (1 << other));
                } else if base == 0xc8 {
                    assert_eq!(cpu.cpu.fpu_top, 5);
                    assert_eq!(cpu.cpu.fpr[other], extended(7, false));
                    assert_eq!(
                        cpu.cpu.fpr[5],
                        extended(if index == 0 { 7 } else { 3 }, false)
                    );
                } else {
                    assert_eq!(cpu.cpu.fpu_top, 6);
                    assert_eq!(cpu.cpu.fpr[other], extended(7, false));
                    assert_eq!(cpu.cpu.fpu_empty, 1 << 5);
                }
            }
        }
        // D9 D8+i is ordinary FSTP on 287, but suppresses #IS on 387/486.
        let (vm, mut cpu) = setup_model(model, wide, real, &[0xd9, 0xdb]);
        seed(&mut cpu, extended(7, false));
        cpu.cpu.fpu_empty = 0xff;
        cpu.cpu.fpu_cw &= !1;
        fixture::steps(&mut cpu, &vm, 1);
        if model == CpuModel::I80286 {
            assert_eq!(cpu.cpu.fpu_sw & 0x41, 0x41);
            assert_eq!(cpu.cpu.fpu_top, 0);
        } else {
            assert_eq!(cpu.cpu.fpu_sw & 0x41, 0);
            assert_eq!(cpu.cpu.fpu_top, 1);
            assert_eq!(cpu.cpu.fpr[3], extended(7, false));
        }
    }
}
