//! 80386 PRM §17 BT/BTC/BTR/BTS pp. 17-29–36, BCD pp. 17-18–21,
//! 17-54–55, and WAIT p. 17-171. Page numbers are printed manual pages.
#[path = "../support/descriptor_fixture.rs"]
#[allow(dead_code)]
mod descriptor_fixture;

use descriptor_fixture::{fault, steps, value};
use x86jit_core::{CpuModel, FpuModel, SegmentCache, SegmentId, Vcpu, Vm};

fn setup(code: &[u8], wide: bool, checked: bool) -> (Vm, Vcpu) {
    let (vm, mut cpu) = descriptor_fixture::setup(CpuModel::I80386, code);
    // Engine fixture: flat CPL-0 protected segments, with CS.D/SS.B independent
    // of the operand/address overrides exercised below. Machine has no JIT tier;
    // descriptor_fixture's backend sentinel catches accidental codegen entry.
    cpu.cpu.cs = 8;
    cpu.cpu.ds = 16;
    cpu.cpu.es = 16;
    cpu.cpu.ss = 16;
    cpu.cpu.fs = 16;
    cpu.cpu.gs = 16;
    for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
        *cache = SegmentCache {
            base: 0,
            limit: u32::MAX,
            attributes: if i == SegmentId::Cs as usize {
                0x9b
            } else {
                0x93
            } | if wide { 0xc000 } else { 0x8000 },
            unusable: 0,
            reserved: 0,
        };
    }
    if checked {
        cpu.cpu.legacy.segments[SegmentId::Es as usize].limit = 0xffff;
    }
    cpu.cpu.flags.cf = true;
    cpu.cpu.flags.of = true;
    cpu.cpu.flags.sf = true;
    cpu.cpu.flags.zf = true;
    cpu.cpu.flags.set_af(true);
    cpu.cpu.flags.set_pf(true);
    (vm, cpu)
}
fn encoding(wide: bool, operand32: bool, address32: bool, body: &[u8]) -> Vec<u8> {
    let mut code = Vec::new();
    if wide != operand32 {
        code.push(0x66);
    }
    if wide != address32 {
        code.push(0x67);
    }
    code.extend(body);
    code
}
fn unchanged_non_cf(cpu: &Vcpu) {
    assert!(cpu.cpu.flags.of && cpu.cpu.flags.sf && cpu.cpu.flags.zf);
    assert!(cpu.cpu.flags.af() && cpu.cpu.flags.pf());
}

#[test]
fn bt_edx_21_firmware_encoding() {
    // PRM §17 BT pp. 17-29–30; Fig. 17-3 p. 17-14.
    for wide in [false, true] {
        for checked in [false, true] {
            for set in [false, true] {
                let code = encoding(wide, true, wide, &[0x0f, 0xba, 0xe2, 0x15]);
                if wide {
                    assert_eq!(code, [0x0f, 0xba, 0xe2, 0x15]);
                }
                let (vm, mut cpu) = setup(&code, wide, checked);
                let edx = if set { 1 << 21 } else { 1 << 20 };
                cpu.cpu.gpr[2] = edx;
                steps(&mut cpu, &vm, 1);
                assert_eq!(cpu.cpu.flags.cf, set);
                assert_eq!(cpu.cpu.gpr[2], edx);
                assert_eq!(cpu.cpu.rip, 0x1000 + code.len() as u64);
                unchanged_non_cf(&cpu);
            }
        }
    }
}

#[test]
fn bt_family_register_and_immediate_modulo() {
    // PRM §17 BT/BTC/BTR/BTS pp. 17-29–36: immediate offsets select
    // within a word/dword; register destinations also wrap at that width.
    for wide in [false, true] {
        for checked in [false, true] {
            for operand32 in [false, true] {
                let width = if operand32 { 32 } else { 16 };
                for (op, group) in [(0xa3, 4), (0xab, 5), (0xb3, 6), (0xbb, 7)] {
                    for immediate in [false, true] {
                        for index in [width + 5, 255, u32::MAX] {
                            for set in [false, true] {
                                let body = if immediate {
                                    vec![0x0f, 0xba, 0xc2 | group << 3, index as u8]
                                } else {
                                    vec![0x0f, op, 0xca]
                                }; // EDX/DX, ECX/CX
                                let code = encoding(wide, operand32, wide, &body);
                                let (vm, mut cpu) = setup(&code, wide, checked);
                                let bit = index % width;
                                let mask = 1u64 << bit;
                                let initial = if set { mask } else { 0 };
                                cpu.cpu.gpr[2] = 0x1234_5678_0000_0000 | initial;
                                cpu.cpu.gpr[1] = u64::from(index);
                                steps(&mut cpu, &vm, 1);
                                assert_eq!(cpu.cpu.flags.cf, set, "{code:02x?}");
                                let result = match group {
                                    5 => initial | mask,
                                    6 => initial & !mask,
                                    7 => initial ^ mask,
                                    _ => initial,
                                };
                                let upper = if operand32 && group != 4 {
                                    0
                                } else {
                                    0x1234_5678_0000_0000
                                };
                                assert_eq!(cpu.cpu.gpr[2], upper | result, "{code:02x?}");
                                unchanged_non_cf(&cpu);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn bt_family_memory_immediate_and_signed_bit_strings() {
    // PRM §17.2.2.4 BIT pp. 17-14–15: signed DIV rounds down; MOD
    // is positive. §17 BT/BTC/BTR/BTS pp. 17-29–36: immediate modulo.
    for wide in [false, true] {
        for checked in [false, true] {
            for operand32 in [false, true] {
                for address32 in [false, true] {
                    for (op, group) in [(0xa3, 4), (0xab, 5), (0xb3, 6), (0xbb, 7)] {
                        for immediate in [true, false] {
                            for index in [-259i32, -11, -1, 37, 255, 513] {
                                for set in [false, true] {
                                    let mut body = vec![0x0f, if immediate { 0xba } else { op }];
                                    body.push(
                                        if address32 { 5 } else { 6 }
                                            | if immediate { group << 3 } else { 1 << 3 },
                                    );
                                    body.extend(if address32 {
                                        0x6000u32.to_le_bytes().to_vec()
                                    } else {
                                        0x6000u16.to_le_bytes().to_vec()
                                    });
                                    if immediate {
                                        body.push(index as u8);
                                    }
                                    let code = encoding(wide, operand32, address32, &body);
                                    let (vm, mut cpu) = setup(&code, wide, checked);
                                    let offset = if immediate {
                                        i32::from(index as u8) % if operand32 { 32 } else { 16 }
                                    } else {
                                        index
                                    };
                                    let address = (0x6000 + offset.div_euclid(8)) as u64;
                                    let mask = 1u8 << offset.rem_euclid(8);
                                    let initial = if set { mask } else { 0 };
                                    vm.write_bytes(address, &[initial]).unwrap();
                                    cpu.cpu.gpr[1] = u64::from(index as u32);
                                    steps(&mut cpu, &vm, 1);
                                    assert_eq!(cpu.cpu.flags.cf, set, "{code:02x?} index={index}");
                                    let result = match group {
                                        5 => initial | mask,
                                        6 => initial & !mask,
                                        7 => initial ^ mask,
                                        _ => initial,
                                    };
                                    assert_eq!(value(&vm, address, 1), u64::from(result));
                                    unchanged_non_cf(&cpu);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn bcd_results_and_defined_flags() {
    // PRM §17 AAA/AAD/AAM/AAS pp. 17-18–21, DAA/DAS pp. 17-54–55.
    // Only defined flags are asserted. SDM Vol 2A §3.2 AAM p. 3-5
    // also specifies bases other than ten (the PRM lists only D4 0A).
    for wide in [false, true] {
        for checked in [false, true] {
            for p66 in [false, true] {
                for p67 in [false, true] {
                    for (body, ax, af, cf, expected, expected_af, expected_cf, szp) in [
                        (&[0x27][..], 0x009b, false, false, 0x0001, true, true, true),
                        (
                            &[0x27][..],
                            0x0045,
                            false,
                            false,
                            0x0045,
                            false,
                            false,
                            true,
                        ),
                        (&[0x2f][..], 0x00ff, true, true, 0x0099, true, true, true),
                        (
                            &[0x2f][..],
                            0x0044,
                            false,
                            false,
                            0x0044,
                            false,
                            false,
                            true,
                        ),
                        (&[0x37][..], 0x000a, false, false, 0x0100, true, true, false),
                        (
                            &[0x37][..],
                            0x0209,
                            false,
                            true,
                            0x0209,
                            false,
                            false,
                            false,
                        ),
                        (&[0x3f][..], 0x01ff, true, true, 0x0009, true, true, false),
                        (
                            &[0x3f][..],
                            0x0209,
                            false,
                            true,
                            0x0209,
                            false,
                            false,
                            false,
                        ),
                        (
                            &[0xd4, 10][..],
                            0x003f,
                            false,
                            false,
                            0x0603,
                            false,
                            false,
                            true,
                        ),
                        (
                            &[0xd4, 16][..],
                            0x00ff,
                            false,
                            false,
                            0x0f0f,
                            false,
                            false,
                            true,
                        ),
                        (
                            &[0xd5, 10][..],
                            0x0603,
                            false,
                            false,
                            0x003f,
                            false,
                            false,
                            true,
                        ),
                        (
                            &[0xd5, 16][..],
                            0x0f0f,
                            false,
                            false,
                            0x00ff,
                            false,
                            false,
                            true,
                        ),
                    ] {
                        let code = encoding(wide, wide != p66, wide != p67, body);
                        let (vm, mut cpu) = setup(&code, wide, checked);
                        cpu.cpu.gpr[0] = 0x1234_5678_abcd_0000 | ax;
                        cpu.cpu.flags.set_af(af);
                        cpu.cpu.flags.cf = cf;
                        steps(&mut cpu, &vm, 1);
                        assert_eq!(
                            cpu.cpu.gpr[0],
                            0x1234_5678_abcd_0000 | expected,
                            "{code:02x?}"
                        );
                        if body[0] < 0xd4 {
                            assert_eq!(cpu.cpu.flags.af(), expected_af);
                            assert_eq!(cpu.cpu.flags.cf, expected_cf);
                        }
                        if szp {
                            let al = expected as u8;
                            assert_eq!(cpu.cpu.flags.sf, al & 0x80 != 0);
                            assert_eq!(cpu.cpu.flags.zf, al == 0);
                            assert_eq!(cpu.cpu.flags.pf(), al.count_ones().is_multiple_of(2));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn aam_zero_is_precise_divide_error() {
    // SDM Vol 2A §3.2 AAM p. 3-5: imm8=0 raises #DE.
    for wide in [false, true] {
        for checked in [false, true] {
            let (vm, mut cpu) = setup(&[0xd4, 0], wide, checked);
            cpu.cpu.gpr[0] = 0x1234_5678_abcd_ffff;
            fault(&mut cpu, &vm, 0, None);
        }
    }
}

#[test]
fn wait_mp_ts_and_absent_coprocessor() {
    // PRM §4.1.3 pp. 4-3–4 and §§11.1.2–11.1.4 pp. 11-1–2:
    // WAIT tests TS only with MP=1, independently of EM. §17 p. 17-171.
    // ASSUMED: FpuModel::None ties BUSY#/ERROR# inactive, following
    // §11.1.2 p. 11-2's required absent-coprocessor wiring.
    for wide in [false, true] {
        for checked in [false, true] {
            for fpu in [FpuModel::None, FpuModel::Present] {
                for bits in 0..8u32 {
                    let (vm, mut cpu) = setup(&[0x9b], wide, checked);
                    cpu.set_fpu_model(fpu);
                    cpu.cpu.legacy.cr0 = 1 | bits << 1;
                    if bits & 5 == 5 {
                        fault(&mut cpu, &vm, 7, None);
                    } else {
                        let before = format!("{:?}", cpu.cpu.flags);
                        steps(&mut cpu, &vm, 1);
                        assert_eq!(format!("{:?}", cpu.cpu.flags), before);
                        assert_eq!(cpu.cpu.rip, 0x1001);
                    }
                }
                // PRM §11.1.2 p. 11-2: an asserted ERROR# raises #MF.
                // Engine fixture: ES models a pending coprocessor error
                // (SDM Vol 1 §8.1.3.3 p. 8-5; §8.6 p. 8-31).
                let (vm, mut cpu) = setup(&[0x9b], wide, checked);
                cpu.set_fpu_model(fpu);
                cpu.cpu.fpu_sw = 0x80;
                if fpu == FpuModel::Present {
                    fault(&mut cpu, &vm, 16, None);
                } else {
                    steps(&mut cpu, &vm, 1);
                    assert_eq!(cpu.cpu.fpu_sw, 0x80);
                }
            }
        }
    }
}
