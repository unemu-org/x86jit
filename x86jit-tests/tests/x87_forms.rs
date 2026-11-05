//! SDM Vol 2A §3.2, FCMOVcc, FCOM/FCOMP/FCOMPP, FUCOM/FUCOMP/FUCOMPP,
//! FTST, FXAM, FSQRT, FPREM1 and FSCALE: executable witnesses for the result
//! tables, condition codes, exceptions and stack effects. The same bodies run
//! on both backends and, on x86-64 Linux, on the native oracle.

use iced_x86::code_asm::*;
use x86jit_core::{CpuMode, GuestCpuFeatures, InterpreterBackend};
use x86jit_tests::oracle::{run_with_backend_mode, RunOutcome, VectorInput};
use x86jit_tests::vector::{CpuSnapshot, ExitKind, MemChunk, MemKind, RunSpec};

const CODE: u64 = 0x21_0000;
const DATA: u64 = 0x22_0000;
const ENV: usize = 128;
const C0: u16 = 0x0100;
const C1: u16 = 0x0200;
const C2: u16 = 0x0400;
const C3: u16 = 0x4000;
const EXCEPTIONS: u16 = 0x80ff;
const CODES: u16 = C0 | C1 | C2 | C3;

fn enc(sig: u64, se: u16) -> [u8; 10] {
    let mut b = [0; 10];
    b[..8].copy_from_slice(&sig.to_le_bytes());
    b[8..].copy_from_slice(&se.to_le_bytes());
    b
}
fn number(n: i64) -> [u8; 10] {
    x86jit_core::f80::F80::from_i64(n).to_bytes()
}
fn classes() -> Vec<[u8; 10]> {
    vec![
        number(3),
        number(-3),
        enc(1, 0),
        enc(1, 0x8000),
        enc(0, 0),
        enc(0, 0x8000),
        enc(1 << 63, 0x7fff),
        enc(1 << 63, 0xffff),
        enc(0xc000_0000_0000_1234, 0x7fff),
        enc(0x8000_0000_0000_1234, 0x7fff),
        enc(1, 0x3fff), // unsupported unnormal, SDM Vol 1 §8.2.2 Table 8-3
    ]
}

fn input(
    op: &[u8],
    a: Option<[u8; 10]>,
    b: Option<[u8; 10]>,
    cw: u16,
    flags: u64,
    bits: u32,
) -> VectorInput {
    let mut asm = CodeAssembler::new(bits).unwrap();
    asm.fninit().unwrap();
    asm.fldcw(word_ptr(DATA + 64)).unwrap();
    if b.is_some() {
        asm.fld(tbyte_ptr(DATA + 16)).unwrap();
    }
    if a.is_some() {
        asm.fld(tbyte_ptr(DATA)).unwrap();
    }
    // Reserve an empty ST(0) with ST(1) live when a is absent.
    if a.is_none() && b.is_some() {
        asm.fldenv(dword_ptr(DATA + 96)).unwrap();
    }
    let mut code = asm.assemble(CODE).unwrap();
    code.extend_from_slice(op);
    let mut tail = CodeAssembler::new(bits).unwrap();
    tail.fnstenv(dword_ptr(DATA + ENV as u64)).unwrap();
    tail.hlt().unwrap();
    code.extend(tail.assemble(CODE + code.len() as u64).unwrap());
    let mut page = vec![0; 4096];
    if let Some(a) = a {
        page[..10].copy_from_slice(&a);
    }
    if let Some(b) = b {
        page[16..26].copy_from_slice(&b);
    }
    page[64..66].copy_from_slice(&cw.to_le_bytes());
    page[96..98].copy_from_slice(&cw.to_le_bytes());
    page[100..102].copy_from_slice(&0x3000u16.to_le_bytes());
    if let Some(b) = b {
        let sig = u64::from_le_bytes(b[..8].try_into().unwrap());
        let exp = u16::from_le_bytes([b[8], b[9]]) & 0x7fff;
        let tag = if exp == 0 && sig == 0 {
            1
        } else if exp == 0 || exp == 0x7fff || sig >> 63 == 0 {
            2
        } else {
            0
        };
        page[104..106].copy_from_slice(&(0x3fffu16 | tag << 14).to_le_bytes());
    }
    VectorInput {
        cpu_init: CpuSnapshot {
            flags: x86jit_tests::vector::SnapFlags {
                cf: flags & 1 != 0,
                pf: flags & 4 != 0,
                zf: flags & 0x40 != 0,
                ..Default::default()
            },
            ..Default::default()
        },
        mem_init: vec![
            MemChunk {
                addr: CODE,
                bytes: code,
                kind: MemKind::Ram,
            },
            MemChunk {
                addr: DATA,
                bytes: page,
                kind: MemKind::Ram,
            },
        ],
        entry: CODE,
        run: RunSpec::UntilExit,
    }
}

fn env(out: &RunOutcome) -> (u16, u16) {
    let m = &out.mem.iter().find(|m| m.addr == DATA).unwrap().bytes;
    (
        u16::from_le_bytes([m[ENV + 4], m[ENV + 5]]),
        u16::from_le_bytes([m[ENV + 8], m[ENV + 9]]),
    )
}

fn seed_codes(p: &mut VectorInput, top: u16, codes: u16) {
    let code = &mut p.mem_init[0].bytes;
    let mut decoder = iced_x86::Decoder::with_ip(64, code, CODE, 0);
    let at = decoder
        .iter()
        .take_while(|i| i.mnemonic() != iced_x86::Mnemonic::Fnstenv)
        .last()
        .unwrap()
        .ip();
    let mut asm = CodeAssembler::new(64).unwrap();
    asm.fnstenv(dword_ptr(DATA + 96)).unwrap();
    asm.mov(word_ptr(DATA + 100), ((top << 11) | codes) as i32)
        .unwrap();
    asm.fldenv(dword_ptr(DATA + 96)).unwrap();
    code.splice(
        (at - CODE) as usize..(at - CODE) as usize,
        asm.assemble(at).unwrap(),
    );
}

fn check(input: VectorInput, mask: u16, mode: CpuMode) -> RunOutcome {
    check_with_native(input, mask, mode, true)
}

fn check_with_native(input: VectorInput, mask: u16, mode: CpuMode, _native: bool) -> RunOutcome {
    let interp = run_with_backend_mode(
        &input,
        Box::new(InterpreterBackend),
        GuestCpuFeatures::default(),
        mode,
    );
    assert_eq!(interp.exit, ExitKind::Hlt, "op must execute");
    let jit = run_with_backend_mode(
        &input,
        Box::new(x86jit_cranelift::JitBackend::new()),
        GuestCpuFeatures::default(),
        mode,
    );
    assert_eq!(jit.exit, ExitKind::Hlt);
    let assert_same = |other: &RunOutcome| {
        let data = &input.mem_init[1].bytes;
        let context = format!(
            "code={:02x?}, operands={:02x?}, cw={:02x?}",
            input.mem_init[0].bytes,
            &data[..32],
            &data[64..66]
        );
        let (sw, tw) = env(&interp);
        let (osw, otw) = env(other);
        assert_eq!(
            sw & (mask | 0x3800),
            osw & (mask | 0x3800),
            "status word: {context}"
        );
        assert_eq!(tw, otw, "physical tag word: {context}");
        for i in 0..8 {
            if (tw >> (2 * ((interp.cpu.fpu_top as usize + i) & 7))) & 3 != 3 {
                assert_eq!(interp.cpu.st[i], other.cpu.st[i], "ST({i}): {context}");
            }
        }
    };
    assert_same(&jit);
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    if _native && mode == CpuMode::Long64 {
        let native = x86jit_tests::native::run_native(&input).expect("native x87 witness");
        assert_same(&native);
    }
    interp
}

fn run(op: &[u8], a: Option<[u8; 10]>, b: Option<[u8; 10]>, mask: u16) -> RunOutcome {
    check(
        input(op, a, b, 0x037f, 2, 64),
        mask | EXCEPTIONS,
        CpuMode::Long64,
    )
}

#[test]
fn fcmov_all_conditions_and_classes() {
    // SDM Vol 2A §3.2 FCMOVcc, opcode table: CF, ZF, CF|ZF, PF and complements.
    for (byte, inverse) in [(0xda, false), (0xdb, true)] {
        for (i, selector) in [1u64, 0x40, 0x41, 4].into_iter().enumerate() {
            for flags in [0, 1, 4, 0x40, 0x45] {
                for b in classes() {
                    let out = check(
                        input(
                            &[byte, 0xc1 + 8 * i as u8],
                            Some(number(7)),
                            Some(b),
                            0x037f,
                            flags,
                            64,
                        ),
                        EXCEPTIONS | C1,
                        CpuMode::Long64,
                    );
                    let take = (flags & selector != 0) ^ inverse;
                    assert_eq!(out.cpu.st[0], if take { b } else { number(7) });
                    assert_eq!(out.cpu.flags.to_rflags(), flags | 2);
                    assert_eq!(
                        env(&out).0 & EXCEPTIONS,
                        0,
                        "move must not signal on NaNs or denormals"
                    );
                }
            }
        }
    }
}

#[test]
fn fcmov_empty_sources_and_destination() {
    for (a, b) in [
        (None, None),
        (Some(number(7)), None),
        (None, Some(number(3))),
    ] {
        for flags in [0, 1] {
            let out = check(
                input(&[0xda, 0xc1], a, b, 0x037f, flags, 64),
                EXCEPTIONS | C1,
                CpuMode::Long64,
            );
            assert_eq!(env(&out).0 & (EXCEPTIONS | C1), 0x41);
            assert_eq!(out.cpu.st[0], enc(0xc000_0000_0000_0000, 0xffff));
        }
    }
}

#[test]
fn fsqrt_precision_rounding_and_exceptions() {
    // SDM Vol 1 §8.1.5.2 and Vol 2A §3.2 FSQRT, result table and C1.
    for pc in [0, 2, 3] {
        for rc in 0..4 {
            for a in classes()
                .into_iter()
                .chain([number(2), number(4), enc(u64::MAX, 0x7ffe)])
            {
                check(
                    input(
                        &[0xd9, 0xfa],
                        Some(a),
                        None,
                        0x7f | pc << 8 | rc << 10,
                        2,
                        64,
                    ),
                    EXCEPTIONS | C1,
                    CpuMode::Long64,
                );
            }
            let (down, up, nearest) = match pc {
                0 => (0xb504_f300_0000_0000, 0xb504_f400_0000_0000, false),
                2 => (0xb504_f333_f9de_6000, 0xb504_f333_f9de_6800, true),
                _ => (0xb504_f333_f9de_6484, 0xb504_f333_f9de_6485, false),
            };
            let round_up = rc == 2 || (rc == 0 && nearest);
            let out = check(
                input(
                    &[0xd9, 0xfa],
                    Some(number(2)),
                    None,
                    0x7f | pc << 8 | rc << 10,
                    2,
                    64,
                ),
                EXCEPTIONS | C1,
                CpuMode::Long64,
            );
            assert_eq!(out.cpu.st[0], enc(if round_up { up } else { down }, 0x3fff));
            assert_eq!(
                env(&out).0 & (EXCEPTIONS | C1),
                0x20 | if round_up { C1 } else { 0 }
            );
        }
    }
    let out = run(&[0xd9, 0xfa], Some(number(2)), None, C1);
    assert_eq!(out.cpu.st[0], enc(0xb504_f333_f9de_6484, 0x3fff));
    assert_eq!(env(&out).0 & (EXCEPTIONS | C1), 0x20);
    let out = run(&[0xd9, 0xfa], Some(number(-1)), None, C1);
    assert_eq!(env(&out).0 & EXCEPTIONS, 1);
    let out = run(&[0xd9, 0xfa], None, None, C1);
    assert_eq!(env(&out).0 & (EXCEPTIONS | C1), 0x41);
    assert_eq!(
        run(&[0xd9, 0xfa], Some(enc(0, 0x8000)), None, C1).cpu.st[0],
        enc(0, 0x8000)
    );
}

#[test]
fn fprem1_nearest_even_quotient_and_exact_remainder() {
    // SDM Vol 2A §3.2 FPREM1: Q2 -> C0, Q1 -> C3, Q0 -> C1; exact, RC independent.
    for (a, b, rem, q) in [
        (7, 2, -1, 4),
        (5, 2, 1, 2),
        (3, 2, -1, 2),
        (11, 2, -1, 6),
        (15, 2, -1, 8),
        (2, 2, 0, 1),
        (6, 2, 0, 3),
        (10, 2, 0, 5),
        (14, 2, 0, 7),
    ] {
        for sign in [1, -1] {
            for rc in 0..4 {
                let out = check(
                    input(
                        &[0xd9, 0xf5],
                        Some(number(a * sign)),
                        Some(number(b)),
                        0x037f | rc << 10,
                        2,
                        64,
                    ),
                    EXCEPTIONS | CODES,
                    CpuMode::Long64,
                );
                assert_eq!(
                    out.cpu.st[0],
                    if rem == 0 && sign < 0 {
                        enc(0, 0x8000)
                    } else {
                        number(rem * sign)
                    }
                );
                let codes = (((q & 4 != 0) as u16) * C0)
                    | (((q & 2 != 0) as u16) * C3)
                    | (((q & 1 != 0) as u16) * C1);
                assert_eq!(env(&out).0 & (CODES | EXCEPTIONS), codes);
            }
        }
    }
    for a in classes().into_iter().map(Some).chain([None]) {
        for b in classes().into_iter().map(Some).chain([None]) {
            run(&[0xd9, 0xf5], a, b, C2 | C1);
        }
    }
    // Exponent differences 63 (complete) and 64+ (partial) straddle the SDM
    // threshold. 2^n mod 3 = (-1)^n for the nearest-integer remainder.
    for exponent in [63, 64, 65, 127, 1024, 16383] {
        let a = enc(1 << 63, (16383 + exponent) as u16);
        let once = check_with_native(
            input(&[0xd9, 0xf5], Some(a), Some(number(3)), 0x037f, 2, 64),
            EXCEPTIONS | C2 | C1,
            CpuMode::Long64,
            exponent - 1 < 64,
        );
        assert_eq!(env(&once).0 & C2, if exponent > 64 { C2 } else { 0 });
        // Partial results may differ with the implementation-dependent N (SDM
        // Vol 2A §3.2 FPREM1); compare the final result against hardware instead.
        let mut asm = CodeAssembler::new(64).unwrap();
        let mut again = asm.create_label();
        asm.set_label(&mut again).unwrap();
        asm.fprem1().unwrap();
        asm.fnstsw(ax).unwrap();
        asm.test(ah, 4).unwrap();
        asm.jnz(again).unwrap();
        let op = asm.assemble(CODE).unwrap(); // position-independent relative branch
        let out = run(&op, Some(a), Some(number(3)), CODES);
        assert_eq!(
            out.cpu.st[0],
            number(if exponent % 2 == 0 { 1 } else { -1 })
        );
        assert_eq!(env(&out).0 & (EXCEPTIONS | C2), 0);
    }
}

#[test]
fn fscale_truncates_and_preserves_extended_precision() {
    // SDM Vol 2A §3.2 FSCALE Table 3-36; PC does not narrow its significand.
    for pc in [0, 2, 3] {
        for rc in 0..4 {
            for a in classes().into_iter().chain([
                enc(u64::MAX, 0x7ffe),
                enc(1 << 63, 1),
                enc(1 << 63 | 1, 1),
            ]) {
                for b in classes().into_iter().chain([
                    number(20000),
                    number(-20000),
                    enc(0xe000_0000_0000_0000, 0x3fff),
                ]) {
                    check(
                        input(
                            &[0xd9, 0xfd],
                            Some(a),
                            Some(b),
                            0x7f | pc << 8 | rc << 10,
                            2,
                            64,
                        ),
                        EXCEPTIONS | C1,
                        CpuMode::Long64,
                    );
                }
            }
        }
    }
    let a = enc(0x8000_0000_0000_0001, 0x3fff);
    let out = run(
        &[0xd9, 0xfd],
        Some(a),
        Some(enc(0xe000_0000_0000_0000, 0x3fff)),
        C1,
    );
    assert_eq!(out.cpu.st[0], enc(0x8000_0000_0000_0001, 0x4000));
    let out = run(
        &[0xd9, 0xfd],
        Some(a),
        Some(enc(0xe000_0000_0000_0000, 0xbfff)),
        C1,
    );
    assert_eq!(out.cpu.st[0], enc(0x8000_0000_0000_0001, 0x3ffe));
    // SDM Vol 1 §4.9.1.4 Table 4-11 and §4.9.1.5: directed overflow,
    // gradual underflow, ties to even, UE only for tiny AND inexact when masked.
    for rc in 0..4 {
        for sign in [false, true] {
            let se = if sign { 0x8000 } else { 0 };
            let inf = rc == 0 || (rc == 1 && sign) || (rc == 2 && !sign);
            let out = check(
                input(
                    &[0xd9, 0xfd],
                    Some(enc(u64::MAX, 0x7ffe | se)),
                    Some(number(1)),
                    0x037f | rc << 10,
                    2,
                    64,
                ),
                EXCEPTIONS | C1,
                CpuMode::Long64,
            );
            assert_eq!(
                out.cpu.st[0],
                if inf {
                    enc(1 << 63, 0x7fff | se)
                } else {
                    enc(u64::MAX, 0x7ffe | se)
                }
            );
            assert_eq!(
                env(&out).0 & (EXCEPTIONS | C1),
                0x28 | if inf { C1 } else { 0 }
            );
            let up = (rc == 1 && sign) || (rc == 2 && !sign);
            let out = check(
                input(
                    &[0xd9, 0xfd],
                    Some(enc(1, se)),
                    Some(number(-1)),
                    0x037f | rc << 10,
                    2,
                    64,
                ),
                EXCEPTIONS | C1,
                CpuMode::Long64,
            );
            assert_eq!(out.cpu.st[0], enc(u64::from(up), se));
            assert_eq!(
                env(&out).0 & (EXCEPTIONS | C1),
                0x32 | if up { C1 } else { 0 }
            );
        }
        let out = check(
            input(
                &[0xd9, 0xfd],
                Some(enc(1 << 63, 1)),
                Some(number(-1)),
                0x037f | rc << 10,
                2,
                64,
            ),
            EXCEPTIONS | C1,
            CpuMode::Long64,
        );
        assert_eq!(out.cpu.st[0], enc(1 << 62, 0));
        assert_eq!(env(&out).0 & (EXCEPTIONS | C1), 0);
    }
    for (a, b) in [
        (None, None),
        (Some(number(1)), None),
        (None, Some(number(1))),
    ] {
        assert_eq!(
            env(&run(&[0xd9, 0xfd], a, b, C1)).0 & (EXCEPTIONS | C1),
            0x41
        );
    }
}

#[test]
fn fcom_fucom_ftst_and_fxam_classes() {
    // SDM Vol 1 §4.9.2: invalid and QNaN responses precede denormal.
    for (op, nan, expected) in [
        (
            [0xd8, 0xd1],
            enc(0xc000_0000_0000_1234, 0x7fff),
            C3 | C2 | C0 | 1,
        ),
        (
            [0xdd, 0xe1],
            enc(0xc000_0000_0000_1234, 0x7fff),
            C3 | C2 | C0,
        ),
        (
            [0xdd, 0xe1],
            enc(0x8000_0000_0000_1234, 0x7fff),
            C3 | C2 | C0 | 1,
        ),
    ] {
        assert_eq!(
            env(&run(&op, Some(nan), Some(enc(1, 0)), CODES)).0 & (CODES | EXCEPTIONS),
            expected
        );
    }
    for op in [[0xd8, 0xd1], [0xdd, 0xe1]] {
        for (a, b, codes) in [
            (number(1), number(2), C0),
            (number(3), number(2), 0),
            (number(0), enc(0, 0x8000), C3),
            (number(2), number(2), C3),
            (enc(1, 0), number(1), C0 | 2),
            (enc(1 << 63, 0x3ffe), number(0), 0),
            (enc(1 << 63, 0xbffe), number(0), C0),
        ] {
            assert_eq!(
                env(&run(&op, Some(a), Some(b), CODES)).0 & (CODES | EXCEPTIONS),
                codes
            );
        }
    }
    for (op, unordered, pops) in [
        (&[0xd8, 0xd1][..], false, 0),
        (&[0xd8, 0xd9], false, 1),
        (&[0xde, 0xd9], false, 2),
        (&[0xdd, 0xe1], true, 0),
        (&[0xdd, 0xe9], true, 1),
        (&[0xda, 0xe9], true, 2),
    ] {
        for a in classes().into_iter().map(Some).chain([None]) {
            for b in classes().into_iter().map(Some).chain([None]) {
                let out = run(op, a, b, CODES);
                if let Some(a) = a {
                    if b == Some(number(3)) && a[8..] == [0xff, 0x7f] {
                        let nan = u64::from_le_bytes(a[..8].try_into().unwrap());
                        if nan > 1 << 63 {
                            assert_eq!(
                                env(&out).0 & (CODES | 1),
                                C3 | C2 | C0 | u16::from(!unordered || nan & (1 << 62) == 0)
                            );
                        }
                    }
                }
                if a.is_some() && b.is_some() {
                    assert_eq!(out.cpu.fpu_top, (6 + pops) & 7);
                }
            }
        }
    }
    for a in classes().into_iter().map(Some).chain([None]) {
        run(&[0xd9, 0xe4], a, None, CODES);
    }
    for (a, codes) in [
        (number(1), 0),
        (number(-1), C0),
        (number(0), C3),
        (enc(0, 0x8000), C3),
        (enc(1, 0), 2),
        (enc(0xc000_0000_0000_1234, 0x7fff), C0 | C2 | C3 | 1),
    ] {
        assert_eq!(
            env(&run(&[0xd9, 0xe4], Some(a), None, CODES)).0 & (CODES | EXCEPTIONS),
            codes
        );
    }
    // SDM Vol 2A §3.2 FXAM Table 3-44, including sign of an empty register.
    for (a, codes) in [
        (number(3), C2),
        (number(0), C3),
        (enc(1, 0), C3 | C2),
        (enc(1 << 63, 0x7fff), C2 | C0),
        (enc(0xc000_0000_0000_1234, 0x7fff), C0),
        (enc(0x8000_0000_0000_1234, 0x7fff), C0),
        (enc(1, 0x3fff), 0),
    ] {
        for sign in [0, 0x80] {
            let mut a = a;
            a[9] |= sign;
            let out = run(&[0xd9, 0xe5], Some(a), None, CODES);
            assert_eq!(
                env(&out).0 & (CODES | EXCEPTIONS),
                codes | if sign != 0 { C1 } else { 0 }
            );
        }
    }
    let out = run(&[0xd9, 0xe5], None, None, CODES);
    assert_eq!(env(&out).0 & (CODES | EXCEPTIONS) & !C1, C3 | C0);
    // Mark a negative register empty, then examine it: C1 still reads its sign.
    let mut p = input(&[0xd9, 0xe5], None, Some(number(-3)), 0x037f, 2, 64);
    p.mem_init[1].bytes[100..102].copy_from_slice(&0x3800u16.to_le_bytes());
    p.mem_init[1].bytes[104..106].copy_from_slice(&0xffffu16.to_le_bytes());
    let out = check(p, EXCEPTIONS | CODES, CpuMode::Long64);
    assert_eq!(env(&out).0 & (CODES | EXCEPTIONS), C3 | C1 | C0);

    // SDM Vol 2A §3.2 FCOM: m32/m64, including a signaling NaN whose
    // quiet bit must survive widening until the compare reports IE.
    for width in [4, 8] {
        let values: Vec<(u64, u16)> = if width == 4 {
            vec![
                (2f32.to_bits() as u64, C0),
                (1f32.to_bits() as u64, C3),
                (0, 0),
                (0x8000_0000, 0),
                (1, 2),
                (0x7f80_0000, C0),
                (0xff80_0000, 0),
                (0x7fc0_1234, C3 | C2 | C0 | 1),
                (0x7f80_1234, C3 | C2 | C0 | 1),
            ]
        } else {
            vec![
                (2f64.to_bits(), C0),
                (1f64.to_bits(), C3),
                (0, 0),
                (1 << 63, 0),
                (1, 2),
                (f64::INFINITY.to_bits(), C0),
                (f64::NEG_INFINITY.to_bits(), 0),
                (0x7ff8_0000_0000_1234, C3 | C2 | C0 | 1),
                (0x7ff0_0000_0000_1234, C3 | C2 | C0 | 1),
            ]
        };
        for pop in [false, true] {
            let mut op = vec![0x48, 0xb8]; // mov rax, memory operand
            op.extend_from_slice(&(DATA + 32).to_le_bytes());
            op.extend_from_slice(&[
                if width == 4 { 0xd8 } else { 0xdc },
                if pop { 0x18 } else { 0x10 },
            ]);
            for (bits, expected) in &values {
                let mut p = input(&op, Some(number(1)), None, 0x037f, 2, 64);
                p.mem_init[1].bytes[32..32 + width].copy_from_slice(&bits.to_le_bytes()[..width]);
                let out = check(p, EXCEPTIONS | CODES, CpuMode::Long64);
                assert_eq!(env(&out).0 & (EXCEPTIONS | CODES), *expected);
                assert_eq!(out.cpu.fpu_top, if pop { 0 } else { 7 });
            }
        }
    }
}

#[test]
fn x87_new_encodings_lift_in_both_modes() {
    let mut forms = vec![
        vec![0xd9, 0xfa],
        vec![0xd9, 0xf5],
        vec![0xd9, 0xfd],
        vec![0xd9, 0xe4],
        vec![0xd9, 0xe5],
        vec![0xde, 0xd9],
        vec![0xda, 0xe9],
    ];
    for i in 0..8 {
        for (prefix, bases) in [
            (0xda, &[0xc0, 0xc8, 0xd0, 0xd8][..]),
            (0xdb, &[0xc0, 0xc8, 0xd0, 0xd8]),
            (0xd8, &[0xd0, 0xd8]),
            (0xdc, &[0xd0, 0xd8]),
            (0xdd, &[0xe0, 0xe8]),
            (0xde, &[0xd0]),
        ] {
            for base in bases {
                forms.push(vec![prefix, base + i]);
            }
        }
    }
    for prefix in [0xd8, 0xdc] {
        for group in [0x10, 0x18] {
            forms.push(vec![prefix, group]);
        }
    }
    let mut missing = Vec::new();
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for form in &forms {
            let mut mem = x86jit_core::Memory::new(x86jit_core::MemoryModel::Flat { size: 4096 });
            mem.map(
                0,
                4096,
                x86jit_core::Prot::RWX,
                x86jit_core::RegionKind::Ram,
            )
            .unwrap();
            mem.write_bytes(0, form).unwrap();
            if x86jit_core::lift::lift_one(&mem, x86jit_core::lift::FetchAddr::flat(0), mode)
                .is_err()
            {
                missing.push(format!("{mode:?} {form:02x?}"));
            }
        }
    }
    assert!(missing.is_empty(), "missing: {missing:?}");
    for op in [
        [0xda, 0xc1],
        [0xdb, 0xc9],
        [0xd9, 0xfa],
        [0xd9, 0xf5],
        [0xd9, 0xfd],
        [0xd8, 0xd1],
        [0xdc, 0xd1],
        [0xdc, 0xd9],
        [0xde, 0xd1],
        [0xde, 0xd9],
        [0xdd, 0xe1],
        [0xdd, 0xe9],
        [0xda, 0xe9],
        [0xd9, 0xe4],
        [0xd9, 0xe5],
    ] {
        for (mode, bits) in [(CpuMode::Long64, 64), (CpuMode::Compat32, 32)] {
            check(
                input(&op, Some(number(3)), Some(number(2)), 0x037f, 1, bits),
                EXCEPTIONS | CODES,
                mode,
            );
        }
    }
}

#[test]
fn new_x87_unmasked_exceptions_commit_per_class_and_defer_mf() {
    // SDM Vol 1 §4.9.2, §8.5.1.1–§8.5.6 and §8.6: pre-operation faults
    // retain operands; post-operation faults store their rounded/biased result.
    let mut cases = vec![
        ([0xda, 0xc1], Some(number(1)), None, 1, 0x41),
        ([0xd9, 0xfa], Some(number(-1)), None, 1, 1),
        ([0xd9, 0xfa], Some(enc(1, 0)), None, 2, 2),
        ([0xd9, 0xfa], Some(number(2)), None, 0x20, 0x20),
        (
            [0xd9, 0xfd],
            Some(enc(u64::MAX, 0x7ffe)),
            Some(number(1)),
            8,
            8,
        ),
        (
            [0xd9, 0xfd],
            Some(enc(1 << 63, 1)),
            Some(number(-1)),
            0x10,
            0x10,
        ),
        ([0xd9, 0xf5], Some(number(1)), Some(number(0)), 1, 1),
        ([0xd9, 0xf5], Some(enc(1, 0)), Some(number(1)), 2, 2),
        (
            [0xd8, 0xd1],
            Some(enc(0xc000_0000_0000_1234, 0x7fff)),
            Some(number(1)),
            1,
            1,
        ),
        ([0xde, 0xd9], Some(number(1)), None, 1, 0x41),
        (
            [0xdd, 0xe9],
            Some(enc(0x8000_0000_0000_1234, 0x7fff)),
            Some(number(1)),
            1,
            1,
        ),
        ([0xda, 0xe9], Some(number(1)), Some(enc(1, 0)), 2, 2),
        ([0xd9, 0xe4], None, None, 1, 0x41),
        ([0xd9, 0xfd], Some(number(1)), Some(number(65536)), 8, 8),
        (
            [0xd9, 0xfd],
            Some(number(1)),
            Some(number(-65536)),
            0x10,
            0x10,
        ),
        (
            [0xd9, 0xfd],
            Some(enc(u64::MAX, 0x7ffe)),
            Some(number(1)),
            0x20,
            0x28,
        ),
        ([0xd9, 0xfd], Some(enc(1, 0)), Some(number(-1)), 0x20, 0x32),
        ([0xd9, 0xf5], Some(enc(3, 0)), Some(enc(2, 0)), 0x10, 0x12),
    ];
    for base in [0xc1, 0xc9, 0xd1, 0xd9] {
        for byte in [0xda, 0xdb] {
            cases.push(([byte, base], Some(number(1)), None, 1, 0x41));
        }
    }
    for (op, a, b, unmask, expected) in cases {
        let mut p = input(&op, a, b, 0x037f & !unmask, 2, 64);
        let comparison = matches!(
            op,
            [0xd8, 0xd1] | [0xde, 0xd9] | [0xdd, 0xe9] | [0xda, 0xe9] | [0xd9, 0xe4]
        );
        if comparison {
            seed_codes(
                &mut p,
                if b.is_some() {
                    6
                } else if a.is_some() {
                    7
                } else {
                    0
                },
                C0 | C1 | C2,
            );
        }
        let result = match (op, unmask) {
            ([0xd9, 0xfa], 0x20) => Some(enc(0xb504_f333_f9de_6484, 0x3fff)),
            ([0xd9, 0xfd], 8) => Some(if b == Some(number(65536)) {
                enc(1 << 63, 0x7fff)
            } else {
                enc(u64::MAX, 0x1fff)
            }),
            ([0xd9, 0xfd], 0x10) => Some(if b == Some(number(-65536)) {
                number(0)
            } else {
                enc(1 << 63, 0x6000)
            }),
            ([0xd9, 0xfd], 0x20) => Some(if a == Some(enc(u64::MAX, 0x7ffe)) {
                enc(1 << 63, 0x7fff)
            } else {
                number(0)
            }),
            ([0xd9, 0xf5], 0x10) => Some(enc(1 << 63, 0xdfc2)),
            _ => a,
        };
        // FNSTENV saves pending status without waiting, so these unmasked
        // cases can also be witnessed natively without delivering SIGFPE.
        let observed = check(
            p.clone(),
            EXCEPTIONS | if comparison { CODES } else { C1 },
            CpuMode::Long64,
        );
        assert_eq!(env(&observed).0 & EXCEPTIONS, expected | 0x8080);
        if comparison {
            assert_eq!(env(&observed).0 & CODES, C0 | C2);
        }
        if let Some(result) = result {
            assert_eq!(observed.cpu.st[0], result);
        }
        let code = &mut p.mem_init[0].bytes;
        let mut decoder = iced_x86::Decoder::with_ip(64, code, CODE, 0);
        let end = decoder
            .iter()
            .filter(|i| i.mnemonic() == iced_x86::Mnemonic::Fnstenv)
            .last()
            .unwrap()
            .ip();
        code.truncate((end - CODE) as usize);
        let mut tail = CodeAssembler::new(64).unwrap();
        tail.fnstsw(word_ptr(DATA + 80)).unwrap();
        let marker = end + tail.assemble(end).unwrap().len() as u64;
        tail.fsqrt().unwrap();
        tail.hlt().unwrap();
        code.extend(tail.assemble(end).unwrap());
        for backend in [
            Box::new(InterpreterBackend) as Box<dyn x86jit_core::Backend>,
            Box::new(x86jit_cranelift::JitBackend::new()),
        ] {
            let out = x86jit_tests::oracle::run_with_backend(&p, backend);
            assert_eq!(
                out.exit,
                ExitKind::Exception {
                    addr: marker,
                    vector: 16
                },
                "op={op:02x?}"
            );
            assert_eq!(out.cpu.rip, marker);
            let data = &out.mem.iter().find(|m| m.addr == DATA).unwrap().bytes;
            let sw = u16::from_le_bytes([data[80], data[81]]);
            assert_eq!(sw & EXCEPTIONS, expected | 0x8080, "op={op:02x?}");
            if let Some(result) = result {
                assert_eq!(out.cpu.st[0], result);
            }
            if let Some(b) = b {
                assert_eq!(out.cpu.st[1], b);
            }
            assert_eq!(
                out.cpu.fpu_top,
                if b.is_some() {
                    6
                } else if a.is_some() {
                    7
                } else {
                    0
                }
            );
        }
    }
}
