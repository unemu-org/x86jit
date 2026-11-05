//! SDM Vol 2A §3.2 FRNDINT, FXTRACT, FLD constants, FFREE, FINCSTP,
//! FDECSTP, FNOP; §3.3 FIST; Vol 2D §6.1 WAIT/FWAIT. Exact x87 witnesses
//! on the interpreter, JIT and (on x86-64 Linux) native oracle.

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

struct TestJit {
    jit: x86jit_cranelift::JitBackend,
    region: bool,
}
impl TestJit {
    fn backend(region: bool) -> Box<dyn x86jit_core::Backend> {
        Box::new(Self {
            jit: x86jit_cranelift::JitBackend::new(),
            region,
        })
    }
}
impl x86jit_core::Backend for TestJit {
    fn materialize(
        &self,
        ir: &x86jit_core::ir::IrBlock,
        consistency: x86jit_core::MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
    ) -> x86jit_core::CachedBlock {
        use x86jit_core::CachedBlock;
        let block = if self.region {
            self.jit.materialize_region(
                &x86jit_core::ir::IrRegion {
                    entry: ir.guest_start,
                    blocks: vec![ir.clone()],
                    has_loop: false,
                },
                consistency,
                mmio,
                guest_base,
            )
        } else {
            self.jit.materialize(ir, consistency, mmio, guest_base)
        };
        assert!(
            matches!(block, CachedBlock::Compiled { .. }),
            "x87 must compile"
        );
        block
    }
}

fn check(input: VectorInput, mask: u16, mode: CpuMode) -> RunOutcome {
    let interp = run_with_backend_mode(
        &input,
        Box::new(InterpreterBackend),
        GuestCpuFeatures::default(),
        mode,
    );
    let jit = run_with_backend_mode(
        &input,
        TestJit::backend(false),
        GuestCpuFeatures::default(),
        mode,
    );
    assert_eq!(
        (&interp.exit, &jit.exit),
        (&ExitKind::Hlt, &ExitKind::Hlt),
        "both tiers must execute"
    );
    let assert_same = |other: &RunOutcome| {
        let data = &input.mem_init[1].bytes;
        let context = format!(
            "code={:02x?}, operands={:02x?}, cw={:02x?}",
            input.mem_init[0].bytes,
            &data[..32],
            &data[64..66]
        );
        assert_eq!(other.exit, ExitKind::Hlt);
        let (sw, tw) = env(&interp);
        let (osw, otw) = env(other);
        assert_eq!(
            sw & (mask | 0x3800),
            osw & (mask | 0x3800),
            "status word: {context}"
        );
        assert_eq!(tw, otw, "physical tag word: {context}");
        assert_eq!(
            &interp.mem[1].bytes[80..88],
            &other.mem[1].bytes[80..88],
            "stored integer: {context}"
        );
        for i in 0..8 {
            if (tw >> (2 * ((interp.cpu.fpu_top as usize + i) & 7))) & 3 != 3 {
                assert_eq!(interp.cpu.st[i], other.cpu.st[i], "ST({i}): {context}");
            }
        }
    };
    assert_same(&jit);
    let region = run_with_backend_mode(
        &input,
        TestJit::backend(true),
        GuestCpuFeatures::default(),
        mode,
    );
    assert_same(&region);
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    if mode == CpuMode::Long64 {
        let native = x86jit_tests::native::run_native(&input).expect("native x87 witness");
        assert_same(&native);
    }
    interp
}

#[test]
fn frndint_exact_rounding_classes_and_precision_independence() {
    // SDM Vol 2A §3.2 FRNDINT; Vol 1 §§4.8.4.1, 8.5.6: RC, PE, C1.
    for (mode, bits) in [(CpuMode::Long64, 64), (CpuMode::Compat32, 32)] {
        for rc in 0..4 {
            for pc in [0, 0x200, 0x300] {
                for (sig, se, rounded, up) in [
                    (0xa000_0000_0000_0000, 0x3fff, [1, 1, 2, 1][rc], rc == 2),
                    (0xe000_0000_0000_0000, 0xbfff, [-2, -2, -1, -1][rc], rc < 2),
                    (0xa000_0000_0000_0000, 0x4000, [2, 2, 3, 2][rc], rc == 2),
                    (
                        0xe000_0000_0000_0000,
                        0x4000,
                        [4, 3, 4, 3][rc],
                        rc == 0 || rc == 2,
                    ),
                ] {
                    let out = check(
                        input(
                            &[0xd9, 0xfc],
                            Some(enc(sig, se)),
                            None,
                            0x7f | pc | ((rc as u16) << 10),
                            2,
                            bits,
                        ),
                        EXCEPTIONS | C1,
                        mode,
                    );
                    assert_eq!(out.cpu.st[0], number(rounded));
                    assert_eq!(
                        env(&out).0 & (EXCEPTIONS | C1),
                        0x20 | if up { C1 } else { 0 }
                    );
                }
                // Integer values above i64 and PC's precision remain exact.
                for a in [
                    enc(u64::MAX, 0x403e),
                    enc(u64::MAX, 0x407f),
                    number(16_777_217),
                ] {
                    let out = check(
                        input(
                            &[0xd9, 0xfc],
                            Some(a),
                            None,
                            0x7f | pc | ((rc as u16) << 10),
                            2,
                            bits,
                        ),
                        EXCEPTIONS | C1,
                        mode,
                    );
                    assert_eq!(out.cpu.st[0], a);
                    assert_eq!(env(&out).0 & (EXCEPTIONS | C1), 0);
                }
            }
            for sign in [0, 0x8000] {
                let up = rc == 0 || (rc == 1 && sign != 0) || (rc == 2 && sign == 0);
                let expected = if up {
                    enc(1 << 63, 0x403e | sign)
                } else {
                    enc(0xffff_ffff_ffff_fffe, 0x403d | sign)
                };
                let out = check(
                    input(
                        &[0xd9, 0xfc],
                        Some(enc(u64::MAX, 0x403d | sign)),
                        None,
                        0x37f | ((rc as u16) << 10),
                        2,
                        bits,
                    ),
                    EXCEPTIONS | C1,
                    mode,
                );
                assert_eq!(out.cpu.st[0], expected);
                assert_eq!(
                    env(&out).0 & (EXCEPTIONS | C1),
                    0x20 | if up { C1 } else { 0 }
                );
            }
            for (a, expected, exc, up) in [
                (
                    enc(1, 0),
                    if rc == 2 { number(1) } else { number(0) },
                    0x22,
                    rc == 2,
                ),
                (
                    enc(1, 0x8000),
                    if rc == 1 { number(-1) } else { enc(0, 0x8000) },
                    0x22,
                    rc == 1,
                ),
                (enc(0, 0x8000), enc(0, 0x8000), 0, false),
                (enc(1 << 63, 0x7fff), enc(1 << 63, 0x7fff), 0, false),
                (
                    enc(0xc000_0000_0000_1234, 0x7fff),
                    enc(0xc000_0000_0000_1234, 0x7fff),
                    0,
                    false,
                ),
                (
                    enc(0x8000_0000_0000_1234, 0x7fff),
                    enc(0xc000_0000_0000_1234, 0x7fff),
                    1,
                    false,
                ),
                (enc(1, 0x3fff), enc(0xc000_0000_0000_0000, 0xffff), 1, false),
            ] {
                let out = check(
                    input(
                        &[0xd9, 0xfc],
                        Some(a),
                        None,
                        0x37f | ((rc as u16) << 10),
                        2,
                        bits,
                    ),
                    EXCEPTIONS | C1,
                    mode,
                );
                assert_eq!(out.cpu.st[0], expected);
                assert_eq!(
                    env(&out).0 & (EXCEPTIONS | C1),
                    exc | if up { C1 } else { 0 }
                );
            }
        }
    }
}

#[test]
fn fxtract_exact_components_and_special_values() {
    // SDM Vol 2A §3.2 FXTRACT; Vol 1 §§4.8.3.3, 4.8.3.5, 8.5.1.2 Table 8-10, 8.5.3.
    let nan = enc(0xc000_0000_0000_1234, 0x7fff);
    let indefinite = enc(0xc000_0000_0000_0000, 0xffff);
    for (mode, bits) in [(CpuMode::Long64, 64), (CpuMode::Compat32, 32)] {
        for rc in 0..4 {
            for pc in [0, 0x200, 0x300] {
                for (a, sig, exp, exc) in [
                    (
                        enc(0xdead_beef_0000_0001, 0xc012),
                        enc(0xdead_beef_0000_0001, 0xbfff),
                        number(19),
                        0,
                    ),
                    (enc(1, 0), number(1), number(-16445), 2),
                    (enc(1, 0x8000), number(-1), number(-16445), 2),
                    (enc(0, 0), number(0), enc(1 << 63, 0xffff), 4),
                    (enc(0, 0x8000), enc(0, 0x8000), enc(1 << 63, 0xffff), 4),
                    (
                        enc(1 << 63, 0x7fff),
                        enc(1 << 63, 0x7fff),
                        enc(1 << 63, 0x7fff),
                        0,
                    ),
                    (
                        enc(1 << 63, 0xffff),
                        enc(1 << 63, 0xffff),
                        enc(1 << 63, 0x7fff),
                        0,
                    ),
                    (nan, nan, nan, 0),
                    (enc(0x8000_0000_0000_1234, 0x7fff), nan, nan, 1),
                    (enc(1, 0x3fff), indefinite, indefinite, 1),
                ] {
                    let out = check(
                        input(
                            &[0xd9, 0xf4],
                            Some(a),
                            None,
                            0x7f | pc | (rc << 10),
                            2,
                            bits,
                        ),
                        EXCEPTIONS,
                        mode,
                    );
                    assert_eq!(out.cpu.st[0], sig);
                    assert_eq!(out.cpu.st[1], exp);
                    assert_eq!(out.cpu.fpu_top, 6);
                    assert_eq!(env(&out).0 & EXCEPTIONS, exc);
                }
            }
        }
    }
}

#[test]
fn fld_constants_every_rc_and_pc_without_precision_exception() {
    // SDM Vol 1 §§8.3.4, 8.3.8; Vol 2A §3.2 FLD1/.../FLDZ: 66-bit
    // internal constants rounded to extended precision; no PE or C1 rounding flag.
    for (mode, bits) in [(CpuMode::Long64, 64), (CpuMode::Compat32, 32)] {
        for rc in 0..4 {
            for pc in [0, 0x200, 0x300] {
                for (op, sigs, se) in [
                    (0xe8, [1 << 63; 4], 0x3fff),
                    (
                        0xe9,
                        [
                            0xd49a_784b_cd1b_8afe,
                            0xd49a_784b_cd1b_8afe,
                            0xd49a_784b_cd1b_8aff,
                            0xd49a_784b_cd1b_8afe,
                        ],
                        0x4000,
                    ),
                    (
                        0xea,
                        [
                            0xb8aa_3b29_5c17_f0bc,
                            0xb8aa_3b29_5c17_f0bb,
                            0xb8aa_3b29_5c17_f0bc,
                            0xb8aa_3b29_5c17_f0bb,
                        ],
                        0x3fff,
                    ),
                    (
                        0xeb,
                        [
                            0xc90f_daa2_2168_c235,
                            0xc90f_daa2_2168_c234,
                            0xc90f_daa2_2168_c235,
                            0xc90f_daa2_2168_c234,
                        ],
                        0x4000,
                    ),
                    (
                        0xec,
                        [
                            0x9a20_9a84_fbcf_f799,
                            0x9a20_9a84_fbcf_f798,
                            0x9a20_9a84_fbcf_f799,
                            0x9a20_9a84_fbcf_f798,
                        ],
                        0x3ffd,
                    ),
                    (
                        0xed,
                        [
                            0xb172_17f7_d1cf_79ac,
                            0xb172_17f7_d1cf_79ab,
                            0xb172_17f7_d1cf_79ac,
                            0xb172_17f7_d1cf_79ab,
                        ],
                        0x3ffe,
                    ),
                    (0xee, [0; 4], 0),
                ] {
                    let out = check(
                        input(
                            &[0xd9, op],
                            None,
                            None,
                            0x5f | pc | ((rc as u16) << 10),
                            2,
                            bits,
                        ),
                        EXCEPTIONS | C1,
                        mode,
                    );
                    assert_eq!(out.cpu.st[0], enc(sigs[rc], se));
                    assert_eq!(env(&out).0 & (EXCEPTIONS | C1), 0);
                }
            }
        }
    }
}

#[test]
fn fist_nonpop_widths_rounding_invalid_and_empty() {
    // SDM Vol 2A §3.3 FIST/FISTP: FIST has m16/m32 forms and never pops.
    for (mode, bits) in [(CpuMode::Long64, 64), (CpuMode::Compat32, 32)] {
        for rc in 0..4 {
            for (op, width) in [(vec![0xdf, 0x10], 2), (vec![0xdb, 0x10], 4)] {
                for (a, expected, exc) in [
                    (
                        Some(enc(0xe000_0000_0000_0000, 0x3fff)),
                        [2i64, 1, 2, 1][rc],
                        0x20,
                    ),
                    (
                        Some(enc(0xe000_0000_0000_0000, 0xbfff)),
                        [-2, -2, -1, -1][rc],
                        0x20,
                    ),
                    (Some(enc(1 << 63, 0x7fff)), i64::MIN, 1),
                    (Some(enc(0xc000_0000_0000_1234, 0x7fff)), i64::MIN, 1),
                    (
                        Some(number(if width == 2 { 40000 } else { 3_000_000_000 })),
                        i64::MIN,
                        1,
                    ),
                    (None, i64::MIN, 0x41),
                ] {
                    for unmask in [0u16, 1, 0x20] {
                        let mut p = input(
                            &op,
                            a,
                            None,
                            (0x37f | ((rc as u16) << 10)) & !unmask,
                            2,
                            bits,
                        );
                        p.cpu_init.gpr[0] = DATA + 80;
                        p.mem_init[1].bytes[80..88].fill(0xa5);
                        let out = check(p, EXCEPTIONS | C1, mode);
                        let m = &out.mem[1].bytes;
                        if unmask & exc & 1 != 0 {
                            assert_eq!(&m[80..88], &[0xa5; 8]);
                        } else {
                            let value = if exc & 1 != 0 {
                                1u64 << (width * 8 - 1)
                            } else {
                                expected as u64
                            };
                            assert_eq!(&m[80..80 + width], &value.to_le_bytes()[..width]);
                            assert_eq!(&m[80 + width..88], &vec![0xa5; 8 - width]);
                        }
                        assert_eq!(out.cpu.fpu_top, if a.is_some() { 7 } else { 0 });
                        let sw = env(&out).0;
                        assert_eq!(
                            sw & EXCEPTIONS,
                            exc | if unmask & exc != 0 { 0x8080 } else { 0 }
                        );
                        if let Some(a) = a {
                            assert_eq!(out.cpu.st[0], a);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn fwait_reports_pending_mf_at_its_own_rip() {
    // SDM Vol 1 §8.6 and Vol 2D §6.1 WAIT/FWAIT: ES causes #MF at WAIT.
    for bits in [64, 32] {
        for (op, cw, faults) in [
            (&[0x9b][..], 0x37e, true),
            (&[0xd9, 0xd0][..], 0x37e, true),
            (&[0xdd, 0xc0][..], 0x37e, true),
            (&[0xd9, 0xf6][..], 0x37e, true),
            (&[0xd9, 0xf7][..], 0x37e, true),
            (&[0x9b][..], 0x37f, false),
        ] {
            let mut p = input(&[0xd9, 0xfa], Some(number(-1)), None, cw, 2, bits);
            let code = &mut p.mem_init[0].bytes;
            let mut d = iced_x86::Decoder::with_ip(bits, code, CODE, 0);
            let at = d
                .iter()
                .find(|i| i.mnemonic() == iced_x86::Mnemonic::Fnstenv)
                .unwrap()
                .ip();
            code.truncate((at - CODE) as usize);
            code.extend_from_slice(op);
            code.push(0xf4);
            let mode = if bits == 64 {
                CpuMode::Long64
            } else {
                CpuMode::Compat32
            };
            let mut exits = Vec::new();
            for backend in [
                Box::new(InterpreterBackend) as Box<dyn x86jit_core::Backend>,
                TestJit::backend(false),
                TestJit::backend(true),
            ] {
                let out = run_with_backend_mode(&p, backend, GuestCpuFeatures::default(), mode);
                let expected = if faults {
                    ExitKind::Exception {
                        addr: at,
                        vector: 16,
                    }
                } else {
                    ExitKind::Hlt
                };
                exits.push((out.exit, expected, out.cpu.rip));
            }
            assert!(
                exits
                    .iter()
                    .all(|(a, b, rip)| a == b && (!faults || *rip == at)),
                "{exits:?}"
            );
        }
    }
}

#[test]
fn generic_x87_arithmetic_commits_post_operation_results() {
    // SDM Vol 1 §§4.9.2, 8.5.3–8.5.6: pre-operation exceptions preserve
    // operands, post-operation register exceptions commit (and FADDP pops).
    for (op, a, b, cw, result, exc, top, c1) in [
        (
            [0xd8, 0xf1],
            number(1),
            number(3),
            0x35f,
            enc(0xaaaa_aaaa_aaaa_aaab, 0x3ffd),
            0x20,
            6,
            C1,
        ),
        (
            [0xde, 0xc1],
            enc(0x8000_0000_0000_0001, 0x3fff),
            number(1),
            0x35f,
            number(2),
            0x20,
            7,
            0,
        ),
        (
            [0xd8, 0xc9],
            enc(u64::MAX, 0x7ffe),
            number(2),
            0x377,
            enc(u64::MAX, 0x1fff),
            8,
            6,
            0,
        ),
        (
            [0xd8, 0xc9],
            enc(1 << 63, 1),
            enc(1 << 63, 0x3ffe),
            0x36f,
            enc(1 << 63, 0x6000),
            0x10,
            6,
            0,
        ),
        (
            [0xd8, 0xf1],
            number(1),
            number(0),
            0x37b,
            number(1),
            4,
            6,
            0,
        ),
    ] {
        let out = check(
            input(&op, Some(a), Some(b), cw, 2, 64),
            EXCEPTIONS | C1,
            CpuMode::Long64,
        );
        assert_eq!(out.cpu.st[0], result, "op={op:02x?}");
        assert_eq!(out.cpu.fpu_top, top);
        assert_eq!(env(&out).0 & (EXCEPTIONS | C1), exc | 0x8080 | c1);
    }
}

// SDM Vol 1 §4.9.2: zero divide precedes denormal operands, even when
// both exceptions are masked. The lower-priority DE is suppressed.
#[test]
fn both_unmasked_divzero_must_report_ze() {
    let out = check(
        input(
            &[0xd8, 0xf1],
            Some(enc(1, 0)),
            Some(number(0)),
            0x379,
            2,
            64,
        ),
        EXCEPTIONS,
        CpuMode::Long64,
    );
    assert_eq!(env(&out).0 & EXCEPTIONS, 0x8084);
    assert_eq!(out.cpu.st[0], enc(1, 0));
}

#[test]
fn zero_divide_has_priority_over_denormal() {
    for cw in [0x37f, 0x37d, 0x37b, 0x379] {
        let out = check(
            input(&[0xd8, 0xf1], Some(enc(1, 0)), Some(number(0)), cw, 2, 64),
            EXCEPTIONS,
            CpuMode::Long64,
        );
        assert_eq!(
            env(&out).0 & EXCEPTIONS,
            if cw & 4 == 0 { 0x8084 } else { 4 },
            "cw={cw:04x}"
        );
        assert_eq!(
            out.cpu.st[0],
            if cw & 4 == 0 {
                enc(1, 0)
            } else {
                enc(1 << 63, 0x7fff)
            }
        );
    }
}

// SDM Vol 1 §§8.5.4–8.5.6; Vol 2A §3.2 FMUL, FPU Flags Affected:
// C1 records significand rounding even when an unmasked range exception
// commits a biased result. Exact results clear a previously set C1.
#[test]
fn generic_unmasked_range_round_up_sets_c1() {
    let out = check(
        input(
            &[0xd8, 0xc9],
            Some(enc(u64::MAX, 0x7ffe)),
            Some(enc((1 << 63) | 1, 0x4000)),
            0xb47,
            2,
            64,
        ),
        EXCEPTIONS | C1,
        CpuMode::Long64,
    );
    assert_eq!(out.cpu.st[0], enc((1 << 63) | 1, 0x2000));
    assert_eq!(env(&out).0 & (EXCEPTIONS | C1), 0x82a8);
}

#[test]
fn generic_unmasked_range_exact_clears_c1() {
    let mut p = input(
        &[0xd8, 0xc9],
        Some(enc(u64::MAX, 0x7ffe)),
        Some(number(2)),
        0x377,
        2,
        64,
    );
    seed_codes(&mut p, 6, C1);
    let out = check(p, EXCEPTIONS | C1, CpuMode::Long64);
    assert_eq!(out.cpu.st[0], enc(u64::MAX, 0x1fff));
    assert_eq!(env(&out).0 & (EXCEPTIONS | C1), 0x8088);
}

#[test]
fn generic_arithmetic_c1_all_commit_paths() {
    // SDM Vol 1 §§8.1.5.2, 8.5.6; Vol 2A §3.2 FADD/FADDP/FIADD,
    // FPU Flags Affected: both rounding directions replace the previous C1.
    // Cover all four memory classes and all three register destination paths.
    for (op, memory) in [
        ([0xdc, 0x00], Some(1f64.to_bits().to_le_bytes().to_vec())),
        ([0xd8, 0x00], Some(1f32.to_bits().to_le_bytes().to_vec())),
        ([0xde, 0x00], Some(1i16.to_le_bytes().to_vec())),
        ([0xda, 0x00], Some((-1i32).to_le_bytes().to_vec())),
        ([0xd8, 0xc1], None),
        ([0xdc, 0xc1], None),
        ([0xde, 0xc1], None),
    ] {
        for rc in [2, 3] {
            for precision_mask in [0, 0x20] {
                let mut p = input(
                    &op,
                    Some(if op == [0xda, 0x00] {
                        enc((3 << 62) | (1 << 39), 0x4000)
                    } else {
                        enc((1 << 63) | (1 << 39), 0x3fff)
                    }),
                    Some(number(1)),
                    (0x7f | (rc << 10)) & !precision_mask,
                    2,
                    64,
                );
                p.cpu_init.gpr[0] = DATA + 80;
                if let Some(bytes) = &memory {
                    p.mem_init[1].bytes[80..80 + bytes.len()].copy_from_slice(bytes);
                }
                seed_codes(&mut p, 6, if rc == 2 { 0 } else { C1 });
                let out = check(p, EXCEPTIONS | C1, CpuMode::Long64);
                let up = if rc == 2 { C1 } else { 0 };
                assert_eq!(
                    env(&out).0 & (EXCEPTIONS | C1),
                    0x20 | up | if precision_mask == 0 { 0 } else { 0x8080 },
                    "op={op:02x?}, rc={rc}"
                );
                let destination = usize::from(op == [0xdc, 0xc1]);
                assert_eq!(
                    out.cpu.st[destination],
                    enc((1 << 63) | if rc == 2 { 1 << 40 } else { 0 }, 0x4000)
                );
            }
        }
    }
}

fn full_stack(op: &[u8], cw: u16, bits: u32) -> VectorInput {
    let mut p = input(op, None, None, cw, 2, bits);
    p.mem_init[1].bytes[..10].copy_from_slice(&number(3));
    let mut decoder = iced_x86::Decoder::with_ip(bits, &p.mem_init[0].bytes, CODE, 0);
    let at = decoder.iter().nth(2).unwrap().ip();
    let mut asm = CodeAssembler::new(bits).unwrap();
    for _ in 0..8 {
        asm.fld(tbyte_ptr(DATA)).unwrap();
    }
    p.mem_init[0].bytes.splice(
        (at - CODE) as usize..(at - CODE) as usize,
        asm.assemble(at).unwrap(),
    );
    p
}

#[test]
fn fnop_ffree_and_stack_pointer_rotations() {
    // SDM Vol 2A §3.2 FNOP, FFREE, FINCSTP, FDECSTP: no operand
    // exceptions, physical tags/data preserved except FFREE's selected tag.
    for top in 0..8 {
        for (op, delta, freed) in std::iter::once((vec![0xd9, 0xd0], 0, None))
            .chain([(vec![0xd9, 0xf7], 1, None), (vec![0xd9, 0xf6], 7, None)])
            .chain((0..8).map(|i| (vec![0xdd, 0xc0 + i], 0, Some(i))))
        {
            let mut p = full_stack(&op, 0x37f, 64);
            seed_codes(&mut p, top, CODES);
            let mask = EXCEPTIONS | if delta != 0 { C1 } else { 0 };
            let out = check(p, mask, CpuMode::Long64);
            assert_eq!(out.cpu.fpu_top, ((top + delta) & 7) as u8);
            assert_eq!(
                env(&out).1,
                freed.map_or(0, |i| 3 << (((top + i as u16) & 7) * 2))
            );
            assert_eq!(env(&out).0 & mask, 0);
            for i in 0..8 {
                if Some(i as u8) != freed {
                    assert_eq!(out.cpu.st[i], number(3));
                }
            }
        }
    }
    for op in [[0xd9, 0xd0], [0xdd, 0xc7], [0xd9, 0xf6], [0xd9, 0xf7]] {
        let out = check(
            input(&op, None, None, 0x37e, 2, 32),
            EXCEPTIONS,
            CpuMode::Compat32,
        );
        assert_eq!(env(&out).0 & EXCEPTIONS, 0);
        assert_eq!(env(&out).1, 0xffff);
    }
}

#[test]
fn new_x87_stack_faults_and_constant_overflow() {
    // SDM Vol 1 §8.5.1.1: masked faults write indefinite; unmasked faults
    // preserve TOP/data. Vol 2A §3.2 FRNDINT, FXTRACT and FLD constants: C1.
    let indefinite = enc(0xc000_0000_0000_0000, 0xffff);
    for (mode, bits) in [(CpuMode::Long64, 64), (CpuMode::Compat32, 32)] {
        for unmasked in [false, true] {
            let cw = if unmasked { 0x37e } else { 0x37f };
            for (op, pushes) in [([0xd9, 0xfc], false), ([0xd9, 0xf4], true)] {
                let out = check(input(&op, None, None, cw, 2, bits), EXCEPTIONS | C1, mode);
                assert_eq!(
                    env(&out).0 & (EXCEPTIONS | C1),
                    0x41 | if unmasked { 0x8080 } else { 0 }
                );
                assert_eq!(out.cpu.fpu_top, if pushes && !unmasked { 7 } else { 0 });
                assert_eq!(
                    env(&out).1,
                    if unmasked {
                        0xffff
                    } else if pushes {
                        0xbffe
                    } else {
                        0xfffe
                    }
                );
                if !unmasked {
                    assert_eq!(out.cpu.st[0], indefinite);
                    if pushes {
                        assert_eq!(out.cpu.st[1], indefinite);
                    }
                }
            }
            for op in [0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xf4] {
                let out = check(full_stack(&[0xd9, op], cw, bits), EXCEPTIONS | C1, mode);
                assert_eq!(
                    env(&out).0 & (EXCEPTIONS | C1),
                    0x241 | if unmasked { 0x8080 } else { 0 }
                );
                assert_eq!(out.cpu.fpu_top, if unmasked { 0 } else { 7 });
                if !unmasked {
                    assert_eq!(out.cpu.st[0], indefinite);
                    if op == 0xf4 {
                        assert_eq!(out.cpu.st[1], indefinite);
                    }
                } else {
                    assert_eq!(out.cpu.st[0], number(3));
                }
            }
        }
    }
}

#[test]
fn frndint_fxtract_unmasked_exceptions_and_deferred_wait() {
    // SDM Vol 1 §§4.9.2, 8.5.1–8.5.3, 8.5.6, 8.6: preserve operands
    // on pre-operation exceptions; retain rounded FRNDINT on unmasked PE.
    let snan = enc(0x8000_0000_0000_1234, 0x7fff);
    for (op, a, mask, result) in [
        ([0xd9, 0xfc], snan, 1, snan),
        ([0xd9, 0xfc], enc(1, 0), 2, enc(1, 0)),
        (
            [0xd9, 0xfc],
            enc(0xe000_0000_0000_0000, 0x3fff),
            0x20,
            number(2),
        ),
        ([0xd9, 0xf4], snan, 1, snan),
        ([0xd9, 0xf4], enc(1, 0), 2, enc(1, 0)),
        ([0xd9, 0xf4], enc(0, 0x8000), 4, enc(0, 0x8000)),
    ] {
        for (mode, bits) in [(CpuMode::Long64, 64), (CpuMode::Compat32, 32)] {
            let mut p = input(&op, Some(a), None, 0x37f & !mask, 2, bits);
            let out = check(
                p.clone(),
                EXCEPTIONS | if mask == 0x20 { C1 } else { 0 },
                mode,
            );
            assert_eq!(env(&out).0 & EXCEPTIONS, 0x8080 | mask);
            assert_eq!(out.cpu.fpu_top, 7);
            assert_eq!(out.cpu.st[0], result);
            let mut d = iced_x86::Decoder::with_ip(bits, &p.mem_init[0].bytes, CODE, 0);
            let wait = d
                .iter()
                .find(|i| i.mnemonic() == iced_x86::Mnemonic::Fnstenv)
                .unwrap()
                .ip();
            p.mem_init[0].bytes.truncate((wait - CODE) as usize);
            p.mem_init[0].bytes.extend_from_slice(&[0x9b, 0xf4]);
            for backend in [
                Box::new(InterpreterBackend) as Box<dyn x86jit_core::Backend>,
                TestJit::backend(false),
                TestJit::backend(true),
            ] {
                let out = run_with_backend_mode(&p, backend, GuestCpuFeatures::default(), mode);
                assert_eq!(
                    out.exit,
                    ExitKind::Exception {
                        addr: wait,
                        vector: 16
                    }
                );
                assert_eq!(out.cpu.rip, wait);
                assert_eq!(out.cpu.st[0], result);
                assert_eq!(out.cpu.fpu_top, 7);
            }
        }
    }
}

#[test]
fn undocumented_x87_forms_remain_precise_unknown_instructions() {
    // Project provenance boundary: these encodings have no primary-source
    // semantics in the pinned manuals; see deferred.md.
    for (mode, bits) in [(CpuMode::Long64, 64), (CpuMode::Compat32, 32)] {
        for (prefix, base) in [(0xdf, 0xc0), (0xd9, 0xd8)] {
            for i in 0..8 {
                let p = input(&[prefix, base + i], Some(number(3)), None, 0x37f, 2, bits);
                let mut d = iced_x86::Decoder::with_ip(bits, &p.mem_init[0].bytes, CODE, 0);
                let at = d.iter().nth(3).unwrap().ip();
                for backend in [
                    Box::new(InterpreterBackend) as Box<dyn x86jit_core::Backend>,
                    TestJit::backend(false),
                    TestJit::backend(true),
                ] {
                    let out = run_with_backend_mode(&p, backend, GuestCpuFeatures::default(), mode);
                    assert_eq!(out.exit, ExitKind::UnknownInstruction { addr: at });
                    assert_eq!(out.cpu.rip, at);
                    assert_eq!(out.cpu.st[0], number(3));
                    assert_eq!(out.cpu.fpu_top, 7);
                }
            }
        }
    }
}

#[test]
fn x87_rest_executes_inside_a_multiblock_region() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use x86jit_core::{
        ir::{IrBlock, IrRegion, RegionCaps},
        Backend, CachedBlock, MemConsistency,
    };
    struct Regions {
        jit: x86jit_cranelift::JitBackend,
        compiled: Arc<AtomicUsize>,
    }
    impl Backend for Regions {
        fn region_caps(&self) -> Option<RegionCaps> {
            Some(RegionCaps {
                max_blocks: 16,
                max_icount: 256,
            })
        }
        fn materialize(
            &self,
            ir: &IrBlock,
            c: MemConsistency,
            m: Option<(u64, u64)>,
            b: u64,
        ) -> CachedBlock {
            self.jit.materialize(ir, c, m, b)
        }
        fn materialize_region(
            &self,
            ir: &IrRegion,
            c: MemConsistency,
            m: Option<(u64, u64)>,
            b: u64,
        ) -> CachedBlock {
            assert!(ir.has_loop && ir.blocks.len() >= 2);
            let compiled = self.jit.materialize_region(ir, c, m, b);
            assert!(matches!(compiled, CachedBlock::Compiled { .. }));
            self.compiled.fetch_add(1, Ordering::Relaxed);
            compiled
        }
    }
    // Every documented new family occurs within the loop, including a branch
    // through a second block so this witnesses region compilation explicitly.
    let mut asm = CodeAssembler::new(64).unwrap();
    let mut head = asm.create_label();
    let mut body = asm.create_label();
    asm.mov(ecx, 3).unwrap();
    asm.set_label(&mut head).unwrap();
    asm.fninit().unwrap();
    asm.fldpi().unwrap();
    asm.fxtract().unwrap();
    asm.frndint().unwrap();
    asm.fist(word_ptr(DATA + 80)).unwrap();
    asm.fincstp().unwrap();
    asm.fdecstp().unwrap();
    asm.ffree(st1).unwrap();
    asm.fnop().unwrap();
    asm.wait().unwrap();
    asm.jmp(body).unwrap();
    asm.set_label(&mut body).unwrap();
    asm.dec(ecx).unwrap();
    asm.jnz(head).unwrap();
    asm.fnstenv(dword_ptr(DATA + ENV as u64)).unwrap();
    asm.hlt().unwrap();
    let mut p = input(&[], None, None, 0x37f, 2, 64);
    p.mem_init[0].bytes = asm.assemble(CODE).unwrap();
    let expected = check(p.clone(), EXCEPTIONS | C1, CpuMode::Long64);
    let compiled = Arc::new(AtomicUsize::new(0));
    let actual = run_with_backend_mode(
        &p,
        Box::new(Regions {
            jit: x86jit_cranelift::JitBackend::new(),
            compiled: compiled.clone(),
        }),
        GuestCpuFeatures::default(),
        CpuMode::Long64,
    );
    assert_eq!(actual.exit, ExitKind::Hlt);
    assert!(compiled.load(Ordering::Relaxed) > 0);
    assert_eq!(env(&actual), env(&expected));
    assert_eq!(actual.cpu.st[0], expected.cpu.st[0]);
    assert_eq!(actual.mem[1].bytes[80..82], expected.mem[1].bytes[80..82]);
}
