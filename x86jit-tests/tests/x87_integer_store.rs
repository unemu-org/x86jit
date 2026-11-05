//! SDM Vol 1 §§4.2.1.2, 8.2.1, 8.5.1.2, 8.5.6 and Vol 2A §3.3
//! FIST/FISTP, FISTTP: destination-width integer conversion and exception commits.

use iced_x86::code_asm::*;
use x86jit_core::{Backend, CpuMode, GuestCpuFeatures, InterpreterBackend};
use x86jit_tests::oracle::{run_with_backend_mode, RunOutcome, VectorInput};
use x86jit_tests::vector::{ExitKind, MemChunk, MemKind, RunSpec};

const CODE: u64 = 0x21_0000;
const DATA: u64 = 0x22_0000;
const DEST: usize = 32;
const ENV: usize = 128;
const STATUS: u16 = 0x82ff; // exceptions, ES/B and C1; C0/C2/C3 are undefined

fn enc(sig: u64, se: u16) -> [u8; 10] {
    let mut b = [0; 10];
    b[..8].copy_from_slice(&sig.to_le_bytes());
    b[8..].copy_from_slice(&se.to_le_bytes());
    b
}

fn integer(n: i64) -> [u8; 10] {
    x86jit_core::f80::F80::from_i64(n).to_bytes()
}

fn float(n: f64) -> [u8; 10] {
    x86jit_core::f80::F80::from_f64(n.to_bits()).to_bytes()
}

fn program(
    width: usize,
    trunc: bool,
    value: Option<[u8; 10]>,
    cw: u16,
    bits: u32,
    wait: bool,
) -> (VectorInput, u64) {
    let mut asm = CodeAssembler::new(bits).unwrap();
    asm.fninit().unwrap();
    asm.fldcw(word_ptr(DATA + 64)).unwrap();
    if value.is_some() {
        asm.fld(tbyte_ptr(DATA)).unwrap();
    }
    // Seed C1 so exact, invalid and truncating conversions must actively clear it.
    asm.fnstenv(dword_ptr(DATA + 96)).unwrap();
    asm.or(word_ptr(DATA + 100), 0x200).unwrap();
    asm.fldenv(dword_ptr(DATA + 96)).unwrap();
    match (width, trunc) {
        (2, false) => asm.fistp(word_ptr(DATA + DEST as u64)),
        (4, false) => asm.fistp(dword_ptr(DATA + DEST as u64)),
        (8, false) => asm.fistp(qword_ptr(DATA + DEST as u64)),
        (2, true) => asm.fisttp(word_ptr(DATA + DEST as u64)),
        (4, true) => asm.fisttp(dword_ptr(DATA + DEST as u64)),
        (8, true) => asm.fisttp(qword_ptr(DATA + DEST as u64)),
        _ => unreachable!(),
    }
    .unwrap();
    let marker;
    if wait {
        asm.fnstsw(word_ptr(DATA + ENV as u64 + 4)).unwrap();
        marker = CODE + asm.assemble(CODE).unwrap().len() as u64;
        asm.fsqrt().unwrap();
    } else {
        marker = 0;
        asm.fnstenv(dword_ptr(DATA + ENV as u64)).unwrap();
    }
    asm.hlt().unwrap();
    let mut page = vec![0; 4096];
    if let Some(value) = value {
        page[..10].copy_from_slice(&value);
    }
    page[DEST..DEST + 8].fill(0xa5);
    page[64..66].copy_from_slice(&cw.to_le_bytes());
    (
        VectorInput {
            cpu_init: Default::default(),
            mem_init: vec![
                MemChunk {
                    addr: CODE,
                    bytes: asm.assemble(CODE).unwrap(),
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
        },
        marker,
    )
}

fn data(out: &RunOutcome) -> &[u8] {
    &out.mem.iter().find(|m| m.addr == DATA).unwrap().bytes
}

fn status(out: &RunOutcome) -> u16 {
    u16::from_le_bytes(data(out)[ENV + 4..ENV + 6].try_into().unwrap())
}

fn check(p: &VectorInput, backend: Box<dyn Backend>, bits: u32) -> RunOutcome {
    let out = run_with_backend_mode(
        p,
        backend,
        GuestCpuFeatures::default(),
        if bits == 64 {
            CpuMode::Long64
        } else {
            CpuMode::Compat32
        },
    );
    assert_eq!(out.exit, ExitKind::Hlt);
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    if bits == 64 {
        let native = x86jit_tests::native::run_native(p).expect("native integer-store witness");
        assert_eq!(native.exit, ExitKind::Hlt);
        assert_eq!(&data(&out)[DEST..DEST + 8], &data(&native)[DEST..DEST + 8]);
        assert_eq!(
            status(&out) & (STATUS | 0x3800),
            status(&native) & (STATUS | 0x3800)
        );
        assert_eq!(
            &data(&out)[ENV + 8..ENV + 10],
            &data(&native)[ENV + 8..ENV + 10]
        );
        assert_eq!(out.cpu.fpu_top, native.cpu.fpu_top);
        if out.cpu.fpu_top == 7 {
            assert_eq!(out.cpu.st[0], native.cpu.st[0]);
        }
    }
    out
}

fn invalid(backend: fn() -> Box<dyn Backend>, unmasked: bool) {
    // SDM Vol 1 §8.5.1.2 Table 8-10 includes both NaNs, infinity and unsupported
    // formats. §8.2.1 distinguishes valid signed minima from integer indefinite by IE.
    for bits in [64, 32] {
        for width in [2, 4, 8] {
            for trunc in [false, true] {
                let mut values = match width {
                    2 => vec![integer(40000), integer(-40000)],
                    4 => vec![integer(3_000_000_000), integer(-3_000_000_000)],
                    _ => vec![enc(1 << 63, 0x403e), enc((1 << 63) + 1, 0xc03e)],
                };
                values.extend([
                    float(1e30),
                    float(-1e30),
                    enc(1 << 63, 0x7fff),
                    enc(1 << 63, 0xffff),
                    enc(0xc000_0000_0000_1234, 0x7fff),
                    enc(0x8000_0000_0000_1234, 0x7fff),
                    enc(1, 0x3fff),
                ]);
                for value in values {
                    let (p, _) = program(
                        width,
                        trunc,
                        Some(value),
                        if unmasked { 0x037e } else { 0x037f },
                        bits,
                        false,
                    );
                    let out = check(&p, backend(), bits);
                    let mut expected = [0xa5; 8];
                    if !unmasked {
                        expected[..width].fill(0);
                        expected[width - 1] = 0x80;
                    }
                    assert_eq!(
                        &data(&out)[DEST..DEST + 8],
                        &expected,
                        "width={width}, trunc={trunc}, source={value:02x?}"
                    );
                    assert_eq!(status(&out) & STATUS, if unmasked { 0x8081 } else { 1 });
                    assert_eq!(out.cpu.fpu_top, if unmasked { 7 } else { 0 });
                    let tag = u16::from_le_bytes(data(&out)[ENV + 8..ENV + 10].try_into().unwrap());
                    if unmasked {
                        assert_eq!(tag & 0x3fff, 0x3fff);
                        assert_ne!(tag >> 14, 3);
                    } else {
                        assert_eq!(tag, 0xffff);
                    }
                    if unmasked {
                        assert_eq!(out.cpu.st[0], value);
                    }
                }
            }
        }
    }
}

fn rounding(backend: fn() -> Box<dyn Backend>) {
    // SDM Vol 1 §4.8.4.1 Table 4-9 and §8.5.6: PE and C1 follow the
    // magnitude rounding. Unmasked precision retains the stored result and pop
    // (§4.9.1.6); invalid conversion takes precedence over PE (§4.9.2).
    for width in [2, 4, 8] {
        for trunc in [false, true] {
            for rc in 0..4 {
                for unmask_precision in [false, true] {
                    for (value, rounded, up) in [
                        (1.5, [2, 1, 2, 1], [true, false, true, false]),
                        (-1.5, [-2, -2, -1, -1], [true, true, false, false]),
                        (0.75, [1, 0, 1, 0], [true, false, true, false]),
                        (-0.75, [-1, -1, 0, 0], [true, true, false, false]),
                        (0.5, [0, 0, 1, 0], [false, false, true, false]),
                        (0.25, [0, 0, 1, 0], [false, false, true, false]),
                        (-0.25, [0, -1, 0, 0], [false, true, false, false]),
                        (2.5, [2, 2, 3, 2], [false, false, true, false]),
                    ] {
                        let effective_rc = if trunc { 3 } else { rc };
                        let cw = (0x037f | (rc << 10)) & if unmask_precision { !0x20 } else { !0 };
                        let (p, _) = program(width, trunc, Some(float(value)), cw, 64, false);
                        let out = check(&p, backend(), 64);
                        let n = rounded[effective_rc as usize] as i64;
                        assert_eq!(&data(&out)[DEST..DEST + width], &n.to_le_bytes()[..width]);
                        let c1 = if up[effective_rc as usize] { 0x200 } else { 0 };
                        assert_eq!(
                            status(&out) & STATUS,
                            0x20 | c1 | if unmask_precision { 0x8080 } else { 0 }
                        );
                        assert_eq!(out.cpu.fpu_top, 0);
                    }
                }
            }
            let (min, max) = match width {
                2 => (i16::MIN as i64, i16::MAX as i64),
                4 => (i32::MIN as i64, i32::MAX as i64),
                _ => (i64::MIN, i64::MAX),
            };
            for n in [min, max, 0] {
                let (p, _) = program(width, trunc, Some(integer(n)), 0x037f, 64, false);
                let out = check(&p, backend(), 64);
                assert_eq!(&data(&out)[DEST..DEST + width], &n.to_le_bytes()[..width]);
                assert_eq!(status(&out) & STATUS, 0);
            }
            let (positive_half, negative_half) = if width == 8 {
                (enc(u64::MAX, 0x403d), enc(u64::MAX, 0xc03d))
            } else {
                (float(max as f64 + 0.5), float(min as f64 - 0.5))
            };
            for (source, n, flags) in [
                (
                    positive_half,
                    if trunc { max } else { min },
                    if trunc { 0x20 } else { 1 },
                ),
                (
                    negative_half,
                    if width == 8 && trunc { min + 1 } else { min },
                    if trunc || width != 8 { 0x20 } else { 0x220 },
                ),
            ] {
                let (p, _) = program(width, trunc, Some(source), 0x037f, 64, false);
                let out = check(&p, backend(), 64);
                assert_eq!(&data(&out)[DEST..DEST + width], &n.to_le_bytes()[..width]);
                assert_eq!(status(&out) & STATUS, flags);
            }
        }
    }
}

fn empty_stack(backend: fn() -> Box<dyn Backend>) {
    // SDM Vol 1 §8.5.1.1: masked underflow stores indefinite and pops;
    // unmasked underflow preserves TOP and memory, and clears C1 in both cases.
    for width in [2, 4, 8] {
        for trunc in [false, true] {
            for unmasked in [false, true] {
                let (p, _) = program(
                    width,
                    trunc,
                    None,
                    if unmasked { 0x037e } else { 0x037f },
                    64,
                    false,
                );
                let out = check(&p, backend(), 64);
                let mut expected = [0xa5; 8];
                if !unmasked {
                    expected[..width].fill(0);
                    expected[width - 1] = 0x80;
                }
                assert_eq!(&data(&out)[DEST..DEST + 8], &expected);
                assert_eq!(status(&out) & STATUS, if unmasked { 0x80c1 } else { 0x41 });
                assert_eq!(out.cpu.fpu_top, if unmasked { 0 } else { 1 });
            }
        }
    }
}

fn deferred_mf(backend: fn() -> Box<dyn Backend>) {
    // SDM Vol 1 §8.6: FNSTSW reads pending status; the next waiting op delivers #MF.
    for width in [2, 4, 8] {
        for trunc in [false, true] {
            let (p, marker) = program(width, trunc, Some(float(1e30)), 0x037e, 64, true);
            let out =
                run_with_backend_mode(&p, backend(), GuestCpuFeatures::default(), CpuMode::Long64);
            assert_eq!(
                out.exit,
                ExitKind::Exception {
                    addr: marker,
                    vector: 16
                }
            );
            assert_eq!(out.cpu.rip, marker);
            assert_eq!(status(&out) & STATUS, 0x8081);
            assert_eq!(&data(&out)[DEST..DEST + 8], &[0xa5; 8]);
            assert_eq!(out.cpu.fpu_top, 7);
            assert_eq!(out.cpu.st[0], float(1e30));
        }
    }
}

macro_rules! tier_tests {
    ($module:ident, $backend:expr) => {
        mod $module {
            use super::*;
            fn backend() -> Box<dyn Backend> {
                $backend
            }
            #[test]
            fn masked_invalid_integer_store() {
                invalid(backend, false);
            }
            #[test]
            fn unmasked_invalid_integer_store() {
                invalid(backend, true);
            }
            #[test]
            fn integer_store_rounding_and_boundaries() {
                rounding(backend);
            }
            #[test]
            fn integer_store_stack_underflow() {
                empty_stack(backend);
            }
            #[test]
            fn integer_store_defers_mf() {
                deferred_mf(backend);
            }
        }
    };
}

tier_tests!(interp, Box::new(InterpreterBackend));
tier_tests!(jit, Box::new(x86jit_cranelift::JitBackend::new()));
