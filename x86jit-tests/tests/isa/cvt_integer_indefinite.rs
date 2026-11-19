//! Masked-invalid conversion results: SDM Vol 2A §3.3, CVTSS2SI/CVTSD2SI,
//! CVTTSS2SI/CVTTSD2SI, CVTPS2DQ/CVTPD2DQ, CVTTPS2DQ/CVTTPD2DQ;
//! Vol 2C §5.2, VCVTSS2USI/VCVTSD2USI/VCVTTSS2USI/VCVTTSD2USI.

use iced_x86::code_asm::*;
#[cfg(any(feature = "unicorn", all(target_arch = "x86_64", target_os = "linux")))]
use iced_x86::EncodingKind;
use iced_x86::{Code, Instruction, MemoryOperand, Register};
use x86jit_core::{
    Backend, Exit, GuestCpuFeatures, InterpreterBackend, Prot, Reg, RegionCaps, RegionKind, Vm,
    VmConfig,
};
use x86jit_cranelift::JitBackend;
use x86jit_tests::oracle::VectorInput;
use x86jit_tests::vector::{CpuSnapshot, MemChunk, MemKind, RunSpec};

const ENTRY: u64 = 0x20000;
const DATA: u64 = 0x30000;
const OUTPUT: u64 = 0x40000;

#[derive(Debug)]
struct Form {
    code: Code,
    double: bool,
    trunc: bool,
    signed: bool,
    scalar: bool,
    bits: u32,
    wide: bool,
}

fn forms(scalar: bool) -> Vec<Form> {
    Code::values()
        .filter_map(|code| {
            let n = format!("{code:?}");
            let lower = n.to_ascii_lowercase();
            let scalar_form = [
                "cvtss2si_",
                "cvttss2si_",
                "cvtsd2si_",
                "cvttsd2si_",
                "cvtss2usi_",
                "cvttss2usi_",
                "cvtsd2usi_",
                "cvttsd2usi_",
            ]
            .iter()
            .any(|s| lower.contains(s));
            let packed_form = ["cvtps2dq_", "cvttps2dq_", "cvtpd2dq_", "cvttpd2dq_"]
                .iter()
                .any(|s| lower.contains(s));
            if (scalar && !scalar_form) || (!scalar && !packed_form) || lower.contains("zmm") {
                return None;
            }
            Some(Form {
                code,
                double: lower.contains("sd2") || lower.contains("pd2"),
                trunc: lower.contains("cvtt"),
                signed: !lower.contains("2usi"),
                scalar,
                bits: if lower.contains("_r64_") { 64 } else { 32 },
                wide: lower.contains("ymm"),
            })
        })
        .collect()
}

// Exact integer-boundary witnesses plus NaNs with their quiet/signalling bit preserved.
fn operands(double: bool) -> Vec<u64> {
    let values = [
        f64::INFINITY,
        f64::NEG_INFINITY,
        2147483648.0,
        -2147483648.0,
        9223372036854775808.0,
        -9223372036854775808.0,
        4294967296.0,
        18446744073709551616.0,
        -1.0,
        -0.5,
        -0.75,
        -1.5,
        -0.9999999999999999,
        -1.0000000000000002,
        -0.0,
        2147483647.5,
        2147483647.75,
        -2147483649.0,
        -2147483648.5,
        2147483647.0,
        2147483520.0,
        4294967295.0,
        9223372036854774784.0,
        18446744073709549568.0,
        1.5,
    ];
    let mut out: Vec<_> = values
        .into_iter()
        .map(|f| {
            if double {
                f.to_bits()
            } else {
                (f as f32).to_bits() as u64
            }
        })
        .collect();
    out.extend(if double {
        [0x7ff8000000000001, 0x7ff0000000000001]
    } else {
        [0x7fc00001, 0x7f800001]
    });
    out.extend([
        1,
        if double {
            0x8000000000000001
        } else {
            0x80000001
        },
    ]);
    out
}

fn expected(raw: u64, form: &Form, mode: u32) -> u64 {
    // SDM Vol 1 §10.2.3.4: DAZ replaces subnormal inputs with signed zero.
    let magnitude = raw
        & if form.double {
            i64::MAX as u64
        } else {
            i32::MAX as u64
        };
    let raw = if mode & 0x40 != 0 && magnitude < if form.double { 1 << 52 } else { 1 << 23 } {
        raw & if form.double { 1 << 63 } else { 1 << 31 }
    } else {
        raw
    };
    let rc = (mode >> 13) & 3;
    let f = if form.double {
        f64::from_bits(raw)
    } else {
        f32::from_bits(raw as u32) as f64
    };
    let r = match if form.trunc { 3 } else { rc } {
        0 => f.round_ties_even(),
        1 => f.floor(),
        2 => f.ceil(),
        _ => f.trunc(),
    };
    let (lo, hi, indefinite) = match (form.signed, form.bits) {
        (true, 32) => (-2147483648.0, 2147483648.0, 0x80000000),
        (true, _) => (
            -9223372036854775808.0,
            9223372036854775808.0,
            0x8000000000000000,
        ),
        (false, 32) => (0.0, 4294967296.0, 0xffffffff),
        (false, _) => (0.0, 18446744073709551616.0, u64::MAX),
    };
    if !(r >= lo && r < hi) {
        return indefinite;
    }
    if form.signed {
        (r as i64 as u64)
            & if form.bits == 32 {
                u32::MAX as u64
            } else {
                u64::MAX
            }
    } else {
        r as u64
    }
}

fn fixture(form: &Form, mode: u32, memory: bool) -> (VectorInput, Vec<u8>) {
    let mut a = CodeAssembler::new(64).unwrap();
    let mut body = a.create_label();
    a.mov(ecx, 2).unwrap();
    a.jmp(body).unwrap();
    a.set_label(&mut body).unwrap();
    let mut data = vec![0; 4096];
    let mut result = vec![0; 4096];
    let values = operands(form.double);
    let lanes = if form.scalar {
        1
    } else if form.double {
        if form.wide {
            4
        } else {
            2
        }
    } else if form.wide {
        8
    } else {
        4
    };
    let step = if form.double { 8 } else { 4 };
    for case in 0..values.len() {
        let offset = case * 32;
        for lane in 0..lanes {
            let raw = values[(case + lane) % values.len()];
            data[offset + lane * step..offset + (lane + 1) * step]
                .copy_from_slice(&raw.to_le_bytes()[..step]);
            let v = expected(raw, form, mode).to_le_bytes();
            let size = if form.scalar { 8 } else { 4 };
            result[offset + lane * size..offset + (lane + 1) * size].copy_from_slice(&v[..size]);
        }
        let src_reg = if form.wide {
            Register::YMM1
        } else {
            Register::XMM1
        };
        if !memory {
            if form.wide {
                a.vmovups(ymm1, ymmword_ptr(rbx + offset as i32)).unwrap();
            } else {
                a.movups(xmm1, xmmword_ptr(rbx + offset as i32)).unwrap();
            }
        }
        let dst = if form.scalar {
            if form.bits == 64 {
                Register::RAX
            } else {
                Register::EAX
            }
        } else if form.wide && !form.double {
            Register::YMM0
        } else {
            Register::XMM0
        };
        if form.scalar {
            a.mov(rax, u64::MAX).unwrap();
        }
        let insn = if memory {
            Instruction::with2(
                form.code,
                dst,
                MemoryOperand::with_base_displ(Register::RBX, offset as i64),
            )
            .unwrap()
        } else {
            Instruction::with2(form.code, dst, src_reg).unwrap()
        };
        a.add_instruction(insn).unwrap();
        if form.scalar {
            a.mov(qword_ptr(rdi + offset as i32), rax).unwrap();
        } else if form.wide && !form.double {
            a.vmovups(ymmword_ptr(rdi + offset as i32), ymm0).unwrap();
        } else {
            a.movups(xmmword_ptr(rdi + offset as i32), xmm0).unwrap();
        }
    }
    a.dec(ecx).unwrap();
    a.jnz(body).unwrap();
    a.hlt().unwrap();
    let mut cpu_init = CpuSnapshot {
        rip: ENTRY,
        mxcsr: mode,
        ..Default::default()
    };
    cpu_init.gpr[3] = DATA;
    cpu_init.gpr[7] = OUTPUT;
    let input = VectorInput {
        cpu_init,
        entry: ENTRY,
        run: RunSpec::UntilExit,
        mem_init: vec![
            MemChunk {
                addr: ENTRY,
                bytes: a.assemble(ENTRY).unwrap(),
                kind: MemKind::Ram,
            },
            MemChunk {
                addr: DATA,
                bytes: data,
                kind: MemKind::Ram,
            },
            MemChunk {
                addr: OUTPUT,
                bytes: vec![0; 4096],
                kind: MemKind::Ram,
            },
        ],
    };
    (input, result)
}

fn run_tier(input: &VectorInput, tier: usize) -> Vec<u8> {
    let backend: Box<dyn Backend> = match tier {
        0 => Box::new(InterpreterBackend),
        1 => Box::new(JitBackend::new()),
        _ => Box::new(JitBackend::with_superblocks(RegionCaps {
            max_blocks: 8,
            max_icount: 256,
        })),
    };
    let mut vm = Vm::with_backend(VmConfig::flat(0x50000), backend);
    vm.set_guest_cpu_features(GuestCpuFeatures::v4());
    for m in &input.mem_init {
        vm.map(m.addr, m.bytes.len(), Prot::RWX, RegionKind::Ram)
            .unwrap();
        vm.write_bytes(m.addr, &m.bytes).unwrap();
    }
    let mut cpu = vm.new_vcpu_with_mxcsr(input.cpu_init.mxcsr);
    cpu.set_reg(Reg::Rip, ENTRY);
    cpu.set_reg(Reg::Rbx, DATA);
    cpu.set_reg(Reg::Rdi, OUTPUT);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    if tier > 0 {
        assert!(
            matches!(
                vm.cache.get(x86jit_core::BlockKey::with_mxcsr(
                    ENTRY,
                    x86jit_core::CpuMode::Long64,
                    input.cpu_init.mxcsr
                )),
                Some(x86jit_core::CachedBlock::Compiled { .. })
            ),
            "must execute compiled code"
        );
    }
    if tier == 2 {
        assert!(vm.cache.regions() > 0, "must execute a compiled region");
    }
    let mut out = vec![0; 4096];
    vm.read_bytes(OUTPUT, &mut out).unwrap();
    out
}

fn check_bytes(actual: &[u8], expected: &[u8], context: &str) {
    for (case, (actual, expected)) in actual.chunks(32).zip(expected.chunks(32)).enumerate() {
        assert_eq!(actual, expected, "{context} case={case}");
    }
}

fn witnesses(scalar: bool, tier: usize) {
    let forms = forms(scalar);
    assert_eq!(
        forms.len(),
        if scalar { 32 } else { 20 },
        "lifted encoding coverage"
    );
    for form in forms {
        for mode in (0..4).flat_map(|rc| [0x1f80 | rc << 13, 0x9fc0 | rc << 13]) {
            for memory in [false, true] {
                let (input, expected) = fixture(&form, mode, memory);
                check_bytes(
                    &run_tier(&input, tier),
                    &expected,
                    &format!("{form:?} mode={mode:#x} memory={memory} tier={tier}"),
                );
            }
        }
    }
}

#[test]
fn cvt_indefinite_scalar_interpreter() {
    witnesses(true, 0);
}

#[test]
fn cvt_indefinite_scalar_block() {
    witnesses(true, 1);
}

#[test]
fn cvt_indefinite_scalar_region() {
    witnesses(true, 2);
}

#[test]
fn cvt_indefinite_packed_interpreter() {
    witnesses(false, 0);
}

#[test]
fn cvt_indefinite_packed_block() {
    witnesses(false, 1);
}

#[test]
fn cvt_indefinite_packed_region() {
    witnesses(false, 2);
}

#[cfg(feature = "unicorn")]
#[test]
fn cvt_indefinite_matches_unicorn() {
    use x86jit_tests::oracle::Oracle;
    for scalar in [true, false] {
        for form in forms(scalar)
            .into_iter()
            .filter(|f| f.code.encoding() == EncodingKind::Legacy)
        {
            for mode in (0..4).flat_map(|rc| [0x1f80 | rc << 13, 0x9fc0 | rc << 13]) {
                for memory in [false, true] {
                    let (input, expected) = fixture(&form, mode, memory);
                    let oracle = x86jit_tests::unicorn::UnicornOracle.run(&input);
                    check_bytes(
                        &oracle.mem[2].bytes,
                        &expected,
                        &format!("{form:?} mode={mode:#x} memory={memory}"),
                    );
                    for tier in 0..3 {
                        check_bytes(
                            &run_tier(&input, tier),
                            &oracle.mem[2].bytes,
                            &format!("{form:?} tier={tier}"),
                        );
                    }
                }
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn cvt_indefinite_matches_native() {
    for scalar in [true, false] {
        for form in forms(scalar) {
            match form.code.encoding() {
                EncodingKind::VEX if !std::is_x86_feature_detected!("avx") => continue,
                EncodingKind::EVEX
                    if !std::is_x86_feature_detected!("avx512f")
                        || (!form.scalar && !std::is_x86_feature_detected!("avx512vl")) =>
                {
                    continue
                }
                _ => {}
            }
            for mode in (0..4).flat_map(|rc| [0x1f80 | rc << 13, 0x9fc0 | rc << 13]) {
                for memory in [false, true] {
                    let (input, expected) = fixture(&form, mode, memory);
                    let oracle = x86jit_tests::native::run_native(&input)
                        .expect("advertised host ISA must execute");
                    check_bytes(
                        &oracle.mem[2].bytes,
                        &expected,
                        &format!("{form:?} mode={mode:#x} memory={memory}"),
                    );
                    for tier in 0..3 {
                        check_bytes(
                            &run_tier(&input, tier),
                            &oracle.mem[2].bytes,
                            &format!("{form:?} tier={tier}"),
                        );
                    }
                }
            }
        }
    }
}
