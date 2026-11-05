//! MXCSR numerical controls: SDM Vol 1 §§10.2.3.2–10.2.3.4.

use iced_x86::code_asm::*;
use x86jit_core::{Backend, Exit, InterpreterBackend, Prot, Reg, RegionKind, Vm, VmConfig};
use x86jit_cranelift::JitBackend;

fn scalar(mode: u32, x: u32, y: u32, backend: Box<dyn Backend>) -> u32 {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    a.addss(xmm0, xmm1).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
        .unwrap();
    let mut cpu = vm.new_vcpu_with_mxcsr(mode);
    cpu.set_reg(Reg::Rip, 0x1000);
    cpu.cpu.xmm[0] = x as u128;
    cpu.cpu.xmm[1] = y as u128;
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    cpu.cpu.xmm[0] as u32
}

#[test]
fn mxcsr_modes_add_rounding_witness() {
    // SDM Vol 1 §10.2.3.2 and §4.8.4: 1 + half an ulp, positive.
    for rc in 0..4 {
        for backend in [
            Box::new(InterpreterBackend) as Box<dyn Backend>,
            Box::new(JitBackend::new()),
        ] {
            assert_eq!(
                scalar(0x1f80 | rc << 13, 0x3f800000, 0x33800000, backend),
                if rc == 2 { 0x3f800001 } else { 0x3f800000 },
                "RC={rc}"
            );
        }
    }
}

#[test]
fn mxcsr_modes_daz_witness() {
    // SDM Vol 1 §10.2.3.4: a subnormal source becomes a signed zero.
    for backend in [
        Box::new(InterpreterBackend) as Box<dyn Backend>,
        Box::new(JitBackend::new()),
    ] {
        assert_eq!(scalar(0x1fc0, 1, 0, backend), 0);
    }
}

use x86jit_tests::oracle::{run_with_backend, VectorInput};
use x86jit_tests::vector::{CpuSnapshot, MemChunk, MemKind, RunSpec};

const ENTRY: u64 = 0x210000;

fn input(mode: u32, build: impl FnOnce(&mut CodeAssembler), x: u128, y: u128) -> VectorInput {
    let mut a = CodeAssembler::new(64).unwrap();
    build(&mut a);
    a.hlt().unwrap();
    let mut cpu_init = CpuSnapshot {
        mxcsr: mode,
        rip: ENTRY,
        ..Default::default()
    };
    cpu_init.xmm[0] = x;
    cpu_init.xmm[1] = y;
    cpu_init.xmm[2] = x;
    cpu_init.ymm_hi[0] = x;
    cpu_init.ymm_hi[1] = y;
    cpu_init.ymm_hi[2] = x;
    cpu_init.gpr[0] = 0x0123_4567_89ab_cdef;
    VectorInput {
        cpu_init,
        entry: ENTRY,
        run: RunSpec::UntilExit,
        mem_init: vec![MemChunk {
            addr: ENTRY,
            bytes: a.assemble(ENTRY).unwrap(),
            kind: MemKind::Ram,
        }],
    }
}

fn tiers(input: &VectorInput, name: &str) -> CpuSnapshot {
    let interpreted = run_with_backend(input, Box::new(InterpreterBackend));
    assert_eq!(
        interpreted.exit,
        x86jit_tests::vector::ExitKind::Hlt,
        "{name} must execute through HLT"
    );
    let compiled = run_with_backend(input, Box::new(JitBackend::new()));
    let baseline = run_with_backend(
        input,
        Box::new(JitBackend::with_options(
            None,
            x86jit_cranelift::HostTarget::Baseline,
            x86jit_cranelift::OptLevel::None,
        )),
    );
    assert_eq!(
        baseline.cpu.xmm, interpreted.cpu.xmm,
        "baseline {name} mode={:#x}",
        input.cpu_init.mxcsr
    );
    assert_eq!(
        baseline.cpu.ymm_hi, interpreted.cpu.ymm_hi,
        "baseline {name} mode={:#x}",
        input.cpu_init.mxcsr
    );
    assert_eq!(
        baseline.cpu.gpr, interpreted.cpu.gpr,
        "baseline {name} mode={:#x}",
        input.cpu_init.mxcsr
    );

    assert_eq!(
        compiled.exit, interpreted.exit,
        "{name} mode={:#x}",
        input.cpu_init.mxcsr
    );
    assert_eq!(
        compiled.cpu.xmm, interpreted.cpu.xmm,
        "{name} mode={:#x}",
        input.cpu_init.mxcsr
    );
    assert_eq!(
        compiled.cpu.ymm_hi, interpreted.cpu.ymm_hi,
        "{name} mode={:#x}",
        input.cpu_init.mxcsr
    );
    assert_eq!(
        compiled.cpu.gpr, interpreted.cpu.gpr,
        "{name} mode={:#x}",
        input.cpu_init.mxcsr
    );
    interpreted.cpu
}

fn arithmetic(a: &mut CodeAssembler, op: usize, double: bool, packed: bool, wide: bool) {
    if wide {
        match (op, double) {
            (0, false) => a.vaddps(ymm0, ymm0, ymm1).unwrap(),
            (1, false) => a.vsubps(ymm0, ymm0, ymm1).unwrap(),
            (2, false) => a.vmulps(ymm0, ymm0, ymm1).unwrap(),
            (3, false) => a.vdivps(ymm0, ymm0, ymm1).unwrap(),
            (4, false) => a.vsqrtps(ymm0, ymm0).unwrap(),
            (0, true) => a.vaddpd(ymm0, ymm0, ymm1).unwrap(),
            (1, true) => a.vsubpd(ymm0, ymm0, ymm1).unwrap(),
            (2, true) => a.vmulpd(ymm0, ymm0, ymm1).unwrap(),
            (3, true) => a.vdivpd(ymm0, ymm0, ymm1).unwrap(),
            (4, true) => a.vsqrtpd(ymm0, ymm0).unwrap(),
            _ => unreachable!(),
        }
    } else {
        match (op, double, packed) {
            (0, false, false) => a.addss(xmm0, xmm1).unwrap(),
            (1, false, false) => a.subss(xmm0, xmm1).unwrap(),
            (2, false, false) => a.mulss(xmm0, xmm1).unwrap(),
            (3, false, false) => a.divss(xmm0, xmm1).unwrap(),
            (4, false, false) => a.sqrtss(xmm0, xmm0).unwrap(),
            (0, true, false) => a.addsd(xmm0, xmm1).unwrap(),
            (1, true, false) => a.subsd(xmm0, xmm1).unwrap(),
            (2, true, false) => a.mulsd(xmm0, xmm1).unwrap(),
            (3, true, false) => a.divsd(xmm0, xmm1).unwrap(),
            (4, true, false) => a.sqrtsd(xmm0, xmm0).unwrap(),
            (0, false, true) => a.addps(xmm0, xmm1).unwrap(),
            (1, false, true) => a.subps(xmm0, xmm1).unwrap(),
            (2, false, true) => a.mulps(xmm0, xmm1).unwrap(),
            (3, false, true) => a.divps(xmm0, xmm1).unwrap(),
            (4, false, true) => a.sqrtps(xmm0, xmm0).unwrap(),
            (0, true, true) => a.addpd(xmm0, xmm1).unwrap(),
            (1, true, true) => a.subpd(xmm0, xmm1).unwrap(),
            (2, true, true) => a.mulpd(xmm0, xmm1).unwrap(),
            (3, true, true) => a.divpd(xmm0, xmm1).unwrap(),
            (4, true, true) => a.sqrtpd(xmm0, xmm0).unwrap(),
            _ => unreachable!(),
        }
    }
}

fn broadcast(x: u64, double: bool) -> u128 {
    if double {
        x as u128 | (x as u128) << 64
    } else {
        let x = x as u32 as u128;
        x | x << 32 | x << 64 | x << 96
    }
}

fn ftz_witness(op: usize, double: bool, x: u64, y: u64, mode: u32) -> Option<u64> {
    if mode & 0x8000 == 0 {
        return None;
    }
    let (min, one, sign) = if double {
        (0x0010000000000000, 0x3ff0000000000000, 1 << 63)
    } else {
        (0x00800000, 0x3f800000, 1 << 31)
    };
    // SDM Vol 1 §§10.2.3.3, 11.5.2.5: classify tininess after rounding
    // at destination precision with an unbounded exponent. Let m=min-normal,
    // u=min-subnormal and p=24/53. u-m and u*m are tiny; ±m*(1-2^-p)
    // is exactly representable with an unbounded exponent and remains tiny.
    // (m+u)*(1-2^(1-p)) = m*(1-2^(2-2p)) rounds up to m in RN/RU.
    match (op, x, y) {
        (1, 1, y) if y == min && mode & 0x40 == 0 => Some(sign),
        (2, 1, y) if y == min && mode & 0x40 == 0 => Some(0),
        (2, x, y) if x == min && y == one - 1 => Some(0),
        (2, x, y) if x == min + 1 && y == one - 2 => Some(if matches!((mode >> 13) & 3, 0 | 2) {
            min
        } else {
            0
        }),
        (2, x, y) if x == sign | min && y == one - 1 => Some(sign),
        _ => None,
    }
}

fn arithmetic_cases(mut check: impl FnMut(&VectorInput, &str, Option<u128>)) {
    for rc in 0..4 {
        for flush in [0, 0x40, 0x8000, 0x8040] {
            let mode = 0x1f80 | rc << 13 | flush;
            for double in [false, true] {
                // SDM Vol 1 §§4.8.4, 10.2.3.2–10.2.3.4: inexact,
                // denormal inputs, tiny outputs, and the normal/subnormal boundary.
                let pairs: &[(u64, u64)] = if double {
                    &[
                        (0x3ff0000000000001, 0x3ff0000000000002),
                        (1, 0x0010000000000000),
                        (0x0010000000000000, 0x3fefffffffffffff),
                        (0x0010000000000001, 0x3feffffffffffffe),
                        (0x8010000000000000, 0x3fefffffffffffff),
                    ]
                } else {
                    &[
                        (0x3f800001, 0x3f800002),
                        (1, 0x00800000),
                        (0x00800000, 0x3f7fffff),
                        (0x00800001, 0x3f7ffffe),
                        (0x80800000, 0x3f7fffff),
                    ]
                };
                for &(x, y) in pairs {
                    for op in 0..5 {
                        if op == 4 && ((double && x >> 63 != 0) || (!double && x >> 31 != 0)) {
                            continue;
                        }
                        for (packed, wide) in [(false, false), (true, false), (true, true)] {
                            let vector = input(
                                mode,
                                |a| arithmetic(a, op, double, packed, wide),
                                broadcast(x, double),
                                broadcast(y, double),
                            );
                            let witness = ftz_witness(op, double, x, y, mode).map(|bits| {
                                if packed {
                                    broadcast(bits, double)
                                } else {
                                    let mask = if double {
                                        u64::MAX as u128
                                    } else {
                                        u32::MAX as u128
                                    };
                                    (vector.cpu_init.xmm[0] & !mask) | bits as u128
                                }
                            });
                            check(&vector, &format!("op={op} f64={double} packed={packed} wide={wide} x={x:x} y={y:x}"), witness);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn mxcsr_modes_arithmetic_all_tiers() {
    arithmetic_cases(|v, n, witness| {
        let actual = tiers(v, n);
        if let Some(bits) = witness {
            assert_eq!(actual.xmm[0], bits, "{n} mode={:#x}", v.cpu_init.mxcsr);
        }
    });
}

#[test]
fn mxcsr_modes_ftz_after_rounding_witness() {
    // SDM Vol 1 §10.2.3.3, §11.5.2.5: tininess after rounding.
    // (min-normal + min-subnormal) * (1 - 2^-23) rounds to min-normal
    // with an unbounded exponent in RN/RU.
    for rc in 0..4 {
        for flush in [0x8000, 0x8040] {
            let v = input(
                0x1f80 | flush | rc << 13,
                |a| a.mulss(xmm0, xmm1).unwrap(),
                0x00800001,
                0x3f7ffffe,
            );
            assert_eq!(
                tiers(&v, "after-rounding").xmm[0] as u32,
                if rc == 0 || rc == 2 { 0x00800000 } else { 0 }
            );
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn mxcsr_modes_arithmetic_matches_native() {
    if !std::is_x86_feature_detected!("avx") {
        return;
    }
    arithmetic_cases(|v, n, _| {
        let expected = x86jit_tests::native::run_native(v).expect("SSE/AVX host");
        let actual = tiers(v, n);
        assert_eq!(
            expected.cpu.xmm, actual.xmm,
            "{n} mode={:#x}",
            v.cpu_init.mxcsr
        );
        assert_eq!(
            expected.cpu.ymm_hi, actual.ymm_hi,
            "{n} mode={:#x}",
            v.cpu_init.mxcsr
        );
    });
}

#[cfg(feature = "unicorn")]
#[test]
fn mxcsr_modes_sse_matches_unicorn() {
    use x86jit_tests::oracle::Oracle;
    let mut mismatches = Vec::new();
    arithmetic_cases(|v, n, witness| {
        // Unicorn's x86 decoder does not support these VEX arithmetic forms.
        if n.contains("wide=true") {
            return;
        }
        let mut expected = x86jit_tests::unicorn::UnicornOracle.run(v);
        let actual = tiers(v, n);
        if let Some(bits) = witness {
            assert_eq!(actual.xmm[0], bits, "{n} mode={:#x}", v.cpu_init.mxcsr);
            // SDM Vol 1 §§10.2.3.3, 11.5.2.5: masked FTZ returns signed
            // zero for tiny results, with tininess after unbounded-exponent
            // rounding. Unicorn disagrees for this class: SUBSS x=0x00000001,
            // y=0x00800000, MXCSR=0x9f80 gives 0x807fffff; the engine gives
            // 0x80000000. Only the independently witnessed result is excluded;
            // mxcsr_modes_arithmetic_matches_native is the hardware check.
            if expected.cpu.xmm[0] != bits {
                eprintln!(
                    "Unicorn FTZ divergence: {n} mode={:#x} unicorn={:#034x} engine={bits:#034x}",
                    v.cpu_init.mxcsr, expected.cpu.xmm[0]
                );
                expected.cpu.xmm[0] = bits;
            }
        }
        if expected.cpu.xmm != actual.xmm {
            mismatches.push(format!(
                "{n} mode={:#x} unicorn={:x?} engine={:x?}",
                v.cpu_init.mxcsr, expected.cpu.xmm, actual.xmm
            ));
        }
    });
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

fn conversion_cases(mut check: impl FnMut(&VectorInput, &str)) {
    for rc in 0..4 {
        for flush in [0, 0x40, 0x8000, 0x8040] {
            let mode = 0x1f80 | rc << 13 | flush;
            for op in 0..12 {
                for x in [
                    1u64,
                    0x3ff0000010000000,
                    0xbff8000000000000,
                    0x380fffffe0000000,
                ] {
                    // SDM Vol 2A §3.3, CVTSI2SS/CVTSD2SS/CVTSD2SI/CVTPS2DQ;
                    // SDM Vol 2B §4.3 ROUNDPS; Vol 2C §5.2 VCVTPS2PH, Table 5-3.
                    let v = input(
                        mode,
                        |a| match op {
                            0 => a.cvtsi2ss(xmm0, rax).unwrap(),
                            1 => a.cvtsi2sd(xmm0, rax).unwrap(),
                            2 => a.cvtsd2ss(xmm0, xmm1).unwrap(),
                            3 => a.cvtsd2si(rax, xmm1).unwrap(),
                            4 => a.cvtps2dq(xmm0, xmm1).unwrap(),
                            5 => a.roundps(xmm0, xmm1, 4).unwrap(),
                            6 => a.vcvtps2ph(xmm0, xmm1, 4).unwrap(),
                            7 => a.cvtpd2ps(xmm0, xmm1).unwrap(),
                            8 => a.vcvtpd2ps(xmm0, ymm1).unwrap(),
                            9 => a.vcvtps2pd(ymm0, xmm1).unwrap(),
                            10 => a.vcvtps2dq(ymm0, ymm1).unwrap(),
                            11 => a.cvtdq2ps(xmm0, xmm1).unwrap(),
                            _ => unreachable!(),
                        },
                        x as u128 | (x as u128) << 64,
                        x as u128 | (x as u128) << 64,
                    );
                    check(&v, &format!("conversion={op} x={x:x}"));
                }
            }
        }
    }
}

#[test]
fn mxcsr_modes_conversions_all_tiers() {
    conversion_cases(|v, n| {
        tiers(v, n);
    });
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn mxcsr_modes_conversions_match_native() {
    if !std::is_x86_feature_detected!("avx") || !std::is_x86_feature_detected!("f16c") {
        return;
    }
    conversion_cases(|v, n| {
        let expected = x86jit_tests::native::run_native(v).expect("SSE/AVX/F16C host");
        let actual = tiers(v, n);
        assert_eq!(
            expected.cpu.xmm, actual.xmm,
            "{n} mode={:#x}",
            v.cpu_init.mxcsr
        );
        assert_eq!(
            expected.cpu.gpr, actual.gpr,
            "{n} mode={:#x}",
            v.cpu_init.mxcsr
        );
    });
}

#[test]
fn mxcsr_modes_helpers_and_f16c_witnesses() {
    for rc in 0..4 {
        for flush in [0, 0x40, 0x8000, 0x8040] {
            let mode = 0x1f80 | rc << 13 | flush;
            for op in 0..4 {
                for &(x, y) in &[
                    (1, 0x3f800000),
                    (0x3f800001, 0x3f800001),
                    (0x00800000, 0x3f000000),
                ] {
                    let v = input(
                        mode,
                        |a| match op {
                            0 => a.vfmadd132ps(xmm0, xmm2, xmm1).unwrap(),
                            1 => a.haddps(xmm0, xmm1).unwrap(),
                            2 => a.dpps(xmm0, xmm1, 0xff).unwrap(),
                            3 => a.hsubps(xmm0, xmm1).unwrap(),
                            _ => unreachable!(),
                        },
                        broadcast(x, false),
                        broadcast(y, false),
                    );
                    tiers(&v, &format!("helper={op} x={x:x}"));
                }
            }
            // SDM Vol 2C §5.2 VCVTPS2PH: FTZ is ignored; imm[2] selects RC.
            let v = input(
                mode,
                |a| a.vcvtps2ph(xmm0, xmm1, 4).unwrap(),
                0,
                broadcast(0x33800000, false),
            );
            assert_eq!(tiers(&v, "F16C FTZ override").xmm[0] as u16, 1);
            // SDM Vol 2C §5.2 VCVTPH2PS: DAZ is ignored, even for half subnormals.
            let v = input(mode, |a| a.vcvtph2ps(xmm0, xmm1).unwrap(), 0, 1);
            assert_eq!(tiers(&v, "F16C DAZ override").xmm[0] as u32, 0x33800000);
        }
    }
}

fn fp_control() -> u64 {
    let value;
    // SAFETY: read-only access to this thread's FP control register.
    unsafe {
        #[cfg(target_arch = "aarch64")]
        core::arch::asm!("mrs {}, fpcr", out(reg) value, options(nomem,nostack));
        #[cfg(target_arch = "x86_64")]
        {
            let mut v = 0u32;
            core::arch::asm!("stmxcsr [{}]", in(reg) &mut v, options(nostack));
            value = u64::from(v);
        }
    }
    value
}

#[derive(Debug)]
struct ObserveHelper(Option<x86jit_core::ExitRequest>);
impl x86jit_core::CpuSource for ObserveHelper {
    fn read_tsc(&self, _: u64) -> x86jit_core::TscSample {
        assert_eq!(
            fp_control(),
            x86jit_core::jit_abi::fp::default_control(),
            "Rust helper environment"
        );
        if let Some(request) = &self.0 {
            request.request_exit();
        }
        x86jit_core::TscSample { counter: 7, aux: 0 }
    }
    fn read_random(&self, _: u64, _: x86jit_core::RandomKind, _: u8) -> Option<u64> {
        None
    }
}

#[test]
fn mxcsr_modes_restore_host_and_default_helpers_on_exits() {
    for mode in [0x9fc0, 0x3f80, 0x5f80, 0x7f80] {
        for terminal in 0..10 {
            let mut vm = Vm::with_backend(VmConfig::flat(0x20000), Box::new(JitBackend::new()));
            vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.map(0x18000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
            let mut a = CodeAssembler::new(64).unwrap();
            a.rdtsc().unwrap();
            a.addss(xmm0, xmm1).unwrap();
            match terminal {
                0 => a.hlt().unwrap(),
                1 => a.syscall().unwrap(),
                2 => a.ud2().unwrap(),
                3 => a.mov(rax, qword_ptr(0x20000u64)).unwrap(),
                4 => a.in_(al, dx).unwrap(),
                5 => a.xsave(ptr(0x18000u64)).unwrap(),
                6 => a.mov(rax, qword_ptr(0x18000u64)).unwrap(),
                7 => a.mov(qword_ptr(0x18000u64), rax).unwrap(),
                8 | 9 => a.jmp(0x1000u64).unwrap(),
                _ => unreachable!(),
            }
            vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
                .unwrap();
            let mut cpu = vm.new_vcpu_with_mxcsr(mode);
            cpu.cpu.cpu_source = Some(std::sync::Arc::new(ObserveHelper(if terminal == 9 {
                Some(cpu.exit_request())
            } else {
                None
            })));
            cpu.set_reg(Reg::Rip, 0x1000);
            let before = fp_control();
            let exit = cpu.run(&vm, Some(10));
            assert!(
                match terminal {
                    0 => matches!(exit, Exit::Hlt),
                    1 => matches!(exit, Exit::Syscall),
                    2 => matches!(exit, Exit::Exception { .. }),
                    3 => matches!(exit, Exit::UnmappedMemory { .. }),
                    4 => matches!(exit, Exit::PortIo { .. }),
                    5 => matches!(exit, Exit::UnknownInstruction { .. }),
                    6 => matches!(exit, Exit::MmioRead { .. }),
                    7 => matches!(exit, Exit::MmioWrite { .. }),
                    8 => matches!(exit, Exit::BudgetExhausted),
                    9 => matches!(exit, Exit::Interrupted),
                    _ => false,
                },
                "terminal={terminal} exit={exit:?}"
            );
            assert_eq!(fp_control(), before, "terminal={terminal} mode={mode:x}");
        }
    }
}

#[test]
fn mxcsr_modes_cache_identity_and_mode_write_barrier() {
    // SDM Vol 2B §4.3 ROUNDSS: imm8[2] selects MXCSR.RC.
    for region in [false, true] {
        let backend = if region {
            JitBackend::with_superblocks(x86jit_core::RegionCaps {
                max_blocks: 8,
                max_icount: 128,
            })
        } else {
            JitBackend::new()
        };
        let mut vm = Vm::with_backend(VmConfig::flat(0x10000), Box::new(backend));
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        let mut a = CodeAssembler::new(64).unwrap();
        a.roundss(xmm0, xmm1, 4).unwrap();
        a.ldmxcsr(dword_ptr(0x8000u64)).unwrap();
        a.roundss(xmm2, xmm1, 4).unwrap();
        a.hlt().unwrap();
        vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
            .unwrap();
        vm.write_bytes(0x8000, &0x5f80u32.to_le_bytes()).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, 0x1000);
        cpu.cpu.xmm[0] = 0x3f800000;
        cpu.cpu.xmm[1] = 0x3fa00000;
        cpu.cpu.xmm[2] = 0x3f800000;
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.cpu.xmm[0] as u32, 0x3f800000);
        assert_eq!(cpu.cpu.xmm[2] as u32, 0x40000000);
        let ir = x86jit_core::lift::lift_block(
            &vm.mem,
            x86jit_core::lift::FetchAddr::flat(0x1000),
            x86jit_core::CpuMode::Long64,
        )
        .unwrap();
        assert_eq!(ir.icount, 2, "mode write must end the block");
        let lifted = x86jit_core::lift::lift_region(
            &vm.mem,
            0x1000,
            x86jit_core::RegionCaps {
                max_blocks: 8,
                max_icount: 128,
            },
            x86jit_core::CpuMode::Long64,
        )
        .unwrap();
        assert_eq!(lifted.blocks.len(), 1, "region must stop at the mode write");
        assert_eq!(
            vm.cache.chained(),
            0,
            "mode write must dispatch, never chain"
        );
        for mode in [0x1f80, 0x3f80, 0x5f80, 0x1f80] {
            cpu.set_mxcsr(mode);
            cpu.set_reg(Reg::Rip, 0x1000);
            cpu.cpu.xmm[0] = 0x3f800000;
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            assert_eq!(
                cpu.cpu.xmm[0] as u32,
                if mode == 0x5f80 {
                    0x40000000
                } else {
                    0x3f800000
                }
            );
            assert!(vm
                .cache
                .get(x86jit_core::BlockKey::with_mxcsr(
                    0x1000,
                    x86jit_core::CpuMode::Long64,
                    mode
                ))
                .is_some());
        }
    }
}

#[test]
fn mxcsr_modes_block_region_and_background_tier_up_are_bit_identical() {
    use x86jit_core::{BlockKey, CachedBlock, CpuMode, RegionCaps};
    let caps = RegionCaps {
        max_blocks: 8,
        max_icount: 128,
    };
    for mode in [0x9fc0, 0x1f80, 0x3f80, 0x5f80, 0x7f80] {
        let mut reference = None;
        for tier in 0..4 {
            let jit = if tier >= 2 {
                JitBackend::with_superblocks(caps)
            } else {
                JitBackend::new()
            };
            let handle = jit.tier_up_handle();
            let backend: Box<dyn Backend> = if tier == 0 {
                Box::new(InterpreterBackend)
            } else {
                Box::new(jit)
            };
            let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
            vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
            if tier == 3 {
                vm.set_tier_up_after(Some(2));
                vm.set_tier_up_region_after(Some(4));
                vm.set_tier_up_background(true);
            }
            let mut a = CodeAssembler::new(64).unwrap();
            let mut top = a.create_label();
            let mut tail = a.create_label();
            a.set_label(&mut top).unwrap();
            a.movaps(xmm0, xmm2).unwrap();
            a.mulps(xmm0, xmm1).unwrap();
            a.addps(xmm0, xmm3).unwrap();
            a.jmp(tail).unwrap();
            a.set_label(&mut tail).unwrap();
            a.nop().unwrap();
            a.jmp(top).unwrap();
            vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
                .unwrap();
            let mut cpu = vm.new_vcpu_with_mxcsr(mode);
            cpu.set_reg(Reg::Rip, 0x1000);
            cpu.cpu.xmm[2] = 0x00800001_80800000_00800000_00000001;
            cpu.cpu.xmm[1] = 0x3f7ffffe_3f000000_3f000000_3f800000;
            cpu.cpu.xmm[3] = 0x00000000_80000001_00000001_00000000;
            for iteration in 0..32 {
                assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
                let bits = cpu.cpu.xmm[0];
                let expected = *reference.get_or_insert(bits);
                assert_eq!(
                    bits, expected,
                    "tier={tier} iteration={iteration} mode={mode:x}"
                );
                if tier == 3 && iteration == 15 {
                    handle.wait_idle();
                }
            }
            if tier >= 2 {
                assert!(vm.cache.regions() > 0, "tier={tier} must form a region");
            }
            if tier > 0 {
                assert!(matches!(
                    vm.cache
                        .get(BlockKey::with_mxcsr(0x1000, CpuMode::Long64, mode)),
                    Some(CachedBlock::Compiled { .. })
                ));
            }
        }
    }
}

#[test]
fn mxcsr_modes_estimates_minmax_and_comparisons() {
    for rc in 0..4 {
        for flush in [0, 0x40, 0x8000, 0x8040] {
            for op in 0..5 {
                for &(x, y) in &[
                    (1, 0),
                    (0x80000001, 0),
                    (0x3f800000, 0xbf800000),
                    (0x7f7fffff, 0x3f800000),
                ] {
                    // Preserve the existing default-mode negative-estimate NaN path.
                    if rc == 0 && flush == 0 && op == 1 && y & 0x80000000 != 0 {
                        continue;
                    }
                    let v = input(
                        0x1f80 | rc << 13 | flush,
                        |a| match op {
                            0 => a.vrcpps(xmm0, xmm1).unwrap(),
                            1 => a.vrsqrtps(xmm0, xmm1).unwrap(),
                            2 => a.minps(xmm0, xmm1).unwrap(),
                            3 => a.maxps(xmm0, xmm1).unwrap(),
                            4 => a.cmpps(xmm0, xmm1, 0).unwrap(),
                            _ => unreachable!(),
                        },
                        broadcast(x, false),
                        broadcast(y, false),
                    );
                    tiers(&v, &format!("estimate/minmax/cmp={op} x={x:x} y={y:x}"));
                }
            }
        }
    }
}

#[test]
fn mxcsr_modes_non_afp_fallback_is_only_at_tininess_boundary() {
    use x86jit_cranelift::{HostTarget, OptLevel};
    // SDM Vol 1 §11.5.2.5: unbounded-exponent rounding distinguishes these
    // from a normal result that happens to round to min-normal as a subnormal.
    for &(x, y, boundary) in &[
        (0x3f800000u32, 0x3f800000u32, false),
        (0x00800000, 0x3f000000, false),
        (0x00800000, 0x3f7fffff, true),
        (0x00800001, 0x3f7ffffe, true),
    ] {
        let mut vm = Vm::with_backend(
            VmConfig::flat(0x10000),
            Box::new(JitBackend::with_options(
                None,
                HostTarget::Baseline,
                OptLevel::None,
            )),
        );
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        let mut a = CodeAssembler::new(64).unwrap();
        a.mulss(xmm0, xmm1).unwrap();
        a.hlt().unwrap();
        vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
            .unwrap();
        let mut cpu = vm.new_vcpu_with_mxcsr(0x9fc0);
        cpu.set_reg(Reg::Rip, 0x1000);
        cpu.cpu.xmm[0] = x as u128;
        cpu.cpu.xmm[1] = y as u128;
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        let calls = vm
            .backend
            .helper_calls()
            .into_iter()
            .filter(|(n, _)| *n == "fp_boundary_helper")
            .map(|(_, n)| n)
            .sum::<u64>();
        if cfg!(target_arch = "aarch64") {
            assert_eq!(calls, u64::from(boundary));
        } else {
            assert_eq!(calls, 0);
        }
    }
}

#[test]
fn mxcsr_modes_fxrstor_switches_the_next_instruction() {
    for backend in [
        Box::new(InterpreterBackend) as Box<dyn Backend>,
        Box::new(JitBackend::new()),
    ] {
        let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        let mut a = CodeAssembler::new(64).unwrap();
        a.fxsave(xmmword_ptr(0x8000u64)).unwrap();
        a.ldmxcsr(dword_ptr(0x9000u64)).unwrap();
        a.fxrstor(xmmword_ptr(0x8000u64)).unwrap();
        a.addss(xmm0, xmm1).unwrap();
        a.hlt().unwrap();
        vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
            .unwrap();
        vm.write_bytes(0x9000, &0x1f80u32.to_le_bytes()).unwrap();
        let mut cpu = vm.new_vcpu_with_mxcsr(0x5f80);
        cpu.set_reg(Reg::Rip, 0x1000);
        cpu.cpu.xmm[0] = 0x3f800000;
        cpu.cpu.xmm[1] = 0x33800000;
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.cpu.xmm[0] as u32, 0x3f800001);
    }
}

#[test]
fn mxcsr_modes_compile_limit_is_shared_by_specialisations() {
    let jit = JitBackend::new();
    jit.set_compile_limit(1);
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), Box::new(jit));
    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, &[0xf4]).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    cpu.set_mxcsr(0x9fc0);
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(
        cpu.run(&vm, None),
        Exit::CodeSpaceExhausted { .. }
    ));
}

#[test]
fn mxcsr_modes_nan_payloads_and_invalid_arithmetic() {
    // SDM Vol 1 §§4.8.3.5, 4.8.3.7: first-source NaNs and indefinite.
    for mode in [0x9fc0, 0x3f80, 0x5f80, 0x7f80] {
        for double in [false, true] {
            let pairs: &[(u64, u64)] = if double {
                &[
                    (0x7ff8123456789abc, 0x7ff0000000000001),
                    (0x7ff0000000000000, 0xfff0000000000000),
                    (0, 0),
                    (0xbff0000000000000, 0x3ff0000000000000),
                ]
            } else {
                &[
                    (0x7fc01234, 0x7f800001),
                    (0x7f800000, 0xff800000),
                    (0, 0),
                    (0xbf800000, 0x3f800000),
                ]
            };
            for &(x, y) in pairs {
                for op in 0..5 {
                    let v = input(
                        mode,
                        |a| arithmetic(a, op, double, true, false),
                        broadcast(x, double),
                        broadcast(y, double),
                    );
                    tiers(&v, &format!("NaN op={op} x={x:x} y={y:x}"));
                }
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn mxcsr_modes_helpers_match_native() {
    if !std::is_x86_feature_detected!("fma") || !std::is_x86_feature_detected!("sse4.1") {
        return;
    }
    for rc in 0..4 {
        for flush in [0, 0x40, 0x8000, 0x8040] {
            for op in 0..3 {
                let v = input(
                    0x1f80 | rc << 13 | flush,
                    |a| match op {
                        0 => a.vfmadd132ps(xmm0, xmm2, xmm1).unwrap(),
                        1 => a.haddps(xmm0, xmm1).unwrap(),
                        _ => a.dpps(xmm0, xmm1, 0xff).unwrap(),
                    },
                    broadcast(0x00800001, false),
                    broadcast(0x3f7ffffe, false),
                );
                let expected = x86jit_tests::native::run_native(&v).expect("supported SIMD host");
                let actual = tiers(&v, "native helper");
                assert_eq!(
                    actual.xmm, expected.cpu.xmm,
                    "op={op} RC={rc} flush={flush:x}"
                );
            }
        }
    }
}

#[test]
fn mxcsr_modes_f16c_single_denormal_respects_daz() {
    // SDM Vol 1 §10.2.3.4; Vol 2C §5.2 VCVTPS2PH overrides FTZ, not DAZ.
    for flush in [0, 0x40, 0x8000, 0x8040] {
        let v = input(
            0x5f80 | flush,
            |a| a.vcvtps2ph(xmm0, xmm1, 4).unwrap(),
            0,
            1,
        );
        assert_eq!(
            tiers(&v, "F16C single denormal").xmm[0] as u16,
            if flush & 0x40 != 0 { 0 } else { 1 }
        );
    }
}

#[test]
fn mxcsr_modes_constant_conversion_is_not_folded_at_nearest() {
    // SDM Vol 2A §3.3 CVTSI2SS: RC applies even when the compiler knows the
    // integer input. Cranelift's speed egraph otherwise folds this at RN.
    for rc in 0..4 {
        let v = input(
            0x1f80 | rc << 13,
            |a| {
                a.mov(rax, 16_777_217u64).unwrap();
                a.cvtsi2ss(xmm0, rax).unwrap();
            },
            0,
            0,
        );
        let expected = if rc == 2 { 0x4b800001 } else { 0x4b800000 };
        for backend in [
            Box::new(InterpreterBackend) as Box<dyn Backend>,
            Box::new(JitBackend::with_opt_level(
                x86jit_cranelift::OptLevel::Speed,
            )),
        ] {
            let result = run_with_backend(&v, backend);
            assert_eq!(result.cpu.xmm[0] as u32, expected, "RC={rc}");
        }
    }
}

#[derive(Debug)]
struct ReadHostControl;
impl x86jit_core::CpuSource for ReadHostControl {
    fn read_tsc(&self, _: u64) -> x86jit_core::TscSample {
        x86jit_core::TscSample {
            counter: fp_control(),
            aux: 0,
        }
    }
    fn read_random(&self, _: u64, _: x86jit_core::RandomKind, _: u8) -> Option<u64> {
        None
    }
}

struct RestoreControl(u64);
impl Drop for RestoreControl {
    fn drop(&mut self) {
        // SAFETY: restore this thread's saved environment before leaving the test.
        unsafe { x86jit_core::jit_abi::fp::set_control(self.0) };
    }
}

fn preexisting_host_control(backend: Box<dyn Backend>) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    a.rdtsc().unwrap();
    a.hlt().unwrap();
    vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
        .unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    cpu.cpu.cpu_source = Some(std::sync::Arc::new(ReadHostControl));
    let restore = RestoreControl(fp_control());
    let external = x86jit_core::jit_abi::fp::control(0x9fc0);
    // No Rust floating-point work occurs between this write and run's guard.
    unsafe { x86jit_core::jit_abi::fp::set_control(external) };
    let exit = cpu.run(&vm, None);
    let returned = fp_control();
    drop(restore);
    assert!(matches!(exit, Exit::Hlt));
    let observed = cpu.reg(Reg::Rax) | cpu.reg(Reg::Rdx) << 32;
    assert_eq!(observed, x86jit_core::jit_abi::fp::default_control());
    assert_eq!(returned, external);
}

#[test]
fn mxcsr_modes_preexisting_host_control_interpreter() {
    preexisting_host_control(Box::new(InterpreterBackend));
}

#[test]
fn mxcsr_modes_preexisting_host_control_jit() {
    preexisting_host_control(Box::new(JitBackend::new()));
}

#[test]
fn mxcsr_modes_restore_preexisting_host_control_on_panic() {
    struct PanicBackend(std::sync::Arc<std::sync::atomic::AtomicU64>);
    impl Backend for PanicBackend {
        fn materialize(
            &self,
            _: &x86jit_core::ir::IrBlock,
            _: x86jit_core::MemConsistency,
            _: Option<(u64, u64)>,
            _: u64,
        ) -> x86jit_core::CachedBlock {
            // Compilation callbacks must run in Rust's default environment.
            self.0
                .store(fp_control(), std::sync::atomic::Ordering::Relaxed);
            panic!("exercise run's unwind guard");
        }
    }
    let observed = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(u64::MAX));
    let mut vm = Vm::with_backend(
        VmConfig::flat(0x10000),
        Box::new(PanicBackend(observed.clone())),
    );
    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, &[0xf4]).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    let restore = RestoreControl(fp_control());
    let external = x86jit_core::jit_abi::fp::control(0x7f80);
    unsafe { x86jit_core::jit_abi::fp::set_control(external) };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cpu.run(&vm, None)));
    let returned = fp_control();
    drop(restore);
    assert!(result.is_err());
    assert_eq!(returned, external);
    assert_eq!(
        observed.load(std::sync::atomic::Ordering::Relaxed),
        x86jit_core::jit_abi::fp::default_control()
    );
}

#[test]
fn mxcsr_modes_standalone_default_rounding() {
    standalone_default_rounding(false);
}

#[test]
fn mxcsr_modes_standalone_call_block_default_rounding() {
    standalone_default_rounding(true);
}

fn standalone_default_rounding(raw_call: bool) {
    // SDM Vol 1 §4.8.4: 1 + half an ulp rounds to even in the default mode.
    let backend = JitBackend::new();
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), Box::new(InterpreterBackend));
    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    a.addss(xmm0, xmm1).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
        .unwrap();
    let ir = x86jit_core::lift::lift_block(
        &vm.mem,
        x86jit_core::lift::FetchAddr::flat(0x1000),
        x86jit_core::CpuMode::Long64,
    )
    .unwrap();
    let block = backend.materialize(&ir, x86jit_core::MemConsistency::Fast, None, 0);
    let x86jit_core::CachedBlock::Compiled { entry } = block else {
        panic!("expected compiled")
    };
    let mut cpu = x86jit_core::CpuState::default();
    cpu.xmm[0] = 0x3f800000;
    cpu.xmm[1] = 0x33800000;
    let restore = RestoreControl(fp_control());
    let external = x86jit_core::jit_abi::fp::control(0x5f80);
    unsafe {
        x86jit_core::jit_abi::fp::set_control(external);
    }
    let exit = unsafe {
        if raw_call {
            let mut ctx = x86jit_core::jit_abi::MemCtx::for_memory(&vm.mem);
            assert_eq!(
                x86jit_core::jit_abi::call_block(entry, &mut cpu, &mut ctx),
                x86jit_core::jit_abi::RET_HLT
            );
            x86jit_core::StepResult::Exit(Exit::Hlt)
        } else {
            x86jit_core::jit_abi::run_compiled(
                entry,
                &mut cpu,
                &vm.mem,
                x86jit_core::CpuMode::Long64,
            )
        }
    };
    let returned = fp_control();
    drop(restore);
    assert!(matches!(exit, x86jit_core::StepResult::Exit(Exit::Hlt)));
    assert_eq!(returned, external);
    assert_eq!(
        cpu.xmm[0] as u32, 0x3f800000,
        "standalone default translation must round to nearest"
    );
}

#[test]
fn mxcsr_modes_standalone_mmio_replay_uses_default_fp() {
    // SDM Vol 1 §4.8.4: replay retains round-to-nearest-even for the same tie.
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), Box::new(JitBackend::new()));
    vm.map(0, 0x8000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x8000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    a.addss(xmm0, dword_ptr(0x8000u64)).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
        .unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    cpu.cpu.xmm[0] = 0x3f800000;
    assert!(matches!(cpu.run(&vm, None), Exit::MmioRead { .. }));
    cpu.complete_mmio_read(0x33800000);
    let entry = match vm
        .cache
        .get(x86jit_core::BlockKey::with_mxcsr(
            0x1000,
            x86jit_core::CpuMode::Long64,
            0x1f80,
        ))
        .unwrap()
    {
        x86jit_core::CachedBlock::Compiled { entry } => entry,
        _ => panic!("compiled expected"),
    };
    let restore = RestoreControl(fp_control());
    unsafe {
        x86jit_core::jit_abi::fp::set_control(x86jit_core::jit_abi::fp::control(0x5f80));
    }
    let result = unsafe {
        x86jit_core::jit_abi::run_compiled(
            entry,
            &mut cpu.cpu,
            &vm.mem,
            x86jit_core::CpuMode::Long64,
        )
    };
    drop(restore);
    assert!(matches!(result, x86jit_core::StepResult::Continue));
    assert_eq!(
        cpu.cpu.xmm[0] as u32, 0x3f800000,
        "Rust interpreter replay must use the default host environment"
    );
}

#[test]
fn mxcsr_modes_single_step_normalizes_caller_control() {
    interpreter_entry_normalizes_caller_control(0);
}

#[test]
fn mxcsr_modes_step_one_normalizes_caller_control() {
    interpreter_entry_normalizes_caller_control(1);
}

#[test]
fn mxcsr_modes_interpret_block_normalizes_caller_control() {
    interpreter_entry_normalizes_caller_control(2);
}

fn interpreter_entry_normalizes_caller_control(entry: u8) {
    // SDM Vol 1 §4.8.4 and §10.2.3.2: the default RC rounds this tie to even.
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), Box::new(InterpreterBackend));
    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    a.addss(xmm0, xmm1).unwrap();
    vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
        .unwrap();
    let ir = x86jit_core::lift::lift_one(
        &vm.mem,
        x86jit_core::lift::FetchAddr::flat(0x1000),
        x86jit_core::CpuMode::Long64,
    )
    .unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    cpu.cpu.xmm[0] = 0x3f800000;
    cpu.cpu.xmm[1] = 0x33800000;
    let mut scratch = Vec::new();
    let mut info = x86jit_core::interp::RetireInfo::default();
    let restore = RestoreControl(fp_control());
    let external = x86jit_core::jit_abi::fp::control(0x5f80);
    // SAFETY: the entry guard must normalize and restore this thread's control.
    unsafe {
        x86jit_core::jit_abi::fp::set_control(external);
    }
    let result = match entry {
        0 => cpu.step_instruction(&vm),
        1 => x86jit_core::interp::step_one(
            &vm.mem,
            &mut cpu.cpu,
            x86jit_core::CpuMode::Long64,
            &mut scratch,
            &mut info,
        ),
        _ => x86jit_core::interp::interpret_block(
            &ir,
            &mut cpu.cpu,
            &vm.mem,
            &mut scratch,
            &mut info,
        ),
    };
    let returned = fp_control();
    drop(restore);
    assert!(matches!(result, x86jit_core::StepResult::Continue));
    assert_eq!(returned, external, "entry={entry}");
    assert_eq!(
        cpu.cpu.xmm[0] as u32, 0x3f800000,
        "entry={entry}: default ADDSS must round to nearest under caller round-up"
    );
}
