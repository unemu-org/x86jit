//! SDM Vol 2B §4.3: PABSB/PABSW/PABSD/PABSQ, PACKUSDW,
//! PMINSB/PMINSW, PMINUB/PMINUW, PMAXSB/PMAXSW/PMAXSD/PMAXSQ,
//! PMAXUB/PMAXUW; Vol 2A §3.3 EXTRACTPS and §2.5 exception classes.
use iced_x86::{Code, Encoder, Instruction, OpKind, Register};
use x86jit_core::{
    Backend, CpuMode, Exit, GuestCpuFeatures, InterpreterBackend, Prot, Reg, RegionCaps,
    RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;
use x86jit_tests::compare::compare;
use x86jit_tests::oracle::{run_with_backend_mode, RunOutcome, VectorInput};
use x86jit_tests::vector::{CpuSnapshot, ExitKind, MemChunk, MemKind, RunSpec};

const CODE: u64 = 0x1_0000;
const DATA: u64 = 0x2_0000;
const END: u64 = 0x3_0000;

#[derive(Clone, Copy, Debug)]
struct Case {
    code: Code,
    a: u128,
    b: u128,
    expected: u128,
}

fn cases(bits: u32) -> Vec<Case> {
    // Literal witnesses distinguish signedness, saturation boundaries, zero,
    // signed minima, and lane order independently of the shared compute helpers.
    use Code::*;
    let mut cases = vec![
        Case {
            code: Pabsb_xmm_xmmm128,
            a: 17,
            b: 0xff807f0100fe0280_ff807f0100fe0280,
            expected: 0x01807f0100020280_01807f0100020280,
        },
        Case {
            code: Pabsw_xmm_xmmm128,
            a: 17,
            b: 0xffff80007fff0001_0000fffe00028000,
            expected: 0x000180007fff0001_0000000200028000,
        },
        Case {
            code: Pabsd_xmm_xmmm128,
            a: 17,
            b: 0xffffffff80000000_7fffffff00000000,
            expected: 0x0000000180000000_7fffffff00000000,
        },
        Case {
            code: Packusdw_xmm_xmmm128,
            a: 0x000100000000ffff_00000000ffffffff,
            b: 0x800000007fffffff_0000fffe00000001,
            expected: 0x0000fffffffe0001_ffffffff00000000,
        },
        Case {
            code: Pminsb_xmm_xmmm128,
            a: 0x807fff0001fe40c0_807fff0001fe40c0,
            b: 0x7f8000fffe01c040_7f8000fffe01c040,
            expected: 0x8080fffffefec0c0_8080fffffefec0c0,
        },
        Case {
            code: Pmaxsb_xmm_xmmm128,
            a: 0x807fff0001fe40c0_807fff0001fe40c0,
            b: 0x7f8000fffe01c040_7f8000fffe01c040,
            expected: 0x7f7f000001014040_7f7f000001014040,
        },
        Case {
            code: Pminuw_xmm_xmmm128,
            a: 0x80007fffffff0000_0001fffe4000c000,
            b: 0x7fff80000000ffff_fffe0001c0004000,
            expected: 0x7fff7fff00000000_0001000140004000,
        },
        Case {
            code: Pmaxuw_xmm_xmmm128,
            a: 0x80007fffffff0000_0001fffe4000c000,
            b: 0x7fff80000000ffff_fffe0001c0004000,
            expected: 0x80008000ffffffff_fffefffec000c000,
        },
        Case {
            code: Extractps_rm32_xmm_imm8,
            a: 17,
            b: 0x800000007fc12345_deadbeef01234567,
            expected: 0,
        },
    ];
    if bits == 64 {
        cases.push(Case {
            code: Extractps_r64m32_xmm_imm8,
            ..*cases.last().unwrap()
        });
    }
    cases
}

fn extract(case: Case) -> bool {
    matches!(
        case.code,
        Code::Extractps_rm32_xmm_imm8 | Code::Extractps_r64m32_xmm_imm8
    )
}

fn encode(case: Case, memory: bool, alias: bool, imm: u8, bits: u32, region: bool) -> Vec<u8> {
    let mut i = Instruction::default();
    i.set_code(case.code);
    i.set_op0_register(if case.code == Code::Extractps_r64m32_xmm_imm8 {
        Register::RDX
    } else if extract(case) {
        Register::EDX
    } else {
        Register::XMM0
    });
    i.set_op1_register(if alias {
        Register::XMM0
    } else {
        Register::XMM2
    });
    if extract(case) {
        i.set_op_kind(2, OpKind::Immediate8);
        i.set_immediate8(imm);
    }
    if memory {
        i.set_op_kind(if extract(case) { 0 } else { 1 }, OpKind::Memory);
        i.set_memory_base(if bits == 64 {
            Register::RAX
        } else {
            Register::EAX
        });
    }
    let mut e = Encoder::new(bits);
    let mut bytes = if region {
        // Two blocks with an untaken back edge force region compilation.
        // PUSHF/POPF preserve the seeded flags across the loop condition.
        vec![0x9c, 0x83, 0xf9, 0, 0x74, 0xfa, 0x9d]
    } else {
        vec![]
    };
    e.encode(&i, CODE + bytes.len() as u64).unwrap();
    bytes.extend(e.take_buffer());
    bytes.push(0xf4);
    bytes
}

fn input(case: Case, memory: bool, alias: bool, imm: u8, bits: u32) -> VectorInput {
    let mut cpu = CpuSnapshot {
        rip: CODE,
        ..Default::default()
    };
    cpu.gpr[0] = DATA;
    cpu.gpr[1] = 1;
    cpu.gpr[2] = if bits == 64 {
        0xffff_ffff_8765_4321
    } else {
        0x8765_4321
    };
    cpu.gpr[4] = DATA + 0x100;
    cpu.xmm.fill(0x123456789abcdef0_fedcba9876543210);
    cpu.ymm_hi.fill(0x8877665544332211_00ffeeddccbbaa99);
    cpu.xmm[0] = if alias { case.b } else { case.a };
    cpu.xmm[2] = case.b;
    cpu.flags.cf = true;
    cpu.flags.of = true;
    cpu.flags.zf = true;
    let mut data = vec![0x39; 0x200];
    data[..16].copy_from_slice(&case.b.to_le_bytes());
    VectorInput {
        cpu_init: cpu,
        mem_init: vec![
            MemChunk {
                addr: CODE,
                bytes: encode(case, memory, alias, imm, bits, false),
                kind: MemKind::Ram,
            },
            MemChunk {
                addr: DATA,
                bytes: data,
                kind: MemKind::Ram,
            },
        ],
        entry: CODE,
        run: RunSpec::UntilExit,
    }
}

fn backend(tier: u8) -> Box<dyn Backend> {
    match tier {
        0 => Box::new(InterpreterBackend),
        1 => Box::new(JitBackend::new()),
        _ => Box::new(JitBackend::with_superblocks(RegionCaps {
            max_blocks: 8,
            max_icount: 128,
        })),
    }
}

fn mode(bits: u32) -> CpuMode {
    if bits == 64 {
        CpuMode::Long64
    } else {
        CpuMode::Compat32
    }
}

fn state(cpu: &x86jit_core::state::CpuState) -> Vec<u128> {
    let mut result: Vec<_> = cpu.gpr.iter().map(|&x| u128::from(x)).collect();
    result.push(cpu.flags.to_rflags().into());
    result.push(cpu.mxcsr.into());
    result.extend(cpu.xmm);
    result.extend(cpu.ymm_hi);
    result.extend(cpu.zmm_hi.iter().flatten());
    result.extend(cpu.kmask.map(u128::from));
    result
}

#[test]
fn legacy_sse_integer_witnesses_all_tiers() {
    // SDM Vol 2B §4.3, the entries cited above: legacy writes preserve bits
    // MAXVL-1:128 and flags. Vol 2A §3.3 EXTRACTPS: imm8[1:0], zero-extended GPR.
    for bits in [64, 32] {
        for case in cases(bits) {
            for memory in [false, true] {
                for alias in [false, true] {
                    for imm in if extract(case) {
                        vec![0, 1, 2, 3, 0xfc, 0xfd, 0xfe, 0xff]
                    } else {
                        vec![0]
                    } {
                        let i = input(case, memory, alias, imm, bits);
                        let reference = run_with_backend_mode(
                            &i,
                            backend(0),
                            GuestCpuFeatures::default(),
                            mode(bits),
                        );
                        assert_eq!(reference.exit, ExitKind::Hlt, "{case:?}");
                        same(
                            reference.clone(),
                            run_with_backend_mode(
                                &i,
                                backend(1),
                                GuestCpuFeatures::default(),
                                mode(bits),
                            ),
                            &format!("{case:?} {bits} mem={memory} alias={alias}"),
                        );
                        for tier in 0..3 {
                            let mut vm = Vm::with_backend(VmConfig::flat(END), backend(tier));
                            vm.set_cpu_mode(mode(bits));
                            vm.map(0, END as usize, Prot::RWX, RegionKind::Ram).unwrap();
                            let bytes = encode(case, memory, alias, imm, bits, tier == 2);
                            vm.write_bytes(CODE, &bytes).unwrap();
                            vm.write_bytes(DATA, &i.mem_init[1].bytes).unwrap();
                            let mut cpu = vm.new_vcpu();
                            cpu.cpu.gpr = i.cpu_init.gpr;
                            cpu.cpu.xmm = i.cpu_init.xmm;
                            cpu.cpu.ymm_hi = i.cpu_init.ymm_hi;
                            cpu.cpu.zmm_hi.fill([0x1234, 0x5678]);
                            cpu.cpu.flags = i.cpu_init.flags.into();
                            cpu.set_reg(Reg::Rip, CODE);
                            let exit = cpu.run(&vm, Some(16));
                            assert!(
                                matches!(exit, Exit::Hlt),
                                "{case:?} bits={bits} tier={tier}: {exit:?}"
                            );
                            if tier == 2 {
                                assert!(vm.cache.regions() > 0, "region did not compile");
                            }
                            assert_eq!(cpu.cpu.ymm_hi, i.cpu_init.ymm_hi);
                            assert_eq!(cpu.cpu.zmm_hi, [[0x1234, 0x5678]; 32]);
                            assert_eq!(cpu.cpu.flags.to_rflags(), i.cpu_init.flags.to_rflags());
                            assert_eq!(&cpu.cpu.gpr[..], &reference.cpu.gpr[..]);
                            assert_eq!(cpu.cpu.xmm, reference.cpu.xmm);
                            if extract(case) {
                                let value = ((case.b >> ((imm & 3) * 32)) & 0xffff_ffff) as u32;
                                if memory {
                                    let mut out = [0; 16];
                                    vm.read_bytes(DATA, &mut out).unwrap();
                                    let mut expected = case.b.to_le_bytes();
                                    expected[..4].copy_from_slice(&value.to_le_bytes());
                                    assert_eq!(out, expected);
                                } else {
                                    assert_eq!(cpu.reg(Reg::Rdx), u64::from(value));
                                }
                                assert_eq!(cpu.cpu.xmm, i.cpu_init.xmm);
                            } else {
                                let expected = if !alias
                                    || matches!(
                                        case.code,
                                        Code::Pabsb_xmm_xmmm128
                                            | Code::Pabsw_xmm_xmmm128
                                            | Code::Pabsd_xmm_xmmm128
                                    ) {
                                    case.expected
                                } else if case.code == Code::Packusdw_xmm_xmmm128 {
                                    0x0000fffffffe0001_0000fffffffe0001
                                } else {
                                    case.b
                                };
                                assert_eq!(
                                    cpu.xmm(0),
                                    expected,
                                    "{case:?} tier={tier} alias={alias}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn legacy_sse_integer_exact_width_alignment_and_atomicity() {
    // SDM Vol 2A §2.5: legacy arithmetic m128 requires 16-byte alignment;
    // EXTRACTPS writes exactly m32. Vol 3A §7.5: faults preserve instruction state.
    for bits in [64, 32] {
        for case in cases(bits) {
            let addresses = if extract(case) {
                vec![END - 4, END - 5, END - 3, END - 1, END]
            } else {
                vec![END - 16, END - 32, END, END - 31, END - 24]
            };
            for at in addresses {
                for tier in 0..3 {
                    let mut vm = Vm::with_backend(VmConfig::flat(END), backend(tier));
                    vm.set_cpu_mode(mode(bits));
                    vm.map(0, END as usize, Prot::RWX, RegionKind::Ram).unwrap();
                    vm.write_bytes(CODE, &encode(case, true, false, 0xff, bits, tier == 2))
                        .unwrap();
                    vm.write_bytes(END - 32, &[0x81; 32]).unwrap();
                    let mut cpu = vm.new_vcpu();
                    cpu.cpu.gpr = input(case, true, false, 0xff, bits).cpu_init.gpr;
                    cpu.cpu.gpr[0] = at;
                    cpu.cpu.xmm.fill(case.b);
                    cpu.cpu.ymm_hi.fill(u128::MAX);
                    cpu.cpu.zmm_hi.fill([0x1234, 0x5678]);
                    cpu.set_reg(Reg::Rip, CODE);
                    let before = state(&cpu.cpu);
                    let exit = cpu.run(&vm, Some(16));
                    let fault = at + if extract(case) { 4 } else { 16 } > END
                        || (!extract(case) && at % 16 != 0);
                    let mut after = [0; 32];
                    vm.read_bytes(END - 32, &mut after).unwrap();
                    if fault {
                        if !extract(case) && at % 16 != 0 {
                            assert!(
                                matches!(exit, Exit::Exception { vector: 13, .. }),
                                "{case:?} tier={tier} {exit:?}"
                            );
                        } else {
                            assert!(
                                matches!(exit, Exit::UnmappedMemory { addr: END, .. }),
                                "{case:?} tier={tier} at={at:#x}: {exit:?}"
                            );
                        }
                        assert_eq!(state(&cpu.cpu), before, "{case:?} tier={tier}");
                        assert_eq!(cpu.reg(Reg::Rip), CODE + if tier == 2 { 7 } else { 0 });
                        assert_eq!(after, [0x81; 32]);
                    } else {
                        assert!(matches!(exit, Exit::Hlt), "{case:?}: {exit:?}");
                    }
                    if tier == 2 {
                        assert!(vm.cache.regions() > 0);
                    }
                }
            }
        }
    }
}

fn same(a: RunOutcome, b: RunOutcome, context: &str) {
    assert_eq!(a.exit, ExitKind::Hlt, "{context}: {a:?}");
    assert_eq!(b.exit, ExitKind::Hlt, "{context}: {b:?}");
    assert!(
        compare(&a, &b, &[]).is_none(),
        "{context}: {:?}",
        compare(&a, &b, &[])
    );
}

#[cfg(feature = "unicorn")]
#[test]
fn legacy_sse_integer_matches_unicorn() {
    use x86jit_tests::oracle::Oracle;
    use x86jit_tests::unicorn::{UnicornOracle, UnicornOracle32};
    for bits in [64, 32] {
        for case in cases(bits) {
            for memory in [false, true] {
                for alias in [false, true] {
                    let mut i = input(case, memory, alias, 0xff, bits);
                    // The Unicorn wrapper captures XMM only; upper preservation
                    // has independent architectural witnesses above.
                    i.cpu_init.ymm_hi.fill(0);
                    // Unicorn models XMM0-XMM15 (XMM0-XMM7 in 32-bit mode); the
                    // higher registers exist only with REX or AVX-512.
                    i.cpu_init.xmm[if bits == 64 { 16 } else { 8 }..].fill(0);
                    let u = if bits == 64 {
                        UnicornOracle.run(&i)
                    } else {
                        UnicornOracle32.run(&i)
                    };
                    for tier in [0, 1] {
                        same(
                            u.clone(),
                            run_with_backend_mode(
                                &i,
                                backend(tier),
                                GuestCpuFeatures::default(),
                                mode(bits),
                            ),
                            &format!("{case:?} {bits} mem={memory} alias={alias}"),
                        );
                    }
                }
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn legacy_sse_integer_matches_native() {
    if !std::is_x86_feature_detected!("avx") || !std::is_x86_feature_detected!("sse4.1") {
        return;
    }
    for case in cases(64) {
        for memory in [false, true] {
            for alias in [false, true] {
                for imm in [0, 1, 2, 3, 0xfc, 0xfd, 0xfe, 0xff] {
                    let mut i = input(case, memory, alias, imm, 64);
                    // The native stub cannot seed XMM16-31 (native.rs); zero them on
                    // both sides so the comparison starts from one state.
                    i.cpu_init.xmm[16..].fill(0);
                    i.cpu_init.ymm_hi[16..].fill(0);
                    let n = x86jit_tests::native::run_native(&i).expect("native legacy SSE form");
                    for tier in [0, 1] {
                        same(
                            n.clone(),
                            run_with_backend_mode(
                                &i,
                                backend(tier),
                                GuestCpuFeatures::default(),
                                CpuMode::Long64,
                            ),
                            &format!("{case:?} mem={memory} alias={alias} imm={imm}"),
                        );
                    }
                }
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn legacy_sse_integer_faults_match_native() {
    use x86jit_tests::native::{run_native_fault, NativeFault};
    if !std::is_x86_feature_detected!("avx") || !std::is_x86_feature_detected!("sse4.1") {
        return;
    }
    for case in cases(64) {
        let addresses = if extract(case) {
            vec![END, END - 1, END - 3]
        } else {
            // Aligned unmapped operand and fully mapped misaligned operand:
            // neither case leaves exception priority ambiguous (SDM Vol 2A §2.5).
            vec![END, END - 31]
        };
        for at in addresses {
            let mut i = input(case, true, false, 0xff, 64);
            // The native stub cannot seed XMM16-31 (native.rs).
            i.cpu_init.xmm[16..].fill(0);
            i.cpu_init.ymm_hi[16..].fill(0);
            i.cpu_init.gpr[0] = at;
            i.cpu_init.gpr[4] = END - 0x100;
            i.mem_init[1].addr = END - 0x1000;
            i.mem_init[1].bytes = vec![0x81; 0x1000];
            let fault = if !extract(case) && at % 16 != 0 {
                NativeFault::GeneralProtection
            } else {
                NativeFault::PageFault { addr: END }
            };
            let native = run_native_fault(&i, fault).expect("native legacy SSE memory fault");
            for tier in [0, 1] {
                let result = run_with_backend_mode(
                    &i,
                    backend(tier),
                    GuestCpuFeatures::default(),
                    CpuMode::Long64,
                );
                assert!(
                    compare(&native, &result, &[]).is_none(),
                    "{case:?} tier={tier} at={at:#x}: {:?}",
                    compare(&native, &result, &[])
                );
            }
        }
    }
}
