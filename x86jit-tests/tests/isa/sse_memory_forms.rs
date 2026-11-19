//! Memory widths: SDM Vol 2B, PACKUSWB, PMADDWD, PSHUFHW/PSHUFLW,
//! PSLLW/PSLLD/PSLLQ, PSRAW/PSRAD, PSRLW/PSRLD/PSRLQ, SQRTPD/PS/SD/SS,
//! MPSADBW, PBLENDW, PMADDUBSW and PTEST instruction pages.
//! VEX moves: SDM Vol 2A, LDDQU; masks: SDM Vol 2C, VPMASKMOV.
use iced_x86::{Code, Encoder, Instruction, OpKind, Register};
use x86jit_core::{CpuMode, GuestCpuFeatures, InterpreterBackend};
use x86jit_cranelift::JitBackend;
use x86jit_tests::compare::compare;
use x86jit_tests::oracle::{run_with_backend_mode, RunOutcome, VectorInput};
use x86jit_tests::vector::{CpuSnapshot, ExitKind, MemChunk, MemKind, RunSpec};

const CODE: u64 = 0x1000;
const DATA: u64 = 0x8000;

#[derive(Clone, Copy, Debug)]
struct Case {
    code: Code,
    bytes: usize,
    imm: bool,
    vex: bool,
}

fn cases() -> Vec<Case> {
    use Code::*;
    let mut cases: Vec<_> = [
        Packuswb_xmm_xmmm128,
        Pmaddwd_xmm_xmmm128,
        Pshufhw_xmm_xmmm128_imm8,
        Pshuflw_xmm_xmmm128_imm8,
        Pslld_xmm_xmmm128,
        Psllq_xmm_xmmm128,
        Psllw_xmm_xmmm128,
        Psrad_xmm_xmmm128,
        Psraw_xmm_xmmm128,
        Psrld_xmm_xmmm128,
        Psrlq_xmm_xmmm128,
        Psrlw_xmm_xmmm128,
        Sqrtpd_xmm_xmmm128,
        Sqrtps_xmm_xmmm128,
        Sqrtsd_xmm_xmmm64,
        Sqrtss_xmm_xmmm32,
        Mpsadbw_xmm_xmmm128_imm8,
        Pblendw_xmm_xmmm128_imm8,
        Pmaddubsw_xmm_xmmm128,
        Ptest_xmm_xmmm128,
    ]
    .into_iter()
    .map(|code| Case {
        code,
        bytes: match code {
            Sqrtss_xmm_xmmm32 => 4,
            Sqrtsd_xmm_xmmm64 => 8,
            _ => 16,
        },
        imm: matches!(
            code,
            Pshufhw_xmm_xmmm128_imm8
                | Pshuflw_xmm_xmmm128_imm8
                | Mpsadbw_xmm_xmmm128_imm8
                | Pblendw_xmm_xmmm128_imm8
        ),
        vex: false,
    })
    .collect();
    for (code, bytes) in [
        (VEX_Vpblendw_xmm_xmm_xmmm128_imm8, 16),
        (VEX_Vpblendw_ymm_ymm_ymmm256_imm8, 32),
        (VEX_Vmpsadbw_xmm_xmm_xmmm128_imm8, 16),
        (VEX_Vmpsadbw_ymm_ymm_ymmm256_imm8, 32),
    ] {
        cases.push(Case {
            code,
            bytes,
            imm: true,
            vex: true,
        });
    }
    cases
}

fn encode(case: Case, memory: bool, bits: u32, imm: u8) -> Vec<u8> {
    let mut i = Instruction::default();
    i.set_code(case.code);
    let (dst, src) = if case.bytes == 32 {
        (Register::YMM0, Register::YMM1)
    } else {
        (Register::XMM0, Register::XMM1)
    };
    i.set_op0_register(dst);
    let operand = if case.vex {
        i.set_op1_register(dst);
        2
    } else {
        1
    };
    if memory {
        i.set_op_kind(operand, OpKind::Memory);
        i.set_memory_base(if bits == 64 {
            Register::RAX
        } else {
            Register::EAX
        });
    } else {
        i.set_op_register(operand, src);
    }
    if case.imm {
        i.set_op_kind(operand + 1, OpKind::Immediate8);
        i.set_immediate8(imm);
    }
    let mut e = Encoder::new(bits);
    e.encode(&i, CODE).unwrap();
    let mut bytes = e.take_buffer();
    bytes.push(0xf4);
    bytes
}

fn input(case: Case, memory: bool, bits: u32, imm: u8, seed: u128) -> VectorInput {
    // Keep sqrt witnesses finite and nonnegative: NaN payload/sign policy is
    // covered by the separate floating-point tests.
    let seed = match case.code {
        Code::Sqrtpd_xmm_xmmm128 | Code::Sqrtsd_xmm_xmmm64 => {
            seed & 0x7fef_ffff_ffff_ffff_7fef_ffff_ffff_ffff
        }
        Code::Sqrtps_xmm_xmmm128 | Code::Sqrtss_xmm_xmmm32 => {
            seed & 0x7f7f_ffff_7f7f_ffff_7f7f_ffff_7f7f_ffff
        }
        _ => seed,
    };
    let mut cpu = CpuSnapshot {
        rip: CODE,
        ..Default::default()
    };
    cpu.gpr[0] = DATA;
    cpu.xmm[0] = seed.rotate_left(43);
    cpu.xmm[1] = seed;
    cpu.ymm_hi[0] = !seed;
    cpu.ymm_hi[1] = seed.rotate_left(17);
    cpu.flags.cf = true;
    cpu.flags.of = true;
    let mut data = seed.to_le_bytes().to_vec();
    data.extend_from_slice(&cpu.ymm_hi[1].to_le_bytes());
    VectorInput {
        cpu_init: cpu,
        mem_init: vec![
            MemChunk {
                addr: CODE,
                bytes: encode(case, memory, bits, imm),
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

fn same_state(mut a: RunOutcome, mut b: RunOutcome, context: &str) {
    // Encodings have different lengths; code bytes themselves are not output data.
    a.cpu.rip = 0;
    b.cpu.rip = 0;
    a.mem.retain(|m| m.addr != CODE);
    b.mem.retain(|m| m.addr != CODE);
    assert_eq!(a.exit, ExitKind::Hlt, "{context}: register form");
    assert_eq!(b.exit, ExitKind::Hlt, "{context}: memory form");
    assert!(
        compare(&a, &b, &[]).is_none(),
        "{context}: {:?}",
        compare(&a, &b, &[])
    );
}

fn check_register_forms(cases: &[Case]) {
    for bits in [64, 32] {
        let mode = if bits == 64 {
            CpuMode::Long64
        } else {
            CpuMode::Compat32
        };
        for &case in cases {
            for (imm, seed) in [
                (0, 0),
                (0xa5, 0x40000000408000004010000041100000),
                (0xff, u128::MAX),
                (0x53, 0x80017fff0100fffe0000000000000011),
            ] {
                let reg = input(case, false, bits, imm, seed);
                let mem = input(case, true, bits, imm, seed);
                let r = run_with_backend_mode(
                    &reg,
                    Box::new(InterpreterBackend),
                    GuestCpuFeatures::default(),
                    mode,
                );
                let m = run_with_backend_mode(
                    &mem,
                    Box::new(InterpreterBackend),
                    GuestCpuFeatures::default(),
                    mode,
                );
                let j = run_with_backend_mode(
                    &mem,
                    Box::new(JitBackend::new()),
                    GuestCpuFeatures::default(),
                    mode,
                );
                let jr = run_with_backend_mode(
                    &reg,
                    Box::new(JitBackend::new()),
                    GuestCpuFeatures::default(),
                    mode,
                );
                let context = format!("{:?} {bits} imm={imm}", case.code);
                same_state(r.clone(), m.clone(), &context);
                same_state(r.clone(), jr, &context);
                same_state(m, j, &context);
            }
        }
    }
}

#[test]
fn sse_memory_forms_match_register_and_interpreter() {
    check_register_forms(&cases().into_iter().filter(|c| !c.vex).collect::<Vec<_>>());
}

#[test]
fn aliased_vpblendw_vmpsadbw_memory_match_register_and_interpreter() {
    check_register_forms(&cases().into_iter().filter(|c| c.vex).collect::<Vec<_>>());
}

#[cfg(feature = "unicorn")]
#[test]
fn sse_memory_forms_match_unicorn() {
    use x86jit_tests::oracle::{InterpreterOracle, InterpreterOracle32, Oracle};
    use x86jit_tests::unicorn::{UnicornOracle, UnicornOracle32};
    // Unicorn's VEX decoder is not an oracle for AVX; those use the register-form
    // comparison above and the native x86-64 fuzz lane.
    for bits in [64, 32] {
        for case in cases().into_iter().filter(|c| !c.vex) {
            let mut i = input(case, true, bits, 0xa5, 0x40000000408000004010000041100000);
            // This Unicorn wrapper captures XMM only. Dirty YMM preservation is
            // checked by the interpreter/JIT matrix above.
            i.cpu_init.ymm_hi.fill(0);
            let (u, m) = if bits == 64 {
                (UnicornOracle.run(&i), InterpreterOracle.run(&i))
            } else {
                (UnicornOracle32.run(&i), InterpreterOracle32.run(&i))
            };
            same_state(u, m, &format!("{:?} {bits}", case.code));
        }
    }
}

fn move_code(bytes: usize, qword: bool, store: bool, reference: bool) -> Code {
    use Code::*;
    match (bytes, qword, store, reference) {
        (16, false, false, false) => VEX_Vpmaskmovd_xmm_xmm_m128,
        (32, false, false, false) => VEX_Vpmaskmovd_ymm_ymm_m256,
        (16, true, false, false) => VEX_Vpmaskmovq_xmm_xmm_m128,
        (32, true, false, false) => VEX_Vpmaskmovq_ymm_ymm_m256,
        (16, false, true, false) => VEX_Vpmaskmovd_m128_xmm_xmm,
        (32, false, true, false) => VEX_Vpmaskmovd_m256_ymm_ymm,
        (16, true, true, false) => VEX_Vpmaskmovq_m128_xmm_xmm,
        (32, true, true, false) => VEX_Vpmaskmovq_m256_ymm_ymm,
        (16, false, false, true) => VEX_Vmaskmovps_xmm_xmm_m128,
        (32, false, false, true) => VEX_Vmaskmovps_ymm_ymm_m256,
        (16, true, false, true) => VEX_Vmaskmovpd_xmm_xmm_m128,
        (32, true, false, true) => VEX_Vmaskmovpd_ymm_ymm_m256,
        (16, false, true, true) => VEX_Vmaskmovps_m128_xmm_xmm,
        (32, false, true, true) => VEX_Vmaskmovps_m256_ymm_ymm,
        (16, true, true, true) => VEX_Vmaskmovpd_m128_xmm_xmm,
        (32, true, true, true) => VEX_Vmaskmovpd_m256_ymm_ymm,
        _ => unreachable!(),
    }
}

fn encode_mask(code: Code, bytes: usize, store: bool, bits: u32) -> Vec<u8> {
    let mut i = Instruction::default();
    i.set_code(code);
    let (data, mask) = if bytes == 16 {
        (Register::XMM0, Register::XMM2)
    } else {
        (Register::YMM0, Register::YMM2)
    };
    i.set_op1_register(mask);
    if store {
        i.set_op2_register(data);
    } else {
        i.set_op0_register(data);
    }
    i.set_op_kind(if store { 0 } else { 2 }, OpKind::Memory);
    i.set_memory_base(if bits == 64 {
        Register::RAX
    } else {
        Register::EAX
    });
    let mut e = Encoder::new(bits);
    e.encode(&i, CODE).unwrap();
    let mut result = e.take_buffer();
    result.push(0xf4);
    result
}

fn lddqu_cases() -> Vec<Case> {
    [
        (Code::VEX_Vlddqu_xmm_m128, 16),
        (Code::VEX_Vlddqu_ymm_m256, 32),
    ]
    .into_iter()
    .map(|(code, bytes)| Case {
        code,
        bytes,
        imm: false,
        vex: false,
    })
    .collect()
}

#[test]
fn vlddqu_and_vpmaskmov_match_existing_forms() {
    // SDM Vol 2A, LDDQU: unaligned m128/m256; Vol 2C, VPMASKMOV:
    // sign-bit-selected dword/qword elements, inactive loads zeroed/stores untouched.
    for bits in [64, 32] {
        let mode = if bits == 64 {
            CpuMode::Long64
        } else {
            CpuMode::Compat32
        };
        for case in lddqu_cases() {
            let mut mem = input(case, true, bits, 0, 0x0123456789abcdef_fedcba9876543210);
            // Exercise an unaligned load independently of the boundary witnesses.
            mem.cpu_init.gpr[0] += 1;
            mem.mem_init[1].bytes.push(0x93);
            mem.cpu_init.xmm[1] =
                u128::from_le_bytes(mem.mem_init[1].bytes[1..17].try_into().unwrap());
            if case.bytes == 32 {
                mem.cpu_init.ymm_hi[1] =
                    u128::from_le_bytes(mem.mem_init[1].bytes[17..33].try_into().unwrap());
            }
            let mut reference = mem.clone();
            let reg = Case {
                code: if case.bytes == 16 {
                    Code::VEX_Vmovdqu_xmm_xmmm128
                } else {
                    Code::VEX_Vmovdqu_ymm_ymmm256
                },
                ..case
            };
            reference.mem_init[0].bytes = encode(reg, false, bits, 0);
            let r = run_with_backend_mode(
                &reference,
                Box::new(InterpreterBackend),
                GuestCpuFeatures::default(),
                mode,
            );
            let m = run_with_backend_mode(
                &mem,
                Box::new(InterpreterBackend),
                GuestCpuFeatures::default(),
                mode,
            );
            let j = run_with_backend_mode(
                &mem,
                Box::new(JitBackend::new()),
                GuestCpuFeatures::default(),
                mode,
            );
            same_state(r, m.clone(), &format!("{:?}", case.code));
            same_state(m, j, &format!("{:?}", case.code));
        }
        for bytes in [16, 32] {
            for qword in [false, true] {
                for store in [false, true] {
                    for mask in [
                        0,
                        u128::MAX,
                        0x800000007fffffff_800000007fffffff,
                        0x8000000000000000_7fffffffffffffff,
                    ] {
                        let case = Case {
                            code: move_code(bytes, qword, store, false),
                            bytes,
                            imm: false,
                            vex: false,
                        };
                        let mut i = input(
                            Case {
                                code: if bytes == 16 {
                                    Code::VEX_Vlddqu_xmm_m128
                                } else {
                                    Code::VEX_Vlddqu_ymm_m256
                                },
                                ..case
                            },
                            true,
                            bits,
                            0,
                            0x0123456789abcdef_fedcba9876543210,
                        );
                        i.cpu_init.xmm[2] = mask;
                        i.cpu_init.ymm_hi[2] = mask.rotate_left(64);
                        i.mem_init[0].bytes = encode_mask(case.code, bytes, store, bits);
                        let mut reference = i.clone();
                        reference.mem_init[0].bytes =
                            encode_mask(move_code(bytes, qword, store, true), bytes, store, bits);
                        let r = run_with_backend_mode(
                            &reference,
                            Box::new(InterpreterBackend),
                            GuestCpuFeatures::default(),
                            mode,
                        );
                        let m = run_with_backend_mode(
                            &i,
                            Box::new(InterpreterBackend),
                            GuestCpuFeatures::default(),
                            mode,
                        );
                        let j = run_with_backend_mode(
                            &i,
                            Box::new(JitBackend::new()),
                            GuestCpuFeatures::default(),
                            mode,
                        );
                        same_state(r, m.clone(), &format!("{:?}", case.code));
                        same_state(m, j, &format!("{:?}", case.code));
                    }
                }
            }
        }
    }
}

const SPAN: u64 = 0x1_0000;

fn registers(cpu: &x86jit_core::Vcpu) -> Vec<u128> {
    use x86jit_core::Reg;
    let mut state: Vec<u128> = (0..16)
        .map(|i| cpu.reg(Reg::from_gpr_index(i)) as u128)
        .collect();
    state.push(cpu.flags().to_rflags() as u128);
    for i in 0..32 {
        state.extend([
            cpu.xmm(i),
            cpu.ymm_hi(i),
            cpu.zmm_hi(i, 0),
            cpu.zmm_hi(i, 1),
        ]);
    }
    for i in 0..8 {
        state.push(cpu.kmask(i) as u128);
    }
    state
}

// Out-of-span unmapped page, matching fault_atomicity.rs: the JIT's flat RAM
// model does not consult the interpreter's region map for in-span holes.
fn boundary_run(
    code: &[u8],
    at: u64,
    bits: u32,
    jit: bool,
    mask: (u128, u128),
    fault: bool,
) -> (u128, u128, Vec<u8>) {
    use x86jit_core::{Backend, Exit, Prot, Reg, RegionKind, Vm, VmConfig};
    let backend: Box<dyn Backend> = if jit {
        Box::new(JitBackend::new())
    } else {
        Box::new(InterpreterBackend)
    };
    let mut vm = Vm::with_backend(VmConfig::flat(SPAN), backend);
    vm.set_cpu_mode(if bits == 64 {
        CpuMode::Long64
    } else {
        CpuMode::Compat32
    });
    vm.map(0, SPAN as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.write_bytes(CODE, code).unwrap();
    let mut initial_memory = vec![0; SPAN as usize];
    // Positive finite floats as well as varied integer lanes.
    vm.write_bytes(SPAN - 64, &[0x40; 64]).unwrap();
    vm.read_bytes(0, &mut initial_memory).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    for i in 0..16 {
        cpu.set_reg(Reg::from_gpr_index(i), i as u64 * 37);
    }
    cpu.set_reg(Reg::Rax, at);
    for i in 0..32 {
        cpu.set_xmm(i, 0x1122334455667788_99aabbccddeeff00 ^ i as u128);
        cpu.set_ymm_hi(i, 0x8877665544332211_00ffeeddccbbaa99 ^ i as u128);
        cpu.set_zmm_hi(i, 0, i as u128 + 71);
        cpu.set_zmm_hi(i, 1, i as u128 + 93);
    }
    cpu.set_xmm(2, mask.0);
    cpu.set_ymm_hi(2, mask.1);
    let mut flags = cpu.flags();
    flags.cf = true;
    flags.zf = true;
    flags.sf = true;
    flags.of = true;
    flags.set_af(true);
    flags.set_pf(true);
    cpu.set_flags(flags);
    let before = registers(&cpu);
    let exit = cpu.run(&vm, None);
    let mut after_memory = vec![0; SPAN as usize];
    vm.read_bytes(0, &mut after_memory).unwrap();
    if fault {
        let insn = iced_x86::Decoder::new(bits, code, iced_x86::DecoderOptions::NONE).decode();
        // SDM Vol 2A §2.5, Tables 2-15/2-19/2-21: these legacy m128 forms
        // require alignment. Only the fully mapped case has a unique fault.
        let alignment_fault =
            insn.encoding() == iced_x86::EncodingKind::Legacy && insn.memory_size().size() == 16;
        let expected = (!alignment_fault || at + insn.memory_size().size() as u64 > SPAN)
            && matches!(exit, Exit::UnmappedMemory { addr: SPAN, .. })
            || (alignment_fault
                && matches!(
                    exit,
                    Exit::Exception {
                        addr: CODE,
                        vector: 13,
                        ..
                    }
                ));
        assert!(expected, "bits={bits} jit={jit} at={at:#x}: {exit:?}");
        assert_eq!(cpu.reg(Reg::Rip), CODE);
        assert_eq!(registers(&cpu), before, "registers changed on fault");
        assert_eq!(
            after_memory, initial_memory,
            "partial memory store before fault"
        );
    } else {
        assert!(matches!(exit, Exit::Hlt), "{exit:?}");
    }
    (cpu.xmm(0), cpu.ymm_hi(0), after_memory)
}

#[test]
fn sse_memory_forms_fault_without_committing() {
    for bits in [64, 32] {
        for case in cases().into_iter().chain(lddqu_cases()) {
            let code = encode(case, true, bits, 0xa5);
            // Unaligned legacy m128 faults before the first transfer; other forms
            // fault on the second transfer. Neither may commit its destination.
            let mapped = case.bytes / 2;
            for jit in [false, true] {
                boundary_run(&code, SPAN - mapped as u64, bits, jit, (0, 0), true);
            }
        }
    }
}

#[test]
fn legacy_m128_mapped_misalignment_raises_gp() {
    // SDM Vol 2A §2.5, Tables 2-18/2-19/2-21: no paging-priority ambiguity.
    for bits in [64, 32] {
        for case in cases() {
            if case.bytes == 16 && case.code.encoding() == iced_x86::EncodingKind::Legacy {
                let code = encode(case, true, bits, 0xa5);
                for jit in [false, true] {
                    boundary_run(&code, SPAN - 63, bits, jit, (0, 0), true);
                }
            }
        }
    }
}

#[test]
fn sse_memory_forms_use_exact_operand_width() {
    for bits in [64, 32] {
        for case in cases().into_iter().chain(lddqu_cases()) {
            let code = encode(case, true, bits, 0x53);
            let r = boundary_run(&code, SPAN - case.bytes as u64, bits, false, (0, 0), false);
            let j = boundary_run(&code, SPAN - case.bytes as u64, bits, true, (0, 0), false);
            assert_eq!(r, j, "{:?} bits={bits}", case.code);
        }
    }
}

#[test]
fn vpmaskmov_suppresses_inactive_faults_and_commits_last() {
    for bits in [64, 32] {
        for bytes in [16, 32] {
            for qword in [false, true] {
                let elem = if qword { 8 } else { 4 };
                let first = 1u128 << (elem * 8 - 1);
                for store in [false, true] {
                    let code =
                        encode_mask(move_code(bytes, qword, store, false), bytes, store, bits);
                    for jit in [false, true] {
                        // No selected elements: the whole operand can be unmapped.
                        boundary_run(&code, SPAN, bits, jit, (0, 0), false);
                        // Only the first element is backed and selected. Its end is
                        // exactly the page end; no inactive element may be touched.
                        let (lo, hi, mem) =
                            boundary_run(&code, SPAN - elem as u64, bits, jit, (first, 0), false);
                        if store {
                            assert_eq!(
                                &mem[SPAN as usize - elem..],
                                &0x1122334455667788_99aabbccddeeff00u128.to_le_bytes()[..elem]
                            );
                        } else {
                            assert_eq!(
                                lo,
                                if elem == 4 {
                                    0x40404040
                                } else {
                                    0x4040404040404040
                                }
                            );
                            assert_eq!(hi, 0);
                        }
                        if bytes == 32 {
                            // Fault in the high 128-bit lane after a valid low-lane element.
                            boundary_run(&code, SPAN - 16, bits, jit, (first, first), true);
                        }
                        // First element selected and backed, next selected and
                        // unmapped: no earlier register or store may commit.
                        boundary_run(
                            &code,
                            SPAN - elem as u64,
                            bits,
                            jit,
                            (first | (first << (elem * 8)), 0),
                            true,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn new_memory_fuzz_menu_lifts_and_matches_interpreter() {
    use iced_x86::code_asm::CodeAssembler;
    use x86jit_tests::fuzz::V_VEX;
    for op in &V_VEX[63..] {
        let mut asm = CodeAssembler::new(64).unwrap();
        op.emit_mem.unwrap()(&mut asm, 0, 0, 1, DATA, 0xa5);
        asm.hlt().unwrap();
        let mut i = input(cases()[0], true, 64, 0, 0x4040404040404040_4040404040404040);
        i.mem_init[0].bytes = asm.assemble(CODE).unwrap();
        let m = run_with_backend_mode(
            &i,
            Box::new(InterpreterBackend),
            GuestCpuFeatures::default(),
            CpuMode::Long64,
        );
        let j = run_with_backend_mode(
            &i,
            Box::new(JitBackend::new()),
            GuestCpuFeatures::default(),
            CpuMode::Long64,
        );
        same_state(m, j, op.name);
    }
}

// SDM Vol 2C, VPMASKMOV, Description and Operation: selected dword/qword
// elements access memory, including unaligned addresses. Fault addresses follow §16.
fn vpmaskmov_unaligned_boundary(jit: bool, qword: bool, store: bool) {
    let elem = if qword { 8 } else { 4 };
    let first = 1u128 << (elem * 8 - 1);
    for bits in [64, 32] {
        for bytes in [16, 32] {
            let code = encode_mask(move_code(bytes, qword, store, false), bytes, store, bits);
            // Cover the first selected element and a later one after a valid access.
            for preceding in [0, 1] {
                let active = if preceding == 0 {
                    first
                } else {
                    first | (first << (elem * 8))
                };
                boundary_run(
                    &code,
                    SPAN - 2 - preceding * elem as u64,
                    bits,
                    jit,
                    (active, 0),
                    true,
                );
            }
        }
    }
}

macro_rules! unaligned_maskmov_tests {
    ($($name:ident: ($jit:expr, $qword:expr, $store:expr)),* $(,)?) => {$(
        #[test]
        fn $name() {
            vpmaskmov_unaligned_boundary($jit, $qword, $store);
        }
    )*};
}

unaligned_maskmov_tests! {
    vpmaskmov_unaligned_dword_load_interp: (false, false, false),
    vpmaskmov_unaligned_dword_load_jit: (true, false, false),
    vpmaskmov_unaligned_qword_load_interp: (false, true, false),
    vpmaskmov_unaligned_qword_load_jit: (true, true, false),
    vpmaskmov_unaligned_dword_store_interp: (false, false, true),
    vpmaskmov_unaligned_dword_store_jit: (true, false, true),
    vpmaskmov_unaligned_qword_store_interp: (false, true, true),
    vpmaskmov_unaligned_qword_store_jit: (true, true, true),
}
