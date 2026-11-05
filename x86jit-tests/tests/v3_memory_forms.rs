//! VEX memory operands: SDM Vol 2A §2.5 (exception classes); Vol 2B §4.3,
//! PACKSSWB/PACKSSDW, PACKUSWB/PACKUSDW, PALIGNR, PABSB/PABSW/PABSD,
//! PMOVSX/PMOVZX, PSHUFD/PSHUFHW/PSHUFLW, PSLL/PSRL/PSRA, PTEST;
//! Vol 2C §5.2, VCVTPS2PH, VPBLENDD, VPERM2F128/VPERM2I128,
//! VPERMD/VPERMPS, VPERMILPS/VPERMILPD, VPSLLV/VPSRLV/VPSRAV, VTESTPS/VTESTPD.
use iced_x86::{Code, Encoder, Instruction, OpKind, Register};
use x86jit_core::{CpuMode, GuestCpuFeatures, InterpreterBackend};
use x86jit_cranelift::JitBackend;
use x86jit_tests::compare::compare;
use x86jit_tests::oracle::{run_with_backend_mode, RunOutcome, VectorInput};
use x86jit_tests::vector::{CpuSnapshot, ExitKind, MemChunk, MemKind, RunSpec};

#[cfg(all(
    any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
#[path = "../../x86jit-cranelift/src/unwind_test_signal.rs"]
#[allow(dead_code)]
mod signal;

const CODE: u64 = 0x1_0000;
const DATA: u64 = 0x2_0003;
static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn cases() -> Vec<Code> {
    use Code::*;
    vec![
        VEX_Vcvtps2ph_xmmm128_ymm_imm8,
        VEX_Vcvtps2ph_xmmm64_xmm_imm8,
        VEX_Vpabsb_xmm_xmmm128,
        VEX_Vpabsb_ymm_ymmm256,
        VEX_Vpabsd_xmm_xmmm128,
        VEX_Vpabsd_ymm_ymmm256,
        VEX_Vpabsw_xmm_xmmm128,
        VEX_Vpabsw_ymm_ymmm256,
        VEX_Vpackssdw_ymm_ymm_ymmm256,
        VEX_Vpacksswb_ymm_ymm_ymmm256,
        VEX_Vpackusdw_ymm_ymm_ymmm256,
        VEX_Vpackuswb_ymm_ymm_ymmm256,
        VEX_Vpalignr_xmm_xmm_xmmm128_imm8,
        VEX_Vpalignr_ymm_ymm_ymmm256_imm8,
        VEX_Vpblendd_xmm_xmm_xmmm128_imm8,
        VEX_Vpblendd_ymm_ymm_ymmm256_imm8,
        VEX_Vperm2f128_ymm_ymm_ymmm256_imm8,
        VEX_Vperm2i128_ymm_ymm_ymmm256_imm8,
        VEX_Vpermd_ymm_ymm_ymmm256,
        VEX_Vpermilpd_xmm_xmm_xmmm128,
        VEX_Vpermilpd_ymm_ymm_ymmm256,
        VEX_Vpermilps_xmm_xmm_xmmm128,
        VEX_Vpermilps_ymm_ymm_ymmm256,
        VEX_Vpermps_ymm_ymm_ymmm256,
        VEX_Vpmovsxbd_ymm_xmmm64,
        VEX_Vpmovsxbq_ymm_xmmm32,
        VEX_Vpmovsxbw_ymm_xmmm128,
        VEX_Vpmovsxdq_ymm_xmmm128,
        VEX_Vpmovsxwd_ymm_xmmm128,
        VEX_Vpmovsxwq_ymm_xmmm64,
        VEX_Vpmovzxbd_ymm_xmmm64,
        VEX_Vpmovzxbq_ymm_xmmm32,
        VEX_Vpmovzxbw_ymm_xmmm128,
        VEX_Vpmovzxdq_ymm_xmmm128,
        VEX_Vpmovzxwd_ymm_xmmm128,
        VEX_Vpmovzxwq_ymm_xmmm64,
        VEX_Vpshufd_ymm_ymmm256_imm8,
        VEX_Vpshufhw_xmm_xmmm128_imm8,
        VEX_Vpshufhw_ymm_ymmm256_imm8,
        VEX_Vpshuflw_xmm_xmmm128_imm8,
        VEX_Vpshuflw_ymm_ymmm256_imm8,
        VEX_Vpslld_xmm_xmm_xmmm128,
        VEX_Vpslld_ymm_ymm_xmmm128,
        VEX_Vpsllq_xmm_xmm_xmmm128,
        VEX_Vpsllq_ymm_ymm_xmmm128,
        VEX_Vpsllvd_xmm_xmm_xmmm128,
        VEX_Vpsllvd_ymm_ymm_ymmm256,
        VEX_Vpsllvq_xmm_xmm_xmmm128,
        VEX_Vpsllvq_ymm_ymm_ymmm256,
        VEX_Vpsllw_xmm_xmm_xmmm128,
        VEX_Vpsllw_ymm_ymm_xmmm128,
        VEX_Vpsrad_xmm_xmm_xmmm128,
        VEX_Vpsrad_ymm_ymm_xmmm128,
        VEX_Vpsravd_xmm_xmm_xmmm128,
        VEX_Vpsravd_ymm_ymm_ymmm256,
        VEX_Vpsraw_xmm_xmm_xmmm128,
        VEX_Vpsraw_ymm_ymm_xmmm128,
        VEX_Vpsrld_xmm_xmm_xmmm128,
        VEX_Vpsrld_ymm_ymm_xmmm128,
        VEX_Vpsrlq_xmm_xmm_xmmm128,
        VEX_Vpsrlq_ymm_ymm_xmmm128,
        VEX_Vpsrlvd_xmm_xmm_xmmm128,
        VEX_Vpsrlvd_ymm_ymm_ymmm256,
        VEX_Vpsrlvq_xmm_xmm_xmmm128,
        VEX_Vpsrlvq_ymm_ymm_ymmm256,
        VEX_Vpsrlw_xmm_xmm_xmmm128,
        VEX_Vpsrlw_ymm_ymm_xmmm128,
        VEX_Vptest_xmm_xmmm128,
        VEX_Vptest_ymm_ymmm256,
        VEX_Vtestpd_xmm_xmmm128,
        VEX_Vtestpd_ymm_ymmm256,
        VEX_Vtestps_xmm_xmmm128,
        VEX_Vtestps_ymm_ymmm256,
    ]
}

fn narrow_cases() -> Vec<Code> {
    use Code::*;
    vec![
        VEX_Vpmovsxbd_xmm_xmmm32,
        VEX_Vpmovsxbq_xmm_xmmm16,
        VEX_Vpmovsxbw_xmm_xmmm64,
        VEX_Vpmovsxdq_xmm_xmmm64,
        VEX_Vpmovsxwd_xmm_xmmm64,
        VEX_Vpmovsxwq_xmm_xmmm32,
        VEX_Vpmovzxbd_xmm_xmmm32,
        VEX_Vpmovzxbq_xmm_xmmm16,
        VEX_Vpmovzxbw_xmm_xmmm64,
        VEX_Vpmovzxdq_xmm_xmmm64,
        VEX_Vpmovzxwd_xmm_xmmm64,
        VEX_Vpmovzxwq_xmm_xmmm32,
    ]
}

fn instruction(code: Code, memory: bool, bits: u32, imm: u8, alias: bool) -> Instruction {
    let mut i = Instruction::default();
    i.set_code(code);
    let name = format!("{code:?}");
    for (op, part) in name.split('_').skip(2).enumerate() {
        let op = op as u32;
        if part == "imm8" {
            i.set_op_kind(op, OpKind::Immediate8);
            i.set_immediate8(imm);
        } else if part.contains("mmm") && memory {
            i.set_op_kind(op, OpKind::Memory);
            i.set_memory_base(if bits == 64 {
                Register::RAX
            } else {
                Register::EAX
            });
        } else {
            let reg = if op == 0 {
                0
            } else if part.contains("mmm") {
                2
            } else if alias {
                0
            } else {
                1
            };
            let base = if part.starts_with("ymm") {
                Register::YMM0
            } else {
                Register::XMM0
            };
            i.set_op_register(op, Register::try_from(base as usize + reg).unwrap());
        }
    }
    i
}

fn encode(i: &Instruction, bits: u32) -> Vec<u8> {
    let mut e = Encoder::new(bits);
    e.encode(i, CODE).unwrap();
    let mut bytes = e.take_buffer();
    bytes.push(0xf4);
    bytes
}

fn operand_cases(code: Code) -> Vec<(u8, u128)> {
    let mut inputs = vec![
        (0, 0),
        (0xa5, 0x80017fff0100fffe_0000000000000011),
        (0xff, u128::MAX),
        (0x53, 0x0302010007060504_8000000040000000),
    ];
    if matches!(
        code,
        Code::VEX_Vpalignr_xmm_xmm_xmmm128_imm8 | Code::VEX_Vpalignr_ymm_ymm_ymmm256_imm8
    ) {
        inputs.extend([1, 15, 16, 17, 31].map(|imm| (imm, 0xafaeadacabaaa9a8_a7a6a5a4a3a2a1a0)));
    }
    inputs
}

fn input(code: Code, memory: bool, bits: u32, imm: u8, alias: bool, seed: u128) -> VectorInput {
    let mut cpu = CpuSnapshot {
        rip: CODE,
        ..Default::default()
    };
    cpu.gpr[0] = DATA;
    for n in 0..16 {
        cpu.xmm[n] = seed.rotate_left(n as u32 * 7);
        cpu.ymm_hi[n] = seed.rotate_left(n as u32 * 11) ^ u128::MAX;
    }
    // Exact finite conversions keep MXCSR exception flags outside this operand-form test.
    if format!("{code:?}").contains("Vcvtps2ph") {
        for n in 0..3 {
            cpu.xmm[n] = 0x3f80000040000000_4040000040800000;
            cpu.ymm_hi[n] = 0x4080000040400000_400000003f800000;
        }
    }
    cpu.flags.cf = true;
    cpu.flags.of = true;
    let mut data = cpu.xmm[2].to_le_bytes().to_vec();
    data.extend_from_slice(&cpu.ymm_hi[2].to_le_bytes());
    VectorInput {
        cpu_init: cpu,
        mem_init: vec![
            MemChunk {
                addr: CODE,
                bytes: encode(&instruction(code, memory, bits, imm, alias), bits),
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

fn normalize(mut r: RunOutcome) -> RunOutcome {
    r.cpu.rip = 0;
    r.mem.retain(|m| m.addr != CODE);
    r
}

fn assert_same(a: RunOutcome, b: RunOutcome, context: &str) {
    let (a, b) = (normalize(a), normalize(b));
    assert_eq!(a.exit, ExitKind::Hlt, "{context}: {a:?}");
    assert_eq!(b.exit, ExitKind::Hlt, "{context}: {b:?}");
    assert!(
        compare(&a, &b, &[]).is_none(),
        "{context}: {:?}",
        compare(&a, &b, &[])
    );
}

#[test]
fn v3_memory_forms_match_register_and_jit() {
    let _serial = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let mut missing = Vec::new();
    for bits in [64, 32] {
        let mode = if bits == 64 {
            CpuMode::Long64
        } else {
            CpuMode::Compat32
        };
        for code in cases() {
            for alias in [false, true] {
                for (imm, seed) in operand_cases(code) {
                    let reg = input(code, false, bits, imm, alias, seed);
                    let mem = input(code, true, bits, imm, alias, seed);
                    let run = |i: &VectorInput, jit| {
                        run_with_backend_mode(
                            i,
                            if jit {
                                Box::new(JitBackend::new())
                            } else {
                                Box::new(InterpreterBackend)
                            },
                            GuestCpuFeatures::default(),
                            mode,
                        )
                    };
                    let m = run(&mem, false);
                    let j = run(&mem, true);
                    if m.exit != ExitKind::Hlt || j.exit != ExitKind::Hlt {
                        missing.push(format!(
                            "{code:?} bits={bits} interp={:?} jit={:?}",
                            m.exit, j.exit
                        ));
                        break;
                    }
                    let mut r = run(&reg, false);
                    let jr = run(&reg, true);
                    assert_same(r.clone(), jr, &format!("register {code:?}"));
                    if format!("{code:?}").contains("Vcvtps2ph") {
                        let size = instruction(code, true, bits, imm, alias)
                            .memory_size()
                            .size();
                        r.mem[1].bytes[..size].copy_from_slice(&r.cpu.xmm[0].to_le_bytes()[..size]);
                        r.cpu.xmm[0] = reg.cpu_init.xmm[0];
                        r.cpu.ymm_hi[0] = reg.cpu_init.ymm_hi[0];
                    }
                    assert_same(
                        r,
                        m.clone(),
                        &format!("memory {code:?} bits={bits} alias={alias} imm={imm}"),
                    );
                    assert_same(
                        m,
                        j,
                        &format!("jit {code:?} bits={bits} alias={alias} imm={imm}"),
                    );
                }
            }
        }
    }
    assert!(
        missing.is_empty(),
        "unlifted memory forms ({}):\n{}",
        missing.len(),
        missing.join("\n")
    );
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn v3_memory_forms_match_native() {
    let _serial = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    use x86jit_tests::native::run_native;
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("f16c") {
        return;
    }
    for code in cases() {
        for alias in [false, true] {
            for (imm, seed) in operand_cases(code) {
                let i = input(code, true, 64, imm, alias, seed);
                let native = run_native(&i).expect("native memory form must execute");
                let interp = run_with_backend_mode(
                    &i,
                    Box::new(InterpreterBackend),
                    GuestCpuFeatures::default(),
                    CpuMode::Long64,
                );
                assert_same(
                    native,
                    interp,
                    &format!("native {code:?} imm={imm} alias={alias}"),
                );
            }
        }
    }
}

#[test]
fn v3_palignr_memory_source1_witness() {
    let _serial = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    // SDM Vol 2B §4.3, PALIGNR: each lane concatenates SRC1 above SRC2,
    // shifts by imm8 bytes, and takes the low 16 bytes (zero past byte 31).
    for code in [
        Code::VEX_Vpalignr_xmm_xmm_xmmm128_imm8,
        Code::VEX_Vpalignr_ymm_ymm_ymmm256_imm8,
    ] {
        for bits in [64, 32] {
            for imm in [1, 15, 16, 17, 31, 32] {
                for alias in [false, true] {
                    let mut i = input(code, true, bits, imm, alias, 0);
                    i.cpu_init.xmm[0] =
                        u128::from_le_bytes(std::array::from_fn(|n| 0x40 + n as u8));
                    i.cpu_init.ymm_hi[0] =
                        u128::from_le_bytes(std::array::from_fn(|n| 0x50 + n as u8));
                    i.cpu_init.xmm[1] =
                        u128::from_le_bytes(std::array::from_fn(|n| 0xa0 + n as u8));
                    i.cpu_init.ymm_hi[1] =
                        u128::from_le_bytes(std::array::from_fn(|n| 0xb0 + n as u8));
                    i.mem_init[1].bytes = (0u8..32).collect();
                    let width = instruction(code, true, bits, imm, alias)
                        .memory_size()
                        .size();
                    let mut expected = Vec::new();
                    for lane in 0..width / 16 {
                        let first = if alias { 0x40 } else { 0xa0 } + lane as u8 * 16;
                        for byte in 0..16 {
                            let index = usize::from(imm) + byte;
                            expected.push(match index {
                                0..=15 => (lane * 16 + index) as u8,
                                16..=31 => first + (index - 16) as u8,
                                _ => 0,
                            });
                        }
                    }
                    let check = |r: &RunOutcome| {
                        assert_eq!(r.exit, ExitKind::Hlt, "{code:?}");
                        let got =
                            [r.cpu.xmm[0].to_le_bytes(), r.cpu.ymm_hi[0].to_le_bytes()].concat();
                        assert_eq!(
                            &got[..width],
                            expected,
                            "{code:?} bits={bits} imm={imm} alias={alias}"
                        );
                    };
                    for jit in [false, true] {
                        let r = run_with_backend_mode(
                            &i,
                            if jit {
                                Box::new(JitBackend::new())
                            } else {
                                Box::new(InterpreterBackend)
                            },
                            GuestCpuFeatures::default(),
                            if bits == 64 {
                                CpuMode::Long64
                            } else {
                                CpuMode::Compat32
                            },
                        );
                        check(&r);
                    }
                    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
                    if bits == 64 && std::is_x86_feature_detected!("avx2") {
                        check(
                            &x86jit_tests::native::run_native(&i).expect("native PALIGNR witness"),
                        );
                    }
                }
            }
        }
    }
}

fn machine_state(cpu: &x86jit_core::CpuState) -> Vec<u128> {
    let mut out: Vec<_> = cpu.gpr.iter().map(|&v| u128::from(v)).collect();
    out.push(u128::from(cpu.flags.to_rflags()));
    out.push(u128::from(cpu.mxcsr));
    for i in 0..32 {
        out.extend(cpu.vec_lanes(i));
    }
    out.extend(cpu.kmask.iter().map(|&v| u128::from(v)));
    out
}

const SPAN: u64 = 0x3_0000;

// Two blocks with an untaken back edge force region compilation. PUSHF/POPF
// preserve the seeded flags across the loop condition before the tested operand.
fn region_code(i: &Instruction, bits: u32) -> Vec<u8> {
    let mut bytes = vec![0x9c, 0x83, 0xf9, 0, 0x74, 0xfa, 0x9d];
    let mut op = encode(i, bits);
    op.pop();
    bytes.extend(op);
    bytes.push(0xf4);
    bytes
}

fn boundary_run(
    code: Code,
    bits: u32,
    tier: u8,
    available: usize,
    alias: bool,
    protected: bool,
) -> (Vec<u128>, Vec<u8>) {
    use x86jit_core::{Backend, Exit, Prot, Reg, RegionCaps, RegionKind, Vm, VmConfig};
    let backend: Box<dyn Backend> = match tier {
        0 => Box::new(InterpreterBackend),
        1 => Box::new(JitBackend::new()),
        _ => Box::new(JitBackend::with_superblocks(RegionCaps {
            max_blocks: 8,
            max_icount: 128,
        })),
    };
    let mut vm = Vm::with_backend(VmConfig::flat(SPAN), backend);
    // SAFETY: sysconf has no pointer arguments.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let end = if protected {
        SPAN - 2 * page as u64
    } else {
        SPAN
    };
    vm.mem.set_host_fault_recovery(protected);
    vm.set_cpu_mode(if bits == 64 {
        CpuMode::Long64
    } else {
        CpuMode::Compat32
    });
    vm.map(0, SPAN as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    let i = instruction(code, true, bits, 0xa5, alias);
    let width = i.memory_size().size();
    let op = if tier == 2 {
        region_code(&i, bits)
    } else {
        encode(&i, bits)
    };
    vm.write_bytes(CODE, &op).unwrap();
    let source = input(
        code,
        true,
        bits,
        0xa5,
        alias,
        0x80017fff0100fffe_0000000000000011,
    );
    vm.write_bytes(end - 64, &[0x40; 64]).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.gpr = source.cpu_init.gpr;
    cpu.cpu.gpr[0] = end - available as u64;
    cpu.cpu.gpr[1] = 1;
    cpu.cpu.gpr[4] = SPAN - 0x100;
    cpu.cpu.xmm = source.cpu_init.xmm;
    cpu.cpu.ymm_hi = source.cpu_init.ymm_hi;
    cpu.cpu.flags = source.cpu_init.flags.into();
    // v3 exposes YMM. Dirty the hidden upper state on faults to detect any early commit.
    if available < width {
        for n in 0..32 {
            cpu.cpu.zmm_hi[n] = [0x1234 + n as u128, 0x5678 + n as u128];
        }
    }
    cpu.set_reg(Reg::Rip, CODE);
    let before = machine_state(&cpu.cpu);
    let mut before_mem = vec![0; 64];
    vm.read_bytes(end - 64, &mut before_mem).unwrap();
    if protected {
        // SAFETY: the page is aligned, in this VM's live mapping, and restored below.
        assert_eq!(
            unsafe {
                libc::mprotect(
                    vm.mem.host_base().add(end as usize).cast_mut().cast(),
                    page,
                    libc::PROT_NONE,
                )
            },
            0
        );
    }
    #[cfg(all(
        any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    if protected {
        signal::HANDLE.store(
            cpu.fault_handle().as_usize(),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    let exit = cpu.run(&vm, Some(16));
    if protected {
        // SAFETY: restore the same live page before reading or dropping the mapping.
        assert_eq!(
            unsafe {
                libc::mprotect(
                    vm.mem.host_base().add(end as usize).cast_mut().cast(),
                    page,
                    libc::PROT_READ | libc::PROT_WRITE,
                )
            },
            0
        );
    }
    let mut after_mem = vec![0; 64];
    vm.read_bytes(end - 64, &mut after_mem).unwrap();
    let context = format!("{code:?} bits={bits} tier={tier} available={available} alias={alias}");
    if available < width {
        assert!(
            matches!(exit, Exit::UnmappedMemory { addr, .. } if addr == end),
            "{context}: {exit:?}"
        );
        assert_eq!(
            cpu.reg(Reg::Rip),
            CODE + if tier == 2 { 7 } else { 0 },
            "{context}"
        );
        assert_eq!(
            machine_state(&cpu.cpu),
            before,
            "{context}: fault changed registers or flags"
        );
        assert_eq!(after_mem, before_mem, "{context}: fault committed a store");
    } else {
        assert!(matches!(exit, Exit::Hlt), "{context}: {exit:?}");
    }
    if tier == 2 {
        assert!(
            vm.cache.regions() > 0,
            "{context}: did not execute compiled region"
        );
    }
    (machine_state(&cpu.cpu), after_mem)
}

#[test]
fn v3_memory_forms_exact_width_and_fault_atomicity() {
    let _serial = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    // SDM Vol 2A §2.5: these VEX forms allow unaligned addresses. A fully backed
    // operand succeeds; an operand one byte short faults at the first unavailable byte.
    // Vol 2B §4.3, PSLL/PSRL/PSRA: uniform counts are always m128, including YMM data.
    for bits in [64, 32] {
        for code in cases().into_iter().chain(narrow_cases()) {
            let width = instruction(code, true, bits, 0, false).memory_size().size();
            for alias in [false, true] {
                for available in [width, width + 1, width - 1, 1, 0x1000 + width / 2] {
                    let expected = boundary_run(code, bits, 0, available, alias, false);
                    for tier in [1, 2] {
                        assert_eq!(
                            boundary_run(code, bits, tier, available, alias, false),
                            expected,
                            "{code:?} bits={bits} tier={tier}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn v3_pshuflw_memory_witness() {
    let _serial = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    // SDM Vol 2B §4.3, PSHUFLW, Operation (VEX.128/VEX.256): imm8 selects four
    // low words independently in each 128-bit lane; the high qword is copied.
    for code in [
        Code::VEX_Vpshuflw_xmm_xmmm128_imm8,
        Code::VEX_Vpshuflw_ymm_ymmm256_imm8,
    ] {
        let mut i = input(code, true, 64, 0x1b, false, 0);
        i.mem_init[1].bytes = (0u16..16).flat_map(u16::to_le_bytes).collect();
        for jit in [false, true] {
            let r = run_with_backend_mode(
                &i,
                if jit {
                    Box::new(JitBackend::new())
                } else {
                    Box::new(InterpreterBackend)
                },
                GuestCpuFeatures::default(),
                CpuMode::Long64,
            );
            assert_eq!(r.exit, ExitKind::Hlt);
            assert_eq!(r.cpu.xmm[0], 0x0007_0006_0005_0004_0000_0001_0002_0003);
            assert_eq!(
                r.cpu.ymm_hi[0],
                if code == Code::VEX_Vpshuflw_ymm_ymmm256_imm8 {
                    0x000f_000e_000d_000c_0008_0009_000a_000b
                } else {
                    0
                }
            );
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn v3_memory_forms_fault_match_native() {
    let _serial = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    use x86jit_tests::native::{run_native_fault, NativeFault};
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("f16c") {
        return;
    }
    const END: u64 = 0x3_1000;
    for code in cases() {
        let mut i = input(
            code,
            true,
            64,
            0xa5,
            true,
            0x80017fff0100fffe_0000000000000011,
        );
        let width = instruction(code, true, 64, 0, true).memory_size().size();
        i.cpu_init.gpr[0] = END - width as u64 + 1;
        i.mem_init[1].addr = END - 0x1000;
        i.mem_init[1].bytes = vec![0x40; 0x1000];
        let native = run_native_fault(&i, NativeFault::PageFault { addr: END })
            .expect("native memory operand page fault");
        let interp = run_with_backend_mode(
            &i,
            Box::new(InterpreterBackend),
            GuestCpuFeatures::default(),
            CpuMode::Long64,
        );
        let (native, interp) = (normalize(native), normalize(interp));
        assert!(
            compare(&native, &interp, &[]).is_none(),
            "{code:?}: {:?}",
            compare(&native, &interp, &[])
        );
    }
}

#[cfg(all(
    any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
#[test]
fn v3_memory_forms_protected_page_faults_commit_nothing() {
    let _serial = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    // SDM Vol 3A §7.5: a fault restarts at the instruction with no partial result.
    // Exercise both halves of a vector load, and the narrowing conversion store.
    for code in [
        Code::VEX_Vpshuflw_ymm_ymmm256_imm8,
        Code::VEX_Vpblendd_ymm_ymm_ymmm256_imm8,
        Code::VEX_Vpsllw_ymm_ymm_xmmm128,
        Code::VEX_Vcvtps2ph_xmmm128_ymm_imm8,
    ] {
        let width = instruction(code, true, 64, 0, true).memory_size().size();
        for available in [width - 1, width / 2] {
            for tier in [0, 1, 2] {
                assert_eq!(
                    signal::child(|| {
                        // SAFETY: fault forwarding is installed only in this isolated child.
                        unsafe {
                            signal::install();
                        }
                        boundary_run(code, 64, tier, available, true, true);
                    }),
                    0,
                    "{code:?} tier={tier} available={available}"
                );
            }
        }
    }
}

#[cfg(feature = "unicorn")]
#[test]
fn v3_memory_forms_unicorn_supported_xmm_forms() {
    use x86jit_tests::oracle::Oracle;
    use x86jit_tests::unicorn::UnicornOracle;
    let _serial = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    // The wrapper captures XMM only. YMM forms use the native oracle instead.
    let mut supported = 0;
    let mut unsupported = 0;
    for code in cases()
        .into_iter()
        .filter(|c| !format!("{c:?}").contains("ymm"))
    {
        // Unicorn's VCVTPS2PH m64 store reports UnmappedMemory { addr: 0x10000,
        // access: Read } instead of Hlt. CODE (0x10000) and DATA (0x20003) are
        // both mapped by the wrapper; its error fallback uses RIP as the address,
        // so this is not evidence of an unmapped fixture page. Keep native coverage.
        if code == Code::VEX_Vcvtps2ph_xmmm64_xmm_imm8 {
            unsupported += 1;
            continue;
        }
        let mut i = input(
            code,
            true,
            64,
            0xa5,
            true,
            0x80017fff0100fffe_0000000000000011,
        );
        i.cpu_init.ymm_hi.fill(0);
        let u = UnicornOracle.run(&i);
        // Measured: for forms its CPU model rejects (for example VPBLENDD and
        // the VCVTPS2PH m64 store), the wrapper reports UnmappedMemory { addr:
        // 0x10000, access: Read }. 0x10000 is CODE, which is mapped; the
        // wrapper's error fallback uses RIP as the address. Treat it as an
        // unsupported form, like #UD; the native oracle still covers it.
        if matches!(
            u.exit,
            ExitKind::UnknownInstruction { .. } | ExitKind::Exception { vector: 6, .. }
        ) || matches!(u.exit, ExitKind::UnmappedMemory { addr, .. } if addr == CODE)
        {
            unsupported += 1;
            continue;
        }
        let m = run_with_backend_mode(
            &i,
            Box::new(InterpreterBackend),
            GuestCpuFeatures::default(),
            CpuMode::Long64,
        );
        assert_same(u, m, &format!("Unicorn {code:?}"));
        supported += 1;
    }
    eprintln!("Unicorn XMM forms: {supported} supported, {unsupported} unsupported");
}
