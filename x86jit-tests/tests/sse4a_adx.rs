//! AMD APM Vol 4, EXTRQ/INSERTQ/MOVNTSD/MOVNTSS; SDM Vol 2A §3.3, ADCX/ADOX.

use iced_x86::code_asm::*;
use x86jit_core::{
    Backend, CpuMode, Exit, InterpreterBackend, Prot, Reg, RegionCaps, RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;
#[cfg(feature = "unicorn")]
use x86jit_tests::oracle::Oracle;
use x86jit_tests::oracle::VectorInput;
use x86jit_tests::vector::{CpuSnapshot, MemChunk, MemKind, RunSpec, SnapFlags};

const CODE: u64 = 0x10000;
const DATA: u64 = 0x20000;
const FAULT: u64 = 0x40000;
const LOW: u64 = 0xfedc_ba98_7654_3210;
const HIGH: u64 = 0x1234_5678_9abc_def0;

fn input(bytes: Vec<u8>) -> VectorInput {
    let mut cpu_init = CpuSnapshot {
        flags: SnapFlags::from_rflags(0xcd7),
        ..CpuSnapshot::default()
    };
    cpu_init.xmm[0] = ((HIGH as u128) << 64) | LOW as u128;
    cpu_init.ymm_hi[0] = u128::MAX;
    cpu_init.gpr[6] = DATA;
    VectorInput {
        cpu_init,
        mem_init: vec![
            MemChunk {
                addr: CODE,
                bytes,
                kind: MemKind::Ram,
            },
            MemChunk {
                addr: DATA,
                bytes: vec![0xa5; 4096],
                kind: MemKind::Ram,
            },
        ],
        entry: CODE,
        run: RunSpec::UntilExit,
    }
}

fn code(bits: u32, build: impl FnOnce(&mut CodeAssembler)) -> Vec<u8> {
    let mut a = CodeAssembler::new(bits).unwrap();
    let mut top = a.create_label();
    let mut body = a.create_label();
    a.set_label(&mut top).unwrap();
    a.jmp(body).unwrap();
    a.nop().unwrap();
    a.set_label(&mut body).unwrap();
    build(&mut a);
    // All witnesses preserve ZF=1; the backedge makes this a region candidate.
    a.jne(top).unwrap();
    a.hlt().unwrap();
    a.assemble(CODE).unwrap()
}

fn engines(input: &VectorInput, mode: CpuMode, check: impl Fn(&x86jit_core::Vcpu, &Vm, &Exit)) {
    for tier in 0..3 {
        let backend: Box<dyn Backend> = match tier {
            0 => Box::new(InterpreterBackend),
            1 => Box::new(JitBackend::new()),
            _ => Box::new(JitBackend::with_superblocks(RegionCaps {
                max_blocks: 16,
                max_icount: 256,
            })),
        };
        let mut vm = Vm::with_backend(VmConfig::flat(0x40000), backend);
        vm.set_cpu_mode(mode);
        for chunk in &input.mem_init {
            vm.map(chunk.addr, 4096, Prot::RWX, RegionKind::Ram)
                .unwrap();
            vm.write_bytes(chunk.addr, &chunk.bytes).unwrap();
        }
        let mut cpu = vm.new_vcpu();
        for (i, &v) in input.cpu_init.gpr.iter().enumerate() {
            cpu.set_reg(Reg::from_gpr_index(i), v);
        }
        cpu.set_flags(input.cpu_init.flags.into());
        for i in 0..8 {
            cpu.set_xmm(i, input.cpu_init.xmm[i]);
            cpu.set_ymm_hi(i, input.cpu_init.ymm_hi[i]);
        }
        cpu.set_reg(Reg::Rip, CODE);
        let exit = cpu.run(&vm, None);
        check(&cpu, &vm, &exit);
        if tier == 2 {
            assert!(vm.cache.regions() > 0, "region was not compiled");
        }
    }
}

#[cfg(feature = "unicorn")]
fn unicorn_sse4a_available(mode: CpuMode) -> bool {
    use std::sync::OnceLock;
    use unicorn_engine::unicorn_const::{Arch, Mode, Prot};
    use unicorn_engine::{uc_error, RegisterX86, Unicorn};
    static SUPPORTED: [OnceLock<bool>; 2] = [OnceLock::new(), OnceLock::new()];
    let bits32 = mode == CpuMode::Compat32;
    *SUPPORTED[usize::from(bits32)].get_or_init(|| {
        let mut uc = Unicorn::new(
            Arch::X86,
            if bits32 { Mode::MODE_32 } else { Mode::MODE_64 },
        )
        .unwrap();
        uc.mem_map(CODE, 4096, Prot::ALL).unwrap();
        let bytes = [0x66, 0x0f, 0x78, 0xc0, 8, 8];
        uc.mem_write(CODE, &bytes).unwrap();
        uc.reg_write_long(RegisterX86::XMM0, &(LOW as u128).to_le_bytes())
            .unwrap();
        let result = uc.emu_start(CODE, CODE + bytes.len() as u64, 0, 1);
        eprintln!("Unicorn default CPU ({mode:?}), EXTRQ 66 0f 78 c0 08 08: {result:?}");
        match result {
            Ok(()) => {
                let actual = uc.reg_read_long(RegisterX86::XMM0).unwrap();
                assert_eq!(u64::from_le_bytes(actual[..8].try_into().unwrap()), 0x32);
                true
            }
            Err(uc_error::INSN_INVALID) => {
                eprintln!("SSE4A oracle skipped: Unicorn CPU rejects SSE4A ({mode:?})");
                false
            }
            Err(error) => panic!("Unicorn SSE4A probe failed: {error:?}"),
        }
    })
}

#[cfg(feature = "unicorn")]
#[test]
fn sse4a_unicorn_cpu_model_probe() {
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        let _ = unicorn_sse4a_available(mode);
    }
}

// Only architecturally defined bits are compared with an oracle. Undefined bit
// fields are witnessed against the engine's documented deterministic policy.
fn bitfield_oracles(
    input: &VectorInput,
    mode: CpuMode,
    expected: u64,
    defined: bool,
    unicorn_divergent: bool,
) {
    #[cfg(feature = "unicorn")]
    if unicorn_sse4a_available(mode) {
        use x86jit_tests::unicorn::{UnicornOracle, UnicornOracle32};
        let oracle: &dyn Oracle = if mode == CpuMode::Long64 {
            &UnicornOracle
        } else {
            &UnicornOracle32
        };
        let result = oracle.run(input);
        assert_eq!(result.exit, x86jit_tests::vector::ExitKind::Hlt);
        if unicorn_divergent {
            // Unicorn diverges from the APM here: see the caller.
            eprintln!(
                "SSE4A Unicorn divergence: oracle={:032x}, engine low={expected:016x}",
                result.cpu.xmm[0]
            );
        } else if defined {
            assert_eq!(result.cpu.xmm[0] as u64, expected);
        } else {
            eprintln!(
                "SSE4A undefined result: oracle={:032x}, engine low={expected:016x}",
                result.cpu.xmm[0]
            );
        }
        assert_eq!(result.cpu.flags, input.cpu_init.flags);
    }
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    if mode == CpuMode::Long64 {
        // AMD APM Vol 3 §E.4.2: Fn8000_0001_ECX[6] advertises SSE4A.
        // SAFETY: CPUID is available on x86-64; the leaf is checked first.
        #[allow(unused_unsafe)] // CPUID intrinsics are unsafe on the workspace MSRV.
        let supported = unsafe {
            std::arch::x86_64::__cpuid(0x8000_0000).eax >= 0x8000_0001
                && std::arch::x86_64::__cpuid(0x8000_0001).ecx & (1 << 6) != 0
        };
        if supported {
            let Some(result) = x86jit_tests::native::run_native(input) else {
                eprintln!("native SSE4A witness skipped: native oracle unavailable");
                return;
            };
            assert_eq!(result.exit, x86jit_tests::vector::ExitKind::Hlt);
            if defined {
                assert_eq!(result.cpu.xmm[0] as u64, expected);
            } else {
                eprintln!(
                    "SSE4A undefined result: oracle={:032x}, engine low={expected:016x}",
                    result.cpu.xmm[0]
                );
            }
            assert_eq!(result.cpu.flags, input.cpu_init.flags);
        } else {
            eprintln!("native SSE4A witness skipped: CPUID lacks SSE4A");
        }
    }
    let _ = (input, mode, expected, defined, unicorn_divergent);
}

#[test]
fn sse4a_bitfields_controls_edges_and_undefined_policy() {
    // AMD APM Vol 4, EXTRQ/INSERTQ: zero length means 64; six-bit
    // control fields, upper quadword and overrun results are undefined.
    // Engine policy: preserve upper quadword; shift then mask EXTRQ and
    // truncate INSERTQ's shifted mask/data to 64 bits, including overruns.
    for (bits, mode) in [(64, CpuMode::Long64), (32, CpuMode::Compat32)] {
        for insert in [false, true] {
            for register in [false, true] {
                for (length, index, expected) in [
                    (0u8, 0u8, if insert { 0x0123_4567_89ab_cdef } else { LOW }),
                    (
                        16,
                        48,
                        if insert {
                            0xcdef_ba98_7654_3210
                        } else {
                            0xfedc
                        },
                    ),
                    (8, 8, if insert { 0xfedc_ba98_7654_ef10 } else { 0x32 }),
                    (16, 60, if insert { 0xfedc_ba98_7654_3210 } else { 0xf }),
                    (
                        0,
                        8,
                        if insert {
                            0x2345_6789_abcd_ef10
                        } else {
                            0x00fe_dcba_9876_5432
                        },
                    ),
                ] {
                    let bytes = code(bits, |a| {
                        match (insert, register) {
                            (false, false) => {
                                a.extrq_3(xmm0, u32::from(length | 0xc0), u32::from(index | 0xc0))
                            }
                            (false, true) => a.extrq(xmm0, xmm1),
                            (true, false) => a.insertq_4(
                                xmm0,
                                xmm1,
                                u32::from(length | 0xc0),
                                u32::from(index | 0xc0),
                            ),
                            (true, true) => a.insertq(xmm0, xmm1),
                        }
                        .unwrap();
                    });
                    let mut input = input(bytes);
                    let controls =
                        0xffff_ffff_ffff_c0c0 | u64::from(length) | (u64::from(index) << 8);
                    input.cpu_init.xmm[1] = if insert {
                        ((controls as u128) << 64) | 0x0123_4567_89ab_cdef
                    } else {
                        ((u64::MAX as u128) << 64) | controls as u128
                    };
                    engines(&input, mode, |cpu, _, exit| {
                        assert!(
                            matches!(exit, Exit::Hlt),
                            "{exit:?}, {insert}/{register}/{length}/{index}"
                        );
                        assert_eq!(cpu.xmm(0), ((HIGH as u128) << 64) | expected as u128);
                        assert_eq!(cpu.ymm_hi(0), u128::MAX);
                        assert_eq!(SnapFlags::from(cpu.flags()), input.cpu_init.flags);
                        assert_eq!(cpu.xmm(1), input.cpu_init.xmm[1]);
                    });
                    let defined =
                        (if length == 0 { 64 } else { u16::from(length) }) + u16::from(index) <= 64;
                    // Measured: Unicorn disagrees with AMD APM Vol 4 here. Its
                    // INSERTQ (both forms) differs even for defined controls, e.g.
                    // length 16 at index 48 gives 0x3210000000000000 where the APM
                    // gives 0xcdefba9876543210, and its register-form EXTRQ returns
                    // 0 for a zero length, which the APM defines as 64. Its
                    // immediate EXTRQ agrees with the APM. The APM wins; Unicorn's
                    // value is printed, not compared, and the native SSE4A witness
                    // still compares every defined case.
                    let unicorn_divergent = insert || (register && length == 0);
                    bitfield_oracles(&input, mode, expected, defined, unicorn_divergent);
                }
            }
        }
        for (insert, register, expected) in [
            (false, true, 8),
            (true, true, 0xfedc_ba98_7608_0808),
            (true, false, 0xfedc_ba98_7608_0808),
        ] {
            let bytes = code(bits, |a| {
                match (insert, register) {
                    (false, true) => a.extrq(xmm0, xmm0),
                    (true, true) => a.insertq(xmm0, xmm0),
                    (true, false) => a.insertq_4(xmm0, xmm0, 8u32, 16u32),
                    _ => unreachable!(),
                }
                .unwrap();
            });
            let mut input = input(bytes);
            input.cpu_init.xmm[0] = (0x1234_5678_9abc_1008u128 << 64) | 0xfedc_ba98_7654_0808;
            // EXTRQ controls in the low word: length=8, index=8.
            // INSERTQ controls in the high word: length=8, index=16.
            engines(&input, mode, |cpu, _, exit| {
                assert!(matches!(exit, Exit::Hlt), "{exit:?}");
                assert_eq!(
                    cpu.xmm(0),
                    (input.cpu_init.xmm[0] & (!0u128 << 64)) | expected as u128
                );
                assert_eq!(SnapFlags::from(cpu.flags()), input.cpu_init.flags);
                assert_eq!(cpu.ymm_hi(0), u128::MAX);
            });
            bitfield_oracles(&input, mode, expected, true, false);
        }
    }
}

#[test]
fn sse4a_streaming_scalar_store_width_and_fault() {
    // AMD APM Vol 4, MOVNTSD/MOVNTSS: m64/m32 stores, no flags affected.
    for (bits, mode) in [(64, CpuMode::Long64), (32, CpuMode::Compat32)] {
        for size in [4, 8] {
            for fault in [false, true] {
                let bytes = code(bits, |a| {
                    if size == 4 {
                        a.movntss(dword_ptr(esi), xmm0)
                    } else {
                        a.movntsd(qword_ptr(esi), xmm0)
                    }
                    .unwrap();
                });
                let mut input = input(bytes);
                input.cpu_init.gpr[6] = if fault { FAULT } else { DATA + 7 };
                engines(&input, mode, |cpu, vm, exit| {
                    if fault {
                        assert!(
                            matches!(exit, Exit::UnmappedMemory { addr: FAULT, .. }),
                            "{exit:?}"
                        );
                    } else {
                        assert!(matches!(exit, Exit::Hlt), "{exit:?}");
                        let mut actual = [0; 24];
                        vm.read_bytes(DATA, &mut actual).unwrap();
                        let mut expected = [0xa5; 24];
                        expected[7..7 + size].copy_from_slice(&LOW.to_le_bytes()[..size]);
                        assert_eq!(actual, expected);
                    }
                    assert_eq!(cpu.xmm(0), input.cpu_init.xmm[0]);
                    assert_eq!(SnapFlags::from(cpu.flags()), input.cpu_init.flags);
                });
            }
        }
    }
}

fn adx(a: &mut CodeAssembler, overflow: bool, size: usize, memory: bool) {
    match (overflow, size, memory) {
        (false, 4, false) => a.adcx(eax, ecx),
        (false, 4, true) => a.adcx(eax, dword_ptr(esi)),
        (false, 8, false) => a.adcx(rax, rcx),
        (false, 8, true) => a.adcx(rax, qword_ptr(esi)),
        (true, 4, false) => a.adox(ebx, ecx),
        (true, 4, true) => a.adox(ebx, dword_ptr(esi)),
        (true, 8, false) => a.adox(rbx, rcx),
        (true, 8, true) => a.adox(rbx, qword_ptr(esi)),
        _ => unreachable!(),
    }
    .unwrap();
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn native_adx(input: &VectorInput) -> Option<x86jit_tests::oracle::RunOutcome> {
    // SDM Vol 2A §3.3 ADCX/ADOX: CPUID.07H.00H:EBX[19].
    // SAFETY: CPUID is available on x86-64; maximum basic leaf is checked first.
    #[allow(unused_unsafe)] // CPUID intrinsics are unsafe on the workspace MSRV.
    let supported = unsafe {
        std::arch::x86_64::__cpuid(0).eax >= 7
            && std::arch::x86_64::__cpuid_count(7, 0).ebx & (1 << 19) != 0
    };
    if !supported {
        eprintln!("native ADX witness skipped: CPUID lacks ADX");
        return None;
    }
    let result = x86jit_tests::native::run_native(input);
    if result.is_none() {
        eprintln!("native ADX witness skipped: native oracle unavailable");
    }
    result
}

#[test]
fn adx_single_partial_flags_and_interleaved_chains() {
    // SDM Vol 2A §3.3 ADCX/ADOX, Operation/Flags Affected: both
    // additions are unsigned; each reads and writes only its own carry flag.
    for (bits, mode) in [(64, CpuMode::Long64), (32, CpuMode::Compat32)] {
        for size in [4, 8] {
            if bits == 32 && size == 8 {
                continue;
            }
            let mask = if size == 8 { u64::MAX } else { u32::MAX as u64 };
            for memory in [false, true] {
                for (lhs, rhs) in [
                    (mask, 0),
                    (mask, mask),
                    (mask >> 1, 1),
                    (0, 0),
                    (LOW & mask, 0x1122_3344),
                ] {
                    for cf in [false, true] {
                        for of in [false, true] {
                            // Single-instruction cases also witness the untouched other carry.
                            for count in [1, 4] {
                                for first_overflow in [false, true] {
                                    let bytes = code(bits, |a| {
                                        for i in 0..count {
                                            adx(a, first_overflow ^ (i % 2 != 0), size, memory);
                                            a.setb(byte_ptr(esi + 16 + 2 * i)).unwrap();
                                            a.seto(byte_ptr(esi + 17 + 2 * i)).unwrap();
                                        }
                                    });
                                    let mut input = input(bytes);
                                    input.cpu_init.flags.cf = cf;
                                    input.cpu_init.flags.of = of;
                                    input.cpu_init.flags.sf = cf;
                                    input.cpu_init.flags.pf = of;
                                    input.cpu_init.flags.af = cf ^ of;
                                    // SDM Vol 1 §3.4.1.1: 32-bit writes zero the
                                    // destination's upper half, even with dirty input.
                                    let initial =
                                        lhs | if size == 4 { 0xdead_beef_0000_0000 } else { 0 };
                                    input.cpu_init.gpr[0] = initial;
                                    input.cpu_init.gpr[3] = initial;
                                    input.cpu_init.gpr[1] = rhs;
                                    input.mem_init[1].bytes[..8]
                                        .copy_from_slice(&rhs.to_le_bytes());
                                    let mut sums = [initial, initial];
                                    let mut carries = [cf, of];
                                    let mut history = Vec::new();
                                    for i in 0..count {
                                        let chain = usize::from(first_overflow ^ (i % 2 != 0));
                                        let total = (sums[chain] & mask) as u128
                                            + rhs as u128
                                            + u128::from(carries[chain]);
                                        sums[chain] = total as u64 & mask;
                                        carries[chain] = total > mask as u128;
                                        history
                                            .extend([u8::from(carries[0]), u8::from(carries[1])]);
                                    }
                                    let mut expected_flags = input.cpu_init.flags;
                                    expected_flags.cf = carries[0];
                                    expected_flags.of = carries[1];
                                    engines(&input, mode, |cpu, vm, exit| {
                                        assert!(matches!(exit, Exit::Hlt), "{exit:?}");
                                        assert_eq!([cpu.reg(Reg::Rax), cpu.reg(Reg::Rbx)], sums);
                                        assert_eq!(SnapFlags::from(cpu.flags()), expected_flags);
                                        let mut actual = vec![0; history.len()];
                                        vm.read_bytes(DATA + 16, &mut actual).unwrap();
                                        assert_eq!(actual, history);
                                    });
                                    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
                                    if mode == CpuMode::Long64 {
                                        if let Some(result) = native_adx(&input) {
                                            assert_eq!(
                                                result.exit,
                                                x86jit_tests::vector::ExitKind::Hlt
                                            );
                                            assert_eq!(
                                                [result.cpu.gpr[0], result.cpu.gpr[3]],
                                                sums
                                            );
                                            assert_eq!(result.cpu.flags, expected_flags);
                                            assert_eq!(
                                                &result.mem[1].bytes[16..16 + history.len()],
                                                history.as_slice()
                                            );
                                        }
                                    }
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
fn adx_memory_fault_recovers_both_carries() {
    // SDM Vol 3A §7.5: fault exposes the last completed instruction.
    // Later XOR kills both carries on the successful path, exercising cold recovery.
    for (bits, mode) in [(64, CpuMode::Long64), (32, CpuMode::Compat32)] {
        for size in [4, 8] {
            if bits == 32 && size == 8 {
                continue;
            }
            for fault_overflow in [false, true] {
                let mut fault_rip = 0;
                let bytes = code(bits, |a| {
                    if size == 8 {
                        a.add(rdi, rdi)
                    } else {
                        a.add(edi, edi)
                    }
                    .unwrap();
                    adx(a, false, size, false);
                    adx(a, true, size, false);
                    fault_rip = CODE + a.assemble(CODE).unwrap().len() as u64;
                    adx(a, fault_overflow, size, true);
                    a.xor(eax, eax).unwrap();
                });
                let mut input = input(bytes);
                input.cpu_init.gpr[0] = if size == 8 { u64::MAX } else { u32::MAX as u64 };
                input.cpu_init.gpr[3] = 0;
                input.cpu_init.gpr[6] = FAULT;
                input.cpu_init.gpr[7] = 1 << (size * 8 - 1);
                // The ADD must supply both carry inputs, rather than accidentally
                // reusing the same values from the initial flags.
                input.cpu_init.flags.cf = false;
                input.cpu_init.flags.of = false;
                engines(&input, mode, |cpu, _, exit| {
                    assert!(
                        matches!(exit, Exit::UnmappedMemory { addr: FAULT, .. }),
                        "{exit:?}"
                    );
                    assert_eq!(cpu.reg(Reg::Rip), fault_rip);
                    assert_eq!(cpu.reg(Reg::Rax), 0);
                    assert_eq!(cpu.reg(Reg::Rbx), 1);
                    // ADD produces CF=OF=ZF=PF=1, SF=AF=0; ADCX keeps CF=1,
                    // ADOX clears OF. DF is preserved throughout.
                    assert_eq!(cpu.flags().to_rflags() & 0xcd5, 0x445);
                });
            }
        }
    }
}

#[test]
fn adx_lock_prefix_is_rejected() {
    // SDM Vol 2A §3.3 ADCX/ADOX, Exceptions: LOCK causes #UD.
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for prefix in [0x66, 0xf3] {
            let input = input(vec![0xf0, prefix, 0x0f, 0x38, 0xf6, 0xc1, 0xf4]);
            // Invalid instructions cannot form a region; exercise both backends directly.
            for backend in [
                Box::new(InterpreterBackend) as Box<dyn Backend>,
                Box::new(JitBackend::new()),
            ] {
                let mut vm = Vm::with_backend(VmConfig::flat(0x40000), backend);
                vm.set_cpu_mode(mode);
                vm.map(CODE, 4096, Prot::RX, RegionKind::Ram).unwrap();
                vm.write_bytes(CODE, &input.mem_init[0].bytes).unwrap();
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, CODE);
                assert!(matches!(
                    cpu.run(&vm, None),
                    Exit::UnknownInstruction { .. } | Exit::Exception { vector: 6, .. }
                ));
            }
        }
    }
}
