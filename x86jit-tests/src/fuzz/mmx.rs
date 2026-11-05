//! MM-register differential generator, independent of the SSE mnemonic menu.
use super::*;
#[cfg(feature = "unicorn")]
use crate::oracle::Oracle;
use iced_x86::{Code, Encoder, Instruction, OpCodeOperandKind, OpKind, Register};

/// Every MM-register encoding in the compatibility map, plus EMMS and the bridges.
pub const CODES: &[Code] = &[
    Code::Movd_mm_rm32,
    Code::Movd_rm32_mm,
    Code::Movq_mm_mmm64,
    Code::Movq_mm_rm64,
    Code::Movq_mmm64_mm,
    Code::Movq_rm64_mm,
    Code::Packssdw_mm_mmm64,
    Code::Packsswb_mm_mmm64,
    Code::Packuswb_mm_mmm64,
    Code::Paddb_mm_mmm64,
    Code::Paddd_mm_mmm64,
    Code::Paddq_mm_mmm64,
    Code::Paddsb_mm_mmm64,
    Code::Paddsw_mm_mmm64,
    Code::Paddusb_mm_mmm64,
    Code::Paddusw_mm_mmm64,
    Code::Paddw_mm_mmm64,
    Code::Pand_mm_mmm64,
    Code::Pandn_mm_mmm64,
    Code::Pcmpeqb_mm_mmm64,
    Code::Pcmpeqd_mm_mmm64,
    Code::Pcmpeqw_mm_mmm64,
    Code::Pcmpgtb_mm_mmm64,
    Code::Pcmpgtd_mm_mmm64,
    Code::Pcmpgtw_mm_mmm64,
    Code::Pmaddwd_mm_mmm64,
    Code::Pmulhw_mm_mmm64,
    Code::Pmullw_mm_mmm64,
    Code::Por_mm_mmm64,
    Code::Pslld_mm_imm8,
    Code::Pslld_mm_mmm64,
    Code::Psllq_mm_imm8,
    Code::Psllq_mm_mmm64,
    Code::Psllw_mm_imm8,
    Code::Psllw_mm_mmm64,
    Code::Psrad_mm_imm8,
    Code::Psrad_mm_mmm64,
    Code::Psraw_mm_imm8,
    Code::Psraw_mm_mmm64,
    Code::Psrld_mm_imm8,
    Code::Psrld_mm_mmm64,
    Code::Psrlq_mm_imm8,
    Code::Psrlq_mm_mmm64,
    Code::Psrlw_mm_imm8,
    Code::Psrlw_mm_mmm64,
    Code::Psubb_mm_mmm64,
    Code::Psubd_mm_mmm64,
    Code::Psubsb_mm_mmm64,
    Code::Psubsw_mm_mmm64,
    Code::Psubusb_mm_mmm64,
    Code::Psubusw_mm_mmm64,
    Code::Psubw_mm_mmm64,
    Code::Punpckhbw_mm_mmm64,
    Code::Punpckhdq_mm_mmm64,
    Code::Punpckhwd_mm_mmm64,
    Code::Punpcklbw_mm_mmm32,
    Code::Punpckldq_mm_mmm32,
    Code::Punpcklwd_mm_mmm32,
    Code::Pxor_mm_mmm64,
    Code::Cvtpd2pi_mm_xmmm128,
    Code::Cvtpi2pd_xmm_mmm64,
    Code::Cvtpi2ps_xmm_mmm64,
    Code::Cvtps2pi_mm_xmmm64,
    Code::Cvttpd2pi_mm_xmmm128,
    Code::Cvttps2pi_mm_xmmm64,
    Code::Movntq_m64_mm,
    Code::Pavgb_mm_mmm64,
    Code::Pavgw_mm_mmm64,
    Code::Pextrw_r32_mm_imm8,
    Code::Pextrw_r64_mm_imm8,
    Code::Pinsrw_mm_r32m16_imm8,
    Code::Pinsrw_mm_r64m16_imm8,
    Code::Pmaxsw_mm_mmm64,
    Code::Pmaxub_mm_mmm64,
    Code::Pminsw_mm_mmm64,
    Code::Pminub_mm_mmm64,
    Code::Pmovmskb_r32_mm,
    Code::Pmovmskb_r64_mm,
    Code::Pmulhuw_mm_mmm64,
    Code::Pmuludq_mm_mmm64,
    Code::Psadbw_mm_mmm64,
    Code::Pshufw_mm_mmm64_imm8,
    Code::Psubq_mm_mmm64,
    Code::Pabsb_mm_mmm64,
    Code::Pabsd_mm_mmm64,
    Code::Pabsw_mm_mmm64,
    Code::Palignr_mm_mmm64_imm8,
    Code::Phaddd_mm_mmm64,
    Code::Phaddsw_mm_mmm64,
    Code::Phaddw_mm_mmm64,
    Code::Phsubd_mm_mmm64,
    Code::Phsubsw_mm_mmm64,
    Code::Phsubw_mm_mmm64,
    Code::Pmaddubsw_mm_mmm64,
    Code::Pmulhrsw_mm_mmm64,
    Code::Pshufb_mm_mmm64,
    Code::Psignb_mm_mmm64,
    Code::Psignd_mm_mmm64,
    Code::Psignw_mm_mmm64,
    Code::Emms,
    Code::Movq2dq_xmm_mm,
    Code::Movdq2q_mm_xmm,
];

pub const IMAGE: u64 = SCRATCH + 0x800;

/// Build an actual encoding; randomize all MM slots, immediates and memory offsets.
pub fn instruction(code: Code, seed: u64, memory: bool, mode: CpuMode) -> Instruction {
    let mut rng = Rng::new(seed);
    let mut i = Instruction::default();
    i.set_code(code);
    let info = code.op_code();
    let mut used_memory = false;
    for n in 0..info.op_count() {
        let kind = info.op_kind(n);
        let mem = memory
            && !used_memory
            && matches!(
                kind,
                OpCodeOperandKind::mm_or_mem
                    | OpCodeOperandKind::xmm_or_mem
                    | OpCodeOperandKind::r32_or_mem
                    | OpCodeOperandKind::r64_or_mem
            );
        crate::compat::template_operand(&mut i, n, kind, mode, mem).unwrap();
        if i.op_kind(n) == OpKind::Memory {
            used_memory = true;
            i.set_memory_base(Register::None);
            i.set_memory_index(Register::None);
            i.set_memory_displ_size(4);
            let mut addr = SCRATCH + rng.below(0x700) as u64;
            // SDM Vol 3B §25.25.3 Table 25-4: CVT{T}PD2PI m128 needs
            // 16-byte alignment. Misalignment has a separate #GP witness.
            if matches!(code, Code::Cvtpd2pi_mm_xmmm128 | Code::Cvttpd2pi_mm_xmmm128) {
                addr &= !15;
            }
            i.set_memory_displacement64(addr);
        } else if i.op_register(n).is_mm() {
            i.set_op_register(n, Register::MM0 + rng.below(8) as u32);
        } else if i.op_register(n).is_xmm() {
            i.set_op_register(n, Register::XMM0 + rng.below(8) as u32);
        } else if i.op_kind(n) == OpKind::Immediate8 {
            i.set_immediate8([0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 255][rng.below(13)]);
        }
    }
    i
}

pub fn input(codes: &[Code], seed: u64, memory: bool, mode: CpuMode) -> VectorInput {
    let mut rng = Rng::new(seed);
    let mut init = CpuSnapshot {
        rip: CODE,
        ..Default::default()
    };
    for g in &mut init.gpr[..8] {
        *g = rng.next()
            & if mode == CpuMode::Compat32 {
                0xffff_ffff
            } else {
                u64::MAX
            };
    }
    for v in &mut init.xmm[..8] {
        *v = rng.vec128();
    }
    for v in &mut init.st {
        let raw = match rng.below(3) {
            0 => rng.vec128() as u64,
            1 => [0, 1, 15, 16, 31, 32, 63, 64, 65, 255, u64::MAX][rng.below(11)],
            _ => rng.next(),
        };
        v[..8].copy_from_slice(&raw.to_le_bytes());
    }
    let mut bytes = Vec::new();
    for (n, &code) in codes.iter().enumerate() {
        let i = instruction(code, seed.wrapping_add(n as u64), memory, mode);
        let mut enc = Encoder::new(mode.bits());
        enc.encode(&i, CODE + bytes.len() as u64).unwrap();
        bytes.extend(enc.take_buffer());
    }
    // Capture tags and MM slots through the architectural image on every oracle.
    // SDM Vol 1 §10.5.1: FXSAVE's FTW byte is indexed by physical register.
    let mut a = CodeAssembler::new(mode.bits()).unwrap();
    if mode == CpuMode::Compat32 {
        a.fxsave(dword_ptr(IMAGE as u32)).unwrap();
    } else {
        a.fxsave(qword_ptr(IMAGE)).unwrap();
    }
    a.hlt().unwrap();
    bytes.extend(a.assemble(CODE + bytes.len() as u64).unwrap());
    let mut scratch = vec![0; SCRATCH_LEN];
    for b in &mut scratch[..0x800] {
        *b = rng.next() as u8;
    }
    VectorInput {
        cpu_init: init,
        entry: CODE,
        run: RunSpec::UntilExit,
        mem_init: vec![
            MemChunk {
                addr: CODE,
                bytes,
                kind: MemKind::Ram,
            },
            MemChunk {
                addr: SCRATCH,
                bytes: scratch,
                kind: MemKind::Ram,
            },
        ],
    }
}

/// Compare the MMX-relevant FXSAVE fields, discarding unmodelled pointers and MXCSR flags.
/// Unicorn leaves exponent bits 79:64 unchanged on MMX writes; mask only those bytes.
pub fn normalize(out: &mut RunOutcome, unicorn: bool) {
    for chunk in &mut out.mem {
        if chunk.addr == SCRATCH {
            let image = &mut chunk.bytes[0x800..0xa00];
            for (n, byte) in image.iter_mut().enumerate() {
                let keep = matches!(n, 0..=4)
                    || (32..160).contains(&n) && (n - 32) % 16 < if unicorn { 8 } else { 10 };
                if !keep {
                    *byte = 0;
                }
            }
        }
    }
    if unicorn {
        for slot in &mut out.cpu.st {
            slot[8..].fill(0);
        }
    }
}

/// Used by `cargo xfuzz --class mmx` and bounded regression tests.
pub fn check(input: &VectorInput, mode: CpuMode) -> Result<(), String> {
    let mut interp = run_with_backend_mode(
        input,
        Box::new(InterpreterBackend),
        GuestCpuFeatures::v2(),
        mode,
    );
    let mut jit = run_with_backend_mode(
        input,
        Box::new(JitBackend::new()),
        GuestCpuFeatures::v2(),
        mode,
    );
    normalize(&mut interp, false);
    normalize(&mut jit, false);
    let diff = compare(&interp, &jit, &[], &[]);
    if let Some(diff) = diff {
        return Err(format!("JIT: {diff}"));
    }
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    if mode == CpuMode::Long64 {
        let mut native =
            crate::native::run_native(input).ok_or("native oracle did not complete")?;
        normalize(&mut native, false);
        let mut ours = interp.clone();
        if has_memory_form(input, mode, Code::Cvtpi2ps_xmm_mmm64) {
            // SDM Vol 2A §3.3 CVTPI2PS: "This instruction causes a transition
            // from x87 FPU to MMX technology operation", with no exemption for
            // a memory source (CVTPI2PD exempts xmm,m64; CVTPI2PS does not),
            // and the engine follows the SDM. AMD APM Vol 4 CVTPI2PS is silent.
            // Measured on an AMD Zen 4 CPU, seed=1 Long64 memory case: the CPU
            // leaves the FXSAVE FTW at IMAGE+4 (0x220804) 0x00 where the SDM
            // response is 0xff. Exclude only the transition state (TOP and the
            // tag word) from the native comparison for such programs.
            for out in [&mut native, &mut ours] {
                mask_x87_transition(out);
            }
        }
        let diff = compare(&native, &ours, &[], &[]);
        if let Some(diff) = diff {
            return Err(format!("native: {diff}"));
        }
    }
    #[cfg(feature = "unicorn")]
    if !unicorn_cvtpi2pd_memory(input, mode) {
        let mut reference = if mode == CpuMode::Compat32 {
            crate::unicorn::UnicornOracle32.run(input)
        } else {
            crate::unicorn::UnicornOracle.run(input)
        };
        normalize(&mut reference, true);
        normalize(&mut interp, true);
        let diff = compare(&reference, &interp, &[], &[]);
        if let Some(diff) = diff {
            return Err(format!("Unicorn: {diff}"));
        }
    }
    Ok(())
}

/// Whether the program holds `code` with a memory source operand.
#[cfg_attr(
    not(all(target_arch = "x86_64", target_os = "linux")),
    allow(dead_code)
)]
fn has_memory_form(input: &VectorInput, mode: CpuMode, code: Code) -> bool {
    let chunk = input
        .mem_init
        .iter()
        .find(|chunk| chunk.addr == input.entry)
        .unwrap();
    let mut decoder = iced_x86::Decoder::with_ip(mode.bits(), &chunk.bytes, input.entry, 0);
    while decoder.can_decode() {
        let i = decoder.decode();
        if i.code() == code && i.op_kind(1) == OpKind::Memory {
            return true;
        }
    }
    false
}

/// Clear TOP and the tag word, in the CPU state and in the saved image.
#[cfg_attr(
    not(all(target_arch = "x86_64", target_os = "linux")),
    allow(dead_code)
)]
fn mask_x87_transition(out: &mut RunOutcome) {
    out.cpu.fpu_top = 0;
    for chunk in &mut out.mem {
        if chunk.addr == SCRATCH {
            // FXSAVE image: FSW bits 13:11 hold TOP; byte 4 is the abridged FTW.
            chunk.bytes[0x802] = 0;
            chunk.bytes[0x803] &= !0x38;
            chunk.bytes[0x804] = 0;
        }
    }
}

/// Unicorn diverges on the x87 transition for this operand configuration.
/// Keep interpreter/JIT and native comparisons above, including the saved FTW.
#[cfg(feature = "unicorn")]
fn unicorn_cvtpi2pd_memory(input: &VectorInput, mode: CpuMode) -> bool {
    let code = input
        .mem_init
        .iter()
        .find(|chunk| chunk.addr == input.entry)
        .unwrap();
    let mut decoder = iced_x86::Decoder::with_ip(mode.bits(), &code.bytes, input.entry, 0);
    while decoder.can_decode() {
        let i = decoder.decode();
        if i.code() == Code::Cvtpi2pd_xmm_mmm64 && i.op_kind(1) == OpKind::Memory {
            // SDM Vol 2A §3.3 CVTPI2PD: xmm,m64 neither transitions nor takes
            // x87 exceptions. Measured seed=1 Long64 memory case: Unicorn's
            // FXSAVE FTW at IMAGE+4 (0x220804) is 0xff; engine is 0x00.
            // Skip this program only for Unicorn; never mask the native FTW.
            return true;
        }
    }
    false
}

#[cfg(all(test, feature = "unicorn"))]
mod tests {
    use super::*;

    #[test]
    fn unicorn_exemption_is_limited_to_cvtpi2pd_memory() {
        for mode in [CpuMode::Long64, CpuMode::Compat32] {
            for code in [
                Code::Cvtpi2pd_xmm_mmm64,
                Code::Cvtpi2ps_xmm_mmm64,
                Code::Movq2dq_xmm_mm,
            ] {
                for memory in [false, true] {
                    let p = input(&[code], 1, memory, mode);
                    assert_eq!(
                        unicorn_cvtpi2pd_memory(&p, mode),
                        code == Code::Cvtpi2pd_xmm_mmm64 && memory
                    );
                    if code == Code::Cvtpi2pd_xmm_mmm64 && memory {
                        // Still runs interpreter/JIT and, on x86-64 Linux, native.
                        check(&p, mode).unwrap();
                    }
                }
            }
        }
    }
}
