//! ISA compatibility probe: measure, *mechanically*, which x86-64 instruction forms
//! the lifter (`x86jit_core::lift`) actually handles, bucketed by instruction-set
//! generation (v1/v2/v3/v4 + x87/MMX + optional extensions). The map probes the real lifter
//! — never hand-written prose, which rots immediately: the in-tree CPUID comment it
//! replaced was already wrong when it was written.
//!
//! Method: for every `iced_x86::Code` valid in 64-bit mode and in scope (its CPUID
//! features map to a generation we model), synthesize a canonical instruction with
//! templated operands, encode it, feed the bytes (plus a `ret` terminator) to
//! `lift_block`, and classify Lifted / Unsupported / Unencodable.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use iced_x86::{
    Code, CpuidFeature, Encoder, FlowControl, Instruction, OpCodeOperandKind, OpKind, Register,
};
use serde::{Deserialize, Serialize};
use x86jit_core::ir::{IrBlock, IrOp};
use x86jit_core::lift::{lift_block, CpuMode, FetchAddr, LiftError};
use x86jit_core::{Memory, MemoryModel, Prot, RegionKind};

/// Instruction-set generation buckets we model (x86-64 userland).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Serialize, Deserialize)]
pub enum Gen {
    /// x86-64-v1 baseline: scalar 64-bit + SSE + SSE2.
    V1,
    /// x86-64-v2: SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, CMPXCHG16B, LAHF/SAHF.
    V2,
    /// x86-64-v3: AVX, AVX2, BMI1/2, FMA, F16C, LZCNT, MOVBE.
    V3,
    /// x86-64-v4: AVX-512 F/BW/DQ/VL/CD (the in-progress AVX-512 lift).
    V4,
    /// x87 FPU (fidelity note: implemented f64-backed, not true 80-bit).
    X87,
    /// Legacy MMX.
    Mmx,
    /// Optional extensions outside the psABI micro-architecture levels.
    Optional,
}

impl Gen {
    pub fn label(self) -> &'static str {
        match self {
            Gen::V1 => "x86-64-v1",
            Gen::V2 => "x86-64-v2",
            Gen::V3 => "x86-64-v3",
            Gen::V4 => "x86-64-v4",
            Gen::X87 => "x87",
            Gen::Mmx => "mmx",
            Gen::Optional => "optional",
        }
    }
}

/// Map a single CPUID feature to the generation it belongs to, or `None` if it is
/// out of scope (AVX-512, privileged/system, vendor extensions — deliberately not
/// modeled here; those get their own brief when a target image demands them).
fn feature_gen(f: CpuidFeature) -> Option<Gen> {
    use CpuidFeature::*;
    Some(match f {
        // Baseline scalar userland — a v1 CPU without SIMD features.
        INTEL8086 | INTEL8086_ONLY | INTEL186 | INTEL286 | INTEL286_ONLY | INTEL386
        | INTEL386_ONLY | INTEL386_A0_ONLY | INTEL486 | INTEL486_A_ONLY | CMOV | CLFSH
        | CLFLUSHOPT | CX8 | CPUID | PAUSE | MULTIBYTENOP | FXSR | SSE | SSE2 | SYSCALL | SEP
        | SMAP | X64 => Gen::V1,
        // psABI §3.1.1, Table 3.1 (Micro-Architecture Levels).
        RDTSCP | RDRAND | RDSEED | SSE4A | ADX | XSAVE | XSAVEOPT | XSAVEC => Gen::Optional,
        // x86-64-v2.
        SSE3 | SSSE3 | SSE4_1 | SSE4_2 | POPCNT | CMPXCHG16B | MOVBE => Gen::V2,
        // x86-64-v3.
        AVX | AVX2 | BMI1 | BMI2 | FMA | F16C | LZCNT => Gen::V3,
        // x86-64-v4 (AVX-512 base set; sub-extensions like VBMI/VNNI stay out of scope,
        // so a code needing them is skipped by the all-features-modeled rule).
        AVX512F | AVX512BW | AVX512DQ | AVX512VL | AVX512CD => Gen::V4,
        // x87 FPU (iced has no single X87 feature — the FPU* variants tag it).
        FPU | FPU287 | FPU287XL_ONLY | FPU387 | FPU387SL_ONLY => Gen::X87,
        MMX => Gen::Mmx,
        // Everything else (AVX-512 family, FMA4, privileged, TSX, SHA, AES,
        // vendor) is out of scope for this map.
        _ => return None,
    })
}

/// Generation of a whole instruction: in scope only if *all* its CPUID features are
/// modeled; the generation is the highest among them (an SSE4.1 instruction that
/// also lists SSE2 is v2). MOVBE is v2 but pairs with a baseline feature — the max
/// rule handles it. Returns `None` for out-of-scope instructions.
fn code_gen(code: Code) -> Option<Gen> {
    // SDM Vol 1 §13.3: XSETBV requires CPL 0, outside this user-mode map.
    if code == Code::Xsetbv {
        return None;
    }
    let feats = code.cpuid_features();
    if feats.is_empty() {
        return None;
    }
    let mut acc: Option<Gen> = None;
    for &f in feats {
        let g = feature_gen(f)?; // any out-of-scope feature ⇒ whole insn out of scope
        acc = Some(match acc {
            Some(prev) if prev >= g => prev,
            _ => g,
        });
    }
    acc
}

/// Result of probing one instruction form.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Probe {
    /// The lifter produced IR covering the complete instruction.
    Lifted,
    /// The lifter returned `Unsupported` at the start of the probe — a real gap.
    Unsupported,
    /// No complete probe result: synthesis/encoding failed, decoding faulted, or
    /// lifting stopped partway through the form. Not counted for/against coverage.
    Unencodable,
    /// The register form lifts and the memory form does not.
    ///
    /// iced represents both alternatives of a register-or-memory operand under one
    /// `Code`, so probing only the register form reports the whole `Code` as lifted
    /// while the lifter still rejects `[mem]`. That is not hypothetical: `vextract*`'s
    /// memory destination was covered here until a real guest binary trapped on it.
    LiftedRegOnly,
}

const SCRATCH_BASE: u64 = 0x1000;

/// Re-decoded EVEX modifier probes, including extensions outside the coverage map.
/// Every successful encoding is checked for the requested mask/broadcast bits.
pub fn evex_modifier_forms() -> Vec<Instruction> {
    use iced_x86::{Decoder, DecoderOptions, EncodingKind};
    let mut forms = Vec::new();
    for code in Code::values() {
        let info = code.op_code();
        if info.encoding() != EncodingKind::EVEX || !info.mode64() {
            continue;
        }
        for memory in [false, true] {
            for broadcast in [false, true] {
                for masked in [false, true] {
                    for zero in [false, true] {
                        if (!broadcast && !masked)
                            || (broadcast && (!memory || !info.can_broadcast()))
                            || (masked && !info.can_use_op_mask_register())
                            || (zero && (!masked || !info.can_use_zeroing_masking()))
                        {
                            continue;
                        }
                        let mut i = Instruction::default();
                        i.set_code(code);
                        let mut valid = true;
                        let mut has_memory = false;
                        for op in 0..info.op_count() {
                            let kind = info.op_kind(op);
                            use OpCodeOperandKind::*;
                            if memory && matches!(kind, xmm_or_mem | ymm_or_mem | zmm_or_mem) {
                                i.set_op_kind(op, OpKind::Memory);
                                i.set_memory_base(Register::RAX);
                                has_memory = true;
                            } else if memory
                                && matches!(
                                    kind,
                                    mem_vsib32x
                                        | mem_vsib32y
                                        | mem_vsib32z
                                        | mem_vsib64x
                                        | mem_vsib64y
                                        | mem_vsib64z
                                )
                            {
                                i.set_op_kind(op, OpKind::Memory);
                                i.set_memory_base(Register::RAX);
                                i.set_memory_index(match kind {
                                    mem_vsib32x | mem_vsib64x => Register::XMM4,
                                    mem_vsib32y | mem_vsib64y => Register::YMM4,
                                    _ => Register::ZMM4,
                                });
                                has_memory = true;
                            } else if kind == zmmp3_vvvv {
                                i.set_op_register(op, Register::ZMM4);
                            } else if matches!(kind, zmm_reg | zmm_rm | zmm_vvvv | zmm_or_mem) {
                                i.set_op_register(
                                    op,
                                    Register::try_from(Register::ZMM0 as usize + op as usize)
                                        .unwrap(),
                                );
                            } else if matches!(kind, k_reg | kp1_reg | k_rm | k_vvvv | k_or_mem) {
                                i.set_op_register(
                                    op,
                                    Register::try_from(Register::K2 as usize + op as usize)
                                        .unwrap(),
                                );
                            } else if template_operand(&mut i, op, kind, CpuMode::Long64, false)
                                .is_err()
                            {
                                valid = false;
                            }
                        }
                        has_memory |=
                            (0..info.op_count()).any(|op| i.op_kind(op) == OpKind::Memory);
                        if !valid || memory != has_memory {
                            continue;
                        }
                        i.set_is_broadcast(broadcast);
                        i.set_op_mask(if masked { Register::K1 } else { Register::None });
                        i.set_zeroing_masking(zero);
                        let mut encoder = Encoder::new(64);
                        if encoder.encode(&i, SCRATCH_BASE).is_err() {
                            continue;
                        }
                        let bytes = encoder.take_buffer();
                        let decoded =
                            Decoder::with_ip(64, &bytes, SCRATCH_BASE, DecoderOptions::NONE)
                                .decode();
                        if decoded.is_invalid() {
                            continue;
                        }
                        assert_eq!(decoded.code(), code);
                        assert_eq!(decoded.is_broadcast(), broadcast);
                        assert_eq!(decoded.op_mask(), i.op_mask());
                        assert_eq!(decoded.zeroing_masking(), zero);
                        forms.push(decoded);
                    }
                }
            }
        }
    }
    forms
}

/// Mnemonics executed by the EVEX modifier differential campaign.
pub fn evex_lifted_modifier_mnemonics() -> BTreeSet<String> {
    evex_modifier_forms()
        .into_iter()
        .filter_map(|i| {
            let mut encoder = Encoder::new(64);
            encoder.encode(&i, SCRATCH_BASE).ok()?;
            (probe_encoded(i.code(), &encoder.take_buffer(), CpuMode::Long64)
                == Some(Probe::Lifted))
            .then(|| format!("{:?}", i.mnemonic()))
        })
        .collect()
}

/// Synthesize + encode a canonical instance of `code`, lift it in 64-bit long mode,
/// and classify. Returns `None` if the code is out of scope or invalid in the mode.
pub fn probe_code(code: Code) -> Option<Probe> {
    probe_code_in(code, CpuMode::Long64)
}

/// [`probe_code`] with the CPU mode as a parameter (§17.3 seam): the mode
/// picks the validity gate (`mode64()`/`mode32()`), the encoder bitness, and the
/// lift mode — so the coverage map measures each mode's *own* ISA (legacy-only and
/// 16-bit-operand forms exist only outside long mode). A 16-bit real-mode probe is
/// this same function with a 16-bit `CpuMode`, once such a mode exists.
pub fn probe_code_in(code: Code, mode: CpuMode) -> Option<Probe> {
    code_gen(code)?; // Generation-map scope only; CPUID checks bypass this gate.
    probe_instruction(code, mode)
}

fn probe_instruction(code: Code, mode: CpuMode) -> Option<Probe> {
    probe_instruction_with(code, mode, &mut |_, _| {})
}

fn probe_instruction_with(
    code: Code,
    mode: CpuMode,
    observe: &mut impl FnMut(&IrBlock, Option<u32>),
) -> Option<Probe> {
    let info = code.op_code();
    let valid_in_mode = match mode {
        CpuMode::Long64 => info.mode64(),
        CpuMode::Compat32 => info.mode32(),
        CpuMode::Real16 => info.mode16(),
    };
    if !valid_in_mode || code == Code::INVALID {
        return None;
    }

    // Probe the register form first, then — for a `Code` that also has a memory
    // alternative — the memory form. iced files both under one `Code`, so a probe that
    // only builds the register form reports the whole `Code` as covered even when the
    // lifter rejects `[mem]`.
    let reg_form = probe_form_with(code, info.op_count(), mode, None, &mut |block| {
        observe(block, None)
    })?;
    let mem_idx = (0..info.op_count()).find(|&i| {
        let mut probe = Instruction::default();
        template_operand(&mut probe, i, info.op_kind(i), mode, true).is_ok()
    });
    match (reg_form, mem_idx) {
        (Probe::Lifted, Some(idx)) => {
            match probe_form_with(code, info.op_count(), mode, Some(idx), &mut |block| {
                observe(block, Some(idx))
            }) {
                // A memory form the encoder rejects is not a gap: this `Code` has none.
                Some(Probe::Unencodable) | None => Some(Probe::Lifted),
                Some(Probe::Lifted) => Some(Probe::Lifted),
                Some(_) => Some(Probe::LiftedRegOnly),
            }
        }
        (other, _) => Some(other),
    }
}

/// One probe of `code`: every operand a register, except `mem_at` if given.
fn probe_form_with(
    code: Code,
    op_count: u32,
    mode: CpuMode,
    mem_at: Option<u32>,
    observe: &mut impl FnMut(&IrBlock),
) -> Option<Probe> {
    let info = code.op_code();
    let mut instr = Instruction::default();
    instr.set_code(code);
    // iced needs the operand count set before per-operand setters take effect for
    // some forms; the OpCodeInfo drives how many operands to template.
    for i in 0..op_count {
        if template_operand(&mut instr, i, info.op_kind(i), mode, mem_at == Some(i)).is_err() {
            return Some(Probe::Unencodable);
        }
    }

    let mut enc = Encoder::new(mode.bits());
    let bytes = match enc.encode(&instr, SCRATCH_BASE) {
        Ok(_) => enc.take_buffer(),
        Err(_) => return Some(Probe::Unencodable),
    };
    if bytes.is_empty() {
        return Some(Probe::Unencodable);
    }

    probe_encoded_with(code, &bytes, mode, observe)
}

fn probe_encoded(code: Code, bytes: &[u8], mode: CpuMode) -> Option<Probe> {
    probe_encoded_with(code, bytes, mode, &mut |_| {})
}

fn probe_encoded_with(
    code: Code,
    bytes: &[u8],
    mode: CpuMode,
    observe: &mut impl FnMut(&IrBlock),
) -> Option<Probe> {
    // Append a `ret` (0xC3) so a non-terminator probe instruction still closes the
    // block; a terminator probe (jmp/ret/…) ends it itself.
    let mut prog = bytes.to_vec();
    prog.push(0xC3);
    let mut mem = Memory::new(MemoryModel::Flat { size: 0x4000 });
    // write_bytes needs a mapped region (it bypasses guest Prot but the region must
    // exist); map the whole scratch buffer as executable RAM.
    if mem.map(0, 0x4000, Prot::RX, RegionKind::Ram).is_err() {
        return Some(Probe::Unencodable);
    }
    if mem.write_bytes(SCRATCH_BASE, &prog).is_err() {
        return Some(Probe::Unencodable);
    }
    match lift_block(&mem, FetchAddr::flat(SCRATCH_BASE), mode) {
        Ok(block) => {
            observe(&block);
            // Descriptor-dependent forms decode to an exact embedder exit;
            // count them as unsupported execution, despite successful lifting.
            if block
                .ops
                .iter()
                .any(|op| matches!(op, IrOp::Unsupported { .. }))
            {
                return Some(Probe::Unsupported);
            }
            // A waiting x87 form can lift FWAIT alone and stop before its remainder.
            // Decode-failure IR also returns Ok, but does not implement the probed form.
            // Instructions whose purpose is an exception (e.g. UD2 or INT) still count.
            let intentional_exception = matches!(
                code.flow_control(),
                FlowControl::Exception | FlowControl::Interrupt
            );
            let decode_exception = !intentional_exception
                && block.ops.iter().any(|op| {
                    matches!(
                        op,
                        IrOp::Trap { vector: 6 | 13, .. } | IrOp::IntGate { vector: 6 | 13, .. }
                    )
                });
            if block.guest_len < bytes.len() as u32 || decode_exception {
                // Preserve the inconclusive classification of the corresponding
                // decode error or unsupported instruction past the probe's start.
                Some(Probe::Unencodable)
            } else {
                Some(Probe::Lifted)
            }
        }
        Err(LiftError::Unsupported { addr, .. }) if addr == SCRATCH_BASE => {
            Some(Probe::Unsupported)
        }
        // Unsupported past the probe insn (shouldn't happen with a plain ret), or a
        // decode error on our own bytes: treat as unencodable noise, not a gap.
        Err(_) => Some(Probe::Unencodable),
    }
}

/// Executable Real16 register/memory probes, derived from the coverage templater.
/// Includes legacy operand shapes outside the long-mode coverage map.
pub fn real16_forms() -> Vec<(Code, Vec<u8>)> {
    use iced_x86::OpCodeOperandKind::*;
    let mut forms = Vec::new();
    for code in Code::values().filter(|c| *c != Code::INVALID && c.op_code().mode16()) {
        for memory in [false, true] {
            let mut insn = Instruction::default();
            insn.set_code(code);
            let mut valid = true;
            for i in 0..code.op_code().op_count() {
                let kind = code.op_code().op_kind(i);
                if i > 0 && kind == imm8 && insn.op_kind(i - 1) == OpKind::Immediate16 {
                    insn.set_op_kind(i, OpKind::Immediate8_2nd);
                    insn.set_immediate8_2nd(2);
                    continue;
                }
                if kind == mem_offs {
                    insn.set_op_kind(i, OpKind::Memory);
                    insn.set_memory_base(Register::None);
                    insn.set_memory_displacement64(0x4040);
                    insn.set_memory_displ_size(2);
                    continue;
                }
                let has_memory = matches!(
                    kind,
                    r8_or_mem
                        | r16_or_mem
                        | r32_or_mem
                        | r64_or_mem
                        | mm_or_mem
                        | xmm_or_mem
                        | ymm_or_mem
                        | mem
                        | mem_offs
                );
                if template_operand(&mut insn, i, kind, CpuMode::Real16, memory && has_memory)
                    .is_ok()
                {
                    continue;
                }
                match kind {
                    imm16 => {
                        insn.set_op_kind(i, OpKind::Immediate16);
                        insn.set_immediate16(0x20);
                    }
                    imm32 | imm32sex64 => {
                        insn.set_op_kind(i, OpKind::Immediate32);
                        insn.set_immediate32(0x20);
                    }
                    seg_reg | es | cs | ss | ds | fs | gs => insn.set_op_register(
                        i,
                        match kind {
                            es => Register::ES,
                            cs => Register::CS,
                            ss => Register::SS,
                            fs => Register::FS,
                            gs => Register::GS,
                            _ => Register::DS,
                        },
                    ),
                    br16_1 | br16_2 => {
                        insn.set_op_kind(i, OpKind::NearBranch16);
                        insn.set_near_branch16(0x1020);
                    }
                    br32_1 | br32_4 => {
                        insn.set_op_kind(i, OpKind::NearBranch32);
                        insn.set_near_branch32(0x1020);
                    }
                    farbr2_2 | farbr4_2 => {
                        insn.set_op_kind(
                            i,
                            if kind == farbr2_2 {
                                OpKind::FarBranch16
                            } else {
                                OpKind::FarBranch32
                            },
                        );
                        insn.set_far_branch_selector(0);
                        insn.set_far_branch32(0x1020);
                    }
                    seg_rSI => insn.set_op_kind(i, OpKind::MemorySegSI),
                    es_rDI => insn.set_op_kind(i, OpKind::MemoryESDI),
                    seg_rBX_al => {
                        insn.set_op_kind(i, OpKind::Memory);
                        insn.set_memory_base(Register::BX);
                    }
                    _ => valid = false,
                }
            }
            if !valid {
                continue;
            }
            add_real16_form(&mut forms, code, &insn);
            let mut addr32 = insn;
            let mut address_operand = false;
            for i in 0..insn.op_count() {
                match insn.op_kind(i) {
                    OpKind::Memory => {
                        if insn.memory_base() == Register::BX {
                            addr32.set_memory_base(Register::EBX);
                            address_operand = true;
                        } else if insn.memory_base() == Register::None && kind_is_moffs(code) {
                            addr32.set_memory_displ_size(4);
                            address_operand = true;
                        }
                    }
                    OpKind::MemorySegSI => {
                        addr32.set_op_kind(i, OpKind::MemorySegESI);
                        address_operand = true;
                    }
                    OpKind::MemoryESDI => {
                        addr32.set_op_kind(i, OpKind::MemoryESEDI);
                        address_operand = true;
                    }
                    _ => {}
                }
            }
            if address_operand {
                add_real16_form(&mut forms, code, &addr32);
            }
        }
    }
    forms
}

fn kind_is_moffs(code: Code) -> bool {
    (0..code.op_code().op_count())
        .any(|i| code.op_code().op_kind(i) == iced_x86::OpCodeOperandKind::mem_offs)
}
fn add_real16_form(forms: &mut Vec<(Code, Vec<u8>)>, code: Code, insn: &Instruction) {
    let mut enc = Encoder::new(16);
    if enc.encode(insn, SCRATCH_BASE).is_err() {
        return;
    }
    let bytes = enc.take_buffer();
    let probe = probe_encoded(code, &bytes, CpuMode::Real16);
    // Absent-coprocessor inputs retain CPU operand checks even when
    // the Present engine leaves the ESC form to the embedder (§17.9).
    let esc = bytes
        .iter()
        .find(|b| {
            !matches!(
                b,
                0x26 | 0x2e | 0x36 | 0x3e | 0x64..=0x67 | 0xf0 | 0xf2 | 0xf3
            )
        })
        .is_some_and(|b| (0xd8..=0xdf).contains(b));
    if !bytes.is_empty()
        && (probe == Some(Probe::Lifted) || esc && probe == Some(Probe::Unsupported))
        && !forms.iter().any(|(_, b)| *b == bytes)
    {
        forms.push((code, bytes.clone()));
        if insn.op_kinds().any(|k| {
            matches!(
                k,
                OpKind::MemorySegSI
                    | OpKind::MemoryESDI
                    | OpKind::MemorySegESI
                    | OpKind::MemoryESEDI
            )
        }) {
            for prefix in [0xf2, 0xf3] {
                let mut prefixed = vec![prefix];
                prefixed.extend(&bytes);
                if probe_encoded(code, &prefixed, CpuMode::Real16) == Some(Probe::Lifted) {
                    forms.push((code, prefixed));
                }
            }
        }
    }
}

/// Verify every block synthesized by the ISA probe, including partial lifts.
/// Register and memory forms use the coverage probe's scope and selection rules.
#[cfg(debug_assertions)]
pub fn verify_probe_temp_defs_in(mode: CpuMode) -> Result<usize, Vec<String>> {
    let mut checked = 0;
    let mut failures = Vec::new();
    for code in Code::values().filter(|&code| code_gen(code).is_some()) {
        probe_instruction_with(code, mode, &mut |block, mem_at| {
            checked += 1;
            if let Err(error) = block.verify_temp_defs() {
                failures.push(format!(
                    "{mode:?}/{code:?} memory_operand={mem_at:?}: {error:?}, op={:?}",
                    block.ops[error.op_index]
                ));
            }
        });
    }
    if failures.is_empty() {
        Ok(checked)
    } else {
        Err(failures)
    }
}

/// Pick registers by operand index so a 2-operand form gets distinct registers.
fn nth<const N: usize>(regs: [Register; N], i: u32) -> Register {
    regs[(i as usize) % N]
}

/// Set operand `i` of `instr` to a canonical value for `kind`. `Err(())` means the
/// operand kind is exotic/unsupported by this templater (⇒ Unencodable). `mode`
/// picks the memory base register width (RAX in long mode, EAX at bitness 32).
pub(crate) fn template_operand(
    instr: &mut Instruction,
    i: u32,
    kind: OpCodeOperandKind,
    mode: CpuMode,
    as_memory: bool,
) -> Result<(), ()> {
    use OpCodeOperandKind::*;
    let as_mem = |instr: &mut Instruction| {
        instr.set_op_kind(i, OpKind::Memory);
        instr.set_memory_base(match mode {
            CpuMode::Long64 => Register::RAX,
            CpuMode::Compat32 => Register::EAX,
            CpuMode::Real16 => Register::BX,
        });
        instr.set_memory_displacement64(0x40);
        // The encoder rejects a displacement with displ_size 0 — "Displacement must be
        // 0 if displ_size == 0". Without this every memory form comes back Unencodable,
        // which the probe reads as "this Code has no memory form" and reports the
        // register result as full coverage.
        instr.set_memory_displ_size(1);
    };
    // The register-or-memory kinds, taken as memory when the caller asks. Only one
    // operand is ever asked: x86 encodes at most one memory operand.
    if as_memory {
        return match kind {
            r8_or_mem | r16_or_mem | r16_reg_mem | r16_rm | r32_or_mem | r32_or_mem_mpx
            | r32_reg_mem | r32_rm | r64_or_mem | r64_or_mem_mpx | r64_reg_mem | r64_rm
            | xmm_rm | xmm_or_mem | ymm_rm | ymm_or_mem | mm_rm | mm_or_mem => {
                as_mem(instr);
                Ok(())
            }
            _ => Err(()),
        };
    }
    // Register-form operands (including the register alternative of `*_or_mem`).
    let reg = |instr: &mut Instruction, r: Register| {
        instr.set_op_kind(i, OpKind::Register);
        instr.set_op_register(i, r);
    };
    match kind {
        seg_reg => reg(instr, Register::DS),
        r8_reg | r8_opcode | r8_or_mem => reg(
            instr,
            nth([Register::AL, Register::CL, Register::DL, Register::BL], i),
        ),
        r16_reg | r16_reg_mem | r16_rm | r16_opcode | r16_or_mem => reg(
            instr,
            nth([Register::AX, Register::CX, Register::DX, Register::BX], i),
        ),
        r32_reg | r32_reg_mem | r32_rm | r32_opcode | r32_vvvv | r32_or_mem | r32_or_mem_mpx => {
            reg(
                instr,
                nth(
                    [Register::EAX, Register::ECX, Register::EDX, Register::EBX],
                    i,
                ),
            )
        }
        r64_reg | r64_reg_mem | r64_rm | r64_opcode | r64_vvvv | r64_or_mem | r64_or_mem_mpx => {
            reg(
                instr,
                nth(
                    [Register::RAX, Register::RCX, Register::RDX, Register::RBX],
                    i,
                ),
            )
        }
        xmm_reg | xmm_rm | xmm_vvvv | xmm_is4 | xmm_is5 | xmmp3_vvvv | xmm_or_mem => reg(
            instr,
            nth(
                [
                    Register::XMM0,
                    Register::XMM1,
                    Register::XMM2,
                    Register::XMM3,
                ],
                i,
            ),
        ),
        ymm_reg | ymm_rm | ymm_vvvv | ymm_is4 | ymm_is5 | ymm_or_mem => reg(
            instr,
            nth(
                [
                    Register::YMM0,
                    Register::YMM1,
                    Register::YMM2,
                    Register::YMM3,
                ],
                i,
            ),
        ),
        mm_reg | mm_rm | mm_or_mem => reg(
            instr,
            nth(
                [Register::MM0, Register::MM1, Register::MM2, Register::MM3],
                i,
            ),
        ),
        // Fixed implicit registers.
        al => reg(instr, Register::AL),
        cl => reg(instr, Register::CL),
        ax => reg(instr, Register::AX),
        dx => reg(instr, Register::DX),
        eax => reg(instr, Register::EAX),
        rax => reg(instr, Register::RAX),
        st0 => reg(instr, Register::ST0),
        sti_opcode => reg(instr, Register::ST1),
        // VSIB operands have a vector index distinct from destination 0 and mask 2.
        mem_vsib32x | mem_vsib64x | mem_vsib32y | mem_vsib64y => {
            if mode == CpuMode::Real16 {
                return Err(());
            }
            as_mem(instr);
            instr.set_memory_index(if matches!(kind, mem_vsib32y | mem_vsib64y) {
                Register::YMM1
            } else {
                Register::XMM1
            });
            instr.set_memory_index_scale(4);
        }
        // Pure memory operands.
        mem | mem_offs => {
            instr.set_op_kind(i, OpKind::Memory);
            instr.set_memory_base(match mode {
                CpuMode::Long64 => Register::RAX,
                CpuMode::Compat32 => Register::EAX,
                CpuMode::Real16 => Register::BX,
            });
            instr.set_memory_displacement64(0x40);
            // Non-zero displ_size or the encoder rejects the displacement; see the note
            // in `as_mem` above.
            instr.set_memory_displ_size(1);
        }
        // Immediates.
        imm8 | imm8_const_1 | imm8sex16 | imm8sex32 | imm8sex64 | imm4_m2z => {
            // EXTRQ/INSERTQ carry two consecutive immediate bytes. iced stores
            // the second separately and requires Immediate8_2nd for its operand.
            if i > 0 && instr.op_kind(i - 1) == OpKind::Immediate8 {
                instr.set_op_kind(i, OpKind::Immediate8_2nd);
                instr.set_immediate8_2nd(1);
            } else {
                instr.set_op_kind(i, OpKind::Immediate8);
                instr.set_immediate8(1);
            }
        }
        // Anything else (mask regs, zmm, seg/cr/dr/tr/bnd, moffs, rel, wider
        // immediates, is4 for legacy) — exotic or out of our v1..v3 focus.
        _ => return Err(()),
    }
    Ok(())
}

// --- coverage aggregation + serialized artifacts ---

/// Per-generation counts + the concrete missing/partial code lists (the machine-
/// readable artifact, `backlog/docs/compat/coverage.json`).
#[derive(Serialize, Deserialize, Default)]
pub struct GenCoverage {
    pub lifted: u32,
    pub unsupported: u32,
    pub unencodable: u32,
    /// Mnemonic-ish `Code` names that probed Unsupported (the gap list).
    pub missing: Vec<String>,
    /// Codes whose register form lifts and whose memory form does not. Counted apart
    /// from `lifted` because reporting them as covered is what let `vextract*`'s memory
    /// destination look supported until a real guest binary trapped on it.
    #[serde(default)]
    pub reg_only: u32,
    /// The concrete `Code` names behind `reg_only`.
    #[serde(default)]
    pub missing_mem_form: Vec<String>,
}

#[derive(Serialize, Deserialize, Default)]
pub struct Coverage {
    /// 64-bit long mode (the original map; key name kept for artifact stability).
    pub generations: BTreeMap<String, GenCoverage>,
    /// 32-bit compat mode (`CpuMode::Compat32`): the same generations probed
    /// at bitness 32. Includes what long mode can't encode — the legacy-only forms
    /// (`Pushad`/`Into`/`Les_r32_m1632`/…) and the 16-bit operand-size forms
    /// (`Call_rel16`/`Retnw`/`Pushaw`/…), so the 32-bit gap list is visible.
    #[serde(default)]
    pub compat32: BTreeMap<String, GenCoverage>,
}

/// Probe the whole in-scope ISA in `mode` and aggregate per generation. Real-mode
/// (16-bit) coverage is this same probe once a 16-bit `CpuMode` exists (§17.6: the
/// seam is the parameter; the mode itself arrives with a consumer).
fn mode_coverage(mode: CpuMode) -> BTreeMap<String, GenCoverage> {
    let mut map: BTreeMap<String, GenCoverage> = BTreeMap::new();
    for code in Code::values() {
        let Some(g) = code_gen(code) else { continue };
        let Some(p) = probe_code_in(code, mode) else {
            continue;
        };
        let entry = map.entry(g.label().to_string()).or_default();
        match p {
            Probe::Lifted => entry.lifted += 1,
            Probe::Unsupported => {
                entry.unsupported += 1;
                entry.missing.push(format!("{code:?}"));
            }
            Probe::Unencodable => entry.unencodable += 1,
            Probe::LiftedRegOnly => {
                entry.reg_only += 1;
                entry.missing_mem_form.push(format!("{code:?}"));
            }
        }
    }
    for gc in map.values_mut() {
        gc.missing.sort();
        gc.missing_mem_form.sort();
    }
    map
}

/// Encoding-level coverage for MM registers, independent of SSE mnemonic coverage.
pub fn lifted_mmx_codes() -> BTreeSet<String> {
    Code::values()
        .filter(|&code| {
            let info = code.op_code();
            let mm = (0..info.op_count()).any(|n| {
                matches!(
                    info.op_kind(n),
                    OpCodeOperandKind::mm_reg
                        | OpCodeOperandKind::mm_rm
                        | OpCodeOperandKind::mm_or_mem
                )
            });
            (mm || code == Code::Emms) && probe_code(code) == Some(Probe::Lifted)
        })
        .map(|c| format!("{c:?}"))
        .collect()
}

/// The set of x86 mnemonics the lifter actually handles, measured by the same probe
/// [`compute_coverage`] uses: every in-scope `iced_x86::Code` whose canonical form
/// lifts (`Probe::Lifted`) contributes its mnemonic (via `code.mnemonic()`, which
/// collapses all encodings of one op — `Add_rm64_r64`, `Add_r64_rm64`, … → `Add`).
/// Long-mode probe only; the coverage ratchet keys on mnemonics, not the
/// encoding-specific `Code` names. No Unicorn needed — it's pure lift.
pub fn lifted_mnemonics() -> BTreeSet<String> {
    let mut set = BTreeSet::new();
    for code in Code::values() {
        if code_gen(code).is_none() {
            continue;
        }
        // `LiftedRegOnly` counts here. This set is keyed on MNEMONICS, and at that
        // granularity "we lift this op" is true — the register/memory split is a
        // statement about forms and belongs in the coverage map, not in a ratchet that
        // cannot express it. Excluding them would declare a dozen ops we do lift as
        // stale entries.
        if let Some(Probe::Lifted | Probe::LiftedRegOnly) = probe_code(code) {
            set.insert(format!("{:?}", code.mnemonic()));
        }
    }
    set
}

/// Probe the whole in-scope ISA and aggregate: 64-bit long mode plus 32-bit compat.
pub fn compute_coverage() -> Coverage {
    Coverage {
        generations: mode_coverage(CpuMode::Long64),
        compat32: mode_coverage(CpuMode::Compat32),
    }
}

// --- artifacts: machine-readable JSON + human dashboard ---

/// Directory holding the checked-in compat artifacts (`backlog/docs/compat/`).
pub fn artifact_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("backlog")
        .join("docs")
        .join("compat")
}

impl Coverage {
    /// Pretty JSON for `backlog/docs/compat/coverage.json` (stable key order via BTreeMap).
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("serialize coverage") + "\n"
    }

    /// The human dashboard `backlog/docs/compat/isa-coverage.md`: one row per generation
    /// with the lifted/missing split and percentage, plus the concrete gap lists.
    pub fn to_markdown(&self) -> String {
        let mut s = String::new();
        // Backlog.md doc frontmatter so the generated dashboard is app-listed like the
        // curated docs. `created_date` is fixed (not "now") so regeneration is stable —
        // the bytes must match the checked-in file. `id` is stable too.
        s.push_str(
            "---\nid: doc-19\ntitle: 'ISA compatibility coverage'\ntype: other\n\
             created_date: '2026-07-06 11:25'\n---\n\n",
        );
        // The probe's honest limit ships with the map: this artifact is linked from the
        // README, so an unqualified number here is a published claim.
        s.push_str(
            "> **`reg_only` is the honest column.** A register-or-memory operand is \
             probed BOTH ways: iced files both alternatives under one `Code`, so \
             lifting the register form used to mark the whole `Code` covered even when \
             the lifter rejected `[mem]`. Codes in that state are now counted and \
             listed separately (`missing_mem_form`) rather than folded into `lifted` — \
             that silence is what let `vextract*`'s memory destination look supported \
             until a real guest binary trapped on it.\n>\n> The remaining \
             caveat is `unencodable`: operand shapes this probe still cannot \
             synthesize are neither covered nor counted against coverage.\n\n",
        );
        s.push_str("> **EVEX modifiers are probed separately.** These totals use canonical unmasked, non-broadcast operands. They do not claim support for `{k}`, `{z}`, or `{1toN}`. The `evex_modifiers` tests encode and re-decode those modifiers, sweep all EVEX opcode templates, and compare supported forms across execution tiers.\n\n");
        s.push_str("# ISA compatibility coverage\n\n");
        s.push_str(
            "**Generated** by `cargo run -p x86jit-tests --bin compat -- --write` — do NOT edit \
             by hand. Measured by probing the real lifter (`x86jit-tests/src/compat.rs`): a \
             canonical instance of every in-scope `iced_x86::Code` is encoded and fed to \
             `lift_block`, per CPU mode. `lifted`/`missing` are of the *encodable* forms; \
             `unencodable` are exotic operand shapes the probe can't synthesize (not counted). \
             Kept honest by the `compat_map_is_current` test. The `optional` bucket \
             groups RDTSCP/RDRAND/RDSEED, SSE4A and ADX, which are not psABI level requirements \
             (psABI §3.1.1, Table 3.1, Micro-Architecture Levels).\n\n",
        );
        s.push_str("## 64-bit long mode (Long64)\n\n");
        render_mode_table(&mut s, &self.generations);
        s.push_str(
            "\n## 32-bit compat mode (Compat32)\n\n\
             Probed at bitness 32: also covers the legacy-only forms long mode dropped \
             (`Pushad`/`Into`/`Daa`/…) and the 16-bit operand-size forms \
             (`Call_rm16`/`Retnw`/`Pushaw`/…). `CpuMode::Real16` exists and is \
             interpreter-only; this generated map does not yet include a 16-bit real-mode \
             table. Real16 validation uses the [SingleStepTests 80286 \
             corpus](../../../x86jit-tests/tests/ss286.rs) and [targeted Unicorn \
             comparisons](../../../x86jit-tests/tests/cf16.rs).\n\n",
        );
        render_mode_table(&mut s, &self.compat32);
        render_missing(&mut s, "long64", &self.generations);
        render_missing(&mut s, "compat32", &self.compat32);
        render_reg_only(&mut s, "long64", &self.generations);
        render_reg_only(&mut s, "compat32", &self.compat32);
        s
    }

    /// Write both artifacts into `backlog/docs/compat/`.
    pub fn write_artifacts(&self) -> std::io::Result<()> {
        let dir = artifact_dir();
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join("coverage.json"), self.to_json())?;
        std::fs::write(dir.join("isa-coverage.md"), self.to_markdown())?;
        Ok(())
    }

    /// Load the checked-in JSON (for the staleness test).
    pub fn load_checked_in() -> std::io::Result<Coverage> {
        let text = std::fs::read_to_string(artifact_dir().join("coverage.json"))?;
        Ok(serde_json::from_str(&text).expect("parse coverage.json"))
    }
}

/// One per-generation summary table for a mode's coverage map.
fn render_mode_table(s: &mut String, map: &BTreeMap<String, GenCoverage>) {
    s.push_str("| generation | lifted | reg_only | missing | % of encodable | unencodable |\n");
    s.push_str("|---|---:|---:|---:|---:|---:|\n");
    for (g, c) in map {
        let known = c.lifted + c.unsupported;
        let pct = if known > 0 {
            100.0 * c.lifted as f64 / known as f64
        } else {
            0.0
        };
        s.push_str(&format!(
            "| {g} | {} | {} | {} | {pct:.0}% | {} |\n",
            c.lifted, c.reg_only, c.unsupported, c.unencodable
        ));
    }
}

/// The concrete gap lists for a mode, headed `## <mode> <generation> — missing (n)`.
fn render_missing(s: &mut String, mode: &str, map: &BTreeMap<String, GenCoverage>) {
    for (g, c) in map {
        if c.missing.is_empty() {
            continue;
        }
        s.push_str(&format!(
            "\n## {mode} {g} — missing ({})\n\n",
            c.missing.len()
        ));
        for m in &c.missing {
            s.push_str(&format!("- `{m}`\n"));
        }
    }
}

/// The `reg_only` gap lists — Codes whose register form lifts and whose memory form does
/// not. Rendered because the note at the top of this artifact promises them by name — a
/// bare count is a claim the reader cannot check.
fn render_reg_only(s: &mut String, mode: &str, map: &BTreeMap<String, GenCoverage>) {
    for (g, c) in map {
        if c.missing_mem_form.is_empty() {
            continue;
        }
        s.push_str(&format!(
            "\n## {mode} {g} — register form lifts, memory form does not ({})\n\n",
            c.missing_mem_form.len()
        ));
        for m in &c.missing_mem_form {
            s.push_str(&format!("- `{m}`\n"));
        }
    }
}

// --- CPUID ⇄ coverage consistency ---

/// One architectural instruction-feature bit advertised by a CPUID source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdvertisedCpuidFeature {
    pub leaf: u32,
    pub subleaf: u32,
    pub register: &'static str,
    pub bit: u32,
    pub feature: CpuidFeature,
}

/// Incomplete coverage for one advertised bit. Unencodable probes are reported
/// separately: they are unknown coverage, never silently treated as implemented.
#[derive(Clone, Debug)]
pub struct CpuidLiftGap {
    pub advertised: AdvertisedCpuidFeature,
    pub missing: Vec<Code>,
    pub unencodable: Vec<Code>,
}

/// Check any CPUID source, including `|l, s| preset.cpuid(l, s)`,
/// `|l, s| table.query(l, s)`, or `|l, s| cpu.cpuid(l, s)` with vCPU overrides.
/// Uses iced's per-Code requirements directly, without generation buckets. Only
/// long-mode forms available at CPL 3 are probed. Both register and memory forms
/// must lift. A Code counts only when all of its requirements with known CPUID
/// bits are advertised; iced's HLE_or_RTM and SKINIT_or_SVM remain alternatives.
/// This checks lift coverage, not execution semantics or CPU consistency.
pub fn cpuid_lift_gaps(
    source: impl FnMut(u32, u32) -> x86jit_core::CpuidRegisters,
) -> Vec<CpuidLiftGap> {
    let advertised_features = advertised_instruction_features(source);
    let known: HashSet<_> = advertised_instruction_features(|_, _| {
        x86jit_core::CpuidRegisters::new(u32::MAX, u32::MAX, u32::MAX, u32::MAX)
    })
    .into_iter()
    .map(|f| f.feature)
    .collect();
    let enabled: HashSet<_> = advertised_features.iter().map(|f| f.feature).collect();
    advertised_features
        .into_iter()
        .filter_map(|advertised| {
            let mut missing = Vec::new();
            let mut unencodable = Vec::new();
            for code in Code::values() {
                if !code.op_code().cpl3()
                    || !code.cpuid_features().contains(&advertised.feature)
                    || code
                        .cpuid_features()
                        .iter()
                        .any(|f| known.contains(f) && !enabled.contains(f))
                {
                    continue;
                }
                match probe_instruction(code, CpuMode::Long64) {
                    Some(Probe::Unsupported | Probe::LiftedRegOnly) => missing.push(code),
                    Some(Probe::Unencodable) => unencodable.push(code),
                    _ => {}
                }
            }
            (!missing.is_empty() || !unencodable.is_empty()).then_some(CpuidLiftGap {
                advertised,
                missing,
                unencodable,
            })
        })
        .collect()
}

/// Instruction-feature bits understood by iced and enumerated by Intel/AMD CPUID.
/// Bits for platform capabilities without instruction requirements are not lift
/// claims. Old vendor-specific features absent from the cited Intel/AMD tables
/// (for example Centaur and Cyrix leaves), and bits now reserved by those tables,
/// are outside this checker's scope.
pub fn advertised_instruction_features(
    mut source: impl FnMut(u32, u32) -> x86jit_core::CpuidRegisters,
) -> Vec<AdvertisedCpuidFeature> {
    use CpuidFeature::*;
    let max_basic = source(0, 0).eax;
    let max_extended = source(0x8000_0000, 0).eax;
    let mut out = Vec::new();
    let mut add = |leaf, subleaf, register, bits: &[(u32, CpuidFeature)]| {
        if (leaf < 0x8000_0000 && leaf > max_basic) || (leaf >= 0x8000_0000 && leaf > max_extended)
        {
            return;
        }
        if leaf == 7 && subleaf > source(7, 0).eax {
            return;
        }
        let regs = source(leaf, subleaf);
        let value = match register {
            "eax" => regs.eax,
            "ebx" => regs.ebx,
            "ecx" => regs.ecx,
            "edx" => regs.edx,
            _ => unreachable!(),
        };
        for &(bit, feature) in bits {
            if value & (1 << bit) != 0 {
                out.push(AdvertisedCpuidFeature {
                    leaf,
                    subleaf,
                    register,
                    bit,
                    feature,
                });
            }
        }
    };
    // SDM Vol 1 Tables 21-9 and 21-10.
    add(
        1,
        0,
        "ecx",
        &[
            (0, SSE3),
            (1, PCLMULQDQ),
            (3, MONITOR),
            (5, VMX),
            (6, SMX),
            (9, SSSE3),
            (12, FMA),
            (13, CMPXCHG16B),
            (19, SSE4_1),
            (20, SSE4_2),
            (22, MOVBE),
            (23, POPCNT),
            (25, AES),
            (26, XSAVE),
            (28, AVX),
            (29, F16C),
            (30, RDRAND),
        ],
    );
    add(
        1,
        0,
        "edx",
        &[
            (0, FPU),
            (4, TSC),
            (5, MSR),
            (8, CX8),
            (11, SEP),
            (15, CMOV),
            (19, CLFSH),
            (23, MMX),
            (24, FXSR),
            (25, SSE),
            (26, SSE2),
        ],
    );
    // SDM Vol 1 Tables 21-19, 21-20 and 21-21.
    add(
        7,
        0,
        "ebx",
        &[
            (0, FSGSBASE),
            (3, BMI1),
            (4, HLE),
            (5, AVX2),
            (8, BMI2),
            (10, INVPCID),
            (11, RTM),
            (14, MPX),
            (16, AVX512F),
            (17, AVX512DQ),
            (18, RDSEED),
            (19, ADX),
            (20, SMAP),
            (21, AVX512_IFMA),
            (23, CLFLUSHOPT),
            (24, CLWB),
            (26, AVX512PF),
            (27, AVX512ER),
            (28, AVX512CD),
            (29, SHA),
            (30, AVX512BW),
            (31, AVX512VL),
        ],
    );
    add(
        7,
        0,
        "ecx",
        &[
            (0, PREFETCHWT1),
            (1, AVX512_VBMI),
            (3, PKU),
            (5, WAITPKG),
            (6, AVX512_VBMI2),
            (7, CET_SS),
            (8, GFNI),
            (9, VAES),
            (10, VPCLMULQDQ),
            (11, AVX512_VNNI),
            (12, AVX512_BITALG),
            (14, AVX512_VPOPCNTDQ),
            (22, RDPID),
            (23, KL),
            (25, CLDEMOTE),
            (27, MOVDIRI),
            (28, MOVDIR64B),
            (29, ENQCMD),
        ],
    );
    add(
        7,
        0,
        "edx",
        &[
            (2, AVX512_4VNNIW),
            (3, AVX512_4FMAPS),
            (5, UINTR),
            (8, AVX512_VP2INTERSECT),
            (14, SERIALIZE),
            (16, TSXLDTRK),
            (18, PCONFIG),
            (20, CET_IBT),
            (22, AMX_BF16),
            (23, AVX512_FP16),
            (24, AMX_TILE),
            (25, AMX_INT8),
        ],
    );
    // SDM Vol 1 Tables 21-22, 21-23 and 21-25.
    add(
        7,
        1,
        "eax",
        &[
            (0, SHA512),
            (1, SM3),
            (2, SM4),
            (4, AVX_VNNI),
            (5, AVX512_BF16),
            (7, CMPCCXADD),
            (17, FRED),
            (18, LKGS),
            (19, WRMSRNS),
            (21, AMX_FP16),
            (22, HRESET),
            (23, AVX_IFMA),
            (27, MSRLIST),
        ],
    );
    add(7, 1, "ebx", &[(1, TSE)]);
    add(
        7,
        1,
        "edx",
        &[
            (4, AVX_VNNI_INT8),
            (5, AVX_NE_CONVERT),
            (8, AMX_COMPLEX),
            (10, AVX_VNNI_INT16),
            (14, PREFETCHITI),
        ],
    );
    // SDM Vol 1 Tables 21-35, 21-47, 21-51 and 21-62.
    add(0xd, 1, "eax", &[(0, XSAVEOPT), (1, XSAVEC), (3, XSAVES)]);
    add(0x12, 0, "eax", &[(0, SGX1)]);
    add(0x14, 0, "ebx", &[(4, PTWRITE)]);
    add(0x19, 0, "ebx", &[(0, AESKLE), (2, WIDE_KL)]);
    // AMD APM Vol 3 §E.4.2 (including duplicate standard-feature bits in EDX).
    add(
        0x8000_0001,
        0,
        "ecx",
        &[
            (2, SVM),
            (5, LZCNT),
            (6, SSE4A),
            (8, PREFETCHW),
            (11, XOP),
            (12, SKINIT),
            (15, LWP),
            (16, FMA4),
            (21, TBM),
            (29, MONITORX),
        ],
    );
    add(
        0x8000_0001,
        0,
        "edx",
        &[
            (0, FPU),
            (4, TSC),
            (5, MSR),
            (8, CX8),
            (11, SYSCALL),
            (15, CMOV),
            (23, MMX),
            (24, FXSR),
            (27, RDTSCP),
            (29, X64),
            (30, D3NOWEXT),
            (31, D3NOW),
        ],
    );
    // AMD APM Vol 3 §E.4.7, Fn8000_0008_EBX.
    add(
        0x8000_0008,
        0,
        "ebx",
        &[
            (0, CLZERO),
            (3, INVLPGB),
            (4, RDPRU),
            (8, MCOMMIT),
            (9, WBNOINVD),
        ],
    );
    // iced uses disjunctive requirements for these instruction groups.
    let hle_rtm = out.iter().find(|f| matches!(f.feature, HLE | RTM)).copied();
    if let Some(bit) = hle_rtm {
        out.push(AdvertisedCpuidFeature {
            feature: HLE_or_RTM,
            ..bit
        });
    }
    let skinit_svm = out
        .iter()
        .find(|f| matches!(f.feature, SKINIT | SVM))
        .copied();
    if let Some(bit) = skinit_svm {
        out.push(AdvertisedCpuidFeature {
            feature: SKINIT_or_SVM,
            ..bit
        });
    }
    out
}

/// Path to the checked-in CPUID waiver file (features advertised but not yet fully
/// lifted, each with a reason).
pub fn cpuid_waiver_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("compat")
        .join("cpuid-waivers.ron")
}

/// Load (preset, feature, reason) waivers for partial or unknown lift coverage.
/// Panics on a malformed file (a broken waiver list must not silently
/// pass the consistency test).
pub fn cpuid_waivers() -> Vec<(String, String, String)> {
    let text = std::fs::read_to_string(cpuid_waiver_path()).expect("read cpuid-waivers.ron");
    ron::from_str(&text).expect("parse cpuid-waivers.ron")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_requires_complete_fwait_prefixed_instruction() {
        // Real16 lifts the 16-bit FSAVE/FSTENV images (xj-562.11); its other
        // unsupported waiting forms report Unsupported, which also gives no credit.
        for code in [Code::Fdisi, Code::Feni, Code::Fsave_m108byte, Code::Fsetpm] {
            assert_eq!(
                probe_code_in(code, CpuMode::Real16),
                Some(Probe::Unsupported),
                "Real16: {code:?} must not receive credit for lifting only FWAIT"
            );
        }
        for code in [Code::Fsave_m94byte, Code::Fstenv_m14byte] {
            assert_eq!(
                probe_code_in(code, CpuMode::Real16),
                Some(Probe::Lifted),
                "Real16: {code:?}"
            );
        }
        for mode in [CpuMode::Long64, CpuMode::Compat32] {
            for code in [
                Code::Fdisi,
                Code::Feni,
                Code::Fsave_m94byte,
                Code::Fsave_m108byte,
                Code::Fsetpm,
                Code::Fstenv_m14byte,
            ] {
                assert_eq!(
                    probe_code_in(code, mode),
                    Some(Probe::Unencodable),
                    "{mode:?}: {code:?} must not receive credit for lifting only FWAIT"
                );
            }
            // A waiting form whose remainder lifts still counts as complete.
            assert_eq!(
                probe_code_in(Code::Fstenv_m28byte, mode),
                Some(Probe::Lifted),
                "{mode:?}: supported waiting form"
            );
        }
    }

    #[test]
    fn probe_preserves_intentional_exceptions() {
        for mode in [CpuMode::Long64, CpuMode::Compat32, CpuMode::Real16] {
            for code in [Code::Ud2, Code::Int1, Code::Int3, Code::Int_imm8] {
                assert_eq!(
                    probe_code_in(code, mode),
                    Some(Probe::Lifted),
                    "{mode:?}: {code:?}"
                );
            }
            for vector in [6, 13] {
                assert_eq!(
                    probe_encoded(Code::Int_imm8, &[0xcd, vector], mode),
                    Some(Probe::Lifted),
                    "{mode:?}: INT {vector}"
                );
            }
        }
        assert_eq!(
            probe_code_in(Code::Into, CpuMode::Real16),
            Some(Probe::Lifted)
        );
    }

    #[test]
    fn probe_rejects_invalid_encoding_trap() {
        // SDM Vol 3A §7.15, Event 6: invalid encodings raise #UD.
        for (mode, bytes) in [
            (CpuMode::Long64, [0xf0, 0x90]),
            (CpuMode::Compat32, [0xf0, 0x90]),
            (CpuMode::Real16, [0xc4, 0xc0]),
        ] {
            assert_eq!(
                probe_encoded(Code::Nopd, &bytes, mode),
                Some(Probe::Unencodable),
                "{mode:?}"
            );
        }
    }

    #[test]
    fn probe_rejects_overlength_trap() {
        // SDM Vol 3A §7.15, Event 13: exceeding 15 bytes raises #GP(0).
        // The appended RET would be byte 16: the fault block covers every
        // probed byte, so the byte-length check alone cannot reject it.
        let overlong = [0x66; 15];
        for mode in [CpuMode::Long64, CpuMode::Compat32] {
            assert_eq!(
                probe_encoded(Code::Nopd, &overlong, mode),
                Some(Probe::Unencodable),
                "{mode:?}"
            );
        }
    }
}
