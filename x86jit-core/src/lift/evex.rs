//! Shared EVEX operand/result lowering (§7.1).
use super::*;

// SDM Vol 2C §5.2, VFMADD/VFMSUB/VFNMADD/VFNMSUB/VFMADDSUB/VFMSUBADD:
// packed FMA reads the old destination and masks at the FP element width.
fn fma_element(mnemonic: Mnemonic) -> Option<u8> {
    use Mnemonic::*;
    match mnemonic {
        Vfmadd132ps | Vfmadd213ps | Vfmadd231ps | Vfmsub132ps | Vfmsub213ps | Vfmsub231ps
        | Vfnmadd132ps | Vfnmadd213ps | Vfnmadd231ps | Vfnmsub132ps | Vfnmsub213ps
        | Vfnmsub231ps | Vfmaddsub132ps | Vfmaddsub213ps | Vfmaddsub231ps | Vfmsubadd132ps
        | Vfmsubadd213ps | Vfmsubadd231ps => Some(4),
        Vfmadd132pd | Vfmadd213pd | Vfmadd231pd | Vfmsub132pd | Vfmsub213pd | Vfmsub231pd
        | Vfnmadd132pd | Vfnmadd213pd | Vfnmadd231pd | Vfnmsub132pd | Vfnmsub213pd
        | Vfnmsub231pd | Vfmaddsub132pd | Vfmaddsub213pd | Vfmaddsub231pd | Vfmsubadd132pd
        | Vfmsubadd213pd | Vfmsubadd231pd => Some(8),
        _ => None,
    }
}

// SDM Vol 1 §15.6.1 and Vol 2 §2.7.7: these packed operations have an
// element-wise destination mask. Element widths: Vol 2A §3.3 (packed FP
// arithmetic/logic), Vol 2B §4.3 (PACK/UNPCK/SQRT), Vol 2C §5.2 (VPERMIL/
// VPERM/VPSHUF and packed integer operations). Only lane-wise operations may
// use the same mask to select non-broadcast memory elements.
fn packed_shape(mnemonic: Mnemonic) -> Option<(u8, bool)> {
    use Mnemonic::*;
    Some(match mnemonic {
        Vaddps | Vsubps | Vmulps | Vdivps | Vminps | Vmaxps | Vsqrtps | Vandps | Vandnps
        | Vorps | Vxorps | Vpaddd | Vpsubd | Vpminsd | Vpminud | Vpmaxsd | Vpmaxud | Vpmulld
        | Vpandd | Vpandnd | Vpord | Vpxord | Vpabsd => (4, true),
        Vaddpd | Vsubpd | Vmulpd | Vdivpd | Vminpd | Vmaxpd | Vsqrtpd | Vandpd | Vandnpd
        | Vorpd | Vxorpd | Vpaddq | Vpsubq | Vpminsq | Vpminuq | Vpmaxsq | Vpmaxuq | Vpmullq
        | Vpandq | Vpandnq | Vporq | Vpxorq | Vpabsq => (8, true),
        Vpaddb | Vpsubb | Vpaddsb | Vpaddusb | Vpsubsb | Vpsubusb | Vpminsb | Vpminub | Vpmaxsb
        | Vpmaxub | Vpabsb => (1, true),
        Vpaddw | Vpsubw | Vpaddsw | Vpaddusw | Vpsubsw | Vpsubusw | Vpminsw | Vpminuw | Vpmaxsw
        | Vpmaxuw | Vpmullw | Vpabsw => (2, true),
        Vpermilps | Vpermps | Vpermd | Vshufps | Vmovshdup | Vmovsldup | Vunpcklps | Vunpckhps
        | Vpunpckldq | Vpunpckhdq | Vpshufd => (4, false),
        Vpermilpd | Vpermpd | Vpermq | Vshufpd | Vmovddup | Vunpcklpd | Vunpckhpd | Vpunpcklqdq
        | Vpunpckhqdq => (8, false),
        Vpunpcklbw | Vpunpckhbw | Vpacksswb | Vpackuswb | Vpalignr => (1, false),
        Vpunpcklwd | Vpunpckhwd | Vpackssdw | Vpackusdw | Vpshufhw | Vpshuflw => (2, false),
        // SDM Vol 2A §2.8.2 (Type E2): packed FMA suppresses inactive
        // memory elements, including the broadcast when no lane is active.
        _ => return fma_element(mnemonic).map(|elem| (elem, true)),
    })
}

fn scratch(insn: &Instruction) -> impl Iterator<Item = u8> + '_ {
    (0..32).filter(|&r| {
        !(0..insn.op_count()).any(|i| {
            insn.op_kind(i) == OpKind::Register
                && insn.op_register(i).is_vector_register()
                && insn.op_register(i).number() == usize::from(r)
        })
    })
}

fn save(reg: u8, ops: &mut Vec<IrOp>, tg: &mut TempGen) -> [Temp; 8] {
    std::array::from_fn(|chunk| {
        let dst = tg.fresh();
        ops.push(IrOp::VReadChunk {
            dst,
            src: reg,
            chunk: chunk as u8,
        });
        dst
    })
}

fn restore(reg: u8, saved: [Temp; 8], ops: &mut Vec<IrOp>) {
    for (chunk, src) in saved.into_iter().enumerate() {
        ops.push(IrOp::VWriteChunk {
            dst: reg,
            src: Val::Temp(src),
            chunk: chunk as u8,
        });
    }
}

fn vector_register(reg: u8, bytes: u16) -> Register {
    let base = match bytes {
        16 => Register::XMM0,
        32 => Register::YMM0,
        64 => Register::ZMM0,
        _ => unreachable!(),
    };
    Register::try_from(base as usize + reg as usize).unwrap()
}

pub(super) fn lift(
    insn: &Instruction,
    ops: &mut Vec<IrOp>,
    tg: &mut TempGen,
    mode: CpuMode,
    sequence: &mut mmx::Sequence,
) -> Result<bool, LiftError> {
    let mask = evex_writemask(insn);
    if let (Some((elem, lane_wise)), Some((dst, bytes))) =
        (packed_shape(insn.mnemonic()), vec_operand(insn, 0))
    {
        if bytes == 64
            && insn.mnemonic() == Mnemonic::Vpermilpd
            && insn.op_kind(2) == OpKind::Immediate8
        {
            return Err(unsupported_insn(insn));
        }
        let memory = (0..insn.op_count()).find(|&i| insn.op_kind(i) == OpKind::Memory);
        // SDM Vol 2C §5.2, VPERMD: Type E4NF reads the full table.
        // Preserve existing predicated handlers for such memory dependencies;
        // a VEX-era handler without an explicit mask still fails the guard.
        if memory.is_some() && mask.is_some() && !insn.is_broadcast() && !lane_wise {
            return lift_existing(insn, ops, tg, mode, sequence);
        }
        let mut free = scratch(insn);
        let result = free.next().unwrap();
        let source = free.next().unwrap();
        let mut plain = *insn;
        plain.set_op_mask(Register::None);
        plain.set_zeroing_masking(false);
        plain.set_is_broadcast(false);
        plain.set_op_register(0, vector_register(result, bytes));
        let saved_result = save(result, ops, tg);
        let saved_source = memory.map(|_| save(source, ops, tg));
        if let Some(index) = memory {
            let addr = effective_address(insn, ops, tg)?;
            ops.push(IrOp::VMaskLoadMem {
                fault: Default::default(),
                dst: source,
                addr,
                // SDM Vol 2A §2.8.4, Tables 2-45/2-52: E4NF
                // broadcasts read their scalar even with an empty writemask.
                k: if lane_wise { mask.unwrap_or(0) } else { 0 },
                elem,
                zeroing: true,
                bytes,
                broadcast: insn.is_broadcast().then(|| insn.memory_size().size() as u8),
            });
            plain.set_op_kind(index, OpKind::Register);
            plain.set_op_register(index, vector_register(source, bytes));
        }
        // All potentially faulting reads precede temporary vector writes. The
        // register-only body cannot fault; the scratch registers are then restored.
        if fma_element(insn.mnemonic()).is_some() {
            ops.push(IrOp::VMovWide {
                dst: result,
                src: dst,
                bytes,
            });
        }
        lift_insn_inner(&plain, ops, tg, mode, sequence)?;
        if let Some(k) = mask {
            ops.push(IrOp::VMaskMov {
                dst,
                src: result,
                k,
                elem,
                zeroing: insn.zeroing_masking(),
                bytes,
            });
        } else {
            ops.push(IrOp::VMovWide {
                dst,
                src: result,
                bytes,
            });
        }
        restore(result, saved_result, ops);
        if let Some(saved) = saved_source {
            restore(source, saved, ops);
        }
        return Ok(false);
    }
    // Compare/test and blend sources use element-wise fault suppression
    // (SDM Vol 1 §15.6.1; Vol 2A §2.8.4, Type E4).
    if (insn.op0_register().is_k()
        || matches!(insn.mnemonic(), Mnemonic::Vpblendmd | Mnemonic::Vpblendmq))
        && insn.op_kind(2) == OpKind::Memory
    {
        let (_, bytes) = vec_operand(insn, 1).or_unsupported(insn)?;
        let elem = insn.memory_size().element_size() as u8;
        let source = scratch(insn).next().unwrap();
        let saved = save(source, ops, tg);
        let addr = effective_address(insn, ops, tg)?;
        ops.push(IrOp::VMaskLoadMem {
            fault: Default::default(),
            dst: source,
            addr,
            k: mask.unwrap_or(0),
            elem,
            zeroing: true,
            bytes,
            broadcast: insn.is_broadcast().then(|| insn.memory_size().size() as u8),
        });
        let mut plain = *insn;
        plain.set_is_broadcast(false);
        plain.set_op_kind(2, OpKind::Register);
        plain.set_op_register(2, vector_register(source, bytes));
        let start = ops.len();
        lift_insn_inner(&plain, ops, tg, mode, sequence)?;
        if !ops[start..].iter().any(|op| {
            matches!(
                op,
                IrOp::VPCmpToMask { .. } | IrOp::VPTestToMask { .. } | IrOp::VpBlendm { .. }
            )
        }) {
            return Err(unsupported_insn(insn));
        }
        restore(source, saved, ops);
        return Ok(false);
    }
    if insn.is_broadcast() {
        return Err(unsupported_insn(insn));
    }
    lift_existing(insn, ops, tg, mode, sequence)
}

fn lift_existing(
    insn: &Instruction,
    ops: &mut Vec<IrOp>,
    tg: &mut TempGen,
    mode: CpuMode,
    sequence: &mut mmx::Sequence,
) -> Result<bool, LiftError> {
    let mask = evex_writemask(insn).unwrap();
    let start = ops.len();
    let end = lift_insn_inner(insn, ops, tg, mode, sequence)?;
    if !ops[start..].iter().any(|op| carries_mask(op, mask)) {
        return Err(unsupported_insn(insn));
    }
    Ok(end)
}

// A newly dispatched EVEX form must explicitly carry its predicate in the IR.
// Reusing a VEX-era handler is insufficient, even when its unmasked form works.
fn carries_mask(op: &IrOp, mask: u8) -> bool {
    use IrOp::*;
    match op {
        VMaskMov { k, .. }
        | VMaskLoadMem { k, .. }
        | VMaskStoreMem { k, .. }
        | VMaskedLogic { k, .. }
        | VMaskedPacked { k, .. }
        | VMaskedShift { k, .. }
        | VShiftReg { k, .. }
        | VShiftVar { k, .. }
        | VGf2p8 { k, .. }
        | VGf2p8M { k, .. }
        | VpBlendm { k, .. } => *k == mask,
        VFma { writemask, .. }
        | VFmaM { writemask, .. }
        | VShuffle32Wide { writemask, .. }
        | VPshufbWide { writemask, .. }
        | VPMovExtendWide { writemask, .. }
        | VPAbs { writemask, .. }
        | VpUnaryLane { writemask, .. }
        | VShuffLane { writemask, .. }
        | VpMultishift { writemask, .. }
        | VBroadcastLane { writemask, .. }
        | VBroadcastLaneM { writemask, .. }
        | VPCmpToMask { writemask, .. }
        | VPCmpToMaskM { writemask, .. }
        | VPTestToMask { writemask, .. }
        | VPTestToMaskM { writemask, .. }
        | VPmovNarrow { writemask, .. }
        | VPermT2 { writemask, .. }
        | VPermT2M { writemask, .. }
        | VPerm1 { writemask, .. }
        | VPerm1M { writemask, .. } => *writemask == Some(mask),
        _ => false,
    }
}
