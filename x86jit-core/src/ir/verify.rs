//! Debug-only validation of the block-local temporary contract (§6, §7.2).

use super::{IrBlock, IrOp, Temp, Val};

/// The first temporary read without a preceding definition in this block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UndefinedTemp {
    pub op_index: usize,
    pub guest_addr: u64,
    pub temp: Temp,
}

impl IrBlock {
    /// Walk ops in order, checking all inputs before recording any outputs.
    /// A fault exits the block; a retry starts another walk from its entry.
    /// Architectural registers (including vector/opmask registers) are not temps.
    pub fn verify_temp_defs(&self) -> Result<(), UndefinedTemp> {
        let mut defined = std::collections::HashSet::new();
        let mut guest_addr = self.guest_start;
        for (op_index, op) in self.ops.iter().enumerate() {
            if let IrOp::InsnStart { guest_addr: addr } = op {
                guest_addr = *addr;
            }
            let mut check = |reads: &[Val], writes: &[Temp]| {
                for &value in reads {
                    if let Val::Temp(temp) = value {
                        if !defined.contains(&temp) {
                            return Err(UndefinedTemp {
                                op_index,
                                guest_addr,
                                temp,
                            });
                        }
                    }
                }
                defined.extend(writes.iter().copied());
                Ok(())
            };
            match op {
                IrOp::InsnStart { .. }
                | IrOp::Fence { .. }
                | IrOp::Cpuid
                | IrOp::ReadTsc { .. }
                | IrOp::Xgetbv
                | IrOp::MmxState { .. }
                | IrOp::VMov { .. }
                | IrOp::VMov256 { .. }
                | IrOp::VLogic { .. }
                | IrOp::VPackedBin { .. }
                | IrOp::VPackedShift { .. }
                | IrOp::VByteShift { .. }
                | IrOp::VShuffle32 { .. }
                | IrOp::VBlendW { .. }
                | IrOp::VBlendD { .. }
                | IrOp::VFma { .. }
                | IrOp::VAes { .. }
                | IrOp::VAesImc { .. }
                | IrOp::VAesKeygen { .. }
                | IrOp::VSha { .. }
                | IrOp::VGfni { .. }
                | IrOp::Movq2dq { .. }
                | IrOp::Movdq2q { .. }
                | IrOp::VPclmul { .. }
                | IrOp::VPsign { .. }
                | IrOp::VPackWide { .. }
                | IrOp::VPMAddWd { .. }
                | IrOp::VPMadd { .. }
                | IrOp::VShuffle32Wide { .. }
                | IrOp::VShuffle16 { .. }
                | IrOp::VShufps { .. }
                | IrOp::VPshufb { .. }
                | IrOp::VPshufbWide { .. }
                | IrOp::VAlignr { .. }
                | IrOp::VUnpackLow { .. }
                | IrOp::VPackUsWB { .. }
                | IrOp::VZeroUpper { .. }
                | IrOp::VZeroUpperAll { .. }
                | IrOp::VMovWide { .. }
                | IrOp::VMaskMov { .. }
                | IrOp::VLogic256 { .. }
                | IrOp::VLogicWide { .. }
                | IrOp::VPopcnt { .. }
                | IrOp::VInsertLaneWide { .. }
                | IrOp::VExtractLaneWide { .. }
                | IrOp::VPcmpStr { .. }
                | IrOp::VPcmpStrMask { .. }
                | IrOp::VInsertPs { .. }
                | IrOp::VInsertPs3 { .. }
                | IrOp::VDpps { .. }
                | IrOp::VAlign { .. }
                | IrOp::VPMovExtend { .. }
                | IrOp::VPMovExtendWide { .. }
                | IrOp::VPAbs { .. }
                | IrOp::VpUnaryLane { .. }
                | IrOp::VpBlendm { .. }
                | IrOp::VShuffLane { .. }
                | IrOp::VpMultishift { .. }
                | IrOp::VPBlendV { .. }
                | IrOp::VPBlendVX { .. }
                | IrOp::VBlendI { .. }
                | IrOp::VDppd { .. }
                | IrOp::VDp3 { .. }
                | IrOp::VPRound { .. }
                | IrOp::VMaskedLogic { .. }
                | IrOp::VMaskedPacked { .. }
                | IrOp::VMaskedShift { .. }
                | IrOp::VShiftReg { .. }
                | IrOp::VShiftVar { .. }
                | IrOp::VGf2p8 { .. }
                | IrOp::VPTernlog { .. }
                | IrOp::VPackedBin256 { .. }
                | IrOp::VPackedWide { .. }
                | IrOp::VBroadcast { .. }
                | IrOp::VBroadcastLane { .. }
                | IrOp::VPCmpToMask { .. }
                | IrOp::VPTestToMask { .. }
                | IrOp::VKOrTest { .. }
                | IrOp::VKMovKK { .. }
                | IrOp::VKUnpack { .. }
                | IrOp::VKBinOp { .. }
                | IrOp::VKNot { .. }
                | IrOp::VKShift { .. }
                | IrOp::VPmovNarrow { .. }
                | IrOp::VPermT2 { .. }
                | IrOp::VPerm1 { .. }
                | IrOp::VInsert128 { .. }
                | IrOp::VExtract128 { .. }
                | IrOp::VPshufb256 { .. }
                | IrOp::VPackedShift256 { .. }
                | IrOp::VPermq { .. }
                | IrOp::VPermd { .. }
                | IrOp::VPermilVar { .. }
                | IrOp::VPerm2i128 { .. }
                | IrOp::VPalignr256 { .. }
                | IrOp::VPtest { .. }
                | IrOp::VTestFp { .. }
                | IrOp::VFloatBin { .. }
                | IrOp::VHFloat { .. }
                | IrOp::VHInt { .. }
                | IrOp::VFloatMov { .. }
                | IrOp::VFloatCmpMask { .. }
                | IrOp::VFloatCmpMask256 { .. }
                | IrOp::VPackedCvt { .. }
                | IrOp::VPhMinPosUw { .. }
                | IrOp::VMpsadbw { .. }
                | IrOp::VCvtPh2Ps { .. }
                | IrOp::VCvtPs2Ph { .. }
                | IrOp::VPackedCvtWide256 { .. }
                | IrOp::VFloatUnary { .. }
                | IrOp::VFloatBin256 { .. }
                | IrOp::VFloatUnary256 { .. }
                | IrOp::VPackedCvt256 { .. }
                | IrOp::VShufps256 { .. }
                | IrOp::VUnpack256 { .. }
                | IrOp::VMoveHalf { .. }
                | IrOp::SetDf { .. }
                | IrOp::PushFlags { .. }
                | IrOp::CheckIretNt
                | IrOp::Unsupported { .. }
                | IrOp::SetIf { .. }
                | IrOp::SetSsShadow
                | IrOp::PushfReal
                | IrOp::PopfReal
                | IrOp::IntGate { .. }
                | IrOp::IntoGate { .. }
                | IrOp::IretReal
                | IrOp::SetCf { .. }
                | IrOp::Bcd { .. }
                | IrOp::LoopCx { .. }
                | IrOp::FarRet { .. }
                | IrOp::Lahf
                | IrOp::Sahf
                | IrOp::PushaReal
                | IrOp::PopaReal
                | IrOp::EnterReal { .. }
                | IrOp::Branch { .. }
                | IrOp::Ret { .. }
                | IrOp::Syscall { .. }
                | IrOp::Hlt
                | IrOp::Trap { .. } => check(&[], &[])?,
                IrOp::ReadReg { dst, .. }
                | IrOp::GetCond { dst, .. }
                | IrOp::Random { dst, .. }
                | IrOp::GetMxcsr { dst, .. }
                | IrOp::MmxRead { dst, .. }
                | IrOp::VToGpr { dst, .. }
                | IrOp::VReadChunk { dst, .. }
                | IrOp::VExtractW { dst, .. }
                | IrOp::VExtractLane { dst, .. }
                | IrOp::VMoveMaskB { dst, .. }
                | IrOp::VMoveMaskFp { dst, .. }
                | IrOp::VMoveMaskB256 { dst, .. }
                | IrOp::VKToGpr { dst, .. }
                | IrOp::ReadSelector { dst, .. } => check(&[], &[*dst])?,
                IrOp::WriteReg { src, .. }
                | IrOp::SetMxcsr { src, .. }
                | IrOp::MmxWrite { src, .. }
                | IrOp::VFromGpr { src, .. }
                | IrOp::VWriteChunk { src, .. }
                | IrOp::VInsertW { src, .. }
                | IrOp::VInsertLane { src, .. }
                | IrOp::VBroadcastGpr { src, .. }
                | IrOp::VKFromGpr { src, .. }
                | IrOp::VCvtFromInt { src, .. }
                | IrOp::VCvtFloat { src, .. }
                | IrOp::RestoreFlags { src, .. } => check(&[*src], &[])?,
                IrOp::Add { a, b, dst, .. }
                | IrOp::Sub { a, b, dst, .. }
                | IrOp::Adc { a, b, dst, .. }
                | IrOp::Adox { a, b, dst, .. }
                | IrOp::Sbb { a, b, dst, .. }
                | IrOp::And { a, b, dst, .. }
                | IrOp::Or { a, b, dst, .. }
                | IrOp::Xor { a, b, dst, .. }
                | IrOp::Shl { a, b, dst, .. }
                | IrOp::Shr { a, b, dst, .. }
                | IrOp::Sar { a, b, dst, .. }
                | IrOp::Rol { a, b, dst, .. }
                | IrOp::Ror { a, b, dst, .. }
                | IrOp::Rcl { a, b, dst, .. }
                | IrOp::Rcr { a, b, dst, .. }
                | IrOp::Bmi { a, b, dst, .. }
                | IrOp::MmxCompute { a, b, dst, .. } => check(&[*a, *b], &[*dst])?,
                IrOp::DoubleShift {
                    a, b, count, dst, ..
                } => check(&[*a, *b, *count], &[*dst])?,
                IrOp::Sext { a, dst, .. } | IrOp::Bswap { a, dst, .. } => check(&[*a], &[*dst])?,
                IrOp::Mul { a, b, lo, hi, .. } => check(&[*a, *b], &[*lo, *hi])?,
                IrOp::Div {
                    hi,
                    lo,
                    divisor,
                    quot,
                    rem,
                    ..
                } => check(&[*hi, *lo, *divisor], &[*quot, *rem])?,
                IrOp::Load { addr, dst, .. } => check(&[*addr], &[*dst])?,
                IrOp::Store { addr, src, .. } => check(&[*addr, *src], &[])?,
                IrOp::Bt { a, bit, result, .. } => check(&[*a, *bit], &[*result])?,
                IrOp::X87 {
                    addr,
                    operand_offset,
                    ..
                } => check(&[*addr, *operand_offset], &[])?,
                IrOp::FxState { addr, .. }
                | IrOp::XState { addr, .. }
                | IrOp::AtomicCasPair { addr, .. }
                | IrOp::VLoad { addr, .. }
                | IrOp::VStore { addr, .. }
                | IrOp::VPackedBinM { addr, .. }
                | IrOp::VLogicM { addr, .. }
                | IrOp::VBlendWM { addr, .. }
                | IrOp::VFmaM { addr, .. }
                | IrOp::VAesM { addr, .. }
                | IrOp::VAesImcM { addr, .. }
                | IrOp::VAesKeygenM { addr, .. }
                | IrOp::VShaM { addr, .. }
                | IrOp::VGfniM { addr, .. }
                | IrOp::VPclmulM { addr, .. }
                | IrOp::VPsignM { addr, .. }
                | IrOp::VPackWideM { addr, .. }
                | IrOp::VPMaddM { addr, .. }
                | IrOp::VShuffle16M { addr, .. }
                | IrOp::VShufpsM { addr, .. }
                | IrOp::VPshufbM { addr, .. }
                | IrOp::VAlignrM { addr, .. }
                | IrOp::VUnpackLowM { addr, .. }
                | IrOp::VLoadWide { addr, .. }
                | IrOp::VStoreWide { addr, .. }
                | IrOp::VMaskLoadMem { addr, .. }
                | IrOp::VMaskStoreMem { addr, .. }
                | IrOp::VVecMaskLoadMem { addr, .. }
                | IrOp::VVecMaskStoreMem { addr, .. }
                | IrOp::VLogic256M { addr, .. }
                | IrOp::VLogicWideM { addr, .. }
                | IrOp::VPopcntM { addr, .. }
                | IrOp::VExtractLaneWideM { addr, .. }
                | IrOp::VPcmpStrM { addr, .. }
                | IrOp::VPcmpStrMaskM { addr, .. }
                | IrOp::VInsertPsM { addr, .. }
                | IrOp::VInsertPsM3 { addr, .. }
                | IrOp::VDppsM { addr, .. }
                | IrOp::VPMovExtendM { addr, .. }
                | IrOp::VPBlendVM { addr, .. }
                | IrOp::VPBlendVXM { addr, .. }
                | IrOp::VBlendIM { addr, .. }
                | IrOp::VDppdM { addr, .. }
                | IrOp::VDp3M { addr, .. }
                | IrOp::VPRoundM { addr, .. }
                | IrOp::VShiftRegM { addr, .. }
                | IrOp::VGf2p8M { addr, .. }
                | IrOp::VPTernlogM { addr, .. }
                | IrOp::VPackedBin256M { addr, .. }
                | IrOp::VPackedWideM { addr, .. }
                | IrOp::VBroadcastM { addr, .. }
                | IrOp::VBroadcastLaneM { addr, .. }
                | IrOp::VPCmpToMaskM { addr, .. }
                | IrOp::VPTestToMaskM { addr, .. }
                | IrOp::VPmovNarrowMem { addr, .. }
                | IrOp::VPermT2M { addr, .. }
                | IrOp::VPerm1M { addr, .. }
                | IrOp::VInsert128M { addr, .. }
                | IrOp::VPshufb256M { addr, .. }
                | IrOp::VPtestM { addr, .. }
                | IrOp::VFloatBinM { addr, .. }
                | IrOp::VHFloatM { addr, .. }
                | IrOp::VHIntM { addr, .. }
                | IrOp::VFloatCmpMaskM { addr, .. }
                | IrOp::VFloatCmpMask256M { addr, .. }
                | IrOp::VMpsadbwM { addr, .. }
                | IrOp::VPermilVarM { addr, .. }
                | IrOp::VPerm2i128M { addr, .. }
                | IrOp::VShiftVarM { addr, .. }
                | IrOp::VCvtPs2PhM { addr, .. }
                | IrOp::VFloatBin256M { addr, .. }
                | IrOp::VFloatUnary256M { addr, .. }
                | IrOp::VPackedCvt256M { addr, .. }
                | IrOp::VShufps256M { addr, .. }
                | IrOp::VUnpack256M { addr, .. }
                | IrOp::VLoadHalf { addr, .. }
                | IrOp::VStoreHalf { addr, .. }
                | IrOp::CheckAlign { addr, .. } => check(&[*addr], &[])?,
                IrOp::Popcnt { src, dst, .. } | IrOp::VCvtToInt { src, dst, .. } => {
                    check(&[*src], &[*dst])?
                }
                IrOp::Crc32 { crc, src, dst, .. } => check(&[*crc, *src], &[*dst])?,
                IrOp::BitScan { src, old, dst, .. } => check(&[*src, *old], &[*dst])?,
                IrOp::AtomicRmw { addr, src, old, .. } => check(&[*addr, *src], &[*old])?,
                IrOp::AtomicCas {
                    addr,
                    expected,
                    src,
                    old,
                    ..
                } => check(&[*addr, *expected, *src], &[*old])?,
                IrOp::MmxConvert { lo, hi, .. } => check(&[*lo, *hi], &[])?,
                IrOp::VGather { base, segment, .. } => check(&[*base, *segment], &[])?,
                IrOp::VFloatCmp { a, b, .. } => check(&[*a, *b], &[])?,
                IrOp::VFloatUnaryM { src_addr, .. } => check(&[*src_addr], &[])?,
                IrOp::CheckIret { cs, ip, ss, .. } => {
                    check(&[*cs, *ip, ss.unwrap_or(Val::Imm(0))], &[])?
                }
                IrOp::BoundGate {
                    index,
                    lower,
                    upper,
                    ..
                } => check(&[*index, *lower, *upper], &[])?,
                IrOp::SegLimitCheck { offset, .. } => check(&[*offset], &[])?,
                IrOp::FarJump { cs, ip, .. } | IrOp::FarCall { cs, ip, .. } => {
                    check(&[*cs, *ip], &[])?
                }
                IrOp::RepString { seg_base, .. } => check(&[*seg_base], &[])?,
                IrOp::RepStringReal {
                    ds_base, es_base, ..
                } => check(&[*ds_base, *es_base], &[])?,
                IrOp::Jump { target, .. } | IrOp::Call { target, .. } => check(&[*target], &[])?,
                IrOp::PortIo { port, value, .. } => check(&[*port, *value], &[])?,
                IrOp::PortString { base, .. } => check(&[*base], &[])?,
                IrOp::Legacy(op) => match op {
                    crate::legacy::LegacyOp::WrappedLoad { dst, addr, .. } => {
                        check(&[*addr], &[*dst])?
                    }
                    crate::legacy::LegacyOp::WrappedStore { addr, value, .. } => {
                        check(&[*addr, *value], &[])?
                    }
                    crate::legacy::LegacyOp::Coprocessor20 { addr, .. } => check(&[*addr], &[])?,
                    crate::legacy::LegacyOp::FlagsImage { dst, .. } => check(&[], &[*dst])?,
                    crate::legacy::LegacyOp::LoadFlagsWord { value, .. } => check(&[*value], &[])?,
                    crate::legacy::LegacyOp::DivideTrap {
                        quot,
                        rem,
                        hi,
                        lo,
                        divisor,
                        ..
                    } => check(&[Val::Temp(*hi), *lo, *divisor], &[*quot, *rem])?,
                    crate::legacy::LegacyOp::CheckIndex20 {
                        index,
                        lower,
                        upper,
                        ..
                    } => check(
                        &[Val::Temp(*index), Val::Temp(*lower), Val::Temp(*upper)],
                        &[],
                    )?,
                    crate::legacy::LegacyOp::Reti20 { ip, cs, flags, sp } => check(
                        &[
                            Val::Temp(*ip),
                            Val::Temp(*cs),
                            Val::Temp(*flags),
                            Val::Temp(*sp),
                        ],
                        &[],
                    )?,
                    crate::legacy::LegacyOp::PopPsw20 { value } => {
                        check(&[Val::Temp(*value)], &[])?
                    }
                    crate::legacy::LegacyOp::Fpo2Escape20 { addr, .. } => check(&[*addr], &[])?,
                    crate::legacy::LegacyOp::WrappedString { .. }
                    | crate::legacy::LegacyOp::Interrupt20 { .. }
                    | crate::legacy::LegacyOp::Brkem20 { .. }
                    | crate::legacy::LegacyOp::BitField20 { .. }
                    | crate::legacy::LegacyOp::BcdString20 { .. }
                    | crate::legacy::LegacyOp::InhibitIrq20 => {}
                    crate::legacy::LegacyOp::Address { dst, offset, .. }
                    | crate::legacy::LegacyOp::DirectStackAddress { dst, offset, .. } => {
                        check(&[*offset], &[*dst])?
                    }
                    crate::legacy::LegacyOp::ReadSelector { dst, .. }
                    | crate::legacy::LegacyOp::ReadSystemSegment { dst, .. }
                    | crate::legacy::LegacyOp::Pop { dst, .. } => check(&[], &[*dst])?,
                    crate::legacy::LegacyOp::LoadSegment { value, .. }
                    | crate::legacy::LegacyOp::LoadSystemSegment { value, .. }
                    | crate::legacy::LegacyOp::Push { value, .. } => check(&[*value], &[])?,
                    crate::legacy::LegacyOp::Transfer { cs, ip, .. } => {
                        check(&[cs.unwrap_or(Val::Imm(0)), *ip], &[])?
                    }
                    crate::legacy::LegacyOp::Table { addr, .. } => check(&[*addr], &[])?,
                    crate::legacy::LegacyOp::ReadControl { dst, .. } => check(&[], &[*dst])?,
                    crate::legacy::LegacyOp::WriteControl { value, .. } => check(&[*value], &[])?,
                    crate::legacy::LegacyOp::LoadMsw { value } => check(&[*value], &[])?,
                    crate::legacy::LegacyOp::ClearTs
                    | crate::legacy::LegacyOp::Checkpoint
                    | crate::legacy::LegacyOp::SharedWalk => {}
                    crate::legacy::LegacyOp::QueryDescriptor { selector, .. } => {
                        check(&[*selector], &[])?
                    }
                    crate::legacy::LegacyOp::AdjustRpl {
                        selector,
                        rpl,
                        destination,
                    } => {
                        check(&[*selector, *rpl], &[])?;
                        if let crate::legacy::SelectorDestination::Memory(addr) = destination {
                            check(&[*addr], &[])?;
                        }
                    }
                    crate::legacy::LegacyOp::Bound {
                        index,
                        lower,
                        upper,
                        size: _,
                    } => check(&[*index, *lower, *upper], &[])?,
                    crate::legacy::LegacyOp::Interrupt { .. } => {}
                    crate::legacy::LegacyOp::AdjustStack { .. }
                    | crate::legacy::LegacyOp::Flags { .. }
                    | crate::legacy::LegacyOp::All { .. }
                    | crate::legacy::LegacyOp::Return { .. }
                    | crate::legacy::LegacyOp::Iret { .. }
                    | crate::legacy::LegacyOp::Loop { .. }
                    | crate::legacy::LegacyOp::String { .. }
                    | crate::legacy::LegacyOp::Enter {
                        size: _,
                        alloc: _,
                        level: _,
                    }
                    | crate::legacy::LegacyOp::Coprocessor {
                        bytes: _,
                        len: _,
                        wait: _,
                    }
                    | crate::legacy::LegacyOp::Leave { .. }
                    | crate::legacy::LegacyOp::CheckTaskReturn
                    | crate::legacy::LegacyOp::Refuse { .. } => {}
                },
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod legacy_probes;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuMode, FlagMask, Reg};

    fn block(ops: Vec<IrOp>) -> IrBlock {
        IrBlock {
            mode: CpuMode::Real16,
            guest_start: 0x100,
            ops,
            temp_count: 4,
            guest_len: 8,
            icount: 2,
            legacy_context: None,
        }
    }

    #[test]
    fn reports_first_read_and_instruction_before_defining_destination() {
        let ir = block(vec![
            IrOp::ReadReg {
                dst: 0,
                reg: Reg::Rax,
            },
            IrOp::InsnStart { guest_addr: 0x104 },
            IrOp::Add {
                dst: 1,
                a: Val::Temp(1),
                b: Val::Temp(2),
                size: 2,
                set_flags: FlagMask::ALL,
            },
        ]);
        assert_eq!(
            ir.verify_temp_defs(),
            Err(UndefinedTemp {
                op_index: 2,
                guest_addr: 0x104,
                temp: 1,
            })
        );
    }

    #[test]
    fn legacy_bound_checks_each_input_without_defining_a_destination() {
        for input in 0..3 {
            let mut values = [Val::Imm(0); 3];
            values[input] = Val::Temp(3);
            let ir = block(vec![IrOp::Legacy(crate::legacy::LegacyOp::Bound {
                index: values[0],
                lower: values[1],
                upper: values[2],
                size: 4,
            })]);
            assert_eq!(
                ir.verify_temp_defs(),
                Err(UndefinedTemp {
                    op_index: 0,
                    guest_addr: 0x100,
                    temp: 3,
                })
            );
        }
    }

    #[test]
    fn checks_auxiliary_and_optional_inputs() {
        for op in [
            IrOp::CheckIret {
                cs: Val::Imm(0),
                ip: Val::Imm(0),
                ss: Some(Val::Temp(3)),
            },
            IrOp::Div {
                quot: 0,
                rem: 1,
                hi: Val::Imm(0),
                lo: Val::Imm(4),
                divisor: Val::Temp(3),
                size: 2,
                signed: false,
            },
            IrOp::X87 {
                fault: super::super::MemoryAccess::default(),
                kind: crate::x87::FpuKind::Fld1,
                instruction: 0,
                addr: Val::Imm(0),
                operand_offset: Val::Temp(3),
                sti: 0,
            },
            IrOp::Legacy(crate::legacy::LegacyOp::Transfer {
                cs: Some(Val::Temp(3)),
                ip: Val::Imm(0),
                size: 2,
                return_ip: None,
            }),
            IrOp::Legacy(crate::legacy::LegacyOp::Table {
                addr: Val::Temp(3),
                idtr: false,
                load: true,
                size: 2,
            }),
            IrOp::Legacy(crate::legacy::LegacyOp::WriteControl {
                register: crate::legacy::ControlRegister::Cr0,
                value: Val::Temp(3),
            }),
            IrOp::Legacy(crate::legacy::LegacyOp::LoadMsw {
                value: Val::Temp(3),
            }),
        ] {
            assert_eq!(block(vec![op]).verify_temp_defs().unwrap_err().temp, 3);
        }
    }

    #[test]
    fn legacy_control_read_defines_its_output() {
        use crate::legacy::{ControlRegister, LegacyOp};
        let ops = vec![
            IrOp::Legacy(LegacyOp::ReadControl {
                dst: 3,
                register: ControlRegister::Cr0,
            }),
            IrOp::Legacy(LegacyOp::WriteControl {
                register: ControlRegister::Cr3,
                value: Val::Temp(3),
            }),
        ];
        assert_eq!(block(ops.clone()).verify_temp_defs(), Ok(()));
        assert_eq!(
            block(ops[1..].to_vec())
                .verify_temp_defs()
                .unwrap_err()
                .temp,
            3
        );
    }

    #[test]
    fn direct_stack_address_checks_its_offset_and_defines_its_address() {
        use crate::legacy::LegacyOp;
        let op = |offset| {
            IrOp::Legacy(LegacyOp::DirectStackAddress {
                dst: 3,
                offset,
                bytes: 2,
                vector: 12,
            })
        };
        assert_eq!(
            block(vec![op(Val::Temp(7))])
                .verify_temp_defs()
                .unwrap_err()
                .temp,
            7
        );
        let use_address = IrOp::Load {
            fault: Default::default(),
            dst: 4,
            addr: Val::Temp(3),
            size: 2,
        };
        assert_eq!(
            block(vec![op(Val::Imm(0x8000)), use_address.clone()]).verify_temp_defs(),
            Ok(())
        );
        assert_eq!(
            block(vec![use_address])
                .verify_temp_defs()
                .unwrap_err()
                .temp,
            3
        );
    }

    #[test]
    fn descriptor_ops_check_every_input_and_define_only_the_read_output() {
        use crate::legacy::{DescriptorQuery, LegacyOp, SelectorDestination, SystemRegister};
        for input in 0..3 {
            let mut values = [Val::Imm(0); 3];
            values[input] = Val::Temp(3);
            let op = LegacyOp::AdjustRpl {
                selector: values[0],
                rpl: values[1],
                destination: SelectorDestination::Memory(values[2]),
            };
            assert_eq!(
                block(vec![IrOp::Legacy(op)])
                    .verify_temp_defs()
                    .unwrap_err()
                    .temp,
                3
            );
        }
        for op in [
            LegacyOp::QueryDescriptor {
                selector: Val::Temp(3),
                query: DescriptorQuery::Read,
                allowed_system_types: 0,
                allow_conforming: true,
            },
            LegacyOp::LoadSystemSegment {
                value: Val::Temp(3),
                register: SystemRegister::Ldtr,
                allowed_types: 4,
            },
        ] {
            assert_eq!(
                block(vec![IrOp::Legacy(op)])
                    .verify_temp_defs()
                    .unwrap_err()
                    .temp,
                3
            );
        }
        for register in [SystemRegister::Ldtr, SystemRegister::Tr] {
            let ops = vec![
                IrOp::Legacy(LegacyOp::ReadSystemSegment { dst: 3, register }),
                IrOp::Legacy(LegacyOp::LoadSystemSegment {
                    register,
                    value: Val::Temp(3),
                    allowed_types: 4,
                }),
            ];
            assert_eq!(block(ops.clone()).verify_temp_defs(), Ok(()));
            assert_eq!(
                block(ops[1..].to_vec())
                    .verify_temp_defs()
                    .unwrap_err()
                    .temp,
                3
            );
        }
    }

    #[test]
    fn multiple_outputs_and_redefinitions_are_local_to_one_block() {
        let ops = vec![
            IrOp::Mul {
                lo: 0,
                hi: 1,
                a: Val::Imm(7),
                b: Val::Imm(6),
                size: 2,
                signed: false,
                set_flags: FlagMask::NONE,
            },
            IrOp::InsnStart { guest_addr: 0x104 },
            IrOp::Div {
                quot: 2,
                rem: 3,
                hi: Val::Temp(1),
                lo: Val::Temp(0),
                divisor: Val::Imm(2),
                size: 2,
                signed: false,
            },
            IrOp::Add {
                dst: 2,
                a: Val::Temp(2),
                b: Val::Temp(3),
                size: 2,
                set_flags: FlagMask::NONE,
            },
            IrOp::WriteReg {
                reg: Reg::Rax,
                src: Val::Temp(2),
                size: 2,
            },
        ];
        assert_eq!(block(ops.clone()).verify_temp_defs(), Ok(()));
        assert_eq!(
            block(ops[2..].to_vec())
                .verify_temp_defs()
                .unwrap_err()
                .temp,
            1
        );
    }
}
