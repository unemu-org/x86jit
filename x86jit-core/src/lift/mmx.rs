//! MMX operands and commit ordering (§7.1).
use super::*;
use crate::ir::MmxOp;

/// Knowledge local to a basic block, never shared between region entries.
#[derive(Default)]
pub(super) struct Sequence {
    checked: bool,
    active: bool,
}

impl Sequence {
    pub(super) fn check(&mut self, i: &Instruction, ops: &mut impl LiftBuffer) {
        // SDM Vol 3A §15.5.1; Vol 2A §3.3 CVTPI2PD: only xmm,m64 is exempt.
        let mmx = handles(i)
            || matches!(
                i.mnemonic(),
                Mnemonic::Emms | Mnemonic::Femms | Mnemonic::Movq2dq | Mnemonic::Movdq2q
            );
        let exempt = i.mnemonic() == Mnemonic::Cvtpi2pd && i.op_kind(1) == OpKind::Memory;
        if mmx && !exempt && !self.checked {
            ops.push(IrOp::X87 {
                instruction: 0,
                fault: Default::default(),
                kind: crate::x87::FpuKind::Fwait,
                operand_offset: Val::Imm(0),
                addr: Val::Imm(0),
                sti: 0,
            });
            self.checked = true;
        }
    }

    pub(super) fn enter(&mut self, ops: &mut impl LiftBuffer) {
        // SDM Vol 3A §15.2 Table 15-2: MMX cannot create pending x87 exceptions
        // or change their masks. Once checked/entered, integer/SIMD
        // work cannot invalidate either fact. Every entry starts with fresh knowledge.
        // Call after all faulting accesses, before the first register commit.
        if !self.active {
            ops.push(IrOp::MmxState { empty: false });
            self.active = true;
        }
    }

    pub(super) fn observe(&mut self, ops: &[IrOp]) {
        if ops.iter().any(|op| {
            matches!(
                op,
                IrOp::X87 { .. }
                    | IrOp::FxState { restore: true, .. }
                    | IrOp::XState {
                        kind: crate::x87::xstate::XStateKind::Restore,
                        ..
                    }
                    | IrOp::MmxState { empty: true }
            )
        }) {
            *self = Self::default();
        }
    }
}

pub(super) fn handles(i: &Instruction) -> bool {
    ((0..i.op_count()).any(|n| i.op_register(n).is_mm())
        || matches!(i.mnemonic(), Mnemonic::Cvtpi2ps | Mnemonic::Cvtpi2pd))
        && !matches!(i.mnemonic(), Mnemonic::Movq2dq | Mnemonic::Movdq2q)
}

fn read(
    i: &Instruction,
    n: u32,
    size: u8,
    ops: &mut impl LiftBuffer,
    tg: &mut TempGen,
) -> Result<Val, LiftError> {
    if i.op_kind(n) == OpKind::Register && i.op_register(n).is_mm() {
        let dst = tg.fresh();
        ops.push(IrOp::MmxRead {
            dst,
            reg: reg_mmx(i, n).or_unsupported(i)?,
        });
        Ok(Val::Temp(dst))
    } else if i.op_kind(n) == OpKind::Memory {
        let addr = effective_address(i, ops, tg)?;
        let dst = tg.fresh();
        ops.push(IrOp::Load {
            fault: Default::default(),
            dst,
            addr,
            size,
        });
        Ok(Val::Temp(dst))
    } else {
        lower_read(i, n, ops, tg)
    }
}

fn write(
    i: &Instruction,
    value: Val,
    ops: &mut impl LiftBuffer,
    tg: &mut TempGen,
    sequence: &mut Sequence,
) -> Result<(), LiftError> {
    if i.op_kind(0) == OpKind::Register && i.op_register(0).is_mm() {
        sequence.enter(ops);
        ops.push(IrOp::MmxWrite {
            reg: reg_mmx(i, 0).or_unsupported(i)?,
            src: value,
        });
    } else {
        let target = lower_write_target(i, 0, ops, tg)?;
        if matches!(target, WriteTarget::Mem { .. }) {
            // A store must succeed before its x87 transition can commit.
            emit_write(ops, tg, target, value);
            sequence.enter(ops);
            return Ok(());
        }
        sequence.enter(ops);
        emit_write(ops, tg, target, value);
    }
    Ok(())
}

pub(super) fn lift(
    i: &Instruction,
    ops: &mut impl LiftBuffer,
    tg: &mut TempGen,
    sequence: &mut Sequence,
) -> Result<bool, LiftError> {
    use Mnemonic::*;
    let m = i.mnemonic();
    if matches!(m, Movd | Movq | Movntq) {
        let size = if m == Movd { 4 } else { 8 };
        let value = read(i, 1, size, ops, tg)?;
        // SDM Vol 2B §4.3 MOVD/MOVQ: MOVD clears MMn[63:32].
        let value = if m == Movd {
            alu_none(ops, tg, |dst| IrOp::And {
                dst,
                a: value,
                b: Val::Imm(0xffff_ffff),
                size: 8,
                set_flags: FlagMask::NONE,
            })
        } else {
            value
        };
        write(i, value, ops, tg, sequence)?;
        return Ok(false);
    }
    if matches!(
        m,
        Cvtpd2pi | Cvtps2pi | Cvttpd2pi | Cvttps2pi | Cvtpi2pd | Cvtpi2ps
    ) {
        return lift_conversion(i, ops, tg, sequence);
    }
    let (lane, packed) = match m {
        Paddb => (1, PackedBinOp::Add),
        Paddw => (2, PackedBinOp::Add),
        Paddd => (4, PackedBinOp::Add),
        Paddq => (8, PackedBinOp::Add),
        Psubb => (1, PackedBinOp::Sub),
        Psubw => (2, PackedBinOp::Sub),
        Psubd => (4, PackedBinOp::Sub),
        Psubq => (8, PackedBinOp::Sub),
        Paddsb => (1, PackedBinOp::AddSatS),
        Paddsw => (2, PackedBinOp::AddSatS),
        Paddusb => (1, PackedBinOp::AddSatU),
        Paddusw => (2, PackedBinOp::AddSatU),
        Psubsb => (1, PackedBinOp::SubSatS),
        Psubsw => (2, PackedBinOp::SubSatS),
        Psubusb => (1, PackedBinOp::SubSatU),
        Psubusw => (2, PackedBinOp::SubSatU),
        Pcmpeqb => (1, PackedBinOp::CmpEq),
        Pcmpeqw => (2, PackedBinOp::CmpEq),
        Pcmpeqd => (4, PackedBinOp::CmpEq),
        Pcmpgtb => (1, PackedBinOp::CmpGt),
        Pcmpgtw => (2, PackedBinOp::CmpGt),
        Pcmpgtd => (4, PackedBinOp::CmpGt),
        Pminub => (1, PackedBinOp::MinU),
        Pmaxub => (1, PackedBinOp::MaxU),
        Pminsw => (2, PackedBinOp::MinS),
        Pmaxsw => (2, PackedBinOp::MaxS),
        Pavgb => (1, PackedBinOp::AvgU),
        Pavgw => (2, PackedBinOp::AvgU),
        Pmullw => (2, PackedBinOp::MulLo16),
        Pmulhw => (2, PackedBinOp::MulHiS16),
        Pmulhuw => (2, PackedBinOp::MulHiU16),
        Pmuludq => (8, PackedBinOp::MulU32),
        Pmulhrsw => (2, PackedBinOp::MulHiRoundedS16),
        _ => (0, PackedBinOp::Add),
    };
    let op = if lane != 0 {
        MmxOp::Packed { lane, op: packed }
    } else {
        match m {
            Pand => MmxOp::Logic(VLogicOp::And),
            Pandn => MmxOp::Logic(VLogicOp::Andn),
            Por => MmxOp::Logic(VLogicOp::Or),
            Pxor => MmxOp::Logic(VLogicOp::Xor),
            Psllw | Pslld | Psllq | Psrlw | Psrld | Psrlq | Psraw | Psrad => MmxOp::Shift {
                lane: match m {
                    Psllw | Psrlw | Psraw => 2,
                    Pslld | Psrld | Psrad => 4,
                    _ => 8,
                },
                right: !matches!(m, Psllw | Pslld | Psllq),
                arith: matches!(m, Psraw | Psrad),
            },
            Punpcklbw | Punpcklwd | Punpckldq | Punpckhbw | Punpckhwd | Punpckhdq => {
                MmxOp::Unpack {
                    lane: match m {
                        Punpcklbw | Punpckhbw => 1,
                        Punpcklwd | Punpckhwd => 2,
                        _ => 4,
                    },
                    high: matches!(m, Punpckhbw | Punpckhwd | Punpckhdq),
                }
            }
            Packsswb | Packuswb | Packssdw => MmxOp::Pack {
                from: if m == Packssdw { 4 } else { 2 },
                signed: m != Packuswb,
            },
            Pmaddwd | Pmaddubsw => MmxOp::Madd {
                unsigned: m == Pmaddubsw,
            },
            Phaddw => MmxOp::Horizontal(HIntOp::AddW),
            Phaddd => MmxOp::Horizontal(HIntOp::AddD),
            Phaddsw => MmxOp::Horizontal(HIntOp::AddSw),
            Phsubw => MmxOp::Horizontal(HIntOp::SubW),
            Phsubd => MmxOp::Horizontal(HIntOp::SubD),
            Phsubsw => MmxOp::Horizontal(HIntOp::SubSw),
            Psadbw => MmxOp::Horizontal(HIntOp::Sad),
            Pabsb | Pabsw | Pabsd => MmxOp::Abs {
                lane: match m {
                    Pabsb => 1,
                    Pabsw => 2,
                    _ => 4,
                },
            },
            Psignb | Psignw | Psignd => MmxOp::Sign {
                lane: match m {
                    Psignb => 1,
                    Psignw => 2,
                    _ => 4,
                },
            },
            Pshufb => MmxOp::ShuffleBytes,
            Pshufw => MmxOp::ShuffleWords(i.immediate8()),
            Palignr => MmxOp::Align(i.immediate8()),
            Pinsrw => MmxOp::Insert(i.immediate8() & 3),
            Pextrw => MmxOp::Extract(i.immediate8() & 3),
            Pmovmskb => MmxOp::Mask,
            _ => return Err(unsupported_insn(i)),
        }
    };
    let unary = matches!(
        op,
        MmxOp::Abs { .. } | MmxOp::ShuffleWords(_) | MmxOp::Extract(_) | MmxOp::Mask
    );
    let a = read(i, if unary { 1 } else { 0 }, 8, ops, tg)?;
    // SDM Vol 2B §4.3 PUNPCKLBW/PUNPCKLWD/PUNPCKLDQ: the MMX memory source is m32.
    let size = if matches!(op, MmxOp::Unpack { high: false, .. }) {
        4
    } else if matches!(op, MmxOp::Insert(_)) {
        2
    } else {
        8
    };
    let b = if unary {
        Val::Imm(0)
    } else {
        read(i, 1, size, ops, tg)?
    };
    let dst = tg.fresh();
    ops.push(IrOp::MmxCompute { dst, a, b, op });
    write(i, Val::Temp(dst), ops, tg, sequence)?;
    Ok(false)
}

fn lift_conversion(
    i: &Instruction,
    ops: &mut impl LiftBuffer,
    tg: &mut TempGen,
    sequence: &mut Sequence,
) -> Result<bool, LiftError> {
    use Mnemonic::*;
    let kind = match i.mnemonic() {
        Cvtpi2ps => PackedCvtKind::Dq2Ps,
        Cvtpi2pd => PackedCvtKind::Dq2Pd,
        Cvtps2pi => PackedCvtKind::Ps2Dq,
        Cvttps2pi => PackedCvtKind::Tps2Dq,
        Cvtpd2pi => PackedCvtKind::Pd2Dq,
        Cvttpd2pi => PackedCvtKind::Tpd2Dq,
        _ => unreachable!(),
    };
    let to_mmx = !matches!(i.mnemonic(), Cvtpi2ps | Cvtpi2pd);
    let wide = matches!(i.mnemonic(), Cvtpd2pi | Cvttpd2pi);
    let mut parts = [Val::Imm(0); 2];
    if i.op_kind(1) == OpKind::Memory {
        let addr = effective_address(i, ops, tg)?;
        for (n, part) in parts.iter_mut().enumerate().take(if wide { 2 } else { 1 }) {
            let addr = if n == 0 {
                addr
            } else {
                add_addr(addr, Val::Imm(8), ops, tg)
            };
            let dst = tg.fresh();
            ops.push(IrOp::Load {
                fault: Default::default(),
                dst,
                addr,
                size: 8,
            });
            *part = Val::Temp(dst);
        }
    } else if to_mmx {
        let src = reg_xmm(i, 1).or_unsupported(i)?;
        for (n, part) in parts.iter_mut().enumerate().take(if wide { 2 } else { 1 }) {
            let dst = tg.fresh();
            ops.push(IrOp::VExtractLane {
                dst,
                src,
                index: n as u8,
                size: 8,
            });
            *part = Val::Temp(dst);
        }
    } else {
        parts[0] = read(i, 1, 8, ops, tg)?;
    }
    let dst = if to_mmx { reg_mmx(i, 0) } else { reg_xmm(i, 0) }.or_unsupported(i)?;
    // SDM Vol 2A §3.3 CVTPI2PD: the xmm,m64 form does not enter MMX state.
    if i.mnemonic() != Cvtpi2pd || i.op_kind(1) != OpKind::Memory {
        sequence.enter(ops);
    }
    // SDM Vol 2A §3.3 CVTPI2PS/CVTPI2PD/CVTPD2PI/CVTPS2PI and CVTT forms.
    ops.push(IrOp::MmxConvert {
        dst,
        lo: parts[0],
        hi: parts[1],
        kind,
        to_mmx,
    });
    Ok(false)
}
