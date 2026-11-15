//! System-free Direct lowering (decision-23 §§2–5; spec.md §17.10).
use super::legacy::load;
use super::*;
use crate::legacy::{LegacyOp as L, MachineContext, MachineRegime, SegmentClass, SegmentId};

pub(crate) const CONTEXT: MachineContext = MachineContext {
    segment_class: SegmentClass::Direct,
    generation: crate::CpuGeneration::I8086,
    regime: MachineRegime::Real,
    default_operand_bits: 16,
    default_address_bits: 16,
    stack_address_bits: 16,
    cpl: 0,
    paging_enabled: false,
};

// Internal address value: high word is the segment base; low word is the
// offset. Subsequent pointer-word additions retain the segment identity.
// UM §2.3 p. 2-12: every byte wraps within its segment; §2.8 p. 2-68: 20 bits.
pub(super) fn tagged_address(
    ops: &mut impl LiftBuffer,
    tg: &mut TempGen,
    segment: SegmentId,
    offset: Val,
) -> Val {
    let sel = legacy::selector(ops, tg, segment);
    let base = alu_none(ops, tg, |dst| IrOp::Shl {
        dst,
        a: sel,
        b: Val::Imm(36),
        size: 8,
        set_flags: FlagMask::NONE,
    });
    add_addr(base, offset, ops, tg)
}

fn stack(ops: &mut impl LiftBuffer, tg: &mut TempGen, delta: i64) -> Val {
    let offset = legacy::stack_offset(ops, tg, delta);
    tagged_address(ops, tg, SegmentId::Ss, offset)
}
fn peek(ops: &mut impl LiftBuffer, tg: &mut TempGen, delta: i64) -> Val {
    let addr = stack(ops, tg, delta);
    load(ops, tg, addr, 2)
}
fn image(ops: &mut impl LiftBuffer, tg: &mut TempGen, profile: &crate::ModelProfile) -> Val {
    let dst = tg.fresh();
    let q = profile.quirks;
    ops.push(IrOp::Legacy(L::FlagsImage {
        dst,
        clear: q.real_flags_clear_mask,
        set: q.real_flags_set_mask,
    }));
    Val::Temp(dst)
}
fn pop_flags(ops: &mut impl LiftBuffer, value: Val, delay_irq: bool) {
    // UM §2.2 Fig. 2-9 p. 2-7; §2.7 Fig. 2-32 p. 2-33.
    ops.push(IrOp::Legacy(L::LoadFlagsWord {
        value,
        mask: 0x0fd5,
        delay_irq,
    }));
}
fn write_segment(ops: &mut impl LiftBuffer, segment: SegmentId, value: Val, inhibit: bool) {
    let reg = match segment {
        SegmentId::Cs => Reg::Cs,
        SegmentId::Ds => Reg::Ds,
        SegmentId::Es => Reg::Es,
        SegmentId::Ss => Reg::Ss,
        _ => unreachable!("8086 segment"),
    };
    ops.push(IrOp::WriteReg {
        reg,
        src: value,
        size: 2,
    });
    if inhibit {
        ops.push(IrOp::SetSsShadow);
    }
}

// A temporary holding `value` (NEC operations name their operands by temporary).
fn temp(ops: &mut impl LiftBuffer, tg: &mut TempGen, value: Val) -> Temp {
    match value {
        Val::Temp(t) => t,
        value => {
            let t = tg.fresh();
            ops.push(IrOp::Or {
                dst: t,
                a: value,
                b: Val::Imm(0),
                size: 8,
                set_flags: FlagMask::NONE,
            });
            t
        }
    }
}
// (base + delta) as a sixteen-bit offset.
fn offset16(ops: &mut impl LiftBuffer, tg: &mut TempGen, base: Val, delta: i64) -> Val {
    let sum = alu_none(ops, tg, |dst| IrOp::Add {
        dst,
        a: base,
        b: Val::Imm(delta as u64),
        size: 8,
        set_flags: FlagMask::NONE,
    });
    alu_none(ops, tg, |dst| IrOp::And {
        dst,
        a: sum,
        b: Val::Imm(0xffff),
        size: 8,
        set_flags: FlagMask::NONE,
    })
}
fn store_word(
    ops: &mut impl LiftBuffer,
    tg: &mut TempGen,
    segment: SegmentId,
    offset: Val,
    value: Val,
) {
    let addr = tagged_address(ops, tg, segment, offset);
    ops.push(IrOp::Store {
        fault: Default::default(),
        addr,
        src: value,
        size: 2,
        order: MemOrder::None,
    });
}
fn load_word(ops: &mut impl LiftBuffer, tg: &mut TempGen, segment: SegmentId, offset: Val) -> Val {
    let addr = tagged_address(ops, tg, segment, offset);
    load(ops, tg, addr, 2)
}
fn write16(ops: &mut impl LiftBuffer, reg: Reg, src: Val) {
    ops.push(IrOp::WriteReg { reg, src, size: 2 });
}

// NEC V20/V30 forms with their own lowering (decision-23 C4a). Returns false
// for an instruction the shared 8086 lowering handles. Kept out of line so the
// shared lowering, which every twenty-bit lift runs, keeps its own size.
#[inline(never)]
fn nec_instruction(
    insn: &Instruction,
    ops: &mut impl LiftBuffer,
    tg: &mut TempGen,
    next: u16,
    profile: &'static crate::ModelProfile,
) -> Result<bool, LiftError> {
    use Mnemonic::*;
    match insn.mnemonic() {
        // NEC UM p. 12-148 PUSH R: temp <- SP, then AW CW DW BW temp BP IX IY
        // at SP-2 ... SP-16, SP <- SP - 16.
        Pusha => {
            for (i, reg) in [
                Reg::Rax,
                Reg::Rcx,
                Reg::Rdx,
                Reg::Rbx,
                Reg::Rsp,
                Reg::Rbp,
                Reg::Rsi,
                Reg::Rdi,
            ]
            .into_iter()
            .enumerate()
            {
                let value = read_reg(reg, ops, tg);
                let offset = legacy::stack_offset(ops, tg, -2 * (i as i64 + 1));
                store_word(ops, tg, SegmentId::Ss, offset, value);
            }
            let sp = legacy::stack_offset(ops, tg, -16);
            legacy::write_sp(ops, sp);
        }
        // NEC UM p. 12-152 POP R: IY IX BP from SP..SP+5, BW DW CW AW from
        // SP+8..SP+15; the word at SP+6 is not read. SP <- SP + 16.
        Popa => {
            let mut values = Vec::new();
            for (delta, reg) in [
                (0, Reg::Rdi),
                (2, Reg::Rsi),
                (4, Reg::Rbp),
                (8, Reg::Rbx),
                (10, Reg::Rdx),
                (12, Reg::Rcx),
                (14, Reg::Rax),
            ] {
                values.push((reg, peek(ops, tg, delta)));
            }
            let sp = legacy::stack_offset(ops, tg, 16);
            for (reg, value) in values {
                write16(ops, reg, value);
            }
            legacy::write_sp(ops, sp);
        }
        // NEC UM p. 12-152 PREPARE imm16,imm8. The level is the whole imm8
        // (no modulus is stated). Stack and frame words use SS (§10 p. 10-2).
        Enter => {
            let level = insn.immediate8_2nd();
            let bp = read_reg(Reg::Rbp, ops, tg);
            let mut delta = -2i64;
            let frame = legacy::stack_offset(ops, tg, delta);
            store_word(ops, tg, SegmentId::Ss, frame, bp);
            if level > 0 {
                for i in 1..i64::from(level) {
                    let source = offset16(ops, tg, bp, -2 * i);
                    let value = load_word(ops, tg, SegmentId::Ss, source);
                    delta -= 2;
                    let offset = legacy::stack_offset(ops, tg, delta);
                    store_word(ops, tg, SegmentId::Ss, offset, value);
                }
                delta -= 2;
                let offset = legacy::stack_offset(ops, tg, delta);
                store_word(ops, tg, SegmentId::Ss, offset, frame);
            }
            let sp = legacy::stack_offset(ops, tg, delta - i64::from(insn.immediate16()));
            write16(ops, Reg::Rbp, frame);
            legacy::write_sp(ops, sp);
        }
        // NEC UM p. 12-153 DISPOSE: SP <- BP, BP <- (SP + 1, SP), SP <- SP + 2.
        Leave => {
            let bp = read_reg(Reg::Rbp, ops, tg);
            let offset = offset16(ops, tg, bp, 0);
            let value = load_word(ops, tg, SegmentId::Ss, offset);
            let sp = offset16(ops, tg, bp, 2);
            write16(ops, Reg::Rbp, value);
            legacy::write_sp(ops, sp);
        }
        // NEC UM p. 12-170 CHKIND reg16,mem32: lower bound at mem32, upper at
        // mem32 + 2; BRK 5 when outside. The pushed PC is the next instruction's
        // offset (§3 p. 3-1).
        Bound => {
            let index = lower_read(insn, 0, ops, tg)?;
            let index = temp(ops, tg, index);
            let addr = effective_address(insn, ops, tg)?;
            let lower = load(ops, tg, addr, 2);
            let lower = temp(ops, tg, lower);
            let second = add_addr(addr, Val::Imm(2), ops, tg);
            let upper = load(ops, tg, second, 2);
            let upper = temp(ops, tg, upper);
            ops.push(IrOp::Legacy(L::CheckIndex20 {
                index,
                lower,
                upper,
                saved_ip: next,
                flags_set: profile.quirks.real_flags_set_mask,
            }));
        }
        // NEC UM pp. 12-28–29 INM/OUTM: one port transfer per element through
        // DW; INM stores at DS1:IY, OUTM reads [seg-spec:]IX (default DS0).
        Insb | Insw | Outsb | Outsw => {
            let dir_out = matches!(insn.mnemonic(), Outsb | Outsw);
            let segment = if dir_out {
                legacy::segment(insn.segment_prefix()).unwrap_or(SegmentId::Ds)
            } else {
                SegmentId::Es
            };
            let sel = legacy::selector(ops, tg, segment);
            let base = alu_none(ops, tg, |dst| IrOp::Shl {
                dst,
                a: sel,
                b: Val::Imm(36),
                size: 8,
                set_flags: FlagMask::NONE,
            });
            ops.push(IrOp::PortString {
                elem: if matches!(insn.mnemonic(), Insb | Outsb) {
                    1
                } else {
                    2
                },
                rep: insn.has_rep_prefix(),
                addr_bits: 16,
                base,
                dir_out,
                stack: false,
            });
        }
        // NEC UM p. 12-168 RETI; the MD rule is applied when it commits.
        Iret => {
            let ip = peek(ops, tg, 0);
            let cs = peek(ops, tg, 2);
            let flags = peek(ops, tg, 4);
            let sp = legacy::stack_offset(ops, tg, 6);
            let (ip, cs, flags, sp) = (
                temp(ops, tg, ip),
                temp(ops, tg, cs),
                temp(ops, tg, flags),
                temp(ops, tg, sp),
            );
            ops.push(IrOp::Legacy(L::Reti20 { ip, cs, flags, sp }));
        }
        // NEC UM p. 12-151 POP PSW.
        Popf => {
            let value = peek(ops, tg, 0);
            let value = temp(ops, tg, value);
            let sp = legacy::stack_offset(ops, tg, 2);
            ops.push(IrOp::Legacy(L::PopPsw20 { value }));
            legacy::write_sp(ops, sp);
        }
        // NEC UM §6 p. 6-4 and p. 12-172 EI: INT is held through the next
        // instruction whatever IE was; IE is set as STI sets IF.
        Sti if profile.quirks.ei_delay_unconditional => {
            ops.push(IrOp::SetIf { value: true });
            ops.push(IrOp::Legacy(L::InhibitIrq20));
        }
        // NEC UM §6 p. 6-4: INT and NMI are not acknowledged between MOV
        // reg16/mem16, sreg and the next instruction.
        Mov if insn.op1_register().is_segment_register()
            && profile.quirks.segment_store_inhibits_interrupts =>
        {
            legacy::lower(insn, ops, tg)?;
            ops.push(IrOp::Legacy(L::InhibitIrq20));
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// NEC V20/V30 BRKEM imm8 (NEC UM p. 12-169), decoded by the fetch.
pub(crate) fn brkem(
    start: u16,
    length: u32,
    next: u16,
    vector: u8,
    profile: &'static crate::ModelProfile,
) -> IrBlock {
    IrBlock {
        mode: CpuMode::Real16,
        guest_start: u64::from(start),
        guest_len: length,
        icount: 1,
        temp_count: 0,
        ops: vec![
            IrOp::InsnStart {
                guest_addr: u64::from(start),
            },
            IrOp::Legacy(L::Brkem20 {
                vector,
                saved_ip: next,
                flags_set: profile.quirks.real_flags_set_mask,
            }),
            IrOp::Legacy(L::SharedWalk),
        ],
        legacy_context: Some(CONTEXT),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn instruction(
    insn: &Instruction,
    bytes: &[u8],
    evidence: &[u8],
    start: u16,
    length: u32,
    next: u16,
    restart_ip: u16,
    profile: &'static crate::ModelProfile,
    tg: &mut TempGen,
) -> Result<IrBlock, LiftError> {
    let mut values = vec![IrOp::InsnStart {
        guest_addr: u64::from(start),
    }];
    // Shared lowering reads this entry, never the generation projection.
    let mut ops = legacy::ContextOps {
        ops: &mut values,
        context: CONTEXT,
        profile: Some(profile),
    };
    use Mnemonic::*;
    let opcode = bytes
        .iter()
        .position(|b| !matches!(b, 0x26 | 0x2e | 0x36 | 0x3e | 0xf2 | 0xf3))
        .unwrap();
    if matches!(bytes[opcode], 0xd8..=0xdf | 0x9b) {
        let mut raw = [0; 15];
        raw[..evidence.len()].copy_from_slice(evidence);
        let wait = bytes[opcode] == 0x9b;
        let memory = !wait && bytes[opcode + 1] & 0xc0 != 0xc0;
        let addr = if memory {
            let mut surrogate = bytes.to_vec();
            surrogate[opcode] = 0x8b;
            surrogate[opcode + 1] &= 0xc7;
            let address = Decoder::new(16, &surrogate, DecoderOptions::NO_INVALID_CHECK).decode();
            effective_address(&address, &mut ops, tg)?
        } else {
            Val::Imm(0)
        };
        let mut fp = Vec::new();
        let mut fp_ops = legacy::ContextOps {
            ops: &mut fp,
            context: CONTEXT,
            profile: Some(profile),
        };
        let _ = control::lift_x87(insn, &mut fp_ops, tg);
        let decoded = fp.iter().find_map(|op| match op {
            IrOp::X87 { kind, sti, .. } => Some((*kind, *sti)),
            _ => None,
        });
        ops.push(IrOp::Legacy(L::Coprocessor20 {
            kind: decoded.map(|v| v.0),
            sti: decoded.map_or(0, |v| v.1),
            addr,
            memory,
            wait,
            bytes: raw,
            len: evidence.len() as u8,
        }));
    } else if profile.isa == crate::ModelIsa::NecV20
        && nec_instruction(insn, &mut ops, tg, next, profile)?
    {
    } else {
        match insn.mnemonic() {
            Pushf => {
                let value = image(&mut ops, tg, profile);
                legacy::push_value(&mut ops, tg, value, 2);
            }
            Popf => {
                let value = peek(&mut ops, tg, 0);
                let sp = legacy::stack_offset(&mut ops, tg, 2);
                pop_flags(&mut ops, value, false);
                legacy::write_sp(&mut ops, sp);
            }
            Mov if insn.op0_register().is_segment_register() => {
                let value = lower_read(insn, 1, &mut ops, tg)?;
                write_segment(
                    &mut ops,
                    legacy::segment(insn.op0_register()).unwrap(),
                    value,
                    true,
                );
            }
            Pop if insn.op0_register().is_segment_register() => {
                let value = peek(&mut ops, tg, 0);
                let sp = legacy::stack_offset(&mut ops, tg, 2);
                legacy::write_sp(&mut ops, sp);
                write_segment(
                    &mut ops,
                    legacy::segment(insn.op0_register()).unwrap(),
                    value,
                    true,
                );
            }
            Les | Lds => {
                let addr = effective_address(insn, &mut ops, tg)?;
                let value = load(&mut ops, tg, addr, 2);
                let second = add_addr(addr, Val::Imm(2), &mut ops, tg);
                let sel = load(&mut ops, tg, second, 2);
                let target = lower_write_target(insn, 0, &mut ops, tg)?;
                emit_write(&mut ops, tg, target, value);
                write_segment(
                    &mut ops,
                    if insn.mnemonic() == Les {
                        SegmentId::Es
                    } else {
                        SegmentId::Ds
                    },
                    sel,
                    false,
                );
            }
            Call | Jmp
                if matches!(insn.op0_kind(), OpKind::FarBranch16)
                    || matches!(insn.code(), Code::Call_m1616 | Code::Jmp_m1616) =>
            {
                let (cs, ip) = legacy::far_target(insn, &mut ops, tg, 2)?;
                if insn.mnemonic() == Call {
                    let old_cs = legacy::selector(&mut ops, tg, SegmentId::Cs);
                    let sp = legacy::stack_offset(&mut ops, tg, -4);
                    let addr = tagged_address(&mut ops, tg, SegmentId::Ss, sp);
                    let high = alu_none(&mut ops, tg, |dst| IrOp::Shl {
                        dst,
                        a: old_cs,
                        b: Val::Imm(16),
                        size: 8,
                        set_flags: FlagMask::NONE,
                    });
                    let frame = alu_none(&mut ops, tg, |dst| IrOp::Or {
                        dst,
                        a: high,
                        b: Val::Imm(u64::from(next)),
                        size: 8,
                        set_flags: FlagMask::NONE,
                    });
                    ops.push(IrOp::Store {
                        fault: Default::default(),
                        addr,
                        src: frame,
                        size: 4,
                        order: MemOrder::None,
                    });
                    legacy::write_sp(&mut ops, sp);
                }
                write_segment(&mut ops, SegmentId::Cs, cs, false);
                ops.push(IrOp::Jump { target: ip });
            }
            Int | Int3 | Into => {
                ops.push(IrOp::Legacy(L::Interrupt20 {
                    vector: if insn.mnemonic() == Int {
                        insn.immediate8()
                    } else if insn.mnemonic() == Int3 {
                        3
                    } else {
                        4
                    },
                    saved_ip: next,
                    overflow_only: insn.mnemonic() == Into,
                    flags_set: profile.quirks.real_flags_set_mask,
                }));
            }
            Retf | Iret => {
                let ip = peek(&mut ops, tg, 0);
                let cs = peek(&mut ops, tg, 2);
                let flags = (insn.mnemonic() == Iret).then(|| peek(&mut ops, tg, 4));
                let extra = if insn.mnemonic() == Retf && insn.op_count() != 0 {
                    insn.immediate16()
                } else {
                    0
                };
                let sp = legacy::stack_offset(
                    &mut ops,
                    tg,
                    if flags.is_some() { 6 } else { 4 } + i64::from(extra),
                );
                if let Some(flags) = flags {
                    // 8086 Family UM §2.6 p. 2-24: IRET, like STI,
                    // delays external interrupts through its successor.
                    pop_flags(&mut ops, flags, profile.quirks.iret_delays_interrupts);
                }
                legacy::write_sp(&mut ops, sp);
                write_segment(&mut ops, SegmentId::Cs, cs, false);
                ops.push(IrOp::Jump { target: ip });
            }
            _ => {
                legacy::lower(insn, &mut ops, tg)?;
            }
        }
    }
    let q = profile.quirks;
    if q.shift_count_mask.is_none() {
        let mut normalized = Vec::new();
        for mut op in values {
            unmasked_count(&mut op, &mut normalized, tg);
            normalized.push(op);
        }
        values = normalized;
    }
    legacy::defer_store_flags(&mut values, tg);
    if values
        .iter()
        .any(|op| matches!(op, IrOp::Trap { vector, .. } if *vector > q.max_processor_interrupt))
    {
        return Err(unsupported_insn(insn));
    }
    let mut result = Vec::new();
    for op in values {
        match op {
            IrOp::Load {
                dst, addr, size, ..
            } => result.push(IrOp::Legacy(L::WrappedLoad { dst, addr, size })),
            IrOp::Store {
                addr, src, size, ..
            } => result.push(IrOp::Legacy(L::WrappedStore {
                addr,
                value: src,
                size,
            })),
            IrOp::SegLimitCheck { .. } => {}
            IrOp::Div {
                quot,
                rem,
                hi,
                lo,
                divisor,
                size,
                signed,
            } => {
                let hi = match hi {
                    Val::Temp(temp) => temp,
                    value => {
                        let temp = tg.fresh();
                        result.push(IrOp::Or {
                            dst: temp,
                            a: value,
                            b: Val::Imm(0),
                            size: 8,
                            set_flags: FlagMask::NONE,
                        });
                        temp
                    }
                };
                result.push(IrOp::Legacy(L::DivideTrap {
                    quot,
                    rem,
                    hi,
                    lo,
                    divisor,
                    size,
                    signed,
                    reject_minimum: q.idiv_rejects_signed_minimum,
                    saved_ip: if q.divide_saves_next_ip { next } else { start },
                }));
            }
            IrOp::Legacy(L::String {
                kind,
                elem,
                rep,
                source,
                ..
            }) => result.push(IrOp::Legacy(L::WrappedString {
                kind,
                elem,
                rep,
                source,
                restart_ip: if q.rep_restart_prefixes.is_some() {
                    restart_ip
                } else {
                    start
                },
            })),
            IrOp::Legacy(L::DirectStackAddress { dst, offset, .. }) => {
                let value = tagged_address(&mut result, tg, SegmentId::Ss, offset);
                result.push(IrOp::Or {
                    dst,
                    a: value,
                    b: Val::Imm(0),
                    size: 8,
                    set_flags: FlagMask::NONE,
                });
            }
            IrOp::AtomicRmw {
                old,
                addr,
                src,
                size,
                op,
                ..
            } => {
                result.push(IrOp::Legacy(L::WrappedLoad {
                    dst: old,
                    addr,
                    size,
                }));
                let value = match op {
                    RmwOp::Xchg => src,
                    _ => return Err(unsupported_insn(insn)),
                };
                result.push(IrOp::Legacy(L::WrappedStore { addr, value, size }));
            }
            op => result.push(op),
        }
    }
    // All paths use the existing shared walk. There is no descriptor context.
    if !matches!(
        result.last(),
        Some(
            IrOp::Jump { .. }
                | IrOp::Branch { .. }
                | IrOp::LoopCx { .. }
                | IrOp::Hlt
                | IrOp::Legacy(L::Interrupt20 { .. } | L::CheckIndex20 { .. } | L::Reti20 { .. })
        )
    ) {
        result.push(IrOp::Jump {
            target: Val::Imm(u64::from(next)),
        });
    }
    result.push(IrOp::Legacy(L::SharedWalk));
    Ok(IrBlock {
        mode: CpuMode::Real16,
        guest_start: u64::from(start),
        guest_len: length,
        icount: 1,
        temp_count: tg.count(),
        ops: result,
        legacy_context: Some(CONTEXT),
    })
}

// UM §2.7 p. 2-39: up to 255 shifts. Reduce only to an equivalent count
// accepted by the shared operations; flags undefined for multi-bit counts use
// the engine's existing deterministic rule. Zero remains a true no-op.
fn unmasked_count(op: &mut IrOp, ops: &mut Vec<IrOp>, tg: &mut TempGen) {
    if matches!(op, IrOp::Shl { set_flags, .. } | IrOp::Shr { set_flags, .. } | IrOp::Sar { set_flags, .. } | IrOp::Rol { set_flags, .. } | IrOp::Ror { set_flags, .. } | IrOp::Rcl { set_flags, .. } | IrOp::Rcr { set_flags, .. } if set_flags.is_none())
    {
        return;
    }
    let (b, size, rotate) = match op {
        IrOp::Shl { b, size, .. } | IrOp::Shr { b, size, .. } | IrOp::Sar { b, size, .. } => {
            (b, *size, 0)
        }
        IrOp::Rol { b, size, .. } | IrOp::Ror { b, size, .. } => (b, *size, 1),
        IrOp::Rcl { b, size, .. } | IrOp::Rcr { b, size, .. } => (b, *size, 2),
        _ => return,
    };
    if matches!(b, Val::Imm(0..=1)) {
        return;
    }
    let count = alu_none(ops, tg, |dst| IrOp::And {
        dst,
        a: *b,
        b: Val::Imm(255),
        size: 8,
        set_flags: FlagMask::NONE,
    });
    let width = u64::from(size) * 8;
    let sub = alu_none(ops, tg, |dst| IrOp::Sub {
        dst,
        a: Val::Imm(0),
        b: count,
        size: 8,
        set_flags: FlagMask::NONE,
    });
    let nonzero = alu_none(ops, tg, |dst| IrOp::Shr {
        dst,
        a: sub,
        b: Val::Imm(63),
        size: 8,
        set_flags: FlagMask::NONE,
    });
    let full_mask = alu_none(ops, tg, |dst| IrOp::Sub {
        dst,
        a: Val::Imm(0),
        b: nonzero,
        size: 8,
        set_flags: FlagMask::NONE,
    });
    let reduced = if rotate == 1 {
        let minus = alu_none(ops, tg, |dst| IrOp::Sub {
            dst,
            a: count,
            b: Val::Imm(1),
            size: 8,
            set_flags: FlagMask::NONE,
        });
        let bits = alu_none(ops, tg, |dst| IrOp::And {
            dst,
            a: minus,
            b: Val::Imm(width - 1),
            size: 8,
            set_flags: FlagMask::NONE,
        });
        add_addr(bits, Val::Imm(1), ops, tg)
    } else if rotate == 2 {
        // CL is at most 255; division is unsigned and cannot fail here.
        let quot = tg.fresh();
        let rem = tg.fresh();
        ops.push(IrOp::Div {
            quot,
            rem,
            hi: Val::Imm(0),
            lo: count,
            divisor: Val::Imm(width + 1),
            size: 2,
            signed: false,
        });
        Val::Temp(rem)
    } else {
        // min(count, width+1), using an unsigned borrow mask without guest flags.
        let delta = alu_none(ops, tg, |dst| IrOp::Sub {
            dst,
            a: Val::Imm(width + 1),
            b: count,
            size: 8,
            set_flags: FlagMask::NONE,
        });
        let high = alu_none(ops, tg, |dst| IrOp::Sar {
            dst,
            a: delta,
            b: Val::Imm(63),
            size: 8,
            set_flags: FlagMask::NONE,
        });
        let excess = alu_none(ops, tg, |dst| IrOp::And {
            dst,
            a: delta,
            b: high,
            size: 8,
            set_flags: FlagMask::NONE,
        });
        add_addr(count, excess, ops, tg)
    };
    *b = alu_none(ops, tg, |dst| IrOp::And {
        dst,
        a: reduced,
        b: full_mask,
        size: 8,
        set_flags: FlagMask::NONE,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lowered(code: &[u8], profile: &'static crate::ModelProfile) -> String {
        let insn = Decoder::new(16, code, DecoderOptions::NO_INVALID_CHECK).decode();
        let len = insn.len() as u32;
        let next = 0x100 + len as u16;
        let ir = instruction(
            &insn,
            code,
            code,
            0x100,
            len,
            next,
            0x100,
            profile,
            &mut TempGen::new(),
        )
        .unwrap();
        format!("{:?}", ir.ops)
    }

    // Decision-23 §2: each catalogue entry supplies its own lowering policy.
    // The bus siblings share values today, so change one row and require the
    // lowering made with that row to follow it.
    #[test]
    fn lowering_reads_the_supplied_catalogue_entry() {
        for model in [crate::CpuModel::I8088, crate::CpuModel::I8086] {
            let p = model.profile();
            let row = |quirks: crate::ModelQuirks| -> &'static crate::ModelProfile {
                Box::leak(Box::new(crate::ModelProfile {
                    isa: p.isa,
                    quirks,
                    system: p.system,
                    physical_bits: p.physical_bits,
                    max_insn_len: p.max_insn_len,
                    fpu: p.fpu,
                    timing: p.timing,
                    semantics_id: p.semantics_id,
                    lift: p.lift,
                }))
            };
            // PUSH SP (legacy lowering) and MOV AX,[BX] (address shape).
            let mut quirks = p.quirks;
            quirks.push_sp_after_decrement = false;
            assert_ne!(
                lowered(b"\x54", p),
                lowered(b"\x54", row(quirks)),
                "{model:?}"
            );
            let mut quirks = p.quirks;
            quirks.wrap_segment_offsets = false;
            assert_ne!(
                lowered(b"\x8b\x07", p),
                lowered(b"\x8b\x07", row(quirks)),
                "{model:?}"
            );
        }
    }
}
