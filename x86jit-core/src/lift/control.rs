use super::*;

/// `in`/`out` (imm8 or `dx` form) → `IrOp::PortIo`, a trap-out to the embedder
/// (§5.2). Operand layout (iced): `in acc, port` has op0 = accumulator (`al`/`ax`/
/// `eax`), op1 = the port (imm8 or `dx`); `out port, acc` is the mirror. The access
/// width is the accumulator's operand size (1/2/4). For `out` the accumulator value
/// is read here and carried in the exit; for `in` the embedder writes the result
/// back via `complete_port_in`.
pub(crate) fn lift_port_io(
    insn: &Instruction,
    ops: &mut impl LiftBuffer,
    tg: &mut TempGen,
    dir_out: bool,
) -> Result<(), LiftError> {
    let (acc_idx, port_idx) = if dir_out { (1, 0) } else { (0, 1) };
    let size = operand_size(insn, acc_idx);
    if !matches!(size, 1 | 2 | 4) {
        return Err(unsupported_insn(insn));
    }

    // Port is either an 8-bit immediate or `dx` (low 16 bits).
    let port = match insn.op_kind(port_idx) {
        OpKind::Immediate8 => Val::Imm(insn.immediate(port_idx) & 0xffff),
        OpKind::Register => {
            let dx = read_reg(Reg::Rdx, ops, tg);
            alu_none(ops, tg, |dst| IrOp::And {
                dst,
                a: dx,
                b: Val::Imm(0xffff),
                size: 8,
                set_flags: FlagMask::NONE,
            })
        }
        _ => return Err(unsupported_insn(insn)),
    };

    let value = if dir_out {
        read_reg(Reg::Rax, ops, tg)
    } else {
        Val::Imm(0)
    };

    ops.push(IrOp::PortIo {
        port,
        value,
        size,
        dir_out,
    });
    Ok(())
}

/// `push src` — long-mode default operand size is 8. Store BEFORE committing RSP so
/// a faulting store leaves RSP untouched for the retry (§16 pitfall #0).
pub(crate) fn lift_push(
    insn: &Instruction,
    ops: &mut impl LiftBuffer,
    tg: &mut TempGen,
    mode: CpuMode,
) -> Result<(), LiftError> {
    let size = push_pop_size(insn, mode);
    // `push Sreg` (CS/DS/ES/SS) pushes the 16-bit selector; the generic `lower_read`
    // rejects a segment-register operand, so read the selector directly (§17.6).
    let src = match sreg_operand(insn, 0) {
        Some(seg) => read_reg(seg, ops, tg),
        None => lower_read(insn, 0, ops, tg)?,
    };

    let rsp = read_reg(Reg::Rsp, ops, tg);
    let new_rsp = tg.fresh();
    ops.push(IrOp::Sub {
        dst: new_rsp,
        a: rsp,
        b: Val::Imm(size as u64),
        size: 8,
        set_flags: FlagMask::NONE,
    });
    // Compat32: ESP wraps mod 2^32 before it is used as the store address (a push at
    // ESP < slot must wrap, not carry into the upper half of the backing u64) (§16).
    emit_sp_wrap(ops, new_rsp, mode);
    let store_addr = stack_addr(Val::Temp(new_rsp), ops, tg, mode);
    ops.push(IrOp::Store {
        fault: Default::default(),
        addr: store_addr,
        src,
        size,
        order: MemOrder::None,
    });
    ops.push(IrOp::WriteReg {
        reg: Reg::Rsp,
        src: Val::Temp(new_rsp),
        size: sp_write_size(mode),
    });
    Ok(())
}

/// `pop dst` — Load BEFORE committing so a faulting load leaves state untouched.
///
/// Fault ordering (§16): `pop` is restartable — if the destination write faults, RSP
/// must be left un-advanced (fault-before-commit, matching hardware). So the RSP
/// write-back is ordered relative to the destination write by destination kind:
///
/// * **Register dst** — commit the RSP increment *first*, write the destination
///   register *last*. A register write can't fault, so ordering only matters for
///   `pop rsp`: writing the destination last lets the popped value override the RSP
///   increment.
/// * **Memory dst** — emit the (possibly-faulting) `Store` *first*, RSP write-back
///   *last*. On a store fault RSP is never committed, so a restarted `pop [mem]`
///   re-reads the same stack slot with the original RSP.
pub(crate) fn lift_pop(
    insn: &Instruction,
    ops: &mut impl LiftBuffer,
    tg: &mut TempGen,
    mode: CpuMode,
) -> Result<(), LiftError> {
    let size = push_pop_size(insn, mode);
    let rsp = read_reg(Reg::Rsp, ops, tg);
    let load_addr = stack_addr(rsp, ops, tg, mode);
    let val = tg.fresh();
    ops.push(IrOp::Load {
        fault: Default::default(),
        dst: val,
        addr: load_addr,
        size,
    });
    let new_rsp = tg.fresh();
    ops.push(IrOp::Add {
        dst: new_rsp,
        a: rsp,
        b: Val::Imm(size as u64),
        size: 8,
        set_flags: FlagMask::NONE,
    });
    // A 4-byte RSP write in Compat32 zero-extends → ESP wraps mod 2^32.
    let rsp_writeback = IrOp::WriteReg {
        reg: Reg::Rsp,
        src: Val::Temp(new_rsp),
        size: sp_write_size(mode),
    };
    // `pop Sreg` (DS/ES/SS — `pop cs` is not decodable on the 286) pops a 16-bit
    // selector into the segment register. A segment write can't fault, so it takes the
    // register path below (commit SP first).
    let dst = match sreg_operand(insn, 0) {
        Some(seg) => WriteTarget::Reg { reg: seg, size: 2 },
        None => lower_write_target(insn, 0, ops, tg)?,
    };
    match dst {
        WriteTarget::Mem { .. } => {
            // Store first (can fault), RSP write-back last → restartable on a store
            // fault. The effective address was lowered above from the *pre-pop* RSP; a
            // `pop [rsp+disp]` naming RSP as its own base/index is defined by the SDM
            // against post-increment RSP, but compilers never emit it — assert loudly
            // rather than silently address off the wrong RSP.
            debug_assert!(
                insn.memory_base() != Register::RSP && insn.memory_index() != Register::RSP,
                "pop [mem] with RSP-relative destination address is unsupported",
            );
            emit_write(ops, tg, dst, Val::Temp(val));
            ops.push(rsp_writeback);
        }
        _ => {
            // Register dst (incl. `pop rsp`): commit RSP first, destination last so the
            // popped value overrides the RSP increment.
            ops.push(rsp_writeback);
            emit_write(ops, tg, dst, Val::Temp(val));
        }
    }
    if sreg_operand(insn, 0) == Some(Reg::Ss) {
        ops.push(IrOp::SetSsShadow);
    }
    Ok(())
}

/// `true` for a far `jmp` (§17.6): a direct `ptr16:16` operand (`EA`) or the `m16:16`
/// memory form (`FF /5`). Distinguishes far from the near `jmp r/m16` (`FF /4`), which
/// shares the `Jmp` mnemonic.
pub(crate) fn is_far_jmp(insn: &Instruction) -> bool {
    matches!(insn.op_kind(0), OpKind::FarBranch16 | OpKind::FarBranch32)
        || insn.code() == Code::Jmp_m1616
}

/// `true` for a far `call` (§17.6): a direct `ptr16:16` operand (`9A`) or the `m16:16`
/// memory form (`FF /3`) — as opposed to the near `call r/m16` (`FF /2`).
pub(crate) fn is_far_call(insn: &Instruction) -> bool {
    matches!(insn.op_kind(0), OpKind::FarBranch16 | OpKind::FarBranch32)
        || insn.code() == Code::Call_m1616
}

/// A string op with its repeat prefix. movs/stos/lods take `rep`; scas/cmps take
/// `repe`/`repne` (both share the F3/F2 prefix bytes with the instruction kind).
pub(crate) fn lift_string(
    insn: &Instruction,
    ops: &mut impl LiftBuffer,
    tg: &mut TempGen,
    op: StrOp,
    elem: u8,
    mode: CpuMode,
) -> Result<bool, LiftError> {
    let f3 = insn.has_rep_prefix() || insn.has_repe_prefix();
    let f2 = insn.has_repne_prefix();
    let rep = match op {
        StrOp::Scas | StrOp::Cmps => {
            if f2 {
                RepKind::Repne
            } else if f3 {
                RepKind::Repe
            } else {
                RepKind::None
            }
        }
        _ => {
            if f3 {
                RepKind::Rep
            } else {
                RepKind::None
            }
        }
    };

    // SDM Vol 2B §4.3, REP Description/Operation (pp. 4-563–4-564),
    // STOS Description/Operation (pp. 4-676–4-677), MOVS Description/Operation
    // (pp. 4-104–4-105): in every mode, the
    // address-size attribute selects SI/ESI/RSI, DI/EDI/RDI and CX/ECX/RCX.
    // iced stores implicit string addressing in OpKind, not memory_base().
    let addr_bits: u8 = match (insn.op0_kind(), insn.op1_kind()) {
        (OpKind::MemorySegSI | OpKind::MemorySegDI | OpKind::MemoryESDI, _)
        | (_, OpKind::MemorySegSI | OpKind::MemorySegDI | OpKind::MemoryESDI) => 16,
        (OpKind::MemorySegESI | OpKind::MemorySegEDI | OpKind::MemoryESEDI, _)
        | (_, OpKind::MemorySegESI | OpKind::MemorySegEDI | OpKind::MemoryESEDI) => 32,
        _ => 64,
    };

    // Segment override on the DS-relative *source* pointer (RSI). Only movs/lods/cmps
    // read from DS:[RSI]; a `fs`/`gs` prefix there redirects the read. ES:[RDI]
    // (stos/scas dest, cmps second operand) is never overridable → base 0. FS/GS base
    // comes from the guest segment-base registers, exactly like `with_segment`.
    let reads_ds_source = matches!(op, StrOp::Movs | StrOp::Lods | StrOp::Cmps);

    // Real mode (§17.6): both segment bases are non-zero. The DS-relative source uses
    // `ds<<4` (or the override selector`<<4`); the ES-relative destination uses `es<<4`.
    // Emit the real-mode string op carrying both. Long/compat keep the flat-ES path.
    if mode.wraps_16() {
        // On the 286, an `F2` (REPNZ) prefix on MOVS/STOS/LODS is treated as plain REP —
        // the ZF-terminated repeat only applies to CMPS/SCAS. (The shared `rep` above maps
        // F2 to `None` for these ops, which is the long/compat convention; real mode must
        // repeat instead.) The corpus captures the repeated result.
        let rep = match op {
            StrOp::Movs | StrOp::Stos | StrOp::Lods if f2 => RepKind::Rep,
            _ => rep,
        };
        let src_seg = iced_segment_reg(insn.memory_segment()).unwrap_or(Reg::Ds);
        let ds_base = seg_base_shl4(src_seg, ops, tg);
        let es_base = seg_base_shl4(Reg::Es, ops, tg);
        ops.push(IrOp::RepStringReal {
            op,
            elem,
            rep,
            addr_bits,
            ds_base,
            es_base,
        });
        return Ok(false);
    }

    let seg_base = if reads_ds_source {
        match insn.segment_prefix() {
            Register::FS => read_reg(Reg::FsBase, ops, tg),
            Register::GS => read_reg(Reg::GsBase, ops, tg),
            _ => Val::Imm(0),
        }
    } else {
        Val::Imm(0)
    };

    ops.push(IrOp::RepString {
        fault: Default::default(),
        op,
        elem,
        rep,
        addr_bits,
        seg_base,
    });
    Ok(false)
}

/// Real-mode segment base `selector << 4` as a `Val` (§17.6): read the 16-bit selector
/// and shift it left by 4. Used by string ops and far-pointer loads.
fn seg_base_shl4(seg: Reg, ops: &mut impl LiftBuffer, tg: &mut TempGen) -> Val {
    let sel = read_reg(seg, ops, tg);
    alu_none(ops, tg, |dst| IrOp::Shl {
        dst,
        a: sel,
        b: Val::Imm(4),
        size: 8,
        set_flags: FlagMask::NONE,
    })
}

/// The ST(i) index referenced by an x87 instruction: the highest ST register
/// among its operands (ST0 is index 0, so a non-zero partner wins). Defaults to 1
/// for the implicit-`st1` forms (`faddp`, `fxch`).
pub(crate) fn st_index(insn: &Instruction) -> u8 {
    let mut idx = None;
    for i in 0..insn.op_count() {
        let r = insn.op_register(i);
        if r >= Register::ST0 && r <= Register::ST7 {
            let n = (r as u32 - Register::ST0 as u32) as u8;
            idx = Some(idx.map_or(n, |c: u8| c.max(n)));
        }
    }
    idx.unwrap_or(1)
}

/// For a register-form x87 arithmetic/store instruction, is the destination ST(0)?
/// (`fsub st(0), st(i)` vs `fsub st(i), st(0)` — op0 is the destination.) Chooses
/// between the `*Sti` (ST0-dest) and `*ToSti` (ST(i)-dest) IR kinds.
pub(crate) fn dst_is_st0(insn: &Instruction) -> bool {
    insn.op0_register() == Register::ST0
}

/// Lift one x87 FPU instruction to an `X87` IR op (§14). Memory operands are
/// reduced to an effective address; register forms carry ST(i) in `sti`.
pub(crate) fn lift_x87<B: LiftBuffer>(
    insn: &Instruction,
    ops: &mut B,
    tg: &mut TempGen,
) -> Result<(), LiftError> {
    use crate::x87::FpuKind as K;
    use Mnemonic::*;

    let mem = (0..insn.op_count()).any(|i| insn.op_kind(i) == OpKind::Memory);
    let msz = insn.memory_size().size();
    let sti = st_index(insn);

    // Emit an X87 op with a freshly computed address (memory forms) or a dummy.
    let emit = |kind: K, ops: &mut B, tg: &mut TempGen| -> Result<(), LiftError> {
        let (addr, operand_offset) = if mem {
            effective_address_parts(insn, ops, tg)?
        } else {
            (Val::Imm(0), Val::Imm(0))
        };
        ops.push(IrOp::X87 {
            operand_offset,
            instruction: 0,
            fault: Default::default(),
            kind,
            addr,
            // SDM Vol 1 §8.1.10, Figs. 8-10/8-12: real-address layouts.
            sti: if insn.code_size() == iced_x86::CodeSize::Code16
                && matches!(kind, K::Fldenv | K::Fnstenv | K::Fnsave | K::Frstor)
            {
                crate::x87::X87_ENV_REAL
                    | if matches!(msz, 14 | 94) {
                        crate::x87::X87_ENV16
                    } else {
                        0
                    }
            } else {
                sti
            },
        });
        Ok(())
    };

    match insn.mnemonic() {
        Fld => {
            let k = if mem {
                match msz {
                    4 => K::FldF32,
                    10 => K::FldF80,
                    _ => K::FldF64,
                }
            } else {
                K::FldSti
            };
            emit(k, ops, tg)?;
        }
        Fild => {
            let k = match msz {
                2 => K::FildI16,
                8 => K::FildI64,
                _ => K::FildI32,
            };
            emit(k, ops, tg)?;
        }
        Fst => emit(
            if !mem {
                K::FstSti
            } else if msz == 4 {
                K::FstF32
            } else {
                K::FstF64
            },
            ops,
            tg,
        )?,
        Fstp => {
            let k = if !mem {
                K::FstpSti
            } else {
                match msz {
                    4 => K::FstpF32,
                    10 => K::FstpF80,
                    _ => K::FstpF64,
                }
            };
            emit(k, ops, tg)?;
        }
        Fist => emit(
            match msz {
                2 => K::FistI16,
                4 => K::FistI32,
                _ => return Err(unsupported_insn(insn)),
            },
            ops,
            tg,
        )?,
        Fistp => emit(
            match msz {
                2 => K::FistpI16,
                8 => K::FistpI64,
                _ => K::FistpI32,
            },
            ops,
            tg,
        )?,
        Fisttp => emit(
            match msz {
                2 => K::FisttpI16,
                8 => K::FisttpI64,
                _ => K::FisttpI32,
            },
            ops,
            tg,
        )?,
        Fadd => emit(
            if !mem {
                if dst_is_st0(insn) {
                    K::FaddSti
                } else {
                    K::FaddToSti
                }
            } else if msz == 4 {
                K::FaddMemF32
            } else {
                K::FaddMemF64
            },
            ops,
            tg,
        )?,
        Faddp => emit(K::FaddP, ops, tg)?,
        Fsub => emit(
            if !mem {
                if dst_is_st0(insn) {
                    K::FsubSti
                } else {
                    K::FsubToSti
                }
            } else if msz == 4 {
                K::FsubMemF32
            } else {
                K::FsubMemF64
            },
            ops,
            tg,
        )?,
        Fsubp => emit(K::FsubP, ops, tg)?,
        Fsubr => emit(
            if !mem {
                if dst_is_st0(insn) {
                    K::FsubrSti
                } else {
                    K::FsubrToSti
                }
            } else if msz == 4 {
                K::FsubrMemF32
            } else {
                K::FsubrMemF64
            },
            ops,
            tg,
        )?,
        Fsubrp => emit(K::FsubrP, ops, tg)?,
        Fmul => emit(
            if !mem {
                if dst_is_st0(insn) {
                    K::FmulSti
                } else {
                    K::FmulToSti
                }
            } else if msz == 4 {
                K::FmulMemF32
            } else {
                K::FmulMemF64
            },
            ops,
            tg,
        )?,
        Fmulp => emit(K::FmulP, ops, tg)?,
        Fdiv => emit(
            if !mem {
                if dst_is_st0(insn) {
                    K::FdivSti
                } else {
                    K::FdivToSti
                }
            } else if msz == 4 {
                K::FdivMemF32
            } else {
                K::FdivMemF64
            },
            ops,
            tg,
        )?,
        Fdivp => emit(K::FdivP, ops, tg)?,
        Fdivr => emit(
            if !mem {
                if dst_is_st0(insn) {
                    K::FdivrSti
                } else {
                    K::FdivrToSti
                }
            } else if msz == 4 {
                K::FdivrMemF32
            } else {
                K::FdivrMemF64
            },
            ops,
            tg,
        )?,
        Fdivrp => emit(K::FdivrP, ops, tg)?,
        // `ficom`/`ficomp`: same `DA`/`DE` groups, reporting through the status-word
        // condition codes (C0/C2/C3) rather than EFLAGS.
        Ficom | Ficomp => {
            let k = match (insn.mnemonic(), msz) {
                (Ficom, 2) => K::FicomI16,
                (Ficom, 4) => K::FicomI32,
                (Ficomp, 2) => K::FicompI16,
                (Ficomp, 4) => K::FicompI32,
                _ => return Err(unsupported_insn(insn)),
            };
            emit(k, ops, tg)?;
        }
        // x87 integer-operand arithmetic: `DA /n` takes m32int, `DE /n` takes
        // m16int; there is no 64-bit form and no register form, so any other memory size
        // is refused rather than silently lifted as m32int.
        Fiadd | Fimul | Fisub | Fisubr | Fidiv | Fidivr => {
            let k = match (insn.mnemonic(), msz) {
                (Fiadd, 2) => K::FiaddMemI16,
                (Fiadd, 4) => K::FiaddMemI32,
                (Fimul, 2) => K::FimulMemI16,
                (Fimul, 4) => K::FimulMemI32,
                (Fisub, 2) => K::FisubMemI16,
                (Fisub, 4) => K::FisubMemI32,
                (Fisubr, 2) => K::FisubrMemI16,
                (Fisubr, 4) => K::FisubrMemI32,
                (Fidiv, 2) => K::FidivMemI16,
                (Fidiv, 4) => K::FidivMemI32,
                (Fidivr, 2) => K::FidivrMemI16,
                (Fidivr, 4) => K::FidivrMemI32,
                _ => return Err(unsupported_insn(insn)),
            };
            emit(k, ops, tg)?;
        }
        Fld1 => emit(K::Fld1, ops, tg)?,
        Fldz => emit(K::Fldz, ops, tg)?,
        Fldpi => emit(K::Fldpi, ops, tg)?,
        Fldl2e => emit(K::Fldl2e, ops, tg)?,
        Fldl2t => emit(K::Fldl2t, ops, tg)?,
        Fldlg2 => emit(K::Fldlg2, ops, tg)?,
        Fldln2 => emit(K::Fldln2, ops, tg)?,
        Frndint => emit(K::Frndint, ops, tg)?,
        Fxtract => emit(K::Fxtract, ops, tg)?,
        Fnop => emit(K::Fnop, ops, tg)?,
        Wait => emit(K::Fwait, ops, tg)?,
        Ffree => emit(K::Ffree, ops, tg)?,
        Fincstp => emit(K::Fincstp, ops, tg)?,
        Fdecstp => emit(K::Fdecstp, ops, tg)?,
        Fabs => emit(K::Fabs, ops, tg)?,
        Fchs => emit(K::Fchs, ops, tg)?,
        Fxch => emit(K::Fxch, ops, tg)?,
        Fcmovb => emit(K::Fcmovb, ops, tg)?,
        Fcmove => emit(K::Fcmove, ops, tg)?,
        Fcmovbe => emit(K::Fcmovbe, ops, tg)?,
        Fcmovu => emit(K::Fcmovu, ops, tg)?,
        Fcmovnb => emit(K::Fcmovnb, ops, tg)?,
        Fcmovne => emit(K::Fcmovne, ops, tg)?,
        Fcmovnbe => emit(K::Fcmovnbe, ops, tg)?,
        Fcmovnu => emit(K::Fcmovnu, ops, tg)?,
        Fcom | Fcomp => emit(
            match (insn.mnemonic(), mem, msz) {
                (Fcom, false, _) => K::Fcom,
                (Fcomp, false, _) => K::Fcomp,
                (Fcom, true, 4) => K::FcomMemF32,
                (Fcom, true, 8) => K::FcomMemF64,
                (Fcomp, true, 4) => K::FcompMemF32,
                (Fcomp, true, 8) => K::FcompMemF64,
                _ => return Err(unsupported_insn(insn)),
            },
            ops,
            tg,
        )?,
        Fcompp => emit(K::Fcompp, ops, tg)?,
        Fucom => emit(K::Fucom, ops, tg)?,
        Fucomp => emit(K::Fucomp, ops, tg)?,
        Fucompp => emit(K::Fucompp, ops, tg)?,
        Ftst => emit(K::Ftst, ops, tg)?,
        Fxam => emit(K::Fxam, ops, tg)?,
        Fsqrt => emit(K::Fsqrt, ops, tg)?,
        Fscale => emit(K::Fscale, ops, tg)?,
        Fprem1 => emit(K::Fprem1, ops, tg)?,
        Fucomi => emit(K::Fucomi, ops, tg)?,
        Fucomip => emit(K::Fucomip, ops, tg)?,
        Fcomi => emit(K::Fcomi, ops, tg)?,
        Fcomip => emit(K::Fcomip, ops, tg)?,
        Fldcw => emit(K::Fldcw, ops, tg)?,
        Fnstcw => emit(K::Fnstcw, ops, tg)?,
        // `fnstsw`/`fstsw`: `ax` form writes the status word to AX; the memory form
        // (`fnstsw m16`) stores it to [mem]. Distinct kinds so the exec knows whether
        // to touch AX or the effective address.
        Fnstsw => {
            if mem {
                emit(K::FnstswMem, ops, tg)?;
            } else {
                ops.push(IrOp::X87 {
                    instruction: 0,
                    fault: Default::default(),
                    kind: K::Fnstsw,
                    operand_offset: Val::Imm(0),
                    addr: Val::Imm(0),
                    sti: 0,
                });
            }
        }
        Fnstenv | Fstenv => {
            if msz != 28 && !(msz == 14 && insn.code_size() == iced_x86::CodeSize::Code16) {
                return Err(unsupported_insn(insn));
            }
            emit(K::Fnstenv, ops, tg)?
        }
        Fldenv => {
            if msz != 28 && !(msz == 14 && insn.code_size() == iced_x86::CodeSize::Code16) {
                return Err(unsupported_insn(insn));
            }
            emit(K::Fldenv, ops, tg)?
        }
        Fnsave | Fsave | Frstor if insn.code_size() == iced_x86::CodeSize::Code16 => {
            if msz != 94 {
                return Err(unsupported_insn(insn));
            }
            emit(
                if insn.mnemonic() == Frstor {
                    K::Frstor
                } else {
                    K::Fnsave
                },
                ops,
                tg,
            )?;
        }
        Fprem => emit(K::Fprem, ops, tg)?,
        // Transcendentals: f64-precision, ST(0)/ST(1)-implicit (no operand).
        Fsin => emit(K::Fsin, ops, tg)?,
        Fcos => emit(K::Fcos, ops, tg)?,
        Fptan => emit(K::Fptan, ops, tg)?,
        Fpatan => emit(K::Fpatan, ops, tg)?,
        F2xm1 => emit(K::F2xm1, ops, tg)?,
        Fyl2x => emit(K::Fyl2x, ops, tg)?,
        Fyl2xp1 => emit(K::Fyl2xp1, ops, tg)?,
        Fsincos => emit(K::Fsincos, ops, tg)?,
        // Unit management. The waiting forms (finit/fclex) map to the same kinds:
        // the implicit `fwait` is a no-op since FP exceptions aren't modeled.
        Fninit | Finit => emit(K::Fninit, ops, tg)?,
        Fnclex | Fclex => emit(K::Fnclex, ops, tg)?,
        _ => return Err(unsupported_insn(insn)),
    }
    Ok(())
}

/// The segment `Reg` named by operand `idx`, or `None` if it isn't a segment register.
/// CS/DS/ES/SS map to their selector `Reg`; FS/GS are 386+ and out of scope for the
/// 286 real-mode target (§17.6).
pub(crate) fn sreg_operand(insn: &Instruction, idx: u32) -> Option<Reg> {
    if insn.op_kind(idx) != OpKind::Register {
        return None;
    }
    iced_segment_reg(insn.op_register(idx))
}

pub(crate) fn lift_unsupported(
    insn: &Instruction,
    ops: &mut impl LiftBuffer,
) -> Result<bool, LiftError> {
    let LiftError::Unsupported { bytes, len, .. } = unsupported_insn(insn) else {
        unreachable!()
    };
    ops.push(IrOp::Unsupported { bytes, len });
    Ok(true)
}

pub(crate) fn lift_mov_sreg_user(
    insn: &Instruction,
    ops: &mut impl LiftBuffer,
    tg: &mut TempGen,
) -> Result<bool, LiftError> {
    if insn.op_register(0).is_segment_register() {
        return lift_unsupported(insn, ops);
    }
    let segment = (insn.op_register(1) as u32 - Register::ES as u32) as u8;
    let dst = tg.fresh();
    ops.push(IrOp::ReadSelector { dst, segment });
    // SDM Vol 2B §4.3 MOV: the memory destination is always a word; the
    // register destination follows operand size and zero extends at 32/64 bits.
    let target = lower_write_target(insn, 0, ops, tg)?;
    emit_write(ops, tg, target, Val::Temp(dst));
    Ok(false)
}

fn flags_size(insn: &Instruction) -> u8 {
    match insn.mnemonic() {
        Mnemonic::Pushfq | Mnemonic::Popfq | Mnemonic::Iretq => 8,
        Mnemonic::Pushfd | Mnemonic::Popfd | Mnemonic::Iretd => 4,
        _ => 2,
    }
}

pub(crate) fn lift_flag_stack(
    insn: &Instruction,
    ops: &mut impl LiftBuffer,
    tg: &mut TempGen,
    mode: CpuMode,
    pop: bool,
) -> Result<(), LiftError> {
    let size = flags_size(insn);
    if !pop {
        ops.push(IrOp::PushFlags {
            fault: Default::default(),
            size,
            wrap_sp: mode == CpuMode::Compat32,
        });
        return Ok(());
    }
    let rsp = read_reg(Reg::Rsp, ops, tg);
    let image = tg.fresh();
    ops.push(IrOp::Load {
        fault: Default::default(),
        dst: image,
        addr: rsp,
        size,
    });
    let next = alu_none(ops, tg, |dst| IrOp::Add {
        dst,
        a: rsp,
        b: Val::Imm(size as u64),
        size: 8,
        set_flags: FlagMask::NONE,
    });
    ops.push(IrOp::RestoreFlags {
        src: Val::Temp(image),
        size,
        iret: false,
    });
    ops.push(IrOp::WriteReg {
        reg: Reg::Rsp,
        src: next,
        size: sp_write_size(mode),
    });
    Ok(())
}

pub(crate) fn lift_iret_user(
    insn: &Instruction,
    ops: &mut impl LiftBuffer,
    tg: &mut TempGen,
    mode: CpuMode,
) -> Result<(), LiftError> {
    // SDM Vol 2A §3.2 IRET, Operation, IA-32e-MODE: reject NT before Pop().
    ops.push(IrOp::CheckIretNt);
    // SDM Vol 3A §11.3: serialize prior memory accesses before reading the frame.
    ops.push(IrOp::Fence {
        kind: FenceKind::Full,
    });
    let size = flags_size(insn);
    // SDM Vol 2A §3.2 IRET; Vol 3A §7.14.3: Long64 always restores SS:RSP.
    let count = if mode == CpuMode::Long64 { 5 } else { 3 };
    let rsp = read_reg(Reg::Rsp, ops, tg);
    let mut frame = Vec::new();
    for index in 0..count {
        let addr = tg.fresh();
        ops.push(IrOp::Add {
            dst: addr,
            a: rsp,
            b: Val::Imm(index * size as u64),
            size: 8,
            set_flags: FlagMask::NONE,
        });
        emit_sp_wrap(ops, addr, mode);
        let value = tg.fresh();
        ops.push(IrOp::Load {
            fault: Default::default(),
            dst: value,
            addr: Val::Temp(addr),
            size,
        });
        frame.push(Val::Temp(value));
    }
    ops.push(IrOp::CheckIret {
        cs: frame[1],
        ss: if count == 5 { Some(frame[4]) } else { None },
        ip: frame[0],
    });
    let new_rsp = if count == 5 {
        frame[3]
    } else {
        alu_none(ops, tg, |dst| IrOp::Add {
            dst,
            a: rsp,
            b: Val::Imm(count * size as u64),
            size: 8,
            set_flags: FlagMask::NONE,
        })
    };
    ops.push(IrOp::Fence {
        kind: FenceKind::Full,
    });
    ops.push(IrOp::RestoreFlags {
        src: frame[2],
        size,
        iret: true,
    });
    ops.push(IrOp::WriteReg {
        reg: Reg::Rsp,
        src: new_rsp,
        size: sp_write_size(mode),
    });
    ops.push(IrOp::Jump { target: frame[0] });
    Ok(())
}

/// SDM Vol 2A §3.3 INS (pp. 3-464–466); Vol 2B §4.3 OUTS
/// (pp. 4-173–176), REP (pp. 4-563–564): DX, address-sized indices/count,
/// fixed ES destination, overridable DS source, and byte/word/dword elements.
pub(crate) fn lift_port_string(
    insn: &Instruction,
    ops: &mut impl LiftBuffer,
    tg: &mut TempGen,
    mode: CpuMode,
    dir_out: bool,
    elem: u8,
) -> Result<bool, LiftError> {
    // Only F3 REP is defined for port strings (SDM Vol 2B §4.3 REP).
    if insn.has_repne_prefix() {
        return Err(unsupported_insn(insn));
    }
    let memory = insn.op_kind(if dir_out { 1 } else { 0 });
    let addr_bits = match memory {
        OpKind::MemorySegSI | OpKind::MemoryESDI => 16,
        OpKind::MemorySegESI | OpKind::MemoryESEDI => 32,
        _ => 64,
    };
    // Real16 follows the engine's 80286 address/operand-size scope (§17.6).
    if mode.wraps_16() && (addr_bits != 16 || elem > 2) {
        return Err(unsupported_insn(insn));
    }
    let base = if mode.wraps_16() {
        let seg = if dir_out {
            iced_segment_reg(insn.memory_segment()).unwrap_or(Reg::Ds)
        } else {
            Reg::Es
        };
        seg_base_shl4(seg, ops, tg)
    } else if dir_out {
        match insn.segment_prefix() {
            Register::FS => read_reg(Reg::FsBase, ops, tg),
            Register::GS => read_reg(Reg::GsBase, ops, tg),
            _ => Val::Imm(0),
        }
    } else {
        Val::Imm(0)
    };
    ops.push(IrOp::PortString {
        elem,
        rep: insn.has_rep_prefix(),
        addr_bits,
        base,
        dir_out,
        // SDM Vol 1 §3.3.7.1: in 64-bit mode an SS override on a non-stack reference is
        // ignored, so a non-canonical OUTS source raises #GP, not #SS.
        stack: dir_out && mode != CpuMode::Long64 && insn.memory_segment() == Register::SS,
    });
    Ok(true)
}
