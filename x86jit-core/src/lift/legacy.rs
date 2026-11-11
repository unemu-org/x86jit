//! Explicit-context legacy lowering (decision-18 §1; spec.md §§7, 17).
use super::*;
use crate::legacy::{ControlRegister, LegacyDecodeContext, LegacyOp as L, SegmentId};

struct Ops {
    values: Vec<IrOp>,
    context: LegacyDecodeContext,
}
impl std::ops::Deref for Ops {
    type Target = [IrOp];
    fn deref(&self) -> &[IrOp] {
        &self.values
    }
}
impl std::ops::DerefMut for Ops {
    fn deref_mut(&mut self) -> &mut [IrOp] {
        &mut self.values
    }
}
impl LiftBuffer for Ops {
    fn push(&mut self, op: IrOp) {
        self.values.push(op);
    }
    fn legacy_context(&self) -> Option<LegacyDecodeContext> {
        Some(self.context)
    }
}

pub(crate) fn segment(r: Register) -> Option<SegmentId> {
    Some(match r {
        Register::CS => SegmentId::Cs,
        Register::DS => SegmentId::Ds,
        Register::ES => SegmentId::Es,
        Register::SS => SegmentId::Ss,
        Register::FS => SegmentId::Fs,
        Register::GS => SegmentId::Gs,
        _ => return None,
    })
}

/// Decode exactly the fetched instruction; callers fetch incrementally and check CS.
pub(crate) fn instruction(
    code: &[u8],
    ip: u32,
    context: LegacyDecodeContext,
) -> Result<IrBlock, LiftError> {
    let mut decoder = Decoder::with_ip(
        u32::from(context.default_operand_bits),
        code,
        u64::from(ip),
        // 80386 PRM §17 MOV to/from Special Registers, p. 17-111:
        // recognize test-register MOV so deferred system accesses are typed.
        DecoderOptions::MOV_TR,
    );
    let insn = decoder.decode();
    let mode = if context.default_operand_bits == 16 {
        CpuMode::Real16
    } else {
        CpuMode::Compat32
    };
    let mut ops = Ops {
        values: Vec::new(),
        context,
    };
    let mut tg = TempGen::new();
    ops.push(IrOp::InsnStart {
        guest_addr: u64::from(ip),
    });
    // SDM Vol 2A §3.3 LGDT/LIDT; 80386 PRM §17 pp. 17-95,
    // 17-154: /0../3 require memory. Newer instruction aliases in these
    // ModRM register slots do not make them legal on the I386 generation.
    let opcode = code;
    let prefix_bytes = opcode
        .iter()
        .take_while(|b| {
            matches!(
                b,
                0x26 | 0x2e | 0x36 | 0x3e | 0x64 | 0x65 | 0x66 | 0x67 | 0xf0 | 0xf2 | 0xf3
            )
        })
        .count();
    let table_register_operand = matches!(&opcode[prefix_bytes..],
        [0x0f, 0x01, modrm, ..] if modrm & 0xe0 == 0xc0);
    if insn.is_invalid() || table_register_operand {
        // SDM Vol 3A §7.15, Events 6/13: #UD or an overlong instruction.
        let vector = if code.len() == 15 && exceeds_length_limit(code, mode) {
            13
        } else {
            6
        };
        ops.push(IrOp::Trap { vector, advance: 0 });
    } else {
        lower(&insn, &mut ops, &mut tg)
            .map_err(|e| refill_unsupported_bytes(e, code, u64::from(ip)))?;
        mark_memory_accesses(&insn, mode, &mut ops.values);
    }
    Ok(IrBlock {
        mode,
        guest_start: u64::from(ip),
        guest_len: code.len() as u32,
        icount: 1,
        temp_count: tg.count(),
        ops: ops.values,
        legacy_context: Some(context),
    })
}

fn emit(ops: &mut impl LiftBuffer, op: L) {
    ops.push(IrOp::Legacy(op));
}
fn address(
    insn: &Instruction,
    ops: &mut impl LiftBuffer,
    tg: &mut TempGen,
) -> Result<Val, LiftError> {
    effective_address(insn, ops, tg)
}
fn selector(ops: &mut impl LiftBuffer, tg: &mut TempGen, segment: SegmentId) -> Val {
    let dst = tg.fresh();
    emit(ops, L::ReadSelector { dst, segment });
    Val::Temp(dst)
}
fn load(ops: &mut impl LiftBuffer, tg: &mut TempGen, addr: Val, size: u8) -> Val {
    let dst = tg.fresh();
    ops.push(IrOp::Load {
        fault: Default::default(),
        dst,
        addr,
        size,
    });
    Val::Temp(dst)
}
fn far_target(
    insn: &Instruction,
    ops: &mut impl LiftBuffer,
    tg: &mut TempGen,
    size: u8,
) -> Result<(Val, Val), LiftError> {
    match insn.op_kind(0) {
        OpKind::FarBranch16 => Ok((
            Val::Imm(u64::from(insn.far_branch_selector())),
            Val::Imm(u64::from(insn.far_branch16())),
        )),
        OpKind::FarBranch32 => Ok((
            Val::Imm(u64::from(insn.far_branch_selector())),
            Val::Imm(u64::from(insn.far_branch32())),
        )),
        OpKind::Memory => {
            let a = address(insn, ops, tg)?;
            let ip = load(ops, tg, a, size);
            let a = add_addr(a, Val::Imm(u64::from(size)), ops, tg);
            let cs = load(ops, tg, a, 2);
            Ok((cs, ip))
        }
        _ => Err(unsupported_insn(insn)),
    }
}

pub(super) fn lower(
    insn: &Instruction,
    ops: &mut impl LiftBuffer,
    tg: &mut TempGen,
) -> Result<bool, LiftError> {
    use Mnemonic::*;
    let context = ops.legacy_context().expect("legacy lowering context");
    // SDM Vol 1 §21.1.2; Vol 3B §§23.1.2–23.1.3. This path implements
    // the I386 instruction generation; Application Real16 retains its 286 policy.
    let mode = if context.default_operand_bits == 16 {
        CpuMode::Real16
    } else {
        CpuMode::Compat32
    };
    // Prefixes select effective sizes; CodeSize is only the decoder default.
    let size = insn.code().op_code().operand_size() as u8 / 8;
    let size = if size == 0 {
        context.default_operand_bits / 8
    } else {
        size
    };
    let next = insn.next_ip() as u32;
    match insn.mnemonic() {
        Lgdt | Lidt | Sgdt | Sidt => {
            let addr = address(insn, ops, tg)?;
            emit(
                ops,
                L::Table {
                    addr,
                    idtr: matches!(insn.mnemonic(), Lidt | Sidt),
                    load: matches!(insn.mnemonic(), Lgdt | Lidt),
                    size,
                },
            );
        }
        Smsw => {
            let dst = tg.fresh();
            emit(
                ops,
                L::ReadControl {
                    dst,
                    register: ControlRegister::Cr0,
                },
            );
            let target = lower_write_target(insn, 0, ops, tg)?;
            // SDM Vol 2B §4.3 SMSW; 80386 PRM §17 p. 17-160:
            // memory is always a word. For r32, undefined bits 31:16 retain
            // their prior value as engine policy; the GPR write is still 32-bit.
            let value = if let WriteTarget::Reg { reg, size: 4 } = target {
                let old = read_reg(reg, ops, tg);
                let high = alu_none(ops, tg, |dst| IrOp::And {
                    dst,
                    a: old,
                    b: Val::Imm(0xffff_0000),
                    size: 4,
                    set_flags: FlagMask::NONE,
                });
                let low = alu_none(ops, tg, |out| IrOp::And {
                    dst: out,
                    a: Val::Temp(dst),
                    b: Val::Imm(0xffff),
                    size: 4,
                    set_flags: FlagMask::NONE,
                });
                alu_none(ops, tg, |dst| IrOp::Or {
                    dst,
                    a: high,
                    b: low,
                    size: 4,
                    set_flags: FlagMask::NONE,
                })
            } else {
                Val::Temp(dst)
            };
            emit_write(ops, tg, target, value);
        }

        Lmsw => {
            let value = lower_read(insn, 0, ops, tg)?;
            emit(ops, L::LoadMsw { value });
        }
        Clts => emit(ops, L::ClearTs),
        Mov if insn.op_register(0).is_cr() || insn.op_register(1).is_cr() => {
            let writing = insn.op_register(0).is_cr();
            let register = match insn.op_register(u32::from(!writing)) {
                Register::CR0 => ControlRegister::Cr0,
                Register::CR2 => ControlRegister::Cr2,
                Register::CR3 => ControlRegister::Cr3,
                // 80386 PRM §4.1.3 Fig. 4-2 p. 4-3 (scan checked):
                // CR1 is reserved; CR4 and later controls are not I386.
                _ => {
                    ops.push(IrOp::Trap {
                        vector: 6,
                        advance: 0,
                    });
                    return Ok(true);
                }
            };
            // SDM Vol 2B §4.3 MOV CR; 80386 PRM §17 p. 17-111:
            // always a 32-bit GPR transfer, including with a 66h prefix.
            if writing {
                let value = lower_read(insn, 1, ops, tg)?;
                emit(ops, L::WriteControl { register, value });
            } else {
                let dst = tg.fresh();
                emit(ops, L::ReadControl { dst, register });
                let target = lower_write_target(insn, 0, ops, tg)?;
                emit_write(ops, tg, target, Val::Temp(dst));
            }
        }
        // Debug/test-register execution remains outside the activated system seam.
        Mov if insn.op_register(0).is_dr()
            || insn.op_register(1).is_dr()
            || insn.op_register(0).is_tr()
            || insn.op_register(1).is_tr() =>
        {
            emit(
                ops,
                L::Refuse {
                    operation: crate::UnsupportedSystemOperation::LegacyInstruction,
                },
            );
        }
        Mov if insn.op_register(0).is_segment_register()
            || insn.op_register(1).is_segment_register() =>
        {
            if let Some(seg) = segment(insn.op0_register()) {
                if seg == SegmentId::Cs {
                    ops.push(IrOp::Trap {
                        vector: 6,
                        advance: 0,
                    });
                } else {
                    let value = lower_read(insn, 1, ops, tg)?;
                    emit(
                        ops,
                        L::LoadSegment {
                            segment: seg,
                            value,
                            inhibit: seg == SegmentId::Ss,
                        },
                    );
                }
            } else {
                let value = selector(ops, tg, segment(insn.op1_register()).unwrap());
                let dst = lower_write_target(insn, 0, ops, tg)?;
                emit_write(ops, tg, dst, value);
            }
        }
        Push => {
            let value = if let Some(seg) = segment(insn.op0_register()) {
                selector(ops, tg, seg)
            } else {
                lower_read(insn, 0, ops, tg)?
            };
            emit(ops, L::Push { value, size });
        }
        Pop => {
            let dst = tg.fresh();
            emit(ops, L::Pop { dst, size });
            if let Some(segment) = segment(insn.op0_register()) {
                emit(
                    ops,
                    L::LoadSegment {
                        segment,
                        value: Val::Temp(dst),
                        inhibit: segment == SegmentId::Ss,
                    },
                );
                emit(
                    ops,
                    L::AdjustStack {
                        bytes: u32::from(size),
                    },
                );
            } else {
                // Memory EA for POP uses post-increment ESP (SDM Vol 2B §4.3 POP).
                // Defer this uncommon form rather than commit ESP before a fault.
                if insn.op_kind(0) != OpKind::Register {
                    return Err(unsupported_insn(insn));
                }
                let target = lower_write_target(insn, 0, ops, tg)?;
                emit(
                    ops,
                    L::AdjustStack {
                        bytes: u32::from(size),
                    },
                );
                emit_write(ops, tg, target, Val::Temp(dst));
            }
        }
        Pushf | Pushfd => emit(ops, L::Flags { size, pop: false }),
        Popf | Popfd => emit(ops, L::Flags { size, pop: true }),
        Pusha | Pushad => emit(ops, L::All { size, pop: false }),
        Popa | Popad => emit(ops, L::All { size, pop: true }),
        Les | Lds | Lss | Lfs | Lgs => {
            let segment = match insn.mnemonic() {
                Les => SegmentId::Es,
                Lds => SegmentId::Ds,
                Lss => SegmentId::Ss,
                Lfs => SegmentId::Fs,
                _ => SegmentId::Gs,
            };
            let a = address(insn, ops, tg)?;
            let value = load(ops, tg, a, size);
            let b = add_addr(a, Val::Imm(u64::from(size)), ops, tg);
            let sel = load(ops, tg, b, 2);
            emit(
                ops,
                L::LoadSegment {
                    segment,
                    value: sel,
                    inhibit: false,
                },
            );
            emit_write(
                ops,
                tg,
                WriteTarget::Reg {
                    reg: iced_to_reg(insn.op0_register()).unwrap(),
                    size,
                },
                value,
            );
        }
        Jmp | Call => {
            let far = is_far_jmp(insn)
                || is_far_call(insn)
                || matches!(insn.code(), Code::Jmp_m1632 | Code::Call_m1632);
            let (cs, ip) = if far {
                let (cs, ip) = far_target(insn, ops, tg, size)?;
                (Some(cs), ip)
            } else {
                (
                    None,
                    match insn.op_kind(0) {
                        OpKind::NearBranch16 | OpKind::NearBranch32 => {
                            Val::Imm(insn.near_branch_target())
                        }
                        _ => lower_read(insn, 0, ops, tg)?,
                    },
                )
            };
            emit(
                ops,
                L::Transfer {
                    cs,
                    ip,
                    size,
                    return_ip: (insn.mnemonic() == Call).then_some(
                        (insn.next_ip() & if size == 2 { 0xffff } else { 0xffff_ffff }) as u32,
                    ),
                },
            );
        }
        Ret | Retf => emit(
            ops,
            L::Return {
                size,
                far: insn.mnemonic() == Retf,
                pop_extra: if insn.op_count() == 0 {
                    0
                } else {
                    insn.immediate16()
                },
            },
        ),
        // SDM Vol 2B §4.3 NOP: 0F 1F multi-byte NOP requires a
        // later processor family; only the 90h NOP forms belong to I386.
        Nop if !matches!(insn.code(), Code::Nopw | Code::Nopd) => {
            ops.push(IrOp::Trap {
                vector: 6,
                advance: 0,
            });
        }
        Cpuid => ops.push(IrOp::Trap {
            vector: 6,
            advance: 0,
        }),
        Iret | Iretd => {
            emit(ops, L::CheckTaskReturn);
            emit(
                ops,
                L::Refuse {
                    operation: crate::UnsupportedSystemOperation::LegacyEventDelivery,
                },
            );
        }
        Int | Int3 | Into => emit(
            ops,
            L::Interrupt {
                vector: match insn.mnemonic() {
                    Int => insn.immediate8(),
                    Int3 => 3,
                    _ => 4,
                },
                saved_ip: next,
                overflow_only: insn.mnemonic() == Into,
            },
        ),
        Loop | Loope | Loopne | Jcxz | Jecxz => {
            let count_bits = match insn.code() {
                Code::Loop_rel8_16_CX
                | Code::Loope_rel8_16_CX
                | Code::Loopne_rel8_16_CX
                | Code::Jcxz_rel8_16
                | Code::Loop_rel8_32_CX
                | Code::Loope_rel8_32_CX
                | Code::Loopne_rel8_32_CX
                | Code::Jcxz_rel8_32 => 16,
                _ => 32,
            };
            let kind = match insn.mnemonic() {
                Loop => LoopKind::Loop,
                Loope => LoopKind::Loope,
                Loopne => LoopKind::Loopne,
                _ => LoopKind::Jcxz,
            };
            emit(
                ops,
                L::Loop {
                    kind,
                    count_bits,
                    taken: insn.near_branch_target() as u32,
                    next,
                },
            );
        }
        Movsb | Movsw | Movsd | Stosb | Stosw | Stosd | Lodsb | Lodsw | Lodsd | Cmpsb | Cmpsw
        | Cmpsd | Scasb | Scasw | Scasd
            if insn.op_kinds().any(|k| {
                matches!(
                    k,
                    OpKind::MemorySegSI
                        | OpKind::MemorySegESI
                        | OpKind::MemoryESDI
                        | OpKind::MemoryESEDI
                )
            }) =>
        {
            let kind = match insn.mnemonic() {
                Movsb | Movsw | Movsd => StrOp::Movs,
                Stosb | Stosw | Stosd => StrOp::Stos,
                Lodsb | Lodsw | Lodsd => StrOp::Lods,
                Cmpsb | Cmpsw | Cmpsd => StrOp::Cmps,
                _ => StrOp::Scas,
            };
            let elem = match insn.mnemonic() {
                Movsb | Stosb | Lodsb | Cmpsb | Scasb => 1,
                Movsw | Stosw | Lodsw | Cmpsw | Scasw => 2,
                _ => 4,
            };
            let address_bits = if insn
                .op_kinds()
                .any(|k| matches!(k, OpKind::MemorySegSI | OpKind::MemoryESDI))
            {
                16
            } else {
                32
            };
            let rep = if insn.has_repne_prefix() {
                RepKind::Repne
            } else if insn.has_rep_prefix() {
                if matches!(kind, StrOp::Cmps | StrOp::Scas) {
                    RepKind::Repe
                } else {
                    RepKind::Rep
                }
            } else {
                RepKind::None
            };
            emit(
                ops,
                L::String {
                    kind,
                    elem,
                    rep,
                    address_bits,
                    source: segment(insn.segment_prefix()).unwrap_or(SegmentId::Ds),
                },
            );
        }
        Xlatb => {
            let bx = read_reg(Reg::Rbx, ops, tg);
            let ax = read_reg(Reg::Rax, ops, tg);
            let al = alu_none(ops, tg, |dst| IrOp::And {
                dst,
                a: ax,
                b: Val::Imm(255),
                size: 8,
                set_flags: FlagMask::NONE,
            });
            let offset = add_addr(bx, al, ops, tg);
            let address_bits = if matches!(insn.memory_base(), Register::EBX) {
                32
            } else {
                16
            };
            let offset = alu_none(ops, tg, |dst| IrOp::And {
                dst,
                a: offset,
                b: Val::Imm(if address_bits == 16 {
                    0xffff
                } else {
                    0xffff_ffff
                }),
                size: 8,
                set_flags: FlagMask::NONE,
            });
            let dst = tg.fresh();
            emit(
                ops,
                L::Address {
                    dst,
                    segment: segment(insn.memory_segment()).unwrap_or(SegmentId::Ds),
                    offset,
                    bytes: 1,
                },
            );
            let value = load(ops, tg, Val::Temp(dst), 1);
            emit_write(
                ops,
                tg,
                WriteTarget::Reg {
                    reg: Reg::Rax,
                    size: 1,
                },
                value,
            );
        }
        Leave => emit(ops, L::Leave { size }),
        Cli => ops.push(IrOp::SetIf { value: false }),
        Sti => ops.push(IrOp::SetIf { value: true }),
        // Shared scalar lowering has no legacy control/stack assumptions here.
        // SDM Vol 3B §23.1.3 and 80386 PRM §14.2, pp. 14-2–14-3.
        Mov | Lea | Add | Adc | Sub | Sbb | And | Or | Xor | Cmp | Test | Inc | Dec | Neg | Not
        | Xchg | Shl | Sal | Shr | Sar | Rol | Ror | Rcl | Rcr | Shld | Shrd | Mul | Imul | Div
        | Idiv | Movzx | Movsx | Cbw | Cwde | Cwd | Cdq | Clc | Stc | Cmc | Cld | Std | Lahf
        | Sahf | Nop | Hlt | In | Out | Jb | Jbe | Je | Jg | Jge | Jl | Jle | Jae | Ja | Jne
        | Jno | Jnp | Jns | Jo | Jp | Js | Setb | Setbe | Sete | Setg | Setge | Setl | Setle
        | Setae | Seta | Setne | Setno | Setnp | Setns | Seto | Setp | Sets | Bsf | Bsr => {
            let first = ops.len();
            let result = lift_insn_inner(insn, ops, tg, mode, &mut mmx::Sequence::default());
            for op in &mut ops[first..] {
                if let IrOp::Branch {
                    taken, fallthrough, ..
                } = op
                {
                    *taken = insn.near_branch_target();
                    *fallthrough = u64::from(next);
                }
            }
            return result;
        }
        // Other system facilities are activated by later phases.
        _ => emit(
            ops,
            L::Refuse {
                operation: crate::UnsupportedSystemOperation::LegacyInstruction,
            },
        ),
    }
    Ok(true)
}
