//! Explicit-context legacy lowering (decision-18 §1; spec.md §§7, 17).
use super::*;
use crate::legacy::{
    ControlRegister, DescriptorQuery, LegacyOp as L, MachineContext, SegmentId,
    SelectorDestination, SystemRegister,
};

struct Ops {
    values: Vec<IrOp>,
    context: MachineContext,
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
    fn legacy_context(&self) -> Option<MachineContext> {
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

/// Check opcode availability even when the instruction is not fully fetched.
pub(crate) fn unavailable_i286_opcode(code: &[u8]) -> bool {
    let mut opcode = code
        .iter()
        .skip_while(|b| matches!(b, 0x26 | 0x2e | 0x36 | 0x3e | 0xf0 | 0xf2 | 0xf3));
    // I286 PRM Appendix B #UD p. B-12: invalid first or second opcode bytes
    // suffice for #UD. Appendix B encodings (CLTS p. B-31, LAR p. B-60,
    // LGDT/LIDT p. B-65, LSL p. B-71, LTR p. B-72) use only 0F 00–03/06.
    match opcode.next() {
        Some(0x64..=0x67) => true,
        Some(0x0f) => match opcode.next() {
            // Appendix B SLDT/STR/LLDT/LTR/VERR/VERW pp. B-66, B-72,
            // B-102, B-108, B-111: 0F 00 has only extensions /0–/5.
            Some(0x00) => opcode.next().is_some_and(|b| b & 0x38 >= 0x30),
            // Appendix B LGDT/LIDT, LMSW, SGDT/SIDT, SMSW pp. B-65,
            // B-67, B-101, B-103: /5 and /7 are absent; /0–/3 need memory.
            Some(0x01) => opcode.next().is_some_and(|b| {
                matches!(b & 0x38, 0x28 | 0x38) || (b & 0xc0 == 0xc0 && b & 0x38 <= 0x18)
            }),
            Some(0x02 | 0x03 | 0x06) | None => false,
            Some(_) => true,
        },
        _ => false,
    }
}

/// Decode exactly the fetched instruction; callers fetch incrementally and check CS.
pub(crate) fn instruction(
    code: &[u8],
    ip: u32,
    context: MachineContext,
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
    // I286 PRM §6.2 p. 6-2; Appendix B #UD p. B-12 and MOV p. B-73;
    // §9.6.1 p. 9-10: unavailable opcodes/operands fault at their prefixes.
    // Iced's generation tags classify encodings, not individual Sreg operands.
    let unavailable_286 = context.generation == crate::CpuGeneration::I286
        && (unavailable_i286_opcode(code)
            || (0..insn.op_count())
                .any(|i| matches!(insn.op_register(i), Register::FS | Register::GS))
            || !insn.cpuid_features().iter().all(|feature| {
                matches!(
                    feature,
                    iced_x86::CpuidFeature::INTEL8086
                        | iced_x86::CpuidFeature::INTEL8086_ONLY
                        | iced_x86::CpuidFeature::INTEL186
                        | iced_x86::CpuidFeature::INTEL286
                        | iced_x86::CpuidFeature::INTEL286_ONLY
                        // I286/287 PRM Part II Ch. 2, Processor Control
                        // Instructions, p. 2-16: available coprocessor forms.
                        | iced_x86::CpuidFeature::FPU
                        | iced_x86::CpuidFeature::FPU287
                )
            }));
    let esc = crate::x87::is_esc(code);
    let wait = insn.mnemonic() == Mnemonic::Wait;
    if esc || wait {
        let mut bytes = [0; 15];
        bytes[..code.len()].copy_from_slice(code);
        emit(
            &mut ops,
            L::Coprocessor {
                bytes,
                len: code.len() as u8,
                wait,
            },
        );
    }
    // SDM Vol 1 §§2.1.1–2.1.3; 80386 PRM §17: sharing lowering
    // does not grant instructions introduced after the selected generation.
    // 80386 PRM §17 MOV to/from Special Registers, p. 17-111:
    // TR6/TR7 are I386. Iced also tags the shared encoding with 486-only
    // test registers; that alternative must not hide the I386 operands.
    let i386_test_register = matches!(insn.code(), Code::Mov_r32_tr | Code::Mov_tr_r32)
        && (0..insn.op_count())
            .any(|i| matches!(insn.op_register(i), Register::TR6 | Register::TR7));
    let unavailable_386 = context.generation == crate::CpuGeneration::I386
        && !i386_test_register
        && !insn.cpuid_features().iter().all(|feature| {
            matches!(
                feature,
                iced_x86::CpuidFeature::INTEL8086
                    | iced_x86::CpuidFeature::INTEL8086_ONLY
                    | iced_x86::CpuidFeature::INTEL186
                    | iced_x86::CpuidFeature::INTEL286
                    | iced_x86::CpuidFeature::INTEL286_ONLY
                    | iced_x86::CpuidFeature::INTEL386
                    | iced_x86::CpuidFeature::INTEL386_ONLY
                    | iced_x86::CpuidFeature::INTEL386_A0_ONLY
                    | iced_x86::CpuidFeature::FPU
                    | iced_x86::CpuidFeature::FPU287
                    | iced_x86::CpuidFeature::FPU387
            )
        });
    if insn.is_invalid() || table_register_operand || unavailable_286 || unavailable_386 {
        // SDM Vol 3A §7.15, Events 6/13: #UD or an overlong instruction.
        // I286 PRM Appendix B #GP p. B-9: incomplete at ten bytes is overlong.
        let overlong_286 = context.generation == crate::CpuGeneration::I286
            && code.len() == 10
            && decoder.last_error() == iced_x86::DecoderError::NoMoreBytes
            && !unavailable_286;
        let vector = if overlong_286 || (code.len() == 15 && exceeds_length_limit(code, mode)) {
            13
        } else {
            6
        };
        ops.push(IrOp::Trap { vector, advance: 0 });
    } else {
        let first = ops.values.len();
        // Decision-25: empty I286 outputs and WAIT complete even where
        // executing a present protected-mode coprocessor remains deferred.
        let lowered = if context.generation == crate::CpuGeneration::I286
            && context.regime == crate::MachineRegime::Protected
            && (wait || (esc && crate::x87::esc_output(code)))
        {
            Err(unsupported_insn(&insn))
        } else {
            lower(&insn, &mut ops, &mut tg)
        };
        if let Err(error) = lowered {
            if !esc && !wait {
                return Err(refill_unsupported_bytes(error, code, u64::from(ip)));
            }
            ops.values.truncate(first);
            let mut bytes = [0; 15];
            bytes[..code.len()].copy_from_slice(code);
            ops.push(IrOp::Unsupported {
                bytes,
                len: code.len() as u8,
            });
            if matches!(insn.mnemonic(), Mnemonic::Fnstcw | Mnemonic::Fnstsw) {
                // The absent word stores share real-mode transfer handling;
                // Unsupported preserves Present's protected-mode refusal.
                control::lift_x87(&insn, &mut ops, &mut tg)?;
            } else if insn.op_kinds().any(|k| k == OpKind::Memory) {
                effective_address(&insn, &mut ops, &mut tg)?;
            }
        }
        // Decoder access intent includes RMW operands, which require a writable
        // segment type (SDM Vol 3A §6.4).
        let mut factory = iced_x86::InstructionInfoFactory::new();
        let info = factory.info(&insn);
        let writing = (0..insn.op_count()).any(|i| {
            insn.op_kind(i) == OpKind::Memory
                && matches!(
                    info.op_access(i),
                    iced_x86::OpAccess::Write
                        | iced_x86::OpAccess::CondWrite
                        | iced_x86::OpAccess::ReadWrite
                        | iced_x86::OpAccess::ReadCondWrite
                )
        });
        for op in &mut ops.values {
            if let IrOp::Legacy(L::Address { access, .. }) = op {
                *access = if writing {
                    crate::AccessKind::Write
                } else {
                    crate::AccessKind::Read
                };
            }
        }
        mark_memory_accesses(&insn, mode, &mut ops.values);
        fill_ir_bytes(&insn, code, u64::from(ip), &mut ops.values);
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
    // Generation availability is checked before lowering; decoder width alone
    // never grants the 286 any 386 forms (I286 PRM §6.2 p. 6-2, Appendix B).
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
    // 286 PRM App. B pp. B-21/60/66/71/72/102/108/111;
    // 386 PRM §17 pp. 17-25/89/98/106/108/159/166/169 (scans checked):
    // these instructions are protected-only, before any operand access.
    if context.regime == crate::MachineRegime::Real
        && matches!(
            insn.mnemonic(),
            Lldt | Sldt | Ltr | Str | Lar | Lsl | Verr | Verw | Arpl
        )
    {
        ops.push(IrOp::Trap {
            vector: 6,
            advance: 0,
        });
        return Ok(true);
    }
    match insn.mnemonic() {
        Lldt | Ltr => {
            let value = lower_read(insn, 0, ops, tg)?;
            // SDM Vol 2A §3.3 LLDT/LTR; 286 PRM App. B pp. B-66/B-72;
            // 386 PRM §17 pp. 17-98/17-108: GDT LDT or available TSS.
            let (register, allowed_types) = if insn.mnemonic() == Lldt {
                (SystemRegister::Ldtr, 1 << 2)
            } else {
                (
                    SystemRegister::Tr,
                    if context.generation == crate::CpuGeneration::I286 {
                        1 << 1
                    } else {
                        (1 << 1) | (1 << 9)
                    },
                )
            };
            emit(
                ops,
                L::LoadSystemSegment {
                    register,
                    value,
                    allowed_types,
                },
            );
        }
        Sldt | Str => {
            let dst = tg.fresh();
            emit(
                ops,
                L::ReadSystemSegment {
                    dst,
                    register: if insn.mnemonic() == Sldt {
                        SystemRegister::Ldtr
                    } else {
                        SystemRegister::Tr
                    },
                },
            );
            let mut target = lower_write_target(insn, 0, ops, tg)?;
            // SDM Vol 2B §4.3 SLDT p. 4-656: Intel386 upper bits are
            // undefined; preserve them as deterministic engine policy.
            // STR p. 4-679 requires clearing bits 31:16 for a 32-bit register.
            // PROVENANCE.md §1: this takes precedence over 386 PRM §17
            // p. 17-166, which says the operand-size attribute has no effect.
            if insn.mnemonic() == Sldt {
                if let WriteTarget::Reg { size, .. } = &mut target {
                    *size = 2;
                }
            }
            emit_write(ops, tg, target, Val::Temp(dst));
        }
        Lar | Lsl | Verr | Verw => {
            let selector = lower_read(
                insn,
                u32::from(matches!(insn.mnemonic(), Lar | Lsl)),
                ops,
                tg,
            )?;
            let query = match insn.mnemonic() {
                Lar => DescriptorQuery::Rights {
                    dst: iced_to_reg(insn.op0_register()).or_unsupported(insn)?,
                    size,
                },
                Lsl => DescriptorQuery::Limit {
                    dst: iced_to_reg(insn.op0_register()).or_unsupported(insn)?,
                    size,
                },
                Verr => DescriptorQuery::Read,
                _ => DescriptorQuery::Write,
            };
            // SDM Vol 2A §3.3 LAR Table 3-55, LSL Table 3-59 are primary.
            // 286 PRM §11.3 p. 11-3/App. B p. B-71: 286 system types
            // have no 386 extensions, and LSL requires nonconforming segments.
            let i286 = context.generation == crate::CpuGeneration::I286;
            let allowed_system_types = match insn.mnemonic() {
                Lar => {
                    if i286 {
                        0x003e
                    } else {
                        0x1a3e
                    }
                }
                Lsl => {
                    if i286 {
                        0x000e
                    } else {
                        0x0a0e
                    }
                }
                _ => 0,
            };
            emit(
                ops,
                L::QueryDescriptor {
                    selector,
                    query,
                    allowed_system_types,
                    allow_conforming: !(i286 && insn.mnemonic() == Lsl),
                },
            );
        }
        Arpl => {
            let target = lower_write_target(insn, 0, ops, tg)?;
            let (selector, destination) = match target {
                WriteTarget::Reg { reg, .. } => {
                    (read_reg(reg, ops, tg), SelectorDestination::Register(reg))
                }
                WriteTarget::Mem { addr, .. } => {
                    (load(ops, tg, addr, 2), SelectorDestination::Memory(addr))
                }
                _ => return Err(unsupported_insn(insn)),
            };
            let rpl = lower_read(insn, 1, ops, tg)?;
            emit(
                ops,
                L::AdjustRpl {
                    selector,
                    rpl,
                    destination,
                },
            );
        }
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
                emit(
                    ops,
                    L::AdjustStack {
                        bytes: u32::from(size),
                    },
                );
                // SDM Vol 2B §4.3 POP: a memory EA uses the incremented ESP.
                // The legacy interpreter restores SP if the destination faults.
                let target = lower_write_target(insn, 0, ops, tg)?;
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
        // I386 PRM Appendix A two-byte opcode map p. A-7, §14.6
        // Table 14-1 p. 14-5: 0F FF is undefined and raises vector 6.
        // Iced recognizes this encoding as UD0.
        Cpuid | Ud0 | Ud2 => ops.push(IrOp::Trap {
            vector: 6,
            advance: 0,
        }),
        Iret | Iretd => {
            emit(ops, L::CheckTaskReturn);
            emit(ops, L::Iret { size });
        }
        Bound => {
            // I286 PRM Appendix B BOUND p. B-22; I386 PRM §17 BOUND
            // p. 17-26: signed, inclusive bounds; #BR restarts BOUND.
            let index = lower_read(insn, 0, ops, tg)?;
            let a = address(insn, ops, tg)?;
            let lower = load(ops, tg, a, size);
            let b = add_addr(a, Val::Imm(u64::from(size)), ops, tg);
            let upper = load(ops, tg, b, size);
            emit(
                ops,
                L::Bound {
                    index,
                    lower,
                    upper,
                    size,
                },
            );
        }
        Int | Int3 | Int1 | Into => emit(
            ops,
            L::Interrupt {
                vector: match insn.mnemonic() {
                    Int => insn.immediate8(),
                    Int3 => 3,
                    Int1 => 1,
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
        | Cmpsd | Scasb | Scasw | Scasd | Insb | Insw | Insd | Outsb | Outsw | Outsd
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
                Scasb | Scasw | Scasd => StrOp::Scas,
                Insb | Insw | Insd => StrOp::Ins,
                _ => StrOp::Outs,
            };
            let elem = match insn.mnemonic() {
                Movsb | Stosb | Lodsb | Cmpsb | Scasb | Insb | Outsb => 1,
                Movsw | Stosw | Lodsw | Cmpsw | Scasw | Insw | Outsw => 2,
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
                if matches!(kind, StrOp::Cmps | StrOp::Scas) {
                    RepKind::Repne
                } else {
                    RepKind::Rep
                }
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
                    access: crate::AccessKind::Read,
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
        Bt | Bts | Btr | Btc
            if insn.op_kind(0) == OpKind::Memory && insn.op_kind(1) == OpKind::Register =>
        {
            // SDM Vol 2A §3.3 BT/BTC/BTR/BTS: a register bit index
            // contributes a signed byte displacement before segment checks.
            let bit = lower_read(insn, 1, ops, tg)?;
            let signed = tg.fresh();
            ops.push(IrOp::Sext {
                dst: signed,
                a: bit,
                from: operand_size(insn, 1),
            });
            let displacement = alu_none(ops, tg, |dst| IrOp::Sar {
                dst,
                a: Val::Temp(signed),
                b: Val::Imm(3),
                size: 8,
                set_flags: FlagMask::NONE,
            });
            let base = effective_address_no_segment(insn, ops, tg)?;
            let offset = add_addr(base, displacement, ops, tg);
            let bits = if insn.memory_base().size() == 2
                || insn.memory_index().size() == 2
                || (insn.memory_base() == Register::None
                    && insn.memory_index() == Register::None
                    && insn.memory_displ_size() == 2)
            {
                16
            } else {
                32
            };
            let offset = alu_none(ops, tg, |dst| IrOp::And {
                dst,
                a: offset,
                b: Val::Imm(if bits == 16 { 0xffff } else { 0xffff_ffff }),
                size: 8,
                set_flags: FlagMask::NONE,
            });
            let dst = tg.fresh();
            emit(
                ops,
                L::Address {
                    dst,
                    segment: segment(insn.memory_segment()).unwrap(),
                    offset,
                    bytes: 1,
                    access: if insn.mnemonic() == Bt {
                        crate::AccessKind::Read
                    } else {
                        crate::AccessKind::Write
                    },
                },
            );
            emit_mem_bt(
                ops,
                tg,
                Val::Temp(dst),
                1,
                bit,
                match insn.mnemonic() {
                    Bt => BtOp::Test,
                    Bts => BtOp::Set,
                    Btr => BtOp::Reset,
                    _ => BtOp::Complement,
                },
                insn.has_lock_prefix(),
            );
        }
        Enter => emit(
            ops,
            L::Enter {
                size,
                alloc: insn.immediate16(),
                level: insn.immediate8_2nd() & 31,
            },
        ),
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
        _ if context.regime == crate::legacy::MachineRegime::Real => {
            // Share application instruction semantics; effective_address still
            // emits Legacy::Address and every bus access remains legacy-owned.
            let mut application = *insn;
            if insn.cpuid_features().iter().any(|feature| {
                matches!(
                    feature,
                    iced_x86::CpuidFeature::FPU
                        | iced_x86::CpuidFeature::FPU287
                        | iced_x86::CpuidFeature::FPU387
                )
            }) {
                // SDM Vol 1 §8.1.10: PE selects the real environment image,
                // independently of retained CS.D. Decoded operand/address
                // widths remain in Code and the operand registers; the shared
                // Real16 helper uses CodeSize only to choose this personality.
                application.set_code_size(CodeSize::Code16);
            }
            return lift_insn_inner(
                &application,
                ops,
                tg,
                CpuMode::Real16,
                &mut mmx::Sequence::default(),
            );
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
