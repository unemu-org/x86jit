//! Explicit-context legacy lowering (decision-18 §1; spec.md §§7, 17).
use super::*;
use crate::legacy::{
    ControlRegister, DebugTestRegister, DescriptorQuery, LegacyOp as L, MachineContext, SegmentId,
    SelectorDestination, SystemRegister,
};

struct Ops {
    values: Vec<IrOp>,
    context: MachineContext,
    profile: &'static crate::ModelProfile,
    /// The fields read live, by `LiveFieldKind`, and the temporary each
    /// defines at its first use.
    live: Option<Box<LiveRequest>>,
}
type LiveRequest = [Option<(LiveSpec, Option<Temp>)>; 2];
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
    fn model_profile(&self) -> Option<&'static crate::ModelProfile> {
        Some(self.profile)
    }
    fn live_field(&mut self, kind: LiveFieldKind, tg: &mut TempGen) -> Option<Val> {
        let (spec, temp) = self.live.as_mut()?[kind as usize].as_mut()?;
        // One read per instruction: every use sees the same value.
        let dst = *temp.get_or_insert_with(|| {
            let dst = tg.fresh();
            self.values.push(IrOp::Legacy(L::LiveField {
                dst,
                bus: spec.bus,
                bytes: spec.bytes,
                extend: spec.extend,
            }));
            dst
        });
        Some(Val::Temp(dst))
    }
}

/// Where a live field's bytes are and how the decoder extends them.
#[derive(Clone, Copy, Debug)]
struct LiveSpec {
    bus: u64,
    bytes: u8,
    extend: u8,
}

/// The immediate and displacement fields of an instruction whose lowering
/// can read them live (xj-to9.46): `(offset, bytes)` in the instruction,
/// indexed by `LiveFieldKind`, and its length.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LiveFields {
    pub len: u8,
    pub fields: [Option<(u8, u8)>; 2],
}

/// The live-capable fields of the instruction at the start of `code`. Only
/// the scalar forms whose immediate and displacement reach the IR through
/// the central operand lowering (`lower_read`, `effective_address_no_segment`)
/// qualify; `live_instruction` still proves that no other op depends on the
/// bytes. Branch displacements, ENTER/RET counts, ports, vectors and shift
/// counts stay baked.
#[cold]
#[inline(never)]
pub(crate) fn live_fields(code: &[u8], ip: u32, context: MachineContext) -> Option<LiveFields> {
    use Mnemonic::*;
    let mut decoder = Decoder::with_ip(
        u32::from(context.default_operand_bits),
        code,
        u64::from(ip),
        crate::model::decode::decoder_options_for_generation(context.generation),
    );
    let insn = decoder.decode();
    if insn.is_invalid()
        || !matches!(
            insn.mnemonic(),
            Mov | Lea | Add | Adc | Sub | Sbb | And | Or | Xor | Cmp | Test
        )
        || (0..insn.op_count()).any(|i| {
            let register = insn.op_register(i);
            register.is_segment_register()
                || register.is_cr()
                || register.is_dr()
                || register.is_tr()
        })
    {
        return None;
    }
    let offsets = decoder.get_constant_offsets(&insn);
    if offsets.has_immediate2() {
        return None;
    }
    let mut fields = [None; 2];
    if offsets.has_immediate() {
        fields[LiveFieldKind::Immediate as usize] = Some((
            offsets.immediate_offset() as u8,
            offsets.immediate_size() as u8,
        ));
    }
    if offsets.has_displacement() {
        fields[LiveFieldKind::Displacement as usize] = Some((
            offsets.displacement_offset() as u8,
            offsets.displacement_size() as u8,
        ));
    }
    fields.iter().any(Option::is_some).then_some(LiveFields {
        len: insn.len() as u8,
        fields,
    })
}

/// The decoder's value of the `kind` field of the instruction at the start
/// of `code`.
fn decoded_field(code: &[u8], ip: u32, context: MachineContext, kind: LiveFieldKind) -> u64 {
    let insn = Decoder::with_ip(
        u32::from(context.default_operand_bits),
        code,
        u64::from(ip),
        crate::model::decode::decoder_options_for_generation(context.generation),
    )
    .decode();
    match kind {
        LiveFieldKind::Immediate => (0..insn.op_count())
            .find(|&i| is_immediate(insn.op_kind(i)))
            .map_or(0, |i| insn.immediate(i)),
        LiveFieldKind::Displacement => insn.memory_displacement64(),
    }
}

/// The instruction at the start of `code` with each `requested` field
/// (`LiveFields::fields` index → the bus address of its first byte) read
/// when it executes, or `None` when that cannot be exact. Exactness is
/// checked, not assumed: the field's extension must reproduce the decoder's
/// value, and the lift must be identical for the field's current bytes, their
/// complement and zero, so no op other than `LiveField` depends on them.
#[cold]
#[inline(never)]
pub(crate) fn live_instruction(
    code: &[u8],
    ip: u32,
    context: MachineContext,
    profile: &'static crate::ModelProfile,
    fields: LiveFields,
    requested: [Option<u64>; 2],
) -> Option<IrBlock> {
    if code.len() != usize::from(fields.len) {
        return None;
    }
    let probe = |fill: fn(u8) -> u8| {
        let mut probe = code.to_vec();
        for (field, bus) in fields.fields.iter().zip(requested) {
            if let (Some((offset, bytes)), Some(_)) = (field, bus) {
                for byte in &mut probe[usize::from(*offset)..usize::from(offset + bytes)] {
                    *byte = fill(*byte);
                }
            }
        }
        probe
    };
    let probes = [code.to_vec(), probe(|b| !b), probe(|_| 0)];
    let mut live = [None; 2];
    for (kind, slot) in [LiveFieldKind::Immediate, LiveFieldKind::Displacement]
        .into_iter()
        .zip(&mut live)
    {
        let (Some((offset, bytes)), Some(bus)) =
            (fields.fields[kind as usize], requested[kind as usize])
        else {
            continue;
        };
        let range = usize::from(offset)..usize::from(offset + bytes);
        let extend = [0, 64, 32, 16].into_iter().find(|&extend| {
            probes.iter().all(|code| {
                let mut raw = [0; 8];
                raw[..range.len()].copy_from_slice(&code[range.clone()]);
                crate::legacy::live_field_value(u64::from_le_bytes(raw), bytes, extend)
                    == decoded_field(code, ip, context, kind)
            })
        })?;
        *slot = Some((LiveSpec { bus, bytes, extend }, None));
    }
    let mut lifts = probes
        .iter()
        .map(|code| instruction_with(code, ip, context, profile, Some(Box::new(live))));
    let first = lifts.next()?.ok()?;
    let image = format!("{:?}", first.ops);
    if !first
        .ops
        .iter()
        .any(|op| matches!(op, IrOp::Legacy(L::LiveField { .. })))
        || first.ops.iter().any(|op| {
            matches!(
                op,
                IrOp::Trap { .. } | IrOp::Unsupported { .. } | IrOp::Legacy(L::Refuse { .. })
            )
        })
    {
        return None;
    }
    for lift in lifts {
        let lift = lift.ok()?;
        if lift.guest_len != first.guest_len
            || lift.temp_count != first.temp_count
            || format!("{:?}", lift.ops) != image
        {
            return None;
        }
    }
    Some(first)
}

// The live-field exactness witness: a lowering that bakes its immediate
// somewhere `live_field` does not reach (1: its value, 2: whether it is 0).
#[cfg(test)]
thread_local! {
    static BAKE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn bake_for_test(insn: &Instruction, ops: &mut Vec<IrOp>, tg: &mut TempGen) {
    let Some(value) = (0..insn.op_count())
        .find(|&i| is_immediate(insn.op_kind(i)))
        .map(|i| insn.immediate(i))
    else {
        return;
    };
    let baked = match BAKE.with(std::cell::Cell::get) {
        1 => value,
        2 => u64::from(value == 0),
        _ => return,
    };
    ops.push(IrOp::Add {
        dst: tg.fresh(),
        a: Val::Imm(baked),
        b: Val::Imm(0),
        size: 4,
        set_flags: FlagMask::NONE,
    });
}

#[cfg(test)]
mod live_tests {
    use super::*;

    /// `live_instruction` reads a field live only if no op other than its
    /// `LiveField` depends on the field's bytes: a lowering that bakes the
    /// value, or a predicate on it that the probes (the current bytes,
    /// their complement and zero) distinguish, stays baked. A predicate the
    /// probes cannot tell apart (say, `== 1` with the value 5) would not be
    /// caught; only the forms `live_fields` admits, whose lowering reaches
    /// the field through `lower_read` and `effective_address_no_segment`
    /// alone, are ever probed.
    #[test]
    fn a_lowering_that_bakes_the_field_stays_baked() {
        let context =
            crate::legacy::derive_legacy_context(crate::CpuModel::I80486Dx.legacy_model(), &{
                let mut cpu = crate::CpuState::default();
                cpu.legacy.cr0 = 1;
                for (i, cache) in cpu.legacy.segments.iter_mut().enumerate() {
                    cache.limit = u32::MAX;
                    cache.attributes = if i == 0 { 0x409b } else { 0x4093 };
                }
                cpu
            })
            .unwrap();
        let profile = crate::model::profile_for_generation(context.generation);
        // MOV EAX, 5.
        let code = b"\xb8\x05\x00\x00\x00";
        let fields = live_fields(code, 0x1000, context).unwrap();
        assert_eq!(
            fields.fields[LiveFieldKind::Immediate as usize],
            Some((1, 4))
        );
        let requested = [Some(0x1001), None];
        for (bake, live) in [(0, true), (1, false), (2, false)] {
            BAKE.with(|b| b.set(bake));
            let ir = live_instruction(code, 0x1000, context, profile, fields, requested);
            BAKE.with(|b| b.set(0));
            assert_eq!(ir.is_some(), live, "bake {bake}");
        }
    }
}

pub(crate) fn alias_context() -> MachineContext {
    crate::legacy::REAL16_ALIAS.context
}

pub(crate) fn shared_family(insn: &Instruction) -> bool {
    use Mnemonic::*;
    matches!(
        insn.mnemonic(),
        Push | Pop
            | Pushf
            | Pushfd
            | Popf
            | Popfd
            | Pusha
            | Pushad
            | Popa
            | Popad
            | Call
            | Ret
            | Retf
            | Jmp
            | Iret
            | Iretd
            | Enter
            | Leave
            | Loop
            | Loope
            | Loopne
            | Jcxz
            | Jecxz
            | Les
            | Lds
            | Lss
            | Lfs
            | Lgs
            | Bound
            | Xlatb
    ) || (insn.mnemonic() == Mov
        && (insn.op0_register().is_segment_register() || insn.op1_register().is_segment_register()))
}

// Decision-23 §6 freezes the alias's accepted instruction set. Sharing the
// implementation does not grant forms it previously left to the embedder.
pub(super) fn alias_supported(insn: &Instruction) -> bool {
    use Mnemonic::*;
    !(matches!(
        insn.mnemonic(),
        Pushfd | Popfd | Pushad | Popad | Iretd | Jecxz | Lss | Lfs | Lgs
    ) || (insn.mnemonic() == Bound && operand_size(insn, 0) != 2)
        || (0..insn.op_count()).any(|i| matches!(insn.op_register(i), Register::FS | Register::GS))
        || insn.op0_kind() == OpKind::FarBranch32
        || matches!(insn.code(), Code::Call_m1632 | Code::Jmp_m1632))
}

pub(crate) fn shared_helper(op: &L) -> bool {
    matches!(
        op,
        L::LoadSegment { .. }
            | L::Flags { .. }
            | L::All { .. }
            | L::Transfer { .. }
            | L::Return { .. }
            | L::Iret { .. }
            | L::CheckTaskReturn
            | L::Enter { .. }
            | L::Leave { .. }
            | L::Bound { .. }
    )
}

pub(super) struct ContextOps<'a, B> {
    pub ops: &'a mut B,
    pub context: MachineContext,
    /// The selected catalogue entry where lowering must not use the
    /// generation projection (I8088/I8086; decision-23 §2).
    pub profile: Option<&'static crate::ModelProfile>,
}
impl<B: LiftBuffer> std::ops::Deref for ContextOps<'_, B> {
    type Target = [IrOp];
    fn deref(&self) -> &[IrOp] {
        self.ops
    }
}
impl<B: LiftBuffer> std::ops::DerefMut for ContextOps<'_, B> {
    fn deref_mut(&mut self) -> &mut [IrOp] {
        self.ops
    }
}
impl<B: LiftBuffer> LiftBuffer for ContextOps<'_, B> {
    fn push(&mut self, op: IrOp) {
        self.ops.push(op);
    }
    fn live_field(&mut self, kind: LiveFieldKind, tg: &mut TempGen) -> Option<Val> {
        self.ops.live_field(kind, tg)
    }
    fn legacy_context(&self) -> Option<MachineContext> {
        Some(self.context)
    }
    fn model_profile(&self) -> Option<&'static crate::ModelProfile> {
        self.profile
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
#[cfg(test)]
pub(crate) fn instruction(
    code: &[u8],
    ip: u32,
    context: MachineContext,
) -> Result<IrBlock, LiftError> {
    instruction_for_profile(
        code,
        ip,
        context,
        crate::model::profile_for_generation(context.generation),
    )
}

pub(crate) fn instruction_for_profile(
    code: &[u8],
    ip: u32,
    context: MachineContext,
    profile: &'static crate::ModelProfile,
) -> Result<IrBlock, LiftError> {
    instruction_with(code, ip, context, profile, None)
}

fn instruction_with(
    code: &[u8],
    ip: u32,
    context: MachineContext,
    profile: &'static crate::ModelProfile,
    live: Option<Box<LiveRequest>>,
) -> Result<IrBlock, LiftError> {
    let mut decoder = Decoder::with_ip(
        u32::from(context.default_operand_bits),
        code,
        u64::from(ip),
        // 80386 PRM §17 MOV to/from Special Registers, p. 17-111:
        // recognize the test-register MOV encodings.
        crate::model::decode::decoder_options_for_generation(context.generation),
    );
    let mut insn = decoder.decode();
    let mut locked_mov = false;
    if insn.is_invalid() && profile.quirks.lock_data_movement {
        locked_mov = match relock(&mut insn, code, ip, context, profile) {
            Err(error) if context.regime == crate::MachineRegime::Protected => {
                return Ok(locked_transfer(code, ip, context, error));
            }
            locked => locked?,
        };
    }
    let mode = if context.default_operand_bits == 16 {
        CpuMode::Real16
    } else {
        CpuMode::Compat32
    };
    let mut ops = Ops {
        values: Vec::new(),
        context,
        profile,
        live,
    };
    let mut tg = TempGen::new();
    ops.push(IrOp::InsnStart {
        guest_addr: u64::from(ip),
    });
    let action = profile.decode(code, &insn);
    if let crate::model::decode::DecodeAction::Alias(canonical) = action {
        insn.set_code(canonical);
    }
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
    if insn.is_invalid() || matches!(action, crate::model::decode::DecodeAction::Absent(_)) {
        let overlong_286 = profile.max_insn_len == Some(10)
            && code.len() == 10
            && decoder.last_error() == iced_x86::DecoderError::NoMoreBytes
            && !profile.isa.unavailable_prefix(code);
        if matches!(action, crate::model::decode::DecodeAction::Absent(None)) {
            return Err(refill_unsupported_bytes(
                unsupported_insn(&insn),
                code,
                u64::from(ip),
            ));
        }
        let vector = if overlong_286 || (code.len() == 15 && exceeds_length_limit(code, mode)) {
            13
        } else if let crate::model::decode::DecodeAction::Absent(Some(vector)) = action {
            vector
        } else {
            6
        };
        ops.push(IrOp::Trap { vector, advance: 0 });
    } else {
        let first = ops.values.len();
        // Decision-25: empty I286 outputs and WAIT complete even where
        // executing a present protected-mode coprocessor remains deferred.
        let lowered = if !profile.lift.protected_coprocessor_outputs
            && context.regime == crate::MachineRegime::Protected
            && (wait
                || (esc && crate::x87::esc_output(code))
                || matches!(
                    insn.mnemonic(),
                    Mnemonic::Fninit | Mnemonic::Fnclex | Mnemonic::Fnsetpm
                )) {
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
            if matches!(
                insn.mnemonic(),
                Mnemonic::Fnstcw | Mnemonic::Fnstsw | Mnemonic::Fnstenv | Mnemonic::Fnsave
            ) {
                // Absent control stores share real-mode transfer handling;
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
        let mut pop_stack = insn.mnemonic() == Mnemonic::Pop;
        // SDM Vol 2A §3.3 ARPL p. 3-75: its write is conditional;
        // AdjustRpl checks write permission only in the raising branch.
        let operand_access = if writing && insn.mnemonic() != Mnemonic::Arpl {
            crate::AccessKind::Write
        } else {
            crate::AccessKind::Read
        };
        for op in &mut ops.values {
            if let IrOp::Legacy(L::Address {
                access, segment, ..
            }) = op
            {
                // POP's implicit source is read-only even with a memory destination.
                if pop_stack {
                    pop_stack = false;
                    continue;
                }
                if *access == crate::AccessKind::Execute {
                    continue;
                }
                if *segment == SegmentId::Ss
                    && matches!(insn.mnemonic(), Mnemonic::Push | Mnemonic::Call)
                {
                    continue;
                }
                *access = operand_access;
            }
        }
        if esc {
            if let Some(vector) = profile.quirks.coprocessor_overrun_vector {
                coprocessor_overrun(&insn, &mut ops.values, operand_access, vector);
            }
        }
        if locked_mov {
            lock_stores(&mut ops.values, &mut tg);
        }
        #[cfg(test)]
        if ops.live.is_some() {
            bake_for_test(&insn, &mut ops.values, &mut tg);
        }
        mark_memory_accesses(&insn, mode, &mut ops.values);
        fill_ir_bytes(&insn, code, u64::from(ip), &mut ops.values);
        if context.regime != crate::MachineRegime::Real {
            privilege_guard(
                &insn,
                locked_mov,
                context.generation,
                context.regime,
                &mut ops.values,
            );
        }
    }
    if context.segment_class == crate::SegmentClass::Direct {
        direct_addresses(&mut ops.values, &mut tg, context, u64::from(ip));
        if !ops.values.iter().all(crate::interp::legacy::supported) {
            ops.values.truncate(1);
            ops.values.push(IrOp::Legacy(L::Refuse {
                operation: crate::UnsupportedSystemOperation::LegacyInstruction,
            }));
        }
    }
    if context.segment_class == crate::SegmentClass::Flat32 {
        flat32_addresses(&mut ops.values);
    }
    defer_store_flags(&mut ops.values, &mut tg);
    if context.segment_class != crate::SegmentClass::Direct && rollback_required(&ops.values) {
        ops.values.insert(1, IrOp::Legacy(L::Checkpoint));
    }
    Ok(IrBlock {
        mode,
        guest_start: u64::from(ip),
        guest_len: code.len() as u32,
        icount: 1,
        temp_count: tg.count(),
        ops: ops.values,
        legacy_context: Some(context),
        link_exit: false,
    })
}

// 80286 PRM App. C item 11 p. C-2: "LOCK should only be used with the XCHG,
// MOV, MOVS, INS, and OUTS instructions", so none of them raises #UD. A locked
// MOV takes the engine's atomic path (`lock_stores`). A locked MOVS, INS or
// OUTS holds the bus across its transfers, which the engine cannot do: a
// typed handoff (`UnknownInstruction`) before anything commits, as for the
// twenty-bit models' locked strings (xj-50w.12). Returns whether `insn` is
// now a locked MOV.
#[cold]
#[inline(never)]
fn relock(
    insn: &mut Instruction,
    code: &[u8],
    ip: u32,
    context: MachineContext,
    profile: &crate::ModelProfile,
) -> Result<bool, LiftError> {
    let Some(locked) = crate::model::decode::locked_data_movement(
        code,
        u32::from(context.default_operand_bits),
        u64::from(ip),
        crate::model::decode::decoder_options_for_generation(context.generation),
    ) else {
        return Ok(false);
    };
    *insn = locked;
    // A form the decode set does not define keeps its #UD.
    if matches!(
        profile.decode(code, insn),
        crate::model::decode::DecodeAction::Absent(_)
    ) {
        return Ok(false);
    }
    if insn.mnemonic() != Mnemonic::Mov {
        return Err(refill_unsupported_bytes(
            unsupported_insn(insn),
            code,
            u64::from(ip),
        ));
    }
    Ok(true)
}

// A locked MOV's store is one atomic exchange whose old value is discarded:
// the path `IrOp::AtomicRmw` gives every locked operand on this path,
// including the typed handoff for a device operand. A load is one read,
// which the memory layer already performs as one host access when the
// operand is naturally aligned; a register-only MOV transfers nothing.
#[cold]
#[inline(never)]
fn lock_stores(ops: &mut [IrOp], tg: &mut TempGen) {
    for op in ops {
        if let IrOp::Store {
            fault,
            addr,
            src,
            size,
            ..
        } = *op
        {
            *op = IrOp::AtomicRmw {
                fault,
                old: tg.fresh(),
                addr,
                src,
                size,
                op: crate::ir::RmwOp::Xchg,
            };
        }
    }
}

// 80286 PRM App. B #MP p. B-9: a coprocessor operand larger than one word
// whose second or later words lie outside the segment raises interrupt 9; one
// whose start lies outside, or a one-word operand, takes the CPU's limit fault
// (#GP), as do writes to a read-only segment. §5.2 p. 5-7 and App. C Table C-1
// p. C-1 state the real-mode case ("does not fit in a segment"; "exceeded
// offset 0FFFFH"). The check precedes the operand's CPU limit check; the
// interpreter delivers it as a fault of the ESC (ASSUMED, spec.md §17.10).
#[inline(never)]
fn coprocessor_overrun(
    insn: &Instruction,
    ops: &mut Vec<IrOp>,
    access: crate::AccessKind,
    vector: u8,
) {
    let bytes = insn.memory_size().size();
    if bytes <= 2 || !insn.op_kinds().any(|k| k == OpKind::Memory) {
        return;
    }
    let Some((at, offset)) = ops.iter().enumerate().find_map(|(at, op)| match op {
        IrOp::Legacy(L::Address {
            offset,
            bytes: size,
            ..
        })
        | IrOp::SegLimitCheck { offset, size, .. }
            if usize::from(*size) == bytes =>
        {
            Some((at, *offset))
        }
        _ => None,
    }) else {
        return;
    };
    let kind = ops.iter().find_map(|op| match op {
        IrOp::X87 { kind, .. } => Some(*kind),
        _ => None,
    });
    ops.insert(
        at,
        IrOp::Legacy(L::CoprocessorOverrun {
            segment: segment(insn.memory_segment()).expect("memory segment"),
            offset,
            bytes: bytes as u8,
            access,
            vector,
            kind,
        }),
    );
}

// The CPL and IOPL tests that protected mode makes (286 PRM App. B; 386 PRM
// §6.3.5.1 p. 6-17, §8.3.1 pp. 8-4–5; SDM Vol 3A §6.9 p. 6-23). The test
// runs first, ahead of any memory operand (decision O10: ASSUMED, as LOADALL
// does; the sources give no order), and only in protected or V86 contexts, so the
// Real and Direct lowering is unchanged. A #UD that the lowering raised (a
// reserved control register, or a LOCK prefix on a form the decode set does
// not lock) keeps its place ahead of it (ASSUMED).
#[cold]
#[inline(never)]
fn privilege_guard(
    insn: &Instruction,
    locked_mov: bool,
    generation: crate::CpuGeneration,
    regime: crate::MachineRegime,
    ops: &mut Vec<IrOp>,
) {
    use crate::legacy::PrivilegeCheck;
    use Mnemonic::*;
    let special = |i| {
        let register = insn.op_register(i);
        register.is_cr() || register.is_dr() || register.is_tr()
    };
    let kind = match insn.mnemonic() {
        Hlt | Lgdt | Lidt | Lmsw | Clts | Lldt | Ltr | Invd | Wbinvd | Invlpg => {
            PrivilegeCheck::Cpl0
        }
        Mov if special(0) || special(1) => PrivilegeCheck::Cpl0,
        // SDM Vol 3B §23.2.7 p. 23-10: six sensitive families. Valid
        // LOCK and RET gain no V86 IOPL test (SDM Vol 2A LOCK pp. 3-565–566).
        Pushf | Pushfd | Popf | Popfd | Int | Iret | Iretd
            if regime == crate::MachineRegime::Virtual8086 =>
        {
            PrivilegeCheck::Iopl
        }
        Cli | Sti => PrivilegeCheck::Iopl,
        // SDM Vol 2B REP Operation p. 4-564: with a zero count the string
        // instruction does not execute, so it makes no I/O test.
        Insb | Insw | Insd | Outsb | Outsw | Outsd
            if insn.has_rep_prefix() || insn.has_repne_prefix() =>
        {
            PrivilegeCheck::RepIoPort {
                size: insn.memory_size().size() as u8,
                count_bytes: if insn
                    .op_kinds()
                    .any(|k| matches!(k, OpKind::MemorySegSI | OpKind::MemoryESDI))
                {
                    2
                } else {
                    4
                },
            }
        }
        In | Out => {
            let (acc, port) = if insn.mnemonic() == In {
                (0, 1)
            } else {
                (1, 0)
            };
            PrivilegeCheck::IoPort {
                port: (insn.op_kind(port) == OpKind::Immediate8).then(|| insn.immediate8()),
                size: operand_size(insn, acc),
            }
        }
        Insb | Insw | Insd | Outsb | Outsw | Outsd => PrivilegeCheck::IoPort {
            port: None,
            size: insn.memory_size().size() as u8,
        },
        // 286 PRM App. B LOCK p. B-68 and §3.11.2 p. 3-29: LOCK in protected
        // mode is #GP(0) when CPL > IOPL. The 386 has no such rule (386 PRM
        // §17 LOCK pp. 17-100–101 lists #UD only).
        _ if generation == crate::CpuGeneration::I286 && (locked_mov || insn.has_lock_prefix()) => {
            PrivilegeCheck::Iopl
        }
        _ => return,
    };
    if ops.iter().any(|op| matches!(op, IrOp::Trap { .. })) {
        return;
    }
    ops.insert(1, IrOp::Legacy(L::Privileged { kind }));
}

// A locked MOVS, INS or OUTS on the 80286 in the protected regime (`relock`'s
// handoff): its LOCK test first (286 PRM App. B LOCK p. B-68), so CPL > IOPL
// is #GP(0); with CPL <= IOPL the same typed handoff `relock` gives
// elsewhere, before any effect.
#[cold]
#[inline(never)]
fn locked_transfer(code: &[u8], ip: u32, context: MachineContext, error: LiftError) -> IrBlock {
    debug_assert!(matches!(error, LiftError::Unsupported { .. }), "{error:?}");
    let mode = if context.default_operand_bits == 16 {
        CpuMode::Real16
    } else {
        CpuMode::Compat32
    };
    IrBlock {
        mode,
        guest_start: u64::from(ip),
        guest_len: code.len() as u32,
        icount: 1,
        temp_count: 0,
        ops: vec![
            IrOp::InsnStart {
                guest_addr: u64::from(ip),
            },
            IrOp::Legacy(L::Privileged {
                kind: crate::legacy::PrivilegeCheck::Iopl,
            }),
            IrOp::Legacy(L::LockedTransfer),
        ],
        legacy_context: Some(context),
        link_exit: false,
    }
}

#[cold]
#[inline(never)]
fn flat32_addresses(ops: &mut [IrOp]) {
    for op in ops {
        if let IrOp::Legacy(L::Address {
            dst,
            segment,
            offset,
            bytes,
            access,
        }) = *op
        {
            if !matches!(segment, SegmentId::Fs | SegmentId::Gs)
                && !(segment == SegmentId::Cs && access == crate::AccessKind::Write)
            {
                *op = IrOp::Legacy(L::Flat32Address {
                    dst,
                    segment,
                    offset,
                    bytes,
                    access,
                });
            }
        }
    }
}

// spec.md §16: only a fault after a CPU-state commit needs rollback. Helper
// transactions retain their own memory evidence and task partial-commit rules.
fn rollback_required(ops: &[IrOp]) -> bool {
    let mut committed = false;
    for op in ops {
        let faultable = matches!(
            op,
            IrOp::Load { .. }
                | IrOp::Store { .. }
                | IrOp::AtomicRmw { .. }
                | IrOp::AtomicCas { .. }
                | IrOp::Div { .. }
                | IrOp::Trap { .. }
                | IrOp::BoundGate { .. }
                | IrOp::SegLimitCheck { .. }
                | IrOp::X87 { .. }
                | IrOp::Jump { .. }
                | IrOp::Branch { .. }
                | IrOp::Legacy(_)
        ) && !matches!(op, IrOp::Legacy(L::LiveField { .. }));
        if committed && faultable {
            return true;
        }
        committed |= match op {
            IrOp::WriteReg { .. }
            | IrOp::Bcd { .. }
            | IrOp::Bt { .. }
            | IrOp::SetIf { .. }
            | IrOp::SetDf { .. }
            | IrOp::SetCf { .. }
            | IrOp::Lahf
            | IrOp::Sahf => true,
            IrOp::Add { set_flags, .. }
            | IrOp::Adc { set_flags, .. }
            | IrOp::Sub { set_flags, .. }
            | IrOp::Sbb { set_flags, .. }
            | IrOp::And { set_flags, .. }
            | IrOp::Or { set_flags, .. }
            | IrOp::Xor { set_flags, .. }
            | IrOp::Shl { set_flags, .. }
            | IrOp::Shr { set_flags, .. }
            | IrOp::Sar { set_flags, .. }
            | IrOp::Rol { set_flags, .. }
            | IrOp::Ror { set_flags, .. }
            | IrOp::Rcl { set_flags, .. }
            | IrOp::Rcr { set_flags, .. }
            | IrOp::DoubleShift { set_flags, .. }
            | IrOp::Mul { set_flags, .. } => *set_flags != FlagMask::NONE,
            IrOp::Legacy(op) => !matches!(
                op,
                L::Address { .. }
                    | L::Flat32Address { .. }
                    | L::LiveField { .. }
                    | L::CoprocessorOverrun { .. }
                    | L::DirectStackAddress { .. }
                    | L::ReadSelector { .. }
                    | L::ReadControl { .. }
                    | L::ReadDebugTest { .. }
                    | L::ReadSystemSegment { .. }
                    | L::SharedWalk
                    | L::Checkpoint
                    | L::Bound { .. }
                    | L::CheckTaskReturn
                    | L::Privileged { .. }
                    | L::LockedTransfer
            ),
            _ => false,
        };
    }
    false
}

// The catalogue entry that governs this lowering. I8088/I8086 lowering carries
// its selected entry (decision-23 §2); the generation projection serves the
// system models, whose entries it reproduces exactly.
#[inline(always)]
fn lowering_profile(
    ops: &impl LiftBuffer,
    context: MachineContext,
) -> &'static crate::ModelProfile {
    match ops.model_profile() {
        Some(profile) => profile,
        None => {
            debug_assert!(
                context.generation != crate::CpuGeneration::I8086,
                "8086-family lowering carries its catalogue entry"
            );
            crate::model::profile_for_generation(context.generation)
        }
    }
}

pub(super) fn direct_address(
    ops: &mut impl LiftBuffer,
    tg: &mut TempGen,
    segment: SegmentId,
    offset: Val,
    bytes: u8,
    context: MachineContext,
    ip: u64,
) -> Val {
    // Only an 8086-family context can select the wrapped shape.
    // Keep the catalogue lookup off the existing system-model lift paths.
    if context.generation == crate::CpuGeneration::I8086
        && lowering_profile(ops, context).quirks.wrap_segment_offsets
    {
        return super::i8086::tagged_address(ops, tg, segment, offset);
    }
    // I286 PRM Appendix B MOV p. B-74; I386 PRM §14.6 Table 14-1:
    // check the whole operand before forming a selector-derived linear address.
    if bytes != 0 {
        ops.push(IrOp::SegLimitCheck {
            alias_low16: false,
            offset,
            size: bytes,
            vector: if segment == SegmentId::Ss {
                lowering_profile(ops, context).lift.direct_stack_fault
            } else {
                13
            },
            fault_ip: ip,
        });
    }
    let reg = match segment {
        SegmentId::Cs => Reg::Cs,
        SegmentId::Ds => Reg::Ds,
        SegmentId::Es => Reg::Es,
        SegmentId::Ss => Reg::Ss,
        SegmentId::Fs | SegmentId::Gs => Reg::Cs,
    };
    let sel = match segment {
        SegmentId::Fs | SegmentId::Gs => {
            let dst = tg.fresh();
            ops.push(IrOp::ReadSelector {
                dst,
                segment: if segment == SegmentId::Fs { 4 } else { 5 },
            });
            Val::Temp(dst)
        }
        _ => read_reg(reg, ops, tg),
    };
    let base = alu_none(ops, tg, |dst| IrOp::Shl {
        dst,
        a: sel,
        b: Val::Imm(4),
        size: 8,
        set_flags: FlagMask::NONE,
    });
    add_addr(offset, base, ops, tg)
}

fn direct_addresses(ops: &mut Vec<IrOp>, tg: &mut TempGen, context: MachineContext, ip: u64) {
    let mut result = Vec::with_capacity(ops.len());
    for op in ops.drain(..) {
        if let IrOp::Legacy(L::Address {
            dst,
            segment,
            offset,
            bytes,
            ..
        }) = op
        {
            direct_address(&mut result, tg, segment, offset, bytes, context, ip);
            let Some(IrOp::Add { dst: output, .. }) = result.last_mut() else {
                unreachable!("address addition")
            };
            *output = dst;
        } else {
            result.push(op);
        }
    }
    *ops = result;
}

// Scalar RMW lowering computes its result before the store. Recompute only the
// flag image after that store succeeds, so a scalar needs no rollback
// snapshot (spec.md §16, fault atomicity).
pub(super) fn defer_store_flags(ops: &mut Vec<IrOp>, tg: &mut TempGen) {
    let Some(store) = ops.iter().rposition(|op| matches!(op, IrOp::Store { .. })) else {
        return;
    };
    let mut result = Vec::with_capacity(ops.len());
    let mut bit_flags = Vec::new();
    for (index, op) in ops.drain(..).enumerate() {
        if index < store {
            if let IrOp::Bt {
                result: dst,
                a,
                bit,
                size,
                op: kind,
            } = op
            {
                bit_flags.push(IrOp::Bt {
                    result: dst,
                    a,
                    bit,
                    size,
                    op: kind,
                });
                let shift = alu_none(&mut result, tg, |dst| IrOp::And {
                    dst,
                    a: bit,
                    b: Val::Imm(u64::from(size) * 8 - 1),
                    size: 8,
                    set_flags: FlagMask::NONE,
                });
                let bit_mask = alu_none(&mut result, tg, |dst| IrOp::Shl {
                    dst,
                    a: Val::Imm(1),
                    b: shift,
                    size: 8,
                    set_flags: FlagMask::NONE,
                });
                let b = if matches!(kind, BtOp::Reset) {
                    alu_none(&mut result, tg, |dst| IrOp::Xor {
                        dst,
                        a: bit_mask,
                        b: Val::Imm(u64::MAX),
                        size: 8,
                        set_flags: FlagMask::NONE,
                    })
                } else {
                    bit_mask
                };
                result.push(match kind {
                    BtOp::Set => IrOp::Or {
                        dst,
                        a,
                        b,
                        size,
                        set_flags: FlagMask::NONE,
                    },
                    BtOp::Reset => IrOp::And {
                        dst,
                        a,
                        b,
                        size,
                        set_flags: FlagMask::NONE,
                    },
                    BtOp::Complement => IrOp::Xor {
                        dst,
                        a,
                        b,
                        size,
                        set_flags: FlagMask::NONE,
                    },
                    BtOp::Test => unreachable!("a bit test has no store"),
                });
                continue;
            }
        }
        result.push(op);
        if index == store {
            result.append(&mut bit_flags);
        }
    }
    *ops = result;
    let Some(store) = ops.iter().rposition(|op| matches!(op, IrOp::Store { .. })) else {
        return;
    };
    let mut flags = Vec::new();
    for op in &mut ops[..store] {
        let original = op.clone();
        let mask = match op {
            IrOp::Add { set_flags, .. }
            | IrOp::Adc { set_flags, .. }
            | IrOp::Sub { set_flags, .. }
            | IrOp::Sbb { set_flags, .. }
            | IrOp::And { set_flags, .. }
            | IrOp::Or { set_flags, .. }
            | IrOp::Xor { set_flags, .. }
            | IrOp::Shl { set_flags, .. }
            | IrOp::Shr { set_flags, .. }
            | IrOp::Sar { set_flags, .. }
            | IrOp::Rol { set_flags, .. }
            | IrOp::Ror { set_flags, .. }
            | IrOp::Rcl { set_flags, .. }
            | IrOp::Rcr { set_flags, .. }
            | IrOp::DoubleShift { set_flags, .. } => set_flags,
            _ => continue,
        };
        if *mask != FlagMask::NONE {
            *mask = FlagMask::NONE;
            flags.push(original);
        }
    }
    ops.splice(store + 1..store + 1, flags);
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
pub(super) fn selector(ops: &mut impl LiftBuffer, tg: &mut TempGen, segment: SegmentId) -> Val {
    let reg = match segment {
        SegmentId::Cs => Reg::Cs,
        SegmentId::Ds => Reg::Ds,
        SegmentId::Es => Reg::Es,
        SegmentId::Ss => Reg::Ss,
        SegmentId::Fs | SegmentId::Gs => {
            let dst = tg.fresh();
            ops.push(IrOp::ReadSelector {
                dst,
                segment: segment as u8,
            });
            return Val::Temp(dst);
        }
    };
    read_reg(reg, ops, tg)
}
pub(super) fn load(ops: &mut impl LiftBuffer, tg: &mut TempGen, addr: Val, size: u8) -> Val {
    let dst = tg.fresh();
    ops.push(IrOp::Load {
        fault: Default::default(),
        dst,
        addr,
        size,
    });
    Val::Temp(dst)
}
// SDM Vol 2B §4.3 PUSH/POP/RET; 80386 PRM §17.1.3, pp. 17-1–17-2:
// operand width selects slots; SS.B independently selects SP or ESP.
pub(super) fn stack_offset(ops: &mut impl LiftBuffer, tg: &mut TempGen, delta: i64) -> Val {
    let context = ops.legacy_context().expect("stack context");
    let sp = read_reg(Reg::Rsp, ops, tg);
    let value = if delta == 0 {
        sp
    } else {
        alu_none(ops, tg, |dst| IrOp::Add {
            dst,
            a: sp,
            b: Val::Imm(delta as u64),
            size: 8,
            set_flags: FlagMask::NONE,
        })
    };
    alu_none(ops, tg, |dst| IrOp::And {
        dst,
        a: value,
        b: Val::Imm(if context.stack_address_bits == 16 {
            0xffff
        } else {
            0xffff_ffff
        }),
        size: 8,
        set_flags: FlagMask::NONE,
    })
}
fn stack_address(
    ops: &mut impl LiftBuffer,
    tg: &mut TempGen,
    offset: Val,
    size: u8,
    access: crate::AccessKind,
) -> Val {
    let context = ops.legacy_context().expect("stack context");
    // Only an 8086-family context can select the wrapped shape.
    // Keep the catalogue lookup off the existing system-model lift paths.
    if context.generation == crate::CpuGeneration::I8086
        && lowering_profile(ops, context).quirks.wrap_segment_offsets
    {
        return super::i8086::tagged_address(ops, tg, SegmentId::Ss, offset);
    }
    let dst = tg.fresh();
    if context.segment_class == crate::SegmentClass::Direct {
        emit(
            ops,
            L::DirectStackAddress {
                dst,
                offset,
                bytes: size,
                vector: lowering_profile(ops, context).lift.direct_stack_fault,
            },
        );
    } else {
        emit(
            ops,
            L::Address {
                dst,
                segment: SegmentId::Ss,
                offset,
                bytes: size,
                access,
            },
        );
    }
    Val::Temp(dst)
}
pub(super) fn write_sp(ops: &mut impl LiftBuffer, value: Val) {
    ops.push(IrOp::WriteReg {
        reg: Reg::Rsp,
        src: value,
        size: ops
            .legacy_context()
            .expect("stack context")
            .stack_address_bits
            / 8,
    });
}
pub(super) fn push_value(ops: &mut impl LiftBuffer, tg: &mut TempGen, value: Val, size: u8) {
    let offset = stack_offset(ops, tg, -i64::from(size));
    let addr = stack_address(ops, tg, offset, size, crate::AccessKind::Write);
    ops.push(IrOp::Store {
        fault: crate::ir::MemoryAccess::new(u16::from(size), false, true),
        addr,
        src: value,
        size,
        order: MemOrder::None,
    });
    write_sp(ops, offset);
}
fn check_target(
    ops: &mut impl LiftBuffer,
    tg: &mut TempGen,
    ip: Val,
    size: u8,
    fault_ip: u64,
) -> Val {
    let value = ip;
    let context = ops.legacy_context().expect("transfer context");
    if context.segment_class == crate::SegmentClass::Direct {
        if size == 4 {
            ops.push(IrOp::SegLimitCheck {
                alias_low16: false,
                offset: value,
                size: 1,
                vector: 13,
                fault_ip,
            });
        }
    } else {
        let dst = tg.fresh();
        emit(
            ops,
            L::Address {
                dst,
                segment: SegmentId::Cs,
                offset: value,
                bytes: 1,
                access: crate::AccessKind::Execute,
            },
        );
    }
    value
}

pub(super) fn far_target(
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
    let profile = lowering_profile(ops, context);
    if matches!(
        profile.isa,
        crate::ModelIsa::I80486Dx | crate::ModelIsa::I80486DxCpuid
    ) {
        emit(ops, L::I486Gate);
    }
    // 287 PRM Part II Ch. 2 p. 2-16, App. B item 2 p. B-1;
    // SDM Vol 1 §8.3.13 p. 8-24; Intel486 PRM §17.7 p. 17-8,
    // §25.2.3 / §25.3 p. 25-11: these controls affect no internal state.
    if matches!(insn.mnemonic(), Fneni | Fndisi)
        && context.generation != crate::CpuGeneration::I8086
    {
        return Ok(false);
    }
    if insn.mnemonic() == Fnsetpm && context.generation == crate::CpuGeneration::I386 {
        return Ok(false);
    }
    // 287 FSETPM changes addressing mode (Part II Ch. 2 p. 2-16).
    // Its independent mode latch remains outside the supported 287 scope.
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
    // 287 PRM Part II App. A Table A-2 pp. A-2–5 notes 1/4/6–9;
    // SDM Vol 3B §25.18.9 p. 25-15: these register aliases execute on
    // the attached FPU. Engine policy retains the 286 protected-mode
    // coprocessor refusal, including for packed-decimal transfers.
    if (context.generation != crate::CpuGeneration::I286
        || context.regime == crate::MachineRegime::Real)
        && (matches!(insn.mnemonic(), Fbld | Fbstp | Ffreep | Fstpnce)
            || matches!(
                insn.code(),
                iced_x86::Code::Fxch_st0_sti_DDC8
                    | iced_x86::Code::Fxch_st0_sti_DFC8
                    | iced_x86::Code::Fstp_sti_DFD0
                    | iced_x86::Code::Fstp_sti_DFD8
            ))
    {
        control::lift_x87(insn, ops, tg)?;
        return Ok(false);
    }
    // 286 PRM App. B pp. B-21/60/66/71/72/102/108/111;
    // 386 PRM §17 pp. 17-25/89/98/106/108/159/166/169 (scans checked):
    // these instructions are protected-only, before any operand access.
    // SDM Vol 2A ARPL pp. 3-75–76, LAR p. 3-536, LLDT p. 3-559,
    // LSL p. 3-575, LTR p. 3-577; Vol 2B SLDT p. 4-657, STR p. 4-679;
    // Vol 2C VERR/VERW p. 5-163: V86 also raises #UD before operands.
    if context.regime != crate::MachineRegime::Protected
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
        // Intel486 PRM §26 INVD/INVLPG/WBINVD pp. 26-171–172, 26-281.
        Invlpg if context.paging_enabled => {
            let offset = effective_address_no_segment(insn, ops, tg)?;
            emit(ops, L::InvalidatePage {
                segment: segment(insn.memory_segment()).expect("memory segment"),
                offset,
            });
        }
        // No paged state with PG off: preserve the existing lowering and walk.
        Invd | Wbinvd | Invlpg => emit(ops, L::I486Invalidate),
        // PRM §26 BSWAP p. 26-35: 16-bit result is undefined.
        Bswap if size == 2 => emit(ops, L::Refuse { operation: crate::UnsupportedSystemOperation::LegacyInstruction }),
        Cpuid => emit(ops, L::I486Cpuid { signature: crate::model::I486_DX_CPUID_SIGNATURE }),
        Pushf | Pushfd | Popf | Popfd | Iret | Iretd if !profile.flags_helper_supported() => {
            return Err(unsupported_insn(insn))
        }
        Lldt | Ltr => {
            let value = lower_read(insn, 0, ops, tg)?;
            // SDM Vol 2A §3.3 LLDT/LTR; 286 PRM App. B pp. B-66/B-72;
            // 386 PRM §17 pp. 17-98/17-108: GDT LDT or available TSS.
            let (register, allowed_types) = if insn.mnemonic() == Lldt {
                (SystemRegister::Ldtr, 1 << 2)
            } else {
                (SystemRegister::Tr, profile.lift.available_tss_types)
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
            let allowed_system_types = match insn.mnemonic() {
                Lar => profile.lift.lar_system_types,
                Lsl => profile.lift.lsl_system_types,
                _ => 0,
            };
            emit(
                ops,
                L::QueryDescriptor {
                    selector,
                    query,
                    allowed_system_types,
                    allow_conforming: profile.lift.lsl_conforming || insn.mnemonic() != Lsl,
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
                    (
                        load(ops, tg, addr, 2),
                        SelectorDestination::Memory {
                            addr,
                            segment: segment(insn.memory_segment()).expect("memory segment"),
                        },
                    )
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
        // Intel, "Undocumented iAPX 286 Test Instruction", p. 1: LOADALL
        // belongs to the 80286; another generation keeps the undefined opcode.
        Loadall if context.generation == crate::CpuGeneration::I286 => emit(ops, L::Loadall),
        Loadall => ops.push(IrOp::Trap {
            vector: 6,
            advance: 0,
        }),
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
        Mov if insn.op_register(0).is_dr()
            || insn.op_register(1).is_dr()
            || insn.op_register(0).is_tr()
            || insn.op_register(1).is_tr() =>
        {
            let writing = insn.op_register(0).is_dr() || insn.op_register(0).is_tr();
            let register = match insn.op_register(u32::from(!writing)) {
                Register::DR0 => DebugTestRegister::Dr0,
                Register::DR1 => DebugTestRegister::Dr1,
                Register::DR2 => DebugTestRegister::Dr2,
                Register::DR3 => DebugTestRegister::Dr3,
                // SDM Vol 2B §4.3 MOV DR p. 4-35 explicitly describes
                // Intel386 compatibility: DR4/DR5 alias DR6/DR7.
                Register::DR4 | Register::DR6 => DebugTestRegister::Dr6,
                Register::DR5 | Register::DR7 => DebugTestRegister::Dr7,
                Register::TR6 => DebugTestRegister::Tr6,
                Register::TR7 => DebugTestRegister::Tr7,
                Register::TR3 | Register::TR4 | Register::TR5 => {
                    // 486 PRM §10.7.2 pp. 10-13–17: cache test registers.
                    emit(ops, L::Refuse { operation: crate::UnsupportedSystemOperation::LegacyCacheTest });
                    return Ok(true);
                }
                _ => unreachable!("generation decode excludes other special registers"),
            };
            // 80386 PRM §17 MOV p. 17-111: always a 32-bit GPR transfer,
            // irrespective of 66h. Preserve undefined arithmetic flags (ASSUMED).
            if writing {
                let value = lower_read(insn, 1, ops, tg)?;
                emit(ops, L::WriteDebugTest { register, value });
            } else {
                let dst = tg.fresh();
                emit(ops, L::ReadDebugTest { dst, register });
                let target = lower_write_target(insn, 0, ops, tg)?;
                emit_write(ops, tg, target, Val::Temp(dst));
            }
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
            let value =
                if profile.quirks.push_sp_after_decrement && insn.op0_register() == Register::SP {
                    stack_offset(ops, tg, -2)
                } else {
                    value
                };
            push_value(ops, tg, value, size);
        }
        Pop => {
            let offset = stack_offset(ops, tg, 0);
            let addr = stack_address(ops, tg, offset, size, crate::AccessKind::Read);
            let value = load(ops, tg, addr, size);
            let next_sp = stack_offset(ops, tg, i64::from(size));
            if let Some(segment) = segment(insn.op0_register()) {
                emit(
                    ops,
                    L::LoadSegment {
                        segment,
                        value,
                        inhibit: segment == SegmentId::Ss,
                    },
                );
                write_sp(ops, next_sp);
            } else {
                if insn.op0_kind() == OpKind::Memory {
                    // SDM Vol 2B §4.3 POP: ESP-based destinations use post-pop ESP.
                    // Substitute its staged value; commit SP after the store succeeds.
                    let raw = read_reg(Reg::Rsp, ops, tg);
                    let high = alu_none(ops, tg, |dst| IrOp::And {
                        dst,
                        a: raw,
                        b: Val::Imm(if context.stack_address_bits == 16 {
                            !0xffff
                        } else {
                            0
                        }),
                        size: 8,
                        set_flags: FlagMask::NONE,
                    });
                    let post = alu_none(ops, tg, |dst| IrOp::Or {
                        dst,
                        a: high,
                        b: next_sp,
                        size: 8,
                        set_flags: FlagMask::NONE,
                    });
                    let first = ops.len();
                    let target = lower_write_target(insn, 0, ops, tg)?;
                    for op in &mut ops[first..] {
                        if let IrOp::ReadReg { dst, reg: Reg::Rsp } = op {
                            *op = IrOp::Or {
                                dst: *dst,
                                a: post,
                                b: Val::Imm(0),
                                size: 8,
                                set_flags: FlagMask::NONE,
                            };
                        }
                    }
                    emit_write(ops, tg, target, value);
                    write_sp(ops, next_sp);
                } else {
                    write_sp(ops, next_sp);
                    let target = lower_write_target(insn, 0, ops, tg)?;
                    emit_write(ops, tg, target, value);
                }
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
                        _ => {
                            let value = lower_read(insn, 0, ops, tg)?;
                            if insn.op0_kind() == OpKind::Register {
                                alu_none(ops, tg, |dst| IrOp::And {
                                    dst,
                                    a: value,
                                    b: Val::Imm(if size == 2 { 0xffff } else { 0xffff_ffff }),
                                    size: 8,
                                    set_flags: FlagMask::NONE,
                                })
                            } else {
                                value
                            }
                        }
                    },
                )
            };
            if !far {
                let ip = check_target(ops, tg, ip, size, insn.ip());
                if insn.mnemonic() == Call {
                    push_value(
                        ops,
                        tg,
                        Val::Imm(insn.next_ip() & if size == 2 { 0xffff } else { 0xffff_ffff }),
                        size,
                    );
                }
                ops.push(IrOp::Jump { target: ip });
                return Ok(true);
            }
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
        Ret => {
            let offset = stack_offset(ops, tg, 0);
            let addr = stack_address(ops, tg, offset, size, crate::AccessKind::Read);
            let ip = load(ops, tg, addr, size);
            let ip = check_target(ops, tg, ip, size, insn.ip());
            let extra = if insn.op_count() == 0 {
                0
            } else {
                insn.immediate16()
            };
            let next_sp = stack_offset(ops, tg, i64::from(size) + i64::from(extra));
            write_sp(ops, next_sp);
            ops.push(IrOp::Jump { target: ip });
        }
        Retf => emit(
            ops,
            L::Return {
                size,
                far: true,
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
        Ud0 | Ud2 => ops.push(IrOp::Trap {
            vector: 6,
            advance: 0,
        }),
        Iret | Iretd => {
            // SDM Vol 3A §11.3: IRET is serializing in real and protected mode.
            ops.push(IrOp::Fence {
                kind: FenceKind::Full,
            });
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
            let check_target = matches!(context.segment_class, crate::SegmentClass::Checked | crate::SegmentClass::Segmented)
                || insn.near_branch_target() > 0xffff;
            if count_bits == 16 && !check_target {
                // Both Direct paths use the compact, statically word-sized form.
                ops.push(IrOp::LoopCx {
                    kind,
                    taken: insn.near_branch_target(),
                    fallthrough: u64::from(next),
                });
            } else {
                emit(
                    ops,
                    L::Loop {
                        kind,
                        count_bits,
                        check_target,
                        taken: insn.near_branch_target() as u32,
                        next,
                    },
                );
            }
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
            let addr = if context.segment_class == crate::SegmentClass::Direct {
                direct_address(
                    ops,
                    tg,
                    segment(insn.memory_segment()).unwrap_or(SegmentId::Ds),
                    offset,
                    1,
                    context,
                    insn.ip(),
                )
            } else {
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
                Val::Temp(dst)
            };
            let value = load(ops, tg, addr, 1);
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
            // 80386 PRM §17.2.2.4 pp. 17-14–15: a register bit index
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
        Daa | Das | Aaa | Aas | Aam | Aad
            if context.generation == crate::CpuGeneration::I386 =>
        {
            // 80386 PRM §17 pp. 17-18–21, 17-54–55: these byte/AX
            // operations are independent of CS.D. Reuse real-mode lowering.
            // SDM Vol 2A §3.2 AAM p. 3-5: base zero raises #DE.
            return lift_insn_inner(
                insn,
                ops,
                tg,
                CpuMode::Real16,
                &mut mmx::Sequence::default(),
            );
        }
        Wait if context.generation == crate::CpuGeneration::I386 => {
            // 80386 PRM §17 p. 17-171; §§11.1.2–11.1.4 pp. 11-1–2:
            // the preceding Coprocessor gate tests MP && TS; the shared
            // WAIT operation handles the configured coprocessor's errors.
            control::lift_x87(insn, ops, tg)?;
        }
        _ if context.generation == crate::CpuGeneration::I386
            && insn.cpuid_features().iter().any(|feature| {
                matches!(
                    feature,
                    iced_x86::CpuidFeature::FPU
                        | iced_x86::CpuidFeature::FPU287
                        | iced_x86::CpuidFeature::FPU387
                )
            }) =>
        {
            // 80386 PRM §§11.1.1–11.1.2 p. 11-1: ESC uses the attached
            // 387 in either regime. SDM Vol 1 §8.1.10, Figs. 8-9–8-12
            // pp. 8-11–12; 486 PRM §15.1.6 pp. 15-8–10: V86 uses real
            // formats, independently of operand size and PE.
            use crate::x87::{FpuKind as K, X87_ENV_REAL, X87_ENV16};
            let environment = match insn.mnemonic() {
                Fldenv => Some(K::Fldenv),
                Fnstenv | Fstenv => Some(K::Fnstenv),
                Fnsave | Fsave => Some(K::Fnsave),
                Frstor => Some(K::Frstor),
                _ => None,
            };
            if let Some(kind) = environment {
                let (addr, operand_offset) = effective_address_parts(insn, ops, tg)?;
                ops.push(IrOp::X87 {
                    kind,
                    addr,
                    operand_offset,
                    instruction: 0,
                    fault: Default::default(),
                    sti: if context.regime != crate::MachineRegime::Protected {
                        X87_ENV_REAL
                    } else {
                        0
                    } | if matches!(insn.memory_size().size(), 14 | 94) {
                        X87_ENV16
                    } else {
                        0
                    },
                });
            } else {
                control::lift_x87(insn, ops, tg)?;
            }
        }
        Cli => ops.push(IrOp::SetIf { value: false }),
        Sti => ops.push(IrOp::SetIf { value: true }),
        // Shared scalar lowering has no legacy control/stack assumptions here.
        // SDM Vol 3B §23.1.3 and 80386 PRM §14.2, pp. 14-2–14-3.
        Mov | Lea | Add | Adc | Sub | Sbb | And | Or | Xor | Cmp | Test | Inc | Dec | Neg | Not
        | Xchg | Shl | Sal | Shr | Sar | Rol | Ror | Rcl | Rcr | Shld | Shrd | Mul | Imul | Div
        | Idiv | Movzx | Movsx | Cbw | Cwde | Cwd | Cdq | Clc | Stc | Cmc | Cld | Std | Lahf
        | Sahf | Nop | Hlt | In | Out | Jb | Jbe | Je | Jg | Jge | Jl | Jle | Jae | Ja | Jne
        | Jno | Jnp | Jns | Jo | Jp | Js | Setb | Setbe | Sete | Setg | Setge | Setl | Setle
        | Setae | Seta | Setne | Setno | Setnp | Setns | Seto | Setp | Sets | Bsf | Bsr
        // 80386 PRM §17 pp. 17-29–36: register destinations and immediate
        // indices wrap within the operand width. Memory/register bit strings
        // retain the signed displacement lowering above (§17.2.2.4 pp. 17-14–15).
        // 486 PRM §26 BSWAP p. 26-35, CMPXCHG pp. 26-62–63,
        // XADD pp. 26-282–283: reuse the shared scalar/atomic lowering.
        | Bt | Bts | Btr | Btc | Bswap | Xadd | Cmpxchg => {
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
    Ok(matches!(
        insn.mnemonic(),
        Call | Ret
            | Retf
            | Jmp
            | Iret
            | Iretd
            | Loop
            | Loope
            | Loopne
            | Jcxz
            | Jecxz
            | Bound
            | Enter
            | Leave
            | Pushf
            | Pushfd
            | Popf
            | Popfd
            | Pusha
            | Pushad
            | Popa
            | Popad
            | Les
            | Lds
            | Lss
            | Lfs
            | Lgs
            | Loadall
    ) || (insn.mnemonic() == Mov && insn.op0_register().is_segment_register())
        || (insn.mnemonic() == Pop && insn.op0_register().is_segment_register()))
}

#[cfg(test)]
mod restartability_tests {
    use super::*;

    #[test]
    fn rollback_mark_follows_lowering_order_in_checked_and_direct_contexts() {
        let cpu = crate::CpuState::legacy_real_entry([0; 6]);
        let context = crate::legacy::derive_legacy_context(
            crate::LegacyCpuModel {
                generation: crate::CpuGeneration::I386,
                model_semantics_id: 0,
            },
            &cpu,
        )
        .unwrap();
        for class in [crate::SegmentClass::Direct, crate::SegmentClass::Checked] {
            for code in [
                &b"\x01\x06\x00\x20"[..],     // ADD [2000h],AX
                &b"\x11\x06\x00\x20"[..],     // ADC [2000h],AX
                &b"\x19\x06\x00\x20"[..],     // SBB [2000h],AX
                &b"\x29\x06\x00\x20"[..],     // SUB [2000h],AX
                &b"\x09\x06\x00\x20"[..],     // OR [2000h],AX
                &b"\x21\x06\x00\x20"[..],     // AND [2000h],AX
                &b"\x31\x06\x00\x20"[..],     // XOR [2000h],AX
                &b"\xd1\x26\x00\x20"[..],     // SHL word [2000h],1
                &b"\xd1\x1e\x00\x20"[..],     // RCR word [2000h],1
                &b"\xff\x06\x00\x20"[..],     // INC word [2000h]
                &b"\xff\x0e\x00\x20"[..],     // DEC word [2000h]
                &b"\xf7\x16\x00\x20"[..],     // NOT word [2000h]
                &b"\xf7\x1e\x00\x20"[..],     // NEG word [2000h]
                &b"\x87\x06\x00\x20"[..],     // XCHG [2000h],AX
                &b"\x0f\xab\x06\x00\x20"[..], // BTS [2000h],AX
                &b"\x66\x11\x06\x00\x20"[..], // ADC [2000h],EAX
                &b"\x66\xd1\x26\x00\x20"[..], // SHL dword [2000h],1
                &b"\x03\x06\x00\x20"[..],     // ADD AX,[2000h]
                &b"\x89\x06\x00\x20"[..],     // MOV [2000h],AX
                &b"\x50"[..],
                &b"\x90"[..],
            ] {
                let ir = instruction(
                    code,
                    0x1000,
                    MachineContext {
                        segment_class: class,
                        ..context
                    },
                )
                .unwrap();
                assert!(
                    !matches!(ir.ops.get(1), Some(IrOp::Legacy(L::Checkpoint))),
                    "{class:?} {code:x?}"
                );
            }
        }
        // The conservative helper-commit rule still marks a coprocessor gate
        // followed by a fallible body; scalar flag staging does not erase it.
        let ir = instruction(
            b"\xd9\x3e\x00\x20", // FNSTCW [2000h]
            0x1000,
            MachineContext {
                segment_class: crate::SegmentClass::Checked,
                ..context
            },
        )
        .unwrap();
        assert!(matches!(ir.ops.get(1), Some(IrOp::Legacy(L::Checkpoint))));
    }
}
