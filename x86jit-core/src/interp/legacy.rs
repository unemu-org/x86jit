//! Legacy real interpreter, selected before the op loop (decision-18 §§1, 4;
//! spec.md §§8.1, 17). Shared scalar helpers retain their existing flag semantics.
use super::*;
use crate::legacy::{self, ControlRegister, LegacyDecodeContext, LegacyOp as L, SegmentId};

#[derive(Default)]
pub(crate) struct Outcome {
    pub exit: Option<Exit>,
    pub retired: bool,
    pub context_changed: bool,
    pub arm_sti: bool,
    pub arm_ss: bool,
}

fn refusal(cpu: &CpuState, operation: crate::UnsupportedSystemOperation) -> Exit {
    Exit::UnsupportedSystemContext {
        addr: cpu.rip,
        operation,
    }
}

fn exception(cpu: &CpuState, vector: u8) -> Exit {
    // SDM Vol 3B §23.1.4: real-address faults save the faulting offset
    // and do not carry an error code.
    let mut details = crate::fault::FaultMetadata::architectural_exception(
        cpu.rip,
        cpu.rip,
        vector,
        crate::fault::FaultClass::Fault,
        crate::fault::ErrorCode::NoCode,
    );
    details.path = crate::fault::FaultPath::Interpreter;
    Exit::exception(cpu.rip, vector).with_fault_metadata(details)
}

fn write_cr0(cpu: &mut CpuState, value: u32) -> Result<bool, Exit> {
    // SDM Vol 3A §2.5: PG without PE is illegal, independently of whether
    // this engine can execute the proposed regime (decision-18 §4).
    if value & 0x8000_0001 == 0x8000_0000 {
        return Err(exception(cpu, 13));
    }
    if value & !cpu.legacy.cr0 & 0x8000_0001 != 0 {
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyExecution,
        ));
    }
    let changed = cpu.legacy.cr0 != value;
    cpu.legacy.cr0 = value;
    Ok(changed)
}

fn selector(cpu: &CpuState, segment: SegmentId) -> u16 {
    match segment {
        SegmentId::Cs => cpu.cs,
        SegmentId::Ds => cpu.ds,
        SegmentId::Es => cpu.es,
        SegmentId::Ss => cpu.ss,
        SegmentId::Fs => cpu.fs,
        SegmentId::Gs => cpu.gs,
    }
}

fn reload(cpu: &mut CpuState, segment: SegmentId, value: u16) {
    *match segment {
        SegmentId::Cs => &mut cpu.cs,
        SegmentId::Ds => &mut cpu.ds,
        SegmentId::Es => &mut cpu.es,
        SegmentId::Ss => &mut cpu.ss,
        SegmentId::Fs => &mut cpu.fs,
        SegmentId::Gs => &mut cpu.gs,
    } = value;
    // SDM Vol 3A §12.9.2; Vol 3B §23.1.1; AMD64 APM Vol 2 §4.1:
    // real reload changes base; retained limits/attributes are not synthesized.
    cpu.legacy.segments[segment as usize].base = u32::from(value) << 4;
    cpu.legacy.segments[segment as usize].unusable = 0;
    if segment == SegmentId::Cs {
        cpu.legacy.reset_fetch = 0;
    }
}

fn sp(cpu: &CpuState, context: LegacyDecodeContext) -> u64 {
    cpu.gpr[RSP] & mask(context.stack_address_bits / 8)
}
fn adjust(cpu: &mut CpuState, context: LegacyDecodeContext, bytes: u32) {
    cpu.write_gpr(
        RSP,
        sp(cpu, context).wrapping_add(u64::from(bytes)),
        context.stack_address_bits / 8,
    );
}

fn read<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    addr: u64,
    size: u8,
) -> Result<u64, Exit> {
    if let Some(value) = cpu.mmio_part(cpu.rip, addr) {
        return Ok(value & mask(size));
    }
    let value = mem.read(addr, size).or_else(|trap| {
        // Decision-16 §3: replayed RAM accesses cannot consume the answer
        // belonging to a later MMIO transfer of this instruction.
        if trap == MemTrap::Mmio {
            if let Some(value) = cpu.pending_mmio.take() {
                cpu.record_mmio_part(cpu.rip, addr, value);
                return Ok(value);
            }
        }
        Err(
            match trap_out(cpu, cpu.rip, trap, addr, size, AccessKind::Read, 0).into_step() {
                StepResult::Exit(exit) => exit,
                _ => unreachable!(),
            },
        )
    })?;
    Ok(value & mask(size))
}

fn write<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    addr: u64,
    size: u8,
    value: u64,
) -> Result<(), Exit> {
    // The retry ledger is engine storage. Bit 63 distinguishes answered writes
    // from reads at the same 32-bit linear address; it is never a guest address.
    let key = addr | (1 << 63);
    if cpu.mmio_part(cpu.rip, key).is_some() {
        return Ok(());
    }
    if let Err(trap) = mem.write(addr, value, size) {
        if trap == MemTrap::Mmio && cpu.pending_mmio_write {
            cpu.pending_mmio_write = false;
            cpu.record_mmio_part(cpu.rip, key, value);
        } else {
            return Err(
                match trap_out(cpu, cpu.rip, trap, addr, size, AccessKind::Write, value).into_step()
                {
                    StepResult::Exit(exit) => exit,
                    _ => unreachable!(),
                },
            );
        }
    }
    Ok(())
}

fn probe<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    addr: u64,
    size: u8,
) -> Result<(), Exit> {
    match mem.probe_write(addr, size) {
        Ok(()) | Err(MemTrap::Mmio) => Ok(()),
        Err(trap) => Err(
            match trap_out(cpu, cpu.rip, trap, addr, size, AccessKind::Write, 0).into_step() {
                StepResult::Exit(exit) => exit,
                _ => unreachable!(),
            },
        ),
    }
}

// SDM Vol 1 §6.2.3; Vol 3A §3.4.5; Vol 2B §4.3 PUSH:
// transfer width and SS.B-selected pointer width are independent. CALL's
// CS/EIP slots use that same boundary (Vol 2A §3.3 CALL).
fn push<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    context: LegacyDecodeContext,
    values: &[u64],
    size: u8,
) -> Result<(), Exit> {
    let old = sp(cpu, context);
    let mut addresses = Vec::with_capacity(values.len());
    for i in 1..=values.len() {
        let offset =
            old.wrapping_sub(i as u64 * u64::from(size)) & mask(context.stack_address_bits / 8);
        let address = legacy::access(cpu, SegmentId::Ss, offset, size)?;
        probe(cpu, mem, address, size)?;
        addresses.push(address);
    }
    for (&addr, &value) in addresses.iter().zip(values) {
        write(cpu, mem, addr, size, value)?;
    }
    cpu.write_gpr(
        RSP,
        old.wrapping_sub(values.len() as u64 * u64::from(size)),
        context.stack_address_bits / 8,
    );
    Ok(())
}
fn peek<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    context: LegacyDecodeContext,
    index: u64,
    size: u8,
) -> Result<u64, Exit> {
    let offset = sp(cpu, context).wrapping_add(index * u64::from(size))
        & mask(context.stack_address_bits / 8);
    let addr = legacy::access(cpu, SegmentId::Ss, offset, size)?;
    read(cpu, mem, addr, size)
}
fn target(cpu: &CpuState, value: u64, size: u8) -> Result<u64, Exit> {
    let offset = value & mask(size);
    legacy::access(cpu, SegmentId::Cs, offset, 1)?;
    Ok(offset)
}

/// Shared real-address delivery for software INT and queued maskable IRQs.
/// Faults remain outward until the exception-routing seam (decision-18 §4).
pub(crate) fn deliver_interrupt<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    memory: &Memory,
    context: LegacyDecodeContext,
    vector: u8,
    saved_ip: u32,
) -> Result<(), Exit> {
    let mem = Access::<ROUTED, ORDER>::new(memory);
    deliver(cpu, mem, context, vector, saved_ip)
}

fn deliver<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    context: LegacyDecodeContext,
    vector: u8,
    saved_ip: u32,
) -> Result<(), Exit> {
    // SDM Vol 3B §23.1.4; Vol 2A §3.3 INT, REAL-ADDRESS-MODE:
    // four-byte IDTR-relative entries, inclusive limit, six-byte frame,
    // no error code. Validate/read the target before committing the frame.
    let offset = u32::from(vector) * 4;
    if offset + 3 > u32::from(cpu.legacy.idtr.limit) {
        // 80386 PRM §14.3 p.14-3, §14.6 Table 14-1 p.14-5, Table 14-2 p.14-8:
        // I386 raises vector 8 here, diverging from the current SDM Vol 2A
        // §3.3 INT's #GP. Report the fault outward; escalation/shutdown are
        // deferred to the exception-routing seam (decision-18 §4).
        let fault = if context.generation == crate::CpuGeneration::I386 {
            8
        } else {
            13
        };
        return Err(exception(cpu, fault));
    }
    let address = u64::from(cpu.legacy.idtr.base.wrapping_add(offset));
    let ip = read(cpu, mem, address, 2)?;
    let cs = read(cpu, mem, u64::from((address as u32).wrapping_add(2)), 2)?;
    if ip > u64::from(cpu.legacy.segments[SegmentId::Cs as usize].limit) {
        return Err(exception(cpu, 13));
    }
    let flags = legacy::flags_image(cpu, context.generation);
    push(
        cpu,
        mem,
        context,
        &[u64::from(flags), u64::from(cpu.cs), u64::from(saved_ip)],
        2,
    )?;
    cpu.flags.if_ = false;
    // SDM Vol 3B §23.1.4: clear TF/RF/AC on entry.
    cpu.rflags_system &= !0x50100;
    reload(cpu, SegmentId::Cs, cs as u16);
    cpu.rip = ip;
    Ok(())
}

fn supported(op: &IrOp) -> bool {
    matches!(
        op,
        IrOp::InsnStart { .. }
            | IrOp::Legacy(_)
            | IrOp::ReadReg { .. }
            | IrOp::WriteReg { .. }
            | IrOp::Add { .. }
            | IrOp::Adc { .. }
            | IrOp::Sub { .. }
            | IrOp::Sbb { .. }
            | IrOp::And { .. }
            | IrOp::Or { .. }
            | IrOp::Xor { .. }
            | IrOp::Shl { .. }
            | IrOp::Shr { .. }
            | IrOp::Sar { .. }
            | IrOp::Rol { .. }
            | IrOp::Ror { .. }
            | IrOp::Rcl { .. }
            | IrOp::Rcr { .. }
            | IrOp::DoubleShift { .. }
            | IrOp::Sext { .. }
            | IrOp::Mul { .. }
            | IrOp::Div { .. }
            | IrOp::GetCond { .. }
            | IrOp::BitScan { .. }
            | IrOp::Load { .. }
            | IrOp::Store { .. }
            | IrOp::Jump { .. }
            | IrOp::Branch { .. }
            | IrOp::SetIf { .. }
            | IrOp::SetDf { .. }
            | IrOp::SetCf { .. }
            | IrOp::Lahf
            | IrOp::Sahf
            | IrOp::Hlt
            | IrOp::PortIo { .. }
            | IrOp::Trap { .. }
    )
}

/// One completed instruction or one REP iteration; incomplete work costs zero.
pub(crate) fn execute<const ROUTED: bool, const ORDER: u8>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    memory: &Memory,
    scratch: &mut impl ScratchBuffer,
) -> Result<Outcome, Exit> {
    let context = ir.legacy_context.expect("legacy interpreter context");
    if !ir.ops.iter().all(supported) {
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyInstruction,
        ));
    }
    // Save architectural destinations; device completion/retry ownership stays live
    // on failure. Scalar lowering places writes last, but a RMW can set flags
    // before its final (faulting) store. Legacy performance has no application target.
    let before = (
        cpu.gpr,
        cpu.flags,
        cpu.rflags_system,
        cpu.legacy,
        [cpu.cs, cpu.ds, cpu.es, cpu.ss, cpu.fs, cpu.gs],
    );
    let mem = Access::<ROUTED, ORDER>::new(memory);
    let result = walk(
        ir,
        cpu,
        mem,
        scratch.zeroed(ir.temp_count as usize),
        context,
    );
    if result.is_err() {
        cpu.gpr = before.0;
        cpu.flags = before.1;
        cpu.rflags_system = before.2;
        cpu.legacy = before.3;
        [cpu.cs, cpu.ds, cpu.es, cpu.ss, cpu.fs, cpu.gs] = before.4;
        cpu.rip = ir.guest_start;
    } else if result.as_ref().is_ok_and(|outcome| outcome.retired) {
        // SDM Vol 3B §20.3.1.1; 80386 PRM §12.3.1.1, p. 12-7:
        // completed ordinary instructions clear RF. Retry exits do not retire.
        cpu.rflags_system &= !(1 << 16);
    }
    result
}

fn walk<const ROUTED: bool, const ORDER: u8>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    temps: &mut [u64],
    context: LegacyDecodeContext,
) -> Result<Outcome, Exit> {
    // 80386 PRM §14.7 item 8, p. 14-6 (scan checked): sequential
    // execution across FFFFh faults on the next fetch instead of wrapping IP.
    let next = ir.guest_start.wrapping_add(u64::from(ir.guest_len)) & 0xffff_ffff;
    let mut outcome = Outcome {
        retired: true,
        ..Outcome::default()
    };
    for op in &ir.ops {
        let result = match op {
            IrOp::InsnStart { .. } => None,
            IrOp::ReadReg { dst, reg } => exec_read_reg(cpu, temps, dst, reg),
            IrOp::WriteReg { reg, src, size } => exec_write_reg(cpu, temps, reg, src, size),
            IrOp::Add {
                dst,
                a,
                b,
                size,
                set_flags,
            } => exec_add(cpu, temps, dst, a, b, size, set_flags),
            IrOp::Adc {
                dst,
                a,
                b,
                size,
                set_flags,
            } => exec_adc(cpu, temps, dst, a, b, size, set_flags),
            IrOp::Sub {
                dst,
                a,
                b,
                size,
                set_flags,
            } => exec_sub(cpu, temps, dst, a, b, size, set_flags),
            IrOp::Sbb {
                dst,
                a,
                b,
                size,
                set_flags,
            } => exec_sbb(cpu, temps, dst, a, b, size, set_flags),
            IrOp::And {
                dst,
                a,
                b,
                size,
                set_flags,
            } => exec_and(cpu, temps, dst, a, b, size, set_flags),
            IrOp::Or {
                dst,
                a,
                b,
                size,
                set_flags,
            } => exec_or(cpu, temps, dst, a, b, size, set_flags),
            IrOp::Xor {
                dst,
                a,
                b,
                size,
                set_flags,
            } => exec_xor(cpu, temps, dst, a, b, size, set_flags),
            IrOp::Shl {
                dst,
                a,
                b,
                size,
                set_flags,
            } => exec_shl(cpu, temps, dst, a, b, size, set_flags),
            IrOp::Shr {
                dst,
                a,
                b,
                size,
                set_flags,
            } => exec_shr(cpu, temps, dst, a, b, size, set_flags),
            IrOp::Sar {
                dst,
                a,
                b,
                size,
                set_flags,
            } => exec_sar(cpu, temps, dst, a, b, size, set_flags),
            IrOp::Rol {
                dst,
                a,
                b,
                size,
                set_flags,
            } => exec_rol(cpu, temps, dst, a, b, size, set_flags),
            IrOp::Ror {
                dst,
                a,
                b,
                size,
                set_flags,
            } => exec_ror(cpu, temps, dst, a, b, size, set_flags),
            IrOp::Rcl {
                dst,
                a,
                b,
                size,
                set_flags,
            } => exec_rcl(cpu, temps, dst, a, b, size, set_flags),
            IrOp::Rcr {
                dst,
                a,
                b,
                size,
                set_flags,
            } => exec_rcr(cpu, temps, dst, a, b, size, set_flags),
            IrOp::DoubleShift {
                dst,
                a,
                b,
                count,
                size,
                left,
                set_flags,
            } => exec_double_shift(cpu, temps, dst, a, b, count, size, left, set_flags),
            IrOp::Sext { dst, a, from } => exec_sext(temps, dst, a, from),
            IrOp::Mul {
                lo,
                hi,
                a,
                b,
                size,
                signed,
                set_flags,
            } => exec_mul(cpu, temps, lo, hi, a, b, size, signed, set_flags),
            IrOp::Div {
                quot,
                rem,
                hi,
                lo,
                divisor,
                size,
                signed,
            } => exec_div(
                cpu,
                temps,
                ir.guest_start,
                quot,
                rem,
                hi,
                lo,
                divisor,
                size,
                signed,
            ),
            IrOp::GetCond { dst, cond } => exec_get_cond(cpu, temps, dst, cond),
            IrOp::BitScan {
                dst,
                src,
                old,
                size,
                op,
            } => exec_bit_scan(cpu, temps, dst, src, old, size, op),
            IrOp::Load {
                dst, addr, size, ..
            } => {
                temps[*dst as usize] = read(cpu, mem, read_val(*addr, temps), *size)?;
                None
            }
            IrOp::Store {
                addr, src, size, ..
            } => {
                write(
                    cpu,
                    mem,
                    read_val(*addr, temps),
                    *size,
                    read_val(*src, temps),
                )?;
                None
            }
            IrOp::Jump { target: ip } => {
                cpu.rip = target(cpu, read_val(*ip, temps), 4)?;
                return Ok(outcome);
            }
            IrOp::Branch {
                cond,
                taken,
                fallthrough,
            } => {
                // SDM Vol 2A §3.3 Jcc: only a taken target is checked here;
                // sequential fallthrough is checked by the following fetch.
                cpu.rip = if eval_cond(*cond, &cpu.flags) {
                    target(cpu, *taken, 4)?
                } else {
                    *fallthrough
                };
                return Ok(outcome);
            }
            IrOp::SetIf { value } => {
                outcome.arm_sti = *value && !cpu.flags.if_;
                cpu.flags.if_ = *value;
                None
            }
            IrOp::SetDf { value } => {
                cpu.flags.df = *value;
                None
            }
            IrOp::SetCf { value } => {
                cpu.flags.cf = value.unwrap_or(!cpu.flags.cf);
                None
            }
            IrOp::Lahf => exec_lahf(cpu),
            IrOp::Sahf => exec_sahf(cpu),
            IrOp::Hlt => {
                cpu.rip = next;
                outcome.exit = Some(Exit::Hlt);
                return Ok(outcome);
            }
            IrOp::PortIo {
                port,
                value,
                size,
                dir_out,
            } => {
                let result = exec_port_io(cpu, temps, next, port, value, size, dir_out).unwrap();
                if let StepResult::Exit(exit) = result.into_step() {
                    outcome.exit = Some(exit);
                }
                return Ok(outcome);
            }
            IrOp::Trap { vector, .. } => {
                return Err(exception(cpu, *vector));
            }
            IrOp::Legacy(op) => {
                match op {
                    L::Table {
                        addr,
                        idtr,
                        load,
                        size,
                    } => {
                        let addr = read_val(*addr, temps);
                        let high = u64::from((addr as u32).wrapping_add(2));
                        if *load {
                            let limit = read(cpu, mem, addr, 2)? as u16;
                            let base = read(cpu, mem, high, 4)? as u32;
                            let table = legacy::TableRegister {
                                base: if *size == 2 { base & 0x00ff_ffff } else { base },
                                limit,
                                reserved: 0,
                            };
                            // SDM Vol 2A §3.3 LGDT/LIDT; 80386 PRM §17
                            // pp. 17-94–95 (page images checked): 16+24 or 16+32.
                            let destination = if *idtr {
                                &mut cpu.legacy.idtr
                            } else {
                                &mut cpu.legacy.gdtr
                            };
                            outcome.context_changed = *destination != table;
                            *destination = table;
                        } else {
                            let table = if *idtr {
                                cpu.legacy.idtr
                            } else {
                                cpu.legacy.gdtr
                            };
                            // SDM Vol 2B §4.3 SGDT/SIDT, IA-32 Architecture
                            // Compatibility; 80386 PRM §17 p. 17-154 compatibility
                            // note (scan checked): I386's 16-bit high byte is zero.
                            let base = if *size == 2 {
                                table.base & 0x00ff_ffff
                            } else {
                                table.base
                            };
                            probe(cpu, mem, addr, 2)?;
                            probe(cpu, mem, high, 4)?;
                            write(cpu, mem, addr, 2, u64::from(table.limit))?;
                            write(cpu, mem, high, 4, u64::from(base))?;
                        }
                    }
                    L::ReadControl { dst, register } => {
                        temps[*dst as usize] = u64::from(match register {
                            ControlRegister::Cr0 => cpu.legacy.cr0,
                            ControlRegister::Cr2 => cpu.legacy.cr2,
                            ControlRegister::Cr3 => cpu.legacy.cr3,
                        });
                    }
                    L::WriteControl { register, value } => {
                        let value = read_val(*value, temps) as u32;
                        // SDM Vol 3A §2.5; 80386 PRM §4.1.3 Fig. 4-2
                        // p. 4-3 (scan checked): I386 has CR0.PG/ET/TS/EM/MP/PE,
                        // CR2[31:0], and CR3[31:12]. Ignore reserved write bits;
                        // later CR0 controls and CR3.PCD/PWT are not I386 fields.
                        // ET is writable on I386 (PRM §11.1.1 p. 11-1, scan checked).
                        // MOV CR's arithmetic flags are undefined (SDM Vol 2B
                        // §4.3); retaining them is the engine's deterministic policy.
                        match register {
                            ControlRegister::Cr0 => {
                                outcome.context_changed = write_cr0(cpu, value & 0x8000_001f)?;
                            }
                            ControlRegister::Cr2 => cpu.legacy.cr2 = value,
                            ControlRegister::Cr3 => {
                                cpu.legacy.cr3 = value & 0xffff_f000;
                                // Decision-18 §3: even an equal CR3 write invalidates.
                                outcome.context_changed = true;
                            }
                        }
                    }
                    L::LoadMsw { value } => {
                        let value =
                            legacy::lmsw_value(cpu.legacy.cr0, read_val(*value, temps) as u16);
                        outcome.context_changed = write_cr0(cpu, value)?;
                    }
                    L::ClearTs => {
                        // SDM Vol 2A §3.3 CLTS; 80386 PRM §17 p. 17-48.
                        outcome.context_changed = write_cr0(cpu, cpu.legacy.cr0 & !8)?;
                    }
                    L::Interrupt {
                        vector,
                        saved_ip,
                        overflow_only,
                    } => {
                        if !overflow_only || cpu.flags.of {
                            deliver(cpu, mem, context, *vector, *saved_ip)?;
                            outcome.context_changed = true;
                            return Ok(outcome);
                        }
                    }
                    L::Address {
                        dst,
                        segment,
                        offset,
                        bytes,
                    } => {
                        temps[*dst as usize] =
                            legacy::access(cpu, *segment, read_val(*offset, temps), *bytes)?
                    }
                    L::ReadSelector { dst, segment } => {
                        temps[*dst as usize] = u64::from(selector(cpu, *segment))
                    }
                    L::LoadSegment {
                        segment,
                        value,
                        inhibit,
                    } => {
                        reload(cpu, *segment, read_val(*value, temps) as u16);
                        outcome.context_changed = true;
                        outcome.arm_ss = *inhibit;
                    }
                    L::Push { value, size } => {
                        push(cpu, mem, context, &[read_val(*value, temps)], *size)?
                    }
                    L::Pop { dst, size } => {
                        temps[*dst as usize] = peek(cpu, mem, context, 0, *size)?
                    }
                    L::AdjustStack { bytes } => adjust(cpu, context, *bytes),
                    L::Flags { size, pop } => {
                        if *pop {
                            let image = peek(cpu, mem, context, 0, *size)? as u32;
                            let before = cpu.rflags_system;
                            legacy::restore_flags(cpu, context.generation, image)?;
                            outcome.context_changed |= (before ^ cpu.rflags_system) & 0x7000 != 0;
                            adjust(cpu, context, u32::from(*size));
                        } else {
                            // SDM Vol 2B §4.3 PUSHF: the pushed RF/VM image is zero.
                            push(
                                cpu,
                                mem,
                                context,
                                &[u64::from(
                                    legacy::flags_image(cpu, context.generation) & !0x30000,
                                )],
                                *size,
                            )?;
                        }
                    }
                    L::All { size, pop } => {
                        // SDM Vol 2B §4.3 PUSHA/POPA: original SP/ESP occupies
                        // the fifth slot; POPA skips it without loading a register.
                        if *pop {
                            let mut values = [0; 8];
                            for (i, value) in values.iter_mut().enumerate() {
                                if i != 3 {
                                    *value = peek(cpu, mem, context, i as u64, *size)?;
                                }
                            }
                            for (i, reg) in [7, 6, 5, 4, 3, 2, 1, 0].into_iter().enumerate() {
                                if reg != 4 {
                                    cpu.write_gpr(reg, values[i], *size);
                                }
                            }
                            adjust(cpu, context, 8 * u32::from(*size));
                        } else {
                            // SDM Vol 2B §4.3 PUSHA/PUSHAD; 80386 PRM §17,
                            // p. 17-134: these real-mode cases precede pushes.
                            if matches!(sp(cpu, context), 7 | 9 | 11 | 13 | 15) {
                                return Err(exception(cpu, 13));
                            }
                            let values = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| cpu.gpr[i]);
                            push(cpu, mem, context, &values, *size)?;
                        }
                    }
                    L::Transfer {
                        cs,
                        ip,
                        size,
                        return_ip,
                    } => {
                        let ip = read_val(*ip, temps) & mask(*size);
                        // A real CS reload retains the inclusive limit. Validate
                        // before pushing a CALL frame or changing either CS image.
                        let ip = target(cpu, ip, *size)?;
                        if let Some(return_ip) = return_ip {
                            let mut frame = Vec::new();
                            if cs.is_some() {
                                frame.push(u64::from(cpu.cs));
                            }
                            frame.push(u64::from(*return_ip));
                            push(cpu, mem, context, &frame, *size)?;
                        }
                        if let Some(cs) = cs {
                            reload(cpu, SegmentId::Cs, read_val(*cs, temps) as u16);
                            outcome.context_changed = true;
                        }
                        cpu.rip = ip;
                        return Ok(outcome);
                    }
                    L::Return {
                        size,
                        far,
                        pop_extra,
                    } => {
                        let ip = peek(cpu, mem, context, 0, *size)?;
                        let cs = if *far {
                            Some(peek(cpu, mem, context, 1, *size)? as u16)
                        } else {
                            None
                        };
                        let ip = target(cpu, ip, *size)?;
                        adjust(
                            cpu,
                            context,
                            u32::from(*size) * (if *far { 2 } else { 1 }) + u32::from(*pop_extra),
                        );
                        if let Some(cs) = cs {
                            reload(cpu, SegmentId::Cs, cs);
                            outcome.context_changed = true;
                        }
                        cpu.rip = ip;
                        return Ok(outcome);
                    }
                    L::Loop {
                        kind,
                        count_bits,
                        taken,
                        next,
                    } => {
                        let size = count_bits / 8;
                        let count = cpu.gpr[1] & mask(size);
                        let decrement = !matches!(kind, crate::ir::LoopKind::Jcxz);
                        let count = if decrement {
                            count.wrapping_sub(1) & mask(size)
                        } else {
                            count
                        };
                        let take = match kind {
                            crate::ir::LoopKind::Jcxz => count == 0,
                            crate::ir::LoopKind::Loop => count != 0,
                            crate::ir::LoopKind::Loope => count != 0 && cpu.flags.zf,
                            crate::ir::LoopKind::Loopne => count != 0 && !cpu.flags.zf,
                        };
                        // SDM Vol 2A §3.3 LOOP: an untaken loop completes its
                        // decrement before the next sequential fetch can fault.
                        let ip = if take {
                            target(cpu, u64::from(*taken), 4)?
                        } else {
                            u64::from(*next)
                        };
                        if decrement {
                            cpu.write_gpr(1, count, size);
                        }
                        cpu.rip = ip;
                        return Ok(outcome);
                    }
                    L::String {
                        kind,
                        elem,
                        rep,
                        address_bits,
                        source,
                    } => {
                        // SDM Vol 2B §4.3 MOVS/CMPS/STOS/LODS/SCAS/REP:
                        // address size selects indices and count; operand size selects
                        // element width. ES destination cannot be overridden.
                        let width = address_bits / 8;
                        let repeating = !matches!(rep, RepKind::None);
                        let count = cpu.gpr[1] & mask(width);
                        if !repeating || count != 0 {
                            let si = cpu.gpr[6] & mask(width);
                            let di = cpu.gpr[7] & mask(width);
                            let src = if matches!(kind, StrOp::Movs | StrOp::Cmps | StrOp::Lods) {
                                let a = legacy::access(cpu, *source, si, *elem)?;
                                Some(read(cpu, mem, a, *elem)?)
                            } else {
                                None
                            };
                            let dst = if matches!(
                                kind,
                                StrOp::Movs | StrOp::Stos | StrOp::Cmps | StrOp::Scas
                            ) {
                                Some(legacy::access(cpu, SegmentId::Es, di, *elem)?)
                            } else {
                                None
                            };
                            match kind {
                                StrOp::Movs | StrOp::Stos => {
                                    write(cpu, mem, dst.unwrap(), *elem, src.unwrap_or(cpu.gpr[0]))?
                                }
                                StrOp::Lods => cpu.write_gpr(0, src.unwrap(), *elem),
                                StrOp::Cmps | StrOp::Scas => {
                                    let rhs = read(cpu, mem, dst.unwrap(), *elem)?;
                                    let lhs = src.unwrap_or(cpu.gpr[0]);
                                    apply(
                                        &mut cpu.flags,
                                        FlagMask::ALL,
                                        &alu_sub(lhs, rhs, 0, *elem),
                                    );
                                }
                                _ => unreachable!(),
                            }
                            let delta = if cpu.flags.df {
                                0u64.wrapping_sub(u64::from(*elem))
                            } else {
                                u64::from(*elem)
                            };
                            if src.is_some() {
                                cpu.write_gpr(6, si.wrapping_add(delta), width);
                            }
                            if dst.is_some() {
                                cpu.write_gpr(7, di.wrapping_add(delta), width);
                            }
                            if repeating {
                                let count = count.wrapping_sub(1) & mask(width);
                                cpu.write_gpr(1, count, width);
                                let more = count != 0
                                    && match rep {
                                        RepKind::Repe => cpu.flags.zf,
                                        RepKind::Repne => !cpu.flags.zf,
                                        _ => true,
                                    };
                                if more {
                                    outcome.retired = false;
                                    cpu.rip = ir.guest_start;
                                    return Ok(outcome);
                                }
                            }
                        }
                    }
                    L::Leave { size } => {
                        let offset = cpu.gpr[5] & mask(context.stack_address_bits / 8);
                        let a = legacy::access(cpu, SegmentId::Ss, offset, *size)?;
                        let value = read(cpu, mem, a, *size)?;
                        cpu.write_gpr(
                            4,
                            offset.wrapping_add(u64::from(*size)),
                            context.stack_address_bits / 8,
                        );
                        cpu.write_gpr(5, value, *size);
                    }
                    L::CheckTaskReturn => {
                        // SDM Vol 3A §10.3; decision-18 §2: NT storage
                        // does not license a task return or an ordinary IRET.
                        if cpu.rflags_system & 0x4000 != 0 {
                            return Err(refusal(
                                cpu,
                                crate::UnsupportedSystemOperation::LegacyTaskReturn,
                            ));
                        }
                    }
                    L::Refuse { operation } => return Err(refusal(cpu, *operation)),
                }
                None
            }
            _ => unreachable!("legacy IR preflight"),
        };
        if let Some(InterpResult::Exit(exit)) = result {
            return Err(exit.into_exit());
        }
    }
    cpu.rip = next;
    Ok(outcome)
}
