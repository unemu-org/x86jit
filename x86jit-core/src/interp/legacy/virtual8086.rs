//! Stack transfers for internal V86 execution (design v86 §§4.2–4.3).
use super::*;

/// A frame read whose value can select a different return personality.
/// The paged planner obtains these inputs before planning the reached path.
#[cold]
#[inline(never)]
pub(super) fn peek_frame<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    context: MachineContext,
    index: u64,
    size: u8,
) -> Result<u64, Exit> {
    let offset = sp(cpu, context).wrapping_add(index * u64::from(size))
        & mask(context.stack_address_bits / 8);
    let addr = segment_access(cpu, context, SegmentId::Ss, offset, size, AccessKind::Read)?;
    let result = if mem.paged() {
        paging::read_frame(cpu, mem, addr, size)
    } else {
        read(cpu, mem, addr, size)
    };
    result.map_err(|exit| {
        operand_fault(
            exit,
            cpu.rip,
            addr,
            crate::ir::MemoryAccess::new(u16::from(size), false, true),
        )
    })
}

/// SDM Vol 2A §3.3 IRET pp. 3-492–493; Vol 3B §23.2.5 pp. 23-8–9.
/// The complete nine-slot image is read before any destination is installed.
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(super) fn enter<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    context: MachineContext,
    ip: u64,
    cs: u16,
    image: u32,
    task_switch: &tasks::SwitchState,
) -> Result<(), Exit> {
    #[cfg(not(feature = "test-hooks"))]
    let allowed = false;
    #[cfg(feature = "test-hooks")]
    let allowed = task_switch.allow_virtual8086;
    let _ = task_switch;
    if !allowed {
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::InvalidLegacyContext(
                legacy::MachineContextError::Virtual8086Unsupported,
            ),
        ));
    }
    pop_frame_limit(cpu, mem, context, 9, 4)?;
    let esp = peek_frame(cpu, mem, context, 3, 4)?;
    let ss = peek_frame(cpu, mem, context, 4, 4)? as u16;
    let es = peek_frame(cpu, mem, context, 5, 4)? as u16;
    let ds = peek_frame(cpu, mem, context, 6, 4)? as u16;
    let fs = peek_frame(cpu, mem, context, 7, 4)? as u16;
    let gs = peek_frame(cpu, mem, context, 8, 4)? as u16;
    if ip > 0xffff {
        return Err(fault(cpu, 13, 0));
    }
    legacy::restore_flags(cpu, context.generation, image)?;
    cpu.rflags_system = (cpu.rflags_system & !0x30000) | u64::from(image & 0x30000);
    for (segment, selector) in [
        (SegmentId::Cs, cs),
        (SegmentId::Ss, ss),
        (SegmentId::Es, es),
        (SegmentId::Ds, ds),
        (SegmentId::Fs, fs),
        (SegmentId::Gs, gs),
    ] {
        reload_virtual8086(cpu, segment, selector);
    }
    cpu.gpr[RSP] = esp;
    cpu.legacy.cpl = 3;
    cpu.legacy.reset_fetch = 0;
    cpu.rip = ip;
    Ok(())
}

/// SDM Vol 2A §3.3 IRET pp. 3-491–492: this return remains in V86,
/// independently of NT, preserving VM/IOPL and restoring RF only for IRETD.
#[cold]
#[inline(never)]
pub(super) fn iret<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    context: MachineContext,
    size: u8,
) -> Result<(), Exit> {
    pop_frame_limit(cpu, mem, context, 3, size)?;
    let ip = peek_frame(cpu, mem, context, 0, size)?;
    let cs = peek_frame(cpu, mem, context, 1, size)? as u16;
    let image = peek_frame(cpu, mem, context, 2, size)? as u32;
    if ip > 0xffff {
        return Err(fault(cpu, 13, 0));
    }
    let image = if size == 2 {
        image | (cpu.rflags_system as u32 & 0xffff0000)
    } else {
        image
    };
    privilege::load_flags(cpu, context.generation, image)?;
    if size == 4 {
        cpu.rflags_system = (cpu.rflags_system & !(1 << 16)) | u64::from(image & (1 << 16));
    }
    adjust(cpu, context, 3 * u32::from(size));
    reload_virtual8086(cpu, SegmentId::Cs, cs);
    cpu.rip = ip;
    Ok(())
}

#[cold]
#[inline(never)]
#[cfg(feature = "test-hooks")]
pub(crate) fn execute_block<const ROUTED: bool, const ORDER: u8>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    memory: &Memory,
    scratch: &mut impl ScratchBuffer,
    task_switch: &mut tasks::SwitchState,
) -> Progress {
    let context = ir.legacy_context.expect("V86 context");
    let outcome = execute_instruction::<ROUTED, ORDER, false, true>(
        ir,
        &InstructionView {
            guest_start: ir.guest_start,
            guest_len: ir.guest_len,
            ops: &ir.ops,
        },
        cpu,
        &LegacyAccess::new(memory, legacy::physical_mask(context.generation)),
        scratch.zeroed(ir.temp_count as usize),
        context,
        task_switch,
    );
    Progress {
        sti_shadow: false,
        ss_shadow: 0,
        steps: u64::from(outcome.is_ok()),
        retired: u64::from(outcome.as_ref().is_ok_and(|o| o.retired)),
        outcome: Some(outcome),
    }
}
