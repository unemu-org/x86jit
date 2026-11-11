//! Legacy system interpreter, selected before the op loop (decision-18 §§1, 4;
//! spec.md §§8.1, 17). Shared scalar helpers retain their existing flag semantics.
use super::*;
use crate::legacy::{self, ControlRegister, LegacyDecodeContext, LegacyOp as L, SegmentId};

#[derive(Default)]
pub(crate) struct Outcome {
    pub exit: Option<Exit>,
    pub retired: bool,
    pub context_changed: bool,
    pub translations_invalidated: bool,
    pub arm_sti: bool,
    pub arm_ss: bool,
    pub suppress_debug: bool,
    preserve_rf: bool,
}

fn refusal(cpu: &CpuState, operation: crate::UnsupportedSystemOperation) -> Exit {
    Exit::UnsupportedSystemContext {
        addr: cpu.rip,
        operation,
    }
}

fn exception(cpu: &CpuState, vector: u8) -> Exit {
    fault(cpu, vector, 0)
}

fn fault(cpu: &CpuState, vector: u8, code: u16) -> Exit {
    // SDM Vol 3A §§7.13, 7.15; 80386 PRM §§9.7, 9.10: protected
    // #GP/#SS/#NP carry a code; PRM Fig. 9-7 (p. 9-10, scan checked)
    // leaves bits 31:16 undefined. Real faults have no code (PRM Table 14-1).
    let mut details = crate::fault::FaultMetadata::architectural_exception(
        cpu.rip,
        cpu.rip,
        vector,
        crate::fault::FaultClass::Fault,
        if cpu.legacy.cr0 & 1 != 0 && matches!(vector, 8 | 10..=14) {
            crate::fault::ErrorCode::bits(u32::from(code), 0xffff)
        } else {
            crate::fault::ErrorCode::NoCode
        },
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
    if value & 0x8000_0000 != 0 {
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyExecution,
        ));
    }
    let changed = cpu.legacy.cr0 != value;
    // SDM Vol 3A §12.9.1; 80386 PRM §14.4.1: enter at CPL 0;
    // the six selector/cache pairs remain intact until explicitly reloaded.
    if (value ^ cpu.legacy.cr0) & 1 != 0 {
        cpu.legacy.cpl = 0;
    }
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
    // SDM 325462-092US Vol 3A §§2.2, 12.9.2 and Vol 3B §23.1.1;
    // 80386 PRM §§14.1, 14.5 (pp. 14-1, 14-4, scan checked) describe
    // ordinary 64-KiB real segments and the recommended return sequence.
    // Neither explicitly specifies a later real-mode reload of a nonstandard
    // inherited limit/attributes. The accepted big-real contract retains both
    // and replaces only the base; AMD64 APM Vol 2 §4.1 states this explicitly.
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
    let (physical, part) = legacy::physical_span(cpu, addr, size);
    let low = read_physical(cpu, mem, addr, physical, part)?;
    if part == size {
        return Ok(low);
    }
    let next = u64::from((addr as u32).wrapping_add(u32::from(part)));
    Ok(low | (read(cpu, mem, next, size - part)? << (part * 8)))
}

fn read_physical<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    linear: u64,
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
                StepResult::Exit(exit) => crate::fault::qualify_legacy_bus(exit, linear, addr),
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
    let value = value & mask(size);
    let (physical, part) = legacy::physical_span(cpu, addr, size);
    if part == size {
        return write_physical(cpu, mem, addr, physical, size, value);
    }
    // Validate both destinations before committing the split RAM store.
    probe(cpu, mem, addr, size)?;
    write_physical(cpu, mem, addr, physical, part, value)?;
    let next = u64::from((addr as u32).wrapping_add(u32::from(part)));
    write(cpu, mem, next, size - part, value >> (part * 8))
}

fn write_physical<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    linear: u64,
    addr: u64,
    size: u8,
    value: u64,
) -> Result<(), Exit> {
    // The retry ledger is engine storage. Bit 63 distinguishes answered writes
    // from reads at the same physical bus address; it is never a guest address.
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
                    StepResult::Exit(exit) => crate::fault::qualify_legacy_bus(exit, linear, addr),
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
    let (physical, part) = legacy::physical_span(cpu, addr, size);
    probe_physical(cpu, mem, addr, physical, part)?;
    if part < size {
        let next = u64::from((addr as u32).wrapping_add(u32::from(part)));
        probe(cpu, mem, next, size - part)?;
    }
    Ok(())
}

fn probe_physical<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    linear: u64,
    addr: u64,
    size: u8,
) -> Result<(), Exit> {
    match mem.probe_write(addr, size) {
        Ok(()) | Err(MemTrap::Mmio) => Ok(()),
        Err(trap) => Err(
            match trap_out(cpu, cpu.rip, trap, addr, size, AccessKind::Write, 0).into_step() {
                StepResult::Exit(exit) => crate::fault::qualify_legacy_bus(exit, linear, addr),
                _ => unreachable!(),
            },
        ),
    }
}

struct Descriptor {
    cache: legacy::SegmentCache,
    access_address: Option<u32>,
}

// Retry keys occupy engine-only space above the legacy 32-bit linear address.
// A CALL frame may overwrite its own far pointer or descriptor before a later
// MMIO push hands off. Preserve the inputs, not a second read of that frame.
const DESCRIPTOR_RETRY: u64 = 1 << 62;
const TRANSFER_IP_RETRY: u64 = DESCRIPTOR_RETRY | (1 << 33);
const TRANSFER_CS_RETRY: u64 = TRANSFER_IP_RETRY | 1;

fn descriptor<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    value: u16,
    generation: crate::CpuGeneration,
) -> Result<Descriptor, Exit> {
    // SDM Vol 3A §§3.4.2, 3.4.5; 80386 PRM §§5.1.1–5.1.2,
    // 6.3.1.2: selectors index eight-byte descriptors, including their last byte.
    if value & 4 != 0 {
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyLdtSelector,
        ));
    }
    let offset = u32::from(value & !7);
    if offset + 7 > u32::from(cpu.legacy.gdtr.limit) {
        return Err(fault(cpu, 13, value & !3));
    }
    let address = cpu.legacy.gdtr.base.wrapping_add(offset);
    let key = DESCRIPTOR_RETRY | u64::from(address);
    if let (Some(base_limit), Some(attributes)) = (
        cpu.mmio_part(cpu.rip, key),
        cpu.mmio_part(cpu.rip, key | (1 << 32)),
    ) {
        return Ok(Descriptor {
            cache: legacy::SegmentCache {
                base: base_limit as u32,
                limit: (base_limit >> 32) as u32,
                attributes: attributes as u16,
                ..Default::default()
            },
            access_address: Some(address.wrapping_add(5)),
        });
    }
    let low = read(cpu, mem, u64::from(address), 4)? as u32;
    let high = read(cpu, mem, u64::from(address.wrapping_add(4)), 4)? as u32;
    // I286 PRM §6.3 Fig. 6-3 p. 6-5 (scan checked): bytes 6/7 are
    // reserved; they supply no limit, size, granularity or high base bits.
    let high = if generation == crate::CpuGeneration::I286 {
        high & 0xffff
    } else {
        high
    };
    let limit = (low & 0xffff) | (high & 0x000f_0000);
    let attributes = ((high >> 8) & 0xf0ff) as u16;
    Ok(Descriptor {
        cache: legacy::SegmentCache {
            base: (low >> 16) | ((high & 0xff) << 16) | (high & 0xff00_0000),
            limit: if attributes & 0x8000 != 0 {
                (limit << 12) | 0xfff
            } else {
                limit
            },
            attributes,
            unusable: 0,
            reserved: 0,
        },
        access_address: Some(address.wrapping_add(5)),
    })
}

fn accessed<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    descriptor: &mut Descriptor,
) -> Result<(), Exit> {
    if let Some(address) = descriptor.access_address {
        let key = DESCRIPTOR_RETRY | u64::from(address.wrapping_sub(5));
        cpu.record_mmio_part(
            cpu.rip,
            key,
            u64::from(descriptor.cache.base) | (u64::from(descriptor.cache.limit) << 32),
        );
        cpu.record_mmio_part(
            cpu.rip,
            key | (1 << 32),
            u64::from(descriptor.cache.attributes),
        );
    }
    // SDM Vol 3A §3.4.5.1; 80386 PRM §5.1.1 p. 5-4 (scan checked):
    // successful segment loads set A in memory as well as in the hidden cache.
    // I286 PRM §7.4 p. 7-11 specifies the same accessed-bit update.
    if descriptor.cache.attributes & 1 == 0 {
        if let Some(address) = descriptor.access_address {
            write(
                cpu,
                mem,
                u64::from(address),
                1,
                u64::from(descriptor.cache.attributes as u8 | 1),
            )?;
            descriptor.cache.attributes |= 1;
        }
    }
    Ok(())
}

fn install(cpu: &mut CpuState, segment: SegmentId, value: u16, cache: legacy::SegmentCache) {
    reload(cpu, segment, value);
    cpu.legacy.segments[segment as usize] = cache;
}

fn load_segment<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    segment: SegmentId,
    value: u16,
    generation: crate::CpuGeneration,
) -> Result<(), Exit> {
    if cpu.legacy.cr0 & 1 == 0 {
        reload(cpu, segment, value);
        return Ok(());
    }
    // SDM Vol 2B §4.3 MOV; Vol 3A §§6.3, 6.6, 6.7;
    // 80386 PRM §17 MOV pp. 17-109–110 (scan checked).
    // I286 PRM §7.4.1 p. 7-12; Appendix B MOV pp. B-73–74.
    if value & !3 == 0 {
        if segment == SegmentId::Ss {
            return Err(fault(cpu, 13, 0));
        }
        install(
            cpu,
            segment,
            value,
            legacy::SegmentCache {
                unusable: 1,
                ..Default::default()
            },
        );
        return Ok(());
    }
    let mut d = descriptor(cpu, mem, value, generation)?;
    let attr = d.cache.attributes;
    let code = attr & 8 != 0;
    let dpl = ((attr >> 5) & 3) as u8;
    let rpl = (value & 3) as u8;
    let cpl = cpu.legacy.cpl;
    let valid = attr & 0x10 != 0
        && if segment == SegmentId::Ss {
            !code && attr & 2 != 0 && rpl == cpl && dpl == cpl
        } else {
            (!code || attr & 2 != 0) && ((code && attr & 4 != 0) || dpl >= cpl.max(rpl))
        };
    if !valid {
        return Err(fault(cpu, 13, value & !3));
    }
    if attr & 0x80 == 0 {
        return Err(fault(
            cpu,
            if segment == SegmentId::Ss { 12 } else { 11 },
            value & !3,
        ));
    }
    supported_descriptor(cpu, attr)?;
    accessed(cpu, mem, &mut d)?;
    install(cpu, segment, value, d.cache);
    Ok(())
}

fn code_descriptor<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    value: u16,
    returning: bool,
    generation: crate::CpuGeneration,
) -> Result<Descriptor, Exit> {
    // SDM Vol 3A §6.8; Vol 2A §3.3 CALL/IRET/JMP; Vol 2B §4.3 RET;
    // 80386 PRM §17 pp. 17-39, 17-76–77, 17-85, 17-142–143 (scan checked).
    // I286 PRM §§7.4.2–7.5 pp. 7-13–15; Appendix B IRET pp. B-51–52,
    // RET pp. B-94–95: same-CPL code descriptor checks.
    if value & !3 == 0 {
        return Err(fault(cpu, 13, 0));
    }
    let cpl = cpu.legacy.cpl;
    let rpl = (value & 3) as u8;
    if returning && rpl < cpl {
        return Err(fault(cpu, 13, value & !3));
    }
    if returning && rpl > cpl {
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyPrivilegeTransition,
        ));
    }
    let d = descriptor(cpu, mem, value, generation)?;
    let attr = d.cache.attributes;
    if attr & 0x10 == 0 {
        if !returning && matches!(attr & 0xf, 1 | 3 | 4 | 5 | 9 | 11 | 12) {
            return Err(refusal(
                cpu,
                crate::UnsupportedSystemOperation::LegacyGateOrTaskTransfer,
            ));
        }
        return Err(fault(cpu, 13, value & !3));
    }
    let dpl = ((attr >> 5) & 3) as u8;
    let conforming = attr & 4 != 0;
    if attr & 8 == 0
        || if conforming {
            dpl > cpl
        } else {
            dpl != cpl || rpl > cpl
        }
    {
        return Err(fault(cpu, 13, value & !3));
    }
    if attr & 0x80 == 0 {
        return Err(fault(cpu, 11, value & !3));
    }
    supported_descriptor(cpu, attr)?;
    Ok(d)
}

fn supported_descriptor(cpu: &CpuState, attributes: u16) -> Result<(), Exit> {
    // Decision-18 §1: an unsupported retained size attribute is an engine
    // refusal, after architectural type/privilege/presence checks succeed.
    if attributes & 0x2000 != 0 {
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyEntryFacility,
        ));
    }
    Ok(())
}

fn code_limit(cpu: &CpuState, cache: legacy::SegmentCache, ip: u64) -> Result<(), Exit> {
    if ip > u64::from(cache.limit) {
        return Err(fault(cpu, 13, 0));
    }
    Ok(())
}

// SDM Vol 1 §6.2.3; Vol 3A §3.4.5; Vol 2B §4.3 PUSH:
// transfer width and SS.B-selected pointer width are independent. CALL's
// CS/EIP slots use that same boundary (Vol 2A §3.3 CALL).
fn push_addresses<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    context: LegacyDecodeContext,
    count: usize,
    size: u8,
) -> Result<Vec<u64>, Exit> {
    let old = sp(cpu, context);
    let bytes = count as u8 * size;
    let start = old.wrapping_sub(u64::from(bytes)) & mask(context.stack_address_bits / 8);
    legacy::access(
        cpu,
        context.generation,
        SegmentId::Ss,
        start,
        bytes,
        AccessKind::Write,
    )?;
    let mut addresses = Vec::with_capacity(count);
    for i in 1..=count {
        let offset =
            old.wrapping_sub(i as u64 * u64::from(size)) & mask(context.stack_address_bits / 8);
        let address = legacy::access(
            cpu,
            context.generation,
            SegmentId::Ss,
            offset,
            size,
            AccessKind::Write,
        )?;
        probe(cpu, mem, address, size)?;
        addresses.push(address);
    }
    Ok(addresses)
}

fn push<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    context: LegacyDecodeContext,
    values: &[u64],
    size: u8,
) -> Result<(), Exit> {
    let old = sp(cpu, context);
    let addresses = push_addresses(cpu, mem, context, values.len(), size)?;
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
    let addr = legacy::access(
        cpu,
        context.generation,
        SegmentId::Ss,
        offset,
        size,
        AccessKind::Read,
    )?;
    read(cpu, mem, addr, size)
}
fn target(cpu: &CpuState, context: LegacyDecodeContext, value: u64, size: u8) -> Result<u64, Exit> {
    let offset = value & mask(size);
    legacy::access(
        cpu,
        context.generation,
        SegmentId::Cs,
        offset,
        1,
        AccessKind::Execute,
    )?;
    Ok(offset)
}

/// Shared delivery for software INT, processor exceptions and queued IRQs.
pub(crate) fn deliver_interrupt<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    memory: &Memory,
    context: LegacyDecodeContext,
    event: legacy::LegacyEvent,
) -> Result<(), Exit> {
    let mem = Access::<ROUTED, ORDER>::new(memory);
    deliver(cpu, mem, context, event)
}

fn deliver_protected<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    context: LegacyDecodeContext,
    event: legacy::LegacyEvent,
) -> Result<(), Exit> {
    // I286 PRM §9.4 Fig. 9-3, Table 9-1 pp. 9-4–6;
    // I386 PRM §§9.5–9.7 pp. 9-6–10; §17 INT pp. 17-70–72 (scan checked).
    let offset = u32::from(event.vector) * 8;
    let ext = u16::from(event.external);
    let idt_code = offset as u16 | 2 | ext;
    if offset + 7 > u32::from(cpu.legacy.idtr.limit) {
        return Err(fault(cpu, 13, idt_code));
    }
    let address = cpu.legacy.idtr.base.wrapping_add(offset);
    let key = DESCRIPTOR_RETRY | (1 << 34) | u64::from(address);
    let low = if let Some(value) = cpu.mmio_part(cpu.rip, key) {
        value as u32
    } else {
        read(cpu, mem, u64::from(address), 4)? as u32
    };
    let high = if let Some(value) = cpu.mmio_part(cpu.rip, key | (1 << 32)) {
        value as u32
    } else {
        read(cpu, mem, u64::from(address.wrapping_add(4)), 4)? as u32
    };
    let attr = (high >> 8) as u8;
    let kind = attr & 0x1f;
    let valid = matches!(kind, 5..=7)
        || (context.generation == crate::CpuGeneration::I386 && matches!(kind, 14 | 15));
    if !valid || (event.software && (attr >> 5) & 3 < context.cpl) {
        return Err(fault(cpu, 13, idt_code));
    }
    if attr & 0x80 == 0 {
        return Err(fault(cpu, 11, idt_code));
    }
    if kind == 5 {
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyGateOrTaskTransfer,
        ));
    }
    let cs = (low >> 16) as u16;
    let selector_code = (cs & !3) | ext;
    if cs & !3 == 0 {
        return Err(fault(cpu, 13, ext));
    }
    let mut d = descriptor(cpu, mem, cs, context.generation).map_err(|exit| {
        // Table bounds failures name the selector with the delivery's EXT bit.
        if matches!(exit, Exit::Exception { vector: 13, .. }) {
            fault(cpu, 13, selector_code)
        } else {
            exit
        }
    })?;
    let dpl = ((d.cache.attributes >> 5) & 3) as u8;
    if d.cache.attributes & 0x18 != 0x18 || dpl > context.cpl {
        return Err(fault(cpu, 13, selector_code));
    }
    if d.cache.attributes & 0x80 == 0 {
        return Err(fault(cpu, 11, selector_code));
    }
    if d.cache.attributes & 4 == 0 && dpl < context.cpl {
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyPrivilegeTransition,
        ));
    }
    supported_descriptor(cpu, d.cache.attributes)?;
    let size = if kind & 8 != 0 { 4 } else { 2 };
    let ip = u64::from((low & 0xffff) | if size == 4 { high & 0xffff0000 } else { 0 });
    // I386 PRM §17 INT p. 17-72 checks stack space before the target offset,
    // reporting #SS(0)/#GP(0). Follow SDM Vol 2A §3.3 INT n/INTO/INT3/INT1,
    // Operation, INTRA-PRIVILEGE-LEVEL-INTERRUPT p. 3-478 for the error codes:
    // #SS(EXT) for insufficient stack space, #GP(EXT) for an invalid target offset.
    push_addresses(
        cpu,
        mem,
        context,
        3 + usize::from(event.error_code.is_some()),
        size,
    )
    .map_err(|exit| {
        if matches!(exit, Exit::Exception { vector: 12, .. }) {
            fault(cpu, 12, ext)
        } else {
            exit
        }
    })?;
    if ip > u64::from(d.cache.limit) {
        return Err(fault(cpu, 13, ext));
    }
    cpu.record_mmio_part(cpu.rip, key, u64::from(low));
    cpu.record_mmio_part(cpu.rip, key | (1 << 32), u64::from(high));
    accessed(cpu, mem, &mut d)?;
    let mut flags = legacy::flags_image(cpu, context.generation);
    // I386 PRM §12.3.1.1 p. 12-7: fault frames set RF; traps and INT do not.
    if context.generation == crate::CpuGeneration::I386 && event.fault {
        flags |= 1 << 16;
    }
    let mut frame = vec![
        u64::from(flags),
        u64::from(cpu.cs),
        u64::from(event.saved_ip),
    ];
    if let Some(code) = event.error_code {
        frame.push(u64::from(code));
    }
    push(cpu, mem, context, &frame, size)?;
    if kind & 1 == 0 {
        cpu.flags.if_ = false;
    }
    // I286 PRM §9.4 pp. 9-3–5; I386 PRM §17 INT p. 17-72:
    // both gate kinds clear TF/NT after saving their old values.
    // SDM Vol 3A §7.12.1.3 also clears RF in the live handler flags.
    cpu.rflags_system &= !0x14100;
    install(
        cpu,
        SegmentId::Cs,
        (cs & !3) | u16::from(context.cpl),
        d.cache,
    );
    cpu.rip = ip;
    Ok(())
}

fn deliver<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    context: LegacyDecodeContext,
    event: legacy::LegacyEvent,
) -> Result<(), Exit> {
    if context.regime == legacy::LegacyRegime::Protected {
        return deliver_protected(cpu, mem, context, event);
    }
    let vector = event.vector;
    let saved_ip = event.saved_ip;
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

// x87 retains its existing RAM-only transfer contract. Segment checks happen in
// Legacy::Address; this adapter splits each actual RAM transfer at the A20 gate.
struct PhysicalFpMemory<'a, const ROUTED: bool> {
    memory: &'a Memory,
    a20_enabled: bool,
    fault: std::cell::Cell<Option<u64>>,
}
impl<const ROUTED: bool> crate::x87::FpMem for PhysicalFpMemory<'_, ROUTED> {
    fn load(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.load_checked(addr, buf)
            .map_err(|at| self.fault.set(Some(at)))
            .is_ok()
    }
    fn store(&self, addr: u64, bytes: &[u8]) -> bool {
        self.store_checked(addr, bytes)
            .map_err(|at| self.fault.set(Some(at)))
            .is_ok()
    }
    fn fault_addr(&self, addr: u64) -> u64 {
        self.fault.get().unwrap_or(addr)
    }
    fn host_fault_recovery(&self) -> bool {
        ROUTED
    }
    fn load_checked(&self, mut addr: u64, mut buf: &mut [u8]) -> Result<(), u64> {
        while !buf.is_empty() {
            let (physical, part) =
                legacy::physical_span_with_gate(self.a20_enabled, addr, buf.len().min(255) as u8);
            let (head, tail) = buf.split_at_mut(usize::from(part));
            self.memory
                .read_ram_checked_mode::<ROUTED>(physical, head)
                .map_err(|at| addr + at.saturating_sub(physical))?;
            addr = u64::from((addr as u32).wrapping_add(u32::from(part)));
            buf = tail;
        }
        Ok(())
    }
    fn store_checked(&self, mut addr: u64, mut bytes: &[u8]) -> Result<(), u64> {
        // Check every physical destination before the first RAM commit.
        let mut offset = 0;
        while offset < bytes.len() {
            let linear = u64::from((addr as u32).wrapping_add(offset as u32));
            let (physical, part) = legacy::physical_span_with_gate(
                self.a20_enabled,
                linear,
                (bytes.len() - offset).min(255) as u8,
            );
            if let Err(trap) = self.memory.probe_write_mode::<ROUTED>(physical, part) {
                let at = trap.fault_addr(physical);
                crate::fault::record_fp_fault(
                    at,
                    usize::from(part),
                    if matches!(trap, MemTrap::HostFault) {
                        crate::FaultOrigin::HostAccess
                    } else {
                        crate::FaultOrigin::GuestMap
                    },
                    crate::AddressPrecision::TransferAddress,
                );
                return Err(linear + at.saturating_sub(physical));
            }
            offset += usize::from(part);
        }
        while !bytes.is_empty() {
            let (physical, part) =
                legacy::physical_span_with_gate(self.a20_enabled, addr, bytes.len().min(255) as u8);
            let (head, tail) = bytes.split_at(usize::from(part));
            self.memory
                .write_ram_checked_mode::<ROUTED>(physical, head)
                .map_err(|at| addr + at.saturating_sub(physical))?;
            addr = u64::from((addr as u32).wrapping_add(u32::from(part)));
            bytes = tail;
        }
        Ok(())
    }
}

pub(crate) fn finish_port_input<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    memory: &Memory,
) -> Result<(), Exit> {
    let Some(p) = cpu.port_string_in else {
        return Ok(());
    };
    let Some(value) = p.value else {
        return Err(Exit::port_io(cpu.gpr[RDX] as u16, p.elem, PortDir::In, 0));
    };
    // The handoff already checked the complete ES operand and owns its A20 line.
    write(
        cpu,
        Access::<ROUTED, ORDER>::new(memory),
        u64::from((p.base as u32).wrapping_add(p.offset as u32)),
        p.elem,
        value,
    )?;
    cpu.write_gpr(RDI, p.next_index, p.addr_bits / 8);
    if p.rep {
        cpu.write_gpr(RCX, p.next_count, p.addr_bits / 8);
    }
    cpu.rip = p.next_ip;
    cpu.port_string_in = None;
    cpu.clear_mmio_parts();
    Ok(())
}

fn atomic_device_exit<const ROUTED: bool, const ORDER: u8>(
    cpu: &CpuState,
    mem: &Access<ROUTED, ORDER>,
    ir: &IrBlock,
) -> Exit {
    // Locked device operands require whole-instruction embedder emulation,
    // including an operand split by the board's physical gate (§5.2).
    let mut bytes = [0; 15];
    for i in 0..ir.guest_len as usize {
        let linear = u64::from(
            cpu.legacy.segments[SegmentId::Cs as usize]
                .base
                .wrapping_add((cpu.rip as u32).wrapping_add(i as u32)),
        );
        let (physical, _) = legacy::physical_span(cpu, linear, 1);
        if mem.fetch_code(physical, &mut bytes[i..i + 1]).is_err() {
            break;
        }
    }
    Exit::unknown_instruction(cpu.rip, bytes, ir.guest_len as u8)
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
            | IrOp::Bcd { .. }
            | IrOp::Bt { .. }
            | IrOp::BoundGate { .. }
            | IrOp::X87 { .. }
            | IrOp::AtomicRmw { .. }
            | IrOp::Unsupported { .. }
    )
}

/// One completed instruction or one REP iteration; incomplete work costs zero.
pub(crate) fn execute<const ROUTED: bool, const ORDER: u8>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    memory: &Memory,
    scratch: &mut impl ScratchBuffer,
) -> Result<Outcome, Exit> {
    #[cfg(debug_assertions)]
    debug_assert!(
        ir.verify_temp_defs().is_ok(),
        "undefined interpreter temporary: {:?}",
        ir.verify_temp_defs()
    );
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
    } else if result
        .as_ref()
        .is_ok_and(|outcome| outcome.retired && !outcome.preserve_rf)
    {
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
            IrOp::Bcd { kind } => exec_bcd(cpu, cpu.rip, kind),
            IrOp::Bt {
                result,
                a,
                bit,
                size,
                op,
            } => exec_bt(cpu, temps, result, a, bit, size, op),
            IrOp::BoundGate {
                index,
                lower,
                upper,
                ..
            } => {
                let index = read_val(*index, temps) as i16;
                if index < read_val(*lower, temps) as i16 || index > read_val(*upper, temps) as i16
                {
                    return Err(exception(cpu, 5));
                }
                None
            }
            IrOp::AtomicRmw {
                old,
                addr,
                src,
                size,
                op,
                ..
            } => {
                let linear = read_val(*addr, temps);
                let (physical, part) = legacy::physical_span(cpu, linear, *size);
                if part == *size {
                    match mem.atomic_rmw(physical, read_val(*src, temps), *size, *op) {
                        Ok(value) => temps[*old as usize] = value,
                        Err(MemTrap::Mmio) => return Err(atomic_device_exit(cpu, mem, ir)),
                        Err(trap) => {
                            if let StepResult::Exit(exit) = trap_out(
                                cpu,
                                cpu.rip,
                                trap,
                                physical,
                                *size,
                                AccessKind::Write,
                                read_val(*src, temps),
                            )
                            .into_step()
                            {
                                return Err(crate::fault::qualify_legacy_bus(
                                    exit, linear, physical,
                                ));
                            }
                        }
                    }
                } else {
                    if (0..*size).any(|byte| {
                        let linear = u64::from((linear as u32).wrapping_add(u32::from(byte)));
                        let (physical, _) = legacy::physical_span(cpu, linear, 1);
                        matches!(mem.0.region_kind(physical, 1), Ok(crate::RegionKind::Trap))
                    }) {
                        return Err(atomic_device_exit(cpu, mem, ir));
                    }
                    // As with the memory layer's unaligned RMW fallback, this
                    // discontinuous bus operand has no single host atomic address.
                    probe(cpu, mem, linear, *size)?;
                    let value = read(cpu, mem, linear, *size)?;
                    let src = read_val(*src, temps);
                    let updated = match op {
                        crate::ir::RmwOp::Add => value.wrapping_add(src),
                        crate::ir::RmwOp::Sub => value.wrapping_sub(src),
                        crate::ir::RmwOp::Rsub => src.wrapping_sub(value),
                        crate::ir::RmwOp::And => value & src,
                        crate::ir::RmwOp::Or => value | src,
                        crate::ir::RmwOp::Xor => value ^ src,
                        crate::ir::RmwOp::Xchg => src,
                    };
                    write(cpu, mem, linear, *size, updated)?;
                    temps[*old as usize] = value;
                }
                None
            }
            IrOp::Unsupported { bytes, len } => {
                if cpu.fpu_model != crate::FpuModel::None {
                    return Err(Exit::unknown_instruction(cpu.rip, *bytes, *len));
                }
                None
            }
            IrOp::X87 {
                kind,
                addr,
                sti,
                instruction,
                operand_offset,
                ..
            } => {
                if cpu.fpu_model == crate::FpuModel::None {
                    continue;
                }
                if crate::x87::mf_pending_before(cpu, *kind) {
                    return Err(exception(cpu, 16));
                }
                let address = read_val(*addr, temps);
                let fp = PhysicalFpMemory::<ROUTED> {
                    memory: &mem.0,
                    a20_enabled: cpu.legacy.a20_enabled != 0,
                    fault: std::cell::Cell::new(None),
                };
                let _ordering = ordering::Barrier::<ORDER>::new(
                    if crate::x87::mem_write_bytes(*kind).is_some() {
                        Some(ordering::Direction::Write)
                    } else if crate::x87::mem_operand_bytes(*kind) != 0 {
                        Some(ordering::Direction::Read)
                    } else {
                        None
                    },
                    false,
                );
                if let Some((at, writing)) = crate::x87::exec_x87(cpu, &fp, *kind, address, *sti) {
                    let (physical, _) = legacy::physical_span(cpu, at, 1);
                    let exit = crate::fault::fp_exit(
                        cpu.rip,
                        physical,
                        crate::x87::mem_operand_bytes_format(*kind, *sti) as u16,
                        if writing {
                            AccessKind::Write
                        } else {
                            AccessKind::Read
                        },
                    );
                    return Err(crate::fault::qualify_legacy_bus(exit, at, physical));
                }
                crate::x87::record_x87_instruction(
                    cpu,
                    read_val(*operand_offset, temps),
                    ir.guest_start,
                    *instruction,
                );
                None
            }
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
                cpu.rip = target(cpu, context, read_val(*ip, temps), 4)?;
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
                    target(cpu, context, *taken, 4)?
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
                    L::Coprocessor { bytes, len, wait } => {
                        // SDM Vol 3A §2.5 Table 2-2; 80386 PRM §§11.1.2–11.1.4:
                        // ESC tests EM/TS before operand checks; WAIT tests MP && TS.
                        let msw = cpu.legacy.cr0;
                        if if *wait {
                            msw & 0xa == 0xa
                        } else {
                            msw & 0xc != 0
                        } {
                            return Err(exception(cpu, 7));
                        }
                        // Generation/decoder #UD follows the MSW gate, before
                        // the absent-output handoff (decision-21; §17.9).
                        if !*wait
                            && cpu.fpu_model == crate::FpuModel::None
                            && !ir
                                .ops
                                .iter()
                                .any(|op| matches!(op, IrOp::Trap { vector: 6, .. }))
                            && crate::x87::esc_output(&bytes[..usize::from(*len)])
                        {
                            return Err(Exit::unknown_instruction(cpu.rip, *bytes, *len));
                        }
                    }
                    L::Enter { size, alloc, level } => {
                        // SDM Vol 2A §3.3 ENTER; Vol 1 §6.2.3: operand size
                        // selects frame values independently of stack address size.
                        let stack_mask = mask(context.stack_address_bits / 8);
                        let frame_sp = sp(cpu, context).wrapping_sub(u64::from(*size)) & stack_mask;
                        let frame = if *size == 4 && context.stack_address_bits == 16 {
                            (cpu.gpr[RSP] & 0xffff_0000) | frame_sp
                        } else {
                            frame_sp & mask(*size)
                        };
                        let pushes = if *level == 0 {
                            1
                        } else {
                            usize::from(*level) + 1
                        };
                        let final_sp = sp(cpu, context)
                            .wrapping_sub(pushes as u64 * u64::from(*size))
                            .wrapping_sub(u64::from(*alloc))
                            & stack_mask;
                        legacy::access(
                            cpu,
                            context.generation,
                            SegmentId::Ss,
                            final_sp,
                            1,
                            AccessKind::Write,
                        )
                        .map_err(|_| exception(cpu, 12))?;
                        let addresses = push_addresses(cpu, mem, context, pushes, *size)?;
                        let mut values = vec![cpu.gpr[RBP] & mask(*size)];
                        let mut overlay = Vec::new();
                        for i in 1..*level {
                            // Display reads see earlier frame pushes, even if
                            // BP aliases SP or the A20 gate aliases their backing.
                            // Stage those bytes until every faultable read succeeds.
                            let value = values[usize::from(i - 1)];
                            let pushed = addresses[usize::from(i - 1)];
                            for byte in 0..*size {
                                let linear =
                                    u64::from((pushed as u32).wrapping_add(u32::from(byte)));
                                let (physical, _) = legacy::physical_span(cpu, linear, 1);
                                overlay.push((physical, (value >> (byte * 8)) as u8));
                            }
                            let offset = cpu.gpr[RBP].wrapping_sub(u64::from(i) * u64::from(*size))
                                & stack_mask;
                            let address = legacy::access(
                                cpu,
                                context.generation,
                                SegmentId::Ss,
                                offset,
                                *size,
                                AccessKind::Read,
                            )?;
                            let physical = (0..*size)
                                .map(|byte| {
                                    legacy::physical_span(
                                        cpu,
                                        u64::from((address as u32).wrapping_add(u32::from(byte))),
                                        1,
                                    )
                                    .0
                                })
                                .collect::<Vec<_>>();
                            let value = if physical
                                .iter()
                                .any(|at| overlay.iter().any(|(p, _)| p == at))
                            {
                                let mut value = 0;
                                for (byte, at) in physical.into_iter().enumerate() {
                                    let part = if let Some((_, value)) =
                                        overlay.iter().rev().find(|(p, _)| *p == at)
                                    {
                                        u64::from(*value)
                                    } else {
                                        read(
                                            cpu,
                                            mem,
                                            u64::from((address as u32).wrapping_add(byte as u32)),
                                            1,
                                        )?
                                    };
                                    value |= part << (byte * 8);
                                }
                                value
                            } else {
                                read(cpu, mem, address, *size)?
                            };
                            values.push(value);
                        }
                        if *level != 0 {
                            values.push(frame);
                        }
                        push(cpu, mem, context, &values, *size)?;
                        cpu.write_gpr(RBP, frame, *size);
                        adjust(cpu, context, 0u32.wrapping_sub(u32::from(*alloc)));
                    }
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
                            // I286 PRM Appendix B LGDT/LIDT p. B-65: 16+24,
                            // with the last byte of the six-byte input ignored.
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
                            // I286 PRM Appendix B SGDT/SIDT p. B-101 leaves
                            // byte 5 undefined. SDM Vol 2B §4.3 SGDT/SIDT,
                            // IA-32 Architecture Compatibility specifies 286 ones.
                            let base = if context.generation == crate::CpuGeneration::I286 {
                                (table.base & 0x00ff_ffff) | 0xff00_0000
                            } else if *size == 2 {
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
                            // I286 PRM §10.2.2 p. 10-4; §10.4 p. 10-6:
                            // MSW has four controls. Keep the reset high image of
                            // ones as the engine's fixed reserved-bit policy.
                            ControlRegister::Cr0
                                if context.generation == crate::CpuGeneration::I286 =>
                            {
                                (cpu.legacy.cr0 & 0xf) | 0xfff0
                            }
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
                                outcome.translations_invalidated = true;
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
                    L::Bound {
                        index,
                        lower,
                        upper,
                        size,
                    } => {
                        let signed = |value| sign_extend(value, *size) as i64;
                        let index = signed(read_val(*index, temps));
                        if index < signed(read_val(*lower, temps))
                            || index > signed(read_val(*upper, temps))
                        {
                            return Err(exception(cpu, 5));
                        }
                    }
                    L::Interrupt {
                        vector,
                        saved_ip,
                        overflow_only,
                    } => {
                        if !overflow_only || cpu.flags.of {
                            deliver(
                                cpu,
                                mem,
                                context,
                                legacy::LegacyEvent {
                                    vector: *vector,
                                    saved_ip: *saved_ip,
                                    software: true,
                                    external: false,
                                    error_code: None,
                                    fault: false,
                                },
                            )?;
                            // I286 PRM §9.7.1 p. 9-14, Appendix B INT/INTO
                            // p. B-48; 80386 PRM §12.3.1.4 p. 12-8,
                            // §17 INT/INTO p. 17-69: both models save then
                            // clear TF and stop stepping for taken
                            // INT n/INT 3/INTO. No #DB precedes or follows the
                            // handler's first instruction; IRET restores TF.
                            // Prefer these instruction/debug rules over the
                            // conflicting handler-stepping summaries in I286
                            // Appendix C item 12, I386 §14.7 item 10, and the
                            // final paragraphs of those single-step sections.
                            outcome.suppress_debug = true;
                            outcome.context_changed = true;
                            return Ok(outcome);
                        }
                    }
                    L::Address {
                        dst,
                        segment,
                        offset,
                        bytes,
                        access,
                    } => {
                        temps[*dst as usize] = legacy::access(
                            cpu,
                            context.generation,
                            *segment,
                            read_val(*offset, temps),
                            *bytes,
                            *access,
                        )?
                    }
                    L::ReadSelector { dst, segment } => {
                        temps[*dst as usize] = u64::from(selector(cpu, *segment))
                    }
                    L::LoadSegment {
                        segment,
                        value,
                        inhibit,
                    } => {
                        load_segment(
                            cpu,
                            mem,
                            *segment,
                            read_val(*value, temps) as u16,
                            context.generation,
                        )?;
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
                            if context.regime == legacy::LegacyRegime::Real
                                && matches!(sp(cpu, context), 7 | 9 | 11 | 13 | 15)
                            {
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
                        let ip = cpu
                            .mmio_part(cpu.rip, TRANSFER_IP_RETRY)
                            .unwrap_or_else(|| read_val(*ip, temps))
                            & mask(*size);
                        let cs = cs.map(|cs| {
                            cpu.mmio_part(cpu.rip, TRANSFER_CS_RETRY)
                                .unwrap_or_else(|| read_val(cs, temps))
                                as u16
                        });
                        let mut descriptor = if context.regime == legacy::LegacyRegime::Protected {
                            cs.map(|cs| code_descriptor(cpu, mem, cs, false, context.generation))
                                .transpose()?
                        } else {
                            None
                        };
                        if let Some(d) = &descriptor {
                            if return_ip.is_some() {
                                push_addresses(cpu, mem, context, 2, *size)?;
                            }
                            code_limit(cpu, d.cache, ip)?;
                        } else {
                            target(cpu, context, ip, *size)?;
                        }
                        if let Some(d) = &mut descriptor {
                            if return_ip.is_some() {
                                cpu.record_mmio_part(cpu.rip, TRANSFER_IP_RETRY, ip);
                                cpu.record_mmio_part(
                                    cpu.rip,
                                    TRANSFER_CS_RETRY,
                                    u64::from(cs.unwrap()),
                                );
                            }
                            accessed(cpu, mem, d)?;
                        }
                        if let Some(return_ip) = return_ip {
                            let mut frame = Vec::new();
                            if cs.is_some() {
                                frame.push(u64::from(cpu.cs));
                            }
                            frame.push(u64::from(*return_ip));
                            push(cpu, mem, context, &frame, *size)?;
                        }
                        if let Some(cs) = cs {
                            if let Some(d) = descriptor {
                                install(
                                    cpu,
                                    SegmentId::Cs,
                                    (cs & !3) | u16::from(context.cpl),
                                    d.cache,
                                );
                            } else {
                                reload(cpu, SegmentId::Cs, cs);
                            }
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
                        legacy::access(
                            cpu,
                            context.generation,
                            SegmentId::Ss,
                            sp(cpu, context),
                            *size * if *far { 2 } else { 1 },
                            AccessKind::Read,
                        )?;
                        let ip = peek(cpu, mem, context, 0, *size)?;
                        let cs = if *far {
                            Some(peek(cpu, mem, context, 1, *size)? as u16)
                        } else {
                            None
                        };
                        let ip = ip & mask(*size);
                        let mut descriptor = if context.regime == legacy::LegacyRegime::Protected {
                            cs.map(|cs| code_descriptor(cpu, mem, cs, true, context.generation))
                                .transpose()?
                        } else {
                            None
                        };
                        if let Some(d) = &mut descriptor {
                            code_limit(cpu, d.cache, ip)?;
                            accessed(cpu, mem, d)?;
                        } else {
                            target(cpu, context, ip, *size)?;
                        }
                        adjust(
                            cpu,
                            context,
                            u32::from(*size) * (if *far { 2 } else { 1 }) + u32::from(*pop_extra),
                        );
                        if let Some(cs) = cs {
                            if let Some(d) = descriptor {
                                install(cpu, SegmentId::Cs, cs, d.cache);
                            } else {
                                reload(cpu, SegmentId::Cs, cs);
                            }
                            outcome.context_changed = true;
                        }
                        cpu.rip = ip;
                        return Ok(outcome);
                    }
                    L::Iret { size } => {
                        // SDM Vol 2A §3.3 IRET; 80386 PRM §17 pp. 17-74,
                        // 17-76–78 (scan checked): same-CPL frame, independent SS.B.
                        legacy::access(
                            cpu,
                            context.generation,
                            SegmentId::Ss,
                            sp(cpu, context),
                            3 * *size,
                            AccessKind::Read,
                        )?;
                        let ip = peek(cpu, mem, context, 0, *size)? & mask(*size);
                        let cs = peek(cpu, mem, context, 1, *size)? as u16;
                        let image = peek(cpu, mem, context, 2, *size)? as u32;
                        // SDM Vol 2A §3.3, IRET/IRETD/IRETQ, Operation,
                        // REAL-ADDRESS-MODE: (tempEFLAGS AND 257FD5H) OR
                        // (EFLAGS AND 1A0000H) preserves VM with PE clear.
                        if context.regime == legacy::LegacyRegime::Protected
                            && *size == 4
                            && image & (1 << 17) != 0
                        {
                            return Err(refusal(
                                cpu,
                                crate::UnsupportedSystemOperation::InvalidLegacyContext(
                                    legacy::LegacyContextError::Virtual8086Unsupported,
                                ),
                            ));
                        }
                        let mut descriptor = if context.regime == legacy::LegacyRegime::Protected {
                            Some(code_descriptor(cpu, mem, cs, true, context.generation)?)
                        } else {
                            None
                        };
                        code_limit(
                            cpu,
                            descriptor
                                .as_ref()
                                .map_or(cpu.legacy.segments[0], |d| d.cache),
                            ip,
                        )?;
                        if let Some(d) = &mut descriptor {
                            accessed(cpu, mem, d)?;
                        }
                        legacy::restore_flags(cpu, context.generation, image)?;
                        // IRET restores RF in a 32-bit frame; POPFD does not.
                        if *size == 4 {
                            cpu.rflags_system =
                                (cpu.rflags_system & !(1 << 16)) | u64::from(image & (1 << 16));
                        }
                        adjust(cpu, context, 3 * u32::from(*size));
                        if let Some(d) = descriptor {
                            install(cpu, SegmentId::Cs, cs, d.cache);
                        } else {
                            reload(cpu, SegmentId::Cs, cs);
                        }
                        cpu.rip = ip;
                        outcome.context_changed = true;
                        outcome.preserve_rf = true;
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
                            target(cpu, context, u64::from(*taken), 4)?
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
                        if matches!(kind, StrOp::Ins | StrOp::Outs) {
                            let width = address_bits / 8;
                            let repeating = !matches!(rep, RepKind::None);
                            let count = cpu.gpr[RCX] & mask(width);
                            if repeating && count == 0 {
                                continue;
                            }
                            let output = matches!(kind, StrOp::Outs);
                            let index = if output { RSI } else { RDI };
                            let offset = cpu.gpr[index] & mask(width);
                            let segment = if output { *source } else { SegmentId::Es };
                            let address = legacy::access(
                                cpu,
                                context.generation,
                                segment,
                                offset,
                                *elem,
                                if output {
                                    AccessKind::Read
                                } else {
                                    AccessKind::Write
                                },
                            )?;
                            let value = if output {
                                read(cpu, mem, address, *elem)?
                            } else {
                                0
                            };
                            let delta = if cpu.flags.df {
                                0u64.wrapping_sub(u64::from(*elem))
                            } else {
                                u64::from(*elem)
                            };
                            let next_index = offset.wrapping_add(delta) & mask(width);
                            let next_count = count.wrapping_sub(1) & mask(width);
                            let last = !repeating || next_count == 0;
                            let next_ip = if last { next } else { ir.guest_start };
                            outcome.retired = last;
                            if output {
                                cpu.write_gpr(index, next_index, width);
                                if repeating {
                                    cpu.write_gpr(RCX, next_count, width);
                                }
                                cpu.rip = next_ip;
                            } else {
                                cpu.pending_port_in = Some(*elem);
                                cpu.port_string_in = Some(crate::state::PortStringIn {
                                    rip: ir.guest_start,
                                    next_ip,
                                    instruction_end: next,
                                    base: u64::from(cpu.legacy.segments[segment as usize].base),
                                    offset,
                                    next_index,
                                    next_count,
                                    addr_bits: *address_bits,
                                    elem: *elem,
                                    rep: repeating,
                                    value: None,
                                });
                            }
                            outcome.exit = Some(Exit::PortIo {
                                port: cpu.gpr[RDX] as u16,
                                size: *elem,
                                dir: if output { PortDir::Out } else { PortDir::In },
                                value,
                            });
                            return Ok(outcome);
                        }
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
                                let a = legacy::access(
                                    cpu,
                                    context.generation,
                                    *source,
                                    si,
                                    *elem,
                                    AccessKind::Read,
                                )?;
                                Some(read(cpu, mem, a, *elem)?)
                            } else {
                                None
                            };
                            let dst = if matches!(
                                kind,
                                StrOp::Movs | StrOp::Stos | StrOp::Cmps | StrOp::Scas
                            ) {
                                Some(legacy::access(
                                    cpu,
                                    context.generation,
                                    SegmentId::Es,
                                    di,
                                    *elem,
                                    if matches!(kind, StrOp::Movs | StrOp::Stos) {
                                        AccessKind::Write
                                    } else {
                                        AccessKind::Read
                                    },
                                )?)
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
                        let a = legacy::access(
                            cpu,
                            context.generation,
                            SegmentId::Ss,
                            offset,
                            *size,
                            AccessKind::Read,
                        )?;
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

#[cfg(test)]
mod delivery_tests {
    use super::*;
    use crate::{CpuGeneration, LegacyCpuModel, Prot, RegionKind, Vm, VmConfig};

    #[test]
    fn every_supported_exception_vector_uses_its_gate_width_and_error_code() {
        // I286 PRM §§9.4–9.6 pp. 9-4–10; I386 PRM §§9.6–9.8,
        // Tables 9-6/9-7 pp. 9-20–21: vectors, error codes and frame slots.
        for generation in [CpuGeneration::I286, CpuGeneration::I386] {
            for size in [2u8, 4] {
                if generation == CpuGeneration::I286 && size == 4 {
                    continue;
                }
                let mut vm = Vm::new(VmConfig::flat(0x10000));
                vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
                vm.write_bytes(0x2008, &[0xff, 0xff, 0, 0, 0, 0x9b, 0, 0])
                    .unwrap();
                for vector in [0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16] {
                    let error_code = match vector {
                        8 => Some(0),
                        10..=13 => Some(24),
                        _ => None,
                    };
                    let mut cpu = CpuState::legacy_real_entry([8, 0, 0, 16, 0, 0]);
                    cpu.rip = 0x1000;
                    cpu.gpr[RSP] = 0x8000;
                    cpu.flags.if_ = true;
                    cpu.rflags_system = 0x4100;
                    cpu.legacy.cr0 = 1;
                    cpu.legacy.segments[SegmentId::Cs as usize].base = 0;
                    cpu.legacy.segments[SegmentId::Ss as usize].base = 0;
                    cpu.legacy.gdtr.base = 0x2000;
                    cpu.legacy.gdtr.limit = 15;
                    cpu.legacy.idtr.base = 0x3000;
                    cpu.legacy.idtr.limit = 0x7ff;
                    let context = legacy::derive_legacy_context(
                        LegacyCpuModel {
                            generation,
                            model_semantics_id: 0,
                        },
                        &cpu,
                    )
                    .unwrap();
                    vm.write_bytes(
                        0x3000 + u64::from(vector) * 8,
                        &[0, 0x18, 8, 0, 0, if size == 4 { 0x8e } else { 0x86 }, 0, 0],
                    )
                    .unwrap();
                    deliver_interrupt::<false, 0>(
                        &mut cpu,
                        &vm.mem,
                        context,
                        legacy::LegacyEvent {
                            vector,
                            saved_ip: 0x1000,
                            software: false,
                            external: true,
                            error_code,
                            fault: !matches!(vector, 1 | 3 | 4),
                        },
                    )
                    .unwrap();
                    let sp = 0x8000 - u64::from(size) * (3 + u64::from(error_code.is_some()));
                    assert_eq!((cpu.cs, cpu.rip, cpu.gpr[RSP]), (8, 0x1800, sp));
                    let mut image = [0; 4];
                    vm.read_bytes(sp, &mut image[..size as usize]).unwrap();
                    assert_eq!(u32::from_le_bytes(image), error_code.unwrap_or(0x1000));
                    assert!(!cpu.flags.if_);
                    assert_eq!(cpu.rflags_system & 0x14100, 0);
                }
            }
        }
    }

    #[test]
    fn software_gate_dpl_check_precedes_presence_and_external_delivery_bypasses_dpl() {
        // I286 PRM §9.4 Table 9-1 p. 9-6; I386 PRM §17 INT p. 17-70.
        // The public dispatcher restricts CPL to zero. Exercise the gate check
        // directly at CPL 3, where a lower gate DPL is representable.
        for generation in [CpuGeneration::I286, CpuGeneration::I386] {
            let mut vm = Vm::new(VmConfig::flat(0x10000));
            vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
            let mut cpu = CpuState::legacy_real_entry([3, 0, 0, 3, 0, 0]);
            cpu.rip = 0x1000;
            cpu.legacy.cr0 = 1;
            cpu.legacy.cpl = 3;
            cpu.legacy.idtr.base = 0x3000;
            cpu.legacy.idtr.limit = 0x7ff;
            let context = legacy::derive_legacy_context(
                LegacyCpuModel {
                    generation,
                    model_semantics_id: 0,
                },
                &cpu,
            )
            .unwrap();
            let event = legacy::LegacyEvent {
                vector: 0x30,
                saved_ip: 0x1002,
                software: true,
                external: false,
                error_code: None,
                fault: false,
            };
            for (attr, software, external, expected, code) in [
                (0xc6, true, false, 13, 0x182),
                (0x46, true, false, 13, 0x182),
                (0x46, false, true, 11, 0x183),
                (0x66, true, false, 11, 0x182),
            ] {
                vm.write_bytes(0x3180, &[0, 0x18, 8, 0, 0, attr, 0, 0])
                    .unwrap();
                let exit = deliver_interrupt::<false, 0>(
                    &mut cpu,
                    &vm.mem,
                    context,
                    legacy::LegacyEvent {
                        software,
                        external,
                        ..event
                    },
                )
                .unwrap_err();
                assert!(matches!(exit, Exit::Exception { vector, .. } if vector == expected));
                assert_eq!(
                    exit.fault_metadata().unwrap().error_code(),
                    crate::fault::ErrorCode::bits(code, 0xffff)
                );
                assert_eq!(cpu.rip, 0x1000);
            }
        }
    }
}
