//! Privilege-level transitions on the Machine interpreter (spec.md §17.10):
//! interrupts and exceptions to an inner ring, far CALL and JMP through a call
//! gate, RETF and IRET to an outer ring, and the masked FLAGS load at CPL 1–3.
//!
//! Sources: 80286 and 80287 PRM 210498-005 ("286 PRM"), 80386 PRM
//! 230985-001 ("386 PRM") and SDM 325462-092US, as pinned in `oracles/`.
//! Page numbers are the printed ones. Where the sources disagree, spec.md
//! §17.10 records the rows and the choice (decisions O1–O14).
//!
//! Every fault precedes the commit point; before it the helpers only read and
//! probe. The commit sets the accessed bits of the new CS and SS descriptors,
//! writes the frame and then the registers, so a device stop during the frame
//! writes leaves `CpuState` unchanged and the instruction or event replays.
use super::*;
use crate::legacy::{CpuGeneration, SegmentCache};

/// A stack that is not installed yet: the new SS selector, its descriptor and
/// the (E)SP the transition loads.
struct Stack {
    selector: u16,
    descriptor: Descriptor,
    pointer: u64,
}

impl Stack {
    /// The pointer width the new SS.B selects (SDM Vol 3A §3.4.5).
    fn bytes(&self) -> u8 {
        if self.descriptor.cache.attributes & 0x4000 != 0 {
            4
        } else {
            2
        }
    }
    /// The offset of slot `k` (1-based) of a push of `size`-byte values.
    fn slot(&self, k: u64, size: u8) -> u64 {
        self.pointer.wrapping_sub(k * u64::from(size)) & mask(self.bytes())
    }
    /// Whether a push of `bytes` bytes fits below the pointer, by the
    /// whole-operand rule `legacy::access` applies to the installed SS
    /// (286 PRM §§7.4.1–7.4.2 pp. 7-12–13; 386 PRM §§6.3.1.1–6.3.1.2
    /// pp. 6-4–6).
    fn room(&self, bytes: u64) -> bool {
        let cache = self.descriptor.cache;
        let start = self.pointer.wrapping_sub(bytes) & mask(self.bytes());
        let last = start + bytes - 1;
        let bound = if cache.attributes & 0x4000 != 0 {
            u64::from(u32::MAX)
        } else {
            0xffff
        };
        if cache.attributes & 4 != 0 {
            start > u64::from(cache.limit) && last <= bound
        } else {
            last <= u64::from(cache.limit)
        }
    }
    fn linear(&self, offset: u64) -> u64 {
        u64::from(self.descriptor.cache.base.wrapping_add(offset as u32))
    }
}

/// The ring-`ring` stack from the current TSS, validated (§1.1 of the design).
/// SDM Vol 2A INT p. 3-472–473, CALL p. 3-131; 286 PRM App. B INT p. B-49;
/// 386 PRM §17 INT pp. 17-70–71. The fields are SP_n at 2+4n and SS_n at 4+4n
/// in a 286 TSS (286 PRM §8.2 Fig. 8-1 p. 8-2), ESP_n at 4+8n and SS_n at
/// 8+8n in a 386 TSS (386 PRM §7.1 Fig. 7-1 p. 7-2). A field beyond the TSS
/// limit is #TS(TR selector + EXT) (SDM p. 3-472; DERIVED for both models,
/// whose PRMs are silent); an unusable TR takes the same fault, as
/// `tasks::task_return` does (ASSUMED).
fn inner_stack<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    generation: CpuGeneration,
    ring: u8,
    ext: u16,
) -> Result<Stack, Exit> {
    let tr = cpu.legacy.tr;
    let ring = u32::from(ring);
    let (sp_at, ss_at, sp_bytes) = if tasks::wide(tr) {
        (4 + 8 * ring, 8 + 8 * ring, 4)
    } else {
        (2 + 4 * ring, 4 + 4 * ring, 2)
    };
    if tr.cache.unusable != 0 || ss_at + 1 > tr.cache.limit {
        return Err(fault(cpu, 10, (tr.selector & !3) | ext));
    }
    // SDM p. 3-473 lists NewSS before NewESP; the order is visible only to a
    // device-backed TSS (DERIVED).
    let selector = read(cpu, mem, u64::from(tr.cache.base.wrapping_add(ss_at)), 2)? as u16;
    let pointer = read(
        cpu,
        mem,
        u64::from(tr.cache.base.wrapping_add(sp_at)),
        sp_bytes,
    )?;
    let invalid = |cpu: &CpuState| fault(cpu, 10, (selector & !3) | ext);
    // The checks in SDM order (p. 3-473): null, index, RPL, then DPL and
    // type from the descriptor, then presence. A null SS is #TS(EXT), with
    // 386 PRM p. 17-70's #GP(EXT) recorded in spec.md §17.10 (decision O5).
    if selector & !3 == 0 {
        return Err(fault(cpu, 10, ext));
    }
    let Some(address) = descriptor_address(cpu, selector, generation) else {
        return Err(invalid(cpu));
    };
    if u32::from(selector & 3) != ring {
        return Err(invalid(cpu));
    }
    let descriptor = descriptor_at(cpu, mem, address, generation)?;
    let attr = descriptor.cache.attributes;
    if attr & 0x1a != 0x12 || u32::from((attr >> 5) & 3) != ring {
        return Err(invalid(cpu));
    }
    if attr & 0x80 == 0 {
        return Err(fault(cpu, 12, (selector & !3) | ext));
    }
    supported_descriptor(cpu, attr)?;
    Ok(Stack {
        selector,
        descriptor,
        pointer,
    })
}

/// Decision-24's device preflight: a descriptor whose accessed bit must be
/// set by a locked update is refused on a device before anything commits.
fn preflight<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    descriptor: &Descriptor,
) -> Result<(), Exit> {
    match descriptor.access_address {
        Some(address) if descriptor.cache.attributes & 1 == 0 => {
            tasks::probe_descriptor_cas(cpu, &mem.address_keyed(), address)
        }
        _ => Ok(()),
    }
}

/// The IDT gate as `deliver_protected` read it: its ledger key and words.
pub(super) struct Gate {
    pub key: u64,
    pub low: u32,
    pub high: u32,
}

/// An interrupt or trap gate to a non-conforming handler with DPL < CPL.
/// 286 PRM App. B INT p. B-49 and §9.4 Fig. 9-4 pp. 9-4–5; 386 PRM §17 INT
/// pp. 17-70–71; SDM Vol 2A INT pp. 3-472–474; SDM Vol 3A §7.12.1 p. 7-17.
/// The caller has made the gate and target-CS checks. The new stack is the
/// TSS's; it holds old SS, old (E)SP, (E)FLAGS, CS, (E)IP and the error code,
/// at the gate's width.
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(super) fn inward<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    context: MachineContext,
    event: legacy::LegacyEvent,
    cs: u16,
    mut code: Descriptor,
    ip: u64,
    size: u8,
    interrupt_gate: bool,
    gate: Gate,
) -> Result<(), Exit> {
    let ext = u16::from(event.external);
    let ring = ((code.cache.attributes >> 5) & 3) as u8;
    let mut stack = inner_stack(cpu, mem, context.generation, ring, ext)?;
    // Room for 10 or 20 bytes, and the error code's 2 or 4 more: #SS(new SS
    // + EXT) (SDM p. 3-473; 286 PRM Table 9-1 p. 9-6). Decision O2; the
    // PRM pages B-49 and 17-71 count 10 or 20 bytes and give #SS(0).
    let count = 5 + u64::from(event.error_code.is_some());
    if !stack.room(count * u64::from(size)) {
        return Err(fault(cpu, 12, (stack.selector & !3) | ext));
    }
    // SDM p. 3-474: #GP(EXT), the code of the landed same-level path;
    // 286 PRM p. B-49 gives #GP(0).
    if ip > u64::from(code.cache.limit) {
        return Err(fault(cpu, 13, ext));
    }
    let slots = (1..=count)
        .map(|k| stack.linear(stack.slot(k, size)))
        .collect::<Vec<_>>();
    for &at in &slots {
        probe(cpu, mem, at, size).map_err(|exit| {
            operand_fault(
                exit,
                cpu.rip,
                at,
                crate::ir::MemoryAccess::new(u16::from(size), false, true),
            )
        })?;
    }
    // Commit point.
    cpu.record_mmio_part(cpu.rip, gate.key, u64::from(gate.low));
    cpu.record_mmio_part(cpu.rip, gate.key | (1 << 32), u64::from(gate.high));
    preflight(cpu, mem, &code)?;
    preflight(cpu, mem, &stack.descriptor)?;
    accessed(cpu, mem, &mut code)?;
    accessed(cpu, mem, &mut stack.descriptor)?;
    let mut flags = legacy::flags_image(cpu, context.generation);
    // I386 PRM §12.3.1.1 p. 12-7: fault frames set RF; traps and INT do not.
    if context.generation == CpuGeneration::I386 && event.fault {
        flags |= 1 << 16;
    }
    // The selectors are zero-extended in a 32-bit frame, as the engine's CS
    // pushes are (386 PRM p. 17-71 "3 words padded to 4"; ASSUMED value).
    let mut frame = vec![
        u64::from(cpu.ss),
        cpu.gpr[RSP] & mask(size),
        u64::from(flags),
        u64::from(cpu.cs),
        u64::from(event.saved_ip),
    ];
    if let Some(error) = event.error_code {
        frame.push(u64::from(error));
    }
    for (&at, &value) in slots.iter().zip(&frame) {
        write(cpu, mem, at, size, value)?;
    }
    install(cpu, SegmentId::Ss, stack.selector, stack.descriptor.cache);
    // SDM p. 3-474: ESP := NewESP, then the pushes at the new SS.B width.
    cpu.gpr[RSP] = stack.pointer;
    cpu.write_gpr(RSP, stack.slot(count, size), stack.bytes());
    cpu.legacy.cpl = ring;
    if interrupt_gate {
        cpu.flags.if_ = false;
    }
    // 286 PRM p. B-49; 386 PRM p. 17-71: TF and NT clear. SDM Vol 3A
    // §7.12.1.3 also clears RF in the live handler flags.
    cpu.rflags_system &= !0x14100;
    install(cpu, SegmentId::Cs, (cs & !3) | u16::from(ring), code.cache);
    cpu.rip = ip;
    Ok(())
}

/// Retry keys for the parameters a call gate copies, packed by 64-bit word:
/// four words or two doublewords per key, in the engine-only space of
/// `DESCRIPTOR_RETRY`.
const PARAMETER_RETRY: u64 = DESCRIPTOR_RETRY | (1 << 35);

/// The fields of a call-gate descriptor, decoded from the cache that
/// `descriptor_at` builds from its image: offset 15:0 in bytes 0-1, the
/// target selector in bytes 2-3, the parameter count in byte 4 bits 4:0 and,
/// in a type-12 gate, offset 31:16 in bytes 6-7 (286 PRM §7.5.1.1 Fig. 7-10
/// p. 7-17; 386 PRM §6.3.4 Fig. 6-5 p. 6-11; SDM Vol 3A §6.8.3 Fig. 6-8
/// p. 6-13). The cache splits offset 31:16 across the limit, flags and base
/// fields, and scales the limit when byte-6 bit 7 (offset bit 23) is set; the
/// decode undoes both (DERIVED from that layout). On the 80286 bytes 6-7 are
/// reserved and the cache never holds them.
fn gate_fields(gate: &Descriptor) -> (u16, u32, u64) {
    let cache = gate.cache;
    let attr = u32::from(cache.attributes);
    let raw = if attr & 0x8000 != 0 {
        cache.limit >> 12
    } else {
        cache.limit
    };
    let offset = (raw & 0xf_ffff) | ((attr & 0xf000) << 8) | (cache.base & 0xff00_0000);
    (
        cache.base as u16,
        offset,
        u64::from((cache.base >> 16) & 0x1f),
    )
}

/// A far CALL or JMP whose selector names a call gate: type 4, a 16-bit gate
/// on both generations, or type 12, a 32-bit gate on the 80386. The gate
/// type, not the instruction's operand size, sets the width of the offset, the
/// pushes and the parameters: words for type 4, doublewords for type 12
/// (386 PRM §16.4.2.1 p. 16-6; SDM Vol 3A §6.8.3 p. 6-14). The caller has
/// checked the selector for null and its table limit (`far_descriptor`).
///
/// Check order: 286 PRM App. B CALL pp. B-24–25, JMP p. B-57; 386 PRM §17
/// CALL pp. 17-39–41, JMP pp. 17-85–86; SDM Vol 2A CALL pp. 3-130–133; SDM
/// Vol 3A §6.8.4 Table 6-1 p. 6-16. A CALL to a non-conforming segment with
/// DPL < CPL switches to the TSS stack of that ring and copies the parameters
/// (SDM Vol 3A §6.8.5 steps 1–9 p. 6-18, Fig. 6-13 p. 6-19; 286 PRM
/// §7.5.1.4 p. 7-20). Every other transfer stays at CPL with no stack switch:
/// a conforming target, or DPL = CPL (SDM Vol 3A p. 6-17). A JMP never changes
/// privilege (286 PRM §7.5.1.3 p. 7-19; 386 PRM §6.3.4 p. 6-12).
#[cold]
#[inline(never)]
pub(super) fn call_gate<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    context: MachineContext,
    selector: u16,
    gate: Descriptor,
    far_ip: u64,
    return_ip: Option<u32>,
) -> Result<(), Exit> {
    let generation = context.generation;
    let cpl = cpu.legacy.cpl;
    let attr = gate.cache.attributes;
    let size: u8 = if attr & 8 != 0 { 4 } else { 2 };
    // 1. MAX(CPL, RPL) <= gate DPL, else #GP(gate); 2. the gate present,
    // else #NP(gate).
    let gate_dpl = ((attr >> 5) & 3) as u8;
    if gate_dpl < cpl || gate_dpl < (selector & 3) as u8 {
        return Err(fault(cpu, 13, selector & !3));
    }
    if attr & 0x80 == 0 {
        return Err(fault(cpu, 11, selector & !3));
    }
    // 3–7. The target CS: null #GP(0); index #GP(CS); not code or DPL > CPL
    // #GP(CS); for JMP, non-conforming DPL != CPL #GP(CS) (286 PRM p. B-57;
    // SDM Table 6-1); not present #NP(CS). The 286 and 386 CALL pages omit
    // the presence test; 286 PRM Table 7-3 p. 7-18 and SDM p. 3-131 make it.
    // It follows the type and DPL tests, in the SDM order; Table 7-3 lists
    // it first, and that order is not taken (decision O7).
    let (cs, offset, count) = gate_fields(&gate);
    if cs & !3 == 0 {
        return Err(fault(cpu, 13, 0));
    }
    let mut code = descriptor(cpu, mem, cs, generation)?;
    let code_attr = code.cache.attributes;
    let dpl = ((code_attr >> 5) & 3) as u8;
    let conforming = code_attr & 4 != 0;
    if code_attr & 0x18 != 0x18 || dpl > cpl {
        return Err(fault(cpu, 13, cs & !3));
    }
    // 286 PRM §7.5.1.3 p. 7-19 names the gate in this error code; p. B-57,
    // 386 PRM p. 17-85 and SDM p. 3-509 name the CS selector, which is taken
    // (spec.md §17.10).
    if return_ip.is_none() && !conforming && dpl != cpl {
        return Err(fault(cpu, 13, cs & !3));
    }
    if code_attr & 0x80 == 0 {
        return Err(fault(cpu, 11, cs & !3));
    }
    supported_descriptor(cpu, code_attr)?;
    // A 16-bit gate supplies offset 15:0 only (SDM p. 3-132 "CallGate
    // (InstructionPointer) AND FFFFH").
    let ip = u64::from(offset) & mask(size);
    let Some(return_ip) = return_ip else {
        // JMP: offset beyond the CS limit #GP(0); CS.RPL := CPL (286 PRM
        // p. B-57; 386 PRM p. 17-86).
        code_limit(cpu, code.cache, ip)?;
        accessed(cpu, mem, &mut code)?;
        install(cpu, SegmentId::Cs, (cs & !3) | u16::from(cpl), code.cache);
        cpu.rip = ip;
        return Ok(());
    };
    if conforming || dpl == cpl {
        // SAME-PRIVILEGE: room for 4 bytes, or 8 on a 32-bit gate, #SS(0);
        // then the offset, #GP(0); CS.RPL := CPL (286 PRM p. B-25; 386 PRM
        // pp. 17-40–41; SDM pp. 3-133–134).
        push_addresses(cpu, mem, context, 2, size)?;
        code_limit(cpu, code.cache, ip)?;
        cpu.record_mmio_part(cpu.rip, TRANSFER_IP_RETRY, far_ip);
        cpu.record_mmio_part(cpu.rip, TRANSFER_CS_RETRY, u64::from(selector));
        remember(cpu, &gate);
        accessed(cpu, mem, &mut code)?;
        let frame = [u64::from(cpu.cs), u64::from(return_ip)];
        push(cpu, mem, context, &frame, size)?;
        install(cpu, SegmentId::Cs, (cs & !3) | u16::from(cpl), code.cache);
        cpu.rip = ip;
        return Ok(());
    }
    // MORE-PRIVILEGE. The ring-`dpl` stack from the TSS, checked as for an
    // interrupt with EXT = 0: a null SS is #TS(0) (286 PRM p. B-25; 386 PRM
    // p. 17-40; SDM p. 3-131).
    let mut stack = inner_stack(cpu, mem, generation, dpl, 0)?;
    // Room for the parameters plus 8 bytes, or plus 16 on a 32-bit gate. The
    // fault is #SS(0) on the 80286 (286 PRM p. B-25; §7.5.1.4 p. 7-20 "a
    // stack fault occurs with an error code of 0") and #SS(new SS) on the
    // 80386 (386 PRM §9.8.12 p. 9-16 "overflow of the new stack during an
    // interlevel CALL"; SDM p. 3-132), where 386 PRM p. 17-40 and §6.3.4.1
    // p. 6-15 print #SS(0) (decision O1).
    let slots = 4 + count;
    if !stack.room(slots * u64::from(size)) {
        let code = if generation == CpuGeneration::I286 {
            0
        } else {
            stack.selector & !3
        };
        return Err(fault(cpu, 12, code));
    }
    code_limit(cpu, code.cache, ip)?;
    // The parameters, from the old SS:(E)SP at the old SS.B width. A
    // parameter beyond the old SS limit is #SS(0), after the offset check;
    // no source orders the two (ASSUMED, decision O11). Every parameter is
    // read before the first write, so a new stack that overlaps the old one
    // copies the caller's values (decision O11). They are read in push order,
    // the one at the highest address first (ASSUMED; the order is visible to
    // a device only).
    let width = context.stack_address_bits / 8;
    let old = sp(cpu, context);
    let mut sources = Vec::with_capacity(count as usize);
    for k in (0..count).rev() {
        let offset = old.wrapping_add(k * u64::from(size)) & mask(width);
        sources.push(segment_access(
            cpu,
            context,
            SegmentId::Ss,
            offset,
            size,
            AccessKind::Read,
        )?);
    }
    let targets = (1..=slots)
        .map(|k| stack.linear(stack.slot(k, size)))
        .collect::<Vec<_>>();
    for &at in &targets {
        probe(cpu, mem, at, size).map_err(|exit| {
            operand_fault(
                exit,
                cpu.rip,
                at,
                crate::ir::MemoryAccess::new(u16::from(size), false, true),
            )
        })?;
    }
    let per = 8 / u64::from(size);
    // The retry ledger must hold every entry the frame copy can make, decided
    // before its first device transfer (`frame_fits_ledger`).
    if !frame_fits_ledger(
        cpu,
        mem,
        [&gate, &code, &stack.descriptor],
        count.div_ceil(per),
        &sources,
        &targets,
        size,
    ) {
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyInstruction,
        ));
    }
    // A replay after a device stop in the frame takes the parameters from the
    // ledger, as a frame write over the old stack may have changed them.
    let mut parameters = Vec::with_capacity(count as usize);
    for (i, &at) in sources.iter().enumerate() {
        let i = i as u64;
        let key = PARAMETER_RETRY | (i / per);
        let shift = (i % per) * u64::from(size) * 8;
        if let Some(packed) = cpu.mmio_part(cpu.rip, key) {
            mem.skip_read(cpu, at, size);
            parameters.push((packed >> shift) & mask(size));
            continue;
        }
        parameters.push(read(cpu, mem, at, size).map_err(|exit| {
            operand_fault(
                exit,
                cpu.rip,
                at,
                crate::ir::MemoryAccess::new(u16::from(size), false, false),
            )
        })?);
    }
    // Commit point.
    cpu.record_mmio_part(cpu.rip, TRANSFER_IP_RETRY, far_ip);
    cpu.record_mmio_part(cpu.rip, TRANSFER_CS_RETRY, u64::from(selector));
    remember(cpu, &gate);
    for (group, values) in parameters.chunks(per as usize).enumerate() {
        let packed = values.iter().enumerate().fold(0, |packed, (j, &value)| {
            packed | (value << (j as u64 * u64::from(size) * 8))
        });
        cpu.record_mmio_part(cpu.rip, PARAMETER_RETRY | group as u64, packed);
    }
    preflight(cpu, mem, &code)?;
    preflight(cpu, mem, &stack.descriptor)?;
    accessed(cpu, mem, &mut code)?;
    accessed(cpu, mem, &mut stack.descriptor)?;
    // Old SS, old (E)SP, the parameters in their order, old CS and the return
    // (E)IP (SDM p. 3-132; 286 PRM §7.5.1.4 Fig. 7-12 p. 7-21). The selectors
    // are zero-extended in a 32-bit frame, as the engine's CS pushes are
    // (386 PRM p. 17-40 "6-byte return address (padded to 8 bytes)";
    // ASSUMED value).
    let mut frame = Vec::with_capacity(slots as usize);
    frame.push(u64::from(cpu.ss));
    frame.push(cpu.gpr[RSP] & mask(size));
    frame.extend_from_slice(&parameters);
    frame.push(u64::from(cpu.cs));
    frame.push(u64::from(return_ip));
    for (&at, &value) in targets.iter().zip(&frame) {
        write(cpu, mem, at, size, value)?;
    }
    install(cpu, SegmentId::Ss, stack.selector, stack.descriptor.cache);
    // ESP := NewESP, then the pushes at the new SS.B width (SDM p. 3-132).
    cpu.gpr[RSP] = stack.pointer;
    cpu.write_gpr(RSP, stack.slot(slots, size), stack.bytes());
    // CPL := the CS DPL; CS.RPL := CPL (SDM p. 3-132; 286 PRM p. B-25).
    cpu.legacy.cpl = dpl;
    install(cpu, SegmentId::Cs, (cs & !3) | u16::from(dpl), code.cache);
    cpu.rip = ip;
    Ok(())
}

/// The bus pieces of a `bytes`-byte transfer at `linear`, as `read`,
/// `write` and `skip_read` split it.
fn pieces(
    cpu: &CpuState,
    physical_mask: u32,
    mut linear: u64,
    mut bytes: u8,
    mut visit: impl FnMut(u64, u8),
) {
    while bytes != 0 {
        let (physical, part) = legacy::physical_span(cpu, physical_mask, linear, bytes);
        visit(physical, part);
        linear = u64::from((linear as u32).wrapping_add(u32::from(part)));
        bytes -= part;
    }
}

/// Whether the retry ledger (`CpuState::mmio_parts`, 64 entries) holds every
/// entry an inward call-gate CALL makes. A larger count would drop an
/// answered transfer, so a replay would re-ask it or give its answer to
/// another transfer (spec.md §5.2). The decision comes before the parameter
/// reads, the first device transfer of the frame copy. The reads before it
/// (the far pointer, the gate twice, the CS and SS descriptors and the two
/// TSS fields, each at most two bus pieces) make at most 24 entries, so they
/// cannot fill the ledger alone. The count is
/// the entries already held for this instruction plus those still to come:
/// the far-pointer keys, two per descriptor kept for the replay (gate, CS,
/// SS), one per packed parameter group, one per device piece of a parameter
/// read and one per device piece of a frame write. A piece counts as a device
/// piece if any of its bytes is in a `Trap` region. The count does not depend
/// on how far an earlier attempt got, so every replay decides alike.
fn frame_fits_ledger<const ROUTED: bool, const ORDER: u8>(
    cpu: &CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    descriptors: [&Descriptor; 3],
    groups: u64,
    sources: &[u64],
    targets: &[u64],
    size: u8,
) -> bool {
    let device = |physical: u64, part: u8| {
        (0..u64::from(part)).any(|i| {
            matches!(
                mem.raw().region_kind(physical + i, 1),
                Ok(crate::RegionKind::Trap)
            )
        })
    };
    let mut keys = vec![TRANSFER_IP_RETRY, TRANSFER_CS_RETRY];
    for descriptor in descriptors {
        if let Some(address) = descriptor.access_address {
            let key = DESCRIPTOR_RETRY | u64::from(address.wrapping_sub(5));
            keys.extend([key, key | (1 << 32)]);
        }
    }
    keys.extend((0..groups).map(|group| PARAMETER_RETRY | group));
    let mut ahead = 0;
    for &at in sources {
        pieces(cpu, mem.physical_mask(), at, size, |physical, part| {
            if device(physical, part) {
                keys.push(mem.next_read_key(physical, ahead));
            }
            ahead += 1;
        });
    }
    for &at in targets {
        pieces(cpu, mem.physical_mask(), at, size, |physical, part| {
            if device(physical, part) {
                keys.push(physical | (1 << 63));
            }
        });
    }
    keys.sort_unstable();
    keys.dedup();
    let held = cpu.mmio_retry_active && cpu.owns_mmio_retry(cpu.rip);
    let recorded = if held {
        usize::from(cpu.mmio_parts_len)
    } else {
        0
    };
    let new = keys
        .iter()
        .filter(|&&key| !held || cpu.mmio_part(cpu.rip, key).is_none())
        .count();
    recorded + new <= cpu.mmio_parts.len()
}

/// A far return frame that names an outer ring: RETF or RETF n (`flags` is
/// `None`, `release` is n) or IRET (`flags` is the popped image).
pub(super) struct Return {
    pub ip: u64,
    pub cs: u16,
    pub flags: Option<u32>,
    pub size: u8,
    pub release: u16,
}

fn frame_value<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    context: MachineContext,
    offset: u64,
    size: u8,
) -> Result<u64, Exit> {
    let offset = sp(cpu, context).wrapping_add(offset) & mask(context.stack_address_bits / 8);
    let addr = segment_access(cpu, context, SegmentId::Ss, offset, size, AccessKind::Read)?;
    read(cpu, mem, addr, size).map_err(|exit| {
        operand_fault(
            exit,
            cpu.rip,
            addr,
            crate::ir::MemoryAccess::new(u16::from(size), false, true),
        )
    })
}

/// RETF, RETF n or IRET whose popped CS.RPL is greater than CPL.
/// 286 PRM App. B RET pp. B-94–95, IRET pp. B-51–52, §7.5.2 pp. 7-20–22;
/// 386 PRM §17 RET pp. 17-142–144, IRET pp. 17-76–78, §6.3.4.2 Table 6-3
/// pp. 6-16–17; SDM Vol 2B RET pp. 4-571–574; SDM Vol 2A IRET
/// pp. 3-493–497. The caller has checked the frame's first two slots and
/// read IP, CS and, for IRET, FLAGS.
#[cold]
#[inline(never)]
pub(super) fn outward<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    context: MachineContext,
    frame: Return,
) -> Result<(), Exit> {
    let generation = context.generation;
    let size = u64::from(frame.size);
    // RET: IP, CS, n parameter bytes, SP, SS; IRET: IP, CS, FLAGS, SP, SS
    // (286 PRM p. B-95 "SS selector (at SP+6+imm)", p. B-52 "(at SP+8)";
    // 386 PRM Table 6-3 p. 6-17 "SS selector at ESP + N + 12").
    let sp_at = if frame.flags.is_some() {
        3 * size
    } else {
        2 * size + u64::from(frame.release)
    };
    let ss_at = sp_at + size;
    // Top 8+n or 10 bytes (16+n or 20 with a 32-bit operand) within the
    // stack limits, else #SS(0). The whole range, first and last byte, by the
    // rule that checks every stack operand.
    let top = sp(cpu, context);
    segment_access(cpu, context, SegmentId::Ss, top, 1, AccessKind::Read)?;
    segment_access(
        cpu,
        context,
        SegmentId::Ss,
        top + ss_at + size - 1,
        1,
        AccessKind::Read,
    )?;
    // The return CS. Its RPL is the new CPL. A conforming segment needs
    // DPL <= RPL on RET and on IRET (decision O5: 286 PRM p. B-52 and
    // 386 PRM p. 17-77 print "DPL must be > CPL" for IRET).
    let cs = frame.cs;
    let rpl = (cs & 3) as u8;
    if cs & !3 == 0 {
        return Err(fault(cpu, 13, 0));
    }
    let mut code = descriptor(cpu, mem, cs, generation)?;
    let attr = code.cache.attributes;
    let dpl = ((attr >> 5) & 3) as u8;
    if attr & 0x18 != 0x18 || if attr & 4 != 0 { dpl > rpl } else { dpl != rpl } {
        return Err(fault(cpu, 13, cs & !3));
    }
    if attr & 0x80 == 0 {
        return Err(fault(cpu, 11, cs & !3));
    }
    // The return SS. Selector, type and privilege failures are #GP(SS)
    // (decision O4); a not-present SS is #SS(SS) on both generations
    // (decision O3: the 386 PRM's RET and IRET pages print #NP).
    let pointer = frame_value(cpu, mem, context, sp_at, frame.size)?;
    let ss = frame_value(cpu, mem, context, ss_at, frame.size)? as u16;
    if ss & !3 == 0 {
        return Err(fault(cpu, 13, 0));
    }
    let mut stack = descriptor(cpu, mem, ss, generation)?;
    let stack_attr = stack.cache.attributes;
    if (ss & 3) as u8 != rpl || stack_attr & 0x1a != 0x12 || ((stack_attr >> 5) & 3) as u8 != rpl {
        return Err(fault(cpu, 13, ss & !3));
    }
    if stack_attr & 0x80 == 0 {
        return Err(fault(cpu, 12, ss & !3));
    }
    if frame.ip > u64::from(code.cache.limit) {
        return Err(fault(cpu, 13, 0));
    }
    supported_descriptor(cpu, attr)?;
    supported_descriptor(cpu, stack_attr)?;
    // Commit point.
    preflight(cpu, mem, &code)?;
    preflight(cpu, mem, &stack)?;
    accessed(cpu, mem, &mut code)?;
    accessed(cpu, mem, &mut stack)?;
    if let Some(image) = frame.flags {
        // SDM Vol 3A §11.3: all frame reads complete before retirement.
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        // The old CPL and IOPL decide which flags load: the SDM assigns
        // them before CPL := CS.RPL (IRET p. 3-493).
        load_flags(cpu, generation, image)?;
        if frame.size == 4 {
            cpu.rflags_system = (cpu.rflags_system & !(1 << 16)) | u64::from(image & (1 << 16));
        }
    }
    cpu.legacy.cpl = rpl;
    install(cpu, SegmentId::Cs, cs, code.cache);
    cpu.rip = frame.ip;
    install(cpu, SegmentId::Ss, ss, stack.cache);
    // SDM IRET p. 3-493: a 16-bit pop clears the upper bits, then
    // ESP := tempESP; SDM RET p. 4-573 assigns ESP := tempESP the same way
    // (DERIVED). RET n then releases n bytes at the new SS.B width
    // (SDM p. 4-574; 286 PRM §7.5.2 p. 7-21; 386 PRM §6.3.4.2 p. 6-16),
    // unchecked (decision O9: the PRM pseudocode omits this adjustment).
    cpu.gpr[RSP] = pointer & mask(frame.size);
    if frame.release != 0 {
        let bytes = if stack.cache.attributes & 0x4000 != 0 {
            4
        } else {
            2
        };
        cpu.write_gpr(
            RSP,
            cpu.gpr[RSP].wrapping_add(u64::from(frame.release)) & mask(bytes),
            bytes,
        );
    }
    null_data_segments(cpu, generation);
    Ok(())
}

/// After an outward return, a data segment register that the new CPL may not
/// use becomes null: ES and DS on the 80286; ES, FS, GS and DS on the
/// 80386. The rule is the SDM's, from the hidden cache alone: data or
/// non-conforming code with DPL < CPL (SDM Vol 2B RET p. 4-574, Vol 2A IRET
/// p. 3-495, Vol 3A §6.8.6 p. 6-20). A null selector whose cache passes that
/// test, such as one left by real mode or LOADALL, is kept as it is.
/// Decision O6: the PRM prose says "greater" (286 PRM §7.5.2 p. 7-21; 386 PRM
/// §6.3.4.2 p. 6-16), and the PRM pseudocode's re-validation against the
/// table limit and readability is not modelled. Nothing faults here.
fn null_data_segments(cpu: &mut CpuState, generation: CpuGeneration) {
    let segments: &[SegmentId] = if generation == CpuGeneration::I286 {
        &[SegmentId::Es, SegmentId::Ds]
    } else {
        &[SegmentId::Es, SegmentId::Fs, SegmentId::Gs, SegmentId::Ds]
    };
    for &segment in segments {
        let cache = cpu.legacy.segments[segment as usize];
        let dpl = ((cache.attributes >> 5) & 3) as u8;
        let data_or_nonconforming = cache.attributes & 8 == 0 || cache.attributes & 4 == 0;
        if data_or_nonconforming && dpl < cpu.legacy.cpl {
            install(
                cpu,
                segment,
                0,
                SegmentCache {
                    unusable: 1,
                    ..Default::default()
                },
            );
        }
    }
}

/// POPF and IRET at CPL > 0: IOPL loads only at CPL 0 and IF only when
/// CPL <= IOPL, with no fault (286 PRM App. B POPF p. B-86, IRET p. B-51,
/// §10.1 p. 10-1; 386 PRM §8.3.1 p. 8-5, §17 POPF p. 17-130; SDM Vol 2B POPF
/// pp. 4-407–408, Vol 2A IRET p. 3-493). CPL and IOPL are the values before
/// the instruction. At CPL 0 this is `legacy::restore_flags`.
#[inline]
pub(super) fn load_flags(
    cpu: &mut CpuState,
    generation: CpuGeneration,
    image: u32,
) -> Result<(), Exit> {
    if cpu.legacy.cpl == 0 {
        return legacy::restore_flags(cpu, generation, image);
    }
    restricted_flags(cpu, generation, image)
}

#[cold]
#[inline(never)]
fn restricted_flags(cpu: &mut CpuState, generation: CpuGeneration, image: u32) -> Result<(), Exit> {
    let iopl = ((cpu.rflags_system >> 12) & 3) as u8;
    let interrupts = cpu.flags.if_;
    let level = cpu.rflags_system & 0x3000;
    legacy::restore_flags(cpu, generation, image)?;
    if cpu.legacy.cpl > iopl {
        cpu.flags.if_ = interrupts;
    }
    cpu.rflags_system = (cpu.rflags_system & !0x3000) | level;
    Ok(())
}

/// The run-time test of [`legacy::PrivilegeCheck`], against the live CPL and
/// IOPL: IOPL can change within a context through a CPL-0 POPF.
#[cold]
#[inline(never)]
pub(super) fn check(
    cpu: &CpuState,
    generation: CpuGeneration,
    kind: legacy::PrivilegeCheck,
) -> Result<(), Exit> {
    let cpl = cpu.legacy.cpl;
    let iopl = ((cpu.rflags_system >> 12) & 3) as u8;
    let (permitted, port) = match kind {
        legacy::PrivilegeCheck::Cpl0 => (cpl == 0, false),
        legacy::PrivilegeCheck::Iopl => (cpl <= iopl, false),
        legacy::PrivilegeCheck::IoPort => (cpl <= iopl, true),
        // SDM Vol 2B REP Operation p. 4-564: a zero count executes nothing.
        legacy::PrivilegeCheck::RepIoPort { count_bytes } => {
            (cpl <= iopl || cpu.gpr[RCX] & mask(count_bytes) == 0, true)
        }
    };
    if permitted {
        return Ok(());
    }
    // 386 PRM §17.2.2.5 p. 17-15: the I/O permission function returns FALSE
    // for a 286 TSS, which has no bitmap (SDM Vol 3A §10.6 p. 10-18), so
    // only a 386 TSS's bitmap is consulted. A TR that holds neither is
    // #GP(0) as well (ASSUMED).
    if port && generation == CpuGeneration::I386 && tasks::wide(cpu.legacy.tr) {
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyIoPermissionBitmap,
        ));
    }
    Err(fault(cpu, 13, 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LegacyCpuModel, Prot, RegionKind, Vm, VmConfig};

    #[test]
    fn inward_room_counts_the_error_code_and_faults_with_the_new_ss_and_ext() {
        // Decision O2: room for 10 bytes, 12 with an error code (20 or 24
        // on a 32-bit gate), else #SS(new SS + EXT) (SDM Vol 2A INT
        // p. 3-473; 286 PRM Table 9-1 p. 9-6). SS0 is expand-down data with
        // limit 0FFFh, whose lowest valid offset is 1000h (the sources
        // `Stack::room` cites). An SP0 of 1000h + 10 (or + 20) fits the
        // frame without an error code exactly, and not one with it.
        for generation in [CpuGeneration::I286, CpuGeneration::I386] {
            for size in [2u8, 4] {
                if generation == CpuGeneration::I286 && size == 4 {
                    continue;
                }
                for error_code in [None, Some(0x18)] {
                    let mut vm = Vm::new(VmConfig::flat(0x10000));
                    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
                    // 08h: DPL-0 code; 10h: DPL-0 expand-down data, limit
                    // 0FFFh, accessed bit clear; 18h: DPL-3 code.
                    vm.write_bytes(0x2008, &[0xff, 0xff, 0, 0, 0, 0x9b, 0, 0])
                        .unwrap();
                    vm.write_bytes(0x2010, &[0xff, 0x0f, 0, 0, 0, 0x96, 0, 0])
                        .unwrap();
                    vm.write_bytes(
                        0x3008,
                        &[0, 0x18, 8, 0, 0, if size == 4 { 0x8e } else { 0x86 }, 0, 0],
                    )
                    .unwrap();
                    // A 286 TSS at 4000h: SP0 = 1000h + 5 * size, SS0 = 10h.
                    vm.mem
                        .write(0x4002, 0x1000 + 5 * u64::from(size), 2)
                        .unwrap();
                    vm.mem.write(0x4004, 0x10, 2).unwrap();
                    let mut cpu = CpuState::legacy_real_entry([0x1b, 0x23, 0x23, 0x23, 0, 0]);
                    cpu.rip = 0x1000;
                    cpu.gpr[RSP] = 0x8000;
                    cpu.legacy.cr0 = 1;
                    cpu.legacy.cpl = 3;
                    cpu.legacy.segments[SegmentId::Cs as usize].attributes = 0xfb;
                    cpu.legacy.segments[SegmentId::Ss as usize].attributes = 0xf3;
                    cpu.legacy.gdtr.base = 0x2000;
                    cpu.legacy.gdtr.limit = 0x1f;
                    cpu.legacy.idtr.base = 0x3000;
                    cpu.legacy.idtr.limit = 0x7ff;
                    cpu.legacy.tr = legacy::SystemSegment {
                        selector: 0x28,
                        cache: legacy::SegmentCache {
                            base: 0x4000,
                            limit: 0x2b,
                            attributes: 0x83,
                            ..Default::default()
                        },
                        ..Default::default()
                    };
                    let context = legacy::derive_legacy_context(
                        LegacyCpuModel {
                            generation,
                            model_semantics_id: 0,
                        },
                        &cpu,
                    )
                    .unwrap();
                    let before = cpu.clone();
                    let result = deliver_interrupt::<false, 0>(
                        &mut cpu,
                        &vm.mem,
                        context,
                        legacy::LegacyEvent {
                            vector: 1,
                            saved_ip: 0x1000,
                            software: false,
                            external: true,
                            error_code,
                            fault: true,
                        },
                        &mut tasks::SwitchState::default(),
                    );
                    if error_code.is_none() {
                        result.unwrap();
                        assert_eq!((cpu.cs, cpu.legacy.cpl, cpu.gpr[RSP]), (8, 0, 0x1000));
                    } else {
                        let exit = result.unwrap_err();
                        assert!(matches!(exit, Exit::Exception { vector: 12, .. }));
                        assert_eq!(
                            exit.fault_metadata().unwrap().error_code(),
                            crate::fault::ErrorCode::bits(0x11, 0xffff)
                        );
                        assert_eq!(format!("{cpu:?}"), format!("{before:?}"));
                        assert_eq!(vm.mem.read(0x2015, 1).unwrap(), 0x96);
                    }
                }
            }
        }
    }
}
