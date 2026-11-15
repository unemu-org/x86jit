//! Machine memory and transactional system helpers for the shared interpreter
//! (decision-18 §§1, 4; spec.md §§8.1, 17).
use super::*;
pub(crate) mod tasks;
use crate::legacy::{
    self, ControlRegister, DescriptorQuery, LegacyOp as L, MachineContext, SegmentId,
    SelectorDestination, SystemRegister,
};

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

#[inline(never)]
pub(super) fn direct_stack_address(
    cpu: &mut CpuState,
    offset: u64,
    bytes: u8,
    vector: u8,
    ip: u64,
) -> Result<u64, Exit> {
    // 80386 PRM §17.1.3: SS.B=0 selects SP independently of operand size.
    // §14.7 item 7/Table 14-1; 286 PRM Table C-1: gate the whole operand.
    if offset + u64::from(bytes) > 0x10000 {
        cpu.rip = ip;
        return Err(legacy::segment_fault(
            cpu,
            vector,
            offset,
            bytes,
            AccessKind::Read,
        ));
    }
    Ok((u64::from(cpu.ss) << 4) + offset)
}

#[cold]
pub(super) fn stack_fault_access(mut exit: Exit, ops: &[IrOp], rip: u64) -> Exit {
    let start = ops
        .iter()
        .position(|op| matches!(op, IrOp::InsnStart { guest_addr } if *guest_addr == rip))
        .unwrap_or(0);
    let gate = ops[start..]
        .iter()
        .position(|op| matches!(op, IrOp::Legacy(L::DirectStackAddress { .. })))
        .expect("stack gate");
    let access = ops[start + gate + 1..]
        .iter()
        .find_map(|op| match op {
            IrOp::Load { .. } => Some(AccessKind::Read),
            IrOp::Store { .. } => Some(AccessKind::Write),
            _ => None,
        })
        .expect("stack transfer");
    if let Exit::Exception {
        metadata: Some(m), ..
    } = &mut exit
    {
        m.access = crate::Knowledge::Known(access.into());
    }
    exit
}

#[cold]
#[inline(never)]
pub(super) fn scalar_limit_access(ops: &[IrOp], rip: u64, offset: Val) -> AccessKind {
    let start = ops
        .iter()
        .position(|op| matches!(op, IrOp::InsnStart { guest_addr } if *guest_addr == rip))
        .unwrap_or(0);
    let end = ops[start + 1..]
        .iter()
        .position(|op| matches!(op, IrOp::InsnStart { .. }))
        .map_or(ops.len(), |i| start + 1 + i);
    let instruction = &ops[start..end];
    if instruction
        .iter()
        .any(|op| matches!(op, IrOp::Jump { target } if *target == offset))
    {
        return AccessKind::Execute;
    }
    // An explicit source operand stays read-only when PUSH/CALL also stores
    // its implicit frame. Stack-address results identify those stores exactly.
    let writes = instruction.iter().any(|op| match op {
        IrOp::Store {
            addr: Val::Temp(addr),
            ..
        } => !instruction
            .iter()
            .any(|op| matches!(op, IrOp::Legacy(L::DirectStackAddress { dst, .. }) if dst == addr)),
        IrOp::Store { .. }
        | IrOp::AtomicRmw { .. }
        | IrOp::AtomicCas { .. }
        | IrOp::AtomicCasPair { .. } => true,
        _ => false,
    });
    if writes {
        AccessKind::Write
    } else {
        AccessKind::Read
    }
}

// Only retry-specialized alias walks need this scan. Ordinary alias loads keep
// their frozen completion behavior (decision-23 §6); unified scalar reads use
// the same ledger as Machine (80386 PRM §9.1, instruction restart).
#[cold]
#[inline(never)]
pub(super) fn unified_read_retry(ops: &[IrOp], rip: u64) -> bool {
    let start = ops
        .iter()
        .position(|op| matches!(op, IrOp::InsnStart { guest_addr } if *guest_addr == rip))
        .unwrap_or(0);
    ops[start + 1..]
        .iter()
        .take_while(|op| !matches!(op, IrOp::InsnStart { .. }))
        .any(|op| {
            matches!(
                op,
                IrOp::Legacy(
                    L::DirectStackAddress { .. }
                        | L::LoadSegment { .. }
                        | L::Transfer { .. }
                        | L::Bound { .. }
                )
            )
        })
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

pub(super) fn selector(cpu: &CpuState, segment: SegmentId) -> u16 {
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

#[inline]
fn segment_access(
    cpu: &CpuState,
    context: MachineContext,
    segment: SegmentId,
    offset: u64,
    bytes: u8,
    access: AccessKind,
) -> Result<u64, Exit> {
    if context.segment_class == crate::SegmentClass::Checked {
        return legacy::access(cpu, context.generation, segment, offset, bytes, access);
    }
    // Direct proof was derived at the context boundary (decision-23 §5).
    // Fault vectors: I286 PRM Appendix B MOV p. B-74; I386 PRM §14.6 Table 14-1.
    if offset
        .checked_add(u64::from(bytes.saturating_sub(1)))
        .is_none_or(|last| last > 0xffff)
    {
        return Err(legacy::segment_fault(
            cpu,
            if segment == SegmentId::Ss && context.generation == crate::CpuGeneration::I386 {
                12
            } else {
                13
            },
            offset,
            bytes,
            access,
        ));
    }
    Ok((u64::from(selector(cpu, segment)) << 4) + offset)
}

fn sp(cpu: &CpuState, context: MachineContext) -> u64 {
    cpu.gpr[RSP] & mask(context.stack_address_bits / 8)
}
fn adjust(cpu: &mut CpuState, context: MachineContext, bytes: u32) {
    cpu.write_gpr(
        RSP,
        sp(cpu, context).wrapping_add(u64::from(bytes)),
        context.stack_address_bits / 8,
    );
}

// A device read is identified by its visit as well as its bus address. The
// cursor restarts with the instruction, including reads served from the ledger.
// Writes retain their separate bit-63 namespace; descriptor snapshots use bit 62.
struct LegacyAccess<'a, const ROUTED: bool, const ORDER: u8> {
    access: &'a Access<ROUTED, ORDER>,
    read_index: Cell<u16>,
    writes: Option<&'a dyn Fn(u64, u8)>,
}
impl<'a, const ROUTED: bool, const ORDER: u8> LegacyAccess<'a, ROUTED, ORDER> {
    fn new(memory: &'a Memory) -> Self {
        Self {
            access: Access::new(memory),
            read_index: Cell::new(0),
            writes: None,
        }
    }
    fn skip_read(&self, cpu: &CpuState, mut linear: u64, mut bytes: u8) {
        while bytes != 0 {
            let (_, part) = legacy::physical_span(cpu, linear, bytes);
            self.read_index.set(self.read_index.get() + 1);
            linear = u64::from((linear as u32).wrapping_add(u32::from(part)));
            bytes -= part;
        }
    }
}
impl<const ROUTED: bool, const ORDER: u8> std::ops::Deref for LegacyAccess<'_, ROUTED, ORDER> {
    type Target = Access<ROUTED, ORDER>;
    fn deref(&self) -> &Self::Target {
        self.access
    }
}

// Instructions replay from their first read; task continuations resume saved
// stages, whose existing transfers retain address-based retry keys.
trait LegacyMemory<const ROUTED: bool, const ORDER: u8> {
    fn raw(&self) -> &Access<ROUTED, ORDER>;
    fn read_key(&self, addr: u64) -> u64;
    fn skip_read(&self, cpu: &CpuState, linear: u64, bytes: u8);
    fn note_write(&self, _addr: u64, _size: u8) {}
}
impl<const ROUTED: bool, const ORDER: u8> LegacyMemory<ROUTED, ORDER> for Access<ROUTED, ORDER> {
    fn raw(&self) -> &Access<ROUTED, ORDER> {
        self
    }
    fn read_key(&self, addr: u64) -> u64 {
        addr
    }
    fn skip_read(&self, _cpu: &CpuState, _linear: u64, _bytes: u8) {}
}
impl<const ROUTED: bool, const ORDER: u8> LegacyMemory<ROUTED, ORDER>
    for LegacyAccess<'_, ROUTED, ORDER>
{
    fn raw(&self) -> &Access<ROUTED, ORDER> {
        self.access
    }
    fn read_key(&self, addr: u64) -> u64 {
        let index = self.read_index.get();
        self.read_index.set(index + 1);
        addr | (u64::from(index) << 32)
    }
    fn skip_read(&self, cpu: &CpuState, linear: u64, bytes: u8) {
        LegacyAccess::skip_read(self, cpu, linear, bytes);
    }
    fn note_write(&self, addr: u64, size: u8) {
        if let Some(observe) = self.writes {
            observe(addr, size);
        }
    }
}

fn read<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
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
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    linear: u64,
    addr: u64,
    size: u8,
) -> Result<u64, Exit> {
    let key = mem.read_key(addr);
    if let Some(value) = cpu.mmio_part(cpu.rip, key) {
        return Ok(value & mask(size));
    }
    let value = mem.raw().read(addr, size).or_else(|trap| {
        // Decision-16 §3: replayed RAM accesses cannot consume the answer
        // belonging to a later MMIO transfer of this instruction.
        if trap == MemTrap::Mmio {
            if let Some(value) = cpu.pending_mmio.take() {
                let value = value & mask(size);
                cpu.record_mmio_part(cpu.rip, key, value);
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

// SDM Vol 2A §3.3 ENTER; 80386 PRM §17 ENTER: overlapping frame writes
// must not change a display on restart. Between an MMIO exit and resume,
// edits preserving exactly the architectural inputs at their read widths keep
// the attempt, so no acknowledged transfer repeats; changing an input restarts
// ENTER. BP/EBP is read at operand size and, for displays, stack size; SP/ESP
// is read at stack size for addressing and operand size for FrameTemp. Identity
// also retains the effective SS base/limit/B and limit rule, A20 policy, operand
// size, allocation imm16 and nesting imm8 modulo 32. Checked SS access rights
// participate only as used by ENTER's reads/writes. Unread bits are excluded.
//
// The first two slots retain the exact identity, then become the first two
// display snapshots, keeping identity in their unused key/high-value bits.
// Collection needs at most two identities plus four current byte answers (6)
// until both displays complete, then at most 29 snapshots plus four answers
// (33). Zero/one-display attempts retain two identities plus at most four write
// parts (6). Otherwise thirty snapshots plus 32 frame writes (33 if one crosses
// a bus boundary) need at most 63 slots. No identity phase adds to that peak.
const ENTER_DISPLAY_RETRY: u64 = 1 << 61;
const ENTER_DISPLAY_STARTS: u32 = 37;
const ENTER_DISPLAY_ENDS: u32 = 41;
const ENTER_DISPLAY_PENDING: u64 = 1 << 46;
const ENTER_IDENTITY_CONFIG: u32 = 47;
const ENTER_IDENTITY_CONFIG_MASK: u64 = 0x3fff << ENTER_IDENTITY_CONFIG;
const ENTER_IDENTITY_KEY: u64 = 0xffff_ffff | ENTER_IDENTITY_CONFIG_MASK;

fn enter_display_index(key: u64) -> u8 {
    ((key >> 32) & 0x1f) as u8
}

fn enter_display_key(index: u8) -> u64 {
    ENTER_DISPLAY_RETRY | (u64::from(index) << 32)
}

#[inline(never)]
fn prepare_enter_displays(
    cpu: &mut CpuState,
    context: MachineContext,
    size: u8,
    alloc: u16,
    level: u8,
) {
    let stack_bytes = context.stack_address_bits / 8;
    let bp_bytes = if level > 1 {
        size.max(stack_bytes)
    } else {
        size
    };
    let bp = cpu.gpr[RBP] & mask(bp_bytes);
    let sp = cpu.gpr[RSP] & mask(size.max(stack_bytes));
    let (base, limit, expand_down, readable, writable) =
        if context.segment_class == crate::SegmentClass::Direct {
            (u32::from(cpu.ss) << 4, 0xffff, false, true, true)
        } else {
            let ss = cpu.legacy.segments[SegmentId::Ss as usize];
            let protected = cpu.legacy.cr0 & 1 != 0;
            let code = ss.attributes & 8 != 0;
            (
                ss.base,
                ss.limit,
                protected && !code && ss.attributes & 4 != 0,
                !protected || !code || ss.attributes & 2 != 0,
                !protected || (!code && ss.attributes & 2 != 0),
            )
        };
    let config = u64::from(alloc)
        | (u64::from(level) << 16)
        | (u64::from(stack_bytes == 4) << 21)
        | (u64::from(size == 4) << 22)
        | (u64::from(cpu.legacy.a20_enabled != 0) << 23)
        | (u64::from(expand_down) << 24)
        | (u64::from(readable) << 25)
        | (u64::from(writable) << 26);
    let identities = [
        (bp | ((config & 0x3fff) << ENTER_IDENTITY_CONFIG), sp << 32),
        (
            u64::from(base) | ((config >> 14) << ENTER_IDENTITY_CONFIG),
            u64::from(limit) << 32,
        ),
    ];
    let matches = cpu.owns_mmio_retry(cpu.rip)
        && identities.iter().enumerate().all(|(i, &(key, value))| {
            cpu.mmio_parts[..usize::from(cpu.mmio_parts_len)]
                .iter()
                .any(|&(k, v)| {
                    k >> 61 == 1
                        && enter_display_index(k) == i as u8 + 1
                        && k & ENTER_IDENTITY_KEY == key
                        && v & !0xffff_ffff == value
                })
        });
    if !matches {
        // Abandon the whole attempt before collecting anything new, including
        // its display answers and acknowledged frame writes. Those device
        // answers belong to the earlier exit's evidence, not to this attempt.
        cpu.clear_mmio_parts();
        for (i, (key, value)) in identities.into_iter().enumerate() {
            cpu.record_mmio_part(
                cpu.rip,
                enter_display_key(i as u8 + 1) | ENTER_DISPLAY_PENDING | key,
                value,
            );
        }
    }
}

fn enter_display(cpu: &CpuState, index: u8) -> Option<u64> {
    cpu.mmio_parts[..usize::from(cpu.mmio_parts_len)]
        .iter()
        .find(|&&(key, _)| {
            key >> 61 == 1 && enter_display_index(key) == index && key & ENTER_DISPLAY_PENDING == 0
        })
        .map(|&(_, value)| u64::from(value as u32))
}

fn record_enter_display(
    cpu: &mut CpuState,
    visits: std::ops::Range<u16>,
    index: u8,
    address: u64,
    size: u8,
    partial: bool,
    value: u64,
) {
    let mut key = enter_display_key(index);
    let mut value = value;
    let mut kept = 0;
    for i in 0..usize::from(cpu.mmio_parts_len) {
        let entry = cpu.mmio_parts[i];
        if entry.0 >> 61 == 1 && enter_display_index(entry.0) == index {
            key |= entry.0 & ENTER_IDENTITY_KEY;
            value |= entry.1 & !0xffff_ffff;
            continue;
        }
        let visit = (entry.0 >> 32) as u16;
        if entry.0 >> 48 != 0 || !visits.contains(&visit) {
            cpu.mmio_parts[kept] = entry;
            kept += 1;
        } else {
            // An overlapped display reads uncovered bytes individually; an
            // unoverlapped display uses the normal bus-span split. Preserve
            // each completion's boundaries, not the RAM or staged bytes.
            let byte = (0..size)
                .find(|&byte| {
                    legacy::physical_span(cpu, address.wrapping_add(u64::from(byte)), 1).0
                        == u64::from(entry.0 as u32)
                })
                .expect("current attempt's display answer belongs to its bus span");
            let bytes = if partial {
                1
            } else {
                legacy::physical_span(cpu, address.wrapping_add(u64::from(byte)), size - byte).1
            };
            key |= 1 << (ENTER_DISPLAY_STARTS + u32::from(byte));
            key |= 1 << (ENTER_DISPLAY_ENDS + u32::from(byte + bytes - 1));
        }
    }
    cpu.mmio_parts_len = kept as u8;
    cpu.record_mmio_part(cpu.rip, key, value);
}

// Fault-only projection for helper ledgers, including an incomplete first
// display. Visits and write tags are not bus addresses; descriptor/transfer
// snapshots are not device answers. ENTER's compact provenance retains the
// original answer boundaries and the alias helper's saved A20 policy.
#[cold]
pub(crate) fn mmio_evidence(cpu: &CpuState) -> Box<[(u64, u64)]> {
    let entries = &cpu.mmio_parts[..usize::from(cpu.mmio_parts_len)];
    let identity = |index| {
        entries
            .iter()
            .find(|&&(key, _)| key >> 61 == 1 && enter_display_index(key) == index)
            .copied()
    };
    let display_geometry = identity(1).zip(identity(2)).map(|((bp, _), (base, _))| {
        let config = ((bp & ENTER_IDENTITY_CONFIG_MASK) >> ENTER_IDENTITY_CONFIG)
            | (((base & ENTER_IDENTITY_CONFIG_MASK) >> ENTER_IDENTITY_CONFIG) << 14);
        (bp as u32, base as u32, config)
    });
    let mut transfers = Vec::new();
    for &(key, value) in entries {
        if key & DESCRIPTOR_RETRY != 0 {
            continue;
        }
        if key >> 61 != 1 {
            transfers.push((u64::from(key as u32), value));
            continue;
        }
        let starts = (key >> ENTER_DISPLAY_STARTS) & 0xf;
        let ends = (key >> ENTER_DISPLAY_ENDS) & 0xf;
        for byte in 0..4 {
            if starts & (1 << byte) != 0 {
                let end = (byte..4)
                    .find(|end| ends & (1 << end) != 0)
                    .expect("display answer has an end");
                let (bp, base, config) = display_geometry.expect("ENTER identity");
                let size = if config & (1 << 22) != 0 { 4 } else { 2 };
                let stack_mask = if config & (1 << 21) != 0 {
                    u32::MAX
                } else {
                    0xffff
                };
                let offset =
                    bp.wrapping_sub(u32::from(enter_display_index(key)) * size) & stack_mask;
                let address = u64::from(base.wrapping_add(offset).wrapping_add(byte));
                let (bus, _) = legacy::physical_span_with_gate(
                    config & (1 << 23) != 0,
                    address,
                    (end - byte + 1) as u8,
                );
                transfers.push((bus, (value >> (byte * 8)) & mask((end - byte + 1) as u8)));
            }
        }
    }
    transfers.into_boxed_slice()
}

fn write<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
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
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    linear: u64,
    addr: u64,
    size: u8,
    value: u64,
) -> Result<(), Exit> {
    let value = value & mask(size);
    // The retry ledger is engine storage. Bit 63 distinguishes answered writes
    // from reads at the same physical bus address; it is never a guest address.
    let key = addr | (1 << 63);
    if cpu.mmio_part(cpu.rip, key).is_some() {
        mem.note_write(addr, size);
        return Ok(());
    }
    if let Err(trap) = mem.raw().write(addr, value, size) {
        if trap == MemTrap::Mmio && cpu.pending_mmio_write {
            cpu.pending_mmio_write = false;
            cpu.record_mmio_part(cpu.rip, key, value);
        } else {
            return Err(
                match trap_out(
                    cpu,
                    cpu.rip,
                    trap,
                    store_fault_address(mem, trap, addr, size),
                    size,
                    AccessKind::Write,
                    value,
                )
                .into_step()
                {
                    StepResult::Exit(exit) => crate::fault::qualify_legacy_bus(exit, linear, addr),
                    _ => unreachable!(),
                },
            );
        }
    }
    mem.note_write(addr, size);
    Ok(())
}

fn probe<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
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
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    linear: u64,
    addr: u64,
    size: u8,
) -> Result<(), Exit> {
    match mem.raw().probe_write(addr, size) {
        Ok(()) | Err(MemTrap::Mmio) => Ok(()),
        Err(trap) => Err(
            match trap_out(
                cpu,
                cpu.rip,
                trap,
                store_fault_address(mem, trap, addr, size),
                size,
                AccessKind::Write,
                0,
            )
            .into_step()
            {
                StepResult::Exit(exit) => crate::fault::qualify_legacy_bus(exit, linear, addr),
                _ => unreachable!(),
            },
        ),
    }
}

/// [`probe`] for a LOCK or implicitly locked operand that the A20 gate splits into
/// two bus pieces. Each piece must lie in one region, as an undivided locked operand
/// must (`Memory::atomic_rmw` keeps the containment miss): a piece over adjacent
/// regions reports the miss and its exit, as [`probe_physical`] did before data
/// transfers were performed across adjacent regions. A contained piece is probed
/// as [`probe_physical`] probes it.
#[cold]
fn probe_locked<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    addr: u64,
    size: u8,
) -> Result<(), Exit> {
    let (physical, part) = legacy::physical_span(cpu, addr, size);
    let checked = mem
        .raw()
        .region_kind(physical, part)
        .and_then(|_| mem.raw().probe_write(physical, part));
    match checked {
        Ok(()) | Err(MemTrap::Mmio) => {}
        Err(trap) => {
            return Err(
                match trap_out(
                    cpu,
                    cpu.rip,
                    trap,
                    store_fault_address(mem, trap, physical, part),
                    part,
                    AccessKind::Write,
                    0,
                )
                .into_step()
                {
                    StepResult::Exit(exit) => {
                        crate::fault::qualify_legacy_bus(exit, addr, physical)
                    }
                    _ => unreachable!(),
                },
            )
        }
    }
    if part < size {
        let next = u64::from((addr as u32).wrapping_add(u32::from(part)));
        probe_locked(cpu, mem, next, size - part)?;
    }
    Ok(())
}

#[cold]
fn store_fault_address<const ROUTED: bool, const ORDER: u8>(
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    trap: MemTrap,
    addr: u64,
    size: u8,
) -> u64 {
    if trap != MemTrap::Unmapped {
        return addr;
    }
    (0..size)
        .map(|i| addr + u64::from(i))
        .find(|&at| {
            matches!(
                mem.raw().probe_write_mode::<false>(at, 1),
                Err(MemTrap::Unmapped)
            )
        })
        .unwrap_or(addr)
}

#[cold]
fn operand_fault(exit: Exit, rip: u64, base: u64, fault: crate::ir::MemoryAccess) -> Exit {
    match super::classify_access(InterpResult::exit(exit), rip, base, fault).into_step() {
        StepResult::Exit(exit) => exit,
        _ => unreachable!(),
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

fn descriptor_address(cpu: &CpuState, value: u16) -> Option<u32> {
    // SDM Vol 3A §§3.4.2, 3.5.1; 286 PRM §6.6.2 pp. 6-12–13;
    // 386 PRM §§5.1.1–5.1.2, 6.3.1.2: the last descriptor byte must
    // lie inside the selected table. Only GDT index zero is null.
    let offset = u32::from(value & !7);
    let (base, limit) = if value & 4 == 0 {
        if value & !3 == 0 {
            return None;
        }
        (cpu.legacy.gdtr.base, u32::from(cpu.legacy.gdtr.limit))
    } else {
        let ldtr = cpu.legacy.ldtr;
        if ldtr.selector & !3 == 0 || ldtr.cache.unusable != 0 {
            return None;
        }
        (ldtr.cache.base, ldtr.cache.limit)
    };
    (offset + 7 <= limit).then(|| base.wrapping_add(offset))
}

fn descriptor<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    value: u16,
    generation: crate::CpuGeneration,
) -> Result<Descriptor, Exit> {
    let address = descriptor_address(cpu, value).ok_or_else(|| fault(cpu, 13, value & !3))?;
    descriptor_at(cpu, mem, address, generation)
}

fn descriptor_at<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    address: u32,
    generation: crate::CpuGeneration,
) -> Result<Descriptor, Exit> {
    let key = DESCRIPTOR_RETRY | u64::from(address);
    if let (Some(base_limit), Some(attributes)) = (
        cpu.mmio_part(cpu.rip, key),
        cpu.mmio_part(cpu.rip, key | (1 << 32)),
    ) {
        mem.skip_read(cpu, u64::from(address), 4);
        mem.skip_read(cpu, u64::from(address.wrapping_add(4)), 4);
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

#[cfg(test)]
thread_local! {
    static DESCRIPTOR_UPDATE: std::cell::Cell<Option<(u64, u64)>> = const { std::cell::Cell::new(None) };
}

fn descriptor_cas<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    address: u32,
    expected: u64,
    value: u64,
) -> Result<u64, Exit> {
    let linear = u64::from(address);
    let (physical, _) = legacy::physical_span(cpu, linear, 1);
    #[cfg(test)]
    DESCRIPTOR_UPDATE.with(|update| {
        if let Some((at, replacement)) = update.take() {
            mem.write(at, replacement, 8).unwrap();
        }
    });
    // SDM Vol 2A §3.3 LTR p. 3-576; Vol 3A §11.1.2.1 p. 11-3;
    // 386 PRM §7.6.1 p. 7-10: busy/accessed updates follow LOCK semantics.
    // A byte CAS preserves concurrent edits and is aligned even when
    // the descriptor table is not. Only the access-rights byte changes.
    // SDM Vol 3A §3.4.5.1 p. 3-13: descriptor memory may discard
    // writes. ROM uses the scalar CAS policy; device CAS remains refused.
    let observed = match mem.atomic_cas(physical, expected, value, 1) {
        Ok(observed) => observed,
        // Separate device read/write completions cannot implement CAS.
        Err(MemTrap::Mmio) => {
            return Err(refusal(
                cpu,
                crate::UnsupportedSystemOperation::LegacyInstruction,
            ));
        }
        Err(trap) => {
            return Err(
                match trap_out(cpu, cpu.rip, trap, physical, 1, AccessKind::Write, value)
                    .into_step()
                {
                    StepResult::Exit(exit) => {
                        crate::fault::qualify_legacy_bus(exit, linear, physical)
                    }
                    _ => unreachable!(),
                },
            );
        }
    };
    Ok(observed)
}

fn descriptor_bit<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    descriptor: &mut Descriptor,
    bit: u16,
) -> Result<bool, Exit> {
    if descriptor.cache.attributes & bit == 0 {
        if let Some(address) = descriptor.access_address {
            let expected = u64::from(descriptor.cache.attributes as u8);
            let value = expected | u64::from(bit);
            let observed = descriptor_cas(cpu, mem.raw(), address, expected, value)?;
            if observed != expected {
                descriptor.cache.attributes =
                    (descriptor.cache.attributes & !0xff) | observed as u16;
                return Ok(false);
            }
        }
    }
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
    descriptor.cache.attributes |= bit;
    Ok(true)
}

fn accessed<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    descriptor: &mut Descriptor,
) -> Result<(), Exit> {
    // SDM Vol 3A §§3.4.5.1, 11.1.2.1; 386 PRM §5.1.1 p. 5-4;
    // 286 PRM §7.4 p. 7-11: segment loads update A in memory and cache.
    while !descriptor_bit(cpu, mem, descriptor, 1)? {}
    Ok(())
}

fn load_system_segment<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    register: SystemRegister,
    value: u16,
    allowed_types: u16,
    generation: crate::CpuGeneration,
) -> Result<(), Exit> {
    // 286 PRM App. B pp. B-66/B-72; 386 PRM §17 pp. 17-98/17-108;
    // SDM Vol 2A §3.3 LLDT/LTR: GDT-only system descriptors at CPL 0.
    if cpu.legacy.cpl != 0 {
        return Err(fault(cpu, 13, 0));
    }
    if matches!(register, SystemRegister::Ldtr) && value & !3 == 0 {
        cpu.legacy.ldtr.selector = value;
        cpu.legacy.ldtr.cache.unusable = 1;
        return Ok(());
    }
    if value & 4 != 0 {
        return Err(fault(cpu, 13, value & !3));
    }
    let mut d = descriptor(cpu, mem, value, generation)?;
    loop {
        let attr = d.cache.attributes;
        if attr & 0x10 != 0 || allowed_types & (1 << (attr & 0xf)) == 0 {
            return Err(fault(cpu, 13, value & !3));
        }
        if attr & 0x80 == 0 {
            return Err(fault(cpu, 11, value & !3));
        }
        if matches!(register, SystemRegister::Ldtr) || descriptor_bit(cpu, mem, &mut d, 2)? {
            break;
        }
        // A competing processor may have acquired the TSS or replaced it.
        // Re-read and validate before trying to mark it busy again.
        d = descriptor(cpu, mem, value, generation)?;
    }
    let system = legacy::SystemSegment {
        selector: value,
        reserved: 0,
        cache: d.cache,
    };
    match register {
        SystemRegister::Ldtr => cpu.legacy.ldtr = system,
        SystemRegister::Tr => cpu.legacy.tr = system,
    }
    Ok(())
}

fn query_descriptor<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    selector: u16,
    query: DescriptorQuery,
    allowed_system_types: u16,
    allow_conforming: bool,
    generation: crate::CpuGeneration,
) -> Result<(), Exit> {
    // SDM Vol 2A §3.3 LAR/LSL; Vol 2C §5.2 VERR/VERW pp. 5-162–163;
    // 286 PRM §11.3 pp. 11-3–4; 386 PRM §6.3.6 pp. 6-18–19:
    // invalid selectors clear ZF; memory faults still escape. P is not tested,
    // and querying never loads a segment or sets the descriptor accessed bit.
    let Some(address) = descriptor_address(cpu, selector) else {
        cpu.flags.zf = false;
        return Ok(());
    };
    let cache = descriptor_at(cpu, mem, address, generation)?.cache;
    let attr = cache.attributes;
    let segment = attr & 0x10 != 0;
    let code = segment && attr & 8 != 0;
    let conforming = code && attr & 4 != 0;
    let dpl = (attr >> 5) & 3;
    let visible = if conforming {
        allow_conforming
    } else {
        dpl >= u16::from(cpu.legacy.cpl).max(selector & 3)
    };
    let valid_type = match query {
        DescriptorQuery::Read => segment && (!code || attr & 2 != 0),
        DescriptorQuery::Write => segment && !code && attr & 2 != 0,
        _ => segment || allowed_system_types & (1 << (attr & 0xf)) != 0,
    };
    if visible && valid_type {
        match query {
            DescriptorQuery::Rights { dst, size } => {
                // SDM Vol 2A §3.3 LAR: undefined bits 19:16 use zero as
                // deterministic engine policy. Access byte/flags retain their bits.
                write_reg(cpu, dst, u64::from(attr) << 8, size);
            }
            DescriptorQuery::Limit { dst, size } => {
                write_reg(cpu, dst, u64::from(cache.limit), size)
            }
            _ => {}
        }
    }
    cpu.flags.zf = visible && valid_type;
    Ok(())
}

fn install(cpu: &mut CpuState, segment: SegmentId, value: u16, cache: legacy::SegmentCache) {
    reload(cpu, segment, value);
    cpu.legacy.segments[segment as usize] = cache;
}

fn load_segment<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
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
    mem: &impl LegacyMemory<ROUTED, ORDER>,
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
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    context: MachineContext,
    count: usize,
    size: u8,
) -> Result<Vec<u64>, Exit> {
    let old = sp(cpu, context);
    let bytes = count as u8 * size;
    let start = old.wrapping_sub(u64::from(bytes)) & mask(context.stack_address_bits / 8);
    segment_access(cpu, context, SegmentId::Ss, start, bytes, AccessKind::Write)?;
    let mut addresses = Vec::with_capacity(count);
    for i in 1..=count {
        let offset =
            old.wrapping_sub(i as u64 * u64::from(size)) & mask(context.stack_address_bits / 8);
        let address = segment_access(cpu, context, SegmentId::Ss, offset, size, AccessKind::Write)?;
        probe(cpu, mem, address, size).map_err(|exit| {
            operand_fault(
                exit,
                cpu.rip,
                address,
                crate::ir::MemoryAccess::new(u16::from(size), false, true),
            )
        })?;
        addresses.push(address);
    }
    Ok(addresses)
}

fn push<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    context: MachineContext,
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
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    context: MachineContext,
    index: u64,
    size: u8,
) -> Result<u64, Exit> {
    let offset = sp(cpu, context).wrapping_add(index * u64::from(size))
        & mask(context.stack_address_bits / 8);
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
fn target(cpu: &CpuState, context: MachineContext, value: u64, size: u8) -> Result<u64, Exit> {
    let offset = value & mask(size);
    segment_access(cpu, context, SegmentId::Cs, offset, 1, AccessKind::Execute)?;
    Ok(offset)
}

/// Shared delivery for software INT, processor exceptions and queued IRQs.
pub(crate) fn deliver_interrupt<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    memory: &Memory,
    context: MachineContext,
    event: legacy::LegacyEvent,
    task_switch: &mut tasks::SwitchState,
) -> Result<(), Exit> {
    let mem = &LegacyAccess::<ROUTED, ORDER>::new(memory);
    task_switch.committed = false;
    task_switch.completed = false;
    if task_switch.context().is_some() {
        return tasks::resume(cpu, mem.raw(), task_switch);
    }
    deliver(cpu, mem, context, event, task_switch)
}

fn deliver_protected<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    context: MachineContext,
    event: legacy::LegacyEvent,
    task_switch: &mut tasks::SwitchState,
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
        mem.skip_read(cpu, u64::from(address), 4);
        value as u32
    } else {
        read(cpu, mem, u64::from(address), 4)? as u32
    };
    let high = if let Some(value) = cpu.mmio_part(cpu.rip, key | (1 << 32)) {
        mem.skip_read(cpu, u64::from(address.wrapping_add(4)), 4);
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
        return tasks::event(
            cpu,
            mem.raw(),
            context,
            (low >> 16) as u16,
            event,
            task_switch,
        );
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
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    context: MachineContext,
    event: legacy::LegacyEvent,
    task_switch: &mut tasks::SwitchState,
) -> Result<(), Exit> {
    if context.regime == legacy::MachineRegime::Protected {
        return deliver_protected(cpu, mem, context, event, task_switch);
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
    target(cpu, context, ip, 4)?;
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
// A transfer over adjacent RAM/ROM regions is performed (the `*_spanning` paths).
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
                .read_ram_checked_spanning::<ROUTED>(physical, head)
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
            if let Err(trap) = self.memory.probe_write_spanning::<ROUTED>(physical, part) {
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
                .write_ram_checked_spanning_observed::<ROUTED>(physical, head, self.memory)
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
        &LegacyAccess::<ROUTED, ORDER>::new(memory),
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

struct InstructionView<'a> {
    guest_start: u64,
    guest_len: u32,
    ops: &'a [IrOp],
}

fn atomic_device_exit<const ROUTED: bool, const ORDER: u8>(
    cpu: &CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    ir: &InstructionView<'_>,
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
        if mem
            .raw()
            .fetch_code(physical, &mut bytes[i..i + 1])
            .is_err()
        {
            break;
        }
    }
    Exit::unknown_instruction(cpu.rip, bytes, ir.guest_len as u8)
}

pub(crate) fn supported(op: &IrOp) -> bool {
    matches!(
        op,
        IrOp::ReadSelector { .. }
            | IrOp::SegLimitCheck { .. }
            | IrOp::InsnStart { .. }
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
            | IrOp::LoopCx { .. }
            | IrOp::SetIf { .. }
            | IrOp::SetDf { .. }
            | IrOp::SetCf { .. }
            | IrOp::Lahf
            | IrOp::Sahf
            | IrOp::Hlt
            | IrOp::PortIo { .. }
            | IrOp::Trap { .. }
            | IrOp::Bcd { .. }
            | IrOp::Fence { .. }
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
    task_switch: &mut tasks::SwitchState,
) -> Result<Outcome, Exit> {
    #[cfg(debug_assertions)]
    debug_assert!(
        ir.verify_temp_defs().is_ok(),
        "undefined interpreter temporary: {:?}",
        ir.verify_temp_defs()
    );
    let context = ir.legacy_context.expect("legacy interpreter context");
    execute_instruction::<ROUTED, ORDER>(
        ir,
        &InstructionView {
            guest_start: ir.guest_start,
            guest_len: ir.guest_len,
            ops: &ir.ops,
        },
        cpu,
        &LegacyAccess::new(memory),
        scratch.zeroed(ir.temp_count as usize),
        context,
        task_switch,
    )
}

fn execute_instruction<const ROUTED: bool, const ORDER: u8>(
    block: &IrBlock,
    ir: &InstructionView<'_>,
    cpu: &mut CpuState,
    mem: &LegacyAccess<'_, ROUTED, ORDER>,
    temps: &mut [u64],
    context: MachineContext,
    task_switch: &mut tasks::SwitchState,
) -> Result<Outcome, Exit> {
    mem.read_index.set(0);
    let checked = context.segment_class == crate::SegmentClass::Checked;
    if checked && !ir.ops.iter().all(supported) {
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyInstruction,
        ));
    }
    let checkpoint = matches!(ir.ops.get(1), Some(IrOp::Legacy(L::Checkpoint)));
    let before = if checkpoint {
        Some((
            cpu.gpr,
            cpu.flags,
            cpu.rflags_system,
            cpu.legacy,
            [cpu.cs, cpu.ds, cpu.es, cpu.ss, cpu.fs, cpu.gs],
        ))
    } else {
        None
    };
    task_switch.committed = false;
    task_switch.completed = false;
    let mut result = if task_switch.context().is_some() {
        let suppress_debug = task_switch.suppress_debug();
        tasks::resume(cpu, mem.raw(), task_switch).map(|()| Outcome {
            retired: true,
            context_changed: true,
            preserve_rf: true,
            suppress_debug,
            ..Outcome::default()
        })
    } else {
        let mut execution = InstructionExecution {
            ir,
            mem,
            context,
            task_switch,
            outcome: Outcome {
                retired: true,
                ..Outcome::default()
            },
        };
        super::walk_instruction::<ROUTED, ORDER>(
            block,
            ir.ops,
            ir.guest_start,
            cpu,
            mem.access,
            temps,
            &mut execution,
        )
        .map(|()| execution.outcome)
    };
    if task_switch.committed
        && matches!(
            result,
            Err(Exit::UnsupportedSystemContext {
                operation: crate::UnsupportedSystemOperation::LegacyTaskDebugTrap,
                ..
            })
        )
    {
        result = Ok(Outcome {
            exit: result.err(),
            retired: true,
            context_changed: true,
            preserve_rf: true,
            ..Outcome::default()
        });
    }
    if result.is_err() && !task_switch.committed {
        if let Some(before) = before {
            cpu.gpr = before.0;
            cpu.flags = before.1;
            cpu.rflags_system = before.2;
            cpu.legacy = before.3;
            [cpu.cs, cpu.ds, cpu.es, cpu.ss, cpu.fs, cpu.gs] = before.4;
        }
        cpu.rip = ir.guest_start;
        if let Err(exit) = result {
            // Match the shared walk's evidence only where the instruction's
            // memory effects are represented. Opaque frame/system helpers keep
            // their existing evidence; generic scalar analysis cannot prove
            // that those helpers left RAM untouched.
            let scalar_evidence = context.regime == legacy::MachineRegime::Real
                && ir.ops.iter().all(|op| match op {
                    IrOp::Legacy(op) => matches!(
                        op,
                        L::SharedWalk
                            | L::Checkpoint
                            | L::Address { .. }
                            | L::ReadSelector { .. }
                            | L::DirectStackAddress { .. }
                            | L::Push { .. }
                            | L::Pop { .. }
                            | L::AdjustStack { .. }
                            | L::Transfer { cs: None, .. }
                            | L::Return { far: false, .. }
                            | L::Loop { .. }
                            | L::String { .. }
                            | L::Coprocessor { .. }
                    ),
                    _ => true,
                });
            let mut exit = if scalar_evidence {
                super::finish_instruction_fault::<true>(
                    exit,
                    ir.ops,
                    ir.guest_start,
                    Some(ir.guest_start + u64::from(ir.guest_len)),
                    crate::CpuMode::Real16,
                    cpu,
                    mem.0.normal(),
                    temps,
                    false,
                )
            } else {
                exit
            };
            if let Exit::Exception {
                metadata: Some(m), ..
            }
            | Exit::UnmappedMemory {
                metadata: Some(m), ..
            } = &mut exit
            {
                if cpu.owns_mmio_retry(cpu.rip) {
                    m.progress.acknowledged = mmio_evidence(cpu);
                }
            }
            result = Err(exit);
        }
    } else if result
        .as_ref()
        .is_ok_and(|outcome| outcome.retired && !outcome.preserve_rf)
    {
        // SDM Vol 3B §20.3.1.1; 80386 PRM §12.3.1.1: retirement clears RF.
        cpu.rflags_system &= !(1 << 16);
    }
    result
}

#[derive(Clone, Copy)]
pub(crate) enum BlockWalk {
    Shared,
    Direct,
    Checked,
    Protected,
}
impl BlockWalk {
    pub(crate) fn for_ir<const SYSTEM: bool>(ir: &IrBlock) -> Self {
        if SYSTEM
            && ir
                .legacy_context
                .is_some_and(|c| c.regime == legacy::MachineRegime::Protected)
        {
            return Self::Protected;
        }
        if ir
            .legacy_context
            .is_some_and(|c| c.segment_class == crate::SegmentClass::Direct)
        {
            if matches!(ir.ops.last(), Some(IrOp::Legacy(L::SharedWalk))) {
                Self::Shared
            } else {
                Self::Direct
            }
        } else {
            Self::Checked
        }
    }
}

pub(crate) struct Progress {
    pub sti_shadow: bool,
    pub ss_shadow: bool,
    pub steps: u64,
    pub retired: u64,
    pub outcome: Option<Result<Outcome, Exit>>,
}

impl Progress {
    #[inline]
    pub(crate) fn into_outcome(self) -> Result<Outcome, Exit> {
        self.outcome.unwrap_or_else(|| {
            Ok(Outcome {
                arm_sti: self.sti_shadow,
                arm_ss: self.ss_shadow,
                ..Outcome::default()
            })
        })
    }
}

/// Instruction boundaries for system helpers and their transaction state. All
/// instruction bodies use the shared opcode walk (spec.md §8.1).
pub(crate) fn execute_instructions<const ROUTED: bool, const ORDER: u8>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    memory: &Memory,
    scratch: &mut impl ScratchBuffer,
    limit: u64,
    exit_word: &std::sync::atomic::AtomicU64,
    task_switch: &mut tasks::SwitchState,
) -> Progress {
    debug_assert!(task_switch.context().is_none(), "Direct task continuation");
    #[cfg(debug_assertions)]
    debug_assert!(ir.verify_temp_defs().is_ok(), "{:?}", ir.verify_temp_defs());
    let context = ir.legacy_context.expect("Direct context");
    let temps = scratch.reused(ir.temp_count as usize);
    let mem = &LegacyAccess::new(memory);
    let mut steps = 0;
    let mut retired = 0;
    let mut start = 0;
    let outcome = loop {
        let end = ir.ops[start + 1..]
            .iter()
            .position(|op| matches!(op, IrOp::InsnStart { .. }))
            .map_or(ir.ops.len(), |offset| start + 1 + offset);
        let IrOp::InsnStart { guest_addr } = ir.ops[start] else {
            unreachable!("instruction marker")
        };
        let next = match ir.ops.get(end) {
            Some(IrOp::InsnStart { guest_addr }) => *guest_addr,
            _ => ir.guest_start + u64::from(ir.guest_len),
        };
        cpu.rip = guest_addr;
        match execute_instruction::<ROUTED, ORDER>(
            ir,
            &InstructionView {
                guest_start: guest_addr,
                guest_len: (next - guest_addr) as u32,
                ops: &ir.ops[start..end],
            },
            cpu,
            mem,
            temps,
            context,
            task_switch,
        ) {
            Err(exit) => break Err(exit),
            Ok(outcome) => {
                steps += 1;
                retired += u64::from(outcome.retired);
                if cpu.mmio_retry_active {
                    cpu.finish_mmio_retry(None);
                }
                // Never execute stale successors after a code write. Dispatch
                // drains the notification and invalidates physical byte spans.
                if (outcome.retired && end == ir.ops.len())
                    || steps == limit
                    || outcome.exit.is_some()
                    || outcome.context_changed
                    || outcome.arm_sti
                    || outcome.arm_ss
                    || exit_word.load(std::sync::atomic::Ordering::Relaxed) != 0
                    || memory.has_dirty_code()
                {
                    break Ok(outcome);
                }
                if !outcome.retired {
                    continue;
                }
            }
        }
        start = end;
    };
    Progress {
        sti_shadow: false,
        ss_shadow: false,
        steps,
        retired,
        outcome: Some(outcome),
    }
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_shared_string<const ROUTED: bool, const STOP: bool, M: InterpreterMemory>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    mem: &Access<ROUTED, 0, M>,
    kind: StrOp,
    elem: u8,
    rep: RepKind,
    ip: u64,
    address_bits: u8,
    source: SegmentId,
    exit_word: Option<&std::sync::atomic::AtomicU64>,
    steps: &mut StepContext,
    shadows: &mut ShadowState,
    retired: u64,
) -> Option<WalkResult> {
    if !ir
        .legacy_context
        .is_some_and(|c| c.segment_class == crate::SegmentClass::Direct)
    {
        return Some(WalkResult::exit(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyExecution,
        )));
    }
    let src_base = u64::from(selector(cpu, source)) << 4;
    let dst_base = u64::from(cpu.es) << 4;
    let result = super::string_run_real_bounded::<ROUTED, STOP, _>(
        cpu,
        mem,
        kind,
        elem,
        rep,
        ip,
        address_bits,
        src_base,
        dst_base,
        exit_word,
        steps,
        shadows,
        retired,
    );
    if matches!(result, Some(WalkResult::Fault)) {
        return Some(shared_string_fault(mem, elem));
    }
    result
}

#[cold]
fn shared_string_fault<const ROUTED: bool, M: InterpreterMemory>(
    mem: &Access<ROUTED, 0, M>,
    elem: u8,
) -> WalkResult {
    let mut exit = super::result::consume_current().expect("string fault");
    if let Exit::UnmappedMemory {
        addr,
        access: AccessKind::Write,
        metadata: Some(details),
    } = &mut exit
    {
        if details.origin == crate::FaultOrigin::GuestMap {
            let first = (0..elem)
                .map(|i| addr.wrapping_add(u64::from(i)))
                .find(|&at| matches!(mem.probe_write_mode::<false>(at, 1), Err(MemTrap::Unmapped)))
                .unwrap_or(*addr);
            *addr = first;
            details.address = crate::Knowledge::Known(crate::FaultAddress::new(
                first,
                crate::AddressPrecision::TransferAddress,
                crate::AddressSpace::Linear,
            ));
        }
    }
    WalkResult::exit(exit)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_shared<const ROUTED: bool, const ORDER: u8, const DIRECT20: bool>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    memory: &Memory,
    scratch: &mut impl ScratchBuffer,
    limit: u64,
    exit_word: &std::sync::atomic::AtomicU64,
    sti_shadow: bool,
    ss_shadow: bool,
    pending_irq: bool,
) -> Progress {
    let code_base = if DIRECT20 {
        (((u64::from(cpu.cs) << 4) + ir.guest_start) & 0xfffff).wrapping_sub(ir.guest_start)
    } else {
        u64::from(cpu.cs) << 4
    };
    let mut steps = StepContext::new(limit);
    steps.pending_irq = pending_irq;
    let mut info = RetireInfo {
        code_base,
        exit_request: Some(exit_word),
        sti_shadow,
        ss_shadow,
        ..Default::default()
    };
    let result = super::interpret_direct_steps::<ROUTED, ORDER, DIRECT20>(
        ir, cpu, memory, scratch, &mut info, &mut steps,
    );
    if info.retired != 0 {
        cpu.rflags_system &= !(1 << 16);
    }
    // A twenty-bit ModeChange is a NEC V20/V30 switch to 8080 mode: the
    // dispatcher decides the next boundary (decision-23 C4a).
    if matches!(result, BlockResult::Continue)
        || (!DIRECT20 && matches!(result, BlockResult::ModeChange))
        || (matches!(
            result,
            BlockResult::Exit(super::result::InterpExit::BudgetExhausted)
        ) && !cpu.mmio_retry_active
            && cpu.port_string_in.is_none())
    {
        return Progress {
            steps: steps.steps,
            retired: info.retired,
            sti_shadow: info.sti_shadow,
            ss_shadow: info.ss_shadow,
            outcome: None,
        };
    }
    let outcome = match result {
        BlockResult::Continue | BlockResult::ModeChange => Ok(Outcome {
            arm_sti: info.sti_shadow,
            arm_ss: info.ss_shadow,
            ..Outcome::default()
        }),
        BlockResult::Exit(exit) => match exit.into_exit() {
            Exit::BudgetExhausted => Ok(Outcome {
                arm_sti: info.sti_shadow,
                arm_ss: info.ss_shadow,
                ..Outcome::default()
            }),
            exit @ (Exit::Hlt | Exit::PortIo { .. } | Exit::Interrupted) => Ok(Outcome {
                exit: Some(exit),
                arm_sti: info.sti_shadow,
                arm_ss: info.ss_shadow,
                ..Outcome::default()
            }),
            // The twenty-bit walk moves every data byte through a one-byte
            // transfer (`interp::i8086`; lift/i8086.rs lowers loads, stores and
            // strings to its Wrapped primitives), and a one-byte access lies in
            // one region or in none. No containment miss over adjacent regions
            // reaches this arm, so it keeps its pre-split shape.
            exit if DIRECT20 => Err(qualify_shared_exit(exit)),
            exit => {
                let exit = match split_shared::<ROUTED, ORDER, DIRECT20>(
                    exit, ir, cpu, memory, scratch, &mut info, &mut steps,
                ) {
                    Ok(progress) => return progress,
                    Err(exit) => exit,
                };
                Err(qualify_shared_exit(exit))
            }
        },
    };
    Progress {
        sti_shadow: info.sti_shadow,
        ss_shadow: info.ss_shadow,
        steps: steps.steps,
        retired: info.retired,
        outcome: Some(outcome),
    }
}

#[inline(always)]
fn qualify_shared_exit(exit: Exit) -> Exit {
    match exit {
        Exit::UnmappedMemory { addr, .. }
        | Exit::MmioRead { addr, .. }
        | Exit::MmioWrite { addr, .. } => crate::fault::qualify_legacy_bus(exit, addr, addr),
        _ => exit,
    }
}

/// [`execute_shared`]'s re-execution of the rest of its block after a containment
/// miss over adjacent RAM/ROM regions (`interp::split_direct`): `Ok` with the
/// block's progress over both walks, or `Err` with the first attempt's exit. The
/// twenty-bit walk does not reach it (see `execute_shared`).
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn split_shared<const ROUTED: bool, const ORDER: u8, const DIRECT20: bool>(
    first: Exit,
    ir: &IrBlock,
    cpu: &mut CpuState,
    memory: &Memory,
    scratch: &mut impl ScratchBuffer,
    info: &mut RetireInfo,
    steps: &mut StepContext,
) -> Result<Progress, Exit> {
    let result = super::split_direct::<ROUTED, ORDER, DIRECT20>(
        first, ir, cpu, memory, scratch, info, steps,
    )?;
    if info.retired != 0 {
        cpu.rflags_system &= !(1 << 16);
    }
    let progress = |outcome| Progress {
        steps: steps.steps,
        retired: info.retired,
        sti_shadow: info.sti_shadow,
        ss_shadow: info.ss_shadow,
        outcome,
    };
    let boundary = Outcome {
        arm_sti: info.sti_shadow,
        arm_ss: info.ss_shadow,
        ..Outcome::default()
    };
    Ok(match result {
        BlockResult::Continue | BlockResult::ModeChange => progress(None),
        BlockResult::Exit(exit) => match exit.into_exit() {
            Exit::BudgetExhausted if !cpu.mmio_retry_active && cpu.port_string_in.is_none() => {
                progress(None)
            }
            Exit::BudgetExhausted => progress(Some(Ok(boundary))),
            exit @ (Exit::Hlt | Exit::PortIo { .. } | Exit::Interrupted) => {
                progress(Some(Ok(Outcome {
                    exit: Some(exit),
                    ..boundary
                })))
            }
            exit => progress(Some(Err(qualify_shared_exit(exit)))),
        },
    })
}

// Keep the new operation shapes out of existing walks while sharing their source.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_shared20<const ROUTED: bool, const ORDER: u8>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    memory: &Memory,
    scratch: &mut impl ScratchBuffer,
    limit: u64,
    exit_word: &std::sync::atomic::AtomicU64,
    sti_shadow: bool,
    ss_shadow: bool,
    pending_irq: bool,
) -> Progress {
    execute_shared::<ROUTED, ORDER, true>(
        ir,
        cpu,
        memory,
        scratch,
        limit,
        exit_word,
        sti_shadow,
        ss_shadow,
        pending_irq,
    )
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_block<const ROUTED: bool, const ORDER: u8>(
    walk: BlockWalk,
    ir: &IrBlock,
    cpu: &mut CpuState,
    memory: &Memory,
    scratch: &mut impl ScratchBuffer,
    limit: u64,
    exit_word: &std::sync::atomic::AtomicU64,
    task_switch: &mut tasks::SwitchState,
    sti_shadow: bool,
    ss_shadow: bool,
    pending_irq: bool,
) -> Progress {
    match walk {
        BlockWalk::Shared => execute_shared::<ROUTED, ORDER, false>(
            ir,
            cpu,
            memory,
            scratch,
            limit,
            exit_word,
            sti_shadow,
            ss_shadow,
            pending_irq,
        ),
        BlockWalk::Direct => execute_instructions::<ROUTED, ORDER>(
            ir,
            cpu,
            memory,
            scratch,
            limit,
            exit_word,
            task_switch,
        ),
        BlockWalk::Checked | BlockWalk::Protected => {
            execute_checked::<ROUTED, ORDER>(ir, cpu, memory, scratch, task_switch)
        }
    }
}

#[inline(always)]
pub(crate) fn execute_checked<const ROUTED: bool, const ORDER: u8>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    memory: &Memory,
    scratch: &mut impl ScratchBuffer,
    task_switch: &mut tasks::SwitchState,
) -> Progress {
    let outcome = execute::<ROUTED, ORDER>(ir, cpu, memory, scratch, task_switch);
    Progress {
        sti_shadow: false,
        ss_shadow: false,
        steps: u64::from(outcome.is_ok()),
        retired: u64::from(outcome.as_ref().is_ok_and(|o| o.retired)),
        outcome: Some(outcome),
    }
}

// Application and Machine use the same transactional instruction helpers.
// The compatibility personality has no retained system state and ignores A20.
pub(super) fn execute_shared_op<const ROUTED: bool, const ORDER: u8>(
    op: &L,
    cpu: &mut CpuState,
    memory: &Access<ROUTED, ORDER, impl InterpreterMemory>,
    temps: &mut [u64],
    context: Option<MachineContext>,
    ip: u64,
    next: u64,
) -> Result<Outcome, Exit> {
    let saved = context.is_none().then_some(cpu.legacy);
    let context = context.unwrap_or_else(crate::lift::legacy::alias_context);
    if saved.is_some() {
        cpu.legacy = legacy::LegacyCpuState {
            a20_enabled: 1,
            ..Default::default()
        };
    }
    cpu.rip = ip;
    if matches!(op, L::CheckTaskReturn) {
        if let Some(saved) = saved {
            cpu.legacy = saved;
            return Ok(Outcome::default());
        }
    }
    let ops = [IrOp::Legacy(op.clone())];
    let observe = |addr, size| {
        if !matches!(
            memory.0.normal().region_kind(addr, size),
            Ok(crate::RegionKind::Rom)
        ) {
            memory.0.note_block_write(addr, usize::from(size));
        }
    };
    let mut access = LegacyAccess::new(memory.0.normal());
    access.writes = Some(&observe);
    let mut outcome = Outcome {
        retired: true,
        ..Outcome::default()
    };
    let result = execute_op::<ROUTED, ORDER>(
        &ops[0],
        &InstructionView {
            guest_start: ip,
            guest_len: (next - ip) as u32,
            ops: &ops,
        },
        cpu,
        &access,
        temps,
        context,
        &mut tasks::SwitchState::default(),
        &mut outcome,
    )
    .map(|terminated| {
        if !terminated {
            cpu.rip = next;
        }
        outcome
    });
    if let Some(saved) = saved {
        cpu.legacy = saved;
        if result.as_ref().is_ok_and(|outcome| outcome.context_changed) {
            crate::legacy::REAL16_ALIAS.derive_caches(cpu);
        }
        if result.is_ok() {
            cpu.rip &= 0xffff;
        }
        if result.is_ok() && matches!(op, L::Flags { pop: true, .. } | L::Iret { .. }) {
            // The frozen compatibility personality does not acquire single-step.
            cpu.rflags_system &= !0x7100;
        }
    }
    result
}

// Keep op decoding and the exceptional context out of the scalar walk.
#[inline(never)]
pub(super) fn execute_loop_op(
    cpu: &mut CpuState,
    op: &L,
    context: Option<MachineContext>,
    ip: u64,
) -> Result<(), Exit> {
    let L::Loop {
        kind,
        count_bits,
        check_target,
        taken,
        next,
    } = op
    else {
        unreachable!("loop operation");
    };
    if *check_target {
        return execute_loop_checked_op(cpu, op, context, ip);
    }
    execute_loop(
        cpu,
        crate::lift::legacy::alias_context(),
        *kind,
        *count_bits,
        false,
        *taken,
        *next,
    )
}

#[cold]
#[inline(never)]
fn execute_loop_checked_op(
    cpu: &mut CpuState,
    op: &L,
    context: Option<MachineContext>,
    ip: u64,
) -> Result<(), Exit> {
    let L::Loop {
        kind,
        count_bits,
        taken,
        next,
        ..
    } = op
    else {
        unreachable!("loop operation");
    };
    cpu.rip = ip;
    execute_loop(
        cpu,
        context.unwrap_or_else(crate::lift::legacy::alias_context),
        *kind,
        *count_bits,
        true,
        *taken,
        *next,
    )
}

#[inline(always)]
pub(super) fn execute_loop(
    cpu: &mut CpuState,
    context: MachineContext,
    kind: crate::ir::LoopKind,
    count_bits: u8,
    check_target: bool,
    taken: u32,
    next: u32,
) -> Result<(), Exit> {
    if count_bits == 16 {
        execute_loop_count::<2>(cpu, context, kind, check_target, taken, next)
    } else {
        execute_loop_count::<4>(cpu, context, kind, check_target, taken, next)
    }
}

#[inline(always)]
fn execute_loop_count<const SIZE: u8>(
    cpu: &mut CpuState,
    context: MachineContext,
    kind: crate::ir::LoopKind,
    check_target: bool,
    taken: u32,
    next: u32,
) -> Result<(), Exit> {
    let count = cpu.gpr[1] & mask(SIZE);
    if matches!(kind, crate::ir::LoopKind::Jcxz) {
        cpu.rip = loop_ip(cpu, context, check_target, count == 0, taken, next)?;
        return Ok(());
    }
    let count = count.wrapping_sub(1) & mask(SIZE);
    let take = match kind {
        crate::ir::LoopKind::Loop => count != 0,
        crate::ir::LoopKind::Loope => count != 0 && cpu.flags.zf,
        crate::ir::LoopKind::Loopne => count != 0 && !cpu.flags.zf,
        crate::ir::LoopKind::Jcxz => unreachable!(),
    };
    let ip = loop_ip(cpu, context, check_target, take, taken, next)?;
    cpu.write_gpr(1, count, SIZE);
    cpu.rip = ip;
    Ok(())
}

#[inline(always)]
fn loop_ip(
    cpu: &CpuState,
    context: MachineContext,
    check_target: bool,
    take: bool,
    taken: u32,
    next: u32,
) -> Result<u64, Exit> {
    let ip = u64::from(if take { taken } else { next });
    // SDM Vol 2A §3.3 LOOP: a faulting taken target precedes count commit.
    if take && check_target {
        target(cpu, context, ip, 4)?;
    }
    Ok(ip)
}

// The shared opcode iterator delegates only the Machine-specific accesses and
// system helpers here; scalar arithmetic has one interpretation (spec.md §8.1).
#[allow(clippy::too_many_arguments)]
fn execute_op<const ROUTED: bool, const ORDER: u8>(
    op: &IrOp,
    ir: &InstructionView<'_>,
    cpu: &mut CpuState,
    mem: &LegacyAccess<'_, ROUTED, ORDER>,
    temps: &mut [u64],
    context: MachineContext,
    task_switch: &mut tasks::SwitchState,
    outcome: &mut Outcome,
) -> Result<bool, Exit> {
    let next = ir.guest_start.wrapping_add(u64::from(ir.guest_len)) & 0xffff_ffff;
    let result = match op {
        IrOp::BoundGate {
            index,
            lower,
            upper,
            ..
        } => {
            let index = read_val(*index, temps) as i16;
            if index < read_val(*lower, temps) as i16 || index > read_val(*upper, temps) as i16 {
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
                            return Err(crate::fault::qualify_legacy_bus(exit, linear, physical));
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
                probe_locked(cpu, mem, linear, *size)?;
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
                if context.generation == crate::CpuGeneration::I286
                    && context.regime == crate::MachineRegime::Protected
                    && (crate::x87::esc_output(&bytes[..usize::from(*len)])
                        || bytes[..usize::from(*len)].last() == Some(&0x9b))
                {
                    return Err(refusal(
                        cpu,
                        crate::UnsupportedSystemOperation::LegacyInstruction,
                    ));
                }
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
                if context.generation == crate::CpuGeneration::I286
                    && matches!(
                        kind,
                        crate::x87::FpuKind::Fnstcw | crate::x87::FpuKind::FnstswMem
                    )
                {
                    // I286/287 PRM Part II Appendix B item 10 p. B-2
                    // reads the absent bus through the memory status store;
                    // decision-25 selects the all-high word default.
                    write(cpu, mem, read_val(*addr, temps), 2, 0xffff)?;
                }
                return Ok(false);
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
        IrOp::SegLimitCheck {
            offset,
            size,
            vector,
            ..
        } => {
            if read_val(*offset, temps)
                .checked_add(u64::from(*size))
                .is_none_or(|end| end > 0x1_0000)
            {
                return Err(legacy::segment_fault(
                    cpu,
                    *vector,
                    read_val(*offset, temps),
                    *size,
                    scalar_limit_access(ir.ops, cpu.rip, *offset),
                ));
            }
            None
        }
        IrOp::Load {
            fault,
            dst,
            addr,
            size,
        } => {
            let address = read_val(*addr, temps);
            temps[*dst as usize] = read(cpu, mem, address, *size)
                .map_err(|exit| operand_fault(exit, cpu.rip, address, *fault))?;
            None
        }
        IrOp::Store {
            fault,
            addr,
            src,
            size,
            ..
        } => {
            write(
                cpu,
                mem,
                read_val(*addr, temps),
                *size,
                read_val(*src, temps),
            )
            .map_err(|exit| operand_fault(exit, cpu.rip, read_val(*addr, temps), *fault))?;
            None
        }
        IrOp::LoopCx {
            kind,
            taken,
            fallthrough,
        } => {
            execute_loop(
                cpu,
                context,
                *kind,
                16,
                false,
                *taken as u32,
                *fallthrough as u32,
            )?;
            return Ok(true);
        }
        IrOp::Jump { target: ip } => {
            cpu.rip = target(cpu, context, read_val(*ip, temps), 4)?;
            return Ok(true);
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
            return Ok(true);
        }
        IrOp::SetIf { value } => {
            outcome.arm_sti = *value && !cpu.flags.if_;
            cpu.flags.if_ = *value;
            None
        }
        IrOp::Hlt => {
            cpu.rip = next;
            outcome.exit = Some(Exit::Hlt);
            return Ok(true);
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
            return Ok(true);
        }
        IrOp::Trap { vector, .. } => {
            return Err(exception(cpu, *vector));
        }
        IrOp::Legacy(op) => {
            match op {
                L::Checkpoint | L::SharedWalk => {}
                L::DirectStackAddress {
                    dst,
                    offset,
                    bytes,
                    vector,
                } => {
                    temps[*dst as usize] = direct_stack_address(
                        cpu,
                        read_val(*offset, temps),
                        *bytes,
                        *vector,
                        cpu.rip,
                    )
                    .map_err(|exit| stack_fault_access(exit, ir.ops, cpu.rip))?;
                }
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
                    // Generation/decoder #UD follows the MSW gate. I286
                    // larger memory outputs keep the data channel idle
                    // (I286/287 PRM Part II Ch. 2 p. 2-49, Ch. 3 p. 3-1;
                    // decision-25). The I386 PRM §11.1.2 pp. 11-1–11-2
                    // supplies no absent output value: retain its handoff.
                    if !*wait
                        && cpu.fpu_model == crate::FpuModel::None
                        && !ir
                            .ops
                            .iter()
                            .any(|op| matches!(op, IrOp::Trap { vector: 6, .. }))
                        && crate::x87::esc_output(&bytes[..usize::from(*len)])
                    {
                        if context.generation != crate::CpuGeneration::I286 {
                            return Err(Exit::unknown_instruction(cpu.rip, *bytes, *len));
                        }
                        if bytes[..usize::from(*len)].ends_with(&[0xdf, 0xe0]) {
                            // I286/287 PRM Part I Appendix A p. A-4:
                            // FSTSW AX assumes an all-high empty bus.
                            cpu.write_gpr(RAX, 0xffff, 2);
                        }
                    }
                }
                L::Enter { size, alloc, level } => {
                    prepare_enter_displays(cpu, context, *size, *alloc, *level);
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
                    segment_access(cpu, context, SegmentId::Ss, final_sp, 1, AccessKind::Write)
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
                            let linear = u64::from((pushed as u32).wrapping_add(u32::from(byte)));
                            let (physical, _) = legacy::physical_span(cpu, linear, 1);
                            overlay.push((physical, (value >> (byte * 8)) as u8));
                        }
                        let offset =
                            cpu.gpr[RBP].wrapping_sub(u64::from(i) * u64::from(*size)) & stack_mask;
                        let address = segment_access(
                            cpu,
                            context,
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
                        let cached = enter_display(cpu, i);
                        let first = mem.read_index.get();
                        let partial = physical
                            .iter()
                            .any(|at| overlay.iter().any(|(p, _)| p == at));
                        let value = if partial {
                            let mut value = 0;
                            for (byte, at) in physical.into_iter().enumerate() {
                                let part = if let Some((_, value)) =
                                    overlay.iter().rev().find(|(p, _)| *p == at)
                                {
                                    u64::from(*value)
                                } else if cached.is_some() {
                                    mem.skip_read(
                                        cpu,
                                        u64::from((address as u32).wrapping_add(byte as u32)),
                                        1,
                                    );
                                    0
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
                        } else if cached.is_some() {
                            mem.skip_read(cpu, address, *size);
                            0
                        } else {
                            read(cpu, mem, address, *size)?
                        };
                        let value = cached.unwrap_or(value);
                        if cached.is_none() {
                            record_enter_display(
                                cpu,
                                first..mem.read_index.get(),
                                i,
                                address,
                                *size,
                                partial,
                                value,
                            );
                        }
                        values.push(value);
                    }
                    if *level != 0 {
                        values.push(frame);
                    }
                    push(cpu, mem, context, &values, *size)?;
                    cpu.write_gpr(RBP, frame, *size);
                    adjust(cpu, context, 0u32.wrapping_sub(u32::from(*alloc)));
                    cpu.clear_mmio_parts();
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
                    let value = legacy::lmsw_value(cpu.legacy.cr0, read_val(*value, temps) as u16);
                    outcome.context_changed = write_cr0(cpu, value)?;
                }
                L::ClearTs => {
                    // SDM Vol 2A §3.3 CLTS; 80386 PRM §17 p. 17-48.
                    outcome.context_changed = write_cr0(cpu, cpu.legacy.cr0 & !8)?;
                }
                L::ReadSystemSegment { dst, register } => {
                    temps[*dst as usize] = u64::from(match register {
                        SystemRegister::Ldtr => cpu.legacy.ldtr.selector,
                        SystemRegister::Tr => cpu.legacy.tr.selector,
                    });
                }
                L::LoadSystemSegment {
                    register,
                    value,
                    allowed_types,
                } => {
                    load_system_segment(
                        cpu,
                        mem,
                        *register,
                        read_val(*value, temps) as u16,
                        *allowed_types,
                        context.generation,
                    )?;
                    outcome.context_changed = true;
                }
                L::QueryDescriptor {
                    selector,
                    query,
                    allowed_system_types,
                    allow_conforming,
                } => {
                    query_descriptor(
                        cpu,
                        mem,
                        read_val(*selector, temps) as u16,
                        *query,
                        *allowed_system_types,
                        *allow_conforming,
                        context.generation,
                    )?;
                }
                L::AdjustRpl {
                    selector,
                    rpl,
                    destination,
                } => {
                    // 286 PRM App. B p. B-21; 386 PRM §17 p. 17-25:
                    // increase only RPL, set ZF iff changed; all other flags survive.
                    let value = read_val(*selector, temps) as u16;
                    let rpl = read_val(*rpl, temps) as u16 & 3;
                    let changed = value & 3 < rpl;
                    if changed {
                        let value = u64::from((value & !3) | rpl);
                        match destination {
                            SelectorDestination::Register(reg) => write_reg(cpu, *reg, value, 2),
                            SelectorDestination::Memory(addr) => {
                                write(cpu, mem, read_val(*addr, temps), 2, value)?
                            }
                        }
                    }
                    cpu.flags.zf = changed;
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
                            task_switch,
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
                        // 386 PRM §12.3.1.1 p. 12-7 exempts task-switching
                        // INT from RF clearing, as for JMP and CALL.
                        outcome.preserve_rf = task_switch.committed;
                        outcome.context_changed = true;
                        return Ok(true);
                    }
                }
                L::Address {
                    dst,
                    segment,
                    offset,
                    bytes,
                    access,
                } => {
                    temps[*dst as usize] = segment_access(
                        cpu,
                        context,
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
                L::Pop { dst, size } => temps[*dst as usize] = peek(cpu, mem, context, 0, *size)?,
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
                        // SDM Vol 2B §4.3 POPA/POPAD; 80386 PRM §17 POPA:
                        // preflight the whole frame before any device read,
                        // including the discarded SP/ESP slot.
                        segment_access(
                            cpu,
                            context,
                            SegmentId::Ss,
                            sp(cpu, context),
                            8 * *size,
                            AccessKind::Read,
                        )?;
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
                        if context.regime == legacy::MachineRegime::Real
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
                            .unwrap_or_else(|| read_val(cs, temps)) as u16
                    });
                    if context.regime == legacy::MachineRegime::Protected {
                        if let Some(cs) = cs {
                            if tasks::transfer(
                                cpu,
                                mem.raw(),
                                context,
                                cs,
                                if return_ip.is_some() {
                                    tasks::Cause::Call
                                } else {
                                    tasks::Cause::Jump
                                },
                                next as u32,
                                task_switch,
                            )? {
                                outcome.context_changed = true;
                                outcome.preserve_rf = true;
                                return Ok(true);
                            }
                        }
                    }
                    let mut descriptor = if context.regime == legacy::MachineRegime::Protected {
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
                    return Ok(true);
                }
                L::Return {
                    size,
                    far,
                    pop_extra,
                } => {
                    segment_access(
                        cpu,
                        context,
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
                    let mut descriptor = if context.regime == legacy::MachineRegime::Protected {
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
                    return Ok(true);
                }
                L::Iret { size } => {
                    // SDM Vol 2A §3.3 IRET; 80386 PRM §17 pp. 17-74,
                    // 17-76–78 (scan checked): same-CPL frame, independent SS.B.
                    segment_access(
                        cpu,
                        context,
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
                    if context.regime == legacy::MachineRegime::Protected
                        && *size == 4
                        && image & (1 << 17) != 0
                    {
                        return Err(refusal(
                            cpu,
                            crate::UnsupportedSystemOperation::InvalidLegacyContext(
                                legacy::MachineContextError::Virtual8086Unsupported,
                            ),
                        ));
                    }
                    let mut descriptor = if context.regime == legacy::MachineRegime::Protected {
                        Some(code_descriptor(cpu, mem, cs, true, context.generation)?)
                    } else {
                        None
                    };
                    if let Some(descriptor) = &descriptor {
                        code_limit(cpu, descriptor.cache, ip)?;
                    } else {
                        target(cpu, context, ip, 4)?;
                    }
                    if let Some(d) = &mut descriptor {
                        accessed(cpu, mem, d)?;
                    }
                    // SDM Vol 3A §11.3: all frame reads complete before retirement.
                    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
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
                    return Ok(true);
                }
                L::Loop {
                    kind,
                    count_bits,
                    check_target,
                    taken,
                    next,
                } => {
                    execute_loop(
                        cpu,
                        context,
                        *kind,
                        *count_bits,
                        *check_target,
                        *taken,
                        *next,
                    )?;
                    return Ok(true);
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
                            return Ok(false);
                        }
                        let output = matches!(kind, StrOp::Outs);
                        let index = if output { RSI } else { RDI };
                        let offset = cpu.gpr[index] & mask(width);
                        let segment = if output { *source } else { SegmentId::Es };
                        let address = segment_access(
                            cpu,
                            context,
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
                                base: u64::from((address as u32).wrapping_sub(offset as u32)),
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
                        return Ok(true);
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
                            let a =
                                segment_access(cpu, context, *source, si, *elem, AccessKind::Read)?;
                            Some(read(cpu, mem, a, *elem)?)
                        } else {
                            None
                        };
                        let dst = if matches!(
                            kind,
                            StrOp::Movs | StrOp::Stos | StrOp::Cmps | StrOp::Scas
                        ) {
                            Some(segment_access(
                                cpu,
                                context,
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
                                apply(&mut cpu.flags, FlagMask::ALL, &alu_sub(lhs, rhs, 0, *elem));
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
                                return Ok(true);
                            }
                        }
                    }
                }
                L::Leave { size } => {
                    let offset = cpu.gpr[5] & mask(context.stack_address_bits / 8);
                    let a = segment_access(
                        cpu,
                        context,
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
                    if context.regime == legacy::MachineRegime::Real
                        && cpu.rflags_system & 0x4000 != 0
                    {
                        return Err(refusal(
                            cpu,
                            crate::UnsupportedSystemOperation::LegacyTaskReturn,
                        ));
                    }
                    if context.regime == legacy::MachineRegime::Protected
                        && cpu.rflags_system & 0x4000 != 0
                    {
                        tasks::task_return(cpu, mem, context, next as u32, task_switch)?;
                        outcome.context_changed = true;
                        outcome.preserve_rf = true;
                        return Ok(true);
                    }
                }
                // Twenty-bit operations belong to the dedicated shared walk.
                // Keep that execution machinery out of the system-model walker.
                L::WrappedLoad { .. }
                | L::WrappedStore { .. }
                | L::FlagsImage { .. }
                | L::LoadFlagsWord { .. }
                | L::DivideTrap { .. }
                | L::Coprocessor20 { .. }
                | L::WrappedString { .. }
                | L::Interrupt20 { .. }
                | L::CheckIndex20 { .. }
                | L::Brkem20 { .. }
                | L::Reti20 { .. }
                | L::PopPsw20 { .. }
                | L::BitField20 { .. }
                | L::BcdString20 { .. }
                | L::Fpo2Escape20 { .. }
                | L::InhibitIrq20 => {
                    return Err(refusal(
                        cpu,
                        crate::UnsupportedSystemOperation::LegacyExecution,
                    ))
                }
                L::Refuse { operation } => return Err(refusal(cpu, *operation)),
            }
            None
        }
        _ => unreachable!("Machine-specific op"),
    };
    if let Some(InterpResult::Exit(exit)) = result {
        return Err(exit.into_exit());
    }
    Ok(false)
}

pub(super) fn machine_specific(op: &IrOp) -> bool {
    matches!(
        op,
        IrOp::AtomicRmw { .. }
            | IrOp::BoundGate { .. }
            | IrOp::Branch { .. }
            | IrOp::Hlt
            | IrOp::Jump { .. }
            | IrOp::Load { .. }
            | IrOp::LoopCx { .. }
            | IrOp::PortIo { .. }
            | IrOp::SegLimitCheck { .. }
            | IrOp::SetIf { .. }
            | IrOp::Store { .. }
            | IrOp::Trap { .. }
            | IrOp::Unsupported { .. }
            | IrOp::X87 { .. }
            | IrOp::Legacy(_)
    )
}

struct InstructionExecution<'a, 'b, const ROUTED: bool, const ORDER: u8> {
    ir: &'a InstructionView<'a>,
    mem: &'a LegacyAccess<'a, ROUTED, ORDER>,
    context: MachineContext,
    task_switch: &'b mut tasks::SwitchState,
    outcome: Outcome,
}
impl<const ROUTED: bool, const ORDER: u8> super::InstructionExecutor<'_>
    for &mut InstructionExecution<'_, '_, ROUTED, ORDER>
{
    fn enabled(&self) -> bool {
        true
    }
    fn next_ip(&self) -> u64 {
        self.ir
            .guest_start
            .wrapping_add(u64::from(self.ir.guest_len))
            & 0xffff_ffff
    }
    fn execute(&mut self, op: &IrOp, cpu: &mut CpuState, temps: &mut [u64]) -> Option<WalkResult> {
        match execute_op(
            op,
            self.ir,
            cpu,
            self.mem,
            temps,
            self.context,
            self.task_switch,
            &mut self.outcome,
        ) {
            Ok(false) => None,
            Ok(true) => Some(WalkResult::Continue),
            Err(exit) => Some(WalkResult::exit(exit)),
        }
    }
}

#[cfg(test)]
mod descriptor_update_tests {
    use super::*;
    use crate::{CpuGeneration, Prot, RegionKind, Vm, VmConfig};

    fn image(base: u16, attr: u8) -> u64 {
        u64::from_le_bytes([0xff, 0xff, base as u8, (base >> 8) as u8, 0, attr, 0, 0])
    }

    #[test]
    fn ltr_cas_preserves_competing_descriptor_edits_and_reloads_the_cache() {
        // SDM Vol 2A §3.3 LTR p. 3-576; 386 PRM §7.6.1 p. 7-10:
        // the busy update is locked. Inject an edit after the descriptor read
        // and before CAS; both the competing base/DPL and busy bit must survive.
        for generation in [CpuGeneration::I286, CpuGeneration::I386] {
            let mut vm = Vm::new(VmConfig::flat(0x10000));
            vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.mem.write(0x2008, image(0x4000, 0x81), 8).unwrap();
            let mut cpu = CpuState::default();
            cpu.legacy.gdtr.base = 0x102000;
            cpu.legacy.gdtr.limit = 15;
            cpu.legacy.a20_enabled = 0;
            DESCRIPTOR_UPDATE.with(|update| update.set(Some((0x2008, image(0x5000, 0xa1)))));
            load_system_segment(
                &mut cpu,
                &LegacyAccess::<false, 0>::new(&vm.mem),
                SystemRegister::Tr,
                8,
                1 << 1,
                generation,
            )
            .unwrap();
            assert_eq!(vm.mem.read(0x2008, 8).unwrap(), image(0x5000, 0xa3));
            assert_eq!(cpu.legacy.tr.cache.base, 0x5000);
            assert_eq!(cpu.legacy.tr.cache.attributes, 0xa3);
            DESCRIPTOR_UPDATE.with(|update| assert!(update.get().is_none()));
        }
    }

    #[test]
    fn ltr_cas_failure_rechecks_type_and_presence_without_installing_tr() {
        // SDM Vol 2A §3.3 LTR pp. 3-576–577: a busy or wrong-type
        // descriptor faults with #GP(selector); an absent TSS with #NP(selector).
        for generation in [CpuGeneration::I286, CpuGeneration::I386] {
            for (attr, vector) in [(0x83, 13), (0x82, 13), (0x91, 13), (0x01, 11)] {
                let mut vm = Vm::new(VmConfig::flat(0x10000));
                vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
                vm.mem.write(0x2008, image(0x4000, 0x81), 8).unwrap();
                let mut cpu = CpuState::default();
                cpu.legacy.cr0 = 1;
                cpu.legacy.gdtr.base = 0x2000;
                cpu.legacy.gdtr.limit = 15;
                let before = cpu.legacy.tr;
                DESCRIPTOR_UPDATE.with(|update| update.set(Some((0x2008, image(0x5000, attr)))));
                let exit = load_system_segment(
                    &mut cpu,
                    &LegacyAccess::<false, 0>::new(&vm.mem),
                    SystemRegister::Tr,
                    11,
                    1 << 1,
                    generation,
                )
                .unwrap_err();
                assert!(matches!(exit, Exit::Exception { vector: actual, .. } if actual == vector));
                assert_eq!(
                    exit.fault_metadata().unwrap().error_code(),
                    crate::fault::ErrorCode::bits(8, 0xffff)
                );
                assert_eq!(cpu.legacy.tr, before);
                assert_eq!(vm.mem.read(0x2008, 8).unwrap(), image(0x5000, attr));
            }
        }
    }

    #[test]
    fn accessed_cas_preserves_competing_gdt_and_ldt_edits_through_a20() {
        // SDM Vol 3A §§3.4.5.1, 11.1.2.1 p. 11-3: setting A follows
        // LOCK semantics. Inject a DPL/base edit in the read/write window.
        for generation in [CpuGeneration::I286, CpuGeneration::I386] {
            for ldt in [false, true] {
                let mut vm = Vm::new(VmConfig::flat(0x10000));
                vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
                vm.mem.write(0x2008, image(0x4000, 0x92), 8).unwrap();
                let mut cpu = CpuState::default();
                cpu.legacy.cr0 = 1;
                cpu.legacy.a20_enabled = 0;
                cpu.legacy.gdtr.base = 0x102000;
                cpu.legacy.gdtr.limit = 15;
                cpu.legacy.ldtr.selector = 16;
                cpu.legacy.ldtr.cache.base = 0x102000;
                cpu.legacy.ldtr.cache.limit = 15;
                cpu.legacy.ldtr.cache.unusable = 0;
                DESCRIPTOR_UPDATE.with(|update| update.set(Some((0x2008, image(0x5000, 0xb2)))));
                load_segment(
                    &mut cpu,
                    &LegacyAccess::<false, 0>::new(&vm.mem),
                    SegmentId::Ds,
                    if ldt { 12 } else { 8 },
                    generation,
                )
                .unwrap();
                assert_eq!(vm.mem.read(0x2008, 8).unwrap(), image(0x5000, 0xb3));
                assert_eq!(
                    cpu.legacy.segments[SegmentId::Ds as usize].attributes & 1,
                    1
                );
                DESCRIPTOR_UPDATE.with(|update| assert!(update.get().is_none()));
            }
        }
    }
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
                        &mut tasks::SwitchState::default(),
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
                    &mut tasks::SwitchState::default(),
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
