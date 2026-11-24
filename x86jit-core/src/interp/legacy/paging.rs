//! Internal Machine translation boundary (paging design §§3.1–3.4, spec §17.10).
//! No translation cache: each instruction/iteration owns its physical spans.
use super::*;
use crate::fault::{
    AddressPrecision, AddressSpace, ErrorCode, FaultAddress, FaultCause, FaultClass, FaultMetadata,
    FaultOrigin, FaultPath, Knowledge, MemoryEffects, PagingWalk, PagingWalkLevel,
    RegisterProgress,
};

const PRESENT: u32 = 1;
const WRITE: u32 = 2;
const USER: u32 = 4;
const ACCESSED: u32 = 0x20;
const DIRTY: u32 = 0x40;
const FRAME: u32 = 0xffff_f000;
const IDENTITY: u32 = FRAME | PRESENT | WRITE | USER;

#[cfg(feature = "test-hooks")]
type CasHook = Box<dyn FnMut(&Memory, u64, u32, u32, Option<u32>)>;
#[cfg(feature = "test-hooks")]
thread_local! {
    static CAS_HOOK: std::cell::RefCell<Option<CasHook>> = const { std::cell::RefCell::new(None) };
}

/// Observe/inject table CAS races on this thread only. `None` precedes the CAS;
/// `Some(observed)` follows it, including failed comparisons.
#[cfg(feature = "test-hooks")]
pub fn with_cas_hook_for_test<R>(
    hook: impl FnMut(&Memory, u64, u32, u32, Option<u32>) + 'static,
    run: impl FnOnce() -> R,
) -> R {
    struct Reset(Option<CasHook>);
    impl Drop for Reset {
        fn drop(&mut self) {
            CAS_HOOK.with(|slot| *slot.borrow_mut() = self.0.take());
        }
    }
    let _reset = Reset(CAS_HOOK.with(|slot| slot.replace(Some(Box::new(hook)))));
    run()
}

#[cfg(feature = "test-hooks")]
fn cas_hook(memory: &Memory, bus: u64, expected: u32, bits: u32, observed: Option<u32>) {
    CAS_HOOK.with(|slot| {
        if let Some(hook) = &mut *slot.borrow_mut() {
            hook(memory, bus, expected, bits, observed);
        }
    });
}

// The paged adapter retains the same control-write validation as PG=0.
#[cold]
#[inline(never)]
pub(super) fn write_cr0(cpu: &mut CpuState, value: u32) -> Result<bool, Exit> {
    // SDM Vol 3A §5.1.1 p. 5-2; Intel486 PRM §12.2.1 p. 12-3:
    // PG requires PE and NW requires CD.
    if value & 1 == 0 || (cpu.machine_flags_mask != 0 && value & 0x6000_0000 == 0x2000_0000) {
        return Err(exception(cpu, 13));
    }
    let changed = cpu.legacy.cr0 != value;
    cpu.legacy.cr0 = value;
    Ok(changed)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Privilege {
    Supervisor,
    User,
}
impl Privilege {
    fn user(self) -> bool {
        matches!(self, Self::User)
    }
}

struct PageFault {
    linear: u32,
    access: AccessKind,
    privilege: Privilege,
    protection: bool,
}
impl PageFault {
    fn exit(self, cpu: &CpuState) -> Exit {
        // 80386 PRM §9.8.14, Fig. 9-8 pp. 9-17–18; Intel486 PRM
        // §9.9.14 pp. 9-21–23: CR2 is linear, only P/W/U are defined.
        let code = u32::from(self.protection)
            | (u32::from(self.access == AccessKind::Write) << 1)
            | (u32::from(self.privilege.user()) << 2);
        let mut m = FaultMetadata::architectural_exception(
            cpu.rip,
            cpu.rip,
            14,
            FaultClass::Fault,
            ErrorCode::bits(code, 7),
        );
        m.origin = FaultOrigin::GuestException(FaultCause::PageTranslation);
        m.path = FaultPath::Interpreter;
        m.address = Knowledge::Known(FaultAddress::new(
            u64::from(self.linear),
            AddressPrecision::ExactByte,
            AddressSpace::Linear,
        ));
        m.access = Knowledge::Known(self.access.into());
        m.transfer_access = m.access;
        m.progress.registers = RegisterProgress::InstructionEntry;
        // A updates can precede this failure even with unchanged operand RAM.
        m.progress.ram = MemoryEffects::PossiblyPartial;
        m.progress.device = MemoryEffects::None;
        Exit::exception(cpu.rip, 14).with_fault_metadata(m)
    }
}

pub(crate) fn publish_cr2(cpu: &mut CpuState, exit: &Exit) {
    if let Some(m) = exit.fault_metadata() {
        if m.origin == FaultOrigin::GuestException(FaultCause::PageTranslation) {
            if let Knowledge::Known(address) = m.address {
                cpu.legacy.cr2 = address.value() as u32;
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Translation {
    physical: u32,
    bytes: u8,
    pte_address: u32,
    pte: u32,
}

#[allow(clippy::too_many_arguments)]
fn bus_failure(
    cpu: &mut CpuState,
    trap: MemTrap,
    linear: u32,
    physical: u32,
    bus: u64,
    level: PagingWalkLevel,
    access: AccessKind,
    value: u64,
) -> Exit {
    let StepResult::Exit(mut exit) =
        trap_out(cpu, cpu.rip, trap, bus, 4, access, value).into_step()
    else {
        unreachable!()
    };
    if let Exit::UnmappedMemory {
        metadata: Some(m), ..
    } = &mut exit
    {
        // A table transfer's byte offset is physical evidence, not an offset
        // within the original operand's linear address.
        m.physical_address = match m.address {
            Knowledge::Known(address) => Knowledge::Known(address.value()),
            _ => Knowledge::Known(bus),
        };
        m.address = Knowledge::Known(FaultAddress::new(
            u64::from(linear),
            AddressPrecision::ExactByte,
            AddressSpace::Linear,
        ));
        m.paging_walk = Some(PagingWalk {
            level,
            physical_address: u64::from(physical),
        });
        m.progress.ram = MemoryEffects::PossiblyPartial;
    }
    exit
}

fn table_bus<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    linear: u32,
    physical: u32,
    level: PagingWalkLevel,
) -> Result<u64, Exit> {
    // 80386 PRM §§5.2.3–5.2.4 pp. 5-8–10: table references are physical.
    // Apply the board gate only after selecting the entry (paging design §3.6).
    let (bus, bytes) = legacy::physical_span(cpu, mem.physical_mask(), u64::from(physical), 4);
    debug_assert_eq!(bytes, 4); // aligned entries cannot cross the bus/A20 boundary
    match mem.raw().region_kind(bus, 4) {
        Ok(crate::RegionKind::Ram) => Ok(bus),
        Ok(_) => Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::PagingTableStorage,
        )),
        Err(_)
            if (0..4).any(|byte| {
                matches!(
                    mem.raw().region_kind(bus + byte, 1),
                    Ok(crate::RegionKind::Trap | crate::RegionKind::Rom)
                )
            }) =>
        {
            Err(refusal(
                cpu,
                crate::UnsupportedSystemOperation::PagingTableStorage,
            ))
        }
        Err(trap) => Err(bus_failure(
            cpu,
            trap,
            linear,
            physical,
            bus,
            level,
            AccessKind::Read,
            0,
        )),
    }
}

fn table_read<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    linear: u32,
    physical: u32,
    level: PagingWalkLevel,
) -> Result<u32, Exit> {
    let bus = table_bus(cpu, mem, linear, physical, level)?;
    mem.raw()
        .read(bus, 4)
        .map(|v| v as u32)
        .map_err(|trap| bus_failure(cpu, trap, linear, physical, bus, level, AccessKind::Read, 0))
}

// None means that frame/presence/permissions changed: restart the translation.
fn table_or<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    linear: u32,
    physical: u32,
    level: PagingWalkLevel,
    mut expected: u32,
    bits: u32,
) -> Result<Option<u32>, Exit> {
    let bus = table_bus(cpu, mem, linear, physical, level)?;
    while expected & bits != bits {
        // SDM Vol 3A §11.1.2.1 p. 11-3; 80386 PRM §11.2.2 p. 11-5:
        // individual A/D updates are locked and preserve competing edits.
        let value = expected | bits;
        #[cfg(feature = "test-hooks")]
        cas_hook(mem.raw(), bus, expected, bits, None);
        let observed = mem
            .raw()
            .atomic_cas(bus, u64::from(expected), u64::from(value), 4)
            .map_err(|trap| {
                bus_failure(
                    cpu,
                    trap,
                    linear,
                    physical,
                    bus,
                    level,
                    AccessKind::Write,
                    u64::from(value),
                )
            })? as u32;
        #[cfg(feature = "test-hooks")]
        cas_hook(mem.raw(), bus, expected, bits, Some(observed));
        if observed == expected {
            mem.note_write(bus, 4);
            return Ok(Some(value));
        }
        if (observed ^ expected) & IDENTITY != 0 {
            return Ok(None);
        }
        expected = observed;
    }
    Ok(Some(expected))
}

fn translate_linear<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    linear: u32,
    bytes: u8,
    access: AccessKind,
    privilege: Privilege,
    set_accessed: bool,
) -> Result<Translation, Exit> {
    loop {
        // 80386 PRM §5.2, Figs. 5-8–10 pp. 5-7–9; SDM Vol 3A §5.3
        // pp. 5-9–13: two 1024-entry levels and 4 KiB frames, no PSE.
        let pde_address = (cpu.legacy.cr3 & FRAME) + 4 * (linear >> 22);
        let pde = table_read(cpu, mem, linear, pde_address, PagingWalkLevel::Directory)?;
        let fault = |protection| PageFault {
            linear,
            access,
            privilege,
            protection,
        };
        if pde & PRESENT == 0 {
            return Err(fault(false).exit(cpu));
        }
        let pde = if set_accessed {
            let Some(pde) = table_or(
                cpu,
                mem,
                linear,
                pde_address,
                PagingWalkLevel::Directory,
                pde,
                ACCESSED,
            )?
            else {
                continue;
            };
            pde
        } else {
            pde
        };
        let pte_address = (pde & FRAME) + 4 * ((linear >> 12) & 0x3ff);
        let pte = table_read(cpu, mem, linear, pte_address, PagingWalkLevel::Table)?;
        if pte & PRESENT == 0 {
            return Err(fault(false).exit(cpu));
        }
        // Intel486 PRM §9.9.14 p. 9-21 requires both A bits on operand
        // protection faults. The same order on the 386 is engine policy (§3.3).
        let pte = if set_accessed {
            let Some(pte) = table_or(
                cpu,
                mem,
                linear,
                pte_address,
                PagingWalkLevel::Table,
                pte,
                ACCESSED,
            )?
            else {
                continue;
            };
            pte
        } else {
            pte
        };
        // 80386 PRM §§6.4.1–6.4.2 pp. 6-20–22, Table 6-5;
        // Intel486 PRM §4.1.3 p. 4-7 describes WP for user pages.
        // SDM Vol 3A §§5.1.3, 5.6.1 pp. 5-5, 5-33–35 applies WP to
        // both mapping classes (paging design Q3).
        let rights = pde & pte;
        let wp = cpu.machine_flags_mask != 0 && cpu.legacy.cr0 & (1 << 16) != 0;
        if (privilege.user() && rights & USER == 0)
            || (access == AccessKind::Write && (privilege.user() || wp) && rights & WRITE == 0)
        {
            return Err(fault(true).exit(cpu));
        }
        return Ok(Translation {
            physical: (pte & FRAME) | (linear & 0xfff),
            bytes: u32::from(bytes).min(0x1000 - (linear & 0xfff)) as u8,
            pte_address,
            pte,
        });
    }
}

#[derive(Clone)]
struct Fragment {
    linear: u32,
    bus: u64,
    bytes: u8,
    shift: u8,
    translation: Translation,
    dirty_complete: bool,
}

fn complete_dirty<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    fragments: &mut [Fragment],
) -> Result<bool, Exit> {
    // 80386 PRM §5.2.4.3 p. 5-10; Intel486 PRM §5.3.4.3 pp. 5-21–22:
    // a permission probe does not complete the D update required by a write.
    for f in fragments {
        if f.dirty_complete {
            continue;
        }
        let t = f.translation;
        if table_or(
            cpu,
            mem,
            f.linear,
            t.pte_address,
            PagingWalkLevel::Table,
            t.pte,
            DIRTY,
        )?
        .is_none()
        {
            return Ok(false);
        }
        f.dirty_complete = true;
    }
    Ok(true)
}

fn plan<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    addr: u64,
    size: u8,
    access: AccessKind,
    dirty: bool,
) -> Result<Vec<Fragment>, Exit> {
    // SDM Vol 3A §5.6.1 p. 5-33; 80386 PRM §6.5 p. 6-22:
    // implicit table/TSS and inner-stack accesses are supervisor at any CPL.
    let privilege = if cpu.legacy.cpl == 3 && !mem.supervisor() {
        Privilege::User
    } else {
        Privilege::Supervisor
    };
    let key = PlanKey {
        addr: addr as u32,
        size,
        access,
        privilege,
        cr3: cpu.legacy.cr3,
        cr0: cpu.legacy.cr0,
    };
    if let Some(tx) = mem.transaction() {
        let cached = tx
            .borrow()
            .plans
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, fragments)| fragments.clone());
        if let Some(mut fragments) = cached {
            if !dirty || preparing(mem) {
                return Ok(fragments);
            }
            if complete_dirty(cpu, mem, &mut fragments)? {
                let mut tx = tx.borrow_mut();
                tx.plans.iter_mut().find(|(k, _)| *k == key).unwrap().1 = fragments.clone();
                return Ok(fragments);
            }
            tx.borrow_mut().plans.retain(|(k, _)| *k != key);
        }
    }
    'restart: loop {
        let mut fragments = Vec::new();
        let mut consumed = 0;
        while consumed < size {
            let linear = (addr as u32).wrapping_add(u32::from(consumed));
            let translation =
                translate_linear(cpu, mem, linear, size - consumed, access, privilege, true)?;
            let (bus, bytes) = legacy::physical_span(
                cpu,
                mem.physical_mask(),
                u64::from(translation.physical),
                translation.bytes,
            );
            let bytes = mem
                .raw()
                .region_end(bus)
                .map_or(bytes, |end| u64::from(bytes).min(end - bus) as u8);
            fragments.push(Fragment {
                linear,
                bus,
                bytes,
                shift: consumed * 8,
                translation,
                dirty_complete: translation.pte & DIRTY != 0,
            });
            consumed += bytes;
        }
        if access == AccessKind::Write {
            for fragment in &fragments {
                probe_physical(
                    cpu,
                    mem,
                    u64::from(fragment.linear),
                    fragment.bus,
                    fragment.bytes,
                )?;
            }
        }
        if dirty && !preparing(mem) {
            // 80386 PRM §5.2.4.3 p. 5-10; Intel486 PRM §5.3.4.3
            // pp. 5-21–22: PTE.D precedes the operand write; PDE.D is preserved.
            // Preflight the entire span before D, retaining even aliased frames.
            if !complete_dirty(cpu, mem, &mut fragments)? {
                continue 'restart;
            }
        }
        if let Some(tx) = mem.transaction() {
            tx.borrow_mut().plans.push((key, fragments.clone()));
        }
        return Ok(fragments);
    }
}

fn commit_ram<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
) -> Result<(), Exit> {
    let mut tx = mem.transaction().expect("paged transaction").borrow_mut();
    let linears: Vec<_> = tx
        .plans
        .iter()
        .flat_map(|(_, spans)| spans)
        .map(|f| (f.bus, f.linear))
        .collect();
    for (bus, bytes, value, committed) in &mut tx.writes {
        if !*committed {
            if let Err(trap) = mem.raw().write(*bus, *value, *bytes) {
                let linear = linears.iter().find(|f| f.0 == *bus).map_or(0, |f| f.1);
                let StepResult::Exit(exit) =
                    trap_out(cpu, cpu.rip, trap, *bus, *bytes, AccessKind::Write, *value)
                        .into_step()
                else {
                    unreachable!()
                };
                let mut exit = crate::fault::qualify_legacy_bus(exit, u64::from(linear), *bus);
                if let Exit::UnmappedMemory {
                    metadata: Some(m), ..
                } = &mut exit
                {
                    m.progress.ram = MemoryEffects::PossiblyPartial;
                    m.progress.acknowledged = mmio_evidence(cpu);
                    if !m.progress.acknowledged.is_empty() {
                        m.progress.device = MemoryEffects::PossiblyPartial;
                    }
                }
                return Err(exit);
            }
            mem.note_write(*bus, *bytes);
            *committed = true;
        }
    }
    Ok(())
}

// Transfer visits belong to one retained instruction/iteration. Separate read
// and write cursors distinguish two accesses even when their bus bytes alias.
fn transfer_read<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    f: &Fragment,
) -> Result<u64, Exit> {
    let Some(cell) = mem.transaction() else {
        return read_physical(cpu, mem, u64::from(f.linear), f.bus, f.bytes);
    };
    let index = {
        let mut tx = cell.borrow_mut();
        let i = tx.read_cursor;
        tx.read_cursor += 1;
        i
    };
    if preparing(mem) {
        let result = read_physical(cpu, mem, u64::from(f.linear), f.bus, f.bytes);
        cell.borrow_mut()
            .prepared_reads
            .push((f.bus, f.bytes, result.as_ref().ok().copied()));
        return result;
    }
    if let Some(&(bus, bytes, value)) = cell.borrow().reads.get(index) {
        if (bus, bytes) != (f.bus, f.bytes) {
            return Err(refusal(
                cpu,
                crate::UnsupportedSystemOperation::LegacyEntryFacility,
            ));
        }
        return Ok(value);
    }
    let prepared = cell.borrow().prepared_reads.get(index).copied();
    let mut value = if let Some((bus, bytes, Some(value))) = prepared {
        if (bus, bytes) != (f.bus, f.bytes) {
            return Err(refusal(
                cpu,
                crate::UnsupportedSystemOperation::LegacyEntryFacility,
            ));
        }
        value
    } else {
        match mem.raw().read(f.bus, f.bytes) {
            Ok(value) => value,
            Err(MemTrap::Mmio) if cpu.pending_mmio.is_some() => {
                let value = cpu.pending_mmio.take().unwrap() & mask(f.bytes);
                cpu.record_mmio_part(cpu.rip, f.bus | ((index as u64) << 32), value);
                value
            }
            Err(trap) => return Err(transfer_failure(cpu, mem, trap, f, AccessKind::Read, 0)),
        }
    } & mask(f.bytes);
    // A later read in ENTER or a task save sees this attempt's staged RAM.
    let mut tx = cell.borrow_mut();
    for byte in 0..f.bytes {
        for &(bus, bytes, stored, committed) in tx.writes.iter().rev() {
            let at = f.bus + u64::from(byte);
            if !committed && (bus..bus + u64::from(bytes)).contains(&at) {
                let shift = byte * 8;
                value =
                    (value & !(0xff << shift)) | (((stored >> ((at - bus) * 8)) & 0xff) << shift);
                break;
            }
        }
    }
    tx.reads.push((f.bus, f.bytes, value));
    Ok(value)
}

fn transfer_failure<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    trap: MemTrap,
    f: &Fragment,
    access: AccessKind,
    value: u64,
) -> Exit {
    if trap == MemTrap::Mmio {
        if let Some(tx) = mem.transaction() {
            tx.borrow_mut().pending_mmio_linear = Some(f.linear);
        }
    }
    let StepResult::Exit(exit) =
        trap_out(cpu, cpu.rip, trap, f.bus, f.bytes, access, value).into_step()
    else {
        unreachable!()
    };
    crate::fault::qualify_legacy_bus(exit, u64::from(f.linear), f.bus)
}

fn transfer_write<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    f: &Fragment,
    value: u64,
) -> Result<(), Exit> {
    let Some(cell) = mem.transaction() else {
        return write_physical(cpu, mem, u64::from(f.linear), f.bus, f.bytes, value);
    };
    let value = value & mask(f.bytes);
    let index = {
        let mut tx = cell.borrow_mut();
        let i = tx.write_cursor;
        tx.write_cursor += 1;
        i
    };
    if let Some(&(bus, bytes, stored, _)) = cell.borrow().writes.get(index) {
        if (bus, bytes, stored) != (f.bus, f.bytes, value) {
            return Err(refusal(
                cpu,
                crate::UnsupportedSystemOperation::LegacyEntryFacility,
            ));
        }
        return Ok(());
    }
    let device = mem.raw().region_kind(f.bus, f.bytes) == Ok(crate::RegionKind::Trap);
    if device {
        if cpu.pending_mmio_write {
            cpu.pending_mmio_write = false;
            cpu.record_mmio_part(cpu.rip, f.bus | ((index as u64) << 32) | (1 << 63), value);
            mem.raw().note_write(f.bus, usize::from(f.bytes));
            mem.note_write(f.bus, f.bytes);
        } else {
            return Err(transfer_failure(
                cpu,
                mem,
                MemTrap::Mmio,
                f,
                AccessKind::Write,
                value,
            ));
        }
    }
    cell.borrow_mut()
        .writes
        .push((f.bus, f.bytes, value, device));
    Ok(())
}

#[cold]
#[inline(never)]
pub(super) fn read<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    addr: u64,
    size: u8,
    access: AccessKind,
) -> Result<u64, Exit> {
    let fragments = plan(cpu, mem, addr, size, access, false)?;
    let mut value = 0;
    for f in fragments {
        let part = match transfer_read(cpu, mem, &f) {
            Err(Exit::MmioRead { .. }) if preparing(mem) => {
                if mem.supervisor() || !mem.transaction().unwrap().borrow().device_dummy {
                    return Err(refusal(
                        cpu,
                        crate::UnsupportedSystemOperation::LegacyInstruction,
                    ));
                }
                0
            }
            other => other?,
        };
        let part = if preparing(mem) {
            let mut tx = mem.transaction().unwrap().borrow_mut();
            let mut part = part;
            for byte in 0..f.bytes {
                if let Some((_, value)) = tx
                    .overlay
                    .iter()
                    .rev()
                    .find(|(at, _)| *at == f.bus + u64::from(byte))
                {
                    let shift = byte * 8;
                    part = (part & !(0xff << shift)) | (u64::from(*value) << shift);
                }
            }
            if let Some((_, _, snapshot)) = tx.prepared_reads.last_mut() {
                if snapshot.is_some() {
                    *snapshot = Some(part);
                }
            }
            part
        } else {
            part
        };
        value |= part << f.shift;
    }
    Ok(value)
}

// Control-transfer frame inputs cannot use dummy values: EFLAGS.VM selects
// both the frame extent and the segment-loading rules (design v86 §4.2).
#[cold]
#[inline(never)]
pub(super) fn read_frame<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    addr: u64,
    size: u8,
) -> Result<u64, Exit> {
    if let Some(tx) = mem.transaction() {
        tx.borrow_mut().frame_reads = true;
    }
    let fragments = plan(cpu, mem, addr, size, AccessKind::Read, false)?;
    let mut value = 0;
    for f in fragments {
        let part = transfer_read(cpu, mem, &f).inspect_err(|exit| {
            if matches!(exit, Exit::MmioRead { .. }) {
                if let Some(tx) = mem.transaction() {
                    tx.borrow_mut().pending_mmio_linear = Some(f.linear);
                }
            }
        })?;
        value |= part << f.shift;
    }
    Ok(value)
}

// Only bus completion outputs cross from a preparation preview. Architectural
// destinations remain in the original image until the complete return succeeds.
#[cold]
#[inline(never)]
fn retain_frame_answers(cpu: &mut CpuState, preview: &CpuState) {
    cpu.pending_mmio = preview.pending_mmio;
    cpu.mmio_parts = preview.mmio_parts;
    cpu.mmio_parts_len = preview.mmio_parts_len;
    cpu.mmio_parts_rip = preview.mmio_parts_rip;
    cpu.mmio_retry_active = preview.mmio_retry_active;
    cpu.mmio_retry_cs = preview.mmio_retry_cs;
}

#[cold]
#[inline(never)]
pub(super) fn write<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    addr: u64,
    size: u8,
    value: u64,
) -> Result<(), Exit> {
    let fragments = plan(cpu, mem, addr, size, AccessKind::Write, true)?;
    for f in fragments {
        if preparing(mem) {
            let mut tx = mem.transaction().unwrap().borrow_mut();
            for byte in 0..f.bytes {
                tx.overlay.push((
                    f.bus + u64::from(byte),
                    (value >> (f.shift + byte * 8)) as u8,
                ));
            }
            continue;
        }
        transfer_write(cpu, mem, &f, value >> f.shift)?;
    }
    Ok(())
}

#[cold]
#[inline(never)]
pub(super) fn probe<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    addr: u64,
    size: u8,
) -> Result<(), Exit> {
    plan(cpu, mem, addr, size, AccessKind::Write, false).map(|_| ())
}

#[cold]
#[inline(never)]
pub(super) fn check_write_permission<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    addr: u64,
) -> Result<(), Exit> {
    // SDM Vol 2A §3.3 ENTER pp. 3-295–297 checks whether a write would
    // #PF without transferring an operand. ASSUMED: no A/D updates for
    // this check; neither ENTER nor Vol 3A §5.8 specifies them without
    // an access. Do not retain an operand plan or check physical backing.
    let privilege = if cpu.legacy.cpl == 3 && !mem.supervisor() {
        Privilege::User
    } else {
        Privilege::Supervisor
    };
    translate_linear(
        cpu,
        mem,
        addr as u32,
        1,
        AccessKind::Write,
        privilege,
        false,
    )
    .map(|_| ())
}

// Fetch performs the same read-permission walk but retains execute metadata.
// 80386 PRM §9.8.14 Fig. 9-8 pp. 9-17–18; Intel486 PRM §9.9.14
// Fig. 9-7 pp. 9-21–22: no I/D bit exists on these models.
pub(crate) fn fetch_bus_for_test(
    cpu: &mut CpuState,
    memory: &Memory,
    linear: u32,
    wrote: &Cell<bool>,
) -> Result<u64, Exit> {
    if memory.host_fault_recovery() {
        fetch_bus::<true>(cpu, memory, linear, wrote)
    } else {
        fetch_bus::<false>(cpu, memory, linear, wrote)
    }
}

fn fetch_bus<const ROUTED: bool>(
    cpu: &mut CpuState,
    memory: &Memory,
    linear: u32,
    wrote: &Cell<bool>,
) -> Result<u64, Exit> {
    let privilege = if cpu.legacy.cpl == 3 {
        Privilege::User
    } else {
        Privilege::Supervisor
    };
    let observe = |_, _| wrote.set(true);
    let mut mem = LegacyAccess::<ROUTED, 0, true>::new(memory, u32::MAX);
    mem.writes = Some(&observe);
    let t = translate_linear(cpu, &mem, linear, 1, AccessKind::Execute, privilege, true)?;
    Ok(legacy::physical_span(cpu, u32::MAX, u64::from(t.physical), 1).0)
}

#[cfg(feature = "test-hooks")]
pub(crate) fn scalar_for_test(
    code: &[u8],
    model: crate::CpuModel,
    cpu: &mut CpuState,
    memory: &Memory,
) -> Result<(), Exit> {
    if !matches!(
        model,
        crate::CpuModel::I80386 | crate::CpuModel::I80486Dx | crate::CpuModel::I80486DxCpuid
    ) || cpu.legacy.cr0 & 0x8000_0001 != 0x8000_0001
    {
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyEntryFacility,
        ));
    }
    cpu.machine_flags_mask = model.machine_flags_mask();
    let context = legacy::derive_legacy_context(model.legacy_model(), cpu).map_err(|error| {
        refusal(
            cpu,
            crate::UnsupportedSystemOperation::InvalidLegacyContext(error),
        )
    })?;
    let ir = crate::lift::legacy::instruction_for_profile(
        code,
        cpu.rip as u32,
        context,
        model.profile(),
    )
    .map_err(|_| refusal(cpu, crate::UnsupportedSystemOperation::LegacyInstruction))?;
    execute_for_test::<true, false>(
        &ir,
        cpu,
        memory,
        context,
        &mut tasks::SwitchState::default(),
    )
    .map(|_| ())
}

#[cfg(feature = "test-hooks")]
pub(crate) fn execute_for_test<const PAGED: bool, const V86: bool>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    memory: &Memory,
    context: MachineContext,
    task_switch: &mut tasks::SwitchState,
) -> Result<(Outcome, Option<PortPlan>), Exit>
where
    PagingMode<PAGED>: PagingStorage,
{
    let mut scratch = vec![0; ir.temp_count as usize];
    let mem = LegacyAccess::<false, 0, PAGED>::new(memory, u32::MAX);
    let view = InstructionView {
        guest_start: ir.guest_start,
        guest_len: ir.guest_len,
        ops: &ir.ops,
    };
    if PAGED {
        return execute::<false, 0, PAGED, V86>(
            ir,
            &view,
            cpu,
            &mem,
            &mut scratch,
            context,
            task_switch,
        );
    }
    execute_instruction::<false, 0, PAGED, V86>(
        ir,
        &InstructionView {
            guest_start: ir.guest_start,
            guest_len: ir.guest_len,
            ops: &ir.ops,
        },
        cpu,
        &LegacyAccess::<false, 0, PAGED>::new(memory, u32::MAX),
        &mut scratch,
        context,
        task_switch,
    )
    .map(|outcome| (outcome, None))
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct PlanKey {
    addr: u32,
    size: u8,
    access: AccessKind,
    privilege: Privilege,
    cr3: u32,
    cr0: u32,
}

// One instruction/REP iteration owns these spans. No page-table lookup after
// the first operand store: operand bytes may themselves alias a later PTE.
pub(crate) struct PortPlan {
    inputs: Box<CpuState>,
    mapping_identity: std::sync::Arc<()>,
    spans: Vec<Fragment>,
    completed: usize,
    pending_mmio_linear: Option<u32>,
}

impl PortPlan {
    pub(crate) fn pending_mmio_linear_address(&self) -> Option<u32> {
        self.pending_mmio_linear
    }
    pub(crate) fn matches_owner(&self, cpu: &CpuState, memory: &Memory) -> bool {
        let Some(p) = cpu.port_string_in else {
            return false;
        };
        matches_cpu_inputs(&self.inputs, cpu)
            && std::sync::Arc::ptr_eq(&self.mapping_identity, &memory.mapping_identity())
            && self.inputs.pending_mmio == cpu.pending_mmio
            && (self.pending_mmio_linear.is_some()
                || self.inputs.pending_mmio_write == cpu.pending_mmio_write)
            && self.inputs.port_string_in.is_some_and(|owner| {
                // Only the requested port value may change the retained handoff.
                owner.rip == p.rip
                    && owner.next_ip == p.next_ip
                    && owner.instruction_end == p.instruction_end
                    && owner.base == p.base
                    && owner.offset == p.offset
                    && owner.next_index == p.next_index
                    && owner.next_count == p.next_count
                    && owner.addr_bits == p.addr_bits
                    && owner.elem == p.elem
                    && owner.rep == p.rep
                    && match (owner.value, p.value, cpu.pending_port_in) {
                        (None, None, Some(size)) => size == p.elem,
                        (None, Some(_), None) => true,
                        (Some(before), Some(after), None) => before == after,
                        _ => false,
                    }
            })
    }
}

#[derive(Default)]
pub(crate) struct Transaction {
    mapping_identity: Option<std::sync::Arc<()>>,
    plans: Vec<(PlanKey, Vec<Fragment>)>,
    overlay: Vec<(u64, u8)>,
    preparing: bool,
    device_dummy: bool,
    frame_reads: bool,
    reads: Vec<(u64, u8, u64)>,
    prepared_reads: Vec<(u64, u8, Option<u64>)>,
    writes: Vec<(u64, u8, u64, bool)>,
    read_cursor: usize,
    write_cursor: usize,
    inputs: Option<Box<CpuState>>,
    task_phase: Option<(tasks::Stage, usize, usize)>,
    pending_mmio_linear: Option<u32>,
}

impl Transaction {
    pub(crate) fn matches_memory(&self, memory: &Memory) -> bool {
        same_arc(&self.mapping_identity, &Some(memory.mapping_identity()))
    }

    pub(crate) fn pending_mmio_linear_address(&self) -> Option<u32> {
        self.pending_mmio_linear
    }

    pub(crate) fn matches_inputs(&self, cpu: &CpuState) -> bool {
        let Some(inputs) = &self.inputs else {
            return true;
        };
        matches_cpu_inputs(inputs, cpu)
    }
}

fn matches_cpu_inputs(inputs: &CpuState, cpu: &CpuState) -> bool {
    macro_rules! same {
            ($($field:ident),* $(,)?) => {{
                let CpuState {
                    $($field: _,)*
                    flags: _, fpu_pad: _, pending_port_in: _,
                    pending_mmio: _, pending_mmio_write: _, port_string_in: _,
                    mmio_parts: _, mmio_parts_len: _, mmio_parts_rip: _,
                    mmio_retry_active: _, mmio_retry_cs: _,
                    cpuid_table: _, cpu_source: _,
                } = cpu;
                true $(&& inputs.$field == cpu.$field)*
            }};
        }
    // Completion slots and the transfer ledger are outputs of the bus handoff.
    // Numeric and system inputs own the retained image just as GPRs do.
    same!(
        gpr,
        rip,
        rflags_system,
        legacy,
        cs,
        ds,
        es,
        ss,
        fs,
        gs,
        fs_base,
        gs_base,
        xmm,
        ymm_hi,
        zmm_hi,
        kmask,
        mxcsr,
        fpr,
        fpu_top,
        fpu_cw,
        fpu_sw,
        fpu_empty,
        fpu_fip,
        fpu_fdp,
        fpu_fop,
        fpu_fcs,
        fpu_fds,
        fpu_model,
        fpu_emulation,
        x87_precision,
        x87_pointer_model,
        features,
        machine_flags_mask,
        debug_test_registers,
        numeric_error_route,
        external_numeric_error,
        task_io_map_base,
        task_debug_trap,
        cpuid_overrides,
        source_id,
        random_state,
    ) && inputs.flags == cpu.flags
        && same_arc(&inputs.cpuid_table, &cpu.cpuid_table)
        && same_arc(&inputs.cpu_source, &cpu.cpu_source)
}

fn same_arc<T: ?Sized>(a: &Option<std::sync::Arc<T>>, b: &Option<std::sync::Arc<T>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => std::sync::Arc::ptr_eq(a, b),
        (None, None) => true,
        _ => false,
    }
}

fn preparing<const ROUTED: bool, const ORDER: u8>(mem: &impl LegacyMemory<ROUTED, ORDER>) -> bool {
    mem.transaction().is_some_and(|tx| tx.borrow().preparing)
}

#[cold]
#[inline(never)]
fn finish_plan<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    tx: &std::cell::RefCell<Transaction>,
) -> Result<bool, Exit> {
    let mut tx = tx.borrow_mut();
    let mut dirtied = false;
    for (key, spans) in &tx.plans {
        if key.access != AccessKind::Write {
            continue;
        }
        for f in spans {
            dirtied |= f.translation.pte & DIRTY == 0;
            if table_or(
                cpu,
                mem,
                f.linear,
                f.translation.pte_address,
                PagingWalkLevel::Table,
                f.translation.pte,
                DIRTY,
            )?
            .is_none()
            {
                return Ok(false);
            }
        }
    }
    // A source can alias a PTE whose D bit was just set. Re-evaluate the
    // prepared instruction with those physical walk writes already visible.
    if !dirtied {
        for (key, spans) in &mut tx.plans {
            if key.access == AccessKind::Write {
                for f in spans {
                    f.dirty_complete = true;
                }
            }
        }
    }
    Ok(!dirtied)
}

#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(super) fn execute<const ROUTED: bool, const ORDER: u8, const PAGED: bool, const V86: bool>(
    block: &IrBlock,
    ir: &InstructionView<'_>,
    cpu: &mut CpuState,
    original: &LegacyAccess<'_, ROUTED, ORDER, PAGED>,
    temps: &mut [u64],
    context: MachineContext,
    task_switch: &mut tasks::SwitchState,
) -> Result<(Outcome, Option<PortPlan>), Exit>
where
    PagingMode<PAGED>: PagingStorage,
{
    execute_with_retry::<ROUTED, ORDER, PAGED, V86>(
        block,
        ir,
        cpu,
        original,
        temps,
        context,
        task_switch,
        &mut None,
    )
}

#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn execute_with_retry<const ROUTED: bool, const ORDER: u8, const PAGED: bool, const V86: bool>(
    block: &IrBlock,
    ir: &InstructionView<'_>,
    cpu: &mut CpuState,
    original: &LegacyAccess<'_, ROUTED, ORDER, PAGED>,
    temps: &mut [u64],
    context: MachineContext,
    task_switch: &mut tasks::SwitchState,
    retry: &mut Option<Transaction>,
) -> Result<(Outcome, Option<PortPlan>), Exit>
where
    PagingMode<PAGED>: PagingStorage,
{
    loop {
        if retry
            .as_ref()
            .is_some_and(|tx| !tx.matches_memory(original.raw()))
        {
            return Err(refusal(
                cpu,
                crate::UnsupportedSystemOperation::LegacyEntryFacility,
            ));
        }
        let retained = retry.take();
        let resuming = retained.is_some();
        let tx = std::cell::RefCell::new(retained.unwrap_or_else(|| Transaction {
            mapping_identity: Some(original.raw().mapping_identity()),
            preparing: true,
            device_dummy: ir.ops.iter().all(|op| match op {
                IrOp::Legacy(op) => matches!(
                    op,
                    L::Checkpoint
                        | L::I486Gate
                        | L::Address { .. }
                        | L::ReadSelector { .. }
                        | L::Push { .. }
                        | L::Pop { .. }
                        | L::AdjustStack { .. }
                        | L::String { .. }
                        | L::Enter { .. }
                        | L::Leave { .. }
                        | L::All { .. }
                        | L::Flags { .. }
                        | L::Table { .. }
                        | L::Privileged { .. }
                        | L::LoadMsw { .. }
                        | L::Coprocessor { .. }
                        | L::CoprocessorOverrun { .. }
                ),
                IrOp::Jump { .. } | IrOp::Branch { .. } => false,
                _ => true,
            }),
            ..Transaction::default()
        }));
        {
            let mut tx = tx.borrow_mut();
            // Task continuations resume their saved phase; ordinary instructions
            // replay from their checkpoint and consume their earlier visits.
            let task_resume = resuming && task_switch.context().is_some();
            let (read, write) = if task_resume {
                tx.task_phase
                    .map_or((tx.reads.len(), tx.writes.len()), |(_, read, write)| {
                        (read, write)
                    })
            } else {
                (0, 0)
            };
            tx.read_cursor = read;
            tx.write_cursor = write;
        }
        let mut mem =
            LegacyAccess::<ROUTED, ORDER, true>::new(&original.access.0, original.physical_mask);
        mem.writes = original.writes;
        mem.paging = Some(&tx);
        if !resuming || tx.borrow().preparing {
            if resuming {
                tx.borrow_mut().prepared_reads.clear();
                tx.borrow_mut().overlay.clear();
            }
            let mut preview_cpu = cpu.clone();
            let mut preview_tasks = task_switch.clone();
            let mut preview_temps = temps.to_vec();
            let preview = execute_instruction::<ROUTED, ORDER, true, V86>(
                block,
                ir,
                &mut preview_cpu,
                &mem,
                &mut preview_temps,
                context,
                &mut preview_tasks,
            );
            if let Err(exit) = preview {
                if tx.borrow().frame_reads && matches!(exit, Exit::MmioRead { .. }) {
                    retain_frame_answers(cpu, &preview_cpu);
                    tx.borrow_mut().inputs = Some(Box::new(cpu.clone()));
                    *retry = Some(tx.into_inner());
                    return Err(exit);
                }
                if !preview_tasks.committed {
                    publish_cr2(cpu, &exit);
                    return Err(exit);
                }
                // Hardware task switches have their own commit boundary. Replay the
                // outgoing saves and incoming commit even when the new task faults
                // (80386 PRM §7.5 pp. 7-8–9; design §3.7).
            }
            if !finish_plan(cpu, &mem, &tx)? {
                continue;
            }
            let mut tx = tx.borrow_mut();
            tx.preparing = false;
            tx.read_cursor = 0;
        }
        let before_commit = cpu.clone();
        let result = execute_instruction::<ROUTED, ORDER, true, V86>(
            block,
            ir,
            cpu,
            &mem,
            temps,
            context,
            task_switch,
        );
        let outcome = match result {
            Ok(outcome) => outcome,
            Err(exit) => {
                if matches!(
                    exit,
                    Exit::MmioRead { .. } | Exit::MmioWrite { .. } | Exit::UnmappedMemory { .. }
                ) {
                    tx.borrow_mut().inputs = Some(Box::new(cpu.clone()));
                    *retry = Some(tx.into_inner());
                }
                return Err(exit);
            }
        };
        // RAM stores commit only after every irreversible device transfer has
        // completed. Reads below see the staged stores in instruction order.
        let next_ip = cpu.rip;
        if !task_switch.committed {
            cpu.rip = before_commit.rip;
        }
        if let Err(exit) = commit_ram(cpu, &mem) {
            if !task_switch.committed {
                restore_destinations(cpu, before_commit);
            }
            tx.borrow_mut().inputs = Some(Box::new(cpu.clone()));
            *retry = Some(tx.into_inner());
            return Err(exit);
        }
        cpu.rip = next_ip;
        let input = cpu.port_string_in.map(|p| {
            let linear = (p.base as u32).wrapping_add(p.offset as u32);
            let spans = tx
                .borrow()
                .plans
                .iter()
                .find(|(key, _)| {
                    key.addr == linear && key.size == p.elem && key.access == AccessKind::Write
                })
                .expect("preflighted INS destination")
                .1
                .clone();
            PortPlan {
                inputs: Box::new(cpu.clone()),
                mapping_identity: original.raw().mapping_identity(),
                spans,
                completed: 0,
                pending_mmio_linear: None,
            }
        });
        return Ok((outcome, input));
    }
}

#[cold]
#[inline(never)]
pub(super) fn byte_bus<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    linear: u64,
    access: AccessKind,
) -> Result<u64, Exit> {
    Ok(plan(cpu, mem, linear, 1, access, false)?[0].bus)
}

#[cold]
#[inline(never)]
pub(super) fn probe_descriptor_cas<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    address: u32,
) -> Result<(), Exit> {
    let spans = plan(cpu, mem, u64::from(address), 1, AccessKind::Write, false)?;
    if mem.raw().region_kind(spans[0].bus, 1) == Ok(crate::RegionKind::Trap) {
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyInstruction,
        ));
    }
    Ok(())
}

#[cold]
#[inline(never)]
pub(super) fn descriptor_cas<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    address: u32,
    expected: u64,
    value: u64,
) -> Result<u64, Exit> {
    probe_descriptor_cas(cpu, mem, address)?;
    if preparing(mem) {
        let observed = read(cpu, mem, u64::from(address), 1, AccessKind::Write)?;
        // The committing phase uses a physical CAS rather than an operand
        // read visit. Keep the ordinary operand snapshot cursor aligned.
        if let Some(tx) = mem.transaction() {
            let mut tx = tx.borrow_mut();
            tx.prepared_reads.pop();
            tx.read_cursor -= 1;
        }
        if observed == expected {
            write(cpu, mem, u64::from(address), 1, value)?;
        }
        return Ok(observed);
    }
    let f = &plan(cpu, mem, u64::from(address), 1, AccessKind::Write, true)?[0];
    let observed = mem
        .raw()
        .atomic_cas(f.bus, expected, value, 1)
        .map_err(|trap| {
            let StepResult::Exit(exit) =
                trap_out(cpu, cpu.rip, trap, f.bus, 1, AccessKind::Write, value).into_step()
            else {
                unreachable!()
            };
            crate::fault::qualify_legacy_bus(exit, u64::from(address), f.bus)
        })?;
    if observed == expected {
        mem.note_write(f.bus, 1);
    }
    Ok(observed)
}

#[cold]
#[inline(never)]
pub(super) fn locked<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    linear: u64,
    src: u64,
    size: u8,
    update: super::super::integer::LockedUpdate,
) -> Result<u64, Exit> {
    let spans = plan(cpu, mem, linear, size, AccessKind::Write, true)?;
    if !preparing(mem) && spans.len() == 1 {
        let f = &spans[0];
        let result = match update {
            super::super::integer::LockedUpdate::Cas(expected) => {
                mem.raw().atomic_cas(f.bus, expected, src, size)
            }
            super::super::integer::LockedUpdate::Rmw(op) => {
                mem.raw().atomic_rmw(f.bus, src, size, op)
            }
        };
        match result {
            Ok(value) => {
                mem.note_write(f.bus, size);
                return Ok(value);
            }
            Err(MemTrap::Mmio) => {}
            Err(trap) => {
                let StepResult::Exit(exit) =
                    trap_out(cpu, cpu.rip, trap, f.bus, size, AccessKind::Write, src).into_step()
                else {
                    unreachable!()
                };
                return Err(crate::fault::qualify_legacy_bus(exit, linear, f.bus));
            }
        }
    }
    // SDM Vol 3A §11.1.2.2 pp. 11-4–5; design §3.4: one retained
    // snapshot for the scatter operand, with the existing single-vCPU contract.
    // The transaction retains reads and completed writes across bus handoffs.
    let old = read(cpu, mem, linear, size, AccessKind::Write)?;
    write(cpu, mem, linear, size, update.apply(old, src, size))?;
    Ok(old)
}

// Architectural rollback must not resurrect an already consumed device answer.
// The translated transaction and its transfer visits live outside CpuState.
fn restore_destinations(cpu: &mut CpuState, mut before: CpuState) {
    std::mem::swap(&mut before.pending_mmio, &mut cpu.pending_mmio);
    std::mem::swap(&mut before.pending_mmio_write, &mut cpu.pending_mmio_write);
    std::mem::swap(&mut before.mmio_parts, &mut cpu.mmio_parts);
    std::mem::swap(&mut before.mmio_parts_len, &mut cpu.mmio_parts_len);
    std::mem::swap(&mut before.mmio_parts_rip, &mut cpu.mmio_parts_rip);
    std::mem::swap(&mut before.mmio_retry_active, &mut cpu.mmio_retry_active);
    std::mem::swap(&mut before.mmio_retry_cs, &mut cpu.mmio_retry_cs);
    *cpu = before;
}

struct FpBuffer {
    base: u64,
    bytes: std::cell::RefCell<Vec<u8>>,
    stored: Cell<bool>,
}
impl crate::x87::FpMem for FpBuffer {
    fn load(&self, addr: u64, out: &mut [u8]) -> bool {
        let offset = (addr as u32).wrapping_sub(self.base as u32) as usize;
        out.copy_from_slice(&self.bytes.borrow()[offset..offset + out.len()]);
        true
    }
    fn store(&self, addr: u64, bytes: &[u8]) -> bool {
        let offset = (addr as u32).wrapping_sub(self.base as u32) as usize;
        self.bytes.borrow_mut()[offset..offset + bytes.len()].copy_from_slice(bytes);
        self.stored.set(true);
        true
    }
}

#[cold]
#[inline(never)]
pub(super) fn x87<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    kind: crate::x87::FpuKind,
    address: u64,
    sti: u8,
) -> Result<(), Exit> {
    let bytes = crate::x87::mem_operand_bytes_format(kind, sti) as usize;
    let writing = crate::x87::mem_write_bytes(kind).is_some();
    // Strong image restart is engine policy (design §3.4). SDM Vol 3A
    // §11.1.1 pp. 11-2–3 permits partial stores for some x87 instructions.
    let mut data = vec![0; bytes];
    for offset in (0..bytes).step_by(8) {
        let linear = u64::from((address as u32).wrapping_add(offset as u32));
        let size = (bytes - offset).min(8) as u8;
        if !writing {
            plan(cpu, mem, linear, size, AccessKind::Read, false)?;
        }
    }
    if !writing {
        for offset in (0..bytes).step_by(8) {
            let linear = u64::from((address as u32).wrapping_add(offset as u32));
            let size = (bytes - offset).min(8) as u8;
            let value = read(cpu, mem, linear, size, AccessKind::Read)?;
            data[offset..offset + usize::from(size)]
                .copy_from_slice(&value.to_le_bytes()[..usize::from(size)]);
        }
    }
    let fp = FpBuffer {
        base: address,
        bytes: std::cell::RefCell::new(data),
        stored: Cell::new(false),
    };
    let before = cpu.clone();
    let fault = crate::x87::exec_x87(cpu, &fp, kind, address, sti);
    debug_assert!(fault.is_none());
    if fp.stored.get() {
        for offset in (0..bytes).step_by(8) {
            let linear = u64::from((address as u32).wrapping_add(offset as u32));
            if let Err(exit) = probe(cpu, mem, linear, (bytes - offset).min(8) as u8) {
                restore_destinations(cpu, before);
                return Err(exit);
            }
        }
        let data = fp.bytes.borrow();
        for offset in (0..bytes).step_by(8) {
            let size = (bytes - offset).min(8);
            let mut value = [0; 8];
            value[..size].copy_from_slice(&data[offset..offset + size]);
            let linear = u64::from((address as u32).wrapping_add(offset as u32));
            if let Err(exit) = write(cpu, mem, linear, size as u8, u64::from_le_bytes(value)) {
                restore_destinations(cpu, before);
                return Err(exit);
            }
        }
    }
    Ok(())
}

#[cold]
#[inline(never)]
pub(crate) fn deliver_interrupt<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    memory: &Memory,
    context: MachineContext,
    event: legacy::LegacyEvent,
    task_switch: &mut tasks::SwitchState,
    retry: &mut Option<Transaction>,
) -> Result<(), Exit> {
    // A preflight fault belongs to this delivery, even after a previous task
    // completed. Match the unpaged delivery's per-attempt completion boundary.
    task_switch.committed = false;
    task_switch.completed = false;
    loop {
        if retry.as_ref().is_some_and(|tx| !tx.matches_memory(memory)) {
            return Err(refusal(
                cpu,
                crate::UnsupportedSystemOperation::LegacyEntryFacility,
            ));
        }
        let retained = retry.take();
        let resuming = retained.is_some();
        let tx = std::cell::RefCell::new(retained.unwrap_or_else(|| Transaction {
            mapping_identity: Some(memory.mapping_identity()),
            preparing: true,
            ..Transaction::default()
        }));
        {
            let mut tx = tx.borrow_mut();
            // Task continuations resume their saved phase; ordinary instructions
            // replay from their checkpoint and consume their earlier visits.
            let task_resume = resuming && task_switch.context().is_some();
            let (read, write) = if task_resume {
                tx.task_phase
                    .map_or((tx.reads.len(), tx.writes.len()), |(_, read, write)| {
                        (read, write)
                    })
            } else {
                (0, 0)
            };
            tx.read_cursor = read;
            tx.write_cursor = write;
        }
        let mut mem = LegacyAccess::<ROUTED, ORDER, true>::new(memory, u32::MAX);
        mem.paging = Some(&tx);
        if !resuming {
            let mut preview_cpu = cpu.clone();
            let mut preview_tasks = task_switch.clone();
            let preview = if preview_tasks.context().is_some() {
                tasks::resume(&mut preview_cpu, &mem.address_keyed(), &mut preview_tasks)
            } else {
                deliver(&mut preview_cpu, &mem, context, event, &mut preview_tasks)
            };
            if let Err(exit) = preview {
                if !preview_tasks.committed {
                    publish_cr2(cpu, &exit);
                    return Err(exit);
                }
            }
            if !finish_plan(cpu, &mem, &tx)? {
                continue;
            }
            let mut tx = tx.borrow_mut();
            tx.preparing = false;
            tx.read_cursor = 0;
        }
        let before_commit = cpu.clone();
        let mut result = if task_switch.context().is_some() {
            tasks::resume(cpu, &mem.address_keyed(), task_switch)
        } else {
            deliver(cpu, &mem, context, event, task_switch)
        };
        if result.is_ok() {
            let next_ip = cpu.rip;
            if !task_switch.committed {
                cpu.rip = before_commit.rip;
            }
            if let Err(exit) = commit_ram(cpu, &mem) {
                if !task_switch.committed {
                    restore_destinations(cpu, before_commit);
                }
                result = Err(exit);
            } else {
                cpu.rip = next_ip;
            }
        }
        if let Err(exit) = &result {
            publish_cr2(cpu, exit);
            if matches!(
                exit,
                Exit::MmioRead { .. } | Exit::MmioWrite { .. } | Exit::UnmappedMemory { .. }
            ) {
                tx.borrow_mut().inputs = Some(Box::new(cpu.clone()));
                *retry = Some(tx.into_inner());
            }
        }
        return result;
    }
}

#[cold]
#[inline(never)]
pub(crate) fn finish_port_input<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    memory: &Memory,
    plan: &mut Option<PortPlan>,
) -> Result<(), Exit> {
    let p = cpu.port_string_in.expect("pending input iteration");
    let saved = plan
        .as_mut()
        .filter(|saved| saved.matches_owner(cpu, memory))
        .ok_or_else(|| refusal(cpu, crate::UnsupportedSystemOperation::LegacyEntryFacility))?;
    let Some(value) = p.value else {
        return Err(Exit::port_io(cpu.gpr[RDX] as u16, p.elem, PortDir::In, 0));
    };
    saved.inputs.port_string_in.as_mut().unwrap().value = Some(value);
    let mem = LegacyAccess::<ROUTED, ORDER, true>::new(memory, u32::MAX);
    // The port read is irreversible. Retain this iteration's translated spans
    // over the handoff, even if the embedder edits its PTEs before completion.
    while saved.completed < saved.spans.len() {
        let f = &saved.spans[saved.completed];
        let value = (value >> f.shift) & mask(f.bytes);
        if mem.raw().region_kind(f.bus, f.bytes) == Ok(crate::RegionKind::Trap) {
            if !cpu.pending_mmio_write {
                saved.pending_mmio_linear = Some(f.linear);
                return Err(transfer_failure(
                    cpu,
                    &mem,
                    MemTrap::Mmio,
                    f,
                    AccessKind::Write,
                    value,
                ));
            }
            cpu.pending_mmio_write = false;
            cpu.record_mmio_part(
                cpu.rip,
                f.bus | ((saved.completed as u64) << 32) | (1 << 63),
                value,
            );
            mem.raw().note_write(f.bus, usize::from(f.bytes));
        } else if let Err(trap) = mem.raw().write(f.bus, value, f.bytes) {
            return Err(transfer_failure(
                cpu,
                &mem,
                trap,
                f,
                AccessKind::Write,
                value,
            ));
        }
        mem.note_write(f.bus, f.bytes);
        saved.completed += 1;
        saved.pending_mmio_linear = None;
    }
    *plan = None;
    cpu.write_gpr(RDI, p.next_index, p.addr_bits / 8);
    if p.rep {
        cpu.write_gpr(RCX, p.next_count, p.addr_bits / 8);
    }
    cpu.rip = p.next_ip;
    cpu.port_string_in = None;
    cpu.clear_mmio_parts();
    Ok(())
}

// A task phase can contain several reads (the incoming image or a segment
// descriptor). Retry that phase's visits without replaying earlier phases.
#[cold]
#[inline(never)]
pub(super) fn task_phase<const ROUTED: bool, const ORDER: u8>(
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    stage: tasks::Stage,
) {
    if preparing(mem) {
        return;
    }
    let mut tx = mem
        .transaction()
        .expect("paged task transaction")
        .borrow_mut();
    if tx.task_phase.is_none_or(|(old, _, _)| old != stage) {
        tx.task_phase = Some((stage, tx.read_cursor, tx.write_cursor));
    }
}

// SDM Vol 3A §10.3 steps 6, 10 pp. 10-10–11: outgoing saves precede
// clearing the outgoing busy bit, even when their physical storage overlaps.
#[cold]
#[inline(never)]
pub(super) fn task_saves_complete<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
) -> Result<(), Exit> {
    if preparing(mem) {
        return Ok(());
    }
    commit_ram(cpu, mem)
}

// Task switches have an architectural commit seam; ordinary instruction
// preflight must stop there rather than roll an incoming-task fault back.
#[cold]
#[inline(never)]
pub(super) fn task_commit<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
) -> Result<(), Exit> {
    if preparing(mem) {
        // Internal preflight stop, consumed only after SwitchState::committed.
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyTaskPaging,
        ));
    }
    commit_ram(cpu, mem)?;
    // Keep the outgoing plans through the incoming image read. The saves may
    // alias its paging structures; that image was preflighted before commit.
    Ok(())
}

#[cold]
#[inline(never)]
pub(super) fn task_cr3_loaded<const ROUTED: bool, const ORDER: u8>(
    mem: &impl LegacyMemory<ROUTED, ORDER>,
) {
    // Even an equal CR3 load starts fresh translations. Outgoing TSS saves
    // can alias the page tables used by incoming descriptor loads (§3.7).
    mem.transaction()
        .expect("paged task transaction")
        .borrow_mut()
        .plans
        .clear();
}

#[cold]
#[inline(never)]
pub(super) fn task_resident<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    tss: legacy::SystemSegment,
) -> Result<(), Exit> {
    // SDM Vol 3A §10.3 step 5 p. 10-10: the entire incoming TSS image
    // must be resident before saves/TR. Cover static fields as well as the
    // register fields below; do not read unused fields from devices.
    let bytes = if tasks::wide(tss) { 0x68 } else { 0x2c };
    for offset in (0..bytes).step_by(8) {
        plan(
            cpu,
            &mem.system(),
            u64::from(tss.cache.base.wrapping_add(offset)),
            (bytes - offset).min(8) as u8,
            AccessKind::Read,
            false,
        )?;
    }
    // Plans are keyed by exact span. Retain each image read's translation
    // across outgoing saves that can overwrite the incoming TSS's PTE.
    let mut field = |offset, size| {
        plan(
            cpu,
            &mem.system(),
            u64::from(tss.cache.base.wrapping_add(offset)),
            size,
            AccessKind::Read,
            false,
        )
        .map(|_| ())
    };
    if tasks::wide(tss) {
        // 386 PRM §7.1 Fig. 7-1 pp. 7-2–3: CR3, EIP, EFLAGS,
        // eight GPRs, six selectors, LDTR, T and I/O-map base.
        for offset in (0x1c..0x48).step_by(4) {
            field(offset, 4)?;
        }
        for offset in (0x48..=0x60).step_by(4).chain([0x64, 0x66]) {
            field(offset, 2)?;
        }
    } else {
        // 286 PRM §8.2 Fig. 8-1 pp. 8-2–3: IP through LDTR.
        for offset in (0x0e..0x2c).step_by(2) {
            field(offset, 2)?;
        }
    }
    Ok(())
}

#[cold]
#[inline(never)]
pub(super) fn task_image<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    tss: legacy::SystemSegment,
) -> Result<(tasks::Image, Option<u32>), Exit> {
    task_resident(cpu, mem, tss)?;
    // 80386 PRM §7.1 Fig. 7-1 pp. 7-2–3; §7.5 p. 7-8:
    // CR3 is a static incoming field, read through the outgoing address space.
    let cr3 = if tasks::wide(tss) {
        Some(read(
            cpu,
            &mem.system(),
            u64::from(tss.cache.base.wrapping_add(0x1c)),
            4,
            AccessKind::Read,
        )? as u32)
    } else {
        None
    };
    Ok((tasks::load_image(cpu, mem, tss)?, cr3))
}

#[cold]
#[inline(never)]
pub(crate) fn execute_retained<const ROUTED: bool, const ORDER: u8, const V86: bool>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    memory: &Memory,
    context: MachineContext,
    task_switch: &mut tasks::SwitchState,
    retry: &mut Option<Transaction>,
) -> Result<(Outcome, Option<PortPlan>), Exit> {
    let mut scratch = vec![0; ir.temp_count as usize];
    let mem = LegacyAccess::<ROUTED, ORDER, true>::new(memory, u32::MAX);
    let view = InstructionView {
        guest_start: ir.guest_start,
        guest_len: ir.guest_len,
        ops: &ir.ops,
    };
    execute_with_retry::<ROUTED, ORDER, true, V86>(
        ir,
        &view,
        cpu,
        &mem,
        &mut scratch,
        context,
        task_switch,
        retry,
    )
}
