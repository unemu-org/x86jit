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

// Only an already-paged private interpreter reaches this helper. Public
// PG-setting writes continue through write_cr0 and its facility refusal.
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
        // Paging design §§3.4, 5 step 5: a split device operand needs a stable
        // plan and operand snapshot across handoffs. Refuse before any transfer
        // or D update until that continuation exists, including region splits
        // inside a single translated page.
        let device = fragments.iter().any(|f| {
            (0..f.bytes).any(|byte| {
                mem.raw().region_kind(f.bus + u64::from(byte), 1) == Ok(crate::RegionKind::Trap)
            })
        });
        if device
            && (fragments.len() != 1
                || mem.raw().region_kind(fragments[0].bus, size) != Ok(crate::RegionKind::Trap))
        {
            return Err(refusal(
                cpu,
                crate::UnsupportedSystemOperation::PagingSplitDevice,
            ));
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
        let part = match read_physical(cpu, mem, u64::from(f.linear), f.bus, f.bytes) {
            Err(Exit::MmioRead { .. }) if preparing(mem) => {
                if !mem.transaction().unwrap().borrow().device_dummy {
                    return Err(refusal(
                        cpu,
                        crate::UnsupportedSystemOperation::PagingSplitDevice,
                    ));
                }
                0
            }
            other => other?,
        };
        let part = if preparing(mem) {
            let tx = mem.transaction().unwrap().borrow();
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
            part
        } else {
            part
        };
        value |= part << f.shift;
    }
    Ok(value)
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
        write_physical(
            cpu,
            mem,
            u64::from(f.linear),
            f.bus,
            f.bytes,
            value >> f.shift,
        )?;
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
#[cfg(feature = "test-hooks")]
pub(crate) fn fetch_bus_for_test(
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
    let mut mem = LegacyAccess::<false, 0, true>::new(memory, u32::MAX);
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
    execute_for_test::<true>(
        &ir,
        cpu,
        memory,
        context,
        &mut tasks::SwitchState::default(),
    )
    .map(|_| ())
}

#[cfg(feature = "test-hooks")]
pub(crate) fn execute_for_test<const PAGED: bool>(
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
        return execute(ir, &view, cpu, &mem, &mut scratch, context, task_switch);
    }
    execute_instruction::<false, 0, PAGED>(
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
#[cfg_attr(not(feature = "test-hooks"), allow(dead_code))]
pub(crate) struct PortPlan {
    owner: u64,
    spans: Vec<Fragment>,
}

#[derive(Default)]
pub(crate) struct Transaction {
    plans: Vec<(PlanKey, Vec<Fragment>)>,
    overlay: Vec<(u64, u8)>,
    preparing: bool,
    device_dummy: bool,
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
    let tx = tx.borrow();
    // Multiple device operands need retained completions across handoffs (design
    // §5 step 5). Preflight has already selected any later #PF before refusal.
    let device = tx.plans.iter().any(|(_, spans)| {
        spans.iter().any(|f| {
            (0..f.bytes).any(|b| {
                mem.raw().region_kind(f.bus + u64::from(b), 1) == Ok(crate::RegionKind::Trap)
            })
        })
    });
    if device {
        let first = &tx.plans[0].0;
        if tx
            .plans
            .iter()
            .any(|(k, _)| k.addr != first.addr || k.size != first.size || k.cr3 != first.cr3)
        {
            return Err(refusal(
                cpu,
                crate::UnsupportedSystemOperation::PagingSplitDevice,
            ));
        }
    }
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
    Ok(!dirtied)
}

#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(super) fn execute<const ROUTED: bool, const ORDER: u8, const PAGED: bool>(
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
    loop {
        let tx = std::cell::RefCell::new(Transaction {
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
                ),
                IrOp::Jump { .. } | IrOp::Branch { .. } => false,
                _ => true,
            }),
            ..Transaction::default()
        });
        let mut mem =
            LegacyAccess::<ROUTED, ORDER, true>::new(&original.access.0, original.physical_mask);
        mem.writes = original.writes;
        mem.paging = Some(&tx);
        let mut preview_cpu = cpu.clone();
        let mut preview_tasks = task_switch.clone();
        let mut preview_temps = temps.to_vec();
        let preview = execute_instruction::<ROUTED, ORDER, true>(
            block,
            ir,
            &mut preview_cpu,
            &mem,
            &mut preview_temps,
            context,
            &mut preview_tasks,
        );
        if let Err(exit) = preview {
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
        tx.borrow_mut().preparing = false;
        let outcome = execute_instruction::<ROUTED, ORDER, true>(
            block,
            ir,
            cpu,
            &mem,
            temps,
            context,
            task_switch,
        )?;
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
                owner: p.rip,
                spans,
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
    // Split device shapes were refused by plan before any transfer/D update.
    let old = read(cpu, mem, linear, size, AccessKind::Write)?;
    write(cpu, mem, linear, size, update.apply(old, src, size))?;
    Ok(old)
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
                *cpu = before;
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
                *cpu = before;
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
) -> Result<(), Exit> {
    loop {
        let tx = std::cell::RefCell::new(Transaction {
            preparing: true,
            ..Transaction::default()
        });
        let mut mem = LegacyAccess::<ROUTED, ORDER, true>::new(memory, u32::MAX);
        mem.paging = Some(&tx);
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
        tx.borrow_mut().preparing = false;
        task_switch.committed = false;
        task_switch.completed = false;
        let result = if task_switch.context().is_some() {
            tasks::resume(cpu, &mem.address_keyed(), task_switch)
        } else {
            deliver(cpu, &mem, context, event, task_switch)
        };
        if let Err(exit) = &result {
            publish_cr2(cpu, exit);
        }
        return result;
    }
}

#[cfg(feature = "test-hooks")]
#[cold]
#[inline(never)]
pub(crate) fn finish_port_input(
    cpu: &mut CpuState,
    memory: &Memory,
    plan: &mut Option<PortPlan>,
) -> Result<(), Exit> {
    let p = cpu.port_string_in.expect("pending input iteration");
    let Some(value) = p.value else {
        return Err(Exit::port_io(cpu.gpr[RDX] as u16, p.elem, PortDir::In, 0));
    };
    let mem = LegacyAccess::<false, 0, true>::new(memory, u32::MAX);
    let saved = plan
        .as_ref()
        .filter(|saved| saved.owner == p.rip)
        .ok_or_else(|| refusal(cpu, crate::UnsupportedSystemOperation::LegacyEntryFacility))?;
    // The port read is irreversible. Retain this iteration's translated spans
    // over the handoff, even if the embedder edits its PTEs before completion.
    for f in &saved.spans {
        write_physical(
            cpu,
            &mem,
            u64::from(f.linear),
            f.bus,
            f.bytes,
            value >> f.shift,
        )?;
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

// Task switches have an architectural commit seam; ordinary instruction
// preflight must stop there rather than roll an incoming-task fault back.
#[cold]
#[inline(never)]
pub(super) fn task_commit<const ROUTED: bool, const ORDER: u8>(
    cpu: &CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
) -> Result<(), Exit> {
    if preparing(mem) {
        // Internal preflight stop, consumed only after SwitchState::committed.
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyTaskPaging,
        ));
    }
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
pub(super) fn task_image<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    tss: legacy::SystemSegment,
) -> Result<(tasks::Image, Option<u32>), Exit> {
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
