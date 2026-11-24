//! Internal Machine translation boundary (paging design §§3.1–3.4, spec §17.10).
//! No translation cache: every scalar plan owns its resolved physical spans.
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

#[derive(Clone, Copy)]
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
        let pte_address = (pde & FRAME) + 4 * ((linear >> 12) & 0x3ff);
        let pte = table_read(cpu, mem, linear, pte_address, PagingWalkLevel::Table)?;
        if pte & PRESENT == 0 {
            return Err(fault(false).exit(cpu));
        }
        // Intel486 PRM §9.9.14 p. 9-21 requires both A bits on protection
        // faults. The same failed-walk order on the 386 is engine policy (§3.3).
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
        // 80386 PRM §§6.4.1–6.4.2 pp. 6-20–22, Table 6-5;
        // Intel486 PRM §§6.8.1–6.8.2 pp. 6-23–25, Table 6-4 (WP=0).
        let rights = pde & pte;
        if privilege.user()
            && (rights & USER == 0 || (access == AccessKind::Write && rights & WRITE == 0))
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

struct Fragment {
    linear: u32,
    bus: u64,
    bytes: u8,
    shift: u8,
    translation: Translation,
}

fn plan<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    addr: u64,
    size: u8,
    access: AccessKind,
    dirty: bool,
) -> Result<Vec<Fragment>, Exit> {
    let privilege = if cpu.legacy.cpl == 3 {
        Privilege::User
    } else {
        Privilege::Supervisor
    };
    'restart: loop {
        let mut fragments = Vec::new();
        let mut consumed = 0;
        while consumed < size {
            let linear = (addr as u32).wrapping_add(u32::from(consumed));
            let translation =
                translate_linear(cpu, mem, linear, size - consumed, access, privilege)?;
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
        if dirty {
            // 80386 PRM §5.2.4.3 p. 5-10; Intel486 PRM §5.3.4.3
            // pp. 5-21–22: PTE.D precedes the operand write; PDE.D is preserved.
            // Preflight the entire span before D, retaining even aliased frames.
            for fragment in &fragments {
                let t = fragment.translation;
                if table_or(
                    cpu,
                    mem,
                    fragment.linear,
                    t.pte_address,
                    PagingWalkLevel::Table,
                    t.pte,
                    DIRTY,
                )?
                .is_none()
                {
                    continue 'restart;
                }
            }
        }
        return Ok(fragments);
    }
}

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
        value |= read_physical(cpu, mem, u64::from(f.linear), f.bus, f.bytes)? << f.shift;
    }
    Ok(value)
}

pub(super) fn write<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    addr: u64,
    size: u8,
    value: u64,
) -> Result<(), Exit> {
    let fragments = plan(cpu, mem, addr, size, AccessKind::Write, true)?;
    for f in fragments {
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

pub(super) fn probe<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    addr: u64,
    size: u8,
) -> Result<(), Exit> {
    plan(cpu, mem, addr, size, AccessKind::Write, false).map(|_| ())
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
    let t = translate_linear(cpu, &mem, linear, 1, AccessKind::Execute, privilege)?;
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
        || (model != crate::CpuModel::I80386 && cpu.legacy.cr0 & (1 << 16) != 0)
    {
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyEntryFacility,
        ));
    }
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
    execute_for_test::<true>(&ir, cpu, memory, context, false).map(|_| ())
}

#[cfg(feature = "test-hooks")]
pub(crate) fn execute_for_test<const PAGED: bool>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    memory: &Memory,
    context: MachineContext,
    controls: bool,
) -> Result<Outcome, Exit> {
    // Paging design §3.4: PUSH/POP memory forms need instruction-wide preflight;
    // a load/store pair is scalar only when it reuses one operand address (RMW).
    let addresses = ir
        .ops
        .iter()
        .filter(|op| matches!(op, IrOp::Legacy(L::Address { .. })))
        .count();
    if addresses > 1
        || ir.ops.iter().any(|op| match op {
            IrOp::Legacy(op) => {
                !(controls
                    && matches!(
                        op,
                        L::ReadControl { .. } | L::WriteControl { .. } | L::Privileged { .. }
                    ))
                    && !matches!(
                        op,
                        L::Checkpoint | L::I486Gate | L::Address { .. } | L::ReadSelector { .. }
                    )
            }
            op if machine_specific(op) => !matches!(
                op,
                IrOp::Load { .. } | IrOp::Store { .. } | IrOp::SegLimitCheck { .. }
            ),
            _ => false,
        })
    {
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyInstruction,
        ));
    }
    let mut scratch = vec![0; ir.temp_count as usize];
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
        &mut tasks::SwitchState::default(),
    )
}
