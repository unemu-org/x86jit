//! Machine memory and transactional system helpers for the shared interpreter
//! (decision-18 §§1, 4; spec.md §§8.1, 17).
use super::*;
pub(crate) mod paging;
mod privilege;
pub(crate) mod tasks;
pub(crate) mod virtual8086;
use crate::legacy::{
    self, ControlRegister, DebugTestRegister, DescriptorQuery, LegacyOp as L, MachineContext,
    SegmentId, SelectorDestination, SystemRegister,
};

#[derive(Default)]
pub(crate) struct Outcome {
    pub exit: Option<Exit>,
    pub retired: bool,
    pub context_changed: bool,
    pub translations_invalidated: bool,
    pub arm_sti: bool,
    /// The Vcpu `ss_shadow` bits this instruction arms: bit 0 the MOV/POP SS
    /// hold, bit 1 the NMI hold (spec.md §17.10).
    pub arm_ss: u8,
    pub suppress_debug: bool,
    preserve_rf: bool,
    /// This instruction delivered a software interrupt, whose origin is in
    /// `SwitchState::software_interrupt`, for the delivery log (spec.md
    /// §17.10, Delivery log). One byte, in the padding beside the flags above
    /// (`the_delivery_log_markers_fit_the_existing_layout`).
    pub software_interrupt: bool,
}

/// A delivered software interrupt and the origin it saved (spec.md §17.10,
/// Delivery log): the CS:IP its frame holds, the CS base and CPL before it.
#[derive(Clone, Copy, Default)]
pub(crate) struct SoftwareInterrupt {
    pub vector: u8,
    pub cs: u16,
    pub ip: u32,
    pub cs_base: u32,
    pub cpl: u8,
    pub vm: bool,
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

// Only retry-specialized alias walks need this scan. The unified multi-read
// families use the same ledger as Machine (80386 PRM §9.1, instruction
// restart); other alias loads bind their answers (`exec_load_retry`).
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
    // Intel486 PRM §12.2.1 Table 12-1 p. 12-3: NW=1/CD=0 is #GP(0).
    if cpu.machine_flags_mask != 0 && value & 0x6000_0000 == 0x2000_0000 {
        return Err(exception(cpu, 13));
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

/// Whether alignment checking is enabled: CR0.AM, EFLAGS.AC and CPL 3,
/// virtual-8086 mode included (SDM Vol 3A §7.15, Event 17—Alignment Check
/// Exception (#AC), p. 7-53; 486 PRM §4.1.1 p. 4-2). The 486 gate
/// (`LegacyOp::I486Gate`) refuses an instruction in this regime.
#[inline(always)]
pub(super) fn alignment_regime(cpu: &CpuState) -> bool {
    cpu.legacy.cpl == 3 && cpu.legacy.cr0 & (1 << 18) != 0 && cpu.rflags_system & (1 << 18) != 0
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

// SDM Vol 3B §23.2 p. 23-5; 386 PRM §15.1.2 pp. 15-2–3:
// segment values are paragraph numbers, including zero, for all six loads.
fn reload_virtual8086(cpu: &mut CpuState, segment: SegmentId, value: u16) {
    install(
        cpu,
        segment,
        value,
        legacy::virtual8086_cache(value, segment == SegmentId::Cs),
    );
}

/// A real-mode segment load. On the 80286 it also resets the limit to FFFFH
/// and "access rights is changed to a writable segment", which ends a cache
/// that LOADALL loaded (Intel, "Undocumented iAPX 286 Test Instruction",
/// `oracles/intel/i286-loadall.pdf`, p. 3); every real-mode segment has the
/// limit 0FFFFH (I286 PRM §5.1 p. 5-2, scan checked). The rights are the
/// engine's real-mode labels, the values of `CpuState::legacy_real_entry`,
/// which real-mode access does not check: 93h, writable data, for DS, ES and
/// SS, and 9Bh, execute/read code, for CS. 9Bh is ASSUMED: neither source
/// gives it, and p. 3's "writable" makes a CS-override write succeed where
/// the engine raises #GP if LMSW enters protected mode before CS is loaded
/// again.
fn reload_real(
    cpu: &mut CpuState,
    segment: SegmentId,
    value: u16,
    generation: crate::CpuGeneration,
) {
    reload(cpu, segment, value);
    if generation == crate::CpuGeneration::I286 {
        let cache = &mut cpu.legacy.segments[segment as usize];
        cache.limit = 0xffff;
        cache.attributes = if segment == SegmentId::Cs { 0x9b } else { 0x93 };
    }
}

/// The new IP of a real-mode transfer that loads CS. The 80286 reload sets
/// the CS limit to FFFFH (`reload_real`), so the current cache's limit does
/// not bound the new IP; the 80386 keeps the limit, so there it does.
fn far_real_target(
    cpu: &CpuState,
    context: MachineContext,
    value: u64,
    size: u8,
) -> Result<u64, Exit> {
    let offset = value & mask(size);
    if context.generation == crate::CpuGeneration::I286 && offset <= 0xffff {
        return Ok(offset);
    }
    target(cpu, context, value, size)
}

/// 80286 LOADALL (Intel, "Undocumented iAPX 286 Test Instruction",
/// `oracles/intel/i286-loadall.pdf`, pp. 1-4). p. 2 gives the 51 bus cycles
/// but not their order: the engine reads ascending words, and every read
/// precedes the first state change, so a fault or a device stop restarts
/// the instruction with nothing loaded (ASSUMED). The model is a later
/// stepping: the A1/B1 erratum that may enter protected mode (p. 15) is not
/// modelled (ASSUMED).
fn loadall<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
) -> Result<(), Exit> {
    // p. 1: in protected mode at a privilege level other than 0, exception
    // 13 with an error code of 0.
    if cpu.legacy.cr0 & 1 != 0 && cpu.legacy.cpl != 0 {
        return Err(fault(cpu, 13, 0));
    }
    // p. 2: 51 bus cycles from physical 000800H. No paging precedes the
    // 80286 bus, and the board's A20 gate applies as to any transfer.
    let mut table = [0; legacy::LOADALL_TABLE_BYTES];
    for (i, word) in table.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let address = u64::from(legacy::LOADALL_TABLE) + 2 * i as u64;
        *word = (read(cpu, mem, address, 2)? as u16).to_le_bytes();
    }
    let image = legacy::LoadallImage::parse(&table);
    // p. 2: LOADALL "cannot switch the 80286 from protected mode to real
    // mode", so PE stays set (DERIVED). The MSW loads as LMSW does (I286 PRM
    // App. B LMSW p. B-67): bits 3:0, the higher bits dropped (ASSUMED).
    let cr0 = legacy::lmsw_value(cpu.legacy.cr0, image.msw);
    let protected = cr0 & 1 != 0;
    if !image.supported(protected) {
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::I286LoadallTable,
        ));
    }
    cpu.legacy.cr0 = cr0;
    // p. 3: the DPLs of the CS and SS caches are the CPL; `supported` admits
    // DPL 0 alone in protected mode, and real mode is CPL 0.
    let cs = image.caches[SegmentId::Cs as usize];
    cpu.legacy.cpl = if protected {
        ((cs.attributes >> 5) & 3) as u8
    } else {
        0
    };
    let segments = [SegmentId::Cs, SegmentId::Ds, SegmentId::Es, SegmentId::Ss];
    for ((segment, selector), cache) in segments.into_iter().zip(image.selectors).zip(image.caches)
    {
        install(cpu, segment, selector, cache);
    }
    cpu.legacy.gdtr = image.gdtr;
    cpu.legacy.idtr = image.idtr;
    cpu.legacy.ldtr = image.ldtr;
    cpu.legacy.tr = image.tr;
    for (register, value) in image.gpr.into_iter().enumerate() {
        cpu.write_gpr(register, u64::from(value), 2);
    }
    // p. 1 says only that all registers load. FLAGS loads by the POPF and
    // IRET rule, so IOPL and NT load only in protected mode (I286 PRM App. B
    // IRET pp. B-51–52; §10.1 p. 10-1). ASSUMED.
    legacy::restore_flags(cpu, crate::CpuGeneration::I286, u32::from(image.flags))?;
    cpu.rip = u64::from(image.ip);
    Ok(())
}

// 80386 PRM §§6.3.1.1–6.3.1.2 pp. 6-4–6: retain Checked's
// whole-operand limit policy, including at FFFFFFFFh (decision-27).
#[inline]
fn flat32_address(
    cpu: &CpuState,
    segment: SegmentId,
    offset: u64,
    bytes: u8,
    access: AccessKind,
) -> Result<u64, Exit> {
    if offset
        .checked_add(u64::from(bytes.saturating_sub(1)))
        .is_none_or(|last| last > u64::from(u32::MAX))
    {
        return Err(legacy::segment_fault(
            cpu,
            if segment == SegmentId::Ss { 12 } else { 13 },
            offset,
            bytes,
            access,
        ));
    }
    Ok(offset)
}

#[cold]
#[inline(never)]
fn flat32_segment_access(
    cpu: &CpuState,
    context: MachineContext,
    segment: SegmentId,
    offset: u64,
    bytes: u8,
    access: AccessKind,
) -> Result<u64, Exit> {
    if !matches!(segment, SegmentId::Fs | SegmentId::Gs)
        && !(segment == SegmentId::Cs && access == AccessKind::Write)
    {
        return flat32_address(cpu, segment, offset, bytes, access);
    }
    legacy::access(cpu, context.generation, segment, offset, bytes, access)
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
    if matches!(
        context.segment_class,
        crate::SegmentClass::Checked | crate::SegmentClass::Segmented
    ) {
        return legacy::access(cpu, context.generation, segment, offset, bytes, access);
    }
    if context.segment_class == crate::SegmentClass::Flat32 {
        return flat32_segment_access(cpu, context, segment, offset, bytes, access);
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

// The unpaged adapter has zero paging storage: no extra field initialization
// or transaction-pointer lookup in its existing per-instruction construction.
pub(crate) struct PagingMode<const PAGED: bool>;
pub(crate) trait PagingStorage {
    type State<'a>: Default;
    fn fast_operand<'s>(state: &'s Self::State<'_>) -> Option<&'s paging::FastOperand>;
    fn fast_miss(_: &Self::State<'_>) {}
    fn tlb<'a>(state: &Self::State<'a>) -> Option<&'a std::cell::RefCell<paging::Tlb>>;
    fn transaction<'a>(
        state: &Self::State<'a>,
    ) -> Option<&'a std::cell::RefCell<paging::Transaction>>;
}
impl PagingStorage for PagingMode<false> {
    type State<'a> = ();
    fn fast_operand(_: &()) -> Option<&paging::FastOperand> {
        None
    }
    fn tlb<'a>(_: &Self::State<'a>) -> Option<&'a std::cell::RefCell<paging::Tlb>> {
        None
    }
    fn transaction<'a>(_: &Self::State<'a>) -> Option<&'a std::cell::RefCell<paging::Transaction>> {
        None
    }
}
impl PagingStorage for PagingMode<true> {
    type State<'a> = paging::AdapterState<'a>;
    fn fast_operand<'s>(state: &'s Self::State<'_>) -> Option<&'s paging::FastOperand> {
        state.fast_operand.as_ref()
    }
    fn fast_miss(state: &Self::State<'_>) {
        state.fast_missed.set(true);
    }
    fn tlb<'a>(state: &Self::State<'a>) -> Option<&'a std::cell::RefCell<paging::Tlb>> {
        state.tlb
    }
    fn transaction<'a>(
        state: &Self::State<'a>,
    ) -> Option<&'a std::cell::RefCell<paging::Transaction>> {
        state.transaction
    }
}

// A device read is identified by its visit as well as its bus address. The
// cursor restarts with the instruction, including reads served from the ledger.
// Writes retain their separate bit-63 namespace; descriptor snapshots use bit 62.
// `physical_mask` is the context model's bus mask (`legacy::physical_mask`).
struct LegacyAccess<'a, const ROUTED: bool, const ORDER: u8, const PAGED: bool = false>
where
    PagingMode<PAGED>: PagingStorage,
{
    access: &'a Access<ROUTED, ORDER>,
    read_index: Cell<u16>,
    writes: Option<&'a dyn Fn(u64, u8)>,
    physical_mask: u32,
    /// The frozen Real16 alias (decision-23 §6) keeps its whole-frame stack
    /// checks; see `wraps_in_segment`.
    frozen_alias: bool,
    paging: <PagingMode<PAGED> as PagingStorage>::State<'a>,
}
impl<'a, const ROUTED: bool, const ORDER: u8, const PAGED: bool>
    LegacyAccess<'a, ROUTED, ORDER, PAGED>
where
    PagingMode<PAGED>: PagingStorage,
{
    fn new(memory: &'a Memory, physical_mask: u32) -> Self {
        Self {
            access: Access::new(memory),
            read_index: Cell::new(0),
            writes: None,
            physical_mask,
            frozen_alias: false,
            paging: Default::default(),
        }
    }
    fn skip_read(&self, cpu: &mut CpuState, mut linear: u64, mut bytes: u8) -> Result<(), Exit> {
        if PAGED {
            return paging::read(cpu, self, linear, bytes, AccessKind::Read).map(|_| ());
        }
        while bytes != 0 {
            let (_, part) = legacy::physical_span(cpu, self.physical_mask, linear, bytes);
            self.read_index.set(self.read_index.get() + 1);
            linear = u64::from((linear as u32).wrapping_add(u32::from(part)));
            bytes -= part;
        }
        Ok(())
    }
}
impl<const ROUTED: bool, const ORDER: u8, const PAGED: bool> std::ops::Deref
    for LegacyAccess<'_, ROUTED, ORDER, PAGED>
where
    PagingMode<PAGED>: PagingStorage,
{
    type Target = Access<ROUTED, ORDER>;
    fn deref(&self) -> &Self::Target {
        self.access
    }
}

// Task-switch stages: the context's bus with address-based retry keys.
pub(crate) struct AddressKeyed<'a, M> {
    memory: &'a M,
}

struct Supervisor<'a, M>(&'a M);

// Instructions replay from their first read; task continuations resume saved
// stages, whose existing transfers retain address-based retry keys. Every
// view carries its model's physical-address mask (`legacy::physical_mask`).
trait LegacyMemory<const ROUTED: bool, const ORDER: u8> {
    const PAGED: bool = false;
    fn fast_operand(&self) -> Option<&paging::FastOperand> {
        None
    }
    fn fast_miss(&self) {}
    #[inline(always)]
    fn paged(&self) -> bool {
        Self::PAGED
    }
    fn tlb(&self) -> Option<&std::cell::RefCell<paging::Tlb>> {
        None
    }
    fn transaction(&self) -> Option<&std::cell::RefCell<paging::Transaction>> {
        None
    }
    fn supervisor(&self) -> bool {
        false
    }
    fn system(&self) -> Supervisor<'_, Self>
    where
        Self: Sized,
    {
        Supervisor(self)
    }
    fn raw(&self) -> &Access<ROUTED, ORDER>;
    fn read_key(&self, addr: u64) -> u64;
    /// The key `read_key` gives the read of `addr` that comes `ahead` reads
    /// after the next one, without moving the cursor.
    fn next_read_key(&self, addr: u64, ahead: u16) -> u64;
    fn skip_read(&self, cpu: &mut CpuState, linear: u64, bytes: u8) -> Result<(), Exit>;
    fn note_write(&self, _addr: u64, _size: u8) {}
    fn physical_mask(&self) -> u32;
    fn frozen_alias(&self) -> bool {
        false
    }
    /// The same bus, with address-based retry keys (task-switch stages).
    fn address_keyed(&self) -> AddressKeyed<'_, Self>
    where
        Self: Sized,
    {
        AddressKeyed { memory: self }
    }
}
impl<const ROUTED: bool, const ORDER: u8, M: LegacyMemory<ROUTED, ORDER>>
    LegacyMemory<ROUTED, ORDER> for AddressKeyed<'_, M>
{
    const PAGED: bool = M::PAGED;
    fn fast_operand(&self) -> Option<&paging::FastOperand> {
        let operand = self.memory.fast_operand();
        debug_assert!(
            operand.is_none_or(paging::FastOperand::is_strict),
            "a task-stage access under a scalar operand window"
        );
        operand
    }
    fn fast_miss(&self) {
        self.memory.fast_miss()
    }
    fn raw(&self) -> &Access<ROUTED, ORDER> {
        self.memory.raw()
    }
    fn tlb(&self) -> Option<&std::cell::RefCell<paging::Tlb>> {
        self.memory.tlb()
    }
    fn transaction(&self) -> Option<&std::cell::RefCell<paging::Transaction>> {
        self.memory.transaction()
    }
    fn supervisor(&self) -> bool {
        self.memory.supervisor()
    }
    fn read_key(&self, addr: u64) -> u64 {
        addr
    }
    fn next_read_key(&self, addr: u64, _ahead: u16) -> u64 {
        addr
    }
    fn skip_read(&self, cpu: &mut CpuState, linear: u64, bytes: u8) -> Result<(), Exit> {
        if Self::PAGED {
            paging::read(cpu, self, linear, bytes, AccessKind::Read).map(|_| ())
        } else {
            Ok(())
        }
    }
    fn physical_mask(&self) -> u32 {
        self.memory.physical_mask()
    }
    fn note_write(&self, addr: u64, size: u8) {
        if Self::PAGED {
            self.memory.note_write(addr, size);
        }
    }
}
impl<const ROUTED: bool, const ORDER: u8, M: LegacyMemory<ROUTED, ORDER>>
    LegacyMemory<ROUTED, ORDER> for Supervisor<'_, M>
{
    const PAGED: bool = M::PAGED;
    // Windows carry their privilege; a supervisor access finds only a
    // supervisor-proved window (`paging::FastOperand`). Only strict lanes
    // make system accesses: `fast_instruction` admits none.
    fn fast_operand(&self) -> Option<&paging::FastOperand> {
        let operand = self.0.fast_operand();
        debug_assert!(
            operand.is_none_or(paging::FastOperand::is_strict),
            "a system access under a scalar operand window"
        );
        operand
    }
    fn fast_miss(&self) {
        self.0.fast_miss()
    }
    fn raw(&self) -> &Access<ROUTED, ORDER> {
        self.0.raw()
    }
    fn tlb(&self) -> Option<&std::cell::RefCell<paging::Tlb>> {
        self.0.tlb()
    }
    fn transaction(&self) -> Option<&std::cell::RefCell<paging::Transaction>> {
        self.0.transaction()
    }
    fn supervisor(&self) -> bool {
        true
    }
    fn read_key(&self, addr: u64) -> u64 {
        self.0.read_key(addr)
    }
    fn next_read_key(&self, addr: u64, ahead: u16) -> u64 {
        self.0.next_read_key(addr, ahead)
    }
    fn skip_read(&self, cpu: &mut CpuState, linear: u64, bytes: u8) -> Result<(), Exit> {
        if Self::PAGED {
            paging::read(cpu, self, linear, bytes, AccessKind::Read).map(|_| ())
        } else {
            self.0.skip_read(cpu, linear, bytes)
        }
    }
    fn physical_mask(&self) -> u32 {
        self.0.physical_mask()
    }
    fn note_write(&self, addr: u64, size: u8) {
        self.0.note_write(addr, size);
    }
}
impl<const ROUTED: bool, const ORDER: u8, const PAGED: bool> LegacyMemory<ROUTED, ORDER>
    for LegacyAccess<'_, ROUTED, ORDER, PAGED>
where
    PagingMode<PAGED>: PagingStorage,
{
    const PAGED: bool = PAGED;
    fn fast_operand(&self) -> Option<&paging::FastOperand> {
        PagingMode::<PAGED>::fast_operand(&self.paging)
    }
    fn fast_miss(&self) {
        PagingMode::<PAGED>::fast_miss(&self.paging)
    }
    fn tlb(&self) -> Option<&std::cell::RefCell<paging::Tlb>> {
        PagingMode::<PAGED>::tlb(&self.paging)
    }
    fn transaction(&self) -> Option<&std::cell::RefCell<paging::Transaction>> {
        PagingMode::<PAGED>::transaction(&self.paging)
    }

    fn raw(&self) -> &Access<ROUTED, ORDER> {
        self.access
    }
    fn physical_mask(&self) -> u32 {
        self.physical_mask
    }
    fn read_key(&self, addr: u64) -> u64 {
        let index = self.read_index.get();
        self.read_index.set(index + 1);
        addr | (u64::from(index) << 32)
    }
    fn next_read_key(&self, addr: u64, ahead: u16) -> u64 {
        addr | (u64::from(self.read_index.get() + ahead) << 32)
    }
    fn skip_read(&self, cpu: &mut CpuState, linear: u64, bytes: u8) -> Result<(), Exit> {
        LegacyAccess::skip_read(self, cpu, linear, bytes)
    }
    fn note_write(&self, addr: u64, size: u8) {
        if let Some(observe) = self.writes {
            observe(addr, size);
        }
    }
    fn frozen_alias(&self) -> bool {
        self.frozen_alias
    }
}

fn read<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    addr: u64,
    size: u8,
) -> Result<u64, Exit> {
    if mem.paged() {
        return paging::read(cpu, mem, addr, size, AccessKind::Read);
    }
    let (physical, part) = legacy::physical_span(cpu, mem.physical_mask(), addr, size);
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
    physical_mask: u32,
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
        | (u64::from(writable) << 26)
        // A bus narrower than 32 bits (the 80286's), for mmio_evidence.
        | (u64::from(physical_mask != u32::MAX) << 27);
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

#[allow(clippy::too_many_arguments)]
fn record_enter_display(
    cpu: &mut CpuState,
    physical_mask: u32,
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
                    legacy::physical_span(
                        cpu,
                        physical_mask,
                        address.wrapping_add(u64::from(byte)),
                        1,
                    )
                    .0 == u64::from(entry.0 as u32)
                })
                .expect("current attempt's display answer belongs to its bus span");
            let bytes = if partial {
                1
            } else {
                legacy::physical_span(
                    cpu,
                    physical_mask,
                    address.wrapping_add(u64::from(byte)),
                    size - byte,
                )
                .1
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
        if key & DESCRIPTOR_RETRY != 0 || crate::state::is_bound_key(key) {
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
                    legacy::physical_mask(if config & (1 << 27) != 0 {
                        crate::CpuGeneration::I286
                    } else {
                        crate::CpuGeneration::I386
                    }),
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
    if mem.paged() {
        return paging::write(cpu, mem, addr, size, value);
    }
    let value = value & mask(size);
    let (physical, part) = legacy::physical_span(cpu, mem.physical_mask(), addr, size);
    if part == size {
        return write_physical(cpu, mem, addr, physical, size, value);
    }
    // Validate both destinations before committing the split RAM store.
    probe(cpu, mem, addr, size)?;
    let next = u64::from((addr as u32).wrapping_add(u32::from(part)));
    if split_device_follows_ram(cpu, mem, physical, part, next, size - part) {
        // The device piece goes first, so a device stop leaves the RAM piece
        // uncommitted (spec.md §16, pitfall 0) and the retry of a
        // read-modify-write reads the operand it read the first time. This
        // order is an engine choice for restartability; no pinned manual gives
        // the order of the two bus cycles.
        write(cpu, mem, next, size - part, value >> (part * 8))?;
        return write_physical(cpu, mem, addr, physical, part, value);
    }
    write_physical(cpu, mem, addr, physical, part, value)?;
    write(cpu, mem, next, size - part, value >> (part * 8))
}

#[cold]
#[inline(never)]
fn split_device_follows_ram<const ROUTED: bool, const ORDER: u8>(
    cpu: &CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    physical: u64,
    part: u8,
    next: u64,
    rest: u8,
) -> bool {
    let (next, rest) = legacy::physical_span(cpu, mem.physical_mask(), next, rest);
    let device = |at, bytes| {
        matches!(
            mem.raw().region_kind(at, bytes),
            Ok(crate::RegionKind::Trap)
        )
    };
    !device(physical, part) && device(next, rest)
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
    if mem.paged() {
        return paging::probe(cpu, mem, addr, size);
    }
    let (physical, part) = legacy::physical_span(cpu, mem.physical_mask(), addr, size);
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
    let (physical, part) = legacy::physical_span(cpu, mem.physical_mask(), addr, size);
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

fn descriptor_address(cpu: &CpuState, value: u16, generation: crate::CpuGeneration) -> Option<u32> {
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
        // On the 80286 the LDT cache's valid bit alone says whether there
        // is an LDT. LLDT with a null selector marks the register invalid
        // (286 PRM App. B LLDT p. B-66, scan checked), as a task switch to a
        // null LDT does here, and LOADALL loads the cache with no check
        // against the selector ("Undocumented iAPX 286 Test Instruction",
        // pp. 2-3). DERIVED. The 80386 path keeps the null-selector test.
        let ldtr = cpu.legacy.ldtr;
        if ldtr.cache.unusable != 0
            || (generation != crate::CpuGeneration::I286 && ldtr.selector & !3 == 0)
        {
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
    let address =
        descriptor_address(cpu, value, generation).ok_or_else(|| fault(cpu, 13, value & !3))?;
    descriptor_at(cpu, mem, address, generation)
}

fn descriptor_at<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    address: u32,
    generation: crate::CpuGeneration,
) -> Result<Descriptor, Exit> {
    let mem = &mem.system();
    let key = DESCRIPTOR_RETRY | u64::from(address);
    if let (Some(base_limit), Some(attributes)) = (
        cpu.mmio_part(cpu.rip, key),
        cpu.mmio_part(cpu.rip, key | (1 << 32)),
    ) {
        mem.skip_read(cpu, u64::from(address), 4)?;
        mem.skip_read(cpu, u64::from(address.wrapping_add(4)), 4)?;
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
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    address: u32,
    expected: u64,
    value: u64,
) -> Result<u64, Exit> {
    let mem = &mem.system();
    if mem.paged() {
        return paging::descriptor_cas(cpu, mem, address, expected, value);
    }
    let linear = u64::from(address);
    let (physical, _) = legacy::physical_span(cpu, mem.physical_mask(), linear, 1);
    let mem = mem.raw();
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
            let observed = descriptor_cas(cpu, mem, address, expected, value)?;
            if observed != expected {
                descriptor.cache.attributes =
                    (descriptor.cache.attributes & !0xff) | observed as u16;
                return Ok(false);
            }
        }
    }
    remember(cpu, descriptor);
    descriptor.cache.attributes |= bit;
    Ok(true)
}

/// Keep a descriptor as it was read for the replays of this instruction, so
/// that a frame write over its table cannot change it (`descriptor_at`).
fn remember(cpu: &mut CpuState, descriptor: &Descriptor) {
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
    let Some(address) = descriptor_address(cpu, selector, generation) else {
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
                // 80386 PRM §17 p. 17-89; Intel486 PRM §26 p. 26-188:
                // 00FxFF00H includes reserved bit 21. SDM Vol 2A §3.3
                // LAR p. 3-533: undefined bits 19:16 use zero as
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
        reload_real(cpu, segment, value, generation);
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
    let (d, gate) = far_descriptor(cpu, mem, value, returning, generation)?;
    debug_assert!(!gate, "a return never names a call gate");
    Ok(d)
}

/// The descriptor a far CALL, JMP, RET or IRET names, after the code-segment
/// checks; or, for CALL and JMP only, a call gate (`true`), which the caller
/// hands to `privilege::call_gate` unchecked. Type 4 is a call gate on both
/// generations and type 12 on the 80386 only: on the 80286 it takes the
/// "ELSE #GP (code segment selector)" branch of 286 PRM App. B CALL p. B-25
/// and JMP p. B-58 (DERIVED, as `tasks::transfer` does for types 9 and 11).
fn far_descriptor<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    value: u16,
    returning: bool,
    generation: crate::CpuGeneration,
) -> Result<(Descriptor, bool), Exit> {
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
    // A return to an outer ring takes `privilege::outward` before this.
    debug_assert!(!returning || rpl == cpl, "outward return");
    let d = descriptor(cpu, mem, value, generation)?;
    let attr = d.cache.attributes;
    if attr & 0x10 == 0 {
        // Task gates and TSSs were dispatched by `tasks::transfer` before
        // this; every other system type is #GP(selector).
        if !returning
            && (attr & 0xf == 4 || (generation != crate::CpuGeneration::I286 && attr & 0xf == 12))
        {
            return Ok((d, true));
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
    Ok((d, false))
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
    push_slots(cpu, mem, context, count, size, true)
}

/// `whole_frame` checks the frame as one span before its slots; without it,
/// only a slot that itself leaves the segment faults and SP wraps inside it.
fn push_slots<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    context: MachineContext,
    count: usize,
    size: u8,
    whole_frame: bool,
) -> Result<Vec<u64>, Exit> {
    let old = sp(cpu, context);
    if whole_frame {
        let bytes = count as u8 * size;
        let start = old.wrapping_sub(u64::from(bytes)) & mask(context.stack_address_bits / 8);
        segment_access(cpu, context, SegmentId::Ss, start, bytes, AccessKind::Write)?;
    }
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
    push_values(cpu, mem, context, values, size, true)
}

/// Whether a multi-slot stack frame wraps SP inside the stack segment, so
/// that only a slot which itself leaves the segment faults. True in
/// real-address mode on the I80286 and I80386 Machine models; protected mode
/// and the frozen Real16 alias (decision-23 §6) check the whole frame first.
/// The model is not tested here: only the 286/386 legacy interpreter calls
/// it, and the twenty-bit models take their own path with no stack limit.
///
/// The PRMs fault only on the word at offset FFFFh: INT/INTO shut down only
/// at SP = 1, 3 or 5 (I286 PRM App. B p. B-50; 80386 PRM §17 p. 17-73), PUSHA
/// likewise (p. B-88; p. 17-134); IRET, POPA and CALL raise interrupt 13
/// "for a word operand at offset 0FFFFH" or when "the stack is popped when it
/// has offset 0FFFFH" (pp. B-26, B-53, B-85); the 80386 pages fault "if any
/// part of the operand" lies beyond 0FFFFH (pp. 17-42, 17-79, 17-129,
/// 17-145) and Table 14-2 p. 14-8 names "push with SP=1 during PUSH, CALL, or
/// INT" (§14.7 item 7 p. 14-6). Decision-24 records the SDM's whole-frame
/// wording as the counter-statement.
fn wraps_in_segment<const ROUTED: bool, const ORDER: u8>(
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    context: MachineContext,
) -> bool {
    context.regime == legacy::MachineRegime::Real && !mem.frozen_alias()
}

/// An interrupt frame, a far CALL frame or PUSHA/PUSHAD (`wraps_in_segment`).
/// A wrapped frame word at FFFFh faults; that fault is delivered on the same
/// SP, so at SP = 1, 3 or 5 it crosses again and escalates to shutdown.
fn push_frame<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    context: MachineContext,
    values: &[u64],
    size: u8,
) -> Result<(), Exit> {
    let whole_frame = !wraps_in_segment(mem, context);
    push_values(cpu, mem, context, values, size, whole_frame)
}

/// Check every slot of a `count`-slot pop frame before its first read: the
/// whole span, or each slot alone when the frame wraps (`wraps_in_segment`).
fn pop_frame_limit<const ROUTED: bool, const ORDER: u8>(
    cpu: &CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    context: MachineContext,
    count: u8,
    size: u8,
) -> Result<(), Exit> {
    let base = sp(cpu, context);
    if !wraps_in_segment(mem, context) {
        segment_access(
            cpu,
            context,
            SegmentId::Ss,
            base,
            count * size,
            AccessKind::Read,
        )?;
        return Ok(());
    }
    for i in 0..u64::from(count) {
        let offset = base.wrapping_add(i * u64::from(size)) & mask(context.stack_address_bits / 8);
        segment_access(cpu, context, SegmentId::Ss, offset, size, AccessKind::Read)?;
    }
    Ok(())
}

fn push_values<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    context: MachineContext,
    values: &[u64],
    size: u8,
    whole_frame: bool,
) -> Result<(), Exit> {
    let old = sp(cpu, context);
    let addresses = push_slots(cpu, mem, context, values.len(), size, whole_frame)?;
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
    let mem =
        &LegacyAccess::<ROUTED, ORDER>::new(memory, legacy::physical_mask(context.generation));
    task_switch.committed = false;
    task_switch.completed = false;
    if task_switch.context().is_some() {
        return tasks::resume(cpu, &mem.address_keyed(), task_switch);
    }
    deliver(cpu, mem, context, event, task_switch)
}

fn deliver_protected<const ROUTED: bool, const ORDER: u8, const V86: bool>(
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
        mem.system().skip_read(cpu, u64::from(address), 4)?;
        value as u32
    } else {
        read(cpu, &mem.system(), u64::from(address), 4)? as u32
    };
    let high = if let Some(value) = cpu.mmio_part(cpu.rip, key | (1 << 32)) {
        mem.system()
            .skip_read(cpu, u64::from(address.wrapping_add(4)), 4)?;
        value as u32
    } else {
        read(cpu, &mem.system(), u64::from(address.wrapping_add(4)), 4)? as u32
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
    if V86 && matches!(kind, 6 | 7) {
        // SDM Vol 3B §23.3.1 p. 23-12; design v86 §9 Q2.
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyGateOrTaskTransfer,
        ));
    }
    if kind == 5 {
        return tasks::event(
            cpu,
            &mem.address_keyed(),
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
    if d.cache.attributes & 0x18 != 0x18
        || dpl > context.cpl
        || (V86 && (dpl != 0 || d.cache.attributes & 4 != 0))
    {
        return Err(fault(cpu, 13, selector_code));
    }
    if d.cache.attributes & 0x80 == 0 {
        return Err(fault(cpu, 11, selector_code));
    }
    let size = if kind & 8 != 0 { 4 } else { 2 };
    let ip = u64::from((low & 0xffff) | if size == 4 { high & 0xffff0000 } else { 0 });
    if V86 {
        return privilege::virtual8086_inward(
            cpu,
            mem,
            context,
            event,
            cs,
            d,
            ip,
            kind & 1 == 0,
            privilege::Gate { key, low, high },
        );
    }
    if d.cache.attributes & 4 == 0 && dpl < context.cpl {
        return privilege::inward(
            cpu,
            mem,
            context,
            event,
            cs,
            d,
            ip,
            size,
            kind & 1 == 0,
            privilege::Gate { key, low, high },
        );
    }
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
    if task_switch.record_regs {
        task_switch.capture_regs(cpu, context.generation);
    }
    if context.regime == legacy::MachineRegime::Virtual8086 {
        return deliver_protected::<ROUTED, ORDER, true>(cpu, mem, context, event, task_switch);
    }
    if context.regime == legacy::MachineRegime::Protected {
        return deliver_protected::<ROUTED, ORDER, false>(cpu, mem, context, event, task_switch);
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
    far_real_target(cpu, context, ip, 4)?;
    let flags = legacy::flags_image(cpu, context.generation);
    // Decision-24's source rule: the frame wraps SP as both PRMs' INT pages
    // say (`wraps_in_segment`), not the whole-frame room check in SDM Vol 2A
    // §3.3 INT, REAL-ADDRESS-MODE p. 3-470.
    push_frame(
        cpu,
        mem,
        context,
        &[u64::from(flags), u64::from(cpu.cs), u64::from(saved_ip)],
        2,
    )?;
    cpu.flags.if_ = false;
    // SDM Vol 3B §23.1.4: clear TF/RF/AC on entry.
    cpu.rflags_system &= !0x50100;
    reload_real(cpu, SegmentId::Cs, cs as u16, context.generation);
    cpu.rip = ip;
    Ok(())
}

// x87 retains its existing RAM-only transfer contract. Segment checks happen in
// Legacy::Address; this adapter splits each actual RAM transfer at the A20 gate.
// A transfer over adjacent RAM/ROM regions is performed (the `*_spanning` paths).
struct PhysicalFpMemory<'a, const ROUTED: bool> {
    memory: &'a Memory,
    a20_enabled: bool,
    physical_mask: u32,
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
            let (physical, part) = legacy::physical_span_with_gate(
                self.a20_enabled,
                self.physical_mask,
                addr,
                buf.len().min(255) as u8,
            );
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
                self.physical_mask,
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
            let (physical, part) = legacy::physical_span_with_gate(
                self.a20_enabled,
                self.physical_mask,
                addr,
                bytes.len().min(255) as u8,
            );
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
    physical_mask: u32,
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
        &LegacyAccess::<ROUTED, ORDER>::new(memory, physical_mask),
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
    /// Every op is `supported`, decided once when the block was built: a
    /// `flat32_ordinary` block (the Direct and paged ordinary walks) or a
    /// Direct-class lift, which turns any other op into `Refuse`. The
    /// instruction then skips the run-time scan. False wherever unproved.
    proven: bool,
}

// A locked operand partly in RAM and partly in one Trap region, A20-split or
// not (xj-ir6.34): the read of its pieces in operand byte order, then the
// device piece's write and the RAM pieces' commit (`integer::locked_split_update`,
// which cites the order and the retry rule). The device piece is one transfer,
// as an A20 bus piece is, keyed by its visit (`read_physical`) and its write
// acknowledgement (`write_physical`). An operand over two Trap regions keeps
// embedder emulation; a ROM piece is read and its write dropped, and a byte in
// no region faults before any transfer (`integer::locked_pieces`).
#[cold]
#[inline(never)]
fn locked_split<const ROUTED: bool, const ORDER: u8, const PAGED: bool>(
    cpu: &mut CpuState,
    mem: &LegacyAccess<'_, ROUTED, ORDER, PAGED>,
    ir: &InstructionView<'_>,
    linear: u64,
    src: u64,
    size: u8,
    update: super::integer::LockedUpdate,
) -> Result<u64, Exit>
where
    PagingMode<PAGED>: PagingStorage,
{
    use super::integer::{locked_piece_exit, locked_pieces, locked_split_update, LockedPieces};
    let at = |byte: u8| u64::from((linear as u32).wrapping_add(u32::from(byte)));
    let mut bus = [0; 8];
    for (byte, bus) in bus.iter_mut().enumerate().take(usize::from(size)) {
        *bus = legacy::physical_span(cpu, mem.physical_mask, at(byte as u8), 1).0;
    }
    let fault = |cpu: &mut CpuState, trap, piece: super::integer::LockedPiece, write, value| {
        let access = if write {
            AccessKind::Write
        } else {
            AccessKind::Read
        };
        let exit = trap_out(cpu, cpu.rip, trap, piece.addr, piece.len, access, value);
        let StepResult::Exit(exit) = exit.into_step() else {
            unreachable!("a memory trap exits")
        };
        crate::fault::qualify_legacy_bus(exit, at(piece.first), piece.addr)
    };
    let (pieces, count) = match locked_pieces::<ROUTED>(&mem.0, size, |byte| bus[usize::from(byte)])
    {
        LockedPieces::Split(pieces, count) => (pieces, count),
        LockedPieces::Refused => return Err(atomic_device_exit(cpu, mem, ir)),
        LockedPieces::Miss(trap, piece) => {
            let address = store_fault_address(mem, trap, piece.addr, piece.len);
            let piece = super::integer::LockedPiece {
                addr: address,
                ..piece
            };
            return Err(fault(cpu, trap, piece, true, 0));
        }
    };
    // An exchange whose old value the instruction discards (the locked MOV) is
    // a store: it reads nothing.
    let read = !(matches!(
        update,
        super::integer::LockedUpdate::Rmw(crate::ir::RmwOp::Xchg)
    ) && super::integer::exchange_discards_old(ir.ops, cpu.rip));
    let owner = cpu.rip;
    locked_split_update(
        cpu,
        owner,
        &pieces[..count],
        read,
        |old| update.apply(old, src, size),
        |cpu, piece| {
            read_physical(cpu, mem, at(piece.first), piece.addr, piece.len)
                .map_err(|exit| locked_piece_exit(exit, size, piece))
        },
        |cpu, piece| {
            mem.raw()
                .read(piece.addr, piece.len)
                .map_err(|trap| fault(cpu, trap, piece, false, 0))
        },
        |cpu, piece, value| {
            write_physical(cpu, mem, at(piece.first), piece.addr, piece.len, value)
                .map_err(|exit| locked_piece_exit(exit, size, piece))
        },
        |cpu, piece, value| write_physical(cpu, mem, at(piece.first), piece.addr, piece.len, value),
    )
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
        let (physical, _) = legacy::physical_span(cpu, mem.physical_mask(), linear, 1);
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
            | IrOp::AtomicCas { .. }
            | IrOp::Bswap { .. }
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
    execute_instruction::<ROUTED, ORDER, false, false>(
        ir,
        &InstructionView {
            guest_start: ir.guest_start,
            guest_len: ir.guest_len,
            ops: &ir.ops,
            proven: false,
        },
        cpu,
        &LegacyAccess::new(memory, legacy::physical_mask(context.generation)),
        scratch.zeroed(ir.temp_count as usize),
        context,
        task_switch,
    )
}

fn execute_instruction<const ROUTED: bool, const ORDER: u8, const PAGED: bool, const V86: bool>(
    block: &IrBlock,
    ir: &InstructionView<'_>,
    cpu: &mut CpuState,
    mem: &LegacyAccess<'_, ROUTED, ORDER, PAGED>,
    temps: &mut [u64],
    context: MachineContext,
    task_switch: &mut tasks::SwitchState,
) -> Result<Outcome, Exit>
where
    PagingMode<PAGED>: PagingStorage,
{
    if PAGED && mem.transaction().is_none() && mem.fast_operand().is_none() {
        return paging::execute::<ROUTED, ORDER, PAGED, V86>(
            block,
            ir,
            cpu,
            mem,
            temps,
            context,
            task_switch,
        )
        .map(|(outcome, _)| outcome);
    }
    mem.read_index.set(0);
    let checked = context.segment_class != crate::SegmentClass::Direct;
    debug_assert!(!checked || !ir.proven || ir.ops.iter().all(supported));
    if checked && !ir.proven && !ir.ops.iter().all(supported) {
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyInstruction,
        ));
    }
    let before = enter_instruction(ir, cpu, task_switch);
    let result = if task_switch.context().is_some() {
        let suppress_debug = task_switch.suppress_debug();
        // A software interrupt whose task switch was handed off delivers here,
        // with the origin its INT stored (spec.md §17.10, Delivery log).
        let software_interrupt = task_switch.software_interrupt_pending();
        tasks::resume(cpu, &mem.address_keyed(), task_switch).map(|()| Outcome {
            retired: true,
            context_changed: true,
            preserve_rf: true,
            suppress_debug,
            software_interrupt,
            ..Outcome::default()
        })
    } else {
        count_walk();
        let mut execution = InstructionExecution::<ROUTED, ORDER, PAGED, V86> {
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
    leave_instruction(result, ir, cpu, mem, temps, context, task_switch, before)
}

/// The registers a `Checkpoint` instruction restores when it fails before its
/// task switch commits.
type Rollback = ([u64; 16], Flags, u64, legacy::LegacyCpuState, [u16; 6]);

/// An instruction's entry: its rollback image and fresh task-switch marks.
#[inline(always)]
fn enter_instruction(
    ir: &InstructionView<'_>,
    cpu: &CpuState,
    task_switch: &mut tasks::SwitchState,
) -> Option<Rollback> {
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
    before
}

/// An instruction's exit: a failed instruction rolls back to its first byte
/// with its evidence, a retired one clears RF, and a page fault publishes CR2.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn leave_instruction<const ROUTED: bool, const ORDER: u8, const PAGED: bool>(
    mut result: Result<Outcome, Exit>,
    ir: &InstructionView<'_>,
    cpu: &mut CpuState,
    mem: &LegacyAccess<'_, ROUTED, ORDER, PAGED>,
    temps: &mut [u64],
    context: MachineContext,
    task_switch: &tasks::SwitchState,
    before: Option<Rollback>,
) -> Result<Outcome, Exit>
where
    PagingMode<PAGED>: PagingStorage,
{
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
    match result {
        Err(exit) if !task_switch.committed => {
            result = Err(roll_back(exit, ir, cpu, mem, temps, context, before));
        }
        Ok(ref outcome) if outcome.retired && !outcome.preserve_rf => {
            // SDM Vol 3B §20.3.1.1; 80386 PRM §12.3.1.1: retirement clears RF.
            cpu.rflags_system &= !(1 << 16);
        }
        _ => {}
    }
    if PAGED {
        if let Err(exit) = &result {
            paging::publish_cr2(cpu, exit);
        }
    }
    result
}

#[cold]
#[inline(never)]
fn roll_back<const ROUTED: bool, const ORDER: u8, const PAGED: bool>(
    exit: Exit,
    ir: &InstructionView<'_>,
    cpu: &mut CpuState,
    mem: &LegacyAccess<'_, ROUTED, ORDER, PAGED>,
    temps: &mut [u64],
    context: MachineContext,
    before: Option<Rollback>,
) -> Exit
where
    PagingMode<PAGED>: PagingStorage,
{
    if let Some(before) = before {
        cpu.gpr = before.0;
        cpu.flags = before.1;
        cpu.rflags_system = before.2;
        cpu.legacy = before.3;
        [cpu.cs, cpu.ds, cpu.es, cpu.ss, cpu.fs, cpu.gs] = before.4;
    }
    cpu.rip = ir.guest_start;
    // Match the shared walk's evidence only where the instruction's
    // memory effects are represented. Opaque frame/system helpers keep
    // their existing evidence; generic scalar analysis cannot prove
    // that those helpers left RAM untouched. The 486 alignment gate
    // (`I486Gate`) only refuses before any effect, so it changes nothing.
    let scalar_evidence = context.regime == legacy::MachineRegime::Real
        && ir.ops.iter().all(|op| match op {
            IrOp::Legacy(op) => matches!(
                op,
                L::SharedWalk
                    | L::Checkpoint
                    | L::I486Gate
                    | L::Address { .. }
                    | L::ReadSelector { .. }
                    | L::DirectStackAddress { .. }
                    | L::FusedPush { .. }
                    | L::FusedPop { .. }
                    | L::FusedPush20 { .. }
                    | L::FusedPop20 { .. }
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
    exit
}

/// A live field's value (`LegacyOp::LiveField`, xj-to9.46): its bytes
/// now, in one copy from the bus address the fetch translated, extended as
/// the decoder extends them. The proof holding the op lives no longer than
/// that RAM or ROM mapping (`Memory::mapping_identity`), so the copy cannot
/// come up short; if it did, it would report an execute access to the bus
/// address (`LegacyOp::LiveField`).
#[inline]
pub(super) fn live_field(memory: &Memory, bus: u64, bytes: u8, extend: u8) -> Result<u64, Exit> {
    let mut raw = [0; 8];
    match memory.fetch_code(bus, &mut raw[..usize::from(bytes)]) {
        Ok(copied) if copied == usize::from(bytes) => Ok(legacy::live_field_value(
            u64::from_le_bytes(raw),
            bytes,
            extend,
        )),
        _ => Err(Exit::unmapped_memory(bus, AccessKind::Execute)),
    }
}

pub(crate) fn flat32_ordinary(ir: &IrBlock) -> bool {
    ir.ops.iter().all(|op| match op {
        IrOp::Legacy(op) => {
            matches!(
                op,
                L::Address { .. }
                    | L::LiveField { .. }
                    | L::Flat32Address { .. }
                    | L::SegmentedAddress { .. }
                    | L::I486Gate
                    | L::Checkpoint
                    | L::ReadSelector { .. }
                    | L::Transfer { cs: None, .. }
                    | L::Return { far: false, .. }
                    | L::Loop { .. }
            ) || data_segment_load(op).is_some()
        }
        _ => supported(op),
    })
}

/// A DS, ES, FS or GS load (MOV, POP, LDS, LES, LFS, LGS). It performs the
/// complete architectural load with its own faults and may continue an
/// ordinary block: no event shadow (SDM Vol 3A §7.8.3 p. 7-8 names only SS), and
/// neither the stack width nor CS changes. SS loads keep ending the block.
#[inline]
pub(crate) fn data_segment_load(op: &L) -> Option<SegmentId> {
    match op {
        L::LoadSegment {
            segment: segment @ (SegmentId::Ds | SegmentId::Es | SegmentId::Fs | SegmentId::Gs),
            inhibit: false,
            ..
        } => Some(*segment),
        _ => None,
    }
}

/// Whether an ordinary block continues after an instruction whose outcome
/// changed the context. In an ordinary walk only a data-segment load does
/// that. Segmented and V86 stay admissible after every successful load (later
/// operands of the reloaded segment were lifted as live checks); Flat32
/// continues only if a reloaded DS or ES cache still proves Flat32.
#[cold]
#[inline(never)]
pub(crate) fn reload_continues(ops: &[IrOp], context: MachineContext, cpu: &CpuState) -> bool {
    ops.iter().all(|op| match op {
        IrOp::Legacy(op @ L::LoadSegment { .. }) => data_segment_load(op).is_some_and(|segment| {
            context.segment_class != crate::SegmentClass::Flat32
                || matches!(segment, SegmentId::Fs | SegmentId::Gs)
                || legacy::flat32_cache(
                    &cpu.legacy.segments[segment as usize],
                    segment as usize,
                    cpu.legacy.cpl,
                )
        }),
        _ => true,
    }) && ops
        .iter()
        .any(|op| matches!(op, IrOp::Legacy(op) if data_segment_load(op).is_some()))
}

#[derive(Clone, Copy)]
pub(crate) enum BlockWalk {
    Shared,
    Direct,
    Checked,
    Protected,
    /// A paged Flat32, Segmented or V86 block whose ordinary walk was proved at
    /// lift (`paging::execute_flat32`).
    Paged,
}
impl BlockWalk {
    #[cold]
    #[inline(never)]
    pub(crate) fn for_flat32(ir: &IrBlock) -> Self {
        if flat32_ordinary(ir) {
            Self::Direct
        } else {
            Self::Protected
        }
    }

    pub(crate) fn for_ir<const SYSTEM: bool>(ir: &IrBlock) -> Self {
        if SYSTEM
            && ir
                .legacy_context
                .is_some_and(|c| c.regime != legacy::MachineRegime::Real)
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
    pub ss_shadow: u8,
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
/// instruction bodies use the shared opcode walk (spec.md §8.1), in one pass
/// over the block ([`BlockPass`]).
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
    let mask = legacy::physical_mask(context.generation);
    let mut pass = BlockPass::<ROUTED, ORDER, false, false>::new(
        ir,
        context,
        memory,
        LegacyAccess::new(memory, mask),
        None,
        task_switch,
        exit_word,
        limit,
    );
    let outcome = pass.run(cpu, temps, &mut None);
    Progress {
        sti_shadow: false,
        ss_shadow: 0,
        steps: pass.steps,
        retired: pass.retired,
        outcome: Some(outcome),
    }
}

#[cfg(feature = "test-hooks")]
thread_local! {
    /// Op walks the Machine instruction walks started, for the dispatch
    /// counters (`Vcpu::take_machine_counters_for_test`).
    pub(crate) static WALKS: Cell<u64> = const { Cell::new(0) };
}

#[inline(always)]
fn count_walk() {
    #[cfg(feature = "test-hooks")]
    WALKS.with(|walks| walks.set(walks.get() + 1));
}

/// How one instruction of an ordinary block executes. The unpaged walk has
/// only `Registers`; the paged walk chooses per instruction
/// (`paging::ordinary_lane`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Lane {
    /// The unpaged adapter: no paged operand.
    Registers,
    /// Paged operands preflighted from the TLB (`paging::FastOperand`).
    Fast,
    /// A system lane over proved windows that replays on a miss.
    Strict,
    /// The two-phase operand transaction.
    Transaction,
}

/// Where a [`BlockPass`] continues after an instruction commits.
#[derive(Clone, Copy)]
enum Flow {
    /// The block continues at this op index: the next instruction, or the
    /// same one after an incomplete REP iteration.
    Next(usize),
    /// The block ends with this instruction's outcome (`BlockPass::outcome`).
    Stop,
    /// The prepared instruction runs on its own, outside the walk: a pending
    /// task continuation, or a strict or transaction lane.
    Yield,
}

/// One pass of the shared opcode walk over a Direct-walk block (a Direct-class
/// or `flat32_ordinary` block) or a paged ordinary block. Every instruction
/// keeps its own commit point: it starts at its `InsnStart` (rollback image,
/// read cursor, task-switch marks, outcome) and commits at the next
/// `InsnStart` or where the walk returns, with the same rollback, retirement,
/// budget, exit-word and dirty-code tests at each boundary as an instruction
/// walked by itself (`execute_instruction`). An instruction that fails leaves
/// state as its own walk would; one that cannot run in the walk yields to that
/// walk. `LANES` selects the paged walk (`paging::execute_ordinary`).
pub(super) struct BlockPass<
    'a,
    const ROUTED: bool,
    const ORDER: u8,
    const LANES: bool,
    const V86: bool,
> {
    ir: &'a IrBlock,
    context: MachineContext,
    memory: &'a Memory,
    registers: LegacyAccess<'a, ROUTED, ORDER>,
    /// The paged lanes' adapter; the unpaged walk has none.
    paged: Option<LegacyAccess<'a, ROUTED, ORDER, true>>,
    task_switch: &'a mut tasks::SwitchState,
    exit_word: &'a std::sync::atomic::AtomicU64,
    limit: u64,
    /// The paged walk's TLB and its generation at block entry.
    tlb: Option<(&'a std::cell::RefCell<paging::Tlb>, u64)>,
    /// No transaction is retained; read where each walk segment starts, as
    /// only a yielded instruction changes it.
    retry_none: bool,
    pub(super) steps: u64,
    pub(super) retired: u64,
    reloaded: bool,
    /// The op index where the current walk segment started.
    base: usize,
    view: InstructionView<'a>,
    start: usize,
    end: usize,
    lane: Lane,
    fast: Option<paging::FastOperand>,
    before: Option<Rollback>,
    outcome: Outcome,
    in_flight: bool,
    flow: Option<Flow>,
}

impl<'a, const ROUTED: bool, const ORDER: u8, const LANES: bool, const V86: bool>
    BlockPass<'a, ROUTED, ORDER, LANES, V86>
{
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn new(
        ir: &'a IrBlock,
        context: MachineContext,
        memory: &'a Memory,
        registers: LegacyAccess<'a, ROUTED, ORDER>,
        paged: Option<LegacyAccess<'a, ROUTED, ORDER, true>>,
        task_switch: &'a mut tasks::SwitchState,
        exit_word: &'a std::sync::atomic::AtomicU64,
        limit: u64,
    ) -> Self {
        let tlb = paged
            .as_ref()
            .and_then(|paged| paged.paging.tlb)
            .map(|tlb| (tlb, tlb.borrow().generation));
        Self {
            ir,
            context,
            memory,
            registers,
            paged,
            task_switch,
            exit_word,
            limit,
            tlb,
            retry_none: true,
            steps: 0,
            retired: 0,
            reloaded: false,
            base: 0,
            view: InstructionView {
                guest_start: ir.guest_start,
                guest_len: 0,
                ops: &[],
                proven: true,
            },
            start: 0,
            end: 0,
            lane: Lane::Registers,
            fast: None,
            before: None,
            outcome: Outcome::default(),
            in_flight: false,
            flow: None,
        }
    }

    /// The block from its first instruction to the outcome that ends it.
    #[inline(always)]
    pub(super) fn run(
        &mut self,
        cpu: &mut CpuState,
        temps: &mut [u64],
        retry: &mut Option<paging::Transaction>,
    ) -> Result<Outcome, Exit> {
        let ir = self.ir;
        let access = self.registers.access;
        let mut at = 0;
        loop {
            let IrOp::InsnStart { guest_addr } = ir.ops[at] else {
                unreachable!("instruction marker")
            };
            self.base = at;
            self.retry_none = retry.is_none();
            count_walk();
            let walked = super::walk_instruction::<ROUTED, ORDER>(
                ir,
                &ir.ops[at..],
                guest_addr,
                cpu,
                access,
                temps,
                &mut *self,
            );
            let flow = match self.flow.take() {
                Some(Flow::Next(next)) => {
                    at = next;
                    continue;
                }
                Some(Flow::Stop) => return Ok(std::mem::take(&mut self.outcome)),
                Some(Flow::Yield) => {
                    self.outcome = self.yielded(cpu, temps, retry)?;
                    self.commit(cpu)
                }
                // The instruction returned from the walk or ran off the block.
                None => {
                    if let Err(exit) = walked {
                        self.outcome = self.fail(exit, cpu, temps)?;
                    } else {
                        self.locate();
                        self.complete(cpu);
                    }
                    self.commit(cpu)
                }
            };
            match flow {
                Flow::Next(next) => at = next,
                Flow::Stop => return Ok(std::mem::take(&mut self.outcome)),
                Flow::Yield => unreachable!("commit never yields"),
            }
        }
    }

    /// Enter the instruction at `at`. The paged walk locates it and chooses
    /// its lane at once; the unpaged walk locates it only when an op of its
    /// own, a failure or the walk's end needs its bounds (`locate`), and
    /// otherwise learns its end at the next instruction's marker.
    #[inline(always)]
    fn prepare(&mut self, at: usize, guest_addr: u64, cpu: &mut CpuState, temps: &mut [u64]) {
        self.start = at;
        self.end = 0;
        self.view = InstructionView {
            guest_start: guest_addr,
            guest_len: 0,
            ops: &[],
            // `BlockWalk::Direct`: a Direct-class or `flat32_ordinary` block;
            // the paged ordinary block's `ordinary_block`, decided at lift,
            // implies `flat32_ordinary`.
            proven: true,
        };
        if LANES {
            self.locate();
            let (tlb, _) = self.tlb.expect("paged walk");
            (self.lane, self.fast) = paging::ordinary_lane(
                &self.view,
                cpu,
                self.memory,
                tlb,
                temps,
                self.context,
                self.retry_none,
            );
        }
    }

    /// The current instruction's bounds: its ops end at the next `InsnStart`,
    /// whose address is where it ends, or with the block.
    #[inline(always)]
    fn locate(&mut self) {
        if self.end != 0 {
            return;
        }
        let ops = &self.ir.ops;
        let at = self.start;
        let end = ops[at + 1..]
            .iter()
            .position(|op| matches!(op, IrOp::InsnStart { .. }))
            .map_or(ops.len(), |offset| at + 1 + offset);
        let next = match ops.get(end) {
            Some(IrOp::InsnStart { guest_addr }) => *guest_addr,
            _ => self.ir.guest_start + u64::from(self.ir.guest_len),
        };
        self.bound(end, next);
    }

    /// The current instruction ends at op `end` and guest address `next`.
    #[inline(always)]
    fn bound(&mut self, end: usize, next: u64) {
        self.end = end;
        self.view.ops = &self.ir.ops[self.start..end];
        self.view.guest_len = (next - self.view.guest_start) as u32;
        debug_assert!(
            self.context.segment_class == crate::SegmentClass::Direct
                || self.view.ops.iter().all(supported)
        );
    }

    /// `execute_instruction`'s entry for an instruction the walk runs.
    #[inline(always)]
    fn begin(&mut self, cpu: &CpuState) {
        if LANES && self.lane == Lane::Fast {
            let paged = self.paged.as_mut().expect("paged walk");
            paged.paging.fast_operand = self.fast;
            paged.read_index.set(0);
        } else {
            self.registers.read_index.set(0);
        }
        // `enter_instruction`, writing the rollback image in place: `before`
        // is None between instructions. The op after the marker is the
        // instruction's own, or the next instruction's marker.
        debug_assert!(self.before.is_none());
        if matches!(
            self.ir.ops.get(self.start + 1),
            Some(IrOp::Legacy(L::Checkpoint))
        ) {
            self.before = Some((
                cpu.gpr,
                cpu.flags,
                cpu.rflags_system,
                cpu.legacy,
                [cpu.cs, cpu.ds, cpu.es, cpu.ss, cpu.fs, cpu.gs],
            ));
        }
        self.task_switch.committed = false;
        self.task_switch.completed = false;
        self.outcome = Outcome {
            retired: true,
            ..Outcome::default()
        };
        self.in_flight = true;
    }

    /// `execute_instruction`'s exit for an instruction the walk completed:
    /// retirement clears RF (SDM Vol 3B §20.3.1.1; 80386 PRM §12.3.1.1).
    #[inline(always)]
    fn complete(&mut self, cpu: &mut CpuState) {
        debug_assert!(self.in_flight, "an instruction in the walk");
        self.in_flight = false;
        if self.before.is_some() {
            self.before = None;
        }
        if self.outcome.retired && !self.outcome.preserve_rf {
            cpu.rflags_system &= !(1 << 16);
        }
    }

    /// `execute_instruction`'s exit for an instruction that failed in the walk.
    #[cold]
    #[inline(never)]
    fn fail(&mut self, exit: Exit, cpu: &mut CpuState, temps: &mut [u64]) -> Result<Outcome, Exit> {
        debug_assert!(self.in_flight, "an instruction in the walk");
        self.in_flight = false;
        self.locate();
        let before = self.before.take();
        if LANES && self.lane == Lane::Fast {
            leave_instruction(
                Err(exit),
                &self.view,
                cpu,
                self.paged.as_ref().expect("paged walk"),
                temps,
                self.context,
                self.task_switch,
                before,
            )
        } else {
            leave_instruction(
                Err(exit),
                &self.view,
                cpu,
                &self.registers,
                temps,
                self.context,
                self.task_switch,
                before,
            )
        }
    }

    /// The prepared instruction on its own walk, as before the one-pass walk.
    #[cold]
    #[inline(never)]
    fn yielded(
        &mut self,
        cpu: &mut CpuState,
        temps: &mut [u64],
        retry: &mut Option<paging::Transaction>,
    ) -> Result<Outcome, Exit> {
        let (ir, view, context) = (self.ir, &self.view, self.context);
        match self.lane {
            Lane::Registers => execute_instruction::<ROUTED, ORDER, false, V86>(
                ir,
                view,
                cpu,
                &self.registers,
                temps,
                context,
                self.task_switch,
            ),
            Lane::Fast => {
                self.paged.as_mut().expect("paged walk").paging.fast_operand = self.fast;
                execute_instruction::<ROUTED, ORDER, true, V86>(
                    ir,
                    view,
                    cpu,
                    self.paged.as_ref().expect("paged walk"),
                    temps,
                    context,
                    self.task_switch,
                )
            }
            Lane::Strict => {
                self.paged.as_mut().expect("paged walk").paging.fast_operand = self.fast;
                paging::strict_instruction::<ROUTED, ORDER, V86>(
                    ir,
                    view,
                    cpu,
                    self.paged.as_mut().expect("paged walk"),
                    temps,
                    context,
                    self.task_switch,
                    retry,
                )
            }
            Lane::Transaction => {
                self.paged.as_mut().expect("paged walk").paging.fast_operand = None;
                paging::execute_with_retry::<ROUTED, ORDER, true, V86>(
                    ir,
                    view,
                    cpu,
                    self.paged.as_ref().expect("paged walk"),
                    temps,
                    context,
                    self.task_switch,
                    retry,
                )
                .map(|(outcome, input)| {
                    debug_assert!(input.is_none(), "ordinary Flat32 has no port strings");
                    outcome
                })
            }
        }
    }

    /// The boundary after a completed instruction: its retirement, then
    /// whether the block goes on. Never execute stale successors after a code
    /// write; dispatch drains the notification and invalidates physical spans.
    #[inline(always)]
    fn commit(&mut self, cpu: &mut CpuState) -> Flow {
        debug_assert!(self.end != 0, "a located instruction");
        let outcome = &mut self.outcome;
        self.steps += 1;
        self.retired += u64::from(outcome.retired);
        if cpu.mmio_retry_active {
            cpu.finish_mmio_retry(None);
        }
        let last = self.end == self.ir.ops.len();
        // One load for every event of another thread or of this instruction:
        // a host request, an NMI edge, and a code write, whose notice
        // (`DISPATCH_DIRTY`, `DISPATCH_CHILD_DIRTY`) every writer sets in
        // each registered dispatch word, this one included, before it
        // returns (`Memory::note_code_write`, `ChildDispatch::notify`).
        let polled = || self.exit_word.load(std::sync::atomic::Ordering::Relaxed) != 0;
        if LANES {
            let (tlb, token) = self.tlb.expect("paged walk");
            if last
                || self.steps == self.limit
                || outcome.exit.is_some()
                || outcome.translations_invalidated
                || outcome.arm_sti
                || outcome.arm_ss != 0
                || polled()
                || tlb.borrow().generation != token
            {
                outcome.context_changed |= self.reloaded;
                return Flow::Stop;
            }
            // An in-block data-segment load continues the block; the
            // dispatcher reselects once where it ends (spec.md §17.10).
            if outcome.context_changed {
                if !reload_continues(self.view.ops, self.context, cpu) {
                    return Flow::Stop;
                }
                self.reloaded = true;
            }
            return Flow::Next(self.end);
        }
        let stop = (outcome.retired && last)
            || self.steps == self.limit
            || outcome.exit.is_some()
            || outcome.arm_sti
            || outcome.arm_ss != 0
            || polled();
        if outcome.context_changed {
            // An in-block data-segment load leaves the block running when
            // nothing else ends it (`reload_continues`); the dispatcher
            // reselects the context once, where the block ends (spec.md §17.10).
            if stop || !outcome.retired || !reload_continues(self.view.ops, self.context, cpu) {
                return Flow::Stop;
            }
            self.reloaded = true;
            return Flow::Next(self.end);
        }
        if stop {
            // The block reloaded a segment; reselect where it ends.
            outcome.context_changed |= self.reloaded;
            return Flow::Stop;
        }
        // An incomplete REP iteration runs again at the same instruction.
        Flow::Next(if outcome.retired {
            self.end
        } else {
            self.start
        })
    }
}

impl<const ROUTED: bool, const ORDER: u8, const LANES: bool, const V86: bool>
    super::InstructionExecutor<'_> for &mut BlockPass<'_, ROUTED, ORDER, LANES, V86>
{
    fn enabled(&self) -> bool {
        true
    }
    fn next_ip(&self) -> u64 {
        // Unlocated, the instruction ran off the block: it is the last one.
        let end = if self.end == 0 {
            self.ir.guest_start + u64::from(self.ir.guest_len)
        } else {
            self.view
                .guest_start
                .wrapping_add(u64::from(self.view.guest_len))
        };
        end & 0xffff_ffff
    }
    #[inline(always)]
    fn execute(&mut self, op: &IrOp, cpu: &mut CpuState, temps: &mut [u64]) -> Option<WalkResult> {
        self.locate();
        let result = if LANES && self.lane == Lane::Fast {
            execute_op::<ROUTED, ORDER, true, V86>(
                op,
                &self.view,
                cpu,
                self.paged.as_ref().expect("paged walk"),
                temps,
                self.context,
                self.task_switch,
                &mut self.outcome,
            )
        } else {
            execute_op::<ROUTED, ORDER, false, V86>(
                op,
                &self.view,
                cpu,
                &self.registers,
                temps,
                self.context,
                self.task_switch,
                &mut self.outcome,
            )
        };
        match result {
            Ok(false) => None,
            Ok(true) => Some(WalkResult::Continue),
            Err(exit) => Some(WalkResult::exit(exit)),
        }
    }
    fn boundaries(&self) -> bool {
        true
    }
    #[inline(always)]
    fn boundary(
        &mut self,
        op_index: usize,
        guest_addr: u64,
        cpu: &mut CpuState,
        temps: &mut [u64],
    ) -> Option<WalkResult> {
        let at = self.base + op_index;
        let first = !self.in_flight;
        if self.in_flight {
            // The instruction's ops ran out at this marker: it falls through
            // to this instruction, as its own walk would have ended (`walk_ops`).
            if self.end == 0 {
                self.bound(at, guest_addr);
            }
            cpu.rip = self.next_ip();
            self.complete(cpu);
            match self.commit(cpu) {
                Flow::Next(next) if next == at => {}
                flow => {
                    self.flow = Some(flow);
                    return Some(WalkResult::Continue);
                }
            }
        }
        cpu.rip = guest_addr;
        self.prepare(at, guest_addr, cpu, temps);
        // A task continuation is pending only where a walk starts: no
        // instruction of an ordinary block (`flat32_ordinary`, or the real-mode
        // Direct class) switches tasks.
        debug_assert!(first || self.task_switch.context().is_none());
        if (first && self.task_switch.context().is_some())
            || (LANES && matches!(self.lane, Lane::Strict | Lane::Transaction))
        {
            self.locate();
            self.flow = Some(Flow::Yield);
            return Some(WalkResult::Continue);
        }
        self.begin(cpu);
        None
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
pub(crate) fn execute_shared<
    const ROUTED: bool,
    const ORDER: u8,
    const DIRECT20: bool,
    const DEVICE: bool,
>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    memory: &Memory,
    scratch: &mut impl ScratchBuffer,
    limit: u64,
    exit_word: &std::sync::atomic::AtomicU64,
    sti_shadow: bool,
    ss_shadow: u8,
    pending_irq: bool,
    nmi_blocked: bool,
) -> Progress {
    let code_base = if DIRECT20 {
        (((u64::from(cpu.cs) << 4) + ir.guest_start) & 0xfffff).wrapping_sub(ir.guest_start)
    } else {
        u64::from(cpu.cs) << 4
    };
    let mut steps = StepContext::new(limit);
    steps.pending_irq = pending_irq;
    // The Machine walk carries the whole `ss_shadow` byte in its StepContext.
    steps.ss_shadow = ss_shadow;
    steps.nmi_blocked = nmi_blocked;
    let mut info = RetireInfo {
        code_base,
        exit_request: Some(exit_word),
        sti_shadow,
        ..Default::default()
    };
    let result = super::interpret_direct_steps::<ROUTED, ORDER, DIRECT20, DEVICE>(
        ir, cpu, memory, scratch, &mut info, &mut steps,
    );
    if info.retired != 0 {
        cpu.rflags_system &= !(1 << 16);
    }
    let ss_shadow = steps.ss_shadow;
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
            ss_shadow,
            outcome: None,
        };
    }
    let outcome = match result {
        BlockResult::Continue | BlockResult::ModeChange => Ok(Outcome {
            arm_sti: info.sti_shadow,
            arm_ss: ss_shadow,
            ..Outcome::default()
        }),
        BlockResult::Exit(exit) => match exit.into_exit() {
            Exit::BudgetExhausted => Ok(Outcome {
                arm_sti: info.sti_shadow,
                arm_ss: ss_shadow,
                ..Outcome::default()
            }),
            exit @ (Exit::Hlt | Exit::PortIo { .. } | Exit::Interrupted) => Ok(Outcome {
                exit: Some(exit),
                arm_sti: info.sti_shadow,
                arm_ss: ss_shadow,
                ..Outcome::default()
            }),
            // The twenty-bit walk moves every RAM byte through a one-byte
            // transfer (`interp::i8086`; lift/i8086.rs lowers loads, stores and
            // strings to its Wrapped primitives), and a one-byte access lies in
            // one region or in none. A device word on a sixteen-bit bus
            // (xj-50w.20) is made only when both its bytes lie in one Trap
            // region (`device_word`), so it is a device exit and never a span.
            // The one wider RAM transfer is a locked read-modify-write
            // (`LockedRmw20`), whose containment miss over adjacent regions the
            // re-execution would return unchanged (`SplitMemory` performs no
            // locked operand). So this arm keeps its pre-split shape.
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
        ss_shadow,
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
    let ss_shadow = steps.ss_shadow;
    let progress = |outcome| Progress {
        steps: steps.steps,
        retired: info.retired,
        sti_shadow: info.sti_shadow,
        ss_shadow,
        outcome,
    };
    let boundary = Outcome {
        arm_sti: info.sti_shadow,
        arm_ss: ss_shadow,
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
pub(crate) fn execute_shared20<const ROUTED: bool, const ORDER: u8, const DEVICE: bool>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    memory: &Memory,
    scratch: &mut impl ScratchBuffer,
    limit: u64,
    exit_word: &std::sync::atomic::AtomicU64,
    sti_shadow: bool,
    ss_shadow: u8,
    pending_irq: bool,
    nmi_blocked: bool,
) -> Progress {
    execute_shared::<ROUTED, ORDER, true, DEVICE>(
        ir,
        cpu,
        memory,
        scratch,
        limit,
        exit_word,
        sti_shadow,
        ss_shadow,
        pending_irq,
        nmi_blocked,
    )
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
/// `DEVICE` selects the Shared walk whose memory provider may complete a
/// registered MMIO transfer in its block (`RunningMemory`).
pub(crate) fn execute_block<const ROUTED: bool, const ORDER: u8, const DEVICE: bool>(
    walk: BlockWalk,
    ir: &IrBlock,
    cpu: &mut CpuState,
    memory: &Memory,
    scratch: &mut impl ScratchBuffer,
    limit: u64,
    exit_word: &std::sync::atomic::AtomicU64,
    task_switch: &mut tasks::SwitchState,
    sti_shadow: bool,
    ss_shadow: u8,
    pending_irq: bool,
    nmi_blocked: bool,
) -> Progress {
    match walk {
        BlockWalk::Shared => execute_shared::<ROUTED, ORDER, false, DEVICE>(
            ir,
            cpu,
            memory,
            scratch,
            limit,
            exit_word,
            sti_shadow,
            ss_shadow,
            pending_irq,
            nmi_blocked,
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
        BlockWalk::Checked | BlockWalk::Protected | BlockWalk::Paged => {
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
        ss_shadow: 0,
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
            memory.0.note_unobserved_write(addr, usize::from(size));
        }
    };
    // The alias keeps its 32-bit address formation (decision-23 §6).
    let mut access = LegacyAccess::new(
        memory.0.normal(),
        if saved.is_some() {
            u32::MAX
        } else {
            legacy::physical_mask(context.generation)
        },
    );
    access.writes = Some(&observe);
    access.frozen_alias = saved.is_some();
    let mut outcome = Outcome {
        retired: true,
        ..Outcome::default()
    };
    let result = execute_op::<ROUTED, ORDER, false, false>(
        &ops[0],
        &InstructionView {
            guest_start: ip,
            guest_len: (next - ip) as u32,
            ops: &ops,
            proven: false,
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
fn execute_op<const ROUTED: bool, const ORDER: u8, const PAGED: bool, const V86: bool>(
    op: &IrOp,
    ir: &InstructionView<'_>,
    cpu: &mut CpuState,
    mem: &LegacyAccess<'_, ROUTED, ORDER, PAGED>,
    temps: &mut [u64],
    context: MachineContext,
    task_switch: &mut tasks::SwitchState,
    outcome: &mut Outcome,
) -> Result<bool, Exit>
where
    PagingMode<PAGED>: PagingStorage,
{
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
        IrOp::AtomicCas {
            old,
            addr,
            expected,
            src,
            size,
            ..
        } => {
            // 486 PRM §26 CMPXCHG pp. 26-62–63, §3.11.7 p. 3-47:
            // LOCK makes the memory comparison/update one bus transaction.
            if PAGED {
                temps[*old as usize] = paging::locked(
                    cpu,
                    mem,
                    read_val(*addr, temps),
                    read_val(*src, temps),
                    *size,
                    super::integer::LockedUpdate::Cas(read_val(*expected, temps) & mask(*size)),
                )?;
                return Ok(false);
            }
            let linear = read_val(*addr, temps);
            let expected = read_val(*expected, temps) & mask(*size);
            let src = read_val(*src, temps);
            let update = super::integer::LockedUpdate::Cas(expected);
            let (physical, part) = legacy::physical_span(cpu, mem.physical_mask, linear, *size);
            let host = if part == *size {
                match mem.atomic_cas(physical, expected, src, *size) {
                    Ok(value) => Some(value),
                    Err(MemTrap::Mmio) => {
                        if !matches!(
                            mem.0.region_kind(physical, *size),
                            Ok(crate::RegionKind::Trap)
                        ) {
                            Some(locked_split(cpu, mem, ir, linear, src, *size, update)?)
                        } else {
                            None
                        }
                    }
                    Err(trap) => {
                        let StepResult::Exit(exit) =
                            trap_out(cpu, cpu.rip, trap, physical, *size, AccessKind::Write, src)
                                .into_step()
                        else {
                            unreachable!("a memory trap exits")
                        };
                        return Err(crate::fault::qualify_legacy_bus(exit, linear, physical));
                    }
                }
            } else {
                None
            };
            temps[*old as usize] = if let Some(value) = host {
                value
            } else {
                let trap_bytes = (0..*size)
                    .filter(|&byte| {
                        let at = u64::from((linear as u32).wrapping_add(u32::from(byte)));
                        let (bus, _) = legacy::physical_span(cpu, mem.physical_mask, at, 1);
                        matches!(mem.0.region_kind(bus, 1), Ok(crate::RegionKind::Trap))
                    })
                    .count();
                if trap_bytes != 0 && trap_bytes != usize::from(*size) {
                    locked_split(cpu, mem, ir, linear, src, *size, update)?
                } else {
                    // Same split-RAM atomicity limitation as AtomicRmw below.
                    probe_locked(cpu, mem, linear, *size)?;
                    let value = read(cpu, mem, linear, *size)?;
                    write(cpu, mem, linear, *size, update.apply(value, src, *size))?;
                    value
                }
            };
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
            if PAGED {
                temps[*old as usize] = paging::locked(
                    cpu,
                    mem,
                    read_val(*addr, temps),
                    read_val(*src, temps),
                    *size,
                    super::integer::LockedUpdate::Rmw(*op),
                )?;
                return Ok(false);
            }
            let linear = read_val(*addr, temps);
            let (physical, part) = legacy::physical_span(cpu, mem.physical_mask, linear, *size);
            let completed = if part == *size {
                match mem.atomic_rmw(physical, read_val(*src, temps), *size, *op) {
                    Ok(value) => {
                        temps[*old as usize] = value;
                        true
                    }
                    // A device operand inside one Trap region takes the bus
                    // transfers below (spec.md §5.2); one partly in RAM is
                    // performed by pieces (`locked_split`).
                    Err(MemTrap::Mmio) => {
                        if !matches!(
                            mem.0.region_kind(physical, *size),
                            Ok(crate::RegionKind::Trap)
                        ) {
                            let src = read_val(*src, temps);
                            temps[*old as usize] = locked_split(
                                cpu,
                                mem,
                                ir,
                                linear,
                                src,
                                *size,
                                super::integer::LockedUpdate::Rmw(*op),
                            )?;
                            true
                        } else {
                            false
                        }
                    }
                    Err(trap) => {
                        let StepResult::Exit(exit) = trap_out(
                            cpu,
                            cpu.rip,
                            trap,
                            physical,
                            *size,
                            AccessKind::Write,
                            read_val(*src, temps),
                        )
                        .into_step() else {
                            unreachable!("a memory trap exits")
                        };
                        return Err(crate::fault::qualify_legacy_bus(exit, linear, physical));
                    }
                }
            } else {
                // A20 splits the operand into bus pieces. All in devices: each
                // piece is a transfer of its own. Devices and RAM mixed: the
                // pieces of an operand partly in a Trap region (`locked_split`).
                let trap_bytes = (0..*size)
                    .filter(|&byte| {
                        let linear = u64::from((linear as u32).wrapping_add(u32::from(byte)));
                        let (physical, _) =
                            legacy::physical_span(cpu, mem.physical_mask, linear, 1);
                        matches!(mem.0.region_kind(physical, 1), Ok(crate::RegionKind::Trap))
                    })
                    .count();
                if trap_bytes != 0 && trap_bytes != usize::from(*size) {
                    let src = read_val(*src, temps);
                    temps[*old as usize] = locked_split(
                        cpu,
                        mem,
                        ir,
                        linear,
                        src,
                        *size,
                        super::integer::LockedUpdate::Rmw(*op),
                    )?;
                    true
                } else {
                    false
                }
            };
            if !completed {
                // A discontinuous bus operand has no single host atomic address.
                // LIMITATION: a RAM operand split by A20 takes the policy of an
                // operand that crosses an aligned 16-byte granule
                // (`Memory::locked_unaligned`, spec.md §8.2.3): one read, then one
                // write, not atomic against another vCPU. A
                // device operand is read once, then written once: the LOCK
                // semantics "are followed for as many bus cycles as necessary
                // to update the entire operand" (SDM Vol 3A §11.1.2.2; 80286
                // PRM Appendix B, XCHG p. B-114). Each bus piece must still lie
                // inside one region. The Machine ledger keys these transfers as
                // for any other Machine access.
                probe_locked(cpu, mem, linear, *size)?;
                // An exchange whose old value the instruction discards (the
                // locked MOV) is a store: it reads nothing.
                let value = if matches!(op, crate::ir::RmwOp::Xchg)
                    && super::integer::exchange_discards_old(ir.ops, cpu.rip)
                {
                    0
                } else {
                    read(cpu, mem, linear, *size)?
                };
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
                        || bytes[..usize::from(*len)].last() == Some(&0x9b)
                        || [0xe2, 0xe3, 0xe4]
                            .iter()
                            .any(|op| bytes[..usize::from(*len)].ends_with(&[0xdb, *op])))
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
                absent_coprocessor_store(cpu, mem, *kind, read_val(*addr, temps), *sti)?;
                return Ok(false);
            }
            numeric_error_before(cpu, *kind)?;
            let address = read_val(*addr, temps);
            if PAGED {
                paging::x87(cpu, mem, *kind, address, *sti)?;
            } else {
                let fp = PhysicalFpMemory::<ROUTED> {
                    memory: &mem.0,
                    a20_enabled: cpu.legacy.a20_enabled != 0,
                    physical_mask: mem.physical_mask,
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
                    let (physical, _) = legacy::physical_span(cpu, mem.physical_mask, at, 1);
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
            }
            use crate::x87::FpuKind as K;
            if matches!(
                kind,
                K::Fldcw | K::Fldenv | K::Frstor | K::Fnstenv | K::Fnsave | K::Fninit | K::Fnclex
            ) {
                // SDM Vol 1 §8.1.3.3 p. 8-5; Intel486 PRM §15.1.2
                // pp. 15-2–3 (387 witness, §25.1 pp. 25-1–2): ES/B
                // summarize flags whose current masks are clear. Refresh after
                // the transfer commits so stores retain the pre-change image.
                // 80286/287 PRM Part II Ch. 1, "The NPX Status Word",
                // p. 1-10 gives the same ES rule, but B denotes execution/error
                // busy rather than mirroring ES; retain that model's B policy.
                let summary = match context.generation {
                    crate::CpuGeneration::I386 => 0x8080,
                    crate::CpuGeneration::I286 => 0x0080,
                    crate::CpuGeneration::I8086 => 0,
                };
                let pending = cpu.fpu_sw & !cpu.fpu_cw & 0x3f != 0;
                cpu.fpu_sw = (cpu.fpu_sw & !summary) | if pending { summary } else { 0 };
            }
            crate::x87::record_x87_instruction(
                cpu,
                read_val(*operand_offset, temps),
                ir.guest_start,
                *instruction,
            );
            if context.generation == crate::CpuGeneration::I386
                && instruction & crate::x87::X87_RECORD != 0
            {
                // Intel486 PRM §15.1.6 p. 15-8, second witness for the 387
                // programming model: each non-control ESC records its opcode.
                // SDM Vol 1 §8.1.9.1 p. 8-10 describes this original policy.
                cpu.fpu_fop = *instruction as u16 & 0x7ff;
            }
            // Numeric transfers and pointer updates have committed; the typed
            // exit is instruction-produced and includes this retirement.
            if cpu.numeric_error_route == crate::NumericErrorRoute::External {
                outcome.exit = cpu.reconcile_numeric_error(next);
            }
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
            temps[*dst as usize] = if PAGED {
                let write = ir
                    .ops
                    .iter()
                    .any(|op| matches!(op, IrOp::Store { addr: stored, .. } if stored == addr));
                paging::read(
                    cpu,
                    mem,
                    address,
                    *size,
                    if write {
                        AccessKind::Write
                    } else {
                        AccessKind::Read
                    },
                )
            } else {
                read(cpu, mem, address, *size)
            }
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
                L::I486Gate => {
                    // ASSUMED: conservative refusal of the whole regime,
                    // including aligned accesses.
                    if alignment_regime(cpu) {
                        return Err(refusal(
                            cpu,
                            crate::UnsupportedSystemOperation::LegacyAlignmentCheck,
                        ));
                    }
                }
                // Intel486 PRM §26 INVD/WBINVD pp. 26-171, 26-281: operate
                // on caches. The functional engine models no data cache.
                L::I486Invalidate => {}
                L::InvalidatePage { segment, offset } => {
                    // Intel486 PRM §26 INVLPG p. 26-172; SDM Vol 3A
                    // §5.10.4.1 pp. 5-48–49 permits conservative invalidation.
                    // Central address lowering supplies the wrapped offset.
                    // There is no segment access, page walk or operand read.
                    if PAGED {
                        let _linear = cpu.legacy.segments[*segment as usize]
                            .base
                            .wrapping_add(read_val(*offset, temps) as u32);
                        outcome.translations_invalidated = true;
                    }
                }
                L::I486Cpuid { signature } => {
                    // Embedded 486 DM §§D.2–D.3 pp. D-1–2. ASSUMED:
                    // reserved bits zero; model-1 DX exposes only on-chip FPU.
                    let [eax, ebx, ecx, edx] = match cpu.gpr[RAX] as u32 {
                        0 => [1, 0x756e_6547, 0x6c65_746e, 0x4965_6e69],
                        1 => [*signature, 0, 0, crate::model::I486_DX_CPUID_FEATURES],
                        _ => [0; 4],
                    };
                    for (reg, value) in [(RAX, eax), (RBX, ebx), (RCX, ecx), (RDX, edx)] {
                        cpu.gpr[reg] = u64::from(value);
                    }
                }
                // The instruction executor runs the sequence a header stands for.
                L::Checkpoint
                | L::SharedWalk
                | L::FusedPush { .. }
                | L::FusedPop { .. }
                | L::FusedPush20 { .. }
                | L::FusedPop20 { .. } => {}
                L::Privileged { kind } => {
                    privilege::check::<ROUTED, ORDER, V86>(cpu, mem, context.generation, *kind)?
                }
                L::LockedTransfer => return Err(atomic_device_exit(cpu, mem, ir)),
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
                L::CoprocessorOverrun {
                    segment,
                    offset,
                    bytes,
                    access,
                    vector,
                    kind,
                } => coprocessor_overrun(
                    cpu,
                    context,
                    *segment,
                    read_val(*offset, temps),
                    *bytes,
                    *access,
                    *vector,
                    *kind,
                )?,
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
                    // 80386 PRM §§11.1.2–11.1.4 pp. 11-1–2 describe the
                    // ESC/WAIT gates, with absent BUSY#/ERROR# tied inactive.
                    // 286 PRM §10.2.2 pp. 10-4–5 recommends MP or EM for ESC;
                    // Part II Ch. 3 p. 3-4 describes the same emulation gate.
                    // Neither defines arbitrary absent-socket output data.
                    if !*wait
                        && cpu.fpu_model == crate::FpuModel::None
                        && context.generation != crate::CpuGeneration::I286
                        && !ir
                            .ops
                            .iter()
                            .any(|op| matches!(op, IrOp::Trap { vector: 6, .. }))
                        && crate::x87::esc_output(&bytes[..usize::from(*len)])
                        && !ir.ops.iter().any(|op| {
                            matches!(
                                op,
                                IrOp::X87 {
                                    kind: crate::x87::FpuKind::Fnstcw
                                        | crate::x87::FpuKind::Fnstsw
                                        | crate::x87::FpuKind::FnstswMem
                                        | crate::x87::FpuKind::Fnstenv
                                        | crate::x87::FpuKind::Fnsave,
                                    ..
                                }
                            )
                        })
                    {
                        return Err(Exit::unknown_instruction(cpu.rip, *bytes, *len));
                    }
                }
                L::Enter { size, alloc, level } => {
                    if !PAGED {
                        prepare_enter_displays(
                            cpu,
                            context,
                            mem.physical_mask,
                            *size,
                            *alloc,
                            *level,
                        );
                    }
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
                    let final_address =
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
                            let physical = if PAGED {
                                paging::byte_bus(cpu, mem, linear, AccessKind::Write)?
                            } else {
                                legacy::physical_span(cpu, mem.physical_mask, linear, 1).0
                            };
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
                        let physical = if PAGED {
                            (0..*size)
                                .map(|byte| {
                                    let linear =
                                        u64::from((address as u32).wrapping_add(u32::from(byte)));
                                    paging::byte_bus(cpu, mem, linear, AccessKind::Read)
                                })
                                .collect::<Result<Vec<_>, Exit>>()?
                        } else {
                            (0..*size)
                                .map(|byte| {
                                    legacy::physical_span(
                                        cpu,
                                        mem.physical_mask,
                                        u64::from((address as u32).wrapping_add(u32::from(byte))),
                                        1,
                                    )
                                    .0
                                })
                                .collect::<Vec<_>>()
                        };
                        let cached = if PAGED { None } else { enter_display(cpu, i) };
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
                                    )?;
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
                            mem.skip_read(cpu, address, *size)?;
                            0
                        } else {
                            read(cpu, mem, address, *size)?
                        };
                        let value = cached.unwrap_or(value);
                        if !PAGED && cached.is_none() {
                            record_enter_display(
                                cpu,
                                mem.physical_mask,
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
                    if PAGED {
                        // SDM Vol 2A §3.3 ENTER pp. 3-295–297: display
                        // reads and frame pushes precede the allocation check.
                        paging::check_write_permission(cpu, mem, final_address)?;
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
                            // 486 PRM §4.1.3 pp. 4-6–7: CD/NW/AM/WP/NE;
                            // ET is fixed one. Reserved bits read zero (ASSUMED).
                            let value = if cpu.machine_flags_mask != 0 {
                                (value & 0xe005_003f) | 0x10
                            } else {
                                value & 0x8000_001f
                            };
                            outcome.context_changed = if PAGED && value & 0x8000_0000 != 0 {
                                paging::write_cr0(cpu, value)?
                            } else {
                                write_cr0(cpu, value)?
                            };
                        }
                        ControlRegister::Cr2 => cpu.legacy.cr2 = value,
                        ControlRegister::Cr3 => {
                            // 486 PRM §4.1.3 p. 4-6: CR3.PCD/PWT.
                            cpu.legacy.cr3 = value
                                & if cpu.machine_flags_mask != 0 {
                                    0xffff_f018
                                } else {
                                    0xffff_f000
                                };
                            // Decision-18 §3: even an equal CR3 write invalidates.
                            outcome.context_changed = true;
                            outcome.translations_invalidated = true;
                        }
                    }
                }
                L::ReadDebugTest { dst, register } => {
                    temps[*dst as usize] = u64::from(cpu.debug_test_registers[*register as usize]);
                }
                L::WriteDebugTest { register, value } => {
                    let value = read_val(*value, temps) as u32;
                    let value = match register {
                        DebugTestRegister::Dr6 => {
                            // 80386 PRM §12.2 Fig. 12-1 pp. 12-3–5 defines
                            // B0–B3, BD, BS, BT; reserved bits are "do not define".
                            // ASSUMED: freeze reserved bits to the SDM Vol 3A
                            // §12.1 Table 12-1 p. 12-3 image, including bit 12=0.
                            (value & 0x0000_e00f) | 0xffff_0ff0
                        }
                        DebugTestRegister::Dr7 => {
                            // 80386 PRM §12.2.2 pp. 12-3–4: L0–L3/G0–G3
                            // enable breakpoints. SDM Vol 3B §20.2.4 p. 20-5:
                            // GD enables debug-register access traps.
                            // ASSUMED: refuse before committing, including GD,
                            // so unsupported debug traps cannot be silently lost.
                            if value & 0x20ff != 0 {
                                return Err(refusal(
                                    cpu,
                                    crate::UnsupportedSystemOperation::LegacyDebugBreakpoints,
                                ));
                            }
                            // ASSUMED: freeze reserved bits (PRM Fig. 12-1
                            // p. 12-3) to the SDM Table 12-1 p. 12-3 image.
                            (value & 0xffff_23ff) | 0x400
                        }
                        DebugTestRegister::Tr6 => {
                            // 80386 PRM §10.6.2 p. 10-14: both C=0 (write)
                            // and C=1 (lookup) issue an immediate TLB command.
                            // ASSUMED: refuse every command, also with PG=0.
                            return Err(refusal(
                                cpu,
                                crate::UnsupportedSystemOperation::LegacyTlbTest,
                            ));
                        }
                        _ => value,
                    };
                    cpu.debug_test_registers[*register as usize] = value;
                }
                L::LoadMsw { value } => {
                    let value = legacy::lmsw_value(cpu.legacy.cr0, read_val(*value, temps) as u16);
                    outcome.context_changed = write_cr0(cpu, value)?;
                }
                L::ClearTs => {
                    // SDM Vol 2A §3.3 CLTS; 80386 PRM §17 p. 17-48.
                    outcome.context_changed = write_cr0(cpu, cpu.legacy.cr0 & !8)?;
                }
                L::Loadall => {
                    // The document is silent on events after LOADALL; the
                    // engine's choices are ASSUMED. No interrupt or NMI hold
                    // follows, even when SS loads, and NMI blocking does not
                    // end. A TF set before LOADALL traps after it, saving
                    // the loaded CS:IP; a TF it loads traps after the next
                    // instruction, as for IRET or a task switch (I286 PRM
                    // §9.7.1 p. 9-14).
                    loadall(cpu, mem)?;
                    outcome.context_changed = true;
                    return Ok(true);
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
                            SelectorDestination::Memory { addr, segment } => {
                                // 80386 PRM §17 p. 17-25: "if the result is in a
                                // nonwritable segment" applies when a result is written.
                                // SDM Vol 2A §3.3 ARPL p. 3-75; Intel486 PRM §26
                                // p. 26-27: only the raising branch writes DEST.
                                let addr = read_val(*addr, temps);
                                let offset = (addr as u32)
                                    .wrapping_sub(cpu.legacy.segments[*segment as usize].base);
                                segment_access(
                                    cpu,
                                    context,
                                    *segment,
                                    u64::from(offset),
                                    2,
                                    AccessKind::Write,
                                )?;
                                write(cpu, mem, addr, 2, value)?;
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
                        let origin = SoftwareInterrupt {
                            vector: *vector,
                            cs: cpu.cs,
                            ip: *saved_ip,
                            cs_base: cpu.legacy.segments[SegmentId::Cs as usize].base,
                            cpl: context.cpl,
                            vm: V86,
                        };
                        // Stored before the delivery: a task switch handed off
                        // to the embedder completes in `tasks::resume`, whose
                        // outcome reads it (`execute_instruction`).
                        task_switch.software_interrupt = origin;
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
                        outcome.software_interrupt = true;
                        return Ok(true);
                    }
                }
                L::SegmentedAddress {
                    dst,
                    segment,
                    offset,
                    bytes,
                    access,
                    proof,
                } => {
                    let offset = read_val(*offset, temps);
                    temps[*dst as usize] = proof.address(offset, *bytes).ok_or_else(|| {
                        legacy::segment_fault(
                            cpu,
                            if *segment == SegmentId::Ss { 12 } else { 13 },
                            offset,
                            *bytes,
                            *access,
                        )
                    })?;
                }
                L::Flat32Address {
                    dst,
                    segment,
                    offset,
                    bytes,
                    access,
                } => {
                    temps[*dst as usize] =
                        flat32_address(cpu, *segment, read_val(*offset, temps), *bytes, *access)?;
                }
                L::Address {
                    dst,
                    segment,
                    offset,
                    bytes,
                    access,
                } => {
                    temps[*dst as usize] = if V86 {
                        legacy::virtual8086_access(
                            cpu,
                            *segment,
                            read_val(*offset, temps),
                            *bytes,
                            *access,
                        )?
                    } else {
                        segment_access(
                            cpu,
                            context,
                            *segment,
                            read_val(*offset, temps),
                            *bytes,
                            *access,
                        )?
                    };
                }
                L::ReadSelector { dst, segment } => {
                    temps[*dst as usize] = u64::from(selector(cpu, *segment))
                }
                L::LiveField {
                    dst,
                    bus,
                    bytes,
                    extend,
                } => temps[*dst as usize] = live_field(mem.raw(), *bus, *bytes, *extend)?,
                L::LoadSegment {
                    segment,
                    value,
                    inhibit,
                } => {
                    if V86 {
                        reload_virtual8086(cpu, *segment, read_val(*value, temps) as u16);
                    } else {
                        load_segment(
                            cpu,
                            mem,
                            *segment,
                            read_val(*value, temps) as u16,
                            context.generation,
                        )?;
                    }
                    outcome.context_changed = true;
                    // spec.md §17.10: an SS load also holds the NMI input.
                    outcome.arm_ss = if *inhibit {
                        crate::vm::SS_HOLD | crate::vm::NMI_HOLD
                    } else {
                        0
                    };
                }
                L::Push { value, size } => {
                    push(cpu, mem, context, &[read_val(*value, temps)], *size)?
                }
                L::Pop { dst, size } => temps[*dst as usize] = peek(cpu, mem, context, 0, *size)?,
                L::AdjustStack { bytes } => adjust(cpu, context, *bytes),
                L::Flags { size, pop } => {
                    if *pop {
                        let mut image = peek(cpu, mem, context, 0, *size)? as u32;
                        if *size == 2 {
                            image |= cpu.rflags_system as u32 & cpu.machine_flags_mask;
                        }
                        let before = cpu.rflags_system;
                        privilege::load_flags(cpu, context.generation, image)?;
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
                        // preflight the frame before any device read,
                        // including the discarded SP/ESP slot. In real mode
                        // it wraps (I286 PRM App. B POPA p. B-85).
                        pop_frame_limit(cpu, mem, context, 8, *size)?;
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
                        // An even SP wraps and completes (`push_frame`).
                        if context.regime == legacy::MachineRegime::Real
                            && matches!(sp(cpu, context), 7 | 9 | 11 | 13 | 15)
                        {
                            return Err(exception(cpu, 13));
                        }
                        let values = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| cpu.gpr[i]);
                        push_frame(cpu, mem, context, &values, *size)?;
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
                                &mem.address_keyed(),
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
                    let mut descriptor = None;
                    if let (Some(cs), legacy::MachineRegime::Protected) = (cs, context.regime) {
                        let (d, gate) = far_descriptor(cpu, mem, cs, false, context.generation)?;
                        if gate {
                            privilege::call_gate(cpu, mem, context, cs, d, ip, *return_ip)?;
                            outcome.context_changed = true;
                            return Ok(true);
                        }
                        descriptor = Some(d);
                    }
                    if let Some(d) = &descriptor {
                        if return_ip.is_some() {
                            push_addresses(cpu, mem, context, 2, *size)?;
                        }
                        code_limit(cpu, d.cache, ip)?;
                    } else if cs.is_some() {
                        far_real_target(cpu, context, ip, *size)?;
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
                        // I286 PRM App. B CALL p. B-26; 80386 PRM p. 17-42
                        // and Table 14-2 p. 14-8: the real-mode frame wraps.
                        push_frame(cpu, mem, context, &frame, *size)?;
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
                            if V86 {
                                reload_virtual8086(cpu, SegmentId::Cs, cs);
                            } else {
                                reload_real(cpu, SegmentId::Cs, cs, context.generation);
                            }
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
                    // 80386 PRM §17 RET p. 17-145; I286 PRM App. B RET
                    // p. B-95 read as a word popped at FFFFh (decision-24).
                    pop_frame_limit(cpu, mem, context, if *far { 2 } else { 1 }, *size)?;
                    let ip = peek(cpu, mem, context, 0, *size)?;
                    let cs = if *far {
                        Some(peek(cpu, mem, context, 1, *size)? as u16)
                    } else {
                        None
                    };
                    let ip = ip & mask(*size);
                    let mut descriptor = if context.regime == legacy::MachineRegime::Protected {
                        if let Some(cs) = cs.filter(|cs| (cs & 3) as u8 > cpu.legacy.cpl) {
                            privilege::outward(
                                cpu,
                                mem,
                                context,
                                privilege::Return {
                                    ip,
                                    cs,
                                    flags: None,
                                    size: *size,
                                    release: *pop_extra,
                                },
                            )?;
                            outcome.context_changed = true;
                            return Ok(true);
                        }
                        cs.map(|cs| code_descriptor(cpu, mem, cs, true, context.generation))
                            .transpose()?
                    } else {
                        None
                    };
                    if let Some(d) = &mut descriptor {
                        code_limit(cpu, d.cache, ip)?;
                        accessed(cpu, mem, d)?;
                    } else if cs.is_some() {
                        far_real_target(cpu, context, ip, *size)?;
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
                            if V86 {
                                reload_virtual8086(cpu, SegmentId::Cs, cs);
                            } else {
                                reload_real(cpu, SegmentId::Cs, cs, context.generation);
                            }
                        }
                        outcome.context_changed = true;
                    }
                    cpu.rip = ip;
                    return Ok(true);
                }
                L::Iret { size } => {
                    if V86 {
                        virtual8086::iret(cpu, mem, context, *size)?;
                        outcome.context_changed = true;
                        outcome.preserve_rf = true;
                        return Ok(true);
                    }
                    // SDM Vol 2A §3.3 IRET; 80386 PRM §17 pp. 17-74,
                    // 17-76–78 (scan checked): same-CPL frame, independent SS.B.
                    // Real mode wraps: I286 PRM App. B IRET p. B-53, 80386
                    // PRM §17 IRET p. 17-79 (`wraps_in_segment`).
                    pop_frame_limit(cpu, mem, context, 3, *size)?;
                    let peek = virtual8086::peek_frame::<ROUTED, ORDER>;
                    let ip = peek(cpu, mem, context, 0, *size)? & mask(*size);
                    let cs = peek(cpu, mem, context, 1, *size)? as u16;
                    let image = peek(cpu, mem, context, 2, *size)? as u32;
                    // SDM Vol 2A §3.3, IRET/IRETD/IRETQ, Operation,
                    // REAL-ADDRESS-MODE: (tempEFLAGS AND 257FD5H) OR
                    // (EFLAGS AND 1A0000H) preserves VM with PE clear.
                    if context.regime == legacy::MachineRegime::Protected
                        && *size == 4
                        && image & (1 << 17) != 0
                        && context.cpl == 0
                    {
                        virtual8086::enter(cpu, mem, context, ip, cs, image)?;
                        outcome.context_changed = true;
                        outcome.preserve_rf = true;
                        return Ok(true);
                    }
                    let mut descriptor = if context.regime == legacy::MachineRegime::Protected {
                        if (cs & 3) as u8 > cpu.legacy.cpl {
                            privilege::outward(
                                cpu,
                                mem,
                                context,
                                privilege::Return {
                                    ip,
                                    cs,
                                    flags: Some(image),
                                    size: *size,
                                    release: 0,
                                },
                            )?;
                            outcome.context_changed = true;
                            outcome.preserve_rf = true;
                            return Ok(true);
                        }
                        Some(code_descriptor(cpu, mem, cs, true, context.generation)?)
                    } else {
                        None
                    };
                    if let Some(descriptor) = &descriptor {
                        code_limit(cpu, descriptor.cache, ip)?;
                    } else {
                        far_real_target(cpu, context, ip, 4)?;
                    }
                    if let Some(d) = &mut descriptor {
                        accessed(cpu, mem, d)?;
                    }
                    // SDM Vol 3A §11.3: all frame reads complete before retirement.
                    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
                    let image = if *size == 2 {
                        image | (cpu.rflags_system as u32 & cpu.machine_flags_mask)
                    } else {
                        image
                    };
                    privilege::load_flags(cpu, context.generation, image)?;
                    // IRET restores RF in a 32-bit frame; POPFD does not.
                    if *size == 4 {
                        cpu.rflags_system =
                            (cpu.rflags_system & !(1 << 16)) | u64::from(image & (1 << 16));
                    }
                    adjust(cpu, context, 3 * u32::from(*size));
                    if let Some(d) = descriptor {
                        install(cpu, SegmentId::Cs, cs, d.cache);
                    } else {
                        reload_real(cpu, SegmentId::Cs, cs, context.generation);
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
                        if PAGED && !output {
                            paging::probe(cpu, mem, address, *elem)?;
                        }
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
                    // 80386 PRM §17 IRET p. 17-74: PE=0 pops the stack
                    // before considering NT. Retain the I286 boundary unchanged.
                    if context.regime == legacy::MachineRegime::Real
                        && context.generation == crate::CpuGeneration::I286
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
                        tasks::task_return(
                            cpu,
                            &mem.address_keyed(),
                            context,
                            next as u32,
                            task_switch,
                        )?;
                        outcome.context_changed = true;
                        outcome.preserve_rf = true;
                        return Ok(true);
                    }
                }
                // Twenty-bit operations belong to the dedicated shared walk.
                // Keep that execution machinery out of the system-model walker.
                L::WrappedLoad { .. }
                | L::WrappedStore { .. }
                | L::LockedRmw20 { .. }
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
                | L::InhibitIrq20
                | L::HoldExternal20 => {
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

// ASSUMED (maintainer policy, 2026-11-24): a floating absent data bus reads
// all ones. 286 PRM Part II Ch. 3 p. 3-2 calls floating data undefined;
// 386 PRM §11.1.2 pp. 11-1–2 defines no output value. Preserve the x87 image,
// including after FNSTENV/FNSAVE.
#[cold]
#[inline(never)]
fn absent_coprocessor_store<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    kind: crate::x87::FpuKind,
    address: u64,
    sti: u8,
) -> Result<(), Exit> {
    use crate::x87::FpuKind as K;
    if kind == K::Fnstsw {
        cpu.write_gpr(RAX, 0xffff, 2);
    } else if matches!(kind, K::Fnstcw | K::FnstswMem | K::Fnstenv | K::Fnsave) {
        let bytes = crate::x87::mem_operand_bytes_format(kind, sti) as usize;
        for offset in (0..bytes).step_by(8) {
            let linear = u64::from((address as u32).wrapping_add(offset as u32));
            probe(cpu, mem, linear, (bytes - offset).min(8) as u8)?;
        }
        for offset in (0..bytes).step_by(8) {
            let linear = u64::from((address as u32).wrapping_add(offset as u32));
            write(cpu, mem, linear, (bytes - offset).min(8) as u8, u64::MAX)?;
        }
    }
    Ok(())
}

// 80286 PRM App. B #MP p. B-9: the attached coprocessor's transfer raises
// the overrun "during the execution of the numeric instruction by the 80287".
// App. B #MF p. B-9 tests the ERROR pin "at the beginning of most floating
// point instructions", so a pending numeric error that the instruction tests
// raises interrupt 16 first and the transfer never starts. ASSUMED (accepted
// engine choice): the overrun is delivered as a fault of the ESC, before any
// transfer; §9.6.3 pp. 9-10–11 describes an asynchronous event and Tables
// 2-4/5-2 disagree on its return address (spec.md §17.10). With an empty
// socket the CPU operand check that follows applies unchanged (decision-25).
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn coprocessor_overrun(
    cpu: &CpuState,
    context: MachineContext,
    segment: SegmentId,
    offset: u64,
    bytes: u8,
    access: AccessKind,
    vector: u8,
    kind: Option<crate::x87::FpuKind>,
) -> Result<(), Exit> {
    if cpu.fpu_model == crate::FpuModel::None
        || segment_access(cpu, context, segment, offset, 1, access).is_err()
        || segment_access(cpu, context, segment, offset, bytes, access).is_ok()
    {
        return Ok(());
    }
    if let Some(kind) = kind {
        numeric_error_before(cpu, kind)?;
    }
    Err(legacy::coprocessor_overrun_fault(
        cpu, vector, offset, bytes, access,
    ))
}

/// The operations a Machine instruction walk executes itself (`execute_op`),
/// as a pattern: `walk_ops` matches them in its one op dispatch.
macro_rules! machine_specific_ops {
    () => {
        IrOp::AtomicRmw { .. }
            | IrOp::AtomicCas { .. }
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
    };
}
pub(super) use machine_specific_ops;

struct InstructionExecution<
    'a,
    'b,
    'memory,
    const ROUTED: bool,
    const ORDER: u8,
    const PAGED: bool,
    const V86: bool,
> where
    PagingMode<PAGED>: PagingStorage,
{
    ir: &'a InstructionView<'a>,
    mem: &'a LegacyAccess<'memory, ROUTED, ORDER, PAGED>,
    context: MachineContext,
    task_switch: &'b mut tasks::SwitchState,
    outcome: Outcome,
}
impl<const ROUTED: bool, const ORDER: u8, const PAGED: bool, const V86: bool>
    super::InstructionExecutor<'_>
    for &mut InstructionExecution<'_, '_, '_, ROUTED, ORDER, PAGED, V86>
where
    PagingMode<PAGED>: PagingStorage,
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
        match execute_op::<ROUTED, ORDER, PAGED, V86>(
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

// 80286 PRM Part I Appendix B #MF p. B-9; 80386 PRM §§11.1.2,
// 11.1.5.3: direct ERROR recognition precedes the waiting operation.
// External hold/acknowledgement is an engine abstraction of embedder board wiring.
fn numeric_error_before(cpu: &CpuState, kind: crate::x87::FpuKind) -> Result<(), Exit> {
    if cpu.numeric_error_route == crate::NumericErrorRoute::External {
        if crate::x87::waits_for_pending(kind) && cpu.external_numeric_error.held {
            return Err(Exit::ExternalNumericError {
                addr: cpu.rip,
                event: crate::NumericErrorEvent::Waiting,
            });
        }
    } else if crate::x87::mf_pending_before(cpu, kind) {
        return Err(exception(cpu, 16));
    }
    Ok(())
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
                &LegacyAccess::<false, 0>::new(&vm.mem, legacy::physical_mask(generation)),
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
                    &LegacyAccess::<false, 0>::new(&vm.mem, legacy::physical_mask(generation)),
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
                    &LegacyAccess::<false, 0>::new(&vm.mem, legacy::physical_mask(generation)),
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

#[cfg(test)]
mod loadall_tests {
    use super::*;
    use crate::{CpuGeneration, LegacyCpuModel, Prot, RegionKind, Vm, VmConfig};

    #[test]
    fn loadall_outside_privilege_level_zero_faults_before_reading_the_table() {
        // Intel, "Undocumented iAPX 286 Test Instruction", p. 1: in protected
        // mode at a privilege level other than 0, exception 13 with an error
        // code of 0. The test exercises the instruction directly at each
        // CPL. The table lies in a device region, so a read would stop with
        // an MMIO exit instead.
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Trap).unwrap();
        for cpl in 0..=3u8 {
            let mut cpu = CpuState::legacy_real_entry([u16::from(cpl), 0, 0, u16::from(cpl), 0, 0]);
            cpu.rip = 0x1000;
            cpu.legacy.cr0 = 1;
            cpu.legacy.cpl = cpl;
            let context = legacy::derive_legacy_context(
                LegacyCpuModel {
                    generation: CpuGeneration::I286,
                    model_semantics_id: 0,
                },
                &cpu,
            )
            .unwrap();
            let ir = crate::lift::legacy::instruction(b"\x0f\x05", 0x1000, context).unwrap();
            let before = format!("{cpu:?}");
            let exit = execute::<false, 0>(
                &ir,
                &mut cpu,
                &vm.mem,
                &mut Vec::new(),
                &mut tasks::SwitchState::default(),
            )
            .err()
            .unwrap();
            if cpl == 0 {
                assert!(
                    matches!(
                        exit,
                        Exit::MmioRead {
                            addr: 0x800,
                            size: 2,
                            ..
                        }
                    ),
                    "{exit:?}"
                );
            } else {
                assert!(
                    matches!(exit, Exit::Exception { vector: 13, .. }),
                    "{exit:?}"
                );
                assert_eq!(
                    exit.fault_metadata().unwrap().error_code(),
                    crate::fault::ErrorCode::bits(0, 0xffff)
                );
                assert_eq!(format!("{cpu:?}"), before);
            }
        }
    }
}

#[cfg(test)]
mod delivery_log_layout {
    use super::*;
    use std::mem::size_of;

    // Outcome and Progress as they were before the delivery log (spec.md
    // §17.10, Delivery log): the same fields, without its markers.
    #[allow(dead_code)]
    struct OutcomeBefore {
        exit: Option<Exit>,
        retired: bool,
        context_changed: bool,
        translations_invalidated: bool,
        arm_sti: bool,
        arm_ss: u8,
        suppress_debug: bool,
        preserve_rf: bool,
    }
    #[allow(dead_code)]
    struct ProgressBefore {
        sti_shadow: bool,
        ss_shadow: u8,
        steps: u64,
        retired: u64,
        outcome: Option<Result<OutcomeBefore, Exit>>,
    }

    // Each instruction returns an Outcome and each unit a Progress. The log's
    // one flag fills padding in Outcome, and Progress carries nothing for it,
    // so neither return grows while the log is off.
    #[test]
    fn the_delivery_log_markers_fit_the_existing_layout() {
        assert_eq!(size_of::<Outcome>(), size_of::<OutcomeBefore>());
        assert_eq!(
            size_of::<Result<Outcome, Exit>>(),
            size_of::<Result<OutcomeBefore, Exit>>()
        );
        assert_eq!(size_of::<Progress>(), size_of::<ProgressBefore>());
    }
}

#[cfg(test)]
mod walk_verdict {
    use super::*;

    #[test]
    fn an_op_outside_the_walk_is_refused_at_its_instruction() {
        // The `supported` verdict is taken when a block is built: only a
        // `flat32_ordinary` block gets the Direct walk, whose instructions skip
        // the run-time scan. An instruction carrying any other op (no 80386
        // or 80486 encoding lifts to one; this IR is hand-built) ends the
        // block before it and runs alone on the Protected walk, which refuses
        // it before it changes anything, the earlier instruction retired.
        let mut vm = crate::Vm::new(crate::VmConfig::flat(0x4000));
        vm.map(0, 0x4000, crate::Prot::RWX, crate::RegionKind::Ram)
            .unwrap();
        let mut cpu = CpuState {
            rip: 0x1000,
            ..CpuState::default()
        };
        cpu.legacy.cr0 = 1;
        cpu.legacy.a20_enabled = 1;
        for (i, cache) in cpu.legacy.segments[..4].iter_mut().enumerate() {
            cache.base = 0;
            cache.limit = u32::MAX;
            cache.attributes = if i == 0 { 0x9b } else { 0x93 } | 0x4000;
        }
        let model = crate::CpuModel::I80386.legacy_model();
        let context = legacy::derive_legacy_context(model, &cpu).unwrap();
        assert_eq!(context.segment_class, crate::SegmentClass::Flat32);
        let lift = |bytes: &[u8], ip| crate::lift::legacy::instruction(bytes, ip, context).unwrap();
        let first = lift(b"\x40", 0x1000); // INC EAX
        let mut second = lift(b"\x43", 0x1001); // INC EBX, then the foreign op
        second.ops.push(IrOp::Cpuid);
        assert!(!supported(&IrOp::Cpuid));
        let mut joined = first.clone();
        joined.ops.extend(second.ops.iter().cloned());
        joined.guest_len += second.guest_len;
        joined.icount += 1;

        // Build-time verdicts: the ordinary prefix is proven, the rest is not.
        assert!(matches!(BlockWalk::for_flat32(&first), BlockWalk::Direct));
        assert!(matches!(
            BlockWalk::for_flat32(&second),
            BlockWalk::Protected
        ));
        assert!(matches!(
            BlockWalk::for_flat32(&joined),
            BlockWalk::Protected
        ));

        let refused = |progress: &Progress| {
            matches!(
                progress.outcome,
                Some(Err(Exit::UnsupportedSystemContext {
                    operation: crate::UnsupportedSystemOperation::LegacyInstruction,
                    ..
                }))
            )
        };
        let exit_word = std::sync::atomic::AtomicU64::new(0);
        let mut scratch = Vec::new();
        let mut tasks = tasks::SwitchState::default();
        let mut run = |ir: &IrBlock, cpu: &mut CpuState| {
            execute_block::<false, 0, false>(
                BlockWalk::for_flat32(ir),
                ir,
                cpu,
                &vm.mem,
                &mut scratch,
                u64::MAX,
                &exit_word,
                &mut tasks,
                false,
                0,
                false,
                false,
            )
        };
        let start = cpu.clone();
        // The block the lifter builds stops before the foreign instruction.
        let progress = run(&first, &mut cpu);
        assert_eq!((progress.retired, cpu.rip, cpu.gpr[0]), (1, 0x1001, 1));
        let after_first = cpu.gpr;
        let progress = run(&second, &mut cpu);
        assert!(refused(&progress));
        assert_eq!((progress.steps, progress.retired), (0, 0));
        assert_eq!((cpu.rip, cpu.gpr), (0x1001, after_first));
        // A block holding it whole is refused at its start, as before.
        cpu = start.clone();
        let progress = run(&joined, &mut cpu);
        assert!(refused(&progress));
        assert_eq!((progress.steps, progress.retired), (0, 0));
        assert_eq!((cpu.rip, cpu.gpr), (0x1000, start.gpr));
    }
}
