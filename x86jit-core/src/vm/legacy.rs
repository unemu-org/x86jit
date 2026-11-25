//! Context caches and the shared Application/Machine dispatcher (decision-23 §5;
//! decision-20; spec.md §§9, 17). Machine and the Real16 alias stay interpreted.
use super::*;
use crate::interp::legacy::BlockWalk;
use crate::legacy::{
    derive_legacy_context, CpuGeneration, LegacyCpuState, MachineContext, MachineRegime, SegmentId,
};
use crate::IrOp;
use iced_x86::{Decoder, DecoderError};
use std::hash::{Hash, Hasher};

// The NMI input's cold helpers (spec.md §17.10), in their own module so that
// this module's hot dispatcher code keeps its code-generation unit.
mod nmi;
mod paging;

mod flat32;

/// VM-issued paging proof identity (decision-27). A process-wide issuer keeps
/// domains distinct even when a vCPU moves between VMs; zero is never issued.
pub(crate) fn paging_generation() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .expect("paging proof identifiers exhausted")
}

use nmi::External;
pub(super) use nmi::{NMI_BLOCKED, NMI_REMEMBERED};

// The retained context stamp guards the segment class. Flat32 and paging
// also own separate proof maps, so the shared key need not hash that class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LegacyBlockKey {
    owner_id: u64,
    context_id: u64,
    model_semantics_id: u64,
    decode_bits: u8,
    stack_bits: u8,
    regime: MachineRegime,
    decode_ip: u32,
    linear_fetch: u32,
    mxcsr_mode: u32,
}
impl Hash for LegacyBlockKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.owner_id.hash(state);
        self.context_id.hash(state);
        self.model_semantics_id.hash(state);
        // Preserve RandomState's byte stream without three short hasher calls.
        state.write(&[self.decode_bits, self.stack_bits, self.regime as u8]);
        self.decode_ip.hash(state);
        self.linear_fetch.hash(state);
        self.mxcsr_mode.hash(state);
    }
}
// `eq` below is the derived field-by-field comparison, so the derived `Hash` agrees.
#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Clone, Copy, Eq, Hash)]
struct LegacyContextStamp {
    model: LegacyCpuModel,
    state: LegacyCpuState,
    selectors: [u16; 6],
    system_flags: u64,
}
// The derived comparison, field by field, inlined into the per-run context check
// (a codegen pin: the derived `#[inline]` leaves the choice to the inliner).
impl PartialEq for LegacyContextStamp {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.model == other.model
            && self.state == other.state
            && self.selectors == other.selectors
            && self.system_flags == other.system_flags
    }
}
impl LegacyContextStamp {
    fn new(cpu: &CpuState, model: LegacyCpuModel) -> Self {
        let mut state = cpu.legacy;
        state.cr2 = 0; // Output register, not a translation assumption.
        Self {
            model,
            state,
            selectors: [cpu.cs, cpu.ds, cpu.es, cpu.ss, cpu.fs, cpu.gs],
            system_flags: cpu.rflags_system & 0x27000,
        }
    }
    #[inline(always)]
    fn matches(&self, cpu: &CpuState, model: LegacyCpuModel) -> bool {
        #[inline(always)]
        fn cache_bits(c: &crate::legacy::SegmentCache) -> (u64, u64) {
            (
                u64::from(c.base) | (u64::from(c.limit) << 32),
                u64::from(c.attributes)
                    | (u64::from(c.unusable) << 16)
                    | (u64::from(c.reserved) << 24),
            )
        }
        #[inline(always)]
        fn table_bits(t: &crate::legacy::TableRegister) -> u64 {
            u64::from(t.base) | (u64::from(t.limit) << 32) | (u64::from(t.reserved) << 48)
        }
        #[inline(always)]
        fn system_bits(s: &crate::legacy::SystemSegment) -> (u64, u64) {
            (
                u64::from(s.selector)
                    | (u64::from(s.reserved) << 16)
                    | (u64::from(s.cache.base) << 32),
                u64::from(s.cache.limit)
                    | (u64::from(s.cache.attributes) << 32)
                    | (u64::from(s.cache.unusable) << 48)
                    | (u64::from(s.cache.reserved) << 56),
            )
        }
        #[inline(always)]
        fn tail_bits(s: &LegacyCpuState) -> u32 {
            u32::from(s.cpl)
                | (u32::from(s.a20_enabled) << 8)
                | (u32::from(s.reset_fetch) << 16)
                | (u32::from(s.reserved) << 24)
        }
        // Compare initialized fields, without reading padding or copying the
        // normalized state. CR2 remains output-only (decision-23 §5).
        let a = &self.state;
        let b = &cpu.legacy;
        let mut diff = u64::from(self.model != model)
            | u64::from(self.selectors != [cpu.cs, cpu.ds, cpu.es, cpu.ss, cpu.fs, cpu.gs])
            | (self.system_flags ^ (cpu.rflags_system & 0x27000))
            | u64::from(a.cr0 ^ b.cr0)
            | u64::from(a.cr3 ^ b.cr3)
            | (table_bits(&a.gdtr) ^ table_bits(&b.gdtr))
            | (table_bits(&a.idtr) ^ table_bits(&b.idtr))
            | u64::from(tail_bits(a) ^ tail_bits(b));
        for (a, b) in a.segments.iter().zip(&b.segments) {
            let (a0, a1) = cache_bits(a);
            let (b0, b1) = cache_bits(b);
            diff |= (a0 ^ b0) | (a1 ^ b1);
        }
        for (a, b) in [(&a.ldtr, &b.ldtr), (&a.tr, &b.tr)] {
            let (a0, a1) = system_bits(a);
            let (b0, b1) = system_bits(b);
            diff |= (a0 ^ b0) | (a1 ^ b1);
        }
        diff == 0
    }
}
// Decision-20: the firmware-style transition loop uses four contexts. Sixteen
// leaves room for a handful of code/stack configurations while bounding churn.
const MAX_RETAINED_CONTEXTS: usize = 16;

struct RetainedContext {
    id: u64,
    last_used: u64,
}

struct Block {
    ir: Arc<IrBlock>,
    walk: BlockWalk,
    bytes: Vec<u8>,
}
struct InstructionRetry {
    stamp: LegacyContextStamp,
    ip: u64,
    mxcsr_mode: u32,
    ir: Arc<IrBlock>,
    walk: BlockWalk,
}
#[derive(Clone, Copy)]
enum EventSource {
    SingleStep,
    Irq(u8),
    Exception {
        vector: u8,
        error_code: Option<u32>,
    },
    /// An accepted NMI (spec.md §17.10); `shutdown` when it ends an I80286 or
    /// I80386 shutdown.
    Nmi {
        shutdown: bool,
    },
}
#[derive(Clone, Copy)]
struct PendingEvent {
    irq_consumed: bool,
    /// The unit that raised this single-step trap took a software interrupt
    /// (8086 AP-67 p. A-28; read only with an NMI latched).
    software: bool,
    /// An exception delivered serially after an NMI's delivery faulted: it
    /// carries that NMI's blocking, which ends if this delivery never enters
    /// the guest either (spec.md §17.10).
    after_nmi: bool,
    source: EventSource,
    ip: u32,
    cs: u16,
    /// The return address, CS base and CPL the delivery log records for this
    /// event (spec.md §17.10, Delivery log). It starts as `cs:ip` and the
    /// state around it, and changes only when a fault after a committed task
    /// switch makes the event an exception of the new task: a bus handoff in
    /// the continuation moves `cs:ip` to the new task, not the record.
    origin: Origin,
}

/// The return address a delivery saves, with the CS base and CPL in effect
/// before it (spec.md §17.10, Delivery log).
#[derive(Clone, Copy, Default)]
pub(super) struct Origin {
    cs: u16,
    ip: u32,
    cs_base: u32,
    cpl: u8,
    vm: bool,
}

/// Where the delivery log takes the interrupted registers of a delivery
/// from, when it records them (spec.md §17.10, Delivery log).
#[derive(Clone, Copy)]
pub(super) enum Interrupted {
    /// An 80286/80386 delivery: the copy `interp::legacy::deliver` took when
    /// the delivery began (`SwitchState::regs`).
    Captured,
    /// A twenty-bit delivery, which changes only SP, FLAGS, CS and IP: the
    /// vCPU state with SP before its six-byte frame, and the FLAGS image the
    /// frame holds.
    Frame20 { flags: u16 },
}

// The kind a delivery log records for an event's source. BOUND's #BR is a
// fault that saves BOUND (SDM Vol 3A §7.15, Event 5 p. 7-31).
fn delivery_kind(source: EventSource) -> crate::DeliveryKind {
    match source {
        EventSource::SingleStep => crate::DeliveryKind::Trap,
        EventSource::Irq(_) => crate::DeliveryKind::Irq,
        EventSource::Nmi { .. } => crate::DeliveryKind::Nmi,
        EventSource::Exception { .. } => crate::DeliveryKind::Fault,
    }
}

fn exception_source(exit: &Exit) -> Option<EventSource> {
    let Exit::Exception { vector, .. } = exit else {
        return None;
    };
    let error_code = exit.fault_metadata().and_then(|m| match m.error_code() {
        crate::fault::ErrorCode::Bits { value, .. } => Some(value),
        _ => None,
    });
    Some(EventSource::Exception {
        vector: *vector,
        error_code,
    })
}

fn double_fault(generation: CpuGeneration, first: EventSource, second: EventSource) -> bool {
    let EventSource::Exception { vector: first, .. } = first else {
        return false;
    };
    let EventSource::Exception { vector: second, .. } = second else {
        return false;
    };
    // I286 PRM §9.6.2 p. 9-10 excludes #9 from the first-fault set;
    // I386 PRM §9.8.8 Table 9-3 p. 9-13 includes it as contributory.
    if generation == CpuGeneration::I286 {
        return matches!(first, 0 | 10..=13);
    }
    let contributory = |vector| matches!(vector, 0 | 9..=13);
    (contributory(first) && contributory(second))
        || (first == 14 && (contributory(second) || second == 14))
}

/// Completed event deliveries in a row with no instruction retired between
/// them, at which the run returns `Exit::BudgetExhausted`. Delivery costs no
/// unit (`Vcpu::run_steps`), so a handler whose first instruction faults
/// enters delivery again with nothing retired. It does so until the frames
/// exhaust the stack: an 8086-family stack wraps within its segment and never
/// does (8086 Family UM §2.3 p. 2-12), and a 32-bit stack takes up to
/// 2^32 / 12 frames. The return is at a boundary and resumable; the next run
/// counts afresh. The value is an engine choice (ASSUMED): one boundary takes
/// at most an owned continuation, an NMI, an IRQ and the single-step trap
/// after a unit (spec.md §17.10), so a guest that retires instructions never
/// reaches it. Only the delivery path reads or writes the count.
const DELIVERY_STALL_LIMIT: u32 = 1024;

#[derive(Clone, Copy, Default)]
struct MachineBoundary {
    task_pending: bool,
    single_step: bool,
    one_step: bool,
    service: bool,
    pending_io: bool,
    instruction_retry: bool,
    // After a unit: a queued A20 change, or NMI blocking that an IRET may end
    // (spec.md §17.10). The cold epilogue reads which. The twenty-bit boundary
    // leaves out the NMI term: its models nest NMIs and are never blocked (the
    // assertion in nmi.rs). The slow arm gates its IRET checks
    // (`nmi_after_progress`) on this field, as the fast arm gates its epilogue.
    cold_epilogue: bool,
    irq_pending: bool,
}

// Qualified while MachineContext is unchanged. Keeping the Arc prevents
// address reuse; every probe still follows an authoritative VM cache lookup.
// Edited entry, context-changing helpers/events, faults and reset clear it.
#[derive(Default)]
struct DirectProofs {
    slots: [Option<(Arc<IrBlock>, BlockWalk)>; 8],
}
impl DirectProofs {
    fn slot(ip: u64) -> usize {
        ((ip ^ (ip >> 4)) & 7) as usize
    }
    fn get(&self, ir: &Arc<IrBlock>, ip: u64) -> Option<BlockWalk> {
        self.slots[Self::slot(ip)]
            .as_ref()
            .filter(|(known, _)| Arc::ptr_eq(known, ir))
            .map(|(_, walk)| *walk)
    }
    fn insert(&mut self, ir: &Arc<IrBlock>, ip: u64) -> BlockWalk {
        let walk = BlockWalk::for_ir::<false>(ir);
        self.slots[Self::slot(ip)] = Some((Arc::clone(ir), walk));
        walk
    }
    fn clear(&mut self) {
        self.slots = Default::default();
    }
}

// SDM Vol 2B §4.3 REPE/REPZ p. 4-565 and REPNE/REPNZ p. 4-567:
// comparison faults restore the flags from before the repeated instruction,
// while the completed count/indices remain committed across host step slices.
struct RepFlags {
    ir: Arc<IrBlock>,
    stamp: LegacyContextStamp,
    cursor: [u64; 4],
    flags: crate::state::Flags,
    system_flags: u64,
    origin: crate::state::Flags,
    retired: u64,
}

impl RepFlags {
    fn cursor(cpu: &CpuState) -> [u64; 4] {
        [cpu.gpr[1], cpu.gpr[6], cpu.gpr[7], cpu.gpr[0]]
    }
    fn matches(&self, cpu: &CpuState, model: LegacyCpuModel, retired: u64) -> bool {
        self.retired == retired
            && self.ir.guest_start == cpu.rip
            && self.cursor == Self::cursor(cpu)
            && self.flags == cpu.flags
            && self.system_flags == cpu.rflags_system
            && self.stamp.matches(cpu, model)
    }
}

// 8086 Family UM §2.7 p. 2-42 (NEC V20/V30 UM §6 p. 6-4 for three prefixes):
// an interrupt taken inside a repeated string saves the restart IP of its kept
// prefixes. The owner/restart pair of the string at CS:IP (`rep_restart20`) is
// keyed on that instruction and on the inputs its next element reads, at their
// read widths, as the U3 ENTER identity contract keys an attempt (xj-50w.15): a
// host change of the instruction bytes there, or a host edit of one of those
// inputs, between runs starts a new operation at that address, which an
// interrupt then names by its first prefix; an edit of a register the string
// does not read, or a write of the same bytes, keeps the pair. Only the host
// edits between runs, so when a run returns with a pair pending its string's
// bytes (prefixes and opcode, as lifted), inputs and registers are recorded,
// and the next run with a budget compares them with guest memory at CS:IP and
// the registers; the dispatcher's per-block code is unchanged.
//
// The inputs (UM §2.7 Table 2-13 p. 2-40, pp. 2-42–43): CX and DF; SI and the
// source segment for MOVS, LODS and CMPS; DI and ES for MOVS, STOS, CMPS and
// SCAS; AL or AX as the STOS source and SCAS value. ZF "does not need to be
// initialized" (p. 2-42), so it is not an input.
#[derive(Clone, Copy, Default)]
struct RepInputs20 {
    registers: [u16; 9],
    // The registers the string reads (its description) and its bytes from
    // the first prefix through the opcode.
    mask: [u16; 9],
    bytes: [u8; 15],
    len: u8,
}
impl RepInputs20 {
    // CX, DF, SI, DI, AX and the DS, ES, SS and CS selectors.
    fn registers(cpu: &CpuState) -> [u16; 9] {
        [
            cpu.gpr[1] as u16,
            u16::from(cpu.flags.df),
            cpu.gpr[6] as u16,
            cpu.gpr[7] as u16,
            cpu.gpr[0] as u16,
            cpu.ds,
            cpu.es,
            cpu.ss,
            cpu.cs,
        ]
    }
    // The registers `string` reads. Twenty-bit port strings lower to
    // `IrOp::PortString` and never own a pair.
    fn mask(string: &crate::legacy::LegacyOp) -> [u16; 9] {
        use crate::ir::StrOp::*;
        let crate::legacy::LegacyOp::WrappedString {
            kind, elem, source, ..
        } = string
        else {
            unreachable!("a REP owner/restart pair belongs to a WrappedString")
        };
        let source_read = matches!(kind, Movs | Lods | Cmps);
        let destination_read = matches!(kind, Movs | Stos | Cmps | Scas);
        let value = match kind {
            Stos | Scas if *elem == 1 => 0xff,
            Stos | Scas => 0xffff,
            _ => 0,
        };
        let on = |read: bool| if read { 0xffff } else { 0 };
        let segment = |id: SegmentId| on(source_read && *source == id);
        [
            0xffff,
            0xffff,
            on(source_read),
            on(destination_read),
            value,
            segment(SegmentId::Ds),
            on(destination_read) | segment(SegmentId::Es),
            segment(SegmentId::Ss),
            segment(SegmentId::Cs),
        ]
    }
    // The code byte at CS:offset, read as the twenty-bit fetch reads it.
    fn code_byte(vm: &Vm, cs: u16, offset: u16) -> Option<u8> {
        let mut byte = [0];
        let physical = ((u64::from(cs) << 4) + u64::from(offset)) & 0xfffff;
        vm.mem.fetch_code(physical, &mut byte).ok().map(|_| byte[0])
    }
    // Record the pair's string at run exit: its block's string, with the
    // bytes it was lifted from (`LegacyOp::WrappedString::code`), not a read
    // of guest memory. Pending code writes are applied to the cache first
    // (`handle_smc`). A write after the lift whose notice this drain sees has
    // removed the block, so the pair is dropped here. Any other write, one
    // whose notice comes later included, leaves guest memory different from
    // those bytes at the next entry, which drops the pair there; no order of
    // the reads in this function matters. A string the cache no longer holds
    // (a concurrent code write or eviction), or one longer than fifteen bytes,
    // is not recorded, and the caller drops the pair.
    fn record(&mut self, vm: &Vm, cpu: &CpuState, word: &AtomicU64, pair: (u16, u16, u16)) -> bool {
        let (cs, owner, restart) = pair;
        vm.handle_smc(word);
        let physical = ((u64::from(cs) << 4) + u64::from(owner)) & 0xfffff;
        let key = BlockKey::machine_direct(physical, cpu.mxcsr, false);
        let Some(CachedBlock::Interpreted(ir)) = vm.cache.get_for_dispatch(key, false) else {
            return false;
        };
        let string = ir.ops.iter().find_map(|op| match op {
            IrOp::Legacy(
                op @ crate::legacy::LegacyOp::WrappedString {
                    rep, restart_ip, ..
                },
            ) if !matches!(rep, crate::ir::RepKind::None) && *restart_ip == restart => Some(op),
            _ => None,
        });
        let Some(string @ crate::legacy::LegacyOp::WrappedString { code, code_len, .. }) =
            string.filter(|_| ir.guest_start == u64::from(owner))
        else {
            return false;
        };
        if *code_len == 0 {
            return false;
        }
        self.registers = Self::registers(cpu);
        self.mask = Self::mask(string);
        self.bytes = *code;
        self.len = *code_len;
        true
    }
    // Whether the host changed the recorded string or one of its inputs.
    fn edited(&self, vm: &Vm, cpu: &CpuState, cs: u16, owner: u16) -> bool {
        let registers = Self::registers(cpu);
        (0..9).any(|i| (registers[i] ^ self.registers[i]) & self.mask[i] != 0)
            || (0..self.len).any(|i| {
                Self::code_byte(vm, cs, owner.wrapping_add(u16::from(i)))
                    != Some(self.bytes[usize::from(i)])
            })
    }
}

const HANDOFF_DEBUG: u8 = 1;
const HANDOFF_NUMERIC: u8 = 2;

pub(super) struct Execution {
    owner_id: u64,
    context_epoch: u64,
    context_id: u64,
    stamp: Option<LegacyContextStamp>,
    validated: Option<(LegacyContextStamp, MachineContext, u64)>,
    // The validated stamp covers every selector and the effective A20 line.
    validated_direct_bus: bool,
    contexts: HashMap<LegacyContextStamp, RetainedContext>,
    blocks: HashMap<LegacyBlockKey, Block>,
    instruction_retry: Option<InstructionRetry>,
    rep_flags: Option<Box<RepFlags>>,
    rep_restart20: Option<(u16, u16, u16)>,
    rep_inputs20: RepInputs20,
    /// NEC V20/V30 8080-mode hand-back state (decision-23 C4a): an ED
    /// escape at CS:IP, an 8080 refusal at CS:IP with its opcode, and the
    /// reason MD became 0 in the current call. Valid only while CS:IP match.
    pub(super) nec_escape: Option<(u16, u16)>,
    pub(super) nec_refusal: Option<(u16, u16, u8)>,
    pub(super) nec_entry: Option<crate::I8080Entry>,
    task_switch: crate::interp::legacy::tasks::SwitchState,
    #[cfg(test)]
    lifts: u64,
    // Accepted event delivery can itself hand off an MMIO transfer. Its retry
    // must complete before executing the interrupted instruction (decision-16 §3).
    event_retry: Option<PendingEvent>,
    // Port/numeric handoffs precede the post-instruction trap. External
    // routing shares the service load; Internal does not add a boundary check.
    handoff_service: u8,
    shutdown: bool,
    pub(super) pending_a20_enabled: Option<bool>,
    direct_proofs: DirectProofs,
    smc_pages: Vec<u64>,
    last_direct_fetch: Option<u64>,
    // `Vcpu::delivery_stalled`: the retired count at the last completed
    // delivery, and the deliveries completed since it last moved.
    stall_retired: u64,
    stall_count: u32,
    paging: Option<Box<paging::State>>,
    flat32: flat32::State,
    event_owner: Option<crate::interp::legacy::tasks::Owner>,
}
impl Execution {
    pub(super) fn new() -> Self {
        static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
        let owner_id = NEXT_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .expect("legacy owner identities exhausted");
        Self {
            owner_id,
            context_epoch: 0,
            context_id: 0,
            stamp: None,
            validated: None,
            validated_direct_bus: false,
            contexts: HashMap::new(),
            blocks: HashMap::new(),
            instruction_retry: None,
            rep_flags: None,
            rep_restart20: None,
            rep_inputs20: RepInputs20::default(),
            nec_escape: None,
            nec_refusal: None,
            nec_entry: None,
            task_switch: Default::default(),
            #[cfg(test)]
            lifts: 0,
            event_retry: None,
            event_owner: None,
            handoff_service: 0,
            shutdown: false,
            pending_a20_enabled: None,
            direct_proofs: DirectProofs::default(),
            smc_pages: Vec::new(),
            last_direct_fetch: None,
            stall_retired: 0,
            stall_count: 0,
            paging: None,
            flat32: flat32::State::default(),
        }
    }
    // Whether deliveries copy the registers they begin with for the delivery
    // log (`DeliveryLogOptions::registers`; spec.md §17.10, Delivery log).
    // A delivery already under way when this changes took no copy, and
    // records none.
    pub(super) fn set_record_regs(&mut self, record: bool) {
        self.task_switch.record_regs = record;
        self.task_switch.regs = None;
    }
    pub(super) fn reset_execution_state(&mut self) {
        self.event_retry = None;
        self.event_owner = None;
        self.rep_flags = None;
        self.rep_restart20 = None;
        self.nec_escape = None;
        self.nec_refusal = None;
        self.nec_entry = None;
        self.task_switch.reset();
        self.handoff_service &= !HANDOFF_DEBUG;
        self.shutdown = false;
        self.instruction_retry = None;
        self.direct_proofs.clear();
        self.smc_pages.clear();
        self.last_direct_fetch = None;
        self.stall_count = 0;
        if let Some(paging) = &mut self.paging {
            paging.reset();
        }
    }
    #[inline(always)]
    fn validated_context(&self, cpu: &CpuState, model: LegacyCpuModel) -> Option<MachineContext> {
        // A retained namespace can change while execution returns to the same CPU inputs.
        let (stamp, context, epoch) = self.validated.as_ref()?;
        (*epoch == self.context_epoch && stamp.matches(cpu, model)).then_some(*context)
    }
    pub(super) fn is_shutdown(&self) -> bool {
        self.shutdown
    }
    pub(super) fn task_switch_pending(&self) -> bool {
        self.task_switch.context().is_some()
    }
    pub(super) fn has_continuation(&self) -> bool {
        self.instruction_retry.is_some()
            || self.event_retry.is_some()
            || self.task_switch.context().is_some()
    }
    pub(super) fn debug_handoff_pending(&self) -> bool {
        self.handoff_service & HANDOFF_DEBUG != 0
    }
    #[cold]
    #[inline(never)]
    fn invalidate_translations(&mut self) {
        self.invalidate_paging();
        self.direct_proofs.clear();
        self.blocks.clear();
        self.flat32 = flat32::State::default();
        self.contexts.clear();
        self.stamp = None;
        self.validated = None;
    }
    #[cold]
    #[inline(never)]
    pub(super) fn invalidate_paging(&mut self) {
        if let Some(paging) = &mut self.paging {
            paging.invalidate();
        }
    }
    #[cfg(test)]
    fn sync(&mut self, cpu: &CpuState, model: LegacyCpuModel, force: bool) {
        self.sync_class(cpu, model, force, crate::SegmentClass::Checked);
    }
    fn sync_class(
        &mut self,
        cpu: &CpuState,
        model: LegacyCpuModel,
        force: bool,
        class: crate::SegmentClass,
    ) {
        let stamp = LegacyContextStamp::new(cpu, model);
        if force || self.stamp != Some(stamp) {
            self.context_epoch = if let Some(next) = self.context_epoch.checked_add(1) {
                next
            } else {
                self.invalidate_translations();
                0
            };
            // Decision-18 §3 guard dimensions remain complete. Stable identities
            // implement the accepted transition-cost contract: returning to an
            // identical context reuses its IR instead of discarding other modes.
            // Fetch still revalidates bytes, mappings and policy on every entry.
            self.context_id = if class == crate::SegmentClass::Direct {
                // Direct states do not consume decision-20's Checked LRU slots.
                u64::MAX
            } else if let Some(context) = self.contexts.get_mut(&stamp) {
                context.last_used = self.context_epoch;
                context.id
            } else {
                let id = if self.contexts.len() == MAX_RETAINED_CONTEXTS {
                    let oldest = *self
                        .contexts
                        .iter()
                        .min_by_key(|(_, context)| context.last_used)
                        .unwrap()
                        .0;
                    let evicted = self.contexts.remove(&oldest).unwrap();
                    self.blocks.retain(|key, _| key.context_id != evicted.id);
                    self.flat32.evict_context(evicted.id);
                    evicted.id
                } else {
                    self.contexts.len() as u64
                };
                self.contexts.insert(
                    stamp,
                    RetainedContext {
                        id,
                        last_used: self.context_epoch,
                    },
                );
                id
            };
            self.stamp = Some(stamp);
        }
    }
    fn resolve<const KIND: u8>(
        &mut self,
        vm: &Vm,
        cpu: &CpuState,
        model: LegacyCpuModel,
        context: MachineContext,
        _exit_word: &AtomicU64,
    ) -> Result<(Arc<IrBlock>, BlockWalk), Exit> {
        if self.task_switch.context().is_some() {
            if let Some(retry) = &self.instruction_retry {
                return Ok((retry.ir.clone(), retry.walk));
            }
        }
        if context.segment_class == crate::SegmentClass::Flat32 {
            return self.resolve_flat32(vm, cpu, model, context, _exit_word);
        }
        let linear_fetch = crate::legacy::access(
            cpu,
            context.generation,
            SegmentId::Cs,
            cpu.rip,
            1,
            crate::AccessKind::Execute,
        )? as u32;
        let key = LegacyBlockKey {
            owner_id: self.owner_id,
            context_id: self.context_id,
            model_semantics_id: model.model_semantics_id,
            decode_bits: context.default_operand_bits,
            stack_bits: context.stack_address_bits,
            regime: context.regime,
            decode_ip: cpu.rip as u32,
            linear_fetch,
            mxcsr_mode: cpu.mxcsr & crate::jit_abi::fp::MODE_MASK,
        };
        // Decision-16 §3: finish the already decoded instruction on a device
        // retry. A partial CALL frame can overwrite the CALL's own bytes;
        // its IR must survive eviction and reuse of the translation's context ID.
        if let Some(retry) = &self.instruction_retry {
            if retry.stamp == LegacyContextStamp::new(cpu, model)
                && retry.ip == cpu.rip
                && retry.mxcsr_mode == key.mxcsr_mode
            {
                return Ok((retry.ir.clone(), retry.walk));
            }
        }

        // One-instruction blocks bound fault visibility and SMC. Re-fetch each
        // dependent byte, even on cache hits: permissions, policy and physical
        // aliases are checked before executing that instruction, without page
        // version/TLB assumptions reserved for the paging phase.
        let bytes = fetch(vm, cpu, context)?;
        if let Some(block) = self.blocks.get(&key) {
            if block.bytes == bytes {
                return Ok((block.ir.clone(), block.walk));
            }
        }
        let ir = crate::lift::legacy::instruction_for_profile(
            &bytes,
            cpu.rip as u32,
            context,
            vm.machine_model_profile(context.generation),
        )
        .map_err(|error| match error {
            crate::lift::LiftError::Unsupported { addr, bytes, len } => {
                Exit::unknown_instruction(addr, bytes, len)
            }
            crate::lift::LiftError::DecodeFault { addr } => {
                Exit::unmapped_memory(addr, crate::AccessKind::Execute)
            }
        })?;
        let ir = Arc::new(ir);
        let walk = if KIND >= MACHINE_DISPATCH {
            BlockWalk::for_ir::<true>(&ir)
        } else {
            BlockWalk::for_ir::<false>(&ir)
        };
        #[cfg(test)]
        {
            self.lifts += 1;
        }
        self.blocks.insert(
            key,
            Block {
                ir: ir.clone(),
                walk,
                bytes,
            },
        );
        Ok((ir, walk))
    }
    #[cold]
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn execute_protected_block<const ROUTED: bool, const ORDER: u8>(
        &mut self,
        ir: &Arc<IrBlock>,
        cpu: &mut CpuState,
        memory: &crate::memory::Memory,
        scratch: &mut impl crate::interp::ScratchBuffer,
        model: LegacyCpuModel,
        retired: u64,
    ) -> crate::interp::legacy::Progress {
        let comparing = ir.ops.iter().any(|op| {
            matches!(
                op,
                IrOp::Legacy(crate::legacy::LegacyOp::String {
                    kind: crate::ir::StrOp::Cmps | crate::ir::StrOp::Scas,
                    rep: crate::ir::RepKind::Repe | crate::ir::RepKind::Repne,
                    ..
                })
            )
        });
        if comparing {
            let continuing = self.rep_flags.as_ref().is_some_and(|saved| {
                Arc::ptr_eq(&saved.ir, ir) && saved.matches(cpu, model, retired)
            });
            if !continuing {
                self.rep_flags = Some(Box::new(RepFlags {
                    ir: Arc::clone(ir),
                    stamp: LegacyContextStamp::new(cpu, model),
                    cursor: RepFlags::cursor(cpu),
                    flags: cpu.flags,
                    system_flags: cpu.rflags_system,
                    origin: cpu.flags,
                    retired,
                }));
            }
        } else {
            self.rep_flags = None;
        }
        let progress = if ir
            .legacy_context
            .is_some_and(|c| c.regime == MachineRegime::Virtual8086)
        {
            crate::interp::legacy::virtual8086::execute_block::<ROUTED, ORDER>(
                ir,
                cpu,
                memory,
                scratch,
                &mut self.task_switch,
            )
        } else {
            crate::interp::legacy::execute_checked::<ROUTED, ORDER>(
                ir,
                cpu,
                memory,
                scratch,
                &mut self.task_switch,
            )
        };
        if let Some(saved) = self.rep_flags.as_mut() {
            let fault = progress.outcome.as_ref().is_some_and(|outcome| {
                outcome.as_ref().is_err_and(|exit| {
                    matches!(exit, Exit::UnmappedMemory { .. })
                        || exit
                            .fault_metadata()
                            .is_some_and(|m| m.class() == crate::fault::FaultClass::Fault)
                })
            });
            if fault {
                cpu.flags = saved.origin;
                self.rep_flags = None;
            } else if progress.retired != 0 || cpu.rip != ir.guest_start {
                self.rep_flags = None;
            } else {
                saved.cursor = RepFlags::cursor(cpu);
                saved.flags = cpu.flags;
                saved.system_flags = cpu.rflags_system;
            }
        }
        progress
    }
}

/// The outcome of [`Vcpu::split_retry`].
enum SplitRetry {
    /// The rest of the block completed on its re-execution.
    Done { mode_change: bool },
    /// The exit to handle: the first attempt's, or the re-execution's.
    Exit(Exit),
}

impl Vcpu {
    /// An interpreted application block ended with the containment miss `exit`. If
    /// its operand lies in adjacent RAM/ROM regions, the rest of the block is
    /// re-executed (`interp::split_application`) with the accounting of an
    /// interpreted block; otherwise `exit` is returned unchanged.
    #[cold]
    #[inline(never)]
    fn split_retry<const ROUTED: bool, const STEPPED: bool, const STOP: bool>(
        &mut self,
        vm: &Vm,
        ir: &IrBlock,
        exit: Exit,
        code_base: u64,
        exit_word: &AtomicU64,
        steps: &mut crate::interp::StepContext,
    ) -> SplitRetry {
        let mut info = crate::interp::RetireInfo {
            code_base,
            consistency: vm.consistency,
            sti_shadow: self.sti_shadow,
            ss_shadow: self.ss_shadow & SS_HOLD != 0,
            exit_request: Some(exit_word),
            trace: vm.lockstep_trace.as_deref(),
            ..Default::default()
        };
        let result = match crate::interp::split_application::<ROUTED, STEPPED, STOP>(
            exit,
            ir,
            &mut self.cpu,
            &vm.mem,
            &mut self.interp_scratch,
            &mut info,
            steps,
        ) {
            Ok(result) => result,
            Err(exit) => return SplitRetry::Exit(exit),
        };
        self.retired += info.retired;
        self.executed += info.retired;
        self.sti_shadow = info.sti_shadow;
        self.ss_shadow = u8::from(info.ss_shadow);
        match result {
            crate::interp::BlockResult::Continue => SplitRetry::Done { mode_change: false },
            crate::interp::BlockResult::ModeChange => SplitRetry::Done { mode_change: true },
            crate::interp::BlockResult::Exit(exit) => SplitRetry::Exit(exit.into_exit()),
        }
    }

    /// The Real16 alias dispatcher's containment miss: [`Self::split_retry`], then
    /// the dispatcher's handling of the exit that results. `None` continues the
    /// dispatch loop; `Some` is the stop to return.
    #[cold]
    #[inline(never)]
    fn split_alias<const ROUTED: bool, const STEPPED: bool, const STOP: bool>(
        &mut self,
        vm: &Vm,
        ir: &IrBlock,
        exit: Exit,
        exit_word: &AtomicU64,
        steps: &mut crate::interp::StepContext,
    ) -> Option<StepStop> {
        let code_base = u64::from(self.cpu.cs) << 4;
        let exit = match self
            .split_retry::<ROUTED, STEPPED, STOP>(vm, ir, exit, code_base, exit_word, steps)
        {
            SplitRetry::Done { .. } => return None,
            SplitRetry::Exit(exit) => exit,
        };
        Some(StepStop::Exit(match exit {
            Exit::Syscall => self.syscall_interpreted(ir),
            Exit::Interrupted => {
                let path = if !STEPPED || steps.string_interrupted {
                    crate::FaultPath::Helper {
                        family: crate::FaultHelper::String,
                        caller: crate::HelperCaller::Interpreter,
                    }
                } else {
                    crate::FaultPath::Interpreter
                };
                self.interrupted(path, false)
            }
            Exit::Exception { addr, vector, .. } => {
                match crate::interp::deliver_interrupt(
                    &mut self.cpu,
                    &vm.mem,
                    addr,
                    vector,
                    addr,
                    vm.consistency,
                )
                .into_step()
                {
                    StepResult::Continue => {
                        self.sti_shadow = false;
                        self.ss_shadow = 0;
                        return None;
                    }
                    StepResult::Exit(exit) => exit,
                }
            }
            exit => exit,
        }))
    }

    /// The application dispatcher's containment miss: [`Self::split_retry`], then
    /// the dispatcher's qualification of the exit that results. `Ok` reports
    /// whether the completed block changed the MXCSR mode.
    #[cold]
    #[inline(never)]
    fn split_application<const ROUTED: bool>(
        &mut self,
        vm: &Vm,
        ir: &IrBlock,
        exit: Exit,
        code_base: u64,
        exit_word: &AtomicU64,
    ) -> Result<bool, Exit> {
        match self.split_retry::<ROUTED, false, false>(
            vm,
            ir,
            exit,
            code_base,
            exit_word,
            &mut crate::interp::StepContext::default(),
        ) {
            SplitRetry::Done { mode_change } => Ok(mode_change),
            SplitRetry::Exit(Exit::Syscall) => Err(self.syscall_interpreted(ir)),
            SplitRetry::Exit(Exit::Interrupted) => Err(self.interrupted(
                crate::FaultPath::Helper {
                    family: crate::FaultHelper::String,
                    caller: crate::HelperCaller::Interpreter,
                },
                false,
            )),
            SplitRetry::Exit(exit @ Exit::Exception { .. }) => {
                Err(crate::interp::qualify_lifted_exit(exit, ir))
            }
            SplitRetry::Exit(exit) => Err(exit),
        }
    }
}

fn retry_instruction(ir: &Arc<IrBlock>, ip: u64) -> Arc<IrBlock> {
    let start = ir
        .ops
        .iter()
        .position(|op| matches!(op, IrOp::InsnStart { guest_addr } if *guest_addr == ip))
        .unwrap();
    let end = ir.ops[start + 1..]
        .iter()
        .position(|op| matches!(op, IrOp::InsnStart { .. }))
        .map_or(ir.ops.len(), |offset| start + 1 + offset);
    let next = match ir.ops.get(end) {
        Some(IrOp::InsnStart { guest_addr }) => *guest_addr,
        _ => ir.guest_start + u64::from(ir.guest_len),
    };
    let mut ops = ir.ops[start..end].to_vec();
    if matches!(
        ir.ops.last(),
        Some(IrOp::Legacy(crate::legacy::LegacyOp::SharedWalk))
    ) && !matches!(
        ops.last(),
        Some(IrOp::Legacy(crate::legacy::LegacyOp::SharedWalk))
    ) {
        ops.push(IrOp::Legacy(crate::legacy::LegacyOp::SharedWalk));
    }
    Arc::new(IrBlock {
        mode: ir.mode,
        guest_start: ip,
        guest_len: (next - ip) as u32,
        icount: 1,
        temp_count: ir.temp_count,
        ops,
        legacy_context: ir.legacy_context,
    })
}

fn raw_direct(cpu: &CpuState) -> bool {
    cpu.legacy.a20_enabled != 0
        || [cpu.cs, cpu.ds, cpu.es, cpu.ss, cpu.fs, cpu.gs]
            .iter()
            .all(|selector| *selector <= 0xf000)
}

fn shared_instruction(insn: &iced_x86::Instruction) -> bool {
    use iced_x86::Mnemonic::*;
    // The compatibility decoder faults beyond ten bytes; I386 allows fifteen
    // (80286 PRM App. C item 10 p. C-2; 80386 PRM §14.7 item 6 p. 14-6).
    if insn.len() > 10 {
        return false;
    }
    // The alias wraps sequential IP; Machine retains EIP past FFFFh
    // (80386 PRM §14.7 item 8). Specialize this edge at lift time.
    if insn.ip() + insn.len() as u64 == 0x10000 {
        return false;
    }
    match insn.mnemonic() {
        Mov => !(0..insn.op_count()).any(|i| {
            insn.op_register(i).is_cr()
                || insn.op_register(i).is_dr()
                || insn.op_register(i).is_tr()
        }),
        Push | Pop | Call | Ret | Retf | Jmp | Pushf | Pushfd | Popf | Popfd | Pusha | Pushad
        | Popa | Popad | Iret | Iretd | Enter | Leave | Les | Lds | Lss | Lfs | Lgs | Bound
        | Xlatb | Loop | Loope | Loopne | Jcxz | Jecxz => true,
        Movsb | Movsw | Movsd | Stosb | Stosw | Stosd | Lodsb | Lodsw | Lodsd | Cmpsb | Cmpsw
        | Cmpsd | Scasb | Scasw | Scasd => {
            // The frozen alias has no FS/GS segment personality and uses #GP
            // for SS limits; Machine I386 has FS/GS and #SS (PRM §14.6 Table 14-1).
            insn.is_string_instruction()
                && !matches!(
                    insn.memory_segment(),
                    iced_x86::Register::FS | iced_x86::Register::GS | iced_x86::Register::SS
                )
        }
        Lea | Add | Adc | Sub | Sbb | And | Or | Xor | Cmp | Test | Inc | Dec | Neg | Not
        | Xchg | Shl | Sal | Shr | Sar | Rol | Ror | Rcl | Rcr | Shld | Shrd | Mul | Imul | Div
        | Idiv | Movzx | Movsx | Cbw | Cwde | Cwd | Cdq | Clc | Stc | Cmc | Cld | Std | Lahf
        | Sahf | Nop | Hlt | In | Out | Cli | Sti => true,
        Jb | Jbe | Je | Jg | Jge | Jl | Jle | Jae | Ja | Jne | Jno | Jnp | Jns | Jo | Jp | Js => {
            // The frozen alias's conditional-branch IP wrap is outside the
            // unified stack/control families; Machine checks the full EIP.
            insn.code().op_code().operand_size() != 32
        }
        _ => false,
    }
}

#[inline(always)]
fn resolve_direct(
    vm: &Vm,
    cpu: &CpuState,
    context: MachineContext,
    exit_word: &AtomicU64,
    physical: u64,
    proofs: &mut DirectProofs,
) -> Result<(Arc<IrBlock>, BlockWalk), Exit> {
    let key = BlockKey::machine_direct(physical, cpu.mxcsr, cpu.legacy.a20_enabled != 0);
    if let Some(CachedBlock::Interpreted(ir)) = vm.cache.get_for_dispatch(key, vm.time_enabled) {
        if ir.guest_start == cpu.rip {
            if let Some(walk) = proofs.get(&ir, cpu.rip) {
                return Ok((ir, walk));
            }
            if ir.legacy_context == Some(context) {
                let walk = proofs.insert(&ir, cpu.rip);
                return Ok((ir, walk));
            }
        }
    }
    let ir = resolve_direct_uncached(vm, cpu, context, exit_word, physical, key)?;
    let walk = proofs.insert(&ir, cpu.rip);
    Ok((ir, walk))
}

#[cold]
#[inline(never)]
fn resolve_direct_uncached(
    vm: &Vm,
    cpu: &CpuState,
    context: MachineContext,
    exit_word: &AtomicU64,
    physical: u64,
    key: BlockKey,
) -> Result<Arc<IrBlock>, Exit> {
    if cpu.rip > 0xffff {
        // 80386 PRM §14.7 item 8: EIP outside CS faults before fetching.
        crate::legacy::access(
            cpu,
            context.generation,
            SegmentId::Cs,
            cpu.rip,
            1,
            crate::AccessKind::Execute,
        )?;
    }
    loop {
        let source = crate::memory::LiftSource::new(&vm.mem);
        let generation = vm.cache.publication_generation();
        let permit = vm.mem.fetch_permit();
        let mut failure = None;
        let lifted = crate::lift::lift_block_context(
            &vm.mem,
            FetchAddr {
                pa: physical,
                ip: cpu.rip,
            },
            CpuMode::Real16,
            &mut failure,
            &source,
            Some(context),
        );
        let mut ir = if let Ok(mut ir) = lifted {
            let mut code = vec![0; ir.guest_len as usize];
            let copied = vm.mem.fetch_code(physical, &mut code).unwrap_or(0);
            code.truncate(copied);
            let mut decoder = Decoder::with_ip(
                16,
                &code,
                cpu.rip,
                crate::model::decode::SYSTEM_DECODER_OPTIONS,
            );
            let mut count = 0;
            let mut end = cpu.rip;
            let mut special = None;
            let mut strings = Vec::new();
            while decoder.can_decode() {
                let position = decoder.position();
                let insn = decoder.decode();
                let bytes = &code[position..decoder.position()];
                // Match incremental Machine fetch: an unavailable I286 opcode
                // byte wins before a later ESC can reach its MSW gate.
                let profile = vm.machine_model_profile(context.generation);
                let isa = profile.isa;
                // The catalogue's length limit (xj-50w.13; I286 PRM App. C
                // item 10 p. C-2): a fetch holds at most that many bytes, so
                // the reference sees what the Checked fetch sees, also when an
                // unavailable opcode lies past the limit.
                let bytes = &bytes[..bytes
                    .len()
                    .min(profile.max_insn_len.map_or(15, usize::from))];
                let reference_bytes = if isa.legacy_segments_only() && isa.unavailable_prefix(bytes)
                {
                    let end = (1..=bytes.len())
                        .find(|&end| isa.unavailable_prefix(&bytes[..end]))
                        .expect("unavailable opcode byte");
                    &bytes[..end]
                } else {
                    bytes
                };
                let reference = crate::lift::legacy::instruction_for_profile(
                    reference_bytes,
                    insn.ip() as u32,
                    context,
                    profile,
                );
                let allowed = !insn.is_invalid()
                    && shared_instruction(&insn)
                    && reference.as_ref().is_ok_and(|r| {
                        !r.ops.iter().any(|op| {
                            matches!(
                                op,
                                IrOp::Trap { .. }
                                    | IrOp::Legacy(crate::legacy::LegacyOp::Refuse { .. })
                            )
                        })
                    });
                if !allowed {
                    special = if insn.is_invalid() {
                        None
                    } else {
                        reference.ok()
                    };
                    break;
                }
                if let Some(op) = reference.as_ref().ok().and_then(|ir| {
                    ir.ops.iter().find(|op| {
                        matches!(op, IrOp::Legacy(crate::legacy::LegacyOp::String { .. }))
                    })
                }) {
                    strings.push((insn.ip(), op.clone()));
                }
                count += 1;
                end = insn.ip() + insn.len() as u64;
            }
            if count == 0 {
                if let Some(ir) = special {
                    ir
                } else {
                    let bytes = fetch(vm, cpu, context)?;
                    crate::lift::legacy::instruction_for_profile(
                        &bytes,
                        cpu.rip as u32,
                        context,
                        vm.machine_model_profile(context.generation),
                    )
                    .map_err(|error| direct_refusal(cpu, error))?
                }
            } else {
                if end < ir.guest_start + u64::from(ir.guest_len) {
                    let stop = ir
                        .ops
                        .iter()
                        .position(
                            |op| matches!(op, IrOp::InsnStart { guest_addr } if *guest_addr == end),
                        )
                        .unwrap();
                    ir.ops.truncate(stop);
                }
                ir.guest_len = (end - cpu.rip) as u32;
                ir.icount = count;
                if !strings.is_empty() {
                    // Preserve the width-aware Machine string op. The generic
                    // walk uses its selector bases with the shared string engine;
                    // the frozen alias keeps its existing string IR.
                    let mut ops = Vec::with_capacity(ir.ops.len());
                    let mut string = false;
                    for op in ir.ops {
                        if let IrOp::InsnStart { guest_addr } = op {
                            ops.push(op);
                            string = false;
                            if let Some((_, body)) =
                                strings.iter().find(|(ip, _)| *ip == guest_addr)
                            {
                                ops.push(body.clone());
                                string = true;
                            }
                        } else if !string {
                            ops.push(op);
                        }
                    }
                    ir.ops = ops;
                }
                // Opaque instructions retain the instruction helper walk's fault
                // evidence and context handoff until the walks are unified.
                if !ir.ops.iter().any(|op| {
                    matches!(op,
                    IrOp::Legacy(op) if crate::lift::legacy::shared_helper(op))
                }) {
                    ir.ops
                        .push(IrOp::Legacy(crate::legacy::LegacyOp::SharedWalk));
                }
                ir
            }
        } else {
            let bytes = fetch(vm, cpu, context)?;
            crate::lift::legacy::instruction_for_profile(
                &bytes,
                cpu.rip as u32,
                context,
                vm.machine_model_profile(context.generation),
            )
            .map_err(|error| direct_refusal(cpu, error))?
        };
        ir.legacy_context = Some(context);
        if !source.validate() || permit.as_ref().is_some_and(|p| !p.is_current()) {
            continue;
        }
        if !source.publishable() {
            let CachedBlock::Interpreted(ir) = source.uncached(ir) else {
                unreachable!()
            };
            return Ok(ir);
        }
        let ir = Arc::new(ir);
        #[cfg(test)]
        vm.after_source_read();
        vm.handle_smc(exit_word);
        if vm.cache.insert_checked(
            key,
            CachedBlock::Interpreted(ir.clone()),
            vec![(physical, ir.guest_len)],
            |spans| {
                if vm.cache.publication_generation() != generation
                    || permit.as_ref().is_some_and(|p| !p.is_current())
                {
                    return false;
                }
                for (start, len) in spans {
                    vm.mem.mark_code(*start, *len);
                }
                true
            },
        ) {
            return Ok(ir);
        }
    }
}

// A Direct lift refusal reports the refused instruction's bytes, as the
// Checked resolver does (`resolve`) and the twenty-bit path does.
#[cold]
fn direct_refusal(cpu: &CpuState, error: crate::lift::LiftError) -> Exit {
    match error {
        crate::lift::LiftError::Unsupported { addr, bytes, len } => {
            Exit::unknown_instruction(addr, bytes, len)
        }
        crate::lift::LiftError::DecodeFault { .. } => {
            Exit::unknown_instruction(cpu.rip, [0; 15], 0)
        }
    }
}

fn fetch(vm: &Vm, cpu: &CpuState, context: MachineContext) -> Result<Vec<u8>, Exit> {
    fetch_watched(vm, cpu, context, cpu.rip, None)
}
fn fetch_watched(
    vm: &Vm,
    cpu: &CpuState,
    context: MachineContext,
    ip: u64,
    source: Option<&crate::memory::LiftSource<'_>>,
) -> Result<Vec<u8>, Exit> {
    // The catalogue's instruction-length limit (ten bytes on the 80286,
    // I286 PRM Appendix B #GP p. B-9; fifteen on the 80386) and formed-address
    // mask (`legacy::physical_mask`), resolved at compile time per model so
    // the per-byte loop carries no model data. A model with no limit (the
    // 8086 family) reads at most fifteen bytes, the decoder's maximum.
    const fn limit(model: crate::CpuModel) -> u64 {
        match model.profile().max_insn_len {
            Some(limit) => limit as u64,
            None => 15,
        }
    }
    use crate::legacy::{bus_model, physical_mask};
    match context.generation {
        CpuGeneration::I8086 => fetch_bytes::<
            { physical_mask(CpuGeneration::I8086) },
            { limit(bus_model(CpuGeneration::I8086)) },
        >(vm, cpu, context, ip, source),
        CpuGeneration::I286 => fetch_bytes::<
            { physical_mask(CpuGeneration::I286) },
            { limit(bus_model(CpuGeneration::I286)) },
        >(vm, cpu, context, ip, source),
        CpuGeneration::I386 => fetch_bytes::<
            { physical_mask(CpuGeneration::I386) },
            { limit(bus_model(CpuGeneration::I386)) },
        >(vm, cpu, context, ip, source),
    }
}

fn fetch_bytes<const MASK: u32, const LIMIT: u64>(
    vm: &Vm,
    cpu: &CpuState,
    context: MachineContext,
    ip: u64,
    source: Option<&crate::memory::LiftSource<'_>>,
) -> Result<Vec<u8>, Exit> {
    let mut bytes = Vec::with_capacity(15);
    for i in 0..LIMIT {
        let linear = crate::legacy::access(
            cpu,
            context.generation,
            SegmentId::Cs,
            ip + i,
            1,
            crate::AccessKind::Execute,
        )?;
        let (addr, _) = crate::legacy::physical_span(cpu, MASK, linear, 1);
        if let Some(policy) = &vm.mem.fetch_policy {
            let verdict = policy.approve(addr >> crate::memory::CODE_PAGE_BITS, &vm.mem);
            if verdict != crate::FetchVerdict::Allow {
                return Err(crate::fault::qualify_legacy_bus(
                    crate::fault::fetch_exit(
                        &vm.mem,
                        FetchAddr { pa: addr, ip },
                        CpuMode::Compat32,
                        addr,
                        Some(crate::lift::FetchFailure::Policy(verdict)),
                    ),
                    linear,
                    addr,
                ));
            }
        }
        if let Some(source) = source {
            source.watch(addr, 1);
        }
        let mut byte = [0];
        if let Err(trap) = vm.mem.fetch_code(addr, &mut byte) {
            return Err(crate::fault::qualify_legacy_bus(
                crate::fault::fetch_exit(
                    &vm.mem,
                    FetchAddr { pa: addr, ip },
                    CpuMode::Compat32,
                    addr,
                    Some(crate::lift::FetchFailure::Memory(trap)),
                ),
                linear,
                addr,
            ));
        }
        if let Some(source) = source {
            source.record(addr, &byte);
        }
        bytes.push(byte[0]);
        if crate::model::decode::unavailable_prefix_for_generation(context.generation, &bytes) {
            break;
        }
        let mut decoder = Decoder::with_ip(
            u32::from(context.default_operand_bits),
            &bytes,
            ip,
            crate::model::decode::SYSTEM_DECODER_OPTIONS,
        );
        let insn = decoder.decode();
        if !insn.is_invalid() {
            break;
        }
        if decoder.last_error() != DecoderError::NoMoreBytes
            && !locked_form_continues(&bytes, context, ip)
        {
            break;
        }
    }
    Ok(bytes)
}

// A LOCK form the 80286 executes (80286 PRM App. C item 11 p. C-2) is
// rejected by the strict decoder; keep fetching while it is still incomplete.
#[cold]
#[inline(never)]
fn locked_form_continues(bytes: &[u8], context: MachineContext, ip: u64) -> bool {
    if !crate::legacy::bus_model(context.generation)
        .profile()
        .quirks
        .lock_data_movement
        || !bytes
            .iter()
            .take_while(|b| matches!(b, 0x26 | 0x2e | 0x36 | 0x3e | 0xf0 | 0xf2 | 0xf3))
            .any(|b| *b == 0xf0)
    {
        return false;
    }
    let mut decoder = Decoder::with_ip(
        u32::from(context.default_operand_bits),
        bytes,
        ip,
        crate::model::decode::SYSTEM_DECODER_OPTIONS | iced_x86::DecoderOptions::NO_INVALID_CHECK,
    );
    decoder.decode().is_invalid() && decoder.last_error() == DecoderError::NoMoreBytes
}

#[inline(always)]
fn resolve_alias<const STEPPED: bool>(
    vm: &Vm,
    stats: &mut crate::time_stats::DispatchCounts,
    exit_word: &AtomicU64,
    sections: &AtomicU64,
    at: FetchAddr,
    mxcsr: u32,
) -> Result<Arc<IrBlock>, Exit> {
    stats.fetch_calls = stats.fetch_calls.saturating_add(1);
    let _time_fetch = vm
        .time_enabled
        .then(|| crate::time_stats::ResolveTimer::enter(stats.fetch_calls));
    let default = STEPPED || mxcsr & crate::jit_abi::fp::MODE_MASK == 0;
    let key = if default {
        BlockKey::new(at.pa, CpuMode::Real16)
    } else {
        BlockKey::with_mxcsr(at.pa, CpuMode::Real16, mxcsr)
    };
    loop {
        stats.lookups = stats.lookups.saturating_add(1);
        if let Some(CachedBlock::Interpreted(ir)) = vm.cache.get_for_dispatch(key, vm.time_enabled)
        {
            return Ok(ir);
        }
        stats.misses = stats.misses.saturating_add(1);
        if vm.time_enabled {
            crate::time_stats::ResolveTimer::miss();
        }
        let result = if default {
            resolve_alias_uncached::<true>(vm, exit_word, sections, at, mxcsr)
        } else {
            resolve_alias_uncached::<false>(vm, exit_word, sections, at, mxcsr)
        };
        match result {
            Ok(ir) => return Ok(ir),
            Err(ResolveFailure::Exit(exit)) => return Err(exit),
            Err(ResolveFailure::Retry) => {}
        }
    }
}
#[cold]
#[inline(never)]
fn resolve_alias_uncached<const DEFAULT: bool>(
    vm: &Vm,
    exit_word: &AtomicU64,
    sections: &AtomicU64,
    at: FetchAddr,
    mxcsr: u32,
) -> Result<Arc<IrBlock>, ResolveFailure> {
    let source = crate::memory::LiftSource::new(&vm.mem);
    resolve_interpreted_uncached::<DEFAULT>(
        vm,
        &source,
        exit_word,
        sections,
        at,
        CpuMode::Real16,
        mxcsr,
    )
    .map(|block| match block {
        CachedBlock::Interpreted(ir) => ir,
        _ => unreachable!(),
    })
}

impl Vcpu {
    // Twenty-bit run entry (xj-50w.15): a REP owner/restart pair is pending.
    #[inline(always)]
    pub(super) fn rep_restart20_pending(&self) -> bool {
        self.legacy_execution
            .as_deref()
            .is_some_and(|execution| execution.rep_restart20.is_some())
    }

    // Twenty-bit run entry (xj-50w.15): drop the pending REP owner/restart pair
    // when the host changed the string's bytes at CS:owner or one of its inputs
    // since the last run returned. The translation cache is not consulted, so
    // an eviction keeps the pair.
    #[cold]
    #[inline(never)]
    pub(super) fn forget_edited_rep_restart20(&mut self, vm: &Vm) {
        let execution = self.legacy_execution.as_mut().unwrap();
        let Some((cs, owner, _)) = execution.rep_restart20 else {
            return;
        };
        if execution.rep_inputs20.edited(vm, &self.cpu, cs, owner) {
            execution.rep_restart20 = None;
        }
    }

    // Twenty-bit run exit (xj-50w.15): a REP owner/restart pair is pending;
    // record its string for the next run's entry. Called before the run's
    // dispatch counts are published, while the VM is still in use.
    #[cold]
    #[inline(never)]
    pub(super) fn record_rep_restart20(&mut self, vm: &Vm) {
        let execution = self.legacy_execution.as_mut().unwrap();
        if let Some(pair) = execution.rep_restart20 {
            if !execution
                .rep_inputs20
                .record(vm, &self.cpu, &self.exit_request.word, pair)
            {
                execution.rep_restart20 = None;
            }
        }
    }

    #[inline(always)]
    pub(super) fn finish_comparison_attempt(&mut self, result: &Result<StepRun, StepRunError>) {
        if self.legacy_execution.as_ref().unwrap().rep_flags.is_some() {
            self.finish_owned_comparison_attempt(result);
        }
    }

    #[cold]
    #[inline(never)]
    fn finish_owned_comparison_attempt(&mut self, result: &Result<StepRun, StepRunError>) {
        let saved = self
            .legacy_execution
            .as_mut()
            .unwrap()
            .rep_flags
            .take()
            .unwrap();
        let continuing = match result {
            Ok(StepRun {
                stop: StepStop::IrqReady,
                ..
            }) => true,
            Ok(StepRun {
                stop: StepStop::Exit(exit),
                ..
            }) => match exit {
                Exit::BudgetExhausted | Exit::Interrupted => true,
                Exit::MmioRead { .. } | Exit::MmioWrite { .. } => {
                    self.cpu.mmio_retry_active && self.cpu.owns_mmio_retry(self.cpu.rip)
                }
                _ => false,
            },
            Err(_) => false,
        };
        // Consume by default: faults, entry errors and future exit kinds cannot
        // retain an abandoned attempt through a reusable translation context.
        if continuing && saved.matches(&self.cpu, self.legacy_model.unwrap(), self.retired) {
            self.legacy_execution.as_mut().unwrap().rep_flags = Some(saved);
        }
    }

    #[cold]
    #[inline(never)]
    fn validate_legacy_debug_state(&self) -> Result<(), Exit> {
        // 80386 PRM §12.2.2 p. 12-4: L0–L3/G0–G3 enable breakpoints;
        // SDM Vol 3B §20.2.4 p. 20-5: GD enables debug-register traps.
        // Apply the MOV DR7 refusal policy to embedder-restored images too.
        if self.legacy_model.unwrap().generation == CpuGeneration::I386
            && self.cpu.debug_test_registers[crate::legacy::DebugTestRegister::Dr7 as usize]
                & 0x20ff
                != 0
        {
            return Err(self.unsupported_legacy(UnsupportedSystemOperation::LegacyDebugBreakpoints));
        }
        Ok(())
    }

    /// Compatibility alias for transfer witnesses; uses normal Machine entry.
    #[doc(hidden)]
    #[cfg(feature = "test-hooks")]
    pub fn run_virtual8086_steps_for_test(
        &mut self,
        vm: &Vm,
        max_steps: u64,
        options: StepOptions,
    ) -> Result<StepRun, StepRunError> {
        self.run_steps(vm, max_steps, options)
    }

    /// Hand an outward exception back to the internal V86 delivery seam.
    #[doc(hidden)]
    #[cfg(feature = "test-hooks")]
    pub fn deliver_virtual8086_fault_for_test(&mut self, vm: &Vm, exit: &Exit) -> Result<(), Exit> {
        let context = crate::legacy::derive_virtual8086_context_for_test(
            self.legacy_model.unwrap(),
            &self.cpu,
        )
        .map_err(|e| {
            self.unsupported_legacy(UnsupportedSystemOperation::InvalidLegacyContext(e))
        })?;
        let source = exception_source(exit).expect("exception witness");
        let ip = exit
            .fault_metadata()
            .map_or(self.cpu.rip, |m| m.saved_rip()) as u32;
        self.deliver_legacy_event::<false, 0>(
            vm,
            context,
            PendingEvent {
                irq_consumed: false,
                software: false,
                after_nmi: false,
                source,
                ip,
                cs: self.cpu.cs,
                origin: self.delivery_origin(self.cpu.cs, ip),
            },
        )
    }

    pub(super) fn legacy_context_for_run(&self) -> Result<MachineContext, Exit> {
        self.validate_legacy_debug_state()?;
        let model = self.legacy_model.expect("legacy model");
        if self.twenty_bit_model() {
            return Ok(crate::lift::i8086::CONTEXT);
        }
        if let Some(context) = self
            .legacy_execution
            .as_ref()
            .and_then(|e| e.task_switch.context_for(&self.cpu))
        {
            return Ok(context);
        }
        if self.cpu.task_debug_trap != 0 {
            return Err(self.unsupported_legacy(UnsupportedSystemOperation::LegacyTaskDebugTrap));
        }
        let context = derive_legacy_context(model, &self.cpu).map_err(|e| {
            Exit::UnsupportedSystemContext {
                addr: self.cpu.rip,
                operation: UnsupportedSystemOperation::InvalidLegacyContext(e),
            }
        })?;
        // I286 PRM §10.4 p. 10-6; 80386 PRM §10.2.3 p. 10-3:
        // reset may fetch through the retained high CS base.
        // System-free models have already returned their permanent context.
        if !matches!(model.generation, CpuGeneration::I286 | CpuGeneration::I386)
            || self.cpu.rip > u64::from(u32::MAX)
        {
            return Err(self.unsupported_legacy(UnsupportedSystemOperation::LegacyEntryFacility));
        }
        Ok(context)
    }

    #[inline(always)]
    pub(super) fn warm_machine_context(&mut self, limit: Option<u64>) -> Option<MachineContext> {
        if limit != Some(0)
            && !self.cpu.mmio_retry_active
            && self.cpu.task_debug_trap == 0
            && self.cpu.rip <= u64::from(u32::MAX)
        {
            let execution = self.legacy_execution.as_mut().unwrap();
            if !execution.shutdown && !execution.has_continuation() {
                if let Some(context) =
                    execution.validated_context(&self.cpu, self.legacy_model.unwrap())
                {
                    if context.generation == CpuGeneration::I386
                        && self.cpu.debug_test_registers
                            [crate::legacy::DebugTestRegister::Dr7 as usize]
                            & 0x20ff
                            != 0
                    {
                        return None;
                    }
                    if context.segment_class == crate::SegmentClass::Direct {
                        execution.task_switch.committed = false;
                        execution.task_switch.completed = false;
                    }
                    return Some(context);
                }
            }
        }
        None
    }

    #[cold]
    #[inline(never)]
    pub(super) fn prepare_machine_entry_cold(
        &mut self,
        limit: Option<u64>,
    ) -> Result<MachineContext, Exit> {
        self.validate_legacy_debug_state()?;
        if limit != Some(0) && self.cpu.mmio_retry_active {
            self.abandon_edited_paged_retry();
        }
        // Decision-24: edited state abandons the entire owned operation before
        // entry validation can reject its deliberately unfinished segment caches.
        // A zero-budget probe leaves continuations and completion inputs intact.
        if limit != Some(0)
            && (self
                .legacy_execution
                .as_ref()
                .unwrap()
                .event_owner
                .as_ref()
                .is_some_and(|owner| !owner.matches(&self.cpu))
                || (self.cpu.mmio_retry_active
                    && self
                        .legacy_execution
                        .as_ref()
                        .unwrap()
                        .instruction_retry
                        .as_ref()
                        .is_some_and(|retry| {
                            !retry.stamp.matches(&self.cpu, self.legacy_model.unwrap())
                                || retry.ip != self.cpu.rip
                                || retry.mxcsr_mode
                                    != self.cpu.mxcsr & crate::jit_abi::fp::MODE_MASK
                        }))
                || self
                    .legacy_execution
                    .as_mut()
                    .unwrap()
                    .task_switch
                    .abandon_if_changed(&self.cpu))
        {
            self.legacy_execution.as_mut().unwrap().instruction_retry = None;
            self.legacy_execution.as_mut().unwrap().task_switch.reset();
            self.abandon_legacy_event();
            self.cpu.clear_mmio_parts();
            self.cpu.pending_mmio = None;
            self.cpu.pending_mmio_write = false;
            self.cpu.mmio_parts_len = 0;
            self.cpu.mmio_parts_rip = 0;
            self.cpu.mmio_retry_active = false;
            self.cpu.mmio_retry_cs = 0;
            self.apply_pending_a20();
        }
        let context = self.legacy_context_for_run()?;
        if limit == Some(0) {
            return Err(Exit::BudgetExhausted);
        }
        // spec.md §17.10: a latched NMI ends shutdown unless NMI is blocked;
        // the dispatcher's first boundary admits only that NMI.
        if self.legacy_execution.as_ref().unwrap().shutdown
            && (self.exit_request.word.load(Ordering::Relaxed) & NMI_EDGE == 0
                || self.nmi_state & NMI_BLOCKED != 0)
        {
            return Err(Exit::Shutdown { addr: self.cpu.rip });
        }
        // Decision-16 §§1,3: validate CS:RIP ownership before execution,
        // while leaving every completion slot untouched at zero budget.
        if self.cpu.mmio_retry_active && !self.cpu.owns_mmio_retry(self.cpu.rip) {
            self.cpu.clear_mmio_parts();
        }
        let execution = self.legacy_execution.as_mut().unwrap();
        if !self.cpu.mmio_retry_active && execution.task_switch.context().is_none() {
            execution.instruction_retry = None;
        }
        if let Some(event) = execution.event_retry.filter(|event| {
            (u64::from(event.ip) != self.cpu.rip || event.cs != self.cpu.cs)
                && execution.task_switch.context().is_none()
        }) {
            self.release_legacy_event_boundary();
            // spec.md §17.10: a released NMI is latched again, unblocked; a
            // released exception after a faulting NMI delivery unblocks.
            self.undo_nmi(&event, true);
            self.apply_pending_a20();
        }
        if context.segment_class == crate::SegmentClass::Direct {
            // Direct has no task continuation and shared blocks cannot start a
            // task switch. A Checked transition back starts a fresh execute()
            // attempt, which also clears these stale completion flags.
            let task_switch = &mut self.legacy_execution.as_mut().unwrap().task_switch;
            task_switch.committed = false;
            task_switch.completed = false;
        }
        self.legacy_execution
            .as_mut()
            .unwrap()
            .direct_proofs
            .clear();
        self.legacy_execution.as_mut().unwrap().sync_class(
            &self.cpu,
            self.legacy_model.unwrap(),
            false,
            context.segment_class,
        );
        let execution = self.legacy_execution.as_mut().unwrap();
        execution.validated_direct_bus = raw_direct(&self.cpu);
        execution.validated = Some((
            LegacyContextStamp::new(&self.cpu, self.legacy_model.unwrap()),
            context,
            execution.context_epoch,
        ));
        Ok(context)
    }

    pub(super) fn finish_machine_run(&mut self, result: &mut StepRun) {
        if matches!(result.stop, StepStop::Exit(Exit::BudgetExhausted))
            && !self.cpu.mmio_retry_active
        {
            let execution = self.legacy_execution.as_ref().unwrap();
            if execution.task_switch.context().is_none()
                && execution.instruction_retry.is_none()
                && execution.pending_a20_enabled.is_none()
            {
                return;
            }
        }
        // The Machine bus handoff exposes only this transfer's bytes, including
        // stores whose source register or split operand contains higher bits.
        if let StepStop::Exit(Exit::MmioWrite { size, value, .. }) = &mut result.stop {
            *value &= u64::MAX >> (8 * (8 - *size));
        }
        self.cpu.finish_mmio_retry(match &result.stop {
            StepStop::Exit(exit) => Some(exit),
            StepStop::IrqReady => None,
        });
        if !self.cpu.mmio_retry_active
            && self
                .legacy_execution
                .as_ref()
                .unwrap()
                .task_switch
                .context()
                .is_none()
        {
            self.legacy_execution.as_mut().unwrap().instruction_retry = None;
        }
        self.apply_pending_a20();
        if let StepStop::Exit(exit) = &result.stop {
            // Retry bookkeeping above is engine-owned. Seal the state actually
            // returned to the embedder, allowing only this exit's completion.
            self.legacy_execution
                .as_mut()
                .unwrap()
                .task_switch
                .seal_exit(&self.cpu, exit);
            let execution = self.legacy_execution.as_mut().unwrap();
            execution.event_owner = execution
                .event_retry
                .filter(|event| event.origin.vm)
                .map(|_| crate::interp::legacy::tasks::Owner::at_exit(&self.cpu, exit));
        }
    }

    fn deliver_legacy_event<const ROUTED: bool, const ORDER: u8>(
        &mut self,
        vm: &Vm,
        context: MachineContext,
        event: PendingEvent,
    ) -> Result<(), Exit> {
        if context.paging_enabled {
            return self.deliver_legacy_event_impl::<ROUTED, ORDER, true>(vm, context, event);
        }
        self.deliver_legacy_event_impl::<ROUTED, ORDER, false>(vm, context, event)
    }

    fn deliver_legacy_event_impl<const ROUTED: bool, const ORDER: u8, const PAGED: bool>(
        &mut self,
        vm: &Vm,
        mut context: MachineContext,
        mut event: PendingEvent,
    ) -> Result<(), Exit> {
        if context.generation == CpuGeneration::I8086 {
            return self.deliver_event20::<ROUTED, ORDER>(vm, event);
        }
        if context.generation == CpuGeneration::I386
            && matches!(event.source, EventSource::SingleStep)
        {
            // 80386 PRM §12.2.3 pp. 12-4–5: single-step sets BS;
            // DR6 status bits are sticky, including B0–B3, BD and BT.
            self.cpu.debug_test_registers[crate::legacy::DebugTestRegister::Dr6 as usize] |=
                1 << 14;
        }
        self.legacy_execution.as_mut().unwrap().rep_flags = None;
        self.legacy_execution
            .as_mut()
            .unwrap()
            .direct_proofs
            .clear();
        // Decision-23 §5: processor exceptions can be exposed without delivery.
        if context.regime != MachineRegime::Real
            && self.exception_delivery == crate::ExceptionDelivery::Outward
            && matches!(event.source, EventSource::SingleStep)
        {
            self.release_legacy_event_boundary();
            let mut metadata = crate::fault::FaultMetadata::architectural_exception(
                self.cpu.rip,
                u64::from(event.ip),
                1,
                crate::fault::FaultClass::Trap,
                crate::fault::ErrorCode::NoCode,
            );
            metadata.path = crate::fault::FaultPath::Interpreter;
            return Err(Exit::exception(self.cpu.rip, 1).with_fault_metadata(metadata));
        }
        // Event priority, highest first (debug-register events remain unimplemented):
        // I286 PRM §5.2.1.1, Table 5-1 p. 5-4: instruction exception,
        // single step, NMI, processor-extension segment overrun, INTR.
        // 80386 PRM §9.3, Table 9-2 p. 9-4: non-debug faults, taken
        // INTO/INT n/INT 3, debug traps for this instruction, debug faults
        // for the next instruction, NMI, INTR. Faulting attempts never reach
        // this post-instruction TF delivery; taken software INT suppresses it.
        // I286 PRM §9.7.1 p. 9-14; 80386 PRM §12.3.1.4 p. 12-8:
        // finish single-step delivery before examining lower-priority external
        // interrupts, which may be serviced before the first handler instruction.
        // Interrupt gates clear IF; trap gates retain its prior value.
        // The dispatcher checks a latched NMI after this delivery and before
        // the handler's first instruction, independently of IF (80386 PRM
        // §9.2.2 p. 9-3; spec.md §17.10).
        let (vector, error_code) = loop {
            let (vector, error_code, fault) = match event.source {
                EventSource::SingleStep => (1, None, false),
                EventSource::Irq(vector) => (vector, None, false),
                EventSource::Exception { vector, error_code } => (vector, error_code, true),
                // I286 PRM §9.2 p. 9-3, Table 9-3 p. 9-9; 80386 PRM Table 9-1
                // p. 9-2: vector 2, no error code. External: CPL and gate DPL
                // are not checked (I286 PRM §9.4 p. 9-4).
                EventSource::Nmi { .. } => (2, None, false),
            };
            let delivery = crate::legacy::LegacyEvent {
                vector,
                saved_ip: event.ip,
                software: false,
                external: true,
                error_code,
                fault,
            };
            let result = if PAGED {
                let execution = self.legacy_execution.as_mut().unwrap();
                let state = execution.paging.get_or_insert_with(Default::default);
                state.reconcile(vm, &self.cpu, self.legacy_model.unwrap());
                if !self.cpu.mmio_retry_active {
                    state.event_transaction = None;
                }
                crate::interp::legacy::paging::deliver_interrupt::<ROUTED, ORDER>(
                    &mut self.cpu,
                    &vm.mem,
                    context,
                    delivery,
                    &mut execution.task_switch,
                    &mut state.event_transaction,
                    &state.tlb,
                )
            } else {
                crate::interp::legacy::deliver_interrupt::<ROUTED, ORDER>(
                    &mut self.cpu,
                    &vm.mem,
                    context,
                    delivery,
                    &mut self.legacy_execution.as_mut().unwrap().task_switch,
                )
            };
            match result {
                // A real-address frame holds no error code (SDM Vol 3B
                // §23.1.4); a protected one, gate or task, holds the event's.
                Ok(()) => {
                    break (
                        vector,
                        error_code.filter(|_| context.regime != MachineRegime::Real),
                    )
                }
                Err(exit) => {
                    let committed = self
                        .legacy_execution
                        .as_ref()
                        .unwrap()
                        .task_switch
                        .committed;
                    if committed {
                        // Commit transfers ownership to the task continuation.
                        // A fresh IRQ queued during a later handoff stays queued.
                        self.consume_legacy_irq(&mut event);
                        context = crate::interp::legacy::tasks::fault_context(
                            &self.cpu,
                            context.generation,
                        );
                        event.ip = self.cpu.rip as u32;
                        event.cs = self.cpu.cs;
                    }
                    self.cpu.finish_mmio_retry(Some(&exit));
                    // An accepted event owns its continuation through every
                    // bus handoff and engine refusal, until completion or fault.
                    let execution = self.legacy_execution.as_mut().unwrap();
                    execution.event_retry = (execution.task_switch.context().is_some()
                        || matches!(
                            &exit,
                            Exit::MmioRead { .. }
                                | Exit::MmioWrite { .. }
                                | Exit::UnmappedMemory { .. }
                        ))
                    .then_some(event);
                    let completed = execution.task_switch.completed;
                    if execution.event_retry.is_none() {
                        // A nonresumable exit releases the abandoned boundary,
                        // including inhibition inherited from STI/MOV/POP SS.
                        self.release_legacy_event_boundary();
                    }
                    if completed {
                        if self.delivery_log.is_some()
                            && (event.origin.vm || self.cpu.rflags_system & (1 << 17) != 0)
                        {
                            self.record_delivery(
                                vector,
                                delivery_kind(event.source),
                                event.origin,
                                error_code,
                                Interrupted::Captured,
                            );
                        }
                        self.finish_legacy_event(event, Some(&exit));
                        return Err(exit);
                    }
                    if !matches!(exit, Exit::Exception { .. }) {
                        if self
                            .legacy_execution
                            .as_ref()
                            .unwrap()
                            .event_retry
                            .is_none()
                        {
                            // A nonresumable refusal: as a queued IRQ stays
                            // queued, an NMI is latched again; after a faulting
                            // NMI delivery, its blocking ends (spec.md §17.10).
                            self.undo_nmi(&event, true);
                        }
                        // Accepted delivery owns its inputs through bus handoffs.
                        return Err(exit);
                    }
                    self.apply_pending_a20();
                    if context.regime == MachineRegime::Real
                        && self.exception_delivery == crate::ExceptionDelivery::Outward
                    {
                        // Real-address Outward contract: the fault is returned and a
                        // pending IRQ stays queued for the next IF check; an NMI
                        // is latched again (spec.md §17.10).
                        self.undo_nmi(&event, true);
                        return Err(exit);
                    }
                    self.consume_legacy_irq(&mut event);
                    if self.exception_delivery == crate::ExceptionDelivery::Outward {
                        // 386 PRM Table 9-6 note 3 p. 9-20: the NMI is lost; no
                        // guest handler will IRET (spec.md §17.10).
                        self.undo_nmi(&event, false);
                        return Err(exit);
                    }
                    let next = exception_source(&exit).expect("architectural exception");
                    // I286 PRM §9.6.2 p. 9-10; I386 PRM §9.8.8,
                    // Tables 9-3/9-4 p. 9-13 (scan checked): delivery of #DF
                    // that faults stops the processor until reset. An NMI
                    // taken out of shutdown whose delivery faults re-enters
                    // it, still blocked: only reset ends it (I286 PRM §9.6.2
                    // p. 9-10; SDM Vol 3A §7.15 p. 7-37; spec.md §17.10).
                    // Any other NMI is benign: the fault is delivered
                    // serially and the NMI is lost, still blocked (I286 PRM
                    // §9.6.2 p. 9-10; 386 PRM Table 9-3 p. 9-13).
                    if !committed
                        && matches!(
                            event.source,
                            EventSource::Exception { vector: 8, .. }
                                | EventSource::Nmi { shutdown: true }
                        )
                    {
                        self.legacy_execution.as_mut().unwrap().shutdown = true;
                        return Err(Exit::Shutdown { addr: self.cpu.rip });
                    }
                    // The serial exception carries the lost NMI's blocking
                    // (spec.md §17.10).
                    event.after_nmi |= matches!(event.source, EventSource::Nmi { .. });
                    // 286 PRM §9.6.4 p. 9-11; 386 PRM §7.5 p. 7-8:
                    // descriptor faults after a task switch start a new task's
                    // exception, rather than combining with the old task's event.
                    event.source =
                        if !committed && double_fault(context.generation, event.source, next) {
                            EventSource::Exception {
                                vector: 8,
                                error_code: Some(0),
                            }
                        } else {
                            next
                        };
                    if committed {
                        // The exception belongs to the new task: its frame
                        // holds the new task's CS:IP (spec.md §17.10,
                        // Delivery log).
                        event.origin = self.delivery_origin(event.cs, event.ip);
                    }
                }
            }
        };
        if self.delivery_log.is_some() {
            self.record_delivery(
                vector,
                delivery_kind(event.source),
                event.origin,
                error_code,
                Interrupted::Captured,
            );
        }
        self.finish_legacy_event(event, None);
        Ok(())
    }

    /// Record a delivery that completed and that the dispatch loop continues
    /// from. True at the `DELIVERY_STALL_LIMIT`th one with no instruction
    /// retired since the first; the count then starts again.
    #[cold]
    #[inline(never)]
    fn delivery_stalled(&mut self) -> bool {
        let retired = self.retired;
        let execution = self.legacy_execution.as_mut().unwrap();
        if execution.stall_count == 0 || execution.stall_retired != retired {
            execution.stall_retired = retired;
            execution.stall_count = 0;
        }
        execution.stall_count += 1;
        if execution.stall_count < DELIVERY_STALL_LIMIT {
            return false;
        }
        execution.stall_count = 0;
        true
    }

    // The record origin of a delivery that saves `cs:ip` now (spec.md
    // §17.10, Delivery log), with the CS base and CPL in effect. The
    // twenty-bit models keep no descriptor cache: their base is CS × 16 (8086
    // Family UM §2.3 p. 2-12).
    pub(super) fn delivery_origin(&self, cs: u16, ip: u32) -> Origin {
        if self.twenty_bit_model() {
            Origin {
                cs,
                ip,
                cs_base: u32::from(cs) << 4,
                cpl: 0,
                vm: false,
            }
        } else {
            Origin {
                cs,
                ip,
                cs_base: self.cpu.legacy.segments[SegmentId::Cs as usize].base,
                cpl: self.cpu.legacy.cpl,
                vm: self.cpu.rflags_system & (1 << 17) != 0,
            }
        }
    }

    // Append a delivery of `vector` from `origin` that has just entered its
    // handler to the delivery log (spec.md §17.10, Delivery log), with the
    // interrupted registers from `interrupted` when the log records them.
    // Called only while the log is enabled.
    #[cold]
    #[inline(never)]
    pub(super) fn record_delivery(
        &mut self,
        vector: u8,
        kind: crate::DeliveryKind,
        origin: Origin,
        error_code: Option<u32>,
        interrupted: Interrupted,
    ) {
        let twenty = self.twenty_bit_model();
        let regs = self
            .delivery_log
            .as_ref()
            .is_some_and(|log| log.options().records_registers())
            .then(|| match interrupted {
                Interrupted::Captured => self.legacy_execution.as_ref().unwrap().task_switch.regs,
                // 8086 Family UM §2.6 pp. 2-25–28: the frame is three words
                // below the interrupted SP.
                Interrupted::Frame20 { flags } => Some(crate::delivery_log::regs_at(
                    &self.cpu,
                    crate::CpuGeneration::I8086,
                    u32::from(flags),
                    self.cpu.gpr[4].wrapping_add(6),
                )),
            })
            .flatten()
            .map(Box::new);
        // 8086 Family UM §2.2 p. 2-7: a twenty-bit IP is a sixteen-bit offset.
        let ip = if twenty {
            origin.ip & 0xffff
        } else {
            origin.ip
        };
        let linear = origin.cs_base.wrapping_add(ip);
        let record = crate::DeliveryRecord {
            vector,
            kind,
            saved_cs: origin.cs,
            saved_ip: ip,
            // 8086 Family UM §2.8 p. 2-68: twenty address bits.
            saved_linear: if twenty { linear & 0xfffff } else { linear },
            error_code,
            cpl_before: origin.cpl,
            cpl_after: if twenty { 0 } else { self.cpu.legacy.cpl },
            retired: self.retired,
            regs,
            virtual_8086_before: origin.vm,
            virtual_8086_after: !twenty && self.cpu.rflags_system & (1 << 17) != 0,
        };
        if let Some(log) = &mut self.delivery_log {
            log.push(record);
        }
    }

    // The software interrupt the unit just walked took, as the walk itself
    // reported it (spec.md §17.10, Delivery log). An 80286/80386 INT marks
    // its outcome and leaves the origin it read in the task-switch state; a
    // twenty-bit INT, INT 3, INTO, CHKIND or BRKEM writes the vector and the
    // CS:IP its frame holds into the op's `taken` temporary once the frame is
    // written (`legacy::report_interrupt20`). Called only while the log is
    // enabled.
    #[cold]
    #[inline(never)]
    fn record_unit(
        &mut self,
        twenty: bool,
        ir: &IrBlock,
        progress: &crate::interp::legacy::Progress,
    ) {
        let (vector, origin, interrupted) = if twenty {
            let taken = |op: &IrOp| match op {
                IrOp::Legacy(
                    crate::legacy::LegacyOp::Interrupt20 { taken, .. }
                    | crate::legacy::LegacyOp::CheckIndex20 { taken, .. }
                    | crate::legacy::LegacyOp::Brkem20 { taken, .. },
                ) => Some(*taken),
                _ => None,
            };
            // The interrupt op ends its block (`vm::i8086::resolve`): the block
            // holds one, in its last instruction.
            debug_assert!(
                {
                    let last = ir
                        .ops
                        .iter()
                        .rposition(|op| matches!(op, IrOp::InsnStart { .. }));
                    let mut found = ir
                        .ops
                        .iter()
                        .enumerate()
                        .filter(|(_, op)| taken(op).is_some());
                    match (found.next(), found.next(), last) {
                        (None, _, _) => true,
                        (Some((at, _)), None, Some(last)) => at > last,
                        _ => false,
                    }
                },
                "a twenty-bit interrupt op that is not its block's only one, in its last instruction"
            );
            let Some(taken) = ir.ops.iter().rev().find_map(taken) else {
                return;
            };
            // A walk that retired the whole block reached the op, which wrote
            // its report; one that stopped earlier left the temporary from an
            // earlier walk.
            if progress.retired != u64::from(ir.icount) {
                return;
            }
            let Some((vector, cs, ip, flags)) = self
                .interp_scratch
                .values
                .get(taken as usize)
                .and_then(|&report| crate::legacy::reported_interrupt20(report))
            else {
                return;
            };
            (
                vector,
                self.delivery_origin(cs, u32::from(ip)),
                Interrupted::Frame20 { flags },
            )
        } else {
            if !matches!(&progress.outcome, Some(Ok(outcome)) if outcome.software_interrupt) {
                return;
            }
            let delivered = self
                .legacy_execution
                .as_ref()
                .unwrap()
                .task_switch
                .software_interrupt;
            (
                delivered.vector,
                Origin {
                    cs: delivered.cs,
                    ip: delivered.ip,
                    cs_base: delivered.cs_base,
                    cpl: delivered.cpl,
                    vm: delivered.vm,
                },
                Interrupted::Captured,
            )
        };
        self.record_delivery(
            vector,
            crate::DeliveryKind::Software,
            origin,
            None,
            interrupted,
        );
    }

    fn consume_legacy_irq(&mut self, event: &mut PendingEvent) {
        if !event.irq_consumed {
            if let EventSource::Irq(vector) = event.source {
                if self.pending_irq == Some(vector) {
                    self.pending_irq = None;
                }
                event.irq_consumed = true;
            }
        }
    }

    fn release_legacy_event_boundary(&mut self) {
        // Decision-24: an abandoned event owns no future inhibition or trap.
        // Clear private boundary state; architectural TF belongs to CpuState.
        let execution = self.legacy_execution.as_mut().unwrap();
        execution.event_retry = None;
        execution.event_owner = None;
        execution.handoff_service &= !HANDOFF_DEBUG;
        self.sti_shadow = false;
        self.ss_shadow = 0;
    }

    fn abandon_legacy_event(&mut self) {
        if let Some(mut event) = self.legacy_execution.as_mut().unwrap().event_retry.take() {
            self.consume_legacy_irq(&mut event);
            // spec.md §17.10: an abandoned NMI, or exception after a faulting
            // NMI delivery, is consumed and leaves no blocking.
            self.undo_nmi(&event, false);
        }
        self.release_legacy_event_boundary();
    }

    fn finish_legacy_event(&mut self, mut event: PendingEvent, exit: Option<&Exit>) {
        self.consume_legacy_irq(&mut event);
        self.release_legacy_event_boundary();
        self.cpu.finish_mmio_retry(exit);
        self.apply_pending_a20();
        if self.twenty_bit_model() {
            return;
        }
        let class = self
            .legacy_context_for_run()
            .map_or(crate::SegmentClass::Checked, |context| {
                context.segment_class
            });
        self.legacy_execution.as_mut().unwrap().sync_class(
            &self.cpu,
            self.legacy_model.unwrap(),
            true,
            class,
        );
    }

    fn route_legacy_exception<const ROUTED: bool, const ORDER: u8>(
        &mut self,
        vm: &Vm,
        context: MachineContext,
        exit: Exit,
    ) -> Result<(), Exit> {
        if self.exception_delivery != crate::ExceptionDelivery::InGuest {
            return Err(exit);
        }
        let Some(source) = exception_source(&exit) else {
            return Err(exit);
        };
        self.cpu.finish_mmio_retry(Some(&exit));
        self.legacy_execution.as_mut().unwrap().instruction_retry = None;
        self.apply_pending_a20();
        let ip = exit
            .fault_metadata()
            .map_or(self.cpu.rip, |m| m.saved_rip()) as u32;
        self.deliver_legacy_event::<ROUTED, ORDER>(
            vm,
            context,
            PendingEvent {
                irq_consumed: false,
                software: false,
                after_nmi: false,
                source,
                ip,
                cs: self.cpu.cs,
                origin: self.delivery_origin(self.cpu.cs, ip),
            },
        )
    }

    #[inline]
    fn machine_pending_io(&self, task_pending: bool) -> bool {
        task_pending
            || self.cpu.mmio_retry_active
            || self.cpu.pending_port_in.is_some()
            || self.cpu.pending_mmio.is_some()
            || self.cpu.pending_mmio_write
            || self.cpu.port_string_in.is_some()
    }

    #[inline]
    fn machine_boundary<const SINGLE: bool, const KIND: u8>(
        &self,
        options: StepOptions,
    ) -> MachineBoundary {
        let execution = self.legacy_execution.as_ref().unwrap();
        let task_pending = execution.task_switch_pending();
        // I286 PRM §9.7.1 p. 9-14; 80386 PRM §12.3.1.4 p. 12-8:
        // TF changes only at an instruction/event boundary. Task continuations
        // retain the sample from their original instruction.
        let single_step = if task_pending {
            execution
                .task_switch
                .single_step()
                .unwrap_or(self.cpu.rflags_system & 0x100 != 0)
        } else {
            self.cpu.rflags_system & 0x100 != 0
        };
        let irq_pending = self.pending_irq.is_some();
        let a20_pending = execution.pending_a20_enabled.is_some();
        MachineBoundary {
            task_pending,
            single_step,
            irq_pending,
            // A twenty-bit model is never blocked (`nmi.rs`). An enabled
            // delivery log reads each unit's software interrupt in the
            // epilogue (`record_unit`).
            cold_epilogue: a20_pending
                || (KIND != super::DIRECT20_DISPATCH && self.nmi_state != 0)
                || self.delivery_log.is_some(),
            one_step: SINGLE
                || single_step
                || self.sti_shadow
                || self.ss_shadow & SS_HOLD != 0
                || irq_pending
                || a20_pending
                || options.stop_when_irq_ready,
            service: self.cpu.port_string_in.is_some()
                || execution.handoff_service != 0
                || execution.event_retry.is_some()
                || irq_pending
                || options.stop_when_irq_ready,
            // A one-step budget already caps execution and forces refresh.
            // Its I/O eligibility is queried only when a host request exists.
            pending_io: !SINGLE && self.machine_pending_io(task_pending),
            instruction_retry: execution.instruction_retry.is_some(),
        }
    }

    // Service step: an INS element whose input was answered completes, which
    // ends both shadows and an MMIO retry; a queued A20 change applies, and the
    // boundary is derived again.
    #[inline(always)]
    fn finish_port_string<
        const ROUTED: bool,
        const ORDER: u8,
        const SINGLE: bool,
        const KIND: u8,
    >(
        &mut self,
        vm: &Vm,
        options: StepOptions,
        physical_mask: u32,
        boundary: &mut MachineBoundary,
        direct_bus: &mut bool,
    ) -> Result<(), Exit> {
        if KIND == super::DIRECT20_DISPATCH {
            crate::interp::i8086::finish_port_input::<ROUTED, ORDER>(
                &mut self.cpu,
                &vm.mem,
                self.cpu_model,
            )
        } else if KIND == MACHINE_DISPATCH && self.cpu.legacy.cr0 & 0x8000_0000 != 0 {
            self.finish_paged_port_input::<ROUTED, ORDER>(vm)
        } else {
            crate::interp::legacy::finish_port_input::<ROUTED, ORDER>(
                &mut self.cpu,
                &vm.mem,
                physical_mask,
            )
        }?;
        self.sti_shadow = false;
        self.ss_shadow = 0;
        if self.cpu.mmio_retry_active {
            self.cpu.finish_mmio_retry(None);
        }
        if self
            .legacy_execution
            .as_ref()
            .unwrap()
            .pending_a20_enabled
            .is_some()
        {
            self.apply_pending_a20();
            *direct_bus = raw_direct(&self.cpu);
        }
        *boundary = self.machine_boundary::<SINGLE, KIND>(options);
        Ok(())
    }

    // A handoff's completed transfer releases its owned single-step trap.
    #[inline(always)]
    fn queue_port_debug(&mut self, task_pending: bool) {
        if !task_pending
            && self
                .legacy_execution
                .as_ref()
                .unwrap()
                .debug_handoff_pending()
            && self.cpu.pending_port_in.is_none()
            && self.cpu.port_string_in.is_none()
        {
            self.queue_debug_event();
        }
    }

    #[inline(always)]
    fn queue_debug_event(&mut self) {
        self.legacy_execution.as_mut().unwrap().handoff_service &= !HANDOFF_DEBUG;
        self.legacy_execution.as_mut().unwrap().event_retry = Some(PendingEvent {
            irq_consumed: false,
            software: false,
            after_nmi: false,
            source: EventSource::SingleStep,
            ip: self.cpu.rip as u32,
            cs: self.cpu.cs,
            origin: self.delivery_origin(self.cpu.cs, self.cpu.rip as u32),
        });
    }

    // Profile constants keep compiled chaining free of interpreted context checks.
    #[inline(always)]
    pub(super) fn run_dispatch<
        const ROUTED: bool,
        const DEFAULT: bool,
        const BOUNDED: bool,
        const KIND: u8,
        const STEPPED: bool,
        const STOP: bool,
        const ORDER: u8,
        const SINGLE: bool,
    >(
        &mut self,
        vm: &Vm,
        budget: u64,
        fp_mode: &mut crate::jit_abi::fp::RunMode,
        origin: SharedReason,
        options: StepOptions,
        initial_context: Option<(MachineContext, Option<u64>)>,
    ) -> DispatchResult {
        if KIND != APPLICATION_DISPATCH {
            self.cpu.machine_flags_mask =
                vm.cpu_model.map_or(0, crate::CpuModel::machine_flags_mask);
        }
        // Epoch numbers belong to a domain. A vCPU used with another VM must
        // not carry predictions whose equal epoch names a different backend.
        let mut participant = if KIND == APPLICATION_DISPATCH {
            let lifetime = vm.cache.code_lifetime();
            if !Arc::ptr_eq(&self.fast_lifetime, &lifetime) {
                self.fast_clear();
                self.execution_participant = lifetime.participant();
                self.fast_epoch = lifetime.epoch();
                self.fast_lifetime = lifetime;
            }
            Some(self.execution_participant.for_dispatch())
        } else {
            None
        };
        let mxcsr_mode = if DEFAULT { 0 } else { self.fast_mxcsr_mode };
        let mut shared_reason = origin;
        let mut blocks_run: u64 = 0;
        let exit_word_owner = Arc::clone(&self.exit_request.word);
        let exit_word: &AtomicU64 = &exit_word_owner;
        // Deliberately a LOCAL: taking it as a `&mut` parameter instead measured +5.4%
        // on the dispatch-micro bench, because the compiler stops treating it as one.
        let mut ctx = MemCtx::for_memory(&vm.mem);
        ctx.time_enabled = u64::from(vm.time_enabled);
        self.fault_entry.publish(&mut self.cpu, &mut ctx);
        if KIND == APPLICATION_DISPATCH {
            ctx.dispatch_word_ptr = exit_word as *const AtomicU64 as u64;
            // Hand compiled code this vcpu's shadow return stack. `self.ret_stack` is
            // boxed, so its address is stable for the whole run despite `&mut self`.
            ctx.ret_stack = std::ptr::addr_of_mut!(*self.ret_stack) as u64;
            // ...and the executed-instruction counter it charges into. Stable
            // for the run: `self` cannot move while this call is on the stack. Read only
            // by blocks the backend chose to emit accounting into.
            ctx.icount_ptr = std::ptr::addr_of_mut!(self.executed) as u64;
        }
        if (KIND == APPLICATION_DISPATCH || KIND == ALIAS_DISPATCH) && (!BOUNDED || budget != 0) {
            if let Err(exit) = self.finish_port_string_in::<ROUTED>(vm) {
                return if KIND == APPLICATION_DISPATCH {
                    DispatchResult::Application(Ok(exit))
                } else {
                    DispatchResult::Stepped(StepRun {
                        steps: 0,
                        stop: StepStop::Exit(exit),
                    })
                };
            }
        }
        let mut dispatch = if KIND == APPLICATION_DISPATCH {
            exit_word.load(Ordering::Relaxed)
        } else {
            0
        };
        let limit = if SINGLE {
            Some(1)
        } else if KIND >= MACHINE_DISPATCH {
            if STEPPED {
                Some(budget)
            } else {
                initial_context.unwrap().1
            }
        } else {
            BOUNDED.then_some(budget)
        };
        let mut steps = 0;
        let mut alias_steps = crate::interp::StepContext::new(budget);
        let model = self.legacy_model.unwrap_or(LegacyCpuModel {
            generation: CpuGeneration::I286,
            model_semantics_id: 0,
        });
        // The model's bus mask, once per call: the dispatch loop's page hint
        // and port-input completion form physical addresses with it.
        let physical_mask = crate::legacy::physical_mask(model.generation);
        let mut context =
            initial_context.map_or_else(crate::lift::legacy::alias_context, |initial| initial.0);
        let direct20 = KIND == super::DIRECT20_DISPATCH;
        let mut direct_bus = direct20
            || KIND == ALIAS_DISPATCH
            || (KIND >= MACHINE_DISPATCH
                && self.legacy_execution.as_ref().unwrap().validated_direct_bus);
        if KIND == MACHINE_DISPATCH {
            // CpuState may have been restored or edited between calls. Sample
            // the opt-in route once, never at each instruction boundary. The
            // twenty-bit and Application dispatchers compile this away.
            let execution = self.legacy_execution.as_mut().unwrap();
            execution.handoff_service = (execution.handoff_service & !HANDOFF_NUMERIC)
                | if self.cpu.numeric_error_route == crate::NumericErrorRoute::External {
                    HANDOFF_NUMERIC
                } else {
                    0
                };
        }
        let mut boundary = MachineBoundary::default();
        let mut refresh_boundary = true;
        loop {
            if direct20 {
                // spec.md §17.10: the twenty-bit entries have a permanent
                // Direct context, the one every derivation of theirs returns
                // (`legacy_context_for_run`). As a constant it costs no turn.
                context = crate::lift::i8086::CONTEXT;
            }
            if KIND >= MACHINE_DISPATCH {
                if limit.is_some_and(|limit| steps >= limit) {
                    return DispatchResult::Stepped(StepRun {
                        steps,
                        stop: StepStop::Exit(Exit::BudgetExhausted),
                    });
                }
                if refresh_boundary {
                    boundary = self.machine_boundary::<SINGLE, KIND>(options);
                    // NEC V20/V30 8080 mode decides its boundary after the
                    // service steps (decision-23 C4a).
                    if direct20 && self.cpu.legacy.reserved & crate::legacy::NEC_EMULATION != 0 {
                        boundary.service = true;
                    }
                }
                // Only an ordinary completed shared block can retain this proof.
                // Helpers, faults and event continuations refresh on the next turn.
                refresh_boundary = true;
                let task_pending = boundary.task_pending;
                // A host request, or an NMI edge (spec.md §17.10), in one load.
                // Both are decided out of line, the NMI step included.
                if self.exit_request.word.load(Ordering::Relaxed) & (EXIT_REQUESTED | NMI_EDGE) != 0
                {
                    match self.machine_external::<ROUTED, ORDER, SINGLE, KIND>(
                        vm,
                        options,
                        context,
                        task_pending,
                        boundary.pending_io,
                        direct_bus,
                    ) {
                        External::Proceed => {}
                        External::OneStep => {
                            boundary.service = true;
                            boundary.one_step = true;
                        }
                        External::Serviced(next, bus) => {
                            boundary = next;
                            direct_bus = bus;
                        }
                        External::Delivered => {
                            // Delivery costs no instruction unit. Re-derive
                            // after the CS reload.
                            context = match self.legacy_context_for_run() {
                                Ok(context) => context,
                                Err(exit) => {
                                    return DispatchResult::Stepped(StepRun {
                                        steps,
                                        stop: StepStop::Exit(exit),
                                    })
                                }
                            };
                            direct_bus = raw_direct(&self.cpu);
                            if self.delivery_stalled() {
                                return DispatchResult::Stepped(StepRun {
                                    steps,
                                    stop: StepStop::Exit(Exit::BudgetExhausted),
                                });
                            }
                            continue;
                        }
                        External::Stop(exit) => {
                            return DispatchResult::Stepped(StepRun {
                                steps,
                                stop: StepStop::Exit(exit),
                            })
                        }
                    }
                }
                if boundary.service {
                    if self.cpu.port_string_in.is_some() {
                        if let Err(exit) = self.finish_port_string::<ROUTED, ORDER, SINGLE, KIND>(
                            vm,
                            options,
                            physical_mask,
                            &mut boundary,
                            &mut direct_bus,
                        ) {
                            return DispatchResult::Stepped(StepRun {
                                steps,
                                stop: StepStop::Exit(exit),
                            });
                        }
                    }
                    // This is the existing deferred-debug service gate, with
                    // External folded into its flag. Internal with no deferred
                    // debug skips the entire body, including pending-I/O checks.
                    let service = self.legacy_execution.as_ref().unwrap().handoff_service;
                    if !task_pending
                        && service != 0
                        && self.cpu.pending_port_in.is_none()
                        && self.cpu.port_string_in.is_none()
                    {
                        if service & HANDOFF_NUMERIC != 0 && !self.machine_pending_io(task_pending)
                        {
                            if let Some(exit) = self.cpu.reconcile_numeric_error(self.cpu.rip) {
                                return DispatchResult::Stepped(StepRun {
                                    steps,
                                    stop: StepStop::Exit(exit),
                                });
                            }
                        }
                        if service & HANDOFF_DEBUG != 0 {
                            self.queue_debug_event();
                        }
                    }
                    // spec.md §17.10: the NMI step, after an owned
                    // continuation and before a queued IRQ, ran in
                    // `machine_external`.
                    let event = self
                        .legacy_execution
                        .as_ref()
                        .unwrap()
                        .event_retry
                        .or_else(|| {
                            self.pending_irq
                                .filter(|_| {
                                    !task_pending
                                        && self.irq_delivery_ready()
                                        && !self.cpu.mmio_retry_active
                                })
                                .map(|vector| PendingEvent {
                                    irq_consumed: false,
                                    software: false,
                                    after_nmi: false,
                                    source: EventSource::Irq(vector),
                                    ip: self.cpu.rip as u32,
                                    cs: self.cpu.cs,
                                    origin: self.delivery_origin(self.cpu.cs, self.cpu.rip as u32),
                                })
                        });
                    if let Some(event) = event {
                        if let Err(exit) =
                            self.deliver_legacy_event::<ROUTED, ORDER>(vm, context, event)
                        {
                            return DispatchResult::Stepped(StepRun {
                                steps,
                                stop: StepStop::Exit(exit),
                            });
                        }
                        // Delivery costs no instruction unit. Re-derive after the CS reload.
                        context = match self.legacy_context_for_run() {
                            Ok(context) => context,
                            Err(exit) => {
                                return DispatchResult::Stepped(StepRun {
                                    steps,
                                    stop: StepStop::Exit(exit),
                                })
                            }
                        };
                        direct_bus = raw_direct(&self.cpu);
                        if self.delivery_stalled() {
                            return DispatchResult::Stepped(StepRun {
                                steps,
                                stop: StepStop::Exit(Exit::BudgetExhausted),
                            });
                        }
                        continue;
                    }
                    if options.stop_when_irq_ready
                        && !task_pending
                        && !self.cpu.mmio_retry_active
                        && self.irq_delivery_ready()
                    {
                        return DispatchResult::Stepped(StepRun {
                            steps,
                            stop: StepStop::IrqReady,
                        });
                    }
                    if direct20 && self.cpu.legacy.reserved & crate::legacy::NEC_EMULATION != 0 {
                        // RETEM or CALLN completed as one unit; otherwise stop.
                        if let Err(exit) = self.nec_boundary::<ROUTED, ORDER>(vm) {
                            return DispatchResult::Stepped(StepRun {
                                steps,
                                stop: StepStop::Exit(exit),
                            });
                        }
                        steps += 1;
                        continue;
                    }
                }
                let direct = !task_pending && context.segment_class == crate::SegmentClass::Direct;
                // 8086 Family UM §2.2 p. 2-7: IP is a sixteen-bit offset,
                // including when the embedder has supplied upper storage bits.
                let fetch_offset = if direct20 {
                    self.cpu.rip & 0xffff
                } else {
                    self.cpu.rip
                };
                let linear_fetch = (u64::from(self.cpu.cs) << 4) + fetch_offset;
                let linear_fetch = if direct20 {
                    linear_fetch & 0xfffff
                } else {
                    linear_fetch
                };
                let physical = if direct && direct_bus {
                    linear_fetch
                } else {
                    crate::legacy::physical_span(&self.cpu, physical_mask, linear_fetch, 1).0
                };
                let page = physical >> crate::memory::CODE_PAGE_BITS;
                let dirty = exit_word.load(Ordering::Relaxed) & crate::memory::DISPATCH_DIRTY != 0;
                let dropped = dirty.then(|| vm.cache.invalidated_blocks());
                vm.handle_smc(exit_word);
                let execution = self.legacy_execution.as_mut().unwrap();
                if dropped.is_some_and(|before| before != vm.cache.invalidated_blocks()) {
                    if let Some(last) = execution.last_direct_fetch {
                        // This bounded page hint is only a lowering choice. Precise
                        // source invalidation and byte revalidation remain authoritative.
                        for page in [last >> crate::memory::CODE_PAGE_BITS, page] {
                            if !execution.smc_pages.contains(&page) {
                                if execution.smc_pages.len() == 16 {
                                    execution.smc_pages.remove(0);
                                }
                                execution.smc_pages.push(page);
                            }
                        }
                    }
                }
                let retained = direct && (!direct_bus || execution.smc_pages.contains(&page));
                let resolved = if direct20 {
                    super::i8086::resolve(vm, &self.cpu, exit_word, physical)
                } else if direct && !retained && !boundary.instruction_retry {
                    resolve_direct(
                        vm,
                        &self.cpu,
                        context,
                        exit_word,
                        physical,
                        &mut self.legacy_execution.as_mut().unwrap().direct_proofs,
                    )
                } else if KIND == MACHINE_DISPATCH && context.paging_enabled {
                    // Keep paged selection behind
                    // the existing Direct source-proof fast path.
                    self.resolve_paged(vm, context)
                } else {
                    let execution = self.legacy_execution.as_mut().unwrap();
                    if execution.context_id == u64::MAX {
                        execution.sync_class(&self.cpu, model, true, crate::SegmentClass::Checked);
                    }
                    execution.resolve::<KIND>(
                        vm,
                        &self.cpu,
                        model,
                        MachineContext {
                            // The masked bus lowering retains the Direct proof;
                            // SMC uses the cheaper original Checked instruction IR.
                            segment_class: if context.segment_class == crate::SegmentClass::Flat32 {
                                crate::SegmentClass::Flat32
                            } else if direct && !direct_bus {
                                crate::SegmentClass::Direct
                            } else {
                                crate::SegmentClass::Checked
                            },
                            ..context
                        },
                        exit_word,
                    )
                };
                self.legacy_execution.as_mut().unwrap().last_direct_fetch =
                    (direct && !retained).then_some(linear_fetch);
                let (ir, walk) = match resolved {
                    Ok(ir) => ir,
                    Err(exit) => {
                        self.legacy_execution
                            .as_mut()
                            .unwrap()
                            .direct_proofs
                            .clear();
                        match self.route_legacy_exception::<ROUTED, ORDER>(vm, context, exit) {
                            Ok(()) => {
                                context = match self.legacy_context_for_run() {
                                    Ok(context) => context,
                                    Err(exit) => {
                                        return DispatchResult::Stepped(StepRun {
                                            steps,
                                            stop: StepStop::Exit(exit),
                                        })
                                    }
                                };
                                direct_bus = raw_direct(&self.cpu);
                                if self.delivery_stalled() {
                                    return DispatchResult::Stepped(StepRun {
                                        steps,
                                        stop: StepStop::Exit(Exit::BudgetExhausted),
                                    });
                                }
                                continue;
                            }
                            Err(exit) => {
                                return DispatchResult::Stepped(StepRun {
                                    steps,
                                    stop: StepStop::Exit(exit),
                                })
                            }
                        }
                    }
                };
                let single_step = boundary.single_step;
                let remaining = if boundary.one_step {
                    1
                } else {
                    limit.map_or(u64::MAX, |limit| limit - steps)
                };
                let progress = if direct20 {
                    crate::interp::legacy::execute_shared20::<ROUTED, ORDER>(
                        &ir,
                        &mut self.cpu,
                        &vm.mem,
                        &mut self.interp_scratch,
                        remaining,
                        exit_word,
                        self.sti_shadow,
                        self.ss_shadow,
                        boundary.irq_pending,
                        // A twenty-bit model is never blocked (`nmi.rs`).
                        false,
                    )
                } else if KIND >= MACHINE_DISPATCH && matches!(walk, BlockWalk::Protected) {
                    if context.paging_enabled {
                        self.execute_paged::<ROUTED, ORDER>(vm, &ir, context, remaining)
                    } else {
                        self.legacy_execution
                            .as_mut()
                            .unwrap()
                            .execute_protected_block::<ROUTED, ORDER>(
                                &ir,
                                &mut self.cpu,
                                &vm.mem,
                                &mut self.interp_scratch,
                                model,
                                self.retired,
                            )
                    }
                } else {
                    crate::interp::legacy::execute_block::<ROUTED, ORDER>(
                        walk,
                        &ir,
                        &mut self.cpu,
                        &vm.mem,
                        &mut self.interp_scratch,
                        remaining,
                        exit_word,
                        &mut self.legacy_execution.as_mut().unwrap().task_switch,
                        self.sti_shadow,
                        self.ss_shadow,
                        boundary.irq_pending,
                        self.nmi_state & NMI_BLOCKED != 0,
                    )
                };
                if direct20 {
                    self.cpu.rip &= 0xffff;
                    let restart = ir.ops.iter().find_map(|op| match op {
                        IrOp::Legacy(crate::legacy::LegacyOp::WrappedString {
                            restart_ip,
                            rep,
                            ..
                        }) if !matches!(rep, crate::ir::RepKind::None)
                            && self.cpu.rip == ir.guest_start =>
                        {
                            Some((self.cpu.cs, self.cpu.rip as u16, *restart_ip))
                        }
                        _ => None,
                    });
                    // UM §2.7 p. 2-42: only a string with a completed element is
                    // interrupted inside the instruction. A stop before its first
                    // element keeps the full owner unless an earlier walk already
                    // completed one (the same pair is still pending; a host edit
                    // of the string or its inputs dropped it at entry, xj-50w.15).
                    let execution = self.legacy_execution.as_mut().unwrap();
                    execution.rep_restart20 = restart.filter(|pair| {
                        progress.steps != 0 || execution.rep_restart20 == Some(*pair)
                    });
                }
                if !single_step && progress.outcome.is_none() {
                    steps += progress.steps;
                    self.retired += progress.retired;
                    self.executed += progress.retired;
                    self.sti_shadow = progress.sti_shadow;
                    self.ss_shadow = progress.ss_shadow;
                    if boundary.instruction_retry {
                        self.legacy_execution.as_mut().unwrap().instruction_retry = None;
                        boundary.instruction_retry = false;
                    }
                    // 8086 Family UM §2.6 p. 2-28: a lifted POPF or IRET can load
                    // TF; the instruction after it is the first one trapped.
                    refresh_boundary = boundary.one_step
                        || boundary.pending_io
                        || self.sti_shadow
                        || self.ss_shadow & SS_HOLD != 0
                        || (direct20 && self.cpu.rflags_system & 0x100 != 0);
                    if boundary.cold_epilogue {
                        if self.delivery_log.is_some() {
                            self.record_unit(direct20, &ir, &progress);
                        }
                        // A queued A20 change, or an IRET that ends NMI
                        // blocking: derive the bus and the boundary again.
                        if self.machine_epilogue(&ir, progress.retired) {
                            direct_bus = raw_direct(&self.cpu);
                            refresh_boundary = true;
                        }
                    }
                    if SINGLE && steps != 0 {
                        // A spent one-step budget: the next turn's budget
                        // test returns this, before anything else runs.
                        return DispatchResult::Stepped(StepRun {
                            steps,
                            stop: StepStop::Exit(Exit::BudgetExhausted),
                        });
                    }
                    continue;
                }
                steps += progress.steps;
                self.retired += progress.retired;
                self.executed += progress.retired;
                // spec.md §17.10: while NMI is blocked, an IRET this unit
                // retired, or one that faulted, ends blocking. Blocking sets
                // `cold_epilogue`, as for the fast arm; a twenty-bit model is
                // never blocked (`nmi.rs`).
                if boundary.cold_epilogue {
                    if !direct20 {
                        self.nmi_after_progress(&ir, &progress);
                    }
                    if self.delivery_log.is_some() {
                        self.record_unit(direct20, &ir, &progress);
                    }
                }
                let outcome = progress.into_outcome();
                if !outcome.as_ref().is_ok_and(|outcome| {
                    !outcome.context_changed && !outcome.translations_invalidated
                }) {
                    self.legacy_execution
                        .as_mut()
                        .unwrap()
                        .direct_proofs
                        .clear();
                }
                // NEC V20/V30: BRKEM or RETI made MD 0. Both refuse a start
                // with BRK = 1 before they commit (interp/i8086.rs), so no
                // single-step boundary reaches this point.
                if direct20 && self.cpu.legacy.reserved & crate::legacy::NEC_EMULATION != 0 {
                    debug_assert!(!single_step);
                    let brkem = ir.ops.iter().rev().find_map(|op| match op {
                        IrOp::Legacy(crate::legacy::LegacyOp::Brkem20 { vector, .. }) => {
                            Some(*vector)
                        }
                        _ => None,
                    });
                    self.legacy_execution.as_mut().unwrap().nec_entry = Some(match brkem {
                        Some(vector) => crate::I8080Entry::Brkem { vector },
                        None => crate::I8080Entry::Reti,
                    });
                }
                match outcome {
                    Ok(outcome) => {
                        // Decision-16 §4: the SS class is not rearmed; the NMI
                        // hold is, by every load (spec.md §17.10).
                        let arm_ss = outcome.arm_ss & SS_HOLD != 0 && self.ss_shadow & SS_HOLD == 0;
                        if self.cpu.port_string_in.is_none() {
                            self.sti_shadow = outcome.arm_sti;
                            self.ss_shadow = u8::from(arm_ss) | (outcome.arm_ss & NMI_HOLD);
                        }
                        if self.cpu.mmio_retry_active {
                            self.cpu.finish_mmio_retry(None);
                        }
                        // SDM Vol 2B REP, p. 4-564: a completed iteration is an
                        // interruption window with updated registers (decision-22).
                        self.legacy_execution.as_mut().unwrap().instruction_retry = None;
                        if self
                            .legacy_execution
                            .as_ref()
                            .unwrap()
                            .pending_a20_enabled
                            .is_some()
                        {
                            self.apply_pending_a20();
                            direct_bus = raw_direct(&self.cpu);
                        }
                        if !direct20 && outcome.context_changed {
                            let execution = self.legacy_execution.as_mut().unwrap();
                            if outcome.translations_invalidated {
                                execution.invalidate_translations();
                            }
                            context = match self.legacy_context_for_run() {
                                Ok(context) => context,
                                Err(exit) => {
                                    return DispatchResult::Stepped(StepRun {
                                        steps,
                                        stop: StepStop::Exit(exit),
                                    })
                                }
                            };
                            direct_bus = raw_direct(&self.cpu);
                            self.legacy_execution.as_mut().unwrap().sync_class(
                                &self.cpu,
                                model,
                                true,
                                context.segment_class,
                            );
                        }
                        // I286 PRM Appendix B MOV p. B-73, POP p. B-83;
                        // 80386 PRM §9.2.4 pp. 9-3–4, §17 MOV p. 17-109,
                        // POP p. 17-126: only MOV/POP SS suppress this boundary.
                        // SDM Vol 3A §7.8.3 pp. 7-8–9: consecutive SS loads
                        // guarantee inhibition only for the first load; LSS has none.
                        let debug = single_step && !arm_ss && !outcome.suppress_debug;
                        if debug {
                            if matches!(
                                outcome.exit,
                                Some(
                                    Exit::PortIo { .. }
                                        | Exit::ExternalNumericError {
                                            event: crate::NumericErrorEvent::Asserted,
                                            ..
                                        }
                                )
                            ) {
                                self.legacy_execution.as_mut().unwrap().handoff_service |=
                                    HANDOFF_DEBUG;
                            } else {
                                context =
                                    match self.deliver_step_trap::<ROUTED, ORDER, KIND>(vm, &ir) {
                                        Ok(context) => context,
                                        Err(exit) => {
                                            return DispatchResult::Stepped(StepRun {
                                                steps,
                                                stop: StepStop::Exit(exit),
                                            })
                                        }
                                    };
                                direct_bus = raw_direct(&self.cpu);
                                // HLT completed; the enabled debug trap resumes at
                                // its successor (I286 PRM Appendix B p. B-42;
                                // 80386 PRM §17 HLT p. 17-60).
                                continue;
                            }
                        }
                        if let Some(exit) = outcome.exit {
                            return DispatchResult::Stepped(StepRun {
                                steps,
                                stop: StepStop::Exit(exit),
                            });
                        }
                    }
                    Err(exit) => {
                        // An architectural fault abandons this attempt before it is
                        // returned to the embedder; only resumable transfers keep IR.
                        self.cpu.finish_mmio_retry(Some(&exit));
                        if self.cpu.mmio_retry_active
                            || self
                                .legacy_execution
                                .as_ref()
                                .unwrap()
                                .task_switch
                                .context()
                                .is_some()
                        {
                            let retry =
                                if direct || context.segment_class == crate::SegmentClass::Flat32 {
                                    retry_instruction(&ir, self.cpu.rip)
                                } else {
                                    ir
                                };
                            self.legacy_execution.as_mut().unwrap().instruction_retry =
                                Some(InstructionRetry {
                                    stamp: LegacyContextStamp::new(&self.cpu, model),
                                    ip: self.cpu.rip,
                                    mxcsr_mode: self.cpu.mxcsr & crate::jit_abi::fp::MODE_MASK,
                                    walk: if context.paging_enabled {
                                        BlockWalk::Protected
                                    } else if context.segment_class == crate::SegmentClass::Flat32 {
                                        BlockWalk::for_flat32(&retry)
                                    } else if direct {
                                        BlockWalk::for_ir::<false>(&retry)
                                    } else {
                                        walk
                                    },
                                    ir: retry,
                                });
                        } else {
                            self.legacy_execution.as_mut().unwrap().instruction_retry = None;
                            if self
                                .legacy_execution
                                .as_ref()
                                .unwrap()
                                .pending_a20_enabled
                                .is_some()
                            {
                                self.apply_pending_a20();
                            }
                        }
                        if matches!(exit, Exit::Exception { .. }) {
                            let fault_context = if self
                                .legacy_execution
                                .as_ref()
                                .unwrap()
                                .task_switch
                                .committed
                            {
                                crate::interp::legacy::tasks::fault_context(
                                    &self.cpu,
                                    context.generation,
                                )
                            } else {
                                context
                            };
                            match self.route_legacy_exception::<ROUTED, ORDER>(
                                vm,
                                fault_context,
                                exit,
                            ) {
                                Ok(()) => {
                                    context = match self.legacy_context_for_run() {
                                        Ok(context) => context,
                                        Err(exit) => {
                                            return DispatchResult::Stepped(StepRun {
                                                steps,
                                                stop: StepStop::Exit(exit),
                                            })
                                        }
                                    };
                                    direct_bus = raw_direct(&self.cpu);
                                    if self.delivery_stalled() {
                                        return DispatchResult::Stepped(StepRun {
                                            steps,
                                            stop: StepStop::Exit(Exit::BudgetExhausted),
                                        });
                                    }
                                    continue;
                                }
                                Err(exit) => {
                                    return DispatchResult::Stepped(StepRun {
                                        steps,
                                        stop: StepStop::Exit(exit),
                                    })
                                }
                            }
                        }
                        return DispatchResult::Stepped(StepRun {
                            steps,
                            stop: StepStop::Exit(exit),
                        });
                    }
                }
            } else if KIND == ALIAS_DISPATCH {
                let pending = self.cpu.mmio_retry_active
                    || self.cpu.pending_mmio.is_some()
                    || self.cpu.pending_mmio_write
                    || self.cpu.pending_port_in.is_some();
                if STEPPED && alias_steps.remaining == 0 {
                    return DispatchResult::Stepped(StepRun {
                        steps,
                        stop: StepStop::Exit(Exit::BudgetExhausted),
                    });
                }
                if !pending && exit_word.load(Ordering::Relaxed) & EXIT_REQUESTED != 0 {
                    return DispatchResult::Stepped(StepRun {
                        steps,
                        stop: StepStop::Exit(
                            self.interrupted(crate::FaultPath::Interpreter, false),
                        ),
                    });
                }
                if let Some(exit) = self.code_space_exit(vm) {
                    return DispatchResult::Stepped(StepRun {
                        steps,
                        stop: StepStop::Exit(exit),
                    });
                }
                if !STEPPED && BOUNDED && steps >= budget {
                    return DispatchResult::Stepped(StepRun {
                        steps,
                        stop: StepStop::Exit(Exit::BudgetExhausted),
                    });
                }
                vm.handle_smc(exit_word);
                if !STEPPED && self.cpu.mmio_retry_active {
                    let mut info = crate::interp::RetireInfo {
                        consistency: vm.consistency,
                        sti_shadow: self.sti_shadow,
                        ss_shadow: self.ss_shadow & SS_HOLD != 0,
                        exit_request: Some(exit_word),
                        ..Default::default()
                    };
                    let ip = self.cpu.rip;
                    let result = crate::interp::step_one_mode::<ROUTED>(
                        &vm.mem,
                        &mut self.cpu,
                        self.mode,
                        &mut self.interp_scratch,
                        &mut info,
                    );
                    self.retired += info.retired;
                    self.sti_shadow = info.sti_shadow;
                    self.ss_shadow = u8::from(info.ss_shadow);
                    match result {
                        StepResult::Continue => {
                            if BOUNDED {
                                steps += 1;
                            }
                            continue;
                        }
                        StepResult::Exit(Exit::Syscall) => {
                            return DispatchResult::Stepped(StepRun {
                                steps,
                                stop: StepStop::Exit(
                                    self.syscall(crate::FaultPath::Interpreter, ip),
                                ),
                            })
                        }
                        StepResult::Exit(Exit::Interrupted) => {
                            return DispatchResult::Stepped(StepRun {
                                steps,
                                stop: StepStop::Exit(self.interrupted(
                                    crate::FaultPath::Helper {
                                        family: crate::FaultHelper::String,
                                        caller: crate::HelperCaller::Interpreter,
                                    },
                                    false,
                                )),
                            })
                        }
                        StepResult::Exit(exit) => {
                            return DispatchResult::Stepped(StepRun {
                                steps,
                                stop: StepStop::Exit(exit),
                            })
                        }
                    }
                }
                if self.pending_irq.is_some() && self.irq_ready() {
                    let vector = self.pending_irq.take().unwrap();
                    let ip = self.cpu.rip;
                    match crate::interp::deliver_interrupt(
                        &mut self.cpu,
                        &vm.mem,
                        ip,
                        vector,
                        ip,
                        vm.consistency,
                    )
                    .into_step()
                    {
                        StepResult::Continue => {
                            self.sti_shadow = false;
                            self.ss_shadow = 0;
                            if !STEPPED && BOUNDED {
                                steps += 1;
                            }
                            continue;
                        }
                        StepResult::Exit(exit) => {
                            return DispatchResult::Stepped(StepRun {
                                steps,
                                stop: StepStop::Exit(exit),
                            })
                        }
                    }
                }
                if STOP && self.irq_ready() {
                    return DispatchResult::Stepped(StepRun {
                        steps,
                        stop: StepStop::IrqReady,
                    });
                }
                if !STEPPED && vm.time_enabled {
                    self.record_shared_resolve(shared_reason, self.cpu.rip);
                }
                let fetch = FetchAddr::for_mode(CpuMode::Real16, self.cpu.rip, self.cpu.cs);
                let ir = match resolve_alias::<STEPPED>(
                    vm,
                    &mut self.time_dispatch,
                    exit_word,
                    &self.engine_sections,
                    fetch,
                    self.cpu.mxcsr,
                ) {
                    Ok(ir) => ir,
                    Err(exit) if crate::x87::emulated_esc(&self.cpu, self.mode, &exit) => {
                        match self.step_instruction(vm) {
                            StepResult::Continue => {
                                if !STEPPED && BOUNDED {
                                    steps += 1;
                                }
                                continue;
                            }
                            StepResult::Exit(exit) => {
                                return DispatchResult::Stepped(StepRun {
                                    steps,
                                    stop: StepStop::Exit(exit),
                                })
                            }
                        }
                    }
                    Err(exit) => {
                        return DispatchResult::Stepped(StepRun {
                            steps,
                            stop: StepStop::Exit(exit),
                        })
                    }
                };

                if !STEPPED && vm.time_enabled {
                    shared_reason = SharedReason::Interpreter;
                    crate::time_stats::interpreted();
                }
                let mut info = crate::interp::RetireInfo {
                    code_base: u64::from(self.cpu.cs) << 4,
                    consistency: vm.consistency,
                    sti_shadow: self.sti_shadow,
                    ss_shadow: self.ss_shadow & SS_HOLD != 0,
                    exit_request: Some(exit_word),
                    trace: vm.lockstep_trace.as_deref(),
                    ..Default::default()
                };
                alias_steps.pending_irq = self.pending_irq.is_some();
                let result = if STEPPED {
                    if self.cpu.mxcsr & crate::jit_abi::fp::MODE_MASK == 0 {
                        crate::interp::interpret_block_steps::<ROUTED, true, STOP>(
                            &ir,
                            &mut self.cpu,
                            &vm.mem,
                            &mut self.interp_scratch,
                            &mut info,
                            &mut alias_steps,
                        )
                    } else {
                        crate::interp::interpret_block_steps::<ROUTED, false, STOP>(
                            &ir,
                            &mut self.cpu,
                            &vm.mem,
                            &mut self.interp_scratch,
                            &mut info,
                            &mut alias_steps,
                        )
                    }
                } else if self.cpu.mxcsr & crate::jit_abi::fp::MODE_MASK == 0 {
                    crate::interp::interpret_block_fp::<ROUTED, true>(
                        &ir,
                        &mut self.cpu,
                        &vm.mem,
                        &mut self.interp_scratch,
                        &mut info,
                    )
                } else {
                    crate::interp::interpret_block_fp::<ROUTED, false>(
                        &ir,
                        &mut self.cpu,
                        &vm.mem,
                        &mut self.interp_scratch,
                        &mut info,
                    )
                };
                self.retired += info.retired;
                self.executed += info.retired;
                self.sti_shadow = info.sti_shadow;
                self.ss_shadow = u8::from(info.ss_shadow);
                steps = if STEPPED {
                    alias_steps.steps
                } else {
                    steps + 1
                };
                match result {
                    crate::interp::BlockResult::Continue
                    | crate::interp::BlockResult::ModeChange => continue,
                    crate::interp::BlockResult::Exit(exit) => match exit.into_exit() {
                        Exit::Syscall => {
                            return DispatchResult::Stepped(StepRun {
                                steps,
                                stop: StepStop::Exit(self.syscall_interpreted(&ir)),
                            })
                        }
                        Exit::Interrupted => {
                            let path = if !STEPPED || alias_steps.string_interrupted {
                                crate::FaultPath::Helper {
                                    family: crate::FaultHelper::String,
                                    caller: crate::HelperCaller::Interpreter,
                                }
                            } else {
                                crate::FaultPath::Interpreter
                            };
                            return DispatchResult::Stepped(StepRun {
                                steps,
                                stop: StepStop::Exit(self.interrupted(path, false)),
                            });
                        }
                        Exit::Exception { addr, vector, .. } => {
                            match crate::interp::deliver_interrupt(
                                &mut self.cpu,
                                &vm.mem,
                                addr,
                                vector,
                                addr,
                                vm.consistency,
                            )
                            .into_step()
                            {
                                StepResult::Continue => {
                                    self.sti_shadow = false;
                                    self.ss_shadow = 0;
                                    continue;
                                }
                                StepResult::Exit(exit) => {
                                    return DispatchResult::Stepped(StepRun {
                                        steps,
                                        stop: StepStop::Exit(exit),
                                    })
                                }
                            }
                        }
                        exit @ Exit::UnmappedMemory { .. } => {
                            let split = self.split_alias::<ROUTED, STEPPED, STOP>(
                                vm,
                                &ir,
                                exit,
                                exit_word,
                                &mut alias_steps,
                            );
                            if STEPPED {
                                steps = alias_steps.steps;
                            }
                            match split {
                                None => continue,
                                Some(stop) => {
                                    return DispatchResult::Stepped(StepRun { steps, stop })
                                }
                            }
                        }
                        exit => {
                            return DispatchResult::Stepped(StepRun {
                                steps,
                                stop: StepStop::Exit(exit),
                            })
                        }
                    },
                }
            } else {
                // The outer dispatcher may invalidate code or invoke backend callbacks.
                // Retain guest control only in the integer-only compiled chain below.
                if !DEFAULT {
                    fp_mode.default();
                }
                if dispatch & EXIT_REQUESTED != 0
                    && self.cpu.pending_mmio.is_none()
                    && !self.cpu.pending_mmio_write
                    && self.cpu.pending_port_in.is_none()
                    && self.cpu.port_string_in.is_none()
                {
                    return DispatchResult::Application(Ok(Exit::Interrupted));
                }
                // The sticky flag can be observed before the backend finishes its
                // broadcast. A late bit must not keep an already-notified vcpu in
                // the outer loop forever.
                if self.code_space_reported && dispatch & CODE_SPACE_EXHAUSTED != 0 {
                    exit_word.fetch_and(!CODE_SPACE_EXHAUSTED, Ordering::Acquire);
                }
                if let Some(exit) = self.code_space_exit(vm) {
                    return DispatchResult::Application(Ok(exit));
                }
                if BOUNDED && blocks_run >= budget {
                    return DispatchResult::Application(Ok(Exit::BudgetExhausted));
                }

                // SMC (§10): drop any cached block a prior block's store landed on,
                // before fetching the next one, so re-execution re-lifts fresh bytes.
                vm.handle_smc(exit_word);

                // Participation precedes resolving or probing any executable pointer.
                // Its scope includes cold fault decoding and every chain transfer;
                // continue/return/unwind all announce quiescence through RAII.
                let execution = participant.as_mut().unwrap().enter();

                // Flush the fast-resolve cache if any invalidation happened since we
                // last synced — covers same-thread SMC (just handled above) and
                // cross-thread SMC (another vcpu bumped the epoch). Ordered after
                // `handle_smc` so a probe never predates its own invalidation.
                let epoch = self.fast_lifetime.epoch();
                if epoch != self.fast_epoch {
                    self.fast_clear();
                    self.fast_epoch = epoch;
                    self.fast_empty_reason = PrivateMiss::Epoch;
                }

                // Resume after an MMIO trap the JIT deferred (§5.2): once the
                // embedder has supplied the read value (`complete_mmio_read`) or
                // acknowledged the write (`complete_mmio_write`), single-step the
                // faulting instruction on the interpreter — it consumes the pending
                // value/ack and advances RIP — before dispatching the next block. (Under
                // the interpreter backend this is equivalent to re-dispatching the block;
                // it just does the one faulting instruction first.)
                // A RAM fault may have consumed the pending completion while leaving
                // recorded transfers. Keep that retry on the interpreter until completion;
                // compiled blocks never inherit answers from a partially serviced operand.
                if self.cpu.mmio_retry_active {
                    if !DEFAULT {
                        fp_mode.default();
                    }
                    // Finish the pending transfer before honouring a host request at
                    // an instruction or REP element boundary (SDM Vol 2B §4.3,
                    // REP Description/Operation, pp. 4-563–4-564).
                    let mut info = crate::interp::RetireInfo {
                        consistency: vm.consistency,
                        sti_shadow: self.sti_shadow,
                        ss_shadow: self.ss_shadow & SS_HOLD != 0,
                        exit_request: Some(exit_word),
                        ..Default::default()
                    };
                    let instruction_rip = self.cpu.rip;
                    let r = crate::interp::step_one_mode::<ROUTED>(
                        &vm.mem,
                        &mut self.cpu,
                        self.mode,
                        &mut self.interp_scratch,
                        &mut info,
                    );
                    self.retired += info.retired;
                    self.sti_shadow = info.sti_shadow;
                    self.ss_shadow = u8::from(info.ss_shadow);
                    match r {
                        StepResult::Continue => {
                            if BOUNDED {
                                blocks_run += 1;
                            }
                            if self.cpu.mxcsr & crate::jit_abi::fp::MODE_MASK != mxcsr_mode {
                                return DispatchResult::Application(Err(blocks_run));
                            }
                            dispatch = exit_word.load(Ordering::Relaxed);
                            if dispatch & EXIT_REQUESTED != 0 {
                                return DispatchResult::Application(Ok(
                                    self.interrupted(crate::FaultPath::Interpreter, false)
                                ));
                            }
                            continue;
                        }
                        StepResult::Exit(Exit::Syscall) => {
                            return DispatchResult::Application(Ok(
                                self.syscall(crate::FaultPath::Interpreter, instruction_rip)
                            ));
                        }
                        StepResult::Exit(Exit::Interrupted) => {
                            return DispatchResult::Application(Ok(self.interrupted(
                                crate::FaultPath::Helper {
                                    family: crate::FaultHelper::String,
                                    caller: crate::HelperCaller::Interpreter,
                                },
                                false,
                            )));
                        }
                        StepResult::Exit(exit) => {
                            return DispatchResult::Application(Ok(crate::fault::qualify_real16(
                                exit, &self.cpu,
                            )))
                        }
                    }
                }

                // Application Long64/Compat32 use a flat fetch address. The block
                // cache and lift key on fetch.pa; compiled probes key on RIP.
                let fetch = FetchAddr::for_mode(self.mode, self.cpu.rip, self.cpu.cs);

                // Fast path: a vcpu-private probe replaces the shared cache lookup
                // for compiled blocks. A miss falls back to `resolve` and installs the
                // result; interpreted blocks are never cached here, so they always route
                // through `resolve` (keeping the cache hit/miss counters meaningful).
                let block = match self.fast_get(self.cpu.rip) {
                    Some(entry) => {
                        // Plain (non-atomic) per-vcpu counter — the whole point of the
                        // private probe is to avoid the shared atomic bumps on this path.
                        self.fast_hits = self.fast_hits.saturating_add(1);
                        CachedBlock::Compiled { entry }
                    }
                    None => {
                        if vm.time_enabled {
                            self.record_shared_resolve(shared_reason, self.cpu.rip);
                        }
                        if !DEFAULT {
                            fp_mode.default();
                        }
                        match resolve_dispatch::<DEFAULT>(
                            vm,
                            &mut self.time_dispatch,
                            exit_word,
                            &self.engine_sections,
                            fetch,
                            self.mode,
                            if DEFAULT { 0 } else { self.cpu.mxcsr },
                        ) {
                            Ok(b) => {
                                if let CachedBlock::Compiled { entry, .. } = &b {
                                    self.fast_put(self.cpu.rip, *entry);
                                }
                                b
                            }
                            Err(exit) => return DispatchResult::Application(Ok(exit)),
                        }
                    }
                };

                if let Some(exit) = self.code_space_exit(vm) {
                    return DispatchResult::Application(Ok(exit));
                }
                match block {
                    CachedBlock::Interpreted(ir) => {
                        // Owned IR needs no execution grace. A request deferred by
                        // resolution is delivered by the existing execution boundary.
                        drop(execution);
                        if vm.time_enabled {
                            shared_reason = SharedReason::Interpreter;
                        }
                        if vm.time_enabled {
                            crate::time_stats::interpreted();
                        }
                        if !DEFAULT {
                            fp_mode.default();
                        }
                        let mut info = crate::interp::RetireInfo {
                            code_base: fetch.pa - fetch.ip,
                            consistency: vm.consistency,
                            sti_shadow: self.sti_shadow,
                            ss_shadow: self.ss_shadow & SS_HOLD != 0,
                            exit_request: Some(exit_word),
                            trace: vm.lockstep_trace.as_deref(),
                            ..Default::default()
                        };
                        let r = crate::interp::interpret_block_fp::<ROUTED, DEFAULT>(
                            &ir,
                            &mut self.cpu,
                            &vm.mem,
                            &mut self.interp_scratch,
                            &mut info,
                        );
                        // Publish retirement and shadows from this interpreted block.
                        self.retired += info.retired;
                        // Same counter as compiled code charges, so the total
                        // covers whichever tier actually ran the guest.
                        self.executed += info.retired;
                        self.sti_shadow = info.sti_shadow;
                        self.ss_shadow = u8::from(info.ss_shadow);
                        match r {
                            crate::interp::BlockResult::Continue => {
                                if BOUNDED {
                                    blocks_run += 1;
                                }
                            }
                            crate::interp::BlockResult::ModeChange => {
                                return DispatchResult::Application(Err(if BOUNDED {
                                    blocks_run + 1
                                } else {
                                    0
                                }))
                            }
                            // Application exceptions return to the embedder.
                            crate::interp::BlockResult::Exit(exit) => {
                                let exit = exit.into_exit();
                                match exit {
                                    Exit::Syscall => {
                                        return DispatchResult::Application(Ok(
                                            self.syscall_interpreted(&ir)
                                        ))
                                    }
                                    Exit::Interrupted => {
                                        return DispatchResult::Application(Ok(self.interrupted(
                                            crate::FaultPath::Helper {
                                                family: crate::FaultHelper::String,
                                                caller: crate::HelperCaller::Interpreter,
                                            },
                                            false,
                                        )));
                                    }
                                    exit @ Exit::Exception { .. } => {
                                        return DispatchResult::Application(Ok(
                                            crate::interp::qualify_lifted_exit(exit, &ir),
                                        ))
                                    }
                                    exit @ Exit::UnmappedMemory { .. } => {
                                        match self.split_application::<ROUTED>(
                                            vm,
                                            &ir,
                                            exit,
                                            fetch.pa - fetch.ip,
                                            exit_word,
                                        ) {
                                            // The rest of the block completed: one block.
                                            Ok(false) => {
                                                if BOUNDED {
                                                    blocks_run += 1;
                                                }
                                            }
                                            Ok(true) => {
                                                return DispatchResult::Application(Err(
                                                    if BOUNDED { blocks_run + 1 } else { 0 },
                                                ))
                                            }
                                            Err(exit) => {
                                                return DispatchResult::Application(Ok(exit))
                                            }
                                        }
                                    }
                                    exit => return DispatchResult::Application(Ok(exit)),
                                }
                            }
                        }
                        dispatch = exit_word.load(Ordering::Relaxed);
                        if dispatch & EXIT_REQUESTED != 0 {
                            return DispatchResult::Application(Ok(
                                self.interrupted(crate::FaultPath::Interpreter, false)
                            ));
                        }
                    }
                    CachedBlock::Compiled { entry, .. } => {
                        dispatch = exit_word.load(Ordering::Relaxed);
                        if dispatch != 0 {
                            continue;
                        }
                        if vm.time_enabled {
                            crate::time_stats::compiled();
                        }
                        let mut cur = entry;
                        let (stopped, stopped_code) = loop {
                            // Hand the block its remaining block quantum (superblocks,
                            // §12 M5): a compiled region spends 1 fuel per guest block and
                            // stops at 0; a single block ignores it. Charging the fuel it
                            // consumed (min 1) keeps `blocks_run` an exact guest-block
                            // count — identical to the interpreter, preserving §9.2 and
                            // the `RunSpec::Blocks(n)` oracle.
                            let quantum = if BOUNDED {
                                budget - blocks_run
                            } else {
                                u64::MAX
                            };
                            ctx.fuel = quantum;
                            // SAFETY: `cur` is a block compiled to this ABI, alive in
                            // the JIT arena (owned by `vm`) for the call.
                            if !DEFAULT {
                                fp_mode.compiled(self.cpu.mxcsr);
                            }
                            let code = unsafe {
                                crate::jit_abi::call_block_in_mode(cur, &mut self.cpu, &mut ctx)
                            };
                            if BOUNDED {
                                blocks_run += (quantum - ctx.fuel).max(1);
                            }
                            if !DEFAULT && !matches!(code, RET_CHAIN | RET_LINK | RET_IBTC_MISS) {
                                fp_mode.default();
                            }
                            let next;
                            match code {
                                RET_CONTINUE | RET_INTERRUPTED => break (cur, code),
                                RET_RETURN_MISS => {
                                    self.time_dispatch.indirect =
                                        self.time_dispatch.indirect.saturating_add(1);
                                    break (cur, code);
                                }
                                RET_MXCSR_CHANGED => {
                                    return DispatchResult::Application(Err(blocks_run))
                                }
                                RET_CHAIN => {
                                    // Plain per-vcpu counter, flushed once per `run`
                                    // (see `TranslationCache::add_chained`): this fires
                                    // ~1M times per frame in a real game, so a shared
                                    // atomic here is a locked RMW on the hottest path.
                                    self.chained = self.chained.saturating_add(1);
                                    next = CompiledPtr::new(ctx.next_entry as *const u8);
                                }
                                RET_LINK => {
                                    // Snapshot BEFORE resolving: the publish
                                    // below only happens if the epoch is still this one, so
                                    // an SMC drop that lands while we resolve cannot have its
                                    // slot-clearing undone by our store.
                                    let since_epoch = self.fast_lifetime.epoch();
                                    if since_epoch != execution.epoch() {
                                        break (cur, code);
                                    }
                                    if !DEFAULT {
                                        fp_mode.default();
                                    }
                                    if vm.time_enabled {
                                        crate::time_stats::shared_resolve(
                                            SharedReason::DirectLink,
                                            PrivateMiss::Bypass,
                                        );
                                    }
                                    let resolved = resolve::<DEFAULT>(
                                        vm,
                                        &mut self.time_dispatch,
                                        exit_word,
                                        &self.engine_sections,
                                        FetchAddr::flat(self.cpu.rip),
                                        self.mode,
                                        if DEFAULT { 0 } else { self.cpu.mxcsr },
                                    );
                                    if let Some(exit) = self.code_space_exit(vm) {
                                        return DispatchResult::Application(Ok(exit));
                                    }
                                    match resolved {
                                        Ok(CachedBlock::Compiled { entry, .. }) => {
                                            // SAFETY: `link_slot` is a live `Arc<AtomicU64>`
                                            // in the JIT arena. Relaxed store: another vcpu
                                            // reading the slot sees 0 or a valid entry, never a
                                            // torn value (aligned u64); it pairs with the
                                            // backend's `invalidate_links` clear.
                                            let slot = ctx.link_slot;
                                            let _section = EngineSection::enter(
                                                exit_word,
                                                &self.engine_sections,
                                                CriticalSectionKind::Publish,
                                            );
                                            let published = vm.cache.publish_entry_if_current(
                                                since_epoch,
                                                entry,
                                                || unsafe {
                                                    (*(slot as *const AtomicU64))
                                                        .store(entry.0 as u64, Ordering::Relaxed)
                                                },
                                            );
                                            drop(_section);
                                            if !published {
                                                // Invalidated under us. Publishing now would
                                                // refill a slot the drop just cleared, and
                                                // transferring would run a translation of code
                                                // the guest has rewritten. Back to the outer
                                                // loop, which runs `handle_smc` and re-resolves.
                                                break (cur, code);
                                            }
                                            // Seed the fast-resolve cache too: the next
                                            // outer-loop visit to this RIP skips `resolve`.
                                            self.fast_put(self.cpu.rip, entry);
                                            next = entry;
                                        }
                                        // Mixed backend can't chain — fall back to dispatch.
                                        Ok(CachedBlock::Interpreted(_)) => break (cur, code),
                                        Err(exit) => return DispatchResult::Application(Ok(exit)),
                                    }
                                }
                                // IBTC miss: an indirect edge whose per-site slot was
                                // empty or held a different target. Resolve the computed
                                // RIP and refill the slot with a fresh {target, entry}
                                // descriptor, unless the site is megamorphic.
                                RET_IBTC_MISS => {
                                    self.time_dispatch.indirect =
                                        self.time_dispatch.indirect.saturating_add(1);
                                    #[cfg(test)]
                                    if let Some(hook) = self.ibtc_test_hook.as_mut() {
                                        hook(false);
                                    }
                                    // Generation validation rejects retired tracked entries,
                                    // but retaining backends may use generation-zero pointers.
                                    // Keep the private fill epoch check for those entries and
                                    // for capped refills that skip slot publication entirely.
                                    let since_epoch = self.fast_lifetime.epoch();
                                    if since_epoch != execution.epoch()
                                        || since_epoch != self.fast_epoch
                                    {
                                        break (cur, code);
                                    }
                                    // Probe the vcpu-private fast cache before `resolve`.
                                    // The per-site IBTC holds ONE target, so a site
                                    // with several live targets misses here on nearly every
                                    // call — and `resolve` is an RwLock read plus a hash
                                    // lookup and a clone, where this probes two private
                                    // tags without synchronization.
                                    // Same key, same epoch invariant and same "compiled
                                    // blocks only" content as the probe the outer dispatch
                                    // loop already trusts, so a hit here is equally sound;
                                    // it skips the tier-up drain exactly as that one does.
                                    let entry = match self.fast_get(self.cpu.rip) {
                                        Some(entry) => {
                                            self.fast_hits = self.fast_hits.saturating_add(1);
                                            Some(entry)
                                        }
                                        None => {
                                            if vm.time_enabled {
                                                self.record_shared_resolve(
                                                    SharedReason::IbtcMiss,
                                                    self.cpu.rip,
                                                );
                                            }
                                            if !DEFAULT {
                                                fp_mode.default();
                                            }
                                            let resolved = resolve::<DEFAULT>(
                                                vm,
                                                &mut self.time_dispatch,
                                                exit_word,
                                                &self.engine_sections,
                                                FetchAddr::flat(self.cpu.rip),
                                                self.mode,
                                                if DEFAULT { 0 } else { self.cpu.mxcsr },
                                            );
                                            if let Some(exit) = self.code_space_exit(vm) {
                                                return DispatchResult::Application(Ok(exit));
                                            }
                                            match resolved {
                                                Ok(CachedBlock::Compiled { entry }) => {
                                                    self.fast_put(self.cpu.rip, entry);
                                                    Some(entry)
                                                }
                                                // Indirect edge into an interpreted block.
                                                Ok(CachedBlock::Interpreted(_)) => None,
                                                Err(exit) => {
                                                    return DispatchResult::Application(Ok(exit))
                                                }
                                            }
                                        }
                                    };
                                    #[cfg(test)]
                                    if let Some(hook) = self.ibtc_test_hook.as_mut() {
                                        hook(true);
                                    }
                                    // Even a fast hit can allocate during refill bookkeeping.
                                    if !DEFAULT {
                                        fp_mode.default();
                                    }
                                    match entry {
                                        Some(entry) => {
                                            let slot = ctx.link_slot;
                                            // A full map reallocates on insert: account only then,
                                            // not on every new refill entry.
                                            let full = self.ibtc_refills.len()
                                                >= self.ibtc_refills.capacity();
                                            let count = self.ibtc_refills.entry(slot).or_insert(0);
                                            let under_cap = *count < IBTC_MEGAMORPHIC_CAP;
                                            if under_cap {
                                                *count += 1;
                                            }
                                            if full {
                                                self.allocations.refills.store(
                                                    (self.ibtc_refills.capacity()
                                                        * std::mem::size_of::<(u64, u32)>())
                                                        as u64,
                                                    Ordering::Relaxed,
                                                );
                                            }
                                            if under_cap {
                                                let _section = EngineSection::enter(
                                                    exit_word,
                                                    &self.engine_sections,
                                                    CriticalSectionKind::Publish,
                                                );
                                                let desc = vm
                                                    .cache
                                                    .alloc_ibtc_descriptor(self.cpu.rip, entry);
                                                // SAFETY: `slot` is a live `Box<AtomicU64>` in
                                                // the JIT arena; the published descriptor is
                                                // immutable and retained through execution grace.
                                                // Release (not Relaxed): unlike the RET_LINK
                                                // slot — a single scalar entry — this publishes
                                                // a POINTER to a multi-field {target, entry}
                                                // payload, so the payload's writes must be
                                                // ordered-visible before the pointer. Release
                                                // here pairs with the reader's address
                                                // dependency (the compiled `ibtc_or_miss` loads
                                                // the descriptor fields *through* this pointer),
                                                // giving release/consume ordering; a plain
                                                // Relaxed store would let a weakly-ordered host
                                                // (AArch64) expose the pointer before the fields.
                                                let published = vm.cache.publish_entry_if_current(
                                                    since_epoch,
                                                    entry,
                                                    || unsafe {
                                                        (*(slot as *const AtomicU64))
                                                            .store(desc, Ordering::Release)
                                                    },
                                                );
                                                if !published {
                                                    // Invalidated under us — see RET_LINK.
                                                    break (cur, code);
                                                }
                                            }
                                            if !under_cap && vm.cache.epoch() != since_epoch {
                                                break (cur, code);
                                            }
                                            next = entry;
                                        }
                                        // Interpreted target — leave the chain and dispatch.
                                        None => break (cur, code),
                                    }
                                }
                                RET_SYSCALL => {
                                    let path = crate::codemap::fault_path(
                                        cur.0 as usize,
                                        crate::Knowledge::Known(ctx.next_entry),
                                        false,
                                    );
                                    return DispatchResult::Application(Ok(
                                        self.syscall(path, ctx.next_entry)
                                    ));
                                }
                                RET_HLT => return DispatchResult::Application(Ok(Exit::Hlt)),
                                RET_UNMAPPED => {
                                    return DispatchResult::Application(Ok(
                                        ctx.unmapped_exit(&self.cpu, self.mode)
                                    ))
                                }
                                RET_UNMAPPED_DETAIL => {
                                    return DispatchResult::Application(Ok(ctx
                                        .detailed_unmapped_exit(
                                            &self.cpu,
                                            &vm.mem,
                                            self.mode,
                                            cur.0 as usize,
                                        )))
                                }
                                RET_EXCEPTION_DETAIL => {
                                    let exit = ctx.detailed_exception_exit(
                                        &self.cpu,
                                        &vm.mem,
                                        self.mode,
                                        cur.0 as usize,
                                    );
                                    return DispatchResult::Application(Ok(
                                        crate::fault::qualify_lifted(
                                            exit,
                                            crate::codemap::lookup_fault_info(
                                                cur.0 as usize,
                                                ctx.next_entry,
                                            )
                                            .as_ref(),
                                        ),
                                    ));
                                }
                                RET_MMIO_DEFER | RET_HOST_FAULT | RET_HOST_FAULT_DETAIL => {
                                    // Alignment recovery has no reliable data address.
                                    if code == RET_MMIO_DEFER
                                        || self.fault_entry.fault_kind()
                                            != crate::jit_abi::fault::HostFaultKind::Alignment
                                    {
                                        let pc = if code == RET_HOST_FAULT_DETAIL {
                                            ctx.next_entry as usize
                                        } else {
                                            cur.0 as usize
                                        };
                                        if let Some(info) =
                                            crate::codemap::lookup_fault_info(pc, self.cpu.rip)
                                        {
                                            info.note_partial_store(
                                                &self.cpu,
                                                &vm.mem,
                                                ctx.fault_addr,
                                            );
                                        }
                                    }
                                    let raw_fault = if code == RET_HOST_FAULT_DETAIL {
                                        self.fault_entry.raw_fault()
                                    } else {
                                        None
                                    };
                                    if code == RET_HOST_FAULT || code == RET_HOST_FAULT_DETAIL {
                                        if DEFAULT {
                                            // The landing stub may have restored non-default caller
                                            // control. No guest mode was installed in this specialization.
                                            // SAFETY: normalize before fault decoding and replay.
                                            unsafe {
                                                crate::jit_abi::fp::set_control(
                                                    crate::jit_abi::fp::default_control(),
                                                )
                                            };
                                        } else {
                                            fp_mode.host_fault();
                                        }
                                        if self.fault_entry.fault_kind()
                                            != crate::jit_abi::fault::HostFaultKind::Alignment
                                            && vm.mem.mapped_ram(ctx.fault_addr)
                                        {
                                            let exit = if code == RET_HOST_FAULT_DETAIL {
                                                ctx.detailed_host_exit(
                                                    &self.cpu, &vm.mem, self.mode,
                                                )
                                            } else {
                                                ctx.unmapped_exit(&self.cpu, self.mode)
                                            };
                                            return DispatchResult::Application(Ok(
                                                crate::fault::attach_raw_host_fault(
                                                    exit, raw_fault,
                                                ),
                                            ));
                                        }
                                    }
                                    if !DEFAULT {
                                        fp_mode.default();
                                    }
                                    // Trap-region accesses replay to produce MMIO exits.
                                    // Host faults reach replay only for unmapped addresses
                                    // or alignment faults. A misaligned locked operand
                                    // replays atomically inside one aligned 16-byte granule,
                                    // not across one (`Memory::locked_unaligned`).
                                    // Replay uses this run's memory policy in both modes.
                                    // Interpreter single-step (not the hot compiled loop):
                                    // tick the retired counter for the one instruction it
                                    // may retire (§17.6, sub-seam c).
                                    let mut info = crate::interp::RetireInfo {
                                        consistency: vm.consistency,
                                        exit_request: Some(exit_word),
                                        trace: vm.lockstep_trace.as_deref(),
                                        ..Default::default()
                                    };
                                    let original_info = if code == RET_HOST_FAULT_DETAIL {
                                        crate::codemap::lookup_fault_info(
                                            ctx.next_entry as usize,
                                            self.cpu.rip,
                                        )
                                    } else {
                                        None
                                    };
                                    let instruction_rip = self.cpu.rip;
                                    let r = crate::time_stats::HelperTimer::measure(
                                        vm.time_enabled,
                                        || {
                                            crate::interp::step_one_mode::<ROUTED>(
                                                &vm.mem,
                                                &mut self.cpu,
                                                self.mode,
                                                &mut self.interp_scratch,
                                                &mut info,
                                            )
                                        },
                                    );
                                    self.retired += info.retired;
                                    match r {
                                        StepResult::Continue => {
                                            if self.cpu.mxcsr & crate::jit_abi::fp::MODE_MASK
                                                != mxcsr_mode
                                            {
                                                return DispatchResult::Application(Err(
                                                    blocks_run,
                                                ));
                                            }
                                            break (cur, code);
                                        }
                                        StepResult::Exit(Exit::Syscall) => {
                                            return DispatchResult::Application(Ok(self.syscall(
                                                crate::FaultPath::Interpreter,
                                                instruction_rip,
                                            )));
                                        }
                                        StepResult::Exit(Exit::Interrupted) => {
                                            return DispatchResult::Application(Ok(self
                                                .interrupted(
                                                    crate::FaultPath::Helper {
                                                        family: crate::FaultHelper::String,
                                                        caller: crate::HelperCaller::Interpreter,
                                                    },
                                                    false,
                                                )));
                                        }
                                        StepResult::Exit(exit) => {
                                            return DispatchResult::Application(Ok(
                                                if code == RET_HOST_FAULT
                                                    || code == RET_HOST_FAULT_DETAIL
                                                {
                                                    crate::fault::attach_raw_host_fault(
                                                        ctx.preserve_host_replay(
                                                            exit,
                                                            original_info.as_ref(),
                                                            code == RET_HOST_FAULT_DETAIL,
                                                        ),
                                                        raw_fault,
                                                    )
                                                } else {
                                                    exit
                                                },
                                            ));
                                        }
                                    }
                                }
                                // A port-I/O instruction (`in`/`out`, §5.2): the block set
                                // RIP to the instruction; single-step it on the interpreter
                                // to produce the `Exit::PortIo` (same deferral as MMIO).
                                RET_PORTIO_DEFER => {
                                    if !DEFAULT {
                                        fp_mode.default();
                                    }
                                    let mut info = crate::interp::RetireInfo {
                                        consistency: vm.consistency,
                                        exit_request: Some(exit_word),
                                        trace: vm.lockstep_trace.as_deref(),
                                        ..Default::default()
                                    };
                                    let instruction_rip = self.cpu.rip;
                                    let r = crate::time_stats::HelperTimer::measure(
                                        vm.time_enabled,
                                        || {
                                            crate::interp::step_one_mode::<ROUTED>(
                                                &vm.mem,
                                                &mut self.cpu,
                                                self.mode,
                                                &mut self.interp_scratch,
                                                &mut info,
                                            )
                                        },
                                    );
                                    self.retired += info.retired;
                                    match r {
                                        StepResult::Continue => break (cur, code),
                                        StepResult::Exit(Exit::Syscall) => {
                                            return DispatchResult::Application(Ok(self.syscall(
                                                crate::FaultPath::Interpreter,
                                                instruction_rip,
                                            )));
                                        }
                                        StepResult::Exit(exit) => {
                                            return DispatchResult::Application(Ok(exit))
                                        }
                                    }
                                }
                                // A guest exception (`#DE` div, or a lifted `ud2`/`int3`/
                                // `int1` trap); the block set the saved RIP (fault: on the
                                // instruction, trap: past it) and stored the vector in the
                                // MemCtx out-field.
                                RET_EXCEPTION => {
                                    return DispatchResult::Application(Ok(Exit::exception(
                                        self.cpu.rip,
                                        ctx.exception_vector as u8,
                                    )))
                                }
                                other => panic!("compiled block returned invalid ABI code {other}"),
                            }
                            if BOUNDED && blocks_run >= budget {
                                if !DEFAULT {
                                    fp_mode.default();
                                }
                                return DispatchResult::Application(Ok(Exit::BudgetExhausted));
                            }
                            // Poll before replacing the just-executed unit with its
                            // successor. Only the cold edge needs its identity.
                            if exit_word.load(Ordering::Relaxed) != 0
                                || self.fast_lifetime.epoch() != execution.epoch()
                            {
                                break (cur, code);
                            }
                            cur = next;
                        };
                        if vm.time_enabled {
                            shared_reason = match stopped_code {
                                RET_RETURN_MISS => SharedReason::ReturnMiss,
                                RET_LINK => SharedReason::DirectLink,
                                RET_IBTC_MISS => SharedReason::IbtcMiss,
                                _ => SharedReason::Boundary,
                            };
                        }
                        dispatch = exit_word.load(Ordering::Relaxed);
                        if dispatch & EXIT_REQUESTED != 0 {
                            if !DEFAULT {
                                fp_mode.default();
                            }
                            return DispatchResult::Application(Ok(
                                self.interrupted_compiled(stopped, stopped_code)
                            ));
                        }
                    }
                }
            }
        }
    }
}

impl Vcpu {
    fn deliver_event20<const ROUTED: bool, const ORDER: u8>(
        &mut self,
        vm: &Vm,
        mut event: PendingEvent,
    ) -> Result<(), Exit> {
        // 8086 Family UM §2.6 Table 2-3 p. 2-26 and Single-Step p. 2-28:
        // enter the external handler before recognizing the single-step trap.
        if let Some((cs, owner, restart)) =
            self.legacy_execution.as_mut().unwrap().rep_restart20.take()
        {
            if cs == event.cs && u32::from(owner) == event.ip {
                event.ip = u32::from(restart);
            }
        }
        // Each catalogue entry supplies its own delivery policy (decision-23 §2).
        let profile = self
            .cpu_model
            .expect("twenty-bit dispatch requires an 8086-family catalogue model")
            .profile();
        let quirks = profile.quirks;
        let word_bus = crate::lift::i8086::word_bus(profile);
        // spec.md §17.10: a single-step trap meets an acceptable NMI here,
        // before the boundary's NMI step.
        let step_nmi = matches!(event.source, EventSource::SingleStep)
            && self.exit_request.word.load(Ordering::Relaxed) & NMI_EDGE != 0
            && self.nmi_gate(false);
        if quirks.step_irq_conflict_refused
            && matches!(event.source, EventSource::SingleStep)
            && step_nmi
        {
            // NEC UM Table 6-1 p. 6-1 gives priorities, not the sequence.
            self.legacy_execution.as_mut().unwrap().event_retry = Some(event);
            return Err(Exit::UnsupportedSystemContext {
                addr: self.cpu.rip,
                operation: UnsupportedSystemOperation::NecStepWithNmi,
            });
        }
        if quirks.step_irq_conflict_refused
            && matches!(event.source, EventSource::SingleStep)
            && self.pending_irq.is_some()
            && self.irq_delivery_ready()
        {
            self.legacy_execution.as_mut().unwrap().event_retry = Some(event);
            return Err(Exit::UnsupportedSystemContext {
                addr: self.cpu.rip,
                operation: UnsupportedSystemOperation::NecStepWithInterrupt,
            });
        }
        let mut trap = matches!(event.source, EventSource::SingleStep)
            || (quirks.step_interrupt_handler && self.cpu.rflags_system & 0x100 != 0);
        // 8086: an exception taken with TF meets an acceptable NMI after its
        // own frame, once that delivery has ended the faulting instruction's
        // shadows.
        let mut nmi_after = quirks.step_interrupt_handler
            && trap
            && matches!(event.source, EventSource::Exception { .. });
        if quirks.step_interrupt_handler && step_nmi {
            // 8086 Family UM §2.6 p. 2-28 ("either internal or external");
            // AP-67 Fig. 3E3a p. A-29: the NMI frame, then the step frame, so
            // the step routine returns into the unstepped NMI handler. NMI
            // wins over a queued IRQ, which waits behind IF = 0. After a
            // software interrupt, AP-67 p. A-28 and Fig. 3E3b p. A-29: no step
            // frame; stepping resumes after the interrupting instruction.
            if self.accept_nmi() {
                event.source = EventSource::Nmi { shutdown: false };
                trap = !event.software;
            }
        } else if quirks.step_interrupt_handler
            && matches!(event.source, EventSource::SingleStep)
            && self.irq_delivery_ready()
        {
            if let Some(vector) = self.pending_irq {
                event.source = EventSource::Irq(vector);
            }
        }
        loop {
            let vector = match event.source {
                EventSource::SingleStep => 1,
                EventSource::Irq(vector) | EventSource::Exception { vector, .. } => vector,
                // 8086 Family UM §2.6 p. 2-24, Fig. 2-30 p. 2-25; NEC UM
                // Table 6-1 p. 6-1: vector 2, no INTA cycle.
                EventSource::Nmi { .. } => 2,
            };
            // NEC V20/V30 UM §6 p. 6-1, §8 p. 8-4: an interrupt taken in 8080
            // mode pushes the PSW with MD = 0, then sets MD = 1; standby ends.
            let emulation = self.cpu.legacy.reserved & crate::legacy::NEC_EMULATION != 0;
            let flags = match crate::interp::i8086::interrupt::<ROUTED, ORDER>(
                &mut self.cpu,
                &vm.mem,
                vector,
                event.ip as u16,
                if emulation {
                    quirks.real_flags_set_mask & 0x7fff
                } else {
                    quirks.real_flags_set_mask
                },
                word_bus,
            ) {
                Ok(flags) => flags,
                Err(exit) => {
                    self.legacy_execution.as_mut().unwrap().event_retry = Some(event);
                    return Err(exit);
                }
            };
            if self.delivery_log.is_some() {
                self.record_delivery(
                    vector,
                    delivery_kind(event.source),
                    event.origin,
                    None,
                    Interrupted::Frame20 { flags },
                );
            }
            if emulation {
                self.cpu.legacy.reserved &=
                    !(crate::legacy::NEC_EMULATION | crate::legacy::NEC_STANDBY);
                let execution = self.legacy_execution.as_mut().unwrap();
                execution.nec_escape = None;
                execution.nec_refusal = None;
                execution.nec_entry = None;
            }
            let stepped = matches!(event.source, EventSource::SingleStep);
            self.finish_legacy_event(event, None);
            if std::mem::take(&mut nmi_after)
                && self.exit_request.word.load(Ordering::Relaxed) & NMI_EDGE != 0
                && self.nmi_gate(false)
                && self.accept_nmi()
            {
                // AP-67 p. A-28, Fig. 3E3b p. A-29: the exception's frame,
                // then the NMI frame, and no step frame.
                trap = false;
                event = PendingEvent {
                    irq_consumed: false,
                    software: false,
                    after_nmi: false,
                    source: EventSource::Nmi { shutdown: false },
                    ip: self.cpu.rip as u32,
                    cs: self.cpu.cs,
                    origin: self.delivery_origin(self.cpu.cs, self.cpu.rip as u32),
                };
            } else if trap && !stepped {
                event = PendingEvent {
                    irq_consumed: false,
                    software: false,
                    after_nmi: false,
                    source: EventSource::SingleStep,
                    ip: self.cpu.rip as u32,
                    cs: self.cpu.cs,
                    origin: self.delivery_origin(self.cpu.cs, self.cpu.rip as u32),
                };
            } else {
                return Ok(());
            }
        }
    }
}

#[cfg(test)]
#[allow(deprecated)] // Compatibility selector witnesses.
mod tests {
    use super::*;

    #[test]
    fn v86_boundary_restores_reuse_retained_checked_contexts() {
        // V86 design §4.1/Q3: full canonical images form checked contexts;
        // completed run boundaries retain IR, while context edits isolate it.
        for model in [
            crate::CpuModel::I80386,
            crate::CpuModel::I80486Dx,
            crate::CpuModel::I80486DxCpuid,
        ] {
            let mut vm = Vm::new(VmConfig::flat(0x10000));
            vm.set_cpu_model(model);
            vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0x1000, &[0xeb, 0xfe]).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu = CpuState::legacy_real_entry([0; 6]);
            cpu.cpu.rip = 0x1000;
            cpu.cpu.legacy.cr0 = 1;
            cpu.cpu.legacy.cpl = 3;
            cpu.cpu.rflags_system = (1 << 17) | 0x3000;
            let checkpoint = cpu.cpu.clone();
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1, "{:?}", run.stop());
            let execution = cpu.legacy_execution.as_ref().unwrap();
            assert_eq!(execution.blocks.len(), 1);
            let (key, ir) = execution
                .blocks
                .iter()
                .next()
                .map(|(key, block)| (*key, block.ir.clone()))
                .unwrap();
            for budget in [1, 2, 17] {
                let run = cpu.run_steps(&vm, budget, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), budget, "{:?}", run.stop());
                assert_eq!(cpu.legacy_execution.as_ref().unwrap().lifts, 1);
            }
            cpu.cpu.ds = 0x100;
            cpu.cpu.legacy.segments[SegmentId::Ds as usize].base = 0x1000;
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1, "{:?}", run.stop());
            assert_eq!(cpu.legacy_execution.as_ref().unwrap().lifts, 2);
            cpu.cpu = checkpoint;
            cpu.invalidate_paging();
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1, "{:?}", run.stop());
            let execution = cpu.legacy_execution.as_ref().unwrap();
            assert_eq!(execution.lifts, 2);
            assert!(Arc::ptr_eq(&ir, &execution.blocks[&key].ir));
        }
    }

    #[test]
    fn cached_direct_bus_revalidates_selectors_and_a20() {
        // SDM Vol 3A §11.7.13.4; decision-22 §3: the board gate masks bit 20;
        // entry edits must invalidate both context and bus-derived metadata.
        for model in [crate::CpuModel::I80286, crate::CpuModel::I80386] {
            let mut vm = Vm::new(VmConfig::flat(0x120000));
            vm.set_cpu_model(model);
            vm.map(0, 0x120000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0x1000, b"\xa1\x20\x00\xeb\xfb").unwrap();
            vm.mem.write(0x20, 0x1111, 2).unwrap();
            vm.mem.write(0x10, 0x2222, 2).unwrap();
            vm.mem.write(0x100010, 0x3333, 2).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu = CpuState::legacy_real_entry([0; 6]);
            for (selector, enabled, direct, value) in [
                (0, true, true, 0x1111),
                (0xffff, false, false, 0x2222),
                (0xffff, true, true, 0x3333),
                (0, false, true, 0x1111),
            ] {
                cpu.cpu.ds = selector;
                cpu.cpu.legacy.segments[SegmentId::Ds as usize].base = u32::from(selector) << 4;
                cpu.set_a20_enabled(enabled);
                cpu.cpu.rip = 0x1000;
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 1, "{run:?}");
                assert!(matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)));
                assert_eq!(cpu.cpu.gpr[0], value);
                assert_eq!(
                    cpu.legacy_execution.as_ref().unwrap().validated_direct_bus,
                    direct
                );
            }
        }
    }

    #[test]
    // SDM Vol 2A §3.3 INC: CF is preserved; decision-16 §3:
    // a completed input retires once before the next budget boundary.
    fn partial_shared_budget_completion_keeps_mmio_cleanup() {
        for model in [crate::CpuModel::I80286, crate::CpuModel::I80386] {
            let mut vm = Vm::new(VmConfig::flat(0x10000));
            vm.set_cpu_model(model);
            vm.map(0, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
            vm.write_bytes(0x1000, b"\xa1\x00\x30\x40\xeb\xfa").unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu = CpuState::legacy_real_entry([0; 6]);
            cpu.cpu.rip = 0x1000;
            cpu.cpu.flags.cf = true;
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 0);
            assert!(matches!(
                run.stop(),
                StepStop::Exit(Exit::MmioRead {
                    addr: 0x3000,
                    size: 2,
                    ..
                })
            ));
            cpu.complete_mmio_read(0x1234);
            // Select the authoritative multi-instruction block for completion,
            // exercising its budget boundary instead of a narrowed retry slice.
            cpu.legacy_execution.as_mut().unwrap().instruction_retry = None;
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1);
            assert!(matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)));
            assert_eq!(cpu.cpu.gpr[0], 0x1234);
            assert_eq!(cpu.cpu.rip, 0x1003);
            assert!(!cpu.cpu.mmio_retry_active);
            assert_eq!(cpu.cpu.mmio_parts_len, 0);
            assert_eq!(cpu.retired_instructions(), 1);
            assert_eq!(
                cpu.run_steps(&vm, 1, StepOptions::default())
                    .unwrap()
                    .steps(),
                1
            );
            assert_eq!(cpu.cpu.gpr[0], 0x1235);
            assert!(cpu.cpu.flags.cf);
            assert_eq!(cpu.retired_instructions(), 2);
        }
    }

    #[test]
    fn replaced_direct_arc_revalidates_its_context_before_reusing_a_walk() {
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        vm.set_cpu_model(crate::CpuModel::I80386);
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, b"\xb8\x01\x00\xeb\xfb").unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x1000;
        assert_eq!(
            cpu.run_steps(&vm, 2, StepOptions::default())
                .unwrap()
                .steps(),
            2
        );
        let key = BlockKey::machine_direct(0x1000, 0, true);
        let CachedBlock::Interpreted(valid) = vm.cache.get(key).unwrap() else {
            unreachable!()
        };
        assert!(cpu
            .legacy_execution
            .as_ref()
            .unwrap()
            .direct_proofs
            .get(&valid, 0x1000)
            .is_some());
        let mut wrong = (*valid).clone();
        wrong.legacy_context.as_mut().unwrap().generation = CpuGeneration::I286;
        let mut changed = false;
        for op in &mut wrong.ops {
            if let IrOp::WriteReg { src, .. } = op {
                *src = crate::ir::Val::Imm(0xbad);
                changed = true;
            }
        }
        assert!(changed);
        let len = wrong.guest_len;
        vm.cache.insert(
            key,
            CachedBlock::Interpreted(Arc::new(wrong)),
            vec![(0x1000, len)],
            |_| vm.mem.mark_code(0x1000, len),
        );
        cpu.cpu.rip = 0x1000;
        let (resolved, _) = resolve_direct(
            &vm,
            &cpu.cpu,
            valid.legacy_context.unwrap(),
            &cpu.exit_request.word,
            0x1000,
            &mut cpu.legacy_execution.as_mut().unwrap().direct_proofs,
        )
        .unwrap();
        assert_eq!(
            resolved.legacy_context.unwrap().generation,
            CpuGeneration::I386
        );
        assert_eq!(
            cpu.run_steps(&vm, 2, StepOptions::default())
                .unwrap()
                .steps(),
            2
        );
        assert_eq!(cpu.cpu.gpr[0], 1);
        let CachedBlock::Interpreted(revalidated) = vm.cache.get(key).unwrap() else {
            unreachable!()
        };
        assert_eq!(
            revalidated.legacy_context.unwrap().generation,
            CpuGeneration::I386
        );
    }

    #[test]
    fn warm_stamp_preserves_all_cache_fields_and_output_exclusions() {
        let cpu = CpuState::legacy_real_entry([1, 2, 3, 4, 5, 6]);
        let model = LegacyCpuModel {
            generation: CpuGeneration::I386,
            model_semantics_id: 17,
        };
        let stamp = LegacyContextStamp::new(&cpu, model);
        assert!(stamp.matches(&cpu, model));
        for segment in 0..8 {
            for field in 0..5 {
                for bit in 0..32 {
                    let mut changed = cpu.clone();
                    let cache = match segment {
                        0..=5 => &mut changed.legacy.segments[segment],
                        6 => &mut changed.legacy.ldtr.cache,
                        _ => &mut changed.legacy.tr.cache,
                    };
                    match field {
                        0 => cache.base ^= 1 << bit,
                        1 => cache.limit ^= 1 << bit,
                        2 => cache.attributes ^= (1u32 << bit) as u16,
                        3 => cache.unusable ^= (1u32 << bit) as u8,
                        _ => cache.reserved ^= (1u32 << bit) as u8,
                    }
                    assert_eq!(
                        stamp.matches(&changed, model),
                        stamp == LegacyContextStamp::new(&changed, model),
                        "segment {segment} field {field} bit {bit}"
                    );
                }
            }
        }
        let mut changed = cpu.clone();
        changed.legacy.cr2 = u32::MAX;
        changed.rflags_system ^= !0x27000;
        assert!(stamp.matches(&changed, model));
        for edit in [
            |c: &mut CpuState| c.legacy.cr0 ^= 1,
            |c: &mut CpuState| c.legacy.cr3 ^= 1,
            |c: &mut CpuState| c.legacy.gdtr.reserved ^= 1,
            |c: &mut CpuState| c.legacy.idtr.reserved ^= 1,
            |c: &mut CpuState| c.legacy.ldtr.selector ^= 1,
            |c: &mut CpuState| c.legacy.tr.selector ^= 1,
            |c: &mut CpuState| c.legacy.cpl ^= 1,
            |c: &mut CpuState| c.legacy.a20_enabled ^= 1,
            |c: &mut CpuState| c.legacy.reset_fetch ^= 1,
            |c: &mut CpuState| c.legacy.reserved ^= 1,
            |c: &mut CpuState| c.cs ^= 1,
            |c: &mut CpuState| c.gs ^= 1,
            |c: &mut CpuState| c.rflags_system ^= 0x20000,
        ] {
            let mut changed = cpu.clone();
            edit(&mut changed);
            assert!(!stamp.matches(&changed, model));
        }
    }

    #[test]
    fn redirected_event_discards_every_private_boundary_before_fetch() {
        for model in [crate::CpuModel::I80286, crate::CpuModel::I80386] {
            let mut vm = Vm::new(VmConfig::flat(0x10000));
            vm.set_cpu_model(model);
            vm.map(0, 0x2000, crate::Prot::RWX, crate::RegionKind::Ram)
                .unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu = CpuState::legacy_real_entry([0; 6]);
            cpu.cpu.rip = 0x3000;
            cpu.cpu.flags.if_ = true;
            cpu.sti_shadow = true;
            cpu.ss_shadow = SS_HOLD | NMI_HOLD;
            let execution = cpu.legacy_execution.as_mut().unwrap();
            execution.handoff_service |= HANDOFF_DEBUG;
            execution.event_retry = Some(PendingEvent {
                irq_consumed: false,
                software: false,
                after_nmi: false,
                source: EventSource::SingleStep,
                ip: 0x1000,
                cs: 0,
                origin: Origin::default(),
            });
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 0, "{:?}", run.stop());
            assert!(
                matches!(
                    run.stop(),
                    StepStop::Exit(Exit::UnmappedMemory { addr: 0x3000, .. })
                ),
                "{:?}",
                run.stop()
            );
            assert!(!cpu.sti_shadow);
            assert_eq!(cpu.ss_shadow, 0);
            let execution = cpu.legacy_execution.as_ref().unwrap();
            assert_eq!(execution.handoff_service & HANDOFF_DEBUG, 0);
            assert!(execution.event_retry.is_none());
            assert!(cpu.irq_ready());
        }
    }

    #[test]
    fn shared_direct_fault_does_not_inherit_a_completed_task_commit() {
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        vm.set_cpu_model(crate::CpuModel::I80386);
        vm.set_exception_delivery(crate::ExceptionDelivery::InGuest);
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, b"\xa3\xff\xff").unwrap();
        vm.write_bytes(13 * 4, &[0x00, 0x20, 0, 0]).unwrap();
        vm.write_bytes(0x2000, &[0xf4]).unwrap();
        let mut cpu = vm.new_vcpu();
        // An embedder may return to ordinary real mode after a completed switch.
        cpu.cpu = CpuState::legacy_real_entry([0; 6]);
        cpu.cpu.rip = 0x1000;
        cpu.cpu.gpr[4] = 0x9000;
        let task = &mut cpu.legacy_execution.as_mut().unwrap().task_switch;
        task.committed = true;
        task.completed = true;
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(matches!(run.stop(), StepStop::Exit(Exit::Hlt)), "{run:?}");
        assert_eq!(run.steps(), 1);
        assert_eq!(cpu.cpu.rip, 0x2001);
        assert_eq!(vm.mem.read(0x8ffa, 2).unwrap(), 0x1000);
    }

    #[test]
    fn direct_publication_rejects_a_source_write_before_registration() {
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        vm.set_cpu_model(crate::CpuModel::I80386);
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, b"\xb8\x01\x00\xf4").unwrap();
        *vm.source_read_hook.lock().unwrap() = Some(|vm| {
            vm.write_bytes(0x1001, b"\x02").unwrap();
        });
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x1000;
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 1);
        assert_eq!(cpu.cpu.gpr[0], 2, "a source write must cancel publication");
    }

    #[test]
    fn smc_retains_instruction_contexts_but_nearby_data_keeps_shared_blocks() {
        for (code, smc) in [
            (&b"\xc6\x06\x06\x10\x2a\xb0\x00\xeb\xf7"[..], true),
            (&b"\xa3\x80\x10\xeb\xfb"[..], false),
        ] {
            let mut vm = Vm::new(VmConfig::flat(0x10000));
            vm.set_cpu_model(crate::CpuModel::I80386);
            vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0x1000, code).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu.rip = 0x1000;
            for _ in 0..2 {
                assert_eq!(
                    cpu.run_steps(&vm, 60, StepOptions::default())
                        .unwrap()
                        .steps(),
                    60
                );
                let execution = cpu.legacy_execution.as_ref().unwrap();
                assert_eq!(execution.smc_pages.is_empty(), !smc);
                assert_eq!(execution.lifts, if smc { 3 } else { 0 });
                assert_eq!(execution.blocks.len(), if smc { 3 } else { 0 });
                if smc {
                    assert_eq!(cpu.cpu.gpr[0] & 0xff, 42);
                }
            }
            cpu.reset_execution_state(crate::ResetOptions::default());
            assert!(cpu.legacy_execution.as_ref().unwrap().smc_pages.is_empty());
        }
    }

    #[test]
    fn smc_direct_scalar_fault_is_restartable_without_a_checkpoint() {
        // SDM Vol 3A §7.5: a fault restores the instruction's input state.
        // Decision-16 §3 retains the completed device read as fault evidence.
        for model in [crate::CpuModel::I80286, crate::CpuModel::I80386] {
            let mut vm = Vm::new(VmConfig::flat(0x10000));
            vm.set_cpu_model(model);
            vm.set_exception_delivery(crate::ExceptionDelivery::Outward);
            vm.map(0, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
            // Replace the decoded NOP with HLT before ADC [3000h],AX.
            vm.write_bytes(0x1000, b"\xc6\x06\x09\x10\xf4\x11\x06\x00\x30\x90")
                .unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu.rip = 0x1000;
            cpu.cpu.gpr[0] = 0x7fff;
            cpu.cpu.flags.cf = true;
            cpu.cpu.flags.of = false;
            let before = cpu.cpu.flags;
            assert_eq!(
                cpu.run_steps(&vm, 1, StepOptions::default())
                    .unwrap()
                    .steps(),
                1
            );
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 0);
            assert!(
                matches!(
                    run.stop(),
                    StepStop::Exit(Exit::MmioRead {
                        addr: 0x3000,
                        size: 2,
                        ..
                    })
                ),
                "{run:?}"
            );
            assert_eq!(
                crate::legacy::derive_legacy_context(cpu.legacy_model.unwrap(), &cpu.cpu)
                    .unwrap()
                    .segment_class,
                crate::SegmentClass::Direct
            );
            let execution = cpu.legacy_execution.as_ref().unwrap();
            assert!(!execution.smc_pages.is_empty());
            let ir = &execution
                .blocks
                .values()
                .find(|b| b.ir.guest_start == 0x1005)
                .unwrap()
                .ir;
            assert_eq!(
                ir.legacy_context.unwrap().segment_class,
                crate::SegmentClass::Checked
            );
            assert!(!ir
                .ops
                .iter()
                .any(|op| matches!(op, IrOp::Legacy(crate::legacy::LegacyOp::Checkpoint))));
            cpu.complete_mmio_read(0);
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 0);
            assert!(
                matches!(
                    run.stop(),
                    StepStop::Exit(Exit::MmioWrite {
                        addr: 0x3000,
                        size: 2,
                        value: 0x8000,
                        ..
                    })
                ),
                "{run:?}"
            );
            assert_eq!(cpu.cpu.flags, before);
            vm.unmap(0x3000, 0x1000).unwrap();
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 0);
            let StepStop::Exit(
                exit @ Exit::UnmappedMemory {
                    addr: 0x3000,
                    access: crate::AccessKind::Write,
                    ..
                },
            ) = run.stop()
            else {
                panic!("{run:?}")
            };
            assert_eq!(
                exit.fault_metadata()
                    .unwrap()
                    .progress()
                    .acknowledged_transfers(),
                &[(0x3000, 0)]
            );
            assert_eq!(cpu.cpu.flags, before);
            assert_eq!(
                (cpu.cpu.rip, cpu.cpu.gpr[0], cpu.retired_instructions()),
                (0x1005, 0x7fff, 1)
            );
        }
    }

    #[test]
    fn double_fault_classes_follow_each_generation() {
        // I286 PRM §9.6.2 p. 9-10; I386 PRM §9.8.8 Tables 9-3/9-4 p. 9-13.
        let exception = |vector| EventSource::Exception {
            vector,
            error_code: None,
        };
        let benign = [1, 3, 4, 5, 6, 7, 16];
        for first in benign {
            for second in [0, 9, 10, 11, 12, 13, 14] {
                for generation in [CpuGeneration::I286, CpuGeneration::I386] {
                    assert!(!double_fault(
                        generation,
                        exception(first),
                        exception(second)
                    ));
                }
            }
        }
        for first in [0, 10, 11, 12, 13] {
            for second in [0, 10, 11, 12, 13] {
                for generation in [CpuGeneration::I286, CpuGeneration::I386] {
                    assert!(double_fault(
                        generation,
                        exception(first),
                        exception(second)
                    ));
                }
            }
        }
        assert!(!double_fault(
            CpuGeneration::I286,
            exception(9),
            exception(13)
        ));
        assert!(double_fault(
            CpuGeneration::I386,
            exception(9),
            exception(13)
        ));
        assert!(!double_fault(
            CpuGeneration::I386,
            exception(13),
            exception(14)
        ));
        for second in [0, 9, 10, 11, 12, 13, 14] {
            assert!(double_fault(
                CpuGeneration::I386,
                exception(14),
                exception(second)
            ));
        }
        for first in [EventSource::SingleStep, EventSource::Irq(0)] {
            assert!(!double_fault(CpuGeneration::I386, first, exception(13)));
        }
    }

    #[test]
    fn a20_toggles_reuse_shared_and_retained_physical_blocks() {
        let (vm, mut cpu) = a20_loop();
        let mut originals = Vec::new();
        for enabled in [true, false, true, false] {
            cpu.set_a20_enabled(enabled);
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1);
            let block = if enabled {
                let CachedBlock::Interpreted(ir) = vm
                    .cache
                    .get(BlockKey::machine_direct(0x100000, 0, true))
                    .unwrap()
                else {
                    unreachable!()
                };
                ir
            } else {
                cpu.legacy_execution
                    .as_ref()
                    .unwrap()
                    .blocks
                    .values()
                    .next()
                    .unwrap()
                    .ir
                    .clone()
            };
            if originals.len() < 2 {
                originals.push(block);
            } else {
                assert!(Arc::ptr_eq(&block, &originals[usize::from(!enabled)]));
            }
        }
        let execution = cpu.legacy_execution.as_ref().unwrap();
        assert_eq!(execution.blocks.len(), 1);
        assert_eq!(execution.lifts, 1);
    }

    #[test]
    fn checked_a20_toggles_reuse_only_the_matching_retained_blocks() {
        let (vm, mut cpu) = a20_loop();
        cpu.cpu.legacy.segments[SegmentId::Ds as usize].limit = 0xfffe;
        let mut originals = Vec::new();
        for enabled in [true, false, true, false] {
            cpu.set_a20_enabled(enabled);
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1);
            let execution = cpu.legacy_execution.as_ref().unwrap();
            let block = execution
                .blocks
                .iter()
                .find(|(key, _)| key.context_id == execution.context_id)
                .unwrap()
                .1
                .ir
                .clone();
            if originals.len() < 2 {
                originals.push(block);
            } else {
                assert!(Arc::ptr_eq(&block, &originals[usize::from(!enabled)]));
            }
        }
        let execution = cpu.legacy_execution.as_ref().unwrap();
        assert_eq!(execution.contexts.len(), 2);
        assert_eq!(execution.blocks.len(), 2);
        assert_eq!(execution.lifts, 2);
    }

    // spec.md §17.10: NMI blocking forces no one-step dispatch. A handler that
    // never executes IRET (it far-jumps back) keeps NMI blocked, and the
    // following loop of 10^6 instructions walks exactly the blocks of the same
    // loop entered without an NMI.
    #[test]
    fn a_blocked_nmi_handler_that_never_irets_keeps_ordinary_dispatch() {
        for model in [crate::CpuModel::I80286, crate::CpuModel::I80386] {
            let walk = |nmi: bool| {
                let mut vm = Vm::new(VmConfig::flat(0x40000));
                vm.set_cpu_model(model);
                vm.map(0, 0x40000, Prot::RWX, RegionKind::Ram).unwrap();
                // INC AX; INC BX; JMP $-2 at 0:1000; JMP FAR 0:1000 at 0:1800.
                vm.write_bytes(0x1000, b"\x40\x43\xeb\xfc").unwrap();
                vm.write_bytes(0x1800, b"\xea\x00\x10\x00\x00").unwrap();
                vm.mem.write(8, 0x1800, 4).unwrap();
                let mut cpu = vm.new_vcpu();
                cpu.cpu.rip = 0x1800;
                cpu.cpu.gpr[4] = 0x8000;
                cpu.interp_scratch.fill_before_block = Some(0);
                if nmi {
                    cpu.cpu.rip = 0x1000;
                    cpu.raise_nmi().unwrap();
                }
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 1);
                assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0, 0x1000));
                assert_eq!(cpu.nmi_state & NMI_BLOCKED != 0, nmi);
                if nmi {
                    cpu.raise_nmi().unwrap();
                }
                let entries = cpu.interp_scratch.block_entries;
                let run = cpu.run_steps(&vm, 999_999, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 999_999);
                assert_eq!(cpu.nmi_state & NMI_BLOCKED != 0, nmi, "never unblocked");
                (
                    cpu.interp_scratch.block_entries - entries,
                    cpu.cpu.gpr[0],
                    cpu.cpu.gpr[3],
                    cpu.cpu.rip,
                )
            };
            let (blocked, plain) = (walk(true), walk(false));
            assert!(plain.0 > 1000, "{plain:?}");
            assert_eq!(blocked, plain, "{model:?}");
        }
    }

    // A Machine at 0:1000 (real mode, SP 8000h) with `code` there, the NMI
    // handler `handler` at 0:1800 and the IRQ 20h handler HLT at 0:1900.
    fn blocked_nmi_machine(model: crate::CpuModel, code: &[u8], handler: &[u8]) -> (Vm, Vcpu) {
        let mut vm = Vm::new(VmConfig::flat(0x40000));
        vm.set_cpu_model(model);
        vm.map(0, 0x40000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, code).unwrap();
        vm.write_bytes(0x1800, handler).unwrap();
        vm.write_bytes(0x1900, b"\xf4").unwrap();
        vm.mem.write(8, 0x1800, 4).unwrap();
        vm.mem.write(0x80, 0x1900, 4).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x1000;
        cpu.cpu.gpr[4] = 0x8000;
        (vm, cpu)
    }

    // spec.md §17.10: while NMI is blocked (I80286, I80386) the walk applies
    // the dispatcher's predicate. An edge raised after the dispatcher's word
    // test and before the walk's first boundary (here by the scratch hook, at
    // every block entry) neither stops the walk nor ends the STI shadow the
    // walk carries in. The handler, entered with IF = 0 and never left, is
    // STI; CLI; JMP $-4: its only IF = 1 boundary is the one STI shadows, so a
    // queued IRQ is never taken. The edge storm walks exactly the blocks of
    // the same run with no edges, with or without the IRQ.
    #[test]
    fn a_blocked_edge_inside_a_walk_neither_stops_it_nor_ends_the_sti_shadow() {
        for model in [crate::CpuModel::I80286, crate::CpuModel::I80386] {
            for irq in [false, true] {
                let walk = |storm: bool| {
                    let (vm, mut cpu) = blocked_nmi_machine(model, b"\x90", b"\xfb\xfa\xeb\xfc");
                    cpu.interp_scratch.fill_before_block = Some(0);
                    cpu.raise_nmi().unwrap();
                    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                    assert_eq!((run.steps(), cpu.cpu.rip), (1, 0x1801)); // STI
                    assert!(cpu.cpu.flags.if_ && cpu.sti_shadow);
                    assert_ne!(cpu.nmi_state & NMI_BLOCKED, 0);
                    if irq {
                        cpu.inject_irq(0x20);
                    }
                    if storm {
                        cpu.interp_scratch.nmi_before_block =
                            Some((cpu.nmi_line().unwrap(), 200_000));
                    }
                    let entries = cpu.interp_scratch.block_entries;
                    let run = cpu.run_steps(&vm, 30_000, StepOptions::default()).unwrap();
                    assert!(
                        matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
                        "{model:?} irq {irq} storm {storm}: {:?} at {:#x}",
                        run.stop(),
                        cpu.cpu.rip
                    );
                    assert_eq!(run.steps(), 30_000);
                    assert_eq!(cpu.has_pending_irq(), irq, "never taken in the shadow");
                    assert_ne!(cpu.nmi_state & NMI_BLOCKED, 0);
                    if storm {
                        let left = cpu.interp_scratch.nmi_before_block.as_ref().unwrap().1;
                        assert_ne!(left, 0, "an edge at every walk");
                    }
                    (
                        cpu.interp_scratch.block_entries - entries,
                        format!("{:?}", cpu.cpu),
                    )
                };
                let (calm, storm) = (walk(false), walk(true));
                assert!(calm.0 >= 10_000, "{calm:?}");
                assert_eq!(storm, calm, "{model:?} irq {irq}");
            }
        }
    }

    // spec.md §17.10: an edge raised while blocked that no word test has seen
    // yet (here raised at the entry of the IRET's own walk) is still in the
    // word when that IRET ends blocking. The I80386 ignores it as it ignores
    // the others (80386 PRM §9.2.1 p. 9-3); on the I80286 it stays latched,
    // as a remembered edge would be, and is serviced after that IRET (I286
    // PRM §9.2 p. 9-3).
    #[test]
    fn an_edge_raised_in_the_irets_own_walk_follows_the_blocked_rule() {
        for model in [crate::CpuModel::I80286, crate::CpuModel::I80386] {
            let (vm, mut cpu) = blocked_nmi_machine(model, b"\x90\x90", b"\xcf");
            let line = cpu.nmi_line().unwrap();
            cpu.raise_nmi().unwrap();
            // Delivery costs no step; the hook raises at the IRET's walk.
            cpu.interp_scratch.nmi_before_block = Some((line.clone(), 1));
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1);
            assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[4]), (0x1000, 0x8000));
            assert_eq!(cpu.interp_scratch.nmi_before_block.as_ref().unwrap().1, 0);
            assert_eq!(cpu.nmi_state, 0, "the IRET ended blocking");
            let remember = model == crate::CpuModel::I80286;
            assert_eq!(
                (line.is_raised(), cpu.has_pending_nmi()),
                (remember, remember)
            );
            // The 80286 takes it before the next NOP (the handler's IRET is
            // its one step); the 80386 executes the NOP.
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1);
            assert_eq!(cpu.cpu.rip, if remember { 0x1000 } else { 0x1001 });
            assert_eq!(cpu.retired_instructions(), 2);
            assert!(!line.is_raised() && cpu.nmi_state == 0);
        }
    }

    fn a20_loop() -> (Vm, Vcpu) {
        let mut vm = Vm::new(VmConfig::flat(0x200000));
        vm.set_legacy_cpu_model(LegacyCpuModel {
            generation: CpuGeneration::I386,
            model_semantics_id: 1,
        });
        vm.map(0, 0x200000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0, b"\xeb\xfe").unwrap();
        vm.write_bytes(0x100000, b"\xeb\xfe").unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu = CpuState::legacy_real_entry([0xffff, 0, 0, 0, 0, 0]);
        cpu.cpu.rip = 0x10;
        (vm, cpu)
    }

    #[test]
    #[ignore = "explicit interpreter A20 toggle throughput measurement"]
    fn a20_toggle_throughput() {
        let (vm, mut cpu) = a20_loop();
        for enabled in [false, true] {
            cpu.set_a20_enabled(enabled);
            assert_eq!(
                cpu.run_steps(&vm, 1, StepOptions::default())
                    .unwrap()
                    .steps(),
                1
            );
        }
        for sample in 0..3 {
            let start = std::time::Instant::now();
            for i in 0..200_000 {
                cpu.set_a20_enabled(i & 1 != 0);
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 1);
                assert!(matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)));
            }
            let elapsed = start.elapsed().as_secs_f64();
            let execution = cpu.legacy_execution.as_ref().unwrap();
            assert_eq!(cpu.cpu.rip, 0x10);
            assert_eq!(execution.lifts, 1);
            assert_eq!(execution.blocks.len(), 1);
            println!("sample={sample} toggles=200000 elapsed_seconds={elapsed:.6} toggles_per_second={:.0} ns_per_toggle_and_instruction={:.1} lifts=1 cached_blocks=2", 200_000.0 / elapsed, elapsed * 1e9 / 200_000.0);
        }
        assert_eq!(vm.backend.compile_ns(), 0);
    }

    fn cache_context(
        execution: &mut Execution,
        vm: &Vm,
        cpu: &mut CpuState,
        model: LegacyCpuModel,
        tag: u32,
    ) -> Vec<Arc<IrBlock>> {
        cpu.legacy.segments[SegmentId::Ds as usize].base = tag;
        execution.sync(cpu, model, false);
        let context = derive_legacy_context(model, cpu).unwrap();
        [0x1000, 0x1006]
            .into_iter()
            .map(|ip| {
                cpu.rip = ip;
                execution
                    .resolve::<MACHINE_DISPATCH>(vm, cpu, model, context, &AtomicU64::new(0))
                    .unwrap()
                    .0
            })
            .collect()
    }

    #[test]
    fn context_churn_bounds_retained_contexts_and_their_blocks() {
        let (vm, mut cpu) = pe_transition_loop();
        let model = cpu.legacy_model.unwrap();
        let mut execution = Execution::new();
        for tag in 0..(MAX_RETAINED_CONTEXTS * 8) {
            cache_context(&mut execution, &vm, &mut cpu.cpu, model, tag as u32);
            let retained = (tag + 1).min(MAX_RETAINED_CONTEXTS);
            assert_eq!(execution.contexts.len(), retained);
            assert_eq!(execution.blocks.len(), retained * 2);
            assert!(execution.blocks.keys().all(|key| execution
                .contexts
                .values()
                .any(|context| context.id == key.context_id)));
        }
        assert!(execution.contexts.keys().all(|stamp| {
            stamp.state.segments[SegmentId::Ds as usize].base >= (MAX_RETAINED_CONTEXTS * 7) as u32
        }));
    }

    // Decision O13: a ring-3 loop that enters ring 0 through INT 40h and
    // returns with IRET, or through a call gate and returns with RETF,
    // retains two contexts, ring 3 and ring 0, and relifts nothing after its
    // first round trip.
    fn ring_trip_loop(generation: CpuGeneration, call_gate: bool) -> (Vm, Vcpu) {
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        vm.set_legacy_cpu_model(LegacyCpuModel {
            generation,
            model_semantics_id: 1,
        });
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        if call_gate {
            // call far 0033h:0; jmp $-5; at 5000h retf. GDT 30h is a DPL-3
            // 286 call gate to 0008h:5000h with no parameters.
            vm.write_bytes(0x1000, b"\x9a\x00\x00\x33\x00\xeb\xf9")
                .unwrap();
            vm.write_bytes(0x5000, b"\xcb").unwrap();
            vm.write_bytes(0x2030, &[0, 0x50, 8, 0, 0, 0xe4, 0, 0])
                .unwrap();
        } else {
            vm.write_bytes(0x1000, b"\xcd\x40\xeb\xfc").unwrap(); // int 40h; jmp $-2
            vm.write_bytes(0x5000, b"\xcf").unwrap(); // iret
        }
        for (selector, attr) in [(8u64, 0x9bu8), (16, 0x93), (24, 0xfb), (32, 0xf3)] {
            vm.write_bytes(0x2000 + selector, &[0xff, 0xff, 0, 0, 0, attr, 0, 0])
                .unwrap();
        }
        vm.write_bytes(0x2028, &[0x2b, 0, 0, 0x30, 0, 0x83, 0, 0])
            .unwrap();
        vm.write_bytes(0x3002, &[0x00, 0x90, 0x10, 0x00]).unwrap(); // SP0, SS0
        vm.write_bytes(0x3200, &[0, 0x50, 8, 0, 0, 0xe6, 0, 0])
            .unwrap(); // vector 40h
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x1000;
        cpu.cpu.gpr[4] = 0x8000;
        cpu.cpu.legacy.cr0 = 1;
        cpu.cpu.legacy.cpl = 3;
        [cpu.cpu.cs, cpu.cpu.ds, cpu.cpu.es, cpu.cpu.ss] = [0x1b, 0x23, 0x23, 0x23];
        for (i, attr) in [0xfb, 0xf3, 0xf3, 0xf3].into_iter().enumerate() {
            cpu.cpu.legacy.segments[i].attributes = attr;
        }
        // Null FS and GS, as an outward return leaves them (SDM Vol 2A IRET
        // p. 3-495); usable null caches would be one more context each time.
        for segment in [SegmentId::Fs, SegmentId::Gs] {
            cpu.cpu.legacy.segments[segment as usize] = crate::legacy::SegmentCache {
                unusable: 1,
                ..Default::default()
            };
        }
        cpu.cpu.legacy.gdtr.base = 0x2000;
        cpu.cpu.legacy.gdtr.limit = 0x37;
        cpu.cpu.legacy.idtr.base = 0x3000;
        cpu.cpu.legacy.idtr.limit = 0x7ff;
        cpu.cpu.legacy.tr = crate::legacy::SystemSegment {
            selector: 0x28,
            cache: crate::legacy::SegmentCache {
                base: 0x3000,
                limit: 0x2b,
                attributes: 0x83,
                ..Default::default()
            },
            ..Default::default()
        };
        (vm, cpu)
    }

    #[test]
    fn ring_trips_retain_two_contexts_and_relift_nothing() {
        for (generation, call_gate) in [
            (CpuGeneration::I286, false),
            (CpuGeneration::I386, false),
            (CpuGeneration::I286, true),
            (CpuGeneration::I386, true),
        ] {
            let (vm, mut cpu) = ring_trip_loop(generation, call_gate);
            let warm = cpu.run_steps(&vm, 3, StepOptions::default()).unwrap();
            assert_eq!(warm.steps(), 3);
            let warm = cpu.legacy_execution.as_ref().unwrap();
            let (lifts, blocks) = (warm.lifts, warm.blocks.len());
            let run = cpu.run_steps(&vm, 3000, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 3000, "{:?}", run.stop());
            assert_eq!(
                (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
                (0x1b, 0x1000, 0x8000)
            );
            let execution = cpu.legacy_execution.as_ref().unwrap();
            assert_eq!(execution.lifts, lifts);
            assert_eq!(execution.contexts.len(), 2);
            assert_eq!(execution.blocks.len(), blocks);
        }
    }

    #[test]
    #[ignore = "explicit interpreter ring-trip throughput measurement"]
    fn ring_trip_throughput() {
        for generation in [CpuGeneration::I286, CpuGeneration::I386] {
            let (vm, mut cpu) = ring_trip_loop(generation, false);
            cpu.run_steps(&vm, 3, StepOptions::default()).unwrap();
            for sample in 0..3 {
                let lifts = cpu.legacy_execution.as_ref().unwrap().lifts;
                let begin = std::time::Instant::now();
                let run = cpu.run_steps(&vm, 300_000, StepOptions::default()).unwrap();
                let elapsed = begin.elapsed().as_secs_f64();
                assert_eq!(run.steps(), 300_000);
                let execution = cpu.legacy_execution.as_ref().unwrap();
                println!("generation={generation:?} sample={sample} trips=100000 elapsed_seconds={elapsed:.6} ns_per_trip={:.1} contexts={} lifts={} cached_blocks={}",
                    elapsed * 1e9 / 100_000.0, execution.contexts.len(), execution.lifts - lifts, execution.blocks.len());
            }
        }
    }

    #[test]
    fn context_lru_keeps_hot_contexts_and_their_blocks() {
        let (vm, mut cpu) = pe_transition_loop();
        let model = cpu.legacy_model.unwrap();
        let mut execution = Execution::new();
        let hot: Vec<_> = (0..2)
            .map(|tag| cache_context(&mut execution, &vm, &mut cpu.cpu, model, tag))
            .collect();
        for tag in 2..(MAX_RETAINED_CONTEXTS * 8) {
            cache_context(&mut execution, &vm, &mut cpu.cpu, model, tag as u32);
            let lifts = execution.lifts;
            for (tag, original) in hot.iter().enumerate() {
                let blocks = cache_context(&mut execution, &vm, &mut cpu.cpu, model, tag as u32);
                for (ir, original) in blocks.iter().zip(original) {
                    assert!(Arc::ptr_eq(ir, original));
                }
            }
            assert_eq!(execution.lifts, lifts);
            assert!(execution.contexts.len() <= MAX_RETAINED_CONTEXTS);
            assert!(execution.blocks.len() <= MAX_RETAINED_CONTEXTS * 2);
        }
    }

    #[test]
    fn task_register_and_ldtr_separate_retained_context_namespaces() {
        let (vm, mut cpu) = pe_transition_loop();
        let model = cpu.legacy_model.unwrap();
        let mut execution = Execution::new();
        let original = cache_context(&mut execution, &vm, &mut cpu.cpu, model, 0);
        let original_id = execution.context_id;
        let before = cpu.cpu.legacy;
        cpu.cpu.legacy.tr.selector = 24;
        cpu.cpu.legacy.tr.cache.base = 0x4000;
        let task = cache_context(&mut execution, &vm, &mut cpu.cpu, model, 0);
        assert_ne!(execution.context_id, original_id);
        assert!(!Arc::ptr_eq(&original[0], &task[0]));
        let task_id = execution.context_id;
        cpu.cpu.legacy.ldtr.selector = 32;
        cpu.cpu.legacy.ldtr.cache.base = 0x5000;
        let ldt = cache_context(&mut execution, &vm, &mut cpu.cpu, model, 0);
        assert_ne!(execution.context_id, task_id);
        assert!(!Arc::ptr_eq(&task[0], &ldt[0]));
        cpu.cpu.legacy = before;
        let returned = cache_context(&mut execution, &vm, &mut cpu.cpu, model, 0);
        assert_eq!(execution.context_id, original_id);
        assert!(Arc::ptr_eq(&original[0], &returned[0]));
    }

    fn pe_transition_loop() -> (Vm, Vcpu) {
        // SDM Vol 3A §§12.9.1–12.9.2; 80386 PRM §§14.4–14.5:
        // each round enters USE16 with a far JMP and returns with a real far JMP.
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        vm.set_legacy_cpu_model(LegacyCpuModel {
            generation: CpuGeneration::I386,
            model_semantics_id: 1,
        });
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(
            0x1000,
            b"\x66\xb8\x01\x00\x00\x00\x0f\x22\xc0\xea\x00\x11\x08\x00",
        )
        .unwrap();
        vm.write_bytes(0x1100, b"\x31\xc0\x0f\x22\xc0\xea\x00\x10\x00\x00")
            .unwrap();
        vm.write_bytes(0x2008, b"\xff\xff\x00\x00\x00\x9b\x00\x00")
            .unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.legacy.segments[SegmentId::Ds as usize].attributes = 0x92;
        cpu.cpu.rip = 0x1000;
        cpu.cpu.legacy.gdtr.base = 0x2000;
        cpu.cpu.legacy.gdtr.limit = 15;
        (vm, cpu)
    }

    #[test]
    #[ignore = "explicit interpreter transition throughput measurement"]
    fn pe_transition_throughput() {
        let (vm, mut cpu) = pe_transition_loop();
        let warm = cpu.run_steps(&vm, 600, StepOptions::default()).unwrap();
        assert_eq!(warm.steps(), 600);
        for sample in 0..3 {
            let lifts = cpu.legacy_execution.as_ref().unwrap().lifts;
            let begin = std::time::Instant::now();
            let run = cpu.run_steps(&vm, 600_000, StepOptions::default()).unwrap();
            let elapsed = begin.elapsed().as_secs_f64();
            assert_eq!(run.steps(), 600_000, "{:?}", run.stop());
            assert!(matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)));
            assert_eq!(
                (cpu.cpu.rip, cpu.cpu.cs, cpu.cpu.legacy.cr0),
                (0x1000, 0, 0)
            );
            let execution = cpu.legacy_execution.as_ref().unwrap();
            println!("sample={sample} transitions=200000 elapsed_seconds={elapsed:.6} transitions_per_second={:.0} lifts={} cached_blocks={}",
                200_000.0 / elapsed, execution.lifts - lifts, execution.blocks.len());
        }
        assert_eq!(cpu.retired_instructions(), 1_800_600);
        assert_eq!(vm.backend.compile_ns(), 0);
    }
    #[test]
    fn real_delivery_advances_epoch_before_the_handler_can_execute() {
        for irq in [false, true] {
            let model = LegacyCpuModel {
                generation: CpuGeneration::I386,
                model_semantics_id: 1,
            };
            let mut vm = Vm::new(VmConfig::flat(0x10000));
            vm.set_legacy_cpu_model(model);
            vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0x1000, b"\xcd\x20\x43").unwrap();
            vm.write_bytes(0x2080, b"\x00\x02\x00\x01").unwrap();
            vm.write_bytes(0x1200, b"\xf4").unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu.rip = 0x1000;
            cpu.cpu.gpr[4] = 0x8000;
            cpu.cpu.legacy.idtr.base = 0x2000;
            cpu.cpu.flags.if_ = true;
            if irq {
                cpu.inject_irq(0x20);
            }
            cpu.legacy_execution
                .as_mut()
                .unwrap()
                .sync(&cpu.cpu, model, false);
            let before = cpu.legacy_execution.as_ref().unwrap().context_epoch;
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1);
            assert_eq!(
                cpu.legacy_execution.as_ref().unwrap().context_epoch,
                before + 1
            );
            assert_eq!(cpu.cpu.cs, 0x100);
            assert_eq!(cpu.cpu.rip, if irq { 0x201 } else { 0x200 });
            assert_eq!(cpu.cpu.gpr[3], 0);
        }
    }

    #[test]
    fn system_writes_end_the_block_and_advance_only_the_required_epochs() {
        let model = LegacyCpuModel {
            generation: CpuGeneration::I386,
            model_semantics_id: 1,
        };
        for (code, cr0, eax, advances) in [
            (&b"\x0f\x22\xd8"[..], 0, 0, true), // equal CR3 still invalidates
            (&b"\x0f\x22\xd0"[..], 0, 0x1234, false), // live CR2
            (&b"\x0f\x22\xc0"[..], 0, 0x10, true),
            (&b"\x0f\x22\xc0"[..], 0, 0, false),
            (&b"\x0f\x01\xf0"[..], 0x10, 0xe, true),
            (&b"\x0f\x06"[..], 0x18, 0, true),
            (&b"\x0f\x01\x16\x00\x20"[..], 0, 0, true),
            (&b"\x0f\x01\x1e\x00\x20"[..], 0, 0, true),
        ] {
            let mut vm = Vm::new(VmConfig::flat(0x10000));
            vm.set_legacy_cpu_model(model);
            vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0x1000, code).unwrap();
            vm.write_bytes(0x1000 + code.len() as u64, b"\x43").unwrap();
            vm.write_bytes(0x2000, b"\xff\x03\x00\x30\x00\x00").unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu.legacy.segments[SegmentId::Ds as usize].attributes = 0x92;
            cpu.cpu.rip = 0x1000;
            cpu.cpu.legacy.cr0 = cr0;
            cpu.cpu.gpr[0] = eax;
            cpu.legacy_execution
                .as_mut()
                .unwrap()
                .sync(&cpu.cpu, model, false);
            let before = cpu.legacy_execution.as_ref().unwrap().context_epoch;
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1, "{code:02x?}: {:?}", run.stop());
            assert_eq!(cpu.cpu.gpr[3], 0);
            assert_eq!(cpu.cpu.rip, 0x1000 + code.len() as u64);
            assert_eq!(
                cpu.legacy_execution.as_ref().unwrap().context_epoch,
                before + u64::from(advances)
            );
            if code == b"\x0f\x22\xd8" {
                assert!(cpu.legacy_execution.as_ref().unwrap().blocks.is_empty());
            } else {
                assert_eq!(cpu.legacy_execution.as_ref().unwrap().blocks.len(), 1);
            }
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1);
            assert_eq!(cpu.cpu.gpr[3], 1);
        }
    }

    #[test]
    fn epoch_stamp_guards_context_but_not_live_outputs() {
        let model = LegacyCpuModel {
            generation: CpuGeneration::I386,
            model_semantics_id: 1,
        };
        let cpu = CpuState::legacy_real_entry([0; 6]);
        let mut e = Execution::new();
        e.sync(&cpu, model, false);
        let epoch = e.context_epoch;
        let mut live = cpu.clone();
        live.legacy.cr2 = 0x1234;
        live.flags.if_ = true;
        e.sync(&live, model, false);
        assert_eq!(e.context_epoch, epoch);
        let edits: [fn(&mut CpuState); 12] = [
            |c| c.legacy.cr0 = 1,
            |c| c.legacy.cr3 = 0x1000,
            |c| c.legacy.cpl = 1,
            |c| c.ds = 3,
            |c| c.legacy.segments[4].base = 0x1000,
            |c| c.legacy.gdtr.limit = 0xff,
            |c| c.legacy.idtr.base = 0x2000,
            |c| c.legacy.ldtr.selector = 8,
            |c| c.legacy.tr.selector = 16,
            |c| c.rflags_system = 0x7000,
            |c| c.legacy.a20_enabled = 0,
            |c| c.legacy.reset_fetch = 1,
        ];
        for edit in edits {
            let mut changed = cpu.clone();
            edit(&mut changed);
            let before = e.context_epoch;
            e.sync(&changed, model, false);
            assert_ne!(e.context_epoch, before);
        }
        let before = e.context_epoch;
        e.sync(
            &cpu,
            LegacyCpuModel {
                model_semantics_id: 2,
                ..model
            },
            false,
        );
        assert_ne!(e.context_epoch, before);
        assert_ne!(e.owner_id, Execution::new().owner_id);
    }

    #[test]
    fn ldtr_and_tr_hidden_changes_isolate_retained_instruction_blocks() {
        // Decisions 18 §3 and 20: the full system-register cache participates,
        // even when its visible selector stays unchanged. Returning reuses IR.
        for model in [crate::CpuModel::I80286, crate::CpuModel::I80386] {
            let mut vm = Vm::new(VmConfig::flat(0x10000));
            vm.set_cpu_model(model);
            vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0x1000, b"\xeb\xfe").unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu.rip = 0x1000;
            cpu.cpu.legacy.cr0 = 1;
            let initial = cpu.cpu.legacy;
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1);
            let execution = cpu.legacy_execution.as_ref().unwrap();
            let id = execution.context_id;
            let original = execution.blocks.values().next().unwrap().ir.clone();
            let edits: [fn(&mut LegacyCpuState); 8] = [
                |s| s.ldtr.cache.base = 0x2000,
                |s| s.ldtr.cache.limit = 0xff,
                |s| s.ldtr.cache.attributes = 0x82,
                |s| s.ldtr.cache.unusable = 1,
                |s| s.tr.cache.base = 0x3000,
                |s| s.tr.cache.limit = 0x67,
                |s| s.tr.cache.attributes = 0x83,
                |s| s.tr.cache.unusable = 1,
            ];
            for edit in edits {
                cpu.cpu.legacy = initial;
                edit(&mut cpu.cpu.legacy);
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 1);
                let execution = cpu.legacy_execution.as_ref().unwrap();
                assert_ne!(execution.context_id, id);
                let block = execution
                    .blocks
                    .iter()
                    .find(|(key, _)| key.context_id == execution.context_id)
                    .unwrap()
                    .1;
                assert!(!Arc::ptr_eq(&original, &block.ir));
                cpu.cpu.legacy = initial;
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 1);
                let execution = cpu.legacy_execution.as_ref().unwrap();
                assert_eq!(execution.context_id, id);
                let block = execution
                    .blocks
                    .iter()
                    .find(|(key, _)| key.context_id == id)
                    .unwrap()
                    .1;
                assert!(Arc::ptr_eq(&original, &block.ir));
            }
            assert_eq!(cpu.legacy_execution.as_ref().unwrap().lifts, 9);
        }
    }

    #[test]
    fn epoch_rollover_drops_populated_contexts_before_reusing_ids() {
        let (vm, mut cpu) = pe_transition_loop();
        let model = cpu.legacy_model.unwrap();
        let mut execution = Execution::new();
        let mut original = Vec::new();
        for tag in 0..3 {
            original.push(cache_context(&mut execution, &vm, &mut cpu.cpu, model, tag));
            assert_eq!(execution.context_id, u64::from(tag));
        }
        assert_eq!(execution.contexts.len(), 3);
        assert_eq!(execution.blocks.len(), 6);
        execution.context_epoch = u64::MAX;
        cpu.cpu.legacy.segments[SegmentId::Ds as usize].base = 0;
        execution.sync(&cpu.cpu, model, true);
        assert_eq!(execution.context_epoch, 0);
        assert_eq!(execution.context_id, 0);
        assert_eq!(execution.contexts.len(), 1);
        assert!(execution.blocks.is_empty());
        for (tag, old_blocks) in original.iter().enumerate() {
            let blocks = cache_context(&mut execution, &vm, &mut cpu.cpu, model, tag as u32);
            assert_eq!(execution.context_id, tag as u64);
            for (ir, old) in blocks.iter().zip(old_blocks) {
                assert!(!Arc::ptr_eq(ir, old));
            }
        }
        assert_eq!(execution.blocks.len(), 6);
        assert_eq!(execution.lifts, 12);
    }

    #[test]
    fn pe_transitions_reuse_both_regimes_without_relifting_or_flushing_other_contexts() {
        let (vm, mut cpu) = pe_transition_loop();
        let warm = cpu.run_steps(&vm, 6, StepOptions::default()).unwrap();
        assert_eq!(warm.steps(), 6, "{:?}", warm.stop());
        let execution = cpu.legacy_execution.as_ref().unwrap();
        assert_eq!(execution.contexts.len(), 4);
        assert_eq!(execution.blocks.len(), 6);
        assert_eq!(execution.lifts, 6);
        let blocks: Vec<_> = execution
            .blocks
            .iter()
            .map(|(key, block)| (*key, block.ir.clone()))
            .collect();
        let before_epoch = execution.context_epoch;
        for count in [1, 2, 3, 6, 600] {
            let run = cpu.run_steps(&vm, count, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), count, "{:?}", run.stop());
        }
        let execution = cpu.legacy_execution.as_ref().unwrap();
        assert!(execution.context_epoch > before_epoch);
        assert_eq!(execution.contexts.len(), 4);
        assert_eq!(execution.blocks.len(), 6);
        assert_eq!(execution.lifts, 6);
        for (key, ir) in blocks {
            assert!(Arc::ptr_eq(&ir, &execution.blocks[&key].ir));
        }
    }

    #[test]
    fn direct_protected_round_trips_reuse_both_cache_namespaces() {
        let (vm, mut cpu) = pe_transition_loop();
        cpu.cpu.legacy.segments[SegmentId::Ds as usize].attributes = 0x93;
        let warm = cpu.run_steps(&vm, 6, StepOptions::default()).unwrap();
        assert_eq!(warm.steps(), 6, "{:?}", warm.stop());
        let key = BlockKey::machine_direct(0x1000, 0, true);
        let CachedBlock::Interpreted(direct) = vm.cache.get(key).unwrap() else {
            unreachable!()
        };
        assert_eq!(direct.icount, 1);
        assert_eq!(
            direct.legacy_context.unwrap().segment_class,
            crate::SegmentClass::Direct
        );
        let blocks: Vec<_> = cpu
            .legacy_execution
            .as_ref()
            .unwrap()
            .blocks
            .iter()
            .map(|(key, block)| (*key, block.ir.clone()))
            .collect();
        assert!(!blocks.is_empty());
        let lifts = cpu.legacy_execution.as_ref().unwrap().lifts;
        let run = cpu.run_steps(&vm, 600, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 600, "{:?}", run.stop());
        let CachedBlock::Interpreted(reused) = vm.cache.get(key).unwrap() else {
            unreachable!()
        };
        assert!(Arc::ptr_eq(&direct, &reused));
        let execution = cpu.legacy_execution.as_ref().unwrap();
        assert_eq!(execution.lifts, lifts);
        for (key, ir) in blocks {
            assert!(Arc::ptr_eq(&ir, &execution.blocks[&key].ir));
        }
    }

    #[test]
    fn reused_contexts_revalidate_code_bytes_and_between_run_cache_edits() {
        let (vm, mut cpu) = pe_transition_loop();
        let model = cpu.legacy_model.unwrap();
        let mut execution = Execution::new();
        let mut original = Vec::new();
        for pe in [0, 1] {
            cpu.cpu.legacy.cr0 = pe;
            let context = derive_legacy_context(model, &cpu.cpu).unwrap();
            execution.sync(&cpu.cpu, model, false);
            original.push(
                execution
                    .resolve::<MACHINE_DISPATCH>(&vm, &cpu.cpu, model, context, &AtomicU64::new(0))
                    .unwrap()
                    .0,
            );
        }
        for pe in [0, 1] {
            cpu.cpu.legacy.cr0 = pe;
            let context = derive_legacy_context(model, &cpu.cpu).unwrap();
            execution.sync(&cpu.cpu, model, false);
            let ir = execution
                .resolve::<MACHINE_DISPATCH>(&vm, &cpu.cpu, model, context, &AtomicU64::new(0))
                .unwrap()
                .0;
            assert!(Arc::ptr_eq(&ir, &original[pe as usize]));
        }
        vm.write_bytes(0x1002, &2u32.to_le_bytes()).unwrap();
        for pe in [0, 1] {
            cpu.cpu.legacy.cr0 = pe;
            let context = derive_legacy_context(model, &cpu.cpu).unwrap();
            execution.sync(&cpu.cpu, model, false);
            let ir = execution
                .resolve::<MACHINE_DISPATCH>(&vm, &cpu.cpu, model, context, &AtomicU64::new(0))
                .unwrap()
                .0;
            assert!(!Arc::ptr_eq(&ir, &original[pe as usize]));
        }
        let before_id = execution.context_id;
        cpu.cpu.legacy.segments[SegmentId::Ds as usize].limit = 0x12345;
        execution.sync(&cpu.cpu, model, false);
        assert_ne!(execution.context_id, before_id);
        let context = derive_legacy_context(model, &cpu.cpu).unwrap();
        execution
            .resolve::<MACHINE_DISPATCH>(&vm, &cpu.cpu, model, context, &AtomicU64::new(0))
            .unwrap();
        assert_eq!(execution.blocks.len(), 3);
        cpu.cpu.legacy.segments[SegmentId::Ds as usize].limit = 0xffff;
        execution.sync(&cpu.cpu, model, false);
        assert_eq!(execution.context_id, before_id);
        assert_eq!(execution.lifts, 5);
    }
}

#[cfg(test)]
mod virtual8086_key_tests {
    use super::*;

    #[test]
    fn virtual8086_key_and_stamp_separate_identical_fetch_addresses() {
        let model = crate::CpuModel::I80386.legacy_model();
        let mut cpu = CpuState::legacy_real_entry([0; 6]);
        let real = LegacyContextStamp::new(&cpu, model);
        let key = LegacyBlockKey {
            owner_id: 1,
            context_id: 1,
            model_semantics_id: model.model_semantics_id,
            decode_bits: 16,
            stack_bits: 16,
            regime: MachineRegime::Real,
            decode_ip: 0x1000,
            linear_fetch: 0x1000,
            mxcsr_mode: 0,
        };
        let v86_key = LegacyBlockKey {
            regime: MachineRegime::Virtual8086,
            ..key
        };
        let mut keys = std::collections::HashSet::new();
        keys.insert(key);
        keys.insert(v86_key);
        assert_eq!(keys.len(), 2);
        cpu.rflags_system |= 1 << 17;
        assert!(LegacyContextStamp::new(&cpu, model) != real);
        cpu.rflags_system = 0;
        assert!(LegacyContextStamp::new(&cpu, model) == real);
    }
}
