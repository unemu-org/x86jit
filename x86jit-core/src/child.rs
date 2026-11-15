//! Allocation-free instruction stepping for shared-address-space children
//! (decision-15). Create the view and arena in the parent; see the crate docs
//! for the lifetime and signal-adapter protocol.

use std::cell::{RefCell, UnsafeCell};
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::atomic::{fence, AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::fault::*;
use crate::interp::ordering::{Barrier, Direction};
use crate::ir::{MemoryAccess, RepKind, StrOp};
use crate::memory::{InterpreterMemory, CODE_PAGE_BITS};
use crate::{CpuMode, CpuState, FetchVerdict, IrBlock, IrOp, MemTrap, Memory, RegionKind, Vm};

mod transfer;
pub use transfer::classify;

/// Caller-supplied execution snapshot for [`Vm::child_view_with`].
/// Omitted addresses are holes even when the parent maps them. Construction
/// validates structure and backing identity; the embedder must freeze native
/// execute permissions and refusal tags under its own mapping synchronization.
#[derive(Clone, Debug, Default)]
pub struct ChildViewMetadata {
    pub regions: Vec<ChildRegion>,
    /// Guest 4 KiB page numbers on which stores must be refused. These supplement
    /// the parent's code tags sampled at attachment. Each must intersect child RAM.
    pub code_store_refusals: Vec<u64>,
}

/// A half-open guest interval in a caller's child snapshot. Adjacent RAM
/// intervals are joined for data accesses, retaining their own fetch verdicts.
#[derive(Clone, Debug)]
pub enum ChildRegion {
    Ram {
        range: Range<u64>,
        /// Index of `range.start` in the parent's backing, relative to
        /// [`Memory::host_base`]. Must equal `range.start - Memory::guest_base()`.
        /// No alternate pointer is accepted: child writes must reach parent bytes.
        backing_offset: usize,
        /// Frozen verdict for every fetch byte in this interval. Split at page
        /// boundaries to represent different native execute permissions per page.
        fetch: FetchVerdict,
    },
    /// Traps may narrow any mapped parent interval. Child execution
    /// refuses both fetches and data accesses here without touching the backing.
    Trap { range: Range<u64> },
}

impl ChildRegion {
    pub(crate) fn range(&self) -> &Range<u64> {
        match self {
            Self::Ram { range, .. } | Self::Trap { range } => range,
        }
    }
}

/// Invalid caller metadata; rejected before registering a child bridge journal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChildViewError {
    InvalidRange,
    Overlap,
    Unmapped,
    NotRam,
    BackingMismatch,
    InvalidCodePage,
}

impl std::fmt::Display for ChildViewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InvalidRange => "child interval is empty, reversed or outside the parent span",
            Self::Overlap => "child intervals overlap",
            Self::Unmapped => "child interval crosses an unmapped parent address",
            Self::NotRam => "child RAM interval crosses parent non-RAM memory",
            Self::BackingMismatch => "child RAM offset does not match the parent backing",
            Self::InvalidCodePage => "child refusal page does not intersect child RAM",
        })
    }
}
impl std::error::Error for ChildViewError {}

pub(crate) enum FrozenFetch {
    Pages(Vec<(u64, FetchVerdict)>),
    Ranges(Vec<(Range<u64>, FetchVerdict)>),
}

impl FrozenFetch {
    fn verdict(&self, addr: u64) -> FetchVerdict {
        match self {
            Self::Pages(pages) => pages
                .binary_search_by_key(&(addr >> CODE_PAGE_BITS), |&(p, _)| p)
                .ok()
                .map(|i| pages[i].1),
            Self::Ranges(ranges) => ranges
                .partition_point(|(range, _)| range.start <= addr)
                .checked_sub(1)
                .and_then(|i| ranges[i].0.contains(&addr).then_some(ranges[i].1)),
        }
        .unwrap_or(FetchVerdict::DenyNotPresent)
    }
}

#[derive(Clone)]
pub(crate) struct Region {
    pub start: u64,
    pub end: u64,
    pub kind: RegionKind,
    pub _prot: crate::Prot,
}

pub(crate) const DISPATCH_CHILD_DIRTY: u64 = 1 << 3;

/// Stable, append-only dispatch targets: a child cannot borrow the parent's
/// mutable registry or upgrade/drop weak owners. Reclaimed with Memory.
#[derive(Default)]
pub(crate) struct ChildDispatch {
    enabled: AtomicBool,
    head: OnceLock<Arc<ChildDispatchNode>>,
    nodes: Mutex<Vec<Arc<ChildDispatchNode>>>,
    pub(crate) dirty: AtomicBool,
}

struct ChildDispatchNode {
    word: Arc<AtomicU64>,
    epoch: AtomicU64,
    next: OnceLock<Arc<ChildDispatchNode>>,
}

impl ChildDispatch {
    pub(crate) fn enable(&self) -> bool {
        !self.enabled.swap(true, Ordering::Relaxed)
    }

    pub(crate) fn register(&self, word: &Arc<AtomicU64>) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        let mut nodes = self.nodes.lock().unwrap();
        let node = Arc::new(ChildDispatchNode {
            word: word.clone(),
            epoch: AtomicU64::new(0),
            next: OnceLock::new(),
        });
        let link = nodes.last().map_or(&self.head, |tail| &tail.next);
        assert!(link.set(node.clone()).is_ok());
        nodes.push(node);
        // Pair with notify's fence: after publishing this link, the caller
        // reads dirty. Either that read sees the notification or notify's
        // link read sees this node; both reads cannot miss publication.
        fence(Ordering::SeqCst);
    }

    pub(crate) fn notify(&self) {
        self.dirty.store(true, Ordering::Release);
        fence(Ordering::SeqCst);
        let mut next = self.head.get();
        // OnceLock::get never blocks, including during parent registration.
        while let Some(node) = next {
            node.epoch.fetch_add(1, Ordering::Release);
            node.word.fetch_or(DISPATCH_CHILD_DIRTY, Ordering::Release);
            next = node.next.get();
        }
    }

    pub(crate) fn epoch(&self, word: &AtomicU64) -> Option<&AtomicU64> {
        let mut next = self.head.get();
        while let Some(node) = next {
            if std::ptr::eq(&*node.word, word) {
                return Some(&node.epoch);
            }
            next = node.next.get();
        }
        // Publication drains also use temporary, unregistered words.
        None
    }
}

impl Drop for ChildDispatch {
    fn drop(&mut self) {
        // Unlink iteratively in the parent, independent of Vec's drop order.
        // No child borrows remain; nodes have no owners outside this registry.
        self.head.take();
        let nodes = self.nodes.get_mut().unwrap_or_else(|e| e.into_inner());
        for node in nodes {
            Arc::get_mut(node).unwrap().next.take();
        }
    }
}

pub(crate) struct ChildWrites {
    pages: Vec<(u64, Box<[AtomicU64]>)>,
}

impl ChildWrites {
    pub(crate) fn new(regions: &[Region]) -> Self {
        Self {
            pages: regions
                .iter()
                .filter(|r| r.kind == RegionKind::Ram && r.end > r.start)
                .map(|r| {
                    let first = r.start >> CODE_PAGE_BITS;
                    let count = ((r.end - 1) >> CODE_PAGE_BITS) - first + 1;
                    let words = (0..count.div_ceil(64)).map(|_| AtomicU64::new(0)).collect();
                    (first, words)
                })
                .collect(),
        }
    }

    fn mark(&self, page: u64) {
        for (first, words) in &self.pages {
            if let Some(relative) = page.checked_sub(*first) {
                if let Some(word) = words.get((relative >> 6) as usize) {
                    word.fetch_or(1 << (relative & 63), Ordering::Release);
                    return;
                }
            }
        }
        unreachable!("child write outside snapshotted RAM")
    }

    pub(crate) fn drain(&self, mut visit: impl FnMut(u64)) {
        for (first, words) in &self.pages {
            for (i, word) in words.iter().enumerate() {
                let mut bits = word.swap(0, Ordering::Acquire);
                while bits != 0 {
                    visit(first + (i as u64 * 64) + u64::from(bits.trailing_zeros()));
                    bits &= bits - 1;
                }
            }
        }
    }
}

pub(crate) struct CodeSnapshot {
    pages: Vec<(u64, u64)>,
    pub(crate) consistency: crate::MemConsistency,
}

impl FromIterator<(u64, u64)> for CodeSnapshot {
    fn from_iter<T: IntoIterator<Item = (u64, u64)>>(iter: T) -> Self {
        Self {
            pages: iter.into_iter().collect(),
            consistency: crate::MemConsistency::Fast,
        }
    }
}

/// Metadata snapshot borrowing its VM's stable atomic store bridge. The marker
/// prevents safe Rust from dropping the VM or unmapping/remapping its backing.
/// Stepping borrows only the live code-tag bitmap and atomic store-notification
/// bridge from the VM; mapping, fetch policy and CPU metadata are owned snapshots.
/// Guest bytes remain shared; mapping changes require external synchronization.
/// Drop it in the parent after child exit/exec and written-page reconciliation.
/// Parent-side drop uses Memory to release deferred alias tags and history.
pub struct ChildView<'vm> {
    pub(crate) base: usize,
    pub(crate) guest_base: u64,
    pub(crate) top: u64,
    pub(crate) regions: Vec<Region>,
    pub(crate) code: CodeSnapshot,
    pub(crate) live_code: &'vm [AtomicU64],
    pub(crate) parent: &'vm Memory,
    pub(crate) writes: Arc<ChildWrites>,
    pub(crate) dispatch: &'vm ChildDispatch,
    #[cfg(test)]
    pub(crate) before_tag_read: Option<&'vm (dyn Fn() + Sync)>,
    #[cfg(test)]
    pub(crate) before_journal: Option<&'vm (dyn Fn() + Sync)>,
    pub(crate) fetch: Option<FrozenFetch>,
    pub(crate) recover: bool,
    pub(crate) mode: CpuMode,
    pub(crate) features: crate::GuestCpuFeatures,
    pub(crate) identity: Option<crate::CpuidTable>,
    pub(crate) x87_pointer_model: crate::X87PointerModel,
    pub(crate) lifetime: PhantomData<&'vm Vm>,
}

impl Drop for ChildView<'_> {
    fn drop(&mut self) {
        self.parent.release_child_view(&self.writes);
    }
}

impl ChildView<'_> {
    pub fn host_fault_recovery(&self) -> bool {
        self.recover
    }
    pub fn cpu_mode(&self) -> CpuMode {
        self.mode
    }
    pub fn mem_consistency(&self) -> crate::MemConsistency {
        self.code.consistency
    }
    pub fn guest_cpu_features(&self) -> crate::GuestCpuFeatures {
        self.features
    }
    pub fn cpuid_table(&self) -> Option<&crate::CpuidTable> {
        self.identity.as_ref()
    }
    pub fn x87_pointer_model(&self) -> crate::X87PointerModel {
        self.x87_pointer_model
    }

    fn region(&self, addr: u64, size: u8) -> Option<&Region> {
        let end = addr.checked_add(u64::from(size))?;
        if addr < self.guest_base || end > self.top {
            return None;
        }
        self.regions
            .iter()
            .find(|r| r.start <= addr && end <= r.end)
    }
    /// Kind of a non-atomic data span: the region containing it or, as the
    /// parent's `Memory::adjacent_span` accepts one, the kind shared by the
    /// adjacent regions covering it (all RAM or all ROM). Both kinds share one
    /// backing, so the span is one host range. A gap, a Trap byte and a span that
    /// mixes RAM and ROM are `None`: the child keeps the containment fault there.
    fn data_kind(&self, addr: u64, size: u8) -> Option<RegionKind> {
        match self.region(addr, size) {
            Some(r) => Some(r.kind),
            None => self.kind_across(addr, size),
        }
    }
    #[cold]
    #[inline(never)]
    fn kind_across(&self, addr: u64, size: u8) -> Option<RegionKind> {
        let end = addr.checked_add(u64::from(size))?;
        if size == 0 || addr < self.guest_base || end > self.top {
            return None;
        }
        let mut at = addr;
        let mut kind = None;
        while at < end {
            let r = self.regions.iter().find(|r| r.start <= at && at < r.end)?;
            if r.kind == RegionKind::Trap || kind.is_some_and(|k| k != r.kind) {
                return None;
            }
            kind = Some(r.kind);
            at = end.min(r.end);
        }
        kind
    }
    fn code_page(&self, page: u64) -> bool {
        let relative = page.wrapping_sub(self.guest_base >> CODE_PAGE_BITS);
        self.code
            .pages
            .binary_search_by_key(&(relative >> 6), |&(word, _)| word)
            .ok()
            .is_some_and(|i| self.code.pages[i].1 & (1 << (relative & 63)) != 0)
    }

    fn note_write(&self, addr: u64, size: u8) {
        #[cfg(test)]
        if let Some(hook) = self.before_journal {
            hook();
        }
        let first = addr >> CODE_PAGE_BITS;
        let last = (addr + u64::from(size) - 1) >> CODE_PAGE_BITS;
        for page in first..=last {
            self.writes.mark(page);
        }
        #[cfg(test)]
        if let Some(hook) = self.before_tag_read {
            hook();
        }
        // §10: store -> journal -> full fence -> live tag check. A shared-VM
        // child can be a separate process, outside PRIVATE_EXPEDITED's scope.
        fence(Ordering::SeqCst);
        if (first..=last).any(|page| {
            let relative = page.wrapping_sub(self.guest_base >> CODE_PAGE_BITS);
            self.live_code
                .get((relative >> 6) as usize)
                .is_some_and(|word| word.load(Ordering::Relaxed) & (1 << (relative & 63)) != 0)
        }) {
            self.dispatch.notify();
        }
    }
}

/// Why the step could not be attempted. A refusal leaves CPU and guest bytes
/// unchanged, including when it is discovered in a later vector/string transfer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChildRefusal {
    X87,
    FxState,
    /// Gather helpers require fault metadata and partial vector restart state.
    Gather,
    Helper,
    Mmio,
    CodeStore,
    /// Split cache-line/page operands, or alignment unsupported by the host.
    AtomicAlignment,
    /// Host lacks a required atomic instruction.
    AtomicUnavailable,
    ArenaFull,
    UnsupportedInstruction,
}

/// Fixed-size exception evidence. It has no heap owner and can cross the child
/// boundary by value. Convert it to ordinary owned metadata only in the parent.
#[derive(Clone, Debug)]
pub struct ChildFault {
    pub path: FaultPath,
    pub origin: FaultOrigin,
    pub class: FaultClass,
    pub vector: Knowledge<u8>,
    pub error_code: ErrorCode,
    pub saved_rip: u64,
    pub instruction_rip: u64,
    pub instruction_len: Knowledge<u8>,
    pub address: Knowledge<FaultAddress>,
    pub operand_base: Knowledge<u64>,
    pub operand_size: Knowledge<u16>,
    pub access: Knowledge<FaultAccess>,
    pub transfer_access: Knowledge<FaultAccess>,
    pub transfer_size: Knowledge<u16>,
    pub transfer_unit: TransferUnit,
    pub projection: Option<PageFaultProjection>,
    pub raw_host_fault: Option<RawHostFault>,
    pub registers: RegisterProgress,
    pub ram: MemoryEffects,
    pub rep: Option<RepRestart>,
}
impl ChildFault {
    fn new(rip: u64) -> Self {
        Self {
            path: FaultPath::Interpreter,
            origin: FaultOrigin::Unknown,
            class: FaultClass::Unknown,
            vector: Knowledge::Unknown,
            error_code: ErrorCode::Unknown,
            saved_rip: rip,
            instruction_rip: rip,
            instruction_len: Knowledge::Unknown,
            address: Knowledge::NotApplicable,
            operand_base: Knowledge::NotApplicable,
            operand_size: Knowledge::NotApplicable,
            access: Knowledge::NotApplicable,
            transfer_access: Knowledge::NotApplicable,
            transfer_size: Knowledge::NotApplicable,
            transfer_unit: TransferUnit::Bytes,
            projection: None,
            raw_host_fault: None,
            registers: RegisterProgress::InstructionEntry,
            ram: MemoryEffects::None,
            rep: None,
        }
    }
    /// Parent-side conversion. The child must retain the plain record instead.
    pub fn into_metadata(self) -> FaultMetadata {
        let mut m = FaultMetadata::unknown(self.saved_rip);
        m.path = self.path;
        m.origin = self.origin;
        m.class = self.class;
        m.vector = self.vector;
        m.error_code = self.error_code;
        m.instruction_rip = Knowledge::Known(self.instruction_rip);
        m.instruction_len = self.instruction_len;
        m.address = self.address;
        m.operand_base = self.operand_base;
        m.operand_size = self.operand_size;
        m.access = self.access;
        m.transfer_access = self.transfer_access;
        m.transfer_size = self.transfer_size;
        m.transfer_unit = self.transfer_unit;
        m.projection = self.projection;
        m.raw_host_fault = self.raw_host_fault;
        m.progress.registers = self.registers;
        m.progress.ram = self.ram;
        m.progress.device = MemoryEffects::None;
        m.progress.rep = self.rep;
        m
    }
}

/// A plain child exit, never an allocating `Exit` with boxed metadata.
#[derive(Clone, Debug)]
#[non_exhaustive]
#[allow(clippy::large_enum_variant)]
pub enum ChildStep {
    Continue,
    Syscall {
        evidence: crate::ExecutionEvidence,
        instruction_len: u8,
    },
    Fault(ChildFault),
    Refused(ChildRefusal),
    Hlt,
    PortIo {
        port: u16,
        size: u8,
        dir: crate::PortDir,
        value: u64,
    },
}

#[derive(Clone, Copy, Default)]
struct Store {
    addr: u64,
    value: u64,
    size: u8,
    access: MemoryAccess,
    base: u64,
    canonical: Option<(u64, u64)>,
}
struct StepMemory {
    outcome: Option<ChildStep>,
    rip: u64,
    access: MemoryAccess,
    operand: Option<u64>,
    canonical: Option<(u64, u64)>,
    rmw: bool,
    stores: [Store; 64],
    stores_len: usize,
    written: Vec<u64>,
    writing: bool,
    partial: bool,
}

/// Storage for one child, allocated before the fork. `capacity` bounds the
/// distinct written-page log. IR and temporaries have fixed one-instruction
/// budgets; exhausting any budget refuses before changing guest state.
/// Use one arena per concurrent child. Never drop it in the child; let the
/// embedder exit/exec without running destructors over inherited allocations.
pub struct ChildArena {
    ir: IrBlock,
    temps: Vec<u64>,
    cpu: CpuState,
    memory: RefCell<StepMemory>,
    recovery: UnsafeCell<transfer::Record>,
    atomics: transfer::AtomicFeatures,
}
impl ChildArena {
    pub fn new(capacity: usize) -> Self {
        // iced initializes immutable decoder tables here, before child entry.
        let mut decoder = iced_x86::Decoder::new(64, &[0x90], iced_x86::DecoderOptions::NONE);
        let _ = decoder.decode();
        Self {
            ir: IrBlock {
                legacy_context: None,
                mode: crate::CpuMode::Long64,
                guest_start: 0,
                guest_len: 0,
                icount: 1,
                ops: Vec::with_capacity(1024),
                temp_count: 0,
            },
            temps: Vec::with_capacity(1024),
            cpu: CpuState::default(),
            memory: RefCell::new(StepMemory {
                outcome: None,
                rip: 0,
                access: MemoryAccess::default(),
                operand: None,
                canonical: None,
                rmw: false,
                stores: [Store::default(); 64],
                stores_len: 0,
                written: Vec::with_capacity(capacity),
                writing: false,
                partial: false,
            }),
            recovery: UnsafeCell::new(transfer::Record::default()),
            atomics: transfer::AtomicFeatures::detect(),
        }
    }
    /// Guest 4 KiB page numbers; valid until the parent drops the arena.
    pub fn written_pages(&self) -> std::cell::Ref<'_, [u64]> {
        std::cell::Ref::map(self.memory.borrow(), |m| m.written.as_slice())
    }

    /// Use the exclusive-loop atomic fallback even when the host supports LSE.
    /// Configure in the parent before entering the child.
    #[cfg(target_arch = "aarch64")]
    pub fn disable_lse_atomics(&mut self) {
        self.atomics.lse = false;
    }
}

struct ChildMemory<'a, 'vm, const ORDER: u8> {
    view: &'a ChildView<'vm>,
    state: &'a RefCell<StepMemory>,
    recovery: &'a UnsafeCell<transfer::Record>,
    atomics: &'a transfer::AtomicFeatures,
}
impl<const ORDER: u8> ChildMemory<'_, '_, ORDER> {
    fn refuse(&self, why: ChildRefusal) {
        let mut s = self.state.borrow_mut();
        if s.outcome.is_none() {
            s.outcome = Some(ChildStep::Refused(why));
        }
    }
    fn exception_fault(&self, vector: u8, cause: FaultCause, advance: u8) {
        let mut s = self.state.borrow_mut();
        if s.outcome.is_some() {
            return;
        }
        let mut f = ChildFault::new(s.rip);
        // The shared cause-to-code projection encodes SDM Vol 3A §7.13/§7.15.
        // Its empty retry ledger has no allocation.
        let m =
            FaultMetadata::exception(s.rip.wrapping_add(u64::from(advance)), s.rip, vector, cause);
        if cause == FaultCause::Divide {
            f.path = FaultPath::Helper {
                family: FaultHelper::Divide,
                caller: HelperCaller::Interpreter,
            };
        }
        if advance != 0 {
            f.registers = RegisterProgress::CompletedInstruction;
        }
        f.saved_rip = m.saved_rip;
        f.origin = m.origin;
        f.vector = m.vector;
        f.class = m.class;
        f.error_code = m.error_code;
        s.outcome = Some(ChildStep::Fault(f));
    }
    fn fault(&self, addr: u64, size: u8, direction: FaultAccess, host: bool, exact: bool) {
        let mut s = self.state.borrow_mut();
        if s.outcome.is_some() {
            return;
        }
        let mut f = ChildFault::new(s.rip);
        f.origin = if host {
            FaultOrigin::HostAccess
        } else {
            FaultOrigin::GuestMap
        };
        f.address = Knowledge::Known(FaultAddress::new(
            addr,
            if exact {
                AddressPrecision::ExactByte
            } else {
                AddressPrecision::TransferAddress
            },
            AddressSpace::Linear,
        ));
        f.operand_base = s.operand.map_or(Knowledge::Known(addr), Knowledge::Known);
        f.operand_size = Knowledge::Known(s.access.bytes().max(u16::from(size)));
        f.access = Knowledge::Known(if s.rmw {
            FaultAccess::ReadWrite
        } else {
            direction
        });
        f.transfer_access = Knowledge::Known(direction);
        f.transfer_size = Knowledge::Known(u16::from(size));
        f.ram = if s.partial {
            MemoryEffects::PossiblyPartial
        } else {
            MemoryEffects::None
        };
        if host {
            // SAFETY: synchronous recovery has completed on this arena's thread.
            f.raw_host_fault = unsafe { (*self.recovery.get()).raw.take() };
        } else {
            f.projection = Some(PageFaultProjection {
                code: ErrorCode::bits(0, 1),
                is_fetch: direction == FaultAccess::Execute,
            });
        }
        s.outcome = Some(ChildStep::Fault(f));
    }
    /// `across` admits a data span over adjacent regions ([`ChildView::data_kind`]);
    /// an atomic operand passes `false` and keeps the containment fault, as in the
    /// parent.
    fn check(
        &self,
        addr: u64,
        size: u8,
        direction: FaultAccess,
        across: bool,
    ) -> Option<RegionKind> {
        if self.stopped() {
            return None;
        }
        let mut s = self.state.borrow_mut();
        let base = *s.operand.get_or_insert(addr);
        let (canonical_base, width) = s
            .canonical
            .unwrap_or((base, u64::from(s.access.bytes().max(u16::from(size)))));
        // SDM Vol 1 §3.3.7.1: canonical endpoints; SS selects #SS(0).
        let bad = self.view.mode == CpuMode::Long64
            && width != 0
            && (!crate::memory::is_canonical(canonical_base)
                || !crate::memory::is_canonical(canonical_base.wrapping_add(width - 1)));
        let stack = s.access.stack();
        drop(s);
        if bad {
            self.exception_fault(if stack { 12 } else { 13 }, FaultCause::NonCanonical, 0);
            return None;
        }
        match self.view.region(addr, size) {
            Some(r) if r.kind == RegionKind::Trap => {
                self.refuse(ChildRefusal::Mmio);
                None
            }
            Some(r) => Some(r.kind),
            None => {
                if across {
                    if let Some(kind) = self.view.kind_across(addr, size) {
                        return Some(kind);
                    }
                }
                let at = (0..size)
                    .map(|i| addr.wrapping_add(u64::from(i)))
                    .find(|&a| self.view.region(a, 1).is_none())
                    .unwrap_or(addr);
                self.fault(at, size, direction, false, true);
                None
            }
        }
    }
    fn reserve_write(&self, addr: u64, size: u8) -> bool {
        let first = addr >> CODE_PAGE_BITS;
        let last = addr.saturating_add(u64::from(size) - 1) >> CODE_PAGE_BITS;
        for page in first..=last {
            if self.view.code_page(page) {
                self.refuse(ChildRefusal::CodeStore);
                return false;
            }
        }
        let mut s = self.state.borrow_mut();
        for page in first..=last {
            if !s.written.contains(&page) {
                if s.written.len() == s.written.capacity() {
                    drop(s);
                    self.refuse(ChildRefusal::ArenaFull);
                    return false;
                }
                s.written.push(page);
            }
        }
        true
    }
    fn load(&self, addr: u64, size: u8) -> u64 {
        if self.check(addr, size, FaultAccess::Read, true).is_none() {
            return 0;
        }
        // SAFETY: the snapshot's lifetime keeps this validated backing live.
        match unsafe {
            transfer::load(
                self.view.base + (addr - self.view.guest_base) as usize,
                size,
                self.recovery.get(),
            )
        } {
            Ok(value) => value,
            Err(at) => {
                self.fault(
                    self.view.guest_base + (at - self.view.base) as u64,
                    size,
                    FaultAccess::Read,
                    true,
                    true,
                );
                0
            }
        }
    }
    fn compare_exchange(&self, addr: u64, expected: u128, src: u128, size: u8) -> Option<u128> {
        let kind = self.check(addr, size, FaultAccess::Write, false)?;
        if kind != RegionKind::Ram {
            self.refuse(ChildRefusal::Mmio);
            return None;
        }
        let ptr = self.view.base + (addr - self.view.guest_base) as usize;
        if (addr & 63) + u64::from(size) > 64
            || (ptr & 63) + usize::from(size) > 64
            || ((cfg!(target_arch = "aarch64") || size == 16) && ptr & (usize::from(size) - 1) != 0)
        {
            self.refuse(ChildRefusal::AtomicAlignment);
            return None;
        }
        if size == 16 && !self.atomics.pair {
            self.refuse(ChildRefusal::AtomicUnavailable);
            return None;
        }
        let written_len = self.state.borrow().written.len();
        if !self.reserve_write(addr, size) {
            return None;
        }
        // SAFETY: checked RAM backing, alignment and feature support; the arena's
        // recovery record remains live through the dedicated atomic fixup range.
        match unsafe {
            transfer::compare_exchange(ptr, expected, src, size, self.recovery.get(), self.atomics)
        } {
            Ok(old) => {
                self.view.note_write(addr, size);
                Some(old)
            }
            Err(()) => {
                self.state.borrow_mut().written.truncate(written_len);
                self.fault(addr, size, FaultAccess::Write, true, true);
                None
            }
        }
    }

    fn atomic(&self, op: &IrOp, temps: &mut [u64], cpu: &mut CpuState) {
        self.state.borrow_mut().rmw = true;
        match *op {
            IrOp::AtomicRmw {
                old,
                addr,
                src,
                size,
                op,
                ..
            } => {
                let addr = read_value(addr, temps);
                let src = read_value(src, temps);
                let mask = u64::MAX >> (64 - size * 8);
                let mut expected = 0u64;
                loop {
                    use crate::ir::RmwOp;
                    let next = match op {
                        RmwOp::Add => expected.wrapping_add(src),
                        RmwOp::Sub => expected.wrapping_sub(src),
                        RmwOp::Rsub => src.wrapping_sub(expected),
                        RmwOp::And => expected & src,
                        RmwOp::Or => expected | src,
                        RmwOp::Xor => expected ^ src,
                        RmwOp::Xchg => src,
                    } & mask;
                    let Some(observed) =
                        self.compare_exchange(addr, expected.into(), next.into(), size)
                    else {
                        return;
                    };
                    let observed = observed as u64;
                    if observed == expected {
                        temps[old as usize] = observed;
                        break;
                    }
                    expected = observed;
                }
            }
            IrOp::AtomicCas {
                old,
                addr,
                expected,
                src,
                size,
                ..
            } => {
                let mask = u64::MAX >> (64 - size * 8);
                if let Some(observed) = self.compare_exchange(
                    read_value(addr, temps),
                    (read_value(expected, temps) & mask).into(),
                    read_value(src, temps).into(),
                    size,
                ) {
                    temps[old as usize] = observed as u64;
                }
            }
            IrOp::AtomicCasPair { addr, size, .. } => {
                let addr = read_value(addr, temps);
                // SDM Vol 2A §3.2, CMPXCHG8B/CMPXCHG16B, 64-Bit Mode
                // Exceptions and Operation: 16-byte alignment; only ZF changes.
                if size == 16 && addr & 15 != 0 {
                    self.exception_fault(13, FaultCause::RequiredAlignment, 0);
                    return;
                }
                let bits = size * 4;
                let mask = u64::MAX >> (64 - bits);
                let expected =
                    (u128::from(cpu.gpr[2] & mask) << bits) | u128::from(cpu.gpr[0] & mask);
                let src = (u128::from(cpu.gpr[1] & mask) << bits) | u128::from(cpu.gpr[3] & mask);
                if let Some(old) = self.compare_exchange(addr, expected, src, size) {
                    cpu.flags.zf = old == expected;
                    if old != expected {
                        cpu.write_gpr(0, old as u64, size / 2);
                        cpu.write_gpr(2, (old >> bits) as u64, size / 2);
                    }
                }
            }
            _ => unreachable!(),
        }
    }
    fn store(&self, addr: u64, value: u64, size: u8) {
        let Some(kind) = self.check(addr, size, FaultAccess::Write, true) else {
            return;
        };
        if kind == RegionKind::Rom {
            return;
        }
        let written_len = self.state.borrow().written.len();
        if !self.reserve_write(addr, size) {
            return;
        }
        let mut s = self.state.borrow_mut();
        if !s.writing {
            if s.stores_len == s.stores.len() {
                drop(s);
                self.refuse(ChildRefusal::ArenaFull);
                return;
            }
            let n = s.stores_len;
            s.stores[n] = Store {
                addr,
                value,
                size,
                access: s.access,
                base: s.operand.unwrap_or(addr),
                canonical: s.canonical,
            };
            s.stores_len += 1;
            return;
        }
        drop(s);
        let ptr = self.view.base + (addr - self.view.guest_base) as usize;
        #[cfg(test)]
        crate::interp::ordering::EVENTS.with(|events| events.borrow_mut().push("commit"));
        // SAFETY: validated live backing; the recovery record is exclusive to this step.
        match unsafe { transfer::store(ptr, value, size, self.recovery.get()) } {
            Ok(()) => {
                self.view.note_write(addr, size);
                self.state.borrow_mut().partial = true;
            }
            Err(at) => {
                self.state.borrow_mut().written.truncate(written_len);
                // Cross-page transfers write bytewise; retain only their completed prefix.
                if at > ptr {
                    self.reserve_write(addr, (at - ptr) as u8);
                    self.view.note_write(addr, (at - ptr) as u8);
                    self.state.borrow_mut().partial = true;
                }
                self.fault(
                    self.view.guest_base + (at - self.view.base) as u64,
                    size,
                    FaultAccess::Write,
                    true,
                    true,
                );
            }
        }
    }
}
impl<const ORDER: u8> InterpreterMemory for ChildMemory<'_, '_, ORDER> {
    fn child_atomic(&self, op: &IrOp, temps: &mut [u64], cpu: &mut CpuState) {
        self.atomic(op, temps, cpu);
    }

    fn normal(&self) -> &Memory {
        panic!("parent memory reached by child execution")
    }
    fn read<const ROUTED: bool>(&self, addr: u64, size: u8) -> Result<u64, MemTrap> {
        Ok(self.load(addr, size))
    }
    fn write<const ROUTED: bool>(&self, addr: u64, value: u64, size: u8) -> Result<(), MemTrap> {
        self.store(addr, value, size);
        Ok(())
    }
    fn probe_write<const ROUTED: bool>(&self, addr: u64, size: u8) -> Result<(), MemTrap> {
        self.check(addr, size, FaultAccess::Write, true);
        Ok(())
    }
    fn stopped(&self) -> bool {
        self.state.borrow().outcome.is_some()
    }
    fn begin_op(&self, op: &IrOp, temps: &[u64], cpu: &CpuState) {
        let descriptor = match op {
            IrOp::Load { fault, addr, .. }
            | IrOp::Store { fault, addr, .. }
            | IrOp::X87 { fault, addr, .. }
            | IrOp::FxState { fault, addr, .. }
            | IrOp::AtomicRmw { fault, addr, .. }
            | IrOp::AtomicCas { fault, addr, .. }
            | IrOp::AtomicCasPair { fault, addr, .. }
            | IrOp::VLoad { fault, addr, .. }
            | IrOp::VStore { fault, addr, .. }
            | IrOp::VPackedBinM { fault, addr, .. }
            | IrOp::VLogicM { fault, addr, .. }
            | IrOp::VBlendWM { fault, addr, .. }
            | IrOp::VFmaM { fault, addr, .. }
            | IrOp::VAesM { fault, addr, .. }
            | IrOp::VAesImcM { fault, addr, .. }
            | IrOp::VAesKeygenM { fault, addr, .. }
            | IrOp::VShaM { fault, addr, .. }
            | IrOp::VGfniM { fault, addr, .. }
            | IrOp::VPclmulM { fault, addr, .. }
            | IrOp::VPsignM { fault, addr, .. }
            | IrOp::VPackWideM { fault, addr, .. }
            | IrOp::VPMaddM { fault, addr, .. }
            | IrOp::VShuffle16M { fault, addr, .. }
            | IrOp::VShufpsM { fault, addr, .. }
            | IrOp::VPshufbM { fault, addr, .. }
            | IrOp::VAlignrM { fault, addr, .. }
            | IrOp::VUnpackLowM { fault, addr, .. }
            | IrOp::VLoadWide { fault, addr, .. }
            | IrOp::VStoreWide { fault, addr, .. }
            | IrOp::VMaskLoadMem { fault, addr, .. }
            | IrOp::VMaskStoreMem { fault, addr, .. }
            | IrOp::VVecMaskLoadMem { fault, addr, .. }
            | IrOp::VVecMaskStoreMem { fault, addr, .. }
            | IrOp::VLogic256M { fault, addr, .. }
            | IrOp::VLogicWideM { fault, addr, .. }
            | IrOp::VPopcntM { fault, addr, .. }
            | IrOp::VExtractLaneWideM { fault, addr, .. }
            | IrOp::VPcmpStrM { fault, addr, .. }
            | IrOp::VPcmpStrMaskM { fault, addr, .. }
            | IrOp::VInsertPsM { fault, addr, .. }
            | IrOp::VInsertPsM3 { fault, addr, .. }
            | IrOp::VDppsM { fault, addr, .. }
            | IrOp::VPMovExtendM { fault, addr, .. }
            | IrOp::VPBlendVM { fault, addr, .. }
            | IrOp::VPBlendVXM { fault, addr, .. }
            | IrOp::VBlendIM { fault, addr, .. }
            | IrOp::VDppdM { fault, addr, .. }
            | IrOp::VDp3M { fault, addr, .. }
            | IrOp::VPRoundM { fault, addr, .. }
            | IrOp::VShiftRegM { fault, addr, .. }
            | IrOp::VGf2p8M { fault, addr, .. }
            | IrOp::VPTernlogM { fault, addr, .. }
            | IrOp::VPackedBin256M { fault, addr, .. }
            | IrOp::VPackedWideM { fault, addr, .. }
            | IrOp::VBroadcastM { fault, addr, .. }
            | IrOp::VBroadcastLaneM { fault, addr, .. }
            | IrOp::VPCmpToMaskM { fault, addr, .. }
            | IrOp::VPTestToMaskM { fault, addr, .. }
            | IrOp::VPmovNarrowMem { fault, addr, .. }
            | IrOp::VPermT2M { fault, addr, .. }
            | IrOp::VPerm1M { fault, addr, .. }
            | IrOp::VInsert128M { fault, addr, .. }
            | IrOp::VPshufb256M { fault, addr, .. }
            | IrOp::VPtestM { fault, addr, .. }
            | IrOp::VFloatBinM { fault, addr, .. }
            | IrOp::VHFloatM { fault, addr, .. }
            | IrOp::VHIntM { fault, addr, .. }
            | IrOp::VFloatCmpMaskM { fault, addr, .. }
            | IrOp::VFloatCmpMask256M { fault, addr, .. }
            | IrOp::VMpsadbwM { fault, addr, .. }
            | IrOp::VPermilVarM { fault, addr, .. }
            | IrOp::VPerm2i128M { fault, addr, .. }
            | IrOp::VShiftVarM { fault, addr, .. }
            | IrOp::VCvtPs2PhM { fault, addr, .. }
            | IrOp::VFloatBin256M { fault, addr, .. }
            | IrOp::VFloatUnary256M { fault, addr, .. }
            | IrOp::VPackedCvt256M { fault, addr, .. }
            | IrOp::VShufps256M { fault, addr, .. }
            | IrOp::VUnpack256M { fault, addr, .. }
            | IrOp::VLoadHalf { fault, addr, .. }
            | IrOp::VStoreHalf { fault, addr, .. } => Some((*fault, read_value(*addr, temps))),
            IrOp::VFloatUnaryM {
                fault, src_addr, ..
            } => Some((*fault, read_value(*src_addr, temps))),
            IrOp::PushFlags {
                fault, size: slot, ..
            } => Some((*fault, cpu.gpr[4].wrapping_sub(u64::from(*slot)))),
            IrOp::RepString { fault, .. } => Some((*fault, 0)),
            IrOp::Call { fault, slot, .. } => {
                Some((*fault, cpu.gpr[4].wrapping_sub(u64::from(*slot))))
            }
            IrOp::Ret { fault, .. } => Some((*fault, cpu.gpr[4])),
            _ => None,
        };
        let mut s = self.state.borrow_mut();
        s.access = op.memory_access();
        s.operand = descriptor.map(|(_, base)| base);
        s.canonical = descriptor
            .map(|(fault, base)| {
                let k_mask = |k: u8| {
                    if k == 0 {
                        u64::MAX
                    } else {
                        cpu.kmask[usize::from(k)]
                    }
                };
                let masked = match op {
                    IrOp::VMaskLoadMem {
                        k, elem, broadcast, ..
                    } => Some((broadcast.unwrap_or(*elem), k_mask(*k))),
                    IrOp::VMaskStoreMem { k, elem, .. } => Some((*elem, k_mask(*k))),
                    IrOp::VVecMaskLoadMem {
                        mask, elem, bytes, ..
                    }
                    | IrOp::VVecMaskStoreMem {
                        mask, elem, bytes, ..
                    } => Some((
                        *elem,
                        crate::interp::vec_msb_mask(
                            &cpu.vec_lanes(usize::from(*mask)),
                            *elem,
                            *bytes,
                        ),
                    )),
                    IrOp::VGf2p8M { k, mode, .. }
                        if *k != 0
                            && matches!(crate::GfniOp::from_u8(*mode), crate::GfniOp::Mulb) =>
                    {
                        Some((1, k_mask(*k)))
                    }
                    IrOp::VFmaM {
                        writemask: Some(k),
                        prec,
                        ..
                    } => Some((prec.bytes(), k_mask(*k))),
                    _ => None,
                };
                // Share the normal interpreter's canonical span for active elements
                // (SDM Vol 1 §15.6.1; Vol 2C §5.2 VMASKMOV, Description/Operation).
                match masked {
                    Some((elem, _)) if fault.bytes() == u16::from(elem) => {
                        (base, u64::from(fault.bytes()))
                    }
                    Some((elem, mask)) => crate::memory::active_operand_span(
                        base,
                        u64::from(fault.bytes()),
                        elem,
                        mask,
                    ),
                    None => (base, u64::from(fault.bytes())),
                }
            })
            .filter(|_| !matches!(op, IrOp::RepString { .. }));
    }
    fn exception(&self, vector: u8, cause: FaultCause, advance: u8) {
        self.exception_fault(vector, cause, advance);
    }
    fn string<const ROUTED: bool>(
        &self,
        cpu: &mut CpuState,
        op: StrOp,
        elem: u8,
        rep: RepKind,
        bits: u8,
        base: u64,
    ) {
        let mask = if bits == 32 {
            u64::from(u32::MAX)
        } else {
            u64::MAX
        };
        let count = if matches!(rep, RepKind::None) {
            1
        } else {
            cpu.gpr[1] & mask
        };
        if matches!(op, StrOp::Movs | StrOp::Stos) {
            let written_len = self.state.borrow().written.len();
            // Check every potential refusal before the first committed iteration.
            // Faults remain per-iteration so restart registers retain completed work.
            let delta = if cpu.flags.df {
                0u64.wrapping_sub(u64::from(elem))
            } else {
                u64::from(elem)
            };
            for i in 0..count {
                let dst = cpu.gpr[7].wrapping_add(i.wrapping_mul(delta)) & mask;
                let src = base.wrapping_add(cpu.gpr[6].wrapping_add(i.wrapping_mul(delta)) & mask);
                if self
                    .view
                    .region(dst, elem)
                    .is_some_and(|r| r.kind == RegionKind::Trap)
                    || (matches!(op, StrOp::Movs)
                        && self
                            .view
                            .region(src, elem)
                            .is_some_and(|r| r.kind == RegionKind::Trap))
                {
                    self.refuse(ChildRefusal::Mmio);
                    return;
                }
                if self.view.data_kind(dst, elem) == Some(RegionKind::Ram)
                    && !self.reserve_write(dst, elem)
                {
                    return;
                }
            }
            // Preflight reserves capacity; execution logs only completed transfers.
            self.state.borrow_mut().written.truncate(written_len);
        }
        self.state.borrow_mut().writing = true;
        let rip = self.state.borrow().rip;
        let _ordering = Barrier::<ORDER>::new(
            Some(match op {
                StrOp::Ins | StrOp::Outs => unreachable!("port strings use PortString"),
                StrOp::Movs => Direction::ReadWrite,
                StrOp::Stos => Direction::Write,
                StrOp::Lods | StrOp::Scas | StrOp::Cmps => Direction::Read,
            }),
            true,
        );
        crate::interp::string_run_mode::<true, _>(
            cpu, self, op, elem, rep, rip, bits, base, 0, None,
        );
        if let Some(ChildStep::Fault(f)) = &mut self.state.borrow_mut().outcome {
            f.path = FaultPath::Helper {
                family: FaultHelper::String,
                caller: HelperCaller::Interpreter,
            };
            f.registers = RegisterProgress::RepRestart;
            f.rep = Some(RepRestart {
                source: cpu.gpr[6] & mask,
                destination: cpu.gpr[7] & mask,
                remaining: cpu.gpr[1] & mask,
                address_bits: bits,
                element_size: elem,
                direction_flag: cpu.flags.df,
                flags: cpu.rflags(),
                operation: op,
            });
        }
    }
}
impl<const ORDER: u8> crate::interp::StrMem for ChildMemory<'_, '_, ORDER> {
    fn sload(&self, addr: u64, elem: u8) -> Result<u64, MemTrap> {
        {
            let mut s = self.state.borrow_mut();
            s.operand = None;
            s.canonical = None;
        }
        let v = self.load(addr, elem);
        #[cfg(test)]
        crate::interp::ordering::EVENTS.with(|events| events.borrow_mut().push("read"));
        if self.stopped() {
            Err(MemTrap::Unmapped)
        } else {
            Ok(v)
        }
    }
    fn sstore(&self, addr: u64, val: u64, elem: u8) -> Result<(), MemTrap> {
        {
            let mut s = self.state.borrow_mut();
            s.operand = None;
            s.canonical = None;
        }
        self.store(addr, val, elem);
        #[cfg(test)]
        crate::interp::ordering::EVENTS.with(|events| events.borrow_mut().push("write"));
        if self.stopped() {
            Err(MemTrap::Unmapped)
        } else {
            Ok(())
        }
    }
}

fn refusal(op: &IrOp) -> Option<ChildRefusal> {
    match op {
        IrOp::Legacy(_) => Some(ChildRefusal::Helper),
        IrOp::InsnStart { .. }
        | IrOp::ReadReg { .. }
        | IrOp::WriteReg { .. }
        | IrOp::Add { .. }
        | IrOp::Sub { .. }
        | IrOp::Adc { .. }
        | IrOp::Adox { .. }
        | IrOp::Sbb { .. }
        | IrOp::And { .. }
        | IrOp::Or { .. }
        | IrOp::Xor { .. }
        | IrOp::Shl { .. }
        | IrOp::Shr { .. }
        | IrOp::Sar { .. }
        | IrOp::DoubleShift { .. }
        | IrOp::Rol { .. }
        | IrOp::Ror { .. }
        | IrOp::Rcl { .. }
        | IrOp::Rcr { .. }
        | IrOp::Sext { .. }
        | IrOp::Bswap { .. }
        | IrOp::Mul { .. }
        | IrOp::Div { .. }
        | IrOp::GetCond { .. }
        | IrOp::Fence { .. }
        | IrOp::Load { .. }
        | IrOp::Store { .. }
        | IrOp::Bt { .. }
        | IrOp::GetMxcsr { .. }
        | IrOp::SetMxcsr { .. }
        | IrOp::Popcnt { .. }
        | IrOp::Crc32 { .. }
        | IrOp::BitScan { .. }
        | IrOp::Bmi { .. }
        | IrOp::MmxRead { .. }
        | IrOp::MmxWrite { .. }
        | IrOp::MmxState { .. }
        | IrOp::MmxConvert { .. }
        | IrOp::MmxCompute { .. }
        | IrOp::VLoad { .. }
        | IrOp::VStore { .. }
        | IrOp::VMov { .. }
        | IrOp::VMov256 { .. }
        | IrOp::VFromGpr { .. }
        | IrOp::VToGpr { .. }
        | IrOp::VReadChunk { .. }
        | IrOp::VWriteChunk { .. }
        | IrOp::VLogic { .. }
        | IrOp::VPackedBin { .. }
        | IrOp::VPackedBinM { .. }
        | IrOp::VLogicM { .. }
        | IrOp::VPackedShift { .. }
        | IrOp::VByteShift { .. }
        | IrOp::VShuffle32 { .. }
        | IrOp::VBlendW { .. }
        | IrOp::VBlendWM { .. }
        | IrOp::VBlendD { .. }
        | IrOp::VFma { .. }
        | IrOp::VFmaM { .. }
        | IrOp::VAes { .. }
        | IrOp::VAesM { .. }
        | IrOp::VAesImc { .. }
        | IrOp::VAesImcM { .. }
        | IrOp::VAesKeygen { .. }
        | IrOp::VAesKeygenM { .. }
        | IrOp::VSha { .. }
        | IrOp::VShaM { .. }
        | IrOp::VGfni { .. }
        | IrOp::VGfniM { .. }
        | IrOp::Movq2dq { .. }
        | IrOp::Movdq2q { .. }
        | IrOp::VPclmul { .. }
        | IrOp::VPclmulM { .. }
        | IrOp::VPsign { .. }
        | IrOp::VPsignM { .. }
        | IrOp::VPackWide { .. }
        | IrOp::VPackWideM { .. }
        | IrOp::VPMAddWd { .. }
        | IrOp::VPMadd { .. }
        | IrOp::VPMaddM { .. }
        | IrOp::VShuffle32Wide { .. }
        | IrOp::VShuffle16 { .. }
        | IrOp::VShuffle16M { .. }
        | IrOp::VShufps { .. }
        | IrOp::VShufpsM { .. }
        | IrOp::VPshufb { .. }
        | IrOp::VPshufbM { .. }
        | IrOp::VPshufbWide { .. }
        | IrOp::VAlignr { .. }
        | IrOp::VAlignrM { .. }
        | IrOp::VUnpackLow { .. }
        | IrOp::VUnpackLowM { .. }
        | IrOp::VPackUsWB { .. }
        | IrOp::VInsertW { .. }
        | IrOp::VExtractW { .. }
        | IrOp::VExtractLane { .. }
        | IrOp::VInsertLane { .. }
        | IrOp::VMoveMaskB { .. }
        | IrOp::VMoveMaskFp { .. }
        | IrOp::VZeroUpper { .. }
        | IrOp::VZeroUpperAll { .. }
        | IrOp::VLoadWide { .. }
        | IrOp::VStoreWide { .. }
        | IrOp::VMovWide { .. }
        | IrOp::VMaskMov { .. }
        | IrOp::VMaskLoadMem { .. }
        | IrOp::VMaskStoreMem { .. }
        | IrOp::VVecMaskLoadMem { .. }
        | IrOp::VVecMaskStoreMem { .. }
        | IrOp::VLogic256 { .. }
        | IrOp::VLogic256M { .. }
        | IrOp::VLogicWide { .. }
        | IrOp::VLogicWideM { .. }
        | IrOp::VPopcnt { .. }
        | IrOp::VPopcntM { .. }
        | IrOp::VInsertLaneWide { .. }
        | IrOp::VExtractLaneWide { .. }
        | IrOp::VExtractLaneWideM { .. }
        | IrOp::VPcmpStr { .. }
        | IrOp::VPcmpStrM { .. }
        | IrOp::VPcmpStrMask { .. }
        | IrOp::VPcmpStrMaskM { .. }
        | IrOp::VInsertPs { .. }
        | IrOp::VInsertPsM { .. }
        | IrOp::VInsertPs3 { .. }
        | IrOp::VInsertPsM3 { .. }
        | IrOp::VDpps { .. }
        | IrOp::VDppsM { .. }
        | IrOp::VAlign { .. }
        | IrOp::VPMovExtend { .. }
        | IrOp::VPMovExtendM { .. }
        | IrOp::VPMovExtendWide { .. }
        | IrOp::VPAbs { .. }
        | IrOp::VpUnaryLane { .. }
        | IrOp::VpBlendm { .. }
        | IrOp::VShuffLane { .. }
        | IrOp::VpMultishift { .. }
        | IrOp::VPBlendV { .. }
        | IrOp::VPBlendVM { .. }
        | IrOp::VPBlendVX { .. }
        | IrOp::VPBlendVXM { .. }
        | IrOp::VBlendI { .. }
        | IrOp::VBlendIM { .. }
        | IrOp::VDppd { .. }
        | IrOp::VDppdM { .. }
        | IrOp::VDp3 { .. }
        | IrOp::VDp3M { .. }
        | IrOp::VPRound { .. }
        | IrOp::VPRoundM { .. }
        | IrOp::VMaskedLogic { .. }
        | IrOp::VMaskedPacked { .. }
        | IrOp::VMaskedShift { .. }
        | IrOp::VShiftReg { .. }
        | IrOp::VShiftRegM { .. }
        | IrOp::VShiftVar { .. }
        | IrOp::VGf2p8 { .. }
        | IrOp::VGf2p8M { .. }
        | IrOp::VPTernlog { .. }
        | IrOp::VPTernlogM { .. }
        | IrOp::VPackedBin256 { .. }
        | IrOp::VPackedBin256M { .. }
        | IrOp::VPackedWide { .. }
        | IrOp::VPackedWideM { .. }
        | IrOp::VMoveMaskB256 { .. }
        | IrOp::VBroadcast { .. }
        | IrOp::VBroadcastM { .. }
        | IrOp::VBroadcastGpr { .. }
        | IrOp::VBroadcastLane { .. }
        | IrOp::VBroadcastLaneM { .. }
        | IrOp::VPCmpToMask { .. }
        | IrOp::VPCmpToMaskM { .. }
        | IrOp::VPTestToMask { .. }
        | IrOp::VPTestToMaskM { .. }
        | IrOp::VKOrTest { .. }
        | IrOp::VKFromGpr { .. }
        | IrOp::VKToGpr { .. }
        | IrOp::VKMovKK { .. }
        | IrOp::VKUnpack { .. }
        | IrOp::VKBinOp { .. }
        | IrOp::VKNot { .. }
        | IrOp::VKShift { .. }
        | IrOp::VPmovNarrow { .. }
        | IrOp::VPmovNarrowMem { .. }
        | IrOp::VPermT2 { .. }
        | IrOp::VPermT2M { .. }
        | IrOp::VPerm1 { .. }
        | IrOp::VPerm1M { .. }
        | IrOp::VInsert128 { .. }
        | IrOp::VInsert128M { .. }
        | IrOp::VExtract128 { .. }
        | IrOp::VPshufb256 { .. }
        | IrOp::VPshufb256M { .. }
        | IrOp::VPackedShift256 { .. }
        | IrOp::VPermq { .. }
        | IrOp::VPermd { .. }
        | IrOp::VPermilVar { .. }
        | IrOp::VPerm2i128 { .. }
        | IrOp::VPalignr256 { .. }
        | IrOp::VPtest { .. }
        | IrOp::VPtestM { .. }
        | IrOp::VTestFp { .. }
        | IrOp::VFloatBin { .. }
        | IrOp::VFloatBinM { .. }
        | IrOp::VHFloat { .. }
        | IrOp::VHFloatM { .. }
        | IrOp::VHInt { .. }
        | IrOp::VHIntM { .. }
        | IrOp::VFloatMov { .. }
        | IrOp::VFloatCmp { .. }
        | IrOp::VFloatCmpMask { .. }
        | IrOp::VFloatCmpMaskM { .. }
        | IrOp::VFloatCmpMask256 { .. }
        | IrOp::VFloatCmpMask256M { .. }
        | IrOp::VCvtFromInt { .. }
        | IrOp::VCvtToInt { .. }
        | IrOp::VCvtFloat { .. }
        | IrOp::VPackedCvt { .. }
        | IrOp::VPhMinPosUw { .. }
        | IrOp::VMpsadbw { .. }
        | IrOp::VMpsadbwM { .. }
        | IrOp::VCvtPh2Ps { .. }
        | IrOp::VPermilVarM { .. }
        | IrOp::VPerm2i128M { .. }
        | IrOp::VShiftVarM { .. }
        | IrOp::VCvtPs2PhM { .. }
        | IrOp::VCvtPs2Ph { .. }
        | IrOp::VPackedCvtWide256 { .. }
        | IrOp::VFloatUnary { .. }
        | IrOp::VFloatUnaryM { .. }
        | IrOp::VFloatBin256 { .. }
        | IrOp::VFloatBin256M { .. }
        | IrOp::VFloatUnary256 { .. }
        | IrOp::VFloatUnary256M { .. }
        | IrOp::VPackedCvt256 { .. }
        | IrOp::VPackedCvt256M { .. }
        | IrOp::VShufps256 { .. }
        | IrOp::VShufps256M { .. }
        | IrOp::VUnpack256 { .. }
        | IrOp::VUnpack256M { .. }
        | IrOp::VMoveHalf { .. }
        | IrOp::VLoadHalf { .. }
        | IrOp::VStoreHalf { .. }
        | IrOp::SetDf { .. }
        | IrOp::PushFlags { .. }
        | IrOp::RestoreFlags { .. }
        | IrOp::ReadSelector { .. }
        | IrOp::SetIf { .. }
        | IrOp::SetSsShadow
        | IrOp::SetCf { .. }
        | IrOp::LoopCx { .. }
        | IrOp::RepString { .. }
        | IrOp::Lahf
        | IrOp::Sahf
        | IrOp::Jump { .. }
        | IrOp::Branch { .. }
        | IrOp::Call { .. }
        | IrOp::Ret { .. }
        | IrOp::Syscall { .. }
        | IrOp::Hlt
        | IrOp::Trap { .. }
        | IrOp::PortIo { .. }
        | IrOp::CheckAlign { .. } => None,
        IrOp::Cpuid => Some(ChildRefusal::Helper),
        IrOp::ReadTsc { .. } => Some(ChildRefusal::Helper),
        IrOp::Random { .. } => Some(ChildRefusal::Helper),
        IrOp::Xgetbv => Some(ChildRefusal::Helper),
        IrOp::X87 { .. } => Some(ChildRefusal::X87),
        IrOp::FxState { .. } => Some(ChildRefusal::FxState),
        IrOp::XState { .. } => Some(ChildRefusal::FxState),
        IrOp::VGather { .. } => Some(ChildRefusal::Gather),
        IrOp::AtomicRmw { .. } | IrOp::AtomicCas { .. } | IrOp::AtomicCasPair { .. } => None,
        IrOp::CheckIretNt => Some(ChildRefusal::Helper),
        IrOp::CheckIret { .. } => Some(ChildRefusal::Helper),
        IrOp::Unsupported { .. } => Some(ChildRefusal::UnsupportedInstruction),
        IrOp::PushfReal => Some(ChildRefusal::Helper),
        IrOp::PopfReal => Some(ChildRefusal::Helper),
        IrOp::IntGate { .. } => Some(ChildRefusal::Helper),
        IrOp::IntoGate { .. } => Some(ChildRefusal::Helper),
        IrOp::BoundGate { .. } => Some(ChildRefusal::Helper),
        IrOp::SegLimitCheck { .. } => Some(ChildRefusal::Helper),
        IrOp::IretReal => Some(ChildRefusal::Helper),
        IrOp::Bcd { .. } => Some(ChildRefusal::Helper),
        IrOp::FarJump { .. } => Some(ChildRefusal::Helper),
        IrOp::FarCall { .. } => Some(ChildRefusal::Helper),
        IrOp::FarRet { .. } => Some(ChildRefusal::Helper),
        IrOp::RepStringReal { .. } | IrOp::PortString { .. } => Some(ChildRefusal::Helper),
        IrOp::PushaReal => Some(ChildRefusal::Helper),
        IrOp::PopaReal => Some(ChildRefusal::Helper),
        IrOp::EnterReal { .. } => Some(ChildRefusal::Helper),
    }
}

/// Execute exactly one instruction using only this view, CPU and arena.
/// No allocation, lock, cache access, callback or parent-owned
/// destructor runs here. Snapshot/arena creation and metadata conversion belong in the parent.
/// Committed stores use the live code tags and atomic notification bridge (§10).
/// Install a child handler calling [`classify`] before entering a view configured
/// for host-fault recovery. Keep mappings live for the view's entire lifetime.
pub fn step(view: &ChildView<'_>, cpu: &mut CpuState, arena: &mut ChildArena) -> ChildStep {
    #[cfg(target_arch = "aarch64")]
    match view.mem_consistency() {
        crate::MemConsistency::AcqRel => return step_ordered::<1>(view, cpu, arena),
        crate::MemConsistency::FullTso => return step_ordered::<2>(view, cpu, arena),
        crate::MemConsistency::Fast => {}
    }
    step_ordered::<0>(view, cpu, arena)
}

fn step_ordered<const ORDER: u8>(
    view: &ChildView<'_>,
    cpu: &mut CpuState,
    arena: &mut ChildArena,
) -> ChildStep {
    let _fp = crate::jit_abi::fp::RunMode::new();
    if view.mode == CpuMode::Real16 {
        return ChildStep::Refused(ChildRefusal::Helper);
    }
    if cpu.mmio_retry_active || cpu.pending_mmio.is_some() || cpu.pending_mmio_write {
        return ChildStep::Refused(ChildRefusal::Mmio);
    }
    let ChildArena {
        ir,
        temps,
        cpu: work,
        memory,
        recovery,
        atomics,
    } = arena;
    let entry_log = memory.get_mut().written.len();
    {
        let s = memory.get_mut();
        s.outcome = None;
        s.rip = cpu.rip;
        s.stores_len = 0;
        s.writing = false;
        s.partial = false;
        s.access = MemoryAccess::default();
        s.operand = None;
        s.canonical = None;
        s.rmw = false;
    }
    recovery.get_mut().raw = None;
    let mem = ChildMemory::<ORDER> {
        view,
        state: memory,
        recovery,
        atomics,
    };
    let mut bytes = [0; 15];
    let mut decoded = false;
    for n in 0..15 {
        let addr = cpu.rip.wrapping_add(n as u64);
        if view
            .region(addr, 1)
            .is_some_and(|r| r.kind == RegionKind::Trap)
        {
            mem.refuse(ChildRefusal::Mmio);
            break;
        }
        if let Some(fetch) = &view.fetch {
            let verdict = fetch.verdict(addr);
            if verdict != FetchVerdict::Allow {
                mem.fault(addr, 1, FaultAccess::Execute, false, true);
                if let Some(ChildStep::Fault(f)) = &mut memory.borrow_mut().outcome {
                    f.projection = Some(PageFaultProjection {
                        code: ErrorCode::bits(
                            u32::from(verdict == FetchVerdict::DenyProtection),
                            1,
                        ),
                        is_fetch: true,
                    });
                }
                break;
            }
        }
        bytes[n] = mem.load(addr, 1) as u8;
        if mem.stopped() {
            break;
        }
        memory.borrow_mut().operand = None;
        match crate::lift::lift_child(&bytes[..=n], cpu.rip, view.mode, ir) {
            Ok(None) if n < 14 => continue,
            Ok(Some(false)) if ir.temp_count as usize <= temps.capacity() => {
                decoded = true;
                break;
            }
            Ok(Some(_)) => {
                mem.refuse(ChildRefusal::ArenaFull);
                break;
            }
            Ok(None) => {
                mem.exception_fault(13, FaultCause::InstructionLength, 0);
                break;
            }
            Err(_) => {
                let mut decoder = iced_x86::Decoder::new(
                    view.mode.bits(),
                    &bytes[..=n],
                    iced_x86::DecoderOptions::NONE,
                );
                let instruction = decoder.decode();
                use iced_x86::Mnemonic::*;
                let why = match instruction.mnemonic() {
                    Xsave | Xsave64 | Xsavec | Xsavec64 | Xsaveopt | Xsaveopt64 | Xsaves
                    | Xsaves64 | Xrstor | Xrstor64 | Xrstors | Xrstors64 => ChildRefusal::FxState,
                    Iret | Iretd | Iretq => ChildRefusal::Helper,
                    _ => ChildRefusal::UnsupportedInstruction,
                };
                mem.refuse(why);
                break;
            }
        }
    }
    if !decoded {
        let mut s = memory.borrow_mut();
        let mut result = s
            .outcome
            .take()
            .unwrap_or(ChildStep::Refused(ChildRefusal::UnsupportedInstruction));
        if let ChildStep::Fault(f) = &mut result {
            f.path = FaultPath::LifterFetch {
                first_unavailable: match f.address {
                    Knowledge::Known(a) => a.value(),
                    _ => cpu.rip,
                },
            };
            f.transfer_access = Knowledge::Known(FaultAccess::Execute);
            f.access = f.transfer_access;
            f.transfer_unit = TransferUnit::FetchByte;
            f.operand_base = Knowledge::Known(cpu.rip);
            f.operand_size = Knowledge::Unknown;
            if let Some(projection) = &mut f.projection {
                projection.is_fetch = true;
            }
        }
        s.written.truncate(entry_log);
        return result;
    }
    if let Some(why) = ir.ops.iter().find_map(refusal) {
        return ChildStep::Refused(why);
    }
    memory.borrow_mut().rmw = ir.ops.iter().any(|op| {
        matches!(
            op,
            IrOp::Store { .. } | IrOp::VStore { .. } | IrOp::VStoreWide { .. }
        )
    }) && ir.ops.iter().any(|op| matches!(op, IrOp::Load { .. }));
    copy_cpu(work, cpu);
    let result = crate::interp::walk_child::<ORDER>(ir, work, &mem, temps);
    if !mem.stopped() && !memory.borrow().writing {
        let len = memory.borrow().stores_len;
        {
            let mut s = memory.borrow_mut();
            if len != 0 {
                s.written.truncate(entry_log);
            }
            s.writing = true;
        }
        // The walk's store barriers surround staging, not the actual transfers.
        // Fence each commit so wide stores retain ordering between vector lanes.
        // SDM Vol 3A §11.2.2, §11.2.4.1: order the committed writes too.
        // classify resumes inside transfer::store and returns normally, so Drop
        // retains the post-fence even on a fault with a partially written prefix,
        // just as the ordinary interpreter's access/helper guards do.
        for i in 0..len {
            let store = memory.borrow().stores[i];
            {
                let mut s = memory.borrow_mut();
                s.access = store.access;
                s.operand = Some(store.base);
                s.canonical = store.canonical;
            }
            let _ordering = Barrier::<ORDER>::new(Some(Direction::Write), false);
            mem.store(store.addr, store.value, store.size);
            if mem.stopped() {
                break;
            }
        }
    }
    let outcome = memory.borrow_mut().outcome.take();
    match outcome {
        Some(ChildStep::Refused(why)) => {
            memory.borrow_mut().written.truncate(entry_log);
            ChildStep::Refused(why)
        }
        Some(ChildStep::Fault(mut f)) => {
            if !memory.borrow().writing {
                memory.borrow_mut().written.truncate(entry_log);
            }
            f.instruction_len = Knowledge::Known(ir.guest_len as u8);
            if f.rep.is_some() {
                copy_cpu(cpu, work);
            }
            cpu.rip = f.saved_rip;
            ChildStep::Fault(f)
        }
        Some(_) => unreachable!(),
        None => {
            copy_cpu(cpu, work);
            match result {
                crate::StepResult::Continue => ChildStep::Continue,
                crate::StepResult::Exit(crate::Exit::Syscall) => ChildStep::Syscall {
                    evidence: crate::ExecutionEvidence::new(FaultPath::Interpreter, ir.guest_start),
                    instruction_len: ir.guest_len as u8,
                },
                crate::StepResult::Exit(crate::Exit::Hlt) => ChildStep::Hlt,
                crate::StepResult::Exit(crate::Exit::PortIo {
                    port,
                    size,
                    dir,
                    value,
                }) => ChildStep::PortIo {
                    port,
                    size,
                    dir,
                    value,
                },
                _ => unreachable!("child scope admits only plain exits"),
            }
        }
    }
}

// Copy only plain CPU fields. In particular no Arc is cloned, borrowed or dropped;
// source/table owners supplied by the embedder stay entirely untouched.
fn read_value(v: crate::Val, temps: &[u64]) -> u64 {
    match v {
        crate::Val::Imm(v) => v,
        crate::Val::Temp(t) => temps[t as usize],
    }
}

fn copy_cpu(dst: &mut CpuState, src: &CpuState) {
    macro_rules! fields { ($($name:ident),* $(,)?) => { $(dst.$name = src.$name;)* }; }
    fields!(
        gpr,
        rip,
        fs_base,
        gs_base,
        flags,
        xmm,
        ymm_hi,
        zmm_hi,
        kmask,
        fpr,
        fpu_top,
        fpu_cw,
        fpu_pad,
        pending_mmio,
        pending_mmio_write,
        pending_port_in,
        port_string_in,
        features,
        x87_precision,
        cs,
        ds,
        es,
        ss,
        fpu_empty,
        fpu_sw,
        fpu_fip,
        fpu_fdp,
        fpu_fop,
        fpu_fcs,
        fpu_fds,
        mmio_parts,
        mmio_parts_len,
        mmio_parts_rip,
        mmio_retry_active,
        mmio_retry_cs,
        source_id,
        random_state,
        rflags_system,
        fs,
        gs,
        mxcsr,
        x87_pointer_model
    );
    dst.cpuid_overrides.xcr0 = src.cpuid_overrides.xcr0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Prot, VmConfig};

    #[test]
    fn sparse_child_journal_and_code_snapshot_cover_only_child_ram() {
        for span in [0x10000, 0x1000000] {
            let mut vm = Vm::new(VmConfig::flat(span));
            vm.map(0, span as usize, Prot::RWX, RegionKind::Ram)
                .unwrap();
            vm.set_fetch_policy(Box::new(|_| {
                panic!("sparse attachment called parent policy")
            }));
            // Tags in holes, traps and both ends of a snapshot word must be masked.
            for page in [0, 1, 3, 5, 6, 15] {
                vm.mem.mark_code(page << CODE_PAGE_BITS, 1);
            }
            let view = vm
                .child_view_with(
                    ChildViewMetadata {
                        regions: vec![
                            ChildRegion::Ram {
                                range: 0x3001..0x5001,
                                backing_offset: 0x3001,
                                fetch: FetchVerdict::Allow,
                            },
                            ChildRegion::Trap {
                                range: 0x6000..span,
                            },
                            ChildRegion::Ram {
                                range: 0x1001..0x1fff,
                                backing_offset: 0x1001,
                                fetch: FetchVerdict::Allow,
                            },
                        ],
                        code_store_refusals: vec![4, 4],
                    },
                    CpuMode::Long64,
                )
                .unwrap();
            assert_eq!(view.writes.pages.len(), 2);
            assert_eq!(
                view.writes
                    .pages
                    .iter()
                    .map(|(_, words)| words.len() * 8)
                    .sum::<usize>(),
                16
            );
            assert_eq!(
                view.code.pages,
                vec![(0, (1 << 1) | (1 << 3) | (1 << 4) | (1 << 5))]
            );
            assert_eq!(view.regions.len(), 3);
            assert!(std::ptr::eq(view.parent, &vm.mem));
        }
    }

    #[test]
    fn sparse_child_backing_and_tags_use_nonzero_guest_base() {
        let base = 0x10010;
        let len = 0x4000;
        let backing = vec![0u128; len / 16].into_boxed_slice();
        let ptr = Box::into_raw(backing).cast::<u128>().cast::<u8>();
        let mut vm = Vm::new(VmConfig::flat(0));
        vm.consistency = crate::MemConsistency::FullTso;
        vm.mem = Memory::from_host_ram(
            crate::MemoryModel::Reserved {
                span: base + len as u64,
            },
            crate::HostRam {
                ptr,
                len,
                guest_base: base,
                dtor: Box::new(|ptr, len| {
                    // SAFETY: this is the boxed u128 slice transferred above.
                    unsafe {
                        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                            ptr.cast::<u128>(),
                            len / 16,
                        )))
                    };
                }),
                protect: None,
            },
        )
        .unwrap();
        vm.map(base, len, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(base, &[0x89, 0x18]).unwrap();
        vm.mem.mark_code(base + 0x2000, 1);
        let metadata = ChildViewMetadata {
            regions: vec![ChildRegion::Ram {
                range: base..base + len as u64,
                backing_offset: 0,
                fetch: FetchVerdict::Allow,
            }],
            code_store_refusals: vec![(base + 0x3000) >> CODE_PAGE_BITS],
        };
        let mut wrong = metadata.clone();
        let ChildRegion::Ram { backing_offset, .. } = &mut wrong.regions[0] else {
            unreachable!()
        };
        *backing_offset = base as usize;
        assert!(matches!(
            vm.child_view_with(wrong, CpuMode::Long64),
            Err(ChildViewError::BackingMismatch)
        ));
        let view = vm.child_view_with(metadata, CpuMode::Long64).unwrap();
        assert_eq!(view.mem_consistency(), crate::MemConsistency::FullTso);
        assert_eq!(view.guest_cpu_features(), vm.guest_cpu_features());
        assert_eq!(view.x87_pointer_model(), vm.x87_pointer_model());
        assert!(view.code_page((base + 0x2000) >> CODE_PAGE_BITS));
        assert!(view.code_page((base + 0x3000) >> CODE_PAGE_BITS));
        let mut cpu = CpuState {
            rip: base,
            ..CpuState::default()
        };
        cpu.gpr[0] = base + 0x1000;
        cpu.gpr[3] = 42;
        let mut arena = ChildArena::new(2);
        assert!(matches!(
            step(&view, &mut cpu, &mut arena),
            ChildStep::Continue
        ));
        assert_eq!(vm.mem.read(base + 0x1000, 4), Ok(42));
        assert_eq!(
            &*arena.written_pages(),
            &[(base + 0x1000) >> CODE_PAGE_BITS]
        );
        let mut pages = Vec::new();
        view.writes.drain(|page| pages.push(page));
        assert_eq!(pages, arena.written_pages().as_ref());
        vm.invalidate_written_pages(&arena);
    }

    #[test]
    fn child_ir_and_temporary_budgets_refuse() {
        let mut vm = Vm::new(VmConfig::flat(0x4000));
        vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, &[0x48, 0x83, 0xc0, 1]).unwrap();
        let view = vm.child_view(CpuMode::Long64);
        for ir_full in [true, false] {
            let mut arena = ChildArena::new(4);
            if ir_full {
                arena.ir.ops = Vec::new();
            } else {
                arena.temps = Vec::new();
            }
            let mut cpu = CpuState {
                rip: 0x1000,
                ..CpuState::default()
            };
            let before = format!("{cpu:?}");
            assert!(matches!(
                step(&view, &mut cpu, &mut arena),
                ChildStep::Refused(ChildRefusal::ArenaFull)
            ));
            assert_eq!(before, format!("{cpu:?}"));
        }
    }
}
