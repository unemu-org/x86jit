//! Borrowed Machine device access, spec.md §5.4 and decision-28.
use super::{ExecutionProfile, StepOptions, StepRun, StepRunError, Vcpu, Vm};
use crate::{Exit, PortDir, RegionKind};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, Weak};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeviceSpace {
    Port,
    Mmio,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeviceDirection {
    Read,
    Write,
}
/// Exact position within one run. Steps include REP iterations; retirement is
/// cumulative and advances only when an instruction finishes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct DevicePosition {
    pub retired_instructions: u64,
    pub steps: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeviceDeadline {
    /// Call-local instruction/iteration position.
    Steps(u64),
    /// Cumulative retired-instruction position.
    RetiredInstructions(u64),
}
impl DeviceDeadline {
    fn reached(self, p: DevicePosition) -> bool {
        match self {
            Self::Steps(n) => p.steps >= n,
            Self::RetiredInstructions(n) => p.retired_instructions >= n,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeviceEvent {
    pub vector: u8,
    pub kind: crate::DeliveryKind,
    pub saved_cs: u16,
    pub saved_ip: u32,
    pub error_code: Option<u32>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeviceRequest {
    pub space: DeviceSpace,
    pub address: u64,
    pub direction: DeviceDirection,
    pub size: u8,
    pub value: u64,
    pub total_size: u16,
    pub part: u8,
    pub linear_address: Option<u32>,
    pub instruction_ip: u64,
    pub code_segment: u16,
    /// An event delivery can own the transfer instead of an instruction.
    pub event: Option<DeviceEvent>,
    /// Immutable VM-issued identity, independent of mutable CpuState::source_id.
    pub vcpu_id: u64,
    pub position: DevicePosition,
    pub unit_start: DevicePosition,
    /// Modeled cycles are unavailable until cycle accounting is implemented.
    pub cycles: Option<u64>,
}
impl DeviceRequest {
    /// Reconstruct the ordinary Exit for this transfer, for explicit host
    /// recovery after DeviceFault. No handler is called and no state changes.
    pub fn handoff(&self) -> Exit {
        match (self.space, self.direction) {
            (DeviceSpace::Port, direction) => Exit::port_io(
                self.address as u16,
                self.size,
                if direction == DeviceDirection::Read {
                    PortDir::In
                } else {
                    PortDir::Out
                },
                self.value,
            ),
            (DeviceSpace::Mmio, DeviceDirection::Read) => Exit::MmioRead {
                addr: self.address,
                size: self.size,
                total_size: self.total_size,
                part: self.part,
            },
            (DeviceSpace::Mmio, DeviceDirection::Write) => Exit::MmioWrite {
                addr: self.address,
                size: self.size,
                value: self.value,
                total_size: self.total_size,
                part: self.part,
            },
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeviceError {
    pub code: u64,
}
impl DeviceError {
    pub const fn new(code: u64) -> Self {
        Self { code }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeviceResult {
    /// Low transfer bytes for reads; the value is ignored for writes.
    Handled(u64),
    Fallback,
    DeviceFault(DeviceError),
}
/// A host-facing device-service stop. No variant represents a guest exception.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeviceStop {
    /// Complete or abandon the transfer in `DeviceRun::request()` or
    /// `Vcpu::device_request()`; execution never redispatches it automatically.
    Fault { error: DeviceError },
    /// The callback unwound or acknowledged RAM effects could not be applied.
    /// All vCPUs in this VM are terminally poisoned.
    HandlerFailed,
    /// A preceding handler failure terminally poisoned this VM.
    Poisoned,
    /// INTR is asserted and this boundary accepts a maskable interrupt.
    IntrReady,
    /// An exact instruction/iteration deadline has been reached.
    Deadline { position: DevicePosition },
}
/// Bounded, synchronous, non-reentrant device service. Fallback and DeviceFault
/// must have no device-private effects. Context effects commit only on Handled.
/// No Send/Sync bound is required; cross-vCPU arbitration belongs to the embedder.
pub trait DeviceIo {
    fn access(&mut self, request: &DeviceRequest, context: &mut DeviceContext<'_>) -> DeviceResult;
}
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeviceRange {
    Ports(Range<u32>),
    Mmio(Range<u64>),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceHandle(u64);
static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeviceRegistrationError {
    UnsupportedProfile,
    InvalidRange,
    Overlap,
    NotTrap,
    Busy,
    UnknownHandle,
    Exhausted,
}
impl std::fmt::Display for DeviceRegistrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::UnsupportedProfile => "device ranges require a Machine profile",
            Self::InvalidRange => "device range is empty or out of bounds",
            Self::Overlap => "device range overlaps a registered range",
            Self::NotTrap => "MMIO device range is not entirely Trap memory",
            Self::Busy => "a vCPU is executing or owns an outstanding device handoff",
            Self::UnknownHandle => "device handle is not registered",
            Self::Exhausted => "device handles are exhausted",
        })
    }
}
impl std::error::Error for DeviceRegistrationError {}

const PENDING: u8 = 1;
const OBSERVED: u8 = 2;

#[derive(Default)]
pub(super) struct CpuRegistration {
    status: AtomicU8,
    active: AtomicBool,
    word: Weak<AtomicU64>,
}
impl CpuRegistration {
    #[inline(always)]
    pub fn observed(&self) -> bool {
        self.status.load(Ordering::Acquire) != 0
    }
    pub fn observe(&self) {
        self.status.fetch_or(OBSERVED, Ordering::Release);
    }
    pub fn pending(&self) -> bool {
        self.status.load(Ordering::Acquire) & PENDING != 0
    }
    pub fn set_pending(&self, pending: bool) {
        if self.pending() == pending {
            return;
        }
        if pending {
            self.status.fetch_or(PENDING, Ordering::Release);
        } else {
            self.status.fetch_and(!PENDING, Ordering::Release);
        }
    }
}
#[derive(Default)]
pub(super) struct Cpu {
    pub state: Arc<CpuRegistration>,
    pub id: u64,
    pub fault: Option<Exit>,
    pub request: Option<Box<DeviceRequest>>,
    /// An issued MmioRead/MmioWrite handoff awaiting its completion call.
    pub mmio: bool,
    /// This vCPU's last value of `state`'s pending bit, which only it writes.
    pub published: bool,
    pub intr: bool,
    pub invalidate: bool,
}
/// Marks a vCPU as executing for `Registry::quiescent` for one run. It owns a
/// reference to the registration because the run borrows the vCPU mutably.
pub(super) struct Active(Arc<CpuRegistration>);
impl Active {
    pub fn enter(cpu: &Arc<CpuRegistration>) -> Self {
        cpu.active.store(true, Ordering::Release);
        Self(Arc::clone(cpu))
    }
}
impl Drop for Active {
    fn drop(&mut self) {
        self.0.active.store(false, Ordering::Release);
    }
}
#[derive(Default)]
pub(super) struct Registry {
    cpus: Mutex<Vec<Weak<CpuRegistration>>>,
    poison: AtomicBool,
}
impl Registry {
    pub fn add_cpu(&self, word: &Arc<AtomicU64>, registered: bool) -> Arc<CpuRegistration> {
        let cpu = Arc::new(CpuRegistration {
            word: Arc::downgrade(word),
            status: AtomicU8::new(if !registered && !self.poisoned() {
                0
            } else {
                OBSERVED
            }),
            ..CpuRegistration::default()
        });
        let mut cpus = self.cpus.lock().unwrap();
        cpus.retain(|p| p.strong_count() != 0);
        cpus.push(Arc::downgrade(&cpu));
        cpu
    }
    pub fn quiescent(&self) -> bool {
        self.cpus
            .lock()
            .unwrap()
            .iter()
            .filter_map(Weak::upgrade)
            .all(|p| !p.active.load(Ordering::Acquire) && !p.pending())
    }
    fn poison_execution(&self) {
        self.poison.store(true, Ordering::Release);
        for cpu in self.cpus.lock().unwrap().iter().filter_map(Weak::upgrade) {
            cpu.observe();
            if let Some(word) = cpu.word.upgrade() {
                word.fetch_or(super::EXIT_REQUESTED, Ordering::Release);
            }
        }
    }
    #[inline(always)]
    pub fn poisoned(&self) -> bool {
        self.poison.load(Ordering::Acquire)
    }
}
/// The VM's registered ranges. Registration takes `&mut Vm` and every run
/// borrows `&Vm`, so no run can overlap a change: runs read this without a
/// lock or an atomic. Ports are a bitmap over the 16-bit namespace, MMIO a
/// sorted list of disjoint ranges.
#[derive(Default)]
pub(super) struct Table {
    pub(super) entries: Vec<(DeviceHandle, DeviceRange)>,
    ports: Vec<u64>,
    mmio: Vec<Range<u64>>,
}
impl Table {
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    #[inline(always)]
    pub fn has_mmio(&self) -> bool {
        !self.mmio.is_empty()
    }
    fn rebuild(&mut self) {
        self.ports.clear();
        self.mmio.clear();
        for (_, range) in &self.entries {
            match range {
                DeviceRange::Ports(r) => {
                    self.ports.resize(1024, 0);
                    for port in r.clone() {
                        self.ports[port as usize / 64] |= 1 << (port % 64);
                    }
                }
                DeviceRange::Mmio(r) => self.mmio.push(r.clone()),
            }
        }
        self.mmio.sort_by_key(|r| r.start);
    }
    /// Whether one registration serves this transfer: the addressed port for a
    /// 1/2/4-byte port access, or the whole 1/2/4/8-byte MMIO transfer.
    #[inline]
    pub fn serves(&self, space: DeviceSpace, address: u64, size: u8) -> bool {
        match space {
            DeviceSpace::Port => {
                matches!(size, 1 | 2 | 4)
                    && self
                        .ports
                        .get(address as usize / 64)
                        .is_some_and(|word| word & (1 << (address % 64)) != 0)
            }
            DeviceSpace::Mmio => {
                if !matches!(size, 1 | 2 | 4 | 8) {
                    return false;
                }
                let i = self.mmio.partition_point(|r| r.start <= address);
                i != 0
                    && address
                        .checked_add(u64::from(size))
                        .is_some_and(|end| end <= self.mmio[i - 1].end)
            }
        }
    }
    /// The space, address and size of an issued port or MMIO handoff.
    #[inline]
    pub fn serves_exit(&self, exit: &Exit) -> bool {
        match *exit {
            Exit::PortIo { port, size, .. } => {
                self.serves(DeviceSpace::Port, u64::from(port), size)
            }
            Exit::MmioRead { addr, size, .. } | Exit::MmioWrite { addr, size, .. } => {
                self.serves(DeviceSpace::Mmio, addr, size)
            }
            _ => false,
        }
    }
    fn overlaps_mmio(&self, address: u64, len: usize) -> bool {
        let end = address.saturating_add(len as u64);
        self.mmio.iter().any(|r| address < r.end && r.start < end)
    }
}
impl Vm {
    fn validate_device_range(
        &self,
        range: &DeviceRange,
        replacing: Option<DeviceHandle>,
    ) -> Result<(), DeviceRegistrationError> {
        if self.profile != ExecutionProfile::Machine {
            return Err(DeviceRegistrationError::UnsupportedProfile);
        }
        if !self.devices().quiescent() {
            return Err(DeviceRegistrationError::Busy);
        }
        match range {
            DeviceRange::Ports(r) if r.start < r.end && r.end <= 65536 => {}
            DeviceRange::Mmio(r) if r.start < r.end => {
                let mut p = r.start;
                while p < r.end {
                    if self.mem.region_kind(p, 1) != Ok(RegionKind::Trap) {
                        return Err(DeviceRegistrationError::NotTrap);
                    }
                    p = self
                        .mem
                        .region_end(p)
                        .map_err(|_| DeviceRegistrationError::NotTrap)?
                        .min(r.end);
                }
            }
            _ => return Err(DeviceRegistrationError::InvalidRange),
        }
        if self.device_table.entries.iter().any(|(id, old)| {
            Some(*id) != replacing
                && match (range, old) {
                    (DeviceRange::Ports(a), DeviceRange::Ports(b)) => {
                        a.start < b.end && b.start < a.end
                    }
                    (DeviceRange::Mmio(a), DeviceRange::Mmio(b)) => {
                        a.start < b.end && b.start < a.end
                    }
                    _ => false,
                }
        }) {
            return Err(DeviceRegistrationError::Overlap);
        }
        Ok(())
    }
    pub fn register_device_range(
        &mut self,
        range: DeviceRange,
    ) -> Result<DeviceHandle, DeviceRegistrationError> {
        self.validate_device_range(&range, None)?;
        let id = DeviceHandle(
            NEXT_HANDLE
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                .map_err(|_| DeviceRegistrationError::Exhausted)?,
        );
        self.device_table.entries.push((id, range));
        self.device_table.rebuild();
        for cpu in self
            .devices()
            .cpus
            .lock()
            .unwrap()
            .iter()
            .filter_map(Weak::upgrade)
        {
            cpu.observe();
        }
        Ok(id)
    }
    pub fn replace_device_range(
        &mut self,
        handle: DeviceHandle,
        range: DeviceRange,
    ) -> Result<(), DeviceRegistrationError> {
        if self.profile == ExecutionProfile::Machine
            && !self
                .device_table
                .entries
                .iter()
                .any(|(id, _)| *id == handle)
        {
            return Err(DeviceRegistrationError::UnknownHandle);
        }
        self.validate_device_range(&range, Some(handle))?;
        let entry = self
            .device_table
            .entries
            .iter_mut()
            .find(|(id, _)| *id == handle)
            .ok_or(DeviceRegistrationError::UnknownHandle)?;
        entry.1 = range;
        self.device_table.rebuild();
        Ok(())
    }
    pub fn unregister_device_range(
        &mut self,
        handle: DeviceHandle,
    ) -> Result<(), DeviceRegistrationError> {
        if !self.devices().quiescent() {
            return Err(DeviceRegistrationError::Busy);
        }
        let i = self
            .device_table
            .entries
            .iter()
            .position(|(id, _)| *id == handle)
            .ok_or(DeviceRegistrationError::UnknownHandle)?;
        self.device_table.entries.remove(i);
        self.device_table.rebuild();
        Ok(())
    }
    pub fn device_poisoned(&self) -> bool {
        self.devices().poisoned()
    }
    /// A mapping change over registered MMIO while a vCPU executes or owns an
    /// outstanding handoff.
    pub(super) fn device_mapping_busy(&self, address: u64, len: usize) -> bool {
        self.device_table.overlaps_mmio(address, len) && !self.devices().quiescent()
    }
}

/// Restricted physical RAM and board-signal access. Writes are staged and visible
/// to subsequent reads in this callback; Handled commits them with dirty/SMC
/// notification. No CPU, mappings, completion or execution APIs are exposed.
pub struct DeviceContext<'a> {
    vm: &'a Vm,
    fx: Effects,
}
/// The staged effects of one callback, committed only on Handled.
struct Effects {
    writes: Vec<(u64, Vec<u8>)>,
    intr: Option<bool>,
    nmi: bool,
    stop: bool,
    a20: Option<bool>,
    invalidate: bool,
    deadline: Option<DeviceDeadline>,
}
impl Effects {
    fn new(deadline: Option<DeviceDeadline>) -> Self {
        Self {
            writes: Vec::new(),
            intr: None,
            nmi: false,
            stop: false,
            a20: None,
            invalidate: false,
            deadline,
        }
    }
    /// Whether committing these effects changes nothing the rest of a block
    /// depends on: no RAM write (code or data a later instruction reads), no
    /// INTR level change, NMI, host stop, A20 or paging request, and the same
    /// deadline (the block's step cap was computed from it).
    fn none(&self, intr: bool, deadline: Option<DeviceDeadline>) -> bool {
        self.writes.is_empty()
            && self.intr.is_none_or(|level| level == intr)
            && !self.nmi
            && !self.stop
            && self.a20.is_none()
            && !self.invalidate
            && self.deadline == deadline
    }
}
impl DeviceContext<'_> {
    fn check_ram(&self, address: u64, len: usize) -> Result<(), DeviceError> {
        let end = address.checked_add(len as u64).ok_or(DeviceError::new(1))?;
        let mut p = address;
        while p < end {
            if self.vm.mem.region_kind(p, 1) != Ok(RegionKind::Ram) {
                return Err(DeviceError::new(1));
            }
            p = self
                .vm
                .mem
                .region_end(p)
                .map_err(|_| DeviceError::new(1))?
                .min(end);
        }
        Ok(())
    }
    pub fn read_ram(&self, address: u64, bytes: &mut [u8]) -> Result<(), DeviceError> {
        self.check_ram(address, bytes.len())?;
        let mut offset = 0;
        while offset < bytes.len() {
            let at = address + offset as u64;
            let len = (self.vm.mem.region_end(at).unwrap() - at).min((bytes.len() - offset) as u64)
                as usize;
            self.vm
                .read_bytes(at, &mut bytes[offset..offset + len])
                .map_err(|_| DeviceError::new(1))?;
            offset += len;
        }
        for (at, data) in &self.fx.writes {
            let lo = address.max(*at);
            let hi = (address + bytes.len() as u64).min(*at + data.len() as u64);
            if lo < hi {
                bytes[(lo - address) as usize..(hi - address) as usize]
                    .copy_from_slice(&data[(lo - at) as usize..(hi - at) as usize]);
            }
        }
        Ok(())
    }
    pub fn write_ram(&mut self, address: u64, bytes: &[u8]) -> Result<(), DeviceError> {
        self.check_ram(address, bytes.len())?;
        self.fx.writes.push((address, bytes.to_vec()));
        Ok(())
    }
    pub fn set_intr(&mut self, asserted: bool) {
        self.fx.intr = Some(asserted);
    }
    pub fn assert_nmi(&mut self) {
        self.fx.nmi = true;
    }
    pub fn request_host_stop(&mut self) {
        self.fx.stop = true;
    }
    pub fn set_a20_enabled(&mut self, enabled: bool) {
        self.fx.a20 = Some(enabled);
    }
    pub fn invalidate_paging(&mut self) {
        self.fx.invalidate = true;
    }
    /// Set a deadline or move the same coordinate earlier. Changing coordinates
    /// or moving later is refused, preserving the already-promised service time.
    pub fn set_deadline(&mut self, deadline: DeviceDeadline) -> Result<(), DeviceError> {
        if self.fx.deadline.is_some_and(|old| !matches!((old, deadline),
            (DeviceDeadline::Steps(a), DeviceDeadline::Steps(b)) |
            (DeviceDeadline::RetiredInstructions(a), DeviceDeadline::RetiredInstructions(b)) if b <= a)) {
            return Err(DeviceError::new(2));
        }
        self.fx.deadline = Some(deadline);
        Ok(())
    }
    /// Unknown until modeled cycle accounting is available. Adds no host time.
    pub fn charge_extra_cycles(&mut self, _cycles: u64) -> Option<u64> {
        None
    }
}

/// Whether an op can set IF: STI, or a flags load, far transfer, interrupt or
/// task return, whose IF comes from memory or a new task's image. Conservative:
/// a false positive only runs the block one unit at a time.
fn may_set_if(op: &crate::IrOp) -> bool {
    use crate::legacy::LegacyOp as L;
    matches!(
        op,
        crate::IrOp::SetIf { value: true }
            | crate::IrOp::Legacy(
                L::Flags { pop: true, .. }
                    | L::Transfer { cs: Some(_), .. }
                    | L::Return { far: true, .. }
                    | L::Interrupt { .. }
                    | L::Iret { .. }
                    | L::CheckTaskReturn
                    | L::LoadFlagsWord { .. }
                    | L::Interrupt20 { .. }
                    | L::Brkem20 { .. }
                    | L::Reti20 { .. }
                    | L::PopPsw20 { .. }
                    | L::Loadall
            )
    )
}
/// Whether an instruction before the block's last can set IF. The last one's
/// effect is first observed at the boundary after the block, as when stepped.
fn sets_if_before_last(ir: &crate::ir::IrBlock) -> bool {
    let last = ir
        .ops
        .iter()
        .rposition(|op| matches!(op, crate::IrOp::InsnStart { .. }))
        .unwrap_or(0);
    ir.ops[..last].iter().any(may_set_if)
}
#[derive(Default)]
pub(super) struct Run<'a> {
    pub handler: Option<&'a mut dyn DeviceIo>,
    pub deadline: Option<DeviceDeadline>,
    pub unit_start: DevicePosition,
    pub instruction_ip: u64,
    pub code_segment: u16,
    /// A callback the in-block path made that the block could not continue
    /// past: the walk then issued this transfer's ordinary handoff, which
    /// settles this result instead of calling the handler again.
    decided: Option<Box<Decided>>,
}
struct Decided {
    request: DeviceRequest,
    result: std::thread::Result<DeviceResult>,
    effects: Effects,
}
impl Run<'_> {
    /// Whether this run's handler is consulted for an issued handoff: a
    /// registered transfer, or any transfer once the VM is poisoned. Others
    /// take the ordinary Exit without building a request.
    #[inline(always)]
    pub fn serves(&self, vm: &Vm, exit: &Exit) -> bool {
        vm.devices().poisoned() || (self.handler.is_some() && vm.device_table.serves_exit(exit))
    }
    #[inline(always)]
    pub fn boundary(&self, cpu: &mut Vcpu, steps: u64) -> Option<Exit> {
        // Nothing to decide without a deadline, an asserted INTR or a queued
        // paging invalidation.
        if self.deadline.is_none() && !cpu.device_cpu().intr && !cpu.device_cpu().invalidate {
            return None;
        }
        self.boundary_slow(cpu, steps)
    }
    #[inline(never)]
    fn boundary_slow(&self, cpu: &mut Vcpu, steps: u64) -> Option<Exit> {
        if cpu.legacy_memory_in_flight() || cpu.cpu.pending_port_in.is_some() {
            return None;
        }
        if cpu.device_cpu().invalidate {
            cpu.invalidate_paging();
            cpu.device_cpu_mut().invalidate = false;
        }
        let position = DevicePosition {
            retired_instructions: cpu.retired,
            steps,
        };
        if self.deadline.is_some_and(|d| d.reached(position)) {
            return Some(Exit::Device(crate::DeviceStop::Deadline { position }));
        }
        // SDM Vol 3A §6.9: a single-step trap held for a completed IN/OUT, or
        // any event awaiting delivery, outranks INTR. This boundary accepts a
        // maskable interrupt only once those are delivered. An NMI edge outranks
        // it only while NMI is unblocked; a blocked one is discarded or
        // remembered at this boundary (spec.md §17.10).
        if cpu.device_cpu().intr
            && cpu
                .legacy_execution
                .as_ref()
                .is_none_or(|e| !e.event_waiting())
            && cpu.pending_irq.is_none()
            && {
                let word = cpu.exit_request.word.load(Ordering::Relaxed);
                word & super::EXIT_REQUESTED == 0
                    && (word & super::NMI_EDGE == 0
                        || cpu.nmi_state & super::legacy::NMI_BLOCKED != 0)
            }
            && !cpu.numeric_error_unreported()
            && cpu.irq_delivery_ready()
        {
            return Some(Exit::Device(crate::DeviceStop::IntrReady));
        }
        None
    }
    /// Units that must run one at a time: while INTR is asserted with IF=1 the
    /// next boundary may accept it (after a shadow, a held trap or a pending
    /// completion), and a queued paging invalidation applies at a boundary.
    #[inline(always)]
    pub fn single_step(&self, cpu: &Vcpu) -> bool {
        (cpu.device_cpu().intr && cpu.cpu.flags.if_) || cpu.device_cpu().invalidate
    }
    /// Cap one block's step budget so the walk stops at exactly the boundary a
    /// single-stepped run would report. A deadline caps at its distance: a step
    /// retires at most one instruction. While INTR is asserted with IF=0, only an
    /// instruction that can set IF ends the stretch; STI or POPF does so inside a
    /// block, so such a block runs one unit at a time.
    #[inline(always)]
    pub fn cap(&self, cpu: &Vcpu, ir: &crate::ir::IrBlock, steps: u64, remaining: u64) -> u64 {
        if self.deadline.is_none() && !cpu.device_cpu().intr {
            return remaining;
        }
        let mut remaining = match self.deadline {
            Some(DeviceDeadline::Steps(n)) => remaining.min(n.saturating_sub(steps).max(1)),
            Some(DeviceDeadline::RetiredInstructions(n)) => {
                remaining.min(n.saturating_sub(cpu.retired).max(1))
            }
            None => remaining,
        };
        if remaining > 1 && cpu.device_cpu().intr && !cpu.cpu.flags.if_ && sets_if_before_last(ir) {
            remaining = 1;
        }
        remaining
    }
    pub fn access(
        &mut self,
        cpu: &mut Vcpu,
        vm: &Vm,
        steps: u64,
        exit: &Exit,
    ) -> Option<Result<(), Exit>> {
        if vm.devices().poisoned() {
            return Some(Err(Exit::Device(crate::DeviceStop::Poisoned)));
        }
        self.handler.as_ref()?;
        let (space, address, direction, size, value, total_size, part) = match *exit {
            Exit::PortIo {
                port,
                size,
                dir,
                value,
            } => (
                DeviceSpace::Port,
                u64::from(port),
                if dir == PortDir::In {
                    DeviceDirection::Read
                } else {
                    DeviceDirection::Write
                },
                size,
                value,
                u16::from(size),
                0,
            ),
            Exit::MmioRead {
                addr,
                size,
                total_size,
                part,
            } => (
                DeviceSpace::Mmio,
                addr,
                DeviceDirection::Read,
                size,
                0,
                total_size,
                part,
            ),
            Exit::MmioWrite {
                addr,
                size,
                value,
                total_size,
                part,
            } => (
                DeviceSpace::Mmio,
                addr,
                DeviceDirection::Write,
                size,
                value & (u64::MAX >> (64 - u32::from(size) * 8)),
                total_size,
                part,
            ),
            _ => return None,
        };
        if let Some(decided) = self.decided.take() {
            // The in-block path called the handler for this transfer, and the
            // walk issues its handoff next. Any other handoff drops that result
            // and takes the ordinary path.
            let Decided {
                request,
                result,
                effects,
            } = *decided;
            if (
                request.space,
                request.address,
                request.direction,
                request.size,
                request.value,
            ) == (space, address, direction, size, value)
            {
                return self.settle(cpu, vm, exit, &request, result, effects);
            }
        }
        if !vm.device_table.serves(space, address, size) {
            return None;
        }
        let request = DeviceRequest {
            space,
            address,
            direction,
            size,
            value,
            total_size,
            part,
            linear_address: cpu.pending_mmio_linear_address(),
            instruction_ip: self.instruction_ip,
            code_segment: self.code_segment,
            event: cpu.legacy_execution.as_ref().and_then(|e| e.device_event()),
            vcpu_id: cpu.device_cpu().id,
            position: DevicePosition {
                retired_instructions: cpu.retired,
                steps,
            },
            unit_start: self.unit_start,
            cycles: None,
        };
        let (result, effects) = self.call(vm, &request);
        self.settle(cpu, vm, exit, &request, result, effects)
    }
    fn call(
        &mut self,
        vm: &Vm,
        request: &DeviceRequest,
    ) -> (std::thread::Result<DeviceResult>, Effects) {
        let mut context = DeviceContext {
            vm,
            fx: Effects::new(self.deadline),
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.handler
                .as_deref_mut()
                .unwrap()
                .access(request, &mut context)
        }));
        (result, context.fx)
    }
    fn settle(
        &mut self,
        cpu: &mut Vcpu,
        vm: &Vm,
        exit: &Exit,
        request: &DeviceRequest,
        result: std::thread::Result<DeviceResult>,
        effects: Effects,
    ) -> Option<Result<(), Exit>> {
        match result {
            Ok(DeviceResult::Fallback) => None,
            Ok(DeviceResult::DeviceFault(error)) => {
                cpu.device_cpu_mut().request = Some(Box::new(request.clone()));
                Some(Err(Exit::Device(crate::DeviceStop::Fault { error })))
            }
            Err(_) => {
                vm.devices().poison_execution();
                cpu.device_cpu_mut().request = Some(Box::new(request.clone()));
                Some(Err(Exit::Device(crate::DeviceStop::HandlerFailed)))
            }
            Ok(DeviceResult::Handled(value)) => {
                for (address, bytes) in &effects.writes {
                    let mut offset = 0;
                    while offset < bytes.len() {
                        let at = address + offset as u64;
                        let len = (vm.mem.region_end(at).unwrap() - at)
                            .min((bytes.len() - offset) as u64)
                            as usize;
                        if vm.write_bytes(at, &bytes[offset..offset + len]).is_err() {
                            vm.devices().poison_execution();
                            cpu.device_cpu_mut().request = Some(Box::new(request.clone()));
                            return Some(Err(Exit::Device(crate::DeviceStop::HandlerFailed)));
                        }
                        offset += len;
                    }
                }
                if let Some(asserted) = effects.intr {
                    cpu.device_cpu_mut().intr = asserted;
                }
                if effects.nmi {
                    let _ = cpu.raise_nmi();
                }
                if effects.stop {
                    cpu.exit_request.request_exit();
                }
                if let Some(enabled) = effects.a20 {
                    cpu.set_a20_enabled(enabled);
                }
                cpu.device_cpu_mut().invalidate |= effects.invalidate;
                self.deadline = effects.deadline;
                match exit {
                    Exit::PortIo {
                        dir: PortDir::In, ..
                    } => cpu.complete_port_in(value),
                    Exit::MmioRead { .. } => cpu.complete_mmio_read(value),
                    Exit::MmioWrite { .. } => cpu.complete_mmio_write(),
                    // An OUT needs no completion, but its single-step trap
                    // is now held.
                    _ => cpu.refresh_device_pending(),
                }
                Some(Ok(()))
            }
        }
    }
}

/// Decision-28 stage 3 inside a block: the shared real-mode walk meets a
/// registered MMIO transfer and calls the handler where it stands, instead of
/// abandoning the instruction for a handoff, a completion and a re-execution.
/// The dispatcher arms this for one walk that starts with no handoff state
/// open (no MMIO retry, pending answer or port input); only the walk's memory
/// provider reads it, at a Trap-region miss, so RAM accesses and unarmed
/// walks pay nothing.
///
/// The walk continues past the transfer only when that is the re-execution's
/// result exactly: a scalar transfer inside one registration that is its
/// instruction's only memory access, preceded by operations that fault before
/// it or not at all and followed by register operations that cannot fault (no
/// later stop of the instruction re-executes it), and a Handled callback with
/// no effects (`Effects::none`). Any other result is kept in `Run::decided` and
/// the walk issues the ordinary handoff, which settles it as if the handler
/// were called there: the handler is never called twice for one transfer, and
/// Fallback, DeviceFault, a panic and every effect keep the stage-2 behavior.
/// An IR block, an instruction IP in it and its `sole_access` result.
type Verdict = (Arc<crate::ir::IrBlock>, u64, Option<u64>);
pub(super) struct InBlock {
    run: std::cell::Cell<*mut Run<'static>>,
    ir: std::cell::Cell<*const Arc<crate::ir::IrBlock>>,
    /// The last `sole_access` verdict, for an IR this entry keeps alive.
    verdict: std::cell::RefCell<Option<Verdict>>,
    vm: *const Vm,
    vcpu_id: u64,
    code_segment: std::cell::Cell<u16>,
    intr: std::cell::Cell<bool>,
    retired: std::cell::Cell<u64>,
    steps: std::cell::Cell<u64>,
}
thread_local! {
    static IN_BLOCK: std::cell::Cell<*const InBlock> =
        const { std::cell::Cell::new(std::ptr::null()) };
}
/// Published hooks on all threads: a Trap miss with none published (every
/// run without a handler) returns before the thread-local lookup. Kept over a
/// lookup alone, which costs the Exit path about 22 host instructions per MMIO
/// transfer (`machine_exits`, real16 store and planar, two Trap misses each);
/// the counter changes once per handler run, not per access.
static PUBLISHED: AtomicU64 = AtomicU64::new(0);
#[cfg(test)]
thread_local! {
    /// Tests only: whether the in-block path is disabled, how many callbacks
    /// it made, and how many transfers it completed inside their block.
    pub(super) static IN_BLOCK_TEST: std::cell::Cell<(bool, u64, u64)> =
        const { std::cell::Cell::new((false, 0, 0)) };
}
/// Publishes an `InBlock` for one dispatch call and withdraws it on return.
pub(super) struct InBlockScope<'a> {
    hook: &'a InBlock,
    previous: *const InBlock,
}
impl InBlock {
    pub fn new(vm: &Vm, vcpu_id: u64) -> Self {
        Self {
            run: std::cell::Cell::new(std::ptr::null_mut()),
            ir: std::cell::Cell::new(std::ptr::null()),
            verdict: std::cell::RefCell::new(None),
            vm,
            vcpu_id,
            code_segment: std::cell::Cell::new(0),
            intr: std::cell::Cell::new(false),
            retired: std::cell::Cell::new(0),
            steps: std::cell::Cell::new(0),
        }
    }
    pub fn publish(&self) -> InBlockScope<'_> {
        PUBLISHED.fetch_add(1, Ordering::Relaxed);
        InBlockScope {
            hook: self,
            previous: IN_BLOCK.with(|slot| slot.replace(self)),
        }
    }
    /// Arm for one walk of `ir` that starts at this exact run position, when
    /// no transfer of the vCPU awaits completion or re-execution. The caller
    /// arms no walk that an event delivery owns.
    #[inline(always)]
    pub fn arm(&self, run: &mut Run<'_>, cpu: &Vcpu, ir: &Arc<crate::ir::IrBlock>, steps: u64) {
        let state = &cpu.cpu;
        if state.mmio_retry_active
            || state.pending_mmio.is_some()
            || state.pending_mmio_write
            || state.pending_port_in.is_some()
            || state.port_string_in.is_some()
        {
            return;
        }
        debug_assert!(run.decided.is_none());
        self.run.set((run as *mut Run<'_>).cast());
        self.ir.set(ir);
        self.code_segment.set(state.cs);
        self.intr.set(cpu.device_cpu().intr);
        self.retired.set(cpu.retired);
        self.steps.set(steps);
    }
    #[inline(always)]
    pub fn disarm(&self) {
        self.run.set(std::ptr::null_mut());
        self.ir.set(std::ptr::null());
    }
}
impl Drop for InBlockScope<'_> {
    fn drop(&mut self) {
        self.hook.disarm();
        IN_BLOCK.with(|slot| slot.set(self.previous));
        PUBLISHED.fetch_sub(1, Ordering::Relaxed);
    }
}
/// The completed instructions before the one at `ip` in `ir`, when its only
/// memory access is a Load or Store preceded by operations that fault before
/// it or not at all and followed by register operations that cannot fault.
/// Every earlier instruction of an armed walk completed, retiring once and
/// charging one step, unless a string operation counted its iterations; such
/// a block declines.
fn sole_access(ir: &crate::ir::IrBlock, ip: u64) -> Option<u64> {
    use crate::ir::IrOp;
    use crate::legacy::LegacyOp as L;
    let pure = |op: &IrOp| {
        matches!(
            op,
            IrOp::ReadReg { .. }
                | IrOp::WriteReg { .. }
                | IrOp::Add { .. }
                | IrOp::Sub { .. }
                | IrOp::Adc { .. }
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
                | IrOp::Sext { .. }
                | IrOp::Bswap { .. }
                | IrOp::Mul { .. }
                | IrOp::GetCond { .. }
                | IrOp::Fence { .. }
                | IrOp::Legacy(L::SharedWalk)
        )
    };
    let mut completed = 0u64;
    let mut ops = ir.ops.iter();
    loop {
        match ops.next()? {
            IrOp::InsnStart { guest_addr } if *guest_addr == ip => break,
            IrOp::InsnStart { .. } => completed += 1,
            IrOp::RepString { .. }
            | IrOp::RepStringReal { .. }
            | IrOp::Legacy(L::String { .. } | L::WrappedString { .. }) => return None,
            _ => {}
        }
    }
    let mut accessed = false;
    for op in ops {
        match op {
            IrOp::InsnStart { .. } => break,
            IrOp::Load { .. } | IrOp::Store { .. } if !accessed => accessed = true,
            // A segment limit check faults before the access, so before any call.
            IrOp::SegLimitCheck { .. } if !accessed => {}
            op if pure(op) => {}
            _ => return None,
        }
    }
    accessed.then_some(completed)
}
/// Disarms a hook while its handler runs. Decision-28 forbids execution from
/// inside a callback, but a handler holding the VM and another vCPU can still
/// run it: that run publishes no hook without a handler of its own, so its walk
/// would find this one, armed for the same cached block, and call the running
/// handler again through a second `&mut Run`. Disarmed, the nested run takes
/// the ordinary handoff. It is not refused: once nothing aliases it is sound,
/// and detecting it would add a thread-local check and an error to every run.
struct Disarmed<'a>(&'a InBlock, *mut Run<'static>);
impl<'a> Disarmed<'a> {
    fn new(hook: &'a InBlock) -> Self {
        Self(hook, hook.run.replace(std::ptr::null_mut()))
    }
}
impl Drop for Disarmed<'_> {
    fn drop(&mut self) {
        self.0.run.set(self.1);
    }
}
/// Called by the walk's memory provider at a Trap-region miss. Returns the
/// read value (zero for a write) when the walk continues; None issues the
/// ordinary handoff.
#[cold]
#[inline(never)]
pub(crate) fn in_block_access(
    ir: &crate::ir::IrBlock,
    ip: u64,
    write: bool,
    address: u64,
    size: u8,
    value: u64,
) -> Option<u64> {
    if PUBLISHED.load(Ordering::Relaxed) == 0 {
        return None;
    }
    #[cfg(test)]
    if IN_BLOCK_TEST.with(|t| t.get().0) {
        return None;
    }
    let hook = IN_BLOCK.with(|slot| slot.get());
    if hook.is_null() {
        return None;
    }
    // SAFETY: a published hook outlives its scope, which the dispatch call that
    // owns it holds. `run` is non-null only while the walk of `ir` it armed
    // runs, during which the dispatcher touches neither the Run nor the VM,
    // and never while its handler runs (`Disarmed`).
    let hook = unsafe { &*hook };
    let run = hook.run.get();
    if run.is_null() {
        return None;
    }
    // SAFETY: as `run`: the dispatcher's block handle outlives the walk.
    let block = unsafe { &*hook.ir.get() };
    if !std::ptr::eq(&**block, ir) {
        return None;
    }
    let run = unsafe { &mut *run };
    let vm = unsafe { &*hook.vm };
    if run.decided.is_some()
        || run.handler.is_none()
        || vm.devices().poisoned()
        || !vm.device_table.serves(DeviceSpace::Mmio, address, size)
    {
        return None;
    }
    let completed = {
        let mut verdict = hook.verdict.borrow_mut();
        match &*verdict {
            Some((cached, at, completed)) if Arc::ptr_eq(cached, block) && *at == ip => *completed,
            _ => {
                let completed = sole_access(ir, ip);
                *verdict = Some((block.clone(), ip, completed));
                completed
            }
        }
    }?;
    let mask = u64::MAX >> (64 - u32::from(size) * 8);
    let position = DevicePosition {
        retired_instructions: hook.retired.get() + completed,
        steps: hook.steps.get() + completed,
    };
    let request = DeviceRequest {
        space: DeviceSpace::Mmio,
        address,
        direction: if write {
            DeviceDirection::Write
        } else {
            DeviceDirection::Read
        },
        size,
        value: if write { value & mask } else { 0 },
        total_size: u16::from(size),
        part: 0,
        // An armed walk runs an unpaged real-mode block.
        linear_address: None,
        instruction_ip: ip,
        code_segment: hook.code_segment.get(),
        event: None,
        vcpu_id: hook.vcpu_id,
        position,
        // An MMIO transfer's unit is its own instruction.
        unit_start: position,
        cycles: None,
    };
    let (result, effects) = {
        let _disarmed = Disarmed::new(hook);
        run.call(vm, &request)
    };
    #[cfg(test)]
    IN_BLOCK_TEST.with(|t| t.set((t.get().0, t.get().1 + 1, t.get().2)));
    if let Ok(DeviceResult::Handled(answer)) = result {
        if effects.none(hook.intr.get(), run.deadline) {
            #[cfg(test)]
            IN_BLOCK_TEST.with(|t| t.set((t.get().0, t.get().1, t.get().2 + 1)));
            return Some(if write { 0 } else { answer & mask });
        }
    }
    run.decided = Some(Box::new(Decided {
        request,
        result,
        effects,
    }));
    None
}
/// Counts and stop from the borrowed-device entry. Error evidence is owned by
/// this result; ordinary StepRun/Exit values retain their existing drop paths.
#[derive(Debug)]
#[non_exhaustive]
pub struct DeviceRun {
    run: StepRun,
    request: Option<Box<DeviceRequest>>,
}
impl DeviceRun {
    pub fn steps(&self) -> u64 {
        self.run.steps()
    }
    pub fn stop(&self) -> &super::StepStop {
        self.run.stop()
    }
    pub fn request(&self) -> Option<&DeviceRequest> {
        self.request.as_deref()
    }
    pub fn into_stop(self) -> super::StepStop {
        self.run.into_stop()
    }
    /// Retain counts/stop and discard the owned request snapshot.
    pub fn into_run(self) -> StepRun {
        self.run
    }
}
impl Vcpu {
    /// Publish whether this vCPU owns an outstanding handoff, which blocks fork,
    /// registration and relevant mapping changes: an unanswered IN, an INS whose
    /// store is still pending, an uncompleted MMIO handoff, a latched device
    /// fault, or private state no CpuState carries: an accepted event's
    /// continuation, a task switch's context or a held single-step trap. An OUT
    /// needs no completion; a completed MMIO's replay issues no transfer that
    /// the registry could route differently.
    pub(super) fn refresh_device_pending(&mut self) {
        let devices = self.device_cpu();
        let pending = devices.fault.is_some()
            || devices.mmio
            || self.cpu.pending_port_in.is_some()
            || self.cpu.port_string_in.is_some()
            || self
                .legacy_execution
                .as_ref()
                .is_some_and(|e| e.device_outstanding());
        if devices.published != pending {
            devices.state.set_pending(pending);
            self.device_cpu_mut().published = pending;
        }
    }
    /// End-of-run bookkeeping: a run that executed past an uncompleted MMIO
    /// handoff issued it again or abandoned it; one that executed nothing
    /// leaves it.
    #[cold]
    #[inline(never)]
    pub(super) fn finish_device_pending(&mut self, result: &super::StepRun) {
        let issued = matches!(
            result.stop,
            super::StepStop::Exit(Exit::MmioRead { .. } | Exit::MmioWrite { .. })
        );
        let devices = self.device_cpu_mut();
        devices.mmio = issued || (devices.mmio && result.steps == 0);
        self.refresh_device_pending();
    }
    /// A pending unmasked x87 error the External route reports at this
    /// boundary's service step, before an IRQ is accepted (§17.9).
    fn numeric_error_unreported(&self) -> bool {
        self.cpu.numeric_error_route == crate::NumericErrorRoute::External
            && self.cpu.fpu_model == crate::FpuModel::Present
            && self.cpu.numeric_error_pending()
            && !self.cpu.external_numeric_error.reported
    }
    /// Set the controller's INTR level between calls. Vector arbitration and
    /// acknowledgement remain external; inject the chosen vector with inject_irq.
    pub fn set_intr(&mut self, asserted: bool) {
        if self.profile == ExecutionProfile::Machine {
            self.device_cpu().state.observe();
            self.device_cpu_mut().intr = asserted;
        }
    }
    pub fn intr_asserted(&self) -> bool {
        self.profile == ExecutionProfile::Machine && self.device_cpu().intr
    }
    /// Complete a latched DeviceFault after the embedder services its `request.handoff()`.
    /// Read values use the low transfer bytes; writes ignore `value`.
    /// Returns false when no device fault is pending.
    pub fn complete_device_fault(&mut self, value: u64) -> bool {
        let Some(Exit::Device(crate::DeviceStop::Fault { .. })) =
            self.device_cpu_mut().fault.take()
        else {
            return false;
        };
        let request = self.device_cpu_mut().request.take().expect("fault request");
        match request.handoff() {
            Exit::MmioRead { .. } => self.complete_mmio_read(value),
            Exit::MmioWrite { .. } => self.complete_mmio_write(),
            Exit::PortIo {
                dir: PortDir::In, ..
            } => self.complete_port_in(value),
            Exit::PortIo {
                dir: PortDir::Out, ..
            } => {}
            _ => unreachable!("DeviceFault owns an issued transfer"),
        }
        self.refresh_device_pending();
        true
    }
    /// Request evidence for the current device fault or terminal handler failure.
    /// Explicit completion or execution-state abandonment clears it.
    pub fn device_request(&self) -> Option<&DeviceRequest> {
        self.device_cpu().request.as_deref()
    }
    fn device_run(&self, result: Result<StepRun, StepRunError>) -> Result<DeviceRun, StepRunError> {
        result.map(|run| {
            let request = if matches!(
                run.stop(),
                super::StepStop::Exit(
                    Exit::Device(crate::DeviceStop::Fault { .. })
                        | Exit::Device(crate::DeviceStop::HandlerFailed)
                )
            ) {
                self.device_request().cloned().map(Box::new)
            } else {
                None
            };
            DeviceRun { run, request }
        })
    }
    /// Run Machine instructions with a handler borrowed only for this call.
    /// None retains the ordinary Exit/completion behavior, even for registered
    /// ranges. Application profiles are outside this API. Zero budget services
    /// nothing. Device deadlines are local to this call; INTR remains a level.
    pub fn run_steps_with_devices(
        &mut self,
        vm: &Vm,
        max_steps: u64,
        options: StepOptions,
        handler: Option<&mut dyn DeviceIo>,
    ) -> Result<DeviceRun, StepRunError> {
        if self.profile != ExecutionProfile::Machine {
            return Err(StepRunError::UnsupportedMode);
        }
        if handler.is_none() {
            let result = self.run_steps(vm, max_steps, options);
            return self.device_run(result);
        }
        vm.sync_backing_policy();
        self.sync_dispatch_word(vm);
        if !self.device_cpu().state.observed() {
            self.device_cpu().state.observe();
        }
        let mut devices = Run {
            handler,
            unit_start: DevicePosition {
                retired_instructions: self.retired.saturating_sub(u64::from(
                    self.cpu
                        .port_string_in
                        .as_ref()
                        .is_some_and(|p| p.next_ip != p.rip),
                )),
                steps: 0,
            },
            instruction_ip: self
                .cpu
                .port_string_in
                .as_ref()
                .map_or(self.cpu.rip, |p| p.rip),
            code_segment: self.cpu.cs,
            ..Run::default()
        };
        let result = if self
            .legacy_model
            .is_some_and(|m| m.generation == crate::CpuGeneration::I8086)
        {
            if max_steps != 0 && self.rep_restart20_pending() {
                self.forget_edited_rep_restart20(vm);
            }
            self.run_engine_devices::<{ super::DIRECT20_DISPATCH }, true, true, true>(
                vm,
                Some(max_steps),
                options,
                &mut devices,
            )
        } else {
            self.run_engine_devices::<{ super::MACHINE_DISPATCH }, true, true, true>(
                vm,
                Some(max_steps),
                options,
                &mut devices,
            )
        };
        self.device_run(result)
    }
}

#[cfg(test)]
mod tests;
