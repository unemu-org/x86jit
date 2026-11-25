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
pub(super) struct Active(Arc<CpuRegistration>);
impl Active {
    pub fn enter(cpu: &Arc<CpuRegistration>) -> Self {
        cpu.active.store(true, Ordering::Release);
        Self(cpu.clone())
    }
}
impl Drop for Active {
    fn drop(&mut self) {
        self.0.active.store(false, Ordering::Release);
    }
}
#[derive(Default)]
pub(super) struct Registry {
    entries: Mutex<Vec<(DeviceHandle, DeviceRange)>>,
    /// Whether `entries` is non-empty, so an access with no registered range
    /// takes no lock. Entries change only while every vCPU is quiescent.
    any: AtomicBool,
    cpus: Mutex<Vec<Weak<CpuRegistration>>>,
    poison: AtomicBool,
}
impl Registry {
    pub fn add_cpu(&self, word: &Arc<AtomicU64>) -> Arc<CpuRegistration> {
        let cpu = Arc::new(CpuRegistration {
            word: Arc::downgrade(word),
            status: AtomicU8::new(
                if self.entries.lock().unwrap().is_empty() && !self.poisoned() {
                    0
                } else {
                    2
                },
            ),
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
    pub fn mapping_busy(&self, address: u64, len: usize) -> bool {
        self.entries.lock().unwrap().iter().any(|(_, r)| matches!(r, DeviceRange::Mmio(r) if address < r.end && r.start < address.saturating_add(len as u64))) && !self.quiescent()
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
    pub fn poisoned(&self) -> bool {
        self.poison.load(Ordering::Acquire)
    }
    fn matches(&self, r: &DeviceRequest) -> bool {
        if !self.any.load(Ordering::Acquire) {
            return false;
        }
        self.entries
            .lock()
            .unwrap()
            .iter()
            .any(|(_, range)| match range {
                DeviceRange::Ports(range) => {
                    r.space == DeviceSpace::Port
                        && range.contains(&(r.address as u32))
                        && matches!(r.size, 1 | 2 | 4)
                }
                DeviceRange::Mmio(range) => {
                    r.space == DeviceSpace::Mmio
                        && matches!(r.size, 1 | 2 | 4 | 8)
                        && r.address >= range.start
                        && r.address
                            .checked_add(u64::from(r.size))
                            .is_some_and(|end| end <= range.end)
                }
            })
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
        if self
            .devices()
            .entries
            .lock()
            .unwrap()
            .iter()
            .any(|(id, old)| {
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
            })
        {
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
        let mut entries = self.devices().entries.lock().unwrap();
        entries.push((id, range));
        self.devices().any.store(true, Ordering::Release);
        drop(entries);
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
                .devices()
                .entries
                .lock()
                .unwrap()
                .iter()
                .any(|(id, _)| *id == handle)
        {
            return Err(DeviceRegistrationError::UnknownHandle);
        }
        self.validate_device_range(&range, Some(handle))?;
        let mut entries = self.devices().entries.lock().unwrap();
        let entry = entries
            .iter_mut()
            .find(|(id, _)| *id == handle)
            .ok_or(DeviceRegistrationError::UnknownHandle)?;
        entry.1 = range;
        Ok(())
    }
    pub fn unregister_device_range(
        &mut self,
        handle: DeviceHandle,
    ) -> Result<(), DeviceRegistrationError> {
        if !self.devices().quiescent() {
            return Err(DeviceRegistrationError::Busy);
        }
        let mut entries = self.devices().entries.lock().unwrap();
        let i = entries
            .iter()
            .position(|(id, _)| *id == handle)
            .ok_or(DeviceRegistrationError::UnknownHandle)?;
        entries.remove(i);
        self.devices()
            .any
            .store(!entries.is_empty(), Ordering::Release);
        Ok(())
    }
    pub fn device_poisoned(&self) -> bool {
        self.devices().poisoned()
    }
}

/// Restricted physical RAM and board-signal access. Writes are staged and visible
/// to subsequent reads in this callback; Handled commits them with dirty/SMC
/// notification. No CPU, mappings, completion or execution APIs are exposed.
pub struct DeviceContext<'a> {
    vm: &'a Vm,
    writes: Vec<(u64, Vec<u8>)>,
    intr: Option<bool>,
    nmi: bool,
    stop: bool,
    a20: Option<bool>,
    invalidate: bool,
    deadline: Option<DeviceDeadline>,
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
        for (at, data) in &self.writes {
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
        self.writes.push((address, bytes.to_vec()));
        Ok(())
    }
    pub fn set_intr(&mut self, asserted: bool) {
        self.intr = Some(asserted);
    }
    pub fn assert_nmi(&mut self) {
        self.nmi = true;
    }
    pub fn request_host_stop(&mut self) {
        self.stop = true;
    }
    pub fn set_a20_enabled(&mut self, enabled: bool) {
        self.a20 = Some(enabled);
    }
    pub fn invalidate_paging(&mut self) {
        self.invalidate = true;
    }
    /// Set a deadline or move the same coordinate earlier. Changing coordinates
    /// or moving later is refused, preserving the already-promised service time.
    pub fn set_deadline(&mut self, deadline: DeviceDeadline) -> Result<(), DeviceError> {
        if self.deadline.is_some_and(|old| !matches!((old, deadline),
            (DeviceDeadline::Steps(a), DeviceDeadline::Steps(b)) |
            (DeviceDeadline::RetiredInstructions(a), DeviceDeadline::RetiredInstructions(b)) if b <= a)) {
            return Err(DeviceError::new(2));
        }
        self.deadline = Some(deadline);
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
}
impl Run<'_> {
    pub fn boundary(&self, cpu: &mut Vcpu, steps: u64) -> Option<Exit> {
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
        if !vm.devices().matches(&request) {
            return None;
        }
        let mut context = DeviceContext {
            vm,
            writes: Vec::new(),
            intr: None,
            nmi: false,
            stop: false,
            a20: None,
            invalidate: false,
            deadline: self.deadline,
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.handler
                .as_deref_mut()
                .unwrap()
                .access(&request, &mut context)
        }));
        match result {
            Ok(DeviceResult::Fallback) => None,
            Ok(DeviceResult::DeviceFault(error)) => {
                cpu.device_cpu_mut().request = Some(Box::new(request));
                Some(Err(Exit::Device(crate::DeviceStop::Fault { error })))
            }
            Err(_) => {
                vm.devices().poison_execution();
                cpu.device_cpu_mut().request = Some(Box::new(request));
                Some(Err(Exit::Device(crate::DeviceStop::HandlerFailed)))
            }
            Ok(DeviceResult::Handled(value)) => {
                for (address, bytes) in &context.writes {
                    let mut offset = 0;
                    while offset < bytes.len() {
                        let at = address + offset as u64;
                        let len = (vm.mem.region_end(at).unwrap() - at)
                            .min((bytes.len() - offset) as u64)
                            as usize;
                        if vm.write_bytes(at, &bytes[offset..offset + len]).is_err() {
                            vm.devices().poison_execution();
                            cpu.device_cpu_mut().request = Some(Box::new(request));
                            return Some(Err(Exit::Device(crate::DeviceStop::HandlerFailed)));
                        }
                        offset += len;
                    }
                }
                if let Some(asserted) = context.intr {
                    cpu.device_cpu_mut().intr = asserted;
                }
                if context.nmi {
                    let _ = cpu.raise_nmi();
                }
                if context.stop {
                    cpu.exit_request.request_exit();
                }
                if let Some(enabled) = context.a20 {
                    cpu.set_a20_enabled(enabled);
                }
                cpu.device_cpu_mut().invalidate |= context.invalidate;
                self.deadline = context.deadline;
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
        self.device_cpu().state.observe();
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
