//! Owned exception evidence (§5.2; decision-14). Hardware error-code formats:
//! Intel SDM Vol 3A §5.7 (Figure 5-12), §7.13 (Figure 7-7), §7.15.
//!
//! A flat guest map is not a guest page-table walk. Its labelled projection never
//! supplies an architectural code. Unknown bits must not be replaced with zero.

use crate::ir::{RepKind, StrOp};
use crate::{AccessKind, CpuMode, CpuState, Exit, GuestAccessKind, IrBlock, IrOp, Memory};

/// A known value, unavailable evidence, or a field without a meaning for this event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Knowledge<T> {
    Known(T),
    Unknown,
    NotApplicable,
}

/// Absence of a pushed code differs from a known zero code (SDM Vol 3A §7.13).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorCode {
    NoCode,
    Unknown,
    #[non_exhaustive]
    Bits {
        value: u32,
        known_mask: u32,
    },
}
impl ErrorCode {
    /// Construct partial evidence; bits outside the mask must be zero.
    pub const fn bits(value: u32, known_mask: u32) -> Self {
        assert!(value & !known_mask == 0);
        Self::Bits { value, known_mask }
    }
    /// Invalid public enum literals are treated as unavailable evidence.
    pub const fn bit(self, bit: u8) -> Option<bool> {
        match self {
            Self::Bits { value, known_mask }
                if bit < 32 && value & !known_mask == 0 && known_mask & (1u32 << bit) != 0 =>
            {
                Some(value & (1u32 << bit) != 0)
            }
            _ => None,
        }
    }
}

/// Guest intent or underlying transfer direction, including fetch and RMW.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultAccess {
    Read,
    Write,
    ReadWrite,
    Execute,
}
impl From<AccessKind> for FaultAccess {
    fn from(v: AccessKind) -> Self {
        match v {
            AccessKind::Read => Self::Read,
            AccessKind::Write => Self::Write,
            AccessKind::Execute => Self::Execute,
        }
    }
}
impl From<GuestAccessKind> for FaultAccess {
    fn from(v: GuestAccessKind) -> Self {
        match v {
            GuestAccessKind::Read => Self::Read,
            GuestAccessKind::Write => Self::Write,
            GuestAccessKind::ReadWrite => Self::ReadWrite,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FaultCause {
    Divide,
    InvalidOpcode,
    InstructionLength,
    NonCanonical,
    RequiredAlignment,
    InvalidMxcsr,
    IretNt,
    IretSelector,
    PendingX87,
    Breakpoint,
    DebugTrap,
    Unknown,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FaultOrigin {
    GuestException(FaultCause),
    GuestMap,
    TrapFetch,
    HostAccess,
    SoftwareInterrupt,
    Unknown,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultClass {
    Fault,
    Trap,
    Unknown,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressPrecision {
    ExactByte,
    TransferAddress,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressSpace {
    Linear,
    RealModePhysical,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaultAddress {
    value: u64,
    precision: AddressPrecision,
    space: AddressSpace,
}
impl FaultAddress {
    pub const fn new(value: u64, precision: AddressPrecision, space: AddressSpace) -> Self {
        Self {
            value,
            precision,
            space,
        }
    }
    pub const fn value(self) -> u64 {
        self.value
    }
    pub const fn precision(self) -> AddressPrecision {
        self.precision
    }
    pub const fn space(self) -> AddressSpace {
        self.space
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferUnit {
    Bytes,
    FetchByte,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegisterProgress {
    InstructionEntry,
    CompletedInstruction,
    RepRestart,
    Unknown,
}
/// Effects of the reporting instruction only. PossiblyPartial is not a byte bitmap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryEffects {
    None,
    PossiblyPartial,
    Unknown,
}

/// Exact current restart state, not a count of iterations completed in this attempt.
/// SDM Vol 2B, REP/REPE/REPNE, Description and Operation.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct RepRestart {
    pub(crate) source: u64,
    pub(crate) destination: u64,
    pub(crate) remaining: u64,
    pub(crate) address_bits: u8,
    pub(crate) element_size: u8,
    pub(crate) direction_flag: bool,
    pub(crate) flags: u64,
    pub(crate) operation: StrOp,
}
impl RepRestart {
    pub fn source(&self) -> u64 {
        self.source
    }
    pub fn destination(&self) -> u64 {
        self.destination
    }
    pub fn remaining(&self) -> u64 {
        self.remaining
    }
    pub fn address_bits(&self) -> u8 {
        self.address_bits
    }
    pub fn element_size(&self) -> u8 {
        self.element_size
    }
    pub fn direction_flag(&self) -> bool {
        self.direction_flag
    }
    pub fn flags(&self) -> u64 {
        self.flags
    }
    pub fn operation(&self) -> StrOp {
        self.operation
    }
    pub fn completed_iterations(&self) -> Knowledge<u64> {
        Knowledge::Unknown
    }
}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct FaultProgress {
    pub(crate) registers: RegisterProgress,
    pub(crate) ram: MemoryEffects,
    pub(crate) device: MemoryEffects,
    pub(crate) rep: Option<RepRestart>,
    // Retry acknowledgements are address/value evidence; the existing ledger does
    // not distinguish device reads from writes. Never assert an effect direction.
    pub(crate) acknowledged: Box<[(u64, u64)]>,
}
impl FaultProgress {
    pub fn registers(&self) -> RegisterProgress {
        self.registers
    }
    pub fn ram(&self) -> MemoryEffects {
        self.ram
    }
    pub fn device(&self) -> MemoryEffects {
        self.device
    }
    pub fn rep(&self) -> Option<&RepRestart> {
        self.rep.as_ref()
    }
    /// Current retry ledger only; completed REP iterations may have older effects.
    pub fn acknowledged_transfers(&self) -> &[(u64, u64)] {
        &self.acknowledged
    }
    pub fn device_history_complete(&self) -> Knowledge<bool> {
        Knowledge::Unknown
    }
    pub fn had_partial_progress(&self) -> Knowledge<bool> {
        if self.rep.is_some() {
            return Knowledge::Unknown;
        }
        if !self.acknowledged.is_empty() {
            return Knowledge::Known(true);
        }
        if matches!(
            self.registers,
            RegisterProgress::InstructionEntry | RegisterProgress::CompletedInstruction
        ) && self.ram == MemoryEffects::None
            && self.device == MemoryEffects::None
        {
            Knowledge::Known(false)
        } else {
            Knowledge::Unknown
        }
    }
}

/// Partial map/access facts, explicitly separate from an architectural #PF.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct PageFaultProjection {
    pub(crate) code: ErrorCode,
    pub(crate) is_fetch: bool,
}

/// The ABI that supplied a raw signal record, independent of the guest ISA.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum HostFaultPlatform {
    LinuxX86_64,
    FreeBsdX86_64,
    MacOsX86_64,
    LinuxAarch64,
    FreeBsdAarch64,
    MacOsAarch64,
}

/// Opaque host evidence copied by the embedder's signal adapter. No field is
/// interpreted as a guest exception code (Intel SDM Vol 3A §5.7, §7.13).
/// `None` means the host context did not expose that field; zero is a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawHostFault {
    pub platform: HostFaultPlatform,
    pub trap_number: Option<u64>,
    pub error_code: Option<u64>,
    /// CR2 on x86-64, FAR on AArch64, when exposed by the host context.
    pub fault_address: Option<u64>,
    pub esr: Option<u64>,
    pub signal: Option<i32>,
    pub signal_code: Option<i32>,
    /// Kept separately: some host signals report a PC instead of a data address.
    pub signal_address: Option<u64>,
}

impl RawHostFault {
    pub const fn new(platform: HostFaultPlatform) -> Self {
        Self {
            platform,
            trap_number: None,
            error_code: None,
            fault_address: None,
            esr: None,
            signal: None,
            signal_code: None,
            signal_address: None,
        }
    }
}
impl PageFaultProjection {
    pub const fn error_code(self) -> ErrorCode {
        self.code
    }
    pub const fn is_fetch(self) -> bool {
        self.is_fetch
    }
}

/// The execution tier that called a shared Rust faulting helper.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum HelperCaller {
    Interpreter,
    CompiledBlock { host_pc: usize },
    Region { entry_rip: u64, host_pc: usize },
}

/// Shared helper families represented by the lowered instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FaultHelper {
    X87,
    FxState,
    String,
    Divide,
    CompareExchangePair,
}

/// Owned evidence of where a fault was raised. This describes engine execution,
/// independently of the architectural cause and any underlying host signal.
/// `host_pc` is a code-map lookup key: an exact transfer PC after host recovery,
/// or the compiled function entry for a checked exit. A transfer `site` is present
/// only when an exact code-map site is available; checked exits do not invent one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FaultPath {
    Unknown,
    Interpreter,
    CompiledBlock {
        host_pc: usize,
        site: Option<crate::GuestTrapSite>,
    },
    Region {
        entry_rip: u64,
        host_pc: usize,
        site: Option<crate::GuestTrapSite>,
    },
    Helper {
        family: FaultHelper,
        caller: HelperCaller,
    },
    LifterFetch {
        first_unavailable: u64,
    },
}

/// Immutable evidence owned by an [`Exit`]. Each unavailable field stays explicit.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct FaultMetadata {
    pub(crate) path: FaultPath,
    pub(crate) origin: FaultOrigin,
    pub(crate) class: FaultClass,
    pub(crate) vector: Knowledge<u8>,
    pub(crate) error_code: ErrorCode,
    pub(crate) saved_rip: u64,
    pub(crate) instruction_rip: Knowledge<u64>,
    pub(crate) instruction_len: Knowledge<u8>,
    pub(crate) address: Knowledge<FaultAddress>,
    pub(crate) cs_ip: Knowledge<(u16, u16)>,
    pub(crate) operand_base: Knowledge<u64>,
    pub(crate) operand_size: Knowledge<u16>,
    pub(crate) access: Knowledge<FaultAccess>,
    pub(crate) transfer_access: Knowledge<FaultAccess>,
    pub(crate) transfer_size: Knowledge<u16>,
    pub(crate) transfer_unit: TransferUnit,
    pub(crate) projection: Option<PageFaultProjection>,
    pub(crate) raw_host_fault: Option<RawHostFault>,
    pub(crate) progress: FaultProgress,
}
impl FaultMetadata {
    /// Unqualified evidence; useful to custom backends without a complete model.
    pub fn unknown(saved_rip: u64) -> Self {
        Self {
            path: FaultPath::Unknown,
            origin: FaultOrigin::Unknown,
            class: FaultClass::Unknown,
            vector: Knowledge::Unknown,
            error_code: ErrorCode::Unknown,
            saved_rip,
            instruction_rip: Knowledge::Unknown,
            instruction_len: Knowledge::Unknown,
            address: Knowledge::Unknown,
            cs_ip: Knowledge::NotApplicable,
            operand_base: Knowledge::Unknown,
            operand_size: Knowledge::Unknown,
            access: Knowledge::Unknown,
            transfer_access: Knowledge::Unknown,
            transfer_size: Knowledge::Unknown,
            transfer_unit: TransferUnit::Bytes,
            projection: None,
            raw_host_fault: None,
            progress: FaultProgress {
                registers: RegisterProgress::Unknown,
                ram: MemoryEffects::Unknown,
                device: MemoryEffects::Unknown,
                rep: None,
                acknowledged: Box::new([]),
            },
        }
    }
    /// Cause-specific architectural evidence. Unsupported causes remain Unknown.
    /// SDM Vol 3A §7.13 and §7.15, Events 0/1/3/6/12/13/16.
    pub fn exception(saved_rip: u64, instruction_rip: u64, vector: u8, cause: FaultCause) -> Self {
        let mut m = Self::unknown(saved_rip);
        m.vector = Knowledge::Known(vector);
        m.instruction_rip = Knowledge::Known(instruction_rip);
        let expected = match cause {
            FaultCause::Divide => Some(0),
            FaultCause::InvalidOpcode => Some(6),
            FaultCause::Breakpoint => Some(3),
            FaultCause::DebugTrap => Some(1),
            FaultCause::PendingX87 => Some(16),
            FaultCause::NonCanonical if vector == 12 => Some(12),
            FaultCause::NonCanonical
            | FaultCause::InstructionLength
            | FaultCause::RequiredAlignment
            | FaultCause::InvalidMxcsr
            | FaultCause::IretNt
            | FaultCause::IretSelector => Some(13),
            _ => None,
        };
        if expected == Some(vector) {
            m.origin = FaultOrigin::GuestException(cause);
            m.class = if matches!(cause, FaultCause::Breakpoint | FaultCause::DebugTrap) {
                FaultClass::Trap
            } else {
                FaultClass::Fault
            };
            m.error_code = match cause {
                FaultCause::Divide
                | FaultCause::InvalidOpcode
                | FaultCause::Breakpoint
                | FaultCause::DebugTrap
                | FaultCause::PendingX87 => ErrorCode::NoCode,
                FaultCause::IretSelector => ErrorCode::Unknown,
                _ => ErrorCode::bits(0, u32::MAX),
            };
        }
        m
    }
    /// Assert architectural evidence supplied by a backend with its own guest
    /// exception model. The caller must justify the vector, class and code from
    /// the failing guest operation (SDM Vol 3A §5.7, §7.13, §7.15). A host signal
    /// or flat-map projection alone does not justify using this constructor.
    pub fn architectural_exception(
        saved_rip: u64,
        instruction_rip: u64,
        vector: u8,
        class: FaultClass,
        error_code: ErrorCode,
    ) -> Self {
        let mut m = Self::unknown(saved_rip);
        m.origin = FaultOrigin::GuestException(FaultCause::Unknown);
        m.class = class;
        m.vector = Knowledge::Known(vector);
        m.instruction_rip = Knowledge::Known(instruction_rip);
        m.error_code = error_code;
        m
    }

    /// Execution evidence; legacy/custom exits without a witness remain Unknown.
    pub fn path(&self) -> FaultPath {
        self.path
    }
    pub fn origin(&self) -> FaultOrigin {
        self.origin
    }
    pub fn class(&self) -> FaultClass {
        self.class
    }
    pub fn vector(&self) -> Knowledge<u8> {
        self.vector
    }
    pub fn error_code(&self) -> ErrorCode {
        self.error_code
    }
    /// Complete architectural code only; a flat-map projection never qualifies.
    pub fn architectural_error_code(&self) -> Option<u32> {
        match (self.origin, self.error_code) {
            (
                FaultOrigin::GuestException(_),
                ErrorCode::Bits {
                    value,
                    known_mask: u32::MAX,
                },
            ) => Some(value),
            _ => None,
        }
    }
    pub fn saved_rip(&self) -> u64 {
        self.saved_rip
    }
    pub fn instruction_rip(&self) -> Knowledge<u64> {
        self.instruction_rip
    }
    pub fn instruction_len(&self) -> Knowledge<u8> {
        self.instruction_len
    }
    pub fn linear_address(&self) -> Knowledge<FaultAddress> {
        self.address
    }
    pub fn real_mode_cs_ip(&self) -> Knowledge<(u16, u16)> {
        self.cs_ip
    }
    pub fn operand_base(&self) -> Knowledge<u64> {
        self.operand_base
    }
    pub fn operand_size(&self) -> Knowledge<u16> {
        self.operand_size
    }
    pub fn access(&self) -> Knowledge<FaultAccess> {
        self.access
    }
    pub fn transfer_access(&self) -> Knowledge<FaultAccess> {
        self.transfer_access
    }
    pub fn transfer_size(&self) -> Knowledge<u16> {
        self.transfer_size
    }
    pub fn transfer_unit(&self) -> TransferUnit {
        self.transfer_unit
    }
    pub fn page_fault_projection(&self) -> Option<PageFaultProjection> {
        self.projection
    }
    /// Raw host facts, separate from the guest linear/operand addresses and codes.
    pub fn raw_host_fault(&self) -> Option<&RawHostFault> {
        self.raw_host_fault.as_ref()
    }
    /// Attach evidence supplied by a host signal adapter without qualifying any
    /// guest architectural facts. Use only after returning from signal context.
    pub fn with_raw_host_fault(mut self, raw: RawHostFault) -> Self {
        self.raw_host_fault = Some(raw);
        self
    }
    pub fn progress(&self) -> &FaultProgress {
        &self.progress
    }
}

#[cold]
#[inline(never)]
pub(crate) fn memory_exit(
    rip: u64,
    addr: u64,
    size: u16,
    access: AccessKind,
    host: bool,
    exact: bool,
) -> Exit {
    let mut m = FaultMetadata::unknown(rip);
    m.origin = if host {
        FaultOrigin::HostAccess
    } else {
        FaultOrigin::GuestMap
    };
    m.instruction_rip = Knowledge::Known(rip);
    m.address = Knowledge::Known(FaultAddress::new(
        addr,
        if exact {
            AddressPrecision::ExactByte
        } else {
            AddressPrecision::TransferAddress
        },
        AddressSpace::Linear,
    ));
    m.transfer_size = Knowledge::Known(size);
    m.transfer_access = Knowledge::Known(access.into());
    Exit::unmapped_memory(addr, access).with_fault_metadata(m)
}

#[cold]
#[inline(never)]
pub(crate) fn exception_exit(rip: u64, vector: u8, cause: FaultCause) -> Exit {
    Exit::exception(rip, vector)
        .with_fault_metadata(FaultMetadata::exception(rip, rip, vector, cause))
}

// Retain the existing lowered address calculation, rather than decoding mutable
// guest code or adding address stores to compiled accesses. This covers a single
// explicit operand; implicit/multiple operands retain their existing evidence.
#[derive(Clone, Debug)]
struct OperandAddress {
    prefix: Box<[IrOp]>,
    address: crate::Val,
}
impl OperandAddress {
    fn from_ops(ops: &[IrOp]) -> Option<Self> {
        let mut candidate = None;
        for (index, op) in ops
            .iter()
            .enumerate()
            .filter(|(_, o)| o.memory_access().bytes() != 0)
        {
            let address = match op {
                IrOp::Load { addr, .. }
                | IrOp::Store { addr, .. }
                | IrOp::X87 { addr, .. }
                | IrOp::FxState { addr, .. }
                | IrOp::AtomicRmw { addr, .. }
                | IrOp::AtomicCas { addr, .. }
                | IrOp::AtomicCasPair { addr, .. }
                | IrOp::VLoad { addr, .. }
                | IrOp::VStore { addr, .. }
                | IrOp::VPackedBinM { addr, .. }
                | IrOp::VLogicM { addr, .. }
                | IrOp::VBlendWM { addr, .. }
                | IrOp::VFmaM { addr, .. }
                | IrOp::VAesM { addr, .. }
                | IrOp::VAesImcM { addr, .. }
                | IrOp::VAesKeygenM { addr, .. }
                | IrOp::VShaM { addr, .. }
                | IrOp::VGfniM { addr, .. }
                | IrOp::VPclmulM { addr, .. }
                | IrOp::VPsignM { addr, .. }
                | IrOp::VPackWideM { addr, .. }
                | IrOp::VPMaddM { addr, .. }
                | IrOp::VShuffle16M { addr, .. }
                | IrOp::VShufpsM { addr, .. }
                | IrOp::VPshufbM { addr, .. }
                | IrOp::VAlignrM { addr, .. }
                | IrOp::VUnpackLowM { addr, .. }
                | IrOp::VLoadWide { addr, .. }
                | IrOp::VStoreWide { addr, .. }
                | IrOp::VMaskLoadMem { addr, .. }
                | IrOp::VMaskStoreMem { addr, .. }
                | IrOp::VVecMaskLoadMem { addr, .. }
                | IrOp::VVecMaskStoreMem { addr, .. }
                | IrOp::VLogic256M { addr, .. }
                | IrOp::VLogicWideM { addr, .. }
                | IrOp::VPopcntM { addr, .. }
                | IrOp::VExtractLaneWideM { addr, .. }
                | IrOp::VPcmpStrM { addr, .. }
                | IrOp::VPcmpStrMaskM { addr, .. }
                | IrOp::VInsertPsM { addr, .. }
                | IrOp::VInsertPsM3 { addr, .. }
                | IrOp::VDppsM { addr, .. }
                | IrOp::VPMovExtendM { addr, .. }
                | IrOp::VPBlendVM { addr, .. }
                | IrOp::VPBlendVXM { addr, .. }
                | IrOp::VBlendIM { addr, .. }
                | IrOp::VDppdM { addr, .. }
                | IrOp::VDp3M { addr, .. }
                | IrOp::VPRoundM { addr, .. }
                | IrOp::VShiftRegM { addr, .. }
                | IrOp::VGf2p8M { addr, .. }
                | IrOp::VPTernlogM { addr, .. }
                | IrOp::VPackedBin256M { addr, .. }
                | IrOp::VPackedWideM { addr, .. }
                | IrOp::VBroadcastM { addr, .. }
                | IrOp::VBroadcastLaneM { addr, .. }
                | IrOp::VPCmpToMaskM { addr, .. }
                | IrOp::VPTestToMaskM { addr, .. }
                | IrOp::VPmovNarrowMem { addr, .. }
                | IrOp::VPermT2M { addr, .. }
                | IrOp::VPerm1M { addr, .. }
                | IrOp::VInsert128M { addr, .. }
                | IrOp::VPshufb256M { addr, .. }
                | IrOp::VPtestM { addr, .. }
                | IrOp::VFloatBinM { addr, .. }
                | IrOp::VHFloatM { addr, .. }
                | IrOp::VHIntM { addr, .. }
                | IrOp::VFloatCmpMaskM { addr, .. }
                | IrOp::VFloatCmpMask256M { addr, .. }
                | IrOp::VMpsadbwM { addr, .. }
                | IrOp::VPermilVarM { addr, .. }
                | IrOp::VPerm2i128M { addr, .. }
                | IrOp::VShiftVarM { addr, .. }
                | IrOp::VCvtPs2PhM { addr, .. }
                | IrOp::VFloatBin256M { addr, .. }
                | IrOp::VFloatUnary256M { addr, .. }
                | IrOp::VPackedCvt256M { addr, .. }
                | IrOp::VShufps256M { addr, .. }
                | IrOp::VUnpack256M { addr, .. }
                | IrOp::VLoadHalf { addr, .. }
                | IrOp::VStoreHalf { addr, .. } => *addr,
                _ => return None,
            };
            if let Some((_, prior)) = candidate {
                if !matches!((prior, address), (crate::Val::Temp(a), crate::Val::Temp(b)) if a == b)
                    && !matches!((prior, address), (crate::Val::Imm(a), crate::Val::Imm(b)) if a == b)
                {
                    return None;
                }
            } else {
                candidate = Some((index, address));
            }
        }
        let (index, address) = candidate?;
        Some(Self {
            prefix: ops[..index].into(),
            address,
        })
    }

    fn evaluate(&self, cpu: &CpuState) -> Option<u64> {
        let mut values = std::collections::HashMap::new();
        let read = |v: crate::Val, values: &std::collections::HashMap<crate::Temp, u64>| match v {
            crate::Val::Imm(v) => Some(v),
            crate::Val::Temp(t) => values.get(&t).copied(),
        };
        for op in self.prefix.iter() {
            let (dst, value) = match *op {
                IrOp::InsnStart { .. } | IrOp::CheckAlign { .. } => continue,
                IrOp::ReadReg { dst, reg } => (dst, crate::interp::read_reg(cpu, reg)),
                IrOp::Add {
                    dst, a, b, size: 8, ..
                } => (dst, read(a, &values)?.wrapping_add(read(b, &values)?)),
                IrOp::Sub {
                    dst, a, b, size: 8, ..
                } => (dst, read(a, &values)?.wrapping_sub(read(b, &values)?)),
                IrOp::And {
                    dst, a, b, size: 8, ..
                } => (dst, read(a, &values)? & read(b, &values)?),
                IrOp::Shl {
                    dst, a, b, size: 8, ..
                } => (
                    dst,
                    read(a, &values)?.wrapping_shl((read(b, &values)? & 63) as u32),
                ),
                _ => return None,
            };
            values.insert(dst, value);
        }
        read(self.address, &values)
    }
}

/// Immutable cold instruction data. It owns no guest bytes and survives SMC/unmap.
/// This additive backend interface leaves IrBlock and GuestTrapSite literals intact.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct InstructionFaultInfo {
    pub(crate) rip: u64,
    pub(crate) len: Knowledge<u8>,
    pub(crate) operand_size: Knowledge<u16>,
    pub(crate) rmw: bool,
    pub(crate) writes: bool,
    pub(crate) single_scalar_store: bool,
    pub(crate) preflighted_store: bool,
    pub(crate) single_scalar_transfer: bool,
    pub(crate) memory_ops: usize,
    // Bounds checks cover the operand; lane extraction transfers one 128-bit
    // piece at a time in both tiers, including its interpreter preflight.
    pub(crate) split_transfer_size: Option<u16>,
    pub(crate) rep: Option<(StrOp, u8, u8)>,
    pub(crate) trap: Option<(u8, u8)>,
    pub(crate) length_limit: bool,
    pub(crate) helper: Option<FaultHelper>,
    alignment_check: bool,
    operand_address: Option<OperandAddress>,
}
impl InstructionFaultInfo {
    /// Build per-instruction records, validating lengths within each basic block.
    pub fn for_block(block: &IrBlock) -> Vec<Self> {
        Self::for_block_with_split_stores(block, false)
    }

    /// Build records for the exact split-store policy used by code generation.
    pub fn for_block_with_split_stores(block: &IrBlock, precise_split_stores: bool) -> Vec<Self> {
        let mut infos = Vec::new();
        let mut start = 0;
        while start < block.ops.len() {
            let rip = match block.ops[start] {
                IrOp::InsnStart { guest_addr } => guest_addr,
                _ if start == 0 => block.guest_start,
                _ => break,
            };
            let end = block.ops[start + 1..]
                .iter()
                .position(|o| matches!(o, IrOp::InsnStart { .. }))
                .map_or(block.ops.len(), |n| start + 1 + n);
            let next = match block.ops.get(end) {
                Some(IrOp::InsnStart { guest_addr }) => Some(*guest_addr),
                _ => block.guest_start.checked_add(u64::from(block.guest_len)),
            };
            let info = Self::from_ops_with_split_stores(
                rip,
                next,
                &block.ops[start..end],
                precise_split_stores,
            );
            infos.push(info);
            start = end;
        }
        infos
    }
    pub(crate) fn from_ops(rip: u64, next: Option<u64>, ops: &[IrOp]) -> Self {
        Self::from_ops_with_split_stores(rip, next, ops, false)
    }
    fn from_ops_with_split_stores(
        rip: u64,
        next: Option<u64>,
        ops: &[IrOp],
        precise_split_stores: bool,
    ) -> Self {
        let mut m = Self {
            rip,
            len: next
                .and_then(|n| n.checked_sub(rip))
                .filter(|&n| (1..=15).contains(&n))
                .map_or(Knowledge::Unknown, |n| Knowledge::Known(n as u8)),
            operand_size: Knowledge::Unknown,
            rmw: false,
            writes: false,
            single_scalar_store: false,
            preflighted_store: false,
            single_scalar_transfer: false,
            memory_ops: 0,
            split_transfer_size: None,
            rep: None,
            trap: None,
            length_limit: false,
            alignment_check: ops.iter().any(|o| matches!(o, IrOp::CheckAlign { .. })),
            helper: ops.iter().find_map(|op| match op {
                IrOp::X87 { .. } => Some(FaultHelper::X87),
                IrOp::FxState { .. } => Some(FaultHelper::FxState),
                IrOp::RepString { .. } | IrOp::RepStringReal { .. } => Some(FaultHelper::String),
                IrOp::Div { .. } => Some(FaultHelper::Divide),
                IrOp::AtomicCasPair { .. } => Some(FaultHelper::CompareExchangePair),
                _ => None,
            }),
            operand_address: OperandAddress::from_ops(ops),
        };
        for op in ops {
            let bytes = op.memory_access().bytes();
            if bytes != 0 {
                m.memory_ops += 1;
                m.operand_size = match m.operand_size {
                    Knowledge::Unknown => Knowledge::Known(bytes),
                    Knowledge::Known(old) if old == bytes => Knowledge::Known(old),
                    _ => Knowledge::NotApplicable,
                };
            }
            if matches!(op, IrOp::VExtractLaneWideM { .. }) {
                m.split_transfer_size = Some(16);
            }
            match op {
                IrOp::Store { addr, .. } => {
                    m.writes = true;
                    m.rmw |= ops.iter().any(|o| matches!(o, IrOp::Load { addr: a, .. } if matches!((a, addr), (crate::Val::Temp(a), crate::Val::Temp(b)) if a == b) || matches!((a, addr), (crate::Val::Imm(a), crate::Val::Imm(b)) if a == b)));
                }
                IrOp::AtomicRmw { .. } | IrOp::AtomicCas { .. } | IrOp::AtomicCasPair { .. } => {
                    m.rmw = true;
                    m.writes = true;
                }
                IrOp::VStore { .. }
                | IrOp::VStoreWide { .. }
                | IrOp::VStoreHalf { .. }
                | IrOp::VMaskStoreMem { .. }
                | IrOp::VVecMaskStoreMem { .. }
                | IrOp::VExtractLaneWideM { .. }
                | IrOp::VPmovNarrowMem { .. }
                | IrOp::VCvtPs2PhM { .. }
                | IrOp::Call { .. }
                | IrOp::PushFlags { .. } => m.writes = true,
                IrOp::X87 { kind, .. } => m.writes |= crate::x87::mem_write_bytes(*kind).is_some(),
                IrOp::FxState { restore, .. } => m.writes |= !restore,
                IrOp::RepString {
                    op,
                    elem,
                    addr_bits,
                    rep,
                    ..
                } => {
                    m.writes |= matches!(op, StrOp::Movs | StrOp::Stos);
                    if !matches!(rep, RepKind::None) {
                        m.rep = Some((*op, *elem, *addr_bits));
                    }
                }
                IrOp::RepStringReal { op, elem, rep, .. } => {
                    m.writes |= matches!(op, StrOp::Movs | StrOp::Stos);
                    m.operand_size = Knowledge::Known(u16::from(*elem));
                    if !matches!(rep, RepKind::None) {
                        m.rep = Some((*op, *elem, 16));
                    }
                }
                IrOp::Trap { vector, advance } => {
                    m.trap = Some((*vector, *advance));
                    // The strict lifter's length-limit producer consumes 15 bytes
                    // and emits only this Trap. Generic vector-only exits do not
                    // pass through this static instruction descriptor.
                    m.length_limit = *vector == 13
                        && *advance == 0
                        && m.len == Knowledge::Known(15)
                        && ops
                            .iter()
                            .all(|o| matches!(o, IrOp::InsnStart { .. } | IrOp::Trap { .. }));
                    // Invalid decodings and length-exhaustion share Trap IR. Their
                    // decoded length cannot be proved from a consumed byte count.
                    if *advance == 0 {
                        m.len = Knowledge::Unknown;
                    }
                }
                _ => {}
            }
        }
        m.single_scalar_store = m.memory_ops == 1
            && matches!(m.operand_size, Knowledge::Known(1..=8))
            && ops
                .iter()
                .filter(|o| {
                    matches!(
                        o,
                        IrOp::Store { .. } | IrOp::Call { .. } | IrOp::PushFlags { .. }
                    )
                })
                .count()
                == 1;
        // One destination, with a whole-operand bounds/region preflight and
        // crossing compiled write probes. Multi-destination instructions and
        // REP retain their separately qualified progress contracts.
        m.preflighted_store = precise_split_stores
            && m.memory_ops <= 2
            && ops
                .iter()
                .filter(|op| {
                    matches!(
                        op,
                        IrOp::Store { .. }
                            | IrOp::VStore { .. }
                            | IrOp::VStoreWide { .. }
                            | IrOp::VStoreHalf { .. }
                            | IrOp::VExtractLaneWideM { .. }
                            | IrOp::VCvtPs2PhM { .. }
                            | IrOp::AtomicRmw { .. }
                            | IrOp::AtomicCas { .. }
                            | IrOp::Call { .. }
                            | IrOp::PushFlags { .. }
                    )
                })
                .count()
                == 1;
        m.single_scalar_transfer = m.memory_ops == 1
            && matches!(m.operand_size, Knowledge::Known(1..=8))
            && ops.iter().any(|o| {
                matches!(
                    o,
                    IrOp::Load { .. }
                        | IrOp::Store { .. }
                        | IrOp::AtomicRmw { .. }
                        | IrOp::AtomicCas { .. }
                        | IrOp::Call { .. }
                        | IrOp::Ret { .. }
                        | IrOp::PushFlags { .. }
                )
            });
        if m.operand_size == Knowledge::NotApplicable {
            m.operand_size = Knowledge::Unknown;
        }
        m
    }
}

#[cold]
#[inline(never)]
pub(crate) fn finish(
    mut exit: Exit,
    info: Option<&InstructionFaultInfo>,
    cpu: &CpuState,
    mem: &Memory,
    mode: CpuMode,
    compiled: bool,
) -> Exit {
    let legacy_vector = match &exit {
        Exit::Exception { vector, .. } => Some(*vector),
        Exit::UnmappedMemory { .. } => None,
        _ => return exit,
    };
    let mut m = match &mut exit {
        Exit::Exception { metadata, .. } | Exit::UnmappedMemory { metadata, .. } => metadata
            .take()
            .map(|v| *v)
            .unwrap_or_else(|| FaultMetadata::unknown(cpu.rip)),
        _ => unreachable!(),
    };
    if !compiled {
        let helper = info.and_then(|i| {
            i.helper.filter(|family| {
                // DIV's memory source is loaded before the arithmetic helper; an
                // explicit alignment check also precedes its instruction's helper.
                (*family != FaultHelper::Divide
                    || m.origin == FaultOrigin::GuestException(FaultCause::Divide))
                    && !(i.alignment_check
                        && m.origin == FaultOrigin::GuestException(FaultCause::RequiredAlignment))
            })
        });
        m.path = match helper {
            Some(family) => FaultPath::Helper {
                family,
                caller: HelperCaller::Interpreter,
            },
            None => FaultPath::Interpreter,
        };
    }
    m.saved_rip = cpu.rip;
    if let Some(info) = info {
        m.instruction_rip = Knowledge::Known(info.rip);
        m.instruction_len = info.len;
        if compiled && m.origin == FaultOrigin::HostAccess {
            if let Some(address) = info.operand_address.as_ref().and_then(|a| a.evaluate(cpu)) {
                m.operand_base = Knowledge::Known(address);
            }
        }
        if info.operand_size != Knowledge::Unknown {
            m.operand_size = info.operand_size;
        }
        if let Some(vector) = legacy_vector {
            if m.origin == FaultOrigin::Unknown {
                let cause = match (info.trap, vector) {
                    (Some((6, 0)), 6) => FaultCause::InvalidOpcode,
                    (Some((3, n)), 3) if n != 0 => FaultCause::Breakpoint,
                    (Some((1, n)), 1) if n != 0 => FaultCause::DebugTrap,
                    _ => FaultCause::Unknown,
                };
                let qualified = FaultMetadata::exception(cpu.rip, info.rip, vector, cause);
                m.origin = qualified.origin;
                m.class = qualified.class;
                m.vector = qualified.vector;
                m.error_code = qualified.error_code;
            }
        }
        if let Knowledge::Known(direction) = m.transfer_access {
            m.access = Knowledge::Known(if info.rmw {
                FaultAccess::ReadWrite
            } else {
                direction
            });
        }
        // Restart-state evidence only. No instruction-entry snapshots or counters.
        if let Some((op, elem, bits)) = info.rep {
            let mask = match bits {
                16 => 0xffff,
                32 => 0xffff_ffff,
                _ => u64::MAX,
            };
            m.progress.rep = Some(RepRestart {
                source: cpu.gpr[6] & mask,
                destination: cpu.gpr[7] & mask,
                remaining: cpu.gpr[1] & mask,
                address_bits: bits,
                element_size: elem,
                direction_flag: cpu.flags.df,
                flags: cpu.rflags(),
                operation: op,
            });
            m.progress.registers = RegisterProgress::RepRestart;
            m.progress.ram = if info.writes {
                MemoryEffects::PossiblyPartial
            } else {
                MemoryEffects::None
            };
        } else {
            m.progress.registers = if m.class == FaultClass::Trap {
                RegisterProgress::CompletedInstruction
            } else if legacy_vector.is_none() || m.class == FaultClass::Fault {
                RegisterProgress::InstructionEntry
            } else {
                RegisterProgress::Unknown
            };
            m.progress.ram = if (info.rmw
                && info.memory_ops <= 2
                && m.transfer_access == Knowledge::Known(FaultAccess::Read))
                || ((info.single_scalar_store || info.preflighted_store)
                    && m.origin == FaultOrigin::GuestMap)
                // A compiled one-byte Write in a wider destination is a
                // preflight site. Faults in actual wider stores remain uncertain
                // if protections were revoked after successful preflight.
                || (compiled && info.preflighted_store
                    && m.origin == FaultOrigin::HostAccess
                    && m.transfer_access == Knowledge::Known(FaultAccess::Write)
                    && m.transfer_size == Knowledge::Known(1)
                    && matches!(info.operand_size, Knowledge::Known(2..=64)))
            {
                MemoryEffects::None
            } else if info.writes
                && (legacy_vector.is_none()
                    || (compiled
                        && m.origin == FaultOrigin::GuestException(FaultCause::NonCanonical)))
            {
                MemoryEffects::PossiblyPartial
            } else {
                MemoryEffects::None
            };
        }
    }
    if cpu.owns_mmio_retry(cpu.rip) {
        m.progress.acknowledged = cpu.mmio_parts[..usize::from(cpu.mmio_parts_len)].into();
    }
    // No ledger means no proof about already issued device reads. A regular
    // single-transfer read with no retry cannot have completed a prior transfer.
    m.progress.device = if m.progress.acknowledged.is_empty()
        && info.is_some_and(|i| {
            i.rep.is_none()
                && ((legacy_vector.is_none() && i.single_scalar_transfer)
                    || (i.trap.is_some() && i.memory_ops == 0))
        }) {
        MemoryEffects::None
    } else {
        MemoryEffects::Unknown
    };
    if let Exit::UnmappedMemory { addr, access, .. } = &exit {
        if m.address == Knowledge::Unknown {
            m.address = Knowledge::Known(FaultAddress::new(
                *addr,
                AddressPrecision::TransferAddress,
                AddressSpace::Linear,
            ));
        }
        if m.transfer_access == Knowledge::Unknown {
            m.transfer_access = Knowledge::Known((*access).into());
        }
        if m.origin == FaultOrigin::GuestMap
            && crate::memory::is_canonical(*addr)
            && mem.absent_byte(*addr)
        {
            // The reported byte is verified absent, rather than merely the
            // start of a transfer that failed region containment.
            if let Knowledge::Known(address) = &mut m.address {
                address.precision = AddressPrecision::ExactByte;
            }
            // Labelled map projection only (SDM Vol 3A §5.7). U/S, RSVD,
            // PK, SS, HLAT, SGX and all reserved bits remain unknown. RMW W/R
            // remains unknown until instruction-level primary-source witnesses.
            let fetch = *access == AccessKind::Execute;
            let wr = match m.access {
                Knowledge::Known(FaultAccess::Read) => Some(false),
                Knowledge::Known(FaultAccess::Write) => Some(true),
                _ => None,
            };
            let mask = 1 | if !fetch { 0x10 } else { 0 } | if wr.is_some() { 2 } else { 0 };
            m.projection = Some(PageFaultProjection {
                code: ErrorCode::bits(if wr == Some(true) { 2 } else { 0 }, mask),
                is_fetch: fetch,
            });
        }
    }
    if mode == CpuMode::Real16 {
        m.cs_ip = Knowledge::Known((cpu.cs, cpu.rip as u16));
        if let Knowledge::Known(ref mut address) = m.address {
            address.space = AddressSpace::RealModePhysical;
        }
    }
    exit.with_fault_metadata(m)
}

/// Private cold-ABI encoding, shared with the built-in backend.
#[doc(hidden)]
pub fn cause_to_wire(cause: FaultCause) -> u64 {
    match cause {
        FaultCause::Divide => 1,
        FaultCause::InvalidOpcode => 2,
        FaultCause::InstructionLength => 3,
        FaultCause::NonCanonical => 4,
        FaultCause::RequiredAlignment => 5,
        FaultCause::InvalidMxcsr => 6,
        FaultCause::IretNt => 7,
        FaultCause::IretSelector => 8,
        FaultCause::PendingX87 => 9,
        FaultCause::Breakpoint => 10,
        FaultCause::DebugTrap => 11,
        FaultCause::Unknown => 0,
    }
}
pub(crate) fn cause_from_wire(value: u64) -> FaultCause {
    match value {
        1 => FaultCause::Divide,
        2 => FaultCause::InvalidOpcode,
        3 => FaultCause::InstructionLength,
        4 => FaultCause::NonCanonical,
        5 => FaultCause::RequiredAlignment,
        6 => FaultCause::InvalidMxcsr,
        7 => FaultCause::IretNt,
        8 => FaultCause::IretSelector,
        9 => FaultCause::PendingX87,
        10 => FaultCause::Breakpoint,
        11 => FaultCause::DebugTrap,
        _ => FaultCause::Unknown,
    }
}

#[cold]
#[inline(never)]
pub(crate) fn fetch_exit(
    mem: &Memory,
    at: crate::lift::FetchAddr,
    mode: CpuMode,
    addr: u64,
    trap: Option<crate::lift::FetchFailure>,
) -> Exit {
    use crate::lift::FetchFailure::{Memory, Policy};
    let mut m = FaultMetadata::unknown(at.ip);
    m.path = FaultPath::LifterFetch {
        first_unavailable: addr,
    };
    m.origin = match trap {
        Some(Memory(crate::MemTrap::Mmio)) => FaultOrigin::TrapFetch,
        Some(Memory(crate::MemTrap::HostFault)) => FaultOrigin::HostAccess,
        Some(Memory(crate::MemTrap::Unmapped)) => FaultOrigin::GuestMap,
        Some(Policy(_)) => FaultOrigin::GuestMap,
        _ => FaultOrigin::Unknown,
    };
    m.instruction_rip = Knowledge::Known(at.ip);
    m.address = Knowledge::Known(FaultAddress::new(
        addr,
        AddressPrecision::ExactByte,
        if mode == CpuMode::Real16 {
            AddressSpace::RealModePhysical
        } else {
            AddressSpace::Linear
        },
    ));
    m.access = Knowledge::Known(FaultAccess::Execute);
    m.transfer_access = Knowledge::Known(FaultAccess::Execute);
    m.transfer_size = Knowledge::Known(1);
    m.transfer_unit = TransferUnit::FetchByte;
    m.progress.registers = RegisterProgress::InstructionEntry;
    m.progress.ram = MemoryEffects::None;
    m.progress.device = MemoryEffects::None;
    if mode == CpuMode::Real16 {
        m.cs_ip = Knowledge::Known((((at.pa - at.ip) >> 4) as u16, at.ip as u16));
    }
    if m.origin == FaultOrigin::GuestMap
        && crate::memory::is_canonical(addr)
        && mem.absent_byte(addr)
    {
        m.projection = Some(PageFaultProjection {
            code: ErrorCode::bits(0, 1),
            is_fetch: true,
        });
    }
    if let Some(Policy(verdict)) = trap {
        // SDM Vol 3A §5.7, Fig. 5-12: P=1 denotes protection, P=0 absence.
        // I/D depends on guest paging controls, so only the independent fetch
        // fact is asserted, as with the other fetch projections.
        m.projection = Some(PageFaultProjection {
            code: ErrorCode::bits(u32::from(verdict == crate::FetchVerdict::DenyProtection), 1),
            is_fetch: true,
        });
    }
    Exit::unmapped_memory(addr, AccessKind::Execute).with_fault_metadata(m)
}

#[cold]
#[inline(never)]
pub(crate) fn qualify_real16(mut exit: Exit, cpu: &CpuState) -> Exit {
    if let Exit::UnmappedMemory {
        metadata: Some(m), ..
    }
    | Exit::Exception {
        metadata: Some(m), ..
    } = &mut exit
    {
        m.cs_ip = Knowledge::Known((cpu.cs, cpu.rip as u16));
        if let Knowledge::Known(ref mut address) = m.address {
            address.space = AddressSpace::RealModePhysical;
        }
        // Multi-transfer real-mode instructions may update SP/registers before
        // delivery fails; do not extrapolate the flat-mode commit-last contract.
        if m.progress.rep.is_none() {
            m.progress.registers = RegisterProgress::Unknown;
        }
    }
    exit
}

// Detailed errors for FP traits whose compatibility signature returns only u64.
// Written/read only after failure, outside signal context. Every built-in FP
// memory error initializes it; consumption invalidates the record on the cold path.
#[derive(Clone, Copy)]
pub(crate) struct FpTransferFault {
    pub addr: u64,
    pub size: u16,
    pub origin: FaultOrigin,
    pub precision: AddressPrecision,
}
thread_local! { static FP_TRANSFER_FAULT: std::cell::Cell<Option<FpTransferFault>> = const { std::cell::Cell::new(None) }; }
#[cold]
#[inline(never)]
pub(crate) fn record_fp_fault(
    addr: u64,
    size: usize,
    origin: FaultOrigin,
    precision: AddressPrecision,
) -> u64 {
    FP_TRANSFER_FAULT.with(|slot| {
        slot.set(Some(FpTransferFault {
            addr,
            size: size as u16,
            origin,
            precision,
        }))
    });
    addr
}
#[cold]
#[inline(never)]
pub(crate) fn take_fp_fault() -> Option<FpTransferFault> {
    FP_TRANSFER_FAULT.with(|slot| slot.take())
}
#[cold]
#[inline(never)]
pub(crate) fn fp_exit(rip: u64, addr: u64, size: u16, access: AccessKind) -> Exit {
    let detail = take_fp_fault();
    let mut exit = memory_exit(
        rip,
        addr,
        detail.map_or(size, |d| d.size),
        access,
        false,
        false,
    );
    if let Exit::UnmappedMemory {
        metadata: Some(m), ..
    } = &mut exit
    {
        m.origin = detail.map_or(FaultOrigin::Unknown, |d| d.origin);
        if let Some(d) = detail {
            m.address =
                Knowledge::Known(FaultAddress::new(d.addr, d.precision, AddressSpace::Linear));
        }
    }
    exit
}

// Only VM/step callers that obtained this instruction from the strict lifter
// may qualify a decode cause. A standalone synthetic IrBlock stays Unknown.
#[cold]
#[inline(never)]
pub(crate) fn qualify_lifted(mut exit: Exit, info: Option<&InstructionFaultInfo>) -> Exit {
    if let (
        Some(info),
        Exit::Exception {
            vector,
            metadata: Some(m),
            ..
        },
    ) = (info, &mut exit)
    {
        if info.length_limit && *vector == 13 && m.origin == FaultOrigin::Unknown {
            m.origin = FaultOrigin::GuestException(FaultCause::InstructionLength);
            m.class = FaultClass::Fault;
            m.error_code = ErrorCode::bits(0, u32::MAX);
            m.progress.registers = RegisterProgress::InstructionEntry;
        } else if info
            .trap
            .is_some_and(|(v, advance)| v == *vector && advance != 0 && v != 1 && v != 3)
        {
            // SDM Vol 3A §7.13: INT n does not push a code even for vectors
            // that name exceptions with architectural error codes.
            m.origin = FaultOrigin::SoftwareInterrupt;
            m.class = FaultClass::Trap;
            m.error_code = ErrorCode::NoCode;
            m.progress.registers = RegisterProgress::CompletedInstruction;
        }
    }
    exit
}

#[cold]
pub(crate) fn attach_compiled_path(mut exit: Exit, pc: usize, helper: bool) -> Exit {
    let metadata = match &mut exit {
        Exit::UnmappedMemory { metadata, .. } | Exit::Exception { metadata, .. } => metadata,
        _ => return exit,
    };
    if let Some(m) = metadata {
        m.path = crate::codemap::fault_path(pc, m.instruction_rip, helper);
    }
    exit
}

#[cold]
pub(crate) fn attach_raw_host_fault(mut exit: Exit, raw: Option<RawHostFault>) -> Exit {
    if let Some(raw) = raw {
        match &mut exit {
            Exit::UnmappedMemory { metadata, .. } | Exit::Exception { metadata, .. } => {
                if let Some(metadata) = metadata {
                    metadata.raw_host_fault = Some(raw);
                }
            }
            _ => {}
        }
    }
    exit
}

#[cfg(test)]
mod host_address_tests {
    use super::*;
    use crate::{ir::MemoryAccess, FlagMask, Reg, Val};

    #[test]
    fn host_operand_recipe_reuses_lowered_address_operations() {
        let ops = [
            IrOp::ReadReg {
                dst: 0,
                reg: Reg::Rbx,
            },
            IrOp::ReadReg {
                dst: 1,
                reg: Reg::Rcx,
            },
            IrOp::Shl {
                dst: 2,
                a: Val::Temp(1),
                b: Val::Imm(3),
                size: 8,
                set_flags: FlagMask::NONE,
            },
            IrOp::Add {
                dst: 3,
                a: Val::Temp(0),
                b: Val::Temp(2),
                size: 8,
                set_flags: FlagMask::NONE,
            },
            IrOp::And {
                dst: 4,
                a: Val::Temp(3),
                b: Val::Imm(0xffff_ffff),
                size: 8,
                set_flags: FlagMask::NONE,
            },
            IrOp::ReadReg {
                dst: 5,
                reg: Reg::FsBase,
            },
            IrOp::Add {
                dst: 6,
                a: Val::Temp(4),
                b: Val::Temp(5),
                size: 8,
                set_flags: FlagMask::NONE,
            },
            IrOp::VLoad {
                fault: MemoryAccess::new(16, true, false),
                dst: 0,
                addr: Val::Temp(6),
                size: 16,
            },
        ];
        let mut cpu = CpuState::new();
        cpu.gpr[3] = u64::MAX - 15;
        cpu.gpr[1] = 5;
        cpu.fs_base = 0x10000;
        assert_eq!(
            OperandAddress::from_ops(&ops).unwrap().evaluate(&cpu),
            Some(0x10018)
        );
    }

    #[test]
    fn host_operand_recipe_leaves_unavailable_inputs_unknown() {
        let load = IrOp::Load {
            fault: MemoryAccess::new(8, true, false),
            dst: 0,
            addr: Val::Temp(8),
            size: 8,
        };
        let mut ops = vec![load.clone()];
        let cpu = CpuState::new();
        assert_eq!(OperandAddress::from_ops(&ops).unwrap().evaluate(&cpu), None);
        ops.insert(
            0,
            IrOp::WriteReg {
                reg: Reg::Rdi,
                src: Val::Imm(123),
                size: 8,
            },
        );
        assert_eq!(OperandAddress::from_ops(&ops).unwrap().evaluate(&cpu), None);
        ops.push(IrOp::Load {
            fault: MemoryAccess::new(8, true, false),
            dst: 1,
            addr: Val::Imm(456),
            size: 8,
        });
        assert!(OperandAddress::from_ops(&ops).is_none());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_store_fault_is_not_mistaken_for_split_store_probe() {
        let mut mem = Memory::new(crate::MemoryModel::Flat { size: 0x4000 });
        mem.map(0, 0x4000, crate::Prot::RWX, crate::RegionKind::Ram)
            .unwrap();
        mem.write_bytes(0x1000, &[0x88, 0x07, 0xf4]).unwrap();
        let block =
            crate::lift::lift_block(&mem, crate::lift::FetchAddr::flat(0x1000), CpuMode::Long64)
                .unwrap();
        let mut cpu = CpuState::new();
        cpu.rip = 0x1000;
        for precise in [false, true] {
            let info = InstructionFaultInfo::for_block_with_split_stores(&block, precise);
            let exit = memory_exit(cpu.rip, 0x2000, 1, crate::AccessKind::Write, true, true);
            let exit = finish(exit, Some(&info[0]), &cpu, &mem, CpuMode::Long64, true);
            assert_eq!(
                exit.fault_metadata().unwrap().progress().ram(),
                MemoryEffects::PossiblyPartial
            );
        }
    }

    #[test]
    fn crossing_probe_proves_no_bytes_but_transfer_fault_retains_uncertainty() {
        let mut mem = Memory::new(crate::MemoryModel::Flat { size: 0x4000 });
        mem.map(0, 0x4000, crate::Prot::RWX, crate::RegionKind::Ram)
            .unwrap();
        mem.write_bytes(0x1000, &[0x0f, 0x11, 0x07, 0xf4]).unwrap();
        let block =
            crate::lift::lift_block(&mem, crate::lift::FetchAddr::flat(0x1000), CpuMode::Long64)
                .unwrap();

        let mut cpu = CpuState::new();
        cpu.rip = 0x1000;
        for precise in [false, true] {
            let info = InstructionFaultInfo::for_block_with_split_stores(&block, precise);
            for compiled in [false, true] {
                for (width, host) in [(1, true), (16, true), (16, false)] {
                    let exit =
                        memory_exit(cpu.rip, 0x2000, width, crate::AccessKind::Write, host, true);
                    let exit = finish(exit, Some(&info[0]), &cpu, &mem, CpuMode::Long64, compiled);
                    let expected = if precise && (!host || (compiled && width == 1)) {
                        MemoryEffects::None
                    } else {
                        MemoryEffects::PossiblyPartial
                    };
                    assert_eq!(exit.fault_metadata().unwrap().progress().ram(), expected);
                }
            }
        }
    }
}
