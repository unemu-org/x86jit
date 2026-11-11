//! IR interpreter (§8.1). Walks an `IrBlock`'s ops over a `temps` vector and a
//! `&mut CpuState`, reading/writing shared guest `&Memory`. Slow but simple — the
//! oracle the JIT is validated against.
//!
//! RIP-on-trap convention (§8, §16), identical to the JIT's: on a memory trap RIP
//! is set to the FAULTING instruction (`cur_addr`, from `InsnStart`) so the user
//! can map/handle and retry; after `syscall`/`hlt` RIP is PAST the instruction.

mod mmx;
pub(crate) mod ordering;
mod result;
use result::InterpExit;
pub mod softfloat;

use crate::exit::{AccessKind, Exit, PortDir, StepResult};
use std::cell::Cell;
use std::cmp::Ordering;

use crate::ir::{
    Cond, FPrec, FenceKind, FlagMask, FloatBinOp, FloatUnOp, HFloatOp, HIntOp, IrBlock, IrOp,
    PackedBinOp, RepKind, StrOp, VLogicOp, Val,
};
use crate::memory::{Access, InterpreterMemory, MemTrap, Memory, RunningMemory};
use crate::state::{CpuState, Flags, Reg};

/// `gpr[]` slot for RSP (used by push/pop-style stack ops in Call/Ret).
const RSP: usize = 4;

/// Per-block interpreter bookkeeping the dispatcher needs but that must NOT live on
/// `CpuState` (ABI-frozen, §17.6 sub-seam c). Filled by [`interpret_block`]/[`step_one`]
/// and consumed by [`crate::vm::Vcpu::run`] for the retired-instruction counter and the
/// STI/SS-shadow interrupt-injection gate. Callers must set `consistency` to the
/// execution tier and carry both shadow fields across calls; other inputs may use
/// `RetireInfo::default()`.
#[derive(Default, Clone, Copy, Debug)]
pub struct RetireInfo<'a> {
    /// In: linear code-segment base (CS << 4 in Real16, zero in flat modes). Same-block
    /// self-modifying-code detection compares stores against the block's linear span
    /// `code_base + guest_start`, so a direct caller of [`interpret_block`] running
    /// segmented Real16 code must set it; the default zero is right only for flat modes
    /// or CS = 0. VM dispatch and [`step_one`] set it.
    pub code_base: u64,
    /// In: optional interpreter trace sink. Never enabled from the environment.
    pub trace: Option<&'a crate::lockstep::Trace>,
    /// Guest ordering tier; selected once before entering the op walk.
    pub consistency: crate::MemConsistency,
    /// In: optional per-vcpu notification word, polled by long string operations.
    pub exit_request: Option<&'a std::sync::atomic::AtomicU64>,
    /// Number of guest instructions that **retired** (completed) during this call. A
    /// mid-instruction memory trap (RIP left on the faulting instruction for retry) does
    /// NOT count it — on the retry it retires and is counted once. Straight-line no-trap
    /// blocks count exactly one per decoded instruction.
    pub retired: u64,
    /// In/out: STI inhibition at the current instruction boundary. An incomplete
    /// successor preserves it (SDM Vol 3A §7.8.1; Vol 2B §4.3 STI).
    pub sti_shadow: bool,
    /// In/out: MOV/POP SS inhibition, distinct from STI for consecutive-load policy
    /// (SDM Vol 3A §7.8.3; Vol 2B §4.3 MOV/POP).
    pub ss_shadow: bool,
}

#[derive(Default)]
struct ShadowState {
    // Bits 0..2 hold incoming STI/SS; bits 2..4 hold newly armed STI/SS.
    bits: u8,
    observed_retired: u64,
}

impl ShadowState {
    const STI: u8 = 1;
    const SS: u8 = 2;
    const ARM_SHIFT: u8 = 2;

    fn new(sti: bool, ss: bool) -> Self {
        Self {
            bits: u8::from(sti) | (u8::from(ss) << 1),
            observed_retired: 0,
        }
    }

    fn complete(&mut self) {
        self.bits >>= Self::ARM_SHIFT;
    }

    // Observe completion lazily, using the existing retirement counter. No
    // shadow state needs to be loaded or stored at ordinary InsnStart markers.
    fn complete_before(&mut self, retired: u64) {
        self.bits >>= u64::from(Self::ARM_SHIFT) * (retired - self.observed_retired).min(2);
        self.observed_retired = retired;
    }

    #[cold]
    #[inline(never)]
    fn finish(&mut self, retired: u64, delivered: bool, completed: bool) {
        self.complete_before(retired);
        if delivered {
            // SDM Vol 3A §§7.8.1, 7.8.3: delivered exceptions end inhibition even
            // though the faulting instruction did not retire. A host exit does not.
            self.bits = 0;
        } else if completed {
            self.complete();
        }
    }

    fn arm_sti(&mut self, value: bool, entry_if: bool) {
        let mask = Self::STI << Self::ARM_SHIFT;
        self.bits = (self.bits & !mask) | (u8::from(value && !entry_if) << Self::ARM_SHIFT);
    }

    fn arm_ss(&mut self) {
        let mask = Self::SS << Self::ARM_SHIFT;
        self.bits = (self.bits & !mask) | (u8::from(self.bits & Self::SS == 0) * mask);
    }

    fn complete_rep_prefix(&mut self, before: u64, cpu: &CpuState, retired: u64) {
        // Decision-16 §4: the first completed REP iteration consumes inhibition
        // (SDM Vol 3A §§7.6, 7.8.1, 7.8.3). Sample the committed counter once
        // per helper call, adding no bookkeeping to the iteration loop.
        if cpu.gpr[1] != before {
            self.complete_before(retired);
            self.complete();
        }
    }
}

// Local to one stepped call; absent from CpuState and the compiled ABI.
#[derive(Default)]
pub(crate) struct StepContext {
    pub remaining: u64,
    pub steps: u64,
    pub pending_irq: bool,
    pub string_interrupted: bool,
    rep_counted: bool,
    cut: bool,
}

// A disabled specialization passes a zero-sized value, not an initialized dummy
// ledger or an extra pointer in the helper ABI. Only STEPPED code calls get().
trait BudgetState<T> {
    fn get(&mut self) -> &mut T;
}
impl<T> BudgetState<T> for () {
    fn get(&mut self) -> &mut T {
        unreachable!("unstepped specialization")
    }
}
impl<T> BudgetState<T> for &mut T {
    fn get(&mut self) -> &mut T {
        self
    }
}

impl StepContext {
    pub fn new(remaining: u64) -> Self {
        Self {
            remaining,
            ..Self::default()
        }
    }
    fn charge(&mut self) {
        debug_assert!(self.remaining > 0);
        self.remaining -= 1;
        self.steps += 1;
    }

    fn boundary(
        &self,
        cpu: &CpuState,
        shadows: &ShadowState,
        exit_request: Option<&std::sync::atomic::AtomicU64>,
    ) -> Option<WalkResult> {
        if self.remaining == 0 {
            return Some(WalkResult::exit(Exit::BudgetExhausted));
        }
        let pending = cpu.mmio_retry_active
            || cpu.pending_mmio.is_some()
            || cpu.pending_mmio_write
            || cpu.pending_port_in.is_some()
            || cpu.port_string_in.is_some();
        if !pending
            && exit_request.is_some_and(|word| {
                word.load(std::sync::atomic::Ordering::Relaxed) & crate::vm::EXIT_REQUESTED != 0
            })
        {
            return Some(WalkResult::exit(Exit::Interrupted));
        }
        if self.pending_irq && cpu.flags.if_ && shadows.bits == 0 && !pending {
            // Return to the dispatcher to deliver; no guest instruction is in flight.
            // Reuse the no-retirement result so unstepped result layout stays fixed.
            // Inhibition is already empty at this boundary.
            return Some(WalkResult::FaultDelivered);
        }
        None
    }
}

/// Reusable interpreter temporaries. The VM supplies an allocation-tracked buffer;
/// direct callers can continue to use `Vec<u64>`.
#[doc(hidden)]
pub trait ScratchBuffer {
    /// Return `count` zeroed temporary slots, growing storage only when necessary.
    fn zeroed(&mut self, count: usize) -> &mut [u64];
}

impl ScratchBuffer for Vec<u64> {
    #[inline]
    fn zeroed(&mut self, count: usize) -> &mut [u64] {
        self.clear();
        self.resize(count, 0);
        self
    }
}

/// Cached Real16 walk with instruction/iteration boundaries (decision-16 §3).
pub(crate) fn interpret_block_steps<
    const ROUTED: bool,
    const DEFAULT: bool,
    const STOP_WHEN_IRQ_READY: bool,
>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    mem: &Memory,
    scratch: &mut impl ScratchBuffer,
    info: &mut RetireInfo,
    steps: &mut StepContext,
) -> BlockResult {
    #[cfg(target_arch = "aarch64")]
    match info.consistency {
        crate::MemConsistency::AcqRel => {
            return interpret_steps_ordered::<ROUTED, DEFAULT, 1, STOP_WHEN_IRQ_READY>(
                ir, cpu, mem, scratch, info, steps,
            )
        }
        crate::MemConsistency::FullTso => {
            return interpret_steps_ordered::<ROUTED, DEFAULT, 2, STOP_WHEN_IRQ_READY>(
                ir, cpu, mem, scratch, info, steps,
            )
        }
        crate::MemConsistency::Fast => {}
    }
    interpret_steps_ordered::<ROUTED, DEFAULT, 0, STOP_WHEN_IRQ_READY>(
        ir, cpu, mem, scratch, info, steps,
    )
}

fn interpret_steps_ordered<
    const ROUTED: bool,
    const DEFAULT: bool,
    const ORDER: u8,
    const STOP_WHEN_IRQ_READY: bool,
>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    mem: &Memory,
    scratch: &mut impl ScratchBuffer,
    info: &mut RetireInfo,
    steps: &mut StepContext,
) -> BlockResult {
    if ir.legacy_context.is_some() {
        return BlockResult::exit(Exit::UnsupportedSystemContext {
            addr: cpu.rip,
            operation: crate::UnsupportedSystemOperation::LegacyExecution,
        });
    }
    let mem = Access::<ROUTED, ORDER>::new(mem);
    if cpu.mmio_retry_active {
        interpret_block_inner::<true, ROUTED, DEFAULT, ORDER, true, STOP_WHEN_IRQ_READY>(
            ir, cpu, mem, scratch, info, steps,
        )
    } else {
        interpret_block_inner::<false, ROUTED, DEFAULT, ORDER, true, STOP_WHEN_IRQ_READY>(
            ir, cpu, mem, scratch, info, steps,
        )
    }
}

/// Single-step the interpreter over exactly one instruction at `cpu.rip` (§5.2,
/// M4-T10). The dispatcher calls this to service an MMIO access the JIT deferred:
/// the interpreter re-executes the faulting instruction, which either traps out
/// (`MmioRead`/`MmioWrite`) or — on resume, once the embedder supplied the value
/// via `complete_mmio_read` / acknowledged the write via `complete_mmio_write` —
/// consumes it and advances RIP. A lift/decode error becomes the matching exit.
pub fn step_one(
    mem: &Memory,
    cpu: &mut CpuState,
    mode: crate::lift::CpuMode,
    scratch: &mut impl ScratchBuffer,
    info: &mut RetireInfo,
) -> StepResult {
    let _fp_mode = crate::jit_abi::fp::RunMode::new();
    if mem.host_fault_recovery() {
        step_one_mode::<true>(mem, cpu, mode, scratch, info)
    } else {
        step_one_mode::<false>(mem, cpu, mode, scratch, info)
    }
}

pub(crate) fn step_one_mode<const ROUTED: bool>(
    mem: &Memory,
    cpu: &mut CpuState,
    mode: crate::lift::CpuMode,
    scratch: &mut impl ScratchBuffer,
    info: &mut RetireInfo,
) -> StepResult {
    if cpu.mmio_retry_active && !cpu.owns_mmio_retry(cpu.rip) {
        cpu.clear_mmio_parts();
    }
    // §17.6: in Real16 `cpu.rip` is the 16-bit IP; code is fetched from
    // `cs_base + IP`. `FetchAddr::for_mode` forms the physical fetch address while the
    // decoder still resolves against the IP (Long64/Compat32 are flat: pa == ip).
    let at = crate::lift::FetchAddr::for_mode(mode, cpu.rip, cpu.cs);
    info.code_base = at.pa - at.ip;
    let mut fetch_failure = None;
    let result = match crate::lift::lift_one_with_fault(mem, at, mode, &mut fetch_failure) {
        Ok(ir) => {
            return match interpret_block_mode::<ROUTED>(&ir, cpu, mem, scratch, info) {
                StepResult::Exit(exit @ Exit::Exception { .. }) => {
                    let exit = qualify_lifted_exit(exit, &ir);
                    StepResult::Exit(if mode == crate::CpuMode::Real16 {
                        crate::fault::qualify_real16(exit, cpu)
                    } else {
                        exit
                    })
                }
                StepResult::Exit(exit) if mode == crate::CpuMode::Real16 => {
                    StepResult::Exit(crate::fault::qualify_real16(exit, cpu))
                }
                other => other,
            }
        }
        Err(crate::lift::LiftError::Unsupported { addr, bytes, len }) => {
            StepResult::Exit(Exit::UnknownInstruction { addr, bytes, len })
        }
        Err(crate::lift::LiftError::DecodeFault { addr }) => {
            StepResult::Exit(crate::fault::fetch_exit(mem, at, mode, addr, fetch_failure))
        }
    };
    if let StepResult::Exit(exit) = &result {
        cpu.finish_mmio_retry(Some(exit));
    }
    result
}

/// Execute application IR. Legacy-context IR returns a typed unsupported exit
/// before changing CPU, memory, scratch storage, or retirement bookkeeping.
pub fn interpret_block(
    ir: &IrBlock,
    cpu: &mut CpuState,
    mem: &Memory,
    scratch: &mut impl ScratchBuffer,
    info: &mut RetireInfo,
) -> StepResult {
    if ir.legacy_context.is_some() {
        return StepResult::Exit(Exit::UnsupportedSystemContext {
            addr: cpu.rip,
            operation: crate::UnsupportedSystemOperation::LegacyExecution,
        });
    }
    let _fp_mode = crate::jit_abi::fp::RunMode::new();
    if mem.host_fault_recovery() {
        interpret_block_mode::<true>(ir, cpu, mem, scratch, info)
    } else {
        interpret_block_mode::<false>(ir, cpu, mem, scratch, info)
    }
}

// A private carrier suppresses implicit Drop guards in the successful op walk.
// Every Exit is explicitly transferred to the public result on the cold return;
// no metadata owner is stored or cleared at instruction/iteration entry.
pub(crate) enum InterpResult {
    Continue,
    Exit(InterpExit),
}
impl InterpResult {
    #[inline(always)]
    fn exit(exit: Exit) -> Self {
        Self::Exit(InterpExit::from_exit(exit))
    }
    pub(crate) fn into_step(self) -> StepResult {
        match self {
            Self::Continue => StepResult::Continue,
            Self::Exit(exit) => StepResult::Exit(exit.into_exit()),
        }
    }
}

// Mode writes carry a distinct internal completion tag, so the dispatcher need
// not reload MXCSR after ordinary interpreted blocks. The public result is unchanged.
#[repr(u8)]
pub(crate) enum BlockResult {
    Continue,
    ModeChange,
    Exit(InterpExit),
}

// A fault tag contains no payload: the producer already initialized the cold
// owned record. This keeps even argument copies out of the successful op walk.
enum WalkResult {
    Continue,
    ModeChange,
    FaultDelivered,
    Exit(InterpExit),
    Fault,
}
impl From<InterpResult> for WalkResult {
    #[inline(always)]
    fn from(result: InterpResult) -> Self {
        match result {
            InterpResult::Continue => Self::Continue,
            InterpResult::Exit(
                InterpExit::DetailedUnmapped { .. } | InterpExit::DetailedException { .. },
            ) => Self::Fault,
            InterpResult::Exit(exit) => Self::Exit(exit),
        }
    }
}
impl WalkResult {
    #[inline(always)]
    fn exit(exit: Exit) -> Self {
        InterpResult::exit(exit).into()
    }

    fn delivered_fault(result: InterpResult) -> Self {
        match result {
            InterpResult::Continue => Self::FaultDelivered,
            other => other.into(),
        }
    }
}

impl BlockResult {
    #[inline(always)]
    fn exit(exit: Exit) -> Self {
        Self::Exit(InterpExit::from_exit(exit))
    }

    fn into_step(self) -> StepResult {
        match self {
            Self::Continue | Self::ModeChange => StepResult::Continue,
            Self::Exit(exit) => StepResult::Exit(exit.into_exit()),
        }
    }
}

pub(crate) fn interpret_block_mode<const ROUTED: bool>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    mem: &Memory,
    scratch: &mut impl ScratchBuffer,
    info: &mut RetireInfo,
) -> StepResult {
    if cpu.mxcsr & softfloat::MODE_MASK == 0 {
        interpret_block_fp::<ROUTED, true>(ir, cpu, mem, scratch, info).into_step()
    } else {
        interpret_block_fp::<ROUTED, false>(ir, cpu, mem, scratch, info).into_step()
    }
}

// Decide once per block: the default specialization contains the original host
// arithmetic, with no MXCSR load or mode branch in any floating-point arm.
pub(crate) fn interpret_block_fp<const ROUTED: bool, const DEFAULT: bool>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    mem: &Memory,
    scratch: &mut impl ScratchBuffer,
    info: &mut RetireInfo,
) -> BlockResult {
    #[cfg(target_arch = "aarch64")]
    if info.consistency != crate::MemConsistency::Fast {
        return interpret_block_nonfast::<ROUTED, DEFAULT>(ir, cpu, mem, scratch, info);
    }
    interpret_block_ordered::<ROUTED, DEFAULT, 0>(ir, cpu, mem, scratch, info)
}

// Keep ordered-tier setup out of the Fast block wrapper.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn interpret_block_nonfast<const ROUTED: bool, const DEFAULT: bool>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    mem: &Memory,
    scratch: &mut impl ScratchBuffer,
    info: &mut RetireInfo,
) -> BlockResult {
    match info.consistency {
        crate::MemConsistency::AcqRel => {
            interpret_block_ordered::<ROUTED, DEFAULT, 1>(ir, cpu, mem, scratch, info)
        }
        crate::MemConsistency::FullTso => {
            interpret_block_ordered::<ROUTED, DEFAULT, 2>(ir, cpu, mem, scratch, info)
        }
        crate::MemConsistency::Fast => unreachable!(),
    }
}

fn interpret_block_ordered<const ROUTED: bool, const DEFAULT: bool, const ORDER: u8>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    mem: &Memory,
    scratch: &mut impl ScratchBuffer,
    info: &mut RetireInfo,
) -> BlockResult {
    if ir.legacy_context.is_some() {
        return BlockResult::exit(Exit::UnsupportedSystemContext {
            addr: cpu.rip,
            operation: crate::UnsupportedSystemOperation::LegacyExecution,
        });
    }
    let mem = Access::<ROUTED, ORDER>::new(mem);
    if cpu.mmio_retry_active {
        interpret_block_inner::<true, ROUTED, DEFAULT, ORDER, false, false>(
            ir,
            cpu,
            mem,
            scratch,
            info,
            &mut StepContext::default(),
        )
    } else {
        interpret_block_inner::<false, ROUTED, DEFAULT, ORDER, false, false>(
            ir,
            cpu,
            mem,
            scratch,
            info,
            &mut StepContext::default(),
        )
    }
}

#[inline(always)]
fn interp_mxcsr<const DEFAULT: bool>(cpu: &CpuState) -> u32 {
    if DEFAULT {
        0
    } else {
        cpu.mxcsr
    }
}

fn interpret_block_inner<
    const RETRY: bool,
    const ROUTED: bool,
    const DEFAULT: bool,
    const ORDER: u8,
    const STEPPED: bool,
    const STOP_WHEN_IRQ_READY: bool,
>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    scratch: &mut impl ScratchBuffer,
    info: &mut RetireInfo,
    steps: &mut StepContext,
) -> BlockResult {
    let temps = scratch.zeroed(ir.temp_count as usize);
    let cur_addr = Cell::new(ir.guest_start);
    let started = Cell::new(1u8);
    // Retired-instruction accounting (§17.6, sub-seam c). `InsnStart` marks the start of
    // each guest instruction; when a *later* `InsnStart` runs, the previous instruction
    // provably completed, so we retire it then. The final instruction retires when the
    // walk completes its terminator or falls through, including a branch to itself.
    // SDM Vol 3A §7.5: delivering a fault leaves the instruction incomplete for
    // restart, whereas a trap follows a completed instruction.
    // Exit paths retain the saved-RIP convention: syscall/hlt/port I/O and advancing
    // traps retire; faults, MMIO and interrupted REP leave RIP on the instruction for
    // retry and do not retire it. `started` gates the first `InsnStart` (nothing
    // to retire yet). Shadows retain the incoming inhibition until completion;
    // shadow-producing ops stage new inhibition separately from that input.
    let mut retired: u64 = 0;
    // 0 = refetch after a store, 1 = no instruction yet, 2 = instruction in flight.
    // Bit 1 contributes retirement; the zero test handles only refetch.
    let mut shadows = ShadowState::new(info.sti_shadow, info.ss_shadow);

    // The op walk lives in `walk_ops`; its every early `return` funnels back here for the
    // single final-instruction retirement below. `cur_addr`/`retired`/`started`/
    // `shadows` are shared with the walk and read afterwards.
    if STEPPED {
        steps.cut = false;
    }
    let start = info.code_base + ir.guest_start;
    let host_start = (mem.host_base() as u64).wrapping_add(start.wrapping_sub(mem.guest_base()));
    let running = RunningMemory {
        memory: mem,
        ir,
        start,
        end: start + u64::from(ir.guest_len),
        host_start,
        host_end: host_start + u64::from(ir.guest_len),
        current: &cur_addr,
        started: &started,
    };
    let mem = Access::<ROUTED, ORDER, _>::new(&running);
    let result = if STEPPED {
        walk_ops::<RETRY, ROUTED, DEFAULT, false, ORDER, STEPPED, STOP_WHEN_IRQ_READY>(
            ir,
            cpu,
            mem,
            temps,
            &cur_addr,
            &mut retired,
            &started,
            &mut shadows,
            info.exit_request,
            info.trace,
            &mut *steps,
        )
    } else {
        walk_ops::<RETRY, ROUTED, DEFAULT, false, ORDER, STEPPED, STOP_WHEN_IRQ_READY>(
            ir,
            cpu,
            mem,
            temps,
            &cur_addr,
            &mut retired,
            &started,
            &mut shadows,
            info.exit_request,
            info.trace,
            (),
        )
    };
    if STEPPED && steps.cut {
        started.set(1);
    }
    let completed = match &result {
        WalkResult::Continue | WalkResult::ModeChange => true,
        WalkResult::FaultDelivered => false,
        WalkResult::Exit(_) | WalkResult::Fault => cpu.rip != cur_addr.get(),
    };
    if shadows.bits != 0 {
        shadows.finish(
            retired,
            matches!(&result, WalkResult::FaultDelivered),
            (started.get() == 2) && completed && cpu.pending_port_in.is_none(),
        );
    }

    let result = match result {
        WalkResult::Continue | WalkResult::FaultDelivered => BlockResult::Continue,
        WalkResult::ModeChange => BlockResult::ModeChange,
        WalkResult::Exit(exit) => BlockResult::Exit(exit),
        WalkResult::Fault => {
            let exit =
                result::consume_current().expect("fault return must initialize its cold record");
            BlockResult::exit(finish_fault::<STEPPED>(
                exit,
                ir,
                cur_addr.get(),
                cpu,
                mem,
                temps,
            ))
        }
    };

    // Final-instruction retirement (§17.6, sub-seam c) depends on completion, not
    // whether the branch target differs from the instruction's own address.
    if (started.get() == 2) && completed {
        retired += 1;
        if STEPPED && !steps.rep_counted {
            steps.charge();
        }
    }
    // Retry cleanup already inspects results only on this existing cold route.
    let result = if RETRY
        || matches!(
            &result,
            BlockResult::Exit(InterpExit::MmioRead { .. } | InterpExit::MmioWrite { .. })
        ) {
        match result {
            BlockResult::Exit(exit) => {
                let exit = exit.into_exit();
                cpu.finish_mmio_retry(Some(&exit));
                BlockResult::exit(exit)
            }
            other => {
                cpu.finish_mmio_retry(None);
                other
            }
        }
    } else {
        result
    };
    info.retired = retired;
    info.sti_shadow = shadows.bits & ShadowState::STI != 0;
    info.ss_shadow = shadows.bits & ShadowState::SS != 0;
    result
}

#[cold]
#[inline(never)]
fn finish_fault<const STEPPED: bool>(
    exit: Exit,
    ir: &IrBlock,
    rip: u64,
    cpu: &CpuState,
    mem: &Memory,
    temps: &[u64],
) -> Exit {
    if !matches!(exit, Exit::Exception { .. } | Exit::UnmappedMemory { .. }) {
        return exit;
    }
    let start = ir
        .ops
        .iter()
        .position(|op| matches!(op, IrOp::InsnStart { guest_addr } if *guest_addr == rip))
        .unwrap_or(0);
    let end = ir.ops[start + 1..]
        .iter()
        .position(|op| matches!(op, IrOp::InsnStart { .. }))
        .map_or(ir.ops.len(), |i| start + 1 + i);
    let next = match ir.ops.get(end) {
        Some(IrOp::InsnStart { guest_addr }) => Some(*guest_addr),
        _ => Some(ir.next_ip()),
    };
    let info =
        crate::fault::InstructionFaultInfo::from_ops(rip, next, &ir.ops[start..end], ir.mode);
    // Per-access classification retained Long64/SS evidence. The VM/step caller
    // qualifies Real16 physical addresses with its explicit decode mode.
    let mode = crate::CpuMode::Long64;
    let mut exit = crate::fault::finish(exit, Some(&info), cpu, mem, mode, false);
    if STEPPED && cpu.owns_mmio_retry(cpu.rip) {
        let addresses = ir.ops[start..end].iter().find_map(|op| match op {
            IrOp::RepString {
                addr_bits,
                seg_base,
                ..
            } => {
                let mask = match addr_bits {
                    16 => 0xffff,
                    32 => 0xffff_ffff,
                    _ => u64::MAX,
                };
                Some([
                    read_val(*seg_base, temps).wrapping_add(cpu.gpr[RSI] & mask),
                    cpu.gpr[RDI] & mask,
                ])
            }
            IrOp::RepStringReal {
                ds_base, es_base, ..
            } => Some([
                read_val(*ds_base, temps).wrapping_add(cpu.gpr[RSI] & 0xffff),
                read_val(*es_base, temps).wrapping_add(cpu.gpr[RDI] & 0xffff),
            ]),
            _ => None,
        });
        if let Some(addresses) = addresses {
            match &mut exit {
                Exit::Exception {
                    metadata: Some(m), ..
                }
                | Exit::UnmappedMemory {
                    metadata: Some(m), ..
                } => {
                    m.resolve_string_transfer_addresses(addresses);
                }
                _ => {}
            }
        }
    }
    exit
}

#[cold]
#[inline(never)]
pub(crate) fn qualify_lifted_exit(exit: Exit, ir: &IrBlock) -> Exit {
    let rip = match exit.fault_metadata().map(|m| m.instruction_rip()) {
        Some(crate::Knowledge::Known(rip)) => rip,
        _ => return exit,
    };
    let infos = crate::fault::InstructionFaultInfo::for_block(ir);
    crate::fault::qualify_lifted(exit, infos.iter().find(|i| i.rip == rip))
}

/// The per-op interpreter walk. Every early `return` here exits `walk_ops`; the caller
/// then applies the single final-instruction retirement, reading the post-walk values of
/// the `&mut` retirement counters and shadow state.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn walk_ops<
    const RETRY: bool,
    const ROUTED: bool,
    const DEFAULT: bool,
    const CHILD: bool,
    const ORDER: u8,
    const STEPPED: bool,
    const STOP_WHEN_IRQ_READY: bool,
>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER, impl InterpreterMemory>,
    temps: &mut [u64],
    cur_addr: &Cell<u64>,
    retired: &mut u64,
    started: &Cell<u8>,
    shadows: &mut ShadowState,
    exit_request: Option<&std::sync::atomic::AtomicU64>,
    trace: Option<&crate::lockstep::Trace>,
    mut budget: impl BudgetState<StepContext>,
) -> WalkResult {
    let mut bracket = crate::lockstep::begin_with_trace(if CHILD { None } else { trace });
    let mut entry_flags = cpu.flags;
    for op in &ir.ops {
        if CHILD {
            mem.0.begin_op(op, temps, cpu);
        }
        match op {
            IrOp::InsnStart { guest_addr } => {
                if !CHILD && started.get() == 0 {
                    // The store completed; leave cur_addr on it for retirement.
                    started.set(2);
                    cpu.rip = *guest_addr;
                    if bracket.active() {
                        crate::lockstep::flush(&mut bracket, cpu, mem);
                    }
                    return WalkResult::Continue;
                }
                if STEPPED {
                    let steps = budget.get();
                    if started.get() == 2 {
                        *retired += 1;
                        if !steps.rep_counted {
                            steps.charge();
                        }
                        if cpu.mmio_retry_active {
                            cpu.clear_mmio_parts();
                        }
                    }
                    shadows.complete_before(*retired);
                    // Install the stopped marker's IP before any trace or guest effect.
                    cpu.rip = *guest_addr & 0xffff;
                    if let Some(stop) = steps.boundary(cpu, shadows, exit_request) {
                        steps.cut = true;
                        return stop;
                    }
                    if STOP_WHEN_IRQ_READY && (started.get() == 2) {
                        // Poll Vcpu::irq_ready only after exporting the completed shadows.
                        steps.cut = true;
                        return WalkResult::Continue;
                    }
                    steps.rep_counted = false;
                }
                entry_flags = cpu.flags;
                if RETRY
                    && cpu.mmio_retry_active
                    && ((started.get() == 2) || !cpu.owns_mmio_retry(*guest_addr))
                {
                    cpu.clear_mmio_parts();
                }
                if !STEPPED {
                    *retired += u64::from(started.get() >> 1);
                }
                started.set(2);
                cur_addr.set(*guest_addr);
                if !CHILD && bracket.active() {
                    crate::lockstep::on_insn_start(&mut bracket, cpu, mem, *guest_addr);
                }
            }
            IrOp::MmxConvert {
                dst,
                lo,
                hi,
                kind,
                to_mmx,
            } => {
                let v = u128::from(read_val(*lo, temps)) | (u128::from(read_val(*hi, temps)) << 64);
                let r = vector::packed_cvt128_mode_specialized::<DEFAULT>(
                    v,
                    kind,
                    interp_mxcsr::<DEFAULT>(cpu),
                );
                if *to_mmx {
                    let slot = &mut cpu.fpr[*dst as usize];
                    slot[..8].copy_from_slice(&(r as u64).to_le_bytes());
                    slot[8..].fill(0xff);
                } else if matches!(kind, crate::ir::PackedCvtKind::Dq2Ps) {
                    cpu.xmm[*dst as usize] =
                        (cpu.xmm[*dst as usize] & (u128::MAX << 64)) | u128::from(r as u64);
                } else {
                    cpu.xmm[*dst as usize] = r;
                }
            }
            IrOp::MmxRead { dst, reg } => {
                temps[*dst as usize] =
                    u64::from_le_bytes(cpu.fpr[*reg as usize][..8].try_into().unwrap());
            }
            IrOp::MmxWrite { reg, src } => {
                let slot = &mut cpu.fpr[*reg as usize];
                slot[..8].copy_from_slice(&read_val(*src, temps).to_le_bytes());
                slot[8..].fill(0xff);
            }
            IrOp::MmxState { empty } => {
                // SDM Vol 3A §15.2 Table 15-2: only TOP and tags change.
                cpu.fpu_top = 0;
                cpu.fpu_empty = if *empty { 0xff } else { 0 };
            }
            IrOp::MmxCompute { dst, a, b, op } => {
                temps[*dst as usize] = mmx::compute(read_val(*a, temps), read_val(*b, temps), *op);
            }
            IrOp::ReadReg { dst, reg } => {
                if let Some(r) = exec_read_reg(cpu, temps, dst, reg) {
                    return r.into();
                }
            }
            IrOp::WriteReg { reg, src, size } => {
                if let Some(r) = exec_write_reg(cpu, temps, reg, src, size) {
                    return r.into();
                }
            }
            IrOp::Add {
                dst,
                a,
                b,
                size,
                set_flags,
            } => {
                if let Some(r) = exec_add(cpu, temps, dst, a, b, size, set_flags) {
                    return r.into();
                }
            }
            IrOp::Adc {
                dst,
                a,
                b,
                size,
                set_flags,
            } => {
                if let Some(r) = exec_adc(cpu, temps, dst, a, b, size, set_flags) {
                    return r.into();
                }
            }
            IrOp::Adox {
                dst,
                a,
                b,
                size,
                set_flags,
            } => {
                let r = alu_add(
                    read_val(*a, temps),
                    read_val(*b, temps),
                    u64::from(cpu.flags.of),
                    *size,
                );
                temps[*dst as usize] = r.res;
                if set_flags.0 & 0b10_0000 != 0 {
                    cpu.flags.of = r.cf;
                }
            }
            IrOp::Sub {
                dst,
                a,
                b,
                size,
                set_flags,
            } => {
                if let Some(r) = exec_sub(cpu, temps, dst, a, b, size, set_flags) {
                    return r.into();
                }
            }
            IrOp::Sbb {
                dst,
                a,
                b,
                size,
                set_flags,
            } => {
                if let Some(r) = exec_sbb(cpu, temps, dst, a, b, size, set_flags) {
                    return r.into();
                }
            }
            IrOp::And {
                dst,
                a,
                b,
                size,
                set_flags,
            } => {
                if let Some(r) = exec_and(cpu, temps, dst, a, b, size, set_flags) {
                    return r.into();
                }
            }
            IrOp::Or {
                dst,
                a,
                b,
                size,
                set_flags,
            } => {
                if let Some(r) = exec_or(cpu, temps, dst, a, b, size, set_flags) {
                    return r.into();
                }
            }
            IrOp::Xor {
                dst,
                a,
                b,
                size,
                set_flags,
            } => {
                if let Some(r) = exec_xor(cpu, temps, dst, a, b, size, set_flags) {
                    return r.into();
                }
            }
            IrOp::Shl {
                dst,
                a,
                b,
                size,
                set_flags,
            } => {
                if let Some(r) = exec_shl(cpu, temps, dst, a, b, size, set_flags) {
                    return r.into();
                }
            }
            IrOp::Shr {
                dst,
                a,
                b,
                size,
                set_flags,
            } => {
                if let Some(r) = exec_shr(cpu, temps, dst, a, b, size, set_flags) {
                    return r.into();
                }
            }
            IrOp::DoubleShift {
                dst,
                a,
                b,
                count,
                size,
                left,
                set_flags,
            } => {
                if let Some(r) =
                    exec_double_shift(cpu, temps, dst, a, b, count, size, left, set_flags)
                {
                    return r.into();
                }
            }
            IrOp::Sar {
                dst,
                a,
                b,
                size,
                set_flags,
            } => {
                if let Some(r) = exec_sar(cpu, temps, dst, a, b, size, set_flags) {
                    return r.into();
                }
            }
            IrOp::Sext { dst, a, from } => {
                if let Some(r) = exec_sext(temps, dst, a, from) {
                    return r.into();
                }
            }
            IrOp::Bswap { dst, a, size } => {
                if let Some(r) = exec_bswap(temps, dst, a, size) {
                    return r.into();
                }
            }
            IrOp::Rol {
                dst,
                a,
                b,
                size,
                set_flags,
            } => {
                if let Some(r) = exec_rol(cpu, temps, dst, a, b, size, set_flags) {
                    return r.into();
                }
            }
            IrOp::Ror {
                dst,
                a,
                b,
                size,
                set_flags,
            } => {
                if let Some(r) = exec_ror(cpu, temps, dst, a, b, size, set_flags) {
                    return r.into();
                }
            }
            IrOp::Rcl {
                dst,
                a,
                b,
                size,
                set_flags,
            } => {
                if let Some(r) = exec_rcl(cpu, temps, dst, a, b, size, set_flags) {
                    return r.into();
                }
            }
            IrOp::Rcr {
                dst,
                a,
                b,
                size,
                set_flags,
            } => {
                if let Some(r) = exec_rcr(cpu, temps, dst, a, b, size, set_flags) {
                    return r.into();
                }
            }
            IrOp::Mul {
                lo,
                hi,
                a,
                b,
                size,
                signed,
                set_flags,
            } => {
                if let Some(r) = exec_mul(cpu, temps, lo, hi, a, b, size, signed, set_flags) {
                    return r.into();
                }
            }
            IrOp::Div {
                quot,
                rem,
                hi,
                lo,
                divisor,
                size,
                signed,
            } => {
                if CHILD {
                    match divide(
                        read_val(*hi, temps),
                        read_val(*lo, temps),
                        read_val(*divisor, temps),
                        *size,
                        *signed,
                    ) {
                        Some((q, r)) => {
                            temps[*quot as usize] = q;
                            temps[*rem as usize] = r;
                        }
                        None => {
                            mem.0.exception(0, crate::FaultCause::Divide, 0);
                            return WalkResult::Continue;
                        }
                    }
                    continue;
                }
                if let Some(r) = exec_div(
                    cpu,
                    temps,
                    cur_addr.get(),
                    quot,
                    rem,
                    hi,
                    lo,
                    divisor,
                    size,
                    signed,
                ) {
                    return r.into();
                }
            }
            IrOp::GetCond { dst, cond } => {
                if let Some(r) = exec_get_cond(cpu, temps, dst, cond) {
                    return r.into();
                }
            }
            IrOp::Fence { kind } => {
                use std::sync::atomic::{fence, Ordering};
                let order = match kind {
                    FenceKind::Full => Ordering::SeqCst,
                    FenceKind::Load => Ordering::Acquire,
                    FenceKind::Store => Ordering::Release,
                };
                fence(order);
                #[cfg(test)]
                FENCE_ORDERS.with(|orders| {
                    let mut values = orders.get();
                    values[match kind {
                        FenceKind::Full => 0,
                        FenceKind::Load => 1,
                        FenceKind::Store => 2,
                    }] = Some(order);
                    orders.set(values);
                });
            }
            IrOp::Load {
                fault,
                dst,
                addr,
                size,
            } => {
                if let Some(r) = exec_load(cpu, mem, temps, cur_addr.get(), dst, addr, size) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::Store {
                fault,
                addr,
                src,
                size,
                ..
            } => {
                // SDM Vol 3A §5.6 permits reads from write-protected pages;
                // §7.5 restores pre-instruction state on faults. Expose entry
                // flags during the write, including a host fault inside mem.write.
                let result = cpu.flags;
                cpu.flags = entry_flags;
                if let Some(r) = exec_store(cpu, mem, temps, cur_addr.get(), addr, src, size) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
                cpu.flags = result;
            }
            IrOp::AtomicRmw {
                fault,
                old,
                addr,
                src,
                size,
                op: rmw,
            } => {
                if CHILD {
                    mem.0.child_atomic(op, temps, cpu);
                } else if let Some(r) =
                    exec_atomic_rmw(cpu, mem, temps, cur_addr.get(), old, addr, src, size, rmw)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::AtomicCas {
                fault,
                old,
                addr,
                expected,
                src,
                size,
            } => {
                if CHILD {
                    mem.0.child_atomic(op, temps, cpu);
                } else if let Some(r) = exec_atomic_cas(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    old,
                    addr,
                    expected,
                    src,
                    size,
                ) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::AtomicCasPair {
                fault,
                addr,
                size,
                len,
            } => {
                if CHILD {
                    mem.0.child_atomic(op, temps, cpu);
                } else {
                    let (mem, _ordering) = mem.helper(Some(ordering::Direction::ReadWrite));
                    let addr = read_val(*addr, temps);
                    if let Some(exit) =
                        cmpxchg_pair_mode(cpu, mem, cur_addr.get(), addr, *size, *len)
                    {
                        return walk_classify_access(
                            InterpResult::exit(exit),
                            cur_addr.get(),
                            addr,
                            *fault,
                        );
                    }
                }
            }
            IrOp::Bt {
                result,
                a,
                bit,
                size,
                op,
            } => {
                if let Some(r) = exec_bt(cpu, temps, result, a, bit, size, op) {
                    return r.into();
                }
            }
            IrOp::Cpuid => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                if let Some(r) = exec_cpuid(cpu) {
                    return r.into();
                }
            }
            IrOp::ReadTsc { aux } => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                tsc_run(cpu, *aux)
            }
            IrOp::Random {
                dst,
                kind,
                size,
                set_flags,
            } => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                temps[*dst as usize] = random_run(cpu, *kind, *size, *set_flags);
            }
            IrOp::Xgetbv => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                if let Some(r) = exec_xgetbv(cpu, cur_addr.get()) {
                    return r.into();
                }
            }
            IrOp::X87 {
                fault, addr, kind, ..
            } => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                let (mem, _ordering) =
                    mem.helper(if crate::x87::mem_write_bytes(*kind).is_some() {
                        Some(ordering::Direction::Write)
                    } else if crate::x87::mem_operand_bytes(*kind) != 0 {
                        Some(ordering::Direction::Read)
                    } else {
                        None
                    });
                if let Some(r) = exec_x87(cpu, mem, temps, cur_addr.get(), op) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::GetMxcsr { dst } => temps[*dst as usize] = cpu.mxcsr as u64,
            IrOp::SetMxcsr { src } => {
                let mxcsr = read_val(*src, temps) as u32;
                if mxcsr & !crate::state::MXCSR_MASK != 0 {
                    if CHILD {
                        mem.0.exception(13, crate::FaultCause::InvalidMxcsr, 0);
                        return WalkResult::Continue;
                    }
                    cpu.rip = cur_addr.get();
                    return InterpResult::exit(crate::fault::exception_exit(
                        cur_addr.get(),
                        13,
                        crate::FaultCause::InvalidMxcsr,
                    ))
                    .into();
                }
                cpu.mxcsr = mxcsr;
                cpu.rip = ir.next_ip();
                return WalkResult::ModeChange;
            }
            IrOp::XState {
                fault,
                addr,
                kind,
                wide,
            } => {
                if CHILD {
                    unreachable!("XSAVE family refused before child execution");
                }
                let (mem, _ordering) = mem.helper(Some(match kind {
                    crate::x87::xstate::XStateKind::Restore => ordering::Direction::Read,
                    crate::x87::xstate::XStateKind::SaveCompact => ordering::Direction::Write,
                    _ => ordering::Direction::ReadWrite,
                }));
                let a = read_val(*addr, temps);
                if let Err(error) =
                    crate::x87::xstate::exec_xstate_access(cpu, mem, a, *kind, *wide, *fault)
                {
                    cpu.rip = cur_addr.get();
                    let r = InterpResult::exit(xstate_exit(cur_addr.get(), error));
                    return walk_classify_access(r, cur_addr.get(), a, *fault);
                }
                if *kind == crate::x87::xstate::XStateKind::Restore {
                    cpu.rip = ir.next_ip();
                    return WalkResult::ModeChange;
                }
            }
            IrOp::FxState {
                fault,
                addr,
                restore,
                wide,
            } => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                let (mem, _ordering) = mem.helper(Some(if *restore {
                    ordering::Direction::Read
                } else {
                    ordering::Direction::Write
                }));
                if let Some(r) = exec_fx_state(cpu, mem, temps, cur_addr.get(), addr, restore, wide)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
                if *restore {
                    cpu.rip = ir.next_ip();
                    return WalkResult::ModeChange;
                }
            }
            IrOp::Popcnt { dst, src, size } => {
                if let Some(r) = exec_popcnt(cpu, temps, dst, src, size) {
                    return r.into();
                }
            }
            IrOp::Crc32 {
                dst,
                crc,
                src,
                bytes,
            } => {
                if let Some(r) = exec_crc32(temps, dst, crc, src, bytes) {
                    return r.into();
                }
            }
            IrOp::Bmi {
                dst,
                a,
                b,
                size,
                op,
            } => {
                if let Some(r) = exec_bmi(cpu, temps, dst, a, b, size, op) {
                    return r.into();
                }
            }
            IrOp::BitScan {
                dst,
                src,
                old,
                size,
                op,
            } => {
                if let Some(r) = exec_bit_scan(cpu, temps, dst, src, old, size, op) {
                    return r.into();
                }
            }
            IrOp::VLoad {
                fault,
                dst,
                addr,
                size,
            } => {
                if let Some(r) = exec_v_load(cpu, mem, temps, cur_addr.get(), dst, addr, size) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VStore {
                fault,
                addr,
                src,
                size,
            } => {
                if let Some(r) = exec_v_store(cpu, mem, temps, cur_addr.get(), addr, src, size) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VMov { dst, src } => {
                if let Some(r) = exec_v_mov(cpu, dst, src) {
                    return r.into();
                }
            }
            IrOp::VMov256 { dst, src } => {
                cpu.xmm[*dst as usize] = cpu.xmm[*src as usize];
                cpu.ymm_hi[*dst as usize] = cpu.ymm_hi[*src as usize];
            }
            IrOp::VLoadWide {
                fault,
                dst,
                addr,
                bytes,
            } => {
                if let Some(r) = exec_v_load_wide(cpu, mem, temps, cur_addr.get(), dst, addr, bytes)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VStoreWide {
                fault,
                addr,
                src,
                bytes,
            } => {
                if let Some(r) =
                    exec_v_store_wide(cpu, mem, temps, cur_addr.get(), addr, src, bytes)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VMovWide { dst, src, bytes } => {
                if let Some(r) = exec_v_mov_wide(cpu, dst, src, bytes) {
                    return r.into();
                }
            }
            IrOp::VMaskMov {
                dst,
                src,
                k,
                elem,
                zeroing,
                bytes,
            } => {
                if let Some(r) = exec_v_mask_mov(cpu, dst, src, k, elem, zeroing, bytes) {
                    return r.into();
                }
            }
            IrOp::VMaskLoadMem {
                broadcast,
                fault,
                dst,
                addr,
                k,
                elem,
                zeroing,
                bytes,
            } => {
                let (mem, _ordering) = mem.helper(Some(ordering::Direction::Read));
                if let Some(r) = exec_v_mask_load_mem(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    addr,
                    k,
                    elem,
                    zeroing,
                    bytes,
                    broadcast,
                ) {
                    return walk_classify_masked_access(
                        r,
                        cur_addr.get(),
                        read_val(*addr, temps),
                        *fault,
                        broadcast.unwrap_or(*elem),
                        fault_k_mask(cpu, *k),
                    );
                }
            }
            IrOp::VMaskStoreMem {
                fault,
                src,
                addr,
                k,
                elem,
                bytes,
            } => {
                let (mem, _ordering) = mem.helper(Some(ordering::Direction::Write));
                if let Some(r) = exec_v_mask_store_mem(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    src,
                    addr,
                    k,
                    elem,
                    bytes,
                ) {
                    return walk_classify_masked_access(
                        r,
                        cur_addr.get(),
                        read_val(*addr, temps),
                        *fault,
                        *elem,
                        fault_k_mask(cpu, *k),
                    );
                }
            }
            IrOp::VGather {
                fault,
                dst,
                mask,
                index,
                base,
                segment,
                elem,
                index_elem,
                bytes,
                scale,
                addr32,
            } => {
                let (mem, _ordering) = mem.helper(Some(ordering::Direction::Read));
                if let Some((f, ea)) = gather_run(
                    cpu,
                    mem,
                    *dst,
                    *mask,
                    *index,
                    read_val(*base, temps),
                    read_val(*segment, temps),
                    *elem,
                    *index_elem,
                    *bytes,
                    *scale,
                    *addr32,
                    cur_addr.get(),
                ) {
                    return walk_classify_access(
                        InterpResult::exit(str_fault_exit(f)),
                        cur_addr.get(),
                        ea,
                        *fault,
                    );
                }
            }
            IrOp::VVecMaskLoadMem {
                fault,
                dst,
                addr,
                mask,
                elem,
                bytes,
            } => {
                let (mem, _ordering) = mem.helper(Some(ordering::Direction::Read));
                if let Some(r) = exec_v_vecmask_load_mem(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    addr,
                    mask,
                    elem,
                    bytes,
                ) {
                    return walk_classify_masked_access(
                        r,
                        cur_addr.get(),
                        read_val(*addr, temps),
                        *fault,
                        *elem,
                        fault_vec_mask(cpu, *mask, *elem, *bytes),
                    );
                }
            }
            IrOp::VVecMaskStoreMem {
                fault,
                src,
                addr,
                mask,
                elem,
                bytes,
            } => {
                let (mem, _ordering) = mem.helper(Some(ordering::Direction::Write));
                if let Some(r) = exec_v_vecmask_store_mem(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    src,
                    addr,
                    mask,
                    elem,
                    bytes,
                ) {
                    return walk_classify_masked_access(
                        r,
                        cur_addr.get(),
                        read_val(*addr, temps),
                        *fault,
                        *elem,
                        fault_vec_mask(cpu, *mask, *elem, *bytes),
                    );
                }
            }
            IrOp::VLogic256 { dst, a, b, op } => {
                if let Some(r) = exec_v_logic256(cpu, dst, a, b, op) {
                    return r.into();
                }
            }
            IrOp::VLogicWide {
                dst,
                a,
                b,
                op,
                bytes,
            } => {
                if let Some(r) = exec_v_logic_wide(cpu, dst, a, b, op, bytes) {
                    return r.into();
                }
            }
            IrOp::VLogicWideM {
                fault,
                dst,
                a,
                addr,
                op,
                bytes,
            } => {
                if let Some(r) =
                    exec_v_logic_wide_m(cpu, mem, temps, cur_addr.get(), dst, a, addr, op, bytes)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VPopcnt {
                dst,
                a,
                lane,
                bytes,
            } => {
                if let Some(r) = exec_v_popcnt(cpu, dst, a, lane, bytes) {
                    return r.into();
                }
            }
            IrOp::VPopcntM {
                fault,
                dst,
                addr,
                lane,
                bytes,
            } => {
                if let Some(r) =
                    exec_v_popcnt_m(cpu, mem, temps, cur_addr.get(), dst, addr, lane, bytes)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VPMovExtend {
                dst,
                src,
                from,
                to,
                signed,
            } => {
                if let Some(r) = exec_v_p_mov_extend(cpu, dst, src, from, to, signed) {
                    return r.into();
                }
            }
            IrOp::VPMovExtendM {
                fault,
                dst,
                addr,
                from,
                to,
                signed,
            } => {
                if let Some(r) = exec_v_p_mov_extend_m(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    addr,
                    from,
                    to,
                    signed,
                ) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VPMovExtendWide {
                dst,
                src,
                from,
                to,
                signed,
                dst_width,
                writemask,
                zeroing,
            } => {
                if let Some(r) = exec_v_p_mov_extend_wide(
                    cpu, dst, src, from, to, signed, dst_width, writemask, zeroing,
                ) {
                    return r.into();
                }
            }
            IrOp::VPAbs {
                dst,
                src,
                elem,
                dst_width,
                writemask,
                zeroing,
            } => {
                if let Some(r) = exec_v_p_abs(cpu, dst, src, elem, dst_width, writemask, zeroing) {
                    return r.into();
                }
            }
            IrOp::VpUnaryLane {
                dst,
                src,
                op,
                imm,
                elem,
                dst_width,
                writemask,
                zeroing,
            } => {
                if let Some(r) =
                    exec_v_p_unary_lane(cpu, dst, src, op, imm, elem, dst_width, writemask, zeroing)
                {
                    return r.into();
                }
            }
            IrOp::VpBlendm {
                dst,
                a,
                b,
                k,
                elem,
                dst_width,
                zeroing,
            } => {
                if let Some(r) = exec_v_p_blendm(cpu, dst, a, b, k, elem, dst_width, zeroing) {
                    return r.into();
                }
            }
            IrOp::VShuffLane {
                dst,
                a,
                b,
                imm,
                elem,
                dst_width,
                writemask,
                zeroing,
            } => {
                if let Some(r) =
                    exec_v_shuf_lane(cpu, dst, a, b, imm, elem, dst_width, writemask, zeroing)
                {
                    return r.into();
                }
            }
            IrOp::VpMultishift {
                dst,
                ctrl,
                data,
                dst_width,
                writemask,
                zeroing,
            } => {
                if let Some(r) =
                    exec_v_p_multishift(cpu, dst, ctrl, data, dst_width, writemask, zeroing)
                {
                    return r.into();
                }
            }
            IrOp::VPBlendV { dst, src, lane } => {
                if let Some(r) = exec_v_p_blend_v(cpu, dst, src, lane) {
                    return r.into();
                }
            }
            IrOp::VPBlendVM {
                fault,
                dst,
                addr,
                lane,
            } => {
                if let Some(r) =
                    exec_v_p_blend_v_m(cpu, mem, temps, cur_addr.get(), dst, addr, lane)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VPBlendVX {
                dst,
                a,
                b,
                mask,
                lane,
                bytes,
            } => {
                if let Some(r) = exec_v_p_blend_v_x(cpu, dst, a, b, mask, lane, bytes) {
                    return r.into();
                }
            }
            IrOp::VPBlendVXM {
                fault,
                dst,
                a,
                addr,
                mask,
                lane,
                bytes,
            } => {
                if let Some(r) = exec_v_p_blend_v_xm(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    a,
                    addr,
                    mask,
                    lane,
                    bytes,
                ) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VBlendI {
                dst,
                a,
                b,
                imm,
                lane,
                bytes,
            } => {
                if let Some(r) = exec_v_blend_i(cpu, dst, a, b, imm, lane, bytes) {
                    return r.into();
                }
            }
            IrOp::VBlendIM {
                fault,
                dst,
                a,
                addr,
                imm,
                lane,
                bytes,
            } => {
                if let Some(r) = exec_v_blend_i_m(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    a,
                    addr,
                    imm,
                    lane,
                    bytes,
                ) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VPRound {
                dst,
                a,
                src,
                prec,
                mode,
                scalar,
                bytes,
            } => {
                if let Some(r) =
                    exec_v_p_round::<DEFAULT>(cpu, dst, a, src, prec, mode, scalar, bytes)
                {
                    return r.into();
                }
            }
            IrOp::VPRoundM {
                fault,
                dst,
                addr,
                prec,
                mode,
                scalar,
                bytes,
            } => {
                if let Some(r) = exec_v_p_round_m::<DEFAULT, ROUTED, ORDER>(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    addr,
                    prec,
                    mode,
                    scalar,
                    bytes,
                ) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VMaskedLogic {
                dst,
                a,
                b,
                op,
                k,
                elem,
                zeroing,
                bytes,
            } => {
                if let Some(r) = exec_v_masked_logic(cpu, dst, a, b, op, k, elem, zeroing, bytes) {
                    return r.into();
                }
            }
            IrOp::VMaskedPacked {
                dst,
                a,
                b,
                op,
                k,
                elem,
                zeroing,
                bytes,
            } => {
                if let Some(r) = exec_v_masked_packed(cpu, dst, a, b, op, k, elem, zeroing, bytes) {
                    return r.into();
                }
            }
            IrOp::VInsertLaneWide {
                dst,
                src,
                ins,
                idx,
                num_lanes,
                bytes,
            } => {
                if let Some(r) = exec_v_insert_lane_wide(cpu, dst, src, ins, idx, num_lanes, bytes)
                {
                    return r.into();
                }
            }
            IrOp::VExtractLaneWide {
                dst,
                src,
                idx,
                num_lanes,
            } => {
                if let Some(r) = exec_v_extract_lane_wide(cpu, dst, src, idx, num_lanes) {
                    return r.into();
                }
            }
            IrOp::VExtractLaneWideM {
                fault,
                src,
                addr,
                idx,
                num_lanes,
            } => {
                if let Some(r) = exec_v_extract_lane_wide_m(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    src,
                    addr,
                    idx,
                    num_lanes,
                ) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VPcmpStr {
                a,
                b,
                imm,
                explicit,
            } => {
                if let Some(r) = exec_v_pcmp_str(cpu, a, b, imm, explicit) {
                    return r.into();
                }
            }
            IrOp::VPcmpStrM {
                fault,
                a,
                addr,
                imm,
                explicit,
            } => {
                if let Some(r) =
                    exec_v_pcmp_str_m(cpu, mem, temps, cur_addr.get(), a, addr, imm, explicit)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VPcmpStrMask {
                a,
                b,
                imm,
                explicit,
            } => {
                if let Some(r) = exec_v_pcmp_str_mask(cpu, a, b, imm, explicit) {
                    return r.into();
                }
            }
            IrOp::VPcmpStrMaskM {
                fault,
                a,
                addr,
                imm,
                explicit,
            } => {
                if let Some(r) =
                    exec_v_pcmp_str_mask_m(cpu, mem, temps, cur_addr.get(), a, addr, imm, explicit)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VInsertPs { dst, src, imm } => {
                if let Some(r) = exec_v_insert_ps(cpu, dst, src, imm) {
                    return r.into();
                }
            }
            IrOp::VInsertPsM {
                fault,
                dst,
                addr,
                imm,
            } => {
                if let Some(r) = exec_v_insert_ps_m(cpu, mem, temps, cur_addr.get(), dst, addr, imm)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VInsertPs3 { dst, a, src, imm } => {
                if let Some(r) = exec_v_insert_ps3(cpu, dst, a, src, imm) {
                    return r.into();
                }
            }
            IrOp::VInsertPsM3 {
                fault,
                dst,
                a,
                addr,
                imm,
            } => {
                if let Some(r) =
                    exec_v_insert_ps_m3(cpu, mem, temps, cur_addr.get(), dst, a, addr, imm)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VDpps {
                dst,
                a,
                b,
                imm,
                bytes,
            } => {
                if let Some(r) = exec_v_dpps::<DEFAULT>(cpu, dst, a, b, imm, bytes) {
                    return r.into();
                }
            }
            IrOp::VDppsM {
                fault,
                dst,
                a,
                addr,
                imm,
                bytes,
            } => {
                if let Some(r) = exec_v_dpps_m::<DEFAULT, ROUTED, ORDER>(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    a,
                    addr,
                    imm,
                    bytes,
                ) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VDppd { dst, b, imm } => {
                if let Some(r) = exec_v_dppd::<DEFAULT>(cpu, dst, b, imm) {
                    return r.into();
                }
            }
            IrOp::VDppdM {
                fault,
                dst,
                addr,
                imm,
            } => {
                if let Some(r) = exec_v_dppd_m::<DEFAULT, ROUTED, ORDER>(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    addr,
                    imm,
                ) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VDp3 {
                dst,
                a,
                b,
                imm,
                prec,
            } => {
                if let Some(r) = exec_v_dp3::<DEFAULT>(cpu, dst, a, b, imm, prec) {
                    return r.into();
                }
            }
            IrOp::VDp3M {
                fault,
                dst,
                a,
                addr,
                imm,
                prec,
            } => {
                if let Some(r) = exec_v_dp3_m::<DEFAULT, ROUTED, ORDER>(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    a,
                    addr,
                    imm,
                    prec,
                ) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VAlign {
                dst,
                a,
                b,
                shift,
                elem,
                bytes,
            } => {
                if let Some(r) = exec_v_align(cpu, dst, a, b, shift, elem, bytes) {
                    return r.into();
                }
            }
            IrOp::VPTernlog {
                dst,
                b,
                c,
                imm,
                bytes,
            } => {
                if let Some(r) = exec_v_p_ternlog(cpu, dst, b, c, imm, bytes) {
                    return r.into();
                }
            }
            IrOp::VPTernlogM {
                fault,
                dst,
                b,
                addr,
                imm,
                bytes,
            } => {
                if let Some(r) =
                    exec_v_p_ternlog_m(cpu, mem, temps, cur_addr.get(), dst, b, addr, imm, bytes)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VLogic256M {
                fault,
                dst,
                a,
                addr,
                op,
            } => {
                if let Some(r) =
                    exec_v_logic256_m(cpu, mem, temps, cur_addr.get(), dst, a, addr, op)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VPackedBin256 {
                dst,
                a,
                b,
                lane,
                op,
            } => {
                if let Some(r) = exec_v_packed_bin256(cpu, dst, a, b, lane, op) {
                    return r.into();
                }
            }
            IrOp::VPackedBin256M {
                fault,
                dst,
                a,
                addr,
                lane,
                op,
            } => {
                if let Some(r) =
                    exec_v_packed_bin256_m(cpu, mem, temps, cur_addr.get(), dst, a, addr, lane, op)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VPackedWide {
                dst,
                a,
                b,
                lane,
                op,
                bytes,
            } => {
                if let Some(r) = exec_v_packed_wide(cpu, dst, a, b, lane, op, bytes) {
                    return r.into();
                }
            }
            IrOp::VPackedWideM {
                fault,
                dst,
                a,
                addr,
                lane,
                op,
                bytes,
            } => {
                if let Some(r) = exec_v_packed_wide_m(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    a,
                    addr,
                    lane,
                    op,
                    bytes,
                ) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VMoveMaskB256 { dst, src } => {
                if let Some(r) = exec_v_move_mask_b256(cpu, temps, dst, src) {
                    return r.into();
                }
            }
            IrOp::VFromGpr { dst, src, size } => {
                if let Some(r) = exec_v_from_gpr(cpu, temps, dst, src, size) {
                    return r.into();
                }
            }
            IrOp::VReadChunk { dst, src, chunk } => {
                temps[*dst as usize] = (cpu.vec_lanes(*src as usize)[*chunk as usize / 2]
                    >> ((*chunk % 2) * 64)) as u64;
            }
            IrOp::VWriteChunk { dst, src, chunk } => {
                let mut lanes = cpu.vec_lanes(*dst as usize);
                let shift = (*chunk % 2) * 64;
                let mask = (u64::MAX as u128) << shift;
                lanes[*chunk as usize / 2] = (lanes[*chunk as usize / 2] & !mask)
                    | ((read_val(*src, temps) as u128) << shift);
                cpu.set_vec(*dst as usize, lanes, 64);
            }
            IrOp::VToGpr { dst, src, size } => {
                if let Some(r) = exec_v_to_gpr(cpu, temps, dst, src, size) {
                    return r.into();
                }
            }
            IrOp::VLogic { dst, a, b, op } => {
                if let Some(r) = exec_v_logic(cpu, dst, a, b, op) {
                    return r.into();
                }
            }
            IrOp::VPackedBin {
                dst,
                a,
                b,
                lane,
                op,
            } => {
                if let Some(r) = exec_v_packed_bin(cpu, dst, a, b, lane, op) {
                    return r.into();
                }
            }
            IrOp::VPackedBinM {
                fault,
                dst,
                addr,
                lane,
                op,
            } => {
                if let Some(r) =
                    exec_v_packed_bin_m(cpu, mem, temps, cur_addr.get(), dst, addr, lane, op)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VLogicM {
                fault,
                dst,
                addr,
                op,
            } => {
                if let Some(r) = exec_v_logic_m(cpu, mem, temps, cur_addr.get(), dst, addr, op) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VPackedShift {
                dst,
                a,
                imm,
                lane,
                right,
                arith,
            } => {
                if let Some(r) = exec_v_packed_shift(cpu, dst, a, imm, lane, right, arith) {
                    return r.into();
                }
            }
            IrOp::VMaskedShift {
                dst,
                a,
                imm,
                elem,
                right,
                arith,
                k,
                zeroing,
                bytes,
            } => {
                if let Some(r) =
                    exec_v_masked_shift(cpu, dst, a, imm, elem, right, arith, k, zeroing, bytes)
                {
                    return r.into();
                }
            }
            IrOp::VShiftReg {
                dst,
                a,
                count,
                elem,
                right,
                arith,
                k,
                zeroing,
                bytes,
            } => {
                exec_shift_reg(
                    cpu, *dst, *a, *count, *elem, *right, *arith, *k, *zeroing, *bytes,
                );
            }
            IrOp::VShiftVar {
                dst,
                a,
                count,
                elem,
                right,
                arith,
                k,
                zeroing,
                bytes,
            } => {
                exec_var_shift(
                    cpu, *dst, *a, *count, *elem, *right, *arith, *k, *zeroing, *bytes,
                );
            }
            IrOp::VGf2p8 {
                dst,
                a,
                b,
                imm,
                mode,
                k,
                zeroing,
                bytes,
            } => {
                exec_gf2p8(cpu, *dst, *a, *b, *imm, *mode, *k, *zeroing, *bytes);
            }
            IrOp::VGf2p8M {
                fault,
                dst,
                a,
                addr,
                imm,
                mode,
                k,
                zeroing,
                bytes,
            } => {
                let (mem, _ordering) = mem.helper(Some(ordering::Direction::Read));
                if let Some(r) = exec_v_gf2p8_m(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    a,
                    addr,
                    imm,
                    mode,
                    k,
                    zeroing,
                    bytes,
                ) {
                    if *k != 0
                        && matches!(crate::ir::GfniOp::from_u8(*mode), crate::ir::GfniOp::Mulb)
                    {
                        return walk_classify_masked_access(
                            r,
                            cur_addr.get(),
                            read_val(*addr, temps),
                            *fault,
                            1,
                            fault_k_mask(cpu, *k),
                        );
                    }
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VByteShift {
                dst,
                a,
                shift,
                right,
                width,
            } => {
                if let Some(r) = exec_v_byte_shift(cpu, dst, a, shift, right, width) {
                    return r.into();
                }
            }
            IrOp::VShuffle32 { dst, a, imm, bytes } => {
                if let Some(r) = exec_v_shuffle32(cpu, dst, a, imm, bytes) {
                    return r.into();
                }
            }
            IrOp::VShuffle16M {
                fault,
                dst,
                addr,
                imm,
                high,
            } => {
                let ea = read_val(*addr, temps);
                let bv = match vload(cpu, mem, cur_addr.get(), ea, 16) {
                    Ok(v) => v,
                    Err(f) => {
                        return walk_classify_access(
                            vec_trap_out(cpu, cur_addr.get(), f, AccessKind::Read, ea, 16),
                            cur_addr.get(),
                            read_val(*addr, temps),
                            *fault,
                        )
                    }
                };
                cpu.xmm[*dst as usize] = shuffle16_128(bv, *imm, *high);
            }
            IrOp::VShiftRegM {
                fault,
                dst,
                a,
                bytes,
                addr,
                elem,
                right,
                arith,
            } => {
                let ea = read_val(*addr, temps);
                let bv = match vload(cpu, mem, cur_addr.get(), ea, 16) {
                    Ok(v) => v,
                    Err(f) => {
                        return walk_classify_access(
                            vec_trap_out(cpu, cur_addr.get(), f, AccessKind::Read, ea, 16),
                            cur_addr.get(),
                            read_val(*addr, temps),
                            *fault,
                        )
                    }
                };
                let count = (bv as u64).min(u64::from(*elem) * 8) as u8;
                let av = cpu.vec_lanes(*a as usize);
                let mut r = [0; 4];
                for i in 0..usize::from(*bytes) / 16 {
                    r[i] = packed_shift(av[i], count, *elem, *right, *arith);
                }
                if *bytes == 16 {
                    cpu.set_vec_low(*dst as usize, r, *bytes);
                } else {
                    cpu.set_vec(*dst as usize, r, *bytes);
                }
            }
            IrOp::VPtestM {
                fault,
                a,
                addr,
                bytes,
                elem,
            } => {
                let ea = read_val(*addr, temps);
                let bv = match vload_lanes(cpu, mem, cur_addr.get(), ea, *bytes) {
                    Ok(v) => v,
                    Err(f) => {
                        return walk_classify_access(
                            vec_trap_out(cpu, cur_addr.get(), f, AccessKind::Read, ea, *bytes),
                            cur_addr.get(),
                            ea,
                            *fault,
                        )
                    }
                };
                let av = cpu.vec_lanes(*a as usize);
                let mask = if *elem == 0 {
                    u128::MAX
                } else {
                    fp_sign_mask(*elem)
                };
                let (ahi, bhi) = if *bytes == 32 {
                    (av[1] & mask, bv[1] & mask)
                } else {
                    (0, 0)
                };
                ptest_value(cpu, av[0] & mask, bv[0] & mask, ahi, bhi);
            }
            IrOp::VBlendWM {
                fault,
                dst,
                a,
                addr,
                imm,
                bytes,
            } => {
                let ea = read_val(*addr, temps);
                let bv = match vload_lanes(cpu, mem, cur_addr.get(), ea, *bytes) {
                    Ok(v) => v,
                    Err(f) => {
                        return walk_classify_access(
                            vec_trap_out(cpu, cur_addr.get(), f, AccessKind::Read, ea, *bytes),
                            cur_addr.get(),
                            read_val(*addr, temps),
                            *fault,
                        )
                    }
                };
                blend_w_value(cpu, *dst, cpu.vec_lanes(*a as usize), bv, *imm, *bytes);
            }
            IrOp::VMpsadbwM {
                fault,
                dst,
                a,
                addr,
                imm,
                bytes,
            } => {
                let ea = read_val(*addr, temps);
                let bv = match vload_lanes(cpu, mem, cur_addr.get(), ea, *bytes) {
                    Ok(v) => v,
                    Err(f) => {
                        return walk_classify_access(
                            vec_trap_out(cpu, cur_addr.get(), f, AccessKind::Read, ea, *bytes),
                            cur_addr.get(),
                            read_val(*addr, temps),
                            *fault,
                        )
                    }
                };
                let av = cpu.vec_lanes(*a as usize);
                cpu.xmm[*dst as usize] = mpsadbw(av[0], bv[0], *imm);
                if *bytes == 32 {
                    cpu.ymm_hi[*dst as usize] = mpsadbw(av[1], bv[1], (*imm >> 3) & 7);
                }
            }
            IrOp::VBlendW {
                dst,
                a,
                b,
                imm,
                bytes,
            } => {
                if let Some(r) = exec_v_blend_w(cpu, dst, a, b, imm, bytes) {
                    return r.into();
                }
            }
            IrOp::VBlendD {
                dst,
                a,
                b,
                imm,
                bytes,
            } => {
                if let Some(r) = exec_v_blend_d(cpu, dst, a, b, imm, bytes) {
                    return r.into();
                }
            }
            IrOp::VFma {
                dst,
                x,
                y,
                z,
                prec,
                scalar,
                neg_prod,
                neg_add,
                bytes,
                alt_sign,
                writemask,
                zeroing,
            } => {
                if let Some(r) = exec_v_fma::<DEFAULT>(
                    cpu, dst, x, y, z, prec, scalar, neg_prod, neg_add, bytes, alt_sign, writemask,
                    zeroing,
                ) {
                    return r.into();
                }
            }
            IrOp::VFmaM {
                fault,
                dst,
                x,
                y,
                z,
                addr,
                mem_role,
                prec,
                scalar,
                neg_prod,
                neg_add,
                bytes,
                alt_sign,
                writemask,
                zeroing,
            } => {
                let (mem, _ordering) = mem.helper(Some(ordering::Direction::Read));
                if let Some(r) = exec_v_fma_m::<DEFAULT, ROUTED>(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    x,
                    y,
                    z,
                    addr,
                    mem_role,
                    prec,
                    scalar,
                    neg_prod,
                    neg_add,
                    bytes,
                    alt_sign,
                    writemask,
                    zeroing,
                ) {
                    if let Some(k) = writemask {
                        return walk_classify_masked_access(
                            r,
                            cur_addr.get(),
                            read_val(*addr, temps),
                            *fault,
                            prec.bytes(),
                            fault_k_mask(cpu, *k),
                        );
                    }
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VPackWide {
                dst,
                a,
                b,
                from_elem,
                signed,
                bytes,
            } => {
                if let Some(r) = exec_v_pack_wide(cpu, dst, a, b, from_elem, signed, bytes) {
                    return r.into();
                }
            }
            IrOp::VPackWideM {
                fault,
                dst,
                a,
                addr,
                from_elem,
                signed,
                bytes,
            } => {
                let ea = read_val(*addr, temps);
                let bv = match vload_lanes(cpu, mem, cur_addr.get(), ea, *bytes) {
                    Ok(v) => v,
                    Err(f) => {
                        return walk_classify_access(
                            vec_trap_out(cpu, cur_addr.get(), f, AccessKind::Read, ea, *bytes),
                            cur_addr.get(),
                            ea,
                            *fault,
                        )
                    }
                };
                let av = cpu.vec_lanes(*a as usize);
                let mut r = [0; 4];
                for i in 0..usize::from(*bytes) / 16 {
                    r[i] = pack_lane(av[i], bv[i], *from_elem, *signed);
                }
                cpu.set_vec_low(*dst as usize, r, *bytes);
            }
            IrOp::VShuffle32Wide {
                dst,
                a,
                imm,
                bytes,
                writemask,
                zeroing,
            } => {
                if let Some(r) = exec_v_shuffle32_wide(cpu, dst, a, imm, bytes, writemask, zeroing)
                {
                    return r.into();
                }
            }
            IrOp::VMoveHalf {
                dst,
                src,
                dst_high,
                src_high,
            } => {
                if let Some(r) = exec_v_move_half(cpu, dst, src, dst_high, src_high) {
                    return r.into();
                }
            }
            IrOp::VLoadHalf {
                fault,
                dst,
                addr,
                high,
            } => {
                if let Some(r) = exec_v_load_half(cpu, mem, temps, cur_addr.get(), dst, addr, high)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VStoreHalf {
                fault,
                addr,
                src,
                high,
            } => {
                if let Some(r) = exec_v_store_half(cpu, mem, temps, cur_addr.get(), addr, src, high)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VExtractW { dst, src, index } => {
                if let Some(r) = exec_v_extract_w(cpu, temps, dst, src, index) {
                    return r.into();
                }
            }
            IrOp::VExtractLane {
                dst,
                src,
                index,
                size,
            } => {
                if let Some(r) = exec_v_extract_lane(cpu, temps, dst, src, index, size) {
                    return r.into();
                }
            }
            IrOp::VMoveMaskB { dst, src } => {
                if let Some(r) = exec_v_move_mask_b(cpu, temps, dst, src) {
                    return r.into();
                }
            }
            IrOp::VMoveMaskFp {
                dst,
                src,
                elem,
                bytes,
            } => {
                if let Some(r) = exec_v_move_mask_fp(cpu, temps, dst, src, elem, bytes) {
                    return r.into();
                }
            }
            IrOp::VBroadcast {
                dst,
                src,
                elem,
                w256,
            } => {
                if let Some(r) = exec_v_broadcast(cpu, dst, src, elem, w256) {
                    return r.into();
                }
            }
            IrOp::VBroadcastM {
                fault,
                dst,
                addr,
                elem,
                w256,
            } => {
                if let Some(r) =
                    exec_v_broadcast_m(cpu, mem, temps, cur_addr.get(), dst, addr, elem, w256)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VBroadcastGpr {
                dst,
                src,
                elem,
                width,
            } => {
                if let Some(r) = exec_v_broadcast_gpr(cpu, temps, dst, src, elem, width) {
                    return r.into();
                }
            }
            IrOp::VBroadcastLane {
                dst,
                src,
                chunk,
                elem,
                dst_width,
                writemask,
                zeroing,
            } => {
                if let Some(r) =
                    exec_v_broadcast_lane(cpu, dst, src, chunk, elem, dst_width, writemask, zeroing)
                {
                    return r.into();
                }
            }
            IrOp::VBroadcastLaneM {
                fault,
                dst,
                addr,
                chunk,
                elem,
                dst_width,
                writemask,
                zeroing,
            } => {
                let (mem, _ordering) = mem.helper(Some(ordering::Direction::Read));
                if let Some(r) = exec_v_broadcast_lane_m(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    addr,
                    chunk,
                    elem,
                    dst_width,
                    writemask,
                    zeroing,
                ) {
                    let base = read_val(*addr, temps);
                    if let Some(k) = writemask {
                        let (operand, width) = crate::memory::active_broadcast_span(
                            base,
                            *chunk,
                            *elem,
                            *dst_width,
                            fault_k_mask(cpu, *k),
                        );
                        return walk_classify_access(
                            r,
                            cur_addr.get(),
                            operand,
                            crate::ir::MemoryAccess::new(
                                width as u16,
                                fault.long64(),
                                fault.stack(),
                            ),
                        );
                    }
                    return walk_classify_access(r, cur_addr.get(), base, *fault);
                }
            }
            IrOp::VPCmpToMask {
                k,
                a,
                b,
                elem,
                width,
                pred,
                signed,
                writemask,
            } => {
                if let Some(r) =
                    exec_v_p_cmp_to_mask(cpu, k, a, b, elem, width, pred, signed, writemask)
                {
                    return r.into();
                }
            }
            IrOp::VPCmpToMaskM {
                fault,
                k,
                a,
                addr,
                elem,
                width,
                pred,
                signed,
                writemask,
            } => {
                if let Some(r) = exec_v_p_cmp_to_mask_m(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    k,
                    a,
                    addr,
                    elem,
                    width,
                    pred,
                    signed,
                    writemask,
                ) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VPTestToMask {
                k,
                a,
                b,
                elem,
                width,
                neg,
                writemask,
            } => {
                if let Some(r) = exec_v_p_test_to_mask(cpu, k, a, b, elem, width, neg, writemask) {
                    return r.into();
                }
            }
            IrOp::VPTestToMaskM {
                fault,
                k,
                a,
                addr,
                elem,
                width,
                neg,
                writemask,
            } => {
                if let Some(r) = exec_v_p_test_to_mask_m(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    k,
                    a,
                    addr,
                    elem,
                    width,
                    neg,
                    writemask,
                ) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VKOrTest { a, b, width } => {
                if let Some(r) = exec_v_k_or_test(cpu, a, b, width) {
                    return r.into();
                }
            }
            IrOp::VKFromGpr { k, src, width } => {
                if let Some(r) = exec_v_k_from_gpr(cpu, temps, k, src, width) {
                    return r.into();
                }
            }
            IrOp::VKToGpr { dst, k, width } => {
                if let Some(r) = exec_v_k_to_gpr(cpu, temps, dst, k, width) {
                    return r.into();
                }
            }
            IrOp::VKMovKK { dst, src, width } => {
                if let Some(r) = exec_v_k_mov_k_k(cpu, dst, src, width) {
                    return r.into();
                }
            }
            IrOp::VKUnpack { dst, a, b, half } => {
                if let Some(r) = exec_v_k_unpack(cpu, dst, a, b, half) {
                    return r.into();
                }
            }
            IrOp::VKBinOp {
                dst,
                a,
                b,
                op,
                width,
            } => {
                if let Some(r) = exec_v_k_bin_op(cpu, dst, a, b, op, width) {
                    return r.into();
                }
            }
            IrOp::VKNot { dst, a, width } => {
                if let Some(r) = exec_v_k_not(cpu, dst, a, width) {
                    return r.into();
                }
            }
            IrOp::VKShift {
                dst,
                a,
                amount,
                width,
                left,
            } => {
                if let Some(r) = exec_v_k_shift(cpu, dst, a, amount, width, left) {
                    return r.into();
                }
            }
            IrOp::VPmovNarrow {
                dst,
                src,
                from,
                to,
                src_width,
                writemask,
                zeroing,
            } => {
                if let Some(r) =
                    exec_v_pmov_narrow(cpu, dst, src, from, to, src_width, writemask, zeroing)
                {
                    return r.into();
                }
            }
            IrOp::VPmovNarrowMem {
                fault,
                src,
                addr,
                from,
                to,
                src_width,
            } => {
                let (mem, _ordering) = mem.helper(Some(ordering::Direction::Write));
                if let Some(r) = exec_v_pmov_narrow_mem(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    src,
                    addr,
                    from,
                    to,
                    src_width,
                ) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VPermT2 {
                dst,
                idx,
                tbl,
                elem,
                writemask,
                zeroing,
                bytes,
                imode,
            } => {
                if let Some(r) =
                    exec_v_perm_t2(cpu, dst, idx, tbl, elem, writemask, zeroing, bytes, imode)
                {
                    return r.into();
                }
            }
            IrOp::VPermT2M {
                fault,
                dst,
                idx,
                addr,
                elem,
                writemask,
                zeroing,
                bytes,
                imode,
            } => {
                let (mem, _ordering) = mem.helper(Some(ordering::Direction::Read));
                if let Some(r) = exec_v_perm_t2_m(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    idx,
                    addr,
                    elem,
                    writemask,
                    zeroing,
                    bytes,
                    imode,
                ) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VPerm1 {
                dst,
                idx,
                src,
                elem,
                bytes,
                writemask,
                zeroing,
            } => {
                if let Some(r) = exec_v_perm1(cpu, dst, idx, src, elem, bytes, writemask, zeroing) {
                    return r.into();
                }
            }
            IrOp::VPerm1M {
                fault,
                dst,
                idx,
                addr,
                elem,
                bytes,
                writemask,
                zeroing,
            } => {
                let (mem, _ordering) = mem.helper(Some(ordering::Direction::Read));
                if let Some(r) = exec_v_perm1_m(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    idx,
                    addr,
                    elem,
                    bytes,
                    writemask,
                    zeroing,
                ) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VInsert128 { dst, src, ins, hi } => {
                if let Some(r) = exec_v_insert128(cpu, dst, src, ins, hi) {
                    return r.into();
                }
            }
            IrOp::VInsert128M {
                fault,
                dst,
                src,
                addr,
                hi,
            } => {
                if let Some(r) =
                    exec_v_insert128_m(cpu, mem, temps, cur_addr.get(), dst, src, addr, hi)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VExtract128 { dst, src, hi } => {
                if let Some(r) = exec_v_extract128(cpu, dst, src, hi) {
                    return r.into();
                }
            }
            IrOp::VPshufb256 { dst, a, idx } => {
                if let Some(r) = exec_v_pshufb256(cpu, dst, a, idx) {
                    return r.into();
                }
            }
            IrOp::VPshufbWide {
                dst,
                a,
                idx,
                bytes,
                writemask,
                zeroing,
            } => {
                if let Some(r) = exec_v_pshufb_wide(cpu, dst, a, idx, bytes, writemask, zeroing) {
                    return r.into();
                }
            }
            IrOp::VPshufb256M {
                fault,
                dst,
                a,
                addr,
            } => {
                if let Some(r) = exec_v_pshufb256_m(cpu, mem, temps, cur_addr.get(), dst, a, addr) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VPackedShift256 {
                dst,
                a,
                imm,
                lane,
                right,
                arith,
            } => {
                if let Some(r) = exec_v_packed_shift256(cpu, dst, a, imm, lane, right, arith) {
                    return r.into();
                }
            }
            IrOp::VPermq { dst, src, imm } => {
                if let Some(r) = exec_v_permq(cpu, dst, src, imm) {
                    return r.into();
                }
            }
            IrOp::VPermd { dst, ctrl, src } => {
                if let Some(r) = exec_v_permd(cpu, dst, ctrl, src) {
                    return r.into();
                }
            }
            IrOp::VPermilVar {
                dst,
                src,
                ctrl,
                elem,
                bytes,
            } => {
                if let Some(r) = exec_v_permil_var(cpu, dst, src, ctrl, elem, bytes) {
                    return r.into();
                }
            }
            IrOp::VPerm2i128 { dst, a, b, imm } => {
                if let Some(r) = exec_v_perm2i128(cpu, dst, a, b, imm) {
                    return r.into();
                }
            }
            IrOp::VPalignr256 { dst, a, b, imm } => {
                if let Some(r) = exec_v_palignr256(cpu, dst, a, b, imm) {
                    return r.into();
                }
            }
            IrOp::VPtest { a, b, w256 } => {
                if let Some(r) = exec_v_ptest(cpu, a, b, w256) {
                    return r.into();
                }
            }
            IrOp::VTestFp { a, b, elem, bytes } => {
                if let Some(r) = exec_v_test_fp(cpu, a, b, elem, bytes) {
                    return r.into();
                }
            }
            IrOp::VZeroUpper { reg } => {
                if let Some(r) = exec_v_zero_upper(cpu, reg) {
                    return r.into();
                }
            }
            IrOp::VZeroUpperAll { clear_low } => {
                if let Some(r) = exec_v_zero_upper_all(cpu, *clear_low) {
                    return r.into();
                }
            }
            IrOp::VPshufb { dst, a, idx } => {
                if let Some(r) = exec_v_pshufb(cpu, dst, a, idx) {
                    return r.into();
                }
            }
            IrOp::VPshufbM { fault, dst, addr } => {
                if let Some(r) = exec_v_pshufb_m(cpu, mem, temps, cur_addr.get(), dst, addr) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VAlignr { dst, a, src, imm } => {
                if let Some(r) = exec_v_alignr(cpu, dst, a, src, imm) {
                    return r.into();
                }
            }
            IrOp::VAlignrM {
                fault,
                dst,
                a,
                addr,
                imm,
                bytes,
            } => {
                let ea = read_val(*addr, temps);
                let bv = match vload_lanes(cpu, mem, cur_addr.get(), ea, *bytes) {
                    Ok(v) => v,
                    Err(f) => {
                        return walk_classify_access(
                            vec_trap_out(cpu, cur_addr.get(), f, AccessKind::Read, ea, *bytes),
                            cur_addr.get(),
                            ea,
                            *fault,
                        )
                    }
                };
                let av = cpu.vec_lanes(*a as usize);
                let mut r = [0; 4];
                for i in 0..usize::from(*bytes) / 16 {
                    r[i] = palignr(av[i], bv[i], *imm);
                }
                if *bytes == 16 {
                    cpu.set_vec_low(*dst as usize, r, *bytes);
                } else {
                    cpu.set_vec(*dst as usize, r, *bytes);
                }
            }
            IrOp::VAes { dst, a, b, op } => {
                if let Some(r) = exec_v_aes(cpu, dst, a, b, op) {
                    return r.into();
                }
            }
            IrOp::VAesM {
                fault,
                dst,
                a,
                addr,
                op,
            } => {
                if let Some(r) = exec_v_aes_m(cpu, mem, temps, cur_addr.get(), dst, a, addr, op) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VAesImc { dst, src } => {
                if let Some(r) = exec_v_aes_imc(cpu, dst, src) {
                    return r.into();
                }
            }
            IrOp::VAesImcM { fault, dst, addr } => {
                if let Some(r) = exec_v_aes_imc_m(cpu, mem, temps, cur_addr.get(), dst, addr) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VAesKeygen { dst, src, imm } => {
                if let Some(r) = exec_v_aes_keygen(cpu, dst, src, imm) {
                    return r.into();
                }
            }
            IrOp::VAesKeygenM {
                fault,
                dst,
                addr,
                imm,
            } => {
                if let Some(r) =
                    exec_v_aes_keygen_m(cpu, mem, temps, cur_addr.get(), dst, addr, imm)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VSha { dst, a, b, imm, op } => {
                if let Some(r) = exec_v_sha(cpu, dst, a, b, imm, op) {
                    return r.into();
                }
            }
            IrOp::VShaM {
                fault,
                dst,
                a,
                addr,
                imm,
                op,
            } => {
                if let Some(r) =
                    exec_v_sha_m(cpu, mem, temps, cur_addr.get(), dst, a, addr, imm, op)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VGfni { dst, a, b, imm, op } => {
                if let Some(r) = exec_v_gfni(cpu, dst, a, b, imm, op) {
                    return r.into();
                }
            }
            IrOp::VGfniM {
                fault,
                dst,
                a,
                addr,
                imm,
                op,
            } => {
                if let Some(r) =
                    exec_v_gfni_m(cpu, mem, temps, cur_addr.get(), dst, a, addr, imm, op)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::Movq2dq { dst, src_mm } => exec_movq2dq(cpu, *dst, *src_mm),
            IrOp::Movdq2q { dst_mm, src_xmm } => exec_movdq2q(cpu, *dst_mm, *src_xmm),
            IrOp::VPclmul { dst, a, b, imm } => {
                if let Some(r) = exec_v_pclmul(cpu, dst, a, b, imm) {
                    return r.into();
                }
            }
            IrOp::VPclmulM {
                fault,
                dst,
                a,
                addr,
                imm,
            } => {
                if let Some(r) = exec_v_pclmul_m(cpu, mem, temps, cur_addr.get(), dst, a, addr, imm)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VPsign {
                dst,
                a,
                b,
                lane,
                bytes,
            } => {
                if let Some(r) = exec_v_psign(cpu, dst, a, b, lane, bytes) {
                    return r.into();
                }
            }
            IrOp::VPsignM {
                fault,
                dst,
                a,
                addr,
                lane,
                bytes,
            } => {
                if let Some(r) =
                    exec_v_psign_m(cpu, mem, temps, cur_addr.get(), dst, a, addr, lane, bytes)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VShufps { dst, a, b, imm } => {
                if let Some(r) = exec_v_shufps(cpu, dst, a, b, imm) {
                    return r.into();
                }
            }
            IrOp::VShufpsM {
                fault,
                dst,
                a,
                addr,
                imm,
            } => {
                if let Some(r) = exec_v_shufps_m(cpu, mem, temps, cur_addr.get(), dst, a, addr, imm)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VShuffle16 {
                dst,
                a,
                imm,
                high,
                bytes,
            } => {
                if let Some(r) = exec_v_shuffle16(cpu, dst, a, imm, high, bytes) {
                    return r.into();
                }
            }
            IrOp::VUnpackLow {
                dst,
                a,
                b,
                lane,
                high,
            } => {
                if let Some(r) = exec_v_unpack_low(cpu, dst, a, b, lane, high) {
                    return r.into();
                }
            }
            IrOp::VUnpackLowM {
                fault,
                dst,
                a,
                addr,
                lane,
                high,
            } => {
                if let Some(r) =
                    exec_v_unpack_low_m(cpu, mem, temps, cur_addr.get(), dst, a, addr, lane, high)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VPackUsWB { dst, a, b } => {
                if let Some(r) = exec_v_pack_us_w_b(cpu, dst, a, b) {
                    return r.into();
                }
            }
            IrOp::VPMAddWd { dst, a, b } => {
                exec_pmaddwd(cpu, *dst, *a, *b);
            }
            IrOp::VPMadd {
                dst,
                a,
                b,
                ubsw,
                bytes,
            } => {
                exec_v_pmadd(cpu, *dst, *a, *b, *ubsw, *bytes);
            }
            IrOp::VPMaddM {
                fault,
                dst,
                a,
                addr,
                ubsw,
                bytes,
            } => {
                if let Some(r) =
                    exec_v_pmadd_m(cpu, mem, temps, cur_addr.get(), dst, a, addr, ubsw, bytes)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::SetDf { value } => {
                if let Some(r) = exec_set_df(cpu, value) {
                    return r.into();
                }
            }
            IrOp::RepString {
                fault,
                op,
                elem,
                rep,
                addr_bits,
                seg_base,
            } => {
                if CHILD {
                    mem.0.string::<ROUTED>(
                        cpu,
                        *op,
                        *elem,
                        *rep,
                        *addr_bits,
                        read_val(*seg_base, temps),
                    );
                    if mem.0.stopped() {
                        return WalkResult::Continue;
                    }
                    continue;
                }
                let (mem, _ordering) = mem.helper(Some(match op {
                    StrOp::Ins | StrOp::Outs => unreachable!("port strings use PortString"),
                    StrOp::Movs => ordering::Direction::ReadWrite,
                    StrOp::Stos => ordering::Direction::Write,
                    StrOp::Lods | StrOp::Scas | StrOp::Cmps => ordering::Direction::Read,
                }));
                if STEPPED {
                    let stop = string_run_bounded::<ROUTED, STOP_WHEN_IRQ_READY, _>(
                        cpu,
                        mem,
                        *op,
                        *elem,
                        *rep,
                        cur_addr.get(),
                        *addr_bits,
                        read_val(*seg_base, temps),
                        0,
                        exit_request,
                        budget.get(),
                        shadows,
                        *retired,
                    );
                    if let Some(stop) = stop {
                        return stop;
                    }
                    continue;
                }
                let before = cpu.gpr[1];
                let result = exec_rep_string(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    op,
                    elem,
                    rep,
                    addr_bits,
                    seg_base,
                    exit_request,
                );
                shadows.complete_rep_prefix(before, cpu, *retired);
                if let Some(r) = result {
                    return walk_classify_string_access(r, cur_addr.get(), *fault);
                }
            }
            IrOp::VInsertW { dst, src, index } => {
                if let Some(r) = exec_v_insert_w(cpu, temps, dst, src, index) {
                    return r.into();
                }
            }
            IrOp::VInsertLane {
                dst,
                base,
                src,
                index,
                size,
            } => {
                if let Some(r) = exec_v_insert_lane(cpu, temps, dst, base, src, index, size) {
                    return r.into();
                }
            }
            IrOp::VFloatMov { dst, a, src, prec } => {
                if let Some(r) = exec_v_float_mov(cpu, dst, a, src, prec) {
                    return r.into();
                }
            }
            IrOp::VFloatBin {
                dst,
                a,
                b,
                op,
                prec,
                scalar,
            } => {
                if let Some(r) = exec_v_float_bin::<DEFAULT>(cpu, dst, a, b, op, prec, scalar) {
                    return r.into();
                }
            }
            IrOp::VFloatBinM {
                fault,
                dst,
                addr,
                op,
                prec,
                scalar,
            } => {
                if let Some(r) = exec_v_float_bin_m::<DEFAULT, ROUTED, ORDER>(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    addr,
                    op,
                    prec,
                    scalar,
                ) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VHFloat {
                dst,
                a,
                b,
                op,
                prec,
                bytes,
            } => {
                if let Some(r) = exec_v_h_float::<DEFAULT>(cpu, dst, a, b, op, prec, bytes) {
                    return r.into();
                }
            }
            IrOp::VHFloatM {
                fault,
                dst,
                a,
                addr,
                op,
                prec,
                bytes,
            } => {
                if let Some(r) = exec_v_h_float_m::<DEFAULT, ROUTED, ORDER>(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    a,
                    addr,
                    op,
                    prec,
                    bytes,
                ) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VHInt {
                dst,
                a,
                b,
                op,
                bytes,
            } => {
                if let Some(r) = exec_v_h_int(cpu, dst, a, b, op, bytes) {
                    return r.into();
                }
            }
            IrOp::VHIntM {
                fault,
                dst,
                a,
                addr,
                op,
                bytes,
            } => {
                if let Some(r) =
                    exec_v_h_int_m(cpu, mem, temps, cur_addr.get(), dst, a, addr, op, bytes)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VFloatCmpMask {
                dst,
                a,
                b,
                prec,
                scalar,
                pred,
            } => {
                if let Some(r) =
                    exec_v_float_cmp_mask::<DEFAULT>(cpu, dst, a, b, prec, scalar, pred)
                {
                    return r.into();
                }
            }
            IrOp::VFloatCmpMaskM {
                fault,
                dst,
                addr,
                prec,
                scalar,
                pred,
            } => {
                if let Some(r) = exec_v_float_cmp_mask_m::<DEFAULT, ROUTED, ORDER>(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    addr,
                    prec,
                    scalar,
                    pred,
                ) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VFloatCmpMask256 {
                dst,
                a,
                b,
                prec,
                pred,
            } => {
                if let Some(r) = exec_v_float_cmp_mask256::<DEFAULT>(cpu, dst, a, b, prec, pred) {
                    return r.into();
                }
            }
            IrOp::VFloatCmpMask256M {
                fault,
                dst,
                a,
                addr,
                prec,
                pred,
            } => {
                if let Some(r) = exec_v_float_cmp_mask256_m::<DEFAULT, ROUTED, ORDER>(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    a,
                    addr,
                    prec,
                    pred,
                ) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VFloatCmp { a, b, prec } => {
                if let Some(r) = exec_v_float_cmp::<DEFAULT>(cpu, temps, a, b, prec) {
                    return r.into();
                }
            }
            IrOp::VCvtFromInt {
                dst,
                src,
                int_size,
                prec,
                signed,
            } => {
                if let Some(r) =
                    exec_v_cvt_from_int::<DEFAULT>(cpu, temps, dst, src, int_size, prec, signed)
                {
                    return r.into();
                }
            }
            IrOp::VCvtToInt {
                dst,
                src,
                int_size,
                prec,
                trunc,
                signed,
            } => {
                if let Some(r) = exec_v_cvt_to_int::<DEFAULT>(
                    cpu, temps, dst, src, int_size, prec, trunc, signed,
                ) {
                    return r.into();
                }
            }
            IrOp::VCvtFloat { dst, src, from, to } => {
                if let Some(r) = exec_v_cvt_float::<DEFAULT>(cpu, temps, dst, src, from, to) {
                    return r.into();
                }
            }
            IrOp::VPackedCvt { dst, src, kind } => {
                if let Some(r) = exec_v_packed_cvt::<DEFAULT>(cpu, dst, src, kind) {
                    return r.into();
                }
            }
            IrOp::VFloatBin256 {
                dst,
                a,
                b,
                op,
                prec,
            } => {
                if let Some(r) = exec_v_float_bin256::<DEFAULT>(cpu, dst, a, b, op, prec) {
                    return r.into();
                }
            }
            IrOp::VFloatBin256M {
                fault,
                dst,
                a,
                addr,
                op,
                prec,
            } => {
                if let Some(r) = exec_v_float_bin256_m::<DEFAULT, ROUTED, ORDER>(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    a,
                    addr,
                    op,
                    prec,
                ) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VFloatUnary256 { dst, src, op, prec } => {
                if let Some(r) = exec_v_float_unary256::<DEFAULT>(cpu, dst, src, op, prec) {
                    return r.into();
                }
            }
            IrOp::VFloatUnary256M {
                fault,
                dst,
                addr,
                op,
                prec,
            } => {
                if let Some(r) = exec_v_float_unary256_m::<DEFAULT, ROUTED, ORDER>(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    addr,
                    op,
                    prec,
                ) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VPackedCvt256 { dst, src, kind } => {
                if let Some(r) = exec_v_packed_cvt256::<DEFAULT>(cpu, dst, src, kind) {
                    return r.into();
                }
            }
            IrOp::VPackedCvt256M {
                fault,
                dst,
                addr,
                kind,
            } => {
                if let Some(r) = exec_v_packed_cvt256_m::<DEFAULT, ROUTED, ORDER>(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    addr,
                    kind,
                ) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VPackedCvtWide256 { dst, src, kind } => {
                if let Some(r) = exec_v_packed_cvt_wide256::<DEFAULT>(cpu, dst, src, kind) {
                    return r.into();
                }
            }
            IrOp::VShufps256 {
                dst,
                a,
                b,
                imm_lo,
                imm_hi,
            } => {
                if let Some(r) = exec_v_shufps256(cpu, dst, a, b, imm_lo, imm_hi) {
                    return r.into();
                }
            }
            IrOp::VShufps256M {
                fault,
                dst,
                a,
                addr,
                imm_lo,
                imm_hi,
            } => {
                if let Some(r) = exec_v_shufps256_m(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    a,
                    addr,
                    imm_lo,
                    imm_hi,
                ) {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VUnpack256 {
                dst,
                a,
                b,
                lane,
                high,
            } => {
                if let Some(r) = exec_v_unpack256(cpu, dst, a, b, lane, high) {
                    return r.into();
                }
            }
            IrOp::VUnpack256M {
                fault,
                dst,
                a,
                addr,
                lane,
                high,
            } => {
                if let Some(r) =
                    exec_v_unpack256_m(cpu, mem, temps, cur_addr.get(), dst, a, addr, lane, high)
                {
                    return walk_classify_access(r, cur_addr.get(), read_val(*addr, temps), *fault);
                }
            }
            IrOp::VCvtPh2Ps { dst, src, lanes } => {
                let (lo, hi) = cvtph2ps(cpu.xmm[*src as usize], *lanes as usize);
                cpu.xmm[*dst as usize] = lo;
                cpu.ymm_hi[*dst as usize] = hi; // 0 for the 4-lane form
            }
            IrOp::VPermilVarM {
                fault,
                dst,
                src,
                addr,
                elem,
                bytes,
            } => {
                let ea = read_val(*addr, temps);
                let bv = match vload_lanes(cpu, mem, cur_addr.get(), ea, *bytes) {
                    Ok(v) => v,
                    Err(f) => {
                        return walk_classify_access(
                            vec_trap_out(cpu, cur_addr.get(), f, AccessKind::Read, ea, *bytes),
                            cur_addr.get(),
                            ea,
                            *fault,
                        )
                    }
                };
                let av = cpu.vec_lanes(*src as usize);
                let mut r = [0; 4];
                for i in 0..usize::from(*bytes) / 16 {
                    r[i] = permil_lane(av[i], bv[i], *elem);
                }
                cpu.set_vec(*dst as usize, r, *bytes);
            }
            IrOp::VPerm2i128M {
                fault,
                dst,
                a,
                addr,
                imm,
            } => {
                let ea = read_val(*addr, temps);
                let bv = match vload_lanes(cpu, mem, cur_addr.get(), ea, 32) {
                    Ok(v) => v,
                    Err(f) => {
                        return walk_classify_access(
                            vec_trap_out(cpu, cur_addr.get(), f, AccessKind::Read, ea, 32),
                            cur_addr.get(),
                            ea,
                            *fault,
                        )
                    }
                };
                let av = cpu.vec_lanes(*a as usize);
                let halves = [av[0], av[1], bv[0], bv[1]];
                let r = perm2i128_values(halves, *imm);
                cpu.set_vec(*dst as usize, [r[0], r[1], 0, 0], 32);
            }
            IrOp::VShiftVarM {
                fault,
                dst,
                a,
                addr,
                elem,
                right,
                arith,
                bytes,
            } => {
                let ea = read_val(*addr, temps);
                let bv = match vload_lanes(cpu, mem, cur_addr.get(), ea, *bytes) {
                    Ok(v) => v,
                    Err(f) => {
                        return walk_classify_access(
                            vec_trap_out(cpu, cur_addr.get(), f, AccessKind::Read, ea, *bytes),
                            cur_addr.get(),
                            ea,
                            *fault,
                        )
                    }
                };
                let av = cpu.vec_lanes(*a as usize);
                let mut r = [0; 4];
                for i in 0..usize::from(*bytes) / 16 {
                    r[i] = var_shift_lane(av[i], bv[i], *elem, *right, *arith);
                }
                cpu.set_vec(*dst as usize, r, *bytes);
            }
            IrOp::VCvtPs2PhM {
                fault,
                addr,
                src,
                lanes,
                rc,
            } => {
                let ea = read_val(*addr, temps);
                let mxcsr = interp_mxcsr::<DEFAULT>(cpu);
                let rc = if rc & 4 != 0 {
                    ((mxcsr >> 13) & 3) as u8
                } else {
                    *rc
                };
                let out = cvtps2ph_with_mxcsr(
                    cpu.xmm[*src as usize],
                    cpu.ymm_hi[*src as usize],
                    usize::from(*lanes),
                    rc,
                    mxcsr,
                );
                let size = lanes * 2;
                {
                    // Match the JIT's separate vstore_preflight helper boundary.
                    let (mem, _ordering) = mem.helper(Some(ordering::Direction::Write));
                    // SDM Vol 3A §7.5: check the whole destination before any RAM store.
                    for offset in (0..size).step_by(8) {
                        let at = ea.wrapping_add(u64::from(offset));
                        if let Err(
                            t @ (MemTrap::Unmapped | MemTrap::NonCanonical | MemTrap::HostFault),
                        ) = mem.probe_write_mode::<false>(at, 8)
                        {
                            let at = if t == MemTrap::Unmapped {
                                (0..8)
                                    .map(|i| at.wrapping_add(i))
                                    .find(|&b| {
                                        matches!(
                                            mem.probe_write_mode::<false>(b, 1),
                                            Err(MemTrap::Unmapped)
                                        )
                                    })
                                    .unwrap_or(at)
                            } else {
                                t.fault_addr(at)
                            };
                            return walk_classify_access(
                                trap_out(cpu, cur_addr.get(), t, at, 8, AccessKind::Write, 0),
                                cur_addr.get(),
                                ea,
                                *fault,
                            );
                        }
                    }
                    if mem.store_crosses_host_page(ea, usize::from(size)) {
                        if let Err((at, t)) = mem.preflight_store(ea, u64::MAX, 8, u16::from(size))
                        {
                            return walk_classify_access(
                                trap_out(cpu, cur_addr.get(), t, at, 8, AccessKind::Write, 0),
                                cur_addr.get(),
                                ea,
                                *fault,
                            );
                        }
                    }
                }
                let _ordering =
                    ordering::Barrier::<ORDER>::new(Some(ordering::Direction::Write), false);
                if let Err(f) = vstore_parts(
                    cpu,
                    &Preflighted(mem.unordered()),
                    cur_addr.get(),
                    ea,
                    out,
                    size,
                ) {
                    return walk_classify_access(
                        vec_trap_out(
                            cpu,
                            cur_addr.get(),
                            f,
                            AccessKind::Write,
                            ea,
                            u16::from(size),
                        ),
                        cur_addr.get(),
                        ea,
                        *fault,
                    );
                }
            }
            IrOp::VCvtPs2Ph {
                dst,
                src,
                lanes,
                rc,
            } => {
                let out = cvtps2ph_with_mxcsr(
                    cpu.xmm[*src as usize],
                    cpu.ymm_hi[*src as usize],
                    *lanes as usize,
                    if rc & 4 != 0 {
                        ((interp_mxcsr::<DEFAULT>(cpu) >> 13) & 3) as u8
                    } else {
                        *rc
                    },
                    interp_mxcsr::<DEFAULT>(cpu),
                );
                cpu.xmm[*dst as usize] = out;
                cpu.ymm_hi[*dst as usize] = 0;
            }
            IrOp::VPhMinPosUw { dst, src } => {
                cpu.xmm[*dst as usize] = phminposuw(cpu.xmm[*src as usize]);
            }
            IrOp::VMpsadbw {
                dst,
                a,
                b,
                imm,
                bytes,
            } => {
                cpu.xmm[*dst as usize] = mpsadbw(cpu.xmm[*a as usize], cpu.xmm[*b as usize], *imm);
                if *bytes == 32 {
                    // Per 128-bit lane; imm[5:3] is the high-lane control (imm[2:0] the low).
                    let imm_hi = (*imm >> 3) & 0x7;
                    cpu.ymm_hi[*dst as usize] =
                        mpsadbw(cpu.ymm_hi[*a as usize], cpu.ymm_hi[*b as usize], imm_hi);
                }
                // The VEX.128 form's upper clear is emitted as a separate VZeroUpper.
            }
            IrOp::VFloatUnary {
                dst,
                a,
                src,
                op,
                prec,
                scalar,
            } => {
                if let Some(r) = exec_v_float_unary::<DEFAULT>(cpu, dst, a, src, op, prec, scalar) {
                    return r.into();
                }
            }
            IrOp::VFloatUnaryM {
                fault,
                dst,
                a,
                src_addr,
                op,
                prec,
                scalar,
            } => {
                if let Some(r) = exec_v_float_unary_m::<DEFAULT, ROUTED, ORDER>(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    dst,
                    a,
                    src_addr,
                    op,
                    prec,
                    scalar,
                ) {
                    return walk_classify_access(
                        r,
                        cur_addr.get(),
                        read_val(*src_addr, temps),
                        *fault,
                    );
                }
            }
            IrOp::Jump { target } => {
                if let Some(r) = exec_jump(cpu, temps, target) {
                    return r.into();
                }
            }
            IrOp::Branch {
                cond,
                taken,
                fallthrough,
            } => {
                if let Some(r) = exec_branch(cpu, cond, taken, fallthrough) {
                    return r.into();
                }
            }
            IrOp::Call {
                fault,
                target,
                return_addr,
                slot,
                wrap_sp,
            } => {
                if let Some(r) = exec_call(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    target,
                    return_addr,
                    slot,
                    wrap_sp,
                ) {
                    return match r {
                        InterpResult::Continue => WalkResult::Continue,
                        _ => {
                            let base = cpu.gpr[RSP].wrapping_sub(u64::from(*slot));
                            let base = if *wrap_sp { base & 0xffff_ffff } else { base };
                            stack_fault_return(cur_addr.get(), base, *fault)
                        }
                    };
                }
            }
            IrOp::Ret {
                fault,
                slot,
                pop_extra,
                wrap_sp,
            } => {
                if let Some(r) = exec_ret(cpu, mem, cur_addr.get(), slot, pop_extra, wrap_sp) {
                    return match r {
                        InterpResult::Continue => WalkResult::Continue,
                        _ => stack_fault_return(cur_addr.get(), cpu.gpr[RSP], *fault),
                    };
                }
            }
            IrOp::Syscall { is_amd64 } => {
                if let Some(r) = exec_syscall(cpu, ir.next_ip(), *is_amd64) {
                    return r.into();
                }
            }
            IrOp::PortString {
                elem,
                rep,
                addr_bits,
                base,
                dir_out,
                stack,
            } => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                let amask = mask(*addr_bits / 8);
                let count = cpu.gpr[RCX] & amask;
                if *rep && count == 0 {
                    cpu.rip = ir.next_ip();
                    return WalkResult::Continue;
                }
                let index = if *dir_out { RSI } else { RDI };
                let offset = cpu.gpr[index] & amask;
                let base = read_val(*base, temps);
                // SDM Vol 3B §23.1.4, Table 23-1: real-mode operands
                // crossing the 64 KiB segment limit raise #GP (or #SS).
                if ir.mode == crate::CpuMode::Real16 && offset + u64::from(*elem) > 0x10000 {
                    return WalkResult::delivered_fault(deliver_interrupt_mode(
                        cpu,
                        mem,
                        cur_addr.get(),
                        if *stack { 12 } else { 13 },
                        cur_addr.get(),
                    ));
                }
                let delta = if cpu.flags.df {
                    0u64.wrapping_sub(u64::from(*elem))
                } else {
                    u64::from(*elem)
                };
                let next_index = offset.wrapping_add(delta) & amask;
                let next_count = count.wrapping_sub(1) & amask;
                let last = !*rep || next_count == 0;
                let next_ip = if last { ir.next_ip() } else { cur_addr.get() };
                let value = if *dir_out {
                    let (mem, _ordering) = mem.helper(Some(ordering::Direction::Read));
                    match string_load::<true, _>(
                        cpu,
                        mem,
                        cur_addr.get(),
                        0,
                        base.wrapping_add(offset),
                        *elem,
                    ) {
                        Ok(value) => value & mask(*elem),
                        Err(t) => {
                            // SDM Vol 2B §4.3 OUTS, 64-Bit Mode Exceptions.
                            if matches!(t, MemTrap::Unmapped | MemTrap::NonCanonical) {
                                return WalkResult::exit(crate::memory::unmapped_exit(
                                    cur_addr.get(),
                                    base.wrapping_add(offset),
                                    u64::from(*elem),
                                    ir.mode == crate::CpuMode::Long64,
                                    *stack,
                                    t.fault_addr(base.wrapping_add(offset)),
                                    AccessKind::Read,
                                ));
                            }
                            return WalkResult::exit(str_fault_exit(StrFault {
                                addr: t.fault_addr(base.wrapping_add(offset)),
                                write: false,
                                trap: t,
                                value: 0,
                                elem: *elem,
                            }));
                        }
                    }
                } else {
                    0
                };
                // Decision-16 §2: the handoff owns one step, even with a pending
                // input result. Whole-instruction retirement occurs at the final handoff.
                if STEPPED {
                    budget.get().charge();
                }
                *retired += u64::from(last);
                started.set(1);
                if *dir_out {
                    cpu.write_gpr(index, next_index, *addr_bits / 8);
                    if *rep {
                        cpu.write_gpr(RCX, next_count, *addr_bits / 8);
                    }
                    cpu.rip = next_ip;
                    cpu.clear_mmio_parts();
                    shadows.complete();
                } else {
                    cpu.pending_port_in = Some(*elem);
                    cpu.port_string_in = Some(crate::state::PortStringIn {
                        rip: cur_addr.get(),
                        next_ip,
                        instruction_end: ir.next_ip(),
                        base,
                        offset,
                        next_index,
                        next_count,
                        addr_bits: *addr_bits,
                        elem: *elem,
                        rep: *rep,
                        value: None,
                    });
                    cpu.rip = cur_addr.get();
                }
                return WalkResult::exit(Exit::PortIo {
                    port: cpu.gpr[2] as u16,
                    size: *elem,
                    dir: if *dir_out { PortDir::Out } else { PortDir::In },
                    value,
                });
            }
            IrOp::PortIo {
                port,
                value,
                size,
                dir_out,
            } => {
                if let Some(r) = exec_port_io(cpu, temps, ir.next_ip(), port, value, size, dir_out)
                {
                    return r.into();
                }
            }
            IrOp::Hlt => {
                if let Some(r) = exec_hlt(cpu, ir.next_ip()) {
                    return r.into();
                }
            }
            IrOp::Trap { vector, advance } => {
                if CHILD {
                    let cause = match vector {
                        3 => crate::FaultCause::Breakpoint,
                        6 => crate::FaultCause::InvalidOpcode,
                        13 => crate::FaultCause::InstructionLength,
                        _ => crate::FaultCause::Unknown,
                    };
                    mem.0.exception(*vector, cause, *advance);
                    return WalkResult::Continue;
                }
                if let Some(r) = exec_trap(cpu, cur_addr.get(), ir.mode, vector, advance) {
                    return r.into();
                }
            }
            // --- real-mode interrupt-flag + IVT delivery (§17.6) ---
            IrOp::PushFlags {
                fault,
                size,
                wrap_sp,
            } => {
                let mut sp = cpu.gpr[RSP].wrapping_sub(*size as u64);
                if *wrap_sp {
                    sp &= 0xffff_ffff;
                }
                let image = cpu.rflags() & !0x3_0000;
                if let Err(t) = mem.write(sp, image, *size) {
                    return walk_classify_access(
                        trap_out(cpu, cur_addr.get(), t, sp, *size, AccessKind::Write, image),
                        cur_addr.get(),
                        sp,
                        *fault,
                    );
                }
                cpu.gpr[RSP] = sp;
            }
            IrOp::RestoreFlags { src, size, iret } => {
                cpu.set_user_rflags(read_val(*src, temps), *size, *iret);
            }
            IrOp::ReadSelector { dst, segment } => {
                temps[*dst as usize] = cpu.selector(*segment) as u64;
            }
            IrOp::CheckIretNt => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                if cpu.rflags_system & (1 << 14) != 0 {
                    cpu.rip = cur_addr.get();
                    return InterpResult::exit(crate::fault::exception_exit(
                        cur_addr.get(),
                        13,
                        crate::FaultCause::IretNt,
                    ))
                    .into();
                }
            }
            IrOp::CheckIret { cs, ss, ip } => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                let ip = read_val(*ip, temps);
                if read_val(*cs, temps) as u16 != cpu.cs
                    || ss.is_some_and(|v| read_val(v, temps) as u16 != cpu.ss)
                    || (ss.is_some() && ((ip << 16) as i64 >> 16) as u64 != ip)
                {
                    cpu.rip = cur_addr.get();
                    return InterpResult::exit(crate::fault::exception_exit(
                        cur_addr.get(),
                        13,
                        if read_val(*cs, temps) as u16 != cpu.cs
                            || ss.is_some_and(|v| read_val(v, temps) as u16 != cpu.ss)
                        {
                            crate::FaultCause::IretSelector
                        } else {
                            crate::FaultCause::NonCanonical
                        },
                    ))
                    .into();
                }
            }
            IrOp::Unsupported { bytes, len } => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                cpu.rip = cur_addr.get();
                return WalkResult::exit(Exit::UnknownInstruction {
                    addr: cur_addr.get(),
                    bytes: *bytes,
                    len: *len,
                });
            }
            IrOp::SetIf { value } => {
                // SDM Vol 2B §4.3 STI: only an execution beginning with IF=0
                // inhibits the following instruction boundary.
                // SetIf is the only op of CLI/STI, so IF still has its entry value.
                shadows.complete_before(*retired);
                shadows.arm_sti(*value, cpu.flags.if_);
                cpu.flags.if_ = *value;
            }
            IrOp::SetSsShadow => {
                // SDM Vol 3A §7.8.3; Vol 2B §4.3 MOV/POP: the next instruction
                // consumes inhibition without changing IF. Decision-16 §4 chooses
                // not to rearm an incoming SS shadow on consecutive SS loads.
                shadows.complete_before(*retired);
                shadows.arm_ss();
            }
            IrOp::PushfReal => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                if let Some(r) = exec_pushf_real(cpu, mem, cur_addr.get()) {
                    return r.into();
                }
            }
            IrOp::PopfReal => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                if let Some(r) = exec_popf_real(cpu, mem, cur_addr.get()) {
                    return r.into();
                }
            }
            IrOp::IntGate { vector, saved_ip } => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                // The IR's saved IP classifies the gate before delivery: faults save
                // the instruction start; software INT/INT3 save the next IP (SDM
                // Vol 3A §7.5). The handler's destination CS:IP is independent.
                let fault = *saved_ip == cur_addr.get();
                let result = deliver_interrupt_mode(cpu, mem, cur_addr.get(), *vector, *saved_ip);
                return if fault {
                    WalkResult::delivered_fault(result)
                } else {
                    result.into()
                };
            }
            IrOp::IntoGate { next_ip } => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                return if cpu.flags.of {
                    deliver_interrupt_mode(cpu, mem, cur_addr.get(), 4, *next_ip).into()
                } else {
                    cpu.rip = *next_ip;
                    WalkResult::Continue
                };
            }
            IrOp::SegLimitCheck {
                offset,
                size,
                vector,
                fault_ip,
            } => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                // 80286 real-mode segment-limit fault (§17.6): the access crosses the
                // 0xFFFF byte limit iff offset + size overflows 64 KB. If so, vector the
                // fault in-guest; otherwise fall through to the Load/Store that follows.
                let off = read_val(*offset, temps) & 0xFFFF;
                if off + *size as u64 > 0x1_0000 {
                    return WalkResult::delivered_fault(deliver_interrupt_mode(
                        cpu,
                        mem,
                        cur_addr.get(),
                        *vector,
                        *fault_ip,
                    ));
                }
            }
            IrOp::BoundGate {
                index,
                lower,
                upper,
                fault_ip,
                next_ip,
            } => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                // Signed 16-bit array-bounds check (§17.6). Out of range → #BR (vector 5),
                // a fault whose saved IP is the `bound` instruction itself.
                let idx = read_val(*index, temps) as u16 as i16;
                let lo = read_val(*lower, temps) as u16 as i16;
                let hi = read_val(*upper, temps) as u16 as i16;
                return if idx < lo || idx > hi {
                    WalkResult::delivered_fault(deliver_interrupt_mode(
                        cpu,
                        mem,
                        cur_addr.get(),
                        5,
                        *fault_ip,
                    ))
                } else {
                    cpu.rip = *next_ip;
                    WalkResult::Continue
                };
            }
            IrOp::IretReal => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                return exec_iret_real(cpu, mem, cur_addr.get()).into();
            }
            IrOp::SetCf { value } => {
                // `clc`/`stc`/`cmc`: set/clear/complement CF, nothing else.
                cpu.flags.cf = match value {
                    Some(v) => *v,
                    None => !cpu.flags.cf,
                };
            }
            IrOp::Bcd { kind } => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                if let Some(r) = exec_bcd(cpu, cur_addr.get(), kind) {
                    return r.into();
                }
            }
            IrOp::LoopCx {
                kind,
                taken,
                fallthrough,
            } => return exec_loop_cx(cpu, kind, taken, fallthrough).into(),
            IrOp::FarJump { cs, ip } => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                return exec_far_jump(cpu, temps, cs, ip).into();
            }
            IrOp::FarCall { cs, ip, ret_ip } => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                return exec_far_call(cpu, mem, temps, cur_addr.get(), cs, ip, ret_ip).into();
            }
            IrOp::FarRet { pop_extra } => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                return exec_far_ret(cpu, mem, cur_addr.get(), pop_extra).into();
            }
            IrOp::RepStringReal {
                op,
                elem,
                rep,
                ds_base,
                es_base,
            } => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                let (mem, _ordering) = mem.helper(Some(match op {
                    StrOp::Ins | StrOp::Outs => unreachable!("port strings use PortString"),
                    StrOp::Movs => ordering::Direction::ReadWrite,
                    StrOp::Stos => ordering::Direction::Write,
                    StrOp::Lods | StrOp::Scas | StrOp::Cmps => ordering::Direction::Read,
                }));
                if STEPPED {
                    let stop = string_run_bounded::<ROUTED, STOP_WHEN_IRQ_READY, _>(
                        cpu,
                        mem,
                        *op,
                        *elem,
                        *rep,
                        cur_addr.get(),
                        16,
                        read_val(*ds_base, temps),
                        read_val(*es_base, temps),
                        exit_request,
                        budget.get(),
                        shadows,
                        *retired,
                    );
                    if let Some(stop) = stop {
                        return stop;
                    }
                    continue;
                }
                let before = cpu.gpr[1];
                let result = exec_rep_string_real(
                    cpu,
                    mem,
                    temps,
                    cur_addr.get(),
                    op,
                    elem,
                    rep,
                    ds_base,
                    es_base,
                    exit_request,
                );
                shadows.complete_rep_prefix(before, cpu, *retired);
                if let Some(r) = result {
                    return r.into();
                }
            }
            IrOp::Lahf => {
                if let Some(r) = exec_lahf(cpu) {
                    return r.into();
                }
            }
            IrOp::Sahf => {
                if let Some(r) = exec_sahf(cpu) {
                    return r.into();
                }
            }
            IrOp::PushaReal => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                if let Some(r) = exec_pusha_real(cpu, mem, cur_addr.get()) {
                    return r.into();
                }
            }
            IrOp::PopaReal => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                if let Some(r) = exec_popa_real(cpu, mem, cur_addr.get()) {
                    return r.into();
                }
            }
            IrOp::EnterReal { alloc, level } => {
                if CHILD {
                    unreachable!("helper refused before child execution");
                }
                if let Some(r) = exec_enter_real(cpu, mem, cur_addr.get(), *alloc, *level) {
                    return r.into();
                }
            }
            IrOp::CheckAlign { addr, bytes } => {
                if read_val(*addr, temps) & (u64::from(*bytes) - 1) != 0 {
                    if CHILD {
                        mem.0.exception(13, crate::FaultCause::RequiredAlignment, 0);
                        return WalkResult::Continue;
                    }
                    cpu.rip = cur_addr.get();
                    return InterpResult::exit(crate::fault::exception_exit(
                        cur_addr.get(),
                        13,
                        crate::FaultCause::RequiredAlignment,
                    ))
                    .into();
                }
            }
        }
        if CHILD && mem.0.stopped() {
            return WalkResult::Continue;
        }
    }

    // Straight-line block with no control-flow terminator (code ran out): flow on
    // from just past the decoded bytes.
    cpu.rip = ir.next_ip();
    WalkResult::Continue
}

pub(crate) fn walk_child<const ORDER: u8>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    mem: &impl InterpreterMemory,
    scratch: &mut Vec<u64>,
) -> StepResult {
    scratch.clear();
    scratch.resize(ir.temp_count as usize, 0);
    let cur_addr = Cell::new(ir.guest_start);
    let started = Cell::new(1u8);
    let mut retired = 0;
    let mut shadows = ShadowState::default();
    let mem = Access::<false, ORDER, _>::new(mem);
    let result = if cpu.mxcsr & softfloat::MODE_MASK == 0 {
        walk_ops::<false, false, true, true, ORDER, false, false>(
            ir,
            cpu,
            mem,
            scratch,
            &cur_addr,
            &mut retired,
            &started,
            &mut shadows,
            None,
            None,
            (),
        )
    } else {
        walk_ops::<false, false, false, true, ORDER, false, false>(
            ir,
            cpu,
            mem,
            scratch,
            &cur_addr,
            &mut retired,
            &started,
            &mut shadows,
            None,
            None,
            (),
        )
    };
    match result {
        WalkResult::Continue | WalkResult::ModeChange => StepResult::Continue,
        WalkResult::Exit(exit) => StepResult::Exit(exit.into_exit()),
        WalkResult::Fault | WalkResult::FaultDelivered => {
            unreachable!("child faults use arena records")
        }
    }
}

mod control;
mod integer;
mod vector;

pub(crate) use control::*;
pub(crate) use integer::*;
pub(crate) use vector::*;

/// The four-way SSE bitwise op (`pxor`/`pand`/`por`/`pandn`), shared by the
/// register (`VLogic`) and memory (`VLogicM`) forms so `Andn`'s non-commutative
/// `!a & b` can't drift between two hand-copied matches.
fn vlogic(a: u128, b: u128, op: VLogicOp) -> u128 {
    match op {
        VLogicOp::Xor => a ^ b,
        VLogicOp::And => a & b,
        VLogicOp::Or => a | b,
        VLogicOp::Andn => !a & b,
    }
}

/// `pmovzx`/`pmovsx`: read `16/to` low elements of `from` bytes from `src`, zero- or
/// sign-extend each to `to` bytes, pack into the 128-bit result.
fn pmov_extend(src: u128, from: u8, to: u8, signed: bool) -> u128 {
    let (from, to) = (from as usize, to as usize);
    let count = 16 / to;
    let sb = src.to_le_bytes();
    let mut out = [0u8; 16];
    for i in 0..count {
        let mut val = 0u64;
        for j in 0..from {
            val |= (sb[i * from + j] as u64) << (8 * j);
        }
        // Sign-extend within a u64 when signed; the low `to` bytes are then written.
        let ext = if signed {
            let bits = from as u32 * 8;
            let sign = 1u64 << (bits - 1);
            if val & sign != 0 {
                val | (u64::MAX << bits)
            } else {
                val
            }
        } else {
            val
        };
        let eb = ext.to_le_bytes();
        out[i * to..i * to + to].copy_from_slice(&eb[..to]);
    }
    u128::from_le_bytes(out)
}

/// SSE4.2 `pcmpistri`/`pcmpestri`: the string-compare aggregation that
/// returns an index in ECX plus flags. `len1`/`len2` are the valid element counts (for
/// `pcmpistri` the position of the first null element; for `pcmpestri` the saturated
/// |EAX|/|EDX|). Returns `(ecx, cf, zf, sf, of)`; AF and PF are cleared by the caller.
/// Follows the Intel SDM per-(i,j) validity-override table.
pub fn pcmpstr(
    a: u128,
    b: u128,
    len1: usize,
    len2: usize,
    imm: u8,
) -> (u32, bool, bool, bool, bool) {
    let (intres2, n, cf, zf, sf, of) = pcmpstr_intres2(a, b, len1, len2, imm);
    let msb = imm & 0x40 != 0;
    let ecx = if intres2 == 0 {
        n as u32
    } else if msb {
        31 - intres2.leading_zeros()
    } else {
        intres2.trailing_zeros()
    };
    (ecx, cf, zf, sf, of)
}

/// SSE4.2 `pcmpistrm`/`pcmpestrm`: the same aggregation as [`pcmpstr`] but the
/// per-element result bitmask `intres2` is expanded into an XMM0 mask instead of an index.
/// `imm[6]==0` → bit mask (result bits in the low bytes, zero-extended); `imm[6]==1` → byte
/// (or word) mask (each result bit expands to a full `0x00`/`0xFF..` element). Returns
/// `(mask, cf, zf, sf, of)`; the flags are identical to the index form. AF/PF cleared by
/// callers.
fn pcmpstr_mask(
    a: u128,
    b: u128,
    len1: usize,
    len2: usize,
    imm: u8,
) -> (u128, bool, bool, bool, bool) {
    let (intres2, n, cf, zf, sf, of) = pcmpstr_intres2(a, b, len1, len2, imm);
    let words = imm & 1 != 0;
    let byte_mask = imm & 0x40 != 0;
    let ew = if words { 2usize } else { 1 };
    let elem_mask: u128 = if words { 0xFFFF } else { 0xFF };
    let mask = if byte_mask {
        // Expand each result bit to a full 0x00/0xFF.. element.
        let mut m = 0u128;
        for i in 0..n {
            if intres2 & (1 << i) != 0 {
                m |= elem_mask << (i * ew * 8);
            }
        }
        m
    } else {
        // Bit mask: the result bits live in the low bytes, zero-extended to 128 bits.
        intres2 as u128
    };
    (mask, cf, zf, sf, of)
}

/// Shared core of the `pcmpstr` family: run the aggregation and return `(intres2, n, cf,
/// zf, sf, of)`. `intres2` is the per-src2-element result bitmask (post-polarity); both the
/// index (`pcmp*i`) and mask (`pcmp*m`) forms build on it. `n` is 8 (words) or 16 (bytes).
fn pcmpstr_intres2(
    a: u128,
    b: u128,
    len1: usize,
    len2: usize,
    imm: u8,
) -> (u32, usize, bool, bool, bool, bool) {
    let words = imm & 1 != 0;
    let signed = imm & 2 != 0;
    let agg = (imm >> 2) & 3;
    let polarity = (imm >> 4) & 3;
    let n = if words { 8 } else { 16 };
    let ew = if words { 2 } else { 1 }; // element width in bytes
    let mask = if words { 0xFFFFu128 } else { 0xFF };

    let get = |v: u128, i: usize| -> i32 {
        let raw = ((v >> (i * ew * 8)) & mask) as u32;
        if signed {
            if words {
                raw as u16 as i16 as i32
            } else {
                raw as u8 as i8 as i32
            }
        } else {
            raw as i32
        }
    };
    let a_inv = |j: usize| j >= len1;
    let b_inv = |i: usize| i >= len2;

    // Per-(src2 i, src1 j) boolean after the validity override.
    let overridden = |i: usize, j: usize| -> bool {
        let base = match agg {
            1 => {
                // ranges: even j is the range lower bound (src1[j] <= src2[i]), odd is upper.
                if j & 1 == 0 {
                    get(a, j) <= get(b, i)
                } else {
                    get(a, j) >= get(b, i)
                }
            }
            _ => get(a, j) == get(b, i),
        };
        let (ai, bi) = (a_inv(j), b_inv(i));
        match agg {
            0 | 1 => {
                if ai || bi {
                    false
                } else {
                    base
                }
            }
            2 => {
                if ai && bi {
                    true
                } else if ai != bi {
                    false
                } else {
                    base
                }
            }
            _ => {
                // equal ordered
                if ai {
                    true
                } else if bi {
                    false
                } else {
                    base
                }
            }
        }
    };

    let mut intres1: u32 = 0;
    for i in 0..n {
        let bit = match agg {
            0 => (0..n).any(|j| overridden(i, j)),
            1 => (0..n)
                .step_by(2)
                .any(|j| overridden(i, j) && overridden(i, j + 1)),
            2 => overridden(i, i),
            _ => (0..n).all(|j| {
                if i + j < n {
                    overridden(i + j, j)
                } else {
                    a_inv(j) // past the haystack end: OK only if the needle is exhausted
                }
            }),
        };
        if bit {
            intres1 |= 1 << i;
        }
    }

    // Polarity → IntRes2.
    let nmask: u32 = if words { 0xFF } else { 0xFFFF };
    let intres2 = match polarity {
        1 => (!intres1) & nmask,                    // negate all
        3 => intres1 ^ ((1u32 << len2.min(n)) - 1), // negate only valid src2 positions
        _ => intres1 & nmask,                       // positive
    } & nmask;

    let cf = intres2 != 0;
    let zf = len2 < n;
    let sf = len1 < n;
    let of = intres2 & 1 != 0;
    (intres2, n, cf, zf, sf, of)
}

/// SSE4.1 `insertps`: insert the 32-bit source dword `tmp` into `dst` at the
/// destination lane `imm[5:4]`, then zero each dword `i` whose `imm[i]` bit is set. `dst` is
/// the 128-bit destination register; the source-lane select `imm[7:6]` is resolved by the
/// caller (register form reads `src.dword[imm[7:6]]`; the m32 form uses the loaded dword).
pub fn insertps(dst: u128, tmp: u32, imm: u8) -> u128 {
    let dst_lane = ((imm >> 4) & 3) as usize;
    let mut dw = [0u32; 4];
    for (i, slot) in dw.iter_mut().enumerate() {
        *slot = (dst >> (i * 32)) as u32;
    }
    dw[dst_lane] = tmp;
    for (i, slot) in dw.iter_mut().enumerate() {
        if imm & (1 << i) != 0 {
            *slot = 0;
        }
    }
    let mut out = 0u128;
    for (i, &v) in dw.iter().enumerate() {
        out |= (v as u128) << (i * 32);
    }
    out
}

/// Quiet an f64 NaN by setting the most-significant fraction bit (SDM Vol 1 §4.8.3.5).
fn quiet_f64(v: f64) -> f64 {
    f64::from_bits(v.to_bits() | 0x0008_0000_0000_0000)
}

/// Quiet an f32 NaN by setting the most-significant fraction bit (SDM Vol 1 §4.8.3.5:
/// "when an SNaN is converted to a QNaN, the conversion is handled by setting the
/// most-significant fraction bit of the SNaN to 1"). Idempotent on a QNaN.
fn quiet_f32(v: f32) -> f32 {
    f32::from_bits(v.to_bits() | 0x0040_0000)
}

/// One SSE/AVX binary f32 operation with the architectural NaN result, not Rust's.
///
/// **This exists because Rust and LLVM leave the NaN payload unspecified, and the x86
/// rule is operand-order-dependent.** SDM Vol 1 §4.8.3.5, Table 4-8 — for
/// SSE/SSE2/SSE3/SSE4.1/AVX, a NaN result is the **first source operand** (quieted if it
/// is an SNaN), falling back to the second. LLVM is free to commute `fadd`/`fmul`, which
/// it does at higher optimization levels, so `a * b` in Rust returns one operand's
/// payload in a debug build and the other's in a release build of the same source — and a
/// third answer on another host, since the payload comes from whatever multiply that host
/// happens to have. The interpreter is the JIT's oracle; an oracle whose answer depends on
/// `opt-level` or the host is not one.
///
/// Found via `cargo xfuzz --seed 26816`, where a `vdpps` NaN reached `vpsignd`, which
/// turned the payload difference into a plain integer divergence the NaN-tolerant
/// comparator could not see through.
pub(crate) fn sse_binop_f32(src1: f32, src2: f32, op: impl FnOnce(f32, f32) -> f32) -> f32 {
    let r = op(src1, src2);
    if r.is_nan() {
        repair_sse_nan_f32(src1, src2)
    } else {
        r
    }
}

// Keep operand classification out of LLVM's ordinary arithmetic path.
#[cold]
#[inline(never)]
fn repair_sse_nan_f32(src1: f32, src2: f32) -> f32 {
    if src1.is_nan() {
        quiet_f32(src1)
    } else if src2.is_nan() {
        quiet_f32(src2)
    } else {
        // SDM Vol 1 §4.8.3.7: invalid arithmetic without a NaN source.
        f32::from_bits(F32_INDEFINITE)
    }
}

/// Widen a NaN from f32 to f64 the way `cvtss2sd`/`cvtps2pd` do: sign kept, significand
/// carried by 29 bits (f64 has 29 more significand bits than f32) and the quiet bit set,
/// so an SNaN becomes a QNaN with its payload in place.
///
/// Rust's `as f64` guarantees none of that — not the payload, not the SNaN conversion —
/// and these are the instructions whose entire job is the conversion. Derived from
/// SDM Vol 1 §4.8.3.5, Table 4-8 (a one-operand op returns its NaN source operand,
/// converted to a QNaN) and MEASURED on hardware through the native oracle, which is what
/// fixes the shift: `0x7FC01234 → 0x7FF8024680000000`, `0x7F801234 → 0x7FF8024680000000`
/// (the SNaN quieted to the same value), `0xFFC01234 → 0xFFF8024680000000`.
pub(crate) fn f32_nan_to_f64(v: f32) -> f64 {
    let b = v.to_bits() as u64;
    let sign = (b & 0x8000_0000) << 32;
    let payload = (b & 0x007F_FFFF) << 29;
    f64::from_bits(sign | 0x7FF8_0000_0000_0000 | payload)
}

/// Narrow a NaN from f64 to f32 the way `cvtsd2ss`/`cvtpd2ps` do — the inverse shift, so
/// the low 29 payload bits are lost and the quiet bit is set. Measured the same way:
/// `0x7FF8123456789ABC → 0x7FC091A2`, `0x7FF0123456789ABC → 0x7FC091A2`,
/// `0x7FF8000000000001 → 0x7FC00000` (a payload that does not survive the shift),
/// `0xFFF8123456789ABC → 0xFFC091A2`.
pub(crate) fn f64_nan_to_f32(v: f64) -> f32 {
    let b = v.to_bits();
    let sign = ((b >> 32) as u32) & 0x8000_0000;
    let payload = ((b & 0x000F_FFFF_FFFF_FFFF) >> 29) as u32;
    f32::from_bits(sign | 0x7FC0_0000 | payload)
}

/// [`sse_binop_f32`] at double precision. Same rule, same reason.
pub(crate) fn sse_binop_f64(src1: f64, src2: f64, op: impl FnOnce(f64, f64) -> f64) -> f64 {
    let r = op(src1, src2);
    if r.is_nan() {
        repair_sse_nan_f64(src1, src2)
    } else {
        r
    }
}

// Keep operand classification out of LLVM's ordinary arithmetic path.
#[cold]
#[inline(never)]
fn repair_sse_nan_f64(src1: f64, src2: f64) -> f64 {
    if src1.is_nan() {
        quiet_f64(src1)
    } else if src2.is_nan() {
        quiet_f64(src2)
    } else {
        // SDM Vol 1 §4.8.3.7: invalid arithmetic without a NaN source.
        f64::from_bits(F64_INDEFINITE)
    }
}

/// SSE4.1 `dpps`: single-precision dot product. `imm[7:4]` selects which of the
/// four `a[i]*b[i]` products enter the sum; `imm[3:0]` selects which result dwords receive
/// the broadcast sum (others are zeroed). The four-term sum is evaluated in lane order with
/// IEEE f32 arithmetic to match the CPU (rounding), and with the architectural NaN rule
/// spelled out rather than inherited from the host — see [`sse_binop_f32`]. Returns the
/// 128-bit result. Shared by the interpreter and the JIT helper → jit == interp.
pub fn dpps(a: u128, b: u128, imm: u8) -> u128 {
    dpps_mode(a, b, imm, 0)
}

pub fn dpps_mode(a: u128, b: u128, imm: u8, mxcsr: u32) -> u128 {
    dpps_mode_specialized::<false>(a, b, imm, mxcsr)
}

fn dpps_mode_specialized<const DEFAULT: bool>(a: u128, b: u128, imm: u8, mxcsr: u32) -> u128 {
    let mxcsr = if DEFAULT { 0 } else { mxcsr };
    let lane = |v: u128, i: usize| f32::from_bits((v >> (i * 32)) as u32);
    let mut p = [0.0f32; 4];
    for (i, slot) in p.iter_mut().enumerate() {
        if imm & (0x10 << i) != 0 {
            *slot = apply_f32_mode::<DEFAULT>(lane(a, i), lane(b, i), FloatBinOp::Mul, mxcsr);
        }
    }
    // SDM tree order: (P0+P1) + (P2+P3).
    let s01 = apply_f32_mode::<DEFAULT>(p[0], p[1], FloatBinOp::Add, mxcsr);
    let s23 = apply_f32_mode::<DEFAULT>(p[2], p[3], FloatBinOp::Add, mxcsr);
    let sum = apply_f32_mode::<DEFAULT>(s01, s23, FloatBinOp::Add, mxcsr);
    let mut out = 0u128;
    for i in 0..4 {
        let v = if imm & (1 << i) != 0 { sum } else { 0.0 };
        out |= (v.to_bits() as u128) << (i * 32);
    }
    out
}

/// F16C `vcvtph2ps` half→single. Decode an IEEE-754 binary16 to f32 exactly
/// (f32 has ≥ all f16 significand/exponent range, so every value — including subnormals,
/// inf — maps without rounding). SDM Vol 2C §5.2 VCVTPH2PS and Vol 1
/// §14.4, Table 14-9: NaNs retain sign/payload and signaling NaNs are quieted.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let sign_f = (sign as u32) << 31;
    let bits = if exp == 0 {
        if mant == 0 {
            sign_f // ±0
        } else {
            // Subnormal half = mant × 2^-24 (exact in f32). Compute the value directly and
            // reapply the sign — avoids off-by-one exponent slips in a manual renormalize.
            let v = (mant as f32) * (2.0f32).powi(-24);
            (v.to_bits() & 0x7fff_ffff) | sign_f
        }
    } else if exp == 0x1f {
        sign_f | (0xff << 23) | ((mant as u32) << 13) | if mant != 0 { 0x0040_0000 } else { 0 }
    } else {
        // Normal: rebias exponent (127 - 15 = 112), left-align mantissa.
        let exp32 = (exp as u32) + 112;
        sign_f | (exp32 << 23) | ((mant as u32) << 13)
    };
    f32::from_bits(bits)
}

/// F16C `vcvtps2ph` single→half with the imm8[2:0] rounding control: 0 = round
/// to nearest even, 1 = toward -inf, 2 = toward +inf, 3 = toward zero. Callers
/// resolve imm8[2] from MXCSR.RC (SDM Vol 2C §5.2 VCVTPS2PH, Table 5-3).
pub fn f32_to_f16(f: f32, rc: u8) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x7f_ffff;
    // Inf / NaN.
    if exp == 0xff {
        if mant != 0 {
            // NaN: the top 10 significand bits carry over and bit 9 is forced, which is
            // what `0x7e00` already sets — the result is a QNaN whatever the payload, so
            // nothing needs forcing on top of it. Do NOT force bit 0 (a `.max(1)`) "to
            // keep it a NaN": that corrupts the payload of every source NaN whose carried
            // bits are zero — hardware returns `0x7e00`, forcing gives `0x7e01`
            // (`cargo xfuzz --seed 661`).
            return sign | 0x7e00 | ((mant >> 13) as u16 & 0x3ff);
        }
        return sign | 0x7c00; // ±inf
    }
    // Unbiased exponent for half (bias 15).
    let e = exp - 127 + 15;
    // Rounding: pick the increment based on the discarded low bits and the mode.
    let round = |value: u32, shift: u32, half_sign: u16| -> u32 {
        if shift == 0 {
            return value;
        }
        let lost_mask = (1u32 << shift) - 1;
        let lost = value & lost_mask;
        let truncated = value >> shift;
        let halfway = 1u32 << (shift - 1);
        let up = match rc & 0x3 {
            1 => half_sign != 0 && lost != 0, // toward -inf
            2 => half_sign == 0 && lost != 0, // toward +inf
            3 => false,                       // toward zero
            _ => lost > halfway || (lost == halfway && (truncated & 1) == 1), // nearest even
        };
        truncated + up as u32
    };
    if e >= 0x1f {
        // Overflow: |value| ≥ 2^16 > the max finite half (65504 = 0x7bff). Per the SDM RC rule
        // the result is ±inf when the active mode rounds away from zero for this sign, else it
        // caps at the max finite half (derived from the SDM VCVTPS2PH pseudocode):
        //   nearest-even → inf  (value ≥ 2^16 is past the 65520 round-to-inf tie, so never caps)
        //   toward -inf  → -inf for negatives, +max-finite for positives
        //   toward +inf  → +inf for positives, -max-finite for negatives
        //   toward zero  → max finite, both signs
        let to_inf = match rc & 0x3 {
            1 => sign != 0, // toward -inf
            2 => sign == 0, // toward +inf
            3 => false,     // toward zero
            _ => true,      // nearest-even
        };
        return if to_inf { sign | 0x7c00 } else { sign | 0x7bff };
    }
    if e <= 0 {
        // Subnormal or underflow to zero. Build the full significand (with implicit 1),
        // then shift right by (14 - e) with rounding.
        if e < -10 {
            // Too small for even the smallest subnormal (2^-24). A true zero stays signed zero
            // in every mode; a nonzero magnitude rounds to the smallest subnormal (0x0001 /
            // 0x8001) only when the directed mode rounds away from zero — toward +inf for a
            // positive, toward -inf for a negative. Nearest-even and toward-zero flush
            // to ±0.
            let nonzero = x & 0x7fff_ffff != 0;
            let up = match rc & 0x3 {
                1 => nonzero && sign != 0, // toward -inf
                2 => nonzero && sign == 0, // toward +inf
                _ => false,                // nearest-even / toward zero
            };
            return sign | up as u16;
        }
        let full = mant | 0x80_0000; // 1.mant, 24 bits
        let shift = (14 - e) as u32; // ≥ 14
        let m = round(full, shift, sign);
        // A round-up carry to 0x400 is the smallest NORMAL (exp field 1, mantissa 0), not a
        // wrapped zero — the binary16 encoding is contiguous, so `sign | m` is exact. Masking
        // with 0x3ff would drop the carry and wrap the largest subnormal back to zero.
        return sign | m as u16;
    }
    // Normal: round the 23-bit mantissa down to 10 bits (drop 13).
    let m = round(mant, 13, sign);
    // Rounding may carry into the exponent (mant overflow 0x400 → bump exp).
    let mut half_exp = e as u32;
    let mut half_mant = m;
    if half_mant & 0x400 != 0 {
        half_mant = 0;
        half_exp += 1;
        if half_exp >= 0x1f {
            return sign | 0x7c00; // carried into inf
        }
    }
    sign | ((half_exp as u16) << 10) | (half_mant as u16 & 0x3ff)
}

/// F16C `vcvtph2ps` core: convert `lanes` binary16 elements from the low bits of
/// `src` to f32, packed into a 128/256-bit result (returned as two 128-bit halves).
pub fn cvtph2ps(src: u128, lanes: usize) -> (u128, u128) {
    let mut out = [0u128; 2];
    for i in 0..lanes {
        let h = (src >> (i * 16)) as u16;
        let f = f16_to_f32(h).to_bits() as u128;
        let half = i / 4;
        let pos = (i % 4) * 32;
        out[half] |= f << pos;
    }
    (out[0], out[1])
}

/// F16C `vcvtps2ph` core: convert `lanes` f32 elements from the 128/256-bit
/// source (`slo`/`shi`) to binary16, packed into the low bits of a 128-bit result.
pub fn cvtps2ph(slo: u128, shi: u128, lanes: usize, rc: u8) -> u128 {
    let mut out = 0u128;
    for i in 0..lanes {
        let src = if i < 4 { slo } else { shi };
        let f = f32::from_bits((src >> ((i % 4) * 32)) as u32);
        let h = f32_to_f16(f, rc) as u128;
        out |= h << (i * 16);
    }
    out
}

/// F16C applies DAZ to single-precision sources (SDM Vol 1 §10.2.3.4),
/// while its instruction-specific rule ignores FTZ (Vol 2C §5.2 VCVTPS2PH).
/// `rc` is already resolved from imm8/MXCSR by the caller.
pub fn cvtps2ph_with_mxcsr(slo: u128, shi: u128, lanes: usize, rc: u8, mxcsr: u32) -> u128 {
    cvtps2ph(
        daz_lanes(slo, FPrec::F32, mxcsr),
        daz_lanes(shi, FPrec::F32, mxcsr),
        lanes,
        rc,
    )
}

/// SSE4.1 `phminposuw`: minimum of the eight unsigned 16-bit words → word 0,
/// its (lowest) index → word 1, bits 127:32 zeroed.
pub fn phminposuw(src: u128) -> u128 {
    let mut min = u16::MAX;
    let mut idx = 0u16;
    for i in 0..8 {
        let w = (src >> (i * 16)) as u16;
        if w < min {
            min = w;
            idx = i as u16;
        }
    }
    (min as u128) | ((idx as u128) << 16)
}

/// SSE4.1 `mpsadbw` over one 128-bit lane: `imm[2]` picks the src1 byte offset
/// (`0`/`4`), `imm[1:0]` the src2 dword offset (`0..3` → byte offset `0/4/8/12`). Produces
/// eight unsigned 16-bit sums of absolute byte differences of 4-byte windows.
pub fn mpsadbw(a: u128, b: u128, imm: u8) -> u128 {
    let ab = a.to_le_bytes();
    let bb = b.to_le_bytes();
    let a_off = ((imm >> 2) & 1) as usize * 4;
    let b_off = (imm & 3) as usize * 4;
    let mut out = 0u128;
    for i in 0..8usize {
        let mut sum: u16 = 0;
        for j in 0..4usize {
            let x = ab[a_off + i + j] as i32;
            let y = bb[b_off + j] as i32;
            sum += (x - y).unsigned_abs() as u16;
        }
        out |= (sum as u128) << (i * 16);
    }
    out
}

/// Valid element count for an implicit-length string (`pcmpistri`): the index of the
/// first null element, or `n` if none.
fn pcmpistr_len(v: u128, words: bool) -> usize {
    let (n, ew, mask) = if words {
        (8usize, 2usize, 0xFFFFu128)
    } else {
        (16, 1, 0xFF)
    };
    (0..n)
        .position(|i| (v >> (i * ew * 8)) & mask == 0)
        .unwrap_or(n)
}

/// SSE4.2 `pcmpistri`/`pcmpestri`: run the aggregation over `xmm[a]` and
/// `xmm[b]`, returning `(ecx, cf, zf, sf, of)`. For the explicit form the lengths come
/// from EAX/EDX; otherwise from the first null element. Read-only — the interpreter arm
/// and the JIT helper write ECX/flags through their own state machinery.
pub fn pcmpstr_run(
    cpu: &CpuState,
    a: u8,
    b: u8,
    imm: u8,
    explicit: bool,
) -> (u32, bool, bool, bool, bool) {
    pcmpstr_run_bv(cpu, a, cpu.xmm[b as usize], imm, explicit)
}

/// As [`pcmpstr_run`] but source 2 is supplied as a value (`bv`) rather than read from
/// `cpu.xmm[b]` — the memory-source `pcmpistri/pcmpestri` form. Source 1 is
/// still `cpu.xmm[a]`; the explicit-length EAX/EDX read is unchanged.
pub fn pcmpstr_run_bv(
    cpu: &CpuState,
    a: u8,
    bv: u128,
    imm: u8,
    explicit: bool,
) -> (u32, bool, bool, bool, bool) {
    let av = cpu.xmm[a as usize];
    let (len1, len2) = pcmpstr_lengths(cpu, av, bv, imm, explicit);
    pcmpstr(av, bv, len1, len2, imm)
}

/// Valid element counts `(len1, len2)` for a `pcmpstr`-family run: from EAX/EDX for the
/// explicit form (`pcmp*e*`), else from the first null element in each source.
fn pcmpstr_lengths(cpu: &CpuState, av: u128, bv: u128, imm: u8, explicit: bool) -> (usize, usize) {
    let words = imm & 1 != 0;
    let n = if words { 8 } else { 16 };
    if explicit {
        let eax = cpu.gpr[0] as u32 as i32;
        let edx = cpu.gpr[2] as u32 as i32;
        (
            (eax.unsigned_abs() as usize).min(n),
            (edx.unsigned_abs() as usize).min(n),
        )
    } else {
        (pcmpistr_len(av, words), pcmpistr_len(bv, words))
    }
}

/// SSE4.2 `pcmpistrm`/`pcmpestrm`: run the aggregation over `xmm[a]` and
/// `xmm[b]`, returning `(mask, cf, zf, sf, of)` — the mask goes to XMM0. Read-only; the
/// interpreter arm and JIT helper write XMM0/flags through their own state machinery.
pub fn pcmpstrm_run(
    cpu: &CpuState,
    a: u8,
    b: u8,
    imm: u8,
    explicit: bool,
) -> (u128, bool, bool, bool, bool) {
    pcmpstrm_run_bv(cpu, a, cpu.xmm[b as usize], imm, explicit)
}

/// As [`pcmpstrm_run`] but source 2 is supplied as a value (`bv`) — the memory-source form.
pub fn pcmpstrm_run_bv(
    cpu: &CpuState,
    a: u8,
    bv: u128,
    imm: u8,
    explicit: bool,
) -> (u128, bool, bool, bool, bool) {
    let av = cpu.xmm[a as usize];
    let (len1, len2) = pcmpstr_lengths(cpu, av, bv, imm, explicit);
    pcmpstr_mask(av, bv, len1, len2, imm)
}

/// EVEX `valign{d,q}`: shift the concatenation `a:b` (a high, b low) right
/// by `shift` elements of `elem` bytes, and return the low `bytes` as 128-bit lanes.
fn valign_lanes(a: [u128; 4], b: [u128; 4], shift: u8, elem: u8, bytes: u16) -> [u128; 4] {
    let bytes = bytes as usize;
    // 2*bytes-byte buffer: low half = b, high half = a.
    let mut buf = [0u8; 128];
    for (i, chunk) in b.iter().enumerate().take(bytes / 16) {
        buf[i * 16..i * 16 + 16].copy_from_slice(&chunk.to_le_bytes());
    }
    for (i, chunk) in a.iter().enumerate().take(bytes / 16) {
        buf[bytes + i * 16..bytes + i * 16 + 16].copy_from_slice(&chunk.to_le_bytes());
    }
    let total_elems = (2 * bytes) / elem as usize;
    let shift_bytes = (shift as usize % total_elems) * elem as usize;
    let mut out = [0u128; 4];
    for (i, slot) in out.iter_mut().enumerate().take(bytes / 16) {
        let mut b16 = [0u8; 16];
        b16.copy_from_slice(&buf[shift_bytes + i * 16..shift_bytes + i * 16 + 16]);
        *slot = u128::from_le_bytes(b16);
    }
    out
}

/// `valign` for the JIT helper: shift-and-write into `dst`.
#[allow(clippy::too_many_arguments)]
pub fn exec_valign(cpu: &mut CpuState, dst: u8, a: u8, b: u8, shift: u8, elem: u8, bytes: u16) {
    let r = valign_lanes(
        cpu.vec_lanes(a as usize),
        cpu.vec_lanes(b as usize),
        shift,
        elem,
        bytes,
    );
    cpu.set_vec(dst as usize, r, bytes);
}

/// `vpermt2{b,w,d,q}`: two-table cross-lane permute — the ONE implementation
/// shared by the interpreter and the JIT helper. `idx` holds per-lane indices into the
/// `2*n` lanes of {table0 = `dst`'s old value, table1 = `tbl`}; the result overwrites
/// `dst` under the mask. `masked` selects the write-masked (merge/zero) vs full write.
#[allow(clippy::too_many_arguments)]
pub fn exec_vpermt2(
    cpu: &mut CpuState,
    dst: u8,
    idx: u8,
    tbl: u8,
    elem: u8,
    k: u8,
    masked: bool,
    zeroing: bool,
    bytes: u16,
    imode: bool,
) {
    // vpermt2: index = idx operand, table0 = old dst. vpermi2: index = old dst, table0 =
    // idx operand. Table1 = tbl register in both; the result overwrites dst.
    let (index, table0) = if imode {
        (cpu.vec_lanes(dst as usize), cpu.vec_lanes(idx as usize))
    } else {
        (cpu.vec_lanes(idx as usize), cpu.vec_lanes(dst as usize))
    };
    let table1 = cpu.vec_lanes(tbl as usize);
    let res = permute2(&index, &table0, &table1, elem, bytes);
    if masked {
        cpu.write_masked(dst as usize, res, k, elem, zeroing, bytes);
    } else {
        cpu.set_vec(dst as usize, res, bytes);
    }
}

/// Single-source cross-lane permute `vperm{d,q}` (vector-index): the whole
/// register is one table; `dst[i] = src[idx[i] & (n-1)]`. Masked/zeroing per the write-
/// mask. Shared by interp and the JIT helper → jit == interp.
#[allow(clippy::too_many_arguments)]
pub fn exec_vperm1(
    cpu: &mut CpuState,
    dst: u8,
    idx: u8,
    src: u8,
    elem: u8,
    bytes: u16,
    k: u8,
    masked: bool,
    zeroing: bool,
) {
    let index = cpu.vec_lanes(idx as usize);
    let table = cpu.vec_lanes(src as usize);
    let res = permute1(&index, &table, elem, bytes);
    if masked {
        cpu.write_masked(dst as usize, res, k, elem, zeroing, bytes);
    } else {
        cpu.set_vec(dst as usize, res, bytes);
    }
}

/// Load one 16-byte permute-table lane, or the read fault that stopped it.
///
/// Guest memory is reached through an 8-byte-max accessor, so a 128-bit lane is two
/// `sload`s and **either** can fault — the returned [`StrFault`] names the half that
/// actually failed, which is what lets the guest resume at the right address. `rip` is
/// rewound to the faulting instruction first: these run mid-instruction, and a fault
/// must not leave `rip` past an instruction that never completed.
fn load_lane16<M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    ea: u64,
    cur_addr: u64,
) -> Result<u128, StrFault> {
    let half = |addr: u64, cpu: &mut CpuState| {
        vpart(cpu, mem, cur_addr, addr, 8).map_err(|f| f.into_str_fault(cpu, cur_addr, false))
    };
    let lo = half(ea, cpu)?;
    let hi = half(ea + 8, cpu)?;
    Ok((lo as u128) | ((hi as u128) << 64))
}

/// Memory-source single-table permute `vperm{d,q} v, idx, [mem]`: the table is
/// loaded from `[base]` rather than a register. Generic over [`StrMem`] so interp and the
/// JIT helper share it → jit == interp. A load fault stops before any register write.
#[allow(clippy::too_many_arguments)]
pub fn vperm1_run<M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    dst: u8,
    idx: u8,
    base: u64,
    elem: u8,
    k: u8,
    masked: bool,
    zeroing: bool,
    bytes: u16,
    cur_addr: u64,
) -> Option<StrFault> {
    let mut table = [0u128; 4];
    for (i, slot) in table.iter_mut().enumerate().take(bytes as usize / 16) {
        let ea = base.wrapping_add(i as u64 * 16);
        *slot = match load_lane16(cpu, mem, ea, cur_addr) {
            Ok(v) => v,
            Err(f) => return Some(f),
        };
    }
    let index = cpu.vec_lanes(idx as usize);
    let res = permute1(&index, &table, elem, bytes);
    if masked {
        cpu.write_masked(dst as usize, res, k, elem, zeroing, bytes);
    } else {
        cpu.set_vec(dst as usize, res, bytes);
    }
    None
}

/// SDM Vol 2C §5.2, VPERMD/VPERMPS: each index selects across the whole table.
fn permute1(index: &[u128; 4], table: &[u128; 4], elem: u8, bytes: u16) -> [u128; 4] {
    let n = usize::from(bytes) / usize::from(elem);
    let mut res = [0; 4];
    for i in 0..n {
        let id = get_velem(index, i, elem) as usize & (n - 1);
        set_velem(&mut res, i, elem, get_velem(table, id, elem));
    }
    res
}

/// Two-table cross-lane permute core (shared by `vpermt2`/`vpermi2`, reg + memory src):
/// for each of the `bytes/elem` lanes, `index[i]` (masked to `log2(2n)` bits) selects a
/// lane from the concatenation `table0:table1`.
fn permute2(
    index: &[u128; 4],
    table0: &[u128; 4],
    table1: &[u128; 4],
    elem: u8,
    bytes: u16,
) -> [u128; 4] {
    let n = bytes as usize / elem as usize;
    let sel = 2 * n - 1;
    let mut res = [0u128; 4];
    for i in 0..n {
        let id = get_velem(index, i, elem) as usize & sel;
        let v = if id < n {
            get_velem(table0, id, elem)
        } else {
            get_velem(table1, id - n, elem)
        };
        set_velem(&mut res, i, elem, v);
    }
    res
}

/// Memory-source `vpermt2`/`vpermi2`: table 1 is loaded from `[base]` rather
/// than a register. Generic over [`StrMem`] so interp (`Memory`) and the JIT helper
/// (`RawStrMem`) share it → jit == interp. A load fault stops before any write.
#[allow(clippy::too_many_arguments)]
pub fn permute2_run<M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    dst: u8,
    idx: u8,
    base: u64,
    elem: u8,
    k: u8,
    masked: bool,
    zeroing: bool,
    bytes: u16,
    imode: bool,
    cur_addr: u64,
) -> Option<StrFault> {
    // Load table1 from memory first (fault before any register write).
    let mut table1 = [0u128; 4];
    for (i, slot) in table1.iter_mut().enumerate().take(bytes as usize / 16) {
        let ea = base.wrapping_add(i as u64 * 16);
        *slot = match load_lane16(cpu, mem, ea, cur_addr) {
            Ok(v) => v,
            Err(f) => return Some(f),
        };
    }
    let (index, table0) = if imode {
        (cpu.vec_lanes(dst as usize), cpu.vec_lanes(idx as usize))
    } else {
        (cpu.vec_lanes(idx as usize), cpu.vec_lanes(dst as usize))
    };
    let res = permute2(&index, &table0, &table1, elem, bytes);
    if masked {
        cpu.write_masked(dst as usize, res, k, elem, zeroing, bytes);
    } else {
        cpu.set_vec(dst as usize, res, bytes);
    }
    None
}

/// Masked EVEX logic: compute `op(a, b)` per 128-bit lane, then write it
/// into `dst` under opmask `k` at `elem` granularity (merge or zeroing).
#[allow(clippy::too_many_arguments)]
fn apply_masked_logic(
    cpu: &mut CpuState,
    op: VLogicOp,
    dst: u8,
    a: u8,
    b: u8,
    k: u8,
    elem: u8,
    zeroing: bool,
    bytes: u16,
) {
    let (al, bl) = (cpu.vec_lanes(a as usize), cpu.vec_lanes(b as usize));
    let mut r = [0u128; 4];
    for i in 0..4 {
        r[i] = vlogic(al[i], bl[i], op);
    }
    cpu.write_masked(dst as usize, r, k, elem, zeroing, bytes);
}

/// EVEX `vpshufb`: per-128-bit-lane byte shuffle `dst = pshufb(a, idx)` over
/// `bytes`, byte-granularity masked. Shared by interp and the JIT helper → jit == interp.
#[allow(clippy::too_many_arguments)]
pub fn exec_vpshufb_wide(
    cpu: &mut CpuState,
    dst: u8,
    a: u8,
    idx: u8,
    bytes: u16,
    k: u8,
    masked: bool,
    zeroing: bool,
) {
    let av = cpu.vec_lanes(a as usize);
    let iv = cpu.vec_lanes(idx as usize);
    let mut res = [0u128; 4];
    for l in 0..(bytes as usize / 16) {
        res[l] = pshufb(av[l], iv[l]);
    }
    if masked {
        cpu.write_masked(dst as usize, res, k, 1, zeroing, bytes);
    } else {
        cpu.set_vec(dst as usize, res, bytes);
    }
}

/// EVEX/VEX-256 `vpshufd`: per-128-bit-lane dword shuffle by `imm8` over
/// `bytes`, dword-granularity masked. Shared by interp and the JIT helper → jit == interp.
#[allow(clippy::too_many_arguments)]
pub fn exec_vshuffle32_wide(
    cpu: &mut CpuState,
    dst: u8,
    a: u8,
    imm: u8,
    bytes: u16,
    k: u8,
    masked: bool,
    zeroing: bool,
) {
    let av = cpu.vec_lanes(a as usize);
    let mut res = [0u128; 4];
    for l in 0..(bytes as usize / 16) {
        let v = av[l];
        let mut r = 0u128;
        for i in 0..4 {
            let sel = (imm >> (2 * i)) & 3;
            let lane = (v >> (sel as u32 * 32)) & 0xffff_ffff;
            r |= lane << (i * 32);
        }
        res[l] = r;
    }
    if masked {
        cpu.write_masked(dst as usize, res, k, 4, zeroing, bytes);
    } else {
        cpu.set_vec(dst as usize, res, bytes);
    }
}

/// Masked EVEX packed arithmetic: compute `packed_bin` per 128-bit chunk
/// then merge/zero-mask under `k` at `elem` granularity. Shared by the interpreter and
/// the JIT helper (`exec_masked_packed`) so jit == interp.
#[allow(clippy::too_many_arguments)]
fn apply_masked_packed(
    cpu: &mut CpuState,
    op: PackedBinOp,
    dst: u8,
    a: u8,
    b: u8,
    k: u8,
    elem: u8,
    zeroing: bool,
    bytes: u16,
) {
    let (al, bl) = (cpu.vec_lanes(a as usize), cpu.vec_lanes(b as usize));
    let mut r = [0u128; 4];
    for i in 0..4 {
        r[i] = packed_bin(al[i], bl[i], elem, op);
    }
    cpu.write_masked(dst as usize, r, k, elem, zeroing, bytes);
}

/// EVEX masked packed arithmetic entry for the JIT helper. `op_code`
/// mirrors the codegen encoding; delegates to the same [`apply_masked_packed`] the
/// interpreter uses, guaranteeing jit == interp.
#[allow(clippy::too_many_arguments)]
pub fn exec_masked_packed(
    cpu: &mut CpuState,
    op_code: u8,
    dst: u8,
    a: u8,
    b: u8,
    k: u8,
    elem: u8,
    zeroing: bool,
    bytes: u16,
) {
    let op = match op_code {
        0 => PackedBinOp::Add,
        1 => PackedBinOp::Sub,
        2 => PackedBinOp::MinU,
        3 => PackedBinOp::MaxU,
        4 => PackedBinOp::MinS,
        5 => PackedBinOp::MaxS,
        6 => PackedBinOp::MulLo32,
        7 => PackedBinOp::CmpEq,
        8 => PackedBinOp::CmpGt,
        9 => PackedBinOp::MulLo64,
        10 => PackedBinOp::MulU32,
        11 => PackedBinOp::MulS32,
        12 => PackedBinOp::MulLo16,
        13 => PackedBinOp::MulHiU16,
        14 => PackedBinOp::MulHiS16,
        15 => PackedBinOp::AddSatS,
        16 => PackedBinOp::AddSatU,
        17 => PackedBinOp::SubSatS,
        18 => PackedBinOp::SubSatU,
        19 => PackedBinOp::AvgU,
        _ => PackedBinOp::MulHiRoundedS16,
    };
    apply_masked_packed(cpu, op, dst, a, b, k, elem, zeroing, bytes);
}

/// EVEX packed shift-by-imm over any width with optional write-masking.
/// Computes the full unmasked shift per 128-bit lane, then commits: unmasked (`k == 0`)
/// clears above `bytes` via [`CpuState::set_vec`]; masked routes through
/// [`CpuState::write_masked`] for the merge/zero rule. Shared by interp + JIT.
#[allow(clippy::too_many_arguments)]
fn apply_masked_shift(
    cpu: &mut CpuState,
    dst: u8,
    a: u8,
    imm: u8,
    elem: u8,
    right: bool,
    arith: bool,
    k: u8,
    zeroing: bool,
    bytes: u16,
) {
    let al = cpu.vec_lanes(a as usize);
    let mut r = [0u128; 4];
    for i in 0..4 {
        r[i] = packed_shift(al[i], imm, elem, right, arith);
    }
    if k == 0 {
        cpu.set_vec(dst as usize, r, bytes); // unmasked EVEX: full write, zero-upper
    } else {
        cpu.write_masked(dst as usize, r, k, elem, zeroing, bytes);
    }
}

/// EVEX masked packed shift entry for the JIT helper; delegates to the same
/// [`apply_masked_shift`] the interpreter uses, guaranteeing jit == interp.
#[allow(clippy::too_many_arguments)]
pub fn exec_masked_shift(
    cpu: &mut CpuState,
    dst: u8,
    a: u8,
    imm: u8,
    elem: u8,
    right: bool,
    arith: bool,
    k: u8,
    zeroing: bool,
    bytes: u16,
) {
    apply_masked_shift(cpu, dst, a, imm, elem, right, arith, k, zeroing, bytes);
}

/// Packed shift by a scalar register count `vp{sll,srl,sra}{w,d,q} v,v,xmm`: the low 64
/// bits of `count`'s xmm shift every lane uniformly. A count ≥ the lane width is
/// clamped to the width so the shared `packed_shift` over-shift path yields 0 / sign-smear.
#[allow(clippy::too_many_arguments)]
pub fn exec_shift_reg(
    cpu: &mut CpuState,
    dst: u8,
    a: u8,
    count: u8,
    elem: u8,
    right: bool,
    arith: bool,
    k: u8,
    zeroing: bool,
    bytes: u16,
) {
    let cnt = cpu.xmm[count as usize] as u64; // low 64 bits = the uniform shift amount
    let bits = elem as u64 * 8;
    let eff = cnt.min(bits) as u8; // ≥ width → over-shift (packed_shift returns 0 / sign)
                                   // 128-bit unmasked form: the legacy-SSE `psll* xmm,xmm`
                                   // must leave bits 255:128 intact, so write `cpu.xmm[dst]`
                                   // directly rather than `set_vec` (which zeroes the upper
                                   // halves). The VEX/EVEX form's upper clear comes from a
                                   // `VZeroUpper` the lifter appends. Masked/256/512 forms
                                   // keep the width-aware `apply_masked_shift`.
    if k == 0 && bytes == 16 {
        cpu.xmm[dst as usize] = packed_shift(cpu.xmm[a as usize], eff, elem, right, arith);
        return;
    }
    apply_masked_shift(cpu, dst, a, eff, elem, right, arith, k, zeroing, bytes);
}

/// AVX2/AVX-512 per-element variable shift `vp{sll,srl,sra}v{w,d,q}`: shift each
/// `elem`-byte lane of `a` by the count in the matching lane of `count`, then merge/zero
/// under `k` (`k == 0` = unmasked full-width write). Shared by interp and the JIT helper.
#[allow(clippy::too_many_arguments)]
pub fn exec_var_shift(
    cpu: &mut CpuState,
    dst: u8,
    a: u8,
    count: u8,
    elem: u8,
    right: bool,
    arith: bool,
    k: u8,
    zeroing: bool,
    bytes: u16,
) {
    let al = cpu.vec_lanes(a as usize);
    let cl = cpu.vec_lanes(count as usize);
    let mut r = [0u128; 4];
    for i in 0..usize::from(bytes) / 16 {
        r[i] = var_shift_lane(al[i], cl[i], elem, right, arith);
    }
    if k == 0 {
        cpu.set_vec(dst as usize, r, bytes); // unmasked EVEX: full write, zero-upper
    } else {
        cpu.write_masked(dst as usize, r, k, elem, zeroing, bytes);
    }
}

/// GFNI `gf2p8{mulb,affineqb,affineinvqb}` wide/masked: per-128-bit-lane GF(2⁸)
/// op over `bytes` (reusing the shared [`GfniOp::apply`] math), then merge/zero under `k`
/// (byte-granular; `k == 0` = unmasked). Shared by interp and the JIT helper → jit == interp.
#[allow(clippy::too_many_arguments)]
pub fn exec_gf2p8(
    cpu: &mut CpuState,
    dst: u8,
    a: u8,
    b: u8,
    imm: u8,
    mode: u8,
    k: u8,
    zeroing: bool,
    bytes: u16,
) {
    let op = crate::ir::GfniOp::from_u8(mode);
    let al = cpu.vec_lanes(a as usize);
    let bl = cpu.vec_lanes(b as usize);
    let mut r = [0u128; 4];
    for lane in 0..(bytes as usize / 16) {
        r[lane] = op.apply(al[lane], bl[lane], imm);
    }
    if k == 0 {
        cpu.set_vec(dst as usize, r, bytes);
    } else {
        cpu.write_masked(dst as usize, r, k, 1, zeroing, bytes); // byte-granular mask
    }
}

/// As [`exec_gf2p8`] but the second source (the affine matrix / multiplier) is a memory
/// operand at `addr`: read each 128-bit lane from guest memory, then apply the
/// GF(2⁸) op. Shared by the interpreter and the JIT's `gf2p8_mem` helper via [`StrMem`], so
/// the `dst == src1` aliasing case (openssl's `vgf2p8affineqb ymm,ymm,[mem]`) is exact
/// without a scratch register. Returns `Some(fault)` on an unmapped load.
#[allow(clippy::too_many_arguments)]
pub fn gf2p8_mem_run<M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    dst: u8,
    a: u8,
    addr: u64,
    imm: u8,
    mode: u8,
    k: u8,
    zeroing: bool,
    bytes: u16,
) -> Option<StrFault> {
    let op = crate::ir::GfniOp::from_u8(mode);
    let al = cpu.vec_lanes(a as usize);
    let mut r = [0u128; 4];
    let mut bl = [0u128; 4];
    if k != 0 && matches!(op, crate::ir::GfniOp::Mulb) {
        // SDM Vol 1 §15.6.1; Vol 2A §2.8.4 and GF2P8MULB, Description:
        // masked multiply suppresses memory faults at byte granularity.
        for i in 0..bytes as usize {
            if cpu.kmask[k as usize] & (1 << i) != 0 {
                let ea = addr.wrapping_add(i as u64);
                let v = match vpart(cpu, mem, cpu.rip, ea, 1) {
                    Ok(v) => v,
                    Err(f) => return Some(f.into_str_fault(cpu, cpu.rip, false)),
                };
                set_velem(&mut bl, i, 1, v);
            }
        }
    } else {
        // SDM Vol 2A, GF2P8AFFINEQB/GF2P8AFFINEINVQB, Description:
        // affine forms do not support memory fault suppression.
        for (lane, slot) in bl.iter_mut().enumerate().take(bytes as usize / 16) {
            let ea = addr.wrapping_add(lane as u64 * 16);
            *slot = match load_lane16(cpu, mem, ea, cpu.rip) {
                Ok(v) => v,
                Err(f) => return Some(f),
            };
        }
    }
    for lane in 0..bytes as usize / 16 {
        r[lane] = op.apply(al[lane], bl[lane], imm);
    }
    if k == 0 {
        cpu.set_vec(dst as usize, r, bytes);
    } else {
        cpu.write_masked(dst as usize, r, k, 1, zeroing, bytes);
    }
    None
}

/// One element of a variable shift: `av` (low `bits` bits significant) shifted by `cnt`.
/// A count ≥ `bits` yields 0 (logical/left) or the smeared sign (arithmetic right).
/// One variable-shift lane. SDM Vol 2C §5.2, VPSLLV/VPSRLV/VPSRAV, Operation.
pub fn var_shift_lane(a: u128, count: u128, elem: u8, right: bool, arith: bool) -> u128 {
    let mut r = 0;
    let bits = u32::from(elem) * 8;
    let mask = if elem == 8 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    for i in 0..16 / u32::from(elem) {
        let av = (a >> (i * bits)) as u64 & mask;
        let cv = (count >> (i * bits)) as u64 & mask;
        r |= u128::from(var_shift_one(av, cv, bits, right, arith)) << (i * bits);
    }
    r
}

fn var_shift_one(av: u64, cnt: u64, bits: u32, right: bool, arith: bool) -> u64 {
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let av = av & mask;
    if cnt >= bits as u64 {
        return if right && arith && (av >> (bits - 1)) & 1 != 0 {
            mask
        } else {
            0
        };
    }
    let c = cnt as u32;
    if !right {
        (av << c) & mask
    } else if !arith {
        av >> c
    } else {
        let se = sign_extend_128(av as u128, bits as u8); // i128, sign-extended lane
        ((se >> c) as u64) & mask
    }
}

/// EVEX narrowing move `vpmov{q,d,w}{d,w,b}`: truncate each of the
/// `src_width/from` source lanes to its low `to` bytes, packing them contiguously into
/// dst lanes `0..n`; bits above the packed result are zeroed (EVEX dest). Masking is at
/// `to` granularity over the `n` result lanes — masked-off lanes keep the old dst (merge)
/// or clear (zeroing); `write_masked` can't be reused because a sub-16-byte result has
/// zero 128-bit chunks. Shared by interp and the JIT helper → jit == interp.
#[allow(clippy::too_many_arguments)]
pub fn exec_vpmov_narrow(
    cpu: &mut CpuState,
    dst: u8,
    src: u8,
    from: u8,
    to: u8,
    src_width: u16,
    k: u8,
    masked: bool,
    zeroing: bool,
) {
    let n = src_width as usize / from as usize;
    let s = cpu.vec_lanes(src as usize);
    let old = cpu.vec_lanes(dst as usize);
    let kmask = if masked {
        cpu.kmask[k as usize]
    } else {
        u64::MAX
    };
    let mut res = [0u128; 4];
    for i in 0..n {
        // set_velem masks to `to` bytes, so the wide source lane is truncated on write.
        let out = if (kmask >> i) & 1 != 0 {
            get_velem(&s, i, from)
        } else if zeroing {
            0
        } else {
            get_velem(&old, i, to) // merge: keep the old dst element
        };
        set_velem(&mut res, i, to, out);
    }
    // Lanes above the packed result are always zeroed → store the full 512-bit register.
    cpu.set_vec(dst as usize, res, 64);
}

/// EVEX/VEX-256 widening move `vpmov{s,z}x*` to a wide (or masked) dest:
/// zero/sign-extend each of the `dst_width/to` low `from`-byte source lanes to `to` bytes.
/// Masking is at `to` granularity over the result lanes; bits above the result (to `VL`)
/// are zeroed. Shared by interp and the JIT helper → jit == interp.
#[allow(clippy::too_many_arguments)]
pub fn exec_vpmov_extend_wide(
    cpu: &mut CpuState,
    dst: u8,
    src: u8,
    from: u8,
    to: u8,
    signed: bool,
    dst_width: u16,
    k: u8,
    masked: bool,
    zeroing: bool,
) {
    let n = dst_width as usize / to as usize;
    let s = cpu.vec_lanes(src as usize);
    let old = cpu.vec_lanes(dst as usize);
    let kmask = if masked {
        cpu.kmask[k as usize]
    } else {
        u64::MAX
    };
    let bits = from as u32 * 8;
    let mut res = [0u128; 4];
    for i in 0..n {
        let raw = get_velem(&s, i, from);
        // Sign-extend within a u64 when signed; set_velem then masks to `to` bytes.
        let ext = if signed && bits < 64 && (raw & (1u64 << (bits - 1))) != 0 {
            raw | (u64::MAX << bits)
        } else {
            raw
        };
        let out = if (kmask >> i) & 1 != 0 {
            ext
        } else if zeroing {
            0
        } else {
            get_velem(&old, i, to) // merge: keep the old dst element
        };
        set_velem(&mut res, i, to, out);
    }
    cpu.set_vec(dst as usize, res, dst_width);
}

/// Packed absolute value `vpabs{b,w,d,q}`: per `elem`-byte lane `dst = |src|`
/// (signed; `abs(MIN)` wraps to `MIN`, matching x86). Masking at `elem` granularity; bits
/// above the result (to `VL`) are zeroed. Shared by interp and the JIT helper.
#[allow(clippy::too_many_arguments)]
pub fn exec_vpabs(
    cpu: &mut CpuState,
    dst: u8,
    src: u8,
    elem: u8,
    dst_width: u16,
    k: u8,
    masked: bool,
    zeroing: bool,
) {
    let n = dst_width as usize / elem as usize;
    let s = cpu.vec_lanes(src as usize);
    let old = cpu.vec_lanes(dst as usize);
    let kmask = if masked {
        cpu.kmask[k as usize]
    } else {
        u64::MAX
    };
    let bits = elem as u32 * 8;
    let mut res = [0u128; 4];
    for i in 0..n {
        let raw = get_velem(&s, i, elem);
        // Sign-extend to i64, take absolute value (wrapping so |MIN| == MIN), then mask.
        let sext = if bits < 64 && (raw & (1u64 << (bits - 1))) != 0 {
            raw | (u64::MAX << bits)
        } else {
            raw
        };
        let abs = (sext as i64).wrapping_abs() as u64;
        let out = if (kmask >> i) & 1 != 0 {
            abs
        } else if zeroing {
            0
        } else {
            get_velem(&old, i, elem) // merge: keep the old dst element
        };
        set_velem(&mut res, i, elem, out);
    }
    cpu.set_vec(dst as usize, res, dst_width);
}

/// Masked EVEX unary lane op `vplzcnt{d,q}` / `vprol{d,q}` / `vpconflict{d,q}`:
/// per `elem`-byte lane `dst = f(src)`, masked at `elem` granularity, bits above `VL`
/// zeroed. `op` selects the lane function; `imm` is the rotate count (`vprol` only).
/// Shared by interp and the JIT helper.
#[allow(clippy::too_many_arguments)]
pub fn exec_vp_unary_lane(
    cpu: &mut CpuState,
    dst: u8,
    src: u8,
    op: crate::ir::VpUnaryOp,
    imm: u8,
    elem: u8,
    dst_width: u16,
    k: u8,
    masked: bool,
    zeroing: bool,
) {
    use crate::ir::VpUnaryOp;
    let n = dst_width as usize / elem as usize;
    let s = cpu.vec_lanes(src as usize);
    let old = cpu.vec_lanes(dst as usize);
    let kmask = if masked {
        cpu.kmask[k as usize]
    } else {
        u64::MAX
    };
    let mut res = [0u128; 4];
    for i in 0..n {
        let raw = get_velem(&s, i, elem);
        let val = match op {
            VpUnaryOp::Lzcnt => {
                if elem == 4 {
                    (raw as u32).leading_zeros() as u64
                } else {
                    raw.leading_zeros() as u64
                }
            }
            VpUnaryOp::Rol => {
                if elem == 4 {
                    (raw as u32).rotate_left(imm as u32) as u64
                } else {
                    raw.rotate_left(imm as u32)
                }
            }
            VpUnaryOp::Conflict => {
                // dst[i] = bitmask of lower lanes j<i whose element equals lane i.
                let mut m = 0u64;
                for j in 0..i {
                    if get_velem(&s, j, elem) == raw {
                        m |= 1u64 << j;
                    }
                }
                m
            }
        };
        let out = if (kmask >> i) & 1 != 0 {
            val
        } else if zeroing {
            0
        } else {
            get_velem(&old, i, elem) // merge: keep the old dst element
        };
        set_velem(&mut res, i, elem, out);
    }
    cpu.set_vec(dst as usize, res, dst_width);
}

/// Masked EVEX blend `vpblendm{d,q}`: per `elem`-byte lane
/// `dst[i] = k[i] ? b[i] : (zeroing ? 0 : a[i])`, bits above `VL` zeroed. The opmask `k`
/// is the blend control. Shared by interp and the JIT helper.
#[allow(clippy::too_many_arguments)]
pub fn exec_vp_blendm(
    cpu: &mut CpuState,
    dst: u8,
    a: u8,
    b: u8,
    k: u8,
    elem: u8,
    dst_width: u16,
    zeroing: bool,
) {
    let n = dst_width as usize / elem as usize;
    let av = cpu.vec_lanes(a as usize);
    let bv = cpu.vec_lanes(b as usize);
    // SDM Vol 1 §15.6.1.1: encoded k0 selects the implicit all-ones mask.
    let kmask = if k == 0 {
        u64::MAX
    } else {
        cpu.kmask[k as usize]
    };
    let mut res = [0u128; 4];
    for i in 0..n {
        let out = if (kmask >> i) & 1 != 0 {
            get_velem(&bv, i, elem)
        } else if zeroing {
            0
        } else {
            get_velem(&av, i, elem)
        };
        set_velem(&mut res, i, elem, out);
    }
    cpu.set_vec(dst as usize, res, dst_width);
}

/// Masked EVEX 128-bit-lane shuffle `vshuff32x4` / `vshuff64x2`: imm8 selects
/// whole 128-bit lanes — low half of dst from `a`, high half from `b` — then masked at
/// `elem` granularity. Shared by interp and the JIT helper.
#[allow(clippy::too_many_arguments)]
pub fn exec_vshuf_lane(
    cpu: &mut CpuState,
    dst: u8,
    a: u8,
    b: u8,
    imm: u8,
    elem: u8,
    dst_width: u16,
    k: u8,
    masked: bool,
    zeroing: bool,
) {
    let nlanes = dst_width as usize / 16; // 128-bit lanes (2 for 256, 4 for 512)
    let bits_per = nlanes.trailing_zeros(); // 1 bit/field for 2 lanes, 2 bits for 4 lanes
    let sel_mask = (nlanes as u8) - 1;
    let al = cpu.vec_lanes(a as usize);
    let bl = cpu.vec_lanes(b as usize);
    let mut shuf = [0u128; 4];
    for (i, slot) in shuf.iter_mut().enumerate().take(nlanes) {
        let field = ((imm >> (i as u32 * bits_per)) & sel_mask) as usize;
        *slot = if i < nlanes / 2 {
            al[field] // low half of dst comes from src1
        } else {
            bl[field] // high half from src2
        };
    }
    if masked {
        cpu.write_masked(dst as usize, shuf, k, elem, zeroing, dst_width);
    } else {
        cpu.set_vec(dst as usize, shuf, dst_width);
    }
}

/// Masked EVEX `vpmultishiftqb` (AVX512-VBMI): for each qword `q`, output byte
/// `i` = `data.qword[q]` rotated right by `(ctrl.qword[q].byte[i] & 63)`, low 8 bits.
/// Masked at byte granularity. Shared by interp and the JIT helper.
#[allow(clippy::too_many_arguments)]
pub fn exec_vp_multishift(
    cpu: &mut CpuState,
    dst: u8,
    ctrl: u8,
    data: u8,
    dst_width: u16,
    k: u8,
    masked: bool,
    zeroing: bool,
) {
    let nq = dst_width as usize / 8; // number of qwords
    let cl = cpu.vec_lanes(ctrl as usize);
    let dl = cpu.vec_lanes(data as usize);
    let mut res = [0u128; 4];
    for q in 0..nq {
        let cq = get_velem(&cl, q, 8); // control qword
        let dq = get_velem(&dl, q, 8); // data qword
        let mut outq = 0u64;
        for i in 0..8 {
            let sh = ((cq >> (i * 8)) & 0x3f) as u32; // control byte i, low 6 bits
            let byte = dq.rotate_right(sh) as u8; // 8 bits starting at bit `sh`, wrapping
            outq |= (byte as u64) << (i * 8);
        }
        set_velem(&mut res, q, 8, outq);
    }
    if masked {
        cpu.write_masked(dst as usize, res, k, 1, zeroing, dst_width);
    } else {
        cpu.set_vec(dst as usize, res, dst_width);
    }
}

/// Masked-EVEX-logic entry for the JIT helper. `op_code`: 0=Xor 1=And
/// 2=Or 3=Andn. Delegates to the same [`apply_masked_logic`] the interpreter uses, so
/// JIT and interpreter share one implementation.
#[allow(clippy::too_many_arguments)]
pub fn exec_masked_logic(
    cpu: &mut CpuState,
    op_code: u8,
    dst: u8,
    a: u8,
    b: u8,
    k: u8,
    elem: u8,
    zeroing: bool,
    bytes: u16,
) {
    let op = match op_code {
        0 => VLogicOp::Xor,
        1 => VLogicOp::And,
        2 => VLogicOp::Or,
        _ => VLogicOp::Andn,
    };
    apply_masked_logic(cpu, op, dst, a, b, k, elem, zeroing, bytes);
}

/// SSE4.1 variable blend: for each `lane`-byte lane, pick it from `s` when the lane's
/// top bit in `mask` is set, else from `d`.
fn blendv(d: u128, s: u128, mask: u128, lane: u8) -> u128 {
    let bits = lane as u32 * 8;
    let lm = lane_mask(lane);
    let mut r = 0u128;
    for i in 0..(16 / lane as u32) {
        let sh = i * bits;
        let pick = if (mask >> (sh + bits - 1)) & 1 == 1 {
            s
        } else {
            d
        };
        r |= ((pick >> sh) & lm) << sh;
    }
    r
}

/// SSE4.1 imm8 static blend `blendps`/`blendpd`: for each lane of `lane` bytes,
/// take it from `b` when `imm8[lane_index]` is set, else from `a`. `blendps` has 4 dword
/// lanes (imm bits[3:0]); `blendpd` has 2 qword lanes (imm bits[1:0]). Shared by interp and
/// the JIT helper → jit == interp.
pub fn blendi(a: u128, b: u128, imm: u8, lane: u8) -> u128 {
    let bits = lane as u32 * 8;
    let lm = lane_mask(lane);
    let mut r = 0u128;
    for i in 0..(16 / lane as u32) {
        let sh = i * bits;
        let pick = if (imm >> i) & 1 == 1 { b } else { a };
        r |= ((pick >> sh) & lm) << sh;
    }
    r
}

/// SSE4.1 `dppd`: double-precision dot product. `imm[5:4]` selects which of the
/// two `a[i]*b[i]` products enter the sum; `imm[1:0]` selects which result qwords receive
/// the broadcast sum (others are zeroed). Evaluated with IEEE f64 arithmetic to match the
/// CPU (rounding), and with the architectural NaN rule spelled out — see
/// [`sse_binop_f64`]. Shared by the interpreter and the JIT helper → jit == interp.
pub fn dppd(a: u128, b: u128, imm: u8) -> u128 {
    dppd_mode(a, b, imm, 0)
}

pub fn dppd_mode(a: u128, b: u128, imm: u8, mxcsr: u32) -> u128 {
    dppd_mode_specialized::<false>(a, b, imm, mxcsr)
}

fn dppd_mode_specialized<const DEFAULT: bool>(a: u128, b: u128, imm: u8, mxcsr: u32) -> u128 {
    let mxcsr = if DEFAULT { 0 } else { mxcsr };
    let lane = |v: u128, i: usize| f64::from_bits((v >> (i * 64)) as u64);
    let mut p = [0.0f64; 2];
    for (i, slot) in p.iter_mut().enumerate() {
        if imm & (0x10 << i) != 0 {
            *slot = apply_f64_mode::<DEFAULT>(lane(a, i), lane(b, i), FloatBinOp::Mul, mxcsr);
        }
    }
    let sum = apply_f64_mode::<DEFAULT>(p[0], p[1], FloatBinOp::Add, mxcsr);
    let mut out = 0u128;
    for i in 0..2 {
        let v = if imm & (1 << i) != 0 { sum } else { 0.0 };
        out |= (v.to_bits() as u128) << (i * 64);
    }
    out
}

/// SSE4.1 `round`: round each lane of `s` per the imm8 `mode` (bits[1:0]: 0 nearest-even,
/// 1 floor, 2 ceil, 3 truncate; bit[2] selects MXCSR.RC). When `scalar`, only
/// lane 0 is rounded and the other lanes of `d` are preserved.
#[cfg(test)]
fn vround(d: u128, s: u128, prec: FPrec, mode: u8, scalar: bool) -> u128 {
    vround_mode::<true>(d, s, prec, mode, scalar, 0)
}

fn vround_mode<const DEFAULT: bool>(
    d: u128,
    s: u128,
    prec: FPrec,
    mode: u8,
    scalar: bool,
    mxcsr: u32,
) -> u128 {
    let mxcsr = if DEFAULT { 0 } else { mxcsr };
    if mxcsr & softfloat::MODE_MASK != 0 {
        let rc = if mode & 4 != 0 {
            (mxcsr >> 13) & 3
        } else {
            u32::from(mode & 3)
        };
        let bits = u32::from(prec.bytes()) * 8;
        let mask = (1u128 << bits) - 1;
        let mut out = d;
        for i in 0..if scalar { 1 } else { 128 / bits } {
            let r = softfloat::round((s >> (i * bits)) as u64, prec == FPrec::F64, rc, mxcsr);
            out = (out & !(mask << (i * bits))) | (u128::from(r) << (i * bits));
        }
        return out;
    }

    let m = if mode & 4 != 0 { 0 } else { mode & 3 };
    let rnd = |f: f64| match m {
        1 => f.floor(),
        2 => f.ceil(),
        3 => f.trunc(),
        _ => round_ties_even(f),
    };
    let mut out = d;
    match prec {
        FPrec::F32 => {
            let count = if scalar { 1 } else { 4 };
            for i in 0..count {
                let raw = (s >> (i * 32)) as u32;
                let x = f32::from_bits(raw);
                // A one-operand SSE op delivers its NaN source operand, converted to a
                // QNaN (SDM Vol 1 §4.8.3.5, Table 4-8, the single-operand rows). Spelled
                // out because the f32→f64→f32 round trip below leaves both the payload
                // and the SNaN→QNaN conversion unspecified in Rust, so the answer came
                // from the host and the optimizer — see `sse_binop_f32` for the same
                // defect on the two-operand side.
                let r = if x.is_nan() {
                    quiet_f32(x)
                } else {
                    rnd(x as f64) as f32
                };
                let mask = 0xFFFF_FFFFu128 << (i * 32);
                out = (out & !mask) | ((r.to_bits() as u128) << (i * 32));
            }
        }
        FPrec::F64 => {
            let count = if scalar { 1 } else { 2 };
            for i in 0..count {
                let raw = (s >> (i * 64)) as u64;
                let x = f64::from_bits(raw);
                // As the f32 arm: the NaN source operand, quieted (SDM Vol 1 Table 4-8).
                let r = if x.is_nan() { quiet_f64(x) } else { rnd(x) };
                let mask = (u64::MAX as u128) << (i * 64);
                out = (out & !mask) | ((r.to_bits() as u128) << (i * 64));
            }
        }
    }
    out
}

/// `vpternlog` bitwise ternary logic: each output bit is `imm8[(a<<2)|(b<<1)|c]` of the
/// three input bits. For each of the 8 index combinations whose `imm` bit is set, OR in
/// the bits where `a`/`b`/`c` match that index's polarity.
fn ternlog(a: u128, b: u128, c: u128, imm: u8) -> u128 {
    let mut r = 0u128;
    for j in 0..8u8 {
        if imm & (1 << j) != 0 {
            let pa = if j & 4 != 0 { a } else { !a };
            let pb = if j & 2 != 0 { b } else { !b };
            let pc = if j & 1 != 0 { c } else { !c };
            r |= pa & pb & pc;
        }
    }
    r
}

/// Replicate the low `elem`-byte element of `low` across all 16 bytes (vpbroadcast).
fn broadcast_elem(low: u128, elem: u8) -> u128 {
    let bits = elem as u32 * 8; // elem ∈ {1,2,4,8} → bits ≤ 64
    let e = low & lane_mask(elem);
    let mut r = 0u128;
    for i in 0..(16 / elem as u32) {
        r |= e << (i * bits);
    }
    r
}

/// Low-`width`-bit mask for opmask ops (`width` ∈ {8,16,32,64}).
fn kwidth_mask(width: u8) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// Evaluate a `vpcmp` predicate (imm8 low 3 bits) against a lane ordering.
fn vpcmp_pred(pred: u8, ord: std::cmp::Ordering) -> bool {
    use std::cmp::Ordering::*;
    match pred & 7 {
        0 => ord == Equal,   // EQ
        1 => ord == Less,    // LT
        2 => ord != Greater, // LE
        3 => false,          // FALSE
        4 => ord != Equal,   // NE
        5 => ord != Less,    // GE (NLT)
        6 => ord == Greater, // GT (NLE)
        _ => true,           // TRUE
    }
}

/// EVEX `vptestm`/`vptestnm` → opmask: per `elem`-byte lane, `(a & b) != 0` (or `== 0`
/// when `neg`), one bit per lane across the low `width` bytes.
fn vptest_mask(a: [u128; 4], b: [u128; 4], elem: u8, width: u16, neg: bool) -> u64 {
    let bits = elem as u32 * 8;
    let lane_mask = lane_mask(elem);
    let lanes_per_128 = 16 / elem as u32;
    let mut mask = 0u64;
    let mut idx = 0u32;
    for chunk in 0..(width as usize / 16) {
        for l in 0..lanes_per_128 {
            let sh = l * bits;
            let anded = (a[chunk] >> sh) & (b[chunk] >> sh) & lane_mask;
            let set = if neg { anded == 0 } else { anded != 0 };
            if set {
                mask |= 1u64 << idx;
            }
            idx += 1;
        }
    }
    mask
}

/// EVEX `vpcmp{,u}{b,w,d,q}` → opmask: one bit per `elem`-byte lane across the low
/// `width` bytes of the four 128-bit chunks, comparing signed or unsigned.
fn vpcmp_mask(a: [u128; 4], b: [u128; 4], elem: u8, width: u16, pred: u8, signed: bool) -> u64 {
    let bits = elem as u32 * 8;
    let lane_mask = lane_mask(elem);
    let lanes_per_128 = 16 / elem as u32;
    let mut mask = 0u64;
    let mut idx = 0u32;
    for chunk in 0..(width as usize / 16) {
        for l in 0..lanes_per_128 {
            let sh = l * bits;
            let la = (a[chunk] >> sh) & lane_mask;
            let lb = (b[chunk] >> sh) & lane_mask;
            let ord = if signed {
                sign_extend_128(la, bits as u8).cmp(&sign_extend_128(lb, bits as u8))
            } else {
                la.cmp(&lb)
            };
            if vpcmp_pred(pred, ord) {
                mask |= 1u64 << idx;
            }
            idx += 1;
        }
    }
    mask
}

fn packed_bin(a: u128, b: u128, lane: u8, op: PackedBinOp) -> u128 {
    let bits = lane as u32 * 8;
    let lane_mask = lane_mask(lane);
    let mut res = 0u128;
    let mut i = 0;
    while i < 16 / lane {
        let sh = i as u32 * bits;
        let (la, lb) = ((a >> sh) & lane_mask, (b >> sh) & lane_mask);
        // Signed lane values (sign-extended from `bits`) for the signed ops.
        let (sa, sb) = (
            sign_extend_128(la, bits as u8),
            sign_extend_128(lb, bits as u8),
        );
        let lr = match op {
            PackedBinOp::Add => la.wrapping_add(lb) & lane_mask,
            PackedBinOp::Sub => la.wrapping_sub(lb) & lane_mask,
            PackedBinOp::CmpEq => {
                if la == lb {
                    lane_mask
                } else {
                    0
                }
            }
            PackedBinOp::CmpGt => {
                if sa > sb {
                    lane_mask
                } else {
                    0
                }
            }
            PackedBinOp::MulLo16 | PackedBinOp::MulLo32 | PackedBinOp::MulLo64 => {
                la.wrapping_mul(lb) & lane_mask
            }
            // vpmulhuw/vpmulhw: high `bits` of the unsigned/signed lane×lane product.
            PackedBinOp::MulHiU16 => (la.wrapping_mul(lb) >> bits) & lane_mask,
            PackedBinOp::MulHiS16 => ((sa.wrapping_mul(sb) >> bits) as u128) & lane_mask,
            // vpmuludq: unsigned low-dword × low-dword → full 64-bit lane.
            PackedBinOp::MulU32 => (la & 0xffff_ffff).wrapping_mul(lb & 0xffff_ffff),
            // vpmuldq: signed low-dword × low-dword → full 64-bit lane (sign-extend first).
            PackedBinOp::MulS32 => {
                ((la as u32 as i32 as i64).wrapping_mul(lb as u32 as i32 as i64)) as u64 as u128
            }
            PackedBinOp::MinU => la.min(lb),
            PackedBinOp::MaxU => la.max(lb),
            PackedBinOp::MinS => {
                if sa < sb {
                    la
                } else {
                    lb
                }
            }
            PackedBinOp::MaxS => {
                if sa > sb {
                    la
                } else {
                    lb
                }
            }
            // paddsb/paddsw: signed saturating add, clamped to the signed range of `bits`.
            PackedBinOp::AddSatS => {
                let (lo, hi) = (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1);
                ((sa + sb).clamp(lo, hi) as u128) & lane_mask
            }
            // paddusb/paddusw: unsigned saturating add, clamped to [0, 2^bits - 1].
            PackedBinOp::AddSatU => {
                let hi = (1u128 << bits) - 1;
                (la + lb).min(hi)
            }
            // psubsb/psubsw: signed saturating subtract, clamped to the signed range.
            PackedBinOp::SubSatS => {
                let (lo, hi) = (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1);
                ((sa - sb).clamp(lo, hi) as u128) & lane_mask
            }
            // psubusb/psubusw: unsigned saturating subtract, clamped at 0.
            PackedBinOp::SubSatU => la.saturating_sub(lb),
            // pavgb/pavgw: unsigned rounding average (a + b + 1) >> 1.
            PackedBinOp::AvgU => (la + lb + 1) >> 1,
            // pmulhrsw: signed 16×16 product, take bits [16:1] rounded:
            // (((a*b) >> 14) + 1) >> 1. Lane width is always 2 (word).
            PackedBinOp::MulHiRoundedS16 => {
                let prod = (sa as i16 as i32).wrapping_mul(sb as i16 as i32);
                let r = (((prod >> 14) + 1) >> 1) as u32 as u128;
                r & lane_mask
            }
        };
        res |= lr << sh;
        i += 1;
    }
    res
}

/// Packed logical shift of each `lane`-byte element by `imm`.
fn packed_shift(a: u128, imm: u8, lane: u8, right: bool, arith: bool) -> u128 {
    let bits = lane as u32 * 8;
    let lane_mask = lane_mask(lane);
    let over = imm as u32 >= bits; // count >= element width
                                   // A logical/left over-shift yields 0; an arithmetic right over-shift yields
                                   // each lane's sign bit smeared across the whole element.
    if over && !(right && arith) {
        return 0;
    }
    let mut res = 0u128;
    let mut i = 0;
    while i < 16 / lane {
        let sh = i as u32 * bits;
        let lv = (a >> sh) & lane_mask;
        let lr = if !right {
            (lv << imm as u32) & lane_mask
        } else if !arith {
            lv >> imm as u32
        } else {
            // arithmetic right: sign-extend the lane, shift, re-mask.
            let sv = sign_extend_128(lv, bits as u8);
            let shifted = if over {
                sv >> (bits - 1)
            } else {
                sv >> imm as u32
            };
            (shifted as u128) & lane_mask
        };
        res |= lr << sh;
        i += 1;
    }
    res
}

/// punpckl*: interleave the low 8 bytes of `a` and `b` at `lane`-byte elements.
fn unpack_low(a: u128, b: u128, lane: u8, high: bool) -> u128 {
    let bits = lane as u32 * 8;
    let lane_mask = lane_mask(lane);
    let n = 8 / lane;
    let base = if high { n as u32 } else { 0 }; // start element: high half or low
    let mut res = 0u128;
    let mut i = 0u32;
    while i < n as u32 {
        let ea = (a >> ((base + i) * bits)) & lane_mask;
        let eb = (b >> ((base + i) * bits)) & lane_mask;
        res |= ea << (2 * i * bits);
        res |= eb << ((2 * i + 1) * bits);
        i += 1;
    }
    res
}

/// One 128-bit lane of `pack{ss,us}{wb,dw}`: saturate each `from`-byte source element
/// (read as signed) to `from/2` bytes; `a`'s elements fill the low half, `b`'s the high.
/// Saturate and pack one lane. SDM Vol 2B §4.3, PACKSSWB/PACKSSDW and PACKUSWB/PACKUSDW.
pub fn pack_lane(a: u128, b: u128, from: u8, signed: bool) -> u128 {
    let fb = from as u32 * 8;
    let tb = (from as u32 / 2) * 8; // result element bits
    let count = 16 / from as usize; // elements per source lane
    let (lo, hi): (i128, i128) = if signed {
        (-(1i128 << (tb - 1)), (1i128 << (tb - 1)) - 1)
    } else {
        (0, (1i128 << tb) - 1)
    };
    let tmask: u128 = (1u128 << tb) - 1;
    let elem = |v: u128, i: usize| -> i128 {
        let raw = (v >> (i as u32 * fb)) & ((1u128 << fb) - 1);
        // sign-extend the source element from `fb` bits (x86 packs read source as signed)
        let sign = 1u128 << (fb - 1);
        if raw & sign != 0 {
            (raw | (!0u128 << fb)) as i128
        } else {
            raw as i128
        }
    };
    let mut res = 0u128;
    for i in 0..count {
        let ca = (elem(a, i).clamp(lo, hi) as u128) & tmask;
        let cb = (elem(b, i).clamp(lo, hi) as u128) & tmask;
        res |= ca << (i as u32 * tb);
        res |= cb << ((count + i) as u32 * tb);
    }
    res
}

/// The QNaN floating-point indefinite: sign set, exponent all ones, significand MSB set
/// and the rest zero (SDM Vol 1 §4.8.3.7). What an FMA delivers for `inf * 0` and the
/// other invalid combinations Table 14-17 marks `QNaNIndefinite`.
const F32_INDEFINITE: u32 = 0xFFC0_0000;
const F64_INDEFINITE: u64 = 0xFFF8_0000_0000_0000;

/// One FMA element: `±(x*y) ± z` with a single rounding (`f64`/`f32` `mul_add`), returned
/// as the raw bit pattern. `neg_prod` negates the product, `neg_add` the addend.
///
/// The NaN cases come from SDM Vol 1 Table 14-17, "FMA Numeric Behavior": the result is
/// `Q(x)` when the multiplicand is a NaN, else `Q(y)` for the multiplier, else `Q(z)` for
/// the addend — the same for every sign variant. Two things follow:
///
///   * the answer is the **source** operand, so the `neg_prod`/`neg_add` sign flips must
///     not reach it. Negating first yields a NaN with the sign bit inverted
///     (`cargo xfuzz --seed 22131`);
///   * the payload must be chosen here rather than left to `mul_add`, whose NaN result
///     Rust and LLVM leave unspecified — see [`sse_binop_f32`] for the same defect on the
///     two-operand ops, where it also makes the answer depend on `opt-level`.
///
/// When no source is a NaN but the operation is still invalid (`inf * 0`, or an infinite
/// product added to an infinity of the opposite sign), Table 14-17 gives the QNaN
/// indefinite rather than an arbitrary NaN.
#[cfg(test)]
fn fma_elem(xb: u64, yb: u64, zb: u64, is_f64: bool, neg_prod: bool, neg_add: bool) -> u64 {
    fma_elem_mode::<true>(xb, yb, zb, is_f64, neg_prod, neg_add, 0)
}

fn fma_elem_mode<const DEFAULT: bool>(
    xb: u64,
    yb: u64,
    zb: u64,
    is_f64: bool,
    neg_prod: bool,
    neg_add: bool,
    mxcsr: u32,
) -> u64 {
    let mxcsr = if DEFAULT { 0 } else { mxcsr };
    if mxcsr & softfloat::MODE_MASK != 0 {
        return softfloat::fma(xb, yb, zb, is_f64, neg_prod, neg_add, mxcsr);
    }

    if is_f64 {
        let x0 = f64::from_bits(xb);
        let y = f64::from_bits(yb);
        let z0 = f64::from_bits(zb);
        for src in [x0, y, z0] {
            if src.is_nan() {
                return quiet_f64(src).to_bits();
            }
        }
        let x = if neg_prod { -x0 } else { x0 };
        let z = if neg_add { -z0 } else { z0 };
        let r = x.mul_add(y, z);
        if r.is_nan() {
            F64_INDEFINITE
        } else {
            r.to_bits()
        }
    } else {
        let x0 = f32::from_bits(xb as u32);
        let y = f32::from_bits(yb as u32);
        let z0 = f32::from_bits(zb as u32);
        for src in [x0, y, z0] {
            if src.is_nan() {
                return quiet_f32(src).to_bits() as u64;
            }
        }
        let x = if neg_prod { -x0 } else { x0 };
        let z = if neg_add { -z0 } else { z0 };
        let r = x.mul_add(y, z);
        if r.is_nan() {
            F32_INDEFINITE as u64
        } else {
            r.to_bits() as u64
        }
    }
}

/// FMA3 per-lane compute: `dst[i] = ±(x[i]*y[i]) ± z[i]`. Scalar keeps the low
/// element only (the rest of `old` dst is preserved); packed does `bytes/elem` lanes.
/// Shared by interp and the JIT helper → jit == interp.
#[allow(clippy::too_many_arguments)]
fn fma_lanes<const DEFAULT: bool>(
    xv: [u128; 4],
    yv: [u128; 4],
    zv: [u128; 4],
    old: [u128; 4],
    prec: FPrec,
    scalar: bool,
    neg_prod: bool,
    neg_add: bool,
    bytes: u16,
    alt_sign: u8,
    mxcsr: u32,
) -> [u128; 4] {
    let mxcsr = if DEFAULT { 0 } else { mxcsr };
    let elem = prec.bytes();
    let is_f64 = matches!(prec, FPrec::F64);
    let mut res = if scalar { old } else { [0u128; 4] };
    let n = if scalar {
        1
    } else {
        bytes as usize / elem as usize
    };
    for i in 0..n {
        // FMA alternating-sign family: `vfmaddsub` (alt_sign==1) subtracts z on
        // even lanes and adds on odd; `vfmsubadd` (==2) is the opposite. This overrides the
        // base `neg_add` per lane. `neg_add` here means "negate z" i.e. subtract.
        let na = match alt_sign {
            1 => i % 2 == 0, // fmaddsub: even → subtract, odd → add
            2 => i % 2 != 0, // fmsubadd: even → add, odd → subtract
            _ => neg_add,
        };
        let r = fma_elem_mode::<DEFAULT>(
            get_velem(&xv, i, elem),
            get_velem(&yv, i, elem),
            get_velem(&zv, i, elem),
            is_f64,
            neg_prod,
            na,
            mxcsr,
        );
        set_velem(&mut res, i, elem, r);
    }
    res
}

/// FMA3 entry for the JIT helper (register form): reads x/y/z from vector
/// registers, computes via [`fma_lanes`], writes dst. Guarantees jit == interp.
#[allow(clippy::too_many_arguments)]
pub fn exec_fma(
    cpu: &mut CpuState,
    dst: u8,
    x: u8,
    y: u8,
    z: u8,
    prec_f64: bool,
    scalar: bool,
    neg_prod: bool,
    neg_add: bool,
    bytes: u16,
    alt_sign: u8,
    k: u8,
    masked: bool,
    zeroing: bool,
) {
    let prec = if prec_f64 { FPrec::F64 } else { FPrec::F32 };
    let xv = cpu.vec_lanes(x as usize);
    let yv = cpu.vec_lanes(y as usize);
    let zv = cpu.vec_lanes(z as usize);
    let old = cpu.vec_lanes(dst as usize);
    let res = fma_lanes::<false>(
        xv, yv, zv, old, prec, scalar, neg_prod, neg_add, bytes, alt_sign, cpu.mxcsr,
    );
    // Masked EVEX packed FMA; scalar masked is rejected at lift.
    if masked {
        cpu.write_masked(dst as usize, res, k, prec.bytes(), zeroing, bytes);
    } else {
        let w = if scalar { 16 } else { bytes };
        cpu.set_vec(dst as usize, res, w);
    }
}

/// Replicate the low `chunk` bytes of `src_bytes` across `dst_width` bytes → four lanes.
fn broadcast_lane_lanes(src_bytes: &[u8; 64], chunk: usize, dst_width: usize) -> [u128; 4] {
    let mut out = [0u8; 64];
    let mut i = 0;
    while i + chunk <= dst_width && i + chunk <= 64 {
        out[i..i + chunk].copy_from_slice(&src_bytes[0..chunk]);
        i += chunk;
    }
    let mut r = [0u128; 4];
    for (j, slot) in r.iter_mut().enumerate() {
        *slot = u128::from_le_bytes(out[j * 16..j * 16 + 16].try_into().unwrap());
    }
    r
}

/// EVEX lane-broadcast register form: replicate the low `chunk` bytes of vector
/// `src` across the dest, masked/zeroing at `elem` granularity. Shared by interp + JIT.
#[allow(clippy::too_many_arguments)]
pub fn exec_broadcast_lane(
    cpu: &mut CpuState,
    dst: u8,
    src: u8,
    chunk: u8,
    elem: u8,
    dst_width: u16,
    k: u8,
    masked: bool,
    zeroing: bool,
) {
    let s = cpu.vec_lanes(src as usize);
    let mut sb = [0u8; 64];
    for (j, lane) in s.iter().enumerate() {
        sb[j * 16..j * 16 + 16].copy_from_slice(&lane.to_le_bytes());
    }
    let res = broadcast_lane_lanes(&sb, chunk as usize, dst_width as usize);
    if masked {
        cpu.write_masked(dst as usize, res, k, elem, zeroing, dst_width);
    } else {
        cpu.set_vec(dst as usize, res, dst_width);
    }
}

/// EVEX lane-broadcast memory form: the `chunk`-byte block is loaded from
/// `[base]` via `StrMem` (fault-capable — returns `Some(StrFault)` on unmapped).
#[allow(clippy::too_many_arguments)]
pub fn broadcast_lane_mem_run<M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    dst: u8,
    base: u64,
    chunk: u8,
    elem: u8,
    dst_width: u16,
    k: u8,
    masked: bool,
    zeroing: bool,
    cur_addr: u64,
) -> Option<StrFault> {
    let mut sb = [0u8; 64];
    // SDM Vol 1 §15.6.1; Vol 2A §2.8.6; Vol 2C, VPBROADCAST, Operation:
    // each active output selects its source element modulo the broadcast chunk.
    // Repeated outputs need only one read of their shared source element.
    let transfer = if masked { elem as usize } else { 8 };
    for off in (0..chunk as usize).step_by(transfer) {
        if masked {
            let needed = (off / elem as usize..dst_width as usize / elem as usize)
                .step_by(chunk as usize / elem as usize)
                .any(|i| cpu.kmask[k as usize] & (1 << i) != 0);
            if !needed {
                continue;
            }
        }
        match vpart(
            cpu,
            mem,
            cur_addr,
            base.wrapping_add(off as u64),
            transfer as u8,
        ) {
            Ok(v) => sb[off..off + transfer].copy_from_slice(&v.to_le_bytes()[..transfer]),
            Err(f) => return Some(f.into_str_fault(cpu, cur_addr, false)),
        }
    }
    let res = broadcast_lane_lanes(&sb, chunk as usize, dst_width as usize);
    if masked {
        cpu.write_masked(dst as usize, res, k, elem, zeroing, dst_width);
    } else {
        cpu.set_vec(dst as usize, res, dst_width);
    }
    None
}

/// AES-NI round entry for the JIT helper. Register form: read state `a`
/// and round key `b` from `cpu.xmm`, write `f(a, b)` to `dst`. Shared with interp via
/// [`AesOp::apply`] → jit == interp. `op` is the [`AesOp`] wire value.
pub fn exec_aes(cpu: &mut CpuState, dst: u8, a: u8, b: u8, op: u8) {
    let state = cpu.xmm[a as usize];
    let rk = cpu.xmm[b as usize];
    cpu.xmm[dst as usize] = crate::ir::AesOp::from_u8(op).apply(state, rk);
}

/// AES-NI round entry for the JIT memory-form helper: the round key is the
/// already-loaded 128-bit value `rk` (the load/fault is done natively before the call).
pub fn exec_aes_mem(cpu: &mut CpuState, dst: u8, a: u8, rk: u128, op: u8) {
    let state = cpu.xmm[a as usize];
    cpu.xmm[dst as usize] = crate::ir::AesOp::from_u8(op).apply(state, rk);
}

/// `aesimc`/`vaesimc` register-form JIT entry: `dst = InvMixColumns(src)`.
pub fn exec_aes_imc(cpu: &mut CpuState, dst: u8, src: u8) {
    cpu.xmm[dst as usize] = crate::aes::aes_imc(cpu.xmm[src as usize]);
}

/// `aesimc`/`vaesimc` memory-form JIT entry: source is the loaded value `v`.
pub fn exec_aes_imc_mem(cpu: &mut CpuState, dst: u8, v: u128) {
    cpu.xmm[dst as usize] = crate::aes::aes_imc(v);
}

/// `aeskeygenassist` register-form JIT entry.
pub fn exec_aes_keygen(cpu: &mut CpuState, dst: u8, src: u8, imm: u8) {
    cpu.xmm[dst as usize] = crate::aes::aes_keygen(cpu.xmm[src as usize], imm);
}

/// `aeskeygenassist` memory-form JIT entry: source is the loaded value `v`.
pub fn exec_aes_keygen_mem(cpu: &mut CpuState, dst: u8, v: u128, imm: u8) {
    cpu.xmm[dst as usize] = crate::aes::aes_keygen(v, imm);
}

/// SHA-NI register-form JIT entry: read op1 `a` and op2 `b` from `cpu.xmm`
/// (plus xmm0 for `sha256rnds2`), write `f(a, b, xmm0, imm)` to `dst`. Shared with interp
/// via [`ShaOp::apply`] → jit == interp. `op` is the [`ShaOp`] wire value.
pub fn exec_sha(cpu: &mut CpuState, dst: u8, a: u8, b: u8, imm: u8, op: u8) {
    let x = cpu.xmm[a as usize];
    let y = cpu.xmm[b as usize];
    let xmm0 = cpu.xmm[0];
    cpu.xmm[dst as usize] = crate::ir::ShaOp::from_u8(op).apply(x, y, xmm0, imm);
}

/// SHA-NI memory-form JIT entry: op2 is the already-loaded 128-bit value `v`
/// (the load/fault is done natively before the call).
pub fn exec_sha_mem(cpu: &mut CpuState, dst: u8, a: u8, v: u128, imm: u8, op: u8) {
    let x = cpu.xmm[a as usize];
    let xmm0 = cpu.xmm[0];
    cpu.xmm[dst as usize] = crate::ir::ShaOp::from_u8(op).apply(x, v, xmm0, imm);
}

/// GFNI register-form JIT entry: read op1 `a` and op2 `b` from `cpu.xmm`,
/// write `f(a, b, imm)` to `dst`. Shared with interp via [`GfniOp::apply`] → jit == interp.
/// `op` is the [`GfniOp`] wire value; `imm` is the affine constant (ignored by `mulb`).
pub fn exec_gfni(cpu: &mut CpuState, dst: u8, a: u8, b: u8, imm: u8, op: u8) {
    let x = cpu.xmm[a as usize];
    let y = cpu.xmm[b as usize];
    cpu.xmm[dst as usize] = crate::ir::GfniOp::from_u8(op).apply(x, y, imm);
}

/// GFNI memory-form JIT entry: op2 is the already-loaded 128-bit value `v`
/// (the load/fault is done natively before the call).
pub fn exec_gfni_mem(cpu: &mut CpuState, dst: u8, a: u8, v: u128, imm: u8, op: u8) {
    let x = cpu.xmm[a as usize];
    cpu.xmm[dst as usize] = crate::ir::GfniOp::from_u8(op).apply(x, v, imm);
}

/// PCLMULQDQ register-form JIT entry: `dst = clmul(a, b, imm)`.
pub fn exec_pclmul(cpu: &mut CpuState, dst: u8, a: u8, b: u8, imm: u8) {
    let x = cpu.xmm[a as usize];
    let y = cpu.xmm[b as usize];
    cpu.xmm[dst as usize] = crate::pclmul::pclmul(x, y, imm);
}

/// PCLMULQDQ memory-form JIT entry: op2 is the already-loaded 128-bit value `v`
/// (the load/fault is done natively before the call).
pub fn exec_pclmul_mem(cpu: &mut CpuState, dst: u8, a: u8, v: u128, imm: u8) {
    let x = cpu.xmm[a as usize];
    cpu.xmm[dst as usize] = crate::pclmul::pclmul(x, v, imm);
}

/// `movq2dq dst_xmm, src_mm`: MMX → XMM.
/// SDM Vol 2B §4.3 MOVQ2DQ; Vol 3A §15.2 Table 15-2.
pub fn exec_movq2dq(cpu: &mut CpuState, dst: u8, src_mm: u8) {
    let lo = u64::from_le_bytes(cpu.fpr[src_mm as usize][0..8].try_into().unwrap());
    cpu.xmm[dst as usize] = lo as u128;
}

/// `movdq2q dst_mm, src_xmm`: XMM → MMX.
/// SDM Vol 2B §4.3 MOVDQ2Q; Vol 3A §15.2 Table 15-2: exponent bits become all ones.
pub fn exec_movdq2q(cpu: &mut CpuState, dst_mm: u8, src_xmm: u8) {
    let lo = (cpu.xmm[src_xmm as usize] as u64).to_le_bytes();
    let slot = &mut cpu.fpr[dst_mm as usize];
    slot[0..8].copy_from_slice(&lo);
    slot[8] = 0xff;
    slot[9] = 0xff;
}

/// FMA3 memory-form entry for the JIT helper: one source (`mem_role`) comes
/// from `[base]`, loaded via `RawStrMem`. Fault-capable — writes the fault and returns
/// `Some(StrFault)` on an unmapped load. Shares [`fma_lanes`] with interp.
#[allow(clippy::too_many_arguments)]
pub fn fma_mem_run<M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    dst: u8,
    x: u8,
    y: u8,
    z: u8,
    base: u64,
    mem_role: u8,
    prec_f64: bool,
    scalar: bool,
    neg_prod: bool,
    neg_add: bool,
    bytes: u16,
    alt_sign: u8,
    cur_addr: u64,
    writemask: Option<u8>,
    zeroing: bool,
) -> Option<StrFault> {
    fma_mem_run_specialized::<false, M>(
        cpu, mem, dst, x, y, z, base, mem_role, prec_f64, scalar, neg_prod, neg_add, bytes,
        alt_sign, cur_addr, writemask, zeroing,
    )
}

#[allow(clippy::too_many_arguments)]
fn fma_mem_run_specialized<const DEFAULT: bool, M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    dst: u8,
    x: u8,
    y: u8,
    z: u8,
    base: u64,
    mem_role: u8,
    prec_f64: bool,
    scalar: bool,
    neg_prod: bool,
    neg_add: bool,
    bytes: u16,
    alt_sign: u8,
    cur_addr: u64,
    writemask: Option<u8>,
    zeroing: bool,
) -> Option<StrFault> {
    let prec = if prec_f64 { FPrec::F64 } else { FPrec::F32 };
    let elem = prec.bytes();
    // Load the memory operand: a scalar (low element) or a full `bytes`-wide vector.
    let mut memv = [0u128; 4];
    if let Some(k) = writemask {
        // SDM Vol 1 §15.6.1; Vol 2A §2.8.2, Table 2-48 (Type E2);
        // Vol 2C, VFMADD132PS/VFMADD213PS/VFMADD231PS, Operation:
        // masked-off elements suppress memory faults; active loads keep their width.
        let count = if scalar {
            1
        } else {
            bytes as usize / elem as usize
        };
        for i in 0..count {
            if cpu.kmask[k as usize] & (1 << i) != 0 {
                let ea = base.wrapping_add((i * elem as usize) as u64);
                let v = match vpart(cpu, mem, cur_addr, ea, elem) {
                    Ok(v) => v,
                    Err(f) => return Some(f.into_str_fault(cpu, cur_addr, false)),
                };
                set_velem(&mut memv, i, elem, v);
            }
        }
    } else if scalar {
        memv[0] = match vpart(cpu, mem, cur_addr, base, elem) {
            Ok(v) => v as u128,
            Err(f) => return Some(f.into_str_fault(cpu, cur_addr, false)),
        };
    } else {
        for (i, slot) in memv.iter_mut().enumerate().take(bytes as usize / 16) {
            let ea = base.wrapping_add(i as u64 * 16);
            *slot = match load_lane16(cpu, mem, ea, cur_addr) {
                Ok(v) => v,
                Err(f) => return Some(f),
            };
        }
    }
    let xv = if mem_role == 0 {
        memv
    } else {
        cpu.vec_lanes(x as usize)
    };
    let yv = if mem_role == 1 {
        memv
    } else {
        cpu.vec_lanes(y as usize)
    };
    let zv = if mem_role == 2 {
        memv
    } else {
        cpu.vec_lanes(z as usize)
    };
    let old = cpu.vec_lanes(dst as usize);
    let res = fma_lanes::<DEFAULT>(
        xv,
        yv,
        zv,
        old,
        prec,
        scalar,
        neg_prod,
        neg_add,
        bytes,
        alt_sign,
        interp_mxcsr::<DEFAULT>(cpu),
    );
    // Masked EVEX packed FMA; scalar masked is rejected at lift.
    match writemask {
        Some(k) => cpu.write_masked(dst as usize, res, k, elem, zeroing, bytes),
        None => {
            let w = if scalar { 16 } else { bytes };
            cpu.set_vec(dst as usize, res, w);
        }
    }
    None
}

/// Pack `pack{ss,us}{wb,dw}` over `bytes` (per 128-bit lane), signed/unsigned saturation.
/// Shared by interp and the JIT helper → jit == interp.
pub fn exec_vpack(
    cpu: &mut CpuState,
    dst: u8,
    a: u8,
    b: u8,
    from_elem: u8,
    signed: bool,
    bytes: u16,
) {
    let (av, bv) = (cpu.vec_lanes(a as usize), cpu.vec_lanes(b as usize));
    let mut res = [0u128; 4];
    for l in 0..(bytes as usize / 16) {
        res[l] = pack_lane(av[l], bv[l], from_elem, signed);
    }
    // Preserve bits above `bytes`: VPackWide is shared by legacy SSE (must PRESERVE
    // 255:128) and VEX (must CLEAR). Legacy and VEX.128 both land on bytes=16 with opposite
    // upper rules, so the clear moves to the lift — `set_vec_low` preserves here and the VEX
    // lift appends a trailing `VZeroUpper` (the same pattern as blend/byteshift).
    cpu.set_vec_low(dst as usize, res, bytes);
}

/// 128-bit memory-source pack: `xmm[dst] = pack(a, b)` where `b` is the
/// already-loaded 128-bit memory operand and `a` is op1's VALUE. Used by both the
/// interpreter and the JIT helper so the two paths share the saturation logic.
///
/// `a` is passed in rather than read back from `dst` (§16): op1 used to reach
/// this through a lift-emitted `VMov dst, a`, which wrote the destination before the load
/// that can fault.
pub fn pack_wide_mem_from(
    cpu: &mut CpuState,
    dst: u8,
    a: u128,
    b: u128,
    from_elem: u8,
    signed: bool,
) {
    cpu.xmm[dst as usize] = pack_lane(a, b, from_elem, signed);
}

/// `pmaddwd`: pairwise-multiply the eight signed 16-bit lanes of `a` and `b`,
/// then add adjacent products into four signed 32-bit dwords (two's-complement wrap).
/// Shared by interp and the JIT helper → jit == interp. Legacy SSE: preserves bits 255:128.
pub fn exec_pmaddwd(cpu: &mut CpuState, dst: u8, a: u8, b: u8) {
    let (av, bv) = (cpu.xmm[a as usize], cpu.xmm[b as usize]);
    let mut res = 0u128;
    for i in 0..4u32 {
        let lo = 2 * i * 16;
        let hi = (2 * i + 1) * 16;
        let a0 = ((av >> lo) as u16 as i16) as i32;
        let a1 = ((av >> hi) as u16 as i16) as i32;
        let b0 = ((bv >> lo) as u16 as i16) as i32;
        let b1 = ((bv >> hi) as u16 as i16) as i32;
        // Adjacent products, summed with two's-complement wrap (matches hardware).
        let d = (a0.wrapping_mul(b0)).wrapping_add(a1.wrapping_mul(b1));
        res |= ((d as u32) as u128) << (i * 32);
    }
    cpu.xmm[dst as usize] = res;
}

/// One 128-bit lane of `vpmaddwd`/`vpmaddubsw`. Shared by interp and the JIT
/// helper so jit == interp. `ubsw == false`: eight signed 16-bit lanes → four signed 32-bit
/// dwords (`a.word[2i]*b.word[2i] + a.word[2i+1]*b.word[2i+1]`, two's-complement wrap).
/// `ubsw == true` (`pmaddubsw`): sixteen byte lanes → eight words; per adjacent byte pair
/// `saturate_i16(a.byte(2i)_unsigned * b.byte(2i)_signed + a.byte(2i+1)_unsigned *
/// b.byte(2i+1)_signed)`.
pub fn pmadd_lane(a: u128, b: u128, ubsw: bool) -> u128 {
    let mut res = 0u128;
    if ubsw {
        for i in 0..8u32 {
            let lo = 2 * i * 8;
            let hi = (2 * i + 1) * 8;
            let a0 = (a >> lo) as u8 as i32; // unsigned
            let a1 = (a >> hi) as u8 as i32; // unsigned
            let b0 = (b >> lo) as u8 as i8 as i32; // signed
            let b1 = (b >> hi) as u8 as i8 as i32; // signed
            let s = a0 * b0 + a1 * b1; // exact (fits i32), then saturate to i16
            let w = s.clamp(i16::MIN as i32, i16::MAX as i32) as i16 as u16;
            res |= (w as u128) << (i * 16);
        }
    } else {
        for i in 0..4u32 {
            let lo = 2 * i * 16;
            let hi = (2 * i + 1) * 16;
            let a0 = (a >> lo) as u16 as i16 as i32;
            let a1 = (a >> hi) as u16 as i16 as i32;
            let b0 = (b >> lo) as u16 as i16 as i32;
            let b1 = (b >> hi) as u16 as i16 as i32;
            let d = (a0.wrapping_mul(b0)).wrapping_add(a1.wrapping_mul(b1));
            res |= ((d as u32) as u128) << (i * 32);
        }
    }
    res
}

/// VEX `vpmaddwd`/`vpmaddubsw` register form: width-generic multiply-add,
/// each 128-bit lane via [`pmadd_lane`]. Writes `bytes` (16/32); bits above `bytes` were
/// cleared by a preceding `VZeroUpper` (VEX.128) or are written here (VEX.256).
pub fn exec_v_pmadd(cpu: &mut CpuState, dst: u8, a: u8, b: u8, ubsw: bool, bytes: u16) {
    cpu.xmm[dst as usize] = pmadd_lane(cpu.xmm[a as usize], cpu.xmm[b as usize], ubsw);
    if bytes == 32 {
        cpu.ymm_hi[dst as usize] = pmadd_lane(cpu.ymm_hi[a as usize], cpu.ymm_hi[b as usize], ubsw);
    }
}

/// One 128-bit lane of the JIT `vpmaddubsw`/`vpmaddwd` memory helper:
/// `dst.lane[hi_half] = pmadd(a.lane[hi_half], memv, ubsw)`. `hi_half == 0` targets the
/// low (xmm) lane, `hi_half == 1` the high (ymm_hi) lane — pmadd is per-128-lane, so the
/// JIT loads each half of `[mem]` and calls this once per half.
pub fn exec_v_pmadd_mem_lane(
    cpu: &mut CpuState,
    dst: u8,
    a: u8,
    memv: u128,
    ubsw: bool,
    hi_half: bool,
) {
    if hi_half {
        cpu.ymm_hi[dst as usize] = pmadd_lane(cpu.ymm_hi[a as usize], memv, ubsw);
    } else {
        cpu.xmm[dst as usize] = pmadd_lane(cpu.xmm[a as usize], memv, ubsw);
    }
}

fn packuswb(a: u128, b: u128) -> u128 {
    let clamp = |w: u128| -> u128 {
        let s = w as u16 as i16;
        s.clamp(0, 255) as u128
    };
    let mut res = 0u128;
    for i in 0..8u32 {
        res |= clamp((a >> (i * 16)) & 0xffff) << (i * 8);
        res |= clamp((b >> (i * 16)) & 0xffff) << ((8 + i) * 8);
    }
    res
}

/// A failed vector sub-transfer: which access faulted, how wide it was, and how.
///
/// The width matters as much as the address. A 16-byte vector access is two
/// 8-byte transfers; reporting the operand's 16 bytes alongside the second transfer's
/// address hands the embedder a request it cannot answer — `complete_mmio_read` carries
/// one `u64`.
pub(crate) struct VecFault {
    pub at: u64,
    pub size: u8,
    pub trap: MemTrap,
    /// For a WRITE, the bytes THIS transfer carries — reporting the operand's low 8 bytes
    /// for every half of a 16-byte store would announce half a transaction as the whole of
    /// one. Zero for reads.
    pub value: u64,
}

impl VecFault {
    fn into_str_fault(self, cpu: &mut CpuState, cur_addr: u64, write: bool) -> StrFault {
        cpu.rip = cur_addr;
        StrFault {
            addr: self.at,
            write,
            trap: self.trap,
            value: self.value,
            elem: self.size,
        }
    }
}

/// Attach the memory operand's context after the shared transfer helper identifies
/// the faulting part. Slots include RAM and masked-off elements, so their indices
/// remain stable across retries and partially trapped operands.
fn vec_fault_exit(f: StrFault, base: u64, width: u16) -> Exit {
    let mut exit = str_fault_exit(f);
    match &mut exit {
        Exit::MmioRead {
            addr,
            size,
            total_size,
            part,
        }
        | Exit::MmioWrite {
            addr,
            size,
            total_size,
            part,
            ..
        } => {
            *total_size = width;
            *part = (addr.wrapping_sub(base) / u64::from(*size)) as u8;
        }
        _ => {}
    }
    exit
}

#[cold]
#[inline(never)]
fn vec_interp_fault(f: StrFault, base: u64, width: u16) -> Option<InterpResult> {
    Some(InterpResult::exit(vec_fault_exit(f, base, width)))
}

#[cold]
#[inline(never)]
fn vec_trap_out_optional(
    cpu: &mut CpuState,
    cur_addr: u64,
    f: VecFault,
    access: AccessKind,
    base: u64,
    width: u16,
) -> Option<InterpResult> {
    Some(vec_trap_out(cpu, cur_addr, f, access, base, width))
}

fn vec_trap_out(
    cpu: &mut CpuState,
    cur_addr: u64,
    f: VecFault,
    access: AccessKind,
    base: u64,
    width: u16,
) -> InterpResult {
    let f = f.into_str_fault(cpu, cur_addr, access == AccessKind::Write);
    InterpResult::exit(vec_fault_exit(f, base, width))
}

/// One 8-byte-or-narrower transfer of a vector access, with MMIO resumption.
///
/// `cur_addr` is the faulting instruction's address; it keys the sub-transfers the
/// embedder has already answered (`CpuState::mmio_parts`). Without that key a
/// multi-transfer instruction could never finish an MMIO access: the retry re-executes
/// the whole instruction, so a single `pending_mmio` is consumed by the FIRST transfer
/// and the second traps with nothing left — then the retry starts over and the first
/// traps again. Measured before this existed: four `run` → `complete_mmio_read` rounds
/// produced four identical exits.
#[inline(always)]
fn vpart<M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    cur_addr: u64,
    at: u64,
    size: u8,
) -> Result<u64, VecFault> {
    if let Some(v) = cpu.mmio_part(cur_addr, at) {
        return Ok(v);
    }
    match mem.sload(at, size) {
        Ok(v) => Ok(v),
        Err(MemTrap::Mmio) => match cpu.pending_mmio.take() {
            // The embedder answered THIS transfer since the last attempt. Record it so
            // the transfers after it can trap in turn without losing this one.
            Some(v) => {
                let v = v & mask(size);
                cpu.record_mmio_part(cur_addr, at, v);
                Ok(v)
            }
            None => Err(VecFault {
                at,
                size,
                trap: MemTrap::Mmio,
                value: 0,
            }),
        },
        Err(trap) => {
            let at = trap.fault_addr(at);
            // §16: refine only an unavailable transfer, preserving successful and
            // MMIO access widths. Unlike vpart, sload consumes no MMIO answer and
            // returns a Trap verdict without producing an MMIO exit.
            let at = if matches!(trap, MemTrap::Unmapped) {
                (0..size)
                    .map(|i| at.wrapping_add(u64::from(i)))
                    .find(|&ea| matches!(mem.sload(ea, 1), Err(MemTrap::Unmapped)))
                    .unwrap_or(at)
            } else {
                at
            };
            Err(VecFault {
                at,
                size,
                trap,
                value: 0,
            })
        }
    }
}

struct Preflighted<'a, M>(&'a M);
impl<M: StrMem> StrMem for Preflighted<'_, M> {
    fn sload(&self, addr: u64, elem: u8) -> Result<u64, MemTrap> {
        self.0.sload(addr, elem)
    }
    #[inline(always)]
    fn sstore(&self, addr: u64, val: u64, elem: u8) -> Result<(), MemTrap> {
        self.0.sstore_preflighted(addr, val, elem)
    }
}

/// The store counterpart of [`vpart`]. An acknowledgement is recorded the same way, with
/// a zero value — only its presence matters, and sharing one table means a 16-byte store
/// converges for the same reason a load does.
#[inline(always)]
fn vpart_store<M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    cur_addr: u64,
    at: u64,
    v: u64,
    size: u8,
) -> Result<(), VecFault> {
    if cpu.mmio_part(cur_addr, at).is_some() {
        return Ok(());
    }
    match mem.sstore(at, v, size) {
        Ok(()) => Ok(()),
        Err(MemTrap::Mmio) => {
            if cpu.pending_mmio_write {
                cpu.pending_mmio_write = false;
                cpu.record_mmio_part(cur_addr, at, 0);
                Ok(())
            } else {
                Err(VecFault {
                    at,
                    size,
                    trap: MemTrap::Mmio,
                    value: v,
                })
            }
        }
        Err(trap) => Err(VecFault {
            at: trap.fault_addr(at),
            size,
            trap,
            value: v,
        }),
    }
}

/// Load a 128-bit vector value (16/8/4 bytes; upper bytes zeroed for <16).
///
/// The error carries the sub-transfer that actually faulted, not the operand base: a
/// 16-byte access is two 8-byte operations, and naming the base tells
/// the embedder to map a page it already has — it maps it, retries, faults identically,
/// and loops, with no way out because the information was gone before the `Exit` was
/// built.
fn vload<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER, impl InterpreterMemory>,
    cur_addr: u64,
    addr: u64,
    size: u8,
) -> Result<u128, VecFault> {
    let _ordering = ordering::Barrier::<ORDER>::new(Some(ordering::Direction::Read), false);
    let mem = mem.unordered();
    match size {
        16 => {
            let lo = vpart(cpu, mem, cur_addr, addr, 8)? as u128;
            let hi = vpart(cpu, mem, cur_addr, addr.wrapping_add(8), 8)? as u128;
            Ok(lo | (hi << 64))
        }
        8 => Ok(vpart(cpu, mem, cur_addr, addr, 8)? as u128),
        // SDM Vol 2B §4.3, PMOVSX/PMOVZX: XMM byte-to-qword sources are m16.
        2 => Ok(vpart(cpu, mem, cur_addr, addr, 2)? as u128),
        _ => Ok(vpart(cpu, mem, cur_addr, addr, 4)? as u128),
    }
}

#[cfg(test)]
#[test]
fn vector_fault_probe_preserves_mmio_answers() {
    use crate::memory::{MemoryModel, Prot, RegionKind};

    let mut mem = Memory::new(MemoryModel::Flat { size: 0x10000 });
    mem.map(0x4000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    for answer in [None, Some(0x1122334455667788)] {
        let mut cpu = CpuState::new();
        cpu.pending_mmio = answer;
        cpu.pending_mmio_write = true;
        let Err(fault) = vload(&mut cpu, Access::<false>::new(&mem), 0x1000, 0x4ffc, 8) else {
            panic!("a transfer extending beyond Trap memory must fault");
        };
        assert!(matches!(fault.trap, MemTrap::Unmapped));
        assert_eq!(fault.at, 0x5000);
        assert_eq!(fault.size, 8);
        assert_eq!(cpu.pending_mmio, answer);
        assert!(cpu.pending_mmio_write);
        assert_eq!(cpu.mmio_part(0x1000, 0x4ffc), None);
    }
}

/// Load a `width`-byte vector (`width/16` 128-bit lanes) from `base` into `[u128; 4]`,
/// zero-filling lanes above `width`. On a fault, returns the faulting sub-transfer.
fn vload_lanes<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER, impl InterpreterMemory>,
    cur_addr: u64,
    base: u64,
    width: u16,
) -> Result<[u128; 4], VecFault> {
    let mut lanes = [0u128; 4];
    for (i, slot) in lanes.iter_mut().enumerate().take(width as usize / 16) {
        let ea = base.wrapping_add(i as u64 * 16);
        *slot = vload(cpu, mem, cur_addr, ea, 16)?;
    }
    Ok(lanes)
}

/// Per-lane population count over a 512-bit vector: each `lane`-byte element (`lane` ∈
/// {4,8} for `vpopcnt{d,q}`) is replaced by the count of its set bits.
fn vpopcnt_lanes(a: [u128; 4], lane: u8) -> [u128; 4] {
    let bits = lane as u32 * 8;
    let per_128 = 16 / lane as usize;
    let mut r = [0u128; 4];
    for (chunk, out) in a.iter().zip(r.iter_mut()) {
        for l in 0..per_128 {
            let sh = l as u32 * bits;
            let elem = if lane == 8 {
                (*chunk >> sh) as u64
            } else {
                ((*chunk >> sh) as u64) & 0xffff_ffff
            };
            *out |= (elem.count_ones() as u128) << sh;
        }
    }
    r
}

/// Store a 128-bit vector value. The error carries the faulting sub-transfer — see
/// [`vload`].
fn vstore<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER, impl InterpreterMemory>,
    cur_addr: u64,
    addr: u64,
    v: u128,
    size: u8,
) -> Result<(), VecFault> {
    let _ordering = ordering::Barrier::<ORDER>::new(Some(ordering::Direction::Write), false);
    let mem = mem.unordered();
    if ROUTED && size == 16 {
        if mem.region_kind(addr, 16) == Ok(crate::RegionKind::Ram) {
            if mem.store_crosses_host_page(addr, 16) {
                // Keep the whole value available so the copy can commit an unproven
                // page first, before either of the ordinary eight-byte transfers.
                return vstore_crossing_ram(mem, addr, &[v]);
            }
        } else if let Err((at, trap)) = mem.preflight_store(addr, u64::MAX, 8, u16::from(size)) {
            return Err(VecFault {
                at,
                size: 8,
                trap,
                value: 0,
            });
        }
    }
    vstore_parts(cpu, mem, cur_addr, addr, v, size)
}

#[cold]
fn vstore_crossing_ram<const ROUTED: bool>(
    mem: &Access<ROUTED, 0, impl InterpreterMemory>,
    addr: u64,
    lanes: &[u128],
) -> Result<(), VecFault> {
    let mut bytes = [0; 64];
    for (lane, out) in lanes.iter().zip(bytes.as_chunks_mut::<16>().0) {
        *out = lane.to_le_bytes();
    }
    // The caller validated one contiguous RAM region. Keep the entire payload
    // together so copy_out can order transfers after its actual-access probes.
    mem.write_ram(addr, &bytes[..lanes.len() * 16])
        .map_err(|at| {
            let offset = (at - addr) as usize;
            VecFault {
                at,
                size: 8,
                trap: MemTrap::HostFault,
                value: (lanes[offset / 16] >> (offset % 16 / 8 * 64)) as u64,
            }
        })
}

#[inline(always)]
fn vstore_parts<M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    cur_addr: u64,
    addr: u64,
    v: u128,
    size: u8,
) -> Result<(), VecFault> {
    match size {
        16 => {
            let mem = &Preflighted(mem);
            vpart_store(cpu, mem, cur_addr, addr, v as u64, 8)?;
            vpart_store(cpu, mem, cur_addr, addr + 8, (v >> 64) as u64, 8)
        }
        8 => vpart_store(cpu, mem, cur_addr, addr, v as u64, 8),
        _ => vpart_store(cpu, mem, cur_addr, addr, v as u64 & 0xffff_ffff, 4),
    }
}

#[cold]
#[inline(never)]
fn stack_fault_return(rip: u64, base: u64, fault: crate::ir::MemoryAccess) -> WalkResult {
    let exit = result::consume_current().expect("stack failure must initialize its cold record");
    classify_access(InterpResult::exit(exit), rip, base, fault).into()
}

// Only scalar fault keys cross the cold boundary. Passing a whole carrier to
// an out-of-line classifier gives each dispatch arm an escaping stack slot.
#[inline(always)]
fn walk_classify_access(
    result: InterpResult,
    rip: u64,
    base: u64,
    fault: crate::ir::MemoryAccess,
) -> WalkResult {
    match result {
        InterpResult::Exit(InterpExit::DetailedUnmapped { addr, access }) => {
            record_walk_access(addr, access, rip, base, fault);
            WalkResult::Fault
        }
        other => other.into(),
    }
}
#[cold]
#[inline(never)]
fn record_walk_access(
    addr: u64,
    access: AccessKind,
    rip: u64,
    base: u64,
    fault: crate::ir::MemoryAccess,
) {
    let _ = classify_access(
        InterpResult::Exit(InterpExit::DetailedUnmapped { addr, access }),
        rip,
        base,
        fault,
    );
}
#[inline(always)]
fn walk_classify_masked_access(
    result: InterpResult,
    rip: u64,
    base: u64,
    fault: crate::ir::MemoryAccess,
    elem: u8,
    mask: u64,
) -> WalkResult {
    let key = match result {
        InterpResult::Exit(InterpExit::DetailedUnmapped { addr, access }) => {
            result::memory_key(addr, access)
        }
        InterpResult::Exit(InterpExit::DetailedException { addr, vector }) => {
            (addr, u16::from(vector))
        }
        other => return other.into(),
    };
    record_walk_masked_access(key, rip, base, fault, elem, mask);
    WalkResult::Fault
}
#[cold]
#[inline(never)]
fn record_walk_masked_access(
    key: (u64, u16),
    rip: u64,
    base: u64,
    fault: crate::ir::MemoryAccess,
    elem: u8,
    mask: u64,
) {
    let exit = result::consume(key).expect("fault return must initialize its cold record");
    let _ = classify_masked_access(InterpResult::exit(exit), rip, base, fault, elem, mask);
}
#[inline(always)]
fn walk_classify_string_access(
    result: InterpResult,
    rip: u64,
    fault: crate::ir::MemoryAccess,
) -> WalkResult {
    let base = match &result {
        InterpResult::Exit(InterpExit::DetailedUnmapped { addr, .. }) => *addr,
        _ => 0,
    };
    walk_classify_access(result, rip, base, fault)
}

#[cold]
#[inline(never)]
fn classify_access(
    result: InterpResult,
    rip: u64,
    base: u64,
    fault: crate::ir::MemoryAccess,
) -> InterpResult {
    match result {
        InterpResult::Exit(exit) if matches!(exit, InterpExit::DetailedUnmapped { .. }) => {
            let mut exit = exit.into_exit();
            let (addr, access, old) = match &mut exit {
                Exit::UnmappedMemory {
                    addr,
                    access,
                    metadata,
                } => (*addr, *access, metadata.take()),
                _ => unreachable!(),
            };
            let mut classified = fault.unmapped_exit(rip, base, addr, access);
            if let Some(mut details) = old {
                if matches!(classified, Exit::UnmappedMemory { .. }) {
                    details.operand_base = crate::Knowledge::Known(base);
                    details.operand_size = crate::Knowledge::Known(fault.bytes());
                    classified = classified.with_fault_metadata(*details);
                } else if let Exit::Exception {
                    metadata: Some(m), ..
                } = &mut classified
                {
                    m.operand_base = crate::Knowledge::Known(base);
                    m.operand_size = crate::Knowledge::Known(fault.bytes());
                    m.transfer_size = details.transfer_size;
                    m.transfer_access = details.transfer_access;
                    m.address = details.address;
                }
            }
            InterpResult::exit(classified)
        }
        other => other,
    }
}

// Read vector masks inside the cold function too: inlining vec_lanes here
// otherwise hoists state pointers into scalar-only block prologues.
#[cold]
#[inline(never)]
fn fault_vec_mask(cpu: &CpuState, reg: u8, elem: u8, bytes: u16) -> u64 {
    vec_msb_mask(&cpu.vec_lanes(reg as usize), elem, bytes)
}

#[cold]
#[inline(never)]
fn fault_k_mask(cpu: &CpuState, reg: u8) -> u64 {
    if reg == 0 {
        u64::MAX
    } else {
        cpu.kmask[reg as usize]
    }
}

#[cold]
#[inline(never)]
fn classify_masked_access(
    result: InterpResult,
    rip: u64,
    base: u64,
    fault: crate::ir::MemoryAccess,
    elem: u8,
    mask: u64,
) -> InterpResult {
    // A broadcast carrier is one element regardless of the selected output lane
    // (SDM Vol 2A §2.7.7).
    let (operand, width) = if fault.bytes() == u16::from(elem) {
        (base, u64::from(fault.bytes()))
    } else {
        crate::memory::active_operand_span(base, u64::from(fault.bytes()), elem, mask)
    };
    let result = classify_access(
        result,
        rip,
        operand,
        crate::ir::MemoryAccess::new(width as u16, fault.long64(), fault.stack()),
    );
    match result {
        InterpResult::Exit(exit) => {
            let mut exit = exit.into_exit();
            if let Exit::UnmappedMemory {
                metadata: Some(m), ..
            }
            | Exit::Exception {
                metadata: Some(m), ..
            } = &mut exit
            {
                m.operand_base = crate::Knowledge::Known(base);
                m.operand_size = crate::Knowledge::Known(fault.bytes());
            }
            InterpResult::exit(exit)
        }
        other => other,
    }
}

/// Set RIP to the faulting instruction and turn a `MemTrap` into the matching Exit.
#[cold]
#[inline(never)]
fn trap_out(
    cpu: &mut CpuState,
    cur_addr: u64,
    trap: MemTrap,
    addr: u64,
    size: u8,
    access: AccessKind,
    value: u64,
) -> InterpResult {
    cpu.rip = cur_addr;
    let exit = match (trap, access) {
        (MemTrap::NonCanonical | MemTrap::Unmapped, _) => {
            crate::fault::memory_exit(cur_addr, addr, u16::from(size), access, false, false)
        }
        (t @ MemTrap::HostFault, _) => crate::fault::memory_exit(
            cur_addr,
            t.fault_addr(addr),
            u16::from(size),
            access,
            true,
            true,
        ),
        (MemTrap::Mmio, AccessKind::Read) => Exit::mmio_read(addr, size),
        (MemTrap::Mmio, _) => Exit::mmio_write(addr, size, value),
    };
    InterpResult::exit(exit)
}

// --- register access ---

pub(crate) fn read_reg(cpu: &CpuState, reg: Reg) -> u64 {
    match reg.gpr_index() {
        Some(i) => cpu.gpr[i],
        None => match reg {
            Reg::Rip => cpu.rip,
            Reg::FsBase => cpu.fs_base,
            Reg::GsBase => cpu.gs_base,
            // Real-mode segment selectors (§17.6): raw 16-bit selector; the base is
            // `selector << 4`, computed by the caller (`with_segment`/stack lift).
            Reg::Cs => cpu.cs as u64,
            Reg::Ds => cpu.ds as u64,
            Reg::Es => cpu.es as u64,
            Reg::Ss => cpu.ss as u64,
            _ => unreachable!("gpr_index None only for rip/fs/gs/segments"),
        },
    }
}

fn write_reg(cpu: &mut CpuState, reg: Reg, val: u64, size: u8) {
    match reg.gpr_index() {
        Some(i) => cpu.write_gpr(i, val, size),
        None => match reg {
            Reg::Rip => cpu.rip = val,
            Reg::FsBase => cpu.fs_base = val,
            Reg::GsBase => cpu.gs_base = val,
            Reg::Cs => cpu.cs = val as u16,
            Reg::Ds => cpu.ds = val as u16,
            Reg::Es => cpu.es = val as u16,
            Reg::Ss => cpu.ss = val as u16,
            _ => unreachable!("gpr_index None only for rip/fs/gs/segments"),
        },
    }
}

fn read_val(v: Val, temps: &[u64]) -> u64 {
    match v {
        Val::Temp(t) => temps[t as usize],
        Val::Imm(i) => i,
    }
}

// --- ALU + flags (Variant A, materialized, §3.2) ---

/// Result of an ALU op: the (masked) value plus the six candidate flags.
struct AluResult {
    res: u64,
    cf: bool,
    pf: bool,
    af: bool,
    zf: bool,
    sf: bool,
    of: bool,
}

fn mask(size: u8) -> u64 {
    if size >= 8 {
        u64::MAX
    } else {
        (1u64 << (size * 8)) - 1
    }
}

fn shift_mask(size: u8) -> u64 {
    if size == 8 {
        63
    } else {
        31
    }
}

fn sign_bit(size: u8) -> u64 {
    1u64 << (size * 8 - 1)
}

const RAX: usize = 0;
const RCX: usize = 1;
const RDX: usize = 2;
const RBX: usize = 3;
const RBP: usize = 5;
const RSI: usize = 6;
const RDI: usize = 7;
const R11: usize = 11;

/// Guest-memory access for `string_run_interruptible` (§10). Two implementors give the two
/// backends the memory semantics each already uses for a *scalar* store, so a
/// string op is never quietly weaker than the `mov` next to it:
///
/// * The interpreter passes `&Memory` — every element goes through the same region
///   check + SMC `note_write` as `IrOp::Store`, so `rep stos` onto a code page is
///   caught (§10), a `Trap` region yields MMIO, and an unmapped-but-in-bounds
///   address traps instead of silently scribbling the backing buffer.
/// * The JIT passes [`RawStrMem`] — a bounds-only raw view matching its inlined
///   stores, whose SMC/region handling is the deliberately deferred JIT-side step
///   (§10, §9.1). This keeps the two callers behavior-compatible without pulling
///   the (unavailable) `Memory` into the compiled ABI.
pub trait StrMem {
    fn sload(&self, addr: u64, elem: u8) -> Result<u64, MemTrap>;
    fn sstore(&self, addr: u64, val: u64, elem: u8) -> Result<(), MemTrap>;
    /// Store after a successful whole-destination preflight, without repeating it.
    /// Custom providers retain their ordinary store behavior by default.
    fn sstore_preflighted(&self, addr: u64, val: u64, elem: u8) -> Result<(), MemTrap> {
        self.sstore(addr, val, elem)
    }
    /// In recovery mode, preflight selected store pages before any element commits.
    /// The default implementation adds no checks for custom memory providers.
    fn preflight_store(
        &self,
        _base: u64,
        _mask: u64,
        _elem: u8,
        _bytes: u16,
    ) -> Result<(), (u64, MemTrap)> {
        Ok(())
    }
}

pub(crate) fn preflight_store_pages(
    base: u64,
    mask: u64,
    elem: u8,
    bytes: u16,
    host_base: usize,
    guest_base: u64,
    checks: (
        impl FnMut(u64, u8) -> Result<(), MemTrap>,
        impl FnMut(u64, u8) -> Result<u64, MemTrap>,
    ),
) -> Result<(), (u64, MemTrap)> {
    let (mut region, mut probe) = checks;
    // Guest regions can end inside a host page. Check every selected element
    // before the host-page probes; neither check consumes an MMIO answer.
    for i in 0..bytes as usize / elem as usize {
        if mask & (1u64 << i) != 0 {
            let ea = base.wrapping_add((i * elem as usize) as u64);
            if let Err(trap @ (MemTrap::Unmapped | MemTrap::NonCanonical | MemTrap::HostFault)) =
                region(ea, elem)
            {
                let at = if trap == MemTrap::Unmapped {
                    (0..elem)
                        .map(|j| ea.wrapping_add(u64::from(j)))
                        .find(|&at| matches!(region(at, 1), Err(MemTrap::Unmapped)))
                        .unwrap_or(ea)
                } else {
                    trap.fault_addr(ea)
                };
                return Err((at, trap));
            }
        }
    }
    let page = crate::memory::host_page_size() as u64;
    let count = bytes as usize / elem as usize;
    if count == 0 {
        return Ok(());
    }
    let selected = mask & (u64::MAX >> (64 - count));
    if selected == 0 {
        return Ok(());
    }
    let first = selected.trailing_zeros() as usize * elem as usize;
    let last = (64 - selected.leading_zeros() as usize) * elem as usize - 1;
    let host = (host_base as u64).wrapping_add(base.wrapping_sub(guest_base));
    if (host.wrapping_add(first as u64) ^ host.wrapping_add(last as u64)) & !(page - 1) == 0 {
        return Ok(());
    }
    let mut checked_until = 0u64;
    for i in 0..count {
        if selected & (1u64 << i) == 0 {
            continue;
        }
        let ea = base.wrapping_add((i * elem as usize) as u64);
        let host = (host_base as u64).wrapping_add(ea.wrapping_sub(guest_base));
        let mut j = checked_until.saturating_sub(host);
        while j < u64::from(elem) {
            let at = host.wrapping_add(j);
            let guest = ea.wrapping_add(j);
            let mut next = at + (page - (at & (page - 1)));
            match probe(guest, 1) {
                // MMIO keeps its existing per-element answer/retry path.
                Err(MemTrap::Mmio) => {}
                Err(t) => return Err((t.fault_addr(guest), t)),
                Ok(region_end) => {
                    // A ROM probe cannot prove access to RAM sharing its host page.
                    next = at + (region_end - guest).min(next - at);
                    checked_until = next;
                }
            }
            j += next - at;
        }
    }
    Ok(())
}

impl StrMem for Memory {
    fn preflight_store(
        &self,
        base: u64,
        mask: u64,
        elem: u8,
        bytes: u16,
    ) -> Result<(), (u64, MemTrap)> {
        if self.host_fault_recovery() {
            preflight_store_pages(
                base,
                mask,
                elem,
                bytes,
                self.host_base() as usize,
                self.guest_base(),
                (
                    |a, n| self.probe_write_mode::<false>(a, n),
                    |a, n| {
                        self.probe_write_mode::<true>(a, n)?;
                        self.region_end(a)
                    },
                ),
            )?;
        }
        Ok(())
    }

    fn sstore_preflighted(&self, addr: u64, val: u64, elem: u8) -> Result<(), MemTrap> {
        if self.host_fault_recovery() {
            crate::memory::InterpreterMemory::write_preflighted::<true>(self, addr, val, elem)
        } else {
            self.write(addr, val, elem)
        }
    }
    fn sload(&self, addr: u64, elem: u8) -> Result<u64, MemTrap> {
        self.read(addr, elem)
    }
    fn sstore(&self, addr: u64, val: u64, elem: u8) -> Result<(), MemTrap> {
        self.write(addr, val, elem)
    }
}

impl<const ROUTED: bool, const ORDER: u8, M: InterpreterMemory> StrMem
    for Access<ROUTED, ORDER, M>
{
    #[inline(always)]
    fn preflight_store(
        &self,
        base: u64,
        mask: u64,
        elem: u8,
        bytes: u16,
    ) -> Result<(), (u64, MemTrap)> {
        self.0.preflight_store::<ROUTED>(base, mask, elem, bytes)
    }

    #[inline(always)]
    fn sstore_preflighted(&self, addr: u64, val: u64, elem: u8) -> Result<(), MemTrap> {
        self.write_preflighted(addr, val, elem)
    }
    #[inline(always)]
    fn sload(&self, addr: u64, elem: u8) -> Result<u64, MemTrap> {
        self.read(addr, elem)
    }
    #[inline(always)]
    fn sstore(&self, addr: u64, value: u64, elem: u8) -> Result<(), MemTrap> {
        self.write(addr, value, elem)
    }
}

/// Bounds-only raw guest view for the JIT string helper (deferred JIT-side SMC).
/// It reports bounds and opted-in host faults; without region info it never reports MMIO.
///
/// `base` is the host address of guest `guest_base`; `size` is the exclusive top guest
/// address (`guest_base + span`). A guest address `a` translates to `base + (a -
/// guest_base)` and is valid iff `guest_base <= a` and `a + elem <= size`. The
/// base-relative offset `a - guest_base` (as a wrapping `u64`) exceeds `size -
/// guest_base` when `a < guest_base`, so the single unsigned bound below rejects
/// below-base and above-top in one comparison (mirrors the JIT's `checked_addr`).
pub struct RawStrMem {
    pub base: *mut u8,
    pub size: u64,
    pub guest_base: u64,
    /// Enable fixups when host protections are independently owned by the embedder.
    pub recover_host_faults: bool,
}

impl RawStrMem {
    /// Validate a store through the recovery routines, without committing a value.
    /// The shared masked-store preflight calls this once per selected host page.
    pub fn probe_store(&self, addr: u64, elem: u8) -> Result<(), MemTrap> {
        let off = self.off(addr, elem).ok_or(MemTrap::Unmapped)?;
        // SAFETY: off validates the full span within the live backing mapping.
        unsafe { crate::usercopy::probe(self.base.add(off), elem as usize, true) }.map_err(|at| {
            crate::memory::record_host_fault(self.guest_base + (at - self.base as usize) as u64)
        })
    }

    /// Backing offset for `addr` if `[addr, addr+elem)` lies in `[guest_base, size)`.
    #[inline]
    fn off(&self, addr: u64, elem: u8) -> Option<usize> {
        let end = addr.checked_add(elem as u64)?;
        if addr < self.guest_base || end > self.size {
            return None;
        }
        Some((addr - self.guest_base) as usize)
    }
}

impl RawStrMem {
    #[inline(always)]
    fn load_mode<const ROUTED: bool>(&self, addr: u64, elem: u8) -> Result<u64, MemTrap> {
        let off = self.off(addr, elem).ok_or(MemTrap::Unmapped)?;
        // SAFETY: off validated the full guest range; default mode requires host accessibility.
        if !ROUTED {
            return Ok(unsafe { crate::memory::plain_read(self.base.add(off), elem) });
        }
        unsafe { crate::usercopy::load(self.base.add(off), elem) }.map_err(|at| {
            crate::memory::record_host_fault(self.guest_base + (at - self.base as usize) as u64)
        })
    }
    #[inline(always)]
    fn store_mode<const ROUTED: bool>(&self, addr: u64, val: u64, elem: u8) -> Result<(), MemTrap> {
        let off = self.off(addr, elem).ok_or(MemTrap::Unmapped)?;
        // SAFETY: off validated the full guest range; default mode requires host accessibility.
        if !ROUTED {
            unsafe { crate::memory::plain_write(self.base.add(off), val, elem) };
            return Ok(());
        }
        unsafe { crate::usercopy::store(self.base.add(off), val, elem) }.map_err(|at| {
            crate::memory::record_host_fault(self.guest_base + (at - self.base as usize) as u64)
        })
    }
}
impl StrMem for RawStrMem {
    fn preflight_store(
        &self,
        base: u64,
        mask: u64,
        elem: u8,
        bytes: u16,
    ) -> Result<(), (u64, MemTrap)> {
        if self.recover_host_faults {
            preflight_store_pages(
                base,
                mask,
                elem,
                bytes,
                self.base as usize,
                self.guest_base,
                (
                    |a, n| self.off(a, n).map(|_| ()).ok_or(MemTrap::Unmapped),
                    |a, n| self.probe_store(a, n).map(|()| self.size),
                ),
            )?;
        }
        Ok(())
    }

    fn sstore_preflighted(&self, addr: u64, val: u64, elem: u8) -> Result<(), MemTrap> {
        if !self.recover_host_faults {
            return self.sstore(addr, val, elem);
        }
        let off = self.off(addr, elem).ok_or(MemTrap::Unmapped)?;
        // SAFETY: off validates the range; the caller preflighted selected pages.
        unsafe { crate::usercopy::store_preflighted(self.base.add(off), val, elem) }.map_err(|at| {
            crate::memory::record_host_fault(self.guest_base + (at - self.base as usize) as u64)
        })
    }
    fn sload(&self, addr: u64, elem: u8) -> Result<u64, MemTrap> {
        if self.recover_host_faults {
            self.load_mode::<true>(addr, elem)
        } else {
            self.load_mode::<false>(addr, elem)
        }
    }
    fn sstore(&self, addr: u64, value: u64, elem: u8) -> Result<(), MemTrap> {
        if self.recover_host_faults {
            self.store_mode::<true>(addr, value, elem)
        } else {
            self.store_mode::<false>(addr, value, elem)
        }
    }
}
#[repr(transparent)]
struct RawStrAccess<const ROUTED: bool>(RawStrMem);
impl<const ROUTED: bool> RawStrAccess<ROUTED> {
    fn new(raw: &RawStrMem) -> &Self {
        // SAFETY: transparent shared view; the returned reference retains the source lifetime.
        unsafe { &*(raw as *const RawStrMem).cast::<Self>() }
    }
}
impl<const ROUTED: bool> StrMem for RawStrAccess<ROUTED> {
    #[inline(always)]
    fn sload(&self, addr: u64, elem: u8) -> Result<u64, MemTrap> {
        self.0.load_mode::<ROUTED>(addr, elem)
    }
    #[inline(always)]
    fn sstore(&self, addr: u64, value: u64, elem: u8) -> Result<(), MemTrap> {
        self.0.store_mode::<ROUTED>(addr, value, elem)
    }
}

/// Run a raw JIT string helper, selecting its protection mode once per instruction.
#[allow(clippy::too_many_arguments)]
pub fn raw_string_run_interruptible(
    cpu: &mut CpuState,
    mem: &RawStrMem,
    op: StrOp,
    elem: u8,
    rep: RepKind,
    cur_addr: u64,
    addr_bits: u8,
    src_base: u64,
    dst_base: u64,
    exit_request: Option<&std::sync::atomic::AtomicU64>,
) -> Option<StringStop> {
    if mem.recover_host_faults {
        raw_string_routed(
            cpu,
            mem,
            op,
            elem,
            rep,
            cur_addr,
            addr_bits,
            src_base,
            dst_base,
            exit_request,
        )
    } else {
        string_run_interruptible(
            cpu,
            RawStrAccess::<false>::new(mem),
            op,
            elem,
            rep,
            cur_addr,
            addr_bits,
            src_base,
            dst_base,
            exit_request,
        )
    }
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn raw_string_routed(
    cpu: &mut CpuState,
    mem: &RawStrMem,
    op: StrOp,
    elem: u8,
    rep: RepKind,
    cur_addr: u64,
    addr_bits: u8,
    src_base: u64,
    dst_base: u64,
    exit_request: Option<&std::sync::atomic::AtomicU64>,
) -> Option<StringStop> {
    string_run_interruptible(
        cpu,
        RawStrAccess::<true>::new(mem),
        op,
        elem,
        rep,
        cur_addr,
        addr_bits,
        src_base,
        dst_base,
        exit_request,
    )
}

/// A string op stopped on a memory trap. Carries what the caller needs to build the
/// matching `Exit` (`value`/`elem` matter only for an MMIO write).
pub struct StrFault {
    pub addr: u64,
    pub write: bool,
    pub trap: MemTrap,
    pub value: u64,
    pub elem: u8,
}

/// Why a repeated string operation returned before completion.
pub enum StringStop {
    Fault(StrFault),
    Interrupted,
}

/// Run a string operation without host interruption, using a flat destination
/// segment. See [`string_run_interruptible`] for the interruptible implementation.
#[allow(clippy::too_many_arguments)]
pub fn string_run<M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    op: StrOp,
    elem: u8,
    rep: RepKind,
    cur_addr: u64,
    addr_bits: u8,
    seg_base: u64,
) -> Option<StrFault> {
    string_run_impl(cpu, mem, op, elem, rep, cur_addr, addr_bits, seg_base, 0)
}

/// Run a string operation without host interruption, with separate source and
/// destination segment bases. See [`string_run_interruptible`].
#[allow(clippy::too_many_arguments)]
pub fn string_run_impl<M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    op: StrOp,
    elem: u8,
    rep: RepKind,
    cur_addr: u64,
    addr_bits: u8,
    src_base: u64,
    dst_base: u64,
) -> Option<StrFault> {
    match string_run_interruptible(
        cpu, mem, op, elem, rep, cur_addr, addr_bits, src_base, dst_base, None,
    ) {
        Some(StringStop::Fault(fault)) => Some(fault),
        None => None,
        Some(StringStop::Interrupted) => unreachable!("no exit request supplied"),
    }
}

/// Shared restartable string implementation for the interpreter and JIT (§10).
/// Updates registers/flags after each iteration, preserving progress on a fault
/// or host request. Source and destination segment bases are supplied separately;
/// long/compat mode uses a flat ES destination, real mode passes `es<<4`.
/// Memory access goes through [`StrMem`].
#[allow(clippy::too_many_arguments)]
pub fn string_run_interruptible<M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    op: StrOp,
    elem: u8,
    rep: RepKind,
    cur_addr: u64,
    addr_bits: u8,
    src_base: u64,
    dst_base: u64,
    exit_request: Option<&std::sync::atomic::AtomicU64>,
) -> Option<StringStop> {
    string_run_mode::<false, M>(
        cpu,
        mem,
        op,
        elem,
        rep,
        cur_addr,
        addr_bits,
        src_base,
        dst_base,
        exit_request,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn string_run_mode<const CHILD: bool, M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    op: StrOp,
    elem: u8,
    rep: RepKind,
    cur_addr: u64,
    addr_bits: u8,
    src_base: u64,
    dst_base: u64,
    exit_request: Option<&std::sync::atomic::AtomicU64>,
) -> Option<StringStop> {
    string_run_specialized::<CHILD, false, false, M>(
        cpu,
        mem,
        op,
        elem,
        rep,
        cur_addr,
        addr_bits,
        src_base,
        dst_base,
        exit_request,
        (),
        (),
    )
}

// Bounded REP shares the operations and restart-register updates with the
// unbounded specialization. A returned boundary is checked before any next access.
#[allow(clippy::too_many_arguments)]
fn string_run_bounded<const ROUTED: bool, const STOP_WHEN_IRQ_READY: bool, M: InterpreterMemory>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, 0, M>,
    op: StrOp,
    elem: u8,
    rep: RepKind,
    cur_addr: u64,
    addr_bits: u8,
    src_base: u64,
    dst_base: u64,
    exit_request: Option<&std::sync::atomic::AtomicU64>,
    steps: &mut StepContext,
    shadows: &mut ShadowState,
    retired: u64,
) -> Option<WalkResult> {
    shadows.complete_before(retired);
    match string_run_specialized::<false, true, STOP_WHEN_IRQ_READY, _>(
        cpu,
        mem,
        op,
        elem,
        rep,
        cur_addr,
        addr_bits,
        src_base,
        dst_base,
        exit_request,
        &mut *steps,
        &mut *shadows,
    ) {
        None => None,
        Some(StringStop::Fault(f)) => Some(WalkResult::exit(str_fault_exit(f))),
        Some(StringStop::Interrupted) => {
            let stop = steps.boundary(cpu, shadows, exit_request);
            let stop = if STOP_WHEN_IRQ_READY && stop.is_none() {
                steps.cut = true;
                Some(WalkResult::Continue)
            } else {
                stop
            };
            steps.string_interrupted =
                matches!(stop, Some(WalkResult::Exit(InterpExit::Interrupted)));
            stop
        }
    }
}

// Transfer identities are iteration-local slots, not addresses: read and write
// at the same physical address are distinct device obligations. The ordinary
// vector ledger remains address-keyed; these paths cannot share an owner.
#[inline(always)]
fn string_load<const STEPPED: bool, M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    rip: u64,
    slot: u64,
    addr: u64,
    elem: u8,
) -> Result<u64, MemTrap> {
    if STEPPED {
        if let Some(v) = cpu.mmio_part(rip, slot) {
            return Ok(v);
        }
    }
    match mem.sload(addr, elem) {
        Err(MemTrap::Mmio) if STEPPED => {
            if let Some(v) = cpu.pending_mmio.take() {
                let v = v & mask(elem);
                cpu.record_mmio_part(rip, slot, v);
                Ok(v)
            } else {
                Err(MemTrap::Mmio)
            }
        }
        other => other,
    }
}

#[inline(always)]
fn string_store<const STEPPED: bool, M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    rip: u64,
    addr: u64,
    value: u64,
    elem: u8,
) -> Result<(), MemTrap> {
    if STEPPED && cpu.mmio_part(rip, 1).is_some() {
        return Ok(());
    }
    match mem.sstore(addr, value, elem) {
        Err(MemTrap::Mmio) if STEPPED && cpu.pending_mmio_write => {
            cpu.pending_mmio_write = false;
            cpu.record_mmio_part(rip, 1, 0);
            Ok(())
        }
        other => other,
    }
}

// The unbounded public helpers instantiate false: no per-iteration ledger,
// answer handling, countdown, or boundary polling is added to their loops.
#[allow(clippy::too_many_arguments)]
fn string_run_specialized<
    const CHILD: bool,
    const STEPPED: bool,
    const STOP_WHEN_IRQ_READY: bool,
    M: StrMem,
>(
    cpu: &mut CpuState,
    mem: &M,
    op: StrOp,
    elem: u8,
    rep: RepKind,
    cur_addr: u64,
    addr_bits: u8,
    src_base: u64,
    dst_base: u64,
    exit_request: Option<&std::sync::atomic::AtomicU64>,
    mut budget: impl BudgetState<StepContext>,
    mut inhibition: impl BudgetState<ShadowState>,
) -> Option<StringStop> {
    let step = if cpu.flags.df {
        (elem as i64).wrapping_neg() as u64
    } else {
        elem as u64
    };
    let m = mask(elem);
    // Address-size (§17.5): 64-bit (default) uses full RSI/RDI/RCX; a `67h` prefix
    // selects the 32-bit E-regs (mask 0xFFFF_FFFF, upper bits zeroed on write-back)
    // or 16-bit (mask 0xFFFF, upper bits preserved). `amask` bounds the pointer
    // arithmetic and the RCX counter.
    let amask: u64 = match addr_bits {
        16 => 0xFFFF,
        32 => 0xFFFF_FFFF,
        _ => u64::MAX,
    };
    // Effective linear address of a pointer register at its current (masked) offset:
    // DS-side reads add `src_base` (the FS/GS base under an override in long/compat, or
    // `ds<<4` / the override selector`<<4` in real mode); ES-side (destination) adds
    // `dst_base` (0 in long/compat, `es<<4` in real mode). Offset truncated to width.
    let src_lin = |reg: u64| src_base.wrapping_add(reg & amask);
    let dst_lin = |reg: u64| dst_base.wrapping_add(reg & amask);
    // Advance a pointer register by `step`, wrapping within the address width and
    // writing the result back with the right upper-bits policy: 64-bit → whole reg;
    // 32-bit → zero-extend (upper 32 cleared); 16-bit → merge (upper 48 preserved).
    let advance = |reg: u64| -> u64 {
        let lo = (reg & amask).wrapping_add(step) & amask;
        // 32-bit address size zero-extends the E-register write (upper 32 cleared),
        // matching write_gpr(.,4); 16-bit merges; 64-bit is the whole register.
        if addr_bits == 32 {
            lo
        } else {
            (reg & !amask) | lo
        }
    };
    if STEPPED && !matches!(rep, RepKind::None) && cpu.gpr[RCX] & amask != 0 {
        budget.get().rep_counted = true;
    }
    let mut completed_iteration = false;
    loop {
        if !matches!(rep, RepKind::None) && cpu.gpr[RCX] & amask == 0 {
            break;
        }
        if STEPPED
            && (budget
                .get()
                .boundary(cpu, inhibition.get(), exit_request)
                .is_some()
                || (STOP_WHEN_IRQ_READY && completed_iteration))
        {
            cpu.rip = cur_addr;
            return Some(StringStop::Interrupted);
        }
        // SDM Vol 2B, REP/REPE/REPZ/REPNE/REPNZ—Repeat String Operation
        // Prefix, Description and Operation (REP pp. 4-563–4-564;
        // REPE/REPZ pp. 4-565–4-566; REPNE/REPNZ pp. 4-567–4-568): an interrupted repeat keeps the next source/
        // destination pointers and the count after the last successful iteration;
        // the saved instruction pointer names the string instruction.
        // Relaxed polls only detect requests; the dispatcher consumes with Acquire.
        if !STEPPED
            && !matches!(rep, RepKind::None)
            && exit_request.is_some_and(|word| {
                word.load(std::sync::atomic::Ordering::Relaxed) & crate::vm::EXIT_REQUESTED != 0
            })
        {
            cpu.rip = cur_addr;
            return Some(StringStop::Interrupted);
        }
        match op {
            StrOp::Ins | StrOp::Outs => unreachable!("port strings use PortString"),
            StrOp::Movs => {
                let sa = src_lin(cpu.gpr[RSI]);
                let da = dst_lin(cpu.gpr[RDI]);
                let v = match string_load::<STEPPED, M>(cpu, mem, cur_addr, 0, sa, elem) {
                    Ok(v) => v,
                    Err(t) => {
                        if CHILD {
                            cpu.rip = cur_addr;
                            return Some(StringStop::Interrupted);
                        }
                        return trap(cpu, cur_addr, sa, false, t, 0, elem).map(StringStop::Fault);
                    }
                };
                if let Err(t) = string_store::<STEPPED, M>(cpu, mem, cur_addr, da, v, elem) {
                    if CHILD {
                        cpu.rip = cur_addr;
                        return Some(StringStop::Interrupted);
                    }
                    return trap(cpu, cur_addr, da, true, t, v, elem).map(StringStop::Fault);
                }
                cpu.gpr[RSI] = advance(cpu.gpr[RSI]);
                cpu.gpr[RDI] = advance(cpu.gpr[RDI]);
            }
            StrOp::Stos => {
                let da = dst_lin(cpu.gpr[RDI]);
                let v = cpu.gpr[RAX] & m;
                if let Err(t) = string_store::<STEPPED, M>(cpu, mem, cur_addr, da, v, elem) {
                    if CHILD {
                        cpu.rip = cur_addr;
                        return Some(StringStop::Interrupted);
                    }
                    return trap(cpu, cur_addr, da, true, t, v, elem).map(StringStop::Fault);
                }
                cpu.gpr[RDI] = advance(cpu.gpr[RDI]);
            }
            StrOp::Lods => {
                let sa = src_lin(cpu.gpr[RSI]);
                let v = match string_load::<STEPPED, M>(cpu, mem, cur_addr, 0, sa, elem) {
                    Ok(v) => v,
                    Err(t) => {
                        if CHILD {
                            cpu.rip = cur_addr;
                            return Some(StringStop::Interrupted);
                        }
                        return trap(cpu, cur_addr, sa, false, t, 0, elem).map(StringStop::Fault);
                    }
                };
                cpu.write_gpr(RAX, v, elem);
                cpu.gpr[RSI] = advance(cpu.gpr[RSI]);
            }
            StrOp::Scas => {
                let da = dst_lin(cpu.gpr[RDI]);
                let b = match string_load::<STEPPED, M>(cpu, mem, cur_addr, 1, da, elem) {
                    Ok(v) => v,
                    Err(t) => {
                        if CHILD {
                            cpu.rip = cur_addr;
                            return Some(StringStop::Interrupted);
                        }
                        return trap(cpu, cur_addr, da, false, t, 0, elem).map(StringStop::Fault);
                    }
                };
                let r = alu_sub(cpu.gpr[RAX] & m, b, 0, elem);
                apply(&mut cpu.flags, FlagMask::ALL, &r);
                cpu.gpr[RDI] = advance(cpu.gpr[RDI]);
            }
            StrOp::Cmps => {
                let sa = src_lin(cpu.gpr[RSI]);
                let da = dst_lin(cpu.gpr[RDI]);
                let a = match string_load::<STEPPED, M>(cpu, mem, cur_addr, 0, sa, elem) {
                    Ok(v) => v,
                    Err(t) => {
                        if CHILD {
                            cpu.rip = cur_addr;
                            return Some(StringStop::Interrupted);
                        }
                        return trap(cpu, cur_addr, sa, false, t, 0, elem).map(StringStop::Fault);
                    }
                };
                let b = match string_load::<STEPPED, M>(cpu, mem, cur_addr, 1, da, elem) {
                    Ok(v) => v,
                    Err(t) => {
                        if CHILD {
                            cpu.rip = cur_addr;
                            return Some(StringStop::Interrupted);
                        }
                        return trap(cpu, cur_addr, da, false, t, 0, elem).map(StringStop::Fault);
                    }
                };
                let r = alu_sub(a, b, 0, elem);
                apply(&mut cpu.flags, FlagMask::ALL, &r);
                cpu.gpr[RSI] = advance(cpu.gpr[RSI]);
                cpu.gpr[RDI] = advance(cpu.gpr[RDI]);
            }
        }
        if STEPPED {
            let steps = budget.get();
            let shadows = inhibition.get();
            // Decision-16 §3: iteration ownership ends even at unchanged REP IP.
            // The two transfer slots distinguish MOVS/CMPS operands at equal addresses.
            cpu.clear_mmio_parts();
            if !matches!(rep, RepKind::None) {
                steps.charge();
                if STOP_WHEN_IRQ_READY {
                    completed_iteration = true;
                }
                if shadows.bits != 0 {
                    shadows.complete();
                }
            }
        }
        // Decrement the (address-width) counter with the same upper-bits policy as the
        // pointer registers: 32-bit zero-extends (upper 32 cleared), 16-bit merges.
        let dec = |reg: u64| {
            let lo = (reg & amask).wrapping_sub(1) & amask;
            if addr_bits == 32 {
                lo
            } else {
                (reg & !amask) | lo
            }
        };
        match rep {
            RepKind::None => break,
            RepKind::Rep => cpu.gpr[RCX] = dec(cpu.gpr[RCX]),
            RepKind::Repe => {
                cpu.gpr[RCX] = dec(cpu.gpr[RCX]);
                if !cpu.flags.zf {
                    break;
                }
            }
            RepKind::Repne => {
                cpu.gpr[RCX] = dec(cpu.gpr[RCX]);
                if cpu.flags.zf {
                    break;
                }
            }
        }
    }
    None
}

/// Build the `Exit` for a memory fault reported by a shared mem helper (`string_run_interruptible`,
/// `masked_load_run`, `masked_store_run`). RIP is already set by the helper.
#[cold]
#[inline(never)]
fn str_fault_exit(f: StrFault) -> Exit {
    let access = if f.write {
        AccessKind::Write
    } else {
        AccessKind::Read
    };
    match (f.trap, access) {
        (MemTrap::HostFault, _) => {
            crate::fault::memory_exit(0, f.addr, u16::from(f.elem), access, true, true)
        }
        (MemTrap::Unmapped | MemTrap::NonCanonical, _) => {
            crate::fault::memory_exit(0, f.addr, u16::from(f.elem), access, false, false)
        }
        (MemTrap::Mmio, AccessKind::Read) => Exit::mmio_read(f.addr, f.elem),
        (MemTrap::Mmio, _) => Exit::mmio_write(f.addr, f.elem, f.value),
    }
}

/// Read element `i` (`elem` bytes, `elem` ∈ {1,2,4,8}) from a flattened 512-bit vector.
/// An element never straddles a 128-bit lane (`elem` divides 16), so a single lane
/// suffices.
#[inline]
fn get_velem(lanes: &[u128; 4], i: usize, elem: u8) -> u64 {
    let byte = i * elem as usize;
    let lane = byte / 16;
    let sh = (byte % 16) * 8;
    let mask = if elem == 8 {
        u64::MAX
    } else {
        (1u64 << (elem as u32 * 8)) - 1
    };
    ((lanes[lane] >> sh) as u64) & mask
}

/// Write element `i` (`elem` bytes) into a flattened 512-bit vector (see [`get_velem`]).
#[inline]
fn set_velem(lanes: &mut [u128; 4], i: usize, elem: u8, val: u64) {
    let byte = i * elem as usize;
    let lane = byte / 16;
    let sh = (byte % 16) * 8;
    let mask: u128 = if elem == 8 {
        u64::MAX as u128
    } else {
        (1u128 << (elem as u32 * 8)) - 1
    };
    lanes[lane] = (lanes[lane] & !(mask << sh)) | (((val as u128) & mask) << sh);
}

/// Convert an AVX1 vector mask register's per-element sign bits into an opmask-style
/// bitfield (`vmaskmovps/pd`). For each of the `bytes/elem` elements, bit `i`
/// is set iff element `i`'s most-significant bit is set. Lets the vector-mask conditional
/// load/store reuse [`masked_load_run`]/[`masked_store_run`] verbatim.
pub fn vec_msb_mask(lanes: &[u128; 4], elem: u8, bytes: u16) -> u64 {
    let n = bytes as usize / elem as usize;
    let top = elem as u32 * 8 - 1;
    let mut k = 0u64;
    for i in 0..n {
        if (get_velem(lanes, i, elem) >> top) & 1 != 0 {
            k |= 1 << i;
        }
    }
    k
}

/// SDM Vol 2C §5.2, VPGATHERDD/VPGATHERQD and VPGATHERDQ/VPGATHERQQ,
/// Operation: sign-extend VSIB indices, normalize mask elements, then commit each
/// loaded element and clear its mask. Ascending order delivers the lowest fault
/// with all lower selected elements complete. Floating gathers use the same bits.
#[allow(clippy::too_many_arguments)]
pub fn gather_run<M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    dst: u8,
    mask_reg: u8,
    index: u8,
    base: u64,
    segment: u64,
    elem: u8,
    index_elem: u8,
    bytes: u16,
    scale: u8,
    addr32: bool,
    cur_addr: u64,
) -> Option<(StrFault, u64)> {
    let count = bytes as usize / usize::from(elem.max(index_elem));
    let result_bytes = (count * usize::from(elem)) as u16;
    let indices = cpu.vec_lanes(index as usize);
    let mut values = cpu.vec_lanes(dst as usize);
    if result_bytes == 8 {
        values[0] &= u64::MAX as u128;
    }
    let original_mask = cpu.vec_lanes(mask_reg as usize);
    let mut masks = [0; 4];
    for i in 0..count {
        let active = get_velem(&original_mask, i, elem) >> (elem * 8 - 1) != 0;
        set_velem(&mut masks, i, elem, if active { u64::MAX } else { 0 });
    }
    // The unused portions may be zeroed even if the first selected load faults.
    cpu.set_vec(dst as usize, values, result_bytes.max(16));
    cpu.set_vec(mask_reg as usize, masks, result_bytes.max(16));
    for i in 0..count {
        if get_velem(&masks, i, elem) == 0 {
            continue;
        }
        let raw_index = get_velem(&indices, i, index_elem);
        let signed_index = if index_elem == 4 {
            raw_index as u32 as i32 as i64 as u64
        } else {
            raw_index
        };
        let offset = base.wrapping_add(signed_index.wrapping_mul(u64::from(scale)));
        let ea = segment.wrapping_add(if addr32 { offset as u32 as u64 } else { offset });
        match vpart(cpu, mem, cur_addr, ea, elem) {
            Ok(value) => {
                set_velem(&mut values, i, elem, value);
                set_velem(&mut masks, i, elem, 0);
                cpu.set_vec(dst as usize, values, result_bytes.max(16));
                cpu.set_vec(mask_reg as usize, masks, result_bytes.max(16));
            }
            Err(f) => return Some((f.into_str_fault(cpu, cur_addr, false), ea)),
        }
    }
    None
}

/// EVEX write-masked vector **load** from memory (`vmovdqu{8,16,32,64} v{k}{z}, [mem]`).
/// Element-wise so masked-off lanes never touch memory — hardware fault
/// suppression: a masked-off element straddling an unmapped page must NOT fault. Only
/// active `k` lanes are read; inactive lanes are zeroed (`zeroing`) or kept (merge). On an
/// active-lane fault nothing is committed (a masked load is architecturally atomic): RIP is
/// set to the faulting instruction and the fault returned. Shared by the interpreter and
/// the JIT helper via [`StrMem`], so JIT == interp.
#[allow(clippy::too_many_arguments)]
pub fn masked_load_run<M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    dst: u8,
    base: u64,
    k: u64,
    elem: u8,
    zeroing: bool,
    bytes: u16,
    cur_addr: u64,
) -> Option<StrFault> {
    let n = bytes as usize / elem as usize;
    let mut lanes = cpu.vec_lanes(dst as usize);
    for i in 0..n {
        if k & (1 << i) != 0 {
            let ea = base.wrapping_add((i * elem as usize) as u64);
            match vpart(cpu, mem, cur_addr, ea, elem) {
                Ok(v) => set_velem(&mut lanes, i, elem, v),
                Err(f) => return Some(f.into_str_fault(cpu, cur_addr, false)),
            }
        } else if zeroing {
            set_velem(&mut lanes, i, elem, 0);
        }
    }
    cpu.set_vec(dst as usize, lanes, bytes);
    None
}

/// SDM Vol 2 §2.7.7, Vol 1 §15.6.1: one broadcast element is read if
/// any participating mask bit is set; otherwise the memory access is suppressed.
#[allow(clippy::too_many_arguments)]
pub fn masked_broadcast_load_run<M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    dst: u8,
    base: u64,
    k: u64,
    elem: u8,
    bytes: u16,
    broadcast_elem: u8,
    cur_addr: u64,
) -> Option<StrFault> {
    let count = bytes / u16::from(elem);
    let active = k & (u64::MAX >> (64 - count));
    let value = if active == 0 {
        0
    } else {
        match vpart(cpu, mem, cur_addr, base, broadcast_elem) {
            Ok(v) => v,
            Err(f) => return Some(f.into_str_fault(cpu, cur_addr, false)),
        }
    };
    let mut lanes = [0; 4];
    for i in 0..usize::from(bytes) / usize::from(broadcast_elem) {
        set_velem(&mut lanes, i, broadcast_elem, value);
    }
    cpu.set_vec(dst as usize, lanes, bytes);
    None
}

/// EVEX write-masked vector **store** to memory (`vmovdqu{8,16,32,64} [mem]{k}, v`).
/// Element-wise, in order — active lanes are stored left to right, so a
/// fault mid-way can leave earlier active lanes committed in default mode (there is no
/// zeroing form for stores). Fault suppression: inactive lanes never touch memory. Shared
/// with the JIT helper via [`StrMem`]. Recovery mode preflights selected host pages.
#[allow(clippy::too_many_arguments)]
pub fn masked_store_run<M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    src: u8,
    base: u64,
    k: u64,
    elem: u8,
    bytes: u16,
    cur_addr: u64,
) -> Option<StrFault> {
    let n = bytes as usize / elem as usize;
    if let Err((at, t)) = mem.preflight_store(base, k, elem, bytes) {
        return trap(cpu, cur_addr, at, true, t, 0, elem);
    }
    let mem = &Preflighted(mem);
    let lanes = cpu.vec_lanes(src as usize);
    for i in 0..n {
        if k & (1 << i) != 0 {
            let ea = base.wrapping_add((i * elem as usize) as u64);
            let v = get_velem(&lanes, i, elem);
            if let Err(f) = vpart_store(cpu, mem, cur_addr, ea, v, elem) {
                return Some(f.into_str_fault(cpu, cur_addr, true));
            }
        }
    }
    None
}

/// Unmasked narrowing store `vpmov{q,d,w}{d,w,b} [mem], src`: truncate each of
/// the `src_width/from` source lanes to `to` bytes and store them contiguously at `base`.
/// Default mode stops at the first faulting element; recovery mode first preflights
/// selected host pages. Generic over [`StrMem`] so interp (`Memory`) and the JIT helper (`RawStrMem`)
/// share it → jit == interp.
#[allow(clippy::too_many_arguments)]
pub fn narrow_store_run<M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    src: u8,
    from: u8,
    to: u8,
    src_width: u16,
    base: u64,
    cur_addr: u64,
) -> Option<StrFault> {
    let n = src_width as usize / from as usize;
    if let Err((at, t)) = mem.preflight_store(base, u64::MAX, to, (n * to as usize) as u16) {
        return trap(cpu, cur_addr, at, true, t, 0, to);
    }
    let mem = &Preflighted(mem);
    let lanes = cpu.vec_lanes(src as usize);
    for i in 0..n {
        let v = get_velem(&lanes, i, from); // sstore truncates to `to` bytes on write
        let ea = base.wrapping_add((i * to as usize) as u64);
        if let Err(f) = vpart_store(cpu, mem, cur_addr, ea, v, to) {
            return Some(f.into_str_fault(cpu, cur_addr, true));
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn trap(
    cpu: &mut CpuState,
    cur_addr: u64,
    addr: u64,
    write: bool,
    t: MemTrap,
    value: u64,
    elem: u8,
) -> Option<StrFault> {
    cpu.rip = cur_addr;
    Some(StrFault {
        addr: t.fault_addr(addr),
        write,
        trap: t,
        value,
        elem,
    })
}

/// CRC-32C (Castagnoli, SSE4.2 `crc32`): fold the low `bytes` bytes of `src` into
/// the running CRC `crc` using the reflected polynomial 0x82F63B78. Shared by both
/// backends so the checksum matches bit-for-bit.
pub fn crc32c(mut crc: u32, src: u64, bytes: u8) -> u32 {
    for i in 0..bytes as u32 {
        crc ^= ((src >> (i * 8)) & 0xff) as u32;
        for _ in 0..8 {
            let m = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0x82F6_3B78 & m);
        }
    }
    crc
}

/// Paired compare-exchange shared by the interpreter and JIT helper.
/// SDM Vol 2A, CMPXCHG8B/CMPXCHG16B, Operation, Flags Affected,
/// and 64-Bit Mode Exceptions. All fault checks precede state commits.
pub fn cmpxchg_pair_run(
    cpu: &mut CpuState,
    mem: &Memory,
    cur_addr: u64,
    addr: u64,
    size: u8,
    len: u8,
) -> Option<Exit> {
    if mem.host_fault_recovery() {
        cmpxchg_pair_mode(cpu, Access::<true>::new(mem), cur_addr, addr, size, len)
    } else {
        cmpxchg_pair_mode(cpu, Access::<false>::new(mem), cur_addr, addr, size, len)
    }
}

fn cmpxchg_pair_mode<const ROUTED: bool>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, 0, impl InterpreterMemory>,
    cur_addr: u64,
    addr: u64,
    size: u8,
    len: u8,
) -> Option<Exit> {
    if size == 16 && addr & 15 != 0 {
        cpu.rip = cur_addr;
        return Some(crate::fault::exception_exit(
            cur_addr,
            13,
            crate::FaultCause::RequiredAlignment,
        ));
    }
    let bits = if size == 16 { 64 } else { 32 };
    let mask = if size == 16 {
        u64::MAX
    } else {
        u32::MAX as u64
    };
    let expected = ((cpu.gpr[RDX] & mask) as u128) << bits | (cpu.gpr[RAX] & mask) as u128;
    let src = ((cpu.gpr[RCX] & mask) as u128) << bits | (cpu.gpr[RBX] & mask) as u128;
    let result = if size == 16 {
        mem.atomic_cas128(addr, expected, src)
    } else {
        mem.atomic_cas(addr, expected as u64, src as u64, 8)
            .map(u128::from)
    };
    let old = match result {
        Ok(old) => old,
        Err(MemTrap::Mmio) => {
            // Engine policy: separate device reads/writes cannot implement a
            // paired CAS atomically. Reject it before any guest state commits.
            cpu.rip = cur_addr;
            let mut bytes = [0; 15];
            if let Ok(code) = mem.code_slice(cur_addr, len as usize) {
                bytes[..code.len()].copy_from_slice(&code);
            }
            return Some(Exit::unknown_instruction(cur_addr, bytes, len));
        }
        Err(t) => {
            let InterpResult::Exit(exit) =
                trap_out(cpu, cur_addr, t, addr, size, AccessKind::Write, src as u64)
            else {
                unreachable!()
            };
            return Some(exit.into_exit());
        }
    };
    cpu.flags.zf = old == expected;
    if old != expected {
        cpu.write_gpr(RAX, old as u64, size / 2);
        cpu.write_gpr(RDX, (old >> bits) as u64, size / 2);
    }
    None
}

/// `cpuid` (§14). Shared by both backends (the interpreter calls it directly; the JIT via
/// a helper) so `cpuid` answers identically everywhere. Reads leaf in EAX, subleaf in ECX;
/// writes EAX/EBX/ECX/EDX (32-bit, zero-extended).
pub fn cpuid_run(cpu: &mut CpuState) {
    let regs = cpu.cpuid(cpu.gpr[RAX] as u32, cpu.gpr[RCX] as u32);
    let crate::features::CpuidRegisters { eax, ebx, ecx, edx } = regs;
    cpu.write_gpr(RAX, eax as u64, 4);
    cpu.write_gpr(RBX, ebx as u64, 4);
    cpu.write_gpr(RCX, ecx as u64, 4);
    cpu.write_gpr(RDX, edx as u64, 4);
}

/// Shared RDTSC/RDTSCP execution (§14). SDM Vol 2B RDTSC/RDTSCP,
/// Description/Operation/Flags Affected: zero-extended outputs, flags preserved.
pub fn tsc_run(cpu: &mut CpuState, aux: bool) {
    if aux {
        // SDM Vol 2B RDTSCP, Description: earlier loads must be globally visible
        // before sampling. Order guest memory here; CpuSource must synchronise
        // hardware-counter execution itself (see its ordering contract).
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    }
    let sample = cpu.cpu_source.as_ref().map_or(
        crate::state::TscSample {
            counter: 0x1234_5678,
            aux: 0,
        },
        |source| source.read_tsc(cpu.source_id),
    );
    cpu.write_gpr(RAX, u64::from(sample.counter as u32), 4);
    cpu.write_gpr(RDX, sample.counter >> 32, 4);
    if aux {
        cpu.write_gpr(RCX, u64::from(sample.aux), 4);
    }
}

/// Shared RDRAND/RDSEED execution (§14). SDM Vol 2B RDRAND/RDSEED,
/// Operation/Flags Affected: failure returns zero; CF is success and the other
/// five status flags are clear. The fallback stream is engine policy, not entropy.
pub fn random_run(
    cpu: &mut CpuState,
    kind: crate::state::RandomKind,
    size: u8,
    set_flags: FlagMask,
) -> u64 {
    let value = if let Some(source) = &cpu.cpu_source {
        source.read_random(cpu.source_id, kind, size)
    } else {
        // A fixed wrapping LCG makes the per-vCPU fallback reproducible, including
        // CPU-state snapshots. It is deliberately not a cryptographic generator.
        cpu.random_state = cpu
            .random_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        Some(cpu.random_state)
    };
    let res = value.unwrap_or(0) & mask(size);
    apply(
        &mut cpu.flags,
        set_flags,
        &AluResult {
            res,
            cf: value.is_some(),
            pf: false,
            af: false,
            zf: false,
            sf: false,
            of: false,
        },
    );
    res
}

/// BMI1/BMI2 result + CF for one op. Shared by the interpreter and the
/// JIT's `bmi` helper so both agree exactly; ZF/SF are derived from the result at the
/// call site. `a`/`b` are the raw source values (masked here to `size`).
pub fn bmi_result(a: u64, b: u64, size: u8, op: crate::ir::BmiOp) -> (u64, bool) {
    use crate::ir::BmiOp::*;
    let m = mask(size);
    let bits = size as u32 * 8;
    let (av, bv) = (a & m, b & m);
    match op {
        Andn => (!av & bv & m, false),
        Blsi => (av & av.wrapping_neg() & m, av != 0),
        Blsr => (av & av.wrapping_sub(1) & m, av == 0),
        Blsmsk => ((av ^ av.wrapping_sub(1)) & m, av == 0),
        Bextr => {
            let start = (bv & 0xff) as u32;
            let len = ((bv >> 8) & 0xff) as u32;
            let shifted = if start >= 64 { 0 } else { av >> start };
            let r = if len >= 64 {
                shifted
            } else {
                shifted & ((1u64 << len) - 1)
            };
            (r & m, false)
        }
        Bzhi => {
            let idx = (bv & 0xff) as u32;
            let r = if idx >= bits {
                av
            } else {
                av & ((1u64 << idx) - 1)
            };
            (r & m, idx > bits - 1)
        }
        Pdep => {
            // Deposit a's low bits into the set positions of the mask (no flags).
            let mut r = 0u64;
            let mut k = 0u32;
            for i in 0..bits {
                if (bv >> i) & 1 != 0 {
                    r |= ((av >> k) & 1) << i;
                    k += 1;
                }
            }
            (r & m, false)
        }
        Pext => {
            // Extract a's bits at the set positions of the mask, packed low (no flags).
            let mut r = 0u64;
            let mut k = 0u32;
            for i in 0..bits {
                if (bv >> i) & 1 != 0 {
                    r |= ((av >> i) & 1) << k;
                    k += 1;
                }
            }
            (r & m, false)
        }
    }
}

pub(crate) fn xstate_exit(cur_addr: u64, error: crate::x87::xstate::XStateFault) -> Exit {
    match error {
        crate::x87::xstate::XStateFault::Exception { vector, cause } => {
            crate::fault::exception_exit(cur_addr, vector, cause)
        }
        crate::x87::xstate::XStateFault::Memory { addr, write } => crate::fault::fp_exit(
            cur_addr,
            addr,
            1,
            if write {
                AccessKind::Write
            } else {
                AccessKind::Read
            },
        ),
    }
}

/// Shared `xgetbv`: EDX:EAX = XCR0 for ECX=0, from the per-vCPU override or preset.
/// Called by both backends so they answer identically.
pub fn xgetbv_run(cpu: &mut CpuState) {
    let xcr0 = cpu
        .cpuid_overrides
        .xcr0
        .unwrap_or_else(|| cpu.features.xcr0());
    cpu.write_gpr(RAX, xcr0, 4);
    cpu.write_gpr(RDX, xcr0 >> 32, 4);
}

/// Divide the `size`-width `hi:lo` dividend by `divisor` (§16). Returns the
/// (quotient, remainder), or `None` for `#DE` — a zero divisor or a quotient that
/// overflows the destination width. Shared by the interpreter and the JIT's div
/// helper so both agree exactly.
pub fn divide(hi: u64, lo: u64, divisor: u64, size: u8, signed: bool) -> Option<(u64, u64)> {
    let m = mask(size);
    let n = size * 8;
    let d = divisor & m;
    if d == 0 {
        return None;
    }
    let dividend = ((hi & m) as u128) << n | (lo & m) as u128;
    if signed {
        let dv = sign_extend(d, size) as i64 as i128;
        let sd = sign_extend_128(dividend, 2 * n);
        // `i128::MIN / -1` (64-bit `idiv` of RDX:RAX = i128::MIN by -1) overflows and
        // would panic; the architecture raises #DE there, same as the quotient-range
        // check below — so fold it into the `None` (→ Exit::Exception vector 0) path.
        let (Some(q), Some(r)) = (sd.checked_div(dv), sd.checked_rem(dv)) else {
            return None;
        };
        let lim = 1i128 << (n - 1);
        if q < -lim || q >= lim {
            return None;
        }
        Some((q as u64 & m, r as u64 & m))
    } else {
        let (q, r) = (dividend / d as u128, dividend % d as u128);
        if q > m as u128 {
            return None;
        }
        Some((q as u64, r as u64))
    }
}

/// Sign-extend the low `bits` bits of a `u128` to a signed `i128`.
fn sign_extend_128(v: u128, bits: u8) -> i128 {
    if bits >= 128 {
        return v as i128;
    }
    let shift = 128 - bits;
    ((v << shift) as i128) >> shift
}

/// Sign-extend the low `from` bytes of `v` to a full 64-bit value.
fn sign_extend(v: u64, from: u8) -> u64 {
    if from >= 8 {
        return v;
    }
    let bits = from * 8;
    let shift = 64 - bits;
    (((v << shift) as i64) >> shift) as u64
}

fn parity(v: u64) -> bool {
    (v as u8).count_ones().is_multiple_of(2)
}

/// `pshufb` byte shuffle: each result byte is selected from `data` by the low
/// nibble of the index byte, or zero when the index's top bit is set.
fn pshufb(data: u128, idx: u128) -> u128 {
    let mut r = 0u128;
    for i in 0..16u32 {
        let sel = (idx >> (i * 8)) & 0xff;
        if sel & 0x80 == 0 {
            let byte = (data >> ((sel as u32 & 0xf) * 8)) & 0xff;
            r |= byte << (i * 8);
        }
    }
    r
}

/// `palignr` (SSSE3): concatenate `dst` (high 16 bytes) with `src` (low 16) into a
/// 32-byte value, shift it right by `imm` bytes, and return the low 16. `imm >= 32`
/// shifts everything out (zero). Branches avoid a shift-by-128 (UB on `u128`).
fn palignr(dst: u128, src: u128, imm: u8) -> u128 {
    let shift = imm as u32 * 8; // bit shift over the 256-bit concatenation
    if imm >= 32 {
        0
    } else if shift == 0 {
        src
    } else if shift < 128 {
        (src >> shift) | (dst << (128 - shift))
    } else if shift == 128 {
        dst
    } else {
        dst >> (shift - 128)
    }
}

/// Low-lane mask for a `bytes`-wide element within a 128-bit value.
fn lane_mask(bytes: u8) -> u128 {
    if bytes >= 16 {
        u128::MAX
    } else {
        (1u128 << (bytes as u32 * 8)) - 1
    }
}

/// `pmovmskb` over one 128-bit register: gather the top bit of each of the 16 bytes into
/// the low 16 bits of the result (bit `i` = byte `i`'s sign).
fn movemask_b(v: u128) -> u64 {
    let mut m = 0u64;
    for i in 0..16 {
        if (v >> (i * 8 + 7)) & 1 != 0 {
            m |= 1 << i;
        }
    }
    m
}

/// Scalar/packed float arithmetic. For `scalar`, only lane 0 is computed and the
/// upper bytes of `a` (= `dst`) are preserved; otherwise every `prec`-wide lane.
fn float_bin_mode<const DEFAULT: bool>(
    a: u128,
    b: u128,
    op: FloatBinOp,
    prec: FPrec,
    scalar: bool,
    mxcsr: u32,
) -> u128 {
    let mxcsr = if DEFAULT { 0 } else { mxcsr };
    // A separate scalar path avoids lane-loop bookkeeping around the cold call.
    if scalar {
        return match prec {
            FPrec::F32 => {
                let z = apply_f32_mode::<DEFAULT>(
                    f32::from_bits(a as u32),
                    f32::from_bits(b as u32),
                    op,
                    mxcsr,
                );
                (a & !u128::from(u32::MAX)) | u128::from(z.to_bits())
            }
            FPrec::F64 => {
                let z = apply_f64_mode::<DEFAULT>(
                    f64::from_bits(a as u64),
                    f64::from_bits(b as u64),
                    op,
                    mxcsr,
                );
                (a & !u128::from(u64::MAX)) | u128::from(z.to_bits())
            }
        };
    }
    let bytes = prec.bytes() as u32;
    let lanes = 16 / bytes as usize;
    let mut r = a;
    for i in 0..lanes {
        let sh = i as u32 * bytes * 8;
        match prec {
            FPrec::F32 => {
                let z = apply_f32_mode::<DEFAULT>(
                    f32::from_bits((a >> sh) as u32),
                    f32::from_bits((b >> sh) as u32),
                    op,
                    mxcsr,
                );
                r = (r & !(0xffff_ffffu128 << sh)) | ((z.to_bits() as u128) << sh);
            }
            FPrec::F64 => {
                let z = apply_f64_mode::<DEFAULT>(
                    f64::from_bits((a >> sh) as u64),
                    f64::from_bits((b >> sh) as u64),
                    op,
                    mxcsr,
                );
                r = (r & !(0xffff_ffff_ffff_ffffu128 << sh)) | ((z.to_bits() as u128) << sh);
            }
        }
    }
    r
}

/// SSE3 lane-combining packed float `h{add,sub}p`/`addsubp`. All are packed;
/// `hadd`/`hsub` combine adjacent lanes within each source, `addsub` alternates sub/add
/// between the two sources. Shared by the interpreter and the JIT helper (jit == interp).
pub fn hfloat(a: u128, b: u128, op: HFloatOp, prec: FPrec) -> u128 {
    hfloat_mode(a, b, op, prec, 0)
}

pub fn hfloat_mode(a: u128, b: u128, op: HFloatOp, prec: FPrec, mxcsr: u32) -> u128 {
    hfloat_mode_specialized::<false>(a, b, op, prec, mxcsr)
}

fn hfloat_mode_specialized<const DEFAULT: bool>(
    a: u128,
    b: u128,
    op: HFloatOp,
    prec: FPrec,
    mxcsr: u32,
) -> u128 {
    let mxcsr = if DEFAULT { 0 } else { mxcsr };
    macro_rules! pack {
        ($ty:ty, $bits:literal, $to:ident, $vals:expr) => {{
            let vals: &[$ty] = &$vals;
            let mut r: u128 = 0;
            for (i, v) in vals.iter().enumerate() {
                r |= (v.$to() as u128) << (i as u32 * $bits);
            }
            r
        }};
    }
    match prec {
        FPrec::F32 => {
            let la = |i: u32| f32::from_bits((a >> (i * 32)) as u32);
            let lb = |i: u32| f32::from_bits((b >> (i * 32)) as u32);
            // Every lane goes through the architectural NaN rule: these are ordinary
            // two-operand SSE adds and subtracts, so a NaN result is the first source
            // operand of that lane's pair, quieted (SDM Vol 1 §4.8.3.5, Table 4-8). For
            // the horizontal forms the pair is (lower, upper) within one register — the
            // order the SDM pseudo-code writes, `DEST[31:0] := SRC1[31:0] + SRC1[63:32]`.
            let add = |x, y| apply_f32_mode::<DEFAULT>(x, y, FloatBinOp::Add, mxcsr);
            let sub = |x, y| apply_f32_mode::<DEFAULT>(x, y, FloatBinOp::Sub, mxcsr);
            let out: [f32; 4] = match op {
                HFloatOp::HAdd => [
                    add(la(0), la(1)),
                    add(la(2), la(3)),
                    add(lb(0), lb(1)),
                    add(lb(2), lb(3)),
                ],
                HFloatOp::HSub => [
                    sub(la(0), la(1)),
                    sub(la(2), la(3)),
                    sub(lb(0), lb(1)),
                    sub(lb(2), lb(3)),
                ],
                HFloatOp::AddSub => [
                    sub(la(0), lb(0)),
                    add(la(1), lb(1)),
                    sub(la(2), lb(2)),
                    add(la(3), lb(3)),
                ],
            };
            pack!(f32, 32, to_bits, out)
        }
        FPrec::F64 => {
            let la = |i: u32| f64::from_bits((a >> (i * 64)) as u64);
            let lb = |i: u32| f64::from_bits((b >> (i * 64)) as u64);
            let add = |x, y| apply_f64_mode::<DEFAULT>(x, y, FloatBinOp::Add, mxcsr);
            let sub = |x, y| apply_f64_mode::<DEFAULT>(x, y, FloatBinOp::Sub, mxcsr);
            let out: [f64; 2] = match op {
                HFloatOp::HAdd => [add(la(0), la(1)), add(lb(0), lb(1))],
                HFloatOp::HSub => [sub(la(0), la(1)), sub(lb(0), lb(1))],
                HFloatOp::AddSub => [sub(la(0), lb(0)), add(la(1), lb(1))],
            };
            pack!(f64, 64, to_bits, out)
        }
    }
}

/// Decode the stable op-code used by the JIT `hfloat` helper back to
/// [`HFloatOp`]. Kept next to [`hfloat`] so the encoding lives in one place.
pub fn hfloat_op_from_code(code: u8) -> HFloatOp {
    match code {
        0 => HFloatOp::HAdd,
        1 => HFloatOp::HSub,
        _ => HFloatOp::AddSub,
    }
}

/// Register-form core for `h{add,sub}p`/`addsubp`: per 128-bit lane,
/// `lane[dst] = hfloat(lane[a], lane[b])`. AVX 256-bit horizontal ops run each 128-bit
/// half independently, so each half is a self-contained hadd of its own a/b.
/// Writes only the low `bytes` (16 or 32); the VEX.128 lift adds `VZeroUpper` for the
/// 255:128 clear, and legacy SSE preserves the upper YMM — so this never touches bits
/// above `bytes`. Shared by the interpreter dispatch and the JIT helper (jit == interp).
pub fn hfloat_reg(
    cpu: &mut CpuState,
    dst: u8,
    a: u8,
    b: u8,
    op: HFloatOp,
    prec: FPrec,
    bytes: u16,
) {
    hfloat_reg_specialized::<false>(cpu, dst, a, b, op, prec, bytes)
}

#[allow(clippy::too_many_arguments)]
fn hfloat_reg_specialized<const DEFAULT: bool>(
    cpu: &mut CpuState,
    dst: u8,
    a: u8,
    b: u8,
    op: HFloatOp,
    prec: FPrec,
    bytes: u16,
) {
    let (la, lb) = (cpu.vec_lanes(a as usize), cpu.vec_lanes(b as usize));
    let lanes = bytes as usize / 16;
    for i in 0..lanes {
        let r = hfloat_mode_specialized::<DEFAULT>(
            la[i],
            lb[i],
            op,
            prec,
            interp_mxcsr::<DEFAULT>(cpu),
        );
        match i {
            0 => cpu.xmm[dst as usize] = r,
            _ => cpu.ymm_hi[dst as usize] = r,
        }
    }
}

/// Memory-form core: per 128-bit lane, `lane[dst] = hfloat(lane[a], lane[b])` where `a` is
/// op1 (register source) and `b` is the already-loaded `bytes`-wide memory operand. Reading
/// `a` explicitly (rather than requiring a pre-copy into `dst`) keeps the ymm high lane
/// correct. Shared by the interpreter dispatch and the JIT helper.
pub fn hfloat_mem(
    cpu: &mut CpuState,
    dst: u8,
    a: u8,
    b: [u128; 4],
    op: HFloatOp,
    prec: FPrec,
    bytes: u16,
) {
    hfloat_mem_specialized::<false>(cpu, dst, a, b, op, prec, bytes)
}

#[allow(clippy::too_many_arguments)]
fn hfloat_mem_specialized<const DEFAULT: bool>(
    cpu: &mut CpuState,
    dst: u8,
    a: u8,
    b: [u128; 4],
    op: HFloatOp,
    prec: FPrec,
    bytes: u16,
) {
    let la = cpu.vec_lanes(a as usize);
    let lanes = bytes as usize / 16;
    for i in 0..lanes {
        let r =
            hfloat_mode_specialized::<DEFAULT>(la[i], b[i], op, prec, interp_mxcsr::<DEFAULT>(cpu));
        match i {
            0 => cpu.xmm[dst as usize] = r,
            _ => cpu.ymm_hi[dst as usize] = r,
        }
    }
}

/// SSSE3 packed-integer horizontal `ph{add,sub}{w,d,sw}`. Combines adjacent
/// lane pairs within each source; the low half of the result comes from `a`, the high half
/// from `b`. The `Sw` variants signed-saturate each 16-bit result. Shared by the
/// interpreter and the JIT helper (jit == interp).
pub fn hint(a: u128, b: u128, op: HIntOp) -> u128 {
    // 16-bit lane i of `v` as i32 (widened so add/sub can't overflow before saturating).
    let w = |v: u128, i: u32| ((v >> (i * 16)) as u16 as i16) as i32;
    // 32-bit lane i of `v` as i64.
    let d = |v: u128, i: u32| ((v >> (i * 32)) as u32 as i32) as i64;
    let sat16 = |x: i32| x.clamp(i16::MIN as i32, i16::MAX as i32) as u16 as u128;
    match op {
        HIntOp::AddW | HIntOp::SubW | HIntOp::AddSw | HIntOp::SubSw => {
            // Eight 16-bit results: four adjacent-pair combines from `a`, then from `b`.
            let combine = |x: i32, y: i32| match op {
                HIntOp::AddW => (x.wrapping_add(y)) as u16 as u128,
                HIntOp::SubW => (x.wrapping_sub(y)) as u16 as u128,
                HIntOp::AddSw => sat16(x + y),
                HIntOp::SubSw => sat16(x - y),
                _ => unreachable!(),
            };
            let mut r: u128 = 0;
            for p in 0..4u32 {
                r |= combine(w(a, 2 * p), w(a, 2 * p + 1)) << (p * 16);
                r |= combine(w(b, 2 * p), w(b, 2 * p + 1)) << ((p + 4) * 16);
            }
            r
        }
        HIntOp::AddD | HIntOp::SubD => {
            let combine = |x: i64, y: i64| match op {
                HIntOp::AddD => (x.wrapping_add(y)) as i32 as u32 as u128,
                HIntOp::SubD => (x.wrapping_sub(y)) as i32 as u32 as u128,
                _ => unreachable!(),
            };
            let mut r: u128 = 0;
            for p in 0..2u32 {
                r |= combine(d(a, 2 * p), d(a, 2 * p + 1)) << (p * 32);
                r |= combine(d(b, 2 * p), d(b, 2 * p + 1)) << ((p + 2) * 32);
            }
            r
        }
        HIntOp::Sad => {
            // `psadbw`: for each independent 64-bit half, sum the absolute
            // unsigned-byte differences of the eight bytes; write that 16-bit sum to the
            // low word of the half, zeroing bits 63:16. Max is 8*255 = 2040.
            let byte = |v: u128, i: u32| (v >> (i * 8)) as u8;
            let mut r: u128 = 0;
            for half in 0..2u32 {
                let mut sum: u32 = 0;
                for i in 0..8u32 {
                    let idx = half * 8 + i;
                    let (x, y) = (byte(a, idx), byte(b, idx));
                    sum += (x as i32 - y as i32).unsigned_abs();
                }
                r |= (sum as u128) << (half * 64);
            }
            r
        }
    }
}

/// Decode the stable op-code used by the JIT `hint` helper back to [`HIntOp`].
pub fn hint_op_from_code(code: u8) -> HIntOp {
    match code {
        0 => HIntOp::AddW,
        1 => HIntOp::AddD,
        2 => HIntOp::AddSw,
        3 => HIntOp::SubW,
        4 => HIntOp::SubD,
        5 => HIntOp::SubSw,
        _ => HIntOp::Sad,
    }
}

/// Register-form core: `v[dst] = hint(v[a], v[b])` per 128-bit lane over `bytes` (16 or
/// 32). Horizontal adds/subs pair adjacent lanes *within* each 128-bit lane and psadbw is
/// per-64-bit, so the 256-bit form is `hint` applied to both halves independently. Shared
/// by the interpreter dispatch and the JIT helper (jit == interp).
pub fn hint_reg(cpu: &mut CpuState, dst: u8, a: u8, b: u8, op: HIntOp, bytes: u16) {
    cpu.xmm[dst as usize] = hint(cpu.xmm[a as usize], cpu.xmm[b as usize], op);
    if bytes == 32 {
        cpu.ymm_hi[dst as usize] = hint(cpu.ymm_hi[a as usize], cpu.ymm_hi[b as usize], op);
    }
}

/// Memory-form core: `v[dst] = hint(v[dst], b)` per 128-bit lane, where `blo`/`bhi` are the
/// already-loaded memory operand lanes (`bhi` unused when `bytes == 16`). Shared by the
/// interpreter dispatch and the JIT helper.
pub fn hint_mem(cpu: &mut CpuState, dst: u8, a: u8, blo: u128, bhi: u128, op: HIntOp, bytes: u16) {
    // op1 from `a`, not read back out of `dst` (§16): it used to arrive through
    // a lift-emitted `VMov dst, a`, which wrote the destination before the load that can
    // fault. Both results are computed before either is committed, so `dst == a` stays
    // correct.
    let rlo = hint(cpu.xmm[a as usize], blo, op);
    let rhi = (bytes == 32).then(|| hint(cpu.ymm_hi[a as usize], bhi, op));
    cpu.xmm[dst as usize] = rlo;
    if let Some(rhi) = rhi {
        cpu.ymm_hi[dst as usize] = rhi;
    }
}

/// Scalar/packed float unary op. `dst_old` supplies the preserved upper lanes for
/// the scalar form; `src` is the operand.
fn float_unary(dst_old: u128, src: u128, op: FloatUnOp, prec: FPrec, scalar: bool) -> u128 {
    let bytes = prec.bytes() as u32;
    let lanes = if scalar { 1 } else { 16 / bytes as usize };
    let mut r = dst_old;
    for i in 0..lanes {
        let sh = i as u32 * bytes * 8;
        match prec {
            FPrec::F32 => {
                let v = apply_un_f32(f32::from_bits((src >> sh) as u32), op);
                r = (r & !(0xffff_ffffu128 << sh)) | ((v.to_bits() as u128) << sh);
            }
            FPrec::F64 => {
                let v = apply_un_f64(f64::from_bits((src >> sh) as u64), op);
                r = (r & !(0xffff_ffff_ffff_ffffu128 << sh)) | ((v.to_bits() as u128) << sh);
            }
        }
    }
    r
}

fn float_unary_mode<const DEFAULT: bool>(
    dst_old: u128,
    src: u128,
    op: FloatUnOp,
    prec: FPrec,
    scalar: bool,
    mxcsr: u32,
) -> u128 {
    if DEFAULT {
        return float_unary(dst_old, src, op, prec, scalar);
    }
    let mxcsr = if DEFAULT { 0 } else { mxcsr };
    let bytes = prec.bytes() as u32;
    let lanes = if scalar { 1 } else { 16 / bytes as usize };
    let mut r = dst_old;
    for i in 0..lanes {
        let sh = i as u32 * bytes * 8;
        match prec {
            FPrec::F32 => {
                let v = apply_un_f32_mode::<DEFAULT>(f32::from_bits((src >> sh) as u32), op, mxcsr);
                r = (r & !(0xffff_ffffu128 << sh)) | ((v.to_bits() as u128) << sh);
            }
            FPrec::F64 => {
                let v = apply_un_f64_mode::<DEFAULT>(f64::from_bits((src >> sh) as u64), op, mxcsr);
                r = (r & !(0xffff_ffff_ffff_ffffu128 << sh)) | ((v.to_bits() as u128) << sh);
            }
        }
    }
    r
}

/// One `sqrt`/`rsqrt`/`rcp` element. These take a single operand, so a NaN source is
/// returned quieted (SDM Vol 1 §4.8.3.5, Table 4-8, the single-operand rows) — neither
/// `llvm.sqrt` nor `1.0 / x` pins the payload, and `rsqrt` would launder it through two
/// such operations. A negative operand is an invalid operation with no NaN source, so it
/// yields the QNaN indefinite rather than whatever NaN the host produces (§4.8.3.7).
fn apply_un_f32(mut x: f32, op: FloatUnOp) -> f32 {
    if x.is_nan() {
        return quiet_f32(x);
    }
    // SDM Vol 2B §4.3 RCPPS/RCPSS/RSQRTPS/RSQRTSS: denormal inputs
    // are signed zero regardless of MXCSR.DAZ, before the negative-input test.
    if !matches!(op, FloatUnOp::Sqrt) && x.is_subnormal() {
        x = f32::from_bits(x.to_bits() & 0x8000_0000);
    }
    let indefinite = f32::from_bits(F32_INDEFINITE);
    match op {
        FloatUnOp::Sqrt => {
            if x < 0.0 {
                indefinite
            } else {
                x.sqrt()
            }
        }
        // Exact IEEE reciprocal-sqrt / reciprocal. Real hardware returns an
        // implementation-defined ~12-bit approximation; we compute the exact value, which is
        // within the SDM's guaranteed rel-error bound (1.5*2^-12). See FloatUnOp docs.
        FloatUnOp::Rsqrt => {
            if x < 0.0 {
                indefinite
            } else {
                1.0f32 / x.sqrt()
            }
        }
        FloatUnOp::Rcp => {
            let r = 1.0f32 / x;
            // SDM Vol 2B §4.3 RCPPS/RCPSS: flush tiny results even with FTZ clear.
            if r.is_subnormal() {
                f32::from_bits(r.to_bits() & 0x8000_0000)
            } else {
                r
            }
        }
    }
}

/// [`apply_un_f32`] at double precision.
fn apply_un_f64(x: f64, op: FloatUnOp) -> f64 {
    if x.is_nan() {
        return quiet_f64(x);
    }
    match op {
        FloatUnOp::Sqrt => {
            if x < 0.0 {
                f64::from_bits(F64_INDEFINITE)
            } else {
                x.sqrt()
            }
        }
        // rsqrt/rcp are single-precision only (no rsqrtpd/rcppd encodings) → never lifted F64.
        FloatUnOp::Rsqrt | FloatUnOp::Rcp => {
            unreachable!("rcp/rsqrt are single-precision only")
        }
    }
}

/// One `{add,sub,mul,div,min,max}{ss,sd,ps,pd}` element. `x` is SRC1, `y` is SRC2.
///
/// The arithmetic four go through [`sse_binop_f32`] for the architectural NaN result.
/// Every SSE and AVX float add, subtract, multiply and divide lands here, and a payload
/// difference only becomes a comparison failure once something downstream reads the bits
/// as an integer.
///
/// Min and max deliberately do **not**: their rule is the opposite one and it is already
/// right. "If only one value is a NaN (SNaN or QNaN) for this instruction, the second
/// operand (source operand) ... is written to the result", and "if a value in the second
/// operand is an SNaN, then SNaN is forwarded unchanged ... a QNaN version of the SNaN is
/// not returned" (SDM Vol 2 MINPS/MAXPS). `x < y` is false on a NaN either side, so `y`
/// — SRC2, unquieted — falls out, which is exactly that. Routing these through
/// `sse_binop_f32` would break them.
#[inline(always)]
fn apply_f32(x: f32, y: f32, op: FloatBinOp) -> f32 {
    match op {
        FloatBinOp::Add => sse_binop_f32(x, y, |p, q| p + q),
        FloatBinOp::Sub => sse_binop_f32(x, y, |p, q| p - q),
        FloatBinOp::Mul => sse_binop_f32(x, y, |p, q| p * q),
        FloatBinOp::Div => sse_binop_f32(x, y, |p, q| p / q),
        // x86 min/max: the second operand wins on NaN or equal (`x < y` / `x > y`
        // is false there, yielding `y`).
        FloatBinOp::Min => {
            if x < y {
                x
            } else {
                y
            }
        }
        FloatBinOp::Max => {
            if x > y {
                x
            } else {
                y
            }
        }
    }
}

/// [`apply_f32`] at double precision — same split: the arithmetic four take the NaN rule,
/// min and max keep theirs.
#[inline(always)]
fn apply_f64(x: f64, y: f64, op: FloatBinOp) -> f64 {
    match op {
        FloatBinOp::Add => sse_binop_f64(x, y, |p, q| p + q),
        FloatBinOp::Sub => sse_binop_f64(x, y, |p, q| p - q),
        FloatBinOp::Mul => sse_binop_f64(x, y, |p, q| p * q),
        FloatBinOp::Div => sse_binop_f64(x, y, |p, q| p / q),
        FloatBinOp::Min => {
            if x < y {
                x
            } else {
                y
            }
        }
        FloatBinOp::Max => {
            if x > y {
                x
            } else {
                y
            }
        }
    }
}

/// Round to nearest integer, ties to even (the default MXCSR rounding mode) —
/// `f64::round_ties_even` isn't available at our MSRV. Ties (`|frac| == 0.5`) only
/// occur below 2^52, where `floor as i64` can't overflow.
fn round_ties_even(f: f64) -> f64 {
    let floor = f.floor();
    let diff = f - floor;
    let r = if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else if (floor as i64) & 1 == 0 {
        floor
    } else {
        floor + 1.0
    };
    // A zero result keeps the input's sign (IEEE round-to-nearest): round(-0.5) = -0.0,
    // matching the hardware `roundps`/`roundpd`. Harmless for the int-cast caller.
    if r == 0.0 {
        (0.0f64).copysign(f)
    } else {
        r
    }
}

/// `cmpps`-family predicate on two floats. The low 8 predicates (imm[2:0], the legacy
/// SSE set): 0 EQ, 1 LT, 2 LE, 3 UNORD, 4 NEQ, 5 NLT, 6 NLE, 7 ORD (ordered comparisons
/// are false on a NaN; the "N"/UNORD forms are true). The VEX/AVX form widens `imm8` to
/// 5 bits (0..31, imm[4:0]) with 24 extra predicates; the extra bits distinguish
/// signaling (`_S`) from quiet (`_Q`/`_US`/`_UQ`) — a #IA-on-QNaN nuance we don't model —
/// and add the GE/GT/TRUE/FALSE orderings. Since our `partial_cmp` oracle carries no
/// signaling state, predicates differing only in the S/Q suffix collapse to the same
/// boolean, so the full 32 reduce to the eight distinct outcomes below, keyed on the
/// ordering. `_` handles pred 8..31 per the AVX table rather than aliasing to the low 8.
fn float_pred(ord: Option<Ordering>, pred: u8) -> bool {
    use Ordering::*;
    let unord = ord.is_none();
    match pred & 31 {
        // --- low 8 (legacy SSE) ---
        0x00 => ord == Some(Equal),                 // EQ_OQ
        0x01 => ord == Some(Less),                  // LT_OS
        0x02 => matches!(ord, Some(Less | Equal)),  // LE_OS
        0x03 => unord,                              // UNORD_Q
        0x04 => ord != Some(Equal),                 // NEQ_UQ (true if unordered)
        0x05 => ord != Some(Less),                  // NLT_US
        0x06 => !matches!(ord, Some(Less | Equal)), // NLE_US
        0x07 => !unord,                             // ORD_Q
        // --- extended AVX predicates 8..15 ---
        0x08 => ord == Some(Equal) || unord, // EQ_UQ (unordered-equal)
        0x09 => matches!(ord, Some(Less)) || unord, // NGE_US  = !(a>=b)
        0x0A => matches!(ord, Some(Less | Equal)) || unord, // NGT_US = !(a>b)
        0x0B => false,                       // FALSE_OQ
        0x0C => matches!(ord, Some(Less | Greater)), // NEQ_OQ  (ordered non-equal)
        0x0D => matches!(ord, Some(Greater | Equal)), // GE_OS
        0x0E => matches!(ord, Some(Greater)), // GT_OS
        0x0F => true,                        // TRUE_UQ
        // --- extended AVX predicates 16..23 (same booleans as 0..7, differ only in S/Q) ---
        0x10 => ord == Some(Equal),                 // EQ_OS
        0x11 => ord == Some(Less),                  // LT_OQ
        0x12 => matches!(ord, Some(Less | Equal)),  // LE_OQ
        0x13 => unord,                              // UNORD_S
        0x14 => ord != Some(Equal),                 // NEQ_US
        0x15 => ord != Some(Less),                  // NLT_UQ
        0x16 => !matches!(ord, Some(Less | Equal)), // NLE_UQ
        0x17 => !unord,                             // ORD_S
        // --- extended AVX predicates 24..31 (same booleans as 8..15) ---
        0x18 => ord == Some(Equal) || unord,        // EQ_US
        0x19 => matches!(ord, Some(Less)) || unord, // NGE_UQ
        0x1A => matches!(ord, Some(Less | Equal)) || unord, // NGT_UQ
        0x1B => false,                              // FALSE_OS
        0x1C => matches!(ord, Some(Less | Greater)), // NEQ_OS
        0x1D => matches!(ord, Some(Greater | Equal)), // GE_OQ
        0x1E => matches!(ord, Some(Greater)),       // GT_OQ
        _ => true,                                  // 0x1F TRUE_US
    }
}

/// Per-lane `cmp*` producing an all-ones/zero mask; `scalar` keeps `dst_old`'s
/// upper lanes.
fn float_cmp_mask_mode<const DEFAULT: bool>(
    dst_old: u128,
    a: u128,
    b: u128,
    prec: FPrec,
    scalar: bool,
    pred: u8,
    mxcsr: u32,
) -> u128 {
    let mxcsr = if DEFAULT { 0 } else { mxcsr };
    let a = daz_lanes(a, prec, mxcsr);
    let b = daz_lanes(b, prec, mxcsr);

    let bytes = prec.bytes() as u32;
    let lanes = if scalar { 1 } else { 16 / bytes as usize };
    let mut r = dst_old;
    for i in 0..lanes {
        let sh = i as u32 * bytes * 8;
        let ord = match prec {
            FPrec::F32 => {
                f32::from_bits((a >> sh) as u32).partial_cmp(&f32::from_bits((b >> sh) as u32))
            }
            FPrec::F64 => {
                f64::from_bits((a >> sh) as u64).partial_cmp(&f64::from_bits((b >> sh) as u64))
            }
        };
        let m = lane_mask(bytes as u8) << sh;
        r = (r & !m) | if float_pred(ord, pred) { m } else { 0 };
    }
    r
}

/// `ucomis*`/`comis*` flag result `(ZF, PF, CF)`. Unordered (a NaN operand) sets
/// all three; otherwise EQ→ZF, LT→CF, GT→none (x86 §COMISD).
fn float_compare_mode<const DEFAULT: bool>(
    a: u64,
    b: u64,
    prec: FPrec,
    mxcsr: u32,
) -> (bool, bool, bool) {
    let mxcsr = if DEFAULT { 0 } else { mxcsr };
    let a = softfloat::input(a, prec == FPrec::F64, mxcsr);
    let b = softfloat::input(b, prec == FPrec::F64, mxcsr);

    let ord = match prec {
        FPrec::F32 => f32::from_bits(a as u32).partial_cmp(&f32::from_bits(b as u32)),
        FPrec::F64 => f64::from_bits(a).partial_cmp(&f64::from_bits(b)),
    };
    match ord {
        None => (true, true, true),
        Some(Ordering::Equal) => (true, false, false),
        Some(Ordering::Less) => (false, false, true),
        Some(Ordering::Greater) => (false, false, false),
    }
}

fn alu_add(a: u64, b: u64, carry_in: u64, size: u8) -> AluResult {
    let m = mask(size);
    let (a, b) = (a & m, b & m);
    let wide = a as u128 + b as u128 + carry_in as u128;
    let res = (wide as u64) & m;
    let sb = sign_bit(size);
    AluResult {
        res,
        cf: (wide >> (size * 8)) & 1 != 0,
        pf: parity(res),
        af: ((a & 0xf) + (b & 0xf) + carry_in) & 0x10 != 0,
        zf: res == 0,
        sf: res & sb != 0,
        // signed overflow: operands same sign, result differs.
        of: (!(a ^ b) & (a ^ res)) & sb != 0,
    }
}

fn alu_sub(a: u64, b: u64, borrow_in: u64, size: u8) -> AluResult {
    let m = mask(size);
    let (a, b) = (a & m, b & m);
    let wide = (a as u128)
        .wrapping_sub(b as u128)
        .wrapping_sub(borrow_in as u128);
    let res = (wide as u64) & m;
    let sb = sign_bit(size);
    AluResult {
        res,
        cf: (a as u128) < (b as u128 + borrow_in as u128),
        pf: parity(res),
        af: (a & 0xf) < (b & 0xf) + borrow_in,
        zf: res == 0,
        sf: res & sb != 0,
        // signed overflow: operands differ in sign, result sign != a's sign.
        of: ((a ^ b) & (a ^ res)) & sb != 0,
    }
}

fn rotl(v: u64, cnt: u32, size: u8) -> u64 {
    match size {
        1 => (v as u8).rotate_left(cnt) as u64,
        2 => (v as u16).rotate_left(cnt) as u64,
        4 => (v as u32).rotate_left(cnt) as u64,
        _ => v.rotate_left(cnt),
    }
}

fn rotr(v: u64, cnt: u32, size: u8) -> u64 {
    match size {
        1 => (v as u8).rotate_right(cnt) as u64,
        2 => (v as u16).rotate_right(cnt) as u64,
        4 => (v as u32).rotate_right(cnt) as u64,
        _ => v.rotate_right(cnt),
    }
}

/// Rotate `v` (already masked to `size`) LEFT through CF by `cnt` positions (`cnt`
/// already reduced mod `size*8 + 1`). Returns `(result masked, CF-out)`. Bit-serial —
/// `cnt <= 64`, and rcl/rcr is rare.
fn rcl(mut v: u64, cnt: u32, cf_in: bool, size: u8) -> (u64, bool) {
    let bits = size as u32 * 8;
    let m = mask(size);
    let mut cf = cf_in;
    for _ in 0..cnt {
        let msb = (v >> (bits - 1)) & 1 != 0;
        v = ((v << 1) | cf as u64) & m;
        cf = msb;
    }
    (v, cf)
}

/// Rotate `v` (already masked to `size`) RIGHT through CF by `cnt` positions.
fn rcr(mut v: u64, cnt: u32, cf_in: bool, size: u8) -> (u64, bool) {
    let bits = size as u32 * 8;
    let mut cf = cf_in;
    for _ in 0..cnt {
        let lsb = v & 1 != 0;
        v = (v >> 1) | ((cf as u64) << (bits - 1));
        cf = lsb;
    }
    (v, cf)
}

/// Result carrying only CF/OF (rotates leave the other flags untouched).
fn cf_of(res: u64, cf: bool, of: bool) -> AluResult {
    AluResult {
        res,
        cf,
        pf: false,
        af: false,
        zf: false,
        sf: false,
        of,
    }
}

/// Flags for a shift with a nonzero count: SF/ZF/PF from the result, plus the
/// shift-specific CF and OF (AF is undefined and left out of the SHIFT mask).
fn shift_result(res: u64, size: u8, cf: bool, of: bool) -> AluResult {
    AluResult {
        res,
        cf,
        pf: parity(res),
        af: false,
        zf: res == 0,
        sf: res & sign_bit(size) != 0,
        of,
    }
}

/// Logic ops (and/or/xor/test): CF=OF=0, AF undefined (we clear it), SF/ZF/PF real.
fn alu_logic(res: u64, size: u8) -> AluResult {
    let res = res & mask(size);
    AluResult {
        res,
        cf: false,
        pf: parity(res),
        af: false,
        zf: res == 0,
        sf: res & sign_bit(size) != 0,
        of: false,
    }
}

fn apply(flags: &mut Flags, mask: FlagMask, r: &AluResult) {
    if mask.is_none() {
        return;
    }
    let m = mask.0;
    if m & 0b00_0001 != 0 {
        flags.cf = r.cf;
    }
    if m & 0b00_0010 != 0 {
        flags.set_pf(r.pf);
    }
    if m & 0b00_0100 != 0 {
        flags.set_af(r.af);
    }
    if m & 0b00_1000 != 0 {
        flags.zf = r.zf;
    }
    if m & 0b01_0000 != 0 {
        flags.sf = r.sf;
    }
    if m & 0b10_0000 != 0 {
        flags.of = r.of;
    }
}

fn eval_cond(cond: Cond, f: &Flags) -> bool {
    match cond {
        Cond::Eq => f.zf,
        Cond::Ne => !f.zf,
        Cond::Below => f.cf,
        Cond::AboveEq => !f.cf,
        Cond::BelowEq => f.cf || f.zf,
        Cond::Above => !f.cf && !f.zf,
        Cond::Less => f.sf != f.of,
        Cond::GreaterEq => f.sf == f.of,
        Cond::LessEq => (f.sf != f.of) || f.zf,
        Cond::Greater => (f.sf == f.of) && !f.zf,
        Cond::Sign => f.sf,
        Cond::NoSign => !f.sf,
        Cond::Overflow => f.of,
        Cond::NoOverflow => !f.of,
        Cond::Parity => f.pf(),
        Cond::NoParity => !f.pf(),
    }
}

#[cfg(test)]
mod real16_tests {
    //! Real-mode (16-bit) interpreter smoke tests (§17.6). These drive `step_one` over
    //! hand-assembled 16-bit snippets and check segmented addressing + near control
    //! flow. The full differential validation against Unicorn-16 lives in
    //! `x86jit-tests`; these are the fast in-crate confidence checks.
    use super::*;
    use crate::lift::CpuMode;
    use crate::memory::{MemoryModel, Prot, RegionKind};

    /// Run a Real16 program from CS:IP until it halts (or a budget runs out), returning
    /// the final `CpuState`. Code is placed at physical `cs<<4 + ip`; a flat 64 KiB+
    /// region is mapped so segment bases resolve.
    fn run16(
        cs: u16,
        ds: u16,
        es: u16,
        ss: u16,
        ip: u16,
        sp: u16,
        code: &[u8],
    ) -> (CpuState, Exit) {
        let mut m = Memory::new(MemoryModel::Flat { size: 0x2_0000 });
        m.map(0, 0x2_0000, Prot::RWX, RegionKind::Ram).unwrap();
        let pa = ((cs as u64) << 4) + ip as u64;
        m.write_bytes(pa, code).unwrap();

        let mut cpu = CpuState::new();
        cpu.cs = cs;
        cpu.ds = ds;
        cpu.es = es;
        cpu.ss = ss;
        cpu.rip = ip as u64;
        cpu.gpr[RSP] = sp as u64;
        let mut scratch = Vec::new();
        for _ in 0..64 {
            match step_one(
                &m,
                &mut cpu,
                CpuMode::Real16,
                &mut scratch,
                &mut Default::default(),
            ) {
                StepResult::Continue => {}
                StepResult::Exit(e) => return (cpu, e),
            }
        }
        panic!("run16 exceeded step budget");
    }

    /// BIOS-style: real-mode code executes straight out of a `RegionKind::Rom` region
    /// (mapped at physical `0xF0000`), and a guest store aimed back into the ROM is
    /// silently discarded — the masked-ROM contract — without faulting. Proves ROM is
    /// fetch/read-executable at RAM speed while stores are dropped.
    #[test]
    fn real_mode_executes_from_rom_and_a_store_to_it_is_dropped() {
        let mut m = Memory::new(MemoryModel::Flat { size: 0x10_0000 });
        m.map(0, 0x1000, Prot::RW, RegionKind::Ram).unwrap(); // low RAM (stack)
        m.map(0xF_0000, 0x20, Prot::RX, RegionKind::Rom).unwrap();
        // mov word [0x0000], 0x1234 ; hlt  — DS:0 resolves to phys 0xF0000, i.e. the
        // ROM's own first bytes, so a landed store would overwrite the opcode there.
        let code = [0xC7, 0x06, 0x00, 0x00, 0x34, 0x12, 0xF4];
        m.write_bytes(0xF_0000, &code).unwrap();

        let mut cpu = CpuState::new();
        cpu.cs = 0xF000;
        cpu.ds = 0xF000;
        cpu.ss = 0x0000;
        cpu.rip = 0;
        cpu.gpr[RSP] = 0x0F00;
        let mut scratch = Vec::new();
        let mut exit = None;
        for _ in 0..16 {
            match step_one(
                &m,
                &mut cpu,
                CpuMode::Real16,
                &mut scratch,
                &mut Default::default(),
            ) {
                StepResult::Continue => {}
                StepResult::Exit(e) => {
                    exit = Some(e);
                    break;
                }
            }
        }
        assert!(matches!(exit, Some(Exit::Hlt)), "ran to hlt, got {exit:?}");
        // The store was dropped: phys 0xF0000 still holds the original opcode bytes
        // (`C7 06` → little-endian word 0x06C7), not the stored 0x1234.
        assert_eq!(
            m.read(0xF_0000, 2).unwrap(),
            0x06C7,
            "ROM store must be dropped"
        );
    }

    /// `mov ax, [bx]` (DS:BX load) then `mov [bx+2], ax` (DS store), `hlt`. Seeds DS so
    /// the base is non-zero — proves the `sel<<4 + offset` fold.
    #[test]
    fn ds_segmented_load_store() {
        let cs = 0x0100;
        let ds = 0x0200; // base 0x2000
                         // At DS:0x0010 (phys 0x2010) place the word 0xBEEF.
        let mut m = Memory::new(MemoryModel::Flat { size: 0x2_0000 });
        m.map(0, 0x2_0000, Prot::RWX, RegionKind::Ram).unwrap();
        let cs_pa = (cs as u64) << 4;
        // mov bx,0x10 ; mov ax,[bx] ; mov [bx+2],ax ; hlt
        let code = [
            0xBB, 0x10, 0x00, // mov bx, 0x0010
            0x8B, 0x07, // mov ax, [bx]
            0x89, 0x47, 0x02, // mov [bx+2], ax
            0xF4, // hlt
        ];
        m.write_bytes(cs_pa, &code).unwrap();
        m.write_bytes(0x2010, &[0xEF, 0xBE]).unwrap(); // DS:0x10 = 0xBEEF

        let mut cpu = CpuState::new();
        cpu.cs = cs;
        cpu.ds = ds;
        cpu.ss = 0x0300;
        cpu.rip = 0;
        let mut scratch = Vec::new();
        loop {
            match step_one(
                &m,
                &mut cpu,
                CpuMode::Real16,
                &mut scratch,
                &mut Default::default(),
            ) {
                StepResult::Continue => {}
                StepResult::Exit(Exit::Hlt) => break,
                StepResult::Exit(e) => panic!("unexpected exit {e:?}"),
            }
        }
        assert_eq!(cpu.gpr[0] & 0xFFFF, 0xBEEF, "AX loaded from DS:[BX]");
        let mut back = [0u8; 2];
        m.read_bytes(0x2012, &mut back).unwrap(); // DS:0x12
        assert_eq!(back, [0xEF, 0xBE], "stored 0xBEEF to DS:[BX+2]");
    }

    /// `mov ax, [bp]` uses SS (not DS) implicitly. Seed SS != DS and confirm the SS base
    /// is used.
    #[test]
    fn bp_uses_ss_segment() {
        let cs = 0x0100;
        let ds = 0x0200;
        let ss = 0x0400; // base 0x4000
        let mut m = Memory::new(MemoryModel::Flat { size: 0x2_0000 });
        m.map(0, 0x2_0000, Prot::RWX, RegionKind::Ram).unwrap();
        let cs_pa = (cs as u64) << 4;
        // mov bp, 0x20 ; mov ax, [bp] ; hlt   (8B 46 00 = mov ax,[bp+0])
        let code = [0xBD, 0x20, 0x00, 0x8B, 0x46, 0x00, 0xF4];
        m.write_bytes(cs_pa, &code).unwrap();
        m.write_bytes(0x4020, &[0x34, 0x12]).unwrap(); // SS:0x20 = 0x1234
                                                       // A decoy at DS:0x20 to ensure DS is NOT used.
        m.write_bytes(0x2020, &[0xFF, 0xFF]).unwrap();

        let mut cpu = CpuState::new();
        cpu.cs = cs;
        cpu.ds = ds;
        cpu.ss = ss;
        cpu.rip = 0;
        let mut scratch = Vec::new();
        loop {
            match step_one(
                &m,
                &mut cpu,
                CpuMode::Real16,
                &mut scratch,
                &mut Default::default(),
            ) {
                StepResult::Continue => {}
                StepResult::Exit(Exit::Hlt) => break,
                StepResult::Exit(e) => panic!("unexpected exit {e:?}"),
            }
        }
        assert_eq!(cpu.gpr[0] & 0xFFFF, 0x1234, "AX loaded via SS:[BP]");
    }

    /// push/pop with a 16-bit SP through SS, plus a near call/ret round trip. The stack
    /// lives at SS:SP; the return IP must be popped correctly.
    #[test]
    fn push_pop_and_near_call_ret() {
        let cs = 0x0100;
        let ss = 0x0500; // base 0x5000
                         // Program at CS:0:
                         //   mov ax, 0x1234       B8 34 12
                         //   push ax              50
                         //   call 0x000A          E8 03 00   (target = next_ip(0x0007)+3 = 0x000A)
                         //   pop bx               5B          <- return lands here after ret
                         //   hlt                  F4
                         // at 0x000A:
                         //   mov cx, 0x5678       B9 78 56
                         //   ret                  C3
        let code = [
            0xB8, 0x34, 0x12, // 0x00 mov ax,0x1234
            0x50, // 0x03 push ax
            0xE8, 0x03, 0x00, // 0x04 call 0x000A
            0x5B, // 0x07 pop bx
            0xF4, // 0x08 hlt
            0x90, // 0x09 pad
            0xB9, 0x78, 0x56, // 0x0A mov cx,0x5678
            0xC3, // 0x0D ret
        ];
        let (cpu, exit) = run16(cs, 0x0200, 0x0300, ss, 0x0000, 0x0100, &code);
        assert!(matches!(exit, Exit::Hlt), "halted, got {exit:?}");
        assert_eq!(cpu.gpr[0] & 0xFFFF, 0x1234, "AX");
        assert_eq!(cpu.gpr[1] & 0xFFFF, 0x5678, "CX set inside the call");
        assert_eq!(cpu.gpr[3] & 0xFFFF, 0x1234, "BX popped the pushed AX");
        // SP returned to its start (push+pop and call+ret balance).
        assert_eq!(cpu.gpr[RSP] & 0xFFFF, 0x0100, "SP balanced");
    }

    /// SP wraps mod 2^16: a `push` at SP=0 writes at SS:0xFFFE and leaves SP=0xFFFE.
    #[test]
    fn sp_wraps_at_16_bits() {
        let cs = 0x0100;
        let ss = 0x0600;
        // mov ax,0xCAFE ; push ax ; hlt
        let code = [0xB8, 0xFE, 0xCA, 0x50, 0xF4];
        let (cpu, exit) = run16(cs, 0, 0, ss, 0x0000, 0x0000, &code);
        assert!(matches!(exit, Exit::Hlt));
        assert_eq!(cpu.gpr[RSP] & 0xFFFF, 0xFFFE, "SP wrapped to 0xFFFE");
        // The pushed word landed at SS:0xFFFE = phys 0x6000 + 0xFFFE.
        assert_eq!(cpu.ss, ss);
    }

    /// A near `jmp` wraps the IP to 16 bits (iced masks `near_branch_target`).
    #[test]
    fn near_jmp_sets_ip() {
        let cs = 0x0100;
        // jmp +2 (EB 02) over a ud-ish byte, then mov ax,1 ; hlt
        // 0x00 EB 02      jmp 0x04
        // 0x02 B8 FF FF   (skipped) mov ax,0xFFFF
        // 0x05 ...
        // Simpler: 0x00 EB 03 ; 0x02 90 90 90 ; 0x05 B8 01 00 ; 0x08 F4
        let code = [
            0xEB, 0x03, // jmp 0x05
            0x90, 0x90, 0x90, // skipped
            0xB8, 0x01, 0x00, // mov ax,1
            0xF4, // hlt
        ];
        let (cpu, exit) = run16(cs, 0, 0, 0x0700, 0x0000, 0x0100, &code);
        assert!(matches!(exit, Exit::Hlt));
        assert_eq!(cpu.gpr[0] & 0xFFFF, 0x0001, "AX set after the jmp target");
    }

    // --- sub-seam (b): interrupt-flag + INT/IRET/IVT (§17.6) ---

    /// `cli` clears IF, `sti` sets it (plain set — no STI-shadow, §17.6).
    #[test]
    fn cli_sti_toggle_if() {
        let cs = 0x0100;
        // sti ; cli ; sti ; hlt  → ends with IF set.
        let (cpu, exit) = run16(cs, 0, 0, 0x0700, 0, 0x100, &[0xFB, 0xFA, 0xFB, 0xF4]);
        assert!(matches!(exit, Exit::Hlt));
        assert!(cpu.flags.if_, "IF set by the trailing sti");

        // sti ; cli ; hlt  → ends with IF clear.
        let (cpu, _) = run16(cs, 0, 0, 0x0700, 0, 0x100, &[0xFB, 0xFA, 0xF4]);
        assert!(!cpu.flags.if_, "IF cleared by cli");
    }

    /// `pushf`/`popf` round-trip the FLAGS image including IF (bit 9).
    #[test]
    fn pushf_popf_round_trips_if() {
        let cs = 0x0100;
        let ss = 0x0700;
        // sti ; pushf ; cli ; popf ; hlt
        //   sti sets IF=1; pushf saves image(IF=1); cli clears IF; popf restores IF=1.
        let code = [0xFB, 0x9C, 0xFA, 0x9D, 0xF4];
        let (cpu, exit) = run16(cs, 0, 0, ss, 0, 0x100, &code);
        assert!(matches!(exit, Exit::Hlt));
        assert!(cpu.flags.if_, "popf restored IF=1 from the pushed image");
        assert_eq!(cpu.gpr[RSP] & 0xFFFF, 0x100, "SP balanced after pushf/popf");
        // The pushed image sat at SS:(0x100-2); bit 9 (IF) and bit 1 (reserved) set.
    }

    /// `int n` pushes FLAGS/CS/IP, clears IF, and vectors through the IVT; the handler's
    /// `iret` restores IF and returns. Verifies the stack image and the IF transition.
    #[test]
    fn int_delivers_and_iret_returns() {
        let cs = 0x0100; // caller CS, base 0x1000
        let ss = 0x0700; // stack base 0x7000
        let hcs = 0x0300u16; // handler CS, base 0x3000
        let vector = 0x21u16;

        let mut m = Memory::new(MemoryModel::Flat { size: 0x2_0000 });
        m.map(0, 0x2_0000, Prot::RWX, RegionKind::Ram).unwrap();

        // Caller at CS:0 — sti ; int 0x21 ; mov bx,0xB00B ; hlt
        let caller = [0xFB, 0xCD, 0x21, 0xBB, 0x0B, 0xB0, 0xF4];
        m.write_bytes((cs as u64) << 4, &caller).unwrap();
        // Handler at HCS:0 — mov ax,0x1234 ; iret
        let handler = [0xB8, 0x34, 0x12, 0xCF];
        m.write_bytes((hcs as u64) << 4, &handler).unwrap();
        // IVT[0x21]: IP=0x0000, CS=hcs at phys 0x21*4.
        m.write_bytes(vector as u64 * 4, &[0x00, 0x00]).unwrap();
        m.write_bytes(vector as u64 * 4 + 2, &hcs.to_le_bytes())
            .unwrap();

        let mut cpu = CpuState::new();
        cpu.cs = cs;
        cpu.ss = ss;
        cpu.rip = 0;
        cpu.gpr[RSP] = 0x0100;
        let mut scratch = Vec::new();
        // Step through: sti, int (delivers), mov ax (handler), iret (returns), mov bx, hlt.
        let mut saw_handler = false;
        let exit = loop {
            match step_one(
                &m,
                &mut cpu,
                CpuMode::Real16,
                &mut scratch,
                &mut Default::default(),
            ) {
                StepResult::Continue => {
                    // Right after `int` delivery: IF cleared, CS switched to the handler.
                    if cpu.cs == hcs && !saw_handler {
                        saw_handler = true;
                        assert!(!cpu.flags.if_, "IF cleared on int entry");
                        // Stack image: at SS:SP is IP(next)=3, then CS=cs, then FLAGS.
                        let sp = cpu.gpr[RSP] & 0xFFFF;
                        let base = (ss as u64) << 4;
                        let ip = m.read(base + sp, 2).unwrap();
                        let scs = m.read(base + sp + 2, 2).unwrap();
                        let flg = m.read(base + sp + 4, 2).unwrap();
                        assert_eq!(ip, 3, "pushed return IP = next instr (after int 0x21)");
                        assert_eq!(scs, cs as u64, "pushed caller CS");
                        assert!(
                            flg & (1 << 9) != 0,
                            "pushed FLAGS had IF=1 (sti before int)"
                        );
                    }
                }
                StepResult::Exit(e) => break e,
            }
        };
        assert!(matches!(exit, Exit::Hlt));
        assert!(saw_handler, "handler ran");
        assert_eq!(cpu.cs, cs, "iret returned to the caller CS");
        assert_eq!(cpu.gpr[0] & 0xFFFF, 0x1234, "handler set AX");
        assert_eq!(cpu.gpr[3] & 0xFFFF, 0xB00B, "caller resumed and set BX");
        assert!(cpu.flags.if_, "iret restored IF=1");
        assert_eq!(cpu.gpr[RSP] & 0xFFFF, 0x0100, "SP balanced across int/iret");
    }

    /// A divide-by-zero (`#DE`, vector 0) in real mode vectors through the IVT in-guest
    /// (via the `Vcpu::run` loop), not as an `Exit::Exception`. Long64/Compat32 keep
    /// returning `Exit::Exception`.
    #[test]
    fn divide_error_vectors_through_ivt() {
        use crate::vm::{Vm, VmConfig};
        use crate::MemConsistency;

        let cs = 0x0100u16;
        let ss = 0x0700u16;
        let hcs = 0x0300u16;

        let mut vm = Vm::new(VmConfig {
            memory_model: MemoryModel::Flat { size: 0x2_0000 },
            consistency: MemConsistency::Fast,
        });
        vm.set_cpu_mode(CpuMode::Real16);
        vm.map(0, 0x2_0000, Prot::RWX, RegionKind::Ram).unwrap();

        // Caller: xor dx,dx ; mov ax,1 ; mov cx,0 ; div cx  (→ #DE, divide by 0)
        //         then never reached: hlt.  (div cx = F7 F1)
        let caller = [
            0x31, 0xD2, // xor dx,dx
            0xB8, 0x01, 0x00, // mov ax,1
            0xB9, 0x00, 0x00, // mov cx,0
            0xF7, 0xF1, // div cx  → #DE at this IP (=8)
            0xF4, // hlt (unreached)
        ];
        vm.mem.write_bytes((cs as u64) << 4, &caller).unwrap();
        // #DE handler at HCS:0 — mov bx,0xDEAD ; hlt
        let handler = [0xBB, 0xAD, 0xDE, 0xF4];
        vm.mem.write_bytes((hcs as u64) << 4, &handler).unwrap();
        // IVT[0]: IP=0, CS=hcs.
        vm.mem.write_bytes(0, &[0x00, 0x00]).unwrap();
        vm.mem.write_bytes(2, &hcs.to_le_bytes()).unwrap();

        let mut vcpu = vm.new_vcpu();
        vcpu.cpu.cs = cs;
        vcpu.cpu.ss = ss;
        vcpu.cpu.rip = 0;
        vcpu.cpu.gpr[RSP] = 0x0100;

        let exit = vcpu.run(&vm, Some(64));
        assert!(matches!(exit, Exit::Hlt), "handler halted, got {exit:?}");
        assert_eq!(vcpu.cpu.cs, hcs, "vectored to the #DE handler CS");
        assert_eq!(vcpu.cpu.gpr[3] & 0xFFFF, 0xDEAD, "handler set BX");
        // The saved IP is the faulting `div` (IP 8), not the next instruction. The
        // handler never touched the stack, so SP still points at the pushed frame's IP
        // word (lowest of the FLAGS/CS/IP frame). SP = 0x100 - 6 = 0xFA.
        assert_eq!(
            vcpu.cpu.gpr[RSP] & 0xFFFF,
            0x00FA,
            "frame is 6 bytes (FLAGS/CS/IP)"
        );
        let base = (ss as u64) << 4;
        let ip = vm.mem.read(base + (vcpu.cpu.gpr[RSP] & 0xFFFF), 2).unwrap();
        assert_eq!(ip, 8, "#DE pushed the faulting div's IP (fault, not trap)");
    }

    /// `Vcpu::step_instruction` runs exactly ONE instruction (not the block/run-on the
    /// `run` loop would), and in Real16 delivers a `#DE` in-guest through the IVT
    /// (returning `Continue` on the handler), not as an `Exit::Exception`. This is the
    /// per-instruction primitive the TomHarte 8088 corpus oracle drives.
    #[test]
    fn step_instruction_single_steps_and_vectors_de() {
        use crate::vm::{Vm, VmConfig};
        use crate::MemConsistency;

        let (cs, hcs) = (0x0100u16, 0x0300u16);
        let mut vm = Vm::new(VmConfig {
            memory_model: MemoryModel::Flat { size: 0x2_0000 },
            consistency: MemConsistency::Fast,
        });
        vm.set_cpu_mode(CpuMode::Real16);
        vm.map(0, 0x2_0000, Prot::RWX, RegionKind::Ram).unwrap();

        // `mov ax,1 ; div cl` with CL=0 → #DE on the `div`. NOPs follow so a run-on
        // would keep advancing; a single step must stop after exactly one instruction.
        let code = [
            0xB8, 0x01, 0x00, // mov ax,1        (len 3)
            0xF6, 0xF1, // div cl (CL=0)   (len 2, #DE)
            0x90, 0x90, // nop nop
        ];
        vm.mem.write_bytes((cs as u64) << 4, &code).unwrap();
        // #DE handler at HCS:0; IVT[0] → HCS:0.
        vm.mem.write_bytes((hcs as u64) << 4, &[0x90]).unwrap();
        vm.mem.write_bytes(0, &[0x00, 0x00]).unwrap();
        vm.mem.write_bytes(2, &hcs.to_le_bytes()).unwrap();

        let mut vcpu = vm.new_vcpu();
        vcpu.cpu.cs = cs;
        vcpu.cpu.ss = 0x0700;
        vcpu.cpu.rip = 0;
        vcpu.cpu.gpr[RSP] = 0x0100;
        vcpu.cpu.gpr[1] = 0; // CX (CL=0)

        // Step 1: `mov ax,1` — advances IP by 3 and only that.
        assert!(matches!(vcpu.step_instruction(&vm), StepResult::Continue));
        assert_eq!(vcpu.cpu.rip, 3, "one instruction retired, no run-on");
        assert_eq!(vcpu.cpu.gpr[0] & 0xFFFF, 1, "mov ax,1 executed");

        // Step 2: `div cl` faults #DE and vectors in-guest — Continue, not Exception.
        assert!(matches!(vcpu.step_instruction(&vm), StepResult::Continue));
        assert_eq!(vcpu.cpu.cs, hcs, "#DE vectored to the handler CS");
        assert_eq!(vcpu.cpu.rip, 0, "handler entry IP");
    }

    // --- sub-seam (c): hardware-interrupt injection + retired-instruction counter ---

    use crate::vm::{Vcpu, Vm, VmConfig};
    use crate::MemConsistency;

    /// Build a flat Real16 `Vm` + `Vcpu` seeded at CS:IP=cs:0, SS:SP=ss:0x100, and write
    /// `code` at CS:0. Returns both so the caller can seed extra memory / inspect it.
    fn real16_vm(cs: u16, ss: u16, code: &[u8]) -> (Vm, Vcpu) {
        let mut vm = Vm::new(VmConfig {
            memory_model: MemoryModel::Flat { size: 0x2_0000 },
            consistency: MemConsistency::Fast,
        });
        vm.set_cpu_mode(CpuMode::Real16);
        vm.map(0, 0x2_0000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.mem.write_bytes((cs as u64) << 4, code).unwrap();
        let mut vcpu = vm.new_vcpu();
        vcpu.cpu.cs = cs;
        vcpu.cpu.ss = ss;
        vcpu.cpu.rip = 0;
        vcpu.cpu.gpr[RSP] = 0x0100;
        (vm, vcpu)
    }

    /// Seed IVT[vector] → handler at HCS:0 and write the handler bytes there.
    fn install_handler(vm: &mut Vm, vector: u8, hcs: u16, handler: &[u8]) {
        vm.mem.write_bytes((hcs as u64) << 4, handler).unwrap();
        vm.mem
            .write_bytes(vector as u64 * 4, &[0x00, 0x00])
            .unwrap();
        vm.mem
            .write_bytes(vector as u64 * 4 + 2, &hcs.to_le_bytes())
            .unwrap();
    }

    /// The retired-instruction counter ticks exactly once per straight-line instruction
    /// (§17.6, sub-seam c). `sti; nop; nop; mov ax,1; hlt` = 5 instructions.
    #[test]
    fn retired_counter_counts_straight_line() {
        // FB nop nop B8 01 00 F4  → sti, nop, nop, mov ax,0x0001, hlt
        let code = [0xFB, 0x90, 0x90, 0xB8, 0x01, 0x00, 0xF4];
        let (vm, mut vcpu) = real16_vm(0x0100, 0x0700, &code);
        let exit = vcpu.run(&vm, Some(64));
        assert!(matches!(exit, Exit::Hlt));
        assert_eq!(
            vcpu.retired_instructions(),
            5,
            "sti+nop+nop+mov+hlt = 5 retired"
        );
    }

    /// A memory trap does NOT retire the faulting instruction; on completion+retry it is
    /// counted exactly once. `mov ax,[0]` to an unmapped hole traps; after mapping and
    /// re-running it retires. (Constructs the trap via a hole in a non-flat map.)
    #[test]
    fn retired_counter_excludes_trapping_instruction() {
        // A Vm whose only mapped region is CS's page; DS:0x8000 (phys) is unmapped.
        let mut vm = Vm::new(VmConfig {
            memory_model: MemoryModel::Flat { size: 0x2_0000 },
            consistency: MemConsistency::Fast,
        });
        vm.set_cpu_mode(CpuMode::Real16);
        // Map only [0, 0x4000) — the code lives at 0x1000; DS:0 (phys 0x8000) is a hole.
        vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
        // mov ax,[0x0000] with DS base 0x8000 → phys 0x8000 (unmapped) ; hlt
        let code = [0xA1, 0x00, 0x00, 0xF4];
        vm.mem.write_bytes(0x1000, &code).unwrap();
        let mut vcpu = vm.new_vcpu();
        vcpu.cpu.cs = 0x0100; // base 0x1000
        vcpu.cpu.ds = 0x0800; // base 0x8000 (unmapped)
        vcpu.cpu.ss = 0x0700;
        vcpu.cpu.rip = 0;
        vcpu.cpu.gpr[RSP] = 0x0100;
        let exit = vcpu.run(&vm, Some(64));
        assert!(
            matches!(exit, Exit::UnmappedMemory { .. }),
            "trapped: {exit:?}"
        );
        assert_eq!(
            vcpu.retired_instructions(),
            0,
            "the faulting mov did not retire"
        );
    }

    /// Injection delivers at a run boundary when IF is set: the handler runs, the frame
    /// (FLAGS/CS/IP) is on SS:SP, IF is cleared on entry, and `iret` restores IF and
    /// returns to the interrupted point (§17.6, sub-seam c). Delivery is at a BLOCK
    /// boundary — never mid-block — so the guest is written with a `jmp` that ends the
    /// `sti` block, giving the dispatcher a boundary (with IF now set and the STI shadow
    /// elapsed by the `jmp`) at which to vector.
    #[test]
    fn injection_delivers_when_if_set() {
        // 0000: FB          sti
        // 0001: E9 00 00     jmp 0x0004        (ends the block; boundary with IF=1)
        // 0004: 90          nop                (the interrupted instruction)
        // 0005: F4          hlt
        let cs = 0x0100u16;
        let ss = 0x0700u16;
        let hcs = 0x0300u16;
        let code = [0xFB, 0xE9, 0x00, 0x00, 0x90, 0xF4];
        let (mut vm, mut vcpu) = real16_vm(cs, ss, &code);
        // Handler at HCS:0 — mov ax,0x1234 ; iret
        install_handler(&mut vm, 0x30, hcs, &[0xB8, 0x34, 0x12, 0xCF]);

        vcpu.inject_irq(0x30);
        // Peek the frame the moment it is pushed: step the vcpu to the point the handler
        // is entered by running with a small budget and inspecting SS:SP. Simpler: run to
        // completion and assert the observable effects; the frame contents are validated
        // byte-exact in the cf16 differential.
        let exit = vcpu.run(&vm, Some(64));
        assert!(matches!(exit, Exit::Hlt), "returned {exit:?}");
        assert_eq!(vcpu.cpu.gpr[0] & 0xFFFF, 0x1234, "handler set AX");
        assert_eq!(vcpu.cpu.cs, cs, "iret returned to caller CS");
        assert!(vcpu.cpu.flags.if_, "iret restored IF=1");
        assert_eq!(
            vcpu.cpu.gpr[RSP] & 0xFFFF,
            0x0100,
            "SP balanced across iret"
        );
        assert!(
            !vcpu.has_pending_irq(),
            "the vector was consumed, not left queued"
        );
    }

    /// The pushed interrupt frame (FLAGS/CS/IP) and the IF-cleared-on-entry are checked by
    /// driving the vcpu one block at a time and inspecting SS:SP the instant the handler
    /// is entered (§17.6, sub-seam c).
    #[test]
    fn injection_frame_and_if_clear_on_entry() {
        // 0000: FB          sti
        // 0001: E9 00 00     jmp 0x0004
        // 0004: 90          nop
        // 0005: F4          hlt
        let cs = 0x0100u16;
        let ss = 0x0700u16;
        let hcs = 0x0300u16;
        let code = [0xFB, 0xE9, 0x00, 0x00, 0x90, 0xF4];
        let (mut vm, mut vcpu) = real16_vm(cs, ss, &code);
        install_handler(&mut vm, 0x30, hcs, &[0xB8, 0x34, 0x12, 0xCF]);
        vcpu.inject_irq(0x30);

        // Run block-by-block (budget 1) until we observe CS switch to the handler.
        let mut entered = false;
        for _ in 0..8 {
            let e = vcpu.run(&vm, Some(1));
            if vcpu.cpu.cs == hcs && !entered {
                entered = true;
                assert!(!vcpu.cpu.flags.if_, "IF cleared on interrupt entry");
                let sp = vcpu.cpu.gpr[RSP] & 0xFFFF;
                let base = (ss as u64) << 4;
                let ip = vm.mem.read(base + sp, 2).unwrap();
                let scs = vm.mem.read(base + sp + 2, 2).unwrap();
                let flg = vm.mem.read(base + sp + 4, 2).unwrap();
                assert_eq!(ip, 0x0004, "return IP = interrupted nop");
                assert_eq!(scs, cs as u64, "pushed caller CS");
                assert!(flg & (1 << 9) != 0, "pushed FLAGS had IF=1 (from sti)");
            }
            if matches!(e, Exit::Hlt) {
                break;
            }
        }
        assert!(entered, "handler was entered");
    }

    /// With IF clear (`cli`) an injected IRQ is masked: the guest runs to `hlt` without
    /// vectoring. (The `sti` path is covered by `injection_delivers_when_if_set`.)
    #[test]
    fn injection_masked_while_if_clear() {
        let cs = 0x0100u16;
        let hcs = 0x0300u16;
        // cli ; nop ; hlt  — IF stays clear, so the IRQ never delivers.
        let code = [0xFA, 0x90, 0xF4];
        let (mut vm, mut vcpu) = real16_vm(cs, 0x0700, &code);
        install_handler(&mut vm, 0x30, hcs, &[0xB8, 0x34, 0x12, 0xCF]);
        vcpu.inject_irq(0x30);
        let exit = vcpu.run(&vm, Some(64));
        assert!(matches!(exit, Exit::Hlt));
        assert_eq!(vcpu.cpu.cs, cs, "never vectored to the handler");
        assert_ne!(vcpu.cpu.gpr[0] & 0xFFFF, 0x1234, "handler did not run");
        assert!(vcpu.has_pending_irq(), "vector stays queued (masked)");
    }

    /// The STI shadow deferring delivery by exactly one boundary, driven at the
    /// `Vcpu::run` level. Because this lifter never makes `sti` a block *terminator*, the
    /// deferral is observed on the single-instruction path: budget=1 runs one block; when
    /// a block is exactly `sti` (isolated via a preceding branch boundary so it stands
    /// alone before the next block), the dispatcher reports the shadow and holds a pending
    /// IRQ for that boundary, delivering only after the following block runs an
    /// instruction. Here the guest `jmp`s to an `sti` that is then followed by its own
    /// block, and we assert the handler's return IP is the instruction AFTER the one that
    /// cleared the shadow — never a point mid-way through the `sti` window.
    #[test]
    fn sti_shadow_defers_one_instruction() {
        // 0000: E9 03 00    jmp 0x0006        (skip the handler-marker gap; boundary)
        // 0003: (unused)
        // 0006: FB          sti
        // 0007: 90          nop               (clears the shadow)
        // 0008: F4          hlt
        // The block at 0x0006 is [sti; nop; hlt] — a single block; the run-boundary
        // delivery after this block naturally fires only after the whole block, i.e. the
        // interrupt cannot land between `sti` and `nop`. Since the block ends on `hlt`
        // (not `sti`), the shadow is elapsed and delivery happens at the post-hlt boundary
        // on re-entry (the HLT-wakeup path). We assert the handler runs and control
        // resumes past the hlt — the interrupt never fired inside the sti/nop pair.
        let cs = 0x0100u16;
        let ss = 0x0700u16;
        let hcs = 0x0300u16;
        let code = [
            0xE9, 0x03, 0x00, // jmp 0x0006
            0x00, 0x00, 0x00, // padding (unreached)
            0xFB, // sti      (0x0006)
            0x90, // nop       (0x0007)
            0xF4, // hlt       (0x0008)  first halt
            0xF4, // hlt       (0x0009)  resume-past-hlt lands here (terminates)
        ];
        let (mut vm, mut vcpu) = real16_vm(cs, ss, &code);
        // Handler: mov ax,0xBEEF ; iret
        install_handler(&mut vm, 0x30, hcs, &[0xB8, 0xEF, 0xBE, 0xCF]);
        vcpu.inject_irq(0x30);
        // First run: the sti;nop;hlt block runs to hlt (IRQ deferred — the boundary is the
        // Exit::Hlt). IF is set; the vector stays queued.
        let e1 = vcpu.run(&vm, Some(64));
        assert!(matches!(e1, Exit::Hlt), "halted, got {e1:?}");
        assert!(vcpu.cpu.flags.if_, "IF set by sti");
        assert!(vcpu.has_pending_irq(), "IRQ still queued after the block");
        assert_ne!(
            vcpu.cpu.gpr[0] & 0xFFFF,
            0xBEEF,
            "handler did NOT fire mid-block"
        );
        // Re-entry delivers at the post-hlt boundary (HLT-wakeup): handler runs, iret
        // resumes past the hlt.
        let e2 = vcpu.run(&vm, Some(64));
        assert!(matches!(e2, Exit::Hlt), "second halt, got {e2:?}");
        assert_eq!(
            vcpu.cpu.gpr[0] & 0xFFFF,
            0xBEEF,
            "handler ran on the boundary"
        );
        assert_eq!(vcpu.cpu.cs, cs, "iret returned to caller CS");
    }

    /// A tighter STI-shadow check on the interpreter's `RetireInfo::sti_shadow`: `sti` as
    /// the last dispatched instruction arms the shadow; the next instruction clears it.
    #[test]
    fn sti_shadow_flag_set_only_when_sti_is_last() {
        // Block "sti" alone (single-instruction block via step_one): sti is the last (and
        // only) retired instruction → sti_shadow must be reported.
        let mut m = Memory::new(MemoryModel::Flat { size: 0x2_0000 });
        m.map(0, 0x2_0000, Prot::RWX, RegionKind::Ram).unwrap();
        m.write_bytes(0x1000, &[0xFB]).unwrap(); // sti at CS:0 (base 0x1000)
        let mut cpu = CpuState::new();
        cpu.cs = 0x0100;
        cpu.rip = 0;
        let mut scratch = Vec::new();
        let mut info = RetireInfo::default();
        let r = step_one(&m, &mut cpu, CpuMode::Real16, &mut scratch, &mut info);
        assert!(matches!(r, StepResult::Continue));
        assert_eq!(info.retired, 1, "sti retired");
        assert!(info.sti_shadow, "sti as the last insn arms the shadow");

        // Now a following `nop` clears the shadow.
        m.write_bytes(0x1001, &[0x90]).unwrap(); // nop at IP 1
        let mut info2 = RetireInfo::default();
        let r2 = step_one(&m, &mut cpu, CpuMode::Real16, &mut scratch, &mut info2);
        assert!(matches!(r2, StepResult::Continue));
        assert_eq!(info2.retired, 1, "nop retired");
        assert!(!info2.sti_shadow, "nop clears the STI shadow");
    }

    /// HLT wakeup: a `hlt` with IF set returns `Exit::Hlt`; the embedder then injects and
    /// re-enters `run()`, which delivers the IRQ (vectoring the handler) and, on `iret`,
    /// resumes execution past the `hlt` (§17.6, sub-seam c).
    #[test]
    fn hlt_wakeup_delivers_injected_irq() {
        let cs = 0x0100u16;
        let ss = 0x0700u16;
        let hcs = 0x0300u16;
        // sti ; hlt ; mov bx,0xB00B ; hlt   — first hlt returns Exit::Hlt; after the IRQ
        // handler iret's, execution resumes at `mov bx` then the second hlt.
        let code = [0xFB, 0xF4, 0xBB, 0x0B, 0xB0, 0xF4];
        let (mut vm, mut vcpu) = real16_vm(cs, ss, &code);
        // Handler: mov ax,0xCAFE ; iret
        install_handler(&mut vm, 0x30, hcs, &[0xB8, 0xFE, 0xCA, 0xCF]);

        // First run halts at the `hlt` (IF set, no IRQ pending yet).
        let exit1 = vcpu.run(&vm, Some(64));
        assert!(
            matches!(exit1, Exit::Hlt),
            "first run halted, got {exit1:?}"
        );
        assert_eq!(vcpu.cpu.gpr[3] & 0xFFFF, 0x0000, "mov bx not yet run");
        let rip_after_hlt = vcpu.cpu.rip;
        assert_eq!(rip_after_hlt, 2, "RIP past the hlt (IP 2 = mov bx)");

        // Embedder injects and re-enters: the IRQ is delivered, handler runs, iret returns
        // to the instruction after the hlt, which then runs to the terminating hlt.
        vcpu.inject_irq(0x30);
        let exit2 = vcpu.run(&vm, Some(64));
        assert!(
            matches!(exit2, Exit::Hlt),
            "second run halted, got {exit2:?}"
        );
        assert_eq!(vcpu.cpu.gpr[0] & 0xFFFF, 0xCAFE, "handler ran on wakeup");
        assert_eq!(
            vcpu.cpu.gpr[3] & 0xFFFF,
            0xB00B,
            "resumed past hlt (set BX)"
        );
        assert_eq!(vcpu.cpu.cs, cs, "iret returned to caller CS");
    }

    /// Pending-completion guard: while a `pending_port_in` is outstanding (an `in`
    /// awaiting `complete_port_in`), an injected IRQ is deferred — it is delivered only
    /// after the completion clears (§17.6, sub-seam c).
    #[test]
    fn injection_deferred_while_port_in_pending() {
        let cs = 0x0100u16;
        let ss = 0x0700u16;
        let hcs = 0x0300u16;
        // sti ; in al,0x60 ; mov bx,0x0BAD ; hlt
        //   FB   E4 60       BB AD 0B        F4
        let code = [0xFB, 0xE4, 0x60, 0xBB, 0xAD, 0x0B, 0xF4];
        let (mut vm, mut vcpu) = real16_vm(cs, ss, &code);
        install_handler(&mut vm, 0x30, hcs, &[0xB8, 0x99, 0x99, 0xCF]); // mov ax,0x9999;iret

        vcpu.inject_irq(0x30);
        // First run stops at the `in` (PortIo) with the IRQ still queued (IF is set, but a
        // port-in completion is now pending, blocking delivery).
        let exit1 = vcpu.run(&vm, Some(64));
        assert!(
            matches!(
                exit1,
                Exit::PortIo {
                    dir: crate::exit::PortDir::In,
                    ..
                }
            ),
            "stopped on IN, got {exit1:?}"
        );
        assert!(
            vcpu.cpu.pending_port_in.is_some(),
            "port-in completion pending"
        );
        assert!(vcpu.has_pending_irq(), "IRQ deferred, still queued");

        // Supply the port value; the completion clears and the deferred IRQ now delivers
        // at the next boundary.
        vcpu.complete_port_in(0x42);
        let exit2 = vcpu.run(&vm, Some(64));
        assert!(matches!(exit2, Exit::Hlt), "ran to hlt, got {exit2:?}");
        assert_eq!(
            vcpu.cpu.gpr[0] & 0xFFFF,
            0x9999,
            "handler ran after completion"
        );
        assert_eq!(vcpu.cpu.gpr[3] & 0xFFFF, 0x0BAD, "mov bx ran (post-in)");
    }
}

#[cfg(test)]
mod bmi_tests {
    use super::bmi_result;
    use crate::ir::BmiOp::*;

    #[test]
    fn bmi_result_semantics() {
        // andn: ~a & b
        assert_eq!(bmi_result(0xF0, 0x0F, 1, Andn), (0x0F, false));
        assert_eq!(bmi_result(0xFF, 0x0F, 1, Andn), (0x00, false));
        // blsi: isolate lowest set bit; CF = a != 0
        assert_eq!(bmi_result(0x0C, 0, 4, Blsi), (0x04, true));
        assert_eq!(bmi_result(0, 0, 4, Blsi), (0, false));
        // blsr: clear lowest set bit; CF = a == 0
        assert_eq!(bmi_result(0x0C, 0, 4, Blsr), (0x08, false));
        assert_eq!(bmi_result(0, 0, 4, Blsr), (0, true));
        // blsmsk: mask up to lowest set bit; CF = a == 0
        assert_eq!(bmi_result(0x0C, 0, 4, Blsmsk), (0x07, false));
        assert_eq!(bmi_result(0, 0, 4, Blsmsk), (0xFFFF_FFFF, true));
        // bextr: extract `len` bits from `start` (ctrl = start | len<<8)
        assert_eq!(bmi_result(0xABCD, 4 | (8 << 8), 4, Bextr), (0xBC, false));
        // bzhi: zero bits from index up; CF = index > width-1
        assert_eq!(bmi_result(0xFFFF, 8, 4, Bzhi), (0xFF, false));
        assert_eq!(bmi_result(0xFFFF, 40, 4, Bzhi), (0xFFFF, true)); // idx > 31
                                                                     // pdep: deposit low bits of a into mask positions (1,2,4).
        assert_eq!(bmi_result(0b1011, 0b1_0110, 4, Pdep), (0b0_0110, false));
        // pext: pack a's bits at mask positions (1,2,4) low.
        assert_eq!(bmi_result(0b1_0110, 0b1_0110, 4, Pext), (0b111, false));
    }
}

/// SDM Vol 1 §4.8.3.5 Table 4-8 (SSE/AVX) and Table 14-17 (FMA): which NaN a float
/// operation delivers, pinned BY VALUE.
///
/// These are exact-bit assertions, not "the result is a NaN". Rust and LLVM leave the
/// payload unspecified — LLVM commutes commutative float ops while the x86 rule is
/// operand-order-dependent, so `a * b` yields one operand's payload in a debug build and
/// the other's in a release build of the same source. A test that only checks `is_nan()`
/// passes against every one of those answers. **Run them in release too**: that is where
/// the divergence reproduces.
///
/// **What these tests can and cannot catch.** On an x86 host with the arithmetic actually
/// executed, Rust's `*` compiles to `mulss`, whose NaN rule *is* the rule below — so
/// removing the fix from `apply_f32`, `apply_un_f32` or `dppd` and re-running here still
/// passes (checked). What they pin is drift on a host whose rule differs (aarch64 prefers
/// an SNaN over the first operand) and drift when the optimizer folds or reassociates
/// instead of emitting the instruction.
/// `float_to_float_conversion_carries_the_nan_payload` is the exception: a Rust `as` cast
/// canonicalizes the payload even on x86, so that one does fail against the unfixed code.
/// The tool that catches the rest is `cargo xfuzz` in release against the native oracle.
#[cfg(test)]
mod nan_rule_tests {
    use super::*;

    const QNAN_A: u32 = 0x7FC0_1234;
    const QNAN_B: u32 = 0x7FC0_5678;
    const SNAN_A: u32 = 0x7F80_1234; // exponent all ones, MSB of the fraction clear
    const ORDINARY: u32 = 0x4048_0000; // 3.125

    fn f(b: u32) -> f32 {
        f32::from_bits(b)
    }

    /// The rule itself, and the test that actually pins it: every other test in this
    /// module composes it, and a composition can agree with an unconstrained compiler by
    /// luck. This one cannot — it asserts both operand orders.
    #[test]
    fn sse_binary_ops_deliver_the_first_nan_source_operand() {
        let mul = |x, y| sse_binop_f32(f(x), f(y), |a, b| a * b).to_bits();
        // "Two QNaNs → first source operand" — and the reverse order gives the other one,
        // which is the whole point: the rule is order-dependent.
        assert_eq!(mul(QNAN_A, QNAN_B), QNAN_A);
        assert_eq!(mul(QNAN_B, QNAN_A), QNAN_B);
        // "QNaN and a floating-point value → QNaN source operand", either side.
        assert_eq!(mul(QNAN_A, ORDINARY), QNAN_A);
        assert_eq!(mul(ORDINARY, QNAN_A), QNAN_A);
        // "SNaN and a floating-point value → SNaN source operand, converted into a QNaN".
        assert_eq!(mul(SNAN_A, ORDINARY), SNAN_A | 0x0040_0000);
        // "SNaN and QNaN → first source operand (quieted if it is an SNaN)".
        assert_eq!(mul(SNAN_A, QNAN_B), SNAN_A | 0x0040_0000);
        assert_eq!(mul(QNAN_B, SNAN_A), QNAN_B);
        // No NaN anywhere: ordinary arithmetic, untouched.
        assert_eq!(
            mul(ORDINARY, ORDINARY),
            (f(ORDINARY) * f(ORDINARY)).to_bits()
        );
    }

    #[test]
    fn dpps_propagates_the_first_nan_in_sdm_tree_order() {
        // imm 0x9D: products 0 and 3 selected, result broadcast to lanes 0, 2, 3.
        // P0 is a NaN and P3 is a different NaN, so the sum `(P0+P1)+(P2+P3)` must
        // deliver P0's — the first source operand of the final add.
        let a = (ORDINARY as u128) | ((ORDINARY as u128) << 96);
        let b = (QNAN_A as u128) | ((QNAN_B as u128) << 96);
        let r = dpps(a, b, 0x9D);
        for lane in [0, 2, 3] {
            assert_eq!(
                (r >> (lane * 32)) as u32,
                QNAN_A,
                "lane {lane} takes the first NaN, not the last"
            );
        }
        assert_eq!((r >> 32) as u32, 0, "a deselected output lane is zeroed");
    }

    #[test]
    fn round_quiets_its_nan_source_without_touching_the_payload() {
        // vround, packed f32, mode 3 (truncate). A one-operand op returns its NaN source
        // operand converted to a QNaN (SDM Vol 1 Table 4-8, the single-operand rows).
        let src = (SNAN_A as u128) | ((QNAN_B as u128) << 32);
        let r = vround(0, src, FPrec::F32, 3, false);
        assert_eq!(r as u32, SNAN_A | 0x0040_0000, "SNaN is quieted in place");
        assert_eq!((r >> 32) as u32, QNAN_B, "a QNaN passes through unchanged");
    }

    #[test]
    fn fma_takes_x_then_y_then_z_and_ignores_the_sign_variant() {
        // SDM Vol 1 Table 14-17: Q(x), else Q(y), else Q(z) — the same for every sign
        // variant, so the neg_prod/neg_add flips must not reach the NaN that is returned.
        let e = |x: u32, y: u32, z: u32, np, na| {
            fma_elem(x as u64, y as u64, z as u64, false, np, na) as u32
        };
        assert_eq!(e(QNAN_A, QNAN_B, QNAN_B, false, false), QNAN_A);
        assert_eq!(e(ORDINARY, QNAN_A, QNAN_B, false, false), QNAN_A);
        assert_eq!(e(ORDINARY, ORDINARY, QNAN_B, false, false), QNAN_B);
        // Negating the product must not flip the sign of the NaN that comes back.
        assert_eq!(e(QNAN_A, ORDINARY, ORDINARY, true, false), QNAN_A);
        assert_eq!(e(ORDINARY, ORDINARY, QNAN_B, false, true), QNAN_B);
        // No NaN source, but an invalid operation: the QNaN indefinite, not any NaN.
        let inf = f32::INFINITY.to_bits();
        assert_eq!(e(inf, 0, ORDINARY, false, false), F32_INDEFINITE);
    }

    /// The most reachable instance of the rule: every SSE/AVX add, subtract, multiply and
    /// divide goes through `apply_f32`/`apply_f64`.
    #[test]
    fn packed_and_scalar_arithmetic_takes_the_first_nan_source_operand() {
        for op in [
            FloatBinOp::Add,
            FloatBinOp::Sub,
            FloatBinOp::Mul,
            FloatBinOp::Div,
        ] {
            assert_eq!(apply_f32(f(QNAN_A), f(QNAN_B), op).to_bits(), QNAN_A);
            assert_eq!(apply_f32(f(QNAN_B), f(QNAN_A), op).to_bits(), QNAN_B);
            assert_eq!(
                apply_f32(f(SNAN_A), f(ORDINARY), op).to_bits(),
                SNAN_A | 0x0040_0000
            );
            assert_eq!(apply_f32(f(ORDINARY), f(QNAN_B), op).to_bits(), QNAN_B);
        }
        let d = |b: u64| f64::from_bits(b);
        const DQ_A: u64 = 0x7FF8_1234_5678_9ABC;
        const DQ_B: u64 = 0x7FF8_0000_DEAD_BEEF;
        assert_eq!(apply_f64(d(DQ_A), d(DQ_B), FloatBinOp::Mul).to_bits(), DQ_A);
        assert_eq!(apply_f64(d(DQ_B), d(DQ_A), FloatBinOp::Mul).to_bits(), DQ_B);
    }

    /// min/max keep the OPPOSITE rule and must not be routed through the helper: "if only
    /// one value is a NaN ... the second operand ... is written to the result", and an
    /// SNaN there is "forwarded unchanged" (SDM Vol 2 MINPS/MAXPS). A refactor that
    /// swept every arm of `apply_f32` into `sse_binop_f32` would break exactly this.
    #[test]
    fn min_and_max_return_the_second_operand_unquieted() {
        for op in [FloatBinOp::Min, FloatBinOp::Max] {
            assert_eq!(apply_f32(f(QNAN_A), f(ORDINARY), op).to_bits(), ORDINARY);
            assert_eq!(apply_f32(f(ORDINARY), f(SNAN_A), op).to_bits(), SNAN_A);
            assert_eq!(apply_f32(f(QNAN_A), f(SNAN_A), op).to_bits(), SNAN_A);
        }
    }

    /// One-operand ops: the NaN source quieted, and the QNaN indefinite for an invalid
    /// operand that is not itself a NaN (SDM Vol 1 §4.8.3.7).
    #[test]
    fn sqrt_quiets_its_nan_and_gives_the_indefinite_for_a_negative() {
        assert_eq!(
            apply_un_f32(f(SNAN_A), FloatUnOp::Sqrt).to_bits(),
            SNAN_A | 0x0040_0000
        );
        assert_eq!(apply_un_f32(f(QNAN_B), FloatUnOp::Sqrt).to_bits(), QNAN_B);
        assert_eq!(
            apply_un_f32(-1.0f32, FloatUnOp::Sqrt).to_bits(),
            F32_INDEFINITE
        );
        assert_eq!(
            apply_un_f32(-1.0f32, FloatUnOp::Rsqrt).to_bits(),
            F32_INDEFINITE
        );
        assert_eq!(
            apply_un_f64(-1.0f64, FloatUnOp::Sqrt).to_bits(),
            F64_INDEFINITE
        );
        // A valid operand is untouched.
        assert_eq!(apply_un_f32(4.0f32, FloatUnOp::Sqrt), 2.0f32);
    }

    /// `dppd`, the double-precision twin of `dpps`.
    #[test]
    fn dppd_propagates_the_first_nan() {
        const DQ_A: u64 = 0x7FF8_1234_5678_9ABC;
        let ord = 3.5f64.to_bits() as u128;
        let a = ord | (ord << 64);
        let b = (DQ_A as u128) | ((3.5f64.to_bits() as u128) << 64);
        // imm 0x33: bits 5:4 select both products, bits 1:0 write both output qwords.
        let r = dppd(a, b, 0x33);
        assert_eq!(r as u64, DQ_A, "the first NaN wins the sum");
        assert_eq!((r >> 64) as u64, DQ_A);
    }

    /// f32↔f64 conversions carry the payload by 29 bits and quiet an SNaN. The values are
    /// MEASURED on hardware through the native oracle (see `f32_nan_to_f64`), not derived
    /// from what a Rust `as` cast happens to do — which is the whole point, since `as`
    /// guarantees neither the payload nor the SNaN conversion.
    #[test]
    fn float_to_float_conversion_carries_the_nan_payload() {
        assert_eq!(
            f32_nan_to_f64(f(0x7FC0_1234)).to_bits(),
            0x7FF8_0246_8000_0000
        );
        assert_eq!(
            f32_nan_to_f64(f(0x7F80_1234)).to_bits(),
            0x7FF8_0246_8000_0000
        );
        assert_eq!(
            f32_nan_to_f64(f(0xFFC0_1234)).to_bits(),
            0xFFF8_0246_8000_0000
        );
        assert_eq!(
            f32_nan_to_f64(f(0x7F80_0001)).to_bits(),
            0x7FF8_0000_2000_0000
        );
        let d = f64::from_bits;
        assert_eq!(
            f64_nan_to_f32(d(0x7FF8_1234_5678_9ABC)).to_bits(),
            0x7FC0_91A2
        );
        assert_eq!(
            f64_nan_to_f32(d(0x7FF0_1234_5678_9ABC)).to_bits(),
            0x7FC0_91A2
        );
        assert_eq!(
            f64_nan_to_f32(d(0xFFF8_1234_5678_9ABC)).to_bits(),
            0xFFC0_91A2
        );
        // A payload that does not survive the narrowing is lost, not rounded up.
        assert_eq!(
            f64_nan_to_f32(d(0x7FF8_0000_0000_0001)).to_bits(),
            0x7FC0_0000
        );
    }

    #[test]
    fn cvtps2ph_keeps_the_carried_payload_instead_of_forcing_a_bit() {
        // A source NaN whose top-10 carried significand bits are zero converts to the
        // bare half QNaN 0x7E00; forcing bit 0 to "keep it a NaN" corrupts it (0x7E00
        // already is one).
        assert_eq!(f32_to_f16(f(0x7F80_0001), 0), 0x7E00);
        // A payload that does survive the shift is carried, not overwritten.
        assert_eq!(f32_to_f16(f(0x7FC0_2000), 0), 0x7E01);
    }
}

#[cfg(test)]
mod mmio_cleanup_tests {
    use super::*;

    #[test]
    fn no_mmio_block_does_not_clear_retry_fields() {
        let mut vm = crate::Vm::new(crate::VmConfig::flat(0x10000));
        vm.map(0x1000, 0x1000, crate::Prot::RWX, crate::RegionKind::Ram)
            .unwrap();
        // The instruction count varies while the retry-cleanup cost stays zero.
        for instructions in [1, 32] {
            let mut code = vec![0x90; instructions];
            code.push(0xf4);
            vm.write_bytes(0x1000, &code).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, 0x1000);
            crate::state::MMIO_CLEAR_CALLS.with(|calls| calls.set(0));
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            crate::state::MMIO_CLEAR_CALLS.with(|calls| {
                assert_eq!(
                    calls.get(),
                    0,
                    "inactive retry fields were cleared for {instructions} instructions"
                );
            });
        }
    }
}

#[cfg(test)]
mod exit_request_tests {
    use super::*;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct RequestAfterAccess<'a> {
        word: Option<&'a AtomicU64>,
        accesses: Cell<u64>,
        limit: u64,
        value: u64,
    }

    impl RequestAfterAccess<'_> {
        fn access(&self) {
            let count = self.accesses.get() + 1;
            self.accesses.set(count);
            if count == self.limit {
                if let Some(word) = self.word {
                    word.fetch_or(crate::vm::EXIT_REQUESTED, Ordering::Release);
                }
            }
        }
    }

    impl StrMem for RequestAfterAccess<'_> {
        fn sload(&self, _addr: u64, _elem: u8) -> Result<u64, MemTrap> {
            self.access();
            Ok(self.value)
        }

        fn sstore(&self, _addr: u64, _value: u64, _elem: u8) -> Result<(), MemTrap> {
            self.access();
            Ok(())
        }
    }

    #[test]
    fn exit_request_string_iteration_witness() {
        // SDM Vol 2B, REP/REPE/REPZ/REPNE/REPNZ—Repeat String Operation
        // Prefix, Description and Operation: REP pp. 4-563–4-564,
        // REPE/REPZ pp. 4-565–4-566, REPNE/REPNZ pp. 4-567–4-568.
        // Each loop services interrupts before its next string iteration, keeping
        // the instruction pointer, remaining count and next element pointers.
        for op in [
            StrOp::Movs,
            StrOp::Stos,
            StrOp::Lods,
            StrOp::Cmps,
            StrOp::Scas,
        ] {
            let rep = match op {
                StrOp::Cmps => RepKind::Repe,
                StrOp::Scas => RepKind::Repne,
                _ => RepKind::Rep,
            };
            let word = AtomicU64::new(0);
            let mem = RequestAfterAccess {
                word: Some(&word),
                accesses: Cell::new(0),
                limit: if matches!(op, StrOp::Cmps | StrOp::Movs) {
                    6
                } else {
                    3
                },
                value: if matches!(op, StrOp::Scas) { 0 } else { 0x5a },
            };
            let mut cpu = CpuState::new();
            cpu.rip = 0x1000;
            cpu.gpr[0] = 0x5a;
            cpu.gpr[RCX] = 8;
            cpu.gpr[RSI] = 0x2000;
            cpu.gpr[RDI] = 0x3000;
            let mut reference = cpu.clone();
            assert!(matches!(
                string_run_interruptible(&mut cpu, &mem, op, 1, rep, 0x1234, 64, 0, 0, Some(&word),),
                Some(StringStop::Interrupted)
            ));
            assert_eq!(cpu.rip, 0x1234);
            assert_eq!(cpu.gpr[RCX], 5);
            let reads_source = matches!(op, StrOp::Movs | StrOp::Lods | StrOp::Cmps);
            let uses_destination = !matches!(op, StrOp::Lods);
            assert_eq!(cpu.gpr[RSI], 0x2000 + if reads_source { 3 } else { 0 });
            assert_eq!(cpu.gpr[RDI], 0x3000 + if uses_destination { 3 } else { 0 });
            assert_eq!(mem.accesses.get(), mem.limit);
            let reference_mem = RequestAfterAccess { word: None, ..mem };
            reference.gpr[RCX] = 3;
            assert!(string_run_interruptible(
                &mut reference,
                &reference_mem,
                op,
                1,
                rep,
                0x1234,
                64,
                0,
                0,
                None,
            )
            .is_none());
            reference.gpr[RCX] = 5;
            assert_eq!(cpu.gpr, reference.gpr);
            assert_eq!(cpu.flags.to_rflags(), reference.flags.to_rflags());
        }
    }
}

#[cfg(test)]
mod store_preflight_tests {
    use super::preflight_store_pages;
    #[test]
    fn later_region_fault_precedes_all_host_probes() {
        for trap in [crate::MemTrap::Unmapped, crate::MemTrap::NonCanonical] {
            let fault = preflight_store_pages(
                0x1000,
                u64::MAX,
                8,
                16,
                0,
                0,
                (
                    |at, _| if at >= 0x1008 { Err(trap) } else { Ok(()) },
                    |_, _| panic!("region failure must precede host probes"),
                ),
            );
            assert_eq!(fault, Err((0x1008, trap)));
        }
    }

    #[test]
    fn selected_store_probes_each_touched_page_once() {
        let page = crate::memory::host_page_size() as u64;
        for (mask, elem, expected) in [
            (u64::MAX, 8, vec![page - 4, page]),
            (1, 4, vec![]),
            (0x10, 4, vec![]),
            (0, 1, vec![]),
        ] {
            let mut probes = Vec::new();
            preflight_store_pages(
                page - 4,
                mask,
                elem,
                32,
                0,
                0,
                (
                    |_, _| Ok(()),
                    |at, n| {
                        assert_eq!(n, 1);
                        probes.push(at);
                        Ok(page * 2)
                    },
                ),
            )
            .unwrap();
            assert_eq!(probes, expected);
        }
    }
}

#[cfg(test)]
std::thread_local! {
    // Distinguish IRET completion from its lifted Full pre-fence.
    static FENCE_ORDERS: std::cell::Cell<[Option<std::sync::atomic::Ordering>; 4]> = const { std::cell::Cell::new([None; 4]) };
}

#[cfg(test)]
mod fence_tests {
    use super::FENCE_ORDERS;
    use crate::{Exit, Prot, Reg, RegionKind, Vm, VmConfig};
    use std::sync::atomic::Ordering;

    #[test]
    fn iret_completion_fence_follows_successful_stack_reads() {
        use super::{control::exec_iret_real, Access, CpuState, InterpResult};
        use crate::{Memory, MemoryModel};

        fn check<const ROUTED: bool>() {
            // SDM Vol 3A §11.3: IRET must complete its stack reads before
            // following guest instructions execute. Failed reads do not retire it.
            for words in 0..=3 {
                let mut mem = Memory::new(MemoryModel::Flat { size: 0x10000 });
                if words != 0 {
                    mem.map(0x8000, words * 2, Prot::RW, RegionKind::Ram)
                        .unwrap();
                    for (index, value) in
                        [0x2000, 0x0010, 0x0202].into_iter().take(words).enumerate()
                    {
                        mem.write(0x8000 + index as u64 * 2, value, 2).unwrap();
                    }
                }
                let mut cpu = CpuState::default();
                cpu.gpr[4] = 0x8000;
                FENCE_ORDERS.with(|orders| orders.set([None; 4]));
                let result = exec_iret_real(&mut cpu, Access::<ROUTED>::new(&mem), 0x1000);
                if words == 3 {
                    assert!(matches!(result, InterpResult::Continue));
                    assert_eq!(cpu.rip, 0x2000);
                    assert_eq!(cpu.cs, 0x10);
                    assert_eq!(cpu.gpr[4], 0x8006);
                    assert!(cpu.flags.if_);
                } else {
                    assert!(matches!(result, InterpResult::Exit(_)));
                }
                FENCE_ORDERS.with(|orders| {
                    assert_eq!(
                        orders.get(),
                        [
                            None,
                            None,
                            None,
                            if words == 3 {
                                Some(Ordering::SeqCst)
                            } else {
                                None
                            },
                        ]
                    )
                });
            }
        }
        check::<false>();
        check::<true>();
    }

    #[test]
    fn fence_kinds_execute_expected_host_ordering() {
        // SDM Vol 2B §4.3 MFENCE/SFENCE; Vol 2A §3.3 LFENCE.
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(
            0x1000,
            &[0x0f, 0xae, 0xf0, 0x0f, 0xae, 0xe8, 0x0f, 0xae, 0xf8, 0xf4],
        )
        .unwrap();
        for routed in [false, true] {
            vm.mem.set_host_fault_recovery(routed);
            FENCE_ORDERS.with(|c| c.set([None; 4]));
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, 0x1000);
            cpu.set_reg(Reg::Rax, 42);
            assert!(matches!(cpu.run(&vm, Some(1)), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rax), 42);
            FENCE_ORDERS.with(|c| {
                assert_eq!(
                    c.get(),
                    [
                        Some(Ordering::SeqCst),
                        Some(Ordering::Acquire),
                        Some(Ordering::Release),
                        None,
                    ]
                )
            });
        }
    }
}

fn daz_lanes(v: u128, prec: FPrec, mode: u32) -> u128 {
    if mode & 0x40 == 0 {
        return v;
    }
    let bits = u32::from(prec.bytes()) * 8;
    let mut out = 0;
    for i in 0..128 / bits {
        out |= u128::from(softfloat::input(
            (v >> (i * bits)) as u64,
            prec == FPrec::F64,
            mode,
        )) << (i * bits);
    }
    out
}
fn apply_f32_mode<const DEFAULT: bool>(x: f32, y: f32, op: FloatBinOp, mode: u32) -> f32 {
    let mode = if DEFAULT { 0 } else { mode };
    if mode & softfloat::MODE_MASK == 0 {
        apply_f32(x, y, op)
    } else {
        f32::from_bits(
            softfloat::binary(x.to_bits() as u64, y.to_bits() as u64, false, op, mode) as u32,
        )
    }
}
fn apply_f64_mode<const DEFAULT: bool>(x: f64, y: f64, op: FloatBinOp, mode: u32) -> f64 {
    let mode = if DEFAULT { 0 } else { mode };
    if mode & softfloat::MODE_MASK == 0 {
        apply_f64(x, y, op)
    } else {
        f64::from_bits(softfloat::binary(x.to_bits(), y.to_bits(), true, op, mode))
    }
}
fn apply_un_f32_mode<const DEFAULT: bool>(x: f32, op: FloatUnOp, mode: u32) -> f32 {
    let mode = if DEFAULT { 0 } else { mode };
    if mode & softfloat::MODE_MASK == 0 {
        apply_un_f32(x, op)
    } else {
        f32::from_bits(softfloat::unary(x.to_bits() as u64, false, op, mode) as u32)
    }
}
fn apply_un_f64_mode<const DEFAULT: bool>(x: f64, op: FloatUnOp, mode: u32) -> f64 {
    let mode = if DEFAULT { 0 } else { mode };
    if mode & softfloat::MODE_MASK == 0 {
        apply_un_f64(x, op)
    } else {
        f64::from_bits(softfloat::unary(x.to_bits(), true, op, mode))
    }
}

#[cfg(test)]
mod legacy_refusal_tests {
    use super::*;

    #[test]
    fn cached_and_stepped_walks_refuse_before_scratch_or_bookkeeping() {
        struct NoScratch;
        impl ScratchBuffer for NoScratch {
            fn zeroed(&mut self, _: usize) -> &mut [u64] {
                panic!("legacy IR allocated interpreter scratch")
            }
        }
        let mut mem = Memory::new(crate::MemoryModel::Flat { size: 4096 });
        mem.map(0, 4096, crate::Prot::RWX, crate::RegionKind::Ram)
            .unwrap();
        mem.write_bytes(0, &[0xb8, 0x34, 0x12, 0, 0, 0xf4]).unwrap();
        let mut ir = crate::lift::lift_block(
            &mem,
            crate::lift::FetchAddr::flat(0),
            crate::CpuMode::Long64,
        )
        .unwrap();
        ir.legacy_context = Some(
            crate::derive_legacy_context(
                crate::LegacyCpuModel {
                    generation: crate::CpuGeneration::I386,
                    model_semantics_id: 1,
                },
                &CpuState::legacy_real_entry([0; 6]),
            )
            .unwrap(),
        );
        let mut cpu = CpuState::default();
        let before = format!("{cpu:?}");
        for consistency in [
            crate::MemConsistency::Fast,
            crate::MemConsistency::AcqRel,
            crate::MemConsistency::FullTso,
        ] {
            let mut info = RetireInfo {
                retired: 17,
                sti_shadow: true,
                ss_shadow: true,
                consistency,
                ..RetireInfo::default()
            };
            let bookkeeping = format!("{info:?}");
            let mut steps = StepContext::new(1);
            for result in [
                interpret_block_fp::<false, true>(&ir, &mut cpu, &mem, &mut NoScratch, &mut info),
                interpret_block_fp::<true, false>(&ir, &mut cpu, &mem, &mut NoScratch, &mut info),
                interpret_block_steps::<false, true, false>(
                    &ir,
                    &mut cpu,
                    &mem,
                    &mut NoScratch,
                    &mut info,
                    &mut steps,
                ),
                interpret_block_steps::<true, false, true>(
                    &ir,
                    &mut cpu,
                    &mem,
                    &mut NoScratch,
                    &mut info,
                    &mut steps,
                ),
            ] {
                assert!(matches!(
                    result.into_step(),
                    StepResult::Exit(Exit::UnsupportedSystemContext {
                        operation: crate::UnsupportedSystemOperation::LegacyExecution,
                        ..
                    })
                ));
            }
            assert_eq!(steps.remaining, 1);
            assert_eq!(format!("{info:?}"), bookkeeping);
            assert_eq!(format!("{cpu:?}"), before);
        }
    }
}
