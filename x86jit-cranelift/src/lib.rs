//! Cranelift JIT backend for x86jit (§8.2).
//!
//! Compiles an [`x86jit_core::IrBlock`] to host code. Guest RAM access is inlined
//! (`host_base + guest_addr` after a bounds check); only out-of-range access and
//! syscalls trap out. The compiled-block ABI (signature, result encoding, field
//! offsets) is defined once in `x86jit_core::jit_abi` and shared with the
//! dispatcher; this crate only emits code matching it.
//!
//! Build order (§8.2.3): offsets + ABI + a "returns Continue" block first, then
//! `IrOp`s one at a time, each validated against the interpreter oracle.
//!
//! Optional compile-lock timing is available through
//! [`JitBackend::set_lock_telemetry`], [`JitBackend::lock_telemetry`], and
//! [`JitBackend::reset_lock_telemetry`]. It reports global compilation and module
//! acquisitions, hold/wait totals and maxima in nanoseconds, and fixed log2
//! histograms. Disabled by default, it reads one relaxed switch per acquisition
//! without reading the clock. Hold time includes lazy module construction.
//! Snapshots include released guards only; observers run after their release.
//! [`LockTelemetrySnapshot`] documents lock order and what each entry measures.
//! These are backend diagnostics, not guest state.

#![cfg(feature = "jit")]

#[cfg(test)]
extern crate self as x86jit_cranelift;
#[cfg(all(test, target_arch = "aarch64", target_os = "macos"))]
#[path = "../tests/application_codegen.rs"]
mod application_codegen;
#[cfg(all(test, target_arch = "aarch64", target_os = "macos"))]
application_codegen::relocated_codegen_witnesses!();

#[cfg(test)]
mod batch_tests;
#[cfg(feature = "jit")]
mod code_memory;
mod codegen;
#[cfg(test)]
mod embedder_tests;
#[cfg(feature = "environment")]
mod environment;
mod libcalls;
mod lock_telemetry;
mod memory_stats;
pub mod perfmap;
mod reclamation;
mod unwind;

pub use reclamation::{CodeLifecycleEvent, CodeLifecycleKind};

pub use code_memory::{CodeMemoryFailure, CodeMemoryFailureOp, CodeMemoryPolicy};

use lock_telemetry::{LockKind, Telemetry};
pub use lock_telemetry::{LockTelemetry, LockTelemetrySnapshot};

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::thread::JoinHandle;
use std::time::Instant;

use cranelift::codegen::ir;
use cranelift::prelude::*;
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Linkage, Module, ModuleError};

use x86jit_core::cache::CompiledPtr;
use x86jit_core::jit_abi::{cpu_offsets, CpuOffsets, MAX_GUEST_ACCESS_DESCRIPTORS};
use x86jit_core::x87::{FpMem, RawFpMem};
use x86jit_core::GuestTrapSite;
use x86jit_core::{
    Backend, CachedBlock, IrBlock, IrRegion, MemConsistency, RegionCaps, ShutdownError,
    ShutdownReport, TierUpFinished, TierUpRequest, TierUpSubmit, TierUpUnit,
};

// User trap codes identify transfers independently of machine source locations:
// egraph-created extending loads can have a default location, and x64 can fold
// a load into an ALU instruction carrying the consumer's location. Cranelift
// preserves the load's trap code in both cases (0.115.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GuestTrap {
    guest_rip: u64,
    access: x86jit_core::GuestAccessKind,
    width: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompileError {
    DescriptorLimit,
    SharedAddressSpaceChild,
    Unavailable,
    BackendError,
    /// A caught panic: the payload does not say whether the unit caused it.
    Panicked,
}

impl CompileError {
    fn reason(self) -> x86jit_core::CompileFailureReason {
        match self {
            Self::DescriptorLimit => x86jit_core::CompileFailureReason::DescriptorLimit,
            Self::SharedAddressSpaceChild => x86jit_core::CompileFailureReason::OtherBackendError,
            Self::Unavailable => x86jit_core::CompileFailureReason::CodeSpace,
            Self::BackendError => x86jit_core::CompileFailureReason::OtherBackendError,
            Self::Panicked => x86jit_core::CompileFailureReason::Unclassified,
        }
    }
}

fn check_compile_allowed() -> Result<(), CompileError> {
    x86jit_core::codemap::check_jit_allowed().map_err(|_| CompileError::SharedAddressSpaceChild)
}

fn prepare_guest_traps(
    func: &mut ir::Function,
    guest_rips: &[u64],
) -> Result<Vec<GuestTrap>, CompileError> {
    // The shared lifter caps blocks and regions conservatively. Retain a defensive
    // refusal for oversized synthetic IR or a future lowering that breaks the bound.
    let mut metadata = Vec::new();
    metadata
        .try_reserve_exact(MAX_GUEST_ACCESS_DESCRIPTORS)
        .map_err(|_| CompileError::Unavailable)?;
    let mut block = func.layout.entry_block();
    while let Some(current_block) = block {
        block = func.layout.next_block(current_block);
        let mut next_inst = func.layout.first_inst(current_block);
        while let Some(inst) = next_inst {
            next_inst = func.layout.next_inst(inst);
            let opcode = func.dfg.insts[inst].opcode();
            if !opcode.can_load() && !opcode.can_store() {
                continue;
            }
            let Some(flags) = func.dfg.insts[inst].memflags() else {
                continue;
            };
            let flags_data = func.dfg.mem_flags[flags];
            let Some(code) = flags_data.trap_code() else {
                continue;
            };
            let Some((access, width)) = codegen::guest_trap_access(code) else {
                return Err(CompileError::BackendError);
            };
            let source = func.srcloc(inst);
            let guest_rip = *guest_rips
                .get(source.bits() as usize)
                .filter(|_| !source.is_default())
                .ok_or(CompileError::BackendError)?;
            let descriptor = GuestTrap {
                guest_rip,
                access,
                width,
            };
            let index = match metadata.iter().position(|m| *m == descriptor) {
                Some(index) => index,
                None if metadata.len() < MAX_GUEST_ACCESS_DESCRIPTORS => {
                    metadata.push(descriptor);
                    metadata.len() - 1
                }
                None => return Err(CompileError::DescriptorLimit),
            };
            let owner_flags = func
                .dfg
                .mem_flags
                .insert(flags_data.with_trap_code(ir::TrapCode::user(index as u8 + 1)))
                .map_err(|_| CompileError::Unavailable)?;
            use ir::InstructionData::*;
            let flags = match &mut func.dfg.insts[inst] {
                Load { flags, .. }
                | LoadNoOffset { flags, .. }
                | Store { flags, .. }
                | StoreNoOffset { flags, .. }
                | AtomicCas { flags, .. }
                | AtomicRmw { flags, .. } => flags,
                _ => unreachable!("memflags only exists on memory instructions"),
            };
            *flags = owner_flags;
        }
    }
    Ok(metadata)
}

fn collect_guest_traps(
    code: &cranelift::codegen::CompiledCode,
    metadata: &[GuestTrap],
) -> Result<Vec<GuestTrapSite>, CompileError> {
    // Validate before reserving: no unclassified traps may be silently dropped.
    // Reserve exactly the records retained by the code map.
    for trap in code.buffer.traps() {
        let index = trap.code.as_raw().get() as usize - 1;
        if metadata.get(index).is_none() {
            return Err(CompileError::BackendError);
        }
    }
    let mut sites = Vec::new();
    sites
        .try_reserve_exact(code.buffer.traps().len())
        .map_err(|_| CompileError::Unavailable)?;
    for trap in code.buffer.traps() {
        let descriptor = metadata[trap.code.as_raw().get() as usize - 1];
        sites.push(GuestTrapSite {
            host_offset: trap.offset,
            guest_rip: descriptor.guest_rip,
            access: descriptor.access,
            width: descriptor.width,
        });
    }
    sites.sort_unstable_by_key(|s| s.host_offset);
    Ok(sites)
}

// Keep the disabled path to one flag branch, without constructing or dropping
// a sample guard. The timed invocation runs the same helper body with its flag off.
macro_rules! account_helper {
    ($enabled:expr, $call:expr) => {
        if $enabled != 0 {
            let _time_helper = x86jit_core::time_stats::HelperTimer::enter();
            return $call;
        }
    };
}

/// Division helper called from compiled code (div isn't hot, so a call is fine and
/// avoids 128-bit codegen). Reuses the interpreter's `divide` so both agree.
/// `out` points at `[quot, rem]`; returns 0 on success, 1 on `#DE`.
///
/// # Safety
/// `out` must point at two writable `u64`s. Called only from JIT code with a valid
/// stack-slot pointer.
unsafe extern "C" fn div_helper(
    hi: u64,
    lo: u64,
    divisor: u64,
    size: u64,
    signed: u64,
    out: *mut u64,
    time_enabled: u64,
) -> u64 {
    account_helper!(
        time_enabled,
        div_helper(hi, lo, divisor, size, signed, out, 0)
    );
    match x86jit_core::interp::divide(hi, lo, divisor, size as u8, signed != 0) {
        Some((q, r)) => {
            *out = q;
            *out.add(1) = r;
            0
        }
        None => 1,
    }
}

/// String-op helper: runs the (rep) loop via shared `string_run_interruptible`.
/// Reads `cpu` and the guest buffer (`MemCtx.base`/`size`); a trap writes fault
/// details and returns `RET_UNMAPPED`, a host request returns `RET_INTERRUPTED`,
/// and completion returns `RET_CONTINUE`.
///
/// # Safety
/// `cpu`/`mem` are valid pointers to a `CpuState` / `MemCtx` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn string_helper<const CHECK_TRAPS: bool>(
    cpu: *mut u8,
    mem: *mut u8,
    op: u64,
    elem: u64,
    rep: u64,
    cur_addr: u64,
    addr_bits: u64,
    seg_base: u64,
    time_enabled: u64,
) -> u64 {
    account_helper!(
        time_enabled,
        string_helper::<CHECK_TRAPS>(cpu, mem, op, elem, rep, cur_addr, addr_bits, seg_base, 0)
    );
    use x86jit_core::jit_abi::{
        MemCtx, RET_CONTINUE, RET_INTERRUPTED, RET_MMIO_DEFER, RET_UNMAPPED_DETAIL,
    };
    use x86jit_core::{RepKind, StrOp};

    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let ctx = &mut *(mem as *mut MemCtx);
    let op = [
        StrOp::Movs,
        StrOp::Stos,
        StrOp::Scas,
        StrOp::Cmps,
        StrOp::Lods,
    ][op as usize];
    let rep = [RepKind::None, RepKind::Rep, RepKind::Repe, RepKind::Repne][rep as usize];

    // SAFETY: mem_self retains the live Memory for this helper call.
    let memory = &*(ctx.mem_self as *const x86jit_core::Memory);
    if CHECK_TRAPS
        && x86jit_core::interp::string_touches_trap(
            cpu,
            memory,
            op,
            elem as u8,
            rep,
            addr_bits as u8,
            seg_base,
            0,
        )
    {
        cpu.rip = cur_addr;
        return RET_MMIO_DEFER;
    }

    let raw = x86jit_core::interp::RawStrMem {
        recover_host_faults: memory.host_fault_recovery(),
        base: ctx.base as *mut u8,
        size: ctx.size,
        guest_base: ctx.guest_base,
    };
    let raw = NotifyingStrMem { raw, memory };
    match x86jit_core::interp::string_run_interruptible(
        cpu,
        &raw,
        op,
        elem as u8,
        rep,
        cur_addr,
        addr_bits as u8,
        seg_base,
        0,
        // SAFETY: points to the live vcpu word or the static zero atomic.
        (ctx.dispatch_word_ptr as *const std::sync::atomic::AtomicU64).as_ref(),
    ) {
        None => RET_CONTINUE,
        Some(x86jit_core::interp::StringStop::Interrupted) => RET_INTERRUPTED,
        Some(x86jit_core::interp::StringStop::Fault(f)) => {
            ctx.record_helper_fault(
                f.addr,
                u64::from(f.elem),
                f.write,
                f.addr,
                u64::from(f.elem),
            );
            ctx.record_helper_trap(f.trap);
            ctx.next_entry = cur_addr;
            ctx.link_slot = f.addr;
            RET_UNMAPPED_DETAIL
        }
    }
}

/// Paired CAS helper; guest-memory accesses are checked by `Memory` before asm.
/// Opted-in transfers recover through the core fixup table and return normally.
///
/// # Safety
/// `cpu` and `mem` point to the live CpuState and MemCtx for this call.
unsafe extern "C" fn cas_pair_helper(
    cpu: *mut u8,
    mem: *mut u8,
    addr: u64,
    size: u64,
    cur: u64,
    len: u64,
    time_enabled: u64,
) -> u64 {
    account_helper!(
        time_enabled,
        cas_pair_helper(cpu, mem, addr, size, cur, len, 0)
    );
    use x86jit_core::jit_abi::{MemCtx, RET_CONTINUE, RET_EXCEPTION, RET_MMIO_DEFER, RET_UNMAPPED};
    // SAFETY: the compiled-block ABI supplies these live, per-vcpu pointers.
    let (cpu, ctx) = unsafe {
        (
            &mut *(cpu as *mut x86jit_core::state::CpuState),
            &mut *(mem as *mut MemCtx),
        )
    };
    // SAFETY: for_memory retains the Memory for the whole compiled call.
    let memory = unsafe { &*(ctx.mem_self as *const x86jit_core::Memory) };
    match x86jit_core::interp::cmpxchg_pair_run(cpu, memory, cur, addr, size as u8, len as u8) {
        None => RET_CONTINUE,
        Some(x86jit_core::Exit::Exception { vector, .. }) => {
            ctx.exception_vector = vector as u64;
            RET_EXCEPTION
        }
        Some(exit @ x86jit_core::Exit::UnmappedMemory { addr: at, .. }) => {
            ctx.record_helper_fault(at, size, true, addr, size);
            if let Some(detail) = exit.fault_metadata() {
                ctx.record_helper_trap(match detail.origin() {
                    x86jit_core::FaultOrigin::HostAccess => x86jit_core::MemTrap::HostFault,
                    x86jit_core::FaultOrigin::GuestMap => x86jit_core::MemTrap::Unmapped,
                    _ => x86jit_core::MemTrap::NonCanonical,
                });
            }
            RET_UNMAPPED
        }
        // Re-execute in the interpreter so the dispatcher returns the precise
        // unsupported-operation exit without requesting a device transfer.
        Some(x86jit_core::Exit::UnknownInstruction { .. }) => RET_MMIO_DEFER,
        Some(_) => unreachable!("paired CAS only raises memory/alignment faults or rejects MMIO"),
    }
}

struct NotifyingStrMem<'a> {
    raw: x86jit_core::interp::RawStrMem,
    memory: &'a x86jit_core::Memory,
}

impl x86jit_core::interp::StrMem for NotifyingStrMem<'_> {
    fn preflight_store(
        &self,
        base: u64,
        mask: u64,
        elem: u8,
        bytes: u16,
    ) -> Result<(), (u64, x86jit_core::MemTrap)> {
        self.raw.preflight_store(base, mask, elem, bytes)
    }

    fn sload(&self, addr: u64, elem: u8) -> Result<u64, x86jit_core::MemTrap> {
        self.raw.sload(addr, elem)
    }

    fn sstore_preflighted(
        &self,
        addr: u64,
        value: u64,
        elem: u8,
    ) -> Result<(), x86jit_core::MemTrap> {
        let result = self.raw.store_preflighted_with_progress(addr, value, elem);
        let committed = result
            .as_ref()
            .map_or_else(|&(_, committed)| committed, |()| elem as usize);
        if committed != 0 {
            self.memory.note_write(addr, committed);
        }
        result.map_err(|(trap, _)| trap)
    }

    fn sstore(&self, addr: u64, value: u64, elem: u8) -> Result<(), x86jit_core::MemTrap> {
        let result = self.raw.store_with_progress(addr, value, elem);
        let committed = result
            .as_ref()
            .map_or_else(|&(_, committed)| committed, |()| elem as usize);
        if committed != 0 {
            self.memory.note_write(addr, committed);
        }
        result.map_err(|(trap, _)| trap)
    }
}

struct NotifyingFpMem<'a> {
    raw: RawFpMem,
    memory: &'a x86jit_core::Memory,
}

impl FpMem for NotifyingFpMem<'_> {
    fn load(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.raw.load(addr, buf)
    }

    fn store(&self, addr: u64, bytes: &[u8]) -> bool {
        self.store_checked(addr, bytes).is_ok()
    }

    fn load_checked(&self, addr: u64, buf: &mut [u8]) -> Result<(), u64> {
        self.raw.load_checked(addr, buf)
    }

    fn store_checked(&self, addr: u64, bytes: &[u8]) -> Result<(), u64> {
        let result = self.raw.store_checked_with_progress(addr, bytes);
        let committed = result
            .as_ref()
            .map_or_else(|&(_, committed)| committed, |()| bytes.len());
        if committed != 0 {
            self.memory.note_write(addr, committed);
        }
        result.map_err(|(at, _)| at)
    }

    fn fault_addr(&self, addr: u64) -> u64 {
        self.raw.fault_addr(addr)
    }

    fn host_fault_recovery(&self) -> bool {
        self.raw.host_fault_recovery()
    }
}

// A successful x87 memory helper can suppress FDP recording. Codegen consumes
// this flag locally; it never reaches the compiled-block return-code ABI.
const X87_SKIP_FDP: u64 = 1 << 63;

/// x87 helper: runs one FPU op via the shared `exec_x87`. On a memory fault it
/// writes the fault into `MemCtx`, sets RIP to the faulting instruction, and
/// returns `RET_UNMAPPED`. Success returns `RET_CONTINUE` or the private
/// `X87_SKIP_FDP` flag, which codegen consumes before continuing.
///
/// # Safety
/// `cpu`/`mem` are valid pointers to a `CpuState` / `MemCtx` for the call; `kind`
/// is a valid FpuKind. `CHECK_FDP` is selected only for non-control memory ops.
unsafe extern "C" fn x87_helper<const CHECK_FDP: bool>(
    cpu: *mut u8,
    mem: *mut u8,
    kind: u64,
    addr: u64,
    sti: u64,
    cur_addr: u64,
    time_enabled: u64,
) -> u64 {
    account_helper!(
        time_enabled,
        x87_helper::<CHECK_FDP>(cpu, mem, kind, addr, sti, cur_addr, 0)
    );
    use x86jit_core::jit_abi::{MemCtx, RET_CONTINUE, RET_UNMAPPED};
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let ctx = &mut *(mem as *mut MemCtx);
    // Safe: `kind` came from a real FpuKind baked by the lift.
    let kind: x86jit_core::x87::FpuKind = std::mem::transmute(kind as u16);
    // The deferred #MF report (SDM Vol 1 §8.6), same rule as the interpreter's: an
    // unmasked exception raised earlier traps HERE, before this op runs. `RET_EXCEPTION`
    // and `MemCtx.exception_vector` already exist, so this needs no ABI growth.
    if x86jit_core::x87::mf_pending_before(cpu, kind) {
        ctx.exception_vector = 16;
        cpu.rip = cur_addr;
        return x86jit_core::jit_abi::RET_EXCEPTION;
    }
    // SAFETY: mem_self retains the live Memory for this helper call.
    let memory = unsafe { &*(ctx.mem_self as *const x86jit_core::Memory) };
    let mem = NotifyingFpMem {
        raw: RawFpMem {
            recover_host_faults: memory.host_fault_recovery(),
            base: ctx.base as *mut u8,
            size: ctx.size,
            guest_base: ctx.guest_base,
        },
        memory,
    };
    match x86jit_core::x87::exec_x87(cpu, &mem, kind, addr, sti as u8) {
        None => {
            // Codegen consumes this private success flag before continuing.
            // Keep all pointer stores in the generated success path; this helper
            // only evaluates the live per-vCPU CPUID policy for memory forms.
            if CHECK_FDP && !x86jit_core::x87::updates_data_pointer(cpu) {
                X87_SKIP_FDP
            } else {
                RET_CONTINUE
            }
        }
        Some((fault, write)) => {
            let width = x86jit_core::x87::mem_operand_bytes(kind);
            ctx.record_helper_fault(fault, width, write, addr, width);
            ctx.record_fp_failure();
            cpu.rip = cur_addr;
            RET_UNMAPPED
        }
    }
}

/// fxsave/fxrstor helper: runs the 512-byte save/restore via the shared
/// `exec_fxstate`. On a memory fault it sets RIP + fault fields and returns
/// `RET_UNMAPPED`; reserved MXCSR bits return `RET_EXCEPTION` (#GP).
///
/// # Safety
/// `cpu`/`mem` are valid for the call; `mem` is a `*mut MemCtx`.
unsafe extern "C" fn fxstate_helper(
    cpu: *mut u8,
    mem: *mut u8,
    addr: u64,
    restore: u64,
    cur_addr: u64,
    time_enabled: u64,
) -> u64 {
    account_helper!(
        time_enabled,
        fxstate_helper(cpu, mem, addr, restore, cur_addr, 0)
    );
    use x86jit_core::jit_abi::{MemCtx, RET_CONTINUE, RET_EXCEPTION, RET_UNMAPPED};
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let ctx = &mut *(mem as *mut MemCtx);
    let raw = x86jit_core::x87::RawFpMem {
        // SAFETY: mem_self retains the live Memory for this helper call.
        recover_host_faults: unsafe { &*(ctx.mem_self as *const x86jit_core::Memory) }
            .host_fault_recovery(),
        base: ctx.base as *mut u8,
        size: ctx.size,
        guest_base: ctx.guest_base,
    };
    let raw = NotifyingFpMem {
        raw,
        // SAFETY: mem_self retains the live Memory for this helper call.
        memory: unsafe { &*(ctx.mem_self as *const x86jit_core::Memory) },
    };
    match x86jit_core::x87::exec_fxstate_format(cpu, &raw, addr, restore & 1 != 0, restore & 2 != 0)
    {
        None => RET_CONTINUE,
        Some(x86jit_core::x87::FpStateFault::Memory { addr: fault, write }) => {
            ctx.record_helper_fault(fault, 512, write, addr, 512);
            ctx.record_fp_failure();
            cpu.rip = cur_addr;
            RET_UNMAPPED
        }
        Some(x86jit_core::x87::FpStateFault::GeneralProtection) => {
            cpu.rip = cur_addr;
            ctx.exception_vector = 13;
            RET_EXCEPTION
        }
    }
}

/// Extended-state helper; shared checked transfers retain precise fault addresses.
///
/// # Safety
/// `cpu` and `mem` are the live compiled-block ABI arguments.
unsafe extern "C" fn xstate_helper(
    cpu: *mut u8,
    mem: *mut u8,
    addr: u64,
    kind: u64,
    format: u64,
    cur_addr: u64,
    time_enabled: u64,
) -> u64 {
    account_helper!(
        time_enabled,
        xstate_helper(cpu, mem, addr, kind, format, cur_addr, 0)
    );
    use x86jit_core::jit_abi::{MemCtx, RET_CONTINUE, RET_EXCEPTION_DETAIL, RET_UNMAPPED};
    use x86jit_core::x87::xstate::{exec_xstate, XStateFault, XStateKind};
    // SAFETY: the compiled-block ABI supplies exclusive CPU and live memory context.
    let cpu = unsafe { &mut *(cpu as *mut x86jit_core::CpuState) };
    let ctx = unsafe { &mut *(mem as *mut MemCtx) };
    let memory = unsafe { &*(ctx.mem_self as *const x86jit_core::Memory) };
    let kind = match kind {
        0 => XStateKind::Save,
        1 => XStateKind::Restore,
        2 => XStateKind::SaveOpt,
        _ => XStateKind::SaveCompact,
    };
    match exec_xstate(
        cpu,
        memory,
        addr,
        kind,
        format & 1 != 0,
        x86jit_core::ir::MemoryAccess::new(1, format & 2 != 0, format & 4 != 0),
    ) {
        Ok(()) => RET_CONTINUE,
        Err(XStateFault::Exception { vector, cause }) => {
            cpu.rip = cur_addr;
            ctx.exception_vector = vector as u64;
            ctx.next_entry = cur_addr;
            ctx.fault_access = x86jit_core::fault::cause_to_wire(cause);
            RET_EXCEPTION_DETAIL
        }
        Err(XStateFault::Memory { addr: fault, write }) => {
            ctx.record_helper_fault(fault, 1, write, addr, 1);
            ctx.record_fp_failure();
            cpu.rip = cur_addr;
            RET_UNMAPPED
        }
    }
}

/// `cpuid` helper: delegates to the shared `cpuid_run` so both backends report the
/// same features.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
unsafe extern "C" fn cpuid_helper(cpu: *mut u8, time_enabled: u64) {
    account_helper!(time_enabled, cpuid_helper(cpu, 0));
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::cpuid_run(cpu);
}

/// # Safety
/// `cpu` is a valid exclusive pointer to a `CpuState` for this call.
unsafe extern "C" fn tsc_helper(cpu: *mut u8, aux: u64, time_enabled: u64) {
    account_helper!(time_enabled, tsc_helper(cpu, aux, 0));
    // SAFETY: the compiled-block ABI supplies the live vCPU exclusively.
    let cpu = unsafe { &mut *(cpu as *mut x86jit_core::CpuState) };
    x86jit_core::interp::tsc_run(cpu, aux != 0);
}

/// # Safety
/// `cpu` is a valid exclusive pointer to a `CpuState` for this call.
unsafe extern "C" fn random_helper(
    cpu: *mut u8,
    seed: u64,
    size: u64,
    flags: u64,
    time_enabled: u64,
) -> u64 {
    account_helper!(time_enabled, random_helper(cpu, seed, size, flags, 0));
    // SAFETY: the compiled-block ABI supplies the live vCPU exclusively.
    let cpu = unsafe { &mut *(cpu as *mut x86jit_core::CpuState) };
    let kind = if seed != 0 {
        x86jit_core::RandomKind::Rdseed
    } else {
        x86jit_core::RandomKind::Rdrand
    };
    x86jit_core::interp::random_run(cpu, kind, size as u8, x86jit_core::FlagMask(flags as u8))
}

/// `xgetbv` helper: delegates to the shared `xgetbv_run` so XCR0 tracks the guest
/// feature set identically on both backends.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
unsafe extern "C" fn xgetbv_helper(
    cpu: *mut u8,
    mem: *mut u8,
    cur_addr: u64,
    time_enabled: u64,
) -> u64 {
    account_helper!(time_enabled, xgetbv_helper(cpu, mem, cur_addr, 0));
    use x86jit_core::jit_abi::{MemCtx, RET_CONTINUE, RET_EXCEPTION_DETAIL};
    // SAFETY: the compiled-block ABI supplies exclusive CPU and live memory context.
    let cpu = unsafe { &mut *(cpu as *mut x86jit_core::CpuState) };
    if let Some(x86jit_core::x87::xstate::XStateFault::Exception { vector, cause }) =
        x86jit_core::x87::xstate::xgetbv_fault(cpu)
    {
        let ctx = unsafe { &mut *(mem as *mut MemCtx) };
        cpu.rip = cur_addr;
        ctx.exception_vector = vector as u64;
        ctx.next_entry = cur_addr;
        ctx.fault_access = x86jit_core::fault::cause_to_wire(cause);
        return RET_EXCEPTION_DETAIL;
    }
    x86jit_core::interp::xgetbv_run(cpu);
    RET_CONTINUE
}

/// Record a JIT-inlined guest store against BOTH page facilities: the SMC code-page
/// table and the embedder's watched data ranges.
///
/// Generated store code calls this only after its inline tests say the store *might*
/// have hit a watched page or a page backing translated code — a live `watch_count`
/// load, a bit probe of the watch bitmap, and the code-range watermark.
/// `Memory::note_write` then decides precisely.
///
/// # Safety
/// `mem_self` is the live `*const Memory` for this run (set by `MemCtx::for_memory`).
unsafe extern "C" fn note_write_helper(mem_self: u64, addr: u64, len: u64, time_enabled: u64) {
    account_helper!(time_enabled, note_write_helper(mem_self, addr, len, 0));
    let mem = &*(mem_self as *const x86jit_core::memory::Memory);
    mem.note_write(addr, len as usize);
}

/// `crc32` helper: CRC-32C folding via the shared `crc32c` so both backends agree.
extern "C" fn crc32_helper(crc: u64, src: u64, bytes: u64, time_enabled: u64) -> u64 {
    account_helper!(time_enabled, crc32_helper(crc, src, bytes, 0));
    x86jit_core::interp::crc32c(crc as u32, src, bytes as u8) as u64
}

/// BMI1/BMI2 helper: runs the shared `bmi_result` so the JIT matches
/// the interpreter exactly (the bextr/bzhi variable shift+mask is fiddly to emit
/// natively). Writes `out[0] = result`, `out[1] = CF`; ZF/SF are derived at the call
/// site. `op` is the `BmiOp` discriminant.
///
/// # Safety
/// `out` points to two writable `u64`s for the call.
unsafe extern "C" fn bmi_helper(
    a: u64,
    b: u64,
    op: u64,
    size: u64,
    out: *mut u64,
    time_enabled: u64,
) {
    account_helper!(time_enabled, bmi_helper(a, b, op, size, out, 0));
    use x86jit_core::ir::BmiOp::*;
    let bmiop = match op {
        0 => Andn,
        1 => Blsi,
        2 => Blsr,
        3 => Blsmsk,
        4 => Bextr,
        5 => Bzhi,
        6 => Pdep,
        _ => Pext,
    };
    let (r, cf) = x86jit_core::interp::bmi_result(a, b, size as u8, bmiop);
    *out = r;
    *out.add(1) = cf as u64;
}

/// EVEX masked move helper: runs the shared `CpuState::write_masked`, so
/// the JIT's masking is bit-identical to the interpreter's (masked ops aren't hot, so
/// a helper call beats hand-emitting a per-lane blend). Args are widened to u64.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vmaskmov_helper(
    cpu: *mut u8,
    dst: u64,
    src: u64,
    k: u64,
    elem: u64,
    zeroing: u64,
    bytes: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        vmaskmov_helper(cpu, dst, src, k, elem, zeroing, bytes, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let newval = cpu.vec_lanes(src as usize);
    cpu.write_masked(
        dst as usize,
        newval,
        k as u8,
        elem as u8,
        zeroing != 0,
        bytes as u16,
    );
}

/// Decode a helper's cold operand carrier. The low word remains the transfer
/// width; xj-au4.2 tracks embedded broadcast's transfer/value semantics.
/// SDM Vol 2A §2.7.7: the architectural carrier is the one decoded element.
#[cold]
fn helper_operand_span(base: u64, packed_bytes: u64) -> (u64, u64) {
    let carrier = packed_bytes >> 16;
    (base, if carrier != 0 { carrier } else { packed_bytes })
}

/// EVEX write-masked vector **memory** move `vmovdqu{8,16,32,64} v{k}{z}, [mem]` (load)
/// and `[mem]{k}, v` (store). Element-wise via the shared
/// `masked_load_run`/`masked_store_run` so masked-off lanes never fault (hardware
/// suppression) and JIT == interpreter. On an active-lane fault, writes the fault into
/// `MemCtx` (RIP already set by the shared fn) and returns `RET_UNMAPPED`; else
/// `RET_CONTINUE`. Uses the bounds-only `RawStrMem` view, like `string_helper`.
///
/// # Safety
/// `cpu`/`mem` are valid pointers to a `CpuState` / `MemCtx` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vmaskmov_mem_helper(
    cpu: *mut u8,
    mem: *mut u8,
    reg: u64,
    addr: u64,
    k: u64,
    elem: u64,
    zeroing: u64,
    bytes: u64,
    is_store: u64,
    cur_addr: u64,
    broadcast_elem: u64,
    time_enabled: u64,
) -> u64 {
    account_helper!(
        time_enabled,
        vmaskmov_mem_helper(
            cpu,
            mem,
            reg,
            addr,
            k,
            elem,
            zeroing,
            bytes,
            is_store,
            cur_addr,
            broadcast_elem,
            0
        )
    );
    use x86jit_core::jit_abi::{MemCtx, RET_CONTINUE, RET_UNMAPPED};
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let ctx = &mut *(mem as *mut MemCtx);
    let raw = x86jit_core::interp::RawStrMem {
        // SAFETY: mem_self retains the live Memory for this helper call.
        recover_host_faults: unsafe { &*(ctx.mem_self as *const x86jit_core::Memory) }
            .host_fault_recovery(),
        base: ctx.base as *mut u8,
        size: ctx.size,
        guest_base: ctx.guest_base,
    };
    let raw = NotifyingStrMem {
        raw,
        // SAFETY: mem_self retains the live Memory for this helper call.
        memory: unsafe { &*(ctx.mem_self as *const x86jit_core::Memory) },
    };
    let km = if k == 0 {
        u64::MAX
    } else {
        cpu.kmask[k as usize]
    };
    let fault = if is_store == 1 {
        x86jit_core::interp::masked_store_run(
            cpu,
            &raw,
            reg as u8,
            addr,
            km,
            elem as u8,
            bytes as u16,
            cur_addr,
        )
    } else if broadcast_elem != 0 {
        x86jit_core::interp::masked_broadcast_load_run(
            cpu,
            &raw,
            reg as u8,
            addr,
            km,
            elem as u8,
            bytes as u16,
            broadcast_elem as u8,
            cur_addr,
        )
    } else {
        x86jit_core::interp::masked_load_run(
            cpu,
            &raw,
            reg as u8,
            addr,
            km,
            elem as u8,
            zeroing != 0,
            bytes as u16,
            cur_addr,
        )
    };
    match fault {
        None => RET_CONTINUE,
        Some(f) => {
            let (operand, width) = if broadcast_elem != 0 {
                (addr, broadcast_elem)
            } else {
                x86jit_core::memory::active_operand_span(addr, bytes, elem as u8, km)
            };
            ctx.record_helper_fault(f.addr, u64::from(f.elem), f.write, operand, width);
            ctx.record_helper_trap(f.trap);
            RET_UNMAPPED
        }
    }
}

/// AVX2 VSIB gathers share their restart state machine with the interpreter.
///
/// # Safety
/// `cpu` and `mem` point to live CpuState and MemCtx values for this call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn gather_helper(
    cpu: *mut u8,
    mem: *mut u8,
    dst: u64,
    mask: u64,
    index: u64,
    base: u64,
    segment: u64,
    elem: u64,
    index_elem: u64,
    bytes: u64,
    scale: u64,
    addr32: u64,
    cur_addr: u64,
    time_enabled: u64,
) -> u64 {
    account_helper!(
        time_enabled,
        gather_helper(
            cpu, mem, dst, mask, index, base, segment, elem, index_elem, bytes, scale, addr32,
            cur_addr, 0
        )
    );
    use x86jit_core::jit_abi::{MemCtx, RET_CONTINUE, RET_UNMAPPED};
    // SAFETY: the compiled-block ABI supplies these live pointers.
    let cpu = unsafe { &mut *cpu.cast::<x86jit_core::CpuState>() };
    // SAFETY: mem is the live compiled-block memory context.
    let ctx = unsafe { &mut *mem.cast::<MemCtx>() };
    let raw = x86jit_core::interp::RawStrMem {
        // SAFETY: mem_self retains the live Memory for this helper call.
        recover_host_faults: unsafe { &*(ctx.mem_self as *const x86jit_core::Memory) }
            .host_fault_recovery(),
        base: ctx.base as *mut u8,
        size: ctx.size,
        guest_base: ctx.guest_base,
    };
    match x86jit_core::interp::gather_run(
        cpu,
        &raw,
        dst as u8,
        mask as u8,
        index as u8,
        base,
        segment,
        elem as u8,
        index_elem as u8,
        bytes as u16,
        scale as u8,
        addr32 != 0,
        cur_addr,
    ) {
        None => RET_CONTINUE,
        Some((f, ea)) => {
            ctx.record_helper_fault(f.addr, u64::from(f.elem), false, ea, elem);
            ctx.record_helper_trap(f.trap);
            RET_UNMAPPED
        }
    }
}

/// AVX1 vector-mask conditional memory move `vmaskmovps/pd`: like
/// [`vmaskmov_mem_helper`], but the mask is a *vector register* whose per-element sign
/// bits form the opmask (via `vec_msb_mask`), not an opmask `k`. Load form always zeroes
/// masked-off lanes. Reuses the shared `masked_load_run`/`masked_store_run`, so JIT ==
/// interpreter and masked-off lanes never fault.
///
/// # Safety
/// `cpu`/`mem` are valid pointers to a `CpuState` / `MemCtx` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vec_maskmov_mem_helper(
    cpu: *mut u8,
    mem: *mut u8,
    reg: u64,
    addr: u64,
    mask: u64,
    elem: u64,
    bytes: u64,
    is_store: u64,
    cur_addr: u64,
    time_enabled: u64,
) -> u64 {
    account_helper!(
        time_enabled,
        vec_maskmov_mem_helper(cpu, mem, reg, addr, mask, elem, bytes, is_store, cur_addr, 0)
    );
    use x86jit_core::jit_abi::{MemCtx, RET_CONTINUE, RET_UNMAPPED};
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let ctx = &mut *(mem as *mut MemCtx);
    let raw = x86jit_core::interp::RawStrMem {
        // SAFETY: mem_self retains the live Memory for this helper call.
        recover_host_faults: unsafe { &*(ctx.mem_self as *const x86jit_core::Memory) }
            .host_fault_recovery(),
        base: ctx.base as *mut u8,
        size: ctx.size,
        guest_base: ctx.guest_base,
    };
    let raw = NotifyingStrMem {
        raw,
        // SAFETY: mem_self retains the live Memory for this helper call.
        memory: unsafe { &*(ctx.mem_self as *const x86jit_core::Memory) },
    };
    let km =
        x86jit_core::interp::vec_msb_mask(&cpu.vec_lanes(mask as usize), elem as u8, bytes as u16);
    // §16 commit-last: validate all selected stores before the shared writer.
    // SDM Vol 2C, VPMASKMOV: masked-off elements do not access memory.
    if is_store != 0 {
        for i in 0..bytes / elem {
            if km & (1 << i) != 0 {
                let ea = addr.wrapping_add(i * elem);
                if ea < raw.raw.guest_base
                    || ea.checked_add(elem).is_none_or(|end| end > raw.raw.size)
                {
                    cpu.rip = cur_addr;
                    let at = if ea >= raw.raw.guest_base && ea < raw.raw.size {
                        raw.raw.size
                    } else {
                        ea
                    };
                    let (operand, width) =
                        x86jit_core::memory::active_operand_span(addr, bytes, elem as u8, km);
                    ctx.record_helper_fault(at, elem, true, operand, width);
                    return RET_UNMAPPED;
                }
            }
        }
    }
    let fault = if is_store != 0 {
        x86jit_core::interp::masked_store_run(
            cpu,
            &raw,
            reg as u8,
            addr,
            km,
            elem as u8,
            bytes as u16,
            cur_addr,
        )
    } else {
        x86jit_core::interp::masked_load_run(
            cpu,
            &raw,
            reg as u8,
            addr,
            km,
            elem as u8,
            true,
            bytes as u16,
            cur_addr,
        )
    };
    match fault {
        None => RET_CONTINUE,
        Some(f) => {
            let (operand, width) =
                x86jit_core::memory::active_operand_span(addr, bytes, elem as u8, km);
            ctx.record_helper_fault(f.addr, u64::from(f.elem), f.write, operand, width);
            ctx.record_helper_trap(f.trap);
            RET_UNMAPPED
        }
    }
}

/// Masked EVEX logic: compute `op(a, b)` then masked-write into `dst`,
/// via the shared `exec_masked_logic` so JIT == interpreter.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vmasked_logic_helper(
    cpu: *mut u8,
    op_code: u64,
    dst: u64,
    a: u64,
    b: u64,
    k: u64,
    elem: u64,
    zeroing: u64,
    bytes: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        vmasked_logic_helper(cpu, op_code, dst, a, b, k, elem, zeroing, bytes, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_masked_logic(
        cpu,
        op_code as u8,
        dst as u8,
        a as u8,
        b as u8,
        k as u8,
        elem as u8,
        zeroing != 0,
        bytes as u16,
    );
}

/// Masked EVEX packed arithmetic: compute the packed op then masked-write
/// into `dst`, via the shared `exec_masked_packed` so JIT == interpreter.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vmasked_packed_helper(
    cpu: *mut u8,
    op_code: u64,
    dst: u64,
    a: u64,
    b: u64,
    k: u64,
    elem: u64,
    zeroing: u64,
    bytes: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        vmasked_packed_helper(cpu, op_code, dst, a, b, k, elem, zeroing, bytes, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_masked_packed(
        cpu,
        op_code as u8,
        dst as u8,
        a as u8,
        b as u8,
        k as u8,
        elem as u8,
        zeroing != 0,
        bytes as u16,
    );
}

/// EVEX packed shift-by-imm over any width with optional write-masking:
/// shifts `a` per `elem`-byte lane and commits into `dst`, via the shared
/// `exec_masked_shift` so JIT == interpreter.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vmasked_shift_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    imm: u64,
    elem: u64,
    right: u64,
    arith: u64,
    k: u64,
    zeroing: u64,
    bytes: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        vmasked_shift_helper(cpu, dst, a, imm, elem, right, arith, k, zeroing, bytes, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_masked_shift(
        cpu,
        dst as u8,
        a as u8,
        imm as u8,
        elem as u8,
        right != 0,
        arith != 0,
        k as u8,
        zeroing != 0,
        bytes as u16,
    );
}

/// AVX2/AVX-512 per-element variable shift `vp{sll,srl,sra}v{w,d,q}`, via the
/// shared `exec_var_shift` → jit == interp. `count` is the count-vector register index.
///
/// # Safety
/// `cpu` is a valid `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn var_shift_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    count: u64,
    elem: u64,
    right: u64,
    arith: u64,
    k: u64,
    zeroing: u64,
    bytes: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        var_shift_helper(cpu, dst, a, count, elem, right, arith, k, zeroing, bytes, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_var_shift(
        cpu,
        dst as u8,
        a as u8,
        count as u8,
        elem as u8,
        right != 0,
        arith != 0,
        k as u8,
        zeroing != 0,
        bytes as u16,
    );
}

/// Packed shift by a scalar register count `vp{sll,srl,sra}{w,d,q} v,v,xmm`, via
/// the shared `exec_shift_reg` → jit == interp. `count` is the count-xmm register index.
///
/// # Safety
/// `cpu` is a valid `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn shift_reg_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    count: u64,
    elem: u64,
    right: u64,
    arith: u64,
    k: u64,
    zeroing: u64,
    bytes: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        shift_reg_helper(cpu, dst, a, count, elem, right, arith, k, zeroing, bytes, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_shift_reg(
        cpu,
        dst as u8,
        a as u8,
        count as u8,
        elem as u8,
        right != 0,
        arith != 0,
        k as u8,
        zeroing != 0,
        bytes as u16,
    );
}

/// GFNI wide/masked `gf2p8{mulb,affineqb,affineinvqb}`, via the shared
/// `exec_gf2p8` → jit == interp. `mode` is the [`x86jit_core::GfniOp`] wire value.
///
/// # Safety
/// `cpu` is a valid `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn gf2p8_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    b: u64,
    imm: u64,
    mode: u64,
    k: u64,
    zeroing: u64,
    bytes: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        gf2p8_helper(cpu, dst, a, b, imm, mode, k, zeroing, bytes, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_gf2p8(
        cpu,
        dst as u8,
        a as u8,
        b as u8,
        imm as u8,
        mode as u8,
        k as u8,
        zeroing != 0,
        bytes as u16,
    );
}

/// GFNI wide/masked with a memory matrix `vgf2p8affineqb ymm,ymm,[mem]`, via the
/// shared `gf2p8_mem_run` over the guest buffer. Handles the `dst == src1` aliasing case.
///
/// # Safety
/// `cpu` is a valid `CpuState`; `mem` is a valid `MemCtx` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn gf2p8_mem_helper(
    cpu: *mut u8,
    mem: *mut u8,
    dst: u64,
    a: u64,
    addr: u64,
    imm: u64,
    mode: u64,
    k: u64,
    zeroing: u64,
    bytes: u64,
    cur_addr: u64,
    time_enabled: u64,
) -> u64 {
    account_helper!(
        time_enabled,
        gf2p8_mem_helper(cpu, mem, dst, a, addr, imm, mode, k, zeroing, bytes, cur_addr, 0)
    );
    use x86jit_core::jit_abi::{MemCtx, RET_CONTINUE, RET_UNMAPPED};
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let ctx = &mut *(mem as *mut MemCtx);
    let raw = x86jit_core::interp::RawStrMem {
        // SAFETY: mem_self retains the live Memory for this helper call.
        recover_host_faults: unsafe { &*(ctx.mem_self as *const x86jit_core::Memory) }
            .host_fault_recovery(),
        base: ctx.base as *mut u8,
        size: ctx.size,
        guest_base: ctx.guest_base,
    };
    match x86jit_core::interp::gf2p8_mem_run(
        cpu,
        &raw,
        dst as u8,
        a as u8,
        addr,
        imm as u8,
        mode as u8,
        k as u8,
        zeroing != 0,
        bytes as u16,
    ) {
        None => RET_CONTINUE,
        Some(f) => {
            // SDM Vol 3A §6.5: faults save the owning instruction's RIP.
            cpu.rip = cur_addr;
            let (mut operand, mut width) = helper_operand_span(addr, bytes);
            if k != 0
                && matches!(
                    x86jit_core::ir::GfniOp::from_u8(mode as u8),
                    x86jit_core::ir::GfniOp::Mulb
                )
            {
                (operand, width) =
                    x86jit_core::memory::active_operand_span(addr, width, 1, cpu.kmask[k as usize]);
            }
            ctx.record_helper_fault(f.addr, u64::from(f.elem), f.write, operand, width);
            ctx.record_helper_trap(f.trap);
            RET_UNMAPPED
        }
    }
}

/// SSE4.2 `pcmpistri`/`pcmpestri`: the string-aggregation index + flags,
/// via the shared `pcmpstr_run`. Writes `out[0] = ecx`, `out[1] = cf|zf<<1|sf<<2|of<<3`;
/// the codegen stores ECX and the flags through its own GPR/flag machinery.
///
/// # Safety
/// `cpu` is a valid `CpuState` for the call; `out` points at two writable `u64`s.
unsafe extern "C" fn pcmpstr_helper(
    cpu: *const u8,
    a: u64,
    b: u64,
    imm: u64,
    explicit: u64,
    out: *mut u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        pcmpstr_helper(cpu, a, b, imm, explicit, out, 0)
    );
    let cpu = &*(cpu as *const x86jit_core::state::CpuState);
    let (ecx, cf, zf, sf, of) =
        x86jit_core::interp::pcmpstr_run(cpu, a as u8, b as u8, imm as u8, explicit != 0);
    *out = ecx as u64;
    *out.add(1) = (cf as u64) | ((zf as u64) << 1) | ((sf as u64) << 2) | ((of as u64) << 3);
}

/// Memory-source `pcmpistri`/`pcmpestri`: source 2 is supplied as the loaded
/// 128-bit value (`bv_lo`/`bv_hi`) rather than a register; source 1 is `cpu.xmm[a]`. The
/// JIT loads (and fault-checks) the operand before the call. Out-slot layout matches
/// [`pcmpstr_helper`].
///
/// # Safety
/// `cpu` is a valid `CpuState` for the call; `out` points at two writable `u64`s.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn pcmpstr_mem_helper(
    cpu: *const u8,
    a: u64,
    bv_lo: u64,
    bv_hi: u64,
    imm: u64,
    explicit: u64,
    out: *mut u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        pcmpstr_mem_helper(cpu, a, bv_lo, bv_hi, imm, explicit, out, 0)
    );
    let cpu = &*(cpu as *const x86jit_core::state::CpuState);
    let bv = (bv_lo as u128) | ((bv_hi as u128) << 64);
    let (ecx, cf, zf, sf, of) =
        x86jit_core::interp::pcmpstr_run_bv(cpu, a as u8, bv, imm as u8, explicit != 0);
    *out = ecx as u64;
    *out.add(1) = (cf as u64) | ((zf as u64) << 1) | ((sf as u64) << 2) | ((of as u64) << 3);
}

/// SSE4.2 `pcmpistrm`/`pcmpestrm`: the string-aggregation MASK (written to XMM0)
/// plus flags, via the shared `pcmpstrm_run`. The helper writes XMM0 directly (`&mut cpu`)
/// and returns the flags in `out[1] = cf|zf<<1|sf<<2|of<<3`; the codegen stores the flags.
///
/// # Safety
/// `cpu` is a valid `CpuState` for the call; `out` points at two writable `u64`s.
unsafe extern "C" fn pcmpstrm_helper(
    cpu: *mut u8,
    a: u64,
    b: u64,
    imm: u64,
    explicit: u64,
    out: *mut u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        pcmpstrm_helper(cpu, a, b, imm, explicit, out, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let (mask, cf, zf, sf, of) =
        x86jit_core::interp::pcmpstrm_run(cpu, a as u8, b as u8, imm as u8, explicit != 0);
    cpu.xmm[0] = mask;
    *out.add(1) = (cf as u64) | ((zf as u64) << 1) | ((sf as u64) << 2) | ((of as u64) << 3);
}

/// Memory-source `pcmpistrm`/`pcmpestrm`: source 2 is the loaded 128-bit value.
/// Out-slot + XMM0 layout matches [`pcmpstrm_helper`].
///
/// # Safety
/// `cpu` is a valid `CpuState` for the call; `out` points at two writable `u64`s.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn pcmpstrm_mem_helper(
    cpu: *mut u8,
    a: u64,
    bv_lo: u64,
    bv_hi: u64,
    imm: u64,
    explicit: u64,
    out: *mut u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        pcmpstrm_mem_helper(cpu, a, bv_lo, bv_hi, imm, explicit, out, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let bv = (bv_lo as u128) | ((bv_hi as u128) << 64);
    let (mask, cf, zf, sf, of) =
        x86jit_core::interp::pcmpstrm_run_bv(cpu, a as u8, bv, imm as u8, explicit != 0);
    cpu.xmm[0] = mask;
    *out.add(1) = (cf as u64) | ((zf as u64) << 1) | ((sf as u64) << 2) | ((of as u64) << 3);
}

/// EVEX `valign{d,q}`: cross-lane element shift, via the shared
/// `exec_valign` so JIT == interpreter.
unsafe extern "C" fn valign_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    b: u64,
    shift: u64,
    elem: u64,
    bytes: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        valign_helper(cpu, dst, a, b, shift, elem, bytes, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_valign(
        cpu,
        dst as u8,
        a as u8,
        b as u8,
        shift as u8,
        elem as u8,
        bytes as u16,
    );
}

/// `vpermt2{b,w,d,q}` helper: two-table cross-lane permute via the shared
/// `exec_vpermt2` so JIT == interpreter. Writes the dst vector reg in CpuState directly
/// (vector state is memory-backed); GPRs untouched.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vpermt2_helper(
    cpu: *mut u8,
    dst: u64,
    idx: u64,
    tbl: u64,
    elem: u64,
    k: u64,
    masked: u64,
    zeroing: u64,
    bytes: u64,
    imode: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        vpermt2_helper(cpu, dst, idx, tbl, elem, k, masked, zeroing, bytes, imode, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_vpermt2(
        cpu,
        dst as u8,
        idx as u8,
        tbl as u8,
        elem as u8,
        k as u8,
        masked != 0,
        zeroing != 0,
        bytes as u16,
        imode != 0,
    );
}

/// Memory-source `vpermt2`/`vpermi2` helper: table 1 is loaded from `[addr]`
/// via the shared `permute2_run` (jit == interp). Fault-capable: returns `RET_UNMAPPED`
/// with the fault recorded in the `MemCtx`.
///
/// # Safety
/// `cpu`/`mem` are valid pointers to a `CpuState` / `MemCtx` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vpermt2_mem_helper(
    cpu: *mut u8,
    mem: *mut u8,
    dst: u64,
    idx: u64,
    addr: u64,
    elem: u64,
    k: u64,
    masked: u64,
    zeroing: u64,
    bytes: u64,
    imode: u64,
    cur_addr: u64,
    time_enabled: u64,
) -> u64 {
    account_helper!(
        time_enabled,
        vpermt2_mem_helper(
            cpu, mem, dst, idx, addr, elem, k, masked, zeroing, bytes, imode, cur_addr, 0
        )
    );
    use x86jit_core::jit_abi::{MemCtx, RET_CONTINUE, RET_UNMAPPED};
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let ctx = &mut *(mem as *mut MemCtx);
    let raw = x86jit_core::interp::RawStrMem {
        // SAFETY: mem_self retains the live Memory for this helper call.
        recover_host_faults: unsafe { &*(ctx.mem_self as *const x86jit_core::Memory) }
            .host_fault_recovery(),
        base: ctx.base as *mut u8,
        size: ctx.size,
        guest_base: ctx.guest_base,
    };
    match x86jit_core::interp::permute2_run(
        cpu,
        &raw,
        dst as u8,
        idx as u8,
        addr,
        elem as u8,
        k as u8,
        masked != 0,
        zeroing != 0,
        bytes as u16,
        imode != 0,
        cur_addr,
    ) {
        None => RET_CONTINUE,
        Some(f) => {
            let (operand, width) = helper_operand_span(addr, bytes);
            ctx.record_helper_fault(f.addr, u64::from(f.elem), f.write, operand, width);
            ctx.record_helper_trap(f.trap);
            RET_UNMAPPED
        }
    }
}

/// Single-source cross-lane permute `vperm{d,q}` helper: via the shared
/// `exec_vperm1` so JIT == interpreter. Writes the dst vector reg (memory-backed).
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vperm1_helper(
    cpu: *mut u8,
    dst: u64,
    idx: u64,
    src: u64,
    elem: u64,
    bytes: u64,
    k: u64,
    masked: u64,
    zeroing: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        vperm1_helper(cpu, dst, idx, src, elem, bytes, k, masked, zeroing, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_vperm1(
        cpu,
        dst as u8,
        idx as u8,
        src as u8,
        elem as u8,
        bytes as u16,
        k as u8,
        masked != 0,
        zeroing != 0,
    );
}

/// Memory-source single-table permute `vperm{d,q} v, idx, [mem]` helper: the
/// table is loaded from `[addr]` via the shared `vperm1_run` (jit == interp).
/// Fault-capable: returns `RET_UNMAPPED` with the fault recorded in the `MemCtx`.
///
/// # Safety
/// `cpu`/`mem` are valid pointers to a `CpuState` / `MemCtx` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vperm1_mem_helper(
    cpu: *mut u8,
    mem: *mut u8,
    dst: u64,
    idx: u64,
    addr: u64,
    elem: u64,
    k: u64,
    masked: u64,
    zeroing: u64,
    bytes: u64,
    cur_addr: u64,
    time_enabled: u64,
) -> u64 {
    account_helper!(
        time_enabled,
        vperm1_mem_helper(cpu, mem, dst, idx, addr, elem, k, masked, zeroing, bytes, cur_addr, 0)
    );
    use x86jit_core::jit_abi::{MemCtx, RET_CONTINUE, RET_UNMAPPED};
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let ctx = &mut *(mem as *mut MemCtx);
    let raw = x86jit_core::interp::RawStrMem {
        // SAFETY: mem_self retains the live Memory for this helper call.
        recover_host_faults: unsafe { &*(ctx.mem_self as *const x86jit_core::Memory) }
            .host_fault_recovery(),
        base: ctx.base as *mut u8,
        size: ctx.size,
        guest_base: ctx.guest_base,
    };
    match x86jit_core::interp::vperm1_run(
        cpu,
        &raw,
        dst as u8,
        idx as u8,
        addr,
        elem as u8,
        k as u8,
        masked != 0,
        zeroing != 0,
        bytes as u16,
        cur_addr,
    ) {
        None => RET_CONTINUE,
        Some(f) => {
            let (operand, width) = helper_operand_span(addr, bytes);
            ctx.record_helper_fault(f.addr, u64::from(f.elem), f.write, operand, width);
            ctx.record_helper_trap(f.trap);
            RET_UNMAPPED
        }
    }
}

/// `vpmov{q,d,w}{d,w,b}` narrowing-move helper: truncate + pack via the
/// shared `exec_vpmov_narrow` so JIT == interpreter. Writes the dst vector reg in
/// CpuState directly (vector state is memory-backed); GPRs untouched.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vpmov_narrow_helper(
    cpu: *mut u8,
    dst: u64,
    src: u64,
    from: u64,
    to: u64,
    src_width: u64,
    k: u64,
    masked: u64,
    zeroing: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        vpmov_narrow_helper(cpu, dst, src, from, to, src_width, k, masked, zeroing, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_vpmov_narrow(
        cpu,
        dst as u8,
        src as u8,
        from as u8,
        to as u8,
        src_width as u16,
        k as u8,
        masked != 0,
        zeroing != 0,
    );
}

/// Narrowing-store helper `vpmov{q,d,w}{d,w,b} [mem], src` (unmasked): truncate
/// then store contiguously via the shared `narrow_store_run` (jit == interp). Fault-capable:
/// returns `RET_UNMAPPED` with the fault address recorded in the `MemCtx`.
///
/// # Safety
/// `cpu`/`mem` are valid pointers to a `CpuState` / `MemCtx` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vpmov_narrow_mem_helper(
    cpu: *mut u8,
    mem: *mut u8,
    src: u64,
    addr: u64,
    from: u64,
    to: u64,
    src_width: u64,
    cur_addr: u64,
    time_enabled: u64,
) -> u64 {
    account_helper!(
        time_enabled,
        vpmov_narrow_mem_helper(cpu, mem, src, addr, from, to, src_width, cur_addr, 0)
    );
    use x86jit_core::jit_abi::{MemCtx, RET_CONTINUE, RET_UNMAPPED};
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let ctx = &mut *(mem as *mut MemCtx);
    let raw = x86jit_core::interp::RawStrMem {
        // SAFETY: mem_self retains the live Memory for this helper call.
        recover_host_faults: unsafe { &*(ctx.mem_self as *const x86jit_core::Memory) }
            .host_fault_recovery(),
        base: ctx.base as *mut u8,
        size: ctx.size,
        guest_base: ctx.guest_base,
    };
    let raw = NotifyingStrMem {
        raw,
        // SAFETY: mem_self retains the live Memory for this helper call.
        memory: unsafe { &*(ctx.mem_self as *const x86jit_core::Memory) },
    };
    match x86jit_core::interp::narrow_store_run(
        cpu,
        &raw,
        src as u8,
        from as u8,
        to as u8,
        src_width as u16,
        addr,
        cur_addr,
    ) {
        None => RET_CONTINUE,
        Some(f) => {
            ctx.record_helper_fault(
                f.addr,
                u64::from(f.elem),
                f.write,
                addr,
                src_width / from * to,
            );
            ctx.record_helper_trap(f.trap);
            RET_UNMAPPED
        }
    }
}

/// `vpmov{s,z}x*` widening-move helper for wide/masked dests: zero/sign-extend
/// via the shared `exec_vpmov_extend_wide` so JIT == interpreter. Writes the dst vector
/// reg in CpuState directly (vector state is memory-backed); GPRs untouched.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vpmov_extend_wide_helper(
    cpu: *mut u8,
    dst: u64,
    src: u64,
    from: u64,
    to: u64,
    signed: u64,
    dst_width: u64,
    k: u64,
    masked: u64,
    zeroing: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        vpmov_extend_wide_helper(cpu, dst, src, from, to, signed, dst_width, k, masked, zeroing, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_vpmov_extend_wide(
        cpu,
        dst as u8,
        src as u8,
        from as u8,
        to as u8,
        signed != 0,
        dst_width as u16,
        k as u8,
        masked != 0,
        zeroing != 0,
    );
}

/// `vpabs{b,w,d,q}` packed absolute-value helper: via the shared `exec_vpabs`
/// so JIT == interpreter. Writes the dst vector reg in CpuState directly (memory-backed).
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vpabs_helper(
    cpu: *mut u8,
    dst: u64,
    src: u64,
    elem: u64,
    dst_width: u64,
    k: u64,
    masked: u64,
    zeroing: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        vpabs_helper(cpu, dst, src, elem, dst_width, k, masked, zeroing, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_vpabs(
        cpu,
        dst as u8,
        src as u8,
        elem as u8,
        dst_width as u16,
        k as u8,
        masked != 0,
        zeroing != 0,
    );
}

/// Masked EVEX unary lane helper `vplzcnt/vprol/vpconflict`: via the shared
/// `exec_vp_unary_lane` so JIT == interpreter. `op` is the [`x86jit_core::ir::VpUnaryOp`]
/// wire value; `imm` is the rotate count (vprol only). Register-only, never faults.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vp_unary_lane_helper(
    cpu: *mut u8,
    dst: u64,
    src: u64,
    op: u64,
    imm: u64,
    elem: u64,
    dst_width: u64,
    k: u64,
    masked: u64,
    zeroing: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        vp_unary_lane_helper(cpu, dst, src, op, imm, elem, dst_width, k, masked, zeroing, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_vp_unary_lane(
        cpu,
        dst as u8,
        src as u8,
        x86jit_core::ir::VpUnaryOp::from_u8(op as u8),
        imm as u8,
        elem as u8,
        dst_width as u16,
        k as u8,
        masked != 0,
        zeroing != 0,
    );
}

/// Masked EVEX blend helper `vpblendm{d,q}`: via the shared `exec_vp_blendm`
/// so JIT == interpreter. `k` is the blend-control opmask. Register-only, never faults.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vp_blendm_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    b: u64,
    k: u64,
    elem: u64,
    dst_width: u64,
    zeroing: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        vp_blendm_helper(cpu, dst, a, b, k, elem, dst_width, zeroing, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_vp_blendm(
        cpu,
        dst as u8,
        a as u8,
        b as u8,
        k as u8,
        elem as u8,
        dst_width as u16,
        zeroing != 0,
    );
}

/// Masked EVEX 128-bit-lane shuffle helper `vshuff32x4/64x2`: via the shared
/// `exec_vshuf_lane` so JIT == interpreter. Register-only, never faults.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vshuf_lane_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    b: u64,
    imm: u64,
    elem: u64,
    dst_width: u64,
    k: u64,
    masked: u64,
    zeroing: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        vshuf_lane_helper(cpu, dst, a, b, imm, elem, dst_width, k, masked, zeroing, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_vshuf_lane(
        cpu,
        dst as u8,
        a as u8,
        b as u8,
        imm as u8,
        elem as u8,
        dst_width as u16,
        k as u8,
        masked != 0,
        zeroing != 0,
    );
}

/// Masked EVEX `vpmultishiftqb` helper (VBMI): via the shared
/// `exec_vp_multishift` so JIT == interpreter. Register-only, never faults.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vp_multishift_helper(
    cpu: *mut u8,
    dst: u64,
    ctrl: u64,
    data: u64,
    dst_width: u64,
    k: u64,
    masked: u64,
    zeroing: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        vp_multishift_helper(cpu, dst, ctrl, data, dst_width, k, masked, zeroing, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_vp_multishift(
        cpu,
        dst as u8,
        ctrl as u8,
        data as u8,
        dst_width as u16,
        k as u8,
        masked != 0,
        zeroing != 0,
    );
}

/// `vpshufb` (EVEX) per-lane byte-shuffle helper: via the shared
/// `exec_vpshufb_wide` so JIT == interpreter. Writes the dst vector reg (memory-backed).
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vpshufb_wide_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    idx: u64,
    bytes: u64,
    k: u64,
    masked: u64,
    zeroing: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        vpshufb_wide_helper(cpu, dst, a, idx, bytes, k, masked, zeroing, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_vpshufb_wide(
        cpu,
        dst as u8,
        a as u8,
        idx as u8,
        bytes as u16,
        k as u8,
        masked != 0,
        zeroing != 0,
    );
}

/// `vpshufd` (EVEX/VEX-256) per-lane dword-shuffle helper: via the shared
/// `exec_vshuffle32_wide` so JIT == interpreter. Writes the dst vector reg (memory-backed).
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vshuffle32_wide_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    imm: u64,
    bytes: u64,
    k: u64,
    masked: u64,
    zeroing: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        vshuffle32_wide_helper(cpu, dst, a, imm, bytes, k, masked, zeroing, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_vshuffle32_wide(
        cpu,
        dst as u8,
        a as u8,
        imm as u8,
        bytes as u16,
        k as u8,
        masked != 0,
        zeroing != 0,
    );
}

/// `pack{ss,us}{wb,dw}` saturating-pack helper: via the shared `exec_vpack`
/// so JIT == interpreter. Writes the dst vector reg (memory-backed).
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
unsafe extern "C" fn vpack_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    b: u64,
    from_elem: u64,
    signed: u64,
    bytes: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        vpack_helper(cpu, dst, a, b, from_elem, signed, bytes, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_vpack(
        cpu,
        dst as u8,
        a as u8,
        b as u8,
        from_elem as u8,
        signed != 0,
        bytes as u16,
    );
}

/// Memory-source variant of [`vpack_helper`]: the 128-bit second source is
/// passed as two i64 halves after all guest loads succeed. Read source 1 from `a`
/// at `lane`, so dst can alias it without losing any unread source lane.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vpack_mem_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    lo: u64,
    hi: u64,
    from_elem: u64,
    signed: u64,
    lane: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        vpack_mem_helper(cpu, dst, a, lo, hi, from_elem, signed, lane, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let b = (lo as u128) | ((hi as u128) << 64);
    // op1 by VALUE from `a`, not read back out of `dst` (§16).
    let av = cpu.vec_lanes(a as usize)[lane as usize];
    let r = x86jit_core::interp::pack_lane(av, b, from_elem as u8, signed != 0);
    if lane == 0 {
        cpu.xmm[dst as usize] = r;
    } else {
        cpu.ymm_hi[dst as usize] = r;
    }
}

/// SSE3 lane-combining packed float helper (register form): `haddp`/`hsubp`/
/// `addsubp` via the shared `hfloat_reg` so JIT == interpreter.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vhfloat_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    b: u64,
    op: u64,
    f64_prec: u64,
    bytes: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        vhfloat_helper(cpu, dst, a, b, op, f64_prec, bytes, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let prec = if f64_prec != 0 {
        x86jit_core::FPrec::F64
    } else {
        x86jit_core::FPrec::F32
    };
    let op = x86jit_core::interp::hfloat_op_from_code(op as u8);
    x86jit_core::interp::hfloat_reg(cpu, dst as u8, a as u8, b as u8, op, prec, bytes as u16);
}

/// Memory-source variant of [`vhfloat_helper`]: the `bytes`-wide
/// second source is passed as up-to-four i64 halves (loaded — and fault-checked — in JIT
/// code); `a` is op1 (register source). Combines per 128-bit lane.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vhfloat_mem_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    lo0: u64,
    hi0: u64,
    lo1: u64,
    hi1: u64,
    op: u64,
    f64_prec: u64,
    bytes: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        vhfloat_mem_helper(cpu, dst, a, lo0, hi0, lo1, hi1, op, f64_prec, bytes, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let b = [
        (lo0 as u128) | ((hi0 as u128) << 64),
        (lo1 as u128) | ((hi1 as u128) << 64),
        0,
        0,
    ];
    let prec = if f64_prec != 0 {
        x86jit_core::FPrec::F64
    } else {
        x86jit_core::FPrec::F32
    };
    let op = x86jit_core::interp::hfloat_op_from_code(op as u8);
    x86jit_core::interp::hfloat_mem(cpu, dst as u8, a as u8, b, op, prec, bytes as u16);
}

/// SSSE3 packed-integer horizontal helper (register form): `phaddw/d/sw`,
/// `phsubw/d/sw` via the shared `hint_reg` so JIT == interpreter.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
unsafe extern "C" fn vhint_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    b: u64,
    op: u64,
    bytes: u64,
    time_enabled: u64,
) {
    account_helper!(time_enabled, vhint_helper(cpu, dst, a, b, op, bytes, 0));
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let op = x86jit_core::interp::hint_op_from_code(op as u8);
    x86jit_core::interp::hint_reg(cpu, dst as u8, a as u8, b as u8, op, bytes as u16);
}

/// Memory-source variant of [`vhint_helper`]: the 128/256-bit second source
/// is passed as four i64 halves (loaded — and fault-checked — in JIT code). op1 comes from
/// `a` (explicit, §16). The high lane is ignored for the 128-bit form.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vhint_mem_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    lo0: u64,
    hi0: u64,
    lo1: u64,
    hi1: u64,
    op: u64,
    bytes: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        vhint_mem_helper(cpu, dst, a, lo0, hi0, lo1, hi1, op, bytes, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let blo = (lo0 as u128) | ((hi0 as u128) << 64);
    let bhi = (lo1 as u128) | ((hi1 as u128) << 64);
    let op = x86jit_core::interp::hint_op_from_code(op as u8);
    x86jit_core::interp::hint_mem(cpu, dst as u8, a as u8, blo, bhi, op, bytes as u16);
}

/// F16C `vcvtph2ps` helper: convert `lanes` binary16 halves in the register
/// `src` to f32 via the shared `cvtph2ps` so JIT == interpreter. Writes `dst`'s low 128
/// and ymm_hi (0 for the 4-lane form).
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
unsafe extern "C" fn cvtph2ps_helper(
    cpu: *mut u8,
    dst: u64,
    src: u64,
    lanes: u64,
    time_enabled: u64,
) {
    account_helper!(time_enabled, cvtph2ps_helper(cpu, dst, src, lanes, 0));
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let (lo, hi) = x86jit_core::interp::cvtph2ps(cpu.xmm[src as usize], lanes as usize);
    cpu.xmm[dst as usize] = lo;
    cpu.ymm_hi[dst as usize] = hi;
}

/// Apply one already-loaded memory count lane after all guest reads succeed.
/// # Safety
/// `cpu` points to the active CpuState; `lane` is zero or one.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn var_shift_mem_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    lo: u64,
    hi: u64,
    lane: u64,
    elem: u64,
    right: u64,
    arith: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        var_shift_mem_helper(cpu, dst, a, lo, hi, lane, elem, right, arith, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let av = cpu.vec_lanes(a as usize)[lane as usize];
    let count = u128::from(lo) | (u128::from(hi) << 64);
    let r = x86jit_core::interp::var_shift_lane(av, count, elem as u8, right != 0, arith != 0);
    if lane == 0 {
        cpu.xmm[dst as usize] = r;
    } else {
        cpu.ymm_hi[dst as usize] = r;
    }
}

/// Probe every destination page before a multi-transfer conversion store.
/// # Safety
/// `mem` retains the active MemCtx; generated code has synchronized CpuState.
unsafe extern "C" fn vstore_preflight_helper(
    mem: *mut u8,
    bytes: u64,
    addr: u64,
    time_enabled: u64,
) -> u64 {
    account_helper!(time_enabled, vstore_preflight_helper(mem, bytes, addr, 0));
    use x86jit_core::interp::StrMem;
    use x86jit_core::jit_abi::{MemCtx, RET_UNMAPPED};
    let ctx = &mut *(mem as *mut MemCtx);
    // Called by crossing split stores and conversion stores. Actual-access probes
    // use transfer fixups even when ordinary Rust transfer recovery is disabled.
    let raw = x86jit_core::interp::RawStrMem {
        base: ctx.base as *mut u8,
        size: ctx.size,
        guest_base: ctx.guest_base,
        recover_host_faults: true,
    };
    match raw.preflight_store(addr, u64::MAX, 1, bytes as u16) {
        Ok(()) => ctx.base + (addr - ctx.guest_base),
        Err((at, trap)) => {
            ctx.record_helper_fault(at, 1, true, addr, bytes);
            ctx.record_helper_trap(trap);
            RET_UNMAPPED
        }
    }
}

/// Compute the packed half-precision value without changing guest registers.
/// # Safety
/// `cpu` points to the active CpuState and `out` to a writable 16-byte stack slot.
unsafe extern "C" fn cvtps2ph_value_helper(
    cpu: *mut u8,
    src: u64,
    lanes: u64,
    rc: u64,
    out: *mut u128,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        cvtps2ph_value_helper(cpu, src, lanes, rc, out, 0)
    );
    let cpu = &*(cpu as *const x86jit_core::state::CpuState);
    let rc = if rc & 4 != 0 {
        ((cpu.mxcsr >> 13) & 3) as u8
    } else {
        rc as u8
    };
    *out = x86jit_core::interp::cvtps2ph_with_mxcsr(
        cpu.xmm[src as usize],
        cpu.ymm_hi[src as usize],
        lanes as usize,
        rc,
        cpu.mxcsr,
    );
}

/// F16C `vcvtps2ph` helper: convert `lanes` f32 lanes of the register `src` to
/// binary16 under rounding-control `rc` via the shared `cvtps2ph` so JIT == interpreter.
/// Writes `dst`'s low 128 (upper cleared).
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
unsafe extern "C" fn cvtps2ph_helper(
    cpu: *mut u8,
    dst: u64,
    src: u64,
    lanes: u64,
    rc: u64,
    time_enabled: u64,
) {
    account_helper!(time_enabled, cvtps2ph_helper(cpu, dst, src, lanes, rc, 0));
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let out = x86jit_core::interp::cvtps2ph_with_mxcsr(
        cpu.xmm[src as usize],
        cpu.ymm_hi[src as usize],
        lanes as usize,
        if rc & 4 != 0 {
            ((cpu.mxcsr >> 13) & 3) as u8
        } else {
            rc as u8
        },
        cpu.mxcsr,
    );
    cpu.xmm[dst as usize] = out;
    cpu.ymm_hi[dst as usize] = 0;
}

/// SSE4.1 `phminposuw` helper: via the shared `phminposuw` so JIT == interp.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
unsafe extern "C" fn phminposuw_helper(cpu: *mut u8, dst: u64, src: u64, time_enabled: u64) {
    account_helper!(time_enabled, phminposuw_helper(cpu, dst, src, 0));
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    cpu.xmm[dst as usize] = x86jit_core::interp::phminposuw(cpu.xmm[src as usize]);
}

/// SSE4.1 `mpsadbw`/`vmpsadbw` helper: per 128-bit lane via the shared `mpsadbw`
/// so JIT == interp. The source value is loaded before any destination is written.
/// `half` selects the destination lane; the caller selects that lane's immediate bits.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
unsafe extern "C" fn mpsadbw_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    lo: u64,
    hi: u64,
    imm: u64,
    half: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        mpsadbw_helper(cpu, dst, a, lo, hi, imm, half, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let b = lo as u128 | ((hi as u128) << 64);
    let av = if half == 0 {
        cpu.xmm[a as usize]
    } else {
        cpu.ymm_hi[a as usize]
    };
    let r = x86jit_core::interp::mpsadbw(av, b, imm as u8);
    if half == 0 {
        cpu.xmm[dst as usize] = r;
    } else {
        cpu.ymm_hi[dst as usize] = r;
    }
}

/// `pmaddwd` multiply-add helper: via the shared `exec_pmaddwd` so
/// JIT == interpreter. Writes the dst vector reg (memory-backed).
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
unsafe extern "C" fn pmaddwd_helper(cpu: *mut u8, dst: u64, a: u64, b: u64, time_enabled: u64) {
    account_helper!(time_enabled, pmaddwd_helper(cpu, dst, a, b, 0));
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_pmaddwd(cpu, dst as u8, a as u8, b as u8);
}

/// VEX `vpmaddwd`/`vpmaddubsw` register helper: via the shared `exec_v_pmadd`
/// so JIT == interpreter. Width-generic (`bytes` = 16/32); `ubsw` picks the byte-pair form.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
unsafe extern "C" fn vpmadd_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    b: u64,
    ubsw: u64,
    bytes: u64,
    time_enabled: u64,
) {
    account_helper!(time_enabled, vpmadd_helper(cpu, dst, a, b, ubsw, bytes, 0));
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_v_pmadd(cpu, dst as u8, a as u8, b as u8, ubsw != 0, bytes as u16);
}

/// Memory-source variant of [`vpmadd_helper`]: pmadd is per-128-bit-lane, so one
/// 128-bit `[mem]` lane is passed as two i64 halves (loaded — and fault-checked — in JIT
/// code) and `hi_half` selects the low (xmm) vs high (ymm_hi) lane of `dst`/`a`.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
unsafe extern "C" fn vpmadd_mem_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    lo: u64,
    hi: u64,
    ubsw: u64,
    hi_half: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        vpmadd_mem_helper(cpu, dst, a, lo, hi, ubsw, hi_half, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let memv = (lo as u128) | ((hi as u128) << 64);
    x86jit_core::interp::exec_v_pmadd_mem_lane(
        cpu,
        dst as u8,
        a as u8,
        memv,
        ubsw != 0,
        hi_half != 0,
    );
}

/// EVEX lane-broadcast helper (register form): via the shared
/// `exec_broadcast_lane` so JIT == interpreter. Register-only, never faults.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn broadcast_lane_helper(
    cpu: *mut u8,
    dst: u64,
    src: u64,
    chunk: u64,
    elem: u64,
    dst_width: u64,
    k: u64,
    masked: u64,
    zeroing: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        broadcast_lane_helper(cpu, dst, src, chunk, elem, dst_width, k, masked, zeroing, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_broadcast_lane(
        cpu,
        dst as u8,
        src as u8,
        chunk as u8,
        elem as u8,
        dst_width as u16,
        k as u8,
        masked != 0,
        zeroing != 0,
    );
}

/// EVEX lane-broadcast helper (memory form): loads the chunk from `[base]` via
/// the shared `broadcast_lane_mem_run` (jit == interp). Fault-capable: returns
/// `RET_UNMAPPED` with the fault recorded in the `MemCtx`.
///
/// # Safety
/// `cpu`/`mem` are valid pointers to a `CpuState` / `MemCtx` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn broadcast_lane_mem_helper(
    cpu: *mut u8,
    mem: *mut u8,
    dst: u64,
    base: u64,
    chunk: u64,
    elem: u64,
    dst_width: u64,
    k: u64,
    masked: u64,
    zeroing: u64,
    cur_addr: u64,
    time_enabled: u64,
) -> u64 {
    account_helper!(
        time_enabled,
        broadcast_lane_mem_helper(
            cpu, mem, dst, base, chunk, elem, dst_width, k, masked, zeroing, cur_addr, 0
        )
    );
    use x86jit_core::jit_abi::{MemCtx, RET_CONTINUE, RET_UNMAPPED};
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let ctx = &mut *(mem as *mut MemCtx);
    let raw = x86jit_core::interp::RawStrMem {
        // SAFETY: mem_self retains the live Memory for this helper call.
        recover_host_faults: unsafe { &*(ctx.mem_self as *const x86jit_core::Memory) }
            .host_fault_recovery(),
        base: ctx.base as *mut u8,
        size: ctx.size,
        guest_base: ctx.guest_base,
    };
    match x86jit_core::interp::broadcast_lane_mem_run(
        cpu,
        &raw,
        dst as u8,
        base,
        chunk as u8,
        elem as u8,
        dst_width as u16,
        k as u8,
        masked != 0,
        zeroing != 0,
        cur_addr,
    ) {
        None => RET_CONTINUE,
        Some(f) => {
            let (operand, width) = if masked != 0 {
                x86jit_core::memory::active_broadcast_span(
                    base,
                    chunk as u8,
                    elem as u8,
                    dst_width as u16,
                    cpu.kmask[k as usize],
                )
            } else {
                (base, chunk)
            };
            ctx.record_helper_fault(f.addr, u64::from(f.elem), f.write, operand, width);
            ctx.record_helper_trap(f.trap);
            RET_UNMAPPED
        }
    }
}

/// FMA3 helper (register form): fused multiply-add via the shared `exec_fma`
/// so JIT == interpreter. Writes the dst vector reg (memory-backed).
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn fma_helper(
    cpu: *mut u8,
    dst: u64,
    x: u64,
    y: u64,
    z: u64,
    prec_f64: u64,
    scalar: u64,
    neg_prod: u64,
    neg_add: u64,
    bytes: u64,
    alt_sign: u64,
    k: u64,
    masked: u64,
    zeroing: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        fma_helper(
            cpu, dst, x, y, z, prec_f64, scalar, neg_prod, neg_add, bytes, alt_sign, k, masked,
            zeroing, 0
        )
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_fma(
        cpu,
        dst as u8,
        x as u8,
        y as u8,
        z as u8,
        prec_f64 != 0,
        scalar != 0,
        neg_prod != 0,
        neg_add != 0,
        bytes as u16,
        alt_sign as u8,
        k as u8,
        masked != 0,
        zeroing != 0,
    );
}

/// FMA3 memory-form helper: one source is loaded from `[base]` via the shared
/// `fma_mem_run` (jit == interp). Fault-capable: returns `RET_UNMAPPED` with the fault
/// recorded in the `MemCtx`.
///
/// # Safety
/// `cpu`/`mem` are valid pointers to a `CpuState` / `MemCtx` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn fma_mem_helper(
    cpu: *mut u8,
    mem: *mut u8,
    dst: u64,
    x: u64,
    y: u64,
    z: u64,
    base: u64,
    mem_role: u64,
    prec_f64: u64,
    scalar: u64,
    neg_prod: u64,
    neg_add: u64,
    bytes: u64,
    alt_sign: u64,
    cur_addr: u64,
    k: u64,
    masked: u64,
    zeroing: u64,
    time_enabled: u64,
) -> u64 {
    account_helper!(
        time_enabled,
        fma_mem_helper(
            cpu, mem, dst, x, y, z, base, mem_role, prec_f64, scalar, neg_prod, neg_add, bytes,
            alt_sign, cur_addr, k, masked, zeroing, 0
        )
    );
    use x86jit_core::jit_abi::{MemCtx, RET_CONTINUE, RET_UNMAPPED};
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let ctx = &mut *(mem as *mut MemCtx);
    let raw = x86jit_core::interp::RawStrMem {
        // SAFETY: mem_self retains the live Memory for this helper call.
        recover_host_faults: unsafe { &*(ctx.mem_self as *const x86jit_core::Memory) }
            .host_fault_recovery(),
        base: ctx.base as *mut u8,
        size: ctx.size,
        guest_base: ctx.guest_base,
    };
    let writemask = if masked != 0 { Some(k as u8) } else { None };
    match x86jit_core::interp::fma_mem_run(
        cpu,
        &raw,
        dst as u8,
        x as u8,
        y as u8,
        z as u8,
        base,
        mem_role as u8,
        prec_f64 != 0,
        scalar != 0,
        neg_prod != 0,
        neg_add != 0,
        bytes as u16,
        alt_sign as u8,
        cur_addr,
        writemask,
        zeroing != 0,
    ) {
        None => RET_CONTINUE,
        Some(f) => {
            let (mut operand, mut width) = helper_operand_span(base, bytes);
            if scalar != 0 {
                width = if prec_f64 != 0 { 8 } else { 4 };
            }
            if masked != 0 && bytes >> 16 == 0 {
                (operand, width) = x86jit_core::memory::active_operand_span(
                    base,
                    width,
                    if prec_f64 != 0 { 8 } else { 4 },
                    cpu.kmask[k as usize],
                );
            }
            ctx.record_helper_fault(f.addr, u64::from(f.elem), f.write, operand, width);
            ctx.record_helper_trap(f.trap);
            RET_UNMAPPED
        }
    }
}

/// AES-NI helper (register form): dispatches all 6 AES ops via the shared
/// `x86jit_core::aes` primitives so JIT == interpreter. `op`: 0=enc,1=dec,2=enclast,
/// 3=declast,4=imc,5=keygen. Round ops use `a` (state) + `b` (round key); imc/keygen
/// use `a` as the single source (`imm` = RCON for keygen). Register-only, never faults.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn aes_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    b: u64,
    op: u64,
    imm: u64,
    time_enabled: u64,
) {
    account_helper!(time_enabled, aes_helper(cpu, dst, a, b, op, imm, 0));
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    match op {
        4 => x86jit_core::interp::exec_aes_imc(cpu, dst as u8, a as u8),
        5 => x86jit_core::interp::exec_aes_keygen(cpu, dst as u8, a as u8, imm as u8),
        _ => x86jit_core::interp::exec_aes(cpu, dst as u8, a as u8, b as u8, op as u8),
    }
}

/// AES-NI helper (memory form): the 128-bit memory source is already loaded
/// (fault handled natively before the call) and passed as `lo`/`hi`. Same op dispatch
/// as [`aes_helper`]; round ops use `a` (state) + the loaded key, imc/keygen use the
/// loaded value as the single source.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn aes_mem_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    lo: u64,
    hi: u64,
    op: u64,
    imm: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        aes_mem_helper(cpu, dst, a, lo, hi, op, imm, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let v = (lo as u128) | ((hi as u128) << 64);
    match op {
        4 => x86jit_core::interp::exec_aes_imc_mem(cpu, dst as u8, v),
        5 => x86jit_core::interp::exec_aes_keygen_mem(cpu, dst as u8, v, imm as u8),
        _ => x86jit_core::interp::exec_aes_mem(cpu, dst as u8, a as u8, v, op as u8),
    }
}

/// SHA-NI helper (register form): dispatches all 7 SHA ops via the shared
/// `x86jit_core::sha` primitives so JIT == interpreter. `op` is the [`ShaOp`] wire value;
/// `a` = op1 (dst), `b` = op2, `imm` = `sha1rnds4`'s immediate. `sha256rnds2` reads xmm0
/// implicitly inside the entry point. Register-only, never faults.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn sha_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    b: u64,
    op: u64,
    imm: u64,
    time_enabled: u64,
) {
    account_helper!(time_enabled, sha_helper(cpu, dst, a, b, op, imm, 0));
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_sha(cpu, dst as u8, a as u8, b as u8, imm as u8, op as u8);
}

/// SHA-NI helper (memory form): the 128-bit op2 memory source is already loaded
/// (fault handled natively before the call) and passed as `lo`/`hi`. Same op dispatch as
/// [`sha_helper`]; `sha256rnds2` reads xmm0 implicitly inside the entry point.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn sha_mem_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    lo: u64,
    hi: u64,
    op: u64,
    imm: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        sha_mem_helper(cpu, dst, a, lo, hi, op, imm, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let v = (lo as u128) | ((hi as u128) << 64);
    x86jit_core::interp::exec_sha_mem(cpu, dst as u8, a as u8, v, imm as u8, op as u8);
}

/// GFNI helper (register form): dispatches `gf2p8mulb/affineqb/affineinvqb`
/// via the shared `x86jit_core::gfni` primitives so JIT == interpreter. `op` is the
/// [`x86jit_core::GfniOp`] wire value; `a` = op1, `b` = op2, `imm` = affine constant.
/// Register-only, never faults.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn gfni_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    b: u64,
    op: u64,
    imm: u64,
    time_enabled: u64,
) {
    account_helper!(time_enabled, gfni_helper(cpu, dst, a, b, op, imm, 0));
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_gfni(cpu, dst as u8, a as u8, b as u8, imm as u8, op as u8);
}

/// GFNI helper (memory form): the 128-bit op2 memory source is already loaded
/// (fault handled natively before the call) and passed as `lo`/`hi`. Same op dispatch
/// as [`gfni_helper`].
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn gfni_mem_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    lo: u64,
    hi: u64,
    op: u64,
    imm: u64,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        gfni_mem_helper(cpu, dst, a, lo, hi, op, imm, 0)
    );
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let v = (lo as u128) | ((hi as u128) << 64);
    x86jit_core::interp::exec_gfni_mem(cpu, dst as u8, a as u8, v, imm as u8, op as u8);
}

/// PCLMULQDQ helper (register form): carry-less multiply via the shared
/// `x86jit_core::pclmul` primitive so JIT == interpreter. `a` = op1, `b` = op2, `imm`
/// selects the 64-bit halves. Register-only, never faults.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
unsafe extern "C" fn pclmul_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    b: u64,
    imm: u64,
    time_enabled: u64,
) {
    account_helper!(time_enabled, pclmul_helper(cpu, dst, a, b, imm, 0));
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    x86jit_core::interp::exec_pclmul(cpu, dst as u8, a as u8, b as u8, imm as u8);
}

/// PCLMULQDQ helper (memory form): the 128-bit op2 memory source is already
/// loaded (fault handled natively before the call) and passed as `lo`/`hi`. Same as
/// [`pclmul_helper`] with the loaded value as op2.
///
/// # Safety
/// `cpu` is a valid pointer to a `CpuState` for the call.
unsafe extern "C" fn pclmul_mem_helper(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    lo: u64,
    hi: u64,
    imm: u64,
    time_enabled: u64,
) {
    account_helper!(time_enabled, pclmul_mem_helper(cpu, dst, a, lo, hi, imm, 0));
    let cpu = &mut *(cpu as *mut x86jit_core::state::CpuState);
    let v = (lo as u128) | ((hi as u128) << 64);
    x86jit_core::interp::exec_pclmul_mem(cpu, dst as u8, a as u8, v, imm as u8);
}

/// Upper bound on distinct helpers, for the fixed call-counter array.
/// Asserted against the real count when the table is built, so adding a helper past
/// this fails loudly instead of writing out of bounds.
const MAX_HELPERS: usize = 128;

/// Bounded background-compile queue depth (bg-tier): a full queue makes
/// `tier_up_async` return `Busy` and the block stays interpreted — never an inline
/// compile spike under peak pressure.
const TIER_QUEUE_CAP: usize = 64;

// Bound publication latency without waiting for more work. Cranelift 0.135.8
// memory/system.rs::set_readable_and_executable seals the current mapping, so only
// definitions before the same finalize can share its unused page tail.
const TIER_FINALIZE_BATCH: usize = TIER_QUEUE_CAP;

/// Which *host* instruction set Cranelift may emit for the guest IR — the
/// host-codegen axis, orthogonal to the guest ISA (`GuestCpuFeatures`).
/// Lives on [`JitBackend`], not `VmConfig`: the interpreter has no codegen target (it
/// is plain Rust fixed at compile time), so this would be meaningless on the shared
/// config. Guest-invisible — only instruction selection changes, not results — so the
/// interpreter stays the reference oracle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HostTarget {
    /// Detect the running host and use all its features (`cranelift_native`) — the
    /// default. A hot loop tiers into host-optimal code.
    #[default]
    Native,
    /// Forbid AVX and above (AVX2/AVX-512/FMA): deterministic, AOT-cacheable output
    /// that runs on any SSE4.1 host. Guest AVX/AVX-512 still executes — Cranelift
    /// lowers our 128-bit-lane IR to SSE.
    Baseline,
}

/// Cranelift mid-end optimizer level — how hard Cranelift works on the IR
/// we hand it, orthogonal to [`HostTarget`] (which host instructions it may pick).
///
/// At Cranelift's default of `none` there is no egraph pass, hence no GVN, LICM,
/// constant folding or redundant-load elimination. Lifted x86 is unusually full of
/// exactly what those passes remove — flags recomputed and
/// overwritten unread, the same base+displacement rebuilt per access, reloads of
/// provably unchanged registers — because a lifter translates instruction by
/// instruction and cannot see across that boundary.
///
/// Guest-invisible like `HostTarget`: optimization must not change results, so
/// interp == JIT holds at every level and the interpreter stays the oracle.
///
/// **Pick this from the tier-up policy, not unconditionally** — see
/// [`OptLevel::for_tiering`]. Optimizing is only worth its compile time for code
/// that runs many times; without tier-up every block is compiled on first
/// execution, so the mid-end is pure loss. Measured: turning it on for an eager
/// run cost +19% to +82% on the whole-program tests (sqlite_file 51.7 s -> 93.9 s)
/// and blew a Go server's 120 s startup deadline outright.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OptLevel {
    /// No mid-end optimization — Cranelift's own default. Fastest to compile,
    /// slowest code. The default here too, because a backend
    /// built without a stated tier-up policy has to assume its blocks may be cold.
    #[default]
    None,
    /// Optimize for speed. Measured the best guest throughput of the three — hot
    /// workloads gain 5-25% (sha256 -24.8%, memcpy -18.5%) and the code is 7.8%
    /// smaller — at 45-48% more compile time. Worth it only for blocks that tiered
    /// up, i.e. already proved hot, where tier-up also runs off the vcpu on the
    /// background worker so the longer compile does not stall execution.
    Speed,
    /// Optimize for speed, then for size. Same passes as [`Self::Speed`] plus a
    /// size bias; useful when code-cache footprint matters more than peak speed.
    SpeedAndSize,
}

impl OptLevel {
    /// The level to use given whether the VM tiers up.
    ///
    /// `tiered` true means a block only reaches the compiler after running
    /// `tier_up_after` times, so everything compiled is by construction hot and
    /// worth optimizing. `false` is eager compilation — every block compiled on
    /// first execution, most of them run once — where the mid-end only adds compile
    /// time to code that will never repay it.
    ///
    /// Cranelift bakes `opt_level` into the ISA's `Flags`, and a `JITModule` owns
    /// one ISA, so the choice cannot be made per block without a second module.
    /// Deciding at backend construction, where the tier policy is already known, is
    /// the same decision one step earlier.
    pub fn for_tiering(tiered: bool) -> Self {
        if tiered {
            OptLevel::Speed
        } else {
            OptLevel::None
        }
    }

    /// The `opt_level` value Cranelift's settings builder expects.
    fn flag(self) -> &'static str {
        match self {
            OptLevel::None => "none",
            OptLevel::Speed => "speed",
            OptLevel::SpeedAndSize => "speed_and_size",
        }
    }

    /// Parse the `X86JIT_OPT_LEVEL` spelling; `None` if unrecognized.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "none" => Some(OptLevel::None),
            "speed" => Some(OptLevel::Speed),
            "speed_and_size" => Some(OptLevel::SpeedAndSize),
            _ => None,
        }
    }
}

/// Take a lock, recovering if a previous holder panicked.
///
/// The panic is a real failure — but it is one that already happened and already
/// printed. Poisoning turns it into a *different* panic on every later JIT entry, at a
/// line with nothing to do with the cause, and that secondary symptom then hides the
/// defect that produced it.
///
/// Recovering costs nothing here: the `Jit` is rebuildable and every caller already
/// handles the `None` case. `PoisonError` does not carry the original payload — Rust
/// does not keep it — so this cannot re-report the cause, only stop drowning it and
/// say once that it happened.
fn lock_recovering<'a, T>(m: &'a std::sync::Mutex<T>, what: &str) -> std::sync::MutexGuard<'a, T> {
    m.lock().unwrap_or_else(|poisoned| {
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| {
            use std::io::Write as _;
            let _ = writeln!(
                std::io::stderr(),
                "x86jit: recovering the {what} lock — a previous holder panicked. THAT panic is \
                 the failure; look for it earlier in this log. Continuing so it is not buried \
                 under a cascade of PoisonError."
            );
        });
        poisoned.into_inner()
    })
}

/// Kind of compiled translation unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompiledCodeKind {
    Block,
    Region,
}

/// Published executable host range and the guest entry that produced it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompiledCode {
    pub host_start: usize,
    pub host_len: u32,
    pub guest_rip: u64,
    pub kind: CompiledCodeKind,
}

type Observer<T> = Arc<dyn Fn(T) + Send + Sync>;
type CodeObserver = Observer<CompiledCode>;

/// A failed block or region compilation, including its input size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompileFailure {
    pub guest_rip: u64,
    pub kind: CompiledCodeKind,
    pub reason: x86jit_core::CompileFailureReason,
    /// Sum of the unit's source byte spans.
    pub guest_bytes: u64,
    pub instructions: u64,
    pub ir_ops: usize,
    pub blocks: usize,
}

impl CompileFailure {
    fn for_blocks<'a>(
        guest_rip: u64,
        kind: CompiledCodeKind,
        error: CompileError,
        blocks: impl IntoIterator<Item = &'a IrBlock>,
    ) -> Self {
        let mut event = Self {
            guest_rip,
            kind,
            reason: error.reason(),
            guest_bytes: 0,
            instructions: 0,
            ir_ops: 0,
            blocks: 0,
        };
        for block in blocks {
            event.guest_bytes += u64::from(block.guest_len);
            event.instructions += u64::from(block.icount);
            event.ir_ops += block.ops.len();
            event.blocks += 1;
        }
        event
    }

    fn for_request(req: &TierUpRequest, error: CompileError) -> Self {
        match &req.unit {
            TierUpUnit::Block(block) => {
                Self::for_blocks(req.pc, CompiledCodeKind::Block, error, [block.as_ref()])
            }
            TierUpUnit::Region(region) => {
                Self::for_blocks(req.pc, CompiledCodeKind::Region, error, &region.blocks)
            }
        }
    }
}

type FailureObserver = Observer<CompileFailure>;

/// Engine worker entry point supplied to an embedder's worker spawner.
pub type WorkerTask = Box<dyn FnOnce() + Send + 'static>;
type WorkerSpawner = Box<dyn Fn(WorkerTask) -> std::io::Result<JoinHandle<()>> + Send + Sync>;

fn drop_observer_value<T>(value: T) {
    if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(value))) {
        // The destructor's panic payload may itself panic on drop. Forget that
        // secondary payload so cleanup cannot unwind into compilation.
        std::mem::forget(payload);
        use std::io::Write as _;
        let _ = writeln!(
            std::io::stderr(),
            "x86jit: compiled-code observer cleanup panicked; panic contained"
        );
    }
}

/// The JIT backend. Injected into a `Vm` via `Vm::with_backend` (§4.1) — the core
/// never names this type. Owns the executable-memory arena (`JITModule`) and
/// Cranelift context behind a `Mutex`, so `materialize(&self)` stays `Send + Sync`
/// for a shared `Vm`. With background tier-up it also owns a compiler worker thread
/// and the queues feeding it.
///
/// Constructors are infallible and defer executable storage until compilation.
/// In a PID marked by [`x86jit_core::mark_shared_address_space_child`], a new or
/// inherited backend is inert: compilation refuses before compiler or queue
/// locks, reports [`x86jit_core::CompileFailureReason::OtherBackendError`] to the
/// failure observer, and `Backend` materialization falls back to interpretation.
/// Code storage cannot be allocated, finalized, registered or unmapped there.
/// Observer callbacks still run; the marker does not make inherited callbacks,
/// other backend operations or destructors safe in a child.
pub struct JitBackend {
    shared: Arc<Shared>,
    /// The background compiler thread (bg-tier), spawned lazily on the
    /// first [`tier_up_async`](Backend::tier_up_async) and joined on `Drop`. `None`
    /// until a background tier-up is first requested — eager/sync use never spawns.
    worker: Mutex<Option<JoinHandle<()>>>,
    worker_spawner: WorkerSpawner,
    maintenance: Mutex<Option<JoinHandle<()>>>,
    shutdown_report: ShutdownReport,
    worker_panicked: bool,
}

/// State shared between the vcpu threads (foreground `materialize`, submit, drain)
/// and the background compiler worker (bg-tier). Diagnostic handles share only
/// the separate queue coordination state, never these compiler resources.
struct Shared {
    owner: x86jit_core::codemap::ProcessIdentity,
    system_memory: bool,
    memory: Arc<memory_stats::CodeCounters>,
    lifetime: Arc<x86jit_core::code_lifetime::CodeLifetime>,
    maintenance_stop: AtomicBool,
    collecting: AtomicBool,
    retirement_announced: Mutex<std::collections::HashSet<u64>>,
    lifecycle_observer: Mutex<Option<Observer<CodeLifecycleEvent>>>,
    retired_metadata: Mutex<HashMap<u64, x86jit_core::codemap::RangeRemoval<'static>>>,
    /// The Cranelift module, built lazily on the first compile: its ISA
    /// bakes in `opt_level`, which may have to be derived from a tier-up policy the
    /// `Vm` reports after the backend is constructed. `None` until then.
    inner: Mutex<Option<Jit>>,
    nondefault_inner: Mutex<Option<Jit>>,
    lock_telemetry: Telemetry,
    compile_serial: Mutex<()>,
    compile_attempts: AtomicU64,
    /// Codegen recipe, kept for the deferred build.
    target: HostTarget,
    /// Explicit [`OptLevel`], or `None` to derive it from `tiered` at build time.
    opt: Option<OptLevel>,
    /// Whether the owning `Vm` tiers up, via [`Backend::set_tiering`]. Only read when
    /// the module is built, i.e. on the first compile.
    tiered: AtomicBool,
    /// Per-helper call counts, indexed by the order the `helper!` macro
    /// builds them. Compiled code increments these directly through their baked
    /// addresses, so they must never move: a boxed slice allocated once here.
    /// `helper_names` is filled by the same macro via `stringify!`, so names and
    /// indices cannot drift apart — there is only one list.
    ///
    /// `AtomicU64`, not `u64`: generated code from several vcpus increments the same
    /// address, and this side reads it concurrently. As plain integers that is a host
    /// data race — undefined behaviour in a diagnostic — not just a lossy count.
    /// Generated code uses an atomic add for the same reason; see `call_helper`.
    helper_counters: Box<[AtomicU64]>,
    helper_names: Mutex<Vec<&'static str>>,
    /// Emit the executed-instruction accounting. Off by default: it is one
    /// load/add/store at every guest block boundary, measured at +2.6% to +4.8% on the
    /// block-transfer-heavy bench workloads, which is too much to charge every embedder
    /// for a diagnostic. Read when the module is built, like `tiered`.
    icount: AtomicBool,
    flag_elision: AtomicBool,
    observer: Mutex<Option<CodeObserver>>,
    // The existing publication observer promises backend-lifetime code ranges.
    // Keep that contract until an embedder opts into a lifecycle observer API.
    retain_observed_code: AtomicBool,
    failure_observer: Mutex<Option<FailureObserver>>,
    offsets: CpuOffsets,
    /// Superblock caps (§12 M5-T3), or `None` to compile one block at a time.
    caps: Option<RegionCaps>,
    stopped: AtomicBool,
    background: Arc<CompilerQueue>,
    /// Finished compiles awaiting the core dispatcher's drain.
    done: Mutex<Vec<TierUpFinished>>,
    /// Lock-free "anything to drain?" probe, kept equal to `done.len()` under the
    /// `done` lock — lets `tier_up_finished` early-out without locking.
    ready: AtomicUsize,
    /// Total nanoseconds spent in `compile_with` (every foreground / tier-up / bg
    /// compile), for the bench's compile-vs-run split. Relaxed:
    /// a monotone accumulator, not a synchronization channel.
    compile_ns: AtomicU64,
    time_enabled: AtomicBool,
    background_ns: AtomicU64,
    /// Sticky: a failed module is never finalized or allocated from again.
    code_space_exhausted: AtomicBool,
    /// Registration and the exhaustion broadcast share a lock, so neither a
    /// concurrent nor a later vcpu can miss its notification.
    dispatch_words: Mutex<Vec<Weak<AtomicU64>>>,
    compile_limit: AtomicU64,
    #[cfg(test)]
    fail_compile_stage: AtomicUsize,
    #[cfg(test)]
    fail_worker_spawn: AtomicBool,
}

impl Drop for Shared {
    fn drop(&mut self) {
        let observer = self
            .observer
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        drop_observer_value(observer);
        let observer = self
            .failure_observer
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        drop_observer_value(observer);
        let observer = self
            .lifecycle_observer
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        drop_observer_value(observer);
        if self.owner == x86jit_core::codemap::ProcessIdentity::current()
            && self.lifetime.live_units().is_empty()
            && self.lifetime.retired_units().is_empty()
        {
            for inner in [&mut self.inner, &mut self.nondefault_inner] {
                if let Some(jit) = inner.get_mut().unwrap_or_else(|e| e.into_inner()).take() {
                    // SAFETY: all units and fault readers completed retirement;
                    // no compiler owner remains. Also releases retained system storage.
                    unsafe { jit.module.free_memory() };
                }
            }
        }
    }
}

struct CompileCompletion<'a>(&'a Shared);

impl Drop for CompileCompletion<'_> {
    fn drop(&mut self) {
        let mut queue = self
            .0
            .background
            .queue
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        queue.outstanding -= 1;
        if queue.outstanding == 0 {
            self.0.background.idle_cv.notify_all();
        }
    }
}

struct ForegroundCompile<'a>(&'a CompilerQueue);

impl<'a> ForegroundCompile<'a> {
    fn new(queue: &'a CompilerQueue) -> Self {
        queue
            .queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .foreground += 1;
        Self(queue)
    }
}

impl Drop for ForegroundCompile<'_> {
    fn drop(&mut self) {
        let mut queue = self.0.queue.lock().unwrap_or_else(|e| e.into_inner());
        queue.foreground -= 1;
        if queue.foreground == 0 {
            self.0.work_cv.notify_one();
        }
    }
}

struct QueuedCompile {
    request: TierUpRequest,
    fetch_permit: Option<x86jit_core::FetchPermit>,
}

/// Diagnostics may upgrade this state during destruction without retaining a
/// module, observer or completion. Shutdown empties the queue before counting
/// resource-owning handles; only the inert coordination state can outlive it.
struct CompilerQueue {
    queue: Mutex<Queue>,
    work_cv: Condvar,
    idle_cv: Condvar,
}

/// The background compile queue and its liveness counters (bg-tier).
struct Queue {
    items: VecDeque<QueuedCompile>,
    /// Requests submitted but not yet completed (queued + the active batch).
    /// `wait_idle` blocks until this reaches zero.
    outstanding: usize,
    /// Foreground compiles waiting or holding the compiler lock. The worker parks
    /// between definitions so a queued wave cannot repeatedly overtake a vCPU.
    foreground: usize,
    /// Set by shutdown or `Drop` to unblock and stop the worker.
    shutdown: bool,
    /// Test lever: while true the worker parks without popping, so
    /// requests pile up in the queue. Toggled via `TierUpHandle::pause_compiler`;
    /// `shutdown` still wins so `Drop` never hangs. Never set in production.
    paused: bool,
}

// A definition is not callable until finalization and fault-map registration.
// Keep compiler metadata across clear_context, without exposing an entry pointer.
struct PendingCode {
    id: FuncId,
    code_len: u32,
    srcloc_table: Box<[(u32, u32)]>,
    guest_rips: Box<[u64]>,
    trap_sites: Vec<GuestTrapSite>,
    recipe: Option<x86jit_core::jit_abi::fault::FrameRecipe>,
    instructions: Vec<x86jit_core::fault::InstructionFaultInfo>,
    guest_rip: u64,
    kind: CompiledCodeKind,
}

type Publication<T> = Result<(T, Option<CompiledCode>), CompileError>;

struct Jit {
    allocations: memory_stats::PendingAllocations,
    module: JITModule,
    return_address_bkey: bool,
    fbctx: FunctionBuilderContext,
    next_id: u64,
    reusable_functions: Vec<FuncId>,
    pending_slots: HashMap<FuncId, Vec<Arc<AtomicU64>>>,
    functions: HashMap<u64, FunctionRecord>,
    // Slots for the current definition (§12 M5), transferred to pending_slots
    // and then FunctionRecord. Each `Arc<AtomicU64>` holds a
    // compiled entry pointer (0 = unlinked); its heap address is baked into the
    // code and filled by the dispatcher. Retained through both grace periods.
    // The `Arc` is load-bearing: a bare `Vec` would move its elements on growth,
    // invalidating the addresses already baked into compiled code.
    //
    // `AtomicU64` (not plain `u64`): the dispatcher fill and the SMC-driven clear
    // (`invalidate_links`) both store atomically, so a vcpu reading the slot
    // from compiled code sees 0 or a valid entry, never a torn value. Compiled-code
    // loads are plain machine loads (aligned u64 is naturally atomic on the hosts
    // we target); only the Rust-side writes need the atomic type.
    slots: Vec<Arc<AtomicU64>>,
}

struct FunctionRecord {
    id: FuncId,
    code: CompiledCode,
    slots: Vec<Arc<AtomicU64>>,
}

impl JitBackend {
    /// Poll execution and signal-reader grace without waiting for active users.
    /// Remove completed units' fault metadata and compiler records, and recycle
    /// their function declarations. Acquires the compiler locks. Inherited
    /// backends return zero before touching parent locks. Returns the number of
    /// collected units (at most 64 candidates per call). Dense providers reclaim executable extents when no legacy
    /// code observer has been installed; the system provider retains them. Sealed
    /// anonymous pages remain immutable: only an
    /// entirely free anonymous reservation is released.
    pub fn collect_retired_metadata(&self) -> usize {
        self.shared.collect_retired_metadata()
    }

    /// A JIT whose [`OptLevel`] follows the VM's tier-up policy: the
    /// `Vm` reports it through [`Backend::set_tiering`] and the Cranelift module is
    /// built lazily on the first compile, by which point the policy is known. Pass
    /// an explicit level with [`with_opt_level`](Self::with_opt_level) to override.
    pub fn new() -> Self {
        Self::build(None, HostTarget::Native, None)
    }

    /// Like [`Self::new`], but detect a marked shared-address-space child before
    /// allocating backend state. Infallible constructors remain available and
    /// produce an inert backend in that process.
    pub fn try_new() -> Result<Self, x86jit_core::SharedAddressSpaceChild> {
        x86jit_core::codemap::check_jit_allowed()?;
        Ok(Self::new())
    }

    /// A JIT that forms superblocks (§12 M5-T3): the dispatcher lifts a region and
    /// compiles it as one function, up to `caps`. Opt-in until M5-T3f flips the
    /// default on.
    pub fn with_superblocks(caps: RegionCaps) -> Self {
        Self::build(Some(caps), HostTarget::Native, None)
    }

    /// A JIT at an explicit [`OptLevel`], native host, no superblocks.
    /// An explicit level wins over the tier-up-derived one.
    pub fn with_opt_level(opt: OptLevel) -> Self {
        Self::build(None, HostTarget::Native, Some(opt))
    }

    /// A JIT with every codegen axis chosen explicitly. The three axes are
    /// independent, so the single-axis constructors above cannot express a
    /// combination (superblocks *and* a pinned host target *and* an opt level) —
    /// this one can.
    pub fn with_options(caps: Option<RegionCaps>, target: HostTarget, opt: OptLevel) -> Self {
        Self::build(caps, target, Some(opt))
    }

    /// Compile a block synchronously and report a typed failure instead of
    /// falling back to interpretation. In a marked shared-address-space child,
    /// returns [`x86jit_core::CompileFailureReason::OtherBackendError`] before
    /// taking compiler or queue locks. All failures notify the failure observer.
    pub fn try_materialize(
        &self,
        ir: &IrBlock,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
    ) -> Result<CachedBlock, x86jit_core::CompileFailureReason> {
        self.ensure_maintenance();
        let _time_compile = x86jit_core::time_stats::ForegroundCompileTimer::enter();
        self.shared
            .compile(ir, consistency, mmio, guest_base, 0, false)
            .map(|entry| CachedBlock::Compiled { entry })
            .map_err(CompileError::reason)
    }

    /// Compile a region synchronously with the same refusal contract as
    /// [`Self::try_materialize`].
    pub fn try_materialize_region(
        &self,
        region: &IrRegion,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
    ) -> Result<CachedBlock, x86jit_core::CompileFailureReason> {
        self.ensure_maintenance();
        let _time_compile = x86jit_core::time_stats::ForegroundCompileTimer::enter();
        self.shared
            .compile_region(region, consistency, mmio, guest_base, 0, false)
            .map(|entry| CachedBlock::Compiled { entry })
            .map_err(CompileError::reason)
    }

    /// Use Cranelift's default memory provider for tests and fork-heavy embedders.
    /// Call immediately after construction, before sharing or compiling. Code is
    /// inherited with the system provider's policy; engine workers and VM state
    /// still require the shutdown/rebuild protocol around host fork. Dense arena
    /// mapping bounds and provider storage counters do not apply to this mode.
    /// By default the first compiler module probes and retains a dense code region.
    /// If that first region cannot be created (including unsupported macOS write
    /// protection), Linux/FreeBSD try anonymous W^X storage before selecting this
    /// system provider. macOS selects this system provider directly,
    /// before Cranelift receives any allocation. Once startup selects storage, failures
    /// remain code-space exhaustion; providers are never mixed within a backend.
    /// Automatic fallback and its original operation/errno remain in MemoryStats;
    /// explicit selection here performs no probe and records no fallback.
    /// The first code registration in a fork child replaces the whole inherited
    /// code map, hiding all inherited entries, including still-mapped code from
    /// the system provider. The parent's entries are unaffected.
    pub fn with_system_memory_provider(mut self) -> Self {
        let shared =
            Arc::get_mut(&mut self.shared).expect("configure memory before sharing the backend");
        assert!(
            shared.inner.get_mut().unwrap().is_none()
                && shared.nondefault_inner.get_mut().unwrap().is_none(),
            "configure memory before compiling"
        );
        shared.system_memory = true;
        self
    }

    /// Use code storage without creating any file descriptors.
    /// Configure before sharing or compiling. [`CodeMemoryPolicy::WriteXorExecute`]
    /// permanently seals each publication batch RX and discards its partial-page
    /// tail. [`CodeMemoryPolicy::ReadWriteExecute`] keeps pages writable and
    /// executable, permitting modification of live code in exchange for no
    /// per-batch page waste. Choose it only when that security trade-off is acceptable.
    /// Reservations grow in 128 MiB regions and are excluded from host fork;
    /// shut down and rebuild engines around fork as with the default dense provider.
    /// On Linux/FreeBSD explicit selection never switches providers on failure.
    /// On macOS both policies retain the existing descriptor-free MAP_JIT path,
    /// per-thread write protection, dense tails and startup system fallback.
    pub fn with_no_fd_memory_provider(mut self, policy: CodeMemoryPolicy) -> Self {
        let shared =
            Arc::get_mut(&mut self.shared).expect("configure memory before sharing the backend");
        assert!(
            shared.inner.get_mut().unwrap().is_none()
                && shared.nondefault_inner.get_mut().unwrap().is_none(),
            "configure memory before compiling"
        );
        shared.system_memory = false;
        *shared.memory.provider_choice.lock().unwrap() = if cfg!(target_os = "macos") {
            memory_stats::ProviderChoice::Undecided
        } else {
            memory_stats::ProviderChoice::NoFd(policy)
        };
        self
    }

    /// First dense-provider failure, including failures recovered by fallback.
    /// Does not acquire a compiler lock. None until a provider operation fails.
    pub fn code_first_failure(&self) -> Option<CodeMemoryFailure> {
        self.shared.memory.first_failure()
    }

    /// Most recent dense-provider failure, including failures recovered by fallback.
    pub fn code_last_failure(&self) -> Option<CodeMemoryFailure> {
        self.shared.memory.last_failure()
    }

    /// Choose whether the JIT narrows dead flag writes. Set before compiling.
    pub fn set_flag_elision(&self, enabled: bool) {
        self.shared.flag_elision.store(enabled, Ordering::Relaxed);
    }

    /// Enable or disable compile-lock timing at runtime (off by default).
    /// Each acquisition samples this switch once: disabling does not discard
    /// samples from guards that already started with timing enabled.
    pub fn set_lock_telemetry(&self, enabled: bool) {
        self.shared.lock_telemetry.set_enabled(enabled);
    }

    /// Snapshot completed compile/module lock acquisitions, including errors and unwinds.
    /// Counts include link invalidation even when a module has not been built.
    /// Module wait excludes `compile_serial` wait, which has its own entry.
    /// Active guards and records still being published are not included.
    /// Reading a snapshot never acquires a compile/module mutex or records a sample.
    pub fn lock_telemetry(&self) -> LockTelemetrySnapshot {
        self.shared.lock_telemetry.snapshot()
    }

    /// Clear all three locks' completed samples without changing the enable switch.
    /// Guards still active, waiting, or publishing their sample may report their
    /// entire acquisition after this reset. Quiesce compilation and invalidation
    /// first when a strict interval boundary is needed.
    pub fn reset_lock_telemetry(&self) {
        self.shared.lock_telemetry.reset();
    }

    /// Observe each successfully compiled block or region, including background
    /// tier-up. Called on the compiling thread after code registration and after
    /// releasing the module lock. The callback may re-enter `materialize` or
    /// `invalidate_links`; callbacks on different threads may overlap. Installing an
    /// observer replaces the previous one.
    ///
    /// With unwinding panics, the panic hook reports an observer panic and the
    /// backend catches it, disables that observer, and completes the compilation.
    /// Panicking observers, captured-state destructors, and panic-payload destructors
    /// never stop compilation. Cleanup runs outside backend locks under its own
    /// unwind guard; a secondary panic payload from cleanup is intentionally forgotten.
    /// Already-started notifications may finish. A replacement installed by the callback
    /// is retained. This does not intercept process aborts or double-panic aborts.
    /// A background callback must not call [`TierUpHandle::wait_idle`] on the same
    /// backend: its own compilation is not yet done, so the call never returns.
    ///
    /// Code ranges remain allocated for the backend's lifetime. Installing this
    /// legacy observer permanently pins this backend's executable storage, even
    /// if an observer panic later disables notifications. Metadata collection and
    /// declaration recycling still work. A lifecycle observer with release events
    /// is required before observed executable storage can be reclaimed.
    pub fn set_compiled_code_observer(
        &self,
        observer: impl Fn(CompiledCode) + Send + Sync + 'static,
    ) {
        let observer: CodeObserver = Arc::new(observer);
        self.shared
            .retain_observed_code
            .store(true, Ordering::Release);
        let previous = self.shared.observer.lock().unwrap().replace(observer);
        drop_observer_value(previous);
    }

    /// Observe failed foreground and background compilations, including publication
    /// failures. The event includes the guest entry, unit kind, reason and input size.
    /// Called on the compiling thread outside compiler, queue and observer locks.
    /// Callbacks may overlap and re-enter the backend. The panic containment and
    /// replacement rules are the same as [`Self::set_compiled_code_observer`].
    /// Stale fetch permits are cancellations and produce no failure event.
    ///
    /// A background callback runs before its own compilation is counted as done,
    /// so it must not call [`TierUpHandle::wait_idle`] on the same backend: that
    /// call would wait for the callback itself and never return.
    pub fn set_compile_failure_observer(
        &self,
        observer: impl Fn(CompileFailure) + Send + Sync + 'static,
    ) {
        let observer: FailureObserver = Arc::new(observer);
        let previous = self
            .shared
            .failure_observer
            .lock()
            .unwrap()
            .replace(observer);
        drop_observer_value(previous);
    }

    /// Read legacy diagnostic environment variables explicitly. Requires the
    /// `environment` feature; constructors never read them. Configures flag
    /// elision and, when requested, installs a perf-map observer. Call before
    /// compiling; install a combined observer afterward if other consumers need it.
    #[cfg(feature = "environment")]
    pub fn enable_environment_diagnostics(&self) {
        environment::configure(self);
    }

    /// Limit the number of compilation attempts for this backend's lifetime.
    /// Set during setup, before executing guest code. The default is unlimited.
    /// Reaching the cap has the same terminal behavior as an executable-memory
    /// allocation failure: an `Exit::CodeSpaceExhausted` notification per vcpu and
    /// interpreter fallback for new blocks. Useful for bounding code growth and
    /// testing embedder recovery without exhausting host resources.
    pub fn set_compile_limit(&self, limit: u64) {
        self.shared.compile_limit.store(limit, Ordering::Relaxed);
    }

    /// Per-helper call counts, highest first, zero-count helpers omitted.
    ///
    /// A helper call is a C-ABI exit from compiled code that runs a whole interpreter
    /// operation — tens to hundreds of host cycles against the 1-3 a natively lowered
    /// instruction costs — so a guest whose hot code hits helpers pays a large
    /// per-instruction premium that no mid-end tuning can recover. This says whether
    /// that is happening and, if so, exactly which helpers to lower natively.
    ///
    /// Always on: the counter is one atomic add beside a call that costs orders of
    /// magnitude more. Race-free, so the totals are exact rather than "none, thousands
    /// or millions".
    pub fn helper_calls(&self) -> Vec<(&'static str, u64)> {
        let names = self.shared.helper_names.lock().unwrap();
        let mut out: Vec<(&'static str, u64)> = self
            .shared
            .helper_counters
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .zip(names.iter())
            .filter(|(n, name)| *n > 0 && !name.is_empty())
            .map(|(n, &name)| (name, n))
            .collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        out
    }

    /// Total helper calls across all helpers — the one number that says
    /// whether the helper path matters for a workload at all.
    pub fn helper_calls_total(&self) -> u64 {
        self.shared
            .helper_counters
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .sum()
    }

    /// The [`OptLevel`] this backend will compile at: the explicit one if
    /// the constructor pinned it, otherwise the one derived from the tier-up policy
    /// reported so far. Reading it does not build the module, so the answer can still
    /// change until the first compile if the policy is set later.
    pub fn opt_level(&self) -> OptLevel {
        self.shared
            .opt
            .unwrap_or_else(|| OptLevel::for_tiering(self.shared.tiered.load(Ordering::Relaxed)))
    }

    /// A JIT pinned to a [`HostTarget`]: which *host* instructions Cranelift
    /// may emit for the guest IR — a separate axis from the guest ISA
    /// (`GuestCpuFeatures`). Default is [`HostTarget::Native`] (detect the
    /// running host). Guest-invisible: the emitted code is bit-identical in effect
    /// (only instruction *selection* changes), so interp == JIT holds regardless.
    pub fn with_host_target(target: HostTarget) -> Self {
        Self::build(None, target, None)
    }

    /// Record the codegen recipe. The Cranelift module is NOT built here: `opt`
    /// may be `None`, meaning "derive from the tier-up policy", which the `Vm`
    /// only reports later via [`Backend::set_tiering`]. See [`Shared::jit`].
    fn build(caps: Option<RegionCaps>, target: HostTarget, opt: Option<OptLevel>) -> Self {
        Self {
            shared: Arc::new(Shared {
                owner: x86jit_core::codemap::ProcessIdentity::current(),
                system_memory: false,
                inner: Mutex::new(None),
                nondefault_inner: Mutex::new(None),
                lock_telemetry: Telemetry::default(),
                compile_serial: Mutex::new(()),
                memory: Arc::default(),
                lifetime: Arc::new(x86jit_core::code_lifetime::CodeLifetime::managed()),
                retired_metadata: Mutex::new(HashMap::new()),
                maintenance_stop: AtomicBool::new(false),
                collecting: AtomicBool::new(false),
                retirement_announced: Mutex::default(),
                lifecycle_observer: Mutex::new(None),
                compile_attempts: AtomicU64::new(0),
                target,
                opt,
                tiered: AtomicBool::new(false),
                icount: AtomicBool::new(false),
                flag_elision: AtomicBool::new(true),
                observer: Mutex::new(None),
                retain_observed_code: AtomicBool::new(false),
                failure_observer: Mutex::new(None),
                helper_counters: (0..MAX_HELPERS)
                    .map(|_| AtomicU64::new(0))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                helper_names: Mutex::new(vec![""; MAX_HELPERS]),
                offsets: cpu_offsets(),
                caps,
                stopped: AtomicBool::new(false),
                background: Arc::new(CompilerQueue {
                    queue: Mutex::new(Queue {
                        items: VecDeque::new(),
                        outstanding: 0,
                        foreground: 0,
                        shutdown: false,
                        paused: false,
                    }),
                    work_cv: Condvar::new(),
                    idle_cv: Condvar::new(),
                }),
                done: Mutex::new(Vec::new()),
                ready: AtomicUsize::new(0),
                compile_ns: AtomicU64::new(0),
                time_enabled: AtomicBool::new(false),
                background_ns: AtomicU64::new(0),
                code_space_exhausted: AtomicBool::new(false),
                dispatch_words: Mutex::new(Vec::new()),
                compile_limit: AtomicU64::new(u64::MAX),
                #[cfg(test)]
                fail_compile_stage: AtomicUsize::new(0),
                #[cfg(test)]
                fail_worker_spawn: AtomicBool::new(false),
            }),
            worker: Mutex::new(None),
            maintenance: Mutex::new(None),
            worker_spawner: Box::new(|task| {
                std::thread::Builder::new()
                    .name("x86jit-tier".into())
                    .spawn(task)
            }),
            shutdown_report: ShutdownReport::default(),
            worker_panicked: false,
        }
    }

    /// Emit guest-instruction accounting into compiled code, so
    /// [`Vcpu::executed_instructions`] counts compiled blocks and not only the
    /// interpreter. Off by default — it costs a load/add/store per guest block.
    ///
    /// Must be called BEFORE the first compile: the decision is baked into every
    /// block this backend emits. [`Backend::codegen_description`] reports the state.
    pub fn enable_icount(&self) {
        self.shared.icount.store(true, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn build_jit(target: HostTarget, opt: OptLevel) -> Jit {
        Self::build_jit_with_memory(
            target,
            opt,
            Arc::new(memory_stats::CodeCounters::default()),
            false,
        )
    }

    fn build_jit_with_memory(
        target: HostTarget,
        opt: OptLevel,
        memory: Arc<memory_stats::CodeCounters>,
        system_memory: bool,
    ) -> Jit {
        let mut flags = settings::builder();
        flags.set("use_colocated_libcalls", "false").unwrap();
        flags.set("is_pic", "false").unwrap();
        // Fault recovery unwinds exactly one compiled frame.
        flags.set("preserve_frame_pointers", "true").unwrap();
        flags.set("unwind_info", "true").unwrap();
        flags.set("enable_probestack", "false").unwrap();

        // Set explicitly rather than inheriting Cranelift's default of `none` — see
        // `OptLevel` for why the mid-end passes are worth their compile time on
        // lifted x86.
        flags.set("opt_level", opt.flag()).unwrap();
        // Cranelift's CLIF verifier runs on every compile and only *validates* — it
        // never changes the emitted code — but it costs 20-31% of compile time on
        // every bench workload. Keep it wherever a lowering bug would be
        // introduced (debug + the test suites, which build unoptimized) and drop it
        // from release builds, where the same bug would already have been caught.
        // Without it a malformed-CLIF bug degrades from a clear panic into silently
        // wrong machine code, so this is a debug-build guarantee we are keeping, not
        // a check we are deleting.
        flags
            .set(
                "enable_verifier",
                if cfg!(debug_assertions) {
                    "true"
                } else {
                    "false"
                },
            )
            .unwrap();
        let mut isa_builder = cranelift_native::builder().expect("host ISA");
        if target == HostTarget::Baseline {
            // Pin below the host: forbid AVX and above so codegen is deterministic and
            // portable to any SSE4.1 host (an AOT cache built here runs on older CPUs).
            // Cranelift lowers our 128-bit-lane vector IR to SSE, so guest AVX/AVX-512
            // still executes correctly — just via SSE host instructions. FMA off too:
            // no mul+add contraction, so results stay bit-identical to the interpreter.
            for flag in [
                "has_avx",
                "has_avx512bitalg",
                "has_avx512dq",
                "has_avx512f",
                "has_avx512vbmi",
                "has_avx512vl",
                "has_fma",
            ] {
                let _ = isa_builder.set(flag, "false");
            }
        }
        let isa = isa_builder
            .finish(settings::Flags::new(flags))
            .expect("finish ISA");
        let return_address_bkey = unwind::uses_bkey(isa.as_ref());
        let builder = libcalls::builder_with_memory(
            isa,
            if system_memory {
                code_memory::system_provider(memory)
            } else {
                code_memory::provider(memory)
            },
        );
        let module = JITModule::new(builder);

        Jit {
            allocations: memory_stats::PendingAllocations::default(),
            module,
            return_address_bkey,
            fbctx: FunctionBuilderContext::new(),
            next_id: 0,
            reusable_functions: Vec::new(),
            pending_slots: HashMap::new(),
            functions: HashMap::new(),
            slots: Vec::new(),
        }
    }

    /// Spawn the background compiler thread if it isn't running yet (bg-tier).
    /// Lazy: eager/sync-only use never reaches here, so it never spawns.
    fn submit_tier_up(
        &self,
        req: TierUpRequest,
        fetch_permit: Option<x86jit_core::FetchPermit>,
    ) -> TierUpSubmit {
        self.ensure_maintenance();
        if let Err(error) = check_compile_allowed() {
            self.shared
                .observe_failure(CompileFailure::for_request(&req, error));
            return TierUpSubmit::Unsupported;
        }
        if self.shared.stopped.load(Ordering::Acquire) {
            return TierUpSubmit::Unsupported;
        }
        if self.code_space_exhausted() {
            return TierUpSubmit::Busy;
        }
        if self.ensure_worker().is_err() {
            self.shared.exhaust_code_space::<()>();
            return TierUpSubmit::Busy;
        }
        let mut q = self.shared.background.queue.lock().unwrap();
        if q.shutdown {
            // Racing `Drop` — decline; the caller stays interpreted (correct, slow).
            return TierUpSubmit::Unsupported;
        }
        if q.items.len() >= TIER_QUEUE_CAP {
            // Backpressure: never compile inline in response.
            return TierUpSubmit::Busy;
        }
        q.items.push_back(QueuedCompile {
            request: req,
            fetch_permit,
        });
        q.outstanding += 1;
        drop(q);
        self.shared.background.work_cv.notify_one();
        TierUpSubmit::Queued
    }

    fn ensure_worker(&self) -> std::io::Result<()> {
        let mut w = lock_recovering(&self.worker, "compile worker");
        if self.shared.stopped.load(Ordering::Acquire) {
            return Ok(());
        }
        if w.is_none() {
            let shared = Arc::clone(&self.shared);
            #[cfg(test)]
            let fail = self.shared.fail_worker_spawn.load(Ordering::Relaxed);
            #[cfg(not(test))]
            let fail = false;
            let spawned = if fail {
                Err(std::io::ErrorKind::WouldBlock.into())
            } else {
                (self.worker_spawner)(Box::new(move || shared.worker_loop()))
            };
            *w = Some(spawned?);
        }
        Ok(())
    }

    /// A handle to the background tier-up machinery (bg-tier). Its
    /// [`wait_idle`](TierUpHandle::wait_idle) blocks until every submitted compile
    /// has completed — the determinism lever for tests. Grab it before boxing the
    /// backend into a `Vm`.
    pub fn tier_up_handle(&self) -> TierUpHandle {
        TierUpHandle {
            shared: Arc::downgrade(&self.shared.background),
        }
    }

    /// Install a policy for compiler, maintenance and deferred-reclamation threads
    /// before the first compilation or background submission. The
    /// spawner must return the join handle of the thread executing `task`, or
    /// return an error without starting it. Never detach or retain `task`.
    /// Wrap `task` to configure the worker's signal mask, affinity, priority,
    /// name or thread registration before it runs engine work or callbacks. No library
    /// signal policy is installed. Spawn errors decline background tier-up and
    /// enter the existing code-space-exhausted interpreter fallback.
    pub fn set_worker_spawner(
        &mut self,
        spawner: impl Fn(WorkerTask) -> std::io::Result<JoinHandle<()>> + Send + Sync + 'static,
    ) {
        assert!(self
            .worker
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .is_none());
        assert!(self
            .maintenance
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .is_none());
        assert_eq!(self.shared.compile_attempts.load(Ordering::Relaxed), 0);
        self.worker_spawner = Box::new(spawner);
    }

    fn stop_worker(&mut self) -> Result<ShutdownReport, ShutdownError> {
        self.shared.stopped.store(true, Ordering::Release);
        {
            let mut q = self
                .shared
                .background
                .queue
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            q.shutdown = true;
            let cancelled = q.items.len();
            q.items.clear();
            q.outstanding -= cancelled;
            self.shutdown_report.jobs_cancelled += cancelled;
        }
        self.shared.background.work_cv.notify_all();
        self.shared.background.idle_cv.notify_all();
        if let Some(worker) = self
            .worker
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            if let Err(payload) = worker.join() {
                self.worker_panicked = true;
                drop_observer_value(payload);
            }
            self.shutdown_report.workers_joined += 1;
        }
        {
            let mut q = self
                .shared
                .background
                .queue
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            // A worker panic may have abandoned a request before its completion guard.
            self.shutdown_report.jobs_cancelled += q.outstanding;
            q.outstanding = 0;
        }
        self.shared.background.idle_cv.notify_all();
        let mut done = self.shared.done.lock().unwrap_or_else(|e| e.into_inner());
        for completion in done.iter() {
            self.shared.lifetime.discard(&completion.block);
        }
        done.clear();
        drop(done);
        self.shared.ready.store(0, Ordering::Release);
        self.shared.maintenance_stop.store(true, Ordering::Release);
        self.shared.lifetime.wake_maintenance();
        if let Some(worker) = self
            .maintenance
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            if let Err(payload) = worker.join() {
                self.worker_panicked = true;
                drop_observer_value(payload);
            }
        }
        self.shutdown_report.outstanding_handles = Arc::strong_count(&self.shared) - 1;
        if self.shutdown_report.outstanding_handles != 0 {
            Err(ShutdownError::LiveResources {
                resource: "compiler state",
                report: self.shutdown_report,
            })
        } else if self.worker_panicked {
            Err(ShutdownError::WorkerPanicked {
                report: self.shutdown_report,
            })
        } else {
            Ok(self.shutdown_report)
        }
    }
}

impl Shared {
    fn lock_module(&self, mxcsr_mode: u32) -> lock_telemetry::Guard<'_, Option<Jit>> {
        let (inner, kind) = if mxcsr_mode == 0 {
            (&self.inner, LockKind::DefaultModule)
        } else {
            (&self.nondefault_inner, LockKind::NondefaultModule)
        };
        self.lock_telemetry.lock(inner, kind, "JIT module")
    }

    /// Lock the Cranelift module, building it on first use.
    ///
    /// The ISA bakes in `opt_level`, so the module cannot exist until the level is
    /// decided — and when the level is derived rather than explicit, that needs the
    /// tier-up policy, which the `Vm` only reports after construction. Deferring to
    /// the first compile is what lets `JitBackend::new()` on a tiering `Vm` pick
    /// `Speed` without the embedder asking. A tier-up policy set *after* the first
    /// compile cannot change the level: the module and every block already in it
    /// were built at the old one.
    fn jit(&self, mxcsr_mode: u32) -> impl std::ops::DerefMut<Target = Jit> + '_ {
        struct Guard<'a>(lock_telemetry::Guard<'a, Option<Jit>>);
        impl std::ops::Deref for Guard<'_> {
            type Target = Jit;
            fn deref(&self) -> &Jit {
                self.0.as_ref().expect("module built on lock")
            }
        }
        impl std::ops::DerefMut for Guard<'_> {
            fn deref_mut(&mut self) -> &mut Jit {
                self.0.as_mut().expect("module built on lock")
            }
        }
        let mut guard = self.lock_module(mxcsr_mode);
        if guard.is_none() {
            let opt = self
                .opt
                .unwrap_or_else(|| OptLevel::for_tiering(self.tiered.load(Ordering::Relaxed)));
            *guard = Some(JitBackend::build_jit_with_memory(
                self.target,
                opt,
                self.memory.clone(),
                self.system_memory,
            ));
        }
        Guard(guard)
    }

    fn icount_on(&self) -> bool {
        self.icount.load(Ordering::Relaxed)
    }

    fn compile(
        &self,
        ir: &IrBlock,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
        mxcsr_mode: u32,
        precise_split_stores: bool,
    ) -> Result<CompiledPtr, CompileError> {
        check_compile_allowed()
            .and_then(|()| {
                self.compile_block_with(
                    ir,
                    consistency,
                    mmio,
                    guest_base,
                    mxcsr_mode,
                    precise_split_stores,
                    self.foreground_publication(),
                )
            })
            .inspect_err(|error| {
                self.observe_failure(CompileFailure::for_blocks(
                    ir.guest_start,
                    CompiledCodeKind::Block,
                    *error,
                    [ir],
                ))
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_block_with<T>(
        &self,
        ir: &IrBlock,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
        mxcsr_mode: u32,
        precise_split_stores: bool,
        finish: impl FnOnce(&Self, &mut Jit, PendingCode) -> Publication<T>,
    ) -> Result<T, CompileError> {
        if ir.legacy_context.is_some() {
            return Err(CompileError::BackendError);
        }
        let icount_on = self.icount_on();
        self.define_with_fault_info(
            CompiledCodeKind::Block,
            ir.guest_start,
            mxcsr_mode,
            x86jit_core::fault::InstructionFaultInfo::for_block_with_split_stores(
                ir,
                precise_split_stores,
            ),
            |builder, mut helpers, alloc_slot| {
                helpers.precise_split_stores = precise_split_stores;
                codegen::translate_block(
                    builder,
                    ir,
                    &self.offsets,
                    alloc_slot,
                    helpers,
                    consistency,
                    mmio,
                    guest_base,
                    icount_on,
                )
            },
            finish,
        )
    }

    /// Compile a superblock region (§12 M5-T3) as one function.
    fn compile_region(
        &self,
        region: &IrRegion,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
        mxcsr_mode: u32,
        precise_split_stores: bool,
    ) -> Result<CompiledPtr, CompileError> {
        check_compile_allowed()
            .and_then(|()| {
                self.compile_region_with(
                    region,
                    consistency,
                    mmio,
                    guest_base,
                    mxcsr_mode,
                    precise_split_stores,
                    self.foreground_publication(),
                )
            })
            .inspect_err(|error| {
                self.observe_failure(CompileFailure::for_blocks(
                    region.entry,
                    CompiledCodeKind::Region,
                    *error,
                    &region.blocks,
                ))
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_region_with<T>(
        &self,
        region: &IrRegion,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
        mxcsr_mode: u32,
        precise_split_stores: bool,
        finish: impl FnOnce(&Self, &mut Jit, PendingCode) -> Publication<T>,
    ) -> Result<T, CompileError> {
        if region.blocks.iter().any(|b| b.legacy_context.is_some()) {
            return Err(CompileError::BackendError);
        }
        let icount_on = self.icount_on();
        self.define_with_fault_info(
            CompiledCodeKind::Region,
            region.entry,
            mxcsr_mode,
            region
                .blocks
                .iter()
                .flat_map(|ir| {
                    x86jit_core::fault::InstructionFaultInfo::for_block_with_split_stores(
                        ir,
                        precise_split_stores,
                    )
                })
                .collect(),
            |builder, mut helpers, alloc_slot| {
                helpers.precise_split_stores = precise_split_stores;
                codegen::translate_region(
                    builder,
                    region,
                    &self.offsets,
                    alloc_slot,
                    helpers,
                    consistency,
                    mmio,
                    guest_base,
                    icount_on,
                )
            },
            finish,
        )
    }

    #[cfg(test)]
    fn compile_with(
        &self,
        kind: CompiledCodeKind,
        guest_rip: u64,
        mxcsr_mode: u32,
        translate: impl FnOnce(
            &mut FunctionBuilder,
            codegen::Helpers,
            &mut dyn FnMut() -> u64,
        ) -> Box<[u64]>,
    ) -> Result<CompiledPtr, CompileError> {
        check_compile_allowed()?;
        self.define_with_fault_info(
            kind,
            guest_rip,
            mxcsr_mode,
            Vec::new(),
            translate,
            self.foreground_publication(),
        )
    }

    /// Build a definition and save its metadata before clearing the compiler
    /// context. `finish` either finalizes now or retains it for the worker batch.
    #[allow(clippy::too_many_arguments)]
    fn define_with_fault_info<T>(
        &self,
        // Observer metadata follows the real guest entry, before any source location.
        kind: CompiledCodeKind,
        guest_rip: u64,
        mxcsr_mode: u32,
        instructions: Vec<x86jit_core::fault::InstructionFaultInfo>,
        translate: impl FnOnce(
            &mut FunctionBuilder,
            codegen::Helpers,
            &mut dyn FnMut() -> u64,
        ) -> Box<[u64]>,
        finish: impl FnOnce(&Self, &mut Jit, PendingCode) -> Publication<T>,
    ) -> Result<T, CompileError> {
        check_compile_allowed()?;
        let started = Instant::now();
        if self.stopped.load(Ordering::Acquire) || self.code_space_exhausted.load(Ordering::Acquire)
        {
            return Err(CompileError::Unavailable);
        }
        let _serial = self.lock_telemetry.lock(
            &self.compile_serial,
            LockKind::CompileSerial,
            "JIT compilation",
        );
        let mut module_guard = self.jit(mxcsr_mode);
        let mut jit = reclamation::Definition::new(&mut module_guard);
        // Serialized with every foreground/background compile. After an error the
        // module may contain unfinalized definitions; never retry or publish those.
        if self.code_space_exhausted.load(Ordering::Acquire) {
            return Err(CompileError::Unavailable);
        }
        if self.compile_attempts.load(Ordering::Relaxed)
            >= self.compile_limit.load(Ordering::Relaxed)
        {
            return self.exhaust_code_space().ok_or(CompileError::Unavailable);
        }
        self.compile_attempts.fetch_add(1, Ordering::Relaxed);
        jit.next_id += 1;
        let name = format!("blk_{}", jit.next_id);

        let mut ctx = jit.module.make_context();
        let ptr = jit.module.target_config().pointer_type();
        ctx.func.signature.params.push(AbiParam::new(ptr));
        ctx.func.signature.params.push(AbiParam::new(ptr));
        ctx.func.signature.returns.push(AbiParam::new(types::I64));

        // The Rust helpers, reached from compiled code by `call_indirect` through
        // their baked absolute address rather than a linker-relocated direct call —
        // so the emitted machine code carries no relocations (the prerequisite for a
        // persistable AOT code cache; see backlog/docs/design/aot-plan.md). Build each
        // signature here; `import_signature` + the fn address are wired into
        // `Helpers` below, inside the builder scope.
        // `n` I64 params, plus an I64 return when `ret` (the value-returning helpers);
        // the void helpers (cpuid/vmaskmov/bmi write through a `cpu`/`out` pointer).
        let params = |n: usize, ret: bool| {
            let mut s = jit.module.make_signature();
            for _ in 0..n {
                s.params.push(AbiParam::new(types::I64));
            }
            if ret {
                s.returns.push(AbiParam::new(types::I64));
            }
            s
        };
        let div_sig = params(6, true); // div(hi, lo, divisor, size, signed, out) -> i64
        let cas_pair_sig = params(6, true);
        let str_sig = params(8, true); // string(cpu, mem, op, elem, rep, cur_addr, addr_bits, seg_base) -> i64
        let x87_sig = params(6, true); // x87(cpu, mem, kind, addr, sti, cur_addr) -> status + private FDP policy
        let xstate_sig = params(6, true);
        let fx_sig = params(5, true); // fxstate(cpu, mem, addr, restore, cur_addr) -> i64
        let crc_sig = params(3, true); // crc32(crc, src, bytes) -> i64
        let note_watch_sig = params(3, false); // note_watch(mem_self, addr, len) -> ()
        let cpuid_sig = params(1, false); // cpuid(cpu) -> ()
        let tsc_sig = params(2, false); // tsc(cpu, aux) -> ()
        let random_sig = params(4, true); // random(cpu, seed, size, flags) -> i64
        let xgetbv_sig = params(3, true); // xgetbv(cpu, mem, cur_addr) -> status
        let vmaskmov_sig = params(7, false); // vmaskmov(cpu, dst, src, k, elem, zeroing, bytes) -> ()
        let vmaskmov_mem_sig = params(11, true); // (cpu, mem, reg, addr, k, elem, zeroing, bytes, is_store, cur_addr, broadcast_elem) -> i64
        let gather_sig = params(13, true);
        let vec_maskmov_mem_sig = params(9, true); // (cpu, mem, reg, addr, mask, elem, bytes, is_store, cur_addr) -> i64
        let vmasked_logic_sig = params(9, false); // (cpu, op, dst, a, b, k, elem, zeroing, bytes) -> ()
        let vmasked_packed_sig = params(9, false); // (cpu, op, dst, a, b, k, elem, zeroing, bytes) -> ()
        let vmasked_shift_sig = params(10, false); // (cpu, dst, a, imm, elem, right, arith, k, zeroing, bytes) -> ()
        let var_shift_sig = params(10, false); // (cpu, dst, a, count, elem, right, arith, k, zeroing, bytes) -> ()
        let shift_reg_sig = params(10, false); // (cpu, dst, a, count, elem, right, arith, k, zeroing, bytes) -> ()
        let gf2p8_sig = params(9, false); // (cpu, dst, a, b, imm, mode, k, zeroing, bytes) -> ()
        let gf2p8_mem_sig = params(11, true); // (cpu, mem, dst, a, addr, imm, mode, k, zeroing, bytes, cur_addr) -> ret
        let valign_sig = params(7, false); // valign(cpu, dst, a, b, shift, elem, bytes) -> ()
        let vpermt2_sig = params(10, false); // (cpu, dst, idx, tbl, elem, k, masked, zeroing, bytes, imode) -> ()
        let vpermt2_mem_sig = params(12, true); // (cpu, mem, dst, idx, addr, elem, k, masked, zeroing, bytes, imode, cur_addr) -> ret
        let vperm1_sig = params(9, false); // (cpu, dst, idx, src, elem, bytes, k, masked, zeroing) -> ()
        let vperm1_mem_sig = params(11, true); // (cpu, mem, dst, idx, addr, elem, k, masked, zeroing, bytes, cur_addr) -> ret
        let vpmov_narrow_sig = params(9, false); // (cpu, dst, src, from, to, src_width, k, masked, zeroing) -> ()
        let vpmov_narrow_mem_sig = params(8, true); // (cpu, mem, src, addr, from, to, src_width, cur_addr) -> ret
        let vpmov_extend_wide_sig = params(10, false); // (cpu, dst, src, from, to, signed, dst_width, k, masked, zeroing) -> ()
        let vpabs_sig = params(8, false); // (cpu, dst, src, elem, dst_width, k, masked, zeroing) -> ()
        let vp_unary_lane_sig = params(10, false); // (cpu, dst, src, op, imm, elem, dst_width, k, masked, zeroing) -> ()
        let vp_blendm_sig = params(8, false); // (cpu, dst, a, b, k, elem, dst_width, zeroing) -> ()
        let vshuf_lane_sig = params(10, false); // (cpu, dst, a, b, imm, elem, dst_width, k, masked, zeroing) -> ()
        let vp_multishift_sig = params(8, false); // (cpu, dst, ctrl, data, dst_width, k, masked, zeroing) -> ()
        let vpshufb_wide_sig = params(8, false); // (cpu, dst, a, idx, bytes, k, masked, zeroing) -> ()
        let vshuffle32_wide_sig = params(8, false); // (cpu, dst, a, imm, bytes, k, masked, zeroing) -> ()
        let vpack_sig = params(7, false); // (cpu, dst, a, b, from_elem, signed, bytes) -> ()
        let vpack_mem_sig = params(8, false); // (cpu, dst, a, lo, hi, from_elem, signed, lane) -> ()
        let vhfloat_sig = params(7, false); // (cpu, dst, a, b, op, f64, bytes) -> ()
        let vhfloat_mem_sig = params(10, false); // (cpu, dst, a, lo0, hi0, lo1, hi1, op, f64, bytes) -> ()
        let vhint_sig = params(6, false); // (cpu, dst, a, b, op, bytes) -> ()
        let vhint_mem_sig = params(9, false); // (cpu, dst, a, lo0, hi0, lo1, hi1, op, bytes) -> ()
        let cvtph2ps_sig = params(4, false); // (cpu, dst, src, lanes) -> ()
        let var_shift_mem_sig = params(9, false);
        let vstore_preflight_sig = params(3, true);
        let cvtps2ph_value_sig = params(5, false);
        let cvtps2ph_sig = params(5, false); // (cpu, dst, src, lanes, rc) -> ()
        let phminposuw_sig = params(3, false); // (cpu, dst, src) -> ()
        let mpsadbw_sig = params(7, false); // (cpu, dst, a, lo, hi, imm, half) -> ()
        let pmaddwd_sig = params(4, false); // (cpu, dst, a, b) -> ()
        let fma_sig = params(14, false); // (cpu, dst, x, y, z, prec_f64, scalar, neg_prod, neg_add, bytes, alt_sign, k, masked, zeroing) -> ()
        let fma_mem_sig = params(18, true); // (cpu, mem, dst, x, y, z, base, mem_role, prec_f64, scalar, neg_prod, neg_add, bytes, alt_sign, cur_addr, k, masked, zeroing) -> ret
        let vpmadd_sig = params(6, false); // (cpu, dst, a, b, ubsw, bytes) -> ()
        let vpmadd_mem_sig = params(7, false); // (cpu, dst, a, lo, hi, ubsw, hi_half) -> ()
        let broadcast_lane_sig = params(9, false); // (cpu,dst,src,chunk,elem,dst_width,k,masked,zeroing)
        let broadcast_lane_mem_sig = params(11, true); // (cpu,mem,dst,base,chunk,elem,dst_width,k,masked,zeroing,cur)
        let aes_sig = params(6, false); // aes(cpu, dst, a, b, op, imm) -> ()
        let aes_mem_sig = params(7, false); // aes_mem(cpu, dst, a, lo, hi, op, imm) -> ()
        let sha_sig = params(6, false); // sha(cpu, dst, a, b, op, imm) -> ()
        let sha_mem_sig = params(7, false); // sha_mem(cpu, dst, a, lo, hi, op, imm) -> ()
        let gfni_sig = params(6, false); // gfni(cpu, dst, a, b, op, imm) -> ()
        let gfni_mem_sig = params(7, false); // gfni_mem(cpu, dst, a, lo, hi, op, imm) -> ()
        let pclmul_sig = params(5, false); // pclmul(cpu, dst, a, b, imm) -> ()
        let pclmul_mem_sig = params(6, false); // pclmul_mem(cpu, dst, a, lo, hi, imm) -> ()
        let pcmpstr_sig = params(6, false); // pcmpstr(cpu, a, b, imm, explicit, out) -> ()
        let pcmpstr_mem_sig = params(7, false); // pcmpstr_mem(cpu, a, bv_lo, bv_hi, imm, explicit, out) -> ()
        let pcmpstrm_sig = params(6, false); // pcmpstrm(cpu, a, b, imm, explicit, out) -> ()
        let pcmpstrm_mem_sig = params(7, false); // pcmpstrm_mem(cpu, a, bv_lo, bv_hi, imm, explicit, out) -> ()
        let bmi_sig = params(5, false); // bmi(a, b, op, size, out) -> () — result + CF via `out`

        let fp_signatures = (mxcsr_mode != 0).then(|| (params(1, false), params(8, false)));
        let slots_start = jit.slots.len();
        let guest_rips = {
            let frontend_config = jit.module.isa().frontend_config();
            let Jit { fbctx, slots, .. } = &mut *jit;
            let mut alloc_slot = || {
                let b = Arc::new(AtomicU64::new(0));
                let addr = Arc::as_ptr(&b) as u64;
                self.lifetime.register_slot(&b);
                slots.push(b);
                addr
            };
            let mut builder = FunctionBuilder::new(&mut ctx.func, fbctx);
            // Each helper is `(imported signature ref, baked fn address)`.
            // Each helper also gets a call counter: its index is the macro
            // expansion order, and `stringify!` records the name at the same index, so
            // the counters, the names and the helper table are one list by
            // construction. Compiled code increments the counter through its baked
            // address, so the address must be stable — hence the boxed slice in
            // `Shared`, allocated once.
            let mut hidx = 0usize;
            let mut hnames = self.helper_names.lock().unwrap();
            macro_rules! helper {
                ($sig:expr, $f:expr) => {{
                    assert!(hidx < MAX_HELPERS, "raise MAX_HELPERS");
                    let counter = &self.helper_counters[hidx] as *const AtomicU64 as u64;
                    hnames[hidx] = stringify!($f);
                    #[allow(unused_assignments)] // the last expansion's bump is never read
                    {
                        hidx += 1;
                    }
                    (
                        {
                            let mut sig = $sig;
                            sig.params.push(AbiParam::new(types::I64));
                            builder.import_signature(sig)
                        },
                        $f as *const u8 as u64,
                        counter,
                    )
                }};
            }
            let (switch_fp, fp_boundary) =
                if let Some((switch_sig, fp_boundary_sig)) = fp_signatures {
                    (
                        (
                            builder.import_signature(switch_sig),
                            x86jit_core::jit_abi::fp::set_control as *const u8 as u64,
                        ),
                        helper!(fp_boundary_sig, fp_boundary_helper),
                    )
                } else {
                    // Reserve the counter index in both modules, without importing
                    // signatures that default translations cannot call.
                    hnames[hidx] = "fp_boundary_helper";
                    hidx += 1;
                    (
                        (ir::SigRef::from_u32(0), 0),
                        (ir::SigRef::from_u32(0), 0, 0),
                    )
                };
            let afp = mxcsr_mode != 0
                && self.target != HostTarget::Baseline
                && x86jit_core::jit_abi::fp::has_afp();
            let helpers = codegen::Helpers {
                precise_split_stores: false,
                mxcsr_mode,
                afp,
                fp_control: if mxcsr_mode == 0 {
                    x86jit_core::jit_abi::fp::default_control()
                } else {
                    x86jit_core::jit_abi::fp::control_with_afp(mxcsr_mode, afp)
                },
                switch_fp,
                fp_boundary,
                flag_elision: self.flag_elision.load(Ordering::Relaxed),
                div: helper!(div_sig, div_helper),
                string: helper!(str_sig, string_helper::<true>),
                cas_pair: helper!(cas_pair_sig, cas_pair_helper),
                cpuid: helper!(cpuid_sig, cpuid_helper),
                tsc: helper!(tsc_sig, tsc_helper),
                random: helper!(random_sig, random_helper),
                xgetbv: helper!(xgetbv_sig, xgetbv_helper),
                vmaskmov: helper!(vmaskmov_sig, vmaskmov_helper),
                vmaskmov_mem: helper!(vmaskmov_mem_sig, vmaskmov_mem_helper),
                gather: helper!(gather_sig, gather_helper),
                vec_maskmov_mem: helper!(vec_maskmov_mem_sig, vec_maskmov_mem_helper),
                vmasked_logic: helper!(vmasked_logic_sig, vmasked_logic_helper),
                vmasked_packed: helper!(vmasked_packed_sig, vmasked_packed_helper),
                vmasked_shift: helper!(vmasked_shift_sig, vmasked_shift_helper),
                var_shift: helper!(var_shift_sig, var_shift_helper),
                shift_reg: helper!(shift_reg_sig, shift_reg_helper),
                gf2p8: helper!(gf2p8_sig, gf2p8_helper),
                gf2p8_mem: helper!(gf2p8_mem_sig, gf2p8_mem_helper),
                valign: helper!(valign_sig, valign_helper),
                vpermt2: helper!(vpermt2_sig, vpermt2_helper),
                vpermt2_mem: helper!(vpermt2_mem_sig, vpermt2_mem_helper),
                vperm1: helper!(vperm1_sig, vperm1_helper),
                vperm1_mem: helper!(vperm1_mem_sig, vperm1_mem_helper),
                vpmov_narrow: helper!(vpmov_narrow_sig, vpmov_narrow_helper),
                vpmov_narrow_mem: helper!(vpmov_narrow_mem_sig, vpmov_narrow_mem_helper),
                vpmov_extend_wide: helper!(vpmov_extend_wide_sig, vpmov_extend_wide_helper),
                vpabs: helper!(vpabs_sig, vpabs_helper),
                vp_unary_lane: helper!(vp_unary_lane_sig, vp_unary_lane_helper),
                vp_blendm: helper!(vp_blendm_sig, vp_blendm_helper),
                vshuf_lane: helper!(vshuf_lane_sig, vshuf_lane_helper),
                vp_multishift: helper!(vp_multishift_sig, vp_multishift_helper),
                vpshufb_wide: helper!(vpshufb_wide_sig, vpshufb_wide_helper),
                vshuffle32_wide: helper!(vshuffle32_wide_sig, vshuffle32_wide_helper),
                vpack: helper!(vpack_sig, vpack_helper),
                var_shift_mem: helper!(var_shift_mem_sig, var_shift_mem_helper),
                vstore_preflight: helper!(vstore_preflight_sig, vstore_preflight_helper),
                cvtps2ph_value: helper!(cvtps2ph_value_sig, cvtps2ph_value_helper),
                vpack_mem: helper!(vpack_mem_sig, vpack_mem_helper),
                vhfloat: helper!(vhfloat_sig, vhfloat_helper),
                vhfloat_mem: helper!(vhfloat_mem_sig, vhfloat_mem_helper),
                vhint: helper!(vhint_sig, vhint_helper),
                vhint_mem: helper!(vhint_mem_sig, vhint_mem_helper),
                cvtph2ps: helper!(cvtph2ps_sig, cvtph2ps_helper),
                cvtps2ph: helper!(cvtps2ph_sig, cvtps2ph_helper),
                phminposuw: helper!(phminposuw_sig, phminposuw_helper),
                mpsadbw: helper!(mpsadbw_sig, mpsadbw_helper),
                pmaddwd: helper!(pmaddwd_sig, pmaddwd_helper),
                vpmadd: helper!(vpmadd_sig, vpmadd_helper),
                vpmadd_mem: helper!(vpmadd_mem_sig, vpmadd_mem_helper),
                fma: helper!(fma_sig, fma_helper),
                fma_mem: helper!(fma_mem_sig, fma_mem_helper),
                broadcast_lane: helper!(broadcast_lane_sig, broadcast_lane_helper),
                broadcast_lane_mem: helper!(broadcast_lane_mem_sig, broadcast_lane_mem_helper),
                aes: helper!(aes_sig, aes_helper),
                aes_mem: helper!(aes_mem_sig, aes_mem_helper),
                sha: helper!(sha_sig, sha_helper),
                sha_mem: helper!(sha_mem_sig, sha_mem_helper),
                gfni: helper!(gfni_sig, gfni_helper),
                gfni_mem: helper!(gfni_mem_sig, gfni_mem_helper),
                pclmul: helper!(pclmul_sig, pclmul_helper),
                pclmul_mem: helper!(pclmul_mem_sig, pclmul_mem_helper),
                pcmpstr: helper!(pcmpstr_sig, pcmpstr_helper),
                pcmpstr_mem: helper!(pcmpstr_mem_sig, pcmpstr_mem_helper),
                pcmpstrm: helper!(pcmpstrm_sig, pcmpstrm_helper),
                pcmpstrm_mem: helper!(pcmpstrm_mem_sig, pcmpstrm_mem_helper),
                bmi: helper!(bmi_sig, bmi_helper),
                x87: helper!(x87_sig.clone(), x87_helper::<false>),
                x87_mem: helper!(x87_sig, x87_helper::<true>),
                fxstate: helper!(fx_sig, fxstate_helper),
                xstate: helper!(xstate_sig, xstate_helper),
                crc32: helper!(crc_sig, crc32_helper),
                note_watch: helper!(note_watch_sig, note_write_helper),
            };
            let guest_rips = translate(&mut builder, helpers, &mut alloc_slot);
            builder.finalize(frontend_config);
            guest_rips
        };

        if mxcsr_mode != 0 {
            codegen::preserve_float_environment(&mut ctx.func);
        }
        let trap_metadata = match prepare_guest_traps(&mut ctx.func, &guest_rips) {
            Ok(metadata) => metadata,
            Err(error) => {
                jit.slots.truncate(slots_start);
                return Err(error);
            }
        };
        let id = match jit.reusable_functions.pop() {
            Some(id) => id,
            None => jit
                .module
                .declare_function(&name, Linkage::Export, &ctx.func.signature)
                .expect("declare function"),
        };
        jit.id = Some(id);
        #[cfg(test)]
        if self.fail_compile_stage.load(Ordering::Relaxed) == 1 {
            self.compile_result(Err(ModuleError::Allocation {
                err: std::io::ErrorKind::OutOfMemory.into(),
            }))
            .ok_or(CompileError::Unavailable)?;
        }
        self.compile_result(jit.module.define_function(id, &mut ctx))
            .ok_or(CompileError::Unavailable)?;
        let slots = jit.slots.split_off(slots_start);
        assert!(jit.pending_slots.insert(id, slots).is_none());
        let cc = ctx.compiled_code().expect("compiled code present");
        jit.allocations.define(cc.code_info().total_size as usize);
        // Capture the code size + sorted `(host_off, rip_index)` srcloc table before
        // `clear_context` wipes it, to register in the process-global `CodeMap` once
        // the host entry address is known below.
        let (code_len, srcloc_table, trap_sites, recipe) = {
            let cc = ctx.compiled_code().expect("compiled code present");
            let trap_sites = collect_guest_traps(cc, &trap_metadata);
            #[cfg(test)]
            let trap_sites = if self.fail_compile_stage.load(Ordering::Relaxed) == 4 {
                let mut sites = Vec::<GuestTrapSite>::new();
                sites
                    .try_reserve_exact(usize::MAX)
                    .map(|()| sites)
                    .map_err(|_| CompileError::Unavailable)
            } else {
                trap_sites
            };
            let trap_sites = match trap_sites {
                Ok(sites) => sites,
                Err(CompileError::Unavailable) => {
                    return self.exhaust_code_space().ok_or(CompileError::Unavailable)
                }
                Err(error) => return Err(error),
            };
            let table: Vec<(u32, u32)> = cc
                .buffer
                .get_srclocs_sorted()
                .iter()
                .filter(|s| !s.loc.is_default())
                .map(|s| (s.start, s.loc.bits()))
                .collect();
            (
                cc.code_info().total_size,
                table.into_boxed_slice(),
                trap_sites,
                unwind::recipe(
                    cc,
                    jit.return_address_bkey,
                    ctx.func.signature.call_conv,
                    jit.module.isa().default_call_conv(),
                ),
            )
        };
        jit.module.clear_context(&mut ctx);
        let (result, observed) = finish(
            self,
            &mut jit,
            PendingCode {
                id,
                code_len,
                srcloc_table,
                guest_rips,
                trap_sites,
                recipe,
                instructions,
                guest_rip,
                kind,
            },
        )?;
        jit.keep = true;
        drop(jit);
        drop(module_guard);
        drop(_serial);
        if let Some(code) = observed {
            self.observe(code);
        }
        self.compile_ns
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(result)
    }

    fn finalize_module(&self, jit: &mut Jit) -> Result<(), CompileError> {
        check_compile_allowed()?;
        if self.code_space_exhausted.load(Ordering::Acquire) {
            return Err(CompileError::Unavailable);
        }
        #[cfg(test)]
        if self.fail_compile_stage.load(Ordering::Relaxed) == 2 {
            self.compile_result(Err(ModuleError::Backend(
                region::Error::SystemCall(std::io::ErrorKind::OutOfMemory.into()).into(),
            )))
            .ok_or(CompileError::Unavailable)?;
        }
        let result = jit.module.finalize_definitions();
        jit.allocations.finalize(&self.memory, result.is_ok());
        self.compile_result(result).ok_or(CompileError::Unavailable)
    }

    fn foreground_publication(
        &self,
    ) -> impl FnOnce(&Self, &mut Jit, PendingCode) -> Publication<CompiledPtr> + '_ {
        let foreground = ForegroundCompile::new(&self.background);
        move |shared, jit, code| {
            let result = shared.finalize_one(jit, code);
            // Observer callbacks run after publication and may wait for the worker.
            drop(foreground);
            result
        }
    }

    fn finalize_one(&self, jit: &mut Jit, code: PendingCode) -> Publication<CompiledPtr> {
        self.finalize_module(jit)?;
        let entry = Self::finalized_entry(jit, &code);
        self.publish(entry, code)
            .map(|(entry, observed)| (entry, Some(observed)))
    }

    fn finalized_entry(jit: &mut Jit, code: &PendingCode) -> CompiledPtr {
        let address = jit.module.get_finalized_function(code.id);
        let unit = match jit.module.get_finalized_function_generation(code.id) {
            Some(generation) => x86jit_core::code_lifetime::CodeUnit {
                generation,
                host_start: address as usize,
                host_len: code.code_len,
            },
            // The system provider retains code and has no per-extent identity.
            None => x86jit_core::code_lifetime::CodeUnit::new(address as usize, code.code_len),
        };
        let slots = jit
            .pending_slots
            .remove(&code.id)
            .expect("defined function slots");
        assert!(jit
            .functions
            .insert(
                unit.generation,
                FunctionRecord {
                    id: code.id,
                    code: CompiledCode {
                        host_start: address as usize,
                        host_len: code.code_len,
                        guest_rip: code.guest_rip,
                        kind: code.kind,
                    },
                    slots,
                }
            )
            .is_none());
        unit.entry()
    }

    // Caller has finalized this definition. Register metadata before exposing its entry.
    fn publish(
        &self,
        entry: CompiledPtr,
        code: PendingCode,
    ) -> Result<(CompiledPtr, CompiledCode), CompileError> {
        check_compile_allowed()?;
        let PendingCode {
            id: _,
            code_len,
            srcloc_table,
            guest_rips,
            trap_sites,
            recipe,
            instructions,
            guest_rip,
            kind,
        } = code;
        #[cfg(test)]
        let fail_registration = self.fail_compile_stage.load(Ordering::Relaxed) == 3;
        #[cfg(not(test))]
        let fail_registration = false;
        if fail_registration
            || x86jit_core::codemap::register_with_fault_path(
                entry.0 as usize,
                code_len,
                srcloc_table,
                guest_rips,
                trap_sites,
                recipe,
                instructions,
                match kind {
                    CompiledCodeKind::Block => x86jit_core::codemap::FaultUnit::Block,
                    CompiledCodeKind::Region => x86jit_core::codemap::FaultUnit::Region {
                        entry_rip: guest_rip,
                    },
                },
            )
            .is_err()
        {
            return self.exhaust_code_space().ok_or(CompileError::Unavailable);
        }
        let entry = if entry.generation() == 0 {
            self.lifetime.register(entry, code_len).entry()
        } else {
            let unit = x86jit_core::code_lifetime::CodeUnit {
                generation: entry.generation(),
                host_start: entry.0 as usize,
                host_len: code_len,
            };
            // SAFETY: finalized_entry obtained this fresh provider-owned identity;
            // each PendingCode is consumed exactly once, after registration.
            unsafe { self.lifetime.register_unit(unit) };
            entry
        };
        Ok((
            entry,
            CompiledCode {
                host_start: entry.0 as usize,
                host_len: code_len,
                guest_rip,
                kind,
            },
        ))
    }

    fn observe(&self, code: CompiledCode) {
        if let Some(unit) = self.lifetime.unit_at(code.host_start) {
            Self::notify_observer(
                &self.lifecycle_observer,
                CodeLifecycleEvent {
                    unit,
                    code,
                    kind: CodeLifecycleKind::Publish,
                },
            );
        }
        Self::notify_observer(&self.observer, code);
    }

    fn observe_failure(&self, event: CompileFailure) {
        Self::notify_observer(&self.failure_observer, event);
    }

    fn notify_observer<T>(slot: &Mutex<Option<Observer<T>>>, event: T) {
        let observer = slot.lock().unwrap().clone();
        if let Some(observer) = observer {
            if let Err(payload) =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| observer(event)))
            {
                let disabled = {
                    let mut current = slot.lock().unwrap();
                    if current
                        .as_ref()
                        .is_some_and(|value| Arc::ptr_eq(value, &observer))
                    {
                        current.take()
                    } else {
                        None
                    }
                };
                drop_observer_value(disabled);
                drop_observer_value(payload);
                use std::io::Write as _;
                let _ = writeln!(
                    std::io::stderr(),
                    "x86jit: compiler observer panicked; that observer is disabled"
                );
            }
            drop_observer_value(observer);
        }
    }

    fn exhaust_code_space<T>(&self) -> Option<T> {
        let mut words = self.dispatch_words.lock().unwrap();
        if !self.code_space_exhausted.swap(true, Ordering::AcqRel) {
            words.retain(|weak| {
                if let Some(word) = weak.upgrade() {
                    word.fetch_or(x86jit_core::vm::CODE_SPACE_EXHAUSTED, Ordering::Release);
                    true
                } else {
                    false
                }
            });
        }
        None
    }

    fn compile_result(&self, result: Result<(), ModuleError>) -> Option<()> {
        match result {
            Ok(()) => Some(()),
            Err(ModuleError::Allocation { .. }) => self.exhaust_code_space(),
            // ENOMEM and EAGAIN from OS finalization/protection are resource
            // exhaustion. Other backend errors retain their diagnostic panic.
            Err(ModuleError::Backend(error))
                if error
                    .downcast_ref::<std::io::Error>()
                    .or_else(|| match error.downcast_ref::<region::Error>() {
                        // region does not expose the inner error through source().
                        Some(region::Error::SystemCall(error)) => Some(error),
                        _ => None,
                    })
                    .is_some_and(|e| {
                        matches!(
                            e.kind(),
                            std::io::ErrorKind::OutOfMemory | std::io::ErrorKind::WouldBlock
                        )
                    }) =>
            {
                self.exhaust_code_space()
            }
            Err(error) => panic!("compile: {error}"),
        }
    }

    /// Compile one background request's unit (bg-tier): a single block, or a
    /// hotness-gated superblock region — same off-thread path either way.
    fn define_request(
        &self,
        req: &TierUpRequest,
        permit: Option<&x86jit_core::FetchPermit>,
    ) -> Result<PendingCode, CompileError> {
        check_compile_allowed()?;
        if permit.is_some_and(|p| !p.is_current()) {
            return Err(CompileError::Unavailable);
        }
        #[cfg(test)]
        if self.fail_compile_stage.load(Ordering::Relaxed) == 5 {
            self.compile_result(Err(ModuleError::Backend(
                std::io::Error::from_raw_os_error(libc::EACCES).into(),
            )));
        }
        let defer = |_: &Self, _: &mut Jit, code| Ok((code, None));
        match &req.unit {
            TierUpUnit::Block(ir) => self.compile_block_with(
                ir,
                req.consistency,
                req.mmio,
                req.guest_base,
                req.mxcsr_mode,
                req.precise_split_stores,
                defer,
            ),
            TierUpUnit::Region(region) => self.compile_region_with(
                region,
                req.consistency,
                req.mmio,
                req.guest_base,
                req.mxcsr_mode,
                req.precise_split_stores,
                defer,
            ),
        }
    }

    fn interpreted_request(req: &TierUpRequest) -> CachedBlock {
        match &req.unit {
            TierUpUnit::Block(ir) => CachedBlock::Interpreted(ir.clone()),
            TierUpUnit::Region(region) => {
                CachedBlock::Interpreted(Arc::new(region.blocks[0].clone()))
            }
        }
    }

    /// Drain only work already queued, at most TIER_FINALIZE_BATCH definitions.
    /// Release compilation locks between definitions: a foreground compile can
    /// finalize immediately (including any pending definitions in its module).
    /// Writes use the provider's write scope and RW view. Dense tails remain
    /// available after finalization; publication follows cache synchronization.
    fn worker_loop(&self) {
        loop {
            let mut batch = Vec::with_capacity(TIER_FINALIZE_BATCH);
            {
                let mut q = self.background.queue.lock().unwrap();
                while (q.items.is_empty() || q.paused || q.foreground != 0) && !q.shutdown {
                    q = self.background.work_cv.wait(q).unwrap();
                }
                if q.shutdown && q.items.is_empty() {
                    break;
                }
            }
            let time_batch = self
                .time_enabled
                .load(Ordering::Relaxed)
                .then(x86jit_core::time_stats::MonotonicTime::now);
            loop {
                let queued = {
                    let mut q = self.background.queue.lock().unwrap();
                    if q.shutdown || q.paused || q.foreground != 0 {
                        None
                    } else {
                        q.items.pop_front()
                    }
                };
                let Some(queued) = queued else { break };
                let completion = CompileCompletion(self);
                let code = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.define_request(&queued.request, queued.fetch_permit.as_ref())
                })) {
                    Ok(code) => code,
                    Err(payload) => {
                        drop_observer_value(payload);
                        Err(CompileError::Panicked)
                    }
                };
                let block = Self::interpreted_request(&queued.request);
                batch.push((queued, code, block, completion));
                if batch.len() == TIER_FINALIZE_BATCH {
                    break;
                }
            }
            // Finalize each used MXCSR module once, never exposing unsealed code.
            // If any resource operation failed, the sticky exhaustion state rejects
            // all remaining definitions, exactly as on the synchronous path.
            let started = Instant::now();
            let mut finalized_entries = Vec::with_capacity(batch.len());
            let mut panic_payloads = Vec::new();
            {
                let _serial = self.lock_telemetry.lock(
                    &self.compile_serial,
                    LockKind::CompileSerial,
                    "JIT batch publication",
                );
                for nondefault in [false, true] {
                    if !batch.iter().any(|(queued, code, _, _)| {
                        (queued.request.mxcsr_mode != 0) == nondefault && code.is_ok()
                    }) {
                        continue;
                    }
                    let mut jit = self.jit(u32::from(nondefault));
                    let finalized =
                        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            self.finalize_module(&mut jit)
                        })) {
                            Ok(result) => result,
                            Err(payload) => {
                                panic_payloads.push(payload);
                                Err(CompileError::Panicked)
                            }
                        };
                    for (index, (queued, code, _, _)) in batch.iter_mut().enumerate() {
                        if (queued.request.mxcsr_mode != 0) != nondefault {
                            continue;
                        }
                        if code.is_ok() {
                            let pending = std::mem::replace(
                                code,
                                Err(finalized.err().unwrap_or(CompileError::Unavailable)),
                            )
                            .unwrap();
                            if finalized.is_ok()
                                && !self.code_space_exhausted.load(Ordering::Acquire)
                            {
                                let entry = Self::finalized_entry(&mut jit, &pending);
                                finalized_entries.push((index, entry, pending));
                            } else {
                                reclamation::discard_definition(&mut jit, pending.id);
                            }
                        }
                    }
                }
            }
            for payload in panic_payloads {
                drop_observer_value(payload);
            }
            // Registration can allocate O(log n) codemap nodes per definition;
            // keep that batch work outside the compiler locks so foreground
            // compilation waits only for a definition or module finalization.
            for (index, entry, code) in finalized_entries {
                if !self.code_space_exhausted.load(Ordering::Acquire) {
                    match self.publish(entry, code) {
                        Ok((entry, event)) => {
                            batch[index].2 = CachedBlock::Compiled { entry };
                            self.observe(event);
                        }
                        Err(error) => {
                            batch[index].1 = Err(error);
                            self.discard_unpublished_entry(entry);
                        }
                    }
                } else {
                    self.discard_unpublished_entry(entry);
                }
            }
            self.compile_ns
                .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            for (queued, code, block, _) in &batch {
                if matches!(block, CachedBlock::Interpreted(_))
                    && queued.fetch_permit.as_ref().is_none_or(|p| p.is_current())
                {
                    self.observe_failure(CompileFailure::for_request(
                        &queued.request,
                        *code.as_ref().err().expect("failed compile has a reason"),
                    ));
                }
            }
            let mut done = self.done.lock().unwrap();
            for (queued, code, block, _) in &mut batch {
                let req = &mut queued.request;
                let block = std::mem::replace(block, Self::interpreted_request(req));
                done.push(TierUpFinished {
                    pc: req.pc,
                    mxcsr_mode: req.mxcsr_mode,
                    compile_failure: if matches!(block, CachedBlock::Interpreted(_))
                        && queued.fetch_permit.as_ref().is_none_or(|p| p.is_current())
                    {
                        code.as_ref().err().copied().map(CompileError::reason)
                    } else {
                        None
                    },
                    block,
                    spans: std::mem::take(&mut req.spans),
                    epoch: req.epoch,
                });
            }
            self.ready.store(done.len(), Ordering::Release);
            drop(done);
            if let Some(start) = time_batch {
                let ns = start.elapsed_ns();
                self.background_ns
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                        Some(old.saturating_add(ns))
                    })
                    .ok();
            }
            drop(batch); // completion guards notify wait_idle only after publication
        }
    }
}

/// A cloneable handle to a [`JitBackend`]'s background tier-up machinery (bg-tier),
/// exposing `wait_idle` for deterministic tests without owning the worker thread.
/// Operations retain only queue coordination, never compiler resources, and become
/// inert after backend destruction.
#[derive(Clone)]
pub struct TierUpHandle {
    shared: Weak<CompilerQueue>,
}

impl TierUpHandle {
    /// Block until every submitted background compile has completed or has been
    /// cancelled by shutdown. Before shutdown, results are queued for
    /// `Backend::tier_up_finished`. Returns immediately after backend destruction.
    /// No sleeps — waits on the worker's idle signal.
    pub fn wait_idle(&self) {
        let Some(shared) = self.shared.upgrade() else {
            return;
        };
        let mut q = shared.queue.lock().unwrap_or_else(|e| e.into_inner());
        while q.outstanding > 0 {
            q = shared.idle_cv.wait(q).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Test lever: park the background worker so queued requests pile
    /// up uncompiled until the returned guard drops — lets a race test line up several
    /// in-flight requests for one pc before any of them lands. Sets a queue flag (it
    /// does NOT hold the compiler mutex, so a vcpu can still invalidate/`materialize`
    /// on the same thread). Not for production use.
    pub fn pause_compiler(&self) -> CompilerPause {
        if let Some(shared) = self.shared.upgrade() {
            shared
                .queue
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .paused = true;
        }
        CompilerPause {
            shared: self.shared.clone(),
        }
    }
}

/// Guard from [`TierUpHandle::pause_compiler`]: releasing it unparks the worker.
pub struct CompilerPause {
    shared: Weak<CompilerQueue>,
}

impl Drop for CompilerPause {
    fn drop(&mut self) {
        if let Some(shared) = self.shared.upgrade() {
            shared
                .queue
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .paused = false;
            shared.work_cv.notify_all();
        }
    }
}

impl Default for JitBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for JitBackend {
    fn time_stats(&self) -> x86jit_core::BackendTimeStats {
        let mut stats = x86jit_core::BackendTimeStats::default();
        stats.background_compile_ns = self.shared.background_ns.load(Ordering::Relaxed);
        stats
    }

    fn set_time_accounting(&self, enabled: bool) {
        self.shared.time_enabled.store(enabled, Ordering::Relaxed);
    }

    fn memory_stats(&self) -> x86jit_core::BackendMemoryStats {
        let mut stats = self.shared.memory.snapshot();
        let (live, retired, live_units, retired_units) = self.shared.lifetime.unit_stats();
        stats.code_bytes_live = live;
        stats.code_bytes_retired = retired;
        stats.code_units_live = live_units;
        stats.code_units_retired = retired_units;
        stats.code_ibtc_descriptors = self.shared.lifetime.ibtc_descriptor_count() as u64;
        let queue = self
            .shared
            .background
            .queue
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        stats.background_queue_len = queue.items.len() as u64;
        stats.background_in_flight = (queue.outstanding - queue.items.len()) as u64;
        stats
    }

    fn shutdown(&mut self) -> Result<ShutdownReport, ShutdownError> {
        let report = self.stop_worker();
        // SAFETY: submissions stopped and the compiler joined. Closing the
        // lifetime domain rejects subsequent cache resolution/publication;
        // direct ABI callers must stop using this backend before shutdown.
        unsafe { self.shared.lifetime.retire_all() };
        while self.shared.collect_retired_metadata() != 0 {}
        if self.shared.lifetime.has_pending_reclamation() {
            let mut counts = report.as_ref().copied().unwrap_or(self.shutdown_report);
            counts.outstanding_handles += 1;
            return Err(ShutdownError::LiveResources {
                resource: "code reclamation",
                report: counts,
            });
        }
        report
    }
    fn register_dispatch_word(&self, word: &Arc<AtomicU64>) {
        let mut words = self.shared.dispatch_words.lock().unwrap();
        words.retain(|weak| weak.strong_count() != 0);
        if self.code_space_exhausted() {
            word.fetch_or(x86jit_core::vm::CODE_SPACE_EXHAUSTED, Ordering::Release);
        }
        words.push(Arc::downgrade(word));
    }

    fn code_space_exhausted(&self) -> bool {
        self.shared.code_space_exhausted.load(Ordering::Acquire)
    }

    fn materialize(
        &self,
        ir: &IrBlock,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
    ) -> CachedBlock {
        self.try_materialize(ir, consistency, mmio, guest_base)
            .unwrap_or_else(|_| CachedBlock::Interpreted(Arc::new(ir.clone())))
    }

    fn materialize_with_mxcsr(
        &self,
        ir: &IrBlock,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
        mxcsr_mode: u32,
    ) -> CachedBlock {
        self.ensure_maintenance();
        let _time_compile = x86jit_core::time_stats::ForegroundCompileTimer::enter();
        self.shared
            .compile(ir, consistency, mmio, guest_base, mxcsr_mode, false)
            .map_or_else(
                |_| CachedBlock::Interpreted(Arc::new(ir.clone())),
                |entry| CachedBlock::Compiled { entry },
            )
    }
    fn materialize_region_with_mxcsr(
        &self,
        region: &IrRegion,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
        mxcsr_mode: u32,
    ) -> CachedBlock {
        self.ensure_maintenance();
        let _time_compile = x86jit_core::time_stats::ForegroundCompileTimer::enter();
        self.shared
            .compile_region(region, consistency, mmio, guest_base, mxcsr_mode, false)
            .map_or_else(
                |_| CachedBlock::Interpreted(Arc::new(region.blocks[0].clone())),
                |entry| CachedBlock::Compiled { entry },
            )
    }

    fn materialize_with_split_stores(
        &self,
        ir: &IrBlock,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
        mxcsr_mode: u32,
        precise_split_stores: bool,
    ) -> CachedBlock {
        self.ensure_maintenance();
        let _time_compile = x86jit_core::time_stats::ForegroundCompileTimer::enter();
        self.shared
            .compile(
                ir,
                consistency,
                mmio,
                guest_base,
                mxcsr_mode,
                precise_split_stores,
            )
            .map_or_else(
                |_| CachedBlock::Interpreted(Arc::new(ir.clone())),
                |entry| CachedBlock::Compiled { entry },
            )
    }

    fn materialize_region_with_split_stores(
        &self,
        region: &IrRegion,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
        mxcsr_mode: u32,
        precise_split_stores: bool,
    ) -> CachedBlock {
        self.ensure_maintenance();
        let _time_compile = x86jit_core::time_stats::ForegroundCompileTimer::enter();
        self.shared
            .compile_region(
                region,
                consistency,
                mmio,
                guest_base,
                mxcsr_mode,
                precise_split_stores,
            )
            .map_or_else(
                |_| CachedBlock::Interpreted(Arc::new(region.blocks[0].clone())),
                |entry| CachedBlock::Compiled { entry },
            )
    }

    fn region_caps(&self) -> Option<RegionCaps> {
        self.shared.caps.filter(|_| !self.code_space_exhausted())
    }

    fn materialize_region(
        &self,
        region: &IrRegion,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
    ) -> CachedBlock {
        self.try_materialize_region(region, consistency, mmio, guest_base)
            .unwrap_or_else(|_| CachedBlock::Interpreted(Arc::new(region.blocks[0].clone())))
    }

    fn invalidate_links(&self) {
        // Zero every link slot so no surviving block chains into a unit an SMC
        // invalidation just dropped. Over-invalidation (all slots, not only
        // the victims') is deliberate: invalidation is rare, and a cleared slot
        // simply re-links via `RET_LINK` on its next traversal. Relaxed stores pair
        // with the dispatcher's relaxed fill; compiled-code reads see 0 or a valid
        // entry. Runs under the compiler mutex, off the hot path.
        // Do not use `Shared::jit`: nothing has been compiled if the module was
        // never built, so there are no slots to clear and no reason to build it here.
        for mode in [0, 1] {
            let jit = self.shared.lock_module(mode);
            if let Some(jit) = jit.as_ref() {
                for slot in &jit.slots {
                    slot.store(0, Ordering::Relaxed);
                }
                for slots in jit
                    .pending_slots
                    .values()
                    .chain(jit.functions.values().map(|f| &f.slots))
                {
                    for slot in slots {
                        slot.store(0, Ordering::Relaxed);
                    }
                }
            }
        }
    }

    fn code_lifetime(&self) -> Option<Arc<x86jit_core::code_lifetime::CodeLifetime>> {
        Some(self.shared.lifetime.clone())
    }

    fn tier_up_async(&self, req: TierUpRequest) -> TierUpSubmit {
        self.submit_tier_up(req, None)
    }

    fn tier_up_async_with_fetch_permit(
        &self,
        req: TierUpRequest,
        permit: x86jit_core::FetchPermit,
    ) -> TierUpSubmit {
        self.submit_tier_up(req, Some(permit))
    }

    fn tier_up_finished(&self) -> Vec<TierUpFinished> {
        // Fast path: nothing published since the last drain — no lock, no alloc.
        if self.shared.ready.load(Ordering::Acquire) == 0 {
            return Vec::new();
        }
        let mut done = self.shared.done.lock().unwrap();
        self.shared.ready.store(0, Ordering::Release);
        std::mem::take(&mut *done)
    }

    fn compile_ns(&self) -> u64 {
        self.shared.compile_ns.load(Ordering::Relaxed)
    }

    /// Record the VM's tier-up policy so the lazily-built ISA can derive its
    /// `opt_level` from it. Ignored once the module exists — its blocks
    /// were already compiled at the old level — and ignored entirely when the level
    /// was pinned explicitly at construction.
    fn set_tiering(&self, tiered: bool) {
        self.shared.tiered.store(tiered, Ordering::Relaxed);
    }

    fn helper_calls(&self) -> Vec<(&'static str, u64)> {
        JitBackend::helper_calls(self)
    }

    /// Reachable through `Box<dyn Backend>`, unlike [`JitBackend::opt_level`] — an
    /// embedder that boxes its backend into a `Vm` still needs to confirm which level
    /// the tier-up policy resolved to.
    fn codegen_description(&self) -> String {
        format!(
            "cranelift opt_level={:?} host={:?} superblocks={} verifier={} icount={} preserve_frame_pointers=true flag_elision={}",
            self.opt_level(),
            self.shared.target,
            self.shared.caps.is_some(),
            cfg!(debug_assertions),
            self.shared.icount.load(Ordering::Relaxed),
            self.shared.flag_elision.load(Ordering::Relaxed),
        )
    }
}

impl Drop for JitBackend {
    fn drop(&mut self) {
        let _ = self.stop_worker();
        // SAFETY: closing the domain excludes future compiled cache resolution
        // and publication; submissions stopped and the worker joined. ABI callers must keep
        // the backend alive through calls, chains and cold fault decoding.
        unsafe { self.shared.lifetime.retire_all() };
        self.defer_reclamation();
    }
}

/// Rare normal/subnormal boundary recomputation under default host control.
/// SDM Vol 1 §11.5.2.5 specifies tininess with an unbounded exponent.
///
/// # Safety
/// `out` points at a writable 16-byte JIT stack slot.
unsafe extern "C" fn fp_boundary_helper(
    alo: u64,
    ahi: u64,
    blo: u64,
    bhi: u64,
    op: u64,
    double: u64,
    mode: u64,
    out: *mut u128,
    time_enabled: u64,
) {
    account_helper!(
        time_enabled,
        fp_boundary_helper(alo, ahi, blo, bhi, op, double, mode, out, 0)
    );
    use x86jit_core::interp::softfloat;
    let a = u128::from(alo) | (u128::from(ahi) << 64);
    let b = u128::from(blo) | (u128::from(bhi) << 64);
    let bits = if double != 0 { 64 } else { 32 };
    let mut result = 0;
    if op == 4 {
        for i in 0..2 {
            let r = softfloat::convert((a >> (i * 64)) as u64, true, false, mode as u32);
            result |= u128::from(r) << (i * 32);
        }
        out.write_unaligned(result);
        return;
    }
    for i in 0..128 / bits {
        let x = (a >> (i * bits)) as u64;
        let y = (b >> (i * bits)) as u64;
        let r = if op >= 5 {
            softfloat::unary(
                x,
                false,
                if op == 5 {
                    x86jit_core::FloatUnOp::Rcp
                } else {
                    x86jit_core::FloatUnOp::Rsqrt
                },
                mode as u32,
            )
        } else {
            softfloat::binary(
                x,
                y,
                double != 0,
                match op {
                    0 => x86jit_core::FloatBinOp::Add,
                    1 => x86jit_core::FloatBinOp::Sub,
                    2 => x86jit_core::FloatBinOp::Mul,
                    _ => x86jit_core::FloatBinOp::Div,
                },
                mode as u32,
            )
        };
        result |= u128::from(r) << (i * bits);
    }
    out.write_unaligned(result);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_copy_fault_invalidates_compiled_prefix_only() {
        use x86jit_core::interp::StrMem;
        use x86jit_core::{BlockKey, CpuMode, Exit, HostRam, Prot, Reg, RegionKind, Vm, VmConfig};
        // SAFETY: sysconf has no pointer arguments.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
        for path in ["fp", "string", "preflighted_string"] {
            for commits in [false, true] {
                let len = page * 4;
                let layout = std::alloc::Layout::from_size_align(len, page).unwrap();
                // SAFETY: nonzero layout, owned by HostRam's destructor below.
                let base = unsafe { std::alloc::alloc_zeroed(layout) };
                if base.is_null() {
                    std::alloc::handle_alloc_error(layout);
                }
                let ram = HostRam {
                    ptr: base,
                    len,
                    guest_base: 0,
                    protect: None,
                    dtor: Box::new(move |ptr, _| {
                        // SAFETY: ptr and layout are the allocation transferred above.
                        unsafe { std::alloc::dealloc(ptr, layout) };
                    }),
                };
                let mut vm = x86jit_core::with_barrier_unavailable(|| {
                    Vm::with_backend_host_ram(
                        VmConfig::flat(len as u64),
                        Box::new(JitBackend::new()),
                        ram,
                    )
                    .unwrap()
                });
                vm.map(0, len, Prot::RWX, RegionKind::Ram).unwrap();
                vm.mem.set_host_fault_recovery(true);
                vm.set_tier_up_after(None);
                let second = (page * 2) as u64;
                let first = second - if path == "fp" { 16 } else { 4 };
                vm.write_bytes(first, &[0xb0, 1, 0x90, 0xf4]).unwrap();
                vm.write_bytes(second, &[0xf4]).unwrap();
                let mut cpu = vm.new_vcpu();
                for entry in [first, second] {
                    cpu.set_reg(Reg::Rip, entry);
                    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
                }
                let key = BlockKey::new(first, CpuMode::Long64);
                let suffix_key = BlockKey::new(second, CpuMode::Long64);
                let Some(CachedBlock::Compiled { entry: before }) = vm.cache.get(key) else {
                    panic!("first block must be compiled")
                };
                let Some(CachedBlock::Compiled { entry: suffix }) = vm.cache.get(suffix_key) else {
                    panic!("suffix must be compiled")
                };
                let fault = if commits { second } else { first };
                let failed = x86jit_core::with_copy_out_fault(
                    base as usize + first as usize..base as usize + page * 3,
                    base as usize + fault as usize,
                    || {
                        if path == "fp" {
                            let mem = NotifyingFpMem {
                                raw: RawFpMem {
                                    base,
                                    size: len as u64,
                                    guest_base: 0,
                                    recover_host_faults: true,
                                },
                                memory: &vm.mem,
                            };
                            let mut bytes = [0x90; 32];
                            bytes[..4].copy_from_slice(&[0xb0, 2, 0x90, 0xf4]);
                            mem.store_checked(first, &bytes) == Err(fault)
                        } else {
                            let mem = NotifyingStrMem {
                                raw: x86jit_core::interp::RawStrMem {
                                    base,
                                    size: len as u64,
                                    guest_base: 0,
                                    recover_host_faults: true,
                                },
                                memory: &vm.mem,
                            };
                            let value =
                                u64::from_le_bytes([0xb0, 2, 0x90, 0xf4, 0x90, 0x90, 0x90, 0x90]);
                            let result = if path == "string" {
                                mem.sstore(first, value, 8)
                            } else {
                                mem.sstore_preflighted(first, value, 8)
                            };
                            result == Err(x86jit_core::MemTrap::HostFault)
                        }
                    },
                );
                assert!(failed, "{path}");
                assert_eq!(vm.mem.has_dirty_code(), commits, "{path}");
                cpu.set_reg(Reg::Rip, first);
                assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
                assert_eq!(cpu.reg(Reg::Rax), if commits { 2 } else { 1 }, "{path}");
                let Some(CachedBlock::Compiled { entry: after }) = vm.cache.get(key) else {
                    panic!("first block must be compiled")
                };
                assert_eq!(before.0 == after.0, !commits, "{path}");
                let Some(CachedBlock::Compiled { entry: after }) = vm.cache.get(suffix_key) else {
                    panic!("suffix must stay compiled")
                };
                assert_eq!(
                    suffix.0, after.0,
                    "{path}: untouched suffix was invalidated"
                );
            }
        }
    }

    #[test]
    fn fma_scalar_helper_carriers_use_element_width() {
        // SDM Vol 2C, VFMADD132SS/SD, Operation: m32/m64. Exercise the helper
        // ABI's scalar branches directly, as scalar lifting can inline the load.
        use x86jit_core::jit_abi::{MemCtx, RET_UNMAPPED};
        use x86jit_core::{CpuMode, CpuState, Exit, Memory, MemoryModel};
        let mem = Memory::new(MemoryModel::Flat { size: 0x8000 });
        for (f64, width) in [(0, 4), (1, 8)] {
            for tail in [width, width / 2] {
                let base = (1u64 << 47) - tail;
                let mut cpu = CpuState::default();
                let mut ctx = MemCtx::for_memory(&mem);
                // SAFETY: both ABI pointers refer to live state; the address
                // misses the bounded guest backing before any host access.
                let ret = unsafe {
                    fma_mem_helper(
                        (&mut cpu as *mut CpuState).cast(),
                        (&mut ctx as *mut MemCtx).cast(),
                        0,
                        0,
                        1,
                        0,
                        base,
                        2,
                        f64,
                        1,
                        0,
                        0,
                        16,
                        0,
                        0x1234,
                        0,
                        0,
                        0,
                        0,
                    )
                };
                assert_eq!(ret, RET_UNMAPPED);
                assert_eq!(ctx.helper_operand_addr, base);
                assert_eq!(ctx.helper_operand_size, width);
                let exit = ctx.unmapped_exit(&cpu, CpuMode::Long64);
                if tail == width {
                    assert!(matches!(exit, Exit::UnmappedMemory { addr, .. } if addr == base));
                } else {
                    assert!(matches!(
                        exit,
                        Exit::Exception {
                            addr: 0x1234,
                            vector: 13,
                            ..
                        }
                    ));
                }
            }
        }
    }

    use x86jit_core::jit_abi::run_compiled;
    use x86jit_core::lift::{lift_block, CpuMode};
    use x86jit_core::{CpuState, Memory, MemoryModel, Prot, RegionKind, StepResult};

    // Pause inside a real compiled sub-block, after its watched scalar store.
    // Taking the hook makes later loop iterations run without another handshake.
    thread_local! {
        static STORE_NOTICES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
        static SMC_PAUSE: std::cell::RefCell<Option<(
            std::sync::mpsc::SyncSender<()>, std::sync::mpsc::Receiver<()>
        )>> = const { std::cell::RefCell::new(None) };
    }

    unsafe extern "C" fn paused_store(mem: u64, addr: u64, len: u64, time_enabled: u64) {
        // SAFETY: the generated store passes the same live Memory pointer and
        // checked range as the normal note_write helper.
        unsafe { note_write_helper(mem, addr, len, time_enabled) };
        STORE_NOTICES.with(|count| count.set(count.get() + 1));
        SMC_PAUSE.with(|slot| {
            if let Some((entered, resume)) = slot.borrow_mut().take() {
                let _ = entered.send(());
                let _ = resume.recv();
            }
        });
    }

    struct PausedStoreBackend {
        jit: JitBackend,
        regions: bool,
    }

    impl Backend for PausedStoreBackend {
        fn code_lifetime(
            &self,
        ) -> Option<std::sync::Arc<x86jit_core::code_lifetime::CodeLifetime>> {
            self.jit.code_lifetime()
        }

        fn materialize(
            &self,
            ir: &IrBlock,
            consistency: MemConsistency,
            mmio: Option<(u64, u64)>,
            guest_base: u64,
        ) -> CachedBlock {
            CachedBlock::Compiled {
                entry: self
                    .jit
                    .shared
                    .compile_with(
                        CompiledCodeKind::Block,
                        ir.guest_start,
                        0,
                        |builder, mut helpers, alloc| {
                            helpers.note_watch.1 = paused_store as *const () as u64;
                            codegen::translate_block(
                                builder,
                                ir,
                                &self.jit.shared.offsets,
                                alloc,
                                helpers,
                                consistency,
                                mmio,
                                guest_base,
                                false,
                            )
                        },
                    )
                    .expect("test compile"),
            }
        }

        fn region_caps(&self) -> Option<x86jit_core::RegionCaps> {
            self.regions.then_some(x86jit_core::RegionCaps {
                max_blocks: 8,
                max_icount: 128,
            })
        }

        fn materialize_region(
            &self,
            region: &IrRegion,
            consistency: MemConsistency,
            mmio: Option<(u64, u64)>,
            guest_base: u64,
        ) -> CachedBlock {
            CachedBlock::Compiled {
                entry: self
                    .jit
                    .shared
                    .compile_with(
                        CompiledCodeKind::Region,
                        region.entry,
                        0,
                        |builder, mut helpers, alloc| {
                            helpers.note_watch.1 = paused_store as *const () as u64;
                            codegen::translate_region(
                                builder,
                                region,
                                &self.jit.shared.offsets,
                                alloc,
                                helpers,
                                consistency,
                                mmio,
                                guest_base,
                                false,
                            )
                        },
                    )
                    .expect("test compile"),
            }
        }

        fn invalidate_links(&self) {
            self.jit.invalidate_links();
        }
    }

    fn compiled_helper_patches_cached_translation() {
        use iced_x86::code_asm::*;
        use x86jit_core::{Exit, Reg, Vm, VmConfig};
        for kind in ["vmaskmovps", "kstore", "vpmovqd"] {
            let mut vm = Vm::with_backend(VmConfig::flat(0x10000), Box::new(JitBackend::new()));
            vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.set_guest_cpu_features(x86jit_core::GuestCpuFeatures::v4());
            let mut asm = CodeAssembler::new(64).unwrap();
            // SDM Vol 2C, VMASKMOV/VMOVDQU32/VPMOVQD, Operation:
            // selected dwords commit their low 32 bits to the destination.
            match kind {
                "vmaskmovps" => asm.vmaskmovps(ymmword_ptr(rdi), ymm1, ymm0).unwrap(),
                "kstore" => asm.vmovdqu32(ymmword_ptr(rdi).k1(), ymm0).unwrap(),
                "vpmovqd" => asm.vpmovqd(qword_ptr(rdi), xmm0).unwrap(),
                _ => unreachable!(),
            }
            asm.hlt().unwrap();
            vm.write_bytes(ENTRY, &asm.assemble(ENTRY).unwrap())
                .unwrap();
            let target = 0x4000;
            vm.write_bytes(target, &[0xb8, 1, 0, 0, 0, 0xf4]).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, target);
            assert!(matches!(cpu.run(&vm, Some(16)), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rax), 1);
            assert_eq!(vm.memory_stats().cache.compiled_entries, 1);
            // Only the immediate dword is selected by the masks. Narrowing
            // also rewrites the following dword, preserving the HLT byte.
            cpu.set_xmm(0, 2 | (0xf4 << 64));
            cpu.set_xmm(1, 1 << 31);
            cpu.cpu.kmask[1] = 1;
            cpu.set_reg(Reg::Rdi, target + 1);
            cpu.set_reg(Reg::Rip, ENTRY);
            assert!(matches!(cpu.run(&vm, Some(16)), Exit::Hlt));
            assert!(vm.memory_stats().backend.compiled_units_total >= 2);
            cpu.set_reg(Reg::Rip, target);
            assert!(matches!(cpu.run(&vm, Some(16)), Exit::Hlt), "{kind}");
            assert_eq!(cpu.reg(Reg::Rax), 2, "{kind}");
        }
    }

    #[test]
    fn compiled_masked_and_narrow_stores_notify_in_process_mode() {
        // macOS has no host process barrier. Model successful commands for this
        // single-threaded notifier witness; it makes no remote-ordering claim.
        x86jit_core::with_barrier_success(compiled_helper_patches_cached_translation);
    }

    #[test]
    fn compiled_masked_and_narrow_stores_notify_in_degraded_mode() {
        x86jit_core::with_barrier_unavailable(compiled_helper_patches_cached_translation);
    }

    #[test]
    fn unavailable_barrier_runs_jit_and_notifies_untagged_stores() {
        use x86jit_core::{Exit, Reg, SourceBarrierMode, Vm, VmConfig};
        for regions in [false, true] {
            let mut vm = x86jit_core::with_barrier_unavailable(|| {
                Vm::with_backend(
                    VmConfig::flat(0x80000),
                    Box::new(PausedStoreBackend {
                        jit: JitBackend::new(),
                        regions,
                    }),
                )
            });
            vm.map(0, 0x80000, Prot::RWX, RegionKind::Ram).unwrap();
            // mov [0x40000],42; jmp next; next: mov eax,[0x40000]; hlt.
            // The data is outside the code watermark and has never been watched.
            vm.write_bytes(
                ENTRY,
                &[
                    0xc7, 0x04, 0x25, 0, 0, 4, 0, 42, 0, 0, 0, 0xeb, 0, 0x8b, 0x04, 0x25, 0, 0, 4,
                    0, 0xf4,
                ],
            )
            .unwrap();
            assert!(!vm.mem.is_code_page(0x40));
            STORE_NOTICES.with(|count| count.set(0));
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, ENTRY);
            assert!(matches!(cpu.run(&vm, Some(32)), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rax), 42);
            assert!(
                vm.memory_stats().cache.compiled_entries + vm.memory_stats().cache.region_entries
                    > 0
            );
            STORE_NOTICES.with(|count| assert_eq!(count.get(), 1));
            assert!(!vm.mem.is_code_page(0x40));
            let stats = vm.memory_stats();
            assert_eq!(stats.source_barrier_mode, SourceBarrierMode::Degraded);
            assert_eq!(stats.source_barrier_failures, 1);
            assert_eq!(stats.source_barrier_last_errno, Some(1));
        }
    }

    fn smc_notice_survives_other_vcpu_drain(regions: bool) {
        use std::sync::mpsc;
        use std::time::Duration;
        use x86jit_core::{Exit, Reg, Vm, VmConfig};

        // first: inc rax; mov [0x8000],rax; jmp second
        // second: add rbx,1; cmp rax,3; jb first; hlt
        let code = [
            0x48, 0xff, 0xc0, 0x48, 0x89, 0x04, 0x25, 0x00, 0x80, 0x00, 0x00, 0xeb, 0x00, 0x48,
            0x83, 0xc3, 0x01, 0x48, 0x83, 0xf8, 0x03, 0x72, 0xe9, 0xf4,
        ];
        let mut vm = Vm::with_backend(
            VmConfig::flat(0x10000),
            Box::new(PausedStoreBackend {
                jit: JitBackend::new(),
                regions,
            }),
        );
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(ENTRY, &code).unwrap();
        vm.write_bytes(0x4000, &[0xf4]).unwrap();
        vm.watch_range(0x8000, 8);
        if !regions {
            // Byte-precise invalidation needs an actual cached target. Without
            // this warmup, only first is cached and the write is victim-free.
            let mut warm = vm.new_vcpu();
            warm.set_reg(Reg::Rip, ENTRY + 13);
            warm.set_reg(Reg::Rax, 3);
            assert!(matches!(warm.run(&vm, Some(32)), Exit::Hlt));
        }
        let vm = Arc::new(vm);
        let mut a = vm.new_vcpu();
        a.set_reg(Reg::Rip, ENTRY);
        let mut b = vm.new_vcpu();
        b.set_reg(Reg::Rip, 0x4000);
        let (entered_tx, entered_rx) = mpsc::sync_channel(0);
        let (resume_tx, resume_rx) = mpsc::sync_channel(0);
        let running_vm = Arc::clone(&vm);
        let worker = std::thread::spawn(move || {
            SMC_PAUSE.with(|slot| *slot.borrow_mut() = Some((entered_tx, resume_rx)));
            // A finite fallback bounds a broken gate without an exit request that
            // could hide the missing SMC notice.
            let exit = a.run(&running_vm, Some(32));
            (exit, a)
        });
        entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        if regions {
            assert!(vm.cache.regions() > 0, "A must already be inside a region");
        }
        let epoch = vm.cache.epoch();
        // A is inside first; change second's increment before its next gate.
        vm.mem.write(ENTRY + 16, 10, 1).unwrap();
        assert!(matches!(b.run(&vm, None), Exit::Hlt));
        assert!(!vm.mem.has_dirty_code(), "B must drain the SMC page list");
        assert!(vm.cache.epoch() > epoch, "B must publish invalidation");
        resume_tx.send(()).unwrap();
        let (exit, a) = worker.join().unwrap();
        assert!(matches!(exit, Exit::Hlt), "{exit:?}");
        assert_eq!(a.reg(Reg::Rax), 3);
        assert_eq!(a.reg(Reg::Rbx), 30, "next sub-block must use the new code");
    }

    #[test]
    fn smc_notice_survives_other_vcpu_drain_region() {
        smc_notice_survives_other_vcpu_drain(true);
    }

    #[test]
    fn smc_notice_survives_other_vcpu_drain_chained() {
        smc_notice_survives_other_vcpu_drain(false);
    }

    // `mov eax, 42` then `jmp $` (to self): sets RAX=42 and terminates the block
    // without touching guest memory (the jmp is the block terminator, not a loop).
    const CODE: &[u8] = &[0xb8, 0x2a, 0x00, 0x00, 0x00, 0xeb, 0xf9];
    const ENTRY: u64 = 0x1000;

    fn mem_with_code() -> Memory {
        let mut m = Memory::new(MemoryModel::Flat { size: 0x4000 });
        m.map(ENTRY, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
        m.write_bytes(ENTRY, CODE).unwrap();
        m
    }

    fn request(mem: &Memory) -> TierUpRequest {
        let ir = lift_block(
            mem,
            x86jit_core::lift::FetchAddr::flat(ENTRY),
            CpuMode::Long64,
        )
        .expect("lift the block");
        TierUpRequest {
            precise_split_stores: false,
            mxcsr_mode: 0,
            pc: ENTRY,
            unit: TierUpUnit::Block(Arc::new(ir)),
            consistency: MemConsistency::Fast,
            mmio: None,
            guest_base: 0,
            spans: vec![(ENTRY, CODE.len() as u32)],
            epoch: 0,
        }
    }

    /// Run a compiled block from a fresh CPU and return RAX (should be 42).
    fn run_rax(entry: CompiledPtr, mem: &Memory) -> u64 {
        let mut cpu = CpuState::new();
        cpu.rip = ENTRY;
        let step = unsafe {
            run_compiled(
                entry,
                &mut cpu,
                mem,
                CpuMode::Long64,
                x86jit_core::MemConsistency::Fast,
            )
        };
        assert!(matches!(step, StepResult::Continue), "block continues");
        cpu.gpr[0]
    }

    fn compiled_entry(block: CachedBlock) -> CompiledPtr {
        match block {
            CachedBlock::Compiled { entry } => {
                assert!(!entry.0.is_null(), "compiled entry is non-null");
                entry
            }
            CachedBlock::Interpreted(_) => {
                panic!("expected a compiled block")
            }
        }
    }

    fn assert_compile_failure_recovers(stage: usize) {
        use x86jit_core::{Exit, Reg, Vm, VmConfig};
        let jit = JitBackend::new();
        jit.shared
            .fail_compile_stage
            .store(stage, Ordering::Relaxed);
        let shared = jit.shared.clone();
        let mut vm = Vm::with_backend(VmConfig::flat(0x2000), Box::new(jit));
        vm.map(ENTRY, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
        vm.write_bytes(ENTRY, CODE).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, ENTRY);
        assert!(matches!(
            cpu.run(&vm, Some(1)),
            Exit::CodeSpaceExhausted { addr: ENTRY, .. }
        ));
        assert_eq!(cpu.reg(Reg::Rax), 0);
        assert_eq!(cpu.reg(Reg::Rip), ENTRY);
        assert!(vm.backend.code_space_exhausted());
        let compile_id = shared.jit(0).next_id;
        shared.fail_compile_stage.store(0, Ordering::Relaxed);
        vm.write_bytes(ENTRY + 0x100, CODE).unwrap();
        let fresh = lift_block(
            &vm.mem,
            x86jit_core::lift::FetchAddr::flat(ENTRY + 0x100),
            CpuMode::Long64,
        )
        .unwrap();
        assert!(matches!(
            vm.backend
                .materialize(&fresh, MemConsistency::Fast, None, 0),
            CachedBlock::Interpreted(_)
        ));
        assert_eq!(
            shared.jit(0).next_id,
            compile_id,
            "no compile after exhaustion"
        );
        for _ in 0..3 {
            assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
            assert_eq!(cpu.reg(Reg::Rax), 42);
        }
    }

    #[test]
    fn code_arena_allocation_error_recovers() {
        assert_compile_failure_recovers(1);
    }

    #[test]
    fn executable_memory_finalize_error_recovers() {
        assert_compile_failure_recovers(2);
    }

    #[test]
    fn codemap_registration_failure_recovers() {
        assert_compile_failure_recovers(3);
    }

    #[test]
    fn trap_site_allocation_failure_recovers() {
        assert_compile_failure_recovers(4);
    }

    #[test]
    fn backend_resource_errors_exhaust_code_space() {
        for kind in [
            std::io::ErrorKind::OutOfMemory,
            std::io::ErrorKind::WouldBlock,
        ] {
            let jit = JitBackend::new();
            assert_eq!(
                jit.shared
                    .compile_result(Err(ModuleError::Backend(std::io::Error::from(kind).into()))),
                None
            );
            assert!(jit.code_space_exhausted());
        }
    }

    fn assert_wrapped_backend_resource_error(errno: i32) {
        let jit = JitBackend::new();
        let error = region::Error::SystemCall(std::io::Error::from_raw_os_error(errno));
        assert_eq!(
            jit.shared
                .compile_result(Err(ModuleError::Backend(error.into()))),
            None
        );
        assert!(jit.code_space_exhausted());
    }

    #[test]
    fn wrapped_backend_enomem_exhausts_code_space() {
        assert_wrapped_backend_resource_error(libc::ENOMEM);
    }

    #[test]
    fn wrapped_backend_eagain_exhausts_code_space() {
        assert_wrapped_backend_resource_error(libc::EAGAIN);
    }

    #[test]
    fn wrapped_backend_permission_error_preserves_diagnostics() {
        let jit = JitBackend::new();
        let error = region::Error::SystemCall(std::io::Error::from_raw_os_error(libc::EACCES));
        let diagnostic = error.to_string();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            jit.shared
                .compile_result(Err(ModuleError::Backend(error.into())))
        }))
        .expect_err("permission errors must remain distinct from exhaustion");
        let message = panic.downcast_ref::<String>().expect("diagnostic panic");
        assert!(message.contains(&diagnostic), "{message}");
        assert!(!jit.code_space_exhausted());
    }

    fn running_cycle_notices_exhaustion(region: bool) {
        use std::sync::mpsc;
        use std::time::{Duration, Instant};
        use x86jit_core::{Exit, Reg, Vm, VmConfig};

        const ENTRY: u64 = 0x1000;
        const SIGNAL: u64 = 0x3000;
        const TIMEOUT: Duration = Duration::from_secs(10);
        let jit = if region {
            JitBackend::with_superblocks(RegionCaps {
                max_blocks: 8,
                max_icount: 128,
            })
        } else {
            JitBackend::new()
        };
        let shared = Arc::clone(&jit.shared);
        let mut vm = Vm::with_backend(VmConfig::flat(0x4000), Box::new(jit));
        vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
        // mov r11,1; xchg [SIGNAL],r11; jmp second;
        // second: cmp rax,0; je first; hlt.
        // SDM Vol 3A §11.1.2.1: memory XCHG is automatically locked.
        // The handshake's aligned atomic read witnesses the guest's progress.
        let code = [
            0x49, 0xc7, 0xc3, 1, 0, 0, 0, 0x4c, 0x87, 0x1c, 0x25, 0, 0x30, 0, 0, 0xeb, 0, 0x48,
            0x83, 0xf8, 0, 0x74, 0xe9, 0xf4,
        ];
        vm.write_bytes(ENTRY, &code).unwrap();
        let vm = Arc::new(vm);
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, ENTRY);
        assert!(matches!(cpu.run(&vm, Some(64)), Exit::BudgetExhausted));
        if region {
            assert!(vm.cache.regions() > 0, "must warm a compiled region loop");
        } else {
            assert!(vm.cache.chained() > 0, "must warm compiled chain edges");
        }
        vm.mem.write(SIGNAL, 0, 8).unwrap();
        let waiting = vm.new_vcpu();
        let words: Vec<_> = shared
            .dispatch_words
            .lock()
            .unwrap()
            .iter()
            .map(|word| word.upgrade().unwrap())
            .collect();
        let request = cpu.exit_request();
        let running = Arc::clone(&vm);
        let (tx, rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let exit = cpu.run(&running, None);
            tx.send((exit, cpu)).unwrap();
        });
        let deadline = Instant::now() + TIMEOUT;
        while vm.mem.read(SIGNAL, 8).unwrap() == 0 {
            if Instant::now() >= deadline {
                request.request_exit();
                worker.join().unwrap();
                panic!("warmed guest did not make progress");
            }
            std::thread::yield_now();
        }
        shared.exhaust_code_space::<()>();
        let result = rx.recv_timeout(TIMEOUT);
        // Bound a failing negative control too: release the worker before asserting.
        if result.is_err() {
            request.request_exit();
            rx.recv_timeout(TIMEOUT)
                .expect("exit request stops the guest");
        }
        worker.join().unwrap();
        let (exit, mut cpu) = result.expect("exhaustion must bound an unbudgeted run");
        assert!(matches!(exit, Exit::CodeSpaceExhausted { .. }), "{exit:?}");
        assert_eq!(vm.mem.read(SIGNAL, 8).unwrap(), 1);
        use x86jit_core::vm::CODE_SPACE_EXHAUSTED;
        assert_eq!(words[0].load(Ordering::Acquire), 0);
        assert_eq!(words[1].load(Ordering::Acquire), CODE_SPACE_EXHAUSTED);
        // Consuming one vcpu's notice cannot consume another's, and late
        // registration must inherit the sticky state.
        for mut other in [waiting, vm.new_vcpu()] {
            assert!(matches!(
                other.run(&vm, Some(0)),
                Exit::CodeSpaceExhausted { .. }
            ));
            assert!(matches!(other.run(&vm, Some(0)), Exit::BudgetExhausted));
        }
        // Model a broadcast arriving after the sticky flag was already reported.
        // A simultaneous host request must remain independent of this late bit.
        words[0].fetch_or(CODE_SPACE_EXHAUSTED, Ordering::Release);
        request.request_exit();
        assert!(matches!(cpu.run(&vm, Some(0)), Exit::Interrupted));
        assert_eq!(words[0].load(Ordering::Acquire), CODE_SPACE_EXHAUSTED);
        assert!(matches!(cpu.run(&vm, Some(0)), Exit::BudgetExhausted));
        assert_eq!(words[0].load(Ordering::Acquire), 0);
        // A repeated transition cannot re-arm a vcpu that already reported it.
        shared.exhaust_code_space::<()>();
        cpu.set_reg(Reg::Rax, 1);
        let mut reference_vm = Vm::with_backend(
            VmConfig::flat(0x4000),
            Box::new(x86jit_core::InterpreterBackend),
        );
        reference_vm
            .map(0, 0x4000, Prot::RWX, RegionKind::Ram)
            .unwrap();
        reference_vm.write_bytes(ENTRY, &code).unwrap();
        reference_vm.mem.write(SIGNAL, 1, 8).unwrap();
        let mut reference = reference_vm.new_vcpu();
        reference.cpu = cpu.cpu.clone();
        assert!(matches!(reference.run(&reference_vm, None), Exit::Hlt));
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.cpu.rip, reference.cpu.rip);
        assert_eq!(cpu.cpu.gpr, reference.cpu.gpr);
        assert_eq!(cpu.cpu.flags.to_rflags(), reference.cpu.flags.to_rflags());
    }

    #[test]
    fn code_exhaustion_notifies_running_chained_cycle() {
        running_cycle_notices_exhaustion(false);
    }

    #[test]
    fn code_exhaustion_notifies_running_region_loop() {
        running_cycle_notices_exhaustion(true);
    }

    #[test]
    fn backend_non_resource_errors_preserve_diagnostics() {
        for errno in [libc::EPERM, libc::EACCES] {
            let jit = JitBackend::new();
            let error = std::io::Error::from_raw_os_error(errno);
            let diagnostic = error.to_string();
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                jit.shared
                    .compile_result(Err(ModuleError::Backend(error.into())))
            }))
            .expect_err("permission errors must remain distinct from exhaustion");
            let message = panic.downcast_ref::<String>().expect("diagnostic panic");
            assert!(message.contains(&diagnostic), "{message}");
            assert!(!jit.code_space_exhausted());
        }
    }

    #[test]
    fn background_worker_spawn_failure_is_recoverable() {
        use x86jit_core::{Exit, Reg, Vm, VmConfig};
        let jit = JitBackend::new();
        jit.shared.fail_worker_spawn.store(true, Ordering::Relaxed);
        let handle = jit.tier_up_handle();
        let shared = jit.shared.clone();
        let mut vm = Vm::with_backend(VmConfig::flat(0x2000), Box::new(jit));
        vm.set_tier_up_after(Some(1));
        vm.set_tier_up_background(true);
        vm.map(ENTRY, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
        vm.write_bytes(ENTRY, CODE).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, ENTRY);
        assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
        assert!(matches!(
            cpu.run(&vm, Some(1)),
            Exit::CodeSpaceExhausted { addr: ENTRY, .. }
        ));
        handle.wait_idle();
        assert_eq!(vm.cache.tier_pending_len(), 0);
        assert!(
            shared.inner.lock().unwrap().is_none(),
            "never compiled inline"
        );
        assert!(shared.background.queue.lock().unwrap().items.is_empty());
        shared.fail_worker_spawn.store(false, Ordering::Relaxed);
        assert!(matches!(cpu.run(&vm, Some(3)), Exit::BudgetExhausted));
        assert_eq!(cpu.reg(Reg::Rax), 42);
        assert!(shared.inner.lock().unwrap().is_none(), "sticky after reset");
    }

    const LOOP: &[u8] = &[
        0x31, 0xc0, // xor eax, eax
        0xb9, 7, 0, 0, 0, // mov ecx, 7
        0x01, 0xc8, // add eax, ecx
        0x83, 0xe9, 1, // sub ecx, 1
        0x75, 0xf9, // jnz body
        0xf4, // hlt
    ];
    const TEST_CAPS: RegionCaps = RegionCaps {
        max_blocks: 16,
        max_icount: 256,
    };

    #[test]
    fn background_region_caps_snapshot_survives_exhaustion() {
        use x86jit_core::{Exit, Reg, Vm, VmConfig};
        struct ExhaustOnCaps {
            jit: JitBackend,
            armed: Arc<AtomicBool>,
        }
        impl Backend for ExhaustOnCaps {
            fn code_lifetime(
                &self,
            ) -> Option<std::sync::Arc<x86jit_core::code_lifetime::CodeLifetime>> {
                self.jit.code_lifetime()
            }

            fn code_space_exhausted(&self) -> bool {
                self.jit.code_space_exhausted()
            }
            fn materialize(
                &self,
                ir: &IrBlock,
                consistency: MemConsistency,
                mmio: Option<(u64, u64)>,
                guest_base: u64,
            ) -> CachedBlock {
                self.jit.materialize(ir, consistency, mmio, guest_base)
            }
            fn region_caps(&self) -> Option<RegionCaps> {
                let caps = self.jit.region_caps();
                if self.armed.swap(false, Ordering::Relaxed) {
                    // Deterministically model the worker exhausting storage between
                    // the dispatcher's region decision and region lift.
                    self.jit.shared.exhaust_code_space::<()>();
                }
                caps
            }
            fn tier_up_async(&self, req: TierUpRequest) -> TierUpSubmit {
                assert!(matches!(req.unit, TierUpUnit::Region(_)));
                self.jit.tier_up_async(req)
            }
        }
        let armed = Arc::new(AtomicBool::new(false));
        let backend = ExhaustOnCaps {
            jit: JitBackend::with_superblocks(TEST_CAPS),
            armed: armed.clone(),
        };
        let mut vm = Vm::with_backend(VmConfig::flat(0x2000), Box::new(backend));
        vm.set_tier_up_after(Some(1));
        vm.set_tier_up_background(true);
        vm.map(ENTRY, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
        vm.write_bytes(ENTRY, LOOP).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, ENTRY);
        assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
        cpu.set_reg(Reg::Rip, ENTRY);
        armed.store(true, Ordering::Relaxed);
        assert!(matches!(
            cpu.run(&vm, Some(1)),
            Exit::CodeSpaceExhausted { addr: ENTRY, .. }
        ));
        assert_eq!(vm.cache.tier_pending_len(), 0);
        assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rax), 28);
    }

    fn assert_tier_failure_recovers(stage: usize, background: bool, region: bool) {
        use x86jit_core::{cache::BlockKey, Exit, Reg, Vm, VmConfig};
        let jit = if region {
            JitBackend::with_superblocks(TEST_CAPS)
        } else {
            JitBackend::new()
        };
        jit.shared
            .fail_compile_stage
            .store(stage, Ordering::Relaxed);
        let events = Arc::new(Mutex::new(Vec::new()));
        let observed = events.clone();
        jit.set_compile_failure_observer(move |event| observed.lock().unwrap().push(event));
        let handle = jit.tier_up_handle();
        let compiler = Arc::downgrade(&jit.shared);
        let mut vm = Vm::with_backend(VmConfig::flat(0x2000), Box::new(jit));
        vm.set_tier_up_after(Some(1));
        vm.set_tier_up_background(background);
        vm.set_tier_up_region_after(Some(3));
        vm.map(ENTRY, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
        vm.write_bytes(ENTRY, if region { LOOP } else { CODE })
            .unwrap();
        let key = BlockKey::new(ENTRY, CpuMode::Long64);
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, ENTRY);
        assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
        let original = match vm.cache.get(key).unwrap() {
            CachedBlock::Interpreted(ir) => ir,
            _ => panic!("cold block must be interpreted"),
        };
        // Stall background compilation to inspect the submitted unit and spans.
        let pause = background.then(|| handle.pause_compiler());
        for heat in 1..=if region { 3 } else { 1 } {
            cpu.set_reg(Reg::Rip, ENTRY);
            let exit = cpu.run(&vm, Some(1));
            if background || heat < 3 && region {
                assert!(matches!(exit, Exit::BudgetExhausted));
            } else {
                assert!(matches!(exit, Exit::CodeSpaceExhausted { addr: ENTRY, .. }));
            }
            if region && heat < 3 {
                assert_eq!(vm.cache.tier_pending_len(), 0, "waiting for T2");
            }
        }
        if background {
            let shared = compiler.upgrade().unwrap();
            let q = shared.background.queue.lock().unwrap();
            assert_eq!(q.items.len(), 1);
            assert_eq!(
                matches!(q.items[0].request.unit, TierUpUnit::Region(_)),
                region
            );
            assert_eq!(q.items[0].request.spans.len() > 1, region);
            drop(q);
            drop(pause);
            handle.wait_idle();
            let shared = compiler.upgrade().unwrap();
            assert_eq!(shared.jit(0).next_id, 1, "failure hook was reached");
            let done = shared.done.lock().unwrap();
            assert_eq!(done.len(), 1);
            assert!(matches!(done[0].block, CachedBlock::Interpreted(_)));
            drop(done);
            cpu.set_reg(Reg::Rip, ENTRY);
            assert!(matches!(
                cpu.run(&vm, Some(1)),
                Exit::CodeSpaceExhausted { addr: ENTRY, .. }
            ));
        }
        assert!(vm.backend.code_space_exhausted());
        cpu.set_reg(Reg::Rip, ENTRY);
        assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
        assert_eq!(vm.cache.tier_pending_len(), 0, "failed completion cleared");
        assert_eq!(vm.cache.tier_bg_published(), 0);
        assert_eq!(vm.cache.tier_bg_rejected(), u64::from(background));
        assert_eq!(
            vm.memory_stats().cache.tier_bg_rejected_compile_failed,
            u64::from(background)
        );
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].reason,
            x86jit_core::CompileFailureReason::CodeSpace
        );
        assert_eq!(events[0].guest_rip, ENTRY);
        assert_eq!(
            events[0].kind,
            if region {
                CompiledCodeKind::Region
            } else {
                CompiledCodeKind::Block
            }
        );
        assert!(events[0].guest_bytes > 0 && events[0].instructions > 0);
        drop(events);
        let stats = vm.memory_stats().cache;
        assert_eq!(stats.tier_bg_failed_code_space, u64::from(background));
        assert_eq!(stats.tier_bg_failed_descriptor_limit, 0);
        assert_eq!(stats.tier_bg_failed_other_backend, 0);
        assert_eq!(stats.tier_bg_interpreter_only, 0);
        assert_eq!(vm.cache.regions(), 0);
        let CachedBlock::Interpreted(retained) = vm.cache.get(key).unwrap() else {
            panic!("failed compilation must leave the interpreted cache entry");
        };
        if background {
            assert!(
                Arc::ptr_eq(&retained, &original),
                "no replacement published"
            );
        }
        if region {
            assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rax), 28);
        } else {
            assert_eq!(cpu.reg(Reg::Rax), 42);
        }
    }

    #[test]
    fn inline_tier_real_failures_recover() {
        for stage in 1..=3 {
            assert_tier_failure_recovers(stage, false, false);
        }
    }

    #[test]
    fn background_block_real_failures_recover() {
        for stage in 1..=3 {
            assert_tier_failure_recovers(stage, true, false);
        }
    }

    #[test]
    fn background_loop_region_real_failures_recover() {
        for stage in 1..=3 {
            assert_tier_failure_recovers(stage, true, true);
        }
    }

    #[test]
    fn background_backend_panic_is_not_suppressed_and_worker_survives() {
        use x86jit_core::{Exit, Reg, Vm, VmConfig};
        let jit = JitBackend::new();
        jit.shared.fail_compile_stage.store(5, Ordering::Relaxed);
        let compiler = Arc::downgrade(&jit.shared);
        let events = Arc::new(Mutex::new(Vec::new()));
        let observed = events.clone();
        let weak = compiler.clone();
        jit.set_compile_failure_observer(move |event| {
            let compiler = weak.upgrade().unwrap();
            assert!(compiler.compile_serial.try_lock().is_ok());
            assert!(compiler.inner.try_lock().is_ok());
            assert!(compiler.background.queue.try_lock().is_ok());
            observed.lock().unwrap().push(event);
        });
        let handle = jit.tier_up_handle();
        let mut vm = Vm::with_backend(VmConfig::flat(0x4000), Box::new(jit));
        vm.set_tier_up_after(Some(1));
        vm.set_tier_up_background(true);
        vm.map(ENTRY, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
        vm.write_bytes(ENTRY, CODE).unwrap();
        let mut cpu = vm.new_vcpu();
        for _ in 0..5 {
            cpu.set_reg(Reg::Rip, ENTRY);
            assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
            handle.wait_idle();
        }
        // A caught panic does not prove the unit is at fault (here a host EACCES),
        // so later runs resubmit it rather than suppressing it. The newest failure
        // may still wait in the completion queue for the next dispatch.
        let stats = vm.memory_stats().cache;
        assert!(stats.tier_bg_submitted >= 2, "{stats:?}");
        assert!(stats.tier_bg_failed_other_backend + 1 >= stats.tier_bg_submitted);
        assert_eq!(stats.tier_bg_interpreter_only, 0);
        assert_eq!(stats.tier_bg_failed_code_space, 0);
        assert!(!vm.backend.code_space_exhausted());
        let events = events.lock().unwrap();
        assert_eq!(events.len() as u64, stats.tier_bg_submitted);
        assert!(events
            .iter()
            .all(|e| e.reason == x86jit_core::CompileFailureReason::Unclassified));
        drop(events);
        compiler
            .upgrade()
            .unwrap()
            .fail_compile_stage
            .store(0, Ordering::Relaxed);
        vm.write_bytes(ENTRY, CODE).unwrap();
        for _ in 0..3 {
            cpu.set_reg(Reg::Rip, ENTRY);
            assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
            handle.wait_idle();
        }
        assert_eq!(vm.memory_stats().cache.tier_bg_published, 1);
        assert_eq!(vm.cache.tier_pending_len(), 0);
    }

    #[test]
    fn failure_observer_replacement_survives_panicking_callback() {
        let jit = Arc::new(JitBackend::new());
        let weak = Arc::downgrade(&jit);
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = calls.clone();
        jit.set_compile_failure_observer(move |_| {
            let observed = observed.clone();
            weak.upgrade()
                .unwrap()
                .set_compile_failure_observer(move |_| {
                    observed.fetch_add(1, Ordering::Relaxed);
                });
            panic!("failure observer fixture");
        });
        jit.set_compile_limit(0);
        let mem = mem_with_code();
        let req = request(&mem);
        let TierUpUnit::Block(ir) = req.unit else {
            unreachable!()
        };
        for _ in 0..3 {
            assert!(matches!(
                jit.materialize(&ir, MemConsistency::Fast, None, 0),
                CachedBlock::Interpreted(_)
            ));
        }
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    /// A background-compiled block is drained as a `Compiled` unit and runs correctly
    /// (RAX=42), matching the eager path.
    #[test]
    fn background_compile_runs_correctly() {
        let mem = mem_with_code();
        let jit = JitBackend::new();
        let handle = jit.tier_up_handle();

        assert_eq!(jit.tier_up_async(request(&mem)), TierUpSubmit::Queued);
        handle.wait_idle();

        let mut done = jit.tier_up_finished();
        assert_eq!(done.len(), 1, "exactly one completion drained");
        let fin = done.pop().unwrap();
        assert_eq!(fin.pc, ENTRY);
        assert_eq!(fin.spans, vec![(ENTRY, CODE.len() as u32)]);
        assert_eq!(fin.epoch, 0);
        assert_eq!(run_rax(compiled_entry(fin.block), &mem), 42);

        // Drained: the fast probe is clear and a second drain is empty.
        assert!(jit.tier_up_finished().is_empty());
    }

    /// No worker thread is spawned until the first `tier_up_async`.
    #[test]
    fn worker_spawns_lazily() {
        let jit = JitBackend::new();
        assert!(
            jit.worker.lock().unwrap().is_none(),
            "no thread before any tier-up request"
        );
        let mem = mem_with_code();
        let _ = jit.tier_up_async(request(&mem));
        assert!(
            jit.worker.lock().unwrap().is_some(),
            "the first request spawns the worker"
        );
    }

    #[test]
    fn background_drop_releases_diagnostic_handles() {
        let jit = JitBackend::new();
        let shared = Arc::downgrade(&jit.shared);
        let handle = jit.tier_up_handle();
        let pause = handle.pause_compiler();
        drop(jit);
        assert!(
            shared.upgrade().is_none(),
            "diagnostics must not retain the compiler"
        );
        handle.wait_idle();
        drop(pause);
    }

    #[test]
    fn background_drop_cancels_queued_jobs() {
        let jit = JitBackend::new();
        let shared = jit.shared.clone();
        let handle = jit.tier_up_handle();
        let pause = handle.pause_compiler();
        let mem = mem_with_code();
        for _ in 0..3 {
            assert_eq!(jit.tier_up_async(request(&mem)), TierUpSubmit::Queued);
        }
        drop(jit);
        assert!(
            shared.done.lock().unwrap().is_empty(),
            "shutdown must discard completions"
        );
        assert_eq!(
            shared.compile_attempts.load(Ordering::Relaxed),
            0,
            "queued jobs must be cancelled"
        );
        assert_eq!(shared.background.queue.lock().unwrap().outstanding, 0);
        handle.wait_idle();
        drop(pause);
    }

    #[test]
    fn shutdown_destroys_compiler_with_active_diagnostic_operation() {
        use std::sync::mpsc;
        use std::time::Duration;

        struct DelayedObserver {
            destroyed: Arc<AtomicBool>,
            owner: std::thread::ThreadId,
        }
        impl Drop for DelayedObserver {
            fn drop(&mut self) {
                if std::thread::current().id() != self.owner {
                    std::thread::sleep(Duration::from_millis(5));
                }
                self.destroyed.store(true, Ordering::Release);
            }
        }

        for iteration in 0..1000 {
            let jit = JitBackend::new();
            let destroyed = Arc::new(AtomicBool::new(false));
            let observer = DelayedObserver {
                destroyed: destroyed.clone(),
                owner: std::thread::current().id(),
            };
            jit.set_compiled_code_observer(move |_| {
                std::hint::black_box(&observer);
            });
            let mem = mem_with_code();
            let TierUpUnit::Block(ir) = request(&mem).unit else {
                unreachable!()
            };
            assert!(matches!(
                jit.materialize(&ir, MemConsistency::Fast, None, 0),
                CachedBlock::Compiled { .. }
            ));
            let handle = jit.tier_up_handle();
            let (entered_tx, entered_rx) = mpsc::channel();
            let (release_tx, release_rx) = mpsc::channel();
            let diagnostic = std::thread::spawn(move || {
                // Hold the same upgraded state as wait_idle across all of shutdown.
                let state = handle.shared.upgrade().unwrap();
                entered_tx.send(()).unwrap();
                release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                assert!(state.queue.lock().unwrap().shutdown);
                drop(state);
                handle.wait_idle();
            });
            entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let vm =
                x86jit_core::Vm::with_backend(x86jit_core::VmConfig::flat(0x2000), Box::new(jit));
            let result = vm.shutdown();
            let destroyed_before_return = destroyed.load(Ordering::Acquire);
            release_tx.send(()).unwrap();
            diagnostic.join().unwrap();
            assert_eq!(
                result,
                Ok(ShutdownReport::default()),
                "iteration {iteration}"
            );
            assert!(destroyed_before_return, "iteration {iteration}");
        }
    }

    #[test]
    fn shutdown_joins_inflight_and_cancels_pending_compile() {
        use std::sync::mpsc;
        use std::time::{Duration, Instant};
        let timeout = Duration::from_secs(5);
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        let mut jit = JitBackend::new();
        jit.set_lock_telemetry(true);
        jit.set_compiled_code_observer(move |_| {
            entered_tx.send(()).unwrap();
            release_rx.lock().unwrap().recv_timeout(timeout).unwrap();
        });
        let weak = Arc::downgrade(&jit.shared);
        let handle = jit.tier_up_handle();
        let mem = mem_with_code();
        assert_eq!(jit.tier_up_async(request(&mem)), TierUpSubmit::Queued);
        entered_rx.recv_timeout(timeout).unwrap();
        assert_eq!(jit.tier_up_async(request(&mem)), TierUpSubmit::Queued);
        let (done_tx, done_rx) = mpsc::channel();
        let shutdown = std::thread::spawn(move || {
            let report = jit.shutdown();
            assert!(jit.worker.lock().unwrap().is_none());
            assert!(jit.tier_up_finished().is_empty());
            assert_eq!(jit.tier_up_async(request(&mem)), TierUpSubmit::Unsupported);
            assert!(
                jit.worker.lock().unwrap().is_none(),
                "shutdown cannot respawn"
            );
            assert!(jit.lock_telemetry().compile_serial.acquisitions > 0);
            done_tx.send(report).unwrap();
        });
        let deadline = Instant::now() + timeout;
        while !weak
            .upgrade()
            .unwrap()
            .background
            .queue
            .lock()
            .unwrap()
            .shutdown
        {
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
        assert!(matches!(
            done_rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        release_tx.send(()).unwrap();
        assert_eq!(
            done_rx.recv_timeout(timeout).unwrap().unwrap(),
            ShutdownReport {
                workers_joined: 1,
                jobs_cancelled: 1,
                outstanding_handles: 0,
            }
        );
        shutdown.join().unwrap();
        assert!(weak.upgrade().is_none());
        handle.wait_idle();
    }

    #[test]
    fn vm_shutdown_drains_multiple_teardown_batches_before_success() {
        let mut jit = JitBackend::new();
        let starts = Arc::new(AtomicUsize::new(0));
        let active = Arc::new(AtomicUsize::new(0));
        let worker_starts = starts.clone();
        let worker_active = active.clone();
        jit.set_worker_spawner(move |task| {
            worker_starts.fetch_add(1, Ordering::SeqCst);
            let active = worker_active.clone();
            std::thread::Builder::new().spawn(move || {
                active.fetch_add(1, Ordering::SeqCst);
                task();
                active.fetch_sub(1, Ordering::SeqCst);
            })
        });
        let released = Arc::new(AtomicUsize::new(0));
        let releases = released.clone();
        jit.set_code_lifecycle_observer(move |event| {
            if event.kind == CodeLifecycleKind::Release {
                releases.fetch_add(1, Ordering::SeqCst);
            }
        });
        let mut vm =
            x86jit_core::Vm::with_backend(x86jit_core::VmConfig::flat(0x4000), Box::new(jit));
        vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, &[0xf4]).unwrap();
        let mut ir = lift_block(
            &vm.mem,
            x86jit_core::lift::FetchAddr::flat(0x1000),
            CpuMode::Long64,
        )
        .unwrap();
        for i in 0..130 {
            ir.guest_start = 0x1000 + i;
            let block = vm.backend.materialize(&ir, MemConsistency::Fast, None, 0);
            assert!(matches!(block, CachedBlock::Compiled { .. }));
            vm.cache.insert(
                x86jit_core::BlockKey::new(ir.guest_start, CpuMode::Long64),
                block,
                vec![(ir.guest_start, 1)],
                |_| {},
            );
        }
        assert_eq!(vm.cache.code_lifetime().live_units().len(), 130);
        assert_eq!(released.load(Ordering::SeqCst), 0);
        assert_eq!(vm.shutdown().unwrap().outstanding_handles, 0);
        assert_eq!(released.load(Ordering::SeqCst), 130);
        assert_eq!(active.load(Ordering::SeqCst), 0);
        assert_eq!(
            starts.load(Ordering::SeqCst),
            1,
            "teardown must not start a deferred thread after success"
        );
    }

    #[test]
    fn worker_policy_covers_maintenance_and_deferred_reclamation() {
        use std::cell::Cell;
        use std::sync::mpsc;
        use std::time::Duration;
        thread_local! { static REGISTERED: Cell<bool> = const { Cell::new(false) }; }
        let owner = std::thread::current().id();
        let mut jit = JitBackend::new();
        let starts = Arc::new(AtomicUsize::new(0));
        let worker_starts = starts.clone();
        let (done_tx, done_rx) = mpsc::channel();
        jit.set_worker_spawner(move |task| {
            worker_starts.fetch_add(1, Ordering::SeqCst);
            let done = done_tx.clone();
            std::thread::Builder::new().spawn(move || {
                REGISTERED.set(true);
                task();
                done.send(()).unwrap();
            })
        });
        let (release_tx, release_rx) = mpsc::channel();
        jit.set_code_lifecycle_observer(move |event| {
            if std::thread::current().id() != owner {
                assert!(REGISTERED.get(), "engine thread bypassed registration");
            }
            if event.kind == CodeLifecycleKind::Release {
                release_tx.send(()).unwrap();
            }
        });
        let mem = mem_with_code();
        let TierUpUnit::Block(ir) = request(&mem).unit else {
            unreachable!()
        };
        assert!(matches!(
            jit.materialize(&ir, MemConsistency::Fast, None, 0),
            CachedBlock::Compiled { .. }
        ));
        let domain = jit.shared.lifetime.clone();
        let mut participant = domain.participant();
        let held = participant.enter();
        // Explicit shutdown must report pending grace instead of certifying a
        // fork boundary that would require a deferred worker after destruction.
        assert!(matches!(
            jit.shutdown(),
            Err(ShutdownError::LiveResources {
                resource: "code reclamation",
                ..
            })
        ));
        drop(jit);
        assert_eq!(starts.load(Ordering::SeqCst), 2);
        assert!(release_rx.try_recv().is_err());
        drop(held);
        release_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        for _ in 0..2 {
            done_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        }
    }

    #[test]
    fn shutdown_names_live_compiler_handles_and_is_idempotent() {
        let mut jit = JitBackend::new();
        let shared = jit.shared.clone();
        let error = jit.shutdown().unwrap_err();
        assert_eq!(
            error,
            ShutdownError::LiveResources {
                resource: "compiler state",
                report: ShutdownReport {
                    outstanding_handles: 1,
                    ..ShutdownReport::default()
                },
            }
        );
        assert!(error.to_string().contains("compiler state"));
        drop(shared);
        assert_eq!(jit.shutdown().unwrap(), ShutdownReport::default());
        assert_eq!(jit.shutdown().unwrap(), ShutdownReport::default());
        let mem = mem_with_code();
        let TierUpUnit::Block(ir) = request(&mem).unit else {
            unreachable!()
        };
        assert!(matches!(
            jit.materialize(&ir, MemConsistency::Fast, None, 0),
            CachedBlock::Interpreted(_)
        ));
        assert!(
            jit.shared.inner.lock().unwrap().is_none(),
            "shutdown cannot compile inline"
        );
    }

    #[test]
    fn shutdown_reports_worker_panic_after_join() {
        let mut jit = JitBackend::new();
        // Inject a failed worker without exposing a production panic lever.
        *jit.worker.get_mut().unwrap() = Some(std::thread::spawn(|| panic!("compile failure")));
        let error = jit.shutdown().unwrap_err();
        assert_eq!(
            error,
            ShutdownError::WorkerPanicked {
                report: ShutdownReport {
                    workers_joined: 1,
                    ..ShutdownReport::default()
                },
            }
        );
        assert!(jit.worker.lock().unwrap().is_none());
        assert_eq!(jit.shutdown().unwrap_err(), error);
    }

    /// A full queue returns `Busy` (never an inline compile), and the queued
    /// requests all complete once the compiler is unblocked. The worker is stalled
    /// by holding the JIT mutex it needs, so the queue provably fills.
    #[test]
    fn busy_on_full_queue_then_queued_complete() {
        let mem = mem_with_code();
        let jit = JitBackend::new();
        let handle = jit.tier_up_handle();

        let guard = jit.shared.inner.lock().unwrap(); // stall every compile
        let mut queued = 0usize;
        let mut busy = false;
        for _ in 0..(TIER_QUEUE_CAP + 8) {
            match jit.tier_up_async(request(&mem)) {
                TierUpSubmit::Queued => queued += 1,
                TierUpSubmit::Busy => busy = true,
                TierUpSubmit::Unsupported => panic!("unexpected Unsupported"),
            }
        }
        assert!(busy, "a full queue must report Busy");
        assert!(queued >= TIER_QUEUE_CAP, "queue filled to capacity");

        drop(guard); // unblock the worker
        handle.wait_idle();
        assert_eq!(
            jit.tier_up_finished().len(),
            queued,
            "every queued request completed"
        );
    }

    /// The eager/foreground `materialize` still works (correct output, no
    /// deadlock) while the worker churns a backlog — both take the same JIT mutex.
    #[test]
    fn eager_materialize_works_while_worker_busy() {
        let mem = mem_with_code();
        let jit = JitBackend::new();
        let handle = jit.tier_up_handle();

        for _ in 0..30 {
            let _ = jit.tier_up_async(request(&mem));
        }
        // Foreground compile the same block amid the backlog.
        let ir = lift_block(
            &mem,
            x86jit_core::lift::FetchAddr::flat(ENTRY),
            CpuMode::Long64,
        )
        .unwrap();
        let entry = compiled_entry(jit.materialize(&ir, MemConsistency::Fast, None, 0));
        assert_eq!(run_rax(entry, &mem), 42);

        handle.wait_idle();
        for fin in jit.tier_up_finished() {
            assert_eq!(run_rax(compiled_entry(fin.block), &mem), 42);
        }
    }

    /// The host range a compiled block emits to the perf map starts
    /// at the same host entry `codemap` recorded (both take `entry.0` at the same
    /// call site), and the block is genuinely covered by `codemap` at that guest
    /// RIP. This intentionally does NOT assert on `codemap::lookup(entry.0)`: at the
    /// exact entry offset the srcloc table may have no covering entry (returns
    /// `None`), because the first source location can follow the prologue — which is
    /// *why* perfmap names the symbol from `ir.guest_start` directly, not from
    /// codemap. Instead we scan forward within the block and confirm codemap resolves
    /// it to `ENTRY`, proving the perf-map range and codemap agree on the block.
    /// (Line formatting is covered by `perfmap::tests::format_line_*`; end-to-end
    /// file emission by the bench under `X86JIT_PERF_MAP=1`.)
    #[test]
    fn perfmap_range_matches_codemap() {
        let mem = mem_with_code();
        let jit = JitBackend::new();
        let ir = lift_block(
            &mem,
            x86jit_core::lift::FetchAddr::flat(ENTRY),
            CpuMode::Long64,
        )
        .unwrap();
        let entry = compiled_entry(jit.materialize(&ir, MemConsistency::Fast, None, 0));
        let start = entry.0 as usize;
        // Some host offset inside the compiled block must resolve, via codemap, to
        // the same guest RIP (ENTRY) the perf-map symbol `jit_0x1000` names. Scan a
        // small window — the first srcloc need not be at offset 0.
        let hit = (0..64).find_map(|off| x86jit_core::codemap::lookup(start + off));
        assert_eq!(
            hit,
            Some(ENTRY),
            "codemap covers the compiled block at the guest RIP perfmap names it by"
        );
    }

    /// Dropping with requests queued/mid-compile joins the worker cleanly
    /// (the test hangs on a leaked thread, panics on a double-free — neither here).
    #[test]
    fn drop_joins_with_work_queued() {
        let mem = mem_with_code();
        let jit = JitBackend::new();
        for _ in 0..20 {
            let _ = jit.tier_up_async(request(&mem));
        }
        drop(jit); // must join without hanging
    }

    /// A poisoned queue mutex (a stand-in for a panicked worker) must not
    /// make `Drop` re-panic — a dead worker only means blocks stay interpreted.
    #[test]
    fn drop_survives_poisoned_mutex() {
        let mem = mem_with_code();
        let jit = JitBackend::new();
        let _ = jit.tier_up_async(request(&mem)); // spawn the worker
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = jit.shared.background.queue.lock().unwrap();
            panic!("poison the queue mutex");
        }));
        drop(jit); // must not re-panic despite the poisoned mutex
    }

    /// A panic while the JIT module lock is held must not turn every later JIT entry
    /// into a `PoisonError` panic — the cascade hides the defect that caused it.
    ///
    /// This poisons the same mutex directly rather than panicking inside codegen: the
    /// property under test is what happens *after* poisoning, and reaching it through
    /// a real compile would need a fault-injection hook in the hot path for no extra
    /// confidence.
    #[test]
    fn a_poisoned_jit_lock_does_not_cascade() {
        let jit = JitBackend::new();
        let shared = jit.shared.clone();
        let panicked = std::thread::spawn(move || {
            let _g = shared.inner.lock().unwrap();
            panic!("simulated compile panic while holding the JIT module lock");
        })
        .join();
        assert!(panicked.is_err(), "the helper thread was supposed to panic");
        assert!(
            jit.shared.inner.is_poisoned(),
            "the mutex should be poisoned now — otherwise this test proves nothing"
        );

        // The real assertion: the next entry works instead of panicking.
        let guard = jit.shared.lock_module(0);
        drop(guard);
    }
}
