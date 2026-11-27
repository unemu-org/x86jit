//! Compiled Machine blocks (decision-29, J1): the helpers their loads and stores
//! call, and which instructions they complete themselves.
//!
//! A compiled Machine block is a faster producer of the interpreted walk's
//! result. It completes a prefix of its block and returns `RET_CONTINUE`,
//! [`RET_MACHINE_DEFER`](super::RET_MACHINE_DEFER) or
//! [`RET_MACHINE_STOP`](super::RET_MACHINE_STOP) with the completed count in
//! [`MemCtx::machine_ordinal`](super::MemCtx::machine_ordinal); the dispatcher
//! charges it and, after a defer, walks the rest of the block on the interpreter.
use super::MemCtx;
use crate::ir::IrOp;
use crate::state::CpuState;
use std::sync::atomic::{AtomicU64, Ordering};

/// Marks a run-time defer (an access helper's or an inline check's) in
/// `MemCtx::machine_ordinal`; an unconditional defer leaves it clear.
pub const HELPER_DEFER: u64 = 1 << 63;

/// Helper status: the access completed.
pub const ACCESS_DONE: u64 = 0;
/// Helper status: nothing happened; the block defers the instruction.
pub const ACCESS_DEFER: u64 = 1;
/// Store status: the store completed and the vCPU's dispatch word is set, so
/// the block stops after this instruction, as the interpreted walk does
/// (`BlockPass::commit`).
pub const ACCESS_STOP: u64 = 2;

/// Whether a J1 compiled block lowers this op itself.
pub fn lowered_op(op: &IrOp) -> bool {
    crate::interp::legacy::machine_jit::lowered_op(op)
}

/// Whether a J1 compiled block completes this instruction itself, given its ops
/// after the `InsnStart` marker. Every other instruction is an unconditional
/// defer at its start.
pub fn lowered_instruction(ops: &[IrOp]) -> bool {
    crate::interp::legacy::machine_jit::lowered_instruction(ops)
}

/// The helpers' baked access configuration: the model's physical mask, the
/// run's host-fault routing and its memory-consistency tier.
pub fn access_config(physical_mask: u32, routed: bool, order: u8) -> u64 {
    u64::from(physical_mask) | (u64::from(routed) << 32) | (u64::from(order) << 33)
}

fn split(config: u64) -> (u32, bool, u8) {
    (
        config as u32,
        config & (1 << 32) != 0,
        (config >> 33) as u8 & 3,
    )
}

/// Machine load helper: [`ACCESS_DONE`] with the value in `MemCtx::machine_value`,
/// or [`ACCESS_DEFER`].
///
/// # Safety
/// `cpu` and `mem` are the live `CpuState` and `MemCtx` of a Machine compiled
/// call, whose `mem_self` names the run's `Memory`.
pub unsafe extern "C" fn machine_load(
    cpu: *mut u8,
    mem: *mut u8,
    addr: u64,
    size: u64,
    config: u64,
    _time_enabled: u64,
) -> u64 {
    // SAFETY: the compiled-block ABI supplies these live, per-vCPU pointers.
    let (cpu, ctx) = unsafe { (&mut *(cpu as *mut CpuState), &mut *(mem as *mut MemCtx)) };
    // SAFETY: `for_memory` retains the Memory for the whole compiled call.
    let memory = unsafe { &*(ctx.mem_self as *const crate::Memory) };
    let (mask, routed, order) = split(config);
    match crate::interp::legacy::machine_jit::access(
        cpu, memory, mask, routed, order, addr, size as u8, None,
    ) {
        crate::interp::legacy::machine_jit::Access::Done(value) => {
            ctx.machine_value = value;
            ACCESS_DONE
        }
        crate::interp::legacy::machine_jit::Access::Defer => ACCESS_DEFER,
    }
}

/// Machine store helper: [`ACCESS_DONE`], [`ACCESS_STOP`] or [`ACCESS_DEFER`].
///
/// # Safety
/// As [`machine_load`].
pub unsafe extern "C" fn machine_store(
    cpu: *mut u8,
    mem: *mut u8,
    addr: u64,
    size: u64,
    value: u64,
    config: u64,
    _time_enabled: u64,
) -> u64 {
    // SAFETY: the compiled-block ABI supplies these live, per-vCPU pointers.
    let (cpu, ctx) = unsafe { (&mut *(cpu as *mut CpuState), &mut *(mem as *mut MemCtx)) };
    // SAFETY: `for_memory` retains the Memory for the whole compiled call.
    let memory = unsafe { &*(ctx.mem_self as *const crate::Memory) };
    let (mask, routed, order) = split(config);
    match crate::interp::legacy::machine_jit::access(
        cpu,
        memory,
        mask,
        routed,
        order,
        addr,
        size as u8,
        Some(value),
    ) {
        crate::interp::legacy::machine_jit::Access::Done(_) => {
            // SAFETY: never null; a Machine run points it at its exit word.
            let word = unsafe { &*(ctx.dispatch_word_ptr as *const AtomicU64) };
            if word.load(Ordering::Relaxed) != 0 {
                ACCESS_STOP
            } else {
                ACCESS_DONE
            }
        }
        crate::interp::legacy::machine_jit::Access::Defer => ACCESS_DEFER,
    }
}
