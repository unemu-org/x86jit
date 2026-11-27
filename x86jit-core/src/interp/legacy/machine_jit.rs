//! The interpreter's half of compiled Machine blocks (decision-29, J1): which
//! instructions a compiled block may complete itself, the checked access its
//! loads and stores call, and the walk of a block's remaining instructions after
//! a compiled block deferred one.
use super::*;

/// An op a compiled Machine block lowers itself (decision-29 §4 condition 1,
/// widened by decision-30 §3). None of them sets an [`Outcome`] field in the
/// interpreter (`plain_outcome` below, asserted in the walk), so an
/// instruction made only of these ops completes with the default outcome in
/// both tiers.
pub(crate) fn lowered_op(op: &IrOp) -> bool {
    let gpr = |reg: &crate::Reg| reg.gpr_slot().is_some();
    let width = |size: &u8| matches!(size, 1 | 2 | 4);
    match op {
        IrOp::ReadReg { reg, .. } => {
            gpr(reg)
                || matches!(
                    reg,
                    crate::Reg::Cs | crate::Reg::Ds | crate::Reg::Es | crate::Reg::Ss
                )
        }
        IrOp::WriteReg { reg, .. } => gpr(reg),
        // MOV r/m, Sreg: the selector, read live (decision-30 §3).
        IrOp::ReadSelector { .. } => true,
        IrOp::Load { size, .. } | IrOp::Store { size, .. } => width(size),
        // A 16-bit SHLD/SHRD count above 16 leaves the result undefined (SDM
        // Vol 2B SHRD, "Operation"); the shared lowering's answer differs from
        // the interpreter's, found by the J1 lockstep, so it stays interpreted.
        IrOp::DoubleShift { size, .. } => *size == 4,
        IrOp::Div { size, .. } => width(size),
        IrOp::Add { .. }
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
        | IrOp::Sext { .. }
        | IrOp::Mul { .. }
        | IrOp::GetCond { .. }
        | IrOp::BitScan { .. }
        | IrOp::Bt { .. }
        | IrOp::Bswap { .. }
        | IrOp::SetCf { .. }
        | IrOp::SetDf { .. }
        | IrOp::Lahf
        | IrOp::Sahf
        | IrOp::Jump { .. }
        | IrOp::Branch { .. } => true,
        IrOp::Legacy(
            L::Flat32Address { .. }
            | L::Checkpoint
            | L::LiveField { .. }
            | L::ReadSelector { .. }
            | L::Loop { .. },
        ) => true,
        _ => false,
    }
}

/// Whether a lowered op can end its instruction with a defer in compiled code
/// (decision-30 §3): an access, the 4 GiB operand check, a divide error. A
/// Flat32 jump, branch or loop cannot fail: its target is masked to the
/// operand size and CS's limit is 0xFFFFFFFF.
fn fallible(op: &IrOp) -> bool {
    matches!(
        op,
        IrOp::Load { .. }
            | IrOp::Store { .. }
            | IrOp::Div { .. }
            | IrOp::Legacy(L::Flat32Address { .. })
    )
}

/// Whether a lowered op changes architectural state: registers, flags,
/// memory or RIP. The rest write temporaries only.
fn commits(op: &IrOp) -> bool {
    use crate::ir::FlagMask;
    match op {
        IrOp::ReadReg { .. }
        | IrOp::ReadSelector { .. }
        | IrOp::Load { .. }
        | IrOp::Div { .. }
        | IrOp::Sext { .. }
        | IrOp::GetCond { .. }
        | IrOp::Bswap { .. }
        | IrOp::Legacy(
            L::Flat32Address { .. } | L::Checkpoint | L::LiveField { .. } | L::ReadSelector { .. },
        ) => false,
        IrOp::Add { set_flags, .. }
        | IrOp::Adc { set_flags, .. }
        | IrOp::Sub { set_flags, .. }
        | IrOp::Sbb { set_flags, .. }
        | IrOp::And { set_flags, .. }
        | IrOp::Or { set_flags, .. }
        | IrOp::Xor { set_flags, .. }
        | IrOp::Shl { set_flags, .. }
        | IrOp::Shr { set_flags, .. }
        | IrOp::Sar { set_flags, .. }
        | IrOp::Rol { set_flags, .. }
        | IrOp::Ror { set_flags, .. }
        | IrOp::Rcl { set_flags, .. }
        | IrOp::Rcr { set_flags, .. }
        | IrOp::DoubleShift { set_flags, .. }
        | IrOp::Mul { set_flags, .. } => *set_flags != FlagMask::NONE,
        _ => true,
    }
}

/// Whether a compiled block completes this instruction itself, given its ops
/// after the `InsnStart` marker (decision-30 §3, the compiled-fallibility
/// rule): every op lowered, no commit before an op that can defer, and at
/// most one store. A deferred instruction has then committed nothing when
/// the interpreter re-executes it, so its `Checkpoint`, if any, is not
/// needed by the compiled code.
pub(crate) fn lowered_instruction(ops: &[IrOp]) -> bool {
    let mut committed = false;
    let mut stored = false;
    for op in ops {
        if !lowered_op(op) || (committed && fallible(op)) {
            return false;
        }
        if matches!(op, IrOp::Store { .. }) {
            if stored {
                return false;
            }
            stored = true;
        }
        committed |= commits(op);
    }
    true
}

/// The outcome an instruction made only of lowered ops leaves.
pub(super) fn plain_outcome(outcome: &Outcome) -> bool {
    outcome.exit.is_none()
        && !outcome.context_changed
        && !outcome.translations_invalidated
        && !outcome.arm_sti
        && outcome.arm_ss == 0
        && !outcome.suppress_debug
        && !outcome.preserve_rf
        && !outcome.software_interrupt
}

/// What the compiled access helper reports.
pub(crate) enum Access {
    /// The access completed; a read's value.
    Done(u64),
    /// Nothing happened: the interpreter re-executes the instruction.
    Defer,
}

/// The J1 compiled access (decision-29 §4, "Access helper"): the walk's own
/// unpaged `read`/`write` over `LegacyAccess`, for a transfer whose every bus
/// part is RAM (or ROM, for a read), and for a store, one bus part in a VM
/// without host-fault recovery. Anything else (a device, a hole, a ROM write,
/// a store split across bus parts) is
/// refused before any effect, so the interpreter's re-execution produces the
/// exit, the retry ledger and the fault itself, and a deferred instruction
/// has written nothing.
///
/// `ip` is the instruction's own IP, which the in-block device path reports.
/// The helper runs with `cpu.rip` at the block's start, not at the
/// instruction: the walk's `read`/`write` key their MMIO retry ledger by RIP,
/// so the entry contract (E5) guarantees there is no ledger to consult.
#[allow(clippy::too_many_arguments)]
pub(crate) fn access(
    cpu: &mut CpuState,
    memory: &Memory,
    mask: u32,
    routed: bool,
    order: u8,
    addr: u64,
    size: u8,
    store: Option<u64>,
    ip: u64,
) -> Access {
    debug_assert!(
        !cpu.mmio_retry_active
            && cpu.mmio_parts_len == 0
            && cpu.pending_mmio.is_none()
            && !cpu.pending_mmio_write
            && cpu.pending_port_in.is_none(),
        "a compiled Machine access with transfer retry state"
    );
    let ok = |physical: u64, bytes: u8| match memory.region_kind(physical, bytes) {
        Ok(crate::RegionKind::Ram) => true,
        Ok(crate::RegionKind::Rom) => store.is_none(),
        _ => false,
    };
    let (physical, part) = legacy::physical_span(cpu, mask, addr, size);
    if !ok(physical, part) {
        // Decision-30 §4: a whole transfer to a device served in process
        // completes here when the decision-28 in-block contract allows it;
        // any other result stays in the run's ledger, and the defer's
        // re-execution settles it without calling the handler again.
        if part == size && !routed {
            if let Some(value) = crate::vm::device::in_block_access(
                std::ptr::null(),
                ip,
                store.is_some(),
                physical,
                size,
                store.unwrap_or(0),
            ) {
                return Access::Done(value);
            }
        }
        return Access::Defer;
    }
    // A store in a VM with host-fault recovery could fault part-way through
    // its raw write; it defers, so no store is ever committed twice.
    if store.is_some() && routed {
        return Access::Defer;
    }
    if part != size {
        if store.is_some() {
            return Access::Defer;
        }
        let next = u64::from((addr as u32).wrapping_add(u32::from(part)));
        let (physical, rest) = legacy::physical_span(cpu, mask, next, size - part);
        if rest != size - part || !ok(physical, rest) {
            return Access::Defer;
        }
    }
    macro_rules! go {
        ($routed:literal, $order:literal) => {{
            let mem = LegacyAccess::<$routed, $order>::new(memory, mask);
            match store {
                Some(value) => write(cpu, &mem, addr, size, value).map(|()| 0),
                None => read(cpu, &mem, addr, size),
            }
        }};
    }
    let result = match (routed, order) {
        (false, 0) => go!(false, 0),
        (false, 1) => go!(false, 1),
        (false, _) => go!(false, 2),
        (true, 0) => go!(true, 0),
        (true, 1) => go!(true, 1),
        (true, _) => go!(true, 2),
    };
    // Only a host protection fault (`MemTrap::HostFault`) can still refuse a RAM
    // or ROM part, and only a read can reach one: stores defer in VMs with
    // host-fault recovery, and without it a host fault is not recoverable. A
    // refused read has no effect but RIP (`trap_out`), which the defer exit
    // overwrites; the re-execution faults itself.
    debug_assert!(
        result.is_ok() || (routed && store.is_none()),
        "a preflighted RAM or ROM access trapped: {result:?}"
    );
    result.map_or(Access::Defer, Access::Done)
}

/// The rest of an ordinary block from its instruction at op index `from`, with
/// `limit` steps (decision-29 §4, `RET_MACHINE_DEFER`). The same walk the block
/// takes interpreted, entered at an instruction boundary.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_suffix<const ROUTED: bool, const ORDER: u8>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    memory: &Memory,
    scratch: &mut impl ScratchBuffer,
    limit: u64,
    exit_word: &std::sync::atomic::AtomicU64,
    task_switch: &mut tasks::SwitchState,
    from: usize,
) -> Progress {
    execute_instructions_from::<ROUTED, ORDER>(
        ir,
        cpu,
        memory,
        scratch,
        limit,
        exit_word,
        task_switch,
        from,
    )
}

/// The op index of instruction `ordinal` (its `InsnStart`).
pub(crate) fn instruction_op(ir: &IrBlock, ordinal: u64) -> usize {
    ir.ops
        .iter()
        .enumerate()
        .filter(|(_, op)| matches!(op, IrOp::InsnStart { .. }))
        .nth(ordinal as usize)
        .map(|(at, _)| at)
        .expect("a deferred ordinal inside its block")
}

/// The walk result of `count` completed lowered instructions: the default
/// outcome of the last, as `execute_instructions` reports it.
pub(crate) fn completed(count: u64) -> Progress {
    Progress {
        sti_shadow: false,
        ss_shadow: 0,
        steps: count,
        retired: count,
        outcome: Some(Ok(Outcome {
            retired: true,
            ..Outcome::default()
        })),
    }
}

/// Whether a walk result is `count` completed instructions with the default
/// outcome: what a compiled prefix of that length reports.
pub(crate) fn is_completed(progress: &Progress, count: u64) -> bool {
    progress.steps == count
        && progress.retired == count
        && !progress.sti_shadow
        && progress.ss_shadow == 0
        && matches!(&progress.outcome, Some(Ok(outcome)) if outcome.retired && plain_outcome(outcome))
}

/// How a deferred suffix ended, for the exit-reason census.
pub(crate) fn suffix_kind(progress: &Progress) -> usize {
    match &progress.outcome {
        Some(Ok(outcome)) if outcome.exit.is_some() => 1,
        Some(Ok(_)) | None => 0,
        Some(Err(Exit::MmioRead { .. } | Exit::MmioWrite { .. })) => 2,
        Some(Err(Exit::Exception { .. })) => 3,
        Some(Err(_)) => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{FlagMask, MemOrder, MemoryAccess, Val};

    fn store() -> IrOp {
        IrOp::Store {
            fault: MemoryAccess::new(4, false, false),
            addr: Val::Temp(0),
            src: Val::Temp(1),
            size: 4,
            order: MemOrder::None,
        }
    }

    fn load() -> IrOp {
        IrOp::Load {
            fault: MemoryAccess::new(4, false, false),
            dst: 1,
            addr: Val::Temp(0),
            size: 4,
        }
    }

    fn add(set_flags: FlagMask) -> IrOp {
        IrOp::Add {
            dst: 2,
            a: Val::Temp(1),
            b: Val::Imm(1),
            size: 4,
            set_flags,
        }
    }

    fn write_eax() -> IrOp {
        IrOp::WriteReg {
            reg: crate::Reg::Rax,
            src: Val::Temp(2),
            size: 4,
        }
    }

    /// Decision-30 §3, the compiled-fallibility rule, on the shapes it
    /// refuses: a register or flag commit before an access, a second store.
    /// The lockstep's commit-before-fault programs (POP to MMIO and ROM,
    /// LEAVE from MMIO, RCL and ADC on ROM) lift without these shapes: the
    /// lifter defers flag writes past an instruction's store and writes ESP
    /// last. The rule keeps a lift that has one exact.
    #[test]
    fn a_commit_before_an_op_that_can_defer_stays_interpreted() {
        assert!(lowered_instruction(&[
            load(),
            add(FlagMask::NONE),
            store(),
            write_eax()
        ]));
        assert!(lowered_instruction(&[
            IrOp::Legacy(L::Checkpoint),
            load(),
            add(FlagMask::NONE),
            store(),
            write_eax(),
        ]));
        assert!(!lowered_instruction(&[write_eax(), store()]));
        assert!(!lowered_instruction(&[write_eax(), load()]));
        assert!(!lowered_instruction(&[load(), add(FlagMask::ALL), store()]));
        assert!(!lowered_instruction(&[store(), load()]));
        assert!(!lowered_instruction(&[store(), store()]));
        assert!(!lowered_instruction(&[
            IrOp::SetCf { value: Some(true) },
            IrOp::Legacy(L::Flat32Address {
                dst: 0,
                segment: crate::legacy::SegmentId::Ds,
                offset: Val::Imm(0),
                bytes: 4,
                access: crate::AccessKind::Read,
            }),
        ]));
    }
}
