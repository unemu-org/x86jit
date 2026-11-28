//! Compiled Machine blocks in the Machine turn (decision-29, J1): the entry
//! contract, the call, and the conversion of its return into the walk result
//! the interpreted walk of the same block returns. Admitted only by the
//! test-only [`MachineJitMode`]; production runs never reach this module.
use super::*;
use crate::code_lifetime::MachineTarget;
use crate::interp::legacy::machine_jit;
use crate::interp::legacy::Progress;
use crate::jit_abi::{
    MemCtx, RET_CHAIN, RET_CONTINUE, RET_LINK, RET_MACHINE_DEFER, RET_MACHINE_STOP, RET_RETURN_MISS,
};

use crate::jit_abi::machine::HELPER_DEFER;

/// Which compiled Machine units a VM admits (decision-29 §9). `Off` is the only
/// value production code can hold; the others are set by
/// `Vm::set_machine_jit_for_test` or, with `test-hooks`, the
/// `X86JIT_TEST_MACHINE_JIT` environment variable (`forced`, `delayed:N`,
/// `scripted:SEED`).
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MachineJitMode {
    /// No compiled Machine unit is ever entered.
    #[default]
    Off,
    /// Mode 2: every eligible block compiles synchronously at its first
    /// execution and is entered from then on.
    Forced,
    /// Mode 3: as `Forced`, but a unit is entered only after this many more
    /// eligible turns of its block: publication lands mid-run.
    Delayed(u32),
    /// Mode 4: no generated code. A scripted unit completes a prefix of the
    /// block on the interpreter and returns CONTINUE, DEFER(c) or STOP(c)
    /// chosen from this seed, at every ordinal over a run.
    Scripted(u64),
}

impl MachineJitMode {
    /// The mode named by `X86JIT_TEST_MACHINE_JIT`, read once per process.
    pub(crate) fn from_environment() -> Self {
        #[cfg(all(feature = "environment", any(test, feature = "test-hooks")))]
        {
            static MODE: std::sync::OnceLock<MachineJitMode> = std::sync::OnceLock::new();
            *MODE.get_or_init(|| {
                let Some(value) = crate::environment::test_machine_jit() else {
                    return MachineJitMode::Off;
                };
                let (name, arg) = value.split_once(':').unwrap_or((value.as_str(), ""));
                match name {
                    "forced" => MachineJitMode::Forced,
                    "delayed" => MachineJitMode::Delayed(arg.parse().unwrap_or(1)),
                    "scripted" => MachineJitMode::Scripted(arg.parse().unwrap_or(1)),
                    _ => MachineJitMode::Off,
                }
            })
        }
        #[cfg(not(all(feature = "environment", any(test, feature = "test-hooks"))))]
        MachineJitMode::Off
    }
}

/// Compiled-Machine counters (decision-29 §10 item 2): what compiled units did,
/// so that a forced witness can show it was not vacuous.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MachineJitCounters {
    /// Units this vCPU published: compiled synchronously (modes 2-3), or
    /// compiled by the background worker (production tier-up) and published
    /// when this vCPU drained the completion (`drain`). A unit another vCPU
    /// published, entered here through the shared map, is not counted.
    pub compiled: u64,
    /// Compiled (or scripted) units entered.
    pub entries: u64,
    /// Instructions completed by compiled (or scripted) code.
    pub compiled_retired: u64,
    /// `RET_CONTINUE` returns.
    pub continues: u64,
    /// `RET_MACHINE_STOP` returns.
    pub stops: u64,
    /// `RET_MACHINE_DEFER` returns at an instruction the lowering does not complete.
    pub unconditional_defers: u64,
    /// `RET_MACHINE_DEFER` returns from an access helper.
    pub helper_defers: u64,
    /// Defers at ordinal 0.
    pub defers_at_zero: u64,
    /// How the deferred suffixes ended: completed, a handoff or exit outcome,
    /// an MMIO exit, an exception, another exit.
    pub suffix: [u64; 5],
    /// Turns of an admitted block (E2) that ran the interpreted walk because
    /// another entry condition (E3-E8) failed.
    pub refused: u64,
    /// Units retired: released blocks, a cleared map, a changed code lifetime.
    pub retired: u64,
    /// Compiled-to-compiled transfers without a turn (decision-31).
    pub chained: u64,
    /// Why each compiled turn's last block was the last, indexed by the
    /// `CHAIN_END_*` constants (decision-31 §8 done-when 1).
    pub chain_ends: [u64; CHAIN_ENDS],
    /// Run-time defers by cause (decision-32 §9 done-when 1), named by
    /// [`Self::DEFER_CAUSE_NAMES`].
    pub defer_causes: [u64; machine_jit::CAUSES],
    /// Instructions completed by compiled (or scripted) code at CPL 3.
    pub compiled_retired_cpl3: u64,
    /// Compiled (or scripted) entries with paging on.
    pub paged_entries: u64,
    /// Turns that ran the interpreted walk because an entry condition failed,
    /// by the first failing condition, named by [`Self::REFUSAL_NAMES`]: the
    /// class tests (E2; a block outside the ordinary walk by whether it holds
    /// a string instruction, a port access, or neither), then E3-E10 in the
    /// order `machine_progress` tests them. The E3-E10 entries, from
    /// [`Self::FIRST_ENTRY_REFUSAL`] on, sum to [`Self::refused`].
    pub refusals: [u64; REFUSALS],
    /// Turns of an admitted, eligible block that ran the interpreted walk
    /// because it had no unit to enter, named by [`Self::NO_UNIT_NAMES`].
    pub no_unit: [u64; NO_UNITS],
    /// The instructions the interpreted walks of [`Self::refusals`]' turns
    /// retired, row by row. Counted, as `refusals` is, only with test-hooks;
    /// zero without.
    pub refusals_retired: [u64; REFUSALS],
    /// The instructions the interpreted walks of [`Self::no_unit`]'s turns
    /// retired, row by row. Counted, as `no_unit` is, only with test-hooks;
    /// zero without.
    pub no_unit_retired: [u64; NO_UNITS],
}

/// Entry refusals (`MachineJitCounters::refusals`): E2's class tests, then
/// E3 by its cause, E4, E5 by its input, E6-E8, and the paged entry's three
/// tests (decision-32 §2).
// Read by the test-only census alone (`Vcpu::refusal`, `walk_refusal`).
#[cfg_attr(not(any(test, feature = "test-hooks")), allow(dead_code))]
mod refuse {
    pub(super) const REFUSE_SEGMENT_CLASS: usize = 0;
    pub(super) const REFUSE_CPL: usize = 1;
    pub(super) const REFUSE_WALK_STRING: usize = 2;
    pub(super) const REFUSE_WALK_PORT: usize = 3;
    pub(super) const REFUSE_WALK: usize = 4;
    pub(super) const REFUSE_TRAP_FLAG: usize = 5;
    pub(super) const REFUSE_SHADOW: usize = 6;
    pub(super) const REFUSE_IRQ_PENDING: usize = 7;
    pub(super) const REFUSE_DEVICE_INTR: usize = 8;
    pub(super) const REFUSE_ONE_STEP_OTHER: usize = 9;
    pub(super) const REFUSE_BUDGET: usize = 10;
    pub(super) const REFUSE_RETRY: usize = 11;
    pub(super) const REFUSE_TASK: usize = 12;
    pub(super) const REFUSE_PENDING_IO: usize = 13;
    pub(super) const REFUSE_FETCH_POLICY: usize = 14;
    pub(super) const REFUSE_MXCSR: usize = 15;
    pub(super) const REFUSE_RF: usize = 16;
    pub(super) const REFUSE_PAGE_CROSSING: usize = 17;
    pub(super) const REFUSE_PAGED_BUSY: usize = 18;
    pub(super) const REFUSE_PAGED_PROOF: usize = 19;
    pub(super) const REFUSALS: usize = 20;
}
use refuse::*;

/// Why an eligible block had no unit (`MachineJitCounters::no_unit`).
const NO_UNIT_COLD: usize = 0;
const NO_UNIT_AWAITING: usize = 1;
const NO_UNIT_FAILED: usize = 2;
const NO_UNIT_DEMOTED: usize = 3;
const NO_UNIT_DELAYED: usize = 4;
/// Counting toward T1 again after the backend refused the request as busy.
const NO_UNIT_BUSY: usize = 5;
const NO_UNITS: usize = 6;

/// Test-only: which E2 walk refusal a block outside the ordinary walk is.
#[cfg(any(test, feature = "test-hooks"))]
#[cold]
#[inline(never)]
fn walk_refusal(ir: &IrBlock) -> usize {
    use crate::ir::IrOp;
    use crate::legacy::LegacyOp as L;
    let has = |f: fn(&IrOp) -> bool| ir.ops.iter().any(f);
    if has(|op| matches!(op, IrOp::Legacy(L::String { .. }))) {
        REFUSE_WALK_STRING
    } else if has(|op| {
        matches!(
            op,
            IrOp::PortIo { .. }
                | IrOp::PortString { .. }
                | IrOp::Legacy(L::Privileged {
                    kind: crate::legacy::PrivilegeCheck::IoPort { .. }
                        | crate::legacy::PrivilegeCheck::RepIoPort { .. }
                })
        )
    }) {
        REFUSE_WALK_PORT
    } else {
        REFUSE_WALK
    }
}

/// Test-only tally of a refusal (`MachineJitCounters::refusals`), and the
/// row the turn's walk retires into (`Compiling::row`).
#[cfg(any(test, feature = "test-hooks"))]
#[inline(always)]
fn tally_refusal(state: &mut Compiling, index: usize) {
    state.counters.refusals[index] += 1;
    state.row = Some(index as u8);
}

/// Test-only tally of a missing unit (`MachineJitCounters::no_unit`) in
/// `$counters`, and the row the turn's walk retires into in `$row`
/// (`Compiling::row`), which exists only with test-hooks.
macro_rules! tally_no_unit {
    ($counters:expr, $row:expr, $index:expr) => {
        #[cfg(any(test, feature = "test-hooks"))]
        {
            $counters.no_unit[$index] += 1;
            *$row = Some((REFUSALS + $index) as u8);
        }
        #[cfg(not(any(test, feature = "test-hooks")))]
        let _ = (&$counters, $index);
    };
}

/// Chain-end reasons (decision-31 §8): the edge's budget test, the dispatch
/// word, the execution epoch, the flags compare, an empty slot (`RET_LINK`),
/// a return-ring miss, an unchained exit (`RET_CONTINUE`, or a `RET_CHAIN`
/// with no chain started: chains off or the cold epilogue), DEFER, STOP, and
/// a paged target outside the chain's page or frame (decision-32 §4).
const CHAIN_END_BUDGET: usize = 0;
const CHAIN_END_WORD: usize = 1;
const CHAIN_END_EPOCH: usize = 2;
const CHAIN_END_FLAGS: usize = 3;
const CHAIN_END_EMPTY_SLOT: usize = 4;
const CHAIN_END_RETURN_MISS: usize = 5;
const CHAIN_END_UNCHAINED: usize = 6;
const CHAIN_END_DEFER: usize = 7;
const CHAIN_END_STOP: usize = 8;
/// Decision-32 §4: a paged chain stays inside its first block's page.
const CHAIN_END_PAGE: usize = 9;
const CHAIN_ENDS: usize = 10;

impl MachineJitCounters {
    /// The names of [`Self::defer_causes`]' entries, in order.
    pub const DEFER_CAUSE_NAMES: [&'static str; machine_jit::CAUSES] = [
        "tlb miss",
        "rights",
        "clean write",
        "page crossing",
        "device or hole",
        "ac gate",
        "segment",
        "other",
    ];

    /// The names of [`Self::refusals`]' entries, in order.
    pub const REFUSAL_NAMES: [&'static str; REFUSALS] = [
        "E2 segment class",
        "E2 cpl 1 or 2",
        "E2 walk: string",
        "E2 walk: port io",
        "E2 walk: other",
        "E3 trap flag",
        "E3 sti or ss shadow",
        "E3 irq pending",
        "E3 device intr",
        "E3 other one-step",
        "E4 budget",
        "E5 retry",
        "E5 task switch",
        "E5 pending io",
        "E6 fetch policy",
        "E7 mxcsr",
        "E8 rf",
        "E9 page crossing",
        "E9 paged transaction",
        "E10 paged proof",
    ];

    /// The index of the first E3-E10 entry of [`Self::refusals`].
    pub const FIRST_ENTRY_REFUSAL: usize = REFUSE_TRAP_FLAG;

    /// The names of [`Self::no_unit`]'s entries, in order.
    pub const NO_UNIT_NAMES: [&'static str; NO_UNITS] = [
        "cold",
        "awaiting",
        "failed",
        "demoted",
        "delayed",
        "busy recount",
    ];

    /// The names of [`Self::chain_ends`]' entries, in order.
    pub const CHAIN_END_NAMES: [&'static str; CHAIN_ENDS] = [
        "budget",
        "word",
        "epoch",
        "flags",
        "empty slot",
        "return miss",
        "unchained",
        "defer",
        "stop",
        "page",
    ];
}

/// The blocks a compiled turn completed before its last one (decision-31
/// §2.3): the dispatcher charges `prefix` first, then handles the result as
/// that of `ir`, the last block.
pub(super) struct Chain {
    pub(super) prefix: u64,
    pub(super) ir: Arc<IrBlock>,
}

/// A slot a `RET_LINK` left to fill (decision-31 §2.4): published by the next
/// compiled entry of this vCPU when nothing retired in between, its block
/// starts at `target` under the same key flags, and `epoch` is still current.
#[derive(Clone, Copy)]
struct Fill {
    slot: u64,
    target: u64,
    flags: u32,
    epoch: u64,
    retired: u64,
}

/// Whether compiled Machine blocks chain (decision-31 §8: the test-only
/// chain-off switch, `Vm::set_machine_chains_for_test`).
#[inline(always)]
fn chains(vm: &Vm) -> bool {
    #[cfg(any(test, feature = "test-hooks"))]
    return !vm.machine_chain_off;
    #[cfg(not(any(test, feature = "test-hooks")))]
    {
        let _ = vm;
        true
    }
}

/// Test-only (decision-31 §8, chains broken mid-run): what to raise at a
/// chosen chain edge, before the edge's tests.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainEdgeAction {
    /// The host's exit request (`ExitRequest::request_exit`).
    ExitRequest,
    /// An NMI edge (`NmiLine::raise`).
    NmiEdge,
    /// A store by another vCPU into code at this bus address (its bytes
    /// unchanged): `DISPATCH_DIRTY` in every vCPU's word.
    CodeWrite(u64),
    /// Every link slot cleared and the execution epoch advanced, as a
    /// retirement does.
    Retire,
    /// The backend's code space ran out.
    CodeSpaceExhausted,
}

#[cfg(any(test, feature = "test-hooks"))]
thread_local! {
    // (edges to pass first, the action, the retired count it was raised at).
    static EDGE_HOOK: std::cell::Cell<(u64, Option<ChainEdgeAction>, Option<u64>)> =
        const { std::cell::Cell::new((0, None, None)) };
}

impl Vcpu {
    /// Test-only (decision-31 §8): raise `action` on this thread at the
    /// `edge`-th chain edge from now (0: the next one), once.
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn raise_at_chain_edge_for_test(edge: u64, action: ChainEdgeAction) {
        EDGE_HOOK.with(|hook| hook.set((edge, Some(action), None)));
    }

    /// Test-only: the retired-instruction count at which the action of
    /// [`Self::raise_at_chain_edge_for_test`] was raised, if it was.
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn chain_edge_raised_at_for_test() -> Option<u64> {
        EDGE_HOOK.with(|hook| hook.get().2)
    }
}

#[cfg(any(test, feature = "test-hooks"))]
#[cold]
#[inline(never)]
fn edge_hook(vm: &Vm, exit_word: &AtomicU64, retired: u64) {
    EDGE_HOOK.with(|hook| {
        let (edge, action, raised) = hook.get();
        let Some(action) = action else {
            return;
        };
        if edge != 0 {
            hook.set((edge - 1, Some(action), raised));
            return;
        }
        match action {
            ChainEdgeAction::ExitRequest => {
                exit_word.fetch_or(crate::vm::EXIT_REQUESTED, Ordering::Release);
            }
            ChainEdgeAction::NmiEdge => {
                exit_word.fetch_or(crate::vm::NMI_EDGE, Ordering::Release);
            }
            ChainEdgeAction::CodeWrite(address) => vm.mem.note_write(address, 1),
            ChainEdgeAction::Retire => vm.cache.code_lifetime().flush_predictions(),
            ChainEdgeAction::CodeSpaceExhausted => {
                exit_word.fetch_or(crate::vm::CODE_SPACE_EXHAUSTED, Ordering::Release);
            }
        }
        hook.set((0, None, Some(retired)));
    });
}

#[inline(always)]
fn census(counters: &mut MachineJitCounters, reason: usize) {
    #[cfg(any(test, feature = "test-hooks"))]
    {
        counters.chain_ends[reason] += 1;
    }
    #[cfg(not(any(test, feature = "test-hooks")))]
    let _ = (counters, reason);
}

/// A compiled walk result (decision-30 §6): either `count` lowered
/// instructions completed with the default outcome, which the dispatcher
/// charges on its fast arm, or the walk result of a deferred suffix.
pub(super) enum Compiled {
    /// `RET_CONTINUE` (`stop` false, `count` the block's) or `RET_MACHINE_STOP`.
    Done { count: u64, stop: bool },
    /// A `RET_MACHINE_DEFER`, merged with the interpreted suffix, or the
    /// interpreter's own result for a scripted prefix.
    Walk(Progress),
}

/// The walk result of `count` completed compiled instructions.
pub(super) fn completed(count: u64) -> Progress {
    machine_jit::completed(count)
}

/// Where a block's compiled unit stands, for one vCPU.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Request {
    /// Not requested: counting toward the tier-up threshold.
    None,
    /// Requested or published elsewhere: look the unit up again when the
    /// cache's Machine publication count moves past this value.
    Waiting(u64),
    /// The unit cannot be compiled (a deterministic failure, an unsupported
    /// backend): the block stays interpreted while this slot lives.
    Failed,
}

/// A block's compiled entry as this vCPU caches it: the slot a proof or a
/// link reaches through the block it returns (decision-30 §2, §6).
pub(super) struct Slot {
    // Keeps the block's address from being reused while it names this slot.
    // Weak, so that the memo does not make the block's proof look held
    // (`Arc::strong_count`, the proof maps' bound eviction): a compiling
    // vCPU's 256 slots would otherwise pin up to 256 proofs.
    ir: std::sync::Weak<IrBlock>,
    entry: Option<(crate::CompiledPtr, u64)>,
    hot: u32,
    request: Request,
    delay: u32,
    runs: u64,
    /// Decision-32 Q7: the shared Machine map was looked up once, on the
    /// slot's first eligible turn.
    looked: bool,
    /// The block's bus address, the key's `physical_pc`, once computed.
    bus: u64,
    /// The backend refused this slot's last request as busy, and the slot
    /// counts toward T1 again (the census's "busy recount").
    busy: bool,
}

/// A block's identity in the cold census: its bus address (`u64::MAX` if the
/// TLB holds none), its EIP and its key flags (`MachineBlockKey`'s parts that
/// vary within one run).
#[cfg(any(test, feature = "test-hooks"))]
#[doc(hidden)]
pub type ColdKey = (u64, u32, u32);

/// Test-only: one block key's history in [`ColdCensus`].
#[cfg(any(test, feature = "test-hooks"))]
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ColdKeyStats {
    /// Its "no_unit cold" turns.
    pub cold_turns: u64,
    /// The instructions those turns' interpreted walks retired.
    pub cold_retired: u64,
    /// Memo slots created for it: its first, and one after each conflict or
    /// relift that took its slot.
    pub slots: u64,
    /// The highest hotness one of its slots reached.
    pub max_hot: u32,
    /// Whether one of its slots reached the threshold or found a unit.
    pub reached: bool,
}

/// Test-only: where the "no_unit cold" turns of a vCPU went, by block key,
/// and what each new memo slot replaced. Recorded while
/// `Vcpu::enable_cold_census_for_test` is on.
#[cfg(any(test, feature = "test-hooks"))]
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct ColdCensus {
    pub keys: std::collections::HashMap<ColdKey, ColdKeyStats>,
    /// New slots over an empty index.
    pub slots_empty: u64,
    /// New slots over another live block's slot (a memo conflict).
    pub slots_conflict: u64,
    /// New slots over a dropped block's slot.
    pub slots_dropped: u64,
    /// Conflicts that evicted a slot without a unit and with a count.
    pub conflicts_cold: u64,
    /// Slot changes that took the block's slot back from the victim table.
    pub victims_restored: u64,
    pending: Option<ColdKey>,
}

/// Slots of the per-vCPU memo: a direct-mapped table keyed by the block a
/// proof or a link returned (decision-30 §6: no units lookup). A conflict
/// moves the evicted slot to the victim table (`save_victim`), from which
/// its block's next turn takes it back.
const MEMO: usize = 256;

#[inline(always)]
fn memo_index(ir: &Arc<IrBlock>) -> usize {
    let address = Arc::as_ptr(ir) as usize as u64;
    (address.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> (64 - MEMO.trailing_zeros())) as usize
}

/// The victim table (xj-to9.78): slots that memo conflicts evicted, kept for
/// their blocks. A working set wider than the memo would otherwise restart
/// every count a conflict evicts, and a block run fewer than T1 times between
/// two conflicts would never be compiled. A kept slot holds its block weakly,
/// as the memo does: a dropped proof leaves it dead, so the count still
/// restarts with a new proof (decision-27 §2.1 item 2, "stable").
/// `VICTIM_SETS` sets of `VICTIM_WAYS`, indexed by a hash of the block's
/// address independent of `memo_index`, so that the blocks that share a memo
/// slot spread over the sets.
///
/// Cost: 4,096 slots of at most 88 bytes, about 352 KiB per vCPU that
/// compiles, allocated at its first memo conflict and freed only when its
/// code lifetime changes (`State::begin_run`). The cache's byte accounting
/// does not count it. A kept slot of a dropped block is pruned when its set
/// is next scanned (`prune`); until then its weak reference keeps the
/// block's allocation, which the proof maps count as freed.
const VICTIM_SETS: usize = 512;
const VICTIM_WAYS: usize = 8;
const _: () = assert!(std::mem::size_of::<Option<Slot>>() <= 88);

#[inline(always)]
fn victim_set(ir: *const IrBlock) -> usize {
    let address = ir as usize as u64;
    let set = address.wrapping_mul(0xbf58_476d_1ce4_e5b9) >> (64 - VICTIM_SETS.trailing_zeros());
    set as usize * VICTIM_WAYS
}

/// What keeping a slot saves: nothing for a dead block's; little for one
/// with a unit, which a fresh slot finds again by one lookup (decision-32
/// Q7); its count, request or failure for one without, which a fresh slot
/// would restart.
fn victim_value(slot: &Slot) -> u32 {
    if slot.ir.strong_count() == 0 {
        0
    } else if slot.entry.is_some() {
        1
    } else {
        slot.hot.saturating_add(2)
    }
}

/// Drop the kept slots of dropped blocks in a set: their weak references
/// would keep the blocks' allocations, which the proof maps count as freed.
#[inline(always)]
fn prune(ways: &mut [Option<Slot>]) {
    for way in ways {
        if way.as_ref().is_some_and(|slot| slot.ir.strong_count() == 0) {
            *way = None;
        }
    }
}

/// Keep a slot a memo conflict evicts, over the way of its set that saves
/// the least, if it saves at least as much.
#[cold]
#[inline(never)]
fn save_victim(victims: &mut Option<Box<[Option<Slot>]>>, slot: Slot) {
    let victims =
        victims.get_or_insert_with(|| (0..VICTIM_SETS * VICTIM_WAYS).map(|_| None).collect());
    let set = victim_set(slot.ir.as_ptr());
    let ways = &mut victims[set..set + VICTIM_WAYS];
    prune(ways);
    let (way, least) = ways
        .iter()
        .enumerate()
        .map(|(i, way)| (i, way.as_ref().map_or(0, victim_value)))
        .min_by_key(|&(_, value)| value)
        .unwrap();
    if victim_value(&slot) >= least {
        ways[way] = Some(slot);
    }
}

/// The slot a memo conflict kept for `ir`, taken out of the table.
#[inline(always)]
fn take_victim(victims: &mut Option<Box<[Option<Slot>]>>, ir: &Arc<IrBlock>) -> Option<Slot> {
    let victims = victims.as_deref_mut()?;
    let set = victim_set(Arc::as_ptr(ir));
    let ways = &mut victims[set..set + VICTIM_WAYS];
    prune(ways);
    // A kept slot's weak reference keeps its block's allocation, so an equal
    // address is this very block, alive while `ir` holds it.
    ways.iter_mut()
        .find(|way| {
            way.as_ref()
                .is_some_and(|slot| std::ptr::eq(slot.ir.as_ptr(), Arc::as_ptr(ir)))
        })?
        .take()
}

/// One vCPU's view of the shared compiled Machine units (decision-30 §2).
/// The units belong to the VM's translation cache and its code lifetime; this
/// state caches their entries and holds the vCPU's execution participation,
/// entered around every call, so retired code is not reclaimed while it runs
/// (spec §9.1; decision-17).
/// A page where code writes dropped compiled units (decision-27 §2.5).
#[derive(Clone, Copy, Default)]
struct HotPage {
    page: u64,
    drops: u32,
    first: u64,
    last: u64,
    demoted: bool,
}

/// Demotion window, in this vCPU's retired instructions: two drops within it
/// stop compiling the page until it passes with no further drop. Retired
/// instructions keep the policy deterministic; it affects cost only.
const DEMOTION_WINDOW: u64 = 1 << 20;

#[derive(Default)]
pub(super) struct State {
    /// Allocated by the first run that compiles (`begin_run`): a vCPU that
    /// compiles no Machine code allocates nothing for it.
    pub(super) compiling: Option<Box<Compiling>>,
    lifetime: Option<Arc<crate::code_lifetime::CodeLifetime>>,
    participant: Option<crate::code_lifetime::ExecutionParticipant>,
}

/// The state only a run that compiles uses.
pub(super) struct Compiling {
    pub(super) counters: MachineJitCounters,
    memo: Box<[Option<Slot>]>,
    /// Slots that memo conflicts evicted (`save_victim`), allocated at the
    /// first such conflict.
    victims: Option<Box<[Option<Slot>]>>,
    hot_pages: [HotPage; 16],
    /// The compiled view of the TLB (decision-30 §4), at a stable address
    /// for `MemCtx::ret_stack` (its Machine meaning), and the mapping its entries were filled from.
    pub(super) tlb: crate::jit_abi::machine::MachineTlb,
    mapping: Option<Arc<()>>,
    fill: Option<Fill>,
    /// The epoch the shadow return ring's continuation slots belong to: a new
    /// epoch discards them (decision-31 §5), as Application's `fast_clear`.
    ring_epoch: u64,
    /// Test-only: the census row this turn went to, for its walk's retired
    /// count (`Vcpu::census_retired`): a refusal index, or `REFUSALS` plus a
    /// no-unit index.
    #[cfg(any(test, feature = "test-hooks"))]
    row: Option<u8>,
    #[cfg(any(test, feature = "test-hooks"))]
    cold: Option<Box<ColdCensus>>,
}

impl Default for Compiling {
    fn default() -> Self {
        Self {
            counters: MachineJitCounters::default(),
            memo: (0..MEMO).map(|_| None).collect(),
            victims: None,
            hot_pages: [HotPage::default(); 16],
            tlb: crate::jit_abi::machine::MachineTlb::default(),
            mapping: None,
            fill: None,
            ring_epoch: 0,
            #[cfg(any(test, feature = "test-hooks"))]
            row: None,
            #[cfg(any(test, feature = "test-hooks"))]
            cold: None,
        }
    }
}

impl Compiling {
    /// A drained code write dropped compiled units while this vCPU ran the
    /// code page `page` (decision-27 §2.5).
    #[cold]
    pub(super) fn note_code_drop(&mut self, page: u64, retired: u64) {
        let index = self
            .hot_pages
            .iter()
            .position(|p| p.drops != 0 && p.page == page)
            .unwrap_or_else(|| {
                (0..self.hot_pages.len())
                    .min_by_key(|&i| (self.hot_pages[i].drops != 0, self.hot_pages[i].last))
                    .unwrap()
            });
        let hot = &mut self.hot_pages[index];
        if hot.drops == 0
            || hot.page != page
            || retired.saturating_sub(hot.first) >= DEMOTION_WINDOW
        {
            *hot = HotPage {
                page,
                drops: 0,
                first: retired,
                last: retired,
                demoted: hot.page == page && hot.demoted,
            };
        }
        hot.drops += 1;
        hot.last = retired;
        hot.demoted |= hot.drops >= 2;
    }

    /// Whether `page` is demoted: it stays interpreted.
    fn demoted(&mut self, page: u64, retired: u64) -> bool {
        let Some(hot) = self
            .hot_pages
            .iter_mut()
            .find(|p| p.demoted && p.page == page)
        else {
            return false;
        };
        if retired.saturating_sub(hot.last) < DEMOTION_WINDOW {
            return true;
        }
        *hot = HotPage::default();
        false
    }
}

impl State {
    /// The share of a run that compiles: it exists.
    #[cfg(any(test, feature = "test-hooks"))]
    fn compiling(&mut self) -> &mut Compiling {
        self.compiling
            .as_deref_mut()
            .expect("compiled Machine state outside a compiling run")
    }

    /// The counters of a run that compiles: its share exists.
    pub(super) fn counters(&mut self) -> &mut MachineJitCounters {
        &mut self
            .compiling
            .as_deref_mut()
            .expect("compiled Machine counters outside a compiling run")
            .counters
    }

    /// A drained code write dropped compiled units while this vCPU ran the
    /// code page `page` (decision-27 §2.5).
    #[cold]
    pub(super) fn note_code_drop(&mut self, page: u64, retired: u64) {
        if let Some(compiling) = &mut self.compiling {
            compiling.note_code_drop(page, retired);
        }
    }

    /// At run entry: follow the VM's code lifetime, forgetting every entry of
    /// a previous one, and hold a participation in it.
    pub(super) fn begin_run(
        &mut self,
        lifetime: Arc<crate::code_lifetime::CodeLifetime>,
        mapping: Arc<()>,
    ) {
        let compiling = self.compiling.get_or_insert_with(Box::default);
        // A link is filled only in the run that left it (decision-31 §2.4).
        compiling.fill = None;
        // Regions change only between runs (`Vm::map` takes `&mut Vm`).
        if compiling
            .mapping
            .as_ref()
            .is_none_or(|current| !Arc::ptr_eq(current, &mapping))
        {
            compiling.tlb.flush();
            compiling.mapping = Some(mapping);
        }
        if self
            .lifetime
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, &lifetime))
        {
            return;
        }
        compiling.memo.iter_mut().for_each(|slot| *slot = None);
        compiling.victims = None;
        // Epoch numbers belong to a domain.
        compiling.tlb.ring.sp = 0;
        compiling.ring_epoch = 0;
        self.participant = Some(lifetime.participant());
        self.lifetime = Some(lifetime);
    }
}

/// Whether a VM compiles its Flat32 blocks in production (decision-30 §2,
/// items 1-3): hotness-gated background tier-up and [`admits`].
pub(super) fn production(vm: &Vm) -> bool {
    vm.tier_up_after.is_some() && vm.tier_up_background && admits(vm)
}

/// Whether this run of a Machine VM compiles: in production, or under a test
/// mode. Scripted units run no generated code, so they need no admission.
fn compiles(vm: &Vm) -> bool {
    production(vm)
        || match vm.machine_jit {
            MachineJitMode::Off => false,
            MachineJitMode::Scripted(_) => true,
            _ => admits(vm),
        }
}

/// Whether a VM may run compiled Machine code at all, in production or in
/// a test mode: a backend that compiles Machine IR and has code space left,
/// and memory that does not recover host faults (decision-30 Q7). Sparse
/// backing compiles too (decision-33; the Application profile keeps it on
/// the checked interpreter, spec §4.1).
///
/// Decision-33 invariant: every host byte a compiled Machine unit reads or
/// writes inline lies in a mapped region, hence on a committed host page
/// (`Memory::map` commits before it publishes the region). The view fills
/// only frames inside one RAM or ROM region (`Memory::fast_frame`); live
/// fields lie in their unit's fetched bytes, and `Vm::unmap` drops the
/// units of the frames it unmaps; regions change only under `&mut Vm`, and
/// `State::begin_run` empties the view when the mapping changed. A new
/// inline access class must keep it.
pub(super) fn admits(vm: &Vm) -> bool {
    !vm.mem.host_fault_recovery()
        && vm.backend.compiles_machine()
        && !vm.backend.code_space_exhausted()
}

/// The bus address of a Flat32 block (decision-30 §2, decision-32 §4): its
/// EIP through the model's bus mask and A20 with paging off; with paging on,
/// through the TLB entry its fetch used, which a held paged proof keeps
/// (`paging::State::holds`, the TLB token). `None` if the TLB holds none.
fn block_bus(
    cpu: &CpuState,
    paging: Option<&super::paging::State>,
    context: MachineContext,
    ir: &IrBlock,
) -> Option<u64> {
    let eip = ir.guest_start & 0xffff_ffff;
    if !context.paging_enabled {
        return Some(
            crate::legacy::physical_span(
                cpu,
                crate::legacy::physical_mask(context.generation),
                eip,
                1,
            )
            .0,
        );
    }
    crate::interp::legacy::paging::fetched_bus(cpu, &paging?.tlb.borrow(), eip as u32)
}

/// Decision-32 §2: a paged block may enter a compiled unit when its bytes
/// lie in one 4 KiB linear page (so one frame), no paged transaction or port
/// completion is outstanding (E9), and the block came from a paged proof
/// this turn, by a hit or a link the paged path validated (E10): its TLB
/// token is the entry check that the fetch translation is the lifted one.
#[inline(always)]
fn paged_entry(paging: Option<&super::paging::State>, ir: &Arc<IrBlock>) -> bool {
    (ir.guest_start & 0xfff) + u64::from(ir.guest_len) <= 0x1000
        && paging.is_some_and(|state| state.idle() && state.holds(ir))
}

/// The compiled identity of a Flat32 block (decision-27 §1, decision-30 §2):
/// everything its lift and lowering read besides its bytes.
fn machine_key(
    ir: &IrBlock,
    physical: u64,
    context: MachineContext,
    cpu: &CpuState,
    model_semantics_id: u64,
) -> crate::cache::MachineBlockKey {
    crate::cache::MachineBlockKey {
        physical_pc: physical,
        decode_ip: ir.guest_start as u32,
        model_semantics_id,
        flags: key_flags(context, cpu),
        mxcsr_mode: cpu.mxcsr & crate::jit_abi::fp::MODE_MASK,
    }
}

/// `MachineBlockKey::flags`: the context, A20 and CR0.AM.
#[inline(always)]
fn key_flags(context: MachineContext, cpu: &CpuState) -> u32 {
    let wide = |bits: u8| u32::from(bits == 32);
    (context.regime as u32)
        | wide(context.default_operand_bits) << 2
        | wide(context.default_address_bits) << 3
        | wide(context.stack_address_bits) << 4
        | (context.segment_class as u32) << 5
        | u32::from(context.cpl) << 9
        | u32::from(context.paging_enabled) << 11
        | u32::from(cpu.legacy.a20_enabled != 0) << 12
        | u32::from(cpu.legacy.cr0 & (1 << 18) != 0) << 13
}

/// A chain edge's key (`MachineTarget::edge`): the key flags, and with
/// paging on (flags bit 11) the block's linear page and bus frame
/// (decision-32 §4), packed above them.
#[inline(always)]
fn edge_key(flags: u32, linear: u64, bus: u64) -> u64 {
    let paged = u64::from((flags >> 11) & 1);
    let page = (((linear >> 12) & 0xf_ffff) << 14) | (((bus >> 12) & 0xf_ffff) << 34);
    u64::from(flags) | (paged * page)
}

/// Publish a `RET_LINK`'s slot with the unit this turn enters (decision-31
/// §2.4): a descriptor, stored with Release under the lifetime's lock only
/// while `fill.epoch` is current and `entry` live, so a slot that a
/// retirement cleared is never refilled with a dead entry.
#[cold]
#[inline(never)]
fn fill_slot(vm: &Vm, fill: Fill, entry: crate::CompiledPtr, ir: &Arc<IrBlock>, bus: u64) {
    let target = MachineTarget {
        entry,
        ir: ir.clone(),
        icount: ir.icount,
        flags: fill.flags,
        edge: edge_key(fill.flags, ir.guest_start, bus),
    };
    let slot = fill.slot as *const AtomicU64;
    let _ = vm
        .cache
        .publish_machine_target_if_current(fill.epoch, entry, target, |target| {
            // SAFETY: the slot belongs to a unit live in `fill.epoch`, which is
            // current under the lifetime's lock: no retirement has cleared and
            // released it.
            unsafe { (*slot).store(target, Ordering::Release) };
        });
}

/// The bytes a compiled unit of `ir` depends on: its bus span minus the live
/// fields it reads live (decision-27 §3.1), so a store absorbed by a field
/// keeps the unit.
fn unit_sources(ir: &IrBlock, physical: u64) -> Vec<(u64, u32)> {
    let mut sources = vec![(physical, ir.guest_len)];
    for op in &ir.ops {
        if let IrOp::Legacy(crate::legacy::LegacyOp::LiveField { bus, bytes, .. }) = op {
            let (lo, hi) = (*bus, *bus + u64::from(*bytes));
            sources = sources
                .into_iter()
                .flat_map(|(start, len)| {
                    let end = start + u64::from(len);
                    if hi <= start || end <= lo {
                        return vec![(start, len)];
                    }
                    let mut parts = Vec::new();
                    if start < lo {
                        parts.push((start, (lo - start) as u32));
                    }
                    if hi < end {
                        parts.push((hi, (end - hi) as u32));
                    }
                    parts
                })
                .collect();
        }
    }
    sources
}

/// Publish the Machine compiles the worker finished (decision-30 §2), at the
/// top of a turn, as `drain_tier_up` does for Application.
#[cold]
#[inline(never)]
pub(super) fn drain(vm: &Vm, exit_word: &AtomicU64, sections: &AtomicU64) -> u64 {
    use crate::interruption::{CriticalSectionKind, EngineSection};
    let _section = EngineSection::enter(exit_word, sections, CriticalSectionKind::TierUpWait);
    let mut count = 0;
    for fin in vm.backend.tier_up_finished() {
        vm.handle_smc(exit_word);
        vm.cache.record_tier_bg_completed();
        let Some(key) = vm.cache.machine_request(fin.epoch) else {
            super::super::discard_materialized(vm, &fin.block);
            continue;
        };
        let _publish = EngineSection::enter(exit_word, sections, CriticalSectionKind::Publish);
        let published = vm.cache.publish_machine(
            key,
            fin.block.clone(),
            fin.compile_failure,
            fin.epoch,
            |sp| {
                for &(start, len) in sp {
                    vm.mem.mark_code(start, len);
                }
            },
        );
        if published {
            count += 1;
        } else {
            vm.cache.release_source_tags(&vm.mem, &fin.spans);
            super::super::discard_materialized(vm, &fin.block);
        }
    }
    count
}

impl Vcpu {
    /// Run entry of a Machine run on a VM that requests compilation
    /// (`Vm::machine_compile_requested`): whether this run compiles
    /// ([`compiles`]), and if so `State::begin_run` (decision-29: compiled
    /// units live in the VM's code lifetime). Returns the compiled TLB view
    /// that `MemCtx::ret_stack` holds for compiled loads and stores
    /// (decision-30 §4; Machine code has no shadow return stack), or 0 when
    /// this run does not compile.
    #[cold]
    #[inline(never)]
    pub(super) fn begin_machine_compile(&mut self, vm: &Vm) -> u64 {
        if !compiles(vm) {
            return 0;
        }
        let state = &mut self.legacy_execution.as_mut().unwrap().machine_jit;
        state.begin_run(vm.cache.code_lifetime(), vm.mem.mapping_identity());
        std::ptr::from_ref::<crate::jit_abi::machine::MachineTlb>(
            &state.compiling.as_deref().unwrap().tlb,
        ) as u64
    }

    /// Decision-27 §2.5: a drained code write dropped shared units while this
    /// vCPU ran a compiling turn at its EIP. Notes the page the compiled key
    /// names: the Flat32 EIP through the bus mask, or with paging through the
    /// TLB entry its fetch used ([`block_bus`]).
    #[cold]
    #[inline(never)]
    pub(super) fn note_machine_code_drop(&mut self, paging_enabled: bool, physical_mask: u32) {
        let execution = self.legacy_execution.as_mut().unwrap();
        let eip = self.cpu.rip & 0xffff_ffff;
        let fetched = if paging_enabled {
            execution.paging.as_deref().and_then(|state| {
                crate::interp::legacy::paging::fetched_bus(
                    &self.cpu,
                    &state.tlb.borrow(),
                    eip as u32,
                )
            })
        } else {
            Some(crate::legacy::physical_span(&self.cpu, physical_mask, eip, 1).0)
        };
        if let Some(fetched) = fetched {
            execution
                .machine_jit
                .note_code_drop(fetched >> crate::memory::CODE_PAGE_BITS, self.retired);
        }
    }

    /// The compiled-Machine counters of this vCPU since it was created.
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn machine_jit_counters_for_test(&self) -> MachineJitCounters {
        self.legacy_execution
            .as_ref()
            .and_then(|e| e.machine_jit.compiling.as_deref())
            .map_or_else(MachineJitCounters::default, |c| c.counters)
    }

    /// Test-only: record [`ColdCensus`] on this vCPU from its next turn on.
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn enable_cold_census_for_test(&mut self) {
        if let Some(execution) = self.legacy_execution.as_mut() {
            execution
                .machine_jit
                .compiling
                .get_or_insert_with(Box::default)
                .cold
                .get_or_insert_with(Box::default);
        }
    }

    /// Test-only: the cold census, if it was enabled.
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn cold_census_for_test(&self) -> Option<ColdCensus> {
        self.legacy_execution
            .as_ref()
            .and_then(|e| e.machine_jit.compiling.as_deref())
            .and_then(|c| c.cold.as_deref().cloned())
    }

    /// Test-only: an interpreted walk that followed a census entry retired
    /// `retired` instructions; charge them to the entry's row.
    #[cfg(any(test, feature = "test-hooks"))]
    #[cold]
    #[inline(never)]
    pub(super) fn census_retired(&mut self, retired: u64) {
        let Some(state) = self
            .legacy_execution
            .as_mut()
            .and_then(|e| e.machine_jit.compiling.as_deref_mut())
        else {
            return;
        };
        if let Some(row) = state.row.take() {
            let row = usize::from(row);
            if row < REFUSALS {
                state.counters.refusals_retired[row] += retired;
            } else {
                state.counters.no_unit_retired[row - REFUSALS] += retired;
            }
        }
        if let Some(cold) = state.cold.as_deref_mut() {
            if let Some(key) = cold.pending.take() {
                cold.keys.entry(key).or_default().cold_retired += retired;
            }
        }
    }

    /// Decision-29 §2: the walk result of this turn's block from a compiled
    /// unit, or `None` for the interpreted walk.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn machine_progress<const ROUTED: bool, const ORDER: u8>(
        &mut self,
        vm: &Vm,
        ir: &Arc<IrBlock>,
        walk: BlockWalk,
        context: MachineContext,
        remaining: u64,
        boundary: &MachineBoundary,
        ctx: &mut MemCtx,
        exit_word: &AtomicU64,
        hook: Option<&crate::vm::device::InBlock>,
    ) -> Option<(Compiled, Option<Chain>)> {
        let n = u64::from(ir.icount);
        let execution = self.legacy_execution.as_mut().unwrap();
        // A pending fill is offered to this turn's entry only.
        let fill = execution
            .machine_jit
            .compiling
            .as_deref_mut()
            .and_then(|state| state.fill.take());
        // E2: the admitted class (decision-32 §2): Flat32 at CPL 0 or 3, the
        // unpaged ordinary walk or the paged one. CPL 1 and 2 stay
        // interpreted (Q5). E3-E5: no boundary is due inside the block. E6: no
        // fetch policy. Decision-29 adds two: compiled code assumes the
        // default MXCSR mode and does not clear RF, which a retiring
        // instruction does (SDM Vol 3B §20.3.1.1), so RF=1 stays interpreted.
        // Paged, E9 and E10 (`paged_entry`).
        let paged = context.paging_enabled;
        let cpl = self.cpu.legacy.cpl;
        if !(context.segment_class == crate::SegmentClass::Flat32
            && (cpl == 0 || cpl == 3)
            && if paged {
                matches!(walk, BlockWalk::Paged)
            } else {
                matches!(walk, BlockWalk::Direct)
            })
        {
            #[cfg(any(test, feature = "test-hooks"))]
            {
                let reason = if context.segment_class != crate::SegmentClass::Flat32 {
                    REFUSE_SEGMENT_CLASS
                } else if cpl != 0 && cpl != 3 {
                    REFUSE_CPL
                } else {
                    walk_refusal(ir)
                };
                tally_refusal(execution.machine_jit.compiling(), reason);
            }
            return None;
        }
        let eligible = !boundary.one_step
            && remaining >= n
            && !boundary.instruction_retry
            && !boundary.task_pending
            && !boundary.pending_io
            && execution.event_retry.is_none()
            && self.cpu.port_string_in.is_none()
            && !self.cpu.mmio_retry_active
            && vm.mem.fetch_policy.is_none()
            && self.cpu.mxcsr & crate::jit_abi::fp::MODE_MASK == 0
            && self.cpu.rflags_system & (1 << 16) == 0
            && (!paged || paged_entry(execution.paging.as_deref(), ir));
        if !eligible {
            execution.machine_jit.counters().refused += 1;
            #[cfg(any(test, feature = "test-hooks"))]
            {
                let reason = self.refusal(vm, ir, boundary, remaining, paged);
                let state = self
                    .legacy_execution
                    .as_mut()
                    .unwrap()
                    .machine_jit
                    .compiling();
                tally_refusal(state, reason);
            }
            return None;
        }
        let index = memo_index(ir);
        #[cfg(any(test, feature = "test-hooks"))]
        let census_key = execution
            .machine_jit
            .compiling
            .as_deref()
            .is_some_and(|state| state.cold.is_some())
            .then(|| {
                (
                    block_bus(&self.cpu, execution.paging.as_deref(), context, ir)
                        .unwrap_or(u64::MAX),
                    ir.guest_start as u32,
                    key_flags(context, &self.cpu),
                )
            });
        {
            // `begin_run` allocated the share: this run compiles.
            let state = execution.machine_jit.compiling.as_deref_mut().unwrap();
            state.tlb.reconcile(
                self.cpu.legacy.a20_enabled != 0,
                crate::legacy::physical_mask(context.generation),
                paged,
            );
            if paged {
                // `paged_entry` found the paging state.
                state
                    .tlb
                    .sync(&self.cpu, &execution.paging.as_deref().unwrap().tlb);
            }
            if !state.memo[index]
                .as_ref()
                .is_some_and(|slot| std::ptr::eq(slot.ir.as_ptr(), Arc::as_ptr(ir)))
            {
                #[cfg(any(test, feature = "test-hooks"))]
                if let (Some(key), Some(cold)) = (census_key, state.cold.as_deref_mut()) {
                    match &state.memo[index] {
                        None => cold.slots_empty += 1,
                        Some(old) if old.ir.strong_count() != 0 => {
                            cold.slots_conflict += 1;
                            cold.conflicts_cold += u64::from(
                                old.entry.is_none() && old.request == Request::None && old.hot != 0,
                            );
                        }
                        Some(_) => cold.slots_dropped += 1,
                    }
                    cold.keys.entry(key).or_default().slots += 1;
                }
                // Take this block's kept slot before saving the evicted one,
                // which may land in the same set and would otherwise replace it.
                let kept = take_victim(&mut state.victims, ir);
                if let Some(old) = state.memo[index].take() {
                    if old.ir.strong_count() != 0 {
                        save_victim(&mut state.victims, old);
                    }
                }
                #[cfg(any(test, feature = "test-hooks"))]
                if let Some(cold) = state.cold.as_deref_mut() {
                    cold.victims_restored += u64::from(kept.is_some());
                }
                state.memo[index] = Some(kept.unwrap_or_else(|| Slot {
                    ir: Arc::downgrade(ir),
                    entry: None,
                    hot: 0,
                    request: Request::None,
                    delay: match vm.machine_jit {
                        MachineJitMode::Delayed(k) => k,
                        _ => 0,
                    },
                    runs: 0,
                    looked: false,
                    bus: 0,
                    busy: false,
                }));
            }
            let slot = state.memo[index].as_mut().unwrap();
            slot.runs += 1;
            if let MachineJitMode::Scripted(seed) = vm.machine_jit {
                // A scripted STOP at a random ordinal starts the next turn
                // inside the block, which lifts a block there. The lift-count
                // witnesses read such lifts as lost proofs (the unpaged
                // reduced identity, decision-30 §7, shared across IOPL) or
                // as a paged working set past its bound, so scripted units
                // stay in the J2 class: unpaged, CPL 0. Paged and CPL-3
                // units are witnessed in the Forced, Delayed and background
                // modes.
                if paged || cpl != 0 {
                    return None;
                }
                let runs = slot.runs;
                // A scripted chain resolves its next block, which may lift it:
                // the lift and proof-map witnesses count that. It chains the
                // J2 class only, unpaged at CPL 0; paged and CPL-3 scripted
                // units complete, defer and stop without chaining.
                let chain = chains(vm) && !boundary.cold_epilogue && !paged && cpl == 0;
                let before = state.counters.compiled_retired;
                let result = self.scripted::<ROUTED, ORDER>(
                    vm, ir, context, remaining, seed, runs, exit_word, chain, hook,
                );
                self.count_privilege(paged, cpl, before);
                return Some(result);
            }
        }
        let generation = if paged {
            execution.paging.as_deref().unwrap().proof_generation()
        } else {
            execution.flat32.generation()
        };
        // The block's bus address (`block_bus`).
        let paging = execution.paging.as_deref();
        let physical = || block_bus(&self.cpu, paging, context, ir);
        let model = self.legacy_model.map_or(0, |m| m.model_semantics_id);
        let retired = self.retired;
        let State {
            compiling,
            participant,
            lifetime,
            ..
        } = &mut *execution.machine_jit;
        let state = compiling.as_deref_mut().unwrap();
        let cold = state.memo[index]
            .as_ref()
            .is_some_and(|slot| slot.entry.is_none() && slot.request == Request::None)
            && physical()
                .is_none_or(|bus| state.demoted(bus >> crate::memory::CODE_PAGE_BITS, retired));
        if cold {
            tally_no_unit!(state.counters, &mut state.row, NO_UNIT_DEMOTED);
            return None;
        }
        let Compiling {
            counters,
            memo,
            tlb,
            ring_epoch,
            fill: next_fill,
            #[cfg(any(test, feature = "test-hooks"))]
                row: census_row,
            #[cfg(any(test, feature = "test-hooks"))]
                cold: cold_census,
            ..
        } = state;
        let slot = memo[index].as_mut().unwrap();
        // Participation precedes probing or calling any executable pointer
        // (`code_lifetime::ExecutionParticipant`); it lasts through the call.
        let guard = participant
            .as_mut()
            .expect("a Machine run participant")
            .enter();
        let epoch = guard.epoch();
        let entry = match slot.entry {
            Some((entry, at)) if at == epoch => Some(entry),
            Some((entry, _)) => {
                // A unit's retirement advanced the epoch; this one may still
                // be live. A retired one is dropped, never called.
                let live = guard.admits(entry);
                slot.entry = live.then_some((entry, epoch));
                if !live {
                    slot.request = Request::None;
                }
                live.then_some(entry)
            }
            None => None,
        };
        let entry = match entry {
            Some(entry) if slot.delay == 0 => entry,
            Some(_) => {
                slot.delay -= 1;
                tally_no_unit!(counters, census_row, NO_UNIT_DELAYED);
                return None;
            }
            None => {
                // Decision-32 Q7: a fresh slot (a new lift of a block, after a
                // relift that a CR3 load forced, or a memo conflict) looks the
                // shared map up once, so a unit another lift, address space
                // or vCPU published is entered at once.
                let found = if slot.looked {
                    None
                } else {
                    slot.looked = true;
                    fresh_lookup(vm, ir, physical(), context, &self.cpu, model)
                };
                let ready = match slot.request {
                    _ if found.is_some() => true,
                    Request::None => {
                        slot.hot += 1;
                        vm.machine_jit != MachineJitMode::Off
                            || slot.hot >= vm.tier_up_after.unwrap_or(1)
                    }
                    // Look again after a publication, and retry a request
                    // that was refused or rejected every threshold's turns.
                    Request::Waiting(seen) => {
                        slot.hot += 1;
                        vm.cache.machine_published() != seen
                            || slot.hot >= vm.tier_up_after.unwrap_or(1)
                    }
                    Request::Failed => false,
                };
                if !ready {
                    let reason = match slot.request {
                        Request::None if slot.busy => NO_UNIT_BUSY,
                        Request::None => NO_UNIT_COLD,
                        Request::Waiting(_) => NO_UNIT_AWAITING,
                        Request::Failed => NO_UNIT_FAILED,
                    };
                    tally_no_unit!(counters, census_row, reason);
                    #[cfg(any(test, feature = "test-hooks"))]
                    if let (Some(key), Some(cold)) = (census_key, cold_census.as_deref_mut()) {
                        let stats = cold.keys.entry(key).or_default();
                        stats.max_hot = stats.max_hot.max(slot.hot);
                        if reason == NO_UNIT_COLD {
                            stats.cold_turns += 1;
                            cold.pending = Some(key);
                        }
                    }
                    return None;
                }
                #[cfg(any(test, feature = "test-hooks"))]
                if let (Some(key), Some(cold)) = (census_key, cold_census.as_deref_mut()) {
                    let stats = cold.keys.entry(key).or_default();
                    stats.max_hot = stats.max_hot.max(slot.hot);
                    stats.reached = true;
                }
                let physical = physical()?;
                let (found, compiled) = if let Some(found) = found {
                    (found, false)
                } else {
                    let key = machine_key(ir, physical, context, &self.cpu, model);
                    let config = crate::jit_abi::machine::access_config(
                        crate::legacy::physical_mask(context.generation),
                        ROUTED,
                        ORDER,
                    );
                    let Some(unit) =
                        tier_up(vm, ir, key, physical, generation, config, slot, exit_word)
                    else {
                        let reason = if slot.request == Request::Failed {
                            NO_UNIT_FAILED
                        } else {
                            NO_UNIT_AWAITING
                        };
                        tally_no_unit!(counters, census_row, reason);
                        return None;
                    };
                    unit
                };
                counters.compiled += u64::from(compiled);
                slot.entry = Some((found, epoch));
                slot.bus = physical;
                if slot.delay != 0 {
                    tally_no_unit!(counters, census_row, NO_UNIT_DELAYED);
                    return None;
                }
                found
            }
        };
        if let Some(fill) = fill {
            if fill.retired == retired
                && fill.target == ir.guest_start
                && fill.flags == key_flags(context, &self.cpu)
            {
                fill_slot(vm, fill, entry, ir, slot.bus);
            }
        }
        // Continuation slots pushed in an earlier epoch may name retired
        // units (decision-31 §5).
        if *ring_epoch != epoch {
            tlb.ring.sp = 0;
            *ring_epoch = epoch;
        }
        let lifetime = lifetime.as_deref().expect("a Machine run lifetime");
        // The hot path's counters exist for the witnesses only.
        #[cfg(any(test, feature = "test-hooks"))]
        let before = counters.compiled_retired;
        #[cfg(any(test, feature = "test-hooks"))]
        {
            counters.entries += 1;
        }
        ctx.machine_ordinal = 0;
        if vm.time_enabled {
            crate::time_stats::compiled();
        }
        // SAFETY: `entry` is a Machine unit compiled to this ABI and live in
        // the guard's epoch, which this vCPU announces until the call returns.
        let mut code = unsafe { crate::jit_abi::call_block_in_mode(entry, &mut self.cpu, ctx) };
        // Decision-31 §2.3: the chain loop. No chain starts under the cold
        // epilogue (Q6). `cur` is the running block's target; `None` for the
        // first block, which `ir` names.
        let mut cur: Option<&MachineTarget> = None;
        let mut prefix = 0u64;
        let mut flags = None;
        let chain = code == RET_CHAIN && !boundary.cold_epilogue && chains(vm);
        if code == RET_CHAIN && !chain {
            // No chain starts: the block's end is an unchained exit, which
            // `machine_result` does not count for a `RET_CHAIN`.
            census(counters, CHAIN_END_UNCHAINED);
        }
        if chain {
            let chain_flags = key_flags(context, &self.cpu);
            flags = Some(chain_flags);
            // Decision-32 §4: a paged chain stays in this block's linear page
            // and bus frame. Compiled code changes no TLB entry, so the fetch
            // translation the first entry validated (E10) holds for the whole
            // chain, and a target in the same page and frame is a unit of
            // the bytes a fetch would now read. Other pages return to the
            // dispatcher, whose link or resolve carries their own token.
            let chain_edge = edge_key(chain_flags, ir.guest_start, slot.bus);
            loop {
                let completed = cur.map_or(n, |t| u64::from(t.icount));
                debug_assert!(
                    static_successor(cur.map_or(&**ir, |t| &*t.ir), self.cpu.rip),
                    "a chained block completed at {:#x}, no successor of its block",
                    self.cpu.rip
                );
                // SAFETY: a filled Machine slot holds a `MachineTarget`,
                // published with Release and loaded with acquire; this vCPU
                // has participated in `epoch` since before that load, so the
                // descriptor is not reclaimed until the guard drops.
                let t = unsafe { &*(ctx.next_entry as *const MachineTarget) };
                let done = prefix + completed;
                #[cfg(any(test, feature = "test-hooks"))]
                edge_hook(vm, exit_word, retired + done);
                // §4.3 the dispatch word, §5 the epoch, §2.5 the flags, §4.1
                // the budget (E4 at every chained entry).
                let end = if exit_word.load(Ordering::Relaxed) != 0 {
                    Some(CHAIN_END_WORD)
                } else if lifetime.epoch() != epoch {
                    Some(CHAIN_END_EPOCH)
                } else if t.edge != chain_edge {
                    Some(if t.flags != chain_flags {
                        CHAIN_END_FLAGS
                    } else {
                        CHAIN_END_PAGE
                    })
                } else if remaining - done < u64::from(t.icount) {
                    Some(CHAIN_END_BUDGET)
                } else {
                    None
                };
                if let Some(reason) = end {
                    // A completion of `cur`, at RIP = t's start.
                    census(counters, reason);
                    break;
                }
                debug_assert_eq!(self.cpu.rip, t.ir.guest_start, "a chained block's entry");
                #[cfg(any(test, feature = "test-hooks"))]
                {
                    counters.continues += 1;
                    counters.compiled_retired += completed;
                    counters.entries += 1;
                    counters.chained += 1;
                }
                // §4.4: the in-block device hook follows the running block.
                if let Some(hook) = hook {
                    hook.advance(&t.ir, completed);
                }
                self.chained += 1;
                prefix = done;
                cur = Some(t);
                ctx.machine_ordinal = 0;
                // SAFETY: published only while live (`fill_slot`), and its
                // retirement would have advanced the epoch the edge tested,
                // which the guard pins against reclamation.
                code = unsafe { crate::jit_abi::call_block_in_mode(t.entry, &mut self.cpu, ctx) };
                if code != RET_CHAIN {
                    break;
                }
            }
        }
        let last = cur.map(|t| t.ir.clone());
        let n_last = cur.map_or(n, |t| u64::from(t.icount));
        if code == RET_LINK {
            *next_fill = Some(Fill {
                slot: ctx.link_slot,
                target: self.cpu.rip,
                flags: flags.unwrap_or_else(|| key_flags(context, &self.cpu)),
                epoch,
                retired: retired + prefix + n_last,
            });
        }
        drop(guard);
        if vm.time_enabled {
            crate::time_stats::interpreted();
        }
        let ordinal = ctx.machine_ordinal;
        #[cfg(not(any(test, feature = "test-hooks")))]
        let before = 0;
        let Some(last) = last else {
            let result =
                self.machine_result::<ROUTED, ORDER>(vm, ir, code, ordinal, remaining, exit_word);
            self.count_privilege(paged, cpl, before);
            return Some((result, None));
        };
        // §4.4: a DEFER's suffix serves its transfers for the last block.
        if let Some(hook) = hook {
            hook.advance(&last, 0);
        }
        let result = self.machine_result::<ROUTED, ORDER>(
            vm,
            &last,
            code,
            ordinal,
            remaining - prefix,
            exit_word,
        );
        self.count_privilege(paged, cpl, before);
        Some((result, Some(Chain { prefix, ir: last })))
    }

    /// Test-only: the first entry condition (E3-E10) that refused this
    /// turn's admitted block, as `machine_progress` tests them.
    #[cfg(any(test, feature = "test-hooks"))]
    #[cold]
    #[inline(never)]
    fn refusal(
        &self,
        vm: &Vm,
        ir: &Arc<IrBlock>,
        boundary: &MachineBoundary,
        remaining: u64,
        paged: bool,
    ) -> usize {
        let execution = self.legacy_execution.as_ref().unwrap();
        let paging = execution.paging.as_deref();
        if boundary.one_step {
            return if boundary.single_step {
                REFUSE_TRAP_FLAG
            } else if self.sti_shadow || self.ss_shadow & SS_HOLD != 0 {
                REFUSE_SHADOW
            } else if boundary.irq_pending {
                REFUSE_IRQ_PENDING
            } else if (self.device_cpu().intr && self.cpu.flags.if_) || self.device_cpu().invalidate
            {
                REFUSE_DEVICE_INTR
            } else {
                REFUSE_ONE_STEP_OTHER
            };
        }
        if remaining < u64::from(ir.icount) {
            REFUSE_BUDGET
        } else if boundary.instruction_retry
            || execution.event_retry.is_some()
            || self.cpu.mmio_retry_active
        {
            REFUSE_RETRY
        } else if boundary.task_pending {
            REFUSE_TASK
        } else if boundary.pending_io || self.cpu.port_string_in.is_some() {
            REFUSE_PENDING_IO
        } else if vm.mem.fetch_policy.is_some() {
            REFUSE_FETCH_POLICY
        } else if self.cpu.mxcsr & crate::jit_abi::fp::MODE_MASK != 0 {
            REFUSE_MXCSR
        } else if self.cpu.rflags_system & (1 << 16) != 0 {
            REFUSE_RF
        } else if !paged {
            unreachable!("an unpaged block refused by no condition")
        } else if (ir.guest_start & 0xfff) + u64::from(ir.guest_len) > 0x1000 {
            REFUSE_PAGE_CROSSING
        } else if paging.is_none_or(|state| !state.idle()) {
            REFUSE_PAGED_BUSY
        } else {
            REFUSE_PAGED_PROOF
        }
    }

    /// Test-only: a compiled turn's instructions at CPL 3 and its entries
    /// with paging on (decision-32 §9 done-when 1), from its compiled-retired
    /// count `before` the turn.
    #[inline(always)]
    fn count_privilege(&mut self, paged: bool, cpl: u8, before: u64) {
        #[cfg(any(test, feature = "test-hooks"))]
        {
            let counters = self
                .legacy_execution
                .as_mut()
                .unwrap()
                .machine_jit
                .counters();
            if cpl == 3 {
                counters.compiled_retired_cpl3 += counters.compiled_retired - before;
            }
            counters.paged_entries += u64::from(paged);
        }
        #[cfg(not(any(test, feature = "test-hooks")))]
        let _ = (paged, cpl, before);
    }

    /// A compiled block's return, for `ir`, the block that returned it, with
    /// `remaining` its budget (decision-29 §3; decision-31 Q4: `RET_CHAIN`
    /// that ends a chain, `RET_LINK` and `RET_RETURN_MISS` are completions).
    fn machine_result<const ROUTED: bool, const ORDER: u8>(
        &mut self,
        vm: &Vm,
        ir: &Arc<IrBlock>,
        code: u64,
        ordinal: u64,
        remaining: u64,
        exit_word: &AtomicU64,
    ) -> Compiled {
        let n = u64::from(ir.icount);
        let counters = self
            .legacy_execution
            .as_mut()
            .unwrap()
            .machine_jit
            .counters();
        match code {
            RET_CONTINUE | RET_CHAIN | RET_LINK | RET_RETURN_MISS => {
                debug_assert!(
                    static_successor(ir, self.cpu.rip),
                    "a completed block left RIP at {:#x}, no successor of {:#x}",
                    self.cpu.rip,
                    ir.guest_start
                );
                #[cfg(any(test, feature = "test-hooks"))]
                {
                    counters.continues += 1;
                    counters.compiled_retired += n;
                }
                match code {
                    RET_CONTINUE => census(counters, CHAIN_END_UNCHAINED),
                    RET_LINK => census(counters, CHAIN_END_EMPTY_SLOT),
                    RET_RETURN_MISS => census(counters, CHAIN_END_RETURN_MISS),
                    // Counted at the edge that ended the chain, or as
                    // unchained where no chain started (`machine_progress`).
                    _ => {}
                }
                Compiled::Done {
                    count: n,
                    stop: false,
                }
            }
            RET_MACHINE_STOP => {
                debug_assert_eq!(
                    self.cpu.rip,
                    instruction_addr(ir, ordinal),
                    "a STOP leaves RIP at its ordinal's instruction"
                );
                counters.stops += 1;
                counters.compiled_retired += ordinal;
                census(counters, CHAIN_END_STOP);
                debug_assert!(ordinal != 0 && ordinal < n, "a STOP inside its block");
                Compiled::Done {
                    count: ordinal,
                    stop: true,
                }
            }
            RET_MACHINE_DEFER => {
                let c = ordinal & !HELPER_DEFER;
                if ordinal & HELPER_DEFER != 0 {
                    counters.helper_defers += 1;
                    #[cfg(any(test, feature = "test-hooks"))]
                    {
                        let cause = machine_jit::take_defer_cause(ir, c, &self.cpu);
                        counters.defer_causes[usize::from(cause)] += 1;
                    }
                } else {
                    counters.unconditional_defers += 1;
                }
                census(counters, CHAIN_END_DEFER);
                Compiled::Walk(self.deferred::<ROUTED, ORDER>(vm, ir, c, remaining, exit_word))
            }
            other => panic!("compiled Machine block returned invalid ABI code {other}"),
        }
    }

    /// `RET_MACHINE_DEFER` at ordinal `c`: the interpreter walks the rest of
    /// the block in this turn, and the two counts merge (decision-29 §4).
    fn deferred<const ROUTED: bool, const ORDER: u8>(
        &mut self,
        vm: &Vm,
        ir: &Arc<IrBlock>,
        c: u64,
        remaining: u64,
        exit_word: &AtomicU64,
    ) -> Progress {
        debug_assert!(c < u64::from(ir.icount), "a DEFER inside its block");
        debug_assert_eq!(
            self.cpu.rip,
            match ir.ops[machine_jit::instruction_op(ir, c)] {
                IrOp::InsnStart { guest_addr } => guest_addr,
                _ => unreachable!(),
            },
            "a DEFER leaves RIP at its instruction"
        );
        // A device transfer the suffix serves in its pass is counted from c.
        if c != 0 {
            crate::vm::device::advance_armed(c);
        }
        let from = machine_jit::instruction_op(ir, c);
        // Decision-32 §3: a paged block's rest takes the paged walk, which
        // walks, sets A and D, raises #PF and publishes CR2 itself.
        let suffix = if self.cpu.legacy.cr0 & (1 << 31) != 0 {
            self.execute_paged_from::<ROUTED, ORDER>(vm, ir, remaining - c, from)
        } else {
            let execution = self.legacy_execution.as_mut().unwrap();
            machine_jit::execute_suffix::<ROUTED, ORDER>(
                ir,
                &mut self.cpu,
                &vm.mem,
                &mut self.interp_scratch,
                remaining - c,
                exit_word,
                &mut execution.task_switch,
                from,
            )
        };
        // Ordinary blocks hold no string instruction: every step the suffix
        // took retired one, and RIP stands where the merged count says.
        debug_assert_eq!(
            suffix.steps, suffix.retired,
            "a suffix step that retired nothing"
        );
        debug_assert!(
            c + suffix.retired >= u64::from(ir.icount)
                || self.cpu.rip == instruction_addr(ir, c + suffix.retired),
            "a suffix stopped at {:#x} after {} instructions",
            self.cpu.rip,
            c + suffix.retired
        );
        let execution = self.legacy_execution.as_mut().unwrap();
        let counters = execution.machine_jit.counters();
        counters.compiled_retired += c;
        counters.defers_at_zero += u64::from(c == 0);
        counters.suffix[machine_jit::suffix_kind(&suffix)] += 1;
        Progress {
            steps: c + suffix.steps,
            retired: c + suffix.retired,
            ..suffix
        }
    }

    /// Mode 4: complete a scripted prefix on the interpreter, then return as a
    /// compiled unit would. Deferring or stopping is legal at any instruction
    /// start, so every choice must equal the interpreted walk. A completed
    /// block also ends with a seeded `RET_CHAIN` to the block the resolver
    /// returns next, `RET_LINK` or `RET_CONTINUE` (decision-31 §8), so the
    /// chain loop's half in the dispatcher meets every end reason after every
    /// block, and DEFER and STOP at every ordinal of every chained block.
    #[allow(clippy::too_many_arguments)]
    fn scripted<const ROUTED: bool, const ORDER: u8>(
        &mut self,
        vm: &Vm,
        ir: &Arc<IrBlock>,
        context: MachineContext,
        remaining: u64,
        seed: u64,
        runs: u64,
        exit_word: &AtomicU64,
        chain: bool,
        hook: Option<&crate::vm::device::InBlock>,
    ) -> (Compiled, Option<Chain>) {
        let flags = key_flags(context, &self.cpu);
        let model = self.legacy_model.unwrap_or(LegacyCpuModel {
            generation: CpuGeneration::I286,
            model_semantics_id: 0,
        });
        // The running block, at a stable address the device hook may name.
        let mut cur = ir.clone();
        let mut prefix = 0u64;
        loop {
            let n = u64::from(cur.icount);
            let (result, z) = self.scripted_block::<ROUTED, ORDER>(
                vm,
                &cur,
                remaining - prefix,
                seed,
                runs.wrapping_add(prefix.wrapping_mul(0x1_0000_0001)),
                exit_word,
            );
            let execution = self.legacy_execution.as_mut().unwrap();
            let counters = execution.machine_jit.counters();
            let complete = matches!(result, Compiled::Done { stop: false, .. });
            let end = if !complete {
                None
            } else if !chain {
                Some(CHAIN_END_UNCHAINED)
            } else {
                #[cfg(any(test, feature = "test-hooks"))]
                if (z >> 40) % 3 == 0 {
                    edge_hook(vm, exit_word, self.retired + prefix + n);
                }
                match (z >> 40) % 3 {
                    1 => Some(CHAIN_END_EMPTY_SLOT),
                    2 => Some(CHAIN_END_UNCHAINED),
                    _ if exit_word.load(Ordering::Relaxed) != 0 => Some(CHAIN_END_WORD),
                    _ if key_flags(context, &self.cpu) != flags => Some(CHAIN_END_FLAGS),
                    _ => None,
                }
            };
            if let Some(reason) = end {
                census(counters, reason);
            }
            if !complete || end.is_some() {
                return (result, (prefix != 0).then_some(Chain { prefix, ir: cur }));
            }
            // The block the next turn would resolve here.
            let next = if execution.context_id == u64::MAX {
                None
            } else {
                execution
                    .resolve::<MACHINE_DISPATCH>(vm, &self.cpu, model, context, exit_word)
                    .ok()
                    .filter(|(_, walk)| matches!(walk, BlockWalk::Direct))
            };
            let counters = self
                .legacy_execution
                .as_mut()
                .unwrap()
                .machine_jit
                .counters();
            let Some((next, _)) = next else {
                census(counters, CHAIN_END_UNCHAINED);
                return (result, (prefix != 0).then_some(Chain { prefix, ir: cur }));
            };
            if remaining - prefix - n < u64::from(next.icount) {
                census(counters, CHAIN_END_BUDGET);
                return (result, (prefix != 0).then_some(Chain { prefix, ir: cur }));
            }
            counters.chained += 1;
            self.chained += 1;
            prefix += n;
            cur = next;
            if let Some(hook) = hook {
                hook.advance(&cur, n);
            }
        }
    }

    /// One scripted block (mode 4), and the seeded draw it made.
    fn scripted_block<const ROUTED: bool, const ORDER: u8>(
        &mut self,
        vm: &Vm,
        ir: &Arc<IrBlock>,
        remaining: u64,
        seed: u64,
        runs: u64,
        exit_word: &AtomicU64,
    ) -> (Compiled, u64) {
        let n = u64::from(ir.icount);
        // SplitMix64 over the seed, the block and its run count.
        let mut z = seed
            .wrapping_add(ir.guest_start.wrapping_mul(0x9e37_79b9_7f4a_7c15))
            .wrapping_add(runs.wrapping_mul(0xbf58_476d_1ce4_e5b9));
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        let c = z % (n + 1);
        let kind = (z >> 32) % 3;
        self.legacy_execution
            .as_mut()
            .unwrap()
            .machine_jit
            .counters()
            .entries += 1;
        let prefix = if c == 0 {
            machine_jit::completed(0)
        } else {
            let execution = self.legacy_execution.as_mut().unwrap();
            crate::interp::legacy::execute_instructions::<ROUTED, ORDER>(
                ir,
                &mut self.cpu,
                &vm.mem,
                &mut self.interp_scratch,
                c,
                exit_word,
                &mut execution.task_switch,
            )
        };
        let counters = self
            .legacy_execution
            .as_mut()
            .unwrap()
            .machine_jit
            .counters();
        if !machine_jit::is_completed(&prefix, c) {
            // The interpreter itself ended the prefix: its result is the walk's.
            census(counters, CHAIN_END_UNCHAINED);
            return (Compiled::Walk(prefix), z);
        }
        if c == n {
            counters.continues += 1;
            counters.compiled_retired += n;
            return (
                Compiled::Done {
                    count: n,
                    stop: false,
                },
                z,
            );
        }
        // A compiled store that leaves the dispatch word set stops the block
        // after its instruction (`jit_abi::machine::ACCESS_STOP`); a prefix
        // that did so cannot continue.
        let stop = c != 0 && exit_word.load(Ordering::Relaxed) != 0;
        if (kind == 0 || stop) && c != 0 {
            counters.stops += 1;
            counters.compiled_retired += c;
            census(counters, CHAIN_END_STOP);
            return (
                Compiled::Done {
                    count: c,
                    stop: true,
                },
                z,
            );
        }
        counters.unconditional_defers += 1;
        census(counters, CHAIN_END_DEFER);
        (
            Compiled::Walk(self.deferred::<ROUTED, ORDER>(vm, ir, c, remaining, exit_word)),
            z,
        )
    }
}

/// Decision-32 Q7: the shared unit of a block's key, looked up once when a
/// vCPU first meets that lift.
#[cold]
#[inline(never)]
fn fresh_lookup(
    vm: &Vm,
    ir: &IrBlock,
    physical: Option<u64>,
    context: MachineContext,
    cpu: &CpuState,
    model: u64,
) -> Option<crate::CompiledPtr> {
    vm.cache
        .machine_unit(&machine_key(ir, physical?, context, cpu, model))
}

/// Find or start this block's shared unit (decision-30 §2). A published unit
/// is returned, with whether this call compiled it; otherwise a request is made: background in production, a
/// synchronous compile in the Forced and Delayed test modes (decision-29 §9).
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn tier_up(
    vm: &Vm,
    ir: &Arc<IrBlock>,
    key: crate::cache::MachineBlockKey,
    physical: u64,
    generation: u64,
    config: u64,
    slot: &mut Slot,
    exit_word: &AtomicU64,
) -> Option<(crate::CompiledPtr, bool)> {
    let seen = vm.cache.machine_published();
    if let Some(entry) = vm.cache.machine_unit(&key) {
        return Some((entry, false));
    }
    let mark = |sp: &[(u64, u32)]| {
        for &(start, len) in sp {
            vm.mem.mark_code(start, len);
        }
    };
    let sources = unit_sources(ir, physical);
    let Some(id) = vm
        .cache
        .begin_machine_tier_up(key, generation, sources.clone(), mark)
    else {
        // In flight elsewhere, suppressed, or a code write moved the
        // generation: look again after the next publication.
        slot.hot = 0;
        slot.request = Request::Waiting(seen);
        return None;
    };
    if vm.machine_jit == MachineJitMode::Off {
        let request = crate::TierUpRequest {
            precise_split_stores: vm.precise_split_stores,
            mxcsr_mode: key.mxcsr_mode,
            pc: physical,
            unit: crate::TierUpUnit::Block(ir.clone()),
            consistency: vm.consistency,
            mmio: None,
            guest_base: vm.mem.guest_base(),
            spans: sources.clone(),
            epoch: id,
        };
        match vm.backend.tier_up_async(request) {
            crate::TierUpSubmit::Queued => {
                vm.cache.record_tier_bg_submitted();
                slot.busy = false;
                slot.hot = 0;
                slot.request = Request::Waiting(seen);
            }
            crate::TierUpSubmit::Busy => {
                vm.cache.machine_request(id);
                vm.cache.abandon_machine_tier_up(key, &vm.mem);
                slot.busy = true;
                slot.hot = 0;
            }
            crate::TierUpSubmit::Unsupported => {
                vm.cache.machine_request(id);
                vm.cache.abandon_machine_tier_up(key, &vm.mem);
                slot.request = Request::Failed;
            }
        }
        return None;
    }
    let compiled = vm.backend.materialize_machine(ir, vm.consistency, config);
    vm.cache.machine_request(id);
    vm.handle_smc(exit_word);
    let Some(entry) = compiled else {
        vm.cache.abandon_machine_tier_up(key, &vm.mem);
        slot.request = Request::Failed;
        return None;
    };
    let block = crate::CachedBlock::Compiled { entry };
    if vm.cache.publish_machine(key, block.clone(), None, id, mark) {
        Some((entry, true))
    } else {
        vm.cache.release_source_tags(&vm.mem, &sources);
        super::super::discard_materialized(vm, &block);
        slot.hot = 0;
        slot.request = Request::Waiting(seen);
        None
    }
}

/// The address of instruction `ordinal` of `ir` (its `InsnStart`).
fn instruction_addr(ir: &IrBlock, ordinal: u64) -> u64 {
    match ir.ops[machine_jit::instruction_op(ir, ordinal)] {
        IrOp::InsnStart { guest_addr } => guest_addr,
        _ => unreachable!("an instruction marker"),
    }
}

/// Whether `rip` is where a block that completed may leave it: a static
/// target of its last transfer, its end after a fall-through, or anywhere
/// after an indirect one. A count charged for a block that stopped short
/// would leave RIP inside it (xj-to9.80).
fn static_successor(ir: &IrBlock, rip: u64) -> bool {
    use crate::legacy::LegacyOp;
    let last = ir
        .ops
        .iter()
        .rev()
        .find(|op| !matches!(op, IrOp::Legacy(LegacyOp::SharedWalk)));
    match last {
        Some(IrOp::Branch {
            taken, fallthrough, ..
        })
        | Some(IrOp::LoopCx {
            taken, fallthrough, ..
        }) => rip == *taken || rip == *fallthrough,
        Some(IrOp::Legacy(LegacyOp::Loop { taken, next, .. })) => {
            rip == u64::from(*taken) || rip == u64::from(*next)
        }
        Some(IrOp::Jump {
            target: crate::ir::Val::Imm(target),
        }) => rip == *target,
        Some(IrOp::Jump { .. }) => true,
        _ => rip == ir.guest_start + u64::from(ir.guest_len),
    }
}
