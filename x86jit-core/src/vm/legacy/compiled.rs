//! Compiled Machine blocks in the Machine turn (decision-29, J1): the entry
//! contract, the call, and the conversion of its return into the walk result
//! the interpreted walk of the same block returns. Admitted only by the
//! test-only [`MachineJitMode`]; production runs never reach this module.
use super::*;
use crate::interp::legacy::machine_jit;
use crate::interp::legacy::Progress;
use crate::jit_abi::{MemCtx, RET_CONTINUE, RET_MACHINE_DEFER, RET_MACHINE_STOP};

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
    /// Units compiled (modes 2-3).
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
    ir: Arc<IrBlock>,
    entry: Option<(crate::CompiledPtr, u64)>,
    hot: u32,
    request: Request,
    delay: u32,
    runs: u64,
}

/// Slots of the per-vCPU memo: a direct-mapped table keyed by the block a
/// proof or a link returned (decision-30 §6: no units lookup). A conflict
/// only costs a recount and a cold lookup of the shared unit.
const MEMO: usize = 256;

#[inline(always)]
fn memo_index(ir: &Arc<IrBlock>) -> usize {
    let address = Arc::as_ptr(ir) as usize as u64;
    (address.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> (64 - MEMO.trailing_zeros())) as usize
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
    pub(super) counters: MachineJitCounters,
}

/// The state only a run that compiles uses.
pub(super) struct Compiling {
    memo: Box<[Option<Slot>]>,
    hot_pages: [HotPage; 16],
    /// The compiled view of the TLB (decision-30 §4), at a stable address
    /// for `MemCtx::ret_stack` (its Machine meaning), and the mapping its entries were filled from.
    pub(super) tlb: crate::jit_abi::machine::MachineTlb,
    mapping: Option<Arc<()>>,
}

impl Default for Compiling {
    fn default() -> Self {
        Self {
            memo: (0..MEMO).map(|_| None).collect(),
            hot_pages: [HotPage::default(); 16],
            tlb: crate::jit_abi::machine::MachineTlb::default(),
            mapping: None,
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
#[cold]
#[inline(never)]
pub(super) fn compiles(vm: &Vm) -> bool {
    production(vm)
        || match vm.machine_jit {
            MachineJitMode::Off => false,
            MachineJitMode::Scripted(_) => true,
            _ => admits(vm),
        }
}

/// Whether a VM may run compiled Machine code at all, in production or in
/// a test mode: a backend that compiles Machine IR and has code space left,
/// and memory that is neither sparse (spec §4.1: sparse backing executes
/// through the checked interpreter, even with a JIT backend) nor recovering
/// host faults (decision-30 Q7).
pub(super) fn admits(vm: &Vm) -> bool {
    !vm.mem.is_sparse()
        && !vm.mem.host_fault_recovery()
        && vm.backend.compiles_machine()
        && !vm.backend.code_space_exhausted()
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
    let wide = |bits: u8| u32::from(bits == 32);
    let flags = (context.regime as u32)
        | wide(context.default_operand_bits) << 2
        | wide(context.default_address_bits) << 3
        | wide(context.stack_address_bits) << 4
        | (context.segment_class as u32) << 5
        | u32::from(context.cpl) << 9
        | u32::from(context.paging_enabled) << 11
        | u32::from(cpu.legacy.a20_enabled != 0) << 12
        | u32::from(cpu.legacy.cr0 & (1 << 18) != 0) << 13;
    crate::cache::MachineBlockKey {
        physical_pc: physical,
        decode_ip: ir.guest_start as u32,
        model_semantics_id,
        flags,
        mxcsr_mode: cpu.mxcsr & crate::jit_abi::fp::MODE_MASK,
    }
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
pub(super) fn drain(vm: &Vm, exit_word: &AtomicU64, sections: &AtomicU64) {
    use crate::interruption::{CriticalSectionKind, EngineSection};
    let _section = EngineSection::enter(exit_word, sections, CriticalSectionKind::TierUpWait);
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
        if !published {
            vm.cache.release_source_tags(&vm.mem, &fin.spans);
            super::super::discard_materialized(vm, &fin.block);
        }
    }
}

impl Vcpu {
    /// The compiled-Machine counters of this vCPU since it was created.
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn machine_jit_counters_for_test(&self) -> MachineJitCounters {
        self.legacy_execution
            .as_ref()
            .map_or_else(MachineJitCounters::default, |e| e.machine_jit.counters)
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
    ) -> Option<Compiled> {
        let n = u64::from(ir.icount);
        let execution = self.legacy_execution.as_mut().unwrap();
        // E2: the admitted class. E3-E5: no boundary is due inside the block.
        // E6: no fetch policy. Decision-29 adds two: compiled code assumes the
        // default MXCSR mode and does not clear RF, which a retiring
        // instruction does (SDM Vol 3B §20.3.1.1), so RF=1 stays interpreted.
        if !(matches!(walk, BlockWalk::Direct)
            && context.segment_class == crate::SegmentClass::Flat32
            && !context.paging_enabled
            && self.cpu.legacy.cpl == 0)
        {
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
            && self.cpu.rflags_system & (1 << 16) == 0;
        if !eligible {
            execution.machine_jit.counters.refused += 1;
            return None;
        }
        let index = memo_index(ir);
        {
            // `begin_run` allocated the share: this run compiles.
            let state = execution.machine_jit.compiling.as_deref_mut().unwrap();
            state.tlb.reconcile(
                self.cpu.legacy.a20_enabled != 0,
                crate::legacy::physical_mask(context.generation),
            );
            if !state.memo[index]
                .as_ref()
                .is_some_and(|slot| Arc::ptr_eq(&slot.ir, ir))
            {
                state.memo[index] = Some(Slot {
                    ir: ir.clone(),
                    entry: None,
                    hot: 0,
                    request: Request::None,
                    delay: match vm.machine_jit {
                        MachineJitMode::Delayed(k) => k,
                        _ => 0,
                    },
                    runs: 0,
                });
            }
            let slot = state.memo[index].as_mut().unwrap();
            slot.runs += 1;
            if let MachineJitMode::Scripted(seed) = vm.machine_jit {
                let runs = slot.runs;
                return Some(
                    self.scripted::<ROUTED, ORDER>(vm, ir, remaining, seed, runs, exit_word),
                );
            }
        }
        let generation = execution.flat32.generation();
        // The block's bus address: a Flat32 linear address is its EIP.
        let physical = || {
            crate::legacy::physical_span(
                &self.cpu,
                crate::legacy::physical_mask(context.generation),
                ir.guest_start & 0xffff_ffff,
                1,
            )
            .0
        };
        let model = self.legacy_model.map_or(0, |m| m.model_semantics_id);
        let retired = self.retired;
        let State {
            compiling,
            participant,
            counters,
            ..
        } = &mut *execution.machine_jit;
        let state = compiling.as_deref_mut().unwrap();
        let cold = state.memo[index]
            .as_ref()
            .is_some_and(|slot| slot.entry.is_none() && slot.request == Request::None)
            && state.demoted(physical() >> crate::memory::CODE_PAGE_BITS, retired);
        if cold {
            return None;
        }
        let slot = state.memo[index].as_mut().unwrap();
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
                return None;
            }
            None => {
                let ready = match slot.request {
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
                    return None;
                }
                let physical = physical();
                let key = machine_key(ir, physical, context, &self.cpu, model);
                let config = crate::jit_abi::machine::access_config(
                    crate::legacy::physical_mask(context.generation),
                    ROUTED,
                    ORDER,
                );
                let (found, compiled) =
                    tier_up(vm, ir, key, physical, generation, config, slot, exit_word)?;
                counters.compiled += u64::from(compiled);
                slot.entry = Some((found, epoch));
                if slot.delay != 0 {
                    return None;
                }
                found
            }
        };
        // The hot path's counters exist for the witnesses only.
        #[cfg(any(test, feature = "test-hooks"))]
        {
            counters.entries += 1;
        }
        #[cfg(not(any(test, feature = "test-hooks")))]
        let _ = counters;
        ctx.machine_ordinal = 0;
        if vm.time_enabled {
            crate::time_stats::compiled();
        }
        // SAFETY: `entry` is a Machine unit compiled to this ABI and live in
        // the guard's epoch, which this vCPU announces until the call returns.
        let code = unsafe { crate::jit_abi::call_block_in_mode(entry, &mut self.cpu, ctx) };
        drop(guard);
        if vm.time_enabled {
            crate::time_stats::interpreted();
        }
        let ordinal = ctx.machine_ordinal;
        let counters = &mut self.legacy_execution.as_mut().unwrap().machine_jit.counters;
        Some(match code {
            RET_CONTINUE => {
                #[cfg(any(test, feature = "test-hooks"))]
                {
                    counters.continues += 1;
                    counters.compiled_retired += n;
                }
                Compiled::Done {
                    count: n,
                    stop: false,
                }
            }
            RET_MACHINE_STOP => {
                counters.stops += 1;
                counters.compiled_retired += ordinal;
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
                } else {
                    counters.unconditional_defers += 1;
                }
                Compiled::Walk(self.deferred::<ROUTED, ORDER>(vm, ir, c, remaining, exit_word))
            }
            other => panic!("compiled Machine block returned invalid ABI code {other}"),
        })
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
        let execution = self.legacy_execution.as_mut().unwrap();
        let suffix = machine_jit::execute_suffix::<ROUTED, ORDER>(
            ir,
            &mut self.cpu,
            &vm.mem,
            &mut self.interp_scratch,
            remaining - c,
            exit_word,
            &mut execution.task_switch,
            machine_jit::instruction_op(ir, c),
        );
        let counters = &mut execution.machine_jit.counters;
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
    /// start, so every choice must equal the interpreted walk.
    fn scripted<const ROUTED: bool, const ORDER: u8>(
        &mut self,
        vm: &Vm,
        ir: &Arc<IrBlock>,
        remaining: u64,
        seed: u64,
        runs: u64,
        exit_word: &AtomicU64,
    ) -> Compiled {
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
        let execution = self.legacy_execution.as_mut().unwrap();
        execution.machine_jit.counters.entries += 1;
        let prefix = if c == 0 {
            machine_jit::completed(0)
        } else {
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
        let counters = &mut execution.machine_jit.counters;
        if !machine_jit::is_completed(&prefix, c) {
            // The interpreter itself ended the prefix: its result is the walk's.
            return Compiled::Walk(prefix);
        }
        if c == n {
            counters.continues += 1;
            counters.compiled_retired += n;
            return Compiled::Done {
                count: n,
                stop: false,
            };
        }
        // A compiled store that leaves the dispatch word set stops the block
        // after its instruction (`jit_abi::machine::ACCESS_STOP`); a prefix
        // that did so cannot continue.
        let stop = c != 0 && exit_word.load(Ordering::Relaxed) != 0;
        if (kind == 0 || stop) && c != 0 {
            counters.stops += 1;
            counters.compiled_retired += c;
            return Compiled::Done {
                count: c,
                stop: true,
            };
        }
        counters.unconditional_defers += 1;
        Compiled::Walk(self.deferred::<ROUTED, ORDER>(vm, ir, c, remaining, exit_word))
    }
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
                slot.hot = 0;
                slot.request = Request::Waiting(seen);
            }
            crate::TierUpSubmit::Busy => {
                vm.cache.machine_request(id);
                vm.cache.abandon_machine_tier_up(key, &vm.mem);
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
