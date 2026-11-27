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

pub(super) struct Unit {
    // Keeps the block's address from being reused while it names this unit.
    _ir: Arc<IrBlock>,
    entry: Option<(crate::CompiledPtr, u64)>,
    failed: bool,
    delay: u32,
    runs: u64,
}

/// The compiled Machine units of one vCPU (decision-29 §8, "J1 units are per
/// vCPU"). Each unit is registered with the backend's code lifetime and owned
/// here: it is retired when its block is gone, when the map is cleared, when
/// the VM's code lifetime changes, and when the vCPU is dropped. Every call
/// runs inside this vCPU's execution participation, so retired code is not
/// reclaimed while it runs (spec §9.1; decision-17).
#[derive(Default)]
pub(super) struct State {
    units: EngineMap<usize, Unit>,
    lifetime: Option<Arc<crate::code_lifetime::CodeLifetime>>,
    participant: Option<crate::code_lifetime::ExecutionParticipant>,
    inserted: u64,
    pub(super) counters: MachineJitCounters,
}

/// The map's bound and how often it drops the units of released blocks.
const UNITS: usize = 512;
const SWEEP: u64 = 64;

impl State {
    /// Retire the units `keep` rejects, in one retirement.
    fn retire_where(&mut self, keep: impl Fn(&Unit) -> bool) {
        let mut retired = Vec::new();
        self.units.retain(|_, unit| {
            if keep(unit) {
                return true;
            }
            if let Some((entry, _)) = unit.entry {
                retired.push(crate::CachedBlock::Compiled { entry });
            }
            false
        });
        if let Some(lifetime) = &self.lifetime {
            lifetime.discard_many(&retired);
        }
        self.counters.retired += retired.len() as u64;
    }

    /// At run entry: follow the VM's code lifetime, retiring every unit of a
    /// previous one, and hold a participation in it.
    pub(super) fn begin_run(&mut self, lifetime: Arc<crate::code_lifetime::CodeLifetime>) {
        if self
            .lifetime
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, &lifetime))
        {
            return;
        }
        self.retire_where(|_| false);
        self.participant = Some(lifetime.participant());
        self.lifetime = Some(lifetime);
    }
}

impl Drop for State {
    fn drop(&mut self) {
        self.retire_where(|_| false);
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
    /// unit, or `None` for the interpreted walk. The second value is a STOP,
    /// whose turn must not link its successor.
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
    ) -> Option<(Progress, bool)> {
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
        let state = &mut execution.machine_jit;
        if !eligible {
            state.counters.refused += 1;
            return None;
        }
        let key = Arc::as_ptr(ir) as usize;
        if !state.units.contains_key(&key) {
            state.inserted += 1;
            if state.units.len() >= UNITS {
                state.retire_where(|_| false);
            } else if state.inserted.is_multiple_of(SWEEP) {
                // A unit whose block only this map still holds is never
                // resolved again: its proof and links are gone.
                state.retire_where(|unit| Arc::strong_count(&unit._ir) > 1);
            }
        }
        let delay = match vm.machine_jit {
            MachineJitMode::Delayed(k) => k,
            _ => 0,
        };
        let unit = state.units.entry(key).or_insert_with(|| Unit {
            _ir: ir.clone(),
            entry: None,
            failed: false,
            delay,
            runs: 0,
        });
        unit.runs += 1;
        let runs = unit.runs;
        if let MachineJitMode::Scripted(seed) = vm.machine_jit {
            return Some(self.scripted::<ROUTED, ORDER>(vm, ir, remaining, seed, runs, exit_word));
        }
        // Participation precedes probing or calling any executable pointer
        // (`code_lifetime::ExecutionParticipant`); it lasts through the call.
        let mut participant = state.participant.take().expect("a Machine run participant");
        let guard = participant.enter();
        if let Some((entry, at)) = unit.entry {
            if at != guard.epoch() {
                // Another unit's retirement advanced the epoch; this one may
                // still be live. A retired one is dropped, never called.
                unit.entry = guard.admits(entry).then_some((entry, guard.epoch()));
            }
        }
        if unit.entry.is_none() && !unit.failed {
            let config = crate::jit_abi::machine::access_config(
                crate::legacy::physical_mask(context.generation),
                ROUTED,
                ORDER,
            );
            match vm.backend.materialize_machine(ir, vm.consistency, config) {
                // A unit compiled under an older epoch is checked as above
                // at its next entry.
                Some(entry) => {
                    unit.entry = Some((entry, guard.epoch()));
                    state.counters.compiled += 1;
                }
                None => unit.failed = true,
            }
        }
        let entry = match unit.entry {
            Some((entry, at)) if at == guard.epoch() && unit.delay == 0 => entry,
            _ => {
                if unit.entry.is_some() && unit.delay != 0 {
                    unit.delay -= 1;
                }
                drop(guard);
                state.participant = Some(participant);
                return None;
            }
        };
        state.counters.entries += 1;
        ctx.machine_ordinal = 0;
        if vm.time_enabled {
            crate::time_stats::compiled();
        }
        // SAFETY: `entry` is a Machine unit compiled to this ABI and live in
        // the guard's epoch, which this vCPU announces until the call returns.
        let code = unsafe { crate::jit_abi::call_block_in_mode(entry, &mut self.cpu, ctx) };
        drop(guard);
        self.legacy_execution
            .as_mut()
            .unwrap()
            .machine_jit
            .participant = Some(participant);
        if vm.time_enabled {
            crate::time_stats::interpreted();
        }
        let ordinal = ctx.machine_ordinal;
        Some(match code {
            RET_CONTINUE => {
                let counters = &mut self.legacy_execution.as_mut().unwrap().machine_jit.counters;
                counters.continues += 1;
                counters.compiled_retired += n;
                (machine_jit::completed(n), false)
            }
            RET_MACHINE_STOP => {
                let counters = &mut self.legacy_execution.as_mut().unwrap().machine_jit.counters;
                counters.stops += 1;
                counters.compiled_retired += ordinal;
                debug_assert!(ordinal != 0 && ordinal < n, "a STOP inside its block");
                (machine_jit::completed(ordinal), true)
            }
            RET_MACHINE_DEFER => {
                let c = ordinal & !HELPER_DEFER;
                let counters = &mut self.legacy_execution.as_mut().unwrap().machine_jit.counters;
                if ordinal & HELPER_DEFER != 0 {
                    counters.helper_defers += 1;
                } else {
                    counters.unconditional_defers += 1;
                }
                (
                    self.deferred::<ROUTED, ORDER>(vm, ir, c, remaining, exit_word),
                    false,
                )
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
    ) -> (Progress, bool) {
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
            return (prefix, false);
        }
        if c == n {
            counters.continues += 1;
            counters.compiled_retired += n;
            return (prefix, false);
        }
        // A compiled store that leaves the dispatch word set stops the block
        // after its instruction (`jit_abi::machine::ACCESS_STOP`); a prefix
        // that did so cannot continue.
        let stop = c != 0 && exit_word.load(Ordering::Relaxed) != 0;
        if (kind == 0 || stop) && c != 0 {
            counters.stops += 1;
            counters.compiled_retired += c;
            return (prefix, true);
        }
        counters.unconditional_defers += 1;
        (
            self.deferred::<ROUTED, ORDER>(vm, ir, c, remaining, exit_word),
            false,
        )
    }
}
