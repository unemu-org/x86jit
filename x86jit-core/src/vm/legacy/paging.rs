//! Internal paging fetch/proofs (paging design §§3.2, 3.5;
//! spec.md §17.10). No shared dispatch entry or compiled materialization.
use super::*;
#[cfg(feature = "test-hooks")]
use crate::interp::legacy::paging::execute_for_test;
use crate::interp::legacy::paging::Insn;
use crate::interp::legacy::paging::{publish_cr2, translate_fetch};
use std::cell::Cell;

// Checked V86 loops retain one proof per instruction, including multiple
// unrolled bodies. Keep their working set independently of context identities.
// Proofs outlive TLB evictions (xj-to9.70), so a paged kernel's working set
// across its interrupt, scheduler and driver paths stays proved: up to 4096
// blocks, and at most 16 MiB of IR (`flat32::weight`), which 4096 blocks of
// 4 KiB or 500 of 32 instructions reach.
const MAX_BLOCKS: usize = 4096;
const MAX_BLOCK_BYTES: usize = 16 << 20;
const MAX_CONTEXTS: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq)]
struct Key {
    segment_class: crate::SegmentClass,
    // The singleton reference entry (`interpret_*_for_test`) never shares
    // an ordinary multi-instruction block.
    ordinary: bool,
    decode: LegacyBlockKey,
    cr3: u32,
    // `LegacyContextStamp::new_paged`'s CR0: MP, EM and TS clear.
    cr0: u32,
}
#[cfg(feature = "test-hooks")]
impl super::diag::DiagKey for Key {
    fn site(&self) -> u64 {
        super::diag::DiagKey::site(&self.decode)
    }
    fn fields(&self) -> Vec<(&'static str, u64)> {
        let mut fields = super::diag::DiagKey::fields(&self.decode);
        fields.extend([
            ("class", self.segment_class as u64),
            ("ordinary", u64::from(self.ordinary)),
            ("cr3", u64::from(self.cr3)),
            ("cr0", u64::from(self.cr0)),
        ]);
        fields
    }
}

impl Hash for Key {
    // Every field, packed losslessly: one `BlockHasher` round per word.
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.decode.hash(state);
        state.write_u64(u64::from(self.cr3) | (u64::from(self.cr0) << 32));
        state.write_u64(self.segment_class as u64 | (u64::from(self.ordinary) << 8));
    }
}

struct Proof {
    ir: Arc<IrBlock>,
    // Computed once at lift: whether the block takes the ordinary walk.
    ordinary: bool,
    // An ordinary block's instruction table, built with the proof.
    lanes: Option<Arc<[Insn]>>,
    permit: Option<crate::FetchPermit>,
    // Each code page's first fetched byte: its linear address and the bus
    // address its fetch translated it to (xj-to9.70).
    pages: Box<[(u32, u64)]>,
    // The TLB generation at which every page of `pages` last translated to
    // its bus address. While the generation holds, no entry a fetch used
    // has left the TLB, so the translations still hold.
    validated: Cell<u64>,
    // Ordered post-A20 bus spans, never a linear length applied to one frame.
    _registration: crate::memory::CodeRegistration,
}

#[derive(Default)]
pub(super) struct State {
    // The TLB generation last reconciled.
    generation: u64,
    pub(super) tlb: std::cell::RefCell<crate::interp::legacy::paging::Tlb>,
    mapping: Option<Arc<()>>,
    stamp: Option<LegacyContextStamp>,
    pub(super) context_id: u64,
    // Each retained stamp's identity id and last use (`uses`); ids are never
    // reused before `invalidate`, which restarts `next_context`.
    contexts: EngineMap<LegacyContextStamp, (u64, u64)>,
    next_context: u64,
    model: Option<LegacyCpuModel>,
    memory: usize,
    blocks: super::proof_map::ProofMap<Key, Proof>,
    // The block `resolve` last returned a proof for and the map's version
    // then (`holds`).
    last: Option<(usize, u64)>,
    pub(super) lifts: u64,
    before_publication: Option<fn(&Vm)>,
    port_input: Option<crate::interp::legacy::paging::PortPlan>,
    pub(super) transaction: Option<crate::interp::legacy::paging::Transaction>,
    pub(super) event_transaction: Option<crate::interp::legacy::paging::Transaction>,
    // Every entry is a member of `contexts`; the least recently used one
    // (`uses`) leaves first.
    recent: Box<[Option<Recent>; 4]>,
    uses: u64,
    // `LegacyContextStamp::quick` of `stamp`.
    quick: u64,
    // The selection serial (`Execution::selection`) at which the per-turn
    // resolve last compared `stamp` with the live state; `None` after any
    // other comparison (xj-to9.59).
    synced: Option<u64>,
    // Resolves that found the serial unchanged and compared nothing.
    #[cfg(test)]
    skips: u64,
    // The instruction tables of ordinary blocks resolve returned from a
    // proof, direct-mapped by block address (`lanes_slot`) and empty until
    // the first. A slot names the map's version when it was filled: an
    // unchanged version means the proof that held the block still holds it,
    // alive, so its address names that block (`holds`). Successor links
    // name the version too, so a block run after any change of it was
    // resolved again first and its slot refilled.
    lane_slots: Vec<Option<LaneSlot>>,
    // The table of an ordinary block run without one in `lane_slots`.
    lane_scratch: Vec<Insn>,
}

struct LaneSlot {
    ir: usize,
    version: u64,
    insns: Arc<[Insn]>,
}

const LANE_SLOTS: usize = 512;

/// The `lane_slots` index of the block at `ir`.
#[inline(always)]
fn lanes_slot(ir: *const IrBlock) -> usize {
    ((ir as usize >> 4).wrapping_mul(0x9e37_79b9) >> 7) & (LANE_SLOTS - 1)
}

/// Keep the table of the block a proof of map version `version` just
/// returned, for its run (`State::lane_slots`).
#[inline(always)]
fn remember_lanes(
    slots: &mut Vec<Option<LaneSlot>>,
    version: u64,
    ir: &Arc<IrBlock>,
    lanes: Option<&Arc<[Insn]>>,
) {
    let Some(lanes) = lanes else {
        return;
    };
    if slots.is_empty() {
        slots.resize_with(LANE_SLOTS, || None);
    }
    let address = Arc::as_ptr(ir) as usize;
    let slot = &mut slots[lanes_slot(Arc::as_ptr(ir))];
    if !slot
        .as_ref()
        .is_some_and(|slot| slot.ir == address && slot.version == version)
    {
        *slot = Some(LaneSlot {
            ir: address,
            version,
            insns: lanes.clone(),
        });
    }
}

#[derive(Clone, Copy)]
struct Recent {
    stamp: LegacyContextStamp,
    id: u64,
    quick: u64,
    used: u64,
    // The `Execution` identity id whose complete stamp this is, or
    // `u64::MAX` (`Execution::paging_selection`).
    complete: u64,
}

impl State {
    /// Advances whenever `blocks` may drop or replace a proof, and with it
    /// its registration, or `contexts` may give an identity to another
    /// stamp (it clears `blocks` then); successor links (`links.rs`) name it.
    pub(super) fn version(&self) -> u64 {
        self.blocks.version()
    }

    pub(super) fn reset(&mut self) {
        self.invalidate();
        self.transaction = None;
        self.event_transaction = None;
        self.port_input = None;
    }

    /// Whether a proof holds this very block. Links record the block the
    /// last resolve returned: the map's version advances whenever a proof
    /// may be dropped or replaced, so an unchanged version means the proof
    /// that returned it still holds it, alive, and its address names it.
    /// It does not mark the proof used: the resolve that returned the block
    /// just did (a hit) or just inserted it, and the link made from it then
    /// holds the block, which `make_room`'s `hot` counts while it does.
    pub(super) fn holds(&self, ir: &Arc<IrBlock>) -> bool {
        self.last == Some((Arc::as_ptr(ir) as usize, self.blocks.version()))
    }

    pub(super) fn invalidate(&mut self) {
        self.blocks.clear(why!(Paging));
        // Every slot names an older version now; drop the tables they keep.
        self.lane_slots.clear();
        self.tlb.get_mut().invalidate();
        self.generation = self.tlb.get_mut().generation;
        self.stamp = None;
        self.synced = None;
        self.contexts.clear();
        self.next_context = 0;
        *self.recent = [None; 4];
    }

    /// Brings the context identity, the TLB context and the proofs up to
    /// the live state. `selection` is `Execution::paging_selection` for the
    /// dispatcher's per-turn resolve, `None` for every other caller: while
    /// its serial is the one this identity was last compared at, no stamped
    /// field has changed, and the stamp is not compared again; an identity
    /// id it names is the live state's complete stamp, so the recent entry
    /// recorded for that id is it, unread (xj-to9.59).
    pub(super) fn reconcile(
        &mut self,
        vm: &Vm,
        cpu: &CpuState,
        model: LegacyCpuModel,
        selection: Option<(u64, Option<u64>)>,
    ) {
        let memory = &vm.mem as *const Memory as usize;
        if self.model != Some(model)
            || self.memory != memory
            || self
                .mapping
                .as_ref()
                .is_none_or(|m| !vm.mem.is_mapping_identity(m))
        {
            self.blocks
                .reset(vm.cache.proof_generation(), why!(Mapping));
            self.invalidate();
            self.model = Some(model);
            self.memory = memory;
            self.mapping = Some(vm.mem.mapping_identity());
        }
        // A write drops the proofs whose bus sources it overlaps.
        self.blocks.reconcile(&vm.cache);
        let serial = selection.map(|(serial, _)| serial);
        if serial.is_none() || self.synced != serial || self.stamp.is_none() {
            let complete = selection.and_then(|(_, id)| id);
            self.reconcile_identity(cpu, model, complete);
            self.synced = serial;
        } else {
            // Every writer of a stamped field outside a block walk marks it
            // (`Execution::context_dirty`); a walk reports its own. Every
            // test build checks that none was missed.
            #[cfg(any(test, debug_assertions))]
            assert!(
                self.stamp
                    .as_ref()
                    .is_some_and(|s| s.matches_paged(cpu, model)),
                "a stamped field changed at an unchanged selection serial"
            );
            #[cfg(test)]
            {
                self.skips += 1;
            }
        }
        self.tlb.get_mut().reconcile(cpu);
        self.reconcile_generation();
    }

    /// The TLB context alone, after a walk changed the CPU context: the
    /// identity waits for the next resolve, after the dispatcher's
    /// `sync_class` has advanced the selection serial.
    pub(super) fn reconcile_translation(&mut self, cpu: &CpuState) {
        self.tlb.get_mut().reconcile(cpu);
        self.reconcile_generation();
    }

    // The identity of the complete stamp of `cpu`: the last one, a recent
    // one or a retained one, compared where it is kept (`quick` first), or a
    // new one.
    fn reconcile_identity(&mut self, cpu: &CpuState, model: LegacyCpuModel, complete: Option<u64>) {
        if let Some(complete) = complete {
            if let Some(slot) = self
                .recent
                .iter()
                .position(|r| r.as_ref().is_some_and(|r| r.complete == complete))
            {
                let entry = self.recent[slot].as_mut().unwrap();
                debug_assert!(entry.stamp.matches_paged(cpu, model));
                if self.context_id != entry.id || self.stamp.is_none() {
                    self.uses += 1;
                    entry.used = self.uses;
                    self.context_id = entry.id;
                    self.stamp = Some(entry.stamp);
                    self.quick = entry.quick;
                }
                return;
            }
        }
        let quick = LegacyContextStamp::quick_paged(cpu);
        let complete = complete.unwrap_or(u64::MAX);
        if self.quick == quick
            && self
                .stamp
                .as_ref()
                .is_some_and(|s| s.matches_paged(cpu, model))
        {
            if complete != u64::MAX {
                if let Some(entry) = self
                    .recent
                    .iter_mut()
                    .flatten()
                    .find(|r| r.id == self.context_id)
                {
                    entry.complete = complete;
                }
            }
            return;
        }
        self.uses += 1;
        if let Some(slot) = self.recent.iter().position(|r| {
            r.as_ref()
                .is_some_and(|r| r.quick == quick && r.stamp.matches_paged(cpu, model))
        }) {
            let entry = self.recent[slot].as_mut().unwrap();
            entry.used = self.uses;
            if complete != u64::MAX {
                entry.complete = complete;
            }
            self.context_id = entry.id;
            self.stamp = Some(entry.stamp);
            self.quick = quick;
            return;
        }
        let stamp = LegacyContextStamp::new_paged(cpu, model);
        let id = if let Some((id, used)) = self.contexts.get_mut(&stamp) {
            *used = self.uses;
            *id
        } else {
            // Bound context identities as well as blocks: evict the least
            // recently used identity and only its proofs, as the
            // dispatcher's retained contexts do (`Execution::enter_context`).
            // A guest with many address spaces keeps a user and a kernel
            // stamp per CR3, and its kernel's proofs stay (xj-to9.70).
            if self.contexts.len() == MAX_CONTEXTS {
                self.evict_context();
            }
            let id = self.next_context;
            self.next_context += 1;
            self.contexts.insert(stamp, (id, self.uses));
            id
        };
        let slot = self
            .recent
            .iter()
            .position(Option::is_none)
            .unwrap_or_else(|| {
                (0..self.recent.len())
                    .min_by_key(|&i| self.recent[i].as_ref().unwrap().used)
                    .unwrap()
            });
        self.recent[slot] = Some(Recent {
            stamp,
            id,
            quick,
            used: self.uses,
            complete,
        });
        self.context_id = id;
        self.stamp = Some(stamp);
        self.quick = quick;
    }

    // Every advance, a flush of a new translation context (CR3, the paging
    // bits of CR0, A20, the model) included, only makes the proofs validated
    // before it translate their code pages again on their next hit
    // (`resolve`): a proof whose pages walk to the frames it was fetched
    // from still holds its bytes, which their physical registration guards.
    // Only `invalidate` (a reset, a mapping change or the embedder's
    // explicit invalidation) drops them.
    fn reconcile_generation(&mut self) {
        self.generation = self.tlb.get_mut().generation;
    }

    /// A guest CR3 load, equal values included: flush the TLB and keep the
    /// proofs, which revalidate (xj-to9.70). A load of another value also
    /// changes the context identity, which names CR3.
    pub(super) fn load_cr3(&mut self) {
        self.tlb.get_mut().load_cr3();
        self.generation = self.tlb.get_mut().generation;
        self.synced = None;
    }

    /// The dispatcher restarted its identity ids
    /// (`Execution::invalidate_translations`, after a CR3 load or an epoch
    /// rollover): none names a recent entry any more.
    pub(super) fn forget_dispatch_ids(&mut self) {
        self.synced = None;
        for entry in self.recent.iter_mut().flatten() {
            entry.complete = u64::MAX;
        }
    }

    // The least recently used identity leaves with its proofs. The recent
    // entries' uses are the latest; the map records the rest.
    #[cold]
    fn evict_context(&mut self) {
        for recent in self.recent.iter().flatten() {
            if let Some((_, used)) = self.contexts.get_mut(&recent.stamp) {
                *used = (*used).max(recent.used);
            }
        }
        let Some((&stamp, &(id, _))) = self.contexts.iter().min_by_key(|(_, &(_, used))| used)
        else {
            return;
        };
        self.contexts.remove(&stamp);
        for slot in self.recent.iter_mut() {
            if slot.is_some_and(|r| r.id == id) {
                *slot = None;
            }
        }
        self.blocks
            .remove_where(|key| key.decode.context_id == id, why!(ContextEvicted));
        self.lane_slots.clear();
    }

    fn key(
        &self,
        cpu: &CpuState,
        model: LegacyCpuModel,
        context: MachineContext,
        owner_id: u64,
        ordinary: bool,
    ) -> Result<Key, Exit> {
        let linear_fetch = crate::legacy::access(
            cpu,
            context.generation,
            SegmentId::Cs,
            cpu.rip,
            1,
            crate::AccessKind::Execute,
        )? as u32;
        Ok(Key {
            segment_class: context.segment_class,
            ordinary,
            decode: LegacyBlockKey {
                owner_id,
                context_id: self.context_id,
                model_semantics_id: model.model_semantics_id,
                decode_bits: context.default_operand_bits,
                stack_bits: context.stack_address_bits,
                regime: context.regime,
                decode_ip: cpu.rip as u32,
                linear_fetch,
                mxcsr_mode: cpu.mxcsr & crate::jit_abi::fp::MODE_MASK,
            },
            cr3: cpu.legacy.cr3,
            cr0: cpu.legacy.cr0 & super::PAGED_IDENTITY_CR0,
        })
    }

    #[cold]
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn resolve(
        &mut self,
        vm: &Vm,
        cpu: &mut CpuState,
        model: LegacyCpuModel,
        context: MachineContext,
        owner_id: u64,
        exit_word: &AtomicU64,
        ordinary: bool,
        selection: Option<(u64, Option<u64>)>,
    ) -> Result<(Arc<IrBlock>, bool), Exit> {
        let walk_wrote = Cell::new(false);
        #[cfg(feature = "test-hooks")]
        let mut retry = None;
        loop {
            vm.handle_smc(exit_word);
            self.reconcile(vm, cpu, model, selection);
            // The dispatcher's context notes and the TLB's generation
            // advances, which only this resolver reaches on paged paths.
            #[cfg(feature = "test-hooks")]
            {
                super::diag::drain_contexts(&vm.live_fields.diag);
                vm.live_fields.diag.tlb(&mut self.tlb.get_mut().advances);
            }
            let key = self.key(cpu, model, context, owner_id, ordinary)?;
            let generation = self.tlb.get_mut().generation;
            let mut moved = false;
            if let Some(proof) = self
                .blocks
                .get(&key)
                .filter(|p| p.permit.as_ref().is_none_or(|p| p.is_current()))
            {
                // A fetched entry left the TLB since this proof's code pages
                // last translated: translate them again, as the fetch would.
                let mut held = true;
                if proof.validated.get() != generation {
                    let wrote = Cell::new(false);
                    held = pages_hold(vm, cpu, &proof.pages, &self.tlb, &wrote);
                    walk_wrote.set(walk_wrote.get() | wrote.get());
                    #[cfg(feature = "test-hooks")]
                    vm.live_fields.diag.revalidated(held);
                    if wrote.get() {
                        // A walk's A update may have written code: drain it
                        // and look again (the entries now hit).
                        continue;
                    }
                    if held && self.tlb.get_mut().generation == generation {
                        proof.validated.set(generation);
                    }
                }
                if held {
                    probe!(PROOFS);
                    let version = self.blocks.version();
                    self.last = Some((Arc::as_ptr(&proof.ir) as usize, version));
                    remember_lanes(
                        &mut self.lane_slots,
                        version,
                        &proof.ir,
                        proof.lanes.as_ref(),
                    );
                    return Ok((proof.ir.clone(), proof.ordinary));
                }
                moved = true;
            }
            if moved {
                self.blocks.remove(&key, why!(Translation));
            }
            probe!(LIFTS);
            #[cfg(feature = "test-hooks")]
            self.blocks.record_lift(
                vm,
                super::diag::ProofClass::Paged,
                &key,
                retry.take(),
                key.decode.decode_ip,
                context,
            );
            let source = crate::memory::LiftSource::new(&vm.mem);
            let generation = vm.cache.proof_generation();
            let permit = vm.mem.fetch_permit();
            let mut live = super::live::Plan::default();
            // The generation the lift's translations hold at: a lift whose
            // walks replace a fetched entry revalidates on its first hit.
            let validated = self.tlb.get_mut().generation;
            let (ir, sources, pages) = lift(
                vm,
                cpu,
                context,
                &source,
                &walk_wrote,
                &self.tlb,
                ordinary,
                &mut live,
            )
            .map_err(|mut exit| {
                if walk_wrote.get() {
                    if let Exit::UnmappedMemory {
                        metadata: Some(m), ..
                    }
                    | Exit::Exception {
                        metadata: Some(m), ..
                    } = &mut exit
                    {
                        m.progress.ram = crate::fault::MemoryEffects::PossiblyPartial;
                    }
                }
                exit
            })?;
            if !source.validate() || permit.as_ref().is_some_and(|p| !p.is_current()) {
                #[cfg(feature = "test-hooks")]
                {
                    retry = Some(super::diag::RetryKind::SourceChanged);
                }
                continue;
            }
            if let Some(hook) = self.before_publication.take() {
                hook(vm);
            }
            #[cfg(test)]
            vm.after_source_read();
            vm.handle_smc(exit_word);
            // A/D CAS can touch executable table storage during this fetch.
            // Its notifications must be drained before physical registration.
            if vm.cache.proof_generation() != generation {
                #[cfg(feature = "test-hooks")]
                {
                    retry = Some(super::diag::RetryKind::Registration);
                }
                continue;
            }
            #[cfg(feature = "test-hooks")]
            let promoting = live.promotes();
            if !vm.live_fields.commit(&vm.cache, &vm.mem, live) {
                #[cfg(feature = "test-hooks")]
                {
                    retry = Some(if promoting {
                        super::diag::RetryKind::Promotion
                    } else {
                        super::diag::RetryKind::Demotion
                    });
                }
                continue;
            }
            self.reconcile_generation();
            let key = self.key(cpu, model, context, owner_id, ordinary)?;
            self.lifts += 1;
            let link_exit = super::links::link_exit(&ir);
            let mut ir = ir;
            // The lift's spare capacity would count against the budget.
            ir.ops.shrink_to_fit();
            let ir = Arc::new(IrBlock { link_exit, ..ir });
            if !source.publishable() {
                #[cfg(feature = "test-hooks")]
                vm.live_fields.diag.uncached(super::diag::ProofClass::Paged);
                // The interpreter may use this snapshot once, without a proof.
                self.blocks.remove_unpublished(&key);
                self.last = None;
                let ir = retry_instruction(&ir, cpu.rip);
                let ordinary = ordinary && ordinary_block(context, &ir);
                return Ok((ir, ordinary));
            }
            let ordinary = ordinary && ordinary_block(context, &ir);
            let mut registration = None;
            if !vm.cache.register_private_sources(generation, || {
                registration = Some(vm.mem.register_code_sources(&sources));
            }) || permit.as_ref().is_some_and(|p| !p.is_current())
            {
                #[cfg(feature = "test-hooks")]
                {
                    retry = Some(super::diag::RetryKind::Registration);
                }
                continue;
            }
            let lanes = ordinary.then(|| crate::interp::legacy::paging::lanes(&ir));
            let weight = super::flat32::weight(&ir, sources.len())
                + lanes
                    .as_ref()
                    .map_or(0, |l| l.len() * std::mem::size_of::<Insn>());
            if !self.blocks.contains_key(&key) {
                self.blocks
                    .make_room(MAX_BLOCKS, MAX_BLOCK_BYTES, weight, why!(Bound), |proof| {
                        Arc::strong_count(&proof.ir) > 1
                    });
            }
            self.blocks.insert(
                key,
                Proof {
                    ir: ir.clone(),
                    ordinary,
                    lanes: lanes.clone(),
                    permit,
                    pages: pages.into_boxed_slice(),
                    validated: Cell::new(validated),
                    _registration: registration.unwrap(),
                },
                &sources,
                weight,
            );
            self.last = Some((Arc::as_ptr(&ir) as usize, self.blocks.version()));
            let version = self.blocks.version();
            remember_lanes(&mut self.lane_slots, version, &ir, lanes.as_ref());
            return Ok((ir, ordinary));
        }
    }
}

/// The vCPU's paging state, made on first use.
#[inline(always)]
fn paging_state(paging: &mut Option<Box<State>>) -> &mut State {
    if paging.is_none() {
        new_paging_state(paging);
    }
    paging.as_deref_mut().unwrap()
}

#[cold]
#[inline(never)]
fn new_paging_state(paging: &mut Option<Box<State>>) {
    *paging = Some(Box::default());
}

/// Whether a paged block takes the ordinary Flat32, Segmented or V86 walk.
fn ordinary_block(context: MachineContext, ir: &IrBlock) -> bool {
    ordinary_context(context) && crate::interp::legacy::flat32_ordinary(ir)
}

/// Paged contexts whose ordinary instructions share blocks. V86 addresses
/// are live paragraph arithmetic (`legacy::virtual8086_access`), so it needs
/// no segment proof; its sensitive instructions stay singletons.
fn ordinary_context(context: MachineContext) -> bool {
    matches!(
        context.segment_class,
        crate::SegmentClass::Flat32 | crate::SegmentClass::Segmented
    ) || context.regime == MachineRegime::Virtual8086
}

// Decision-27 order 2: use the singleton fetch boundary for every required
// instruction, retaining one TLB token and physical publication for the block.
#[allow(clippy::too_many_arguments)]
fn lift(
    vm: &Vm,
    cpu: &CpuState,
    context: MachineContext,
    source: &crate::memory::LiftSource<'_>,
    walk_wrote: &Cell<bool>,
    tlb: &std::cell::RefCell<crate::interp::legacy::paging::Tlb>,
    blocks: bool,
    live: &mut super::live::Plan,
) -> Result<Lifted, Exit> {
    let mut cursor = cpu.clone();
    let linear = cpu.legacy.segments[0].base.wrapping_add(cpu.rip as u32);
    let page_end = cpu.rip + (4096 - u64::from(linear & 4095));
    let mut block: Option<IrBlock> = None;
    let mut sources = Vec::new();
    let mut pages = Vec::new();
    let mut reloaded = 0;
    loop {
        if cursor.rip + 15 > page_end {
            if let Some(block) = block.take() {
                return Ok((block, sources, pages));
            }
        }
        let fetched = fetch(vm, &mut cursor, context, source, walk_wrote, tlb);
        let (bytes, spans, fetched_pages) = match fetched {
            Ok(fetched) => fetched,
            Err(exit) => return block.map(|b| (b, sources, pages)).ok_or(exit),
        };
        // Each byte's bus address, as this fetch translated it.
        let bus = |offset: u32| {
            let mut offset = u64::from(offset);
            for &(start, len) in &spans {
                if offset < u64::from(len) {
                    return start + offset;
                }
                offset -= u64::from(len);
            }
            unreachable!("a fetched byte has a bus span")
        };
        let mut instruction = match vm.live_fields.instruction(
            &bytes,
            cursor.rip as u32,
            context,
            vm.machine_model_profile(context.generation),
            crate::lift::legacy::alignment_mask(&cursor),
            bus,
            live,
        ) {
            Ok(ir) => ir,
            Err(error) => {
                return block
                    .map(|b| (b, sources, pages))
                    .ok_or_else(|| direct_refusal(&cursor, error))
            }
        };
        let loads = super::flat32::specialize_segments(&mut instruction, cpu, reloaded);
        let ordinary = blocks
            && ordinary_context(context)
            && crate::interp::legacy::flat32_ordinary(&instruction);
        if !ordinary {
            if let Some(block) = block.take() {
                return Ok((block, sources, pages));
            }
        }
        sources.extend(spans);
        for page in fetched_pages {
            if !pages
                .iter()
                .any(|&(linear, _)| linear >> 12 == page.0 >> 12)
            {
                pages.push(page);
            }
        }
        let mut decoder = Decoder::with_ip(
            u32::from(context.default_operand_bits),
            &bytes,
            cursor.rip,
            crate::model::decode::SYSTEM_DECODER_OPTIONS,
        );
        let insn = decoder.decode();
        cursor.rip += u64::from(instruction.guest_len);
        reloaded = loads;
        if let Some(block) = &mut block {
            block.guest_len += instruction.guest_len;
            block.icount += 1;
            block.temp_count = block.temp_count.max(instruction.temp_count);
            block.ops.extend(instruction.ops);
        } else {
            block = Some(instruction);
        }
        if !ordinary
            || insn.flow_control() != iced_x86::FlowControl::Next
            || cursor.rip >= page_end
            || block.as_ref().unwrap().icount == 32
        {
            return Ok((block.unwrap(), sources, pages));
        }
    }
}

/// A lifted block, its ordered bus spans and its code pages: each page's
/// first fetched byte, linear and as translated (`Proof::pages`).
type Lifted = (IrBlock, Vec<(u64, u32)>, Vec<(u32, u64)>);
type Fetched = (Vec<u8>, Vec<(u64, u32)>, Vec<(u32, u64)>);

/// Whether every code page of a proof still translates its first fetched
/// byte `linear` to the bus address `bus` it was fetched from: through a
/// permitted TLB entry, or else through the walk a fetch would make (which
/// sets A bits, reported in `wrote`, and fills the TLB). A walk that faults
/// is a mismatch: the relift fetches again and takes the fault. Same linear
/// page, another frame (a remapped PTE after INVLPG or an eviction) is a
/// mismatch even when the bytes are equal (paging design §3.5).
fn pages_hold(
    vm: &Vm,
    cpu: &CpuState,
    pages: &[(u32, u64)],
    tlb: &std::cell::RefCell<crate::interp::legacy::paging::Tlb>,
    wrote: &Cell<bool>,
) -> bool {
    let mut walker: Option<CpuState> = None;
    for &(linear, bus) in pages {
        let hit = tlb.borrow_mut().fetch_hit(cpu, linear);
        let found = match hit {
            Some(physical) => crate::legacy::physical_span(cpu, u32::MAX, u64::from(physical), 1).0,
            None => {
                let walker = walker.get_or_insert_with(|| cpu.clone());
                match translate_fetch(walker, &vm.mem, linear, wrote, Some(tlb)) {
                    Ok(found) => found,
                    Err(_) => return false,
                }
            }
        };
        if found != bus {
            return false;
        }
    }
    true
}

#[cold]
#[inline(never)]
fn fetch(
    vm: &Vm,
    cpu: &mut CpuState,
    context: MachineContext,
    source: &crate::memory::LiftSource<'_>,
    walk_wrote: &Cell<bool>,
    tlb: &std::cell::RefCell<crate::interp::legacy::paging::Tlb>,
) -> Result<Fetched, Exit> {
    let ip = cpu.rip;
    let mut bytes = Vec::with_capacity(15);
    let mut sources: Vec<(u64, u32)> = Vec::new();
    let mut pages: Vec<(u32, u64)> = Vec::new();
    // 80386 PRM §9.8.13 p. 9-17; Intel486 PRM §9.9.13 p. 9-21:
    // the maximum instruction length is fifteen bytes.
    for i in 0..15 {
        let linear = crate::legacy::access(
            cpu,
            context.generation,
            SegmentId::Cs,
            ip + i,
            1,
            crate::AccessKind::Execute,
        )?;
        let bus = if context.paging_enabled {
            translate_fetch(cpu, &vm.mem, linear as u32, walk_wrote, Some(tlb))?
        } else {
            crate::legacy::physical_span(cpu, u32::MAX, linear, 1).0
        };
        // Only a translated fetch has pages to revalidate.
        if context.paging_enabled
            && pages
                .last()
                .is_none_or(|&(page, _)| page >> 12 != linear as u32 >> 12)
        {
            pages.push((linear as u32, bus));
        }
        if let Some(policy) = &vm.mem.fetch_policy {
            let verdict = policy.approve(bus >> crate::memory::CODE_PAGE_BITS, &vm.mem);
            if verdict != crate::FetchVerdict::Allow {
                return Err(crate::fault::qualify_legacy_bus(
                    crate::fault::fetch_exit(
                        &vm.mem,
                        FetchAddr { pa: bus, ip },
                        CpuMode::Compat32,
                        bus,
                        Some(crate::lift::FetchFailure::Policy(verdict)),
                    ),
                    linear,
                    bus,
                ));
            }
        }
        source.watch(bus, 1);
        let mut byte = [0];
        vm.mem.fetch_code(bus, &mut byte).map_err(|trap| {
            crate::fault::qualify_legacy_bus(
                crate::fault::fetch_exit(
                    &vm.mem,
                    FetchAddr { pa: bus, ip },
                    CpuMode::Compat32,
                    bus,
                    Some(crate::lift::FetchFailure::Memory(trap)),
                ),
                linear,
                bus,
            )
        })?;
        source.record(bus, &byte);
        if let Some((_, len)) = sources.last_mut().filter(|(start, len)| {
            *start + u64::from(*len) == bus
                && *start >> crate::memory::CODE_PAGE_BITS == bus >> crate::memory::CODE_PAGE_BITS
        }) {
            *len += 1;
        } else {
            sources.push((bus, 1));
        }
        bytes.push(byte[0]);
        if crate::model::decode::unavailable_prefix_for_generation(context.generation, &bytes) {
            break;
        }
        let mut decoder = Decoder::with_ip(
            u32::from(context.default_operand_bits),
            &bytes,
            ip,
            crate::model::decode::SYSTEM_DECODER_OPTIONS,
        );
        if !decoder.decode().is_invalid() || decoder.last_error() != DecoderError::NoMoreBytes {
            break;
        }
    }
    Ok((bytes, sources, pages))
}

impl Vcpu {
    #[cold]
    #[inline(never)]
    pub(in crate::vm) fn validate_paged_continuations(&self, vm: &Vm) -> Result<(), Exit> {
        let Some(state) = self
            .legacy_execution
            .as_ref()
            .and_then(|e| e.paging.as_ref())
        else {
            return Ok(());
        };
        // Refuse before dispatch cleanup can release an acknowledged effect's
        // ledger. A later retry must keep refusing the same stale physical plan.
        if state
            .transaction
            .iter()
            .chain(&state.event_transaction)
            .any(|tx| !tx.matches_memory(&vm.mem))
            || state
                .port_input
                .as_ref()
                .is_some_and(|p| !p.matches_owner(&self.cpu, &vm.mem))
        {
            return Err(Exit::UnsupportedSystemContext {
                addr: self.cpu.rip,
                operation: UnsupportedSystemOperation::LegacyEntryFacility,
            });
        }
        Ok(())
    }

    /// Original linear byte of the pending paged Machine MMIO fragment.
    /// The returned `Exit::MmioRead`/`MmioWrite` carries its physical bus address,
    /// width and value. This evidence survives a held retry and is cleared when
    /// the instruction completes or is abandoned. Other execution paths return None.
    pub fn pending_mmio_linear_address(&self) -> Option<u32> {
        if self.cpu.legacy.cr0 & 0x8000_0000 == 0 {
            return None;
        }
        let state = self.legacy_execution.as_ref()?.paging.as_ref()?;
        state
            .event_transaction
            .as_ref()
            .and_then(|tx| tx.pending_mmio_linear_address())
            .or_else(|| {
                state
                    .port_input
                    .as_ref()
                    .and_then(|p| p.pending_mmio_linear_address())
            })
            .or_else(|| {
                state
                    .transaction
                    .as_ref()
                    .and_then(|tx| tx.pending_mmio_linear_address())
            })
    }

    #[cold]
    #[inline(never)]
    pub(super) fn abandon_edited_paged_retry(&mut self) {
        let execution = self.legacy_execution.as_mut().unwrap();
        let Some(state) = execution.paging.as_mut() else {
            return;
        };
        let event_edited = state
            .event_transaction
            .as_ref()
            .is_some_and(|tx| !tx.matches_inputs(&self.cpu));
        let instruction_edited = state.transaction.as_ref().is_some_and(|tx| {
            !execution.instruction_retry.as_ref().is_some_and(|retry| {
                retry.stamp == LegacyContextStamp::new(&self.cpu, self.legacy_model.unwrap())
                    && retry.ip == self.cpu.rip
                    && retry.mxcsr_mode == self.cpu.mxcsr & crate::jit_abi::fp::MODE_MASK
                    && tx.matches_inputs(&self.cpu)
            })
        });
        if event_edited || instruction_edited {
            state.transaction = None;
            state.event_transaction = None;
            execution.instruction_retry = None;
            execution.task_switch.reset();
            if event_edited {
                self.abandon_legacy_event();
            }
            self.cpu.clear_mmio_parts();
        }
    }

    #[cold]
    #[inline(never)]
    pub(super) fn resolve_paged(
        &mut self,
        vm: &Vm,
        context: MachineContext,
    ) -> Result<(Arc<IrBlock>, crate::interp::legacy::BlockWalk), Exit> {
        let execution = self.legacy_execution.as_mut().unwrap();
        // A bus handoff owns its decoded bytes and translated spans until the
        // instruction finishes or entry abandons it (decision-16).
        if let Some(retry) = &execution.instruction_retry {
            if retry.stamp == LegacyContextStamp::new(&self.cpu, self.legacy_model.unwrap())
                && retry.ip == self.cpu.rip
                && retry.mxcsr_mode == self.cpu.mxcsr & crate::jit_abi::fp::MODE_MASK
                && execution
                    .paging
                    .as_ref()
                    .and_then(|s| s.transaction.as_ref())
                    .is_none_or(|tx| tx.matches_inputs(&self.cpu))
            {
                return Ok((retry.ir.clone(), retry.walk));
            }
            self.cpu.clear_mmio_parts();
            execution.instruction_retry = None;
        }
        let selection = execution.paging_selection();
        let state = paging_state(&mut execution.paging);
        state.transaction = None;
        state
            .resolve(
                vm,
                &mut self.cpu,
                self.legacy_model.unwrap(),
                context,
                execution.owner_id,
                &self.exit_request.word,
                true,
                Some(selection),
            )
            // Paging always uses a protected or V86 context. Reuse the
            // dispatcher's system-walk gate before selecting paged execution.
            .map(|(ir, ordinary)| {
                (
                    ir,
                    if ordinary {
                        crate::interp::legacy::BlockWalk::Paged
                    } else {
                        crate::interp::legacy::BlockWalk::Protected
                    },
                )
            })
            .inspect_err(|exit| publish_cr2(&mut self.cpu, exit))
    }

    #[cold]
    #[inline(never)]
    pub(super) fn execute_paged<const ROUTED: bool, const ORDER: u8>(
        &mut self,
        vm: &Vm,
        ir: &Arc<IrBlock>,
        context: MachineContext,
        limit: u64,
        ordinary: bool,
    ) -> crate::interp::legacy::Progress {
        let execution = self.legacy_execution.as_mut().unwrap();
        // A retained retry slice arrives as Protected; classify it here.
        if ordinary || ordinary_block(context, ir) {
            execution.rep_flags = None;
            let state = paging_state(&mut execution.paging);
            let version = state.blocks.version();
            let insns: &[Insn] = match state.lane_slots.get(lanes_slot(Arc::as_ptr(ir))) {
                Some(Some(slot))
                    if slot.ir == Arc::as_ptr(ir) as usize && slot.version == version =>
                {
                    &slot.insns
                }
                // A retained retry slice, a snapshot run once unpublished, or
                // a slot another block took.
                _ => {
                    crate::interp::legacy::paging::lanes_into(ir, &mut state.lane_scratch);
                    &state.lane_scratch
                }
            };
            let progress = crate::interp::legacy::paging::execute_flat32::<ROUTED, ORDER>(
                ir,
                insns,
                &mut self.cpu,
                &vm.mem,
                &mut self.interp_scratch,
                limit,
                &self.exit_request.word,
                &mut execution.task_switch,
                &mut state.transaction,
                &state.tlb,
            );
            // A walk that reloaded a data segment and then stopped with an
            // exit reports no context change (`BlockPass::reloaded`): the
            // next resolve compares the paging stamp again (xj-to9.59).
            if progress.outcome.as_ref().is_some_and(|o| o.is_err()) {
                execution.context_dirty();
            }
            return progress;
        }
        let comparing = ir.ops.iter().any(|op| {
            matches!(
                op,
                IrOp::Legacy(crate::legacy::LegacyOp::String {
                    kind: crate::ir::StrOp::Cmps | crate::ir::StrOp::Scas,
                    rep: crate::ir::RepKind::Repe | crate::ir::RepKind::Repne,
                    ..
                })
            )
        });
        if comparing {
            let continuing = execution.rep_flags.as_ref().is_some_and(|saved| {
                Arc::ptr_eq(&saved.ir, ir)
                    && saved.matches(&self.cpu, self.legacy_model.unwrap(), self.retired)
            });
            if !continuing {
                execution.rep_flags = Some(Box::new(RepFlags {
                    ir: ir.clone(),
                    stamp: LegacyContextStamp::new(&self.cpu, self.legacy_model.unwrap()),
                    cursor: RepFlags::cursor(&self.cpu),
                    flags: self.cpu.flags,
                    system_flags: self.cpu.rflags_system,
                    origin: self.cpu.flags,
                    retired: self.retired,
                }));
            }
        } else {
            execution.rep_flags = None;
        }
        let state = paging_state(&mut execution.paging);
        let v86 = context.regime == MachineRegime::Virtual8086;
        // Same-level far transfers of the ordinary classes skip the operand
        // transaction when their descriptor and frame are proved RAM.
        let fast = if v86 {
            crate::interp::legacy::paging::execute_far_fast::<ROUTED, ORDER, true>(
                ir,
                &mut self.cpu,
                &vm.mem,
                context,
                &mut execution.task_switch,
                &state.transaction,
                &state.tlb,
            )
        } else if matches!(
            context.segment_class,
            crate::SegmentClass::Flat32 | crate::SegmentClass::Segmented
        ) {
            crate::interp::legacy::paging::execute_far_fast::<ROUTED, ORDER, false>(
                ir,
                &mut self.cpu,
                &vm.mem,
                context,
                &mut execution.task_switch,
                &state.transaction,
                &state.tlb,
            )
        } else {
            None
        };
        let result = if let Some(result) = fast {
            result.map(|outcome| (outcome, None))
        } else if v86 {
            crate::interp::legacy::paging::execute_retained::<ROUTED, ORDER, true>(
                ir,
                &mut self.cpu,
                &vm.mem,
                context,
                &mut execution.task_switch,
                &mut state.transaction,
                &state.tlb,
            )
        } else {
            execute_retained::<ROUTED, ORDER>(
                ir,
                &mut self.cpu,
                &vm.mem,
                context,
                &mut execution.task_switch,
                &mut state.transaction,
                &state.tlb,
            )
        };
        let outcome = result.map(|(outcome, input)| {
            state.port_input = input;
            if outcome.translations_invalidated {
                state.load_cr3();
            } else if outcome.context_changed {
                // The dispatcher's `sync_class` follows.
                state.reconcile_translation(&self.cpu);
            }
            outcome
        });
        if !outcome
            .as_ref()
            .is_ok_and(|o| !o.context_changed && !o.translations_invalidated)
        {
            execution.context_dirty();
        }
        if let Some(saved) = execution.rep_flags.as_mut() {
            let fault = outcome.as_ref().is_err_and(|exit| {
                matches!(exit, Exit::UnmappedMemory { .. })
                    || exit
                        .fault_metadata()
                        .is_some_and(|m| m.class() == crate::fault::FaultClass::Fault)
            });
            if fault {
                self.cpu.flags = saved.origin;
                execution.rep_flags = None;
            } else if outcome.as_ref().is_ok_and(|o| o.retired) || self.cpu.rip != ir.guest_start {
                execution.rep_flags = None;
            } else {
                saved.cursor = RepFlags::cursor(&self.cpu);
                saved.flags = self.cpu.flags;
                saved.system_flags = self.cpu.rflags_system;
            }
        }
        let retired = u64::from(outcome.as_ref().is_ok_and(|o| o.retired));
        crate::interp::legacy::Progress {
            steps: u64::from(outcome.is_ok()),
            retired,
            sti_shadow: self.sti_shadow,
            ss_shadow: self.ss_shadow,
            outcome: Some(outcome),
        }
    }

    #[cold]
    #[inline(never)]
    pub(super) fn finish_paged_port_input<const ROUTED: bool, const ORDER: u8>(
        &mut self,
        vm: &Vm,
    ) -> Result<(), Exit> {
        let state = self
            .legacy_execution
            .as_mut()
            .unwrap()
            .paging
            .get_or_insert_with(Default::default);
        crate::interp::legacy::paging::finish_port_input::<ROUTED, ORDER>(
            &mut self.cpu,
            &vm.mem,
            &mut state.port_input,
        )
    }

    /// Internal single-instruction fetch/execute entry for focused witnesses.
    /// Uses the same paged adapter as normal Machine dispatch.
    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    #[cfg(feature = "test-hooks")]
    pub fn interpret_paged_for_test(&mut self, vm: &Vm) -> Result<(), Exit> {
        self.interpret_personality_for_test::<false>(vm)
    }

    /// Private V86 boundary for coherent images; public Machine entry stays gated.
    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    #[cfg(feature = "test-hooks")]
    pub fn interpret_virtual8086_for_test(&mut self, vm: &Vm) -> Result<(), Exit> {
        self.interpret_personality_for_test::<true>(vm)
    }

    #[cfg(feature = "test-hooks")]
    fn interpret_personality_for_test<const V86: bool>(&mut self, vm: &Vm) -> Result<(), Exit> {
        let model = vm
            .cpu_model
            .filter(|m| {
                matches!(
                    m,
                    crate::CpuModel::I80386
                        | crate::CpuModel::I80486Dx
                        | crate::CpuModel::I80486DxCpuid
                )
            })
            .ok_or(Exit::UnsupportedSystemContext {
                addr: self.cpu.rip,
                operation: UnsupportedSystemOperation::LegacyEntryFacility,
            })?;
        if self.legacy_execution.is_none() {
            return Err(Exit::UnsupportedSystemContext {
                addr: self.cpu.rip,
                operation: UnsupportedSystemOperation::LegacyEntryFacility,
            });
        }
        self.cpu.machine_flags_mask = model.machine_flags_mask();
        let context = if V86 {
            crate::legacy::derive_virtual8086_context_for_test(model.legacy_model(), &self.cpu)
        } else {
            derive_legacy_context(model.legacy_model(), &self.cpu)
        }
        .map_err(|error| Exit::UnsupportedSystemContext {
            addr: self.cpu.rip,
            operation: UnsupportedSystemOperation::InvalidLegacyContext(error),
        })?;
        // This test entry remains the singleton Checked reference tier.
        let context = MachineContext {
            segment_class: crate::SegmentClass::Checked,
            ..context
        };
        if self.cpu.port_string_in.is_some() {
            if !context.paging_enabled {
                return crate::interp::legacy::finish_port_input::<false, 0>(
                    &mut self.cpu,
                    &vm.mem,
                    u32::MAX,
                );
            }
            let state = self
                .legacy_execution
                .as_mut()
                .unwrap()
                .paging
                .get_or_insert_with(Default::default);
            return crate::interp::legacy::paging::finish_port_input::<false, 0>(
                &mut self.cpu,
                &vm.mem,
                &mut state.port_input,
            );
        }
        let execution = self.legacy_execution.as_mut().unwrap();
        let state = execution.paging.get_or_insert_with(Default::default);
        let result = state.resolve(
            vm,
            &mut self.cpu,
            model.legacy_model(),
            context,
            execution.owner_id,
            &self.exit_request.word,
            false,
            None,
        );
        let ir = match result {
            Ok((ir, _)) => ir,
            Err(exit) => {
                publish_cr2(&mut self.cpu, &exit);
                if let Some(saved) = execution.rep_flags.take() {
                    self.cpu.flags = saved.origin;
                }
                return self.route_paged_exception(vm, context, exit);
            }
        };
        let comparing = ir.ops.iter().any(|op| {
            matches!(
                op,
                IrOp::Legacy(crate::legacy::LegacyOp::String {
                    kind: crate::ir::StrOp::Cmps | crate::ir::StrOp::Scas,
                    rep: crate::ir::RepKind::Repe | crate::ir::RepKind::Repne,
                    ..
                })
            )
        });
        if comparing {
            let continuing = execution.rep_flags.as_ref().is_some_and(|saved| {
                Arc::ptr_eq(&saved.ir, &ir)
                    && saved.matches(&self.cpu, model.legacy_model(), self.retired)
            });
            if !continuing {
                execution.rep_flags = Some(Box::new(RepFlags {
                    ir: Arc::clone(&ir),
                    stamp: LegacyContextStamp::new(&self.cpu, model.legacy_model()),
                    cursor: RepFlags::cursor(&self.cpu),
                    flags: self.cpu.flags,
                    system_flags: self.cpu.rflags_system,
                    origin: self.cpu.flags,
                    retired: self.retired,
                }));
            }
        } else {
            execution.rep_flags = None;
        }
        let outcome = if context.paging_enabled {
            crate::interp::legacy::paging::execute_retained::<false, 0, V86>(
                &ir,
                &mut self.cpu,
                &vm.mem,
                context,
                &mut execution.task_switch,
                &mut state.transaction,
                &state.tlb,
            )
        } else {
            execute_for_test::<false, V86>(
                &ir,
                &mut self.cpu,
                &vm.mem,
                context,
                &mut execution.task_switch,
            )
        };
        if let Some(saved) = execution.rep_flags.as_mut() {
            let fault = outcome.as_ref().is_err_and(|exit| {
                matches!(exit, Exit::UnmappedMemory { .. })
                    || exit
                        .fault_metadata()
                        .is_some_and(|m| m.class() == crate::fault::FaultClass::Fault)
            });
            if fault {
                self.cpu.flags = saved.origin;
                execution.rep_flags = None;
            } else if outcome.as_ref().is_ok_and(|(o, _)| o.retired)
                || self.cpu.rip != ir.guest_start
            {
                execution.rep_flags = None;
            } else {
                saved.cursor = RepFlags::cursor(&self.cpu);
                saved.flags = self.cpu.flags;
                saved.system_flags = self.cpu.rflags_system;
            }
        }
        let (outcome, input) = match outcome {
            Ok(outcome) => outcome,
            Err(exit) => return self.route_paged_exception(vm, context, exit),
        };
        state.port_input = input;
        // 80386 PRM §5.2.5 p. 5-11; Intel486 PRM §5.3.5 p. 5-22:
        // loading CR3 flushes even when its value is unchanged.
        if outcome.translations_invalidated {
            state.load_cr3();
        } else if outcome.context_changed {
            state.reconcile(vm, &self.cpu, model.legacy_model(), None);
        }
        self.cpu.finish_mmio_retry(outcome.exit.as_ref());
        outcome.exit.map_or(Ok(()), Err)
    }

    #[cfg(feature = "test-hooks")]
    fn route_paged_exception(
        &mut self,
        vm: &Vm,
        mut context: MachineContext,
        exit: Exit,
    ) -> Result<(), Exit> {
        self.cpu.finish_mmio_retry(Some(&exit));
        if self.exception_delivery != crate::ExceptionDelivery::InGuest {
            return Err(exit);
        }
        let Some(source) = exception_source(&exit) else {
            return Err(exit);
        };
        if self
            .legacy_execution
            .as_ref()
            .unwrap()
            .task_switch
            .committed
        {
            context = crate::interp::legacy::tasks::fault_context(&self.cpu, context.generation);
        }
        let ip = exit
            .fault_metadata()
            .map_or(self.cpu.rip, |m| m.saved_rip()) as u32;
        self.deliver_legacy_event_impl::<false, 0, true>(
            vm,
            context,
            PendingEvent {
                irq_consumed: false,
                software: false,
                after_nmi: false,
                source,
                ip,
                cs: self.cpu.cs,
                origin: self.delivery_origin(self.cpu.cs, ip),
            },
        )
    }

    /// Internal proof diagnostics: translation generation, retained blocks, lifts.
    #[doc(hidden)]
    #[cfg(feature = "test-hooks")]
    pub fn paging_cache_for_test(&self) -> (u64, usize, u64) {
        self.legacy_execution
            .as_ref()
            .and_then(|e| e.paging.as_ref())
            .map_or((0, 0, 0), |s| (s.generation, s.blocks.len(), s.lifts))
    }

    /// Internal TLB contents: slot, page, frame, PTE, rights, fetched.
    #[doc(hidden)]
    #[cfg(feature = "test-hooks")]
    pub fn paging_tlb_for_test(&self) -> Vec<(usize, u32, u32, u32, u32, bool)> {
        self.legacy_execution
            .as_ref()
            .and_then(|e| e.paging.as_ref())
            .map_or_else(Vec::new, |s| s.tlb.borrow().entries_for_test())
    }

    /// Inject a physical write after fetch and before source registration.
    #[doc(hidden)]
    #[cfg(feature = "test-hooks")]
    pub fn paging_publication_hook_for_test(&mut self, hook: fn(&Vm)) {
        self.legacy_execution
            .as_mut()
            .unwrap()
            .paging
            .get_or_insert_with(Default::default)
            .before_publication = Some(hook);
    }
}

#[cold]
#[inline(never)]
fn execute_retained<const ROUTED: bool, const ORDER: u8>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    memory: &Memory,
    context: MachineContext,
    task_switch: &mut crate::interp::legacy::tasks::SwitchState,
    retry: &mut Option<crate::interp::legacy::paging::Transaction>,
    tlb: &std::cell::RefCell<crate::interp::legacy::paging::Tlb>,
) -> Result<
    (
        crate::interp::legacy::Outcome,
        Option<crate::interp::legacy::paging::PortPlan>,
    ),
    Exit,
> {
    crate::interp::legacy::paging::execute_retained::<ROUTED, ORDER, false>(
        ir,
        cpu,
        memory,
        context,
        task_switch,
        retry,
        tlb,
    )
}

#[cfg(test)]
impl Vcpu {
    /// Paged resolves that compared no stamp (xj-to9.59).
    pub(in crate::vm) fn paging_skips_for_test(&self) -> u64 {
        self.legacy_execution
            .as_ref()
            .and_then(|e| e.paging.as_ref())
            .map_or(0, |s| s.skips)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // xj-to9.59: a paged USE16 ring-0 loop whose every iteration writes each
    // class of stamped field from inside a block walk: data-segment loads
    // with two descriptors, LIDT and LGDT with two tables each, INT n and
    // IRET through a 286 interrupt gate, LLDT with two LDTs, a far CALL into
    // a second code segment and RETF, MOV CR0 toggling TS, POPF toggling
    // IOPL, and a #DE delivered between walks; an STI window takes a queued
    // IRQ between walks. LTR runs once. Every following resolve finds the
    // selection serial advanced or the stamp unchanged (the reconcile
    // assertion): a writer that left the serial still fails it.
    fn paged_writers(model: crate::CpuModel) -> (Vm, Vcpu) {
        let mut vm = Vm::new(crate::VmConfig::flat(0x20000));
        vm.set_cpu_model(model);
        vm.set_exception_delivery(crate::ExceptionDelivery::InGuest);
        vm.map(0, 0x20000, crate::Prot::RWX, crate::RegionKind::Ram)
            .unwrap();
        vm.write_bytes(0x1000, &0x2007u32.to_le_bytes()).unwrap();
        for page in 0..32u32 {
            vm.write_bytes(
                0x2000 + u64::from(page * 4),
                &((page << 12) | 7).to_le_bytes(),
            )
            .unwrap();
        }
        let descriptor = |base: u32, limit: u16, access: u8| -> [u8; 8] {
            let [l0, l1] = limit.to_le_bytes();
            let [b0, b1, b2, b3] = base.to_le_bytes();
            [l0, l1, b0, b1, b2, access, 0, b3]
        };
        let gdt = [
            [0; 8],
            descriptor(0, 0xffff, 0x9b),      // 08h: code
            descriptor(0x6000, 0xffff, 0x93), // 10h: data A
            descriptor(0x7000, 0xffff, 0x93), // 18h: data B
            descriptor(0, 0xffff, 0x9b),      // 20h: second code segment
            descriptor(0x5400, 7, 0x82),      // 28h: LDT A
            descriptor(0x5408, 7, 0x82),      // 30h: LDT B
            descriptor(0, 0xffff, 0x93),      // 38h: stack
            descriptor(0x5600, 0x2b, 0x81),   // 40h: 286 TSS
        ]
        .concat();
        vm.write_bytes(0x5000, &gdt).unwrap();
        vm.write_bytes(0x5200, &gdt).unwrap();
        // 286 interrupt gates to 20h: #DE at 4820h, 40h at 4810h, 41h at 4830h.
        for idt in [0x5800u64, 0x5c00] {
            for (vector, ip) in [(0u64, 0x4820u16), (0x40, 0x4810), (0x41, 0x4830)] {
                let [i0, i1] = ip.to_le_bytes();
                vm.write_bytes(idt + vector * 8, &[i0, i1, 0x20, 0, 0, 0x86, 0, 0])
                    .unwrap();
            }
        }
        // LIDT and LGDT operands, DS-relative to data A.
        vm.write_bytes(
            0x6100,
            &[
                0xff, 3, 0, 0x58, 0, 0, 0xff, 3, 0, 0x5c, 0, 0, 0x47, 0, 0, 0x50, 0, 0, 0x47, 0, 0,
                0x52, 0, 0,
            ],
        )
        .unwrap();
        #[rustfmt::skip]
        let code: &[u8] = &[
            0xb8, 0x40, 0x00,             // 4000: mov ax,40h
            0x0f, 0x00, 0xd8,             //       ltr ax
            0xb8, 0x10, 0x00,             // 4006: mov ax,10h
            0x8e, 0xd8,                   //       mov ds,ax
            0xff, 0x06, 0x10, 0x00,       //       inc word [10h]
            0x0f, 0x01, 0x1e, 0x00, 0x01, //       lidt [100h]
            0x0f, 0x01, 0x16, 0x0c, 0x01, //       lgdt [10ch]
            0xcd, 0x40,                   //       int 40h
            0x0f, 0x01, 0x1e, 0x06, 0x01, //       lidt [106h]
            0x0f, 0x01, 0x16, 0x12, 0x01, //       lgdt [112h]
            0xb8, 0x28, 0x00,             //       mov ax,28h
            0x0f, 0x00, 0xd0,             //       lldt ax
            0xb8, 0x18, 0x00,             //       mov ax,18h
            0x8e, 0xd8,                   //       mov ds,ax
            0xff, 0x06, 0x10, 0x00,       //       inc word [10h]
            0x9a, 0x00, 0x48, 0x20, 0x00, //       call 20h:4800h
            0xb8, 0x30, 0x00,             //       mov ax,30h
            0x0f, 0x00, 0xd0,             //       lldt ax
            0x0f, 0x20, 0xc0,             //       mov eax,cr0
            0x66, 0x35, 0x08, 0, 0, 0,    //       xor eax,8 (TS)
            0x0f, 0x22, 0xc0,             //       mov cr0,eax
            0x9c, 0x58,                   //       pushf; pop ax
            0x35, 0x00, 0x10,             //       xor ax,1000h (IOPL)
            0x50, 0x9d,                   //       push ax; popf
            0xb3, 0x00,                   //       mov bl,0
            0xb8, 0x07, 0x00,             //       mov ax,7
            0xf6, 0xf3,                   //       div bl (#DE once)
            0xfb, 0x90, 0xfa,             //       sti; nop; cli
            0x49,                         //       dec cx
            0x75, 0xa7,                   //       jnz 4006h
            0xf4,                         //       hlt
        ];
        vm.write_bytes(0x4000, code).unwrap();
        vm.write_bytes(0x4800, &[0x46, 0xcb]).unwrap(); // inc si; retf
        vm.write_bytes(0x4810, &[0x47, 0xcf]).unwrap(); // inc di; iret
        vm.write_bytes(0x4820, &[0x45, 0xb3, 0x01, 0xcf]).unwrap(); // inc bp; mov bl,1; iret
        vm.write_bytes(0x4830, &[0x42, 0xcf]).unwrap(); // inc dx; iret
        let mut cpu = vm.new_vcpu();
        let cache = |base: u32, attributes: u16| crate::SegmentCache {
            base,
            limit: 0xffff,
            attributes,
            ..Default::default()
        };
        let l = &mut cpu.cpu.legacy;
        l.cr0 = 0x8000_0001;
        l.cr3 = 0x1000;
        l.cpl = 0;
        l.segments = [
            cache(0, 0x9b),
            cache(0x6000, 0x93),
            cache(0x6000, 0x93),
            cache(0, 0x93),
            crate::SegmentCache {
                unusable: 1,
                ..Default::default()
            },
            crate::SegmentCache {
                unusable: 1,
                ..Default::default()
            },
        ];
        l.gdtr = crate::legacy::TableRegister {
            base: 0x5000,
            limit: 0x47,
            reserved: 0,
        };
        l.idtr = crate::legacy::TableRegister {
            base: 0x5800,
            limit: 0x3ff,
            reserved: 0,
        };
        [
            cpu.cpu.cs, cpu.cpu.ds, cpu.cpu.es, cpu.cpu.ss, cpu.cpu.fs, cpu.cpu.gs,
        ] = [8, 0x10, 0x10, 0x38, 0, 0];
        cpu.cpu.rip = 0x4000;
        cpu.cpu.gpr[4] = 0x9000;
        cpu.cpu.flags.if_ = false;
        cpu.set_a20_enabled(true);
        (vm, cpu)
    }

    // The counters the loop keeps: data A and B, SI (far calls), DI (INT
    // 40h), BP (#DE), DX (IRQ 41h), CX (iterations left).
    fn writer_counts(vm: &Vm, cpu: &Vcpu) -> [u64; 6] {
        let word = |at| {
            let mut w = [0; 2];
            vm.read_bytes(at, &mut w).unwrap();
            u64::from(u16::from_le_bytes(w))
        };
        [
            word(0x6010),
            word(0x7010),
            cpu.reg(Reg::Rsi) & 0xffff,
            cpu.reg(Reg::Rdi) & 0xffff,
            cpu.reg(Reg::Rbp) & 0xffff,
            cpu.reg(Reg::Rdx) & 0xffff,
        ]
    }

    fn paging_skips(cpu: &Vcpu) -> u64 {
        cpu.paging_skips_for_test()
    }

    #[test]
    fn every_writer_of_a_stamped_field_advances_the_selection() {
        for model in [crate::CpuModel::I80386, crate::CpuModel::I80486Dx] {
            let (vm, mut cpu) = paged_writers(model);
            cpu.set_reg(Reg::Rcx, 40);
            let options = crate::StepOptions::default();
            // A run takes the IRQ queued before it in its first STI window.
            let mut runs = 0;
            loop {
                cpu.inject_irq(0x41);
                runs += 1;
                let run = cpu.run_steps(&vm, 400, options).unwrap();
                if matches!(run.stop(), crate::StepStop::Exit(Exit::Hlt)) {
                    break;
                }
                assert!(runs < 100, "{model:?}: {:?}", run.stop());
            }
            assert_eq!(cpu.reg(Reg::Rcx) & 0xffff, 0, "{model:?}");
            let counts = writer_counts(&vm, &cpu);
            assert_eq!(counts[..5], [40; 5], "{model:?}");
            assert!((1..=runs).contains(&counts[5]), "{model:?}: {counts:?}");
            // An even count of toggles leaves TS and IOPL as they were.
            assert_eq!(cpu.cpu.legacy.cr0 & 8, 0);
            assert_eq!(cpu.cpu.rflags_system & 0x3000, 0);
            assert!(
                paging_skips(&cpu) > 40 * 5,
                "{model:?}: {}",
                paging_skips(&cpu)
            );
        }
    }

    // Host edits between runs: another table, IOPL, a restored image.
    #[test]
    fn a_host_edit_between_runs_advances_the_selection() {
        let (vm, mut cpu) = paged_writers(crate::CpuModel::I80386);
        cpu.set_reg(Reg::Rcx, 1000);
        let options = crate::StepOptions::default();
        assert_eq!(cpu.run_steps(&vm, 300, options).unwrap().steps(), 300);
        let saved = cpu.cpu.clone();
        for edit in 0..12 {
            match edit % 4 {
                0 => cpu.cpu.legacy.idtr.base ^= 0x5800 ^ 0x5c00,
                1 => cpu.cpu.legacy.gdtr.base ^= 0x5000 ^ 0x5200,
                2 => cpu.cpu.rflags_system ^= 0x1000,
                _ => cpu.cpu = saved.clone(),
            }
            assert_eq!(cpu.run_steps(&vm, 97, options).unwrap().steps(), 97);
        }
        assert!(paging_skips(&cpu) > 0);
    }

    #[test]
    fn sliced_far_calls_retain_blocks_and_translations_with_distant_data() {
        for model in [crate::CpuModel::I80386, crate::CpuModel::I80486Dx] {
            let mut vm = Vm::new(crate::VmConfig::flat(0x80000));
            vm.set_cpu_model(model);
            vm.map(0, 0x80000, crate::Prot::RWX, crate::RegionKind::Ram)
                .unwrap();
            vm.write_bytes(0x1000, &0x2007u32.to_le_bytes()).unwrap();
            for page in 0..128u32 {
                vm.write_bytes(
                    0x2000 + u64::from(page * 4),
                    &((page * 4096) | 7).to_le_bytes(),
                )
                .unwrap();
            }
            // Descriptor page 0 and data page 64 must coexist in the TLB.
            for index in 1..=4 {
                let descriptor = if index <= 2 {
                    0x00cf_fb00_0000_ffffu64
                } else {
                    0x00cf_f300_0000_ffffu64
                };
                vm.write_bytes(0x800 + index * 8, &descriptor.to_le_bytes())
                    .unwrap();
            }
            let mut code = vec![0xa1, 0, 0, 4, 0]; // MOV EAX,[0x40000].
            code.extend([0x90; 8]);
            code.extend([0x9a, 0, 0x10, 1, 0, 0x13, 0]); // CALL FAR 13:11000.
            code.extend([0xeb, 0xea]); // JMP 10000.
            vm.write_bytes(0x10000, &code).unwrap();
            vm.write_bytes(0x11000, &[0x90, 0xcb]).unwrap();
            vm.write_bytes(0x40000, &0x12345678u32.to_le_bytes())
                .unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu.legacy.cr0 = 0x80000001;
            cpu.cpu.legacy.cr3 = 0x1000;
            cpu.cpu.legacy.cpl = 3;
            cpu.cpu.legacy.gdtr.base = 0x800;
            cpu.cpu.legacy.gdtr.limit = 39;
            cpu.cpu.cs = 11;
            cpu.cpu.ds = 27;
            cpu.cpu.es = 27;
            cpu.cpu.ss = 35;
            for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
                cache.limit = u32::MAX;
                cache.attributes = if i == 0 { 0xc0fb } else { 0xc0f3 };
            }
            cpu.set_a20_enabled(true);
            cpu.set_reg(Reg::Rip, 0x10000);
            cpu.set_reg(Reg::Rsp, 0x8800);
            let slice = |cpu: &mut Vcpu, budget| {
                cpu.disable_delivery_log();
                cpu.set_a20_enabled(true);
                for reg in [Reg::Rip, Reg::Rax, Reg::Rsp] {
                    cpu.set_reg(reg, cpu.reg(reg));
                }
                let run = cpu
                    .run_steps(&vm, budget, crate::StepOptions::default())
                    .unwrap();
                assert_eq!(run.steps(), budget);
                assert!(matches!(
                    run.into_stop(),
                    crate::StepStop::Exit(Exit::BudgetExhausted)
                ));
            };
            for budget in [1, 17, 64, 3].into_iter().cycle().take(104) {
                slice(&mut cpu, budget);
            }
            let snapshot = |cpu: &Vcpu| {
                let state = cpu
                    .legacy_execution
                    .as_ref()
                    .unwrap()
                    .paging
                    .as_ref()
                    .unwrap();
                (state.generation, state.lifts)
            };
            let warm = snapshot(&cpu);
            let retired = cpu.retired_instructions();
            for budget in [1, 17, 64, 3].into_iter().cycle().take(104) {
                slice(&mut cpu, budget);
                assert_eq!(snapshot(&cpu), warm, "{model:?}: warm slice rebuilt proofs");
            }
            assert_eq!(cpu.retired_instructions() - retired, 2210);
            assert_eq!(cpu.reg(Reg::Rax), 0x12345678);
            assert_eq!(cpu.reg(Reg::Rsp), 0x8800);
            assert_eq!(cpu.reg(Reg::Rip), 0x10000);
        }
    }

    /// Paged V86 at IOPL 0 with a ring-0 #GP monitor (SDM Vol 3B §23.2.8.1
    /// p. 23-11, §23.3.1.1 pp. 23-12–13; Vol 2A IRET pp. 3-492–493). The
    /// V86 stack page 47h and the ring-0 stack page 146h (frame 6) share a
    /// TLB set (`Tlb::index`).
    fn v86_monitor(model: crate::CpuModel, v86: &[u8], paged: bool) -> (Vm, Vcpu) {
        let mut vm = Vm::new(crate::VmConfig::flat(0x60000));
        vm.set_cpu_model(model);
        vm.map(0, 0x60000, crate::Prot::RWX, crate::RegionKind::Ram)
            .unwrap();
        if paged {
            vm.write_bytes(0x1000, &0x2007u32.to_le_bytes()).unwrap();
            for page in 0..0x60u32 {
                let user = matches!(page, 0x10 | 0x30 | 0x47 | 0x51);
                let entry = (page << 12) | if user { 7 } else { 3 };
                vm.write_bytes(0x2000 + u64::from(page * 4), &entry.to_le_bytes())
                    .unwrap();
            }
            vm.write_bytes(0x2000 + 0x146 * 4, &0x6003u32.to_le_bytes())
                .unwrap();
        }
        vm.write_bytes(0x3808, &0x00cf_9b00_0000_ffffu64.to_le_bytes())
            .unwrap();
        vm.write_bytes(0x3810, &0x00cf_9300_0000_ffffu64.to_le_bytes())
            .unwrap();
        // Vector 13: 32-bit interrupt gate, DPL 0, to 8:9000h.
        vm.write_bytes(0x3000 + 13 * 8, &0x00008e00_00089000u64.to_le_bytes())
            .unwrap();
        // ESP0 = 147000h paged (7000h unpaged), SS0 = 10h.
        let esp0: u32 = if paged { 0x147000 } else { 0x7000 };
        vm.write_bytes(0x5004, &esp0.to_le_bytes()).unwrap();
        vm.write_bytes(0x5008, &0x10u16.to_le_bytes()).unwrap();
        vm.write_bytes(0x5066, &0x68u16.to_le_bytes()).unwrap();
        // PUSH EAX; MOV AX,10h; MOV DS,AX; INC DWORD [6000h];
        // ADD DWORD [ESP+8],2; POP EAX; ADD ESP,4; IRETD.
        vm.write_bytes(
            0x9000,
            &[
                0x50, 0x66, 0xb8, 0x10, 0x00, 0x8e, 0xd8, 0xff, 0x05, 0x00, 0x60, 0x00, 0x00, 0x83,
                0x44, 0x24, 0x08, 0x02, 0x58, 0x83, 0xc4, 0x04, 0xcf,
            ],
        )
        .unwrap();
        vm.write_bytes(0x10000, v86).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu = CpuState::legacy_real_entry([0x1000, 0x5100, 0x5100, 0x4000, 0, 0]);
        cpu.cpu.legacy.cr0 = if paged { 0x80000001 } else { 1 };
        cpu.cpu.legacy.cr3 = 0x1000;
        cpu.cpu.legacy.cpl = 3;
        cpu.cpu.rflags_system = 0x20000;
        cpu.cpu.legacy.gdtr.base = 0x3800;
        cpu.cpu.legacy.gdtr.limit = 0x17;
        cpu.cpu.legacy.idtr.base = 0x3000;
        cpu.cpu.legacy.idtr.limit = 0x7ff;
        cpu.cpu.legacy.tr.selector = 0x18;
        cpu.cpu.legacy.tr.cache = crate::SegmentCache {
            base: 0x5000,
            limit: 0x67,
            attributes: 0x8b,
            ..Default::default()
        };
        cpu.cpu.gpr[4] = 0x8000;
        cpu.set_a20_enabled(true);
        (vm, cpu)
    }

    fn proof_snapshot(cpu: &Vcpu) -> (u64, u64) {
        let state = cpu
            .legacy_execution
            .as_ref()
            .unwrap()
            .paging
            .as_ref()
            .unwrap();
        (state.generation, state.lifts)
    }

    #[test]
    fn v86_monitor_round_trips_keep_both_contexts_proofs() {
        // IN AL,80h (#GP to the monitor); PUSH AX; POP AX; JMP to the IN.
        let v86 = [0xe4, 0x80, 0x50, 0x58, 0xeb, 0xfa];
        for model in [crate::CpuModel::I80386, crate::CpuModel::I80486Dx] {
            let (vm, mut cpu) = v86_monitor(model, &v86, true);
            let options = crate::StepOptions::default();
            // One period: three V86 steps and the monitor's eight.
            assert_eq!(cpu.run_steps(&vm, 22, options).unwrap().steps(), 22);
            // The monitor (null DS on entry from V86) is Segmented, with
            // blocks that start where a slice starts: one sliced period
            // lifts them, and every later period reuses them.
            let period = |cpu: &mut Vcpu| {
                for budget in [11, 5, 6] {
                    cpu.disable_delivery_log();
                    assert_eq!(cpu.run_steps(&vm, budget, options).unwrap().steps(), budget);
                }
            };
            period(&mut cpu);
            let warm = proof_snapshot(&cpu);
            let retired = cpu.retired_instructions();
            for _ in 0..50 {
                period(&mut cpu);
                assert_eq!(
                    proof_snapshot(&cpu),
                    warm,
                    "{model:?}: round trip rebuilt proofs"
                );
            }
            assert_eq!(cpu.retired_instructions() - retired, 50 * 22);
            assert_eq!(cpu.cpu.rflags_system & 0x20000, 0x20000);
            assert_eq!(
                (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
                (0x1000, 0, 0x8000)
            );
            let mut count = [0; 4];
            vm.read_bytes(0x6000, &mut count).unwrap();
            assert_eq!(u32::from_le_bytes(count), 104);
        }
    }

    #[test]
    fn replacing_a_fetch_translation_still_expires_its_proofs() {
        // SDM Vol 3A §5.10.2: an edited PTE may stay unobserved until its
        // translation leaves the TLB. Data page 10Ch (FFFFh:C010h, A20 on)
        // shares code page Eh's set, so MOV AL,[C010h] replaces the fetch
        // entry each iteration and the next fetch must walk to the edited
        // frame. Code page Eh maps frame 10h, then 30h; page 10Ch frame 51h.
        // MOV AL,[C010h]; INC CX (DEC CX at frame 30h); JMP to the MOV.
        use crate::interp::legacy::paging::Tlb;
        assert_eq!(Tlb::index(0xe000), Tlb::index(0x10c000));
        for model in [crate::CpuModel::I80386, crate::CpuModel::I80486Dx] {
            let (vm, mut cpu) = v86_monitor(model, &[0xa0, 0x10, 0xc0, 0x41, 0xeb, 0xfa], true);
            vm.write_bytes(0x30000, &[0xa0, 0x10, 0xc0, 0x49, 0xeb, 0xfa])
                .unwrap();
            vm.write_bytes(0x2000 + 0xe * 4, &0x10007u32.to_le_bytes())
                .unwrap();
            vm.write_bytes(0x2000 + 0x10c * 4, &0x51007u32.to_le_bytes())
                .unwrap();
            for (segment, selector) in [(SegmentId::Cs, 0xe00u16), (SegmentId::Ds, 0xffff)] {
                cpu.cpu.legacy.segments[segment as usize].base = u32::from(selector) << 4;
            }
            (cpu.cpu.cs, cpu.cpu.ds) = (0xe00, 0xffff);
            let options = crate::StepOptions::default();
            assert_eq!(cpu.run_steps(&vm, 30, options).unwrap().steps(), 30);
            assert_eq!(cpu.cpu.gpr[1], 10);
            vm.write_bytes(0x2000 + 0xe * 4, &0x30007u32.to_le_bytes())
                .unwrap();
            assert_eq!(cpu.run_steps(&vm, 30, options).unwrap().steps(), 30);
            assert_eq!(
                cpu.cpu.gpr[1], 0,
                "{model:?}: stale proof after replacement"
            );
        }
    }

    #[test]
    fn unrolled_v86_slices_fit_the_retained_block_map() {
        let mut vm = Vm::new(crate::VmConfig::flat(0x8000));
        vm.set_cpu_model(crate::CpuModel::I80386);
        vm.map(0, 0x8000, crate::Prot::RWX, crate::RegionKind::Ram)
            .unwrap();
        vm.write_bytes(0x1000, &0x2007u32.to_le_bytes()).unwrap();
        vm.write_bytes(0x2010, &0x4007u32.to_le_bytes()).unwrap();
        let mut code = vec![0x90; 200];
        code.extend([0xe9, 0x35, 0xff]); // JMP back across 203 bytes.
        vm.write_bytes(0x4000, &code).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu = CpuState::legacy_real_entry([0x400, 0, 0, 0, 0, 0]);
        cpu.cpu.legacy.cr0 = 0x80000001;
        cpu.cpu.legacy.cr3 = 0x1000;
        cpu.cpu.legacy.cpl = 3;
        cpu.cpu.rflags_system = 0x23000;
        let options = crate::StepOptions::default();
        assert_eq!(cpu.run_steps(&vm, 402, options).unwrap().steps(), 402);
        let snapshot = |cpu: &Vcpu| {
            let state = cpu
                .legacy_execution
                .as_ref()
                .unwrap()
                .paging
                .as_ref()
                .unwrap();
            (state.generation, state.lifts, state.blocks.len())
        };
        // Ordinary V86 blocks start wherever a 64-step slice begins: after
        // one pass over every start (64 is prime to 201) the working set,
        // at most one block per instruction, stays retained.
        let slice = |cpu: &mut Vcpu| {
            cpu.disable_delivery_log();
            cpu.set_a20_enabled(true);
            assert_eq!(cpu.run_steps(&vm, 64, options).unwrap().steps(), 64);
        };
        for _ in 0..201 {
            slice(&mut cpu);
        }
        let warm = snapshot(&cpu);
        assert_eq!(warm.1, warm.2 as u64);
        assert!(warm.2 <= 201, "{warm:?}");
        for _ in 0..201 {
            slice(&mut cpu);
            assert_eq!(snapshot(&cpu), warm);
        }
        assert_eq!(cpu.cpu.rip, 0);
    }

    #[test]
    fn paged_flat32_warm_block_retains_translation_and_physical_registration() {
        for model in [crate::CpuModel::I80386, crate::CpuModel::I80486Dx] {
            let mut vm = Vm::new(crate::VmConfig::flat(0x8000));
            vm.set_cpu_model(model);
            vm.map(0, 0x8000, crate::Prot::RWX, crate::RegionKind::Ram)
                .unwrap();
            // Linear code page 1 maps physical frame 4.
            vm.write_bytes(0x2000, &0x3007u32.to_le_bytes()).unwrap();
            vm.write_bytes(0x3004, &0x4007u32.to_le_bytes()).unwrap();
            vm.write_bytes(0x4000, b"\x40\x43\x42\xeb\x00").unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu.rip = 0x1000;
            cpu.cpu.legacy.cr0 = 0x80000001;
            cpu.cpu.legacy.cr3 = 0x2000;
            for (i, cache) in cpu.cpu.legacy.segments[..4].iter_mut().enumerate() {
                cache.base = 0;
                cache.limit = u32::MAX;
                cache.attributes = if i == 0 { 0x409b } else { 0x4093 };
            }
            let context = derive_legacy_context(model.legacy_model(), &cpu.cpu).unwrap();
            assert_eq!(context.segment_class, crate::SegmentClass::Flat32);
            let mut state = State::default();
            let exit = AtomicU64::new(0);
            *vm.source_read_hook.lock().unwrap() = Some(|vm| {
                vm.write_bytes(0x4001, b"\x90").unwrap();
            });
            let a = state
                .resolve(
                    &vm,
                    &mut cpu.cpu,
                    model.legacy_model(),
                    context,
                    1,
                    &exit,
                    true,
                    None,
                )
                .unwrap()
                .0;
            assert_eq!(a.icount, 4);
            assert!(!a.ops.iter().any(|op| matches!(
                op,
                IrOp::WriteReg {
                    reg: crate::Reg::Rbx,
                    ..
                }
            )));
            assert!(vm.mem.is_code_page(4));
            let token = state.generation;
            let lifts = state.lifts;
            let registration = &state.blocks.values().next().unwrap()._registration as *const _;
            // SDM Vol 3A §5.8 / paging design §3.2: A may stay clear on a
            // retained translation. Clearing it also witnesses no warm walk.
            vm.write_bytes(0x3004, &0x4007u32.to_le_bytes()).unwrap();
            let b = state
                .resolve(
                    &vm,
                    &mut cpu.cpu,
                    model.legacy_model(),
                    context,
                    1,
                    &exit,
                    true,
                    None,
                )
                .unwrap()
                .0;
            assert!(Arc::ptr_eq(&a, &b));
            assert_eq!(state.generation, token);
            assert_eq!(state.lifts, lifts);
            assert_eq!(
                &state.blocks.values().next().unwrap()._registration as *const _,
                registration
            );
            let mut pte = [0; 4];
            vm.read_bytes(0x3004, &mut pte).unwrap();
            assert_eq!(u32::from_le_bytes(pte), 0x4007);
            vm.write_bytes(0x4001, b"\x43").unwrap();
            let c = state
                .resolve(
                    &vm,
                    &mut cpu.cpu,
                    model.legacy_model(),
                    context,
                    1,
                    &exit,
                    true,
                    None,
                )
                .unwrap()
                .0;
            assert!(!Arc::ptr_eq(&a, &c));
        }
    }

    #[test]
    fn a_paged_code_write_drops_only_the_proofs_whose_frame_bytes_it_overlaps() {
        for model in [crate::CpuModel::I80386, crate::CpuModel::I80486Dx] {
            let mut vm = Vm::new(crate::VmConfig::flat(0x8000));
            vm.set_cpu_model(model);
            vm.map(0, 0x8000, crate::Prot::RWX, crate::RegionKind::Ram)
                .unwrap();
            // Linear code page 1 maps physical frame 4. Block A: INC EAX;
            // JMP $+2. B: INC EBX; JMP $+2. Both in the one frame.
            vm.write_bytes(0x2000, &0x3007u32.to_le_bytes()).unwrap();
            vm.write_bytes(0x3004, &0x4007u32.to_le_bytes()).unwrap();
            vm.write_bytes(0x4000, b"\x40\xeb\x00\x43\xeb\x00").unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu.legacy.cr0 = 0x80000001;
            cpu.cpu.legacy.cr3 = 0x2000;
            for (i, cache) in cpu.cpu.legacy.segments[..4].iter_mut().enumerate() {
                cache.base = 0;
                cache.limit = u32::MAX;
                cache.attributes = if i == 0 { 0x409b } else { 0x4093 };
            }
            let context = derive_legacy_context(model.legacy_model(), &cpu.cpu).unwrap();
            let mut state = State::default();
            let exit = AtomicU64::new(0);
            let mut at = |state: &mut State, ip| {
                cpu.cpu.rip = ip;
                state
                    .resolve(
                        &vm,
                        &mut cpu.cpu,
                        model.legacy_model(),
                        context,
                        1,
                        &exit,
                        true,
                        None,
                    )
                    .unwrap()
                    .0
            };
            let (a, b) = (at(&mut state, 0x1000), at(&mut state, 0x1003));
            assert_eq!(state.lifts, 2);
            // INC EBX becomes INC EDX in the frame: only B's proof overlaps.
            vm.write_bytes(0x4003, b"\x42").unwrap();
            assert!(Arc::ptr_eq(&a, &at(&mut state, 0x1000)));
            assert_eq!(state.lifts, 2, "{model:?}");
            let b2 = at(&mut state, 0x1003);
            assert!(!Arc::ptr_eq(&b, &b2));
            assert!(b2.ops.iter().any(|op| matches!(
                op,
                IrOp::WriteReg {
                    reg: crate::Reg::Rdx,
                    ..
                }
            )));
            assert_eq!(state.lifts, 3);
            // An invalidation outside the drain is unrecorded: every proof drops.
            vm.cache
                .invalidate_overlapping(0, u64::MAX, || vm.mem.clear_all_code_pages());
            assert!(!Arc::ptr_eq(&a, &at(&mut state, 0x1000)));
            assert!(!Arc::ptr_eq(&b2, &at(&mut state, 0x1003)));
            assert_eq!(state.lifts, 5);
        }
    }
}
