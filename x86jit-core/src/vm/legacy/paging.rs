//! Internal paging fetch/proofs (paging design §§3.2, 3.5;
//! spec.md §17.10). No shared dispatch entry or compiled materialization.
use super::*;
#[cfg(feature = "test-hooks")]
use crate::interp::legacy::paging::execute_for_test;
use crate::interp::legacy::paging::{fetch_bus_for_test, publish_cr2};
use std::cell::Cell;

// Checked V86 loops retain one proof per instruction, including multiple
// unrolled bodies. Keep their working set independently of context identities.
const MAX_BLOCKS: usize = 256;
const MAX_CONTEXTS: usize = 64;

#[derive(PartialEq, Eq)]
struct Key {
    segment_class: crate::SegmentClass,
    // The singleton reference entry (`interpret_*_for_test`) never shares
    // an ordinary multi-instruction block.
    ordinary: bool,
    decode: LegacyBlockKey,
    generation: u64,
    cr3: u32,
    cr0: u32,
}
impl Hash for Key {
    // Every field, packed losslessly: one `BlockHasher` round per word.
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.decode.hash(state);
        state.write_u64(self.generation);
        state.write_u64(u64::from(self.cr3) | (u64::from(self.cr0) << 32));
        state.write_u64(self.segment_class as u64 | (u64::from(self.ordinary) << 8));
    }
}

struct Proof {
    ir: Arc<IrBlock>,
    // Computed once at lift: whether the block takes the ordinary walk.
    ordinary: bool,
    permit: Option<crate::FetchPermit>,
    // Ordered post-A20 bus spans, never a linear length applied to one frame.
    _registration: crate::memory::CodeRegistration,
}

#[derive(Default)]
pub(super) struct State {
    generation: u64,
    pub(super) tlb: std::cell::RefCell<crate::interp::legacy::paging::Tlb>,
    mapping: Option<Arc<()>>,
    stamp: Option<LegacyContextStamp>,
    pub(super) context_id: u64,
    contexts: EngineMap<LegacyContextStamp, u64>,
    model: Option<LegacyCpuModel>,
    memory: usize,
    source_generation: u64,
    blocks: EngineMap<Key, Proof>,
    lifts: u64,
    before_publication: Option<fn(&Vm)>,
    port_input: Option<crate::interp::legacy::paging::PortPlan>,
    pub(super) transaction: Option<crate::interp::legacy::paging::Transaction>,
    pub(super) event_transaction: Option<crate::interp::legacy::paging::Transaction>,
    // Most recent first; every entry is a member of `contexts`.
    recent: Box<[Option<(LegacyContextStamp, u64)>; 4]>,
    // Advances whenever `blocks` may drop or replace a proof, and with it
    // its registration, or `contexts` may give an identity to another stamp;
    // successor links (`links.rs`) name it.
    pub(super) version: u64,
}

impl State {
    pub(super) fn reset(&mut self) {
        self.invalidate();
        self.transaction = None;
        self.event_transaction = None;
        self.port_input = None;
    }

    /// Whether a proof holds this very block.
    pub(super) fn holds(&self, ir: &Arc<IrBlock>) -> bool {
        self.blocks.values().any(|proof| Arc::ptr_eq(&proof.ir, ir))
    }

    pub(super) fn invalidate(&mut self) {
        self.version = self.version.wrapping_add(1);
        self.blocks.clear();
        self.tlb.get_mut().invalidate();
        self.generation = self.tlb.get_mut().generation;
        self.stamp = None;
        self.contexts.clear();
        *self.recent = [None; 4];
    }

    pub(super) fn reconcile(&mut self, vm: &Vm, cpu: &CpuState, model: LegacyCpuModel) {
        let stamp = LegacyContextStamp::new(cpu, model);
        let memory = &vm.mem as *const Memory as usize;
        let source_generation = vm.cache.publication_generation();
        let mapping = vm.mem.mapping_identity();
        if self.model != Some(model)
            || self.memory != memory
            || self
                .mapping
                .as_ref()
                .is_none_or(|m| !Arc::ptr_eq(m, &mapping))
        {
            self.invalidate();
            self.model = Some(model);
            self.memory = memory;
            self.mapping = Some(mapping);
        }
        if self.source_generation != source_generation {
            self.version = self.version.wrapping_add(1);
            self.blocks.clear();
            self.source_generation = source_generation;
        }
        if self.stamp != Some(stamp) {
            self.context_id = if let Some(slot) = self
                .recent
                .iter()
                .position(|r| r.as_ref().is_some_and(|(s, _)| *s == stamp))
            {
                self.recent[..=slot].rotate_right(1);
                self.recent[0].unwrap().1
            } else {
                let id = if let Some(id) = self.contexts.get(&stamp) {
                    *id
                } else {
                    // Bound context identities as well as blocks. Clear both before
                    // reusing an ID; this does not expire cached translations.
                    if self.contexts.len() == MAX_CONTEXTS {
                        self.version = self.version.wrapping_add(1);
                        self.blocks.clear();
                        self.contexts.clear();
                        *self.recent = [None; 4];
                    }
                    let id = self.contexts.len() as u64;
                    self.contexts.insert(stamp, id);
                    id
                };
                self.recent.rotate_right(1);
                self.recent[0] = Some((stamp, id));
                id
            };
            self.stamp = Some(stamp);
        }
        self.tlb.get_mut().reconcile(cpu);
        self.reconcile_generation();
    }

    fn reconcile_generation(&mut self) {
        let generation = self.tlb.get_mut().generation;
        if self.generation != generation {
            self.version = self.version.wrapping_add(1);
            self.blocks.clear();
            self.generation = generation;
        }
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
            generation: self.generation,
            cr3: cpu.legacy.cr3,
            cr0: cpu.legacy.cr0,
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
    ) -> Result<(Arc<IrBlock>, bool), Exit> {
        let walk_wrote = Cell::new(false);
        loop {
            vm.handle_smc(exit_word);
            self.reconcile(vm, cpu, model);
            let key = self.key(cpu, model, context, owner_id, ordinary)?;
            if let Some(proof) = self
                .blocks
                .get(&key)
                .filter(|p| p.permit.as_ref().is_none_or(|p| p.is_current()))
            {
                probe!(PROOFS);
                return Ok((proof.ir.clone(), proof.ordinary));
            }
            probe!(LIFTS);
            let source = crate::memory::LiftSource::new(&vm.mem);
            let generation = vm.cache.publication_generation();
            let permit = vm.mem.fetch_permit();
            let (ir, sources) = lift(vm, cpu, context, &source, &walk_wrote, &self.tlb, ordinary)
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
            if vm.cache.publication_generation() != generation {
                continue;
            }
            self.reconcile_generation();
            let key = self.key(cpu, model, context, owner_id, ordinary)?;
            self.lifts += 1;
            let direct_exit = ir.ends_direct();
            let ir = Arc::new(IrBlock { direct_exit, ..ir });
            if !source.publishable() {
                // The interpreter may use this snapshot once, without a proof.
                self.version = self.version.wrapping_add(1);
                self.blocks.remove(&key);
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
                continue;
            }
            if self.blocks.len() == MAX_BLOCKS && !self.blocks.contains_key(&key) {
                self.version = self.version.wrapping_add(1);
                self.blocks.clear();
            }
            if self
                .blocks
                .insert(
                    key,
                    Proof {
                        ir: ir.clone(),
                        ordinary,
                        permit,
                        _registration: registration.unwrap(),
                    },
                )
                .is_some()
            {
                self.version = self.version.wrapping_add(1);
            }
            return Ok((ir, ordinary));
        }
    }
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
fn lift(
    vm: &Vm,
    cpu: &CpuState,
    context: MachineContext,
    source: &crate::memory::LiftSource<'_>,
    walk_wrote: &Cell<bool>,
    tlb: &std::cell::RefCell<crate::interp::legacy::paging::Tlb>,
    blocks: bool,
) -> Result<(IrBlock, Vec<(u64, u32)>), Exit> {
    let mut cursor = cpu.clone();
    let linear = cpu.legacy.segments[0].base.wrapping_add(cpu.rip as u32);
    let page_end = cpu.rip + (4096 - u64::from(linear & 4095));
    let mut block: Option<IrBlock> = None;
    let mut sources = Vec::new();
    let mut reloaded = 0;
    loop {
        if cursor.rip + 15 > page_end {
            if let Some(block) = block.take() {
                return Ok((block, sources));
            }
        }
        let fetched = fetch(vm, &mut cursor, context, source, walk_wrote, tlb);
        let (bytes, spans) = match fetched {
            Ok(fetched) => fetched,
            Err(exit) => return block.map(|b| (b, sources)).ok_or(exit),
        };
        let mut instruction = match crate::lift::legacy::instruction_for_profile(
            &bytes,
            cursor.rip as u32,
            context,
            vm.machine_model_profile(context.generation),
        ) {
            Ok(ir) => ir,
            Err(error) => {
                return block
                    .map(|b| (b, sources))
                    .ok_or_else(|| direct_refusal(&cursor, error))
            }
        };
        let loads = super::flat32::specialize_segments(&mut instruction, cpu, reloaded);
        let ordinary = blocks
            && ordinary_context(context)
            && crate::interp::legacy::flat32_ordinary(&instruction);
        if !ordinary {
            if let Some(block) = block.take() {
                return Ok((block, sources));
            }
        }
        sources.extend(spans);
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
            return Ok((block.unwrap(), sources));
        }
    }
}

type Fetched = (Vec<u8>, Vec<(u64, u32)>);

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
            fetch_bus_for_test(cpu, &vm.mem, linear as u32, walk_wrote, Some(tlb))?
        } else {
            crate::legacy::physical_span(cpu, u32::MAX, linear, 1).0
        };
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
    Ok((bytes, sources))
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
        let state = execution.paging.get_or_insert_with(Default::default);
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
            let state = execution.paging.get_or_insert_with(Default::default);
            return crate::interp::legacy::paging::execute_flat32::<ROUTED, ORDER>(
                ir,
                &mut self.cpu,
                &vm.mem,
                &mut self.interp_scratch,
                limit,
                &self.exit_request.word,
                &mut execution.task_switch,
                &mut state.transaction,
                &state.tlb,
            );
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
        let state = execution.paging.get_or_insert_with(Default::default);
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
                state.invalidate();
            } else if outcome.context_changed {
                state.reconcile(vm, &self.cpu, self.legacy_model.unwrap());
            }
            outcome
        });
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
            state.invalidate();
        } else if outcome.context_changed {
            state.reconcile(vm, &self.cpu, model.legacy_model());
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
mod tests {
    use super::*;

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
    /// V86 stack page 47h and the ring-0 stack page 6 share a TLB set.
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
        }
        vm.write_bytes(0x3808, &0x00cf_9b00_0000_ffffu64.to_le_bytes())
            .unwrap();
        vm.write_bytes(0x3810, &0x00cf_9300_0000_ffffu64.to_le_bytes())
            .unwrap();
        // Vector 13: 32-bit interrupt gate, DPL 0, to 8:9000h.
        vm.write_bytes(0x3000 + 13 * 8, &0x00008e00_00089000u64.to_le_bytes())
            .unwrap();
        vm.write_bytes(0x5004, &[0, 0x70, 0, 0, 0x10, 0]).unwrap();
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
        // translation leaves the TLB. Data page 51h shares code page 10h's
        // set, so MOV AL,[0] replaces the fetch entry each iteration and the
        // next fetch must walk to the edited frame.
        // MOV AL,[0]; INC CX (DEC CX at frame 30h); JMP to the MOV.
        for model in [crate::CpuModel::I80386, crate::CpuModel::I80486Dx] {
            let (vm, mut cpu) = v86_monitor(model, &[0xa0, 0, 0, 0x41, 0xeb, 0xfa], true);
            vm.write_bytes(0x30000, &[0xa0, 0, 0, 0x49, 0xeb, 0xfa])
                .unwrap();
            let options = crate::StepOptions::default();
            assert_eq!(cpu.run_steps(&vm, 30, options).unwrap().steps(), 30);
            assert_eq!(cpu.cpu.gpr[1], 10);
            vm.write_bytes(0x2000 + 0x10 * 4, &0x30007u32.to_le_bytes())
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
                )
                .unwrap()
                .0;
            assert!(!Arc::ptr_eq(&a, &c));
        }
    }
}
