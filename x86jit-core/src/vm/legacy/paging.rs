//! Internal one-instruction paging fetch/proofs (paging design §§3.2, 3.5;
//! spec.md §17.10). No shared dispatch entry or compiled materialization.
use super::*;
use crate::interp::legacy::paging::{execute_for_test, fetch_bus_for_test, publish_cr2};
use std::cell::Cell;

const MAX_BLOCKS: usize = 64;

#[derive(PartialEq, Eq, Hash)]
struct Key {
    decode: LegacyBlockKey,
    generation: u64,
    cr3: u32,
    cr0: u32,
}

struct Proof {
    ir: Arc<IrBlock>,
    bytes: Vec<u8>,
    // Ordered post-A20 bus spans, never a linear length applied to one frame.
    sources: Vec<(u64, u32)>,
    _registration: crate::memory::CodeRegistration,
}

#[derive(Default)]
pub(super) struct State {
    generation: u64,
    stamp: Option<LegacyContextStamp>,
    memory: usize,
    source_generation: u64,
    blocks: HashMap<Key, Proof>,
    lifts: u64,
    before_publication: Option<fn(&Vm)>,
}

impl State {
    pub(super) fn invalidate(&mut self) {
        self.blocks.clear();
        // Dropping entries before wrap prevents identifier reuse retaining proofs.
        self.generation = self.generation.wrapping_add(1);
        self.stamp = None;
    }

    fn reconcile(&mut self, vm: &Vm, cpu: &CpuState, model: LegacyCpuModel) {
        let stamp = LegacyContextStamp::new(cpu, model);
        let memory = &vm.mem as *const Memory as usize;
        let source_generation = vm.cache.publication_generation();
        if self.stamp != Some(stamp)
            || self.memory != memory
            || self.source_generation != source_generation
        {
            self.invalidate();
            self.stamp = Some(stamp);
            self.memory = memory;
            self.source_generation = source_generation;
        }
    }

    #[cold]
    #[inline(never)]
    fn resolve(
        &mut self,
        vm: &Vm,
        cpu: &mut CpuState,
        model: crate::CpuModel,
        context: MachineContext,
        owner_id: u64,
        exit_word: &AtomicU64,
    ) -> Result<Arc<IrBlock>, Exit> {
        let walk_wrote = Cell::new(false);
        loop {
            vm.handle_smc(exit_word);
            self.reconcile(vm, cpu, model.legacy_model());
            let source = crate::memory::LiftSource::new(&vm.mem);
            let generation = vm.cache.publication_generation();
            let permit = vm.mem.fetch_permit();
            let (bytes, sources) =
                fetch(vm, cpu, context, &source, &walk_wrote).map_err(|mut exit| {
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
            let linear_fetch = crate::legacy::access(
                cpu,
                context.generation,
                SegmentId::Cs,
                cpu.rip,
                1,
                crate::AccessKind::Execute,
            )? as u32;
            let key = Key {
                decode: LegacyBlockKey {
                    owner_id,
                    context_id: self.generation,
                    model_semantics_id: model.legacy_model().model_semantics_id,
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
            };
            let ir = if let Some(proof) = self
                .blocks
                .get(&key)
                .filter(|p| p.bytes == bytes && p.sources == sources)
            {
                proof.ir.clone()
            } else {
                let ir = crate::lift::legacy::instruction_for_profile(
                    &bytes,
                    cpu.rip as u32,
                    context,
                    model.profile(),
                )
                .map_err(|e| direct_refusal(cpu, e))?;
                self.lifts += 1;
                Arc::new(ir)
            };
            if !source.publishable() {
                // The interpreter may use this snapshot once, without a proof.
                self.blocks.remove(&key);
                return Ok(ir);
            }
            let mut registration = None;
            if !vm.cache.register_private_sources(generation, || {
                registration = Some(vm.mem.register_code_sources(&sources));
            }) || permit.as_ref().is_some_and(|p| !p.is_current())
            {
                continue;
            }
            if self.blocks.len() == MAX_BLOCKS && !self.blocks.contains_key(&key) {
                self.blocks.clear();
            }
            self.blocks.insert(
                key,
                Proof {
                    ir: ir.clone(),
                    bytes,
                    sources,
                    _registration: registration.unwrap(),
                },
            );
            return Ok(ir);
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
            fetch_bus_for_test(cpu, &vm.mem, linear as u32, walk_wrote)?
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
    /// Internal single-instruction fetch/execute entry; public PG=1 stays refused.
    /// Only step-1 scalar instructions and control-register transfers are allowed.
    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    pub fn interpret_paged_for_test(&mut self, vm: &Vm) -> Result<(), Exit> {
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
        if self.legacy_execution.is_none()
            || (model != crate::CpuModel::I80386 && self.cpu.legacy.cr0 & (1 << 16) != 0)
        {
            return Err(Exit::UnsupportedSystemContext {
                addr: self.cpu.rip,
                operation: UnsupportedSystemOperation::LegacyEntryFacility,
            });
        }
        self.cpu.machine_flags_mask = model.machine_flags_mask();
        let context = derive_legacy_context(model.legacy_model(), &self.cpu).map_err(|error| {
            Exit::UnsupportedSystemContext {
                addr: self.cpu.rip,
                operation: UnsupportedSystemOperation::InvalidLegacyContext(error),
            }
        })?;
        let execution = self.legacy_execution.as_mut().unwrap();
        let state = execution.paging.get_or_insert_with(Default::default);
        let result = state.resolve(
            vm,
            &mut self.cpu,
            model,
            context,
            execution.owner_id,
            &self.exit_request.word,
        );
        let ir = match result {
            Ok(ir) => ir,
            Err(exit) => {
                publish_cr2(&mut self.cpu, &exit);
                return Err(exit);
            }
        };
        let outcome = if context.paging_enabled {
            execute_for_test::<true>(&ir, &mut self.cpu, &vm.mem, context, true)
        } else {
            execute_for_test::<false>(&ir, &mut self.cpu, &vm.mem, context, true)
        }?;
        // 80386 PRM §5.2.5 p. 5-11; Intel486 PRM §5.3.5 p. 5-22:
        // loading CR3 flushes even when its value is unchanged.
        if outcome.translations_invalidated || outcome.context_changed {
            state.invalidate();
        }
        outcome.exit.map_or(Ok(()), Err)
    }

    /// Internal proof diagnostics: translation generation, retained blocks, lifts.
    #[doc(hidden)]
    pub fn paging_cache_for_test(&self) -> (u64, usize, u64) {
        self.legacy_execution
            .as_ref()
            .and_then(|e| e.paging.as_ref())
            .map_or((0, 0, 0), |s| (s.generation, s.blocks.len(), s.lifts))
    }

    /// Inject a physical write after fetch and before source registration.
    #[doc(hidden)]
    pub fn paging_publication_hook_for_test(&mut self, hook: fn(&Vm)) {
        self.legacy_execution
            .as_mut()
            .unwrap()
            .paging
            .get_or_insert_with(Default::default)
            .before_publication = Some(hook);
    }
}
