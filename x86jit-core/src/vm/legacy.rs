//! Private legacy translation namespace and dispatch (decision-18 §§3–4;
//! spec.md §§9, 17). No application cache/backend entry is used here.
use super::*;
use crate::legacy::{
    derive_legacy_context, CpuGeneration, LegacyCpuState, LegacyDecodeContext, LegacyRegime,
    SegmentId,
};
use iced_x86::{Decoder, DecoderError, DecoderOptions};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct LegacyBlockKey {
    owner_id: u64,
    context_epoch: u64,
    model_semantics_id: u64,
    decode_bits: u8,
    stack_bits: u8,
    regime: LegacyRegime,
    decode_ip: u32,
    linear_fetch: u32,
    mxcsr_mode: u32,
}
#[derive(Clone, Copy, PartialEq, Eq)]
struct LegacyContextStamp {
    model: LegacyCpuModel,
    state: LegacyCpuState,
    selectors: [u16; 6],
    system_flags: u64,
}
impl LegacyContextStamp {
    fn new(cpu: &CpuState, model: LegacyCpuModel) -> Self {
        let mut state = cpu.legacy;
        state.cr2 = 0; // Output register, not a translation assumption.
        Self {
            model,
            state,
            selectors: [cpu.cs, cpu.ds, cpu.es, cpu.ss, cpu.fs, cpu.gs],
            system_flags: cpu.rflags_system & 0x7000,
        }
    }
}
struct Block {
    ir: Arc<IrBlock>,
    bytes: Vec<u8>,
}
pub(super) struct Execution {
    owner_id: u64,
    context_epoch: u64,
    stamp: Option<LegacyContextStamp>,
    blocks: HashMap<LegacyBlockKey, Block>,
}
impl Execution {
    pub(super) fn new() -> Self {
        static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
        let owner_id = NEXT_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .expect("legacy owner identities exhausted");
        Self {
            owner_id,
            context_epoch: 0,
            stamp: None,
            blocks: HashMap::new(),
        }
    }
    fn sync(&mut self, cpu: &CpuState, model: LegacyCpuModel, force: bool) {
        let stamp = LegacyContextStamp::new(cpu, model);
        if force || self.stamp != Some(stamp) {
            self.blocks.clear();
            self.context_epoch = self.context_epoch.checked_add(1).unwrap_or(0);
            self.stamp = Some(stamp);
        }
    }
    fn resolve(
        &mut self,
        vm: &Vm,
        cpu: &CpuState,
        model: LegacyCpuModel,
        context: LegacyDecodeContext,
    ) -> Result<Arc<IrBlock>, Exit> {
        let linear_fetch = crate::legacy::access(cpu, SegmentId::Cs, cpu.rip, 1)? as u32;
        let key = LegacyBlockKey {
            owner_id: self.owner_id,
            context_epoch: self.context_epoch,
            model_semantics_id: model.model_semantics_id,
            decode_bits: context.default_operand_bits,
            stack_bits: context.stack_address_bits,
            regime: context.regime,
            decode_ip: cpu.rip as u32,
            linear_fetch,
            mxcsr_mode: cpu.mxcsr & crate::jit_abi::fp::MODE_MASK,
        };
        // One-instruction blocks bound fault visibility and SMC. Re-fetch each
        // dependent byte, even on cache hits: permissions, policy and physical
        // aliases are checked before executing that instruction, without page
        // version/TLB assumptions reserved for the paging phase.
        let bytes = fetch(vm, cpu, context)?;
        if let Some(block) = self.blocks.get(&key) {
            if block.bytes == bytes {
                return Ok(block.ir.clone());
            }
        }
        let ir =
            crate::lift::legacy::instruction(&bytes, cpu.rip as u32, context).map_err(|error| {
                match error {
                    crate::lift::LiftError::Unsupported { addr, bytes, len } => {
                        Exit::unknown_instruction(addr, bytes, len)
                    }
                    crate::lift::LiftError::DecodeFault { addr } => {
                        Exit::unmapped_memory(addr, crate::AccessKind::Execute)
                    }
                }
            })?;
        let ir = Arc::new(ir);
        self.blocks.insert(
            key,
            Block {
                ir: ir.clone(),
                bytes,
            },
        );
        Ok(ir)
    }
}

fn fetch(vm: &Vm, cpu: &CpuState, context: LegacyDecodeContext) -> Result<Vec<u8>, Exit> {
    let mut bytes = Vec::with_capacity(15);
    for i in 0..15 {
        let addr = crate::legacy::access(cpu, SegmentId::Cs, cpu.rip + i, 1)?;
        if let Some(policy) = &vm.mem.fetch_policy {
            let verdict = policy.approve(addr >> crate::memory::CODE_PAGE_BITS, &vm.mem);
            if verdict != crate::FetchVerdict::Allow {
                return Err(Exit::unmapped_memory(addr, crate::AccessKind::Execute));
            }
        }
        let mut byte = [0];
        if let Err(trap) = vm.mem.fetch_code(addr, &mut byte) {
            return Err(crate::fault::fetch_exit(
                &vm.mem,
                FetchAddr {
                    pa: u64::from(cpu.legacy.segments[0].base) + cpu.rip,
                    ip: cpu.rip,
                },
                CpuMode::Compat32,
                addr,
                Some(crate::lift::FetchFailure::Memory(trap)),
            ));
        }
        bytes.push(byte[0]);
        let mut decoder = Decoder::with_ip(
            u32::from(context.default_operand_bits),
            &bytes,
            cpu.rip,
            DecoderOptions::MOV_TR,
        );
        let insn = decoder.decode();
        if !insn.is_invalid() || decoder.last_error() != DecoderError::NoMoreBytes {
            break;
        }
    }
    Ok(bytes)
}

impl Vcpu {
    fn legacy_context_for_run(&self) -> Result<LegacyDecodeContext, Exit> {
        let model = self.legacy_model.expect("legacy model");
        let context = derive_legacy_context(model, &self.cpu).map_err(|e| {
            Exit::UnsupportedSystemContext {
                addr: self.cpu.rip,
                operation: UnsupportedSystemOperation::InvalidLegacyContext(e),
            }
        })?;
        if context.regime != LegacyRegime::Real {
            return Err(self.unsupported_legacy(UnsupportedSystemOperation::LegacyExecution));
        }
        // Other generations, A20 gating and reset-origin execution are not activated by the real-forms phase.
        if model.generation != CpuGeneration::I386
            || self.cpu.legacy.a20_enabled == 0
            || self.cpu.legacy.reset_fetch != 0
            || self.cpu.rip > u64::from(u32::MAX)
        {
            return Err(self.unsupported_legacy(UnsupportedSystemOperation::LegacyEntryFacility));
        }
        Ok(context)
    }

    pub(super) fn run_legacy(
        &mut self,
        vm: &Vm,
        limit: Option<u64>,
        options: StepOptions,
        stepped: bool,
    ) -> Result<StepRun, StepRunError> {
        if let Err(exit) = self.legacy_context_for_run() {
            return Ok(StepRun {
                steps: 0,
                stop: StepStop::Exit(exit),
            });
        }
        if limit == Some(0) {
            return Ok(StepRun {
                steps: 0,
                stop: StepStop::Exit(Exit::BudgetExhausted),
            });
        }
        // Decision-16 §§1,3: validate CS:RIP ownership before execution,
        // while leaving every completion slot untouched at zero budget.
        if self.cpu.mmio_retry_active && !self.cpu.owns_mmio_retry(self.cpu.rip) {
            self.cpu.clear_mmio_parts();
        }
        if stepped && self.cpu.pending_port_in.is_some() {
            return Err(StepRunError::PendingPortIn);
        }
        self.interruption_evidence = None;
        self.syscall_evidence = None;
        self.legacy_execution
            .as_mut()
            .unwrap()
            .sync(&self.cpu, self.legacy_model.unwrap(), false);
        let time_run = crate::time_stats::RunGuard::enter(
            self.time_record.clone(),
            vm.time_enabled,
            self.time_outside.take(),
        );
        let _fp = crate::jit_abi::fp::RunMode::new();
        let dispatch = DispatchGuard::new(self, vm);
        let cpu = &mut *dispatch.cpu;
        cpu.fault_entry.prepare_run();
        struct FaultRunGuard(*mut crate::jit_abi::fault::EntryRecord);
        impl Drop for FaultRunGuard {
            fn drop(&mut self) {
                // SAFETY: the boxed entry remains fixed throughout the exclusive
                // vCPU borrow, including if a fetch-policy callback unwinds.
                unsafe {
                    (*self.0).deactivate();
                }
            }
        }
        let guard = FaultRunGuard(&mut *cpu.fault_entry);
        let mut ctx = MemCtx::for_memory(&vm.mem);
        cpu.fault_entry.publish(&mut cpu.cpu, &mut ctx);
        let mut result = match (vm.mem.host_fault_recovery(), vm.consistency) {
            (true, MemConsistency::Fast) => cpu.legacy_loop::<true, 0>(vm, limit, options, stepped),
            (false, MemConsistency::Fast) => {
                cpu.legacy_loop::<false, 0>(vm, limit, options, stepped)
            }
            (true, MemConsistency::AcqRel) => {
                cpu.legacy_loop::<true, 1>(vm, limit, options, stepped)
            }
            (false, MemConsistency::AcqRel) => {
                cpu.legacy_loop::<false, 1>(vm, limit, options, stepped)
            }
            (true, MemConsistency::FullTso) => {
                cpu.legacy_loop::<true, 2>(vm, limit, options, stepped)
            }
            (false, MemConsistency::FullTso) => {
                cpu.legacy_loop::<false, 2>(vm, limit, options, stepped)
            }
        };
        drop(guard);
        if let StepStop::Exit(exit @ (Exit::UnmappedMemory { .. } | Exit::Exception { .. })) =
            result.stop
        {
            result.stop = StepStop::Exit(cpu.fault_entry.finish_transfer_fault(exit, &vm.mem));
        }
        cpu.cpu.finish_mmio_retry(match &result.stop {
            StepStop::Exit(exit) => Some(exit),
            StepStop::IrqReady => None,
        });
        drop(dispatch);
        self.time_outside = time_run.finish(match &result.stop {
            StepStop::Exit(exit) => Some(exit),
            StepStop::IrqReady => None,
        });
        Ok(result)
    }

    fn legacy_loop<const ROUTED: bool, const ORDER: u8>(
        &mut self,
        vm: &Vm,
        limit: Option<u64>,
        _options: StepOptions,
        _stepped: bool,
    ) -> StepRun {
        let model = self.legacy_model.unwrap();
        let mut steps = 0;
        loop {
            let stop = if limit.is_some_and(|limit| steps >= limit) {
                Some(Exit::BudgetExhausted)
            } else if !self.cpu.mmio_retry_active
                && self.cpu.pending_port_in.is_none()
                && self.cpu.pending_mmio.is_none()
                && !self.cpu.pending_mmio_write
                && self.cpu.port_string_in.is_none()
                && self.exit_request.word.load(Ordering::Relaxed) & EXIT_REQUESTED != 0
            {
                Some(self.interrupted(crate::FaultPath::Interpreter, false))
            } else {
                None
            };
            if let Some(exit) = stop {
                return StepRun {
                    steps,
                    stop: StepStop::Exit(exit),
                };
            }
            let context = match self.legacy_context_for_run() {
                Ok(c) => c,
                Err(exit) => {
                    return StepRun {
                        steps,
                        stop: StepStop::Exit(exit),
                    }
                }
            };
            if self.pending_irq.is_some()
                && self.irq_delivery_ready()
                && !self.cpu.mmio_retry_active
            {
                return StepRun {
                    steps,
                    stop: StepStop::Exit(
                        self.unsupported_legacy(UnsupportedSystemOperation::LegacyEventDelivery),
                    ),
                };
            }
            self.legacy_execution
                .as_mut()
                .unwrap()
                .sync(&self.cpu, model, false);
            let ir = match self
                .legacy_execution
                .as_mut()
                .unwrap()
                .resolve(vm, &self.cpu, model, context)
            {
                Ok(ir) => ir,
                Err(exit) => {
                    return StepRun {
                        steps,
                        stop: StepStop::Exit(exit),
                    }
                }
            };
            match crate::interp::legacy::execute::<ROUTED, ORDER>(
                &ir,
                &mut self.cpu,
                &vm.mem,
                &mut self.interp_scratch,
            ) {
                Ok(outcome) => {
                    steps += 1;
                    self.retired += u64::from(outcome.retired);
                    self.executed += u64::from(outcome.retired);
                    self.sti_shadow = outcome.arm_sti;
                    self.ss_shadow = outcome.arm_ss && !self.ss_shadow;
                    self.cpu.finish_mmio_retry(None);
                    if outcome.context_changed {
                        self.legacy_execution
                            .as_mut()
                            .unwrap()
                            .sync(&self.cpu, model, true);
                    }
                    if let Some(exit) = outcome.exit {
                        return StepRun {
                            steps,
                            stop: StepStop::Exit(exit),
                        };
                    }
                }
                Err(exit) => {
                    return StepRun {
                        steps,
                        stop: StepStop::Exit(exit),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn epoch_stamp_guards_context_but_not_live_outputs_and_rolls_over_empty() {
        let model = LegacyCpuModel {
            generation: CpuGeneration::I386,
            model_semantics_id: 1,
        };
        let cpu = CpuState::legacy_real_entry([0; 6]);
        let mut e = Execution::new();
        e.sync(&cpu, model, false);
        let epoch = e.context_epoch;
        let mut live = cpu.clone();
        live.legacy.cr2 = 0x1234;
        live.flags.if_ = true;
        e.sync(&live, model, false);
        assert_eq!(e.context_epoch, epoch);
        let edits: [fn(&mut CpuState); 12] = [
            |c| c.legacy.cr0 = 1,
            |c| c.legacy.cr3 = 0x1000,
            |c| c.legacy.cpl = 1,
            |c| c.ds = 3,
            |c| c.legacy.segments[4].base = 0x1000,
            |c| c.legacy.gdtr.limit = 0xff,
            |c| c.legacy.idtr.base = 0x2000,
            |c| c.legacy.ldtr.selector = 8,
            |c| c.legacy.tr.selector = 16,
            |c| c.rflags_system = 0x7000,
            |c| c.legacy.a20_enabled = 0,
            |c| c.legacy.reset_fetch = 1,
        ];
        for edit in edits {
            let mut changed = cpu.clone();
            edit(&mut changed);
            let before = e.context_epoch;
            e.sync(&changed, model, false);
            assert_ne!(e.context_epoch, before);
        }
        let before = e.context_epoch;
        e.sync(
            &cpu,
            LegacyCpuModel {
                model_semantics_id: 2,
                ..model
            },
            false,
        );
        assert_ne!(e.context_epoch, before);
        e.context_epoch = u64::MAX;
        e.sync(&cpu, model, true);
        assert_eq!(e.context_epoch, 0);
        assert!(e.blocks.is_empty());
        assert_ne!(e.owner_id, Execution::new().owner_id);
    }
}
