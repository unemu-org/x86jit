//! Machine dispatch with shared Direct blocks and retained Checked contexts
//! (decision-23 §5; decision-20; spec.md §§9, 17). All blocks remain interpreted.
use super::*;
use crate::legacy::{
    derive_legacy_context, CpuGeneration, LegacyCpuState, MachineContext, MachineRegime, SegmentId,
};
use crate::IrOp;
use iced_x86::{Decoder, DecoderError, DecoderOptions};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct LegacyBlockKey {
    owner_id: u64,
    context_id: u64,
    model_semantics_id: u64,
    decode_bits: u8,
    stack_bits: u8,
    regime: MachineRegime,
    decode_ip: u32,
    linear_fetch: u32,
    mxcsr_mode: u32,
}
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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
// Decision-20: the firmware-style transition loop uses four contexts. Sixteen
// leaves room for a handful of code/stack configurations while bounding churn.
const MAX_RETAINED_CONTEXTS: usize = 16;

struct RetainedContext {
    id: u64,
    last_used: u64,
}

struct Block {
    ir: Arc<IrBlock>,
    bytes: Vec<u8>,
}
struct InstructionRetry {
    stamp: LegacyContextStamp,
    ip: u64,
    mxcsr_mode: u32,
    ir: Arc<IrBlock>,
}
#[derive(Clone, Copy)]
enum EventSource {
    SingleStep,
    Irq(u8),
    Exception { vector: u8, error_code: Option<u32> },
}
#[derive(Clone, Copy)]
struct PendingEvent {
    irq_consumed: bool,
    source: EventSource,
    ip: u32,
    cs: u16,
}
fn exception_source(exit: &Exit) -> Option<EventSource> {
    let Exit::Exception { vector, .. } = exit else {
        return None;
    };
    let error_code = exit.fault_metadata().and_then(|m| match m.error_code() {
        crate::fault::ErrorCode::Bits { value, .. } => Some(value),
        _ => None,
    });
    Some(EventSource::Exception {
        vector: *vector,
        error_code,
    })
}

fn double_fault(generation: CpuGeneration, first: EventSource, second: EventSource) -> bool {
    let EventSource::Exception { vector: first, .. } = first else {
        return false;
    };
    let EventSource::Exception { vector: second, .. } = second else {
        return false;
    };
    // I286 PRM §9.6.2 p. 9-10 excludes #9 from the first-fault set;
    // I386 PRM §9.8.8 Table 9-3 p. 9-13 includes it as contributory.
    if generation == CpuGeneration::I286 {
        return matches!(first, 0 | 10..=13);
    }
    let contributory = |vector| matches!(vector, 0 | 9..=13);
    (contributory(first) && contributory(second))
        || (first == 14 && (contributory(second) || second == 14))
}

pub(super) struct Execution {
    owner_id: u64,
    context_epoch: u64,
    context_id: u64,
    stamp: Option<LegacyContextStamp>,
    contexts: HashMap<LegacyContextStamp, RetainedContext>,
    blocks: HashMap<LegacyBlockKey, Block>,
    instruction_retry: Option<InstructionRetry>,
    task_switch: crate::interp::legacy::tasks::SwitchState,
    #[cfg(test)]
    lifts: u64,
    // Accepted event delivery can itself hand off an MMIO transfer. Its retry
    // must complete before executing the interrupted instruction (decision-16 §3).
    event_retry: Option<PendingEvent>,
    // Port handoffs must reach the embedder before the post-instruction trap.
    port_debug: bool,
    shutdown: bool,
    pub(super) pending_a20_enabled: Option<bool>,
    smc_pages: Vec<u64>,
    last_direct_fetch: Option<u64>,
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
            context_id: 0,
            stamp: None,
            contexts: HashMap::new(),
            blocks: HashMap::new(),
            instruction_retry: None,
            task_switch: Default::default(),
            #[cfg(test)]
            lifts: 0,
            event_retry: None,
            port_debug: false,
            shutdown: false,
            pending_a20_enabled: None,
            smc_pages: Vec::new(),
            last_direct_fetch: None,
        }
    }
    pub(super) fn reset_execution_state(&mut self) {
        self.event_retry = None;
        self.task_switch.reset();
        self.port_debug = false;
        self.shutdown = false;
        self.instruction_retry = None;
        self.smc_pages.clear();
        self.last_direct_fetch = None;
    }
    pub(super) fn is_shutdown(&self) -> bool {
        self.shutdown
    }
    pub(super) fn task_switch_pending(&self) -> bool {
        self.task_switch.context().is_some()
    }
    pub(super) fn has_continuation(&self) -> bool {
        self.instruction_retry.is_some()
            || self.event_retry.is_some()
            || self.task_switch.context().is_some()
    }
    fn invalidate_translations(&mut self) {
        self.blocks.clear();
        self.contexts.clear();
        self.stamp = None;
    }
    #[cfg(test)]
    fn sync(&mut self, cpu: &CpuState, model: LegacyCpuModel, force: bool) {
        self.sync_class(cpu, model, force, crate::SegmentClass::Checked);
    }
    fn sync_class(
        &mut self,
        cpu: &CpuState,
        model: LegacyCpuModel,
        force: bool,
        class: crate::SegmentClass,
    ) {
        let stamp = LegacyContextStamp::new(cpu, model);
        if force || self.stamp != Some(stamp) {
            self.context_epoch = if let Some(next) = self.context_epoch.checked_add(1) {
                next
            } else {
                self.invalidate_translations();
                0
            };
            // Decision-18 §3 guard dimensions remain complete. Stable identities
            // implement the accepted transition-cost contract: returning to an
            // identical context reuses its IR instead of discarding other modes.
            // Fetch still revalidates bytes, mappings and policy on every entry.
            self.context_id = if class == crate::SegmentClass::Direct {
                // Direct states do not consume decision-20's Checked LRU slots.
                u64::MAX
            } else if let Some(context) = self.contexts.get_mut(&stamp) {
                context.last_used = self.context_epoch;
                context.id
            } else {
                let id = if self.contexts.len() == MAX_RETAINED_CONTEXTS {
                    let oldest = *self
                        .contexts
                        .iter()
                        .min_by_key(|(_, context)| context.last_used)
                        .unwrap()
                        .0;
                    let evicted = self.contexts.remove(&oldest).unwrap();
                    self.blocks.retain(|key, _| key.context_id != evicted.id);
                    evicted.id
                } else {
                    self.contexts.len() as u64
                };
                self.contexts.insert(
                    stamp,
                    RetainedContext {
                        id,
                        last_used: self.context_epoch,
                    },
                );
                id
            };
            self.stamp = Some(stamp);
        }
    }
    fn resolve(
        &mut self,
        vm: &Vm,
        cpu: &CpuState,
        model: LegacyCpuModel,
        context: MachineContext,
        _exit_word: &AtomicU64,
    ) -> Result<Arc<IrBlock>, Exit> {
        if self.task_switch.context().is_some() {
            if let Some(retry) = &self.instruction_retry {
                return Ok(retry.ir.clone());
            }
        }
        let linear_fetch = crate::legacy::access(
            cpu,
            context.generation,
            SegmentId::Cs,
            cpu.rip,
            1,
            crate::AccessKind::Execute,
        )? as u32;
        let key = LegacyBlockKey {
            owner_id: self.owner_id,
            context_id: self.context_id,
            model_semantics_id: model.model_semantics_id,
            decode_bits: context.default_operand_bits,
            stack_bits: context.stack_address_bits,
            regime: context.regime,
            decode_ip: cpu.rip as u32,
            linear_fetch,
            mxcsr_mode: cpu.mxcsr & crate::jit_abi::fp::MODE_MASK,
        };
        // Decision-16 §3: finish the already decoded instruction on a device
        // retry. A partial CALL frame can overwrite the CALL's own bytes;
        // its IR must survive eviction and reuse of the translation's context ID.
        if let Some(retry) = &self.instruction_retry {
            if retry.stamp == LegacyContextStamp::new(cpu, model)
                && retry.ip == cpu.rip
                && retry.mxcsr_mode == key.mxcsr_mode
            {
                return Ok(retry.ir.clone());
            }
        }

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
        #[cfg(test)]
        {
            self.lifts += 1;
        }
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

fn retry_instruction(ir: &Arc<IrBlock>, ip: u64) -> Arc<IrBlock> {
    let start = ir
        .ops
        .iter()
        .position(|op| matches!(op, IrOp::InsnStart { guest_addr } if *guest_addr == ip))
        .unwrap();
    let end = ir.ops[start + 1..]
        .iter()
        .position(|op| matches!(op, IrOp::InsnStart { .. }))
        .map_or(ir.ops.len(), |offset| start + 1 + offset);
    let next = match ir.ops.get(end) {
        Some(IrOp::InsnStart { guest_addr }) => *guest_addr,
        _ => ir.guest_start + u64::from(ir.guest_len),
    };
    let mut ops = ir.ops[start..end].to_vec();
    if matches!(
        ir.ops.last(),
        Some(IrOp::Legacy(crate::legacy::LegacyOp::SharedWalk))
    ) && !matches!(
        ops.last(),
        Some(IrOp::Legacy(crate::legacy::LegacyOp::SharedWalk))
    ) {
        ops.push(IrOp::Legacy(crate::legacy::LegacyOp::SharedWalk));
    }
    Arc::new(IrBlock {
        mode: ir.mode,
        guest_start: ip,
        guest_len: (next - ip) as u32,
        icount: 1,
        temp_count: ir.temp_count,
        ops,
        legacy_context: ir.legacy_context,
    })
}

fn raw_direct(cpu: &CpuState) -> bool {
    cpu.legacy.a20_enabled != 0
        || [cpu.cs, cpu.ds, cpu.es, cpu.ss, cpu.fs, cpu.gs]
            .iter()
            .all(|selector| *selector <= 0xf000)
}

fn shared_instruction(insn: &iced_x86::Instruction) -> bool {
    use iced_x86::Mnemonic::*;
    // The compatibility decoder faults beyond ten bytes; I386 allows fifteen
    // (80286 PRM App. C item 10 p. C-2; 80386 PRM §14.7 item 6 p. 14-6).
    if insn.len() > 10 {
        return false;
    }
    // The alias wraps sequential IP; Machine retains EIP past FFFFh
    // (80386 PRM §14.7 item 8). Specialize this edge at lift time.
    if insn.ip() + insn.len() as u64 == 0x10000 {
        return false;
    }
    match insn.mnemonic() {
        Mov => !(0..insn.op_count()).any(|i| {
            insn.op_register(i).is_cr()
                || insn.op_register(i).is_dr()
                || insn.op_register(i).is_tr()
        }),
        Push | Pop | Call | Ret | Retf | Jmp | Pushf | Pushfd | Popf | Popfd | Pusha | Pushad
        | Popa | Popad | Iret | Iretd | Enter | Leave | Les | Lds | Lss | Lfs | Lgs | Bound
        | Xlatb | Loop | Loope | Loopne | Jcxz | Jecxz => true,
        Movsb | Movsw | Movsd | Stosb | Stosw | Stosd | Lodsb | Lodsw | Lodsd | Cmpsb | Cmpsw
        | Cmpsd | Scasb | Scasw | Scasd => {
            // The frozen alias has no FS/GS segment personality and uses #GP
            // for SS limits; Machine I386 has FS/GS and #SS (PRM §14.6 Table 14-1).
            insn.is_string_instruction()
                && !matches!(
                    insn.memory_segment(),
                    iced_x86::Register::FS | iced_x86::Register::GS | iced_x86::Register::SS
                )
        }
        Lea | Add | Adc | Sub | Sbb | And | Or | Xor | Cmp | Test | Inc | Dec | Neg | Not
        | Xchg | Shl | Sal | Shr | Sar | Rol | Ror | Rcl | Rcr | Shld | Shrd | Mul | Imul | Div
        | Idiv | Movzx | Movsx | Cbw | Cwde | Cwd | Cdq | Clc | Stc | Cmc | Cld | Std | Lahf
        | Sahf | Nop | Hlt | In | Out | Cli | Sti => true,
        Jb | Jbe | Je | Jg | Jge | Jl | Jle | Jae | Ja | Jne | Jno | Jnp | Jns | Jo | Jp | Js => {
            // The frozen alias's conditional-branch IP wrap is outside the
            // unified stack/control families; Machine checks the full EIP.
            insn.code().op_code().operand_size() != 32
        }
        _ => false,
    }
}

#[inline(always)]
fn resolve_direct(
    vm: &Vm,
    cpu: &CpuState,
    context: MachineContext,
    exit_word: &AtomicU64,
    physical: u64,
) -> Result<Arc<IrBlock>, Exit> {
    let key = BlockKey::machine_direct(physical, cpu.mxcsr, cpu.legacy.a20_enabled != 0);
    if let Some(CachedBlock::Interpreted(ir)) = vm.cache.get_for_dispatch(key, vm.time_enabled) {
        if ir.guest_start == cpu.rip && ir.legacy_context == Some(context) {
            return Ok(ir);
        }
    }
    resolve_direct_uncached(vm, cpu, context, exit_word, physical, key)
}

#[cold]
#[inline(never)]
fn resolve_direct_uncached(
    vm: &Vm,
    cpu: &CpuState,
    context: MachineContext,
    exit_word: &AtomicU64,
    physical: u64,
    key: BlockKey,
) -> Result<Arc<IrBlock>, Exit> {
    if cpu.rip > 0xffff {
        // 80386 PRM §14.7 item 8: EIP outside CS faults before fetching.
        crate::legacy::access(
            cpu,
            context.generation,
            SegmentId::Cs,
            cpu.rip,
            1,
            crate::AccessKind::Execute,
        )?;
    }
    loop {
        let source = crate::memory::LiftSource::new(&vm.mem);
        let generation = vm.cache.publication_generation();
        let permit = vm.mem.fetch_permit();
        let mut failure = None;
        let lifted = crate::lift::lift_block_context(
            &vm.mem,
            FetchAddr {
                pa: physical,
                ip: cpu.rip,
            },
            CpuMode::Real16,
            &mut failure,
            &source,
            Some(context),
        );
        let mut ir = if let Ok(mut ir) = lifted {
            let mut code = vec![0; ir.guest_len as usize];
            let copied = vm.mem.fetch_code(physical, &mut code).unwrap_or(0);
            code.truncate(copied);
            let mut decoder = Decoder::with_ip(16, &code, cpu.rip, DecoderOptions::MOV_TR);
            let mut count = 0;
            let mut end = cpu.rip;
            let mut special = None;
            let mut strings = Vec::new();
            while decoder.can_decode() {
                let position = decoder.position();
                let insn = decoder.decode();
                let bytes = &code[position..decoder.position()];
                // Match incremental Machine fetch: an unavailable I286 opcode
                // byte wins before a later ESC can reach its MSW gate.
                let reference_bytes = if context.generation == CpuGeneration::I286
                    && crate::lift::legacy::unavailable_i286_opcode(bytes)
                {
                    let end = (1..=bytes.len())
                        .find(|&end| crate::lift::legacy::unavailable_i286_opcode(&bytes[..end]))
                        .expect("unavailable opcode byte");
                    &bytes[..end]
                } else {
                    bytes
                };
                let reference =
                    crate::lift::legacy::instruction(reference_bytes, insn.ip() as u32, context);
                let allowed = !insn.is_invalid()
                    && shared_instruction(&insn)
                    && reference.as_ref().is_ok_and(|r| {
                        !r.ops.iter().any(|op| {
                            matches!(
                                op,
                                IrOp::Trap { .. }
                                    | IrOp::Legacy(crate::legacy::LegacyOp::Refuse { .. })
                            )
                        })
                    });
                if !allowed {
                    special = if insn.is_invalid() {
                        None
                    } else {
                        reference.ok()
                    };
                    break;
                }
                if let Some(op) = reference.as_ref().ok().and_then(|ir| {
                    ir.ops.iter().find(|op| {
                        matches!(op, IrOp::Legacy(crate::legacy::LegacyOp::String { .. }))
                    })
                }) {
                    strings.push((insn.ip(), op.clone()));
                }
                count += 1;
                end = insn.ip() + insn.len() as u64;
            }
            if count == 0 {
                if let Some(ir) = special {
                    ir
                } else {
                    let bytes = fetch(vm, cpu, context)?;
                    crate::lift::legacy::instruction(&bytes, cpu.rip as u32, context)
                        .map_err(|_| Exit::unknown_instruction(cpu.rip, [0; 15], 0))?
                }
            } else {
                if end < ir.guest_start + u64::from(ir.guest_len) {
                    let stop = ir
                        .ops
                        .iter()
                        .position(
                            |op| matches!(op, IrOp::InsnStart { guest_addr } if *guest_addr == end),
                        )
                        .unwrap();
                    ir.ops.truncate(stop);
                }
                ir.guest_len = (end - cpu.rip) as u32;
                ir.icount = count;
                if !strings.is_empty() {
                    // Preserve the width-aware Machine string op. The generic
                    // walk uses its selector bases with the shared string engine;
                    // the frozen alias keeps its existing string IR.
                    let mut ops = Vec::with_capacity(ir.ops.len());
                    let mut string = false;
                    for op in ir.ops {
                        if let IrOp::InsnStart { guest_addr } = op {
                            ops.push(op);
                            string = false;
                            if let Some((_, body)) =
                                strings.iter().find(|(ip, _)| *ip == guest_addr)
                            {
                                ops.push(body.clone());
                                string = true;
                            }
                        } else if !string {
                            ops.push(op);
                        }
                    }
                    ir.ops = ops;
                }
                // Opaque instructions retain the instruction helper walk's fault
                // evidence and context handoff until the walks are unified.
                if !ir.ops.iter().any(|op| {
                    matches!(op,
                    IrOp::Legacy(op) if crate::lift::legacy::shared_helper(op))
                }) {
                    ir.ops
                        .push(IrOp::Legacy(crate::legacy::LegacyOp::SharedWalk));
                }
                ir
            }
        } else {
            let bytes = fetch(vm, cpu, context)?;
            crate::lift::legacy::instruction(&bytes, cpu.rip as u32, context)
                .map_err(|_| Exit::unknown_instruction(cpu.rip, [0; 15], 0))?
        };
        ir.legacy_context = Some(context);
        if !source.validate() || permit.as_ref().is_some_and(|p| !p.is_current()) {
            continue;
        }
        if !source.publishable() {
            let CachedBlock::Interpreted(ir) = source.uncached(ir) else {
                unreachable!()
            };
            return Ok(ir);
        }
        let ir = Arc::new(ir);
        #[cfg(test)]
        vm.after_source_read();
        vm.handle_smc(exit_word);
        if vm.cache.insert_checked(
            key,
            CachedBlock::Interpreted(ir.clone()),
            vec![(physical, ir.guest_len)],
            |spans| {
                if vm.cache.publication_generation() != generation
                    || permit.as_ref().is_some_and(|p| !p.is_current())
                {
                    return false;
                }
                for (start, len) in spans {
                    vm.mem.mark_code(*start, *len);
                }
                true
            },
        ) {
            return Ok(ir);
        }
    }
}

fn fetch(vm: &Vm, cpu: &CpuState, context: MachineContext) -> Result<Vec<u8>, Exit> {
    fetch_watched(vm, cpu, context, cpu.rip, None)
}
fn fetch_watched(
    vm: &Vm,
    cpu: &CpuState,
    context: MachineContext,
    ip: u64,
    source: Option<&crate::memory::LiftSource<'_>>,
) -> Result<Vec<u8>, Exit> {
    let mut bytes = Vec::with_capacity(15);
    // I286 PRM Appendix B #GP p. B-9: instructions are at most ten bytes.
    let length_limit = if context.generation == CpuGeneration::I286 {
        10
    } else {
        15
    };
    for i in 0..length_limit {
        let linear = crate::legacy::access(
            cpu,
            context.generation,
            SegmentId::Cs,
            ip + i,
            1,
            crate::AccessKind::Execute,
        )?;
        let (addr, _) = crate::legacy::physical_span(cpu, linear, 1);
        if let Some(policy) = &vm.mem.fetch_policy {
            let verdict = policy.approve(addr >> crate::memory::CODE_PAGE_BITS, &vm.mem);
            if verdict != crate::FetchVerdict::Allow {
                return Err(crate::fault::qualify_legacy_bus(
                    crate::fault::fetch_exit(
                        &vm.mem,
                        FetchAddr { pa: addr, ip },
                        CpuMode::Compat32,
                        addr,
                        Some(crate::lift::FetchFailure::Policy(verdict)),
                    ),
                    linear,
                    addr,
                ));
            }
        }
        if let Some(source) = source {
            source.watch(addr, 1);
        }
        let mut byte = [0];
        if let Err(trap) = vm.mem.fetch_code(addr, &mut byte) {
            return Err(crate::fault::qualify_legacy_bus(
                crate::fault::fetch_exit(
                    &vm.mem,
                    FetchAddr { pa: addr, ip },
                    CpuMode::Compat32,
                    addr,
                    Some(crate::lift::FetchFailure::Memory(trap)),
                ),
                linear,
                addr,
            ));
        }
        if let Some(source) = source {
            source.record(addr, &byte);
        }
        bytes.push(byte[0]);
        if context.generation == CpuGeneration::I286
            && crate::lift::legacy::unavailable_i286_opcode(&bytes)
        {
            break;
        }
        let mut decoder = Decoder::with_ip(
            u32::from(context.default_operand_bits),
            &bytes,
            ip,
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
    pub(super) fn legacy_context_for_run(&self) -> Result<MachineContext, Exit> {
        let model = self.legacy_model.expect("legacy model");
        if let Some(context) = self
            .legacy_execution
            .as_ref()
            .and_then(|e| e.task_switch.context_for(&self.cpu))
        {
            return Ok(context);
        }
        if self.cpu.task_debug_trap != 0 {
            return Err(self.unsupported_legacy(UnsupportedSystemOperation::LegacyTaskDebugTrap));
        }
        let context = derive_legacy_context(model, &self.cpu).map_err(|e| {
            Exit::UnsupportedSystemContext {
                addr: self.cpu.rip,
                operation: UnsupportedSystemOperation::InvalidLegacyContext(e),
            }
        })?;
        if context.paging_enabled || context.cpl != 0 {
            return Err(self.unsupported_legacy(UnsupportedSystemOperation::LegacyEntryFacility));
        }
        // I286 PRM §10.4 p. 10-6: reset may fetch through the retained high CS base.
        // The I8086 system personality remains deferred.
        if !matches!(model.generation, CpuGeneration::I286 | CpuGeneration::I386)
            || (self.cpu.legacy.reset_fetch != 0 && model.generation != CpuGeneration::I286)
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
        // Decision-24: edited state abandons the entire owned operation before
        // entry validation can reject its deliberately unfinished segment caches.
        // A zero-budget probe leaves continuations and completion inputs intact.
        if limit != Some(0)
            && self
                .legacy_execution
                .as_mut()
                .unwrap()
                .task_switch
                .abandon_if_changed(&self.cpu)
        {
            self.legacy_execution.as_mut().unwrap().instruction_retry = None;
            self.abandon_legacy_event();
            self.cpu.clear_mmio_parts();
            self.cpu.pending_mmio = None;
            self.cpu.pending_mmio_write = false;
            self.cpu.mmio_parts_len = 0;
            self.cpu.mmio_parts_rip = 0;
            self.cpu.mmio_retry_active = false;
            self.cpu.mmio_retry_cs = 0;
            self.apply_pending_a20();
        }
        let context = match self.legacy_context_for_run() {
            Ok(context) => context,
            Err(exit) => {
                return Ok(StepRun {
                    steps: 0,
                    stop: StepStop::Exit(exit),
                })
            }
        };
        if limit == Some(0) {
            return Ok(StepRun {
                steps: 0,
                stop: StepStop::Exit(Exit::BudgetExhausted),
            });
        }
        if self.legacy_execution.as_ref().unwrap().shutdown {
            return Ok(StepRun {
                steps: 0,
                stop: StepStop::Exit(Exit::Shutdown { addr: self.cpu.rip }),
            });
        }
        // Decision-16 §§1,3: validate CS:RIP ownership before execution,
        // while leaving every completion slot untouched at zero budget.
        if self.cpu.mmio_retry_active && !self.cpu.owns_mmio_retry(self.cpu.rip) {
            self.cpu.clear_mmio_parts();
        }
        let execution = self.legacy_execution.as_mut().unwrap();
        if !self.cpu.mmio_retry_active && execution.task_switch.context().is_none() {
            execution.instruction_retry = None;
        }
        if execution.event_retry.is_some_and(|event| {
            (u64::from(event.ip) != self.cpu.rip || event.cs != self.cpu.cs)
                && execution.task_switch.context().is_none()
        }) {
            self.release_legacy_event_boundary();
            self.apply_pending_a20();
        }
        if stepped && self.cpu.pending_port_in.is_some() {
            return Err(StepRunError::PendingPortIn);
        }
        self.interruption_evidence = None;
        self.syscall_evidence = None;
        if context.segment_class == crate::SegmentClass::Direct {
            // Direct has no task continuation and shared blocks cannot start a
            // task switch. A Checked transition back starts a fresh execute()
            // attempt, which also clears these stale completion flags.
            let task_switch = &mut self.legacy_execution.as_mut().unwrap().task_switch;
            task_switch.committed = false;
            task_switch.completed = false;
        }
        self.legacy_execution.as_mut().unwrap().sync_class(
            &self.cpu,
            self.legacy_model.unwrap(),
            false,
            context.segment_class,
        );
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
            (true, MemConsistency::Fast) => {
                cpu.legacy_loop::<true, 0>(vm, limit, options, stepped, context)
            }
            (false, MemConsistency::Fast) => {
                cpu.legacy_loop::<false, 0>(vm, limit, options, stepped, context)
            }
            (true, MemConsistency::AcqRel) => {
                cpu.legacy_loop::<true, 1>(vm, limit, options, stepped, context)
            }
            (false, MemConsistency::AcqRel) => {
                cpu.legacy_loop::<false, 1>(vm, limit, options, stepped, context)
            }
            (true, MemConsistency::FullTso) => {
                cpu.legacy_loop::<true, 2>(vm, limit, options, stepped, context)
            }
            (false, MemConsistency::FullTso) => {
                cpu.legacy_loop::<false, 2>(vm, limit, options, stepped, context)
            }
        };
        drop(guard);
        if let StepStop::Exit(exit @ (Exit::UnmappedMemory { .. } | Exit::Exception { .. })) =
            result.stop
        {
            result.stop = StepStop::Exit(cpu.fault_entry.finish_transfer_fault(exit, &vm.mem));
        }
        // The Machine bus handoff exposes only this transfer's bytes, including
        // stores whose source register or split operand contains higher bits.
        if let StepStop::Exit(Exit::MmioWrite { size, value, .. }) = &mut result.stop {
            *value &= u64::MAX >> (8 * (8 - *size));
        }
        cpu.cpu.finish_mmio_retry(match &result.stop {
            StepStop::Exit(exit) => Some(exit),
            StepStop::IrqReady => None,
        });
        if !cpu.cpu.mmio_retry_active
            && cpu
                .legacy_execution
                .as_ref()
                .unwrap()
                .task_switch
                .context()
                .is_none()
        {
            cpu.legacy_execution.as_mut().unwrap().instruction_retry = None;
        }
        cpu.apply_pending_a20();
        if let StepStop::Exit(exit) = &result.stop {
            // Retry bookkeeping above is engine-owned. Seal the state actually
            // returned to the embedder, allowing only this exit's completion.
            cpu.legacy_execution
                .as_mut()
                .unwrap()
                .task_switch
                .seal_exit(&cpu.cpu, exit);
        }
        drop(dispatch);
        self.time_outside = time_run.finish(match &result.stop {
            StepStop::Exit(exit) => Some(exit),
            StepStop::IrqReady => None,
        });
        Ok(result)
    }

    fn deliver_legacy_event<const ROUTED: bool, const ORDER: u8>(
        &mut self,
        vm: &Vm,
        mut context: MachineContext,
        mut event: PendingEvent,
    ) -> Result<(), Exit> {
        // Decision-23 §5: processor exceptions can be exposed without delivery.
        if context.regime == MachineRegime::Protected
            && self.exception_delivery == crate::ExceptionDelivery::Outward
            && matches!(event.source, EventSource::SingleStep)
        {
            self.release_legacy_event_boundary();
            let mut metadata = crate::fault::FaultMetadata::architectural_exception(
                self.cpu.rip,
                u64::from(event.ip),
                1,
                crate::fault::FaultClass::Trap,
                crate::fault::ErrorCode::NoCode,
            );
            metadata.path = crate::fault::FaultPath::Interpreter;
            return Err(Exit::exception(self.cpu.rip, 1).with_fault_metadata(metadata));
        }
        // Event priority, highest first (debug-register events remain unimplemented):
        // I286 PRM §5.2.1.1, Table 5-1 p. 5-4: instruction exception,
        // single step, NMI, processor-extension segment overrun, INTR.
        // 80386 PRM §9.3, Table 9-2 p. 9-4: non-debug faults, taken
        // INTO/INT n/INT 3, debug traps for this instruction, debug faults
        // for the next instruction, NMI, INTR. Faulting attempts never reach
        // this post-instruction TF delivery; taken software INT suppresses it.
        // I286 PRM §9.7.1 p. 9-14; 80386 PRM §12.3.1.4 p. 12-8:
        // finish single-step delivery before examining lower-priority external
        // interrupts, which may be serviced before the first handler instruction.
        // Interrupt gates clear IF; trap gates retain its prior value.
        // NMI input/delivery is deferred. A future NMI path must check eligible
        // NMIs after this delivery and before handler execution, independently
        // of IF (80386 PRM §9.2.2 p. 9-3);
        // MOV/POP SS inhibit its load boundary (I286 PRM Appendix B MOV
        // p. B-73, POP p. B-83; 80386 PRM §9.2.4 pp. 9-3–4).
        loop {
            let (vector, error_code, fault) = match event.source {
                EventSource::SingleStep => (1, None, false),
                EventSource::Irq(vector) => (vector, None, false),
                EventSource::Exception { vector, error_code } => (vector, error_code, true),
            };
            let delivery = crate::legacy::LegacyEvent {
                vector,
                saved_ip: event.ip,
                software: false,
                external: true,
                error_code,
                fault,
            };
            match crate::interp::legacy::deliver_interrupt::<ROUTED, ORDER>(
                &mut self.cpu,
                &vm.mem,
                context,
                delivery,
                &mut self.legacy_execution.as_mut().unwrap().task_switch,
            ) {
                Ok(()) => break,
                Err(exit) => {
                    let committed = self
                        .legacy_execution
                        .as_ref()
                        .unwrap()
                        .task_switch
                        .committed;
                    if committed {
                        // Commit transfers ownership to the task continuation.
                        // A fresh IRQ queued during a later handoff stays queued.
                        self.consume_legacy_irq(&mut event);
                        context = crate::interp::legacy::tasks::fault_context(
                            &self.cpu,
                            context.generation,
                        );
                        event.ip = self.cpu.rip as u32;
                        event.cs = self.cpu.cs;
                    }
                    self.cpu.finish_mmio_retry(Some(&exit));
                    // An accepted event owns its continuation through every
                    // bus handoff and engine refusal, until completion or fault.
                    let execution = self.legacy_execution.as_mut().unwrap();
                    execution.event_retry = (execution.task_switch.context().is_some()
                        || matches!(
                            &exit,
                            Exit::MmioRead { .. }
                                | Exit::MmioWrite { .. }
                                | Exit::UnmappedMemory { .. }
                        ))
                    .then_some(event);
                    let completed = execution.task_switch.completed;
                    if execution.event_retry.is_none() {
                        // A nonresumable exit releases the abandoned boundary,
                        // including inhibition inherited from STI/MOV/POP SS.
                        self.release_legacy_event_boundary();
                    }
                    if completed {
                        self.finish_legacy_event(event, Some(&exit));
                        return Err(exit);
                    }
                    if !matches!(exit, Exit::Exception { .. }) {
                        // Accepted delivery owns its inputs through bus handoffs.
                        return Err(exit);
                    }
                    self.apply_pending_a20();
                    if context.regime == MachineRegime::Real
                        && self.exception_delivery == crate::ExceptionDelivery::Outward
                    {
                        // Real-address Outward contract: the fault is returned and a
                        // pending IRQ stays queued for the next IF check.
                        return Err(exit);
                    }
                    self.consume_legacy_irq(&mut event);
                    if self.exception_delivery == crate::ExceptionDelivery::Outward {
                        return Err(exit);
                    }
                    let next = exception_source(&exit).expect("architectural exception");
                    // I286 PRM §9.6.2 p. 9-10; I386 PRM §9.8.8,
                    // Tables 9-3/9-4 p. 9-13 (scan checked): delivery of #DF
                    // that faults stops the processor until reset.
                    if !committed
                        && matches!(event.source, EventSource::Exception { vector: 8, .. })
                    {
                        self.legacy_execution.as_mut().unwrap().shutdown = true;
                        return Err(Exit::Shutdown { addr: self.cpu.rip });
                    }
                    // 286 PRM §9.6.4 p. 9-11; 386 PRM §7.5 p. 7-8:
                    // descriptor faults after a task switch start a new task's
                    // exception, rather than combining with the old task's event.
                    event.source =
                        if !committed && double_fault(context.generation, event.source, next) {
                            EventSource::Exception {
                                vector: 8,
                                error_code: Some(0),
                            }
                        } else {
                            next
                        };
                }
            }
        }
        self.finish_legacy_event(event, None);
        Ok(())
    }

    fn consume_legacy_irq(&mut self, event: &mut PendingEvent) {
        if !event.irq_consumed {
            if let EventSource::Irq(vector) = event.source {
                if self.pending_irq == Some(vector) {
                    self.pending_irq = None;
                }
                event.irq_consumed = true;
            }
        }
    }

    fn release_legacy_event_boundary(&mut self) {
        // Decision-24: an abandoned event owns no future inhibition or trap.
        // Clear private boundary state; architectural TF belongs to CpuState.
        let execution = self.legacy_execution.as_mut().unwrap();
        execution.event_retry = None;
        execution.port_debug = false;
        self.sti_shadow = false;
        self.ss_shadow = false;
    }

    fn abandon_legacy_event(&mut self) {
        if let Some(mut event) = self.legacy_execution.as_mut().unwrap().event_retry.take() {
            self.consume_legacy_irq(&mut event);
        }
        self.release_legacy_event_boundary();
    }

    fn finish_legacy_event(&mut self, mut event: PendingEvent, exit: Option<&Exit>) {
        self.consume_legacy_irq(&mut event);
        self.release_legacy_event_boundary();
        self.cpu.finish_mmio_retry(exit);
        self.apply_pending_a20();
        let class = self
            .legacy_context_for_run()
            .map_or(crate::SegmentClass::Checked, |context| {
                context.segment_class
            });
        self.legacy_execution.as_mut().unwrap().sync_class(
            &self.cpu,
            self.legacy_model.unwrap(),
            true,
            class,
        );
    }

    fn route_legacy_exception<const ROUTED: bool, const ORDER: u8>(
        &mut self,
        vm: &Vm,
        context: MachineContext,
        exit: Exit,
    ) -> Result<(), Exit> {
        if self.exception_delivery != crate::ExceptionDelivery::InGuest {
            return Err(exit);
        }
        let Some(source) = exception_source(&exit) else {
            return Err(exit);
        };
        self.cpu.finish_mmio_retry(Some(&exit));
        self.legacy_execution.as_mut().unwrap().instruction_retry = None;
        self.apply_pending_a20();
        self.deliver_legacy_event::<ROUTED, ORDER>(
            vm,
            context,
            PendingEvent {
                irq_consumed: false,
                source,
                ip: exit
                    .fault_metadata()
                    .map_or(self.cpu.rip, |m| m.saved_rip()) as u32,
                cs: self.cpu.cs,
            },
        )
    }

    fn legacy_loop<const ROUTED: bool, const ORDER: u8>(
        &mut self,
        vm: &Vm,
        limit: Option<u64>,
        options: StepOptions,
        _stepped: bool,
        mut context: MachineContext,
    ) -> StepRun {
        let model = self.legacy_model.unwrap();
        let mut steps = 0;
        let mut direct_bus = raw_direct(&self.cpu);
        let exit_word_owner = self.exit_request.word.clone();
        let exit_word = &*exit_word_owner;
        loop {
            // Task state changes only during execution or event delivery; an
            // event restarts this iteration before dispatching its new context.
            let task_pending = self
                .legacy_execution
                .as_ref()
                .unwrap()
                .task_switch_pending();
            let stop = if limit.is_some_and(|limit| steps >= limit) {
                Some(Exit::BudgetExhausted)
            } else if !self.cpu.mmio_retry_active
                && !task_pending
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
            if self.cpu.port_string_in.is_some() {
                if let Err(exit) = crate::interp::legacy::finish_port_input::<ROUTED, ORDER>(
                    &mut self.cpu,
                    &vm.mem,
                ) {
                    return StepRun {
                        steps,
                        stop: StepStop::Exit(exit),
                    };
                }
                self.sti_shadow = false;
                self.ss_shadow = false;
                if self.cpu.mmio_retry_active {
                    self.cpu.finish_mmio_retry(None);
                }
                if self
                    .legacy_execution
                    .as_ref()
                    .unwrap()
                    .pending_a20_enabled
                    .is_some()
                {
                    self.apply_pending_a20();
                    direct_bus = raw_direct(&self.cpu);
                }
            }
            if !task_pending
                && self.legacy_execution.as_ref().unwrap().port_debug
                && self.cpu.pending_port_in.is_none()
                && self.cpu.port_string_in.is_none()
            {
                self.legacy_execution.as_mut().unwrap().port_debug = false;
                self.legacy_execution.as_mut().unwrap().event_retry = Some(PendingEvent {
                    irq_consumed: false,
                    source: EventSource::SingleStep,
                    ip: self.cpu.rip as u32,
                    cs: self.cpu.cs,
                });
            }
            let event = self
                .legacy_execution
                .as_ref()
                .unwrap()
                .event_retry
                .or_else(|| {
                    self.pending_irq
                        .filter(|_| {
                            !task_pending
                                && self.irq_delivery_ready()
                                && !self.cpu.mmio_retry_active
                        })
                        .map(|vector| PendingEvent {
                            irq_consumed: false,
                            source: EventSource::Irq(vector),
                            ip: self.cpu.rip as u32,
                            cs: self.cpu.cs,
                        })
                });
            if let Some(event) = event {
                if let Err(exit) = self.deliver_legacy_event::<ROUTED, ORDER>(vm, context, event) {
                    return StepRun {
                        steps,
                        stop: StepStop::Exit(exit),
                    };
                }
                // Delivery costs no instruction unit. Re-derive after the CS reload.
                context = match self.legacy_context_for_run() {
                    Ok(context) => context,
                    Err(exit) => {
                        return StepRun {
                            steps,
                            stop: StepStop::Exit(exit),
                        }
                    }
                };
                direct_bus = raw_direct(&self.cpu);
                continue;
            }
            if options.stop_when_irq_ready
                && !task_pending
                && !self.cpu.mmio_retry_active
                && self.irq_delivery_ready()
            {
                return StepRun {
                    steps,
                    stop: StepStop::IrqReady,
                };
            }
            let direct = !task_pending && context.segment_class == crate::SegmentClass::Direct;
            let linear_fetch = (u64::from(self.cpu.cs) << 4) + self.cpu.rip;
            let physical = if direct && direct_bus {
                linear_fetch
            } else {
                crate::legacy::physical_span(&self.cpu, linear_fetch, 1).0
            };
            let page = physical >> crate::memory::CODE_PAGE_BITS;
            let dirty = exit_word.load(Ordering::Relaxed) & crate::memory::DISPATCH_DIRTY != 0;
            let dropped = dirty.then(|| vm.cache.invalidated_blocks());
            vm.handle_smc(exit_word);
            let execution = self.legacy_execution.as_mut().unwrap();
            if dropped.is_some_and(|before| before != vm.cache.invalidated_blocks()) {
                if let Some(last) = execution.last_direct_fetch {
                    // This bounded page hint is only a lowering choice. Precise
                    // source invalidation and byte revalidation remain authoritative.
                    for page in [last >> crate::memory::CODE_PAGE_BITS, page] {
                        if !execution.smc_pages.contains(&page) {
                            if execution.smc_pages.len() == 16 {
                                execution.smc_pages.remove(0);
                            }
                            execution.smc_pages.push(page);
                        }
                    }
                }
            }
            let retained = direct && (!direct_bus || execution.smc_pages.contains(&page));
            let resolved = if direct
                && !retained
                && self
                    .legacy_execution
                    .as_ref()
                    .unwrap()
                    .instruction_retry
                    .is_none()
            {
                resolve_direct(vm, &self.cpu, context, exit_word, physical)
            } else {
                let execution = self.legacy_execution.as_mut().unwrap();
                if execution.context_id == u64::MAX {
                    execution.sync_class(&self.cpu, model, true, crate::SegmentClass::Checked);
                }
                execution.resolve(
                    vm,
                    &self.cpu,
                    model,
                    MachineContext {
                        // The masked bus lowering retains the Direct proof;
                        // SMC uses the cheaper original Checked instruction IR.
                        segment_class: if direct && !direct_bus {
                            crate::SegmentClass::Direct
                        } else {
                            crate::SegmentClass::Checked
                        },
                        ..context
                    },
                    exit_word,
                )
            };
            self.legacy_execution.as_mut().unwrap().last_direct_fetch =
                (direct && !retained).then_some(linear_fetch);
            let ir = match resolved {
                Ok(ir) => ir,
                Err(exit) => {
                    match self.route_legacy_exception::<ROUTED, ORDER>(vm, context, exit) {
                        Ok(()) => {
                            context = match self.legacy_context_for_run() {
                                Ok(context) => context,
                                Err(exit) => {
                                    return StepRun {
                                        steps,
                                        stop: StepStop::Exit(exit),
                                    }
                                }
                            };
                            direct_bus = raw_direct(&self.cpu);
                            continue;
                        }
                        Err(exit) => {
                            return StepRun {
                                steps,
                                stop: StepStop::Exit(exit),
                            }
                        }
                    }
                }
            };
            // I286 PRM §9.7.1 p. 9-14; 80386 PRM §12.3.1.4 p. 12-8:
            // sample TF before execution, including POPF/IRET that change it.
            let single_step = if task_pending {
                self.legacy_execution
                    .as_ref()
                    .unwrap()
                    .task_switch
                    .single_step()
                    .unwrap_or(self.cpu.rflags_system & 0x100 != 0)
            } else {
                self.cpu.rflags_system & 0x100 != 0
            };
            let progress = if direct
                && ir
                    .legacy_context
                    .is_some_and(|c| c.segment_class == crate::SegmentClass::Direct)
            {
                let remaining = limit.map_or(u64::MAX, |limit| limit - steps);
                let remaining = if single_step
                    || self.sti_shadow
                    || self.ss_shadow
                    || self.pending_irq.is_some()
                    || self
                        .legacy_execution
                        .as_ref()
                        .unwrap()
                        .pending_a20_enabled
                        .is_some()
                    || options.stop_when_irq_ready
                {
                    1
                } else {
                    remaining
                };
                if matches!(
                    ir.ops.last(),
                    Some(IrOp::Legacy(crate::legacy::LegacyOp::SharedWalk))
                ) {
                    let shared = crate::interp::legacy::execute_shared::<ROUTED, ORDER>(
                        &ir,
                        &mut self.cpu,
                        &vm.mem,
                        &mut self.interp_scratch,
                        remaining,
                        exit_word,
                        self.sti_shadow,
                        self.ss_shadow,
                        self.pending_irq.is_some(),
                    );
                    if !single_step
                        && matches!(
                            shared.result,
                            crate::interp::BlockResult::Continue
                                | crate::interp::BlockResult::ModeChange
                        )
                    {
                        steps += shared.steps;
                        self.retired += shared.retired;
                        self.executed += shared.retired;
                        self.sti_shadow = shared.sti_shadow;
                        self.ss_shadow = shared.ss_shadow;
                        self.legacy_execution.as_mut().unwrap().instruction_retry = None;
                        if self
                            .legacy_execution
                            .as_ref()
                            .unwrap()
                            .pending_a20_enabled
                            .is_some()
                        {
                            self.apply_pending_a20();
                            direct_bus = raw_direct(&self.cpu);
                        }
                        continue;
                    }
                    shared.into_direct()
                } else {
                    crate::interp::legacy::execute_direct::<ROUTED, ORDER>(
                        &ir,
                        &mut self.cpu,
                        &vm.mem,
                        &mut self.interp_scratch,
                        remaining,
                        exit_word,
                        &mut self.legacy_execution.as_mut().unwrap().task_switch,
                    )
                }
            } else {
                let outcome = crate::interp::legacy::execute::<ROUTED, ORDER>(
                    &ir,
                    &mut self.cpu,
                    &vm.mem,
                    &mut self.interp_scratch,
                    &mut self.legacy_execution.as_mut().unwrap().task_switch,
                );
                crate::interp::legacy::DirectProgress {
                    steps: u64::from(outcome.is_ok()),
                    retired: u64::from(outcome.as_ref().is_ok_and(|outcome| outcome.retired)),
                    outcome,
                }
            };
            steps += progress.steps;
            self.retired += progress.retired;
            self.executed += progress.retired;
            match progress.outcome {
                Ok(outcome) => {
                    let arm_ss = outcome.arm_ss && !self.ss_shadow;
                    if self.cpu.port_string_in.is_none() {
                        self.sti_shadow = outcome.arm_sti;
                        self.ss_shadow = arm_ss;
                    }
                    if self.cpu.mmio_retry_active {
                        self.cpu.finish_mmio_retry(None);
                    }
                    // SDM Vol 2B REP, p. 4-564: a completed iteration is an
                    // interruption window with updated registers (decision-22).
                    self.legacy_execution.as_mut().unwrap().instruction_retry = None;
                    if self
                        .legacy_execution
                        .as_ref()
                        .unwrap()
                        .pending_a20_enabled
                        .is_some()
                    {
                        self.apply_pending_a20();
                        direct_bus = raw_direct(&self.cpu);
                    }
                    if outcome.context_changed {
                        let execution = self.legacy_execution.as_mut().unwrap();
                        if outcome.translations_invalidated {
                            execution.invalidate_translations();
                        }
                        context = match self.legacy_context_for_run() {
                            Ok(context) => context,
                            Err(exit) => {
                                return StepRun {
                                    steps,
                                    stop: StepStop::Exit(exit),
                                }
                            }
                        };
                        direct_bus = raw_direct(&self.cpu);
                        self.legacy_execution.as_mut().unwrap().sync_class(
                            &self.cpu,
                            model,
                            true,
                            context.segment_class,
                        );
                    }
                    // I286 PRM Appendix B MOV p. B-73, POP p. B-83;
                    // 80386 PRM §9.2.4 pp. 9-3–4, §17 MOV p. 17-109,
                    // POP p. 17-126: only MOV/POP SS suppress this boundary.
                    // SDM Vol 3A §7.8.3 pp. 7-8–9: consecutive SS loads
                    // guarantee inhibition only for the first load; LSS has none.
                    let debug = single_step && !arm_ss && !outcome.suppress_debug;
                    if debug {
                        if matches!(outcome.exit, Some(Exit::PortIo { .. })) {
                            self.legacy_execution.as_mut().unwrap().port_debug = true;
                        } else {
                            let event = PendingEvent {
                                irq_consumed: false,
                                source: EventSource::SingleStep,
                                ip: self.cpu.rip as u32,
                                cs: self.cpu.cs,
                            };
                            let delivered = self.legacy_context_for_run().and_then(|context| {
                                self.deliver_legacy_event::<ROUTED, ORDER>(vm, context, event)
                            });
                            if let Err(exit) = delivered {
                                return StepRun {
                                    steps,
                                    stop: StepStop::Exit(exit),
                                };
                            }
                            context = match self.legacy_context_for_run() {
                                Ok(context) => context,
                                Err(exit) => {
                                    return StepRun {
                                        steps,
                                        stop: StepStop::Exit(exit),
                                    }
                                }
                            };
                            direct_bus = raw_direct(&self.cpu);
                            // HLT completed; the enabled debug trap resumes at
                            // its successor (I286 PRM Appendix B p. B-42;
                            // 80386 PRM §17 HLT p. 17-60).
                            continue;
                        }
                    }
                    if let Some(exit) = outcome.exit {
                        return StepRun {
                            steps,
                            stop: StepStop::Exit(exit),
                        };
                    }
                }
                Err(exit) => {
                    // An architectural fault abandons this attempt before it is
                    // returned to the embedder; only resumable transfers keep IR.
                    self.cpu.finish_mmio_retry(Some(&exit));
                    if self.cpu.mmio_retry_active
                        || self
                            .legacy_execution
                            .as_ref()
                            .unwrap()
                            .task_switch
                            .context()
                            .is_some()
                    {
                        self.legacy_execution.as_mut().unwrap().instruction_retry =
                            Some(InstructionRetry {
                                stamp: LegacyContextStamp::new(&self.cpu, model),
                                ip: self.cpu.rip,
                                mxcsr_mode: self.cpu.mxcsr & crate::jit_abi::fp::MODE_MASK,
                                ir: if direct {
                                    retry_instruction(&ir, self.cpu.rip)
                                } else {
                                    ir
                                },
                            });
                    } else {
                        self.legacy_execution.as_mut().unwrap().instruction_retry = None;
                        if self
                            .legacy_execution
                            .as_ref()
                            .unwrap()
                            .pending_a20_enabled
                            .is_some()
                        {
                            self.apply_pending_a20();
                        }
                    }
                    if matches!(exit, Exit::Exception { .. }) {
                        let fault_context = if self
                            .legacy_execution
                            .as_ref()
                            .unwrap()
                            .task_switch
                            .committed
                        {
                            crate::interp::legacy::tasks::fault_context(
                                &self.cpu,
                                context.generation,
                            )
                        } else {
                            context
                        };
                        match self.route_legacy_exception::<ROUTED, ORDER>(vm, fault_context, exit)
                        {
                            Ok(()) => {
                                context = match self.legacy_context_for_run() {
                                    Ok(context) => context,
                                    Err(exit) => {
                                        return StepRun {
                                            steps,
                                            stop: StepStop::Exit(exit),
                                        }
                                    }
                                };
                                direct_bus = raw_direct(&self.cpu);
                                continue;
                            }
                            Err(exit) => {
                                return StepRun {
                                    steps,
                                    stop: StepStop::Exit(exit),
                                }
                            }
                        }
                    }
                    return StepRun {
                        steps,
                        stop: StepStop::Exit(exit),
                    };
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(deprecated)] // Compatibility selector witnesses.
mod tests {
    use super::*;

    #[test]
    fn redirected_event_discards_every_private_boundary_before_fetch() {
        for model in [crate::CpuModel::I80286, crate::CpuModel::I80386] {
            let mut vm = Vm::new(VmConfig::flat(0x10000));
            vm.set_cpu_model(model);
            vm.map(0, 0x2000, crate::Prot::RWX, crate::RegionKind::Ram)
                .unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu = CpuState::legacy_real_entry([0; 6]);
            cpu.cpu.rip = 0x3000;
            cpu.cpu.flags.if_ = true;
            cpu.sti_shadow = true;
            cpu.ss_shadow = true;
            let execution = cpu.legacy_execution.as_mut().unwrap();
            execution.port_debug = true;
            execution.event_retry = Some(PendingEvent {
                irq_consumed: false,
                source: EventSource::SingleStep,
                ip: 0x1000,
                cs: 0,
            });
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 0, "{:?}", run.stop());
            assert!(
                matches!(
                    run.stop(),
                    StepStop::Exit(Exit::UnmappedMemory { addr: 0x3000, .. })
                ),
                "{:?}",
                run.stop()
            );
            assert!(!cpu.sti_shadow);
            assert!(!cpu.ss_shadow);
            let execution = cpu.legacy_execution.as_ref().unwrap();
            assert!(!execution.port_debug);
            assert!(execution.event_retry.is_none());
            assert!(cpu.irq_ready());
        }
    }

    #[test]
    fn shared_direct_fault_does_not_inherit_a_completed_task_commit() {
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        vm.set_cpu_model(crate::CpuModel::I80386);
        vm.set_exception_delivery(crate::ExceptionDelivery::InGuest);
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, b"\xa3\xff\xff").unwrap();
        vm.write_bytes(13 * 4, &[0x00, 0x20, 0, 0]).unwrap();
        vm.write_bytes(0x2000, &[0xf4]).unwrap();
        let mut cpu = vm.new_vcpu();
        // An embedder may return to ordinary real mode after a completed switch.
        cpu.cpu = CpuState::legacy_real_entry([0; 6]);
        cpu.cpu.rip = 0x1000;
        cpu.cpu.gpr[4] = 0x9000;
        let task = &mut cpu.legacy_execution.as_mut().unwrap().task_switch;
        task.committed = true;
        task.completed = true;
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(matches!(run.stop(), StepStop::Exit(Exit::Hlt)), "{run:?}");
        assert_eq!(run.steps(), 1);
        assert_eq!(cpu.cpu.rip, 0x2001);
        assert_eq!(vm.mem.read(0x8ffa, 2).unwrap(), 0x1000);
    }

    #[test]
    fn direct_publication_rejects_a_source_write_before_registration() {
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        vm.set_cpu_model(crate::CpuModel::I80386);
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, b"\xb8\x01\x00\xf4").unwrap();
        *vm.source_read_hook.lock().unwrap() = Some(|vm| {
            vm.write_bytes(0x1001, b"\x02").unwrap();
        });
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x1000;
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 1);
        assert_eq!(cpu.cpu.gpr[0], 2, "a source write must cancel publication");
    }

    #[test]
    fn smc_retains_instruction_contexts_but_nearby_data_keeps_shared_blocks() {
        for (code, smc) in [
            (&b"\xc6\x06\x06\x10\x2a\xb0\x00\xeb\xf7"[..], true),
            (&b"\xa3\x80\x10\xeb\xfb"[..], false),
        ] {
            let mut vm = Vm::new(VmConfig::flat(0x10000));
            vm.set_cpu_model(crate::CpuModel::I80386);
            vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0x1000, code).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu.rip = 0x1000;
            for _ in 0..2 {
                assert_eq!(
                    cpu.run_steps(&vm, 60, StepOptions::default())
                        .unwrap()
                        .steps(),
                    60
                );
                let execution = cpu.legacy_execution.as_ref().unwrap();
                assert_eq!(execution.smc_pages.is_empty(), !smc);
                assert_eq!(execution.lifts, if smc { 3 } else { 0 });
                assert_eq!(execution.blocks.len(), if smc { 3 } else { 0 });
                if smc {
                    assert_eq!(cpu.cpu.gpr[0] & 0xff, 42);
                }
            }
            cpu.reset_execution_state(crate::ResetOptions::default());
            assert!(cpu.legacy_execution.as_ref().unwrap().smc_pages.is_empty());
        }
    }

    #[test]
    fn double_fault_classes_follow_each_generation() {
        // I286 PRM §9.6.2 p. 9-10; I386 PRM §9.8.8 Tables 9-3/9-4 p. 9-13.
        let exception = |vector| EventSource::Exception {
            vector,
            error_code: None,
        };
        let benign = [1, 3, 4, 5, 6, 7, 16];
        for first in benign {
            for second in [0, 9, 10, 11, 12, 13, 14] {
                for generation in [CpuGeneration::I286, CpuGeneration::I386] {
                    assert!(!double_fault(
                        generation,
                        exception(first),
                        exception(second)
                    ));
                }
            }
        }
        for first in [0, 10, 11, 12, 13] {
            for second in [0, 10, 11, 12, 13] {
                for generation in [CpuGeneration::I286, CpuGeneration::I386] {
                    assert!(double_fault(
                        generation,
                        exception(first),
                        exception(second)
                    ));
                }
            }
        }
        assert!(!double_fault(
            CpuGeneration::I286,
            exception(9),
            exception(13)
        ));
        assert!(double_fault(
            CpuGeneration::I386,
            exception(9),
            exception(13)
        ));
        assert!(!double_fault(
            CpuGeneration::I386,
            exception(13),
            exception(14)
        ));
        for second in [0, 9, 10, 11, 12, 13, 14] {
            assert!(double_fault(
                CpuGeneration::I386,
                exception(14),
                exception(second)
            ));
        }
        for first in [EventSource::SingleStep, EventSource::Irq(0)] {
            assert!(!double_fault(CpuGeneration::I386, first, exception(13)));
        }
    }

    #[test]
    fn a20_toggles_reuse_shared_and_retained_physical_blocks() {
        let (vm, mut cpu) = a20_loop();
        let mut originals = Vec::new();
        for enabled in [true, false, true, false] {
            cpu.set_a20_enabled(enabled);
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1);
            let block = if enabled {
                let CachedBlock::Interpreted(ir) = vm
                    .cache
                    .get(BlockKey::machine_direct(0x100000, 0, true))
                    .unwrap()
                else {
                    unreachable!()
                };
                ir
            } else {
                cpu.legacy_execution
                    .as_ref()
                    .unwrap()
                    .blocks
                    .values()
                    .next()
                    .unwrap()
                    .ir
                    .clone()
            };
            if originals.len() < 2 {
                originals.push(block);
            } else {
                assert!(Arc::ptr_eq(&block, &originals[usize::from(!enabled)]));
            }
        }
        let execution = cpu.legacy_execution.as_ref().unwrap();
        assert_eq!(execution.blocks.len(), 1);
        assert_eq!(execution.lifts, 1);
    }

    #[test]
    fn checked_a20_toggles_reuse_only_the_matching_retained_blocks() {
        let (vm, mut cpu) = a20_loop();
        cpu.cpu.legacy.segments[SegmentId::Ds as usize].limit = 0xfffe;
        let mut originals = Vec::new();
        for enabled in [true, false, true, false] {
            cpu.set_a20_enabled(enabled);
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1);
            let execution = cpu.legacy_execution.as_ref().unwrap();
            let block = execution
                .blocks
                .iter()
                .find(|(key, _)| key.context_id == execution.context_id)
                .unwrap()
                .1
                .ir
                .clone();
            if originals.len() < 2 {
                originals.push(block);
            } else {
                assert!(Arc::ptr_eq(&block, &originals[usize::from(!enabled)]));
            }
        }
        let execution = cpu.legacy_execution.as_ref().unwrap();
        assert_eq!(execution.contexts.len(), 2);
        assert_eq!(execution.blocks.len(), 2);
        assert_eq!(execution.lifts, 2);
    }

    fn a20_loop() -> (Vm, Vcpu) {
        let mut vm = Vm::new(VmConfig::flat(0x200000));
        vm.set_legacy_cpu_model(LegacyCpuModel {
            generation: CpuGeneration::I386,
            model_semantics_id: 1,
        });
        vm.map(0, 0x200000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0, b"\xeb\xfe").unwrap();
        vm.write_bytes(0x100000, b"\xeb\xfe").unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu = CpuState::legacy_real_entry([0xffff, 0, 0, 0, 0, 0]);
        cpu.cpu.rip = 0x10;
        (vm, cpu)
    }

    #[test]
    #[ignore = "explicit interpreter A20 toggle throughput measurement"]
    fn a20_toggle_throughput() {
        let (vm, mut cpu) = a20_loop();
        for enabled in [false, true] {
            cpu.set_a20_enabled(enabled);
            assert_eq!(
                cpu.run_steps(&vm, 1, StepOptions::default())
                    .unwrap()
                    .steps(),
                1
            );
        }
        for sample in 0..3 {
            let start = std::time::Instant::now();
            for i in 0..200_000 {
                cpu.set_a20_enabled(i & 1 != 0);
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 1);
                assert!(matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)));
            }
            let elapsed = start.elapsed().as_secs_f64();
            let execution = cpu.legacy_execution.as_ref().unwrap();
            assert_eq!(cpu.cpu.rip, 0x10);
            assert_eq!(execution.lifts, 1);
            assert_eq!(execution.blocks.len(), 1);
            println!("sample={sample} toggles=200000 elapsed_seconds={elapsed:.6} toggles_per_second={:.0} ns_per_toggle_and_instruction={:.1} lifts=1 cached_blocks=2", 200_000.0 / elapsed, elapsed * 1e9 / 200_000.0);
        }
        assert_eq!(vm.backend.compile_ns(), 0);
    }

    fn cache_context(
        execution: &mut Execution,
        vm: &Vm,
        cpu: &mut CpuState,
        model: LegacyCpuModel,
        tag: u32,
    ) -> Vec<Arc<IrBlock>> {
        cpu.legacy.segments[SegmentId::Ds as usize].base = tag;
        execution.sync(cpu, model, false);
        let context = derive_legacy_context(model, cpu).unwrap();
        [0x1000, 0x1006]
            .into_iter()
            .map(|ip| {
                cpu.rip = ip;
                execution
                    .resolve(vm, cpu, model, context, &AtomicU64::new(0))
                    .unwrap()
            })
            .collect()
    }

    #[test]
    fn context_churn_bounds_retained_contexts_and_their_blocks() {
        let (vm, mut cpu) = pe_transition_loop();
        let model = cpu.legacy_model.unwrap();
        let mut execution = Execution::new();
        for tag in 0..(MAX_RETAINED_CONTEXTS * 8) {
            cache_context(&mut execution, &vm, &mut cpu.cpu, model, tag as u32);
            let retained = (tag + 1).min(MAX_RETAINED_CONTEXTS);
            assert_eq!(execution.contexts.len(), retained);
            assert_eq!(execution.blocks.len(), retained * 2);
            assert!(execution.blocks.keys().all(|key| execution
                .contexts
                .values()
                .any(|context| context.id == key.context_id)));
        }
        assert!(execution.contexts.keys().all(|stamp| {
            stamp.state.segments[SegmentId::Ds as usize].base >= (MAX_RETAINED_CONTEXTS * 7) as u32
        }));
    }

    #[test]
    fn context_lru_keeps_hot_contexts_and_their_blocks() {
        let (vm, mut cpu) = pe_transition_loop();
        let model = cpu.legacy_model.unwrap();
        let mut execution = Execution::new();
        let hot: Vec<_> = (0..2)
            .map(|tag| cache_context(&mut execution, &vm, &mut cpu.cpu, model, tag))
            .collect();
        for tag in 2..(MAX_RETAINED_CONTEXTS * 8) {
            cache_context(&mut execution, &vm, &mut cpu.cpu, model, tag as u32);
            let lifts = execution.lifts;
            for (tag, original) in hot.iter().enumerate() {
                let blocks = cache_context(&mut execution, &vm, &mut cpu.cpu, model, tag as u32);
                for (ir, original) in blocks.iter().zip(original) {
                    assert!(Arc::ptr_eq(ir, original));
                }
            }
            assert_eq!(execution.lifts, lifts);
            assert!(execution.contexts.len() <= MAX_RETAINED_CONTEXTS);
            assert!(execution.blocks.len() <= MAX_RETAINED_CONTEXTS * 2);
        }
    }

    #[test]
    fn task_register_and_ldtr_separate_retained_context_namespaces() {
        let (vm, mut cpu) = pe_transition_loop();
        let model = cpu.legacy_model.unwrap();
        let mut execution = Execution::new();
        let original = cache_context(&mut execution, &vm, &mut cpu.cpu, model, 0);
        let original_id = execution.context_id;
        let before = cpu.cpu.legacy;
        cpu.cpu.legacy.tr.selector = 24;
        cpu.cpu.legacy.tr.cache.base = 0x4000;
        let task = cache_context(&mut execution, &vm, &mut cpu.cpu, model, 0);
        assert_ne!(execution.context_id, original_id);
        assert!(!Arc::ptr_eq(&original[0], &task[0]));
        let task_id = execution.context_id;
        cpu.cpu.legacy.ldtr.selector = 32;
        cpu.cpu.legacy.ldtr.cache.base = 0x5000;
        let ldt = cache_context(&mut execution, &vm, &mut cpu.cpu, model, 0);
        assert_ne!(execution.context_id, task_id);
        assert!(!Arc::ptr_eq(&task[0], &ldt[0]));
        cpu.cpu.legacy = before;
        let returned = cache_context(&mut execution, &vm, &mut cpu.cpu, model, 0);
        assert_eq!(execution.context_id, original_id);
        assert!(Arc::ptr_eq(&original[0], &returned[0]));
    }

    fn pe_transition_loop() -> (Vm, Vcpu) {
        // SDM Vol 3A §§12.9.1–12.9.2; 80386 PRM §§14.4–14.5:
        // each round enters USE16 with a far JMP and returns with a real far JMP.
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        vm.set_legacy_cpu_model(LegacyCpuModel {
            generation: CpuGeneration::I386,
            model_semantics_id: 1,
        });
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(
            0x1000,
            b"\x66\xb8\x01\x00\x00\x00\x0f\x22\xc0\xea\x00\x11\x08\x00",
        )
        .unwrap();
        vm.write_bytes(0x1100, b"\x31\xc0\x0f\x22\xc0\xea\x00\x10\x00\x00")
            .unwrap();
        vm.write_bytes(0x2008, b"\xff\xff\x00\x00\x00\x9b\x00\x00")
            .unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.legacy.segments[SegmentId::Ds as usize].attributes = 0x92;
        cpu.cpu.rip = 0x1000;
        cpu.cpu.legacy.gdtr.base = 0x2000;
        cpu.cpu.legacy.gdtr.limit = 15;
        (vm, cpu)
    }

    #[test]
    #[ignore = "explicit interpreter transition throughput measurement"]
    fn pe_transition_throughput() {
        let (vm, mut cpu) = pe_transition_loop();
        let warm = cpu.run_steps(&vm, 600, StepOptions::default()).unwrap();
        assert_eq!(warm.steps(), 600);
        for sample in 0..3 {
            let lifts = cpu.legacy_execution.as_ref().unwrap().lifts;
            let begin = std::time::Instant::now();
            let run = cpu.run_steps(&vm, 600_000, StepOptions::default()).unwrap();
            let elapsed = begin.elapsed().as_secs_f64();
            assert_eq!(run.steps(), 600_000, "{:?}", run.stop());
            assert!(matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)));
            assert_eq!(
                (cpu.cpu.rip, cpu.cpu.cs, cpu.cpu.legacy.cr0),
                (0x1000, 0, 0)
            );
            let execution = cpu.legacy_execution.as_ref().unwrap();
            println!("sample={sample} transitions=200000 elapsed_seconds={elapsed:.6} transitions_per_second={:.0} lifts={} cached_blocks={}",
                200_000.0 / elapsed, execution.lifts - lifts, execution.blocks.len());
        }
        assert_eq!(cpu.retired_instructions(), 1_800_600);
        assert_eq!(vm.backend.compile_ns(), 0);
    }
    #[test]
    fn real_delivery_advances_epoch_before_the_handler_can_execute() {
        for irq in [false, true] {
            let model = LegacyCpuModel {
                generation: CpuGeneration::I386,
                model_semantics_id: 1,
            };
            let mut vm = Vm::new(VmConfig::flat(0x10000));
            vm.set_legacy_cpu_model(model);
            vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0x1000, b"\xcd\x20\x43").unwrap();
            vm.write_bytes(0x2080, b"\x00\x02\x00\x01").unwrap();
            vm.write_bytes(0x1200, b"\xf4").unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu.rip = 0x1000;
            cpu.cpu.gpr[4] = 0x8000;
            cpu.cpu.legacy.idtr.base = 0x2000;
            cpu.cpu.flags.if_ = true;
            if irq {
                cpu.inject_irq(0x20);
            }
            cpu.legacy_execution
                .as_mut()
                .unwrap()
                .sync(&cpu.cpu, model, false);
            let before = cpu.legacy_execution.as_ref().unwrap().context_epoch;
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1);
            assert_eq!(
                cpu.legacy_execution.as_ref().unwrap().context_epoch,
                before + 1
            );
            assert_eq!(cpu.cpu.cs, 0x100);
            assert_eq!(cpu.cpu.rip, if irq { 0x201 } else { 0x200 });
            assert_eq!(cpu.cpu.gpr[3], 0);
        }
    }

    #[test]
    fn system_writes_end_the_block_and_advance_only_the_required_epochs() {
        let model = LegacyCpuModel {
            generation: CpuGeneration::I386,
            model_semantics_id: 1,
        };
        for (code, cr0, eax, advances) in [
            (&b"\x0f\x22\xd8"[..], 0, 0, true), // equal CR3 still invalidates
            (&b"\x0f\x22\xd0"[..], 0, 0x1234, false), // live CR2
            (&b"\x0f\x22\xc0"[..], 0, 0x10, true),
            (&b"\x0f\x22\xc0"[..], 0, 0, false),
            (&b"\x0f\x01\xf0"[..], 0x10, 0xe, true),
            (&b"\x0f\x06"[..], 0x18, 0, true),
            (&b"\x0f\x01\x16\x00\x20"[..], 0, 0, true),
            (&b"\x0f\x01\x1e\x00\x20"[..], 0, 0, true),
        ] {
            let mut vm = Vm::new(VmConfig::flat(0x10000));
            vm.set_legacy_cpu_model(model);
            vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0x1000, code).unwrap();
            vm.write_bytes(0x1000 + code.len() as u64, b"\x43").unwrap();
            vm.write_bytes(0x2000, b"\xff\x03\x00\x30\x00\x00").unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu.legacy.segments[SegmentId::Ds as usize].attributes = 0x92;
            cpu.cpu.rip = 0x1000;
            cpu.cpu.legacy.cr0 = cr0;
            cpu.cpu.gpr[0] = eax;
            cpu.legacy_execution
                .as_mut()
                .unwrap()
                .sync(&cpu.cpu, model, false);
            let before = cpu.legacy_execution.as_ref().unwrap().context_epoch;
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1, "{code:02x?}: {:?}", run.stop());
            assert_eq!(cpu.cpu.gpr[3], 0);
            assert_eq!(cpu.cpu.rip, 0x1000 + code.len() as u64);
            assert_eq!(
                cpu.legacy_execution.as_ref().unwrap().context_epoch,
                before + u64::from(advances)
            );
            if code == b"\x0f\x22\xd8" {
                assert!(cpu.legacy_execution.as_ref().unwrap().blocks.is_empty());
            } else {
                assert_eq!(cpu.legacy_execution.as_ref().unwrap().blocks.len(), 1);
            }
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1);
            assert_eq!(cpu.cpu.gpr[3], 1);
        }
    }

    #[test]
    fn epoch_stamp_guards_context_but_not_live_outputs() {
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
        assert_ne!(e.owner_id, Execution::new().owner_id);
    }

    #[test]
    fn ldtr_and_tr_hidden_changes_isolate_retained_instruction_blocks() {
        // Decisions 18 §3 and 20: the full system-register cache participates,
        // even when its visible selector stays unchanged. Returning reuses IR.
        for model in [crate::CpuModel::I80286, crate::CpuModel::I80386] {
            let mut vm = Vm::new(VmConfig::flat(0x10000));
            vm.set_cpu_model(model);
            vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0x1000, b"\xeb\xfe").unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu.rip = 0x1000;
            cpu.cpu.legacy.cr0 = 1;
            let initial = cpu.cpu.legacy;
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1);
            let execution = cpu.legacy_execution.as_ref().unwrap();
            let id = execution.context_id;
            let original = execution.blocks.values().next().unwrap().ir.clone();
            let edits: [fn(&mut LegacyCpuState); 8] = [
                |s| s.ldtr.cache.base = 0x2000,
                |s| s.ldtr.cache.limit = 0xff,
                |s| s.ldtr.cache.attributes = 0x82,
                |s| s.ldtr.cache.unusable = 1,
                |s| s.tr.cache.base = 0x3000,
                |s| s.tr.cache.limit = 0x67,
                |s| s.tr.cache.attributes = 0x83,
                |s| s.tr.cache.unusable = 1,
            ];
            for edit in edits {
                cpu.cpu.legacy = initial;
                edit(&mut cpu.cpu.legacy);
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 1);
                let execution = cpu.legacy_execution.as_ref().unwrap();
                assert_ne!(execution.context_id, id);
                let block = execution
                    .blocks
                    .iter()
                    .find(|(key, _)| key.context_id == execution.context_id)
                    .unwrap()
                    .1;
                assert!(!Arc::ptr_eq(&original, &block.ir));
                cpu.cpu.legacy = initial;
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 1);
                let execution = cpu.legacy_execution.as_ref().unwrap();
                assert_eq!(execution.context_id, id);
                let block = execution
                    .blocks
                    .iter()
                    .find(|(key, _)| key.context_id == id)
                    .unwrap()
                    .1;
                assert!(Arc::ptr_eq(&original, &block.ir));
            }
            assert_eq!(cpu.legacy_execution.as_ref().unwrap().lifts, 9);
        }
    }

    #[test]
    fn epoch_rollover_drops_populated_contexts_before_reusing_ids() {
        let (vm, mut cpu) = pe_transition_loop();
        let model = cpu.legacy_model.unwrap();
        let mut execution = Execution::new();
        let mut original = Vec::new();
        for tag in 0..3 {
            original.push(cache_context(&mut execution, &vm, &mut cpu.cpu, model, tag));
            assert_eq!(execution.context_id, u64::from(tag));
        }
        assert_eq!(execution.contexts.len(), 3);
        assert_eq!(execution.blocks.len(), 6);
        execution.context_epoch = u64::MAX;
        cpu.cpu.legacy.segments[SegmentId::Ds as usize].base = 0;
        execution.sync(&cpu.cpu, model, true);
        assert_eq!(execution.context_epoch, 0);
        assert_eq!(execution.context_id, 0);
        assert_eq!(execution.contexts.len(), 1);
        assert!(execution.blocks.is_empty());
        for (tag, old_blocks) in original.iter().enumerate() {
            let blocks = cache_context(&mut execution, &vm, &mut cpu.cpu, model, tag as u32);
            assert_eq!(execution.context_id, tag as u64);
            for (ir, old) in blocks.iter().zip(old_blocks) {
                assert!(!Arc::ptr_eq(ir, old));
            }
        }
        assert_eq!(execution.blocks.len(), 6);
        assert_eq!(execution.lifts, 12);
    }

    #[test]
    fn pe_transitions_reuse_both_regimes_without_relifting_or_flushing_other_contexts() {
        let (vm, mut cpu) = pe_transition_loop();
        let warm = cpu.run_steps(&vm, 6, StepOptions::default()).unwrap();
        assert_eq!(warm.steps(), 6, "{:?}", warm.stop());
        let execution = cpu.legacy_execution.as_ref().unwrap();
        assert_eq!(execution.contexts.len(), 4);
        assert_eq!(execution.blocks.len(), 6);
        assert_eq!(execution.lifts, 6);
        let blocks: Vec<_> = execution
            .blocks
            .iter()
            .map(|(key, block)| (*key, block.ir.clone()))
            .collect();
        let before_epoch = execution.context_epoch;
        for count in [1, 2, 3, 6, 600] {
            let run = cpu.run_steps(&vm, count, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), count, "{:?}", run.stop());
        }
        let execution = cpu.legacy_execution.as_ref().unwrap();
        assert!(execution.context_epoch > before_epoch);
        assert_eq!(execution.contexts.len(), 4);
        assert_eq!(execution.blocks.len(), 6);
        assert_eq!(execution.lifts, 6);
        for (key, ir) in blocks {
            assert!(Arc::ptr_eq(&ir, &execution.blocks[&key].ir));
        }
    }

    #[test]
    fn direct_protected_round_trips_reuse_both_cache_namespaces() {
        let (vm, mut cpu) = pe_transition_loop();
        cpu.cpu.legacy.segments[SegmentId::Ds as usize].attributes = 0x93;
        let warm = cpu.run_steps(&vm, 6, StepOptions::default()).unwrap();
        assert_eq!(warm.steps(), 6, "{:?}", warm.stop());
        let key = BlockKey::machine_direct(0x1000, 0, true);
        let CachedBlock::Interpreted(direct) = vm.cache.get(key).unwrap() else {
            unreachable!()
        };
        assert_eq!(direct.icount, 1);
        assert_eq!(
            direct.legacy_context.unwrap().segment_class,
            crate::SegmentClass::Direct
        );
        let blocks: Vec<_> = cpu
            .legacy_execution
            .as_ref()
            .unwrap()
            .blocks
            .iter()
            .map(|(key, block)| (*key, block.ir.clone()))
            .collect();
        assert!(!blocks.is_empty());
        let lifts = cpu.legacy_execution.as_ref().unwrap().lifts;
        let run = cpu.run_steps(&vm, 600, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 600, "{:?}", run.stop());
        let CachedBlock::Interpreted(reused) = vm.cache.get(key).unwrap() else {
            unreachable!()
        };
        assert!(Arc::ptr_eq(&direct, &reused));
        let execution = cpu.legacy_execution.as_ref().unwrap();
        assert_eq!(execution.lifts, lifts);
        for (key, ir) in blocks {
            assert!(Arc::ptr_eq(&ir, &execution.blocks[&key].ir));
        }
    }

    #[test]
    fn reused_contexts_revalidate_code_bytes_and_between_run_cache_edits() {
        let (vm, mut cpu) = pe_transition_loop();
        let model = cpu.legacy_model.unwrap();
        let mut execution = Execution::new();
        let mut original = Vec::new();
        for pe in [0, 1] {
            cpu.cpu.legacy.cr0 = pe;
            let context = derive_legacy_context(model, &cpu.cpu).unwrap();
            execution.sync(&cpu.cpu, model, false);
            original.push(
                execution
                    .resolve(&vm, &cpu.cpu, model, context, &AtomicU64::new(0))
                    .unwrap(),
            );
        }
        for pe in [0, 1] {
            cpu.cpu.legacy.cr0 = pe;
            let context = derive_legacy_context(model, &cpu.cpu).unwrap();
            execution.sync(&cpu.cpu, model, false);
            let ir = execution
                .resolve(&vm, &cpu.cpu, model, context, &AtomicU64::new(0))
                .unwrap();
            assert!(Arc::ptr_eq(&ir, &original[pe as usize]));
        }
        vm.write_bytes(0x1002, &2u32.to_le_bytes()).unwrap();
        for pe in [0, 1] {
            cpu.cpu.legacy.cr0 = pe;
            let context = derive_legacy_context(model, &cpu.cpu).unwrap();
            execution.sync(&cpu.cpu, model, false);
            let ir = execution
                .resolve(&vm, &cpu.cpu, model, context, &AtomicU64::new(0))
                .unwrap();
            assert!(!Arc::ptr_eq(&ir, &original[pe as usize]));
        }
        let before_id = execution.context_id;
        cpu.cpu.legacy.segments[SegmentId::Ds as usize].limit = 0x12345;
        execution.sync(&cpu.cpu, model, false);
        assert_ne!(execution.context_id, before_id);
        let context = derive_legacy_context(model, &cpu.cpu).unwrap();
        execution
            .resolve(&vm, &cpu.cpu, model, context, &AtomicU64::new(0))
            .unwrap();
        assert_eq!(execution.blocks.len(), 3);
        cpu.cpu.legacy.segments[SegmentId::Ds as usize].limit = 0xffff;
        execution.sync(&cpu.cpu, model, false);
        assert_eq!(execution.context_id, before_id);
        assert_eq!(execution.lifts, 5);
    }
}
