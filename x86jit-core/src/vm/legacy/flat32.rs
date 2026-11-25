//! Interpreted unpaged Flat32 and Seg16 blocks (decision-27; spec.md §17.10).
use super::*;

impl Execution {
    #[cold]
    #[inline(never)]
    pub(super) fn resolve_flat32(
        &mut self,
        vm: &Vm,
        cpu: &CpuState,
        model: LegacyCpuModel,
        context: MachineContext,
        exit_word: &AtomicU64,
    ) -> Result<(Arc<IrBlock>, BlockWalk), Exit> {
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
        if let Some(retry) = &self.instruction_retry {
            if retry.stamp == LegacyContextStamp::new(cpu, model)
                && retry.ip == cpu.rip
                && retry.mxcsr_mode == key.mxcsr_mode
            {
                return Ok((retry.ir.clone(), retry.walk));
            }
        }
        self.flat32.resolve(vm, cpu, context, key, exit_word)
    }
}

struct Proof {
    ir: Arc<IrBlock>,
    walk: BlockWalk,
    permit: Option<crate::FetchPermit>,
    _registration: crate::memory::CodeRegistration,
}

#[derive(Default)]
pub(super) struct State {
    generation: u64,
    mapping: Option<Arc<()>>,
    blocks: HashMap<LegacyBlockKey, Proof>,
}

impl State {
    pub(super) fn evict_context(&mut self, id: u64) {
        self.blocks.retain(|key, _| key.context_id != id);
    }

    #[cold]
    #[inline(never)]
    pub(super) fn resolve(
        &mut self,
        vm: &Vm,
        cpu: &CpuState,
        context: MachineContext,
        key: LegacyBlockKey,
        exit_word: &AtomicU64,
    ) -> Result<(Arc<IrBlock>, BlockWalk), Exit> {
        loop {
            vm.handle_smc(exit_word);
            let generation = vm.cache.publication_generation();
            let mapping = vm.mem.mapping_identity();
            if generation != self.generation
                || self
                    .mapping
                    .as_ref()
                    .is_none_or(|old| !Arc::ptr_eq(old, &mapping))
            {
                self.blocks.clear();
                self.generation = generation;
                self.mapping = Some(mapping);
            }
            if let Some(proof) = self
                .blocks
                .get(&key)
                .filter(|p| p.permit.as_ref().is_none_or(|p| p.is_current()))
            {
                probe!(PROOFS);
                return Ok((proof.ir.clone(), proof.walk));
            }
            probe!(LIFTS);
            let source = crate::memory::LiftSource::new(&vm.mem);
            let permit = vm.mem.fetch_permit();
            let ir = lift(vm, cpu, context, &source)?;
            if !source.validate() || permit.as_ref().is_some_and(|p| !p.is_current()) {
                continue;
            }
            if !source.publishable() {
                self.blocks.remove(&key);
                let CachedBlock::Interpreted(ir) = source.uncached(ir) else {
                    unreachable!()
                };
                // Dropping the source releases its temporary tags. Refetch
                // after this instruction rather than trusting a stale successor.
                let ir = retry_instruction(&ir, cpu.rip);
                let walk = BlockWalk::for_flat32(&ir);
                return Ok((ir, walk));
            }
            let walk = BlockWalk::for_flat32(&ir);
            #[cfg(test)]
            vm.after_source_read();
            vm.handle_smc(exit_word);
            let mut sources = Vec::new();
            for offset in 0..ir.guest_len {
                let (physical, _) = crate::legacy::physical_span(
                    cpu,
                    u32::MAX,
                    u64::from(
                        cpu.legacy.segments[0]
                            .base
                            .wrapping_add((cpu.rip + u64::from(offset)) as u32),
                    ),
                    1,
                );
                if let Some((start, len)) = sources.last_mut() {
                    if *start + u64::from(*len) == physical {
                        *len += 1;
                        continue;
                    }
                }
                sources.push((physical, 1u32));
            }
            let mut registration = None;
            if !vm.cache.register_private_sources(generation, || {
                registration = Some(vm.mem.register_code_sources(&sources));
            }) || permit.as_ref().is_some_and(|p| !p.is_current())
            {
                continue;
            }
            // Keep retention bounded independently of the context LRU. Each
            // proof owns its counted physical registration until eviction.
            if self.blocks.len() == 64 {
                self.blocks.clear();
            }
            let ir = Arc::new(ir);
            self.blocks.insert(
                key,
                Proof {
                    ir: ir.clone(),
                    walk,
                    permit,
                    _registration: registration.unwrap(),
                },
            );
            return Ok((ir, walk));
        }
    }
}

fn lift(
    vm: &Vm,
    cpu: &CpuState,
    context: MachineContext,
    source: &crate::memory::LiftSource<'_>,
) -> Result<IrBlock, Exit> {
    let mut ip = cpu.rip;
    let mut block: Option<IrBlock> = None;
    // A block never crosses a linear page; a straddling first instruction
    // retains incremental fetch and has its complete physical source span.
    let linear = cpu.legacy.segments[0].base.wrapping_add(ip as u32);
    let page_end = ip + (4096 - u64::from(linear & 4095));
    let mut reloaded = 0;
    loop {
        // Do not speculatively fetch a successor across this page. A first
        // instruction may straddle and is fetched incrementally as Checked.
        if ip != cpu.rip && ip + 15 > page_end {
            return Ok(block.unwrap());
        }
        let bytes = match fetch_watched(vm, cpu, context, ip, Some(source)) {
            Ok(bytes) => bytes,
            Err(exit) => return block.ok_or(exit),
        };
        let mut instruction = match crate::lift::legacy::instruction_for_profile(
            &bytes,
            ip as u32,
            context,
            vm.machine_model_profile(context.generation),
        ) {
            Ok(ir) => ir,
            Err(error) => return block.ok_or_else(|| direct_refusal(cpu, error)),
        };
        let loads = specialize_segments(&mut instruction, cpu, reloaded);
        let ordinary = crate::interp::legacy::flat32_ordinary(&instruction);
        if !ordinary || ip + u64::from(instruction.guest_len) > page_end {
            if let Some(block) = block.take() {
                return Ok(block);
            }
        }
        let mut decoder = Decoder::with_ip(
            u32::from(context.default_operand_bits),
            &bytes,
            ip,
            crate::model::decode::SYSTEM_DECODER_OPTIONS,
        );
        let insn = decoder.decode();
        ip += u64::from(instruction.guest_len);
        reloaded = loads;
        if let Some(block) = &mut block {
            block.guest_len += instruction.guest_len;
            block.icount += 1;
            // Temporaries never escape an instruction; reuse its local slots.
            block.temp_count = block.temp_count.max(instruction.temp_count);
            block.ops.extend(instruction.ops);
        } else {
            block = Some(instruction);
        }
        if !ordinary
            || insn.flow_control() != iced_x86::FlowControl::Next
            || ip >= page_end
            || block.as_ref().unwrap().icount == 32
        {
            return Ok(block.unwrap());
        }
    }
}

/// Only scalar address IR is specialized. Context-changing helpers other
/// than data-segment loads remain singleton Checked instructions. `reloaded`
/// holds the segments an earlier instruction of the same block loads: their
/// later operands keep the live Checked access, which reads the cache that
/// load installed, never the entry stamp's proof (spec.md §17.10).
/// Returns `reloaded` extended by this instruction's own loads.
pub(super) fn specialize_segments(ir: &mut IrBlock, cpu: &CpuState, reloaded: u8) -> u8 {
    let loads = ir.ops.iter().fold(reloaded, |mask, op| match op {
        IrOp::Legacy(op) => crate::interp::legacy::data_segment_load(op)
            .map_or(mask, |segment| mask | (1 << segment as u8)),
        _ => mask,
    });
    if ir
        .legacy_context
        .is_none_or(|c| c.segment_class != crate::SegmentClass::Seg16)
    {
        return loads;
    }
    for op in &mut ir.ops {
        if let IrOp::Legacy(crate::legacy::LegacyOp::Address {
            dst,
            segment,
            offset,
            bytes,
            access,
        }) = *op
        {
            if reloaded & (1 << segment as u8) != 0 {
                continue;
            }
            *op = IrOp::Legacy(crate::legacy::LegacyOp::Seg16Address {
                dst,
                segment,
                offset,
                bytes,
                access,
                proof: crate::legacy::SegmentAccessProof::new(
                    cpu.legacy.segments[segment as usize],
                    access,
                ),
            });
        }
    }
    loads
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat32_proof_publishes_and_reuses_multi_instruction_ir() {
        for model in [crate::CpuModel::I80386, crate::CpuModel::I80486Dx] {
            for class in [crate::SegmentClass::Flat32, crate::SegmentClass::Seg16] {
                let mut vm = Vm::new(crate::VmConfig::flat(0x4000));
                vm.set_cpu_model(model);
                vm.map(0, 0x4000, crate::Prot::RWX, crate::RegionKind::Ram)
                    .unwrap();
                vm.write_bytes(0x1000, b"\x40\x43\x42\xeb\x00").unwrap();
                let mut cpu = vm.new_vcpu();
                cpu.cpu.rip = 0x1000;
                cpu.cpu.legacy.cr0 = 1;
                for (i, cache) in cpu.cpu.legacy.segments[..4].iter_mut().enumerate() {
                    cache.base = 0;
                    cache.limit = if class == crate::SegmentClass::Flat32 {
                        u32::MAX
                    } else {
                        0xffff
                    };
                    cache.attributes = if i == 0 { 0x9b } else { 0x93 }
                        | if class == crate::SegmentClass::Flat32 {
                            0x4000
                        } else {
                            0
                        };
                }
                let context = derive_legacy_context(model.legacy_model(), &cpu.cpu).unwrap();
                assert_eq!(context.segment_class, class);
                let mut execution = Execution::new();
                execution.sync_class(&cpu.cpu, model.legacy_model(), false, context.segment_class);
                let exit = AtomicU64::new(0);
                *vm.source_read_hook.lock().unwrap() = Some(|vm| {
                    vm.write_bytes(0x1001, b"\x90").unwrap();
                });
                let (a, walk) = execution
                    .resolve::<MACHINE_DISPATCH>(
                        &vm,
                        &cpu.cpu,
                        model.legacy_model(),
                        context,
                        &exit,
                    )
                    .unwrap();
                assert!(matches!(walk, BlockWalk::Direct));
                assert_eq!(a.icount, 4);
                assert!(
                    !a.ops.iter().any(|op| matches!(
                        op,
                        IrOp::WriteReg {
                            reg: crate::Reg::Rbx,
                            ..
                        }
                    )),
                    "a source write before registration must cancel publication"
                );
                let (b, _) = execution
                    .resolve::<MACHINE_DISPATCH>(
                        &vm,
                        &cpu.cpu,
                        model.legacy_model(),
                        context,
                        &exit,
                    )
                    .unwrap();
                assert!(Arc::ptr_eq(&a, &b));
                vm.write_bytes(0x1001, b"\x43").unwrap();
                let (c, _) = execution
                    .resolve::<MACHINE_DISPATCH>(
                        &vm,
                        &cpu.cpu,
                        model.legacy_model(),
                        context,
                        &exit,
                    )
                    .unwrap();
                assert!(!Arc::ptr_eq(&a, &c));
            }
        }
    }
}
