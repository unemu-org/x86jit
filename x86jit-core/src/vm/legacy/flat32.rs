//! Interpreted unpaged Flat32 and Segmented blocks (decision-27; spec.md §17.10).
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
    mapping: Option<Arc<()>>,
    blocks: super::proof_map::ProofMap<LegacyBlockKey, Proof>,
    #[cfg(test)]
    pub(super) lifts: u64,
}

impl State {
    /// Advances whenever the map may drop or replace a proof, and with it
    /// its registration; successor links (`links.rs`) name it.
    pub(super) fn version(&self) -> u64 {
        self.blocks.version()
    }

    /// Whether a proof holds this very block.
    pub(super) fn holds(&self, ir: &Arc<IrBlock>) -> bool {
        self.blocks.values().any(|proof| Arc::ptr_eq(&proof.ir, ir))
    }

    #[cold]
    #[inline(never)]
    pub(super) fn evict_context(&mut self, id: u64) {
        self.blocks.retain(|key| key.context_id != id);
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
            let mapping = vm.mem.mapping_identity();
            if self
                .mapping
                .as_ref()
                .is_none_or(|old| !Arc::ptr_eq(old, &mapping))
            {
                self.blocks.reset(vm.cache.publication_generation());
                self.mapping = Some(mapping);
            }
            // A write drops the proofs whose physical sources it overlaps.
            let generation = self.blocks.reconcile(&vm.cache);
            if let Some(proof) = self
                .blocks
                .get(&key)
                .filter(|p| p.permit.as_ref().is_none_or(|p| p.is_current()))
            {
                probe!(PROOFS);
                return Ok((proof.ir.clone(), proof.walk));
            }
            probe!(LIFTS);
            #[cfg(test)]
            {
                self.lifts += 1;
            }
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
                // The model's bus mask, as fetch used it: an 80286 CS.base + IP
                // past FFFFFFh wraps to the first megabyte (I286 PRM §6.4 p. 6-6).
                let (physical, _) = crate::legacy::physical_span(
                    cpu,
                    crate::legacy::physical_mask(context.generation),
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
            let link_exit = super::links::link_exit(&ir);
            let ir = Arc::new(IrBlock { link_exit, ..ir });
            self.blocks.insert(
                key,
                Proof {
                    ir: ir.clone(),
                    walk,
                    permit,
                    _registration: registration.unwrap(),
                },
                &sources,
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
        let bytes = match fetch_watched(vm, cpu, context, ip, Some(source), &[]) {
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
        .is_none_or(|c| c.segment_class != crate::SegmentClass::Segmented)
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
            *op = IrOp::Legacy(crate::legacy::LegacyOp::SegmentedAddress {
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
            for class in [crate::SegmentClass::Flat32, crate::SegmentClass::Segmented] {
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

    // xj-to9.45: blocks A, B and C share one page; C sits 2 KiB on.
    // Block A: INC EAX; JMP $+2. B: INC EBX; JMP $+2. C: INC ECX; JMP $+2.
    const A: u64 = 0x1000;
    const B: u64 = 0x1003;
    const C: u64 = 0x1800;

    fn proof_fixture(
        model: crate::CpuModel,
        class: crate::SegmentClass,
    ) -> (Vm, crate::Vcpu, MachineContext) {
        let mut vm = Vm::new(crate::VmConfig::flat(0x10000));
        vm.set_cpu_model(model);
        vm.map(0, 0x4000, crate::Prot::RWX, crate::RegionKind::Ram)
            .unwrap();
        vm.write_bytes(A, b"\x40\xeb\x00\x43\xeb\x00").unwrap();
        vm.write_bytes(C, b"\x41\xeb\x00").unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.legacy.cr0 = 1;
        for (i, cache) in cpu.cpu.legacy.segments[..4].iter_mut().enumerate() {
            cache.base = 0;
            let flat = class == crate::SegmentClass::Flat32;
            cache.limit = if flat { u32::MAX } else { 0xffff };
            cache.attributes = if i == 0 { 0x9b } else { 0x93 } | if flat { 0x4000 } else { 0 };
        }
        let context = derive_legacy_context(model.legacy_model(), &cpu.cpu).unwrap();
        assert_eq!(context.segment_class, class);
        (vm, cpu, context)
    }

    fn resolve_at(
        execution: &mut Execution,
        vm: &Vm,
        cpu: &mut crate::Vcpu,
        context: MachineContext,
        exit: &AtomicU64,
        ip: u64,
    ) -> Arc<IrBlock> {
        cpu.cpu.rip = ip;
        let model = vm.cpu_model.unwrap().legacy_model();
        execution
            .resolve::<MACHINE_DISPATCH>(vm, &cpu.cpu, model, context, exit)
            .unwrap()
            .0
    }

    fn writes(ir: &IrBlock, reg: crate::Reg) -> bool {
        ir.ops
            .iter()
            .any(|op| matches!(op, IrOp::WriteReg { reg: r, .. } if *r == reg))
    }

    #[test]
    fn a_code_write_drops_only_the_proofs_whose_bytes_it_overlaps() {
        for model in [crate::CpuModel::I80386, crate::CpuModel::I80486Dx] {
            for class in [crate::SegmentClass::Flat32, crate::SegmentClass::Segmented] {
                let (vm, mut cpu, context) = proof_fixture(model, class);
                let mut execution = Execution::new();
                execution.sync_class(&cpu.cpu, model.legacy_model(), false, class);
                let exit = AtomicU64::new(0);
                let at = |e: &mut Execution, cpu: &mut crate::Vcpu, ip| {
                    resolve_at(e, &vm, cpu, context, &exit, ip)
                };
                let [a, b, c] = [A, B, C].map(|ip| at(&mut execution, &mut cpu, ip));
                assert_eq!(execution.flat32.lifts, 3);
                // INC EBX becomes INC EDX: only B's proof overlaps the byte.
                vm.write_bytes(B, b"\x42").unwrap();
                assert!(Arc::ptr_eq(&a, &at(&mut execution, &mut cpu, A)));
                assert!(Arc::ptr_eq(&c, &at(&mut execution, &mut cpu, C)));
                assert_eq!(execution.flat32.lifts, 3, "{model:?} {class:?}");
                let b2 = at(&mut execution, &mut cpu, B);
                assert!(!Arc::ptr_eq(&b, &b2));
                assert!(writes(&b2, crate::Reg::Rdx) && !writes(&b2, crate::Reg::Rbx));
                assert_eq!(execution.flat32.lifts, 4);
                // A data store on the page, next to every proof, drops none.
                vm.write_bytes(0x1f00, &[1; 4]).unwrap();
                for (ip, ir) in [(A, &a), (B, &b2), (C, &c)] {
                    assert!(Arc::ptr_eq(ir, &at(&mut execution, &mut cpu, ip)));
                }
                assert_eq!(execution.flat32.lifts, 4);
                // A store into A's last byte (its JMP displacement) drops A.
                vm.write_bytes(A + 2, b"\x00").unwrap();
                assert!(!Arc::ptr_eq(&a, &at(&mut execution, &mut cpu, A)));
                assert!(Arc::ptr_eq(&b2, &at(&mut execution, &mut cpu, B)));
                assert_eq!(execution.flat32.lifts, 5);
            }
        }
    }

    #[test]
    fn unrecorded_advances_and_mapping_changes_drop_every_proof() {
        for class in [crate::SegmentClass::Flat32, crate::SegmentClass::Segmented] {
            let model = crate::CpuModel::I80486Dx;
            let (mut vm, mut cpu, context) = proof_fixture(model, class);
            let mut execution = Execution::new();
            execution.sync_class(&cpu.cpu, model.legacy_model(), false, class);
            let exit = AtomicU64::new(0);
            let warm = |e: &mut Execution, vm: &Vm, cpu: &mut crate::Vcpu| {
                [A, C].map(|ip| resolve_at(e, vm, cpu, context, &exit, ip))
            };
            let fresh = |old: &[Arc<IrBlock>; 2], new: &[Arc<IrBlock>; 2]| {
                old.iter().zip(new).all(|(o, n)| !Arc::ptr_eq(o, n))
            };
            // An invalidation outside the drain (here a wide one) is
            // unrecorded: a gap, which no proof survives.
            let before = warm(&mut execution, &vm, &mut cpu);
            vm.cache
                .invalidate_overlapping(0, u64::MAX, || vm.mem.clear_all_code_pages());
            let after = warm(&mut execution, &vm, &mut cpu);
            assert!(fresh(&before, &after));
            // More advances than the record retains: their bytes are unknown.
            for i in 0..300 {
                vm.write_bytes(0x1c00 + i, &[0]).unwrap();
            }
            let lost = warm(&mut execution, &vm, &mut cpu);
            assert!(fresh(&after, &lost));
            // A mapping change drops every proof.
            vm.map(0x8000, 0x1000, crate::Prot::RWX, crate::RegionKind::Ram)
                .unwrap();
            let mapped = warm(&mut execution, &vm, &mut cpu);
            assert!(fresh(&lost, &mapped));
            // A reset of the shared cache (a closed code lifetime rebound)
            // records no bytes either.
            let next = Arc::new(crate::code_lifetime::CodeLifetime::default());
            // SAFETY: the interpreter backend has no compiled code to stop.
            unsafe { vm.cache.code_lifetime().retire_all() };
            assert!(vm.cache.reconcile_code_lifetime(next, |_| {}));
            let reset = warm(&mut execution, &vm, &mut cpu);
            assert!(fresh(&mapped, &reset));
            let again = warm(&mut execution, &vm, &mut cpu);
            assert!(!fresh(&reset, &again));
        }
    }

    #[test]
    fn another_vcpu_draining_a_code_write_drops_only_its_proofs() {
        let model = crate::CpuModel::I80386;
        let class = crate::SegmentClass::Flat32;
        let (vm, mut cpu, context) = proof_fixture(model, class);
        let mut first = Execution::new();
        let mut second = Execution::new();
        first.sync_class(&cpu.cpu, model.legacy_model(), false, class);
        second.sync_class(&cpu.cpu, model.legacy_model(), false, class);
        let (one, two) = (AtomicU64::new(0), AtomicU64::new(0));
        let [a, b, c] =
            [A, B, C].map(|ip| resolve_at(&mut first, &vm, &mut cpu, context, &one, ip));
        // The second vCPU's dispatch drains the write first.
        vm.write_bytes(B, b"\x42").unwrap();
        assert!(vm.mem.has_dirty_code());
        resolve_at(&mut second, &vm, &mut cpu, context, &two, C);
        assert!(!vm.mem.has_dirty_code());
        assert!(Arc::ptr_eq(
            &a,
            &resolve_at(&mut first, &vm, &mut cpu, context, &one, A)
        ));
        assert!(Arc::ptr_eq(
            &c,
            &resolve_at(&mut first, &vm, &mut cpu, context, &one, C)
        ));
        let b2 = resolve_at(&mut first, &vm, &mut cpu, context, &one, B);
        assert!(!Arc::ptr_eq(&b, &b2) && writes(&b2, crate::Reg::Rdx));
        assert_eq!(first.flat32.lifts, 4);
    }
}
