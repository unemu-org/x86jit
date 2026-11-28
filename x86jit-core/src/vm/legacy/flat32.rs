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
        self.flat32
            .resolve(vm, cpu, context, key, exit_word, &mut self.links)
    }
}

/// Proofs retained per vCPU (xj-to9.53, xj-to9.56): a protected-mode
/// program's working set across its renderer, game logic and an
/// extended-memory runtime's interrupt path, which enters its blocks under
/// many contexts, reaches several thousand blocks; at 1024 proofs a PC-class
/// embedder's game relifted 130 to 150 thousand blocks with unchanged keys
/// every five seconds.
pub(super) const MAX_PROOFS: usize = 8192;

/// The proofs' IR bytes retained per vCPU (`weight`), the bound that holds
/// when blocks are large: an op is 72 bytes, a block of 32 instructions at
/// 10–15 ops each is 25–35 KiB, so 8192 such proofs would hold 200–280 MiB.
/// 32 MiB keeps 1000 such blocks or 8192 of 4 KiB.
pub(super) const MAX_PROOF_BYTES: usize = 32 << 20;

/// A proof's weight against `MAX_PROOF_BYTES`: its IR (the block, its ops
/// at their capacity) and what its key over `sources` adds to the map's
/// source index (`proof_map::index_weight`).
pub(super) fn weight<K: super::proof_map::ProofKey>(
    ir: &IrBlock,
    key: &K,
    sources: &[(u64, u32)],
) -> usize {
    std::mem::size_of::<IrBlock>()
        + ir.ops.capacity() * std::mem::size_of::<IrOp>()
        + super::proof_map::index_weight(key, sources)
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
    // The block `resolve` last returned a proof for and the map's version
    // then (`holds`).
    last: Option<(usize, u64)>,
    #[cfg(test)]
    pub(super) lifts: u64,
}

impl State {
    /// Advances whenever the map may drop or replace a proof, and with it
    /// its registration; successor links (`links.rs`) name it.
    /// The proof generation the proofs were last reconciled to.
    pub(super) fn generation(&self) -> u64 {
        self.blocks.generation()
    }

    pub(super) fn version(&self) -> u64 {
        self.blocks.version()
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

    /// The proofs' weight now and its peak since the last call.
    #[cfg(feature = "test-hooks")]
    pub(super) fn take_weight(&mut self) -> (usize, usize) {
        (self.blocks.weight(), self.blocks.take_peak_weight())
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
        links: &mut super::links::Links,
    ) -> Result<(Arc<IrBlock>, BlockWalk), Exit> {
        #[cfg(feature = "test-hooks")]
        let class = super::diag::ProofClass::of(context.segment_class, false);
        #[cfg(feature = "test-hooks")]
        let mut retry = None;
        loop {
            vm.handle_smc(exit_word);
            let mapping = vm.mem.mapping_identity();
            if self
                .mapping
                .as_ref()
                .is_none_or(|old| !Arc::ptr_eq(old, &mapping))
            {
                self.blocks
                    .reset(vm.cache.proof_generation(), why!(Mapping));
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
                self.last = Some((Arc::as_ptr(&proof.ir) as usize, self.blocks.version()));
                return Ok((proof.ir.clone(), proof.walk));
            }
            probe!(LIFTS);
            #[cfg(feature = "test-hooks")]
            self.blocks
                .record_lift(vm, class, &key, retry.take(), key.decode_ip, context);
            #[cfg(test)]
            {
                self.lifts += 1;
            }
            let source = crate::memory::LiftSource::new(&vm.mem);
            let permit = vm.mem.fetch_permit();
            let mut live = super::live::Plan::default();
            let ir = lift(vm, cpu, context, &source, &mut live)?;
            if !source.validate() || permit.as_ref().is_some_and(|p| !p.is_current()) {
                #[cfg(feature = "test-hooks")]
                {
                    retry = Some(super::diag::RetryKind::SourceChanged);
                }
                continue;
            }
            if !source.publishable() {
                #[cfg(feature = "test-hooks")]
                vm.live_fields.diag.uncached(class);
                self.blocks.remove_unpublished(&key);
                self.last = None;
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
                #[cfg(feature = "test-hooks")]
                {
                    retry = Some(super::diag::RetryKind::Registration);
                }
                continue;
            }
            // Keep retention bounded independently of the context LRU. Each
            // proof owns its counted physical registration until eviction.
            let mut ir = ir;
            // The lift's spare capacity would count against the budget.
            ir.ops.shrink_to_fit();
            let weight = weight(&ir, &key, &sources);
            if self
                .blocks
                .make_room(MAX_PROOFS, MAX_PROOF_BYTES, weight, why!(Bound), |proof| {
                    Arc::strong_count(&proof.ir) > 1
                })
            {
                // The sweep advanced the version every Flat link names
                // (`links.rs`): the stale links would only hold their blocks
                // and count them as hot at the next sweep.
                links.drop_path(super::links::LinkPath::Flat);
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
                weight,
            );
            self.last = Some((Arc::as_ptr(&ir) as usize, self.blocks.version()));
            return Ok((ir, walk));
        }
    }
}

fn lift(
    vm: &Vm,
    cpu: &CpuState,
    context: MachineContext,
    source: &crate::memory::LiftSource<'_>,
    live: &mut super::live::Plan,
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
        // Each byte's bus address, as fetch and the registered sources use it.
        let bus = |offset: u32| {
            crate::legacy::physical_span(
                cpu,
                crate::legacy::physical_mask(context.generation),
                u64::from(
                    cpu.legacy.segments[0]
                        .base
                        .wrapping_add((ip as u32).wrapping_add(offset)),
                ),
                1,
            )
            .0
        };
        let mut instruction = match vm.live_fields.instruction(
            &bytes,
            ip as u32,
            context,
            vm.machine_model_profile(context.generation),
            crate::lift::legacy::alignment_mask(cpu),
            bus,
            live,
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

    // Blocks of INC EAX; JMP $+2 from 0x4000, more than MAX_PROOFS, on RAM
    // the fixture maps from 0x4000 to 0x10000.
    const LOOP: u64 = 0x4000;
    const PAST: u64 = (MAX_PROOFS + MAX_PROOFS / 4) as u64;

    fn past_the_bound(vm: &mut Vm) {
        vm.map(LOOP, 0xc000, crate::Prot::RWX, crate::RegionKind::Ram)
            .unwrap();
        vm.write_bytes(LOOP, &b"\x40\xeb\x00".repeat(PAST as usize))
            .unwrap();
    }

    // xj-to9.53: a working set of more blocks than the old bound of 64
    // proofs relifted every block it entered once the map cleared.
    #[test]
    fn a_working_set_past_the_bound_keeps_most_of_its_proofs() {
        for class in [crate::SegmentClass::Flat32, crate::SegmentClass::Segmented] {
            let model = crate::CpuModel::I80486Dx;
            let (mut vm, mut cpu, context) = proof_fixture(model, class);
            past_the_bound(&mut vm);
            let exit = AtomicU64::new(0);
            for (blocks, most) in [(80, 0), (300, 0), (PAST, PAST / 2)] {
                let mut execution = Execution::new();
                execution.sync_class(&cpu.cpu, model.legacy_model(), false, class);
                let mut pass = |e: &mut Execution| {
                    let before = e.flat32.lifts;
                    for i in 0..blocks {
                        resolve_at(e, &vm, &mut cpu, context, &exit, LOOP + 3 * i);
                    }
                    e.flat32.lifts - before
                };
                assert_eq!(pass(&mut execution), blocks);
                let again = pass(&mut execution);
                assert!(again <= most, "{class:?}: {again} of {blocks} relifted");
            }
        }
    }

    // xj-to9.53: a full map evicts the proofs no lookup returned since its
    // last sweep first. Twenty hot blocks entered between blocks entered
    // once (an interrupt path through many contexts) stay proved.
    #[test]
    fn a_flood_of_cold_blocks_keeps_the_hot_proofs() {
        let model = crate::CpuModel::I80486Dx;
        let class = crate::SegmentClass::Flat32;
        let (mut vm, mut cpu, context) = proof_fixture(model, class);
        // PAST cold blocks from LOOP and 20 hot ones from 0x3800.
        past_the_bound(&mut vm);
        vm.write_bytes(0x3800, &b"\x40\xeb\x00".repeat(20)).unwrap();
        let exit = AtomicU64::new(0);
        let mut execution = Execution::new();
        execution.sync_class(&cpu.cpu, model.legacy_model(), false, class);
        let mut hot_lifts = 0;
        for i in 0..PAST {
            if i % 10 == 0 {
                for h in 0..20 {
                    let before = execution.flat32.lifts;
                    resolve_at(
                        &mut execution,
                        &vm,
                        &mut cpu,
                        context,
                        &exit,
                        0x3800 + 3 * h,
                    );
                    hot_lifts += execution.flat32.lifts - before;
                }
            }
            resolve_at(&mut execution, &vm, &mut cpu, context, &exit, LOOP + 3 * i);
        }
        assert_eq!(hot_lifts, 20, "only the first lift of each hot block");
    }

    // xj-to9.56: an unpaged Flat32 or Segmented identity names what its lift
    // reads. States that differ only in the IDTR, GDTR, LDTR, TR, CR3, other
    // CR0 bits, IOPL, NT or the selector values share one identity and its
    // proofs; a cache the class bakes, CR0.AM and the A20 line do not.
    #[test]
    fn an_unpaged_identity_names_only_what_its_lift_reads() {
        type Edit = fn(&mut CpuState);
        let shared: [Edit; 12] = [
            |c| {
                c.legacy.idtr = crate::legacy::TableRegister {
                    base: 0,
                    limit: 0x3ff,
                    reserved: 0,
                }
            },
            |c| c.legacy.gdtr.base = 0x6000,
            |c| c.legacy.ldtr.selector = 0x28,
            |c| c.legacy.ldtr.cache.base = 0x5000,
            |c| c.legacy.tr.cache.attributes = 0x8b,
            |c| c.legacy.cr3 = 0x1000,
            |c| c.legacy.cr0 |= 1 << 3,
            |c| c.rflags_system |= 0x3000 | 0x4000,
            |c| c.cs = 0x170,
            |c| c.ds = 0x178,
            |c| c.ss = 0x18,
            |c| c.fs = 0x20,
        ];
        let distinct: [Edit; 2] = [|c| c.legacy.cr0 |= 1 << 18, |c| c.legacy.a20_enabled ^= 1];
        for class in [crate::SegmentClass::Flat32, crate::SegmentClass::Segmented] {
            let model = crate::CpuModel::I80486Dx;
            let (vm, mut cpu, context) = proof_fixture(model, class);
            cpu.cpu.legacy.a20_enabled = 1;
            let legacy = model.legacy_model();
            let mut execution = Execution::new();
            execution.sync_class(&cpu.cpu, legacy, false, class);
            let exit = AtomicU64::new(0);
            let a = resolve_at(&mut execution, &vm, &mut cpu, context, &exit, A);
            let id = execution.context_id;
            let initial = cpu.cpu.clone();
            let segment_base: Edit = |c| c.legacy.segments[SegmentId::Ds as usize].base = 0x10;
            let distinct = distinct
                .iter()
                .chain((class == crate::SegmentClass::Segmented).then_some(&segment_base));
            let edits = shared.iter().map(|e| (true, e));
            for (shares, edit) in edits.chain(distinct.map(|e| (false, e))) {
                cpu.cpu = initial.clone();
                edit(&mut cpu.cpu);
                let edited = derive_legacy_context(legacy, &cpu.cpu).unwrap();
                assert_eq!(edited.segment_class, class);
                execution.sync_class(&cpu.cpu, legacy, false, class);
                let b = resolve_at(&mut execution, &vm, &mut cpu, edited, &exit, A);
                assert_eq!(execution.context_id == id, shares, "{class:?}");
                assert_eq!(Arc::ptr_eq(&a, &b), shares, "{class:?}");
            }
            cpu.cpu = initial;
            execution.sync_class(&cpu.cpu, legacy, false, class);
            assert_eq!(execution.context_id, id);
        }
    }

    // xj-to9.56: a new identity is never an evicted one's, so no key names
    // the evicted identity's proofs again and no walk over the proof map
    // drops them; the bound sweeps them as unused.
    #[test]
    fn an_evicted_identity_is_never_reused() {
        let model = crate::CpuModel::I80486Dx;
        let class = crate::SegmentClass::Segmented;
        let (vm, mut cpu, context) = proof_fixture(model, class);
        let legacy = model.legacy_model();
        let mut execution = Execution::new();
        let exit = AtomicU64::new(0);
        let mut first = None;
        for k in 0..=super::MAX_RETAINED_CONTEXTS as u32 {
            cpu.cpu.legacy.segments[SegmentId::Ds as usize].base = 0x10 * k;
            execution.sync_class(&cpu.cpu, legacy, false, class);
            assert_eq!(execution.context_id, u64::from(k));
            let ir = resolve_at(&mut execution, &vm, &mut cpu, context, &exit, A);
            first.get_or_insert(ir);
        }
        assert_eq!(execution.contexts.len(), super::MAX_RETAINED_CONTEXTS);
        // The first stamp was evicted: it returns under a fresh identity.
        cpu.cpu.legacy.segments[SegmentId::Ds as usize].base = 0;
        execution.sync_class(&cpu.cpu, legacy, false, class);
        assert_eq!(
            execution.context_id,
            super::MAX_RETAINED_CONTEXTS as u64 + 1
        );
        let again = resolve_at(&mut execution, &vm, &mut cpu, context, &exit, A);
        assert!(!Arc::ptr_eq(first.as_ref().unwrap(), &again));
    }

    // xj-to9.56: a bound sweep advances the proof map's version, which every
    // Flat link names; the stale links are dropped so that they do not hold
    // their blocks as hot at the next sweep. Other paths' links stay.
    #[test]
    fn a_bound_sweep_drops_the_stale_flat_links() {
        let model = crate::CpuModel::I80486Dx;
        let class = crate::SegmentClass::Flat32;
        let (mut vm, mut cpu, context) = proof_fixture(model, class);
        past_the_bound(&mut vm);
        let mut execution = Execution::new();
        execution.sync_class(&cpu.cpu, model.legacy_model(), false, class);
        let exit = AtomicU64::new(0);
        let a = resolve_at(&mut execution, &vm, &mut cpu, context, &exit, A);
        execution
            .links
            .plant(super::links::LinkPath::Flat, &a, context);
        execution
            .links
            .plant(super::links::LinkPath::Direct, &a, context);
        // A and MAX_PROOFS - 1 more fill the map; the next lift sweeps.
        for i in 0..MAX_PROOFS as u64 - 1 {
            resolve_at(&mut execution, &vm, &mut cpu, context, &exit, LOOP + 3 * i);
        }
        assert_eq!(execution.links.count(super::links::LinkPath::Flat), 1);
        resolve_at(
            &mut execution,
            &vm,
            &mut cpu,
            context,
            &exit,
            LOOP + 3 * MAX_PROOFS as u64,
        );
        assert_eq!(execution.links.count(super::links::LinkPath::Flat), 0);
        assert_eq!(execution.links.count(super::links::LinkPath::Direct), 1);
    }

    // xj-to9.56: the map is bounded by its proofs' weight as well as their
    // count: heavy proofs sweep it before the count is reached, and the
    // weight never exceeds the budget.
    #[test]
    fn heavy_proofs_stay_within_the_byte_budget() {
        let mut map = super::super::proof_map::ProofMap::<u64, ()>::default();
        let (bound, budget, heavy) = (1000, 1 << 20, (1 << 20) / 10 + 1);
        let mut swept = 0;
        for key in 0..100u64 {
            if map.make_room(bound, budget, heavy, why!(Bound), |_| false) {
                swept += 1;
            }
            map.insert(key, (), &[(0x1000 * key, 1)], heavy);
            assert!(map.weight() <= budget, "{key}: {}", map.weight());
            assert!(map.len() < 10);
        }
        assert!(swept > 0);
        // Light proofs sweep at the count bound alone.
        let mut map = super::super::proof_map::ProofMap::<u64, ()>::default();
        for key in 0..2 * bound as u64 {
            map.make_room(bound, budget, 1, why!(Bound), |_| false);
            map.insert(key, (), &[(0x1000 * key, 1)], 1);
            assert!(map.len() <= bound);
        }
        assert!(map.len() > bound / 2);
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
