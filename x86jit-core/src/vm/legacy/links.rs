//! Successor links (xj-to9.27): a small direct-mapped memo, per vCPU, of the
//! blocks that the Machine dispatcher's lookups returned, consulted on the
//! turn after a block whose last transfer is direct or a near RET (xj-to9.30),
//! before the resolver that turn would call. A link stands in for that
//! resolver only when every input the resolver reads is the one the link was
//! made under, so it returns a block the resolver would return:
//!
//! * **Address.** The IP, the physical fetch address and MXCSR, and the
//!   resolver path (twenty-bit, Direct, unpaged Flat32/Segmented, paged).
//!   The Direct path adds its context and A20 (its cache key).
//! * **Context.** The unpaged protected and paged paths name the retained
//!   context identity (`Execution::context_id`), which every context change
//!   re-derives (`sync_class`) and whose reuse after an eviction, or a
//!   translation flush, empties the links. The paged path also names the
//!   paging state's identity, proof-map version and TLB generation, so the
//!   paging reconciliation a full lookup performs would change nothing.
//! * **Bytes.** The proof generation read before the lookup that made the
//!   link, after this turn's `handle_smc`: every drained store into a tagged
//!   code page advances it (cache.rs `invalidate_range`) except one absorbed
//!   by a live instruction field (xj-to9.46, `live.rs`), which every block
//!   over those bytes reads when it executes and no shared block covers. A
//!   twenty-bit or Direct link names a block of the shared cache, whose
//!   blocks leave only by an invalidation that advances it; an unpaged
//!   protected or paged link names a block of a proof map, whose version
//!   advances whenever it may drop a proof and with it the registration that
//!   keeps the page tags. Unpublished blocks are not linked.
//! * **Run.** A link is followed only in the run that made it: no link meets
//!   a fork, snapshot, restore, register or memory edit, mapping change or
//!   fetch policy made between runs.
//!
//! Only the turn directly after an ordinary completed block whose last
//! transfer is direct or a near RET (`IrBlock::link_exit`: a `Branch`,
//! `LoopCx`, an immediate `Jump`, or a jump to the IP a near RET popped)
//! consults the links. Nothing about the transfer that reached a turn is
//! part of a link: it is found by the IP the turn starts at, and validated
//! by the inputs above, so a near RET needs no return-address stack. A
//! rewritten return address, a stack switch, a `longjmp` past pending
//! returns and recursion of any depth only make a RET land on another IP,
//! whose own link, or full lookup, serves it (xj-to9.30). Far returns load
//! CS and are not linked; indirect jumps and calls are not linked either.
//! Event delivery, faults, context changes, device handoffs and completed
//! port-string inputs, the host request and NMI word, a service step that
//! runs a handoff's numeric or debug service, single-step traps, A20 changes
//! and a turn that drained a store into a code page take the full lookup. A
//! one-step turn (an interrupt or SS shadow, a pending IRQ the guest masks)
//! and a service turn that delivers nothing still consult the links: neither
//! changes an input a link names. A retained instruction (self-modifying
//! code), a continuation, a pending task switch and, under a fetch policy,
//! the protected proofs, whose permits are checked on every hit, are never
//! linked.
//! Twenty-bit blocks of retained pages are checked against their bytes on
//! every lookup and are never linked either.
//!
//! A link is made on a successor's second miss under one generation and
//! identity, into a slot whose link was not made or followed since it was
//! last spared: code whose stores keep advancing the generation, or whose TLB
//! or proof map keeps turning over, makes none, and two blocks that share a
//! slot keep one link instead of replacing each other. A block cut at the
//! lifter's bound continues sequentially and is not a direct exit
//! (`IrBlock::ends_direct`). The return sites of a subroutine take one slot
//! each, as any successor does. This bears on cost, not validity.
use super::*;

const SLOTS: usize = 32;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum LinkPath {
    Twenty,
    Direct,
    Flat,
    Paged,
}

/// A turn's lookup inputs that choose the resolver.
pub(super) struct Lookup {
    pub(super) context: MachineContext,
    pub(super) physical: u64,
    pub(super) twenty: bool,
    pub(super) direct: bool,
    pub(super) retained: bool,
    pub(super) retry: bool,
    pub(super) paged: bool,
}

/// A consultation that found no link and may make one: the path, the lookup
/// inputs and the proof generation read before the full lookup.
#[derive(Clone, Copy)]
struct Miss {
    path: LinkPath,
    generation: u64,
    context: MachineContext,
    physical: u64,
}

struct Link {
    rip: u64,
    physical: u64,
    run: u64,
    generation: u64,
    // Path-specific identity (`Execution::link_identity`).
    identity: [u64; 4],
    context: MachineContext,
    mxcsr: u32,
    path: LinkPath,
    walk: BlockWalk,
    // Made or followed since it was last spared: a conflicting fill spares
    // it once instead of replacing it.
    used: bool,
    ir: Arc<IrBlock>,
}

#[derive(Default)]
pub(super) struct Links {
    slots: [Option<Link>; SLOTS],
    // Per slot, the IP, generation, and run and identity (folded) of the
    // last miss that made no link.
    seen: [(u64, u64, u64); SLOTS],
    run: u64,
    // Whether the protected proof maps may be linked: no fetch policy.
    protected: bool,
    // The consultation whose full lookup this turn may link.
    pending: Option<Miss>,
    #[cfg(test)]
    pub(super) followed: u64,
    #[cfg(test)]
    pub(super) made: u64,
    #[cfg(test)]
    pub(super) cleared: u64,
    // Whether the block whose exit the next consultation follows ended in a
    // near RET, and the links followed after one (`return_witness.rs`).
    #[cfg(test)]
    pub(super) after_return: bool,
    #[cfg(test)]
    pub(super) returns: u64,
}

impl Links {
    /// A new run: no earlier link matches its tag.
    pub(super) fn begin_run(&mut self, protected: bool) {
        self.run += 1;
        self.protected = protected;
    }
    /// Drop the links of `path`, which a version advance made stale.
    #[cold]
    #[inline(never)]
    pub(super) fn drop_path(&mut self, path: LinkPath) {
        for slot in &mut self.slots {
            if slot.as_ref().is_some_and(|link| link.path == path) {
                *slot = None;
            }
        }
    }
    /// A link of `path` to `ir` in the first free slot (tests only).
    #[cfg(test)]
    pub(super) fn plant(&mut self, path: LinkPath, ir: &Arc<IrBlock>, context: MachineContext) {
        let slot = self.slots.iter_mut().find(|s| s.is_none()).unwrap();
        *slot = Some(Link {
            rip: ir.guest_start,
            physical: ir.guest_start,
            run: self.run,
            generation: 0,
            identity: [0; 4],
            context,
            mxcsr: 0,
            path,
            walk: BlockWalk::for_flat32(ir),
            used: true,
            ir: Arc::clone(ir),
        });
    }
    /// The links of `path` (tests only).
    #[cfg(test)]
    pub(super) fn count(&self, path: LinkPath) -> usize {
        self.slots
            .iter()
            .flatten()
            .filter(|l| l.path == path)
            .count()
    }
    #[cold]
    #[inline(never)]
    pub(super) fn clear(&mut self) {
        self.slots = Default::default();
        self.run += 1;
        #[cfg(test)]
        {
            self.cleared += 1;
        }
    }
}

// A test-only switch: the same dispatcher with every link refused, the
// reference the link witnesses compare against.
#[cfg(test)]
thread_local! {
    pub(super) static OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[inline(always)]
pub(super) fn off() -> bool {
    #[cfg(test)]
    return OFF.with(std::cell::Cell::get);
    #[cfg(not(test))]
    false
}

/// Whether the block ends in a near RET (`lift/legacy.rs`, `Ret`, with
/// or without imm16): the IP loaded from the stack (a twenty-bit model's
/// wrapped load included), SP written, then a jump to the loaded IP, with
/// no store after the load. An indirect near CALL pushes after it loads
/// its target, an indirect JMP writes no SP, and a far return or IRET
/// loads CS (a legacy operation, or a CS write after SP), so none of them
/// is a near return. This bears on cost only: a successor link is keyed
/// on the IP the RET popped, not on the block that made the transfer.
pub(in crate::vm) fn ends_near_return(ir: &IrBlock) -> bool {
    let mut ops = ir
        .ops
        .iter()
        .rev()
        .skip_while(|op| matches!(op, IrOp::Legacy(crate::legacy::LegacyOp::SharedWalk)));
    let (
        Some(IrOp::Jump {
            target: crate::ir::Val::Temp(ip),
        }),
        Some(IrOp::WriteReg {
            reg: crate::Reg::Rsp,
            ..
        }),
    ) = (ops.next(), ops.next())
    else {
        return false;
    };
    for op in ops {
        match op {
            IrOp::Load { dst, .. }
            | IrOp::Legacy(crate::legacy::LegacyOp::WrappedLoad { dst, .. })
                if dst == ip =>
            {
                return true;
            }
            IrOp::Store { .. } | IrOp::Legacy(crate::legacy::LegacyOp::WrappedStore { .. }) => {
                return false
            }
            _ => {}
        }
    }
    false
}

/// Whether the Machine dispatcher consults the successor links after
/// `ir` (`IrBlock::link_exit`): a direct exit or a near RET.
pub(in crate::vm) fn link_exit(ir: &IrBlock) -> bool {
    ir.ends_direct() || ends_near_return(ir)
}

// The derived comparison is out of line; a context is eight bytes.
#[inline(always)]
fn same_context(a: MachineContext, b: MachineContext) -> bool {
    a.segment_class == b.segment_class
        && a.generation == b.generation
        && a.regime == b.regime
        && a.default_operand_bits == b.default_operand_bits
        && a.default_address_bits == b.default_address_bits
        && a.stack_address_bits == b.stack_address_bits
        && a.cpl == b.cpl
        && a.paging_enabled == b.paging_enabled
}

// Fibonacci hashing: neighbouring block starts, a few bytes apart, spread
// over the slots.
fn slot(rip: u64) -> usize {
    (rip.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> (64 - SLOTS.trailing_zeros())) as usize
}

impl Execution {
    /// The path a link may stand in for this turn's resolver, or none.
    fn link_path(&self, lookup: &Lookup) -> Option<LinkPath> {
        if lookup.twenty {
            // A retained twenty-bit page is checked against its bytes. Its
            // blocks live outside the shared cache and are never published;
            // under a fetch policy it keeps the shared resolver, whose blocks
            // a link would return unchanged, so this refusal shows only in
            // the link counters.
            (!lookup.retained).then_some(LinkPath::Twenty)
        } else if lookup.direct {
            (!lookup.retained && !lookup.retry).then_some(LinkPath::Direct)
        } else if !self.links.protected
            || self.instruction_retry.is_some()
            || self.task_switch_pending()
            || self.context_id == u64::MAX
        {
            None
        } else if lookup.paged {
            self.paging.is_some().then_some(LinkPath::Paged)
        } else {
            matches!(
                lookup.context.segment_class,
                crate::SegmentClass::Flat32 | crate::SegmentClass::Segmented
            )
            .then_some(LinkPath::Flat)
        }
    }

    /// The resolver state a link names besides its address.
    fn link_identity(&mut self, path: LinkPath, cpu: &CpuState) -> [u64; 4] {
        match path {
            LinkPath::Twenty => [0; 4],
            LinkPath::Direct => [u64::from(cpu.legacy.a20_enabled != 0), 0, 0, 0],
            LinkPath::Flat => [self.context_id, self.flat32.version(), 0, 0],
            LinkPath::Paged => {
                let state = self.paging.as_deref_mut().unwrap();
                [
                    self.context_id,
                    state.context_id,
                    state.version(),
                    state.tlb.get_mut().generation,
                ]
            }
        }
    }

    /// The block a successor link holds for this turn's lookup, or whether
    /// the block its full lookup returns may be linked (`record_link`).
    #[inline(never)]
    pub(super) fn follow_link(
        &mut self,
        vm: &Vm,
        cpu: &CpuState,
        lookup: Lookup,
    ) -> Result<(Arc<IrBlock>, BlockWalk), bool> {
        let Some(path) = self.link_path(&lookup) else {
            return Err(false);
        };
        // After this turn's `handle_smc`.
        let generation = vm.cache.proof_generation();
        let identity = self.link_identity(path, cpu);
        let run = self.links.run;
        let index = slot(cpu.rip);
        let Some(link) = self.links.slots[index].as_mut().filter(|link| {
            link.rip == cpu.rip
                && link.physical == lookup.physical
                && link.run == run
                && link.generation == generation
                && link.mxcsr == cpu.mxcsr
                && link.path == path
                && link.identity == identity
                // Redundant today: a real-mode Direct context follows from the
                // model (cpl 0, 16-bit caches), the twenty-bit context is a
                // constant, and a protected or paged context follows from its
                // identity (`Execution::context_id`). Kept as the resolver's
                // own input.
                && same_context(link.context, lookup.context)
        }) else {
            // A successor is linked on its second miss under one
            // generation and identity: code whose stores advance the
            // generation, or whose TLB or proof map turns over, every
            // iteration, or that runs once, makes no link it cannot follow.
            // One miss is remembered per slot: two unlinked successors that
            // share a slot and miss in turn, such as a RET's changing
            // targets and a direct successor, keep each other from linking.
            // That costs a full lookup per visit, never a wrong block.
            let tag = identity
                .iter()
                .fold(run, |tag, word| tag.rotate_left(16) ^ word);
            let seen = (cpu.rip, generation, tag);
            if self.links.seen[index] != seen {
                self.links.seen[index] = seen;
                return Err(false);
            }
            self.links.pending = Some(Miss {
                path,
                generation,
                context: lookup.context,
                physical: lookup.physical,
            });
            return Err(true);
        };
        link.used = true;
        let found = (Arc::clone(&link.ir), link.walk);
        if path == LinkPath::Paged {
            // As `resolve_paged`: no operand transaction survives into
            // another block.
            self.paging.as_mut().unwrap().transaction = None;
        }
        probe!(LINKS);
        #[cfg(test)]
        {
            self.links.followed += 1;
            self.links.returns += u64::from(self.links.after_return);
        }
        Ok(found)
    }

    /// Link the published block a full lookup returned, unless the slot holds
    /// a live link made or followed since it was last spared.
    #[cold]
    #[inline(never)]
    pub(super) fn record_link(
        &mut self,
        vm: &Vm,
        cpu: &CpuState,
        resolved: &Result<(Arc<IrBlock>, BlockWalk), Exit>,
    ) {
        let Some(Miss {
            path,
            generation,
            context,
            physical,
        }) = self.links.pending.take()
        else {
            return;
        };
        let Ok((ir, walk)) = resolved else {
            return;
        };
        let walk = *walk;
        let run = self.links.run;
        let index = slot(cpu.rip);
        if let Some(link) = self.links.slots[index]
            .as_mut()
            .filter(|link| link.run == run && link.generation == generation && link.used)
        {
            link.used = false;
            return;
        }
        // Only a published block is linked: the shared cache or the proof map
        // holds this very block, and with it the source tags.
        let published = match path {
            LinkPath::Twenty | LinkPath::Direct => {
                let a20 = path == LinkPath::Direct && cpu.legacy.a20_enabled != 0;
                matches!(
                    vm.cache.get_for_dispatch(BlockKey::machine_direct(physical, cpu.mxcsr, a20), false),
                    Some(CachedBlock::Interpreted(cached)) if Arc::ptr_eq(&cached, ir)
                )
            }
            LinkPath::Flat => self.flat32.holds(ir),
            LinkPath::Paged => self.paging.as_ref().is_some_and(|state| state.holds(ir)),
        };
        if !published {
            return;
        }
        let identity = self.link_identity(path, cpu);
        #[cfg(test)]
        {
            self.links.made += 1;
        }
        self.links.slots[index] = Some(Link {
            rip: cpu.rip,
            physical,
            run,
            generation,
            identity,
            context,
            mxcsr: cpu.mxcsr,
            path,
            walk,
            used: true,
            ir: Arc::clone(ir),
        });
    }
}
