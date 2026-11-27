//! The Machine lowering of the shared translator (decision-29, J1).
//!
//! A compiled Machine block completes a prefix of its block and returns where
//! the next instruction starts: `RET_CONTINUE` after the whole block,
//! `RET_MACHINE_DEFER` before an instruction it does not complete (nothing of
//! which is committed), `RET_MACHINE_STOP` after a store that left the
//! dispatch word set. The completed count goes to `MemCtx::machine_ordinal`,
//! baked at the exit site; the dispatcher charges it. Loads and stores call
//! the `jit_abi::machine` helpers, which run the interpreter's checked access.
//!
//! Decision-31 (J2b): every static exit chains. A JMP, Jcc, LOOP-family,
//! direct CALL or fall-through exit loads its link slot and returns
//! `RET_CHAIN` (filled) or `RET_LINK` (empty); a near CALL pushes the shadow
//! return ring and a near RET predicts through it (`RET_RETURN_MISS` on a
//! miss). Each of them means that the whole block completed and RIP holds
//! the successor. Indirect exits return `RET_CONTINUE`.
use super::*;
use x86jit_core::jit_abi::machine as abi;
use x86jit_core::jit_abi::MEMCTX_GUEST_BASE;
use x86jit_core::jit_abi::{MemCtx, RET_MACHINE_DEFER, RET_MACHINE_STOP};
use x86jit_core::legacy::LegacyOp;

pub(crate) struct MachineLowering {
    config: u64,
    /// The ordinal of the instruction being translated.
    ordinal: u64,
    started: bool,
    /// Set by a store whose helper reported `ACCESS_STOP`.
    stop: Variable,
    /// A store has been translated, so instruction starts test `stop`.
    stored: bool,
    /// The block's last transfer (decision-31 §3).
    exit: abi::MachineExit,
    /// The view's read and write tag offsets for the unit's CPL
    /// (decision-32 §3): its key names the CPL, so the U/S check is free.
    tags: (i32, i32),
}

impl MachineLowering {
    pub(crate) fn new(builder: &mut FunctionBuilder, config: u64, ir: &IrBlock) -> Self {
        let stop = builder.declare_var(types::I8);
        let clear = builder.ins().iconst(types::I8, 0);
        builder.def_var(stop, clear);
        Self {
            config,
            ordinal: 0,
            started: false,
            stop,
            stored: false,
            exit: abi::machine_exit(ir),
            tags: abi::tag_offsets(ir.legacy_context.map_or(0, |c| c.cpl)),
        }
    }
}

/// Translate one ordinary Machine block (decision-29 §4).
#[allow(clippy::too_many_arguments)]
pub fn translate_machine_block(
    builder: &mut FunctionBuilder,
    ir: &IrBlock,
    offsets: &CpuOffsets,
    alloc_slot: &mut dyn FnMut() -> u64,
    helpers: Helpers,
    consistency: MemConsistency,
    access_config: u64,
) -> Box<[u64]> {
    super::translate_block_for_target(
        builder,
        ir,
        offsets,
        alloc_slot,
        helpers,
        consistency,
        None,
        0,
        false,
        cfg!(target_arch = "aarch64"),
        Some(access_config),
    )
}

impl Translator<'_, '_> {
    /// At op `index`: an instruction start tests a pending STOP, then defers an
    /// instruction the lowering does not complete. Returns whether the block
    /// ends here.
    pub(super) fn machine_instruction(&mut self, ops: &[IrOp], index: usize) -> bool {
        let IrOp::InsnStart { guest_addr } = ops[index] else {
            return false;
        };
        let machine = self.machine.as_mut().unwrap();
        let ordinal = if machine.started {
            machine.ordinal + 1
        } else {
            0
        };
        machine.started = true;
        machine.ordinal = ordinal;
        if machine.stored {
            let stop = self.builder.use_var(machine.stop);
            let stopped = self.builder.create_block();
            let next = self.builder.create_block();
            self.builder.set_cold_block(stopped);
            self.builder.ins().brif(stop, stopped, &[], next, &[]);
            self.builder.seal_block(stopped);
            self.builder.seal_block(next);
            self.builder.switch_to_block(stopped);
            self.machine_exit(RET_MACHINE_STOP, guest_addr, ordinal);
            self.builder.switch_to_block(next);
        }
        let body = &ops[index + 1..];
        let end = body
            .iter()
            .position(|op| matches!(op, IrOp::InsnStart { .. }))
            .unwrap_or(body.len());
        if abi::lowered_instruction(&body[..end]) {
            return false;
        }
        self.emit_insn_start(&guest_addr);
        self.machine_exit(RET_MACHINE_DEFER, guest_addr, ordinal);
        true
    }

    fn machine_exit(&mut self, code: u64, rip: u64, ordinal: u64) {
        let rip = self.iconst(rip);
        self.store_cpu(self.offsets.rip, rip);
        let ordinal = self.iconst(ordinal);
        self.store_mem(
            std::mem::offset_of!(MemCtx, machine_ordinal) as i32,
            ordinal,
        );
        self.ret(code);
    }

    /// A run-time defer of the current instruction, in a cold block that ends.
    fn machine_defer_if(&mut self, deferred: Value) {
        let ok = self.begin_trap_fork(deferred);
        let cold = self.builder.current_block().unwrap();
        self.builder.set_cold_block(cold);
        let ordinal = self.machine.as_ref().unwrap().ordinal | abi::HELPER_DEFER;
        self.machine_exit(RET_MACHINE_DEFER, self.cur_addr, ordinal);
        self.builder.switch_to_block(ok);
    }

    /// The Machine lowering of an op, or `None` for the shared lowering.
    pub(super) fn machine_op(&mut self, op: &IrOp) -> Option<bool> {
        let config = self.machine.as_ref().unwrap().config;
        match op {
            IrOp::Load {
                dst, addr, size, ..
            } => {
                let addr = self.val(*addr);
                let value = if config & abi::CONFIG_ROUTED == 0 {
                    self.machine_inline_load(addr, *size, config)
                } else {
                    self.machine_helper_load(addr, *size, config)
                };
                self.set(*dst, value);
                Some(false)
            }
            IrOp::Store {
                addr, src, size, ..
            } => {
                let addr = self.val(*addr);
                let value = self.val(*src);
                let stopped = if config & abi::CONFIG_ROUTED == 0 {
                    self.machine_inline_store(addr, value, *size, config)
                } else {
                    self.machine_helper_store(addr, value, *size, config)
                };
                let machine = self.machine.as_mut().unwrap();
                machine.stored = true;
                let stop = machine.stop;
                let before = self.builder.use_var(stop);
                let after = self.builder.ins().bor(before, stopped);
                self.builder.def_var(stop, after);
                Some(false)
            }
            // The whole-operand check at the end of the 32-bit linear space
            // (`interp::legacy::flat32_address`); the fault is the interpreter's.
            IrOp::Legacy(LegacyOp::Flat32Address {
                dst, offset, bytes, ..
            }) => {
                let offset = self.val(*offset);
                let last = u64::from(u32::MAX) - u64::from(bytes.saturating_sub(1));
                let deferred =
                    self.builder
                        .ins()
                        .icmp_imm_s(IntCC::UnsignedGreaterThan, offset, last as i64);
                self.machine_defer_if(deferred);
                self.set(*dst, offset);
                Some(false)
            }
            // The 486 alignment gate (decision-32 §6), emitted at CPL 3 with
            // CR0.AM set: #AC needs CPL 3, CR0.AM and EFLAGS.AC (SDM Vol 3A
            // §7.15, Event 17, p. 7-53), so with both set the instruction
            // defers before any effect and the interpreter decides it. AC
            // changes only through deferred instructions (POPF), but both
            // bits are read live. There is no CPL test: the lifter emits the
            // gate only at CPL 3 (`lift::legacy::lower`), and the unit's key
            // names its CPL, so it never runs at another.
            IrOp::Legacy(LegacyOp::I486Gate) => {
                use x86jit_core::CpuState;
                let legacy = std::mem::offset_of!(CpuState, legacy);
                let cr0 = self.builder.ins().uload32(
                    MemFlagsData::trusted(),
                    self.cpu,
                    (legacy + std::mem::offset_of!(x86jit_core::legacy::LegacyCpuState, cr0))
                        as i32,
                );
                let rflags = self.builder.ins().load(
                    types::I64,
                    MemFlagsData::trusted(),
                    self.cpu,
                    std::mem::offset_of!(CpuState, rflags_system) as i32,
                );
                let both = self.builder.ins().band(cr0, rflags);
                let ac = self.builder.ins().band_imm_s(both, 1 << 18);
                let deferred = self.builder.ins().icmp_imm_s(IntCC::NotEqual, ac, 0);
                self.machine_defer_if(deferred);
                Some(false)
            }
            // The live segment check of a Flat32 FS or GS operand (decision-32
            // §6): the passing subset of the interpreter's `legacy::access`
            // (SDM Vol 3A §§3.4.5.1, 6.3-6.4; 80386 PRM §§6.3.1.1-6.3.1.2
            // pp. 6-4-6). A usable expand-up data segment, writable for a
            // store, whose limit holds the whole operand, gives base + offset
            // mod 2^32; anything else defers, and the interpreter raises the
            // #GP or #SS. An operand that wraps past 4 GiB crosses a page and
            // defers at its access.
            IrOp::Legacy(LegacyOp::Address {
                dst,
                segment,
                offset,
                bytes,
                access,
            }) => {
                use x86jit_core::legacy::{LegacyCpuState, SegmentCache};
                use x86jit_core::CpuState;
                let cache = (std::mem::offset_of!(CpuState, legacy)
                    + std::mem::offset_of!(LegacyCpuState, segments)
                    + *segment as usize * std::mem::size_of::<SegmentCache>())
                    as i32;
                let field = |f: usize| cache + f as i32;
                let offset = self.val(*offset);
                let base = self.builder.ins().uload32(
                    MemFlagsData::trusted(),
                    self.cpu,
                    field(std::mem::offset_of!(SegmentCache, base)),
                );
                let limit = self.builder.ins().uload32(
                    MemFlagsData::trusted(),
                    self.cpu,
                    field(std::mem::offset_of!(SegmentCache, limit)),
                );
                let attributes = self.builder.ins().uload16(
                    types::I64,
                    MemFlagsData::trusted(),
                    self.cpu,
                    field(std::mem::offset_of!(SegmentCache, attributes)),
                );
                let unusable = self.builder.ins().uload8(
                    types::I64,
                    MemFlagsData::trusted(),
                    self.cpu,
                    field(std::mem::offset_of!(SegmentCache, unusable)),
                );
                // Type bits 3:1: code, expand-down, writable.
                let (mask, want) = if *access == x86jit_core::AccessKind::Write {
                    (0xe, 0x2)
                } else {
                    (0xc, 0)
                };
                let kind = self.builder.ins().band_imm_s(attributes, mask);
                let wrong = self.builder.ins().icmp_imm_s(IntCC::NotEqual, kind, want);
                let unusable = self.builder.ins().icmp_imm_s(IntCC::NotEqual, unusable, 0);
                let last = self
                    .builder
                    .ins()
                    .iadd_imm_s(offset, i64::from(bytes.saturating_sub(1)));
                let beyond = self
                    .builder
                    .ins()
                    .icmp(IntCC::UnsignedGreaterThan, last, limit);
                let refused = self.builder.ins().bor(wrong, unusable);
                let refused = self.builder.ins().bor(refused, beyond);
                self.machine_defer_if(refused);
                let linear = self.builder.ins().iadd(base, offset);
                let linear = self.builder.ins().band_imm_s(linear, 0xffff_ffff);
                self.set(*dst, linear);
                Some(false)
            }
            // A live instruction field (decision-30 §5): one load of its bytes
            // from the bus address the fetch translated, where the op stands.
            IrOp::Legacy(LegacyOp::LiveField {
                dst,
                bus,
                bytes,
                extend,
            }) => {
                let base = self.load_mem(MEMCTX_BASE);
                let guest_base = self.load_mem(MEMCTX_GUEST_BASE);
                let at = self.builder.ins().iadd_imm_s(base, *bus as i64);
                let host = self.builder.ins().isub(at, guest_base);
                let ty = int_ty(*bytes);
                let raw = self
                    .builder
                    .ins()
                    .load(ty, MemFlagsData::trusted(), host, 0);
                let value = if *extend == 0 {
                    self.builder.ins().uextend(types::I64, raw)
                } else {
                    let signed = self.builder.ins().sextend(types::I64, raw);
                    if *extend >= 64 {
                        signed
                    } else {
                        self.builder
                            .ins()
                            .band_imm_s(signed, ((1u64 << *extend) - 1) as i64)
                    }
                };
                self.set(*dst, value);
                Some(false)
            }
            // MOV r/m, Sreg: the selector, read live (decision-30 §3).
            IrOp::Legacy(LegacyOp::ReadSelector { dst, segment }) => {
                use x86jit_core::legacy::SegmentId;
                let index = match segment {
                    SegmentId::Es => 0,
                    SegmentId::Cs => 1,
                    SegmentId::Ss => 2,
                    SegmentId::Ds => 3,
                    SegmentId::Fs => 4,
                    SegmentId::Gs => 5,
                };
                self.machine_selector(*dst, index);
                Some(false)
            }
            IrOp::ReadSelector { dst, segment } => {
                self.machine_selector(*dst, *segment);
                Some(false)
            }
            IrOp::ReadReg { dst, reg } if reg.gpr_index().is_none() => {
                let index = match reg {
                    Reg::Es => 0,
                    Reg::Cs => 1,
                    Reg::Ss => 2,
                    Reg::Ds => 3,
                    _ => unreachable!("a lowered register read: {reg:?}"),
                };
                self.machine_selector(*dst, index);
                Some(false)
            }
            // A checkpoint saves the interpreter's rollback image. The
            // classifier admits an instruction holding one only when nothing
            // that can defer follows its first commit (decision-30 §3).
            IrOp::Legacy(LegacyOp::Checkpoint) => Some(false),
            // DIV and IDIV: the shared helper; #DE defers, and the
            // interpreter raises it (decision-30 §3).
            IrOp::Div {
                quot,
                rem,
                hi,
                lo,
                divisor,
                size,
                signed,
            } => {
                let (hi, lo, divisor) = (self.val(*hi), self.val(*lo), self.val(*divisor));
                let width = self.iconst(u64::from(*size));
                let signed = self.iconst(u64::from(*signed));
                let (slot, call) =
                    self.call_with_out_slot(self.helpers.div, &[hi, lo, divisor, width, signed]);
                let fault = self.builder.inst_results(call)[0];
                self.machine_defer_if(fault);
                let q = self
                    .builder
                    .ins()
                    .stack_load(types::I64, types::I64, slot, 0);
                let r = self
                    .builder
                    .ins()
                    .stack_load(types::I64, types::I64, slot, 8);
                self.set(*quot, q);
                self.set(*rem, r);
                Some(false)
            }
            // LOOP, LOOPE, LOOPNE, JECXZ: the count and the branch, as the
            // interpreter's `execute_loop`. The target cannot fault: CS's
            // limit is 0xFFFFFFFF (SDM Vol 2A LOOP/LOOPcc p. 3-571).
            IrOp::Legacy(LegacyOp::Loop {
                kind,
                count_bits,
                taken,
                next,
                ..
            }) => {
                use x86jit_core::ir::LoopKind;
                let width = *count_bits / 8;
                let mask = if width == 2 { 0xffff } else { 0xffff_ffff };
                let ecx = self.read_gpr(1);
                let count = self.builder.ins().band_imm_s(ecx, mask);
                let take = if matches!(kind, LoopKind::Jcxz) {
                    self.builder.ins().icmp_imm_s(IntCC::Equal, count, 0)
                } else {
                    let less = self.builder.ins().iadd_imm_s(count, -1);
                    let less = self.builder.ins().band_imm_s(less, mask);
                    let more = self.builder.ins().icmp_imm_s(IntCC::NotEqual, less, 0);
                    let take = match kind {
                        LoopKind::Loope => {
                            let zf = self.eval_cond(Cond::Eq);
                            self.builder.ins().band(more, zf)
                        }
                        LoopKind::Loopne => {
                            let nz = self.eval_cond(Cond::Ne);
                            self.builder.ins().band(more, nz)
                        }
                        _ => more,
                    };
                    self.write_gpr(1, less, width);
                    take
                };
                let tk = self.builder.create_block();
                let fl = self.builder.create_block();
                self.builder.ins().brif(take, tk, &[], fl, &[]);
                self.builder.seal_block(tk);
                self.builder.seal_block(fl);
                self.builder.switch_to_block(tk);
                let target = self.iconst(u64::from(*taken));
                self.store_cpu(self.offsets.rip, target);
                self.machine_chain();
                self.builder.switch_to_block(fl);
                let target = self.iconst(u64::from(*next));
                self.store_cpu(self.offsets.rip, target);
                self.machine_chain();
                Some(true)
            }
            // A near transfer in a Flat32 context: the 32-bit target, whose CS
            // limit check cannot fail (`interp::legacy::target`). It ends the
            // block. A direct target chains (decision-31 §3); a near CALL
            // pushes the return ring first, a near RET predicts through it,
            // and any other indirect target returns to the dispatcher's turn.
            IrOp::Jump { target: operand } => {
                let target = self.val(*operand);
                let target = self.builder.ins().band_imm_s(target, 0xffff_ffff);
                self.store_cpu(self.offsets.rip, target);
                let exit = self.machine.as_ref().unwrap().exit;
                if let abi::MachineExit::Call(return_addr) = exit {
                    let cont_slot = (self.alloc_slot)();
                    self.emit_ret_push(return_addr, cont_slot);
                }
                match (operand, exit) {
                    (Val::Imm(_), _) => self.machine_chain(),
                    (_, abi::MachineExit::Return) => self.emit_ret_predict(target),
                    _ => self.ret(RET_CONTINUE),
                }
                Some(true)
            }
            IrOp::Branch {
                cond,
                taken,
                fallthrough,
                ..
            } => {
                let c = self.eval_cond(*cond);
                let tk = self.builder.create_block();
                let fl = self.builder.create_block();
                self.builder.ins().brif(c, tk, &[], fl, &[]);
                self.builder.seal_block(tk);
                self.builder.seal_block(fl);
                self.builder.switch_to_block(tk);
                let target = self.iconst(*taken & 0xffff_ffff);
                self.store_cpu(self.offsets.rip, target);
                self.machine_chain();
                self.builder.switch_to_block(fl);
                let target = self.iconst(*fallthrough);
                self.store_cpu(self.offsets.rip, target);
                self.machine_chain();
                Some(true)
            }
            _ => None,
        }
    }

    /// A static exit (decision-31 §2.1): RIP holds the target; chain through
    /// a fresh link slot.
    fn machine_chain(&mut self) {
        let slot = (self.alloc_slot)();
        self.chain_or_link(slot);
    }

    /// The J1 load: the checked helper, a defer when it refuses.
    fn machine_helper_load(&mut self, addr: Value, size: u8, config: u64) -> Value {
        let ip = self.iconst(self.cur_addr);
        self.store_mem(std::mem::offset_of!(MemCtx, machine_value) as i32, ip);
        let size = self.iconst(u64::from(size));
        let config = self.iconst(config);
        let call = self.call_memory_helper(
            self.helpers.machine_load,
            &[self.cpu, self.mem, addr, size, config],
            Some(GuestAccessKind::Read),
        );
        let status = self.builder.inst_results(call)[0];
        let deferred =
            self.builder
                .ins()
                .icmp_imm_s(IntCC::NotEqual, status, abi::ACCESS_DONE as i64);
        self.machine_defer_if(deferred);
        self.load_mem(std::mem::offset_of!(MemCtx, machine_value) as i32)
    }

    /// The J1 store: the checked helper, a defer when it refuses. Returns an
    /// I8 that is set when the helper reported `ACCESS_STOP`.
    fn machine_helper_store(&mut self, addr: Value, value: Value, size: u8, config: u64) -> Value {
        let ip = self.iconst(self.cur_addr);
        self.store_mem(std::mem::offset_of!(MemCtx, machine_value) as i32, ip);
        let size = self.iconst(u64::from(size));
        let config = self.iconst(config);
        let call = self.call_memory_helper(
            self.helpers.machine_store,
            &[self.cpu, self.mem, addr, size, value, config],
            Some(GuestAccessKind::Write),
        );
        let status = self.builder.inst_results(call)[0];
        let deferred =
            self.builder
                .ins()
                .icmp_imm_s(IntCC::Equal, status, abi::ACCESS_DEFER as i64);
        self.machine_defer_if(deferred);
        self.builder
            .ins()
            .icmp_imm_s(IntCC::Equal, status, abi::ACCESS_STOP as i64)
    }

    /// Decision-30 §4: the slot of `addr`'s page in the vCPU's compiled TLB
    /// view, and whether `tag_offset`'s tag names that page with the whole
    /// `size`-byte access inside it.
    fn machine_probe(&mut self, addr: Value, size: u8, tag_offset: i32) -> (Value, Value) {
        let page = self.builder.ins().ushr_imm_s(addr, 12);
        let index = self
            .builder
            .ins()
            .band_imm_s(page, (abi::TLB_SLOTS - 1) as i64);
        let offset = self.builder.ins().ishl_imm_s(index, abi::ENTRY_SHIFT);
        let tlb = self.load_mem(std::mem::offset_of!(MemCtx, ret_stack) as i32);
        let entry = self.builder.ins().iadd(tlb, offset);
        let tag = self
            .builder
            .ins()
            .uload32(MemFlagsData::trusted(), entry, tag_offset);
        let want = self.builder.ins().iadd_imm_s(page, 1);
        let hit = self.builder.ins().icmp(IntCC::Equal, tag, want);
        let within = self.builder.ins().band_imm_s(addr, 0xfff);
        let inside = self.builder.ins().icmp_imm_s(
            IntCC::UnsignedLessThanOrEqual,
            within,
            0x1000 - i64::from(size),
        );
        let ok = self.builder.ins().band(hit, inside);
        (entry, ok)
    }

    /// A load through the view, the helper on a miss or a page-crossing access.
    fn machine_inline_load(&mut self, addr: Value, size: u8, config: u64) -> Value {
        self.sync_guest_fault_state();
        let tag = self.machine.as_ref().unwrap().tags.0;
        let (entry, ok) = self.machine_probe(addr, size, tag);
        let fast = self.builder.create_block();
        let slow = self.builder.create_block();
        let join = self.builder.create_block();
        self.builder.append_block_param(join, types::I64);
        self.builder.set_cold_block(slow);
        self.builder.ins().brif(ok, fast, &[], slow, &[]);
        self.builder.seal_block(fast);
        self.builder.seal_block(slow);
        self.builder.switch_to_block(fast);
        let addend = self.builder.ins().load(
            types::I64,
            MemFlagsData::trusted(),
            entry,
            abi::ENTRY_ADDEND,
        );
        let host = self.builder.ins().iadd(addend, addr);
        let value = self.load_guest(host, size);
        self.builder.ins().jump(join, &[value.into()]);
        self.builder.switch_to_block(slow);
        let value = self.machine_helper_load(addr, size, config);
        self.builder.ins().jump(join, &[value.into()]);
        self.builder.seal_block(join);
        self.builder.switch_to_block(join);
        self.builder.block_params(join)[0]
    }

    /// A store through the view, then the store barrier: the frame's live
    /// code-tag bit, or any watched page, calls `machine_note`, whose status
    /// may stop the block after this instruction. The helper on a miss.
    fn machine_inline_store(&mut self, addr: Value, value: Value, size: u8, config: u64) -> Value {
        self.sync_guest_fault_state();
        let tag = self.machine.as_ref().unwrap().tags.1;
        let (entry, ok) = self.machine_probe(addr, size, tag);
        let fast = self.builder.create_block();
        let slow = self.builder.create_block();
        let note = self.builder.create_block();
        let join = self.builder.create_block();
        self.builder.append_block_param(join, types::I8);
        self.builder.set_cold_block(slow);
        self.builder.set_cold_block(note);
        self.builder.ins().brif(ok, fast, &[], slow, &[]);
        self.builder.seal_block(fast);
        self.builder.seal_block(slow);
        self.builder.switch_to_block(fast);
        let addend = self.builder.ins().load(
            types::I64,
            MemFlagsData::trusted(),
            entry,
            abi::ENTRY_ADDEND,
        );
        let host = self.builder.ins().iadd(addend, addr);
        self.store_guest(host, value, size);
        // Code tag: the frame's live bit, read after the store as the
        // Application barrier reads its watermark.
        let word_ptr = self.builder.ins().load(
            types::I64,
            MemFlagsData::trusted(),
            entry,
            abi::ENTRY_TAG_WORD,
        );
        let word = self
            .builder
            .ins()
            .atomic_load(types::I64, MemFlagsData::trusted(), word_ptr);
        let bit = self
            .builder
            .ins()
            .uload32(MemFlagsData::trusted(), entry, abi::ENTRY_TAG_BIT);
        let shifted = self.builder.ins().ushr(word, bit);
        let code = self.builder.ins().band_imm_s(shifted, 1);
        let watch_ptr = self.load_mem(MEMCTX_WATCH_COUNT_PTR);
        let watched = self
            .builder
            .ins()
            .load(types::I64, MemFlagsData::trusted(), watch_ptr, 0);
        let either = self.builder.ins().bor(code, watched);
        let clear = self.builder.ins().iconst(types::I8, 0);
        self.builder
            .ins()
            .brif(either, note, &[], join, &[clear.into()]);
        self.builder.seal_block(note);
        self.builder.switch_to_block(note);
        let frame = self
            .builder
            .ins()
            .uload32(MemFlagsData::trusted(), entry, abi::ENTRY_FRAME);
        let base = self.builder.ins().ishl_imm_s(frame, 12);
        let within = self.builder.ins().band_imm_s(addr, 0xfff);
        let physical = self.builder.ins().bor(base, within);
        let width = self.iconst(u64::from(size));
        let call = self.call_helper(
            self.helpers.machine_note,
            &[self.cpu, self.mem, physical, width],
        );
        let status = self.builder.inst_results(call)[0];
        let stopped = self
            .builder
            .ins()
            .icmp_imm_s(IntCC::Equal, status, abi::ACCESS_STOP as i64);
        self.builder.ins().jump(join, &[stopped.into()]);
        self.builder.switch_to_block(slow);
        let stopped = self.machine_helper_store(addr, value, size, config);
        self.builder.ins().jump(join, &[stopped.into()]);
        self.builder.seal_block(join);
        self.builder.switch_to_block(join);
        self.builder.block_params(join)[0]
    }

    /// A selector register (ES, CS, SS, DS, FS, GS by `index`), read live.
    fn machine_selector(&mut self, dst: u32, index: u8) {
        use x86jit_core::CpuState;
        let offset = match index {
            0 => std::mem::offset_of!(CpuState, es),
            1 => std::mem::offset_of!(CpuState, cs),
            2 => std::mem::offset_of!(CpuState, ss),
            3 => std::mem::offset_of!(CpuState, ds),
            4 => std::mem::offset_of!(CpuState, fs),
            _ => std::mem::offset_of!(CpuState, gs),
        } as i32;
        let value =
            self.builder
                .ins()
                .uload16(types::I64, MemFlagsData::trusted(), self.cpu, offset);
        self.set(dst, value);
    }
}
