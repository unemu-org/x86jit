//! The Machine lowering of the shared translator (decision-29, J1).
//!
//! A compiled Machine block completes a prefix of its block and returns where
//! the next instruction starts: `RET_CONTINUE` after the whole block,
//! `RET_MACHINE_DEFER` before an instruction it does not complete (nothing of
//! which is committed), `RET_MACHINE_STOP` after a store that left the
//! dispatch word set. The completed count goes to `MemCtx::machine_ordinal`,
//! baked at the exit site; the dispatcher charges it. Loads and stores call
//! the `jit_abi::machine` helpers, which run the interpreter's checked access.
use super::*;
use x86jit_core::jit_abi::machine as abi;
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
}

impl MachineLowering {
    pub(crate) fn new(builder: &mut FunctionBuilder, config: u64) -> Self {
        let stop = builder.declare_var(types::I8);
        let clear = builder.ins().iconst(types::I8, 0);
        builder.def_var(stop, clear);
        Self {
            config,
            ordinal: 0,
            started: false,
            stop,
            stored: false,
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
                let size = self.iconst(u64::from(*size));
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
                let value = self.load_mem(std::mem::offset_of!(MemCtx, machine_value) as i32);
                self.set(*dst, value);
                Some(false)
            }
            IrOp::Store {
                addr, src, size, ..
            } => {
                let addr = self.val(*addr);
                let value = self.val(*src);
                let size = self.iconst(u64::from(*size));
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
                let stopped =
                    self.builder
                        .ins()
                        .icmp_imm_s(IntCC::Equal, status, abi::ACCESS_STOP as i64);
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
            // A near transfer in a Flat32 context: the 32-bit target, whose CS
            // limit check cannot fail (`interp::legacy::target`). It ends the
            // block; the dispatcher's turn follows its successor.
            IrOp::Jump { target, .. } => {
                let target = self.val(*target);
                let target = self.builder.ins().band_imm_s(target, 0xffff_ffff);
                self.store_cpu(self.offsets.rip, target);
                self.ret(RET_CONTINUE);
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
                self.ret(RET_CONTINUE);
                self.builder.switch_to_block(fl);
                let target = self.iconst(*fallthrough);
                self.store_cpu(self.offsets.rip, target);
                self.ret(RET_CONTINUE);
                Some(true)
            }
            _ => None,
        }
    }
}
