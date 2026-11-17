//! The NMI input's cold helpers (spec.md §17.10): acceptance, blocking, the
//! boundary's NMI step and the IRET checks. The dispatcher calls them only
//! when an edge is latched or NMI is blocked.
use super::*;

/// `Vcpu::nmi_state`: blocked from NMI acceptance to the next IRET
/// (I80286, I80386), and one edge remembered meanwhile (I80286).
pub(in crate::vm) const NMI_BLOCKED: u8 = 1;
pub(in crate::vm) const NMI_REMEMBERED: u8 = 2;

/// Whether the instruction marked at `ip` is an IRET (its lowering ends with
/// `LegacyOp::Iret`, after the task-return check). Cold: read only while NMI
/// is blocked.
#[cold]
#[inline(never)]
fn iret_at(ir: &crate::ir::IrBlock, ip: u64) -> bool {
    let mut current = false;
    for op in &ir.ops {
        match op {
            IrOp::InsnStart { guest_addr } => current = *guest_addr == ip,
            IrOp::Legacy(crate::legacy::LegacyOp::Iret { .. }) if current => return true,
            _ => {}
        }
    }
    false
}

/// Whether a walk that retired `retired` instructions from the start of `ir`
/// retired an IRET. IRET is a control transfer, so it ends its block; it
/// retired when every instruction of the block did.
#[cold]
#[inline(never)]
fn iret_retired(ir: &crate::ir::IrBlock, retired: u64) -> bool {
    let mut count = 0;
    let mut last = 0;
    for op in &ir.ops {
        if let IrOp::InsnStart { guest_addr } = op {
            count += 1;
            last = *guest_addr;
        }
    }
    retired != 0 && retired == count && iret_at(ir, last)
}

/// The 8086 unit that ran before a single-step trap took a software
/// interrupt: INT n, INT 3, or an INTO whose OF test passed (OF is unchanged
/// by INTO). Only the first instruction of the block runs under TF.
#[cold]
#[inline(never)]
pub(super) fn took_software_interrupt20(ir: &crate::ir::IrBlock, cpu: &CpuState) -> bool {
    for op in ir.ops.iter().skip(1) {
        match op {
            IrOp::InsnStart { .. } => return false,
            IrOp::Legacy(crate::legacy::LegacyOp::Interrupt20 { overflow_only, .. }) => {
                return !overflow_only || cpu.flags.of
            }
            _ => {}
        }
    }
    false
}
impl Vcpu {
    /// The catalogue's NMI nesting policy (spec.md §17.10): a system model's
    /// generation entry, or a twenty-bit model's own entry.
    pub(in crate::vm) fn nmi_nesting(&self) -> crate::NmiNesting {
        if self.twenty_bit_model() {
            return self
                .cpu_model
                .expect("twenty-bit model")
                .profile()
                .quirks
                .nmi_nesting;
        }
        crate::model::profile_for_generation(
            self.legacy_model.expect("Machine has a model").generation,
        )
        .quirks
        .nmi_nesting
    }

    /// The dispatcher's NMI predicate without the latch test (spec.md §17.10):
    /// the walk's `nmi_gate` plus the task continuation. Shutdown, and the NEC
    /// 8080-mode BRK refusal, are decided by the boundary step before it.
    #[inline]
    pub(in crate::vm) fn nmi_gate(&self, task_pending: bool) -> bool {
        crate::interp::nmi_gate(
            self.nmi_state & NMI_BLOCKED != 0,
            self.ss_shadow & NMI_HOLD != 0,
            self.machine_pending_io(task_pending),
        )
    }

    // Acceptance (spec.md §17.10): the consuming RMW selects the edge, so
    // none is lost or delivered twice. Blocking starts here, before delivery
    // (I286 PRM §9.6.2 p. 9-10; SDM Vol 3A §7.7.2 p. 7-6).
    pub(super) fn accept_nmi(&mut self) -> bool {
        let prev = self
            .exit_request
            .word
            .fetch_and(!NMI_EDGE, Ordering::Acquire);
        if prev & NMI_EDGE == 0 {
            return false;
        }
        if self.nmi_nesting() != crate::NmiNesting::Nested {
            self.nmi_state |= NMI_BLOCKED;
        }
        true
    }

    // The end of NMI blocking (spec.md §17.10). An I80286 edge remembered
    // meanwhile is latched again, "serviced after the first IRET" (I286 PRM
    // §9.2 p. 9-3), and one still in the word stays latched; an I80386 edge
    // raised while blocked is ignored (80386 PRM §9.2.1 p. 9-3), including
    // one still in the word. That also drops an edge raised between the
    // IRET's retirement and this call, the engine's granularity (ASSUMED).
    pub(super) fn nmi_unblock(&mut self) {
        let state = std::mem::take(&mut self.nmi_state);
        if state & NMI_BLOCKED == 0 {
            return;
        }
        if self.nmi_nesting() == crate::NmiNesting::BlockedIgnore {
            self.exit_request
                .word
                .fetch_and(!NMI_EDGE, Ordering::Relaxed);
        } else if state & NMI_REMEMBERED != 0 {
            self.exit_request.word.fetch_or(NMI_EDGE, Ordering::Release);
        }
    }

    // An accepted NMI that never entered the guest leaves no blocking behind
    // (spec.md §17.10). `relatch` puts its edge back, and an NMI taken out of
    // shutdown back into it, as if it had not been accepted. The exception
    // delivered serially after a faulting NMI delivery carries that NMI's
    // blocking: when it never enters the guest either, no handler will IRET,
    // so blocking ends; the NMI itself was lost at the fault (386 PRM Table
    // 9-6 note 3 p. 9-20) and is not latched again.
    pub(super) fn undo_nmi(&mut self, event: &PendingEvent, relatch: bool) {
        match event.source {
            EventSource::Nmi { shutdown } => {
                self.nmi_unblock();
                if relatch {
                    self.exit_request.word.fetch_or(NMI_EDGE, Ordering::Release);
                    self.legacy_execution.as_mut().unwrap().shutdown |= shutdown;
                }
            }
            _ if event.after_nmi => self.nmi_unblock(),
            _ => {}
        }
    }

    // The dispatch word holds a host request or an NMI edge. `Err`: return
    // the request. `Ok(true)`: a latched NMI for the boundary's NMI step.
    // While blocked, the edge leaves the word here: the I80286 remembers one
    // (I286 PRM §9.2 p. 9-3), the I80386 ignores it (80386 PRM §9.2.1
    // p. 9-3), so no boundary test keeps tripping on it (spec.md §17.10).
    #[cold]
    #[inline(never)]
    pub(super) fn machine_external(&mut self, pending_io: bool) -> Result<bool, Exit> {
        let word = self.exit_request.word.load(Ordering::Relaxed);
        if word & EXIT_REQUESTED != 0 && !pending_io {
            return Err(self.interrupted(crate::FaultPath::Interpreter, false));
        }
        if word & NMI_EDGE == 0 {
            return Ok(false);
        }
        if self.nmi_state & NMI_BLOCKED == 0 {
            return Ok(true);
        }
        let prev = self
            .exit_request
            .word
            .fetch_and(!NMI_EDGE, Ordering::Acquire);
        if prev & NMI_EDGE != 0 && self.nmi_nesting() == crate::NmiNesting::BlockedRemember {
            self.nmi_state |= NMI_REMEMBERED;
        }
        Ok(false)
    }

    // Boundary steps 4a, 4b and 5 (spec.md §17.10), with an NMI edge seen and
    // no owned continuation. `None`: no NMI here; the IRQ step follows.
    // `Some(Ok)`: delivered, with the context derived again after the CS
    // reload; delivery costs no instruction unit.
    #[cold]
    #[inline(never)]
    pub(super) fn machine_nmi<const ROUTED: bool, const ORDER: u8>(
        &mut self,
        vm: &Vm,
        context: MachineContext,
        task_pending: bool,
    ) -> Option<Result<MachineContext, Exit>> {
        let shutdown = self.legacy_execution.as_ref().unwrap().shutdown;
        if self.twenty_bit_model() && self.cpu.legacy.reserved & crate::legacy::NEC_EMULATION != 0 {
            if !matches!(
                self.cpu_model,
                Some(crate::CpuModel::NecV20 | crate::CpuModel::NecV30)
            ) {
                // The 8080-mode boundary refuses the mode byte.
                return None;
            }
            // 4b: single-step in 8080 mode is not described; an NMI
            // delivered here would hide the refusal. The latch is kept.
            if self.cpu.rflags_system & 0x100 != 0 {
                return Some(Err(Exit::UnsupportedSystemContext {
                    addr: self.cpu.rip,
                    operation: UnsupportedSystemOperation::NecEmulationSingleStep,
                }));
            }
        }
        if !(self.nmi_gate(task_pending) && self.accept_nmi()) {
            // 4a: nothing else executes in shutdown.
            return shutdown.then_some(Err(Exit::Shutdown { addr: self.cpu.rip }));
        }
        // I286 PRM §9.6.2 p. 9-10: NMI forces the CPU out of shutdown; the
        // saved CS:IP is the one held at shutdown (ASSUMED).
        self.legacy_execution.as_mut().unwrap().shutdown = false;
        Some(
            self.deliver_legacy_event::<ROUTED, ORDER>(
                vm,
                context,
                PendingEvent {
                    irq_consumed: false,
                    software: false,
                    after_nmi: false,
                    source: EventSource::Nmi { shutdown },
                    ip: self.cpu.rip as u32,
                    cs: self.cpu.cs,
                },
            )
            .and_then(|()| self.legacy_context_for_run()),
        )
    }

    // The fast arm's cold epilogue: a queued A20 change applies, and while NMI
    // is blocked an IRET the unit retired ends blocking. True when either
    // happened.
    #[cold]
    #[inline(never)]
    pub(super) fn machine_epilogue(&mut self, ir: &crate::ir::IrBlock, retired: u64) -> bool {
        let unblocked = self.nmi_after_unit(ir, retired);
        let a20 = self
            .legacy_execution
            .as_ref()
            .unwrap()
            .pending_a20_enabled
            .is_some();
        if a20 {
            self.apply_pending_a20();
        }
        unblocked || a20
    }

    // After a unit while NMI is blocked: an IRET it retired ends blocking
    // (I286 PRM §9.2 p. 9-3 "the first IRET"; 80386 PRM §9.2.1 p. 9-3).
    // True when it did, so the next boundary is derived again.
    #[cold]
    #[inline(never)]
    pub(super) fn nmi_after_unit(&mut self, ir: &crate::ir::IrBlock, retired: u64) -> bool {
        let unblock = self.nmi_state & NMI_BLOCKED != 0 && iret_retired(ir, retired);
        if unblock {
            self.nmi_unblock();
        }
        unblock
    }

    // A faulting IRET ends blocking too (SDM Vol 3A §7.7.2 p. 7-6 for the
    // I80386; ASSUMED for the I80286). A bus handoff inside IRET does not; its
    // completion does. A refused IRET has no effect.
    #[cold]
    #[inline(never)]
    pub(super) fn nmi_after_fault(&mut self, ir: &crate::ir::IrBlock, exit: &Exit) {
        if self.nmi_state & NMI_BLOCKED != 0
            && matches!(exit, Exit::Exception { .. })
            && iret_at(
                ir,
                exit.fault_metadata()
                    .map_or(self.cpu.rip, |m| m.saved_rip()),
            )
        {
            self.nmi_unblock();
        }
    }
}
