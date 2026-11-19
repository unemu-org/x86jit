//! The NMI input's cold helpers (spec.md §17.10): acceptance, blocking, the
//! boundary's NMI step and the IRET checks. The Machine dispatcher's loop
//! holds no NMI decision: it calls these when its dispatch word holds a host
//! request or an edge, after a unit while NMI is blocked, and to deliver a
//! post-unit single-step trap.
use super::*;

// Every twenty-bit catalogue entry nests NMIs (spec.md §17.10), so a vCPU on
// the twenty-bit dispatcher is never blocked: that dispatcher gives its walk
// `false` for blocking, which equals its own predicate, and watches for no
// IRET. The run entries select that dispatcher by the model's generation
// (`CpuGeneration::I8086`); a model of that generation that blocks fails this
// build. The match names every `CpuModel` with no wildcard, so a new model
// fails the build here until it is added to the list below.
const fn nests_if_twenty_bit(model: crate::CpuModel) -> bool {
    match model {
        crate::CpuModel::I80286
        | crate::CpuModel::I80386
        | crate::CpuModel::I8088
        | crate::CpuModel::I8086
        | crate::CpuModel::NecV20
        | crate::CpuModel::NecV30 => {
            !matches!(model.legacy_model().generation, crate::CpuGeneration::I8086)
                || matches!(
                    model.profile().quirks.nmi_nesting,
                    crate::NmiNesting::Nested
                )
        }
    }
}
const _: () = {
    let models = [
        crate::CpuModel::I80286,
        crate::CpuModel::I80386,
        crate::CpuModel::I8088,
        crate::CpuModel::I8086,
        crate::CpuModel::NecV20,
        crate::CpuModel::NecV30,
    ];
    let mut i = 0;
    while i < models.len() {
        assert!(nests_if_twenty_bit(models[i]));
        i += 1;
    }
};

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
fn took_software_interrupt20(ir: &crate::ir::IrBlock, cpu: &CpuState) -> bool {
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
/// What the dispatcher's out-of-line host-request and NMI decision leaves to
/// its loop (`Vcpu::machine_external`).
pub(super) enum External {
    /// Continue the boundary unchanged.
    Proceed,
    /// A latched NMI not delivered here: the service steps run and one unit
    /// is executed, until it is.
    OneStep,
    /// An INS completion ran with an NMI latched and not delivered: continue
    /// the service steps with this boundary, derived again after it, and this
    /// bus proof.
    Serviced(MachineBoundary, bool),
    /// An NMI was delivered; the loop derives the context again after the CS
    /// reload, and the order runs again before the handler's first
    /// instruction.
    Delivered,
    /// Return this exit.
    Stop(Exit),
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

    // The dispatch word holds a host request or an NMI edge: boundary steps 2
    // to 5 of spec.md §17.10 for a latched NMI, out of the dispatcher loop.
    // A host request when no completion is pending stops the run, the latch
    // kept. While NMI is blocked, the edge leaves the word here: the I80286
    // remembers one (I286 PRM §9.2 p. 9-3), the I80386 ignores it (80386 PRM
    // §9.2.1 p. 9-3), so no boundary test keeps tripping on it. Otherwise the
    // service steps that precede the NMI step run here, then the NMI step
    // after an owned continuation; an NMI not delivered forces one-step mode
    // until it is. The loop's own service steps follow, and the ones run here
    // leave them nothing to do.
    #[cold]
    #[inline(never)]
    pub(super) fn machine_external<
        const ROUTED: bool,
        const ORDER: u8,
        const SINGLE: bool,
        const KIND: u8,
    >(
        &mut self,
        vm: &Vm,
        options: StepOptions,
        context: MachineContext,
        task_pending: bool,
        pending_io: bool,
        mut direct_bus: bool,
    ) -> External {
        // A one-step budget's boundary leaves I/O eligibility to this query.
        let pending_io = if SINGLE {
            self.machine_pending_io(task_pending)
        } else {
            pending_io
        };
        let word = self.exit_request.word.load(Ordering::Relaxed);
        if word & EXIT_REQUESTED != 0 && !pending_io {
            return External::Stop(self.interrupted(crate::FaultPath::Interpreter, false));
        }
        if word & NMI_EDGE == 0 {
            return External::Proceed;
        }
        if self.nmi_state & NMI_BLOCKED != 0 {
            let prev = self
                .exit_request
                .word
                .fetch_and(!NMI_EDGE, Ordering::Acquire);
            if prev & NMI_EDGE != 0 && self.nmi_nesting() == crate::NmiNesting::BlockedRemember {
                self.nmi_state |= NMI_REMEMBERED;
            }
            return External::Proceed;
        }
        // The completion ends the boundary derived before it; the loop's
        // service block then runs on with the boundary derived again.
        let mut serviced = None;
        if self.cpu.port_string_in.is_some() {
            let mut boundary = MachineBoundary::default();
            let model = self.legacy_model.unwrap_or(LegacyCpuModel {
                generation: CpuGeneration::I286,
                model_semantics_id: 0,
            });
            if let Err(exit) = self.finish_port_string::<ROUTED, ORDER, SINGLE, KIND>(
                vm,
                options,
                crate::legacy::physical_mask(model.generation),
                &mut boundary,
                &mut direct_bus,
            ) {
                return External::Stop(exit);
            }
            boundary.service = true;
            serviced = Some(boundary);
        }
        self.queue_port_debug(task_pending);
        if self.exit_request.word.load(Ordering::Relaxed) & NMI_EDGE != 0
            && self
                .legacy_execution
                .as_ref()
                .unwrap()
                .event_retry
                .is_none()
        {
            match self.machine_nmi::<ROUTED, ORDER>(vm, context, task_pending) {
                None => {}
                Some(Ok(())) => return External::Delivered,
                Some(Err(exit)) => return External::Stop(exit),
            }
        }
        match serviced {
            Some(boundary) => External::Serviced(boundary, direct_bus),
            // A latched NMI not delivered forces one-step mode until it is.
            None => External::OneStep,
        }
    }

    // Boundary steps 4a, 4b and 5 (spec.md §17.10), with an NMI edge seen and
    // no owned continuation. `None`: no NMI here; the IRQ step follows.
    // `Some(Ok)`: delivered; the caller derives the context again after the
    // CS reload. Delivery costs no instruction unit.
    #[cold]
    #[inline(never)]
    pub(super) fn machine_nmi<const ROUTED: bool, const ORDER: u8>(
        &mut self,
        vm: &Vm,
        context: MachineContext,
        task_pending: bool,
    ) -> Option<Result<(), Exit>> {
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
        Some(self.deliver_legacy_event::<ROUTED, ORDER>(
            vm,
            context,
            PendingEvent {
                irq_consumed: false,
                software: false,
                after_nmi: false,
                source: EventSource::Nmi { shutdown },
                ip: self.cpu.rip as u32,
                cs: self.cpu.cs,
                origin: self.delivery_origin(self.cpu.cs, self.cpu.rip as u32),
            },
        ))
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

    // The slow arm's cold epilogue: while NMI is blocked, an IRET the unit
    // retired, or a faulting IRET, ends blocking.
    #[cold]
    #[inline(never)]
    pub(super) fn nmi_after_progress(
        &mut self,
        ir: &crate::ir::IrBlock,
        progress: &crate::interp::legacy::Progress,
    ) {
        match &progress.outcome {
            Some(Err(exit)) => self.nmi_after_fault(ir, exit),
            _ => {
                self.nmi_after_unit(ir, progress.retired);
            }
        }
    }

    // The post-instruction single-step trap of a unit run with TF (I286 PRM
    // §9.7.1 p. 9-14; 80386 PRM §12.3.1.4 p. 12-8; 8086 UM §2.6 p. 2-28),
    // with the context derived again after the CS reload. On the twenty-bit
    // models the trap records whether the unit took a software interrupt
    // (AP-67 p. A-28), which decides the NMI frame order there.
    #[cold]
    #[inline(never)]
    pub(super) fn deliver_step_trap<const ROUTED: bool, const ORDER: u8, const KIND: u8>(
        &mut self,
        vm: &Vm,
        ir: &crate::ir::IrBlock,
    ) -> Result<MachineContext, Exit> {
        let event = PendingEvent {
            irq_consumed: false,
            software: KIND == crate::vm::DIRECT20_DISPATCH
                && took_software_interrupt20(ir, &self.cpu),
            after_nmi: false,
            source: EventSource::SingleStep,
            ip: self.cpu.rip as u32,
            cs: self.cpu.cs,
            origin: self.delivery_origin(self.cpu.cs, self.cpu.rip as u32),
        };
        let context = self.legacy_context_for_run()?;
        self.deliver_legacy_event::<ROUTED, ORDER>(vm, context, event)?;
        self.legacy_context_for_run()
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
