//! NEC V20/V30 native side of 8080 emulation mode (decision-23 §3 and C4a;
//! spec.md §17.10). The engine executes no 8080 instruction. At a boundary
//! with PSW.MD = 0 it returns [`Exit::I8080Mode`]; the embedder runs an 8080
//! core and hands the state back. RETEM and CALLN, which the 8080 stream
//! carries behind opcode EDh (NEC V20/V30 UM §8 p. 8-1), are decoded here.
use super::*;
use crate::legacy::{NEC_EMULATION, NEC_MD_WRITE, NEC_STANDBY};
use crate::{I8080Entry, I8080Handback, I8080HandbackError, I8080View};

impl Vcpu {
    #[inline]
    fn nec_model(&self) -> bool {
        matches!(
            self.cpu_model,
            Some(crate::CpuModel::NecV20 | crate::CpuModel::NecV30)
        )
    }

    fn i8080_view(&self) -> I8080View {
        I8080View {
            a: self.cpu.gpr[0] as u8,
            // NEC UM §8 Fig. 8-2 p. 8-1: S Z 0 AC 0 P 1 CY.
            flags: (self.cpu.flags.to_flags16() as u8 & 0xd5) | 0x02,
            bc: self.cpu.gpr[1] as u16,
            de: self.cpu.gpr[2] as u16,
            hl: self.cpu.gpr[3] as u16,
            sp: self.cpu.gpr[5] as u16,
            pc: self.cpu.rip as u16,
            interrupt_enable: self.cpu.flags.if_,
            ei_delay: self.sti_shadow,
            halted: self.cpu.legacy.reserved & NEC_STANDBY != 0,
            code_segment: self.cpu.cs,
            data_segment: self.cpu.ds,
        }
    }

    /// Write back the state of an 8080 core that ran on the view of the last
    /// [`Exit::I8080Mode`] (decision-23 C4a). Writes AL (AH is kept), BX, CX,
    /// DX, BP, IP, the five 8080 flags of FLAGS 7:0, IF, the EI delay
    /// (verbatim, as the STI shadow) and the standby latch. Never writes AH,
    /// SP, SI, DI, a segment register, V, DIR, BRK, MD or the MD write-enable
    /// latch. `how` says how the 8080 core stopped. Adds no step and no
    /// retired instruction; a rejected hand-back changes nothing. A hand-back
    /// is rejected while a frame access of RETEM, CALLN or an interrupt taken
    /// in 8080 mode waits for its MMIO completion
    /// ([`I8080HandbackError::InFlight`]).
    pub fn hand_back_i8080(
        &mut self,
        view: &I8080View,
        how: I8080Handback,
    ) -> Result<(), I8080HandbackError> {
        if !self.nec_model() || self.cpu.legacy.reserved & NEC_EMULATION == 0 {
            return Err(I8080HandbackError::NotInI8080Mode);
        }
        if self.cpu.mmio_retry_active
            || self.cpu.pending_mmio.is_some()
            || self.cpu.pending_mmio_write
        {
            return Err(I8080HandbackError::InFlight);
        }
        if view.code_segment != self.cpu.cs || view.data_segment != self.cpu.ds {
            return Err(I8080HandbackError::SegmentChanged);
        }
        if view.halted && !matches!(how, I8080Handback::Boundary) {
            return Err(I8080HandbackError::HaltedStop);
        }
        if matches!(how, I8080Handback::Refused { opcode: 0xed }) {
            return Err(I8080HandbackError::RefusedEscape);
        }
        self.cpu.write_gpr(0, u64::from(view.a), 1);
        self.cpu.write_gpr(1, u64::from(view.bc), 2);
        self.cpu.write_gpr(2, u64::from(view.de), 2);
        self.cpu.write_gpr(3, u64::from(view.hl), 2);
        self.cpu.write_gpr(5, u64::from(view.sp), 2);
        self.cpu.rip = u64::from(view.pc);
        let flags = (self.cpu.flags.to_flags16() & !0x00d5) | (u16::from(view.flags) & 0x00d5);
        self.cpu.flags.set_flags16(flags);
        // The 8080 EI and DI act on IE (ASSUMED, contract draft S4).
        self.cpu.flags.if_ = view.interrupt_enable;
        self.sti_shadow = view.ei_delay;
        self.ss_shadow = 0;
        if view.halted {
            self.cpu.legacy.reserved |= NEC_STANDBY;
        } else {
            self.cpu.legacy.reserved &= !NEC_STANDBY;
        }
        let execution = self.legacy_execution.as_mut().unwrap();
        execution.nec_escape = None;
        execution.nec_refusal = None;
        match how {
            I8080Handback::EdEscape => execution.nec_escape = Some((view.code_segment, view.pc)),
            I8080Handback::Boundary => {}
            I8080Handback::Refused { opcode } => {
                execution.nec_refusal = Some((view.code_segment, view.pc, opcode));
            }
        }
        Ok(())
    }

    /// The 8080-mode boundary, after budget, host request, IRQ delivery and
    /// the readiness stop (decision-16 §2 order). `Ok` means RETEM or CALLN
    /// completed as one unit; anything else stops the call.
    pub(super) fn nec_boundary<const ROUTED: bool, const ORDER: u8>(
        &mut self,
        vm: &Vm,
    ) -> Result<(), Exit> {
        if !self.nec_model() {
            // A nonzero mode byte on another twenty-bit model.
            return Err(Exit::UnsupportedSystemContext {
                addr: self.cpu.rip,
                operation: UnsupportedSystemOperation::InvalidLegacyContext(
                    crate::MachineContextError::InvalidStorage,
                ),
            });
        }
        let cs = self.cpu.cs;
        let ip = self.cpu.rip as u16;
        let execution = self.legacy_execution.as_mut().unwrap();
        if let Some((at_cs, at_ip, opcode)) = execution.nec_refusal {
            if (at_cs, at_ip) == (cs, ip) {
                let mut bytes = [0; 15];
                bytes[0] = opcode;
                return Err(Exit::unknown_instruction(u64::from(ip), bytes, 1));
            }
            execution.nec_refusal = None;
        }
        if execution.nec_escape == Some((cs, ip)) {
            return self.nec_escape::<ROUTED, ORDER>(vm, ip);
        }
        execution.nec_escape = None;
        if self.cpu.rflags_system & 0x100 != 0 {
            return Err(Exit::UnsupportedSystemContext {
                addr: self.cpu.rip,
                operation: UnsupportedSystemOperation::NecEmulationSingleStep,
            });
        }
        let entry = execution.nec_entry.take().unwrap_or(I8080Entry::Resume);
        Err(Exit::I8080Mode {
            view: self.i8080_view(),
            entry,
        })
    }

    fn nec_fetch(&self, vm: &Vm, insn: u16, offset: u16) -> Result<u8, Exit> {
        let pa = ((u64::from(self.cpu.cs) << 4) + u64::from(offset)) & 0xf_ffff;
        let at = FetchAddr {
            pa,
            ip: u64::from(insn),
        };
        if let Some(policy) = &vm.mem.fetch_policy {
            let verdict = policy.approve(pa >> crate::memory::CODE_PAGE_BITS, &vm.mem);
            if verdict != crate::FetchVerdict::Allow {
                return Err(crate::fault::fetch_exit(
                    &vm.mem,
                    at,
                    CpuMode::Real16,
                    pa,
                    Some(crate::lift::FetchFailure::Policy(verdict)),
                ));
            }
        }
        let mut byte = [0];
        vm.mem.fetch_code(pa, &mut byte).map_err(|t| {
            crate::fault::fetch_exit(
                &vm.mem,
                at,
                CpuMode::Real16,
                pa,
                Some(crate::lift::FetchFailure::Memory(t)),
            )
        })?;
        Ok(byte[0])
    }

    // RETEM (ED FD, NEC UM p. 12-177) or CALLN (ED ED imm8, p. 12-176),
    // decoded from the ED byte the 8080 core stopped on (a second read of it).
    fn nec_escape<const ROUTED: bool, const ORDER: u8>(
        &mut self,
        vm: &Vm,
        ip: u16,
    ) -> Result<(), Exit> {
        let first = self.nec_fetch(vm, ip, ip)?;
        let mut bytes = [0; 15];
        bytes[0] = first;
        if first != 0xed {
            return Err(Exit::unknown_instruction(u64::from(ip), bytes, 1));
        }
        bytes[1] = self.nec_fetch(vm, ip, ip.wrapping_add(1))?;
        let mut calln = None;
        match bytes[1] {
            0xfd => {
                let word_bus =
                    crate::lift::i8086::word_bus(self.cpu_model.expect("NEC model").profile());
                let [pc, ps, psw] = crate::interp::i8086::read_frame::<ROUTED, ORDER>(
                    &mut self.cpu,
                    &vm.mem,
                    word_bus,
                )?;
                // Only the frame BRKEM saved (MD = 1) is described. p. 12-177
                // reads "When the BRKEM instruction is executed, the MD flag is
                // write disabled"; taken as a misprint for RETEM, the
                // instruction the page describes (ASSUMED).
                if psw & 0x8000 == 0 {
                    return Err(Exit::UnsupportedSystemContext {
                        addr: u64::from(ip),
                        operation: UnsupportedSystemOperation::NecRetemToEmulation,
                    });
                }
                // Every flag is restored from the frame; "Write disable MD".
                self.cpu.flags.set_flags16(psw & 0x0fd5);
                self.cpu.rflags_system = u64::from(psw & 0x100);
                let sp = self.cpu.gpr[4].wrapping_add(6);
                self.cpu.write_gpr(4, sp, 2);
                self.cpu.cs = ps;
                self.cpu.rip = u64::from(pc);
                self.cpu.legacy.reserved &= !(NEC_EMULATION | NEC_MD_WRITE | NEC_STANDBY);
            }
            0xed => {
                let vector = self.nec_fetch(vm, ip, ip.wrapping_add(2))?;
                let profile = self.cpu_model.expect("NEC model").profile();
                let quirks = profile.quirks;
                // The saved PC follows ED ED imm8 (§3 p. 3-1; ASSUMED for
                // CALLN, as x80's CALLN_PC_ADVANCE). The PSW image keeps
                // MD = 0 (p. 12-176); IE and BRK clear under rule R2.
                // p. 12-176 reads "SP <- PS - 6"; taken as a misprint for
                // SP - 6, the three-word frame its other lines push (ASSUMED).
                let origin = self.delivery_origin(self.cpu.cs, u32::from(ip.wrapping_add(3)));
                let flags = crate::interp::i8086::interrupt::<ROUTED, ORDER>(
                    &mut self.cpu,
                    &vm.mem,
                    vector,
                    ip.wrapping_add(3),
                    quirks.real_flags_set_mask & 0x7fff,
                    crate::lift::i8086::word_bus(profile),
                )?;
                self.cpu.legacy.reserved &= !(NEC_EMULATION | NEC_STANDBY);
                calln = Some((vector, origin, flags));
            }
            _ => return Err(Exit::unknown_instruction(u64::from(ip), bytes, 2)),
        }
        // One completed unit (decision-16 §2); the caller charges its step.
        self.retired += 1;
        self.executed += 1;
        if let Some((vector, origin, flags)) = calln.filter(|_| self.delivery_log.is_some()) {
            self.record_delivery(
                vector,
                crate::DeliveryKind::Software,
                origin,
                None,
                super::legacy::Interrupted::Frame20 { flags },
            );
        }
        self.sti_shadow = false;
        self.ss_shadow = 0;
        if self.cpu.mmio_retry_active {
            self.cpu.finish_mmio_retry(None);
        }
        let execution = self.legacy_execution.as_mut().unwrap();
        execution.nec_escape = None;
        execution.nec_entry = None;
        Ok(())
    }
}
