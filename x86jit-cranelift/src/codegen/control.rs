use super::*;

impl Translator<'_, '_> {
    pub(crate) fn emit_insn_start(&mut self, guest_addr: &u64) -> bool {
        self.cur_addr = *guest_addr;
        // SourceLoc indexes this function's full-width RIP array, including all
        // region blocks. Only the instruction count is bounded by u32; reserve
        // u32::MAX for Cranelift's default (unattributed) source location.
        let index = u32::try_from(self.guest_rips.len()).expect("too many source locations");
        assert_ne!(index, u32::MAX, "too many source locations");
        self.guest_rips.push(*guest_addr);
        self.builder.set_srcloc(ir::SourceLoc::new(index));
        false
    }

    pub(crate) fn emit_read_reg(&mut self, dst: &u32, reg: &Reg) -> bool {
        let v = self.read_reg(*reg);
        self.set(*dst, v);
        false
    }

    pub(crate) fn emit_write_reg(&mut self, reg: &Reg, src: &Val, size: &u8) -> bool {
        let v = self.val(*src);
        self.write_reg(*reg, v, *size);
        false
    }

    pub(crate) fn emit_get_cond(&mut self, dst: &u32, cond: &Cond) -> bool {
        let c = self.eval_cond(*cond);
        let v = self.builder.ins().uextend(types::I64, c);
        self.set(*dst, v);
        false
    }

    pub(crate) fn emit_set_df(&mut self, value: &bool) -> bool {
        let v = self.builder.ins().iconst(types::I8, *value as i64);
        self.store_flag(self.offsets.df, v);
        false
    }

    pub(crate) fn emit_jump(&mut self, target: &Val) -> bool {
        let t = self.val(*target);
        self.store_cpu(self.offsets.rip, t);
        match target {
            // Direct jump: known target, so chain through a link slot.
            Val::Imm(_) => {
                let slot = (self.alloc_slot)();
                self.chain_or_link(slot);
            }
            // Indirect jump: target unknown at compile time. Probe the
            // per-site IBTC — chain if the target repeats, else miss.
            Val::Temp(_) => {
                let slot = (self.alloc_slot)();
                self.ibtc_or_miss(slot, t);
            }
        }
        true
    }

    pub(crate) fn emit_branch(&mut self, cond: &Cond, taken: &u64, fallthrough: &u64) -> bool {
        let c = self.eval_cond(*cond);
        let tk = self.builder.create_block();
        let fl = self.builder.create_block();
        self.builder.ins().brif(c, tk, &[], fl, &[]);
        self.builder.seal_block(tk);
        self.builder.seal_block(fl);
        self.builder.switch_to_block(tk);
        let ta = self.iconst(*taken);
        self.store_cpu(self.offsets.rip, ta);
        let tslot = (self.alloc_slot)();
        self.chain_or_link(tslot);
        self.builder.switch_to_block(fl);
        let fa = self.iconst(*fallthrough);
        self.store_cpu(self.offsets.rip, fa);
        let fslot = (self.alloc_slot)();
        self.chain_or_link(fslot);
        true
    }

    pub(crate) fn emit_call(
        &mut self,
        target: &Val,
        return_addr: &u64,
        slot: &u8,
        wrap_sp: &bool,
    ) -> bool {
        let rsp = self.read_gpr(RSP);
        let delta = self.iconst(*slot as u64);
        let mut newsp = self.builder.ins().isub(rsp, delta);
        if *wrap_sp {
            newsp = self.builder.ins().band_imm(newsp, 0xffff_ffff);
        }
        let host = self.checked_addr(newsp, *slot, 1);
        let ra = self.iconst(*return_addr);
        self.store_guest(host, ra, *slot);
        self.note_store(newsp, *slot);
        self.write_gpr(RSP, newsp, if *wrap_sp { 4 } else { 8 });
        let tgt = self.val(*target);
        self.store_cpu(self.offsets.rip, tgt);
        // Return prediction: push (return_addr, continuation slot) onto
        // the shadow ring before transferring to the callee. The slot is an
        // ordinary link slot for the block at `return_addr`; the matching
        // `ret` chains through it. Done for both direct and indirect calls.
        let cont_slot = (self.alloc_slot)();
        self.emit_ret_push(*return_addr, cont_slot);
        match target {
            // Direct call: the callee entry is known, so chain to it the
            // same way a direct jump does. Only the transfer to the callee
            // is chained; the return-address push above already happened.
            Val::Imm(_) => {
                let slot = (self.alloc_slot)();
                self.chain_or_link(slot);
            }
            // Indirect call: IBTC-probe the computed callee, same as an
            // indirect jump.
            Val::Temp(_) => {
                let slot = (self.alloc_slot)();
                self.ibtc_or_miss(slot, tgt);
            }
        }
        true
    }

    pub(crate) fn emit_ret(&mut self, slot: &u8, pop_extra: &u16, wrap_sp: &bool) -> bool {
        let rsp = self.read_gpr(RSP);
        let host = self.checked_addr(rsp, *slot, 0);
        let ret = self.load_guest(host, *slot);
        let delta = self.iconst(*slot as u64 + *pop_extra as u64);
        let mut newsp = self.builder.ins().iadd(rsp, delta);
        if *wrap_sp {
            newsp = self.builder.ins().band_imm(newsp, 0xffff_ffff);
        }
        self.write_gpr(RSP, newsp, if *wrap_sp { 4 } else { 8 });
        self.store_cpu(self.offsets.rip, ret);
        // Return prediction: pop the shadow ring and chain to the
        // caller's continuation if the predicted address matches the actual
        // popped target; otherwise fall back to dispatch.
        self.emit_ret_predict(ret);
        true
    }

    pub(crate) fn emit_syscall(&mut self, is_amd64: bool) -> bool {
        let end = self.iconst(self.guest_end);
        // The AMD64 `syscall` instruction latches RCX <- next-instruction RIP and
        // R11 <- RFLAGS (hardware). The i386 `int 0x80` gate must NOT (its ABI passes
        // args in ECX/…). Mirrors the interpreter's `exec_syscall`.
        if is_amd64 {
            self.write_gpr(RCX, end, 8);
            let rflags = self.user_rflags_image(false);
            self.write_gpr(R11, rflags, 8);
        }
        self.store_cpu(self.offsets.rip, end);
        self.ret(RET_SYSCALL);
        true
    }

    /// Assemble the architectural RFLAGS word from the modeled flag fields — the
    /// JIT counterpart of `x86jit_core::state::Flags::to_rflags` (kept bit-for-bit
    /// identical): CF(0), reserved(1, always set), PF(2), AF(4), ZF(6), SF(7),
    /// DF(10), OF(11).
    fn assemble_rflags(&mut self) -> Value {
        // reserved bit 1, always set
        let mut acc = self.iconst(1 << 1);
        // PF and AF are stored as sources, so they are derived here rather than
        // loaded — this path (`syscall`, `pushf`, `lahf`) is the reason they are
        // kept at all.
        for (v, shift) in [(self.load_pf(), 2), (self.load_af(), 4)] {
            let w = self.builder.ins().uextend(types::I64, v);
            let shifted = self.builder.ins().ishl_imm(w, shift);
            acc = self.builder.ins().bor(acc, shifted);
        }
        for (off, shift) in [
            (self.offsets.cf, 0),
            (self.offsets.zf, 6),
            (self.offsets.sf, 7),
            (self.offsets.df, 10),
            (self.offsets.of, 11),
        ] {
            let b = self.load_flag_u64(off); // I64, 0 or 1
            let shifted = self.builder.ins().ishl_imm(b, shift);
            acc = self.builder.ins().bor(acc, shifted);
        }
        acc
    }

    /// `lahf` (0x9F): AH = the low byte of the FLAGS image — SF(7) ZF(6) 0(5) AF(4)
    /// 0(3) PF(2) 1(1) CF(0). OF is not in the byte and no flag is modified. Mirrors
    /// `interp::control::exec_lahf`, which takes the same bits out of `to_flags16`.
    pub(crate) fn emit_lahf(&mut self) -> bool {
        // reserved bit 1, always set
        let mut byte = self.iconst(1 << 1);
        let (cf_off, zf_off, sf_off) = (self.offsets.cf, self.offsets.zf, self.offsets.sf);
        // PF and AF are stored as sources, so they are derived, not loaded.
        for (v, shift) in [
            (self.load_flag(cf_off), 0),
            (self.load_pf(), 2),
            (self.load_af(), 4),
            (self.load_flag(zf_off), 6),
            (self.load_flag(sf_off), 7),
        ] {
            let w = self.builder.ins().uextend(types::I64, v);
            let shifted = self.builder.ins().ishl_imm(w, shift);
            byte = self.builder.ins().bor(byte, shifted);
        }
        let ah = self.builder.ins().ishl_imm(byte, 8);
        let rax = self.read_gpr(RAX);
        let keep = self.builder.ins().band_imm(rax, !0xff00i64);
        let new = self.builder.ins().bor(keep, ah);
        self.write_gpr(RAX, new, 8);
        false
    }

    /// `sahf` (0x9E): set SF/ZF/AF/PF/CF from AH bits 7/6/4/2/0. OF, DF and IF are
    /// untouched. Mirrors `interp::control::exec_sahf`.
    pub(crate) fn emit_sahf(&mut self) -> bool {
        let rax = self.read_gpr(RAX);
        let ah = self.builder.ins().ushr_imm(rax, 8);
        let bit = |t: &mut Self, n: i64| {
            let s = t.builder.ins().ushr_imm(ah, n);
            let b = t.builder.ins().band_imm(s, 1);
            t.builder.ins().ireduce(types::I8, b)
        };
        let cf = bit(self, 0);
        let pf = bit(self, 2);
        let zf = bit(self, 6);
        let sf = bit(self, 7);
        // AF's source encoding IS bit 4 of a byte, so AH masked to 0x10 is already it.
        let af_masked = self.builder.ins().band_imm(ah, 0x10);
        let af_src = self.builder.ins().ireduce(types::I8, af_masked);
        let pf_src = self.pf_src_from_bool(pf);
        let (cf_off, zf_off, sf_off) = (self.offsets.cf, self.offsets.zf, self.offsets.sf);
        let (pf_off, af_off) = (self.offsets.pf_src, self.offsets.af_src);
        self.store_flag(cf_off, cf);
        self.store_flag(pf_off, pf_src);
        self.store_flag(af_off, af_src);
        self.store_flag(zf_off, zf);
        self.store_flag(sf_off, sf);
        false
    }

    pub(crate) fn emit_hlt(&mut self) -> bool {
        let end = self.iconst(self.guest_end);
        self.store_cpu(self.offsets.rip, end);
        self.ret(RET_HLT);
        true
    }

    pub(crate) fn emit_trap(&mut self, vector: &u8, advance: &u8) -> bool {
        // A lifted architectural exception. x86 saved-RIP: fault (advance 0)
        // stays on the instruction, trap (advance = length) resumes past it —
        // matching the interpreter. The vector goes to the MemCtx out-field the
        // dispatcher reads (which sets `Exit::Exception.addr` from this RIP).
        let rip = self.iconst(self.cur_addr + *advance as u64);
        self.store_cpu(self.offsets.rip, rip);
        let vec = self.iconst(*vector as u64);
        self.store_mem(MEMCTX_EXCEPTION_VECTOR, vec);
        self.ret(RET_EXCEPTION);
        true
    }

    pub(crate) fn emit_port_io(&mut self) -> bool {
        // Port I/O (§5.2) — deferred to the interpreter, exactly like an
        // inlined MMIO access: set RIP to the instruction and hand back
        // `RET_PORTIO_DEFER`. The dispatcher single-steps it on the interp,
        // which produces `Exit::PortIo` (and, for `in`, records the pending
        // accumulator width). Rare and self-contained, so no need to open-code
        // the accumulator read / out-field plumbing in the backend.
        let rip = self.iconst(self.cur_addr);
        self.store_cpu(self.offsets.rip, rip);
        self.ret(RET_PORTIO_DEFER);
        true
    }
}

// CpuState fields are appended, preserving the existing fixed ABI offsets.
const SYSTEM_FLAGS: i32 = core::mem::offset_of!(x86jit_core::CpuState, rflags_system) as i32;
const IF_FLAG: i32 = (core::mem::offset_of!(x86jit_core::CpuState, flags)
    + core::mem::offset_of!(x86jit_core::Flags, if_)) as i32;

impl Translator<'_, '_> {
    pub(crate) fn emit_set_cf(&mut self, value: Option<bool>) -> bool {
        // SDM Vol 2A §3.2 CLC/CMC; Vol 2B §4.3 STC: preserve every other flag.
        let cf = match value {
            Some(v) => self.builder.ins().iconst(types::I8, v as i64),
            None => {
                let cf = self.load_flag(self.offsets.cf);
                self.builder.ins().bxor_imm(cf, 1)
            }
        };
        self.store_flag(self.offsets.cf, cf);
        false
    }

    fn user_rflags_image(&mut self, push: bool) -> Value {
        // SDM Vol 3B §20.3.1.1: RF clears before either image is captured.
        // SDM Vol 2B §4.3 PUSHF additionally clears VM in its stack image.
        let arithmetic = self.assemble_rflags();
        let system = self.load_cpu(SYSTEM_FLAGS);
        let system = self.builder.ins().band_imm(
            system,
            (x86jit_core::state::RFLAGS_SYSTEM_MASK & if push { !0x3_0000 } else { !0x1_0000 })
                as i64,
        );
        let if_ = self.load_flag_u64(IF_FLAG);
        let if_ = self.builder.ins().ishl_imm(if_, 9);
        let image = self.builder.ins().bor(arithmetic, system);
        self.builder.ins().bor(image, if_)
    }

    pub(crate) fn emit_push_flags(&mut self, size: u8, wrap_sp: bool) -> bool {
        let image = self.user_rflags_image(true);
        let rsp = self.read_gpr(RSP);
        let mut sp = self.builder.ins().iadd_imm(rsp, -(size as i64));
        if wrap_sp {
            sp = self.builder.ins().band_imm(sp, 0xffff_ffff);
        }
        let host = self.checked_addr(sp, size, 1);
        self.store_guest(host, image, size);
        self.note_store(sp, size);
        self.write_gpr(RSP, sp, if wrap_sp { 4 } else { 8 });
        false
    }

    pub(crate) fn emit_restore_flags(&mut self, src: Val, size: u8, iret: bool) -> bool {
        // SDM Vol 2B §4.3 POPF Table 4-20; Vol 2A §3.2 IRET, Operation.
        let image = self.val(src);
        for (offset, bit) in [
            (self.offsets.cf, 0),
            (self.offsets.zf, 6),
            (self.offsets.sf, 7),
            (self.offsets.df, 10),
            (self.offsets.of, 11),
        ] {
            let shifted = self.builder.ins().ushr_imm(image, bit);
            let value = self.builder.ins().band_imm(shifted, 1);
            let value = self.builder.ins().ireduce(types::I8, value);
            self.store_flag(offset, value);
        }
        // PF source is a parity byte; AF source carries the architectural bit 4.
        let pf = self.builder.ins().ushr_imm(image, 2);
        let pf = self.builder.ins().band_imm(pf, 1);
        let pf = self.builder.ins().bxor_imm(pf, 1);
        let pf = self.builder.ins().ireduce(types::I8, pf);
        self.store_flag(self.offsets.pf_src, pf);
        let af = self.builder.ins().band_imm(image, 0x10);
        let af = self.builder.ins().ireduce(types::I8, af);
        self.store_flag(self.offsets.af_src, af);
        let system = self.load_cpu(SYSTEM_FLAGS);
        let iopl = self.builder.ins().band_imm(system, 0x3000);
        let permitted = self.builder.ins().icmp_imm(IntCC::Equal, iopl, 0x3000);
        let old_if = self.load_flag(IF_FLAG);
        let new_if = self.builder.ins().ushr_imm(image, 9);
        let new_if = self.builder.ins().band_imm(new_if, 1);
        let new_if = self.builder.ins().ireduce(types::I8, new_if);
        let if_ = self.builder.ins().select(permitted, new_if, old_if);
        self.store_flag(IF_FLAG, if_);
        let mask = x86jit_core::state::user_rflags_mask(size, iret);
        let preserved = self
            .builder
            .ins()
            .band_imm(system, !(mask | if iret { 0 } else { 1 << 16 }) as i64);
        let changed = self.builder.ins().band_imm(image, mask as i64);
        let updated = self.builder.ins().bor(preserved, changed);
        self.store_cpu(SYSTEM_FLAGS, updated);
        false
    }

    fn selector_offset(segment: u8) -> i32 {
        use x86jit_core::CpuState;
        (match segment {
            0 => core::mem::offset_of!(CpuState, es),
            1 => core::mem::offset_of!(CpuState, cs),
            2 => core::mem::offset_of!(CpuState, ss),
            3 => core::mem::offset_of!(CpuState, ds),
            4 => core::mem::offset_of!(CpuState, fs),
            5 => core::mem::offset_of!(CpuState, gs),
            _ => unreachable!("segment encoding"),
        }) as i32
    }

    fn read_selector(&mut self, segment: u8) -> Value {
        let value = self.builder.ins().load(
            types::I16,
            MemFlags::trusted(),
            self.cpu,
            Self::selector_offset(segment),
        );
        self.builder.ins().uextend(types::I64, value)
    }

    pub(crate) fn emit_read_selector(&mut self, dst: u32, segment: u8) -> bool {
        let value = self.read_selector(segment);
        self.set(dst, value);
        false
    }

    pub(crate) fn emit_check_iret(&mut self, cs: Val, ss: Option<Val>, ip: Val) -> bool {
        let cs = self.val(cs);
        let cs = self.builder.ins().band_imm(cs, 0xffff);
        let configured = self.read_selector(1);
        let mut bad = self.builder.ins().icmp(IntCC::NotEqual, cs, configured);
        if let Some(ss) = ss {
            let ss = self.val(ss);
            let ss = self.builder.ins().band_imm(ss, 0xffff);
            let configured = self.read_selector(2);
            let mismatch = self.builder.ins().icmp(IntCC::NotEqual, ss, configured);
            bad = self.builder.ins().bor(bad, mismatch);
            let ip = self.val(ip);
            let canonical = self.builder.ins().ishl_imm(ip, 16);
            let canonical = self.builder.ins().sshr_imm(canonical, 16);
            let mismatch = self.builder.ins().icmp(IntCC::NotEqual, ip, canonical);
            bad = self.builder.ins().bor(bad, mismatch);
        }
        self.iret_gp_if(bad)
    }

    pub(crate) fn emit_check_iret_nt(&mut self) -> bool {
        let system = self.load_cpu(SYSTEM_FLAGS);
        let nt = self.builder.ins().band_imm(system, 1 << 14);
        let nt = self.builder.ins().icmp_imm(IntCC::NotEqual, nt, 0);
        self.iret_gp_if(nt)
    }

    fn iret_gp_if(&mut self, bad: Value) -> bool {
        let ok = self.begin_trap_fork(bad);
        let rip = self.iconst(self.cur_addr);
        self.store_cpu(self.offsets.rip, rip);
        let vector = self.iconst(13);
        self.store_mem(MEMCTX_EXCEPTION_VECTOR, vector);
        self.ret(RET_EXCEPTION);
        self.builder.switch_to_block(ok);
        false
    }
}
