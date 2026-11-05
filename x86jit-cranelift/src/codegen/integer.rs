use super::*;

impl Translator<'_, '_> {
    pub(crate) fn emit_add(
        &mut self,
        dst: &u32,
        a: &Val,
        b: &Val,
        size: &u8,
        set_flags: &FlagMask,
    ) -> bool {
        let (a, b) = (self.val(*a), self.val(*b));
        let zero = self.iconst(0);
        self.add_sub(*dst, a, b, zero, *size, *set_flags, false);
        false
    }

    pub(crate) fn emit_adc(
        &mut self,
        dst: &u32,
        a: &Val,
        b: &Val,
        size: &u8,
        set_flags: &FlagMask,
    ) -> bool {
        let (a, b) = (self.val(*a), self.val(*b));
        let cin = self.load_flag_u64(self.offsets.cf);
        self.add_sub(*dst, a, b, cin, *size, *set_flags, false);
        false
    }

    /// SDM Vol 2A §3.3 ADOX: OF carries unsigned overflow.
    pub(crate) fn emit_adox(&mut self, dst: u32, a: Val, b: Val, size: u8, mask: FlagMask) -> bool {
        let a = self.val(a);
        let b = self.val(b);
        let a = self.mask(a, size);
        let b = self.mask(b, size);
        let cin = self.load_flag_u64(self.offsets.of);
        let sum = self.builder.ins().iadd(a, b);
        let result = self.builder.ins().iadd(sum, cin);
        let carry = if size == 8 {
            let first = self.builder.ins().icmp(IntCC::UnsignedLessThan, sum, a);
            let second = self
                .builder
                .ins()
                .icmp(IntCC::UnsignedLessThan, result, sum);
            self.builder.ins().bor(first, second)
        } else {
            let high = self.builder.ins().ushr_imm(result, i64::from(size * 8));
            self.builder.ins().ireduce(types::I8, high)
        };
        let result = self.mask(result, size);
        self.set(dst, result);
        self.flag_recipe = Some(FlagRecipe::Adox { carry });
        self.record_omitted_flags(mask);
        if mask.0 & 0b10_0000 != 0 {
            self.store_flag(self.offsets.of, carry);
        }
        false
    }

    pub(crate) fn emit_sub(
        &mut self,
        dst: &u32,
        a: &Val,
        b: &Val,
        size: &u8,
        set_flags: &FlagMask,
    ) -> bool {
        let (a, b) = (self.val(*a), self.val(*b));
        let zero = self.iconst(0);
        self.add_sub(*dst, a, b, zero, *size, *set_flags, true);
        false
    }

    pub(crate) fn emit_sbb(
        &mut self,
        dst: &u32,
        a: &Val,
        b: &Val,
        size: &u8,
        set_flags: &FlagMask,
    ) -> bool {
        let (a, b) = (self.val(*a), self.val(*b));
        let cin = self.load_flag_u64(self.offsets.cf);
        self.add_sub(*dst, a, b, cin, *size, *set_flags, true);
        false
    }

    pub(crate) fn emit_and(
        &mut self,
        dst: &u32,
        a: &Val,
        b: &Val,
        size: &u8,
        set_flags: &FlagMask,
    ) -> bool {
        let (a, b) = (self.val(*a), self.val(*b));
        let r = self.builder.ins().band(a, b);
        self.logic(*dst, r, *size, *set_flags);
        false
    }

    pub(crate) fn emit_or(
        &mut self,
        dst: &u32,
        a: &Val,
        b: &Val,
        size: &u8,
        set_flags: &FlagMask,
    ) -> bool {
        let (a, b) = (self.val(*a), self.val(*b));
        let r = self.builder.ins().bor(a, b);
        self.logic(*dst, r, *size, *set_flags);
        false
    }

    pub(crate) fn emit_xor(
        &mut self,
        dst: &u32,
        a: &Val,
        b: &Val,
        size: &u8,
        set_flags: &FlagMask,
    ) -> bool {
        let (a, b) = (self.val(*a), self.val(*b));
        let r = self.builder.ins().bxor(a, b);
        self.logic(*dst, r, *size, *set_flags);
        false
    }

    pub(crate) fn emit_shl(
        &mut self,
        dst: &u32,
        a: &Val,
        b: &Val,
        size: &u8,
        set_flags: &FlagMask,
    ) -> bool {
        let (a, b) = (self.val(*a), self.val(*b));
        self.emit_shift(*dst, ShiftKind::Shl, a, b, *size, *set_flags);
        false
    }

    pub(crate) fn emit_shr(
        &mut self,
        dst: &u32,
        a: &Val,
        b: &Val,
        size: &u8,
        set_flags: &FlagMask,
    ) -> bool {
        let (a, b) = (self.val(*a), self.val(*b));
        self.emit_shift(*dst, ShiftKind::Shr, a, b, *size, *set_flags);
        false
    }

    pub(crate) fn emit_sar(
        &mut self,
        dst: &u32,
        a: &Val,
        b: &Val,
        size: &u8,
        set_flags: &FlagMask,
    ) -> bool {
        let (a, b) = (self.val(*a), self.val(*b));
        self.emit_shift(*dst, ShiftKind::Sar, a, b, *size, *set_flags);
        false
    }

    pub(crate) fn emit_rol(
        &mut self,
        dst: &u32,
        a: &Val,
        b: &Val,
        size: &u8,
        set_flags: &FlagMask,
    ) -> bool {
        let (a, b) = (self.val(*a), self.val(*b));
        self.emit_shift(*dst, ShiftKind::Rol, a, b, *size, *set_flags);
        false
    }

    pub(crate) fn emit_ror(
        &mut self,
        dst: &u32,
        a: &Val,
        b: &Val,
        size: &u8,
        set_flags: &FlagMask,
    ) -> bool {
        let (a, b) = (self.val(*a), self.val(*b));
        self.emit_shift(*dst, ShiftKind::Ror, a, b, *size, *set_flags);
        false
    }

    pub(crate) fn emit_rcl(
        &mut self,
        dst: &u32,
        a: &Val,
        b: &Val,
        size: &u8,
        set_flags: &FlagMask,
    ) -> bool {
        let (a, b) = (self.val(*a), self.val(*b));
        self.emit_rcx(*dst, a, b, *size, *set_flags, true);
        false
    }

    pub(crate) fn emit_rcr(
        &mut self,
        dst: &u32,
        a: &Val,
        b: &Val,
        size: &u8,
        set_flags: &FlagMask,
    ) -> bool {
        let (a, b) = (self.val(*a), self.val(*b));
        self.emit_rcx(*dst, a, b, *size, *set_flags, false);
        false
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_double_shift_arm(
        &mut self,
        dst: &u32,
        a: &Val,
        b: &Val,
        count: &Val,
        size: &u8,
        left: &bool,
        set_flags: &FlagMask,
    ) -> bool {
        let (a, b, count) = (self.val(*a), self.val(*b), self.val(*count));
        self.emit_double_shift(*dst, a, b, count, *size, *left, *set_flags);
        false
    }

    pub(crate) fn emit_sext(&mut self, dst: &u32, a: &Val, from: &u8) -> bool {
        let a = self.val(*a);
        let r = self.sign_extend(a, *from);
        self.set(*dst, r);
        false
    }

    pub(crate) fn emit_bswap(&mut self, dst: &u32, a: &Val, size: &u8) -> bool {
        let a = self.val(*a);
        let r = if *size >= 8 {
            self.builder.ins().bswap(a)
        } else {
            let s = self.builder.ins().ireduce(int_ty(*size), a);
            let sw = self.builder.ins().bswap(s);
            self.builder.ins().uextend(types::I64, sw)
        };
        self.set(*dst, r);
        false
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_mul_arm(
        &mut self,
        lo: &u32,
        hi: &u32,
        a: &Val,
        b: &Val,
        size: &u8,
        signed: &bool,
        set_flags: &FlagMask,
    ) -> bool {
        let (a, b) = (self.val(*a), self.val(*b));
        self.emit_mul(*lo, *hi, a, b, *size, *signed, *set_flags);
        false
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_div_arm(
        &mut self,
        quot: &u32,
        rem: &u32,
        hi: &Val,
        lo: &Val,
        divisor: &Val,
        size: &u8,
        signed: &bool,
    ) -> bool {
        let (hi, lo, dv) = (self.val(*hi), self.val(*lo), self.val(*divisor));
        self.emit_div(*quot, *rem, hi, lo, dv, *size, *signed);
        false
    }

    pub(crate) fn emit_bt(
        &mut self,
        result: &u32,
        a: &Val,
        bit: &Val,
        size: &u8,
        op: &BtOp,
    ) -> bool {
        let av = self.val(*a);
        let bits = (*size as i64) * 8;
        let b = self.val(*bit);
        let b = self.builder.ins().band_imm(b, bits - 1);
        // CF = (a >> b) & 1
        let sh = self.builder.ins().ushr(av, b);
        let cf = self.builder.ins().band_imm(sh, 1);
        let cf = self.builder.ins().ireduce(types::I8, cf);
        self.store_flag(self.offsets.cf, cf);
        // result = a with the bit set / cleared / toggled
        let one = self.iconst(1);
        let mask = self.builder.ins().ishl(one, b);
        let r = match op {
            BtOp::Test => av,
            BtOp::Set => self.builder.ins().bor(av, mask),
            BtOp::Reset => {
                let nm = self.builder.ins().bnot(mask);
                self.builder.ins().band(av, nm)
            }
            BtOp::Complement => self.builder.ins().bxor(av, mask),
        };
        self.set(*result, r);
        false
    }

    pub(crate) fn emit_cpuid(&mut self) -> bool {
        self.flush_gprs(); // helper reads RAX/RCX from CpuState
        let cpu = self.cpu;
        self.call_helper(self.helpers.cpuid, &[cpu]);
        self.reload_gprs(); // helper wrote RAX/RBX/RCX/RDX
        false
    }

    pub(crate) fn emit_xgetbv(&mut self) -> bool {
        self.flush_gprs(); // preserve pending region writes before reloading GPRs
        let cpu = self.cpu;
        self.call_helper(self.helpers.xgetbv, &[cpu]);
        self.reload_gprs(); // helper wrote RAX/RDX (XCR0)
        false
    }

    pub(crate) fn emit_tsc(&mut self, aux: bool) -> bool {
        self.flush_gprs();
        let aux = self.iconst(u64::from(aux));
        self.call_helper(self.helpers.tsc, &[self.cpu, aux]);
        self.reload_gprs();
        false
    }

    pub(crate) fn emit_random(
        &mut self,
        dst: u32,
        kind: x86jit_core::RandomKind,
        size: u8,
        set_flags: FlagMask,
    ) -> bool {
        let seed = self.iconst(u64::from(kind == x86jit_core::RandomKind::Rdseed));
        let size = self.iconst(u64::from(size));
        let mask = self.iconst(u64::from(set_flags.0));
        let call = self.call_helper(self.helpers.random, &[self.cpu, seed, size, mask]);
        let value = self.builder.inst_results(call)[0];
        self.set(dst, value);
        // The helper's writes supersede any cold flag recovery recipe.
        for (i, pending) in self.pending_flags.iter_mut().enumerate() {
            if set_flags.0 & (1 << i) != 0 {
                *pending = None;
            }
        }
        false
    }

    pub(crate) fn emit_x87(
        &mut self,
        kind: &x86jit_core::x87::FpuKind,
        addr: &Val,
        sti: &u8,
    ) -> bool {
        let a = self.val(*addr);
        let kc = self.iconst(*kind as u16 as u64);
        let stic = self.iconst(*sti as u64);
        let cur = self.iconst(self.cur_addr);
        let args = [self.cpu, self.mem, kc, a, stic, cur];
        self.flush_gprs(); // helper reads/writes CpuState
        let access = if x86jit_core::x87::mem_write_bytes(*kind).is_some() {
            Some(GuestAccessKind::Write)
        } else if x86jit_core::x87::mem_operand_bytes(*kind) != 0 {
            Some(GuestAccessKind::Read)
        } else {
            None
        };
        let inst = self.call_memory_helper(self.helpers.x87, &args, access);
        self.trap_if_unmapped_or_exception(inst);
        self.reload_gprs(); // e.g. fnstsw wrote AX
        false
    }

    pub(crate) fn emit_fx_state(&mut self, addr: &Val, restore: &bool) -> bool {
        let a = self.val(*addr);
        let rc = self.iconst(*restore as u64);
        let cur = self.iconst(self.cur_addr);
        let args = [self.cpu, self.mem, a, rc, cur];
        self.flush_gprs(); // helper reads CpuState (XMM/x87)
        let access = if *restore {
            GuestAccessKind::Read
        } else {
            GuestAccessKind::Write
        };
        let inst = self.call_memory_helper(self.helpers.fxstate, &args, Some(access));
        self.trap_if_unmapped_or_exception(inst);
        false
    }

    pub(crate) fn emit_popcnt(&mut self, dst: &u32, src: &Val, size: &u8) -> bool {
        let s = self.val(*src);
        let s = self.mask(s, *size);
        let cnt = self.builder.ins().popcnt(s);
        self.set(*dst, cnt);
        let zero = self.iconst(0);
        let z = self.builder.ins().icmp(IntCC::Equal, s, zero);
        self.store_flag(self.offsets.zf, z);
        let z8 = self.builder.ins().iconst(types::I8, 0);
        for off in [
            self.offsets.cf,
            self.offsets.of,
            self.offsets.sf,
            self.offsets.af_src,
        ] {
            self.store_flag(off, z8);
        }
        // A zero PF SOURCE would mean PF=1; clearing PF needs an odd-parity byte.
        let pf0 = self.pf_src_const(false);
        self.store_flag(self.offsets.pf_src, pf0);
        false
    }

    pub(crate) fn emit_crc32(&mut self, dst: &u32, crc: &Val, src: &Val, bytes: &u8) -> bool {
        let c = self.val(*crc);
        let s = self.val(*src);
        let n = self.iconst(*bytes as u64);
        let inst = self.call_helper(self.helpers.crc32, &[c, s, n]);
        let r = self.builder.inst_results(inst)[0];
        self.set(*dst, r);
        false
    }

    pub(crate) fn emit_bit_scan(
        &mut self,
        dst: &u32,
        src: &Val,
        old: &Val,
        size: &u8,
        op: &BitScanOp,
    ) -> bool {
        let s = self.val(*src);
        let s = self.mask(s, *size);
        let zero = self.iconst(0);
        let is_zero = self.builder.ins().icmp(IntCC::Equal, s, zero);
        let bits = *size as i64 * 8;
        let r = match op {
            BitScanOp::Bsf | BitScanOp::Bsr => {
                self.store_flag(self.offsets.zf, is_zero); // only ZF defined
                let idx = if matches!(op, BitScanOp::Bsr) {
                    let clz = self.builder.ins().clz(s);
                    self.builder.ins().irsub_imm(clz, 63) // 63 - clz
                } else {
                    self.builder.ins().ctz(s)
                };
                let ov = self.val(*old);
                let ov = self.mask(ov, *size);
                self.builder.ins().select(is_zero, ov, idx) // src==0 -> keep old
            }
            BitScanOp::Tzcnt => {
                // Defined on zero (= bit-width). CF=src==0, ZF=result==0.
                let ctz = self.builder.ins().ctz(s);
                let bc = self.iconst(bits as u64);
                let r = self.builder.ins().select(is_zero, bc, ctz);
                self.store_flag(self.offsets.cf, is_zero);
                let rz = self.builder.ins().icmp_imm(IntCC::Equal, r, 0);
                self.store_flag(self.offsets.zf, rz);
                r
            }
            BitScanOp::Lzcnt => {
                // clz over the full I64 minus the padding above `bits`.
                let clz = self.builder.ins().clz(s);
                let r = self.builder.ins().iadd_imm(clz, -(64 - bits));
                self.store_flag(self.offsets.cf, is_zero);
                let rz = self.builder.ins().icmp_imm(IntCC::Equal, r, 0);
                self.store_flag(self.offsets.zf, rz);
                r
            }
        };
        self.set(*dst, r);
        false
    }

    pub(crate) fn emit_bmi(
        &mut self,
        dst: &u32,
        a: &Val,
        b: &Val,
        size: &u8,
        op: &x86jit_core::ir::BmiOp,
    ) -> bool {
        // Result + CF from the shared bmi_result helper (out-slot, like div);
        // ZF/SF derived from the result here. Guarantees jit == interp.
        let av = self.val(*a);
        let bv = self.val(*b);
        let opc = self.iconst(*op as u64);
        let sz = self.iconst(*size as u64);
        let (ss, _) = self.call_with_out_slot(self.helpers.bmi, &[av, bv, opc, sz]);
        let r = self.builder.ins().stack_load(types::I64, ss, 0);
        let cf = self.builder.ins().stack_load(types::I64, ss, 8);
        self.set(*dst, r);
        if op.writes_flags() {
            let cfb = self.builder.ins().icmp_imm(IntCC::NotEqual, cf, 0);
            self.store_flag(self.offsets.cf, cfb);
            let zero = self.iconst(0);
            let zf = self.builder.ins().icmp(IntCC::Equal, r, zero);
            self.store_flag(self.offsets.zf, zf);
            let bits = *size as i64 * 8;
            let top = self.builder.ins().ushr_imm(r, bits - 1);
            let sfv = self.builder.ins().band_imm(top, 1);
            let sf = self.builder.ins().icmp_imm(IntCC::NotEqual, sfv, 0);
            self.store_flag(self.offsets.sf, sf);
            let z8 = self.builder.ins().iconst(types::I8, 0);
            self.store_flag(self.offsets.of, z8);
        }
        false
    }
}
