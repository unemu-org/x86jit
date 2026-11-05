//! Inline physical MMX state and one-lane packed arithmetic (§8.2.3).
use super::*;
use x86jit_core::ir::MmxOp;

impl Translator<'_, '_> {
    pub(super) fn mmx_state(&mut self, empty: bool) {
        // SDM Vol 3A §15.2 Table 15-2. Emit only after fault-capable accesses.
        let z = self.builder.ins().iconst(types::I32, 0);
        let tags = self
            .builder
            .ins()
            .iconst(types::I8, if empty { 255 } else { 0 });
        self.builder
            .ins()
            .store(MemFlags::trusted(), z, self.cpu, self.offsets.fpu_top);
        self.builder
            .ins()
            .store(MemFlags::trusted(), tags, self.cpu, self.offsets.fpu_empty);
    }

    pub(super) fn mmx_read(&mut self, reg: u8) -> Value {
        self.builder.ins().load(
            types::I64,
            MemFlags::new().with_notrap(),
            self.cpu,
            self.offsets.fpr + i32::from(reg) * 10,
        )
    }

    pub(super) fn mmx_write(&mut self, reg: u8, src: Value) {
        let off = self.offsets.fpr + i32::from(reg) * 10;
        let exp = self.builder.ins().iconst(types::I16, -1);
        self.builder
            .ins()
            .store(MemFlags::new().with_notrap(), src, self.cpu, off);
        self.builder
            .ins()
            .store(MemFlags::new().with_notrap(), exp, self.cpu, off + 8);
    }

    fn mmx_elem(&mut self, v: Value, n: u8, lane: u8, signed: bool) -> Value {
        let shifted = self.builder.ins().ushr_imm(v, i64::from(n * lane * 8));
        let ty = match lane {
            1 => types::I8,
            2 => types::I16,
            4 => types::I32,
            _ => return shifted,
        };
        let narrow = self.builder.ins().ireduce(ty, shifted);
        if signed {
            self.builder.ins().sextend(types::I64, narrow)
        } else {
            self.builder.ins().uextend(types::I64, narrow)
        }
    }

    fn mmx_insert(&mut self, acc: Value, v: Value, n: u8, lane: u8) -> Value {
        let mask = u64::MAX >> (64 - lane * 8);
        let v = self.builder.ins().band_imm(v, mask as i64);
        let v = self.builder.ins().ishl_imm(v, i64::from(n * lane * 8));
        self.builder.ins().bor(acc, v)
    }

    pub(super) fn mmx_compute(&mut self, a: Value, b: Value, op: MmxOp) -> Value {
        match op {
            MmxOp::Logic(op) => self.emit_vlogic(a, b, op),
            MmxOp::Packed { lane, op } => {
                let av = self.builder.ins().uextend(types::I128, a);
                let bv = self.builder.ins().uextend(types::I128, b);
                let av = self.bitcast_v(av, vec_ty(lane));
                let bv = self.bitcast_v(bv, vec_ty(lane));
                let r = self.emit_packed_bin(av, bv, op);
                let r = self.bitcast_i128(r);
                self.builder.ins().ireduce(types::I64, r)
            }
            MmxOp::Shift { lane, right, arith } => {
                let av = self.builder.ins().uextend(types::I128, a);
                let count = self.builder.ins().uextend(types::I128, b);
                let r = self.shift_count_value(av, count, lane, right, arith);
                self.builder.ins().ireduce(types::I64, r)
            }
            MmxOp::Sign { lane } | MmxOp::Abs { lane } => {
                let av = self.builder.ins().uextend(types::I128, a);
                let bv = if matches!(op, MmxOp::Abs { .. }) {
                    av
                } else {
                    self.builder.ins().uextend(types::I128, b)
                };
                let r = if matches!(op, MmxOp::Abs { .. }) {
                    let v = self.bitcast_v(av, vec_ty(lane));
                    let v = self.builder.ins().iabs(v);
                    self.bitcast_i128(v)
                } else {
                    self.emit_psign(av, bv, lane)
                };
                self.builder.ins().ireduce(types::I64, r)
            }
            MmxOp::Align(imm) => {
                if imm >= 16 {
                    return self.iconst(0);
                }
                let av = self.builder.ins().uextend(types::I128, a);
                let bv = self.builder.ins().uextend(types::I128, b);
                let av = self.builder.ins().ishl_imm(av, 64);
                let v = self.builder.ins().bor(av, bv);
                let r = self.builder.ins().ushr_imm(v, i64::from(imm) * 8);
                self.builder.ins().ireduce(types::I64, r)
            }
            MmxOp::Extract(index) => self.mmx_elem(a, index, 2, false),
            MmxOp::Insert(index) => {
                let v = self.builder.ins().band_imm(a, !(0xffffi64 << (index * 16)));
                self.mmx_insert(v, b, index, 2)
            }
            MmxOp::Mask => {
                let a = self.builder.ins().uextend(types::I128, a);
                let v = self.bitcast_v(a, types::I8X16);
                let mask = self.builder.ins().vhigh_bits(types::I16, v);
                self.builder.ins().uextend(types::I64, mask)
            }
            MmxOp::ShuffleBytes => {
                let a = self.builder.ins().uextend(types::I128, a);
                let b = self
                    .builder
                    .ins()
                    .band_imm(b, 0x8787_8787_8787_8787u64 as i64);
                let b = self.builder.ins().uextend(types::I128, b);
                let v = self.emit_pshufb(a, b);
                self.builder.ins().ireduce(types::I64, v)
            }
            MmxOp::ShuffleWords(imm) => {
                let mut r = self.iconst(0);
                for n in 0..4 {
                    let v = self.mmx_elem(a, (imm >> (2 * n)) & 3, 2, false);
                    r = self.mmx_insert(r, v, n, 2);
                }
                r
            }
            MmxOp::Unpack { lane, high } => {
                let mut r = self.iconst(0);
                for n in 0..4 / lane {
                    let index = n + if high { 4 / lane } else { 0 };
                    let av = self.mmx_elem(a, index, lane, false);
                    let bv = self.mmx_elem(b, index, lane, false);
                    r = self.mmx_insert(r, av, 2 * n, lane);
                    r = self.mmx_insert(r, bv, 2 * n + 1, lane);
                }
                r
            }
            MmxOp::Pack { from, signed } => {
                let mut r = self.iconst(0);
                let bits = from * 4;
                let lo = self.iconst(if signed {
                    (-(1i64 << (bits - 1))) as u64
                } else {
                    0
                });
                let hi = self.iconst(if signed {
                    (1 << (bits - 1)) - 1
                } else {
                    (1 << bits) - 1
                });
                for (half, src) in [a, b].into_iter().enumerate() {
                    for n in 0..8 / from {
                        let v = self.mmx_elem(src, n, from, true);
                        let v = self.builder.ins().smin(v, hi);
                        let v = self.builder.ins().smax(v, lo);
                        r = self.mmx_insert(r, v, half as u8 * (8 / from) + n, from / 2);
                    }
                }
                r
            }
            MmxOp::Madd { unsigned } => {
                let lane = if unsigned { 1 } else { 2 };
                let mut r = self.iconst(0);
                for n in 0..4 / lane {
                    let mut pair = self.iconst(0);
                    for j in 0..2 {
                        let av = self.mmx_elem(a, 2 * n + j, lane, !unsigned);
                        let bv = self.mmx_elem(b, 2 * n + j, lane, true);
                        let prod = self.builder.ins().imul(av, bv);
                        pair = self.builder.ins().iadd(pair, prod);
                    }
                    if unsigned {
                        let lo = self.iconst(i16::MIN as u64);
                        let hi = self.iconst(i16::MAX as u64);
                        pair = self.builder.ins().smin(pair, hi);
                        pair = self.builder.ins().smax(pair, lo);
                    }
                    r = self.mmx_insert(r, pair, n, lane * 2);
                }
                r
            }
            MmxOp::Horizontal(op) => {
                let mut r = self.iconst(0);
                if matches!(op, HIntOp::Sad) {
                    for n in 0..8 {
                        let av = self.mmx_elem(a, n, 1, false);
                        let bv = self.mmx_elem(b, n, 1, false);
                        let diff = self.builder.ins().isub(av, bv);
                        let diff = self.builder.ins().iabs(diff);
                        r = self.builder.ins().iadd(r, diff);
                    }
                } else {
                    let lane = if matches!(op, HIntOp::AddD | HIntOp::SubD) {
                        4
                    } else {
                        2
                    };
                    for (half, src) in [a, b].into_iter().enumerate() {
                        for n in 0..4 / lane {
                            let x = self.mmx_elem(src, 2 * n, lane, true);
                            let y = self.mmx_elem(src, 2 * n + 1, lane, true);
                            let mut v = if matches!(op, HIntOp::SubW | HIntOp::SubD | HIntOp::SubSw)
                            {
                                self.builder.ins().isub(x, y)
                            } else {
                                self.builder.ins().iadd(x, y)
                            };
                            if matches!(op, HIntOp::AddSw | HIntOp::SubSw) {
                                let lo = self.iconst(i16::MIN as u64);
                                let hi = self.iconst(i16::MAX as u64);
                                v = self.builder.ins().smin(v, hi);
                                v = self.builder.ins().smax(v, lo);
                            }
                            r = self.mmx_insert(r, v, half as u8 * (4 / lane) + n, lane);
                        }
                    }
                }
                r
            }
        }
    }
}
