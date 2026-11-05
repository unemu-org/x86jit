//! Integer arithmetic for non-default SIMD modes (spec §16).
//!
//! SDM Vol 1 §§10.2.3.2–10.2.3.4: RC rounds the result, DAZ replaces input
//! subnormals with signed zero, and FTZ flushes results tiny after rounding.
//! The 127-bit working significand retains exact products of two binary64
//! significands. Right shifts jam discarded bits into a sticky low bit.

use crate::ir::{FloatBinOp, FloatUnOp};

pub const MODE_MASK: u32 = 0xe040;

#[derive(Clone, Copy)]
struct Format {
    frac: u32,
    bias: i32,
    exp_mask: u64,
    sign: u64,
}

impl Format {
    fn new(double: bool) -> Self {
        if double {
            Self {
                frac: 52,
                bias: 1023,
                exp_mask: 0x7ff0_0000_0000_0000,
                sign: 1 << 63,
            }
        } else {
            Self {
                frac: 23,
                bias: 127,
                exp_mask: 0x7f80_0000,
                sign: 1 << 31,
            }
        }
    }
    fn fraction(self) -> u64 {
        (1 << self.frac) - 1
    }
    fn quiet(self, x: u64) -> u64 {
        x | (1 << (self.frac - 1))
    }
    fn nan(self, x: u64) -> bool {
        x & self.exp_mask == self.exp_mask && x & self.fraction() != 0
    }
    fn inf(self, x: u64) -> bool {
        x & !self.sign == self.exp_mask
    }
    fn indefinite(self) -> u64 {
        self.sign | self.exp_mask | (1 << (self.frac - 1))
    }
    fn input(self, x: u64, mode: u32) -> u64 {
        let x = x & (self.sign | self.exp_mask | self.fraction());
        if mode & 0x40 != 0 && x & self.exp_mask == 0 {
            x & self.sign
        } else {
            x
        }
    }
    fn unpack(self, x: u64) -> Wide {
        let ef = ((x & self.exp_mask) >> self.frac) as i32;
        let sig = (x & self.fraction()) | if ef == 0 { 0 } else { 1 << self.frac };
        Wide::new(
            x & self.sign != 0,
            sig as u128,
            ef.max(1) - self.bias - self.frac as i32,
        )
    }
    fn pack(self, w: Wide, mode: u32) -> u64 {
        let sign = if w.neg { self.sign } else { 0 };
        if w.sig == 0 {
            return sign;
        }
        let min_exp = 1 - self.bias;
        // SDM Vol 1 §11.5.2.5: FTZ classifies the result rounded at the
        // destination precision with an UNBOUNDED exponent, before denormal
        // rounding. A bounded-exponent carry alone does not suppress flushing.
        if mode & 0x8000 != 0 && w.exp < min_exp {
            let shift = 126 - self.frac;
            let kept = w.sig >> shift;
            let lost = w.sig & ((1u128 << shift) - 1);
            let half = 1u128 << (shift - 1);
            let up = match (mode >> 13) & 3 {
                1 => w.neg && lost != 0,
                2 => !w.neg && lost != 0,
                3 => false,
                _ => lost > half || (lost == half && kept & 1 != 0),
            };
            if w.exp + i32::from(kept + u128::from(up) >= 1u128 << (self.frac + 1)) < min_exp {
                return sign;
            }
            return sign | (1 << self.frac);
        }

        let mut exp = w.exp;
        let sig = jam(w.sig, (min_exp - exp).max(0) as u32);
        exp = exp.max(min_exp);
        let shift = 126 - self.frac;
        let mut rounded = sig >> shift;
        let lost = sig & ((1u128 << shift) - 1);
        let half = 1u128 << (shift - 1);
        let rc = (mode >> 13) & 3;
        let up = match rc {
            1 => w.neg && lost != 0,
            2 => !w.neg && lost != 0,
            3 => false,
            _ => lost > half || (lost == half && rounded & 1 != 0),
        };
        rounded += u128::from(up);
        if rounded >= 1u128 << (self.frac + 1) {
            rounded >>= 1;
            exp += 1;
        }
        if exp > self.bias {
            let inf = rc == 0 || (rc == 1 && w.neg) || (rc == 2 && !w.neg);
            return sign
                | if inf {
                    self.exp_mask
                } else {
                    self.exp_mask - 1
                };
        }
        let normal = rounded >= 1u128 << self.frac;
        if !normal && mode & 0x8000 != 0 {
            return sign;
        }
        let ef = if normal { (exp + self.bias) as u64 } else { 0 };
        sign | (ef << self.frac) | (rounded as u64 & self.fraction())
    }
}

#[derive(Clone, Copy)]
struct Wide {
    neg: bool,
    sig: u128,
    exp: i32,
}

impl Wide {
    // `sig * 2^scale`, normalized with its leading bit at bit 126.
    fn new(neg: bool, sig: u128, scale: i32) -> Self {
        if sig == 0 {
            return Self { neg, sig, exp: 0 };
        }
        let leading = 127 - sig.leading_zeros();
        let shift = 126i32 - leading as i32;
        let sig = if shift >= 0 {
            sig << shift
        } else {
            jam(sig, (-shift) as u32)
        };
        Self {
            neg,
            sig,
            exp: scale + leading as i32,
        }
    }
    fn add(mut a: Self, mut b: Self, rc: u32) -> Self {
        if a.sig == 0 && b.sig == 0 {
            return Self {
                neg: if a.neg == b.neg { a.neg } else { rc == 1 },
                ..a
            };
        }
        if a.sig == 0 {
            return b;
        }
        if b.sig == 0 {
            return a;
        }
        if (a.exp, a.sig) < (b.exp, b.sig) {
            core::mem::swap(&mut a, &mut b);
        }
        let bs = jam(b.sig, (a.exp - b.exp) as u32);
        let sig = if a.neg == b.neg {
            a.sig + bs
        } else {
            a.sig - bs
        };
        if sig == 0 {
            return Self {
                neg: rc == 1,
                sig: 0,
                exp: 0,
            };
        }
        Self::new(a.neg, sig, a.exp - 126)
    }
}

fn jam(x: u128, n: u32) -> u128 {
    if n == 0 {
        x
    } else if n >= 128 {
        u128::from(x != 0)
    } else {
        (x >> n) | u128::from(x << (128 - n) != 0)
    }
}

pub fn input(x: u64, double: bool, mode: u32) -> u64 {
    Format::new(double).input(x, mode)
}

pub fn binary(x: u64, y: u64, double: bool, op: FloatBinOp, mode: u32) -> u64 {
    let f = Format::new(double);
    let x = f.input(x, mode);
    let y = f.input(y, mode);
    if matches!(op, FloatBinOp::Min | FloatBinOp::Max) {
        if f.nan(x) || f.nan(y) || (x & !f.sign == 0 && y & !f.sign == 0) {
            return y;
        }
        let key = |v: u64| {
            if v & f.sign != 0 {
                !v & (f.sign | f.exp_mask | f.fraction())
            } else {
                v | f.sign
            }
        };
        let take_x = if matches!(op, FloatBinOp::Min) {
            key(x) < key(y)
        } else {
            key(x) > key(y)
        };
        return if take_x { x } else { y };
    }
    if f.nan(x) {
        return f.quiet(x);
    }
    if f.nan(y) {
        return f.quiet(y);
    }
    let y = if matches!(op, FloatBinOp::Sub) {
        y ^ f.sign
    } else {
        y
    };
    let a = f.unpack(x);
    let b = f.unpack(y);
    let neg = (x ^ y) & f.sign;
    let rc = (mode >> 13) & 3;
    let w = match op {
        FloatBinOp::Add | FloatBinOp::Sub => {
            if f.inf(x) || f.inf(y) {
                return if f.inf(x) && f.inf(y) && neg != 0 {
                    f.indefinite()
                } else if f.inf(x) {
                    x
                } else {
                    y
                };
            }
            Wide::add(a, b, rc)
        }
        FloatBinOp::Mul => {
            if f.inf(x) || f.inf(y) {
                return if (f.inf(x) && y & !f.sign == 0) || (f.inf(y) && x & !f.sign == 0) {
                    f.indefinite()
                } else {
                    neg | f.exp_mask
                };
            }
            if a.sig == 0 || b.sig == 0 {
                return neg;
            }
            let shift = 126 - f.frac;
            Wide::new(
                neg != 0,
                (a.sig >> shift) * (b.sig >> shift),
                a.exp + b.exp - 2 * f.frac as i32,
            )
        }
        FloatBinOp::Div => {
            if (f.inf(x) && f.inf(y)) || (x & !f.sign == 0 && y & !f.sign == 0) {
                return f.indefinite();
            }
            if f.inf(x) || y & !f.sign == 0 {
                return neg | f.exp_mask;
            }
            if f.inf(y) || x & !f.sign == 0 {
                return neg;
            }
            let asig = a.sig >> (126 - f.frac);
            let bsig = b.sig >> (126 - f.frac);
            let numerator = asig << 74;
            let q = (numerator / bsig) | u128::from(!numerator.is_multiple_of(bsig));
            Wide::new(neg != 0, q, a.exp - b.exp - 74)
        }
        _ => unreachable!(),
    };
    f.pack(w, mode)
}

pub fn unary(x: u64, double: bool, op: FloatUnOp, mode: u32) -> u64 {
    // SDM Vol 2B §4.3 RCPPS/RSQRTPS: RC is ignored; input denormals
    // are signed zero and tiny reciprocal results are flushed regardless of FTZ.
    let mode = if matches!(op, FloatUnOp::Rcp | FloatUnOp::Rsqrt) {
        0x8040
    } else {
        mode
    };
    let f = Format::new(double);
    let x = f.input(x, mode);
    if f.nan(x) {
        return f.quiet(x);
    }
    if matches!(op, FloatUnOp::Rcp) {
        return binary((f.bias as u64) << f.frac, x, double, FloatBinOp::Div, mode);
    }
    if x & f.sign != 0 && x & !f.sign != 0 {
        return f.indefinite();
    }
    if f.inf(x) {
        return if matches!(op, FloatUnOp::Rsqrt) { 0 } else { x };
    }
    if x & !f.sign == 0 {
        return if matches!(op, FloatUnOp::Rsqrt) {
            (x & f.sign) | f.exp_mask
        } else {
            x
        };
    }
    let a = f.unpack(x);
    let sig = a.sig >> (126 - f.frac);
    let n = sig << (124 - f.frac + (a.exp & 1) as u32);
    let mut root = 0u128;
    let mut rem = n;
    let mut bit = 1u128 << 126;
    while bit > rem {
        bit >>= 2;
    }
    while bit != 0 {
        if rem >= root + bit {
            rem -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    let r = f.pack(
        Wide::new(false, root | u128::from(rem != 0), (a.exp >> 1) - 62),
        mode,
    );
    if matches!(op, FloatUnOp::Rsqrt) {
        binary((f.bias as u64) << f.frac, r, double, FloatBinOp::Div, mode)
    } else {
        r
    }
}

pub fn from_int(raw: u64, signed: bool, double: bool, mode: u32) -> u64 {
    let neg = signed && (raw as i64) < 0;
    let mag = if neg { raw.wrapping_neg() } else { raw };
    Format::new(double).pack(Wide::new(neg, mag as u128, 0), mode)
}

pub fn convert(raw: u64, from_double: bool, to_double: bool, mode: u32) -> u64 {
    let from = Format::new(from_double);
    let to = Format::new(to_double);
    let raw = from.input(raw, mode);
    let sign = if raw & from.sign != 0 { to.sign } else { 0 };
    if from.nan(raw) {
        let payload = raw & from.fraction();
        let payload = if to.frac > from.frac {
            payload << (to.frac - from.frac)
        } else {
            payload >> (from.frac - to.frac)
        };
        return to.quiet(sign | to.exp_mask | payload);
    }
    if from.inf(raw) {
        return sign | to.exp_mask;
    }
    to.pack(from.unpack(raw), mode)
}

pub fn round(raw: u64, double: bool, rc: u32, mode: u32) -> u64 {
    let f = Format::new(double);
    let raw = f.input(raw, mode);
    if f.nan(raw) {
        return f.quiet(raw);
    }
    if f.inf(raw) {
        return raw;
    }
    let w = f.unpack(raw);
    if w.sig == 0 || w.exp >= f.frac as i32 {
        return raw;
    }
    if w.exp < 0 {
        let up = match rc {
            1 => w.neg,
            2 => !w.neg,
            3 => false,
            _ => w.exp == -1 && w.sig > 1 << 126,
        };
        return (raw & f.sign) | if up { (f.bias as u64) << f.frac } else { 0 };
    }
    let sh = (126 - w.exp) as u32;
    let integer = w.sig >> sh;
    let lost = w.sig & ((1u128 << sh) - 1);
    let half = 1u128 << (sh - 1);
    let up = match rc {
        1 => w.neg && lost != 0,
        2 => !w.neg && lost != 0,
        3 => false,
        _ => lost > half || (lost == half && integer & 1 != 0),
    };
    f.pack(Wide::new(w.neg, integer + u128::from(up), 0), 0)
}

pub fn fma(x: u64, y: u64, z: u64, double: bool, np: bool, na: bool, mode: u32) -> u64 {
    let f = Format::new(double);
    let x = f.input(x, mode);
    let y = f.input(y, mode);
    let z = f.input(z, mode);
    // SDM Vol 1 §14.5.2, Table 14-17: source-order NaNs, before negation.
    for v in [x, y, z] {
        if f.nan(v) {
            return f.quiet(v);
        }
    }
    let product_neg = ((x ^ y) & f.sign != 0) ^ np;
    let z = z ^ if na { f.sign } else { 0 };
    if f.inf(x) || f.inf(y) {
        if x & !f.sign == 0 || y & !f.sign == 0 {
            return f.indefinite();
        }
        if f.inf(z) && (z & f.sign != 0) != product_neg {
            return f.indefinite();
        }
        return f.exp_mask | if product_neg { f.sign } else { 0 };
    }
    if f.inf(z) {
        return z;
    }
    let a = f.unpack(x);
    let b = f.unpack(y);
    let sh = 126 - f.frac;
    let product = Wide::new(
        product_neg,
        (a.sig >> sh) * (b.sig >> sh),
        a.exp + b.exp - 2 * f.frac as i32,
    );
    f.pack(Wide::add(product, f.unpack(z), (mode >> 13) & 3), mode)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn software_simd_arithmetic_matches_nearest_hardware() {
        // Numerical witness for SDM Vol 1 §4.8.4, using the host IEEE operations
        // only as an oracle. Integer computation above never changes host control.
        let mut state = 0x2fb9_77ab_0de3_1245u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..2000 {
            let (x, y, z) = (next(), next(), next());
            let (a, b, c) = (f64::from_bits(x), f64::from_bits(y), f64::from_bits(z));
            if !a.is_finite() || !b.is_finite() || !c.is_finite() {
                continue;
            }
            for (op, want) in [
                (FloatBinOp::Add, a + b),
                (FloatBinOp::Sub, a - b),
                (FloatBinOp::Mul, a * b),
                (FloatBinOp::Div, a / b),
            ] {
                if !want.is_nan() {
                    assert_eq!(
                        binary(x, y, true, op, 0),
                        want.to_bits(),
                        "{op:?} x={x:x} y={y:x}"
                    );
                }
            }
            if a >= 0.0 {
                assert_eq!(
                    unary(x, true, FloatUnOp::Sqrt, 0),
                    a.sqrt().to_bits(),
                    "sqrt x={x:x}"
                );
            }
            let want = a.mul_add(b, c);
            if !want.is_nan() {
                assert_eq!(
                    fma(x, y, z, true, false, false, 0),
                    want.to_bits(),
                    "fma x={x:x} y={y:x} z={z:x}"
                );
            }
        }
    }

    #[test]
    fn software_simd_rc_overflow_signed_zero_and_fused_cancellation() {
        // SDM Vol 1 §§4.8.3.7, 4.8.4, 10.2.3.2; §14.5.1: one FMA rounding.
        for rc in 0..4 {
            let mode = rc << 13;
            assert_eq!(
                binary(0x3f800000, 0xbf800000, false, FloatBinOp::Add, mode),
                if rc == 1 { 0x80000000 } else { 0 }
            );
            assert_eq!(
                binary(0x7f7fffff, 0x40000000, false, FloatBinOp::Mul, mode),
                if rc == 0 || rc == 2 {
                    0x7f800000
                } else {
                    0x7f7fffff
                }
            );
            // (1+2^-23)*(1-2^-23)-1 = -2^-46, exact after cancellation.
            assert_eq!(
                fma(0x3f800001, 0x3f7ffffe, 0xbf800000, false, false, false, mode),
                0xa8800000
            );
        }
    }
}
