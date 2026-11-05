//! MMX uses one 64-bit lane of the shared SSE integer primitives (§8.1).
use super::*;
use crate::ir::MmxOp;

pub(super) fn compute(a: u64, b: u64, op: MmxOp) -> u64 {
    let (av, bv) = (a as u128, b as u128);
    // SDM Vol 2B §4.3, PACK*/PHADD*/PHSUB*: concatenate the two half-width results.
    let halves = |r: u128| (r as u32 as u64) | ((r >> 64) as u32 as u64) << 32;
    match op {
        MmxOp::Packed { lane, op } => packed_bin(av, bv, lane, op) as u64,
        MmxOp::Logic(op) => match op {
            VLogicOp::And => a & b,
            VLogicOp::Andn => !a & b,
            VLogicOp::Or => a | b,
            VLogicOp::Xor => a ^ b,
        },
        MmxOp::Shift { lane, right, arith } => {
            packed_shift(av, b.min(u64::from(lane) * 8) as u8, lane, right, arith) as u64
        }
        MmxOp::Unpack { lane, high } => unpack_low(
            if high { av >> 32 } else { av },
            if high { bv >> 32 } else { bv },
            lane,
            false,
        ) as u64,
        MmxOp::Pack { from, signed } => halves(pack_lane(av, bv, from, signed)),
        MmxOp::Madd { unsigned } => pmadd_lane(av, bv, unsigned) as u64,
        MmxOp::Horizontal(HIntOp::Sad) => hint(av, bv, HIntOp::Sad) as u64,
        MmxOp::Horizontal(op) => halves(hint(av, bv, op)),
        MmxOp::Abs { lane } => vector::psign(av, av, lane) as u64,
        MmxOp::Sign { lane } => vector::psign(av, bv, lane) as u64,
        // SDM Vol 2B §4.3 PSHUFB: MMX indices use bits 2:0, bit 7 zeroes a byte.
        MmxOp::ShuffleBytes => pshufb(av, bv & 0x8787_8787_8787_8787) as u64,
        MmxOp::ShuffleWords(imm) => (0..4).fold(0, |r, n| {
            r | ((a >> (((imm >> (2 * n)) & 3) * 16)) & 0xffff) << (n * 16)
        }),
        // SDM Vol 2B §4.3 PALIGNR: concatenate two 64-bit operands, zero for count >= 16.
        MmxOp::Align(imm) => {
            if imm >= 16 {
                0
            } else {
                ((av << 64 | bv) >> (u32::from(imm) * 8)) as u64
            }
        }
        MmxOp::Insert(index) => (a & !(0xffffu64 << (index * 16))) | ((b & 0xffff) << (index * 16)),
        MmxOp::Extract(index) => (a >> (index * 16)) & 0xffff,
        MmxOp::Mask => movemask_b(av),
    }
}
