//! One-dispatch headers for the real-mode stack sequences (xj-to9.29).
//!
//! The real-mode lowering spells a 16-bit-SP push as six ops and a pop as
//! eight (PUSH, POP, the return-address push of a near CALL and the pop of a
//! near RET), the twenty-bit lowering (`lift::i8086`) as eight and ten.
//! [`fuse_instruction`] puts a [`L::FusedPush`], [`L::FusedPop`],
//! [`L::FusedPush20`] or [`L::FusedPop20`] in front of each exact occurrence
//! and leaves the sequence itself in place, so fault reconstruction and the
//! other readers that look for particular ops find the same ops as before.
//! The alias retry fingerprint (`interp::integer::instruction_fingerprint`)
//! hashes every op, the header included, so a fused instruction's fingerprint
//! differs from that of an unfused lift of it (the one-instruction step lift).
//! That is harmless only because no fused instruction reaches it: a Real16
//! instruction with a `DirectStackAddress` takes `unified_read_retry`'s
//! Machine load, which binds no fingerprint, and the twenty-bit forms are
//! Machine walks. The
//! interpreter's ordinary walk executes the header and steps over the
//! sequence; a walk that does not (a child walk, or a Machine walk with its
//! own instruction executor) treats the header as nothing and runs the
//! sequence. A header never spans an instruction boundary: the sequence lies
//! between one `InsnStart` and the next.
//!
//! Real16 IR is interpreter-only (it never reaches a JIT backend), so this
//! pass runs only on Real16 blocks.

use crate::ir::{FlagMask, IrOp, Temp, Val};
use crate::legacy::LegacyOp as L;
use crate::state::Reg;

/// Ops a [`L::FusedPush`] header stands for.
pub(crate) const PUSH_OPS: usize = 6;
/// Ops a [`L::FusedPop`] header stands for.
pub(crate) const POP_OPS: usize = 8;
/// Ops a [`L::FusedPush20`] header stands for.
pub(crate) const PUSH20_OPS: usize = 8;
/// Ops a [`L::FusedPop20`] header stands for.
pub(crate) const POP20_OPS: usize = 10;

fn read_sp(op: &IrOp) -> Option<Temp> {
    match *op {
        IrOp::ReadReg { dst, reg: Reg::Rsp } => Some(dst),
        _ => None,
    }
}

/// `dst = Temp(a) <op> Imm(_)` at 64 bits without flags; the immediate.
fn address_op(op: &IrOp, add: bool, dst: Temp, a: Temp) -> Option<u64> {
    match address_operands(op, add, dst)? {
        (Val::Temp(x), Val::Imm(imm)) if x == a => Some(imm),
        _ => None,
    }
}

/// The operands of `dst = a + b` (`add`) or `dst = a & b` at 64 bits
/// without flags.
fn address_operands(op: &IrOp, add: bool, dst: Temp) -> Option<(Val, Val)> {
    let (d, x, y, size, flags) = match *op {
        IrOp::Add {
            dst,
            a,
            b,
            size,
            set_flags,
        } if add => (dst, a, b, size, set_flags),
        IrOp::And {
            dst,
            a,
            b,
            size,
            set_flags,
        } if !add => (dst, a, b, size, set_flags),
        _ => return None,
    };
    (d == dst && size == 8 && flags == FlagMask::NONE).then_some((x, y))
}

/// `dst = (SS << 36) + Temp(offset)` through `ReadReg` SS at `dst - 2`: the
/// twenty-bit segment tag (`lift::i8086::tagged_address`), three ops.
fn stack_tag(ops: &[IrOp], dst: Temp, offset: Temp) -> bool {
    let [selector, shift, sum] = ops else {
        return false;
    };
    matches!(*selector, IrOp::ReadReg { dst: d, reg: Reg::Ss } if d + 2 == dst)
        && matches!(
            *shift,
            IrOp::Shl {
                dst: d,
                a: Val::Temp(a),
                b: Val::Imm(36),
                size: 8,
                set_flags: FlagMask::NONE,
            } if d + 1 == dst && a + 2 == dst
        )
        && address_operands(sum, true, dst) == Some((Val::Temp(dst - 1), Val::Temp(offset)))
}

fn stack_gate(op: &IrOp, dst: Temp, offset: Temp, size: u8) -> Option<u8> {
    match *op {
        IrOp::Legacy(L::DirectStackAddress {
            dst: d,
            offset: Val::Temp(o),
            bytes,
            vector,
        }) if d == dst && o == offset && bytes == size => Some(vector),
        _ => None,
    }
}

fn write_sp(op: &IrOp, src: Temp) -> bool {
    matches!(*op, IrOp::WriteReg { reg: Reg::Rsp, src: Val::Temp(s), size: 2 } if s == src)
}

/// The push header for `ops[..PUSH_OPS]`, if they are exactly the sequence.
pub(crate) fn push_header(ops: &[IrOp]) -> Option<IrOp> {
    let [sp, sum, masked, gate, store, write] = ops.get(..PUSH_OPS)? else {
        return None;
    };
    let first = read_sp(sp)?;
    let IrOp::Store {
        fault,
        addr: Val::Temp(addr),
        src,
        size,
        ..
    } = *store
    else {
        return None;
    };
    first.checked_add(3)?;
    let delta = address_op(sum, true, first + 1, first)?;
    let vector = stack_gate(gate, first + 3, first + 2, size)?;
    (delta == u64::from(size).wrapping_neg()
        && address_op(masked, false, first + 2, first + 1)? == 0xffff
        && addr == first + 3
        && write_sp(write, first + 2))
    .then_some(IrOp::Legacy(L::FusedPush {
        first,
        src,
        bytes: size,
        vector,
        fault,
    }))
}

/// The pop header for `ops[..POP_OPS]`, if they are exactly the sequence.
pub(crate) fn pop_header(ops: &[IrOp]) -> Option<IrOp> {
    let [sp, masked, gate, load, again, sum, wrapped, write] = ops.get(..POP_OPS)? else {
        return None;
    };
    let first = read_sp(sp)?;
    let IrOp::Load {
        fault,
        dst,
        addr: Val::Temp(addr),
        size,
    } = *load
    else {
        return None;
    };
    let last = first.checked_add(6)?;
    let vector = stack_gate(gate, first + 2, first + 1, size)?;
    let adjust = address_op(sum, true, first + 5, first + 4)?;
    (address_op(masked, false, first + 1, first)? == 0xffff
        && addr == first + 2
        && dst == first + 3
        && read_sp(again) == Some(first + 4)
        && address_op(wrapped, false, last, first + 5)? == 0xffff
        && write_sp(write, last))
    .then(|| {
        Some(IrOp::Legacy(L::FusedPop {
            first,
            adjust: u16::try_from(adjust).ok()?,
            bytes: size,
            vector,
            fault,
        }))
    })
    .flatten()
}

/// The twenty-bit push header for `ops[..PUSH20_OPS]`
/// (`lift::i8086`: no stack gate, the store through the segment tag).
pub(crate) fn push20_header(ops: &[IrOp]) -> Option<IrOp> {
    let [sp, sum, masked, _, _, _, store, write] = ops.get(..PUSH20_OPS)? else {
        return None;
    };
    let first = read_sp(sp)?;
    let IrOp::Legacy(L::WrappedStore {
        addr: Val::Temp(addr),
        value,
        size,
        word_bus,
    }) = *store
    else {
        return None;
    };
    first.checked_add(5)?;
    let delta = address_op(sum, true, first + 1, first)?;
    (delta == u64::from(size).wrapping_neg()
        && address_op(masked, false, first + 2, first + 1)? == 0xffff
        && stack_tag(&ops[3..6], first + 5, first + 2)
        && addr == first + 5
        && write_sp(write, first + 2))
    .then_some(IrOp::Legacy(L::FusedPush20 {
        first,
        src: value,
        bytes: size,
        word_bus,
    }))
}

/// The twenty-bit pop header for `ops[..POP20_OPS]`.
pub(crate) fn pop20_header(ops: &[IrOp]) -> Option<IrOp> {
    let [sp, masked, _, _, _, load, again, sum, wrapped, write] = ops.get(..POP20_OPS)? else {
        return None;
    };
    let first = read_sp(sp)?;
    let IrOp::Legacy(L::WrappedLoad {
        dst,
        addr: Val::Temp(addr),
        size,
        word_bus,
    }) = *load
    else {
        return None;
    };
    let last = first.checked_add(8)?;
    let adjust = address_op(sum, true, first + 7, first + 6)?;
    (address_op(masked, false, first + 1, first)? == 0xffff
        && stack_tag(&ops[2..5], first + 4, first + 1)
        && addr == first + 4
        && dst == first + 5
        && read_sp(again) == Some(first + 6)
        && address_op(wrapped, false, last, first + 7)? == 0xffff
        && write_sp(write, last))
    .then(|| {
        Some(IrOp::Legacy(L::FusedPop20 {
            first,
            adjust: u16::try_from(adjust).ok()?,
            bytes: size,
            word_bus,
        }))
    })
    .flatten()
}

#[cfg(any(test, feature = "test-hooks"))]
thread_local! {
    static UNFUSED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static TAKEN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Witness count: the walk ran one header in place of its sequence.
#[cfg(any(test, feature = "test-hooks"))]
pub(crate) fn note_taken() {
    TAKEN.with(|taken| taken.set(taken.get() + 1));
}

/// Witness switch: the headers this thread's walks ran in place of their
/// sequences since the last call, so a test can tell a walk that steps over
/// the sequence from one that ignores the header.
#[cfg(any(test, feature = "test-hooks"))]
#[doc(hidden)]
pub fn take_fused_headers_for_test() -> u64 {
    TAKEN.with(|taken| taken.replace(0))
}

/// Witness switch: whether this thread's lifts add the headers, so one guest
/// program can run fused and unfused and the results be compared.
#[cfg(any(test, feature = "test-hooks"))]
#[doc(hidden)]
pub fn set_stack_fusion_for_test(enabled: bool) {
    UNFUSED.with(|unfused| unfused.set(!enabled));
}

/// Whether `insn` can lower to a stack sequence that takes a header: the
/// lift calls [`fuse_instruction`] only for these, so other instructions
/// cost one mnemonic test.
pub(crate) fn candidate(insn: &iced_x86::Instruction) -> bool {
    use iced_x86::Mnemonic::*;
    matches!(insn.mnemonic(), Push | Pop | Call | Ret)
}

/// Put a header in front of every exact real-mode stack sequence of the
/// instruction whose ops are `ops[start..]`.
pub(crate) fn fuse_instruction(ops: &mut Vec<IrOp>, start: usize) {
    #[cfg(any(test, feature = "test-hooks"))]
    if UNFUSED.with(std::cell::Cell::get) {
        return;
    }
    let mut i = start;
    while i < ops.len() {
        let found = if matches!(ops[i], IrOp::ReadReg { reg: Reg::Rsp, .. }) {
            let at = &ops[i..];
            push_header(at)
                .map(|h| (h, PUSH_OPS))
                .or_else(|| pop_header(at).map(|h| (h, POP_OPS)))
                .or_else(|| push20_header(at).map(|h| (h, PUSH20_OPS)))
                .or_else(|| pop20_header(at).map(|h| (h, POP20_OPS)))
        } else {
            None
        };
        match found {
            Some((header, len)) => {
                ops.insert(i, header);
                i += 1 + len;
            }
            None => i += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lift::{lift_block, FetchAddr};
    use crate::memory::{Memory, MemoryModel};
    use crate::{CpuMode, Prot, RegionKind};

    fn lifted(code: &[u8]) -> Vec<IrOp> {
        let mut mem = Memory::new(MemoryModel::Flat { size: 0x2_0000 });
        mem.map(0, 0x2_0000, Prot::RWX, RegionKind::Ram).unwrap();
        mem.write_bytes(0x1000, code).unwrap();
        lift_block(&mem, FetchAddr::real16(0, 0x1000), CpuMode::Real16)
            .unwrap()
            .ops
    }

    fn header_kinds(ops: &[IrOp]) -> Vec<&'static str> {
        ops.iter()
            .filter_map(|op| match op {
                IrOp::Legacy(L::FusedPush { .. }) => Some("push"),
                IrOp::Legacy(L::FusedPop { .. }) => Some("pop"),
                IrOp::Legacy(L::FusedPush20 { .. }) => Some("push20"),
                IrOp::Legacy(L::FusedPop20 { .. }) => Some("pop20"),
                _ => None,
            })
            .collect()
    }

    fn without_headers(ops: &[IrOp]) -> String {
        let kept: Vec<_> = ops
            .iter()
            .filter(|op| header_kinds(std::slice::from_ref(*op)).is_empty())
            .collect();
        format!("{kept:?}")
    }

    #[test]
    fn real16_stack_forms_get_one_header_each_and_keep_their_ops() {
        let segments = ["push"; 4];
        for (code, expected) in [
            (&b"\x50\x5b\xf4"[..], &["push", "pop"][..]),
            (b"\x54\x5c\xf4", &["push", "pop"]),
            (b"\x1e\x06\x16\x0e\xf4", &segments),
            // POP to memory spells its pop differently and keeps its ops.
            (b"\xff\x37\x8f\x07\xf4", &["push"]),
            (b"\x68\x34\x12\x6a\x80\xf4", &["push", "push"]),
            (b"\x66\x50\x66\x5b\xf4", &["push", "pop"]),
            (b"\xe8\x00\x00", &["push"]),
            (b"\xc3", &["pop"]),
            (b"\xc2\x04\x00", &["pop"]),
        ] {
            let ops = lifted(code);
            assert_eq!(header_kinds(&ops), expected, "{code:02x?}: {ops:?}");
            set_stack_fusion_for_test(false);
            let unfused = lifted(code);
            set_stack_fusion_for_test(true);
            // Removing the headers gives the unfused lift, op for op.
            assert_eq!(
                without_headers(&ops),
                format!("{:?}", unfused.iter().collect::<Vec<_>>())
            );
            // A header and its sequence lie inside one instruction.
            for (i, op) in ops.iter().enumerate() {
                let len = match op {
                    IrOp::Legacy(L::FusedPush { .. }) => PUSH_OPS,
                    IrOp::Legacy(L::FusedPop { .. }) => POP_OPS,
                    _ => continue,
                };
                assert!(ops[i + 1..=i + len]
                    .iter()
                    .all(|op| !matches!(op, IrOp::InsnStart { .. })));
            }
        }
    }

    #[test]
    fn a_changed_sequence_gets_no_header() {
        let ops = lifted(b"\x50\x5b\xf4");
        let push = ops
            .iter()
            .position(|op| matches!(op, IrOp::Legacy(L::FusedPush { .. })))
            .unwrap();
        let sequence = &ops[push + 1..push + 1 + PUSH_OPS];
        assert!(push_header(sequence).is_some());
        for index in 0..PUSH_OPS {
            let mut changed = sequence.to_vec();
            match &mut changed[index] {
                IrOp::ReadReg { reg, .. } => *reg = Reg::Rbp,
                IrOp::Add { b, .. } | IrOp::And { b, .. } => *b = Val::Imm(0xfffe),
                IrOp::Legacy(L::DirectStackAddress { bytes, .. }) => *bytes = 4,
                IrOp::Store { size, .. } => *size = 1,
                IrOp::WriteReg { size, .. } => *size = 4,
                op => panic!("unexpected {op:?}"),
            }
            assert!(push_header(&changed).is_none(), "{index}: {changed:?}");
        }
    }
}
