//! Extracted `interpret_block` dispatch arm bodies (integer); see `super`.

use super::*;
use crate::ir::*;

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn exec_read_reg(
    cpu: &mut CpuState,
    temps: &mut [u64],
    dst: &Temp,
    reg: &crate::state::Reg,
) -> Option<InterpResult> {
    temps[*dst as usize] = read_reg(cpu, *reg);
    None
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn exec_write_reg(
    cpu: &mut CpuState,
    temps: &mut [u64],
    reg: &crate::state::Reg,
    src: &Val,
    size: &u8,
) -> Option<InterpResult> {
    let v = read_val(*src, &*temps);
    write_reg(cpu, *reg, v, *size);
    None
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn exec_add(
    cpu: &mut CpuState,
    temps: &mut [u64],
    dst: &Temp,
    a: &Val,
    b: &Val,
    size: &u8,
    set_flags: &FlagMask,
) -> Option<InterpResult> {
    let r = alu_add(read_val(*a, &*temps), read_val(*b, &*temps), 0, *size);
    temps[*dst as usize] = r.res;
    apply(&mut cpu.flags, *set_flags, &r);
    None
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn exec_adc(
    cpu: &mut CpuState,
    temps: &mut [u64],
    dst: &Temp,
    a: &Val,
    b: &Val,
    size: &u8,
    set_flags: &FlagMask,
) -> Option<InterpResult> {
    let c = cpu.flags.cf as u64;
    let r = alu_add(read_val(*a, &*temps), read_val(*b, &*temps), c, *size);
    temps[*dst as usize] = r.res;
    apply(&mut cpu.flags, *set_flags, &r);
    None
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn exec_sub(
    cpu: &mut CpuState,
    temps: &mut [u64],
    dst: &Temp,
    a: &Val,
    b: &Val,
    size: &u8,
    set_flags: &FlagMask,
) -> Option<InterpResult> {
    let r = alu_sub(read_val(*a, &*temps), read_val(*b, &*temps), 0, *size);
    temps[*dst as usize] = r.res;
    apply(&mut cpu.flags, *set_flags, &r);
    None
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn exec_sbb(
    cpu: &mut CpuState,
    temps: &mut [u64],
    dst: &Temp,
    a: &Val,
    b: &Val,
    size: &u8,
    set_flags: &FlagMask,
) -> Option<InterpResult> {
    let c = cpu.flags.cf as u64;
    let r = alu_sub(read_val(*a, &*temps), read_val(*b, &*temps), c, *size);
    temps[*dst as usize] = r.res;
    apply(&mut cpu.flags, *set_flags, &r);
    None
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn exec_and(
    cpu: &mut CpuState,
    temps: &mut [u64],
    dst: &Temp,
    a: &Val,
    b: &Val,
    size: &u8,
    set_flags: &FlagMask,
) -> Option<InterpResult> {
    let r = alu_logic(read_val(*a, &*temps) & read_val(*b, &*temps), *size);
    temps[*dst as usize] = r.res;
    apply(&mut cpu.flags, *set_flags, &r);
    None
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn exec_or(
    cpu: &mut CpuState,
    temps: &mut [u64],
    dst: &Temp,
    a: &Val,
    b: &Val,
    size: &u8,
    set_flags: &FlagMask,
) -> Option<InterpResult> {
    let r = alu_logic(read_val(*a, &*temps) | read_val(*b, &*temps), *size);
    temps[*dst as usize] = r.res;
    apply(&mut cpu.flags, *set_flags, &r);
    None
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn exec_xor(
    cpu: &mut CpuState,
    temps: &mut [u64],
    dst: &Temp,
    a: &Val,
    b: &Val,
    size: &u8,
    set_flags: &FlagMask,
) -> Option<InterpResult> {
    let r = alu_logic(read_val(*a, &*temps) ^ read_val(*b, &*temps), *size);
    temps[*dst as usize] = r.res;
    apply(&mut cpu.flags, *set_flags, &r);
    None
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn exec_shl(
    cpu: &mut CpuState,
    temps: &mut [u64],
    dst: &Temp,
    a: &Val,
    b: &Val,
    size: &u8,
    set_flags: &FlagMask,
) -> Option<InterpResult> {
    let vm = read_val(*a, &*temps) & mask(*size);
    let cnt = read_val(*b, &*temps) & shift_mask(*size);
    let res = vm.wrapping_shl(cnt as u32) & mask(*size);
    temps[*dst as usize] = res;
    if !set_flags.is_none() && cnt != 0 {
        let n = (*size * 8) as u64;
        let cf = cnt <= n && (vm >> (n - cnt)) & 1 != 0;
        let of = (res & sign_bit(*size) != 0) ^ cf; // count==1 rule
        apply(
            &mut cpu.flags,
            *set_flags,
            &shift_result(res, *size, cf, of),
        );
    }
    None
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn exec_shr(
    cpu: &mut CpuState,
    temps: &mut [u64],
    dst: &Temp,
    a: &Val,
    b: &Val,
    size: &u8,
    set_flags: &FlagMask,
) -> Option<InterpResult> {
    let vm = read_val(*a, &*temps) & mask(*size);
    let cnt = read_val(*b, &*temps) & shift_mask(*size);
    let res = vm.wrapping_shr(cnt as u32);
    temps[*dst as usize] = res;
    if !set_flags.is_none() && cnt != 0 {
        let cf = (vm >> (cnt - 1)) & 1 != 0;
        let of = vm & sign_bit(*size) != 0; // count==1 rule: original MSB
        apply(
            &mut cpu.flags,
            *set_flags,
            &shift_result(res, *size, cf, of),
        );
    }
    None
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_double_shift(
    cpu: &mut CpuState,
    temps: &mut [u64],
    dst: &Temp,
    a: &Val,
    b: &Val,
    count: &Val,
    size: &u8,
    left: &bool,
    set_flags: &FlagMask,
) -> Option<InterpResult> {
    let n = (*size * 8) as u64;
    let av = read_val(*a, &*temps) & mask(*size);
    let bv = read_val(*b, &*temps) & mask(*size);
    let cnt = read_val(*count, &*temps) & shift_mask(*size);
    if cnt == 0 {
        // Masked count 0 is a no-op; flags unchanged.
        temps[*dst as usize] = av;
    } else {
        let (res, cf) = if *left {
            let lo = av.wrapping_shl(cnt as u32);
            let hi = if cnt < n { bv >> (n - cnt) } else { 0 };
            let cf = cnt <= n && (av >> (n - cnt)) & 1 != 0;
            ((lo | hi) & mask(*size), cf)
        } else {
            let lo = av >> cnt;
            let hi = if cnt < n {
                bv.wrapping_shl((n - cnt) as u32)
            } else {
                0
            };
            let cf = (av >> (cnt - 1)) & 1 != 0;
            ((lo | hi) & mask(*size), cf)
        };
        temps[*dst as usize] = res;
        if !set_flags.is_none() {
            // OF (count==1): the result's sign bit flipped vs the source's.
            let of = (res & sign_bit(*size) != 0) ^ (av & sign_bit(*size) != 0);
            apply(
                &mut cpu.flags,
                *set_flags,
                &shift_result(res, *size, cf, of),
            );
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn exec_sar(
    cpu: &mut CpuState,
    temps: &mut [u64],
    dst: &Temp,
    a: &Val,
    b: &Val,
    size: &u8,
    set_flags: &FlagMask,
) -> Option<InterpResult> {
    let vm = read_val(*a, &*temps) & mask(*size);
    let cnt = read_val(*b, &*temps) & shift_mask(*size);
    let res = (sign_extend(vm, *size) as i64 >> cnt) as u64 & mask(*size);
    temps[*dst as usize] = res;
    if !set_flags.is_none() && cnt != 0 {
        // CF = last bit shifted out. For SAR the operand is sign-extended, so once the count
        // reaches the operand width the bit shifted out is the sign bit — use the sign-extended
        // value, not the width-masked `vm` (which would read 0 past its top bit).
        let cf = (sign_extend(vm, *size) >> (cnt - 1)) & 1 != 0;
        apply(
            &mut cpu.flags,
            *set_flags,
            &shift_result(res, *size, cf, false),
        );
    }
    None
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_sext(temps: &mut [u64], dst: &Temp, a: &Val, from: &u8) -> Option<InterpResult> {
    temps[*dst as usize] = sign_extend(read_val(*a, &*temps), *from);
    None
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_bswap(
    temps: &mut [u64],
    dst: &Temp,
    a: &Val,
    size: &u8,
) -> Option<InterpResult> {
    let v = read_val(*a, &*temps);
    // `size` is the operand width: 8/4 for real `bswap`, 2 for a 16-bit `movbe`
    // (which reuses this op). The swap must be over exactly `size` bytes — a 16-bit
    // movbe needs a 2-byte swap, NOT a 32-bit one, which leaves the stored low half
    // zero and disagrees with hardware on `movbe [mem],r16`.
    temps[*dst as usize] = match *size {
        8 => v.swap_bytes(),
        2 => (v as u16).swap_bytes() as u64,
        _ => (v as u32).swap_bytes() as u64,
    };
    None
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_rol(
    cpu: &mut CpuState,
    temps: &mut [u64],
    dst: &Temp,
    a: &Val,
    b: &Val,
    size: &u8,
    set_flags: &FlagMask,
) -> Option<InterpResult> {
    let vm = read_val(*a, &*temps) & mask(*size);
    let cnt = read_val(*b, &*temps) & shift_mask(*size);
    let res = rotl(vm, cnt as u32, *size);
    temps[*dst as usize] = res;
    if !set_flags.is_none() && cnt != 0 {
        let cf = res & 1 != 0;
        let of = (res & sign_bit(*size) != 0) ^ cf; // count==1 rule
        apply(&mut cpu.flags, *set_flags, &cf_of(res, cf, of));
    }
    None
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_ror(
    cpu: &mut CpuState,
    temps: &mut [u64],
    dst: &Temp,
    a: &Val,
    b: &Val,
    size: &u8,
    set_flags: &FlagMask,
) -> Option<InterpResult> {
    let vm = read_val(*a, &*temps) & mask(*size);
    let cnt = read_val(*b, &*temps) & shift_mask(*size);
    let res = rotr(vm, cnt as u32, *size);
    temps[*dst as usize] = res;
    if !set_flags.is_none() && cnt != 0 {
        let n = *size * 8;
        let cf = res & sign_bit(*size) != 0;
        let of = cf ^ (res >> (n - 2) & 1 != 0); // top two bits differ
        apply(&mut cpu.flags, *set_flags, &cf_of(res, cf, of));
    }
    None
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_rcl(
    cpu: &mut CpuState,
    temps: &mut [u64],
    dst: &Temp,
    a: &Val,
    b: &Val,
    size: &u8,
    set_flags: &FlagMask,
) -> Option<InterpResult> {
    // Rotate left through CF: a (size*8 + 1)-bit rotate including carry-in.
    let vm = read_val(*a, &*temps) & mask(*size);
    let bits = *size as u32 * 8;
    let cnt = (read_val(*b, &*temps) as u32 & shift_mask(*size) as u32) % (bits + 1);
    let (res, cf) = rcl(vm, cnt, cpu.flags.cf, *size);
    temps[*dst as usize] = res;
    if !set_flags.is_none() && cnt != 0 {
        // Left rotate: OF = CF-out XOR MSB(result) (defined for count 1).
        let of = cf ^ (res & sign_bit(*size) != 0);
        apply(&mut cpu.flags, *set_flags, &cf_of(res, cf, of));
    }
    None
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_rcr(
    cpu: &mut CpuState,
    temps: &mut [u64],
    dst: &Temp,
    a: &Val,
    b: &Val,
    size: &u8,
    set_flags: &FlagMask,
) -> Option<InterpResult> {
    // Rotate right through CF (Go's div-by-constant carry fold).
    let vm = read_val(*a, &*temps) & mask(*size);
    let bits = *size as u32 * 8;
    let cnt = (read_val(*b, &*temps) as u32 & shift_mask(*size) as u32) % (bits + 1);
    let (res, cf) = rcr(vm, cnt, cpu.flags.cf, *size);
    temps[*dst as usize] = res;
    if !set_flags.is_none() && cnt != 0 {
        // Right rotate: OF = XOR of the top two result bits (defined for count 1).
        let n = *size * 8;
        let of = (res & sign_bit(*size) != 0) ^ (res >> (n - 2) & 1 != 0);
        apply(&mut cpu.flags, *set_flags, &cf_of(res, cf, of));
    }
    None
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_mul(
    cpu: &mut CpuState,
    temps: &mut [u64],
    lo: &Temp,
    hi: &Temp,
    a: &Val,
    b: &Val,
    size: &u8,
    signed: &bool,
    set_flags: &FlagMask,
) -> Option<InterpResult> {
    let m = mask(*size);
    let n = *size * 8;
    let (va, vb) = (read_val(*a, &*temps) & m, read_val(*b, &*temps) & m);
    let (lo_v, hi_v, overflow) = if *signed {
        let p = sign_extend(va, *size) as i64 as i128 * sign_extend(vb, *size) as i64 as i128;
        let lo_v = p as u64 & m;
        let hi_v = (p >> n) as u64 & m;
        (lo_v, hi_v, p != sign_extend(lo_v, *size) as i64 as i128)
    } else {
        let p = va as u128 * vb as u128;
        let lo_v = p as u64 & m;
        let hi_v = (p >> n) as u64 & m;
        (lo_v, hi_v, hi_v != 0)
    };
    temps[*lo as usize] = lo_v;
    temps[*hi as usize] = hi_v;
    if !set_flags.is_none() {
        // Only CF/OF are defined (the CF_OF mask); the rest are ignored.
        let r = AluResult {
            res: lo_v,
            cf: overflow,
            pf: false,
            af: false,
            zf: false,
            sf: false,
            of: overflow,
        };
        apply(&mut cpu.flags, *set_flags, &r);
    }
    None
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_div(
    cpu: &mut CpuState,
    temps: &mut [u64],
    cur_addr: u64,
    quot: &Temp,
    rem: &Temp,
    hi: &Val,
    lo: &Val,
    divisor: &Val,
    size: &u8,
    signed: &bool,
) -> Option<InterpResult> {
    let hv = read_val(*hi, &*temps);
    let lv = read_val(*lo, &*temps);
    let dv = read_val(*divisor, &*temps);
    match divide(hv, lv, dv, *size, *signed) {
        Some((q, r)) => {
            temps[*quot as usize] = q;
            temps[*rem as usize] = r;
        }
        // #DE: RIP on the faulting div; nothing committed to registers yet.
        None => {
            cpu.rip = cur_addr;
            return Some(InterpResult::exit(crate::fault::exception_exit(
                cur_addr,
                0,
                crate::FaultCause::Divide,
            )));
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_get_cond(
    cpu: &mut CpuState,
    temps: &mut [u64],
    dst: &Temp,
    cond: &Cond,
) -> Option<InterpResult> {
    temps[*dst as usize] = eval_cond(*cond, &cpu.flags) as u64;
    None
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_load<const MACHINE: bool, const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER, impl InterpreterMemory>,
    temps: &mut [u64],
    cur_addr: u64,
    dst: &Temp,
    addr: &Val,
    size: &u8,
) -> Option<InterpResult> {
    let _ordering = ordering::Barrier::<ORDER>::new(Some(ordering::Direction::Read), false);
    let mem = mem.unordered();
    let a = read_val(*addr, &*temps);
    if MACHINE {
        // Decision-16 §3: a later write can suspend this same instruction.
        // Keep its completed device read, and consume answers only at MMIO.
        if let Some(value) = cpu.mmio_part(cur_addr, a) {
            temps[*dst as usize] = value & mask(*size);
            return None;
        }
        match mem.read(a, *size) {
            Ok(value) => temps[*dst as usize] = value,
            Err(MemTrap::Mmio) if cpu.pending_mmio.is_some() => {
                let value = cpu.pending_mmio.take().unwrap() & mask(*size);
                cpu.record_mmio_part(cur_addr, a, value);
                temps[*dst as usize] = value;
            }
            Err(trap) => return Some(trap_out(cpu, cur_addr, trap, a, *size, AccessKind::Read, 0)),
        }
        return None;
    }
    // Resume after an MMIO read (§5.2): the block re-executes from the
    // faulting instruction, so this first load consumes the value the
    // embedder supplied via `complete_mmio_read` instead of re-trapping.
    if let Some(v) = cpu.pending_mmio.take() {
        temps[*dst as usize] = v & mask(*size);
    } else {
        match mem.read(a, *size) {
            Ok(v) => temps[*dst as usize] = v,
            Err(t) => return Some(trap_out(cpu, cur_addr, t, a, *size, AccessKind::Read, 0)),
        }
    }
    None
}

/// The load of an application walk with an MMIO retry open (spec.md §5.2).
///
/// The answer the retry consumes is recorded in the retry ledger, bound to its
/// request: the owning instruction (RIP, CS and an IR fingerprint) and the
/// access's address, size and direction (`bound_answer_key`). A later stop of the
/// same instruction, such as the write of a read-modify-write, then re-executes it
/// without asking the device again. A changed instruction at the same RIP, or any
/// stop other than this instruction's own device stops, drops the answer
/// (`CpuState::finish_mmio_retry`). The recorded answer is consulted before the
/// pending one: it is the answer to this access, while a pending answer belongs
/// to the access that stopped last. It is used only while the access still stops
/// at a Trap region; a region remapped as RAM, or unmapped, drops it
/// (`bound_answer`). The alias forms that keep a multi-read ledger
/// (`unified_read_retry`) take the Machine load. The split re-execution after a
/// containment miss also runs here, with no answer pending or recorded.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_load_retry<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER, impl InterpreterMemory>,
    temps: &mut [u64],
    cur_addr: u64,
    ir: &IrBlock,
    dst: &Temp,
    addr: &Val,
    size: &u8,
) -> Option<InterpResult> {
    if ir.mode == crate::CpuMode::Real16 && super::legacy::unified_read_retry(&ir.ops, cur_addr) {
        return exec_load::<true, ROUTED, ORDER>(cpu, mem, temps, cur_addr, dst, addr, size);
    }
    let _ordering = ordering::Barrier::<ORDER>::new(Some(ordering::Direction::Read), false);
    let mem = mem.unordered();
    let a = read_val(*addr, &*temps);
    let key = crate::state::bound_answer_key(a, *size, false);
    if let Some(v) = bound_answer(cpu, ir, cur_addr, key, mem.0.normal(), a, *size) {
        temps[*dst as usize] = v & mask(*size);
    } else if let Some(v) = cpu.pending_mmio.take() {
        let v = v & mask(*size);
        if cpu.mmio_retry_active {
            record_bound_answer(cpu, ir, cur_addr, key, v);
        }
        temps[*dst as usize] = v;
    } else {
        match mem.read(a, *size) {
            Ok(v) => temps[*dst as usize] = v,
            Err(t) => return Some(trap_out(cpu, cur_addr, t, a, *size, AccessKind::Read, 0)),
        }
    }
    None
}

/// A fingerprint of the instruction at `cur_addr` in `ir`: its lifted operations
/// and the decode mode. Equal fingerprints mean the same accesses from the same
/// registers; code rewritten at the same RIP gives another fingerprint.
#[cold]
#[inline(never)]
fn instruction_fingerprint(ir: &IrBlock, cur_addr: u64) -> u64 {
    struct Fnv(u64);
    impl std::fmt::Write for Fnv {
        fn write_str(&mut self, text: &str) -> std::fmt::Result {
            for byte in text.bytes() {
                self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
            }
            Ok(())
        }
    }
    use std::fmt::Write as _;
    let mut hash = Fnv(0xcbf2_9ce4_8422_2325);
    let start = ir
        .ops
        .iter()
        .position(|op| matches!(op, IrOp::InsnStart { guest_addr } if *guest_addr == cur_addr))
        .unwrap_or(0);
    let _ = write!(hash, "{:?}", ir.mode);
    for (i, op) in ir.ops[start..].iter().enumerate() {
        if i != 0 && matches!(op, IrOp::InsnStart { .. }) {
            break;
        }
        let _ = write!(hash, "{op:?};");
    }
    hash.0
}

/// The recorded answer for `key`, the access of `size` bytes at `addr`, if the
/// owning instruction is unchanged and the access still stops at a Trap region.
/// A different instruction at the owner's RIP drops every bound answer. A
/// recorded answer is the device's: remapped as RAM, or unmapped, the access
/// drops it and runs as it would without one.
#[cold]
#[inline(never)]
fn bound_answer(
    cpu: &mut CpuState,
    ir: &IrBlock,
    cur_addr: u64,
    key: u64,
    memory: &Memory,
    addr: u64,
    size: u8,
) -> Option<u64> {
    let value = cpu.mmio_part(cur_addr, key)?;
    if cpu.owns_mmio_retry(cur_addr)
        && cpu.mmio_part(cur_addr, crate::state::bound_fingerprint_key())
            == Some(instruction_fingerprint(ir, cur_addr))
        && matches!(memory.region_kind(addr, size), Ok(crate::RegionKind::Trap))
    {
        Some(value)
    } else {
        cpu.drop_bound_answers();
        None
    }
}

/// Record a consumed answer bound to its request and to the owning instruction.
#[cold]
#[inline(never)]
fn record_bound_answer(cpu: &mut CpuState, ir: &IrBlock, cur_addr: u64, key: u64, value: u64) {
    let fingerprint = instruction_fingerprint(ir, cur_addr);
    match cpu.mmio_part(cur_addr, crate::state::bound_fingerprint_key()) {
        Some(recorded) if recorded == fingerprint => {}
        Some(_) => {
            cpu.drop_bound_answers();
            cpu.record_mmio_part(cur_addr, crate::state::bound_fingerprint_key(), fingerprint);
        }
        None => cpu.record_mmio_part(cur_addr, crate::state::bound_fingerprint_key(), fingerprint),
    }
    cpu.record_mmio_part(cur_addr, key, value);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_store<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER, impl InterpreterMemory>,
    temps: &mut [u64],
    cur_addr: u64,
    addr: &Val,
    src: &Val,
    size: &u8,
) -> Option<InterpResult> {
    let _ordering = ordering::Barrier::<ORDER>::new(Some(ordering::Direction::Write), false);
    let mem = mem.unordered();
    let a = read_val(*addr, &*temps);
    let v = read_val(*src, &*temps);
    if let Err(t) = mem.write(a, v, *size) {
        // Resume after an MMIO write (§5.2): the block re-executes from
        // the faulting store. If the embedder acknowledged it via
        // `complete_mmio_write`, the side effect is already done — consume
        // the ack and continue instead of re-trapping.
        if t == MemTrap::Mmio && cpu.pending_mmio_write {
            cpu.pending_mmio_write = false;
        } else {
            // §16; SDM Vol 3A §7.5: a store crossing a mapping boundary reports the first
            // unavailable byte after the whole transfer has failed atomically.
            let at = if t == MemTrap::Unmapped {
                (0..*size)
                    .map(|i| a.wrapping_add(u64::from(i)))
                    .find(|&at| {
                        matches!(mem.probe_write_mode::<false>(at, 1), Err(MemTrap::Unmapped))
                    })
                    .unwrap_or(a)
            } else {
                a
            };
            return Some(trap_out(cpu, cur_addr, t, at, *size, AccessKind::Write, v));
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_atomic_rmw<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER, impl InterpreterMemory>,
    temps: &mut [u64],
    cur_addr: u64,
    ir: &IrBlock,
    old: &Temp,
    addr: &Val,
    src: &Val,
    size: &u8,
    op: &RmwOp,
) -> Option<InterpResult> {
    let a = read_val(*addr, &*temps);
    let s = read_val(*src, &*temps);
    match mem.atomic_rmw(a, s, *size, *op) {
        Ok(prev) => temps[*old as usize] = prev,
        Err(MemTrap::Mmio) if locked_bus_transfers(ir) => {
            match locked_device_operand(cpu, mem, cur_addr, ir, a, s, *size, LockedUpdate::Rmw(*op))
            {
                Ok(prev) => temps[*old as usize] = prev,
                Err(exit) => return Some(exit),
            }
        }
        Err(MemTrap::Mmio) => return Some(atomic_mmio_exit(cpu, mem, cur_addr, ir)),
        Err(t) => return Some(trap_out(cpu, cur_addr, t, a, *size, AccessKind::Write, s)),
    }
    None
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_atomic_cas<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER, impl InterpreterMemory>,
    temps: &mut [u64],
    cur_addr: u64,
    ir: &IrBlock,
    old: &Temp,
    addr: &Val,
    expected: &Val,
    src: &Val,
    size: &u8,
) -> Option<InterpResult> {
    let a = read_val(*addr, &*temps);
    let (exp, s) = (read_val(*expected, &*temps), read_val(*src, &*temps));
    match mem.atomic_cas(a, exp, s, *size) {
        Ok(prev) => temps[*old as usize] = prev,
        Err(MemTrap::Mmio) if locked_bus_transfers(ir) => {
            match locked_device_operand(cpu, mem, cur_addr, ir, a, s, *size, LockedUpdate::Cas(exp))
            {
                Ok(prev) => temps[*old as usize] = prev,
                Err(exit) => return Some(exit),
            }
        }
        Err(MemTrap::Mmio) => return Some(atomic_mmio_exit(cpu, mem, cur_addr, ir)),
        Err(t) => return Some(trap_out(cpu, cur_addr, t, a, *size, AccessKind::Write, s)),
    }
    None
}

/// Whether a locked operand on a Trap region becomes bus transfers: Real16
/// (the alias) and Machine IR. Long64 and Compat32 keep
/// `Exit::UnknownInstruction` (`atomic_mmio_exit`): their compiled code is
/// unchanged and the embedder emulates the instruction (spec.md §5.2).
#[inline(always)]
pub(crate) fn locked_bus_transfers(ir: &IrBlock) -> bool {
    ir.mode == crate::CpuMode::Real16 || ir.legacy_context.is_some()
}

/// The value a locked operand on a Trap region is written with.
#[derive(Clone, Copy)]
pub(crate) enum LockedUpdate {
    /// XCHG, XADD and the LOCK-prefixed ALU and bit forms.
    Rmw(RmwOp),
    /// CMPXCHG, with the expected value: the destination receives a write cycle
    /// either way, and a failed comparison writes the value read back (SDM Vol 2A
    /// §3.3 CMPXCHG, Description).
    Cas(u64),
}

/// A locked or implicitly locked operand whose bytes touch a Trap region.
///
/// Real16 and Machine IR perform the bus transaction as the processor does: one
/// read of the operand, then one write of the updated value (SDM Vol 3A
/// §§11.1.2.1–11.1.2.2; Vol 2A §3.3 CMPXCHG: "the processor never produces a
/// locked read without also producing a locked write"; 80286 PRM Appendix B,
/// XCHG p. B-114). Each half is an ordinary MMIO exit, and LOCK# is not
/// modelled: the vCPU is stopped across both exits, so ordering against devices
/// and DMA stays with the embedder (spec.md §5.2). No host memory is accessed
/// here, so no ordering barrier applies. The read answer is bound in the retry
/// ledger (see [`exec_load_retry`]), so the retry after the write acknowledgement
/// does not read the device again. An operand that is not inside one Trap region
/// returns `Exit::UnknownInstruction`, as before: the processor performs it in as
/// many bus cycles as necessary (SDM Vol 3A §11.1.2.2), which separate exits do
/// not express yet. Returns the value read.
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn locked_device_operand(
    cpu: &mut CpuState,
    mem: &Memory,
    cur_addr: u64,
    ir: &IrBlock,
    addr: u64,
    src: u64,
    size: u8,
    update: LockedUpdate,
) -> Result<u64, InterpResult> {
    if !matches!(mem.region_kind(addr, size), Ok(crate::RegionKind::Trap)) {
        return Err(unknown_atomic_exit(cpu, mem, cur_addr, ir));
    }
    let m = mask(size);
    let key = crate::state::bound_answer_key(addr, size, false);
    let prev = if let Some(value) = bound_answer(cpu, ir, cur_addr, key, mem, addr, size) {
        value & m
    } else if let Some(value) = cpu.pending_mmio.take() {
        let value = value & m;
        record_bound_answer(cpu, ir, cur_addr, key, value);
        value
    } else {
        return Err(trap_out(
            cpu,
            cur_addr,
            MemTrap::Mmio,
            addr,
            size,
            AccessKind::Read,
            0,
        ));
    };
    let value = match update {
        LockedUpdate::Rmw(RmwOp::Add) => prev.wrapping_add(src),
        LockedUpdate::Rmw(RmwOp::Sub) => prev.wrapping_sub(src),
        LockedUpdate::Rmw(RmwOp::Rsub) => src.wrapping_sub(prev),
        LockedUpdate::Rmw(RmwOp::And) => prev & src,
        LockedUpdate::Rmw(RmwOp::Or) => prev | src,
        LockedUpdate::Rmw(RmwOp::Xor) => prev ^ src,
        LockedUpdate::Rmw(RmwOp::Xchg) => src,
        LockedUpdate::Cas(expected) if prev == expected & m => src,
        LockedUpdate::Cas(_) => prev,
    } & m;
    if cpu.pending_mmio_write {
        cpu.pending_mmio_write = false;
        Ok(prev)
    } else {
        Err(trap_out(
            cpu,
            cur_addr,
            MemTrap::Mmio,
            addr,
            size,
            AccessKind::Write,
            value,
        ))
    }
}

#[cold]
fn atomic_mmio_exit<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER, impl InterpreterMemory>,
    cur_addr: u64,
    ir: &IrBlock,
) -> InterpResult {
    unknown_atomic_exit(cpu, mem, cur_addr, ir)
}

/// `Exit::UnknownInstruction` with the instruction's exact bytes and length, RIP
/// left on it: embedder emulation of an atomic operation on a device operand.
#[cold]
#[inline(never)]
fn unknown_atomic_exit(
    cpu: &mut CpuState,
    mem: &Memory,
    cur_addr: u64,
    ir: &IrBlock,
) -> InterpResult {
    // Engine policy (§5.2) for Long64/Compat32, and for a Real16 or Machine
    // operand only partly in a Trap region: separate device exits do not express
    // the transaction there. Leave emulation to the embedder, as for paired
    // compare-exchange.
    let next = ir.ops.iter().find_map(|op| match op {
        IrOp::InsnStart { guest_addr } if *guest_addr > cur_addr => Some(*guest_addr),
        _ => None,
    });
    let len = (next.unwrap_or(ir.guest_start + u64::from(ir.guest_len)) - cur_addr) as u8;
    let at = crate::lift::FetchAddr::for_mode(ir.mode, cur_addr, cpu.cs).pa;
    let mut bytes = [0; 15];
    let mut copied = 0;
    while copied < usize::from(len) {
        let Ok(code) = mem.code_slice(at + copied as u64, usize::from(len) - copied) else {
            break;
        };
        if code.is_empty() {
            break;
        }
        bytes[copied..copied + code.len()].copy_from_slice(&code);
        copied += code.len();
    }
    cpu.rip = cur_addr;
    InterpResult::exit(Exit::unknown_instruction(cur_addr, bytes, len))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_bt(
    cpu: &mut CpuState,
    temps: &mut [u64],
    result: &Temp,
    a: &Val,
    bit: &Val,
    size: &u8,
    op: &BtOp,
) -> Option<InterpResult> {
    let av = read_val(*a, &*temps);
    let b = read_val(*bit, &*temps) & (*size as u64 * 8 - 1);
    cpu.flags.cf = (av >> b) & 1 != 0;
    let m = 1u64 << b;
    let r = match op {
        BtOp::Test => av,
        BtOp::Set => av | m,
        BtOp::Reset => av & !m,
        BtOp::Complement => av ^ m,
    };
    temps[*result as usize] = r & mask(*size);
    None
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_cpuid(cpu: &mut CpuState) -> Option<InterpResult> {
    cpuid_run(cpu);
    None
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_xgetbv(cpu: &mut CpuState, cur_addr: u64) -> Option<InterpResult> {
    if let Some(error) = crate::x87::xstate::xgetbv_fault(cpu) {
        cpu.rip = cur_addr;
        return Some(InterpResult::exit(xstate_exit(cur_addr, error)));
    }
    xgetbv_run(cpu);
    None
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_popcnt(
    cpu: &mut CpuState,
    temps: &mut [u64],
    dst: &Temp,
    src: &Val,
    size: &u8,
) -> Option<InterpResult> {
    let s = read_val(*src, &*temps) & mask(*size);
    temps[*dst as usize] = s.count_ones() as u64;
    cpu.flags.zf = s == 0;
    cpu.flags.cf = false;
    cpu.flags.of = false;
    cpu.flags.sf = false;
    cpu.flags.set_af(false);
    cpu.flags.set_pf(false);
    None
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_crc32(
    temps: &mut [u64],
    dst: &Temp,
    crc: &Val,
    src: &Val,
    bytes: &u8,
) -> Option<InterpResult> {
    let c = read_val(*crc, &*temps) as u32;
    let s = read_val(*src, &*temps);
    temps[*dst as usize] = crc32c(c, s, *bytes) as u64;
    None
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_bmi(
    cpu: &mut CpuState,
    temps: &mut [u64],
    dst: &Temp,
    a: &Val,
    b: &Val,
    size: &u8,
    op: &BmiOp,
) -> Option<InterpResult> {
    let av = read_val(*a, &*temps);
    let bv = read_val(*b, &*temps);
    let (r, cf) = bmi_result(av, bv, *size, *op);
    if op.writes_flags() {
        let bits = *size as u32 * 8;
        cpu.flags.cf = cf;
        cpu.flags.zf = r == 0;
        cpu.flags.sf = (r >> (bits - 1)) & 1 != 0;
        cpu.flags.of = false;
    }
    temps[*dst as usize] = r;
    None
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_bit_scan(
    cpu: &mut CpuState,
    temps: &mut [u64],
    dst: &Temp,
    src: &Val,
    old: &Val,
    size: &u8,
    op: &BitScanOp,
) -> Option<InterpResult> {
    use crate::ir::BitScanOp::*;
    let bits = *size as u64 * 8;
    let s = read_val(*src, &*temps) & mask(*size);
    let r = match op {
        Bsf | Bsr if s == 0 => {
            // Destination preserved; only ZF set.
            cpu.flags.zf = true;
            read_val(*old, &*temps) & mask(*size)
        }
        Bsf => {
            cpu.flags.zf = false;
            s.trailing_zeros() as u64
        }
        Bsr => {
            cpu.flags.zf = false;
            63 - s.leading_zeros() as u64
        }
        Tzcnt => {
            let r = if s == 0 {
                bits
            } else {
                s.trailing_zeros() as u64
            };
            cpu.flags.cf = s == 0;
            cpu.flags.zf = r == 0;
            r
        }
        Lzcnt => {
            let r = s.leading_zeros() as u64 - (64 - bits);
            cpu.flags.cf = s == 0;
            cpu.flags.zf = r == 0;
            r
        }
    };
    temps[*dst as usize] = r;
    None
}

/// Decimal/ASCII accumulator adjust (§17.6): `daa`/`das`/`aaa`/`aas`/`aam`/`aad`,
/// following the 80286-defined flag behaviour. The flags each leaves *undefined* (OF
/// for daa/das; OF/SF/ZF/PF for aaa/aas; OF/AF/CF for aam/aad) are irrelevant to both
/// oracles — the 8088 corpus masks them off, and the Unicorn `MODE_16` differential
/// does not compare flags — so only the defined flags and the AL/AH/AX result are
/// pinned. `aam` with a zero base divides by zero and raises `#DE` (a fault: RIP stays
/// on the instruction, delivered in-guest through the IVT by `step_instruction`).
pub(crate) fn exec_bcd(cpu: &mut CpuState, cur_addr: u64, kind: &BcdKind) -> Option<InterpResult> {
    let al = (cpu.gpr[RAX] & 0xFF) as u8;
    match kind {
        BcdKind::Daa | BcdKind::Das => {
            let sub = matches!(kind, BcdKind::Das);
            let old_al = al;
            let old_cf = cpu.flags.cf;
            let mut new_al = al;
            cpu.flags.cf = false;
            if (al & 0x0F) > 9 || cpu.flags.af() {
                let (r, carry) = if sub {
                    new_al.overflowing_sub(6)
                } else {
                    new_al.overflowing_add(6)
                };
                new_al = r;
                cpu.flags.cf = old_cf || carry;
                cpu.flags.set_af(true);
            } else {
                cpu.flags.set_af(false);
            }
            if old_al > 0x99 || old_cf {
                new_al = if sub {
                    new_al.wrapping_sub(0x60)
                } else {
                    new_al.wrapping_add(0x60)
                };
                cpu.flags.cf = true;
            }
            cpu.gpr[RAX] = (cpu.gpr[RAX] & !0xFF) | new_al as u64;
            cpu.flags.sf = new_al & 0x80 != 0;
            cpu.flags.zf = new_al == 0;
            cpu.flags.set_pf(parity(new_al as u64));
            cpu.flags.of = false;
        }
        BcdKind::Aaa | BcdKind::Aas => {
            // Intel: adjust ⇒ `AX ± 0x106` (the ±6 on AL carries into AH, plus the ±1 on
            // AH), then `AL &= 0x0F`. AAA adds, AAS subtracts.
            let mut ax = (cpu.gpr[RAX] & 0xFFFF) as u16;
            if (al & 0x0F) > 9 || cpu.flags.af() {
                ax = if matches!(kind, BcdKind::Aas) {
                    ax.wrapping_sub(0x106)
                } else {
                    ax.wrapping_add(0x106)
                };
                cpu.flags.set_af(true);
                cpu.flags.cf = true;
            } else {
                cpu.flags.set_af(false);
                cpu.flags.cf = false;
            }
            ax &= 0xFF0F;
            cpu.gpr[RAX] = (cpu.gpr[RAX] & !0xFFFF) | ax as u64;
        }
        BcdKind::Aam(base) => {
            if *base == 0 {
                // #DE (divide error): RIP on the faulting instruction (80286 fault).
                cpu.rip = cur_addr;
                return Some(InterpResult::exit(crate::fault::exception_exit(
                    cur_addr,
                    0,
                    crate::FaultCause::Divide,
                )));
            }
            let ah = al / base;
            let new_al = al % base;
            cpu.gpr[RAX] = (cpu.gpr[RAX] & !0xFFFF) | ((ah as u64) << 8) | new_al as u64;
            cpu.flags.sf = new_al & 0x80 != 0;
            cpu.flags.zf = new_al == 0;
            cpu.flags.set_pf(parity(new_al as u64));
        }
        BcdKind::Aad(base) => {
            let ah = ((cpu.gpr[RAX] >> 8) & 0xFF) as u8;
            let new_al = al.wrapping_add(ah.wrapping_mul(*base));
            cpu.gpr[RAX] = (cpu.gpr[RAX] & !0xFFFF) | new_al as u64;
            cpu.flags.sf = new_al & 0x80 != 0;
            cpu.flags.zf = new_al == 0;
            cpu.flags.set_pf(parity(new_al as u64));
        }
    }
    None
}
