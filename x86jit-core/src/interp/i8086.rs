//! Lift-resolved real-mode primitives (decision-23 §3; spec.md §17.10).
use super::*;
use crate::legacy::{LegacyOp as L, SegmentId};
use std::cell::RefCell;

// 8086 Family UM §2.3 p. 2-12 and §2.8 p. 2-68.
pub(super) fn physical(tag: u64, byte: u8) -> u64 {
    ((tag >> 32) + (tag.wrapping_add(u64::from(byte)) & 0xffff)) & 0xfffff
}
fn bus_tag(selector: u16, offset: u64) -> u64 {
    (u64::from(selector) << 36) | (offset & 0xffff)
}

/// A tag bit: the model's data bus moves a word in one cycle (the catalogue's
/// `data_bus_bits`, carried by the lifted operation as `word_bus`). A tag holds
/// its segment base at bit 36 and its offset in the low sixteen bits, and an
/// offset sum carries at most into bit 16, so [`physical`] never sees bit 24.
pub(crate) const WORD_BUS: u64 = 1 << 24;
pub(crate) const fn bus(word_bus: bool) -> u64 {
    (word_bus as u64) << 24
}
// The retry-ledger key of a word device transfer at `addr` (bit 63 marks a
// write, as for a byte transfer). Bit 60 is outside every byte, string,
// coprocessor and bound-answer key (`state::is_bound_key`).
const WORD_KEY: u64 = 1 << 60;

// The word of the operand that byte `i` belongs to, as (its address, the
// byte's index in it), when the bus moves that word in one cycle: a sixteen-bit
// bus, the word's low byte at an even address (8086 Family UM §2.3 p. 2-15;
// NEC V20/V30 UM §4 pp. 4-2–4-3: "Word to even address", one bus cycle), and
// both bytes inside one Trap region. The operand's words start at its even
// byte indices, as a frame or a far pointer is a sequence of words. A word at
// an odd address is two byte cycles, the low byte at the address first, then
// the high byte at the address plus one (UM §4.2 p. 4-15; NEC UM §4 p. 4-3),
// which is the byte order of this walk. An even-addressed word never wraps:
// offset FFFFh and address FFFFFh are odd. A word over two regions, or over a
// device and RAM, keeps its byte transfers (an engine limit: one exit names
// one region).
#[cold]
#[inline(never)]
fn device_word(mem: &Memory, tag: u64, i: u8, size: u8) -> Option<(u64, u8)> {
    let first = i & !1;
    let addr = physical(tag, first);
    if tag & WORD_BUS == 0
        || first + 1 >= size
        || addr & 1 != 0
        || physical(tag, first + 1) != addr + 1
        || !matches!(mem.region_kind(addr, 2), Ok(crate::RegionKind::Trap))
    {
        return None;
    }
    Some((addr, i - first))
}

#[cold]
#[inline(never)]
fn transfer_exit(
    cpu: &mut CpuState,
    trap: MemTrap,
    addr: u64,
    size: u8,
    write: bool,
    value: u64,
) -> Exit {
    let StepResult::Exit(exit) = trap_out(
        cpu,
        cpu.rip,
        trap,
        addr,
        size,
        if write {
            AccessKind::Write
        } else {
            AccessKind::Read
        },
        value,
    )
    .into_step() else {
        unreachable!()
    };
    exit
}
fn memory_exit(cpu: &mut CpuState, trap: MemTrap, addr: u64, write: bool, value: u64) -> Exit {
    transfer_exit(cpu, trap, addr, 1, write, value)
}

// Byte `i` of a read that met a Trap region on a sixteen-bit bus. A byte of a
// device word takes its value from the word's one transfer: its recorded
// answer, the pending answer (at the low byte), or a word `MmioRead`. Any other
// byte is the byte transfer of the eight-bit bus.
fn device_byte(cpu: &mut CpuState, mem: &Memory, tag: u64, i: u8, size: u8) -> Result<u64, Exit> {
    let Some((word, byte)) = device_word(mem, tag, i, size) else {
        let addr = physical(tag, i);
        let Some(value) = cpu.pending_mmio.take() else {
            return Err(memory_exit(cpu, MemTrap::Mmio, addr, false, 0));
        };
        let value = value & 255;
        cpu.record_mmio_part(cpu.rip, addr, value);
        return Ok(value);
    };
    let key = WORD_KEY | word;
    let value = if let Some(value) = cpu.mmio_part(cpu.rip, key) {
        value
    } else if let Some(value) = cpu.pending_mmio.filter(|_| byte == 0) {
        cpu.pending_mmio = None;
        let value = value & 0xffff;
        cpu.record_mmio_part(cpu.rip, key, value);
        value
    } else {
        return Err(transfer_exit(cpu, MemTrap::Mmio, word, 2, false, 0));
    };
    Ok((value >> (8 * byte)) & 255)
}
// The rest of a read, from byte `from`, once byte `from` met a Trap region on a
// sixteen-bit bus; `result` holds the bytes before it. `read` returns into
// this, so its own loop keeps the shape it has without a device path.
#[cold]
#[inline(never)]
fn device_read<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Memory,
    tag: u64,
    size: u8,
    from: u8,
    mut result: u64,
) -> Result<u64, Exit> {
    let access = Access::<ROUTED, ORDER, _>::new(mem);
    for i in from..size {
        let addr = physical(tag, i);
        let value = if let Some(value) = cpu.mmio_part(cpu.rip, addr) {
            value
        } else {
            match access.read(addr, 1) {
                Ok(value) => value,
                Err(MemTrap::Mmio) => device_byte(cpu, mem, tag, i, size)?,
                Err(trap) => return Err(memory_exit(cpu, trap, addr, false, 0)),
            }
        };
        result |= (value & 255) << (8 * i);
    }
    Ok(result)
}
fn read<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Memory,
    tag: u64,
    size: u8,
) -> Result<u64, Exit> {
    let mem = Access::<ROUTED, ORDER, _>::new(mem);
    let mut result = 0;
    for i in 0..size {
        let addr = physical(tag, i);
        let value = if let Some(value) = cpu.mmio_part(cpu.rip, addr) {
            value
        } else {
            match mem.read(addr, 1) {
                Ok(value) => value,
                Err(MemTrap::Mmio) if tag & WORD_BUS != 0 => {
                    return device_read::<ROUTED, ORDER>(cpu, mem, tag, size, i, result)
                }
                Err(MemTrap::Mmio) if cpu.pending_mmio.is_some() => {
                    let value = cpu.pending_mmio.take().unwrap() & 255;
                    cpu.record_mmio_part(cpu.rip, addr, value);
                    value
                }
                Err(trap) => return Err(memory_exit(cpu, trap, addr, false, 0)),
            }
        };
        result |= (value & 255) << (8 * i);
    }
    Ok(result)
}
fn write<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Memory,
    tag: u64,
    size: u8,
    value: u64,
) -> Result<(), Exit> {
    // Validate the complete RAM destination before the first byte commits.
    for i in 0..size {
        let addr = physical(tag, i);
        if let Err(trap) = mem.probe_write_mode::<ROUTED>(addr, 1) {
            if trap != MemTrap::Mmio {
                return Err(memory_exit(cpu, trap, addr, true, (value >> (8 * i)) & 255));
            }
            return write_devices_first::<ROUTED, ORDER>(cpu, mem, tag, size, value, i);
        }
    }
    let access = Access::<ROUTED, ORDER, _>::new(mem);
    for i in 0..size {
        let addr = physical(tag, i);
        let key = addr | (1 << 63);
        let byte = (value >> (8 * i)) & 255;
        if cpu.mmio_part(cpu.rip, key).is_some() {
            continue;
        }
        if let Err(trap) = access.write(addr, byte, 1) {
            if trap == MemTrap::Mmio && cpu.pending_mmio_write {
                cpu.pending_mmio_write = false;
                cpu.record_mmio_part(cpu.rip, key, byte);
            } else {
                return Err(memory_exit(cpu, trap, addr, true, byte));
            }
        }
    }
    Ok(())
}

// A store whose byte `first` is its first device byte, with the bytes before it
// validated: the rest are validated, then the device bytes are written first, in
// order, and the RAM bytes once every device byte is acknowledged. A device stop
// then leaves the instruction's RAM bytes uncommitted (spec.md §16, pitfall 0),
// so the retry of a read-modify-write over RAM and device bytes reads the operand
// it read the first time. The order is an engine choice for restartability; the
// device sees no RAM transfer either way. Every store with a device byte comes
// here, so the validation loop in `write` keeps no state of its own. On a
// sixteen-bit bus a device word (`device_word`) is one transfer, made at its
// low byte and keyed as a word.
#[cold]
#[inline(never)]
fn write_devices_first<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Memory,
    tag: u64,
    size: u8,
    value: u64,
    first: u8,
) -> Result<(), Exit> {
    for i in first + 1..size {
        let addr = physical(tag, i);
        match mem.probe_write_mode::<ROUTED>(addr, 1) {
            Ok(()) | Err(MemTrap::Mmio) => {}
            Err(trap) => {
                return Err(memory_exit(cpu, trap, addr, true, (value >> (8 * i)) & 255));
            }
        }
    }
    let access = Access::<ROUTED, ORDER, _>::new(mem);
    for pass_device in [true, false] {
        for i in 0..size {
            let addr = physical(tag, i);
            if matches!(mem.probe_write_mode::<ROUTED>(addr, 1), Err(MemTrap::Mmio)) != pass_device
            {
                continue;
            }
            let (addr, size, key, data) = match device_word(mem, tag, i, size) {
                Some((_, 1)) => continue,
                Some((word, _)) => (
                    word,
                    2,
                    word | WORD_KEY | (1 << 63),
                    (value >> (8 * i)) & 0xffff,
                ),
                None => (addr, 1, addr | (1 << 63), (value >> (8 * i)) & 255),
            };
            if cpu.mmio_part(cpu.rip, key).is_some() {
                continue;
            }
            if let Err(trap) = access.write(addr, data, size) {
                if trap == MemTrap::Mmio && cpu.pending_mmio_write {
                    cpu.pending_mmio_write = false;
                    cpu.record_mmio_part(cpu.rip, key, data);
                } else {
                    return Err(transfer_exit(cpu, trap, addr, size, true, data));
                }
            }
        }
    }
    Ok(())
}

#[inline(never)]
pub(super) fn primitive<const ROUTED: bool, const ORDER: u8>(
    op: &L,
    cpu: &mut CpuState,
    mem: &Memory,
    temps: &mut [u64],
    ip: u64,
) -> Result<(), Exit> {
    match op {
        L::WrappedLoad {
            dst,
            addr,
            size,
            word_bus,
        } => {
            temps[*dst as usize] =
                read::<ROUTED, ORDER>(cpu, mem, read_val(*addr, temps) | bus(*word_bus), *size)?
        }
        L::WrappedStore {
            addr,
            value,
            size,
            word_bus,
        } => write::<ROUTED, ORDER>(
            cpu,
            mem,
            read_val(*addr, temps) | bus(*word_bus),
            *size,
            read_val(*value, temps),
        )?,
        L::LockedRmw20 {
            old,
            addr,
            src,
            size,
            op,
            bytes,
            len,
            word_bus,
        } => {
            temps[*old as usize] = locked_rmw::<ROUTED, ORDER>(
                cpu,
                mem,
                read_val(*addr, temps) | bus(*word_bus),
                read_val(*src, temps),
                *size,
                *op,
                (ip, bytes, *len),
            )?
        }
        L::FlagsImage { dst, clear, set } => {
            temps[*dst as usize] = (cpu.rflags() & !u64::from(*clear)) | u64::from(*set)
        }
        L::LoadFlagsWord { value, mask, .. } => {
            let value = read_val(*value, temps) as u16 & mask;
            cpu.flags.set_flags16(value);
            cpu.rflags_system = u64::from(value & 0x100);
        }
        // UM §2.7 p. 2-37: -128/-32768 are outside the signed quotient
        // range. §2.6 p. 2-26: the return address names the next instruction.
        L::DivideTrap {
            quot,
            rem,
            hi,
            lo,
            divisor,
            size,
            signed,
            reject_minimum,
            saved_ip,
        } => {
            let result = divide(
                temps[*hi as usize],
                read_val(*lo, temps),
                read_val(*divisor, temps),
                *size,
                *signed,
            )
            .filter(|(q, _)| !(*signed && *reject_minimum && *q == 1 << (size * 8 - 1)));
            if let Some((q, r)) = result {
                temps[*quot as usize] = q;
                temps[*rem as usize] = r;
            } else {
                cpu.rip = ip;
                let mut metadata = crate::FaultMetadata::architectural_exception(
                    u64::from(*saved_ip),
                    ip,
                    0,
                    crate::FaultClass::Trap,
                    crate::ErrorCode::NoCode,
                );
                metadata.path = crate::FaultPath::Interpreter;
                metadata.origin = crate::FaultOrigin::GuestException(crate::FaultCause::Divide);
                return Err(Exit::exception(ip, 0).with_fault_metadata(metadata));
            }
        }
        L::Coprocessor20 {
            kind,
            addr,
            sti,
            memory,
            wait,
            bytes,
            len,
            word_bus,
        } => {
            // UM §2.7 p. 2-48: CPU ESC reads/discards its memory operand.
            // iAPX 86/88,186/188 PRM §6.3 p. 6-8 identifies this as a word.
            // EM is an engine input on a system-free model, never a fake #NM.
            if !wait && cpu.fpu_emulation {
                return Err(Exit::unknown_instruction(ip, *bytes, *len));
            }
            if *memory {
                read::<ROUTED, ORDER>(cpu, mem, read_val(*addr, temps) | bus(*word_bus), 2)?;
            }
            if cpu.fpu_model == crate::FpuModel::None {
            } else if let Some(kind) = kind {
                if crate::x87::mf_pending_before(cpu, *kind) {
                    return Err(Exit::unknown_instruction(ip, *bytes, *len));
                }
                cpu.begin_mmio_retry(ip);
                // The coprocessor's transfers are byte-wide on every model
                // (ASSUMED), and a load reads its first word again after the
                // dummy read above: a known divergence from PRM §6.3 p. 6-8,
                // where the 8087 captures that word (spec.md §5.2, xj-50w.26).
                let memory = FpMemory::<ROUTED, ORDER> {
                    memory: mem,
                    fault: Cell::new(None),
                    parts: RefCell::new(fp_parts(cpu)),
                    pending_read: Cell::new(cpu.pending_mmio),
                    pending_write: Cell::new(cpu.pending_mmio_write),
                };
                let failed =
                    crate::x87::exec_x87(cpu, &memory, *kind, read_val(*addr, temps), *sti);
                cpu.pending_mmio = memory.pending_read.get();
                cpu.pending_mmio_write = memory.pending_write.get();
                save_fp_parts(cpu, ip, &memory.parts.borrow());
                if failed.is_some() {
                    let (trap, at, value, write) = memory
                        .fault
                        .get()
                        .expect("failed coprocessor transfer retains its memory error");
                    return Err(memory_exit(cpu, trap, at, write, value));
                }
            } else {
                return Err(Exit::unknown_instruction(ip, *bytes, *len));
            }
        }
        _ => unreachable!("resolved real-mode primitive"),
    }
    Ok(())
}

// 8086 Family UM §2.5 pp. 2-17–18 and NEC V20/V30 UM p. 12-173: a locked
// read-modify-write takes the engine's atomic path, as `IrOp::AtomicRmw` does
// on the 286 and 386 Machine paths (decision-23, xj-50w.12): one host atomic
// of the operand's width (`Memory::atomic_rmw`), the one RAM transfer of this
// walk wider than a byte. A device operand inside one Trap region is one read,
// then one write, of its bus transfers (`locked_device`, spec.md §5.2); one
// only partly in a Trap region is a typed refusal, and an operand over two
// adjacent regions keeps the containment miss of every locked operand. A word
// at an odd address is the same atomic operation: atomic inside one aligned
// 16-byte granule, and a plain read and write when its first byte is at 15 mod
// 16 (`Memory::locked_unaligned`, spec.md §8.2.3). A word whose bytes wrap
// their segment (offset FFFFh, UM §2.3 p. 2-12) or the twenty-bit bus (FFFFFh,
// §2.8 p. 2-68) has no single host address (`locked_split`).
#[cold]
#[inline(never)]
fn locked_rmw<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Memory,
    tag: u64,
    src: u64,
    size: u8,
    op: crate::ir::RmwOp,
    (ip, bytes, len): (u64, &[u8; 15], u8),
) -> Result<u64, Exit> {
    let first = physical(tag, 0);
    if physical(tag, size - 1) != first + u64::from(size) - 1 {
        return locked_split::<ROUTED, ORDER>(cpu, mem, tag, src, size, op, (ip, bytes, len));
    }
    match mem.atomic_rmw(first, src, size, op) {
        Ok(old) => Ok(old),
        Err(MemTrap::Mmio)
            if matches!(mem.region_kind(first, size), Ok(crate::RegionKind::Trap)) =>
        {
            locked_device(cpu, mem, tag, src, size, op)
        }
        Err(MemTrap::Mmio) => Err(Exit::unknown_instruction(ip, *bytes, len)),
        Err(trap) => {
            let StepResult::Exit(exit) =
                trap_out(cpu, cpu.rip, trap, first, size, AccessKind::Write, src).into_step()
            else {
                unreachable!()
            };
            Err(exit)
        }
    }
}

// A locked operand inside one Trap region: the read, then the write, of the
// whole operand (SDM Vol 3A §11.1.2.2 p. 11-4; 8086 Family UM §2.5 p. 2-18:
// LOCK holds the bus through the instruction). Each is the model's bus
// transfers, as every device transfer of this walk is (spec.md §5.2): one word
// transfer for an even word on a sixteen-bit bus, else one per byte, keyed in
// the instruction's retry ledger. No RAM is touched, so no host-fault routing or
// ordering barrier applies.
#[cold]
#[inline(never)]
fn locked_device(
    cpu: &mut CpuState,
    mem: &Memory,
    tag: u64,
    src: u64,
    size: u8,
    op: crate::ir::RmwOp,
) -> Result<u64, Exit> {
    let old = read::<false, 0>(cpu, mem, tag, size)?;
    write::<false, 0>(cpu, mem, tag, size, updated(op, old, src))?;
    Ok(old)
}

fn updated(op: crate::ir::RmwOp, old: u64, src: u64) -> u64 {
    use crate::ir::RmwOp;
    match op {
        RmwOp::Add => old.wrapping_add(src),
        RmwOp::Sub => old.wrapping_sub(src),
        RmwOp::Rsub => src.wrapping_sub(old),
        RmwOp::And => old & src,
        RmwOp::Or => old | src,
        RmwOp::Xor => old ^ src,
        RmwOp::Xchg => src,
    }
}

// A locked word that wraps its segment or the bus, so its first byte is at 15
// mod 16. LIMITATION: like a word that crosses an aligned 16-byte granule, it
// has no host atomic, so it is the read, then the write, of its byte
// transfers. It completes with the single-processor result, which LOCK holds
// for the whole instruction (UM §2.5 p. 2-18), but is not atomic against
// another vCPU. This is the policy of the 286 and 386 for an A20-split operand
// (spec.md §8.2.3). As in the atomic path, a word wholly in Trap regions is
// `locked_device` and one only partly in them is a typed refusal.
#[cold]
#[inline(never)]
fn locked_split<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Memory,
    tag: u64,
    src: u64,
    size: u8,
    op: crate::ir::RmwOp,
    (ip, bytes, len): (u64, &[u8; 15], u8),
) -> Result<u64, Exit> {
    let devices = (0..size)
        .filter(|&i| {
            matches!(
                mem.region_kind(physical(tag, i), 1),
                Ok(crate::RegionKind::Trap)
            )
        })
        .count();
    if devices == usize::from(size) {
        return locked_device(cpu, mem, tag, src, size, op);
    }
    if devices != 0 {
        return Err(Exit::unknown_instruction(ip, *bytes, len));
    }
    let old = read::<ROUTED, ORDER>(cpu, mem, tag, size)?;
    write::<ROUTED, ORDER>(cpu, mem, tag, size, updated(op, old, src))?;
    Ok(old)
}

/// How a NEC V20/V30 control primitive ended its instruction (decision-23 C4a).
pub(super) enum Nec {
    /// The remaining operations of the instruction run.
    Fallthrough,
    /// IP and CS are final; the walk ends at this instruction.
    Transfer,
    /// As `Transfer`, and PSW.MD is now 0: the dispatcher decides the boundary.
    ModeChange,
}

fn nec_refusal(cpu: &mut CpuState, ip: u64, operation: crate::UnsupportedSystemOperation) -> Exit {
    cpu.rip = ip;
    Exit::UnsupportedSystemContext {
        addr: ip,
        operation,
    }
}

// NEC UM p. 12-168 RETI and p. 12-151 POP PSW: the flags of the 8086 image
// (§3 p. 3-3); MD is loaded only while the write-enable latch is set.
fn nec_load_psw(cpu: &mut CpuState, value: u16) {
    cpu.flags.set_flags16(value & 0x0fd5);
    cpu.rflags_system = u64::from(value & 0x100);
}

#[inline(never)]
pub(super) fn nec<const ROUTED: bool, const ORDER: u8>(
    op: &L,
    cpu: &mut CpuState,
    mem: &Memory,
    temps: &[u64],
    ip: u64,
) -> Result<Nec, Exit> {
    use crate::legacy::{NEC_EMULATION, NEC_MD_WRITE, NEC_STANDBY};
    use crate::UnsupportedSystemOperation as U;
    let latch = cpu.legacy.reserved & NEC_MD_WRITE != 0;
    match op {
        L::CheckIndex20 {
            index,
            lower,
            upper,
            saved_ip,
            flags_set,
            word_bus,
        } => {
            let index = temps[*index as usize] as u16;
            let lower = temps[*lower as usize] as u16;
            let upper = temps[*upper as usize] as u16;
            let unsigned = lower > index || upper < index;
            let signed = (lower as i16) > (index as i16) || (upper as i16) < (index as i16);
            if unsigned != signed {
                return Err(nec_refusal(cpu, ip, U::NecIndexComparison));
            }
            if unsigned {
                interrupt::<ROUTED, ORDER>(cpu, mem, 5, *saved_ip, *flags_set, *word_bus)?;
            } else {
                cpu.rip = u64::from(*saved_ip);
            }
            Ok(Nec::Transfer)
        }
        // NEC UM p. 12-169; IE and BRK cleared as interrupt entry does
        // (p. 6-1; decision-23 rule R2, ASSUMED). With BRK = 1 at its start the
        // general break rule (p. 6-3) would meet the mode switch, which no page
        // describes: refused before commit.
        L::Brkem20 {
            vector,
            saved_ip,
            flags_set,
            word_bus,
        } => {
            if latch {
                return Err(nec_refusal(cpu, ip, U::NecNestedEmulation));
            }
            if cpu.rflags_system & 0x100 != 0 {
                return Err(nec_refusal(cpu, ip, U::NecEmulationSingleStep));
            }
            interrupt::<ROUTED, ORDER>(cpu, mem, *vector, *saved_ip, *flags_set, *word_bus)?;
            cpu.legacy.reserved =
                (cpu.legacy.reserved | NEC_EMULATION | NEC_MD_WRITE) & !NEC_STANDBY;
            Ok(Nec::ModeChange)
        }
        L::Reti20 {
            ip: target,
            cs,
            flags,
            sp,
        } => {
            let value = temps[*flags as usize] as u16;
            // A RETI that enters 8080 mode with BRK = 1 at its start: the
            // break after it is not described for emulation mode. Refused
            // before commit, as POP PSW below.
            if latch && value & 0x8000 == 0 && cpu.rflags_system & 0x100 != 0 {
                return Err(nec_refusal(cpu, ip, U::NecEmulationSingleStep));
            }
            nec_load_psw(cpu, value);
            cpu.write_gpr(RSP, temps[*sp as usize], 2);
            cpu.cs = temps[*cs as usize] as u16;
            cpu.rip = temps[*target as usize] & 0xffff;
            if latch && value & 0x8000 == 0 {
                cpu.legacy.reserved |= NEC_EMULATION;
                Ok(Nec::ModeChange)
            } else {
                Ok(Nec::Transfer)
            }
        }
        L::PopPsw20 { value } => {
            let value = temps[*value as usize] as u16;
            if latch && value & 0x8000 == 0 {
                return Err(nec_refusal(cpu, ip, U::NecPswModeWrite));
            }
            nec_load_psw(cpu, value);
            Ok(Nec::Fallthrough)
        }
        L::BitField20 {
            insert,
            offset,
            length,
            length_register,
            word_bus,
        } => {
            bit_field::<ROUTED, ORDER>(
                cpu,
                mem,
                ip,
                (*insert, *word_bus),
                *offset,
                *length,
                *length_register,
            )?;
            Ok(Nec::Fallthrough)
        }
        L::BcdString20 { op, source } => {
            bcd_string::<ROUTED, ORDER>(cpu, mem, ip, *op, *source)?;
            Ok(Nec::Fallthrough)
        }
        // NEC UM pp. 12-174–175 FPO2: the CPU computes the address and
        // starts the read; the external chip decides what the instruction
        // does. With an empty socket the CPU's part is the whole effect, as
        // for FPO1 (ESC). The generation-neutral engine defines no 66h/67h
        // instruction, so with it attached the result is not stated: a typed
        // refusal before any transfer, as is the engine's EM input.
        L::Fpo2Escape20 {
            addr,
            memory,
            bytes,
            len,
            word_bus,
        } => {
            if cpu.fpu_emulation || cpu.fpu_model != crate::FpuModel::None {
                return Err(Exit::unknown_instruction(ip, *bytes, *len));
            }
            if *memory {
                // One word read: p. 12-175 gives "Transfers: 1" and Clocks
                // 15 (V20), 15 (V30, odd address), 11 (V30, even address);
                // the manual prints that odd/even split only for word
                // transfers (TEST1 mem16 p. 12-83: 16/16/12; mem8 p. 12-82:
                // 12). FPO1 on p. 12-174 gives the same clocks.
                read::<ROUTED, ORDER>(cpu, mem, read_val(*addr, temps) | bus(*word_bus), 2)?;
            }
            Ok(Nec::Fallthrough)
        }
        _ => unreachable!("NEC control primitive"),
    }
}

// A Table 12-6 reg8 code: AL CL DL BL AH CH DH BH.
fn reg8(cpu: &CpuState, code: u8) -> u8 {
    (cpu.gpr[usize::from(code & 3)] >> if code & 4 != 0 { 8 } else { 0 }) as u8
}
fn set_reg8(cpu: &mut CpuState, code: u8, value: u8) {
    let shift = if code & 4 != 0 { 8 } else { 0 };
    let reg = &mut cpu.gpr[usize::from(code & 3)];
    *reg = (*reg & !(0xff << shift)) | (u64::from(value) << shift);
}

// NEC UM pp. 12-21–24 INS and EXT. The field is `length` + 1 bits at bit
// `offset` of the word at DS1:IY (INS) or DS0:IX (EXT), continuing into the
// next word ("Bit field data may overlap the byte boundary"); "Transfers: 2
// or 4" (INS) and "1 or 2" (EXT) read the second word only when the field
// reaches it. INS stores AW bits length:0 there; EXT loads them into AW and
// zeros the rest. "After the transfer", the offset register and IY (IX)
// are "updated to point to the next bit field": the worked examples give
// offset 3 + 5 bits -> offset 8, IY unchanged, and offset 8 + 13 bits ->
// offset 5, IY + 2. A field ending exactly at bit 16 is derived from the same
// sentence and the 0-15 offset rule: the next field is bit 0 of the next
// word, which no other offset value names. Every byte wraps inside its
// segment (§10 Fig. 10-1). Flags are undefined; the
// engine keeps them. Upper offset or length bits must be zero "for correct
// operation" and no other result is described: refused before any transfer.
#[allow(clippy::too_many_arguments)]
fn bit_field<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Memory,
    ip: u64,
    (insert, word_bus): (bool, bool),
    offset_register: u8,
    length: u8,
    length_register: bool,
) -> Result<(), Exit> {
    let offset = reg8(cpu, offset_register);
    let length = if length_register {
        reg8(cpu, length)
    } else {
        length
    };
    if (offset | length) & 0xf0 != 0 {
        return Err(nec_refusal(
            cpu,
            ip,
            crate::UnsupportedSystemOperation::NecBitFieldOperand,
        ));
    }
    let (offset, bits) = (u32::from(offset), u32::from(length) + 1);
    let (selector, index) = if insert { (cpu.es, RDI) } else { (cpu.ds, RSI) };
    let pointer = cpu.gpr[index] & 0xffff;
    let tag = bus_tag(selector, pointer) | bus(word_bus);
    let end = offset + bits;
    let size = if end > 16 { 4 } else { 2 };
    let mask = ((1u64 << bits) - 1) << offset;
    if insert {
        let old = read::<ROUTED, ORDER>(cpu, mem, tag, size)?;
        let field = ((cpu.gpr[RAX] & 0xffff) << offset) & mask;
        write::<ROUTED, ORDER>(cpu, mem, tag, size, (old & !mask) | field)?;
    } else {
        let value = read::<ROUTED, ORDER>(cpu, mem, tag, size)?;
        cpu.write_gpr(RAX, (value & mask) >> offset, 2);
    }
    set_reg8(cpu, offset_register, (end & 15) as u8);
    if end >= 16 {
        cpu.write_gpr(index, pointer.wrapping_add(2) & 0xffff, 2);
    }
    Ok(())
}

// Whether destination byte i of the strings at `src` and `dst` is, at its
// physical address, source byte j of a later step (j > i).
#[cold]
#[inline(never)]
fn bcd_order_shows(src: u64, dst: u64, bytes: u64) -> bool {
    (1..bytes).any(|j| {
        let at = physical(src + j, 0);
        (0..j).any(|i| physical(dst + i, 0) == at)
    })
}

// NEC UM pp. 12-42–44 ADD4S, SUB4S, CMP4S: the packed BCD string of CL
// digits at DS1:IY ("[DS1-spec:]dst-string"), with the one at seg:IX
// (default DS0) added, subtracted or compared, from the low byte up
// ("Transfers: 3n" for n = CL/2 bytes). Z and CY follow the result for an
// even count; V, S, AC and P are undefined and kept. IX, IY and CL are not
// named as changed. Every byte wraps inside its segment. Refused before any
// store or device transfer: CL outside 1-254 or odd (undefined digit and
// flags), an overlap that would show the unstated transfer order, a device
// byte (engine limit), a digit above 9 (found by RAM reads). An unmapped or
// unwritable byte faults before the first store.
fn bcd_string<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Memory,
    ip: u64,
    op: crate::legacy::BcdStringOp,
    source: SegmentId,
) -> Result<(), Exit> {
    use crate::UnsupportedSystemOperation as U;
    let digits = cpu.gpr[RCX] as u8;
    if digits == 0 || digits & 1 != 0 {
        return Err(nec_refusal(cpu, ip, U::NecBcdStringLength));
    }
    let store = op != crate::legacy::BcdStringOp::Cmp;
    let src = bus_tag(legacy::selector(cpu, source), cpu.gpr[RSI]);
    let dst = bus_tag(cpu.es, cpu.gpr[RDI]);
    let bytes = u64::from(digits / 2);
    // "Transfers: 3n" gives no order. If destination byte i is source byte j
    // of a later step, a store before that read and all reads first differ:
    // refused. Disjoint and identical strings, and an overlap whose shared
    // bytes are read no later than they are stored, have one result.
    if store && bcd_order_shows(src, dst, bytes) {
        return Err(nec_refusal(cpu, ip, U::NecBcdStringOverlap));
    }
    for i in 0..bytes {
        for tag in [src + i, dst + i] {
            if matches!(
                mem.region_kind(physical(tag, 0), 1),
                Ok(crate::RegionKind::Trap)
            ) {
                return Err(nec_refusal(cpu, ip, U::NecBcdStringDevice));
            }
        }
    }
    let valid = |byte: u64| byte & 0x0f <= 9 && byte >> 4 <= 9;
    for i in 0..bytes {
        let a = read::<ROUTED, ORDER>(cpu, mem, dst + i, 1)?;
        let b = read::<ROUTED, ORDER>(cpu, mem, src + i, 1)?;
        if !valid(a) || !valid(b) {
            return Err(nec_refusal(cpu, ip, U::NecBcdDigit));
        }
        if store {
            let at = physical(dst + i, 0);
            if let Err(trap) = mem.probe_write_mode::<ROUTED>(at, 1) {
                return Err(memory_exit(cpu, trap, at, true, a));
            }
        }
    }
    // One byte at a time. With the overlap refused above, the order of the
    // reads and stores no longer changes the result.
    let (mut carry, mut zero) = (0, true);
    for i in 0..bytes {
        let a = read::<ROUTED, ORDER>(cpu, mem, dst + i, 1)?;
        let b = read::<ROUTED, ORDER>(cpu, mem, src + i, 1)?;
        let mut digit = |a: u64, b: u64| {
            let value = if op == crate::legacy::BcdStringOp::Add {
                a + b + carry
            } else {
                (a + 10).wrapping_sub(b + carry)
            };
            carry = u64::from(if op == crate::legacy::BcdStringOp::Add {
                value > 9
            } else {
                value < 10
            });
            value % 10
        };
        let low = digit(a & 0x0f, b & 0x0f);
        let high = digit(a >> 4, b >> 4);
        let result = (high << 4) | low;
        zero &= result == 0;
        if store {
            write::<ROUTED, ORDER>(cpu, mem, dst + i, 1, result)?;
        }
    }
    cpu.flags.zf = zero;
    cpu.flags.cf = carry != 0;
    Ok(())
}

/// The three words at SS:SP, each byte wrapping inside the segment
/// (NEC V20/V30 UM p. 12-177 RETEM; §10 Fig. 10-1).
pub(crate) fn read_frame<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Memory,
    word_bus: bool,
) -> Result<[u16; 3], Exit> {
    let tag = bus_tag(cpu.ss, cpu.gpr[RSP]) | bus(word_bus);
    let mut words = [0; 3];
    for (i, word) in words.iter_mut().enumerate() {
        *word = read::<ROUTED, ORDER>(cpu, mem, tag + 2 * i as u64, 2)? as u16;
    }
    Ok(words)
}

// UM §2.6 pp. 2-25–28: fixed IVT, FLAGS/CS/IP frame, clear IF and TF.
// `word_bus` is the catalogue's sixteen-bit data bus: the IVT words (always
// even, NEC UM §4 p. 4-3) and an even-addressed frame are word transfers on a
// device. The frame is stored from SP-6 upward (IP, CS, FLAGS): a known
// divergence from the processor's FLAGS, CS, IP push order (UM §2.6 p. 2-26;
// UM p. A-28), visible to a device stack (spec.md §5.2, xj-50w.25).
#[inline(never)]
pub(crate) fn interrupt<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Memory,
    vector: u8,
    saved_ip: u16,
    flags_set: u16,
    word_bus: bool,
) -> Result<(), Exit> {
    let addr = (u64::from(vector) * 4) | bus(word_bus);
    let ip = read::<ROUTED, ORDER>(cpu, mem, addr, 2)?;
    let cs = read::<ROUTED, ORDER>(cpu, mem, addr + 2, 2)?;
    let sp = cpu.gpr[RSP].wrapping_sub(6) & 0xffff;
    let flags = cpu.rflags() | u64::from(flags_set);
    let frame = u64::from(saved_ip) | (u64::from(cpu.cs) << 16) | ((flags & 0xffff) << 32);
    write::<ROUTED, ORDER>(cpu, mem, bus_tag(cpu.ss, sp) | bus(word_bus), 6, frame)?;
    cpu.write_gpr(RSP, sp, 2);
    cpu.flags.if_ = false;
    cpu.rflags_system &= !0x100;
    cpu.cs = cs as u16;
    cpu.rip = ip;
    Ok(())
}

#[inline(never)]
pub(super) fn string_load<M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    rip: u64,
    slot: u64,
    tag: u64,
    elem: u8,
) -> Result<u64, MemTrap> {
    let mut value = 0;
    for i in 0..elem {
        let key = 32 + slot * 8 + u64::from(i);
        let byte = if let Some(v) = cpu.mmio_part(rip, key) {
            v
        } else {
            match mem.sload(physical(tag, i), 1) {
                Ok(v) => v,
                Err(MemTrap::Mmio) if tag & WORD_BUS != 0 => {
                    return string_load_rest(cpu, mem, (rip, slot), tag, elem, i, value)
                }
                Err(MemTrap::Mmio) if cpu.pending_mmio.is_some() => {
                    let v = cpu.pending_mmio.take().unwrap() & 255;
                    cpu.record_mmio_part(rip, key, v);
                    v
                }
                Err(trap) => {
                    record_fault_byte(cpu, rip, i);
                    return Err(trap);
                }
            }
        };
        value |= (byte & 255) << (8 * i);
    }
    Ok(value)
}
#[inline(never)]
pub(super) fn string_store<M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    rip: u64,
    tag: u64,
    value: u64,
    elem: u8,
) -> Result<(), MemTrap> {
    for i in 0..elem {
        if let Err((_, trap)) = mem.preflight_store(physical(tag, i), 1, 1, 1) {
            record_fault_byte(cpu, rip, i);
            return Err(trap);
        }
    }
    for i in 0..elem {
        let key = 64 + u64::from(i);
        if cpu.mmio_part(rip, key).is_some() {
            continue;
        }
        match mem.sstore(physical(tag, i), (value >> (8 * i)) & 255, 1) {
            Ok(()) => {}
            Err(MemTrap::Mmio) if tag & WORD_BUS != 0 => {
                return string_store_rest(cpu, mem, rip, tag, value, elem, i)
            }
            Err(MemTrap::Mmio) if cpu.pending_mmio_write => {
                cpu.pending_mmio_write = false;
                cpu.record_mmio_part(rip, key, (value >> (8 * i)) & 255);
            }
            Err(trap) => {
                record_fault_byte(cpu, rip, i);
                return Err(trap);
            }
        }
    }
    Ok(())
}
// A string element's device word (`device_word`), for the string slots: the
// element is at most a word, so its word is its whole element, keyed in the
// slot after its two byte keys. `None` is a byte of the eight-bit bus. The
// slot provider has no region map; a two-byte probe answers the Trap verdict,
// which only a span inside one Trap region gives (spec.md §5.2).
#[cold]
#[inline(never)]
fn string_word<M: StrMem>(mem: &M, tag: u64, i: u8, elem: u8) -> Option<(u64, u8)> {
    let addr = physical(tag, 0);
    if elem != 2 || addr & 1 != 0 || !matches!(mem.sload(addr, 2), Err(MemTrap::Mmio)) {
        return None;
    }
    Some((addr, i))
}
fn string_device_load<M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    rip: u64,
    slot: u64,
    tag: u64,
    i: u8,
    elem: u8,
) -> Result<u64, MemTrap> {
    let Some((_, byte)) = string_word(mem, tag, i, elem) else {
        let Some(value) = cpu.pending_mmio.take() else {
            record_fault_byte(cpu, rip, i);
            return Err(MemTrap::Mmio);
        };
        let value = value & 255;
        cpu.record_mmio_part(rip, 32 + slot * 8 + u64::from(i), value);
        return Ok(value);
    };
    let key = 32 + slot * 8 + 2;
    let value = if let Some(value) = cpu.mmio_part(rip, key) {
        value
    } else if let Some(value) = cpu.pending_mmio.filter(|_| byte == 0) {
        cpu.pending_mmio = None;
        let value = value & 0xffff;
        cpu.record_mmio_part(rip, key, value);
        value
    } else {
        record_fault_marker(cpu, rip, FAULT_WORD);
        return Err(MemTrap::Mmio);
    };
    Ok((value >> (8 * byte)) & 255)
}
fn string_device_store<M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    rip: u64,
    tag: u64,
    value: u64,
    i: u8,
    elem: u8,
) -> Result<(), MemTrap> {
    let key = match string_word(mem, tag, i, elem) {
        Some((_, 1)) => return Ok(()),
        Some(_) => 66,
        None => 64 + u64::from(i),
    };
    if cpu.mmio_part(rip, key).is_some() {
        return Ok(());
    }
    if !cpu.pending_mmio_write {
        record_fault_marker(cpu, rip, if key == 66 { FAULT_WORD } else { u64::from(i) });
        return Err(MemTrap::Mmio);
    }
    cpu.pending_mmio_write = false;
    let data = if key == 66 {
        value & 0xffff
    } else {
        (value >> (8 * i)) & 255
    };
    cpu.record_mmio_part(rip, key, data);
    Ok(())
}

// The rest of a string element, from byte `from`, once byte `from` met a Trap
// region on a sixteen-bit bus (`value` holds the bytes before it). The
// element's hot loops return into these, so they keep the shape they have
// without a device path.
#[cold]
#[inline(never)]
fn string_load_rest<M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    (rip, slot): (u64, u64),
    tag: u64,
    elem: u8,
    from: u8,
    mut value: u64,
) -> Result<u64, MemTrap> {
    for i in from..elem {
        let key = 32 + slot * 8 + u64::from(i);
        let byte = if let Some(v) = cpu.mmio_part(rip, key) {
            v
        } else {
            match mem.sload(physical(tag, i), 1) {
                Ok(v) => v,
                Err(MemTrap::Mmio) => string_device_load(cpu, mem, rip, slot, tag, i, elem)?,
                Err(trap) => {
                    record_fault_byte(cpu, rip, i);
                    return Err(trap);
                }
            }
        };
        value |= (byte & 255) << (8 * i);
    }
    Ok(value)
}
#[cold]
#[inline(never)]
fn string_store_rest<M: StrMem>(
    cpu: &mut CpuState,
    mem: &M,
    rip: u64,
    tag: u64,
    value: u64,
    elem: u8,
    from: u8,
) -> Result<(), MemTrap> {
    for i in from..elem {
        if cpu.mmio_part(rip, 64 + u64::from(i)).is_some() {
            continue;
        }
        match mem.sstore(physical(tag, i), (value >> (8 * i)) & 255, 1) {
            Ok(()) => {}
            Err(MemTrap::Mmio) => string_device_store(cpu, mem, rip, tag, value, i, elem)?,
            Err(trap) => {
                record_fault_byte(cpu, rip, i);
                return Err(trap);
            }
        }
    }
    Ok(())
}

// NEC V20/V30 OUTM source fault: name the wrapped byte, or word, the bus refused.
pub(super) fn port_string_fault(cpu: &mut CpuState, trap: MemTrap, tag: u64) -> Exit {
    let marker = cpu.mmio_part(cpu.rip, WRAPPED_FAULT_BYTE).unwrap_or(0);
    if trap != MemTrap::Mmio && marker & FAULT_OPENED_RETRY != 0 {
        cpu.clear_mmio_parts();
    }
    let size = if marker & FAULT_WORD != 0 { 2 } else { 1 };
    transfer_exit(cpu, trap, physical(tag, marker as u8), size, false, 0)
}

/// Complete a NEC V20/V30 INM element with the supplied port value
/// (NEC UM p. 12-28): DS1:IY, whose bytes wrap inside the segment. The store
/// completes outside the walk, so it reads the bus width from the vCPU's
/// catalogue entry (`model`).
pub(crate) fn finish_port_input<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Memory,
    model: Option<crate::CpuModel>,
) -> Result<(), Exit> {
    let word_bus = model.is_some_and(|model| crate::lift::i8086::word_bus(model.profile()));
    let Some(p) = cpu.port_string_in else {
        return Ok(());
    };
    let Some(value) = p.value else {
        return Err(Exit::port_io(
            cpu.gpr[RDX] as u16,
            p.elem,
            crate::PortDir::In,
            0,
        ));
    };
    write::<ROUTED, ORDER>(
        cpu,
        mem,
        p.base.wrapping_add(p.offset) | bus(word_bus),
        p.elem,
        value,
    )?;
    cpu.write_gpr(RDI, p.next_index, 2);
    if p.rep {
        cpu.write_gpr(RCX, p.next_count, 2);
    }
    cpu.rip = p.next_ip & 0xffff;
    cpu.port_string_in = None;
    cpu.clear_mmio_parts();
    Ok(())
}

// This is current-fault state, not an acknowledged transfer. Replace it when
// the retry advances to another byte; record_mmio_part deliberately deduplicates.
// FAULT_OPENED_RETRY marks a retry opened only to carry this marker.
fn record_fault_byte(cpu: &mut CpuState, rip: u64, byte: u8) {
    record_fault_marker(cpu, rip, u64::from(byte))
}
// `fault` is the refused byte, or FAULT_WORD for the element's device word.
fn record_fault_marker(cpu: &mut CpuState, rip: u64, fault: u64) {
    let opened = if cpu.mmio_retry_active && cpu.owns_mmio_retry(rip) {
        0
    } else {
        FAULT_OPENED_RETRY
    };
    cpu.begin_mmio_retry(rip);
    let marker = fault | opened;
    if let Some((_, value)) = cpu.mmio_parts[..usize::from(cpu.mmio_parts_len)]
        .iter_mut()
        .find(|(key, _)| *key == WRAPPED_FAULT_BYTE)
    {
        *value = marker;
    } else {
        cpu.record_mmio_part(rip, WRAPPED_FAULT_BYTE, marker);
    }
}
const WRAPPED_FAULT_BYTE: u64 = 80;
const FAULT_OPENED_RETRY: u64 = 1 << 8;
// The refused transfer is the element's device word, at its byte 0.
const FAULT_WORD: u64 = 1 << 9;

pub(super) fn string_evidence(cpu: &CpuState, source: SegmentId) -> Box<[(u64, u64)]> {
    if !cpu.owns_mmio_retry(cpu.rip) {
        return Box::new([]);
    }
    let src = bus_tag(legacy::selector(cpu, source), cpu.gpr[RSI]);
    let dst = bus_tag(cpu.es, cpu.gpr[RDI]);
    cpu.mmio_parts[..usize::from(cpu.mmio_parts_len)]
        .iter()
        .filter_map(|&(key, value)| {
            // Key 2 of a slot is its element's device word, at byte 0.
            let (tag, byte) = match key {
                32..=34 => (src, key - 32),
                40..=42 => (dst, key - 40),
                64..=66 => (dst, key - 64),
                _ => return None,
            };
            Some((physical(tag, (byte & 1) as u8), value))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(super) fn string<const ROUTED: bool, const STOP: bool, M: InterpreterMemory>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, 0, M>,
    kind: StrOp,
    elem: u8,
    rep: RepKind,
    (carry, word_bus): (bool, bool),
    source: SegmentId,
    ip: u64,
    exit_request: Option<&std::sync::atomic::AtomicU64>,
    steps: &mut StepContext,
    shadows: &mut ShadowState,
    retired: u64,
) -> Option<WalkResult> {
    shadows.complete_before(retired);
    let src = bus_tag(legacy::selector(cpu, source), 0) | bus(word_bus);
    let dst = bus_tag(cpu.es, 0) | bus(word_bus);
    // NEC UM p. 12-14: REPC/REPNC end on CY as REPE/REPNE end on Z.
    let result = string_run_specialized::<true, false, true, true, STOP, _>(
        cpu,
        mem,
        kind,
        elem,
        rep,
        carry,
        ip,
        16,
        src,
        dst,
        exit_request,
        &mut *steps,
        &mut *shadows,
    );
    match result {
        None => None,
        Some(StringStop::Fault(mut f)) => {
            let marker = cpu.mmio_part(ip, WRAPPED_FAULT_BYTE).unwrap_or(0);
            let byte = marker as u8;
            // A RAM-side fault answers no device transfer. Like the scalar path
            // and the ordinary string walk, it leaves no retry of its own, so a
            // queued IRQ is not held behind the restarted string.
            if f.trap != MemTrap::Mmio && marker & FAULT_OPENED_RETRY != 0 {
                cpu.clear_mmio_parts();
            }
            f.addr = physical(f.addr, byte);
            if marker & FAULT_WORD != 0 {
                f.value &= 0xffff;
                f.elem = 2;
            } else {
                f.value = (f.value >> (8 * byte)) & 255;
                f.elem = 1;
            }
            Some(WalkResult::exit(str_fault_exit(f)))
        }
        Some(StringStop::Interrupted) => {
            let result = steps.boundary(cpu, shadows, exit_request);
            if matches!(result, Some(WalkResult::Exit(InterpExit::Interrupted))) {
                steps.string_interrupted = true;
            }
            // Every stop keeps the complete instruction as the owner. Only an
            // interrupt that is taken saves the last prefix (UM §2.7 p. 2-42):
            // the dispatcher records the owner/restart pair and delivery applies it.
            if STOP && result.is_none() {
                steps.cut = true;
                Some(WalkResult::Continue)
            } else {
                result
            }
        }
    }
}

// Pack up to eight acknowledged coprocessor bus bytes per ledger entry. Large
// environment images also retry within the ABI-frozen 64-entry CpuState ledger.
// Bits 20..27 are the valid-byte mask; the low 20 bits name physical memory.
const FP_DIRECTIONS: u64 = 3 << 62;
const FP_VALID: u64 = 255 << 20;
fn fp_parts(cpu: &CpuState) -> Vec<(u64, u64)> {
    let mut parts = Vec::new();
    for &(key, value) in &cpu.mmio_parts[..usize::from(cpu.mmio_parts_len)] {
        if key & FP_DIRECTIONS == 0 {
            parts.push((key, value));
        } else {
            for byte in 0..8 {
                if key & (1 << (20 + byte)) != 0 {
                    parts.push(((key & !FP_VALID) + byte, (value >> (byte * 8)) & 255));
                }
            }
        }
    }
    parts
}
fn save_fp_parts(cpu: &mut CpuState, ip: u64, parts: &[(u64, u64)]) {
    let mut packed: Vec<(u64, u64)> = Vec::new();
    for &(key, value) in parts {
        if key & FP_DIRECTIONS == 0 {
            packed.push((key, value));
        } else {
            let base = key & !7;
            let byte = key & 7;
            if let Some((mask, data)) = packed.iter_mut().find(|(k, _)| *k & !FP_VALID == base) {
                *mask |= 1 << (20 + byte);
                *data |= (value & 255) << (byte * 8);
            } else {
                packed.push((base | (1 << (20 + byte)), (value & 255) << (byte * 8)));
            }
        }
    }
    cpu.mmio_parts_len = 0;
    for (key, value) in packed {
        cpu.record_mmio_part(ip, key, value);
    }
}

pub(super) fn bus_evidence(cpu: &CpuState, coprocessor: bool) -> Box<[(u64, u64)]> {
    if !cpu.owns_mmio_retry(cpu.rip) {
        return Box::new([]);
    }
    let parts = if coprocessor {
        fp_parts(cpu)
    } else {
        cpu.mmio_parts[..usize::from(cpu.mmio_parts_len)].to_vec()
    };
    parts
        .into_iter()
        .map(|(key, value)| (key & 0xfffff, value))
        .collect()
}

// The compatibility FpMem trait returns bool. Retain the complete error here,
// and keep coprocessor acknowledgements distinct from the CPU ESC dummy read.
// The local ledger avoids borrowing CpuState while exec_x87 updates its FPU.
struct FpMemory<'a, const ROUTED: bool, const ORDER: u8> {
    memory: &'a Memory,
    fault: Cell<Option<(MemTrap, u64, u64, bool)>>,
    parts: RefCell<Vec<(u64, u64)>>,
    pending_read: Cell<Option<u64>>,
    pending_write: Cell<bool>,
}
impl<const ROUTED: bool, const ORDER: u8> crate::x87::FpMem for FpMemory<'_, ROUTED, ORDER> {
    fn load(&self, addr: u64, buf: &mut [u8]) -> bool {
        let memory = Access::<ROUTED, ORDER, _>::new(self.memory);
        for (i, byte) in buf.iter_mut().enumerate() {
            let at = physical(addr, i as u8);
            let key = (1 << 62) | at;
            let answered = self
                .parts
                .borrow()
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| *v);
            let value = if let Some(value) = answered {
                value
            } else {
                match memory.read(at, 1) {
                    Ok(value) => value,
                    Err(MemTrap::Mmio) if self.pending_read.get().is_some() => {
                        let value = self.pending_read.take().unwrap() & 255;
                        self.parts.borrow_mut().push((key, value));
                        value
                    }
                    Err(trap) => {
                        self.fault.set(Some((trap, at, 0, false)));
                        return false;
                    }
                }
            };
            *byte = value as u8;
        }
        true
    }
    fn store(&self, addr: u64, bytes: &[u8]) -> bool {
        for (i, byte) in bytes.iter().enumerate() {
            let at = physical(addr, i as u8);
            if let Err(trap) = self.memory.probe_write_mode::<ROUTED>(at, 1) {
                if trap != MemTrap::Mmio {
                    self.fault.set(Some((trap, at, u64::from(*byte), true)));
                    return false;
                }
            }
        }
        let memory = Access::<ROUTED, ORDER, _>::new(self.memory);
        for (i, byte) in bytes.iter().enumerate() {
            let at = physical(addr, i as u8);
            let key = (1 << 63) | at;
            if self.parts.borrow().iter().any(|(k, _)| *k == key) {
                continue;
            }
            if let Err(trap) = memory.write(at, u64::from(*byte), 1) {
                if trap == MemTrap::Mmio && self.pending_write.replace(false) {
                    self.parts.borrow_mut().push((key, u64::from(*byte)));
                } else {
                    self.fault.set(Some((trap, at, u64::from(*byte), true)));
                    return false;
                }
            }
        }
        true
    }
    fn fault_addr(&self, addr: u64) -> u64 {
        self.fault.get().map_or(addr, |(_, at, _, _)| at)
    }
    fn host_fault_recovery(&self) -> bool {
        ROUTED
    }
}
