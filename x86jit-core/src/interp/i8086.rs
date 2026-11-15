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
fn memory_exit(cpu: &mut CpuState, trap: MemTrap, addr: u64, write: bool, value: u64) -> Exit {
    let StepResult::Exit(exit) = trap_out(
        cpu,
        cpu.rip,
        trap,
        addr,
        1,
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

#[inline(never)]
pub(super) fn primitive<const ROUTED: bool, const ORDER: u8>(
    op: &L,
    cpu: &mut CpuState,
    mem: &Memory,
    temps: &mut [u64],
    ip: u64,
) -> Result<(), Exit> {
    match op {
        L::WrappedLoad { dst, addr, size } => {
            temps[*dst as usize] = read::<ROUTED, ORDER>(cpu, mem, read_val(*addr, temps), *size)?
        }
        L::WrappedStore { addr, value, size } => write::<ROUTED, ORDER>(
            cpu,
            mem,
            read_val(*addr, temps),
            *size,
            read_val(*value, temps),
        )?,
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
        } => {
            // UM §2.7 p. 2-48: CPU ESC reads/discards its memory operand.
            // iAPX 86/88,186/188 PRM §6.3 p. 6-8 identifies this as a word.
            // EM is an engine input on a system-free model, never a fake #NM.
            if !wait && cpu.fpu_emulation {
                return Err(Exit::unknown_instruction(ip, *bytes, *len));
            }
            if *memory {
                read::<ROUTED, ORDER>(cpu, mem, read_val(*addr, temps), 2)?;
            }
            if cpu.fpu_model == crate::FpuModel::None {
            } else if let Some(kind) = kind {
                if crate::x87::mf_pending_before(cpu, *kind) {
                    return Err(Exit::unknown_instruction(ip, *bytes, *len));
                }
                cpu.begin_mmio_retry(ip);
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
                interrupt::<ROUTED, ORDER>(cpu, mem, 5, *saved_ip, *flags_set)?;
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
        } => {
            if latch {
                return Err(nec_refusal(cpu, ip, U::NecNestedEmulation));
            }
            if cpu.rflags_system & 0x100 != 0 {
                return Err(nec_refusal(cpu, ip, U::NecEmulationSingleStep));
            }
            interrupt::<ROUTED, ORDER>(cpu, mem, *vector, *saved_ip, *flags_set)?;
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
        _ => unreachable!("NEC control primitive"),
    }
}

/// The three words at SS:SP, each byte wrapping inside the segment
/// (NEC V20/V30 UM p. 12-177 RETEM; §10 Fig. 10-1).
pub(crate) fn read_frame<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Memory,
) -> Result<[u16; 3], Exit> {
    let tag = bus_tag(cpu.ss, cpu.gpr[RSP]);
    let mut words = [0; 3];
    for (i, word) in words.iter_mut().enumerate() {
        *word = read::<ROUTED, ORDER>(cpu, mem, tag + 2 * i as u64, 2)? as u16;
    }
    Ok(words)
}

// UM §2.6 pp. 2-25–28: fixed IVT, FLAGS/CS/IP frame, clear IF and TF.
#[inline(never)]
pub(crate) fn interrupt<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Memory,
    vector: u8,
    saved_ip: u16,
    flags_set: u16,
) -> Result<(), Exit> {
    let addr = u64::from(vector) * 4;
    let ip = read::<ROUTED, ORDER>(cpu, mem, addr, 2)?;
    let cs = read::<ROUTED, ORDER>(cpu, mem, addr + 2, 2)?;
    let sp = cpu.gpr[RSP].wrapping_sub(6) & 0xffff;
    let flags = cpu.rflags() | u64::from(flags_set);
    let frame = u64::from(saved_ip) | (u64::from(cpu.cs) << 16) | ((flags & 0xffff) << 32);
    write::<ROUTED, ORDER>(cpu, mem, bus_tag(cpu.ss, sp), 6, frame)?;
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
// NEC V20/V30 OUTM source fault: name the wrapped byte the bus refused.
pub(super) fn port_string_fault(cpu: &mut CpuState, trap: MemTrap, tag: u64) -> Exit {
    let marker = cpu.mmio_part(cpu.rip, WRAPPED_FAULT_BYTE).unwrap_or(0);
    if trap != MemTrap::Mmio && marker & FAULT_OPENED_RETRY != 0 {
        cpu.clear_mmio_parts();
    }
    memory_exit(cpu, trap, physical(tag, marker as u8), false, 0)
}

/// Complete a NEC V20/V30 INM element with the supplied port value
/// (NEC UM p. 12-28): DS1:IY, whose bytes wrap inside the segment.
pub(crate) fn finish_port_input<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Memory,
) -> Result<(), Exit> {
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
    write::<ROUTED, ORDER>(cpu, mem, p.base.wrapping_add(p.offset), p.elem, value)?;
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
    let opened = if cpu.mmio_retry_active && cpu.owns_mmio_retry(rip) {
        0
    } else {
        FAULT_OPENED_RETRY
    };
    cpu.begin_mmio_retry(rip);
    let marker = u64::from(byte) | opened;
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

pub(super) fn string_evidence(cpu: &CpuState, source: SegmentId) -> Box<[(u64, u64)]> {
    if !cpu.owns_mmio_retry(cpu.rip) {
        return Box::new([]);
    }
    let src = bus_tag(legacy::selector(cpu, source), cpu.gpr[RSI]);
    let dst = bus_tag(cpu.es, cpu.gpr[RDI]);
    cpu.mmio_parts[..usize::from(cpu.mmio_parts_len)]
        .iter()
        .filter_map(|&(key, value)| {
            let (tag, byte) = match key {
                32..=33 => (src, key - 32),
                40..=41 => (dst, key - 40),
                64..=65 => (dst, key - 64),
                _ => return None,
            };
            Some((physical(tag, byte as u8), value))
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
    source: SegmentId,
    ip: u64,
    exit_request: Option<&std::sync::atomic::AtomicU64>,
    steps: &mut StepContext,
    shadows: &mut ShadowState,
    retired: u64,
) -> Option<WalkResult> {
    shadows.complete_before(retired);
    let src = bus_tag(legacy::selector(cpu, source), 0);
    let dst = bus_tag(cpu.es, 0);
    let result = string_run_specialized::<true, false, true, true, STOP, _>(
        cpu,
        mem,
        kind,
        elem,
        rep,
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
            f.value = (f.value >> (8 * byte)) & 255;
            f.elem = 1;
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
