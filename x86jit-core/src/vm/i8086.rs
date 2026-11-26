//! Permanent Direct fetch and shared caching (decision-23 §5).
use super::*;
use crate::interp::legacy::BlockWalk;
use crate::model::decode::DecodeAction;
use crate::IrOp;
use iced_x86::{Decoder, DecoderError, Instruction};

fn prefix(b: u8) -> bool {
    matches!(b, 0x26 | 0x2e | 0x36 | 0x3e | 0xf0 | 0xf2 | 0xf3)
}
fn refusal(ip: u64, bytes: &[u8]) -> Exit {
    let mut out = [0; 15];
    let len = bytes.len().min(15);
    out[..len].copy_from_slice(&bytes[..len]);
    Exit::unknown_instruction(ip, out, len as u8)
}

struct Decoded {
    insn: Instruction,
    len: u32,
    restart: u16,
    /// A NEC V20/V30 form the iced tables do not describe.
    nec: Option<crate::lift::i8086::NecForm>,
    /// NEC V20/V30 REPC or REPNC: the repeat condition tests CY.
    carry: bool,
    /// A LOCK (BUSLOCK) prefix: `lift::i8086::locked` decides the form.
    lock: bool,
}

enum NecFetch {
    More,
    Refuse,
    /// The form and its operand surrogate.
    Form(crate::lift::i8086::NecForm, Instruction),
}

// NEC V20/V30 forms at 0Fh and 66h/67h (decision-23 C4a, C4b). The byte
// table (`known_absent_nec`) has already refused the forms it rules out.
#[cold]
#[inline(never)]
fn nec_form(suffix: &[u8], prefixes: u32, segment: Option<u8>, repeat: bool) -> NecFetch {
    use crate::legacy::BcdStringOp as Bcd;
    use crate::lift::i8086::NecForm as F;
    // Operand-free forms carry a NOP surrogate: sequential flow, so only
    // `NecForm::terminates` ends a block after them.
    let mut nop = Instruction::default();
    nop.set_code(iced_x86::Code::Nopw);
    // The r/m operand of `suffix[at..]` as a surrogate `opcode` with the same
    // ModRM and displacement, and their length; `None` until all bytes arrived.
    let surrogate = |opcode: u8, at: usize| {
        let mut bytes = Vec::from_iter(segment);
        bytes.push(opcode);
        bytes.extend_from_slice(&suffix[at..]);
        let mut decoder = Decoder::new(16, &bytes, iced_x86::DecoderOptions::NO_INVALID_CHECK);
        let insn = decoder.decode();
        (decoder.last_error() != DecoderError::NoMoreBytes)
            .then(|| (insn, insn.len() - usize::from(segment.is_some()) - 1))
    };
    if suffix[0] != 0x0f {
        // FPO2 (UM pp. 12-174–175) takes the ESC address and prefix policy.
        let Some(&modrm) = suffix.get(1) else {
            return NecFetch::More;
        };
        return if repeat {
            NecFetch::Refuse
        } else if modrm & 0xc0 == 0xc0 {
            NecFetch::Form(F::Fpo2 { memory: false }, nop)
        } else {
            match surrogate(0x8b, 1) {
                Some((address, _)) => NecFetch::Form(F::Fpo2 { memory: true }, address),
                None => NecFetch::More,
            }
        };
    }
    let Some(&second) = suffix.get(1) else {
        return NecFetch::More;
    };
    match second {
        // NEC UM p. 12-169: BRKEM imm8 is 0F FF imm8. No prefix use is
        // described, so a prefixed form is a typed refusal.
        0xff => {
            if prefixes != 0 {
                NecFetch::Refuse
            } else if suffix.len() < 3 {
                NecFetch::More
            } else {
                NecFetch::Form(F::Brkem(suffix[2]), nop)
            }
        }
        // pp. 12-42–44: the source segment may be overridden; no repeat
        // prefix is described for these instructions.
        0x20 | 0x22 | 0x26 if !repeat => NecFetch::Form(
            F::BcdString {
                op: match second {
                    0x20 => Bcd::Add,
                    0x22 => Bcd::Sub,
                    _ => Bcd::Cmp,
                },
                source: match segment {
                    Some(0x26) => crate::legacy::SegmentId::Es,
                    Some(0x2e) => crate::legacy::SegmentId::Cs,
                    Some(0x36) => crate::legacy::SegmentId::Ss,
                    _ => crate::legacy::SegmentId::Ds,
                },
            },
            nop,
        ),
        // pp. 12-21–24: register operands and fixed DS1:IY or DS0:IX; no
        // prefix use is described.
        0x31 | 0x33 | 0x39 | 0x3b if prefixes == 0 => {
            let insert = second & 2 == 0;
            match (second & 8 != 0, suffix.get(2), suffix.get(3)) {
                (false, Some(&modrm), _) => NecFetch::Form(
                    F::BitField {
                        insert,
                        offset: modrm & 7,
                        length: (modrm >> 3) & 7,
                        length_register: true,
                    },
                    nop,
                ),
                (true, Some(&modrm), Some(&imm)) => NecFetch::Form(
                    F::BitField {
                        insert,
                        offset: modrm & 7,
                        length: imm,
                        length_register: false,
                    },
                    nop,
                ),
                _ => NecFetch::More,
            }
        }
        // pp. 12-81–98 and 12-45–46: an r/m operand, then imm3/imm4 for
        // 0F 18h–1Fh. A segment prefix applies to a memory operand.
        0x10..=0x1f | 0x28 | 0x2a if !repeat => {
            if suffix.len() < 3 {
                return NecFetch::More;
            }
            let word = matches!(second, 0x10..=0x1f) && second & 1 != 0;
            let Some((operand, len)) = surrogate(if word { 0x89 } else { 0x88 }, 2) else {
                return NecFetch::More;
            };
            let imm = if matches!(second, 0x18..=0x1f) {
                match suffix.get(2 + len) {
                    Some(&imm) => imm,
                    None => return NecFetch::More,
                }
            } else {
                0
            };
            NecFetch::Form(
                F::Operand {
                    opcode: second,
                    imm,
                },
                operand,
            )
        }
        _ => NecFetch::Refuse,
    }
}

// The lifted string of a REPC/REPNC instruction tests CY (NEC UM p. 12-14).
#[cold]
#[inline(never)]
fn carry_string(part: &mut IrBlock) {
    for op in &mut part.ops {
        if let IrOp::Legacy(crate::legacy::LegacyOp::WrappedString { mode, .. }) = op {
            mode.set_carry();
        }
    }
}

// NEC V20/V30 REPC (65h) and REPNC (64h), NEC UM p. 12-14: the iced decode
// sees REPE or REPNE and the string tests CY. Two different repeat prefixes
// are not described: refused. A prefix run longer than the 15 bytes of
// evidence the fetch keeps cannot be checked and is refused too, an engine
// limit (REP/REPE/REPNE accept any count).
#[cold]
#[inline(never)]
fn carry_repeat(original: &[u8], prefixes: u32) -> Option<u8> {
    let run = original.get(..prefixes as usize)?;
    let mut repeats = run
        .iter()
        .filter(|b| matches!(b, 0x64 | 0x65 | 0xf2 | 0xf3));
    let first = *repeats.next()?;
    if repeats.any(|b| *b != first) {
        return None;
    }
    match first {
        0x65 => Some(0xf3),
        0x64 => Some(0xf2),
        _ => None,
    }
}

/// Where an instruction fetch reports the bytes it reads: the shared cache's
/// source tracking, or the snapshot a retained block is checked against.
enum Sink<'s, 'm> {
    Shared(&'s crate::memory::LiftSource<'m>),
    Retained(&'s mut Vec<u8>),
}

/// Drop the last `len` fetched bytes, of an instruction the block leaves
/// out: a store into them does not change the block.
fn unfetch(spans: &mut Vec<(u64, u32)>, sink: &mut Sink<'_, '_>, len: u32) {
    if let Sink::Retained(snapshot) = sink {
        snapshot.truncate(snapshot.len() - len as usize);
    }
    let mut left = len;
    while left != 0 {
        let last = spans.last_mut().expect("the instruction's fetched bytes");
        let take = last.1.min(left);
        last.1 -= take;
        left -= take;
        if last.1 == 0 {
            spans.pop();
        }
    }
}

/// A fetch's byte strings, kept by the lift of one block so that its
/// instructions reuse them: the fetched bytes (the first fifteen, the
/// evidence), the bytes after the prefixes, and the decoded form.
struct Buffers {
    original: Vec<u8>,
    suffix: Vec<u8>,
    bytes: Vec<u8>,
}

impl Buffers {
    fn new() -> Self {
        // Fifteen bytes of evidence; longer prefix runs grow the buffers.
        Self {
            original: Vec::with_capacity(15),
            suffix: Vec::with_capacity(16),
            bytes: Vec::with_capacity(16),
        }
    }
}

/// Fetch the instruction at `ip`, leaving its bytes and evidence in
/// `buffers.bytes` and `buffers.original`.
#[allow(clippy::too_many_arguments)]
fn fetch<const NEC: bool>(
    vm: &Vm,
    cpu: &CpuState,
    ip: u16,
    buffers: &mut Buffers,
    sink: &mut Sink<'_, '_>,
    spans: &mut Vec<(u64, u32)>,
    word: &AtomicU64,
    profile: &crate::ModelProfile,
) -> Result<Decoded, Exit> {
    let mut segment = None;
    let mut repeat = None;
    let mut carry = false;
    let mut lock = false;
    let mut prefixes = 0u32;
    let Buffers {
        original,
        suffix,
        bytes,
    } = buffers;
    original.clear();
    suffix.clear();
    // A requested host stop is taken here only with no transfer of the
    // instruction open: the Machine models finish the open retry first
    // (spec.md §5.2, Accesses of several transfers), and a stop at the fetch of
    // the retry would drop its pending completion and ask the device again.
    // The exemption ends once the prefixes have gone round the whole 64 KiB
    // segment: the retry's own instruction cannot be that long, so its code is
    // gone (the host rewrote it during the stop), and this loop, which counts
    // prefixes without a limit, would otherwise never return.
    let retrying = cpu.mmio_retry_active
        || cpu.pending_mmio.is_some()
        || cpu.pending_mmio_write
        || cpu.pending_port_in.is_some();
    loop {
        if prefixes.is_multiple_of(256)
            && (!retrying || prefixes >= 0x1_0000)
            && word.load(Ordering::Relaxed) & EXIT_REQUESTED != 0
        {
            return Err(Exit::Interrupted);
        }
        let offset = ip.wrapping_add((prefixes + suffix.len() as u32) as u16);
        let pa = ((u64::from(cpu.cs) << 4) + u64::from(offset)) & 0xfffff;
        if let Some(policy) = &vm.mem.fetch_policy {
            let verdict = policy.approve(pa >> crate::memory::CODE_PAGE_BITS, &vm.mem);
            if verdict != crate::FetchVerdict::Allow {
                return Err(crate::fault::fetch_exit(
                    &vm.mem,
                    FetchAddr {
                        pa,
                        ip: u64::from(ip),
                    },
                    CpuMode::Real16,
                    pa,
                    Some(crate::lift::FetchFailure::Policy(verdict)),
                ));
            }
        }
        if let Sink::Shared(source) = sink {
            source.watch(pa, 1);
        }
        let mut b = [0];
        vm.mem.fetch_code(pa, &mut b).map_err(|t| {
            crate::fault::fetch_exit(
                &vm.mem,
                FetchAddr {
                    pa,
                    ip: u64::from(ip),
                },
                CpuMode::Real16,
                pa,
                Some(crate::lift::FetchFailure::Memory(t)),
            )
        })?;
        match sink {
            Sink::Shared(source) => source.record(pa, &b),
            Sink::Retained(snapshot) => snapshot.push(b[0]),
        }
        if let Some((at, len)) = spans
            .last_mut()
            .filter(|(at, len)| *at + u64::from(*len) == pa)
        {
            let _ = at;
            *len += 1;
        } else {
            spans.push((pa, 1));
        }
        if original.len() < 15 {
            original.push(b[0]);
        }
        if suffix.is_empty() && prefix(b[0]) {
            prefixes = prefixes.wrapping_add(1);
            match b[0] {
                // 8086 Family UM §2.5 pp. 2-17–18; NEC V20/V30 UM p. 12-173.
                0xf0 => lock = true,
                0xf2 | 0xf3 => repeat = Some(b[0]),
                _ => segment = Some(b[0]),
            }
            continue;
        }
        suffix.push(b[0]);
        if profile.isa.unavailable_prefix(suffix) {
            // NEC UM p. 12-14: REPNC (64h) and REPC (65h) are repeat prefixes.
            if NEC && matches!(suffix[..], [0x64 | 0x65]) {
                suffix.clear();
                prefixes = prefixes.wrapping_add(1);
                carry = true;
                continue;
            }
            return Err(refusal(u64::from(ip), original));
        }
        // NEC V20/V30 only: the 8086 byte table refuses these opcodes.
        if NEC && matches!(suffix[0], 0x0f | 0x66 | 0x67) {
            match nec_form(suffix, prefixes, segment, carry || repeat.is_some()) {
                NecFetch::More => continue,
                NecFetch::Refuse => return Err(refusal(u64::from(ip), original)),
                NecFetch::Form(form, insn) => {
                    bytes.clear();
                    bytes.extend_from_slice(suffix);
                    return Ok(Decoded {
                        insn,
                        len: prefixes + suffix.len() as u32,
                        restart: ip,
                        nec: Some(form),
                        carry: false,
                        lock,
                    });
                }
            }
        }
        if NEC && carry {
            match carry_repeat(original, prefixes) {
                Some(substitute) => repeat = Some(substitute),
                None => return Err(refusal(u64::from(ip), original)),
            }
        }
        // iced's fifteen-byte representation is a decoding tool, not the
        // processor's length limit. Retain the last prefix in each class,
        // then decode the bounded opcode/operand suffix at its real end IP.
        bytes.clear();
        if lock {
            bytes.push(0xf0);
        }
        bytes.extend(segment);
        bytes.extend(repeat);
        bytes.extend_from_slice(suffix);
        let removed = prefixes - (bytes.len() - suffix.len()) as u32;
        // ESC's CPU address exists even when no x87 instruction uses the
        // extension (UM §4.2 Table 4-12 p. 4-27). Decode its address with a
        // MOV surrogate so an x87-invalid extension cannot omit displacement
        // bytes from the CPU instruction or its dummy word read.
        if matches!(suffix[0], 0xd8..=0xdf) {
            let mut address = bytes.clone();
            let opcode = address.len() - suffix.len();
            address[opcode] = 0x8b;
            if let Some(modrm) = address.get_mut(opcode + 1) {
                *modrm &= 0xc7;
            }
            let mut decoder = Decoder::new(16, &address, profile.isa.decoder_options());
            let _ = decoder.decode();
            if decoder.last_error() == DecoderError::NoMoreBytes {
                continue;
            }
        }
        let mut decoder = Decoder::with_ip(
            16,
            bytes,
            u64::from(ip).wrapping_add(u64::from(removed)),
            profile.isa.decoder_options(),
        );
        let mut insn = decoder.decode();
        if insn.is_invalid()
            && decoder.last_error() == DecoderError::NoMoreBytes
            && !matches!(suffix[0], 0xd8..=0xdf)
        {
            continue;
        }
        match profile.isa.decode(bytes, &insn) {
            DecodeAction::Defined => {}
            DecodeAction::Alias(code) => insn.set_code(code),
            DecodeAction::Absent(_) => return Err(refusal(u64::from(ip), original)),
        }
        // NEC UM p. 12-14: REPC and REPNC repeat a block comparison (CMPBK,
        // CMPM) only.
        if NEC
            && carry
            && !matches!(
                insn.mnemonic(),
                iced_x86::Mnemonic::Cmpsb
                    | iced_x86::Mnemonic::Cmpsw
                    | iced_x86::Mnemonic::Scasb
                    | iced_x86::Mnemonic::Scasw
            )
        {
            return Err(refusal(u64::from(ip), original));
        }
        // A port block transfer resumes at its first byte. With more prefixes
        // than the model keeps (NEC UM §6 p. 6-4: three) it would resume at a
        // later prefix, which this engine's port strings do not model.
        if matches!(
            insn.mnemonic(),
            iced_x86::Mnemonic::Insb
                | iced_x86::Mnemonic::Insw
                | iced_x86::Mnemonic::Outsb
                | iced_x86::Mnemonic::Outsw
        ) && prefixes > u32::from(profile.quirks.rep_restart_prefixes.unwrap_or(0))
        {
            return Err(refusal(u64::from(ip), original));
        }
        let len = prefixes + suffix.len() as u32;
        // 8086 Family UM §2.7 p. 2-42 keeps the last prefix; NEC V20/V30 UM
        // §6 p. 6-4 keeps up to three, one address back per prefix.
        let kept = u32::from(profile.quirks.rep_restart_prefixes.unwrap_or(0));
        let restart = ip.wrapping_add(prefixes.saturating_sub(kept) as u16);
        return Ok(Decoded {
            insn,
            len,
            restart,
            nec: None,
            carry,
            lock,
        });
    }
}

/// The block after one of its instructions: what lifting the rest of the
/// block again from the following instruction starts from.
pub(super) struct Boundary {
    /// Fetched bytes up to the end of this instruction.
    bytes: usize,
    /// The block's ops after this instruction, its terminator included.
    ops: usize,
    /// This instruction's sequential terminator and walk marker, once a
    /// following instruction has replaced them in the block.
    tail: Option<[IrOp; 2]>,
    guest_len: u32,
    icount: u32,
    temp_count: u32,
    temps: u32,
    next: u16,
}

/// A block lifted up to and including its `done`th instruction, its ops in
/// the lift's buffer.
struct Resume {
    ir: IrBlock,
    tg: crate::ir::TempGen,
    ip: u16,
    done: usize,
}

/// Lift the block at `cpu.rip`, or its remainder after `resume`, gathering
/// its ops in `ops` (left empty). Returns the block and whether it ended by
/// its own instructions rather than at a fetch that failed after its first
/// instruction.
#[allow(clippy::too_many_arguments)]
fn lift_block(
    vm: &Vm,
    cpu: &CpuState,
    word: &AtomicU64,
    profile: &'static crate::ModelProfile,
    sink: &mut Sink<'_, '_>,
    spans: &mut Vec<(u64, u32)>,
    ops: &mut Vec<IrOp>,
    resume: Option<Resume>,
    mut bounds: Option<&mut Vec<Boundary>>,
) -> Result<(IrBlock, bool), Exit> {
    let nec_model = profile.isa == crate::ModelIsa::NecV20;
    let (mut combined, mut tg, mut ip, done) = match resume {
        Some(resume) => (Some(resume.ir), resume.tg, resume.ip, resume.done),
        None => {
            ops.clear();
            (None, crate::ir::TempGen::new(), cpu.rip as u16, 0)
        }
    };
    let mut buffers = Buffers::new();
    let mut complete = true;
    for _ in done..16 {
        let fetched = if nec_model {
            fetch::<true>(vm, cpu, ip, &mut buffers, sink, spans, word, profile)
        } else {
            fetch::<false>(vm, cpu, ip, &mut buffers, sink, spans, word, profile)
        };
        let Decoded {
            insn,
            len,
            restart,
            nec,
            carry,
            lock,
        } = match fetched {
            Ok(value) => value,
            Err(Exit::Interrupted) => return Err(Exit::Interrupted),
            Err(_) if combined.is_some() => {
                complete = false;
                break;
            }
            Err(exit) => return Err(exit),
        };
        let (bytes, evidence) = (&buffers.bytes, &buffers.original);
        let next = if profile.quirks.wrap_sequential_ip {
            ip.wrapping_add(len as u16)
        } else {
            u16::try_from(u64::from(ip) + u64::from(len))
                .map_err(|_| refusal(u64::from(ip), bytes))?
        };
        let mut lifted = match &nec {
            Some(form) => crate::lift::i8086::nec_form(
                form, &insn, evidence, ip, len, next, profile, lock, &mut tg,
            ),
            None => crate::lift::i8086::instruction(
                &insn, bytes, evidence, ip, len, next, restart, profile, &mut tg,
            ),
        };
        if lock {
            lifted = lifted.and_then(|ir| crate::lift::i8086::locked(ir, &insn));
        }
        let mut part = match lifted {
            Ok(ir) => ir,
            Err(_) if combined.is_some() => {
                unfetch(spans, sink, len);
                break;
            }
            Err(_) => return Err(refusal(cpu.rip, evidence)),
        };
        if carry {
            carry_string(&mut part);
        }
        // REP interruption ownership starts at its own prefix sequence.
        // Keep strings at block entry so a yield before their first element
        // cannot acquire an in-progress repeat's restart IP.
        if combined.is_some()
            && part.ops.iter().any(|op| {
                matches!(
                    op,
                    IrOp::Legacy(crate::legacy::LegacyOp::WrappedString { .. })
                )
            })
        {
            unfetch(spans, sink, len);
            break;
        }
        // RunningMemory's active block range is contiguous: an instruction
        // whose bytes continue past the IP or the bus wrap starts its own
        // block, so that a store by an earlier instruction into those bytes
        // ends the walk before it (spec.md §10).
        let linear = ((u64::from(cpu.cs) << 4) + u64::from(ip)) & 0xfffff;
        if combined.is_some()
            && (u64::from(ip) + u64::from(len) > 0x10000 || linear + u64::from(len) > 0x100000)
        {
            unfetch(spans, sink, len);
            break;
        }
        let terminates = nec.is_some_and(|form| form.terminates())
            || insn.flow_control() != iced_x86::FlowControl::Next
            || matches!(
                insn.mnemonic(),
                iced_x86::Mnemonic::Hlt | iced_x86::Mnemonic::Popf
            )
            // `IrOp::PortIo` ends its block: the walk hands off with IP
            // at the block end, which must be the next instruction.
            || part.ops.iter().any(|op| {
                matches!(
                    op,
                    IrOp::PortIo { .. }
                        | IrOp::PortString { .. }
                        | IrOp::Legacy(
                            crate::legacy::LegacyOp::WrappedString { .. }
                                | crate::legacy::LegacyOp::Interrupt20 { .. }
                                | crate::legacy::LegacyOp::CheckIndex20 { .. }
                        )
                )
            })
            || u64::from(ip) + u64::from(len) >= 0x10000
            // RunningMemory's active block range is contiguous. End at
            // the bus wrap so stores into the following low addresses
            // cannot execute a stale instruction within the same walk.
            || (((u64::from(cpu.cs) << 4) + u64::from(ip)) & 0xfffff)
                + u64::from(len)
                >= 0x100000;
        if let Some(ir) = &mut combined {
            // Remove the previous sequential terminator and walk marker.
            let marker = ops.pop().unwrap();
            let terminator = ops.pop().unwrap();
            if let Some(bounds) = bounds.as_deref_mut() {
                bounds.last_mut().unwrap().tail = Some([terminator, marker]);
            }
            ops.append(&mut part.ops);
            ir.guest_len += part.guest_len;
            ir.icount += 1;
            ir.temp_count = tg.count();
        } else {
            ops.append(&mut part.ops);
            combined = Some(part);
        }
        if let Some(bounds) = bounds.as_deref_mut() {
            let ir = combined.as_ref().unwrap();
            bounds.push(Boundary {
                bytes: spans.iter().map(|(_, len)| *len as usize).sum(),
                ops: ops.len(),
                tail: None,
                guest_len: ir.guest_len,
                icount: ir.icount,
                temp_count: ir.temp_count,
                temps: tg.count(),
                next,
            });
        }
        if terminates {
            break;
        }
        ip = next;
    }
    let mut ir = combined.expect("first instruction decoded");
    // The dispatcher looks for a REP restart pair only in a one-instruction
    // block (`legacy::run_dispatch`).
    debug_assert!(
        ir.icount == 1
            || !ops.iter().any(|op| matches!(
                op,
                IrOp::Legacy(crate::legacy::LegacyOp::WrappedString { .. })
            )),
        "a twenty-bit string instruction in a block of {} instructions at {:#x}: \
         the REP-restart lookup reads one-instruction blocks only",
        ir.icount,
        ir.guest_start
    );
    // One allocation of the block's size; `ops` keeps its capacity for the
    // next lift.
    let mut block = Vec::with_capacity(ops.len());
    block.append(ops);
    ir.ops = block;
    Ok((ir, complete))
}

fn model_profile(vm: &Vm) -> &'static crate::ModelProfile {
    vm.cpu_model
        .expect("twenty-bit dispatch requires an 8086-family catalogue model")
        .profile()
}

pub(super) fn resolve(
    vm: &Vm,
    cpu: &CpuState,
    word: &AtomicU64,
    physical: u64,
) -> Result<(Arc<IrBlock>, BlockWalk), Exit> {
    let key = BlockKey::machine_direct(physical, cpu.mxcsr, false);
    if let Some(CachedBlock::Interpreted(ir)) = vm.cache.get_for_dispatch(key, vm.time_enabled) {
        if ir.guest_start == cpu.rip {
            return Ok((ir, BlockWalk::Shared));
        }
    }
    let mut ops = Vec::with_capacity(64);
    loop {
        let source = crate::memory::LiftSource::new(&vm.mem);
        let generation = vm.cache.publication_generation();
        let permit = vm.mem.fetch_permit();
        let mut spans = Vec::new();
        let profile = model_profile(vm);
        let (ir, _) = lift_block(
            vm,
            cpu,
            word,
            profile,
            &mut Sink::Shared(&source),
            &mut spans,
            &mut ops,
            None,
            None,
        )?;
        if !source.validate() || permit.as_ref().is_some_and(|p| !p.is_current()) {
            continue;
        }
        if !source.publishable() {
            let CachedBlock::Interpreted(ir) = source.uncached(ir) else {
                unreachable!()
            };
            return Ok((ir, BlockWalk::Shared));
        }
        let direct_exit = ir.ends_direct();
        let ir = Arc::new(IrBlock { direct_exit, ..ir });
        vm.handle_smc(word);
        if vm
            .cache
            .insert_checked(key, CachedBlock::Interpreted(ir.clone()), spans, |spans| {
                if vm.cache.publication_generation() != generation
                    || permit.as_ref().is_some_and(|p| !p.is_current())
                {
                    return false;
                }
                for (start, len) in spans {
                    vm.mem.mark_code(*start, *len);
                }
                true
            })
        {
            return Ok((ir, BlockWalk::Shared));
        }
    }
}

/// A block of a page that self-modifying code rewrites, kept by its vCPU
/// outside the shared cache with the bytes it was lifted from.
struct Retained {
    rip: u64,
    mxcsr: u32,
    profile: &'static crate::ModelProfile,
    ir: Arc<IrBlock>,
    spans: Vec<(u64, u32)>,
    bytes: Vec<u8>,
    bounds: Vec<Boundary>,
    /// The code-alias epoch at which no page of `spans` had an alias.
    alias_epoch: u64,
    /// Entries since the block was lifted, all finding its bytes unchanged.
    matched: u32,
    /// `RetainedBlocks::clock` at the block's last entry.
    used: u64,
}

/// One vCPU's retained twenty-bit blocks, by physical entry address, and
/// the ops buffer their lifts share.
#[derive(Default)]
pub(super) struct RetainedBlocks {
    blocks: std::collections::HashMap<
        u64,
        Retained,
        std::hash::BuildHasherDefault<crate::cache::cache_hash::BlockHasher>,
    >,
    ops: Vec<IrOp>,
    /// Counts entries; the least recently entered block is evicted.
    clock: u64,
    /// Lifts that kept the instructions before a changed byte.
    #[cfg(test)]
    resumed: usize,
}

impl RetainedBlocks {
    pub(super) fn clear(&mut self) {
        self.blocks.clear();
    }
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.blocks.len()
    }
}

// Bounds on what one vCPU retains: blocks, and bytes per block (a prefix run
// can make one instruction arbitrarily long).
const RETAINED_BLOCKS: usize = 128;
const RETAINED_BYTES: usize = 256;
// Entries of one block with unchanged bytes after which its page returns to
// the shared cache; a store that drops a block there marks it again.
pub(super) const RETAINED_MATCHES: u32 = 1 << 16;

/// The first fetched byte of `retained` that no longer reads as it did, or
/// that no longer reads at all.
fn changed(vm: &Vm, retained: &Retained) -> Option<usize> {
    let mut at = 0;
    let mut current = [0; 64];
    for &(pa, len) in &retained.spans {
        let mut offset = 0;
        while offset < len as usize {
            let n = (len as usize - offset).min(current.len());
            if vm.mem.fetch_code(pa + offset as u64, &mut current[..n]) != Ok(n) {
                return Some(at);
            }
            if let Some(i) = current[..n]
                .iter()
                .zip(&retained.bytes[at..at + n])
                .position(|(now, then)| now != then)
            {
                return Some(at + i);
            }
            at += n;
            offset += n;
        }
    }
    None
}

/// Return the page of `physical` to the shared cache: it leaves the
/// self-modifying set with every retained block that starts on it.
fn demote(blocks: &mut RetainedBlocks, smc_pages: &mut Vec<u64>, physical: u64) {
    let page = physical >> crate::memory::CODE_PAGE_BITS;
    smc_pages.retain(|p| *p != page);
    blocks
        .blocks
        .retain(|start, _| *start >> crate::memory::CODE_PAGE_BITS != page);
}

/// `resolve` for a page in the vCPU's self-modifying-code set (the 80286
/// path's `smc_pages`): the block is not published to the shared cache, so
/// stores to its page leave no invalidation work. Each entry compares the
/// bytes the block was lifted from with memory; a block whose bytes changed
/// keeps the instructions before the first changed byte and lifts the rest
/// again. Lifting is a function of the bytes, the entry IP and the model, so
/// the result equals a lift of the whole block from memory. Without a fetch
/// policy only: its page approvals are not rechecked here.
///
/// The block is held by this vCPU alone: no store invalidates it or raises
/// `DISPATCH_DIRTY`, its bytes are checked here instead. A dispatcher memo or
/// block link that skips resolve must not keep or reuse a block resolved
/// through this function (the dispatcher's `retained20`).
///
/// A page with a code alias stays with the shared cache: a store through the
/// other view reaches a running block only through the code tags that a
/// published block keeps on both views.
pub(super) fn resolve_retained(
    vm: &Vm,
    cpu: &CpuState,
    word: &AtomicU64,
    physical: u64,
    retained: &mut RetainedBlocks,
    smc_pages: &mut Vec<u64>,
) -> Result<(Arc<IrBlock>, BlockWalk), Exit> {
    let profile = model_profile(vm);
    let mxcsr = cpu.mxcsr & crate::interp::softfloat::MODE_MASK;
    // Read before any alias check, so a declaration after it is noticed.
    let alias_epoch = vm.mem.code_alias_epoch();
    let mut resume = None;
    let mut entry = None;
    if let Some(block) = retained.blocks.get_mut(&physical).filter(|block| {
        block.rip == cpu.rip && block.mxcsr == mxcsr && std::ptr::eq(block.profile, profile)
    }) {
        if block.alias_epoch != alias_epoch {
            if vm.mem.code_aliased(&block.spans) {
                demote(retained, smc_pages, physical);
                return resolve(vm, cpu, word, physical);
            }
            block.alias_epoch = alias_epoch;
        }
        let Some(at) = changed(vm, block) else {
            retained.clock += 1;
            block.used = retained.clock;
            block.matched += 1;
            let ir = block.ir.clone();
            let tired = block.matched >= RETAINED_MATCHES;
            if tired {
                demote(retained, smc_pages, physical);
            }
            return Ok((ir, BlockWalk::Shared));
        };
        let mut block = retained.blocks.remove(&physical).unwrap();
        let kept = block.bounds.iter().take_while(|b| b.bytes <= at).count();
        if kept != 0 {
            #[cfg(test)]
            {
                retained.resumed += 1;
            }
            let ops = &mut retained.ops;
            let whole = kept == block.bounds.len();
            let last = &mut block.bounds[kept - 1];
            ops.clear();
            if whole {
                ops.extend_from_slice(&block.ir.ops);
            } else {
                // The kept instruction's terminator, which the next one
                // replaced.
                let tail = last.tail.take().expect("a replaced terminator");
                ops.extend_from_slice(&block.ir.ops[..last.ops - 2]);
                ops.extend(tail);
            }
            resume = Some(Resume {
                ir: IrBlock {
                    mode: block.ir.mode,
                    guest_start: block.ir.guest_start,
                    guest_len: last.guest_len,
                    icount: last.icount,
                    temp_count: last.temp_count,
                    ops: Vec::new(),
                    legacy_context: block.ir.legacy_context,
                    // Retained blocks are never linked (`legacy/links.rs`).
                    direct_exit: false,
                },
                tg: crate::ir::TempGen::starting_at(last.temps),
                ip: last.next,
                done: kept,
            });
            let mut left = last.bytes;
            block.bytes.truncate(left);
            block.spans.retain_mut(|(_, len)| {
                let take = (*len as usize).min(left);
                *len = take as u32;
                left -= take;
                take != 0
            });
        } else {
            block.bytes.clear();
            block.spans.clear();
        }
        block.bounds.truncate(kept);
        entry = Some(block);
    }
    lift_retained(
        vm,
        cpu,
        word,
        physical,
        retained,
        smc_pages,
        (entry, resume),
        (profile, mxcsr, alias_epoch),
    )
}

/// `resolve_retained`'s lift: of the block's rest after `resume`, or of the
/// whole block without an `entry`.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn lift_retained(
    vm: &Vm,
    cpu: &CpuState,
    word: &AtomicU64,
    physical: u64,
    retained: &mut RetainedBlocks,
    smc_pages: &mut Vec<u64>,
    (entry, resume): (Option<Retained>, Option<Resume>),
    (profile, mxcsr, alias_epoch): (&'static crate::ModelProfile, u32, u64),
) -> Result<(Arc<IrBlock>, BlockWalk), Exit> {
    let page = physical >> crate::memory::CODE_PAGE_BITS;
    if entry.is_none() && vm.mem.code_aliased(&[(physical, 1)]) {
        // Checked before a lift: the shared path would lift it again.
        demote(retained, smc_pages, physical);
        return resolve(vm, cpu, word, physical);
    }
    // The previous block's vectors, reused.
    let (mut spans, mut snapshot, mut bounds) = match entry {
        Some(block) => (block.spans, block.bytes, block.bounds),
        None => Default::default(),
    };
    let (ir, complete) = lift_block(
        vm,
        cpu,
        word,
        profile,
        &mut Sink::Retained(&mut snapshot),
        &mut spans,
        &mut retained.ops,
        resume,
        Some(&mut bounds),
    )?;
    // The entry page was checked at this epoch, through the kept block's
    // spans or before the lift.
    let elsewhere = spans.iter().any(|&(pa, len)| {
        pa >> crate::memory::CODE_PAGE_BITS != page
            || (pa + u64::from(len) - 1) >> crate::memory::CODE_PAGE_BITS != page
    });
    if elsewhere && vm.mem.code_aliased(&spans) {
        demote(retained, smc_pages, physical);
        return resolve(vm, cpu, word, physical);
    }
    let ir = Arc::new(ir);
    retained.clock += 1;
    let retained_clock = retained.clock;
    let blocks = &mut retained.blocks;
    // A block cut short by a failed fetch could grow once that fetch
    // succeeds, which its bytes do not show: lift it again on each entry.
    if complete && snapshot.len() <= RETAINED_BYTES {
        if blocks.len() >= RETAINED_BLOCKS && !blocks.contains_key(&physical) {
            // Make room by the least recently entered block; an evicted
            // block is lifted again.
            let victim = *blocks.iter().min_by_key(|(_, block)| block.used).unwrap().0;
            blocks.remove(&victim);
        }
        blocks.insert(
            physical,
            Retained {
                rip: cpu.rip,
                mxcsr,
                profile,
                ir: ir.clone(),
                spans,
                bytes: snapshot,
                bounds,
                alias_epoch,
                matched: 0,
                used: retained_clock,
            },
        );
    } else {
        blocks.remove(&physical);
    }
    Ok((ir, BlockWalk::Shared))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Prot, RegionKind, VmConfig};

    const IP: u64 = 0x100;
    // MOV AL,1; ADD AL,2; ADD DL,3; MOV byte [2000h],5; ES: MOV AL,[3000h];
    // NOP; INC AX; LOCK INC byte [2001h]; JMP $.
    const CODE: &[u8] = b"\xb0\x01\x04\x02\x80\xc2\x03\xc6\x06\x00\x20\x05\x26\xa0\x00\x30\x90\x40\xf0\xfe\x06\x01\x20\xeb\xfe";

    fn vm(model: crate::CpuModel, code: &[u8]) -> (Vm, Vcpu) {
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        vm.set_cpu_model(model);
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(IP, code).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = IP;
        (vm, cpu)
    }

    fn shown(result: Result<(Arc<IrBlock>, BlockWalk), Exit>) -> String {
        match result {
            Ok((ir, _)) => format!("{ir:?}"),
            Err(exit) => format!("{exit:?}"),
        }
    }

    #[test]
    fn a_retained_relift_equals_a_lift_of_the_whole_block() {
        // Every byte of the block, rewritten with opcodes, prefixes and
        // operand values that keep, lengthen or shorten its instructions:
        // the retained relift (the kept prefix plus a lift of the rest)
        // equals the shared path's lift of the rewritten bytes.
        let word = AtomicU64::new(0);
        for model in [crate::CpuModel::I8086, crate::CpuModel::NecV30] {
            let mut resumed = 0;
            for at in 0..CODE.len() {
                for value in [
                    0x90,
                    0x04,
                    0x26,
                    0xf0,
                    0xf3,
                    0xeb,
                    0xf4,
                    0xcd,
                    0x0f,
                    CODE[at] ^ 1,
                ] {
                    let (vm, cpu) = self::vm(model, CODE);
                    let mut retained = RetainedBlocks::default();
                    let first =
                        resolve_retained(&vm, &cpu.cpu, &word, IP, &mut retained, &mut Vec::new());
                    assert_eq!(shown(first), shown(resolve(&vm, &cpu.cpu, &word, IP)));
                    vm.write_bytes(IP + at as u64, &[value]).unwrap();
                    let relift = shown(resolve_retained(
                        &vm,
                        &cpu.cpu,
                        &word,
                        IP,
                        &mut retained,
                        &mut Vec::new(),
                    ));
                    let mut code = CODE.to_vec();
                    code[at] = value;
                    let (fresh, cpu) = self::vm(model, &code);
                    assert_eq!(
                        relift,
                        shown(resolve(&fresh, &cpu.cpu, &word, IP)),
                        "{model:?} byte {at} = {value:#x}"
                    );
                    // An unchanged block is served as retained.
                    if relift.starts_with("IrBlock") {
                        let again = shown(resolve_retained(
                            &vm,
                            &cpu.cpu,
                            &word,
                            IP,
                            &mut retained,
                            &mut Vec::new(),
                        ));
                        assert_eq!(again, relift);
                    }
                    resumed += retained.resumed;
                }
            }
            assert!(resumed > 100, "{resumed}");
        }
    }

    #[test]
    fn retained_blocks_are_bounded_by_eviction_and_their_page_returns_when_unchanged() {
        let word = AtomicU64::new(0);
        // HLT at each of RETAINED_BLOCKS + 1 addresses: one more block evicts
        // one block, not the set.
        let (vm, mut cpu) = self::vm(crate::CpuModel::I8086, &[0xf4; 0x400]);
        let mut retained = RetainedBlocks::default();
        let mut pages = vec![IP >> crate::memory::CODE_PAGE_BITS];
        for i in 0..=RETAINED_BLOCKS as u64 {
            cpu.cpu.rip = IP + i;
            resolve_retained(&vm, &cpu.cpu, &word, IP + i, &mut retained, &mut pages).unwrap();
        }
        assert_eq!(retained.len(), RETAINED_BLOCKS);
        // A block whose bytes keep matching returns its page to the shared
        // cache, with the page's other blocks.
        cpu.cpu.rip = IP + RETAINED_BLOCKS as u64;
        for _ in 0..RETAINED_MATCHES {
            assert_eq!(pages.len(), 1);
            resolve_retained(&vm, &cpu.cpu, &word, cpu.cpu.rip, &mut retained, &mut pages).unwrap();
        }
        assert!(pages.is_empty());
        assert_eq!(retained.len(), 0);
    }

    #[test]
    fn a_code_alias_keeps_a_page_with_the_shared_cache() {
        let word = AtomicU64::new(0);
        let (vm, cpu) = self::vm(crate::CpuModel::I8086, CODE);
        let mut retained = RetainedBlocks::default();
        let mut pages = vec![IP >> crate::memory::CODE_PAGE_BITS];
        resolve_retained(&vm, &cpu.cpu, &word, IP, &mut retained, &mut pages).unwrap();
        assert_eq!(retained.len(), 1);
        // Declared after the block was retained: the next entry notices.
        vm.mem.declare_code_alias(0, 0x8000, 0x1000).unwrap();
        resolve_retained(&vm, &cpu.cpu, &word, IP, &mut retained, &mut pages).unwrap();
        assert!(pages.is_empty());
        assert_eq!(retained.len(), 0);
        assert!(vm.mem.is_code_page(0));
        // Marked again with no retained block: the entry page's alias is
        // found before a lift for retention, which leaves its buffer unused.
        let mut retained = RetainedBlocks::default();
        let mut pages = vec![IP >> crate::memory::CODE_PAGE_BITS];
        let (ir, _) =
            resolve_retained(&vm, &cpu.cpu, &word, IP, &mut retained, &mut pages).unwrap();
        assert_eq!(format!("{ir:?}"), shown(resolve(&vm, &cpu.cpu, &word, IP)));
        assert!(pages.is_empty());
        assert_eq!(retained.ops.capacity(), 0);
    }

    #[test]
    fn bytes_of_an_instruction_left_out_of_a_block_are_not_checked() {
        // NOP; REP STOSB (a string starts its own block), and NOP at FFFDh
        // before MOV AX,imm16 across the IP wrap (which starts its own
        // block): the next instruction's fetched bytes leave the snapshot,
        // so a store into them keeps the retained block.
        let word = AtomicU64::new(0);
        for (ip, code, store) in [
            (
                IP,
                &b"\x90\xf3\xaa"[..],
                &[(IP + 1, 0x90), (IP + 2, 0x90)][..],
            ),
            (
                0xfffd,
                &b"\x90\xb8\x01"[..],
                &[(0xfffe, 0x90), (0xffff, 0x02), (0, 0x03)][..],
            ),
        ] {
            let (vm, mut cpu) = self::vm(crate::CpuModel::I8086, &[]);
            vm.write_bytes(ip, code).unwrap();
            cpu.cpu.rip = ip;
            let mut retained = RetainedBlocks::default();
            let mut pages = vec![ip >> crate::memory::CODE_PAGE_BITS];
            let (first, _) =
                resolve_retained(&vm, &cpu.cpu, &word, ip, &mut retained, &mut pages).unwrap();
            assert_eq!(first.icount, 1);
            assert_eq!(retained.blocks[&ip].bytes, [0x90]);
            assert_eq!(retained.blocks[&ip].spans, [(ip, 1)]);
            for &(at, value) in store {
                vm.write_bytes(at, &[value]).unwrap();
                let (again, _) =
                    resolve_retained(&vm, &cpu.cpu, &word, ip, &mut retained, &mut pages).unwrap();
                assert!(Arc::ptr_eq(&first, &again), "{ip:#x}: store at {at:#x}");
            }
            assert_eq!(retained.resumed, 0);
        }
    }

    #[test]
    fn eviction_takes_the_least_recently_entered_block() {
        let word = AtomicU64::new(0);
        let (vm, mut cpu) = self::vm(crate::CpuModel::I8086, &[0xf4; 0x400]);
        let mut retained = RetainedBlocks::default();
        let mut pages = vec![IP >> crate::memory::CODE_PAGE_BITS];
        let mut enter = |cpu: &mut Vcpu, at: u64, retained: &mut RetainedBlocks| {
            cpu.cpu.rip = at;
            resolve_retained(&vm, &cpu.cpu, &word, at, retained, &mut pages).unwrap();
        };
        for i in 0..RETAINED_BLOCKS as u64 {
            enter(&mut cpu, IP + i, &mut retained);
        }
        // The first block, entered again, outlives the second; every new
        // block evicts the oldest entered one.
        for i in 0..4 {
            enter(&mut cpu, IP, &mut retained);
            enter(&mut cpu, IP + RETAINED_BLOCKS as u64 + i, &mut retained);
            assert!(retained.blocks.contains_key(&IP));
            assert!(!retained.blocks.contains_key(&(IP + 1 + i)));
            assert!(retained.blocks.contains_key(&(IP + 2 + i)));
        }
        assert_eq!(retained.len(), RETAINED_BLOCKS);
    }
}
