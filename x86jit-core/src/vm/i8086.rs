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
    bytes: Vec<u8>,
    evidence: Vec<u8>,
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

fn fetch<const NEC: bool>(
    vm: &Vm,
    cpu: &CpuState,
    ip: u16,
    source: &crate::memory::LiftSource<'_>,
    spans: &mut Vec<(u64, u32)>,
    word: &AtomicU64,
    profile: &crate::ModelProfile,
) -> Result<Decoded, Exit> {
    let mut segment = None;
    let mut repeat = None;
    let mut carry = false;
    let mut lock = false;
    let mut prefixes = 0u32;
    let mut original = Vec::new();
    let mut suffix = Vec::new();
    // A requested host stop is taken here only with no transfer of the
    // instruction open: the Machine models finish the open retry first
    // (spec.md §5.2, Accesses of several transfers), and a stop at the fetch of
    // the retry would drop its pending completion and ask the device again.
    let retrying = cpu.mmio_retry_active
        || cpu.pending_mmio.is_some()
        || cpu.pending_mmio_write
        || cpu.pending_port_in.is_some();
    loop {
        if prefixes.is_multiple_of(256)
            && !retrying
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
        source.watch(pa, 1);
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
        source.record(pa, &b);
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
        if profile.isa.unavailable_prefix(&suffix) {
            // NEC UM p. 12-14: REPNC (64h) and REPC (65h) are repeat prefixes.
            if NEC && matches!(suffix[..], [0x64 | 0x65]) {
                suffix.clear();
                prefixes = prefixes.wrapping_add(1);
                carry = true;
                continue;
            }
            return Err(refusal(u64::from(ip), &original));
        }
        // NEC V20/V30 only: the 8086 byte table refuses these opcodes.
        if NEC && matches!(suffix[0], 0x0f | 0x66 | 0x67) {
            match nec_form(&suffix, prefixes, segment, carry || repeat.is_some()) {
                NecFetch::More => continue,
                NecFetch::Refuse => return Err(refusal(u64::from(ip), &original)),
                NecFetch::Form(form, insn) => {
                    return Ok(Decoded {
                        insn,
                        len: prefixes + suffix.len() as u32,
                        restart: ip,
                        bytes: suffix.clone(),
                        evidence: original,
                        nec: Some(form),
                        carry: false,
                        lock,
                    });
                }
            }
        }
        if NEC && carry {
            match carry_repeat(&original, prefixes) {
                Some(substitute) => repeat = Some(substitute),
                None => return Err(refusal(u64::from(ip), &original)),
            }
        }
        // iced's fifteen-byte representation is a decoding tool, not the
        // processor's length limit. Retain the last prefix in each class,
        // then decode the bounded opcode/operand suffix at its real end IP.
        let mut bytes = Vec::new();
        if lock {
            bytes.push(0xf0);
        }
        bytes.extend(segment);
        bytes.extend(repeat);
        bytes.extend_from_slice(&suffix);
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
            &bytes,
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
        match profile.isa.decode(&bytes, &insn) {
            DecodeAction::Defined => {}
            DecodeAction::Alias(code) => insn.set_code(code),
            DecodeAction::Absent(_) => return Err(refusal(u64::from(ip), &original)),
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
            return Err(refusal(u64::from(ip), &original));
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
            return Err(refusal(u64::from(ip), &original));
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
            bytes,
            evidence: original,
            nec: None,
            carry,
            lock,
        });
    }
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
    loop {
        let source = crate::memory::LiftSource::new(&vm.mem);
        let generation = vm.cache.publication_generation();
        let permit = vm.mem.fetch_permit();
        let mut spans = Vec::new();
        let profile = vm
            .cpu_model
            .expect("twenty-bit dispatch requires an 8086-family catalogue model")
            .profile();
        let nec_model = profile.isa == crate::ModelIsa::NecV20;
        let mut ip = cpu.rip as u16;
        let mut tg = crate::ir::TempGen::new();
        let mut combined: Option<IrBlock> = None;
        for _ in 0..16 {
            let fetched = if nec_model {
                fetch::<true>(vm, cpu, ip, &source, &mut spans, word, profile)
            } else {
                fetch::<false>(vm, cpu, ip, &source, &mut spans, word, profile)
            };
            let Decoded {
                insn,
                len,
                restart,
                bytes,
                evidence,
                nec,
                carry,
                lock,
            } = match fetched {
                Ok(value) => value,
                Err(Exit::Interrupted) => return Err(Exit::Interrupted),
                Err(_) if combined.is_some() => break,
                Err(exit) => return Err(exit),
            };
            let next = if profile.quirks.wrap_sequential_ip {
                ip.wrapping_add(len as u16)
            } else {
                u16::try_from(u64::from(ip) + u64::from(len))
                    .map_err(|_| refusal(u64::from(ip), &bytes))?
            };
            let mut lifted = match &nec {
                Some(form) => crate::lift::i8086::nec_form(
                    form, &insn, &evidence, ip, len, next, profile, lock, &mut tg,
                ),
                None => crate::lift::i8086::instruction(
                    &insn, &bytes, &evidence, ip, len, next, restart, profile, &mut tg,
                ),
            };
            if lock {
                lifted = lifted.and_then(|ir| crate::lift::i8086::locked(ir, &insn));
            }
            let mut part = match lifted {
                Ok(ir) => ir,
                Err(_) if combined.is_some() => break,
                Err(_) => return Err(refusal(cpu.rip, &evidence)),
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
                ir.ops.truncate(ir.ops.len() - 2);
                ir.ops.extend(part.ops);
                ir.guest_len += part.guest_len;
                ir.icount += 1;
                ir.temp_count = tg.count();
            } else {
                combined = Some(part);
            }
            if terminates {
                break;
            }
            ip = next;
        }
        let ir = combined.expect("first instruction decoded");
        if !source.validate() || permit.as_ref().is_some_and(|p| !p.is_current()) {
            continue;
        }
        if !source.publishable() {
            let CachedBlock::Interpreted(ir) = source.uncached(ir) else {
                unreachable!()
            };
            return Ok((ir, BlockWalk::Shared));
        }
        let ir = Arc::new(ir);
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
