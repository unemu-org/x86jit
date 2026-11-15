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
    /// NEC V20/V30 BRKEM imm8, which the iced tables do not describe.
    brkem: Option<u8>,
}

fn fetch(
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
    let mut prefixes = 0u32;
    let mut original = Vec::new();
    let mut suffix = Vec::new();
    loop {
        if prefixes.is_multiple_of(256) && word.load(Ordering::Relaxed) & EXIT_REQUESTED != 0 {
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
                0xf0 => return Err(refusal(u64::from(ip), &original)),
                0xf2 | 0xf3 => repeat = Some(b[0]),
                _ => segment = Some(b[0]),
            }
            continue;
        }
        suffix.push(b[0]);
        if profile.isa.unavailable_prefix(&suffix) {
            return Err(refusal(u64::from(ip), &original));
        }
        // NEC UM p. 12-169: BRKEM imm8 is 0F FF imm8. No prefix use is
        // described, so a prefixed form is a typed refusal.
        if suffix[0] == 0x0f {
            if suffix.len() < 3 {
                continue;
            }
            if prefixes != 0 {
                return Err(refusal(u64::from(ip), &original));
            }
            return Ok(Decoded {
                insn: Instruction::default(),
                len: 3,
                restart: ip,
                bytes: suffix.clone(),
                evidence: original,
                brkem: Some(suffix[2]),
            });
        }
        // iced's fifteen-byte representation is a decoding tool, not the
        // processor's length limit. Retain the last prefix in each class,
        // then decode the bounded opcode/operand suffix at its real end IP.
        let mut bytes = Vec::new();
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
            brkem: None,
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
        let mut ip = cpu.rip as u16;
        let mut tg = crate::ir::TempGen::new();
        let mut combined: Option<IrBlock> = None;
        for _ in 0..16 {
            let fetched = fetch(vm, cpu, ip, &source, &mut spans, word, profile);
            let Decoded {
                insn,
                len,
                restart,
                bytes,
                evidence,
                brkem,
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
            let lifted = match brkem {
                Some(vector) => Ok(crate::lift::i8086::brkem(ip, len, next, vector, profile)),
                None => crate::lift::i8086::instruction(
                    &insn, &bytes, &evidence, ip, len, next, restart, profile, &mut tg,
                ),
            };
            let part = match lifted {
                Ok(ir) => ir,
                Err(_) if combined.is_some() => break,
                Err(_) => return Err(refusal(cpu.rip, &evidence)),
            };
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
            let terminates = brkem.is_some()
                || insn.flow_control() != iced_x86::FlowControl::Next
                || matches!(
                    insn.mnemonic(),
                    iced_x86::Mnemonic::Hlt | iced_x86::Mnemonic::Popf
                )
                || part.ops.iter().any(|op| {
                    matches!(
                        op,
                        IrOp::PortString { .. }
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
