//! Extended state transfers (§14). SDM Vol 1 §§13.4–13.10.
use crate::{CpuState, FaultCause, Memory};
use std::cell::RefCell;

/// Operation in the XSAVE family (SDM Vol 2D §6.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum XStateKind {
    Save,
    Restore,
    SaveOpt,
    SaveCompact,
}

/// Components implemented by the register file (SDM Vol 1 §§13.5.1–13.5.5).
pub(crate) const MODELED: u64 = 0xe7;
// Standard offsets are engine policy, enumerated through CPUID.0DH.i.
pub(crate) const COMPONENTS: [(u32, u32, u32); 4] =
    [(2, 256, 576), (5, 64, 832), (6, 512, 896), (7, 1024, 1408)];

pub(crate) fn area_size(mask: u64) -> u32 {
    COMPONENTS.iter().fold(576, |end, &(bit, size, off)| {
        if mask & (1 << bit) != 0 {
            end.max(off + size)
        } else {
            end
        }
    })
}

/// A transfer failure retains its actual byte address and direction.
#[derive(Clone, Copy, Debug)]
pub enum XStateFault {
    Memory { addr: u64, write: bool },
    Exception { vector: u8, cause: FaultCause },
}
fn gp() -> XStateFault {
    XStateFault::Exception {
        vector: 13,
        cause: FaultCause::Unknown,
    }
}

pub(crate) fn xcr0(cpu: &CpuState) -> u64 {
    cpu.cpuid_overrides
        .xcr0
        .unwrap_or_else(|| cpu.features.xcr0())
}

/// Validate XGETBV before changing either output register (SDM Vol 2D §6.3 XGETBV).
pub fn xgetbv_fault(cpu: &CpuState) -> Option<XStateFault> {
    if cpu.cpuid(1, 0).ecx & (1 << 27) == 0 {
        return Some(XStateFault::Exception {
            vector: 6,
            cause: FaultCause::InvalidOpcode,
        });
    }
    let index = cpu.gpr[1] as u32;
    if index != 0 && !(index == 1 && cpu.cpuid(0xd, 1).eax & 4 != 0) {
        return Some(gp());
    }
    None
}

struct FxImage(RefCell<[u8; 512]>);
impl super::FpMem for FxImage {
    fn load(&self, _: u64, buf: &mut [u8]) -> bool {
        buf.copy_from_slice(&*self.0.borrow());
        true
    }
    fn store(&self, _: u64, buf: &[u8]) -> bool {
        self.0.borrow_mut().copy_from_slice(buf);
        true
    }
}

fn memory_fault(addr: u64, write: bool) -> XStateFault {
    crate::fault::record_fp_fault(
        addr,
        1,
        crate::FaultOrigin::GuestMap,
        crate::AddressPrecision::ExactByte,
    );
    XStateFault::Memory { addr, write }
}
fn address(base: u64, off: usize, fault: crate::ir::MemoryAccess) -> Result<u64, XStateFault> {
    let addr = base.checked_add(off as u64).ok_or_else(gp)?;
    // SDM Vol 1 §3.3.7.1: each accessed linear address must be canonical.
    if fault.long64() && !crate::memory::is_canonical(addr) {
        return Err(XStateFault::Exception {
            vector: if fault.stack() { 12 } else { 13 },
            cause: FaultCause::NonCanonical,
        });
    }
    Ok(addr)
}
fn load(
    mem: &Memory,
    base: u64,
    off: usize,
    buf: &mut [u8],
    fault: crate::ir::MemoryAccess,
) -> Result<(), XStateFault> {
    for (i, byte) in buf.iter_mut().enumerate() {
        let at = address(base, off + i, fault)?;
        mem.read_ram_checked(at, std::slice::from_mut(byte))
            .map_err(|addr| {
                // Preserve host-access evidence when recovery supplied it.
                if mem.host_fault_recovery() {
                    XStateFault::Memory { addr, write: false }
                } else {
                    memory_fault(addr, false)
                }
            })?;
    }
    Ok(())
}

/// Shared interpreter/JIT implementation. All reads and destination probes complete
/// before committing state or save bytes (§16). No init/modified optimization is
/// required: XINUSE=XMODIFIED=all ones is permitted by SDM Vol 1 §13.6.
pub fn exec_xstate(
    cpu: &mut CpuState,
    mem: &Memory,
    base: u64,
    kind: XStateKind,
    wide: bool,
    fault: crate::ir::MemoryAccess,
) -> Result<(), XStateFault> {
    if mem.host_fault_recovery() {
        exec_xstate_access(
            cpu,
            crate::memory::Access::<true>::new(mem),
            base,
            kind,
            wide,
            fault,
        )
    } else {
        exec_xstate_access(
            cpu,
            crate::memory::Access::<false>::new(mem),
            base,
            kind,
            wide,
            fault,
        )
    }
}

pub(crate) fn exec_xstate_access<const ROUTED: bool>(
    cpu: &mut CpuState,
    mem: &crate::memory::Access<ROUTED, 0, impl crate::memory::InterpreterMemory>,
    base: u64,
    kind: XStateKind,
    wide: bool,
    fault: crate::ir::MemoryAccess,
) -> Result<(), XStateFault> {
    let long = fault.long64();
    let ecx = cpu.cpuid(1, 0).ecx;
    let extra = cpu.cpuid(0xd, 1).eax;
    if ecx & (1 << 26) == 0
        || ecx & (1 << 27) == 0
        || (kind == XStateKind::SaveOpt && extra & 1 == 0)
        || (kind == XStateKind::SaveCompact && extra & 2 == 0)
    {
        return Err(XStateFault::Exception {
            vector: 6,
            cause: FaultCause::InvalidOpcode,
        });
    }
    // SDM Vol 1 §§13.7–13.10: OSXSAVE precedes the 64-byte alignment check.
    if base & 63 != 0 {
        return Err(XStateFault::Exception {
            vector: 13,
            cause: FaultCause::RequiredAlignment,
        });
    }
    let enabled = xcr0(cpu);
    let requested =
        enabled & ((cpu.gpr[0] as u32 as u64) | ((cpu.gpr[2] as u32 as u64) << 32)) & MODELED;
    let restore = kind == XStateKind::Restore;
    let n = if long { 16 } else { 8 };
    let compact_save = kind == XStateKind::SaveCompact;
    let mut header = [0u8; 64];
    if restore {
        load(mem, base, 512, &mut header, fault)?;
    } else if !compact_save {
        // SDM Vol 1 §§13.7, 13.9: XSAVE/XSAVEOPT read XSTATE_BV to
        // preserve unrequested bits. XSAVEC (§13.10) has no header read.
        load(mem, base, 512, &mut header[..8], fault)?;
    }
    let bv = u64::from_le_bytes(header[..8].try_into().unwrap());
    let comp = u64::from_le_bytes(header[8..16].try_into().unwrap());
    let compact = compact_save || (restore && comp >> 63 != 0);
    if restore {
        let invalid = if compact {
            extra & 2 == 0
                || (comp & !(1 << 63)) & !enabled != 0
                || bv & !comp != 0
                || header[16..].iter().any(|&v| v != 0)
        } else {
            bv & !enabled != 0 || header[8..24].iter().any(|&v| v != 0)
        };
        if invalid {
            return Err(gp());
        }
    }
    // Stage the selected state components, preserving bytes outside those components.
    let image = FxImage(RefCell::new([0; 512]));
    super::exec_fxstate_format(cpu, &image, 0, false, wide);
    let mut legacy = image.0.into_inner();
    let mut writes: Vec<(usize, Vec<u8>)> = Vec::new();
    let mut transfer = |off: usize, buf: &mut [u8]| -> Result<(), XStateFault> {
        if restore {
            load(mem, base, off, buf, fault)
        } else {
            writes.push((off, buf.to_vec()));
            Ok(())
        }
    };
    let active = if restore { requested & bv } else { requested };
    if active & 1 != 0 {
        transfer(0, &mut legacy[..5])?;
        transfer(6, &mut legacy[6..24])?;
        for i in 0..8 {
            transfer(32 + i * 16, &mut legacy[32 + i * 16..42 + i * 16])?;
        }
    }
    // SDM Vol 1 §§13.7, 13.8, 13.10: standard AVX requests transfer MXCSR too.
    let mxcsr = if compact {
        active & 2 != 0
    } else {
        requested & 6 != 0
    };
    if mxcsr {
        transfer(24, &mut legacy[24..28])?;
        if !restore {
            transfer(28, &mut legacy[28..32])?;
        }
    }
    if active & 2 != 0 {
        transfer(160, &mut legacy[160..160 + n * 16])?;
    }
    let mut next = cpu.clone();
    if restore {
        if mxcsr {
            let value = u32::from_le_bytes(legacy[24..28].try_into().unwrap());
            if value & !crate::state::MXCSR_MASK != 0 {
                return Err(XStateFault::Exception {
                    vector: 13,
                    cause: FaultCause::InvalidMxcsr,
                });
            }
            next.mxcsr = value;
        } else if compact && requested & 2 != 0 {
            next.mxcsr = 0x1f80;
        }
        if requested & 1 != 0 {
            if active & 1 != 0 {
                let mut decoded = cpu.clone();
                // The legacy decoder also checks MXCSR; supply the already-validated value.
                legacy[24..28].copy_from_slice(&next.mxcsr.to_le_bytes());
                super::exec_fxstate_format(
                    &mut decoded,
                    &FxImage(RefCell::new(legacy)),
                    0,
                    true,
                    wide,
                );
                next.fpu_cw = decoded.fpu_cw;
                next.fpu_sw = decoded.fpu_sw;
                next.fpu_top = decoded.fpu_top;
                next.fpu_empty = decoded.fpu_empty;
                next.fpr = decoded.fpr;
                next.fpu_fop = decoded.fpu_fop;
                next.fpu_fip = decoded.fpu_fip;
                next.fpu_fdp = decoded.fpu_fdp;
                next.fpu_fcs = decoded.fpu_fcs;
                next.fpu_fds = decoded.fpu_fds;
            } else {
                next.fpu_cw = 0x037f;
                next.fpu_sw = 0;
                next.fpu_top = 0;
                next.fpu_empty = 0xff;
                next.fpr = [[0; 10]; 8];
                next.fpu_fop = 0;
                next.fpu_fip = 0;
                next.fpu_fdp = 0;
                next.fpu_fcs = 0;
                next.fpu_fds = 0;
            }
        }
        if requested & 2 != 0 {
            for i in 0..n {
                next.xmm[i] = if active & 2 == 0 {
                    0
                } else {
                    u128::from_le_bytes(legacy[160 + i * 16..176 + i * 16].try_into().unwrap())
                };
            }
        }
    }
    let format = if restore { comp } else { requested };
    let mut compact_offset = 576usize;
    for (bit, size, _) in COMPONENTS {
        let desc = cpu.cpuid(0xd, bit);
        let off = if compact {
            if desc.ecx & 2 != 0 {
                compact_offset = compact_offset.div_ceil(64) * 64;
            }
            let off = compact_offset;
            if format & (1 << bit) != 0 {
                compact_offset += desc.eax as usize;
            }
            off
        } else {
            desc.ebx as usize
        };
        if requested & (1 << bit) == 0 {
            continue;
        }
        // Tables describe offsets, but the modeled architectural component has a fixed width.
        // Default layout comes from COMPONENTS; a custom table must enumerate that width.
        if desc.eax != size {
            return Err(gp());
        }
        let count = match bit {
            2 => n * 16,
            5 => 64,
            6 => n * 32,
            _ => {
                if long {
                    1024
                } else {
                    0
                }
            }
        };
        if count == 0 {
            continue;
        }
        let mut buf = vec![0; count];
        if !restore {
            match bit {
                2 => {
                    for i in 0..n {
                        buf[i * 16..i * 16 + 16].copy_from_slice(&cpu.ymm_hi[i].to_le_bytes());
                    }
                }
                5 => {
                    for i in 0..8 {
                        buf[i * 8..i * 8 + 8].copy_from_slice(&cpu.kmask[i].to_le_bytes());
                    }
                }
                6 => {
                    for i in 0..n {
                        for j in 0..2 {
                            buf[i * 32 + j * 16..i * 32 + j * 16 + 16]
                                .copy_from_slice(&cpu.zmm_hi[i][j].to_le_bytes());
                        }
                    }
                }
                _ => {
                    for i in 0..16 {
                        for (j, v) in cpu.vec_lanes(16 + i).iter().enumerate() {
                            buf[i * 64 + j * 16..i * 64 + j * 16 + 16]
                                .copy_from_slice(&v.to_le_bytes());
                        }
                    }
                }
            }
        }
        if active & (1 << bit) != 0 {
            transfer(off, &mut buf)?;
        }
        if restore {
            match bit {
                2 => {
                    for i in 0..n {
                        next.ymm_hi[i] =
                            u128::from_le_bytes(buf[i * 16..i * 16 + 16].try_into().unwrap());
                    }
                }
                5 => {
                    for i in 0..8 {
                        next.kmask[i] =
                            u64::from_le_bytes(buf[i * 8..i * 8 + 8].try_into().unwrap());
                    }
                }
                6 => {
                    for i in 0..n {
                        for j in 0..2 {
                            next.zmm_hi[i][j] = u128::from_le_bytes(
                                buf[i * 32 + j * 16..i * 32 + j * 16 + 16]
                                    .try_into()
                                    .unwrap(),
                            );
                        }
                    }
                }
                _ => {
                    if long {
                        for i in 0..16 {
                            let mut v = [0; 4];
                            for (j, lane) in v.iter_mut().enumerate() {
                                *lane = u128::from_le_bytes(
                                    buf[i * 64 + j * 16..i * 64 + j * 16 + 16]
                                        .try_into()
                                        .unwrap(),
                                );
                            }
                            next.set_vec(16 + i, v, 64);
                        }
                    }
                }
            }
        }
    }
    if restore {
        *cpu = next;
        return Ok(());
    }
    let saved_bv = if compact_save {
        requested
    } else {
        bv | requested
    };
    writes.push((512, saved_bv.to_le_bytes().to_vec()));
    if compact_save {
        writes.push((520, ((1u64 << 63) | requested).to_le_bytes().to_vec()));
    }
    // Validate exact guest destinations, including discontiguous components.
    // Only crossing saves need actual-access preflight. ROM representatives do
    // not touch backing memory, so each host page also needs a RAM representative.
    let page = crate::memory::host_page_size();
    let mut pages = Vec::new();
    for (off, buf) in &writes {
        for i in 0..buf.len() {
            let at = address(base, off + i, fault)?;
            mem.probe_write_mode::<false>(at, 1).map_err(|trap| {
                if matches!(trap, crate::MemTrap::HostFault) {
                    let addr = trap.fault_addr(at);
                    crate::fault::record_fp_fault(
                        addr,
                        1,
                        crate::FaultOrigin::HostAccess,
                        crate::AddressPrecision::ExactByte,
                    );
                    XStateFault::Memory { addr, write: true }
                } else {
                    memory_fault(at, true)
                }
            })?;
            if mem.host_fault_recovery() {
                let host = (mem.host_base() as usize)
                    .wrapping_add(at.wrapping_sub(mem.guest_base()) as usize);
                let host_page = host / page;
                let kind = mem.region_kind(at, 1).map_err(|_| memory_fault(at, true))?;
                if !pages.iter().any(|&(p, k, _)| p == host_page && k == kind) {
                    pages.push((host_page, kind, at));
                }
            }
        }
    }
    if pages.len() > 1 {
        for (_, _, at) in pages {
            mem.probe_write(at, 1).map_err(|trap| {
                let addr = trap.fault_addr(at);
                crate::fault::record_fp_fault(
                    addr,
                    1,
                    crate::FaultOrigin::HostAccess,
                    crate::AddressPrecision::ExactByte,
                );
                XStateFault::Memory { addr, write: true }
            })?;
        }
    }
    for (off, buf) in writes {
        for (i, byte) in buf.iter().enumerate() {
            let at = base + (off + i) as u64;
            mem.write_ram(at, std::slice::from_ref(byte))
                .map_err(|addr| XStateFault::Memory { addr, write: true })?;
        }
    }
    Ok(())
}
