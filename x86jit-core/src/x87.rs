//! x87 FPU (§14), backed by the architectural 80-bit extended precision ([`F80`],
//! `f80.rs`) — each register holds the full sign + 15-bit exponent + 64-bit
//! significand and every op rounds to nearest-even at 64 significand bits, matching
//! real hardware. One `exec_x87` routine drives both backends (the interpreter calls
//! it directly; the JIT via a helper), so they agree bit-for-bit with each other and
//! with Unicorn — including the extra 11 mantissa bits that an `f64`-backed register
//! file would drop (e.g. `printf("%Lf")` long-double formatting).
//!
//! The register file is a stack: `ST(i)` = `fpr[(fpu_top + i) & 7]`. `fld`-style
//! ops decrement `fpu_top` then write `ST(0)`; `fstp`-style ops read `ST(0)` then
//! increment. Memory operands go through [`FpMem`], so the interpreter gets the
//! same region check + SMC `note_write` as a scalar store while the JIT keeps a raw
//! bounds-only view; a fault returns `Some((addr, is_write))` so the caller traps
//! with RIP on the instruction (§8, §16), exactly like the string helper.

use crate::f80::{Ctl, Exc, F80};
use crate::state::CpuState;

/// Guest-memory access for the x87 helpers. Two implementors give the two backends
/// the memory semantics each already uses for a scalar store:
///
/// * The interpreter passes `&Memory` — reads/writes go through a mapped-RAM region
///   check and, on a write, the SMC `note_write` (§10), so a self-modifying x87
///   store onto a code page invalidates just like `IrOp::Store`.
/// * The JIT passes [`RawFpMem`] — a bounds-only raw view matching its inlined
///   stores; JIT-side SMC is the deferred "mark host code dead" step (§10, §9.1).
///
/// A `Trap` (MMIO) region faults as unmapped here: an x87 store's value (up to a
/// 10-byte f80 / 512-byte fxsave) can't fit `Exit::MmioWrite`, so x87→MMIO is
/// deferred rather than misreported (§5.2).
pub trait FpMem {
    /// Fill `buf` from guest memory; `false` on a fault (unmapped / non-RAM).
    fn load(&self, addr: u64, buf: &mut [u8]) -> bool;
    /// Write `bytes` to guest memory (recording SMC); `false` on a fault.
    fn store(&self, addr: u64, bytes: &[u8]) -> bool;
    fn fault_addr(&self, addr: u64) -> u64 {
        addr
    }
    fn host_fault_recovery(&self) -> bool {
        false
    }
    /// Checked form retaining an exact host-protection fault address.
    fn load_checked(&self, addr: u64, buf: &mut [u8]) -> Result<(), u64> {
        self.load(addr, buf).then_some(()).ok_or(addr)
    }
    fn store_checked(&self, addr: u64, bytes: &[u8]) -> Result<(), u64> {
        self.store(addr, bytes).then_some(()).ok_or(addr)
    }
}

impl FpMem for crate::memory::Memory {
    fn fault_addr(&self, addr: u64) -> u64 {
        if self.host_fault_recovery() {
            crate::memory::MemTrap::HostFault.fault_addr(addr)
        } else {
            addr
        }
    }
    fn host_fault_recovery(&self) -> bool {
        self.host_fault_recovery()
    }

    fn load_checked(&self, addr: u64, buf: &mut [u8]) -> Result<(), u64> {
        self.read_ram_checked(addr, buf)
    }
    fn store_checked(&self, addr: u64, bytes: &[u8]) -> Result<(), u64> {
        self.write_ram_checked(addr, bytes)
    }

    fn load(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.read_ram_checked(addr, buf)
            .map_err(|at| {
                if self.host_fault_recovery() {
                    crate::memory::record_host_fault(at);
                }
            })
            .is_ok()
    }
    fn store(&self, addr: u64, bytes: &[u8]) -> bool {
        self.write_ram_checked(addr, bytes)
            .map_err(|at| {
                if self.host_fault_recovery() {
                    crate::memory::record_host_fault(at);
                }
            })
            .is_ok()
    }
}

impl<const ROUTED: bool> FpMem for crate::memory::Access<ROUTED> {
    #[inline(always)]
    fn load(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.0
            .read_ram_checked_mode::<ROUTED>(addr, buf)
            .map_err(|at| {
                if ROUTED {
                    crate::memory::record_host_fault(at);
                }
            })
            .is_ok()
    }
    #[inline(always)]
    fn store(&self, addr: u64, buf: &[u8]) -> bool {
        self.0
            .write_ram_checked_mode::<ROUTED>(addr, buf)
            .map_err(|at| {
                if ROUTED {
                    crate::memory::record_host_fault(at);
                }
            })
            .is_ok()
    }
    fn fault_addr(&self, addr: u64) -> u64 {
        if ROUTED {
            crate::memory::MemTrap::HostFault.fault_addr(addr)
        } else {
            addr
        }
    }
    fn host_fault_recovery(&self) -> bool {
        ROUTED
    }
}

/// Raw guest view for JIT x87/fxstate helpers: bounds plus optional host fixups
/// (deferred JIT SMC).
/// `base` is the host address of guest `guest_base`; `size` is the exclusive top guest
/// address. A guest address `a` reads/writes `base + (a - guest_base)`, valid iff
/// `guest_base <= a` and `a + len <= size` (see [`crate::interp::RawStrMem`]).
pub struct RawFpMem {
    pub base: *mut u8,
    pub size: u64,
    pub guest_base: u64,
    /// Enable fixups when host protections are independently owned by the embedder.
    pub recover_host_faults: bool,
}

impl RawFpMem {
    /// Backing offset for `addr` if `[addr, addr+len)` lies in `[guest_base, size)`.
    #[inline]
    fn off(&self, addr: u64, len: usize) -> Option<usize> {
        let end = addr.checked_add(len as u64)?;
        if addr < self.guest_base || end > self.size {
            return None;
        }
        Some((addr - self.guest_base) as usize)
    }
}

impl FpMem for RawFpMem {
    fn host_fault_recovery(&self) -> bool {
        self.recover_host_faults
    }
    fn fault_addr(&self, addr: u64) -> u64 {
        if self.recover_host_faults {
            crate::memory::MemTrap::HostFault.fault_addr(addr)
        } else {
            addr
        }
    }

    fn load(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.load_checked(addr, buf).is_ok()
    }
    fn store(&self, addr: u64, bytes: &[u8]) -> bool {
        self.store_checked(addr, bytes).is_ok()
    }
    fn load_checked(&self, addr: u64, buf: &mut [u8]) -> Result<(), u64> {
        if self.recover_host_faults {
            self.load_mode::<true>(addr, buf)
        } else {
            self.load_mode::<false>(addr, buf)
        }
    }
    fn store_checked(&self, addr: u64, buf: &[u8]) -> Result<(), u64> {
        if self.recover_host_faults {
            self.store_mode::<true>(addr, buf)
        } else {
            self.store_mode::<false>(addr, buf)
        }
    }
}
impl RawFpMem {
    #[inline(always)]
    fn load_mode<const ROUTED: bool>(&self, addr: u64, buf: &mut [u8]) -> Result<(), u64> {
        let off = self.off(addr, buf.len()).ok_or_else(|| {
            if ROUTED {
                crate::memory::record_host_fault(addr);
            }
            addr
        })?;
        // SAFETY: off checked the complete span; default mode requires host accessibility.
        if !ROUTED {
            unsafe {
                std::ptr::copy_nonoverlapping(self.base.add(off), buf.as_mut_ptr(), buf.len())
            };
            return Ok(());
        }
        unsafe { crate::usercopy::copy_in(self.base.add(off), buf) }.map_err(|at| {
            let addr = self.guest_base + (at - self.base as usize) as u64;
            crate::memory::record_host_fault(addr);
            addr
        })
    }
    #[inline(always)]
    fn store_mode<const ROUTED: bool>(&self, addr: u64, bytes: &[u8]) -> Result<(), u64> {
        let off = self.off(addr, bytes.len()).ok_or_else(|| {
            if ROUTED {
                crate::memory::record_host_fault(addr);
            }
            addr
        })?;
        // SAFETY: off checked the complete span; default mode requires host accessibility.
        if !ROUTED {
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.base.add(off), bytes.len())
            };
            return Ok(());
        }
        unsafe { crate::usercopy::copy_out(self.base.add(off), bytes) }.map_err(|at| {
            let addr = self.guest_base + (at - self.base as usize) as u64;
            crate::memory::record_host_fault(addr);
            addr
        })
    }
}

// Select once per helper call so direct FXRSTOR is a visible full memcpy.
#[repr(transparent)]
struct FpAccess<const ROUTED: bool>(RawFpMem);
impl<const ROUTED: bool> FpAccess<ROUTED> {
    fn new(raw: &RawFpMem) -> &Self {
        // SAFETY: transparent shared view, with the source borrow's lifetime.
        unsafe { &*(raw as *const RawFpMem).cast::<Self>() }
    }
}
impl<const ROUTED: bool> FpMem for FpAccess<ROUTED> {
    #[inline(always)]
    fn load(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.0.load_mode::<ROUTED>(addr, buf).is_ok()
    }
    #[inline(always)]
    fn store(&self, addr: u64, buf: &[u8]) -> bool {
        self.0.store_mode::<ROUTED>(addr, buf).is_ok()
    }
    fn host_fault_recovery(&self) -> bool {
        ROUTED
    }
    fn fault_addr(&self, addr: u64) -> u64 {
        if ROUTED {
            crate::MemTrap::HostFault.fault_addr(addr)
        } else {
            addr
        }
    }
}
impl RawFpMem {
    /// Run one FP state transfer, selecting its protection mode once on entry.
    pub fn exec_fxstate(
        &self,
        cpu: &mut CpuState,
        addr: u64,
        restore: bool,
    ) -> Option<FpStateFault> {
        if self.recover_host_faults {
            fxstate_routed(cpu, self, addr, restore)
        } else {
            exec_fxstate(cpu, FpAccess::<false>::new(self), addr, restore)
        }
    }
    /// Run one x87 operation, selecting its protection mode once on entry.
    pub fn exec_x87(
        &self,
        cpu: &mut CpuState,
        kind: FpuKind,
        addr: u64,
        sti: u8,
    ) -> Option<(u64, bool)> {
        if self.recover_host_faults {
            x87_routed(cpu, self, kind, addr, sti)
        } else {
            exec_x87(cpu, FpAccess::<false>::new(self), kind, addr, sti)
        }
    }
}

#[inline(never)]
fn x87_routed(
    cpu: &mut CpuState,
    raw: &RawFpMem,
    kind: FpuKind,
    addr: u64,
    sti: u8,
) -> Option<(u64, bool)> {
    exec_x87(cpu, FpAccess::<true>::new(raw), kind, addr, sti)
}

// Keep the routed copy buffer and saved registers out of the default helper frame.
#[inline(never)]
fn fxstate_routed(
    cpu: &mut CpuState,
    raw: &RawFpMem,
    addr: u64,
    restore: bool,
) -> Option<FpStateFault> {
    exec_fxstate(cpu, FpAccess::<true>::new(raw), addr, restore)
}

/// One x87 operation. Memory forms carry their access in `addr`/size via the op
/// variant; register/stack forms use the `sti` argument to `exec_x87`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum FpuKind {
    // memory load, push ST(0)
    FldF64,
    FldF32,
    FldF80,
    FildI16,
    FildI32,
    FildI64,
    // memory store from ST(0); the `P` forms pop
    FstpF64,
    FstpF32,
    FstpF80,
    FistpI16,
    FistpI32,
    FistpI64,
    // fisttp (SSE3): store integer truncating toward zero (ignores the FPU rounding
    // control), then pop — glibc number formatting uses it.
    FisttpI16,
    FisttpI32,
    FisttpI64,
    FstF64,
    FstF32,
    // ST(0) op= memory
    FaddMemF64,
    FaddMemF32,
    FsubMemF64,
    FsubMemF32,
    FsubrMemF64,
    FsubrMemF32,
    FmulMemF64,
    FmulMemF32,
    FdivMemF64,
    FdivMemF32,
    FdivrMemF64,
    FdivrMemF32,
    // ST(0) op= integer memory: `DA /n` = m32int, `DE /n` = m16int. Any i16/i32 converts
    // exactly to F80, so the only rounding is the arithmetic itself, taken on the same F80
    // path as the float-memory forms above. The `r` forms reverse the operands — `fisub`
    // is ST(0) - mem, `fisubr` is mem - ST(0), and likewise `fidiv`/`fidivr` (measured on
    // hardware, SDM Vol 2 FISUB/FIDIV).
    //
    // `ficom`/`ficomp` (`DA /2 /3`, `DE /2 /3`) report their result in the status-word
    // condition codes C0/C2/C3 rather than in EFLAGS; the `Fcomi`/`Fucomi` family never
    // needs them.
    FicomI16,
    FicomI32,
    FicompI16,
    FicompI32,
    FiaddMemI16,
    FiaddMemI32,
    FisubMemI16,
    FisubMemI32,
    FisubrMemI16,
    FisubrMemI32,
    FimulMemI16,
    FimulMemI32,
    FidivMemI16,
    FidivMemI32,
    FidivrMemI16,
    FidivrMemI32,
    // push a copy of ST(i)
    FldSti,
    // register/stack store: ST(i) = ST(0); the `p` form pops
    FstSti,
    FstpSti,
    // ST(0) op= ST(i) (register forms, ST(0) destination, no pop)
    FsubSti,  // ST(0) -= ST(i)
    FsubrSti, // ST(0) = ST(i) - ST(0)
    FdivSti,  // ST(0) /= ST(i)
    FdivrSti, // ST(0) = ST(i) / ST(0)
    // ST(i) op= ST(0) (register forms, ST(i) destination, no pop) — the `p` forms
    // above with the pop removed (e.g. `fmul st(1), st(0)`).
    FaddToSti,  // ST(i) += ST(0)
    FsubToSti,  // ST(i) -= ST(0)
    FsubrToSti, // ST(i) = ST(0) - ST(i)
    FmulToSti,  // ST(i) *= ST(0)
    FdivToSti,  // ST(i) /= ST(0)
    FdivrToSti, // ST(i) = ST(0) / ST(i)
    // push a constant
    Fld1,
    Fldz,
    // register/stack arithmetic (use `sti`)
    FaddP,   // ST(i) += ST(0); pop
    FsubP,   // ST(i) -= ST(0); pop
    FsubrP,  // ST(i) = ST(0) - ST(i); pop
    FmulP,   // ST(i) *= ST(0); pop
    FdivP,   // ST(i) /= ST(0); pop
    FdivrP,  // ST(i) = ST(0) / ST(i); pop
    FaddSti, // ST(0) += ST(i)
    FmulSti, // ST(0) *= ST(i)
    Fxch,    // swap ST(0), ST(i)
    // compare ST(0) with ST(i), set EFLAGS (ZF/PF/CF); the `P` forms pop
    Fucomi,
    Fucomip,
    Fcomi,
    Fcomip,
    Fcmovb,
    Fcmove,
    Fcmovbe,
    Fcmovu,
    Fcmovnb,
    Fcmovne,
    Fcmovnbe,
    Fcmovnu,
    Fcom,
    Fcomp,
    Fcompp,
    Fucom,
    Fucomp,
    Fucompp,
    FcomMemF32,
    FcomMemF64,
    FcompMemF32,
    FcompMemF64,
    Ftst,
    Fxam,
    Fsqrt,
    Fscale,
    Fprem1,
    // unary on ST(0)
    Fabs,
    Fchs,
    // control word / status word
    Fldcw,     // load control word from memory
    Fnstcw,    // store control word to memory
    Fnstsw,    // store status word to AX (register form)
    FnstswMem, // store status word to [mem] (memory form)
    /// `fnstenv m28byte`: store the 28-byte x87 environment, then mask every FP
    /// exception. Only the 28-byte (32-bit-layout) image is lifted — see
    /// [`env28`] for the field-by-field fidelity note.
    Fnstenv,
    /// `fldenv m28byte`: load the 28-byte x87 environment — the restore half of the
    /// pair FreeBSD's `<fenv.h>` wraps around `powf`/`expf`. Environment only: the
    /// eight data registers are untouched (that is `frstor`). See [`load_env28`] for
    /// the field-by-field decision.
    Fldenv,
    Fprem, // ST(0) = ST(0) rem ST(1)
    // Transcendentals. f64-precision (see `F80` transcendental methods);
    // validated to a bounded ULP vs libm/Unicorn. The reduction-domain ops (fsin/fcos/
    // fptan/fsincos) leave the operand unchanged when |ST(0)| >= 2^63 (hardware sets C2,
    // which is not modeled — the -i compares set EFLAGS, so guests rarely read C0-C3).
    Fsin,    // ST(0) = sin(ST(0))
    Fcos,    // ST(0) = cos(ST(0))
    Fptan,   // ST(0) = tan(ST(0)); then push 1.0
    Fpatan,  // ST(1) = atan2(ST(1), ST(0)); pop
    F2xm1,   // ST(0) = 2^ST(0) - 1
    Fyl2x,   // ST(1) = ST(1) * log2(ST(0)); pop
    Fyl2xp1, // ST(1) = ST(1) * log2(ST(0) + 1); pop
    Fsincos, // ST(0) = sin(ST(0)); then push cos(ST(0))
    // x87 unit management. `fninit` reinitializes the FPU (control word 0x037F,
    // status word 0, tag word all-empty, TOP 0); `fnclex` clears the exception
    // flags. The waiting forms (`finit`/`fclex`) map to the same kinds — FP
    // exceptions and the wait are not modeled, so the wait is a no-op.
    Fninit, // reset control word / status word / TOP
    Fnclex, // clear the exception flags
}

/// x87 argument-reduction domain for `fsin`/`fcos`/`fptan`/`fsincos`: hardware reduces
/// only when the operand is finite with `|ST(0)| < 2^63` (`exp < 63`); outside it, C2 is
/// set and the operand is left unchanged. Inf/NaN flow through the compute path (→ NaN).
fn in_reduction_domain(v: F80) -> bool {
    use crate::f80::Class;
    match v.class {
        Class::Zero | Class::Nan | Class::Inf | Class::Unsupported => true,
        Class::Normal => v.exp < 63,
    }
}

// gpr[] slot for RAX (fnstsw ax).
const RAX: usize = 0;

// `fpr[]` holds raw 80-bit bytes; x87 arithmetic decodes to/from `F80` at the stack
// boundary here. The round-trip is exact for the normal floats x87 produces.
fn push(cpu: &mut CpuState, v: F80) {
    push_raw(cpu, v.to_bytes());
}

/// Push ten raw bytes without going through [`F80`] — for `fld m80`, which is a move.
///
/// Detects **stack overflow** (#IS): "an instruction attempts to load a non-empty x87 FPU
/// register", where non-empty means any tag other than 11 (SDM Vol 1 §8.5.1.1). It sets
/// IE and SF, and C1 to 1 — C1 is what distinguishes overflow from underflow, since both
/// set the same two flags. Masked, the destination receives the QNaN indefinite; unmasked,
/// the instruction is abandoned and TOP does not move, so a handler finds the stack as it
/// was (§8.6).
///
/// The check belongs here rather than at the call sites because every push funnels
/// through it — `fld`, `fild`, `fld1`, `fldz` and the transcendentals that push a second
/// result all reach the same three lines.
fn push_raw(cpu: &mut CpuState, bytes: [u8; 10]) {
    let dst = (cpu.fpu_top.wrapping_sub(1)) & 7;
    if cpu.fpu_empty & (1 << dst) == 0 {
        // C1 = 1 (bit 9): overflow rather than underflow.
        cpu.fpu_sw |= 1 << 9;
        if raise(cpu, Exc::IE.with(Exc::SF)) {
            return; // unmasked: TOP unmoved, destination untouched
        }
        cpu.fpu_top = dst;
        cpu.fpr[dst as usize] = F80::indefinite().to_bytes();
        cpu.fpu_empty &= !(1 << dst);
        return;
    }
    // A successful push clears C1 — it is only meaningful for the fault it reports.
    cpu.fpu_sw &= !(1 << 9);
    cpu.fpu_top = dst;
    cpu.fpr[dst as usize] = bytes;
    cpu.fpu_empty &= !(1 << dst);
}

fn pop(cpu: &mut CpuState) -> F80 {
    let v = F80::from_bytes(&cpu.fpr[cpu.fpu_top as usize]);
    // The popped register is tagged empty; its bytes are left alone, which is what
    // hardware does and what `fnstenv`'s tag word then reports.
    cpu.fpu_empty |= 1 << cpu.fpu_top;
    cpu.fpu_top = (cpu.fpu_top + 1) & 7;
    v
}

/// Read `ST(i)` as an OPERAND, detecting stack underflow (#IS).
///
/// "An instruction references an empty x87 FPU register as a source operand, including
/// attempting to write the contents of an empty register to memory" (SDM Vol 1 §8.5.1.1).
/// The detection point is therefore the READ — not the pop, which is the tempting place
/// and the wrong one: it would catch `fdivp` on an empty stack and miss `fadd st0, st3`
/// with ST(3) empty, which reads without popping.
///
/// Sets IE and SF, and C1 to **0** — overflow sets the same two flags with C1 at 1, so C1
/// is the only thing that distinguishes them.
///
/// `None` means the instruction must be ABANDONED: the exception was unmasked, so TOP and
/// the source operands stay as they were and `#MF` arrives on the next waiting op (§8.6).
/// Returning an `Option` rather than a value is what makes that non-optional at the call
/// site — every one of the thirty readers has to say what it does about it.
fn st_operand(cpu: &mut CpuState, i: u8) -> Option<F80> {
    let phys = ((cpu.fpu_top + i as u32) & 7) as usize;
    if cpu.fpu_empty & (1 << phys) != 0 {
        cpu.fpu_sw &= !(1 << 9); // C1 = 0: underflow
        if raise(cpu, Exc::IE.with(Exc::SF)) {
            return None;
        }
        // Masked: the operation proceeds on the QNaN indefinite.
        return Some(F80::indefinite());
    }
    Some(F80::from_bytes(&cpu.fpr[phys]))
}

/// `ST(i)` as an operand, abandoning the instruction when the read faults unmasked.
macro_rules! operand {
    ($cpu:expr, $i:expr) => {
        match st_operand($cpu, $i) {
            Some(v) => v,
            None => return None,
        }
    };
}

fn set_st(cpu: &mut CpuState, i: u8, v: F80) {
    let phys = ((cpu.fpu_top + i as u32) & 7) as usize;
    cpu.fpr[phys] = v.to_bytes();
    cpu.fpu_empty &= !(1 << phys);
}

// --- raw guest memory access (bounds-checked; matches the string helper) ---

/// Failure of a legacy FP/SSE image transfer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FpStateFault {
    Memory {
        addr: u64,
        write: bool,
    },
    /// Reserved MXCSR bits (SDM Vol 1 §10.2.3; Vol 2A §3.3 FXRSTOR).
    GeneralProtection,
}

/// fxsave/fxrstor (§14): transfer the 512-byte legacy FP/SSE area at `addr`.
/// XMM, x87 and MXCSR round-trip. MXCSR is stored but does not yet govern
/// arithmetic. Reserved bits fault before any CPU state is restored
/// (SDM Vol 2A §3.3 FXRSTOR, 64-Bit Mode Exceptions).
pub fn exec_fxstate<M: FpMem>(
    cpu: &mut CpuState,
    mem: &M,
    addr: u64,
    restore: bool,
) -> Option<FpStateFault> {
    if restore {
        let mut buf = [0u8; 512];
        if !mem.load(addr, &mut buf) {
            return Some(FpStateFault::Memory {
                addr: mem.fault_addr(addr),
                write: false,
            });
        }
        let mxcsr = u32::from_le_bytes(buf[24..28].try_into().unwrap());
        if mxcsr & !crate::state::MXCSR_MASK != 0 {
            return Some(FpStateFault::GeneralProtection);
        }
        cpu.mxcsr = mxcsr;
        cpu.fpu_cw = normalize_cw(u16::from_le_bytes([buf[0], buf[1]]));
        let sw = u16::from_le_bytes([buf[2], buf[3]]);
        cpu.fpu_top = ((sw >> 11) & 7) as u32;
        cpu.fpu_sw = sw & !0x3800;
        // Abridged tag word, byte 4: bit j is 0 when register j is empty. The SDM writes
        // this as "STj" (Vol 1 §10.5.1.1), but MEASURED against hardware the index is the
        // PHYSICAL register, not the top-relative one: after `fninit; fld1` — which
        // leaves TOP at 7, so ST(0) is R7 — hardware saves `0x80`, and a top-relative
        // reading would have given `0x01`.
        cpu.fpu_empty = !buf[4];
        // The ST slots, by contrast, ARE top-relative: the same measurement finds 1.0 in
        // slot 0, which is ST(0) = R7. Loading slot i straight into `fpr[i]` would rotate
        // the whole stack on every fxsave/fxrstor pair taken with TOP != 0.
        for j in 0..8u32 {
            let off = 32 + j as usize * 16;
            cpu.fpr[((cpu.fpu_top + j) & 7) as usize] = buf[off..off + 10].try_into().unwrap();
        }
        for i in 0..16 {
            let off = 160 + i * 16;
            cpu.xmm[i] = u128::from_le_bytes(buf[off..off + 16].try_into().unwrap());
        }
    } else {
        let mut buf = [0u8; 512];
        buf[0..2].copy_from_slice(&cpu.fpu_cw.to_le_bytes());
        buf[2..4].copy_from_slice(&status_word(cpu).to_le_bytes());
        // Abridged FTW, physical-register indexed — see the restore path for the
        // measurement.
        buf[4] = !cpu.fpu_empty;
        // SDM Vol 2A §3.3 FXSAVE, Tables 3-48/3-49: MXCSR at 24, mask at 28.
        buf[24..28].copy_from_slice(&cpu.mxcsr.to_le_bytes());
        buf[28..32].copy_from_slice(&crate::state::MXCSR_MASK.to_le_bytes());
        // ST slots are top-relative: slot j holds ST(j) = `fpr[(top + j) & 7]`.
        for j in 0..8u32 {
            let off = 32 + j as usize * 16;
            buf[off..off + 10].copy_from_slice(&cpu.fpr[((cpu.fpu_top + j) & 7) as usize]);
        }
        for i in 0..16 {
            let off = 160 + i * 16;
            buf[off..off + 16].copy_from_slice(&cpu.xmm[i].to_le_bytes());
        }
        if !mem.store(addr, &buf) {
            return Some(FpStateFault::Memory {
                addr: mem.fault_addr(addr),
                write: true,
            });
        }
    }
    None
}

/// Read `n` bytes of a memory operand. `None` means the load FAULTED.
///
/// Do not propagate that `None` out of [`exec_x87`] with `?`: this function's `None` and
/// `exec_x87`'s `None` mean opposite things — here it is a fault, there it is success — so
/// `?` turns an unmapped operand into a silent no-op that advances RIP. Translate it to
/// `Some((addr, false))` at the call site, the way the store paths return
/// `Some((addr, true))`.
fn read_n<M: FpMem>(mem: &M, addr: u64, n: usize) -> Option<[u8; 10]> {
    let mut buf = [0u8; 10];
    if mem.load(addr, &mut buf[..n]) {
        Some(buf)
    } else {
        None
    }
}

/// Rounding control: control-word bits 11:10 (SDM Vol 1 §4.8.4, Table 4-9) — `00` nearest
/// (ties to even), `01` toward −∞, `10` toward +∞, `11` toward zero. Witnessed end to end
/// by `x87_fldenv_restores_control_word_matches_unicorn`, which `fistp`s the pair
/// `(0.75, -0.75)` under each mode: the pair separates all four, so a mis-decoded field
/// cannot pass.
fn rc(cpu: &CpuState) -> u8 {
    ((cpu.fpu_cw >> 10) & 0b11) as u8
}

/// The guest's control word, as the rounding/precision pair the arithmetic takes.
/// Every add, sub, mul, div and sqrt must go through this: a guest that sets
/// round-toward-zero or 24/53-bit precision — which is the entire purpose of `fldcw` —
/// would otherwise silently get nearest-even at 64 bits, with no trap.
fn ctl(cpu: &CpuState) -> Ctl {
    Ctl(cpu.fpu_cw)
}

/// Normalize a control word the way the hardware does when it is loaded.
///
/// MEASURED on this host through the native oracle, because the SDM describes bits 6, 7
/// and 15:13 only as "reserved" and does not say what a load makes of them:
///
/// | `fldcw` | `fnstcw` reads back |
/// |---|---|
/// | `0x0000` | `0x0040` |
/// | `0x033F` | `0x037F` |
/// | `0x0FFF` | `0x0F7F` |
/// | `0xFFFF` | `0x1F7F` |
///
/// So bit 6 is forced set, bit 7 forced clear, and bits 15:13 cleared — the infinity
/// control at bit 12 survives. Storing the raw value instead made every `fldcw` in a
/// guest visibly disagree with hardware on a field nothing reads, which is harmless right
/// up until a guest round-trips its control word and compares.
fn normalize_cw(raw: u16) -> u16 {
    (raw | 0x0040) & 0x1F7F
}

/// The x87 tag word (SDM Vol 1 §8.1.7): two bits per **physical** register `R(i)` at
/// bits `2i+1:2i` — `00` valid, `01` zero, `10` special (denormal, unnormal, infinity
/// or NaN), `11` empty. The three non-empty encodings are derived from the live `fpr[]`
/// bytes (verified against hardware); `11` comes from `fpu_empty`, the architectural
/// stack-emptiness state that [`exec_fxstate`] also writes as the abridged FTW.
///
/// Pinned against measured hardware by `x87_tag_word_after_fninit_matches_hardware`:
/// `fninit; fnstenv` gives `0xffff` (all empty), `fninit; fld1; fnstenv` gives `0x3fff`.
fn tag_word(cpu: &CpuState) -> u16 {
    let mut tw = 0u16;
    for (i, r) in cpu.fpr.iter().enumerate() {
        let tag: u16 = if cpu.fpu_empty & (1 << i) != 0 {
            3 // empty
        } else {
            let exp = u16::from_le_bytes([r[8], r[9]]) & 0x7fff;
            let mant = u64::from_le_bytes(r[0..8].try_into().unwrap());
            if exp == 0 && mant == 0 {
                1 // zero
            } else if exp == 0 || exp == 0x7fff || mant & (1 << 63) == 0 {
                2 // special: denormal, unnormal, infinity or NaN
            } else {
                0 // valid
            }
        };
        tw |= tag << (2 * i);
    }
    tw
}

/// The 28-byte x87 environment image `fnstenv m28byte` writes (SDM Vol 1 Figure 8-9,
/// the 32-bit protected-mode layout — the one 64-bit mode uses at the default operand
/// size). Each architectural 16-bit word occupies the low half of a dword whose
/// reserved upper half hardware fills with `0xFFFF`, not zero.
///
/// Fidelity, field by field:
///
/// * `0` control word — exact (`fpu_cw`).
/// * `4` status word — exact: TOP from `fpu_top`, the rest from `fpu_sw`, the same value
///   `fnstsw` reports (see [`status_word`]).
/// * `8` tag word — derived, see [`tag_word`].
/// * `12` FPU instruction-pointer offset, `16` CS selector + last opcode, `20` FPU data
///   -pointer offset, `24` data selector — **not modeled**. This FPU tracks no
///   last-instruction pointer, selector or opcode, so the image carries their reset
///   (post-`fninit`) value; that is the whole of what an unmodeled field can say here,
///   since one store cannot partially refuse. [`load_env28`] ignores them symmetrically,
///   so the round trip a `fenv` save/restore performs loses nothing observable.
fn env28(cpu: &CpuState) -> [u8; 28] {
    let mut buf = [0u8; 28];
    buf[0..2].copy_from_slice(&cpu.fpu_cw.to_le_bytes());
    buf[4..6].copy_from_slice(&status_word(cpu).to_le_bytes());
    buf[8..10].copy_from_slice(&tag_word(cpu).to_le_bytes());
    // FIP / CS+FOP / FDP / FDS: carried verbatim from whatever `fldenv` last loaded.
    // Nothing updates them — this FPU tracks no last-instruction pointer — but keeping
    // them makes a save/restore round trip exact instead of zeroing four fields the
    // guest saved.
    buf[12..28].copy_from_slice(&cpu.fpu_env_tail);
    for off in [2, 6, 10, 26] {
        buf[off..off + 2].copy_from_slice(&0xffffu16.to_le_bytes());
    }
    buf
}

/// The full status word: TOP from `fpu_top`, everything else from `fpu_sw`
/// (SDM Vol 1 §8.1.3). `fpu_top` stays the single source of truth for the stack pointer,
/// so the two cannot drift apart.
fn status_word(cpu: &CpuState) -> u16 {
    (cpu.fpu_sw & !0x3800) | ((cpu.fpu_top as u16 & 7) << 11)
}

/// Apply the 28-byte environment image `fldenv m28byte` reads — the inverse of
/// [`env28`], and deliberately narrower than it.
///
/// `fldenv` loads the *environment only*: the eight data registers keep their contents
/// (restoring those is `frstor`), and unlike `fnclex` it clears nothing — whatever the
/// image holds is what lands.
///
/// Field by field, and what this FPU can act on:
///
/// * `0` control word — **honored in full**, the same assignment `fldcw` makes. It
///   carries the rounding control [`rc`] reads for `fist`/`fistp`, so discarding it
///   would be wrong arithmetic with no trap; FreeBSD's `fenv` restore around
///   `powf`/`expf` exists precisely to put this word back.
/// * `4` status word — **honored in full**: TOP goes to `fpu_top`, the rest to `fpu_sw`,
///   so the `fnstenv`/`fldenv` pair a `fenv_t` restore performs is a lossless round trip.
/// * `8` tag word — **only the `11`/empty tags**, into `fpu_empty`. The other three
///   encodings are derived from the live `fpr[]` bytes at every store ([`tag_word`], and
///   the abridged FTW in [`exec_fxstate`]), so there is no state for them to load into,
///   and `fldenv` may not touch `fpr[]` to manufacture one.
/// * `12` FIP, `16` CS selector + last opcode, `20` FDP, `24` data selector —
///   **ignored**, symmetrically with [`env28`] never having produced a real one.
///
/// A loaded control word can **unmask** an exception whose flag is already set in the
/// loaded status word, which on hardware raises `#MF` on the next FP instruction. Here ES
/// is recomputed only when something new is raised (see [`raise`]), so a newly-unmasked
/// pending flag does not by itself trap; the next raise does see the new masks. The SSE
/// half of the same `fenv_t` uses LDMXCSR/STMXCSR: MXCSR is stored and round-trips,
/// but does not yet govern arithmetic.
///
/// `fnsave`/`frstor` stay unlifted. They move the eight data registers, and their image
/// holds those in *stack* order `ST(0)..ST(7)` while `fpr[]` is indexed physically and
/// the only other register-image path here ([`exec_fxstate`]) copies it physically.
/// Choosing a convention there is a register-file question with its own differential
/// surface, not a consequence of the environment decision above, so they keep trapping.
fn load_env28(cpu: &mut CpuState, buf: &[u8; 28]) {
    cpu.fpu_cw = normalize_cw(u16::from_le_bytes([buf[0], buf[1]]));
    let sw = u16::from_le_bytes([buf[4], buf[5]]);
    cpu.fpu_top = ((sw >> 11) & 7) as u32;
    // The rest of the status word — exception flags, SF, ES, C0–C3, B — is stored rather
    // than dropped: a `fenv_t` restore is a round trip, and dropping half of it makes the
    // pair `fnstenv`/`fldenv` lossy in a way a guest can see with a comparison.
    cpu.fpu_sw = sw & !0x3800;
    // Only `11`/empty is taken from the image; the non-empty encodings (`00` valid,
    // `01` zero, `10` special) are derived from the live bytes at store time.
    let tw = u16::from_le_bytes([buf[8], buf[9]]);
    let mut empty = 0u8;
    for i in 0..8 {
        if (tw >> (2 * i)) & 3 == 3 {
            empty |= 1 << i;
        }
    }
    cpu.fpu_empty = empty;
    cpu.fpu_env_tail.copy_from_slice(&buf[12..28]);
}

/// Record the exceptions an operation raised into the status word.
///
/// The six flags are **sticky** — "once set, they remain set until explicitly cleared"
/// (SDM Vol 1 §8.1.3.3) — so this ORs rather than assigns; `fnclex`, `fninit` and
/// `fldenv` are what clear them.
///
/// ES (bit 7) is NOT one of the six. It is set when any **unmasked** exception flag is
/// set: "if an exception flag is masked, the x87 FPU will still set the appropriate flag
/// if the associated exception occurs, but it will not set the ES flag" (same section).
/// So it is recomputed from the whole status word against the current masks, not ORed in
/// per operation — a later `fldcw` that unmasks something does not retroactively set it,
/// but the next raise sees the new masks. B (bit 15) "reflects the contents of the ES
/// flag" and is "included for 8087 compatibility only", so it simply mirrors it.
///
/// Returns whether an exception is unmasked. Pre-operation exceptions abandon the
/// instruction (SDM Vol 1 §4.9.2); register post-operation exceptions retain their
/// result (§8.5.4–§8.5.6). The caller chooses the corresponding commit behaviour.
#[must_use]
fn raise(cpu: &mut CpuState, exc: Exc) -> bool {
    if exc.is_empty() {
        return false;
    }
    // SF (bit 6) rides along with IE but is not one of the six maskable flags, so it is
    // recorded outside the 0x3f window and never consulted for masking.
    cpu.fpu_sw |= exc.0 & 0x7f;
    let unmasked = cpu.fpu_sw & !cpu.fpu_cw & 0x3f;
    if unmasked != 0 {
        cpu.fpu_sw |= (1 << 7) | (1 << 15);
        return true;
    }
    false
}

/// Whether raw double-extended bytes encode a **denormal** operand: biased exponent zero
/// with a non-zero significand (SDM Vol 1 §4.8.3.2). Pseudo-denormals — integer bit set at
/// exponent zero — land here too, which is right: Table 8-3 calls them supported and
/// "handled correctly, considering the biased exponent as 1", i.e. as denormals.
///
/// Read from the RAW bytes rather than from an [`F80`], because the working form has
/// already normalized them: `from_bytes` folds a denormal into `Class::Normal` with a
/// lower exponent, so by the time arithmetic sees the value the fact is gone.
fn f80_bytes_denormal(b: &[u8; 10]) -> bool {
    let sig = u64::from_le_bytes(b[0..8].try_into().unwrap());
    let se = u16::from_le_bytes([b[8], b[9]]);
    se & 0x7fff == 0 && sig != 0
}

/// The same test for an `f64` or `f32` memory operand, by width.
fn float_bytes_denormal(b: &[u8], width: usize) -> bool {
    match width {
        8 => {
            let v = u64::from_le_bytes(b[0..8].try_into().unwrap());
            v & 0x7ff0_0000_0000_0000 == 0 && v & 0x000f_ffff_ffff_ffff != 0
        }
        4 => {
            let v = u32::from_le_bytes(b[0..4].try_into().unwrap());
            v & 0x7f80_0000 == 0 && v & 0x007f_ffff != 0
        }
        _ => false, // integer operands cannot be denormal
    }
}

/// DE for `ST(i)` as an ARITHMETIC operand.
///
/// "The processor reports the denormal-operand exception if an ARITHMETIC instruction
/// attempts to operate on a denormal operand" (SDM Vol 1 §4.9.1.2) — so this is called
/// from the arithmetic arms and deliberately not from [`st_operand`], which every read
/// goes through including `fld`, `fst` and the compares.
fn st_denormal(cpu: &CpuState, i: u8) -> bool {
    let phys = ((cpu.fpu_top + i as u32) & 7) as usize;
    cpu.fpu_empty & (1 << phys) == 0 && f80_bytes_denormal(&cpu.fpr[phys])
}

/// Write the condition codes C3, C2 and C0, and clear C1 (SDM Vol 1 §8.1.3, Figure 8-4:
/// C0 is bit 8, C1 bit 9, C2 bit 10, C3 bit 14 — not contiguous, which is the only thing
/// awkward about them).
fn set_codes(cpu: &mut CpuState, c3: bool, c2: bool, c0: bool) {
    let mut sw = cpu.fpu_sw & !((1 << 8) | (1 << 9) | (1 << 10) | (1 << 14));
    if c0 {
        sw |= 1 << 8;
    }
    if c2 {
        sw |= 1 << 10;
    }
    if c3 {
        sw |= 1 << 14;
    }
    cpu.fpu_sw = sw;
}

/// Whether `kind` performs the implicit wait that reports a pending unmasked exception.
///
/// "All of the x87 FPU instructions except a few special control instructions perform a
/// wait operation ... before they perform their primary operation" (SDM Vol 1 §8.3.12),
/// and §8.6 enumerates the exceptions: FNINIT, FNSTENV, FNSAVE, FNSTSW, FNSTCW, FNCLEX.
///
/// That list is load-bearing, not a detail. A handler reads the status word with FNSTSW
/// and clears it with FNCLEX — if those trapped, a guest would have no way out of its own
/// exception handler.
///
/// **Approximation, stated because it is invisible otherwise:** the lift folds each
/// WAITING form onto the same `FpuKind` as its no-wait twin (`fclex` → `Fnclex`), so the
/// waiting variants of these six do not check either. The difference is only observable
/// to a guest that uses `fclex` rather than `fnclex` *while* an unmasked exception is
/// pending and expects the trap to come from the `fclex` itself.
fn waits_for_pending(kind: FpuKind) -> bool {
    use FpuKind::*;
    !matches!(
        kind,
        Fninit | Fnstenv | Fnstsw | FnstswMem | Fnstcw | Fnclex
    )
}

/// Whether this op must trap `#MF` INSTEAD of executing (SDM Vol 1 §8.6).
///
/// The FPU signals an unmasked exception on the faulting instruction, but the processor
/// "checks the ES flag ... on the next occurrence of a floating-point instruction or a
/// WAIT/FWAIT" and traps there. Reporting on the instruction that caused it would leave
/// the guest's RIP one instruction early — which is the entire reason the rule exists,
/// and why this is a check at the head of the NEXT op rather than a return value from the
/// one that raised.
pub fn mf_pending_before(cpu: &CpuState, kind: FpuKind) -> bool {
    waits_for_pending(kind) && cpu.fpu_sw & (1 << 7) != 0
}

/// ST(0)-destination arithmetic against a memory operand `m` (already widened to
/// F80). The `r` variants reverse the operands.
fn mem_arith(kind: FpuKind, a: F80, m: F80, c: Ctl) -> (F80, Exc) {
    use FpuKind::*;
    match kind {
        FaddMemF64 | FaddMemF32 | FiaddMemI16 | FiaddMemI32 => F80::add_ctl(a, m, c),
        FsubMemF64 | FsubMemF32 | FisubMemI16 | FisubMemI32 => F80::sub_ctl(a, m, c),
        FsubrMemF64 | FsubrMemF32 | FisubrMemI16 | FisubrMemI32 => F80::sub_ctl(m, a, c),
        FmulMemF64 | FmulMemF32 | FimulMemI16 | FimulMemI32 => F80::mul_ctl(a, m, c),
        // A zero divisor raises ZE and yields a correctly-signed infinity (SDM) rather
        // than faulting — that is `F80::div`'s `(_, Zero) => inf(a.sign ^ b.sign)` arm,
        // shared with the float-memory forms.
        FdivMemF64 | FdivMemF32 | FidivMemI16 | FidivMemI32 => F80::div_ctl(a, m, c),
        _ => F80::div_ctl(m, a, c), // FdivrMem* / FidivrMemI*
    }
}

/// Memory-destination width for this op, or `None` for ops without one.
///
/// An unmasked exception can suppress the store (SDM Vol 1 §8.5.1.2), so this
/// width does not imply a committed write. JIT write notifications follow
/// successful [`FpMem::store`] calls.
pub fn mem_write_bytes(kind: FpuKind) -> Option<usize> {
    use FpuKind::*;
    Some(match kind {
        FstpF32 | FstF32 => 4,
        FstpF64 | FstF64 => 8,
        FstpF80 => 10,
        FistpI16 | FisttpI16 => 2,
        FistpI32 | FisttpI32 => 4,
        FistpI64 | FisttpI64 => 8,
        // The control and status words are 16 bits; `fnstenv` writes the 28-byte
        // (32-bit-layout) environment image — see [`env28`].
        Fnstcw | FnstswMem => 2,
        Fnstenv => 28,
        _ => return None,
    })
}

/// Whole memory operand width for the shared x87 fault path. Register forms
/// return zero. Widths mirror the instruction forms in exec_x87 (SDM Vol 2A,
/// FLD/FILD/FADD/FIADD/FLDCW/FLDENV, Description and Operation).
pub fn mem_operand_bytes(kind: FpuKind) -> u64 {
    use FpuKind::*;
    if let Some(bytes) = mem_write_bytes(kind) {
        return bytes as u64;
    }
    match kind {
        FldF80 => 10,
        FldF64 | FildI64 | FaddMemF64 | FsubMemF64 | FsubrMemF64 | FmulMemF64 | FdivMemF64
        | FdivrMemF64 | FcomMemF64 | FcompMemF64 => 8,
        FldF32 | FildI32 | FaddMemF32 | FsubMemF32 | FsubrMemF32 | FmulMemF32 | FdivMemF32
        | FdivrMemF32 | FicomI32 | FicompI32 | FiaddMemI32 | FisubMemI32 | FisubrMemI32
        | FimulMemI32 | FidivMemI32 | FidivrMemI32 | FcomMemF32 | FcompMemF32 => 4,
        FildI16 | Fldcw | FicomI16 | FicompI16 | FiaddMemI16 | FisubMemI16 | FisubrMemI16
        | FimulMemI16 | FidivMemI16 | FidivrMemI16 => 2,
        Fldenv => 28,
        _ => 0,
    }
}

/// Execute one x87 op. `mem` is the guest memory (see [`FpMem`]); `addr` is the
/// (already computed) effective address for memory forms; `sti` selects `ST(i)`
/// for register forms. Returns `Some((addr, is_write))` on a memory fault.
pub fn exec_x87<M: FpMem>(
    cpu: &mut CpuState,
    mem: &M,
    kind: FpuKind,
    addr: u64,
    sti: u8,
) -> Option<(u64, bool)> {
    // SDM Vol 3A §7.5: a memory fault retains instruction-entry state. Operand
    // conversion can set masked x87 status before its destination store faults.
    let status = cpu.fpu_sw;
    let fault = exec_x87_inner(cpu, mem, kind, addr, sti);
    if fault.is_some() {
        cpu.fpu_sw = status;
    }
    fault
}

fn exec_x87_inner<M: FpMem>(
    cpu: &mut CpuState,
    mem: &M,
    kind: FpuKind,
    addr: u64,
    sti: u8,
) -> Option<(u64, bool)> {
    use FpuKind::*;
    // Transcendental precision: Extended = full-80-bit F80 series, else f64.
    let ext = cpu.x87_precision == crate::state::X87Precision::Extended;
    match kind {
        FldF64 => {
            let Some(b) = read_n(mem, addr, 8) else {
                return Some((mem.fault_addr(addr), false));
            };
            // #D on the LOAD, not just on later arithmetic. §4.9.1.2 says "arithmetic
            // instruction", which reads as excluding `fld` — MEASURED otherwise: `fld
            // qword` of a denormal, with nothing else in the program, leaves the host's
            // status word at 0x3802. `fld m32`/`m64` CONVERTS to double extended, and the
            // conversion is what sees the denormal; after it the register holds an
            // ordinary 80-bit value, so nothing downstream could ever notice.
            if float_bytes_denormal(&b, 8) && raise(cpu, Exc::DE) {
                return None;
            }
            push(
                cpu,
                F80::from_f64(u64::from_le_bytes(b[0..8].try_into().unwrap())),
            );
        }
        FldF32 => {
            let Some(b) = read_n(mem, addr, 4) else {
                return Some((mem.fault_addr(addr), false));
            };
            // #D on the conversion — see `FldF64`.
            if float_bytes_denormal(&b, 4) && raise(cpu, Exc::DE) {
                return None;
            }
            let v = f32::from_le_bytes(b[0..4].try_into().unwrap());
            push(cpu, F80::from_f64((v as f64).to_bits())); // f32 -> f80 is exact
        }
        FldF80 => {
            let Some(b) = read_n(mem, addr, 10) else {
                return Some((mem.fault_addr(addr), false));
            };
            // The ten bytes go into the register VERBATIM — no decode/re-encode. `fld
            // m80` is a move, not a conversion: measured against hardware, an unnormal
            // and a pseudo-denormal both land in the register exactly as they were in
            // memory, and only reach a rule when arithmetic reads them. A round trip
            // through `F80` would renormalize the pseudo-denormal's exponent from 0 to 1.
            push_raw(cpu, b);
        }
        FildI16 => {
            let Some(b) = read_n(mem, addr, 2) else {
                return Some((mem.fault_addr(addr), false));
            };
            push(
                cpu,
                F80::from_i64(i16::from_le_bytes(b[0..2].try_into().unwrap()) as i64),
            );
        }
        FildI32 => {
            let Some(b) = read_n(mem, addr, 4) else {
                return Some((mem.fault_addr(addr), false));
            };
            push(
                cpu,
                F80::from_i64(i32::from_le_bytes(b[0..4].try_into().unwrap()) as i64),
            );
        }
        FildI64 => {
            let Some(b) = read_n(mem, addr, 8) else {
                return Some((mem.fault_addr(addr), false));
            };
            push(
                cpu,
                F80::from_i64(i64::from_le_bytes(b[0..8].try_into().unwrap())),
            );
        }
        FstpF64 | FstF64 => {
            let v = operand!(cpu, 0).to_f64();
            if !mem.store(addr, &v.to_le_bytes()) {
                return Some((mem.fault_addr(addr), true));
            }
            if kind == FstpF64 {
                pop(cpu);
            }
        }
        FstpF32 | FstF32 => {
            let v = f64::from_bits(operand!(cpu, 0).to_f64()) as f32;
            if !mem.store(addr, &v.to_le_bytes()) {
                return Some((mem.fault_addr(addr), true));
            }
            if kind == FstpF32 {
                pop(cpu);
            }
        }
        FstpF80 => {
            // The store side of the same move: `fstp m80` writes the register's bytes
            // VERBATIM, which is why this arm does not go through `operand!` — a round
            // trip through `F80` would renormalize a pseudo-denormal and destroy an
            // unnormal, and hardware keeps both.
            //
            // The underflow check therefore has to be written out here rather than
            // inherited from `operand!`: SDM Vol 1 §8.5.1.1 names the case outright —
            // "including attempting to write the contents of an empty register to
            // memory" — so an empty ST(0) must not store ten bytes of stale register
            // data.
            let phys = cpu.fpu_top as usize;
            let bytes = if cpu.fpu_empty & (1 << phys) != 0 {
                cpu.fpu_sw &= !(1 << 9); // C1 = 0: underflow, not overflow
                if raise(cpu, Exc::IE.with(Exc::SF)) {
                    return None; // unmasked: nothing stored, TOP unmoved
                }
                // Masked: the destination gets the indefinite, in the destination's own
                // format — which for `m80` is the 80-bit encoding, not a converted one.
                F80::indefinite().to_bytes()
            } else {
                cpu.fpr[phys]
            };
            if !mem.store(addr, &bytes) {
                return Some((mem.fault_addr(addr), true));
            }
            pop(cpu);
        }
        FistpI16 | FistpI32 | FistpI64 | FisttpI16 | FisttpI32 | FisttpI64 => {
            // SDM Vol 2A §3.3 FIST/FISTP and FISTTP: invalid conversion stores
            // the destination-width indefinite only when IM is masked. FISTTP
            // ignores RC and always clears C1; FISTP uses C1 for rounding.
            let bytes = mem_write_bytes(kind).unwrap();
            let trunc = matches!(kind, FisttpI16 | FisttpI32 | FisttpI64);
            cpu.fpu_sw &= !(1 << 9);
            let a = operand!(cpu, 0);
            let (v, exc, up) = a.to_int_rc((bytes * 8) as u32, if trunc { 3 } else { rc(cpu) });
            if raise(cpu, exc) && exc.has(Exc::IE) {
                return None;
            }
            if !trunc && up {
                cpu.fpu_sw |= 1 << 9;
            }
            if !mem.store(addr, &v.to_le_bytes()[..bytes]) {
                return Some((mem.fault_addr(addr), true));
            }
            // SDM Vol 1 §§4.9.1.6, 8.5.6: an unmasked precision exception
            // retains the rounded result, unlike an invalid conversion.
            pop(cpu);
        }
        FaddMemF64 | FsubMemF64 | FsubrMemF64 | FmulMemF64 | FdivMemF64 | FdivrMemF64 => {
            let Some(b) = read_n(mem, addr, 8) else {
                return Some((mem.fault_addr(addr), false));
            };
            let m = F80::from_f64(u64::from_le_bytes(b[0..8].try_into().unwrap()));
            let a = operand!(cpu, 0);
            let c = ctl(cpu);
            // #D: either operand of an ARITHMETIC instruction being denormal. The memory
            // operand is checked from the bytes just read; ST(0) from its raw register
            // image, since `operand!` returns an already-normalized `F80`.
            let den = float_bytes_denormal(&b, 8) || st_denormal(cpu, 0);
            let (r, e) = mem_arith(kind, a, m, c);
            let e = if den { e.with(Exc::DE) } else { e };
            if raise(cpu, e) {
                return None;
            }
            set_st(cpu, 0, r);
        }
        FaddMemF32 | FsubMemF32 | FsubrMemF32 | FmulMemF32 | FdivMemF32 | FdivrMemF32 => {
            let Some(b) = read_n(mem, addr, 4) else {
                return Some((mem.fault_addr(addr), false));
            };
            let v = f32::from_le_bytes(b[0..4].try_into().unwrap());
            let m = F80::from_f64((v as f64).to_bits());
            let a = operand!(cpu, 0);
            let c = ctl(cpu);
            // #D: either operand of an ARITHMETIC instruction being denormal. The memory
            // operand is checked from the bytes just read; ST(0) from its raw register
            // image, since `operand!` returns an already-normalized `F80`.
            let den = float_bytes_denormal(&b, 4) || st_denormal(cpu, 0);
            let (r, e) = mem_arith(kind, a, m, c);
            let e = if den { e.with(Exc::DE) } else { e };
            if raise(cpu, e) {
                return None;
            }
            set_st(cpu, 0, r);
        }
        // Integer-memory arithmetic. The operand is read at its architectural width,
        // sign-extended, and widened to F80 — exactly what `FildI16`/`FildI32` do, and
        // exact for every i16/i32 — so the only rounding happens inside `mem_arith`.
        FiaddMemI16 | FisubMemI16 | FisubrMemI16 | FimulMemI16 | FidivMemI16 | FidivrMemI16 => {
            let Some(b) = read_n(mem, addr, 2) else {
                return Some((mem.fault_addr(addr), false));
            };
            let m = F80::from_i64(i16::from_le_bytes(b[0..2].try_into().unwrap()) as i64);
            let a = operand!(cpu, 0);
            let c = ctl(cpu);
            // #D: either operand of an ARITHMETIC instruction being denormal. The memory
            // operand is checked from the bytes just read; ST(0) from its raw register
            // image, since `operand!` returns an already-normalized `F80`.
            let den = float_bytes_denormal(&b, 2) || st_denormal(cpu, 0);
            let (r, e) = mem_arith(kind, a, m, c);
            let e = if den { e.with(Exc::DE) } else { e };
            if raise(cpu, e) {
                return None;
            }
            set_st(cpu, 0, r);
        }
        FiaddMemI32 | FisubMemI32 | FisubrMemI32 | FimulMemI32 | FidivMemI32 | FidivrMemI32 => {
            let Some(b) = read_n(mem, addr, 4) else {
                return Some((mem.fault_addr(addr), false));
            };
            let m = F80::from_i64(i32::from_le_bytes(b[0..4].try_into().unwrap()) as i64);
            let a = operand!(cpu, 0);
            let c = ctl(cpu);
            // #D: either operand of an ARITHMETIC instruction being denormal. The memory
            // operand is checked from the bytes just read; ST(0) from its raw register
            // image, since `operand!` returns an already-normalized `F80`.
            let den = float_bytes_denormal(&b, 4) || st_denormal(cpu, 0);
            let (r, e) = mem_arith(kind, a, m, c);
            let e = if den { e.with(Exc::DE) } else { e };
            if raise(cpu, e) {
                return None;
            }
            set_st(cpu, 0, r);
        }
        FldSti => {
            let v = operand!(cpu, sti);
            push(cpu, v);
        }
        Fld1 => push(cpu, F80::from_i64(1)),
        Fldz => push(cpu, F80::zero(false)),
        FaddP | FsubP | FsubrP | FmulP | FdivP | FdivrP => {
            let (s0, si) = (operand!(cpu, 0), operand!(cpu, sti));
            // #D on either arithmetic operand, from the raw register images — the
            // working `F80` has already normalized a denormal away.
            let den = st_denormal(cpu, 0) || st_denormal(cpu, sti);
            let c = ctl(cpu);
            let r = match kind {
                FaddP => F80::add_ctl(si, s0, c),
                FsubP => F80::sub_ctl(si, s0, c),
                FsubrP => F80::sub_ctl(s0, si, c),
                FmulP => F80::mul_ctl(si, s0, c),
                FdivP => F80::div_ctl(si, s0, c),
                _ => F80::div_ctl(s0, si, c),
            };
            let (r, e) = r;
            let e = if den { e.with(Exc::DE) } else { e };
            if raise(cpu, e) {
                // Unmasked: the instruction is abandoned, so ST(i) keeps its value and
                // TOP does not move (SDM Vol 1 §8.6, and §8.5.1.1 for the stack cases).
                return None;
            }
            set_st(cpu, sti, r);
            pop(cpu);
        }
        FstSti | FstpSti => {
            // fst/fstp st(i): copy ST(0) into ST(i); the `p` form then pops.
            let v = operand!(cpu, 0);
            set_st(cpu, sti, v);
            if kind == FstpSti {
                pop(cpu);
            }
        }
        FaddSti | FsubSti | FsubrSti | FmulSti | FdivSti | FdivrSti => {
            // Register-form arithmetic with ST(0) as the destination (no pop).
            let (s0, si) = (operand!(cpu, 0), operand!(cpu, sti));
            // #D on either arithmetic operand, from the raw register images — the
            // working `F80` has already normalized a denormal away.
            let den = st_denormal(cpu, 0) || st_denormal(cpu, sti);
            let c = ctl(cpu);
            let r = match kind {
                FaddSti => F80::add_ctl(s0, si, c),
                FsubSti => F80::sub_ctl(s0, si, c),
                FsubrSti => F80::sub_ctl(si, s0, c),
                FmulSti => F80::mul_ctl(s0, si, c),
                FdivSti => F80::div_ctl(s0, si, c),
                _ => F80::div_ctl(si, s0, c),
            };
            let (r, e) = r;
            let e = if den { e.with(Exc::DE) } else { e };
            if raise(cpu, e) {
                return None;
            }
            set_st(cpu, 0, r);
        }
        FaddToSti | FsubToSti | FsubrToSti | FmulToSti | FdivToSti | FdivrToSti => {
            // Register-form arithmetic with ST(i) as the destination (no pop).
            let (s0, si) = (operand!(cpu, 0), operand!(cpu, sti));
            // #D on either arithmetic operand, from the raw register images — the
            // working `F80` has already normalized a denormal away.
            let den = st_denormal(cpu, 0) || st_denormal(cpu, sti);
            let c = ctl(cpu);
            let r = match kind {
                FaddToSti => F80::add_ctl(si, s0, c),
                FsubToSti => F80::sub_ctl(si, s0, c),
                FsubrToSti => F80::sub_ctl(s0, si, c),
                FmulToSti => F80::mul_ctl(si, s0, c),
                FdivToSti => F80::div_ctl(si, s0, c),
                _ => F80::div_ctl(s0, si, c),
            };
            let (r, e) = r;
            let e = if den { e.with(Exc::DE) } else { e };
            if raise(cpu, e) {
                return None;
            }
            set_st(cpu, sti, r);
        }
        Fxch => {
            let (a, b) = (operand!(cpu, 0), operand!(cpu, sti));
            set_st(cpu, 0, b);
            set_st(cpu, sti, a);
        }
        // SDM Vol 2A §3.2 FCMOVcc: only stack faults can signal; all 80 bits
        // move unchanged, including SNaNs and unsupported encodings.
        Fcmovb | Fcmove | Fcmovbe | Fcmovu | Fcmovnb | Fcmovne | Fcmovnbe | Fcmovnu => {
            let dst = cpu.fpu_top as usize;
            let src = (dst + sti as usize) & 7;
            if cpu.fpu_empty & ((1 << src) | (1 << dst)) != 0 {
                cpu.fpu_sw &= !(1 << 9);
                if !raise(cpu, Exc::IE.with(Exc::SF)) {
                    set_st(cpu, 0, F80::indefinite());
                }
                return None;
            }
            let take = match kind {
                Fcmovb => cpu.flags.cf,
                Fcmove => cpu.flags.zf,
                Fcmovbe => cpu.flags.cf || cpu.flags.zf,
                Fcmovu => cpu.flags.pf(),
                Fcmovnb => !cpu.flags.cf,
                Fcmovne => !cpu.flags.zf,
                Fcmovnbe => !cpu.flags.cf && !cpu.flags.zf,
                _ => !cpu.flags.pf(),
            };
            if take {
                cpu.fpr[dst] = cpu.fpr[src];
            }
        }
        // SDM Vol 2A §3.2 FXAM, Table 3-44. Classification and sign use the
        // raw register even when its tag is empty; FXAM raises no FP exceptions.
        Fxam => {
            use crate::f80::Class;
            let phys = cpu.fpu_top as usize;
            let raw = cpu.fpr[phys];
            let v = F80::from_bytes(&raw);
            let (c3, c2, c0) = if cpu.fpu_empty & (1 << phys) != 0 {
                (true, false, true)
            } else if f80_bytes_denormal(&raw) {
                (true, true, false)
            } else {
                match v.class {
                    Class::Unsupported => (false, false, false),
                    Class::Nan => (false, false, true),
                    Class::Normal => (false, true, false),
                    Class::Inf => (false, true, true),
                    Class::Zero => (true, false, false),
                }
            };
            set_codes(cpu, c3, c2, c0);
            if raw[9] & 0x80 != 0 {
                cpu.fpu_sw |= 1 << 9;
            }
        }
        // SDM Vol 2A §3.2 FCOM/FUCOM/FTST: invalid precedes denormal,
        // unmasked exceptions preserve the codes and suppress every pop.
        Fcom | Fcomp | Fcompp | Fucom | Fucomp | Fucompp | Ftst | FcomMemF32 | FcomMemF64
        | FcompMemF32 | FcompMemF64 => {
            let memory = matches!(kind, FcomMemF32 | FcomMemF64 | FcompMemF32 | FcompMemF64);
            let mut den = st_denormal(cpu, 0);
            let b = if memory {
                let width = if matches!(kind, FcomMemF32 | FcompMemF32) {
                    4
                } else {
                    8
                };
                let Some(raw) = read_n(mem, addr, width) else {
                    return Some((mem.fault_addr(addr), false));
                };
                den |= float_bytes_denormal(&raw, width);
                if width == 4 {
                    // Preserve the signaling bit during widening (SDM Vol 1 §4.8.3.5).
                    let bits = u32::from_le_bytes(raw[..4].try_into().unwrap());
                    if bits & 0x7f80_0000 == 0x7f80_0000 && bits & 0x7f_ffff != 0 {
                        F80 {
                            sign: bits >> 31 != 0,
                            class: crate::f80::Class::Nan,
                            exp: 0,
                            sig: (1 << 63) | ((bits as u64 & 0x7f_ffff) << 40),
                        }
                    } else {
                        F80::from_f64((f32::from_bits(bits) as f64).to_bits())
                    }
                } else {
                    F80::from_f64(u64::from_le_bytes(raw[..8].try_into().unwrap()))
                }
            } else if kind == Ftst {
                F80::ZERO
            } else {
                den |= st_denormal(cpu, sti);
                operand!(cpu, sti)
            };
            let a = operand!(cpu, 0);
            let ((c3, c2, c0), mut exc) =
                F80::compare_x87(a, b, matches!(kind, Fucom | Fucomp | Fucompp));
            cpu.fpu_sw &= !(1 << 9);
            // SDM Vol 1 §4.9.2: QNaN handling also precedes denormal operands.
            if !exc.has(Exc::IE) && !a.is_nan() && !b.is_nan() && den {
                exc = exc.with(Exc::DE);
            }
            if raise(cpu, exc) {
                return None;
            }
            set_codes(cpu, c3, c2, c0);
            if matches!(
                kind,
                Fcomp | Fcompp | Fucomp | Fucompp | FcompMemF32 | FcompMemF64
            ) {
                pop(cpu);
            }
            if matches!(kind, Fcompp | Fucompp) {
                pop(cpu);
            }
        }
        Fsqrt => {
            // SDM Vol 2A §3.2 FSQRT: RC/PC, DE, IE, PE and C1 rounding.
            let den = st_denormal(cpu, 0);
            let a = operand!(cpu, 0);
            let (r, exc) = F80::sqrt_ctl(a, ctl(cpu));
            cpu.fpu_sw &= !(1 << 9);
            if !exc.has(Exc::IE) && den && raise(cpu, Exc::DE) {
                return None;
            }
            // SDM Vol 1 §8.5.6: unmasked precision retains the rounded result.
            if raise(cpu, exc) && exc.has(Exc::IE) {
                return None;
            }
            if exc.has(Exc::PE) && r.class == crate::f80::Class::Normal {
                let square = (r.sig as u128) * (r.sig as u128);
                let exact = (a.sig as u128) << (63 + a.exp - 2 * r.exp);
                if square > exact {
                    cpu.fpu_sw |= 1 << 9;
                }
            }
            set_st(cpu, 0, r);
        }
        Fscale | Fprem1 => {
            let den = st_denormal(cpu, 0) || st_denormal(cpu, 1);
            let a = operand!(cpu, 0);
            let b = operand!(cpu, 1);
            let (r, exc, partial, q, up) = if kind == Fscale {
                let (r, e, up) = F80::scale(a, b, ctl(cpu));
                (r, e, false, 0, up)
            } else {
                let (r, e, partial, q) = F80::rem_nearest(a, b, ctl(cpu));
                (r, e, partial, q, false)
            };
            cpu.fpu_sw &= !(1 << 9);
            if !exc.has(Exc::IE) && !r.is_nan() && den && raise(cpu, Exc::DE) {
                return None;
            }
            // SDM Vol 1 §8.5.4–§8.5.6: post-operation register faults
            // retain the rounded or biased result.
            if raise(cpu, exc) && exc.has(Exc::IE) {
                return None;
            }
            if kind == Fscale {
                if up {
                    cpu.fpu_sw |= 1 << 9;
                }
            }
            // SDM Vol 3B §25.18.2.1: incomplete reductions clear C0/C1/C3.
            else if partial {
                set_codes(cpu, false, true, false);
            } else if !r.is_nan() {
                set_codes(cpu, q & 2 != 0, false, q & 4 != 0);
                if q & 1 != 0 {
                    cpu.fpu_sw |= 1 << 9;
                }
            }
            // SDM Vol 1 §8.5.1.2 Table 8-10: invalid remainder clears C2.
            else {
                cpu.fpu_sw &= !(1 << 10);
            }
            set_st(cpu, 0, r);
        }
        Fucomi | Fucomip | Fcomi | Fcomip => {
            let (zf, pf, cf) = F80::compare(operand!(cpu, 0), operand!(cpu, sti));
            cpu.flags.zf = zf;
            cpu.flags.set_pf(pf);
            cpu.flags.cf = cf;
            cpu.flags.of = false;
            cpu.flags.sf = false;
            cpu.flags.set_af(false);
            if matches!(kind, Fucomip | Fcomip) {
                pop(cpu);
            }
        }
        // `ficom m16int` / `ficom m32int` and their popping forms.
        //
        // Unlike `fcomi`, these report through the status-word condition codes.
        // SDM Vol 2A Table 3-28:
        //
        // | condition      | C3 | C2 | C0 |
        // |----------------|----|----|----|
        // | ST(0) > SRC    |  0 |  0 |  0 |
        // | ST(0) < SRC    |  0 |  0 |  1 |
        // | ST(0) = SRC    |  1 |  0 |  0 |
        // | Unordered      |  1 |  1 |  1 |
        //
        // with C1 set to 0. This is an ORDERED compare: "#IA — One or both operands are
        // NaN values or have unsupported formats", so a QUIET NaN raises invalid here
        // where `fucom` would stay silent. The integer source cannot be a NaN, so only
        // ST(0) can make it unordered.
        FicomI16 | FicomI32 | FicompI16 | FicompI32 => {
            let width = if matches!(kind, FicomI16 | FicompI16) {
                2
            } else {
                4
            };
            let Some(b) = read_n(mem, addr, width) else {
                return Some((mem.fault_addr(addr), false));
            };
            let m = if width == 2 {
                F80::from_i64(i16::from_le_bytes(b[0..2].try_into().unwrap()) as i64)
            } else {
                F80::from_i64(i32::from_le_bytes(b[0..4].try_into().unwrap()) as i64)
            };
            let a = operand!(cpu, 0);
            // `F80::compare` already returns exactly this triple. It is written as
            // `(zf, pf, cf)` for `fcomi` because the architectural mapping IS
            // ZF <- C3, PF <- C2, CF <- C0 (SDM Vol 1 §8.1.4, Figure 8-5) — the same
            // three bits under two names, so there is one comparison rule and not two.
            let (c3, c2, c0) = F80::compare(a, m);
            // Unordered is the all-ones case, and for an ORDERED compare it is #IA. The
            // integer source cannot be a NaN, so only ST(0) reaches it.
            if c3 && c2 && c0 && raise(cpu, Exc::IE) {
                return None;
            }
            set_codes(cpu, c3, c2, c0);
            if matches!(kind, FicompI16 | FicompI32) {
                pop(cpu);
            }
        }
        Fabs => {
            let v = operand!(cpu, 0).abs();
            set_st(cpu, 0, v);
        }
        Fchs => {
            let v = operand!(cpu, 0).neg();
            set_st(cpu, 0, v);
        }
        Fldcw => {
            let Some(b) = read_n(mem, addr, 2) else {
                return Some((mem.fault_addr(addr), false));
            };
            cpu.fpu_cw = normalize_cw(u16::from_le_bytes([b[0], b[1]]));
        }
        Fnstcw => {
            if !mem.store(addr, &cpu.fpu_cw.to_le_bytes()) {
                return Some((mem.fault_addr(addr), true));
            }
        }
        Fnstsw | FnstswMem => {
            // Status word: TOP in bits 11–13, the rest from `fpu_sw`. The `-i` compares
            // write EFLAGS directly, so guests rarely read C0–C3 from here.
            let sw = status_word(cpu);
            if kind == FnstswMem {
                // `fnstsw m16`: store the 16-bit status word to memory.
                if !mem.store(addr, &sw.to_le_bytes()) {
                    return Some((mem.fault_addr(addr), true));
                }
            } else {
                // `fnstsw ax`: write it to AX.
                cpu.write_gpr(RAX, sw as u64, 2);
            }
        }
        Fnstenv => {
            if !mem.store(addr, &env28(cpu)) {
                return Some((mem.fault_addr(addr), true));
            }
            // SDM: "…and then masks all floating-point exceptions" — the store is
            // followed by setting all six exception-mask bits in the control word.
            // Confirmed on hardware (CW 0x0360 -> 0x037F across one `fnstenv`).
            cpu.fpu_cw |= 0x3f;
        }
        Fldenv => {
            let mut buf = [0u8; 28];
            if !mem.load(addr, &mut buf) {
                return Some((mem.fault_addr(addr), false));
            }
            load_env28(cpu, &buf);
        }
        Fninit => {
            // Reinitialize the x87 unit: control word 0x037F (round-to-nearest,
            // all exceptions masked, 64-bit precision), status word 0, tag word
            // all-empty, TOP 0.
            cpu.fpu_cw = 0x037F;
            cpu.fpu_top = 0;
            // "The status word is cleared ... The data registers in the register stack
            // are left unchanged, but they are all tagged as empty" (SDM Vol 2A
            // FINIT/FNINIT).
            cpu.fpu_sw = 0;
            cpu.fpu_empty = 0xFF;
            cpu.fpu_env_tail = [0; 16];
        }
        Fnclex => {
            // "Clears the floating-point exception flags (PE, UE, OE, ZE, DE, and IE),
            // the exception summary status flag (ES), the stack fault flag (SF), and the
            // busy flag (B) in the FPU status word" — SDM Vol 2A FCLEX/FNCLEX. TOP is
            // left alone.
            //
            // The mask also clears bit 8 (C0), which the same entry's Operation clause
            // (`FPUStatusWord[0:7] := 0; FPUStatusWord[15] := 0`) leaves standing. That
            // is within spec, not an accident: "FPU Flags Affected" declares C0, C1, C2
            // and C3 UNDEFINED after this instruction, so a guest may not read C0 across
            // it. C1-C3 are left alone only because nothing needs them zeroed.
            cpu.fpu_sw &= !0b1000_0001_1111_1111;
        }
        Fprem => {
            let (a, b) = (operand!(cpu, 0), operand!(cpu, 1));
            set_st(cpu, 0, F80::rem(a, b));
        }
        // --- Transcendentals ---
        Fsin => {
            let x = operand!(cpu, 0);
            if in_reduction_domain(x) {
                set_st(cpu, 0, if ext { x.sin_ext() } else { x.sin() });
            }
        }
        Fcos => {
            let x = operand!(cpu, 0);
            if in_reduction_domain(x) {
                set_st(cpu, 0, if ext { x.cos_ext() } else { x.cos() });
            }
        }
        Fptan => {
            let x = operand!(cpu, 0);
            if in_reduction_domain(x) {
                set_st(cpu, 0, if ext { x.tan_ext() } else { x.tan() });
                push(cpu, F80::from_i64(1)); // fptan pushes 1.0 (→ ST(1)=tan, ST(0)=1.0)
            }
        }
        Fsincos => {
            let x = operand!(cpu, 0);
            if in_reduction_domain(x) {
                let (s, c) = if ext {
                    (x.sin_ext(), x.cos_ext())
                } else {
                    (x.sin(), x.cos())
                };
                set_st(cpu, 0, s);
                push(cpu, c); // → ST(0)=cos, ST(1)=sin
            }
        }
        Fpatan => {
            let (a, b) = (operand!(cpu, 1), operand!(cpu, 0));
            let r = if ext {
                F80::atan2_ext(a, b)
            } else {
                F80::atan2(a, b)
            }; // atan(ST1/ST0), full quadrant
            set_st(cpu, 1, r);
            pop(cpu);
        }
        F2xm1 => {
            let x = operand!(cpu, 0);
            set_st(cpu, 0, if ext { x.exp2m1_ext() } else { x.exp2m1() });
        }
        Fyl2x => {
            let (y, x) = (operand!(cpu, 1), operand!(cpu, 0));
            let r = if ext {
                F80::ylog2x_ext(y, x)
            } else {
                F80::ylog2x(y, x)
            };
            set_st(cpu, 1, r);
            pop(cpu);
        }
        Fyl2xp1 => {
            let (y, x) = (operand!(cpu, 1), operand!(cpu, 0));
            let r = if ext {
                F80::ylog2xp1_ext(y, x)
            } else {
                F80::ylog2xp1(y, x)
            };
            set_st(cpu, 1, r);
            pop(cpu);
        }
    }
    None
}
