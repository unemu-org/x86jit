//! Guest CPU state: registers and flags (§3).

use std::fmt;

#[cfg(test)]
std::thread_local! {
    pub(crate) static MMIO_CLEAR_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

use iced_x86::Register;

/// Named guest registers exposed through the public API.
///
/// FS/GS bases are present from the start because real programs use them for
/// thread-local storage (§3.1).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Reg {
    Rax,
    Rbx,
    Rcx,
    Rdx,
    Rsi,
    Rdi,
    Rbp,
    Rsp,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
    Rip,
    FsBase,
    GsBase,
    /// Segment selectors used by Real16 addressing (§17.6) and configured
    /// protected-mode selector images. Kept outside `jit_abi::CpuOffsets`.
    Cs,
    Ds,
    Es,
    Ss,
}

/// `gpr[]` slots in x86 encoding order — the inverse of [`Reg::gpr_index`].
const GPR_BY_INDEX: [Reg; 16] = [
    Reg::Rax,
    Reg::Rcx,
    Reg::Rdx,
    Reg::Rbx,
    Reg::Rsp,
    Reg::Rbp,
    Reg::Rsi,
    Reg::Rdi,
    Reg::R8,
    Reg::R9,
    Reg::R10,
    Reg::R11,
    Reg::R12,
    Reg::R13,
    Reg::R14,
    Reg::R15,
];

impl Reg {
    /// The 64-bit register occupying `gpr[]` slot `index` (x86 encoding order).
    /// Inverse of [`Reg::gpr_index`]. Panics on `index >= 16`.
    pub fn from_gpr_index(index: usize) -> Reg {
        GPR_BY_INDEX[index]
    }

    /// Index into [`CpuState::gpr`] in x86 ENCODING order (RAX=0, RCX=1, RDX=2,
    /// RBX=3, RSP=4, RBP=5, RSI=6, RDI=7, R8=8 … R15=15) — NOT this enum's
    /// declaration order. `None` for registers that aren't in `gpr[]`
    /// (`Rip`/`FsBase`/`GsBase` live in their own `CpuState` fields). (§3.1)
    ///
    /// This and [`iced_gpr_index`] are the ONE place register numbering lives.
    pub fn gpr_index(self) -> Option<usize> {
        Some(match self {
            Reg::Rax => 0,
            Reg::Rcx => 1,
            Reg::Rdx => 2,
            Reg::Rbx => 3,
            Reg::Rsp => 4,
            Reg::Rbp => 5,
            Reg::Rsi => 6,
            Reg::Rdi => 7,
            Reg::R8 => 8,
            Reg::R9 => 9,
            Reg::R10 => 10,
            Reg::R11 => 11,
            Reg::R12 => 12,
            Reg::R13 => 13,
            Reg::R14 => 14,
            Reg::R15 => 15,
            Reg::Rip | Reg::FsBase | Reg::GsBase => return None,
            // Segment selectors live in their own `CpuState` fields, not `gpr[]`.
            Reg::Cs | Reg::Ds | Reg::Es | Reg::Ss => return None,
        })
    }
}

/// Map an iced `Register` of any width (RAX/EAX/AX/AL/AH, R8/R8D/R8W/R8L, …) to
/// its [`CpuState::gpr`] index in x86 encoding order. Sub-width and high-byte
/// registers normalize to their 64-bit parent via iced's `full_register`.
/// `None` for anything that isn't a general-purpose register (RIP, segment,
/// XMM, …). (§3.1)
///
/// iced numbers its 64-bit GPRs (`RAX`..`R15`) contiguously in encoding order,
/// so subtracting the `RAX` discriminant yields the `gpr[]` index directly. The
/// unit tests pin this mapping so an iced change can't silently break it.
pub fn iced_gpr_index(reg: Register) -> Option<usize> {
    if !reg.is_gpr() {
        return None;
    }
    Some(reg.full_register() as usize - Register::RAX as usize)
}

/// Arithmetic + direction flags (Variant A: materialized, §3.2).
///
/// `#[repr(C)]` because the JIT stores/loads individual flag fields at stable
/// offsets inside `CpuState` (§8.2.1) — same contract as `CpuState` itself.
/// One-byte bools are simple and correct; a packed RFLAGS-style `u64` bitfield
/// (fewer stores in codegen) is an M4/M5 optimization, not a day-one requirement.
///
/// Lazy flags (Variant B) are a later optimization and deliberately not modeled
/// here — **except** for PF and AF, which are stored as SOURCES rather than as
/// derived bits. Neither has a hot reader: x86 has no conditional
/// branch on AF at all, and PF is reachable only through `jp`/`setp`/`pushf`/
/// `lahf` and the BCD instructions. Deriving them eagerly cost 15 of the ~33 host
/// instructions the JIT emits for a flag update, which is paid once per compiled
/// block because the last update is live across the block boundary. See
/// [`Flags::pf`] / [`Flags::af`] for the encoding.
#[repr(C)]
#[derive(Copy, Clone, Eq)]
pub struct Flags {
    pub cf: bool,
    /// PF source: the low byte of the value PF was computed from. `pf()` is its
    /// **even** parity. Occupies the byte the derived `pf` bool used to, so no
    /// `CpuState` offset moves.
    pub pf_src: u8,
    /// AF source: `a ^ b ^ result`. `af()` is bit 4. The same expression yields the
    /// carry out of bit 3 for addition and the borrow into it for subtraction, so
    /// one encoding covers both (and `adc`/`sbb`, whose carry-in is already folded
    /// into `result`).
    pub af_src: u8,
    pub zf: bool,
    pub sf: bool,
    pub of: bool,
    pub df: bool,
    /// Interrupt-enable flag (SDM Vol 1 §3.4.3). Real16 CLI/STI and interrupt
    /// delivery update it; user-mode POPF/IRET obey the configured IOPL. Stored
    /// after DF to retain the established JIT ABI offsets.
    pub if_: bool,
}

/// The `pf_src` byte that encodes PF=0 — any odd-parity byte does, `1` is the
/// canonical one. PF=1 is `PF_SRC_ONE` (0, even parity).
pub const PF_SRC_ZERO: u8 = 1;
/// The `pf_src` byte that encodes PF=1.
pub const PF_SRC_ONE: u8 = 0;
/// The `af_src` byte that encodes AF=1 (bit 4 set). AF=0 is any byte without it.
pub const AF_SRC_ONE: u8 = 0x10;
/// The `af_src` byte that encodes AF=0.
pub const AF_SRC_ZERO: u8 = 0;

/// Two states are equal when their ARCHITECTURAL flags match. PF and AF are stored
/// as sources, and the same flag value has many source encodings (any odd-parity
/// byte clears PF), so a derived comparison is the only correct one — a byte-wise
/// derive would report two identical machines as different.
impl PartialEq for Flags {
    fn eq(&self, o: &Self) -> bool {
        self.cf == o.cf
            && self.zf == o.zf
            && self.sf == o.sf
            && self.of == o.of
            && self.df == o.df
            && self.if_ == o.if_
            && self.pf() == o.pf()
            && self.af() == o.af()
    }
}

/// Prints the architectural flags, not the sources — a failing state comparison has
/// to be readable as flags.
impl fmt::Debug for Flags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Flags")
            .field("cf", &self.cf)
            .field("pf", &self.pf())
            .field("af", &self.af())
            .field("zf", &self.zf)
            .field("sf", &self.sf)
            .field("of", &self.of)
            .field("df", &self.df)
            .field("if_", &self.if_)
            .finish()
    }
}

/// All flags clear — `pf_src` must therefore be an ODD-parity byte, which is why
/// this cannot be `#[derive(Default)]`.
impl Default for Flags {
    fn default() -> Self {
        Flags {
            cf: false,
            pf_src: PF_SRC_ZERO,
            af_src: AF_SRC_ZERO,
            zf: false,
            sf: false,
            of: false,
            df: false,
            if_: false,
        }
    }
}

impl Flags {
    /// PF — even parity of the low byte of the last result.
    pub fn pf(&self) -> bool {
        self.pf_src.count_ones().is_multiple_of(2)
    }

    /// Set PF to a known value (`popf`, the BCD instructions, forced-flag sites).
    pub fn set_pf(&mut self, v: bool) {
        self.pf_src = if v { PF_SRC_ONE } else { PF_SRC_ZERO };
    }

    /// AF — bit 4 of `a ^ b ^ result`.
    pub fn af(&self) -> bool {
        self.af_src & 0x10 != 0
    }

    /// Set AF to a known value.
    pub fn set_af(&mut self, v: bool) {
        self.af_src = if v { AF_SRC_ONE } else { AF_SRC_ZERO };
    }

    /// Assemble the architectural RFLAGS word from the modeled flag fields
    /// (the low bits + the always-set reserved bit 1). This arithmetic/direction
    /// image carries CF(0), reserved(1), PF(2), AF(4), ZF(6), SF(7), DF(10), OF(11).
    /// `CpuState::rflags` adds the system flags for PUSHF and SYSCALL; the JIT's
    /// `assemble_rflags` reproduces this arithmetic image.
    pub fn to_rflags(&self) -> u64 {
        (self.cf as u64)
            | (1 << 1) // reserved bit, always 1
            | ((self.pf() as u64) << 2)
            | ((self.af() as u64) << 4)
            | ((self.zf as u64) << 6)
            | ((self.sf as u64) << 7)
            | ((self.df as u64) << 10)
            | ((self.of as u64) << 11)
    }

    /// The 16-bit real-mode FLAGS image (§17.6) — the value `pushf`/`int`/exception
    /// delivery pushes and `popf`/`iret` restore. Same low bits as [`to_rflags`], plus
    /// **IF (bit 9)**; the reserved bit 1 is always set, bits 12-15 (IOPL/NT) are 0.
    /// TF (bit 8) is not modeled, so it reads back 0 (see the `Flags` docs). Returned as
    /// a `u16` since real-mode frames are 2 bytes wide.
    ///
    /// Protected modes use `CpuState::rflags`, which also carries system flags.
    pub fn to_flags16(&self) -> u16 {
        (self.cf as u16)
            | (1 << 1) // reserved bit, always 1
            | ((self.pf() as u16) << 2)
            | ((self.af() as u16) << 4)
            | ((self.zf as u16) << 6)
            | ((self.sf as u16) << 7)
            | ((self.if_ as u16) << 9)
            | ((self.df as u16) << 10)
            | ((self.of as u16) << 11)
    }

    /// Restore the modeled flags from a real-mode FLAGS image (`popf`/`iret`, §17.6).
    /// The inverse of [`to_flags16`]: CF/PF/AF/ZF/SF/DF/OF **and IF** are restored; TF
    /// (bit 8) and the system bits (IOPL/NT/…) are not modeled and ignored.
    pub fn set_flags16(&mut self, image: u16) {
        self.cf = image & (1 << 0) != 0;
        self.set_pf(image & (1 << 2) != 0);
        self.set_af(image & (1 << 4) != 0);
        self.zf = image & (1 << 6) != 0;
        self.sf = image & (1 << 7) != 0;
        self.if_ = image & (1 << 9) != 0;
        self.df = image & (1 << 10) != 0;
        self.of = image & (1 << 11) != 0;
    }
}

/// Flat, hot-path guest register file.
///
/// `#[repr(C)]` keeps field offsets stable so the JIT can read/write fields at
/// known offsets (§8.2.1). GPRs are indexed by x86 encoding order
/// (RAX=0, RCX=1, ...), NOT the enum's declaration order (§3.1).
#[repr(C)]
#[derive(Clone, Debug)]
pub struct CpuState {
    pub gpr: [u64; 16],
    pub rip: u64,
    pub fs_base: u64,
    pub gs_base: u64,
    pub flags: Flags,
    /// SSE/AVX/AVX-512 vector registers — the low 128 bits (XMM) (§3.1, M8).
    /// `#[repr(C)]` + 16-byte-aligned `u128` so the JIT loads/stores at stable
    /// offsets. 32 registers: EVEX (AVX-512) can address XMM/YMM/ZMM 16–31.
    pub xmm: [u128; 32],
    /// Bits 255:128 of each vector register. Legacy SSE writes leave
    /// this untouched; a VEX.128 write zeroes it; a VEX.256 write sets it;
    /// `vzeroupper` clears bits 255:128 of ZMM0–15.
    pub ymm_hi: [u128; 32],
    /// Bits 511:256 of each ZMM register (AVX-512). `zmm_hi[i][0]` is
    /// bits 383:256, `zmm_hi[i][1]` bits 511:384. A ZMM-width EVEX write sets these;
    /// a shorter (128/256) write zeroes them.
    pub zmm_hi: [[u128; 2]; 32],
    /// AVX-512 opmask registers k0–k7. `k0` reads as all-ones when used
    /// as a write-mask (i.e. "no masking"); it is still a real, writable register.
    pub kmask: [u64; 8],
    /// x87 FPU register file (§14) as RAW 80-bit values. `ST(i)` = `fpr[(fpu_top + i) & 7]`.
    /// Raw bytes are the source of truth (not a decoded `F80`) so MMX — which aliases the
    /// low 64 bits of the *physical* registers, `MM(i)` = `fpr[i][0..8]` — can
    /// round-trip arbitrary payloads (an `F80` decode/encode would corrupt NaN/MMX bit
    /// patterns). x87 ops decode to [`crate::f80::F80`] for arithmetic and re-encode; that
    /// round-trip is exact for the normal floats x87 produces. `fpu_top` is the stack top,
    /// `fpu_cw` the control word (round-trips through `fldcw`/`fnstcw`).
    pub fpr: [[u8; 10]; 8],
    pub fpu_top: u32,
    pub fpu_cw: u16,
    pub fpu_pad: u16,
    /// An MMIO-read value delivered by `Vcpu::complete_mmio_read`, consumed by the
    /// retried load when the block re-executes from the faulting instruction (§5.2).
    /// On `CpuState` (not `Vcpu`) so the interpreter's `Load` can see it.
    pub pending_mmio: Option<u64>,
    /// An MMIO-write acknowledgement from `Vcpu::complete_mmio_write`: the embedder
    /// performed the write's side effect, so the retried store consumes this and
    /// continues instead of re-trapping (the write counterpart of `pending_mmio`).
    pub pending_mmio_write: bool,
    /// Width (1/2/4) of a port input awaiting an embedder answer.
    /// `Vcpu::complete_port_in` writes IN's accumulator using sub-register
    /// semantics, or retains INS's answer in `port_string_in` for its store.
    /// `None` after the answer is supplied; an INS store can still be pending.
    pub pending_port_in: Option<u8>,
    /// Guest CPU feature set the embedder selected. Read by `cpuid_run` and
    /// the `xgetbv` handler when no table / XCR0 override is installed. Kept out of
    /// `jit_abi::CpuOffsets` — the JIT never field-loads it (only the cpuid/xgetbv
    /// helpers read it via Rust), so it needs no stable ABI offset.
    pub features: crate::features::GuestCpuFeatures,
    /// Precision of the x87 transcendentals. `Fast` (default) uses the f64
    /// libm path; `Extended` uses the full-80-bit F80 series. Read only by `exec_x87`.
    pub x87_precision: X87Precision,
    /// Real-mode segment selectors (§17.6). Appended at the very END of the struct so
    /// every pre-existing `#[repr(C)]` field offset stays byte-identical — the JIT ABI
    /// (`jit_abi::CpuOffsets`) and its offset-assertion tests are untouched. In real
    /// mode a segment's base is `selector << 4`; the interpreter reads these to segment
    /// data (`with_segment`) and stack (push/pop/call/ret) accesses. Protected modes
    /// expose these configured selectors through MOV and validate same-selector
    /// IRET against CS/SS; descriptor tables are not modeled.
    pub cs: u16,
    pub ds: u16,
    pub es: u16,
    pub ss: u16,
    /// Bit `i` set = physical x87 register `R(i)` is **empty** (tag `11`, SDM Vol 1
    /// §8.1.7). Appended at the END like the real-mode selectors above, so every
    /// pre-existing `#[repr(C)]` offset — and the JIT ABI built on them — is untouched.
    ///
    /// Without it the tag word could report `00`/`01`/`10` but never `11`, so
    /// `fninit; fnstenv` said "all eight registers hold zero" where hardware says "all
    /// eight are empty". A guest reads that field to find out how many slots are
    /// occupied, which is the one thing it could not learn.
    pub fpu_empty: u8,
    /// Status-word bits **outside** TOP: the six exception flags (5:0), SF (6), ES (7),
    /// C0/C1/C2 (10:8), C3 (14) and B (15). TOP (13:11) is derived from `fpu_top`, which
    /// stays the single source of truth for the stack pointer.
    ///
    /// The engine raises no FP exception and computes no condition code, so nothing here
    /// is ever *set* by execution — it exists so an environment image survives
    /// `fnstenv`/`fldenv` unchanged, which is what a `fenv_t` save/restore needs.
    pub fpu_sw: u16,
    /// x87 instruction/data offsets and opcode (SDM Vol 1 §§8.1.8–8.1.9).
    pub fpu_fip: u64,
    pub fpu_fdp: u64,
    pub fpu_fop: u16,
    /// Selectors are retained: the default CPUID does not deprecate FCS/FDS.
    pub fpu_fcs: u16,
    pub fpu_fds: u16,
    /// Sub-transfers of the currently-trapping instruction that the embedder has already
    /// answered, as `(guest address, value)`. Appended at the END like the
    /// fields above, so every pre-existing `#[repr(C)]` offset and the JIT ABI built on
    /// them stay byte-identical. Not in `jit_abi::CpuOffsets` — no compiled block reads
    /// it; the JIT defers MMIO to the interpreter (`RET_MMIO_DEFER`), so fixing the
    /// interpreter fixes both tiers.
    ///
    /// **Why a multi-access instruction needs this at all.** A 16-byte vector access is
    /// two 8-byte transfers, while `Exit::MmioRead`'s answer channel carries ONE value.
    /// The retry re-executes the whole instruction, so consuming a single
    /// `pending_mmio` per attempt cannot converge: answer the second half, re-enter, and
    /// the FIRST half traps again with nothing pending — forever. Measured before the
    /// fix: four rounds of `run` → `complete_mmio_read` produced four identical exits.
    ///
    /// Keyed by address rather than by transfer index so it does not depend on the
    /// handlers visiting halves in a particular order. The maximum is 64 transfers:
    /// masked vector operands visit at most 64 / elem elements (elem >= 1), narrowing
    /// stores visit at most 64 / 2 elements, and unmasked operands use at most eight
    /// 8-byte transfers.
    /// The fixed array keeps MMIO retry storage allocation-free and costs 1024 bytes, 896 more
    /// than an eight-entry array.
    /// The interpreter clears these at instruction completion, including ordinary
    /// retries after a RAM fault. The dispatcher keeps partially serviced instructions
    /// on that path and clears abandoned retries at exits; explicit RIP writes also
    /// clear the entries and pending completion.
    pub mmio_parts: [(u64, u64); 64],
    pub mmio_parts_len: u8,
    /// Guest RIP owning the retry, including a pending completion with no parts yet.
    pub mmio_parts_rip: u64,
    /// A retry owns either an outstanding MMIO transfer, a pending completion, or parts.
    pub mmio_retry_active: bool,
    /// Code segment owning the retry; a state replacement may redirect code at the same IP.
    pub mmio_retry_cs: u16,
    /// Complete CPUID description, shared immutably by vCPUs and VM forks.
    /// Appended so existing JIT ABI field offsets do not move.
    pub cpuid_table: Option<std::sync::Arc<crate::features::CpuidTable>>,
    pub cpuid_overrides: crate::features::CpuidOverrides,
    /// Source inherited at vCPU creation; may be replaced between runs.
    pub cpu_source: Option<std::sync::Arc<dyn CpuSource>>,
    /// Engine-allocated identity for the source, independent of guest CPUID IDs.
    /// State clones retain this identity; embedders may replace it between runs.
    pub source_id: u64,
    /// Deterministic fallback stream, copied by CPU-state cloning.
    pub random_state: u64,
    /// Protected-mode RFLAGS bits outside `Flags`: TF, IOPL, NT, RF, VM, AC,
    /// VIF, VIP and ID (SDM Vol 1 §3.4.3). The embedder configures IOPL and
    /// `flags.if_`; execution uses CPL 3. TF is stored without single-step delivery.
    /// Appended to preserve every existing JIT ABI offset.
    pub rflags_system: u64,
    /// Visible FS/GS selectors, independent of the configured linear bases.
    pub fs: u16,
    pub gs: u16,
    /// Guest SSE control/status image (SDM Vol 1 §10.2.3). Stored and round-tripped;
    /// arithmetic applies RC/FTZ/DAZ but does not set sticky exception flags.
    /// This is separate from host FPCR/MXCSR saved by the fault landing stub.
    /// Appended to preserve every existing JIT ABI offset.
    pub mxcsr: u32,
    /// Pointer/opcode model, inherited from the VM and editable between runs.
    /// Appended to preserve every existing JIT ABI offset.
    pub x87_pointer_model: X87PointerModel,
    /// An INS handoff, including an answered value awaiting its faultable store.
    /// Appended to preserve existing JIT ABI offsets.
    #[doc(hidden)]
    pub port_string_in: Option<PortStringIn>,
}

/// Selects x87 opcode recording and pointer/opcode saving independently of CPUID.
/// Environment loading is unchanged. FNSTENV saves the recorded environment in
/// both models (AMD64 APM Vol 5 FSTENV/FNSTENV).
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
#[repr(u8)]
pub enum X87PointerModel {
    /// Record FOP on unmasked exceptions (SDM Vol 1 §8.1.9.1, without FOP
    /// compatibility mode); always save recorded pointers/opcode.
    #[default]
    Sdm,
    /// Record FOP for every non-control instruction (AMD64 APM Vol 1 §6.2.5.2).
    /// FXSAVE and direct `exec_fxstate` export save pointers/opcode only with
    /// FSW.ES set, otherwise zeros (AMD64 APM Vol 2 §11.4.4.3; Vol 5 FXSAVE).
    Amd,
}

/// Captured at handoff so a store retry never repeats a device read.
#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct PortStringIn {
    pub rip: u64,
    pub next_ip: u64,
    pub instruction_end: u64,
    pub base: u64,
    pub offset: u64,
    pub next_index: u64,
    pub next_count: u64,
    pub addr_bits: u8,
    pub elem: u8,
    pub rep: bool,
    pub value: Option<u64>,
}

/// One atomic TSC/TSC_AUX sample (SDM Vol 3B §20.17.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TscSample {
    pub counter: u64,
    pub aux: u32,
}

/// Which random instruction requested a sample (SDM Vol 2B RDRAND/RDSEED).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RandomKind {
    Rdrand,
    Rdseed,
}

/// Embedder-owned time and entropy, shared by a VM's vCPUs (§14).
///
/// The engine makes exactly one method call per executed instruction, never at
/// lift/compile time. `vcpu_id` is `CpuState::source_id`; `size` is 2, 4 or 8 bytes.
/// `None` means random data is unavailable, not a request for a fallback sample.
/// Methods must not panic or reenter execution: JIT calls cross an extern-C boundary.
/// The source owns rate, offset, cross-vCPU monotonicity, entropy quality and replay.
/// Its TSC behavior must agree with any installed CPUID frequency/invariant fields
/// (SDM Vol 3B §20.17/§20.17.1). Cloning CPU state or forking a VM shares this
/// handle; it does not snapshot the source's internal state.
///
/// For RDTSCP, the engine retires all earlier guest instructions and orders their
/// memory accesses with a host fence before calling `read_tsc`. The source must
/// itself use an execution-synchronising hardware-counter read to prevent sampling
/// ahead of earlier guest loads: on x86-64, RDTSCP or LFENCE;RDTSC (SDM Vol 2B
/// RDTSCP/RDTSC, Description); on AArch64, ISB before CNTVCT_EL0, or the
/// self-synchronising CNTVCTSS_EL0 with FEAT_ECV (see the Arm Architecture Reference
/// Manual, generic timer). The memory fence alone does not synchronise counter
/// execution. RDTSC has no ordering requirement (SDM Vol 2B RDTSC, Description).
/// Both instructions use this same method, so a hardware-counter source must
/// provide the RDTSCP ordering on every call.
pub trait CpuSource: Send + Sync + std::fmt::Debug {
    fn read_tsc(&self, vcpu_id: u64) -> TscSample;
    fn read_random(&self, vcpu_id: u64, kind: RandomKind, size: u8) -> Option<u64>;
}

/// Precision selection for the x87 transcendentals (fsin/fcos/…/fyl2x). The
/// speed/accuracy trade-off the embedder chooses per run (like [`GuestCpuFeatures`]).
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum X87Precision {
    /// f64/libm precision (~53-bit) — fast. The default.
    #[default]
    Fast,
    /// Full-80-bit F80 precision (range reduction + Taylor series) — slower, ~80-bit
    /// accurate (closer to a physical x87 FPU).
    Extended,
}

/// A reset x87/SSE CPU: MXCSR is `0x1f80` (SDM Vol 1 §11.6.4, Table 11-2).
/// The x87 control word is post-`finit` `0x037F` — round to nearest even, 64-bit significand, all six exceptions
/// masked (SDM Vol 2A FINIT/FNINIT).
///
/// `Default` is written out field by field rather than derived for two reasons: a derived
/// zero control word means
/// **precision control 00, i.e. 24-bit single precision** (SDM Vol 1 §8.1.5.2, Table
/// 8-2). That was harmless only while the control word reached nothing: the moment
/// arithmetic started consulting it, every guest that never ran `fldcw`
/// computed at single precision. busybox `awk`'s float `printf` caught it in the
/// real-program ladder, which is the one place a wrong default like this shows. And
/// spelling the fields out means adding one to `CpuState` is a compile error here rather
/// than a silent zero.
impl Default for CpuState {
    fn default() -> Self {
        CpuState {
            gpr: [0; 16],
            rip: 0,
            fs_base: 0,
            gs_base: 0,
            flags: Flags::default(),
            xmm: [0; 32],
            ymm_hi: [0; 32],
            zmm_hi: [[0; 2]; 32],
            kmask: [0; 8],
            fpr: [[0; 10]; 8],
            fpu_top: 0,
            fpu_cw: crate::f80::Ctl::RESET.0,
            fpu_pad: 0,
            pending_mmio: None,
            pending_mmio_write: false,
            pending_port_in: None,
            features: crate::features::GuestCpuFeatures::default(),
            x87_precision: X87Precision::default(),
            cs: 0,
            ds: 0,
            es: 0,
            ss: 0,
            // `finit` leaves every register tagged empty (SDM Vol 2A FINIT/FNINIT:
            // "The data registers ... are all tagged as empty").
            fpu_empty: 0xFF,
            fpu_sw: 0,
            fpu_fip: 0,
            fpu_fdp: 0,
            fpu_fop: 0,
            fpu_fcs: 0,
            fpu_fds: 0,
            mmio_parts: [(0, 0); 64],
            mmio_parts_len: 0,
            mmio_parts_rip: 0,
            mmio_retry_active: false,
            mmio_retry_cs: 0,
            cpuid_table: None,
            cpuid_overrides: crate::features::CpuidOverrides::default(),
            cpu_source: None,
            source_id: 0,
            random_state: 0x4d59_5df4_d0f3_3173,
            rflags_system: 0,
            fs: 0,
            gs: 0,
            mxcsr: MXCSR_RESET,
            x87_pointer_model: X87PointerModel::default(),
            port_string_in: None,
        }
    }
}

/// SDM Vol 1 §11.6.4, Table 11-2: SIMD control/status after power-up or reset.
pub const MXCSR_RESET: u32 = 0x1f80;
/// Supported MXCSR bits, reported by FXSAVE (SDM Vol 1 §11.6.6).
/// DAZ is supported as stored state; bits 31:16 are reserved (§10.2.3).
pub const MXCSR_MASK: u32 = 0xffff;

/// SDM Vol 1 §3.4.3: defined system bits carried outside the arithmetic flags.
pub const RFLAGS_SYSTEM_MASK: u64 = 0x3f_7100;
/// SDM Vol 2B §4.3 POPF, Table 4-20; Vol 2A §3.2 IRET, Operation:
/// CPL 3 may write TF/NT, and wide forms also AC/ID (IRET also RF).
pub const fn user_rflags_mask(size: u8, iret: bool) -> u64 {
    0x4100
        | if size > 2 {
            0x24_0000 | if iret { 0x1_0000 } else { 0 }
        } else {
            0
        }
}

impl CpuState {
    /// Full configured user-mode image (SDM Vol 1 §3.4.3).
    pub fn rflags(&self) -> u64 {
        self.flags.to_rflags()
            | ((self.flags.if_ as u64) << 9)
            | (self.rflags_system & RFLAGS_SYSTEM_MASK)
    }

    /// CPL 3 POPF/IRET policy; privilege bits come from the old state.
    /// SDM Vol 2B §4.3 POPF Table 4-20; Vol 2A §3.2 IRET, Operation.
    pub fn set_user_rflags(&mut self, image: u64, size: u8, iret: bool) {
        let old_if = self.flags.if_;
        let can_set_if = self.rflags_system & 0x3000 == 0x3000;
        self.flags.set_flags16(image as u16);
        if !can_set_if {
            self.flags.if_ = old_if;
        }
        let mask = user_rflags_mask(size, iret);
        self.rflags_system = (self.rflags_system & !mask) | (image & mask);
        if !iret {
            self.rflags_system &= !(1 << 16);
        }
    }

    /// Selector encoding follows the ModRM Sreg field (SDM Vol 2B §4.3 MOV).
    pub fn selector(&self, segment: u8) -> u16 {
        match segment {
            0 => self.es,
            1 => self.cs,
            2 => self.ss,
            3 => self.ds,
            4 => self.fs,
            5 => self.gs,
            _ => unreachable!("segment encoding"),
        }
    }
    pub fn new() -> Self {
        Self::default()
    }

    /// The effective CPUID source used by both backends and coverage tools.
    // Keep CPUID resolution out of the large interpreter dispatch loop: inlining
    // it changes register allocation even for instruction arms that never use it.
    #[inline(never)]
    pub fn cpuid(&self, leaf: u32, subleaf: u32) -> crate::features::CpuidRegisters {
        let (leaf, subleaf, regs) = match &self.cpuid_table {
            Some(table) => table.resolve(leaf, subleaf),
            None => (
                leaf,
                if matches!(leaf, 7 | 0xd) { subleaf } else { 0 },
                self.features.cpuid(leaf, subleaf),
            ),
        };
        let mut regs = regs;
        if self.cpuid_table.is_none()
            && leaf == 0xd
            && subleaf <= 1
            && self.features.has(crate::Feature::Xsave)
        {
            regs.ebx = crate::x87::xstate::area_size(
                crate::x87::xstate::xcr0(self) & self.features.xcr0(),
            );
        }
        self.cpuid_overrides.apply(leaf, subleaf, regs)
    }

    /// A sub-transfer of the instruction at `rip` that the embedder has already
    /// answered, or `None`. Entries for any other RIP are ignored.
    pub fn mmio_part(&self, rip: u64, addr: u64) -> Option<u64> {
        if self.mmio_parts_rip != rip {
            return None;
        }
        self.mmio_parts[..self.mmio_parts_len as usize]
            .iter()
            .find(|&&(a, _)| a == addr)
            .map(|&(_, v)| v)
    }

    /// Record an answered sub-transfer for the instruction at `rip`. Switching RIP
    /// discards the previous instruction's entries. Completion and abandonment also
    /// clear them explicitly, since the next execution can have the same RIP.
    ///
    /// Lifted vector instructions produce at most 64 transfers. Exceeding that bound
    /// asserts in debug builds; release builds drop the excess entry, so a retry asks
    /// again instead of reusing another transfer's value.
    pub fn record_mmio_part(&mut self, rip: u64, addr: u64, value: u64) {
        self.begin_mmio_retry(rip);
        if self.mmio_part(rip, addr).is_some() {
            return;
        }
        debug_assert!(
            (self.mmio_parts_len as usize) < self.mmio_parts.len(),
            "vector instruction exceeded the MMIO transfer bound"
        );
        if let Some(slot) = self.mmio_parts.get_mut(self.mmio_parts_len as usize) {
            *slot = (addr, value);
            self.mmio_parts_len += 1;
        }
    }

    /// Drop recorded transfers and the pending completion when an instruction finishes
    /// or is abandoned, including a completion made unnecessary by a changed mask.
    pub fn clear_mmio_parts(&mut self) {
        #[cfg(test)]
        MMIO_CLEAR_CALLS.with(|calls| calls.set(calls.get() + 1));
        if !self.mmio_retry_active {
            return;
        }
        self.pending_mmio = None;
        self.pending_mmio_write = false;
        self.mmio_parts_len = 0;
        self.mmio_parts_rip = 0;
        self.mmio_retry_active = false;
        self.mmio_retry_cs = 0;
    }

    pub(crate) fn owns_mmio_retry(&self, rip: u64) -> bool {
        self.mmio_parts_rip == rip && self.mmio_retry_cs == self.cs
    }

    pub(crate) fn begin_mmio_retry(&mut self, rip: u64) {
        if self.mmio_retry_active && !self.owns_mmio_retry(rip) {
            self.clear_mmio_parts();
        }
        self.mmio_retry_active = true;
        self.mmio_parts_rip = rip;
        self.mmio_retry_cs = self.cs;
    }

    /// Common retry/abandonment policy for interpretation, stepping and dispatch exits.
    pub(crate) fn finish_mmio_retry(&mut self, exit: Option<&crate::exit::Exit>) {
        use crate::exit::Exit;
        let retry = matches!(
            exit,
            Some(
                Exit::MmioRead { .. }
                    | Exit::MmioWrite { .. }
                    | Exit::UnmappedMemory { .. }
                    | Exit::BudgetExhausted
            )
        );
        if self.mmio_retry_active && (!retry || !self.owns_mmio_retry(self.rip)) {
            self.clear_mmio_parts();
        }
        if matches!(exit, Some(Exit::MmioRead { .. } | Exit::MmioWrite { .. })) {
            self.begin_mmio_retry(self.rip);
        }
    }

    /// The full 512-bit value of vector register `reg` as four 128-bit lanes, low→high:
    /// `[xmm, ymm_hi, zmm_hi.0, zmm_hi.1]`. One place to gather the scattered lane
    /// arrays instead of indexing all four inline.
    #[inline]
    pub fn vec_lanes(&self, reg: usize) -> [u128; 4] {
        [
            self.xmm[reg],
            self.ymm_hi[reg],
            self.zmm_hi[reg][0],
            self.zmm_hi[reg][1],
        ]
    }

    /// Write the low `bytes` (16/32/64) of vector register `reg` from lanes `v`,
    /// zeroing everything above `bytes` — the unmasked EVEX write rule (a 128/256-bit
    /// write clears the upper bits). Centralizes the zero-upper logic the 512-bit
    /// handlers would otherwise open-code.
    #[inline]
    pub fn set_vec(&mut self, reg: usize, v: [u128; 4], bytes: u16) {
        self.xmm[reg] = v[0];
        self.ymm_hi[reg] = if bytes >= 32 { v[1] } else { 0 };
        self.zmm_hi[reg] = if bytes >= 64 { [v[2], v[3]] } else { [0, 0] };
    }

    /// Write the low `bytes` (16/32) of vector register `reg` from lanes `v`,
    /// **preserving** everything above `bytes`. The legacy-SSE / VEX.128 write rule for
    /// per-128-bit-lane ops that share one IR op across widths: the SSE form must keep the
    /// upper YMM bits, and the VEX.128 form clears them via a separate trailing `VZeroUpper`.
    /// (Contrast [`set_vec`](Self::set_vec), which zeroes above `bytes`.)
    #[inline]
    pub fn set_vec_low(&mut self, reg: usize, v: [u128; 4], bytes: u16) {
        self.xmm[reg] = v[0];
        if bytes >= 32 {
            self.ymm_hi[reg] = v[1];
        }
        if bytes >= 64 {
            self.zmm_hi[reg] = [v[2], v[3]];
        }
    }

    /// AVX-512 write-masking: commit `newval` into vector
    /// register `reg` under opmask `k` at `elem`-byte (1/2/4/8) granularity across the
    /// low `bytes` (16/32/64). For each lane `i`: `dst[i] = k[i] ? newval[i] :
    /// (zeroing ? 0 : dst[i])`. The single place the merge/zero rule lives — a maskable
    /// op computes its full unmasked `newval` and routes the write here instead of
    /// [`set_vec`](Self::set_vec) (which is the unmasked/k0 fast path).
    pub fn write_masked(
        &mut self,
        reg: usize,
        newval: [u128; 4],
        k: u8,
        elem: u8,
        zeroing: bool,
        bytes: u16,
    ) {
        let kmask = self.kmask[k as usize];
        let old = if zeroing {
            [0u128; 4]
        } else {
            self.vec_lanes(reg)
        };
        let bits = elem as u32 * 8;
        let lane_mask: u128 = if bits >= 128 {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        };
        let per_chunk = 128 / bits; // elements in one 128-bit lane
        let chunks = (bytes as u32 / 16) as usize;
        let mut result = [0u128; 4];
        let mut e = 0u32; // running element index across the whole register
        for (chunk, slot) in result.iter_mut().enumerate() {
            if chunk >= chunks {
                break; // above `bytes` stays zero (set_vec re-asserts this)
            }
            let mut acc = 0u128;
            for j in 0..per_chunk {
                let sh = j * bits;
                let src = if (kmask >> e) & 1 != 0 {
                    newval[chunk]
                } else {
                    old[chunk]
                };
                acc |= ((src >> sh) & lane_mask) << sh;
                e += 1;
            }
            *slot = acc;
        }
        self.set_vec(reg, result, bytes);
    }

    /// Write a general-purpose register with x86 sub-register semantics (§7.1, §16
    /// — the #1 silent porting bug). `size` is the destination operand width in
    /// bytes; `index` is the `gpr[]` slot (see [`Reg::gpr_index`]).
    ///
    /// - 8: full write.
    /// - 4: write low 32 bits and **zero** bits 32–63 (`mov eax` clears the rest of rax).
    /// - 2 / 1: write low 16 / 8 bits and **preserve** the upper bits (`mov ax`/`al`).
    ///
    /// This asymmetry (4 zero-extends, 1/2 merge) is exactly the trap. High-byte
    /// registers (AH/BH/CH/DH, which write bits 8–15) are NOT expressible here —
    /// the lift rejects them rather than mis-lowering to the low byte.
    pub fn write_gpr(&mut self, index: usize, val: u64, size: u8) {
        let cur = self.gpr[index];
        self.gpr[index] = match size {
            8 => val,
            4 => val & 0xFFFF_FFFF,
            2 => (cur & !0xFFFF) | (val & 0xFFFF),
            1 => (cur & !0xFF) | (val & 0xFF),
            _ => unreachable!("gpr write size must be 1/2/4/8, got {size}"),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reg_gpr_index_is_encoding_order() {
        assert_eq!(Reg::Rax.gpr_index(), Some(0));
        assert_eq!(Reg::Rcx.gpr_index(), Some(1));
        assert_eq!(Reg::Rdx.gpr_index(), Some(2));
        assert_eq!(Reg::Rbx.gpr_index(), Some(3));
        assert_eq!(Reg::Rsp.gpr_index(), Some(4));
        assert_eq!(Reg::Rbp.gpr_index(), Some(5));
        assert_eq!(Reg::Rsi.gpr_index(), Some(6));
        assert_eq!(Reg::Rdi.gpr_index(), Some(7));
        assert_eq!(Reg::R8.gpr_index(), Some(8));
        assert_eq!(Reg::R15.gpr_index(), Some(15));
    }

    #[test]
    fn write_gpr_size_semantics() {
        let mut c = CpuState::new();
        c.gpr[0] = 0x1111_2222_3333_4444;
        // 4-byte write zeroes the upper 32 bits.
        c.write_gpr(0, 0xAABB_CCDD_EEFF_0011, 4);
        assert_eq!(c.gpr[0], 0x0000_0000_EEFF_0011);
        // 2-byte write preserves upper 48 bits.
        c.gpr[0] = 0x1111_2222_3333_4444;
        c.write_gpr(0, 0xFFFF_FFFF_FFFF_9999, 2);
        assert_eq!(c.gpr[0], 0x1111_2222_3333_9999);
        // 1-byte write preserves upper 56 bits.
        c.gpr[0] = 0x1111_2222_3333_4444;
        c.write_gpr(0, 0x77, 1);
        assert_eq!(c.gpr[0], 0x1111_2222_3333_4477);
        // 8-byte write replaces everything.
        c.write_gpr(0, 0xDEAD_BEEF_CAFE_BABE, 8);
        assert_eq!(c.gpr[0], 0xDEAD_BEEF_CAFE_BABE);
    }

    #[test]
    fn non_gpr_regs_have_no_index() {
        assert_eq!(Reg::Rip.gpr_index(), None);
        assert_eq!(Reg::FsBase.gpr_index(), None);
        assert_eq!(Reg::GsBase.gpr_index(), None);
    }

    #[test]
    fn iced_gpr_index_matches_encoding_order() {
        assert_eq!(iced_gpr_index(Register::RAX), Some(0));
        assert_eq!(iced_gpr_index(Register::RCX), Some(1));
        assert_eq!(iced_gpr_index(Register::RDX), Some(2));
        assert_eq!(iced_gpr_index(Register::RBX), Some(3));
        assert_eq!(iced_gpr_index(Register::RSP), Some(4));
        assert_eq!(iced_gpr_index(Register::RBP), Some(5));
        assert_eq!(iced_gpr_index(Register::RSI), Some(6));
        assert_eq!(iced_gpr_index(Register::RDI), Some(7));
        assert_eq!(iced_gpr_index(Register::R8), Some(8));
        assert_eq!(iced_gpr_index(Register::R15), Some(15));
    }

    #[test]
    fn iced_subwidth_regs_normalize_to_parent() {
        // EAX/AX/AL/AH all alias RAX -> index 0.
        assert_eq!(iced_gpr_index(Register::EAX), Some(0));
        assert_eq!(iced_gpr_index(Register::AX), Some(0));
        assert_eq!(iced_gpr_index(Register::AL), Some(0));
        assert_eq!(iced_gpr_index(Register::AH), Some(0));
        // R8D/R8W/R8L all alias R8 -> index 8.
        assert_eq!(iced_gpr_index(Register::R8D), Some(8));
        assert_eq!(iced_gpr_index(Register::R8W), Some(8));
        assert_eq!(iced_gpr_index(Register::R8L), Some(8));
        // SPL is RSP's byte -> index 4.
        assert_eq!(iced_gpr_index(Register::SPL), Some(4));
    }

    #[test]
    fn iced_non_gpr_regs_have_no_index() {
        assert_eq!(iced_gpr_index(Register::RIP), None);
        assert_eq!(iced_gpr_index(Register::FS), None);
        assert_eq!(iced_gpr_index(Register::XMM0), None);
        assert_eq!(iced_gpr_index(Register::None), None);
    }
}
