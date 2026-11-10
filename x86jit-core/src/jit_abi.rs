//! Compiled-block ABI (§8.2.1–8.2.2) — the contract shared by the interpreter's
//! dispatcher (which runs compiled blocks) and the Cranelift backend (which emits
//! them). It lives in the core because `execute()` must run a `CachedBlock`
//! without naming the JIT crate (the dependency points the other way, §4.1).
//!
//! A compiled block has signature `fn(*mut CpuState, *mut MemCtx) -> u64`:
//! - reads/writes guest registers as fields of `*CpuState` at stable
//!   `#[repr(C)]` offsets ([`CpuOffsets`]);
//! - inlines RAM access as `MemCtx.base + guest_addr` after a bounds check;
//! - returns an encoded [`StepResult`]/[`Exit`] (`0` = Continue, see the `RET_*`
//!   codes); fault details land in `MemCtx` out-fields.

use crate::cache::CompiledPtr;
use crate::exit::{AccessKind, Exit, StepResult};
use crate::memory::Memory;
use crate::state::CpuState;

pub mod fault;
pub mod fp;

/// Every compiled block shares this signature so the dispatcher jumps in
/// uniformly (§8.2.1).
pub type CompiledFn = unsafe extern "C" fn(cpu: *mut u8, mem: *mut u8) -> u64;

// --- return codes (§8.2.2). RIP is always written by the block. ---
pub const RET_CONTINUE: u64 = 0;
/// Syscall exit; [`MemCtx::next_entry`] contains the guest instruction RIP.
pub const RET_SYSCALL: u64 = 1;
pub const RET_HLT: u64 = 2;
pub const RET_UNMAPPED: u64 = 3;
/// Block chaining (§12 M5): the block resolved its direct successor through a
/// filled link slot; `MemCtx.next_entry` holds it — the dispatcher jumps straight
/// there, skipping the cache lookup.
pub const RET_CHAIN: u64 = 4;
/// A direct edge whose link slot is still empty. `MemCtx.link_slot` holds the
/// slot address; the dispatcher compiles RIP's block and fills the slot so the
/// edge chains next time.
pub const RET_LINK: u64 = 5;
/// A guest CPU exception (`#DE` div, or a lifted `ud2`/`int3`/`int1`). The block
/// set the saved RIP (on the instruction for a fault, past it for a trap) and stored
/// the vector in `MemCtx.exception_vector`; the dispatcher raises `Exit::Exception`.
pub const RET_EXCEPTION: u64 = 6;
/// Indirect-branch target cache miss (fast-dispatch R4): an indirect jmp/call whose
/// per-site IBTC slot was empty or held a descriptor for a *different* target.
/// `MemCtx.link_slot` holds the slot address; the dispatcher resolves RIP,
/// (re)fills the slot with an immutable `{target, entry}` descriptor unless the
/// site is megamorphic, and continues. A hit instead returns `RET_CHAIN`.
pub const RET_IBTC_MISS: u64 = 7;
/// An inlined load/store landed in a `Trap` (MMIO) region (§5.2, M4-T10). The
/// block set RIP to the faulting instruction and committed nothing of it; the
/// dispatcher single-steps that one instruction on the interpreter, which produces
/// the `MmioRead`/`MmioWrite` exit (and, on resume, consumes the pending value or
/// write-ack) before control returns to compiled code. Fault out-fields are set
/// like `RET_UNMAPPED`.
pub const RET_MMIO_DEFER: u64 = 8;
/// A port-I/O instruction (`in`/`out`, §5.2). Like `RET_MMIO_DEFER`, the block set
/// RIP to the port instruction and committed nothing of it; the dispatcher
/// single-steps that one instruction on the interpreter, which produces
/// `Exit::PortIo` (and, for `in`, records the pending accumulator width the embedder
/// answers via `complete_port_in`). Append-only ABI growth.
pub const RET_PORTIO_DEFER: u64 = 9;
/// A string helper stopped between iterations; the dispatcher consumes the request.
pub const RET_INTERRUPTED: u64 = 10;
/// Returned only by the signal-context landing pad; see [`fault`].
pub const RET_HOST_FAULT: u64 = 11;
/// An MXCSR write completed. Dispatch selects the successor's FP specialization;
/// ordinary transfers need no MXCSR load or comparison.
pub const RET_MXCSR_CHANGED: u64 = 12;
/// Detailed cold miss: next_entry is instruction RIP, link_slot the transfer base.
/// Original transfer width is in fault_access bits 24..31. Helper context, if
/// nonzero, supplies the operand base/width. No chain meaning on this return.
pub const RET_UNMAPPED_DETAIL: u64 = 13;
/// Detailed exception: next_entry is reporting RIP, fault_access is FaultCause's
/// private wire encoding. Legacy RET_EXCEPTION never reads these aliases.
pub const RET_EXCEPTION_DETAIL: u64 = 14;
/// Classifier initialized next_entry to exact host PC and link_slot to site intent.
/// Alignment may have no valid fault_addr; replay must precede address publication.
pub const RET_HOST_FAULT_DETAIL: u64 = 15;

/// Maximum distinct guest-access descriptors in one compiled block or region.
/// Cranelift user trap codes below 250 are reserved for these descriptors.
pub const MAX_GUEST_ACCESS_DESCRIPTORS: usize = 249;

// --- MemCtx: guest memory context + fault out-params. `#[repr(C)]`; codegen
// addresses these fields by the byte offsets below. Growth is APPEND-ONLY: a new
// field goes at the end so every offset above it — and therefore every
// previously-baked block — stays valid. ---
#[repr(C)]
pub struct MemCtx {
    /// Host base of the guest buffer (`host_base + guest_addr` for inlined access).
    pub base: u64,
    /// Guest buffer size; a guest address `>= size` traps instead of host-UB.
    pub size: u64,
    /// Out: faulting guest address (written before returning `RET_UNMAPPED`).
    pub fault_addr: u64,
    /// Out: access width in bytes.
    pub fault_size: u64,
    /// Out: bit 0 = write (zero means read), bit 1 = SS reference.
    /// Inline cold misses may encode a decoded carrier width in bits 8..23 and
    /// the original transfer width in bits 24..31 (embedded broadcast, xj-au4.2).
    pub fault_access: u64,
    /// Out: next block entry pointer on `RET_CHAIN`; instruction RIP on `RET_SYSCALL`.
    pub next_entry: u64,
    /// Out: address of the link slot to fill, set on `RET_LINK`.
    pub link_slot: u64,
    /// In/out: block budget for this call (§9.2, superblocks M5-T3). The dispatcher
    /// writes the remaining block quantum before each call; a compiled **region**
    /// decrements it once per guest block it enters and exits when it hits 0, so a
    /// multi-block region charges the exact same block count as the interpreter
    /// (preserving the `RunSpec::Blocks(n)` oracle). A single block never touches
    /// it, so `quantum - fuel == 0` and the dispatcher charges 1 as before.
    pub fuel: u64,
    /// In: pointer to this vcpu's [`RetStack`] shadow return stack (fast-dispatch R5).
    /// Compiled `call`s push `(return_addr, continuation_slot)` here; compiled
    /// `ret`s pop and, on a matching prediction, chain straight to the caller's
    /// continuation. Never null: the dispatcher points it at the vcpu's ring, and
    /// `run_compiled` at a local scratch ring.
    pub ret_stack: u64,
    /// Guest address the RAM buffer (`base`) represents at offset 0 (§4.1, identity
    /// mapping). `0` is the historical zero-based layout; non-zero means a guest
    /// address `a` maps to `base + (a - guest_base)`. The inlined RAM path bakes this
    /// as a compile-time constant (byte-identical codegen when 0); the string/x87
    /// helpers read it here to rebase their raw accesses.
    pub guest_base: u64,
    /// Out: x86 exception vector, set before returning `RET_EXCEPTION`. `#DE` (div)
    /// stores 0; a lifted `IrOp::Trap` stores its vector (`ud2`→6, `int3`→3,
    /// `int1`→1). The dispatcher reads it into `Exit::Exception { vector }`.
    pub exception_vector: u64,
    /// In: address of the live `Memory::watch_count` atomic. Generated store code loads
    /// this pointer then loads the count **through** it — live, never a run-start
    /// snapshot — to gate the watched-range dirty check. A snapshot misses the 0→nonzero
    /// transition when another thread installs the first watch mid-run (a multi-vCPU
    /// race); the live load sees it on the next store. Zero means nothing is watched, so
    /// an unwatched run pays only a pointer load + branch on the inlined store path (the
    /// atomic stays shared-clean in L1).
    pub watch_count_ptr: u64,
    /// In: raw `*const Memory` for the note-watched-write helper the JIT calls when
    /// `watch_count != 0`. Dereferenced only on that gated path; `for_memory` always
    /// sets it from a live `&Memory`.
    pub mem_self: u64,
    /// In: pointer to this vcpu's executed-instruction counter, or 0 when the
    /// accounting is off. Compiled code adds each block's `IrBlock::icount` THROUGH
    /// this pointer, the same shape as `ret_stack`/`watch_count_ptr`, so the
    /// dispatcher needs no per-run flush and `MemCtx` stays a local in `run_inner` —
    /// passing it in by `&mut` instead measured +5.4% on the dispatch-micro bench,
    /// because the compiler can no longer treat it as a local. Each compiled block
    /// adds its own count on entry — one add per block, never per instruction, so the
    /// cost is a load/add/store at a block boundary rather than accounting inside the
    /// block body (which the codegen deliberately avoids). A region adds each guest
    /// block's count at the fuel gate it already passes through, so a multi-block unit
    /// stays exact.
    ///
    /// Separate from `Vcpu::retired_instructions`, which is documented as a
    /// deterministic virtual-time base and ticks only on the interpreter: making
    /// that one jump by a whole block would change the granularity any scheduler
    /// reads it at. This counter answers "how much guest work actually ran" —
    /// guest IPC against wall time, and `executed / chained` for the average
    /// compiled-unit length.
    pub icount_ptr: u64,
    /// In: base of the per-page watch bitmap, so a generated store can test its own
    /// page's bit inline and call the note-watched helper only when it is set.
    /// Gating on the process-wide `watch_count` alone turned EVERY store out of
    /// compiled code into a call that almost always found the page unwatched, once any
    /// single page anywhere was watched — an embedder measured 388M such calls in 10 s
    /// and, later, 7.7% of a retail title's cycles in this write barrier.
    ///
    /// The table starts at the backing's first page: guest page `p` is table index
    /// `i = p - (guest_base >> CODE_PAGE_BITS)`, at word `+ (i >> 6) * 8`, bit
    /// `i & 63`, and generated code subtracts the same base page. Read live, like
    /// `watch_count_ptr`, so a mid-run watch is seen by the next store.
    /// Indexed WITHOUT a bounds check, which is sound because inlined stores are
    /// bounds-checked against `size` first and the table covers that span —
    /// `Memory::watch_bits_cover_size` asserts it at run start.
    pub watch_bits_ptr: u64,
    /// In: address of the live `Memory::code_range` watermark, so a generated store can
    /// decide with one unsigned compare whether it might have landed on a page backing
    /// translated code, and call the note-write helper only then.
    ///
    /// Without it, compiled stores consulted the SMC table through nothing at all: a
    /// guest that patched another block and called it ran the stale translation, while
    /// the interpreter running the same program observed the patch.
    ///
    /// Deliberately NOT the shape of `watch_bits_ptr`. That one can gate on
    /// `watch_count != 0` before touching its table, and that count is zero for almost
    /// every guest; code pages exist as soon as anything has executed, so the same
    /// two-stage gate would degenerate to a table probe on every store — and a table
    /// probe in the hot store stream is what got the watch bitmap's first cut reverted.
    /// A watermark keeps the hot path to a subtract and a compare that stack and heap
    /// stores fall straight through.
    ///
    /// Packed `(lo << 32) | len` over absolute 64 KiB granules (`addr >> 16`), read
    /// live like
    /// `watch_count_ptr` so a page marked by a background compile mid-run is seen by the
    /// next store. One `u64` so both halves arrive in a single load: a torn read could
    /// yield a NARROWED range, which would silently skip a page. Each marked granule
    /// also widens the range to its predecessor, so a store of at most 64 bytes is
    /// gated on its first byte alone (`Memory::widen_code_range`); the precise
    /// note-write helper still tests 4 KiB pages.
    pub code_range_ptr: u64,
    /// In: live per-vcpu dispatch notification word (SMC | exit request).
    /// Points to an immutable zero atomic for standalone compiled calls. Rust polls
    /// Relaxed, generated regions use Cranelift's SeqCst atomic_load, and only the
    /// dispatcher consumes requests (Acquire). Always non-null.
    pub dispatch_word_ptr: u64,
    /// Cold helper-only operand base. Inline generated code never touches this.
    /// Together with helper_operand_size this preserves the whole operand while
    /// fault_addr continues to name the exact failing byte. Consumed on exit.
    pub helper_operand_addr: u64,
    /// Cold helper-only whole-operand width; zero selects fault_addr/fault_size.
    /// Appended to preserve every pre-existing compiled-code field offset.
    /// Only helper miss paths set it; unmapped_exit consumes and clears both fields.
    pub helper_operand_size: u64,
}

/// Number of frames in the shadow return stack ring (R5). A power of two so the
/// index is a mask; wrap-and-overwrite on overflow — a lost frame only costs a
/// misprediction, never a wrong transfer (see [`RetStack`]).
pub const RET_STACK_LEN: usize = 64;

/// Per-vcpu shadow return stack (fast-dispatch R5): a fixed-size ring of
/// `(predicted_return_addr, continuation_slot_addr)` frames pushed on `call` and
/// popped on `ret`. `#[repr(C)]`; codegen addresses `sp` and `entries` by the byte
/// offsets below.
///
/// **Correctness does not depend on ring integrity.** A `ret` follows a prediction
/// only when the popped frame's `predicted_return_addr` equals the *actual* guest
/// return target AND the continuation slot holds the compiled entry that `resolve`
/// filled for that exact address. The ring only supplies a *candidate*; overflow
/// (wrap), underflow, stale frames after an epoch change, and cross-`run()` reuse
/// can each cause a missed prediction but never a wrong control transfer. No RSP
/// tracking is needed.
#[repr(C)]
pub struct RetStack {
    /// Monotonic push count; the live frame index is `sp & (RET_STACK_LEN - 1)`.
    pub sp: u64,
    /// Ring frames: `[predicted_return_addr, continuation_slot_addr]`.
    pub entries: [[u64; 2]; RET_STACK_LEN],
}

impl RetStack {
    pub fn new() -> Self {
        RetStack {
            sp: 0,
            entries: [[0; 2]; RET_STACK_LEN],
        }
    }
}

impl Default for RetStack {
    fn default() -> Self {
        Self::new()
    }
}

pub const MEMCTX_BASE: i32 = 0;
pub const MEMCTX_SIZE: i32 = 8;
pub const MEMCTX_FAULT_ADDR: i32 = 16;
pub const MEMCTX_FAULT_SIZE: i32 = 24;
pub const MEMCTX_FAULT_ACCESS: i32 = 32;
pub const MEMCTX_NEXT_ENTRY: i32 = 40;
pub const MEMCTX_LINK_SLOT: i32 = 48;
pub const MEMCTX_FUEL: i32 = 56;
pub const MEMCTX_RET_STACK: i32 = 64;
pub const MEMCTX_GUEST_BASE: i32 = 72;
pub const MEMCTX_EXCEPTION_VECTOR: i32 = 80;
pub const MEMCTX_WATCH_COUNT_PTR: i32 = 88;
pub const MEMCTX_MEM_SELF: i32 = 96;
pub const MEMCTX_ICOUNT_PTR: i32 = 104;
pub const MEMCTX_WATCH_BITS_PTR: i32 = 112;
pub const MEMCTX_CODE_RANGE_PTR: i32 = 120;
pub const MEMCTX_DISPATCH_WORD_PTR: i32 = 128;

// RetStack field offsets (R5): `sp` then the ring of 16-byte frames.
pub const RETSTACK_SP: i32 = 0;
pub const RETSTACK_ENTRIES: i32 = 8;
/// Byte stride between ring frames (`[u64; 2]`).
pub const RETSTACK_STRIDE: i32 = 16;

/// Byte offsets of `CpuState` fields for codegen (§8.2.1). Computed by measuring a
/// live `#[repr(C)]` value, so no unstable `offset_of!` / MSRV bump is needed —
/// the layout is a contract either way.
#[derive(Copy, Clone, Debug)]
pub struct CpuOffsets {
    pub gpr: i32,
    pub rip: i32,
    pub fs_base: i32,
    pub gs_base: i32,
    pub cf: i32,
    pub pf_src: i32,
    pub af_src: i32,
    pub zf: i32,
    pub sf: i32,
    pub of: i32,
    pub df: i32,
    pub xmm: i32,
    pub ymm_hi: i32,
    pub zmm_hi: i32,
    pub kmask: i32,
    pub mxcsr: i32,
    pub fpr: i32,
    pub fpu_top: i32,
    pub fpu_empty: i32,
    pub fpu_sw: i32,
    pub fpu_fip: i32,
    pub fpu_fdp: i32,
    pub fpu_fds: i32,
    pub fpu_fcs: i32,
    pub fpu_fop: i32,
    pub cs: i32,
}

impl CpuOffsets {
    /// GPR slot `index` (x86 encoding order) lives at `gpr + index*8`.
    pub fn gpr(&self, index: usize) -> i32 {
        self.gpr + (index as i32) * 8
    }

    /// XMM register `index` lives at `xmm + index*16`.
    pub fn xmm(&self, index: usize) -> i32 {
        self.xmm + (index as i32) * 16
    }

    /// Upper 128 bits of YMM register `index`.
    pub fn ymm_hi(&self, index: usize) -> i32 {
        self.ymm_hi + (index as i32) * 16
    }

    /// Bits 511:256 of ZMM register `index`, `half` 0 = 383:256, 1 = 511:384.
    /// Each register occupies two contiguous 16-byte slots.
    pub fn zmm_hi(&self, index: usize, half: usize) -> i32 {
        self.zmm_hi + (index as i32) * 32 + (half as i32) * 16
    }

    /// Opmask register k`index` (k0–k7) lives at `kmask + index*8`.
    pub fn kmask(&self, index: usize) -> i32 {
        self.kmask + (index as i32) * 8
    }
}

/// Measure the `#[repr(C)]` field offsets of `CpuState`.
pub fn cpu_offsets() -> CpuOffsets {
    let s = CpuState::new();
    let base = &s as *const CpuState as usize;
    let off = |p: *const u8| -> i32 { (p as usize - base) as i32 };
    CpuOffsets {
        gpr: off(s.gpr.as_ptr() as *const u8),
        rip: off(&s.rip as *const u64 as *const u8),
        fs_base: off(&s.fs_base as *const u64 as *const u8),
        gs_base: off(&s.gs_base as *const u64 as *const u8),
        cf: off(&s.flags.cf as *const bool as *const u8),
        pf_src: off(&s.flags.pf_src as *const u8),
        af_src: off(&s.flags.af_src as *const u8),
        zf: off(&s.flags.zf as *const bool as *const u8),
        sf: off(&s.flags.sf as *const bool as *const u8),
        of: off(&s.flags.of as *const bool as *const u8),
        df: off(&s.flags.df as *const bool as *const u8),
        xmm: off(s.xmm.as_ptr() as *const u8),
        ymm_hi: off(s.ymm_hi.as_ptr() as *const u8),
        zmm_hi: off(s.zmm_hi.as_ptr() as *const u8),
        kmask: off(s.kmask.as_ptr() as *const u8),
        fpr: off(s.fpr.as_ptr() as *const u8),
        fpu_top: off(&s.fpu_top as *const u32 as *const u8),
        fpu_empty: off(&s.fpu_empty as *const u8),
        fpu_sw: off(&s.fpu_sw as *const u16 as *const u8),
        fpu_fip: off(&s.fpu_fip as *const u64 as *const u8),
        fpu_fdp: off(&s.fpu_fdp as *const u64 as *const u8),
        fpu_fds: off(&s.fpu_fds as *const u16 as *const u8),
        fpu_fcs: off(&s.fpu_fcs as *const u16 as *const u8),
        fpu_fop: off(&s.fpu_fop as *const u16 as *const u8),
        cs: off(&s.cs as *const u16 as *const u8),
        mxcsr: off(&s.mxcsr as *const u32 as *const u8),
    }
}

// Standalone compiled calls have no notifier; this word is never written.
static NO_DISPATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl MemCtx {
    /// Build the guest-memory context for a run (fault/chain fields cleared).
    pub fn for_memory(mem: &Memory) -> Self {
        // Generated code indexes the watch bitmap with no bounds check, so the table
        // must cover every page a bounds-checked store can reach. Fail here, loudly and
        // at run start, rather than emit an out-of-bounds load.
        debug_assert!(
            mem.watch_bits_cover_size(),
            "watch bitmap does not cover the guest span; the inlined watch-bit test \
             would read out of bounds"
        );
        MemCtx {
            base: mem.host_base() as u64,
            size: mem.size(),
            helper_operand_addr: 0,
            helper_operand_size: 0,
            fault_addr: 0,
            fault_size: 0,
            fault_access: 0,
            next_entry: 0,
            link_slot: 0,
            fuel: u64::MAX,
            ret_stack: 0,
            guest_base: mem.guest_base(),
            exception_vector: 0,
            watch_count_ptr: mem.watch_count_ptr(),
            mem_self: mem as *const Memory as u64,
            icount_ptr: 0,
            watch_bits_ptr: mem.watch_bits_ptr(),
            code_range_ptr: mem.code_range_ptr(),
            dispatch_word_ptr: &NO_DISPATCH as *const std::sync::atomic::AtomicU64 as u64,
        }
    }

    /// Record a helper miss without losing its whole operand or failing byte.
    #[cold]
    pub fn record_helper_fault(
        &mut self,
        addr: u64,
        size: u64,
        write: bool,
        base: u64,
        width: u64,
    ) {
        self.fault_addr = addr;
        self.fault_size = size.max(1);
        self.fault_access = u64::from(write);
        self.helper_operand_addr = base;
        self.helper_operand_size = width;
    }

    /// Preserve a helper's detailed reason without enlarging MemTrap/Result.
    #[cold]
    pub fn record_helper_trap(&mut self, trap: crate::MemTrap) {
        self.fault_access |= match trap {
            crate::MemTrap::HostFault => 1 << 40,
            crate::MemTrap::Unmapped => 1 << 41,
            _ => 0,
        };
    }

    /// Consume the FP trait's detailed cold channel immediately after its error.
    #[cold]
    pub fn record_fp_failure(&mut self) {
        if let Some(detail) = crate::fault::take_fp_fault() {
            self.fault_size = u64::from(detail.size);
            self.fault_access |= match detail.origin {
                crate::FaultOrigin::HostAccess => 1 << 40,
                crate::FaultOrigin::GuestMap => 1 << 41,
                _ => 0,
            };
        }
    }

    /// Consume an RET_UNMAPPED miss. Canonicality is classified only here in
    /// cold Rust; host faults cannot be non-canonical because spans exclude it.
    #[cold]
    pub fn unmapped_exit(&mut self, cpu: &CpuState, mode: crate::lift::CpuMode) -> Exit {
        let helper_addr = std::mem::take(&mut self.helper_operand_addr);
        let helper_size = std::mem::take(&mut self.helper_operand_size);
        let (base, size) = if helper_size != 0 {
            (helper_addr, helper_size)
        } else {
            let carrier = (self.fault_access >> 8) & 0xffff;
            if carrier != 0 {
                // checked_addr names the exact failing byte and remaining tail.
                // Recover the transfer's first address before using the decoded
                // broadcast element width (SDM Vol 2A §2.7.7; xj-au4.2).
                let transfer = (self.fault_access >> 24) & 0xff;
                (
                    self.fault_addr.wrapping_sub(transfer - self.fault_size),
                    carrier,
                )
            } else {
                (self.fault_addr, self.fault_size)
            }
        };
        let mut exit = crate::memory::unmapped_exit(
            cpu.rip,
            base,
            size,
            mode == crate::lift::CpuMode::Long64,
            self.fault_access & 2 != 0,
            self.fault_addr,
            if self.fault_access & 1 == 0 {
                AccessKind::Read
            } else {
                AccessKind::Write
            },
        );
        if let Exit::UnmappedMemory {
            metadata: Some(m), ..
        } = &mut exit
        {
            m.origin = crate::FaultOrigin::Unknown;
            m.transfer_size = crate::Knowledge::Unknown;
        }
        exit
    }
    /// Decode the tagged cold payload. No out-word aliases are read for legacy tags.
    #[cold]
    pub fn detailed_unmapped_exit(
        &mut self,
        cpu: &CpuState,
        mem: &Memory,
        mode: crate::CpuMode,
        pc: usize,
    ) -> Exit {
        use crate::fault::*;
        let reporting = self.next_entry;
        let owned_info = crate::codemap::lookup_fault_info(pc, reporting);
        let info = owned_info.as_ref();
        let helper = self.helper_operand_size != 0;
        let operand = (self.helper_operand_addr, self.helper_operand_size);
        let inline_width = (self.fault_access >> 24) & 0xff;
        let request = if helper {
            self.fault_size
        } else {
            inline_width
        };
        let base = self.link_slot;
        let mut exit = self.unmapped_exit(cpu, mode);
        if let Exit::UnmappedMemory {
            metadata: Some(m), ..
        }
        | Exit::Exception {
            metadata: Some(m), ..
        } = &mut exit
        {
            m.instruction_rip = Knowledge::Known(reporting);
            m.transfer_access = Knowledge::Known(if self.fault_access & 1 != 0 {
                FaultAccess::Write
            } else {
                FaultAccess::Read
            });
            if m.origin == FaultOrigin::Unknown && (!helper || mem.absent_byte(self.fault_addr)) {
                m.origin = FaultOrigin::GuestMap;
            }
            m.transfer_size = if request != 0 {
                let width = if helper {
                    request as u16
                } else {
                    info.and_then(|i| i.split_transfer_size)
                        .map_or(request as u16, |piece| piece.min(request as u16))
                };
                Knowledge::Known(width)
            } else {
                Knowledge::Unknown
            };
            if helper {
                m.operand_base = Knowledge::Known(operand.0);
                m.operand_size = Knowledge::Known(operand.1 as u16);
                // A helper can fail containment even when its reported byte is
                // mapped. Do not promote that ambiguous failure to a host #PF.
                if !matches!(m.origin, FaultOrigin::GuestException(_)) {
                    m.origin = if self.fault_access & (1 << 40) != 0 {
                        FaultOrigin::HostAccess
                    } else if self.fault_access & (1 << 41) != 0 || mem.absent_byte(self.fault_addr)
                    {
                        FaultOrigin::GuestMap
                    } else {
                        FaultOrigin::Unknown
                    };
                }
                if info.is_some_and(|i| i.operand_size != Knowledge::Known(operand.1 as u16)) {
                    m.operand_base = Knowledge::Unknown;
                }
            } else if info.is_some_and(|i| i.operand_size == Knowledge::Known(request as u16)) {
                m.operand_base = Knowledge::Known(base);
            } else {
                m.operand_size = Knowledge::Unknown;
            }
            if let Knowledge::Known(ref mut address) = m.address {
                *address = FaultAddress::new(
                    address.value(),
                    if m.origin == FaultOrigin::HostAccess || mem.absent_byte(address.value()) {
                        AddressPrecision::ExactByte
                    } else {
                        AddressPrecision::TransferAddress
                    },
                    AddressSpace::Linear,
                );
            }
        }
        let exit = crate::fault::finish(exit, info, cpu, mem, mode, true);
        crate::fault::attach_compiled_path(exit, pc, helper)
    }

    #[cold]
    pub fn detailed_exception_exit(
        &self,
        cpu: &CpuState,
        mem: &Memory,
        mode: crate::CpuMode,
        pc: usize,
    ) -> Exit {
        let reporting = self.next_entry;
        let m = crate::fault::FaultMetadata::exception(
            cpu.rip,
            reporting,
            self.exception_vector as u8,
            crate::fault::cause_from_wire(self.fault_access),
        );
        let exit = Exit::exception(cpu.rip, self.exception_vector as u8).with_fault_metadata(m);
        let owned_info = crate::codemap::lookup_fault_info(pc, reporting);
        let info = owned_info.as_ref();
        let helper = info.is_some_and(|i| match i.helper {
            Some(crate::FaultHelper::CompareExchangePair) => true,
            Some(_) => matches!(
                crate::fault::cause_from_wire(self.fault_access),
                crate::FaultCause::Divide
                    | crate::FaultCause::PendingX87
                    | crate::FaultCause::InvalidMxcsr
            ),
            None => false,
        });
        let exit = crate::fault::finish(exit, info, cpu, mem, mode, true);
        crate::fault::attach_compiled_path(exit, pc, helper)
    }

    #[cold]
    pub(crate) fn preserve_host_replay(
        &self,
        mut exit: Exit,
        info: Option<&crate::fault::InstructionFaultInfo>,
        detailed: bool,
    ) -> Exit {
        use crate::fault::*;
        if let Exit::UnmappedMemory {
            metadata: Some(m), ..
        }
        | Exit::Exception {
            metadata: Some(m), ..
        } = &mut exit
        {
            m.progress.ram = match info {
                Some(i)
                    if !i.writes
                        || (i.rmw && detailed && self.link_slot & 0xff == 0)
                        || (i.preflighted_store
                            && detailed
                            && self.link_slot & 0xff == 1
                            && self.fault_size == 1
                            && matches!(i.operand_size, Knowledge::Known(2..=64))) =>
                {
                    MemoryEffects::None
                }
                Some(_) => MemoryEffects::PossiblyPartial,
                None => MemoryEffects::Unknown,
            };
            if let Some(i) = info {
                m.instruction_rip = Knowledge::Known(i.rip);
                m.instruction_len = i.len;
                m.operand_size = i.operand_size;
                if i.rmw {
                    m.access = Knowledge::Known(FaultAccess::ReadWrite);
                    if let Some(projection) = &mut m.projection {
                        if let ErrorCode::Bits { value, known_mask } = projection.code {
                            projection.code = ErrorCode::bits(value & !2, known_mask & !2);
                        }
                    }
                }
            }
            if detailed {
                m.transfer_size = Knowledge::Known(self.fault_size as u16);
                m.transfer_access = Knowledge::Known(match self.link_slot & 0xff {
                    0 => FaultAccess::Read,
                    1 => FaultAccess::Write,
                    _ => FaultAccess::ReadWrite,
                });
            }
        }
        if detailed {
            crate::fault::attach_compiled_path(exit, self.next_entry as usize, false)
        } else {
            // Legacy recovery has no reliable compiled PC.
            if let Exit::UnmappedMemory {
                metadata: Some(m), ..
            }
            | Exit::Exception {
                metadata: Some(m), ..
            } = &mut exit
            {
                m.path = FaultPath::Unknown;
            }
            exit
        }
    }

    /// Call only after a detailed Access landing, never before Alignment replay.
    #[cold]
    pub fn detailed_host_exit(&self, cpu: &CpuState, mem: &Memory, mode: crate::CpuMode) -> Exit {
        use crate::fault::*;
        let pc = self.next_entry as usize;
        let owned_info = crate::codemap::lookup_fault_info(pc, cpu.rip);
        let info = owned_info.as_ref();
        let mut exit = crate::fault::memory_exit(
            cpu.rip,
            self.fault_addr,
            self.fault_size as u16,
            if self.fault_access & 1 != 0 {
                AccessKind::Write
            } else {
                AccessKind::Read
            },
            true,
            true,
        );
        if let Exit::UnmappedMemory {
            metadata: Some(m), ..
        } = &mut exit
        {
            m.transfer_access = Knowledge::Known(match self.link_slot {
                0 => FaultAccess::Read,
                1 => FaultAccess::Write,
                _ => FaultAccess::ReadWrite,
            });
        }
        let exit = crate::fault::finish(exit, info, cpu, mem, mode, true);
        crate::fault::attach_compiled_path(exit, pc, false)
    }
}

/// Call one compiled block; returns its raw ABI code (chain/link details land in
/// `ctx`). The dispatcher's chain loop (§9.2, §12 M5) interprets the code.
///
/// # Safety
/// `entry` must point at a block compiled to this exact ABI, alive in the JIT
/// arena for the call. `cpu` is exclusive; `ctx` wraps the shared guest buffer.
/// Recovery additionally requires the frame-pointer contract in [`fault::classify`].
pub unsafe fn call_block(entry: CompiledPtr, cpu: &mut CpuState, ctx: &mut MemCtx) -> u64 {
    let _fp_mode = fp::RunMode::new();
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    if cpu.mxcsr & fp::MODE_MASK != 0 {
        return fp::call(
            cpu as *mut CpuState as *mut u8,
            ctx as *mut MemCtx as *mut u8,
            entry.0,
            fp::control(cpu.mxcsr),
        );
    }
    call_block_in_mode(entry, cpu, ctx)
}

/// Dispatcher entry: its run-scoped guard already installed the host mode.
///
/// # Safety
/// As `call_block`, with host FP control matching the translation's mode.
#[inline]
pub(crate) unsafe fn call_block_in_mode(
    entry: CompiledPtr,
    cpu: &mut CpuState,
    ctx: &mut MemCtx,
) -> u64 {
    let f: CompiledFn = core::mem::transmute(entry.0);
    f(
        cpu as *mut CpuState as *mut u8,
        ctx as *mut MemCtx as *mut u8,
    )
}

/// Convenience: run a single compiled block and decode to a `StepResult` (used
/// where chaining isn't wired). Chain/link codes are treated as `Continue` — the
/// RIP is set either way, so the dispatcher re-resolves.
/// `consistency` must match the tier used to compile the block; interpreter replays
/// for deferred MMIO and host alignment faults use the same tier.
///
/// # Safety
/// As [`call_block`].
pub unsafe fn run_compiled(
    entry: CompiledPtr,
    cpu: &mut CpuState,
    mem: &Memory,
    mode: crate::lift::CpuMode,
    consistency: crate::MemConsistency,
) -> StepResult {
    let _fp_mode = fp::RunMode::new();
    let mut ctx = MemCtx::for_memory(mem);
    // A block may push/pop the shadow return stack (R5); give it a live scratch ring
    // so the pointer is never null. Predictions here are inert — a single block is
    // decoded to Continue regardless — but the memory must be valid.
    let mut scratch = RetStack::new();
    ctx.ret_stack = &mut scratch as *mut RetStack as u64;
    match call_block(entry, cpu, &mut ctx) {
        RET_CONTINUE | RET_CHAIN | RET_LINK | RET_IBTC_MISS | RET_MXCSR_CHANGED => {
            StepResult::Continue
        }
        RET_SYSCALL => StepResult::Exit(Exit::Syscall),
        RET_HLT => StepResult::Exit(Exit::Hlt),
        RET_INTERRUPTED => StepResult::Exit(Exit::Interrupted),
        RET_UNMAPPED => StepResult::Exit(ctx.unmapped_exit(cpu, mode)),
        RET_UNMAPPED_DETAIL => {
            StepResult::Exit(ctx.detailed_unmapped_exit(cpu, mem, mode, entry.0 as usize))
        }
        RET_EXCEPTION_DETAIL => {
            StepResult::Exit(ctx.detailed_exception_exit(cpu, mem, mode, entry.0 as usize))
        }
        RET_HOST_FAULT_DETAIL if ctx.link_slot & 0x100 != 0 => {
            let fault_info = crate::codemap::lookup_fault_info(ctx.next_entry as usize, cpu.rip);
            let mut scratch = Vec::new();
            let mut info = crate::interp::RetireInfo {
                consistency,
                ..Default::default()
            };
            match crate::interp::step_one(mem, cpu, mode, &mut scratch, &mut info) {
                StepResult::Exit(exit) => {
                    StepResult::Exit(ctx.preserve_host_replay(exit, fault_info.as_ref(), true))
                }
                other => other,
            }
        }
        RET_HOST_FAULT => StepResult::Exit(ctx.unmapped_exit(cpu, mode)),
        RET_HOST_FAULT_DETAIL => StepResult::Exit(ctx.detailed_host_exit(cpu, mem, mode)),
        // An inlined access hit a Trap region (M4-T10): single-step the faulting
        // instruction on the interpreter, which yields the MmioRead/Write exit.
        RET_MMIO_DEFER => {
            let mut temps = Vec::new();
            // This standalone helper has no `Vcpu` to tick; the retired-instruction
            // counter (§17.6, sub-seam c) is maintained only on `Vcpu::run`, so discard
            // the per-step retirement outputs here while retaining the ordering input.
            let mut info = crate::interp::RetireInfo {
                consistency,
                ..Default::default()
            };
            crate::interp::step_one(mem, cpu, mode, &mut temps, &mut info)
        }
        // A compiled block raised a guest exception; the block set the saved RIP
        // (fault: on the instruction; trap: past it) and stored the vector (`#DE`→0,
        // `ud2`→6, `int3`→3, `int1`→1) into `ctx.exception_vector`.
        RET_EXCEPTION => StepResult::Exit(
            Exit::exception(cpu.rip, ctx.exception_vector as u8)
                .with_fault_metadata(crate::FaultMetadata::unknown(cpu.rip)),
        ),
        other => panic!("compiled block returned an invalid ABI code: {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mxcsr_is_appended_and_existing_cpu_offsets_are_unchanged() {
        let o = cpu_offsets();
        assert_eq!(o.gpr, 0);
        assert_eq!(o.rip, 128);
        assert_eq!(o.fs_base, 136);
        assert_eq!(o.gs_base, 144);
        assert_eq!(o.cf, 152);
        assert_eq!(o.pf_src, 153);
        assert_eq!(o.af_src, 154);
        assert_eq!(o.zf, 155);
        assert_eq!(o.sf, 156);
        assert_eq!(o.of, 157);
        assert_eq!(o.df, 158);
        assert_eq!(o.xmm, 160);
        assert_eq!(o.ymm_hi, 672);
        assert_eq!(o.zmm_hi, 1184);
        assert_eq!(o.kmask, 2208);
        let s = CpuState::new();
        let base = &s as *const CpuState as usize;
        assert_eq!(o.mxcsr as usize, &s.mxcsr as *const u32 as usize - base);
        assert!(o.mxcsr as usize >= &s.gs as *const u16 as usize - base + 2);
        // SDM Vol 1 §11.6.4, Table 11-2; §11.6.6: our DAZ-capable mask.
        assert_eq!(s.mxcsr, 0x1f80);
        assert_eq!(crate::state::MXCSR_MASK, 0xffff);
    }

    /// The real-mode IF flag (§17.6, sub-seam b) must sit in the pad byte at offset 159
    /// so `xmm` stays 16-byte-aligned at 160 and every `CpuOffsets` value is unchanged.
    /// This pins the ABI-safe placement: a reorder that shifts `df`/`if_`/`xmm` fails here.
    #[test]
    fn if_flag_uses_pad_byte_and_keeps_xmm_at_160() {
        let s = CpuState::new();
        let base = &s as *const CpuState as usize;
        let off = |p: *const u8| p as usize - base;
        assert_eq!(
            off(&s.flags.df as *const bool as *const u8),
            158,
            "df at 158"
        );
        assert_eq!(
            off(&s.flags.if_ as *const bool as *const u8),
            159,
            "if_ occupies the former pad byte at 159"
        );
        assert_eq!(off(s.xmm.as_ptr() as *const u8), 160, "xmm stays at 160");
        // And the measured CpuOffsets are unchanged by the IF addition.
        let o = cpu_offsets();
        assert_eq!(o.df, 158);
        assert_eq!(o.xmm, 160);
    }

    /// SDM Vol 2A §2.7.7: the broadcast carrier describes one element.
    /// The wider transfer remains tracked separately in xj-au4.2.
    #[test]
    fn inline_broadcast_carrier_recovers_base_from_a_failing_tail() {
        use crate::lift::CpuMode;
        let mem = Memory::new(crate::memory::MemoryModel::Flat { size: 4096 });
        let mut ctx = MemCtx::for_memory(&mem);
        let mut cpu = CpuState::new();
        cpu.rip = 0x1000;
        let edge = 1u64 << 47;
        ctx.fault_addr = edge;
        ctx.fault_size = 12;
        ctx.fault_access = (4 << 8) | (16 << 24);
        assert!(matches!(
            ctx.unmapped_exit(&cpu, CpuMode::Long64),
            Exit::UnmappedMemory {
                addr,
                access: AccessKind::Read
            , .. } if addr == edge
        ));
        ctx.fault_size = 14;
        assert!(matches!(
            ctx.unmapped_exit(&cpu, CpuMode::Long64),
            Exit::Exception {
                addr,
                vector: 13
            , .. } if addr == cpu.rip
        ));
    }

    #[test]
    fn helper_operand_is_consumed_before_a_later_inline_miss() {
        use crate::lift::CpuMode;
        let mem = Memory::new(crate::memory::MemoryModel::Flat { size: 0x1000 });
        let cpu = CpuState {
            rip: 0x120,
            ..CpuState::default()
        };
        let mut ctx = MemCtx::for_memory(&mem);
        ctx.record_helper_fault(0x7fff_ffff_fff8, 8, false, 0x7fff_ffff_fff8, 16);
        ctx.fault_access |= 2;
        assert!(matches!(
            ctx.unmapped_exit(&cpu, CpuMode::Long64),
            Exit::Exception {
                addr: 0x120,
                vector: 12,
                ..
            }
        ));
        assert_eq!((ctx.helper_operand_addr, ctx.helper_operand_size), (0, 0));
        ctx.fault_addr = 0xffff_8000_0000_0000;
        ctx.fault_size = 8;
        ctx.fault_access = 0;
        assert!(matches!(
            ctx.unmapped_exit(&cpu, CpuMode::Long64),
            Exit::UnmappedMemory {
                addr: 0xffff_8000_0000_0000,
                access: AccessKind::Read,
                ..
            }
        ));
        ctx.record_helper_fault(0x1010, 8, true, 0x1000, 32);
        assert!(matches!(
            ctx.unmapped_exit(&cpu, CpuMode::Long64),
            Exit::UnmappedMemory {
                addr: 0x1010,
                access: AccessKind::Write,
                ..
            }
        ));
    }

    /// The `MEMCTX_*` offsets are a codegen contract (the JIT bakes them). Verify
    /// them against the real `#[repr(C)]` layout so a field reorder can't silently
    /// desync the backend. `fuel` (superblocks M5-T3) must sit at 56.
    #[test]
    fn memctx_offsets_match_layout() {
        let m = MemCtx::for_memory(&Memory::new(crate::memory::MemoryModel::Flat {
            size: 0x1000,
        }));
        let base = &m as *const MemCtx as usize;
        let off = |p: *const u64| (p as usize - base) as i32;
        assert_eq!(off(&m.helper_operand_addr), 136);
        assert_eq!(off(&m.helper_operand_size), 144);
        assert_eq!(std::mem::size_of::<MemCtx>(), 152);
        assert_eq!(off(&m.dispatch_word_ptr), MEMCTX_DISPATCH_WORD_PTR);
        assert_eq!(MEMCTX_DISPATCH_WORD_PTR, 128);
        assert_eq!(m.dispatch_word_ptr, &NO_DISPATCH as *const _ as u64);
        assert_eq!(NO_DISPATCH.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(off(&m.base), MEMCTX_BASE);
        assert_eq!(off(&m.size), MEMCTX_SIZE);
        assert_eq!(off(&m.fault_addr), MEMCTX_FAULT_ADDR);
        assert_eq!(off(&m.fault_size), MEMCTX_FAULT_SIZE);
        assert_eq!(off(&m.fault_access), MEMCTX_FAULT_ACCESS);
        assert_eq!(off(&m.next_entry), MEMCTX_NEXT_ENTRY);
        assert_eq!(off(&m.link_slot), MEMCTX_LINK_SLOT);
        assert_eq!(off(&m.icount_ptr), MEMCTX_ICOUNT_PTR);
        assert_eq!(off(&m.watch_bits_ptr), MEMCTX_WATCH_BITS_PTR);
        assert_eq!(MEMCTX_WATCH_BITS_PTR, 112);
        assert_eq!(off(&m.fuel), MEMCTX_FUEL);
        assert_eq!(MEMCTX_FUEL, 56);
        assert_eq!(off(&m.ret_stack), MEMCTX_RET_STACK);
        assert_eq!(MEMCTX_RET_STACK, 64);
        assert_eq!(off(&m.guest_base), MEMCTX_GUEST_BASE);
        assert_eq!(MEMCTX_GUEST_BASE, 72);
        assert_eq!(off(&m.exception_vector), MEMCTX_EXCEPTION_VECTOR);
        assert_eq!(MEMCTX_EXCEPTION_VECTOR, 80);
        assert_eq!(off(&m.watch_count_ptr), MEMCTX_WATCH_COUNT_PTR);
        assert_eq!(MEMCTX_WATCH_COUNT_PTR, 88);
        assert_eq!(off(&m.mem_self), MEMCTX_MEM_SELF);
        assert_eq!(MEMCTX_MEM_SELF, 96);
        assert_eq!(off(&m.code_range_ptr), MEMCTX_CODE_RANGE_PTR);
        assert_eq!(MEMCTX_CODE_RANGE_PTR, 120);
    }

    /// `RetStack` field offsets are a codegen contract too (R5).
    #[test]
    fn retstack_offsets_match_layout() {
        let rs = RetStack::new();
        let base = &rs as *const RetStack as usize;
        assert_eq!(&rs.sp as *const u64 as usize - base, RETSTACK_SP as usize);
        assert_eq!(
            rs.entries.as_ptr() as usize - base,
            RETSTACK_ENTRIES as usize
        );
        // Frame stride: consecutive `[u64; 2]` entries are 16 bytes apart.
        assert_eq!(
            rs.entries[1].as_ptr() as usize - rs.entries[0].as_ptr() as usize,
            RETSTACK_STRIDE as usize
        );
    }
}
