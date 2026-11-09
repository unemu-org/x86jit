//! `Vm` (shared) and `Vcpu` (per-thread) — the KVM-style split (§2), plus the
//! dispatcher loop (§9.2).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::cache::{BlockKey, CachedBlock, CompiledPtr, TranslationCache};
use crate::exit::{Exit, StepResult};
use crate::interruption::{CriticalSectionKind, EngineSection};
use crate::ir::{IrBlock, IrRegion, RegionCaps};
use crate::jit_abi::{
    MemCtx, RetStack, RET_CHAIN, RET_CONTINUE, RET_EXCEPTION, RET_EXCEPTION_DETAIL, RET_HLT,
    RET_HOST_FAULT, RET_HOST_FAULT_DETAIL, RET_IBTC_MISS, RET_INTERRUPTED, RET_LINK,
    RET_MMIO_DEFER, RET_MXCSR_CHANGED, RET_PORTIO_DEFER, RET_SYSCALL, RET_UNMAPPED,
    RET_UNMAPPED_DETAIL,
};
use crate::lift::{
    lift_block_with_fault, lift_region, lift_region_with_fault, CpuMode, FetchAddr, LiftError,
};
use crate::memory::{HostRam, MapError, MemError, Memory, MemoryModel, Prot, RegionKind};
use crate::state::{CpuState, Flags, Reg};

/// Materializes IR into an executable `CachedBlock` (§8). The ONLY
/// backend-dependent operation; execution is uniform.
///
/// Injected as a trait object (§4.1) — NOT a config enum. The core can't name
/// the downstream JIT crate (dependency points the other way), so an
/// `enum Backend { Interpreter, Jit }` is unbuildable. The interpreter impl lives
/// here; `x86jit-cranelift` exports a `JitBackend` implementing this same trait
/// and the user injects it via `Vm::with_backend`.
///
/// `materialize` takes `&self` (not `&mut self`) so a `Vm` can be shared across
/// vcpus behind `Arc`. A JIT impl that needs a mutable compiler context wraps it
/// in interior mutability (e.g. `Mutex`).
pub trait Backend: Send + Sync {
    /// Permanently stop admission, cancel queued work, join every owned worker,
    /// and discard completions. Called before VM cache and memory destruction.
    /// Implementations with workers must override this; the default owns none.
    /// Repeated calls must be safe, including after an error. Never call from
    /// a worker callback or a signal/atfork handler.
    fn shutdown(&mut self) -> Result<ShutdownReport, ShutdownError> {
        Ok(ShutdownReport::default())
    }

    /// Whether code storage has exhausted its capacity. Once true, it stays true
    /// for this backend's lifetime: materialization falls back to interpretation.
    /// Each vcpu reports the transition through [`Exit::CodeSpaceExhausted`].
    fn code_space_exhausted(&self) -> bool {
        false
    }

    /// Register a vcpu's dispatch word for asynchronous code-space notification.
    /// Exhaustion sets [`CODE_SPACE_EXHAUSTED`] in every live registered word,
    /// including words registered after the sticky transition. Other bits belong
    /// to SMC and host exit requests and must be preserved.
    fn register_dispatch_word(&self, _word: &Arc<AtomicU64>) {}

    /// Compile `ir` for the given consistency tier. `consistency` only affects a
    /// JIT on a weak host (it picks the barrier strategy for ordinary guest
    /// loads/stores, §8.2.3); the interpreter and x86 hosts ignore it.
    /// `mmio` is the guest's `Trap`-region window (`[lo, hi)`), or `None` if the VM
    /// has no MMIO regions (§5.2). A JIT bakes it as a compile-time constant
    /// and, when `Some`, adds a per-access range check that defers a Trap-region
    /// load/store to the interpreter; `None` means no check and zero overhead.
    /// `guest_base` is the guest address the RAM buffer's first byte represents (§4.1,
    /// identity mapping). A JIT bakes it as a compile-time constant: `0` (the common
    /// case) emits the historical `host = base + guest_addr`; a non-zero base emits the
    /// rebased `host = base + (guest_addr - guest_base)` plus a lower-bound reject, so
    /// the zero-base hot path is byte-identical.
    fn materialize(
        &self,
        ir: &IrBlock,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
    ) -> CachedBlock;

    /// Materialize a translation specialised to RC | FTZ | DAZ. Backends that
    /// do not specialise compiled floating-point code interpret non-default modes.
    fn materialize_with_mxcsr(
        &self,
        ir: &IrBlock,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
        mxcsr_mode: u32,
    ) -> CachedBlock {
        if mxcsr_mode != 0 {
            return CachedBlock::Interpreted(Arc::new(ir.clone()));
        }
        self.materialize(ir, consistency, mmio, guest_base)
    }

    /// Region counterpart of [`Self::materialize_with_mxcsr`].
    fn materialize_region_with_mxcsr(
        &self,
        region: &IrRegion,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
        mxcsr_mode: u32,
    ) -> CachedBlock {
        if mxcsr_mode != 0 {
            return CachedBlock::Interpreted(Arc::new(region.blocks[0].clone()));
        }
        self.materialize_region(region, consistency, mmio, guest_base)
    }

    /// Compile with the VM's split-store policy. The default preserves existing
    /// backends when disabled and interprets when precision is requested. Wrapping
    /// backends should forward this method to retain compiled execution.
    #[allow(clippy::too_many_arguments)]
    fn materialize_with_split_stores(
        &self,
        ir: &IrBlock,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
        mxcsr_mode: u32,
        precise_split_stores: bool,
    ) -> CachedBlock {
        if precise_split_stores {
            return CachedBlock::Interpreted(Arc::new(ir.clone()));
        }
        self.materialize_with_mxcsr(ir, consistency, mmio, guest_base, mxcsr_mode)
    }

    /// Region counterpart of [`Self::materialize_with_split_stores`].
    #[allow(clippy::too_many_arguments)]
    fn materialize_region_with_split_stores(
        &self,
        region: &IrRegion,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
        mxcsr_mode: u32,
        precise_split_stores: bool,
    ) -> CachedBlock {
        if precise_split_stores {
            return CachedBlock::Interpreted(Arc::new(region.blocks[0].clone()));
        }
        self.materialize_region_with_mxcsr(region, consistency, mmio, guest_base, mxcsr_mode)
    }

    /// Superblock caps if this backend forms regions (§12 M5), else `None`
    /// (the default). When `Some`, the dispatcher lifts a region and calls
    /// [`materialize_region`](Backend::materialize_region); a one-block region
    /// falls back to `materialize`. This can change from `Some` to `None` at
    /// runtime after code-space exhaustion, including on a background worker.
    /// Callers must snapshot the caps for each region-lifting operation.
    fn region_caps(&self) -> Option<RegionCaps> {
        None
    }

    /// Compile a multi-block region into one unit. Only called when
    /// [`region_caps`](Backend::region_caps) is `Some`; the default is unreachable.
    fn materialize_region(
        &self,
        _region: &IrRegion,
        _consistency: MemConsistency,
        _mmio: Option<(u64, u64)>,
        _guest_base: u64,
    ) -> CachedBlock {
        unreachable!("materialize_region called on a backend without region_caps")
    }

    /// Invalidate every backend-owned cached code pointer (link slots, and later
    /// IBTC / return-continuation slots) after an SMC invalidation dropped one or
    /// more compiled units. The default is a no-op (the
    /// interpreter has no such state). A JIT clears all its slots: a cleared slot
    /// re-links via the existing `RET_LINK` path on the next traversal, so
    /// over-invalidation is safe and avoids a reverse target→slot index. Called
    /// only when [`TranslationCache::invalidate_overlapping`] actually drops a unit,
    /// which is rare (a write landing on a code page).
    fn invalidate_links(&self) {}

    /// Submit a hot block for **background** compilation off the vcpu's critical
    /// path. The default is [`TierUpSubmit::Unsupported`] — a backend that doesn't run
    /// a compiler thread (the interpreter, or the JIT with background tier-up disabled)
    /// never queues, and the dispatcher falls back to its existing inline/eager path. A
    /// backend that accepts the work returns [`TierUpSubmit::Queued`];
    /// [`TierUpSubmit::Busy`] means "queue full, stay interpreted and retry" — never an
    /// inline compile spike under peak pressure. Takes `&self` (like
    /// [`materialize`](Backend::materialize)) — the compiler state is interior-mutable.
    fn tier_up_async(&self, _req: TierUpRequest) -> TierUpSubmit {
        TierUpSubmit::Unsupported
    }

    /// Submit background work approved by an opt-in fetch policy. Backends that
    /// support cancellation must check `permit` before compiling queued work.
    /// The default declines, letting the dispatcher compile inline and retain
    /// its invalidation checks. Existing backends need no changes when no policy
    /// is installed.
    fn tier_up_async_with_fetch_permit(
        &self,
        _req: TierUpRequest,
        _permit: crate::FetchPermit,
    ) -> TierUpSubmit {
        TierUpSubmit::Unsupported
    }

    /// Drain finished background compiles for the core dispatcher to publish via
    /// `cache.upgrade`; the backend never touches the cache. The default returns an
    /// empty `Vec` (no allocation) for a backend that never queues. Called at the top
    /// of the dispatch loop; each result carries the epoch snapshot taken at submit, so
    /// a stale compile is rejected on publish.
    /// An `Interpreted` completion means compilation failed: the dispatcher
    /// publishes nothing and clears the request's in-flight marker.
    fn tier_up_finished(&self) -> Vec<TierUpFinished> {
        Vec::new()
    }

    /// Total time spent compiling (in `materialize`/`materialize_region`) over this
    /// backend's lifetime, in nanoseconds — for the bench's compile-vs-run split. The
    /// default is `0`: a backend that does no compilation (the interpreter) has no
    /// compile cost to subtract. A JIT accumulates it with interior mutability.
    /// Observability only — never on the hot path.
    fn compile_ns(&self) -> u64 {
        0
    }

    /// Tell the backend whether this VM tiers up, from
    /// [`Vm::set_tier_up_after`]. `true` means a block is only compiled once it has
    /// run `tier_up_after` times — it is hot by construction, so heavier codegen
    /// pays for itself. `false` is eager compilation: every block is compiled on
    /// first execution and most run once, so optimization is pure compile cost.
    ///
    /// A backend may only act on this before its first compile (the Cranelift JIT
    /// bakes the level into its ISA), so call it during setup. The default is a
    /// no-op — the interpreter has no codegen to tune.
    fn set_tiering(&self, _tiered: bool) {}

    /// One-line description of the codegen this backend resolved to, for an embedder
    /// to log or assert on. Needed because a `Vm` owns its backend as a
    /// `Box<dyn Backend>`, so a concrete accessor like `JitBackend::opt_level` is out
    /// of reach exactly where it matters — an embedder cannot otherwise confirm that
    /// the level derived from its tier-up policy is the one it expected. The default
    /// is the type's own name; format is diagnostic, not a stable contract.
    fn codegen_description(&self) -> String {
        "interpreter".to_string()
    }

    /// Calls out of compiled code into interpreter helpers, per helper, highest first.
    /// Empty for a backend with no such path — the interpreter is the helper.
    ///
    /// A helper call runs a whole interpreter operation behind a C-ABI boundary, so a
    /// guest whose hot code hits helpers pays a per-instruction premium that mid-end
    /// tuning cannot recover. On the [`Backend`] trait rather than the concrete type
    /// because a `Vm` owns its backend as `Box<dyn Backend>` — which is exactly where
    /// an embedder needs to read this.
    fn helper_calls(&self) -> Vec<(&'static str, u64)> {
        Vec::new()
    }
}

/// A hot block handed to a backend for background compilation. Plain data — no
/// threads or channels cross the [`Backend`] boundary, so `x86jit-core`'s dependency
/// set stays `{iced-x86}` (§15). Mirrors the arguments the inline tier-up already
/// passes to [`Backend::materialize`], plus the `span`/`epoch` the dispatcher needs to
/// publish the result safely.
pub struct TierUpRequest {
    /// Split-store policy snapshotted when the request was submitted.
    /// Backends must honor it or decline/interpret the request.
    pub precise_split_stores: bool,
    /// RC | FTZ | DAZ assumed by the submitted translation.
    pub mxcsr_mode: u32,
    /// Guest entry address of the block/region (its cache key).
    pub pc: u64,
    /// The already-lifted IR to compile — a single block, or a hotness-gated
    /// superblock region compiled off-thread.
    pub unit: TierUpUnit,
    /// Consistency tier to compile for (§8.2.3).
    pub consistency: MemConsistency,
    /// The guest `Trap`-region window, baked as a constant (§5.2).
    pub mmio: Option<(u64, u64)>,
    /// The guest base (host addr of `ptr[0]`), baked as a constant (§4.1). `0` is the
    /// common zero-based layout; non-zero drives identity mapping.
    pub guest_base: u64,
    /// The guest byte span(s) `(start, len)` for re-establishing SMC coverage on
    /// publish — one for a block, one per sub-block for a region (matches
    /// [`TranslationCache::insert`]'s span list).
    pub spans: Vec<(u64, u32)>,
    /// Invalidation epoch snapshotted at submit; a publish is rejected if the cache
    /// epoch has moved past it (an SMC drop invalidated the unit mid-compile).
    pub epoch: u64,
}

/// The IR unit a background tier-up compiles. Plain data across the [`Backend`]
/// boundary — no cranelift types leak into the core.
pub enum TierUpUnit {
    /// A single already-lifted block.
    Block(Arc<IrBlock>),
    /// A hotness-gated multi-block superblock region: only proven-hot loops form one,
    /// and only off the vcpu — region compile is too heavy inline (superblock-plan.md
    /// T3f).
    Region(Arc<IrRegion>),
}

/// A finished background compile, ready for the core dispatcher to publish. Carries
/// everything `cache.upgrade` needs; the backend returns these from
/// [`Backend::tier_up_finished`] and never writes the cache itself.
pub struct TierUpFinished {
    /// RC | FTZ | DAZ assumed by the submitted translation.
    pub mxcsr_mode: u32,
    /// Guest entry address (cache key), echoing the request's `pc`.
    pub pc: u64,
    /// The compiled unit to swap in for the interpreted block. `Interpreted`
    /// means compilation failed: nothing is published and the in-flight marker
    /// is cleared.
    pub block: CachedBlock,
    /// The guest byte span(s) `(start, len)` — one per sub-block for a region.
    pub spans: Vec<(u64, u32)>,
    /// The epoch snapshotted at submit, checked against the live cache epoch.
    pub epoch: u64,
}

/// Outcome of a [`Backend::tier_up_async`] submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierUpSubmit {
    /// Accepted — the block will be compiled on the backend's worker thread.
    Queued,
    /// Rejected for backpressure (queue full) — stay interpreted and retry later;
    /// the dispatcher must NOT compile inline in response.
    Busy,
    /// The backend runs no background compiler (interpreter, or JIT with background
    /// tier-up off) — the dispatcher uses its existing inline/eager path.
    Unsupported,
}

/// Default backend: wrap the IR in an `Arc` and interpret it (§8.1).
pub struct InterpreterBackend;

impl Backend for InterpreterBackend {
    fn materialize(
        &self,
        ir: &IrBlock,
        _consistency: MemConsistency,
        _mmio: Option<(u64, u64)>,
        _guest_base: u64,
    ) -> CachedBlock {
        CachedBlock::Interpreted(Arc::new(ir.clone()))
    }
}

/// Memory-consistency tier for guest accesses on weak hosts (§8.2.3).
/// aarch64 Fast uses LDR/STR; AcqRel places DMB ISH before stores and after
/// loads; FullTso places DMB ISH after each store and load. x86 stays plain.
/// Memory helpers are bracketed by direction: AcqRel fences before writes and
/// after reads; FullTso fences before reads and after reads or writes.
/// Explicit MFENCE is a full barrier in every tier, host and backend; the JIT
/// emits LFENCE/SFENCE barriers in every aarch64 tier to cover helpers too.
/// Locked operations are atomic in every tier, but aarch64 LL/SC paths lack
/// full x86 locked-op ordering, including XCHG even on LSE-capable hosts.
/// Distinct from `MemoryModel` (address-space layout). Baked into compiled
/// blocks: changing it requires flushing the translation cache.
#[derive(Default, Copy, Clone, PartialEq, Eq, Debug)]
pub enum MemConsistency {
    /// Bare STR/LDR, no barriers. Fastest. Correct only for code that doesn't
    /// synchronize through memory (single-threaded, or non-communicating threads).
    #[default]
    Fast,
    /// Inline interpreter/JIT stores: DMB ISH; STR. Inline loads: LDR; DMB ISH.
    AcqRel,
    /// Inline interpreter/JIT stores: STR; DMB ISH. Inline loads: LDR; DMB ISH.
    /// No leading inline-store fence.
    FullTso,
}

pub struct VmConfig {
    pub memory_model: MemoryModel,
    /// Consistency tier for weak hosts (§4.1, §8.2.3). Start: `Fast`.
    pub consistency: MemConsistency,
}

impl VmConfig {
    /// A `Flat` guest of `size` bytes at the default `Fast` consistency — the common
    /// case. Refine `consistency` on the returned value if a weak host needs a stronger
    /// tier.
    pub fn flat(size: u64) -> Self {
        VmConfig {
            memory_model: MemoryModel::Flat { size },
            consistency: MemConsistency::Fast,
        }
    }

    /// A `Reserved` (embedder-mmap'd) span of `span` bytes at `Fast` consistency.
    /// Pair with [`Vm::with_backend_host_ram`].
    pub fn reserved(span: u64) -> Self {
        VmConfig {
            memory_model: MemoryModel::Reserved { span },
            consistency: MemConsistency::Fast,
        }
    }
}

/// Failure to prepare a deterministic region fixture.
#[derive(Debug)]
#[non_exhaustive]
pub enum ForceRegionError {
    UnsupportedMode,
    Lift(Exit),
    NotCompiled,
    Invalidated,
}

/// Failure to allocate or seed a vCPU source identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceIdError {
    /// This identity was already allocated, even if its vCPU has been dropped.
    InUse(u64),
    /// Seeding cannot move the automatic allocator backwards.
    Rewind { current: u64, requested: u64 },
    /// Automatic allocation has consumed `u64::MAX` and cannot wrap.
    Exhausted,
}

struct SourceIds {
    next: Option<u64>,
    allocated: HashSet<u64>,
}

impl Default for SourceIds {
    fn default() -> Self {
        Self {
            next: Some(0),
            allocated: HashSet::new(),
        }
    }
}

impl SourceIds {
    fn allocate(&mut self, explicit: Option<u64>) -> Result<u64, SourceIdError> {
        let id = explicit.or(self.next).ok_or(SourceIdError::Exhausted)?;
        if !self.allocated.insert(id) {
            return Err(SourceIdError::InUse(id));
        }
        if self.next.is_some_and(|next| id >= next) {
            self.next = id.checked_add(1);
        }
        Ok(id)
    }
}

/// Resources accounted for by an ordered engine shutdown.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShutdownReport {
    pub workers_joined: usize,
    pub jobs_cancelled: usize,
    /// Resource-owning handles still live after joining workers; zero on success.
    pub outstanding_handles: usize,
}

/// Shutdown completed its joins but could not certify a clean engine teardown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShutdownError {
    WorkerPanicked {
        report: ShutdownReport,
    },
    LiveResources {
        resource: &'static str,
        report: ShutdownReport,
    },
}

impl std::fmt::Display for ShutdownError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WorkerPanicked { .. } => {
                write!(f, "background compiler panicked (worker joined)")
            }
            Self::LiveResources { resource, report } => write!(
                f,
                "{} outstanding handles to {resource}",
                report.outstanding_handles
            ),
        }
    }
}

impl std::error::Error for ShutdownError {}

/// Shared per-machine state: guest memory + translation cache + backend (§2).
pub struct Vm {
    vcpu_lifetime: Arc<()>,
    precise_split_stores: bool,
    compile_hold: Option<(std::time::Duration, Arc<std::sync::atomic::AtomicBool>)>,
    lockstep_trace: Option<Arc<crate::lockstep::Trace>>,
    // Fields drop in declaration order, after Drop has joined backend workers.
    pub backend: Box<dyn Backend>,
    pub cache: TranslationCache,
    pub mem: Memory,
    pub consistency: MemConsistency,
    /// Hotness-gated tier-up: when `Some(n)`, a freshly-lifted block
    /// runs on the interpreter and is JIT-compiled only after it executes `n` times.
    /// `None` (default) keeps the eager behavior — compile every block on first
    /// sight. Cuts one-shot compile cost (run-once blocks never reach the backend)
    /// while hot loops still tier up. Only meaningful with a compiling backend.
    tier_up_after: Option<u32>,
    /// Background tier-up: when true — and `tier_up_after` is `Some` with an
    /// async-capable backend — a hot block is compiled on the backend's worker thread
    /// and swapped in when ready, instead of compiling inline on the vcpu's critical
    /// path. Default false: opt-in, so the differential/fuzz corpus never depends on
    /// *when* the interp→compiled switch lands. Falls back to inline tier-up on a
    /// backend that returns `Unsupported`.
    tier_up_background: bool,
    /// Adaptive region tier-up threshold T2: with a region-forming backend, a hot
    /// **loop** stays interpreted until it has run `Some(n)` times before tiering up to
    /// a background superblock region — a much higher bar than `tier_up_after` (T1),
    /// because a region's heavy compile only pays off on a long-running loop (measured —
    /// a premature region regresses, superblock-plan.md T3f). Non-loop blocks tier the
    /// single block at T1 as usual. `None` → use T1: a loop regions as soon as it's hot.
    ///
    /// **Partial.** A region-candidate loop stays *interpreted* until T2 — it does NOT
    /// take a single-block baseline tier in the meantime, so a hot-but-shorter-than-T2
    /// loop that a region wouldn't help interprets the whole time (no baseline speedup).
    /// The fix is a compiled-in **backedge counter** (true OSR): baseline-compile at T1,
    /// then promote the *compiled* loop to a region at T2 — a dispatcher counter can't
    /// do it because chained compiled blocks never return to the dispatcher. Until then
    /// keep T2 `None` (or use a region-forming backend without setting T2).
    tier_up_region_after: Option<u32>,
    /// Guest CPU feature set every new vcpu starts with. Default is
    /// [`GuestCpuFeatures::default`](crate::features::GuestCpuFeatures); an embedder
    /// selects a different ISA level via [`Vm::set_guest_cpu_features`] before spawning
    /// vcpus.
    features: crate::features::GuestCpuFeatures,
    x87_pointer_model: crate::state::X87PointerModel,
    cpuid_table: Option<Arc<crate::features::CpuidTable>>,
    cpu_source: Option<Arc<dyn crate::state::CpuSource>>,
    source_ids: Arc<Mutex<SourceIds>>,
    /// x87 transcendental precision every new vcpu starts with. Default `Fast`
    /// (f64/libm); select `Extended` (full-80-bit) via [`Vm::set_x87_precision`].
    x87_precision: crate::state::X87Precision,
    /// Guest decode/lift mode (§17.3): the effective operand/address-size default every
    /// block lifts under, and part of the block-cache key (§17.4). A `Vm` is constructed
    /// in one mode (the §17 scope fence — no runtime mode switching); vcpus inherit it.
    /// Default [`CpuMode::Long64`]; select via [`Vm::set_cpu_mode`] before spawning vcpus.
    mode: CpuMode,
}

impl Vm {
    /// Snapshot child execution metadata before a shared-address-space fork.
    /// Allocation and fetch-policy callbacks run here, in the parent. `mode`
    /// selects decoding; Real16 helpers are outside the initial child scope.
    /// The view holds only owned metadata and a lifetime marker, keeping this
    /// VM's mapping alive and preventing safe unmap/remap until the view is dropped.
    /// See the crate docs for the FFI and parent invalidation protocol.
    pub fn child_view(&self, mode: CpuMode) -> crate::ChildView<'_> {
        let mut view = self.mem.child_view(self, mode);
        view.code.consistency = self.consistency;
        view
    }

    /// Reconcile a finished child's written-page log with concurrent translation
    /// and execution, using the ordinary address-scoped invalidation protocol.
    /// Call after child exit/exec, while its view and backing still live. Watched
    /// pages are also reported to the parent's normal dirty-range consumer.
    pub fn invalidate_written_pages(&self, arena: &crate::ChildArena) {
        let mut dropped = false;
        for &page in arena.written_pages().iter() {
            let lo = page << crate::memory::CODE_PAGE_BITS;
            self.mem
                .note_write(lo, crate::memory::CODE_PAGE_SIZE as usize);
            dropped |= !self
                .cache
                .invalidate_child_written(lo, lo + crate::memory::CODE_PAGE_SIZE, || {
                    self.mem.clear_code_page(page)
                })
                .is_empty();
        }
        if dropped {
            self.backend.invalidate_links();
        }
    }

    /// Consume the engine before a host fork, then build independent engines in
    /// parent and child. No translations or worker state are reused afterward.
    ///
    /// First stop and join embedder-owned vCPU threads: request their exits with
    /// [`ExitRequest`], wake any blocked outside `run`, and drop their vCPUs.
    /// Recover exclusive VM ownership (e.g. `Arc::try_unwrap`) before calling.
    /// This ownership boundary prevents new runs or foreground compilations.
    /// Then the backend stops submissions, cancels queued compiles, finishes any
    /// in-flight compile, joins its workers and discards completions. Finally the
    /// backend, cache and memory views are destroyed, in that order, even on error.
    /// A successful report guarantees no engine-owned worker remains.
    /// A vCPU that was not dropped is reported as a live resource, even if idle.
    ///
    /// Diagnostic handles may outlive shutdown but must not retain engine state.
    /// The report counts resource-owning handles, not inert diagnostic tokens.
    /// This is a normal-thread operation, not a signal or `pthread_atfork` hook.
    /// Shut down every VM in the process before the host fork. The embedder owns
    /// signal routing, blocked vCPU wakeups and reconstruction in both processes.
    /// Executable arenas and signal-safe code-map records retain their existing
    /// process lifetime; translation reclamation is a separate contract.
    pub fn shutdown(mut self) -> Result<ShutdownReport, ShutdownError> {
        let live_vcpus = Arc::strong_count(&self.vcpu_lifetime) - 1;
        let mut report = self.backend.shutdown();
        if live_vcpus != 0 {
            let (mut counts, resource) = match report {
                Ok(counts) => (counts, "vCPUs"),
                Err(ShutdownError::LiveResources { report, .. }) => {
                    (report, "vCPUs and compiler state")
                }
                Err(ShutdownError::WorkerPanicked { report }) => {
                    (report, "vCPUs (compiler worker panicked)")
                }
            };
            counts.outstanding_handles += live_vcpus;
            report = Err(ShutdownError::LiveResources {
                resource,
                report: counts,
            });
        }
        drop(self);
        report
    }

    /// Construct with the default interpreter backend (lives in the core).
    pub fn new(config: VmConfig) -> Self {
        Self::with_backend(config, Box::new(InterpreterBackend))
    }

    /// Supply a lockstep trace writer for interpreted execution, including tier-up
    /// warmup. Compiled execution does not produce interpreter trace records.
    pub fn set_lockstep_trace(&mut self, trace: Option<Arc<crate::lockstep::Trace>>) {
        self.lockstep_trace = trace;
    }

    /// Install an optional instruction-fetch policy. `page` is the aligned guest
    /// address of a 4 KiB page, regardless of the host page size. Allowed pages
    /// are remembered until code-page invalidation; refusals are not cached.
    /// Installing or replacing a policy invalidates existing translations.
    ///
    /// Calls occur on the fetching vCPU, never in signal context and with no
    /// engine lock held. The callback may block or make syscalls. Different
    /// vCPUs may call it concurrently; calls for the same page are coalesced.
    /// Invalidation racing a call discards its verdict and asks again. Callbacks
    /// must not recursively fetch code from this VM. As with code invalidation,
    /// embedders must synchronize protection changes with running vCPUs; already
    /// executing instructions are not recalled.
    ///
    /// This does not enforce advisory region `Prot` or model guest paging. A
    /// refusal supplies the embedder's presence fact in the existing #PF
    /// projection and sets `is_fetch`; other paging-control facts remain unknown.
    /// With no policy, compiled execution and dispatch retain their existing paths.
    pub fn set_fetch_policy(
        &mut self,
        policy: Box<dyn Fn(u64) -> crate::FetchVerdict + Send + Sync>,
    ) {
        self.cache.invalidate_pending_fetches = true;
        self.cache
            .invalidate_overlapping(0, u64::MAX, || self.mem.clear_all_code_pages());
        self.backend.invalidate_links();
        self.mem.set_fetch_policy(Arc::from(policy));
    }

    /// Opt into write preflight for compiled stores that can cross a host page.
    /// Disabled by default, preserving the existing generated store paths and
    /// their possibly partial RAM effects on a split host-transfer fault.
    ///
    /// When enabled, a failed probe commits no destination bytes. Protections
    /// must remain stable through the actual stores; probes can trigger COW and
    /// dirty tracking. Interpreter/helpers retain their existing policies.
    /// All vCPUs use this VM policy, and forks inherit it. Changing it invalidates
    /// translations, linked entries and pending background compilations.
    pub fn set_precise_split_stores(&mut self, enabled: bool) {
        if self.precise_split_stores == enabled {
            return;
        }
        self.precise_split_stores = enabled;
        // Advance even with no cached spans: a background compile may be pending.
        let pending = self.cache.invalidate_pending_fetches;
        self.cache.invalidate_pending_fetches = true;
        self.cache
            .invalidate_overlapping(0, u64::MAX, || self.mem.clear_all_code_pages());
        self.cache.invalidate_pending_fetches = pending;
        self.backend.invalidate_links();
    }

    /// Test support only; not for production. Hold each synchronous vCPU compile
    /// section for `duration` (at most one second). The returned flag is true
    /// during the hold, before the backend callback; poll it with Acquire.
    /// Configure before running vCPUs. Zero disables the hold. Forks disable it.
    pub fn test_support_hold_compile(
        &mut self,
        duration: std::time::Duration,
    ) -> Arc<std::sync::atomic::AtomicBool> {
        assert!(duration <= std::time::Duration::from_secs(1));
        let entered = Arc::new(std::sync::atomic::AtomicBool::new(false));
        self.compile_hold = (!duration.is_zero()).then(|| (duration, entered.clone()));
        entered
    }

    fn hold_compile(&self) {
        if let Some((duration, entered)) = &self.compile_hold {
            entered.store(true, Ordering::Release);
            std::thread::sleep(*duration);
            entered.store(false, Ordering::Release);
        }
    }

    /// Test support only; not for production. Synchronously compile and cache a
    /// region at `entry`, even for a single block or a non-loop, without warmup.
    /// The next dispatch to this entry with the given MXCSR control mode uses it.
    /// Compilation fallback is an error, never silent interpreter execution.
    ///
    /// Requires a region-forming backend. Call after writing fixture code and
    /// before running vCPUs. This discards existing translations, links and
    /// pending publications. Normal SMC/fetch
    /// invalidation still applies; this does not pin a translation after changes.
    /// No dispatch checks, counters or generated instructions are installed.
    pub fn test_support_force_region(
        &mut self,
        entry: u64,
        caps: RegionCaps,
        mxcsr: u32,
    ) -> Result<(), ForceRegionError> {
        if self.mode == CpuMode::Real16 {
            return Err(ForceRegionError::UnsupportedMode);
        }
        if self.backend.region_caps().is_none() {
            return Err(ForceRegionError::NotCompiled);
        }
        let pending = self.cache.invalidate_pending_fetches;
        self.cache.invalidate_pending_fetches = true;
        self.cache
            .invalidate_overlapping(0, u64::MAX, || self.mem.clear_all_code_pages());
        self.cache.invalidate_pending_fetches = pending;
        self.backend.invalidate_links();
        let epoch = self.cache.epoch();
        let permit = self.mem.fetch_permit();
        let mut failure = None;
        let region = lift_region_with_fault(&self.mem, entry, caps, self.mode, &mut failure)
            .map_err(|e| {
                ForceRegionError::Lift(lift_exit(
                    e,
                    failure,
                    &self.mem,
                    FetchAddr::flat(entry),
                    self.mode,
                ))
            })?;
        let key = BlockKey::with_mxcsr(entry, self.mode, mxcsr);
        let compiled = self.backend.materialize_region_with_split_stores(
            &region,
            self.consistency,
            self.mem.trap_window(),
            self.mem.guest_base(),
            key.mxcsr_mode,
            self.precise_split_stores,
        );
        if !matches!(compiled, CachedBlock::Compiled { .. }) {
            return Err(ForceRegionError::NotCompiled);
        }
        if !self
            .cache
            .insert_checked(key, compiled, region.spans(), |spans| {
                if self.cache.epoch() != epoch || permit.as_ref().is_some_and(|p| !p.is_current()) {
                    return false;
                }
                for &(start, len) in spans {
                    self.mem.mark_code(start, len);
                }
                true
            })
        {
            return Err(ForceRegionError::Invalidated);
        }
        self.cache.record_region();
        Ok(())
    }

    /// Whether compiled split-store preflight is enabled (default: false).
    pub fn precise_split_stores(&self) -> bool {
        self.precise_split_stores
    }

    /// Enable hotness-gated tier-up: interpret each block until it has run `n`
    /// times, then JIT-compile it. `None` restores eager compilation. Returns
    /// `self` for builder-style setup.
    pub fn set_tier_up_after(&mut self, n: Option<u32>) {
        self.tier_up_after = n;
        // The backend tunes its codegen to this: with tier-up everything it compiles
        // has already proved hot. Reported here rather than read at compile time so a
        // JIT can bake it into an ISA it builds once.
        self.backend.set_tiering(n.is_some());
    }

    /// Enable background tier-up: a hot block is compiled off the vcpu on the
    /// backend's worker thread and swapped in when it lands, so the hot dispatch never
    /// stalls for a compile. Only meaningful together with
    /// [`set_tier_up_after`](Vm::set_tier_up_after) and a backend that runs a
    /// compiler thread; on an `Unsupported` backend it degrades to inline tier-up.
    pub fn set_tier_up_background(&mut self, on: bool) {
        self.tier_up_background = on;
    }

    /// Set the adaptive region tier-up threshold T2: a hot loop tiers up to a
    /// background superblock region only after `Some(n)` executions — a higher bar than
    /// [`set_tier_up_after`](Vm::set_tier_up_after) (T1), so short loops never pay a
    /// wasted region compile. `None` uses T1. Only meaningful with a region-forming
    /// backend + background tier-up.
    pub fn set_tier_up_region_after(&mut self, n: Option<u32>) {
        self.tier_up_region_after = n;
    }

    /// Select the guest CPU feature set that vcpus spawned from this VM start with —
    /// the ISA level CPUID/`xgetbv` advertise. Call before [`new_vcpu`](Vm::new_vcpu).
    /// Default is [`GuestCpuFeatures::default`].
    /// Advertising past what the lifter executes is a documented caller risk — a guest
    /// then traps on the unimplemented instruction (a legal `Exit`).
    pub fn set_guest_cpu_features(&mut self, features: crate::features::GuestCpuFeatures) {
        self.features = features;
    }

    /// Select the x87 pointer/opcode model inherited by new vCPUs.
    /// Defaults to [`Sdm`](crate::state::X87PointerModel::Sdm); independent of
    /// CPUID tables and overrides. Existing vCPUs retain their setting, which the
    /// embedder can edit through [`CpuState::x87_pointer_model`]. VM forks inherit it.
    /// Applies to FXSAVE (both widths) and direct `exec_fxstate` export.
    /// The AMD model also records every non-control instruction opcode.
    /// FNSTENV saves the recorded environment; restore operations are unchanged.
    pub fn set_x87_pointer_model(&mut self, policy: crate::state::X87PointerModel) {
        self.x87_pointer_model = policy;
    }

    /// The x87 pointer/opcode model new vCPUs inherit.
    pub fn x87_pointer_model(&self) -> crate::state::X87PointerModel {
        self.x87_pointer_model
    }

    /// Install a complete CPUID table for new vCPUs. `None` restores the selected
    /// `GuestCpuFeatures` projection. Existing vCPUs retain their CPU description.
    /// A table replaces all CPUID replies; `set_guest_cpu_features` still selects
    /// the fallback XCR0 value unless the vCPU supplies an XCR0 override.
    pub fn set_cpuid_table(&mut self, table: Option<crate::features::CpuidTable>) {
        self.cpuid_table = table.map(Arc::new);
    }

    pub fn cpuid_table(&self) -> Option<&crate::features::CpuidTable> {
        self.cpuid_table.as_deref()
    }

    /// Install time/entropy for new vCPUs; existing vCPUs retain their handle.
    /// `None` selects constant TSC 0x12345678, TSC_AUX zero and deterministic
    /// successful random reads. See [`CpuSource`](crate::state::CpuSource).
    pub fn set_cpu_source(&mut self, source: Option<Arc<dyn crate::state::CpuSource>>) {
        self.cpu_source = source;
    }

    /// Select the x87 transcendental precision new vcpus inherit: `Fast`
    /// (f64/libm, default) or `Extended` (full-80-bit F80). Set before spawning vcpus.
    pub fn set_x87_precision(&mut self, p: crate::state::X87Precision) {
        self.x87_precision = p;
    }

    /// The guest CPU feature set new vcpus inherit.
    pub fn guest_cpu_features(&self) -> crate::features::GuestCpuFeatures {
        self.features
    }

    /// Select the guest decode/lift mode (§17.3) this Vm — and every vcpu spawned from
    /// it — runs under. Call before [`new_vcpu`](Vm::new_vcpu). Default is
    /// [`CpuMode::Long64`]. [`CpuMode::Real16`] always uses the interpreter, regardless
    /// of the injected backend or tier-up/region policy.
    pub fn set_cpu_mode(&mut self, mode: CpuMode) {
        self.mode = mode;
    }

    /// The guest decode/lift mode new vcpus inherit (§17.3).
    pub fn cpu_mode(&self) -> CpuMode {
        self.mode
    }

    /// Fork this VM: a child with an independent deep-copy of guest memory (§4.2),
    /// a fresh translation cache, and the given backend, inheriting the consistency
    /// tier and tier-up policy. The guest-agnostic primitive behind an OS `fork` —
    /// the embedder clones the vcpu's `CpuState` (child RAX = 0) and drives the
    /// child. Knows nothing about processes, pids, or fds.
    ///
    /// `None` when the guest memory can't be deep-copied by the core — a host-backed
    /// (embedder-`mmap`ed) `Reserved` span, which only the embedder can re-allocate.
    /// The embedder surfaces that as a typed error to the guest rather than aborting.
    pub fn fork_with_backend(&self, backend: Box<dyn Backend>) -> Option<Vm> {
        Some(Vm {
            vcpu_lifetime: Arc::new(()),
            precise_split_stores: self.precise_split_stores,
            compile_hold: None,
            mem: self.mem.deep_copy()?,
            cache: {
                let mut cache = TranslationCache::new();
                cache.invalidate_pending_fetches = self.mem.fetch_policy.is_some();
                cache
            },
            backend,
            consistency: self.consistency,
            tier_up_after: self.tier_up_after,
            tier_up_background: self.tier_up_background,
            tier_up_region_after: self.tier_up_region_after,
            features: self.features,
            x87_pointer_model: self.x87_pointer_model,
            cpuid_table: self.cpuid_table.clone(),
            cpu_source: self.cpu_source.clone(),
            source_ids: self.source_ids.clone(),
            x87_precision: self.x87_precision,
            mode: self.mode,
            lockstep_trace: self.lockstep_trace.clone(),
        })
    }

    /// Assemble a fresh `Vm` around already-built guest memory — the shared body of the
    /// public constructors, holding the default tier-up policy and CPU features in one place.
    fn from_mem(mem: Memory, backend: Box<dyn Backend>, consistency: MemConsistency) -> Self {
        Self {
            vcpu_lifetime: Arc::new(()),
            precise_split_stores: false,
            compile_hold: None,
            mem,
            cache: TranslationCache::new(),
            backend,
            consistency,
            tier_up_after: None,
            tier_up_background: false,
            tier_up_region_after: None,
            features: crate::features::GuestCpuFeatures::default(),
            x87_pointer_model: crate::state::X87PointerModel::default(),
            cpuid_table: None,
            cpu_source: None,
            source_ids: Arc::new(Mutex::new(SourceIds::default())),
            x87_precision: crate::state::X87Precision::default(),
            mode: CpuMode::Long64,
            lockstep_trace: None,
        }
    }

    /// Construct with an injected backend — this is how the JIT gets in (§4.1).
    pub fn with_backend(config: VmConfig, backend: Box<dyn Backend>) -> Self {
        Self::from_mem(
            Memory::new(config.memory_model),
            backend,
            config.consistency,
        )
    }

    /// Like [`Vm::with_backend`] but for a `Reserved` model backed by an
    /// embedder-provided host mapping (a `MAP_NORESERVE` span the core can't allocate
    /// itself). `config.memory_model` should be `Reserved { span: ram.len }`.
    /// The host pointer must equal `ram.guest_base` modulo 16. Returns an error
    /// for misaligned backing or a backing that cannot contain the guest extent.
    pub fn with_backend_host_ram(
        config: VmConfig,
        backend: Box<dyn Backend>,
        ram: HostRam,
    ) -> Result<Self, crate::memory::HostRamError> {
        Ok(Self::from_mem(
            Memory::from_host_ram(config.memory_model, ram)?,
            backend,
            config.consistency,
        ))
    }

    pub fn map(
        &mut self,
        guest_addr: u64,
        size: usize,
        prot: Prot,
        kind: RegionKind,
    ) -> Result<(), MapError> {
        self.mem.map(guest_addr, size, prot, kind)?;
        // Mapping a Trap (MMIO) region changes the compile-time window a JIT bakes
        // into its Trap-range check (§5.2). Any block already compiled with
        // a narrower (or empty) window would miss the new region, so drop the whole
        // cache; MMIO regions are set up rarely (usually before execution), making
        // this near-free. `unmap` handles the shrinking case.
        if kind == RegionKind::Trap {
            self.cache.invalidate_overlapping(0, u64::MAX, || {
                self.mem.clear_all_code_pages();
            });
            self.backend.invalidate_links();
        }
        Ok(())
    }

    pub fn write_bytes(&self, guest_addr: u64, bytes: &[u8]) -> Result<(), MemError> {
        self.mem.write_bytes(guest_addr, bytes)
    }

    pub fn read_bytes(&self, guest_addr: u64, buf: &mut [u8]) -> Result<(), MemError> {
        self.mem.read_bytes(guest_addr, buf)
    }

    /// Register a watched guest DATA range: guest writes to it are recorded
    /// and drained by [`Self::take_dirty_ranges`], independent of SMC code-page
    /// tracking. For an embedder that caches guest-backed resources (e.g. a GPU
    /// resource cache) and re-uploads lazily on write. Zero write-path cost when nothing
    /// is watched; poll-and-drain at a frame/submit boundary under `MemConsistency::Fast`.
    pub fn watch_range(&self, guest_addr: u64, size: u64) {
        self.mem.watch_range(guest_addr, size);
    }

    /// Stop watching a guest DATA range previously passed to [`Self::watch_range`].
    pub fn unwatch_range(&self, guest_addr: u64, size: u64) {
        self.mem.unwatch_range(guest_addr, size);
    }

    /// Drain the watched ranges written since the last call, coalesced into
    /// `(guest_addr, byte_len)`. Empty and lock-free when nothing watched was written.
    pub fn take_dirty_ranges(&self) -> Vec<(u64, u64)> {
        self.mem.take_dirty_ranges()
    }

    pub fn unmap(&mut self, guest_addr: u64, size: usize) -> Result<(), MapError> {
        self.mem.unmap(guest_addr, size)?;
        // A block cached from the now-unmapped range must not stay executable (§10):
        // drop every unit overlapping it, clear the code-page tags, and flush the
        // backend's link/IBTC slots — mirroring `handle_smc`. Without this a stale
        // block runs (or a chained edge jumps into it) instead of faulting
        // `Exit::UnmappedMemory`.
        let lo = guest_addr;
        let hi = guest_addr.saturating_add(size as u64);
        // Tags and fetch approvals are page-wide. Invalidate the same whole pages
        // before clearing, including translations in surviving sub-page regions.
        let mask = crate::memory::CODE_PAGE_SIZE - 1;
        let (lo, hi) = (lo & !mask, (hi.saturating_sub(1) | mask).saturating_add(1));
        let dropped = !self
            .cache
            .invalidate_overlapping(lo, hi, || {
                let last = hi.saturating_sub(1);
                for page in
                    (lo >> crate::memory::CODE_PAGE_BITS)..=(last >> crate::memory::CODE_PAGE_BITS)
                {
                    self.mem.clear_code_page(page);
                }
            })
            .is_empty();
        if dropped {
            self.backend.invalidate_links();
        }
        Ok(())
    }

    /// Process pending self-modifying-code writes (§10): for each code page a
    /// store landed on, drop every cached block overlapping it and clear the tag
    /// (re-execution re-lifts and re-tags). No-op unless a write actually hit a
    /// code page.
    ///
    /// Interpreter stores, compiled stores, helpers and embedder writes all report
    /// through `Memory::note_write`, which expands declared aliases to code pages.
    fn handle_smc(&self, word: &AtomicU64) {
        self.mem.dispatch_smc(word, |dirty| {
            let mut invalidated = false;
            for page in dirty {
                let lo = page << crate::memory::CODE_PAGE_BITS;
                let hi = lo + (1 << crate::memory::CODE_PAGE_BITS);
                // A dropped unit's inbound link slots (in other, surviving blocks) still
                // point at its now-stale compiled code. Note whether anything was
                // dropped so we can clear the backend's slots once, below.
                // The page tag is cleared *inside* `invalidate_overlapping`, under the
                // spans lock and only if no block still spans the page — so it can't race
                // a concurrent insert's mark.
                invalidated |= !self
                    .cache
                    .invalidate_overlapping(lo, hi, || self.mem.clear_code_page(page))
                    .is_empty();
            }
            // Clear all backend-owned cached code pointers so no surviving block chains
            // into a dropped unit. Cleared slots re-link on their next traversal.
            if invalidated {
                self.backend.invalidate_links();
            }
        });
    }

    /// One execution context per guest thread (§4.3). Shares this `Vm`.
    ///
    /// # Panics
    /// Panics if automatic source identities are exhausted; identities never wrap.
    pub fn new_vcpu(&self) -> Vcpu {
        self.new_vcpu_with_cpuid_overrides(crate::features::CpuidOverrides::default())
    }

    /// Create a vCPU with an embedder-selected initial guest MXCSR, for example
    /// `0x9fc0` to preserve FTZ and DAZ across a thread's initial state.
    /// The ordinary constructor uses the reset value `0x1f80` (SDM Vol 1 §11.6.4).
    /// Mode bits are stored but do not yet govern SIMD arithmetic.
    ///
    /// # Panics
    /// Panics if any reserved bit (31:16) is set; see [`Vcpu::set_mxcsr`].
    /// Also panics if automatic source identities are exhausted.
    pub fn new_vcpu_with_mxcsr(&self, mxcsr: u32) -> Vcpu {
        let mut cpu = self.new_vcpu();
        cpu.set_mxcsr(mxcsr);
        cpu
    }

    /// Create a vCPU with its logical-processor and runtime CPUID fields.
    /// Overrides are owned by `CpuState`, so cloning state preserves them.
    /// For an explicit source identity, use [`Self::new_vcpu_with_source_id`]
    /// and assign [`CpuState::cpuid_overrides`] before execution.
    ///
    /// # Panics
    /// Panics if automatic source identities are exhausted; identities never wrap.
    pub fn new_vcpu_with_cpuid_overrides(
        &self,
        overrides: crate::features::CpuidOverrides,
    ) -> Vcpu {
        let id = self.source_ids.lock().unwrap().allocate(None);
        let id = id.expect("vCPU source identities exhausted");
        self.make_vcpu(id, overrides)
    }

    /// Create a vCPU with an explicit [`CpuState::source_id`]. Rejects any ID
    /// already allocated by this VM or a VM sharing its allocator through
    /// [`Self::fork_with_backend`], including IDs whose vCPUs have been dropped.
    /// Success advances automatic allocation past `id` if necessary. An unused
    /// ID below the current automatic cursor is allowed, including after seeding.
    /// All constructors serialize ID allocation; no bookkeeping runs during execution.
    /// Direct changes to public CPU state and state clones are outside this registry.
    ///
    /// The ID is the key passed to both time and random methods on
    /// [`CpuSource`](crate::state::CpuSource), whose
    /// [`TscSample::aux`](crate::state::TscSample::aux) supplies RDTSCP's ECX
    /// (SDM Vol 3B §20.17.2). Install the source with [`Self::set_cpu_source`]
    /// before construction, or replace [`CpuState::cpu_source`] between runs.
    /// The default source returns AUX zero regardless of the source ID.
    /// [`Self::set_cpuid_table`] supplies the shared CPUID description;
    /// [`Self::new_vcpu_with_cpuid_overrides`] or [`CpuState::cpuid_overrides`]
    /// supplies per-vCPU IDs separately:
    /// leaf 1 `EBX[31:24]` carries the 8-bit initial APIC ID; leaves 0xB/0x1F EDX
    /// carry the 32-bit x2APIC ID on each configured subleaf (SDM Vol 1 §21.1.4,
    /// §21.3, Tables 21-8/21-31/21-70). The embedder chooses the mapping from
    /// the 64-bit source identity to those narrower fields and supplies topology.
    /// RDPID also reads TSC_AUX (SDM Vol 2B §4.3, RDPID, Description), but is
    /// currently unsupported by the lifter.
    pub fn new_vcpu_with_source_id(&self, id: u64) -> Result<Vcpu, SourceIdError> {
        self.source_ids.lock().unwrap().allocate(Some(id))?;
        Ok(self.make_vcpu(id, crate::features::CpuidOverrides::default()))
    }

    /// Next automatic source ID, or `None` after consuming `u64::MAX`.
    /// Quiesce vCPU creation before taking a snapshot for a rebuild.
    pub fn next_source_id(&self) -> Option<u64> {
        self.source_ids.lock().unwrap().next
    }

    /// Reserve the prefix below `next` from future automatic allocation.
    /// Equal cursors are accepted; rewinds and exhaustion are reported without
    /// changing the allocator. Explicit allocation of unused IDs below `next`
    /// remains possible. VM forks share this cursor and the allocated-ID registry.
    ///
    /// To rebuild after a host fork, quiesce creation, save [`Self::next_source_id`]
    /// and the surviving vCPU's source ID, then on a fresh VM call this method
    /// with the saved cursor and [`Self::new_vcpu_with_source_id`] for the survivor.
    /// Install the shared CPUID table and CPU source before construction, then
    /// restore the survivor's CPU state (including CPUID overrides and source
    /// handle) before execution. If replacing state wholesale, keep its source ID
    /// equal to the ID registered by the constructor.
    /// The next automatic vCPU then receives the saved cursor.
    pub fn set_next_source_id(&self, next: u64) -> Result<(), SourceIdError> {
        let mut ids = self.source_ids.lock().unwrap();
        let current = ids.next.ok_or(SourceIdError::Exhausted)?;
        if next < current {
            return Err(SourceIdError::Rewind {
                current,
                requested: next,
            });
        }
        ids.next = Some(next);
        Ok(())
    }

    fn make_vcpu(&self, id: u64, overrides: crate::features::CpuidOverrides) -> Vcpu {
        let mut cpu = CpuState::new();
        cpu.features = self.features; // ISA level the embedder chose
        cpu.x87_pointer_model = self.x87_pointer_model;
        cpu.cpuid_table = self.cpuid_table.clone();
        cpu.cpuid_overrides = overrides;
        cpu.cpu_source = self.cpu_source.clone();
        cpu.source_id = id;
        cpu.x87_precision = self.x87_precision; // transcendental precision
        let word = Arc::new(AtomicU64::new(0));
        self.mem.register_dispatch_word(&word);
        self.backend.register_dispatch_word(&word);
        Vcpu {
            _vm_lifetime: self.vcpu_lifetime.clone(),
            cpu,
            exit_request: ExitRequest { word },
            interruption_evidence: None,
            syscall_evidence: None,
            engine_sections: AtomicU64::new(0),
            mode: self.mode, // decode/lift mode the embedder chose (§17.3)
            fast: Box::new(
                [FastEntry {
                    rip: 0,
                    entry: CompiledPtr(std::ptr::null()),
                }; FAST_N],
            ),
            fast_epoch: self.cache.epoch(),
            fast_mxcsr_mode: 0,
            ibtc_refills: HashMap::new(),
            ret_stack: Box::new(RetStack::new()),
            fault_entry: Box::default(),
            fast_hits: 0,
            chained: 0,
            executed: 0,
            interp_scratch: Vec::new(),
            pending_irq: None,
            retired: 0,
            sti_shadow: false,
            code_space_reported: false,
        }
    }
}

impl Drop for Vm {
    fn drop(&mut self) {
        // Join before releasing either the cache or guest-memory views.
        let _ = self.backend.shutdown();
    }
}

/// Number of slots in the per-vcpu fast-resolve cache. A power of two so the
/// index is a mask. 1024 entries × 16 bytes = 16 KiB per vcpu — cheap.
const FAST_BITS: u32 = 10;
const FAST_N: usize = 1 << FAST_BITS;

/// After this many IBTC refills a site is treated as megamorphic: the
/// dispatcher stops refilling its slot (it stays empty → the site pays the
/// baseline dispatch forever) so a polymorphic indirect branch can't churn the
/// descriptor arena without bound.
const IBTC_MEGAMORPHIC_CAP: u32 = 8;

/// One direct-mapped fast-resolve entry: a guest RIP tag and its compiled entry.
/// A null `CompiledPtr` marks the slot empty (guest RIP 0 never collides with a
/// real block because an empty slot's pointer is null, checked first).
#[derive(Copy, Clone)]
struct FastEntry {
    rip: u64,
    entry: CompiledPtr,
}

pub(crate) const EXIT_REQUESTED: u64 = 2;

/// Backend-to-vcpu notification in the dispatch word shared with SMC and exits.
pub const CODE_SPACE_EXHAUSTED: u64 = 4;

/// Thread-safe, cloneable request to return from a vcpu's next running boundary.
/// Requests coalesce until consumed by `run`; requests made outside `run` persist.
/// A running vcpu checks within one compiled block or chain edge, one region
/// sub-block, or one REP iteration. A handle that outlives its vcpu has no effect.
/// Everything written by a thread before `request_exit` is visible to the caller
/// once `run` returns the `Exit::Interrupted` that consumed that request.
#[derive(Clone, Debug)]
pub struct ExitRequest {
    word: Arc<AtomicU64>,
}

// `request_exit`'s async-signal-safety rests on a native, lock-free 64-bit RMW.
#[cfg(not(target_has_atomic = "64"))]
compile_error!("ExitRequest::request_exit needs a lock-free 64-bit atomic");

impl ExitRequest {
    /// Ask the vcpu to return [`Exit::Interrupted`] with resumable guest state.
    /// Checks occur within one compiled block or chain edge, one region sub-block,
    /// or one REP iteration. A handle outliving its vcpu has no effect.
    /// Writes preceding this call are visible after `run` returns the
    /// [`Exit::Interrupted`] that consumes it. Engine work defers the request until
    /// its locks have been released; [`Vcpu::interruption_evidence`] reports this.
    ///
    /// Async-signal-safe: this call is one lock-free atomic read-modify-write on
    /// a word the handle already owns. It does not allocate, lock, touch the
    /// reference count or call into libc, so a host signal handler may call it
    /// on a handle obtained (and kept alive) before the signal can arrive.
    /// Cloning or dropping an `ExitRequest` is not async-signal-safe: do both
    /// outside the handler.
    pub fn request_exit(&self) {
        // Release publishes preceding writes; run's consuming Acquire pairs with
        // this RMW. Preserve SMC and leave requests after consumption pending.
        self.word.fetch_or(EXIT_REQUESTED, Ordering::Release);
    }
}

/// Per-guest-thread execution context: CPU state + its own `run()` loop (§2).
pub struct Vcpu {
    _vm_lifetime: Arc<()>,
    pub cpu: CpuState,
    exit_request: ExitRequest,
    interruption_evidence: Option<crate::InterruptionEvidence>,
    syscall_evidence: Option<crate::ExecutionEvidence>,
    engine_sections: AtomicU64,
    code_space_reported: bool,
    /// Decode/lift mode (§17.3), inherited from the `Vm`: threaded into every
    /// `resolve`/`lift`/`step_one` call and combined with the guest RIP to form the
    /// [`BlockKey`] a translation is cached under (§17.4). Constant for the vcpu's life
    /// (no runtime mode switching — the §17 scope fence).
    mode: CpuMode,
    /// Fast-resolve cache: a vcpu-private, direct-mapped RIP→compiled
    /// entry map that replaces the shared `RwLock<HashMap>` lookup (plus its two
    /// atomic counter bumps) for the transfers the chain loop can't chain — returns,
    /// indirect jumps, and cold outer-loop re-dispatch. Only `Compiled` entries are
    /// cached; interpreted blocks always route through `resolve` (so the cache
    /// counters stay meaningful). Flushed whenever the cache invalidation epoch
    /// moves — the coherence channel for cross-thread SMC that `invalidate_links`
    /// (backend-owned slots) cannot reach.
    fast: Box<[FastEntry; FAST_N]>,
    /// Snapshot of `TranslationCache::epoch` matching the current `fast` contents.
    fast_epoch: u64,
    fast_mxcsr_mode: u32,
    /// Per-IBTC-slot refill count, keyed by slot address. Guards against a
    /// megamorphic indirect site churning the descriptor arena: once a slot hits
    /// [`IBTC_MEGAMORPHIC_CAP`] refills the dispatcher stops filling it. Cleared on
    /// an invalidation-epoch change alongside the fast cache.
    ibtc_refills: HashMap<u64, u32>,
    /// Shadow return stack: compiled `call`s push predicted returns
    /// here, compiled `ret`s pop and chain on a match. Persists across `run()`
    /// calls (syscall exits re-enter `run()` constantly); its `sp` resets on an
    /// invalidation-epoch change. Boxed for a stable address and a small `Vcpu`.
    ret_stack: Box<RetStack>,
    fault_entry: Box<crate::jit_abi::fault::EntryRecord>,
    /// Fast-resolve cache hits over this vcpu's lifetime. A plain counter, not atomic —
    /// a shared atomic here would reintroduce exactly the contention the private cache
    /// removes. Read via [`Vcpu::fast_hits`].
    fast_hits: u64,
    /// Guest x86 instructions executed since the last flush, on BOTH paths: the
    /// interpreter adds what it retired, compiled code adds each block's
    /// `IrBlock::icount` on entry via `MemCtx.icount`. Deliberately separate from
    /// `retired`, which is a deterministic virtual-time base for a scheduler and must
    /// keep its interpreter-only, per-instruction granularity. This one answers "how
    /// much guest work actually ran" — guest IPC against wall time, and
    /// `executed / chained` for the average compiled-unit length.
    executed: u64,
    /// Chained block transfers taken since the last flush. Plain and private for the
    /// same reason as `fast_hits`, and far more urgently: chaining is the engine's
    /// hottest event (~1M per frame in a real game vs ~5k indirect misses), so it
    /// must not touch a shared line. `run` flushes it into the shared counter.
    chained: u64,
    /// Reused temps buffer for `interpret_block` — grows to the largest block's
    /// `temp_count` and is zero-filled per block, avoiding a per-dispatch allocation.
    interp_scratch: Vec<u64>,
    /// Pending maskable hardware-interrupt vector (§17.6, sub-seam c). Set by
    /// [`Vcpu::inject_irq`]; delivered at a `run()` boundary through the real-mode IVT
    /// once IF is set, no memory/port completion is outstanding, and the STI shadow has
    /// elapsed (see [`Vcpu::run`]). A plain `Vcpu` field, NOT in `#[repr(C)] CpuState`:
    /// it is embedder-facing async state, never read by compiled code, so it stays off
    /// the `jit_abi` layout entirely. Real-mode only (Long64/Compat32 never inject).
    /// Stays queued while blocked — never silently dropped.
    pending_irq: Option<u8>,
    /// Retired-instruction counter (§17.6, sub-seam c): a monotone `u64` bumped once per
    /// guest instruction that **retires** (completes) on the INTERPRETER path. This is
    /// the deterministic virtual-time base the embedder schedules against; it never reads
    /// a wall clock. Because Real16 is interpreter-only (the JIT/region tier is never
    /// reached in real mode) this counts every real-mode instruction. Compiled
    /// Long64/Compat32 blocks do NOT tick it — charging retirement inside compiled code
    /// would need codegen changes we deliberately avoid — so on that embedder's path it counts
    /// only the occasional interpreter single-step (MMIO retry). Read via
    /// [`Vcpu::retired_instructions`].
    retired: u64,
    /// STI-shadow latch (§17.6, sub-seam c). `sti` masks interrupt delivery for the
    /// duration of the *following* instruction, so `sti; hlt` and `sti; cli` behave
    /// atomically on real hardware. Set true when the just-run interpreted block ended
    /// with `sti` as its final retired instruction (the interpreter reports this); while
    /// set, a pending IRQ is held for one more block-dispatch boundary. Cleared once the
    /// next block runs an instruction (which clears the shadow). Delivery at a boundary is
    /// blocked while this is set. See [`Vcpu::run`].
    sti_shadow: bool,
}

impl Vcpu {
    /// Obtain a handle that may request an exit from any host thread.
    pub fn exit_request(&self) -> ExitRequest {
        self.exit_request.clone()
    }

    /// Evidence for the [`Exit::Interrupted`] returned by the most recent [`Self::run`].
    /// `None` before the first run, during a run, or after any other exit. Cleared
    /// as soon as the next `run` is entered, even if it unwinds. The returned value
    /// is a copy and can be retained across resumption. Deferred evidence for a
    /// still-pending request survives unrelated exits internally.
    pub fn interruption_evidence(&self) -> Option<crate::InterruptionEvidence> {
        self.interruption_evidence
    }

    /// Evidence for the [`Exit::Syscall`] returned by the most recent [`Self::run`].
    /// The RIP identifies the syscall instruction, while `cpu.rip` points past it.
    /// `None` before the first run or after any other exit. Cleared as soon as
    /// the next `run` is entered, even if it unwinds; the copy can be retained.
    /// Chain-link evidence is not collected for syscalls.
    pub fn syscall_evidence(&self) -> Option<crate::ExecutionEvidence> {
        self.syscall_evidence
    }

    /// Fast-resolve cache hits served without a shared-cache lookup.
    pub fn fast_hits(&self) -> u64 {
        self.fast_hits
    }

    /// Queue a pending maskable **hardware interrupt** for real-mode delivery (§17.6,
    /// sub-seam c). `vector` is the IVT entry the embedder's PIC resolved (INTA);
    /// x86jit does NOT model a PIC/8259 — computing the vector and updating the PIC's
    /// in-service/ISR state on acknowledge is the **embedder's** responsibility, done
    /// before it calls this. The interrupt is delivered at the next [`Vcpu::run`]
    /// boundary (never mid-block) once ALL hold: IF is set, no memory/port completion is
    /// outstanding (`pending_mmio`/`pending_mmio_write`/`pending_port_in`), and the STI
    /// shadow has elapsed. If injection is currently blocked the vector stays queued —
    /// it is never silently dropped — and fires at the first boundary the gates open,
    /// including the boundary after a `hlt` wakeup. Only one interrupt is queued at a
    /// time (a real PIC re-asserts INTR for further lines after the embedder EOIs, so a
    /// later `inject_irq` overwrites a still-pending vector). Real-mode only.
    pub fn inject_irq(&mut self, vector: u8) {
        self.pending_irq = Some(vector);
    }

    /// Whether an injected hardware-interrupt vector is currently queued and not yet
    /// delivered (§17.6, sub-seam c) — e.g. still masked by IF, held by the STI shadow,
    /// or waiting on a memory/port completion. The embedder can consult this before
    /// re-injecting; a fresh [`Vcpu::inject_irq`] overwrites a still-queued vector.
    pub fn has_pending_irq(&self) -> bool {
        self.pending_irq.is_some()
    }

    /// The retired-instruction counter (§17.6, sub-seam c): guest instructions that have
    /// **retired** (completed) on the interpreter path since this vcpu was created. This
    /// is a deterministic, wall-clock-free virtual-time base for the embedder's scheduler.
    /// In real mode (interpreter-only) it counts every executed instruction; compiled
    /// Long64/Compat32 blocks do not tick it (see the field docs). Monotone, never reset.
    pub fn retired_instructions(&self) -> u64 {
        self.retired
    }

    /// Direct-mapped index for `rip` (Fibonacci hash — one multiply, good spread
    /// even for densely-packed short blocks).
    #[inline]
    fn fast_index(rip: u64) -> usize {
        (rip.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - FAST_BITS)) as usize
    }

    /// Probe the fast-resolve cache; `Some(entry)` on a hit.
    #[inline]
    fn fast_get(&self, rip: u64) -> Option<CompiledPtr> {
        let e = self.fast[Self::fast_index(rip)];
        if !e.entry.0.is_null() && e.rip == rip {
            Some(e.entry)
        } else {
            None
        }
    }

    /// Install a compiled entry for `rip` (overwrites any collision — a stale tag
    /// can only cost a miss, never a wrong transfer, and within an epoch a RIP maps
    /// to a stable compiled entry).
    #[inline]
    fn fast_put(&mut self, rip: u64, entry: CompiledPtr) {
        self.fast[Self::fast_index(rip)] = FastEntry { rip, entry };
    }

    /// Drop every fast-resolve entry and reset the IBTC refill counts (on an
    /// invalidation-epoch change). The backend already zeroed the slots
    /// (`invalidate_links`), so a previously-capped site gets a fresh chance to
    /// re-cache against the rewritten code.
    fn fast_clear(&mut self) {
        for e in self.fast.iter_mut() {
            e.entry = CompiledPtr(std::ptr::null());
        }
        self.ibtc_refills.clear();
        // Drop return predictions too (R5). Not required for correctness — the
        // `ret` addr-compare already guards every prediction — but it keeps the ring
        // from carrying frames whose continuation slots were just zeroed.
        self.ret_stack.sp = 0;
    }
}

impl Vcpu {
    /// Set a guest register. GPRs route through the central index map (§3.1);
    /// RIP and the FS/GS bases live in their own `CpuState` fields (§4.3).
    /// Size-dependent GPR write semantics (32-bit zeroing) are an M1 concern in
    /// the lift's write path — this API sets the full 64-bit value.
    pub fn set_reg(&mut self, reg: Reg, val: u64) {
        match reg.gpr_index() {
            Some(i) => self.cpu.gpr[i] = val,
            None => match reg {
                Reg::Rip => {
                    self.cpu.clear_mmio_parts();
                    self.cpu.rip = val;
                }
                Reg::FsBase => self.cpu.fs_base = val,
                Reg::GsBase => self.cpu.gs_base = val,
                // Real-mode segment selectors (§17.6): the embedder seeds these before a
                // Real16 run; the base is `selector << 4`.
                Reg::Cs => self.cpu.cs = val as u16,
                Reg::Ds => self.cpu.ds = val as u16,
                Reg::Es => self.cpu.es = val as u16,
                Reg::Ss => self.cpu.ss = val as u16,
                _ => unreachable!("gpr_index() only returns None for Rip/FsBase/GsBase/segments"),
            },
        }
    }

    /// Read a guest register. Mirror of [`Self::set_reg`]. (§4.3)
    pub fn reg(&self, reg: Reg) -> u64 {
        match reg.gpr_index() {
            Some(i) => self.cpu.gpr[i],
            None => match reg {
                Reg::Rip => self.cpu.rip,
                Reg::FsBase => self.cpu.fs_base,
                Reg::GsBase => self.cpu.gs_base,
                Reg::Cs => self.cpu.cs as u64,
                Reg::Ds => self.cpu.ds as u64,
                Reg::Es => self.cpu.es as u64,
                Reg::Ss => self.cpu.ss as u64,
                _ => unreachable!("gpr_index() only returns None for Rip/FsBase/GsBase/segments"),
            },
        }
    }

    pub fn set_flags(&mut self, flags: Flags) {
        self.cpu.flags = flags;
    }

    pub fn flags(&self) -> Flags {
        self.cpu.flags
    }

    pub fn set_xmm(&mut self, index: usize, value: u128) {
        self.cpu.xmm[index] = value;
    }

    pub fn xmm(&self, index: usize) -> u128 {
        self.cpu.xmm[index]
    }

    /// Upper 128 bits of YMM `index`.
    pub fn set_ymm_hi(&mut self, index: usize, value: u128) {
        self.cpu.ymm_hi[index] = value;
    }

    pub fn ymm_hi(&self, index: usize) -> u128 {
        self.cpu.ymm_hi[index]
    }

    /// Bits 511:256 of ZMM `index`: `half` 0 = 383:256, 1 = 511:384.
    pub fn set_zmm_hi(&mut self, index: usize, half: usize, value: u128) {
        self.cpu.zmm_hi[index][half] = value;
    }

    pub fn zmm_hi(&self, index: usize, half: usize) -> u128 {
        self.cpu.zmm_hi[index][half]
    }

    /// Opmask register k`index` (k0–k7).
    pub fn set_kmask(&mut self, index: usize, value: u64) {
        self.cpu.kmask[index] = value;
    }

    pub fn kmask(&self, index: usize) -> u64 {
        self.cpu.kmask[index]
    }

    /// Raw 10-byte 80-bit value of the PHYSICAL x87 register `index` (0..8), i.e.
    /// `fpr[index]`. `ST(i)` is `fpr[(fpu_top() + i) & 7]`; callers that
    /// want architectural order rotate by [`Self::fpu_top`].
    pub fn fpr_bytes(&self, index: usize) -> [u8; 10] {
        self.cpu.fpr[index]
    }

    /// Set the physical x87 register `index` from a raw 10-byte 80-bit value.
    pub fn set_fpr_bytes(&mut self, index: usize, bytes: &[u8; 10]) {
        self.cpu.fpr[index] = *bytes;
    }

    /// The x87 stack-top pointer: the physical register that is `ST(0)`.
    pub fn fpu_top(&self) -> u32 {
        self.cpu.fpu_top
    }

    pub fn set_fpu_top(&mut self, top: u32) {
        self.cpu.fpu_top = top & 7;
    }

    /// The x87 control word (round-trips `fldcw`/`fnstcw`).
    pub fn fpu_cw(&self) -> u16 {
        self.cpu.fpu_cw
    }

    pub fn set_fpu_cw(&mut self, cw: u16) {
        self.cpu.fpu_cw = cw;
    }

    /// Guest SSE control/status register, including the stored sticky flags
    /// (SDM Vol 1 §10.2.3). Round-trips through LDMXCSR/STMXCSR and FXSAVE/FXRSTOR;
    /// RC/FTZ/DAZ govern SIMD arithmetic; arithmetic does not set sticky flags.
    pub fn mxcsr(&self) -> u32 {
        self.cpu.mxcsr
    }

    /// Replace guest MXCSR between runs, for example when restoring thread state.
    /// This does not change host floating-point control state.
    ///
    /// # Panics
    /// Panics without changing MXCSR if any reserved bit (31:16) is set
    /// (SDM Vol 1 §§10.2.3, 11.6.6). Guest instructions report #GP instead.
    pub fn set_mxcsr(&mut self, mxcsr: u32) {
        assert_eq!(mxcsr & !crate::state::MXCSR_MASK, 0, "reserved MXCSR bits");
        self.cpu.mxcsr = mxcsr;
    }

    /// Deliver an MMIO read result after `Exit::MmioRead`, then resume (§5.2). The
    /// block re-executes from the faulting instruction (RIP was left there), and its
    /// first load consumes this value instead of re-trapping. Stored on `CpuState`,
    /// not a temp (temps die when the block returns). Interpreter path today —
    /// JIT-side MMIO trap/resume is deferred, and the JIT never emits an
    /// `Exit::MmioRead` for an inlined load, so this is only reached under the interp.
    pub fn complete_mmio_read(&mut self, value: u64) {
        // Preserve the owner recorded at the exit, even if public state changed
        // before completion. The next interpreter entry validates that owner.
        if !self.cpu.mmio_retry_active {
            self.cpu.begin_mmio_retry(self.cpu.rip);
        }
        self.cpu.pending_mmio = Some(value);
    }

    /// Acknowledge an `Exit::MmioWrite` after performing its side effect, then
    /// resume (§5.2). Symmetric to [`Self::complete_mmio_read`]: RIP was left on the
    /// faulting store, so the block re-executes from it; this flag tells the retried
    /// store the write is already done — skip it (no re-trap) and continue. Because
    /// the instruction re-runs, its non-store effects (RSP for `push`, flags) commit
    /// exactly once, preserving instruction atomicity (§7 pitfall 0). Interpreter
    /// path today; JIT-side MMIO is deferred.
    ///
    /// A read-modify-write to a `Trap` region (`add [mmio], reg`) is not supported:
    /// on retry its load re-traps as a fresh `MmioRead`. Model such a device with a
    /// pure load then a pure store instead.
    pub fn complete_mmio_write(&mut self) {
        if !self.cpu.mmio_retry_active {
            self.cpu.begin_mmio_retry(self.cpu.rip);
        }
        self.cpu.pending_mmio_write = true;
    }

    /// Deliver the value read from a port after `Exit::PortIo { dir: In, .. }`, then
    /// resume (§5.2). Unlike MMIO, RIP already advanced past the `in` (as for a
    /// syscall), so this only writes the accumulator — the next `run()` continues
    /// from the following instruction. The write honours x86 sub-register semantics
    /// (32-bit zeroes the upper 32; 16/8-bit merge), reusing the central GPR write
    /// path. Only the low `size` bytes of `value` are used. Calling this without an
    /// outstanding `in` (e.g. after an `out`) is a no-op.
    pub fn complete_port_in(&mut self, value: u64) {
        if let Some(size) = self.cpu.pending_port_in.take() {
            self.cpu
                .write_gpr(Reg::Rax.gpr_index().unwrap(), value, size);
        }
    }

    /// Interpret **exactly one** guest instruction at the current RIP and return how
    /// it ended, without running the following instruction. Always uses the
    /// interpreter (never the JIT), so it is the per-instruction primitive a
    /// single-step oracle needs — e.g. the TomHarte / SingleStepTests 8088 corpus,
    /// whose tests are one instruction each (`x86jit-tests/src/harte.rs`).
    ///
    /// Real-mode CPU exceptions are delivered in-guest here exactly as [`Vcpu::run`]
    /// does (§17.6, sub-seam b): a `#DE` (divide error) or a lifted `int n`/`int3`/
    /// `ud2` vectors through the IVT and returns `Continue` with RIP/CS on the
    /// handler — it does NOT escape as `Exit::Exception`. In Long64/Compat32 an
    /// exception still surfaces as `Exit::Exception`. An unsupported opcode returns
    /// `Exit::UnknownInstruction`; a memory/MMIO/port trap returns its usual exit
    /// (this primitive does not resume a deferred access — call [`Vcpu::run`] for
    /// that).
    pub fn step_instruction(&mut self, vm: &Vm) -> StepResult {
        if vm.mem.fetch_policy.is_some() {
            vm.handle_smc(&self.exit_request.word);
        }
        let _fp_mode = crate::jit_abi::fp::RunMode::new();
        let mut info = crate::interp::RetireInfo {
            consistency: vm.consistency,
            ..Default::default()
        };
        let r = crate::interp::step_one(
            &vm.mem,
            &mut self.cpu,
            self.mode,
            &mut self.interp_scratch,
            &mut info,
        );
        self.retired += info.retired;
        self.sti_shadow = info.sti_shadow;
        let result = match r {
            // §17.6 (sub-seam b): re-deliver a real-mode exception through the IVT,
            // mirroring the interpreted-block arm of `run`.
            StepResult::Exit(Exit::Exception { addr, vector, .. })
                if self.mode == CpuMode::Real16 =>
            {
                crate::interp::deliver_interrupt(
                    &mut self.cpu,
                    &vm.mem,
                    addr,
                    vector,
                    addr,
                    vm.consistency,
                )
                .into_step()
            }
            other => other,
        };
        let result = match result {
            StepResult::Exit(exit) if self.mode == CpuMode::Real16 => {
                StepResult::Exit(crate::fault::qualify_real16(exit, &self.cpu))
            }
            other => other,
        };
        let result = match result {
            StepResult::Exit(exit @ (Exit::UnmappedMemory { .. } | Exit::Exception { .. })) => {
                StepResult::Exit(self.fault_entry.finish_transfer_fault(exit, &vm.mem))
            }
            other => other,
        };
        self.cpu.finish_mmio_retry(match &result {
            StepResult::Exit(exit) => Some(exit),
            StepResult::Continue => None,
        });
        result
    }

    /// Obtain a stable signal-handler handle. It remains valid until this vCPU is
    /// dropped, including across moves. See `jit_abi::fault::classify` for safety.
    pub fn fault_handle(&self) -> crate::jit_abi::fault::FaultHandle {
        self.fault_entry.handle()
    }

    /// Execute until an exit event or budget exhaustion (§5.1, §9.2).
    /// `budget` is measured in blocks (§5.1 recommendation).
    ///
    /// Compiled blocks are chained (§12 M5): a direct edge whose link slot is
    /// filled hands the next entry back via `MemCtx.next_entry` and the inner loop
    /// jumps straight there, skipping the cache lookup. The budget still ticks per
    /// block, so a tight chained loop yields `BudgetExhausted` (preemption, §9.2).
    /// A host request yields `Interrupted` at the next checked boundary; REP may
    /// have made partial progress and resumes at the string instruction. Evidence
    /// is available through [`Self::interruption_evidence`] until `run` is next entered.
    pub fn run(&mut self, vm: &Vm, budget: Option<u64>) -> Exit {
        self.interruption_evidence = None;
        self.syscall_evidence = None;
        self.fault_entry.prepare_run();
        let mut fp_mode = crate::jit_abi::fp::RunMode::new();
        struct FaultRunGuard(*mut crate::jit_abi::fault::EntryRecord);
        impl Drop for FaultRunGuard {
            fn drop(&mut self) {
                // SAFETY: the boxed record stays live and fixed throughout run's
                // exclusive vCPU borrow, including while run_inner unwinds.
                unsafe {
                    (*self.0).deactivate();
                }
            }
        }
        let fault_run = FaultRunGuard(&mut *self.fault_entry);
        let exit = if vm.mem.host_fault_recovery() {
            self.run_inner::<true>(vm, budget, &mut fp_mode)
        } else {
            self.run_inner::<false>(vm, budget, &mut fp_mode)
        };
        drop(fault_run);
        let exit = match exit {
            exit @ (Exit::UnmappedMemory { .. } | Exit::Exception { .. }) => {
                self.fault_entry.finish_transfer_fault(exit, &vm.mem)
            }
            other => other,
        };
        self.cpu.finish_mmio_retry(Some(&exit));
        if matches!(exit, Exit::Interrupted) && self.interruption_evidence.is_none() {
            // Acquire the writes published by the consumed request. A later
            // request remains pending; a concurrent SMC bit survives the mask.
            self.interruption_evidence =
                Some(
                    crate::interruption::consume(&self.exit_request.word, &self.engine_sections)
                        .at(crate::FaultPath::Unknown, false, self.cpu.rip),
                );
        }
        // Published here rather than at each of the inner loop's many exits, so no
        // path can forget it: one shared atomic per run instead of one per transfer.
        // (`executed` needs no flush — compiled code updates it live through
        // `MemCtx.icount_ptr`.)
        vm.cache.add_chained(std::mem::take(&mut self.chained));
        exit
    }

    /// Guest x86 instructions executed by this vcpu, counting compiled blocks and
    /// regions as well as the interpreter.
    ///
    /// Distinct from [`Vcpu::retired_instructions`], which stays an interpreter-only,
    /// per-instruction virtual-time base. Compiled code charges a whole block's count
    /// at its entry, so this advances in block-sized steps. Mid-block faults and
    /// interrupted REP operations can charge an unexecuted tail. Divided by wall
    /// time it gives guest IPC; divided by
    /// `TranslationCache::chained` it gives the average length of a compiled unit.
    pub fn executed_instructions(&self) -> u64 {
        self.executed
    }

    #[cold]
    fn syscall(&mut self, path: crate::FaultPath, rip: u64) -> Exit {
        self.syscall_evidence = Some(crate::ExecutionEvidence::new(path, rip));
        Exit::Syscall
    }

    #[cold]
    fn syscall_interpreted(&mut self, ir: &crate::IrBlock) -> Exit {
        let rip = ir
            .ops
            .iter()
            .rev()
            .find_map(|op| match op {
                crate::IrOp::InsnStart { guest_addr } => Some(*guest_addr),
                _ => None,
            })
            .unwrap_or(ir.guest_start);
        self.syscall(crate::FaultPath::Interpreter, rip)
    }

    #[cold]
    fn interrupted(&mut self, path: crate::FaultPath, chain_link: bool) -> Exit {
        self.interruption_evidence = Some(
            crate::interruption::consume(&self.exit_request.word, &self.engine_sections).at(
                path,
                chain_link,
                self.cpu.rip,
            ),
        );
        Exit::Interrupted
    }

    #[cold]
    fn interrupted_compiled(&mut self, entry: CompiledPtr, code: u64) -> Exit {
        if matches!(
            code,
            RET_MMIO_DEFER | RET_HOST_FAULT | RET_HOST_FAULT_DETAIL | RET_PORTIO_DEFER
        ) {
            return self.interrupted(crate::FaultPath::Interpreter, false);
        }
        let path = crate::codemap::fault_path(
            entry.0 as usize,
            crate::Knowledge::Known(self.cpu.rip),
            code == RET_INTERRUPTED,
        );
        self.interrupted(path, code == RET_CHAIN)
    }

    fn code_space_exit(&mut self, vm: &Vm) -> Option<Exit> {
        if !self.code_space_reported && vm.backend.code_space_exhausted() {
            self.code_space_reported = true;
            self.exit_request
                .word
                .fetch_and(!CODE_SPACE_EXHAUSTED, Ordering::Acquire);
            Some(Exit::code_space_exhausted(self.cpu.rip))
        } else {
            None
        }
    }

    fn run_inner<const ROUTED: bool>(
        &mut self,
        vm: &Vm,
        budget: Option<u64>,
        fp_mode: &mut crate::jit_abi::fp::RunMode,
    ) -> Exit {
        let mut remaining = budget;
        loop {
            fp_mode.default();
            let mode = self.cpu.mxcsr & crate::jit_abi::fp::MODE_MASK;
            if self.fast_mxcsr_mode != mode {
                self.fast_clear();
                self.fast_mxcsr_mode = mode;
            }
            let result = if mode == 0 {
                self.run_dispatch::<ROUTED, true>(vm, remaining, fp_mode)
            } else {
                self.run_dispatch::<ROUTED, false>(vm, remaining, fp_mode)
            };
            match result {
                Ok(exit) => return exit,
                Err(consumed) => remaining = remaining.map(|b| b - consumed),
            }
        }
    }

    // A mode write returns to run_inner to select a new specialization and flush
    // private predictions. Default block transfers contain no FP guard checks.
    #[inline(never)]
    fn run_dispatch<const ROUTED: bool, const DEFAULT: bool>(
        &mut self,
        vm: &Vm,
        budget: Option<u64>,
        fp_mode: &mut crate::jit_abi::fp::RunMode,
    ) -> Result<Exit, u64> {
        let mxcsr_mode = if DEFAULT { 0 } else { self.fast_mxcsr_mode };
        let mut blocks_run: u64 = 0;
        let exit_word_owner = Arc::clone(&self.exit_request.word);
        let exit_word: &AtomicU64 = &exit_word_owner;
        // Deliberately a LOCAL: taking it as a `&mut` parameter instead measured +5.4%
        // on the dispatch-micro bench, because the compiler stops treating it as one.
        let mut ctx = MemCtx::for_memory(&vm.mem);
        self.fault_entry.publish(&mut self.cpu, &mut ctx);
        ctx.dispatch_word_ptr = exit_word as *const AtomicU64 as u64;
        // Hand compiled code this vcpu's shadow return stack. `self.ret_stack` is
        // boxed, so its address is stable for the whole run despite `&mut self`.
        ctx.ret_stack = std::ptr::addr_of_mut!(*self.ret_stack) as u64;
        // ...and the executed-instruction counter it charges into. Stable
        // for the run: `self` cannot move while this call is on the stack. Read only
        // by blocks the backend chose to emit accounting into.
        ctx.icount_ptr = std::ptr::addr_of_mut!(self.executed) as u64;

        let mut dispatch = exit_word.load(Ordering::Relaxed);
        loop {
            // The outer dispatcher may invalidate code or invoke backend callbacks.
            // Retain guest control only in the integer-only compiled chain below.
            if !DEFAULT {
                fp_mode.default();
            }
            if dispatch & EXIT_REQUESTED != 0
                && self.cpu.pending_mmio.is_none()
                && !self.cpu.pending_mmio_write
                && self.cpu.pending_port_in.is_none()
            {
                return Ok(Exit::Interrupted);
            }
            // The sticky flag can be observed before the backend finishes its
            // broadcast. A late bit must not keep an already-notified vcpu in
            // the outer loop forever.
            if self.code_space_reported && dispatch & CODE_SPACE_EXHAUSTED != 0 {
                exit_word.fetch_and(!CODE_SPACE_EXHAUSTED, Ordering::Acquire);
            }
            if let Some(exit) = self.code_space_exit(vm) {
                return Ok(exit);
            }
            if budget.is_some_and(|b| blocks_run >= b) {
                return Ok(Exit::BudgetExhausted);
            }

            // SMC (§10): drop any cached block a prior block's store landed on,
            // before fetching the next one, so re-execution re-lifts fresh bytes.
            vm.handle_smc(exit_word);

            // Flush the fast-resolve cache if any invalidation happened since we
            // last synced — covers same-thread SMC (just handled above) and
            // cross-thread SMC (another vcpu bumped the epoch). Ordered after
            // `handle_smc` so a probe never predates its own invalidation.
            let epoch = vm.cache.epoch();
            if epoch != self.fast_epoch {
                self.fast_clear();
                self.fast_epoch = epoch;
            }

            // Resume after an MMIO trap the JIT deferred (§5.2): once the
            // embedder has supplied the read value (`complete_mmio_read`) or
            // acknowledged the write (`complete_mmio_write`), single-step the
            // faulting instruction on the interpreter — it consumes the pending
            // value/ack and advances RIP — before dispatching the next block. (Under
            // the interpreter backend this is equivalent to re-dispatching the block;
            // it just does the one faulting instruction first.)
            // A RAM fault may have consumed the pending completion while leaving
            // recorded transfers. Keep that retry on the interpreter until completion;
            // compiled blocks never inherit answers from a partially serviced operand.
            if self.cpu.mmio_retry_active {
                if !DEFAULT {
                    fp_mode.default();
                }
                // Finish the pending instruction before honouring a host request.
                let mut info = crate::interp::RetireInfo {
                    consistency: vm.consistency,
                    ..Default::default()
                };
                let instruction_rip = self.cpu.rip;
                let r = crate::interp::step_one_mode::<ROUTED>(
                    &vm.mem,
                    &mut self.cpu,
                    self.mode,
                    &mut self.interp_scratch,
                    &mut info,
                );
                self.retired += info.retired;
                self.sti_shadow = info.sti_shadow;
                match r {
                    StepResult::Continue => {
                        blocks_run += 1;
                        if self.cpu.mxcsr & crate::jit_abi::fp::MODE_MASK != mxcsr_mode {
                            return Err(blocks_run);
                        }
                        dispatch = exit_word.load(Ordering::Relaxed);
                        if dispatch & EXIT_REQUESTED != 0 {
                            return Ok(self.interrupted(crate::FaultPath::Interpreter, false));
                        }
                        continue;
                    }
                    StepResult::Exit(Exit::Syscall) => {
                        return Ok(self.syscall(crate::FaultPath::Interpreter, instruction_rip));
                    }
                    StepResult::Exit(exit) => {
                        return Ok(crate::fault::qualify_real16(exit, &self.cpu))
                    }
                }
            }

            // §17.6 (sub-seam c): hardware-interrupt injection is delivered here, at a
            // run() boundary — never mid-block. Gate: a vector is pending, IF is set, no
            // memory/port completion is outstanding (a partially-serviced MMIO/`in` must
            // finish first), and the one-instruction STI shadow has elapsed. All false-
            // gating leaves the vector queued (never dropped) for a later boundary — this
            // is also the point that services a post-`hlt` wakeup: after an `Exit::Hlt`
            // the embedder injects and re-enters, and RIP already sits past the `hlt`, so
            // delivery vectors the handler and `iret` resumes past the `hlt`. The
            // embedder owns the PIC: it must have run INTA / updated the ISR before
            // `inject_irq`. Real-mode only (Long64/Compat32 never set `pending_irq`).
            if let Some(vector) = self.pending_irq {
                let deliverable = self.cpu.flags.if_
                    && self.cpu.pending_mmio.is_none()
                    && !self.cpu.pending_mmio_write
                    && self.cpu.pending_port_in.is_none()
                    && !self.sti_shadow;
                if deliverable {
                    if !DEFAULT {
                        fp_mode.default();
                    }
                    self.pending_irq = None;
                    // Saved IP = current RIP (the maskable IRQ is an async trap; execution
                    // resumes at the next instruction, which is where RIP already points).
                    let saved = self.cpu.rip;
                    match crate::interp::deliver_interrupt(
                        &mut self.cpu,
                        &vm.mem,
                        saved,
                        vector,
                        saved,
                        vm.consistency,
                    )
                    .into_step()
                    {
                        StepResult::Continue => {
                            // Delivery is itself an atomic hardware event, not a retired
                            // guest instruction — do not tick `retired`; do charge a block
                            // for §9.2 budget accounting. It also clears any stale shadow.
                            self.sti_shadow = false;
                            blocks_run += 1;
                            dispatch = exit_word.load(Ordering::Relaxed);
                            continue;
                        }
                        StepResult::Exit(exit) => {
                            return Ok(crate::fault::qualify_real16(exit, &self.cpu))
                        }
                    }
                }
            }

            // §17.6: the physical fetch address (`cs_base + IP` in Real16, else `rip`).
            // The block cache and lift key on `fetch.pa`; the fast-probe stays keyed on
            // the raw `rip` (it only ever holds *compiled* blocks, which Real16 never
            // produces — so an IP-keyed probe is harmless and identical outside Real16).
            let fetch = FetchAddr::for_mode(self.mode, self.cpu.rip, self.cpu.cs);

            // Fast path: a vcpu-private probe replaces the shared cache lookup
            // for compiled blocks. A miss falls back to `resolve` and installs the
            // result; interpreted blocks are never cached here, so they always route
            // through `resolve` (keeping the cache hit/miss counters meaningful).
            let block = match self.fast_get(self.cpu.rip) {
                Some(entry) => {
                    // Plain (non-atomic) per-vcpu counter — the whole point of the
                    // private probe is to avoid the shared atomic bumps on this path.
                    self.fast_hits += 1;
                    CachedBlock::Compiled { entry }
                }
                None => {
                    if !DEFAULT {
                        fp_mode.default();
                    }
                    match resolve_dispatch::<DEFAULT>(
                        vm,
                        exit_word,
                        &self.engine_sections,
                        fetch,
                        self.mode,
                        if DEFAULT { 0 } else { self.cpu.mxcsr },
                    ) {
                        Ok(b) => {
                            if let CachedBlock::Compiled { entry, .. } = &b {
                                self.fast_put(self.cpu.rip, *entry);
                            }
                            b
                        }
                        Err(exit) => return Ok(exit),
                    }
                }
            };

            if let Some(exit) = self.code_space_exit(vm) {
                return Ok(exit);
            }
            match block {
                CachedBlock::Interpreted(ir) => {
                    if !DEFAULT {
                        fp_mode.default();
                    }
                    let mut info = crate::interp::RetireInfo {
                        consistency: vm.consistency,
                        exit_request: Some(exit_word),
                        trace: vm.lockstep_trace.as_deref(),
                        ..Default::default()
                    };
                    let r = crate::interp::interpret_block_fp::<ROUTED, DEFAULT>(
                        &ir,
                        &mut self.cpu,
                        &vm.mem,
                        &mut self.interp_scratch,
                        &mut info,
                    );
                    // §17.6 (sub-seam c): tick the retired-instruction counter and latch
                    // the STI shadow from this interpreted block (Real16 is all
                    // interpreter, so this is the whole real-mode instruction stream).
                    self.retired += info.retired;
                    // Same counter as compiled code charges, so the total
                    // covers whichever tier actually ran the guest.
                    self.executed += info.retired;
                    self.sti_shadow = info.sti_shadow;
                    match r {
                        crate::interp::BlockResult::Continue => blocks_run += 1,
                        crate::interp::BlockResult::ModeChange => return Err(blocks_run + 1),
                        // §17.6 (sub-seam b): in real mode a CPU exception is delivered
                        // in-guest through the IVT, not surfaced to the embedder. The
                        // only `Exit::Exception` that escapes an interpreted Real16 block
                        // is `#DE` (divide error, vector 0) from `IrOp::Div` — `int n`/
                        // `int3`/`ud2` already vector in-guest via `IrOp::IntGate`. Its
                        // `addr` is the faulting instruction's IP (the saved IP for a
                        // fault), so re-deliver through the same IVT path and continue.
                        // Long64/Compat32 still return `Exit::Exception` unchanged.
                        crate::interp::BlockResult::Exit(exit) => {
                            let exit = exit.into_exit();
                            match exit {
                                Exit::Syscall => return Ok(self.syscall_interpreted(&ir)),
                                Exit::Interrupted => {
                                    return Ok(self.interrupted(
                                        crate::FaultPath::Helper {
                                            family: crate::FaultHelper::String,
                                            caller: crate::HelperCaller::Interpreter,
                                        },
                                        false,
                                    ));
                                }
                                Exit::Exception { addr, vector, .. }
                                    if self.mode == CpuMode::Real16 =>
                                {
                                    match crate::interp::deliver_interrupt(
                                        &mut self.cpu,
                                        &vm.mem,
                                        addr,
                                        vector,
                                        addr,
                                        vm.consistency,
                                    )
                                    .into_step()
                                    {
                                        StepResult::Continue => blocks_run += 1,
                                        StepResult::Exit(exit) => {
                                            return Ok(crate::fault::qualify_real16(
                                                exit, &self.cpu,
                                            ))
                                        }
                                    }
                                }
                                exit @ Exit::Exception { .. } => {
                                    return Ok(crate::interp::qualify_lifted_exit(exit, &ir))
                                }
                                exit => {
                                    return Ok(if self.mode == CpuMode::Real16 {
                                        crate::fault::qualify_real16(exit, &self.cpu)
                                    } else {
                                        exit
                                    })
                                }
                            }
                        }
                    }
                    dispatch = exit_word.load(Ordering::Relaxed);
                    if dispatch & EXIT_REQUESTED != 0 {
                        return Ok(self.interrupted(crate::FaultPath::Interpreter, false));
                    }
                }
                CachedBlock::Compiled { entry, .. } => {
                    dispatch = exit_word.load(Ordering::Relaxed);
                    if dispatch != 0 {
                        continue;
                    }
                    let mut cur = entry;
                    let (stopped, stopped_code) = loop {
                        // Hand the block its remaining block quantum (superblocks,
                        // §12 M5): a compiled region spends 1 fuel per guest block and
                        // stops at 0; a single block ignores it. Charging the fuel it
                        // consumed (min 1) keeps `blocks_run` an exact guest-block
                        // count — identical to the interpreter, preserving §9.2 and
                        // the `RunSpec::Blocks(n)` oracle.
                        let quantum = budget.map_or(u64::MAX, |b| b - blocks_run);
                        ctx.fuel = quantum;
                        // SAFETY: `cur` is a block compiled to this ABI, alive in
                        // the JIT arena (owned by `vm`) for the call.
                        if !DEFAULT {
                            fp_mode.compiled(self.cpu.mxcsr);
                        }
                        let code = unsafe {
                            crate::jit_abi::call_block_in_mode(cur, &mut self.cpu, &mut ctx)
                        };
                        blocks_run += (quantum - ctx.fuel).max(1);
                        if !DEFAULT && !matches!(code, RET_CHAIN | RET_LINK | RET_IBTC_MISS) {
                            fp_mode.default();
                        }
                        let next;
                        match code {
                            RET_CONTINUE | RET_INTERRUPTED => break (cur, code),
                            RET_MXCSR_CHANGED => return Err(blocks_run),
                            RET_CHAIN => {
                                // Plain per-vcpu counter, flushed once per `run`
                                // (see `TranslationCache::add_chained`): this fires
                                // ~1M times per frame in a real game, so a shared
                                // atomic here is a locked RMW on the hottest path.
                                self.chained += 1;
                                next = CompiledPtr(ctx.next_entry as *const u8);
                            }
                            RET_LINK => {
                                // Snapshot BEFORE resolving: the publish
                                // below only happens if the epoch is still this one, so
                                // an SMC drop that lands while we resolve cannot have its
                                // slot-clearing undone by our store.
                                let since_epoch = vm.cache.epoch();
                                if !DEFAULT {
                                    fp_mode.default();
                                }
                                let resolved = resolve::<DEFAULT>(
                                    vm,
                                    exit_word,
                                    &self.engine_sections,
                                    FetchAddr::flat(self.cpu.rip),
                                    self.mode,
                                    if DEFAULT { 0 } else { self.cpu.mxcsr },
                                );
                                if let Some(exit) = self.code_space_exit(vm) {
                                    return Ok(exit);
                                }
                                match resolved {
                                    Ok(CachedBlock::Compiled { entry, .. }) => {
                                        // SAFETY: `link_slot` is a live `Box<AtomicU64>`
                                        // in the JIT arena. Relaxed store: another vcpu
                                        // reading the slot sees 0 or a valid entry, never a
                                        // torn value (aligned u64); it pairs with the
                                        // backend's `invalidate_links` clear.
                                        let slot = ctx.link_slot;
                                        let _section = EngineSection::enter(
                                            exit_word,
                                            &self.engine_sections,
                                            CriticalSectionKind::Publish,
                                        );
                                        let published =
                                            vm.cache.publish_if_current(since_epoch, || unsafe {
                                                (*(slot as *const AtomicU64))
                                                    .store(entry.0 as u64, Ordering::Relaxed)
                                            });
                                        drop(_section);
                                        if !published {
                                            // Invalidated under us. Publishing now would
                                            // refill a slot the drop just cleared, and
                                            // transferring would run a translation of code
                                            // the guest has rewritten. Back to the outer
                                            // loop, which runs `handle_smc` and re-resolves.
                                            break (cur, code);
                                        }
                                        // Seed the fast-resolve cache too: the next
                                        // outer-loop visit to this RIP skips `resolve`.
                                        self.fast_put(self.cpu.rip, entry);
                                        next = entry;
                                    }
                                    // Mixed backend can't chain — fall back to dispatch.
                                    Ok(CachedBlock::Interpreted(_)) => break (cur, code),
                                    Err(exit) => return Ok(exit),
                                }
                            }
                            // IBTC miss: an indirect edge whose per-site slot was
                            // empty or held a different target. Resolve the computed
                            // RIP and refill the slot with a fresh {target, entry}
                            // descriptor, unless the site is megamorphic.
                            RET_IBTC_MISS => {
                                // Snapshot before resolving, as for RET_LINK.
                                let since_epoch = vm.cache.epoch();
                                // Probe the vcpu-private fast cache before `resolve`.
                                // The per-site IBTC holds ONE target, so a site
                                // with several live targets misses here on nearly every
                                // call — and `resolve` is an RwLock read plus a hash
                                // lookup, a clone and two shared atomic bumps, where
                                // this is a direct-mapped array index and a compare.
                                // Same key, same epoch invariant and same "compiled
                                // blocks only" content as the probe the outer dispatch
                                // loop already trusts, so a hit here is equally sound;
                                // it skips the tier-up drain exactly as that one does.
                                let entry = match self.fast_get(self.cpu.rip) {
                                    Some(entry) => {
                                        self.fast_hits += 1;
                                        Some(entry)
                                    }
                                    None => {
                                        if !DEFAULT {
                                            fp_mode.default();
                                        }
                                        let resolved = resolve::<DEFAULT>(
                                            vm,
                                            exit_word,
                                            &self.engine_sections,
                                            FetchAddr::flat(self.cpu.rip),
                                            self.mode,
                                            if DEFAULT { 0 } else { self.cpu.mxcsr },
                                        );
                                        if let Some(exit) = self.code_space_exit(vm) {
                                            return Ok(exit);
                                        }
                                        match resolved {
                                            Ok(CachedBlock::Compiled { entry }) => {
                                                self.fast_put(self.cpu.rip, entry);
                                                Some(entry)
                                            }
                                            // Indirect edge into an interpreted block.
                                            Ok(CachedBlock::Interpreted(_)) => None,
                                            Err(exit) => return Ok(exit),
                                        }
                                    }
                                };
                                // Even a fast hit can allocate during refill bookkeeping.
                                if !DEFAULT {
                                    fp_mode.default();
                                }
                                match entry {
                                    Some(entry) => {
                                        let slot = ctx.link_slot;
                                        let count = self.ibtc_refills.entry(slot).or_insert(0);
                                        if *count < IBTC_MEGAMORPHIC_CAP {
                                            *count += 1;
                                            let _section = EngineSection::enter(
                                                exit_word,
                                                &self.engine_sections,
                                                CriticalSectionKind::Publish,
                                            );
                                            let desc =
                                                vm.cache.alloc_ibtc_descriptor(self.cpu.rip, entry);
                                            // SAFETY: `slot` is a live `Box<AtomicU64>` in
                                            // the JIT arena; the published descriptor is
                                            // immutable and never freed.
                                            // Release (not Relaxed): unlike the RET_LINK
                                            // slot — a single scalar entry — this publishes
                                            // a POINTER to a multi-field {target, entry}
                                            // payload, so the payload's writes must be
                                            // ordered-visible before the pointer. Release
                                            // here pairs with the reader's address
                                            // dependency (the compiled `ibtc_or_miss` loads
                                            // the descriptor fields *through* this pointer),
                                            // giving release/consume ordering; a plain
                                            // Relaxed store would let a weakly-ordered host
                                            // (AArch64) expose the pointer before the fields.
                                            let published = vm.cache.publish_if_current(
                                                since_epoch,
                                                || unsafe {
                                                    (*(slot as *const AtomicU64))
                                                        .store(desc, Ordering::Release)
                                                },
                                            );
                                            if !published {
                                                // Invalidated under us — see RET_LINK.
                                                break (cur, code);
                                            }
                                        }
                                        next = entry;
                                    }
                                    // Interpreted target — leave the chain and dispatch.
                                    None => break (cur, code),
                                }
                            }
                            RET_SYSCALL => {
                                let path = crate::codemap::fault_path(
                                    cur.0 as usize,
                                    crate::Knowledge::Known(ctx.next_entry),
                                    false,
                                );
                                return Ok(self.syscall(path, ctx.next_entry));
                            }
                            RET_HLT => return Ok(Exit::Hlt),
                            RET_UNMAPPED => return Ok(ctx.unmapped_exit(&self.cpu, self.mode)),
                            RET_UNMAPPED_DETAIL => {
                                return Ok(ctx.detailed_unmapped_exit(
                                    &self.cpu,
                                    &vm.mem,
                                    self.mode,
                                    cur.0 as usize,
                                ))
                            }
                            RET_EXCEPTION_DETAIL => {
                                let exit = ctx.detailed_exception_exit(
                                    &self.cpu,
                                    &vm.mem,
                                    self.mode,
                                    cur.0 as usize,
                                );
                                return Ok(crate::fault::qualify_lifted(
                                    exit,
                                    crate::codemap::lookup_fault_info(
                                        cur.0 as usize,
                                        ctx.next_entry,
                                    ),
                                ));
                            }
                            RET_MMIO_DEFER | RET_HOST_FAULT | RET_HOST_FAULT_DETAIL => {
                                // Alignment recovery has no reliable data address.
                                if code == RET_MMIO_DEFER
                                    || self.fault_entry.fault_kind()
                                        != crate::jit_abi::fault::HostFaultKind::Alignment
                                {
                                    let pc = if code == RET_HOST_FAULT_DETAIL {
                                        ctx.next_entry as usize
                                    } else {
                                        cur.0 as usize
                                    };
                                    if let Some(info) =
                                        crate::codemap::lookup_fault_info(pc, self.cpu.rip)
                                    {
                                        info.note_partial_store(&self.cpu, &vm.mem, ctx.fault_addr);
                                    }
                                }
                                let raw_fault = if code == RET_HOST_FAULT_DETAIL {
                                    self.fault_entry.raw_fault()
                                } else {
                                    None
                                };
                                if code == RET_HOST_FAULT || code == RET_HOST_FAULT_DETAIL {
                                    if DEFAULT {
                                        // The landing stub may have restored non-default caller
                                        // control. No guest mode was installed in this specialization.
                                        // SAFETY: normalize before fault decoding and replay.
                                        unsafe {
                                            crate::jit_abi::fp::set_control(
                                                crate::jit_abi::fp::default_control(),
                                            )
                                        };
                                    } else {
                                        fp_mode.host_fault();
                                    }
                                    if self.fault_entry.fault_kind()
                                        != crate::jit_abi::fault::HostFaultKind::Alignment
                                        && vm.mem.mapped_ram(ctx.fault_addr)
                                    {
                                        let exit = if code == RET_HOST_FAULT_DETAIL {
                                            ctx.detailed_host_exit(&self.cpu, &vm.mem, self.mode)
                                        } else {
                                            ctx.unmapped_exit(&self.cpu, self.mode)
                                        };
                                        return Ok(crate::fault::attach_raw_host_fault(
                                            exit, raw_fault,
                                        ));
                                    }
                                }
                                if !DEFAULT {
                                    fp_mode.default();
                                }
                                // Trap-region accesses replay to produce MMIO exits.
                                // Host faults reach replay only for unmapped addresses
                                // or alignment faults. Misaligned locked operations
                                // currently replay non-atomically (tracked in xj-0w3.5).
                                // Replay uses this run's memory policy in both modes.
                                // Interpreter single-step (not the hot compiled loop):
                                // tick the retired counter for the one instruction it
                                // may retire (§17.6, sub-seam c).
                                let mut info = crate::interp::RetireInfo {
                                    consistency: vm.consistency,
                                    exit_request: Some(exit_word),
                                    trace: vm.lockstep_trace.as_deref(),
                                    ..Default::default()
                                };
                                let original_info = if code == RET_HOST_FAULT_DETAIL {
                                    crate::codemap::lookup_fault_info(
                                        ctx.next_entry as usize,
                                        self.cpu.rip,
                                    )
                                } else {
                                    None
                                };
                                let instruction_rip = self.cpu.rip;
                                let r = crate::interp::step_one_mode::<ROUTED>(
                                    &vm.mem,
                                    &mut self.cpu,
                                    self.mode,
                                    &mut self.interp_scratch,
                                    &mut info,
                                );
                                self.retired += info.retired;
                                match r {
                                    StepResult::Continue => {
                                        if self.cpu.mxcsr & crate::jit_abi::fp::MODE_MASK
                                            != mxcsr_mode
                                        {
                                            return Err(blocks_run);
                                        }
                                        break (cur, code);
                                    }
                                    StepResult::Exit(Exit::Syscall) => {
                                        return Ok(self.syscall(
                                            crate::FaultPath::Interpreter,
                                            instruction_rip,
                                        ));
                                    }
                                    StepResult::Exit(Exit::Interrupted) => {
                                        return Ok(self.interrupted(
                                            crate::FaultPath::Helper {
                                                family: crate::FaultHelper::String,
                                                caller: crate::HelperCaller::Interpreter,
                                            },
                                            false,
                                        ));
                                    }
                                    StepResult::Exit(exit) => {
                                        return Ok(
                                            if code == RET_HOST_FAULT
                                                || code == RET_HOST_FAULT_DETAIL
                                            {
                                                crate::fault::attach_raw_host_fault(
                                                    ctx.preserve_host_replay(
                                                        exit,
                                                        original_info,
                                                        code == RET_HOST_FAULT_DETAIL,
                                                    ),
                                                    raw_fault,
                                                )
                                            } else {
                                                exit
                                            },
                                        );
                                    }
                                }
                            }
                            // A port-I/O instruction (`in`/`out`, §5.2): the block set
                            // RIP to the instruction; single-step it on the interpreter
                            // to produce the `Exit::PortIo` (same deferral as MMIO).
                            RET_PORTIO_DEFER => {
                                if !DEFAULT {
                                    fp_mode.default();
                                }
                                let mut info = crate::interp::RetireInfo {
                                    consistency: vm.consistency,
                                    exit_request: Some(exit_word),
                                    trace: vm.lockstep_trace.as_deref(),
                                    ..Default::default()
                                };
                                let instruction_rip = self.cpu.rip;
                                let r = crate::interp::step_one_mode::<ROUTED>(
                                    &vm.mem,
                                    &mut self.cpu,
                                    self.mode,
                                    &mut self.interp_scratch,
                                    &mut info,
                                );
                                self.retired += info.retired;
                                match r {
                                    StepResult::Continue => break (cur, code),
                                    StepResult::Exit(Exit::Syscall) => {
                                        return Ok(self.syscall(
                                            crate::FaultPath::Interpreter,
                                            instruction_rip,
                                        ));
                                    }
                                    StepResult::Exit(exit) => return Ok(exit),
                                }
                            }
                            // A guest exception (`#DE` div, or a lifted `ud2`/`int3`/
                            // `int1` trap); the block set the saved RIP (fault: on the
                            // instruction, trap: past it) and stored the vector in the
                            // MemCtx out-field.
                            RET_EXCEPTION => {
                                return Ok(Exit::exception(
                                    self.cpu.rip,
                                    ctx.exception_vector as u8,
                                ))
                            }
                            other => panic!("compiled block returned invalid ABI code {other}"),
                        }
                        if budget.is_some_and(|b| blocks_run >= b) {
                            if !DEFAULT {
                                fp_mode.default();
                            }
                            return Ok(Exit::BudgetExhausted);
                        }
                        // Poll before replacing the just-executed unit with its
                        // successor. Only the cold edge needs its identity.
                        if exit_word.load(Ordering::Relaxed) != 0 {
                            break (cur, code);
                        }
                        cur = next;
                    };
                    dispatch = exit_word.load(Ordering::Relaxed);
                    if dispatch & EXIT_REQUESTED != 0 {
                        if !DEFAULT {
                            fp_mode.default();
                        }
                        return Ok(self.interrupted_compiled(stopped, stopped_code));
                    }
                }
            }
        }
    }
}

/// Publish every completed background compile into the cache: the dispatcher
/// publishes, the backend never touches the cache. Each is epoch-checked by `upgrade`
/// (a stale compile whose block was SMC-dropped is rejected); the in-flight marker is
/// always cleared so a rejected block can be re-lifted and re-submitted.
/// `tier_up_finished` short-circuits when idle.
fn drain_tier_up(vm: &Vm, exit_word: &AtomicU64, sections: &AtomicU64) {
    let _section = EngineSection::enter(exit_word, sections, CriticalSectionKind::TierUpWait);
    for fin in vm.backend.tier_up_finished() {
        // A Vm runs in a single mode (§17 scope fence), so the finished compile's
        // block key is its echoed pc under the Vm's mode (§17.4).
        let key = BlockKey::with_mxcsr(fin.pc, vm.mode, fin.mxcsr_mode);
        if matches!(fin.block, CachedBlock::Interpreted(_)) {
            // A failed background compile leaves the already-cached interpreter
            // block in place; only clear its in-flight marker.
            vm.cache.end_tier_up(key);
            if vm.cache.epoch() != fin.epoch {
                vm.cache.record_tier_bg_rejected();
            }
            continue;
        }
        // Multi-span publish: a region carries one span per sub-block; a block carries
        // one. `on_mark` re-tags the pages under the spans lock.
        let multi_span = fin.spans.len() > 1;
        let _publish = EngineSection::enter(exit_word, sections, CriticalSectionKind::Publish);
        let published = vm
            .cache
            .upgrade_region(key, fin.block, fin.spans, fin.epoch, |sp| {
                for (start, len) in sp {
                    vm.mem.mark_code(*start, *len);
                }
            });
        vm.cache.end_tier_up(key);
        if published {
            vm.cache.record_tier_bg_published();
            // A multi-span unit is a superblock region — count it like the
            // eager region path does, so `cache.regions()` reflects background regions.
            if multi_span {
                vm.cache.record_region();
            }
        } else {
            vm.cache.record_tier_bg_rejected();
        }
    }
}

/// Fetch a block from the cache or lift+materialize it (miss). Lift errors are legal
/// exits (not `run()` failures) telling the user what to add (§9.2).
enum ResolveFailure {
    Retry,
    Exit(Exit),
}

#[inline(always)]
fn resolve_dispatch<const DEFAULT: bool>(
    vm: &Vm,
    exit_word: &AtomicU64,
    sections: &AtomicU64,
    at: FetchAddr,
    mode: CpuMode,
    mxcsr: u32,
) -> Result<CachedBlock, Exit> {
    // Keep compilation and publication state out of non-tiered cache hits.
    if !vm.tier_up_background && vm.tier_up_after.is_none() && mode != CpuMode::Real16 {
        let key = if DEFAULT {
            BlockKey::new(at.pa, mode)
        } else {
            BlockKey::with_mxcsr(at.pa, mode, mxcsr)
        };
        loop {
            if let Some(block) = vm.cache.get(key) {
                return Ok(block);
            }
            // No IR exists on a miss; snapshot before fetching its bytes.
            let epoch = vm.cache.epoch();
            match resolve_uncached::<DEFAULT>(vm, exit_word, sections, at, mode, mxcsr, epoch) {
                Ok(block) => return Ok(block),
                Err(ResolveFailure::Exit(exit)) => return Err(exit),
                Err(ResolveFailure::Retry) => continue,
            }
        }
    }
    resolve::<DEFAULT>(vm, exit_word, sections, at, mode, mxcsr)
}

#[inline(never)]
fn resolve<const DEFAULT: bool>(
    vm: &Vm,
    exit_word: &AtomicU64,
    sections: &AtomicU64,
    at: FetchAddr,
    mode: CpuMode,
    mxcsr: u32,
) -> Result<CachedBlock, Exit> {
    loop {
        match resolve_attempt::<DEFAULT>(vm, exit_word, sections, at, mode, mxcsr) {
            Ok(block) => return Ok(block),
            Err(ResolveFailure::Exit(exit)) => return Err(exit),
            Err(ResolveFailure::Retry) => continue,
        }
    }
}

fn resolve_attempt<const DEFAULT: bool>(
    vm: &Vm,
    exit_word: &AtomicU64,
    sections: &AtomicU64,
    at: FetchAddr,
    mode: CpuMode,
    mxcsr: u32,
) -> Result<CachedBlock, ResolveFailure> {
    // §17.4: cache maps key on { physical-fetch-address, mode }. In Real16 the physical
    // fetch address (`cs_base + IP`) is what keys the cache and tags SMC pages, so
    // blocks never collide across segments; the IR's `guest_start` stays the decode IP
    // (`at.ip`) so fall-through / branch targets remain 16-bit IPs (§17.6). Outside
    // Real16 `at.pa == at.ip == pc`, so this is byte-identical to the old `pc` keying.
    let pc = at.pa;
    let key = if DEFAULT {
        BlockKey::new(pc, mode)
    } else {
        BlockKey::with_mxcsr(pc, mode, mxcsr)
    };
    // §17.6: Real16 has no compiled tier. Bypass tier-up and region formation even
    // when the embedder injected a compiling backend.
    if mode == CpuMode::Real16 {
        if let Some(block) = vm.cache.get(key) {
            return Ok(block);
        }
        let compile_epoch = vm.cache.epoch();
        let permit = vm.mem.fetch_permit();
        let mut fetch_failure = None;
        let lifted = {
            let _section = EngineSection::enter(exit_word, sections, CriticalSectionKind::Lift);
            lift_block_with_fault(&vm.mem, at, mode, &mut fetch_failure)
        };
        return lifted
            .map_err(|e| ResolveFailure::Exit(lift_exit(e, fetch_failure, &vm.mem, at, mode)))
            .and_then(|ir| {
                finish_single(
                    vm,
                    exit_word,
                    sections,
                    key,
                    pc,
                    ir,
                    permit.as_ref(),
                    compile_epoch,
                )
                .ok_or(ResolveFailure::Retry)
            });
    }
    loop {
        // Publish any completed background compiles first, so a
        // freshly-landed unit is seen by the lookup below. Cheap when idle (the
        // backend's ready-probe short-circuits an empty drain).
        if vm.tier_up_background {
            drain_tier_up(vm, exit_word, sections);
        }
        // Snapshot the invalidation epoch BEFORE the lookup: `upgrade` rejects the
        // tier-up if an SMC drop moved it in between. Otherwise a tier-up racing a
        // concurrent `invalidate_overlapping` would resurrect a stale block with no
        // span — permanently uninvalidatable.
        let epoch = vm.cache.epoch();
        let Some(block) = vm.cache.get(key) else {
            return resolve_uncached::<DEFAULT>(vm, exit_word, sections, at, mode, mxcsr, epoch);
        };
        // Hotness-gated tier-up: a cached *interpreted* block that has
        // now run `tier_up_after` times gets JIT-compiled from its already-lifted
        // IR and swapped in, so cold one-shot blocks never pay compile cost while
        // hot blocks still tier up.
        let (Some(thr), CachedBlock::Interpreted(ir)) = (vm.tier_up_after, &block) else {
            return Ok(block);
        };
        if vm.backend.code_space_exhausted() {
            return Ok(block);
        }
        let count = vm.cache.bump_hotness(key);
        if count < thr {
            return Ok(block);
        }
        let mut permit = None;
        if vm.mem.fetch_policy.is_some() {
            let _section = EngineSection::enter(exit_word, sections, CriticalSectionKind::Publish);
            if !vm
                .cache
                .publish_if_current(epoch, || permit = vm.mem.fetch_permit())
            {
                continue;
            }
        }
        // Background tier-up: compile off the vcpu and keep interpreting
        // until the result lands (published by `drain_tier_up` above on a later
        // dispatch). Submit once — `try_begin_tier_up` gates re-submission.
        if vm.tier_up_background {
            // Adaptive per-block tier. A region-forming backend decides once,
            // per pc, whether this is a multi-block loop worth a region. A loop stays
            // interpreted until a much higher backedge threshold T2 (a premature region
            // on a short loop regresses, T3f) — the OSR analogue; a non-loop block (or a
            // `None`-caps backend) tiers the single block at T1 as before.
            let region_caps = vm.backend.region_caps();
            let region_candidate = match region_caps {
                Some(caps) => match vm.cache.region_decision(key) {
                    Some(c) => c,
                    None => {
                        let _section = EngineSection::enter(
                            exit_word,
                            sections,
                            CriticalSectionKind::RegionBuild,
                        );
                        let c = matches!(
                            lift_region(&vm.mem, pc, caps, mode),
                            Ok(r) if r.blocks.len() > 1 && r.has_loop
                        );
                        vm.cache.set_region_decision(key, c);
                        c
                    }
                },
                None => false,
            };
            let (unit, spans) = if region_candidate {
                let t2 = vm.tier_up_region_after.unwrap_or(thr);
                if count < t2 {
                    return Ok(block); // a hot loop, still warming toward the region tier
                }
                if !vm.cache.try_begin_tier_up(key) {
                    return Ok(block); // a compile for this pc is already in flight
                }
                let caps = region_caps.expect("region candidate ⇒ region_caps");
                let mut fetch_failure = None;
                let lifted = {
                    let _section =
                        EngineSection::enter(exit_word, sections, CriticalSectionKind::RegionBuild);
                    lift_region_with_fault(&vm.mem, pc, caps, mode, &mut fetch_failure)
                };
                match lifted {
                    Ok(region) if region.blocks.len() > 1 && region.has_loop => {
                        let spans = region.spans();
                        (TierUpUnit::Region(Arc::new(region)), spans)
                    }
                    // SMC turned it non-loop since the decision — tier the single block.
                    _ => (
                        TierUpUnit::Block(ir.clone()),
                        vec![(ir.guest_start, ir.guest_len)],
                    ),
                }
            } else {
                if !vm.cache.try_begin_tier_up(key) {
                    return Ok(block); // a compile for this pc is already in flight
                }
                (
                    TierUpUnit::Block(ir.clone()),
                    vec![(ir.guest_start, ir.guest_len)],
                )
            };
            if permit.as_ref().is_some_and(|p| !p.is_current()) {
                vm.cache.end_tier_up(key);
                continue;
            }
            let req = TierUpRequest {
                precise_split_stores: vm.precise_split_stores,
                mxcsr_mode: key.mxcsr_mode,
                pc,
                unit,
                consistency: vm.consistency,
                mmio: vm.mem.trap_window(),
                guest_base: vm.mem.guest_base(),
                spans,
                epoch,
            };
            let submitted = {
                let _section =
                    EngineSection::enter(exit_word, sections, CriticalSectionKind::TierUpWait);
                if let Some(permit) = permit.clone() {
                    vm.backend.tier_up_async_with_fetch_permit(req, permit)
                } else {
                    vm.backend.tier_up_async(req)
                }
            };
            match submitted {
                // Off to the worker — this block stays interpreted for now.
                TierUpSubmit::Queued => return Ok(block),
                // Queue full: don't compile inline (that reintroduces the spike);
                // drop the marker so hotness re-submits on a later dispatch.
                TierUpSubmit::Busy => {
                    vm.cache.end_tier_up(key);
                    return Ok(block);
                }
                // No worker (interpreter, or the JIT with bg off): fall through to
                // today's inline tier-up.
                TierUpSubmit::Unsupported => vm.cache.end_tier_up(key),
            }
        }
        if permit.as_ref().is_some_and(|p| !p.is_current()) {
            continue;
        }
        let compiled = {
            let _section = EngineSection::enter(exit_word, sections, CriticalSectionKind::Compile);
            vm.hold_compile();
            vm.backend.materialize_with_split_stores(
                ir,
                vm.consistency,
                vm.mem.trap_window(),
                vm.mem.guest_base(),
                key.mxcsr_mode,
                vm.precise_split_stores,
            )
        };
        let _section = EngineSection::enter(exit_word, sections, CriticalSectionKind::Publish);
        if vm
            .cache
            .upgrade(key, compiled.clone(), (ir.guest_start, ir.guest_len), epoch)
        {
            return Ok(compiled);
        }
        // Lost the race: an SMC drop invalidated the block mid-tier-up. Loop to
        // re-fetch / re-lift from current memory rather than run a stale block.
    }
}

#[cold]
#[inline(never)]
fn resolve_uncached<const DEFAULT: bool>(
    vm: &Vm,
    exit_word: &AtomicU64,
    sections: &AtomicU64,
    at: FetchAddr,
    mode: CpuMode,
    mxcsr: u32,
    compile_epoch: u64,
) -> Result<CachedBlock, ResolveFailure> {
    let pc = at.pa;
    let key = if DEFAULT {
        BlockKey::new(pc, mode)
    } else {
        BlockKey::with_mxcsr(pc, mode, mxcsr)
    };
    let permit = vm.mem.fetch_permit();
    // Region path (§12 M5): a region-forming backend lifts a superblock EAGERLY on
    // first sight. When background tier-up is on, skip this — regions then form only
    // for proven-hot loops, off-thread (in the hotness path above), never the heavy
    // inline compile T3f flagged. A multi-block region compiles
    // as one unit spanning all its sub-blocks; a one-block region falls through to the
    // single-block path (reusing the block already lifted, so no double lift).
    if let Some(caps) = vm.backend.region_caps().filter(|_| !vm.tier_up_background) {
        let mut fetch_failure = None;
        let lifted = {
            let _section =
                EngineSection::enter(exit_word, sections, CriticalSectionKind::RegionBuild);
            lift_region_with_fault(&vm.mem, pc, caps, mode, &mut fetch_failure)
        };
        match lifted {
            // NOTE: the region path is JIT-only (a region-forming backend); Real16 stays
            // on the interpreter and never reaches here, so `pc` (flat) is correct.
            // Only a multi-block region *with a loop* is worth its heavier compile
            // (it amortizes over the iterations); everything else stays single-block.
            Ok(region) if region.blocks.len() > 1 && region.has_loop => {
                let spans = region.spans();
                if permit.as_ref().is_some_and(|p| !p.is_current()) {
                    return Err(ResolveFailure::Retry);
                }
                let materialized = {
                    let _section =
                        EngineSection::enter(exit_word, sections, CriticalSectionKind::Compile);
                    vm.hold_compile();
                    vm.backend.materialize_region_with_split_stores(
                        &region,
                        vm.consistency,
                        vm.mem.trap_window(),
                        vm.mem.guest_base(),
                        key.mxcsr_mode,
                        vm.precise_split_stores,
                    )
                };
                let _section =
                    EngineSection::enter(exit_word, sections, CriticalSectionKind::Publish);
                // §10: tag every sub-block's pages — under the spans lock (#12).
                if !vm
                    .cache
                    .insert_checked(key, materialized.clone(), spans, |sp| {
                        if vm.cache.epoch() != compile_epoch
                            || permit.as_ref().is_some_and(|p| !p.is_current())
                        {
                            return false;
                        }
                        for (start, len) in sp {
                            vm.mem.mark_code(*start, *len);
                        }
                        true
                    })
                {
                    return Err(ResolveFailure::Retry);
                }
                if matches!(materialized, CachedBlock::Compiled { .. }) {
                    vm.cache.record_region();
                }
                return Ok(materialized);
            }
            Ok(region) => {
                return finish_single(
                    vm,
                    exit_word,
                    sections,
                    key,
                    pc,
                    region.blocks.into_iter().next().unwrap(),
                    permit.as_ref(),
                    compile_epoch,
                )
                .ok_or(ResolveFailure::Retry)
            }
            Err(e) => {
                return Err(ResolveFailure::Exit(lift_exit(
                    e,
                    fetch_failure,
                    &vm.mem,
                    at,
                    mode,
                )))
            }
        }
    }
    let mut fetch_failure = None;
    let lifted = {
        let _section = EngineSection::enter(exit_word, sections, CriticalSectionKind::Lift);
        lift_block_with_fault(&vm.mem, at, mode, &mut fetch_failure)
    };
    match lifted {
        Ok(ir) => finish_single(
            vm,
            exit_word,
            sections,
            key,
            pc,
            ir,
            permit.as_ref(),
            compile_epoch,
        )
        .ok_or(ResolveFailure::Retry),
        Err(e) => Err(ResolveFailure::Exit(lift_exit(
            e,
            fetch_failure,
            &vm.mem,
            at,
            mode,
        ))),
    }
}

/// Materialize a single block, cache it under its `key` with its one span, and tag
/// its pages. `span_start` is the physical fetch address (== `ir.guest_start` outside
/// Real16, but the `cs_base + IP` physical address in Real16, where SMC page-tagging
/// and the cache key must both be physical — see `resolve`).
#[allow(clippy::too_many_arguments)]
fn finish_single(
    vm: &Vm,
    exit_word: &AtomicU64,
    sections: &AtomicU64,
    key: BlockKey,
    span_start: u64,
    ir: IrBlock,
    permit: Option<&crate::FetchPermit>,
    compile_epoch: u64,
) -> Option<CachedBlock> {
    if permit.is_some_and(|p| !p.is_current()) {
        return None;
    }
    let (start, len) = (span_start, ir.guest_len);
    // Real16 stays interpreted (§17.6); other modes defer compilation only until
    // the block proves hot under a tier-up policy (see `resolve`).
    let materialized = if key.mode == CpuMode::Real16 || vm.tier_up_after.is_some() {
        CachedBlock::Interpreted(Arc::new(ir))
    } else {
        let _section = EngineSection::enter(exit_word, sections, CriticalSectionKind::Compile);
        vm.hold_compile();
        vm.backend.materialize_with_split_stores(
            &ir,
            vm.consistency,
            vm.mem.trap_window(),
            vm.mem.guest_base(),
            key.mxcsr_mode,
            vm.precise_split_stores,
        )
    };
    let _section = EngineSection::enter(exit_word, sections, CriticalSectionKind::Publish);
    // §10: tag the block's pages under the spans lock, so the tag can't be cleared
    // by a concurrent SMC invalidation between insert and mark.
    vm.cache
        .insert_checked(key, materialized.clone(), vec![(start, len)], |_| {
            if vm.cache.epoch() != compile_epoch || permit.is_some_and(|p| !p.is_current()) {
                return false;
            }
            vm.mem.mark_code(start, len);
            true
        })
        .then_some(materialized)
}

/// Map a lift error to its dispatcher exit (a legal exit, not a `run()` failure).
fn lift_exit(
    e: LiftError,
    details: Option<crate::lift::FetchFailure>,
    mem: &Memory,
    at: FetchAddr,
    mode: CpuMode,
) -> Exit {
    match e {
        LiftError::Unsupported { addr, bytes, len } => {
            Exit::UnknownInstruction { addr, bytes, len }
        }
        LiftError::DecodeFault { addr } => crate::fault::fetch_exit(mem, at, mode, addr, details),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiled_chain_does_not_probe_exhaustion_per_edge() {
        struct ChainBackend(Arc<AtomicU64>);
        unsafe extern "C" fn chain(cpu: *mut u8, mem: *mut u8) -> u64 {
            // SAFETY: the dispatcher supplies live CpuState/MemCtx pointers per ABI.
            unsafe {
                (*(cpu as *mut CpuState)).gpr[0] += 1;
                (*(mem as *mut MemCtx)).next_entry = chain as *const () as u64;
            }
            RET_CHAIN
        }
        impl Backend for ChainBackend {
            fn code_space_exhausted(&self) -> bool {
                self.0.fetch_add(1, Ordering::Relaxed);
                false
            }

            fn materialize(
                &self,
                _: &IrBlock,
                _: MemConsistency,
                _: Option<(u64, u64)>,
                _: u64,
            ) -> CachedBlock {
                CachedBlock::Compiled {
                    entry: CompiledPtr(chain as *const u8),
                }
            }
        }
        let probes = Arc::new(AtomicU64::new(0));
        let mut vm = Vm::with_backend(
            VmConfig::flat(0x2000),
            Box::new(ChainBackend(probes.clone())),
        );
        vm.map(0x1000, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, &[0xeb, 0xfe]).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
        let before = probes.load(Ordering::Relaxed);
        assert!(matches!(cpu.run(&vm, Some(64)), Exit::BudgetExhausted));
        assert_eq!(cpu.reg(Reg::Rax), 65);
        assert!(
            probes.load(Ordering::Relaxed) - before <= 2,
            "exhaustion probes must not scale with compiled chain length"
        );
    }

    fn vcpu() -> Vcpu {
        let vm = Vm::new(VmConfig {
            memory_model: MemoryModel::Flat { size: 0x1000 },
            consistency: MemConsistency::Fast,
        });
        vm.new_vcpu()
    }

    /// A fresh Vm defaults to long mode; vcpus inherit it (§17.3).
    #[test]
    fn vm_defaults_to_long_mode() {
        let vm = Vm::new(VmConfig {
            memory_model: MemoryModel::Flat { size: 0x1000 },
            consistency: MemConsistency::Fast,
        });
        assert_eq!(vm.cpu_mode(), CpuMode::Long64);
        assert_eq!(vm.new_vcpu().mode, CpuMode::Long64);
    }

    #[test]
    fn set_cpu_mode_accepts_compat32() {
        let mut vm = Vm::new(VmConfig {
            memory_model: MemoryModel::Flat { size: 0x1000 },
            consistency: MemConsistency::Fast,
        });
        vm.set_cpu_mode(CpuMode::Compat32);
        assert_eq!(vm.cpu_mode(), CpuMode::Compat32);
        assert_eq!(vm.new_vcpu().mode, CpuMode::Compat32);
    }

    #[test]
    fn fork_gives_the_child_independent_memory() {
        let mut vm = Vm::new(VmConfig {
            memory_model: MemoryModel::Flat { size: 0x2000 },
            consistency: MemConsistency::Fast,
        });
        vm.map(0, 0x2000, Prot::RW, RegionKind::Ram).unwrap();
        vm.mem.write_bytes(0x100, &[1, 2, 3, 4]).unwrap();

        // Child inherits the snapshot...
        let child = vm
            .fork_with_backend(Box::new(InterpreterBackend))
            .expect("Flat memory is deep-copyable");
        let mut buf = [0u8; 4];
        child.mem.read_bytes(0x100, &mut buf).unwrap();
        assert_eq!(buf, [1, 2, 3, 4], "child sees the forked contents");

        // ...but writes don't cross over.
        child.mem.write_bytes(0x100, &[9, 9, 9, 9]).unwrap();
        let mut pbuf = [0u8; 4];
        vm.mem.read_bytes(0x100, &mut pbuf).unwrap();
        assert_eq!(pbuf, [1, 2, 3, 4], "parent memory unchanged by child write");
        let mut cbuf = [0u8; 4];
        child.mem.read_bytes(0x100, &mut cbuf).unwrap();
        assert_eq!(cbuf, [9, 9, 9, 9], "child memory changed");
    }

    #[test]
    fn gpr_roundtrip_through_index_map() {
        let mut c = vcpu();
        c.set_reg(Reg::Rax, 0xAA);
        c.set_reg(Reg::Rbx, 0xBB);
        c.set_reg(Reg::Rsp, 0x5050);
        c.set_reg(Reg::R15, 0xF15);
        assert_eq!(c.reg(Reg::Rax), 0xAA);
        assert_eq!(c.reg(Reg::Rbx), 0xBB);
        assert_eq!(c.reg(Reg::Rsp), 0x5050);
        assert_eq!(c.reg(Reg::R15), 0xF15);
    }

    #[test]
    fn gpr_writes_land_at_encoding_order_indices() {
        let mut c = vcpu();
        c.set_reg(Reg::Rbx, 0xB); // encoding index 3, not enum position 1
        assert_eq!(c.cpu.gpr[3], 0xB);
        assert_eq!(c.cpu.gpr[1], 0); // Rcx's slot untouched
    }

    #[test]
    fn rip_and_segment_bases_use_own_fields() {
        let mut c = vcpu();
        c.set_reg(Reg::Rip, 0x400000);
        c.set_reg(Reg::FsBase, 0x7fff_0000);
        c.set_reg(Reg::GsBase, 0x7fff_1000);
        assert_eq!(c.reg(Reg::Rip), 0x400000);
        assert_eq!(c.reg(Reg::FsBase), 0x7fff_0000);
        assert_eq!(c.reg(Reg::GsBase), 0x7fff_1000);
        assert_eq!(c.cpu.rip, 0x400000);
        assert_eq!(c.cpu.fs_base, 0x7fff_0000);
        assert_eq!(c.cpu.gs_base, 0x7fff_1000);
    }
}
