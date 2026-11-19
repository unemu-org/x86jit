//! `Vm` (shared) and `Vcpu` (per-thread) — the KVM-style split (§2), plus the
//! dispatcher loop (§9.2).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::cache::{BlockKey, CachedBlock, CompiledPtr, TranslationCache};
use crate::exit::{Exit, StepResult, UnsupportedSystemOperation};
use crate::interruption::{CriticalSectionKind, EngineSection};
use crate::ir::{IrBlock, IrRegion, RegionCaps};
use crate::jit_abi::{
    MemCtx, RetStack, RET_CHAIN, RET_CONTINUE, RET_EXCEPTION, RET_EXCEPTION_DETAIL, RET_HLT,
    RET_HOST_FAULT, RET_HOST_FAULT_DETAIL, RET_IBTC_MISS, RET_INTERRUPTED, RET_LINK,
    RET_MMIO_DEFER, RET_MXCSR_CHANGED, RET_PORTIO_DEFER, RET_RETURN_MISS, RET_SYSCALL,
    RET_UNMAPPED, RET_UNMAPPED_DETAIL,
};
mod i8086;
mod legacy;
mod nec;
use crate::legacy::{ExecutionProfile, LegacyCpuModel};
use crate::lift::{
    lift_block_watched, lift_region, lift_region_watched, CpuMode, FetchAddr, LiftError,
};
use crate::memory::{HostRam, MapError, MemError, Memory, MemoryModel, Prot, RegionKind};
use crate::state::{CpuState, Flags, Reg};
use crate::time_stats::{PrivateMiss, SharedReason};

/// Options for [`Vcpu::reset_execution_state`]. By default x87 state is preserved.
#[derive(Clone, Copy, Debug, Default)]
pub struct ResetOptions {
    x87_init: bool,
}

impl ResetOptions {
    /// Also initialize the x87 environment as FNINIT does (SDM Vol 2A §3.3,
    /// FINIT/FNINIT). The x87 data registers and MXCSR are preserved.
    pub fn with_x87_init(mut self, enabled: bool) -> Self {
        self.x87_init = enabled;
        self
    }
}

/// Options for instruction-exact Real16 execution (decision-16 §1).
/// Default execution stops on the budget or an ordinary exit.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepOptions {
    stop_when_irq_ready: bool,
}

impl StepOptions {
    /// Stop at the first boundary where [`Vcpu::irq_ready`] is true.
    /// A queued eligible IRQ is delivered first; delivery costs no step.
    /// This is a level check: resuming without injection can return ready again.
    pub fn with_stop_when_irq_ready(mut self, enabled: bool) -> Self {
        self.stop_when_irq_ready = enabled;
        self
    }
}

/// Completed accounting units and the reason execution stopped.
#[derive(Debug)]
pub struct StepRun {
    steps: u64,
    stop: StepStop,
}
impl StepRun {
    pub fn steps(&self) -> u64 {
        self.steps
    }
    pub fn stop(&self) -> &StepStop {
        &self.stop
    }
    pub fn into_stop(self) -> StepStop {
        self.stop
    }
}

/// A step limit uses `Exit::BudgetExhausted`; instruction exits take precedence.
#[derive(Debug)]
#[non_exhaustive]
pub enum StepStop {
    Exit(Exit),
    /// No unit is in flight and [`Vcpu::irq_ready`] is true.
    IrqReady,
}

/// Refusal before any execution or completion consumption.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StepRunError {
    UnsupportedMode,
    PendingPortIn,
}
impl std::fmt::Display for StepRunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::UnsupportedMode => "instruction-exact execution requires Real16",
            Self::PendingPortIn => "complete the pending port input before execution",
        })
    }
}
impl std::error::Error for StepRunError {}

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
    /// Worker timing snapshot; must not acquire a compiler lock. Wrappers forward it.
    fn time_stats(&self) -> crate::BackendTimeStats {
        crate::BackendTimeStats::default()
    }

    /// Enable detailed worker timing. Counts remain enabled independently.
    /// Wrappers should forward this setting.
    fn set_time_accounting(&self, _enabled: bool) {}

    /// Backend allocation counters. Implementations with no tracked storage use
    /// zeroes; wrappers should forward this method. Must not take compiler locks.
    fn memory_stats(&self) -> crate::BackendMemoryStats {
        crate::BackendMemoryStats::default()
    }

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

    /// Compiling backends wrap materialization in
    /// [`crate::time_stats::ForegroundCompileTimer`] for per-vCPU timing.
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
    /// Legacy-context IR must be rejected before entering application codegen.
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
        assert!(
            ir.legacy_context.is_none(),
            "legacy IR cannot enter application materialization"
        );
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
        assert!(
            region.blocks.iter().all(|b| b.legacy_context.is_none()),
            "legacy IR cannot enter application materialization"
        );
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
        assert!(
            ir.legacy_context.is_none(),
            "legacy IR cannot enter application materialization"
        );
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
        assert!(
            region.blocks.iter().all(|b| b.legacy_context.is_none()),
            "legacy IR cannot enter application materialization"
        );
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

    /// Lifetime domain for compiled entries and link slots. A wrapping backend
    /// must forward this method to its wrapped backend along with materialization.
    /// Without forwarding, the VM cache uses a separate domain and refuses the
    /// wrapped backend's tracked compiled entries. Cold resolution instead caches
    /// and executes the original single-block IR (or the first block of an eager
    /// region) through the interpreter; rejected tier-up leaves the existing
    /// interpreted block in place. No signal-handler use.
    fn code_lifetime(&self) -> Option<Arc<crate::code_lifetime::CodeLifetime>> {
        None
    }

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
    /// of the dispatch loop; each result carries the source-publication generation
    /// from its request, so a stale compile is rejected on publish.
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
    /// Opaque request identity echoed in the completion. The dispatcher tracks
    /// this request's source spans and cancels it on overlapping invalidation.
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

/// Why a backend could not compile or publish a translation unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompileFailureReason {
    /// The unit exceeds the backend's guest-access descriptor capacity.
    DescriptorLimit,
    /// Executable storage, metadata allocation, or code-space budget exhausted.
    CodeSpace,
    /// Invalid backend metadata or another deterministic backend error, including
    /// JIT admission refused for a marked shared-address-space child PID.
    OtherBackendError,
    /// A failure the backend cannot attribute to the unit, such as a panic on a
    /// host error during compilation or publication. Counted with other backend
    /// errors, but never suppresses a later request for the same unit.
    Unclassified,
}

impl CompileFailureReason {
    /// These failures cannot be repaired by compiling the same source again.
    pub fn is_deterministic(self) -> bool {
        matches!(self, Self::DescriptorLimit | Self::OtherBackendError)
    }
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
    /// is cleared. A classified deterministic failure suppresses further tier-up
    /// attempts until any of the unit's source bytes are invalidated.
    pub block: CachedBlock,
    /// Failure reason when `block` is interpreted. `None` supports backends that
    /// cannot classify failures, and canceled fetch permits. Unknown failures are
    /// counted as other backend errors but do not suppress future requests.
    pub compile_failure: Option<CompileFailureReason>,
    /// The guest byte span(s) `(start, len)` — one per sub-block for a region.
    pub spans: Vec<(u64, u32)>,
    /// The opaque request identity, echoed unchanged from the request.
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
        assert!(
            ir.legacy_context.is_none(),
            "legacy IR cannot enter application materialization"
        );
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
    /// Pair with [`Vm::with_backend_host_ram`]. Through [`Vm::with_backend`] the
    /// whole span is instead one zeroed allocation, and a failed allocation aborts
    /// the process as [`Memory::new`] documents.
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
    /// Compiler workers joined (legacy counter; maintenance joins are excluded).
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
    time_enabled: bool,
    time_registry: Mutex<crate::time_stats::Registry>,
    vcpu_lifetime: Arc<()>,
    vcpu_memory: Mutex<Vec<std::sync::Weak<VcpuAllocations>>>,
    precise_split_stores: bool,
    compile_hold: Option<(std::time::Duration, Arc<std::sync::atomic::AtomicBool>)>,
    #[cfg(test)]
    source_read_hook: Mutex<Option<fn(&Vm)>>,
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
    profile: ExecutionProfile,
    legacy_model: Option<LegacyCpuModel>,
    exception_delivery: crate::ExceptionDelivery,
    cpu_model: Option<crate::CpuModel>,
    fpu_model: crate::FpuModel,
    profile_locked: std::sync::atomic::AtomicBool,
}

impl Vm {
    /// Sum cumulative vCPU counters without a compiler lock. See [`crate::TimeStats`]
    /// for units, timing methods, snapshot lag and overlapping categories.
    pub fn time_stats(&self) -> crate::TimeStats {
        let mut stats = self.time_registry.lock().unwrap().snapshot();
        stats.background_compile_ns = self.backend.time_stats().background_compile_ns;
        let (ns, count) = self.cache.tier_up_time_stats();
        stats.tier_up_latency_ns = ns;
        stats.tier_up_latency_count = count;
        stats
    }

    /// Enable transition timing and detailed helper accounting (default false).
    /// Dispatch and exit counters remain enabled; helper counts, split resolve
    /// counts and detailed timing are collected only while enabled.
    /// Existing totals are retained; subsequent run
    /// entries use the new setting. Outside intervals spanning a disabled run are excluded.
    /// Does not change generated code. See [`crate::TimeStats`] for precision.
    pub fn set_time_accounting(&mut self, enabled: bool) {
        self.time_enabled = enabled;
        self.mem.time_enabled = enabled;
        self.backend.set_time_accounting(enabled);
    }

    /// Snapshot allocation categories without stopping execution. See
    /// [`crate::MemoryStats`] for precision, scope and omitted allocations.
    pub fn memory_stats(&self) -> crate::MemoryStats {
        let mut vcpus = crate::VcpuMemoryStats::default();
        self.vcpu_memory.lock().unwrap().retain(|weak| {
            let Some(allocation) = weak.upgrade() else {
                return false;
            };
            vcpus.count += 1;
            vcpus.state_bytes += (std::mem::size_of::<Vcpu>()
                + std::mem::size_of::<RetStack>()
                + std::mem::size_of::<crate::jit_abi::fault::EntryRecord>()
                + 2 * std::mem::size_of::<AtomicU64>()
                + std::mem::size_of::<VcpuAllocations>()
                + std::mem::size_of::<crate::time_stats::Record>()
                + 8 * std::mem::size_of::<usize>()) as u64;
            vcpus.fast_resolve_bytes += std::mem::size_of::<[FastEntry; FAST_N]>() as u64;
            vcpus.scratch_bytes += allocation.scratch.load(Ordering::Relaxed);
            vcpus.refill_bytes += allocation.refills.load(Ordering::Relaxed);
            true
        });
        let (guest_ram_reserved, guest_ram_committed, memory_tracking_bytes) =
            self.mem.allocation_stats();
        let (source_barrier_mode, source_barrier_failures, source_barrier_last_errno) =
            self.mem.source_barrier.stats();
        crate::MemoryStats {
            source_barrier_mode,
            source_barrier_failures,
            source_barrier_last_errno,
            source_barrier_uncached_executions: self
                .mem
                .source_barrier
                .uncached_executions
                .load(Ordering::Relaxed),
            backend: self.backend.memory_stats(),
            cache: self.cache.memory_stats(),
            vcpus,
            codemap_bytes: crate::codemap::memory_bytes(),
            guest_ram_reserved,
            guest_ram_committed,
            memory_tracking_bytes,
            source_barrier_calls: self.mem.source_barrier.calls.load(Ordering::Relaxed),
            source_barrier_ns: self.mem.source_barrier.ns.load(Ordering::Relaxed),
            source_barrier_tagged_pages: self
                .mem
                .source_barrier
                .tagged_pages
                .load(Ordering::Relaxed),
        }
    }

    /// Snapshot child execution metadata before a shared-address-space fork.
    /// Allocation and fetch-policy callbacks run here, in the parent. `mode`
    /// selects decoding; Real16 helpers are outside the initial child scope.
    /// The view owns mapping/CPU metadata and borrows stable atomic code tags and
    /// a store-notification bridge. Its lifetime keeps this
    /// VM's mapping alive and preventing safe unmap/remap until the view is dropped.
    /// See the crate docs for the FFI and parent invalidation protocol.
    pub fn child_view(&self, mode: CpuMode) -> crate::ChildView<'_> {
        // Register atfork in the parent, never on the allocation-free child path.
        let _ = crate::codemap::ProcessIdentity::current();
        let mut view = self.mem.child_view(self, mode);
        view.code.consistency = self.consistency;
        view
    }

    /// Attach a caller's sparse child snapshot to this running VM's SMC bridge.
    /// RAM intervals must lie in parent RAM and name the same backing offsets;
    /// traps may narrow any mapped parent interval. Holes, frozen fetch verdicts
    /// and extra store-refusal tags come from `metadata`, without consulting the
    /// parent's fetch callback or enumerating its broad mappings. The journal
    /// uses one bit per child RAM page, rounded to 64 bits per contiguous interval.
    /// CPU policy, live tags, notifications, alias retention and drop accounting
    /// all belong to this VM. Existing parent code tags also refuse child stores.
    ///
    /// Call in the parent under the embedder's mapping synchronization before
    /// fork. Native permission snapshots are the caller's responsibility. Like
    /// [`Self::child_view`], drop the view in the parent after child exit/exec and
    /// final written-page reconciliation on this VM.
    pub fn child_view_with(
        &self,
        metadata: crate::ChildViewMetadata,
        mode: CpuMode,
    ) -> Result<crate::ChildView<'_>, crate::ChildViewError> {
        let _ = crate::codemap::ProcessIdentity::current();
        let mut view = self.mem.child_view_with(self, metadata, mode)?;
        view.code.consistency = self.consistency;
        Ok(view)
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
    /// Dense JIT arenas are excluded from host-fork inheritance: compiled entries
    /// and provider state from the parent are unavailable in the child. A new
    /// backend reserves fresh storage there. The first child code registration
    /// starts a fresh signal-safe code map; parent records remain retained but
    /// inactive. Translation reclamation is a separate contract.
    pub fn shutdown(mut self) -> Result<ShutdownReport, ShutdownError> {
        let live_vcpus = Arc::strong_count(&self.vcpu_lifetime) - 1;
        let mut report = self.backend.shutdown();
        if let Err(error) = self.cache.join_reclamation_workers() {
            let counted = matches!(
                report,
                Err(ShutdownError::LiveResources {
                    resource: "code reclamation",
                    ..
                })
            );
            let mut counts = match report {
                Ok(counts)
                | Err(ShutdownError::LiveResources { report: counts, .. })
                | Err(ShutdownError::WorkerPanicked { report: counts }) => counts,
            };
            report = Err(match error {
                ShutdownError::LiveResources { resource, .. } => {
                    // The backend may already have counted this same domain.
                    if !counted {
                        counts.outstanding_handles += 1;
                    }
                    ShutdownError::LiveResources {
                        resource,
                        report: counts,
                    }
                }
                ShutdownError::WorkerPanicked { .. } => {
                    ShutdownError::WorkerPanicked { report: counts }
                }
            });
        }
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
    /// Write notifications revoke page-wide approval and evict every translation
    /// on the notified page. Without a policy, writes invalidate only overlapping
    /// source bytes; compiled execution and dispatch retain their existing paths.
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

    #[cfg(test)]
    fn after_source_read(&self) {
        let hook = self.source_read_hook.lock().unwrap().take();
        if let Some(hook) = hook {
            hook(self);
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
        if self.profile == ExecutionProfile::Machine || self.mode == CpuMode::Real16 {
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
        let generation = self.cache.publication_generation();
        let permit = self.mem.fetch_permit();
        let mut failure = None;
        let source = crate::memory::LiftSource::new(&self.mem);
        let region = lift_region_watched(&self.mem, entry, caps, self.mode, &mut failure, &source)
            .map_err(|e| {
                ForceRegionError::Lift(lift_exit(
                    e,
                    failure,
                    &self.mem,
                    FetchAddr::flat(entry),
                    self.mode,
                ))
            })?;
        if !source.publishable() {
            return Err(ForceRegionError::Invalidated);
        }
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
        self.handle_smc(&AtomicU64::new(0));
        if !self
            .cache
            .insert_checked(key, compiled, region.spans(), |spans| {
                if self.cache.publication_generation() != generation
                    || permit.as_ref().is_some_and(|p| !p.is_current())
                {
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
    /// Long64/Compat32 select Application. Real16 selects the frozen internal
    /// Machine entry while reporting Application(Real16) (decision-23 §6, M4).
    /// Its hidden caches are derived from visible selectors at run entry.
    /// Clears any previously selected public CPU model.
    /// Existing Application vCPUs retain their contexts when this selects the
    /// mode for later vCPUs. Panics when leaving Machine after vCPU creation.
    pub fn set_cpu_mode(&mut self, mode: CpuMode) {
        assert!(
            self.profile != ExecutionProfile::Machine
                || !self.profile_locked.load(Ordering::Relaxed),
            "select execution profile before vCPU creation"
        );
        self.profile = ExecutionProfile::Application(mode);
        self.legacy_model = None;
        self.cpu_model = None;
        self.mode = mode;
    }

    /// Select a catalogue CPU model and Machine execution (decision-23 §2).
    /// Panics after vCPU creation, including after all such vCPUs are dropped.
    ///
    /// Selects [`crate::ExceptionDelivery::InGuest`] (decision-23 §5). Call
    /// [`Self::set_exception_delivery`] afterwards to override exception routing.
    pub fn set_cpu_model(&mut self, model: crate::CpuModel) {
        self.select_machine_model(model.legacy_model());
        self.cpu_model = Some(model);
        self.exception_delivery = crate::ExceptionDelivery::InGuest;
    }

    /// Set the FPU default inherited by new vCPUs (decision-23 §4).
    /// Defaults to [`crate::FpuModel::Present`]. The per-vCPU setter remains an
    /// override. Panics after any vCPU has been created, even if later dropped.
    pub fn set_fpu_model(&mut self, model: crate::FpuModel) {
        assert!(
            !self.profile_locked.load(Ordering::Relaxed),
            "select FPU default before vCPU creation"
        );
        self.fpu_model = model;
    }

    /// The default FPU policy inherited by new vCPUs (decision-23 §4).
    pub fn fpu_model(&self) -> crate::FpuModel {
        self.fpu_model
    }

    /// Compatibility selector with Outward exception delivery (decision-23 §5).
    /// Preserves the supplied generation and semantics identity exactly.
    /// Panics after vCPU creation, including after all such vCPUs are dropped.
    /// Call [`Self::set_exception_delivery`] afterwards to override exception routing.
    #[deprecated(note = "use Vm::set_cpu_model with CpuModel (decision-23 §2)")]
    pub fn set_legacy_cpu_model(&mut self, model: LegacyCpuModel) {
        self.select_machine_model(model);
        self.cpu_model = None;
    }

    fn select_machine_model(&mut self, model: LegacyCpuModel) {
        assert!(
            !self.profile_locked.load(Ordering::Relaxed),
            "select execution profile before vCPU creation"
        );
        self.profile = ExecutionProfile::Machine;
        self.legacy_model = Some(model);
        self.exception_delivery = crate::ExceptionDelivery::Outward;
    }

    /// Select processor-exception routing before creating any vCPU.
    /// Applies to Machine in both real and protected mode (decision-23 §5).
    /// Software INT always uses the guest table. Application profiles ignore it.
    pub fn set_exception_delivery(&mut self, delivery: crate::ExceptionDelivery) {
        assert!(
            !self.profile_locked.load(Ordering::Relaxed),
            "select exception delivery before vCPU creation"
        );
        self.exception_delivery = delivery;
    }

    pub fn exception_delivery(&self) -> crate::ExceptionDelivery {
        self.exception_delivery
    }

    pub fn execution_profile(&self) -> ExecutionProfile {
        self.profile
    }

    /// The selected catalogue model, or `None` for Application and the legacy
    /// compatibility selector (decision-23 §2, Migration M1).
    pub fn cpu_model(&self) -> Option<crate::CpuModel> {
        self.cpu_model
    }

    pub fn legacy_cpu_model(&self) -> Option<LegacyCpuModel> {
        self.legacy_model
    }

    /// Application decode metadata (§17.3), unchanged by selecting Machine.
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
            time_enabled: false,
            time_registry: Mutex::default(),
            vcpu_lifetime: Arc::new(()),
            vcpu_memory: Mutex::new(Vec::new()),
            precise_split_stores: self.precise_split_stores,
            compile_hold: None,
            #[cfg(test)]
            source_read_hook: Mutex::new(None),
            mem: self.mem.deep_copy()?,
            cache: {
                let mut cache = TranslationCache::with_code_lifetime(backend.code_lifetime());
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
            profile: self.profile,
            legacy_model: self.legacy_model,
            exception_delivery: self.exception_delivery,
            cpu_model: self.cpu_model,
            fpu_model: self.fpu_model,
            profile_locked: std::sync::atomic::AtomicBool::new(false),
            lockstep_trace: self.lockstep_trace.clone(),
        })
    }

    /// Assemble a fresh `Vm` around already-built guest memory — the shared body of the
    /// public constructors, holding the default tier-up policy and CPU features in one place.
    fn from_mem(mem: Memory, backend: Box<dyn Backend>, consistency: MemConsistency) -> Self {
        Self {
            time_enabled: false,
            time_registry: Mutex::default(),
            vcpu_lifetime: Arc::new(()),
            vcpu_memory: Mutex::new(Vec::new()),
            precise_split_stores: false,
            compile_hold: None,
            #[cfg(test)]
            source_read_hook: Mutex::new(None),
            mem,
            cache: TranslationCache::with_code_lifetime(backend.code_lifetime()),
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
            profile: ExecutionProfile::Application(CpuMode::Long64),
            legacy_model: None,
            exception_delivery: crate::ExceptionDelivery::Outward,
            cpu_model: None,
            fpu_model: crate::FpuModel::Present,
            profile_locked: std::sync::atomic::AtomicBool::new(false),
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
    /// for misaligned backing, a backing that cannot contain the guest extent, or a
    /// span above `1 << 47`. An identity mapping therefore needs a host mapping below
    /// `1 << 47` (see [`HostRam::guest_base`](crate::memory::HostRam::guest_base)).
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

    /// Process pending self-modifying-code writes (§10): drop units overlapping
    /// written bytes, retaining page tags while any cached unit remains. With a
    /// fetch policy, invalidate whole pages because fetch approvals are page-wide.
    ///
    /// Ordinary stores report through `Memory::note_write`; child stores use the
    /// atomic journal. Both expand writes through declared aliases when drained.
    /// Registration/publication also drain here: a queued notification has not yet
    /// advanced the source generation or canceled pending source spans.
    fn handle_smc(&self, word: &AtomicU64) {
        self.mem.dispatch_smc(word, |dirty| {
            let mut invalidated = false;
            for (lo, hi) in dirty {
                let page = lo >> crate::memory::CODE_PAGE_BITS;
                let (lo, hi) = if self.mem.fetch_policy.is_some() {
                    let start = page << crate::memory::CODE_PAGE_BITS;
                    (start, start.saturating_add(crate::memory::CODE_PAGE_SIZE))
                } else {
                    (lo, hi)
                };
                // A dropped unit's inbound link slots (in other, surviving blocks) still
                // point at its now-stale compiled code. Note whether anything was
                // dropped so we can clear the backend's slots once, below.
                // The page tag is cleared *inside* `invalidate_written`, under the
                // spans lock and only if no block still spans the page — so it can't race
                // a concurrent insert's mark.
                invalidated |= !self
                    .cache
                    .invalidate_written(lo, hi, |empty| {
                        if empty {
                            self.mem.clear_code_page(page);
                        }
                    })
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
        if let Some(model) = self.cpu_model {
            assert!(
                model.profile().fpu.contains(&self.fpu_model),
                "FPU default is incompatible with the CPU model"
            );
        }
        self.profile_locked.store(true, Ordering::Relaxed);
        let mut cpu = if self.profile == ExecutionProfile::Machine {
            CpuState::legacy_real_entry([0; 6])
        } else {
            CpuState::new()
        };
        cpu.fpu_model = self.fpu_model;
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
        let allocations = Arc::new(VcpuAllocations::default());
        let mut registry = self.vcpu_memory.lock().unwrap();
        registry.retain(|weak| weak.strong_count() != 0);
        registry.push(Arc::downgrade(&allocations));
        drop(registry);
        Vcpu {
            time_dispatch: crate::time_stats::DispatchCounts::default(),
            time_record: self.time_registry.lock().unwrap().register(),
            time_outside: None,
            allocations: allocations.clone(),
            _vm_lifetime: self.vcpu_lifetime.clone(),
            cpu,
            exit_request: ExitRequest { word },
            interruption_evidence: None,
            syscall_evidence: None,
            engine_sections: AtomicU64::new(0),
            mode: self.mode, // decode/lift mode the embedder chose (§17.3)
            profile: self.profile,
            legacy_model: self.legacy_model,
            exception_delivery: if self.profile == ExecutionProfile::Application(CpuMode::Real16) {
                crate::legacy::REAL16_ALIAS.exception_delivery
            } else {
                self.exception_delivery
            },
            cpu_model: self.cpu_model,
            legacy_execution: (self.profile == ExecutionProfile::Machine)
                .then(|| Box::new(legacy::Execution::new())),
            fast: Box::new(
                [FastEntry {
                    rip: 0,
                    entry: CompiledPtr::new(std::ptr::null()),
                }; FAST_N],
            ),
            #[cfg(test)]
            ibtc_test_hook: None,
            fast_epoch: self.cache.epoch(),
            fast_lifetime: self.cache.code_lifetime().clone(),
            execution_participant: self.cache.code_lifetime().participant(),
            fast_mxcsr_mode: 0,
            fast_empty_reason: PrivateMiss::Cold,
            ibtc_refills: HashMap::new(),
            ret_stack: Box::new(RetStack::new()),
            fault_entry: Box::default(),
            fast_hits: 0,
            chained: 0,
            executed: 0,
            interp_scratch: VcpuScratch {
                values: Box::default(),
                allocations,
                #[cfg(test)]
                fill_before_block: None,
                #[cfg(test)]
                block_entries: 0,
                #[cfg(test)]
                nmi_before_block: None,
            },
            pending_irq: None,
            retired: 0,
            sti_shadow: false,
            code_space_reported: false,
            ss_shadow: 0,
            nmi_state: 0,
            delivery_log: None,
        }
    }
}

impl Drop for Vm {
    fn drop(&mut self) {
        // Join before releasing either the cache or guest-memory views.
        let _ = self.backend.shutdown();
    }
}

/// Two ways in each of 1024 sets: 2048 entries × 24 bytes = 48 KiB per vcpu.
const FAST_BITS: u32 = 10;
const FAST_N: usize = 2 << FAST_BITS;

/// After this many IBTC refills a site is treated as megamorphic: the
/// dispatcher stops refilling its slot (it stays empty → the site pays the
/// baseline dispatch forever) so a polymorphic indirect branch can't churn the
/// descriptor arena without bound.
const IBTC_MEGAMORPHIC_CAP: u32 = 8;

/// One fast-resolve entry: a guest RIP tag and its compiled entry.
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

/// The NMI edge latch of a Machine vCPU, in its dispatch word (spec.md §17.10).
/// Application vCPUs never carry it: their compiled code exits on any nonzero
/// word. Bit 33, so that `EXIT_REQUESTED | NMI_EDGE` (2_0000_0002h, one bit
/// repeated every 32) is a single AArch64 test immediate: the boundary test
/// keeps its one load and one test.
pub(crate) const NMI_EDGE: u64 = 1 << 33;

/// The Vcpu `ss_shadow` bits: the MOV/POP SS class that holds INT and the
/// single-step trap (decision-16 §4), and the hold of the NMI input that every
/// segment-register load arms (spec.md §17.10).
pub(crate) const SS_HOLD: u8 = 1;
pub(crate) const NMI_HOLD: u8 = 2;

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

    /// Whether a request is pending, without consuming it. Code that the
    /// embedder runs between calls, such as an 8080 core during
    /// [`Exit::I8080Mode`], ORs this into its own stop test. The request stays
    /// pending, and the next `run` or positive `run_steps` consumes it.
    ///
    /// Async-signal-safe for the same reasons as [`Self::request_exit`]: one
    /// lock-free atomic load of a word the handle already owns. The Acquire load
    /// makes writes that preceded the request visible.
    pub fn is_requested(&self) -> bool {
        self.word.load(Ordering::Acquire) & EXIT_REQUESTED != 0
    }
}

/// Thread-safe, cloneable handle on the NMI input of an
/// [`ExecutionProfile::Machine`] vCPU (spec.md §17.10). Each [`Self::raise`]
/// is one rising edge. Edges coalesce until the model accepts one at an
/// instruction or REP-iteration boundary; no exit is returned for it. A handle
/// that outlives its vCPU has no effect.
#[derive(Clone, Debug)]
pub struct NmiLine {
    word: Arc<AtomicU64>,
}

impl NmiLine {
    /// Latch one rising edge. A running vCPU observes it at its next
    /// instruction or REP-iteration boundary. Writes preceding this call are
    /// visible to the guest's NMI handler.
    ///
    /// Async-signal-safe: one lock-free atomic read-modify-write on a word the
    /// handle already owns, like [`ExitRequest::request_exit`]. Cloning or
    /// dropping an `NmiLine` is not async-signal-safe.
    pub fn raise(&self) {
        // Release publishes preceding writes; acceptance's Acquire RMW pairs.
        self.word.fetch_or(NMI_EDGE, Ordering::Release);
    }

    /// The raw latch: an edge raised and not yet accepted or discarded. It can
    /// report an edge that the model will ignore (an I80386 inside its NMI
    /// handler) or that is not deliverable yet; [`Vcpu::has_pending_nmi`] is
    /// the model's view. Code that the embedder runs between calls, such as
    /// an 8080 core during [`Exit::I8080Mode`], ORs this into its stop test.
    ///
    /// Async-signal-safe: one lock-free Acquire load.
    pub fn is_raised(&self) -> bool {
        self.word.load(Ordering::Acquire) & NMI_EDGE != 0
    }
}

/// [`Vcpu::raise_nmi`] on a vCPU that is not [`ExecutionProfile::Machine`]:
/// the Application profiles have no NMI input (spec.md §17.10).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NmiUnsupported;

impl std::fmt::Display for NmiUnsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the NMI input exists only on ExecutionProfile::Machine vCPUs")
    }
}

impl std::error::Error for NmiUnsupported {}

#[derive(Default)]
struct VcpuAllocations {
    scratch: AtomicU64,
    refills: AtomicU64,
}

struct VcpuScratch {
    values: Box<[u64]>,
    allocations: Arc<VcpuAllocations>,
    #[cfg(test)]
    fill_before_block: Option<u64>,
    #[cfg(test)]
    block_entries: u64,
    // Raises an NMI edge at each of the next `.1` block entries: after the
    // dispatcher's word test and before the walk's first boundary.
    #[cfg(test)]
    nmi_before_block: Option<(NmiLine, u64)>,
}

impl VcpuScratch {
    #[cold]
    fn grow(&mut self, count: usize) {
        let capacity = count.max(self.values.len() * 2).max(4);
        self.values = vec![0; capacity].into_boxed_slice();
        self.allocations.scratch.store(
            (capacity * std::mem::size_of::<u64>()) as u64,
            Ordering::Relaxed,
        );
    }
}

impl crate::interp::ScratchBuffer for VcpuScratch {
    #[inline]
    fn zeroed(&mut self, count: usize) -> &mut [u64] {
        let temps = self.reused(count);
        temps.fill(0);
        temps
    }

    #[inline]
    fn reused(&mut self, count: usize) -> &mut [u64] {
        if count > self.values.len() {
            self.grow(count);
        }
        #[cfg(test)]
        if let Some((line, count @ 1..)) = &mut self.nmi_before_block {
            *count -= 1;
            line.raise();
        }
        #[cfg(test)]
        if let Some(pattern) = self.fill_before_block {
            self.block_entries += 1;
            for (index, slot) in self.values[..count].iter_mut().enumerate() {
                *slot = if pattern == 0 {
                    0
                } else {
                    pattern.rotate_left((self.block_entries as u32) % 64)
                        ^ (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                };
            }
        }
        &mut self.values[..count]
    }
}

/// Per-guest-thread execution context: CPU state + its own `run()` loop (§2).
pub struct Vcpu {
    time_dispatch: crate::time_stats::DispatchCounts,
    time_record: Arc<crate::time_stats::Record>,
    time_outside: crate::time_stats::Outside,
    allocations: Arc<VcpuAllocations>,
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
    profile: ExecutionProfile,
    legacy_model: Option<LegacyCpuModel>,
    exception_delivery: crate::ExceptionDelivery,
    cpu_model: Option<crate::CpuModel>,
    legacy_execution: Option<Box<legacy::Execution>>,
    /// Fast-resolve cache: a vcpu-private, two-way RIP→compiled entry map
    /// that replaces the shared lookup for transfers the chain loop can't chain — returns,
    /// indirect jumps, and cold outer-loop re-dispatch. Only `Compiled` entries are
    /// cached; interpreted blocks always route through `resolve` (so the cache
    /// counters stay meaningful). Flushed whenever the cache invalidation epoch
    /// moves — the coherence channel for cross-thread SMC that `invalidate_links`
    /// (backend-owned slots) cannot reach.
    fast: Box<[FastEntry; FAST_N]>,
    #[cfg(test)]
    ibtc_test_hook: Option<Box<dyn FnMut(bool) + Send>>,
    /// Snapshot of `TranslationCache::epoch` matching the current `fast` contents.
    fast_epoch: u64,
    fast_lifetime: Arc<crate::code_lifetime::CodeLifetime>,
    execution_participant: crate::code_lifetime::ExecutionParticipant,
    fast_mxcsr_mode: u32,
    fast_empty_reason: PrivateMiss,
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
    /// `temp_count`, avoiding a per-dispatch allocation. Real16 blocks define temps
    /// before use and retain scratch values; other modes zero the buffer per block.
    interp_scratch: VcpuScratch,
    /// Pending maskable hardware-interrupt vector (§17.6, sub-seam c). Set by
    /// [`Vcpu::inject_irq`]; delivered at a `run()` boundary through the real-mode IVT
    /// once IF is set, no memory/port completion is outstanding, and both shadows have
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
    /// STI inhibition persists through incomplete successors and is consumed at
    /// completion or event delivery (SDM Vol 3A §7.8.1; Vol 2B §4.3 STI).
    sti_shadow: bool,
    /// MOV/POP SS inhibition, kept distinct for consecutive-load policy
    /// (SDM Vol 3A §7.8.3; decision-16 §4), in bit 0 ([`SS_HOLD`]); bit 1
    /// ([`NMI_HOLD`]) holds the NMI input after every segment-register load,
    /// rearmed by consecutive loads (spec.md §17.10). Interpreter-only,
    /// outside `CpuState`.
    ss_shadow: u8,
    /// NMI blocking of the I80286 and I80386 (spec.md §17.10):
    /// [`legacy::NMI_BLOCKED`] from acceptance to the next IRET, and on the
    /// I80286 [`legacy::NMI_REMEMBERED`], one edge raised meanwhile. Private,
    /// outside `CpuState` and the retained-context stamp.
    nmi_state: u8,
    /// The opt-in record of in-guest deliveries (spec.md §17.10, Delivery log):
    /// `None` unless [`Vcpu::enable_delivery_log`] started it. Host-side
    /// diagnostics, outside `CpuState` and `jit_abi`; written only by the cold
    /// recording calls at the Machine delivery sites.
    delivery_log: Option<Box<crate::DeliveryLog>>,
}

const APPLICATION_DISPATCH: u8 = 0;
const ALIAS_DISPATCH: u8 = 1;
const MACHINE_DISPATCH: u8 = 2;
const DIRECT20_DISPATCH: u8 = 3;

enum DispatchResult {
    Application(Result<Exit, u64>),
    Stepped(StepRun),
}
impl DispatchResult {
    fn application(self) -> Result<Exit, u64> {
        match self {
            Self::Application(result) => result,
            Self::Stepped(_) => unreachable!(),
        }
    }
    fn stepped(self) -> StepRun {
        match self {
            Self::Stepped(result) => result,
            Self::Application(_) => unreachable!(),
        }
    }
}

struct DispatchGuard<'a> {
    cpu: &'a mut Vcpu,
    vm: &'a Vm,
    fast_before: u64,
}

impl<'a> DispatchGuard<'a> {
    fn new(cpu: &'a mut Vcpu, vm: &'a Vm) -> Self {
        let fast_before = cpu.fast_hits;
        Self {
            cpu,
            vm,
            fast_before,
        }
    }
}

impl DispatchGuard<'_> {
    /// The run's chain and dispatch counts, with the shared cache's totals
    /// updated; `drop` publishes them to the run's time statistics.
    #[inline(always)]
    fn take_counts(&mut self) -> (u64, crate::time_stats::DispatchCounts) {
        let counts = self.cpu.finish_dispatch_counts(self.vm, self.fast_before);
        let chains = std::mem::take(&mut self.cpu.chained);
        self.vm.cache.add_chained(chains);
        (chains, counts)
    }

    /// The normal end of a run: the counts go to `finish`, in the same
    /// thread-local access as the run's exit (`RunGuard::finish_with_counts`).
    #[inline(always)]
    fn finish(
        mut self,
        time_run: &crate::time_stats::RunGuard,
        exit: Option<&Exit>,
    ) -> crate::time_stats::Outside {
        let (chains, counts) = self.take_counts();
        std::mem::forget(self);
        time_run.finish_with_counts(exit, chains, &counts)
    }
}

impl Drop for DispatchGuard<'_> {
    fn drop(&mut self) {
        let (chains, counts) = self.take_counts();
        crate::time_stats::dispatch_counts(chains, counts);
    }
}

impl Vcpu {
    /// Abandon pending execution and device completions between runs, in any
    /// execution profile. Clears queued IRQs, STI/SS interrupt shadows, IN/INS and
    /// MMIO handoffs, accepted legacy IRQ deliveries, host exit requests and their
    /// evidence, syscall evidence, host-fault records, and per-vCPU fast lookup,
    /// IBTC refill and return-stack predictions. Shared translations and their
    /// validated links are retained. A Machine vCPU also loses its NMI latch,
    /// its NMI hold and its NMI blocking (spec.md §17.10); an edge racing with
    /// this call is discarded if it precedes the atomic latch clear.
    ///
    /// Call this first, then write the CPU model's reset registers into [`Self::cpu`]
    /// (including CS and IP), then resume execution. This method supplies no CPU-model
    /// reset register table. Architectural state is preserved except for the optional
    /// FNINIT environment selected by [`ResetOptions::with_x87_init`].
    /// Retirement, step/execution and diagnostic accounting, source identity,
    /// configuration, memory mappings, and code caches are preserved, and so is
    /// the delivery log ([`Self::enable_delivery_log`]) with its records and
    /// counts: a reset is not a delivery, and nothing is recorded for it. The embedder's
    /// A20 request is preserved because it is a board signal; a queued change
    /// applies when reset abandons its instruction. Existing exit
    /// and fault handles remain valid. An exit request racing with this call is
    /// discarded if it precedes the atomic request clear; later requests remain pending.
    pub fn reset_execution_state(&mut self, options: ResetOptions) {
        crate::interruption::consume(&self.exit_request.word, &self.engine_sections);
        // spec.md §17.10: reset ends NMI blocking (ASSUMED; no PRM states it).
        self.exit_request
            .word
            .fetch_and(!NMI_EDGE, Ordering::Acquire);
        self.nmi_state = 0;
        self.interruption_evidence = None;
        self.syscall_evidence = None;
        self.pending_irq = None;
        if let Some(execution) = &mut self.legacy_execution {
            execution.reset_execution_state();
        }
        self.sti_shadow = false;
        self.ss_shadow = 0;
        self.cpu.pending_port_in = None;
        self.cpu.port_string_in = None;
        self.cpu.pending_mmio = None;
        self.cpu.pending_mmio_write = false;
        self.cpu.mmio_parts.fill((0, 0));
        self.cpu.mmio_parts_len = 0;
        self.cpu.mmio_parts_rip = 0;
        self.cpu.mmio_retry_active = false;
        self.cpu.mmio_retry_cs = 0;
        self.apply_pending_a20();
        *self.fault_entry = Default::default();
        self.fast_clear();
        if options.x87_init {
            crate::x87::fninit(&mut self.cpu);
        }
    }

    /// Obtain a handle that may request an exit from any host thread.
    pub fn exit_request(&self) -> ExitRequest {
        self.exit_request.clone()
    }

    /// Evidence for the [`Exit::Interrupted`] returned by the most recent [`Self::run`] or positive [`Self::run_steps`].
    /// `None` before the first run, during a run, or after any other exit. Cleared
    /// as soon as the next `run` or positive `run_steps` is entered, even if it unwinds. The returned value
    /// is a copy and can be retained across resumption. Deferred evidence for a
    /// still-pending request survives unrelated exits internally.
    pub fn interruption_evidence(&self) -> Option<crate::InterruptionEvidence> {
        self.interruption_evidence
    }

    /// Evidence for the [`Exit::Syscall`] returned by the most recent [`Self::run`] or positive [`Self::run_steps`].
    /// The RIP identifies the syscall instruction, while `cpu.rip` points past it.
    /// `None` before the first run or after any other exit. Cleared as soon as
    /// the next `run` or positive `run_steps` is entered, even if it unwinds; the copy can be retained.
    /// Chain-link evidence is not collected for syscalls.
    pub fn syscall_evidence(&self) -> Option<crate::ExecutionEvidence> {
        self.syscall_evidence
    }

    /// Fast-resolve cache hits served without a shared-cache lookup.
    pub fn fast_hits(&self) -> u64 {
        self.fast_hits
    }

    pub fn execution_profile(&self) -> ExecutionProfile {
        self.profile
    }

    /// The selected catalogue model, or `None` for Application and the legacy
    /// compatibility selector (decision-23 §2, Migration M1).
    pub fn cpu_model(&self) -> Option<crate::CpuModel> {
        self.cpu_model
    }

    pub fn legacy_cpu_model(&self) -> Option<LegacyCpuModel> {
        self.legacy_model
    }

    /// Drive this vCPU's legacy physical A20 line (enabled at ordinary legacy entry).
    /// An embedder with a shared board line must update each vCPU between runs.
    /// Safe after a port-I/O exit; no controller or port policy is implemented here.
    /// Application profiles ignore this line. Execution-state reset preserves it.
    /// A memory continuation, pending acknowledgment, partial split transfer or
    /// decoded retry keeps its effective line until the next architectural
    /// interruption point. Queue changes while that instruction or string
    /// element is in flight. Apply them on completion, between REP iterations,
    /// or when an architectural fault abandons the attempt, before returning or
    /// delivering the fault. With nothing in flight, apply changes immediately.
    /// An IRQ handler between REP iterations can change the line; the resumed
    /// REP uses the new line, whether or not a request preceded the pause.
    /// An accepted interrupt-delivery continuation finishes before a queued
    /// change applies. A handler-fetch failure cannot retain an abandoned retry.
    /// Calls requesting the effective value are no-ops, including while a change
    /// is pending. [`Self::pending_a20_enabled`] exposes a queued change.
    ///
    /// This architectural interruption point rule is an engine contract
    /// (decision-22). SDM Vol 2B REP, p. 4-564, preserves updated registers at
    /// iteration interruption windows. An OUT used by an embedder's A20 controller
    /// completes at a boundary before the next instruction, allowing an immediate
    /// change. A zero-budget run leaves an in-flight request pending.
    pub fn set_a20_enabled(&mut self, enabled: bool) {
        if self.twenty_bit_model() {
            return;
        }
        if self.cpu.legacy.a20_enabled == u8::from(enabled) {
            return;
        }
        if self.legacy_memory_in_flight() {
            self.legacy_execution.as_mut().unwrap().pending_a20_enabled = Some(enabled);
        } else {
            self.cpu.legacy.a20_enabled = u8::from(enabled);
        }
    }

    /// The effective legacy A20 line, excluding any queued change. Application
    /// execution does not consult this value (decision-22).
    pub fn a20_enabled(&self) -> bool {
        self.cpu.legacy.a20_enabled != 0
    }

    /// A requested A20 value waiting for the next architectural interruption
    /// point, or `None`: instruction completion, a completed REP iteration, or
    /// abandonment by an architectural fault. Execution-state reset also abandons
    /// the instruction and applies this value. Application profiles never queue
    /// a change. See [`Self::set_a20_enabled`] for the boundary contract.
    pub fn pending_a20_enabled(&self) -> Option<bool> {
        self.legacy_execution
            .as_ref()
            .and_then(|e| e.pending_a20_enabled)
    }

    fn legacy_memory_in_flight(&self) -> bool {
        self.legacy_execution.as_ref().is_some_and(|e| {
            self.cpu.mmio_retry_active
                || self.cpu.port_string_in.is_some()
                || self.cpu.mmio_parts_len != 0
                || self.cpu.pending_mmio.is_some()
                || self.cpu.pending_mmio_write
                || e.has_continuation()
        })
    }

    fn apply_pending_a20(&mut self) {
        if !self.legacy_memory_in_flight() {
            if let Some(enabled) = self
                .legacy_execution
                .as_mut()
                .and_then(|e| e.pending_a20_enabled.take())
            {
                self.cpu.legacy.a20_enabled = u8::from(enabled);
            }
        }
    }

    fn unsupported_legacy(&self, operation: UnsupportedSystemOperation) -> Exit {
        let model = self.legacy_model.expect("Machine has a model");
        let operation = crate::legacy::derive_legacy_context(model, &self.cpu)
            .err()
            .map_or(operation, UnsupportedSystemOperation::InvalidLegacyContext);
        Exit::UnsupportedSystemContext {
            addr: self.cpu.rip,
            operation,
        }
    }

    /// Queue one manual maskable IRQ vector; a subsequent injection overwrites it.
    /// Queueing neither acknowledges a controller nor constructs a delivery frame.
    /// Application Real16 retains its existing delivery/readiness contract.
    /// Application Long64/Compat32 positive runs refuse with the vector retained;
    /// Legacy I286/I386 real execution delivers through its current IDTR.
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

    /// Whether a maskable IRQ could be accepted at this boundary.
    /// Independent of whether a vector is queued; does not acknowledge or deliver.
    /// Supported for Application Real16 and Machine I286/I386 execution.
    /// Pending IN completion, the entire MMIO retry (including answered but
    /// unfinished transfers), and Machine task-switch continuations prevent acceptance.
    ///
    /// IF and the STI/SS shadows follow SDM Vol 3A §7.8.1 and §7.8.3;
    /// device-completion blocking follows decision-16 §§1, 4.
    /// While an NMI is pending and ready ([`Self::has_pending_nmi`] and
    /// [`Self::nmi_ready`]) this is false: NMI has priority and INT is not
    /// accepted at that boundary (NEC UM p. 2-2; 8086 Family UM p. 2-24;
    /// 286 PRM §9.2 p. 9-3).
    pub fn irq_ready(&self) -> bool {
        (self.profile == ExecutionProfile::Application(CpuMode::Real16)
            || (self.profile == ExecutionProfile::Machine
                && self.legacy_context_for_run().is_ok()
                && !self.legacy_execution.as_ref().unwrap().is_shutdown()))
            && !self.cpu.mmio_retry_active
            && self.irq_delivery_ready()
            && !(self.has_pending_nmi() && self.nmi_ready())
    }

    /// Latch one rising edge on the NMI input of an
    /// [`ExecutionProfile::Machine`] vCPU (spec.md §17.10). Edges coalesce
    /// until one is accepted. Takes no step, has no guest effect and returns
    /// no exit; the next positive [`Self::run`] or [`Self::run_steps`] delivers
    /// it at the first boundary where [`Self::nmi_ready`] holds, ahead of any
    /// queued IRQ and independently of IF. Application vCPUs refuse with no
    /// change. [`Self::nmi_line`] raises from another thread or a signal handler.
    pub fn raise_nmi(&mut self) -> Result<(), NmiUnsupported> {
        let line = self.nmi_line().ok_or(NmiUnsupported)?;
        line.raise();
        Ok(())
    }

    /// A thread-safe handle on this vCPU's NMI input, or `None` outside
    /// [`ExecutionProfile::Machine`] (spec.md §17.10).
    pub fn nmi_line(&self) -> Option<NmiLine> {
        (self.profile == ExecutionProfile::Machine).then(|| NmiLine {
            word: Arc::clone(&self.exit_request.word),
        })
    }

    /// The model's view of the NMI input (spec.md §17.10): an edge is latched,
    /// or remembered while an I80286 services an earlier NMI, and not yet
    /// accepted. On an I80386 inside its NMI handler it is false, because the
    /// processor ignores edges until the next IRET. Always false outside
    /// [`ExecutionProfile::Machine`].
    pub fn has_pending_nmi(&self) -> bool {
        if self.profile != ExecutionProfile::Machine {
            return false;
        }
        let latched = self.exit_request.word.load(Ordering::Acquire) & NMI_EDGE != 0;
        if self.nmi_state & legacy::NMI_BLOCKED == 0 {
            return latched;
        }
        match self.nmi_nesting() {
            crate::NmiNesting::BlockedIgnore => false,
            _ => latched || self.nmi_state & legacy::NMI_REMEMBERED != 0,
        }
    }

    /// Whether a latched NMI would be accepted at this boundary (spec.md
    /// §17.10). Independent of IF and the STI/EI shadow, and of whether an edge
    /// is latched. False while an I80286/I80386 services an earlier NMI, after
    /// a segment-register load or store that holds NMI (the model table of
    /// spec.md §17.10), while an IN, INS or MMIO completion (including an
    /// answered but unfinished retry) or a task-switch continuation is
    /// outstanding, outside [`ExecutionProfile::Machine`] or a valid Machine
    /// context, and at an NEC V20/V30 8080-mode boundary with BRK = 1.
    /// Shutdown does not prevent it: NMI ends an I80286/I80386 shutdown.
    pub fn nmi_ready(&self) -> bool {
        self.profile == ExecutionProfile::Machine
            && self.legacy_context_for_run().is_ok()
            && self.nmi_gate(
                self.legacy_execution
                    .as_ref()
                    .unwrap()
                    .task_switch_pending(),
            )
            && !(self.twenty_bit_model()
                && self.cpu.legacy.reserved & crate::legacy::NEC_EMULATION != 0
                && self.cpu.rflags_system & 0x100 != 0)
    }

    // The dispatcher handles active MMIO retries before reaching its IRQ gate;
    // irq_ready checks that latch explicitly because it can be called during a retry.
    #[inline]
    fn irq_delivery_ready(&self) -> bool {
        self.legacy_execution
            .as_ref()
            .is_none_or(|e| !e.task_switch_pending())
            && self.cpu.flags.if_
            && self.cpu.pending_mmio.is_none()
            && !self.cpu.pending_mmio_write
            && self.cpu.pending_port_in.is_none()
            && self.cpu.port_string_in.is_none()
            && !self.sti_shadow
            && self.ss_shadow & SS_HOLD == 0
    }

    /// The retired-instruction counter (§17.6, sub-seam c): guest instructions that have
    /// **retired** (completed) on the interpreter path since this vcpu was created. This
    /// is a deterministic, wall-clock-free virtual-time base for the embedder's scheduler.
    /// In real mode (interpreter-only) it counts every executed instruction; compiled
    /// Long64/Compat32 blocks do not tick it (see the field docs). Monotone, never reset.
    pub fn retired_instructions(&self) -> u64 {
        self.retired
    }

    /// Start recording the exceptions and interrupts this
    /// [`ExecutionProfile::Machine`] vCPU delivers in the guest, in a ring that
    /// keeps the last `capacity` records (1 to [`crate::DeliveryLog::MAX_CAPACITY`]),
    /// read between runs through [`Self::delivery_log`] (spec.md §17.10,
    /// Delivery log). The log is off by default, and while it is off nothing
    /// in execution, retirement or generated code depends on it. A record is
    /// written when a delivery enters its handler, so the same run gives the
    /// same records however it is sliced into runs. Enabling it again starts an
    /// empty ring. An Application vCPU, the Real16 alias among them, refuses with
    /// [`crate::DeliveryLogError::NotMachine`], an out-of-range capacity with
    /// [`crate::DeliveryLogError::Capacity`], both with no change.
    ///
    /// Under [`crate::ExceptionDelivery::Outward`] an exception the run returns
    /// as an exit is not delivered and not recorded; IRQs, NMIs and software
    /// interrupts are still delivered in the guest and recorded.
    ///
    /// The log is outside [`Self::cpu`] and outside any vCPU checkpoint an
    /// embedder takes by saving and restoring `CpuState`: a restore neither
    /// rewinds nor clears it, so afterwards the log still holds the records of
    /// the abandoned timeline. An embedder that checkpoints takes the records
    /// at the save, and takes or [`crate::DeliveryLog::clear`]s them at the
    /// restore.
    ///
    /// ```
    /// use x86jit_core::{CpuModel, DeliveryKind, Prot, RegionKind, StepOptions, Vm, VmConfig};
    /// let mut vm = Vm::new(VmConfig::flat(0x10_0000));
    /// vm.set_cpu_model(CpuModel::I8086);
    /// vm.map(0, 0x10_0000, Prot::RWX, RegionKind::Ram).unwrap();
    /// vm.write_bytes(0x21 * 4, &[0x00, 0x05, 0x00, 0x00]).unwrap(); // INT 21h -> 0000:0500
    /// vm.write_bytes(0x500, &[0xf4]).unwrap(); // HLT
    /// vm.write_bytes(0x1000, &[0xcd, 0x21]).unwrap(); // INT 21h at 0000:1000
    /// let mut cpu = vm.new_vcpu();
    /// cpu.cpu.rip = 0x1000;
    /// cpu.cpu.gpr[4] = 0x8000;
    /// cpu.enable_delivery_log(64).unwrap();
    /// cpu.run_steps(&vm, 2, StepOptions::default()).unwrap();
    /// let batch = cpu.delivery_log_mut().unwrap().take();
    /// assert_eq!((batch.dropped, batch.total), (0, 1));
    /// let record = &batch.records[0];
    /// assert_eq!((record.vector, record.kind), (0x21, DeliveryKind::Software));
    /// assert_eq!((record.saved_cs, record.saved_ip, record.saved_linear), (0, 0x1002, 0x1002));
    /// assert_eq!(record.retired, 1);
    /// assert_eq!(record.regs, None); // see `enable_delivery_log_with`
    /// ```
    pub fn enable_delivery_log(&mut self, capacity: usize) -> Result<(), crate::DeliveryLogError> {
        self.enable_delivery_log_with(capacity, crate::DeliveryLogOptions::default())
    }

    /// [`Self::enable_delivery_log`] with `options`, which choose what each
    /// record holds besides the default fields: with
    /// [`crate::DeliveryLogOptions::registers`] set, every record carries the
    /// interrupted code's registers in [`crate::DeliveryRecord::regs`]. With
    /// the default options this is [`Self::enable_delivery_log`]; the options
    /// change no execution, retirement or generated code. The options are
    /// fixed for the log's life: this call is the only way to change them,
    /// and it starts an empty ring. A delivery that began before it, an
    /// 80286/80386 task switch handed off to the embedder, records `regs:
    /// None`.
    ///
    /// ```
    /// use x86jit_core::{CpuModel, DeliveryLogOptions, Prot, RegionKind, StepOptions, Vm, VmConfig};
    /// let mut vm = Vm::new(VmConfig::flat(0x10_0000));
    /// vm.set_cpu_model(CpuModel::I8086);
    /// vm.map(0, 0x10_0000, Prot::RWX, RegionKind::Ram).unwrap();
    /// vm.write_bytes(0x21 * 4, &[0x00, 0x05, 0x00, 0x00]).unwrap(); // INT 21h -> 0000:0500
    /// vm.write_bytes(0x500, &[0xf4]).unwrap(); // HLT
    /// // MOV AH, 09h; INT 21h at 0000:1000: INT n with a function number in AH.
    /// vm.write_bytes(0x1000, &[0xb4, 0x09, 0xcd, 0x21]).unwrap();
    /// let mut cpu = vm.new_vcpu();
    /// cpu.cpu.rip = 0x1000;
    /// cpu.cpu.gpr[4] = 0x8000;
    /// cpu.cpu.ds = 0x1234;
    /// let options = DeliveryLogOptions::default().registers(true);
    /// cpu.enable_delivery_log_with(64, options).unwrap();
    /// cpu.run_steps(&vm, 3, StepOptions::default()).unwrap();
    /// let log = cpu.delivery_log().unwrap();
    /// assert!(log.options().records_registers());
    /// let regs = log.iter().next().unwrap().regs.as_deref().copied().unwrap();
    /// // The INT's inputs: AH as MOV set it, SP before the frame.
    /// assert_eq!((regs.eax >> 8, regs.esp, regs.ds), (0x09, 0x8000, 0x1234));
    /// assert_eq!((regs.fs, regs.gs), (None, None)); // FS and GS exist on the 80386 only.
    /// ```
    pub fn enable_delivery_log_with(
        &mut self,
        capacity: usize,
        options: crate::DeliveryLogOptions,
    ) -> Result<(), crate::DeliveryLogError> {
        if self.profile != ExecutionProfile::Machine {
            return Err(crate::DeliveryLogError::NotMachine);
        }
        self.delivery_log = Some(Box::new(crate::DeliveryLog::new(capacity, options)?));
        self.legacy_execution
            .as_mut()
            .expect("a Machine vCPU has its execution state")
            .set_record_regs(options.records_registers());
        Ok(())
    }

    /// Stop recording, and return the log with the records it still holds.
    pub fn disable_delivery_log(&mut self) -> Option<crate::DeliveryLog> {
        if let Some(execution) = self.legacy_execution.as_mut() {
            execution.set_record_regs(false);
        }
        self.delivery_log.take().map(|log| *log)
    }

    /// The delivery log, while it is enabled (spec.md §17.10, Delivery log).
    pub fn delivery_log(&self) -> Option<&crate::DeliveryLog> {
        self.delivery_log.as_deref()
    }

    /// The delivery log, to [`crate::DeliveryLog::take`],
    /// [`crate::DeliveryLog::drain_into`] or [`crate::DeliveryLog::clear`] its
    /// records between runs.
    pub fn delivery_log_mut(&mut self) -> Option<&mut crate::DeliveryLog> {
        self.delivery_log.as_deref_mut()
    }

    // Fold alignment and page bits into the set index: short aligned thunks
    // must not all compete for the same few sets. Full RIP tags decide hits.
    #[inline]
    fn fast_index(rip: u64) -> usize {
        let folded = crate::cache::cache_hash::dispatch_hash(rip);
        (folded as usize & ((1 << FAST_BITS) - 1)) * 2
    }

    #[inline]
    fn fast_get(&mut self, rip: u64) -> Option<CompiledPtr> {
        let index = Self::fast_index(rip);
        let first = self.fast[index];
        if !first.entry.0.is_null() && first.rip == rip {
            return Some(first.entry);
        }
        let second = self.fast[index + 1];
        if !second.entry.0.is_null() && second.rip == rip {
            self.fast.swap(index, index + 1);
            return Some(second.entry);
        }
        None
    }

    // A set is cold only while it has an empty way; a full-set miss is a collision.
    fn record_shared_resolve(&self, reason: SharedReason, rip: u64) {
        let index = Self::fast_index(rip);
        let private =
            if self.fast[index].entry.0.is_null() || self.fast[index + 1].entry.0.is_null() {
                self.fast_empty_reason
            } else {
                PrivateMiss::Collision
            };
        crate::time_stats::shared_resolve(reason, private);
    }

    // Replace the least recently used way once full.
    // Way zero is MRU; a way-one hit promotes it. Epoch/mode flushes clear both.
    #[inline]
    fn fast_put(&mut self, rip: u64, entry: CompiledPtr) {
        let index = Self::fast_index(rip);
        if self.fast[index].rip == rip && !self.fast[index].entry.0.is_null() {
            self.fast[index].entry = entry;
            return;
        }
        if self.fast[index + 1].rip == rip && !self.fast[index + 1].entry.0.is_null() {
            self.fast.swap(index, index + 1);
        } else {
            self.fast[index + 1] = self.fast[index];
        }
        self.fast[index] = FastEntry { rip, entry };
    }

    /// Drop every fast-resolve entry and reset the IBTC refill counts (on an
    /// invalidation-epoch change). The backend already zeroed the slots
    /// (`invalidate_links`), so a previously-capped site gets a fresh chance to
    /// re-cache against the rewritten code.
    fn fast_clear(&mut self) {
        self.fast_empty_reason = PrivateMiss::Cold;
        for e in self.fast.iter_mut() {
            e.entry = CompiledPtr::new(std::ptr::null());
        }
        self.ibtc_refills.clear();
        // Return predictions carry continuation-slot pointers. They must be
        // discarded before resolving again in a later execution epoch.
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
                    self.cpu.port_string_in = None;
                    self.cpu.pending_port_in = None;
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

    /// Coprocessor policy for this vCPU. Currently consumed only in Real16.
    pub fn fpu_model(&self) -> crate::FpuModel {
        self.cpu.fpu_model
    }

    /// Select between runs. The x87 register image and cached IR are retained.
    /// Long64/Compat32 continue to use the existing x87 engine.
    pub fn set_fpu_model(&mut self, model: crate::FpuModel) {
        self.cpu.fpu_model = model;
    }

    /// Engine emulation input for the Real16 alias and models without system
    /// registers (decision-23 §4). MSW/CR0 models use their architectural EM bit.
    pub fn fpu_emulation(&self) -> bool {
        self.cpu.fpu_emulation
    }

    /// Set the emulation input between runs, independently of FPU presence.
    /// On I8088/I8086 this requests a typed ESC handoff; these processors have
    /// no interrupt 7. The Real16 alias retains its existing MSW.EM policy.
    pub fn set_fpu_emulation(&mut self, enabled: bool) {
        self.cpu.fpu_emulation = enabled;
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
    /// first load consumes this value instead of re-trapping; the retry ledger keeps
    /// it across the instruction's own device stops (spec.md §5.2). Stored on `CpuState`,
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
    /// A read-modify-write to a `Trap` region (`add [mmio], reg`) stops twice: one
    /// `MmioRead`, then one `MmioWrite` of the updated value. The retry after this
    /// acknowledgement reuses the value the read was answered with; the device is
    /// not read again, also after a budget or host stop taken before the retry. A
    /// different instruction written at the same RIP in between, or a fault of the
    /// instruction, discards that value, and the re-execution reads the device
    /// again. On Real16 and the
    /// Machine models, XCHG with memory and a LOCK-prefixed form do the same for an
    /// operand inside one Trap region; Long64 and Compat32 return
    /// `Exit::UnknownInstruction` for those (spec.md §5.2).
    pub fn complete_mmio_write(&mut self) {
        if !self.cpu.mmio_retry_active {
            self.cpu.begin_mmio_retry(self.cpu.rip);
        }
        self.cpu.pending_mmio_write = true;
    }

    /// Supply the value for an outstanding IN or INS (§5.2).
    /// IN writes AL/AX/EAX immediately, using the central sub-register policy.
    /// INS retains the value until the next execution call attempts its faultable
    /// ES destination store, then advances DI/EDI/RDI and the REP count. No new
    /// step is charged. A failed store retains the answer, so retry never reads
    /// the port again. IRQ acceptance stays blocked until the store completes.
    /// Calling this without an outstanding input request is a no-op.
    pub fn complete_port_in(&mut self, value: u64) {
        if let Some(size) = self.cpu.pending_port_in.take() {
            if let Some(p) = &mut self.cpu.port_string_in {
                p.value = Some(value & (u64::MAX >> (64 - u32::from(size) * 8)));
            } else {
                self.cpu
                    .write_gpr(Reg::Rax.gpr_index().unwrap(), value, size);
                self.sti_shadow = false;
                self.ss_shadow = 0;
            }
        }
    }

    fn finish_port_string_in<const ROUTED: bool>(&mut self, vm: &Vm) -> Result<(), Exit> {
        if self.cpu.port_string_in.is_none() {
            return Ok(());
        }
        match crate::interp::finish_port_string_in::<ROUTED>(
            &mut self.cpu,
            &vm.mem,
            self.mode,
            vm.consistency,
        ) {
            Ok(()) => {
                self.sti_shadow = false;
                self.ss_shadow = 0;
                Ok(())
            }
            Err(exit) => {
                let exit = if self.mode == CpuMode::Real16 {
                    crate::fault::qualify_real16(exit, &self.cpu)
                } else {
                    exit
                };
                self.cpu.finish_mmio_retry(Some(&exit));
                Err(self.fault_entry.finish_transfer_fault(exit, &vm.mem))
            }
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
    /// (resume deferred MMIO through [`Vcpu::run`]). An answered INS first
    /// attempts its pending store and returns without executing another element.
    /// Legacy real execution performs one run_steps unit, including one REP iteration.
    pub fn step_instruction(&mut self, vm: &Vm) -> StepResult {
        if self.profile == ExecutionProfile::Machine {
            return match self.run_steps(vm, 1, StepOptions::default()) {
                Ok(run) => match run.stop {
                    StepStop::Exit(Exit::BudgetExhausted) | StepStop::IrqReady => {
                        StepResult::Continue
                    }
                    StepStop::Exit(exit) => StepResult::Exit(exit),
                },
                Err(_) => StepResult::Exit(
                    self.unsupported_legacy(UnsupportedSystemOperation::LegacyStepping),
                ),
            };
        }
        if self.mode == CpuMode::Real16 {
            crate::legacy::REAL16_ALIAS.derive_caches(&mut self.cpu);
        }
        if self.cpu.port_string_in.is_some() {
            let result = if vm.mem.host_fault_recovery() {
                self.finish_port_string_in::<true>(vm)
            } else {
                self.finish_port_string_in::<false>(vm)
            };
            return match result {
                Ok(()) => StepResult::Continue,
                Err(exit) => StepResult::Exit(exit),
            };
        }
        if vm.mem.fetch_policy.is_some() {
            vm.handle_smc(&self.exit_request.word);
        }
        let _fp_mode = crate::jit_abi::fp::RunMode::new();
        let mut info = crate::interp::RetireInfo {
            consistency: vm.consistency,
            sti_shadow: self.sti_shadow,
            ss_shadow: self.ss_shadow & SS_HOLD != 0,
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
        self.ss_shadow = u8::from(info.ss_shadow);
        let result = match r {
            // §17.6 (sub-seam b): re-deliver a real-mode exception through the IVT,
            // mirroring the interpreted-block arm of `run`.
            StepResult::Exit(Exit::Exception { addr, vector, .. })
                if self.mode == CpuMode::Real16 =>
            {
                let result = crate::interp::deliver_interrupt(
                    &mut self.cpu,
                    &vm.mem,
                    addr,
                    vector,
                    addr,
                    vm.consistency,
                )
                .into_step();
                if matches!(result, StepResult::Continue) {
                    self.sti_shadow = false;
                    self.ss_shadow = 0;
                }
                result
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

    /// Execute at most `max_steps` units for [`ExecutionProfile::Application`]
    /// with [`CpuMode::Real16`], including when the VM has a compiling backend
    /// (decision-16 §§1–4). Application Long64/Compat32 return
    /// [`StepRunError::UnsupportedMode`], including with a zero budget.
    /// [`ExecutionProfile::Machine`] also supports same-CPL I286/I386 real and
    /// protected execution, using derived code/stack sizes and a private
    /// interpreter cache. Unavailable event delivery returns typed unsupported exits.
    /// No legacy context enters a compiling backend.
    ///
    /// A unit is one completed instruction or one completed REP iteration. Empty
    /// REP counts once. IN/OUT count at handoff; supplying an IN value adds no unit.
    /// A positive call with pending IN is refused. For Application Real16,
    /// zero returns budget exhaustion without fetching, delivering, consuming a
    /// completion, changing shadows, clearing retries, or consuming host requests.
    ///
    /// Instruction exits win over exhaustion. At other boundaries: budget, host
    /// request (after pending completion), a latched Machine NMI that
    /// [`Self::nmi_ready`] accepts, queued eligible IRQ, optional readiness
    /// stop, then execution.
    /// Delivery costs no unit. An incomplete instruction/iteration contributes zero;
    /// its already-permitted partial effects remain. The next unit never begins.
    /// `steps = delta(retired_instructions) + completed_rep_iterations
    /// - completed_nonempty_rep_instructions`, local to this call.
    /// A Machine run, here and in `run`, bounds delivery that retires nothing:
    /// after 1024 consecutive completed deliveries with no instruction retired
    /// between them, counted across runs, as when a handler's first
    /// instruction faults again, it returns [`Exit::BudgetExhausted`] at that
    /// boundary, resumable, with `steps` below the budget. The next run counts
    /// afresh. A host exit request remains the earlier escape.
    /// As with `run`, this vCPU must belong to `vm`.
    pub fn run_steps(
        &mut self,
        vm: &Vm,
        max_steps: u64,
        options: StepOptions,
    ) -> Result<StepRun, StepRunError> {
        if self.profile == ExecutionProfile::Machine {
            if self
                .legacy_model
                .is_some_and(|model| model.generation == crate::CpuGeneration::I8086)
            {
                // xj-50w.15: a REP owner/restart pair was pending when the last
                // run returned; the host may have edited its string or inputs.
                // A zero-budget call is no attempt boundary (`run_engine`).
                if max_steps != 0 && self.rep_restart20_pending() {
                    self.forget_edited_rep_restart20(vm);
                }
                self.run_direct20::<true>(vm, Some(max_steps), options)
            } else {
                self.run_engine::<MACHINE_DISPATCH, true>(vm, Some(max_steps), options)
            }
        } else if self.mode == CpuMode::Real16 {
            self.run_engine::<ALIAS_DISPATCH, true>(vm, Some(max_steps), options)
        } else {
            Err(StepRunError::UnsupportedMode)
        }
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
        if self.profile != ExecutionProfile::Machine
            && self.mode != CpuMode::Real16
            && budget != Some(0)
            && self.pending_irq.is_some()
        {
            return Exit::UnsupportedSystemContext {
                addr: self.cpu.rip,
                operation: UnsupportedSystemOperation::ApplicationIrqDelivery,
            };
        }
        if self.profile != ExecutionProfile::Machine && !vm.cache.code_lifetime().is_open() {
            if let Some(next) = vm.backend.code_lifetime() {
                if vm
                    .cache
                    .reconcile_code_lifetime(next, |page| vm.mem.clear_code_page(page))
                {
                    vm.backend.invalidate_links();
                }
            }
        }
        let run = if self.profile == ExecutionProfile::Machine {
            if self
                .legacy_model
                .is_some_and(|model| model.generation == crate::CpuGeneration::I8086)
            {
                if budget != Some(0) && self.rep_restart20_pending() {
                    self.forget_edited_rep_restart20(vm);
                }
                self.run_direct20::<false>(vm, budget, StepOptions::default())
            } else {
                self.run_engine::<MACHINE_DISPATCH, false>(vm, budget, StepOptions::default())
            }
        } else if self.mode == CpuMode::Real16 {
            self.run_engine::<ALIAS_DISPATCH, false>(vm, budget, StepOptions::default())
        } else {
            self.run_engine::<APPLICATION_DISPATCH, false>(vm, budget, StepOptions::default())
        };
        match run {
            Ok(StepRun {
                stop: StepStop::Exit(exit),
                ..
            }) => exit,
            Ok(_) => unreachable!(),
            Err(_) => self.unsupported_legacy(UnsupportedSystemOperation::LegacyExecution),
        }
    }

    // I8088/I8086 run in a permanent Direct context. The deprecated selector's
    // I8086 generation keeps its C1 run-entry refusal until removal
    // (decision-23 Migration); only set_cpu_model selects these models.
    // The run entries keep their generation test; the Direct20 entry refuses.
    #[inline(always)]
    fn twenty_bit_model(&self) -> bool {
        matches!(
            self.cpu_model,
            Some(
                crate::CpuModel::I8088
                    | crate::CpuModel::I8086
                    | crate::CpuModel::NecV20
                    | crate::CpuModel::NecV30
            )
        )
    }

    // Isolate the permanent-context specialization from the existing entries.
    #[inline(never)]
    fn run_direct20<const STEPPED: bool>(
        &mut self,
        vm: &Vm,
        budget: Option<u64>,
        options: StepOptions,
    ) -> Result<StepRun, StepRunError> {
        self.run_engine::<DIRECT20_DISPATCH, STEPPED>(vm, budget, options)
    }

    #[inline(always)]
    fn run_engine<const KIND: u8, const STEPPED: bool>(
        &mut self,
        vm: &Vm,
        budget: Option<u64>,
        options: StepOptions,
    ) -> Result<StepRun, StepRunError> {
        // Decision-16 §3: every execution exit passes the attempt's ownership
        // boundary. Only explicit continuations can return a REP flag origin.
        // The twenty-bit walk never sets a REP flag origin (only
        // `execute_protected_block` does); its REP restart pair is recorded
        // at the end of `run_engine_attempt` (xj-50w.15).
        let result = self.run_engine_attempt::<KIND, STEPPED>(vm, budget, options);
        if KIND >= MACHINE_DISPATCH && KIND != DIRECT20_DISPATCH && budget != Some(0) {
            self.finish_comparison_attempt(&result);
        }
        result
    }

    #[inline(always)]
    fn run_engine_attempt<const KIND: u8, const STEPPED: bool>(
        &mut self,
        vm: &Vm,
        budget: Option<u64>,
        options: StepOptions,
    ) -> Result<StepRun, StepRunError> {
        if KIND == ALIAS_DISPATCH && STEPPED && budget == Some(0) {
            let time_run = crate::time_stats::RunGuard::enter(
                self.time_record.clone(),
                vm.time_enabled,
                self.time_outside.take(),
            );
            drop(DispatchGuard::new(self, vm));
            self.time_outside = time_run.finish(Some(&Exit::BudgetExhausted));
            return Ok(StepRun {
                steps: 0,
                stop: StepStop::Exit(Exit::BudgetExhausted),
            });
        }
        let context = if KIND >= MACHINE_DISPATCH {
            let context = if KIND == DIRECT20_DISPATCH {
                // The deprecated selector's I8086 generation keeps the C1
                // run-entry refusal (decision-23 Migration).
                if !self.twenty_bit_model() {
                    let Err(exit) = self.prepare_machine_entry_cold(budget) else {
                        unreachable!("the I8086 generation has no system entry")
                    };
                    return Ok(StepRun {
                        steps: 0,
                        stop: StepStop::Exit(exit),
                    });
                }
                if budget == Some(0) {
                    return Ok(StepRun {
                        steps: 0,
                        stop: StepStop::Exit(Exit::BudgetExhausted),
                    });
                }
                crate::lift::i8086::CONTEXT
            } else if let Some(context) = self.warm_machine_context(budget) {
                context
            } else {
                match self.prepare_machine_entry_cold(budget) {
                    Ok(context) => context,
                    Err(exit) => {
                        return Ok(StepRun {
                            steps: 0,
                            stop: StepStop::Exit(exit),
                        });
                    }
                }
            };
            Some((context, budget))
        } else {
            None
        };
        if STEPPED && self.cpu.pending_port_in.is_some() {
            return Err(StepRunError::PendingPortIn);
        }
        if KIND == ALIAS_DISPATCH && budget != Some(0) {
            crate::legacy::REAL16_ALIAS.derive_caches(&mut self.cpu);
        }
        let time_run = crate::time_stats::RunGuard::enter(
            self.time_record.clone(),
            vm.time_enabled,
            self.time_outside.take(),
        );
        let dispatch = DispatchGuard::new(self, vm);
        let cpu = &mut *dispatch.cpu;
        cpu.interruption_evidence = None;
        cpu.syscall_evidence = None;
        cpu.fault_entry.prepare_run();
        let mut fp_mode = crate::jit_abi::fp::RunMode::new();
        struct FaultRunGuard(*mut crate::jit_abi::fault::EntryRecord);
        impl Drop for FaultRunGuard {
            fn drop(&mut self) {
                // SAFETY: the boxed entry remains fixed for the exclusive vCPU borrow.
                unsafe {
                    (*self.0).deactivate();
                }
            }
        }
        let guard = FaultRunGuard(&mut *cpu.fault_entry);
        let mut result = if KIND == APPLICATION_DISPATCH {
            let exit = if vm.mem.host_fault_recovery() {
                cpu.run_inner::<true>(vm, budget, &mut fp_mode)
            } else {
                cpu.run_inner::<false>(vm, budget, &mut fp_mode)
            };
            StepRun {
                steps: 0,
                stop: StepStop::Exit(exit),
            }
        } else if KIND == ALIAS_DISPATCH {
            macro_rules! alias {
                ($routed:literal, $bounded:literal, $stop:literal) => {
                    cpu.run_dispatch::<$routed, true, $bounded, ALIAS_DISPATCH, STEPPED, $stop, 0, false>(
                        vm,
                        budget.unwrap_or(u64::MAX),
                        &mut fp_mode,
                        SharedReason::Entry,
                        options,
                        None,
                    )
                    .stepped()
                };
                ($routed:literal) => {
                    match (budget.is_some(), options.stop_when_irq_ready) {
                        (true, true) => alias!($routed, true, true),
                        (true, false) => alias!($routed, true, false),
                        (false, true) => alias!($routed, false, true),
                        (false, false) => alias!($routed, false, false),
                    }
                };
            }
            if vm.mem.host_fault_recovery() {
                alias!(true)
            } else {
                alias!(false)
            }
        } else {
            macro_rules! enter {
                ($routed:literal, $order:literal, $single:literal) => {
                    cpu.run_dispatch::<$routed, true, false, KIND, false, false, $order, $single>(
                        vm,
                        if $single {
                            1
                        } else {
                            budget.unwrap_or(u64::MAX)
                        },
                        &mut fp_mode,
                        SharedReason::Entry,
                        options,
                        context,
                    )
                    .stepped()
                };
                ($routed:literal, $order:literal) => {{
                    if budget == Some(1) {
                        enter!($routed, $order, true)
                    } else {
                        enter!($routed, $order, false)
                    }
                }};
            }
            match (vm.mem.host_fault_recovery(), vm.consistency) {
                (true, MemConsistency::Fast) => enter!(true, 0),
                (false, MemConsistency::Fast) => enter!(false, 0),
                (true, MemConsistency::AcqRel) => enter!(true, 1),
                (false, MemConsistency::AcqRel) => enter!(false, 1),
                (true, MemConsistency::FullTso) => enter!(true, 2),
                (false, MemConsistency::FullTso) => enter!(false, 2),
            }
        };
        drop(guard);
        if KIND == ALIAS_DISPATCH {
            if let StepStop::Exit(exit) = result.stop {
                result.stop = StepStop::Exit(crate::fault::qualify_real16(exit, &cpu.cpu));
            }
        }
        if let StepStop::Exit(exit @ (Exit::UnmappedMemory { .. } | Exit::Exception { .. })) =
            result.stop
        {
            result.stop = StepStop::Exit(cpu.fault_entry.finish_transfer_fault(exit, &vm.mem));
        }
        if KIND >= MACHINE_DISPATCH {
            cpu.finish_machine_run(&mut result);
        } else if let StepStop::Exit(exit) = &result.stop {
            cpu.cpu.finish_mmio_retry(Some(exit));
        }
        if let StepStop::Exit(Exit::Interrupted) = &result.stop {
            if cpu.interruption_evidence.is_none() {
                cpu.interruption_evidence =
                    Some(
                        crate::interruption::consume(&cpu.exit_request.word, &cpu.engine_sections)
                            .at(crate::FaultPath::Unknown, false, cpu.cpu.rip),
                    );
            }
        }
        // xj-50w.15: a twenty-bit run returns with a REP owner/restart pair
        // pending; record its string while the VM is still in use. A
        // zero-budget call returned before the dispatch (no attempt boundary).
        if KIND == DIRECT20_DISPATCH && cpu.rep_restart20_pending() {
            cpu.record_rep_restart20(vm);
        }
        let outside = dispatch.finish(
            &time_run,
            match &result.stop {
                StepStop::Exit(exit) => Some(exit),
                StepStop::IrqReady => None,
            },
        );
        self.time_outside = outside;
        Ok(result)
    }

    fn finish_dispatch_counts(
        &mut self,
        vm: &Vm,
        fast_before: u64,
    ) -> crate::time_stats::DispatchCounts {
        let mut counts = std::mem::take(&mut self.time_dispatch);
        vm.cache
            .add_dispatch_counts(counts.lookups.saturating_sub(counts.misses), counts.misses);
        counts.lookups = counts
            .lookups
            .saturating_add(self.fast_hits.saturating_sub(fast_before));
        counts
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

    #[inline(never)]
    fn run_application_dispatch<const ROUTED: bool, const DEFAULT: bool, const BOUNDED: bool>(
        &mut self,
        vm: &Vm,
        budget: u64,
        fp_mode: &mut crate::jit_abi::fp::RunMode,
        origin: SharedReason,
    ) -> Result<Exit, u64> {
        self.run_dispatch::<ROUTED, DEFAULT, BOUNDED, APPLICATION_DISPATCH, false, false, 0, false>(
            vm,
            budget,
            fp_mode,
            origin,
            StepOptions::default(),
            None,
        )
        .application()
    }

    fn run_inner<const ROUTED: bool>(
        &mut self,
        vm: &Vm,
        budget: Option<u64>,
        fp_mode: &mut crate::jit_abi::fp::RunMode,
    ) -> Exit {
        let mut remaining = budget;
        let mut origin = SharedReason::Entry;
        loop {
            fp_mode.default();
            let mode = self.cpu.mxcsr & crate::jit_abi::fp::MODE_MASK;
            if self.fast_mxcsr_mode != mode {
                self.fast_clear();
                self.fast_mxcsr_mode = mode;
                self.fast_empty_reason = PrivateMiss::Mode;
            }
            let limit = remaining.unwrap_or(u64::MAX);
            let result = match (mode == 0, remaining.is_some()) {
                (true, true) => {
                    self.run_application_dispatch::<ROUTED, true, true>(vm, limit, fp_mode, origin)
                }
                (true, false) => {
                    self.run_application_dispatch::<ROUTED, true, false>(vm, limit, fp_mode, origin)
                }
                (false, true) => {
                    self.run_application_dispatch::<ROUTED, false, true>(vm, limit, fp_mode, origin)
                }
                (false, false) => self
                    .run_application_dispatch::<ROUTED, false, false>(vm, limit, fp_mode, origin),
            };
            match result {
                Ok(exit) => return exit,
                Err(consumed) => {
                    remaining = remaining.map(|b| b - consumed);
                    if vm.time_enabled {
                        origin = SharedReason::ModeRestart;
                    }
                }
            }
        }
    }
}

/// Publish every completed background compile into the cache: the dispatcher
/// publishes, the backend never touches the cache. Source-span validation includes
/// writes to uncached successors. Only the matching completion releases a slot,
/// so an old completion cannot clear a newer request's ownership.
/// `tier_up_finished` short-circuits when idle.
fn drain_tier_up(vm: &Vm, exit_word: &AtomicU64, sections: &AtomicU64) {
    let _section = EngineSection::enter(exit_word, sections, CriticalSectionKind::TierUpWait);
    for fin in vm.backend.tier_up_finished() {
        vm.handle_smc(exit_word);
        vm.cache.record_tier_bg_completed();
        // A Vm runs in a single mode (§17 scope fence), so the finished compile's
        // block key is its echoed pc under the Vm's mode (§17.4).
        let key = BlockKey::with_mxcsr(fin.pc, vm.mode, fin.mxcsr_mode);
        let multi_span = fin.spans.len() > 1;
        let source_spans = fin.spans;
        let _publish = EngineSection::enter(exit_word, sections, CriticalSectionKind::Publish);
        let published = vm.cache.publish_tier_up(
            key,
            fin.block.clone(),
            fin.compile_failure,
            fin.epoch,
            |sp| {
                for (start, len) in sp {
                    vm.mem.mark_code(*start, *len);
                }
            },
        );
        if !published {
            vm.cache.release_source_tags(&vm.mem, &source_spans);
            discard_materialized(vm, &fin.block);
        }
        if published {
            // A multi-span unit is a superblock region — count it like the
            // eager region path does, so `cache.regions()` reflects background regions.
            if multi_span {
                vm.cache.record_region();
            }
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
    stats: &mut crate::time_stats::DispatchCounts,
    exit_word: &AtomicU64,
    sections: &AtomicU64,
    at: FetchAddr,
    mode: CpuMode,
    mxcsr: u32,
) -> Result<CachedBlock, Exit> {
    // Keep compilation and publication state out of non-tiered cache hits.
    if !vm.tier_up_background && vm.tier_up_after.is_none() && mode != CpuMode::Real16 {
        stats.fetch_calls = stats.fetch_calls.saturating_add(1);
        let _time_fetch = vm
            .time_enabled
            .then(|| crate::time_stats::ResolveTimer::enter(stats.fetch_calls));
        let key = if DEFAULT {
            BlockKey::new(at.pa, mode)
        } else {
            BlockKey::with_mxcsr(at.pa, mode, mxcsr)
        };
        loop {
            stats.lookups = stats.lookups.saturating_add(1);
            if let Some(block) = vm.cache.get_for_dispatch(key, vm.time_enabled) {
                return Ok(block);
            }
            stats.misses = stats.misses.saturating_add(1);
            if vm.time_enabled {
                crate::time_stats::ResolveTimer::miss();
            }
            // No IR exists on a miss; snapshot before fetching its bytes.
            let generation = vm.cache.publication_generation();
            match resolve_uncached::<DEFAULT>(vm, exit_word, sections, at, mode, mxcsr, generation)
            {
                Ok(block) => return Ok(block),
                Err(ResolveFailure::Exit(exit)) => return Err(exit),
                Err(ResolveFailure::Retry) => continue,
            }
        }
    }
    resolve::<DEFAULT>(vm, stats, exit_word, sections, at, mode, mxcsr)
}

#[inline(never)]
fn resolve<const DEFAULT: bool>(
    vm: &Vm,
    stats: &mut crate::time_stats::DispatchCounts,
    exit_word: &AtomicU64,
    sections: &AtomicU64,
    at: FetchAddr,
    mode: CpuMode,
    mxcsr: u32,
) -> Result<CachedBlock, Exit> {
    stats.fetch_calls = stats.fetch_calls.saturating_add(1);
    let _time_fetch = vm
        .time_enabled
        .then(|| crate::time_stats::ResolveTimer::enter(stats.fetch_calls));
    loop {
        match resolve_attempt::<DEFAULT>(vm, stats, exit_word, sections, at, mode, mxcsr) {
            Ok(block) => return Ok(block),
            Err(ResolveFailure::Exit(exit)) => return Err(exit),
            Err(ResolveFailure::Retry) => continue,
        }
    }
}

fn resolve_attempt<const DEFAULT: bool>(
    vm: &Vm,
    stats: &mut crate::time_stats::DispatchCounts,
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
    let source = crate::memory::LiftSource::new(&vm.mem);
    let pc = at.pa;
    let key = if DEFAULT {
        BlockKey::new(pc, mode)
    } else {
        BlockKey::with_mxcsr(pc, mode, mxcsr)
    };
    // §17.6: Real16 has no compiled tier. Bypass tier-up and region formation even
    // when the embedder injected a compiling backend.
    if mode == CpuMode::Real16 {
        stats.lookups = stats.lookups.saturating_add(1);
        if let Some(block) = vm.cache.get_for_dispatch(key, vm.time_enabled) {
            return Ok(block);
        }
        stats.misses = stats.misses.saturating_add(1);
        if vm.time_enabled {
            crate::time_stats::ResolveTimer::miss();
        }
        return resolve_real16_uncached::<DEFAULT>(vm, &source, exit_word, sections, at, mxcsr);
    }
    loop {
        // Publish any completed background compiles first, so a
        // freshly-landed unit is seen by the lookup below. Cheap when idle (the
        // backend's ready-probe short-circuits an empty drain).
        if vm.tier_up_background {
            let _publication_time = vm
                .time_enabled
                .then(crate::time_stats::PublicationTimer::enter);
            drain_tier_up(vm, exit_word, sections);
        }
        // Snapshot before reading cached IR or fetching uncached successors:
        // even a victim-free write must cancel source publication.
        let generation = vm.cache.publication_generation();
        let dispatch_epoch = vm.cache.epoch();
        stats.lookups = stats.lookups.saturating_add(1);
        let Some(block) = vm.cache.get_for_dispatch(key, vm.time_enabled) else {
            stats.misses = stats.misses.saturating_add(1);
            if vm.time_enabled {
                crate::time_stats::ResolveTimer::miss();
            }
            return resolve_uncached::<DEFAULT>(
                vm, exit_word, sections, at, mode, mxcsr, generation,
            );
        };
        // Hotness-gated tier-up: a cached *interpreted* block that has
        // now run `tier_up_after` times gets JIT-compiled from its already-lifted
        // IR and swapped in, so cold one-shot blocks never pay compile cost while
        // hot blocks still tier up.
        let (Some(thr), CachedBlock::Interpreted(ir)) = (vm.tier_up_after, &block) else {
            return Ok(block);
        };
        if !vm.cache.code_lifetime().is_open() || vm.backend.code_space_exhausted() {
            return Ok(block);
        }
        let count = vm.cache.bump_hotness(key);
        if count < thr {
            return Ok(block);
        }
        if !vm.tier_up_background && vm.cache.tier_up_interpreter_only(key) {
            return Ok(block);
        }
        let mut permit = None;
        if vm.mem.fetch_policy.is_some() {
            let _section = EngineSection::enter(exit_word, sections, CriticalSectionKind::Publish);
            if !vm
                .cache
                .publish_if_current(dispatch_epoch, || permit = vm.mem.fetch_permit())
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
                        let c = {
                            let _time_lift = vm
                                .time_enabled
                                .then(crate::time_stats::FetchPhaseTimer::lift);
                            matches!(
                                lift_region(&vm.mem, pc, caps, mode),
                                Ok(r) if r.blocks.len() > 1 && r.has_loop
                            )
                        };
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
                    let _time_lift = vm
                        .time_enabled
                        .then(crate::time_stats::FetchPhaseTimer::lift);
                    lift_region_watched(&vm.mem, pc, caps, mode, &mut fetch_failure, &source)
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
            if !source.publishable() {
                vm.cache.end_tier_up(key);
                return Ok(block);
            }
            #[cfg(test)]
            vm.after_source_read();
            vm.handle_smc(exit_word);
            if permit.as_ref().is_some_and(|p| !p.is_current()) {
                vm.cache.end_tier_up(key);
                continue;
            }
            let Some(request_id) = vm
                .cache
                .prepare_tier_up(key, spans.clone(), generation, |sp| {
                    for &(start, len) in sp {
                        vm.mem.mark_code(start, len);
                    }
                })
            else {
                vm.cache.end_tier_up(key);
                continue;
            };
            let req = TierUpRequest {
                precise_split_stores: vm.precise_split_stores,
                mxcsr_mode: key.mxcsr_mode,
                pc,
                unit,
                consistency: vm.consistency,
                mmio: vm.mem.trap_window(),
                guest_base: vm.mem.guest_base(),
                spans,
                epoch: request_id,
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
                TierUpSubmit::Queued => {
                    vm.cache.record_tier_bg_submitted();
                    return Ok(block);
                }
                // Queue full: don't compile inline (that reintroduces the spike);
                // drop the marker so hotness re-submits on a later dispatch.
                TierUpSubmit::Busy => {
                    vm.cache.abandon_tier_up(key, &vm.mem);
                    return Ok(block);
                }
                // No worker (interpreter, or the JIT with bg off): fall through to
                // today's inline tier-up.
                TierUpSubmit::Unsupported => {
                    vm.cache.abandon_tier_up(key, &vm.mem);
                }
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
        vm.handle_smc(exit_word);
        if vm.cache.upgrade(
            key,
            compiled.clone(),
            (ir.guest_start, ir.guest_len),
            generation,
        ) {
            return Ok(compiled);
        }
        discard_materialized(vm, &compiled);
        if vm.cache.publication_generation() == generation
            && permit.as_ref().is_none_or(|p| p.is_current())
        {
            return Ok(block);
        }
        // An SMC drop invalidated the source mid-tier-up. Re-fetch fresh bytes.
    }
}

// Lift/publication state belongs to a miss, keeping cache hits independent of
// the materializer's stack frame and drop guards.
#[cold]
#[inline(never)]
fn resolve_real16_uncached<const DEFAULT: bool>(
    vm: &Vm,
    source: &crate::memory::LiftSource<'_>,
    exit_word: &AtomicU64,
    sections: &AtomicU64,
    at: FetchAddr,
    mxcsr: u32,
) -> Result<CachedBlock, ResolveFailure> {
    let mode = CpuMode::Real16;
    let pc = at.pa;
    let key = if DEFAULT {
        BlockKey::new(pc, mode)
    } else {
        BlockKey::with_mxcsr(pc, mode, mxcsr)
    };
    let compile_generation = vm.cache.publication_generation();
    let permit = vm.mem.fetch_permit();
    let mut fetch_failure = None;
    let lifted = {
        let _section = EngineSection::enter(exit_word, sections, CriticalSectionKind::Lift);
        let _time_lift = vm
            .time_enabled
            .then(crate::time_stats::FetchPhaseTimer::lift);
        lift_block_watched(&vm.mem, at, mode, &mut fetch_failure, source)
    };
    lifted
        .map_err(|e| ResolveFailure::Exit(lift_exit(e, fetch_failure, &vm.mem, at, mode)))
        .and_then(|ir| {
            finish_single(
                vm,
                source,
                exit_word,
                sections,
                key,
                pc,
                ir,
                permit.as_ref(),
                compile_generation,
            )
            .ok_or(ResolveFailure::Retry)
        })
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
    compile_generation: u64,
) -> Result<CachedBlock, ResolveFailure> {
    let source = crate::memory::LiftSource::new(&vm.mem);
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
    if let Some(caps) = vm
        .backend
        .region_caps()
        .filter(|_| !vm.tier_up_background && vm.cache.code_lifetime().is_open())
    {
        let mut fetch_failure = None;
        let lifted = {
            let _section =
                EngineSection::enter(exit_word, sections, CriticalSectionKind::RegionBuild);
            let _time_lift = vm
                .time_enabled
                .then(crate::time_stats::FetchPhaseTimer::lift);
            lift_region_watched(&vm.mem, pc, caps, mode, &mut fetch_failure, &source)
        };
        match lifted {
            // NOTE: the region path is JIT-only (a region-forming backend); Real16 stays
            // on the interpreter and never reaches here, so `pc` (flat) is correct.
            // Only a multi-block region *with a loop* is worth its heavier compile
            // (it amortizes over the iterations); everything else stays single-block.
            Ok(region) if region.blocks.len() > 1 && region.has_loop => {
                if !source.publishable() {
                    if permit.as_ref().is_some_and(|p| !p.is_current()) {
                        return Err(ResolveFailure::Retry);
                    }
                    return Ok(source.uncached(region.blocks.into_iter().next().unwrap()));
                }
                let spans = region.spans();
                #[cfg(test)]
                vm.after_source_read();
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
                vm.handle_smc(exit_word);
                // §10: tag every sub-block's pages — under the spans lock (#12).
                if !vm
                    .cache
                    .insert_checked(key, materialized.clone(), spans, |sp| {
                        if vm.cache.publication_generation() != compile_generation
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
                    discard_materialized(vm, &materialized);
                    let ir = region.blocks.into_iter().next().unwrap();
                    let len = ir.guest_len;
                    let fallback = CachedBlock::Interpreted(Arc::new(ir));
                    return vm
                        .cache
                        .insert_checked(key, fallback.clone(), vec![(pc, len)], |_| {
                            if vm.cache.publication_generation() != compile_generation
                                || permit.as_ref().is_some_and(|p| !p.is_current())
                            {
                                return false;
                            }
                            vm.mem.mark_code(pc, len);
                            true
                        })
                        .then_some(fallback)
                        .ok_or(ResolveFailure::Retry);
                }
                if matches!(materialized, CachedBlock::Compiled { .. }) {
                    vm.cache.record_region();
                }
                return Ok(materialized);
            }
            Ok(region) => {
                return finish_single(
                    vm,
                    &source,
                    exit_word,
                    sections,
                    key,
                    pc,
                    region.blocks.into_iter().next().unwrap(),
                    permit.as_ref(),
                    compile_generation,
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
        let _time_lift = vm
            .time_enabled
            .then(crate::time_stats::FetchPhaseTimer::lift);
        lift_block_watched(&vm.mem, at, mode, &mut fetch_failure, &source)
    };
    match lifted {
        Ok(ir) => finish_single(
            vm,
            &source,
            exit_word,
            sections,
            key,
            pc,
            ir,
            permit.as_ref(),
            compile_generation,
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

fn discard_materialized(vm: &Vm, block: &CachedBlock) {
    // The cache may belong to a replaced backend. Only the allocator's domain
    // can retire an unpublished entry rejected by that cache.
    if let Some(lifetime) = vm.backend.code_lifetime() {
        lifetime.discard(block);
    }
}

/// Materialize a single block, cache it under its `key` with its one span, and tag
/// its pages. `span_start` is the physical fetch address (== `ir.guest_start` outside
/// Real16, but the `cs_base + IP` physical address in Real16, where SMC page-tagging
/// and the cache key must both be physical — see `resolve`).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn finish_single(
    vm: &Vm,
    source: &crate::memory::LiftSource<'_>,
    exit_word: &AtomicU64,
    sections: &AtomicU64,
    key: BlockKey,
    span_start: u64,
    ir: IrBlock,
    permit: Option<&crate::FetchPermit>,
    compile_generation: u64,
) -> Option<CachedBlock> {
    assert!(
        ir.legacy_context.is_none(),
        "legacy IR cannot enter application materialization"
    );
    #[cfg(test)]
    vm.after_source_read();
    if permit.is_some_and(|p| !p.is_current()) {
        return None;
    }
    if !source.publishable() {
        return Some(source.uncached(ir));
    }
    let (start, len) = (span_start, ir.guest_len);
    // Real16 stays interpreted (§17.6); other modes defer compilation only until
    // the block proves hot under a tier-up policy (see `resolve`).
    let mut ir = Some(ir);
    let materialized = if key.mode == CpuMode::Real16
        || vm.tier_up_after.is_some()
        || !vm.cache.code_lifetime().is_open()
    {
        CachedBlock::Interpreted(Arc::new(ir.take().unwrap()))
    } else {
        let _section = EngineSection::enter(exit_word, sections, CriticalSectionKind::Compile);
        vm.hold_compile();
        vm.backend.materialize_with_split_stores(
            ir.as_ref().unwrap(),
            vm.consistency,
            vm.mem.trap_window(),
            vm.mem.guest_base(),
            key.mxcsr_mode,
            vm.precise_split_stores,
        )
    };
    let _section = EngineSection::enter(exit_word, sections, CriticalSectionKind::Publish);
    vm.handle_smc(exit_word);
    // §10: tag the block's pages under the spans lock, so the tag can't be cleared
    // by a concurrent SMC invalidation between insert and mark.
    let publish = |_: &[(u64, u32)]| {
        if vm.cache.publication_generation() != compile_generation
            || permit.is_some_and(|p| !p.is_current())
        {
            return false;
        }
        vm.mem.mark_code(start, len);
        true
    };
    if vm
        .cache
        .insert_checked(key, materialized.clone(), vec![(start, len)], publish)
    {
        return Some(materialized);
    }
    discard_materialized(vm, &materialized);
    // A domain mismatch is permanent for this run. Publish valid source as
    // interpreted once, rather than repeatedly allocating rejected machine code.
    let fallback = CachedBlock::Interpreted(Arc::new(ir?));
    vm.cache
        .insert_checked(key, fallback.clone(), vec![(start, len)], publish)
        .then_some(fallback)
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
    use crate::AccessKind;

    #[test]
    fn real16_slices_ignore_previous_scratch_values() {
        fn setup() -> (Vm, Vcpu) {
            let mut vm = Vm::new(VmConfig::flat(0x10000));
            vm.set_cpu_mode(CpuMode::Real16);
            vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
            let mut code = vec![
                0xb8, 0x34, 0x12, // mov ax,0x1234
                0xb4, 0xab, // mov ah,0xab
                0x01, 0xd8, // add ax,bx
                0xd1, 0xe0, // shl ax,1
                0xa3, 0, 0x30, // mov [0x3000],ax
                0x50, 0x5b, // push ax; pop bx
                0xbe, 0, 0x30, // mov si,0x3000
                0xbf, 0, 0x40, // mov di,0x4000
                0xb9, 4, 0, // mov cx,4
                0xf3, 0xa5, // rep movsw
            ];
            let back = (-(code.len() as i16 + 3)).to_le_bytes();
            code.extend_from_slice(&[0xe9, back[0], back[1]]);
            vm.write_bytes(0x1000, &code).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Cs, 0x100);
            cpu.set_reg(Reg::Rsp, 0x8000);
            cpu.set_reg(Reg::Rax, 0xdead_beef_0000_0000);
            cpu.set_reg(Reg::Rbx, 0xcafe_babe_0000_0000);
            cpu.interp_scratch.grow(4096);
            (vm, cpu)
        }
        let (zero_vm, mut zero) = setup();
        let (dirty_vm, mut dirty) = setup();
        zero.interp_scratch.fill_before_block = Some(0);
        dirty.interp_scratch.fill_before_block = Some(0xa5a5_5a5a_dead_beef);
        for budget in [0, 1, 7, 64, 256].into_iter().cycle().take(100) {
            let a = zero
                .run_steps(&zero_vm, budget, StepOptions::default())
                .unwrap();
            let b = dirty
                .run_steps(&dirty_vm, budget, StepOptions::default())
                .unwrap();
            assert_eq!(a.steps(), budget);
            assert_eq!(format!("{a:?}"), format!("{b:?}"));
            assert_eq!(format!("{:?}", zero.cpu), format!("{:?}", dirty.cpu));
            assert_eq!(zero.retired_instructions(), dirty.retired_instructions());
            let mut a = [0; 0x6000];
            let mut b = [0; 0x6000];
            zero_vm.read_bytes(0x2000, &mut a).unwrap();
            dirty_vm.read_bytes(0x2000, &mut b).unwrap();
            assert_eq!(a, b);
        }
        assert!(dirty.interp_scratch.block_entries > 100);
        assert_eq!(
            zero.interp_scratch.block_entries,
            dirty.interp_scratch.block_entries
        );
    }

    fn scratch_pair(code: &[u8]) -> [(Vm, Vcpu); 2] {
        [0, 0xa5a5_5a5a_dead_beef].map(|pattern| {
            let mut vm = Vm::new(VmConfig::flat(0x10000));
            vm.set_cpu_mode(CpuMode::Real16);
            for page in (0..0x10000).step_by(4096) {
                vm.map(page, 4096, Prot::RWX, RegionKind::Ram).unwrap();
            }
            vm.write_bytes(0x1000, code).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Cs, 0x100);
            cpu.set_reg(Reg::Ss, 0x200);
            cpu.set_reg(Reg::Rsp, 0x800);
            cpu.interp_scratch.fill_before_block = Some(pattern);
            (vm, cpu)
        })
    }

    fn compare_scratch_steps(pair: &mut [(Vm, Vcpu); 2], budget: u64) -> StepRun {
        let [(a_vm, a), (b_vm, b)] = pair;
        let left = a.run_steps(a_vm, budget, StepOptions::default()).unwrap();
        let right = b.run_steps(b_vm, budget, StepOptions::default()).unwrap();
        assert_eq!(format!("{left:?}"), format!("{right:?}"));
        assert_eq!(format!("{:?}", a.cpu), format!("{:?}", b.cpu));
        assert_eq!(a.retired_instructions(), b.retired_instructions());
        assert_eq!((a.sti_shadow, a.ss_shadow), (b.sti_shadow, b.ss_shadow));
        assert_eq!(
            a.interp_scratch.block_entries,
            b.interp_scratch.block_entries
        );
        left
    }

    fn compare_scratch_memory(pair: &[(Vm, Vcpu); 2], addr: u64, len: usize) {
        let mut left = vec![0; len];
        let mut right = vec![0; len];
        pair[0].0.read_bytes(addr, &mut left).unwrap();
        pair[1].0.read_bytes(addr, &mut right).unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn poisoned_scratch_fault_retry_and_x87() {
        // Fault atomicity/restart: SDM Vol 3A §7.5; x87 memory and pointer
        // state: Vol 1 §§8.1.8, 8.6. Compare zeroed and poisoned block entries.
        for (code, ip, access) in [
            (
                &[0xb8, 0x34, 0x12, 0xa3, 0, 0x70, 0x43, 0xf4][..],
                3,
                AccessKind::Write,
            ),
            // fld1; fadd dword [0x7000]; fstp dword [0x6000]; hlt
            (
                &[0xd9, 0xe8, 0xd8, 6, 0, 0x70, 0xd9, 0x1e, 0, 0x60, 0xf4],
                2,
                AccessKind::Read,
            ),
            // fld1; fstp dword [0x7000]; hlt
            (
                &[0xd9, 0xe8, 0xd9, 0x1e, 0, 0x70, 0xf4],
                2,
                AccessKind::Write,
            ),
            // fld1; o32 fnstenv [0x7000]; hlt
            (
                &[0xd9, 0xe8, 0x66, 0xd9, 0x36, 0, 0x70, 0xf4],
                2,
                AccessKind::Write,
            ),
        ] {
            let mut pair = scratch_pair(code);
            for (vm, _) in &mut pair {
                vm.unmap(0x7000, 4096).unwrap();
            }
            for attempt in 0..2 {
                let result = compare_scratch_steps(&mut pair, 16);
                assert_eq!(result.steps(), u64::from(attempt == 0));
                assert!(
                    matches!(result.stop(), StepStop::Exit(Exit::UnmappedMemory {
                    addr: 0x7000, access: actual, ..
                }) if *actual == access),
                    "{result:?}"
                );
                assert_eq!(pair[0].1.reg(Reg::Rip), ip);
            }
            for (vm, _) in &mut pair {
                vm.map(0x7000, 4096, Prot::RW, RegionKind::Ram).unwrap();
                vm.write_bytes(0x7000, &2.0_f32.to_le_bytes()).unwrap();
            }
            let result = compare_scratch_steps(&mut pair, 16);
            assert!(matches!(result.stop(), StepStop::Exit(Exit::Hlt)));
            compare_scratch_memory(&pair, 0x6000, 0x1000);
            compare_scratch_memory(&pair, 0x7000, 0x1000);
            if access == AccessKind::Read {
                let mut value = [0; 4];
                pair[0].0.read_bytes(0x6000, &mut value).unwrap();
                assert_eq!(f32::from_le_bytes(value), 3.0);
            }
            assert!(pair[1].1.interp_scratch.block_entries >= 3);
        }
    }

    #[test]
    fn poisoned_scratch_mmio_read_write_retries() {
        // mov ax,[0x7000]; add ax,1; mov [0x7002],ax; hlt
        let mut pair = scratch_pair(&[0xa1, 0, 0x70, 0x83, 0xc0, 1, 0xa3, 2, 0x70, 0xf4]);
        for (vm, _) in &mut pair {
            vm.unmap(0x7000, 4096).unwrap();
            vm.map(0x7000, 4096, Prot::RW, RegionKind::Trap).unwrap();
        }
        let result = compare_scratch_steps(&mut pair, 16);
        assert_eq!(result.steps(), 0);
        assert!(matches!(
            result.stop(),
            StepStop::Exit(Exit::MmioRead {
                addr: 0x7000,
                size: 2,
                ..
            })
        ));
        for (_, cpu) in &mut pair {
            cpu.complete_mmio_read(0x1234);
        }
        compare_scratch_steps(&mut pair, 0);
        let result = compare_scratch_steps(&mut pair, 16);
        assert_eq!(result.steps(), 2);
        assert!(matches!(
            result.stop(),
            StepStop::Exit(Exit::MmioWrite {
                addr: 0x7002,
                size: 2,
                value: 0x1235,
                ..
            })
        ));
        for (_, cpu) in &mut pair {
            cpu.complete_mmio_write();
        }
        compare_scratch_steps(&mut pair, 0);
        let result = compare_scratch_steps(&mut pair, 16);
        assert_eq!(result.steps(), 2);
        assert!(matches!(result.stop(), StepStop::Exit(Exit::Hlt)));
        assert_eq!(pair[0].1.reg(Reg::Rax), 0x1235);
        assert!(pair[1].1.interp_scratch.block_entries >= 3);
    }

    #[test]
    fn poisoned_scratch_delivered_divide_and_x87_faults() {
        // SDM Vol 3A §7.5; Vol 1 §8.6: #DE and pending #MF delivery.
        for (code, vector) in [(&[0xf6, 0xf3][..], 0), (&[0x9b][..], 16)] {
            let mut pair = scratch_pair(code);
            for (vm, cpu) in &mut pair {
                vm.write_bytes(vector * 4, &[0, 0, 0, 3]).unwrap();
                vm.write_bytes(0x3000, &[0x43, 0xf4]).unwrap(); // inc bx; hlt
                if vector == 16 {
                    cpu.cpu.fpu_cw &= !1;
                    cpu.cpu.fpu_sw = 0x81;
                }
            }
            let result = compare_scratch_steps(&mut pair, 16);
            assert!(matches!(result.stop(), StepStop::Exit(Exit::Hlt)));
            assert_eq!(result.steps(), 2);
            assert_eq!(pair[0].1.reg(Reg::Rbx), 1);
            compare_scratch_memory(&pair, 0x2000, 0x1000);
            assert!(pair[1].1.interp_scratch.block_entries >= 2);
        }
    }

    #[test]
    fn private_strides_spread_across_sets() {
        for base in [0x10000_u64, 0xffff_8000_0001_0000] {
            for stride in [16, 4096, 65536, 1 << 20, 4 << 20, 16 << 20] {
                let spread: std::collections::HashSet<_> = (0..64)
                    .map(|i| Vcpu::fast_index(base + i * stride))
                    .collect();
                assert!(
                    spread.len() >= 48,
                    "base={base:x} stride={stride} spread={}",
                    spread.len()
                );
            }
        }
    }

    #[test]
    fn invalidation_before_private_probe_or_publication_never_resurrects_ibtc() {
        struct IndirectBackend;
        unsafe extern "C" fn indirect(cpu: *mut u8, mem: *mut u8) -> u64 {
            // SAFETY: the dispatcher supplies live CpuState/MemCtx pointers.
            unsafe {
                let cpu = &mut *(cpu as *mut CpuState);
                let mem = &mut *(mem as *mut MemCtx);
                mem.link_slot = cpu.gpr[1];
                cpu.rip = 0x2000;
            }
            RET_IBTC_MISS
        }
        impl Backend for IndirectBackend {
            fn materialize(
                &self,
                ir: &IrBlock,
                _: MemConsistency,
                _: Option<(u64, u64)>,
                _: u64,
            ) -> CachedBlock {
                CachedBlock::Compiled {
                    entry: CompiledPtr::new(if ir.guest_start == 0x1000 {
                        indirect as *const u8
                    } else {
                        source_two as *const u8
                    }),
                }
            }
        }
        for after_probe in [false, true] {
            let mut vm = Vm::with_backend(VmConfig::flat(0x4000), Box::new(IndirectBackend));
            vm.map(0x1000, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0x1000, &[0xf4]).unwrap();
            vm.write_bytes(0x2000, &[0xf4]).unwrap();
            let key = BlockKey::new(0x2000, CpuMode::Long64);
            let old = CompiledPtr::new(source_one as *const u8);
            vm.cache.insert(
                key,
                CachedBlock::Compiled { entry: old },
                vec![(0x2000, 1)],
                |_| {},
            );
            let vm = Arc::new(vm);
            let slot = Arc::new(AtomicU64::new(0));
            let mut cpu = vm.new_vcpu();
            cpu.fast_put(0x2000, old);
            cpu.set_reg(Reg::Rcx, Arc::as_ptr(&slot) as u64);
            cpu.set_reg(Reg::Rip, 0x1000);
            cpu.ibtc_test_hook = Some(Box::new({
                let vm = vm.clone();
                let slot = slot.clone();
                move |probed| {
                    if probed == after_probe {
                        assert_eq!(
                            vm.cache.invalidate_overlapping(0x2000, 0x2001, || {}),
                            vec![key]
                        );
                        slot.store(0, Ordering::Relaxed);
                    }
                }
            }));
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rax), 2, "stale target executed");
            assert_eq!(
                slot.load(Ordering::Acquire),
                0,
                "stale descriptor republished"
            );
            cpu.fast_clear();
            assert_eq!(slot.load(Ordering::Acquire), 0);
        }
    }

    #[test]
    fn private_two_way_replacement_promotes_hits_and_flushes_both_ways() {
        let vm = Vm::new(VmConfig::flat(4096));
        let mut cpu = vm.new_vcpu();
        let keys: Vec<_> = (0..0x10000)
            .filter(|&rip| Vcpu::fast_index(rip) == Vcpu::fast_index(0))
            .take(3)
            .collect();
        assert_eq!(keys.len(), 3);
        let entry = CompiledPtr::new(std::ptr::dangling());
        assert!(cpu.fast_get(0).is_none());
        cpu.fast_put(keys[0], entry);
        cpu.fast_put(keys[1], entry);
        cpu.fast_put(keys[1], entry); // Reinstalling must not consume a second way.
        assert!(cpu.fast_get(keys[1]).is_some());
        cpu.fast_put(keys[2], entry);
        assert!(cpu.fast_get(keys[0]).is_none());
        for _ in 0..1000 {
            assert!(cpu.fast_get(keys[1]).is_some());
            assert!(cpu.fast_get(keys[2]).is_some());
        }
        cpu.fast_put(keys[1], entry); // Reinstalling MRU preserves the other way.
        assert!(cpu.fast_get(keys[2]).is_some());
        cpu.fast_clear();
        for key in keys {
            assert!(cpu.fast_get(key).is_none());
        }
        assert!(cpu.fast.iter().all(|e| e.entry.0.is_null()));
        assert_eq!(std::mem::size_of_val(&*cpu.fast), 48 * 1024);
    }

    #[test]
    fn legacy_ir_is_rejected_before_application_backend_and_region_formation() {
        use crate::legacy::{CpuGeneration, LegacyCpuModel};
        use std::panic::{catch_unwind, AssertUnwindSafe};
        struct Probe(AtomicU64);
        impl Backend for Probe {
            fn materialize(
                &self,
                _: &IrBlock,
                _: MemConsistency,
                _: Option<(u64, u64)>,
                _: u64,
            ) -> CachedBlock {
                self.0.fetch_add(1, Ordering::Relaxed);
                panic!("legacy IR entered backend");
            }
        }
        let context = crate::derive_legacy_context(
            LegacyCpuModel {
                generation: CpuGeneration::I386,
                model_semantics_id: 1,
            },
            &CpuState::legacy_real_entry([0; 6]),
        )
        .unwrap();
        let ir = IrBlock {
            mode: CpuMode::Compat32,
            guest_start: 0,
            ops: Vec::new(),
            temp_count: 0,
            guest_len: 0,
            icount: 0,
            legacy_context: Some(context),
        };
        let region = IrRegion {
            entry: 0,
            blocks: vec![ir.clone()],
            has_loop: true,
        };
        let backend = Probe(AtomicU64::new(0));
        assert!(
            catch_unwind(AssertUnwindSafe(|| backend.materialize_with_mxcsr(
                &ir,
                MemConsistency::Fast,
                None,
                0,
                0
            )))
            .is_err()
        );
        assert!(
            catch_unwind(AssertUnwindSafe(|| backend.materialize_with_split_stores(
                &ir,
                MemConsistency::Fast,
                None,
                0,
                0,
                true
            )))
            .is_err()
        );
        assert!(
            catch_unwind(AssertUnwindSafe(|| backend.materialize_region_with_mxcsr(
                &region,
                MemConsistency::Fast,
                None,
                0,
                0
            )))
            .is_err()
        );
        assert!(catch_unwind(AssertUnwindSafe(|| backend
            .materialize_region_with_split_stores(
                &region,
                MemConsistency::Fast,
                None,
                0,
                0,
                true
            )))
        .is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| crate::lift::static_succs(&ir))).is_err());
        assert_eq!(backend.0.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn unsupported_irq_preserves_the_exact_queued_vector() {
        for mode in [CpuMode::Compat32, CpuMode::Long64] {
            let mut vm = Vm::new(VmConfig::flat(4096));
            vm.set_cpu_mode(mode);
            let mut cpu = vm.new_vcpu();
            cpu.inject_irq(0x41);
            cpu.inject_irq(0x82);
            for budget in [Some(1), None] {
                assert!(matches!(
                    cpu.run(&vm, budget),
                    Exit::UnsupportedSystemContext {
                        operation: UnsupportedSystemOperation::ApplicationIrqDelivery,
                        ..
                    }
                ));
                assert_eq!(cpu.pending_irq, Some(0x82));
            }
        }
    }

    #[test]
    fn irq_refusal_clears_evidence_from_the_previous_run() {
        for mode in [CpuMode::Compat32, CpuMode::Long64] {
            for interrupted in [false, true] {
                let mut vm = Vm::new(VmConfig::flat(0x2000));
                vm.set_cpu_mode(mode);
                vm.map(0, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
                vm.write_bytes(0x1000, &[0x0f, 0x05, 0xf4]).unwrap();
                let mut cpu = vm.new_vcpu();
                cpu.cpu.rip = 0x1000;
                if interrupted {
                    cpu.exit_request().request_exit();
                    assert!(matches!(cpu.run(&vm, None), Exit::Interrupted));
                    assert!(cpu.interruption_evidence().is_some());
                } else {
                    assert!(matches!(cpu.run(&vm, None), Exit::Syscall));
                    assert!(cpu.syscall_evidence().is_some());
                }
                cpu.cpu.pending_mmio = Some(42);
                cpu.cpu.pending_port_in = Some(1);
                cpu.cpu.mmio_retry_active = true;
                cpu.inject_irq(0x82);
                let state = format!("{:?}", cpu.cpu);
                let retired = cpu.retired_instructions();
                let executed = cpu.executed_instructions();
                let mut bytes = vec![0; 0x2000];
                vm.read_bytes(0, &mut bytes).unwrap();
                assert!(matches!(
                    cpu.run(&vm, None),
                    Exit::UnsupportedSystemContext {
                        operation: UnsupportedSystemOperation::ApplicationIrqDelivery,
                        ..
                    }
                ));
                assert!(cpu.syscall_evidence().is_none());
                assert!(cpu.interruption_evidence().is_none());
                assert_eq!(cpu.pending_irq, Some(0x82));
                assert_eq!(format!("{:?}", cpu.cpu), state);
                assert_eq!(cpu.retired_instructions(), retired);
                assert_eq!(cpu.executed_instructions(), executed);
                let mut after = vec![0; 0x2000];
                vm.read_bytes(0, &mut after).unwrap();
                assert_eq!(after, bytes);
            }
        }
    }

    use super::*;

    const SOURCE_ENTRY: u64 = 0x1000;
    const SOURCE_SUCCESSOR: u64 = 0x3000;

    // A deterministic compiler for the fixture's result. Executing a stale
    // publication returns 1, while a fresh publication returns 2.
    unsafe extern "C" fn source_one(cpu: *mut u8, _: *mut u8) -> u64 {
        // SAFETY: the dispatcher supplies a live CpuState per the compiled ABI.
        unsafe { (*(cpu as *mut CpuState)).gpr[0] = 1 };
        RET_HLT
    }

    unsafe extern "C" fn source_two(cpu: *mut u8, _: *mut u8) -> u64 {
        // SAFETY: the dispatcher supplies a live CpuState per the compiled ABI.
        unsafe { (*(cpu as *mut CpuState)).gpr[0] = 2 };
        RET_HLT
    }

    struct SourceBackend {
        regions: bool,
        values: Arc<Mutex<Vec<u64>>>,
        ready: Mutex<Vec<TierUpFinished>>,
    }

    impl SourceBackend {
        fn compile<'a>(&self, blocks: impl Iterator<Item = &'a IrBlock>) -> CachedBlock {
            let value = blocks
                .flat_map(|b| &b.ops)
                .find_map(|op| match op {
                    crate::IrOp::WriteReg {
                        reg: Reg::Rax,
                        src: crate::Val::Imm(v),
                        ..
                    } => Some(*v),
                    _ => None,
                })
                .expect("fixture writes a constant result");
            self.values.lock().unwrap().push(value);
            CachedBlock::Compiled {
                entry: CompiledPtr::new(match value {
                    1 => source_one as *const u8,
                    2 => source_two as *const u8,
                    _ => panic!("unexpected fixture result"),
                }),
            }
        }
    }

    impl Backend for SourceBackend {
        fn region_caps(&self) -> Option<RegionCaps> {
            self.regions.then_some(RegionCaps {
                max_blocks: 8,
                max_icount: 128,
            })
        }

        fn materialize(
            &self,
            ir: &IrBlock,
            _: MemConsistency,
            _: Option<(u64, u64)>,
            _: u64,
        ) -> CachedBlock {
            self.compile(std::iter::once(ir))
        }

        fn materialize_region(
            &self,
            region: &IrRegion,
            _: MemConsistency,
            _: Option<(u64, u64)>,
            _: u64,
        ) -> CachedBlock {
            self.compile(region.blocks.iter())
        }

        fn tier_up_async(&self, req: TierUpRequest) -> TierUpSubmit {
            let block = match &req.unit {
                TierUpUnit::Block(ir) => self.compile(std::iter::once(ir.as_ref())),
                TierUpUnit::Region(region) => {
                    assert!(region
                        .blocks
                        .iter()
                        .any(|b| b.guest_start == SOURCE_SUCCESSOR));
                    self.compile(region.blocks.iter())
                }
            };
            self.ready.lock().unwrap().push(TierUpFinished {
                pc: req.pc,
                mxcsr_mode: req.mxcsr_mode,
                block,
                compile_failure: None,
                spans: req.spans,
                epoch: req.epoch,
            });
            TierUpSubmit::Queued
        }

        fn tier_up_finished(&self) -> Vec<TierUpFinished> {
            std::mem::take(&mut *self.ready.lock().unwrap())
        }
    }

    #[test]
    fn persistent_barrier_failure_runs_uncached_and_refetches() {
        for regions in [false, true] {
            let values = Arc::new(Mutex::new(Vec::new()));
            // Model a successful construction probe on every host, followed by
            // a thread-local EPERM (e.g. a filter installed on the lifting thread).
            let mut vm = crate::with_barrier_success(|| {
                Vm::with_backend(
                    VmConfig::flat(0x5000),
                    Box::new(SourceBackend {
                        regions,
                        values: values.clone(),
                        ready: Mutex::new(Vec::new()),
                    }),
                )
            });
            vm.map(0, 0x5000, Prot::RWX, RegionKind::Ram).unwrap();
            let mut cpu = vm.new_vcpu();
            let before = vm.memory_stats().source_barrier_failures;
            crate::with_barrier_flush_failure(|| {
                for value in 1..=3 {
                    vm.write_bytes(SOURCE_ENTRY, &[0xb8, value, 0, 0, 0, 0xf4])
                        .unwrap();
                    cpu.set_reg(Reg::Rip, SOURCE_ENTRY);
                    assert!(matches!(cpu.run(&vm, Some(16)), Exit::Hlt));
                    assert_eq!(cpu.reg(Reg::Rax), u64::from(value));
                    let stats = vm.memory_stats();
                    assert_eq!(stats.source_barrier_mode, crate::SourceBarrierMode::Process);
                    assert_eq!(stats.source_barrier_failures - before, u64::from(value) * 3);
                    assert_eq!(stats.source_barrier_last_errno, Some(1));
                    assert_eq!(stats.source_barrier_uncached_executions, u64::from(value));
                    assert_eq!(stats.cache.interpreted_entries, 0);
                    assert_eq!(stats.cache.compiled_entries, 0);
                    assert_eq!(stats.cache.region_entries, 0);
                    assert!(!vm.mem.is_code_page(1));
                }
            });
            assert!(values.lock().unwrap().is_empty(), "fallback never compiles");
        }
    }

    #[test]
    fn failed_region_barrier_neither_compiles_nor_submits_successors() {
        for background in [false, true] {
            let values = Arc::new(Mutex::new(Vec::new()));
            let mut vm = crate::with_barrier_success(|| {
                Vm::with_backend(
                    VmConfig::flat(0x5000),
                    Box::new(SourceBackend {
                        regions: true,
                        values: values.clone(),
                        ready: Mutex::new(Vec::new()),
                    }),
                )
            });
            vm.map(0, 0x5000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(SOURCE_ENTRY, &[0xe9, 0xfb, 0x1f, 0, 0])
                .unwrap();
            vm.write_bytes(
                SOURCE_SUCCESSOR,
                &[
                    0xb8, 2, 0, 0, 0, 0xff, 0xc9, 0x0f, 0x85, 0xf3, 0xdf, 0xff, 0xff, 0xf4,
                ],
            )
            .unwrap();
            if background {
                vm.set_tier_up_after(Some(1));
                vm.set_tier_up_background(true);
                crate::with_barrier_success(|| {
                    let ir = crate::lift::lift_block(
                        &vm.mem,
                        FetchAddr::flat(SOURCE_ENTRY),
                        CpuMode::Long64,
                    )
                    .unwrap();
                    let span = (SOURCE_ENTRY, ir.guest_len);
                    vm.cache.insert(
                        BlockKey::new(SOURCE_ENTRY, CpuMode::Long64),
                        CachedBlock::Interpreted(Arc::new(ir)),
                        vec![span],
                        |_| vm.mem.mark_code(span.0, span.1),
                    );
                });
            }
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, SOURCE_ENTRY);
            cpu.set_reg(Reg::Rcx, 2);
            crate::with_barrier_flush_failure(|| {
                assert!(matches!(cpu.run(&vm, Some(32)), Exit::Hlt));
            });
            assert_eq!(cpu.reg(Reg::Rax), 2);
            let stats = vm.memory_stats();
            assert_eq!(stats.source_barrier_mode, crate::SourceBarrierMode::Process);
            assert!(stats.source_barrier_uncached_executions > 0);
            assert_eq!(stats.cache.interpreted_entries, u64::from(background));
            assert_eq!(stats.cache.compiled_entries + stats.cache.region_entries, 0);
            assert_eq!(stats.cache.tier_bg_submitted, 0);
            assert!(!vm.mem.is_code_page(SOURCE_SUCCESSOR >> 12));
            assert!(values.lock().unwrap().is_empty());
        }
    }

    #[test]
    fn buffered_store_missed_tag_is_validated_before_execution() {
        buffered_store_missed_tag(false);
    }

    #[test]
    fn degraded_buffered_store_missed_tag_is_validated_before_execution() {
        crate::with_barrier_unavailable(|| buffered_store_missed_tag(true));
    }

    fn buffered_store_missed_tag(degraded: bool) {
        let mut vm = Vm::new(VmConfig::flat(0x5000));
        if degraded {
            let stats = vm.memory_stats();
            assert_eq!(
                stats.source_barrier_mode,
                crate::SourceBarrierMode::Degraded
            );
            assert_eq!(stats.source_barrier_failures, 1);
            assert_eq!(stats.source_barrier_last_errno, Some(1));
        }
        vm.map(0, 0x5000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(SOURCE_ENTRY, &[0xb8, 1, 0, 0, 0, 0xf4])
            .unwrap();
        // Rust synchronization cannot keep a real store buffered across a
        // deterministic handshake. Model that visibility separately: the writer's
        // post-store check runs while untagged, while its byte becomes visible
        // only when the process barrier drains the modeled store buffer.
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    assert!(!vm.mem.is_code_page(1));
                    vm.mem.note_write(SOURCE_ENTRY + 1, 1);
                    assert!(!vm.mem.has_dirty_code());
                    *vm.mem.source_barrier_hook.lock().unwrap() = Some(|mem| {
                        assert!(mem.is_code_page(1));
                        // SAFETY: one mapped byte, no concurrent access during this hook.
                        unsafe {
                            crate::memory::atomic_store_le(
                                mem.host_base().add(SOURCE_ENTRY as usize + 1).cast_mut(),
                                2,
                                1,
                            );
                        }
                    });
                })
                .join()
                .unwrap();
        });
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, SOURCE_ENTRY);
        assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rax), 2);
        assert_eq!(vm.mem.source_barriers.load(Ordering::Relaxed), 1);
        cpu.set_reg(Reg::Rip, SOURCE_ENTRY);
        assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rax), 2);
    }

    #[test]
    fn source_publication_races_repeated_writes() {
        source_publication_writes();
    }

    #[test]
    fn degraded_source_publication_races_repeated_writes() {
        crate::with_barrier_unavailable(source_publication_writes);
    }

    fn source_publication_writes() {
        let mut vm = Vm::new(VmConfig::flat(0x5000));
        vm.map(0, 0x5000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(SOURCE_ENTRY, &[0xb8, 1, 0, 0, 0, 0xf4])
            .unwrap();
        let mut cpu = vm.new_vcpu();
        std::thread::scope(|scope| {
            let writer = scope.spawn(|| {
                for i in 0..10_000 {
                    vm.write_bytes(SOURCE_ENTRY + 1, &[1 + (i & 1) as u8])
                        .unwrap();
                    if i % 16 == 0 {
                        std::thread::yield_now();
                    }
                }
            });
            for _ in 0..1_000 {
                cpu.set_reg(Reg::Rip, SOURCE_ENTRY);
                assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
                assert!((1..=2).contains(&cpu.reg(Reg::Rax)));
            }
            writer.join().unwrap();
        });
        for _ in 0..3 {
            cpu.set_reg(Reg::Rip, SOURCE_ENTRY);
            assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rax), 2);
        }
    }

    #[test]
    fn discarded_region_successors_and_lookahead_release_tags() {
        let mut vm = Vm::new(VmConfig::flat(0x5000));
        vm.map(0, 0x5000, Prot::RWX, RegionKind::Ram).unwrap();
        // A conditional entry has a valid successor and an unsupported WBINVD.
        vm.write_bytes(SOURCE_ENTRY, &[0x0f, 0x84, 0xfa, 0x1f, 0, 0])
            .unwrap();
        vm.write_bytes(SOURCE_ENTRY + 6, &[0xf4]).unwrap();
        vm.write_bytes(SOURCE_SUCCESSOR, &[0x0f, 0x09]).unwrap();
        {
            let source = crate::memory::LiftSource::new(&vm.mem);
            let region = lift_region_watched(
                &vm.mem,
                SOURCE_ENTRY,
                RegionCaps {
                    max_blocks: 8,
                    max_icount: 100,
                },
                CpuMode::Long64,
                &mut None,
                &source,
            )
            .unwrap();
            assert!(vm.mem.is_code_page(3));
            assert!(!region
                .blocks
                .iter()
                .any(|b| b.guest_start == SOURCE_SUCCESSOR));
            for (start, len) in region.spans() {
                vm.mem.mark_code(start, len);
            }
            // A multi-page lift batches all new tags behind one barrier.
            assert_eq!(vm.mem.source_barriers.load(Ordering::Relaxed), 1);
        }
        assert!(vm.mem.is_code_page(1));
        assert!(!vm.mem.is_code_page(3));
        vm.write_bytes(SOURCE_SUCCESSOR, &[0x90]).unwrap();
        assert!(!vm.mem.has_dirty_code());
        // Candidacy probes have no persistent owner, even on success.
        let mut fresh = Memory::new(MemoryModel::Flat { size: 0x5000 });
        fresh.map(0, 0x5000, Prot::RWX, RegionKind::Ram).unwrap();
        fresh.write_bytes(0x1fff, &[0xf4, 0x90]).unwrap();
        crate::lift::lift_block(&fresh, FetchAddr::flat(0x1fff), CpuMode::Long64).unwrap();
        assert!(!fresh.is_code_page(1));
        assert!(!fresh.is_code_page(2));
    }

    #[test]
    fn overlapping_lifts_keep_tags_until_last_owner() {
        let mem = Memory::new(MemoryModel::Flat { size: 0x5000 });
        let first = crate::memory::LiftSource::new(&mem);
        let second = crate::memory::LiftSource::new(&mem);
        first.watch(SOURCE_SUCCESSOR, 1);
        second.watch(SOURCE_SUCCESSOR, 1);
        drop(first);
        assert!(mem.is_code_page(3));
        mem.clear_code_page(3);
        assert!(mem.is_code_page(3));
        drop(second);
        assert!(!mem.is_code_page(3));
    }

    fn source_race(background: bool, regions: bool, drain: bool) {
        let values = Arc::new(Mutex::new(Vec::new()));
        let mut vm = Vm::with_backend(
            VmConfig::flat(0x5000),
            Box::new(SourceBackend {
                regions,
                values: values.clone(),
                ready: Mutex::new(Vec::new()),
            }),
        );
        vm.map(0, 0x5000, Prot::RWX, RegionKind::Ram).unwrap();
        if background {
            vm.set_tier_up_after(Some(1));
            vm.set_tier_up_background(true);
        }
        if regions {
            // Entry jumps to a different, initially untagged page. The successor
            // sets the result, decrements ECX and loops to the entry, then halts.
            vm.write_bytes(SOURCE_ENTRY, &[0xe9, 0xfb, 0x1f, 0, 0])
                .unwrap();
            vm.write_bytes(
                SOURCE_SUCCESSOR,
                &[
                    0xb8, 1, 0, 0, 0, 0xff, 0xc9, 0x0f, 0x85, 0xf3, 0xdf, 0xff, 0xff, 0xf4,
                ],
            )
            .unwrap();
        } else {
            vm.write_bytes(SOURCE_ENTRY, &[0xb8, 1, 0, 0, 0, 0xf4])
                .unwrap();
        }
        let source = if regions {
            SOURCE_SUCCESSOR
        } else {
            SOURCE_ENTRY
        };
        if background {
            let ir =
                crate::lift::lift_block(&vm.mem, FetchAddr::flat(SOURCE_ENTRY), CpuMode::Long64)
                    .unwrap();
            let span = (SOURCE_ENTRY, ir.guest_len);
            vm.cache.insert(
                BlockKey::new(SOURCE_ENTRY, CpuMode::Long64),
                CachedBlock::Interpreted(Arc::new(ir)),
                vec![span],
                |_| vm.mem.mark_code(span.0, span.1),
            );
        }
        if regions || !background {
            assert!(!vm.mem.is_code_page(source >> crate::memory::CODE_PAGE_BITS));
        }
        fn write_source(vm: &Vm) {
            let source = if vm.backend.region_caps().is_some() {
                SOURCE_SUCCESSOR
            } else {
                SOURCE_ENTRY
            };
            std::thread::scope(|scope| {
                scope
                    .spawn(|| vm.write_bytes(source + 1, &[2]).unwrap())
                    .join()
                    .unwrap();
            });
        }
        *vm.source_read_hook.lock().unwrap() = Some(if drain {
            |vm| {
                let generation = vm.cache.publication_generation();
                write_source(vm);
                vm.handle_smc(&AtomicU64::new(0));
                assert!(vm.cache.publication_generation() > generation);
            }
        } else {
            write_source
        });
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, SOURCE_ENTRY);
        cpu.set_reg(Reg::Rcx, 2);
        assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rax), 2, "stale source must never execute");
        // Background registration rejects the old source before submitting it.
        // Foreground compilation may finish, but publication must retry it.
        assert_eq!(
            *values.lock().unwrap(),
            match (background, regions) {
                (true, true) => vec![2],
                (true, false) => vec![],
                (false, _) => vec![1, 2],
            }
        );
        for _ in 0..2 {
            cpu.set_reg(Reg::Rip, SOURCE_ENTRY);
            cpu.set_reg(Reg::Rcx, 2);
            assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rax), 2);
        }
        if background {
            assert_eq!(vm.cache.tier_bg_published(), 1);
        }
    }

    #[test]
    fn background_region_watches_successor_before_registration() {
        source_race(true, true, false);
    }

    #[test]
    fn background_region_rejects_successor_write_drained_before_registration() {
        source_race(true, true, true);
    }

    #[test]
    fn foreground_region_watches_successor_before_publication() {
        source_race(false, true, false);
    }

    #[test]
    fn foreground_region_rejects_coverage_miss_drained_before_publication() {
        source_race(false, true, true);
    }

    #[test]
    fn foreground_block_rejects_coverage_miss_drained_before_publication() {
        source_race(false, false, true);
    }

    #[test]
    fn foreground_block_watches_source_before_publication() {
        source_race(false, false, false);
    }

    #[test]
    fn background_block_rejects_source_write_before_registration() {
        source_race(true, false, false);
    }

    #[test]
    fn closed_domain_without_a_replacement_interprets_with_budget_and_exit() {
        struct ClosedBackend(Arc<crate::code_lifetime::CodeLifetime>);
        impl Backend for ClosedBackend {
            fn code_lifetime(&self) -> Option<Arc<crate::code_lifetime::CodeLifetime>> {
                Some(self.0.clone())
            }
            fn materialize(
                &self,
                _: &IrBlock,
                _: MemConsistency,
                _: Option<(u64, u64)>,
                _: u64,
            ) -> CachedBlock {
                panic!("closed domain must not compile");
            }
        }
        for tiered in [false, true] {
            let domain = Arc::new(crate::code_lifetime::CodeLifetime::managed());
            let mut vm = Vm::with_backend(
                VmConfig::flat(0x2000),
                Box::new(ClosedBackend(domain.clone())),
            );
            vm.map(0, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0x1000, &[0x48, 0xff, 0xc0, 0xeb, 0xfb])
                .unwrap();
            if tiered {
                vm.set_tier_up_after(Some(1));
            }
            // SAFETY: no compilation or executable allocation has occurred.
            unsafe { domain.retire_all() };
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, 0x1000);
            assert!(matches!(cpu.run(&vm, Some(0)), Exit::BudgetExhausted));
            assert_eq!(cpu.reg(Reg::Rax), 0);
            for expected in 1..=2 {
                assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
                assert_eq!(cpu.reg(Reg::Rax), expected);
            }
            cpu.exit_request().request_exit();
            assert!(matches!(cpu.run(&vm, None), Exit::Interrupted));
            assert_eq!(cpu.reg(Reg::Rax), 2);
        }
    }

    #[test]
    fn indirect_miss_after_retirement_does_not_probe_or_refill() {
        struct Fixture<'a> {
            cache: &'a TranslationCache,
            slot: Arc<AtomicU64>,
        }
        unsafe extern "C" fn source(cpu: *mut u8, mem: *mut u8) -> u64 {
            // SAFETY: this test retains Fixture and the VM throughout the ABI call.
            unsafe {
                let cpu = &mut *(cpu as *mut CpuState);
                let fixture = &*(cpu.gpr[15] as *const Fixture<'_>);
                fixture.cache.invalidate_overlapping(0x2000, 0x2001, || {});
                cpu.rip = 0x2000;
                (*(mem as *mut MemCtx)).link_slot = Arc::as_ptr(&fixture.slot) as u64;
            }
            RET_IBTC_MISS
        }
        unsafe extern "C" fn target(_: *mut u8, _: *mut u8) -> u64 {
            RET_HLT
        }
        let mut vm = Vm::new(VmConfig::flat(0x4000));
        vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
        let domain = vm.cache.code_lifetime();
        let source = domain
            .register(CompiledPtr::new(source as *const u8), 1)
            .entry();
        let target = domain
            .register(CompiledPtr::new(target as *const u8), 1)
            .entry();
        for (rip, entry) in [(0x1000, source), (0x2000, target)] {
            vm.cache.insert(
                BlockKey::new(rip, CpuMode::Long64),
                CachedBlock::Compiled { entry },
                vec![(rip, 1)],
                |_| {},
            );
        }
        let fixture = Fixture {
            cache: &vm.cache,
            slot: Arc::new(AtomicU64::new(0)),
        };
        domain.register_slot(&fixture.slot);
        let mut cpu = vm.new_vcpu();
        cpu.cpu.gpr[15] = &fixture as *const Fixture<'_> as u64;
        cpu.cpu.rip = 0x1000;
        cpu.fast_put(0x2000, target);
        let hits = cpu.fast_hits;
        assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
        assert_eq!(cpu.fast_hits, hits);
        assert!(cpu.ibtc_refills.is_empty());
        assert_eq!(vm.cache.ibtc_filled(), 0);
        assert_eq!(fixture.slot.load(Ordering::Acquire), 0);
    }

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
                    entry: CompiledPtr::new(chain as *const u8),
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

    #[test]
    fn reset_drops_private_predictors_and_deferrals_but_keeps_lifetime_state() {
        let mut cpu = vcpu();
        cpu.fast_put(0x1234, CompiledPtr::new(std::ptr::dangling()));
        cpu.ibtc_refills.insert(0x5678, IBTC_MEGAMORPHIC_CAP);
        cpu.ret_stack.sp = 1;
        cpu.ret_stack.entries[0] = [0x1234, 0x5678];
        cpu.fast_epoch = 42;
        cpu.fast_mxcsr_mode = 0x2000;
        cpu.fast_hits = 13;
        cpu.retired = 17;
        cpu.executed = 19;
        cpu.chained = 23;
        cpu.time_dispatch.lookups = 29;
        cpu.code_space_reported = true;
        let handle = cpu.fault_handle().as_usize();
        cpu.fault_entry.seed_reset_witness();
        cpu.exit_request
            .word
            .store(1 | CODE_SPACE_EXHAUSTED, Ordering::Relaxed);
        let section = EngineSection::enter(
            &cpu.exit_request.word,
            &cpu.engine_sections,
            CriticalSectionKind::Lift,
        );
        cpu.exit_request.request_exit();
        drop(section);
        assert_ne!(cpu.engine_sections.load(Ordering::Relaxed), 0);
        // Public state may contain completions without an active retry owner.
        cpu.cpu.pending_mmio = Some(0x55);
        cpu.cpu.pending_mmio_write = true;
        assert!(!cpu.cpu.mmio_retry_active);
        cpu.reset_execution_state(ResetOptions::default());
        assert!(cpu.fast.iter().all(|e| e.entry.0.is_null()));
        assert!(cpu.ibtc_refills.is_empty());
        assert_eq!(cpu.ret_stack.sp, 0);
        assert_eq!(cpu.engine_sections.load(Ordering::Relaxed), 0);
        assert_eq!(
            cpu.exit_request.word.load(Ordering::Relaxed),
            1 | CODE_SPACE_EXHAUSTED
        );
        assert!(cpu.cpu.pending_mmio.is_none());
        assert!(!cpu.cpu.pending_mmio_write);
        assert_eq!(cpu.fault_handle().as_usize(), handle);
        cpu.fault_entry.assert_reset_witness_cleared();
        assert_eq!(cpu.fast_epoch, 42);
        assert_eq!(cpu.fast_mxcsr_mode, 0x2000);
        assert_eq!(cpu.fast_hits, 13);
        assert_eq!(cpu.retired, 17);
        assert_eq!(cpu.executed, 19);
        assert_eq!(cpu.chained, 23);
        assert_eq!(cpu.time_dispatch.lookups, 29);
        assert!(cpu.code_space_reported);
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
