//! Guest memory model (§4.1, §4.2, §8.1).

use std::cell::{Cell, RefCell, UnsafeCell};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};

use crate::ir::RmwOp;

mod sparse;
pub use sparse::SparseError;

/// SDM Vol 1 §3.3.7.1: no LA57; bits 63:47 must all agree.
pub fn is_canonical(addr: u64) -> bool {
    canonical(addr) == addr
}

/// `addr` with bit 47 copied into bits 63:48: the engine's 48-bit linear-address width.
/// A processor without 5-level paging is 48-bit CPU canonical (SDM Vol 3A §4.5 p. 4-4).
/// The engine enumerates CPUID.(EAX=07H,ECX=0):ECX[16] (LA57) clear and
/// CPUID.80000001H:EDX[29] (Intel 64) set, the pair that reports a 48-bit width
/// (Vol 3A §5.1.4 p. 5-6).
pub(crate) fn canonical(addr: u64) -> u64 {
    ((addr as i64) << 16 >> 16) as u64
}

#[cold]
fn address_miss(addr: u64, size: u8) -> MemTrap {
    if !is_canonical(addr) || !is_canonical(addr.wrapping_add(u64::from(size).saturating_sub(1))) {
        MemTrap::NonCanonical
    } else {
        MemTrap::Unmapped
    }
}

/// Classify one cold memory miss, retaining its precise failing byte for paging.
/// SDM Vol 1 §3.3.7.1; AMD APM Vol 2 §5.3.1: every byte of a
/// Long64 reference must be canonical; SS selects #SS(0), otherwise #GP(0).
#[cold]
pub fn unmapped_exit(
    rip: u64,
    base: u64,
    bytes: u64,
    long64: bool,
    stack: bool,
    fault_addr: u64,
    access: crate::exit::AccessKind,
) -> crate::exit::Exit {
    let last = base.wrapping_add(bytes.max(1) - 1);
    if long64 && (!is_canonical(base) || !is_canonical(last)) {
        crate::fault::exception_exit(
            rip,
            if stack { 12 } else { 13 },
            crate::fault::FaultCause::NonCanonical,
        )
    } else {
        crate::fault::memory_exit(rip, fault_addr, bytes as u16, access, false, false)
    }
}

/// Cold span of the active memory elements of a vector reference.
/// SDM Vol 1 §15.6.1; Vol 2A §§2.8.1/2.8.2/2.8.4; Vol 2C,
/// VMASKMOV/VPMASKMOV: inactive elements suppress memory faults. Within a
/// vector (at most 64 bytes), canonical endpoints also cover any selected
/// elements between them. An all-ones mask retains the whole operand span.
#[cold]
pub fn active_operand_span(base: u64, bytes: u64, elem: u8, mask: u64) -> (u64, u64) {
    let count = bytes / u64::from(elem);
    let selected = mask & (u64::MAX >> (64 - count));
    if selected == 0 {
        return (base, 0);
    }
    let first = u64::from(selected.trailing_zeros());
    let last = 63 - u64::from(selected.leading_zeros());
    (
        base.wrapping_add(first * u64::from(elem)),
        (last - first + 1) * u64::from(elem),
    )
}

/// Active source elements of an explicit chunk broadcast. Repeated output
/// elements select the source modulo the chunk (SDM Vol 2A §2.8.6; Vol 2C,
/// VBROADCAST, Operation). Read only on a cold miss.
#[cold]
pub fn active_broadcast_span(base: u64, chunk: u8, elem: u8, width: u16, mask: u64) -> (u64, u64) {
    let elements = u32::from(chunk / elem);
    let mut source = 0;
    for i in 0..u32::from(width / u16::from(elem)) {
        if mask & (1 << i) != 0 {
            source |= 1 << (i % elements);
        }
    }
    active_operand_span(base, u64::from(chunk), elem, source)
}

/// Memory model selection. Start with `Flat`; use `Reserved` when the guest wants a
/// sparse, huge address space (Go's ~768 GiB arena hints) (§4.1).
#[derive(Clone, Copy, Debug)]
pub enum MemoryModel {
    /// One contiguous host buffer of `size` bytes representing guest space
    /// `[0, size)`. Translation is `host_base + guest_addr`. `map()` only
    /// tags regions; it does not allocate. The span must fit below `1 << 47`.
    /// Canonical addresses outside the span are unmapped. In Long64, a data
    /// operand containing a non-canonical byte instead raises #GP/#SS
    /// (SDM Vol 1 §3.3.7.1; AMD APM Vol 2 §5.3.1).
    Flat { size: u64 },
    /// A huge (`span`-byte) guest address space `[0, span)`, same one-add translation
    /// as `Flat` but sparsely committed: untouched guest VA (a `PROT_NONE`
    /// reservation, Go's 768 GiB arena hints) costs no physical memory. To actually
    /// reserve hundreds of GiB on a normal host use [`Memory::sparse`] or provide
    /// a host reservation via [`Memory::from_host_ram`]. Constructing this model
    /// through [`Memory::new`] instead uses a plain allocation (fine only for a modest
    /// span or a test). The exclusive span top may be at most 2^47, the lower
    /// canonical half (SDM Vol 1 §3.3.7.1). Forecloses per-page guest protections.
    Reserved { span: u64 },
    /// Sparse address space via a page/region table. `map()` allocates pages.
    SoftMmu,
}

/// Invalid embedder-provided guest backing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostRamError {
    Misaligned,
    GuestBaseOutOfBounds,
    BackingTooSmall,
    NonCanonicalSpan,
}

impl std::fmt::Display for HostRamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Misaligned => "host RAM pointer must equal guest_base modulo 16",
            Self::GuestBaseOutOfBounds => "host RAM guest_base exceeds the model span",
            Self::NonCanonicalSpan => "guest backing overlaps non-canonical addresses",
            Self::BackingTooSmall => "host RAM backing is smaller than the guest extent",
        })
    }
}
impl std::error::Error for HostRamError {}

/// Host-provided backing for a `Reserved` address space. The
/// embedder can supply its own allocation policy and hand core the raw region through
/// [`Memory::from_host_ram`]. `dtor(ptr, len)` frees it when the `Memory` drops.
///
/// # Safety
/// `ptr` must retain its reserved address span `[ptr, ptr+len)` for the lifetime
/// of the `Memory` it backs, and `dtor` must correctly release exactly that region.
/// RAM must permit each performed host access. With host-fault recovery enabled,
/// host protections may reject accesses; a pure store requires only write access.
/// Guard-page hook: `protect(page_ptr, len, accessible)` `mprotect`s a page-aligned
/// sub-range of a [`HostRam`] mapping RW (`true`) or `PROT_NONE` (`false`).
pub type ProtectFn = Box<dyn Fn(*mut u8, usize, bool) + Send + Sync>;

pub struct HostRam {
    pub ptr: *mut u8,
    pub len: usize,
    /// Guest address the buffer's first byte represents (§4.1, identity mapping). The
    /// guest space is `[guest_base, guest_base + span)`; `ptr[0]` is guest `guest_base`,
    /// so translation is `host = ptr + (guest_addr - guest_base)`. `0` (the default)
    /// reproduces the historical `host = ptr + guest_addr`. An embedder sets this
    /// non-zero to make a guest address equal its own host address — `ptr as u64 -
    /// guest_base == 0`, i.e. the numeric host base is 0, so `*(guest_addr as *const u8)`
    /// from the embedder reads the same byte the guest sees. The subtraction is done in
    /// integers (never as a null-adjacent raw pointer, which would be UB).
    ///
    /// An identity mapping makes the host span the guest span, so the embedder must
    /// place the whole mapping below `1 << 47`, the top of the lower canonical half
    /// (SDM Vol 1 §3.3.7.1 p. 3-10). [`Memory::from_host_ram`] rejects a higher span
    /// with [`HostRamError::NonCanonicalSpan`]. `mmap(NULL, …)` does not place a
    /// mapping there on every host; aarch64 Linux places it above `1 << 47`. Request a
    /// low address explicitly (on Linux, a hint with `MAP_FIXED_NOREPLACE`).
    pub guest_base: u64,
    pub dtor: Box<dyn FnMut(*mut u8, usize) + Send>,
    /// Optional guard-page hook: `protect(page_ptr, len, accessible)` flips a
    /// page-aligned sub-range of this mapping between accessible (`true` →
    /// `PROT_READ|PROT_WRITE`) and inaccessible (`false` → `PROT_NONE`). `Memory::map`/
    /// `unmap` call it so an in-span-but-unmapped access hardware-faults, matching the
    /// interpreter's `region_at` trap. `None` (the default) → no guard pages. Only a
    /// host mapping can be `mprotect`ed — a `Vec` backing leaves this `None`.
    pub protect: Option<ProtectFn>,
}

impl Drop for HostRam {
    fn drop(&mut self) {
        (self.dtor)(self.ptr, self.len);
    }
}

/// Keeps the backing allocation alive for the `Memory`'s lifetime. `Backing` reads
/// bytes through a raw `ptr`/`len` (uniform hot path); this only owns the storage.
enum Owner {
    /// The aligned heap allocation, owned as a raw pointer so moving this owner
    /// does not invalidate the access pointer in Backing::ptr.
    Boxed {
        ptr: *mut u8,
        len: usize,
    },
    Host(#[allow(dead_code)] HostRam),
    Sparse(sparse::Reservation),
}

impl Drop for Owner {
    fn drop(&mut self) {
        if let Owner::Boxed { ptr, len } = *self {
            // SAFETY: Backing::boxed allocated with this exact layout and owns it.
            unsafe {
                std::alloc::dealloc(
                    ptr,
                    std::alloc::Layout::from_size_align(len.max(1), 16).unwrap(),
                )
            };
        }
    }
}

/// The contiguous host byte range backing guest RAM, translated by
/// `ptr + (guest_addr - guest_base)`. `ptr`/`len` are the access path (identical for
/// an owned `Box` or a host-provided mapping); `owner` keeps the storage alive.
/// `guest_base` is the guest address `ptr[0]` represents (0 for the historical
/// zero-based layout).
struct Backing {
    ptr: *mut u8,
    len: usize,
    guest_base: u64,
    owner: Owner,
}

// SAFETY: the raw pointer names memory this `Backing` exclusively owns (freed only
// via `owner` on drop). The bytes are guest RAM, raced across vcpus exactly like the
// old `Box<[u8]>` — ordering comes from emitted TSO barriers, not Rust aliasing (§8).
unsafe impl Send for Backing {}
unsafe impl Sync for Backing {}
unsafe impl Send for HostRam {}
unsafe impl Sync for HostRam {}

impl Backing {
    /// Allocate zeroed, 16-byte-aligned owned storage.
    fn zeroed(len: usize) -> Backing {
        let layout = std::alloc::Layout::from_size_align(len.max(1), 16).unwrap();
        // SAFETY: a nonzero, valid layout; Backing owns the resulting allocation.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Backing {
            ptr,
            len,
            guest_base: 0,
            owner: Owner::Boxed { ptr, len },
        }
    }

    /// Copy into 16-byte-aligned owned storage. Owned buffers are zero-based.
    fn boxed(b: &[u8]) -> Backing {
        let backing = Self::zeroed(b.len());
        // SAFETY: both allocations contain b.len() bytes and do not overlap.
        unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), backing.ptr, b.len()) };
        backing
    }

    /// Adopt an embedder-provided host mapping (the `Reserved` NORESERVE path),
    /// carrying its `guest_base` (the guest address `ptr[0]` represents).
    fn host(ram: HostRam) -> Backing {
        let (ptr, len, guest_base) = (ram.ptr, ram.len, ram.guest_base);
        Backing {
            ptr,
            len,
            guest_base,
            owner: Owner::Host(ram),
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    /// Guest address the first backing byte represents (§4.1).
    fn guest_base(&self) -> u64 {
        self.guest_base
    }
    fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Is this a real host mapping (the `MAP_NORESERVE` `Reserved` path), as opposed
    /// to an owned `Box`/`Vec`? Only a host mapping's physical pages can be released
    /// to the OS via `madvise(MADV_DONTNEED)`; a `Vec` backing has none, so the
    /// embedder keeps its explicit write-zero fallback there.
    fn is_host(&self) -> bool {
        matches!(self.owner, Owner::Host(_))
    }
    /// Flip host protection on `[ptr+off, ptr+off+len)` via the embedder's guard-page
    /// hook, if this is a host mapping that installed one. A no-op for a `Vec` backing
    /// or a host mapping with no hook.
    fn protect(&self, off: usize, len: usize, accessible: bool) {
        if let Owner::Host(ram) = &self.owner {
            if let Some(f) = &ram.protect {
                // SAFETY: `[off, off+len)` is caller-clamped inside `[0, self.len)`; the
                // embedder `mprotect`s a page-aligned sub-range of the mapping it owns.
                f(unsafe { self.ptr.add(off) }, len, accessible);
            }
        }
    }
}

/// Host page size assumed for guard-page `mprotect` rounding. 4 KiB on
/// every host this project targets (x86-64 and the 4 KiB aarch64 CI). A 16 KiB-page
/// host would need this parameterized — recorded as a limitation, not a config we run.
const HOST_PAGE: u64 = 4096;

/// Access protection for a mapped region (§4.2).
///
/// **ADVISORY. Nothing checks it, on either backend.** A guest store into a
/// region mapped `R` or `RX` succeeds and changes the bytes — measured, not assumed:
///
/// ```text
/// interp: exit=Hlt ro_page_now=0xdead
/// jit:    exit=Hlt ro_page_now=0xdead
/// ```
///
/// Recorded here because the name promises otherwise. An embedder that maps its guest's
/// `.text` read-only and expects a trap gets silent corruption, and this is the only
/// place it would think to look.
///
/// The engine models no permission fault at all: `MemTrap` distinguishes only
/// mapped/unmapped/MMIO plus an opted-in host fault, and `Exit` has no protection-fault variant. `map` uses this to
/// tag the region and to bound-check; `RegionKind` — not `Prot` — is what routes RAM
/// versus MMIO.
///
/// **What enforcing it would take**, so the next reader does not have to re-derive it:
/// not a check in the access path. The interpreter could consult `region_at` cheaply,
/// but the JIT's inlined accesses bound against `MemCtx.size` alone and have no region
/// map by design — adding one would put a lookup on every load and store. The mechanism
/// that already solves this shape is guard pages:
/// the embedder's [`HostRam::protect`] hook `mprotect`s the host mapping, so an
/// in-span-unmapped access hardware-faults under the JIT at no hot-path cost. Reusing it
/// for `Prot` means widening that hook from `accessible: bool`
/// (`PROT_READ|PROT_WRITE` / `PROT_NONE`) to a permission, which is an embedder contract
/// change across repositories — and it would still not cover a `Vec`-backed `Flat`
/// span, which has no host pages to protect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prot {
    R,
    RW,
    RX,
    RWX,
}

/// Region behavior (§4.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionKind {
    /// Ordinary RAM. Access is inlined into generated code — no trap-out.
    Ram,
    /// Trapped region: data accesses yield `Exit::MmioRead`/`MmioWrite`;
    /// instruction fetch yields `Exit::UnmappedMemory` with `AccessKind::Execute`.
    Trap,
    /// Read-only executable memory (a masked BIOS/option ROM mapped at, e.g.,
    /// physical `0xF0000`). Reads and instruction fetches behave EXACTLY like
    /// [`RegionKind::Ram`] — they take the same inlined backing path, at RAM
    /// speed, and code executes directly out of the region. Only stores differ:
    /// a write is **silently discarded** (`Ok(())` with the backing untouched),
    /// modelling a masked ROM whose data lines ignore the store — the guest does
    /// NOT fault, matching shadow-ROM-as-read-only. Region *contents* are loaded
    /// via [`Memory::write_bytes`] (the host loader path, which bypasses region
    /// kind), so a Rom region is populated at map time and immutable thereafter.
    ///
    /// The write-drop is enforced on the interpreter's scalar/atomic store path,
    /// including the ROM bytes of a store that also covers an adjacent RAM region
    /// (whose RAM bytes are written). Real-mode guests (the BIOS/ROM consumer) run
    /// entirely on the interpreter, so this fully covers ROM. `Rom` is deliberately
    /// **excluded from [`Memory::trap_window`]** so its reads/fetches stay inlined;
    /// under the Long64/Compat32 JIT the inlined store fast path does not consult
    /// region kind, so ROM's write-drop is an interpreter-path guarantee.
    Rom,
}

/// Embedder verdict for an instruction fetch from a guest 4 KiB page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FetchVerdict {
    Allow,
    DenyProtection,
    DenyNotPresent,
}

/// Cancellation witness for a translation approved under a fetch policy.
/// Backends must check this before starting queued compilation. Invalidation
/// during compilation is handled by the cache's publication generation, which
/// every invalidation advances, including one that drops no cached block.
#[derive(Clone)]
pub struct FetchPermit {
    generation: Arc<AtomicU64>,
    observed: u64,
}
impl FetchPermit {
    pub fn is_current(&self) -> bool {
        self.generation.load(Ordering::Acquire) == self.observed
    }
}

pub(crate) struct FetchPolicy {
    callback: Arc<dyn Fn(u64) -> FetchVerdict + Send + Sync>,
    approved: Box<[AtomicU64]>,
    generation: Arc<AtomicU64>,
    state: Mutex<FetchPolicyState>,
    wake: Condvar,
}
#[derive(Default)]
struct FetchPolicyState {
    pending: BTreeSet<(u64, u64)>,
    touched_words: BTreeSet<usize>,
}
impl FetchPolicy {
    fn new(callback: Arc<dyn Fn(u64) -> FetchVerdict + Send + Sync>, base: u64, top: u64) -> Self {
        Self {
            callback,
            approved: fresh_code_pages(base, top),
            generation: Arc::new(AtomicU64::new(0)),
            state: Mutex::new(FetchPolicyState::default()),
            wake: Condvar::new(),
        }
    }

    pub(crate) fn approve(&self, page: u64, mem: &Memory) -> FetchVerdict {
        let base_page = mem.guest_base >> CODE_PAGE_BITS;
        let (w, mask) = word_bit(page.wrapping_sub(base_page));
        let Some(word) = self.approved.get(w) else {
            return FetchVerdict::DenyNotPresent;
        };
        if word.load(Ordering::Acquire) & mask != 0 {
            return FetchVerdict::Allow;
        }
        let _time_policy = mem
            .time_enabled
            .then(crate::time_stats::FetchPhaseTimer::policy);
        loop {
            let mut state = self.state.lock().unwrap();
            let generation = self.generation.load(Ordering::Acquire);
            if word.load(Ordering::Acquire) & mask != 0 {
                return FetchVerdict::Allow;
            }
            let key = (generation, page);
            if !state.pending.insert(key) {
                drop(self.wake.wait(state).unwrap());
                continue;
            }
            // Track in-flight and step/replay fetches before the callback.
            // Clear and mark share this lock, so invalidation cannot wipe a
            // new tag after a new approval has already started.
            mem.mark_fetch_code(page, true);
            drop(state);
            // The guard also releases waiters if the embedder callback unwinds.
            struct Pending<'a>(&'a FetchPolicy, (u64, u64), &'a Memory);
            impl Drop for Pending<'_> {
                fn drop(&mut self) {
                    let mut state = self.0.state.lock().unwrap();
                    state.pending.remove(&self.1);
                    let (generation, page) = self.1;
                    let (w, mask) =
                        word_bit(page.wrapping_sub(self.2.guest_base >> CODE_PAGE_BITS));
                    if self.0.generation.load(Ordering::Acquire) == generation
                        && self.0.approved[w].load(Ordering::Acquire) & mask == 0
                    {
                        self.2.mark_fetch_code(page, false);
                    }
                    self.0.wake.notify_all();
                }
            }
            let guard = Pending(self, key, mem);
            let verdict = {
                let _time = mem
                    .time_enabled
                    .then(crate::time_stats::FetchPhaseTimer::approval);
                (self.callback)(page << CODE_PAGE_BITS)
            };
            let mut state = self.state.lock().unwrap();
            if self.generation.load(Ordering::Acquire) != generation {
                drop(state);
                drop(guard);
                continue;
            }
            if verdict == FetchVerdict::Allow {
                state.touched_words.insert(w);
                word.fetch_or(mask, Ordering::Release);
            }
            drop(state);
            drop(guard);
            return verdict;
        }
    }

    fn clear(&self, page: Option<u64>, base_page: u64, on_clear: impl FnOnce()) {
        let mut state = self.state.lock().unwrap();
        if let Some(page) = page {
            let (w, mask) = word_bit(page.wrapping_sub(base_page));
            if let Some(word) = self.approved.get(w) {
                word.fetch_and(!mask, Ordering::Release);
            }
        } else {
            for w in std::mem::take(&mut state.touched_words) {
                self.approved[w].store(0, Ordering::Release);
            }
        }
        on_clear();
        // Publish the generation only after both bitmaps are clear. Otherwise
        // a new snapshot could observe the new generation with an old approval.
        self.generation.fetch_add(1, Ordering::Release);
        self.wake.notify_all();
    }
}

/// Why a memory access could not complete inline (§8.1). The interpreter and
/// codegen turn this into the appropriate `Exit`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum MemTrap {
    Unmapped,
    /// Non-canonical linear operand (SDM Vol 1 §3.3.7.1). The tier selects
    /// #SS(0) for SS references and #GP(0) otherwise; error code zero is implied.
    NonCanonical,
    /// Host protection fault at the exact guest byte.
    HostFault,
    Mmio,
}
impl MemTrap {
    /// Exact protected byte, or the caller's original transfer address.
    /// Read this immediately after `HostFault` on the thread that made the access:
    /// another failing access on that thread replaces the cold address slot.
    pub fn fault_addr(self, transfer: u64) -> u64 {
        match self {
            Self::HostFault => last_host_fault(),
            _ => transfer,
        }
    }
}

// Like errno, this cold return channel belongs to the calling thread. The signal
// classifier never accesses TLS: the routine returns its address to Rust first.
thread_local! { static HOST_FAULT_ADDRESS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) }; }
pub(crate) fn record_host_fault(at: u64) -> MemTrap {
    HOST_FAULT_ADDRESS.with(|slot| slot.set(at));
    MemTrap::HostFault
}
fn last_host_fault() -> u64 {
    HOST_FAULT_ADDRESS.with(std::cell::Cell::get)
}

#[derive(Debug)]
pub enum MapError {
    /// A vCPU is executing or owns an outstanding device handoff.
    DeviceBusy,
    Overlap,
    OutOfBounds,
    BadAlign,
    Sparse(SparseError),
}

#[derive(Debug)]
pub enum MemError {
    Unmapped,
    Protection,
}

/// A mapped guest region. In `Flat` this only TAGS a slice of the pre-allocated
/// backing buffer with permissions/kind — it does not own memory (§4.1).
#[derive(Clone)]
struct Region {
    start: u64,
    size: usize,
    // `prot` is advisory — nothing consults it on any access path (see [`Prot`]).
    // `kind` routes RAM vs Trap (MMIO). (§4.2)
    #[allow(dead_code)]
    prot: Prot,
    kind: RegionKind,
}

/// Guest page size for SMC tracking (§10): the granularity at which a page is
/// tagged "backs translated code". Invalidation within a tagged page is byte-precise,
/// except that with a fetch policy installed a write notification invalidates the whole
/// page, so revoked execute approval cannot leave a cached block runnable. Independent of
/// any host page size.
pub const CODE_PAGE_BITS: u32 = 12;
pub(crate) const CODE_PAGE_SIZE: u64 = 1 << CODE_PAGE_BITS;

/// Historical upper bound on the guest range the SMC code-page table tracked, kept
/// only to explain why it is gone.
///
/// The table used to be one `AtomicBool` per 4 KiB page capped at 4 GiB, because one
/// bool per page across a 1 TiB `Reserved` span would commit hundreds of MiB and defeat
/// the sparse backing. `mark_code`/`note_write` past the cap simply no-opped — so code
/// placed above it, or a block straddling the boundary, was not SMC-tracked at all.
/// "Guest code always lives low" held for the fixtures and was never a guarantee; a
/// dynamic image mapped high breaks it outright.
///
/// The table is now a PACKED bitset over the whole guest span, the shape `WatchPages`
/// already uses for exactly this reason: 64 pages per word, allocated through
/// `zeroed_words` so the OS commits only the words actually touched. A 1 TiB span costs
/// 32 MiB of address space and a handful of committed pages, instead of 256 MiB of
/// bools.
#[allow(dead_code)]
const CODE_WINDOW: u64 = 4 << 30;

/// Shift for the SMC watermark's 64 KiB granules. The precise bitmaps retain
/// 4 KiB pages; this range is only a conservative pre-filter for generated stores.
/// A 2^47-byte span occupies 2^31 granules, leaving room in each packed u32 for
/// the preceding-granule skew. One atomic u64 holds both bounds: separate loads
/// could combine old and new halves into a narrowed range and miss invalidation.
/// The canonical span bound is enforced by both Memory constructors (SDM Vol 1
/// §3.3.7.1); SMC imposes no smaller limit.
pub const CODE_RANGE_BITS: u32 = 16;

/// The code-page indices a `[addr, addr+len)` access touches.
///
/// Written once because getting it subtly wrong in one of two places is how a store
/// stops invalidating a translation: `len.max(1)` so a zero-length access still names
/// the page it starts on, `saturating_add` so an access at the top of the address
/// space cannot wrap to page 0, and an **inclusive** range so an access spanning a
/// page boundary marks both.
#[inline]
fn code_page_range(addr: u64, len: u64) -> std::ops::RangeInclusive<u64> {
    let last = addr.saturating_add(len.max(1) - 1);
    (addr >> CODE_PAGE_BITS)..=(last >> CODE_PAGE_BITS)
}

/// `(word index, bit mask)` for a guest page in a page bitset.
#[inline]
const fn word_bit(page: u64) -> (usize, u64) {
    ((page >> 6) as usize, 1u64 << (page & 63))
}

/// A zero-filled `AtomicU64` array that keeps the allocator's lazy commit.
///
/// `vec![0u64; n]` goes through `alloc_zeroed`, so for a large table the
/// allocator hands back fresh anonymous pages that the OS only commits on first
/// touch. Reinterpreting that buffer (rather than building it element-by-element
/// with `collect()`) is what preserves the sparseness: a per-element initializer
/// would write every word and fault in the whole table — 33 MiB resident for a
/// 1 TiB span instead of a few KiB.
fn zeroed_words(words: usize) -> Box<[AtomicU64]> {
    let v = vec![0u64; words].into_boxed_slice();
    // SAFETY: `AtomicU64` has the same size and alignment as `u64` and no invalid
    // bit patterns, so the zero-initialized allocation is a valid `[AtomicU64]`.
    unsafe { Box::from_raw(Box::into_raw(v) as *mut [AtomicU64]) }
}

fn try_zeroed_words(words: usize) -> Result<Box<[AtomicU64]>, SparseError> {
    if words == 0 {
        return Ok(Box::default());
    }
    let layout =
        std::alloc::Layout::array::<AtomicU64>(words).map_err(|_| SparseError::Allocation)?;
    // SAFETY: zero is a valid AtomicU64 representation, with its native alignment.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) }.cast::<AtomicU64>();
    if ptr.is_null() {
        return Err(SparseError::Allocation);
    }
    // SAFETY: the allocation has exactly the layout Box<[AtomicU64]> will release.
    Ok(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, words)) })
}

/// Embedder-registered DATA-range dirty tracking.
///
/// Covers backed guest pages relative to the guest base, one bit per 4 KiB page.
/// A full zero-based 2^47 span reserves 4 GiB per table, lazily committed. It has
/// to reach that far: watched DATA is precisely the embedder's big buffers in the high
/// heap (a downstream embedder watches GPU buffers around 41 GiB), and while this table
/// borrowed the SMC code-page table's `CODE_WINDOW` cap, `watch_range` was a silent
/// no-op there and `take_dirty_ranges` always empty.
struct WatchPages {
    base_page: u64,
    /// Bit set = page is a watched data page.
    watch: Box<[AtomicU64]>,
    /// Bit set = watched page written since the last drain.
    dirty: Box<[AtomicU64]>,
    /// Indices of `watch` words holding at least one watched page — the drain's
    /// scan list, so a drain costs O(watched pages / 64) rather than O(span).
    /// Registration changes and scan-list updates share this lock.
    words: Mutex<BTreeSet<u64>>,
    /// Number of watched pages. Gates the store path (and is what the JIT loads
    /// through `watch_count_ptr`), so an unwatched memory pays one relaxed load.
    count: AtomicUsize,
}

impl WatchPages {
    /// Only pages from `guest_base` to the exclusive guest `top` need bits.
    fn new(guest_base: u64, top: u64) -> Self {
        Self {
            base_page: guest_base >> CODE_PAGE_BITS,
            watch: fresh_code_pages(guest_base, top),
            dirty: fresh_code_pages(guest_base, top),
            words: Mutex::new(BTreeSet::new()),
            count: AtomicUsize::new(0),
        }
    }

    #[inline]
    fn is_watched(&self, page: u64) -> bool {
        let (w, m) = word_bit(page.wrapping_sub(self.base_page));
        self.watch
            .get(w)
            .is_some_and(|x| x.load(Ordering::Relaxed) & m != 0)
    }

    /// Record a write to a watched `page`.
    ///
    /// Test-and-test-and-set: only the first write to a page in a drain epoch
    /// does the read-modify-write, later ones are a shared load. An unconditional
    /// RMW would take the cache line exclusive on every store, and several guest
    /// threads writing one buffer would bounce it between cores — measured at
    /// ~13.5 ns/store versus ~0.25 ns for this form (4 threads, one buffer).
    ///
    /// The `Release` on the 0->1 transition pairs with the `Acquire` in
    /// [`Self::take_dirty`], so a drain that observes a page dirty also observes
    /// the store that dirtied it. A writer that finds the bit already set adds no
    /// such edge — that is deliberate and costs nothing in practice, because
    /// dirty tracking reports WHICH pages changed and never claimed to hand out a
    /// consistent snapshot of their CONTENTS: the guest can write a page the
    /// instant after a drain returns. Embedders that need coherent bytes must
    /// quiesce the vcpus (poll-and-drain at a frame/submit boundary), exactly as
    /// before. No page is ever lost: a write racing the drain either sets the bit
    /// before the swap (reported this epoch) or after it (reported the next).
    #[inline]
    fn mark_dirty(&self, page: u64) {
        let (w, m) = word_bit(page.wrapping_sub(self.base_page));
        if let Some(word) = self.dirty.get(w) {
            if word.load(Ordering::Relaxed) & m == 0 {
                word.fetch_or(m, Ordering::Release);
            }
        }
    }

    fn watch(&self, page: u64) {
        let (w, m) = word_bit(page.wrapping_sub(self.base_page));
        let Some(word) = self.watch.get(w) else {
            // Unreachable: the table spans the whole guest address space, and
            // `map()` bounds every guest address against that span. Assert rather
            // than degrade silently — dropping the registration here is what made
            // this facility dead for high addresses in the first place.
            debug_assert!(false, "watch_range page {page:#x} outside the guest span");
            return;
        };
        let mut words = self.words.lock().unwrap();
        // Count only unwatched->watched transitions, so `count` is exact.
        if word.fetch_or(m, Ordering::Relaxed) & m == 0 {
            self.count.fetch_add(1, Ordering::Relaxed);
            words.insert(w as u64);
        }
    }

    fn unwatch(&self, page: u64) {
        let (w, m) = word_bit(page.wrapping_sub(self.base_page));
        let Some(word) = self.watch.get(w) else {
            return;
        };
        let mut words = self.words.lock().unwrap();
        if word.fetch_and(!m, Ordering::Relaxed) & m != 0 {
            self.count.fetch_sub(1, Ordering::Relaxed);
            // Drop any pending dirty bit: an unwatched page must not surface in a
            // later drain, nor come back stale if the range is watched again.
            self.dirty[w].fetch_and(!m, Ordering::Relaxed);
            if word.load(Ordering::Relaxed) == 0 {
                words.remove(&(w as u64));
            }
        }
    }

    /// Drain the dirtied watched pages, ascending. Scans only the words that hold
    /// a watched page, so the output is already sorted and duplicate-free.
    fn take_dirty(&self) -> Vec<u64> {
        // Nothing watched: the common case stays lock-free.
        if self.count.load(Ordering::Relaxed) == 0 {
            return Vec::new();
        }
        let words = self.words.lock().unwrap();
        let mut pages = Vec::new();
        for &w in words.iter() {
            let mut bits = self.dirty[w as usize].swap(0, Ordering::Acquire);
            while bits != 0 {
                pages.push(self.base_page + ((w << 6) | bits.trailing_zeros() as u64));
                bits &= bits - 1;
            }
        }
        pages
    }
}

pub(crate) const DISPATCH_DIRTY: u64 = 1;

/// The next `Memory::dispatch_identity`; 64 bits never wrap in practice.
static NEXT_DISPATCH_IDENTITY: AtomicU64 = AtomicU64::new(1);

type AliasRegistrations = BTreeMap<u64, BTreeSet<(u64, u64, usize)>>;

#[derive(Default)]
struct CodeAliases {
    pairs: BTreeSet<(u64, u64, usize)>,
    registrations: AliasRegistrations,
    // Reference counts retain an edge shared by several sub-page declarations.
    pages: BTreeMap<u64, BTreeMap<u64, usize>>,
    translated: BTreeSet<u64>,
    lifting: BTreeMap<u64, usize>,
    fetching: BTreeSet<u64>,
    live_children: usize,
    // Removed relationships and their live tags survive every drain/clear until
    // the last view drops: a child can pause before publishing its journal bit.
    retired_aliases: AliasRegistrations,
    retained_code: BTreeSet<u64>,
}

impl CodeAliases {
    fn connected(&self, page: u64) -> BTreeSet<u64> {
        let mut found = BTreeSet::from([page]);
        let mut pending = vec![page];
        while let Some(page) = pending.pop() {
            if let Some(neighbors) = self.pages.get(&page) {
                for &other in neighbors.keys() {
                    if found.insert(other) {
                        pending.push(other);
                    }
                }
            }
        }
        found
    }

    fn written_ranges(&self, lo: u64, hi: u64) -> Vec<(u64, u64)> {
        self.written_ranges_with_retired(lo, hi, &BTreeMap::new())
    }

    fn written_ranges_with_retired(
        &self,
        lo: u64,
        hi: u64,
        retired: &AliasRegistrations,
    ) -> Vec<(u64, u64)> {
        let page = lo >> CODE_PAGE_BITS;
        if !self.registrations.contains_key(&page) && !retired.contains_key(&page) {
            return vec![(lo, hi)];
        }
        let mut seen: BTreeMap<u64, Vec<(u64, u64)>> = BTreeMap::new();
        let mut pending = vec![(lo, hi)];
        while let Some((lo, hi)) = pending.pop() {
            for page in code_page_range(lo, hi - lo) {
                let lo = lo.max(page << CODE_PAGE_BITS);
                let hi = hi.min((page + 1) << CODE_PAGE_BITS);
                let covered = seen.entry(page).or_default();
                let mut missing = Vec::new();
                let mut cursor = lo;
                for &(start, end) in covered.iter() {
                    if cursor < start.min(hi) {
                        missing.push((cursor, start.min(hi)));
                    }
                    cursor = cursor.max(end);
                    if cursor >= hi {
                        break;
                    }
                }
                if cursor < hi {
                    missing.push((cursor, hi));
                }
                if missing.is_empty() {
                    continue;
                }
                covered.push((lo, hi));
                covered.sort_unstable();
                let mut merged: Vec<(u64, u64)> = Vec::new();
                for &(start, end) in covered.iter() {
                    if let Some(last) = merged.last_mut() {
                        if start <= last.1 {
                            last.1 = last.1.max(end);
                            continue;
                        }
                    }
                    merged.push((start, end));
                }
                *covered = merged;
                for pairs in self
                    .registrations
                    .get(&page)
                    .into_iter()
                    .chain(retired.get(&page))
                {
                    for &(a, b, len) in pairs {
                        for (from, to) in [(a, b), (b, a)] {
                            for &(start, end) in &missing {
                                let start = start.max(from);
                                let end = end.min(from + len as u64);
                                if start < end {
                                    pending.push((to + (start - from), to + (end - from)));
                                }
                            }
                        }
                    }
                }
            }
        }
        seen.into_values().flatten().collect()
    }

    fn edges(a: u64, b: u64, len: usize, mut visit: impl FnMut(u64, u64)) {
        let mut offset = 0;
        while offset < len as u64 {
            let (a, b) = (a + offset, b + offset);
            visit(a >> CODE_PAGE_BITS, b >> CODE_PAGE_BITS);
            offset += (CODE_PAGE_SIZE - a % CODE_PAGE_SIZE)
                .min(CODE_PAGE_SIZE - b % CODE_PAGE_SIZE)
                .min(len as u64 - offset);
        }
    }
}

/// Guest memory. Owns the host-side backing buffer and region metadata.
///
/// The `host_base` of the backing buffer is what the JIT adds to a guest
/// address to inline RAM access (§8.2.1).
///
/// **Interior mutability (§8 pitfall).** Guest RAM is written through `&self`,
/// NOT `&mut self`: one `Memory` is shared across vcpus, which write concurrently
/// and race exactly like real hardware — ordering comes from TSO barriers
/// (§8.2.3, §11), not from Rust's `&mut`. This is the one place the core is
/// deliberately `unsafe`. `CpuState` stays `&mut` and per-vcpu; only `Memory` is shared.
pub struct Memory {
    // Selects the mapping strategy in `map()`; retained for the SoftMmu switch (§4.1).
    model: MemoryModel,
    // Flat backing store behind UnsafeCell so `write(&self)` is sound; SoftMmu
    // region table comes later. Access must be bounds-checked (§8.2.3) — no raw
    // out-of-range indexing, that would be host UB.
    backing: UnsafeCell<Backing>,
    // Allocation policy is immutable; dispatch must not inspect the backing owner.
    sparse: bool,
    // Region tags (prot/kind + bounds). `map()`/`unmap()` mutate this through
    // `&mut self` before execution; per-access lookups read it through `&self`.
    regions: Vec<Region>,
    // A retained physical plan owns this exact map. Replacing the token on
    // edits also distinguishes new Memory instances without address reuse or wrap.
    mapping_identity: Arc<()>,
    // SMC tracking (§10). `code_page[p]` = page `p` backs a translated block, so a
    // write to it must invalidate the cache. Atomic (not `&mut`) because it's set
    // through `&self` at block-resolve time and read on every store — the same
    // shared-through-`&self` discipline as `backing`. `dirty` collects page-bounded
    // byte ranges written since the last drain; `dirty_flag` lets the hot path skip the lock.
    /// Bit set = page backs a translated block. Page indices are relative to
    /// `guest_base >> CODE_PAGE_BITS`, including a partially backed first page.
    code_page: CodePages,
    code_aliases: Arc<Mutex<CodeAliases>>,
    // Any declaration is live: gates the walks' alias expansion of stores made
    // without an observer (`InterpreterMemory::note_unobserved_write`).
    code_aliases_declared: AtomicBool,
    code_tag_epoch: AtomicU64,
    code_barrier_epoch: AtomicU64,
    pub(crate) source_barrier: crate::process_barrier::ProcessBarrier,
    #[cfg(test)]
    pub(crate) source_barrier_hook: Mutex<Option<fn(&Memory)>>,
    #[cfg(test)]
    pub(crate) source_barriers: AtomicUsize,
    pub(crate) fetch_policy: Option<FetchPolicy>,
    pub(crate) time_enabled: bool,
    // Packed `(lo << 32) | len` over 64 KiB granules — the *watermark* the JIT's
    // inlined stores gate on, so a store outside it skips the table entirely. See
    // `code_range_ptr` for why a watermark and not a table probe.
    code_range: AtomicU64,
    dirty: Mutex<Vec<(u64, u64)>>,
    dirty_flag: AtomicBool,
    dispatch_words: Mutex<Vec<Weak<AtomicU64>>>,
    // Distinguishes this instance from every other Memory of the process, so
    // a vCPU notices a replaced `Vm::mem` and registers its word again.
    dispatch_identity: u64,
    child_dispatch: crate::child::ChildDispatch,
    child_writes: Mutex<Vec<Weak<crate::child::ChildWrites>>>,
    // Embedder-registered DATA-range dirty tracking — a parallel facility to the SMC
    // code-page mechanism above, independent of it (a watched page need not be code).
    // See `WatchPages`.
    watch: WatchPages,
    // Guest address the backing buffer's first byte represents (§4.1). The guest space
    // is `[guest_base, guest_base + span)`; a guest address translates to a backing
    // index by subtracting this (integer arithmetic — never a null-adjacent pointer).
    // `0` (the historical default) is the zero-based layout; an embedder sets it via
    // `HostRam.guest_base` for identity mapping. Cached from the backing (it never
    // changes for a given `Memory`).
    guest_base: u64,
    // Last region index `region_at` hit — a locality cache to skip the linear scan on
    // the common case (consecutive accesses to the same region). Interior-mutable
    // (`region_at` is `&self`); only ever a hint, so `Relaxed` and staleness are fine.
    last_region: AtomicUsize,
    recover_host_faults: bool,
    // Sorted, non-overlapping [start, end) device intervals, built at map time.
    trap_intervals: Vec<(u64, u64)>,
}

// SAFETY: concurrent guest stores are intended to race like real hardware; the
// guest's expected ordering is provided by emitted TSO barriers, not by Rust
// aliasing rules. No host-side invariant is broken by concurrent access to the
// flat byte buffer (bounds-checked per access).
unsafe impl Sync for Memory {}

/// Shared bitmap ownership with ordinary slice metadata on the store path.
/// The allocation never moves or resizes. Keep the data pointer and length
/// together so probes have the same loads as an owned boxed slice.
#[repr(C)]
struct CodePages {
    data: *const AtomicU64,
    len: usize,
    owner: Arc<[AtomicU64]>,
}

// SAFETY: data points into owner, which keeps the fixed allocation alive.
// Its elements are atomic and can be shared or moved between threads.
unsafe impl Send for CodePages {}
// SAFETY: the same owned atomic allocation is safe to access concurrently.
unsafe impl Sync for CodePages {}

impl CodePages {
    fn new(words: Box<[AtomicU64]>) -> Self {
        let owner: Arc<[AtomicU64]> = Arc::from(words);
        Self {
            data: owner.as_ptr(),
            len: owner.len(),
            owner,
        }
    }
}

impl std::ops::Deref for CodePages {
    type Target = [AtomicU64];

    fn deref(&self) -> &Self::Target {
        // SAFETY: data/len describe owner's fixed allocation. Borrowing the
        // slice from self keeps owner alive for the entire returned borrow.
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }
}

/// Counted source tags for private proofs. Tracking storage can outlive Memory:
/// a vCPU may be dropped after its VM, or reused with a different VM.
pub(crate) struct CodeRegistration {
    code_page: Arc<[AtomicU64]>,
    aliases: Arc<Mutex<CodeAliases>>,
    base_page: u64,
    pages: BTreeSet<u64>,
}

impl Drop for CodeRegistration {
    fn drop(&mut self) {
        let mut aliases = self.aliases.lock().unwrap();
        for page in &self.pages {
            let count = aliases.lifting.get_mut(page).unwrap();
            *count -= 1;
            if *count != 0 {
                continue;
            }
            aliases.lifting.remove(page);
            let connected = aliases.connected(*page);
            if connected.iter().any(|p| {
                aliases.translated.contains(p)
                    || aliases.lifting.contains_key(p)
                    || aliases.fetching.contains(p)
            }) {
                continue;
            }
            for page in connected {
                if !aliases.retained_code.contains(&page) {
                    let (w, m) = word_bit(page.wrapping_sub(self.base_page));
                    if let Some(word) = self.code_page.get(w) {
                        word.fetch_and(!m, Ordering::Relaxed);
                    }
                }
            }
        }
    }
}

/// Temporary source ownership lasts through registration/publication. It is
/// separate from durable cache/request/fetch-policy tags, so discarded lookahead
/// and failed speculative successors cannot leave permanent code pages (§10).
pub(crate) struct LiftSource<'a> {
    mem: &'a Memory,
    pages: RefCell<BTreeSet<u64>>,
    copies: RefCell<Vec<(u64, Vec<u8>)>>,
    epoch: Cell<u64>,
    needs_validation: Cell<bool>,
    publishable: Cell<bool>,
}

impl<'a> LiftSource<'a> {
    pub(crate) fn new(mem: &'a Memory) -> Self {
        Self {
            mem,
            pages: RefCell::new(BTreeSet::new()),
            copies: RefCell::new(Vec::new()),
            epoch: Cell::new(0),
            needs_validation: Cell::new(false),
            publishable: Cell::new(true),
        }
    }

    pub(crate) fn watch(&self, addr: u64, len: usize) {
        let mut pages = self.pages.borrow_mut();
        let mut aliases = self.mem.code_aliases.lock().unwrap();
        for page in code_page_range(addr, len as u64) {
            if pages.insert(page) {
                *aliases.lifting.entry(page).or_default() += 1;
                self.mem.refresh_code_page(&aliases, page);
            }
        }
        // Also cover a transition owned by another lifter whose barrier has not
        // completed yet. The mutex orders tag publication and this epoch snapshot.
        let epoch = self.mem.code_tag_epoch.load(Ordering::Acquire);
        self.epoch.set(epoch);
        if self.mem.code_barrier_epoch.load(Ordering::Acquire) < epoch {
            self.needs_validation.set(true);
        }
    }

    pub(crate) fn record(&self, addr: u64, bytes: &[u8]) {
        if self.needs_validation.get() {
            self.copies.borrow_mut().push((addr, bytes.to_vec()));
        }
    }

    pub(crate) fn validate(&self) -> bool {
        let epoch = self.epoch.get();
        let copies = self.copies.take();
        if !self.needs_validation.replace(false) {
            return true;
        }
        // One process barrier covers every newly tagged page in this lift,
        // including successors. Re-read after it: a writer that missed both the
        // watermark and tag must now have made its earlier byte store visible.
        // Writers observing a tag still record dirty ranges before publication's
        // generation check; no cache-coverage rejection is assumed here.
        if self.mem.code_barrier_epoch.load(Ordering::Acquire) < epoch {
            if !self.mem.source_barrier.flush() {
                self.publishable.set(false);
                // This snapshot can run once in the interpreter. It must never
                // enter a cache or synchronize tags for another lifter.
                return true;
            }
            #[cfg(test)]
            {
                self.mem.source_barriers.fetch_add(1, Ordering::Relaxed);
                let hook = self.mem.source_barrier_hook.lock().unwrap().take();
                if let Some(hook) = hook {
                    hook(self.mem);
                }
            }
            self.mem
                .code_barrier_epoch
                .fetch_max(epoch, Ordering::Release);
        }
        copies.into_iter().all(|(addr, bytes)| {
            let mut current = vec![0; bytes.len()];
            self.mem.fetch_code(addr, &mut current) == Ok(bytes.len()) && current == bytes
        })
    }

    pub(crate) fn publishable(&self) -> bool {
        self.publishable.get()
    }

    pub(crate) fn uncached(&self, ir: crate::IrBlock) -> crate::CachedBlock {
        self.mem
            .source_barrier
            .uncached_executions
            .fetch_add(1, Ordering::Relaxed);
        crate::CachedBlock::Interpreted(std::sync::Arc::new(ir))
    }
}

impl Drop for LiftSource<'_> {
    fn drop(&mut self) {
        if self.pages.get_mut().is_empty() {
            return;
        }
        let mut aliases = self.mem.code_aliases.lock().unwrap();
        for &page in self.pages.get_mut().iter() {
            let count = aliases.lifting.get_mut(&page).unwrap();
            *count -= 1;
            if *count == 0 {
                aliases.lifting.remove(&page);
                self.mem.refresh_code_page(&aliases, page);
            }
        }
    }
}

impl Memory {
    /// Reserve a contiguous guest span and commit only mapped regions (§4.1).
    /// Supports Flat and Reserved on Linux, macOS, and Windows. Host page edges
    /// shared by regions stay committed until their last region is unmapped.
    /// Sparse VMs execute through the checked interpreter, even with a JIT backend.
    pub fn sparse(model: MemoryModel) -> Result<Self, SparseError> {
        let span = match model {
            MemoryModel::Flat { size } | MemoryModel::Reserved { span: size } => size,
            MemoryModel::SoftMmu => return Err(SparseError::UnsupportedModel),
        };
        let reservation = sparse::Reservation::new(span)?;
        let words = span.div_ceil(CODE_PAGE_SIZE).div_ceil(64) as usize;
        let code_page = try_zeroed_words(words)?;
        let watch = WatchPages {
            base_page: 0,
            watch: try_zeroed_words(words)?,
            dirty: try_zeroed_words(words)?,
            words: Mutex::new(BTreeSet::new()),
            count: AtomicUsize::new(0),
        };
        let backing = Backing {
            ptr: reservation.ptr,
            len: span as usize,
            guest_base: 0,
            owner: Owner::Sparse(reservation),
        };
        Ok(Self::from_backing_with_tracking(
            model, backing, code_page, watch,
        ))
    }

    /// Whether the host span contains uncommitted pages.
    #[inline(always)]
    pub fn is_sparse(&self) -> bool {
        self.sparse
    }

    pub(crate) fn allocation_stats(&self) -> (u64, Option<u64>, u64) {
        // SAFETY: only immutable allocation metadata is read; guest writes mutate bytes.
        let backing = unsafe { &*self.backing.get() };
        let reserved = backing.len as u64;
        let committed = match &backing.owner {
            Owner::Boxed { .. } => Some(reserved),
            Owner::Sparse(reservation) => Some(self.sparse_committed(reservation.page as u64)),
            _ => None,
        };
        let tracking = (self.code_page.len() + self.watch.watch.len() + self.watch.dirty.len())
            * std::mem::size_of::<AtomicU64>();
        (reserved, committed, tracking as u64)
    }

    /// Host bytes committed for a sparse backing: the union of the mapped regions
    /// rounded out to host pages, which is exactly what `map` commits and `unmap`
    /// leaves committed (shared edge pages count once).
    fn sparse_committed(&self, page: u64) -> u64 {
        let mut spans: Vec<(u64, u64)> = self
            .regions
            .iter()
            .filter(|r| r.size != 0)
            .map(|r| {
                (
                    r.start / page * page,
                    (r.start + r.size as u64).div_ceil(page) * page,
                )
            })
            .collect();
        spans.sort_unstable();
        let mut total = 0;
        let mut open: Option<(u64, u64)> = None;
        for (lo, hi) in spans {
            match &mut open {
                Some((_, end)) if lo <= *end => *end = (*end).max(hi),
                _ => {
                    if let Some((a, b)) = open.replace((lo, hi)) {
                        total += b - a;
                    }
                }
            }
        }
        total + open.map_or(0, |(a, b)| b - a)
    }

    pub(crate) fn child_view<'vm>(
        &'vm self,
        vm: &'vm crate::Vm,
        mode: crate::CpuMode,
    ) -> crate::ChildView<'vm> {
        let regions: Vec<_> = self
            .regions
            .iter()
            .map(|r| crate::child::Region {
                start: r.start,
                end: r.start + r.size as u64,
                kind: r.kind,
                _prot: r.prot,
            })
            .collect();
        let first_page = self.guest_base >> CODE_PAGE_BITS;
        let fetch = self.fetch_policy.as_ref().map(|policy| {
            let mut pages = BTreeSet::new();
            for r in &regions {
                if r.end > r.start && r.kind != RegionKind::Trap {
                    pages.extend((r.start >> CODE_PAGE_BITS)..=((r.end - 1) >> CODE_PAGE_BITS));
                }
            }
            pages
                .into_iter()
                .map(|page| {
                    let (w, mask) = word_bit(page.wrapping_sub(first_page));
                    let allowed = policy
                        .approved
                        .get(w)
                        .is_some_and(|v| v.load(Ordering::Acquire) & mask != 0);
                    // Snapshot a verdict without marking untouched data pages as code.
                    let verdict = if allowed {
                        FetchVerdict::Allow
                    } else {
                        let _time = self
                            .time_enabled
                            .then(crate::time_stats::FetchPhaseTimer::approval);
                        (policy.callback)(page << CODE_PAGE_BITS)
                    };
                    (page, verdict)
                })
                .collect()
        });
        self.attach_child_view(
            vm,
            mode,
            regions,
            fetch.map(crate::child::FrozenFetch::Pages),
            || {
                // Only the widened code extent can contain set bits. Retain nonzero
                // words instead of copying the full sparse address-space bitmap.
                let range = self.code_range.load(Ordering::Acquire);
                let lo = (range >> 32) << CODE_RANGE_BITS;
                let hi = ((range >> 32) + (range & 0xffff_ffff)) << CODE_RANGE_BITS;
                let first_word = ((lo >> CODE_PAGE_BITS).saturating_sub(first_page) >> 6) as usize;
                let end_word =
                    (((hi >> CODE_PAGE_BITS).saturating_sub(first_page) + 63) >> 6) as usize;
                (first_word..end_word.min(self.code_page.len()))
                    .filter_map(|i| {
                        let bits = self.code_page[i].load(Ordering::Acquire);
                        (bits != 0).then_some((i as u64, bits))
                    })
                    .collect()
            },
        )
    }

    pub(crate) fn child_view_with<'vm>(
        &'vm self,
        vm: &'vm crate::Vm,
        mut metadata: crate::ChildViewMetadata,
        mode: crate::CpuMode,
    ) -> Result<crate::ChildView<'vm>, crate::ChildViewError> {
        use crate::{ChildRegion, ChildViewError};

        metadata.regions.sort_unstable_by_key(|r| r.range().start);
        let mut parent_regions: Vec<_> = self.regions.iter().filter(|r| r.size != 0).collect();
        parent_regions.sort_unstable_by_key(|r| r.start);
        let mut regions: Vec<crate::child::Region> = Vec::new();
        let mut fetch = Vec::new();
        let mut previous_end = self.guest_base;
        for region in metadata.regions {
            let range = region.range();
            if range.start >= range.end || range.start < self.guest_base || range.end > self.size()
            {
                return Err(ChildViewError::InvalidRange);
            }
            if range.start < previous_end {
                return Err(ChildViewError::Overlap);
            }
            previous_end = range.end;
            let ram = matches!(region, ChildRegion::Ram { .. });
            let mut cursor = range.start;
            let first = parent_regions.partition_point(|r| r.start + r.size as u64 <= cursor);
            for parent in &parent_regions[first..] {
                if cursor == range.end {
                    break;
                }
                if parent.start > cursor {
                    break;
                }
                if ram && parent.kind != RegionKind::Ram {
                    return Err(ChildViewError::NotRam);
                }
                cursor = range.end.min(parent.start + parent.size as u64);
            }
            if cursor != range.end {
                return Err(ChildViewError::Unmapped);
            }
            if let ChildRegion::Ram {
                backing_offset,
                fetch: verdict,
                ..
            } = &region
            {
                if *backing_offset != (range.start - self.guest_base) as usize {
                    return Err(ChildViewError::BackingMismatch);
                }
                fetch.push((range.clone(), *verdict));
            }
            let kind = if ram {
                RegionKind::Ram
            } else {
                RegionKind::Trap
            };
            if let Some(last) = regions
                .last_mut()
                .filter(|r| r.end == range.start && r.kind == kind)
            {
                last.end = range.end;
            } else {
                regions.push(crate::child::Region {
                    start: range.start,
                    end: range.end,
                    kind,
                    _prot: Prot::RWX,
                });
            }
        }
        let ram_pages: Vec<_> = regions
            .iter()
            .filter(|r| r.kind == RegionKind::Ram)
            .map(|r| (r.start >> CODE_PAGE_BITS, (r.end - 1) >> CODE_PAGE_BITS))
            .collect();
        for &page in &metadata.code_store_refusals {
            if !ram_pages
                .iter()
                .any(|&(first, last)| first <= page && page <= last)
            {
                return Err(ChildViewError::InvalidCodePage);
            }
        }
        let first_page = self.guest_base >> CODE_PAGE_BITS;
        Ok(self.attach_child_view(
            vm,
            mode,
            regions,
            Some(crate::child::FrozenFetch::Ranges(fetch)),
            || {
                let mut words = BTreeMap::new();
                // Sample only words intersecting sparse RAM, masking unrelated pages.
                // The alias mutex held by attachment also covers snapshot/tag retention.
                for (first, last) in ram_pages {
                    let first = first - first_page;
                    let last = last - first_page;
                    for word in (first >> 6)..=(last >> 6) {
                        let mut bits = self.code_page[word as usize].load(Ordering::Acquire);
                        if word == first >> 6 {
                            bits &= u64::MAX << (first & 63);
                        }
                        if word == last >> 6 {
                            bits &= u64::MAX >> (63 - (last & 63));
                        }
                        if bits != 0 {
                            *words.entry(word).or_default() |= bits;
                        }
                    }
                }
                for page in metadata.code_store_refusals {
                    let relative = page - first_page;
                    *words.entry(relative >> 6).or_default() |= 1 << (relative & 63);
                }
                words.into_iter().collect()
            },
        ))
    }

    fn attach_child_view<'vm>(
        &'vm self,
        vm: &'vm crate::Vm,
        mode: crate::CpuMode,
        regions: Vec<crate::child::Region>,
        fetch: Option<crate::child::FrozenFetch>,
        snapshot: impl FnOnce() -> crate::child::CodeSnapshot,
    ) -> crate::ChildView<'vm> {
        let writes = Arc::new(crate::child::ChildWrites::new(&regions));
        {
            let words = self.dispatch_words.lock().unwrap();
            if self.child_dispatch.enable() {
                for word in words.iter().filter_map(Weak::upgrade) {
                    self.child_dispatch.register(&word);
                }
            }
        }
        let mut aliases = self.code_aliases.lock().unwrap();
        let code = snapshot();
        let mut journals = self.child_writes.lock().unwrap();
        journals.retain(|journal| journal.strong_count() != 0);
        journals.push(Arc::downgrade(&writes));
        aliases.live_children += 1;
        crate::ChildView {
            base: self.host_base() as usize,
            guest_base: self.guest_base,
            top: self.size(),
            regions,
            code,
            live_code: &self.code_page,
            parent: self,
            writes,
            dispatch: &self.child_dispatch,
            #[cfg(test)]
            before_tag_read: None,
            #[cfg(test)]
            before_journal: None,
            fetch,
            recover: self.recover_host_faults,
            mode,
            features: vm.guest_cpu_features(),
            identity: vm.cpuid_table().cloned(),
            x87_pointer_model: vm.x87_pointer_model(),
            lifetime: std::marker::PhantomData,
        }
    }

    pub(crate) fn release_child_view(&self, writes: &Arc<crate::child::ChildWrites>) {
        let mut aliases = self.code_aliases.lock().unwrap();
        let mut journals = self.child_writes.lock().unwrap();
        journals.retain(|journal| journal.as_ptr() != Arc::as_ptr(writes));
        aliases.live_children -= 1;
        if aliases.live_children == 0 {
            let retained = std::mem::take(&mut aliases.retained_code);
            let mut refreshed = BTreeSet::new();
            for page in retained {
                if !refreshed.contains(&page) {
                    let component = aliases.connected(page);
                    self.refresh_code_aliases(&aliases, &component);
                    refreshed.extend(component);
                }
            }
            aliases.retired_aliases.clear();
        }
    }

    /// Route Rust guest transfers through signal-context fixups when the embedder
    /// owns host protections. Disabled by default. Install a handler calling
    /// `jit_abi::fault::classify` on every accessing thread before enabling this
    /// mode. Configure before run. Wide-store atomicity across concurrent protection
    /// revocation requires embedder synchronization across the entire transfer.
    pub fn set_host_fault_recovery(&mut self, enabled: bool) {
        self.recover_host_faults = enabled;
    }

    /// Whether Rust data transfers use host-fault fixups.
    pub fn host_fault_recovery(&self) -> bool {
        self.recover_host_faults
    }

    /// Allocate backing for the selected model; mapping regions is a separate step.
    ///
    /// # Panics
    /// Panics if a Flat/Reserved span exceeds `1 << 47`, since it would include
    /// non-canonical addresses (SDM Vol 1 §3.3.7.1). Backing allocation failure aborts
    /// through `std::alloc::handle_alloc_error`; it is not a catchable panic.
    pub fn new(model: MemoryModel) -> Self {
        let span = match model {
            MemoryModel::Flat { size } | MemoryModel::Reserved { span: size } => size,
            MemoryModel::SoftMmu => 0,
        };
        assert!(
            span <= 1 << 47,
            "guest span overlaps non-canonical addresses"
        );
        // Sparse allocation is opt-in through sparse or from_host_ram.
        let len = match &model {
            MemoryModel::Flat { size } | MemoryModel::Reserved { span: size } => *size as usize,
            MemoryModel::SoftMmu => 0,
        };
        Self::from_backing(model, Backing::zeroed(len))
    }

    /// Build a `Reserved` memory over an embedder-provided host mapping.
    /// An alternative to [`Memory::sparse`] for caller-managed host storage; the
    /// `HostRam` dtor frees it on drop. `ram.len` is the span the JIT bounds against.
    /// The host pointer must be congruent to `guest_base` modulo 16, so aligned
    /// guest CMPXCHG16B operands are aligned on the host. Page-aligned mappings
    /// and identity mappings satisfy this. Invalid backings return an error.
    pub fn from_host_ram(model: MemoryModel, ram: HostRam) -> Result<Self, HostRamError> {
        // The JIT and `map()` bound guest accesses against the model span (`size()`),
        // but the backing is only `ram.len` bytes and represents guest `[guest_base,
        // guest_base + ram.len)`. A guest address `a` indexes the backing at `a -
        // guest_base`, so the guest extent above the base (`span - guest_base`) must fit
        // the mapping; otherwise an in-span access could dereference past it (OOB/UB in
        // the JIT, slice panic in the interpreter). `guest_base` must itself be `<= span`.
        let span = match &model {
            MemoryModel::Flat { size } | MemoryModel::Reserved { span: size } => *size,
            MemoryModel::SoftMmu => 0,
        };
        // SDM Vol 1 §3.3.7.1: this engine implements 48-bit canonical addresses.
        // Reject at configuration time so a non-canonical guest pointer can never
        // reach a host access/trap record or the host-fault landing pad.
        if span > 1 << 47 {
            return Err(HostRamError::NonCanonicalSpan);
        }
        if ram.guest_base > span {
            return Err(HostRamError::GuestBaseOutOfBounds);
        }
        if ram.ptr as usize & 15 != ram.guest_base as usize & 15 {
            return Err(HostRamError::Misaligned);
        }
        let extent = (span - ram.guest_base) as usize;
        if extent > ram.len {
            return Err(HostRamError::BackingTooSmall);
        }
        Ok(Self::from_backing(model, Backing::host(ram)))
    }

    fn from_backing(model: MemoryModel, backing: Backing) -> Self {
        // Owned storage is 16-byte aligned; host storage preserves guest alignment.
        let guest_base = backing.guest_base();
        let top = match model {
            MemoryModel::Flat { size } | MemoryModel::Reserved { span: size } => size,
            MemoryModel::SoftMmu => 0,
        };
        let code_page = fresh_code_pages(guest_base, top);
        let watch = WatchPages::new(guest_base, top);
        Self::from_backing_with_tracking(model, backing, code_page, watch)
    }

    fn from_backing_with_tracking(
        model: MemoryModel,
        backing: Backing,
        code_page: Box<[AtomicU64]>,
        watch: WatchPages,
    ) -> Self {
        let guest_base = backing.guest_base();
        let source_barrier = crate::process_barrier::ProcessBarrier::new();
        let code_page = CodePages::new(code_page);
        let code_aliases = Mutex::new(CodeAliases::default());
        let code_aliases = Arc::new(code_aliases);
        Self {
            model,
            sparse: matches!(backing.owner, Owner::Sparse(_)),
            backing: UnsafeCell::new(backing),
            regions: Vec::new(),
            mapping_identity: Arc::new(()),
            trap_intervals: Vec::new(),
            code_page,
            code_aliases,
            code_aliases_declared: AtomicBool::new(false),
            code_tag_epoch: AtomicU64::new(0),
            code_barrier_epoch: AtomicU64::new(0),
            source_barrier,
            #[cfg(test)]
            source_barrier_hook: Mutex::new(None),
            #[cfg(test)]
            source_barriers: AtomicUsize::new(0),
            fetch_policy: None,
            time_enabled: false,
            code_range: AtomicU64::new(0),
            dirty: Mutex::new(Vec::new()),
            dirty_flag: AtomicBool::new(false),
            dispatch_words: Mutex::new(Vec::new()),
            dispatch_identity: NEXT_DISPATCH_IDENTITY.fetch_add(1, Ordering::Relaxed),
            child_dispatch: crate::child::ChildDispatch::default(),
            child_writes: Mutex::new(Vec::new()),
            watch,
            guest_base,
            last_region: AtomicUsize::new(0),
            recover_host_faults: false,
        }
    }

    /// Deep-copy this memory into an independent `Memory` with identical contents
    /// and region tags but its own (empty) SMC/dirty state — the guest-agnostic
    /// primitive behind `fork` (§4.2). The child's byte buffer is a fresh
    /// allocation; writes on either side don't affect the other.
    pub fn deep_copy(&self) -> Option<Memory> {
        // SAFETY: snapshot at a quiescent point (between guest steps); no concurrent
        // vcpu writes the parent during a fork.
        // A boxed child is always zero-based (`guest_base = 0`); a non-zero-based
        // memory (identity mapping) can't be faithfully re-homed into one, and it is
        // always host-backed anyway (the two `None`-returning arms below already cover
        // the shipped identity-mapping case). Reject it rather than miscopy.
        if self.guest_base != 0 || self.is_sparse() {
            return None;
        }
        let owner = &unsafe { &*self.backing.get() }.owner;
        let bytes: Box<[u8]> = match (self.model, owner) {
            // A caller-owned `Reserved` span has an external allocation policy,
            // and cloning it into a `Vec` would
            // commit the whole span. Fork of such a memory is unsupported — `None` lets
            // the embedder surface a typed error instead of aborting the host. Go, the
            // only huge-`Reserved` guest, never forks; forking guests use `Flat`.
            (MemoryModel::Reserved { .. }, Owner::Host(_)) => return None,
            // A host-backed guarded `Flat` or an owned `Reserved` (Vec-backed, modest
            // span): copy only tagged regions into a fresh demand-zero span. For the
            // guarded host case this is also *required* — the unmapped holes are
            // `PROT_NONE`, so a whole-span `to_vec` would fault. The child is
            // `Vec`-backed (no guards — the documented residual; a forking guest
            // typically execve's immediately, which reloads a fresh guarded span).
            (MemoryModel::Reserved { span }, Owner::Boxed { .. })
            | (MemoryModel::Flat { size: span }, Owner::Host(_)) => {
                let mut child = vec![0u8; span as usize].into_boxed_slice();
                for r in &self.regions {
                    let s = r.start as usize;
                    let e = s + r.size;
                    unsafe { self.copy_from_guest(self.host_base().add(s), &mut child[s..e]) }
                        .ok()?;
                }
                child
            }
            _ => {
                let mut child = vec![0; self.size() as usize].into_boxed_slice();
                unsafe { self.copy_from_guest(self.host_base(), &mut child) }.ok()?;
                child
            }
        };
        let mut child = Memory::from_backing(self.model, Backing::boxed(&bytes))
            .with_regions(self.regions.clone());
        child.recover_host_faults = self.recover_host_faults;
        if let Some(policy) = &self.fetch_policy {
            child.set_fetch_policy(policy.callback.clone());
        }
        Some(child)
    }

    fn with_regions(mut self, regions: Vec<Region>) -> Self {
        self.trap_intervals = regions
            .iter()
            .filter(|r| r.kind == RegionKind::Trap && r.size != 0)
            .map(|r| (r.start, r.start + r.size as u64))
            .collect();
        self.trap_intervals.sort_unstable();
        self.regions = regions;
        self
    }

    pub(crate) fn set_fetch_policy(
        &mut self,
        callback: Arc<dyn Fn(u64) -> FetchVerdict + Send + Sync>,
    ) {
        self.fetch_policy = Some(FetchPolicy::new(callback, self.guest_base, self.size()));
    }

    pub(crate) fn fetch_permit(&self) -> Option<FetchPermit> {
        self.fetch_policy.as_ref().map(|policy| FetchPermit {
            generation: policy.generation.clone(),
            observed: policy.generation.load(Ordering::Acquire),
        })
    }

    pub(crate) fn mapping_identity(&self) -> Arc<()> {
        self.mapping_identity.clone()
    }

    /// Tag every page spanned by `[addr, addr+len)` as backing translated code
    /// (§10). Called through `&self` when a block is cached, so a later store to
    /// the page is caught. Idempotent.
    pub fn mark_code(&self, addr: u64, len: u32) {
        let mut aliases = self.code_aliases.lock().unwrap();
        // The table covers the whole guest span, so every mappable page has a bit; a
        // page outside it no-ops (`code_page.get` → None) rather than aborting.
        for page in code_page_range(addr, len as u64) {
            let (w, _) = word_bit(page.wrapping_sub(self.guest_base >> CODE_PAGE_BITS));
            if self.code_page.get(w).is_some() {
                aliases.translated.insert(page);
                self.refresh_code_page(&aliases, page);
            }
        }
    }

    pub(crate) fn register_code_sources(&self, sources: &[(u64, u32)]) -> CodeRegistration {
        let mut aliases = self.code_aliases.lock().unwrap();
        let pages: BTreeSet<_> = sources
            .iter()
            .flat_map(|&(addr, len)| code_page_range(addr, u64::from(len)))
            .collect();
        for &page in &pages {
            *aliases.lifting.entry(page).or_default() += 1;
            self.refresh_code_page(&aliases, page);
        }
        CodeRegistration {
            code_page: self.code_page.owner.clone(),
            aliases: self.code_aliases.clone(),
            base_page: self.guest_base >> CODE_PAGE_BITS,
            pages,
        }
    }

    /// Declare that corresponding bytes of two guest ranges share host storage (§10).
    /// The embedder must create the actual shared mappings and report every pair;
    /// undeclared aliases cannot be discovered by the engine. Either view may execute
    /// or be written. Duplicate declarations are idempotent; overlapping declarations
    /// and multiple views are supported. Page tags are conservative; written ranges
    /// retain their byte offsets in every declared view.
    ///
    /// Both nonempty ranges must fit the backing. Registration is safe during guest
    /// execution; synchronize with the vCPU before relying on completion, as with
    /// publishing the code bytes themselves. This does not synchronize guest bytes.
    pub fn declare_code_alias(&self, a: u64, b: u64, len: usize) -> Result<(), MapError> {
        for start in [a, b] {
            if len == 0
                || start < self.guest_base
                || start
                    .checked_add(len as u64)
                    .is_none_or(|end| end > self.size())
            {
                return Err(MapError::OutOfBounds);
            }
        }
        let mut aliases = self.code_aliases.lock().unwrap();
        let pair = (a.min(b), a.max(b), len);
        if !aliases.pairs.insert(pair) {
            return Ok(());
        }
        self.code_aliases_declared.store(true, Ordering::Relaxed);
        let mut affected = BTreeSet::new();
        CodeAliases::edges(a, b, len, |a, b| {
            *aliases.pages.entry(a).or_default().entry(b).or_default() += 1;
            *aliases.pages.entry(b).or_default().entry(a).or_default() += 1;
            aliases.registrations.entry(a).or_default().insert(pair);
            aliases.registrations.entry(b).or_default().insert(pair);
            affected.extend([a, b]);
        });
        let mut connected = BTreeSet::new();
        for page in affected {
            if !connected.contains(&page) {
                let component = aliases.connected(page);
                self.refresh_code_aliases(&aliases, &component);
                connected.extend(component);
            }
        }
        Ok(())
    }

    /// Remove an exact declaration, accepting either order of its views. Returns
    /// whether it existed. Other declarations and direct code tags remain active.
    /// While any child view lives, retain the removed relationship and its live
    /// tags for deferred writes. The last view's parent-side drop clears them;
    /// child notifications and invalidation may be conservative until then.
    pub fn remove_code_alias(&self, a: u64, b: u64, len: usize) -> bool {
        let mut aliases = self.code_aliases.lock().unwrap();
        let pair = (a.min(b), a.max(b), len);
        if !aliases.pairs.remove(&pair) {
            return false;
        }
        if aliases.pairs.is_empty() {
            self.code_aliases_declared.store(false, Ordering::Relaxed);
        }
        let mut affected = BTreeSet::new();
        CodeAliases::edges(a, b, len, |a, b| {
            affected.extend([a, b]);
            if aliases.live_children != 0 {
                for page in [a, b] {
                    aliases
                        .retired_aliases
                        .entry(page)
                        .or_default()
                        .insert(pair);
                }
            }
        });
        if aliases.live_children != 0 {
            let mut connected = BTreeSet::new();
            for &page in &affected {
                if !connected.contains(&page) {
                    connected.extend(aliases.connected(page));
                }
            }
            aliases.retained_code.extend(
                connected
                    .into_iter()
                    .filter(|&page| self.is_code_page(page)),
            );
        }
        CodeAliases::edges(a, b, len, |a, b| {
            for (from, to) in [(a, b), (b, a)] {
                let neighbors = aliases.pages.get_mut(&from).unwrap();
                let count = neighbors.get_mut(&to).unwrap();
                *count -= 1;
                if *count == 0 {
                    neighbors.remove(&to);
                }
                if neighbors.is_empty() {
                    aliases.pages.remove(&from);
                }
            }
        });
        let mut refreshed = BTreeSet::new();
        for page in affected {
            let registrations = aliases.registrations.get_mut(&page).unwrap();
            registrations.remove(&pair);
            if registrations.is_empty() {
                aliases.registrations.remove(&page);
            }
            if !refreshed.contains(&page) {
                let component = aliases.connected(page);
                self.refresh_code_aliases(&aliases, &component);
                refreshed.extend(component);
            }
        }
        true
    }

    fn refresh_code_page(&self, aliases: &CodeAliases, page: u64) {
        if aliases.pages.contains_key(&page) {
            self.refresh_code_aliases(aliases, &aliases.connected(page));
        } else {
            // Keep ordinary registration and coverage-miss clears allocation-free.
            self.refresh_code_aliases(aliases, &[page]);
        }
    }

    fn refresh_code_aliases<'p>(
        &self,
        aliases: &CodeAliases,
        pages: impl IntoIterator<Item = &'p u64> + Copy,
    ) {
        let live = pages.into_iter().any(|p| {
            aliases.translated.contains(p)
                || aliases.lifting.contains_key(p)
                || aliases.fetching.contains(p)
        });
        let mut transitioned = false;
        for &page in pages {
            let (w, m) = word_bit(page.wrapping_sub(self.guest_base >> CODE_PAGE_BITS));
            if let Some(word) = self.code_page.get(w) {
                if live {
                    if word.fetch_or(m, Ordering::Release) & m == 0 {
                        transitioned = true;
                        self.source_barrier
                            .tagged_pages
                            .fetch_add(1, Ordering::Relaxed);
                    }
                } else if !aliases.retained_code.contains(&page) {
                    word.fetch_and(!m, Ordering::Relaxed);
                }
            }
        }
        // Publish all precise tags before any widened gate. Generated stores acquire
        // the watermark before probing tags, including on AArch64. The alias mutex
        // serializes registration, marking, clearing and dirty-page expansion. When
        // the watermark already covers a new alias, the caller's vCPU synchronization
        // publishes completion; a store racing registration may precede it.
        if live {
            for &page in pages {
                self.widen_code_range((page << CODE_PAGE_BITS) >> CODE_RANGE_BITS);
            }
        }
        if transitioned {
            self.code_tag_epoch.fetch_add(1, Ordering::Release);
        }
    }

    /// Union the watermark with the code granule `p` and its predecessor.
    ///
    /// A generated store is at most 64 bytes, so it can cross at most one 64 KiB
    /// boundary. Including the preceding granule lets the gate test only the first
    /// byte, with one load and one unsigned compare. The precise helper still tests
    /// 4 KiB pages. Union the stored ranges directly, including the low skew, so
    /// marking a previously skew-only granule also covers its predecessor.
    ///
    /// CAS publishes both halves together and only widens the observed range, even
    /// when another vcpu marks code concurrently. Clearing tags never resets this
    /// watermark: that could undo a concurrent mark and skip a later invalidation.
    fn widen_code_range(&self, p: u64) {
        let mut cur = self.code_range.load(Ordering::Relaxed);
        loop {
            let (lo, len) = (cur >> 32, cur & 0xffff_ffff);
            let new_lo = if len == 0 {
                p.saturating_sub(1)
            } else {
                lo.min(p.saturating_sub(1))
            };
            let new_end = (lo + len).max(p + 1);
            let packed = (new_lo << 32) | (new_end - new_lo);
            if packed == cur {
                return;
            }
            match self.code_range.compare_exchange_weak(
                cur,
                packed,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(seen) => cur = seen,
            }
        }
    }

    /// Address of the live [`Self::code_range`] watermark, handed to generated code
    /// through `MemCtx` so an inlined store can decide with one unsigned compare
    /// whether it might have landed on a code page.
    ///
    /// Without it, JIT-compiled guest stores never consult the SMC table at all: a
    /// guest that patched another block and called it ran the STALE translation, while
    /// the same program on the interpreter observed the patch. The suite could not see
    /// it — every JIT-backed SMC test writes from the *embedder* side, which routes
    /// through [`Self::write_bytes`] and its `note_write`.
    ///
    /// A watermark rather than a bitmap probe: the watch facility can gate on
    /// `watch_count != 0`, which is zero for almost every guest, but code pages always
    /// exist once anything has run, so there is no equivalent zero-gate here. Putting a
    /// table load in the hot store stream is what got the watch bitmap's first cut
    /// reverted. Stack and heap stores fall outside the code extent of any real image,
    /// so they pay a subtract and a compare and branch over the rest.
    ///
    /// Layout: `(lo << 32) | len` over 64 KiB granules, with the one-granule low skew
    /// documented on [`Self::widen_code_range`]. A single `u64` so generated code reads
    /// both halves in one load and can never see a torn `lo`/`len` pair — a narrowed
    /// pair would silently skip a page. `len == 0` means no code has been marked, and
    /// `(page - 0) < 0` is false for every page, so an empty range needs no special case.
    ///
    /// Degraded source ordering returns an always-open range covering all valid
    /// RAM, forcing every compiled store through the fenced notifier (§10).
    pub fn code_range_ptr(&self) -> u64 {
        // All valid RAM lies below 2^47. Keeping a separate always-open JIT gate
        // avoids inflating the precise extent used by sparse metadata walks.
        static FENCED_STORES: AtomicU64 = AtomicU64::new(1 << (47 - CODE_RANGE_BITS));
        let range = if self.source_barrier.mode == crate::SourceBarrierMode::Degraded {
            &FENCED_STORES
        } else {
            &self.code_range
        };
        range as *const AtomicU64 as u64
    }

    fn mark_fetch_code(&self, page: u64, live: bool) {
        let mut aliases = self.code_aliases.lock().unwrap();
        if live {
            aliases.fetching.insert(page);
        } else {
            aliases.fetching.remove(&page);
        }
        self.refresh_code_page(&aliases, page);
    }

    pub(crate) fn release_code_registration(&self, page: u64) {
        let mut aliases = self.code_aliases.lock().unwrap();
        aliases.translated.remove(&page);
        self.refresh_code_page(&aliases, page);
    }

    /// Clear a page's code tag after its blocks have been invalidated (§10).
    pub fn clear_code_page(&self, page: u64) {
        let clear = || {
            let mut aliases = self.code_aliases.lock().unwrap();
            aliases.translated.remove(&page);
            aliases.fetching.remove(&page);
            self.refresh_code_page(&aliases, page);
        };
        if let Some(policy) = &self.fetch_policy {
            policy.clear(Some(page), self.guest_base >> CODE_PAGE_BITS, clear);
        } else {
            clear();
        }
    }

    /// Clear every code-page tag — for a whole-cache invalidation (e.g. mapping a
    /// Trap region, §5.2 M4-T10), the bulk counterpart of [`Self::clear_code_page`].
    pub fn clear_all_code_pages(&self) {
        let clear = || {
            let mut aliases = self.code_aliases.lock().unwrap();
            aliases.translated.clear();
            aliases.fetching.clear();
            let range = self.code_range.load(Ordering::Relaxed);
            let (lo, len) = (range >> 32, range & 0xffff_ffff);
            if len == 0 {
                return;
            }
            let base_page = self.guest_base >> CODE_PAGE_BITS;
            let first_page = (lo << (CODE_RANGE_BITS - CODE_PAGE_BITS)).max(base_page);
            let end_page = (lo + len) << (CODE_RANGE_BITS - CODE_PAGE_BITS);
            let first_word = ((first_page - base_page) >> 6) as usize;
            let end_word = end_page.saturating_sub(base_page).div_ceil(64) as usize;
            // An empty watermark touches nothing. Otherwise only words in the code
            // extent are written, preserving lazy commit across a sparse guest span.
            for (i, word) in self
                .code_page
                .iter()
                .enumerate()
                .take(end_word)
                .skip(first_word)
            {
                let first = base_page + (i as u64 * 64);
                let retained = aliases
                    .retained_code
                    .range(first..first + 64)
                    .fold(0, |mask, &page| mask | (1 << (page - first)));
                word.fetch_and(retained, Ordering::Relaxed);
            }
            for &page in aliases.lifting.keys() {
                self.refresh_code_page(&aliases, page);
            }
        };
        if let Some(policy) = &self.fetch_policy {
            policy.clear(None, self.guest_base >> CODE_PAGE_BITS, clear);
        } else {
            clear();
        }
    }

    /// Note a store of `size` bytes at `addr`: if it lands on a code page, record
    /// the written byte ranges for the dispatcher to invalidate (§10). The common
    /// case (a non-code page) costs one relaxed atomic load and returns.
    ///
    /// `pub` because generated code reaches it too, through the helper the inlined
    /// store calls once its watermark test says the store *might* have hit a code or
    /// watched page. Both halves are precise here; the inline test is not, and is not
    /// meant to be.
    pub fn note_write(&self, addr: u64, len: usize) {
        self.note_write_observed(addr, len, self);
    }

    #[inline(always)]
    fn note_write_observed(&self, addr: u64, len: usize, observer: &impl InterpreterMemory) {
        self.source_barrier.writer_fence();
        let host = (self.host_base() as u64).wrapping_add(addr.wrapping_sub(self.guest_base));
        observer.note_host_block_write(host, addr, len);
        // One relaxed load gates the (rare) data-watch path: an unwatched memory pays
        // nothing beyond this branch.
        let watched = self.watch.count.load(Ordering::Relaxed) != 0;
        for page in code_page_range(addr, len as u64) {
            if self.is_code_page(page) {
                return self.note_code_write_range(addr, len, observer);
            }
            if watched && self.watch.is_watched(page) {
                self.watch.mark_dirty(page);
            }
        }
    }

    #[cold]
    #[inline(never)]
    fn note_code_write_range(&self, addr: u64, len: usize, observer: &impl InterpreterMemory) {
        // Finish the whole range here so the non-code path need not preserve
        // addr/len across a cold call. Repeated watch marks are idempotent.
        let watched = self.watch.count.load(Ordering::Relaxed) != 0;
        for page in code_page_range(addr, len as u64) {
            if self.is_code_page(page) {
                let lo = addr.max(page << CODE_PAGE_BITS);
                let hi = addr
                    .saturating_add(len.max(1) as u64)
                    .min((page + 1) << CODE_PAGE_BITS);
                self.note_code_write(lo, hi, observer);
            }
            if watched && self.watch.is_watched(page) {
                self.watch.mark_dirty(page);
            }
        }
    }

    // Keep alias expansion, allocation and notification locks out of data stores'
    // frame and register allocation, even when the code-page branch is never taken.
    #[cold]
    #[inline(never)]
    fn note_code_write(&self, lo: u64, hi: u64, observer: &impl InterpreterMemory) {
        // Alias expansion and dirty-range growth reach the embedder's allocator.
        let _fp = crate::jit_abi::fp::DefaultEnvironment::enter();
        let aliases = self.code_aliases.lock().unwrap();
        let mut dirty = aliases.written_ranges(lo, hi);
        dirty.retain(|&(lo, hi)| {
            observer.note_block_write(lo, (hi - lo) as usize);
            aliases.translated.contains(&(lo >> CODE_PAGE_BITS))
                || aliases.lifting.contains_key(&(lo >> CODE_PAGE_BITS))
                || aliases.fetching.contains(&(lo >> CODE_PAGE_BITS))
        });
        drop(aliases);
        if dirty.is_empty() {
            return;
        }
        // Serialize notification and clearing with the drain. Exit bits in
        // these words are independent and must survive every SMC update.
        let mut words = self.dispatch_words.lock().unwrap();
        self.dirty.lock().unwrap().extend(dirty);
        self.dirty_flag.store(true, Ordering::Relaxed);
        words.retain(|weak| {
            if let Some(word) = weak.upgrade() {
                if word.load(Ordering::Relaxed) & DISPATCH_DIRTY == 0 {
                    word.fetch_or(DISPATCH_DIRTY, Ordering::Relaxed);
                }
                true
            } else {
                false
            }
        });
    }

    /// This instance's identity among the process's Memory instances, never
    /// reused: the one a dispatch word was registered with (`Vcpu::run`).
    #[inline(always)]
    pub(crate) fn dispatch_identity(&self) -> u64 {
        self.dispatch_identity
    }

    pub(crate) fn register_dispatch_word(&self, word: &Arc<AtomicU64>) {
        let mut words = self.dispatch_words.lock().unwrap();
        self.child_dispatch.register(word);
        // register's SeqCst fence precedes this pending read, paired with the
        // child's dirty-store/fence/link-read handshake (decision-19).
        words.retain(|weak| weak.strong_count() != 0);
        if self.has_dirty_code() {
            word.fetch_or(DISPATCH_DIRTY, Ordering::Relaxed);
        }
        words.push(Arc::downgrade(word));
    }

    /// Whether any code page has been written since the last `take_dirty_code` — the
    /// cheap global SMC test. Compiled chain edges poll their per-vcpu notification
    /// word, which folds this condition with exit requests. Atomic polls; false
    /// for every guest that does not modify its own code.
    pub fn has_dirty_code(&self) -> bool {
        self.dirty_flag.load(Ordering::Relaxed) || self.child_dispatch.dirty.load(Ordering::Acquire)
    }

    /// Whether any code alias is declared ([`Self::declare_code_alias`]).
    #[inline(always)]
    pub(crate) fn has_code_aliases(&self) -> bool {
        self.code_aliases_declared.load(Ordering::Relaxed)
    }

    /// Every declared view of the written bytes `addr..addr + len` on a code
    /// page, the written range included, as `(lo, hi)` ranges; `visit` sees
    /// each. A byte on a page with no code tag is not expanded: an alias of
    /// live code shares its tag (`refresh_code_aliases`).
    #[cold]
    #[inline(never)]
    pub(crate) fn visit_code_alias_views(
        &self,
        addr: u64,
        len: usize,
        mut visit: impl FnMut(u64, u64),
    ) {
        let end = addr.saturating_add(len.max(1) as u64);
        let mut aliases = None;
        for page in code_page_range(addr, len as u64) {
            if !self.is_code_page(page) {
                continue;
            }
            let lo = addr.max(page << CODE_PAGE_BITS);
            let hi = end.min((page + 1) << CODE_PAGE_BITS);
            let aliases = aliases.get_or_insert_with(|| self.code_aliases.lock().unwrap());
            for (lo, hi) in aliases.written_ranges(lo, hi) {
                visit(lo, hi);
            }
        }
    }

    /// Whether page `p` currently backs translated code — the precise test behind the
    /// inline watermark, for callers that already know the exact page.
    pub fn is_code_page(&self, p: u64) -> bool {
        let (w, m) = word_bit(p.wrapping_sub(self.guest_base >> CODE_PAGE_BITS));
        self.code_page
            .get(w)
            .is_some_and(|word| word.load(Ordering::Relaxed) & m != 0)
    }

    /// Address of the live `watch_count` atomic, stored into `MemCtx` at run start so the
    /// JIT's inlined store gate loads the count **live** through it — not a run-start
    /// snapshot. A snapshot misses the 0→nonzero transition when another thread installs
    /// the first watch while this vCPU is mid-run in JIT'd code; a live load through this
    /// pointer sees the new count on the next store (coherence). The pointer is stable
    /// for the run: `Memory` is heap-pinned behind the embedder's `Arc`. Reading it costs
    /// one extra L1-cached load on the store fast path (the atomic is uncontended and
    /// shared-clean while unwatched), so an unwatched memory still pays ~nothing.
    pub fn watch_count_ptr(&self) -> u64 {
        &self.watch.count as *const AtomicUsize as u64
    }

    /// Base of the per-page watch BITMAP, so generated stores can test the bit for
    /// their own page inline instead of calling [`Self::note_write`] only to discover
    /// the page is not watched.
    ///
    /// Subtract `guest_base >> CODE_PAGE_BITS` from guest page `p`, then the word
    /// is at `base + (p >> 6) * 8`, bit `p & 63` — the layout `word_bit` uses.
    /// Read live, like [`Self::watch_count_ptr`]: a watch installed by another
    /// thread mid-run must be visible to the next store.
    ///
    /// # Safety contract for the caller
    /// Generated code indexes this WITHOUT a bounds check, so the table must cover
    /// every address a store can reach. It does: `WatchPages::new` is sized over
    /// the pages intersecting `[guest_base, size)`, while inlined stores check both
    /// bounds first. The first page may be partially backed.
    /// [`Self::watch_bits_cover_size`] re-checks that chain at run start.
    pub fn watch_bits_ptr(&self) -> u64 {
        self.watch.watch.as_ptr() as u64
    }

    /// Whether the watch bitmap covers every page an inlined, bounds-checked store can
    /// reach. Asserted where the JIT is handed [`Self::watch_bits_ptr`]: if
    /// this ever goes false the generated bit test would read out of bounds, which is
    /// host UB rather than a panic, so it must fail loudly instead.
    pub fn watch_bits_cover_size(&self) -> bool {
        if self.size() == self.guest_base {
            return true;
        }
        let top_page = (self.size() - 1) >> CODE_PAGE_BITS;
        let relative = top_page - (self.guest_base >> CODE_PAGE_BITS);
        (relative >> 6) < self.watch.watch.len() as u64
    }

    /// Register `[addr, addr + size)` as a watched DATA range: subsequent guest writes
    /// to any page it spans are recorded and drained by [`Self::take_dirty_ranges`].
    /// Independent of the SMC code-page path — a watched page need not be code.
    /// Idempotent per page (re-watching a page is a no-op). Registration works at ANY
    /// guest address the embedder can map.
    pub fn watch_range(&self, addr: u64, size: u64) {
        let last = addr.saturating_add(size.max(1) - 1);
        for page in (addr >> CODE_PAGE_BITS)..=(last >> CODE_PAGE_BITS) {
            self.watch.watch(page);
        }
    }

    /// Stop watching `[addr, addr + size)`. Symmetric to [`Self::watch_range`];
    /// when the last watched page is cleared the write-path check turns off again.
    pub fn unwatch_range(&self, addr: u64, size: u64) {
        let last = addr.saturating_add(size.max(1) - 1);
        for page in (addr >> CODE_PAGE_BITS)..=(last >> CODE_PAGE_BITS) {
            self.watch.unwatch(page);
        }
    }

    /// Drain the watched pages written since the last call, coalesced into
    /// `(guest_addr, byte_len)` ranges. Empty and lock-free in the common
    /// case (nothing watched was written). Intended for poll-and-drain at a frame /
    /// submit boundary; needs no ordering beyond `MemConsistency::Fast`.
    pub fn take_dirty_ranges(&self) -> Vec<(u64, u64)> {
        // Already ascending and duplicate-free: the scan walks the watched words in
        // order and the bits within each word low-to-high.
        let pages = self.watch.take_dirty();
        // Coalesce runs of consecutive pages into a single range.
        let mut out: Vec<(u64, u64)> = Vec::new();
        for p in pages {
            let start = p << CODE_PAGE_BITS;
            if let Some(last) = out.last_mut() {
                if start == last.0 + last.1 {
                    last.1 += CODE_PAGE_SIZE;
                    continue;
                }
            }
            out.push((start, CODE_PAGE_SIZE));
        }
        out
    }

    /// Drain the set of code pages written since the last call (§10). Empty and
    /// lock-free in the common case (nothing self-modified).
    pub fn take_dirty_code(&self) -> Vec<u64> {
        // Common case (nothing self-modified): shared loads, not a swap — the
        // latter takes the cache line exclusive on every dispatch even when clean.
        if !self.has_dirty_code() {
            return Vec::new();
        }
        let _words = self.dispatch_words.lock().unwrap();
        self.take_dirty_code_locked()
            .into_iter()
            .map(|(lo, _)| lo >> CODE_PAGE_BITS)
            .collect()
    }

    /// Serialize invalidation and this vcpu's acknowledgement with notifications
    /// and other dispatchers. A running vcpu must retain its own notice until it
    /// reaches dispatch, even if another vcpu already drained the global list.
    pub(crate) fn dispatch_smc(&self, word: &AtomicU64, invalidate: impl FnOnce(Vec<(u64, u64)>)) {
        let notices = DISPATCH_DIRTY | crate::child::DISPATCH_CHILD_DIRTY;
        if word.load(Ordering::Acquire) & notices == 0 && !self.has_dirty_code() {
            return;
        }
        let _words = self.dispatch_words.lock().unwrap();
        let child_epoch = self.child_dispatch.epoch(word);
        let before = child_epoch.map(|epoch| epoch.load(Ordering::Acquire));
        invalidate(self.take_dirty_code_locked());
        // The lock makes prior invalidation visible before acknowledgement and
        // serializes ordinary writers; child writers use the epoch check below.
        word.fetch_and(!notices, Ordering::AcqRel);
        // Children cannot take dispatch_words. If one notified while we drained,
        // retain a notice even when its fetch_or preceded our acknowledgement.
        // Assumes fewer than 2^64 notifications to this node between epoch
        // samples: a single wrap differs, but a full modulus restores equality.
        if child_epoch.map(|epoch| epoch.load(Ordering::Acquire)) != before {
            word.fetch_or(crate::child::DISPATCH_CHILD_DIRTY, Ordering::Release);
        }
    }

    // Caller holds dispatch_words through any invalidation and acknowledgement.
    fn take_dirty_code_locked(&self) -> Vec<(u64, u64)> {
        // Something's dirty: claim it. A racing vcpu may have drained it already.
        let mut ranges = if self.dirty_flag.swap(false, Ordering::Relaxed) {
            std::mem::take(&mut *self.dirty.lock().unwrap())
        } else {
            Vec::new()
        };
        if self.child_dispatch.dirty.swap(false, Ordering::AcqRel) {
            let aliases = self.code_aliases.lock().unwrap();
            let mut journals = self.child_writes.lock().unwrap();
            journals.retain(|journal| {
                if let Some(writes) = journal.upgrade() {
                    writes.drain(|page| {
                        let lo = page << CODE_PAGE_BITS;
                        ranges.extend(aliases.written_ranges_with_retired(
                            lo,
                            lo + CODE_PAGE_SIZE,
                            &aliases.retired_aliases,
                        ));
                    });
                    true
                } else {
                    false
                }
            });
        }
        ranges
    }

    /// Highest mapped guest address (exclusive end of the top region) at or below
    /// `limit`, or 0 if nothing is mapped there. Lets an embedder place the heap just
    /// above a loaded image's segments instead of at a fixed guess. A region that
    /// *straddles* `limit` (starts below it but extends past) has its end **clamped to
    /// `limit`** — otherwise the result would exceed `limit` and the caller would place
    /// the heap past the boundary it asked to stay under.
    pub fn highest_mapped_below(&self, limit: u64) -> u64 {
        self.regions
            .iter()
            .filter(|r| r.start < limit)
            .map(|r| (r.start + r.size as u64).min(limit))
            .max()
            .unwrap_or(0)
    }

    /// The mapped region wholly containing `[addr, addr + len)`, if any.
    fn region_for(&self, addr: u64, len: usize) -> Option<&Region> {
        let end = addr.checked_add(len as u64)?;
        self.regions
            .iter()
            .find(|r| r.start <= addr && end <= r.start + r.size as u64)
    }

    /// Base pointer of the guest RAM buffer — the host address of guest `guest_base`
    /// (the JIT inlines `host_base + (addr - guest_base)`).
    pub fn host_base(&self) -> *const u8 {
        // SAFETY: pointer to the backing buffer's start; callers bounds-check.
        unsafe { (*self.backing.get()).as_ptr() }
    }

    /// Guest address the backing buffer's first byte represents (§4.1). The guest
    /// space is `[guest_base, size())`. `0` is the historical zero-based layout; an
    /// embedder sets it (via `HostRam.guest_base`) for identity mapping.
    pub fn guest_base(&self) -> u64 {
        self.guest_base
    }

    /// Host backing pointer for a guest range `[addr, addr+len)`, but **only** when the
    /// range lies wholly inside a mapped **RAM** region *and* the backing is a real host
    /// mapping (the `MAP_NORESERVE` `Reserved` path) — not an owned `Box`/`Vec`.
    ///
    /// Returns `Some(host_ptr)` for such a range, `None` otherwise (Vec-backed backing,
    /// unmapped range, or a `Trap`/MMIO region). The one intended caller is the embedder's
    /// `madvise(MADV_DONTNEED)` passthrough: with a real host mapping it can
    /// `libc::madvise` the returned host span to release physical pages to the OS. A `None`
    /// return tells the embedder to fall back to zeroing the guest bytes itself.
    ///
    /// This does **not** touch the JIT's `host_base` translation — it's a read-only
    /// pointer computation over the same `ptr + (addr - guest_base)` arithmetic, gated on
    /// region membership and backing kind. `len == 0` still validates the region so a
    /// zero-length range in RAM returns `Some`.
    pub fn host_ram_ptr(&self, addr: u64, len: usize) -> Option<*mut u8> {
        match self.region_for(addr, len) {
            Some(r) if matches!(r.kind, RegionKind::Ram) => {
                // SAFETY: `region_for` bounds-checked `[addr, addr+len)` into a mapped RAM
                // region, hence inside the backing buffer. `is_host` gates to a real mapping.
                let backing = unsafe { &*self.backing.get() };
                if !backing.is_host() {
                    return None;
                }
                let off = self.host_off(addr);
                // SAFETY: `off <= backing.len` (bounds-checked above); forming an in-bounds
                // (or one-past-the-end when off == len) pointer into the mapping.
                Some(unsafe { backing.as_ptr().add(off) as *mut u8 })
            }
            _ => None,
        }
    }

    /// Backing-buffer index for a guest address (§4.1): `addr - guest_base`, done in
    /// integers so a null-adjacent host pointer is never materialized. Callers must have
    /// established `addr >= guest_base` (via a region tag or a bounds check); below-base
    /// addresses never reach a mapped region, and the JIT's bounds check rejects them.
    #[inline]
    fn host_off(&self, addr: u64) -> usize {
        (addr - self.guest_base) as usize
    }

    /// Size of the backing buffer in bytes — the bound the JIT checks a guest
    /// address against before an inlined access (§8.2.3). The buffer is allocated
    /// from this value, so it equals `backing.len()`. For `Reserved` it is the full
    /// (sparse) span, so the one-add translation reaches the whole reserved range.
    pub fn size(&self) -> u64 {
        match self.model {
            MemoryModel::Flat { size } => size,
            MemoryModel::Reserved { span } => span,
            MemoryModel::SoftMmu => 0,
        }
    }

    /// The `[lo, hi)` guest-address span enclosing every `Trap` (MMIO) region, or
    /// `None` if there are none (§5.2, M4-T10). The JIT bakes this window as a
    /// compile-time constant and, when present, adds a range check that defers an
    /// inlined access to the interpreter. It is the *bounding* window: an address in
    /// a RAM gap between two Trap regions is conservatively deferred too — correct
    /// (the interpreter handles RAM), only slightly slower. `None` means zero
    /// per-access cost, the common case. Sparse VMs execute only checked
    /// interpreter code (§4.1).
    pub fn trap_window(&self) -> Option<(u64, u64)> {
        Some((
            self.trap_intervals.first()?.0,
            self.trap_intervals.last()?.1,
        ))
    }

    /// Whether the inclusive, non-wrapping byte range intersects a device.
    /// O(log T) comparisons over T sorted Trap intervals; no allocation or lock.
    pub fn intersects_trap(&self, first: u64, last: u64) -> bool {
        debug_assert!(first <= last);
        if self.trap_intervals.is_empty() {
            return false;
        }
        let i = self
            .trap_intervals
            .partition_point(|&(_, end)| end <= first);
        self.trap_intervals
            .get(i)
            .is_some_and(|&(start, _)| start <= last)
    }

    /// Reserve a region. In `Flat` this only tags `[guest_addr, guest_addr+size)`
    /// with prot/kind and bounds-checks it against the backing buffer — it does
    /// NOT allocate (`map(high_addr)` in Flat is a tag, not a 128 TB alloc). (§4.1)
    pub fn map(
        &mut self,
        guest_addr: u64,
        size: usize,
        prot: Prot,
        kind: RegionKind,
    ) -> Result<(), MapError> {
        if size != 0
            && (!is_canonical(guest_addr)
                || !guest_addr
                    .checked_add(size as u64 - 1)
                    .is_some_and(is_canonical)
                || (guest_addr < 1 << 47 && guest_addr.saturating_add(size as u64) > 1 << 47))
        {
            return Err(MapError::OutOfBounds);
        }
        match self.model {
            // `Reserved` tags and bounds-checks exactly like `Flat`; it only differs
            // in how the backing bytes are allocated (a sparse mmap vs a `Vec`).
            MemoryModel::Flat { size: total } | MemoryModel::Reserved { span: total } => {
                // The guest space is `[guest_base, total)`; a region below the base has
                // no backing (its bytes live before `ptr[0]`), so reject it — this is the
                // "reject map below guest_base" guard for identity mapping.
                if guest_addr < self.guest_base {
                    return Err(MapError::OutOfBounds);
                }
                let end = guest_addr
                    .checked_add(size as u64)
                    .ok_or(MapError::OutOfBounds)?;
                if end > total {
                    return Err(MapError::OutOfBounds);
                }
                let overlaps = self
                    .regions
                    .iter()
                    .any(|r| guest_addr < r.start + r.size as u64 && r.start < end);
                if overlaps {
                    return Err(MapError::Overlap);
                }
                // Commit before publishing a tag: every tagged byte must be
                // accessible, including host inspection of Trap regions.
                if let Owner::Sparse(reservation) = &unsafe { &*self.backing.get() }.owner {
                    self.regions
                        .try_reserve(1)
                        .map_err(|_| MapError::Sparse(SparseError::Allocation))?;
                    if kind == RegionKind::Trap && size != 0 {
                        self.trap_intervals
                            .try_reserve(1)
                            .map_err(|_| MapError::Sparse(SparseError::Allocation))?;
                    }
                    reservation
                        .commit(guest_addr as usize, size)
                        .map_err(MapError::Sparse)?;
                    // Shared edge pages can retain bytes from an earlier mapping.
                    // Zero only this region; never walk the uncommitted span.
                    unsafe {
                        std::ptr::write_bytes(reservation.ptr.add(guest_addr as usize), 0, size)
                    };
                }
                self.mapping_identity = Arc::new(());
                self.regions.push(Region {
                    start: guest_addr,
                    size,
                    prot,
                    kind,
                });
                if kind == RegionKind::Trap && size != 0 {
                    let i = self
                        .trap_intervals
                        .partition_point(|&(start, _)| start < guest_addr);
                    self.trap_intervals.insert(i, (guest_addr, end));
                }
                // Guard pages: open the region's host pages. No-op unless the backing
                // installed a protect hook.
                self.reprotect(guest_addr, size, true);
                Ok(())
            }
            MemoryModel::SoftMmu => todo!("SoftMmu: allocate pages for the region (§4.1)"),
        }
    }

    /// Load bytes into an already-mapped region (e.g. an ELF segment). Host-side
    /// loader path: it bypasses guest `Prot` (you write code into an RX region),
    /// so it only checks that the range is mapped. (§4.2)
    pub fn write_bytes(&self, guest_addr: u64, bytes: &[u8]) -> Result<(), MemError> {
        if self.region_for(guest_addr, bytes.len()).is_none() {
            return Err(MemError::Unmapped);
        }
        let start = self.host_off(guest_addr);
        // `&self`, interior-mutable (§8), mirroring `write`/`write_ram_guest`: the guest
        // memory model is shared-mutable so the syscall shim can write results through an
        // `Arc<Vm>` on a worker thread (M7 threaded embedder). The range sits inside a
        // mapped region `map()` bounds-checked.
        // SAFETY: the range is bounds-checked into a mapped region of the backing.
        let result = unsafe { self.copy_to_guest(self.host_base().cast_mut().add(start), bytes) };
        let committed = result
            .as_ref()
            .map_or_else(|fault| fault.committed, |()| bytes.len());
        // SMC: an embedder write (loader, syscall passthrough) over a code page
        // must invalidate too (§10).
        if committed != 0 {
            self.note_write(guest_addr, committed);
        }
        result.map_err(|_| MemError::Unmapped)
    }

    /// Guest-side wide read for the x87 helpers (`&self`, interior-mutable model):
    /// copy `buf.len()` bytes out of a mapped **RAM** region. `false` if the range
    /// escapes RAM (unmapped, or a `Trap`/MMIO region) — the interpreter turns that
    /// into a fault, matching a scalar `read`. Used for f64/f80/fxsave loads whose
    /// width exceeds the 8-byte scalar path.
    pub fn read_ram_guest(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.read_ram_checked(addr, buf).is_ok()
    }

    /// Guest-side wide write for the x87 helpers: copy `bytes` into a mapped **RAM**
    /// region and record the SMC `note_write` (§10) — so a self-modifying x87 store
    /// onto a code page invalidates, exactly like a scalar `Store`. `false` if the
    /// range escapes RAM. An x87 store into a `Trap` region can't be expressed as
    /// `Exit::MmioWrite` (its value exceeds 8 bytes), so it faults here rather than
    /// silently scribbling backing — MMIO for x87 stays deferred (§5.2, §10).
    pub fn write_ram_guest(&self, addr: u64, bytes: &[u8]) -> bool {
        self.write_ram_checked(addr, bytes).is_ok()
    }

    /// Read bytes back out (inspection / HLE reading guest structures). (§4.2)
    pub fn read_bytes(&self, guest_addr: u64, buf: &mut [u8]) -> Result<(), MemError> {
        if self.region_for(guest_addr, buf.len()).is_none() {
            return Err(MemError::Unmapped);
        }
        let start = self.host_off(guest_addr);
        // SAFETY: the range lies inside a mapped, bounds-checked region; this is a
        // host-side read with no concurrent guest store to the same bytes.
        unsafe { self.copy_from_guest(self.host_base().add(start), buf) }
            .map_err(|_| MemError::Unmapped)?;
        Ok(())
    }

    /// Drop a region's tag. With ordinary `Flat` backing the buffer is untouched — only the
    /// permission/kind tag goes away. Partial unmap isn't modeled in M0, so the
    /// `(guest_addr, size)` must match a mapped region exactly. Sparse backing
    /// discards unshared host pages; on a host decommit error the tag is still
    /// removed and the reservation retained. (§4.1)
    pub fn unmap(&mut self, guest_addr: u64, size: usize) -> Result<(), MapError> {
        match self
            .regions
            .iter()
            .position(|r| r.start == guest_addr && r.size == size)
        {
            Some(pos) => {
                self.mapping_identity = Arc::new(());
                let region = self.regions.remove(pos);
                if region.kind == RegionKind::Trap && size != 0 {
                    let i = self
                        .trap_intervals
                        .partition_point(|&(start, _)| start < guest_addr);
                    self.trap_intervals.remove(i);
                }
                let end = guest_addr.saturating_add(size as u64);
                let pages = code_page_range(guest_addr, size as u64);
                let removed: BTreeSet<_> = self
                    .code_aliases
                    .lock()
                    .unwrap()
                    .registrations
                    .range(pages)
                    .flat_map(|(_, pairs)| pairs)
                    .copied()
                    .filter(|&(a, b, len)| {
                        [a, b]
                            .into_iter()
                            .any(|start| start < end && guest_addr < start + len as u64)
                    })
                    .collect();
                for (a, b, len) in removed {
                    self.remove_code_alias(a, b, len);
                }
                if self.fetch_policy.is_some() {
                    for page in code_page_range(guest_addr, size as u64) {
                        self.clear_code_page(page);
                    }
                }
                // Guard pages: close the region's host pages, except any page still
                // touched by a surviving region.
                self.reprotect(guest_addr, size, false);
                self.decommit_sparse(guest_addr, size)?;
                Ok(())
            }
            None => Err(MapError::OutOfBounds),
        }
    }

    fn decommit_sparse(&self, start: u64, size: usize) -> Result<(), MapError> {
        let Owner::Sparse(reservation) = &unsafe { &*self.backing.get() }.owner else {
            return Ok(());
        };
        if size == 0 {
            return Ok(());
        }
        let page = reservation.page as u64;
        let mut lo = start / page * page;
        let mut hi = (start + size as u64).div_ceil(page) * page;
        let shared = |first, last| {
            self.regions
                .iter()
                .any(|r| r.size != 0 && r.start < last && first < r.start + r.size as u64)
        };
        // Non-overlap of guest regions means only the two edge pages can be shared.
        if shared(lo, lo + page) {
            lo += page;
        }
        if lo < hi && shared(hi - page, hi) {
            hi -= page;
        }
        if lo < hi {
            reservation
                .decommit(lo as usize, (hi - lo) as usize)
                .map_err(MapError::Sparse)?;
        }
        Ok(())
    }

    /// Flip host guard-page protection for the pages `[start, start+size)` touches.
    /// On map (`accessible = true`) the region's pages, **rounded
    /// outward**, become `PROT_READ|PROT_WRITE`. On unmap (`false`) they become
    /// `PROT_NONE`, **except** a boundary page still overlapped by a surviving region
    /// (a page shared with a live neighbor stays accessible). A no-op unless the
    /// backing is a host mapping with a protect hook. Ranges are clamped to the backing.
    fn reprotect(&self, start: u64, size: usize, accessible: bool) {
        let backing = unsafe { &*self.backing.get() };
        let cap = backing.len() as u64;
        // Work in backing-offset space: the protect hook `mprotect`s the mapping, whose
        // first byte is guest `guest_base`. A region below the base never maps (rejected
        // in `map`), so `start >= guest_base` holds here.
        let base = self.guest_base;
        let ostart = start.saturating_sub(base);
        let oend = start
            .saturating_add(size as u64)
            .saturating_sub(base)
            .min(cap);
        let lo = (ostart / HOST_PAGE) * HOST_PAGE;
        let hi = oend.div_ceil(HOST_PAGE) * HOST_PAGE;
        if lo >= hi {
            return;
        }
        if accessible {
            backing.protect(lo as usize, (hi.min(cap) - lo) as usize, true);
            return;
        }
        // Inaccessible: page by page (backing offsets), skipping any page a surviving
        // region overlaps. Region bounds are converted to offsets for the comparison.
        let mut page = lo;
        while page < hi {
            let pend = (page + HOST_PAGE).min(cap);
            let shared = self.regions.iter().any(|r| {
                let rlo = r.start.saturating_sub(base);
                let rhi = (r.start + r.size as u64).saturating_sub(base);
                rlo < pend && page < rhi
            });
            if !shared && page < pend {
                backing.protect(page as usize, (pend - page) as usize, false);
            }
            page += HOST_PAGE;
        }
    }

    /// Check a WRITE of `size` bytes at `addr` without consuming an MMIO answer.
    /// In default mode this only checks region metadata. Recovery mode probes
    /// actual access; only a denied read takes the cold VM-permission fallback.
    /// Callers must skip host probes for a single-page destination and keep
    /// protections stable until the store commits.
    ///
    /// The wide-store helpers pre-probe their whole destination so a fault leaves nothing
    /// committed. Probing with a load instead — on the reasoning that a load shares the
    /// region and `Trap` checks with a store — is not sufficient once the MMIO answer
    /// channels split: a load probe consumes `pending_mmio`, while the embedder answers a
    /// `MmioWrite` through `pending_mmio_write`, so the retry re-probes forever.
    /// Reporting the region's verdict without consuming anything is what the probe wants.
    pub fn probe_write(&self, addr: u64, size: u8) -> Result<(), MemTrap> {
        if self.recover_host_faults {
            self.probe_write_spanning::<true>(addr, size)
        } else {
            self.probe_write_spanning::<false>(addr, size)
        }
    }

    /// Containment only: a span that no single region contains is `Unmapped`.
    /// [`Self::probe_write_spanning`] also accepts a span over adjacent regions.
    #[inline]
    pub(crate) fn probe_write_mode<const ROUTED: bool>(
        &self,
        addr: u64,
        size: u8,
    ) -> Result<(), MemTrap> {
        match self.region_at(addr, size)?.kind {
            RegionKind::Trap => Err(MemTrap::Mmio),
            RegionKind::Rom => Ok(()),
            RegionKind::Ram if !ROUTED => Ok(()),
            RegionKind::Ram => {
                // SAFETY: region_at validated the entire range.
                unsafe {
                    crate::usercopy::probe(
                        self.host_base().cast_mut().add(self.host_off(addr)),
                        size as usize,
                        true,
                    )
                }
                .map_err(|at| self.host_fault(at))
            }
        }
    }

    /// [`Self::probe_write_mode`] whose containment miss probes a span over adjacent
    /// RAM/ROM regions ([`Self::probe_write_across`]). Only `region_at` reports
    /// `Unmapped` there; a host fault is `HostFault`.
    #[inline]
    pub(crate) fn probe_write_spanning<const ROUTED: bool>(
        &self,
        addr: u64,
        size: u8,
    ) -> Result<(), MemTrap> {
        match self.probe_write_mode::<ROUTED>(addr, size) {
            Err(miss @ MemTrap::Unmapped) if size > 1 => {
                self.probe_write_across(addr, size, miss, ROUTED)
            }
            result => result,
        }
    }

    /// Whether the region holding `addr` is RAM or ROM and the byte after its end
    /// lies in another RAM or ROM region. Every transfer whose span
    /// [`Self::adjacent_span`] accepts starts in such a region, so a containment miss
    /// at its base passes; a gap or a Trap byte at `addr` or at that end does not.
    #[cold]
    #[inline(never)]
    pub(crate) fn continues_into_adjacent(&self, addr: u64) -> bool {
        let data = |at: u64| {
            self.regions
                .iter()
                .find(|r| r.start <= at && at - r.start < r.size as u64)
                .filter(|r| r.kind != RegionKind::Trap)
        };
        data(addr)
            .and_then(|r| r.start.checked_add(r.size as u64))
            .is_some_and(|end| data(end).is_some())
    }

    /// Kinds of the adjacent regions covering `[addr, addr + len)` when no single
    /// region contains it: `Some(true)` when some byte is ROM, `Some(false)` when
    /// every byte is RAM, and `None` when a byte is unmapped (a gap) or lies in a
    /// Trap region. Only `None` keeps the containment miss (§4.2).
    ///
    /// SDM Vol 1 §4.1.1: an unaligned operand can need more than one bus cycle, so
    /// one guest operand can cover bytes of two regions. Every region tags a slice of
    /// the one contiguous backing (`Flat` and `Reserved` alike; `map` never gives a
    /// region host storage of its own), so such a span is still one contiguous host
    /// range at `host_base + (addr - guest_base)`.
    #[cold]
    #[inline(never)]
    fn adjacent_span(&self, addr: u64, len: usize) -> Option<bool> {
        if len == 0 {
            return None;
        }
        let end = addr.checked_add(len as u64)?;
        let mut at = addr;
        let mut rom = false;
        while at < end {
            let region = self
                .regions
                .iter()
                .find(|r| r.start <= at && at - r.start < r.size as u64)?;
            match region.kind {
                RegionKind::Trap => return None,
                RegionKind::Rom => rom = true,
                RegionKind::Ram => {}
            }
            at = end.min(region.start + region.size as u64);
        }
        Some(rom)
    }

    /// Visit the per-region pieces of a span [`Self::adjacent_span`] accepted, in
    /// ascending address order. The order is an engine rule, not a hardware claim:
    /// the SDM does not order the bus cycles of one unaligned operand.
    fn span_pieces<E>(
        &self,
        addr: u64,
        len: usize,
        mut visit: impl FnMut(u64, usize, RegionKind) -> Result<(), E>,
    ) -> Result<(), E> {
        let end = addr + len as u64;
        let mut at = addr;
        while at < end {
            let region = self
                .regions
                .iter()
                .find(|r| r.start <= at && at - r.start < r.size as u64)
                .expect("adjacent_span accepted every byte");
            let next = end.min(region.start + region.size as u64);
            visit(at, (next - at) as usize, region.kind)?;
            at = next;
        }
        Ok(())
    }

    /// Host-probe every RAM piece of an accepted span for write access, in ascending
    /// order, before any byte is stored: a protected byte faults first and nothing
    /// commits. ROM pieces are not probed; their stores are dropped, as in a
    /// contained ROM store, which never touches the host bytes either.
    fn probe_ram_pieces(&self, addr: u64, len: usize) -> Result<(), usize> {
        self.span_pieces(addr, len, |at, n, kind| {
            if kind != RegionKind::Ram {
                return Ok(());
            }
            // SAFETY: adjacent_span validated the piece inside the backing.
            unsafe {
                crate::usercopy::probe(self.host_base().cast_mut().add(self.host_off(at)), n, true)
            }
        })
    }

    // Spans over adjacent regions. The transfer paths (`read_mode`,
    // `write_mode_observed`, `probe_write_mode` and the wide checked transfers) are
    // containment-only. A containment miss reaches the functions below only one
    // level up: the `*_spanning` entry points (the public scalar and wide API and
    // the `Memory` interpreter provider, which serves the Machine bus helpers) and
    // `SplitMemory`, the provider of the interpreter's re-execution of an
    // instruction after such a miss (`interp::split_block`). Each is out of line and
    // non-generic (the recovery mode is a run-time flag; stores report SMC/watch
    // notes through a callback). Any span `adjacent_span` rejects returns the
    // containment miss unchanged. A one-byte access lies in one region or in none,
    // so the scalar `*_spanning` entry points call them only for a wider access;
    // byte-wise callers (the twenty-bit bus helpers) keep the containment-only shape.

    /// Containment miss of [`Self::read_mode`]. A span over adjacent RAM/ROM
    /// regions is one contiguous host range and ROM reads exactly like RAM, so it is
    /// read with the same single whole-width host load as a contained access.
    /// SDM Vol 3A §11.1.1: that keeps an aligned word/doubleword/quadword atomic, as
    /// on hardware and in compiled code (one host `mov` over the same bytes). A
    /// misaligned span takes the byte copy a contained misaligned access takes;
    /// Vol 3A §11.1.1 does not make accesses split across cache lines or pages
    /// atomic, and a region boundary may fall anywhere.
    #[cold]
    #[inline(never)]
    fn read_across(
        &self,
        addr: u64,
        size: u8,
        miss: MemTrap,
        routed: bool,
    ) -> Result<u64, MemTrap> {
        if self.adjacent_span(addr, usize::from(size)).is_none() {
            return Err(miss);
        }
        // SAFETY: adjacent_span validated the whole span inside the backing.
        let ptr = unsafe { (*self.backing.get()).as_ptr().add(self.host_off(addr)) };
        if routed {
            unsafe { crate::usercopy::load(ptr, size) }.map_err(|at| self.host_fault(at))
        } else {
            Ok(unsafe { atomic_load_le(ptr, size) })
        }
    }

    /// Containment miss of [`Self::probe_write_mode`]: each RAM piece is probed as a
    /// contained RAM probe is; ROM pieces report what a contained ROM probe does.
    #[cold]
    #[inline(never)]
    fn probe_write_across(
        &self,
        addr: u64,
        size: u8,
        miss: MemTrap,
        routed: bool,
    ) -> Result<(), MemTrap> {
        if self.adjacent_span(addr, usize::from(size)).is_none() {
            return Err(miss);
        }
        if !routed {
            return Ok(());
        }
        self.probe_ram_pieces(addr, usize::from(size))
            .map_err(|at| self.host_fault(at))
    }

    /// Names the observer for [`Self::write_across`].
    #[cold]
    #[inline(never)]
    fn write_across_observed<const ROUTED: bool, const PREFLIGHTED: bool>(
        &self,
        addr: u64,
        val: u64,
        size: u8,
        observer: &impl InterpreterMemory,
        miss: MemTrap,
    ) -> Result<(), MemTrap> {
        let mut note = |at, len| self.note_write_observed(at, len, observer);
        self.write_across(addr, val, size, miss, (ROUTED, PREFLIGHTED), &mut note)
    }

    /// Containment miss of [`Self::write_mode_observed`]. An all-RAM span over
    /// adjacent regions takes the same single whole-width host store as a contained
    /// access: SDM Vol 3A §11.1.1 keeps an aligned store atomic, and a misaligned
    /// one takes the contained path's byte copy. With ROM bytes in the span, those
    /// bytes are dropped as in a contained ROM store and each RAM piece is stored
    /// separately (SDM Vol 1 §4.1.1: separate bus cycles) in ascending address order
    /// (an engine rule); with host-fault recovery every RAM piece is probed first,
    /// so a protected byte faults before any byte is written.
    #[cold]
    #[inline(never)]
    fn write_across(
        &self,
        addr: u64,
        val: u64,
        size: u8,
        miss: MemTrap,
        (routed, preflighted): (bool, bool),
        note: &mut dyn FnMut(u64, usize),
    ) -> Result<(), MemTrap> {
        let Some(rom) = self.adjacent_span(addr, usize::from(size)) else {
            return Err(miss);
        };
        if rom && routed {
            self.probe_ram_pieces(addr, usize::from(size))
                .map_err(|at| self.host_fault(at))?;
        }
        let mut store = |at: u64, val: u64, size: u8, preflighted: bool| {
            // SAFETY: adjacent_span validated the piece inside the backing; with
            // recovery, `preflighted` is set only after its host probe.
            let ptr = unsafe { ((*self.backing.get()).as_ptr() as *mut u8).add(self.host_off(at)) };
            if routed {
                unsafe {
                    if preflighted {
                        crate::usercopy::store_preflighted(ptr, val, size)
                    } else {
                        crate::usercopy::store(ptr, val, size)
                    }
                }
                .map_err(|fault| {
                    if fault.committed != 0 {
                        note(at, fault.committed);
                    }
                    self.host_fault(fault.at)
                })?;
            } else {
                unsafe { atomic_store_le(ptr, val, size) };
            }
            note(at, usize::from(size));
            Ok(())
        };
        if !rom {
            return store(addr, val, size, preflighted);
        }
        self.span_pieces(addr, usize::from(size), |at, n, kind| {
            if kind != RegionKind::Ram {
                return Ok(());
            }
            store(at, val >> ((at - addr) * 8), n as u8, true)
        })
    }

    /// Containment miss of [`Self::read_ram_checked_mode`]: a span over adjacent
    /// RAM/ROM regions is one host range and is copied as a contained span is.
    #[cold]
    #[inline(never)]
    fn read_ram_across(&self, addr: u64, buf: &mut [u8], routed: bool) -> Result<(), u64> {
        if self.adjacent_span(addr, buf.len()).is_none() {
            return Err(crate::fault::record_fp_fault(
                addr,
                buf.len(),
                crate::FaultOrigin::GuestMap,
                crate::AddressPrecision::TransferAddress,
            ));
        }
        // SAFETY: adjacent_span validated the whole span; the local output is live.
        unsafe {
            let ptr = self.host_base().add(self.host_off(addr));
            if routed {
                crate::usercopy::copy_in(ptr, buf)
            } else {
                std::ptr::copy_nonoverlapping(ptr, buf.as_mut_ptr(), buf.len());
                Ok(())
            }
        }
        .map_err(|at| {
            let addr = self.guest_base + (at - self.host_base() as usize) as u64;
            record_host_fault(addr);
            crate::fault::record_fp_fault(
                addr,
                buf.len(),
                crate::FaultOrigin::HostAccess,
                crate::AddressPrecision::ExactByte,
            )
        })
    }

    /// Names the observer for [`Self::write_ram_across`].
    #[cold]
    #[inline(never)]
    fn write_ram_across_observed<const ROUTED: bool>(
        &self,
        addr: u64,
        bytes: &[u8],
        observer: &impl InterpreterMemory,
    ) -> Result<(), u64> {
        let mut note = |at, len| self.note_write_observed(at, len, observer);
        self.write_ram_across(addr, bytes, ROUTED, &mut note)
    }

    /// Containment miss of [`Self::write_ram_checked_mode_observed`]. An all-RAM
    /// span is copied as one contained span is. A span with ROM bytes drops them, as
    /// a contained ROM store does, and copies the RAM pieces only after every RAM
    /// piece passed its host probe, so a protected byte faults with nothing written.
    #[cold]
    #[inline(never)]
    fn write_ram_across(
        &self,
        addr: u64,
        bytes: &[u8],
        routed: bool,
        note: &mut dyn FnMut(u64, usize),
    ) -> Result<(), u64> {
        let Some(rom) = self.adjacent_span(addr, bytes.len()) else {
            return Err(crate::fault::record_fp_fault(
                addr,
                bytes.len(),
                crate::FaultOrigin::GuestMap,
                crate::AddressPrecision::TransferAddress,
            ));
        };
        let fault = |at: usize| {
            let at = self.guest_base + (at - self.host_base() as usize) as u64;
            record_host_fault(at);
            crate::fault::record_fp_fault(
                at,
                bytes.len(),
                crate::FaultOrigin::HostAccess,
                crate::AddressPrecision::ExactByte,
            )
        };
        if rom && routed {
            self.probe_ram_pieces(addr, bytes.len()).map_err(fault)?;
        }
        let mut copy = |at: u64, src: &[u8]| {
            // SAFETY: adjacent_span validated the piece; the local source is live.
            let result = unsafe {
                let ptr = self.host_base().cast_mut().add(self.host_off(at));
                if routed {
                    crate::usercopy::copy_out(ptr, src)
                } else {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len());
                    Ok(())
                }
            };
            let committed = result
                .as_ref()
                .map_or_else(|fault| fault.committed, |()| src.len());
            if committed != 0 {
                note(at, committed);
            }
            result.map_err(|f| fault(f.at))
        };
        if !rom {
            return copy(addr, bytes);
        }
        self.span_pieces(addr, bytes.len(), |at, n, kind| {
            if kind != RegionKind::Ram {
                return Ok(());
            }
            let offset = (at - addr) as usize;
            copy(at, &bytes[offset..offset + n])
        })
    }

    /// Cold evidence about one byte, independent of permissions and host VM state.
    #[cold]
    pub(crate) fn absent_byte(&self, addr: u64) -> bool {
        !self
            .regions
            .iter()
            .any(|r| r.start <= addr && addr - r.start < r.size as u64)
    }

    pub(crate) fn mapped_ram(&self, addr: u64) -> bool {
        self.region_at(addr, 1)
            .is_ok_and(|r| !matches!(r.kind, RegionKind::Trap))
    }

    pub(crate) fn region_kind(&self, addr: u64, size: u8) -> Result<RegionKind, MemTrap> {
        self.region_at(addr, size).map(|r| r.kind)
    }

    pub(crate) fn region_end(&self, addr: u64) -> Result<u64, MemTrap> {
        self.region_at(addr, 1).map(|r| r.start + r.size as u64)
    }

    /// The region containing `[addr, addr + size)`, or `Unmapped` if the range
    /// escapes every mapped region; a non-canonical miss is `NonCanonical`.
    /// Shared by scalar read/write. Containment only: a span covered by adjacent
    /// RAM/ROM regions is performed one level up, by the `*_spanning` entry points
    /// and `SplitMemory` (see [`Self::adjacent_span`]); atomic operands keep the
    /// miss.
    fn region_at(&self, addr: u64, size: u8) -> Result<&Region, MemTrap> {
        let end = addr
            .checked_add(size as u64)
            .ok_or_else(|| address_miss(addr, size))?;
        let contains = |r: &Region| r.start <= addr && end <= r.start + r.size as u64;
        // Fast path: accesses are highly local, so the last region we hit usually
        // still contains this one — check it before the linear scan. The index stays
        // valid because `regions` only mutates through `&mut self` (map/unmap), never
        // during `&self` execution; a stale index just misses the fast path.
        let hint = self.last_region.load(Ordering::Relaxed);
        if let Some(r) = self.regions.get(hint) {
            if contains(r) {
                return Ok(r);
            }
        }
        match self.regions.iter().position(contains) {
            Some(i) => {
                self.last_region.store(i, Ordering::Relaxed);
                Ok(&self.regions[i])
            }
            None => Err(address_miss(addr, size)),
        }
    }

    fn atomic_region_at(&self, addr: u64, size: u8) -> Result<&Region, MemTrap> {
        let region = self.region_at(addr, size).map_err(|fault| {
            // §5.2: an operand partly in a device takes the device path: the
            // embedder's emulation on Long64 and Compat32, its pieces on
            // Real16 and Machine IR.
            if let Some(end) = addr.checked_add(u64::from(size)) {
                if self.regions.iter().any(|r| {
                    matches!(r.kind, RegionKind::Trap)
                        && r.start < end
                        && addr < r.start + r.size as u64
                }) {
                    return MemTrap::Mmio;
                }
            }
            fault
        })?;
        if matches!(region.kind, RegionKind::Trap) {
            return Err(MemTrap::Mmio);
        }
        Ok(region)
    }

    unsafe fn copy_from_guest(&self, ptr: *const u8, buf: &mut [u8]) -> Result<(), usize> {
        // SAFETY: caller validates the whole span and local buffer. The default
        // contract requires tagged RAM to stay accessible on the host.
        unsafe {
            if self.recover_host_faults {
                crate::usercopy::copy_in(ptr, buf)
            } else {
                std::ptr::copy_nonoverlapping(ptr, buf.as_mut_ptr(), buf.len());
                Ok(())
            }
        }
    }

    unsafe fn copy_to_guest(
        &self,
        ptr: *mut u8,
        buf: &[u8],
    ) -> Result<(), crate::usercopy::WriteFault> {
        // SAFETY: same contract as copy_from_guest, with writable guest bytes.
        unsafe {
            if self.recover_host_faults {
                crate::usercopy::copy_out(ptr, buf)
            } else {
                std::ptr::copy_nonoverlapping(buf.as_ptr(), ptr, buf.len());
                Ok(())
            }
        }
    }

    fn host_fault(&self, at: usize) -> MemTrap {
        record_host_fault(self.guest_base + (at - self.host_base() as usize) as u64)
    }

    /// Wide checked transfer retaining the faulting byte for the FP helpers.
    pub(crate) fn read_ram_checked(&self, addr: u64, buf: &mut [u8]) -> Result<(), u64> {
        if self.recover_host_faults {
            self.read_ram_checked_spanning::<true>(addr, buf)
        } else {
            self.read_ram_checked_spanning::<false>(addr, buf)
        }
    }

    /// [`Self::read_ram_checked_mode`] that also copies a span over adjacent RAM/ROM
    /// regions ([`Self::read_ram_across`]).
    #[inline(always)]
    pub(crate) fn read_ram_checked_spanning<const ROUTED: bool>(
        &self,
        addr: u64,
        buf: &mut [u8],
    ) -> Result<(), u64> {
        if self.region_for(addr, buf.len()).is_none() {
            return self.read_ram_across(addr, buf, ROUTED);
        }
        self.read_ram_checked_mode::<ROUTED>(addr, buf)
    }

    /// Containment only, as the scalar transfers.
    #[inline(always)]
    pub(crate) fn read_ram_checked_mode<const ROUTED: bool>(
        &self,
        addr: u64,
        buf: &mut [u8],
    ) -> Result<(), u64> {
        let region = self.region_for(addr, buf.len()).ok_or_else(|| {
            crate::fault::record_fp_fault(
                addr,
                buf.len(),
                crate::FaultOrigin::GuestMap,
                crate::AddressPrecision::TransferAddress,
            )
        })?;
        if matches!(region.kind, RegionKind::Trap) {
            return Err(crate::fault::record_fp_fault(
                addr,
                buf.len(),
                crate::FaultOrigin::Unknown,
                crate::AddressPrecision::TransferAddress,
            ));
        }
        // SAFETY: region_for validates the entire range; the local output is live.
        unsafe {
            let ptr = self.host_base().add(self.host_off(addr));
            if ROUTED {
                crate::usercopy::copy_in(ptr, buf)
            } else {
                std::ptr::copy_nonoverlapping(ptr, buf.as_mut_ptr(), buf.len());
                Ok(())
            }
        }
        .map_err(|at| {
            let addr = self.guest_base + (at - self.host_base() as usize) as u64;
            record_host_fault(addr);
            crate::fault::record_fp_fault(
                addr,
                buf.len(),
                crate::FaultOrigin::HostAccess,
                crate::AddressPrecision::ExactByte,
            )
        })
    }

    pub(crate) fn write_ram_checked(&self, addr: u64, bytes: &[u8]) -> Result<(), u64> {
        if self.recover_host_faults {
            self.write_ram_checked_spanning_observed::<true>(addr, bytes, self)
        } else {
            self.write_ram_checked_spanning_observed::<false>(addr, bytes, self)
        }
    }

    /// [`Self::write_ram_checked_mode_observed`] that also stores a span over
    /// adjacent RAM/ROM regions ([`Self::write_ram_across`]).
    #[inline(always)]
    pub(crate) fn write_ram_checked_spanning_observed<const ROUTED: bool>(
        &self,
        addr: u64,
        bytes: &[u8],
        observer: &impl InterpreterMemory,
    ) -> Result<(), u64> {
        if self.region_for(addr, bytes.len()).is_none() {
            return self.write_ram_across_observed::<ROUTED>(addr, bytes, observer);
        }
        self.write_ram_checked_mode_observed::<ROUTED>(addr, bytes, observer)
    }

    /// Containment only, as the scalar transfers.
    #[inline(always)]
    fn write_ram_checked_mode_observed<const ROUTED: bool>(
        &self,
        addr: u64,
        bytes: &[u8],
        observer: &impl InterpreterMemory,
    ) -> Result<(), u64> {
        let region = self.region_for(addr, bytes.len()).ok_or_else(|| {
            crate::fault::record_fp_fault(
                addr,
                bytes.len(),
                crate::FaultOrigin::GuestMap,
                crate::AddressPrecision::TransferAddress,
            )
        })?;
        if matches!(region.kind, RegionKind::Trap) {
            return Err(crate::fault::record_fp_fault(
                addr,
                bytes.len(),
                crate::FaultOrigin::Unknown,
                crate::AddressPrecision::TransferAddress,
            ));
        }
        if matches!(region.kind, RegionKind::Rom) {
            return Ok(());
        }
        // SAFETY: region_for validates the entire range; the local source is live.
        let result = unsafe {
            let ptr = self.host_base().cast_mut().add(self.host_off(addr));
            if ROUTED {
                crate::usercopy::copy_out(ptr, bytes)
            } else {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
                Ok(())
            }
        };
        let committed = result
            .as_ref()
            .map_or_else(|fault| fault.committed, |()| bytes.len());
        if committed != 0 {
            self.note_write_observed(addr, committed, observer);
        }
        result.map_err(|fault| {
            let addr = self.guest_base + (fault.at - self.host_base() as usize) as u64;
            record_host_fault(addr);
            crate::fault::record_fp_fault(
                addr,
                bytes.len(),
                crate::FaultOrigin::HostAccess,
                crate::AddressPrecision::ExactByte,
            )
        })
    }

    /// Scalar read used by the interpreter and trap-out path (§8.1). Little-endian.
    /// MUST bounds-check (§8.2.3) — an out-of-range addr is a MemTrap, never a panic/UB.
    /// A `Trap` region yields `MemTrap::Mmio` (routed out as `Exit::MmioRead`).
    pub fn read(&self, addr: u64, size: u8) -> Result<u64, MemTrap> {
        if self.recover_host_faults {
            self.read_spanning::<true>(addr, size)
        } else {
            self.read_spanning::<false>(addr, size)
        }
    }

    /// [`Self::read_mode`] whose containment miss reads a span over adjacent
    /// RAM/ROM regions ([`Self::read_across`]).
    #[inline]
    pub(crate) fn read_spanning<const ROUTED: bool>(
        &self,
        addr: u64,
        size: u8,
    ) -> Result<u64, MemTrap> {
        match self.read_mode::<ROUTED>(addr, size) {
            Err(miss @ MemTrap::Unmapped) if size > 1 => self.read_across(addr, size, miss, ROUTED),
            result => result,
        }
    }

    /// Containment only: a span that no single region contains is `Unmapped`.
    #[inline]
    pub(crate) fn read_mode<const ROUTED: bool>(
        &self,
        addr: u64,
        size: u8,
    ) -> Result<u64, MemTrap> {
        let region = self.region_at(addr, size)?;
        if matches!(region.kind, RegionKind::Trap) {
            return Err(MemTrap::Mmio);
        }
        // SAFETY: bounds-checked into a mapped RAM region (§8.2.3). Single-copy atomic
        // for a naturally-aligned access so a store racing on another vcpu can't tear it
        // (a plain byte copy could) — matching the guest `mov` and the JIT's host `mov`.
        let ptr = unsafe { (*self.backing.get()).as_ptr().add(self.host_off(addr)) };
        if ROUTED {
            unsafe { crate::usercopy::load(ptr, size) }.map_err(|at| self.host_fault(at))
        } else {
            Ok(unsafe { atomic_load_le(ptr, size) })
        }
    }

    /// Copy a decode window, retaining a valid prefix before the first host fault.
    /// No live guest slice is passed to the decoder.
    pub(crate) fn fetch_code(&self, addr: u64, buf: &mut [u8]) -> Result<usize, MemTrap> {
        // SAFETY: the initialized output slice has this writable extent.
        unsafe { self.fetch_code_into(addr, buf.as_mut_ptr(), buf.len()) }
    }

    /// Copy into a writable allocation without constructing references to uninitialized bytes.
    pub(crate) unsafe fn fetch_code_into(
        &self,
        addr: u64,
        out: *mut u8,
        capacity: usize,
    ) -> Result<usize, MemTrap> {
        let len = self.code_len(addr, capacity)?;
        // SAFETY: code_len validates the contiguous RAM/ROM window.
        if !self.recover_host_faults {
            // SAFETY: the requested chunk stays in registered accessible RAM.
            unsafe {
                std::ptr::copy_nonoverlapping(self.host_base().add(self.host_off(addr)), out, len);
            }
            return Ok(len);
        }
        match unsafe {
            crate::usercopy::copy_in_raw(self.host_base().add(self.host_off(addr)), out, len)
        } {
            Ok(()) => Ok(len),
            Err(at) => {
                let copied = at - self.host_base() as usize - self.host_off(addr);
                if copied == 0 {
                    Err(self.host_fault(at))
                } else {
                    Ok(copied)
                }
            }
        }
    }

    /// Owned code snapshot for inspection/decoding (§7.3).
    ///
    /// This returns `Vec<u8>` rather than a borrowed slice of live guest RAM.
    /// Borrow the returned vector when passing it to iced. Concurrent guest stores
    /// cannot alias a Rust reference to the backing. Returns up to
    /// `max_len` bytes from `addr`, continuing across adjacent RAM/ROM regions
    /// until the first gap or Trap region. `Unmapped` if
    /// `addr` isn't inside a mapped region. `Trap` regions return `Mmio` without
    /// exposing backing bytes; the lift maps this to an execute fault. With host
    /// fault recovery enabled, an inaccessible first byte returns `HostFault`;
    /// a later inaccessible byte ends the returned readable prefix.
    pub fn code_slice(&self, addr: u64, max_len: usize) -> Result<Vec<u8>, MemTrap> {
        let mut code = vec![0; self.code_len(addr, max_len)?];
        let len = self.fetch_code(addr, &mut code)?;
        code.truncate(len);
        Ok(code)
    }

    pub(crate) fn fetch_chunk_len(&self, addr: u64, max: usize) -> Result<usize, MemTrap> {
        let region = self.region_at(addr, 1)?;
        if matches!(region.kind, RegionKind::Trap) {
            return Err(MemTrap::Mmio);
        }
        let host = self.host_base() as usize + self.host_off(addr);
        let page = host_page_size();
        Ok(max
            .min(page - host % page)
            .min((region.start + region.size as u64 - addr) as usize))
    }

    fn code_len(&self, addr: u64, max_len: usize) -> Result<usize, MemTrap> {
        let region = self
            .regions
            .iter()
            .find(|r| r.start <= addr && addr < r.start + r.size as u64)
            .ok_or(MemTrap::Unmapped)?;
        if matches!(region.kind, RegionKind::Trap) {
            return Err(MemTrap::Mmio);
        }
        let limit = addr.saturating_add(max_len as u64);
        let mut region_end = region.start + region.size as u64;
        while region_end < limit {
            let Some(next) = self.regions.iter().find(|r| {
                r.start == region_end && r.size > 0 && !matches!(r.kind, RegionKind::Trap)
            }) else {
                break;
            };
            region_end += next.size as u64;
        }
        let end = limit.min(region_end);
        Ok((end - addr) as usize)
    }

    /// `&self`, not `&mut self` (§8 pitfall) — guest RAM is interior-mutable and shared.
    /// Little-endian. Bounds-checked; a `Trap` region yields `MemTrap::Mmio`.
    pub fn write(&self, addr: u64, val: u64, size: u8) -> Result<(), MemTrap> {
        if self.recover_host_faults {
            self.write_mode::<true>(addr, val, size)
        } else {
            self.write_mode::<false>(addr, val, size)
        }
    }

    /// [`Self::write_mode_observed`] whose containment miss stores a span over
    /// adjacent RAM/ROM regions, observed by `self`.
    #[inline]
    pub(crate) fn write_mode<const ROUTED: bool>(
        &self,
        addr: u64,
        val: u64,
        size: u8,
    ) -> Result<(), MemTrap> {
        self.write_spanning_observed::<ROUTED, false>(addr, val, size, self)
    }

    /// [`Self::write_mode_observed`] whose containment miss stores a span over
    /// adjacent RAM/ROM regions ([`Self::write_across_observed`]). Only `region_at`
    /// reports `Unmapped` there; nothing is stored before it.
    #[inline(always)]
    fn write_spanning_observed<const ROUTED: bool, const PREFLIGHTED: bool>(
        &self,
        addr: u64,
        val: u64,
        size: u8,
        observer: &impl InterpreterMemory,
    ) -> Result<(), MemTrap> {
        match self.write_mode_observed::<ROUTED, PREFLIGHTED>(addr, val, size, observer) {
            Err(miss @ MemTrap::Unmapped) if size > 1 => {
                self.write_across_observed::<ROUTED, PREFLIGHTED>(addr, val, size, observer, miss)
            }
            result => result,
        }
    }

    /// Containment only: a span that no single region contains is `Unmapped`.
    #[inline(always)]
    fn write_mode_observed<const ROUTED: bool, const PREFLIGHTED: bool>(
        &self,
        addr: u64,
        val: u64,
        size: u8,
        observer: &impl InterpreterMemory,
    ) -> Result<(), MemTrap> {
        let region = self.region_at(addr, size)?;
        if matches!(region.kind, RegionKind::Trap) {
            return Err(MemTrap::Mmio);
        }
        // A masked ROM discards the store: drop it silently (no fault, backing
        // untouched, no SMC note — nothing changed). ROM contents arrive via
        // `write_bytes` (the loader path), which ignores region kind.
        if matches!(region.kind, RegionKind::Rom) {
            return Ok(());
        }
        // SAFETY: the one deliberate interior-mutable write (§8). Guest stores race
        // like real hardware; ordering comes from TSO barriers, not `&mut`. The range
        // is bounds-checked to lie inside a mapped RAM region. Single-copy atomic for a
        // naturally-aligned access (like the guest `mov` and the JIT's host `mov`), so a
        // concurrent load on another vcpu never sees a torn word.
        let ptr = unsafe { ((*self.backing.get()).as_ptr() as *mut u8).add(self.host_off(addr)) };
        if ROUTED {
            // SAFETY: region_at validates the full span; PREFLIGHTED is set only
            // after a whole-destination probe, with stable host protections.
            unsafe {
                if PREFLIGHTED {
                    crate::usercopy::store_preflighted(ptr, val, size)
                } else {
                    crate::usercopy::store(ptr, val, size)
                }
            }
            .map_err(|fault| {
                if fault.committed != 0 {
                    self.note_write_observed(addr, fault.committed, observer);
                }
                self.host_fault(fault.at)
            })?;
        } else {
            unsafe { atomic_store_le(ptr, val, size) };
        }
        self.note_write_observed(addr, size as usize, observer); // SMC: catch a store onto a code page (§10)
        Ok(())
    }

    /// Atomic read-modify-write on a mapped RAM location (§8.2.3, §11). Returns the
    /// prior value (size-masked). Uses Rust SeqCst host atomics. Full x86 locked-op
    /// ordering (SDM Vol 3A §11.2.3.9) is not guaranteed for aarch64 LL/SC paths
    /// without an additional full fence. A naturally-aligned access uses a real
    /// host atomic. A misaligned one is atomic when it lies inside one aligned
    /// 16-byte granule, and is not when it crosses one (`Memory::locked_unaligned`).
    pub fn atomic_rmw(&self, addr: u64, src: u64, size: u8, op: RmwOp) -> Result<u64, MemTrap> {
        if self.recover_host_faults {
            self.atomic_rmw_mode::<true>(addr, src, size, op)
        } else {
            self.atomic_rmw_mode::<false>(addr, src, size, op)
        }
    }

    #[inline]
    pub(crate) fn atomic_rmw_mode<const ROUTED: bool>(
        &self,
        addr: u64,
        src: u64,
        size: u8,
        op: RmwOp,
    ) -> Result<u64, MemTrap> {
        self.atomic_rmw_mode_observed::<ROUTED>(addr, src, size, op, self)
    }

    #[inline(always)]
    fn atomic_rmw_mode_observed<const ROUTED: bool>(
        &self,
        addr: u64,
        src: u64,
        size: u8,
        op: RmwOp,
        observer: &impl InterpreterMemory,
    ) -> Result<u64, MemTrap> {
        let region = self.atomic_region_at(addr, size)?;
        // SAFETY: bounds-checked into a mapped RAM region; `ptr` is inside the
        // backing buffer. Interior-mutable shared access is the intended model (§8).
        let ptr = unsafe { ((*self.backing.get()).as_ptr() as *mut u8).add(self.host_off(addr)) };
        // A locked RMW on masked ROM discards the modify half; the read half still
        // returns the current (unchanged) value. No store, no fault, no SMC note.
        if matches!(region.kind, RegionKind::Rom) {
            return self
                .read_mode::<ROUTED>(addr, size)
                .map(|v| v & mask_bits(size));
        }
        if !ROUTED {
            let old = unsafe { atomic_rmw_raw(self, ptr, src, size, op) };
            self.note_write_observed(addr, size as usize, observer);
            return Ok(old & mask_bits(size));
        }
        let old = if (ptr as usize).is_multiple_of(size as usize) {
            let mut old =
                unsafe { crate::usercopy::load(ptr, size) }.map_err(|at| self.host_fault(at))?;
            loop {
                let new = apply_rmw(old, src, op, size);
                let observed = unsafe { crate::usercopy::cas(ptr, old, new, size) }
                    .map_err(|at| self.host_fault(at))?;
                if observed == old {
                    break;
                }
                old = observed;
            }
            old
        } else {
            self.locked_unaligned::<true>(ptr, size, LockedUpdate::Rmw { src, op })
                .map_err(|at| self.host_fault(at))?
        };
        self.note_write_observed(addr, size as usize, observer);
        Ok(old & mask_bits(size))
    }

    /// Aligned 16-byte CAS shared by interpreted and compiled CMPXCHG16B.
    /// The caller checks guest alignment before invoking this operation.
    pub fn atomic_cas128(&self, addr: u64, expected: u128, src: u128) -> Result<u128, MemTrap> {
        if self.recover_host_faults {
            self.atomic_cas128_mode::<true>(addr, expected, src)
        } else {
            self.atomic_cas128_mode::<false>(addr, expected, src)
        }
    }

    #[inline]
    pub(crate) fn atomic_cas128_mode<const ROUTED: bool>(
        &self,
        addr: u64,
        expected: u128,
        src: u128,
    ) -> Result<u128, MemTrap> {
        self.atomic_cas128_mode_observed::<ROUTED>(addr, expected, src, self)
    }

    #[inline(always)]
    fn atomic_cas128_mode_observed<const ROUTED: bool>(
        &self,
        addr: u64,
        expected: u128,
        src: u128,
        observer: &impl InterpreterMemory,
    ) -> Result<u128, MemTrap> {
        let region = self.region_at(addr, 16)?;
        if matches!(region.kind, RegionKind::Trap) {
            return Err(MemTrap::Mmio);
        }
        // SAFETY: region_at checked all 16 bytes; backing preserves guest alignment.
        let ptr = unsafe { ((*self.backing.get()).as_ptr() as *mut u8).add(self.host_off(addr)) };
        assert_eq!(
            ptr as usize & 15,
            0,
            "16-byte CAS requires an aligned operand"
        );
        if matches!(region.kind, RegionKind::Rom) {
            // ROM discards the write half, matching the scalar CAS policy.
            // SAFETY: the full read is in the checked region.
            let lo = self.read_mode::<ROUTED>(addr, 8)?;
            let hi = self.read_mode::<ROUTED>(addr + 8, 8)?;
            return Ok(u128::from(lo) | (u128::from(hi) << 64));
        }
        // SAFETY: ptr names 16 mapped, aligned, interior-mutable RAM bytes.
        let old = if ROUTED {
            unsafe { crate::usercopy::cas_pair(ptr, expected, src) }
                .map_err(|at| self.host_fault(at))?
        } else {
            unsafe { atomic_cas128_raw(ptr, expected, src) }
        };
        // SDM Vol 2A CMPXCHG8B/CMPXCHG16B, Description: failed compares write back too.
        self.note_write_observed(addr, 16, observer);
        Ok(old)
    }

    /// Atomic compare-exchange (`cmpxchg`, §8.2.3). If `[addr] == expected`, store
    /// `src`; return the prior value either way (size-masked).
    pub fn atomic_cas(&self, addr: u64, expected: u64, src: u64, size: u8) -> Result<u64, MemTrap> {
        if self.recover_host_faults {
            self.atomic_cas_mode::<true>(addr, expected, src, size)
        } else {
            self.atomic_cas_mode::<false>(addr, expected, src, size)
        }
    }

    #[inline]
    pub(crate) fn atomic_cas_mode<const ROUTED: bool>(
        &self,
        addr: u64,
        expected: u64,
        src: u64,
        size: u8,
    ) -> Result<u64, MemTrap> {
        self.atomic_cas_mode_observed::<ROUTED>(addr, expected, src, size, self)
    }

    #[inline(always)]
    fn atomic_cas_mode_observed<const ROUTED: bool>(
        &self,
        addr: u64,
        expected: u64,
        src: u64,
        size: u8,
        observer: &impl InterpreterMemory,
    ) -> Result<u64, MemTrap> {
        let region = self.atomic_region_at(addr, size)?;
        // SAFETY: as in `atomic_rmw`.
        let ptr = unsafe { ((*self.backing.get()).as_ptr() as *mut u8).add(self.host_off(addr)) };
        // A `cmpxchg` against masked ROM never stores; it returns the current value
        // (so a matching `expected` reports success but leaves the ROM unchanged).
        if matches!(region.kind, RegionKind::Rom) {
            return self
                .read_mode::<ROUTED>(addr, size)
                .map(|v| v & mask_bits(size));
        }
        let old = if ROUTED {
            if (ptr as usize).is_multiple_of(size as usize) {
                unsafe { crate::usercopy::cas(ptr, expected & mask_bits(size), src, size) }
            } else {
                let expected = expected & mask_bits(size);
                self.locked_unaligned::<true>(ptr, size, LockedUpdate::Cas { expected, src })
            }
            .map_err(|at| self.host_fault(at))?
        } else {
            unsafe { atomic_cas_raw(self, ptr, expected & mask_bits(size), src, size) }
        };
        self.note_write_observed(addr, size as usize, observer);
        Ok(old & mask_bits(size))
    }

    /// A locked operand that is not naturally aligned, at host address `ptr` in
    /// the backing; returns the operand's prior value. "The integrity of a bus
    /// lock is not affected by the alignment of the memory field" and "Locked
    /// operations are atomic with respect to all other memory operations" (SDM
    /// Vol 3A §11.1.2.2 p. 11-4; Vol 2A LOCK p. 3-565).
    ///
    /// Inside one aligned 16-byte granule, the update is a compare-and-swap
    /// loop on the containing aligned host word: the 8-byte word when the
    /// operand fits in one, else the 16-byte granule. Each pass takes a
    /// snapshot of the word, updates the operand's bytes in it, and swaps it in
    /// only if the word still holds the snapshot, so the bytes outside the
    /// operand are written back exactly as they were. Any store to any byte of
    /// the word after the snapshot (an aligned atomic on the same bytes, or a
    /// plain store to a neighbour byte) fails the swap, and the loop retries
    /// from the value it saw. The swaps are the host primitives the aligned
    /// paths use: for 8 bytes a SeqCst `compare_exchange` or the routed `cas8`,
    /// for 16 bytes the CMPXCHG16B primitive [`atomic_cas128_raw`] or the
    /// routed pair routine, both of which return a validated snapshot. A failed
    /// `cmpxchg` compare writes the word back unchanged, as the instruction
    /// writes its destination back (SDM Vol 2A CMPXCHG p. 3-194). Host
    /// alignment equals guest alignment modulo 16 ([`Self::from_host_ram`]), so
    /// the host granule is the guest's.
    ///
    /// The argument is the host architecture's. On an x86-64 host the swaps
    /// are LOCK CMPXCHG and LOCK CMPXCHG16B, atomic against every other memory
    /// operation by the SDM rule above. On an aarch64 host it relies on the Arm
    /// architecture's single-copy atomicity of CAS and of the exclusive pair
    /// against an overlapping write of any size: ASSUMED, because the Arm ARM
    /// is not yet pinned in `oracles/`; `locked_misaligned` measures it.
    ///
    /// LIMITATION: an operand that crosses an aligned 16-byte granule has no
    /// host primitive that covers it, so it is a plain read, update and write.
    /// It completes with the single-processor result but is not atomic against
    /// other vCPUs. The same holds when the containing word extends past the
    /// backing, whose start or end need not be 16-byte aligned. A cross-vCPU
    /// rendezvous for these operands is a follow-up (spec.md §8.2.3).
    #[cold]
    #[inline(never)]
    fn locked_unaligned<const ROUTED: bool>(
        &self,
        ptr: *mut u8,
        size: u8,
        update: LockedUpdate,
    ) -> Result<u64, usize> {
        use std::sync::atomic::{AtomicU64, Ordering::Relaxed, Ordering::SeqCst};
        let mask = mask_bits(size);
        let host = ptr as usize;
        let len = size as usize;
        // SAFETY: shared read of the backing's fixed pointer and length (§8).
        let backing = unsafe { &*self.backing.get() };
        let base = backing.as_ptr() as usize;
        let inside =
            |start: usize, bytes: usize| base <= start && start + bytes <= base + backing.len();
        // A routed fault on the containing word is a fault on the operand's
        // page, which a granule never leaves: report the operand's first byte,
        // as a probe of the operand would.
        let at_operand = |_: usize| host;
        if (host & 7) + len <= 8 && inside(host & !7, 8) {
            let word = (host & !7) as *mut u8;
            let shift = (host & 7) * 8;
            // SAFETY: the aligned word lies inside the backing (checked above).
            let mut seen = if ROUTED {
                unsafe { crate::usercopy::load(word, 8) }.map_err(at_operand)?
            } else {
                unsafe { (*(word as *const AtomicU64)).load(Relaxed) }
            };
            loop {
                let old = (seen >> shift) & mask;
                let next = (seen & !(mask << shift)) | (update.apply(old, size) << shift);
                // SAFETY: as above; the swap is an aligned 8-byte host atomic.
                let now = if ROUTED {
                    unsafe { crate::usercopy::cas(word, seen, next, 8) }.map_err(at_operand)?
                } else {
                    match unsafe { &*(word as *const AtomicU64) }
                        .compare_exchange(seen, next, SeqCst, SeqCst)
                    {
                        Ok(v) | Err(v) => v,
                    }
                };
                if now == seen {
                    return Ok(old);
                }
                seen = now;
            }
        }
        if cfg!(any(target_arch = "aarch64", target_arch = "x86_64"))
            && (host & 15) + len <= 16
            && inside(host & !15, 16)
        {
            let granule = (host & !15) as *mut u8;
            let shift = (host & 15) * 8;
            let field = u128::from(mask) << shift;
            // The first guess is two aligned 8-byte loads; the swap validates it.
            // SAFETY: the aligned granule lies inside the backing (checked above).
            let (lo, hi) = if ROUTED {
                unsafe {
                    (
                        crate::usercopy::load(granule, 8).map_err(at_operand)?,
                        crate::usercopy::load(granule.add(8), 8).map_err(at_operand)?,
                    )
                }
            } else {
                unsafe {
                    (
                        (*(granule as *const AtomicU64)).load(Relaxed),
                        (*(granule.add(8) as *const AtomicU64)).load(Relaxed),
                    )
                }
            };
            let mut seen = u128::from(lo) | (u128::from(hi) << 64);
            loop {
                let old = (seen >> shift) as u64 & mask;
                let next = (seen & !field) | (u128::from(update.apply(old, size)) << shift);
                // SAFETY: as above; the swap is an aligned 16-byte host atomic.
                let now = if ROUTED {
                    unsafe { crate::usercopy::cas_pair(granule, seen, next) }.map_err(at_operand)?
                } else {
                    unsafe { atomic_cas128_raw(granule, seen, next) }
                };
                if now == seen {
                    return Ok(old);
                }
                seen = now;
            }
        }
        // The limitation above: one read, then one write when the update stores.
        // SAFETY: the caller validated the operand's `size` bytes.
        unsafe {
            if ROUTED {
                crate::usercopy::update_plain(ptr, size, |old| update.store(old, size))
            } else {
                let old = plain_read(ptr, size);
                if let Some(new) = update.store(old, size) {
                    plain_write(ptr, new, size);
                }
                Ok(old)
            }
        }
    }
}

/// What a locked operand does to its prior value (`Memory::locked_unaligned`).
#[derive(Clone, Copy)]
enum LockedUpdate {
    /// `atomic_rmw`: always stores.
    Rmw { src: u64, op: RmwOp },
    /// `atomic_cas`, `expected` already masked: stores `src` on a match.
    Cas { expected: u64, src: u64 },
}

impl LockedUpdate {
    /// The operand's new value; a failed compare leaves it unchanged.
    fn apply(self, old: u64, size: u8) -> u64 {
        self.store(old, size).unwrap_or(old)
    }

    /// The value the instruction stores, or `None` for a failed compare.
    fn store(self, old: u64, size: u8) -> Option<u64> {
        match self {
            Self::Rmw { src, op } => Some(apply_rmw(old, src, op, size)),
            Self::Cas { expected, src } => (old == expected).then_some(src & mask_bits(size)),
        }
    }
}

/// # Safety
/// ptr names 16 valid, aligned, writable bytes shared by the vcpus.
#[cfg(target_arch = "aarch64")]
unsafe fn atomic_cas128_raw(ptr: *mut u8, expected: u128, src: u128) -> u128 {
    let (lo, hi): (u64, u64);
    // The successful ST(L)XP validates the entire pair, including on a mismatch.
    // Always store: LDXP alone need not give a single-copy-atomic 128-bit snapshot.
    // DMBs supply the full ordering of a guest locked instruction in every tier.
    // SDM Vol 3A §11.2.3.9, Loads and Stores Are Not Reordered with Locked Instructions.
    // SAFETY: caller guarantees alignment and validity; outputs cannot alias inputs.
    unsafe {
        core::arch::asm!(
            "dmb ish",
            "2:",
            "ldaxp {lo}, {hi}, [{ptr}]",
            "cmp {lo}, {elo}",
            "ccmp {hi}, {ehi}, #0, eq",
            "csel {nlo}, {slo}, {lo}, eq",
            "csel {nhi}, {shi}, {hi}, eq",
            "stlxp {status:w}, {nlo}, {nhi}, [{ptr}]",
            "cbnz {status:w}, 2b",
            "dmb ish",
            ptr = in(reg) ptr,
            elo = in(reg) expected as u64, ehi = in(reg) (expected >> 64) as u64,
            slo = in(reg) src as u64, shi = in(reg) (src >> 64) as u64,
            lo = out(reg) lo, hi = out(reg) hi,
            nlo = out(reg) _, nhi = out(reg) _, status = out(reg) _,
            options(nostack),
        )
    };
    (hi as u128) << 64 | lo as u128
}

/// # Safety
/// ptr names 16 valid, aligned, writable bytes shared by the vcpus.
#[cfg(target_arch = "x86_64")]
unsafe fn atomic_cas128_raw(ptr: *mut u8, expected: u128, src: u128) -> u128 {
    assert!(
        std::is_x86_feature_detected!("cmpxchg16b"),
        "host requires CMPXCHG16B"
    );
    let mut lo = expected as u64;
    let mut hi = (expected >> 64) as u64;
    // LLVM reserves RBX; exchange it with a scratch register and restore it.
    // SDM Vol 2A CMPXCHG8B/CMPXCHG16B, Operation.
    // SAFETY: caller guarantees the aligned 16-byte access; RBX is restored.
    unsafe {
        core::arch::asm!(
            "xchg {scratch}, rbx",
            "lock cmpxchg16b [{ptr}]",
            "xchg {scratch}, rbx",
            ptr = in(reg) ptr,
            scratch = inout(reg) src as u64 => _,
            inout("rax") lo, inout("rdx") hi, in("rcx") (src >> 64) as u64,
            options(nostack),
        )
    };
    (hi as u128) << 64 | lo as u128
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
unsafe fn atomic_cas128_raw(_: *mut u8, _: u128, _: u128) -> u128 {
    unimplemented!("16-byte CAS requires an aarch64 or x86-64 host")
}

/// Raw atomic RMW dispatch over a guest pointer in `mem`'s backing. Aligned →
/// real host atomic; misaligned → [`Memory::locked_unaligned`].
///
/// # Safety
/// `ptr` must point to `size` valid, mapped bytes inside the backing buffer.
unsafe fn atomic_rmw_raw(mem: &Memory, ptr: *mut u8, src: u64, size: u8, op: RmwOp) -> u64 {
    use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, AtomicU8, Ordering::SeqCst};

    macro_rules! rmw {
        ($atom:ty, $int:ty) => {{
            if ptr as usize % std::mem::size_of::<$int>() == 0 {
                let a = unsafe { &*(ptr as *const $atom) };
                let s = src as $int;
                let old = match op {
                    RmwOp::Add => a.fetch_add(s, SeqCst),
                    RmwOp::Sub => a.fetch_sub(s, SeqCst),
                    // No native reverse-subtract atomic: CAS loop (new = s - cur).
                    RmwOp::Rsub => {
                        let mut cur = a.load(SeqCst);
                        loop {
                            match a.compare_exchange_weak(cur, s.wrapping_sub(cur), SeqCst, SeqCst)
                            {
                                Ok(v) => break v,
                                Err(v) => cur = v,
                            }
                        }
                    }
                    RmwOp::And => a.fetch_and(s, SeqCst),
                    RmwOp::Or => a.fetch_or(s, SeqCst),
                    RmwOp::Xor => a.fetch_xor(s, SeqCst),
                    RmwOp::Xchg => a.swap(s, SeqCst),
                };
                old as u64
            } else {
                mem.locked_unaligned::<false>(ptr, size, LockedUpdate::Rmw { src, op })
                    .unwrap_or_else(|_| unreachable!("only a routed transfer faults"))
            }
        }};
    }

    match size {
        1 => rmw!(AtomicU8, u8),
        2 => rmw!(AtomicU16, u16),
        4 => rmw!(AtomicU32, u32),
        _ => rmw!(AtomicU64, u64),
    }
}

/// # Safety
/// As `atomic_rmw_raw`.
unsafe fn atomic_cas_raw(mem: &Memory, ptr: *mut u8, expected: u64, src: u64, size: u8) -> u64 {
    use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, AtomicU8, Ordering::SeqCst};

    macro_rules! cas {
        ($atom:ty, $int:ty) => {{
            if ptr as usize % std::mem::size_of::<$int>() == 0 {
                let a = unsafe { &*(ptr as *const $atom) };
                // Failure returns the current value; success returns `expected`.
                match a.compare_exchange(expected as $int, src as $int, SeqCst, SeqCst) {
                    Ok(v) => v as u64,
                    Err(v) => v as u64,
                }
            } else {
                mem.locked_unaligned::<false>(ptr, size, LockedUpdate::Cas { expected, src })
                    .unwrap_or_else(|_| unreachable!("only a routed transfer faults"))
            }
        }};
    }

    match size {
        1 => cas!(AtomicU8, u8),
        2 => cas!(AtomicU16, u16),
        4 => cas!(AtomicU32, u32),
        _ => cas!(AtomicU64, u64),
    }
}

pub(crate) unsafe fn plain_read(ptr: *const u8, size: u8) -> u64 {
    let mut buf = [0u8; 8];
    unsafe { std::ptr::copy_nonoverlapping(ptr, buf.as_mut_ptr(), size as usize) };
    u64::from_le_bytes(buf)
}

pub(crate) unsafe fn plain_write(ptr: *mut u8, val: u64, size: u8) {
    let bytes = val.to_le_bytes();
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, size as usize) };
}

/// Little-endian scalar load that is **single-copy atomic** for a naturally-aligned
/// 1/2/4/8-byte access — the same guarantee real x86 gives a single `mov`, which the
/// JIT inherits by lowering a guest load to one host `mov`. The interpreter must match:
/// a byte-wise `plain_read` racing a concurrent store on another vcpu can observe a
/// *torn* word (half old, half new), which a lock-free guest reader (a Go runtime
/// pointer/length) never tolerates. A misaligned or odd-size access falls back to the
/// byte copy. LIMITATION: x86 also makes an unaligned 16-, 32- or 64-bit access to
/// cached memory atomic when it fits in one cache line (SDM Vol 3A §11.1.1 p. 11-2),
/// and the byte copy is not, so such a load can tear here where it cannot on
/// hardware: against a plain store, or against a locked update of the same field
/// (`Memory::locked_unaligned`). `Relaxed` is sufficient: this fixes single-copy
/// atomicity (no tearing), not inter-location ordering, which x86 TSO handles separately.
///
/// # Safety
/// `ptr` must point to `size` valid, mapped bytes inside the backing buffer.
pub(crate) unsafe fn atomic_load_le(ptr: *const u8, size: u8) -> u64 {
    use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, AtomicU8, Ordering::Relaxed};
    match size {
        1 => unsafe { (*(ptr as *const AtomicU8)).load(Relaxed) as u64 },
        2 if (ptr as usize).is_multiple_of(2) => unsafe {
            (*(ptr as *const AtomicU16)).load(Relaxed) as u64
        },
        4 if (ptr as usize).is_multiple_of(4) => unsafe {
            (*(ptr as *const AtomicU32)).load(Relaxed) as u64
        },
        8 if (ptr as usize).is_multiple_of(8) => unsafe {
            (*(ptr as *const AtomicU64)).load(Relaxed)
        },
        _ => unsafe { plain_read(ptr, size) },
    }
}

/// Little-endian scalar store, the write-side mirror of [`atomic_load_le`]: a
/// naturally-aligned 1/2/4/8-byte store is single-copy atomic (one host atomic store),
/// so a concurrent load on another vcpu never sees a torn word. Misaligned/odd-size
/// falls back to the byte copy.
///
/// # Safety
/// `ptr` must point to `size` valid, mapped bytes inside the backing buffer.
pub(crate) unsafe fn atomic_store_le(ptr: *mut u8, val: u64, size: u8) {
    use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, AtomicU8, Ordering::Relaxed};
    match size {
        1 => unsafe { (*(ptr as *const AtomicU8)).store(val as u8, Relaxed) },
        2 if (ptr as usize).is_multiple_of(2) => unsafe {
            (*(ptr as *const AtomicU16)).store(val as u16, Relaxed)
        },
        4 if (ptr as usize).is_multiple_of(4) => unsafe {
            (*(ptr as *const AtomicU32)).store(val as u32, Relaxed)
        },
        8 if (ptr as usize).is_multiple_of(8) => unsafe {
            (*(ptr as *const AtomicU64)).store(val, Relaxed)
        },
        _ => unsafe { plain_write(ptr, val, size) },
    }
}

/// One bit per page intersecting `[guest_base, top)`, including partial pages.
/// A high base with a small backing needs only a small table. A zero-based 2^47
/// span reserves 4 GiB per bitmap (12 GiB for code/watch/dirty together), lazily
/// committed by `zeroed_words`; hosts with strict allocation limits can refuse it.
fn fresh_code_pages(guest_base: u64, top: u64) -> Box<[AtomicU64]> {
    let pages = if top == guest_base {
        0
    } else {
        top.div_ceil(CODE_PAGE_SIZE) - (guest_base >> CODE_PAGE_BITS)
    };
    zeroed_words(pages.div_ceil(64) as usize)
}

fn mask_bits(size: u8) -> u64 {
    if size >= 8 {
        u64::MAX
    } else {
        (1u64 << (size * 8)) - 1
    }
}

fn apply_rmw(old: u64, src: u64, op: RmwOp, size: u8) -> u64 {
    let m = mask_bits(size);
    let r = match op {
        RmwOp::Add => old.wrapping_add(src),
        RmwOp::Sub => old.wrapping_sub(src),
        RmwOp::Rsub => src.wrapping_sub(old),
        RmwOp::And => old & src,
        RmwOp::Or => old | src,
        RmwOp::Xor => old ^ src,
        RmwOp::Xchg => src,
    };
    r & m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_barrier_stats_batch_new_pages_and_skip_synchronized_tags() {
        crate::with_barrier_success(|| {
            let vm = crate::Vm::new(crate::VmConfig::flat(0x6000));
            let empty = LiftSource::new(&vm.mem);
            assert!(empty.validate());
            assert_eq!(vm.memory_stats().source_barrier_calls, 0);
            assert_eq!(vm.memory_stats().source_barrier_tagged_pages, 0);

            let first = LiftSource::new(&vm.mem);
            first.watch(0x1000, 0x3000);
            first.watch(0x2000, 1);
            assert!(first.validate());
            let stats = vm.memory_stats();
            assert_eq!(stats.source_barrier_calls, 1);
            assert_eq!(stats.source_barrier_tagged_pages, 3);

            let second = LiftSource::new(&vm.mem);
            second.watch(0x1000, 0x3000);
            assert!(second.validate());
            assert!(first.validate());
            let unchanged = vm.memory_stats();
            assert_eq!(unchanged.source_barrier_calls, stats.source_barrier_calls);
            assert_eq!(unchanged.source_barrier_ns, stats.source_barrier_ns);
            assert_eq!(unchanged.source_barrier_tagged_pages, 3);

            let third = LiftSource::new(&vm.mem);
            third.watch(0x4000, 1);
            assert!(third.validate());
            let stats = vm.memory_stats();
            assert_eq!(stats.source_barrier_calls, 2);
            assert_eq!(stats.source_barrier_tagged_pages, 4);
        });
    }

    #[test]
    fn source_barrier_stats_cover_coalesced_lifts_and_alias_pages() {
        crate::with_barrier_success(|| {
            let vm = crate::Vm::new(crate::VmConfig::flat(0x6000));
            vm.mem.declare_code_alias(0x1000, 0x3000, 0x2000).unwrap();
            let first = LiftSource::new(&vm.mem);
            first.watch(0x1000, 1);
            let second = LiftSource::new(&vm.mem);
            second.watch(0x2000, 1);
            assert!(second.validate());
            assert!(first.validate());
            let stats = vm.memory_stats();
            assert_eq!(stats.source_barrier_calls, 1);
            assert_eq!(stats.source_barrier_tagged_pages, 4);
        });
    }

    #[test]
    fn degraded_source_barrier_stats_count_pages_without_calls() {
        crate::with_barrier_unavailable(|| {
            let vm = crate::Vm::new(crate::VmConfig::flat(0x5000));
            let source = LiftSource::new(&vm.mem);
            source.watch(0x1000, 0x3000);
            assert!(source.validate());
            let stats = vm.memory_stats();
            assert_eq!(
                stats.source_barrier_mode,
                crate::SourceBarrierMode::Degraded
            );
            assert_eq!(stats.source_barrier_calls, 0);
            assert_eq!(stats.source_barrier_ns, 0);
            assert_eq!(stats.source_barrier_tagged_pages, 3);
        });
    }

    #[test]
    fn source_barrier_stats_count_retags_before_original_lift_validates() {
        crate::with_barrier_success(|| {
            let vm = crate::Vm::new(crate::VmConfig::flat(0x5000));
            let source = LiftSource::new(&vm.mem);
            source.watch(0x1000, 1);
            assert_eq!(vm.memory_stats().source_barrier_tagged_pages, 1);
            vm.mem.clear_all_code_pages();
            assert!(vm.mem.is_code_page(1));
            assert_eq!(vm.memory_stats().source_barrier_tagged_pages, 2);
            assert!(source.validate());
            assert_eq!(vm.memory_stats().source_barrier_calls, 1);
            assert_eq!(vm.memory_stats().source_barrier_tagged_pages, 2);
            drop(source);

            vm.mem.clear_all_code_pages();
            assert!(!vm.mem.is_code_page(1));
            let source = LiftSource::new(&vm.mem);
            source.watch(0x1000, 1);
            assert!(source.validate());
            assert_eq!(vm.memory_stats().source_barrier_calls, 2);
            assert_eq!(vm.memory_stats().source_barrier_tagged_pages, 3);
        });
    }

    #[test]
    fn source_barrier_stats_sum_concurrent_disjoint_lifts() {
        crate::with_barrier_success(|| {
            for degraded in [false, true] {
                let create = || crate::Vm::new(crate::VmConfig::flat(0x6000));
                let vm = if degraded {
                    crate::with_barrier_unavailable(create)
                } else {
                    create()
                };
                let rendezvous = std::sync::Barrier::new(2);
                std::thread::scope(|scope| {
                    for addr in [0x1000, 0x3000] {
                        let vm = &vm;
                        let rendezvous = &rendezvous;
                        scope.spawn(move || {
                            crate::with_barrier_success(|| {
                                let source = LiftSource::new(&vm.mem);
                                source.watch(addr, 0x2000);
                                rendezvous.wait();
                                assert!(source.validate());
                                rendezvous.wait();
                                assert_eq!(vm.memory_stats().source_barrier_tagged_pages, 4);
                            });
                        });
                    }
                });
                let stats = vm.memory_stats();
                assert_eq!(stats.source_barrier_tagged_pages, 4);
                if degraded {
                    assert_eq!(stats.source_barrier_calls, 0);
                    assert_eq!(stats.source_barrier_ns, 0);
                } else {
                    assert!((1..=2).contains(&stats.source_barrier_calls));
                }
            }
        });
    }

    #[test]
    fn source_barrier_stats_do_not_recount_tags_on_validation_retry() {
        crate::with_barrier_success(|| {
            let mut vm = crate::Vm::new(crate::VmConfig::flat(0x5000));
            vm.map(0, 0x5000, Prot::RWX, RegionKind::Ram).unwrap();
            let source = LiftSource::new(&vm.mem);
            source.watch(0x1000, 1);
            source.record(0x1000, &[0]);
            vm.write_bytes(0x1000, &[1]).unwrap();
            assert!(!source.validate());
            let retry = LiftSource::new(&vm.mem);
            retry.watch(0x1000, 1);
            retry.record(0x1000, &[1]);
            assert!(retry.validate());
            let stats = vm.memory_stats();
            assert_eq!(stats.source_barrier_calls, 1);
            assert_eq!(stats.source_barrier_tagged_pages, 1);
        });
    }

    #[test]
    fn failed_source_barrier_stats_count_every_attempt() {
        crate::with_barrier_success(|| {
            let vm = crate::Vm::new(crate::VmConfig::flat(0x5000));
            let source = LiftSource::new(&vm.mem);
            source.watch(0x1000, 0x2000);
            crate::with_barrier_flush_failure(|| assert!(source.validate()));
            let stats = vm.memory_stats();
            assert!(!source.publishable());
            assert_eq!(stats.source_barrier_calls, 3);
            assert_eq!(stats.source_barrier_failures, 3);
            assert_eq!(stats.source_barrier_tagged_pages, 2);
            drop(source);

            let source = LiftSource::new(&vm.mem);
            source.watch(0x1000, 1);
            assert!(source.validate());
            let stats = vm.memory_stats();
            assert_eq!(stats.source_barrier_calls, 4);
            assert_eq!(stats.source_barrier_tagged_pages, 3);
        });
    }

    #[test]
    fn failed_barrier_does_not_synchronize_other_lifters() {
        crate::with_barrier_success(|| {
            let mut mem = Memory::new(MemoryModel::Flat { size: 0x5000 });
            mem.map(0, 0x5000, Prot::RWX, RegionKind::Ram).unwrap();
            let first = LiftSource::new(&mem);
            first.watch(0x1000, 1);
            first.record(0x1000, &[0]);
            crate::with_barrier_flush_failure(|| assert!(first.validate()));
            assert!(!first.publishable());
            assert_eq!(mem.code_barrier_epoch.load(Ordering::Relaxed), 0);
            let second = LiftSource::new(&mem);
            second.watch(0x1000, 1);
            second.record(0x1000, &[0]);
            assert!(second.validate());
            assert!(second.publishable());
            assert!(mem.code_barrier_epoch.load(Ordering::Relaxed) > 0);
            assert!(
                !first.publishable(),
                "another lifter cannot license the failed snapshot"
            );
        });
    }

    #[test]
    fn smc_dispatch_acknowledges_only_its_own_notice_after_invalidation() {
        let m = Memory::new(MemoryModel::Flat { size: 0x4000 });
        let other_notices = crate::vm::EXIT_REQUESTED | crate::vm::CODE_SPACE_EXHAUSTED;
        let a = Arc::new(AtomicU64::new(other_notices));
        let b = Arc::new(AtomicU64::new(other_notices));
        m.register_dispatch_word(&a);
        m.register_dispatch_word(&b);
        m.mark_code(0x1000, 1);
        m.note_write(0x1000, 1);
        let pending = DISPATCH_DIRTY | other_notices;
        m.dispatch_smc(&b, |pages| {
            assert_eq!(pages, vec![(0x1000, 0x1001)]);
            assert_eq!(a.load(Ordering::Relaxed), pending);
            assert_eq!(b.load(Ordering::Relaxed), pending);
            assert!(m.dispatch_words.try_lock().is_err());
        });
        assert_eq!(a.load(Ordering::Relaxed), pending);
        assert_eq!(b.load(Ordering::Relaxed), other_notices);
        let mut acknowledged = false;
        m.dispatch_smc(&a, |pages| {
            assert!(pages.is_empty(), "another vcpu already drained the list");
            assert_eq!(a.load(Ordering::Relaxed), pending);
            acknowledged = true;
        });
        assert!(acknowledged);
        assert_eq!(a.load(Ordering::Relaxed), other_notices);

        m.note_write(0x1000, 1);
        assert_eq!(m.take_dirty_code(), vec![1]);
        assert_eq!(a.load(Ordering::Relaxed), pending);
        assert_eq!(b.load(Ordering::Relaxed), pending);
    }

    #[test]
    fn dirty_bytes_expand_unaligned_transitive_aliases_without_losing_offsets() {
        let m = Memory::new(MemoryModel::Flat { size: 0x10000 });
        m.declare_code_alias(0x1003, 0x4ffe, 8).unwrap();
        m.declare_code_alias(0x5000, 0x8007, 4).unwrap();
        for addr in [0x1000, 0x4000, 0x5000, 0x8000] {
            m.mark_code(addr, 1);
        }
        let word = Arc::new(AtomicU64::new(0));
        m.register_dispatch_word(&word);
        m.note_write(0x1004, 4);
        m.dispatch_smc(&word, |ranges| {
            assert_eq!(
                ranges,
                vec![
                    (0x1004, 0x1008),
                    (0x4fff, 0x5000),
                    (0x5000, 0x5003),
                    (0x8007, 0x800a)
                ]
            );
        });
        m.note_write(0x4ffe, 1);
        m.dispatch_smc(&word, |ranges| {
            assert_eq!(ranges, vec![(0x1003, 0x1004), (0x4ffe, 0x4fff)]);
        });
    }

    fn flat(size: u64) -> Memory {
        Memory::new(MemoryModel::Flat { size })
    }

    // SDM Vol 1 §3.3.7.1: 48-bit canonicality, both boundaries by value.
    #[test]
    fn canonical_address_boundaries() {
        for addr in [0, 0x0000_7fff_ffff_ffff, 0xffff_8000_0000_0000, u64::MAX] {
            assert!(is_canonical(addr));
        }
        for addr in [0x0000_8000_0000_0000, 0xffff_7fff_ffff_ffff] {
            assert!(!is_canonical(addr));
            let mut mem = Memory::new(MemoryModel::Flat { size: 0x1000 });
            assert!(matches!(
                mem.map(addr, 1, Prot::RW, RegionKind::Ram),
                Err(MapError::OutOfBounds)
            ));
            assert_eq!(mem.read(addr, 1), Err(MemTrap::NonCanonical));
        }
    }

    // SDM Vol 3A §4.5 p. 4-4: 48-bit CPU canonical means bits 63:48 copy bit 47.
    #[test]
    fn canonical_sign_extends_bit_47() {
        for (addr, expected) in [
            (0x0000_7fff_ffff_fff0, 0x0000_7fff_ffff_fff0),
            (0xffff_8765_4321_0fed, 0xffff_8765_4321_0fed),
            (0x1234_5678_9abc_def0, 0x0000_5678_9abc_def0),
            (0x0123_8765_4321_0fed, 0xffff_8765_4321_0fed),
        ] {
            assert_eq!(canonical(addr), expected, "{addr:#x}");
        }
    }

    #[test]
    fn reserved_map_rejects_noncanonical_and_straddling_regions() {
        let mut mem = Memory::new(MemoryModel::Reserved { span: 0x1000 });
        // Exercise map's validation independently of constructor validation.
        // No access is made through this synthetic oversized Reserved model.
        mem.model = MemoryModel::Reserved { span: 1 << 48 };
        for (addr, size) in [(1 << 47, 1), ((1 << 47) - 8, 16)] {
            assert!(matches!(
                mem.map(addr, size, Prot::RW, RegionKind::Ram),
                Err(MapError::OutOfBounds)
            ));
        }
        assert!(mem.regions.is_empty());
    }

    #[test]
    fn noncanonical_span_rejected_before_allocation() {
        assert!(std::panic::catch_unwind(|| Memory::new(MemoryModel::Flat {
            size: (1 << 47) + 1
        }))
        .is_err());
        let ram = HostRam {
            ptr: std::ptr::NonNull::<u128>::dangling().as_ptr().cast(),
            len: 0,
            guest_base: 0,
            dtor: Box::new(|_, _| {}),
            protect: None,
        };
        assert!(matches!(
            Memory::from_host_ram(
                MemoryModel::Reserved {
                    span: (1 << 47) + 1
                },
                ram
            ),
            Err(HostRamError::NonCanonicalSpan)
        ));
    }

    fn reserved(span: u64) -> Memory {
        Memory::new(MemoryModel::Reserved { span })
    }

    /// A `Reserved` `Memory` over a caller-owned buffer, exercising the embedder
    /// (`from_host_ram`) path without a real host mmap: leak a `Box` and hand its
    /// raw region in with a dtor that reclaims it. (The huge-span NORESERVE sparsity
    /// proof lives in the embedder crate, where the mmap does.)
    fn reserved_host(span: usize) -> Memory {
        reserved_host_based(span, 0)
    }

    /// Like [`reserved_host`] but the leaked buffer represents guest `[guest_base,
    /// guest_base + extent)` (the identity-mapping layout, without a real mmap): the
    /// backing is `extent = span - guest_base` bytes and the model span is `span`.
    fn reserved_host_based(span: usize, guest_base: u64) -> Memory {
        let extent = span - guest_base as usize;
        let buf = vec![0u8; extent].into_boxed_slice();
        let ptr = Box::into_raw(buf) as *mut u8;
        let ram = HostRam {
            ptr,
            len: extent,
            guest_base,
            dtor: Box::new(|p, l| {
                // SAFETY: reconstruct exactly the boxed slice we leaked, then drop it.
                let slice = unsafe { std::slice::from_raw_parts_mut(p, l) };
                drop(unsafe { Box::from_raw(slice as *mut [u8]) });
            }),
            protect: None,
        };
        Memory::from_host_ram(MemoryModel::Reserved { span: span as u64 }, ram).unwrap()
    }

    #[test]
    fn highest_mapped_below_clamps_a_straddling_region() {
        let mut m = flat(0x10000);
        assert_eq!(m.highest_mapped_below(0x8000), 0, "nothing mapped → 0");
        // A region wholly below the limit contributes its exclusive end.
        m.map(0x1000, 0x1000, Prot::RW, RegionKind::Ram).unwrap(); // [0x1000, 0x2000)
        assert_eq!(m.highest_mapped_below(0x8000), 0x2000);
        // A region that STRADDLES the limit (starts below, ends above) must not push
        // the result past the limit — it is clamped to the limit (the bug: it returned
        // start+size = 0x6000 > limit).
        m.map(0x3000, 0x3000, Prot::RW, RegionKind::Ram).unwrap(); // [0x3000, 0x6000)
        assert_eq!(
            m.highest_mapped_below(0x5000),
            0x5000,
            "a straddling region clamps to the limit, never above it"
        );
        // A region starting at/above the limit is excluded entirely.
        assert_eq!(m.highest_mapped_below(0x3000), 0x2000);
    }

    /// Recorded `(guest_page_off, len, accessible)` guard-page calls.
    type ProtectLog = std::sync::Arc<std::sync::Mutex<Vec<(u64, usize, bool)>>>;

    /// A host-backed `Memory` whose `protect` hook records `(guest_page_off, len,
    /// accessible)` — for testing the guard-page rounding without a real `mprotect`.
    /// Leaks a `Box`, reclaimed by the dtor on drop.
    fn recording_host(span: usize) -> (Memory, ProtectLog) {
        use std::sync::{Arc, Mutex};
        let buf = vec![0u8; span].into_boxed_slice();
        let base = Box::into_raw(buf) as *mut u8;
        let base_addr = base as usize;
        let log: Arc<Mutex<Vec<(u64, usize, bool)>>> = Arc::new(Mutex::new(Vec::new()));
        let rec = Arc::clone(&log);
        let ram = HostRam {
            ptr: base,
            len: span,
            guest_base: 0,
            dtor: Box::new(|p, l| {
                let slice = unsafe { std::slice::from_raw_parts_mut(p, l) };
                drop(unsafe { Box::from_raw(slice as *mut [u8]) });
            }),
            protect: Some(Box::new(move |page_ptr, len, accessible| {
                rec.lock()
                    .unwrap()
                    .push(((page_ptr as usize - base_addr) as u64, len, accessible));
            })),
        };
        (
            Memory::from_host_ram(MemoryModel::Reserved { span: span as u64 }, ram).unwrap(),
            log,
        )
    }

    #[test]
    fn guard_pages_map_opens_and_unmap_closes_region_pages() {
        let (mut m, log) = recording_host(0x10000);
        // A page-aligned region → exactly its one page opened.
        m.map(0x2000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
        m.unmap(0x2000, 0x1000).unwrap();
        assert_eq!(
            *log.lock().unwrap(),
            vec![(0x2000, 0x1000, true), (0x2000, 0x1000, false)]
        );
    }

    #[test]
    fn guard_pages_shared_edge_page_stays_open_until_last_region_unmaps() {
        let (mut m, log) = recording_host(0x10000);
        // Two sub-page regions sharing host page 0x2000.
        m.map(0x2000, 0x400, Prot::RW, RegionKind::Ram).unwrap(); // [0x2000,0x2400)
        m.map(0x2400, 0x400, Prot::RW, RegionKind::Ram).unwrap(); // [0x2400,0x2800)
        log.lock().unwrap().clear();
        // Unmapping the first must NOT close page 0x2000 — the second still lives there.
        m.unmap(0x2000, 0x400).unwrap();
        assert!(
            log.lock().unwrap().is_empty(),
            "a page shared with a surviving region must stay accessible"
        );
        // Unmapping the last closes it.
        m.unmap(0x2400, 0x400).unwrap();
        assert_eq!(*log.lock().unwrap(), vec![(0x2000, 0x1000, false)]);
    }

    #[test]
    fn reserved_maps_and_roundtrips_like_flat() {
        let mut m = reserved(1 << 30); // 1 GiB span
        m.map(0x1000, 0x2000, Prot::RW, RegionKind::Ram).unwrap();
        m.write(0x1100, 0x1122_3344_5566_7788, 8).unwrap();
        assert_eq!(m.read(0x1100, 8).unwrap(), 0x1122_3344_5566_7788);
        assert!(matches!(m.read(0x50, 4), Err(MemTrap::Unmapped))); // unmapped gap
    }

    #[test]
    fn from_host_ram_backs_reserved() {
        // The embedder-backing path: map, write, read on host-provided RAM, and the
        // dtor runs on drop (no leak — miri/leak-checker would catch a bad dtor).
        let mut m = reserved_host(0x10000);
        m.map(0x1000, 0x2000, Prot::RW, RegionKind::Ram).unwrap();
        m.write(0x1040, 0xfeed_face, 8).unwrap();
        assert_eq!(m.read(0x1040, 8).unwrap(), 0xfeed_face);
        assert!(matches!(m.read(0x40, 4), Err(MemTrap::Unmapped)));
    }

    #[test]
    fn host_ram_ptr_targets_host_mapped_ram_and_skips_vec_and_unmapped() {
        // `host_ram_ptr` returns the backing pointer only for a range wholly inside a
        // mapped RAM region *and* a real host mapping — the gate the embedder's
        // `madvise(MADV_DONTNEED)` passthrough uses to decide host-madvise vs write-zero.

        // Host-mapped (Owner::Host) RAM: a mapped, in-region range resolves to the backing.
        let mut host = reserved_host(0x10000);
        host.map(0x1000, 0x2000, Prot::RW, RegionKind::Ram).unwrap();
        let base = host.host_base() as usize;
        let p = host
            .host_ram_ptr(0x1000, 0x1000)
            .expect("host-mapped RAM range resolves");
        assert_eq!(
            p as usize,
            base + 0x1000,
            "host = base + (addr - guest_base)"
        );
        // A sub-range deeper into the region translates by its offset.
        let p2 = host
            .host_ram_ptr(0x1500, 0x400)
            .expect("sub-range resolves");
        assert_eq!(p2 as usize, base + 0x1500);
        // A range escaping the mapped region → None (write-zero fallback territory).
        assert!(
            host.host_ram_ptr(0x40, 4).is_none(),
            "unmapped range → None"
        );
        assert!(
            host.host_ram_ptr(0x2800, 0x1000).is_none(),
            "range spilling past the region end → None"
        );

        // Vec-backed (Owner::Boxed) RAM: mapped and readable, but NO host mapping to
        // madvise, so `host_ram_ptr` returns None → embedder keeps the write-zero path.
        let mut vecbacked = reserved(1 << 20);
        vecbacked
            .map(0x1000, 0x2000, Prot::RW, RegionKind::Ram)
            .unwrap();
        assert!(
            vecbacked.host_ram_ptr(0x1000, 0x1000).is_none(),
            "a Vec-backed region has no host mapping — None even though it's mapped RAM"
        );
    }

    #[test]
    fn host_ram_ptr_rejects_trap_regions() {
        // A Trap/MMIO region is host-mapped storage-wise but must not be madvised as RAM.
        let mut host = reserved_host(0x10000);
        host.map(0x1000, 0x1000, Prot::RW, RegionKind::Trap)
            .unwrap();
        assert!(
            host.host_ram_ptr(0x1000, 0x100).is_none(),
            "a Trap region is not RAM — None"
        );
    }

    #[test]
    fn guest_base_translates_and_rejects_below_base() {
        // Guest space `[0x10000, 0x20000)` over a 0x10000-byte backing at guest_base.
        let base = 0x10000u64;
        let mut m = reserved_host_based(0x20000, base);
        assert_eq!(m.guest_base(), base);
        assert_eq!(
            m.size(),
            0x20000,
            "size() is the exclusive top guest address"
        );

        // A region below the base is rejected (no backing exists there).
        assert!(matches!(
            m.map(0x8000, 0x1000, Prot::RW, RegionKind::Ram),
            Err(MapError::OutOfBounds)
        ));

        // Map at guest 0x11000, write/read: the value lands at backing offset 0x1000
        // (`addr - guest_base`), not 0x11000.
        m.map(0x11000, 0x2000, Prot::RW, RegionKind::Ram).unwrap();
        m.write(0x11040, 0x1122_3344_5566_7788, 8).unwrap();
        assert_eq!(m.read(0x11040, 8).unwrap(), 0x1122_3344_5566_7788);
        // The byte truly lives at backing offset 0x1040 (identity proof at the core).
        let mut probe = [0u8; 8];
        // SAFETY: read the raw backing offset directly (test-only introspection).
        unsafe {
            let hb = m.host_base().add(0x1040);
            std::ptr::copy_nonoverlapping(hb, probe.as_mut_ptr(), 8);
        }
        assert_eq!(u64::from_le_bytes(probe), 0x1122_3344_5566_7788);

        // Unmapped in-space gap still traps; below-base scalar access traps too.
        assert!(matches!(m.read(0x10500, 4), Err(MemTrap::Unmapped)));
        assert!(matches!(m.read(0x8000, 4), Err(MemTrap::Unmapped)));
        // Top-of-space straddle traps (0x1FFFE + 4 > 0x20000).
        assert!(matches!(m.read(0x1FFFE, 4), Err(MemTrap::Unmapped)));
    }

    #[test]
    fn guest_base_memory_is_not_deep_copyable() {
        // A non-zero-based memory can't be re-homed into a zero-based boxed child.
        let m = reserved_host_based(0x20000, 0x10000);
        assert!(m.deep_copy().is_none());
    }

    #[test]
    fn reserved_deep_copy_is_independent() {
        let mut parent = reserved(1 << 30);
        parent
            .map(0x1000, 0x1000, Prot::RW, RegionKind::Ram)
            .unwrap();
        parent.write(0x1000, 0xaa, 8).unwrap();
        let child = parent.deep_copy().expect("owned Reserved is deep-copyable");
        // Child sees the copied bytes...
        assert_eq!(child.read(0x1000, 8).unwrap(), 0xaa);
        // ...but writes don't cross over.
        parent.write(0x1000, 0xbb, 8).unwrap();
        assert_eq!(parent.read(0x1000, 8).unwrap(), 0xbb);
        assert_eq!(child.read(0x1000, 8).unwrap(), 0xaa);
    }

    #[test]
    fn map_within_bounds_ok() {
        let mut m = flat(0x1000);
        assert!(m.map(0x100, 0x200, Prot::RW, RegionKind::Ram).is_ok());
    }

    #[test]
    fn map_past_end_is_out_of_bounds() {
        let mut m = flat(0x1000);
        assert!(matches!(
            m.map(0xF00, 0x200, Prot::RW, RegionKind::Ram),
            Err(MapError::OutOfBounds)
        ));
    }

    #[test]
    fn map_overflowing_end_is_out_of_bounds() {
        let mut m = flat(0x1000);
        assert!(matches!(
            m.map(u64::MAX, 0x10, Prot::RW, RegionKind::Ram),
            Err(MapError::OutOfBounds)
        ));
    }

    #[test]
    fn overlapping_map_rejected() {
        let mut m = flat(0x1000);
        m.map(0x100, 0x200, Prot::RW, RegionKind::Ram).unwrap();
        assert!(matches!(
            m.map(0x200, 0x100, Prot::RW, RegionKind::Ram),
            Err(MapError::Overlap)
        ));
    }

    #[test]
    fn adjacent_maps_allowed() {
        let mut m = flat(0x1000);
        m.map(0x100, 0x100, Prot::RW, RegionKind::Ram).unwrap();
        assert!(m.map(0x200, 0x100, Prot::RW, RegionKind::Ram).is_ok());
    }

    #[test]
    fn write_then_read_roundtrips() {
        let mut m = flat(0x1000);
        m.map(0x100, 0x200, Prot::RW, RegionKind::Ram).unwrap();
        m.write_bytes(0x110, &[1, 2, 3, 4]).unwrap();
        let mut buf = [0u8; 4];
        m.read_bytes(0x110, &mut buf).unwrap();
        assert_eq!(buf, [1, 2, 3, 4]);
    }

    #[test]
    fn write_outside_mapped_region_is_unmapped() {
        let mut m = flat(0x1000);
        m.map(0x100, 0x10, Prot::RW, RegionKind::Ram).unwrap();
        assert!(matches!(
            m.write_bytes(0x108, &[0; 0x10]),
            Err(MemError::Unmapped)
        ));
    }

    #[test]
    fn read_unmapped_is_unmapped() {
        let m = flat(0x1000);
        let mut buf = [0u8; 4];
        assert!(matches!(
            m.read_bytes(0x100, &mut buf),
            Err(MemError::Unmapped)
        ));
    }

    #[test]
    fn unmap_then_access_is_unmapped() {
        let mut m = flat(0x1000);
        m.map(0x100, 0x200, Prot::RW, RegionKind::Ram).unwrap();
        m.unmap(0x100, 0x200).unwrap();
        let mut buf = [0u8; 4];
        assert!(matches!(
            m.read_bytes(0x110, &mut buf),
            Err(MemError::Unmapped)
        ));
    }

    #[test]
    fn unmap_must_match_a_region_exactly() {
        let mut m = flat(0x1000);
        m.map(0x100, 0x200, Prot::RW, RegionKind::Ram).unwrap();
        assert!(matches!(m.unmap(0x100, 0x100), Err(MapError::OutOfBounds)));
    }

    #[test]
    fn unmap_frees_the_range_for_remapping() {
        let mut m = flat(0x1000);
        m.map(0x100, 0x200, Prot::RW, RegionKind::Ram).unwrap();
        m.unmap(0x100, 0x200).unwrap();
        assert!(m.map(0x100, 0x200, Prot::RX, RegionKind::Ram).is_ok());
    }

    #[test]
    fn scalar_write_read_little_endian() {
        let mut m = flat(0x1000);
        m.map(0x100, 0x100, Prot::RW, RegionKind::Ram).unwrap();
        m.write(0x110, 0x1122_3344_5566_7788, 8).unwrap();
        assert_eq!(m.read(0x110, 8).unwrap(), 0x1122_3344_5566_7788);
        // Sub-word reads see the low bytes (LE).
        assert_eq!(m.read(0x110, 1).unwrap(), 0x88);
        assert_eq!(m.read(0x110, 2).unwrap(), 0x7788);
        assert_eq!(m.read(0x110, 4).unwrap(), 0x5566_7788);
        // Individual bytes in memory, low byte first.
        let mut raw = [0u8; 8];
        m.read_bytes(0x110, &mut raw).unwrap();
        assert_eq!(raw, [0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11]);
    }

    #[test]
    fn scalar_access_outside_region_traps_unmapped() {
        let m = flat(0x1000);
        assert!(matches!(m.read(0x10, 4), Err(MemTrap::Unmapped)));
        assert!(matches!(m.write(0x10, 0, 4), Err(MemTrap::Unmapped)));
    }

    #[test]
    fn scalar_access_straddling_region_end_traps() {
        let mut m = flat(0x1000);
        m.map(0x100, 0x10, Prot::RW, RegionKind::Ram).unwrap();
        // Last valid 4-byte read starts at 0x10c; 0x10e straddles the end.
        assert!(m.read(0x10c, 4).is_ok());
        assert!(matches!(m.read(0x10e, 4), Err(MemTrap::Unmapped)));
    }

    #[test]
    fn atomic_operand_crossing_into_gap_stays_unmapped() {
        let mut m = flat(0x1000);
        m.map(0x100, 0x10, Prot::RW, RegionKind::Ram).unwrap();
        m.write(0x10c, 7, 4).unwrap();
        assert_eq!(
            m.atomic_rmw(0x10e, 9, 4, RmwOp::Add),
            Err(MemTrap::Unmapped)
        );
        assert_eq!(m.atomic_cas(0x10e, 7, 9, 4), Err(MemTrap::Unmapped));
        assert_eq!(m.read(0x10c, 4), Ok(7));
    }

    // xj-0w3.8: a misaligned locked operand at every offset of a 16-byte
    // granule, inside it (the containing-word swap) and across it (the plain
    // fallback), routed and not. The operand's result is the instruction's,
    // and every byte outside the operand keeps its value; a failed compare
    // changes nothing (SDM Vol 2A CMPXCHG p. 3-194).
    #[test]
    fn locked_unaligned_operands_keep_their_neighbours() {
        let pattern: [u8; 48] = std::array::from_fn(|i| 0x31u8.wrapping_mul(i as u8 + 7));
        for routed in [false, true] {
            let mut m = flat(0x1000);
            m.map(0x100, 0x40, Prot::RW, RegionKind::Ram).unwrap();
            m.set_host_fault_recovery(routed);
            let read = |m: &Memory| {
                let mut b = [0u8; 48];
                m.read_ram_checked(0x100, &mut b).unwrap();
                b
            };
            for size in [2u8, 4, 8] {
                for at in 0x111..0x120u64 {
                    if at % u64::from(size) == 0 {
                        continue;
                    }
                    let bytes = (at - 0x100) as usize..(at - 0x100) as usize + size as usize;
                    let mut v = [0u8; 8];
                    v[..size as usize].copy_from_slice(&pattern[bytes.clone()]);
                    let old = u64::from_le_bytes(v);
                    let with = |v: u64| {
                        let mut b = pattern;
                        b[bytes.clone()].copy_from_slice(&v.to_le_bytes()[..size as usize]);
                        b
                    };
                    let src = 0x0123_4567_89ab_cdef;
                    for op in [RmwOp::Add, RmwOp::Rsub, RmwOp::Xor, RmwOp::Xchg] {
                        m.write_ram_checked(0x100, &pattern).unwrap();
                        assert_eq!(m.atomic_rmw(at, src, size, op), Ok(old), "{op:?} {at:#x}");
                        let new = apply_rmw(old, src, op, size);
                        assert_eq!(read(&m), with(new), "{op:?} {size} {at:#x}");
                    }
                    m.write_ram_checked(0x100, &pattern).unwrap();
                    assert_eq!(m.atomic_cas(at, old ^ 1, src, size), Ok(old));
                    assert_eq!(read(&m), pattern, "failed compare {size} {at:#x}");
                    assert_eq!(m.atomic_cas(at, old | !mask_bits(size), src, size), Ok(old));
                    assert_eq!(read(&m), with(src & mask_bits(size)), "{size} {at:#x}");
                }
            }
        }
    }

    // xj-0w3.8: a containing word that would extend past a backing whose length
    // is not a multiple of 16 is never touched; the operand takes the plain
    // fallback, with the same result.
    #[test]
    fn a_granule_past_the_backing_takes_the_plain_fallback() {
        for routed in [false, true] {
            let mut m = flat(0x1009);
            m.map(0x1000, 9, Prot::RW, RegionKind::Ram).unwrap();
            m.set_host_fault_recovery(routed);
            m.write_ram_checked(0x1000, &[1, 2, 3, 4, 5, 6, 7, 8, 9])
                .unwrap();
            assert_eq!(m.atomic_rmw(0x1005, 0x10, 4, RmwOp::Add), Ok(0x0908_0706));
            assert_eq!(m.atomic_cas(0x1007, 0x0908, 0x5555, 2), Ok(0x0908));
            let mut b = [0; 9];
            m.read_ram_checked(0x1000, &mut b).unwrap();
            assert_eq!(b, [1, 2, 3, 4, 5, 0x16, 7, 0x55, 0x55]);
        }
    }

    // xj-0w3.8: concurrent locked updates of one misaligned field inside a
    // granule, against a second locked update of the same bytes and plain
    // stores to a neighbour byte of the containing word, lose nothing (SDM
    // Vol 3A §11.1.2.2 p. 11-4). Both adders add to the field's low byte and
    // the sum stays below 2^16, so no carry leaves the field and the order of
    // the updates does not matter.
    #[test]
    fn locked_unaligned_updates_are_atomic_across_threads() {
        const N: u64 = 20_000;
        // (field offset, size; second adder's offset, size; neighbour offset)
        let rows = [
            (1u64, 2u8, 0u64, 4u8, 5u64),
            (7, 2, 6, 4, 12),
            (6, 4, 4, 4, 3),
            (4, 8, 4, 4, 2),
        ];
        for (field, size, other, other_size, neighbour) in rows {
            for routed in [false, true] {
                let mut m = flat(0x1000);
                m.map(0x100, 0x40, Prot::RW, RegionKind::Ram).unwrap();
                m.set_host_fault_recovery(routed);
                let base = 0x110;
                let shift = 8 * (field - other);
                std::thread::scope(|s| {
                    s.spawn(|| {
                        for _ in 0..N {
                            m.atomic_rmw(base + field, 1, size, RmwOp::Add).unwrap();
                        }
                    });
                    s.spawn(|| {
                        for _ in 0..N {
                            m.atomic_rmw(base + other, 1 << shift, other_size, RmwOp::Add)
                                .unwrap();
                        }
                    });
                    s.spawn(|| {
                        for i in (1..=N).rev() {
                            m.write(base + neighbour, i & 0xff, 1).unwrap();
                        }
                    });
                });
                let tag = format!("field +{field} size {size} routed {routed}");
                assert_eq!(m.read(base + field, size), Ok(2 * N), "{tag}");
                assert_eq!(m.read(base + neighbour, 1), Ok(1), "{tag}");
            }
        }
    }

    #[test]
    fn adjacent_ram_regions_join_for_data_transfers() {
        let mut m = flat(0x1000);
        m.map(0x100, 0x10, Prot::RW, RegionKind::Ram).unwrap();
        m.map(0x110, 0x10, Prot::RW, RegionKind::Ram).unwrap();
        for size in [2u8, 4, 8] {
            for at in 0x111 - u64::from(size)..0x110 {
                m.write(at, 0x8877_6655_4433_2211, size).unwrap();
                assert_eq!(
                    m.read(at, size),
                    Ok(0x8877_6655_4433_2211 & mask_bits(size))
                );
                assert_eq!(m.probe_write(at, size), Ok(()));
            }
        }
        m.write_ram_checked(0x10b, &[7; 10]).unwrap();
        let mut buf = [0; 10];
        m.read_ram_checked(0x10b, &mut buf).unwrap();
        assert_eq!(buf, [7; 10]);
    }

    #[test]
    fn adjacent_rom_bytes_read_like_ram_and_drop_stores() {
        let mut m = flat(0x1000);
        m.map(0x100, 0x10, Prot::RW, RegionKind::Ram).unwrap();
        m.map(0x110, 0x10, Prot::RX, RegionKind::Rom).unwrap();
        m.map(0x120, 0x10, Prot::RW, RegionKind::Ram).unwrap();
        m.write_bytes(0x110, &[0xaa; 0x10]).unwrap();
        m.write(0x10e, 0x4433_2211, 4).unwrap();
        assert_eq!(m.read(0x10e, 4), Ok(0xaaaa_2211));
        m.write(0x11e, 0x4433_2211, 4).unwrap();
        assert_eq!(m.read(0x11e, 4), Ok(0x4433_aaaa));
        m.write_ram_checked(0x10c, &[1, 2, 3, 4, 5, 6, 7, 8])
            .unwrap();
        let mut buf = [0; 8];
        m.read_ram_checked(0x10c, &mut buf).unwrap();
        assert_eq!(buf, [1, 2, 3, 4, 0xaa, 0xaa, 0xaa, 0xaa]);
        // RAM, a whole ROM region, then RAM: only the RAM pieces change.
        let mut wide = [0x5a; 0x14];
        m.write_ram_checked(0x10e, &wide).unwrap();
        m.read_ram_checked(0x10e, &mut wide).unwrap();
        let mut expected = [0xaa; 0x14];
        expected[..2].fill(0x5a);
        expected[0x12..].fill(0x5a);
        assert_eq!(wide, expected);
    }

    #[test]
    fn gap_trap_and_atomic_spans_keep_the_containment_miss() {
        let mut m = flat(0x1000);
        m.map(0x100, 0x10, Prot::RW, RegionKind::Ram).unwrap();
        m.map(0x111, 0x10, Prot::RW, RegionKind::Ram).unwrap();
        m.map(0x130, 0x10, Prot::RW, RegionKind::Ram).unwrap();
        m.map(0x140, 0x10, Prot::RW, RegionKind::Trap).unwrap();
        m.map(0x200, 0x10, Prot::RW, RegionKind::Ram).unwrap();
        m.map(0x210, 0x10, Prot::RW, RegionKind::Ram).unwrap();
        // 0x10e covers the gap byte 0x110; 0x13e covers the Trap byte 0x140.
        for at in [0x10e, 0x13e] {
            assert_eq!(m.read(at, 4), Err(MemTrap::Unmapped));
            assert_eq!(m.write(at, u64::MAX, 4), Err(MemTrap::Unmapped));
            assert_eq!(m.probe_write(at, 4), Err(MemTrap::Unmapped));
            assert!(m.read_ram_checked(at, &mut [0; 4]).is_err());
            assert!(m.write_ram_checked(at, &[0xff; 4]).is_err());
            assert_eq!(m.read(at, 2), Ok(0), "nothing written at {at:#x}");
        }
        assert_eq!(
            m.atomic_rmw(0x20e, 9, 4, RmwOp::Add),
            Err(MemTrap::Unmapped)
        );
        assert_eq!(m.atomic_cas(0x20e, 0, 9, 4), Err(MemTrap::Unmapped));
        assert_eq!(m.read(0x20e, 4), Ok(0));
    }

    #[test]
    fn trap_region_routes_to_mmio() {
        let mut m = flat(0x1000);
        m.map(0x200, 0x10, Prot::RW, RegionKind::Trap).unwrap();
        assert!(matches!(m.read(0x200, 4), Err(MemTrap::Mmio)));
        assert!(matches!(m.write(0x200, 0, 4), Err(MemTrap::Mmio)));
    }

    #[test]
    fn trap_intervals_track_mapping_clone_and_unmap() {
        let mut m = flat(0x1000);
        m.map(0x600, 0x10, Prot::RW, RegionKind::Trap).unwrap();
        m.map(0x200, 0x10, Prot::RW, RegionKind::Trap).unwrap();
        m.map(0x400, 0x10, Prot::RW, RegionKind::Ram).unwrap();
        m.map(0x800, 0, Prot::RW, RegionKind::Trap).unwrap();
        assert_eq!(m.trap_window(), Some((0x200, 0x610)));
        for (first, last, expected) in [
            (0, 0x1ff, false),
            (0x1ff, 0x200, true),
            (0x20f, 0x20f, true),
            (0x210, 0x5ff, false),
            (0x210, 0x600, true),
            (0x610, u64::MAX, false),
        ] {
            assert_eq!(m.intersects_trap(first, last), expected);
        }
        let copy = m.deep_copy().unwrap();
        assert!(copy.intersects_trap(0x200, 0x200));
        m.unmap(0x200, 0x10).unwrap();
        assert!(!m.intersects_trap(0x200, 0x200));
        assert_eq!(m.trap_window(), Some((0x600, 0x610)));
        m.unmap(0x600, 0x10).unwrap();
        assert!(m.trap_window().is_none());
        assert!(!m.intersects_trap(0, u64::MAX));
    }

    #[test]
    fn rom_reads_and_fetches_like_ram_but_drops_writes() {
        let mut m = flat(0x1000);
        m.map(0x200, 0x20, Prot::RX, RegionKind::Rom).unwrap();
        // ROM contents are loaded via the host loader path (bypasses region kind).
        m.write_bytes(0x200, &[0x90, 0x90, 0xc3, 0x11, 0x22, 0x33, 0x44, 0x55])
            .unwrap();
        // Reads see the loaded bytes at RAM speed (no MMIO trap-out).
        assert_eq!(m.read(0x200, 1).unwrap(), 0x90);
        assert_eq!(m.read(0x203, 4).unwrap(), 0x4433_2211);
        // Instruction fetch works: a decoder slice comes straight out of the region.
        let s = m.code_slice(0x200, 8).unwrap();
        assert_eq!(&s[..3], &[0x90, 0x90, 0xc3]);
        // A guest store is silently discarded — no fault, backing unchanged.
        assert!(
            m.write(0x203, 0xdead_beef, 4).is_ok(),
            "ROM write must not fault"
        );
        assert_eq!(
            m.read(0x203, 4).unwrap(),
            0x4433_2211,
            "ROM store was dropped"
        );
        // A locked RMW on ROM returns the current value without modifying it.
        assert_eq!(
            m.atomic_rmw(0x203, 0xff, 4, RmwOp::Add).unwrap(),
            0x4433_2211
        );
        assert_eq!(
            m.read(0x203, 4).unwrap(),
            0x4433_2211,
            "ROM RMW was dropped"
        );
        // A cmpxchg on ROM never stores, even on a matching compare.
        assert_eq!(m.atomic_cas(0x203, 0x4433_2211, 0, 4).unwrap(), 0x4433_2211);
        assert_eq!(
            m.read(0x203, 4).unwrap(),
            0x4433_2211,
            "ROM CAS was dropped"
        );
    }

    #[test]
    fn rom_is_excluded_from_the_trap_window() {
        // ROM stays inlined (RAM-speed reads/fetch): it must NOT widen the JIT's
        // trap-range check, so a memory with only a Rom region has no trap window.
        let mut m = flat(0x1000);
        m.map(0x200, 0x20, Prot::RX, RegionKind::Rom).unwrap();
        assert!(m.trap_window().is_none(), "ROM is not an MMIO trap region");
        // Adding an actual Trap region produces a window bounded by the Trap only.
        m.map(0x400, 0x10, Prot::RW, RegionKind::Trap).unwrap();
        assert_eq!(m.trap_window(), Some((0x400, 0x410)));
    }

    #[test]
    fn code_slice_caps_at_region_end() {
        let mut m = flat(0x1000);
        m.map(0x100, 0x8, Prot::RX, RegionKind::Ram).unwrap();
        m.write_bytes(0x100, &[0x90, 0x90, 0x90, 0xc3, 0, 0, 0, 0])
            .unwrap();
        // Asking for more than the region holds caps at its end (8 bytes).
        let s = m.code_slice(0x100, 64).unwrap();
        assert_eq!(s.len(), 8);
        assert_eq!(&s[..4], &[0x90, 0x90, 0x90, 0xc3]);
        // Mid-region start shortens accordingly.
        assert_eq!(m.code_slice(0x104, 64).unwrap().len(), 4);
        assert!(matches!(m.code_slice(0x50, 4), Err(MemTrap::Unmapped)));
    }

    /// A `Reserved` memory whose backing is only `len` bytes but represents guest
    /// `[guest_base, guest_base + len)` — an identity-mapped arena high in the address
    /// space, without allocating everything below it.
    fn high_base_mem(guest_base: u64, len: usize) -> Memory {
        let mut buf = vec![0u8; len].into_boxed_slice();
        let (ptr, len) = (buf.as_mut_ptr(), buf.len());
        std::mem::forget(buf);
        let ram = HostRam {
            ptr,
            len,
            guest_base,
            dtor: Box::new(|p, l| {
                // SAFETY: `p`/`l` are the leaked `Box<[u8]>` allocated just above.
                drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(p, l)) });
            }),
            protect: None,
        };
        Memory::from_host_ram(
            MemoryModel::Reserved {
                span: guest_base + len as u64,
            },
            ram,
        )
        .unwrap()
    }

    #[test]
    fn smc_47bit_top_uses_backing_relative_bitmaps() {
        // SDM Vol 1 §3.3.7.1: the lower 48-bit canonical half ends at 2^47.
        const TOP: u64 = 1 << 47;
        const LEN: usize = 2 << 20;
        let base = TOP - LEN as u64;
        let mut m = high_base_mem(base, LEN);
        m.map(base, LEN, Prot::RWX, RegionKind::Ram).unwrap();
        assert!(is_canonical(TOP - 1));
        assert!(!is_canonical(TOP));
        for table in [&*m.code_page, &*m.watch.watch, &*m.watch.dirty] {
            assert_eq!(table.len(), 8, "512 backed pages need eight bitmap words");
        }
        assert!(m.watch_bits_cover_size());
        let code = TOP - CODE_PAGE_SIZE;
        m.mark_code(code, 16);
        m.watch_range(code, CODE_PAGE_SIZE);
        m.write_bytes(code, &[0xaa]).unwrap();
        assert_eq!(m.take_dirty_code(), vec![code >> CODE_PAGE_BITS]);
        assert_eq!(m.take_dirty_ranges(), vec![(code, CODE_PAGE_SIZE)]);
        m.clear_all_code_pages();
        assert!(!m.is_code_page(code >> CODE_PAGE_BITS));
    }

    #[test]
    fn smc_47bit_partial_first_page_keeps_bitmap_indices_aligned() {
        const TOP: u64 = 1 << 47;
        let base = TOP - 0x40000 + 16;
        let mut m = high_base_mem(base, (TOP - base) as usize);
        m.map(base, (TOP - base) as usize, Prot::RWX, RegionKind::Ram)
            .unwrap();
        assert_eq!(m.code_page.len(), 1);
        assert!(m.watch_bits_cover_size());
        for addr in [base, TOP - 1] {
            m.mark_code(addr, 1);
            m.watch_range(addr, 1);
            m.write_bytes(addr, &[1]).unwrap();
            let page = addr >> CODE_PAGE_BITS;
            assert_eq!(m.take_dirty_code(), vec![page]);
            assert_eq!(
                m.take_dirty_ranges(),
                vec![(page << CODE_PAGE_BITS, CODE_PAGE_SIZE)]
            );
            m.clear_code_page(page);
            assert!(!m.is_code_page(page));
            m.unwatch_range(addr, 1);
        }
        assert_eq!(m.watch.count.load(Ordering::Relaxed), 0);
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn smc_47bit_zero_based_tables_stay_sparse_when_cleared() {
        const TOP: u64 = 1 << 47;
        // Exercise the full-span metadata without reserving 128 TiB of host RAM.
        // The backing is never accessed; mark/clear only touch the page tables.
        let mut m = Memory::new(MemoryModel::SoftMmu);
        let code_page = fresh_code_pages(0, TOP);
        let code_page = CodePages::new(code_page);
        m.code_page = code_page;
        m.watch = WatchPages::new(0, TOP);
        assert_eq!(m.code_page.len() * 8, 4usize << 30);
        m.clear_all_code_pages();
        m.mark_code(TOP - CODE_PAGE_SIZE, 1);
        m.watch.watch((TOP - CODE_PAGE_SIZE) >> CODE_PAGE_BITS);
        m.clear_all_code_pages();
        assert!(!m.is_code_page((TOP - CODE_PAGE_SIZE) >> CODE_PAGE_BITS));
        assert_eq!(m.watch.watch.len() * 8, 4usize << 30);
        assert_eq!(m.watch.dirty.len() * 8, 4usize << 30);
    }

    #[test]
    fn smc_watermark_granule_edges_are_conservative_and_exact() {
        let m = Memory::new(MemoryModel::Flat { size: 0x80000 });
        let code = 0x40000;
        m.mark_code(code, 1);
        let gate = |addr: u64| {
            let range = m.code_range.load(Ordering::Relaxed);
            (addr >> 16).wrapping_sub(range >> 32) < (range & 0xffff_ffff)
        };
        assert!(gate(code));
        assert!(
            gate(code - 1),
            "a wide store can cross into the first code page"
        );
        assert!(!gate(0x2ffff), "outside the preceding granule stays fast");
        assert!(!gate(0x50000), "outside the last granule stays fast");
        m.note_write(code - 1, 2);
        assert_eq!(m.take_dirty_code(), vec![code >> CODE_PAGE_BITS]);
        m.note_write(code + CODE_PAGE_SIZE, 1);
        assert!(
            m.take_dirty_code().is_empty(),
            "the helper still tests 4 KiB pages"
        );
        // Marking the previously skew-only granule must also cover its predecessor.
        m.mark_code(code - 0x10000, 1);
        assert!(gate(code - 0x10001));
        m.mark_code(0, 1);
        let first = m.code_range.load(Ordering::Relaxed);
        m.mark_code(0, 1);
        assert_eq!(m.code_range.load(Ordering::Relaxed), first);
        m.widen_code_range((1 << (47 - 16)) - 1);
        assert_eq!(m.code_range.load(Ordering::Relaxed), 1 << 31);
    }

    #[test]
    fn smc_47bit_concurrent_marks_never_narrow_the_store_gate() {
        const TOP: u64 = 1 << 47;
        let m = high_base_mem(TOP - (2 << 20), 2 << 20);
        let published = AtomicU64::new(0);
        let barrier = std::sync::Barrier::new(3);
        std::thread::scope(|scope| {
            for parity in 0..2 {
                let (m, published, barrier) = (&m, &published, &barrier);
                scope.spawn(move || {
                    barrier.wait();
                    for i in (parity..32).step_by(2) {
                        let addr = TOP - ((i + 1) << 16);
                        m.mark_code(addr, 1);
                        published.fetch_or(1 << i, Ordering::Release);
                        std::thread::yield_now();
                    }
                });
            }
            barrier.wait();
            let mut previous = 0;
            loop {
                let marked = published.load(Ordering::Acquire);
                let range = m.code_range.load(Ordering::Relaxed);
                let (lo, len) = (range >> 32, range & 0xffff_ffff);
                if previous != 0 {
                    let (old_lo, old_len) = (previous >> 32, previous & 0xffff_ffff);
                    assert!(lo <= old_lo && lo + len >= old_lo + old_len);
                }
                previous = range;
                for i in 0..32 {
                    if marked & (1 << i) != 0 {
                        let addr = TOP - ((i + 1) << 16);
                        assert!((addr >> 16).wrapping_sub(lo) < len);
                        assert!(((addr - 1) >> 16).wrapping_sub(lo) < len);
                        m.note_write(addr, 1);
                        assert!(m.take_dirty_code().contains(&(addr >> CODE_PAGE_BITS)));
                    }
                }
                if marked == u32::MAX as u64 {
                    break;
                }
            }
        });
    }

    // Embedder-registered watched data-range dirty tracking.
    fn watched_mem() -> Memory {
        let mut m = Memory::new(MemoryModel::Flat { size: 0x10_0000 });
        m.map(0x1000, 0xF000, Prot::RW, RegionKind::Ram).unwrap();
        m
    }

    #[test]
    fn unwatched_writes_record_nothing() {
        let m = watched_mem();
        m.write_bytes(0x2000, &[1, 2, 3, 4]).unwrap();
        assert!(
            m.take_dirty_ranges().is_empty(),
            "no range watched → nothing dirty"
        );
    }

    #[test]
    fn watched_write_is_drained_once() {
        let m = watched_mem();
        m.watch_range(0x2000, 16);
        m.write_bytes(0x2000, &[0xAA; 8]).unwrap();
        let d = m.take_dirty_ranges();
        assert_eq!(d, vec![(0x2000, CODE_PAGE_SIZE)], "watched page reported");
        assert!(m.take_dirty_ranges().is_empty(), "drained on read");
    }

    #[test]
    fn writes_outside_watch_are_ignored() {
        let m = watched_mem();
        m.watch_range(0x2000, 16); // watches only page 0x2000
        m.write_bytes(0x5000, &[1; 4]).unwrap(); // different page
        assert!(m.take_dirty_ranges().is_empty());
    }

    #[test]
    fn unwatch_turns_tracking_off() {
        let m = watched_mem();
        m.watch_range(0x2000, 16);
        m.unwatch_range(0x2000, 16);
        m.write_bytes(0x2000, &[1; 4]).unwrap();
        assert!(m.take_dirty_ranges().is_empty(), "unwatched → no record");
    }

    #[test]
    fn concurrent_watch_unwatch_keeps_dirty_word_scannable() {
        use std::sync::{atomic::AtomicBool, Barrier};
        use std::time::{Duration, Instant};

        // Different pages in one word: only page 2 is written or drained.
        // Keeping page 3 watched is the control: the word can never reach zero.
        for keep_word_nonzero in [true, false] {
            let m = watched_mem();
            if keep_word_nonzero {
                m.watch_range(0x3000, CODE_PAGE_SIZE);
            }
            let start = Barrier::new(2);
            let stop = AtomicBool::new(false);
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut lost = 0;
            let mut drains = 0;
            std::thread::scope(|s| {
                s.spawn(|| {
                    start.wait();
                    while !stop.load(Ordering::Relaxed) && Instant::now() < deadline {
                        m.watch_range(0x1000, CODE_PAGE_SIZE);
                        m.unwatch_range(0x1000, CODE_PAGE_SIZE);
                    }
                });
                start.wait();
                for i in 0..200_000 {
                    if i % 1024 == 0 && Instant::now() >= deadline {
                        break;
                    }
                    m.watch_range(0x2000, CODE_PAGE_SIZE);
                    m.note_write(0x2000, 1);
                    if m.take_dirty_ranges() != [(0x2000, CODE_PAGE_SIZE)] {
                        lost += 1;
                    }
                    m.unwatch_range(0x2000, CODE_PAGE_SIZE);
                    drains += 1;
                }
                stop.store(true, Ordering::Relaxed);
            });
            eprintln!("keep_word_nonzero={keep_word_nonzero}: {lost} lost / {drains} drains");
            assert!(
                drains > 0,
                "the stress probe must perform writes and drains"
            );
            assert_eq!(lost, 0, "keep_word_nonzero={keep_word_nonzero}");
        }
    }

    #[test]
    fn consecutive_watched_pages_coalesce() {
        let m = watched_mem();
        m.watch_range(0x2000, 2 * CODE_PAGE_SIZE); // pages 0x2000 and 0x3000
        m.write_bytes(0x2FFC, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap(); // spans both pages
        let d = m.take_dirty_ranges();
        assert_eq!(
            d,
            vec![(0x2000, 2 * CODE_PAGE_SIZE)],
            "two consecutive dirty pages coalesce into one range"
        );
    }

    // The watch table used to be sized with `fresh_code_pages`, capping it at
    // CODE_WINDOW (4 GiB). Registering above that was silently dropped, so a high-heap
    // buffer — the whole point of the facility — never reported dirty.
    #[test]
    fn watch_works_above_the_4gib_code_window() {
        const HIGH: u64 = 41 << 30; // where a downstream embedder's GPU buffers live
                                    // The probe is only meaningful beyond the code window.
        const _: () = assert!(HIGH > CODE_WINDOW);
        // A SMALL backing that represents guest `[HIGH, HIGH + 0x2000)` — the shape an
        // embedder gets from an identity-mapped high arena. Building this as a
        // `Reserved` span of 41 GiB instead would try to allocate 41 GiB of backing,
        // which a CI runner refuses outright (and is not what the embedder does).
        // It also exercises the non-zero `guest_base` indexing the table is sized for.
        let mut m = high_base_mem(HIGH, 0x2000);
        m.map(HIGH, 0x2000, Prot::RW, RegionKind::Ram).unwrap();
        m.watch_range(HIGH, 0x1000);
        m.write_bytes(HIGH, &[0xAA; 8]).unwrap();
        assert_eq!(
            m.take_dirty_ranges(),
            vec![(HIGH, CODE_PAGE_SIZE)],
            "a watched range above CODE_WINDOW must still report dirty"
        );
    }

    // The table spans the whole guest address space, so it MUST stay sparse —
    // `zeroed_words` exists to keep the allocator's lazy commit. A 1 TiB span needs
    // 2 x 32 MiB of virtual address space; touching a handful of pages must not make
    // that resident.
    #[cfg(target_os = "linux")]
    #[test]
    fn watch_table_over_a_huge_span_stays_sparse() {
        fn rss_bytes() -> u64 {
            let s = std::fs::read_to_string("/proc/self/statm").unwrap();
            let pages: u64 = s.split_whitespace().nth(1).unwrap().parse().unwrap();
            pages * 4096
        }
        // A 1 TiB span: 268M pages -> 4.2M words -> 33.5 MiB per bitset.
        const SPAN: u64 = 1 << 40;
        let before = rss_bytes();
        let w = WatchPages::new(0, SPAN);
        for p in 0..64 {
            w.watch((SPAN >> 1 >> CODE_PAGE_BITS) + p);
        }
        let grew = rss_bytes().saturating_sub(before);
        // Two 33.5 MiB tables; only the touched words should be resident. Allow a
        // generous margin for allocator slack and the test harness itself.
        assert!(
            grew < 4 << 20,
            "watch tables for a 1 TiB span made {grew} bytes resident — the lazy \
             zero-page commit was defeated (element-wise init?)"
        );
        assert_eq!(w.take_dirty().len(), 0);
    }

    #[test]
    fn rewatching_a_page_does_not_double_count() {
        let m = watched_mem();
        m.watch_range(0x2000, 16);
        m.watch_range(0x2000, 16); // idempotent
        m.unwatch_range(0x2000, 16);
        m.write_bytes(0x2000, &[1; 4]).unwrap();
        assert!(
            m.take_dirty_ranges().is_empty(),
            "one unwatch must undo one watch, however many times it was registered"
        );
    }

    // A page dirtied and then unwatched must not surface later, nor come back stale
    // when the range is watched again.
    #[test]
    fn unwatch_drops_a_pending_dirty_bit() {
        let m = watched_mem();
        m.watch_range(0x2000, 16);
        m.write_bytes(0x2000, &[1; 4]).unwrap(); // page is now dirty, undrained
        m.unwatch_range(0x2000, 16);
        assert!(
            m.take_dirty_ranges().is_empty(),
            "unwatched page not reported"
        );
        m.watch_range(0x2000, 16);
        assert!(
            m.take_dirty_ranges().is_empty(),
            "re-watching must not resurrect the pre-unwatch dirty bit"
        );
    }

    // Several pages inside ONE bitset word (a word covers 64 pages) must all be
    // reported — the bit-packing must not lose any of them.
    #[test]
    fn many_pages_within_one_bitset_word_all_report() {
        let mut m = Memory::new(MemoryModel::Flat { size: 0x100_0000 });
        m.map(0x0, 0x100_0000, Prot::RW, RegionKind::Ram).unwrap();
        let pages: [u64; 4] = [3, 17, 40, 63]; // all in word 0
        for p in pages {
            m.watch_range(p << CODE_PAGE_BITS, 16);
        }
        for p in pages {
            m.write_bytes(p << CODE_PAGE_BITS, &[1; 4]).unwrap();
        }
        let got: Vec<u64> = m.take_dirty_ranges().iter().map(|&(a, _)| a).collect();
        assert_eq!(
            got,
            pages
                .iter()
                .map(|p| p << CODE_PAGE_BITS)
                .collect::<Vec<_>>(),
            "every dirty page in a shared word is reported, ascending"
        );
    }

    #[test]
    fn non_adjacent_watched_pages_stay_separate() {
        let m = watched_mem();
        m.watch_range(0x2000, 16);
        m.watch_range(0x6000, 16);
        m.write_bytes(0x2000, &[1; 4]).unwrap();
        m.write_bytes(0x6000, &[1; 4]).unwrap();
        assert_eq!(
            m.take_dirty_ranges(),
            vec![(0x2000, CODE_PAGE_SIZE), (0x6000, CODE_PAGE_SIZE)],
        );
    }

    /// Re-marking code must not accumulate the conservative low skew.
    #[test]
    fn re_marking_a_code_page_converges() {
        for page in [0u64, 1, 7, 16, 31] {
            let m = Memory::new(MemoryModel::Flat { size: 0x20_000 });
            let addr = page << CODE_PAGE_BITS;
            m.mark_code(addr, 16);
            let first = m.code_range.load(Ordering::Relaxed);
            for _ in 0..8 {
                m.mark_code(addr, 16);
            }
            assert_eq!(
                m.code_range.load(Ordering::Relaxed),
                first,
                "page {page}: re-marking widened the range from {first:#x}"
            );
        }
    }
}

/// Host protection granule, shared by recovery routines and compiled crossing checks.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "freebsd"))]
pub fn host_page_size() -> usize {
    #[cfg(test)]
    if let Some(page) = TEST_HOST_PAGE_SIZE.with(Cell::get) {
        return page;
    }
    static PAGE: AtomicUsize = AtomicUsize::new(0);
    let cached = PAGE.load(Ordering::Relaxed);
    if cached != 0 {
        return cached;
    }
    unsafe extern "C" {
        fn getpagesize() -> std::ffi::c_int;
    }
    // SAFETY: POSIX getpagesize has no pointer arguments; initialization precedes copying.
    let page = unsafe { getpagesize() } as usize;
    PAGE.store(page, Ordering::Relaxed);
    page
}
/// Host protection granule on platforms without host-fault recovery.
#[cfg(target_os = "windows")]
pub fn host_page_size() -> usize {
    #[cfg(test)]
    if let Some(page) = TEST_HOST_PAGE_SIZE.with(Cell::get) {
        return page;
    }
    #[repr(C)]
    struct SystemInfo {
        architecture: u32,
        page_size: u32,
        min_address: *mut std::ffi::c_void,
        max_address: *mut std::ffi::c_void,
        processor_mask: usize,
        processors: u32,
        processor_type: u32,
        allocation_granularity: u32,
        processor_level: u16,
        processor_revision: u16,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetSystemInfo(info: *mut SystemInfo);
    }
    let mut info = std::mem::MaybeUninit::<SystemInfo>::uninit();
    // SAFETY: GetSystemInfo fills the complete SYSTEM_INFO structure.
    unsafe {
        GetSystemInfo(info.as_mut_ptr());
        info.assume_init().page_size as usize
    }
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "freebsd",
    target_os = "windows"
)))]
pub fn host_page_size() -> usize {
    #[cfg(test)]
    if let Some(page) = TEST_HOST_PAGE_SIZE.with(Cell::get) {
        return page;
    }
    4096
}

#[cfg(test)]
std::thread_local! {
    static TEST_HOST_PAGE_SIZE: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Only for simulated faults over writable backing; OS protections still use real pages.
#[cfg(test)]
pub(crate) fn with_host_page_size_for_test<T>(page: usize, run: impl FnOnce() -> T) -> T {
    assert!(page.is_power_of_two() && page >= 4096);
    struct Reset(Option<usize>);
    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_HOST_PAGE_SIZE.with(|value| value.set(self.0));
        }
    }
    let _reset = Reset(TEST_HOST_PAGE_SIZE.with(|value| value.replace(Some(page))));
    run()
}

// A borrowed specialization retains its backing's representation, avoiding a
// pointer-to-pointer on normal out-of-line interpreter helpers.
#[repr(transparent)]
pub(crate) struct Access<const ROUTED: bool, const ORDER: u8 = 0, M = Memory>(pub M);

pub(crate) trait InterpreterMemory {
    /// Whether this provider's scalar transfers perform a containment miss over
    /// adjacent RAM/ROM regions (`Memory` and [`SplitMemory`]). Only such a provider
    /// compiles the 16-byte vector-store path that checks both transfers before
    /// either is written (`interp::vstore16_split`); it then also overrides
    /// [`Self::write_preflighted_contained`].
    const SPLIT: bool = false;
    fn normal(&self) -> &Memory;
    fn note_block_write(&self, _addr: u64, _len: usize) {}
    fn note_host_block_write(&self, _host: u64, _addr: u64, _len: usize) {}
    /// A store made through [`Self::normal`], whose code-page notification
    /// reached `Memory` rather than this provider: report the written bytes
    /// and, when the provider tracks a running block, their declared aliases.
    #[inline(always)]
    fn note_unobserved_write(&self, addr: u64, len: usize) {
        self.note_block_write(addr, len)
    }
    fn read<const ROUTED: bool>(&self, addr: u64, size: u8) -> Result<u64, MemTrap>;
    fn write<const ROUTED: bool>(&self, addr: u64, value: u64, size: u8) -> Result<(), MemTrap>;
    fn write_preflighted<const ROUTED: bool>(
        &self,
        addr: u64,
        value: u64,
        size: u8,
    ) -> Result<(), MemTrap> {
        self.write::<ROUTED>(addr, value, size)
    }
    /// [`Self::write_preflighted`] for the first transfer of a multi-transfer store:
    /// a span that no single region contains returns the containment miss instead
    /// of being stored across adjacent regions, so the caller can check the
    /// operand's other transfers before any byte is written. A provider whose
    /// `write_preflighted` stores such a span (see [`Self::SPLIT`]) overrides this.
    fn write_preflighted_contained<const ROUTED: bool>(
        &self,
        addr: u64,
        value: u64,
        size: u8,
    ) -> Result<(), MemTrap> {
        self.write_preflighted::<ROUTED>(addr, value, size)
    }
    fn probe_write<const ROUTED: bool>(&self, addr: u64, size: u8) -> Result<(), MemTrap>;
    /// [`Self::read`] that also performs a span over adjacent RAM/ROM regions, for a
    /// helper that commits state between its transfers (the real-mode stack frames
    /// in `interp/control.rs`). Its re-execution after a containment miss would
    /// repeat that commit, so the span is performed here, as the walk performed it
    /// before the split moved off the hot path. A provider that already performs
    /// spans (see [`Self::SPLIT`]) reads unchanged.
    #[inline(always)]
    fn read_across_regions<const ROUTED: bool>(&self, addr: u64, size: u8) -> Result<u64, MemTrap> {
        match self.read::<ROUTED>(addr, size) {
            Err(miss @ MemTrap::Unmapped) if !Self::SPLIT && size > 1 => {
                self.normal().read_across(addr, size, miss, ROUTED)
            }
            result => result,
        }
    }
    /// [`Self::write`] that also performs a span over adjacent RAM/ROM regions, with
    /// its SMC/watch notes sent to this provider; see [`Self::read_across_regions`].
    #[inline(always)]
    fn write_across_regions<const ROUTED: bool>(
        &self,
        addr: u64,
        value: u64,
        size: u8,
    ) -> Result<(), MemTrap>
    where
        Self: Sized,
    {
        match self.write::<ROUTED>(addr, value, size) {
            Err(miss @ MemTrap::Unmapped) if !Self::SPLIT && size > 1 => self
                .normal()
                .write_across_observed::<ROUTED, false>(addr, value, size, self, miss),
            result => result,
        }
    }
    /// Wide checked read (x87 and state images), retaining the faulting byte. The
    /// default also copies a span over adjacent RAM/ROM regions.
    #[inline(always)]
    fn wide_read<const ROUTED: bool>(&self, addr: u64, buf: &mut [u8]) -> Result<(), u64> {
        self.normal().read_ram_checked_spanning::<ROUTED>(addr, buf)
    }
    /// Wide checked store with its SMC/watch notes sent to this provider. The
    /// default also stores a span over adjacent RAM/ROM regions.
    #[inline(always)]
    fn wide_write<const ROUTED: bool>(&self, addr: u64, bytes: &[u8]) -> Result<(), u64>
    where
        Self: Sized,
    {
        self.normal()
            .write_ram_checked_spanning_observed::<ROUTED>(addr, bytes, self)
    }
    #[inline(always)]
    fn preflight_store<const ROUTED: bool>(
        &self,
        base: u64,
        mask: u64,
        elem: u8,
        bytes: u16,
    ) -> Result<(), (u64, MemTrap)> {
        if ROUTED {
            // A contiguous operand in one region needs one metadata check and
            // one recovery-routine call, regardless of its number of transfers.
            let memory = self.normal();
            if mask == u64::MAX && bytes <= u16::from(u8::MAX) {
                if let Ok(region) = memory.region_at(base, bytes as u8) {
                    match region.kind {
                        RegionKind::Rom => return Ok(()),
                        RegionKind::Trap => {}
                        RegionKind::Ram => {
                            let host = (memory.host_base() as usize)
                                .wrapping_add(base.wrapping_sub(memory.guest_base()) as usize);
                            let page = host_page_size();
                            if usize::from(bytes) <= page - (host & (page - 1)) {
                                return Ok(());
                            }
                            // SAFETY: region_at validated the whole RAM destination.
                            return unsafe {
                                crate::usercopy::probe(host as *mut u8, usize::from(bytes), true)
                            }
                            .map_err(|at| {
                                let trap = memory.host_fault(at);
                                (trap.fault_addr(base), trap)
                            });
                        }
                    }
                }
            }
            crate::interp::preflight_store_pages(
                base,
                mask,
                elem,
                bytes,
                self.normal().host_base() as usize,
                self.normal().guest_base(),
                (
                    |a, n| self.probe_write::<false>(a, n),
                    |a, n| {
                        self.probe_write::<true>(a, n)?;
                        memory.region_end(a)
                    },
                ),
            )?;
        }
        Ok(())
    }
    fn child_atomic(&self, _op: &crate::IrOp, _temps: &mut [u64], _cpu: &mut crate::CpuState) {
        unreachable!("child atomic on normal memory")
    }
    fn stopped(&self) -> bool {
        false
    }
    fn begin_op(&self, _op: &crate::IrOp, _temps: &[u64], _cpu: &crate::CpuState) {}
    fn exception(&self, _vector: u8, _cause: crate::FaultCause, _advance: u8) {
        unreachable!("child exception on normal memory")
    }
    fn string<const ROUTED: bool>(
        &self,
        _cpu: &mut crate::CpuState,
        _op: crate::ir::StrOp,
        _elem: u8,
        _rep: crate::ir::RepKind,
        _bits: u8,
        _base: u64,
    ) {
        unreachable!("child string on normal memory")
    }
}
pub(crate) struct RunningMemory<'a> {
    pub memory: &'a Memory,
    pub ir: &'a crate::IrBlock,
    pub start: u64,
    pub end: u64,
    pub host_start: u64,
    pub host_end: u64,
    pub current: &'a Cell<u64>,
    pub started: &'a Cell<u8>,
}

impl RunningMemory<'_> {
    #[cold]
    #[inline(never)]
    fn overlap(&self, addr: u64, len: usize) {
        // SDM Vol 3A §14.6: discard decoded bytes after the storing instruction.
        // Scan only after an overlap; ordinary instruction markers do no extra work.
        let next = self.ir.ops.iter().find_map(|op| match op {
            crate::IrOp::InsnStart { guest_addr } if *guest_addr > self.current.get() => Some(
                self.start
                    .wrapping_add(guest_addr.wrapping_sub(self.ir.guest_start)),
            ),
            _ => None,
        });
        if next.is_some_and(|next| addr.saturating_add(len as u64) > next) {
            self.started.set(0);
        }
    }
}

impl InterpreterMemory for RunningMemory<'_> {
    #[inline(always)]
    fn normal(&self) -> &Memory {
        self.memory
    }
    #[inline(always)]
    fn note_block_write(&self, addr: u64, len: usize) {
        if addr < self.end && addr.saturating_add(len as u64) > self.start {
            self.overlap(addr, len);
        }
    }
    #[inline(always)]
    fn note_host_block_write(&self, host: u64, addr: u64, len: usize) {
        if host < self.host_end && host.saturating_add(len as u64) > self.host_start {
            self.overlap(addr, len);
        }
    }
    // An observed store's code-page path expands aliases before it calls
    // `note_block_write`; a store made through `Memory` has only its own
    // view here, so another view of the running block is expanded here.
    // One relaxed load keeps the expansion off a store with no alias declared.
    #[inline(always)]
    fn note_unobserved_write(&self, addr: u64, len: usize) {
        if addr < self.end && addr.saturating_add(len as u64) > self.start {
            self.overlap(addr, len);
        } else if self.memory.has_code_aliases() {
            self.memory.visit_code_alias_views(addr, len, |lo, hi| {
                self.note_block_write(lo, (hi - lo) as usize)
            });
        }
    }
    // The interpreter walk's transfers are containment-only. A miss over adjacent
    // regions ends the attempt with its usual exit; the instruction is then
    // re-executed through `SplitMemory` (`interp::split_block`).
    #[inline(always)]
    fn read<const ROUTED: bool>(&self, addr: u64, size: u8) -> Result<u64, MemTrap> {
        self.memory.read_mode::<ROUTED>(addr, size)
    }
    #[inline(always)]
    fn write<const ROUTED: bool>(&self, addr: u64, value: u64, size: u8) -> Result<(), MemTrap> {
        self.memory
            .write_mode_observed::<ROUTED, false>(addr, value, size, self)
    }
    #[inline(always)]
    fn write_preflighted<const ROUTED: bool>(
        &self,
        addr: u64,
        value: u64,
        size: u8,
    ) -> Result<(), MemTrap> {
        self.memory
            .write_mode_observed::<ROUTED, true>(addr, value, size, self)
    }
    #[inline(always)]
    fn probe_write<const ROUTED: bool>(&self, addr: u64, size: u8) -> Result<(), MemTrap> {
        self.memory.probe_write_mode::<ROUTED>(addr, size)
    }
    #[inline(always)]
    fn wide_read<const ROUTED: bool>(&self, addr: u64, buf: &mut [u8]) -> Result<(), u64> {
        self.memory.read_ram_checked_mode::<ROUTED>(addr, buf)
    }
    #[inline(always)]
    fn wide_write<const ROUTED: bool>(&self, addr: u64, bytes: &[u8]) -> Result<(), u64> {
        self.memory
            .write_ram_checked_mode_observed::<ROUTED>(addr, bytes, self)
    }
}

// The provider of out-of-walk helpers (the Machine bus helpers, interrupt
// delivery, state images): a containment miss over adjacent RAM/ROM regions is
// performed here, out of line, where the helper already handles its miss.
impl InterpreterMemory for Memory {
    const SPLIT: bool = true;
    #[inline(always)]
    fn normal(&self) -> &Memory {
        self
    }
    #[inline(always)]
    fn read<const ROUTED: bool>(&self, addr: u64, size: u8) -> Result<u64, MemTrap> {
        self.read_spanning::<ROUTED>(addr, size)
    }
    #[inline(always)]
    fn write<const ROUTED: bool>(&self, addr: u64, value: u64, size: u8) -> Result<(), MemTrap> {
        self.write_mode::<ROUTED>(addr, value, size)
    }
    #[inline(always)]
    fn write_preflighted<const ROUTED: bool>(
        &self,
        addr: u64,
        value: u64,
        size: u8,
    ) -> Result<(), MemTrap> {
        self.write_spanning_observed::<ROUTED, true>(addr, value, size, self)
    }
    #[inline(always)]
    fn write_preflighted_contained<const ROUTED: bool>(
        &self,
        addr: u64,
        value: u64,
        size: u8,
    ) -> Result<(), MemTrap> {
        self.write_mode_observed::<ROUTED, true>(addr, value, size, self)
    }
    #[inline(always)]
    fn probe_write<const ROUTED: bool>(&self, addr: u64, size: u8) -> Result<(), MemTrap> {
        self.probe_write_spanning::<ROUTED>(addr, size)
    }
}

/// The provider of the interpreter's re-execution of one instruction after a
/// containment miss (`interp::split_block`), wrapping the walk's provider. A
/// contained access takes the inner provider unchanged; an `Unmapped` miss is
/// handed to the cold `*_across` paths, which perform a span over adjacent RAM/ROM
/// regions and return any other miss unchanged. It never serves the first attempt
/// of an instruction, so that walk keeps its containment-only transfers.
pub(crate) struct SplitMemory<'a, M>(pub &'a M);

impl<M: InterpreterMemory> InterpreterMemory for SplitMemory<'_, M> {
    const SPLIT: bool = true;
    #[inline(always)]
    fn normal(&self) -> &Memory {
        self.0.normal()
    }
    #[inline(always)]
    fn note_block_write(&self, addr: u64, len: usize) {
        self.0.note_block_write(addr, len)
    }
    #[inline(always)]
    fn note_host_block_write(&self, host: u64, addr: u64, len: usize) {
        self.0.note_host_block_write(host, addr, len)
    }
    #[inline(always)]
    fn note_unobserved_write(&self, addr: u64, len: usize) {
        self.0.note_unobserved_write(addr, len)
    }
    #[inline(always)]
    fn read<const ROUTED: bool>(&self, addr: u64, size: u8) -> Result<u64, MemTrap> {
        match self.0.read::<ROUTED>(addr, size) {
            Err(miss @ MemTrap::Unmapped) => self.normal().read_across(addr, size, miss, ROUTED),
            result => result,
        }
    }
    #[inline(always)]
    fn write<const ROUTED: bool>(&self, addr: u64, value: u64, size: u8) -> Result<(), MemTrap> {
        match self.0.write::<ROUTED>(addr, value, size) {
            Err(miss @ MemTrap::Unmapped) => self
                .normal()
                .write_across_observed::<ROUTED, false>(addr, value, size, self.0, miss),
            result => result,
        }
    }
    #[inline(always)]
    fn write_preflighted<const ROUTED: bool>(
        &self,
        addr: u64,
        value: u64,
        size: u8,
    ) -> Result<(), MemTrap> {
        match self.0.write_preflighted::<ROUTED>(addr, value, size) {
            Err(miss @ MemTrap::Unmapped) => self
                .normal()
                .write_across_observed::<ROUTED, true>(addr, value, size, self.0, miss),
            result => result,
        }
    }
    #[inline(always)]
    fn write_preflighted_contained<const ROUTED: bool>(
        &self,
        addr: u64,
        value: u64,
        size: u8,
    ) -> Result<(), MemTrap> {
        self.0.write_preflighted::<ROUTED>(addr, value, size)
    }
    #[inline(always)]
    fn probe_write<const ROUTED: bool>(&self, addr: u64, size: u8) -> Result<(), MemTrap> {
        match self.0.probe_write::<ROUTED>(addr, size) {
            Err(miss @ MemTrap::Unmapped) => {
                self.normal().probe_write_across(addr, size, miss, ROUTED)
            }
            result => result,
        }
    }
    #[inline(always)]
    fn wide_read<const ROUTED: bool>(&self, addr: u64, buf: &mut [u8]) -> Result<(), u64> {
        self.normal().read_ram_checked_spanning::<ROUTED>(addr, buf)
    }
    #[inline(always)]
    fn wide_write<const ROUTED: bool>(&self, addr: u64, bytes: &[u8]) -> Result<(), u64> {
        self.normal()
            .write_ram_checked_spanning_observed::<ROUTED>(addr, bytes, self.0)
    }
}
impl<const ROUTED: bool, const ORDER: u8, M: InterpreterMemory> std::ops::Deref
    for Access<ROUTED, ORDER, M>
{
    type Target = Memory;
    #[inline(always)]
    fn deref(&self) -> &Memory {
        self.0.normal()
    }
}
impl<const ROUTED: bool, const ORDER: u8, M: InterpreterMemory> Access<ROUTED, ORDER, M> {
    #[inline(always)]
    pub fn new(mem: &M) -> &Self {
        // SAFETY: transparent wrapper; shared borrowing retains the backing's
        // lifetime and interior-mutability contract. No owned Access is constructed.
        unsafe { &*(mem as *const M).cast::<Self>() }
    }
    #[inline(always)]
    pub fn read(&self, addr: u64, size: u8) -> Result<u64, MemTrap> {
        let _ordering = crate::interp::ordering::Barrier::<ORDER>::new(
            Some(crate::interp::ordering::Direction::Read),
            false,
        );
        let result = self.0.read::<ROUTED>(addr, size);
        #[cfg(test)]
        crate::interp::ordering::EVENTS.with(|events| events.borrow_mut().push("read"));
        result
    }
    #[inline(always)]
    pub fn write(&self, addr: u64, value: u64, size: u8) -> Result<(), MemTrap> {
        let _ordering = crate::interp::ordering::Barrier::<ORDER>::new(
            Some(crate::interp::ordering::Direction::Write),
            false,
        );
        let result = self.0.write::<ROUTED>(addr, value, size);
        #[cfg(test)]
        crate::interp::ordering::EVENTS.with(|events| events.borrow_mut().push("write"));
        result
    }
    /// [`Self::read`] through [`InterpreterMemory::read_across_regions`].
    #[inline(always)]
    pub(crate) fn read_across_regions(&self, addr: u64, size: u8) -> Result<u64, MemTrap> {
        let _ordering = crate::interp::ordering::Barrier::<ORDER>::new(
            Some(crate::interp::ordering::Direction::Read),
            false,
        );
        let result = self.0.read_across_regions::<ROUTED>(addr, size);
        #[cfg(test)]
        crate::interp::ordering::EVENTS.with(|events| events.borrow_mut().push("read"));
        result
    }
    /// [`Self::write`] through [`InterpreterMemory::write_across_regions`].
    #[inline(always)]
    pub(crate) fn write_across_regions(
        &self,
        addr: u64,
        value: u64,
        size: u8,
    ) -> Result<(), MemTrap>
    where
        M: Sized,
    {
        let _ordering = crate::interp::ordering::Barrier::<ORDER>::new(
            Some(crate::interp::ordering::Direction::Write),
            false,
        );
        let result = self.0.write_across_regions::<ROUTED>(addr, value, size);
        #[cfg(test)]
        crate::interp::ordering::EVENTS.with(|events| events.borrow_mut().push("write"));
        result
    }
    #[inline(always)]
    pub(crate) fn unordered(&self) -> &Access<ROUTED, 0, M> {
        Access::new(&self.0)
    }
    #[inline(always)]
    pub(crate) fn helper(
        &self,
        direction: Option<crate::interp::ordering::Direction>,
    ) -> (
        &Access<ROUTED, 0, M>,
        crate::interp::ordering::Barrier<ORDER>,
    ) {
        (
            self.unordered(),
            crate::interp::ordering::Barrier::new(direction, true),
        )
    }
    #[inline(always)]
    pub fn write_preflighted(&self, addr: u64, value: u64, size: u8) -> Result<(), MemTrap> {
        let _ordering = crate::interp::ordering::Barrier::<ORDER>::new(
            Some(crate::interp::ordering::Direction::Write),
            false,
        );
        let result = self.0.write_preflighted::<ROUTED>(addr, value, size);
        #[cfg(test)]
        crate::interp::ordering::EVENTS.with(|events| events.borrow_mut().push("write"));
        result
    }
    #[inline(always)]
    pub fn write_preflighted_contained(
        &self,
        addr: u64,
        value: u64,
        size: u8,
    ) -> Result<(), MemTrap> {
        let _ordering = crate::interp::ordering::Barrier::<ORDER>::new(
            Some(crate::interp::ordering::Direction::Write),
            false,
        );
        let result = self
            .0
            .write_preflighted_contained::<ROUTED>(addr, value, size);
        #[cfg(test)]
        crate::interp::ordering::EVENTS.with(|events| events.borrow_mut().push("write"));
        result
    }
    #[inline(always)]
    pub fn store_crosses_host_page(&self, addr: u64, bytes: usize) -> bool {
        if !ROUTED {
            return false;
        }
        let memory = self.0.normal();
        let host = (memory.host_base() as usize)
            .wrapping_add(addr.wrapping_sub(memory.guest_base()) as usize);
        if bytes <= 4096 - (host & 4095) {
            return false;
        }
        let page = host_page_size();
        bytes > page - (host & (page - 1))
    }
    #[inline(always)]
    pub fn probe_write(&self, addr: u64, size: u8) -> Result<(), MemTrap> {
        self.0.probe_write::<ROUTED>(addr, size)
    }
    #[inline(always)]
    pub fn probe_write_mode<const RECOVER: bool>(
        &self,
        addr: u64,
        size: u8,
    ) -> Result<(), MemTrap> {
        self.0.probe_write::<RECOVER>(addr, size)
    }
    pub(crate) fn write_ram(&self, addr: u64, bytes: &[u8]) -> Result<(), u64> {
        self.0.wide_write::<ROUTED>(addr, bytes)
    }
    pub fn atomic_rmw(&self, addr: u64, src: u64, size: u8, op: RmwOp) -> Result<u64, MemTrap> {
        self.0
            .normal()
            .atomic_rmw_mode_observed::<ROUTED>(addr, src, size, op, &self.0)
    }
    pub fn atomic_cas128(&self, addr: u64, expected: u128, src: u128) -> Result<u128, MemTrap> {
        self.0
            .normal()
            .atomic_cas128_mode_observed::<ROUTED>(addr, expected, src, &self.0)
    }
    pub fn atomic_cas(&self, addr: u64, expected: u64, src: u64, size: u8) -> Result<u64, MemTrap> {
        self.0
            .normal()
            .atomic_cas_mode_observed::<ROUTED>(addr, expected, src, size, &self.0)
    }
}

#[cfg(test)]
mod fault_layout_tests {
    use super::*;
    #[test]
    fn hot_fault_layouts_remain_compact() {
        assert_eq!(std::mem::size_of::<MemTrap>(), 1);
        assert_eq!(std::mem::size_of::<Result<(), MemTrap>>(), 1);
    }
}

#[cfg(test)]
mod child_isolation_tests {
    use super::*;
    use crate::{child, ChildArena, ChildStep, CpuMode, CpuState, Vm, VmConfig};

    fn child_ram(range: std::ops::Range<u64>) -> crate::ChildRegion {
        crate::ChildRegion::Ram {
            backing_offset: range.start as usize,
            range,
            fetch: FetchVerdict::Allow,
        }
    }

    #[test]
    fn invalid_sparse_child_metadata_never_registers_a_journal() {
        use crate::{ChildRegion, ChildViewError as E, ChildViewMetadata};
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        // Deliberately unsorted parent mappings, including adjacent RAM and a gap.
        vm.map(0x4000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        vm.map(0x2000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x5000, 0x1000, Prot::RX, RegionKind::Rom).unwrap();
        let cases = [
            (vec![child_ram(0x1000..0x1000)], vec![], E::InvalidRange),
            (
                vec![child_ram(std::ops::Range {
                    start: 0x2000,
                    end: 0x1000,
                })],
                vec![],
                E::InvalidRange,
            ),
            (vec![child_ram(0xffff..u64::MAX)], vec![], E::InvalidRange),
            (
                vec![
                    child_ram(0x1000..0x2000),
                    ChildRegion::Trap {
                        range: 0x1800..0x1900,
                    },
                ],
                vec![],
                E::Overlap,
            ),
            (vec![child_ram(0x1800..0x4001)], vec![], E::Unmapped),
            (
                vec![ChildRegion::Trap {
                    range: 0x3000..0x4000,
                }],
                vec![],
                E::Unmapped,
            ),
            (vec![child_ram(0x4000..0x5000)], vec![], E::NotRam),
            (vec![child_ram(0x5000..0x6000)], vec![], E::NotRam),
            (
                vec![ChildRegion::Ram {
                    range: 0x1000..0x2000,
                    backing_offset: 0,
                    fetch: FetchVerdict::Allow,
                }],
                vec![],
                E::BackingMismatch,
            ),
            (vec![child_ram(0x1000..0x2000)], vec![2], E::InvalidCodePage),
            (
                vec![ChildRegion::Trap {
                    range: 0x4000..0x5000,
                }],
                vec![4],
                E::InvalidCodePage,
            ),
            (vec![], vec![u64::MAX], E::InvalidCodePage),
        ];
        for (regions, code_store_refusals, error) in cases {
            assert!(
                matches!(vm.child_view_with(ChildViewMetadata { regions, code_store_refusals }, CpuMode::Long64), Err(e) if e == error)
            );
            assert!(vm.mem.child_writes.lock().unwrap().is_empty());
            assert_eq!(vm.mem.code_aliases.lock().unwrap().live_children, 0);
        }
        let view = vm
            .child_view_with(
                ChildViewMetadata {
                    regions: vec![
                        child_ram(0x1800..0x2800),
                        ChildRegion::Trap {
                            range: 0x4000..0x6000,
                        },
                    ],
                    code_store_refusals: vec![],
                },
                CpuMode::Long64,
            )
            .unwrap();
        assert_eq!(vm.mem.code_aliases.lock().unwrap().live_children, 1);
        drop(view);
        assert_eq!(vm.mem.code_aliases.lock().unwrap().live_children, 0);
        assert!(vm.mem.child_writes.lock().unwrap().is_empty());
        let empty = vm
            .child_view_with(ChildViewMetadata::default(), CpuMode::Long64)
            .unwrap();
        drop(empty);
    }

    #[test]
    fn sparse_child_uses_parent_alias_history_until_last_view_drop() {
        use crate::ChildViewMetadata;
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, &[0x89, 0x18]).unwrap();
        let view = vm
            .child_view_with(
                ChildViewMetadata {
                    regions: vec![child_ram(0x1000..0x2000), child_ram(0x3000..0x4000)],
                    code_store_refusals: vec![],
                },
                CpuMode::Long64,
            )
            .unwrap();
        let sibling_view = vm.child_view(CpuMode::Long64);
        vm.mem.declare_code_alias(0x3000, 0x8000, 0x1000).unwrap();
        vm.mem.mark_code(0x8000, 1);
        let word = Arc::new(AtomicU64::new(0));
        vm.mem.register_dispatch_word(&word);
        let mut cpu = CpuState {
            rip: 0x1000,
            ..CpuState::default()
        };
        cpu.gpr[0] = 0x3000;
        cpu.gpr[3] = 42;
        let mut arena = ChildArena::new(2);
        assert!(matches!(
            child::step(&view, &mut cpu, &mut arena),
            ChildStep::Continue
        ));
        assert!(vm.mem.remove_code_alias(0x3000, 0x8000, 0x1000));
        vm.mem.dispatch_smc(&word, |ranges| {
            assert_eq!(ranges, vec![(0x3000, 0x4000), (0x8000, 0x9000)])
        });
        vm.invalidate_written_pages(&arena);
        drop(view);
        assert_eq!(vm.mem.code_aliases.lock().unwrap().live_children, 1);
        assert!(vm.mem.is_code_page(3));
        drop(sibling_view);
        let aliases = vm.mem.code_aliases.lock().unwrap();
        assert_eq!(aliases.live_children, 0);
        assert!(aliases.retired_aliases.is_empty());
        assert!(!vm.mem.is_code_page(3));
        assert!(vm.mem.is_code_page(8));
    }

    #[test]
    fn alias_removal_before_child_journal_publication() {
        alias_removal_during_child_store(true);
    }

    #[test]
    fn alias_removal_after_child_journal_before_tag_read() {
        alias_removal_during_child_store(false);
    }

    fn alias_removal_during_child_store(before_journal: bool) {
        for drain_in_hook in [false, true] {
            let mut vm = Vm::new(VmConfig::flat(0x6000));
            vm.map(0, 0x6000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0x1000, &[0x89, 0x18]).unwrap();
            vm.write_bytes(0x4000, &[0xb8, 1, 0, 0, 0, 0xf4]).unwrap();
            let key = crate::cache::BlockKey::new(0x4000, CpuMode::Long64);
            let other = crate::vm::EXIT_REQUESTED | crate::vm::CODE_SPACE_EXHAUSTED;
            let a = Arc::new(AtomicU64::new(other));
            let b = Arc::new(AtomicU64::new(other));
            let invalidate = |ranges: Vec<(u64, u64)>| {
                for (lo, hi) in ranges {
                    vm.cache.invalidate_written(lo, hi, |empty| {
                        if empty {
                            vm.mem.clear_code_page(lo >> CODE_PAGE_BITS);
                        }
                    });
                }
            };
            let remove = || {
                assert_eq!(vm.mem.read(0x3000, 4), Ok(2), "guest store committed");
                assert!(!vm.mem.has_dirty_code(), "child has not read its tag yet");
                assert!(vm.mem.remove_code_alias(0x3000, 0x4000, 0x1000));
                assert!(vm.mem.is_code_page(3), "removal retains the live tag");
                assert!(!vm.mem.has_dirty_code(), "removal needs no forced drain");
                assert!(!vm
                    .mem
                    .code_aliases
                    .lock()
                    .unwrap()
                    .pairs
                    .contains(&(0x3000, 0x4000, 0x1000)));
                for word in [&a, &b] {
                    assert_eq!(word.load(Ordering::Acquire), other);
                }
                if drain_in_hook {
                    // An unrelated notice lets dispatch drain/acknowledge while
                    // this child is paused, possibly with its journal still empty.
                    vm.mem.child_dispatch.notify();
                    vm.mem.dispatch_smc(&a, |ranges| {
                        let expected = if before_journal {
                            vec![(0x5000, 0x6000)]
                        } else {
                            vec![(0x3000, 0x4000), (0x4000, 0x5000), (0x5000, 0x6000)]
                        };
                        assert_eq!(ranges, expected);
                        invalidate(ranges);
                    });
                    assert_eq!(a.load(Ordering::Acquire), other);
                    assert!(!vm.mem.has_dirty_code());
                    if before_journal {
                        assert!(
                            vm.cache.get(key).is_some(),
                            "empty journal cannot evict the translation"
                        );
                    }
                }
            };
            let mut view = vm.child_view(CpuMode::Long64);
            let sibling_view = vm.child_view(CpuMode::Long64);
            vm.mem.register_dispatch_word(&a);
            vm.mem.register_dispatch_word(&b);
            let mut cpu = CpuState {
                rip: 0x1000,
                ..CpuState::default()
            };
            cpu.gpr[0] = 0x5000;
            cpu.gpr[3] = 1;
            let mut arena = ChildArena::new(4);
            assert!(matches!(
                child::step(&sibling_view, &mut cpu, &mut arena),
                ChildStep::Continue
            ));
            vm.mem.declare_code_alias(0x3000, 0x4000, 0x1000).unwrap();
            let mut parent = vm.new_vcpu();
            parent.cpu.rip = 0x4000;
            assert!(matches!(parent.run(&vm, Some(100)), crate::Exit::Hlt));
            assert!(
                vm.cache.get(key).is_some(),
                "formerly aliased translation is cached"
            );
            if before_journal {
                view.before_journal = Some(&remove);
            } else {
                view.before_tag_read = Some(&remove);
            }
            cpu.rip = 0x1000;
            cpu.gpr[0] = 0x3000;
            cpu.gpr[3] = 2;
            assert!(matches!(
                child::step(&view, &mut cpu, &mut arena),
                ChildStep::Continue
            ));
            for word in [&a, &b] {
                assert_eq!(
                    word.load(Ordering::Acquire),
                    other | child::DISPATCH_CHILD_DIRTY
                );
                assert_eq!(
                    vm.mem
                        .child_dispatch
                        .epoch(word)
                        .unwrap()
                        .load(Ordering::Acquire),
                    if drain_in_hook { 2 } else { 1 },
                    "the resumed child always notifies"
                );
            }
            vm.mem.dispatch_smc(&b, |ranges| {
                let expected = if !drain_in_hook {
                    vec![(0x3000, 0x4000), (0x4000, 0x5000), (0x5000, 0x6000)]
                } else if before_journal {
                    vec![(0x3000, 0x4000), (0x4000, 0x5000)]
                } else {
                    vec![]
                };
                assert_eq!(ranges, expected);
                invalidate(ranges);
            });
            assert!(
                vm.cache.get(key).is_none(),
                "no formerly aliased translation survives"
            );
            vm.mem.dispatch_smc(&a, |ranges| assert!(ranges.is_empty()));
            assert!(!vm.mem.has_dirty_code());
            assert_eq!(a.load(Ordering::Acquire), other);
            assert_eq!(b.load(Ordering::Acquire), other);
        }
    }

    #[test]
    fn removed_alias_tags_and_history_survive_until_the_last_view_drops() {
        let mut vm = Vm::new(VmConfig::flat(0x7000));
        vm.map(0, 0x7000, Prot::RWX, RegionKind::Ram).unwrap();
        let first = vm.child_view(CpuMode::Long64);
        let second = vm.child_view(CpuMode::Long64);
        vm.mem.declare_code_alias(0x3000, 0x4000, 0x1000).unwrap();
        vm.mem.declare_code_alias(0x4000, 0x5000, 0x1000).unwrap();
        vm.mem.mark_code(0x5000, 1);
        assert!(vm.mem.remove_code_alias(0x3000, 0x4000, 0x1000));
        // All tag-clear paths must honor the deferred clear, not just removal.
        vm.mem.clear_code_page(5);
        vm.mem.release_code_registration(4);
        vm.mem.clear_all_code_pages();
        for page in 3..=5 {
            assert!(vm.mem.is_code_page(page));
        }
        drop(first);
        let third = vm.child_view(CpuMode::Long64);
        drop(second);
        assert!(
            vm.mem.is_code_page(3),
            "a later view also holds the deferred clear"
        );
        assert!(!vm
            .mem
            .code_aliases
            .lock()
            .unwrap()
            .retired_aliases
            .is_empty());
        vm.mem.mark_code(0x6000, 1);
        drop(third);
        for page in 3..=5 {
            assert!(!vm.mem.is_code_page(page));
        }
        assert!(vm.mem.is_code_page(6), "unrelated active tag survives");
        let aliases = vm.mem.code_aliases.lock().unwrap();
        assert_eq!(aliases.live_children, 0);
        assert!(aliases.retired_aliases.is_empty());
        assert!(aliases.retained_code.is_empty());
        assert!(vm.mem.child_writes.lock().unwrap().is_empty());
    }

    #[test]
    fn alias_removal_after_all_views_drop_clears_tags_immediately() {
        let mut vm = Vm::new(VmConfig::flat(0x5000));
        vm.map(0, 0x5000, Prot::RWX, RegionKind::Ram).unwrap();
        let first = vm.child_view(CpuMode::Long64);
        let second = vm.child_view(CpuMode::Long64);
        vm.mem.declare_code_alias(0x3000, 0x4000, 0x1000).unwrap();
        vm.mem.mark_code(0x4000, 1);
        drop(first);
        drop(second);
        assert!(vm.mem.is_code_page(3));
        assert!(vm.mem.remove_code_alias(0x3000, 0x4000, 0x1000));
        assert!(!vm.mem.is_code_page(3));
        assert!(vm.mem.is_code_page(4), "direct translation stays tagged");
        assert!(!vm.mem.has_dirty_code());
        assert!(vm
            .mem
            .code_aliases
            .lock()
            .unwrap()
            .retired_aliases
            .is_empty());
    }

    #[test]
    fn child_notice_survives_a_store_during_parent_acknowledgement() {
        child_notice_during_parent_acknowledgement(1);
    }

    #[test]
    fn child_notice_survives_epoch_wrap_during_parent_acknowledgement() {
        child_notice_during_parent_acknowledgement(u64::MAX);
    }

    fn child_notice_during_parent_acknowledgement(epoch: u64) {
        let mut vm = Vm::new(VmConfig::flat(0x5000));
        vm.map(0, 0x5000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, &[0x89, 0x18]).unwrap();
        let view = vm.child_view(CpuMode::Long64);
        let other = crate::vm::EXIT_REQUESTED | crate::vm::CODE_SPACE_EXHAUSTED;
        let a = Arc::new(AtomicU64::new(other));
        let b = Arc::new(AtomicU64::new(other));
        vm.mem.register_dispatch_word(&a);
        vm.mem.register_dispatch_word(&b);
        vm.mem.mark_code(0x3000, 1);
        let mut cpu = CpuState {
            rip: 0x1000,
            ..CpuState::default()
        };
        cpu.gpr[0] = 0x3000;
        cpu.gpr[3] = 1;
        let mut arena = ChildArena::new(4);
        assert!(matches!(
            child::step(&view, &mut cpu, &mut arena),
            ChildStep::Continue
        ));
        let pending = other | crate::child::DISPATCH_CHILD_DIRTY;
        assert_eq!(a.load(Ordering::Acquire), pending);
        assert_eq!(b.load(Ordering::Acquire), pending);
        let child_epoch = vm.mem.child_dispatch.epoch(&b).unwrap();
        child_epoch.store(epoch, Ordering::Release);
        vm.mem.dispatch_smc(&b, |ranges| {
            assert_eq!(ranges, vec![(0x3000, 0x4000)]);
            assert!(vm.mem.dispatch_words.try_lock().is_err());
            cpu.rip = 0x1000;
            cpu.gpr[3] = 2;
            assert!(matches!(
                child::step(&view, &mut cpu, &mut arena),
                ChildStep::Continue
            ));
        });
        assert_eq!(child_epoch.load(Ordering::Acquire), epoch.wrapping_add(1));
        assert_eq!(b.load(Ordering::Acquire), pending, "racing notice retained");
        vm.mem
            .dispatch_smc(&a, |ranges| assert_eq!(ranges, vec![(0x3000, 0x4000)]));
        assert_eq!(a.load(Ordering::Acquire), other);
        vm.mem.dispatch_smc(&b, |ranges| assert!(ranges.is_empty()));
        assert_eq!(b.load(Ordering::Acquire), other);
    }

    #[test]
    fn child_store_notifies_a_concurrently_registered_dispatch_word() {
        use std::sync::Barrier;

        let mut vm = Vm::new(VmConfig::flat(0x5000));
        vm.map(0, 0x5000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, &[0x89, 0x18]).unwrap();
        let view = vm.child_view(CpuMode::Long64);
        vm.mem.mark_code(0x3000, 1);
        let start = Barrier::new(2);
        let done = Barrier::new(2);
        let other = crate::vm::EXIT_REQUESTED | crate::vm::CODE_SPACE_EXHAUSTED;
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let mut cpu = CpuState::default();
                cpu.gpr[0] = 0x3000;
                let mut arena = ChildArena::new(2);
                for value in 1..=128 {
                    cpu.rip = 0x1000;
                    cpu.gpr[3] = value;
                    start.wait();
                    assert!(matches!(
                        child::step(&view, &mut cpu, &mut arena),
                        ChildStep::Continue
                    ));
                    done.wait();
                }
            });
            for _ in 0..128 {
                let sibling = Arc::new(AtomicU64::new(other));
                start.wait();
                vm.mem.register_dispatch_word(&sibling);
                done.wait();
                assert_ne!(
                    sibling.load(Ordering::Acquire)
                        & (DISPATCH_DIRTY | crate::child::DISPATCH_CHILD_DIRTY),
                    0,
                    "registration or notification must set a notice"
                );
                vm.mem.dispatch_smc(&sibling, |ranges| {
                    assert_eq!(ranges, vec![(0x3000, 0x4000)]);
                });
                assert_eq!(sibling.load(Ordering::Acquire), other);
            }
        });
    }

    #[test]
    fn child_journal_preserves_removed_transitive_aliases() {
        let mut vm = Vm::new(VmConfig::flat(0x6000));
        vm.map(0, 0x6000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, &[0x89, 0x18]).unwrap();
        let view = vm.child_view(CpuMode::Long64);
        vm.mem.declare_code_alias(0x3000, 0x4000, 0x1000).unwrap();
        vm.mem.declare_code_alias(0x4000, 0x5000, 0x1000).unwrap();
        vm.mem.mark_code(0x5000, 1);
        let word = Arc::new(AtomicU64::new(0));
        vm.mem.register_dispatch_word(&word);
        let mut cpu = CpuState {
            rip: 0x1000,
            ..CpuState::default()
        };
        cpu.gpr[0] = 0x3000;
        cpu.gpr[3] = 1;
        let mut arena = ChildArena::new(2);
        assert!(matches!(
            child::step(&view, &mut cpu, &mut arena),
            ChildStep::Continue
        ));
        assert!(vm.mem.remove_code_alias(0x3000, 0x4000, 0x1000));
        assert!(vm.mem.remove_code_alias(0x4000, 0x5000, 0x1000));
        assert!(vm.mem.is_code_page(3));
        vm.mem.dispatch_smc(&word, |ranges| {
            assert_eq!(
                ranges,
                vec![(0x3000, 0x4000), (0x4000, 0x5000), (0x5000, 0x6000)]
            );
        });
        assert!(!vm.mem.has_dirty_code());
        assert_eq!(word.load(Ordering::Acquire), 0);
        drop(view);
        assert!(!vm.mem.is_code_page(3));
        assert!(!vm.mem.is_code_page(4));
        assert!(
            vm.mem.is_code_page(5),
            "direct translated owner survives last drop"
        );
    }

    #[test]
    fn child_step_does_not_consult_poisoned_parent_engine() {
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, &[0x48, 0x89, 0x18, 0x0f, 0x05])
            .unwrap();
        vm.set_fetch_policy(Box::new(|_| FetchVerdict::Allow));
        let view = vm.child_view(CpuMode::Long64);
        vm.cache.poison_for_child_test();
        // All these locks belong to the parent; poison them before child entry.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _dirty = vm.mem.dirty.lock().unwrap();
            let _dispatch = vm.mem.dispatch_words.lock().unwrap();
            let _watch = vm.mem.watch.words.lock().unwrap();
            let _fetch = vm.mem.fetch_policy.as_ref().unwrap().state.lock().unwrap();
            panic!("poison parent memory and policy");
        }));
        assert!(result.is_err());
        assert!(vm.mem.dirty.is_poisoned() && vm.mem.dispatch_words.is_poisoned());
        assert!(vm.mem.watch.words.is_poisoned());
        assert!(vm.mem.fetch_policy.as_ref().unwrap().state.is_poisoned());
        // Late tags use the atomic bridge; unrelated parent metadata stays untouched.
        vm.mem.mark_code(0x3000, 1);
        vm.mem.last_region.store(usize::MAX, Ordering::Relaxed);
        vm.mem.watch.count.store(1, Ordering::Relaxed);
        vm.mem.watch.watch[0].store(u64::MAX, Ordering::Relaxed);
        let mut cpu = CpuState {
            rip: 0x1000,
            ..CpuState::default()
        };
        cpu.gpr[0] = 0x3000;
        cpu.gpr[3] = 0x1234;
        let mut arena = ChildArena::new(4);
        crate::codemap::with_child_test_poison(|| {
            assert!(matches!(
                child::step(&view, &mut cpu, &mut arena),
                ChildStep::Continue
            ));
            assert!(matches!(
                child::step(&view, &mut cpu, &mut arena),
                ChildStep::Syscall { .. }
            ));
        });
        assert_eq!(vm.mem.last_region.load(Ordering::Relaxed), usize::MAX);
        assert!(!vm.mem.dirty_flag.load(Ordering::Relaxed));
        assert!(vm.mem.child_dispatch.dirty.load(Ordering::Acquire));
        assert_eq!(vm.mem.watch.dirty[0].load(Ordering::Relaxed), 0);
        assert_eq!(&*arena.written_pages(), &[3]);
    }
}
