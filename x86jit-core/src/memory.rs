//! Guest memory model (§4.1, §4.2, §8.1).

use std::cell::UnsafeCell;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

use crate::ir::RmwOp;

/// SDM Vol 1 §3.3.7.1: no LA57; bits 63:47 must all agree.
pub fn is_canonical(addr: u64) -> bool {
    ((addr as i64) << 16 >> 16) as u64 == addr
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
    /// reserve hundreds of GiB on a normal host the backing must be a `MAP_NORESERVE`
    /// mapping — which the guest-agnostic core can't allocate (no OS dep), so the
    /// embedder provides it via [`Memory::from_host_ram`]. Constructing this model
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
/// guest-agnostic core must not depend on an OS allocator, so an embedder that
/// needs a huge sparse mapping (a `MAP_NORESERVE` mmap for Go's 768 GiB arena hints)
/// allocates it host-side and hands core the raw region through
/// [`Memory::from_host_ram`]. `dtor(ptr, len)` frees it when the `Memory` drops.
///
/// # Safety
/// `ptr` must be valid for reads and writes over `[ptr, ptr+len)` for the lifetime
/// of the `Memory` it backs, and `dtor` must correctly release exactly that region.
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
    /// The write-drop is enforced on the interpreter's scalar/atomic store path.
    /// Real-mode guests (the BIOS/ROM consumer) run entirely on the interpreter,
    /// so this fully covers ROM. `Rom` is deliberately **excluded from
    /// [`Memory::trap_window`]** so its reads/fetches stay inlined; under the
    /// Long64/Compat32 JIT the inlined store fast path does not consult region
    /// kind, so ROM's write-drop is an interpreter-path guarantee.
    Rom,
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
    Overlap,
    OutOfBounds,
    BadAlign,
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
/// tagged "backs translated code" and at which a code-overlapping write triggers
/// cache invalidation. Independent of any host page size.
pub const CODE_PAGE_BITS: u32 = 12;
const CODE_PAGE_SIZE: u64 = 1 << CODE_PAGE_BITS;

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
        // Count only unwatched->watched transitions, so `count` is exact.
        if word.fetch_or(m, Ordering::Relaxed) & m == 0 {
            self.count.fetch_add(1, Ordering::Relaxed);
            self.words.lock().unwrap().insert(w as u64);
        }
    }

    fn unwatch(&self, page: u64) {
        let (w, m) = word_bit(page.wrapping_sub(self.base_page));
        let Some(word) = self.watch.get(w) else {
            return;
        };
        if word.fetch_and(!m, Ordering::Relaxed) & m != 0 {
            self.count.fetch_sub(1, Ordering::Relaxed);
            // Drop any pending dirty bit: an unwatched page must not surface in a
            // later drain, nor come back stale if the range is watched again.
            self.dirty[w].fetch_and(!m, Ordering::Relaxed);
            if word.load(Ordering::Relaxed) == 0 {
                self.words.lock().unwrap().remove(&(w as u64));
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
    // Region tags (prot/kind + bounds). `map()`/`unmap()` mutate this through
    // `&mut self` before execution; per-access lookups read it through `&self`.
    regions: Vec<Region>,
    // SMC tracking (§10). `code_page[p]` = page `p` backs a translated block, so a
    // write to it must invalidate the cache. Atomic (not `&mut`) because it's set
    // through `&self` at block-resolve time and read on every store — the same
    // shared-through-`&self` discipline as `backing`. `dirty` collects code pages
    // written since the last drain; `dirty_flag` lets the hot path skip the lock.
    /// Bit set = page backs a translated block. Page indices are relative to
    /// `guest_base >> CODE_PAGE_BITS`, including a partially backed first page.
    code_page: Box<[AtomicU64]>,
    // Packed `(lo << 32) | len` over 64 KiB granules — the *watermark* the JIT's
    // inlined stores gate on, so a store outside it skips the table entirely. See
    // `code_range_ptr` for why a watermark and not a table probe.
    code_range: AtomicU64,
    dirty: Mutex<Vec<u64>>,
    dirty_flag: AtomicBool,
    dispatch_words: Mutex<Vec<Weak<AtomicU64>>>,
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
}

// SAFETY: concurrent guest stores are intended to race like real hardware; the
// guest's expected ordering is provided by emitted TSO barriers, not by Rust
// aliasing rules. No host-side invariant is broken by concurrent access to the
// flat byte buffer (bounds-checked per access).
unsafe impl Sync for Memory {}

impl Memory {
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
        // Huge sparse mappings are supplied by the embedder through from_host_ram.
        let len = match &model {
            MemoryModel::Flat { size } | MemoryModel::Reserved { span: size } => *size as usize,
            MemoryModel::SoftMmu => 0,
        };
        Self::from_backing(model, Backing::zeroed(len))
    }

    /// Build a `Reserved` memory over an embedder-provided host mapping.
    /// Core can't allocate a `MAP_NORESERVE` span itself (it must stay guest-agnostic,
    /// depending only on the decoder), so the embedder hands in the raw region; the
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
        Self {
            model,
            backing: UnsafeCell::new(backing),
            regions: Vec::new(),
            code_page,
            code_range: AtomicU64::new(0),
            dirty: Mutex::new(Vec::new()),
            dirty_flag: AtomicBool::new(false),
            dispatch_words: Mutex::new(Vec::new()),
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
        if self.guest_base != 0 {
            return None;
        }
        let owner = &unsafe { &*self.backing.get() }.owner;
        let bytes: Box<[u8]> = match (self.model, owner) {
            // A host-backed (NORESERVE) `Reserved` span can't be re-allocated by the
            // core (that's the embedder's job), and cloning it into a `Vec` would
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
        Some(child)
    }

    fn with_regions(mut self, regions: Vec<Region>) -> Self {
        self.regions = regions;
        self
    }

    /// Tag every page spanned by `[addr, addr+len)` as backing translated code
    /// (§10). Called through `&self` when a block is cached, so a later store to
    /// the page is caught. Idempotent.
    pub fn mark_code(&self, addr: u64, len: u32) {
        // The table covers the whole guest span, so every mappable page has a bit; a
        // page outside it no-ops (`code_page.get` → None) rather than aborting.
        for page in code_page_range(addr, len as u64) {
            let (w, m) = word_bit(page.wrapping_sub(self.guest_base >> CODE_PAGE_BITS));
            if let Some(word) = self.code_page.get(w) {
                word.fetch_or(m, Ordering::Relaxed);
                self.widen_code_range((page << CODE_PAGE_BITS) >> CODE_RANGE_BITS);
            }
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
                Ordering::Relaxed,
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
    pub fn code_range_ptr(&self) -> u64 {
        &self.code_range as *const AtomicU64 as u64
    }

    /// Clear a page's code tag after its blocks have been invalidated (§10).
    pub fn clear_code_page(&self, page: u64) {
        let (w, m) = word_bit(page.wrapping_sub(self.guest_base >> CODE_PAGE_BITS));
        if let Some(word) = self.code_page.get(w) {
            word.fetch_and(!m, Ordering::Relaxed);
        }
    }

    /// Clear every code-page tag — for a whole-cache invalidation (e.g. mapping a
    /// Trap region, §5.2 M4-T10), the bulk counterpart of [`Self::clear_code_page`].
    pub fn clear_all_code_pages(&self) {
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
        for word in &self.code_page
            [first_word.min(self.code_page.len())..end_word.min(self.code_page.len())]
        {
            word.store(0, Ordering::Relaxed);
        }
    }

    /// Note a store of `size` bytes at `addr`: if it lands on a code page, record
    /// the page(s) as dirty for the dispatcher to invalidate (§10). The common
    /// case (a non-code page) costs one relaxed atomic load and returns.
    ///
    /// `pub` because generated code reaches it too, through the helper the inlined
    /// store calls once its watermark test says the store *might* have hit a code or
    /// watched page. Both halves are precise here; the inline test is not, and is not
    /// meant to be.
    pub fn note_write(&self, addr: u64, len: usize) {
        // One relaxed load gates the (rare) data-watch path: an unwatched memory pays
        // nothing beyond this branch.
        let watched = self.watch.count.load(Ordering::Relaxed) != 0;
        for page in code_page_range(addr, len as u64) {
            if self.is_code_page(page) {
                // Serialize notification and clearing with the drain. Exit bits in
                // these words are independent and must survive every SMC update.
                let mut words = self.dispatch_words.lock().unwrap();
                self.dirty.lock().unwrap().push(page);
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
            if watched && self.watch.is_watched(page) {
                self.watch.mark_dirty(page);
            }
        }
    }

    pub(crate) fn register_dispatch_word(&self, word: &Arc<AtomicU64>) {
        let mut words = self.dispatch_words.lock().unwrap();
        words.retain(|weak| weak.strong_count() != 0);
        if self.has_dirty_code() {
            word.fetch_or(DISPATCH_DIRTY, Ordering::Relaxed);
        }
        words.push(Arc::downgrade(word));
    }

    /// Whether any code page has been written since the last `take_dirty_code` — the
    /// cheap global SMC test. Compiled chain edges poll their per-vcpu notification
    /// word, which folds this condition with exit requests. One relaxed load; false
    /// for every guest that does not modify its own code.
    pub fn has_dirty_code(&self) -> bool {
        self.dirty_flag.load(Ordering::Relaxed)
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
        // Common case (nothing self-modified): a shared `load`, not a `swap` — the
        // latter takes the cache line exclusive on every dispatch even when clean.
        if !self.dirty_flag.load(Ordering::Relaxed) {
            return Vec::new();
        }
        let _words = self.dispatch_words.lock().unwrap();
        self.take_dirty_code_locked()
    }

    /// Serialize invalidation and this vcpu's acknowledgement with notifications
    /// and other dispatchers. A running vcpu must retain its own notice until it
    /// reaches dispatch, even if another vcpu already drained the global list.
    pub(crate) fn dispatch_smc(&self, word: &AtomicU64, invalidate: impl FnOnce(Vec<u64>)) {
        if word.load(Ordering::Relaxed) & DISPATCH_DIRTY == 0 && !self.has_dirty_code() {
            return;
        }
        let _words = self.dispatch_words.lock().unwrap();
        invalidate(self.take_dirty_code_locked());
        // The lock makes prior invalidation visible before acknowledging this
        // notice and prevents a new write from losing its notification here.
        word.fetch_and(!DISPATCH_DIRTY, Ordering::Relaxed);
    }

    // Caller holds dispatch_words through any invalidation and acknowledgement.
    fn take_dirty_code_locked(&self) -> Vec<u64> {
        // Something's dirty: claim it. A racing vcpu may have drained it already.
        if !self.dirty_flag.swap(false, Ordering::Relaxed) {
            return Vec::new();
        }
        std::mem::take(&mut *self.dirty.lock().unwrap())
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
    /// per-access cost, the common case.
    pub fn trap_window(&self) -> Option<(u64, u64)> {
        let mut lo = u64::MAX;
        let mut hi = 0u64;
        for r in &self.regions {
            if matches!(r.kind, RegionKind::Trap) {
                lo = lo.min(r.start);
                hi = hi.max(r.start.saturating_add(r.size as u64));
            }
        }
        (lo < hi).then_some((lo, hi))
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
                self.regions.push(Region {
                    start: guest_addr,
                    size,
                    prot,
                    kind,
                });
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
        unsafe { self.copy_to_guest(self.host_base().cast_mut().add(start), bytes) }
            .map_err(|_| MemError::Unmapped)?;
        // SMC: an embedder write (loader, syscall passthrough) over a code page
        // must invalidate too (§10).
        self.note_write(guest_addr, bytes.len());
        Ok(())
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

    /// Drop a region's tag. In `Flat` the backing buffer is untouched — only the
    /// permission/kind tag goes away. Partial unmap isn't modeled in M0, so the
    /// `(guest_addr, size)` must match a mapped region exactly. (§4.1)
    pub fn unmap(&mut self, guest_addr: u64, size: usize) -> Result<(), MapError> {
        match self
            .regions
            .iter()
            .position(|r| r.start == guest_addr && r.size == size)
        {
            Some(pos) => {
                self.regions.remove(pos);
                // Guard pages: close the region's host pages, except any page still
                // touched by a surviving region.
                self.reprotect(guest_addr, size, false);
                Ok(())
            }
            None => Err(MapError::OutOfBounds),
        }
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
    /// In default mode this only checks region metadata. Recovery mode also performs
    /// an atomic same-value write to one byte per touched host page: guest data is
    /// unchanged, but this breaks copy-on-write and sets dirty/soft-dirty tracking
    /// bits, including on earlier pages when a later page faults.
    ///
    /// The wide-store helpers pre-probe their whole destination so a fault leaves nothing
    /// committed. Probing with a load instead — on the reasoning that a load shares the
    /// region and `Trap` checks with a store — is not sufficient once the MMIO answer
    /// channels split: a load probe consumes `pending_mmio`, while the embedder answers a
    /// `MmioWrite` through `pending_mmio_write`, so the retry re-probes forever.
    /// Reporting the region's verdict without consuming anything is what the probe wants.
    pub fn probe_write(&self, addr: u64, size: u8) -> Result<(), MemTrap> {
        if self.recover_host_faults {
            self.probe_write_mode::<true>(addr, size)
        } else {
            self.probe_write_mode::<false>(addr, size)
        }
    }

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

    /// The region containing `[addr, addr + size)`, or `Unmapped` if the range
    /// escapes every mapped region; a non-canonical miss is `NonCanonical`.
    /// Shared by scalar read/write.
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

    unsafe fn copy_to_guest(&self, ptr: *mut u8, buf: &[u8]) -> Result<(), usize> {
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
            self.read_ram_checked_mode::<true>(addr, buf)
        } else {
            self.read_ram_checked_mode::<false>(addr, buf)
        }
    }

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
            self.write_ram_checked_mode::<true>(addr, bytes)
        } else {
            self.write_ram_checked_mode::<false>(addr, bytes)
        }
    }

    #[inline(always)]
    pub(crate) fn write_ram_checked_mode<const ROUTED: bool>(
        &self,
        addr: u64,
        bytes: &[u8],
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
        unsafe {
            let ptr = self.host_base().cast_mut().add(self.host_off(addr));
            if ROUTED {
                crate::usercopy::copy_out(ptr, bytes)
            } else {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
                Ok(())
            }
        }
        .map_err(|at| {
            let addr = self.guest_base + (at - self.host_base() as usize) as u64;
            record_host_fault(addr);
            crate::fault::record_fp_fault(
                addr,
                bytes.len(),
                crate::FaultOrigin::HostAccess,
                crate::AddressPrecision::ExactByte,
            )
        })?;
        self.note_write(addr, bytes.len());
        Ok(())
    }

    /// Scalar read used by the interpreter and trap-out path (§8.1). Little-endian.
    /// MUST bounds-check (§8.2.3) — an out-of-range addr is a MemTrap, never a panic/UB.
    /// A `Trap` region yields `MemTrap::Mmio` (routed out as `Exit::MmioRead`).
    pub fn read(&self, addr: u64, size: u8) -> Result<u64, MemTrap> {
        if self.recover_host_faults {
            self.read_mode::<true>(addr, size)
        } else {
            self.read_mode::<false>(addr, size)
        }
    }

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

    #[inline]
    pub(crate) fn write_mode<const ROUTED: bool>(
        &self,
        addr: u64,
        val: u64,
        size: u8,
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
            unsafe { crate::usercopy::store(ptr, val, size) }.map_err(|at| self.host_fault(at))?;
        } else {
            unsafe { atomic_store_le(ptr, val, size) };
        }
        self.note_write(addr, size as usize); // SMC: catch a store onto a code page (§10)
        Ok(())
    }

    /// Atomic read-modify-write on a mapped RAM location (§8.2.3, §11). Returns the
    /// prior value (size-masked). Uses Rust SeqCst host atomics. Full x86 locked-op
    /// ordering (SDM Vol 3A §11.2.3.9) is not guaranteed for aarch64 LL/SC paths
    /// without an additional full fence. A naturally-aligned access uses a real
    /// host atomic; a misaligned
    /// one (rare; x86 permits it via a bus lock) falls back to a plain RMW — the
    /// *value* is identical, only cross-thread atomicity is lost, which aligned
    /// guest atomics never hit.
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
        let region = self.region_at(addr, size)?;
        if matches!(region.kind, RegionKind::Trap) {
            return Err(MemTrap::Mmio);
        }
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
            let old = unsafe { atomic_rmw_raw(ptr, src, size, op) };
            self.note_write(addr, size as usize);
            return Ok(old & mask_bits(size));
        }
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
        self.note_write(addr, size as usize);
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
        self.note_write(addr, 16);
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
        let region = self.region_at(addr, size)?;
        if matches!(region.kind, RegionKind::Trap) {
            return Err(MemTrap::Mmio);
        }
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
            unsafe { crate::usercopy::cas(ptr, expected & mask_bits(size), src, size) }
                .map_err(|at| self.host_fault(at))?
        } else {
            unsafe { atomic_cas_raw(ptr, expected & mask_bits(size), src, size) }
        };
        self.note_write(addr, size as usize);
        Ok(old & mask_bits(size))
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

/// Raw atomic RMW dispatch over a guest pointer. Aligned → real host atomic;
/// misaligned → plain read/modify/write (see `Memory::atomic_rmw`).
///
/// # Safety
/// `ptr` must point to `size` valid, mapped bytes inside the backing buffer.
unsafe fn atomic_rmw_raw(ptr: *mut u8, src: u64, size: u8, op: RmwOp) -> u64 {
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
                let old = unsafe { plain_read(ptr, size) };
                let new = apply_rmw(old, src, op, size);
                unsafe { plain_write(ptr, new, size) };
                old
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
unsafe fn atomic_cas_raw(ptr: *mut u8, expected: u64, src: u64, size: u8) -> u64 {
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
                let old = unsafe { plain_read(ptr, size) };
                if old == expected {
                    unsafe { plain_write(ptr, src, size) };
                }
                old
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
/// byte copy — aligned guest accesses never rely on atomicity there (same rule as
/// [`atomic_rmw_raw`]). `Relaxed` is sufficient: this fixes single-copy atomicity
/// (no tearing), not inter-location ordering, which x86 TSO handles separately.
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
            assert_eq!(pages, vec![1]);
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
    fn trap_region_routes_to_mmio() {
        let mut m = flat(0x1000);
        m.map(0x200, 0x10, Prot::RW, RegionKind::Trap).unwrap();
        assert!(matches!(m.read(0x200, 4), Err(MemTrap::Mmio)));
        assert!(matches!(m.write(0x200, 0, 4), Err(MemTrap::Mmio)));
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
        for table in [&m.code_page, &m.watch.watch, &m.watch.dirty] {
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
        m.code_page = fresh_code_pages(0, TOP);
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

#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(crate) fn host_page_size() -> usize {
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
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(crate) fn host_page_size() -> usize {
    4096
}

// A borrowed specialization has exactly Memory's representation, avoiding a
// pointer-to-pointer on out-of-line interpreter helpers.
#[repr(transparent)]
pub(crate) struct Access<const ROUTED: bool>(pub Memory);
impl<const ROUTED: bool> std::ops::Deref for Access<ROUTED> {
    type Target = Memory;
    fn deref(&self) -> &Memory {
        &self.0
    }
}
impl<const ROUTED: bool> Access<ROUTED> {
    #[inline(always)]
    pub fn new(mem: &Memory) -> &Self {
        // SAFETY: transparent wrapper; shared borrowing retains Memory's lifetime
        // and interior-mutability contract. No owned Access is constructed.
        unsafe { &*(mem as *const Memory).cast::<Self>() }
    }
    #[inline(always)]
    pub fn read(&self, addr: u64, size: u8) -> Result<u64, MemTrap> {
        self.0.read_mode::<ROUTED>(addr, size)
    }
    #[inline(always)]
    pub fn write(&self, addr: u64, value: u64, size: u8) -> Result<(), MemTrap> {
        self.0.write_mode::<ROUTED>(addr, value, size)
    }
    #[inline(always)]
    pub fn probe_write(&self, addr: u64, size: u8) -> Result<(), MemTrap> {
        self.0.probe_write_mode::<ROUTED>(addr, size)
    }
    pub fn atomic_rmw(&self, addr: u64, src: u64, size: u8, op: RmwOp) -> Result<u64, MemTrap> {
        self.0.atomic_rmw_mode::<ROUTED>(addr, src, size, op)
    }
    pub fn atomic_cas128(&self, addr: u64, expected: u128, src: u128) -> Result<u128, MemTrap> {
        self.0.atomic_cas128_mode::<ROUTED>(addr, expected, src)
    }
    pub fn atomic_cas(&self, addr: u64, expected: u64, src: u64, size: u8) -> Result<u64, MemTrap> {
        self.0.atomic_cas_mode::<ROUTED>(addr, expected, src, size)
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
