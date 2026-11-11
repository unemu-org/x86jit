//! Dense code storage with generation-checked reclamation (spec §8.2; decision-17).
//! The backend excludes execution and signal readers before releasing an extent.
use crate::memory_stats::{CodeCounters, ProviderChoice};
use cranelift_jit::{BranchProtection, JITMemoryKind, JITMemoryProvider};
use cranelift_module::{ModuleError, ModuleResult};
use std::collections::HashMap;
use std::io;
use std::ops::Range;
use std::sync::{atomic::Ordering, Arc};

use x86jit_core::codemap::ProcessIdentity;

const REGION_BYTES: usize = 128 << 20;

/// Protection policy for anonymous code storage without backing descriptors.
/// On macOS the backend retains MAP_JIT and per-thread protection for both policies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CodeMemoryPolicy {
    /// Pack each publication batch into RW pages, then permanently seal them RX.
    /// The final partial page of each batch cannot accept further allocations.
    #[default]
    WriteXorExecute,
    /// Keep code pages RWX, retaining every unused tail across publication batches.
    /// This permits writes to executable code and weakens code-integrity protection.
    ReadWriteExecute,
}

/// A dense code-memory operation that failed. Numeric values are stable diagnostic IDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[repr(u32)]
pub enum CodeMemoryFailureOp {
    /// Create the backing object: memfd_create or shm_open(SHM_ANON).
    ObjectCreate = 1,
    /// Size the backing object with ftruncate.
    ObjectSize = 2,
    /// Reserve anonymous PROT_NONE address space.
    ReserveAddressSpace = 3,
    /// Map the writable view with mmap.
    MapWritable = 4,
    /// Map the executable or read-only alias with mmap.
    MapExecutable = 5,
    /// Exclude a mapping with madvise or minherit.
    ExcludeFromFork = 6,
    /// Query macOS per-thread JIT write-protection support.
    WriteProtectSupport = 7,
    /// Enter the macOS JIT write window.
    WriteProtectDisable = 8,
    /// Restore macOS JIT execute protection.
    WriteProtectRestore = 9,
    /// Clean/invalidate the instruction-cache views.
    CacheMaintenance = 10,
    /// Synchronize instruction streams before publication.
    PipelineSync = 11,
    /// Create the macOS MAP_JIT arena with mmap.
    MapJit = 12,
    /// Create the macOS read-only alias with mach_vm_remap.
    RemapReadOnly = 13,
    /// Protect the macOS read-only alias with mprotect.
    ProtectReadOnly = 14,
    /// Commit anonymous pages as RW (or explicitly requested RWX) with mprotect.
    CommitAnonymous = 15,
    /// Seal newly written anonymous pages RX or read-only with mprotect.
    SealAnonymous = 16,
    /// Query/reject a Linux personality that makes readable pages executable.
    AnonymousPersonality = 17,
    /// Allocate storage through the system provider.
    SystemAllocate = 18,
    /// Protect system-provider code RX (including requested BTI).
    SystemProtectExecutable = 19,
    /// Protect system-provider data read-only.
    SystemProtectReadOnly = 20,
    /// Discard wholly free backing pages without changing live protections.
    DecommitPages = 21,
}

/// A provider failure, retained even when automatic fallback succeeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct CodeMemoryFailure {
    pub operation: CodeMemoryFailureOp,
    /// The original OS errno, or None for APIs/errors without an errno.
    /// Mach return codes are not errno values.
    pub errno: Option<i32>,
}

impl CodeMemoryFailure {
    pub(crate) fn decode(value: u64) -> Option<Self> {
        use CodeMemoryFailureOp::*;
        let operation = match (value >> 32) as u32 {
            1 => ObjectCreate,
            2 => ObjectSize,
            3 => ReserveAddressSpace,
            4 => MapWritable,
            5 => MapExecutable,
            6 => ExcludeFromFork,
            7 => WriteProtectSupport,
            8 => WriteProtectDisable,
            9 => WriteProtectRestore,
            10 => CacheMaintenance,
            11 => PipelineSync,
            12 => MapJit,
            13 => RemapReadOnly,
            14 => ProtectReadOnly,
            15 => CommitAnonymous,
            16 => SealAnonymous,
            17 => AnonymousPersonality,
            18 => SystemAllocate,
            19 => SystemProtectExecutable,
            20 => SystemProtectReadOnly,
            21 => DecommitPages,
            _ => return None,
        };
        let errno = value as i32;
        Some(Self {
            operation,
            errno: (errno != 0).then_some(errno),
        })
    }
}

pub(crate) fn provider(counters: Arc<CodeCounters>) -> Box<dyn JITMemoryProvider + Send> {
    // Probe before handing any pointer or relocation policy to Cranelift. Both
    // MXCSR modules share this decision; a successful region makes it permanent.
    let mut choice = counters.provider_choice.lock().unwrap();
    if *choice == ProviderChoice::System {
        return system_provider(counters.clone());
    }
    let mut memory = CodeMemory::new(counters.clone());
    if let ProviderChoice::NoFd(policy) = *choice {
        memory.no_fd = Some(policy);
    }
    if *choice == ProviderChoice::Undecided {
        match memory.reserve_region(REGION_BYTES, Kind::Code, 16) {
            Ok(_) => *choice = ProviderChoice::Dense,
            Err(_) => {
                memory.counters.failures.fetch_add(1, Ordering::Relaxed);
                counters.fallbacks.fetch_add(1, Ordering::Relaxed);
                // macOS keeps its existing MAP_JIT -> system path. Anonymous
                // fallback is for hosts whose alias backing requires descriptors.
                #[cfg(any(test, target_os = "linux", target_os = "freebsd"))]
                if anonymous_fallback_enabled() {
                    memory.no_fd = Some(CodeMemoryPolicy::WriteXorExecute);
                    match memory.probe_anonymous() {
                        Ok(()) => {
                            *choice = ProviderChoice::NoFd(CodeMemoryPolicy::WriteXorExecute);
                            return Box::new(memory);
                        }
                        Err(_) => {
                            memory.counters.failures.fetch_add(1, Ordering::Relaxed);
                            counters.fallbacks.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                *choice = ProviderChoice::System;
                return system_provider(counters.clone());
            }
        }
    }
    Box::new(memory)
}

pub(crate) fn system_provider(counters: Arc<CodeCounters>) -> Box<dyn JITMemoryProvider + Send> {
    Box::new(SystemMemory::new(counters))
}

/// Keep system storage accounting opaque while recording all failed operations.
struct SystemMemory {
    inner: cranelift_jit::SystemMemoryProvider,
    diagnostics: Arc<CodeCounters>,
    counters: Arc<crate::memory_stats::RegionCounters>,
}

impl SystemMemory {
    fn new(diagnostics: Arc<CodeCounters>) -> Self {
        Self {
            inner: cranelift_jit::SystemMemoryProvider::new(),
            counters: diagnostics.register(),
            diagnostics,
        }
    }
}

impl JITMemoryProvider for SystemMemory {
    fn allocate(&mut self, size: usize, align: u64, kind: JITMemoryKind) -> io::Result<*mut u8> {
        let result = operation(
            &self.diagnostics,
            CodeMemoryFailureOp::SystemAllocate,
            || self.inner.allocate(size, align, kind),
        );
        if result.is_err() {
            self.counters.failures.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    fn finalize(&mut self, branch: BranchProtection) -> ModuleResult<()> {
        use cranelift_jit::SystemMemoryOperation;
        let result = self.inner.finalize_with(branch, |op, action| {
            let op = match op {
                SystemMemoryOperation::ProtectReadOnly => {
                    CodeMemoryFailureOp::SystemProtectReadOnly
                }
                SystemMemoryOperation::ProtectExecutable => {
                    CodeMemoryFailureOp::SystemProtectExecutable
                }
                SystemMemoryOperation::CacheMaintenance => CodeMemoryFailureOp::CacheMaintenance,
                SystemMemoryOperation::PipelineSync => CodeMemoryFailureOp::PipelineSync,
            };
            operation(&self.diagnostics, op, action)
        });
        if result.is_err() {
            self.counters.failures.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    unsafe fn free_memory(&mut self) {
        // SAFETY: the caller guarantees no published pointers remain in use.
        unsafe { self.inner.free_memory() };
    }
}

fn check_anonymous_personality(counters: &CodeCounters) -> io::Result<()> {
    operation(counters, CodeMemoryFailureOp::AnonymousPersonality, || {
        #[cfg(test)]
        if TEST_READ_IMPLIES_EXEC.with(|flag| flag.get()) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "READ_IMPLIES_EXEC prevents anonymous W^X",
            ));
        }
        #[cfg(target_os = "linux")]
        {
            // personality(2), DESCRIPTION: 0xffffffff queries without changing
            // the personality; READ_IMPLIES_EXEC adds execute to readable maps.
            // Linux v6.12 mm/mprotect.c, do_mprotect_pkey: expands PROT_READ
            // with PROT_EXEC for this personality before applying protection.
            // https://man7.org/linux/man-pages/man2/personality.2.html
            // SAFETY: this query takes no pointers and changes no process state.
            let personality = unsafe { libc::personality(0xffff_ffff) };
            if personality == -1 {
                return Err(io::Error::last_os_error());
            }
            if personality & libc::READ_IMPLIES_EXEC != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "READ_IMPLIES_EXEC prevents anonymous W^X",
                ));
            }
        }
        Ok(())
    })
}

#[cfg(test)]
thread_local! {
    static TEST_READ_IMPLIES_EXEC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static INJECT_FAILURE: std::cell::Cell<Option<(CodeMemoryFailureOp, i32)>> = const { std::cell::Cell::new(None) };
    static INJECT_NEXT_FAILURE: std::cell::Cell<Option<(CodeMemoryFailureOp, i32)>> = const { std::cell::Cell::new(None) };
    static ANONYMOUS_PROTECTIONS: std::cell::RefCell<Vec<(usize, usize, i32)>> = const { std::cell::RefCell::new(Vec::new()) };
    // Exercise the Unix alias fallback on macOS without changing its production
    // MAP_JIT policy. ObjectCreate still uses the normal operation injection.
    static TEST_ALIAS_PROVIDER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(any(test, target_os = "linux", target_os = "freebsd"))]
fn anonymous_fallback_enabled() -> bool {
    #[cfg(test)]
    if TEST_ALIAS_PROVIDER.with(|slot| slot.get()) {
        return true;
    }
    cfg!(any(target_os = "linux", target_os = "freebsd"))
}

fn operation<T>(
    counters: &CodeCounters,
    op: CodeMemoryFailureOp,
    action: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    #[cfg(test)]
    let injected = INJECT_FAILURE.with(|slot| {
        if let Some((target, errno)) = slot.get() {
            if target == op {
                slot.set(INJECT_NEXT_FAILURE.with(|next| next.take()));
                return Some(if errno == 0 {
                    io::ErrorKind::Unsupported.into()
                } else {
                    io::Error::from_raw_os_error(errno)
                });
            }
        }
        None
    });
    #[cfg(not(test))]
    let injected: Option<io::Error> = None;
    let result = match injected {
        Some(err) => Err(err),
        None => action(),
    };
    if let Err(err) = &result {
        counters.record_failure(op, err);
    }
    result
}

#[cfg(any(test, all(target_os = "linux", target_arch = "aarch64")))]
mod linux_sync {
    use std::io;

    // Linux uapi/linux/membarrier.h, enum membarrier_cmd.
    const SYNC_CORE: i32 = 1 << 5;
    const REGISTER_SYNC_CORE: i32 = 1 << 6;

    #[derive(Debug, PartialEq, Eq)]
    pub(super) enum Protocol {
        SyncCore,
        Receiver,
    }

    impl Protocol {
        pub(super) fn register(mut membarrier: impl FnMut(i32) -> io::Result<()>) -> Self {
            if membarrier(REGISTER_SYNC_CORE).is_ok() {
                Self::SyncCore
            } else {
                Self::Receiver
            }
        }

        pub(super) fn flush(
            &self,
            mut membarrier: impl FnMut(i32) -> io::Result<()>,
        ) -> io::Result<()> {
            match self {
                Self::SyncCore => membarrier(SYNC_CORE),
                // jit_abi's generation/receiver ISB protocol covers this path.
                Self::Receiver => Ok(()),
            }
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    pub(super) fn membarrier(command: i32) -> io::Result<()> {
        // SAFETY: membarrier(2) takes a command, zero flags and an unused CPU ID.
        if unsafe { libc::syscall(libc::SYS_membarrier, command, 0i32, 0i32) } == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Code,
    ReadOnly,
    Writable,
}

struct Region {
    rw: usize,
    rx: usize,
    len: usize,
    used: usize,
    // Disjoint, coalesced holes below the bump watermark.
    free: Vec<Range<usize>>,
    // Dirty writes can lie below that watermark after reuse.
    dirty: Vec<Range<usize>>,
    // Page-aligned holes whose backing was successfully discarded.
    decommitted: Vec<Range<usize>>,
    live: usize,
    // Retained for shared-object hole punching behind both views.
    backing: Option<std::os::fd::OwnedFd>,
    // Anonymous mappings have [0, sealed) immutable RX/R, [sealed, committed)
    // writable, and [committed, len) PROT_NONE. A sealed page never becomes RW
    // again in this reservation, even after execution and reader grace. Reuse
    // requires fresh reservations; only the RW frontier accepts new writes.
    sealed: usize,
    committed: usize,
    no_fd: bool,
    kind: Kind,
}

/// Provider-owned extent, including definitions that fail before publication.
/// Generation identity is independent of both record and region indices.
#[derive(Debug)]
struct Allocation {
    region: usize,
    offset: usize,
    // Include bump-alignment padding in the allocation's reclaimable extent.
    start: usize,
    len: usize,
    generation: u64,
}

// Sorted disjoint intervals; insertion coalesces adjacency, subtraction splits.
// Their storage scales with fragmentation, rather than the number of retirements.
fn insert_range(ranges: &mut Vec<Range<usize>>, mut range: Range<usize>) {
    if range.is_empty() {
        return;
    }
    let first = ranges.partition_point(|r| r.end < range.start);
    let mut last = first;
    while last < ranges.len() && ranges[last].start <= range.end {
        range.start = range.start.min(ranges[last].start);
        range.end = range.end.max(ranges[last].end);
        last += 1;
    }
    ranges.splice(first..last, [range]);
}

fn subtract_range(ranges: &mut Vec<Range<usize>>, range: Range<usize>) {
    let mut i = 0;
    while i < ranges.len() {
        let old = ranges[i].clone();
        if old.start >= range.end {
            break;
        }
        if old.end <= range.start {
            i += 1;
            continue;
        }
        ranges.remove(i);
        if old.start < range.start {
            ranges.insert(i, old.start..range.start);
            i += 1;
        }
        if old.end > range.end {
            ranges.insert(i, range.end..old.end);
            break;
        }
    }
}

fn range_bytes(ranges: &[Range<usize>]) -> usize {
    ranges.iter().map(|r| r.len()).sum()
}

impl Region {
    fn reusable(&self) -> bool {
        // Descriptor-free reservations are reclaimed as a whole. In particular,
        // sealed RX/R pages never become writable in the same reservation.
        !self.no_fd
    }
}

pub(crate) struct CodeMemory {
    owner: ProcessIdentity,
    diagnostics: Arc<CodeCounters>,
    regions: Vec<Region>,
    allocations: Vec<Allocation>,
    allocation_by_rx: HashMap<usize, usize>,
    counters: Arc<crate::memory_stats::RegionCounters>,
    no_fd: Option<CodeMemoryPolicy>,
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    instruction_sync: linux_sync::Protocol,
}

impl CodeMemory {
    pub(crate) fn new(counters: Arc<CodeCounters>) -> Self {
        Self {
            owner: ProcessIdentity::current(),
            regions: Vec::new(),
            allocations: Vec::new(),
            allocation_by_rx: HashMap::new(),
            counters: counters.register(),
            diagnostics: counters,
            no_fd: None,
            #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
            instruction_sync: linux_sync::Protocol::register(linux_sync::membarrier),
        }
    }

    fn reserve_region(&mut self, len: usize, kind: Kind, align: usize) -> io::Result<usize> {
        // Vacant region indices are stable: another live allocation can retain
        // its index when an empty neighbor is unmapped.
        let index = self.regions.iter().position(|r| r.len == 0);
        if index.is_none() {
            self.regions.try_reserve(1).map_err(io::Error::other)?;
        }
        let (rw, rx, backing) = if self.no_fd.is_some() {
            let address = os::reserve_anonymous(len, &self.diagnostics)?;
            (address, address, None)
        } else {
            os::reserve(len, kind, align, &self.diagnostics)?
        };
        let region = Region {
            rw,
            rx,
            len,
            used: 0,
            free: Vec::new(),
            dirty: Vec::new(),
            decommitted: Vec::new(),
            live: 0,
            backing,
            sealed: 0,
            committed: 0,
            no_fd: self.no_fd.is_some(),
            kind,
        };
        let index = if let Some(index) = index {
            self.regions[index] = region;
            index
        } else {
            self.regions.push(region);
            self.regions.len() - 1
        };
        self.update_counters();
        Ok(index)
    }

    #[cfg(any(test, target_os = "linux", target_os = "freebsd"))]
    fn probe_anonymous(&mut self) -> io::Result<()> {
        check_anonymous_personality(&self.diagnostics)?;
        self.reserve_region(REGION_BYTES, Kind::Code, 16)?;
        let r = self.regions.last_mut().unwrap();
        let page = region::page::size();
        // Probe both writer and executable permissions before selecting the
        // backend-wide policy. This empty sealed page is never reopened.
        let result = os::commit_anonymous(r.rw, page, false, &self.diagnostics)
            .and_then(|()| os::seal_anonymous(r.rw, page, true, &self.diagnostics));
        if let Err(err) = result {
            let r = self.regions.pop().unwrap();
            // SAFETY: failed startup storage has never escaped this provider.
            unsafe { os::release(&r) };
            // The failed provider is discarded; its retained diagnostic counters
            // must not report vector storage that is about to be dropped.
            self.regions = Vec::new();
            self.update_counters();
            return Err(err);
        }
        r.used = page;
        r.sealed = page;
        r.committed = page;
        self.update_counters();
        Ok(())
    }

    fn allocate_inner(
        &mut self,
        size: usize,
        align: u64,
        kind: JITMemoryKind,
    ) -> io::Result<*mut u8> {
        let kind = match kind {
            JITMemoryKind::Executable => Kind::Code,
            JITMemoryKind::ReadOnly => Kind::ReadOnly,
            JITMemoryKind::Writable => Kind::Writable,
        };
        let align = usize::try_from(align).map_err(|_| io::ErrorKind::InvalidInput)?;
        if size == 0 || !align.is_power_of_two() {
            return Err(io::ErrorKind::InvalidInput.into());
        }
        let host_len = u32::try_from(size).map_err(|_| io::ErrorKind::InvalidInput)?;
        // Reserve the bookkeeping before consuming an extent. Allocation failure
        // must use the same sticky CodeSpaceExhausted path as mmap failure.
        self.allocations.try_reserve(1).map_err(io::Error::other)?;
        self.allocation_by_rx
            .try_reserve(1)
            .map_err(io::Error::other)?;
        let fits = |r: &Region| {
            let offset = r.rw.checked_add(r.used)?.checked_add(align - 1)? & !(align - 1);
            let offset = offset.checked_sub(r.rw)?;
            (offset.checked_add(size)? <= r.len && (r.rx.checked_add(offset)? & (align - 1)) == 0)
                .then_some(offset)
        };
        let hole = self.regions.iter().enumerate().find_map(|(i, r)| {
            if r.kind != kind || !r.reusable() {
                return None;
            }
            r.free.iter().find_map(|hole| {
                let offset = r.rw.checked_add(hole.start)?.checked_add(align - 1)? & !(align - 1);
                let offset = offset.checked_sub(r.rw)?;
                let end = if hole.end == r.used { r.len } else { hole.end };
                (offset.checked_add(size)? <= end && (r.rx + offset) & (align - 1) == 0)
                    .then_some((i, offset))
            })
        });
        let found = hole.or_else(|| {
            self.regions
                .iter()
                .enumerate()
                .rev()
                .find_map(|(i, r)| (r.kind == kind).then(|| fits(r)).flatten().map(|p| (i, p)))
        });
        let (index, offset) = match found {
            Some(found) => found,
            None => {
                let page = region::page::size();
                let len = size
                    .checked_add(align - 1)
                    .and_then(|v| v.checked_add(page - 1))
                    .map(|v| (v & !(page - 1)).max(REGION_BYTES))
                    .ok_or(io::ErrorKind::OutOfMemory)?;
                let index = self.reserve_region(len, kind, align)?;
                (
                    index,
                    fits(&self.regions[index]).expect("new region accommodates alignment"),
                )
            }
        };
        let r = &mut self.regions[index];
        if let Some(policy) = self.no_fd {
            let end = (offset + size).div_ceil(region::page::size()) * region::page::size();
            if end > r.committed {
                if policy == CodeMemoryPolicy::WriteXorExecute {
                    check_anonymous_personality(&self.diagnostics)?;
                }
                // Never change protection on previously committed/sealed pages.
                os::commit_anonymous(
                    r.rw + r.committed,
                    end - r.committed,
                    kind == Kind::Code && policy == CodeMemoryPolicy::ReadWriteExecute,
                    &self.diagnostics,
                )?;
                r.committed = end;
            }
            debug_assert!(r.reusable() || offset >= r.sealed);
        }
        let start = if hole.is_some() {
            offset.min(r.used)
        } else {
            r.used
        };
        if hole.is_some() {
            subtract_range(&mut r.free, offset..offset + size);
        }
        r.used = r.used.max(offset + size);
        r.live += 1;
        insert_range(&mut r.dirty, offset..offset + size);
        let page = region::page::size();
        subtract_range(
            &mut r.decommitted,
            offset / page * page..(offset + size).div_ceil(page) * page,
        );
        let unit = x86jit_core::code_lifetime::CodeUnit::new(r.rx + offset, host_len);
        self.allocation_by_rx
            .insert(unit.host_start, self.allocations.len());
        self.allocations.push(Allocation {
            region: index,
            offset,
            start,
            len: size,
            generation: unit.generation,
        });
        let ptr = (r.rw + offset) as *mut u8;
        self.update_counters();
        Ok(ptr)
    }

    fn check_process(&self) -> io::Result<()> {
        x86jit_core::codemap::check_jit_allowed()
            .map_err(|error| io::Error::new(io::ErrorKind::PermissionDenied, error))?;
        if self.owner != ProcessIdentity::current() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "code memory is unavailable after fork; rebuild the engine",
            ));
        }
        Ok(())
    }

    fn update_counters(&self) {
        let c = &self.counters;
        let mut regions = 0;
        let mut reserved = 0;
        let mut used = 0;
        let mut committed = 0;
        let mut largest = 0;
        for r in self
            .regions
            .iter()
            .filter(|r| r.kind == Kind::Code && r.len != 0)
        {
            regions += 1;
            reserved += r.len as u64;
            let holes = if r.reusable() {
                range_bytes(&r.free)
            } else {
                0
            };
            used += (r.used - holes) as u64;
            committed += (if r.no_fd {
                r.committed as u64
            } else {
                (r.used.div_ceil(region::page::size()) * region::page::size()) as u64
            }) - range_bytes(&r.decommitted) as u64;
            largest = largest.max((r.len - r.used) as u64);
            if r.reusable() {
                largest = largest.max(
                    r.free
                        .iter()
                        .map(|hole| {
                            let end = if hole.end == r.used { r.len } else { hole.end };
                            (end - hole.start) as u64
                        })
                        .max()
                        .unwrap_or(0),
                );
            }
        }
        c.regions.store(regions, Ordering::Relaxed);
        c.views.store(
            if self.no_fd.is_some() {
                regions
            } else if cfg!(target_os = "macos") {
                u64::from(regions != 0)
            } else {
                regions * 2
            },
            Ordering::Relaxed,
        );
        c.reserved.store(reserved, Ordering::Relaxed);
        c.used.store(used, Ordering::Relaxed);
        c.committed.store(committed, Ordering::Relaxed);
        c.free.store(reserved - used, Ordering::Relaxed);
        c.reusable.store(
            self.regions
                .iter()
                .filter(|r| r.reusable())
                .map(|r| range_bytes(&r.free) as u64)
                .sum(),
            Ordering::Relaxed,
        );
        c.largest.store(largest, Ordering::Relaxed);
        c.allocation_metadata.store(
            (self.allocations.capacity() * std::mem::size_of::<Allocation>()
                + self.allocation_by_rx.capacity() * 2 * std::mem::size_of::<usize>()
                + self.regions.capacity() * std::mem::size_of::<Region>()
                + self
                    .regions
                    .iter()
                    .map(|r| {
                        (r.free.capacity() + r.dirty.capacity() + r.decommitted.capacity())
                            * std::mem::size_of::<Range<usize>>()
                    })
                    .sum::<usize>()) as u64,
            Ordering::Relaxed,
        );
    }
}

impl JITMemoryProvider for CodeMemory {
    unsafe fn release_allocation(&mut self, executable: *const u8, generation: u64) -> bool {
        if self.check_process().is_err() {
            return false;
        }
        let address = executable as usize;
        let Some(&index) = self.allocation_by_rx.get(&address) else {
            return false;
        };
        if self.allocations[index].generation != generation {
            return false;
        }
        let allocation = self.allocations.swap_remove(index);
        self.allocation_by_rx.remove(&address);
        if let Some(moved) = self.allocations.get(index) {
            let address = self.regions[moved.region].rx + moved.offset;
            *self.allocation_by_rx.get_mut(&address).unwrap() = index;
        }
        let r = &mut self.regions[allocation.region];
        r.live -= 1;
        insert_range(
            &mut r.free,
            allocation.start..allocation.offset + allocation.len,
        );
        subtract_range(
            &mut r.dirty,
            allocation.offset..allocation.offset + allocation.len,
        );
        if r.live == 0 && !(cfg!(target_os = "macos") && !r.no_fd && r.kind == Kind::Code) {
            // SAFETY: every allocation in these views has passed both grace
            // periods. Do not shift another region's stable allocation index.
            unsafe { os::release(r) };
            r.backing = None;
            r.len = 0;
            r.used = 0;
            r.committed = 0;
            r.sealed = 0;
            r.free.clear();
            r.dirty.clear();
            r.decommitted.clear();
        } else if r.reusable() {
            // Only pages wholly inside coalesced free extents may be discarded.
            // Include the unwritten tail of the last touched page. No protection
            // change, backing operation or write touches a live unit's page.
            let page = region::page::size();
            for hole in &r.free {
                let start = hole.start.div_ceil(page) * page;
                let end = if hole.end == r.used {
                    hole.end.div_ceil(page) * page
                } else {
                    hole.end / page * page
                };
                if start >= end {
                    continue;
                }
                let mut pending = std::iter::once(start..end).collect::<Vec<_>>();
                for old in &r.decommitted {
                    subtract_range(&mut pending, old.clone());
                }
                for pages in pending {
                    match operation(
                        &self.diagnostics,
                        CodeMemoryFailureOp::DecommitPages,
                        || os::decommit(r, pages.clone()),
                    ) {
                        Ok(()) => insert_range(&mut r.decommitted, pages),
                        Err(_) => {
                            // Failed decommit retains commitment. The extent is
                            // still safe to reuse after the caller's grace periods.
                            self.counters.failures.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            }
        }
        self.update_counters();
        true
    }

    fn allocation_generation(&self, executable: *const u8) -> Option<u64> {
        self.check_process().ok()?;
        let allocation = &self.allocations[*self.allocation_by_rx.get(&(executable as usize))?];
        let region = &self.regions[allocation.region];
        debug_assert_eq!(region.rx + allocation.offset, executable as usize);
        debug_assert!(allocation.offset + allocation.len <= region.used);
        Some(allocation.generation)
    }

    fn allocate(&mut self, size: usize, align: u64, kind: JITMemoryKind) -> io::Result<*mut u8> {
        self.check_process()?;
        let result = self.allocate_inner(size, align, kind);
        if result.is_err() {
            self.counters.failures.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    fn supports_relocation(&self, kind: cranelift::codegen::binemit::Reloc) -> bool {
        use cranelift::codegen::binemit::Reloc;
        matches!(
            kind,
            Reloc::Abs4
                | Reloc::Abs8
                | Reloc::X86PCRel4
                | Reloc::X86CallPCRel4
                | Reloc::Arm64Call
                | Reloc::Aarch64AdrPrelPgHi21
                | Reloc::Aarch64AddAbsLo12Nc
        )
    }

    fn executable_ptr(&self, writable: *mut u8) -> *const u8 {
        self.check_process()
            .expect("provider belongs to this process");
        let address = writable as usize;
        let r = self
            .regions
            .iter()
            .rev()
            .find(|r| address >= r.rw && address - r.rw < r.used)
            .expect("provider-owned allocation");
        (r.rx + address - r.rw) as *const u8
    }

    fn with_write_access(&self, action: &mut dyn FnMut()) -> io::Result<()> {
        self.check_process()?;
        let result = if self.no_fd.is_some() {
            action();
            Ok(())
        } else {
            os::with_write_access(action, &self.diagnostics)
        };
        if result.is_err() {
            self.counters.failures.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    fn finalize(&mut self, branch_protection: BranchProtection) -> ModuleResult<()> {
        self.check_process()
            .map_err(|err| ModuleError::Allocation { err })?;
        let result = (|| -> io::Result<()> {
            // BTI is not enabled by x86jit's ISA. Reject rather than silently
            // publishing mappings without a requested protection policy.
            if branch_protection != BranchProtection::None {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "JIT BTI mappings are unsupported",
                ));
            }
            for r in &self.regions {
                if r.kind == Kind::Code {
                    for dirty in &r.dirty {
                        operation(
                            &self.diagnostics,
                            CodeMemoryFailureOp::CacheMaintenance,
                            || os::clear_cache(r.rw + dirty.start, r.rx + dirty.start, dirty.len()),
                        )?;
                    }
                }
            }
            for r in &mut self.regions {
                if self.no_fd.is_some()
                    && r.kind != Kind::Writable
                    && !(r.kind == Kind::Code
                        && self.no_fd == Some(CodeMemoryPolicy::ReadWriteExecute))
                    && r.committed > r.sealed
                {
                    os::seal_anonymous(
                        r.rw + r.sealed,
                        r.committed - r.sealed,
                        r.kind == Kind::Code,
                        &self.diagnostics,
                    )?;
                    // Advance immediately: even if pipeline sync fails, no
                    // later operation can allocate/write into these RX pages.
                    r.sealed = r.committed;
                    r.used = r.committed;
                }
            }
            // Require SYNC_CORE after registration; never accept the pinned
            // helper's GLOBAL fallback as instruction-stream synchronization.
            #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
            operation(&self.diagnostics, CodeMemoryFailureOp::PipelineSync, || {
                self.instruction_sync.flush(linux_sync::membarrier)
            })?;
            #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
            operation(&self.diagnostics, CodeMemoryFailureOp::PipelineSync, || {
                #[cfg(target_os = "linux")]
                {
                    wasmtime_jit_icache_coherence::pipeline_flush_mt()
                }
                #[cfg(not(target_os = "linux"))]
                {
                    wasmtime_jit_icache_coherence::pipeline_flush_mt().map_err(io::Error::other)
                }
            })?;
            Ok(())
        })();
        self.update_counters();
        if let Err(err) = result {
            self.counters.failures.fetch_add(1, Ordering::Relaxed);
            return Err(ModuleError::Allocation { err });
        }
        x86jit_core::jit_abi::code_publication_barrier();
        for r in &mut self.regions {
            r.dirty.clear();
        }
        Ok(())
    }

    unsafe fn free_memory(&mut self) {
        // SAFETY: the caller must exclude execution and signal readers. Normal
        // backend teardown calls this only after both grace periods.
        if self.check_process().is_err() {
            return;
        }
        for r in self.regions.drain(..) {
            if r.len != 0 {
                unsafe { os::release(&r) };
            }
        }
        self.allocations.clear();
        self.allocation_by_rx.clear();
        self.update_counters();
    }
}

#[cfg(any(target_os = "linux", target_os = "freebsd", target_os = "macos"))]
mod os {
    use super::*;

    // mach/vm_statistics.h: application-specific VM tag, preserved in region
    // queries even when an allocation fails before it reaches provider accounting.
    #[cfg(target_os = "macos")]
    #[allow(deprecated)]
    const ANONYMOUS_FD: i32 = libc::VM_MEMORY_APPLICATION_SPECIFIC_16 << 24;
    #[cfg(not(target_os = "macos"))]
    const ANONYMOUS_FD: i32 = -1;

    #[cfg(all(test, not(target_os = "macos")))]
    fn observe_mapping(ptr: *mut libc::c_void, len: usize) {
        crate::code_memory_probe::observe_mapping(ptr as usize, len);
    }

    #[cfg(any(not(test), target_os = "macos"))]
    fn observe_mapping(_: *mut libc::c_void, _: usize) {}

    pub(super) fn reserve_anonymous(len: usize, counters: &CodeCounters) -> io::Result<usize> {
        let ptr = operation(counters, CodeMemoryFailureOp::ReserveAddressSpace, || {
            // SAFETY: the kernel chooses a fresh address; no descriptor or fixed
            // mapping is used. The reservation has no accessible pages yet.
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_NONE,
                    // One anonymous shared backing keeps adjacent sealed pages
                    // mergeable without per-frontier private COW identities.
                    // No descriptor is created in the process descriptor table.
                    libc::MAP_SHARED | libc::MAP_ANON,
                    ANONYMOUS_FD,
                    0,
                )
            };
            if ptr == libc::MAP_FAILED {
                Err(io::Error::last_os_error())
            } else {
                observe_mapping(ptr, len);
                Ok(ptr)
            }
        })?;
        exclude_or_unmap(ptr, len, counters).map(|ptr| ptr as usize)
    }

    fn protect_anonymous(
        start: usize,
        len: usize,
        prot: i32,
        counters: &CodeCounters,
        op: CodeMemoryFailureOp,
    ) -> io::Result<()> {
        operation(counters, op, || {
            // SAFETY: the caller supplies a page-aligned extent wholly inside
            // its owned reservation, excluding previously sealed pages.
            if unsafe { libc::mprotect(start as *mut _, len, prot) } != 0 {
                Err(io::Error::last_os_error())
            } else {
                #[cfg(test)]
                ANONYMOUS_PROTECTIONS.with(|log| log.borrow_mut().push((start, len, prot)));
                Ok(())
            }
        })
    }

    pub(super) fn commit_anonymous(
        start: usize,
        len: usize,
        rwx: bool,
        counters: &CodeCounters,
    ) -> io::Result<()> {
        protect_anonymous(
            start,
            len,
            libc::PROT_READ | libc::PROT_WRITE | if rwx { libc::PROT_EXEC } else { 0 },
            counters,
            CodeMemoryFailureOp::CommitAnonymous,
        )
    }

    pub(super) fn seal_anonymous(
        start: usize,
        len: usize,
        executable: bool,
        counters: &CodeCounters,
    ) -> io::Result<()> {
        protect_anonymous(
            start,
            len,
            libc::PROT_READ | if executable { libc::PROT_EXEC } else { 0 },
            counters,
            CodeMemoryFailureOp::SealAnonymous,
        )
    }

    #[cfg(target_os = "macos")]
    pub(super) fn jit_write_protection_supported() -> bool {
        // SAFETY: pthread API has no arguments or memory preconditions.
        unsafe { libc::pthread_jit_write_protect_supported_np() != 0 }
    }

    pub(super) fn reserve(
        len: usize,
        kind: Kind,
        align: usize,
        counters: &CodeCounters,
    ) -> io::Result<(usize, usize, Option<std::os::fd::OwnedFd>)> {
        #[cfg(test)]
        if TEST_ALIAS_PROVIDER.with(|slot| slot.get()) {
            return operation(counters, CodeMemoryFailureOp::ObjectCreate, || {
                Err(io::ErrorKind::PermissionDenied.into())
            });
        }
        #[cfg(target_os = "macos")]
        {
            let (rw, rx) = if kind == Kind::Code {
                mac::reserve(len, align, counters)?
            } else {
                mac::reserve_data(len, kind, align, counters)?
            };
            Ok((rw, rx, None))
        }
        #[cfg(not(target_os = "macos"))]
        {
            // Linux memfd_create(2); FreeBSD shm_open(2), SHM_ANON. Separate
            // aliases retain all dense tails without changing executable pages.
            let fd = operation(counters, CodeMemoryFailureOp::ObjectCreate, || {
                #[cfg(target_os = "linux")]
                // SAFETY: static NUL-terminated name and valid flags.
                let fd = unsafe { libc::memfd_create(c"x86jit-code".as_ptr(), libc::MFD_CLOEXEC) };
                #[cfg(target_os = "freebsd")]
                // SAFETY: SHM_ANON requests an unnamed shared object.
                let fd = unsafe {
                    libc::shm_open(libc::SHM_ANON, libc::O_RDWR | libc::O_CLOEXEC, 0o600)
                };
                if fd < 0 {
                    Err(io::Error::last_os_error())
                } else {
                    Ok(fd)
                }
            })?;
            use std::os::fd::FromRawFd;
            // SAFETY: the newly opened descriptor is uniquely owned here.
            let backing = unsafe { std::os::fd::OwnedFd::from_raw_fd(fd) };
            // SAFETY: mapping calls use this owned backing. Every partial
            // failure unmaps already created views and drops the descriptor.
            unsafe {
                let length = libc::off_t::try_from(len).map_err(|_| io::ErrorKind::OutOfMemory)?;
                operation(counters, CodeMemoryFailureOp::ObjectSize, || {
                    if libc::ftruncate(fd, length) != 0 {
                        Err(io::Error::last_os_error())
                    } else {
                        Ok(())
                    }
                })?;
                let rw = map_aligned(
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    fd,
                    align,
                    counters,
                    CodeMemoryFailureOp::MapWritable,
                )?;
                if kind == Kind::Writable {
                    return Ok((rw as usize, rw as usize, Some(backing)));
                }
                let prot = libc::PROT_READ
                    | if kind == Kind::Code {
                        libc::PROT_EXEC
                    } else {
                        0
                    };
                let rx = match map_aligned(
                    len,
                    prot,
                    libc::MAP_SHARED,
                    fd,
                    align,
                    counters,
                    CodeMemoryFailureOp::MapExecutable,
                ) {
                    Ok(rx) => rx,
                    Err(error) => {
                        libc::munmap(rw, len);
                        return Err(error);
                    }
                };
                Ok((rw as usize, rx as usize, Some(backing)))
            }
        }
    }

    // Overaligned views use a private PROT_NONE reservation first; MAP_FIXED
    // replaces only addresses owned by that reservation, never arbitrary memory.
    fn map_aligned(
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        align: usize,
        counters: &CodeCounters,
        op: CodeMemoryFailureOp,
    ) -> io::Result<*mut libc::c_void> {
        // SAFETY: size is page-rounded, fd/flags describe the desired backing;
        // all partial reservations are released before returning an error.
        unsafe {
            if align <= region::page::size() {
                let ptr = operation(counters, op, || {
                    let ptr = libc::mmap(std::ptr::null_mut(), len, prot, flags, fd, 0);
                    if ptr == libc::MAP_FAILED {
                        Err(io::Error::last_os_error())
                    } else {
                        observe_mapping(ptr, len);
                        Ok(ptr)
                    }
                })?;
                return exclude_or_unmap(ptr, len, counters);
            }
            let total = len.checked_add(align).ok_or(io::ErrorKind::OutOfMemory)?;
            let reservation =
                operation(counters, CodeMemoryFailureOp::ReserveAddressSpace, || {
                    let reservation = libc::mmap(
                        std::ptr::null_mut(),
                        total,
                        libc::PROT_NONE,
                        libc::MAP_PRIVATE | libc::MAP_ANON,
                        ANONYMOUS_FD,
                        0,
                    );
                    if reservation == libc::MAP_FAILED {
                        Err(io::Error::last_os_error())
                    } else {
                        observe_mapping(reservation, total);
                        Ok(reservation)
                    }
                })?;
            let start = reservation as usize;
            let aligned = (start + align - 1) & !(align - 1);
            let ptr = match operation(counters, op, || {
                let ptr = libc::mmap(aligned as *mut _, len, prot, flags | libc::MAP_FIXED, fd, 0);
                if ptr == libc::MAP_FAILED {
                    Err(io::Error::last_os_error())
                } else {
                    Ok(ptr)
                }
            }) {
                Ok(ptr) => ptr,
                Err(error) => {
                    libc::munmap(reservation, total);
                    return Err(error);
                }
            };
            if aligned > start {
                libc::munmap(reservation, aligned - start);
            }
            if aligned + len < start + total {
                libc::munmap((aligned + len) as *mut _, start + total - aligned - len);
            }
            exclude_or_unmap(ptr, len, counters)
        }
    }

    fn exclude_or_unmap(
        ptr: *mut libc::c_void,
        len: usize,
        counters: &CodeCounters,
    ) -> io::Result<*mut libc::c_void> {
        if let Err(error) = exclude_from_fork(ptr, len, counters) {
            // SAFETY: this mapping has not escaped.
            unsafe { libc::munmap(ptr, len) };
            return Err(error);
        }
        Ok(ptr)
    }

    fn exclude_from_fork(
        ptr: *mut libc::c_void,
        len: usize,
        counters: &CodeCounters,
    ) -> io::Result<()> {
        operation(counters, CodeMemoryFailureOp::ExcludeFromFork, || {
            // SAFETY: the complete page-aligned mapping is owned by this provider.
            #[cfg(target_os = "linux")]
            let result = unsafe { libc::madvise(ptr, len, libc::MADV_DONTFORK) };
            #[cfg(any(target_os = "macos", target_os = "freebsd"))]
            let result = {
                unsafe extern "C" {
                    fn minherit(address: *mut libc::c_void, len: usize, inheritance: i32) -> i32;
                }
                // VM_INHERIT_NONE (Mach) and INHERIT_NONE (FreeBSD) are both 2.
                // SAFETY: ptr and len describe the complete owned mapping.
                unsafe { minherit(ptr, len, 2) }
            };
            if result != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        })
    }

    pub(super) fn with_write_access(
        action: &mut dyn FnMut(),
        counters: &CodeCounters,
    ) -> io::Result<()> {
        #[cfg(target_os = "macos")]
        return mac::write(action, counters);
        #[cfg(not(target_os = "macos"))]
        {
            let _ = counters;
            action();
            Ok(())
        }
    }

    #[cfg(all(test, target_os = "macos"))]
    pub(super) fn fork_with_locked_arena() -> libc::pid_t {
        mac::fork_with_locked_arena()
    }

    pub(super) fn clear_cache(rw: usize, rx: usize, len: usize) -> io::Result<()> {
        #[cfg(target_os = "macos")]
        {
            unsafe extern "C" {
                fn sys_icache_invalidate(start: *mut libc::c_void, len: usize);
            }
            // SAFETY: both bounds describe newly initialized code. Apple's
            // cache(3) / Apple silicon JIT porting guide requires this OS API.
            unsafe {
                sys_icache_invalidate(rx as *mut _, len);
            }
            let _ = rw;
        }
        #[cfg(not(target_os = "macos"))]
        {
            // SAFETY: both views cover the initialized code extent. Clean data
            // through the writer VA before invalidating the executable VA.
            #[cfg(target_arch = "aarch64")]
            unsafe {
                let ctr: usize;
                core::arch::asm!("mrs {ctr}, ctr_el0", ctr = out(reg) ctr, options(nostack, preserves_flags));
                let line_bytes = 4 << ((ctr >> 16) & 15);
                let mut line = rw & !(line_bytes - 1);
                while line < rw + len {
                    core::arch::asm!("dc cvau, {line}", line = in(reg) line, options(nostack));
                    line += line_bytes;
                }
                core::arch::asm!("dsb ish", options(nostack));
            }
            let _ = rw;
            // SAFETY: rx is a valid readable/executable view of len bytes.
            let result = unsafe { wasmtime_jit_icache_coherence::clear_cache(rx as *const _, len) };
            #[cfg(target_os = "linux")]
            result?;
            #[cfg(not(target_os = "linux"))]
            result.map_err(io::Error::other)?;
        }
        Ok(())
    }

    pub(super) fn decommit(r: &Region, pages: Range<usize>) -> io::Result<()> {
        debug_assert!(!r.no_fd);
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            // Linux fallocate(2), FALLOC_FL_PUNCH_HOLE | KEEP_SIZE: remove
            // backing pages, invalidating mappings of that object in both views.
            // https://man7.org/linux/man-pages/man2/fallocate.2.html
            // SAFETY: the owned file backs the page-aligned, wholly free range.
            let result = unsafe {
                libc::fallocate(
                    r.backing.as_ref().unwrap().as_raw_fd(),
                    libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                    pages.start as libc::off_t,
                    pages.len() as libc::off_t,
                )
            };
            if result != 0 {
                return Err(io::Error::last_os_error());
            }
        }
        #[cfg(target_os = "freebsd")]
        {
            use std::os::fd::AsRawFd;
            // FreeBSD releng/15.0 sys/kern/uipc_shm.c, shm_deallocate:
            // removes full pages from the shared VM object and its mappings.
            // https://cgit.freebsd.org/src/tree/sys/kern/uipc_shm.c?h=releng/15.0
            let mut remaining = libc::spacectl_range {
                r_offset: pages.start as libc::off_t,
                r_len: pages.len() as libc::off_t,
            };
            while remaining.r_len != 0 {
                let request = remaining;
                // SAFETY: owned shm descriptor and initialized in/out ranges.
                let result = unsafe {
                    libc::fspacectl(
                        r.backing.as_ref().unwrap().as_raw_fd(),
                        libc::SPACECTL_DEALLOC,
                        &request,
                        0,
                        &mut remaining,
                    )
                };
                if result != 0 {
                    return Err(io::Error::last_os_error());
                }
                if remaining.r_len >= request.r_len {
                    return Err(io::Error::other("backing decommit made no progress"));
                }
            }
        }
        #[cfg(target_os = "macos")]
        {
            if r.rw != r.rx {
                return Err(io::ErrorKind::Unsupported.into());
            }
            // XNU vm_map.c, VM_BEHAVIOR_FREE -> VM_SYNC_KILLPAGES, and
            // vm_object_deactivate_pages: discard pages while keeping MAP_JIT.
            // Dirty backing is surrendered; clean pages may remain resident until
            // kernel reclamation. DONTNEED only deactivates dirty pages.
            // https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/vm/vm_map.c
            // SAFETY: page-aligned, wholly free pages in our retained mapping.
            let result = unsafe {
                libc::madvise((r.rw + pages.start) as *mut _, pages.len(), libc::MADV_FREE)
            };
            if result != 0 {
                return Err(io::Error::last_os_error());
            }
        }
        Ok(())
    }

    pub(super) unsafe fn release(r: &Region) {
        #[cfg(target_os = "macos")]
        if r.kind == Kind::Code && !r.no_fd {
            // SAFETY: this logical region has no remaining execution or readers.
            unsafe { libc::madvise(r.rw as *mut _, r.len, libc::MADV_FREE) };
            mac::release(r.rw, r.len);
            return;
        } // retain the process reservation; logical regions pass to other backends
          // SAFETY: inherited from free_memory; exact original views and lengths.
        unsafe {
            libc::munmap(r.rw as *mut _, r.len);
            if r.rx != r.rw {
                libc::munmap(r.rx as *mut _, r.len);
            }
        }
    }

    #[cfg(target_os = "macos")]
    mod mac {
        use super::*;
        use std::sync::{atomic::AtomicPtr, Mutex};
        // One MAP_JIT per process, also shared by the two MXCSR modules and all
        // backends. Logical 128 MiB regions grow inside this lazy 1 TiB extent.
        const RESERVATION: usize = 1 << 40;
        struct Arena {
            owner: ProcessIdentity,
            state: Mutex<(usize, usize, Vec<Range<usize>>)>,
        }
        static ARENA: AtomicPtr<Arena> = AtomicPtr::new(std::ptr::null_mut());

        fn arena() -> &'static Arena {
            let owner = ProcessIdentity::current();
            loop {
                let old = ARENA.load(Ordering::Acquire);
                // SAFETY: published arenas live for the process lifetime.
                if !old.is_null() && unsafe { (*old).owner == owner } {
                    // SAFETY: the arena is immutable outside its mutex and retained.
                    return unsafe { &*old };
                }
                // Check process identity before taking any mutex: a copied parent mutex
                // may be locked by a thread that does not exist in the child.
                let new = Box::into_raw(Box::new(Arena {
                    owner,
                    state: Mutex::new((0, 0, Vec::new())),
                }));
                if ARENA
                    .compare_exchange(old, new, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    // SAFETY: the winning arena is retained permanently.
                    return unsafe { &*new };
                }
                // SAFETY: this unpublished candidate is owned only by us.
                unsafe { drop(Box::from_raw(new)) };
            }
        }

        pub(super) fn release(start: usize, len: usize) {
            let mut state = arena().state.lock().unwrap();
            insert_range(&mut state.2, start..start + len);
        }

        #[cfg(test)]
        pub(super) fn fork_with_locked_arena() -> libc::pid_t {
            let guard = arena().state.lock().unwrap();
            // SAFETY: the child must replace the inherited allocator, even when
            // its copied mutex stays locked. The subprocess watchdog bounds it.
            let pid = unsafe { libc::fork() };
            if pid == 0 {
                std::mem::forget(guard);
            }
            pid
        }

        pub(super) fn reserve(
            len: usize,
            align: usize,
            counters: &CodeCounters,
        ) -> io::Result<(usize, usize)> {
            operation(counters, CodeMemoryFailureOp::WriteProtectSupport, || {
                if jit_write_protection_supported() {
                    Ok(())
                } else {
                    Err(io::ErrorKind::Unsupported.into())
                }
            })?;
            let mut arena = arena().state.lock().unwrap();
            if arena.0 == 0 {
                let ptr = operation(counters, CodeMemoryFailureOp::MapJit, || {
                    // SAFETY: anonymous reservation with per-thread MAP_JIT W^X.
                    let ptr = unsafe {
                        libc::mmap(
                            std::ptr::null_mut(),
                            RESERVATION,
                            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                            libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_JIT,
                            ANONYMOUS_FD,
                            0,
                        )
                    };
                    if ptr == libc::MAP_FAILED {
                        Err(io::Error::last_os_error())
                    } else {
                        Ok(ptr)
                    }
                })?;
                // XNU otherwise walks the entire private JIT object at fork,
                // including its sparse tail. Exclude the whole reservation once.
                if let Err(error) = exclude_from_fork(ptr, RESERVATION, counters) {
                    // SAFETY: no reservation address has escaped yet.
                    unsafe { libc::munmap(ptr, RESERVATION) };
                    return Err(error);
                }
                arena.0 = ptr as usize;
            }
            for index in 0..arena.2.len() {
                let hole = arena.2[index].clone();
                let start = (hole.start + align - 1) & !(align - 1);
                if start.checked_add(len).is_some_and(|end| end <= hole.end) {
                    subtract_range(&mut arena.2, start..start + len);
                    return Ok((start, start));
                }
            }
            let offset = (arena.0 + arena.1)
                .checked_add(align - 1)
                .ok_or(io::ErrorKind::OutOfMemory)?
                & !(align - 1);
            let offset = offset - arena.0;
            let end = offset
                .checked_add(len)
                .filter(|end| *end <= RESERVATION)
                .ok_or(io::ErrorKind::OutOfMemory)?;
            let ptr = arena.0 + offset;
            arena.1 = end;
            Ok((ptr, ptr))
        }

        pub(super) fn write(action: &mut dyn FnMut(), counters: &CodeCounters) -> io::Result<()> {
            operation(counters, CodeMemoryFailureOp::WriteProtectSupport, || {
                if jit_write_protection_supported() {
                    Ok(())
                } else {
                    Err(io::ErrorKind::Unsupported.into())
                }
            })?;
            struct RestoreExecute;
            impl Drop for RestoreExecute {
                fn drop(&mut self) {
                    // SAFETY: restore execution on this compiling thread,
                    // including when copying or relocation unwinds.
                    unsafe { libc::pthread_jit_write_protect_np(1) };
                }
            }
            // SAFETY: private compiler scopes never execute generated code or
            // nest write windows. Embedders must omit jit-write-allowlist, which
            // disables this API (Apple's JIT porting guide).
            operation(counters, CodeMemoryFailureOp::WriteProtectDisable, || {
                unsafe { libc::pthread_jit_write_protect_np(0) };
                Ok(())
            })?;
            let restore = RestoreExecute;
            action();
            // The guard restores execution even if the injected operation fails.
            operation(counters, CodeMemoryFailureOp::WriteProtectRestore, || {
                drop(restore);
                Ok(())
            })
        }

        pub(super) fn reserve_data(
            len: usize,
            kind: Kind,
            align: usize,
            counters: &CodeCounters,
        ) -> io::Result<(usize, usize)> {
            unsafe extern "C" {
                fn mach_vm_remap(
                    target: u32,
                    address: *mut u64,
                    size: u64,
                    mask: u64,
                    flags: i32,
                    source: u32,
                    source_address: u64,
                    copy: i32,
                    current: *mut i32,
                    maximum: *mut i32,
                    inheritance: u32,
                ) -> i32;
            }
            // SAFETY: create an anonymous shared data backing, then remap the
            // same object (copy=false) read-only. No MAP_JIT/data-kind mixing.
            #[allow(deprecated)]
            unsafe {
                let rw = map_aligned(
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED | libc::MAP_ANON,
                    ANONYMOUS_FD,
                    align,
                    counters,
                    CodeMemoryFailureOp::MapWritable,
                )?;
                if kind == Kind::Writable {
                    return Ok((rw as usize, rw as usize));
                }
                let mut rx = 0;
                let (mut current, mut maximum) = (0, 0);
                let task = libc::mach_task_self();
                let remap = operation(counters, CodeMemoryFailureOp::RemapReadOnly, || {
                    let result = mach_vm_remap(
                        task,
                        &mut rx,
                        len as u64,
                        (align - 1) as u64,
                        1,
                        task,
                        rw as u64,
                        0,
                        &mut current,
                        &mut maximum,
                        2,
                    ); // VM_FLAGS_ANYWHERE, VM_INHERIT_NONE
                    if result != 0 {
                        Err(io::Error::other(format!("mach_vm_remap: {result}")))
                    } else {
                        Ok(())
                    }
                });
                if let Err(error) = remap {
                    libc::munmap(rw, len);
                    return Err(error);
                }
                let protect = operation(counters, CodeMemoryFailureOp::ProtectReadOnly, || {
                    if libc::mprotect(rx as *mut _, len, libc::PROT_READ) != 0 {
                        Err(io::Error::last_os_error())
                    } else {
                        Ok(())
                    }
                });
                if let Err(error) = protect {
                    libc::munmap(rw, len);
                    libc::munmap(rx as *mut _, len);
                    return Err(error);
                }
                Ok((rw as usize, rx as usize))
            }
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "freebsd", target_os = "macos")))]
mod os {
    use super::*;
    pub(super) fn reserve_anonymous(_: usize, counters: &CodeCounters) -> io::Result<usize> {
        operation(counters, CodeMemoryFailureOp::ReserveAddressSpace, || {
            Err(io::ErrorKind::Unsupported.into())
        })
    }
    pub(super) fn commit_anonymous(
        _: usize,
        _: usize,
        _: bool,
        _: &CodeCounters,
    ) -> io::Result<()> {
        Err(io::ErrorKind::Unsupported.into())
    }
    pub(super) fn seal_anonymous(_: usize, _: usize, _: bool, _: &CodeCounters) -> io::Result<()> {
        Err(io::ErrorKind::Unsupported.into())
    }
    pub(super) fn reserve(
        _: usize,
        _: Kind,
        _: usize,
        counters: &CodeCounters,
    ) -> io::Result<(usize, usize, Option<std::os::fd::OwnedFd>)> {
        operation(counters, CodeMemoryFailureOp::ReserveAddressSpace, || {
            Err(io::ErrorKind::Unsupported.into())
        })
    }
    pub(super) fn with_write_access(action: &mut dyn FnMut(), _: &CodeCounters) -> io::Result<()> {
        action();
        Ok(())
    }
    pub(super) fn clear_cache(_: usize, _: usize, _: usize) -> io::Result<()> {
        Ok(())
    }
    pub(super) fn decommit(_: &Region, _: Range<usize>) -> io::Result<()> {
        Err(io::ErrorKind::Unsupported.into())
    }
    pub(super) unsafe fn release(_: &Region) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stress;
    #[cfg(target_os = "macos")]
    #[test]
    fn empty_map_jit_logical_regions_pass_between_provider_owners() {
        let mut first = CodeMemory::new(Arc::default());
        first.allocate(16, 16, JITMemoryKind::Executable).unwrap();
        let region = first.regions[0].rx;
        // SAFETY: this provider's allocation was never exposed to execution or readers.
        unsafe { first.free_memory() };
        let mut next = CodeMemory::new(Arc::default());
        next.allocate(16, 16, JITMemoryKind::Executable).unwrap();
        assert_eq!(next.regions[0].rx, region);
        // SAFETY: the replacement allocation also remains unpublished.
        unsafe { next.free_memory() };
    }

    use cranelift::codegen::binemit::Reloc;
    use cranelift_jit::JITModule;
    use cranelift_module::{Linkage, Module, ModuleReloc, ModuleRelocTarget};
    use std::sync::{
        atomic::{AtomicU8, AtomicUsize},
        Mutex,
    };

    fn inject(op: CodeMemoryFailureOp, errno: i32) {
        INJECT_FAILURE.with(|slot| slot.set(Some((op, errno))));
    }

    fn assert_failure(counters: &CodeCounters, op: CodeMemoryFailureOp, errno: i32) {
        assert!(
            INJECT_FAILURE.with(|slot| slot.get().is_none()),
            "hook not reached: {op:?}"
        );
        let failure = CodeMemoryFailure {
            operation: op,
            errno: (errno != 0).then_some(errno),
        };
        assert_eq!(counters.last_failure(), Some(failure));
        let stats = counters.snapshot();
        assert_eq!(stats.code_last_failure_op, op as u32);
        assert_eq!(stats.code_last_failure_errno, errno);
    }

    #[test]
    fn repeated_provider_step_failures_leave_no_allocations() {
        stress::isolated(
            "code_memory::tests::repeated_provider_step_failures_leave_no_allocations",
            |deadline| {
                #[cfg(target_os = "macos")]
                if !os::jit_write_protection_supported() {
                    eprintln!("SKIP dense failure witness: per-thread JIT write protection unsupported; system-provider fallback applies");
                    return;
                }
                let mut mapping_ops = vec![
                    CodeMemoryFailureOp::ReserveAddressSpace,
                    CodeMemoryFailureOp::MapWritable,
                    CodeMemoryFailureOp::ExcludeFromFork,
                ];
                #[cfg(target_os = "macos")]
                mapping_ops.extend([
                    CodeMemoryFailureOp::RemapReadOnly,
                    CodeMemoryFailureOp::ProtectReadOnly,
                ]);
                #[cfg(not(target_os = "macos"))]
                mapping_ops.extend([
                    CodeMemoryFailureOp::ObjectCreate,
                    CodeMemoryFailureOp::ObjectSize,
                    CodeMemoryFailureOp::MapExecutable,
                ]);
                let page = region::page::size();
                let mut unrelated = memmap2::MmapMut::map_anon(27 * page).unwrap();
                let unrelated_ptr = unrelated.as_mut_ptr();
                let _mapping_watch = crate::code_memory_probe::MappingWatch::new();
                let before = crate::code_memory_probe::resources();
                let mut warmed = None;
                for cycle in 0..13 {
                    // Split unrelated memory into more VM entries after every
                    // sample, reproducing lazy allocator/thread mapping noise.
                    // SAFETY: a page wholly inside the retained unrelated map;
                    // no references access it after its protection changes.
                    assert_eq!(
                        unsafe {
                            libc::mprotect(
                                unrelated_ptr.add(2 * cycle * page).cast(),
                                page,
                                libc::PROT_READ,
                            )
                        },
                        0
                    );
                    for op in mapping_ops.iter().copied() {
                        stress::remaining(deadline);
                        let counters = Arc::new(CodeCounters::default());
                        let mut memory = CodeMemory::new(counters.clone());
                        let errno = [libc::ENOMEM, libc::EMFILE, libc::EACCES][cycle % 3];
                        inject(op, errno);
                        assert!(memory
                            .allocate(
                                64,
                                (4 * region::page::size()) as u64,
                                JITMemoryKind::ReadOnly
                            )
                            .is_err());
                        assert_failure(&counters, op, errno);
                        assert!(memory.regions.is_empty());
                        assert!(memory.allocations.is_empty());
                        assert!(memory.allocation_by_rx.is_empty());
                        // A subsequent allocation must survive the failed attempt's
                        // cleanup and remain usable, including its final alias.
                        let rw = memory.allocate(64, 16, JITMemoryKind::ReadOnly).unwrap();
                        memory
                            .with_write_access(&mut || {
                                // SAFETY: writable extent owned by this raw provider.
                                unsafe {
                                    rw.write_bytes(cycle as u8, 64);
                                }
                            })
                            .unwrap();
                        memory.finalize(BranchProtection::None).unwrap();
                        if cycle == 0 {
                            let live = crate::code_memory_probe::resources();
                            assert!(
                                live.mappings >= 2,
                                "both provider aliases must be visible: {live:?}"
                            );
                        }
                        // SAFETY: finalized readable alias of the owned allocation.
                        assert_eq!(
                            unsafe { memory.executable_ptr(rw).read_volatile() },
                            cycle as u8
                        );
                        release(&mut memory, rw);
                        // SAFETY: no pointers escaped or code was registered.
                        unsafe {
                            memory.free_memory();
                        }
                        assert_eq!(counters.snapshot().code_bytes_used, 0);
                        assert_eq!(counters.snapshot().code_bytes_committed, 0);
                        assert_eq!(counters.snapshot().code_bytes_reserved, 0);
                    }
                    for op in [
                        CodeMemoryFailureOp::ReserveAddressSpace,
                        CodeMemoryFailureOp::ExcludeFromFork,
                        CodeMemoryFailureOp::CommitAnonymous,
                        CodeMemoryFailureOp::SealAnonymous,
                        CodeMemoryFailureOp::CacheMaintenance,
                        CodeMemoryFailureOp::PipelineSync,
                    ] {
                        stress::remaining(deadline);
                        let (counters, mut memory) =
                            anonymous_memory(CodeMemoryPolicy::WriteXorExecute);
                        inject(op, libc::ENOMEM);
                        match memory.allocate(64, 16, JITMemoryKind::Executable) {
                            Ok(rw) => {
                                assert!(memory.finalize(BranchProtection::None).is_err());
                                assert_failure(&counters, op, libc::ENOMEM);
                                release(&mut memory, rw);
                            }
                            Err(_) => assert_failure(&counters, op, libc::ENOMEM),
                        }
                        // SAFETY: failed raw allocations never escape to execution.
                        unsafe {
                            memory.free_memory();
                        }
                        assert!(memory.allocations.is_empty());
                        assert!(memory.allocation_by_rx.is_empty());
                        let stats = counters.snapshot();
                        assert_eq!(stats.code_allocation_failures, 1);
                        assert_eq!(
                            stats.code_bytes_used
                                + stats.code_bytes_committed
                                + stats.code_bytes_reserved,
                            0
                        );
                    }
                    if cycle == 0 {
                        warmed = Some(crate::code_memory_probe::resources());
                    }
                }
                let after = crate::code_memory_probe::resources();
                let warmed = warmed.unwrap();
                eprintln!(
                    "MEASURED OS resources before={before:?} warmed={warmed:?} after={after:?}"
                );
                assert_eq!(after.descriptors, warmed.descriptors, "descriptor leak");
                assert!(
                    after.mappings <= warmed.mappings,
                    "mapping leak: {warmed:?} -> {after:?}"
                );
                assert!(
                    after.mapped_bytes <= warmed.mapped_bytes,
                    "mapped-byte leak: {warmed:?} -> {after:?}"
                );
            },
        );
    }

    #[test]
    fn repeated_decommit_failures_reuse_and_eventually_discard_pages() {
        stress::isolated(
            "code_memory::tests::repeated_decommit_failures_reuse_and_eventually_discard_pages",
            |deadline| {
                #[cfg(target_os = "macos")]
                if !os::jit_write_protection_supported() {
                    eprintln!("SKIP dense decommit witness: per-thread JIT write protection unsupported; system-provider fallback applies");
                    return;
                }
                let counters = Arc::new(CodeCounters::default());
                let mut memory = CodeMemory::new(counters.clone());
                let page = region::page::size();
                let anchor = memory
                    .allocate(page, page as u64, JITMemoryKind::Executable)
                    .unwrap();
                for cycle in 0..48 {
                    stress::remaining(deadline);
                    let rw = memory
                        .allocate(page, page as u64, JITMemoryKind::Executable)
                        .unwrap();
                    memory
                        .with_write_access(&mut || {
                            // SAFETY: unpublished page within this provider's write window.
                            unsafe {
                                rw.write_bytes(cycle as u8, page);
                            }
                        })
                        .unwrap();
                    memory.finalize(BranchProtection::None).unwrap();
                    inject(CodeMemoryFailureOp::DecommitPages, libc::EIO);
                    release(&mut memory, rw);
                    assert_failure(&counters, CodeMemoryFailureOp::DecommitPages, libc::EIO);
                    assert_eq!(counters.snapshot().code_bytes_committed, (2 * page) as u64);
                    let reused = memory
                        .allocate(page, page as u64, JITMemoryKind::Executable)
                        .unwrap();
                    assert_eq!(reused, rw);
                    memory
                        .with_write_access(&mut || {
                            // SAFETY: fresh identity owns this reused page.
                            unsafe {
                                reused.write_bytes(0x5a, page);
                            }
                        })
                        .unwrap();
                    memory.finalize(BranchProtection::None).unwrap();
                    // SAFETY: finalized readable executable alias.
                    assert_eq!(
                        unsafe { memory.executable_ptr(reused).read_volatile() },
                        0x5a
                    );
                    release(&mut memory, reused);
                    assert_discarded(rw as usize, page);
                    assert_eq!(counters.snapshot().code_bytes_committed, page as u64);
                }
                release(&mut memory, anchor);
                assert_eq!(counters.snapshot().code_bytes_committed, 0);
                assert_eq!(counters.snapshot().code_allocation_failures, 48);
                // SAFETY: every never-published allocation has been returned.
                unsafe {
                    memory.free_memory();
                }
            },
        );
    }

    fn fixture(jit: crate::JitBackend) -> x86jit_core::Vm {
        use x86jit_core::{Prot, RegionKind, Vm, VmConfig};
        let mut vm = Vm::with_backend(VmConfig::flat(65536), Box::new(jit));
        vm.map(0, 65536, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, &[0xb8, 7, 0, 0, 0, 0xf4]).unwrap();
        vm
    }

    #[test]
    fn first_region_failure_falls_back_and_executes_both_modules() {
        use x86jit_core::{Exit, Reg};
        // Fresh processes also exercise mmap/minherit before the macOS arena
        // exists. Each child runs only this test, with at most two harness threads.
        const CHILD: &str = "X86JIT_TEST_PROVIDER_FAILURE";
        if let Ok(value) = std::env::var(CHILD) {
            let op = CodeMemoryFailure::decode(value.parse::<u64>().unwrap() << 32)
                .unwrap()
                .operation;
            let jit = crate::JitBackend::new();
            let shared = jit.shared.clone();
            let errno = if op == CodeMemoryFailureOp::WriteProtectSupport {
                0
            } else {
                libc::EPERM
            };
            inject(op, errno);
            drop(shared.jit(0));
            assert_failure(&shared.memory, op, errno);
            assert_eq!(
                jit.code_first_failure(),
                Some(CodeMemoryFailure {
                    operation: op,
                    errno: (errno != 0).then_some(errno)
                })
            );
            assert_eq!(jit.code_last_failure(), jit.code_first_failure());
            let vm = fixture(jit);
            let mut cpu = vm.new_vcpu();
            for mxcsr in [0x1f80, 0x3f80] {
                cpu.set_mxcsr(mxcsr);
                cpu.set_reg(Reg::Rip, 0x1000);
                assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
                assert_eq!(cpu.reg(Reg::Rax), 7);
            }
            let stats = vm.memory_stats().backend;
            assert_eq!(stats.compiled_units_total, 2);
            assert_eq!(stats.code_provider_fallback, 1);
            assert_eq!(stats.code_allocation_failures, 1);
            if cfg!(any(target_os = "linux", target_os = "freebsd")) {
                assert!(matches!(
                    *shared.memory.provider_choice.lock().unwrap(),
                    ProviderChoice::NoFd(CodeMemoryPolicy::WriteXorExecute)
                ));
                assert_eq!(stats.code_regions, 2);
                assert_eq!(stats.code_mapping_views, 2);
                assert_eq!(stats.code_bytes_reserved, 2 * REGION_BYTES as u64);
            } else {
                assert_eq!(stats.code_regions, 0);
                assert_eq!(stats.code_bytes_reserved, 0);
            }
            assert_eq!(stats.code_first_failure_op, op as u32);
            assert_eq!(stats.code_first_failure_errno, errno);
            return;
        }
        #[cfg(target_os = "macos")]
        let ops: &[CodeMemoryFailureOp] = if os::jit_write_protection_supported() {
            &[
                CodeMemoryFailureOp::WriteProtectSupport,
                CodeMemoryFailureOp::MapJit,
                CodeMemoryFailureOp::ExcludeFromFork,
            ]
        } else {
            &[CodeMemoryFailureOp::WriteProtectSupport]
        };
        #[cfg(any(target_os = "linux", target_os = "freebsd"))]
        let ops = &[
            CodeMemoryFailureOp::ObjectCreate,
            CodeMemoryFailureOp::ObjectSize,
            CodeMemoryFailureOp::MapWritable,
            CodeMemoryFailureOp::MapExecutable,
            CodeMemoryFailureOp::ExcludeFromFork,
        ];
        for &op in ops {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "code_memory::tests::first_region_failure_falls_back_and_executes_both_modules",
                    "--test-threads",
                    "2",
                ])
                .env(CHILD, (op as u32).to_string())
                .status()
                .unwrap();
            assert!(status.success(), "fallback failed for {op:?}");
        }
    }

    #[test]
    fn both_startup_failures_select_system_and_keep_both_diagnostics() {
        use x86jit_core::{Exit, Reg};
        TEST_ALIAS_PROVIDER.with(|slot| slot.set(true));
        for op in [
            CodeMemoryFailureOp::ReserveAddressSpace,
            CodeMemoryFailureOp::ExcludeFromFork,
            CodeMemoryFailureOp::CommitAnonymous,
            CodeMemoryFailureOp::SealAnonymous,
        ] {
            let jit = crate::JitBackend::new();
            let shared = jit.shared.clone();
            inject(CodeMemoryFailureOp::ObjectCreate, libc::EPERM);
            INJECT_NEXT_FAILURE.with(|slot| slot.set(Some((op, libc::ENOMEM))));
            drop(shared.jit(0));
            assert_eq!(
                *shared.memory.provider_choice.lock().unwrap(),
                ProviderChoice::System
            );
            assert_eq!(
                jit.code_first_failure().unwrap().operation,
                CodeMemoryFailureOp::ObjectCreate
            );
            assert_eq!(jit.code_first_failure().unwrap().errno, Some(libc::EPERM));
            assert_failure(&shared.memory, op, libc::ENOMEM);
            let vm = fixture(jit);
            let mut cpu = vm.new_vcpu();
            for mxcsr in [0x1f80, 0x3f80] {
                cpu.set_mxcsr(mxcsr);
                cpu.set_reg(Reg::Rip, 0x1000);
                assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
                assert_eq!(cpu.reg(Reg::Rax), 7);
            }
            let stats = vm.memory_stats().backend;
            assert_eq!(stats.code_provider_fallback, 2);
            assert_eq!(stats.code_allocation_failures, 2);
            assert_eq!(stats.code_regions, 0);
            assert_eq!(stats.code_bytes_reserved, 0);
            assert_eq!(stats.code_bytes_committed, 0);
            assert_eq!(stats.code_allocation_metadata_bytes, 0);
            assert_eq!(stats.compiled_units_total, 2);
        }
    }

    #[test]
    fn alias_creation_failure_selects_anonymous_and_executes_both_modules() {
        use x86jit_core::{Exit, Reg};
        TEST_ALIAS_PROVIDER.with(|slot| slot.set(true));
        let jit = crate::JitBackend::new();
        let shared = jit.shared.clone();
        inject(CodeMemoryFailureOp::ObjectCreate, libc::EPERM);
        drop(shared.jit(0));
        assert_failure(
            &shared.memory,
            CodeMemoryFailureOp::ObjectCreate,
            libc::EPERM,
        );
        assert_eq!(
            *shared.memory.provider_choice.lock().unwrap(),
            ProviderChoice::NoFd(CodeMemoryPolicy::WriteXorExecute)
        );
        let vm = fixture(jit);
        let mut cpu = vm.new_vcpu();
        for mxcsr in [0x1f80, 0x3f80] {
            cpu.set_mxcsr(mxcsr);
            cpu.set_reg(Reg::Rip, 0x1000);
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rax), 7);
        }
        let stats = vm.memory_stats().backend;
        assert_eq!(stats.compiled_units_total, 2);
        assert_eq!(stats.code_regions, 2);
        assert_eq!(stats.code_mapping_views, 2);
        assert_eq!(stats.code_provider_fallback, 1);
        assert_eq!(stats.code_allocation_failures, 1);
    }

    fn anonymous_memory(policy: CodeMemoryPolicy) -> (Arc<CodeCounters>, CodeMemory) {
        let counters = Arc::new(CodeCounters::default());
        *counters.provider_choice.lock().unwrap() = ProviderChoice::NoFd(policy);
        let mut memory = CodeMemory::new(counters.clone());
        memory.no_fd = Some(policy);
        (counters, memory)
    }

    #[test]
    fn read_implies_exec_refuses_strict_commit_and_automatic_probe() {
        use x86jit_core::{Exit, Reg};
        let (counters, mut memory) = anonymous_memory(CodeMemoryPolicy::WriteXorExecute);
        let first = memory.allocate(32, 16, JITMemoryKind::Executable).unwrap();
        memory.finalize(BranchProtection::None).unwrap();
        TEST_READ_IMPLIES_EXEC.with(|flag| flag.set(true));
        ANONYMOUS_PROTECTIONS.with(|log| log.borrow_mut().clear());
        assert!(memory.allocate(32, 16, JITMemoryKind::Executable).is_err());
        assert_failure(&counters, CodeMemoryFailureOp::AnonymousPersonality, 0);
        assert!(ANONYMOUS_PROTECTIONS.with(|log| log.borrow().is_empty()));
        // OS queries are test-only: Linux region::query reads /proc/self/maps
        // and opens a descriptor, which production no-fd storage must avoid.
        assert_eq!(
            region::query(first).unwrap().protection(),
            region::Protection::READ_EXECUTE
        );
        let frontier = (memory.regions[0].rw + memory.regions[0].committed) as *const u8;
        assert_eq!(
            region::query(frontier).unwrap().protection(),
            region::Protection::NONE
        );
        // SAFETY: no pointer escaped this allocator witness.
        unsafe { memory.free_memory() };

        let jit = crate::JitBackend::new();
        let shared = jit.shared.clone();
        // Exercise explicit strict selection on every host without changing the
        // public macOS MAP_JIT route.
        *shared.memory.provider_choice.lock().unwrap() =
            ProviderChoice::NoFd(CodeMemoryPolicy::WriteXorExecute);
        let vm = fixture(jit);
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(
            cpu.run(&vm, None),
            Exit::CodeSpaceExhausted { .. }
        ));
        assert_failure(&shared.memory, CodeMemoryFailureOp::AnonymousPersonality, 0);
        assert_eq!(vm.memory_stats().backend.code_bytes_committed, 0);
        assert_eq!(vm.memory_stats().backend.code_provider_fallback, 0);

        TEST_ALIAS_PROVIDER.with(|flag| flag.set(true));
        let jit = crate::JitBackend::new();
        let shared = jit.shared.clone();
        inject(CodeMemoryFailureOp::ObjectCreate, libc::EPERM);
        drop(shared.jit(0));
        assert_eq!(
            *shared.memory.provider_choice.lock().unwrap(),
            ProviderChoice::System
        );
        assert_failure(&shared.memory, CodeMemoryFailureOp::AnonymousPersonality, 0);
        assert_eq!(shared.memory.snapshot().code_provider_fallback, 2);
        assert_eq!(shared.memory.snapshot().code_allocation_failures, 2);
        TEST_ALIAS_PROVIDER.with(|flag| flag.set(false));
        TEST_READ_IMPLIES_EXEC.with(|flag| flag.set(false));
    }

    #[test]
    fn anonymous_personality_query_failure_prevents_commit() {
        let (counters, mut memory) = anonymous_memory(CodeMemoryPolicy::WriteXorExecute);
        inject(CodeMemoryFailureOp::AnonymousPersonality, libc::EPERM);
        assert!(memory.allocate(32, 16, JITMemoryKind::Executable).is_err());
        assert_failure(
            &counters,
            CodeMemoryFailureOp::AnonymousPersonality,
            libc::EPERM,
        );
        assert_eq!(counters.snapshot().code_bytes_committed, 0);
        // SAFETY: failed storage was never published.
        unsafe { memory.free_memory() };
    }

    #[test]
    fn system_provider_failures_are_counted_and_sticky_after_fallback_or_selection() {
        use x86jit_core::{Exit, Reg};
        for automatic in [false, true] {
            for op in [
                CodeMemoryFailureOp::SystemAllocate,
                CodeMemoryFailureOp::SystemProtectExecutable,
                CodeMemoryFailureOp::CacheMaintenance,
                CodeMemoryFailureOp::PipelineSync,
            ] {
                let jit = crate::JitBackend::new();
                let jit = if automatic {
                    jit
                } else {
                    jit.with_system_memory_provider()
                };
                let shared = jit.shared.clone();
                if automatic {
                    TEST_ALIAS_PROVIDER.with(|flag| flag.set(true));
                    inject(CodeMemoryFailureOp::ObjectCreate, libc::EPERM);
                    INJECT_NEXT_FAILURE.with(|slot| {
                        slot.set(Some((
                            CodeMemoryFailureOp::ReserveAddressSpace,
                            libc::ENOMEM,
                        )))
                    });
                    drop(shared.jit(0));
                    TEST_ALIAS_PROVIDER.with(|flag| flag.set(false));
                }
                let vm = fixture(jit);
                inject(op, libc::EACCES);
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, 0x1000);
                assert!(
                    matches!(cpu.run(&vm, None), Exit::CodeSpaceExhausted { .. }),
                    "{op:?}"
                );
                assert_failure(&shared.memory, op, libc::EACCES);
                let stats = vm.memory_stats().backend;
                assert_eq!(stats.compiled_units_total, 0);
                assert_eq!(
                    stats.code_allocation_failures,
                    if automatic { 3 } else { 1 }
                );
                assert_eq!(stats.code_regions, 0);
                assert_eq!(stats.code_bytes_committed, 0);
                assert!(shared.code_space_exhausted.load(Ordering::Acquire));
                for mxcsr in [0x1f80, 0x3f80] {
                    let mut cpu = vm.new_vcpu();
                    cpu.set_mxcsr(mxcsr);
                    cpu.set_reg(Reg::Rip, 0x1000);
                    assert!(matches!(
                        cpu.run(&vm, None),
                        Exit::CodeSpaceExhausted { .. }
                    ));
                }
                assert_eq!(
                    vm.memory_stats().backend.code_allocation_failures,
                    stats.code_allocation_failures
                );
            }
        }
    }

    #[test]
    fn system_readonly_failure_returns_allocation_with_original_errno() {
        let counters = Arc::new(CodeCounters::default());
        let mut memory = SystemMemory::new(counters.clone());
        memory.allocate(32, 16, JITMemoryKind::ReadOnly).unwrap();
        inject(CodeMemoryFailureOp::SystemProtectReadOnly, libc::EPERM);
        match memory.finalize(BranchProtection::None) {
            Err(ModuleError::Allocation { err, .. }) => {
                assert_eq!(err.raw_os_error(), Some(libc::EPERM))
            }
            result => panic!("unexpected finalization result: {result:?}"),
        }
        assert_failure(
            &counters,
            CodeMemoryFailureOp::SystemProtectReadOnly,
            libc::EPERM,
        );
        assert_eq!(counters.snapshot().code_allocation_failures, 1);
        // SAFETY: no storage was published.
        unsafe { memory.free_memory() };
    }

    #[test]
    fn anonymous_wx_frontier_never_reopens_executable_pages() {
        let (counters, mut memory) = anonymous_memory(CodeMemoryPolicy::WriteXorExecute);
        ANONYMOUS_PROTECTIONS.with(|log| log.borrow_mut().clear());
        let page = region::page::size();
        let mut pages = Vec::new();
        for _ in 0..8 {
            let first = memory.allocate(32, 16, JITMemoryKind::Executable).unwrap();
            let second = memory.allocate(32, 16, JITMemoryKind::Executable).unwrap();
            assert_eq!(second as usize - first as usize, 32);
            assert_eq!(
                region::query(first).unwrap().protection(),
                region::Protection::READ_WRITE
            );
            memory
                .with_write_access(&mut || {
                    // SAFETY: both allocations remain in the current RW frontier.
                    unsafe { first.write_bytes(0x55, 64) };
                })
                .unwrap();
            memory.finalize(BranchProtection::None).unwrap();
            pages.push(first);
            for &old in &pages {
                assert_eq!(
                    region::query(old).unwrap().protection(),
                    region::Protection::READ_EXECUTE
                );
                // SAFETY: sealed code remains readable after every later batch.
                assert_eq!(unsafe { old.read() }, 0x55);
            }
            let r = &memory.regions[0];
            assert_eq!(r.used, pages.len() * page);
            assert_eq!(r.sealed, r.used);
            assert_eq!(
                region::query((r.rw + r.used) as *const u8)
                    .unwrap()
                    .protection(),
                region::Protection::NONE
            );
        }
        ANONYMOUS_PROTECTIONS.with(|log| {
            let mut executable = std::collections::BTreeSet::new();
            for &(start, len, prot) in log.borrow().iter() {
                assert!(prot & libc::PROT_WRITE == 0 || prot & libc::PROT_EXEC == 0);
                for address in (start..start + len).step_by(page) {
                    if prot & libc::PROT_WRITE != 0 {
                        assert!(!executable.contains(&address), "RX page made writable");
                    }
                    if prot & libc::PROT_EXEC != 0 {
                        executable.insert(address);
                    }
                }
            }
            assert_eq!(executable.len(), 8);
        });
        let stats = counters.snapshot();
        assert_eq!(stats.code_regions, 1);
        assert_eq!(stats.code_mapping_views, 1);
        assert_eq!(stats.code_process_reservation_bytes, 0);
        assert_eq!(stats.code_bytes_used, (8 * page) as u64);
        assert_eq!(stats.code_bytes_committed, stats.code_bytes_used);
        assert_eq!(
            stats.code_bytes_used + stats.code_bytes_free,
            REGION_BYTES as u64
        );
        // SAFETY: no addresses from this raw allocator witness were published.
        unsafe { memory.free_memory() };
    }

    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    #[test]
    fn anonymous_rwx_keeps_partial_pages_and_data_protections() {
        let (counters, mut memory) = anonymous_memory(CodeMemoryPolicy::ReadWriteExecute);
        let first = memory.allocate(32, 16, JITMemoryKind::Executable).unwrap();
        memory.finalize(BranchProtection::None).unwrap();
        let second = memory.allocate(32, 16, JITMemoryKind::Executable).unwrap();
        assert_eq!(second as usize - first as usize, 32);
        assert_eq!(
            region::query(first).unwrap().protection(),
            region::Protection::READ_WRITE_EXECUTE
        );
        let ro = memory.allocate(8, 8, JITMemoryKind::ReadOnly).unwrap();
        let rw = memory.allocate(8, 8, JITMemoryKind::Writable).unwrap();
        memory.finalize(BranchProtection::None).unwrap();
        assert_eq!(
            region::query(ro).unwrap().protection(),
            region::Protection::READ
        );
        assert_eq!(
            region::query(rw).unwrap().protection(),
            region::Protection::READ_WRITE
        );
        assert_eq!(counters.snapshot().code_bytes_used, 64);
        // SAFETY: this witness never publishes any entry.
        unsafe { memory.free_memory() };
    }

    #[test]
    fn anonymous_alignment_growth_and_failure_paths() {
        let page = region::page::size();
        let (counters, mut memory) = anonymous_memory(CodeMemoryPolicy::WriteXorExecute);
        for op in [
            CodeMemoryFailureOp::ReserveAddressSpace,
            CodeMemoryFailureOp::ExcludeFromFork,
            CodeMemoryFailureOp::CommitAnonymous,
        ] {
            inject(op, libc::EPERM);
            assert!(memory
                .allocate(16, (page * 4) as u64, JITMemoryKind::Executable)
                .is_err());
            assert_failure(&counters, op, libc::EPERM);
        }
        let first = memory
            .allocate(16, (page * 4) as u64, JITMemoryKind::Executable)
            .unwrap();
        assert_eq!(first as usize % (page * 4), 0);
        inject(CodeMemoryFailureOp::SealAnonymous, libc::EACCES);
        assert!(memory.finalize(BranchProtection::None).is_err());
        assert_failure(&counters, CodeMemoryFailureOp::SealAnonymous, libc::EACCES);
        memory.finalize(BranchProtection::None).unwrap();
        // Consume virtual capacity without populating it, then require growth.
        memory
            .allocate(REGION_BYTES, 16, JITMemoryKind::Executable)
            .unwrap();
        assert_eq!(memory.regions.len(), 2);
        assert_eq!(counters.snapshot().code_mapping_views, 2);
        assert_eq!(
            region::query(first).unwrap().protection(),
            region::Protection::READ_EXECUTE
        );
        // SAFETY: no code from this witness was published or executed.
        unsafe { memory.free_memory() };
    }

    #[test]
    fn explicit_anonymous_policies_execute_both_modules() {
        use x86jit_core::{Exit, Reg};
        for policy in [CodeMemoryPolicy::WriteXorExecute]
            .into_iter()
            .chain((!cfg!(target_os = "macos")).then_some(CodeMemoryPolicy::ReadWriteExecute))
        {
            let jit = crate::JitBackend::new().with_no_fd_memory_provider(policy);
            let vm = fixture(jit);
            let mut cpu = vm.new_vcpu();
            for mxcsr in [0x1f80, 0x3f80] {
                cpu.set_mxcsr(mxcsr);
                cpu.set_reg(Reg::Rip, 0x1000);
                assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
                assert_eq!(cpu.reg(Reg::Rax), 7);
            }
            let stats = vm.memory_stats().backend;
            assert_eq!(stats.compiled_units_total, 2);
            assert_eq!(stats.code_regions, 2);
            assert_eq!(
                stats.code_mapping_views,
                if cfg!(target_os = "macos") { 1 } else { 2 }
            );
            assert_eq!(stats.code_provider_fallback, 0);
        }
    }

    #[test]
    fn anonymous_failures_never_publish_or_switch_provider() {
        use x86jit_core::{Exit, Reg};
        for op in [
            CodeMemoryFailureOp::ReserveAddressSpace,
            CodeMemoryFailureOp::ExcludeFromFork,
            CodeMemoryFailureOp::CommitAnonymous,
            CodeMemoryFailureOp::SealAnonymous,
            CodeMemoryFailureOp::CacheMaintenance,
            CodeMemoryFailureOp::PipelineSync,
        ] {
            let jit = crate::JitBackend::new();
            // Exercise the anonymous allocator directly on macOS too, while
            // the public option preserves that platform's MAP_JIT path.
            *jit.shared.memory.provider_choice.lock().unwrap() =
                ProviderChoice::NoFd(CodeMemoryPolicy::WriteXorExecute);
            let shared = jit.shared.clone();
            let vm = fixture(jit);
            inject(op, libc::EPERM);
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, 0x1000);
            assert!(
                matches!(cpu.run(&vm, None), Exit::CodeSpaceExhausted { .. }),
                "{op:?}"
            );
            assert_failure(&shared.memory, op, libc::EPERM);
            assert!(vm.backend.code_space_exhausted());
            let stats = vm.memory_stats().backend;
            assert_eq!(stats.code_provider_fallback, 0);
            assert_eq!(stats.compiled_units_total, 0);
            assert_eq!(stats.code_allocation_failures, 1);
            assert_eq!(
                *shared.memory.provider_choice.lock().unwrap(),
                ProviderChoice::NoFd(CodeMemoryPolicy::WriteXorExecute)
            );
        }
    }

    #[test]
    fn background_compilation_proceeds_after_provider_fallback() {
        use x86jit_core::{Exit, Reg};
        let jit = crate::JitBackend::new();
        let handle = jit.tier_up_handle();
        let shared = jit.shared.clone();
        #[cfg(target_os = "macos")]
        let op = CodeMemoryFailureOp::WriteProtectSupport;
        #[cfg(not(target_os = "macos"))]
        let op = CodeMemoryFailureOp::ObjectCreate;
        inject(op, libc::EPERM);
        drop(shared.jit(0));
        assert_failure(&shared.memory, op, libc::EPERM);
        let mut vm = fixture(jit);
        vm.set_tier_up_after(Some(1));
        vm.set_tier_up_background(true);
        let mut cpu = vm.new_vcpu();
        for _ in 0..3 {
            cpu.set_reg(Reg::Rip, 0x1000);
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        }
        handle.wait_idle();
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rax), 7);
        let stats = vm.memory_stats();
        assert_eq!(stats.backend.code_provider_fallback, 1);
        assert_eq!(stats.backend.compiled_units_total, 1);
        assert_eq!(stats.cache.tier_bg_published, 1);
        assert_eq!(stats.cache.tier_bg_failed_code_space, 0);
    }

    #[test]
    fn failure_after_first_region_exhausts_without_switching() {
        use x86jit_core::{Exit, Reg};
        #[cfg(target_os = "macos")]
        if !os::jit_write_protection_supported() {
            return;
        }
        let jit = crate::JitBackend::new();
        let shared = jit.shared.clone();
        let vm = fixture(jit);
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(vm.memory_stats().backend.code_regions, 1);
        #[cfg(target_os = "macos")]
        let op = CodeMemoryFailureOp::WriteProtectSupport;
        #[cfg(not(target_os = "macos"))]
        let op = CodeMemoryFailureOp::ObjectCreate;
        inject(op, libc::ENOMEM);
        // The other module has no region yet, but this backend already has one.
        cpu.set_mxcsr(0x3f80);
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(
            cpu.run(&vm, None),
            Exit::CodeSpaceExhausted { .. }
        ));
        assert_failure(&shared.memory, op, libc::ENOMEM);
        let stats = vm.memory_stats().backend;
        assert_eq!(stats.code_provider_fallback, 0);
        assert_eq!(stats.code_regions, 1);
        assert_eq!(stats.compiled_units_total, 1);
        assert!(vm.backend.code_space_exhausted());
        // A vCPU reports exhaustion once, then resumes in the interpreter.
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(vm.memory_stats().backend.compiled_units_total, 1);
        assert_eq!(vm.memory_stats().backend.code_allocation_failures, 1);
    }

    #[test]
    fn publication_failures_record_errno_and_exhaust() {
        use x86jit_core::{Exit, Reg};
        #[cfg(target_os = "macos")]
        if !os::jit_write_protection_supported() {
            return;
        }
        let mut ops = vec![
            CodeMemoryFailureOp::CacheMaintenance,
            CodeMemoryFailureOp::PipelineSync,
        ];
        #[cfg(target_os = "macos")]
        ops.extend([
            CodeMemoryFailureOp::WriteProtectDisable,
            CodeMemoryFailureOp::WriteProtectRestore,
        ]);
        for op in ops.drain(..) {
            let jit = crate::JitBackend::new();
            let shared = jit.shared.clone();
            let vm = fixture(jit);
            inject(op, libc::EACCES);
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, 0x1000);
            assert!(
                matches!(cpu.run(&vm, None), Exit::CodeSpaceExhausted { .. }),
                "{op:?}"
            );
            assert_failure(&shared.memory, op, libc::EACCES);
            let stats = vm.memory_stats().backend;
            assert_eq!(stats.code_provider_fallback, 0);
            assert_eq!(stats.compiled_units_total, 0);
            assert_eq!(stats.code_allocation_failures, 1);
        }
    }

    #[test]
    fn aligned_mapping_failures_keep_first_and_last_errno() {
        let counters = Arc::new(CodeCounters::default());
        let mut memory = CodeMemory::new(counters.clone());
        let align = (region::page::size() * 4) as u64;
        let mut ops = vec![
            CodeMemoryFailureOp::ReserveAddressSpace,
            CodeMemoryFailureOp::MapWritable,
            CodeMemoryFailureOp::ExcludeFromFork,
        ];
        #[cfg(target_os = "macos")]
        ops.extend([
            CodeMemoryFailureOp::RemapReadOnly,
            CodeMemoryFailureOp::ProtectReadOnly,
        ]);
        #[cfg(not(target_os = "macos"))]
        ops.push(CodeMemoryFailureOp::MapExecutable);
        for (i, op) in ops.into_iter().enumerate() {
            let errno = if i == 0 { libc::ENOMEM } else { libc::EPERM };
            inject(op, errno);
            assert_eq!(
                memory
                    .allocate(16, align, JITMemoryKind::ReadOnly)
                    .unwrap_err()
                    .raw_os_error(),
                Some(errno)
            );
            assert_failure(&counters, op, errno);
            assert_eq!(
                counters.first_failure(),
                Some(CodeMemoryFailure {
                    operation: CodeMemoryFailureOp::ReserveAddressSpace,
                    errno: Some(libc::ENOMEM)
                })
            );
            assert_eq!(counters.snapshot().code_first_failure_errno, libc::ENOMEM);
            assert!(memory.regions.is_empty());
        }
        // Explicit selection does not probe, consume a hook or report a fallback.
        let jit = crate::JitBackend::new().with_system_memory_provider();
        drop(jit.shared.jit(0));
        assert_eq!(jit.shared.memory.snapshot().code_provider_fallback, 0);
        assert_eq!(jit.code_first_failure(), None);
    }

    #[test]
    fn shared_child_marker_refuses_jit_and_preserves_code() {
        const TEST: &str = "code_memory::tests::shared_child_marker_refuses_jit_and_preserves_code";
        const WORKER: &str = "X86JIT_SHARED_CHILD_MARKER_WORKER";
        if std::env::var_os(WORKER).is_none() {
            let mut child = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", TEST, "--test-threads", "2", "--nocapture"])
                .env(WORKER, "1")
                .spawn()
                .unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
            loop {
                if let Some(status) = child.try_wait().unwrap() {
                    assert!(status.success(), "shared-child marker worker: {status}");
                    return;
                }
                if std::time::Instant::now() >= deadline {
                    child.kill().unwrap();
                    child.wait().unwrap();
                    panic!("shared-child marker worker exceeded 20 seconds");
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
        use crate::{CompileError, HostTarget, JitBackend, OptLevel};
        use x86jit_core::{Backend, MemConsistency, RegionCaps, SharedAddressSpaceChild};
        let caps = RegionCaps {
            max_blocks: 8,
            max_icount: 128,
        };
        let jit = std::mem::ManuallyDrop::new(JitBackend::try_new().unwrap());
        let mut mem = x86jit_core::Memory::new(x86jit_core::MemoryModel::Flat { size: 0x4000 });
        mem.map(
            0x1000,
            0x1000,
            x86jit_core::Prot::RWX,
            x86jit_core::RegionKind::Ram,
        )
        .unwrap();
        mem.write_bytes(0x1000, &[0xb8, 42, 0, 0, 0, 0xf4]).unwrap();
        let ir = x86jit_core::lift::lift_block(
            &mem,
            x86jit_core::lift::FetchAddr::flat(0x1000),
            x86jit_core::CpuMode::Long64,
        )
        .unwrap();
        let region = x86jit_core::IrRegion {
            entry: ir.guest_start,
            blocks: vec![ir.clone()],
            has_loop: false,
        };
        let x86jit_core::CachedBlock::Compiled { entry } = jit
            .try_materialize(&ir, MemConsistency::Fast, None, 0)
            .unwrap()
        else {
            panic!("checked compilation must produce code")
        };
        assert!(matches!(
            jit.try_materialize_region(&region, MemConsistency::Fast, None, 0)
                .unwrap(),
            x86jit_core::CachedBlock::Compiled { .. }
        ));
        assert!(x86jit_core::codemap::contains(entry.0 as usize));
        let mut memory = CodeMemory::new(Arc::default());
        let address = memory.allocate(16, 16, JITMemoryKind::Executable).unwrap();
        memory
            .with_write_access(&mut || {
                // SAFETY: this provider owns the writable allocation and no code executes it.
                unsafe { address.write(0x7f) };
            })
            .unwrap();
        memory.finalize(BranchProtection::None).unwrap();
        let regions = memory.regions.len();
        let used = memory.regions[0].used;
        let allocations = memory.allocations.len();
        let request = || x86jit_core::TierUpRequest {
            precise_split_stores: false,
            mxcsr_mode: 0,
            pc: ir.guest_start,
            spans: vec![(ir.guest_start, ir.guest_len)],
            epoch: 0,
            unit: x86jit_core::TierUpUnit::Block(Arc::new(ir.clone())),
            consistency: MemConsistency::Fast,
            mmio: None,
            guest_base: 0,
        };
        let failures = Arc::new(Mutex::new(Vec::new()));
        let observed = failures.clone();
        jit.set_compile_failure_observer(move |event| observed.lock().unwrap().push(event));
        let _serial = jit.shared.compile_serial.lock().unwrap();
        let _queue = jit.shared.background.queue.lock().unwrap();
        x86jit_core::mark_shared_address_space_child().unwrap();
        assert!(matches!(
            JitBackend::try_new(),
            Err(SharedAddressSpaceChild)
        ));
        assert!(matches!(
            jit.shared
                .compile(&ir, MemConsistency::Fast, None, 0, 0, false),
            Err(CompileError::SharedAddressSpaceChild)
        ));
        assert!(matches!(
            jit.shared
                .compile_region(&region, MemConsistency::Fast, None, 0, 0, false),
            Err(CompileError::SharedAddressSpaceChild)
        ));
        assert!(matches!(
            jit.shared.define_request(&request(), None),
            Err(CompileError::SharedAddressSpaceChild)
        ));
        assert!(matches!(
            jit.try_materialize(&ir, MemConsistency::Fast, None, 0),
            Err(x86jit_core::CompileFailureReason::OtherBackendError)
        ));
        assert!(matches!(
            jit.try_materialize_region(&region, MemConsistency::Fast, None, 0),
            Err(x86jit_core::CompileFailureReason::OtherBackendError)
        ));
        assert_eq!(
            jit.tier_up_async(request()),
            x86jit_core::TierUpSubmit::Unsupported
        );
        assert!(matches!(
            jit.materialize(&ir, MemConsistency::Fast, None, 0),
            x86jit_core::CachedBlock::Interpreted(_)
        ));
        assert_eq!(
            CompileError::SharedAddressSpaceChild.reason(),
            x86jit_core::CompileFailureReason::OtherBackendError
        );
        assert_eq!(failures.lock().unwrap().len(), 6);
        assert!(failures.lock().unwrap().iter().all(|event| {
            event.reason == x86jit_core::CompileFailureReason::OtherBackendError
                && event.guest_rip == ir.guest_start
                && event.blocks == 1
        }));
        for backend in [
            JitBackend::new(),
            JitBackend::default(),
            JitBackend::with_superblocks(caps),
            JitBackend::with_opt_level(OptLevel::Speed),
            JitBackend::with_host_target(HostTarget::Native),
            JitBackend::with_options(Some(caps), HostTarget::Native, OptLevel::Speed),
            JitBackend::new().with_system_memory_provider(),
        ] {
            let failures = Arc::new(Mutex::new(Vec::new()));
            let observed = failures.clone();
            backend.set_compile_failure_observer(move |event| {
                observed.lock().unwrap().push(event);
            });
            let publications = Arc::new(AtomicUsize::new(0));
            let observed = publications.clone();
            backend.set_compiled_code_observer(move |_| {
                observed.fetch_add(1, Ordering::Relaxed);
            });
            assert!(matches!(
                backend.try_materialize(&ir, MemConsistency::Fast, None, 0),
                Err(x86jit_core::CompileFailureReason::OtherBackendError)
            ));
            assert!(matches!(
                backend.try_materialize_region(&region, MemConsistency::Fast, None, 0),
                Err(x86jit_core::CompileFailureReason::OtherBackendError)
            ));
            for block in [
                backend.materialize(&ir, MemConsistency::Fast, None, 0),
                backend.materialize_with_mxcsr(&ir, MemConsistency::Fast, None, 0, 1),
                backend.materialize_with_split_stores(&ir, MemConsistency::Fast, None, 0, 1, true),
                backend.materialize_region(&region, MemConsistency::Fast, None, 0),
                backend.materialize_region_with_mxcsr(&region, MemConsistency::Fast, None, 0, 1),
                backend.materialize_region_with_split_stores(
                    &region,
                    MemConsistency::Fast,
                    None,
                    0,
                    1,
                    true,
                ),
            ] {
                assert!(matches!(block, x86jit_core::CachedBlock::Interpreted(_)));
            }
            assert_eq!(
                backend.tier_up_async(request()),
                x86jit_core::TierUpSubmit::Unsupported
            );
            let mut region_request = request();
            region_request.unit = x86jit_core::TierUpUnit::Region(Arc::new(region.clone()));
            assert_eq!(
                backend.tier_up_async(region_request),
                x86jit_core::TierUpSubmit::Unsupported
            );
            let failures = failures.lock().unwrap();
            assert_eq!(failures.len(), 10);
            assert!(failures.iter().all(|event| {
                event.reason == x86jit_core::CompileFailureReason::OtherBackendError
            }));
            assert_eq!(backend.shared.compile_attempts.load(Ordering::Relaxed), 0);
            assert!(backend.shared.inner.lock().unwrap().is_none());
            assert!(backend.shared.nondefault_inner.lock().unwrap().is_none());
            assert!(backend.worker.lock().unwrap().is_none());
            assert_eq!(backend.memory_stats().code_bytes_reserved, 0);
            assert_eq!(publications.load(Ordering::Relaxed), 0);
        }
        let mut vm = x86jit_core::Vm::with_backend(
            x86jit_core::VmConfig::flat(0x4000),
            Box::new(JitBackend::default()),
        );
        vm.map(
            0x1000,
            0x1000,
            x86jit_core::Prot::RWX,
            x86jit_core::RegionKind::Ram,
        )
        .unwrap();
        vm.write_bytes(0x1000, &[0xb8, 42, 0, 0, 0, 0xf4]).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(x86jit_core::Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&vm, Some(10)), x86jit_core::Exit::Hlt));
        assert_eq!(cpu.reg(x86jit_core::Reg::Rax), 42);
        assert_eq!(vm.cache.memory_stats().compiled_entries, 0);
        for kind in [
            JITMemoryKind::Executable,
            JITMemoryKind::ReadOnly,
            JITMemoryKind::Writable,
        ] {
            let error = memory.allocate(16, 16, kind).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
            assert!(error.get_ref().unwrap().is::<SharedAddressSpaceChild>());
        }
        assert!(memory
            .with_write_access(&mut || panic!("child must not enter write access"))
            .is_err());
        assert!(memory.finalize(BranchProtection::None).is_err());
        // SAFETY: the provider must refuse before releasing any live mapping.
        unsafe { memory.free_memory() };
        assert_eq!(memory.regions.len(), regions);
        assert_eq!(memory.regions[0].used, used);
        assert_eq!(memory.allocations.len(), allocations);
        assert_eq!(memory.counters.failures.load(Ordering::Relaxed), 0);
        assert!(!jit.shared.code_space_exhausted.load(Ordering::Relaxed));
        // SAFETY: refusal left the existing mapping readable and unchanged.
        assert_eq!(unsafe { address.read() }, 0x7f);
        assert!(x86jit_core::codemap::contains(entry.0 as usize));
        assert_eq!(
            x86jit_core::codemap::register(0x1000, 8, Box::new([]), Box::new([])),
            Err(x86jit_core::codemap::CodeMapFull)
        );
        assert!(!x86jit_core::codemap::unregister(entry.0 as usize, 1));
        assert!(x86jit_core::codemap::contains(entry.0 as usize));
    }

    #[test]
    fn linux_sync_core_registration_selects_receiver_on_failure() {
        use linux_sync::Protocol;
        let mut commands = Vec::new();
        let protocol = Protocol::register(|command| {
            commands.push(command);
            Ok(())
        });
        assert_eq!(protocol, Protocol::SyncCore);
        protocol
            .flush(|command| {
                commands.push(command);
                Ok(())
            })
            .unwrap();
        assert_eq!(commands, [64, 32]);
        assert!(protocol
            .flush(|_| Err(io::ErrorKind::PermissionDenied.into()))
            .is_err());

        for errno in [libc::EINVAL, libc::ENOSYS, libc::EPERM] {
            let protocol = Protocol::register(|command| {
                assert_eq!(command, 64);
                Err(io::Error::from_raw_os_error(errno))
            });
            assert_eq!(protocol, Protocol::Receiver);
            protocol
                .flush(|_| panic!("receiver protocol must not issue GLOBAL"))
                .unwrap();
        }
    }

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "freebsd"))]
    #[test]
    fn fork_excludes_arena_and_rebuilds_in_child_and_grandchild() {
        #[cfg(target_os = "macos")]
        if !os::jit_write_protection_supported() {
            return;
        }
        const TEST: &str =
            "code_memory::tests::fork_excludes_arena_and_rebuilds_in_child_and_grandchild";
        const WORKER: &str = "X86JIT_CODE_MEMORY_FORK_WORKER";
        if std::env::var_os(WORKER).is_none() {
            use std::os::unix::process::CommandExt;
            for mode in ["dense", "wx"]
                .into_iter()
                .chain((!cfg!(target_os = "macos")).then_some("rwx"))
            {
                let mut child = std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", TEST, "--test-threads", "2", "--nocapture"])
                    .env(WORKER, mode)
                    .process_group(0)
                    .spawn()
                    .unwrap();
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
                loop {
                    if let Some(status) = child.try_wait().unwrap() {
                        assert!(status.success(), "fork worker: {status}");
                        break;
                    }
                    if std::time::Instant::now() >= deadline {
                        // SAFETY: the worker owns this group, including all descendants.
                        unsafe { libc::kill(-(child.id() as libc::pid_t), libc::SIGKILL) };
                        child.wait().unwrap();
                        panic!("code-memory fork worker exceeded 20 seconds");
                    }
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            }

            return;
        }

        fn fork_backend() -> crate::JitBackend {
            let jit = crate::JitBackend::new();
            match std::env::var("X86JIT_CODE_MEMORY_FORK_WORKER")
                .unwrap()
                .as_str()
            {
                "wx" => jit.with_no_fd_memory_provider(CodeMemoryPolicy::WriteXorExecute),
                "rwx" => jit.with_no_fd_memory_provider(CodeMemoryPolicy::ReadWriteExecute),
                _ => jit,
            }
        }
        fn fork_memory() -> CodeMemory {
            match std::env::var("X86JIT_CODE_MEMORY_FORK_WORKER")
                .unwrap()
                .as_str()
            {
                "wx" => anonymous_memory(CodeMemoryPolicy::WriteXorExecute).1,
                "rwx" => anonymous_memory(CodeMemoryPolicy::ReadWriteExecute).1,
                _ => CodeMemory::new(Arc::new(CodeCounters::default())),
            }
        }

        fn assert_absent(address: usize) {
            #[cfg(target_os = "macos")]
            {
                unsafe extern "C" {
                    fn mach_vm_region(
                        task: u32,
                        address: *mut u64,
                        size: *mut u64,
                        flavor: i32,
                        info: *mut i32,
                        count: *mut u32,
                        object: *mut u32,
                    ) -> i32;
                    fn mach_port_deallocate(task: u32, port: u32) -> i32;
                }
                let mut found = address as u64;
                let mut size = 0;
                let mut info = [0i32; 9];
                let mut count = info.len() as u32;
                let mut object = 0;
                // SAFETY: VM_REGION_BASIC_INFO_64 has nine integer words; all
                // output buffers are live, and the returned object port is released.
                #[allow(deprecated)]
                unsafe {
                    let task = libc::mach_task_self();
                    let result = mach_vm_region(
                        task,
                        &mut found,
                        &mut size,
                        9,
                        info.as_mut_ptr(),
                        &mut count,
                        &mut object,
                    );
                    if object != 0 {
                        assert_eq!(mach_port_deallocate(task, object), 0);
                    }
                    assert!(result == 1 || (result == 0 && found > address as u64),
                        "mapping at {address:#x}: result={result}, found={found:#x}, size={size:#x}, inheritance={}", info[2]);
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                let mut residency = 0u8;
                // SAFETY: mincore queries the address without dereferencing it.
                assert_eq!(
                    unsafe {
                        libc::mincore(
                            address as *mut _,
                            region::page::size(),
                            (&mut residency as *mut u8).cast(),
                        )
                    },
                    -1
                );
                assert_eq!(
                    io::Error::last_os_error().raw_os_error(),
                    Some(libc::ENOMEM)
                );
            }
        }
        fn build_and_execute() {
            use x86jit_core::{Exit, Prot, Reg, RegionKind, Vm, VmConfig};
            let mut vm = Vm::with_backend(VmConfig::flat(65536), Box::new(fork_backend()));
            vm.map(0, 65536, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0x1000, &[0xb8, 42, 0, 0, 0, 0xf4]).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, 0x1000);
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rax), 42);
            assert_eq!(vm.memory_stats().backend.compiled_units_total, 1);
        }
        fn child_work(memory: &mut CodeMemory, depth: u8) {
            assert_eq!(x86jit_core::codemap::memory_bytes(), 0);
            assert_eq!(x86jit_core::codemap::memory_stats().total(), 0);
            assert!(!x86jit_core::codemap::unregister(memory.regions[0].rx, 1));
            for r in &memory.regions {
                assert_absent(r.rw);
                assert_absent(r.rx);
                assert_absent(r.rw + r.len - region::page::size());
            }
            assert_eq!(
                memory
                    .allocate(16, 16, JITMemoryKind::Executable)
                    .unwrap_err()
                    .kind(),
                io::ErrorKind::PermissionDenied
            );
            memory
                .with_write_access(&mut || panic!("inherited write closure ran"))
                .unwrap_err();
            assert!(memory.finalize(BranchProtection::None).is_err());
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || memory.executable_ptr(memory.regions[0].rw as *mut u8)
            ))
            .is_err());
            // A fresh arena may reuse the old virtual address; it must contain
            // new backing, and freeing inherited metadata must not unmap it.
            build_and_execute();
            let mut fresh = fork_memory();
            let ptr = fresh.allocate(4096, 16, JITMemoryKind::Executable).unwrap();
            fresh
                .with_write_access(&mut || {
                    // SAFETY: ptr is a fresh writable allocation in this scope.
                    unsafe { ptr.write(0x55) };
                })
                .unwrap();
            // SAFETY: inherited storage is absent; this must leave fresh storage alone.
            unsafe { memory.free_memory() };
            // SAFETY: fresh storage remains readable after inherited release.
            assert_eq!(unsafe { ptr.read() }, 0x55);
            if depth != 0 {
                // SAFETY: no engine worker is active; child exits without copied destructors.
                let pid = unsafe { libc::fork() };
                assert!(pid >= 0);
                if pid == 0 {
                    child_work(&mut fresh, depth - 1);
                    // SAFETY: avoid copied harness state and parent destructors.
                    unsafe { libc::_exit(0) };
                }
                let mut status = 0;
                // SAFETY: pid is our child and status is writable.
                assert_eq!(unsafe { libc::waitpid(pid, &mut status, 0) }, pid);
                assert_eq!(status, 0);
            }
        }
        let mut memory = fork_memory();
        let first = memory
            .allocate(REGION_BYTES, 16, JITMemoryKind::Executable)
            .unwrap();
        memory
            .with_write_access(&mut || {
                // SAFETY: populate the entire 128 MiB logical region in a write scope.
                unsafe { first.write_bytes(0x33, REGION_BYTES) };
            })
            .unwrap();
        memory.allocate(8, 8, JITMemoryKind::ReadOnly).unwrap();
        memory.allocate(8, 8, JITMemoryKind::Writable).unwrap();
        memory.finalize(BranchProtection::None).unwrap();
        // Include a real compiled entry before fork, as well as populated storage.
        build_and_execute();
        let mut pipe = [0; 2];
        // SAFETY: pipe receives two descriptors owned by this test.
        assert_eq!(unsafe { libc::pipe(pipe.as_mut_ptr()) }, 0);
        let start = std::time::Instant::now();
        #[cfg(target_os = "macos")]
        let pid = os::fork_with_locked_arena();
        // SAFETY: provider locks are released; the subprocess watchdog bounds fork itself.
        #[cfg(not(target_os = "macos"))]
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0);
        if pid == 0 {
            // SAFETY: wait for parent-first allocation, then close our pipe ends.
            unsafe {
                libc::close(pipe[1]);
                let mut ready = 0u8;
                assert_eq!(libc::read(pipe[0], (&mut ready as *mut u8).cast(), 1), 1);
                libc::close(pipe[0]);
            }
            child_work(&mut memory, 1);
            // SAFETY: avoid running the copied harness/destructors.
            unsafe { libc::_exit(0) };
        }
        eprintln!(
            "fork: {:?}, populated bytes: {REGION_BYTES}, macOS reservation bytes: {}",
            start.elapsed(),
            if cfg!(target_os = "macos") {
                1u64 << 40
            } else {
                0
            }
        );
        // Populate the next region before the child checks its mappings, then
        // keep allocating while it rebuilds and executes new code.
        let tail = memory
            .allocate(4096, 16, JITMemoryKind::Executable)
            .unwrap();
        memory
            .with_write_access(&mut || {
                // SAFETY: tail covers 4096 bytes in the current write scope.
                unsafe { tail.write_bytes(0x44, 4096) };
            })
            .unwrap();
        // SAFETY: signal parent-first completion and release our descriptors.
        unsafe {
            libc::close(pipe[0]);
            assert_eq!(libc::write(pipe[1], [1u8].as_ptr().cast(), 1), 1);
            libc::close(pipe[1]);
        }
        for _ in 0..1000 {
            memory.allocate(64, 16, JITMemoryKind::Executable).unwrap();
        }
        let mut status = 0;
        // SAFETY: pid is our child and status is writable.
        assert_eq!(unsafe { libc::waitpid(pid, &mut status, 0) }, pid);
        assert_eq!(status, 0);
        // SAFETY: both parent allocations remain live and readable.
        assert_eq!(unsafe { first.read() }, 0x33);
        // SAFETY: tail remains owned by the parent provider.
        assert_eq!(unsafe { tail.read() }, 0x44);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_unsupported_map_jit_fails_closed() {
        if os::jit_write_protection_supported() {
            return;
        }
        let counters = Arc::new(CodeCounters::default());
        assert!(CodeMemory::new(counters.clone())
            .allocate(16, 16, JITMemoryKind::Executable)
            .is_err());
        let mut fallback = provider(counters.clone());
        let rw = fallback
            .allocate(16, 16, JITMemoryKind::Executable)
            .unwrap();
        fallback
            .with_write_access(&mut || {
                // SAFETY: fresh writable allocation in the default provider.
                unsafe { rw.write_bytes(0, 16) };
            })
            .unwrap();
        fallback.finalize(BranchProtection::None).unwrap();
        assert_eq!(counters.snapshot().code_regions, 0);
        // SAFETY: the allocation was never executed or published.
        unsafe { fallback.free_memory() };
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_write_window_restores_execute_after_unwind() {
        if !os::jit_write_protection_supported() {
            return;
        }
        use x86jit_core::{Exit, Prot, Reg, RegionKind, Vm, VmConfig};
        let mut vm = Vm::with_backend(VmConfig::flat(65536), Box::new(crate::JitBackend::new()));
        vm.map(0, 65536, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, &[0xb8, 7, 0, 0, 0, 0xf4]).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(vm.memory_stats().backend.compiled_units_total, 1);
        let panic = std::panic::catch_unwind(|| {
            os::with_write_access(
                &mut || {
                    panic!("copy aborted");
                },
                &CodeCounters::default(),
            )
            .unwrap();
        });
        assert!(panic.is_err());
        // Reenter the already compiled block: a still-writable MAP_JIT thread
        // cannot execute it. Avoid private pthread state-query APIs.
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rax), 7);
    }

    // These raw-provider witnesses publish no callable pointer or fault range.
    // Their release calls therefore meet both grace periods without participants.
    fn release(memory: &mut CodeMemory, rw: *mut u8) -> (usize, u64) {
        let rx = memory.executable_ptr(rw);
        let generation = memory.allocation_generation(rx).unwrap();
        // SAFETY: the raw extent has never escaped or executed.
        assert!(unsafe { memory.release_allocation(rx, generation) });
        (rx as usize, generation)
    }

    fn page_states(address: usize, len: usize) -> Vec<u8> {
        let mut pages = vec![0u8; len.div_ceil(region::page::size())];
        // SAFETY: mincore only inspects the retained mapping; the vector covers
        // every page. It does not read discarded bytes back into physical memory.
        assert_eq!(
            unsafe { libc::mincore(address as *mut _, len, pages.as_mut_ptr().cast()) },
            0
        );
        pages
    }

    fn resident(address: usize, len: usize) -> usize {
        page_states(address, len)
            .iter()
            .filter(|page| **page & 1 != 0)
            .count()
    }

    fn assert_discarded(address: usize, len: usize) {
        #[cfg(not(target_os = "macos"))]
        assert_eq!(resident(address, len), 0, "shared backing must be removed");
        #[cfg(target_os = "macos")]
        {
            // XNU MADV_FREE surrenders dirty backing; clean resident pages can
            // remain until the kernel reclaims them. Do not equate this with RSS.
            let dirty = (libc::MINCORE_MODIFIED | libc::MINCORE_MODIFIED_OTHER) as u8;
            assert!(page_states(address, len)
                .iter()
                .all(|page| page & dirty == 0));
        }
    }

    #[test]
    fn released_extents_coalesce_discard_backing_and_republish_below_watermark() {
        #[cfg(target_os = "macos")]
        if !os::jit_write_protection_supported() {
            return;
        }
        let counters = Arc::new(CodeCounters::default());
        let mut memory = CodeMemory::new(counters.clone());
        let page = region::page::size();
        let first = memory
            .allocate(page, page as u64, JITMemoryKind::Executable)
            .unwrap();
        let second = memory
            .allocate(page, page as u64, JITMemoryKind::Executable)
            .unwrap();
        let live = memory
            .allocate(page, page as u64, JITMemoryKind::Executable)
            .unwrap();
        let rx = memory.executable_ptr(first) as usize;
        memory
            .with_write_access(&mut || {
                // SAFETY: all three contiguous pages belong to this write window.
                unsafe { first.write_bytes(0xa5, 3 * page) };
            })
            .unwrap();
        memory.finalize(BranchProtection::None).unwrap();
        assert_eq!(resident(first as usize, 2 * page), 2);
        #[cfg(target_os = "macos")]
        {
            let dirty = (libc::MINCORE_MODIFIED | libc::MINCORE_MODIFIED_OTHER) as u8;
            assert!(page_states(first as usize, 2 * page)
                .iter()
                .all(|page| page & dirty != 0));
        }
        // Fault in the RX view too: decommit must remove shared backing rather
        // than just one view's page-table entries.
        // SAFETY: finalized readable pages; neither has been released yet.
        assert_eq!(unsafe { (rx as *const u8).read_volatile() }, 0xa5);
        let (_, generation) = release(&mut memory, first);
        release(&mut memory, second);
        assert_eq!(memory.regions[0].free.len(), 1);
        assert_eq!(memory.regions[0].free[0], 0..2 * page);
        assert_discarded(first as usize, 2 * page);
        assert_discarded(rx, 2 * page);
        assert_eq!(counters.snapshot().code_bytes_committed, page as u64);
        // SAFETY: a live page outside the discarded ranges remains readable.
        assert_eq!(unsafe { live.read_volatile() }, 0xa5);
        let reused = memory
            .allocate(2 * page, page as u64, JITMemoryKind::Executable)
            .unwrap();
        assert_eq!(reused, first);
        assert_ne!(
            memory.allocation_generation(rx as *const u8),
            Some(generation)
        );
        // SAFETY: an obsolete handle must not free its replacement.
        assert!(!unsafe { memory.release_allocation(rx as *const u8, generation) });
        assert_eq!(memory.regions[0].dirty.len(), 1);
        assert_eq!(memory.regions[0].dirty[0], 0..2 * page);
        memory
            .with_write_access(&mut || {
                // SAFETY: the new generation owns the reused writable extent.
                unsafe { reused.write_bytes(0x5a, 2 * page) };
            })
            .unwrap();
        inject(CodeMemoryFailureOp::CacheMaintenance, libc::EIO);
        assert!(
            memory.finalize(BranchProtection::None).is_err(),
            "reused code needs cache maintenance"
        );
        memory.finalize(BranchProtection::None).unwrap();
        assert_eq!(counters.snapshot().code_bytes_committed, (3 * page) as u64);
        release(&mut memory, reused);
        release(&mut memory, live);
        assert_eq!(counters.snapshot().code_bytes_committed, 0);
        assert!(memory.allocations.is_empty());
        assert!(memory.allocation_by_rx.is_empty());
        #[cfg(any(target_os = "linux", target_os = "freebsd"))]
        {
            assert_eq!(counters.snapshot().code_regions, 0);
            assert_eq!(counters.snapshot().code_bytes_reserved, 0);
        }
        #[cfg(target_os = "macos")]
        {
            assert_eq!(counters.snapshot().code_regions, 1);
            assert_discarded(rx, 3 * page);
            assert!(
                region::query(rx as *const u8).is_ok(),
                "MAP_JIT reservation retained"
            );
        }
        // SAFETY: no published pointers; releases remaining non-code views, if any.
        unsafe { memory.free_memory() };
    }

    #[test]
    fn decommit_failure_retains_commitment_but_allows_safe_reuse() {
        #[cfg(target_os = "macos")]
        if !os::jit_write_protection_supported() {
            return;
        }
        let counters = Arc::new(CodeCounters::default());
        let mut memory = CodeMemory::new(counters.clone());
        let page = region::page::size();
        let old = memory
            .allocate(page, 16, JITMemoryKind::Executable)
            .unwrap();
        memory
            .allocate(page, 16, JITMemoryKind::Executable)
            .unwrap();
        inject(CodeMemoryFailureOp::DecommitPages, libc::EIO);
        release(&mut memory, old);
        assert_eq!(counters.snapshot().code_bytes_committed, (2 * page) as u64);
        assert_eq!(counters.snapshot().code_allocation_failures, 1);
        assert_failure(&counters, CodeMemoryFailureOp::DecommitPages, libc::EIO);
        assert!(memory.regions[0].decommitted.is_empty());
        assert_eq!(
            memory
                .allocate(page, 16, JITMemoryKind::Executable)
                .unwrap(),
            old
        );
        // SAFETY: raw allocations have never escaped.
        unsafe { memory.free_memory() };
    }

    #[test]
    fn aligned_hole_splitting_preserves_neighbors_and_stable_region_indices() {
        #[cfg(target_os = "macos")]
        if !os::jit_write_protection_supported() {
            return;
        }
        let mut memory = CodeMemory::new(Arc::new(CodeCounters::default()));
        let a = memory.allocate(7, 1, JITMemoryKind::Executable).unwrap();
        let hole = memory.allocate(200, 1, JITMemoryKind::Executable).unwrap();
        let neighbor = memory.allocate(9, 1, JITMemoryKind::Executable).unwrap();
        let huge = memory
            .allocate(REGION_BYTES + 1, 16, JITMemoryKind::Executable)
            .unwrap();
        let neighbor_rx = memory.executable_ptr(neighbor);
        let neighbor_gen = memory.allocation_generation(neighbor_rx);
        let huge_rx = memory.executable_ptr(huge);
        let huge_gen = memory.allocation_generation(huge_rx);
        release(&mut memory, hole);
        let aligned = memory.allocate(32, 64, JITMemoryKind::Executable).unwrap();
        assert_eq!(aligned as usize % 64, 0);
        assert_eq!(memory.executable_ptr(aligned) as usize % 64, 0);
        assert!(aligned as usize >= hole as usize && (aligned as usize) < neighbor as usize);
        assert_eq!(memory.regions[0].free.len(), 2);
        release(&mut memory, aligned);
        release(&mut memory, a);
        assert_eq!(memory.regions[0].free.len(), 1);
        assert_eq!(memory.regions[0].free[0], 0..207);
        release(&mut memory, neighbor);
        assert_eq!(memory.allocation_generation(huge_rx), huge_gen);
        assert_eq!(memory.allocation_generation(neighbor_rx), None);
        assert!(neighbor_gen.is_some());
        // Reuse the vacant region record without changing the huge unit's index.
        let ro = memory.allocate(8, 8, JITMemoryKind::ReadOnly).unwrap();
        assert_eq!(memory.allocation_generation(huge_rx), huge_gen);
        release(&mut memory, ro);
        release(&mut memory, huge);
        assert!(memory.allocations.is_empty());
        // SAFETY: no raw pointer has escaped.
        unsafe { memory.free_memory() };
    }

    #[test]
    fn anonymous_release_only_replaces_wholly_free_reservations() {
        let (counters, mut memory) = anonymous_memory(CodeMemoryPolicy::WriteXorExecute);
        let page = region::page::size();
        let first = memory.allocate(40, 16, JITMemoryKind::Executable).unwrap();
        memory.finalize(BranchProtection::None).unwrap();
        let live = memory.allocate(40, 16, JITMemoryKind::Executable).unwrap();
        memory.finalize(BranchProtection::None).unwrap();
        let old_rx = memory.executable_ptr(first);
        let old_generation = memory.allocation_generation(old_rx).unwrap();
        release(&mut memory, first);
        assert_eq!(
            region::query(old_rx).unwrap().protection(),
            region::Protection::READ_EXECUTE
        );
        let third = memory.allocate(40, 16, JITMemoryKind::Executable).unwrap();
        assert!(third as usize >= first as usize + 2 * page);
        memory.finalize(BranchProtection::None).unwrap();
        release(&mut memory, live);
        release(&mut memory, third);
        assert_eq!(counters.snapshot().code_regions, 0);
        assert_eq!(counters.snapshot().code_bytes_committed, 0);
        assert!(memory.allocations.is_empty());
        assert!(
            region::query(old_rx).is_err(),
            "the old anonymous reservation was unmapped"
        );
        let fresh = memory.allocate(40, 16, JITMemoryKind::Executable).unwrap();
        let fresh_rx = memory.executable_ptr(fresh);
        assert_ne!(memory.allocation_generation(fresh_rx), Some(old_generation));
        // SAFETY: the old identity cannot release storage even if mmap reused its address.
        assert!(!unsafe { memory.release_allocation(old_rx, old_generation) });
        release(&mut memory, fresh);
    }

    #[test]
    fn allocation_generations_include_unpublished_and_failed_extents() {
        #[cfg(target_os = "macos")]
        if !os::jit_write_protection_supported() {
            return;
        }
        let counters = Arc::new(CodeCounters::default());
        let mut memory = CodeMemory::new(counters);
        let rw = memory.allocate(40, 16, JITMemoryKind::Executable).unwrap();
        let rx = memory.executable_ptr(rw);
        let generation = memory.allocation_generation(rx).unwrap();
        assert_ne!(generation, 0);
        assert_eq!(generation, memory.allocations[0].generation);
        assert_eq!(memory.allocation_generation(rx.wrapping_add(1)), None);
        assert!(memory.finalize(BranchProtection::BTI).is_err());
        assert_eq!(memory.allocation_generation(rx), Some(generation));
        let next_rw = memory.allocate(40, 16, JITMemoryKind::Executable).unwrap();
        let next_rx = memory.executable_ptr(next_rw);
        assert_ne!(memory.allocation_generation(next_rx), Some(generation));
        memory.finalize(BranchProtection::None).unwrap();
        assert_eq!(memory.allocation_generation(rx), Some(generation));
        assert_eq!(memory.allocations.len(), 2);
        // SAFETY: no entry was executed or registered in the code map.
        unsafe { memory.free_memory() };
        assert_eq!(memory.allocation_generation(rx), None);
        assert!(memory.allocation_by_rx.is_empty());
    }

    #[test]
    fn dense_tails_growth_kinds_and_failure_accounting() {
        #[cfg(target_os = "macos")]
        if !os::jit_write_protection_supported() {
            return;
        }
        let counters = Arc::new(CodeCounters::default());
        let mut memory = CodeMemory::new(counters.clone());
        let a = memory.allocate(40, 16, JITMemoryKind::Executable).unwrap();
        memory
            .with_write_access(&mut || {
                // SAFETY: a is a writable allocation, and the write scope is active.
                unsafe { a.write_bytes(0, 40) };
            })
            .unwrap();
        memory.finalize(BranchProtection::None).unwrap();
        let b = memory.allocate(40, 16, JITMemoryKind::Executable).unwrap();
        assert_eq!(b as usize - a as usize, 48);
        let ro = memory.allocate(8, 8, JITMemoryKind::ReadOnly).unwrap();
        let rw = memory.allocate(8, 8, JITMemoryKind::Writable).unwrap();
        assert_ne!(ro, rw);
        assert_ne!(memory.executable_ptr(ro), ro);
        assert_eq!(memory.executable_ptr(rw), rw);
        // A request larger than one region grows without discarding earlier tails.
        memory
            .allocate(REGION_BYTES + 1, 16, JITMemoryKind::Executable)
            .unwrap();
        let stats = counters.snapshot();
        assert_eq!(stats.code_regions, 2);
        assert_eq!(stats.code_bytes_used, (REGION_BYTES + 89) as u64);
        assert_eq!(
            stats.code_bytes_free,
            stats.code_bytes_reserved - stats.code_bytes_used
        );
        assert_eq!(stats.code_largest_free_extent, (REGION_BYTES - 88) as u64);
        assert!(memory
            .allocate(usize::MAX, 16, JITMemoryKind::Executable)
            .is_err());
        assert_eq!(counters.snapshot().code_allocation_failures, 1);
        assert_eq!(memory.allocations[0].offset, 0);
        assert_eq!(memory.allocations[1].offset, 48);
        assert_eq!(memory.allocations[1].len, 40);
        let align = region::page::size() * 4;
        let aligned = memory
            .allocate(40, align as u64, JITMemoryKind::Executable)
            .unwrap();
        assert_eq!(aligned as usize % align, 0);
        assert_eq!(memory.executable_ptr(aligned) as usize % align, 0);
        // SAFETY: nothing escaped or executed; no signal metadata was registered.
        unsafe { memory.free_memory() };
    }

    /// Use real shared RW/R mappings even on macOS to exercise nonzero alias
    /// offsets. No instructions are executed here; byte witnesses cover both ISAs.
    struct TestMemory {
        memory: CodeMemory,
        observed: Arc<Observed>,
        aliases: bool,
    }
    #[derive(Default)]
    struct Observed {
        pairs: Mutex<Vec<(usize, usize)>>,
        generations: Mutex<Vec<u64>>,
        failure: AtomicU8,
        writes: AtomicUsize,
        finalizations: AtomicUsize,
    }
    impl JITMemoryProvider for TestMemory {
        fn allocate(
            &mut self,
            size: usize,
            align: u64,
            kind: JITMemoryKind,
        ) -> io::Result<*mut u8> {
            let kind = if self.aliases {
                JITMemoryKind::ReadOnly
            } else {
                kind
            };
            let rw = self.memory.allocate(size, align, kind)?;
            // Read reservation bookkeeping independently of executable_ptr and
            // the module's returned pointer; either may be the source of a bug.
            let allocation = self.memory.allocations.last().unwrap();
            let region = &self.memory.regions[allocation.region];
            self.observed
                .pairs
                .lock()
                .unwrap()
                .push((rw as usize, region.rx + allocation.offset));
            self.observed
                .generations
                .lock()
                .unwrap()
                .push(allocation.generation);
            Ok(rw)
        }
        fn allocation_generation(&self, executable: *const u8) -> Option<u64> {
            self.memory.allocation_generation(executable)
        }
        unsafe fn release_allocation(&mut self, executable: *const u8, generation: u64) -> bool {
            // SAFETY: forwarded module retirement contract; these fixtures never execute.
            unsafe { self.memory.release_allocation(executable, generation) }
        }
        fn supports_relocation(&self, kind: Reloc) -> bool {
            self.memory.supports_relocation(kind)
        }
        fn executable_ptr(&self, ptr: *mut u8) -> *const u8 {
            self.memory.executable_ptr(ptr)
        }
        fn with_write_access(&self, action: &mut dyn FnMut()) -> io::Result<()> {
            self.observed.writes.fetch_add(1, Ordering::Relaxed);
            if self.observed.failure.load(Ordering::Relaxed) == 1 {
                return Err(io::ErrorKind::PermissionDenied.into());
            }
            if self.aliases {
                action();
                Ok(())
            } else {
                self.memory.with_write_access(action)
            }
        }
        fn finalize(&mut self, branch: BranchProtection) -> ModuleResult<()> {
            self.observed.finalizations.fetch_add(1, Ordering::Relaxed);
            if self.observed.failure.load(Ordering::Relaxed) == 2 {
                return Err(ModuleError::Allocation {
                    err: io::ErrorKind::PermissionDenied.into(),
                });
            }
            if self.aliases {
                Ok(())
            } else {
                self.memory.finalize(branch)
            }
        }
        unsafe fn free_memory(&mut self) {
            // SAFETY: forwarded contract; these fixtures never execute.
            unsafe { self.memory.free_memory() };
        }
    }

    fn module(aliases: bool) -> (JITModule, Arc<Observed>) {
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(cranelift::codegen::settings::Flags::new(
                cranelift::codegen::settings::builder(),
            ))
            .unwrap();
        let observed = Arc::new(Observed::default());
        let mut builder = crate::libcalls::builder_with_memory(
            isa,
            Box::new(TestMemory {
                memory: CodeMemory::new(Arc::new(CodeCounters::default())),
                observed: observed.clone(),
                aliases,
            }),
        );
        builder.symbol("far_target", 0x1000usize as *const u8);
        (JITModule::new(builder), observed)
    }

    #[test]
    fn absolute_and_cross_unit_pcrel_relocations_use_runtime_addresses() {
        let (mut m, observed) = module(true);
        let sig = m.make_signature();
        let target = m.declare_function("target", Linkage::Local, &sig).unwrap();
        let caller = m.declare_function("caller", Linkage::Local, &sig).unwrap();
        let name = ModuleRelocTarget::user(0, target.as_u32());
        m.define_function_bytes(target, 16, &[0; 16], &[]).unwrap();
        m.define_function_bytes(
            caller,
            16,
            &[0; 16],
            &[
                ModuleReloc {
                    offset: 0,
                    kind: Reloc::Abs8,
                    name: name.clone(),
                    addend: 3,
                },
                ModuleReloc {
                    offset: 8,
                    kind: Reloc::X86PCRel4,
                    name,
                    addend: -4,
                },
            ],
        )
        .unwrap();
        m.finalize_definitions().unwrap();
        let target = m.get_finalized_function(target);
        let caller = m.get_finalized_function(caller);
        let pairs = observed.pairs.lock().unwrap();
        assert_eq!(target as usize, pairs[0].1);
        assert_eq!(caller as usize, pairs[1].1);
        assert_ne!(pairs[0].0, pairs[0].1);
        assert_ne!(pairs[1].0, pairs[1].1);
        // SAFETY: finalized, readable 16-byte allocation, checked unaligned fields.
        unsafe {
            assert_eq!(caller.cast::<u64>().read_unaligned(), pairs[0].1 as u64 + 3);
            assert_eq!(
                caller.add(8).cast::<i32>().read_unaligned() as isize,
                pairs[0].1 as isize - 4 - (pairs[1].1 as isize + 8)
            );
            m.free_memory();
        }
    }

    #[test]
    fn byte_definitions_are_counted_through_retirement_and_redefinition() {
        let (mut m, _) = module(true);
        let sig = m.make_signature();
        let anchor = m.declare_function("anchor", Linkage::Local, &sig).unwrap();
        let reused = m.declare_function("reused", Linkage::Local, &sig).unwrap();
        m.define_function_bytes(anchor, 16, &[0; 16], &[]).unwrap();
        assert_eq!(m.defined_function_count(), 1, "pending anchor definition");
        m.finalize_definitions().unwrap();
        for _ in 0..32 {
            m.define_function_bytes(reused, 16, &[0; 16], &[]).unwrap();
            assert_eq!(m.defined_function_count(), 2, "pending replacement");
            m.finalize_definitions().unwrap();
            assert_eq!(m.defined_function_count(), 2, "finalized replacement");
            let entry = m.get_finalized_function(reused);
            let generation = m.get_finalized_function_generation(reused).unwrap();
            // SAFETY: fixture bytes are never executed or registered, and no
            // relocation or external consumer refers to the retired definition.
            unsafe {
                m.retire_function_definition(reused);
                assert_eq!(
                    m.defined_function_count(),
                    1,
                    "live anchor after retirement"
                );
                assert!(m.release_function_allocation(entry, generation));
            }
            assert_eq!(m.declarations().get_functions().count(), 2);
        }
        // SAFETY: neither fixture definition has any execution or readers.
        unsafe { m.free_memory() };
    }

    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    #[test]
    fn native_alias_relocations_survive_retirement_and_reuse() {
        stress::isolated(
            "code_memory::tests::native_alias_relocations_survive_retirement_and_reuse",
            |deadline| {
                // Unlike the read-only dual-view byte fixtures, this path uses the
                // production executable alias and its actual publication protocol.
                let (mut m, observed) = module(false);
                let sig = m.make_signature();
                let anchor = m.declare_function("anchor", Linkage::Local, &sig).unwrap();
                let target = m.declare_function("target", Linkage::Local, &sig).unwrap();
                let caller = m.declare_function("caller", Linkage::Local, &sig).unwrap();
                m.define_function_bytes(anchor, 16, &[0; 16], &[]).unwrap();
                m.finalize_definitions().unwrap();
                let mut previous = None;
                for cycle in 0..32 {
                    stress::remaining(deadline);
                    m.define_function_bytes(target, 16, &[0; 16], &[]).unwrap();
                    let name = ModuleRelocTarget::user(0, target.as_u32());
                    m.define_function_bytes(
                        caller,
                        16,
                        &[0; 16],
                        &[
                            ModuleReloc {
                                offset: 0,
                                kind: Reloc::Abs8,
                                name: name.clone(),
                                addend: cycle,
                            },
                            ModuleReloc {
                                offset: 8,
                                kind: Reloc::X86PCRel4,
                                name,
                                addend: -4,
                            },
                        ],
                    )
                    .unwrap();
                    m.finalize_definitions().unwrap();
                    let target_rx = m.get_finalized_function(target);
                    let caller_rx = m.get_finalized_function(caller);
                    let target_generation = m.get_finalized_function_generation(target).unwrap();
                    let caller_generation = m.get_finalized_function_generation(caller).unwrap();
                    let pairs = observed.pairs.lock().unwrap();
                    let pair = pairs[pairs.len() - 1];
                    assert_ne!(pair.0, pair.1);
                    assert_eq!(pair.1, caller_rx as usize);
                    drop(pairs);
                    if let Some((address, generation)) = previous {
                        assert_eq!(caller_rx as usize, address);
                        assert_ne!(caller_generation, generation);
                    }
                    previous = Some((caller_rx as usize, caller_generation));
                    // SAFETY: finalized readable bytes; no fixture code is executed.
                    unsafe {
                        assert_eq!(
                            caller_rx.cast::<u64>().read_unaligned(),
                            target_rx as u64 + cycle as u64
                        );
                        assert_eq!(
                            caller_rx.add(8).cast::<i32>().read_unaligned() as isize,
                            target_rx as isize - 4 - (caller_rx as isize + 8)
                        );
                        // Clear both records before releasing either relocation target.
                        m.retire_function_definition(caller);
                        m.retire_function_definition(target);
                        assert!(m.release_function_allocation(caller_rx, caller_generation));
                        assert!(m.release_function_allocation(target_rx, target_generation));
                        assert!(!m.release_function_allocation(caller_rx, caller_generation));
                    }
                    assert_eq!(m.defined_function_count(), 1);
                    assert_eq!(m.declarations().get_functions().count(), 3);
                }
                // SAFETY: anchor also has no execution, readers or external consumers.
                unsafe {
                    m.free_memory();
                }
            },
        );
    }

    #[test]
    fn arm64_direct_call_and_far_veneer_use_runtime_addresses() {
        let (mut m, observed) = module(true);
        let sig = m.make_signature();
        let far = m
            .declare_function("far_target", Linkage::Import, &sig)
            .unwrap();
        let target = m
            .declare_function("near_target", Linkage::Local, &sig)
            .unwrap();
        let caller = m.declare_function("caller", Linkage::Local, &sig).unwrap();
        m.define_function_bytes(target, 16, &[0; 16], &[]).unwrap();
        let calls = [0x94000000u32.to_le_bytes(), 0x94000000u32.to_le_bytes()].concat();
        m.define_function_bytes(
            caller,
            16,
            &calls,
            &[
                ModuleReloc {
                    offset: 0,
                    kind: Reloc::Arm64Call,
                    name: ModuleRelocTarget::user(0, target.as_u32()),
                    addend: 0,
                },
                ModuleReloc {
                    offset: 4,
                    kind: Reloc::Arm64Call,
                    name: ModuleRelocTarget::user(0, far.as_u32()),
                    addend: 0,
                },
            ],
        )
        .unwrap();
        m.finalize_definitions().unwrap();
        let target = m.get_finalized_function(target);
        let caller = m.get_finalized_function(caller);
        let pairs = observed.pairs.lock().unwrap();
        assert_eq!(target as usize, pairs[0].1);
        assert_eq!(caller as usize, pairs[1].1);
        assert_ne!(pairs[0].0, pairs[0].1);
        assert_ne!(pairs[1].0, pairs[1].1);
        // SAFETY: two calls plus veneer capacity reserved by CompiledBlob.
        unsafe {
            let instruction = caller.cast::<u32>().read_unaligned();
            let displacement = ((instruction << 6) as i32 >> 6) as isize * 4;
            assert_eq!(pairs[1].1 as isize + displacement, pairs[0].1 as isize);
            assert_eq!(caller.add(4).cast::<u32>().read_unaligned(), 0x94000001);
            assert_eq!(caller.add(8).cast::<u32>().read_unaligned(), 0x58000050);
            assert_eq!(caller.add(16).cast::<u64>().read_unaligned(), 0x1000);
            m.free_memory();
        }
    }

    #[test]
    fn unsupported_relocations_fail_before_publication() {
        let (mut m, _) = module(true);
        let sig = m.make_signature();
        let f = m
            .declare_function("unsupported", Linkage::Local, &sig)
            .unwrap();
        assert!(m
            .define_function_bytes(
                f,
                16,
                &[0; 8],
                &[ModuleReloc {
                    offset: 0,
                    kind: Reloc::X86GOTPCRel4,
                    name: ModuleRelocTarget::user(0, f.as_u32()),
                    addend: 0
                }]
            )
            .is_err());
        // SAFETY: no pointers escaped.
        unsafe { m.free_memory() };
    }

    #[test]
    fn executable_entries_equal_recorded_rx_addresses() {
        #[cfg(target_os = "macos")]
        if !os::jit_write_protection_supported() {
            return;
        }
        let (mut m, observed) = module(false);
        let sig = m.make_signature();
        let f = m.declare_function("entry", Linkage::Local, &sig).unwrap();
        m.define_function_bytes(f, 16, &[0; 16], &[]).unwrap();
        m.finalize_definitions().unwrap();
        let pairs = observed.pairs.lock().unwrap();
        let (rw, rx) = pairs[0];
        assert_eq!(m.get_finalized_function(f) as usize, rx);
        assert_eq!(
            m.get_finalized_function_generation(f),
            Some(observed.generations.lock().unwrap()[0])
        );
        if cfg!(target_os = "macos") {
            assert_eq!(rw, rx);
        } else {
            assert_ne!(rw, rx);
        }
        // SAFETY: these byte fixtures never execute and no pointer escaped.
        unsafe { m.free_memory() };
    }

    #[test]
    fn failed_finalization_is_terminal_for_functions_and_data() {
        use cranelift_module::DataDescription;
        for failure in [1, 2] {
            let (mut m, observed) = module(true);
            let sig = m.make_signature();
            let prior = m.declare_function("prior", Linkage::Local, &sig).unwrap();
            m.define_function_bytes(prior, 16, &[0; 8], &[]).unwrap();
            m.finalize_definitions().unwrap();
            assert!(!m.get_finalized_function(prior).is_null());
            let f = m.declare_function("entry", Linkage::Local, &sig).unwrap();
            m.define_function_bytes(
                f,
                16,
                &[0; 8],
                &[ModuleReloc {
                    offset: 0,
                    kind: Reloc::Abs8,
                    name: ModuleRelocTarget::user(0, f.as_u32()),
                    addend: 3,
                }],
            )
            .unwrap();
            let data = m
                .declare_data("pointer", Linkage::Local, false, false)
                .unwrap();
            let mut description = DataDescription::new();
            description.define_zeroinit(8);
            let target = m.declare_func_in_data(f, &mut description);
            description.write_function_addr(0, target);
            m.define_data(data, &description).unwrap();
            observed.failure.store(failure, Ordering::Relaxed);
            assert!(m.finalize_definitions().is_err());
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || m.get_finalized_function(f)
            ))
            .is_err());
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || m.get_finalized_data(data)
            ))
            .is_err());
            observed.failure.store(0, Ordering::Relaxed);
            let writes = observed.writes.load(Ordering::Relaxed);
            let finalizations = observed.finalizations.load(Ordering::Relaxed);
            assert!(m.finalize_definitions().is_err());
            assert_eq!(observed.writes.load(Ordering::Relaxed), writes);
            assert_eq!(
                observed.finalizations.load(Ordering::Relaxed),
                finalizations
            );
            for function in [prior, f] {
                assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || m.get_finalized_function(function)
                ))
                .is_err());
            }
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || m.get_finalized_data(data)
            ))
            .is_err());
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || m.get_address(&ModuleRelocTarget::user(0, f.as_u32()))
            ))
            .is_err());
            // SAFETY: these byte fixtures never execute and no pointer escaped.
            unsafe {
                m.free_memory();
            }
        }
    }
}
