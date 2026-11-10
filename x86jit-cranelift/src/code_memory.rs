//! Dense retained code storage (spec §8.2; decision-17, phase 2).
//!
//! Allocation records deliberately outlive invalidation. Reuse/free lists and
//! execution/signal-reader grace periods belong to decision-17 phase 3.
use crate::memory_stats::CodeCounters;
use cranelift_jit::{BranchProtection, JITMemoryKind, JITMemoryProvider};
use cranelift_module::{ModuleError, ModuleResult};
use std::io;
use std::sync::{atomic::Ordering, Arc};

use x86jit_core::codemap::ProcessIdentity;

const REGION_BYTES: usize = 128 << 20;

pub(crate) fn provider(counters: Arc<CodeCounters>) -> Box<dyn JITMemoryProvider + Send> {
    #[cfg(target_os = "macos")]
    if !os::jit_write_protection_supported() {
        // Unsupported hosts must use separate RW -> RX allocation epochs,
        // rather than MAP_JIT without per-thread W^X enforcement.
        return Box::new(cranelift_jit::SystemMemoryProvider::new());
    }
    Box::new(CodeMemory::new(counters))
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
    flushed: usize,
    kind: Kind,
}

/// Stable identity for a retained allocation: region index, offset and length.
/// Indices are never reused; the monotonically increasing record index is its
/// generation until phase 3 supplies retirement notifications and recycling.
#[derive(Debug)]
#[allow(dead_code)]
struct Allocation {
    region: usize,
    offset: usize,
    len: usize,
}

pub(crate) struct CodeMemory {
    owner: ProcessIdentity,
    regions: Vec<Region>,
    allocations: Vec<Allocation>,
    counters: Arc<crate::memory_stats::RegionCounters>,
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    instruction_sync: linux_sync::Protocol,
}

impl CodeMemory {
    pub(crate) fn new(counters: Arc<CodeCounters>) -> Self {
        Self {
            owner: ProcessIdentity::current(),
            regions: Vec::new(),
            allocations: Vec::new(),
            counters: counters.register(),
            #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
            instruction_sync: linux_sync::Protocol::register(linux_sync::membarrier),
        }
    }

    fn allocate_inner(
        &mut self,
        size: usize,
        align: u64,
        kind: JITMemoryKind,
    ) -> io::Result<*mut u8> {
        self.check_process()?;
        let kind = match kind {
            JITMemoryKind::Executable => Kind::Code,
            JITMemoryKind::ReadOnly => Kind::ReadOnly,
            JITMemoryKind::Writable => Kind::Writable,
        };
        let align = usize::try_from(align).map_err(|_| io::ErrorKind::InvalidInput)?;
        if size == 0 || !align.is_power_of_two() {
            return Err(io::ErrorKind::InvalidInput.into());
        }
        // Reserve the bookkeeping before consuming an extent. Allocation failure
        // must use the same sticky CodeSpaceExhausted path as mmap failure.
        self.allocations.try_reserve(1).map_err(io::Error::other)?;
        let fits = |r: &Region| {
            let offset = r.rw.checked_add(r.used)?.checked_add(align - 1)? & !(align - 1);
            let offset = offset.checked_sub(r.rw)?;
            (offset.checked_add(size)? <= r.len && (r.rx.checked_add(offset)? & (align - 1)) == 0)
                .then_some(offset)
        };
        let found = self
            .regions
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, r)| (r.kind == kind).then(|| fits(r)).flatten().map(|p| (i, p)));
        let (index, offset) = match found {
            Some(found) => found,
            None => {
                self.regions.try_reserve(1).map_err(io::Error::other)?;
                let page = region::page::size();
                let len = size
                    .checked_add(align - 1)
                    .and_then(|v| v.checked_add(page - 1))
                    .map(|v| (v & !(page - 1)).max(REGION_BYTES))
                    .ok_or(io::ErrorKind::OutOfMemory)?;
                let (rw, rx) = os::reserve(len, kind, align)?;
                self.regions.push(Region {
                    rw,
                    rx,
                    len,
                    used: 0,
                    flushed: 0,
                    kind,
                });
                let index = self.regions.len() - 1;
                (
                    index,
                    fits(&self.regions[index]).expect("new region accommodates alignment"),
                )
            }
        };
        let r = &mut self.regions[index];
        r.used = offset + size;
        self.allocations.push(Allocation {
            region: index,
            offset,
            len: size,
        });
        let ptr = (r.rw + offset) as *mut u8;
        self.update_counters();
        Ok(ptr)
    }

    fn check_process(&self) -> io::Result<()> {
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
        for r in self.regions.iter().filter(|r| r.kind == Kind::Code) {
            regions += 1;
            reserved += r.len as u64;
            used += r.used as u64;
            committed += (r.used.div_ceil(region::page::size()) * region::page::size()) as u64;
            largest = largest.max((r.len - r.used) as u64);
        }
        c.regions.store(regions, Ordering::Relaxed);
        c.views.store(
            if cfg!(target_os = "macos") {
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
        c.largest.store(largest, Ordering::Relaxed);
        c.allocation_metadata.store(
            (self.allocations.capacity() * std::mem::size_of::<Allocation>()
                + self.regions.capacity() * std::mem::size_of::<Region>()) as u64,
            Ordering::Relaxed,
        );
    }
}

impl JITMemoryProvider for CodeMemory {
    fn allocate(&mut self, size: usize, align: u64, kind: JITMemoryKind) -> io::Result<*mut u8> {
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
        os::with_write_access(action)
    }

    fn finalize(&mut self, branch_protection: BranchProtection) -> ModuleResult<()> {
        let result = (|| -> io::Result<()> {
            self.check_process()?;
            // BTI is not enabled by x86jit's ISA. Reject rather than silently
            // publishing mappings without a requested protection policy.
            if branch_protection != BranchProtection::None {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "JIT BTI mappings are unsupported",
                ));
            }
            for r in &self.regions {
                if r.kind == Kind::Code && r.used > r.flushed {
                    os::clear_cache(r.rw + r.flushed, r.rx + r.flushed, r.used - r.flushed)?;
                }
            }
            // Require SYNC_CORE after registration; never accept the pinned
            // helper's GLOBAL fallback as instruction-stream synchronization.
            #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
            self.instruction_sync.flush(linux_sync::membarrier)?;
            #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
            wasmtime_jit_icache_coherence::pipeline_flush_mt().map_err(io::Error::other)?;
            Ok(())
        })();
        if let Err(err) = result {
            self.counters.failures.fetch_add(1, Ordering::Relaxed);
            return Err(ModuleError::Allocation { err });
        }
        x86jit_core::jit_abi::code_publication_barrier();
        for r in &mut self.regions {
            r.flushed = r.used;
        }
        Ok(())
    }

    unsafe fn free_memory(&mut self) {
        // SAFETY: the caller must exclude execution and signal readers. Normal
        // backend drop intentionally retains all addresses until phase 3.
        if self.check_process().is_err() {
            return;
        }
        for r in self.regions.drain(..) {
            unsafe { os::release(&r) };
        }
        self.allocations.clear();
        self.update_counters();
    }
}

#[cfg(any(target_os = "linux", target_os = "freebsd", target_os = "macos"))]
mod os {
    use super::*;

    #[cfg(target_os = "macos")]
    pub(super) fn jit_write_protection_supported() -> bool {
        // SAFETY: pthread API has no arguments or memory preconditions.
        unsafe { libc::pthread_jit_write_protect_supported_np() != 0 }
    }

    pub(super) fn reserve(len: usize, kind: Kind, align: usize) -> io::Result<(usize, usize)> {
        #[cfg(target_os = "macos")]
        {
            if kind == Kind::Code {
                mac::reserve(len, align)
            } else {
                mac::reserve_data(len, kind, align)
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            // Linux memfd_create(2); FreeBSD shm_open(2), SHM_ANON. Separate
            // aliases retain all dense tails without changing executable pages.
            #[cfg(target_os = "linux")]
            // SAFETY: static NUL-terminated name and valid flags.
            let fd = unsafe { libc::memfd_create(c"x86jit-code".as_ptr(), libc::MFD_CLOEXEC) };
            #[cfg(target_os = "freebsd")]
            // SAFETY: SHM_ANON requests an unnamed shared object.
            let fd =
                unsafe { libc::shm_open(libc::SHM_ANON, libc::O_RDWR | libc::O_CLOEXEC, 0o600) };
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: fd is uniquely owned here; the successful views own the backing
            // after close. Every partial failure unmaps already created views.
            unsafe {
                let result = (|| {
                    let length =
                        libc::off_t::try_from(len).map_err(|_| io::ErrorKind::OutOfMemory)?;
                    if libc::ftruncate(fd, length) != 0 {
                        return Err(io::Error::last_os_error());
                    }
                    let rw = map_aligned(
                        len,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_SHARED,
                        fd,
                        align,
                    )?;
                    if kind == Kind::Writable {
                        return Ok((rw as usize, rw as usize));
                    }
                    let prot = libc::PROT_READ
                        | if kind == Kind::Code {
                            libc::PROT_EXEC
                        } else {
                            0
                        };
                    let rx = match map_aligned(len, prot, libc::MAP_SHARED, fd, align) {
                        Ok(rx) => rx,
                        Err(error) => {
                            libc::munmap(rw, len);
                            return Err(error);
                        }
                    };
                    Ok((rw as usize, rx as usize))
                })();
                libc::close(fd);
                result
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
    ) -> io::Result<*mut libc::c_void> {
        // SAFETY: size is page-rounded, fd/flags describe the desired backing;
        // all partial reservations are released before returning an error.
        unsafe {
            if align <= region::page::size() {
                let ptr = libc::mmap(std::ptr::null_mut(), len, prot, flags, fd, 0);
                return if ptr == libc::MAP_FAILED {
                    Err(io::Error::last_os_error())
                } else {
                    exclude_or_unmap(ptr, len)
                };
            }
            let total = len.checked_add(align).ok_or(io::ErrorKind::OutOfMemory)?;
            let reservation = libc::mmap(
                std::ptr::null_mut(),
                total,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            );
            if reservation == libc::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
            let start = reservation as usize;
            let aligned = (start + align - 1) & !(align - 1);
            let ptr = libc::mmap(aligned as *mut _, len, prot, flags | libc::MAP_FIXED, fd, 0);
            if ptr == libc::MAP_FAILED {
                let error = io::Error::last_os_error();
                libc::munmap(reservation, total);
                return Err(error);
            }
            if aligned > start {
                libc::munmap(reservation, aligned - start);
            }
            if aligned + len < start + total {
                libc::munmap((aligned + len) as *mut _, start + total - aligned - len);
            }
            exclude_or_unmap(ptr, len)
        }
    }

    fn exclude_or_unmap(ptr: *mut libc::c_void, len: usize) -> io::Result<*mut libc::c_void> {
        if let Err(error) = exclude_from_fork(ptr, len) {
            // SAFETY: this mapping has not escaped.
            unsafe { libc::munmap(ptr, len) };
            return Err(error);
        }
        Ok(ptr)
    }

    fn exclude_from_fork(ptr: *mut libc::c_void, len: usize) -> io::Result<()> {
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
    }

    pub(super) fn with_write_access(action: &mut dyn FnMut()) -> io::Result<()> {
        #[cfg(target_os = "macos")]
        return mac::write(action);
        #[cfg(not(target_os = "macos"))]
        {
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
            unsafe { wasmtime_jit_icache_coherence::clear_cache(rx as *const _, len) }
                .map_err(io::Error::other)?;
        }
        Ok(())
    }

    pub(super) unsafe fn release(r: &Region) {
        #[cfg(target_os = "macos")]
        if r.kind == Kind::Code {
            return;
        } // process reservation is never reused
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
            state: Mutex<(usize, usize)>,
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
                    state: Mutex::new((0, 0)),
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

        pub(super) fn reserve(len: usize, align: usize) -> io::Result<(usize, usize)> {
            if !jit_write_protection_supported() {
                return Err(io::ErrorKind::Unsupported.into());
            }
            let mut arena = arena().state.lock().unwrap();
            if arena.0 == 0 {
                // SAFETY: anonymous reservation with per-thread MAP_JIT W^X.
                let ptr = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        RESERVATION,
                        libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                        libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_JIT,
                        -1,
                        0,
                    )
                };
                if ptr == libc::MAP_FAILED {
                    return Err(io::Error::last_os_error());
                }
                // XNU otherwise walks the entire private JIT object at fork,
                // including its sparse tail. Exclude the whole reservation once.
                if let Err(error) = exclude_from_fork(ptr, RESERVATION) {
                    // SAFETY: no reservation address has escaped yet.
                    unsafe { libc::munmap(ptr, RESERVATION) };
                    return Err(error);
                }
                arena.0 = ptr as usize;
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

        pub(super) fn write(action: &mut dyn FnMut()) -> io::Result<()> {
            if !jit_write_protection_supported() {
                return Err(io::ErrorKind::Unsupported.into());
            }
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
            unsafe { libc::pthread_jit_write_protect_np(0) };
            let _restore = RestoreExecute;
            action();
            Ok(())
        }

        pub(super) fn reserve_data(
            len: usize,
            kind: Kind,
            align: usize,
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
                    -1,
                    align,
                )?;
                if kind == Kind::Writable {
                    return Ok((rw as usize, rw as usize));
                }
                let mut rx = 0;
                let (mut current, mut maximum) = (0, 0);
                let task = libc::mach_task_self();
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
                    libc::munmap(rw, len);
                    return Err(io::Error::other(format!("mach_vm_remap: {result}")));
                }
                if libc::mprotect(rx as *mut _, len, libc::PROT_READ) != 0 {
                    let error = io::Error::last_os_error();
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
    pub(super) fn reserve(_: usize, _: Kind, _: usize) -> io::Result<(usize, usize)> {
        Err(io::ErrorKind::Unsupported.into())
    }
    pub(super) fn with_write_access(action: &mut dyn FnMut()) -> io::Result<()> {
        action();
        Ok(())
    }
    pub(super) fn clear_cache(_: usize, _: usize, _: usize) -> io::Result<()> {
        Ok(())
    }
    pub(super) unsafe fn release(_: &Region) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use cranelift::codegen::binemit::Reloc;
    use cranelift_jit::JITModule;
    use cranelift_module::{Linkage, Module, ModuleReloc, ModuleRelocTarget};
    use std::sync::{
        atomic::{AtomicU8, AtomicUsize},
        Mutex,
    };

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
            let mut child = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", TEST, "--test-threads", "2", "--nocapture"])
                .env(WORKER, "1")
                .process_group(0)
                .spawn()
                .unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
            loop {
                if let Some(status) = child.try_wait().unwrap() {
                    assert!(status.success(), "fork worker: {status}");
                    return;
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
            let mut vm =
                Vm::with_backend(VmConfig::flat(65536), Box::new(crate::JitBackend::new()));
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
            let mut fresh = CodeMemory::new(Arc::new(CodeCounters::default()));
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
        let mut memory = CodeMemory::new(Arc::new(CodeCounters::default()));
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
            os::with_write_access(&mut || {
                panic!("copy aborted");
            })
            .unwrap();
        });
        assert!(panic.is_err());
        // Reenter the already compiled block: a still-writable MAP_JIT thread
        // cannot execute it. Avoid private pthread state-query APIs.
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rax), 7);
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
            Ok(rw)
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
        let mut builder = crate::libcalls::builder(isa);
        let observed = Arc::new(Observed::default());
        builder.memory_provider(Box::new(TestMemory {
            memory: CodeMemory::new(Arc::new(CodeCounters::default())),
            observed: observed.clone(),
            aliases,
        }));
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
