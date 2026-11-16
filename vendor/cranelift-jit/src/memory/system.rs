// Modified by the x86jit project. This file is changed from cranelift-jit 0.135.8
// (crates.io, sha256 31d67d6c6a02ffd860599e55b03d4f37c6cf9be678c1d68dbfa6d1f61946e4ae;
// wasmtime commit 7b6c29ec3e9a49c97da231665cbd8eead9ccbda0, path cranelift/jit),
// licensed Apache-2.0 WITH LLVM-exception (LICENSE in this directory).
// Changed: `SystemMemoryProvider::finalize_with`, a callback around each
// protection, cache-maintenance and pipeline-sync operation; those failures are
// returned as errors instead of panicking.
// README.md in this directory lists every changed file.

use std::io;
use std::mem;
use std::ptr;

use cranelift_module::{ModuleError, ModuleResult};
use memmap2::MmapMut;

use super::{BranchProtection, JITMemoryKind, JITMemoryProvider};

/// A simple struct consisting of a pointer and length.
struct PtrLen {
    map: Option<MmapMut>,
    ptr: *mut u8,
    len: usize,
}

impl PtrLen {
    /// Create a new empty `PtrLen`.
    fn new() -> Self {
        Self {
            map: None,
            ptr: ptr::null_mut(),
            len: 0,
        }
    }

    /// Create a new `PtrLen` pointing to at least `size` bytes of memory,
    /// suitably sized and aligned for memory protection.
    fn with_size(size: usize) -> io::Result<Self> {
        let alloc_size = region::page::ceil(size as *const ()) as usize;
        MmapMut::map_anon(alloc_size).map(|mut mmap| {
            // The order here is important; we assign the pointer first to get
            // around compile time borrow errors.
            Self {
                ptr: mmap.as_mut_ptr(),
                map: Some(mmap),
                len: alloc_size,
            }
        })
    }
}

/// JIT memory manager. This manages pages of suitably aligned and
/// accessible memory. Memory will be leaked by default to have
/// function pointers remain valid for the remainder of the
/// program's life.
pub(crate) struct Memory {
    allocations: Vec<PtrLen>,
    already_protected: usize,
    current: PtrLen,
    position: usize,
}

unsafe impl Send for Memory {}

impl Memory {
    pub(crate) fn new() -> Self {
        Self {
            allocations: Vec::new(),
            already_protected: 0,
            current: PtrLen::new(),
            position: 0,
        }
    }

    fn finish_current(&mut self) {
        self.allocations
            .push(mem::replace(&mut self.current, PtrLen::new()));
        self.position = 0;
    }

    pub(crate) fn allocate(&mut self, size: usize, align: u64) -> io::Result<*mut u8> {
        let align = usize::try_from(align).expect("alignment too big");
        if self.position % align != 0 {
            self.position += align - self.position % align;
            debug_assert!(self.position % align == 0);
        }

        if size <= self.current.len - self.position {
            // TODO: Ensure overflow is not possible.
            let ptr = unsafe { self.current.ptr.add(self.position) };
            self.position += size;
            return Ok(ptr);
        }

        self.finish_current();

        // TODO: Allocate more at a time.
        self.current = PtrLen::with_size(size)?;
        self.position = size;

        Ok(self.current.ptr)
    }

    /// Set all memory allocated in this `Memory` up to now as readable and executable.
    pub(crate) fn set_readable_and_executable(
        &mut self,
        branch_protection: BranchProtection,
        perform: &mut super::Perform<'_>,
    ) -> io::Result<()> {
        self.finish_current();

        for &PtrLen { ptr, len, .. } in self.non_protected_allocations_iter() {
            super::set_readable_and_executable_with(ptr, len, branch_protection, perform)?;
        }

        // Flush any in-flight instructions from the pipeline
        perform(super::SystemMemoryOperation::PipelineSync, &mut || {
            let result = wasmtime_jit_icache_coherence::pipeline_flush_mt();
            #[cfg(target_os = "linux")]
            {
                result
            }
            #[cfg(not(target_os = "linux"))]
            {
                result.map_err(io::Error::other)
            }
        })?;

        self.already_protected = self.allocations.len();
        Ok(())
    }

    /// Set all memory allocated in this `Memory` up to now as readonly.
    pub(crate) fn set_readonly(&mut self, perform: &mut super::Perform<'_>) -> io::Result<()> {
        self.finish_current();

        for &PtrLen { ptr, len, .. } in self.non_protected_allocations_iter() {
            perform(super::SystemMemoryOperation::ProtectReadOnly, &mut || {
                // SAFETY: the provider owns this page-aligned extent.
                unsafe { region::protect(ptr, len, region::Protection::READ) }
                    .map_err(super::protection_error)
            })?;
        }

        self.already_protected = self.allocations.len();
        Ok(())
    }

    /// Iterates non protected memory allocations that are of not zero bytes in size.
    fn non_protected_allocations_iter(&self) -> impl Iterator<Item = &PtrLen> {
        self.allocations[self.already_protected..]
            .iter()
            .filter(|&PtrLen { map, len, .. }| *len != 0 && map.is_some())
    }

    /// Frees all allocated memory regions that would be leaked otherwise.
    /// Likely to invalidate existing function pointers, causing unsafety.
    pub(crate) unsafe fn free_memory(&mut self) {
        self.allocations.clear();
        self.already_protected = 0;
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        // leak memory to guarantee validity of function pointers
        mem::replace(&mut self.allocations, Vec::new())
            .into_iter()
            .for_each(mem::forget);
    }
}

/// A memory provider that allocates memory on-demand using the system
/// allocator.
///
/// Note: Memory will be leaked by default unless
/// [`JITMemoryProvider::free_memory`] is called to ensure function pointers
/// remain valid for the remainder of the program's life.
pub struct SystemMemoryProvider {
    code: Memory,
    readonly: Memory,
    writable: Memory,
}

impl SystemMemoryProvider {
    /// Create a new memory handle with the given branch protection.
    pub fn new() -> Self {
        Self {
            code: Memory::new(),
            readonly: Memory::new(),
            writable: Memory::new(),
        }
    }

    /// Finalize while allowing an embedder to observe and diagnose each OS operation.
    /// The callback must execute the supplied action or return an error.
    pub fn finalize_with(
        &mut self,
        branch_protection: BranchProtection,
        mut perform: impl FnMut(
            super::SystemMemoryOperation,
            &mut dyn FnMut() -> io::Result<()>,
        ) -> io::Result<()>,
    ) -> ModuleResult<()> {
        self.readonly
            .set_readonly(&mut perform)
            .and_then(|()| {
                self.code
                    .set_readable_and_executable(branch_protection, &mut perform)
            })
            .map_err(|err| ModuleError::Allocation { err })
    }
}

impl JITMemoryProvider for SystemMemoryProvider {
    unsafe fn free_memory(&mut self) {
        self.code.free_memory();
        self.readonly.free_memory();
        self.writable.free_memory();
    }

    fn finalize(&mut self, branch_protection: BranchProtection) -> ModuleResult<()> {
        self.finalize_with(branch_protection, |_, action| action())
    }

    fn allocate(&mut self, size: usize, align: u64, kind: JITMemoryKind) -> io::Result<*mut u8> {
        match kind {
            JITMemoryKind::Executable => self.code.allocate(size, align),
            JITMemoryKind::Writable => self.writable.allocate(size, align),
            JITMemoryKind::ReadOnly => self.readonly.allocate(size, align),
        }
    }
}
