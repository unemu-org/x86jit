use cranelift_module::{ModuleError, ModuleResult};
use std::io;

mod arena;
mod system;

pub use arena::ArenaMemoryProvider;
pub use system::SystemMemoryProvider;

/// A fallible system-provider publication operation.
#[derive(Clone, Copy, Debug)]
pub enum SystemMemoryOperation {
    /// Remove write permission from constant data.
    ProtectReadOnly,
    /// Seal code RX, including any requested branch protection.
    ProtectExecutable,
    /// Clean or invalidate newly written instruction-cache lines.
    CacheMaintenance,
    /// Synchronize instruction streams before publishing code.
    PipelineSync,
}

type Perform<'a> =
    dyn FnMut(SystemMemoryOperation, &mut dyn FnMut() -> io::Result<()>) -> io::Result<()> + 'a;

fn protection_error(error: region::Error) -> io::Error {
    match error {
        region::Error::SystemCall(error) => error,
        error => io::Error::other(error),
    }
}

/// Type of branch protection to apply to executable memory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BranchProtection {
    /// No protection.
    None,
    /// Use the Branch Target Identification extension of the Arm architecture.
    BTI,
}

/// The kind of memory allocation requested by a [`JITMemoryProvider`].
pub enum JITMemoryKind {
    /// Allocate memory that will be executable once finalized.
    Executable,
    /// Allocate writable memory.
    Writable,
    /// Allocate memory that will be read-only once finalized.
    ReadOnly,
}

/// A provider of memory for the JIT.
pub trait JITMemoryProvider {
    /// Allocate memory
    fn allocate(&mut self, size: usize, align: u64, kind: JITMemoryKind) -> io::Result<*mut u8>;

    /// Translate an allocation's writable address to its final runtime address.
    /// The translation must remain stable for the allocation's lifetime.
    fn executable_ptr(&self, writable: *mut u8) -> *const u8 {
        writable
    }

    /// Identity of the allocation starting at this executable address, when
    /// provided by an allocator with per-allocation lifetime tracking. The ID
    /// must remain stable through finalization and never identify another extent.
    fn allocation_generation(&self, _executable: *const u8) -> Option<u64> {
        None
    }

    /// Release an exact allocation after unlinking it and completing execution
    /// and fault-reader grace periods. Return false for an unsupported provider
    /// or a stale address/generation pair. The default retains storage.
    ///
    /// # Safety
    /// No execution, reader, compiler record or relocation may still use it.
    unsafe fn release_allocation(&mut self, _executable: *const u8, _generation: u64) -> bool {
        false
    }

    /// Perform copying/relocation with thread-local JIT write access enabled.
    /// Implementations must restore the caller's protection even on unwind.
    fn with_write_access(&self, action: &mut dyn FnMut()) -> io::Result<()> {
        action();
        Ok(())
    }

    /// Whether this provider/integration supports the requested relocation.
    fn supports_relocation(&self, _kind: cranelift_codegen::binemit::Reloc) -> bool {
        true
    }

    /// Free the memory region.
    unsafe fn free_memory(&mut self);
    /// Finalize the memory region and apply memory protections.
    fn finalize(&mut self, branch_protection: BranchProtection) -> ModuleResult<()>;
}

/// Marks the memory region as readable and executable.
///
/// This function deals with applies branch protection and clears the icache,
/// but *doesn't* flush the pipeline. Callers have to ensure that
/// [`wasmtime_jit_icache_coherence::pipeline_flush_mt`] is called before the
/// mappings are used.
pub(crate) fn set_readable_and_executable(
    ptr: *mut u8,
    len: usize,
    branch_protection: BranchProtection,
) -> ModuleResult<()> {
    set_readable_and_executable_with(ptr, len, branch_protection, &mut |_, action| action())
        .map_err(|err| ModuleError::Allocation { err })
}

fn set_readable_and_executable_with(
    ptr: *mut u8,
    len: usize,
    branch_protection: BranchProtection,
    perform: &mut Perform<'_>,
) -> io::Result<()> {
    // Keep cache maintenance before RX protection: some CPUs have errata
    // requiring writable memory for this operation.
    perform(SystemMemoryOperation::CacheMaintenance, &mut || {
        // SAFETY: the provider owns this initialized code extent.
        let result = unsafe { wasmtime_jit_icache_coherence::clear_cache(ptr.cast(), len) };
        #[cfg(target_os = "linux")]
        {
            result
        }
        #[cfg(not(target_os = "linux"))]
        {
            result.map_err(io::Error::other)
        }
    })?;
    perform(SystemMemoryOperation::ProtectExecutable, &mut || {
        // SAFETY: this extent belongs to the provider and is page aligned.
        unsafe { region::protect(ptr, len, region::Protection::READ_EXECUTE) }
            .map_err(protection_error)?;
        if branch_protection == BranchProtection::BTI {
            #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
            if std::arch::is_aarch64_feature_detected!("bti") {
                let prot = libc::PROT_EXEC | libc::PROT_READ | /* PROT_BTI */ 0x10;
                // SAFETY: the same owned, page-aligned extent; add BTI protection.
                if unsafe { libc::mprotect(ptr.cast(), len, prot) } < 0 {
                    return Err(io::Error::last_os_error());
                }
            }
        }
        Ok(())
    })
}
