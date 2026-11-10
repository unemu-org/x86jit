use cranelift_module::{ModuleError, ModuleResult};
use std::io;

mod arena;
mod system;

pub use arena::ArenaMemoryProvider;
pub use system::SystemMemoryProvider;

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
    // Clear all the newly allocated code from cache if the processor requires it
    //
    // Do this before marking the memory as R+X, technically we should be able to do it after
    // but there are some CPU's that have had errata about doing this with read only memory.
    unsafe {
        wasmtime_jit_icache_coherence::clear_cache(ptr as *const libc::c_void, len)
            .expect("Failed cache clear")
    };

    unsafe {
        region::protect(ptr, len, region::Protection::READ_EXECUTE).map_err(|e| {
            ModuleError::Backend(
                anyhow::Error::new(e).context("unable to make memory readable+executable"),
            )
        })?;
    }

    // If BTI is requested, and the architecture supports it, use mprotect to set the PROT_BTI flag.
    if branch_protection == BranchProtection::BTI {
        #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
        if std::arch::is_aarch64_feature_detected!("bti") {
            let prot = libc::PROT_EXEC | libc::PROT_READ | /* PROT_BTI */ 0x10;

            unsafe {
                if libc::mprotect(ptr as *mut libc::c_void, len, prot) < 0 {
                    return Err(ModuleError::Backend(
                        anyhow::Error::new(io::Error::last_os_error())
                            .context("unable to make memory readable+executable"),
                    ));
                }
            }
        }
    }

    Ok(())
}
