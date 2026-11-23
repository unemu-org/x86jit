//! Host virtual-address reservation for §4.1. No guest pointer is a Rust slice.

/// Failure to reserve, commit, or release sparse guest backing.
#[derive(Debug)]
pub enum SparseError {
    UnsupportedHost,
    InvalidSpan,
    UnsupportedModel,
    Allocation,
    Reserve(std::io::Error),
    Commit(std::io::Error),
    Decommit(std::io::Error),
}

impl std::fmt::Display for SparseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedHost => f.write_str("sparse backing is unsupported on this host"),
            Self::InvalidSpan => {
                f.write_str("sparse span must fit the host and lower canonical half")
            }
            Self::UnsupportedModel => {
                f.write_str("sparse backing requires Flat or Reserved memory")
            }
            Self::Allocation => f.write_str("sparse memory metadata allocation failed"),
            Self::Reserve(e) => write!(f, "sparse reservation failed: {e}"),
            Self::Commit(e) => write!(f, "sparse commit failed: {e}"),
            Self::Decommit(e) => write!(f, "sparse decommit failed: {e}"),
        }
    }
}
impl std::error::Error for SparseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Reserve(e) | Self::Commit(e) | Self::Decommit(e) => Some(e),
            _ => None,
        }
    }
}

pub(super) struct Reservation {
    pub ptr: *mut u8,
    pub len: usize,
    pub page: usize,
}

impl Reservation {
    pub fn new(span: u64) -> Result<Self, SparseError> {
        let page = super::host_page_size();
        let len = usize::try_from(span).map_err(|_| SparseError::InvalidSpan)?;
        if len == 0 || span > 1 << 47 || len > isize::MAX as usize - page {
            return Err(SparseError::InvalidSpan);
        }
        let len = len.div_ceil(page) * page;
        let ptr = host::reserve(len)?;
        Ok(Self { ptr, len, page })
    }

    pub fn commit(&self, start: usize, len: usize) -> Result<(), SparseError> {
        #[cfg(test)]
        if FAIL_COMMIT.with(|fail| fail.replace(false)) {
            return Err(SparseError::Commit(std::io::Error::other(
                "injected commit failure",
            )));
        }
        if len == 0 {
            return Ok(());
        }
        let lo = start / self.page * self.page;
        let hi = (start + len).div_ceil(self.page) * self.page;
        // SAFETY: Memory::map bounds-checks the range before committing it.
        host::commit(unsafe { self.ptr.add(lo) }, hi - lo)
    }

    pub fn decommit(&self, start: usize, len: usize) -> Result<(), SparseError> {
        #[cfg(test)]
        if FAIL_DECOMMIT.with(|fail| fail.replace(false)) {
            return Err(SparseError::Decommit(std::io::Error::other(
                "injected decommit failure",
            )));
        }
        // SAFETY: the caller supplies whole, unshared pages within the reservation.
        host::decommit(unsafe { self.ptr.add(start) }, len)
    }
}

#[cfg(test)]
std::thread_local! {
    static FAIL_COMMIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FAIL_DECOMMIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(all(
    test,
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
mod tests {
    use super::*;
    use crate::{MapError, Memory, MemoryModel, Prot, RegionKind};

    #[test]
    fn commit_failure_does_not_publish_a_region_and_can_be_retried() {
        let mut memory = Memory::sparse(MemoryModel::Reserved { span: 0x10000 }).unwrap();
        FAIL_COMMIT.with(|fail| fail.set(true));
        assert!(matches!(
            memory.map(0, 4096, Prot::RW, RegionKind::Ram),
            Err(MapError::Sparse(SparseError::Commit(_)))
        ));
        assert!(matches!(memory.read(0, 1), Err(crate::MemTrap::Unmapped)));
        memory.map(0, 4096, Prot::RW, RegionKind::Ram).unwrap();
        assert_eq!(memory.read(0, 8).unwrap(), 0);
    }

    #[test]
    fn decommit_failure_removes_the_tag_and_invalidates_cached_code() {
        use crate::{Exit, Vm, VmConfig};
        let mut vm = Vm::sparse(VmConfig::reserved(0x10000)).unwrap();
        vm.map(0, 4096, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0, &[0xf4]).unwrap();
        let mut cpu = vm.new_vcpu();
        assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
        FAIL_DECOMMIT.with(|fail| fail.set(true));
        assert!(matches!(
            vm.unmap(0, 4096),
            Err(MapError::Sparse(SparseError::Decommit(_)))
        ));
        cpu.cpu.rip = 0;
        assert!(matches!(
            cpu.run(&vm, Some(10)),
            Exit::UnmappedMemory { .. }
        ));
        vm.map(0, 4096, Prot::RWX, RegionKind::Ram).unwrap();
        assert_eq!(vm.mem.read(0, 8).unwrap(), 0);
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        host::release(self.ptr, self.len);
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod host {
    use super::*;
    use std::ffi::{c_int, c_void};

    const PRIVATE: c_int = 2;
    #[cfg(target_os = "linux")]
    const ANON_NORESERVE: c_int = 0x20 | 0x4000;
    #[cfg(target_os = "macos")]
    const ANON_NORESERVE: c_int = 0x1000 | 0x40;

    unsafe extern "C" {
        fn mmap(
            p: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            off: i64,
        ) -> *mut c_void;
        fn mprotect(p: *mut c_void, len: usize, prot: c_int) -> c_int;
        fn madvise(p: *mut c_void, len: usize, advice: c_int) -> c_int;
        fn munmap(p: *mut c_void, len: usize) -> c_int;
    }

    pub fn reserve(len: usize) -> Result<*mut u8, SparseError> {
        // SAFETY: anonymous PROT_NONE mapping; no existing mapping is replaced.
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                0,
                PRIVATE | ANON_NORESERVE,
                -1,
                0,
            )
        };
        if ptr as isize == -1 {
            Err(SparseError::Reserve(std::io::Error::last_os_error()))
        } else if ptr.is_null() {
            // Rust guest transfers require non-null pointers, even on a host
            // that could otherwise map address zero successfully.
            unsafe { munmap(ptr, len) };
            Err(SparseError::Reserve(std::io::Error::other(
                "host returned a null reservation",
            )))
        } else {
            Ok(ptr.cast())
        }
    }
    pub fn commit(ptr: *mut u8, len: usize) -> Result<(), SparseError> {
        // SAFETY: page-aligned subrange of our anonymous reservation.
        if unsafe { mprotect(ptr.cast(), len, 3) } != 0 {
            return Err(SparseError::Commit(std::io::Error::last_os_error()));
        }
        #[cfg(target_os = "macos")]
        // SAFETY: reacquire pages previously marked MADV_FREE_REUSABLE; this
        // also accepts pages that have never been made reusable.
        if unsafe { madvise(ptr.cast(), len, 8) } != 0 {
            // MADV_FREE_REUSE
            return Err(SparseError::Commit(std::io::Error::last_os_error()));
        }
        Ok(())
    }
    pub fn decommit(ptr: *mut u8, len: usize) -> Result<(), SparseError> {
        // Keep the address reservation even if discard fails. Removing/replacing
        // it could let another host thread claim a hole after an allocation error.
        #[cfg(target_os = "linux")]
        let discard = 4; // MADV_DONTNEED
        #[cfg(target_os = "macos")]
        let discard = 7; // MADV_FREE_REUSABLE; Memory::map zeroes reused bytes.
                         // SAFETY: only untagged, unshared whole pages reach this call. Darwin's
                         // reusable-page operation requires writable pages, before PROT_NONE.
        if unsafe { madvise(ptr.cast(), len, discard) } != 0 {
            Err(SparseError::Decommit(std::io::Error::last_os_error()))
        } else {
            // SAFETY: retain the address reservation with inaccessible pages.
            if unsafe { mprotect(ptr.cast(), len, 0) } != 0 {
                Err(SparseError::Decommit(std::io::Error::last_os_error()))
            } else {
                Ok(())
            }
        }
    }
    pub fn release(ptr: *mut u8, len: usize) {
        // SAFETY: this reservation owns the complete range, including replaced pages.
        unsafe { munmap(ptr.cast(), len) };
    }
}

#[cfg(target_os = "windows")]
mod host {
    use super::*;
    use std::ffi::c_void;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn VirtualAlloc(
            p: *mut c_void,
            len: usize,
            allocation: u32,
            protection: u32,
        ) -> *mut c_void;
        fn VirtualFree(p: *mut c_void, len: usize, operation: u32) -> i32;
    }
    pub fn reserve(len: usize) -> Result<*mut u8, SparseError> {
        // SAFETY: reserve address space without commit, PAGE_NOACCESS.
        let ptr = unsafe { VirtualAlloc(std::ptr::null_mut(), len, 0x2000, 1) };
        if ptr.is_null() {
            Err(SparseError::Reserve(std::io::Error::last_os_error()))
        } else {
            Ok(ptr.cast())
        }
    }
    pub fn commit(ptr: *mut u8, len: usize) -> Result<(), SparseError> {
        // SAFETY: MEM_COMMIT / PAGE_READWRITE within the existing reservation.
        if unsafe { VirtualAlloc(ptr.cast(), len, 0x1000, 4) }.is_null() {
            Err(SparseError::Commit(std::io::Error::last_os_error()))
        } else {
            Ok(())
        }
    }
    pub fn decommit(ptr: *mut u8, len: usize) -> Result<(), SparseError> {
        // SAFETY: MEM_DECOMMIT retains address ownership and discards the bytes.
        if unsafe { VirtualFree(ptr.cast(), len, 0x4000) } == 0 {
            Err(SparseError::Decommit(std::io::Error::last_os_error()))
        } else {
            Ok(())
        }
    }
    pub fn release(ptr: *mut u8, _: usize) {
        // SAFETY: MEM_RELEASE requires the reservation base and a zero size.
        unsafe { VirtualFree(ptr.cast(), 0, 0x8000) };
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
mod host {
    use super::*;
    pub fn reserve(_: usize) -> Result<*mut u8, SparseError> {
        Err(SparseError::UnsupportedHost)
    }
    pub fn commit(_: *mut u8, _: usize) -> Result<(), SparseError> {
        Err(SparseError::UnsupportedHost)
    }
    pub fn decommit(_: *mut u8, _: usize) -> Result<(), SparseError> {
        Err(SparseError::UnsupportedHost)
    }
    pub fn release(_: *mut u8, _: usize) {}
}
