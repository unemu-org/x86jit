//! Cold fallback for read-denied crossing-store probes (§8.1, decision-13).
//!
//! VM metadata does not prove an access will succeed (guard regions, backing-file
//! SIGBUS and thread-local protections can still fault). Only a failed read probe
//! uses it. Contiguous copies transfer the first unproven page before proven
//! pages. Additional unproven pages, and preflight-only callers that cannot
//! reorder the real transfer, retain a partial-commit risk on later write faults.
//! The embedder must keep backing and protections stable through commit.

#[cfg(test)]
std::thread_local! {
    pub(crate) static FAIL_QUERY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(crate) static QUERIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cold]
#[inline(never)]
pub(crate) fn writable(at: usize) -> bool {
    // Host permission queries can allocate; store probes may run under guest control.
    let _fp = crate::jit_abi::fp::DefaultEnvironment::enter();
    #[cfg(test)]
    {
        QUERIES.with(|n| n.set(n.get() + 1));
        if FAIL_QUERY.with(std::cell::Cell::get) {
            return false;
        }
    }
    writable_region_end(at).is_some_and(|end| end > at)
}

#[cfg(target_os = "macos")]
fn writable_region_end(at: usize) -> Option<usize> {
    // Darwin SDK mach/vm_region.h: VM_REGION_BASIC_INFO_64 = 9, nine
    // 32-bit words, with current protection first. mach/mach_vm.defs: region.
    unsafe extern "C" {
        static mach_task_self_: u32;
        fn mach_vm_region(
            task: u32,
            address: *mut u64,
            size: *mut u64,
            flavor: i32,
            info: *mut i32,
            count: *mut u32,
            object: *mut u32,
        ) -> i32;
        fn mach_port_deallocate(task: u32, name: u32) -> i32;
    }
    let mut address = at as u64;
    let mut size = 0;
    let mut info = [0i32; 9];
    let mut count = info.len() as u32;
    let mut object = 0;
    // SAFETY: all out-pointers cover the declared Mach messages; no guest
    // pointer is dereferenced or passed as an output buffer.
    let result = unsafe {
        let task = mach_task_self_;
        let result = mach_vm_region(
            task,
            &mut address,
            &mut size,
            9,
            info.as_mut_ptr(),
            &mut count,
            &mut object,
        );
        if object != 0 {
            mach_port_deallocate(task, object);
        }
        result
    };
    // VM_PROT_WRITE = 2 (mach/vm_prot.h). region can return the next mapping.
    (result == 0 && count == 9 && address <= at as u64 && info[0] & 2 != 0)
        .then(|| address.checked_add(size).map(|end| end as usize))
        .flatten()
}

#[cfg(all(target_os = "linux", feature = "environment"))]
fn writable_region_end(at: usize) -> Option<usize> {
    crate::environment::host_writable_region_end(at)
}

// Without filesystem access, preserve the failed probe instead of guessing.
#[cfg(all(target_os = "linux", not(feature = "environment")))]
fn writable_region_end(_: usize) -> Option<usize> {
    None
}

#[cfg(target_os = "freebsd")]
fn writable_region_end(at: usize) -> Option<usize> {
    // FreeBSD 14.3 sys/sys/sysctl.h: CTL_KERN/KERN_PROC/KERN_PROC_VMMAP.
    // sys/sys/user.h: variable-length kinfo_vmentry records; the fixed prefix
    // has start/end at bytes 8/16 and kve_protection at byte 56 (WRITE = 2).
    unsafe extern "C" {
        fn getpid() -> i32;
        fn sysctl(
            name: *const i32,
            n: u32,
            out: *mut u8,
            len: *mut usize,
            new: *const u8,
            newlen: usize,
        ) -> i32;
    }
    // SAFETY: getpid has no pointer arguments.
    let mib = [1, 14, 32, unsafe { getpid() }];
    let mut len = 0;
    // SAFETY: size query, with a live length output and no new value.
    if unsafe {
        sysctl(
            mib.as_ptr(),
            4,
            std::ptr::null_mut(),
            &mut len,
            std::ptr::null(),
            0,
        )
    } != 0
    {
        return None;
    }
    // Leave room for maps created while allocating the output buffer.
    let mut data = vec![0; len.checked_add(len / 3)?.checked_add(4096)?];
    len = data.len();
    // SAFETY: the output buffer has the supplied capacity; the kernel bounds copyout.
    if unsafe {
        sysctl(
            mib.as_ptr(),
            4,
            data.as_mut_ptr(),
            &mut len,
            std::ptr::null(),
            0,
        )
    } != 0
    {
        return None;
    }
    let mut records = data.get(..len)?;
    while records.len() >= 60 {
        let size = u32::from_ne_bytes(records[..4].try_into().ok()?) as usize;
        let record = records.get(..size)?;
        if size < 60 {
            return None;
        }
        let start = u64::from_ne_bytes(record[8..16].try_into().ok()?) as usize;
        let end = u64::from_ne_bytes(record[16..24].try_into().ok()?) as usize;
        let protection = u32::from_ne_bytes(record[56..60].try_into().ok()?);
        if start <= at && at < end {
            return (protection & 2 != 0).then_some(end);
        }
        records = &records[size..];
    }
    None
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "freebsd")))]
fn writable_region_end(_: usize) -> Option<usize> {
    None
}
