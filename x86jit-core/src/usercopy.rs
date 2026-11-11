//! Host transfer fixups (§8.1, decision-13). Routines return through their ordinary
//! ABI frame; the classifier only reads immutable relative range descriptors.
//! Plain scalar atomicity: Intel SDM Vol 3A §11.1.1; LOCK atomicity: §11.1.2.2.
//! Locked ordering: Intel SDM Vol 3A §11.2.3.9.
//! Host x86 transfers: SDM Vol 3A §11.1.1. Host AArch64 atomicity and exclusive
//! pairs: Arm ARM DDI0487J.a §B2.2.1; fault address: DDI0487I.a
//! §D17.2.40 (FAR_EL1), corrected by 102105_I.a_06 §2.85.
//! AArch64 entries include BTI c for indirect dispatch; the encoding is witnessed.
//! Guarded-text semantics remain source/runtime-unverified (decision-13).
//! Metadata follows BTI and its branch.
//! Host ordering: Arm ARM DDI0487I.a §B2.3.11 (Memory Barriers).
//! Cross-page transfers retry bytes after a wide fault, avoiding reliance on the
//! wide access's reported FAR being the first unavailable byte.
#[repr(C)]
struct Status {
    failed: usize,
    value: usize,
}
#[cfg(target_arch = "aarch64")]
type Routine = unsafe extern "C" fn(*mut u8, usize, usize, usize) -> Status;
#[cfg(target_arch = "x86_64")]
type Routine = unsafe extern "sysv64" fn(*mut u8, usize, usize, usize) -> Status;
// Preserve default-mode builds on other hosts; these have no recoverable routines.
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
type Routine = unsafe extern "C" fn(*mut u8, usize, usize, usize) -> Status;
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
macro_rules! unavailable_routines {
    ($($name:ident),*) => {$(
        unsafe extern "C" fn $name(ptr:*mut u8,_:usize,_:usize,_:usize)->Status {
            Status {failed:1,value:ptr as usize}
        }
    )*};
}
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
unavailable_routines!(
    copy_read, copy_write, probe_read, probe_rmw, load1, load2, load4, load8, store1, store2,
    store4, store8, cas1, cas2, cas4, cas8, pair
);
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
static FIXUP_TABLE: [Routine; 17] = [
    copy_read, copy_write, probe_read, probe_rmw, load1, load2, load4, load8, store1, store2,
    store4, store8, cas1, cas2, cas4, cas8, pair,
];
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(crate) fn lookup(pc: usize) -> Option<usize> {
    for routine in FIXUP_TABLE {
        let entry = routine as *const () as usize;
        #[cfg(target_arch = "aarch64")]
        let header = entry + 8;
        #[cfg(target_arch = "x86_64")]
        let header = entry + 2;
        // SAFETY: six immutable words in this routine's linked instruction image.
        let words = unsafe { std::ptr::read_unaligned(header as *const [i32; 6]) };
        let at = |i: usize| header.wrapping_add_signed(words[i] as isize);
        if (at(0)..at(1)).contains(&pc) {
            return Some(at(2));
        }
        if (at(3)..at(4)).contains(&pc) {
            return Some(at(5));
        }
    }
    None
}
unsafe fn invoke(
    routine: Routine,
    ptr: *mut u8,
    data: usize,
    len: usize,
    extra: usize,
) -> Result<usize, usize> {
    #[cfg(any(test, feature = "test-hooks"))]
    if deny_test_read(routine, ptr as usize) {
        return Err(ptr as usize);
    }
    #[cfg(any(test, feature = "test-hooks"))]
    if routine as usize == copy_write as *const () as usize {
        if let Some(at) = WRITE_FAULT.with(std::cell::Cell::get) {
            if (ptr as usize..ptr as usize + len).contains(&at) {
                // SAFETY: the injected fault is inside the validated copy; commit
                // its prefix through the real transfer routine before failing.
                let result = unsafe { routine(ptr, data, at - ptr as usize, extra) };
                assert_eq!(result.failed, 0);
                return Err(at);
            }
        }
    }
    // SAFETY: callers validate the guest span and keep local buffers live.
    let result = unsafe { routine(ptr, data, len, extra) };
    if result.failed == 0 {
        Ok(result.value)
    } else {
        Err(result.value)
    }
}
pub(crate) unsafe fn copy_in(ptr: *const u8, buf: &mut [u8]) -> Result<(), usize> {
    // SAFETY: the output slice is valid and the caller validates guest memory.
    unsafe { copy_in_raw(ptr, buf.as_mut_ptr(), buf.len()) }
}
/// Copy into storage that need not be initialized; only a successful prefix is readable.
pub(crate) unsafe fn copy_in_raw(ptr: *const u8, out: *mut u8, len: usize) -> Result<(), usize> {
    // SAFETY: callers keep the writable output allocation and guest span live.
    unsafe { invoke(copy_read, ptr.cast_mut(), out as usize, len, 0).map(|_| ()) }
}
#[inline(always)]
pub(crate) unsafe fn probe(ptr: *mut u8, len: usize, write: bool) -> Result<(), usize> {
    let page = crate::memory::host_page_size();
    if !write {
        // SAFETY: caller validates the guest range.
        return unsafe { invoke(probe_read, ptr, 0, len, page).map(|_| ()) };
    }
    // SAFETY: caller validates the entire span.
    let result = unsafe { probe_write_span(ptr, len, page) };
    if result.failed == 0 {
        return Ok(());
    }
    // SAFETY: preserve the validated span and the original probe's failure.
    unsafe { resume_write_probe(ptr, len, result) }
}

#[inline(always)]
unsafe fn probe_write_span(ptr: *mut u8, len: usize, page: usize) -> Status {
    #[cfg(any(test, feature = "test-hooks"))]
    // SAFETY: the test wrapper preserves the routine's validated-span contract.
    unsafe {
        test_probe_rmw(ptr, len, page)
    }
    #[cfg(not(any(test, feature = "test-hooks")))]
    // SAFETY: caller validates the span; the routine probes each touched page once.
    unsafe {
        probe_rmw(ptr, 0, len, page)
    }
}

// Keep fallback state out of successful transfers' register saves and stack frames.
#[cold]
#[inline(never)]
unsafe fn resume_write_probe(
    mut at: *mut u8,
    mut remaining: usize,
    mut result: Status,
) -> Result<(), usize> {
    let page = crate::memory::host_page_size();
    loop {
        if result.failed != 2 || !crate::host_write::writable(result.value) {
            return Err(result.value);
        }
        // The failed read is covered by the cold permission fallback. Resume
        // at the following page without repeating any successful RMW probes.
        let consumed = result.value - at as usize + page - (result.value & (page - 1));
        if consumed >= remaining {
            return Ok(());
        }
        at = at.wrapping_add(consumed);
        remaining -= consumed;
        // SAFETY: the remaining suffix is inside the original validated span.
        result = unsafe { probe_write_span(at, remaining, page) };
        if result.failed == 0 {
            return Ok(());
        }
    }
}

/// Progress assumes backing and protections stay stable through commit (§8.1).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct WriteFault {
    pub(crate) at: usize,
    pub(crate) committed: usize,
}

pub(crate) unsafe fn copy_out(ptr: *mut u8, buf: &[u8]) -> Result<(), WriteFault> {
    let page = crate::memory::host_page_size();
    // Preflight all touched pages if more than one is touched. A single-page
    // store faults on its first write before committing bytes. Protection revocation
    // during the subsequent copy still requires embedder synchronization.
    // SAFETY: caller validates the range and the source remains live.
    unsafe {
        if buf.len() > page - ptr as usize % page {
            let result = probe_write_span(ptr, buf.len(), page);
            if result.failed != 0 {
                return copy_out_after_probe_fault(ptr, buf, result);
            }
        }
        invoke(copy_write, ptr, buf.as_ptr() as usize, buf.len(), 0)
            .map(|_| ())
            .map_err(|at| WriteFault {
                at,
                committed: at - ptr as usize,
            })
    }
}

#[cold]
#[inline(never)]
unsafe fn copy_out_after_probe_fault(
    ptr: *mut u8,
    buf: &[u8],
    result: Status,
) -> Result<(), WriteFault> {
    let unproven = result.value;
    // SAFETY: the original probe validated the prefix; resume validates the suffix.
    unsafe { resume_write_probe(ptr, buf.len(), result) }
        .map_err(|at| WriteFault { at, committed: 0 })?;
    let start = unproven - ptr as usize;
    let page = crate::memory::host_page_size();
    let end = buf.len().min(start + page - (unproven & (page - 1)));
    // A permission query cannot prove write access (e.g. SIGBUS beyond EOF).
    // Transfer the first read-faulting page before committing any proven page.
    // With stable backing/protections, successful RMW probes prove the rest.
    // If multiple pages read-fault, the extra pages still rely on permission
    // queries: a later transfer fault there can leave the first unproven page
    // and intervening proven pages committed. Concurrent backing/protection
    // changes likewise remain the embedder's synchronization responsibility.
    // SAFETY: each transfer stays within the original validated destination;
    // the first transfer stays on one page and faults before committing bytes.
    unsafe {
        invoke(
            copy_write,
            ptr.add(start),
            buf[start..end].as_ptr() as usize,
            end - start,
            0,
        )
        .map_err(|at| WriteFault { at, committed: 0 })?;
        // Successful RMW probes prove this prefix under the stable-backing contract.
        invoke(copy_write, ptr, buf.as_ptr() as usize, start, 0).map_err(|at| WriteFault {
            at,
            committed: at - ptr as usize,
        })?;
        invoke(
            copy_write,
            ptr.add(end),
            buf[end..].as_ptr() as usize,
            buf.len() - end,
            0,
        )
        .map_err(|at| WriteFault {
            at,
            committed: at - ptr as usize,
        })?;
    }
    Ok(())
}
pub(crate) unsafe fn load(ptr: *const u8, size: u8) -> Result<u64, usize> {
    if !matches!(size, 1 | 2 | 4 | 8) || (ptr as usize & 4095) + size as usize > 4096 {
        let mut data = [0; 8];
        // SAFETY: local buffer is live; byte fallback refines a crossing wide fault.
        unsafe {
            copy_in(ptr, &mut data[..size as usize])?;
        }
        return Ok(u64::from_le_bytes(data));
    }
    let routine = match size {
        1 => load1,
        2 => load2,
        4 => load4,
        _ => load8,
    };
    // SAFETY: caller validates the naturally aligned or plain unaligned scalar range.
    unsafe { invoke(routine, ptr.cast_mut(), 0, 0, 0).map(|v| v as u64) }
}
pub(crate) unsafe fn store(ptr: *mut u8, value: u64, size: u8) -> Result<(), WriteFault> {
    // Every supported host protection granule is a multiple of 4096. The common
    // scalar path needs no page-size query; only a possible crossing does.
    if (ptr as usize & 4095) + size as usize > 4096 {
        // SAFETY: copy_out checks the actual page size and preflights only crossings.
        return unsafe { copy_out(ptr, &value.to_le_bytes()[..size as usize]) };
    }
    // SAFETY: the full span is on one host page and needs no preflight.
    unsafe { store_preflighted(ptr, value, size) }
}
/// The caller has already preflighted every touched host page.
#[inline]
pub(crate) unsafe fn store_preflighted(
    ptr: *mut u8,
    value: u64,
    size: u8,
) -> Result<(), WriteFault> {
    if !matches!(size, 1 | 2 | 4 | 8) || (ptr as usize & 4095) + size as usize > 4096 {
        let data = value.to_le_bytes();
        // SAFETY: caller validates the full span and has probed all touched pages.
        return unsafe {
            invoke(copy_write, ptr, data.as_ptr() as usize, size as usize, 0)
                .map(|_| ())
                .map_err(|at| WriteFault {
                    at,
                    committed: at - ptr as usize,
                })
        };
    }
    let routine = match size {
        1 => store1,
        2 => store2,
        4 => store4,
        _ => store8,
    };
    // SAFETY: caller validates the entire scalar span.
    unsafe { invoke(routine, ptr, value as usize, 0, 0).map(|_| ()) }
        .map_err(|at| WriteFault { at, committed: 0 })
}
pub(crate) unsafe fn cas(ptr: *mut u8, expected: u64, value: u64, size: u8) -> Result<u64, usize> {
    let mask = u64::MAX >> (64 - size * 8);
    let expected = expected & mask;
    if !(ptr as usize).is_multiple_of(size as usize) {
        // SAFETY: retains the existing non-atomic misaligned fallback; no unaligned exclusive.
        unsafe {
            invoke(
                probe_rmw,
                ptr,
                0,
                size as usize,
                crate::memory::host_page_size(),
            )?;
            let old = load(ptr, size)?;
            if old == expected {
                store(ptr, value, size).map_err(|fault| fault.at)?;
            }
            return Ok(old);
        }
    }
    let routine = match size {
        1 => cas1,
        2 => cas2,
        4 => cas4,
        _ => cas8,
    };
    // SAFETY: caller validates the aligned operand; expected is width-masked on both hosts.
    unsafe { invoke(routine, ptr, expected as usize, value as usize, 0).map(|v| v as u64) }
}
pub(crate) unsafe fn cas_pair(ptr: *mut u8, expected: u128, value: u128) -> Result<u128, usize> {
    #[cfg(target_arch = "x86_64")]
    assert!(
        std::is_x86_feature_detected!("cmpxchg16b"),
        "host requires CMPXCHG16B"
    );
    let mut data = [
        expected as u64,
        (expected >> 64) as u64,
        value as u64,
        (value >> 64) as u64,
    ];
    // SAFETY: caller validates sixteen aligned writable guest bytes; local data is live.
    unsafe {
        invoke(pair, ptr, data.as_mut_ptr() as usize, 0, 0)?;
    }
    Ok(u128::from(data[0]) | (u128::from(data[1]) << 64))
}
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn copy_read(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        "hint #34",
        "b 2f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 6f - 99b",
        ".long 7f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "mov x4, #0",
        "5:",
        "cmp x2, #8",
        "b.lo 6f",
        "ldr x5, [x0, x4]",
        "str x5, [x1, x4]",
        "add x4, x4, #8",
        "sub x2, x2, #8",
        "b 5b",
        "4:",
        "mov x0,#0",
        "mov x1,#0",
        "ret",
        "6:",
        "cbz x2,4b",
        "ldrb w5, [x0, x4]",
        "strb w5, [x1, x4]",
        "add x4,x4,#1",
        "sub x2,x2,#1",
        "b 6b",
        "7:",
        "b 4b",
        "9:",
        "b 6b",
        "8:",
        "mov x0,#1",
        "mov x1,x9",
        "ret",
    );
}
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn copy_write(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        "hint #34",
        "b 2f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 6f - 99b",
        ".long 7f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "mov x4, #0",
        "5:",
        "cmp x2, #8",
        "b.lo 6f",
        "ldr x5, [x1, x4]",
        "str x5, [x0, x4]",
        "add x4, x4, #8",
        "sub x2, x2, #8",
        "b 5b",
        "4:",
        "mov x0,#0",
        "mov x1,#0",
        "ret",
        "6:",
        "cbz x2,4b",
        "ldrb w5, [x1, x4]",
        "strb w5, [x0, x4]",
        "add x4,x4,#1",
        "sub x2,x2,#1",
        "b 6b",
        "7:",
        "b 4b",
        "9:",
        "b 6b",
        "8:",
        "mov x0,#1",
        "mov x1,x9",
        "ret",
    );
}
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn probe_read(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        "hint #34",
        "b 2f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "cbz x2,4f",
        "mov x4,#0",
        "5:",
        "add x6,x0,x4",
        "ldrb w5,[x6]",
        "sub x5,x3,#1",
        "and x5,x6,x5",
        "sub x5,x3,x5",
        "add x4,x4,x5",
        "cmp x4,x2",
        "b.lo 5b",
        "4:",
        "mov x0,#0",
        "mov x1,#0",
        "ret",
        "9:",
        "8:",
        "clrex",
        "mov x0,#1",
        "mov x1,x9",
        "ret",
    );
}
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn probe_rmw(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        "hint #34",
        "b 2f",
        "99:",
        ".long 6f - 99b",
        ".long 7f - 99b",
        ".long 9f - 99b",
        ".long 7f - 99b",
        ".long 8f - 99b",
        ".long 10f - 99b",
        "2:",
        "3:",
        "cbz x2,4f",
        "mov x4,#0",
        "5:",
        "add x6,x0,x4",
        "6:",
        "ldaxrb w5,[x6]",
        "7:",
        "stlxrb w7,w5,[x6]",
        "8:",
        "cbnz w7,6b",
        "sub x5,x3,#1",
        "and x5,x6,x5",
        "sub x5,x3,x5",
        "add x4,x4,x5",
        "cmp x4,x2",
        "b.lo 5b",
        "4:",
        "mov x0,#0",
        "mov x1,#0",
        "ret",
        "9:",
        "clrex",
        "mov x0,#2",
        "mov x1,x9",
        "ret",
        "10:",
        "clrex",
        "mov x0,#1",
        "mov x1,x9",
        "ret",
    );
}
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn load1(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        "hint #34",
        "b 2f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "ldrb w5,[x0]",
        "4:",
        "mov x0,#0",
        "mov x1,x5",
        "ret",
        "9:",
        "8:",
        "clrex",
        "mov x0,#1",
        "mov x1,x9",
        "ret",
    );
}
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn load2(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        "hint #34",
        "b 2f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "ldrh w5,[x0]",
        "4:",
        "mov x0,#0",
        "mov x1,x5",
        "ret",
        "9:",
        "8:",
        "clrex",
        "mov x0,#1",
        "mov x1,x9",
        "ret",
    );
}
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn load4(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        "hint #34",
        "b 2f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "ldr w5,[x0]",
        "4:",
        "mov x0,#0",
        "mov x1,x5",
        "ret",
        "9:",
        "8:",
        "clrex",
        "mov x0,#1",
        "mov x1,x9",
        "ret",
    );
}
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn load8(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        "hint #34",
        "b 2f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "ldr x5,[x0]",
        "4:",
        "mov x0,#0",
        "mov x1,x5",
        "ret",
        "9:",
        "8:",
        "clrex",
        "mov x0,#1",
        "mov x1,x9",
        "ret",
    );
}
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn store1(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        "hint #34",
        "b 2f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "strb w1,[x0]",
        "4:",
        "mov x0,#0",
        "mov x1,#0",
        "ret",
        "9:",
        "8:",
        "clrex",
        "mov x0,#1",
        "mov x1,x9",
        "ret",
    );
}
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn store2(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        "hint #34",
        "b 2f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "strh w1,[x0]",
        "4:",
        "mov x0,#0",
        "mov x1,#0",
        "ret",
        "9:",
        "8:",
        "clrex",
        "mov x0,#1",
        "mov x1,x9",
        "ret",
    );
}
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn store4(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        "hint #34",
        "b 2f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "str w1,[x0]",
        "4:",
        "mov x0,#0",
        "mov x1,#0",
        "ret",
        "9:",
        "8:",
        "clrex",
        "mov x0,#1",
        "mov x1,x9",
        "ret",
    );
}
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn store8(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        "hint #34",
        "b 2f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "str x1,[x0]",
        "4:",
        "mov x0,#0",
        "mov x1,#0",
        "ret",
        "9:",
        "8:",
        "clrex",
        "mov x0,#1",
        "mov x1,x9",
        "ret",
    );
}
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn cas1(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        "hint #34",
        "b 2f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "dmb ish",
        "5:",
        "ldaxrb w5,[x0]",
        "cmp w5,w1",
        "csel w6,w2,w5,eq",
        "stlxrb w7,w6,[x0]",
        "cbnz w7,5b",
        "dmb ish",
        "4:",
        "mov x0,#0",
        "mov x1,x5",
        "ret",
        "9:",
        "8:",
        "clrex",
        "mov x0,#1",
        "mov x1,x9",
        "ret",
    );
}
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn cas2(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        "hint #34",
        "b 2f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "dmb ish",
        "5:",
        "ldaxrh w5,[x0]",
        "cmp w5,w1",
        "csel w6,w2,w5,eq",
        "stlxrh w7,w6,[x0]",
        "cbnz w7,5b",
        "dmb ish",
        "4:",
        "mov x0,#0",
        "mov x1,x5",
        "ret",
        "9:",
        "8:",
        "clrex",
        "mov x0,#1",
        "mov x1,x9",
        "ret",
    );
}
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn cas4(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        "hint #34",
        "b 2f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "dmb ish",
        "5:",
        "ldaxr w5,[x0]",
        "cmp w5,w1",
        "csel w6,w2,w5,eq",
        "stlxr w7,w6,[x0]",
        "cbnz w7,5b",
        "dmb ish",
        "4:",
        "mov x0,#0",
        "mov x1,x5",
        "ret",
        "9:",
        "8:",
        "clrex",
        "mov x0,#1",
        "mov x1,x9",
        "ret",
    );
}
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn cas8(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        "hint #34",
        "b 2f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "dmb ish",
        "5:",
        "ldaxr x5,[x0]",
        "cmp x5,x1",
        "csel x6,x2,x5,eq",
        "stlxr w7,x6,[x0]",
        "cbnz w7,5b",
        "dmb ish",
        "4:",
        "mov x0,#0",
        "mov x1,x5",
        "ret",
        "9:",
        "8:",
        "clrex",
        "mov x0,#1",
        "mov x1,x9",
        "ret",
    );
}
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn pair(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        "hint #34",
        "b 2f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "ldp x6,x7,[x1]",
        "ldp x10,x11,[x1,#16]",
        "dmb ish",
        "5:",
        "ldaxp x4,x5,[x0]",
        "cmp x4,x6",
        "ccmp x5,x7,#0,eq",
        "csel x12,x10,x4,eq",
        "csel x13,x11,x5,eq",
        "stlxp w14,x12,x13,[x0]",
        "cbnz w14,5b",
        "dmb ish",
        "stp x4,x5,[x1]",
        "4:",
        "mov x0,#0",
        "mov x1,#0",
        "ret",
        "9:",
        "8:",
        "clrex",
        "mov x0,#1",
        "mov x1,x9",
        "ret",
    );
}
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "sysv64" fn copy_read(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        ".byte 0xeb",
        ".byte 2f - 99f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 6f - 99b",
        ".long 7f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "xor r8d,r8d",
        "5:",
        "cmp rdx,8",
        "jb 6f",
        "mov rax,[rdi+r8]",
        "mov [rsi+r8],rax",
        "add r8,8",
        "sub rdx,8",
        "jmp 5b",
        "4:",
        "xor eax,eax",
        "xor edx,edx",
        "ret",
        "6:",
        "test rdx,rdx",
        "jz 4b",
        "mov al,[rdi+r8]",
        "mov [rsi+r8],al",
        "inc r8",
        "dec rdx",
        "jmp 6b",
        "7:",
        "jmp 4b",
        "9:",
        "jmp 6b",
        "8:",
        "mov eax,1",
        "mov rdx,r11",
        "ret",
    );
}
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "sysv64" fn copy_write(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        ".byte 0xeb",
        ".byte 2f - 99f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 6f - 99b",
        ".long 7f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "xor r8d,r8d",
        "5:",
        "cmp rdx,8",
        "jb 6f",
        "mov rax,[rsi+r8]",
        "mov [rdi+r8],rax",
        "add r8,8",
        "sub rdx,8",
        "jmp 5b",
        "4:",
        "xor eax,eax",
        "xor edx,edx",
        "ret",
        "6:",
        "test rdx,rdx",
        "jz 4b",
        "mov al,[rsi+r8]",
        "mov [rdi+r8],al",
        "inc r8",
        "dec rdx",
        "jmp 6b",
        "7:",
        "jmp 4b",
        "9:",
        "jmp 6b",
        "8:",
        "mov eax,1",
        "mov rdx,r11",
        "ret",
    );
}
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "sysv64" fn probe_read(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        ".byte 0xeb",
        ".byte 2f - 99f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "test rdx,rdx",
        "jz 4f",
        "xor r8d,r8d",
        "5:",
        "mov al,[rdi+r8]",
        "lea r9,[rdi+r8]",
        "mov r10,rcx",
        "dec r10",
        "and r9,r10",
        "mov r10,rcx",
        "sub r10,r9",
        "add r8,r10",
        "cmp r8,rdx",
        "jb 5b",
        "4:",
        "xor eax,eax",
        "xor edx,edx",
        "ret",
        "9:",
        "8:",
        "mov eax,1",
        "mov rdx,r11",
        "ret",
    );
}
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "sysv64" fn probe_rmw(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        ".byte 0xeb",
        ".byte 2f - 99f",
        "99:",
        ".long 5f - 99b",
        ".long 6f - 99b",
        ".long 9f - 99b",
        ".long 6f - 99b",
        ".long 7f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "test rdx,rdx",
        "jz 4f",
        "xor r8d,r8d",
        "5:",
        "mov al,[rdi+r8]",
        "6:",
        "lock cmpxchg byte ptr [rdi+r8],al",
        "7:",
        "jne 6b",
        "lea r9,[rdi+r8]",
        "mov r10,rcx",
        "dec r10",
        "and r9,r10",
        "mov r10,rcx",
        "sub r10,r9",
        "add r8,r10",
        "cmp r8,rdx",
        "jb 5b",
        "4:",
        "xor eax,eax",
        "xor edx,edx",
        "ret",
        "9:",
        "mov eax,2",
        "mov rdx,r11",
        "ret",
        "8:",
        "mov eax,1",
        "mov rdx,r11",
        "ret",
    );
}
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "sysv64" fn load1(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        ".byte 0xeb",
        ".byte 2f - 99f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "movzx eax,byte ptr [rdi]",
        "mov rdx,rax",
        "4:",
        "xor eax,eax",
        "ret",
        "9:",
        "8:",
        "mov eax,1",
        "mov rdx,r11",
        "ret",
    );
}
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "sysv64" fn load2(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        ".byte 0xeb",
        ".byte 2f - 99f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "movzx eax,word ptr [rdi]",
        "mov rdx,rax",
        "4:",
        "xor eax,eax",
        "ret",
        "9:",
        "8:",
        "mov eax,1",
        "mov rdx,r11",
        "ret",
    );
}
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "sysv64" fn load4(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        ".byte 0xeb",
        ".byte 2f - 99f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "mov eax,dword ptr [rdi]",
        "mov rdx,rax",
        "4:",
        "xor eax,eax",
        "ret",
        "9:",
        "8:",
        "mov eax,1",
        "mov rdx,r11",
        "ret",
    );
}
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "sysv64" fn load8(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        ".byte 0xeb",
        ".byte 2f - 99f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "mov rax,qword ptr [rdi]",
        "mov rdx,rax",
        "4:",
        "xor eax,eax",
        "ret",
        "9:",
        "8:",
        "mov eax,1",
        "mov rdx,r11",
        "ret",
    );
}
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "sysv64" fn store1(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        ".byte 0xeb",
        ".byte 2f - 99f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "mov [rdi],sil",
        "xor edx,edx",
        "4:",
        "xor eax,eax",
        "ret",
        "9:",
        "8:",
        "mov eax,1",
        "mov rdx,r11",
        "ret",
    );
}
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "sysv64" fn store2(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        ".byte 0xeb",
        ".byte 2f - 99f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "mov [rdi],si",
        "xor edx,edx",
        "4:",
        "xor eax,eax",
        "ret",
        "9:",
        "8:",
        "mov eax,1",
        "mov rdx,r11",
        "ret",
    );
}
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "sysv64" fn store4(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        ".byte 0xeb",
        ".byte 2f - 99f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "mov [rdi],esi",
        "xor edx,edx",
        "4:",
        "xor eax,eax",
        "ret",
        "9:",
        "8:",
        "mov eax,1",
        "mov rdx,r11",
        "ret",
    );
}
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "sysv64" fn store8(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        ".byte 0xeb",
        ".byte 2f - 99f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "mov [rdi],rsi",
        "xor edx,edx",
        "4:",
        "xor eax,eax",
        "ret",
        "9:",
        "8:",
        "mov eax,1",
        "mov rdx,r11",
        "ret",
    );
}
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "sysv64" fn cas1(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        ".byte 0xeb",
        ".byte 2f - 99f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "mov rax,rsi",
        "lock cmpxchg [rdi],dl",
        "movzx eax,al",
        "mov rdx,rax",
        "4:",
        "xor eax,eax",
        "ret",
        "9:",
        "8:",
        "mov eax,1",
        "mov rdx,r11",
        "ret",
    );
}
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "sysv64" fn cas2(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        ".byte 0xeb",
        ".byte 2f - 99f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "mov rax,rsi",
        "lock cmpxchg [rdi],dx",
        "movzx eax,ax",
        "mov rdx,rax",
        "4:",
        "xor eax,eax",
        "ret",
        "9:",
        "8:",
        "mov eax,1",
        "mov rdx,r11",
        "ret",
    );
}
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "sysv64" fn cas4(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        ".byte 0xeb",
        ".byte 2f - 99f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "mov rax,rsi",
        "lock cmpxchg [rdi],edx",
        "mov rdx,rax",
        "4:",
        "xor eax,eax",
        "ret",
        "9:",
        "8:",
        "mov eax,1",
        "mov rdx,r11",
        "ret",
    );
}
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "sysv64" fn cas8(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        ".byte 0xeb",
        ".byte 2f - 99f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "3:",
        "mov rax,rsi",
        "lock cmpxchg [rdi],rdx",
        "mov rdx,rax",
        "4:",
        "xor eax,eax",
        "ret",
        "9:",
        "8:",
        "mov eax,1",
        "mov rdx,r11",
        "ret",
    );
}
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "sysv64" fn pair(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
    core::arch::naked_asm!(
        ".byte 0xeb",
        ".byte 2f - 99f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        ".long 3f - 99b",
        ".long 3f - 99b",
        ".long 8f - 99b",
        "2:",
        "push rbx",
        "3:",
        "mov rax,[rsi]",
        "mov rdx,[rsi+8]",
        "mov rbx,[rsi+16]",
        "mov rcx,[rsi+24]",
        "lock cmpxchg16b [rdi]",
        "mov [rsi],rax",
        "mov [rsi+8],rdx",
        "4:",
        "pop rbx",
        "xor eax,eax",
        "xor edx,edx",
        "ret",
        "9:",
        "8:",
        "pop rbx",
        "mov eax,1",
        "mov rdx,r11",
        "ret",
    );
}
#[cfg(all(
    test,
    any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
mod tests {
    use super::*;
    #[test]
    fn classifier_fixup_ranges_are_bounded_and_handle_independent() {
        use crate::jit_abi::fault::{classify, HostFaultKind, HostRegisters};
        for routine in FIXUP_TABLE {
            let entry = routine as *const () as usize;
            #[cfg(target_arch = "aarch64")]
            let header = {
                // SAFETY: the routine entry is a linked AArch64 instruction.
                assert_eq!(unsafe { (entry as *const u32).read() }, 0xd503245f);
                entry + 8
            };
            #[cfg(target_arch = "x86_64")]
            let header = entry + 2;
            // SAFETY: immutable descriptor in our linked instruction image.
            let offsets = unsafe { std::ptr::read_unaligned(header as *const [i32; 6]) };
            let start = header.wrapping_add_signed(offsets[0] as isize);
            let end = header.wrapping_add_signed(offsets[1] as isize);
            let fixup = header.wrapping_add_signed(offsets[2] as isize);
            assert!(start < end && end <= fixup);
            assert_eq!(lookup(start - 1), None);
            assert_eq!(lookup(start), Some(fixup));
            assert_eq!(lookup(end - 1), Some(fixup));
            let second_start = header.wrapping_add_signed(offsets[3] as isize);
            let second_end = header.wrapping_add_signed(offsets[4] as isize);
            let second_fixup = header.wrapping_add_signed(offsets[5] as isize);
            assert_eq!(
                lookup(end),
                (end == second_start && second_start < second_end).then_some(second_fixup)
            );
            if second_start < second_end {
                assert_eq!(lookup(second_start), Some(second_fixup));
                assert_eq!(lookup(second_end - 1), Some(second_fixup));
                assert_eq!(lookup(second_end), None);
            }
            assert_eq!(lookup(fixup), None);
            for kind in [
                HostFaultKind::Access,
                HostFaultKind::Alignment,
                HostFaultKind::Unsupported,
            ] {
                let mut regs = HostRegisters {
                    pc: start as u64,
                    ..Default::default()
                };
                let before = regs;
                // SAFETY: fixup lookup precedes and does not dereference the zero handle.
                let ours = unsafe { classify(0, start, 0x12345678, kind, &mut regs) };
                assert_eq!(ours, kind == HostFaultKind::Access);
                if ours {
                    assert_eq!(regs.pc, fixup as u64);
                    assert_eq!(regs.scratch, 0x12345678);
                    regs.pc = before.pc;
                    regs.scratch = before.scratch;
                }
                assert_eq!(regs, before);
            }
        }
    }
}

#[cfg(all(test, any(target_arch = "aarch64", target_arch = "x86_64")))]
mod scalar_tests {
    use super::*;
    // Enter exactly the state a word-copy fixup receives, without requiring a
    // concurrent protection revocation to reach copy_write's retry path.
    #[cfg(target_arch = "aarch64")]
    #[unsafe(naked)]
    unsafe extern "C" fn retry_copy(_: *mut u8, _: usize, _: usize, _: usize, _: usize) -> Status {
        core::arch::naked_asm!("ret x3");
    }
    #[cfg(target_arch = "x86_64")]
    #[unsafe(naked)]
    unsafe extern "sysv64" fn retry_copy(
        _: *mut u8,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
    ) -> Status {
        core::arch::naked_asm!("jmp rcx");
    }
    #[test]
    fn both_word_copy_fixups_retry_bytes_from_the_current_word() {
        for routine in [copy_read as Routine, copy_write as Routine] {
            let entry = routine as *const () as usize;
            #[cfg(target_arch = "aarch64")]
            let header = entry + 8;
            #[cfg(target_arch = "x86_64")]
            let header = entry + 2;
            // SAFETY: immutable range metadata of this linked routine.
            let offsets = unsafe { std::ptr::read_unaligned(header as *const [i32; 6]) };
            let fixup = header.wrapping_add_signed(offsets[2] as isize);
            let input: [u8; 31] = std::array::from_fn(|i| 0xa0 + i as u8);
            let mut output = [0x55; 31];
            output[..8].copy_from_slice(&input[..8]);
            let (ptr, data) = if routine as *const () == copy_read as *const () {
                (input.as_ptr().cast_mut(), output.as_mut_ptr() as usize)
            } else {
                (output.as_mut_ptr(), input.as_ptr() as usize)
            };
            // SAFETY: both arrays cover offset 8 plus 23 remaining bytes. The
            // trampoline preserves the routine ABI and the caller's return address.
            let result = unsafe { retry_copy(ptr, data, 23, fixup, 8) };
            assert_eq!(result.failed, 0);
            assert_eq!(output, input);
        }
    }
    #[test]
    fn cas_masks_expected_at_every_width() {
        let mut data = [0x42u64; 2];
        for size in [1, 2, 4, 8] {
            for offset in [0, 1] {
                let ptr = data.as_mut_ptr().cast::<u8>().wrapping_add(offset);
                // SAFETY: both aligned and unaligned accesses stay in the local allocation.
                unsafe {
                    store(ptr, 0x42, size).unwrap();
                    let expected = if size == 8 {
                        0x42
                    } else {
                        0x42 | (1u64 << (size * 8))
                    };
                    assert_eq!(cas(ptr, expected, 0x99, size), Ok(0x42));
                    assert_eq!(load(ptr, size), Ok(0x99));
                }
            }
        }
    }
}

#[cfg(any(test, feature = "test-hooks"))]
std::thread_local! {
    static WRITE_FAULT: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    static WRITE_ONLY: std::cell::Cell<std::ops::Range<usize>> = const { std::cell::Cell::new(0..0) };
}

#[cfg(any(test, feature = "test-hooks"))]
unsafe fn test_probe_rmw(ptr: *mut u8, len: usize, page: usize) -> Status {
    // Inject read denial at any touched page, including inside a multi-page call.
    let mut offset = 0;
    while offset < len {
        let at = ptr.wrapping_add(offset);
        if deny_test_read(probe_rmw, at as usize) {
            return Status {
                failed: 2,
                value: at as usize,
            };
        }
        // SAFETY: this byte is inside the caller's validated span.
        let result = unsafe { probe_rmw(at, 0, 1, page) };
        if result.failed != 0 {
            return result;
        }
        offset += page - (at as usize & (page - 1));
    }
    Status {
        failed: 0,
        value: 0,
    }
}

#[cfg(any(test, feature = "test-hooks"))]
fn deny_test_read(routine: Routine, ptr: usize) -> bool {
    let reads = [
        copy_read, probe_read, probe_rmw, load1, load2, load4, load8, cas1, cas2, cas4, cas8, pair,
    ];
    WRITE_ONLY.with(|range| {
        let span = range.replace(0..0);
        let denied = span.contains(&ptr) && reads.iter().any(|r| *r as usize == routine as usize);
        range.set(span);
        denied
    })
}

/// Inject denied reads and a copy-write fault, committing preceding transfers.
#[cfg(any(test, feature = "test-hooks"))]
#[doc(hidden)]
pub fn with_copy_out_fault<T>(
    read_denied: std::ops::Range<usize>,
    write_fault: usize,
    run: impl FnOnce() -> T,
) -> T {
    struct Reset(std::ops::Range<usize>, Option<usize>);
    impl Drop for Reset {
        fn drop(&mut self) {
            WRITE_ONLY.with(|v| v.set(self.0.clone()));
            WRITE_FAULT.with(|v| v.set(self.1));
        }
    }
    let _reset = Reset(
        WRITE_ONLY.with(|v| v.replace(read_denied)),
        WRITE_FAULT.with(|v| v.replace(Some(write_fault))),
    );
    run()
}

#[cfg(test)]
mod store_no_read_tests {
    use super::*;
    use crate::{Exit, MemConsistency, Prot, Reg, RegionKind, Vm, VmConfig};

    fn vm_with_host_offset(config: VmConfig, offset: usize) -> Vm {
        let len = 0x10000;
        let layout = std::alloc::Layout::from_size_align(len + 0x4000, 0x4000).unwrap();
        assert!(offset < 0x4000 && offset.is_multiple_of(16));
        // SAFETY: a nonzero layout; the HostRam destructor owns the allocation.
        let backing = unsafe { std::alloc::alloc_zeroed(layout) };
        if backing.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        // SAFETY: the offset leaves len writable bytes and preserves 16-byte alignment.
        unsafe {
            Vm::with_backend_host_ram(
                config,
                Box::new(crate::InterpreterBackend),
                crate::HostRam {
                    ptr: backing.add(offset),
                    len,
                    guest_base: 0,
                    protect: None,
                    dtor: Box::new(move |ptr, _| {
                        std::alloc::dealloc(ptr.sub(offset), layout);
                    }),
                },
            )
            .unwrap()
        }
    }

    #[test]
    fn partial_copy_fault_notifies_only_committed_code() {
        crate::with_barrier_success(partial_copy_fault_notifies_only_committed_code_in_mode);
        crate::with_barrier_unavailable(partial_copy_fault_notifies_only_committed_code_in_mode);
    }

    fn partial_copy_fault_notifies_only_committed_code_in_mode() {
        use crate::lift::CpuMode;
        use crate::{BlockKey, CachedBlock};
        for path in ["loader", "checked", "guest", "scalar"] {
            for proven_prefix in [0, 16] {
                for commits in [false, true] {
                    let page = crate::memory::host_page_size();
                    let mut vm = vm_with_host_offset(VmConfig::flat(0x10000), 0);
                    vm.mem.set_host_fault_recovery(true);
                    vm.set_tier_up_after(None);
                    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
                    let first = page as u64;
                    let second = first + page as u64;
                    let addr = if path == "scalar" {
                        second - 4
                    } else {
                        first - proven_prefix
                    };
                    let first = if path == "scalar" { addr } else { first };
                    let len = if path == "scalar" {
                        8
                    } else {
                        (second - addr) as usize + 16
                    };
                    // mov eax,1; hlt, or a four-byte block ending at the page boundary.
                    let original = if path == "scalar" {
                        &[0xb0, 1, 0x90, 0xf4][..]
                    } else {
                        &[0xb8, 1, 0, 0, 0, 0xf4][..]
                    };
                    vm.write_bytes(first, original).unwrap();
                    vm.write_bytes(second, &[0xf4]).unwrap();
                    let mut cpu = vm.new_vcpu();
                    for entry in [first, second] {
                        cpu.set_reg(Reg::Rip, entry);
                        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
                    }
                    let key = BlockKey::new(first, CpuMode::Long64);
                    let suffix_key = BlockKey::new(second, CpuMode::Long64);
                    let Some(CachedBlock::Interpreted(before)) = vm.cache.get(key) else {
                        panic!("first block must be cached")
                    };
                    let Some(CachedBlock::Interpreted(suffix)) = vm.cache.get(suffix_key) else {
                        panic!("suffix block must be cached")
                    };
                    let mut old = vec![0; len];
                    vm.mem.read_bytes(addr, &mut old).unwrap();
                    let mut bytes = vec![0x90; len];
                    let offset = (first - addr) as usize;
                    bytes[offset..offset + original.len()].copy_from_slice(original);
                    bytes[offset + 1] = 2;
                    let host = vm.mem.host_base() as usize;
                    let denied = if path == "scalar" { addr } else { page as u64 };
                    let fault = if commits { second } else { denied };
                    let committed = if commits { (second - addr) as usize } else { 0 };
                    let failed = with_copy_out_fault(
                        host + denied as usize..host + second as usize + page,
                        host + fault as usize,
                        || match path {
                            "loader" => vm.mem.write_bytes(addr, &bytes).is_err(),
                            "checked" => vm.mem.write_ram_checked(addr, &bytes) == Err(fault),
                            "guest" => !vm.mem.write_ram_guest(addr, &bytes),
                            "scalar" => vm
                                .mem
                                .write(addr, u64::from_le_bytes(bytes[..8].try_into().unwrap()), 8)
                                .is_err(),
                            _ => unreachable!(),
                        },
                    );
                    assert!(failed, "{path}");
                    let mut actual = vec![0; len];
                    vm.mem.read_bytes(addr, &mut actual).unwrap();
                    assert_eq!(&actual[..committed], &bytes[..committed], "{path}");
                    assert_eq!(&actual[committed..], &old[committed..], "{path}");
                    assert_eq!(vm.mem.has_dirty_code(), commits, "{path}");
                    cpu.set_reg(Reg::Rip, first);
                    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
                    assert_eq!(cpu.reg(Reg::Rax), if commits { 2 } else { 1 }, "{path}");
                    let Some(CachedBlock::Interpreted(after)) = vm.cache.get(key) else {
                        panic!("first block must be cached")
                    };
                    assert_eq!(std::sync::Arc::ptr_eq(&before, &after), !commits, "{path}");
                    let Some(CachedBlock::Interpreted(after)) = vm.cache.get(suffix_key) else {
                        panic!("uncommitted suffix must stay cached")
                    };
                    assert!(
                        std::sync::Arc::ptr_eq(&suffix, &after),
                        "{path}: suffix was invalidated"
                    );
                }
            }
        }
    }

    #[test]
    fn raw_fp_and_string_stores_retain_partial_copy_progress() {
        use crate::x87::RawFpMem;
        let page = crate::memory::host_page_size();
        let mut vm = vm_with_host_offset(VmConfig::flat(0x10000), 0);
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        let base = vm.mem.host_base().cast_mut();
        let first = page as u64;
        let fault = first + page as u64;
        let fp = RawFpMem {
            base,
            size: 0x10000,
            guest_base: 0,
            recover_host_faults: true,
        };
        let result = with_copy_out_fault(
            base as usize + page..base as usize + page * 3,
            base as usize + page * 2,
            || fp.store_checked_with_progress(first - 16, &vec![0x77; page + 32]),
        );
        assert_eq!(result, Err((fault, page + 16)));
        let raw = crate::interp::RawStrMem {
            base,
            size: 0x10000,
            guest_base: 0,
            recover_host_faults: true,
        };
        let result = with_copy_out_fault(
            base as usize + page * 2 - 4..base as usize + page * 3,
            base as usize + page * 2,
            || raw.store_with_progress(fault - 4, 0x9999_9999_9999_9999, 8),
        );
        assert_eq!(result, Err((crate::MemTrap::HostFault, 4)));
    }

    #[test]
    fn xsavec_guest_alignment_does_not_imply_host_page_alignment() {
        for (page, expected_queries) in [(4096, 2), (16384, 0)] {
            crate::memory::with_host_page_size_for_test(page, || {
                let mut vm = vm_with_host_offset(VmConfig::flat(0x10000), 4096 - 512);
                vm.mem.set_host_fault_recovery(true);
                vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
                vm.write_bytes(0x1000, &[0x48, 0x0f, 0xc7, 0x27, 0xf4])
                    .unwrap();
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, 0x1000);
                cpu.set_reg(Reg::Rdi, 0x8000);
                cpu.set_reg(Reg::Rax, 7);
                let start = vm.mem.host_base() as usize + 0x8000;
                assert_eq!(start % page, 4096 - 512);
                WRITE_ONLY.with(|r| r.set(start..start + 0x7000));
                crate::host_write::QUERIES.with(|n| n.set(0));
                let exit = cpu.run(&vm, None);
                WRITE_ONLY.with(|r| r.set(0..0));
                let queries = crate::host_write::QUERIES.with(std::cell::Cell::get);
                eprintln!(
                    "MEASURED: XSAVEC guest=0x8000 host_offset={} page={page} queries={queries}",
                    start % page
                );
                assert!(matches!(exit, Exit::Hlt), "{exit:?}");
                assert_eq!(queries, expected_queries, "page={page}");
                assert_eq!(vm.mem.read(0x8200, 8).unwrap(), 7);
                assert_eq!(vm.mem.read(0x8208, 8).unwrap(), (1 << 63) | 7);
            });
        }
    }

    #[test]
    #[cfg(all(
        any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    fn accepted_probe_read_fault_cannot_replace_later_write_evidence() {
        use crate::jit_abi::fault::{
            classify_with_raw_fault, EntryRecord, HostFaultKind, HostRegisters,
        };
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        let mut record = EntryRecord::default();
        let ptr = vm.mem.host_base().wrapping_add(0x8000).cast_mut();
        let entry = probe_rmw as *const () as usize;
        #[cfg(target_arch = "aarch64")]
        let header = entry + 8;
        #[cfg(target_arch = "x86_64")]
        let header = entry + 2;
        // SAFETY: immutable descriptor of the linked probe routine.
        let offset = unsafe { (header as *const i32).read_unaligned() };
        let pc = header.wrapping_add_signed(offset as isize);
        let read = crate::RawHostFault {
            error_code: Some(0x10),
            fault_address: Some(ptr as u64),
            ..crate::RawHostFault::new(crate::HostFaultPlatform::MacOsAarch64)
        };
        let write = crate::RawHostFault {
            error_code: Some(0x20),
            ..read
        };
        let mut regs = HostRegisters {
            pc: pc as u64,
            ..Default::default()
        };
        // SAFETY: the live record is thread-local; this fixup reads no frames or CPU.
        assert!(unsafe {
            classify_with_raw_fault(
                record.handle().as_usize(),
                pc,
                ptr as usize,
                HostFaultKind::Access,
                &mut regs,
                read,
            )
        });
        WRITE_ONLY.with(|r| r.set(ptr as usize..ptr as usize + 1));
        crate::host_write::QUERIES.with(|n| n.set(0));
        // SAFETY: mapped byte; the hook returns the denied probe-read status.
        let accepted = unsafe { probe(ptr, 1, true) };
        WRITE_ONLY.with(|r| r.set(0..0));
        assert_eq!(accepted, Ok(()));
        assert_eq!(crate::host_write::QUERIES.with(std::cell::Cell::get), 1);

        // A later compiled write landing already owns the newer signal record.
        let exit =
            crate::fault::memory_exit(0x1000, 0x8000, 8, crate::AccessKind::Write, true, true);
        let metadata = exit
            .fault_metadata()
            .unwrap()
            .clone()
            .with_raw_host_fault(write);
        let exit = record.finish_transfer_fault(exit.with_fault_metadata(metadata), &vm.mem);
        assert_eq!(
            exit.fault_metadata().unwrap().raw_host_fault(),
            Some(&write)
        );
    }

    #[test]
    fn scalar_guest_boundary_refinement_does_not_probe_host_memory() {
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        vm.mem.set_host_fault_recovery(true);
        vm.map(0, 0x8004, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, &[0x48, 0x89, 0x07, 0xf4]).unwrap();
        vm.write_bytes(0x8000, &[0x55; 4]).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, 0x1000);
        cpu.set_reg(Reg::Rdi, 0x8000);
        let ptr = vm.mem.host_base() as usize + 0x8000;
        WRITE_ONLY.with(|r| r.set(ptr..ptr + 4));
        crate::host_write::QUERIES.with(|n| n.set(0));
        let exit = cpu.run(&vm, None);
        WRITE_ONLY.with(|r| r.set(0..0));
        assert!(matches!(exit, Exit::UnmappedMemory { addr: 0x8004, .. }));
        assert_eq!(crate::host_write::QUERIES.with(std::cell::Cell::get), 0);
        assert_eq!(vm.mem.read(0x8000, 4).unwrap(), 0x5555_5555);
    }

    #[test]
    fn crossing_second_page_read_denied_and_query_failure_preserves_original_fault() {
        let page = crate::memory::host_page_size();
        let mut data = vec![0x55u8; page * 4];
        let edge = (data.as_mut_ptr() as usize + page).next_multiple_of(page);
        let ptr = (edge - 8) as *mut u8;
        WRITE_ONLY.with(|r| r.set(edge..edge + page));
        crate::host_write::QUERIES.with(|n| n.set(0));
        crate::host_write::FAIL_QUERY.with(|f| f.set(true));
        // SAFETY: both pages are within data; only the hook denies the second read.
        let result = unsafe { copy_out(ptr, &[0x99; 16]) };
        crate::host_write::FAIL_QUERY.with(|f| f.set(false));
        assert_eq!(
            result,
            Err(WriteFault {
                at: edge,
                committed: 0
            }),
            "retain the original second-page read fault"
        );
        assert!(data.iter().all(|&v| v == 0x55), "no partial store");
        assert_eq!(crate::host_write::QUERIES.with(std::cell::Cell::get), 1);
        crate::host_write::QUERIES.with(|n| n.set(0));
        // SAFETY: same allocation; the writable-page query now succeeds.
        assert_eq!(unsafe { copy_out(ptr, &[0x99; 16]) }, Ok(()));
        WRITE_ONLY.with(|r| r.set(0..0));
        assert_eq!(crate::host_write::QUERIES.with(std::cell::Cell::get), 1);
        let offset = ptr as usize - data.as_ptr() as usize;
        assert_eq!(&data[offset..offset + 16], &[0x99; 16]);
        WRITE_ONLY.with(|r| r.set(edge..edge + page));
        crate::host_write::QUERIES.with(|n| n.set(0));
        let bytes = vec![0x77; page + 16];
        // SAFETY: the larger copy stays inside data and must continue past the denied page.
        assert_eq!(unsafe { copy_out(ptr, &bytes) }, Ok(()));
        WRITE_ONLY.with(|r| r.set(0..0));
        assert_eq!(crate::host_write::QUERIES.with(std::cell::Cell::get), 1);
        assert_eq!(&data[offset..offset + bytes.len()], bytes);
    }

    #[test]
    fn pure_stores_with_reads_denied() {
        for page in [4096, 16384] {
            crate::memory::with_host_page_size_for_test(
                page,
                pure_stores_with_reads_denied_on_page,
            );
        }
    }

    fn pure_stores_with_reads_denied_on_page() {
        // The transfer hook models a write-only page even where PROT_WRITE implies
        // read access. Reject reads (including same-value RMW probes) before access.
        // SDM Vol 3A §11.1.1 permits splitting VMOVDQU, unlike aligned MOVAPS on
        // AVX processors. SDM Vol 2C, VMASKMOV/VPMASKMOV, Operation: selected
        // elements are stores; inactive elements do not access memory.
        let cases: &[(&str, &[u8])] = &[
            ("mov8", &[0x88, 0x07]),
            ("mov16", &[0x66, 0x89, 0x07]),
            ("mov32", &[0x89, 0x07]),
            ("mov64", &[0x48, 0x89, 0x07]),
            ("movq", &[0x66, 0x0f, 0xd6, 0x07]),
            ("movups", &[0x0f, 0x11, 0x07]),
            ("movaps", &[0x0f, 0x29, 0x07]),
            ("vmovdqu128", &[0xc5, 0xfa, 0x7f, 0x07]),
            ("vmovdqu256", &[0xc5, 0xfe, 0x7f, 0x07]),
            ("vmovdqu512", &[0x62, 0xf1, 0x7e, 0x48, 0x7f, 0x07]),
            ("movnti", &[0x48, 0x0f, 0xc3, 0x07]),
            ("movntdq", &[0x66, 0x0f, 0xe7, 0x07]),
            ("movntps", &[0x0f, 0x2b, 0x07]),
            ("movntpd", &[0x66, 0x0f, 0x2b, 0x07]),
            ("vmovntdq", &[0xc5, 0xfd, 0xe7, 0x07]),
            ("stosb", &[0xf3, 0xaa]),
            ("stosq", &[0xf3, 0x48, 0xab]),
            ("vmaskmovps", &[0xc4, 0xe2, 0x75, 0x2e, 0x07]),
            ("vpmaskmovq", &[0xc4, 0xe2, 0xf5, 0x8e, 0x07]),
            ("masked8", &[0x62, 0xf1, 0x7f, 0x49, 0x7f, 0x07]),
            ("masked32", &[0x62, 0xf1, 0x7e, 0x49, 0x7f, 0x07]),
            ("masked64", &[0x62, 0xf1, 0xfe, 0x49, 0x7f, 0x07]),
            ("narrow", &[0x62, 0xf2, 0x7e, 0x48, 0x31, 0x07]),
            ("vcvtps2ph", &[0xc4, 0xe3, 0x7d, 0x1d, 0x07, 0]),
            ("fxsave", &[0x48, 0x0f, 0xae, 0x07]),
            ("xsavec", &[0x48, 0x0f, 0xc7, 0x27]),
        ];
        let mut failures = Vec::new();
        for tier in [
            MemConsistency::Fast,
            MemConsistency::AcqRel,
            MemConsistency::FullTso,
        ] {
            for &(name, code) in cases {
                for crossing in [false, true] {
                    let mut config = VmConfig::flat(0x10000);
                    config.consistency = tier;
                    let mut vm = vm_with_host_offset(config, crate::memory::host_page_size() - 512);
                    vm.mem.set_host_fault_recovery(true);
                    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
                    let mut code = code.to_vec();
                    code.push(0xf4);
                    vm.write_bytes(0x1000, &code).unwrap();
                    let mut cpu = vm.new_vcpu();
                    cpu.set_reg(Reg::Rip, 0x1000);
                    let page = crate::memory::host_page_size();
                    let host = vm.mem.host_base() as usize;
                    let edge = (host + 0x9000).div_ceil(page) * page - host;
                    let address = if !crossing {
                        // Guest alignment alone cannot keep a heap-backed store
                        // on one host page. Leave room for the largest fixture,
                        // XSAVEC with RAX=7, and retain guest 64-byte alignment.
                        let address =
                            ((host + 0x8000).next_multiple_of(page) - host).next_multiple_of(64);
                        assert!(
                            (host + address) % page + crate::x87::xstate::area_size(7) as usize
                                <= page
                        );
                        address
                    } else {
                        match name {
                            "fxsave" | "xsavec" => (edge - 256) & !63,
                            "movaps" | "movntdq" | "movntps" | "movntpd" => (edge - 16) & !15,
                            "vmovntdq" => (edge - 16) & !31,
                            "mov16" => edge - 1,
                            _ => edge - 3,
                        }
                    };
                    cpu.set_reg(Reg::Rdi, address as u64);
                    cpu.set_reg(Reg::Rax, 7);
                    cpu.set_reg(Reg::Rcx, 3);
                    cpu.cpu.set_vec(0, [u128::MAX; 4], 64);
                    cpu.cpu.set_vec(1, [u128::MAX; 4], 64);
                    cpu.cpu.kmask[1] = 0x5555_5555_5555_5555;
                    let start = vm.mem.host_base() as usize + 0x8000;
                    WRITE_ONLY.with(|r| r.set(start..start + 0x7000));
                    crate::host_write::QUERIES.with(|n| n.set(0));
                    let exit = cpu.run(&vm, None);
                    WRITE_ONLY.with(|r| r.set(0..0));
                    if !crossing {
                        assert_eq!(
                            crate::host_write::QUERIES.with(std::cell::Cell::get),
                            0,
                            "{name} {tier:?} page={page}: single-page preflight"
                        );
                    }
                    if crossing
                        && matches!(
                            name,
                            "fxsave" | "xsavec" | "vmovdqu512" | "masked8" | "vcvtps2ph"
                        )
                    {
                        assert!(
                            crate::host_write::QUERIES.with(std::cell::Cell::get) <= 2,
                            "{name} page={page}: probes must be per page"
                        );
                    }
                    if !matches!(exit, Exit::Hlt) {
                        failures.push(format!("{name} {tier:?} crossing={crossing}: {exit:?}"));
                    }
                }
            }
        }
        assert!(
            failures.is_empty(),
            "reads or unexpected exits: {failures:#?}"
        );
    }
}

#[cfg(test)]
mod architectural_reads {
    use super::*;
    use crate::{AccessKind, Exit, Prot, Reg, RegionKind, Vm, VmConfig};

    #[test]
    fn xsave_and_xsaveopt_require_readable_header() {
        // SDM Vol 1 §§13.7, 13.9: these save forms read XSTATE_BV;
        // XSAVEC (§13.10) does not. Keep the architectural read and its fault.
        for modrm in [0x27, 0x37] {
            let mut vm = Vm::new(VmConfig::flat(0x10000));
            vm.mem.set_host_fault_recovery(true);
            vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0x1000, &[0x48, 0x0f, 0xae, modrm, 0xf4])
                .unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, 0x1000);
            cpu.set_reg(Reg::Rdi, 0x8000);
            cpu.set_reg(Reg::Rax, 7);
            let header = vm.mem.host_base() as usize + 0x8200;
            WRITE_ONLY.with(|r| r.set(header..header + 8));
            let exit = cpu.run(&vm, None);
            WRITE_ONLY.with(|r| r.set(0..0));
            assert!(
                matches!(
                    exit,
                    Exit::UnmappedMemory {
                        addr: 0x8200,
                        access: AccessKind::Read,
                        ..
                    }
                ),
                "{exit:?}"
            );
            assert_eq!(cpu.reg(Reg::Rip), 0x1000);
        }
    }
}
