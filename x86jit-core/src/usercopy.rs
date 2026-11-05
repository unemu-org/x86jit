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
    copy_read,
    copy_write,
    probe_read,
    probe_write,
    load1,
    load2,
    load4,
    load8,
    store1,
    store2,
    store4,
    store8,
    cas1,
    cas2,
    cas4,
    cas8,
    pair
);
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
static FIXUP_TABLE: [Routine; 17] = [
    copy_read,
    copy_write,
    probe_read,
    probe_write,
    load1,
    load2,
    load4,
    load8,
    store1,
    store2,
    store4,
    store8,
    cas1,
    cas2,
    cas4,
    cas8,
    pair,
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
pub(crate) unsafe fn probe(ptr: *mut u8, len: usize, write: bool) -> Result<(), usize> {
    // SAFETY: caller validates the guest range. Page size is obtained in normal Rust.
    unsafe {
        invoke(
            if write { probe_write } else { probe_read },
            ptr,
            0,
            len,
            crate::memory::host_page_size(),
        )
        .map(|_| ())
    }
}
pub(crate) unsafe fn copy_out(ptr: *mut u8, buf: &[u8]) -> Result<(), usize> {
    let page = crate::memory::host_page_size();
    // Preflight all touched pages if more than one is touched. A single-page
    // store faults on its first write before committing bytes. Protection revocation
    // during the subsequent copy still requires embedder synchronization.
    // SAFETY: caller validates the range and the source remains live.
    unsafe {
        if buf.len() > page - ptr as usize % page {
            probe(ptr, buf.len(), true)?;
        }
        invoke(copy_write, ptr, buf.as_ptr() as usize, buf.len(), 0).map(|_| ())
    }
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
pub(crate) unsafe fn store(ptr: *mut u8, value: u64, size: u8) -> Result<(), usize> {
    if !matches!(size, 1 | 2 | 4 | 8) || (ptr as usize & 4095) + size as usize > 4096 {
        // SAFETY: crossing accesses preflight every touched page before copying.
        return unsafe { copy_out(ptr, &value.to_le_bytes()[..size as usize]) };
    }
    let routine = match size {
        1 => store1,
        2 => store2,
        4 => store4,
        _ => store8,
    };
    // SAFETY: caller validates the entire scalar span.
    unsafe { invoke(routine, ptr, value as usize, 0, 0).map(|_| ()) }
}
pub(crate) unsafe fn cas(ptr: *mut u8, expected: u64, value: u64, size: u8) -> Result<u64, usize> {
    let mask = u64::MAX >> (64 - size * 8);
    let expected = expected & mask;
    if !(ptr as usize).is_multiple_of(size as usize) {
        // SAFETY: retains the existing non-atomic misaligned fallback; no unaligned exclusive.
        unsafe {
            probe(ptr, size as usize, true)?;
            let old = load(ptr, size)?;
            if old == expected {
                store(ptr, value, size)?;
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
unsafe extern "C" fn probe_write(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
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
        "6:",
        "ldaxrb w5,[x6]",
        "stlxrb w7,w5,[x6]",
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
        "8:",
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
unsafe extern "sysv64" fn probe_write(_: *mut u8, _: usize, _: usize, _: usize) -> Status {
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
        "6:",
        "lock cmpxchg byte ptr [rdi+r8],al",
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
    any(target_os = "linux", target_os = "macos"),
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
            assert_eq!(lookup(end), None);
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
