//! Static transfer recovery; no code map, TLS lookup or parent run record.

use crate::jit_abi::fault::HostRegisters;
use crate::RawHostFault;

#[derive(Default)]
pub(super) struct Record {
    pub raw: Option<RawHostFault>,
}
#[repr(C)]
struct Status {
    failed: usize,
    value: usize,
}

/// Classify a synchronous child access fault. Returns false without changing
/// anything for a PC outside the dedicated transfer range. On true, copy the
/// changed registers back to the signal context and return from the handler.
/// Uses only immutable linked text and the arena record carried in scratch.
/// No TLS, allocation, atomics, locks, callbacks or parent engine objects.
///
/// # Safety
/// Call only on the accessing thread for a verified synchronous access fault
/// (never a user-sent signal), with the actual interrupted PC and scratch register
/// (x9/r11) in `regs`. Keep that step's CPU, view, arena and mapping alive.
/// The signal adapter must preserve scratch on entry and copy the returned PC
/// back to the host context. Do not reenter a step from its signal handler.
pub unsafe fn classify(pc: usize, regs: &mut HostRegisters, raw: RawHostFault) -> bool {
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    {
        for entry in [
            access as *const () as usize,
            atomic_access as *const () as usize,
        ] {
            #[cfg(target_arch = "aarch64")]
            let header = entry + 8;
            #[cfg(target_arch = "x86_64")]
            let header = entry + 2;
            // SAFETY: three immutable relative words embedded in linked routine text.
            let offsets = unsafe { std::ptr::read_unaligned(header as *const [i32; 3]) };
            let address = |i: usize| header.wrapping_add_signed(offsets[i] as isize);
            if !(address(0)..address(1)).contains(&pc) {
                continue;
            }
            // SAFETY: within this range scratch is this thread's live arena record.
            let record = unsafe { &mut *(regs.scratch as *mut Record) };
            record.raw = Some(raw);
            regs.pc = address(2) as u64;
            return true;
        }
        false
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = (pc, regs, raw);
        false
    }
}

unsafe fn invoke(
    ptr: usize,
    value: u64,
    size: u8,
    record: *mut Record,
    write: bool,
) -> Result<u64, usize> {
    // SAFETY: caller validated the whole transfer against the live snapshot.
    let status = unsafe {
        access(
            ptr,
            value as usize,
            size as usize,
            record,
            usize::from(write),
        )
    };
    if status.failed == 0 {
        Ok(status.value as u64)
    } else {
        Err(ptr)
    }
}

pub(super) unsafe fn load(ptr: usize, size: u8, record: *mut Record) -> Result<u64, usize> {
    if ptr & 4095 <= 4096 - usize::from(size) {
        // SAFETY: inherited from the caller's backing bounds/lifetime check.
        return unsafe { invoke(ptr, 0, size, record, false) };
    }
    let mut result = 0;
    for i in 0..size {
        // SAFETY: each byte belongs to the validated transfer; this refines FAR.
        let v = unsafe { invoke(ptr + usize::from(i), 0, 1, record, false) }?;
        result |= v << (i * 8);
    }
    Ok(result)
}
pub(super) unsafe fn store(
    ptr: usize,
    value: u64,
    size: u8,
    record: *mut Record,
) -> Result<(), usize> {
    if ptr & 4095 <= 4096 - usize::from(size) {
        // SAFETY: inherited from the caller's backing bounds/lifetime check.
        return unsafe { invoke(ptr, value, size, record, true) }.map(|_| ());
    }
    for i in 0..size {
        // SAFETY: each byte is in the validated backing; faults can leave a prefix.
        unsafe { invoke(ptr + usize::from(i), value >> (i * 8), 1, record, true) }?;
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn access(_: usize, _: usize, _: usize, _: *mut Record, _: usize) -> Status {
    core::arch::naked_asm!(
        "hint #34",
        "b 2f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        "2:",
        "mov x9, x3",
        "3:",
        "cbnz x4, 5f",
        "cmp x2, #1",
        "b.eq 11f",
        "cmp x2, #2",
        "b.eq 12f",
        "cmp x2, #4",
        "b.eq 13f",
        "ldr x1, [x0]",
        "b 4f",
        "11:",
        "ldrb w1, [x0]",
        "b 4f",
        "12:",
        "ldrh w1, [x0]",
        "b 4f",
        "13:",
        "ldr w1, [x0]",
        "b 4f",
        "5:",
        "cmp x2, #1",
        "b.eq 21f",
        "cmp x2, #2",
        "b.eq 22f",
        "cmp x2, #4",
        "b.eq 23f",
        "str x1, [x0]",
        "b 4f",
        "21:",
        "strb w1, [x0]",
        "b 4f",
        "22:",
        "strh w1, [x0]",
        "b 4f",
        "23:",
        "str w1, [x0]",
        "4:",
        "mov x0, #0",
        "ret",
        "9:",
        "mov x0, #1",
        "ret",
    );
}
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "sysv64" fn access(_: usize, _: usize, _: usize, _: *mut Record, _: usize) -> Status {
    core::arch::naked_asm!(
        "jmp 2f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        "2:",
        "mov r11, rcx",
        "mov r10, rdx",
        "3:",
        "test r8, r8",
        "jnz 5f",
        "cmp r10, 1",
        "je 11f",
        "cmp r10, 2",
        "je 12f",
        "cmp r10, 4",
        "je 13f",
        "mov rdx, [rdi]",
        "jmp 4f",
        "11:",
        "movzx edx, byte ptr [rdi]",
        "jmp 4f",
        "12:",
        "movzx edx, word ptr [rdi]",
        "jmp 4f",
        "13:",
        "mov edx, [rdi]",
        "jmp 4f",
        "5:",
        "cmp r10, 1",
        "je 21f",
        "cmp r10, 2",
        "je 22f",
        "cmp r10, 4",
        "je 23f",
        "mov [rdi], rsi",
        "jmp 4f",
        "21:",
        "mov [rdi], sil",
        "jmp 4f",
        "22:",
        "mov [rdi], si",
        "jmp 4f",
        "23:",
        "mov [rdi], esi",
        "4:",
        "xor eax, eax",
        "ret",
        "9:",
        "mov eax, 1",
        "ret",
    );
}
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
unsafe extern "C" fn access(
    ptr: usize,
    _: usize,
    _: usize,
    record: *mut Record,
    _: usize,
) -> Status {
    let _ = (ptr, record);
    Status {
        failed: 1,
        value: 0,
    }
}

// Feature probing is parent-side: the step never consults runtime detection's
// shared cache (nor any TLS/engine object).
pub(super) struct AtomicFeatures {
    pub pair: bool,
    pub(super) lse: bool,
}
impl AtomicFeatures {
    pub fn detect() -> Self {
        Self {
            #[cfg(target_arch = "x86_64")]
            pair: std::is_x86_feature_detected!("cmpxchg16b"),
            #[cfg(not(target_arch = "x86_64"))]
            pair: cfg!(target_arch = "aarch64"),
            #[cfg(target_arch = "aarch64")]
            lse: std::arch::is_aarch64_feature_detected!("lse"),
            #[cfg(not(target_arch = "aarch64"))]
            lse: false,
        }
    }
}

#[repr(C)]
struct AtomicTransfer {
    ptr: usize,
    expected: [u64; 2],
    src: [u64; 2],
    old: [u64; 2],
    size: usize,
    lse: usize,
}

pub(super) unsafe fn compare_exchange(
    ptr: usize,
    expected: u128,
    src: u128,
    size: u8,
    record: *mut Record,
    features: &AtomicFeatures,
) -> Result<u128, ()> {
    let mut transfer = AtomicTransfer {
        ptr,
        expected: [expected as u64, (expected >> 64) as u64],
        src: [src as u64, (src >> 64) as u64],
        old: [0; 2],
        size: size.into(),
        lse: features.lse.into(),
    };
    // SAFETY: caller checked the backing, alignment and instruction support.
    // Only the guest atomic instructions are inside the classified PC range.
    let failed = unsafe { atomic_access(&mut transfer, record) };
    if failed == 0 {
        Ok(u128::from(transfer.old[0]) | (u128::from(transfer.old[1]) << 64))
    } else {
        Err(())
    }
}

// SDM Vol 3A §11.2.3.8/§11.2.3.9: locked operations have a total
// order and order earlier/later loads and stores. Every attempt uses LOCK;
// ARM uses full barriers on both sides, including comparison failure.
// SDM Vol 2A §3.2, CMPXCHG and CMPXCHG8B/CMPXCHG16B, Description:
// comparison failure still requires a write cycle. The exclusive path stores
// back the observed value on mismatch, also validating the whole 128-bit pair.
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "sysv64" fn atomic_access(_: *mut AtomicTransfer, _: *mut Record) -> usize {
    core::arch::naked_asm!(
        "jmp 2f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        "2:",
        "push rbx",
        "mov r11, rsi",
        "mov rsi, [rdi]",
        "mov rax, [rdi + 8]",
        "mov rdx, [rdi + 16]",
        "mov rbx, [rdi + 24]",
        "mov rcx, [rdi + 32]",
        "mov r10, [rdi + 56]",
        "cmp r10, 16",
        "je 16f",
        "xor edx, edx",
        "cmp r10, 1",
        "je 11f",
        "cmp r10, 2",
        "je 12f",
        "cmp r10, 4",
        "je 13f",
        "jmp 18f",
        "3:",
        "11:",
        "lock cmpxchg byte ptr [rsi], bl",
        "movzx eax, al",
        "jmp 4f",
        "12:",
        "lock cmpxchg word ptr [rsi], bx",
        "movzx eax, ax",
        "jmp 4f",
        "13:",
        "lock cmpxchg dword ptr [rsi], ebx",
        "mov eax, eax",
        "jmp 4f",
        "18:",
        "lock cmpxchg qword ptr [rsi], rbx",
        "jmp 4f",
        "16:",
        "lock cmpxchg16b [rsi]",
        "4:",
        "mov [rdi + 40], rax",
        "mov [rdi + 48], rdx",
        "xor eax, eax",
        "pop rbx",
        "ret",
        "9:",
        "mov eax, 1",
        "pop rbx",
        "ret",
    );
}
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn atomic_access(_: *mut AtomicTransfer, _: *mut Record) -> usize {
    core::arch::naked_asm!(
        "hint #34",
        "b 2f",
        "99:",
        ".long 3f - 99b",
        ".long 4f - 99b",
        ".long 9f - 99b",
        "2:",
        "mov x10, x0",
        "mov x9, x1",
        "ldr x11, [x10]",
        "ldp x0, x1, [x10, #8]",
        "ldp x2, x3, [x10, #24]",
        "ldr x12, [x10, #56]",
        "ldr x13, [x10, #64]",
        "dmb ish",
        "cmp x12, #16",
        "b.eq 16f",
        "cmp x12, #1",
        "b.eq 11f",
        "cmp x12, #2",
        "b.eq 12f",
        "cmp x12, #4",
        "b.eq 13f",
        "b 18f",
        "3:",
        "11:",
        "cbz x13, 31f",
        ".arch_extension lse",
        "casalb w0, w2, [x11]",
        "mov w4, w0",
        "mov x5, #0",
        "b 4f",
        "31:",
        "ldaxrb w4, [x11]",
        "cmp w4, w0",
        "csel w6, w2, w4, eq",
        "stlxrb w14, w6, [x11]",
        "cbnz w14, 31b",
        "mov x5, #0",
        "b 4f",
        "12:",
        "cbz x13, 32f",
        ".arch_extension lse",
        "casalh w0, w2, [x11]",
        "mov w4, w0",
        "mov x5, #0",
        "b 4f",
        "32:",
        "ldaxrh w4, [x11]",
        "cmp w4, w0",
        "csel w6, w2, w4, eq",
        "stlxrh w14, w6, [x11]",
        "cbnz w14, 32b",
        "mov x5, #0",
        "b 4f",
        "13:",
        "cbz x13, 33f",
        ".arch_extension lse",
        "casal w0, w2, [x11]",
        "mov w4, w0",
        "mov x5, #0",
        "b 4f",
        "33:",
        "ldaxr w4, [x11]",
        "cmp w4, w0",
        "csel w6, w2, w4, eq",
        "stlxr w14, w6, [x11]",
        "cbnz w14, 33b",
        "mov x5, #0",
        "b 4f",
        "18:",
        "cbz x13, 38f",
        ".arch_extension lse",
        "casal x0, x2, [x11]",
        "mov x4, x0",
        "mov x5, #0",
        "b 4f",
        "38:",
        "ldaxr x4, [x11]",
        "cmp x4, x0",
        "csel x6, x2, x4, eq",
        "stlxr w14, x6, [x11]",
        "cbnz w14, 38b",
        "mov x5, #0",
        "b 4f",
        "16:",
        "cbz x13, 36f",
        "caspal x0, x1, x2, x3, [x11]",
        "mov x4, x0",
        "mov x5, x1",
        "b 4f",
        "36:",
        "ldaxp x4, x5, [x11]",
        "cmp x4, x0",
        "ccmp x5, x1, #0, eq",
        "csel x6, x2, x4, eq",
        "csel x7, x3, x5, eq",
        "stlxp w14, x6, x7, [x11]",
        "cbnz w14, 36b",
        "4:",
        "dmb ish",
        "stp x4, x5, [x10, #40]",
        "mov x0, #0",
        "ret",
        "9:",
        "clrex",
        "mov x0, #1",
        "ret",
    );
}
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
unsafe extern "C" fn atomic_access(_: *mut AtomicTransfer, _: *mut Record) -> usize {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_transfer_layout_matches_assembly() {
        assert_eq!(std::mem::offset_of!(AtomicTransfer, ptr), 0);
        assert_eq!(std::mem::offset_of!(AtomicTransfer, expected), 8);
        assert_eq!(std::mem::offset_of!(AtomicTransfer, src), 24);
        assert_eq!(std::mem::offset_of!(AtomicTransfer, old), 40);
        assert_eq!(std::mem::offset_of!(AtomicTransfer, size), 56);
        assert_eq!(std::mem::offset_of!(AtomicTransfer, lse), 64);
    }

    #[test]
    fn atomic_exclusive_and_detected_paths_preserve_neighbours() {
        let detected = AtomicFeatures::detect();
        for lse in [false, detected.lse] {
            let features = AtomicFeatures {
                pair: detected.pair,
                lse,
            };
            for size in [1, 2, 4, 8, 16] {
                if size == 16 && !features.pair {
                    continue;
                }
                let mut value = 0xa5a5_a5a5_a5a5_a5a5_a5a5_a5a5_a5a5_a5a5u128;
                let mask = u128::MAX >> (128 - size * 8);
                let initial = value;
                let mut record = Record::default();
                let ptr = std::ptr::addr_of_mut!(value) as usize;
                // SAFETY: live, naturally aligned writable stack operand; only
                // this routine touches it until the calls return.
                unsafe {
                    assert_eq!(
                        compare_exchange(ptr, 0, 7, size, &mut record, &features),
                        Ok(initial & mask)
                    );
                    assert_eq!(value, initial);
                    assert_eq!(
                        compare_exchange(ptr, initial & mask, 7, size, &mut record, &features),
                        Ok(initial & mask)
                    );
                    assert_eq!(value, (initial & !mask) | 7);
                }
            }
        }
    }

    #[test]
    fn atomic_pair_paths_do_not_tear_under_contention() {
        let detected = AtomicFeatures::detect();
        if !detected.pair {
            return;
        }
        for lse in [false, detected.lse] {
            let features = AtomicFeatures { pair: true, lse };
            let mut value = 0u128;
            let ptr = std::ptr::addr_of_mut!(value) as usize;
            let barrier = std::sync::Barrier::new(4);
            std::thread::scope(|scope| {
                for _ in 0..4 {
                    let (features, barrier) = (&features, &barrier);
                    scope.spawn(move || {
                        let mut record = Record::default();
                        let mut expected = 0u128;
                        barrier.wait();
                        for _ in 0..1000 {
                            loop {
                                let count = expected as u64 + 1;
                                let next = (u128::from(count) << 64) | u128::from(count);
                                // SAFETY: the parent retains the aligned operand;
                                // every concurrent access uses this atomic routine.
                                let old = unsafe {
                                    compare_exchange(ptr, expected, next, 16, &mut record, features)
                                }
                                .unwrap();
                                assert_eq!(old as u64, (old >> 64) as u64);
                                if old == expected {
                                    expected = next;
                                    break;
                                }
                                expected = old;
                            }
                        }
                    });
                }
            });
            assert_eq!(value, (4000u128 << 64) | 4000);
        }
    }
}
