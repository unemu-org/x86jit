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
        let entry = access as *const () as usize;
        #[cfg(target_arch = "aarch64")]
        let header = entry + 8;
        #[cfg(target_arch = "x86_64")]
        let header = entry + 2;
        // SAFETY: three immutable relative words embedded in linked routine text.
        let offsets = unsafe { std::ptr::read_unaligned(header as *const [i32; 3]) };
        let address = |i: usize| header.wrapping_add_signed(offsets[i] as isize);
        if !(address(0)..address(1)).contains(&pc) {
            return false;
        }
        // SAFETY: within this range scratch is this thread's live arena record.
        let record = unsafe { &mut *(regs.scratch as *mut Record) };
        record.raw = Some(raw);
        regs.pc = address(2) as u64;
        true
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
