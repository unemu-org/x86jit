//! Host ABI extraction for the example/test signal adapter. Core accepts the
//! resulting fixed-size record and does not depend on libc or decode host ABIs.
use super::{HostFaultPlatform, RawHostFault};

// glibc sys/ucontext.h (x86-64): REG_ERR=19, REG_TRAPNO=20, REG_CR2=22.
#[cfg(any(test, all(target_os = "linux", target_arch = "x86_64")))]
fn linux_x86(gregs: &[i64; 23]) -> RawHostFault {
    let mut raw = RawHostFault::new(HostFaultPlatform::LinuxX86_64);
    raw.trap_number = Some(gregs[20] as u64);
    raw.error_code = Some(gregs[19] as u64);
    raw.fault_address = Some(gregs[22] as u64);
    raw
}

// FreeBSD 14.3 sys/x86/include/ucontext.h, amd64 __mcontext prefix.
#[cfg(any(test, all(target_os = "freebsd", target_arch = "x86_64")))]
#[repr(C)]
struct FreeBsdX86Context {
    gprs: [u64; 16],
    trapno: u32,
    fs: u16,
    gs: u16,
    addr: u64,
    flags: u32,
    es: u16,
    ds: u16,
    err: u64,
}
#[cfg(any(test, all(target_os = "freebsd", target_arch = "x86_64")))]
fn freebsd_x86(context: &FreeBsdX86Context) -> RawHostFault {
    let mut raw = RawHostFault::new(HostFaultPlatform::FreeBsdX86_64);
    raw.trap_number = Some(u64::from(context.trapno));
    raw.error_code = Some(context.err);
    raw.fault_address = Some(context.addr);
    raw
}

// Linux v6.12 arch/arm64/include/uapi/asm/sigcontext.h: ESR_MAGIC,
// _aarch64_ctx and esr_context. Records terminate at a zero header; reject
// malformed sizes before advancing. ESR is opaque, including all high bits.
#[cfg(any(test, all(target_os = "linux", target_arch = "aarch64")))]
fn linux_esr(reserved: &[u8]) -> Option<u64> {
    let mut offset = 0;
    while offset + 8 <= reserved.len() {
        let magic = u32::from_ne_bytes(reserved[offset..offset + 4].try_into().ok()?);
        let size = u32::from_ne_bytes(reserved[offset + 4..offset + 8].try_into().ok()?) as usize;
        if size < 16 || size & 15 != 0 || size > reserved.len() - offset {
            return None;
        }
        if magic == 0x4553_5201 {
            return Some(u64::from_ne_bytes(
                reserved[offset + 8..offset + 16].try_into().ok()?,
            ));
        }
        offset += size;
    }
    None
}

// SAFETY: called only with the kernel-provided SA_SIGINFO signal image, before
// recovery rewrites it. Every read is of initialized host ABI fields.
pub unsafe fn extract(
    signal: i32,
    info: &libc::siginfo_t,
    context: *mut libc::c_void,
) -> RawHostFault {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    let mut raw = {
        let g = &(*context.cast::<libc::ucontext_t>()).uc_mcontext.gregs;
        linux_x86(g)
    };
    #[cfg(all(target_os = "freebsd", target_arch = "x86_64"))]
    let mut raw = {
        // Pin the portable prefix against the actual libc ABI at compile time.
        const _: () = {
            assert!(std::mem::offset_of!(libc::mcontext_t, mc_trapno) == 128);
            assert!(std::mem::offset_of!(libc::mcontext_t, mc_addr) == 136);
            assert!(std::mem::offset_of!(libc::mcontext_t, mc_err) == 152);
        };
        let m = &(*context.cast::<libc::ucontext_t>()).uc_mcontext;
        freebsd_x86(&*(std::ptr::from_ref(m).cast::<FreeBsdX86Context>()))
    };
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    let mut raw = {
        let m = &(*context.cast::<libc::ucontext_t>()).uc_mcontext;
        let mut raw = RawHostFault::new(HostFaultPlatform::LinuxAarch64);
        raw.fault_address = Some(m.fault_address);
        // libc keeps __reserved private; the Linux ABI aligns its 4096 bytes
        // after pstate to 16, as in the FPSIMD adapter.
        let start = (std::ptr::addr_of!(m.pstate).add(1) as usize + 15) & !15;
        raw.esr = linux_esr(std::slice::from_raw_parts(start as *const u8, 4096));
        raw
    };
    #[cfg(all(target_os = "freebsd", target_arch = "aarch64"))]
    let mut raw = {
        let _ = context;
        // FreeBSD 14.3 sys/arm64/include/ucontext.h exposes GP/FP registers,
        // flags and extension storage, but no ESR or FAR in mcontext.
        RawHostFault::new(HostFaultPlatform::FreeBsdAarch64)
    };
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let mut raw = {
        // Darwin SDK arm/_structs.h, __darwin_arm_exception_state64.
        let m = &*(*context.cast::<libc::ucontext_t>()).uc_mcontext;
        let mut raw = RawHostFault::new(HostFaultPlatform::MacOsAarch64);
        raw.esr = Some(u64::from(m.__es.__esr));
        raw.fault_address = Some(m.__es.__far);
        raw
    };
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    let mut raw = {
        let m = &*(*context.cast::<libc::ucontext_t>()).uc_mcontext;
        let mut raw = RawHostFault::new(HostFaultPlatform::MacOsX86_64);
        raw.trap_number = Some(u64::from(m.__es.__trapno));
        raw.error_code = Some(u64::from(m.__es.__err));
        raw.fault_address = Some(m.__es.__faultvaddr);
        raw
    };
    raw.signal = Some(signal);
    raw.signal_code = Some(info.si_code);
    raw.signal_address = Some(info.si_addr() as u64);
    raw
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_linux_x86_synthetic_mcontext() {
        let mut gregs = std::array::from_fn(|i| -100 - i as i64);
        gregs[19] = 0x8000_0000_0000_0007u64 as i64;
        gregs[20] = 14;
        gregs[22] = 0xffff_8000_1234_5678u64 as i64;
        let raw = linux_x86(&gregs);
        assert_eq!(raw.platform, HostFaultPlatform::LinuxX86_64);
        assert_eq!(raw.trap_number, Some(14));
        assert_eq!(raw.error_code, Some(0x8000_0000_0000_0007));
        assert_eq!(raw.fault_address, Some(0xffff_8000_1234_5678));
        assert_eq!(raw.esr, None);
        gregs[19] = 0;
        assert_eq!(linux_x86(&gregs).error_code, Some(0));
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            // SAFETY: zero is valid for the libc context's integer/pointer fields.
            let mut uc: libc::ucontext_t = unsafe { std::mem::zeroed() };
            uc.uc_mcontext.gregs = gregs;
            assert_eq!(libc::REG_ERR, 19);
            assert_eq!(libc::REG_TRAPNO, 20);
            assert_eq!(libc::REG_CR2, 22);
            assert_eq!(linux_x86(&uc.uc_mcontext.gregs), linux_x86(&gregs));
        }
    }

    #[test]
    fn raw_freebsd_x86_synthetic_mcontext() {
        let mut image = [0x55aa_0011_2233_4455u64; 20];
        let bytes = image.as_mut_ptr().cast::<u8>();
        // Distinct sentinels in surrounding fields detect offset/width mistakes.
        // SAFETY: image has the size/alignment of the integer-only ABI prefix.
        let m = unsafe { &mut *bytes.cast::<FreeBsdX86Context>() };
        m.trapno = 12;
        m.addr = 0xffff_8000_1122_3344;
        m.err = 0x8000_0000_0000_0005;
        let raw = freebsd_x86(m);
        assert_eq!(std::mem::offset_of!(FreeBsdX86Context, trapno), 128);
        assert_eq!(std::mem::offset_of!(FreeBsdX86Context, addr), 136);
        assert_eq!(std::mem::offset_of!(FreeBsdX86Context, err), 152);
        assert_eq!(raw.platform, HostFaultPlatform::FreeBsdX86_64);
        assert_eq!(raw.trap_number, Some(12));
        assert_eq!(raw.error_code, Some(0x8000_0000_0000_0005));
        assert_eq!(raw.fault_address, Some(0xffff_8000_1122_3344));
        assert_eq!(raw.esr, None);
    }

    #[test]
    fn raw_linux_arm64_esr_reserved_records() {
        let mut bytes = [0u8; 4096];
        // An unrelated context before ESR must be skipped by its declared size.
        bytes[..4].copy_from_slice(&0x4650_8001u32.to_ne_bytes());
        bytes[4..8].copy_from_slice(&528u32.to_ne_bytes());
        bytes[528..532].copy_from_slice(&0x4553_5201u32.to_ne_bytes());
        bytes[532..536].copy_from_slice(&16u32.to_ne_bytes());
        bytes[536..544].copy_from_slice(&0xffff_aaaa_1234_5678u64.to_ne_bytes());
        assert_eq!(linux_esr(&bytes), Some(0xffff_aaaa_1234_5678));
        for size in [0u32, 8, 17, 4096, u32::MAX] {
            bytes[532..536].copy_from_slice(&size.to_ne_bytes());
            assert_eq!(linux_esr(&bytes), None);
        }
        assert_eq!(linux_esr(&[0; 4096]), None);
        assert_eq!(linux_esr(&[]), None);
        bytes[532..536].copy_from_slice(&16u32.to_ne_bytes());
        bytes[536..544].fill(0);
        assert_eq!(linux_esr(&bytes), Some(0));
    }
}
