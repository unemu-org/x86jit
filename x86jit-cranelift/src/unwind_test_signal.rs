//! Test-only signal adapter shared by compiled-frame and guest recovery tests.
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use x86jit_core::jit_abi::fault::{classify_with_raw_fault, HostFaultKind, HostRegisters};
use x86jit_core::{HostFaultPlatform, RawHostFault};

#[path = "unwind_test_fault_context.rs"]
mod fault_context;
// Darwin SDK 14.4 sys/signal.h, Linux v6.12 include/uapi/asm-generic/siginfo.h,
// and FreeBSD 14.3 sys/sys/signal.h (not pinned in oracles/).
// libc does not expose these names on every supported host.
pub const SEGV_MAPERR: libc::c_int = 1;
pub const SEGV_ACCERR: libc::c_int = 2;
// FreeBSD 14.3 sys/vm/vm_fault.c, vm_fault_trap: compatibility protection faults
// use UCODE_PAGEFLT=12. This is not a BUS_* name in sys/sys/signal.h.
#[cfg(any(test, target_os = "freebsd"))]
const FREEBSD_PAGE_FAULT: libc::c_int = 12;
#[cfg(all(target_os = "freebsd", target_arch = "aarch64"))]
const _: () = assert!(std::mem::size_of::<libc::sigset_t>() == 16);

// FreeBSD 14.3 sys/arm64/include/ucontext.h, gpregs/fpregs/__mcontext.
// These host sources are not pinned in oracles/. libc 0.2.186 declares fp_q as
// one u128 rather than [u128; 32], shifting mc_flags and truncating mcontext_t.
// Use the OS layout for the signal image, never libc's truncated FP storage.
#[cfg(any(test, all(target_os = "freebsd", target_arch = "aarch64")))]
mod freebsd_arm64 {
    use super::HostRegisters;

    #[repr(C)]
    pub struct GpRegs {
        pub gp_x: [u64; 30],
        pub gp_lr: u64,
        pub gp_sp: u64,
        pub gp_elr: u64,
        pub gp_spsr: u64,
    }

    #[repr(C)]
    pub struct FpRegs {
        pub fp_q: [u128; 32],
        pub fp_sr: u32,
        pub fp_cr: u32,
        pub fp_flags: i32,
        pub fp_pad: i32,
    }

    #[repr(C)]
    pub struct Mcontext {
        pub mc_gpregs: GpRegs,
        pub mc_fpregs: FpRegs,
        pub mc_flags: i32,
        pub mc_pad: i32,
        pub mc_ptr: u64,
        pub mc_spare: [u64; 7],
    }

    impl Mcontext {
        pub fn read(&self) -> Option<HostRegisters> {
            // _MC_FP_VALID: the compiler may need any of d8-d15 restored.
            if self.mc_flags & 1 == 0 {
                return None;
            }
            let g = &self.mc_gpregs;
            Some(HostRegisters {
                pc: g.gp_elr,
                sp: g.gp_sp,
                fp: g.gp_x[29],
                lr: g.gp_lr,
                scratch: g.gp_x[9],
                callee_saved: g.gp_x[19..29].try_into().unwrap(),
                simd: std::array::from_fn(|i| self.mc_fpregs.fp_q[8 + i] as u64),
            })
        }

        pub fn write(&mut self, regs: &HostRegisters) {
            let g = &mut self.mc_gpregs;
            g.gp_elr = regs.pc;
            g.gp_sp = regs.sp;
            g.gp_x[29] = regs.fp;
            g.gp_lr = regs.lr;
            g.gp_x[9] = regs.scratch;
            g.gp_x[19..29].copy_from_slice(&regs.callee_saved);
            for (q, low) in self.mc_fpregs.fp_q[8..16].iter_mut().zip(regs.simd) {
                *q = (*q & (!0u128 << 64)) | u128::from(low);
            }
        }
    }

    #[test]
    fn freebsd_arm64_context_preserves_unmodified_registers_and_simd_halves() {
        use std::mem::{align_of, offset_of, size_of};
        assert_eq!(size_of::<GpRegs>(), 272);
        assert_eq!(size_of::<FpRegs>(), 528);
        assert_eq!(offset_of!(Mcontext, mc_fpregs), 272);
        assert_eq!(offset_of!(Mcontext, mc_flags), 800);
        assert_eq!(size_of::<Mcontext>(), 880);
        assert_eq!(align_of::<Mcontext>(), 16);
        // Distinct words witness every selected slot and every untouched byte.
        let mut words: [u128; 55] = std::array::from_fn(|i| {
            (u128::from(0xfeed_0000u64 + i as u64) << 64) | (0xdead_0000 + i as u128)
        });
        // SAFETY: the aligned array covers the entire OS layout; all fields are integers.
        let ctx = unsafe { &mut *words.as_mut_ptr().cast::<Mcontext>() };
        ctx.mc_flags = 1;
        let original = ctx.read().unwrap();
        assert_eq!(original.pc, ctx.mc_gpregs.gp_elr);
        assert_eq!(original.sp, ctx.mc_gpregs.gp_sp);
        assert_eq!(original.fp, ctx.mc_gpregs.gp_x[29]);
        assert_eq!(original.lr, ctx.mc_gpregs.gp_lr);
        assert_eq!(original.scratch, ctx.mc_gpregs.gp_x[9]);
        assert_eq!(original.callee_saved, ctx.mc_gpregs.gp_x[19..29]);
        for i in 0..8 {
            assert_eq!(original.simd[i], 0xdead_0000 + 17 + 8 + i as u64);
        }
        ctx.mc_flags = 0;
        assert_eq!(ctx.read(), None);
        ctx.mc_flags = 1;
        let before = words;
        let changed = HostRegisters {
            pc: 0x101,
            sp: 0x102,
            fp: 0x103,
            lr: 0x104,
            scratch: 0x105,
            callee_saved: std::array::from_fn(|i| 0x200 + i as u64),
            simd: std::array::from_fn(|i| 0x300 + i as u64),
        };
        // SAFETY: same aligned, initialized signal image as above.
        let ctx = unsafe { &mut *words.as_mut_ptr().cast::<Mcontext>() };
        ctx.write(&changed);
        assert_eq!(ctx.read(), Some(changed));
        ctx.write(&original);
        assert_eq!(words, before);
    }
}
pub static LEGACY_CLASSIFIER: AtomicBool = AtomicBool::new(false);
pub static IN_CLASSIFIER: AtomicBool = AtomicBool::new(false);
pub static SIMD_SLOTS: AtomicUsize = AtomicUsize::new(0);
pub static LR_CHANGED: AtomicBool = AtomicBool::new(false);
pub static ACCESS_FAULTS: AtomicUsize = AtomicUsize::new(0);
pub static ALIGNMENTS: AtomicUsize = AtomicUsize::new(0);
pub unsafe fn checked_classify(
    handle: usize,
    pc: usize,
    addr: usize,
    kind: HostFaultKind,
    regs: &mut HostRegisters,
    raw: RawHostFault,
) -> bool {
    if LEGACY_CLASSIFIER.load(Ordering::Relaxed) {
        x86jit_core::jit_abi::fault::classify(handle, pc, addr, kind, regs)
    } else {
        classify_with_raw_fault(handle, pc, addr, kind, regs, raw)
    }
}

pub static HANDLE: AtomicUsize = AtomicUsize::new(0);
pub static FOREIGN: AtomicUsize = AtomicUsize::new(0);
pub static FAULTS: AtomicUsize = AtomicUsize::new(0);
pub static WIDTH: AtomicUsize = AtomicUsize::new(0);
// The adapters intentionally live in tests: core neither depends on libc nor
// knows an OS ucontext layout. Every write matches a field read below.
// Linux v6.12 arch/arm64/include/uapi/asm/sigcontext.h: __reserved is
// 4096 bytes aligned to 16; FPSIMD_MAGIC records contain vregs at byte 16.
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
unsafe fn linux_fpsimd(context: *mut libc::ucontext_t) -> Option<*mut u128> {
    let s = &mut (*context).uc_mcontext;
    let start = (std::ptr::addr_of_mut!(s.pstate).add(1) as usize + 15) & !15;
    let mut offset = 0;
    while offset + 8 <= 4096 {
        let head = (start + offset) as *const u32;
        let magic = head.read();
        let size = head.add(1).read() as usize;
        if size < 16 || size & 15 != 0 || size > 4096 - offset {
            return None;
        }
        if magic == 0x4650_8001 {
            return (size >= 528).then_some((start + offset + 16) as *mut u128);
        }
        offset += size;
    }
    None
}

unsafe fn recover(
    context: *mut libc::c_void,
    addr: usize,
    kind: HostFaultKind,
    raw: RawHostFault,
) -> bool {
    // Keep the kernel's allocation provenance: arm64 libc's mcontext is too small.
    // FreeBSD 14.3 sys/sys/_ucontext.h places mcontext after the 16-byte sigset
    // (sys/sys/_sigset.h); its required alignment is 16.
    #[cfg(all(target_os = "freebsd", target_arch = "aarch64"))]
    let context = &mut *context
        .byte_add(std::mem::size_of::<libc::sigset_t>())
        .cast::<freebsd_arm64::Mcontext>();
    #[cfg(not(all(target_os = "freebsd", target_arch = "aarch64")))]
    let context = &mut *context.cast::<libc::ucontext_t>();
    #[cfg(all(target_os = "freebsd", target_arch = "aarch64"))]
    let Some(mut regs) = context.read() else {
        return false;
    };
    #[cfg(not(all(target_os = "freebsd", target_arch = "aarch64")))]
    let mut regs = HostRegisters::default();
    #[cfg(all(target_os = "freebsd", target_arch = "x86_64"))]
    {
        // FreeBSD 14.3 sys/x86/include/ucontext.h, __mcontext (amd64).
        let s = &context.uc_mcontext;
        regs.pc = s.mc_rip as u64;
        regs.sp = s.mc_rsp as u64;
        regs.fp = s.mc_rbp as u64;
        regs.scratch = s.mc_r11 as u64;
        regs.callee_saved[..5].copy_from_slice(&[
            s.mc_rbx as u64,
            s.mc_r12 as u64,
            s.mc_r13 as u64,
            s.mc_r14 as u64,
            s.mc_r15 as u64,
        ]);
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        let g = &context.uc_mcontext.gregs;
        regs.pc = g[libc::REG_RIP as usize] as u64;
        regs.sp = g[libc::REG_RSP as usize] as u64;
        regs.fp = g[libc::REG_RBP as usize] as u64;
        regs.scratch = g[libc::REG_R11 as usize] as u64;
        for (i, r) in [
            libc::REG_RBX,
            libc::REG_R12,
            libc::REG_R13,
            libc::REG_R14,
            libc::REG_R15,
        ]
        .into_iter()
        .enumerate()
        {
            regs.callee_saved[i] = g[r as usize] as u64;
        }
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        let s = &context.uc_mcontext;
        regs.pc = s.pc;
        regs.sp = s.sp;
        regs.fp = s.regs[29];
        regs.lr = s.regs[30];
        regs.scratch = s.regs[9];
        regs.callee_saved.copy_from_slice(&s.regs[19..29]);
        let Some(v) = linux_fpsimd(context) else {
            return false;
        };
        for i in 0..8 {
            regs.simd[i] = v.add(8 + i).read() as u64;
        }
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let s = &(*context.uc_mcontext).__ss;
        regs.pc = s.__pc;
        regs.sp = s.__sp;
        regs.fp = s.__fp;
        regs.lr = s.__lr;
        regs.scratch = s.__x[9];
        let v = &(*context.uc_mcontext).__ns.__v;
        for i in 0..8 {
            regs.simd[i] = v[8 + i] as u64;
        }
        regs.callee_saved.copy_from_slice(&s.__x[19..29]);
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        let s = &(*context.uc_mcontext).__ss;
        regs.pc = s.__rip;
        regs.sp = s.__rsp;
        regs.fp = s.__rbp;
        regs.scratch = s.__r11;
        regs.callee_saved[..5].copy_from_slice(&[s.__rbx, s.__r12, s.__r13, s.__r14, s.__r15]);
    }
    // Inactive/foreign handles and unrelated PCs must leave the context alone.
    let original = regs;
    let foreign = FOREIGN.load(Ordering::Relaxed);
    if foreign != 0
        && x86jit_core::codemap::lookup_recovery(regs.pc as usize).is_some()
        && (checked_classify(foreign, regs.pc as usize, addr, kind, &mut regs, raw)
            || regs != original)
    {
        libc::_exit(93);
    }
    if !checked_classify(
        HANDLE.load(Ordering::Relaxed),
        regs.pc as usize,
        addr,
        kind,
        &mut regs,
        raw,
    ) {
        return false;
    }
    FAULTS.fetch_add(1, Ordering::Relaxed);
    ACCESS_FAULTS.fetch_add(
        usize::from(kind == HostFaultKind::Access),
        Ordering::Relaxed,
    );
    ALIGNMENTS.fetch_add(
        usize::from(kind == HostFaultKind::Alignment),
        Ordering::Relaxed,
    );
    let slots = x86jit_core::codemap::lookup_recovery(original.pc as usize)
        .map_or(0, |(_, recipe)| recipe.simd.iter().flatten().count());
    SIMD_SLOTS.store(slots, Ordering::Relaxed);
    LR_CHANGED.store(original.lr != regs.lr, Ordering::Relaxed);
    WIDTH.store(
        x86jit_core::codemap::lookup_trap(original.pc as usize).map_or(0, |s| s.width as usize),
        Ordering::Relaxed,
    );
    #[cfg(all(target_os = "freebsd", target_arch = "x86_64"))]
    {
        let s = &mut context.uc_mcontext;
        s.mc_rip = regs.pc as libc::register_t;
        s.mc_rsp = regs.sp as libc::register_t;
        s.mc_rbp = regs.fp as libc::register_t;
        s.mc_r11 = regs.scratch as libc::register_t;
        [s.mc_rbx, s.mc_r12, s.mc_r13, s.mc_r14, s.mc_r15] =
            std::array::from_fn(|i| regs.callee_saved[i] as libc::register_t);
    }
    #[cfg(all(target_os = "freebsd", target_arch = "aarch64"))]
    context.write(&regs);
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        let g = &mut context.uc_mcontext.gregs;
        g[libc::REG_RIP as usize] = regs.pc as libc::greg_t;
        g[libc::REG_RSP as usize] = regs.sp as libc::greg_t;
        g[libc::REG_RBP as usize] = regs.fp as libc::greg_t;
        g[libc::REG_R11 as usize] = regs.scratch as libc::greg_t;
        for (i, r) in [
            libc::REG_RBX,
            libc::REG_R12,
            libc::REG_R13,
            libc::REG_R14,
            libc::REG_R15,
        ]
        .into_iter()
        .enumerate()
        {
            g[r as usize] = regs.callee_saved[i] as libc::greg_t;
        }
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        let s = &mut context.uc_mcontext;
        s.pc = regs.pc;
        s.sp = regs.sp;
        s.regs[29] = regs.fp;
        s.regs[30] = regs.lr;
        s.regs[9] = regs.scratch;
        s.regs[19..29].copy_from_slice(&regs.callee_saved);
        let v = linux_fpsimd(context).unwrap();
        for i in 0..8 {
            let p = v.add(8 + i);
            p.write((p.read() & (!0u128 << 64)) | u128::from(regs.simd[i]));
        }
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let s = &mut (*context.uc_mcontext).__ss;
        s.__pc = regs.pc;
        s.__sp = regs.sp;
        s.__fp = regs.fp;
        s.__lr = regs.lr;
        s.__x[9] = regs.scratch;
        let v = &mut (*context.uc_mcontext).__ns.__v;
        for i in 0..8 {
            v[8 + i] = (v[8 + i] & (!0u128 << 64)) | u128::from(regs.simd[i]);
        }
        s.__x[19..29].copy_from_slice(&regs.callee_saved);
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        let s = &mut (*context.uc_mcontext).__ss;
        s.__rip = regs.pc;
        s.__rsp = regs.sp;
        s.__rbp = regs.fp;
        s.__r11 = regs.scratch;
        [s.__rbx, s.__r12, s.__r13, s.__r14, s.__r15] = regs.callee_saved[..5].try_into().unwrap();
    }
    true
}

// POSIX.1-2024 sigaction, signal-specific si_code values. Only synchronous
// memory faults enter recovery; SI_USER/SI_QUEUE and unsupported causes do not.
#[cfg(not(target_os = "freebsd"))]
pub fn fault_kind(signal: libc::c_int, code: libc::c_int) -> HostFaultKind {
    if code <= 0 {
        return HostFaultKind::Unsupported;
    }
    match (signal, code) {
        (libc::SIGSEGV, SEGV_MAPERR | SEGV_ACCERR) | (libc::SIGBUS, libc::BUS_ADRERR) => {
            HostFaultKind::Access
        }
        (libc::SIGBUS, libc::BUS_ADRALN) => HostFaultKind::Alignment,
        _ => HostFaultKind::Unsupported,
    }
}

#[cfg(any(test, target_os = "freebsd"))]
fn freebsd_fault_kind(signal: libc::c_int, code: libc::c_int) -> HostFaultKind {
    match (signal, code) {
        (libc::SIGSEGV, SEGV_MAPERR | SEGV_ACCERR) | (libc::SIGBUS, FREEBSD_PAGE_FAULT) => {
            HostFaultKind::Access
        }
        // FreeBSD 14.3 sys/sys/signal.h: BUS_ADRALN=1.
        (libc::SIGBUS, 1) => HostFaultKind::Alignment,
        _ => HostFaultKind::Unsupported,
    }
}

#[cfg(target_os = "freebsd")]
pub fn fault_kind(signal: libc::c_int, code: libc::c_int) -> HostFaultKind {
    freebsd_fault_kind(signal, code)
}

#[test]
fn freebsd_fault_codes_are_allowlisted() {
    // FreeBSD 14.3 sys/sys/signal.h and sys/vm/vm_fault.c, vm_fault_trap.
    for (signal, code, expected) in [
        (libc::SIGSEGV, 1, HostFaultKind::Access),
        (libc::SIGSEGV, 2, HostFaultKind::Access),
        (libc::SIGBUS, 12, HostFaultKind::Access),
        (libc::SIGBUS, 1, HostFaultKind::Alignment),
        (libc::SIGBUS, 2, HostFaultKind::Unsupported),
        (libc::SIGBUS, 3, HostFaultKind::Unsupported),
        (libc::SIGBUS, 100, HostFaultKind::Unsupported),
        (libc::SIGSEGV, 100, HostFaultKind::Unsupported),
        (libc::SIGILL, 12, HostFaultKind::Unsupported),
    ] {
        assert_eq!(
            freebsd_fault_kind(signal, code),
            expected,
            "{signal}/{code}"
        );
    }
    // SI_USER, SI_QUEUE, SI_TIMER, SI_ASYNCIO, SI_MESGQ, SI_KERNEL, SI_LWP
    // are positive on FreeBSD; a positive code alone does not authorize recovery.
    for signal in [libc::SIGSEGV, libc::SIGBUS] {
        for code in [
            -1, 0, 4, 0x10001, 0x10002, 0x10003, 0x10004, 0x10005, 0x10006, 0x10007,
        ] {
            assert_eq!(freebsd_fault_kind(signal, code), HostFaultKind::Unsupported);
        }
    }
}

// Arm ARM DDI0487I.a §D17.2.37, ESR_EL1 Data Abort ISS: EC 0x24/0x25;
// DFSC 0x04..0x0f = translation/access-flag/permission, 0x21 = alignment.
#[cfg(target_arch = "aarch64")]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub fn esr_fault_kind(esr: u32) -> HostFaultKind {
    if ![0x24, 0x25].contains(&(esr >> 26)) {
        return HostFaultKind::Unsupported;
    }
    match esr & 0x3f {
        0x21 => HostFaultKind::Alignment,
        0x04..=0x0f => HostFaultKind::Access,
        _ => HostFaultKind::Unsupported,
    }
}

extern "C" fn handler(signal: libc::c_int, info: *mut libc::siginfo_t, context: *mut libc::c_void) {
    // SAFETY: this test installs SA_SIGINFO only in children, with live handles.
    unsafe {
        IN_CLASSIFIER.store(true, Ordering::Relaxed);
        let kind = fault_kind(signal, (*info).si_code);
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let kind = if kind == HostFaultKind::Unsupported {
            kind
        } else {
            // Darwin may report protection failures as BUS_ADRALN too. The
            // saved data-abort syndrome distinguishes them from alignment.
            let esr = (*(*(context as *const libc::ucontext_t)).uc_mcontext)
                .__es
                .__esr;
            esr_fault_kind(esr)
        };
        #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
        let kind = if kind == HostFaultKind::Alignment {
            HostFaultKind::Unsupported
        } else {
            kind
        };
        if kind != HostFaultKind::Unsupported
            && recover(
                context,
                (*info).si_addr() as usize,
                kind,
                fault_context::extract(signal, &*info, context),
            )
        {
            IN_CLASSIFIER.store(false, Ordering::Relaxed);
            return;
        }
        IN_CLASSIFIER.store(false, Ordering::Relaxed);
        // Restore default and return: the synchronous instruction faults again.
        libc::signal(signal, libc::SIG_DFL);
        if kind == HostFaultKind::Unsupported {
            libc::raise(signal);
        }
    }
}

pub unsafe fn install() {
    let mut action: libc::sigaction = std::mem::zeroed();
    action.sa_sigaction = handler as *const () as usize;
    action.sa_flags = libc::SA_SIGINFO;
    libc::sigemptyset(&mut action.sa_mask);
    assert_eq!(
        libc::sigaction(libc::SIGSEGV, &action, std::ptr::null_mut()),
        0
    );
    assert_eq!(
        libc::sigaction(libc::SIGBUS, &action, std::ptr::null_mut()),
        0
    );
}

pub fn child(f: impl FnOnce()) -> i32 {
    // SAFETY: signal installation and deliberate host faults stay in the child.
    unsafe {
        let pid = libc::fork();
        assert!(pid >= 0);
        if pid == 0 {
            libc::alarm(15);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
            libc::_exit(i32::from(result.is_err()));
        }
        let mut status = 0;
        assert_eq!(libc::waitpid(pid, &mut status, 0), pid);
        status
    }
}
