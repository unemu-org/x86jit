//! Test-only signal adapter shared by compiled-frame and guest recovery tests.
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use x86jit_core::jit_abi::fault::{classify, HostFaultKind, HostRegisters};
// Darwin SDK 14.4 sys/signal.h and Linux v6.12 include/uapi/asm-generic/siginfo.h.
// libc does not expose these names on every supported host.
pub const SEGV_MAPERR: libc::c_int = 1;
pub const SEGV_ACCERR: libc::c_int = 2;
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
) -> bool {
    IN_CLASSIFIER.store(true, Ordering::Relaxed);
    let result = classify(handle, pc, addr, kind, regs);
    IN_CLASSIFIER.store(false, Ordering::Relaxed);
    result
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

unsafe fn recover(context: *mut libc::c_void, addr: usize, kind: HostFaultKind) -> bool {
    let context = &mut *context.cast::<libc::ucontext_t>();
    let mut regs = HostRegisters::default();
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
        && (checked_classify(foreign, regs.pc as usize, addr, kind, &mut regs) || regs != original)
    {
        libc::_exit(93);
    }
    if !checked_classify(
        HANDLE.load(Ordering::Relaxed),
        regs.pc as usize,
        addr,
        kind,
        &mut regs,
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
    let recipe = x86jit_core::codemap::lookup_recovery(original.pc as usize)
        .unwrap()
        .1;
    SIMD_SLOTS.store(recipe.simd.iter().flatten().count(), Ordering::Relaxed);
    LR_CHANGED.store(original.lr != regs.lr, Ordering::Relaxed);
    WIDTH.store(
        x86jit_core::codemap::lookup_trap(original.pc as usize).map_or(0, |s| s.width as usize),
        Ordering::Relaxed,
    );
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
        if recover(
            context,
            if kind == HostFaultKind::Unsupported {
                0
            } else {
                (*info).si_addr() as usize
            },
            kind,
        ) {
            return;
        }
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
