//! Host FP control transitions at the compiled-code boundary (§8.2.1).
//!
//! The dispatcher retains guest control across integer-only block transfers.
//! Compilation, interpretation and generated Rust-helper calls enter the default
//! environment first. A run-scoped guard restores the caller on exit or unwind.

pub const MODE_MASK: u32 = crate::interp::softfloat::MODE_MASK;

#[cfg(target_arch = "aarch64")]
pub fn has_afp() -> bool {
    static AFP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AFP.get_or_init(|| {
        let saved: u64;
        let probe: u64;
        // SAFETY: Arm ARM DDI0487J.a §C5.2.7, FPCR.AH/FIZ are RES0 without FEAT_AFP.
        // Probe only those bits and restore the thread's control in the same asm.
        unsafe {
            core::arch::asm!(
                "mrs {saved}, fpcr", "orr {probe}, {saved}, #3",
                "msr fpcr, {probe}", "mrs {probe}, fpcr", "msr fpcr, {saved}",
                saved = out(reg) saved, probe = out(reg) probe,
                options(nomem, nostack),
            );
        }
        let _ = saved;
        probe & 3 == 3
    })
}

#[cfg(not(target_arch = "aarch64"))]
pub fn has_afp() -> bool {
    false
}

pub fn control(mode: u32) -> u64 {
    control_with_afp(mode, has_afp())
}

/// Host control for an explicitly selected compilation target.
pub fn control_with_afp(mode: u32, afp: bool) -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        // Arm ARM DDI0487J.a §C5.2.7, FPCR.RMode: RP and RM swap the x86 RC encodings
        // (SDM Vol 1 §4.8.4). AH selects after-rounding tininess and makes FZ
        // output-only; FIZ independently selects input flushing.
        let rc = (mode >> 13) & 3;
        let rc = ((rc & 1) << 1) | ((rc & 2) >> 1);
        let mut fpcr = u64::from(rc) << 22;
        if afp {
            fpcr |= 2 | u64::from(mode & 0x40 != 0);
            fpcr |= u64::from(mode & 0x8000 != 0) << 24;
        }
        fpcr
    }
    #[cfg(target_arch = "x86_64")]
    {
        let _ = afp;
        u64::from(0x1f80 | mode & MODE_MASK)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = (mode, afp);
        0
    }
}

pub const fn default_control() -> u64 {
    if cfg!(target_arch = "x86_64") {
        0x1f80
    } else {
        0
    }
}

/// Assembly-only control write, for calls emitted by the JIT.
///
/// # Safety
/// No Rust floating-point operation may run until default control is restored.
/// Callers must supply a valid control value with host exceptions masked.
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
pub unsafe extern "C" fn set_control(_control: u64) {
    core::arch::naked_asm!("msr fpcr, x0", "ret");
}

/// # Safety
/// See the AArch64 `set_control` contract.
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
pub unsafe extern "C" fn set_control(_control: u64) {
    core::arch::naked_asm!(
        "sub rsp, 8",
        "mov [rsp], edi",
        "ldmxcsr [rsp]",
        "add rsp, 8",
        "ret"
    );
}

/// # Safety
/// Same pointer/lifetime requirements as `call_block`. `control` must mask host
/// exceptions, and the entry must restore default control around Rust helpers.
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
pub(super) unsafe extern "C" fn call(
    _cpu: *mut u8,
    _ctx: *mut u8,
    _entry: *const u8,
    _control: u64,
) -> u64 {
    core::arch::naked_asm!(
        "stp x29, x30, [sp, #-32]!", "mov x29, sp",
        "mrs x9, fpcr", "str x9, [sp, #16]", "msr fpcr, x3",
        "blr x2", "cmp x0, #{fault}", "b.eq 2f", "ldr x9, [sp, #16]", "msr fpcr, x9", "2:",
        "ldp x29, x30, [sp], #32", "ret",
        fault = const super::RET_HOST_FAULT,
    );
}

/// # Safety
/// Same contract as the AArch64 `call` entry.
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
pub(super) unsafe extern "C" fn call(
    _cpu: *mut u8,
    _ctx: *mut u8,
    _entry: *const u8,
    _control: u64,
) -> u64 {
    core::arch::naked_asm!(
        "sub rsp, 24", "stmxcsr [rsp]", "mov [rsp+8], ecx", "ldmxcsr [rsp+8]",
        "call rdx", "cmp rax, {fault}", "je 2f", "ldmxcsr [rsp]", "2:", "add rsp, 24", "ret",
        fault = const super::RET_HOST_FAULT,
    );
}

/// Read the calling thread's control for run and standalone-call restoration.
#[inline]
fn read_control() -> u64 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let value;
        core::arch::asm!("mrs {}, fpcr", out(reg) value, options(nomem, nostack));
        value
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let mut value = 0u32;
        core::arch::asm!("stmxcsr [{}]", in(reg) &mut value, options(nostack));
        u64::from(value)
    }
}

/// Owns the host environment while the dispatcher executes compiled code.
/// Only integer dispatch may run under guest control. Restore default control
/// before compilation, interpretation, or any other Rust floating-point work.
pub(crate) struct RunMode {
    saved: u64,
    active: u32,
    changed: bool,
}

impl RunMode {
    pub(crate) fn new() -> Self {
        let saved = read_control();
        let changed = saved != default_control();
        if changed {
            // SAFETY: enter Rust's default environment before any arithmetic.
            unsafe { set_control(default_control()) };
        }
        Self {
            saved,
            active: 0,
            changed,
        }
    }

    #[inline]
    pub(crate) fn compiled(&mut self, mxcsr: u32) {
        let mode = mxcsr & MODE_MASK;
        if self.active != mode {
            let value = if mode == 0 {
                default_control()
            } else {
                control(mode)
            };
            // SAFETY: until default() only generated code and integer dispatch
            // execute; generated helpers bracket their own default environment.
            unsafe { set_control(value) };
            self.active = mode;
            self.changed = true;
        }
    }

    #[inline]
    pub(crate) fn default(&mut self) {
        if self.active != 0 {
            // SAFETY: restore before calling Rust floating-point code.
            unsafe { set_control(default_control()) };
            self.active = 0;
        }
    }

    pub(crate) fn host_fault(&mut self) {
        // The landing stub restored the pre-run control, which need not be
        // default. Forget the active mode so replay and re-entry both reset it.
        unsafe { set_control(default_control()) };
        self.active = 0;
        self.changed = true;
    }
}

impl Drop for RunMode {
    fn drop(&mut self) {
        if self.changed {
            // SAFETY: no guest code survives run, including during unwinding.
            unsafe { set_control(self.saved) };
        }
    }
}
