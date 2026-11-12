//! Host FP control transitions at the compiled-code boundary (§8.2.1).
//!
//! The dispatcher retains guest control across integer-only block transfers.
//! Compilation, interpretation and FP-using Rust-helper calls enter the default
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
    core::arch::naked_asm!(
        "mrs x9, fpcr",
        "cmp x9, x0",
        "b.eq 2f",
        "msr fpcr, x0",
        "2:",
        "ret"
    );
}

/// Assembly-only write after a cached comparison proved the target differs.
///
/// # Safety
/// Same environment and exception-mask requirements as [`set_control`]. The
/// caller must compare its known current value first to avoid redundant writes.
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
pub unsafe extern "C" fn set_control_changed(_control: u64) {
    core::arch::naked_asm!("msr fpcr, x0", "ret");
}

#[cfg(not(target_arch = "aarch64"))]
pub use set_control as set_control_changed;

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
/// exceptions, and the entry must restore default control around FP-using helpers.
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
        "mrs x9, fpcr", "str x9, [sp, #16]",
        "cmp x9, x3", "b.eq 3f", "msr fpcr, x3", "3:",
        "blr x2", "cmp x0, #{fault}", "b.eq 2f",
        "ldr x9, [sp, #16]", "mrs x10, fpcr", "cmp x9, x10", "b.eq 2f", "msr fpcr, x9", "2:",
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

/// Temporarily enter Rust's default environment before reaching embedder code.
/// Keep this inside cold helper paths so ordinary integer work needs no control read.
pub(crate) struct DefaultEnvironment {
    saved: u64,
}

impl DefaultEnvironment {
    pub(crate) fn enter() -> Self {
        let saved = read_control();
        if saved != default_control() {
            // SAFETY: normalize before allocation or embedder floating-point work.
            unsafe { set_control_changed(default_control()) };
        }
        Self { saved }
    }
}

impl Drop for DefaultEnvironment {
    fn drop(&mut self) {
        if self.saved != default_control() {
            // SAFETY: the saved control came from this thread; the surrounding
            // helper resumes only integer work before returning to compiled code.
            unsafe { set_control_changed(self.saved) };
        }
    }
}

/// Owns the host environment while the dispatcher executes compiled code.
/// Only integer dispatch may run under guest control. Restore default control
/// before compilation, interpretation, or any other Rust floating-point work.
pub(crate) struct RunMode {
    saved: u64,
    active: u32,
    current: u64,
}

impl RunMode {
    pub(crate) fn new() -> Self {
        let saved = read_control();
        let changed = saved != default_control();
        if changed {
            // SAFETY: enter Rust's default environment before any arithmetic.
            unsafe { set_control_changed(default_control()) };
        }
        Self {
            saved,
            active: 0,
            current: default_control(),
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
            if self.current != value {
                unsafe {
                    if value == default_control() {
                        set_control(value);
                    } else {
                        set_control_changed(value);
                    }
                };
                self.current = value;
            }
            self.active = mode;
        }
    }

    #[inline]
    pub(crate) fn default(&mut self) {
        if self.current != default_control() {
            // SAFETY: restore before calling Rust floating-point code.
            // Baseline translations can install a narrower control than the
            // dispatcher's cached guest value. Read before restoring default.
            unsafe { set_control(default_control()) };
            self.current = default_control();
        }
        self.active = 0;
    }

    pub(crate) fn host_fault(&mut self) {
        // The landing stub restored the pre-run control, which need not be
        // default. Forget the active mode so replay and re-entry both reset it.
        self.current = self.saved;
        self.default();
    }
}

impl Drop for RunMode {
    fn drop(&mut self) {
        if self.current != self.saved || self.active != 0 {
            // A compiled entry can narrow FPCR without updating this cache.
            // SAFETY: no guest code survives run, including during unwinding.
            unsafe { set_control(self.saved) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_environment_restores_nested_and_early_return_scopes() {
        fn early_return() -> u64 {
            let _guard = DefaultEnvironment::enter();
            let _nested = DefaultEnvironment::enter();
            read_control()
        }
        for mode in [0x1f80, 0x5f80, 0x9fc0] {
            let saved = read_control();
            let guest = control(mode);
            // SAFETY: only integer work until the guard normalizes control;
            // restore the caller before running assertions.
            unsafe { set_control(guest) };
            let observed = early_return();
            let restored = read_control();
            unsafe { set_control(saved) };
            assert_eq!(observed, default_control());
            assert_eq!(restored, guest);
        }
    }

    #[test]
    fn default_environment_restores_on_unwind() {
        let saved = read_control();
        let guest = control(0x5f80);
        // SAFETY: the scoped guard normalizes before the panic machinery runs.
        unsafe { set_control(guest) };
        let result = std::panic::catch_unwind(|| {
            let _guard = DefaultEnvironment::enter();
            panic!("exercise scoped restoration");
        });
        let restored = read_control();
        unsafe { set_control(saved) };
        assert!(result.is_err());
        assert_eq!(restored, guest);
    }

    #[test]
    fn run_mode_restores_caller_after_a_compiled_entry_narrows_control() {
        let saved = read_control();
        let caller = control(0xbfc0);
        // SAFETY: keep all host traps masked and restore before assertions.
        unsafe {
            set_control(caller);
        }
        {
            let mut mode = RunMode::new();
            mode.compiled(0xbfc0);
            // Model a baseline entry selecting its narrower compilation control.
            unsafe {
                set_control(default_control());
            }
        }
        let restored = read_control();
        unsafe {
            set_control(saved);
        }
        assert_eq!(restored, caller);
    }

    #[test]
    fn run_mode_tracks_control_and_restores_caller() {
        let saved = read_control();
        // SAFETY: mask traps and restore the thread before any assertions.
        unsafe {
            set_control(control(0x9fc0));
        }
        let caller = read_control();
        let observed = {
            let mut mode = RunMode::new();
            let initial = mode.current;
            mode.compiled(0x9fc0);
            let guest = (mode.current, read_control());
            mode.compiled(0x9fc0);
            let repeated = (mode.current, read_control());
            mode.default();
            let default = (mode.current, read_control());
            mode.compiled(0x3f80);
            // Model the landing stub's restoration of the pre-run FPCR.
            unsafe {
                set_control(caller);
            }
            mode.host_fault();
            (
                initial,
                guest,
                repeated,
                default,
                mode.current,
                read_control(),
            )
        };
        let restored = read_control();
        unsafe {
            set_control(saved);
        }
        assert_eq!(observed.0, default_control());
        assert_eq!(observed.1, (control(0x9fc0), control(0x9fc0)));
        assert_eq!(observed.1, observed.2);
        assert_eq!(observed.3, (default_control(), default_control()));
        assert_eq!(observed.4, default_control());
        assert_eq!(observed.5, default_control());
        assert_eq!(restored, caller);
    }
}
