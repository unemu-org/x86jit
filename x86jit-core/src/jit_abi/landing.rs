//! Mangled, per-crate-instance landing stubs for compiled fault returns (§8.2).

#[cfg(all(target_arch = "aarch64", any(target_os = "linux", target_os = "macos")))]
#[unsafe(naked)]
pub(super) unsafe extern "C" fn landing_stub() {
    core::arch::naked_asm!(
        "ldr x10, [x9]", "msr fpcr, x10", "mov x0, #{fault_code}", "ret",
        fault_code = const super::super::RET_HOST_FAULT,
    );
}

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
#[unsafe(naked)]
pub(super) unsafe extern "C" fn landing_stub() {
    // Intel SDM Vol 1 §10.2.3; x86-64 psABI §3.2.1: restore host MXCSR control.
    core::arch::naked_asm!(
        "ldmxcsr [r11]", "mov eax, {fault_code}", "ret",
        fault_code = const super::super::RET_HOST_FAULT,
    );
}
