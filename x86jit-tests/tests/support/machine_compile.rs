//! Decision-30 §10: run a Machine witness file with compiled Flat32 blocks.
//! `X86JIT_TEST_MACHINE_COMPILE=forced` compiles every eligible block at its
//! first execution on the Cranelift backend; `=background` uses the
//! production configuration (hotness-gated background tier-up at T1 = 1).
//! Unset, the witness keeps the backend it names.
use x86jit_core::{Backend, MachineJitMode, Vm};

/// The compile mode the environment selects, if any.
pub fn mode() -> Option<&'static str> {
    match std::env::var("X86JIT_TEST_MACHINE_COMPILE").as_deref() {
        Ok("forced") => Some("forced"),
        Ok("background") => Some("background"),
        _ => None,
    }
}

/// `fallback` unless a compile mode is selected, then the Cranelift backend.
pub fn backend(fallback: Box<dyn Backend>) -> Box<dyn Backend> {
    if mode().is_some() {
        Box::new(x86jit_cranelift::JitBackend::new())
    } else {
        fallback
    }
}

/// Apply the selected compile mode to a VM built on [`backend`].
pub fn configure(vm: &mut Vm) {
    match mode() {
        Some("forced") => vm.set_machine_jit_for_test(MachineJitMode::Forced),
        Some("background") => {
            vm.set_tier_up_after(Some(1));
            vm.set_tier_up_background(true);
        }
        _ => {}
    }
}
