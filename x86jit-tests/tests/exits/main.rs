//! Exits and stepping: host requests, deferral, decode faults, retirement,
//! interrupt shadows, vCPU reset and the flags an exit exposes.
//!
//! One module per former test binary; a test's path starts with its module.

mod decode_faults;
mod deferred_exit;
mod exit_request;
mod flags_at_exit;
mod irq_shadow;
mod retired_count;
mod run_steps;
mod syscall_path;
mod vcpu_reset;
