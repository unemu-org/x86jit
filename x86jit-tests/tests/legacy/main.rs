//! Legacy execution: the CPU models and the Machine profile (every `legacy_*` witness).
//!
//! One module per former test binary; a test's path starts with its module.

// Each module keeps the `#[path]` helpers of its former binary, so a helper
// that several modules include is compiled once per module, as before.
#![allow(clippy::duplicate_mod)]

mod legacy_a20;
mod legacy_alias;
mod legacy_bus_width;
mod legacy_contract;
mod legacy_debug_test;
mod legacy_delivery_liveness;
mod legacy_delivery_log;
mod legacy_direct;
mod legacy_direct_parity;
mod legacy_i286;
mod legacy_i286_loadall;
mod legacy_i286_quirks;
mod legacy_i386_reset;
mod legacy_i8086;
mod legacy_idt;
mod legacy_ldt;
mod legacy_model_decode;
mod legacy_nec_v20;
mod legacy_nec_v20_bcd_overlap;
mod legacy_nec_v20_extensions;
mod legacy_privilege;
mod legacy_protected;
mod legacy_real;
mod legacy_real_exceptions;
mod legacy_real_parity;
mod legacy_real_sp_wrap;
mod legacy_rep_fault_flags;
mod legacy_rollback_sweep;
mod legacy_sse_integer;
mod legacy_task_registers;
mod legacy_task_switch;
mod legacy_u3_review;
mod legacy_upper_audit;
