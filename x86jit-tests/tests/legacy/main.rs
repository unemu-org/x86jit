//! Legacy execution: the CPU models and the Machine profile (every `legacy_*` witness).
//!
//! One module per former test binary; a test's path starts with its module.

// Each module keeps the `#[path]` helpers of its former binary, so a helper
// that several modules include is compiled once per module, as before.
#![allow(clippy::duplicate_mod)]

mod legacy_a20;
mod legacy_alias;
mod legacy_boundary_events;
mod legacy_bus_width;
mod legacy_census_386;
mod legacy_code_alias_smc;
mod legacy_contract;
mod legacy_debug_test;
mod legacy_delivery_liveness;
mod legacy_delivery_log;
mod legacy_direct;
mod legacy_direct_parity;
mod legacy_flat32;
mod legacy_fused_stack;
mod legacy_i286;
mod legacy_i286_loadall;
mod legacy_i286_quirks;
mod legacy_i386_integer;
mod legacy_i386_reset;
mod legacy_i386_x87;
mod legacy_i486;
mod legacy_i8086;
mod legacy_idt;
mod legacy_io_bitmap;
mod legacy_ldt;
mod legacy_model_decode;
mod legacy_nec_v20;
mod legacy_nec_v20_bcd_overlap;
mod legacy_nec_v20_extensions;
mod legacy_one_pass;
mod legacy_paging;
mod legacy_privilege;
mod legacy_protected;
mod legacy_real;
mod legacy_real_exceptions;
mod legacy_real_parity;
mod legacy_real_sp_wrap;
mod legacy_redundant_prefix;
mod legacy_relift_diag;
mod legacy_rep_fault_flags;
mod legacy_rollback_sweep;
mod legacy_seg16;
mod legacy_seg286;
mod legacy_seg32;
mod legacy_segment_blocks;
mod legacy_smc_positions;
mod legacy_smc_selective;
mod legacy_sse_integer;
mod legacy_task_registers;
mod legacy_task_switch;
mod legacy_tf_loop;
mod legacy_u3_review;
mod legacy_upper_audit;
mod legacy_v86;
mod legacy_v86_round_trip;
mod legacy_x87_absent;
mod legacy_x87_bcd;
mod machine_j1_lockstep;
