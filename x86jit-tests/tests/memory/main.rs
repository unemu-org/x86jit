//! Memory and faults: regions, fetch policy, host faults, fault state and
//! atomicity, locked and misaligned operands, MMIO and port I/O.
//!
//! One module per former test binary; a test's path starts with its module.

// Each module keeps the `#[path]` helpers of its former binary, so a helper
// that several modules include is compiled once per module, as before.
#![allow(clippy::duplicate_mod)]

mod address_exceptions;
mod cmpxchg_pair;
mod cross_region_host_fault;
mod cross_region_ram;
mod fault_atomicity;
mod fault_path;
mod fetch_policy;
mod locked_misaligned;
mod locked_mmio;
mod locked_split_device;
mod mmio_rmw;
mod port_io;
mod port_string;
mod prot_advisory;
mod real16_wrap;
mod region_fault_state;
mod rust_memory_faults;
mod sparse_backing;
mod store_no_read;
mod string_mmio;
mod trap_fetch;
mod vector_mmio_resume;
