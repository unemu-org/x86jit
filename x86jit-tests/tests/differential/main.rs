//! Differential oracles: Unicorn, the SingleStepTests 80286 corpus, the
//! RON vector corpus, the ISA compat map and the differential fuzz drivers.
//!
//! One module per former test binary; a test's path starts with its module.

mod addr32;
mod cf16;
mod cf32;
mod compat;
mod corpus;
mod coverage_ratchet;
mod differential;
mod differential32;
mod disasm_matches_objdump;
mod fuzz;
mod fuzz_avx;
mod ss286;
