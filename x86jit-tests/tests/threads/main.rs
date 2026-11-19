//! Threads, memory ordering, self-modifying and cross-modifying code, and
//! background tier-up.
//!
//! One module per former test binary; a test's path starts with its module.

// Each module keeps the `#[path]` helpers of its former binary, so a helper
// that several modules include is compiled once per module, as before.
#![allow(clippy::duplicate_mod)]

mod bg_tier;
mod cross_modifying;
mod fences;
mod interp_ordering;
mod mt;
mod mt_atomic_store_coherence;
mod same_block_smc;
mod smc;
mod threads;
mod tso;
