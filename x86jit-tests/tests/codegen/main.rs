//! The JIT and the interpreter: code generation, tiers, the translation cache,
//! guest trap sites, child execution and the architectural tripwires.
//!
//! One module per former test binary; a test's path starts with its module.

mod boundary;
mod cache;
mod child_step;
mod code_exhaustion;
mod codemap;
mod cpu_source;
mod environment_boundary;
mod guest_trap_sites;
mod interpreter;
mod jit;
mod lockstep_tool;
mod real16_jit_guard;
mod semantics_regressions;
mod smoke;
mod source_ids;
mod superblock;
