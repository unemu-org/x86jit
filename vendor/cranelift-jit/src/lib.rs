// Modified by the x86jit project. This file is changed from cranelift-jit 0.135.8
// (crates.io, sha256 31d67d6c6a02ffd860599e55b03d4f37c6cf9be678c1d68dbfa6d1f61946e4ae;
// wasmtime commit 7b6c29ec3e9a49c97da231665cbd8eead9ccbda0, path cranelift/jit),
// licensed Apache-2.0 WITH LLVM-exception (LICENSE in this directory).
// Changed: re-exports `SystemMemoryOperation`.
// README.md in this directory lists every changed file.

//! Top-level lib.rs for `cranelift_jit`.
//!
//! There is an [example project](https://github.com/bytecodealliance/cranelift-jit-demo/)
//! which shows how to use some of the features of `cranelift_jit`.

#![deny(missing_docs, unreachable_pub)]
#![expect(unsafe_op_in_unsafe_fn, reason = "crate isn't migrated yet")]

mod backend;
mod compiled_blob;
mod memory;

pub use crate::backend::{JITBuilder, JITModule};
pub use crate::memory::{
    ArenaMemoryProvider, BranchProtection, JITMemoryKind, JITMemoryProvider, SystemMemoryOperation,
    SystemMemoryProvider,
};

/// Version number of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
