# x86jit-cranelift-jit — a modified fork of cranelift-jit 0.135.8

**This is not the upstream crate.** It is
[`cranelift-jit`](https://crates.io/crates/cranelift-jit) 0.135.8 as modified by
the x86jit project. It lives at `vendor/cranelift-jit` in the x86jit repository
(<https://github.com/unemu-org/x86jit>) and is used only by `x86jit-cranelift`.
The package is named `x86jit-cranelift-jit`; the library keeps the upstream name
`cranelift_jit`.

- **Upstream:** crates.io `cranelift-jit` 0.135.8, sha256
  `31d67d6c6a02ffd860599e55b03d4f37c6cf9be678c1d68dbfa6d1f61946e4ae`, published
  from wasmtime commit `7b6c29ec3e9a49c97da231665cbd8eead9ccbda0`, path
  `cranelift/jit`.
- **Licence:** unchanged, Apache-2.0 WITH LLVM-exception; the full text is in
  [`LICENSE`](LICENSE).
- **Why:** x86jit owns its JIT code memory through a memory provider
  (decision-17 in the x86jit repository's `backlog/decisions/`). `PROVENANCE.md`
  §5 there describes the licence surface.

## Changed files

Every file below was changed by the x86jit project, and each one carries a notice
saying so.

| File | Change |
|---|---|
| `Cargo.toml` | Package name, description and repository; the upstream documentation link is removed. |
| `README.md` | This section, above the upstream text. |
| `src/lib.rs` | Re-exports `SystemMemoryOperation`. |
| `src/backend.rs` | Relocation under the provider's scoped write access; a failed finalization is terminal; retirement of finalized definitions and discard of unpublished ones; allocation-generation queries; code-range entries for functions defined from bytes. |
| `src/compiled_blob.rs` | Separate writable and executable addresses, with relocations resolved against the executable address; unsupported relocations rejected before any copy; copies under scoped write access; release of an allocation whose initialization failed. |
| `src/memory/mod.rs` | `SystemMemoryOperation`; the `JITMemoryProvider` hooks `executable_ptr`, `allocation_generation`, `release_allocation`, `with_write_access` and `supports_relocation`; cache maintenance and RX protection report failures as `ModuleError::Allocation` instead of panicking. |
| `src/memory/system.rs` | `SystemMemoryProvider::finalize_with`, a callback around each protection, cache-maintenance and pipeline-sync operation; those failures are returned as errors instead of panicking. |

Unchanged from upstream: `LICENSE`, `examples/jit-minimal.rs`,
`src/memory/arena.rs` and `tests/basic.rs`.

## Upstream README

This crate provides a JIT library that uses
[Cranelift](https://crates.io/crates/cranelift).

This crate is extremely experimental.

See the [example program] for a brief overview of how to use this.

[example program]: https://github.com/bytecodealliance/wasmtime/blob/main/cranelift/jit/examples/jit-minimal.rs
