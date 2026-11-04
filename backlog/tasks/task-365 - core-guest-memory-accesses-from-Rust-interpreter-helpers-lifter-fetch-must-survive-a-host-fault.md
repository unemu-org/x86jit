---
id: TASK-365
title: >-
  core: guest-memory accesses from Rust (interpreter, helpers, lifter fetch)
  must survive a host fault
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - fault-precision
  - embedder-api
dependencies:
  - TASK-364
priority: high
ordinal: 401000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Under decision-5, an in-span hole faults on a PROT_NONE guard page. The interpreter still traps before it touches host memory, because `region_at` walks the region list first (memory.rs:1256-1278, used by `read` at 1283-1292 and by `write` at 1316-1320). That works only while the region list and the host protections agree.

An embedder that runs a full native userspace keeps the real mappings and protections itself. mmap, mprotect and munmap come from many threads. There are PROT_NONE guard pages below thread stacks, and garbage collectors protect pages to trap writes. Such an embedder registers one RAM region over the whole span. Every guest access made by Rust code then reaches host memory unchecked, and a fault there is a host crash inside Rust:

- `Memory::read`/`write` and the atomic paths, through `atomic_load_le`/`atomic_store_le` (memory.rs:1500-1531);
- the x87 and fxsave paths through `FpMem` (x87.rs:31-45): the interpreter's `Memory` through `read_ram_guest`/`write_ram_guest`, the JIT's helpers through the bounds-only `RawFpMem`;
- the raw view the string-op helpers use (`RawStrMem`, interp/mod.rs:5348), which compiled `rep movs`/`stos` call into;
- the lifter. `code_slice` (memory.rs:1295-1312) returns a `&[u8]` borrowed from live guest memory, and `lift_block` hands it straight to iced (lift/mod.rs:237-242). A fault during decode is a host crash. A concurrent guest store into those bytes from another vcpu is, in Rust's memory model, a data race on a shared slice.

Leaving these with a longjmp is the undefined behaviour TASK-364 removes.

The standard answer is the one OS kernels use for user copies: a few small per-host assembly routines, each with an entry in a fault-fixup table. The set is probe, copy-in, copy-out, a single-copy-atomic 1/2/4/8-byte load and store, and a compare-exchange for the locked paths. When one of them faults, the embedder's handler asks the TASK-364 classifier. The classifier finds the PC in the fixup table and rewrites the context to the routine's fixup label, and the routine returns `Err(fault address)` to its Rust caller. Nothing unwinds and every `Drop` runs normally. Under the commit-last rule (TASK-305) the instruction has committed nothing, so the interpreter returns the same resumable `Exit::UnmappedMemory` the JIT does.

The lifter copies its fetch window into a local buffer through such a routine and decodes the copy. A fault in the copy becomes an instruction-fetch fault at the first unreadable byte, and the private copy also closes the data race. No lock is held while lifting today (`resolve`, vm.rs:1430-1596). Keep it that way, so a fault in the fetch can never strand a lock.

Constraints:
- x86jit-core gains no dependency: `core::arch::global_asm!`/`asm!` only (x86jit-tests/tests/boundary.rs).
- The routines keep the single-copy atomicity that `atomic_load_le`/`atomic_store_le` guarantee for aligned accesses (memory.rs:1500-1508). A byte-wise copy loop would break it.
- There are x86-64 and aarch64 implementations.

Trade-off: the interpreter's scalar access becomes an out-of-line call where today it is an inline atomic. Price it. If it matters, make the routed path an embedder opt-in ("the embedder owns the host protections") and keep `region_at` as the default.

RELATION. TASK-336 (three interpreter handlers still commit before a faulting access) must land for the "committed nothing" half of the guarantee to hold. TASK-337 is the MMIO side of the same handlers. TASK-378 builds the fetch-permission check into the same fetch path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The setup is one region over the whole span with a PROT_NONE page inside it. Each of these, run on the interpreter, returns `Exit::UnmappedMemory` with the faulting byte's address and no committed state: a load, a store, a locked RMW, an x87 store, `fxsave`, a `rep movsb` that crosses into the page, and a 256-bit vector load. At 96fd5c6 the test's forked child crashes instead
- [ ] #2 The same instructions under the JIT, whose helpers now use the routines, give the same exits
- [ ] #3 A block whose bytes run into a PROT_NONE page lifts its valid prefix and reports a fetch fault at the first unreadable byte. The lifter decodes a private copy, and no path hands iced a slice of live guest memory
- [ ] #4 The routines keep single-copy atomicity for aligned 1/2/4/8-byte accesses, and the existing coherence tests (e.g. mt_atomic_store_coherence.rs) pass unchanged
- [ ] #5 There are x86-64 and aarch64 implementations, and both are exercised (the aarch64 CI lane runs #1-#3). x86jit-core's dependency set stays exactly `{iced-x86}`
- [ ] #6 Interpreter throughput with the routed path is measured against bench/baseline.json and recorded on this task. If an opt-in is introduced, spec.md §8.1 documents its default and the reason for it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 cargo nextest run (--features unicorn) green, minus fuzz_robustness
- [ ] #2 cargo clippy --all-targets --all-features -- -D warnings clean
- [ ] #3 cargo fmt --check clean (nix-pinned rustfmt)
<!-- DOD:END -->
