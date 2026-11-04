---
id: TASK-383
title: 'jit: retire JITModule for an owned code arena (doc-17 B0.2)'
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - aot
dependencies:
  - TASK-360
  - TASK-376
priority: medium
ordinal: 419000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
doc-17 B0.1 is done. Helpers are reached by `call_indirect` through a baked address (x86jit-cranelift/src/codegen/mod.rs:42-48; lib.rs:2500-2505), so emitted code carries no helper relocation. Everything else still goes through cranelift-jit's `JITModule`:
- the `Jit` struct owns one (lib.rs:2075-2076);
- `build_jit` creates it with the default libcall names (lib.rs:2271, 2348-2356);
- `compile_with` gets its context from it and declares, defines and finalizes through it (lib.rs:2489-2510, 2687-2711).

What that costs:
- cranelift-jit owns and lays out the code memory, and never frees it. codemap.rs:8-12 and perfmap.rs:18-21 both depend on that. task-352 needs code it can release, and nothing here can release any.
- `JITModule` is `!Sync`, and finalize takes `&mut`. So one mutex is shared between the foreground compile and the background worker, which is why deferred.md:112-115 keeps a single compile worker "until FD-AOT B0.2 removes the shared module".
- Persisted code cannot be loaded into a `JITModule`.
- Any leftover libcall is resolved through `JITModule`'s name lookup (TASK-360).
- spec.md §9 (line 893) already describes the intended model: "an arena of executable memory (`memmap2`, W^X; on macOS `pthread_jit_write_protect`) … belongs to the `Vm`". The implementation does not match it.

Do B0.2 (aot-plan.md:48-59):
1. Build `ctx.func` directly and compile it with `ctx.compile` against the ISA.
2. Assert that `compiled_code().buffer.relocs()` is empty. That proves B0.1, and it settles the libcall question of aot-plan.md:98-104 with a measurement on both host architectures instead of an assumption.
3. Copy the bytes into an arena the backend owns.
4. Do the instruction-cache maintenance before the entry is published.

ONE CORRECTION TO THE PLAN. B0.2's arena flips a chunk from RW to RX around each append: mprotect RW, memcpy, mprotect RX (aot-plan.md:54-56). With background tier-up, other vcpus can be executing code in that same chunk while the worker appends to it. Flipping the chunk to RW makes them fault. Three ways out:
- Never append into a page that already holds published code. This is page-granular and wastes up to a page per unit unless units are batched.
- Map the memory twice, one RX view and one RW view. That needs a shared-memory object (on Linux, a descriptor, which conflicts with TASK-360's no-descriptor build unless a short-lived memfd is acceptable) or `mach_vm_remap` on macOS.
- On macOS arm64, use `MAP_JIT` with `pthread_jit_write_protect_np`, which switches writability per thread rather than per page.
Pick one per host and document it.

INSTRUCTION-CACHE MAINTENANCE AND PUBLICATION. On AArch64 the writer cleans the data cache to the point of unification and invalidates the instruction cache for the range (DC CVAU, DSB ISH, IC IVAU, DSB ISH). Every other core that will run the code needs a context-synchronization event (ISB) after it observes the publication (Arm Architecture Reference Manual, "Concurrent modification and execution of instructions" and the instruction-cache maintenance rules). The cross-thread half is TASK-376, and this task consumes its result. On x86-64 a store to code followed by a jump to it is coherent on the same core (SDM Vol. 3A, "Self-Modifying Code"). Code published to another thread falls under the rules for cross-modifying code.

Trade-off: owning the arena means owning W^X, instruction-cache maintenance and the macOS JIT-memory story that cranelift-jit hid. It buys four things:
- code that can be released (task-352);
- parallel compile workers (deferred.md:112-115);
- loadable persisted units (TASK-384, TASK-385);
- no dynamic symbol lookup at all.

Dependencies: TASK-360 registers the libcalls explicitly, which the no-relocation assert needs. TASK-376 proves cross-thread code publication on aarch64 outside Linux.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 cranelift-jit and cranelift-module are no longer dependencies of x86jit-cranelift. Compiled code lives in an arena that the backend owns
- [ ] #2 Every compile asserts that the emitted buffer has no relocations. Run across the full differential corpus on x86-64 and on an aarch64 host, it never fires, and any remaining libcall is resolved explicitly
- [ ] #3 No mapping is ever writable and executable at the same time, and no page that holds published code loses execute permission while another thread may run it. A test keeps one vcpu in a chained loop on a page while the worker appends units to the arena, on both host architectures
- [ ] #4 AArch64 instruction-cache maintenance is explicit and follows TASK-376's cross-thread rule. A test runs freshly published code from a thread other than the compiler's, on an aarch64 host
- [ ] #5 The full differential corpus and the jit == interp suites are green on both host architectures (the doc-17 gate)
- [ ] #6 Compile time and run time are measured against bench/baseline.json before and after, and recorded on this task
- [ ] #7 spec.md §9 "Ownership of JIT code memory" and aot-plan.md B0.2 describe the arena that was built, including the W^X scheme chosen for each host. The deferred.md entry on the compiler-thread pool is updated
- [ ] #8 codemap.rs and perfmap.rs say what lifetime the arena guarantees: append-only until task-352 lands
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
