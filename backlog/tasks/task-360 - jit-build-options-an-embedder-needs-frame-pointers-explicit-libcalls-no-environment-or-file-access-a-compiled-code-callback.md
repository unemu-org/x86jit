---
id: TASK-360
title: >-
  jit: build options an embedder needs: frame pointers, explicit libcalls, no
  environment or file access, a compiled-code callback
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - embedding
  - tooling
dependencies: []
priority: high
ordinal: 396000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
An embedder that runs a full native userspace inside its own process, with real POSIX signals and several guest threads, needs four things from the JIT backend that it cannot get today. All four are build or API options. None changes what guest code computes.

1. FRAME POINTERS. `build_jit` (x86jit-cranelift/src/lib.rs:2223-2271) sets `use_colocated_libcalls`, `is_pic`, `opt_level` and `enable_verifier`, and nothing else. Cranelift's `preserve_frame_pointers` keeps its default (off), and nothing in the workspace sets it. Without that flag Cranelift builds a frame record only in a function that needs a frame. A compiled block that calls no helper and spills nothing has no frame record, so a frame-pointer walk that starts at a sample inside it loses a frame. The walkers affected are a sampling profiler, a crash reporter and the embedder's own backtrace at signal time. DWARF unwinding does not cover the gap either: perfmap.rs:22-25 records that unwind information does not cross JIT frames. On an AArch64 host, frame records are the walkable chain defined by the procedure-call standard (AAPCS64, "The Frame Pointer"), and Apple's arm64 ABI asks that x29 always address a valid frame record. Make frame pointers an option and measure what they cost. The cost is a prologue and an epilogue per compiled function, so per block entry, not per instruction.

2. LIBCALLS. The `JITBuilder` is built with `cranelift_module::default_libcall_names()` (lib.rs:2271). When a lowering emits a Cranelift `LibCall`, the call is resolved by name at finalize. A name not registered through `builder.symbol` goes to cranelift-jit's default lookup in the host's dynamic symbol table. A statically linked or fixed-base host binary need not export `floorf`, `ceilf` or `memcpy`. In such a host, the first block that needs one fails at finalize, and only on the host ISA that happens to need it. These lowerings can do it at HEAD:
   - The SSE4.1 `round*` lowering, `emit_round`, uses Cranelift `floor`/`ceil`/`trunc`/`nearest` (codegen/mod.rs:3362, 3374-3377).
   - The non-truncating float-to-integer converts apply `nearest` before converting: cvtps2dq at codegen/vector.rs:3761, and the other sites at vector.rs:4308, 4377, 4407 and 4476. The libcall table must cover them too.
   - The byte shuffle uses `swizzle` (codegen/mod.rs:3017).
   - Cranelift's x86-64 backend lowers these to libcalls when the host lacks SSE4.1 or SSSE3.
   aot-plan.md:98-104 raises the same problem for the owned arena (TASK-383). The fix:
   - Register a host implementation for every `LibCall` the lowerings can produce.
   - Install a symbol lookup that resolves only registered names.
   - Fail loudly, with the name, on anything else.
   Also: the 62 `builder.symbol("x86jit_*", …)` registrations (lib.rs:2272-2347) resolve nothing. Compiled code reaches every helper by `call_indirect` through a baked address (codegen/mod.rs:42-48), and the only `declare_function` is the `Export` at lib.rs:2689. They are dead code, and they would hide an unknown-symbol check behind names that look registered.

3. ENVIRONMENT AND FILES. The library reads its process environment and opens files on its own. Three paths do it:
   - perfmap.rs:55-69: `X86JIT_PERF_MAP=1` creates `/tmp/perf-<pid>.map` and keeps the descriptor open for the life of the process.
   - lockstep.rs:335-366: `X86JIT_LOCKSTEP=<path>` creates a file at that path, and `X86JIT_LOCKSTEP_LO`/`_HI`/`_MAX` are parsed too. The check runs from `interpret_block` on every interpreted block (interp/mod.rs:144).
   - lift/mod.rs:859-865: `X86JIT_NO_FLAG_ELISION` changes what every lift emits.
   The two `std::env` reads in codegen/mod.rs (around line 4709) are inside `#[cfg(test)]` diagnostics, which is fine.
   Three kinds of embedder are affected:
   - One that passes guest descriptor numbers through to host descriptors 1:1 cannot let the engine open descriptors of its own.
   - One that runs without a writable `/tmp` gets a silent no-op.
   - One that implements a guest `exec` by re-executing itself with the guest's environment gives the guest control of these variables. That includes a path the engine will create with the host's privileges, and a switch that changes codegen.
   Add a cargo feature, on by default so nothing changes for existing users, whose absence compiles all three paths out. Give each an API equivalent:
   - flag elision as a lift or `Vm` option;
   - the perf map through the observer in item 4;
   - the lockstep sink as a writer the embedder supplies, or TASK-390's API mode.

4. A COMPILED-CODE OBSERVER. Add a callback on `JitBackend` that receives the host start, host length, guest entry RIP and unit kind (block or region, later AOT unit) of every compiled unit. That is exactly what `perfmap::record` receives at lib.rs:2718. With it the embedder can feed its own profiler, symboliser or crash reporter, and the perf map becomes one implementation of the callback. Document two constraints:
   - The callback runs on whichever thread compiled the unit (a vcpu or the tier-up worker), and the module lock taken at lib.rs:2490 is still held, so it must not re-enter the backend.
   - Once translations start being released (task-352, TASK-383), it needs a matching "released" event.

Existing work: task-140 (done) added the env-gated perf map. task-354 (done) reads the elision variable once per process instead of on every lift. Neither compiles these paths out or provides the options above.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `JitBackend` has a frame-pointer option that sets Cranelift's `preserve_frame_pointers`. A test walks the frame-pointer chain from a helper called by compiled code (and from a leaf block, via a signal) back to the dispatcher frame, on x86-64 and on an aarch64 host
- [ ] #2 The frame-pointer cost is measured against bench/baseline.json on both host architectures and recorded on this task, and the default is chosen from that number
- [ ] #3 Every `LibCall` the lowerings can emit resolves to a registered host function with no dynamic symbol lookup. An unregistered name fails with a message that names it. A test compiles the round and shuffle forms for an x86-64 target with SSE4.1/SSSE3 disabled and asserts they finalize through the registered table
- [ ] #4 The dead `x86jit_*` `builder.symbol` registrations are removed, or shown to be used
- [ ] #5 With the new feature off, x86jit-core and x86jit-cranelift contain no `std::env` read and no file creation or opening outside `#[cfg(test)]`, and a CI step proves it. With default features, behaviour is unchanged
- [ ] #6 Flag elision and the lockstep sink can be controlled through the API with the feature off, and a test covers each
- [ ] #7 The compiled-code observer receives every compiled block and region with its host range, guest RIP and kind, and a test asserts it sees both kinds. The perf map is rebuilt on top of it, and the threading and locking contract is documented on the API
- [ ] #8 `codegen_description` reports the frame-pointer setting, since it changes the emitted code
- [ ] #9 spec.md §17 lists the observer and the feature as embedder seams, and README documents the feature
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
