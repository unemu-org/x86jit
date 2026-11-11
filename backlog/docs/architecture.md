---
id: doc-1
title: 'Architecture'
type: guide
created_date: '2026-07-06 11:25'
---

# Architecture

> **The Linux userland moved out (2026-08-04).** The syscall shim, process model,
> OCI/ELF runner and the whole-program test ladder now live in a separate project,
> [`unemulinux`](https://github.com/unemu-org/unemulinux), which embeds this library.
> Passages below describing `x86jit-linux`, `x86jit-cli`, or running busybox/sqlite/
> CPython/Go describe work that happened in this repository's history and now lives
> there. What stays here is the recompiler: core, the Cranelift backend, the ELF
> loader helpers, and the ISA-level test and bench harnesses.

Workspace shape, data flow, key modules. **Descriptive of the current state**, not aspirational. For *why* the architecture is what it is, see [`../decisions/`](../decisions/) and [`../design/spec.md`](design/spec.md).

## Layout

Cargo workspace, five crates (spec.md §2):

```
x86jit/
├── x86jit-core/        # Vm, Vcpu, IR, lift, cache, dispatcher, interpreter — the engine
├── x86jit-cranelift/   # Cranelift JIT backend: JitBackend + codegen (feature `jit`, default-on)
├── x86jit-elf/         # ELF64 loader (goblin): load_static_elf + setup_stack (SysV argv/auxv); convenience, NOT core
├── x86jit-tests/ # harness: RON vectors, compare, Unicorn + native oracles, 80286 corpus, fuzzers
├── x86jit-bench/ # synthetic workload timings (interp vs JIT), recorded per commit
├── oracles/ # submodule: the pinned manuals and tables the code cites (PROVENANCE.md)
├── flake.nix           # Nix devShell + package (rust-overlay toolchain)
└── spec.md          # authoritative design spec
```

The syscall shim, the process model and the OCI/ELF runner are **not** here — they live in
[`unemulinux`](https://github.com/unemu-org/unemulinux), which embeds this library. `x86jit-core`'s
dependency set is exactly `{iced-x86}`, and `x86jit-tests/tests/boundary.rs` fails the build if
that changes.

`x86jit-core` module map (`x86jit-core/src/`):

| Module | Purpose |
|--------|---------|
| `legacy` | Application/system execution profiles, shared legacy model boundary, pure context derivation, appended system/descriptor storage, checked real-address accesses and model-aware flag images (decision-18 §§1–2) |
| `state` | `Reg`, `CpuState` (`#[repr(C)]`, flat GPR file), `Flags` (§3) |
| `memory_stats` | Public non-exhaustive allocation snapshots: VM/cache/backend categories and live vCPU gauges; page-rounding model for the pinned Cranelift allocator, process-wide retained fault metadata (§4.1, §9.1) |
| `memory` | `Memory`, `MemoryModel` (Flat/SoftMmu), `Prot`, `RegionKind`, `MemTrap`; opt-in host-fault recovery, owned fetch snapshots (§4, §8.1) |
| `process_barrier` | Fixed per-VM source ordering: Linux/FreeBSD private expedited membarrier; probe failure (including macOS) selects fenced stores and an always-open compiled-store notification gate. Later process-barrier failure runs one lifted block uncached after bounded retries (§10) |
| `child` | Owned VM snapshot with a lifetime marker, preallocated per-child arena, const-specialized shared lift/interpreter, transactional refusals, written-page reconciliation and static host-fault fixups (decision-15) |
| `usercopy` | x86-64/AArch64 naked transfer routines, bounded immutable relative fixup table, scalar load/store and CAS, copy and probe operations (§8.1, decision-13) |
| `ir` | `IrOp`, `Val`, `Temp`, `Cond`, `MemOrder`, `IrBlock`, `TempGen`; `MemoryAccess` occupies access-op padding and selects cold #SS/#GP exits on a miss; `CheckAlign` raises #GP for required alignment (§6) |
| `lift` | x86 → IR: `lift_block`, operand lowering, `CpuMode` seam, `LiftError` (§7) |
| `host_write` | Cold host VM write-permission queries after a crossing-store read probe faults (§8.1, decision-13) |
| `lift::legacy`, `interp::legacy`, `vm::legacy` | Explicit-context I386 real lowering and interpreter, centralized stack/segmented accesses, private per-vCPU cache/epoch; protected execution and event delivery remain refused (decision-18 §§1, 3–4) |
| `interp` | IR interpreter: `interpret_block` over a Vcpu-reused temps buffer, Variant-A flags, RIP-on-trap; also the shared `string_run`/`divide`/`crc32c` helpers the JIT calls; `interp::softfloat` implements non-default SIMD numerical modes with integer arithmetic (§8.1) |
| `x87` | x87 FPU exec (`exec_x87`/`exec_fxstate`), shared by both backends; register file is `[F80; 8]` (§14) |
| `f80` | software 80-bit extended float (`F80`) — true x87 precision, rounds each op to a 64-bit significand; pure Rust (portable across x86/ARM) (§14) |
| `jit_abi` | compiled-block ABI (shared with the JIT crate): `MemCtx`, `CpuOffsets`, result codes, `run_compiled`; `jit_abi::fp` owns host FP-control transitions around compiled code and helpers; `jit_abi::fault` owns per-run records, one-frame unwind recipes, mangled naked landing stubs and signal-context classification (§8.2.1-2) |
| `codemap` | immutable host ranges, source locations, exact transfer sites and compiler-derived unwind recipes, published together for signal-safe lookup; PID-scoped shared-child JIT admission marker (§8.2) |
| `disasm` | decode-and-print helper: `disassemble`, `print_disassembly`, `DecodedInsn` (inspection only, §12 M0) |
| `exit` | `Exit`, `AccessKind`, `StepResult` (§5, §8) |
| `interruption` | Per-vCPU engine-section markers and host-request deferral evidence (§9.2) |
| `cache` | `TranslationCache`, `CachedBlock`, `CompiledPtr`; `BlockKey` includes decode mode and MXCSR.RC/FTZ/DAZ; SMC page-tag mark/clear run through `insert`/`invalidate_overlapping` callbacks under the spans lock (§9) |
| `code_lifetime` | Published-unit generations, cache publication ownership, retirement tickets and dispatcher execution participation; shared slots clear before epoch advance. Execution grace is separate from signal-reader grace; storage remains retained (decision-17 phase 3) |
| `cache::cache_hash` | Local integer hasher for shared translation keys; guest address keys use no HashDoS protection (§9.1) |
| `cache::cache_shards` | Sixteen independent reader maps; map mutations take spans → publication gate → all shards → lifetime registry; slot fills omit shards (§9.1, §10) |
| `time_stats` | `TimeStats`, transition timers and sampled helper timing; per-vCPU records summed without compiler locks (§2, §9.2) |
| `vm` | `Vm` (shared), `Vcpu` (per-thread), `run()` dispatcher loop; 48 KiB two-way private compiled cache with epoch/MXCSR flushes (§2, §9.2) |

## Data flow

```
guest bytes → iced-x86 decode → lift → IR (IrBlock) → backend.materialize → CachedBlock
                                                                  │
                       translation cache (guest addr → CachedBlock)
                                                                  │
   dispatcher loop: cache_get(pc) → execute(block) → StepResult ──┴→ Continue (loop) | Exit (return to user)
```

Hot path (RAM access) is **inlined** into generated code — never a callback. Rare events (syscall, MMIO, unknown instruction) **trap out** through `Exit` (§1 boundary rule).

Compiled fault recovery keeps each dispatcher call direct. The Cranelift backend's
`unwind` module extracts one-frame save-slot recipes at compilation and publishes
them with exact trap sites. `Vcpu::run` publishes its record once; an embedder's
signal adapter derives Access/Alignment from the OS reason (saved ESR on macOS
AArch64), restores the outermost compiled frame and returns through a mangled
naked landing stub. Alignment faults replay in the interpreter; run-record cleanup
also runs on panic. AArch64 adapters copy LR, scratch and d8–d15 as well as
preserved GPRs. Nested compiled frames are rejected. An immutable fixup-table
lookup precedes run-record validation: Access faults in opted-in Rust guest
transfers resume at the routine's error return with the address in scratch,
without unwinding a Rust frame or needing an active vCPU. The lifter copies its
fetch window starting with at most 128 page/region-bounded bytes, extending it
when decoding needs more bytes, outside cache locks; the decoder sees only
local bytes.
A const-generic access view selects direct or routed interpreter transfers once
per run. Protected mapped-RAM compiled faults return directly in either mode;
compiled split stores can retain completed host transfers (xj-obx.8). Routed Rust
masked/narrow stores preflight selected host pages. Compiled masked/narrow,
FX/x87 and REP helpers notify each committed element or image through the fenced
writer hook. Child-view stores still use a frozen tag snapshot; live-tag source
ordering for children is tracked separately (xj-aau.5). See
[decision-13](../decisions/decision-13%20-%20Signal-context-landing-pads-for-compiled-guest-faults.md).

`Vm::time_stats()` sums cumulative per-vCPU diagnostics, retaining totals after a
vCPU is dropped. Counts are always collected; `Vm::set_time_accounting(true)`
enables detailed elapsed-time measurements. Interpreter/compiled segments use
transition timing, helper and fetch/translate calls use deterministic 1-in-64
sampling, and compilation uses call/batch boundaries. Outside intervals close at
the next accepted run entry. AArch64 uses CNTVCT_EL0/CNTFRQ_EL0; other hosts use
`Instant`. These are elapsed durations including host descheduling. Running vCPUs
publish at run exit, so a snapshot may lag a whole run. Field docs specify which
categories overlap and which generated fast paths are excluded.

The KVM-style split: **`Vm`** owns shared state (memory + cache); **`Vcpu`** owns per-thread `CpuState` and its own `run()` loop. Many `Vcpu`s over one `Vm` is the path to multithreading (§2, §11).

## Key modules / contracts

- **Backend is not one `execute`.** `materialize(&IrBlock) -> CachedBlock` is backend-dependent; `execute(&CachedBlock)` is uniform and matches on the variant (§8). The interpreter wraps `Arc<IrBlock>`; the JIT compiles to host code.
- **Tier-up is core-driven; background tier-up adds a backend worker (bg-tier, doc-22).** A block starts interpreted and is JIT-compiled once hot (`Vm::set_tier_up_after`). With `set_tier_up_background`, the dispatcher submits the hot block via `Backend::tier_up_async` to a compiler thread the `JitBackend` owns (all threading stays in `x86jit-cranelift`; core stays `{iced-x86}`); the vcpu keeps interpreting and the dispatcher drains completions and publishes each via `cache.upgrade`, guarded by the cache's publication generation (advanced by every invalidation) — the backend never touches the cache (decision-4).
- **`StepResult`, not `Exit`, from the execution layer** — distinguishes "continue" from "trap out" (§8).
- **Operand lowering (§7.1) sits *beneath* per-mnemonic lift.** Every operand reduces to a `Val`; memory operands expand to effective-address arithmetic + `Load`/`Store`. This is the load-bearing layer — nothing lifts without it.

`x86jit-core::code_lifetime` owns generation identities, execution participants,
shared prediction slots and immutable IBTC descriptors. The Cranelift backend's
`reclamation` module polls retirement on a separate maintenance thread, removes
code-map ranges after execution grace, and releases compiler/provider storage
after signal-reader grace. Deferred teardown retains compiler and helper state
until both complete. See [the code-memory contract](design/code-memory-provider.md#final-phase-3-ownership-and-maintenance).

## Layering rules

- `x86jit-core` depends on `iced-x86` only. No cranelift, no memmap2 in core.
- `x86jit-cranelift` depends on `x86jit-core` + cranelift crates (feature-gated).
- `x86jit-elf` depends on `x86jit-core` only; it is a convenience, the core never parses formats.
- Nothing depends on `x86jit-tests`.
