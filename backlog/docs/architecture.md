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
| `state` | `Reg`, `CpuState` (`#[repr(C)]`, flat GPR file), `Flags` (§3) |
| `memory` | `Memory`, `MemoryModel` (Flat/SoftMmu), `Prot`, `RegionKind`, `MemTrap`; opt-in host-fault recovery, owned fetch snapshots (§4, §8.1) |
| `usercopy` | x86-64/AArch64 naked transfer routines, bounded immutable relative fixup table, scalar load/store and CAS, copy and probe operations (§8.1, decision-13) |
| `ir` | `IrOp`, `Val`, `Temp`, `Cond`, `MemOrder`, `IrBlock`, `TempGen`; `MemoryAccess` occupies access-op padding and selects cold #SS/#GP exits on a miss; `CheckAlign` raises #GP for required alignment (§6) |
| `lift` | x86 → IR: `lift_block`, operand lowering, `CpuMode` seam, `LiftError` (§7) |
| `interp` | IR interpreter: `interpret_block` over a Vcpu-reused temps buffer, Variant-A flags, RIP-on-trap; also the shared `string_run`/`divide`/`crc32c` helpers the JIT calls (§8.1) |
| `x87` | x87 FPU exec (`exec_x87`/`exec_fxstate`), shared by both backends; register file is `[F80; 8]` (§14) |
| `f80` | software 80-bit extended float (`F80`) — true x87 precision, rounds each op to a 64-bit significand; pure Rust (portable across x86/ARM) (§14) |
| `jit_abi` | compiled-block ABI (shared with the JIT crate): `MemCtx`, `CpuOffsets`, result codes, `run_compiled`; `jit_abi::fault` owns per-run records, one-frame unwind recipes, mangled naked landing stubs and signal-context classification (§8.2.1-2) |
| `codemap` | immutable host ranges, source locations, exact transfer sites and compiler-derived unwind recipes, published together for signal-safe lookup (§8.2) |
| `disasm` | decode-and-print helper: `disassemble`, `print_disassembly`, `DecodedInsn` (inspection only, §12 M0) |
| `exit` | `Exit`, `AccessKind`, `StepResult` (§5, §8) |
| `cache` | `TranslationCache`, `CachedBlock`, `CompiledPtr`; SMC page-tag mark/clear run through `insert`/`invalidate_overlapping` callbacks under the spans lock (§9) |
| `vm` | `Vm` (shared), `Vcpu` (per-thread), `run()` dispatcher loop (§2, §9.2) |

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
masked/narrow stores preflight selected host pages; routed FX/x87/REP helper
instances stay out of the default helper frames. See
[decision-13](../decisions/decision-13%20-%20Signal-context-landing-pads-for-compiled-guest-faults.md).

The KVM-style split: **`Vm`** owns shared state (memory + cache); **`Vcpu`** owns per-thread `CpuState` and its own `run()` loop. Many `Vcpu`s over one `Vm` is the path to multithreading (§2, §11).

## Key modules / contracts

- **Backend is not one `execute`.** `materialize(&IrBlock) -> CachedBlock` is backend-dependent; `execute(&CachedBlock)` is uniform and matches on the variant (§8). The interpreter wraps `Arc<IrBlock>`; the JIT compiles to host code.
- **Tier-up is core-driven; background tier-up adds a backend worker (bg-tier, doc-22).** A block starts interpreted and is JIT-compiled once hot (`Vm::set_tier_up_after`). With `set_tier_up_background`, the dispatcher submits the hot block via `Backend::tier_up_async` to a compiler thread the `JitBackend` owns (all threading stays in `x86jit-cranelift`; core stays `{iced-x86}`); the vcpu keeps interpreting and the dispatcher drains completions and publishes each via the epoch-guarded `cache.upgrade` — the backend never touches the cache (decision-4).
- **`StepResult`, not `Exit`, from the execution layer** — distinguishes "continue" from "trap out" (§8).
- **Operand lowering (§7.1) sits *beneath* per-mnemonic lift.** Every operand reduces to a `Val`; memory operands expand to effective-address arithmetic + `Load`/`Store`. This is the load-bearing layer — nothing lifts without it.

## Layering rules

- `x86jit-core` depends on `iced-x86` only. No cranelift, no memmap2 in core.
- `x86jit-cranelift` depends on `x86jit-core` + cranelift crates (feature-gated).
- `x86jit-elf` depends on `x86jit-core` only; it is a convenience, the core never parses formats.
- Nothing depends on `x86jit-tests`.
