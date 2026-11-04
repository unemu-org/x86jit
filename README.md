# x86jit

[![CI](https://github.com/unemu-org/x86jit/actions/workflows/ci.yml/badge.svg)](https://github.com/unemu-org/x86jit/actions/workflows/ci.yml)

An x86-64 → host recompiler (JIT), delivered as a pure-Rust library.

> ⚠️ **Early-stage — not production quality.** Started July 2026, under active development.
> It almost certainly has bugs and missing instructions. Be clear-eyed about what the
> testing buys you: a differential oracle validates the instructions that **are** lifted
> (interpreter vs JIT vs a real CPU), but it can't tell you what's *missing* — gaps surface
> when real code hits an unimplemented instruction and traps. See [Status](#status).
>
> **Need a production-grade x86 emulator today? Use [QEMU](https://www.qemu.org/) or
> [Unicorn](https://www.unicorn-engine.org/).** x86jit is for people who want an
> embeddable, hackable CPU core in pure Rust and can live with gaps.

`x86jit` executes x86-64 guest code on any host (x86-64 or ARM64) via JIT recompilation. The core is **guest-agnostic** — it knows nothing about PS4, ELF, the syscalls of any concrete OS, or GPUs. It's a "CPU engine": you give it memory plus an entry point, it runs instructions and yields control every time it hits something it doesn't handle itself.

- **In scope:** x86-64 decoding (via `iced-x86`), lift to a custom IR, two backends (interpreter + Cranelift JIT), translation cache, dispatcher loop, guest memory + CPU state, return-based `Exit` API.
- **Out of scope (the embedder's job):** file-format parsing (ELF/SELF/PE), OS syscall emulation (HLE), MMIO/devices/GPU, loaders, high-level thread scheduling.

The full design lives in [`spec.md`](backlog/docs/design/spec.md).

## Workspace

The **core** is guest-agnostic; everything else is an embedder or tooling crate.

```
x86jit-core/        # Vm, Vcpu, IR, lift, cache, dispatcher, interpreter, x87/f80 — the engine
x86jit-cranelift/   # Cranelift JIT backend (the second `Backend`)
x86jit-elf/         # ELF loader helpers (static / static-PIE / dynamic + stack setup)
x86jit-tests/       # differential testing (vs Unicorn + native), instruction corpus, fuzzing, harness
x86jit-bench/       # workload timings (interp vs JIT vs native), recorded per commit
```

This repository is the **recompiler only** — it emulates no operating system. A
Linux syscall shim, a process model, an OCI/ELF runner and the whole-program test
ladder (busybox, sqlite, CPython, Go servers) live in
[`unemulinux`](https://github.com/unemu-org/unemulinux), which embeds this library.
That split is not aspirational: `x86jit-core`'s dependency set is exactly
`{iced-x86}`, and a test (`x86jit-tests/tests/boundary.rs`) fails the build if that
ever changes.

## Status

Actively developed, with a strong oracle for the instructions it *does* implement. A
hand-written instruction corpus and a fuzzer cross-check the **lifted** instructions three
ways — interpreter vs JIT, and both against a real CPU (Unicorn + native execution) — on
both an **x86-64 and an AArch64** CI runner, so the ARM host path is validated, not
assumed.

One limit on that, stated because it is easy to read past. The corpus validates *what's
lifted*; it does **not** tell you what's missing — that surfaces only when real code hits
an unimplemented instruction.

The native comparison covers general registers, flags, the whole vector file
(XMM/YMM/ZMM**0–31**), the opmasks, the x87 register stack and control word, and MXCSR.
Two things are captured but deliberately **not compared**, because the engine does not
model them: the x87 status-word condition codes and tag word (`TASK-324`), and MXCSR's
six sticky exception flags (`deferred.md`). Both show up in full in a divergence report.

**Unmodified real programs run on this engine** — busybox applets (`sha256sum`, `wc`,
`sort`, `awk`, gzip), sqlite3, lua, libjpeg-turbo `djpeg`, **CPython 3.13**, Go servers,
static/static-PIE/dynamic executables against both musl and glibc, and multi-process shell
pipelines out of a Docker/OCI image. Interpreter and JIT produce the same output as running
them natively.

Those tests **live in [`unemulinux`](https://github.com/unemu-org/unemulinux)**, not here,
because running a real program needs an operating system and this repository deliberately
has none. They still run — a lifter change is validated against that ladder before it is
considered good; the check simply spans two repositories now, so it is a CI-plumbing
problem rather than a coverage one. What runs *in this repository's own* CI is the
ISA-level validation below.

**Instruction coverage:** the full scalar integer set plus SSE/SSE2 up through the
common AVX/AVX2 vector set — SSE3/SSSE3/SSE4.1/SSE4.2, AVX, AVX2, BMI1/BMI2,
`tzcnt`/`lzcnt`/`movbe`, and **true 80-bit x87** computed in software (so x87 results
are bit-identical on x86-64 and ARM64). AVX-512/EVEX is partial and growing. The
guest CPU feature set is selectable per run (`baseline` / `v2` / `v3` / `v4`, the way
`qemu -cpu` works) rather than hardcoded. The exact per-generation breakdown of which
encodings lift is a generated, CI-checked artifact — see the
[**instruction-coverage map**](backlog/docs/compat/isa-coverage.md). **Read it as an
upper bound:** an instruction with both a register and a memory form is probed as the
register form, so a missing memory-operand form can still be reported as lifted
(`TASK-312`). That is how `vextract*`'s memory destination stayed invisible until a real
binary trapped on it.

**CPU modes:** `CpuMode::Long64` (64-bit long mode), `CpuMode::Compat32`
(32-bit protected/compatibility mode with flat segments), and `CpuMode::Real16`
(16-bit real mode). **Real16 is interpreter-only**; an embedder uses it. Its interpreter is validated against the hardware-captured
[SingleStepTests 80286 corpus](x86jit-tests/tests/ss286.rs), comparing final CPU and
memory state and in-guest exception delivery. The corpus reports remaining gaps;
it is not a claim of full 286 conformance. Targeted
[Unicorn comparisons](x86jit-tests/tests/cf16.rs) also cover real-mode addressing
and control flow.

**Engine:** two interchangeable backends — a portable interpreter and a Cranelift
JIT — over a single IR, with a translation cache, hotness-gated tier-up, superblock
regions, and block chaining + indirect-branch caching for fast dispatch. Self-modifying
code stays coherent, multiple guest threads share one VM, and x86-TSO memory-ordering
is preserved on weak (ARM) hosts — all exercised on the AArch64 runner.

**Performance.** Not yet optimized — expect roughly an **order of magnitude slower than
native** for hot code (a tight scalar loop is ~20× native on the JIT; the interpreter is
~40–250×), and worse for startup-heavy or run-once code, where the JIT pays to compile
everything up front. Throughput work is ongoing. The `x86jit-bench` crate records
interp/JIT/native timings per commit if you want real numbers.

**Known gaps** (deliberately absent or partial today):

- AVX-512 / EVEX is partial and growing; MMX is minimal (guests generally use SSE instead).
- **EVEX prefix bits can be silently dropped**, because the lifter dispatches on the
  mnemonic and a pre-AVX-512 handler for that mnemonic may never look at them. Measured by
  encoding each form, re-decoding to confirm the bit survived, and diffing the lifted IR:
  the **write mask** is ignored by 76 mnemonics (`{k1}` and `{k1}{z}` alike, `TASK-333`),
  and **embedded broadcast `{1toN}`** by 139 — no handler in the engine reads it at all
  (`TASK-334`). Broadcast is both a wrong result *and* a spurious fault, since the engine
  reads the full operand width where hardware reads one element. Embedded rounding `{er}`
  is likewise unread (`deferred.md` covers MXCSR.RC, which is a different thing). None of
  this is visible to the compat map or the differential corpus: both probe the plain form.
- Real16 has no JIT tier yet, and 286 fidelity gaps remain — see
  [deferred work](backlog/docs/deferred.md#real-mode-jit-tier-286-fidelity-and-protected-mode-segmentation).
- Protected-mode segmentation has no full segment-descriptor model; `Compat32`
  uses flat segments with `FS`/`GS` bases. Real16 has segmented addressing.
- Signals and fork/exec *after* a process spawns threads are not fully modeled (single-threaded fork/exec works; the threaded case returns a defined error rather than guessing).
- OS emulation (syscalls, devices, loaders) is the embedder's job, not the core's — see [`unemulinux`](https://github.com/unemu-org/unemulinux) for a Linux userland built on this library.
- **`Prot` is advisory — nothing enforces it.** A guest store into a region mapped `R` or `RX` succeeds and changes the bytes, on both backends; the engine models no permission fault (`TASK-330`). Map read-only expecting a trap and you get silent corruption instead.
- **MXCSR governs nothing** on the SSE side — `stmxcsr` returns the reset value, `ldmxcsr` is a no-op, and no SSE floating-point exception is raised or reported (`deferred.md`). The x87 half is modelled: all six status-word flags, the stack-fault flag with C1, the condition codes, and `#MF` delivery, each validated against a real CPU.
- **The x87 store and convert paths have known defects**, each measured against a real
  CPU: `fist`/`fistp` store 0 rather than the integer indefinite when the value does not
  fit, and raise no `IE` (`TASK-339`); `fst`/`fstp` to `m32`/`m64` ignore rounding
  control, never set `PE`, and double-round through `f64` (`TASK-340`); `fnstenv` leaves a
  masked-off exception armed, so an `#MF` handler that opens with it can livelock
  (`TASK-341`); `fprem` never writes `C2`, so the architectural reduction loop exits early
  or spins (`TASK-342`).
- **Nothing reclaims a dropped translation.** A guest that repeatedly patches code — a
  guest-level JIT rewriting an inline cache — was measured growing host RSS by about
  8 KiB and 1.3 executable mappings per patch before the indexed CodeMap. The executable
  arena and CodeMap metadata still grow with translations. The immutable CodeMap
  index retains O(n log n) nodes across n registrations: measured index cost is
  approximately 64 B × (log2 n + 1) per registration, about 1–1.4 KiB per compile
  at the measured sizes, in addition to the per-patch growth above. Its 4,194,304-entry
  capacity permits several GiB of retained index storage; this is a structural ceiling,
  not a practical memory budget. Use the compile limit below to stop earlier.
  IBTC descriptors continue growing per invalidation epoch per live indirect-branch
  site, even after compilation stops. Reclaim is tracked separately in `xj-aau.4`.
  Returned code-arena allocation errors, executable
  memory finalization/protection resource failures (ENOMEM/EAGAIN, including Linux's
  mapping-count limit), CodeMap capacity exhaustion, and failure to start the background
  compiler now permanently stop new compilation. Each vcpu
  returns `Exit::CodeSpaceExhausted { addr, .. }` once at a resumable block boundary;
  call `run()` again to continue with interpreter fallback for uncompiled blocks.
  Existing translations remain usable. `JitBackend::set_compile_limit` can cap
  compilation attempts before host resources run out. On a weak host, interpreter
  fallback provides only the interpreter's ordering for ordinary loads/stores and
  does not honor `MemConsistency::AcqRel` or `FullTso`. An embedder relying on either
  tier for a multithreaded guest should treat `CodeSpaceExhausted` as fatal.
  Other finalization/protection errors retain their diagnostic panic. Global allocator
  failure retains Rust's process-abort behavior.
- **Self-modifying code is observed one block late.** A block that writes into the page it is itself executing runs to the end of that block on the old bytes; the re-lift takes effect on the next dispatch. This matches QEMU and is the deviation `spec.md` §10 records. Everything else — another block's page, `rep stos`, an x87 store — invalidates on both backends.

**API stability.** Pre-1.0 (`0.x`). The embedding API (`Vm`, `Vcpu`, `Exit`, …) is not
frozen and will have breaking changes between releases.

## Getting started

With Nix (recommended — pins the whole toolchain):

```sh
nix develop            # or: direnv allow, then it auto-loads
cargo build
cargo nextest run
```

Neither the build nor the tests need the network. The `oracles/` submodule holds the
pinned manuals the code cites (see [`PROVENANCE.md`](PROVENANCE.md)); it is only needed
if you want to *read* a source, so a clone without it builds and tests fine:

```sh
git submodule update --init      # optional: the cited sources, for humans
```

Without Nix:

```sh
rustup toolchain install stable   # MSRV 1.75
cargo build
cargo test
```

## Embedding

The core is a library. Give it a memory map and an entry point; it hands control
back through `Exit` whenever it hits something you own (a syscall, an MMIO
access, an unsupported instruction):

```rust
use x86jit_core::{Exit, Prot, Reg, RegionKind, Vm, VmConfig};

let mut vm = Vm::new(VmConfig::flat(0x1_0000));   // flat guest space, interpreter backend
vm.map(0, 0x1_0000, Prot::RWX, RegionKind::Ram).unwrap();
vm.write_bytes(0x1000, &[0xB8, 0x05, 0x00, 0x00, 0x00, 0xF4]).unwrap(); // mov eax,5 ; hlt

let mut cpu = vm.new_vcpu();
cpu.set_reg(Reg::Rip, 0x1000);
assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
assert_eq!(cpu.reg(Reg::Rax) as u32, 5);
```

`cpu.exit_request()` returns a cloneable, `Send + Sync` handle. Another host thread
can call `handle.request_exit()` to make `run` return `Exit::Interrupted`, even
without a block budget. The vCPU yields at its next checked instruction boundary:
block/chain edges, region sub-block entries, or between REP string iterations.
Guest state is precise; call `run` again to resume. Requests made outside `run`
remain pending, and duplicate pending requests coalesce into one exit. Each handle
targets one vCPU; a handle outliving that vCPU has no effect. Everything a thread
wrote before `request_exit()` is visible to the caller once `run` returns the
`Exit::Interrupted` that consumed that request. Compiled blocks charge their whole
instruction count at entry, so an interrupted REP can over-charge that counter,
like a mid-block fault.

`Exit` and its struct-like variants are non-exhaustive: matches need a wildcard
arm, and struct-like patterns need `..` to allow future exit reasons and fields.
Replace struct literals with the stable constructors
`Exit::unmapped_memory(addr, access)`, `Exit::mmio_read(addr, size)`,
`Exit::mmio_write(addr, size, value)`, `Exit::unknown_instruction(addr, bytes, len)`,
`Exit::exception(addr, vector)` and `Exit::port_io(port, size, dir, value)`;
fields added later receive defaults through these constructors.

Trap data accesses return `Exit::MmioRead` or `Exit::MmioWrite` with `addr` and
`size` for the current transfer (at most 8 bytes), plus `total_size: u16` for the
full memory operand width and `part: u8` for its zero-based transfer slot.
For example, a 32-byte load returns four 8-byte reads at offsets 0, 8, 16 and 24,
each with `total_size == 32` and `part` 0 through 3. RIP stays on the instruction
until all transfers finish. Supply each read through `complete_mmio_read(value)`;
perform each write's side effect and call `complete_mmio_write()` before resuming.

`total_size` includes masked-off bytes, rather than counting active bytes, and
describes the memory operand (the source chunk for a broadcast, the destination
for a narrowing store). `part` is `addr.wrapping_sub(operand_base) / size`:
per-element masked memory transfers and narrowing stores use element-sized slots,
so inactive elements leave gaps in the indices. Whole-table reads, including
masked VPERMD and GFNI affine memory forms, use 8-byte chunks, so a 32-byte table
reports parts 0, 1, 2 and 3, regardless of the element width.
Slots served by RAM also keep their indices when only part of an operand traps.
Scalar loads/stores of 8 bytes or less and the existing MMIO constructors use
`total_size == size` and `part == 0`.
Both backends report the same metadata.

Instruction fetch from a `RegionKind::Trap` region returns `Exit::UnmappedMemory`
with `AccessKind::Execute`, even when the region is executable. Its `addr` and RIP
identify the attempted instruction's start (the decode IP in Real16). An instruction
straddling a region boundary faults at its start without reading bytes from the next
region; earlier instructions complete first. Trap data reads and writes still use
the MMIO exits and completion methods.

Swap in the JIT with `Vm::with_backend(cfg, Box::new(JitBackend::new()))` — same
API, identical guest state. Runnable examples:

```sh
cargo run -p x86jit-core      --example raw_bytes      # smallest embedding
cargo run -p x86jit-core      --example mmio_device    # a trapped MMIO device
cargo run -p x86jit-cranelift --example jit_vs_interp  # wiring in the JIT
cargo run -p x86jit-elf       --example run_elf -- ELF # load + run a static ELF
```

## Documentation

- [`spec.md`](backlog/docs/design/spec.md) — authoritative design spec (contract, IR, backends, semantics traps).
- [`PROVENANCE.md`](PROVENANCE.md) — where the engine's behaviour comes from: what is authoritative (the Intel SDM, the AMD APM, the psABI), what is merely an oracle, what we deliberately do not model, and the licence surface.
- [`backlog/`](backlog/) — load-on-demand knowledge tree (agent + user docs, ADRs, decision log).
- [`AGENTS.md`](AGENTS.md) / [`CLAUDE.md`](CLAUDE.md) — pointer table for coding agents.

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE))
- MIT license ([`LICENSE-MIT`](LICENSE-MIT))

at your option. All core dependencies are permissive (MIT/Apache), so there are no copyleft constraints (`spec.md` §15).
