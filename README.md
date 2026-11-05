# x86jit

A guest-agnostic x86-64 → host recompiler (JIT), delivered as a pure-Rust library.

You give it a memory map and an entry point. It runs the guest's x86-64 code on an AArch64
host (the primary target) or an x86-64 host, and hands control back through an `Exit` whenever
it meets something it does not handle itself: a syscall, an MMIO access, port I/O, a guest
exception, or an instruction it cannot lift yet. File formats, operating-system emulation and
devices belong to the embedder. A Linux userland built on this library is
[`unemulinux`](https://github.com/unemu-org/unemulinux).

> **Early-stage.** x86jit is under active development and has bugs and missing instructions.
> If you need a production-grade x86 emulator today, use [QEMU](https://www.qemu.org/) or
> [Unicorn](https://www.unicorn-engine.org/). x86jit is for people who want an embeddable,
> hackable CPU core in pure Rust and can live with gaps.

## What it does

- **One IR, two backends.** Guest code is decoded with `iced-x86` and lifted to a small IR. A
  portable interpreter and a Cranelift JIT both execute it and must agree bit for bit; the
  interpreter is the JIT's oracle.
- **Tiered execution.** Blocks start interpreted and compile once hot (`Vm::set_tier_up_after`),
  optionally on a background compiler thread (`Vm::set_tier_up_background`). Hot loops can tier
  up further into superblock regions (`JitBackend::with_superblocks`). Compiled blocks chain
  directly and indirect branches are cached.
- **x86-TSO memory-ordering tiers.** On weak hosts, `MemConsistency` (`Fast`, `AcqRel`,
  `FullTso`) selects the barriers around JIT inline loads and stores. `mfence` is a full barrier
  on both backends and locked operations stay atomic. Many vCPUs on host threads share one `Vm`.
- **Precise faults, resumable exits.** A fault leaves RIP on the faulting instruction with
  earlier instructions complete; handle the cause and call `run()` again. Host faults on guest
  memory in compiled code are recovered through a signal-context landing pad, with the signal
  handler owned by the embedder ([decision-13]). Another host thread can stop a running vCPU
  through `Vcpu::exit_request()`.
- **Self-modifying code.** Guest stores into translated code invalidate it on both backends. A
  block that rewrites itself sees the change from the next block ([spec §10][spec]).
- **Configurable guest CPU.** `Long64`, `Compat32` and `Real16` modes; feature presets
  `GuestCpuFeatures::{baseline, v2, v3, v4}` or a complete CPUID table ([decision-11]);
  embedder-supplied time and entropy sources ([decision-12]). x87 runs as true 80-bit arithmetic
  in software, so its results are identical on every host.
- **Guest-agnostic core.** `x86jit-core` depends on `iced-x86` and nothing else;
  [`boundary.rs`](x86jit-tests/tests/boundary.rs) fails the build if that changes.

## Workspace

| Crate | What it is |
| --- | --- |
| [`x86jit-core`](x86jit-core/) | The engine: `Vm`, `Vcpu`, lift, IR, interpreter, translation cache, dispatcher, x87. |
| [`x86jit-cranelift`](x86jit-cranelift/) | The Cranelift JIT backend, `JitBackend` (default `jit` feature). |
| [`x86jit-elf`](x86jit-elf/) | Optional ELF loader helpers: static, static-PIE and dynamic images, initial stack. |
| [`x86jit-tests`](x86jit-tests/) | Differential harness: vectors, Unicorn and native (x86-64 Linux) oracles, fuzzers, the 80286 corpus, the ISA coverage map. |
| [`x86jit-bench`](x86jit-bench/) | Interpreter and JIT workload timings, recorded per commit. |

## Embedding

```rust
use x86jit_core::{Exit, Prot, Reg, RegionKind, Vm, VmConfig};
use x86jit_cranelift::JitBackend;

// A 64 KiB flat guest on the Cranelift JIT; `Vm::new(config)` runs the interpreter.
let mut vm = Vm::with_backend(VmConfig::flat(0x1_0000), Box::new(JitBackend::new()));
vm.map(0, 0x1_0000, Prot::RWX, RegionKind::Ram).unwrap();
// mov eax, 1 ; syscall ; hlt
vm.write_bytes(0x1000, &[0xB8, 0x01, 0x00, 0x00, 0x00, 0x0F, 0x05, 0xF4]).unwrap();

let mut cpu = vm.new_vcpu();
cpu.set_reg(Reg::Rip, 0x1000);
loop {
    match cpu.run(&vm, None) {
        // RIP is already past `syscall`: service the call, then run again.
        Exit::Syscall => {
            let number = cpu.reg(Reg::Rax);
            cpu.set_reg(Reg::Rax, number + 41); // the call's return value
        }
        Exit::Hlt => break,
        // `Exit` is #[non_exhaustive]: keep a catch-all arm.
        other => panic!("unhandled exit: {other:?}"),
    }
}
assert_eq!(cpu.reg(Reg::Rax), 42);
```

Runnable examples:

```sh
cargo run -p x86jit-core      --example raw_bytes      # smallest embedding
cargo run -p x86jit-core      --example mmio_device    # a trapped MMIO device
cargo run -p x86jit-cranelift --example jit_vs_interp  # wiring in the JIT
cargo run -p x86jit-elf       --example run_elf -- ELF # load and run a static ELF
```

## Documentation

- [`spec.md`][spec]: the authoritative design (contract, IR, backends, semantics traps).
- [`architecture.md`](backlog/docs/architecture.md): workspace shape, data flow and module map.
- [`backlog/decisions/`](backlog/decisions/): design decisions, including the embedder contracts
  for host-fault recovery ([decision-13]), CPUID ([decision-11]) and time and entropy ([decision-12]).
- [`status.md`](backlog/docs/status.md) and [`deferred.md`](backlog/docs/deferred.md): what
  works, and what is deliberately left out.
- [`isa-coverage.md`][coverage]: the generated map of which encodings lift, per x86-64 level.
- [`commands.md`](backlog/docs/commands.md): build, test, fuzz, bench and lint commands.
- [`PROVENANCE.md`](PROVENANCE.md): where the engine's behaviour comes from, and the licence surface.
- [`AGENTS.md`](AGENTS.md): orientation for coding agents.

## Building and testing

```sh
nix develop                   # pinned toolchain; or `direnv allow`
git submodule update --init   # optional: the manuals the code cites (oracles/)
cargo build
cargo nextest run --features unicorn -E 'not binary(fuzz_robustness)'
cargo clippy --all-targets --all-features -- -D warnings
cargo deny --all-features check licenses bans sources
```

`fuzz_robustness` runs for more than seven minutes by design, hence the filter. The `unicorn`
feature (also enabled by `--all-features`) links the native Unicorn library from the dev shell;
without Nix, use stable Rust 1.88 or newer and leave it out. Tests need neither the network nor
the `oracles/` submodule: witness tests pin the cited values in code.

## Known gaps

- **Missing instructions.** Anything the lifter does not handle returns
  `Exit::UnknownInstruction`. AVX-512 is partial: some EVEX handlers ignore the write mask, and
  embedded broadcast is not yet read. MMX is minimal. The tests validate what is lifted, not
  what is missing; read the [coverage map][coverage] as an upper bound.
- **`Prot` is advisory.** A guest store into an `R` or `RX` region succeeds on both backends
  ([deferred.md](backlog/docs/deferred.md)).
- **FP control is partly modelled.** MXCSR's RC, FTZ and DAZ govern SSE/AVX arithmetic on every
  tier, but arithmetic does not set its sticky exception flags and unmasked SIMD exceptions are
  not raised. Some x87 paths, such as stores to memory, ignore rounding control, and legacy forms
  such as `fnsave`/`frstor` are refused.
- **Memory ordering covers JIT inline accesses only.** Interpreter accesses and helper-executed
  accesses (string ops, x87, FXSAVE, masked stores) get no tier barriers, and AArch64 LL/SC
  locked operations lack full x86 ordering.
- **Translations are never reclaimed.** A guest that keeps patching its code grows host memory
  with every patch. When JIT code space runs out, each vCPU returns `Exit::CodeSpaceExhausted`
  once and continues interpreted; `JitBackend::set_compile_limit` caps compilation earlier.
- **`Real16` is interpreter-only**, and there is no protected-mode segment-descriptor model:
  `Compat32` is flat apart from the FS/GS bases.
- **Host-fault recovery depends on the host.** [decision-13] lists which hosts are
  runtime-validated, cross-checked or pending, and the remaining partial-store gap.
- **Real programs are tested elsewhere.** Whole programs (busybox, sqlite, CPython, Go servers)
  run in `unemulinux`. [`scripts/ladder.sh`](scripts/ladder.sh) drives that ladder from here,
  but this repository's CI does not gate on it yet.
- **Pre-1.0 API.** Expect breaking changes between `0.x` releases.

## License

Dual-licensed under the [Apache License, Version 2.0](LICENSE-APACHE) or the
[MIT license](LICENSE-MIT), at your option. Every dependency of the library crates
(`x86jit-core`, `x86jit-cranelift`, `x86jit-elf`) is permissively licensed; the GPL-2.0 Unicorn
oracle is an optional, off-by-default feature of the unpublished test crate. `cargo deny`
enforces this, and [`PROVENANCE.md`](PROVENANCE.md) §5 has the analysis.

[spec]: backlog/docs/design/spec.md
[coverage]: backlog/docs/compat/isa-coverage.md
[decision-11]: backlog/decisions/decision-11%20-%20CPUID-tables-and-per-vCPU-overrides-amend-decision-9.md
[decision-12]: backlog/decisions/decision-12%20-%20Shared-CPU-source-handles-for-time-and-entropy.md
[decision-13]: backlog/decisions/decision-13%20-%20Signal-context-landing-pads-for-compiled-guest-faults.md
