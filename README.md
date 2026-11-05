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
ways — interpreter vs JIT, and both against a real CPU (Unicorn + native execution). The
suite runs on an **x86-64** CI runner and on an **AArch64** host before every merge that
touches codegen or host-specific code, so the ARM host path is validated, not assumed.

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
(16-bit real mode). **Real16 always runs on the interpreter**, even when the VM is
configured with a JIT backend, tier-up, or superblocks. Its interpreter is validated
against the hardware-captured
[SingleStepTests 80286 corpus](x86jit-tests/tests/ss286.rs), comparing final CPU and
memory state and in-guest exception delivery. The corpus reports remaining gaps;
it is not a claim of full 286 conformance. Targeted
[Unicorn comparisons](x86jit-tests/tests/cf16.rs) also cover real-mode addressing
and control flow.

**Engine:** two interchangeable backends — a portable interpreter and a Cranelift
JIT — over a single IR, with a translation cache, hotness-gated tier-up, superblock
regions, and block chaining + indirect-branch caching for fast dispatch. Self-modifying
code stays coherent and multiple guest threads share one VM. On weak (ARM) hosts,
`MemConsistency` governs JIT **inline** loads/stores only: `Fast` uses bare
LDR/STR; `AcqRel` adds DMB ISH before stores and after loads; `FullTso` adds
DMB ISH after each load and store. Helper-executed accesses lack these tier
barriers (xj-0w3.7). The interpreter, including cold blocks before tier-up and
fallback execution, ignores the tier and leaves ordinary accesses Relaxed
(xj-0w3.6); neither ordered tier currently guarantees interpreter TSO.

Explicit `mfence` and the serializing instructions we lift (`cpuid` and Real16
`iret`) execute a full host barrier on both backends in every tier. The JIT also
emits DMB ISH for `lfence`/`sfence` in every aarch64 tier, covering helper accesses;
they need no extra host barrier on x86-64 while guest non-temporal stores use
ordinary host stores. Locked operations remain atomic in every tier, but full
x86 locked-op ordering on aarch64 is not guaranteed: Cranelift's non-LSE LL/SC
sequences have no trailing DMB, and memory `xchg` uses LL/SC even with LSE.

**Performance.** Not yet optimized — expect roughly an **order of magnitude slower than
native** for hot code (a tight scalar loop is ~20× native on the JIT; the interpreter is
~40–250×), and worse for startup-heavy or run-once code, where the JIT pays to compile
everything up front. Throughput work is ongoing. The `x86jit-bench` crate records
interp/JIT/native timings per commit if you want real numbers.

Long64 data references enforce 48-bit canonical addressing: a non-canonical SS
reference raises `Exit::Exception` vector 12, other references vector 13. Required
SIMD and FXSAVE/FXRSTOR operand alignment raises vector 13 before any state commits;
error code zero is implied. Legacy SSE follows the Intel alignment rule. AMD's
optional MXCSR.MM relaxation is not modeled (MXCSR itself is not modeled).
Sources: Intel SDM Vol 1 §§3.3.7.1, 14.9, 15.7; AMD64 APM Vol 1 §4.3.2.

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
  `MOV r/m16|r32|r64, Sreg` reads the configured `CpuState` selector (including
  FS/GS). Protected-mode `MOV Sreg, r/m` and `POP Sreg` lift on every tier and
  return an exact `Exit::UnknownInstruction` before any state change or operand
  access; the embedder validates descriptors and emulates these loads.
- User-mode PUSHF/POPF and IRET run at CPL 3. The embedder configures valid
  CPL 3 selectors, `flags.if_` and `rflags_system` (IOPL and other system bits); defaults are zero.
  POPF preserves IOPL/VM/VIF/VIP, clears RF, and changes IF only when IOPL is 3;
  AC/ID are writable only by wide forms. TF and NT are stored; single-step
  and alignment-check delivery are absent. IRET accepts only the configured
  same-mode CS/SS pair; Long64 restores RIP, CS, RFLAGS, RSP and SS, while Compat32 restores EIP, CS
  and EFLAGS and advances ESP. Different selectors, a nested-task return or a
  noncanonical Long64 RIP raise #GP before state commits. The exception exit
  currently carries the vector and instruction address, without a selector error
  code. Sources: Intel SDM Vol 2B §4.3 MOV, POP, PUSHF, POPF (Table 4-20);
  Vol 2A §3.2 IRET; Vol 3A §7.14.3.
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
  does not honor `MemConsistency::AcqRel` or `FullTso` (xj-0w3.6). An embedder relying on either
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

`Memory::from_host_ram` and `Vm::with_backend_host_ram` return a `Result`.
Host backing must preserve 16-byte guest alignment: `ram.ptr as usize % 16`
must equal `ram.guest_base % 16`. Invalid alignment returns
`HostRamError::Misaligned`; invalid extents also return a `HostRamError`.
Page-aligned backing with a page-aligned guest base, and identity mappings
(`guest_base == ptr as u64`), satisfy this requirement. Owned memory is allocated
with 16-byte alignment. This lets interpreted and compiled `cmpxchg16b` use the
same host atomic operation.

Guard-backed JIT memory can recover inline guest-access host faults through
`cpu.fault_handle()` and `jit_abi::fault::classify`. The embedder owns the
SIGSEGV/SIGBUS handler and copies the returned host registers into its signal
context, including the scratch register the classifier sets (x9 / r11), and on
AArch64 LR and d8–d15. The embedder derives `HostFaultKind` from the OS fault
reason; macOS AArch64 needs the saved ESR to distinguish access from alignment.
FreeBSD amd64/arm64 accepts SIGSEGV with SEGV_MAPERR (1) or SEGV_ACCERR (2),
and SIGBUS with compatibility page-fault code 12 (UCODE_PAGEFLT), as Access;
SIGBUS/BUS_ADRALN (1) is Alignment. Other codes, including user-sent signals,
BUS_ADRERR (2), BUS_OBJERR (3) and BUS_OOMERR (100), are Unsupported. FreeBSD
arm64 alignment recovery ignores `si_addr`, which names the host PC, and replays
from guest RIP after validating the active handle, exact site and compiled frame.
Access faults on that host, and every kind on the other hosts, require an address
in the guest span. These facts follow FreeBSD 14.3 `sys/sys/signal.h`,
`sys/vm/vm_fault.c` and the amd64/arm64 trap sources; they are not pinned in
`oracles/`. See decision-13 below for source links and ucontext layouts.
The handler must not depend on the interrupted FP mode. Returning from it reaches
a mangled core landing stub, which restores host FP control before control returns
to `call_block`'s caller. Access faults produce a resumable `Exit::UnmappedMemory`;
alignment faults replay in the interpreter. Handles must
remain live, and a rejected fault must chain to the previous signal disposition.
See [decision-13](backlog/decisions/decision-13%20-%20Signal-context-landing-pads-for-compiled-guest-faults.md)
for supported hosts, signal safety, and the protected-mapped-page fallback.
Compiled calls stay direct; recovery uses compiler-provided unwind recipes and
one record published per run and cleared on return or panic. macOS x86-64 is
compile-unverified.
FreeBSD amd64/arm64 recovery and test adapters are implemented; runtime validation
of `fault_landing_pad` and `rust_memory_faults` on both FreeBSD architectures is
pending.

Embedders that protect RAM independently of the region table enable
`vm.mem.set_host_fault_recovery(true)` before execution;
`Memory::host_fault_recovery()` reads the setting. Default mode performs no Rust
transfer routing: registered RAM must remain host-accessible to interpreter,
helper and fetch accesses. Compiled-code faults recover in both modes through
decision-13. Routed Rust transfers use the same classifier, including with a zero
or inactive handle on threads with no active vCPU. Copy back PC and scratch and
return normally from the handler; the routine returns its fault to Rust and
ordinary destructors run. Alignment faults inside a fixup routine are rejected.

Instruction decoding always uses a private snapshot. Fetch starts with at most
128 bytes, bounded by the host page and region; it extends only when decoding
needs more bytes. Single-step fetch needs at most 15 bytes. In default mode these
copies use `ptr::copy_nonoverlapping`. A routed copy retains its readable prefix
on a fault. No cache lock is held while copying.

The public API changes are:

- `MemTrap` is `#[non_exhaustive]` and includes payload-free `HostFault`.
  Downstream matches need a wildcard. `MemTrap::fault_addr(transfer)`
  returns the exact byte for a host fault; call it immediately on the accessing
  thread, before another failing transfer can replace the thread's return slot.
- `Memory::code_slice` returns `Result<Vec<u8>, MemTrap>`; callers borrow the
  owned snapshot when decoding. `Memory::set_host_fault_recovery` and
  `Memory::host_fault_recovery` select and inspect the opt-in policy.
- `RawFpMem` and `RawStrMem` have a public `recover_host_faults: bool` field;
  existing struct literals must initialize it. `RawFpMem::exec_fxstate` and
  `RawFpMem::exec_x87` select the policy once per helper call.
  `RawStrMem::probe_store` explicitly invokes a checked write probe.
- `FpMem` provides `fault_addr`, `host_fault_recovery`, `load_checked` and
  `store_checked` methods, with defaults for existing implementors.
  `StrMem::preflight_store` provides a default no-op hook for selected store pages.
  `interp::raw_string_run_interruptible` selects a raw string helper's policy
  once per instruction.

Routed Rust copies and vector stores preflight destination pages. A write probe
atomically writes one byte's current value back on each touched host page: guest
data is unchanged, but copy-on-write and dirty tracking are affected even if a
later page faults. Serialize protection revocation with the complete transfer if
all-or-nothing writes are required. Compiled split vector stores and AArch64
misaligned scalar stores crossing a host page can retain completed writes on a
fault and rewrite those bytes on resume; this gap is tracked in xj-obx.8. Routed
REP MOVS/STOS remains per element and retains completed iterations. An x87
memory fault restores instruction-entry status in both modes.

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

Instruction fetch continues across adjacent mapped RAM/ROM regions. An unmapped
byte or a `RegionKind::Trap` byte stops fetch, even when the Trap region is
executable; Trap backing is never decoded. The resulting `Exit::UnmappedMemory`
uses `AccessKind::Execute` and reports the first unavailable or first Trap byte
as `addr`, while RIP stays on the attempted instruction's start. In Real16, `addr`
is the physical fetch address and RIP is the instruction's IP offset within CS.
Earlier instructions complete first; the faulting instruction has no effects.
Trap data reads and writes use the MMIO exits and completion methods.

In Long64 and Compat32, an invalid opcode raises `Exit::Exception` with vector 6
(#UD); an instruction exceeding 15 bytes raises vector 13 (#GP). Both leave `addr`
and RIP on the faulting instruction after earlier instructions complete (Intel SDM
Vol 3A §7.15, Events 6 and 13; Vol 2A §2.3.11). A valid encoding the lifter does not
handle returns `Exit::UnknownInstruction`.

For a complete CPU identity, install a CPUID table before creating vCPUs:

```rust
use x86jit_core::{CpuidLeaf, CpuidOutOfRange, CpuidOverride, CpuidOverrides,
                  CpuidRegisters as R, CpuidTable};

let mut table = CpuidTable::new(CpuidOutOfRange::HighestBasicLeaf);
// SDM Vol 1 Tables 21-4/6/8: vendor, signature, and leaf-1 fields.
table.leaves.insert(0, CpuidLeaf::Constant(R::new(
    1, 0x756e_6547, 0x6c65_746e, 0x4965_6e69)));
table.leaves.insert(1, CpuidLeaf::Constant(R::new(0x0003_06c3, 0x0002_0800, 0, 0)));
vm.set_cpuid_table(Some(table));

let mut overrides = CpuidOverrides::default();
// Initial APIC ID: SDM Vol 1 Table 21-8.
overrides.leaves.insert((1, 0), CpuidOverride {
    mask: R::new(0, 0xff00_0000, 0, 0),
    value: R::new(0, 1 << 24, 0, 0),
});
let vcpu = vm.new_vcpu_with_cpuid_overrides(overrides);
```

This is a minimal API example, not a complete CPU profile. `CpuidLeaf::Indexed`
holds a `(subleaf, registers)` map and an `invalid_subleaf` reply; `Constant` ignores
input ECX. The table also specifies replies for missing leaves and queries outside
the maxima in leaves 0 and 0x8000_0000. `HighestBasicLeaf` follows SDM Vol 1
§21.1.3; `Fixed` supplies a profile-specific reply. Explicit entries can describe
software-defined ranges. A table replaces all preset CPUID replies. Set `None`
to restore the selected `GuestCpuFeatures` projection for new vCPUs; existing vCPUs
retain their description. That legacy projection returns zeros outside its ranges,
a compatibility deviation from the SDM rule.

Overrides apply masked fields after lookup, keyed by the resolved leaf and subleaf
(zero for leaves that ignore ECX). They can carry logical-processor IDs and runtime
fields such as OSXSAVE, leaf 0xD sizes, 07H ECX[4] OSPKE and 19H EBX[0]
AESKLE (SDM Vol 1 §21.1.5, Table 21-1), and 80000001H EDX[11] SYSCALL
(Table 21-90). The legacy projection keeps SYSCALL set in every mode, a frozen
deviation from Table 21-90, which specifies zero outside 64-bit mode.
`CpuidOverrides::xcr0` supplies XGETBV(0); otherwise the feature preset supplies it.
The embedder keeps those fields consistent and can update `vcpu.cpu.cpuid_overrides`
when its modeled state changes. Tables and overrides survive CPU-state cloning;
VM forks inherit the table.

Advertising a bit does not implement its instructions. In development tooling,
`x86jit_tests::compat::cpuid_lift_gaps(|leaf, subleaf| vcpu.cpu.cpuid(leaf, subleaf))`
reports incomplete or unknown user-mode lift coverage for an effective CPU source.
The same helper accepts a preset's `cpuid` or a table's `query`. It covers Intel/AMD
instruction-feature bits understood by iced, not legacy vendor-specific leaves or
CPU-description consistency. The preset tests carry explicit per-preset waivers,
including CX8, XSAVE and CX16 gaps. In particular, supplying leaf 0xD sizes can
select XSAVE/XRSTOR paths that currently trap. See
[decision-11](backlog/decisions/decision-11%20-%20CPUID-tables-and-per-vCPU-overrides-amend-decision-9.md).

Embedders can install time and entropy through a shared `CpuSource` handle:

```rust
use std::{sync::Arc, time::Instant};
use x86jit_core::{CpuSource, RandomKind, TscSample};

#[derive(Debug)]
struct Source(Instant);

impl CpuSource for Source {
    fn read_tsc(&self, vcpu_id: u64) -> TscSample {
        TscSample {
            counter: self.0.elapsed().as_nanos() as u64,
            aux: vcpu_id as u32,
        }
    }

    fn read_random(&self, _vcpu_id: u64, _kind: RandomKind, _size: u8) -> Option<u64> {
        None // Supply entropy here; None makes the instruction report CF=0.
    }
}

vm.set_cpu_source(Some(Arc::new(Source(Instant::now()))));
let mut cpu = vm.new_vcpu();
```

The engine samples the source exactly once per executed RDTSC, RDTSCP, RDRAND
or RDSEED. `source_id` identifies the requesting vCPU independently of CPUID IDs;
RDTSCP returns the source's paired `aux` in ECX. The source owns rate, offset,
cross-vCPU monotonicity and entropy quality. Its behavior must agree with any
advertised invariant-TSC bit and CPUID frequency leaves 0x15/0x16 (SDM Vol 3B
§20.17/§20.17.1; Vol 1 §21.3 CPUID.15H/16H). Installing a source does not
advertise these instructions: select `Feature::Rdtscp`, `Rdrand` and `Rdseed`
or supply a consistent CPUID table. They remain absent from the default preset.

For RDTSCP, the engine guarantees that all earlier guest instructions have retired
and their memory accesses are ordered by a host fence before calling `read_tsc`.
A source reading a hardware counter must itself use an execution-synchronising
read so that sampling cannot run ahead of earlier guest loads: on x86-64, RDTSCP
or LFENCE;RDTSC (SDM Vol 2B RDTSCP/RDTSC, Description); on AArch64, ISB before
CNTVCT_EL0, or the self-synchronising CNTVCTSS_EL0 with FEAT_ECV (see the Arm
Architecture Reference Manual, generic timer). A memory fence alone does not
synchronise the counter read. RDTSC has no ordering requirement (SDM Vol 2B
RDTSC, Description). Both instructions call the same method, so hardware-counter
sources must provide RDTSCP ordering on every call.

With no source, RDTSC and RDTSCP return `0x12345678`, TSC_AUX is zero, and
RDRAND/RDSEED share a deterministic per-vCPU sequence with CF=1. This fallback
is test data and provides no cryptographic entropy. Existing vCPUs retain their
handle when the VM setting changes; `CpuState::cpu_source` and
`CpuState::source_id` can be replaced between runs. CPU-state clones retain the
source identity and share the handle; VM forks share the handle and fresh-ID
allocator. External source state is not snapshotted by either operation. Source
methods must not panic or reenter
execution across the JIT helper boundary. The accepted design is reflected in
spec §1/§14; see
[decision-12](backlog/decisions/decision-12%20-%20Shared-CPU-source-handles-for-time-and-entropy.md).

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
