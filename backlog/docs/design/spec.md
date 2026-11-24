---
id: doc-13
title: 'Specification: `x86jit` — an x86-64 → host recompiler as a Rust library'
type: specification
created_date: '2026-07-06 11:25'
---

# Specification: `x86jit` — an x86-64 → host recompiler as a Rust library

Working draft 0.4 · design document for implementation with Claude Code

> **Changes 0.3 → 0.4 (implementation audit — holes that would break at build time):** IR gains flag-as-data (`GetCond`) and carry-consuming ops (`Adc`/`Sbb`) — without them `adc`/`sbb`/`setcc`/`cmovcc` cannot be lifted; `set_flags: bool` replaced by a `FlagMask` (`inc`/`dec` preserve CF; shift-by-0 leaves all flags untouched — a runtime-conditional update); added pitfall #0 **instruction atomicity** (every potential trap of an instruction must precede its first state commit, or the RIP-retry convention corrupts state — e.g. `push` moving RSP before a faulting store); `complete_mmio_read` redesigned as a **pending value consumed by the retried load** (the old "write into the temp" contradicted temps dying when a block returns, and could never work in the JIT); guest-RAM access contract switched to **interior mutability** (`&Memory` + `UnsafeCell`, manual `Sync`) — shared RAM written by multiple vcpus cannot flow through `&mut` (hits already in M1 as a borrow conflict); backend selection changed from a config enum to an **injected `Backend` trait object** (the core cannot name the downstream JIT crate — dependency direction); §8.2 gains a mandatory **memory-safety strategy for inlined access** (bounds/permission check vs guard pages) — raw `host_base+addr` without checks is host UB on any out-of-range guest address; notes added on budget-vs-block-chaining, the benign double-lift race, and same-block SMC; the native-oracle `hlt` caveat documented in testing.md (hlt is privileged); §14 gains open decisions (#DE representation, TSO tagging, breakpoint mechanism); `enforce_tso: bool` replaced by the three-tier **`MemConsistency`** knob (`Fast` bare STR/LDR · `AcqRel` DMB-before-store/after-load · `FullTso` STR+DMB) — a per-workload escalation ladder on weak hosts, identical code on x86 hosts, locked ops/`mfence` tier-independent, tier switch = cache flush (§4.1, §8.2.3).
>
> <details><summary>Changes 0.2 → 0.3 (structural review — minimizing surprises)</summary>
>
> **Changes 0.2 → 0.3 (structural review — minimizing surprises):** added section 7.1 **operand lowering** (effective address, memory operand, read-modify-write) — the biggest gap so far; added 7.2 temporaries generator + the `IrBlock.temp_count` field; separated the backend's responsibilities (backend-dependent materialization + uniform `execute` over `CachedBlock`) — removes the false `execute(&IrBlock)`; added 8.2.1–8.2.3 **compiled block ABI** (access to `CpuState` via pointer + offsets, result encoding, M4 build order); fixed `CachedBlock`/`CompiledPtr` with respect to `Send`/`Sync` (pitfall M7) and ownership of code memory; added a dependency map, table of contents, and a consolidated "where the surprises live" section; added section 17 **extensibility points (seams)** with `CpuMode`/`BlockKey` sketches and three `SEAM` markers in the code (decoder bitness, cache key, effective-address helper).
>
> <details><summary>Changes 0.1 → 0.2 (technical review)</summary>
>
> Fixed the contradiction in the backend's return type (`StepResult` instead of a direct `Exit`); split `Exit::Mmio` into `MmioRead`/`MmioWrite` with the correct data direction + `complete_mmio_read`; unified the types in `IrOp::Branch` (both branches static) and clarified `Cond` (signed vs unsigned); rewrote the dispatcher loop into something type-consistent and compilable (lift error → `Exit`, not `?`); added `LiftError`, the `Memory`/`MemTrap` contract, clarified `Flat` vs `map()`; documented two x86-64 semantic pitfalls (zeroing of the upper 32 bits, RIP-relative).
> </details>
> </details>

---

## How to read this document

Sections 1–5 are the **public contract** (what a user of the library sees) — read them first, because they define the boundaries. Sections 6–11 are the **internals** (how it works). Section 12 is the **plan** — implement in milestone order, not section order. Section 16 ("where the surprises live") gathers all the pitfalls in one place — check it before every milestone.

**Dependency map (what must exist before what):**

```
CpuState + Memory (3,4)
      │
      ▼
IR: IrOp + Val + TempGen (6, 7.2)
      │
      ▼
operand lowering (7.1) ──► mnemonic lift (7.3)
      │
      ▼
interpreter (8.1) ◄─── shared StepResult contract (8)
      │                        │
      │                        ▼
      │                 cache + dispatcher (9)
      ▼                        │
differential test (13)         ▼
                        JIT/Cranelift (8.2) ── needs block ABI (8.2.1)
                               │
                               ▼
                        SMC (10) ─► multithreading + TSO (11)
```

Key observation: **operand lowering (7.1) is the foundation of the lift** — without it you can't lift even `mov`, because you can't reduce a memory operand to a `Val`. Build it *before* lifting concrete instructions.

**Table of contents:**
1. Goal and scope · 2. Architecture · 3. Guest state · 4. Input API · 5. Output API · 6. IR · 7. Lift (7.1 operand lowering, 7.2 temps, 7.3 loop) · 8. Backends (8.1 interpreter, 8.2 JIT+ABI) · 9. Cache + dispatcher · 10. SMC · 11. Multithreading · 12. Milestones · 13. Testing · 14. Open decisions · 15. Dependencies · 16. Where the surprises live · 17. Extensibility points (seams)

---

## 1. Goal and scope

`x86jit` is a pure-Rust library that **executes x86-64 code on any host** (x86-64 or ARM64), via JIT recompilation. The core is **guest-agnostic** — it knows nothing about guest platforms, ELF, the syscalls of any particular OS, or the GPU. It is a "CPU engine" that receives memory + a starting point and executes instructions, handing back control every time it encounters something it can't handle on its own.

### What is in scope (library core)

- x86-64 decoding (via `iced-x86`).
- Lifting instructions to a custom IR.
- Two backends: **interpreter** (reference, slow, debuggable) and **JIT** (via Cranelift).
- A translation cache keyed by guest address.
- A dispatcher loop.
- Guest memory model (flat buffer + softmmu later).
- Guest CPU state (registers, flags, RIP).
- Return-based output API (`run()` → `Exit`).

### What is OUT of scope (belongs to the user / separate crates)

- Parsing file formats (ELF, SELF, PE) — the user provides an already unpacked memory map.
- Handling the syscalls of a specific OS (HLE) — the user reacts to `Exit::Syscall`.
- MMIO / devices / GPU — the user reacts to `Exit::MmioRead` / `Exit::MmioWrite`.
- Decryption, loaders, higher-level guest thread scheduling.

> **Boundary rule:** the core communicates with the embedder through **exit reasons** (`Exit`) and **Trap-type memory regions**. The one exception is an optional `CpuSource` handle for time and entropy (RDTSC, RDTSCP, RDRAND, RDSEED), described in §14 and decision-12. Ordinary guest RAM access is always inlined into the generated code and never goes through a callback.

---

## 2. High-level architecture

```
                         ┌─────────────────────────────────────────┐
   x86 bytes (guest RAM) │  iced-x86 ──► lift ──► IR ──► backend    │  ──► host code / execution
                         │                              ├ interpreter│
                         │                              └ Cranelift  │
                         │        translation cache (guest→host)     │
                         │        dispatcher loop                    │
                         └─────────────────────────────────────────┘
                                        ▲            │
                                        │ Exit       │ inline mem access (hot)
                                   user loop     guest RAM buffer
```

### Split into two entities (modeled on KVM: VM vs VCPU)

- **`Vm`** — shared: guest memory + translation cache. One instance per emulated machine.
- **`Vcpu`** — per guest thread: CPU state (registers, flags, RIP) + its own `run()` loop. Shares the `Vm`.

This split opens the road to multithreading from the start (many `Vcpu` over a single `Vm`), even if the first version is single-threaded.

### Workspace layout (proposal)

```
x86jit/
├── x86jit-core/        # Vm, Vcpu, IR, lift, cache, dispatcher, interpreter
├── x86jit-cranelift/   # JIT backend (feature-gated, optional)
├── x86jit-elf/          # OPTIONAL helper: ELF parser → memory map (convenience, not core)
└── x86jit-tests/        # differential testing, instruction corpus, fuzzing
```

---

## 3. Guest state model

### 3.1 Registers

```rust
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Reg {
    Rax, Rbx, Rcx, Rdx, Rsi, Rdi, Rbp, Rsp,
    R8, R9, R10, R11, R12, R13, R14, R15,
    Rip,
    // segment bases — FS/GS are used for TLS (thread-local storage),
    // real programs require this, so they must be present from the start.
    FsBase, GsBase,
}
```

Internally we keep state as a flat struct (not a `HashMap` — this is a hot path):

```rust
pub struct CpuState {
    pub gpr: [u64; 16],     // indexed by x86 order (RAX=0, RCX=1, ...)
    pub rip: u64,
    pub fs_base: u64,
    pub gs_base: u64,
    pub flags: Flags,       // see 3.2
    // XMM/YMM (SIMD) — added in a later milestone, not at the start:
    // pub xmm: [u128; 16],
}
```

> **Note on GPR numbering:** the register order in x86 encoding (RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8..R15) differs from the "natural" one. `iced-x86` returns a `Register` enum — map it once, in one place, to the `gpr[]` index.

### 3.2 Flags

The contract requires that at every exit, including a mid-block fault, flags are the architectural flags after the last retired instruction (Intel SDM Vol 3A §7.5; AMD64 APM Vol 2 §8.1.3).

A host guard-page fault inside compiled code does not yet meet this contract: it reports flags as of the last flag store until xj-obx.2/xj-obx.4 land.

x86 has arithmetic flags: **CF** (carry), **PF** (parity), **AF** (adjust), **ZF** (zero), **SF** (sign), **OF** (overflow), and **DF** (direction, for string instructions).

Two strategies — the spec assumes the **simple version at the start, lazy as an optimization**:

**Variant A (start): materialized flags.** After every instruction that sets flags, you compute them right away and store them as bits.

```rust
#[repr(C)]   // JIT stores/loads flag fields at stable offsets in CpuState (§8.2.1)
pub struct Flags {
    pub cf: bool, pub pf: bool, pub af: bool,
    pub zf: bool, pub sf: bool, pub of: bool,
    pub df: bool,
}
```

> **`#[repr(C)]` on `Flags` is part of the ABI, not cosmetic.** `CpuState` is `#[repr(C)]` so codegen can address fields by offset (§8.2.1) — but a nested non-`repr(C)` struct has unspecified field layout, so the flag offsets the JIT bakes in would be undefined behavior to rely on. Mark `Flags` `#[repr(C)]` too. One-byte bools are fine to start; a packed RFLAGS-style `u64` (fewer stores per flag update in codegen) is an M4/M5 optimization.

**Variant B (optimization later): lazy flags.** Instead of always computing flags, you store "the last operation and its operands", and you compute a flag only when someone reads it (because most set flags are never read). This is a significant performance gain, but it complicates the IR — **don't do it at the start.**

```rust
// Variant B — sketch, to be implemented only after a working JIT:
pub struct LazyFlags {
    pub last_op: FlagOp,   // Add, Sub, Logic, ...
    pub a: u64, pub b: u64, pub result: u64, pub size: u8,
}
```

---

## 4. Input API (input)

### 4.1 Construction and configuration

```rust
pub struct VmConfig {
    /// Memory model. Start: Flat. Later: SoftMmu.
    pub memory_model: MemoryModel,
    /// Memory-CONSISTENCY tier for generated code on weak hosts (ARM) — three
    /// speed/strictness levels, per-Vm (a "per game" knob). On an x86 host all
    /// tiers emit identical code (native TSO). See §8.2.3.
    pub consistency: MemConsistency,
}

/// How faithfully guest accesses reproduce x86-TSO ordering on a weak host.
/// Escalation ladder: run Fast; if a multithreaded workload misbehaves, bump one
/// tier. Distinct from `MemoryModel` (address-space layout) — this is ORDERING.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum MemConsistency {
    /// Bare STR/LDR, no barriers. Fastest. Correct ONLY for code that doesn't
    /// synchronize through memory: single-threaded guests, or multithreaded ones
    /// whose threads don't communicate via shared structures.
    Fast,
    /// JIT inline stores: DMB ISH; STR. Inline loads: LDR; DMB ISH.
    /// Interpreter accesses and guest-memory helpers use matching barriers (§8.2.3).
    AcqRel,
    /// JIT inline stores: STR; DMB ISH. Inline loads: LDR; DMB ISH.
    /// No leading store fence. Interpreter accesses match; read helpers are fenced
    /// before and after the call (§8.2.3).
    FullTso,
}

pub enum MemoryModel {
    /// One contiguous host buffer of size `size`, representing the guest
    /// address space [0, size). Address translation = host_base + guest_addr
    /// (one addition — fastest). `map()` in this model only assigns
    /// permissions/type (Ram/Trap) to regions and checks bounds; it does not
    /// allocate separate blocks. Canonical addresses >= size are UnmappedMemory;
    /// see Data address exits for non-canonical Long64 operands.
    /// Good at the start and when the guest space is dense and fits in host RAM.
    Flat { size: u64 },

    /// The same contiguous translation over a potentially huge reserved span.
    /// Use a sparse constructor or HostRam for sparse host commitment.
    Reserved { span: u64 },

    /// Sparse address space via a page table / region map.
    /// `map()` allocates separate pages. Slower address translation (lookup),
    /// but handles scattered, high addresses without allocating the whole thing.
    /// You implement this when Flat stops being enough (e.g. the guest uses
    /// addresses near the top of the 64-bit space).
    SoftMmu,
}
impl Vm {
    /// Default: the interpreter backend (lives in the core).
    pub fn new(config: VmConfig) -> Self;
    /// Inject a backend — this is how the JIT gets in (see the note below).
    pub fn with_backend(config: VmConfig, backend: Box<dyn Backend>) -> Self;
}
```

> **Backend selection is an injected trait object, NOT a config enum.** An earlier draft had `pub enum Backend { Interpreter, Jit }` in `VmConfig` — that cannot work: `x86jit-cranelift` depends on the core, so the core **cannot name or construct** the JIT backend (the dependency points the other way). The `Backend` *trait* (§8) is defined in the core; the interpreter implements it in the core; `x86jit-cranelift` exports a `JitBackend` implementing the same trait, and the user passes it into `Vm::with_backend`. This keeps the JIT crate optional without a feature-flag knot in the core.

> **Flat vs map():** in the `Flat` model, `size` is the total size of the allocated host buffer. `map()` does NOT add memory — it divides the already existing buffer into regions with different permissions and type (Ram/Trap). In `SoftMmu`, `map()` really allocates pages. Keep this distinction clear, because otherwise `map(guest_addr = 0x7fff_0000_0000, ...)` in Flat mode would mean an attempt to allocate 128 TB.

**Sparse host backing.** Backing allocation is separate from the memory model:
`Vm::sparse(config) -> Result<Vm, SparseError>` selects the interpreter;
`Vm::with_backend_sparse(config, backend)` selects an injected backend. Both accept
the existing `VmConfig::flat(size)` or `VmConfig::reserved(span)` and reserve the
span without committing it. `Memory::sparse(model)` supplies the same backing for
direct memory users. The old constructors retain their allocation policy, and
`HostRam` remains available for caller-managed reservations. There is no new core
dependency. Zero spans, spans above the lower canonical half, and spans that cannot
fit a host pointer are `SparseError::InvalidSpan`; SoftMmu is
`SparseError::UnsupportedModel`. Other hosts return `SparseError::UnsupportedHost`.

Linux and macOS use direct FFI to `mmap(PROT_NONE, MAP_PRIVATE | MAP_ANON |
MAP_NORESERVE)`, then `mprotect(PROT_READ | PROT_WRITE)` on map. Linux unmap uses
`madvise(MADV_DONTNEED)`; macOS uses `MADV_FREE_REUSABLE` (and `MADV_FREE_REUSE`
before reuse); both close the pages with `mprotect(PROT_NONE)` and release the
reservation with `munmap` on drop. Windows uses `VirtualAlloc(MEM_RESERVE,
PAGE_NOACCESS)`, `VirtualAlloc(MEM_COMMIT, PAGE_READWRITE)`,
`VirtualFree(MEM_DECOMMIT)` and `VirtualFree(MEM_RELEASE)` respectively. Host page
size comes from `getpagesize` or `GetSystemInfo`, independent of the 4 KiB guest
tracking granule. Mapping commits rounded host pages before publishing the region
tag, and zeroes only the new region's bytes. Partial pages shared with other live
regions stay accessible; unmap discards only unshared whole pages. Trap regions
also have backing for the host inspection/loader API, but guest accesses still
dispatch to the device. `Prot` retains its existing advisory semantics; select
`RegionKind::Rom` to discard guest writes to firmware.

Reservation/metadata failures return `SparseError`; commit/decommit failures return
`MapError::Sparse(SparseError)`. A failed map publishes no tag. A failed unmap
removes the tag and invalidates translations even if the OS could not discard the
pages, retaining address ownership until drop. These errors report synchronous
host API failures: anonymous Unix memory still follows the host's overcommit/OOM
policy, so success cannot guarantee that the OS will service every future page
fault. No whole-span zeroing or copying is performed.

Sparse backing executes **all** guest instructions through the checked interpreter,
even with an injected JIT backend. Block compilation, region formation and tier-up
are bypassed. This keeps retirement and step accounting exact without charging
compiled suffixes that later defer to the interpreter, and requires no host signal
handler. Ordinary backing's generated code is unchanged. Because `Vm::mem` is
public, run entry detects a sparse/ordinary backing-policy change and invalidates
translations, chained links, vCPU predictions and pending publications before
dispatch; switching back to ordinary backing can compile fresh code.
The interpreter checks containment/adjacent RAM pieces and device kinds before
dereferencing; fetch and lifter snapshots stop at region boundaries. Aligned
atomic containing granules stay within committed host pages even for sub-page
regions. Machine bus dispatch and child execution use their checked region views;
child views borrow the VM, preventing safe unmap while a view lives. SMC and dirty
tracking inspect metadata and fetched/mapped bytes, never scan guest backing.
`CpuState` copies, instruction checkpoints and `reset_execution_state` touch CPU
and execution metadata only. The existing whole-buffer deep-copy/fork path is
refused: `Memory::deep_copy` / `Vm::fork_with_backend` return `None`, while
`Vm::try_fork_with_backend` returns `ForkError::SparseBacking`. An embedder needing
a memory snapshot can inspect mapped regions explicitly with `read_bytes`.

### 4.2 Memory mapping

This is the **main input**: the user (or the ELF/SELF loader) maps segments to guest addresses. The core does not parse any format — it receives ready bytes and addresses.

```rust
pub enum Prot { R, RW, RX, RWX }

pub enum RegionKind {
    /// Ordinary RAM. Access is INLINED into the generated code — no trap-out.
    Ram,
    /// Trapped region. Every access causes Exit::MmioRead / Exit::MmioWrite
    /// (for devices, MMIO).
    Trap,
}

impl Vm {
    /// Reserve a region in the guest address space.
    pub fn map(&mut self, guest_addr: u64, size: usize, prot: Prot, kind: RegionKind)
        -> Result<(), MapError>;

    /// Write content (e.g. an ELF .text/.data segment) into an already mapped region.
    pub fn write_bytes(&mut self, guest_addr: u64, bytes: &[u8]) -> Result<(), MemError>;

    /// Read (for inspection / debugging / HLE reading guest structures).
    pub fn read_bytes(&self, guest_addr: u64, buf: &mut [u8]) -> Result<(), MemError>;

    pub fn unmap(&mut self, guest_addr: u64, size: usize) -> Result<(), MapError>;
}
```

A typical loading flow on the user's side (outside the core):

```rust
// pseudo ELF loader (lives in x86jit-elf or at the user's)
for seg in elf.load_segments() {
    vm.map(seg.vaddr, seg.memsz, seg.prot(), RegionKind::Ram)?;
    vm.write_bytes(seg.vaddr, &seg.data)?;
}
```

### 4.3 Creating a vcpu and setting the initial state

```rust
impl Vm {
    /// Create a new execution context (one per guest thread).
    pub fn new_vcpu(&self) -> Vcpu;
}

impl Vcpu {
    pub fn set_reg(&mut self, reg: Reg, val: u64);
    pub fn reg(&self, reg: Reg) -> u64;
    pub fn set_flags(&mut self, flags: Flags);
    pub fn flags(&self) -> Flags;
}
```

Setting the entry point = simply `set_reg(Reg::Rip, entry)`. Stack: the user allocates a RAM region and sets `Rsp`.

---

## 5. Execution and output API (output)

### 5.1 Return-based model

The core **does not call the user through a trait** in the hot path. Instead, `run()` executes until it encounters an event it can't handle, and **returns a reason**. The user handles the reason in a loop and calls `run()` again.

```rust
impl Vcpu {
    /// Execute guest code until an exit event or exhaustion of the budget.
    /// `budget` = maximum number of executed instructions (or blocks —
    /// see the note below). None = no limit (until the first Exit).
    pub fn run(&mut self, budget: Option<u64>) -> Exit;
}
```

> **The budget** serves cooperative scheduling of many guest threads: you execute N instructions of one vcpu, then switch to another. Without it, one thread in an infinite loop would starve the rest. Decide whether you count instructions or blocks — blocks are cheaper to count (increment once per block, not once per instruction), instructions give finer granularity. Recommendation: count **blocks**, budget in blocks.

### 5.2 Exit-reason enum

```rust
pub enum Exit {
    /// The guest executed a syscall instruction (syscall/sysenter/int 0x80).
    /// The arguments are in guest registers — the user reads them via vcpu.reg().
    /// After handling, the user sets the result register and calls run() again.
    Syscall,

    /// The guest executed a halt instruction (hlt).
    Hlt,

    /// Access to an address that is not mapped.
    UnmappedMemory { addr: u64, access: AccessKind },

    /// READ from a Trap region (MMIO). The guest is WAITING for a value.
    /// The user must supply the result via vcpu.complete_mmio_read(value)
    /// BEFORE the next run(), otherwise the guest state is inconsistent.
    MmioRead { addr: u64, size: u8 },

    /// WRITE to a Trap region (MMIO). The guest SUPPLIES a value.
    /// The user handles the side effect and simply calls run() again.
    MmioWrite { addr: u64, size: u8, value: u64 },

    /// Encountered an instruction that the lift does not yet handle.
    /// Invaluable during development: it tells you exactly what to add to the lift next.
    UnknownInstruction { addr: u64, bytes: [u8; 15], len: u8 },

    /// A guest CPU exception. Invalid encodings raise #UD (vector 6); an instruction
    /// longer than 15 bytes raises #GP(0) (vector 13). Long64 non-canonical data
    /// operands raise #GP(0), or #SS(0) (vector 12) for SS references. Required
    /// operand alignment raises #GP(0). Error code zero is implied for these
    /// #GP/#SS exits; addr and RIP identify the failing instruction.
    Exception { addr: u64, vector: u8 },

    /// Hit a breakpoint set by the user (debug).
    Breakpoint { addr: u64 },

    /// Executed `budget` blocks — cooperative yield of control.
    BudgetExhausted,

    /// Internal error (e.g. corrupted state, inconsistent cache).
    Fault(FaultKind),
}

pub enum AccessKind { Read, Write, Execute }
```

**Data address exits.** In Long64, every byte of a data operand must be canonical
(SDM Vol 1 §3.3.7.1; AMD APM Vol 2 §5.3.1). A non-canonical operand raises
`Exit::Exception { vector: 13, addr: RIP }`, or vector 12 for an implicit stack
access or an RSP/RBP-based operand. CS/DS/ES/SS overrides are ignored; FS/GS
references use vector 13 (SDM Vol 1 §3.3.7.1). Canonical misses, including the
canonical upper half, remain `UnmappedMemory` with the exact failing byte.
For forms with memory fault suppression, only active elements contribute to the
reference (SDM Vol 1 §15.6.1; Vol 2C, VMASKMOV/VPMASKMOV). Embedded-broadcast
fault carriers describe the decoded element (SDM Vol 2A §2.7.7); xj-au4.2 owns
the existing broadcast transfer/value implementation gap.
The lifter records mode, SS selection and whole operand width in access-op padding;
classification runs only after a memory miss. Configured spans and regions exclude
the canonical hole, so generated ordinary loads/stores need no canonicality test.
`CheckAlign` precedes the first access of forms requiring explicit alignment:
legacy SSE m128 except the unaligned and string-compare exceptions, aligned
vector moves at vector width, and FXSAVE/FXRSTOR at 16 bytes (SDM Vol 2A §2.5,
Tables 2-14/2-15/2-18/2-19/2-21; §2.8, Tables 2-45/2-46/2-47; FXSAVE/FXRSTOR).
Alignment failures raise vector 13. These #GP/#SS exits imply error code zero and
leave the instruction's registers, flags, memory and RIP uncommitted. This
contract does not specify priority when an operand violates multiple conditions.

**Decode and fetch exits.** In Long64 and Compat32, an architecturally invalid
encoding produces `Exit::Exception { vector: 6, addr }` (#UD), with `addr` and the
vcpu's RIP on that instruction (SDM Vol 3A §7.15, Event 6; AMD APM Vol 2 §8.2.7).
An instruction that exceeds the 15-byte length limit (SDM Vol 2A §2.3.11) produces
`Exit::Exception { vector: 13, addr }` (#GP(0), SDM Vol 3A §7.15, Event 13) when
all 15 bytes are fetchable; if a byte inside the first 15 cannot be fetched, the
fetch fault below is reported instead.
A valid encoding the lifter does not support produces `Exit::UnknownInstruction`,
also with `addr == RIP` on that instruction. Both preserve all completed earlier
instructions and leave the failing instruction without effects. Real16 delivers
invalid-opcode #UD through the guest IVT (§17.6).

An instruction fetch that runs out of mapped bytes produces
`Exit::UnmappedMemory { access: Execute, addr }`: `addr` is the first unavailable
byte, while RIP stays on the instruction's start (SDM Vol 3A §7.15, Event 14,
CR2 and Saved Instruction Pointer; AMD APM Vol 2 §8.2.15). Fetch continues across
adjacent mapped RAM/ROM regions, but never reads Trap backing. A fetch starting
in or crossing into Trap produces the same execute exit at the first Trap byte.
These rules apply to block, region and single-instruction execution: an unliftable
instruction ends the valid prefix, which executes before dispatch retries that
instruction; any partial IR emitted for the unliftable instruction is discarded.

**Data accesses across adjacent regions.** A data operand need not lie in one
region: an unaligned operand can take more than one bus cycle (SDM Vol 1 §4.1.1).
When every byte lies in adjacent RAM or ROM regions with no gap, the access is
performed: a load returns the bytes, a store commits the RAM bytes and drops the
ROM bytes exactly as a store inside a ROM region does. Every region tags the one
contiguous backing, so an all-RAM span takes the same single host operation as a
contained access, which keeps an aligned word, doubleword or quadword atomic
(SDM Vol 3A §11.1.1). A span with ROM bytes stores its RAM pieces one region at a
time in ascending address order (an engine rule, not a hardware claim); with host
fault recovery every RAM piece is probed before any byte commits. The rule holds
per transfer: a transfer touching a gap or a Trap byte keeps the containment miss
and its exit; splitting device accesses is xj-ir6.30. A 16-byte vector store is two
8-byte transfers; when the first is not inside one region, both are checked before
either is written, with or without recovery, so a gap in the second faults there
with nothing written. A LOCK or implicitly locked operand keeps the containment
miss, including each bus piece of one that the A20 gate splits; on the
twenty-bit models a locked word that wraps its segment or the bus is one
transfer per byte (§8.2.3, §17.10). An automatic child
view joins adjacent regions of one kind for non-atomic data accesses; a span that
mixes RAM and ROM keeps the containment fault in the child. Inline compiled
accesses bound only against the backing and never consult the region map, so they
perform such spans natively, and the `RegionKind::Rom` store limit applies to them
as it does to a contained ROM store.

> **Why MmioRead/MmioWrite are separated:** on an MMIO read, data flows FROM the user TO the guest (the guest wrote the address into the destination register and is waiting for a value); on a write — from the guest to the user. A single `value` field does not model both directions. A read additionally requires a mechanism for injecting the result:

```rust
impl Vcpu {
    /// Called after Exit::MmioRead: supplies the value the guest "read".
    /// Stored as a PENDING VALUE for (addr, size); the retried load consumes it.
    /// See the resume mechanism below.
    pub fn complete_mmio_read(&mut self, value: u64);
}
```

> **MMIO-read resume mechanism — pending value, NOT "write into the temp".** A naive contract ("the value lands in the destination temp, then run() resumes") cannot work: temps live in the executing block's local storage and **die the moment the block returns** with `Exit::MmioRead` — and a compiled JIT block cannot be re-entered mid-way at all. The working design, consistent with the RIP convention (§8):
>
> 1. On a load from a Trap region, the backend sets **RIP = the faulting instruction** and returns `Exit::MmioRead { addr, size }`. Per the instruction-atomicity rule (§7 pitfall 3), nothing of that instruction has committed yet.
> 2. `complete_mmio_read(value)` stores `(addr, size, value)` as a **pending MMIO value** on the `Vcpu`.
> 3. The next `run()` resumes at the faulting instruction — the dispatcher lifts a block starting there (any address can start a block, so partial-block progress is fine). The re-executed `Load` hits the Trap region again, the memory layer sees a matching pending value, **consumes it and returns it as the load result** instead of trapping. Execution flows on.
>
> This works identically in the interpreter and the JIT (the retried load is a fresh execution in both), costs one `Option` check on the Trap-slow-path only, and never requires resuming a half-executed block.

**Read-modify-write operands on Trap regions.** An instruction that reads and
writes a memory operand transfers it once in each direction: one `MmioRead`, then
one `MmioWrite` of the result. The 8086 Family User's Manual (9800722-03) Table 2-21
(pp. 2-51–2-67) counts two transfers for every `memory, register`,
`memory, immediate` and one-operand `memory` form of the arithmetic, logical,
shift, rotate, INC/DEC/NOT/NEG and XCHG instructions, and one for a
`register, memory` form and for CMP (p. 2-53). RIP stays on the instruction across
both exits, and none of its register or flag effects commit before the write is
acknowledged. This holds in every mode and at every tier: compiled code defers an
instruction with a Trap operand to the interpreter.

The retry after each completion re-executes the instruction. In the application
walks (the Real16 alias, Long64, Compat32) the answer it consumes is recorded in
the retry ledger, bound to its request: the owning instruction (RIP, CS and a
fingerprint of its lifted operations) and the access's address, size and
direction. The retry after the write acknowledgement takes the recorded answer
instead of reading the device again. A different instruction at the same RIP
(code written between the exits) does not match the fingerprint and reads the
device itself, and a recorded answer stands only while its access still stops at
a Trap region: remapped as RAM, or unmapped, between the exits, the access drops
it and runs as without one. A recorded answer lives across the instruction's own
`MmioRead`/`MmioWrite` stops and across the boundary stops taken before its retry
runs — `BudgetExhausted` (also from `run(Some(0))` or `run_steps(0)`) and a host
stop (`Interrupted`) — since the instruction has neither completed nor aborted
there. Any other stop of that instruction — a fault such as `UnmappedMemory`, a
delivered exception — ends it, as a fault aborts the instruction on the
processor, and the re-execution reads the device again; it is not reported as an
acknowledged transfer. The Machine walks keep their existing
ledger (decision-16 §3), and the alias's multi-read families share it (U3). The
pending answer of `complete_mmio_read` is still taken by the first load of the
retry that has no recorded answer; binding the pending answer itself to its
request is xj-ir6.13.

**Accesses of several transfers.** A vector operand wider than eight bytes is one
8-byte transfer per quadword, and a masked form one transfer per active element.
A string element between two operands is a read and a write (MOVS) or two reads
(CMPS). Each transfer is its own `MmioRead` or `MmioWrite`. The retry ledger keeps
the answer or acknowledgement of each completed transfer under that transfer: its
address for a vector operand, its operand for a string element. The retry then
stops only at the transfer that asked last, and the next completion answers that
transfer. These records live across the same boundary stops as a recorded
answer: `BudgetExhausted` and a host stop (`Interrupted`) taken before the retry
runs. Long64 and Compat32 take a requested host stop there. The alias and the
Machine models finish the open retry first, and repeat an unanswered request
instead of stopping. A fault of the instruction reports these records as its
acknowledged transfers (`FaultProgress::acknowledged_transfers`).

Some stores are split into a RAM part and a device part: the twenty-bit models'
transfers of a word over a RAM byte and a device byte, an operand that
the A20 gate splits into bus pieces on the I286/I386 Machine path, and a locked
operand partly in RAM and partly in a device (below). Such a store
writes its device part first and commits the RAM part once the device write is
acknowledged, so a stop at the device leaves the instruction's RAM uncommitted
and the retry of a read-modify-write computes from the operand it read first.
That order is an engine choice for restartability (ASSUMED; no pinned manual
gives the order of the two bus cycles), not a bus order; the device sees the same
transfers either way, and only an `MmioWrite` handler that reads guest RAM can see
the RAM part still unwritten.

**Twenty-bit device transfers follow the data bus (xj-50w.20).** The catalogue's
`data_bus_bits` sets the width of a twenty-bit model's device transfer, and the
lift copies it into each transfer it lowers. The 8088 "always accesses memory in
bytes. Word operands are accessed in two bus cycles regardless of their
alignment"; the 8086 accesses a word at an even address "in one bus cycle" and a
word at an odd address "one byte at a time in two consecutive bus cycles" (8086
Family User's Manual §2.3 p. 2-15). NEC V20/V30 UM §4 pp. 4-2–4-3 says the same
of the uPD70108's 8-bit bus and the uPD70116's 16-bit bus. Hence:
- I8088 and NecV20 (8 bits): every device byte is one `MmioRead`/`MmioWrite` of
  size 1, as before.
- I8086 and NecV30 (16 bits): a word of an operand whose low byte is at an even
  address and whose two bytes lie in one Trap region is one transfer of size 2 at
  that address. An operand's words start at its even byte indices: a far pointer,
  a stack frame and an interrupt vector are sequences of words.
- A word at an odd address is two byte transfers on every model: the low byte at
  the address, then the high byte at the address plus one (UM §4.2 p. 4-15; NEC
  UM §4 p. 4-3). A word that wraps its segment (offset FFFFh) or the bus
  (FFFFFh) starts at an odd address, so it keeps its two byte transfers, the
  second at the wrapped address (DERIVED).
- A word over two Trap regions, or over a Trap byte and a RAM byte, keeps its
  byte transfers on every model, the device part first as above. On the 8086 and
  V30 that is an engine limit (ASSUMED): one exit names one region, while the
  processor makes one bus cycle of an even word (UM §4.2 p. 4-15, Table 4-5).
  A locked word over a Trap byte and a RAM byte is performed the same way, with
  its RAM byte read once (below); over two Trap regions it keeps its typed
  refusal.
- Frames are written in the engine's order: an interrupt frame is stored from
  SP-6 upward (IP, CS, FLAGS), a far-call frame from SP-4 upward (IP, CS), PUSH R
  from SP-2 downward; IRET, RETF and POP R read upward. On a 16-bit bus each even
  word of them is one transfer. The store order of the interrupt and far-call
  frames is a known divergence from the sources, on every twenty-bit model: the
  processor pushes FLAGS, then CS, then IP (UM §2.6 p. 2-26; the bus sequence of
  AP-67, UM p. A-28: IVT IP read, IVT CS read, push FLAGS, push CS, push IP), and
  an intersegment CALL pushes CS, then IP (UM §2.7 p. 2-44). A device that sees
  the stores sees the engine's order. Owned by xj-50w.25; the IVT reads (IP,
  then CS) and the IRET/RETF pops already follow the sources.
- Strings: each word element at an even address inside one Trap region is one
  transfer (MOVSW, STOSW, LODSW, CMPSW, SCASW, and the V30's INM/OUTM memory
  side), once per element of a repeated string.
- The CPU's dummy read of a memory ESC or FPO2 operand is one word transfer
  under the same rule. The coprocessor's own operand transfers stay byte-wide on
  every model (ASSUMED: no pinned source gives the 8087's cycle width on an 8086
  bus). For a load, the coprocessor then reads the operand's first word again;
  on the processor the 8087 captures that word from the CPU's dummy read and
  reads only the rest "in consecutive bus cycles" (iAPX 86/88, 186/188 PRM §6.3
  p. 6-8). That second read is a known divergence, owned by xj-50w.26.
A word transfer is recorded in the retry ledger under its word: the retry after
its completion takes the recorded answer for both bytes, and stops again only
at the operand's next transfer, also across a budget or host stop before the
retry. Only a transfer inside one Trap region is a word transfer, so it never
reaches the adjacent-region rule above: RAM transfers stay byte-wide and
containment-only, and the twenty-bit walk still meets no containment miss over
adjacent regions. The 8088 and V20 identities are unchanged; the I8086 and NecV30
identities hash the new bus width (§17.10).
On the I8086 and NecV30 an entry of `FaultProgress::acknowledged_transfers` may
stand for a word transfer: it names the even address of the word's low byte, and
its value is the word. The entry carries no size; the value's width is the only
clue (the I8088 and V20 report the same word as two byte entries).

**Locked operands on Trap regions — Real16 and Machine.** On the Real16 alias and
every Machine model, XCHG with memory, CMPXCHG, XADD and the LOCK-prefixed forms
perform the same one read, then one write, of an operand inside one Trap region.
XCHG with memory is locked automatically (SDM Vol 3A §11.1.2.1 p. 11-3; 80286 PRM
Appendix B XCHG p. B-114; 8086 Family User's Manual §2.5 p. 2-18); a locked
operation holds the bus "for as many bus cycles as necessary to update the entire
operand" (SDM Vol 3A §11.1.2.2 p. 11-4); "the processor never produces a locked
read without also producing a locked write", and a CMPXCHG writes the value read
back when its comparison fails and the source when it succeeds (SDM Vol 2A §3.3
CMPXCHG, p. 3-194). A locked exchange whose old value the instruction
discards, the 80286's locked MOV store, is one write and no read, as the
unlocked store is. LOCK# is not modelled: the vCPU is stopped across both exits
— no interrupt is accepted between them — so ordering against devices and DMA
stays with the embedder. The twenty-bit models move device data at their bus
width (above), so a word operand there is one word read, then one word write, on
the 8086 and V30 at an even address, and two byte reads, then two byte writes,
otherwise; an XCHG over a RAM byte and a device byte transfers the device byte. A locked
operand on RAM keeps the atomic path. An A20-split operand whose pieces all lie
in devices is one transfer per bus piece. The alias's CMPXCHG8B keeps embedder
emulation: `Exit::UnknownInstruction`, as below.

**Locked operands partly in RAM (xj-ir6.34).** The processor performs a locked
operand whose bytes lie partly in RAM and partly in a device: the lock "is not
affected by the alignment of the memory field" and is held "for as many bus
cycles as necessary to update the entire operand" (SDM Vol 3A §11.1.2.2 p. 11-4;
80386 PRM §11.2.1 p. 11-5). On the Real16 alias and every Machine model such an
operand — a word over the last RAM byte and the first device byte, a doubleword
with one to three bytes on either side, or an A20-split operand with a RAM piece
and a device piece — is split into pieces, each a run of consecutive bus
addresses in one region. A piece in ROM is handled as a RAM piece whose write
the memory drops (the ROM bullet below):
- Order. The pieces are read in operand byte order, the lowest address first:
  the RAM pieces directly, the device piece as one `MmioRead` of its bytes (one
  transfer, as an A20 bus piece is; a byte transfer on the twenty-bit models,
  whose split word is never inside one Trap region). Then the device piece is
  written as one `MmioWrite`, and the RAM pieces are committed once it is
  acknowledged. On the twenty-bit models the read order is the bus order of a
  word at an odd address, the low byte first (8086 Family UM §4.2 p. 4-15;
  DERIVED). For the 80286, the 80386 and the alias no pinned source orders the
  bus cycles of a misaligned operand (the 80386 PRM §2.2 p. 2-4 says only that
  the processor converts it into "the appropriate sequences of requests"), so the
  order is ASSUMED: the one the engine's plain split read uses. The device-first
  write is the engine's order for a split store (above, ASSUMED). Each device
  exit reports the operand's width, RAM bytes included, as `total_size`, and
  the piece's byte offset in the operand over its size as `part`.
- Retry. The value of each RAM piece is read once per attempt and recorded in
  the retry ledger as a bound key: it lives across the instruction's own device
  stops and a budget or host stop before the retry, and every retry computes from
  it, so a store to that RAM between the exits does not change the result and is
  overwritten by the commit. The device answer and acknowledgement are recorded
  as every other device transfer of the path is, so the retry after the write's
  acknowledgement reads no device. A fault, or any other stop, drops the RAM
  values, and the re-execution reads the RAM again; they are never reported as
  acknowledged transfers. The device answer follows the path's rule for its
  transfers: on the alias it is a bound answer, dropped with the RAM values, so
  the re-execution reads the device again too; on the twenty-bit and 286/386
  paths it is an ordinary transfer record, which an `UnmappedMemory` stop keeps
  and reports as an acknowledged transfer, so the re-execution does not read
  the device again, and which any other stop drops.
- ROM. A piece in ROM is performed as a locked operand inside a ROM region is:
  it is read with the other memory pieces, and its write is dropped, while the
  device piece is read and written as above. Nothing faults.
- Faults. Every piece is checked before the first transfer. A byte in no region
  is the containment miss of a locked operand over two regions (`UnmappedMemory`
  for a write, at that byte), and a RAM piece the host refuses to write faults
  there too: no device transfer is issued and nothing is committed, a RAM fault
  coming before any device read (ASSUMED, as the twenty-bit store and the
  A20-split locked operand validate first). Bytes at consecutive bus addresses
  in two adjacent memory regions, RAM or ROM, keep the containment miss they
  take without a device byte (above, Data accesses across adjacent regions),
  reported at the second region's first byte. Before xj-ir6.34 an operand over
  a device and ROM, a device and no region, or a device and two adjacent memory
  regions ended in `Exit::UnknownInstruction` for the embedder to emulate; over
  a device and ROM it is now performed (above), and over a device and no region
  or two adjacent memory regions it now ends in this `UnmappedMemory`.
- Known inconsistency. A locked operand over adjacent ROM and RAM regions with
  no device byte keeps the containment miss (`UnmappedMemory` for a write, at
  the operand), while a locked operand inside a ROM region, and a ROM piece
  beside a device piece, are performed with the ROM write dropped. Resolving it
  is separate work.
- CMPXCHG writes its destination either way: the source when the comparison
  succeeds, the value read when it fails (SDM Vol 2A §3.3 CMPXCHG, p. 3-194), so a
  failed comparison writes the device piece and the RAM piece back.
- Atomicity. The vCPU is stopped across both exits and takes no interrupt
  between them, so the result is the single-processor one. LIMITATION: the RAM
  pieces are a plain read and a later write, not one host atomic, so a store by
  another vCPU between them is lost: the crossing-operand policy of §8.2.3. The
  Machine models run one vCPU; the alias may run several.
- An operand over two Trap regions, with or without RAM bytes, keeps
  `Exit::UnknownInstruction` with the instruction's bytes: one exit names one
  region. The alias's CMPXCHG8B keeps it too.

**Atomic operations on Trap regions — Long64 and Compat32:** memory CMPXCHG, XADD,
XCHG, CMPXCHG8B/CMPXCHG16B and supported LOCK-prefixed ALU/bit modifications return
`Exit::UnknownInstruction` with the instruction's exact bytes and length, leaving
RIP on that instruction and its guest state unchanged. No device transfer is
requested; their compiled code has no device path. The embedder emulates the
instruction and advances RIP; `complete_mmio_read`/`complete_mmio_write` do not
acknowledge this exit. Memory CMPXCHG and XADD use this policy with or without an
explicit LOCK prefix, matching their atomic IR lowering.

> **Note on MMIO vs JIT:** in generated code, ordinary RAM is inlined, but access to a Trap region must leave the block. This complicates codegen (the JIT must generate a check "is this address a Trap" or know it statically). At the start (interpreter) this is trivial. With the JIT, consider: either a runtime check on accesses that may hit a Trap, or keeping MMIO out of the hot path and accepting that accesses to unknown addresses go the slow route. Resolve this in M4 together with the inline-access safety check (§8.2.3) — one range/permission check can serve both.

### 5.3 Inspecting and modifying state between calls

After `run()` (and before the next one) the user reads/writes the guest state — this is how syscall HLE reads arguments and writes the result:

```rust
// example of handling the "write" syscall (Linux x86-64: nr in RAX, args in RDI/RSI/RDX)
match cpu.run(Some(100_000)) {
    Exit::Syscall => {
        let nr  = cpu.reg(Reg::Rax);
        let fd  = cpu.reg(Reg::Rdi);
        let buf = cpu.reg(Reg::Rsi);
        let len = cpu.reg(Reg::Rdx);
        let ret = host_handle_syscall(nr, fd, buf, len, &vm);
        cpu.set_reg(Reg::Rax, ret);      // result back to the guest
        // the loop returns to run()
    }
    Exit::UnknownInstruction { addr, bytes, len } => {
        panic!("no lift for instruction @ {addr:#x}: {:02x?}", &bytes[..len as usize]);
    }
    other => { /* ... */ }
}
```

---

## 6. IR (intermediate representation)

The IR is **custom** — it is the layer "smart about x86 semantics" that you either interpret or translate to Cranelift. It is what lets you have an interpreter *and* a JIT behind the same frontend.

### 6.1 Value model

The IR is a list of operations on **temporary values** (temporaries) — three-address code, not a tree. Each operation produces a value, subsequent ones consume it.

```rust
pub type Temp = u32;

pub enum Val {
    Temp(Temp),   // result of an earlier operation
    Imm(u64),     // constant
}
```

### 6.2 IR operations (sketch — you extend it as you lift)

```rust
/// Which flags an operation updates. x86 is NOT "all or nothing":
/// inc/dec update everything EXCEPT CF; and/or/xor force CF=OF=0 and leave AF
/// undefined; shifts touch flags ONLY when count != 0 (runtime-conditional!).
/// A plain bool cannot express any of that.
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct FlagMask(pub u8);   // bit per flag: CF|PF|AF|ZF|SF|OF
impl FlagMask {
    pub const NONE: FlagMask;        // e.g. mov, lea
    pub const ALL: FlagMask;         // add, sub, cmp
    pub const ALL_BUT_CF: FlagMask;  // inc, dec
    // logic ops: ALL, but the backend forces CF=OF=0 per the op's rules
}

pub enum IrOp {
    // --- instruction boundary marker ---
    // Emitted by the lift at the START of each guest instruction. Carries the
    // guest address so a backend can set cpu.rip to the FAULTING instruction on a
    // memory trap or exception (§8). Without it, RIP-on-trap has no address —
    // guest_len gives only the block END (right for syscall, wrong for a mid-block
    // fault). Interp: hold it in a `cur_addr` local. JIT: bake it as a const for
    // the following trapping accesses. Also delimits instructions for SMC.
    InsnStart { guest_addr: u64 },

    // --- data movement ---
    ReadReg  { dst: Temp, reg: Reg },   // Reg::Rip is FORBIDDEN here — see the note below
    WriteReg { reg: Reg, src: Val },

    // --- arithmetic / logic (size: 1,2,4,8 bytes) ---
    Add { dst: Temp, a: Val, b: Val, size: u8, set_flags: FlagMask },
    Sub { dst: Temp, a: Val, b: Val, size: u8, set_flags: FlagMask },
    // Adc/Sbb CONSUME CF as an input (a + b + CF): flags can't be add-then-add,
    // the carry chain and OF/CF must be computed over the full three-operand sum.
    Adc { dst: Temp, a: Val, b: Val, size: u8, set_flags: FlagMask },
    Sbb { dst: Temp, a: Val, b: Val, size: u8, set_flags: FlagMask },
    And { dst: Temp, a: Val, b: Val, size: u8, set_flags: FlagMask },
    Or  { dst: Temp, a: Val, b: Val, size: u8, set_flags: FlagMask },
    Xor { dst: Temp, a: Val, b: Val, size: u8, set_flags: FlagMask },
    // Shifts: flag update happens ONLY if the (masked) count != 0 — the backend
    // implements the runtime condition (interp: an if; JIT: a conditional).
    Shl { dst: Temp, a: Val, b: Val, size: u8, set_flags: FlagMask },
    Shr { dst: Temp, a: Val, b: Val, size: u8, set_flags: FlagMask },
    // ... Mul, Div, Neg, Not, Sar, Rol, Ror, etc.

    // --- flags as DATA (not only as branch conditions) ---
    // setcc, cmovcc, adc/sbb lowering, rcl/rcr all need to READ flags into a value.
    // GetCond materializes a condition as 0/1. (CF alone = GetCond(Below).)
    GetCond { dst: Temp, cond: Cond },

    // --- memory ---
    Load  { dst: Temp, addr: Val, size: u8 },
    Store { addr: Val, src: Val, size: u8, order: MemOrder },
    Fence { kind: FenceKind },

    // --- control: EACH of these operations ENDS the block ---
    // Note: for jmp/jcc to a constant target, addresses are KNOWN statically at
    // lift time (Imm). For indirect jmp (jmp rax, ret) the target is dynamic (Temp).
    Jump   { target: Val },                                 // jmp (direct: Imm, indirect: Temp)
    Branch { cond: Cond, taken: u64, fallthrough: u64 },    // jcc — BOTH branches are known addresses
    Call   { target: Val, return_addr: u64 },               // return_addr = address after call (to push on stack)
    Ret,                                                    // dynamic target — from the stack
    Syscall,                                                // trap-out to the user
    Hlt,
}

pub enum MemOrder { None }  // ordinary accesses use the Vm consistency tier
pub enum FenceKind { Full, Load, Store }  // IrOp::Fence carries explicit guest fences

// NOTE — ReadReg(Rip) is forbidden in the IR: cpu.rip is updated only at block
// end, so a mid-block read would see a stale value. Everything that "reads RIP"
// (RIP-relative addressing, call's return address) is known statically at lift
// time and lowered to an Imm. Assert this in the lift.

// jcc conditions. They correspond to flag combinations, NOT to "sign" abstractly —
// x86 distinguishes signed comparisons (l/g: SF,OF,ZF) and unsigned ones (b/a: CF,ZF).
pub enum Cond {
    Eq, Ne,               // ZF        (je/jne)
    Below, BelowEq,       // CF, CF|ZF (jb/jbe — unsigned)
    Above, AboveEq,       // !CF&!ZF   (ja/jae — unsigned)
    Less, LessEq,         // SF!=OF    (jl/jle — signed)
    Greater, GreaterEq,   // SF==OF    (jg/jge — signed)
    Sign, NoSign,         // SF        (js/jns)
    Overflow, NoOverflow, // OF        (jo/jno)
    Parity, NoParity,     // PF        (jp/jnp)
}
```

### 6.3 IR block

```rust
pub struct IrBlock {
    pub guest_start: u64,   // guest address (key in the cache)
    pub ops: Vec<IrOp>,     // operations, the last always ends the block (Jump/Branch/Ret/...)
    pub temp_count: u32,    // how many temporaries were allocated (the backend reserves that many slots)
    pub guest_len: u32,     // how many guest bytes the block spanned (for SMC invalidation)
    pub icount: u32,        // how many x86 instructions (for budget/statistics)
}
```

---

## 7. Lift: x86 → IR

The lift is a `match` on the mnemonic from `iced-x86`, producing IR operations. Use `InstructionInfo` from iced (registers read/written, flags touched, control-flow classification) — it shortens the lift and tells you directly when to close the block.

### 7.1 Operand lowering — READ THIS BEFORE YOU START THE LIFT

> **This is the biggest source of surprises during implementation.** The IR examples in section 6 show operands as ready `Val`s — that's a simplification. In reality an x86 operand has three forms, and one of them (memory) itself requires several IR operations. The lift MUST have an "operand lowering" layer *beneath* the mnemonic lift, otherwise `Add`/`Mov`/the rest won't come together.

An x86 operand is one of three things:

1. **Register** — `rax`. Lowering: `ReadReg` → `Temp`.
2. **Immediate** — `0x42`. Lowering: `Val::Imm`.
3. **Memory** — `[rbx + rcx*4 + 0x10]`. Lowering: compute the **effective address** (this is arithmetic: `base + index*scale + disp`), then `Load`. This is several IR operations per one operand.

That's why the lift of each instruction is really **two levels**: (a) reduce the operands to `Val` (lowering), (b) generate the operation on those `Val`s. Design two helper functions and use them everywhere:

```rust
/// Reduce a SOURCE operand to a Val (reads a register / immediate / loads from memory).
/// For a memory operand it emits the effective-address computation + Load.
fn lower_read(insn: &Instruction, op_idx: u32, ops: &mut Vec<IrOp>, tg: &mut TempGen) -> Val;

/// Reduce a DESTINATION operand to a "write location" (register or memory address).
/// Returns a handle you later use to write the result (WriteReg or Store).
fn lower_write_target(insn: &Instruction, op_idx: u32, ops: &mut Vec<IrOp>, tg: &mut TempGen) -> WriteTarget;

enum WriteTarget {
    Reg(Reg),
    Mem { addr: Val, size: u8 },   // address already computed as a Val
}
```

**The effective address** is a separate helper — called by `lower_read`/`lower_write_target` when the operand is memory:

```rust
/// Emits IR operations computing base + index*scale + disp, returns a Temp with the address.
/// iced gives the components: memory_base(), memory_index(), memory_index_scale(),
/// memory_displacement(). Watch out for RIP-relative (see pitfall no. 2) and for the
/// FS/GS segment base (TLS) — if the instruction has a segment prefix, add fs_base/gs_base.
///
/// SEAM (§17.5): this is the ONLY place a memory address is computed. No lift computes
/// an address itself. Today in Long64 we skip segment bases except FS/GS; if a 32-bit
/// mode were added, segmentation would be added HERE (one function), not in every lift.
fn effective_address(insn: &Instruction, ops: &mut Vec<IrOp>, tg: &mut TempGen) -> Val;
```

**Read-modify-write** — instructions such as `add [mem], rax` read the memory operand, modify it, and write it back. The pattern: `effective_address` (once!) → `Load` → operation → `Store` to the same address. **Compute the effective address only once** and use it for both Load and Store — don't compute it twice (it would be a bug if the address depended on something that changes, and wasteful).

Example — the full lift of `add rax, [rbx+8]` (destination operand register, source memory):

```rust
fn lift_add(insn: &Instruction, ops: &mut Vec<IrOp>, tg: &mut TempGen) {
    let dst = lower_write_target(insn, 0, ops, tg);   // op0 = rax → WriteTarget::Reg(Rax)
    let a   = lower_read(insn, 0, ops, tg);           // read the current value of rax
    let b   = lower_read(insn, 1, ops, tg);           // op1 = [rbx+8] → effective_address + Load
    let res = tg.fresh();
    let size = operand_size(insn);                    // 8 for rax
    ops.push(IrOp::Add { dst: res, a, b, size, set_flags: true });
    // write the result to the destination:
    match dst {
        WriteTarget::Reg(r)          => ops.push(IrOp::WriteReg { reg: r, src: Val::Temp(res) }),
        WriteTarget::Mem { addr, size } => ops.push(IrOp::Store { addr, src: Val::Temp(res), size, order: MemOrder::None }),
    }
}
```

> You can see why without this layer the lift wouldn't work: `add` is not "add two Vals", but "reduce both operands (perhaps loading from memory via a computed address), add, write the result to the destination (perhaps writing to memory)". This structure repeats for ALL two-operand instructions — write it once, use it everywhere.

### 7.2 Allocating temporaries

`Temp` is simply an increasing counter within the block. A simple generator:

```rust
pub struct TempGen(u32);
impl TempGen {
    fn new() -> Self { TempGen(0) }
    fn fresh(&mut self) -> Temp { let t = self.0; self.0 += 1; t }
    fn count(&self) -> u32 { self.0 }   // how many temps were allocated (the backend reserves that many slots)
}
```

The interpreter keeps `temps: Vec<u64>` of size `count()`. The JIT maps `Temp → cranelift Value` (see 8.2). Temps are local to the block — you reset `TempGen` for each block.

### 7.3 Skeleton of the block lift loop

```rust
pub fn lift_block(vm: &Vm, start: u64) -> Result<IrBlock, LiftError> {
    let mut decoder = /* iced Decoder set to guest bytes from `start`. SEAM (§17.3):
                         bitness from CpuMode (mode.bits()), NOT the literal 64 — today always Long64. */;
    let mut ops = Vec::new();
    let mut tg = TempGen::new();
    let mut icount = 0u32;
    let mut guest_len = 0u32;
    loop {
        let insn = decoder.decode();
        if insn.is_invalid() { return Err(LiftError::DecodeFault { addr: /* current PC */ }); }
        icount += 1;
        guest_len += insn.len() as u32;
        // Mark the instruction boundary so a mem-trap / exception can set RIP to
        // THIS instruction (§8). Every per-mnemonic lift below emits its trapping
        // ops after this marker.
        ops.push(IrOp::InsnStart { guest_addr: insn.ip() });
        // tg is NOT reset between instructions in the block — temps grow through the whole block.
        match insn.mnemonic() {
            Mnemonic::Mov => lift_mov(&insn, &mut ops, &mut tg),
            Mnemonic::Add => lift_add(&insn, &mut ops, &mut tg),
            Mnemonic::Jmp => { lift_jmp(&insn, &mut ops, &mut tg); break; }   // end of block
            Mnemonic::Je | Mnemonic::Jne /* ... */ => { lift_jcc(&insn, &mut ops); break; }
            Mnemonic::Call => { lift_call(&insn, &mut ops, &mut tg); break; }
            Mnemonic::Ret  => { ops.push(IrOp::Ret); break; }
            Mnemonic::Syscall => { ops.push(IrOp::Syscall); break; }
            // NOTE: the catch-all arm must consult iced's flow-control info BEFORE
            // returning Unsupported — otherwise a rarer jump variant not listed above
            // falls through to Unsupported and a trailing `if is_flow_control` check
            // after the match would be dead code (an earlier draft had that bug).
            _ if is_flow_control(&insn) => { lift_flow(&insn, &mut ops, &mut tg)?; break; }
            _ => return Err(LiftError::Unsupported {
                addr: insn.ip(),
                bytes: copy_insn_bytes(&insn),   // 15-byte buffer + len
                len: insn.len() as u8,
            }),
        }
    }
    Ok(IrBlock { guest_start: start, ops, temp_count: tg.count(), guest_len, icount })
}

// Lift errors — mapped in the dispatcher to an Exit, NOT to a panic.
pub enum LiftError {
    /// Instruction recognized by iced, but the lift does not handle it yet.
    Unsupported { addr: u64, bytes: [u8; 15], len: u8 },
    /// Can't even be decoded (e.g. bytes outside mapped memory, garbage).
    DecodeFault { addr: u64 },
}
```

> **Block-closing rule:** a block ends at the first instruction changing control flow (jmp/jcc/call/ret/syscall/hlt). `iced-x86` provides info about control flow — use it instead of a manual list of opcodes.

> **Flag semantics are the essence of the lift.** `Add` with `set_flags: FlagMask::ALL` must generate the computation of the six flags per the ADD rules. This is the place where you actually encode "what each x86 instruction means". iced tells you *which* flags are touched; *how* to compute them — that's your lift. Remember the non-uniform cases baked into `FlagMask` (§6.2): `inc`/`dec` keep CF, logic ops force CF=OF=0, shifts update flags only when the masked count ≠ 0.

> **Pitfall no. 1 when porting x86-64 — a write to a 32-bit register ZEROES the upper 32 bits.** `mov eax, ...` (or any operation on `eax`) sets the whole `rax`, zeroing bits 32–63. But an operation on `ax` (16-bit) or `al` (8-bit) preserves the higher bits. This is NOT symmetric and is a source of subtle bugs. Your `WriteReg`/interpreter must respect this depending on the size of the destination operand. Encode this rule once, centrally, in the write to a GPR — don't scatter it across the lifts of individual instructions.

> **Pitfall no. 2 — RIP-relative addressing.** x86-64 has a RIP-relative addressing mode (`lea rax, [rip+0x1234]`). iced gives it to you computed (`insn.ip_rel_memory_address()` / `memory_displacement`), but you must use the address of the *next* instruction as the base, not the current one. iced computes this — just use its value, don't compute it by hand.

> **Pitfall no. 3 — instruction atomicity vs the RIP-retry convention (silent state corruption).** The trap-out convention (§8) says: on a memory fault, RIP points at the *faulting instruction* so the user can map/handle and retry. Retry means **re-executing the instruction from scratch** — so **no effect of that instruction may commit before its last potential trap**. The lift must order the emitted IR accordingly:
>
> - `push rax` must NOT be lowered as "WriteReg rsp-8, then Store" — if the Store faults, RSP has already moved and the retry corrupts the stack. Lower as: compute the new-RSP temp → `Store` (may trap) → only then `WriteReg rsp`.
> - RMW (`add [mem], rax`): the `Store` can trap — flags and any register write must be emitted *after* it, or recomputed on retry (they are, since the whole instruction re-executes — as long as nothing was committed).
> - General rule: within one guest instruction, emit **all ops that can trap (loads/stores) before all ops that commit state (WriteReg, flag updates)** — or prove the commit is idempotent under re-execution.
>
> Partial-block progress is fine (each *completed* instruction's effects are committed as the block runs, and a retried block simply starts at the faulting instruction — any address can start a block). The invariant is strictly *intra*-instruction. Violations are invisible until a fault-retry actually happens, then corrupt state at a distance — encode the discipline in the lowering helpers, and fuzz with faulting addresses (testing.md).

---

## 8. Backends

Both backends consume an `IrBlock`. A shared interface.

> **Important — the return type.** The backend does NOT return `Exit` directly. It must distinguish two cases: (a) the block ended normally and execution should flow on (it gives a new RIP), (b) the block traps out to the user (`Exit`). That's why we introduce `StepResult`. This resolves the contradiction where the dispatcher loop must know whether to continue or to return.

> **Important — two different responsibilities, not one `execute`.** The backend does not have a single "execute" method, because the interpreter and the JIT execute *different things*: the interpreter walks the `IrBlock`, the JIT jumps into a compiled `code_ptr`. We split this into:
> - **materialization** (backend-dependent): from an `IrBlock` make a `CachedBlock` — the interpreter wraps it in an `Arc`, the JIT compiles it to host code;
> - **execution** (uniform): the dispatcher `match`es on the `CachedBlock` variant and either interprets or jumps into the code.
>
> Thanks to this there is no false `execute(&IrBlock)`, which makes no sense for the JIT (after materialization the JIT no longer touches the `IrBlock`). The return type in both cases is `StepResult`.

```rust
pub enum StepResult {
    /// The block executed to the end (jump/ret/branch resolved internally).
    /// The new RIP is already stored in CpuState; execution should flow on.
    Continue,
    /// The block traps out to the user. Execution stopped; RIP per the convention below.
    Exit(Exit),
}

/// The backend materializes IR into an executable form. This is the only backend-dependent
/// operation. Injected into the Vm as a `Box<dyn Backend>` (§4.1): the interpreter impl lives
/// in the core, the JIT impl in x86jit-cranelift.
pub trait Backend {
    fn materialize(&mut self, ir: &IrBlock) -> CachedBlock;
}

/// Execution is uniform — the dispatcher calls it on a CachedBlock regardless of the backend.
/// NOTE: `mem` is `&Memory`, NOT `&mut Memory` — guest RAM uses interior mutability
/// (see the pitfall below). Stores go through `&self` methods.
fn execute(
    block: &CachedBlock,
    cpu: &mut CpuState,
    mem: &Memory,
    mode: CpuMode,
    consistency: MemConsistency,
) -> StepResult {
    match block {
        CachedBlock::Interpreted(ir) => {
            let mut info = RetireInfo { consistency, ..Default::default() };
            interpret_block(ir, cpu, mem, &mut Vec::new(), &mut info)
        },
        CachedBlock::Compiled { entry, .. } => {
            // compiled block ABI — see 8.2.
            unsafe { run_compiled(*entry, cpu, mem, mode, consistency) }
        }
    }
}
```

> **Pitfall — guest RAM cannot be `&mut Memory` (hits in M1, not M7).** A store needs to *write* guest RAM, so the obvious signature is `&mut Memory`. But: (a) the dispatcher reads the cache from the same `Vm` while a block executes, and (b) in M7 **multiple vcpus write the same RAM concurrently** — that's what real hardware does, and `&mut` (exclusive) cannot model it at all. Start with `&mut Memory` and you rewrite every backend signature at M7. Decide it now: the guest RAM buffer uses **interior mutability** — `UnsafeCell<[u8]>` (or a raw base pointer) behind `&Memory`, with `read(&self, …)`/`write(&self, …)` and a hand-written `unsafe impl Sync`. Document the contract: concurrent guest stores race exactly like real hardware, and the TSO barriers (§8.2.3, §11) — not Rust's `&mut` — provide the ordering the guest expects. This is the one place the core is deliberately `unsafe`; wrap it tightly and keep `CpuState` (`&mut`, per-vcpu) separate from `Memory` (`&`, shared).

> **RIP semantics on trap-out (to be resolved and held consistently):** when a block traps out on `syscall`, RIP should point to the instruction *after* `syscall` (because after handling you resume past it). When it traps out on `UnmappedMemory`/`Mmio`, RIP points to the instruction *causing* the access (because the user may want to retry it after mapping/handling). Write down this convention and stick to it in both backends — otherwise the JIT and the interpreter will diverge on trap-outs.

### 8.1 Interpreter (build FIRST)

Walks the list of `IrOp` in a loop and executes each operation with native Rust code. Slow, but simple, fully debuggable, and on an x86 host comparable with native execution (oracle).

A pure store wholly within one host page needs no host-access preflight.
Crossing stores retain same-value RMW probes once per touched host page before
committing bytes (decision-13). Only a probe's read fault queries host VM write
permission; a writable verdict permits the store without that read. In this cold
fallback, all-or-nothing relies on page protection as for a single-page store;
VM metadata cannot rule out guard, backing-file or thread-local protection faults.
A negative or failed query preserves the original access fault. The embedder must
keep host protections stable through commit. Compiled crossing preflight remains
opt-in through precise split stores and uses the real host page size. Ordering
stays in the existing access barriers. XSAVE and XSAVEOPT retain their architectural
XSTATE_BV read (SDM Vol 1 §§13.7, 13.9); XSAVEC is a pure store (§13.10).

Memory-access contract (used by both backends in the interpreter version and by the trap-out):

```rust
#[non_exhaustive]
pub enum MemTrap { Unmapped, HostFault, Mmio }

impl MemTrap {
    // HostFault carries no payload, so MemTrap and Result<(), MemTrap> stay one
    // byte on the hot path. The faulting byte's address lives in the accessing
    // thread's return slot: call this on that thread, before its next failing
    // routed access. Other variants return `transfer` unchanged.
    pub fn fault_addr(self, transfer: u64) -> u64;
}

impl Memory {
    // Both take &self — guest RAM is interior-mutable (see §8 pitfall). `write`
    // does NOT need &mut, which is what lets one Memory be shared across vcpus.
    pub fn read(&self,  addr: u64, size: u8) -> Result<u64, MemTrap>;
    pub fn write(&self, addr: u64, val: u64, size: u8) -> Result<(), MemTrap>;

    // Opt-in host-fault recovery for Rust guest accesses (decision-13). Default off.
    pub fn set_host_fault_recovery(&mut self, enabled: bool);
    pub fn host_fault_recovery(&self) -> bool;

    // An owned snapshot of up to `max_len` bytes from `addr`, never a borrowed
    // view of live guest memory: a concurrent guest store into a slice iced is
    // reading would be a data race. A readable prefix is returned when a later
    // byte is unavailable. The lift (§7.3) does not call this: it decodes from
    // its own private buffer, filled on demand (see below).
    pub fn code_slice(&self, addr: u64, max_len: usize) -> Result<Vec<u8>, MemTrap>;
}
```

**Host faults inside Rust accesses.** By default the interpreter, the Rust helpers
that compiled code calls (string, x87, FXSAVE/FXRSTOR, masked and narrowing
stores, paired CAS) and the lifter's fetch access guest RAM directly. They check
the region table, not host protections, so **registered RAM must stay
host-accessible** to them. Compiled code is different: its host faults recover in
both modes through the signal-context landing stub (decision-13).

An embedder that owns host protections calls `set_host_fault_recovery(true)` before
execution. Examples: PROT_NONE stack guards, write-protected pages for a GC barrier,
`mprotect` from other threads. The setting routes every Rust guest access through
small fault-fixup routines (`usercopy`), one per operation and host (x86-64,
AArch64), listed in an immutable fixup table:
- probes;
- word-wise copies with byte retry;
- single-copy-atomic 1/2/4/8-byte loads and stores;
- CAS and paired CAS.

The embedder's handler calls the same `jit_abi::fault::classify`. A PC inside a
routine is OURS for an access fault even with no active vcpu run. The context is
rewritten to the routine's fixup, which returns the faulting byte's address to
its Rust caller as an ordinary return value. Nothing unwinds, every `Drop` runs,
and the access becomes the resumable `Exit::UnmappedMemory` the interpreter
gives for an unmapped page. The interpreter picks the access path once per run
(const-generic), so the default path carries no per-access branch.

Routed writes probe each touched host page before committing (one atomic
same-value byte write per page). Under protections that stay stable, a faulting
instruction therefore commits nothing. Known limits, recorded in decision-13:
- protections revoked concurrently by another thread need the embedder's
  serialisation;
- routed REP MOVS/STOS keeps completed iterations and runs per element;
- compiled stores the host splits can commit their completed parts (xj-obx.8).

The opt-in exists because routing costs. Measured on AArch64 against no routing:
interpreter memcpy +4.6%, routed FXSAVE/x87/REP helper loops +20–140%. The
default path adds no per-access work. Against the version before recovery
existed it measures between −20% and +3% by workload median; the differences
are code layout. Decision-13 holds the tables.

**Fetch.** The lifter always decodes a private copy. The first chunk is at most 128
bytes, bounded by the host page and the guest region. Later chunks are copied
only when the decoder asks for more bytes. A single step needs at most 15 bytes.
A fetch fault is reported at the first unreadable byte, and the valid prefix
still lifts. No lock is held across the copy.

```rust
// `cur_addr` tracks the current guest instruction so a trap can set RIP to it.
IrOp::InsnStart { guest_addr } => { cur_addr = *guest_addr; }
IrOp::Add { dst, a, b, size, set_flags } => {
    let va = read_val(a, &temps);
    let vb = read_val(b, &temps);
    let res = va.wrapping_add(vb) & mask(size);
    temps[*dst] = res;
    // FlagMask, not bool: update only the flags in the mask (§6.2).
    compute_add_flags(va, vb, res, *size, *set_flags, &mut cpu.flags);
}
IrOp::Store { addr, src, size, .. } => {
    // On a trap, RIP = the faulting instruction (cur_addr, from InsnStart) — NOT
    // block.guest_start + guest_len (that's the block end). Retry re-executes it.
    // A HostFault's address is the faulting byte: read it on this thread, at once.
    let a = read_val(addr, &temps);
    match mem.write(a, read_val(src, &temps), *size) {
        Ok(())            => {}
        Err(MemTrap::Mmio) => { cpu.rip = cur_addr; return StepResult::Exit(Exit::MmioWrite { .. }); }
        Err(t)            => { cpu.rip = cur_addr; let addr = t.fault_addr(a);
                               return StepResult::Exit(Exit::UnmappedMemory { addr, .. }); }
    }
}
IrOp::Jump { target } => {
    cpu.rip = read_val(target, &temps);   // store the new RIP in the state
    return StepResult::Continue;          // the dispatcher loop will pick up from the new RIP
}
IrOp::Syscall => {
    cpu.rip = block.guest_start + block.guest_len as u64;  // past the syscall instruction
    return StepResult::Exit(Exit::Syscall);
}
```

### 8.2 JIT via Cranelift (build SECOND)

Process-owned JIT storage and the process code map use PID plus fork generation.
For a raw host fork that shares the address space, the embedder calls
`x86jit_core::mark_shared_address_space_child()` on child entry, before engine use
or starting child threads. A fixed lock-free table retains up to 1024 live markers,
keyed by PID plus fork generation; repeated marking is idempotent. After definitive
child exit/detachment and before PID reuse, the parent calls
`x86jit_core::retire_shared_address_space_child(pid)` to reclaim the marker.
Slot versions prevent stale reads or retirements from matching a reused slot.
Copying-fork descendants ignore inherited markers and reclaim their slots on
marking because their atfork handler bumps the fork generation.
Raw-syscall shared-VM children bypass that handler and
retain the parent's generation. `Vm::child_view` initializes fork tracking in the
parent. If the table is full, the child must exit/exec without engine use.
The identity distinguishes the child from its parent despite their shared statics,
and covers every thread in the child without relying on inherited TLS or locks.

`JitBackend` constructors and `Default` are infallible and defer executable storage
until compilation. New and inherited backends are inert in a marked PID: compilation
refuses before parent queue/compiler locks, reports the existing typed reason
`CompileFailureReason::OtherBackendError` to the failure observer, and retains the
`Backend` interpreted fallback. Reusing this reason preserves exhaustive downstream
matches. Observer callbacks still run and must be safe to invoke in the child.
The additive `try_new` detects the mark before constructing backend state;
`try_materialize` and `try_materialize_region` return typed compilation failures.
Code-memory operations refuse allocation, finalization, write access and release.
Process-global code registration retains `Result<(), CodeMapFull>`, refusing with
`CodeMapFull` before replacing the map or entering its writer. Unregistration
refuses as well. Signal lookup, fault classification and the step-one child
interpreter remain available. The marker is admission control, not a substitute
for the child stack/TLS and
no-inherited-destructors contract in `child`.

#### 8.2.1 Compiled block ABI and access to guest state — READ BEFORE M4

> **This is the black box that has to be opened before you write the JIT backend.** The "baked-in base address" from section 1 is not enough — you must decide exactly *how* the generated host code reads and writes guest registers and memory. Without this you won't move.

The model (standard for JITs): **the guest register file is a struct in host memory, and the compiled block receives a pointer to it.** The block does not keep guest registers in host registers permanently — it reads/writes them as fields of the `CpuState` struct at known offsets. Cranelift allocates host registers only for *temporaries within the block*.

The compiled block has a fixed signature (ABI). Proposal:

```rust
/// The signature of every compiled block. All blocks have THE SAME signature,
/// so that the dispatcher can jump into them uniformly.
type CompiledFn = unsafe extern "C" fn(
    cpu: *mut CpuState,   // pointer to the guest register file
    mem: *mut MemCtx,     // memory context (host_base of the guest buffer + metadata)
) -> u64;                 // encoded StepResult/Exit — see below
```

Inside the block:
- **`ReadReg { reg }`** → Cranelift emits a `load` from `cpu + offset_of(reg)`.
- **`WriteReg { reg, .. }`** → `store` to `cpu + offset_of(reg)` (with the upper-bit zeroing rule for a 32-bit write!).
- **`Load { addr, size }`** (RAM) → compute `mem.host_base + guest_addr`, `load`. Inline, no callback.
- **`Store`** analogously.
- **`InsnStart { guest_addr }`** → no code emitted; remember `guest_addr` as a compile-time const so the trapping accesses that follow can store it to `cpu.rip` before returning an `Exit`.
- **End of block** (`Jump`/`Branch`/`Ret`) → store the new RIP to `cpu.rip`, return the "Continue" code.
- **`Syscall`/access to Trap/exception** → store RIP per the convention (past `syscall`; the remembered `InsnStart` address for a fault), return the code of the appropriate `Exit`.

> **The offsets of `CpuState` fields** must be stable and known in codegen. Use `#[repr(C)]` on `CpuState`, compute the offsets once (via `core::mem::offset_of!` or constants) and pass them to the backend. This ties the struct layout to the generated code — document it as a contract.

#### 8.2.2 Block result encoding

The compiled block returns a `u64`, because `extern "C"` won't conveniently return an enum. Define an encoding: e.g. the value `0` = `Continue`, and nonzero values encode an `Exit` variant (the type in the upper bits, data such as the syscall number/address in the lower bits — or write the details to a field in `CpuState`/`MemCtx` and return only the discriminator). `run_compiled` from section 8 decodes this `u64` back into a `StepResult`. Keep this encoding in one place, shared between codegen and `run_compiled`.

#### 8.2.3 Translating IR operations to Cranelift

The same `match` on `IrOp`, but instead of *executing*, it **describes operations to the Cranelift builder**, which generates host code. Cranelift does register allocation and instruction emission per-target (x86-64 on a desktop, ARM64 on a Mac — from the same IR). You build the `Temp → cranelift Value` map per-block (a vector of size `ir.temp_count`).

```rust
IrOp::Add { dst, a, b, .. } => {
    let va = clif_val(a, &mut builder, &temp_map);   // Temp→Value or iconst for Imm
    let vb = clif_val(b, &mut builder, &temp_map);
    let res = builder.ins().iadd(va, vb);            // does NOT compute — describes
    temp_map[*dst] = res;
    // flags: emit Cranelift operations computing CF/ZF/SF/OF and store to the flag fields in CpuState
}
IrOp::ReadReg { dst, reg } => {
    let off = reg_offset(*reg);
    let v = builder.ins().load(types::I64, MemFlags::trusted(), cpu_ptr, off);
    temp_map[*dst] = v;
}
```

**Guest memory access must be inlined** — the JIT emits direct `load`/`store` to the guest buffer (`mem.host_base + guest_addr`), NOT a callback. Only `Trap` regions and syscalls cause a trap-out. This is the "thick for the rare, thin for the hot" boundary from section 1.

> **Inline access needs a safety strategy — this is a zero-th-class M4 decision, not a detail.** `host_base + guest_addr` with `MemFlags::trusted()` and no check is **host-process UB** the moment the guest computes an address outside `[0, size)` (a guest bug, a wild pointer, an attack) — it reads/writes outside the host `Vec`, corrupting the emulator. `Exit::UnmappedMemory` exists in the API but *inlined code has no way to reach it* unless you emit something. Pick one, up front:
> - **Explicit bounds/permission check** (recommended start): before each inlined access, emit `cmp guest_addr, size` + a branch to a slow-path stub that returns `Exit::UnmappedMemory`/`MmioRead`/`MmioWrite`. Cheap (a predictable branch), portable, and the *same* check routes Trap/MMIO out — so it does double duty with §5.2. A per-page permission bitmap (R/W/X, Ram/Trap) makes it one lookup.
> - **Guard pages + SIGSEGV handler** (qemu-style): map the guest space with `PROT_NONE` guards, catch the fault, translate the host fault PC back to a guest exit. Fast (no per-access branch) but heavy, deeply `unsafe`, and platform-specific — defer.
> - **Reserve the full space** (only viable for a 32-bit guest, where 4 GB is mmap-able): no per-access check at all. Not an option for a 64-bit guest.
>
> Corollary: in `Flat`, guest address `0` is a valid buffer offset, so a guest null-deref does **not** fault unless a per-page permission bit says "unmapped". If you want faithful `#GP`/`#PF` on null, you need that bitmap — the "one addition" hot path silently assumed checks that must actually exist.

**Memory-consistency tiers (`MemConsistency`, §4.1):** these policies govern guest loads/stores on aarch64 hosts. JIT inline lowering:

| Tier | Inline store | Inline load |
|------|--------------|-------------|
| `Fast` | `STR` | `LDR` |
| `AcqRel` | `DMB ISH; STR` | `LDR; DMB ISH` |
| `FullTso` | `STR; DMB ISH` | `LDR; DMB ISH` |

`FullTso` has no leading store fence. x86 hosts use plain loads/stores in every tier. A leaner STLR/LDAPR AcqRel mapping is a future optimization, not the current lowering.

**Coverage:** JIT guest-memory helpers (string operations, masked stores, x87, FXSAVE and other memory helpers) receive tier barriers through `call_memory_helper`. On aarch64, `AcqRel` fences before write calls and after read calls; `FullTso` fences before read calls and after all memory calls; `Fast` adds none. Post-call barriers precede fault/exit handling, including partial progress. The interpreter honours `MemConsistency` for ordinary accesses and helpers with matching SeqCst fences on aarch64, including cold blocks, fallback execution, MMIO replay and the allocation-free child step; x86-64 execution adds no tier barriers. The child snapshots the VM's tier and selects its ORDER specialization once per step. Each actual transfer in its deferred scalar/vector store batch receives the tier's write guard, in addition to the walk's staging barriers. This conservatively fences both 8-byte halves of a 128-bit vector lane and preserves the normal interpreter's barriers between lanes; Fast and x86-64 add no tier barriers; child code-tag fences remain independent (§10). REP transfers commit directly under direction-specific helper barriers, bypassing the deferred batch. Host-fault classification resumes within the transfer routine and returns through the same Drop guards, retaining post-fences even when a fault leaves partial writes. These boundaries preserve ordering across helper calls (SDM Vol 3A §11.2.2, §11.2.4.1). The locked-op limitation below remains.

**Explicit ordering:** `IrOp::Fence { kind: Full }` carries MFENCE and precedes CPUID and Real16 IRET (SDM Vol 3A §11.3). It executes a SeqCst host fence in the interpreter and a Cranelift full fence in the JIT (`DMB ISH` on aarch64, `MFENCE` on x86-64), in every tier. LFENCE/SFENCE carry `Load`/`Store`; the interpreter uses Acquire/Release fences. The JIT emits DMB ISH for both in **every aarch64 tier**, using a conservative full fence to enforce explicit ordering independently of the ordinary-access tier, and emits no extra barrier for them on x86-64. Guest non-temporal stores currently use ordinary host stores; host non-temporal lowering would require SFENCE on x86 too (SDM Vol 3A §11.2.2, §11.2.5; Vol 2B §4.3 MFENCE/SFENCE; Vol 2A §3.3 LFENCE).

**Locked operations:** atomicity is implemented on both backends in every tier for a naturally aligned operand. The bus lock does not depend on the operand's alignment (SDM Vol 3A §11.1.2.2 p. 11-4; Vol 2A LOCK p. 3-565). The interpreter, the shared helper that compiled CMPXCHG8B and CMPXCHG16B call, and the replay of an instruction whose compiled access took a host alignment fault (decision-13) update a misaligned operand with a compare-and-swap loop on the containing aligned 8- or 16-byte host word (`Memory::locked_unaligned`), which preserves the bytes outside the operand and retries after any store to the word: atomic inside one aligned 16-byte granule. **Limitation:** on that path no host primitive covers an operand that crosses an aligned 16-byte granule (a word at 15, a doubleword at 13–15, a quadword or CMPXCHG8B operand at 9–15, mod 16), or one whose containing word would pass the start or end of the backing (an owned backing whose length is not a multiple of 16, or an embedder's `HostRam` span whose start or end is not 16-byte aligned), so it is a plain read, update and write: it completes with the single-processor result but is not atomic against another vCPU. The A20-split operand of the 286 and 386, the twenty-bit word that wraps its segment or the bus (§17.10) and the RAM piece of a locked operand partly in a Trap region (§5.2) have the same policy. A cross-vCPU rendezvous for them is a follow-up. Other compiled locked forms issue the host atomic at the operand's address, so a misaligned operand there depends on the host. An x86-64 host performs it atomically at any alignment, a crossing one as a split lock (SDM Vol 3A §11.1.2.2), so a crossing operand is atomic in compiled code and not on the path above: the tiers diverge. On an aarch64 host the access either completes in hardware or takes an alignment fault: FEAT_LSE2 lets the LSE atomics complete inside a 16-byte granule, while XCHG's exclusives and a crossing operand are not covered by it (Arm architecture behaviour, not pinned in `oracles/`). A faulting access completes only through the embedder's decision-13 handler, whose replay takes the path above; without one the process receives SIGBUS. Full x86 locked-op ordering (SDM Vol 3A §11.2.3.9) is not guaranteed on aarch64 LL/SC paths: Cranelift 0.135.8 emits LDAXR/STLXR loops without a trailing DMB when LSE is unavailable, and XCHG uses LL/SC even with LSE. This limitation is separate from explicit fences, which always enforce their requested ordering.

> **Switching tiers = flushing the translation cache.** Barriers are baked into compiled blocks, so a block compiled under `Fast` is wrong under `FullTso`. The tier is per-`Vm` config; if you ever make it switchable at runtime, the switch must invalidate the whole cache (and it's the cheap, correct way to implement it — don't try to key the cache by tier). Decide the tagging mechanics (lift-tags vs codegen-applies-blanket) in M7 — the lift sketches emit `MemOrder::None` today on purpose, which makes codegen-applies-blanket the natural fit.

> **M4 build order:** first just the offsets + ABI + "a block that only returns Continue with a new RIP" (without real operations) — verify that the dispatcher jumps into it and returns. Then add the translation of `IrOp` one by one, validating each against the interpreter (oracle). Don't write the whole backend at once.

---

## 9. Translation cache and dispatcher

### 9.1 Cache

Key = guest address. Value = the translated block (for the JIT: a pointer to executable host code; for the interpreter: an `IrBlock` to walk in a loop).

```rust
pub struct TranslationCache {
    // Shared between vcpus (multithreading) — hence the synchronization.
    // SEAM (§17.4): the key is u64 (guest address). If processor modes ever get
    // added, switch to BlockKey { guest_addr, mode } — today mode is always Long64.
    map: RwLock<HashMap<u64, CachedBlock>>,
}

#[derive(Clone)]
pub enum CachedBlock {
    Interpreted(Arc<IrBlock>),
    Compiled {
        entry: CompiledPtr,   // pointer to host code — see the note on Send/Sync
        guest_len: u32,
    },
}
```

> **Send/Sync — pitfall M7 (multithreading).** A raw pointer `*const u8`/`fn` is **not `Send + Sync`**, so a `CachedBlock` with a bare pointer CANNOT be put into a cache shared between threads behind an `Arc<TranslationCache>` — the compiler will reject it only in M7, when you add threads, and this will be surprising, because single-threaded it worked. Solve it right away in M4: wrap the pointer in a type for which you manually attest safety (the code is immutable after compilation and executable from all threads):
> ```rust
> #[derive(Copy, Clone)]
> pub struct CompiledPtr(pub *const u8);
> // Safe: compiled code is read-only and executable from any thread;
> // its lifetime is guaranteed by the owner of the code memory (the JIT arena lives as long as the Vm).
> unsafe impl Send for CompiledPtr {}
> unsafe impl Sync for CompiledPtr {}
> ```
> The interpreter (`Arc<IrBlock>`) is `Send + Sync` automatically. Make this type in M4, even if multithreading comes in M7 — otherwise you'll be reworking signatures later.

> **Ownership of JIT code memory.** Compiled blocks live in an arena of executable memory (`memmap2`, W^X; on macOS `pthread_jit_write_protect`). The arena belongs to the `Vm` and lives as long as it. `CompiledPtr` is a borrow into this arena — don't free it as long as the cache may point to it. SMC invalidation (section 10) removes the entry from the map, but the code memory is freed only when the arena is destroyed or via a separate recycling mechanism (a late optimization).

### 9.2 Dispatcher loop (the heart)

```rust
fn run(&mut self, budget: Option<u64>) -> Exit {
    let mut blocks_run: u64 = 0;
    loop {
        if let Some(b) = budget {
            if blocks_run >= b { return Exit::BudgetExhausted; }
        }

        let pc = self.cpu.rip;

        // Fetch from the cache or lift (miss). CachedBlock is cheap to clone
        // (Arc<IrBlock> or pointer+len) — we don't hold a reference into the RwLock's
        // interior for the duration of execution, because the backend mutates memory/cache (SMC).
        let block: CachedBlock = match self.vm.cache_get(pc) {
            Some(b) => b,                        // HIT — clone of the Arc/pointer
            None => {
                // MISS. A lift error (unknown instruction) is NOT a run() error —
                // it's a legal exit that tells the user "add this instruction to the lift".
                match lift_block(&self.vm, pc) {
                    Ok(ir) => {
                        let materialized = self.backend.materialize(&ir);
                        self.vm.cache_insert(pc, materialized.clone());
                        materialized
                    }
                    Err(LiftError::Unsupported { addr, bytes, len }) => {
                        return Exit::UnknownInstruction { addr, bytes, len };
                    }
                    Err(LiftError::DecodeFault { addr }) => {
                        return Exit::UnmappedMemory { addr, access: AccessKind::Execute };
                    }
                }
            }
        };

        // &self.vm.mem — NOT &mut: guest RAM is interior-mutable (§8 pitfall).
        match execute(&block, &mut self.cpu, &self.vm.mem) {
            StepResult::Continue => { blocks_run += 1; }   // RIP already updated in the block
            StepResult::Exit(exit) => { return exit; }     // syscall, mmio, breakpoint, ...
        }
    }
}
```

> **Budget vs block chaining (a trap that lands in M5, kills M7).** The budget is counted here, once per block. **Block chaining** (§12 M5) makes blocks jump straight into each other *without returning to the dispatcher* — so `blocks_run` stops ticking and a tight guest loop never yields `BudgetExhausted`, starving every other vcpu. When you add chaining, keep a preemption path: either a periodic counter check compiled into chained edges, or an external "please exit" flag the chained code polls at back-edges. Decide it with chaining, not after.

> **Benign double-lift race (M7).** Two vcpus can miss the same PC simultaneously, both lift, both insert. Harmless — lifting is pure and the result is identical, one insert just overwrites the other with an equal value. Don't add locking to prevent it; at most dedupe opportunistically. Worth one sentence so nobody "fixes" it into a bottleneck.

> **Cache ownership model.** `cache_get` returns a `CachedBlock` by clone (Arc for the interpreter, a copy of pointer+len for the JIT), NOT a reference into the interior of the `RwLock`. The reason: during `execute` the backend mutates guest memory, which may trigger SMC invalidation (a write to the cache) — holding a live `&` reference into the interior of the same lock would be a deadlock/borrow conflict. The clone breaks that dependency. `cache_get`/`cache_insert` take and release the lock briefly inside.

---

## 10. Self-modifying code (SMC) — cache invalidation

Because guest code lies in a *data* buffer (the guest may modify it), a write to a region from which you have a translated block invalidates that entry. At the start you can ignore it; add it when you hit a game/program that modifies its own code.

- Track which guest pages have translated blocks (a "code" bit per page).
- On a write to such a page → remove the affected entries from the cache (with the JIT: also mark the host code as dead).
- On the next execution → miss → re-lift from the changed bytes.

Invalidation uses a page index to find candidate units, then compares their source
byte spans with the accumulated write ranges. A sparse radix directory stores an
exact 4096-bit coverage bitmap per indexed page, for cached units and registered
background sources. Ordinary writes check coverage under the spans read lock; a
miss advances the publication generation before releasing that lock, without
allocating, hashing keys or acquiring the write lock. Publishers require the write
lock, so they either precede the coverage check or observe the generation advance.
Overlapping writes reacquire the write lock and recheck the touched pages. Per-page
entries carry local byte spans, so candidate checks and coverage rebuilding iterate
those entries directly without rehashing each key into the unit table.
Disjoint writes preserve cached units, page tags and the dispatch epoch, so per-vCPU predictors remain warm. Every drained
range advances a separate source-publication generation under the cache spans lock.
Synchronous lifts and tier-ups snapshot it before reading source bytes or IR and reject
publication if it changes, including when an uncached region successor was written and
no cached unit overlapped. Background requests use that check only until their source
spans are registered under the same lock; subsequent invalidation cancels only requests
whose source spans overlap the write. Unregistered requests also retain their claim
generation, so cancellation during source reads needs no pending-map mutation.
Every block lift tags its fetch pages before
copying source bytes, including uncached region successors. These tags have counted
**temporary lift owners**, held through registration/publication. Dropping a failed or
speculative lift releases its tags, including rejected successors and unused lookahead;
cached units, registered requests, fetch approvals and interpreter-only suppression
retain their independent ownership. Retired request sources are reconciled under the
spans lock; the normal publication path does not scan existing units.

The untagged-to-tagged transition requires ordering on **both** sides:
a writer's byte store can remain buffered while its post-store tag/watermark load
observes the old clear value (SDM 325462-092US Vol 3A §11.2.3.4, Example 11-3).
A lift batches all new tags, orders writers, and validates its copied bytes by
re-reading them afterwards; a mismatch retries the lift. Tag and barrier epochs
cover concurrent lifters too. Already synchronized tags need no new barrier.
Dropping speculative tags allows the same page to transition again: repeated
rejected successors or unused lookahead can therefore cost one barrier per lift.
There is no negative cache: a rejection can depend on the entry/decode mode or
mutable source bytes, and cannot safely suppress the next page's synchronization.

`Memory` selects its source protocol during construction, before guest writes or
vCPU execution. The mode remains fixed for that memory's lifetime, independently
of the guest consistency tier. `Vm::memory_stats()` exposes `source_barrier_mode`,
`source_barrier_failures` (failed attempts, including probing/registration),
`source_barrier_last_errno` (retained even after a successful retry), and
`source_barrier_uncached_executions` (blocks dispatched without publication after
a later process-barrier failure).
Always-on source-publication counters exclude construction probing/registration:
`source_barrier_calls` counts each process-barrier attempt, including failures
and retries; `source_barrier_ns` measures elapsed host nanoseconds inside those
calls only, excluding tagging, validation and retry yields.
`source_barrier_tagged_pages` counts pages newly tagged as code (4 KiB), including
aliases, at each untagged-to-tagged transition in both Process and Degraded modes.
Re-tagging a cleared page counts again; failed attempts and retries do not add
pages. With `source_barrier_calls`, the ratio gives pages covered per barrier,
because one barrier covers every page tagged since the previous one. There is no
per-barrier page attribution. Degraded mode makes no process-barrier calls and
leaves calls and nanoseconds at zero while still counting page transitions.
These counters are independent relaxed snapshots.

- **Process mode (Linux/FreeBSD):** probe private registration and an actual
  `membarrier(PRIVATE_EXPEDITED)` before choosing this mode. Only registration
  success is cached process-wide. ENOSYS/EINVAL/EPERM select degraded immediately;
  other errors, including ENOMEM and an interposed EINTR, get at most three attempts
  per probe operation. A later VM can retry registration after an earlier failure.
  Interpreter/embedder stores use a compiler barrier before checking tags;
  compiled stores retain their ordered IR store and live atomic watermark load.
  A newly tagged source uses a process barrier plus local full fences.
  [Linux membarrier(2), DESCRIPTION and RETURN VALUE](https://man7.org/linux/man-pages/man2/membarrier.2.html)
  guarantees full ordering on running siblings; non-running threads already meet
  the condition, and the result for fixed zero-flag commands is invariant after
  a successful probe under unchanged calling conditions. A per-thread filter can
  still reject a later invocation. Each publication barrier gets at most three
  attempts; persistent failure leaves the synchronized tag epoch unchanged and
  executes only the entry block once in the interpreter, without compiling or
  caching it. The next execution re-lifts current bytes. Failed region lifts
  cannot publish or enter the background queue. The VM keeps its original mode:
  changing an active VM to degraded would leave earlier unfenced stores uncovered. [FreeBSD membarrier(2)](https://man.freebsd.org/cgi/man.cgi?query=membarrier&sektion=2)
  provides the corresponding private expedited operation (14.1 and later).
- **Degraded mode (probe failure, macOS, other hosts):** ordinary interpreter/embedder
  memory stores reach a hardware SeqCst fence before its precise tag check.
  The JIT receives a constant always-open watermark covering all valid RAM below
  2^47, so every compiled store calls the same fenced notifier, even on untagged
  data pages. The precise code extent stays separate for sparse metadata walks.
  The lifter's local full fence and source re-read now have matching writer-side
  fences. If the writer misses the tag, its preceding byte store is visible at
  re-read; if it observes the tag, ordinary dirty-range/publication validation
  handles the write. Cost: a hardware fence on every store and a Rust helper call
  on every compiled inline store; store-heavy guests can be substantially slower.
  Translation and interpretation remain available after a failed probe. Compiled
  masked, narrowing and REP helpers notify each committed element through the
  same fenced writer hook; x87/FXSAVE notify committed images and XSAVE uses
  notifying `Memory` transfers.

**Shared-address-space children (decision-19):** guest bytes remain shared with
live parent vCPUs. The frozen code snapshot is only a preflight refusal set, not
the SMC notification source. Every committed scalar/vector/REP/atomic transfer,
including a partially completed host store, journals its written pages into a
preallocated atomic bitmap, executes a SeqCst fence, then checks the live code
tags. A hit sets the bridge's dirty flag and notifies every registered dispatch
word. Children always use the full writer fence: a separate host process sharing
the address space is outside a private process barrier's scope. Thus a missed
tag leaves bytes visible for the lifter's fenced source re-read, and an observed
tag queues normal invalidation before the child can publish a guest signal.

The child only accesses stable atomic tags, journal words and append-only dispatch
nodes. It neither allocates nor takes parent locks. Nodes cover existing and newly
created vCPUs and retain their dispatch words until Memory drops. The parent
drains the atomic journals under its ordinary SMC serialization, expands active
and per-view retired alias declarations, and conservatively invalidates whole
written pages. Removed declarations remain in parent-side journal history until
the view is dropped and its journal pruned, covering deferred and racing writes.
Notification fences between publishing the bridge-global dirty flag and reading
dispatch links; registration fences between publishing its link and reading the
flag. These paired SeqCst fences prevent both sides from missing publication.
Per-vCPU child notice epochs bracket drain/acknowledgement, preserving racing
notifications provided fewer than 2^64 notifications reach the node between
samples; independent exit/exhaustion bits survive.
The post-exit arena log remains required for watched data and final reconciliation.

**macOS:** no sibling is suspended or enumerated. `mach_thread_self()` is checked
against MACH_PORT_NULL/MACH_PORT_DEAD and its acquired right is released before
choosing degraded mode (EINVAL for failed self-identification, ENOTSUP otherwise).

**Register-query investigation: Outcome B (ordering proof incomplete).** The
following is READ evidence from Apple's **XNU `xnu-12377.1.9`**, not an assertion
about every shipping kernel. The question is whether, for every sibling, successful
query completion orders its preceding guest stores before the caller's source
re-read, and orders the caller's preceding tag stores before the sibling's later
tag loads. Both directions are required. A passing race witness or another
runtime's use of the operation cannot establish that contract.

- [libsyscall/wrappers/thread_register_state.c, thread_get_register_pointer_values](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.1.9/libsyscall/wrappers/thread_register_state.c#L32):
  on arm64 the wrapper requests `ARM_THREAD_STATE64` with `thread_get_state`, then
  extracts pointer-shaped register values. It adds no explicit memory barrier.
  Zero output capacity still performs the query but returns
  `KERN_INSUFFICIENT_BUFFER_SIZE`; that return must not be confused with a failed
  kernel query in an experiment. The
  [Mach interface, osfmk/mach/thread_act.defs](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.1.9/osfmk/mach/thread_act.defs#L116)
  allows stale state for an executing target, including the old-style
  `thread_get_state` operation. That wording is not a memory-ordering promise;
  it also does not prove that this implementation skips stopping a running target.
- [osfmk/kern/thread_act.c, thread_get_state_internal](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.1.9/osfmk/kern/thread_act.c#L612):
  a live remote target is held, its mutex is dropped, and
  `thread_stop(thread, FALSE)` is called. Success reacquires the mutex, reads
  machine state, and calls `thread_unstop`; both success and interruption call
  `thread_release` before returning. The read path does not call `act_set_state`;
  the setter shares hold/stop machinery but is a separate operation. An inactive
  non-inspection target returns `KERN_TERMINATED`. The current-thread branch only
  reads machine state and does not stop the caller.
- [osfmk/kern/thread_act.c, thread_hold, thread_set_apc_ast_locked and thread_apc_ast](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.1.9/osfmk/kern/thread_act.c#L351):
  the first hold sets `AST_APC`; an active remote processor receives an AST check.
  The target's APC handler parks it while the hold count is positive.
  [osfmk/kern/sched_prim.c, thread_stop](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.1.9/osfmk/kern/sched_prim.c#L1424)
  sets `TH_SUSP`, requests another AST check while the target is on CPU, and waits
  on `wake_active`. `FALSE` means it need not wait for the target to become
  non-runnable. `thread_dispatch` updates the outgoing thread's state and wakes
  waiters after the context switch.
- [osfmk/arm64/machine_routines.c, cause_ast_check](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.1.9/osfmk/arm64/machine_routines.c#L1745)
  sends `SIGPast`.
  [osfmk/arm/cpu_common.c, cpu_signal_internal and cpu_signal_handler_internal](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.1.9/osfmk/arm/cpu_common.c#L409)
  publishes the signal with release ordering and executes `DSB ISHST` on the
  **sending** CPU before signaling. The receiving handler uses acquire operations
  and invokes `ast_check`. This supplies a caller-to-target ordering edge when
  that IPI path is taken. The sender's DSB does not drain the target's stores.

**Apple silicon ordering:** exception entry/return and ISB synchronize execution
context; they do not by themselves promise completion of preceding ordinary RAM
stores. DMB orders accesses; DSB additionally requires completion within its
specified domain. A software context switch must be examined for these operations,
not treated as an additional architectural barrier by name. The selected
[osfmk/arm64/cswitch.s, Switch_context and set_process_dependent_keys_and_sync_context](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.1.9/osfmk/arm64/cswitch.s#L227)
saves/restores registers and conditionally uses ISB; it does not contain an
unconditional target-side DMB/DSB. Architectural distinctions: Arm
[102336_0100_01_en, version 1.0 (2022), §§5–6, 10–11](https://documentation-service.arm.com/static/62a304f231ea212bb662321d).
Visibility here means visibility to coherent CPU observers, not writeback to DRAM.

**The missing step is universal scheduler-handshake coverage.** A target-side DSB
is not the only possible proof: a release/acquire acknowledgement chain can also
order stores. For the ordinary switch-out path, the candidate chain is guest store
→ outgoing-thread scheduler-lock release on its old CPU → caller's acquisition of
that lock → caller's source re-read. In the opposite direction the candidate is
tag store → caller's lock/signal publication → target's acquire before returning
to guest loads. The locks are real ordering operations:
[osfmk/kern/thread.h, thread_lock and wake_lock](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.1.9/osfmk/kern/thread.h#L1180)
map through
[osfmk/arm/simple_lock.h](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.1.9/osfmk/arm/simple_lock.h#L119)
and [osfmk/arm/locks_arm.c](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.1.9/osfmk/arm/locks_arm.c#L219)
to hardware locks or, when configured, ticket locks.
[osfmk/kern/locks.c, hw_lock atomic acquisition and hw_lock_unlock_internal](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.1.9/osfmk/kern/locks.c#L340)
and [osfmk/kern/lock_ticket.c, hw_lck_ticket_reserve_orig and hw_lck_ticket_unlock_inner](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.1.9/osfmk/kern/lock_ticket.c#L422)
use acquire and release ordering. It would be incorrect to dismiss these
chains merely because there is no explicit DSB in `Switch_context`.

What is **not established** is that every successful register query necessarily
crosses those two chains, including the already-off-CPU, run-queue, handoff and
migration branches. In particular,
[osfmk/kern/sched_prim.c, thread_isoncpu](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.1.9/osfmk/kern/sched_prim.c#L1376)
can return false without requesting a new target interrupt; its run-queue test has
explicit TODOs concerning dequeue-before-context-switch and handoff cases.
Proving the relevant old-core release was observed, and the future target acquire
cannot bypass the caller's publication, is still necessary on these paths. The
public register-state interface supplies no substitute guarantee. Thus the
argument stops at that quantified step, rather than claiming that exception entry
flushes a store buffer. This is an incomplete proof, not a measured counterexample
or a claim that the scheduler's locking is broken. Applying the inspected source
to an uninspected kernel version would be a further ASSUMED premise. macOS keeps
paired fences until these premises are discharged.

**Enumeration and lifecycle:**
[osfmk/kern/task.c, task_threads_internal](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.1.9/osfmk/kern/task.c#L3954)
takes a reference to each listed thread under the task lock. Port rights keep the
objects identifiable but do not prevent exit. A query that loses an exit race can
fail; failure is not a remote ordering acknowledgement. Treating a disappearing
thread as covered would require tracing termination's final release as well.
A thread omitted because it is inserted after the snapshot is different: the
caller stores the tag and fences **before** taking and releasing the enumeration
task lock. In
[osfmk/kern/thread.c, thread_create_internal](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.1.9/osfmk/kern/thread.c#L1436),
insertion into the task list takes that same lock before the thread is started.
The release/acquire handoff propagates the tag to the creating CPU. Its
`thread_create_running_internal2` calls `thread_start` before releasing the task
lock; `thread_start` uses `clear_wait` to make the new thread eligible to run.
The scheduler's
[osfmk/kern/sched_prim.c, thread_invoke](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.1.9/osfmk/kern/sched_prim.c#L2928)
acquires the new thread's scheduling lock before switching to it. These lock
handoffs explain publication to the new thread's first tag check, rather than
relying on elapsed wall time. A thread inserted before the snapshot is
enumerated even if it has not started. A blocked kernel thread need not receive a
fresh IPI; its prior switch-out and eventual resume must satisfy the scheduler
chains above. Neither enumeration nor creation resolves that outstanding proof.

**Caller and signals:** retain full local fences before enumeration and after all
queries; querying the caller itself supplies no remote barrier and can be skipped.
In the inspected read path, the kernel removes its temporary stop and hold before
returning to user space, including interruption cleanup. Therefore a normal signal
handler on the caller cannot run in a user-space suspend/resume window introduced
by this operation. This establishes the desired absence of that deadlock window,
not general async-signal safety of enumeration, allocation or embedding code.
The implemented degraded protocol holds no sibling at all. A future process-mode
implementation must also resolve errors after construction: simply switching an
active VM to degraded cannot retroactively fence its earlier compiled stores.

**Timing evidence (MEASURED, not ordering evidence):** a separate macOS arm64
experiment on Darwin 27.0.0, kernel `xnu-13432.1.9`, timed 101 passes after five
warmups, one process at a time. Each pass fenced locally, enumerated threads,
queried every sibling with capacity for 64 register pointers, released all rights
and the list, and fenced locally. Counts include the caller, whose query was
skipped. For 1/8/32 total threads, median pass times were 2.25/7.38/24.79 µs with
condition-variable-blocked siblings and 2.38/129.50/302.67 µs with spinning
siblings. Mean remote-query times were 0.585/0.576 µs (7/31 blocked siblings) and
28.03/32.22 µs (7/31 spinning siblings); active-pass p95 was 646.17/4852.21 µs.
This is the cost of the proposed operation sequence, not a supported normal-mode
barrier or a guaranteed latency bound. The measured kernel's source was not
available at that version tag; its equivalence to the inspected version is not
assumed. No normal-mode performance claim follows from this experiment.

A fresh seven-process-sample comparison of the unchanged automatic degraded mode
against the previously built, instrumented `55b9a49` baseline gave median Real16
cold-pass times of 3.357/3.593 ms and 8M compiled-store times of 37.043/10.071 ms
(current/baseline). Scoped process instructions retired, read with
`proc_pid_rusage(RUSAGE_INFO_V4)` including VM setup, were 66,249,053/65,938,207
and 809,378,823/330,283,127 respectively. Each workload ran in its own process,
serially, with alternating variant order. These short-run timings describe this
measurement only; the compiled-store instruction increase remains about 145%.

**Stopped, exited and newly created threads:** degraded mode needs no enumeration.
A stopped writer cannot pass its next tag check without first passing its fence;
a writer that already missed the tag has ordered its byte store before that check.
An exited writer has no future check/store, and its completed checks have the same
ordering. Threads created after a tag store start after its publication and pass
their first store/check through the same fence. In process mode a hypothetical
thread created after barrier enumeration starts after the tag store: kernel
thread creation/start synchronization orders its first tag load after publication,
so it either sees that tag (or a later ownership update) or its earlier source
write was already covered by the barrier. Existing sleeping/exited threads are
covered by the private expedited contract's non-running-thread guarantee. No
assumption that a thread list remains fixed is needed.

Registration and publication drain pending write notifications before checking source validity;
registered successors retain write tracking while pending. Single-instruction
step/replay lifts are transient and need no additional tags. Canceled requests keep their
pending slot until the matching completion drains, preventing duplicate compilation.
An opaque request identity prevents a stale completion from releasing a newer slot.

`Vm::memory_stats().cache` exposes lifetime invalidation calls and blocks dropped,
coverage bitmap/directory allocation bytes, background requests accepted, duplicate
attempts skipped, completions received and published, and rejections split by source invalidation, backend compilation failure,
and supersession. These relaxed counters are independent snapshots; a region counts
as one cache entry and one background request.

With an installed fetch policy, write notifications invalidate the whole 4 KiB page:
approval is page-wide, and `note_write` also reports protection changes. Surviving
same-page cached code must not bypass a revoked approval. Byte precision applies only
without a fetch policy. Unmap and child write-log invalidation remain conservative.

**Shared mappings.** The embedder creates the shared host mappings and declares each
pair of corresponding guest ranges with `Memory::declare_code_alias(a, b, len)`.
Declarations are symmetric and idempotent; overlapping declarations and multiple views
are supported. `remove_code_alias` removes an exact pair in either order. `unmap`
removes declarations overlapping the unmapped bytes in either view. The engine cannot
discover undeclared host aliases; those can retain stale translations. External writers
must still report completed writes with `note_write` on the view they wrote.

The sorted page index connects all declared views at conservative 4 KiB granularity.
Marking a translated page tags its connected views before widening the existing 64 KiB
watermark. Stores use the same range gate; its atomic load acquires the published tags.
Only a write hitting a tagged page expands aliases and reports the translated pages to
the dispatcher. Clearing a translation preserves tags needed by other live views, and
fetch approvals are forgotten with the corresponding translations. Whole-page unmap
invalidation likewise prevents a surviving sub-page translation from losing tracking.
The same cache spans cover interpreter blocks, compiled blocks, regions and background
tier-up publication. Registration can occur during execution; the embedder synchronizes
with the vCPU before relying on its completion. A store racing registration may precede
it. Far-apart views stretch the watermark across intervening data, increasing helper
traffic even when those data pages are not code. No alias lookup is added to ordinary
stores outside tagged pages.

**Who has to report the write.** The bullet above says "a write to such a page", and the trap is that a *write* is not one code path. The interpreter's stores go through `Memory::write*`, which calls `note_write`; but compiled code inlines stores as raw host writes, and the string, masked/narrowing vector, x87 and fxsave helpers write through raw bounds-checked views of guest RAM. Each of those has to report its own write or it silently invalidates nothing. That is not hypothetical — the JIT shipped for a long time reporting **none** of them, so a guest that patched another block and called it ran the stale translation, and the suite was green because every JIT-backed SMC test wrote from the *embedder* side instead of the guest's (task-329). The compiled store path reports through an inline range test against a code watermark; the helpers report each committed transfer through notifying memory views. Child-view stores use the atomic journal and live-tag protocol above.

**Chained execution has to be able to notice.** Invalidation is applied by `Vm::handle_smc` in the dispatcher loop, and compiled code follows chain/link/IBTC edges *without returning to it*. Marking a page dirty is therefore only half the job — the compiled inner loop leaves the chain when a code page has been dirtied, which bounds its deferral below to one block. Without that, "deferred to the end of the running block" quietly becomes "deferred until the chain happens to end".

**Cross-modifying code (another vcpu writes) is bounded by the guest's own protocol.** SDM Vol 3A §11.1.3 calls the *unsynchronized* form model-specific — real silicon may run stale bytes too, depending how far ahead of the executing processor the write landed — and defines the case that is guaranteed: the modifier stores the code then raises a flag, and the executing processor polls it, executes a serializing instruction, and only then runs the code. A guest following that is served correctly here, and `x86jit-tests/tests/threads/cross_modifying.rs` pins it on both backends. CPUID carries a full memory fence (SDM Vol 3A §11.3); the dirty-page chain leave above supplies translation invalidation, which a memory fence alone cannot provide. Do **not** add a validity check at the chain transfer itself to make the unsynchronized case behave — it would put a load, a compare and a branch on the path fast dispatch exists to keep free, in exchange for a promise the architecture does not make.

**Same-block SMC on the interpreter.** SDM Vol 3A §14.6, *Self-Modifying Code*,
describes invalidating prefetched/decoded instructions
affected by a write. In every decode mode, a same-linear-address interpreter store
overlapping bytes of a later instruction ends the block after the storing
instruction completes and retires. Dispatch invalidates and re-lifts at the next
RIP, so `run_steps` budgets and block formation cannot change the observed bytes.
Direct writes confined to already executed instructions, the storing instruction,
or bytes outside the block do not end it.

A per-walk memory view carries the running block's linear and host-address spans. Every successful
interpreter store checks that span independently of the shared code-page tag,
including bulk FP/state transfers, string iterations, vector lanes and locked
stores. A direct overlap takes a cold path that checks whether any later instruction
bytes were touched and zeros the walk's packed instruction state. Its other values
encode the first marker and an instruction in flight, supplying retirement accounting
with a shift. At the next marker, a single zero test recognizes the cut, finishes
retirement of the storing instruction and returns to dispatch. This replaces the
existing started-flag test; it adds no TLS access or shared atomic load.

The code-page cold path expands declared aliases before checking overlap, at
conservative 4 KiB page granularity. It can end a block whose remaining bytes were
not changed. This alias check relies on the store's alias page still being tagged;
direct same-linear-address detection does not. SDM Vol 3A §14.6 guarantees the
cross-alias case when a serializing operation, such as CPUID, precedes execution
of the modified instruction; the interpreter serves that case within a single block.
Undeclared host aliases remain outside this guarantee.

Another vCPU clearing the running block's tag cannot hide a subsequent direct
self-modifying store: the local span check still detects it. Cache invalidation alone
does not cut the walk, preserving the unsynchronized cross-vCPU contract above.
Real16 converts CS:IP to linear addresses for overlap while retaining IP for
retirement and refetch. `same_block_smc.rs` pins these properties without Unicorn,
including deterministic two-vCPU invalidation and shared host alias views.

> **Compiled same-block SMC remains deferred to the next dispatch.** A compiled
> block keeps executing old translations through its end, including the remainder
> of a region sub-block. A follow-up needs per-vcpu pending refetch state set by
> inline and helper stores, then an exit at the next instruction marker after all
> register/flag updates. That exit must flush region state, save the next RIP and
> account for the completed prefix instead of charging the whole block's instruction
> count. Returning directly from the store notification would skip part of the
> storing instruction. The existing watermark and non-code store gates must stay
> unchanged.

---

## 11. Multithreading (late milestone)

- **Per-vcpu:** `CpuState` (registers, flags, RIP), its own `run()` loop, its own host thread.
- **Shared (behind an `Arc`):** guest memory and the translation cache.
- **Cache:** translation happens once per block (whoever hits it first); synchronization via `RwLock` or a lock-free structure. Consider a per-vcpu cache for reading + a shared one for writing.
- **Memory model:** `MemConsistency` orders JIT inline accesses, guest-memory helper calls and interpreter accesses on aarch64 (§4.1, §8.2.3). aarch64 LL/SC locked-op ordering remains incomplete. Explicit fences apply on both backends.

---

## 12. Implementation plan (milestones)

The order is chosen so as to **have a working, testable core as fast as possible**, and to defer the hard things.

### M0 — Skeleton (days)
- Workspace, `Vm`/`Vcpu`, `CpuState`, `Memory` (flat buffer + map/write/read).
- Integrate `iced-x86`: a decoding loop that only prints instructions. Zero execution.
- **Test:** load hand-assembled bytes, decode, print — matches `objdump`.

### M1 — IR + interpreter, minimal instruction set (1–2 weeks)
- IR (the `IrOp`, `IrBlock` enums), lift for: `mov`, `add`, `sub`, `cmp`, `and`/`or`/`xor`, `push`/`pop`, `jmp`, `jcc`, `call`, `ret`, `lea`, `load`/`store`.
- Interpreter executing the IR. Flags variant A (materialized).
- Return-based `run()` + `Exit` (for now: `UnknownInstruction`, `Syscall`, `BudgetExhausted`, `Hlt`).
- **Test — differential (crucial):** on an x86 host execute the same block natively (a small asm stub) and compare the register/flag state with the interpreter. This is your free oracle.

### M2 — First real program (1–2 weeks)
- Extend instruction coverage until you build and run a **static x86-64 "hello world" ELF (Linux)** under the interpreter.
- A minimal syscall shim on the test side: `write`, `exit` (reacting to `Exit::Syscall`).
- (Optionally) `x86jit-elf` — a simple segment loader.
- **Psychological milestone:** you see "hello world" printed by emulated code. The whole loop works end-to-end.
- **⚠️ Pick a nolibc / freestanding binary, NOT a static glibc one.** A static-glibc "hello world" runs `__libc_start_main`, whose very first steps call SSE2 `memcpy`/`strlen` — so "static glibc hello world" secretly requires a chunk of M8 (SIMD) before it prints anything. For M2 use a freestanding binary that issues `write`/`exit` via raw `syscall` (hand asm or `-nostdlib`), or musl built with SIMD disabled. Note it in the M2 tasks; otherwise "1–2 weeks" hits a SIMD wall.

### M3 — Translation cache (days–a week)
- A cache keyed by guest address, a dispatcher with hit/miss.
- For now the value = `IrBlock` (interpreter with cache — you skip the re-lift, but not yet the JIT).
- **Test:** a loop in guest code does not re-lift blocks (count miss/hit).

### M4 — Cranelift / JIT backend (2–4 weeks)
- `x86jit-cranelift`: the same `match` on `IrOp`, but describing to Cranelift.
- RAM access inlined; syscall/trap trap-out.
- **Test:** the JIT gives identical state to the interpreter over the whole corpus (interpreter = oracle for the JIT). Measure the speedup.

### M5 — Performance (ongoing)
- Block chaining (stitching blocks without returning to the dispatcher).
- Lazy flags (variant B).
- Superblocks / traces, if worth it.

### M6 — SMC invalidation
- Tracking pages with code, invalidation on write.

### M7 — Multithreading + TSO
- Many `Vcpu` over a shared `Vm`, cache synchronization.
- `MemConsistency` tiers + barriers in codegen (`Fast`/`AcqRel`/`FullTso`, §8.2.3).
- **Test:** a multithreaded program communicating through memory produces a deterministic result.

### M8+ — SIMD (SSE/AVX)
- XMM/YMM in the state, lift of vector instructions. A big, separate chapter. Games require it, but not at the start.

---

## 13. Testing strategy

1. **Differential testing (the most important).** On an x86 host execute a block natively and compare with the interpreter/JIT. Automate: generate random instruction sequences, execute both, compare state. This catches 90% of semantic bugs.
2. **Interpreter as oracle for the JIT.** For every block: the interpreter and the JIT must produce identical state.
3. **Per-instruction unit tests.** For each lifted instruction: a set of inputs (including edges: overflows, zero, sign) → expected state + flags.
4. **Decoder fuzzing.** Random bytes → the lift must not panic (at most `UnknownInstruction`).
5. **Corpus of real binaries.** A growing set of static ELFs as end-to-end tests.

---

## 14. Open design decisions (to be resolved along the way)

- **Budget in instructions or blocks?** Recommendation: blocks (cheaper). Resolve in M1.
- **Lazy flags right away or later?** Recommendation: later (M5). Simplicity first.
- **Flat memory or softmmu?** Flat at the start (M0), softmmu when you hit a sparse/scattered address space.
- **Time and entropy sources (decided, decision-12, 2026-11-04):** `Vm::set_cpu_source` installs a shared `CpuSource` handle for new vCPUs. Both backends sample it exactly once per executed RDTSC, RDTSCP, RDRAND or RDSEED. A TSC read returns a paired counter/AUX sample; a random read returns a value or unavailability. Without a source, TSC is `0x12345678`, AUX is zero, and random reads use a deterministic per-vCPU stream with CF=1. The source owns rate, offset, cross-vCPU monotonicity, entropy quality and replay; the embedder keeps invariant-TSC and CPUID leaves 0x15/0x16 consistent with it. Installing a source does not advertise features. Clones and forks share the external source state without snapshotting it. Source methods must not panic or re-enter execution. CPUID stays data-driven (decision-11).

**Hooks (model A) alongside return-based?** General debugging hooks remain undecided; the core stays return-based apart from the `CpuSource` exception above.
- **How to represent Cranelift values vs IR temps?** A `Temp → cranelift Value` map built per-block in the backend.
- **How to represent CPU exceptions (`#DE` divide-by-zero, `#UD` from `ud2`, `int3`)?** They're guest-visible faults, not lift failures. Options: a dedicated `Exit::Exception { vector, error_code }` (extensible, honest) vs. folding into `Exit::Fault`. Recommendation: a dedicated exit — real programs `SIGFPE`/`SIGILL` on these and HLE must see the vector. Resolve before lifting `div` (M1/M2).
- **Inline memory-access safety: bounds check vs guard pages?** (§8.2.3) Recommendation: explicit per-access bounds+permission check first (portable, doubles as the MMIO trap path), guard pages only as a measured M5 perf optimization. Resolve in M4.
- **Consistency-tier mechanics** (§8.2.3): does the lift tag every `Load`/`Store`, or does codegen apply the tier as a blanket policy to `MemOrder::None` accesses? Recommendation: codegen-applies-blanket (lift stays tier-agnostic; sketches emit `None` today). Selective tagging (only provably-shared pages) is a later optimization that *reopens the AcqRel gap* — measure before doing it. Resolve in M7.
- **Breakpoint mechanism** (`Exit::Breakpoint`): a set breakpoint must split/bound the block at that address and invalidate any cached block covering it, so execution actually stops there. Needs a `set_breakpoint`/`clear_breakpoint` API + block-boundary handling. Design when debug support is actually built (post-M4); today the enum variant is a placeholder.
- **`hlt` in unit-test snippets** (testing.md): `hlt` is privileged and faults in the native oracle's user-mode process — vectors that terminate with `hlt` need a different terminator for the native oracle (int3+handler, or a ret trampoline). Unicorn/interpreter handle `hlt` fine. Resolve when building the native oracle (M1).

---

## 15. Dependencies

- `iced-x86` (MIT) — x86 decoder. Core's **only** dependency, and `x86jit-tests/tests/codegen/boundary.rs` fails the build if that changes.
- `cranelift-*` (Apache-2.0 WITH LLVM-exception) — JIT backend. Feature-gated (`x86jit-cranelift`).
- `memmap2` (MIT OR Apache-2.0) — allocation of executable memory for the JIT cache (W^X; on macOS `pthread_jit_write_protect`).
- `goblin` (MIT) — ELF parsing, in the loader helper (`x86jit-elf`) only.
- `unicorn-engine` (**GPL-2.0**) — the differential oracle. Test-only, optional, off by default; see below.

> **License.** Every dependency of the *shipped* crates — core, the Cranelift backend, the
> ELF helper — is permissive (MIT / Apache-2.0), so a consumer is free to choose their own
> licence, copyleft included.
>
> **The one exception, stated because an earlier version of this section wrongly said there
> was none:** `unicorn-engine` is **GPL-2.0**. It is optional, behind the `unicorn` feature,
> confined to `x86jit-tests` (which is `publish = false` and which nothing depends on at
> build time), and dynamically linked against the system `libunicorn.so`. CI does enable it,
> so the default CI test binary links it — but nothing shipped from this repository does.
> Full analysis in [`PROVENANCE.md`](../../../PROVENANCE.md) §5.

---

## 16. Where the surprises live (consolidated list of pitfalls)

One place gathering all the mines scattered across the document. Review it before every milestone — each of these things will block progress or give a silent, hard-to-find bug if you overlook it.

**Instruction atomicity vs the retry convention (pitfall #0 — silent, cross-cutting, hits in M1):**
- **No state may commit before an instruction's last possible trap.** RIP points at the faulting instruction and the user retries it (§8), so a `push` that moves RSP before a faulting store, or an RMW that writes flags before a faulting store, corrupts state on retry. Within one guest instruction, emit all trapping ops (load/store) **before** all committing ops (WriteReg, flags), or prove idempotence. Encode in the lowering helpers (§7 pitfall 3). Invisible until a fault-retry happens.
- **RIP-on-trap needs the instruction's address, which the IR must carry.** A trapping `Load`/`Store` (or an exception) must set `cpu.rip` to the *faulting* instruction — but `IrOp`s don't otherwise know their address, and `guest_start + guest_len` is the block END (right only for the block-terminating `syscall`). The lift emits `IrOp::InsnStart { guest_addr }` at each instruction boundary; the interpreter tracks it in `cur_addr`, the JIT bakes it as a const for the following trapping accesses (§6.2, §8.1, §8.2.1). Miss this and every mid-block fault resumes at the wrong RIP.

**x86-64 semantics (hits in M1–M2, at lift time):**
- **Zeroing of the upper 32 bits.** A write to `eax` zeroes bits 32–63 of `rax`; a write to `ax`/`al` — does not. Asymmetric. Encode it centrally in the write to a GPR (§7.1, §8.2.1). The most common silent bug.
- **A memory operand is not a `Val`.** It requires computing the effective address (`base + index*scale + disp`) + `Load`. Without the operand lowering layer (§7.1) you can't lift even `mov`.
- **Read-modify-write** (`add [mem], rax`): compute the effective address **once**, use it for Load and Store (§7.1).
- **RIP-relative** is computed relative to the *next* instruction. Use the value from iced, don't compute it by hand (§7 pitfall 2). `ReadReg(Rip)` is forbidden in IR — RIP is stale mid-block; lower to `Imm` (§6.2).
- **FS/GS segment base** for TLS — if the instruction has a segment prefix, add `fs_base`/`gs_base` to the address (§7.1).
- **Flags: `FlagMask`, not `bool`, and not uniform.** `inc`/`dec` keep CF; logic ops force CF=OF=0; shifts update flags **only when the count ≠ 0** (runtime-conditional). A `set_flags: bool` cannot express these — use `FlagMask` (§6.2). iced tells you *which* flags, you encode *how* (§7).
- **Flags as input, not just output.** `adc`/`sbb` consume CF into the arithmetic; `setcc`/`cmovcc`/`rcl`/`rcr` read flags as data. IR needs `Adc`/`Sbb` and `GetCond` — without them you can't lift 128-bit add chains that glibc/compilers emit constantly (§6.2).
- **glibc static "hello world" needs SSE.** `__libc_start_main` uses SSE2 `memcpy`/`strlen` immediately — M2 must use a nolibc/freestanding binary, or M2 secretly requires M8 (§12 M2).
- **`#DE`/`#UD`/`int3` are guest exceptions, not lift errors** — represent them as an `Exit`, decide the shape before lifting `div` (§14).

**Rust aliasing / ownership (hits in M1, gets worse in M7):**
- **Guest RAM is `&Memory` with interior mutability, never `&mut Memory`.** A store writes through `&self`; multiple vcpus share one `Memory` and race like real hardware (ordered by TSO barriers, not `&mut`). Starting with `&mut` forces a full signature rewrite at M7. One deliberate `unsafe` (`UnsafeCell` + manual `Sync`), tightly wrapped (§8).
- **Backend is an injected `Box<dyn Backend>`, not a config enum.** The core can't name the downstream JIT crate — `enum Backend { Interpreter, Jit }` in config is unbuildable. Inject the trait object (§4.1, §8).

**Type contracts (hits at compile time, early):**
- **The backend has no `execute(&IrBlock)`.** Materialization is backend-dependent, execution is uniform over `CachedBlock` (§8). You won't glue the interpreter and the JIT together with a single `execute` on IR.
- **`StepResult`, not `Exit`, from the execution layer.** It distinguishes "flow on" from "trap-out" (§8).
- **The RIP convention on trap-out** must be identical in the interpreter and the JIT (past the instruction for syscall, on the instruction for a memory access), otherwise the backends will diverge (§8).

**JIT / Cranelift (hits in M4):**
- **The block ABI is a decision, not a detail.** `CpuState` via pointer + stable offsets (`#[repr(C)]`), memory via `host_base`, the result encoded in a `u64` (§8.2.1–8.2.2). Open this box before you write codegen.
- **Build the JIT incrementally** against the interpreter-oracle: first an empty block "return Continue with RIP", then `IrOp` one by one (§8.2.3).
- **Inlined memory access is host UB without a safety strategy.** Raw `host_base + guest_addr` reads/writes outside the host buffer on any out-of-range guest address. Emit a bounds+permission check (which also routes Trap/MMIO out) or use guard pages — decide up front (§8.2.3). In `Flat`, address 0 is valid, so a faithful null-`#PF` needs a per-page permission bitmap.
- **MMIO in the JIT** requires a runtime check or a conscious decision that unknown addresses go the slow route (§5.2). The MMIO-read resume is a **pending value consumed by the retried load**, not a write into a dead temp (§5.2).
- **Budget stops ticking under block chaining** — a chained loop never yields `BudgetExhausted`, starving other vcpus. Add a preemption path with chaining (§9.2, M5).

**Multithreading and memory (hits in M7, but prepare in M4):**
- **`CompiledPtr` must be `Send + Sync`** manually, otherwise the cache won't go behind an `Arc` between threads. Make this type in M4, not in M7 (§9.1).
- **The TSO memory model** — `MemConsistency` tiers in codegen and interpretation (`Fast`/`AcqRel`/`FullTso`, §8.2.3). Without barriers, multithreaded programs produce nondeterministic bugs that aren't there single-threaded — the worst kind to chase. Escalate the tier before debugging "impossible" races. Locked ops remain atomic, with the aarch64 LL/SC ordering limitation in §8.2.3; explicit fences apply in every tier. Interpreter accesses and JIT guest-memory helper calls honour the tier on aarch64. Switching tiers flushes the translation cache (barriers are baked into blocks).
- **Ownership of code memory** — the arena lives with the `Vm`, `CompiledPtr` is a borrow; don't free it while the cache points to it (§9.1).

**Guest memory model (hits in M0, then again at large addresses):**
- **`Flat` vs `map()`** — in `Flat`, `map()` only assigns permissions, does not allocate. `map(high_address)` in `Flat` is not an attempt to allocate 128 TB (§4.1). High/sparse addresses → only with `SoftMmu`.

**SMC (hits when you meet a game modifying its own code — can be ignored for a long time):**
- **Consistency of cache↔guest buffer.** A write to a page with translated code invalidates the entry (§10). Ignore it until you actually hit such a case — but then remember it comes from here.

---

## 17. Extensibility points (seams, not a framework)

> **Status (2026-11-04).** The seams below were used: `CpuMode` now has three variants — `Long64`, `Compat32` (32-bit code) and `Real16` (real mode; it always executes through the interpreter, whatever backend is configured — JIT and region compilation of Real16 are not implemented). The text below is the original design rationale and still describes why the seams exist; where it says "today always `Long64`" or calls real mode future work, read it as history.

This library supports **x86-64 long mode only** (see §1). This section defines how to leave clean seams in case someone (or you in a few years) wants to extend it — **without building any framework now**. Goal: don't cement the "always 64" assumption where leaving a parameter is cheap. Nothing here mandates implementing other modes — it only keeps the door open.

### 17.1 Three kinds of extension — two are already supported, one needs seams

- **More instructions (same long mode).** E.g. SSE/AVX, rarer instructions. **Already supported by the architecture** — a new arm in the lift `match` + possibly a new IR operation + its handling in the backends. Zero structural change. The pattern "hit `UnknownInstruction` → add the lift". Needs no seams.
- **A different guest architecture (ARM, MIPS, 6502…).** **Already supported by the IR layer** — someone writes a new front-end (decoder + lift → the same IR), and the whole backend (interpreter, Cranelift, cache, dispatcher) comes for free. A lot of work (new decoder + lift), but clean: it adds a parallel front-end, doesn't touch the existing one. Needs no seams in that sense — just a second lift.
- **A different x86 processor MODE (protected/compatibility 32-bit, theoretically real mode).** **This is the only case that requires seams left in advance**, because it touches semantics *scattered* across the core (decoding, addressing, cache). The rest of this section is about those three seams.

### 17.2 Why 32-bit mode is invasive (so you know what the seam protects)

In long mode: the decoder is always in 64-bit mode, segments (except FS/GS) have base 0, an address is a flat offset. In 32-bit protected mode: **the same bytes decode differently**, and **CS/DS/ES/SS have meaningful bases and limits** — every memory access is `segment.base + offset` with a limit check. On top of that the mode can **change at runtime** (mode transitions), so the same guest address means different code in different modes. These are three concrete touch points — and exactly where we leave seams.

### 17.3 Seam 1 — mode as an explicit field, not the constant `64`

Don't scatter the literal `64` (at `Decoder::new`) or the "segments are always 0" assumption across the code. Introduce an explicit mode type — **today with a single value** — and read it where you decode and compute addresses.

```rust
/// Guest execution mode. TODAY always Long64. The enum exists solely so the
/// decoder and address computation ask for the mode instead of hardcoding "64".
/// Adding a variant later then does NOT require rewriting scattered assumptions.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum CpuMode {
    Long64,
    // Protected32,   // future: decoder in 32-bit mode, segmentation active
    // Real16,        // future: if someone ever wanted a full machine
}
```

Where it lives: logically the mode is a hidden "mode register" of the guest (because in real x86 it can change via a mode transition), so its home is next to the CPU state / lift context. Today it's effectively a constant — but pass it as a value, don't hardcode `64`.

Cost now: negligible (one field, one value, `Decoder::new(mode.bits(), ...)` instead of `Decoder::new(64, ...)`). Payoff later: adding a mode is adding an enum variant, not hunting for `64` in a hundred places.

### 17.4 Seam 2 — mode in the cache key

The cache is keyed by guest address (§9.1). The same address in 32- and 64-bit mode is **different** translated code (because the bytes decode differently). So the key is conceptually an `(address, mode)` pair, not the address alone — even if the second field has a single value today.

```rust
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct BlockKey {
    pub guest_addr: u64,
    pub mode: CpuMode,      // today always Long64; part of the key so modes don't get confused
}
// cache: RwLock<HashMap<BlockKey, CachedBlock>>
```

Cost now: practically zero (wrapping a `u64` in a struct with a single-value field). Payoff later: adding a mode doesn't require redesigning the cache or risking a block from one mode being used in another.

### 17.5 Seam 3 — all memory access through one helper

Segmentation in 32-bit mode touches **every** memory access. The only way its addition isn't surgery on a hundred lifts is for *every* access to go through one address-computation helper — `effective_address` from §7.1. You already have this seam for another reason (RMW, FS/GS); here it's about **keeping the discipline**: no lift computes an address itself.

Today `effective_address` in `Long64` skips segment bases (except FS/GS). In a future `Protected32` the same function would add `segment.base` and check the limit — **a change in one function**, not in every memory-touching instruction. As long as the "address only through the helper" discipline is ironclad, this seam is free.

### 17.6 Directive: leave seams, do NOT build machinery

This is the line you must not cross. The seams above cost pennies and are simply good code (a parameter instead of a constant, a key struct, a single choke-point for the address). **Machinery** is a different thing and is forbidden here until a second implementation exists:

- **DON'T** write `trait AddressingMode` / `trait ExecutionMode` with a single implementation.
- **DON'T** parametrize things that are identical in 32- and 64-bit.
- **DON'T** add "just in case" layers, mode configuration, plugins.
- **DON'T** design an API for `Protected32` that nobody has written.

The reason isn't only about time: **empty abstractions never validated by a second implementation usually turn out wrong when the second one finally arrives** — because you guessed its shape instead of knowing it. Better one solid mode with three clean seams than a multi-mode framework tested with a single mode. When a second mode really arrives, then — with the concrete case in front of you — you'll design the right abstraction; now you'd only make it up.

### 17.7 The other side of the seam: today reject other modes LOUDLY

The seam opens a path to the future; but today another mode must not "almost work", it must be clearly rejected — because silently mis-decoding 32-bit code in 64-bit mode yields garbage, not an error (the same bytes are a different instruction). The loader (outside the core) checks in the header that the file is 64-bit (ELF: `ELFCLASS64` + `EM_X86_64`; SELF analogously) and rejects others with a clear message "only x86-64 long mode supported". The library name communicates the contract, header validation enforces it, the seams leave a back door — three layers, each doing its job.

### 17.8 Embedder diagnostics and compiled-code observation

`environment` is a default-enabled Cargo feature in the core and Cranelift crates.
It exposes explicitly invoked legacy diagnostic APIs; VM/backend construction and
execution never enable them from environment variables. Disabling the feature
compiles out diagnostic environment reads and file access. Flag elision has an
explicit lift/JIT option, and interpreter lockstep accepts a caller-supplied writer
through `lockstep::Trace` and `Vm::set_lockstep_trace`.

`JitBackend::set_compiled_code_observer` receives each successfully compiled block
or region's host range, guest entry RIP, and kind. Calls run synchronously on the
compiling thread (vcpu or tier-up worker) after registration and releasing the
module lock. Re-entry is supported, and callbacks on different threads may overlap.
An unwinding observer panic is reported and caught, that observer is disabled, and
compilation completes without stopping the worker. `perfmap::PerfMap` implements this seam with a
supplied writer. Code is retained for the backend's lifetime; releasing or reusing
translations requires adding a matching released event before that lifecycle
changes. Frame pointers remain mandatory for fault recovery (decision-13).

`JitBackend::set_compile_failure_observer` reports unsuccessful compilations with
`CompileFailure`: guest entry RIP, block/region kind, `CompileFailureReason`, guest
byte count, instruction count, IR operation count and block count. Callbacks run
outside compiler, queue and observer locks with the same re-entry, replacement and
panic-containment contract as the compiled-code observer. Canceled fetch permits
are not compile errors. Background backend panics retain the panic-hook diagnostic
and become `OtherBackendError` completions so the worker can continue.

`MemoryStats.cache.tier_bg_rejected_compile_failed` is split into
`tier_bg_failed_descriptor_limit`, `tier_bg_failed_code_space` (including allocation
and compile-budget failures), and `tier_bg_failed_other_backend` (including backends
that return no classification). These reason counters sum to the aggregate;
invalidated and superseded completions remain separate rejections. Deterministic
failures (`DescriptorLimit` or `OtherBackendError`) mark the unit interpreter-only.
`tier_bg_interpreter_only` counts these lifetime decisions and
`tier_bg_interpreter_only_skipped` counts suppressed tier-up attempts. Unknown
failures do not suppress retries; code-space exhaustion uses the backend's existing
sticky exhaustion state. All failed region source spans remain indexed and tagged,
including uncached successors, so invalidating any source byte releases suppression.
A stale failure cannot suppress a new source snapshot or a newer pending request.

The shared lifter bounds blocks at instruction boundaries and regions before adding
a successor using a conservative count of guest-access descriptors. The limit is
`jit_abi::MAX_GUEST_ACCESS_DESCRIPTORS` (249). This keeps decision-13's exact trap
ownership and host-PC recovery mapping intact; synthetic oversized IR is refused
and reported through the failure seam.

Cranelift libcalls resolve through a closed table of statically linked host
addresses. Unknown names panic with the missing name before dynamic symbol lookup
can run. Indirect Rust helpers are outside that table.

---

- **Input:** an unpacked memory map + entry point. Format parsing OUTSIDE the core.
- **Core:** iced → lift → custom IR → interpreter/Cranelift → cache(guest→host) → dispatcher.
- **Output:** `run()` returns an `Exit` (syscall/mmio/unknown/…); guest state readable/writable between calls.
- **Hot path (RAM):** inline in codegen, never a callback.
- **Rare events (syscall, MMIO):** trap-out via `Exit`.
### 17.9 Real16 coprocessor model and x87 limit checks

`Vcpu::set_fpu_model` selects `FpuModel::Present` (the default, the existing x87
engine) or `FpuModel::None` (no attached coprocessor); `Vcpu::set_fpu_emulation`
supplies MSW.EM until legacy CR0 state exists. With EM set, every ESC opcode
(D8h-DFh, including reserved encodings) raises #NM before any #UD (SDM Vol 3A §2.5,
Table 2-2; 80386 PRM §§11.1.2-11.1.3). With `None` and EM clear, ESC inputs keep the
CPU's operand checks and leave the x87 image unchanged; coprocessor outputs return
`Exit::UnknownInstruction`, because the pinned sources do not specify empty-socket
values.

In both models, an x87 memory operand that does not fit below the Real16 segment
end faults. An SS-relative operand faults with #GP (vector 13), although SDM Vol 2A
§3.3 FLD (Real-Address Mode Exceptions) and 80386 PRM §14.7 item 7 prescribe #SS
(vector 12). This deliberate divergence keeps x87 consistent with the engine's
existing Real16 scalar policy.

**Machine numeric-error routing.** `NumericErrorRoute::{Internal, External}` is
per-vCPU runtime state; Internal remains the default. The VM setter supplies
the construction default, inherited by new vCPUs and preserved by VM forks.
External is supported only for I80286/I80386 Machine execution with
`FpuModel::Present`, for the forms already supported in Direct and Checked
execution. Typed configuration errors reject unsupported profiles/models,
absent coprocessors, and changes while an instruction/device/event continuation
is in flight. Routing is independent of `ExceptionDelivery` and cached IR.

Internal preserves exception 16 at the later recognition instruction (80286 PRM
Part I Appendix B, #MF p. B-9; 80386 PRM §§11.1.2, 11.1.5.3). External returns
`Exit::ExternalNumericError { addr, event: NumericErrorEvent::Asserted }` at the
first committed pending boundary, before the next instruction executes. The
producer retires and contributes one step; `addr == cpu.rip` is its successor.
A pending image installed by the embedder reports at positive run entry with
zero steps. This deterministic notification boundary is engine policy, not
signal-cycle timing. `numeric_error_pending()` reads committed SW.ES (80286/287
PRM Part II Ch. 1, Fig. 1-4 p. 1-10), independently of routing, IF and notification
history. Failed memory transfers publish no event. Numeric clearing re-arms
notification, including a clear and new assertion within one run.

External assertion latches a CPU-side board hold. At a supported waiting
recognition point, a held instruction returns `Waiting` at its unchanged RIP,
with zero progress and unchanged numeric pointers, registers and memory.
`ack_external_numeric_error()` releases that hold idempotently, preserving the
complete numeric image, even if ES is still set. ES alone does not re-latch an
acknowledged episode or produce #MF in External. Clearing ES alone does not
release the board hold. No-wait controls retain handler access (80286 PRM
Appendix B #MF; 80386 PRM §11.1.2). Pending recognition retains its precedence
over the existing 286 coprocessor-overrun policy.

The hold/acknowledgement abstraction is an **engine choice and embedder
requirement**: AT-class boards supply IRQ13 wiring, a board latch and port F0h
acknowledgement. The embedder decodes ports, arbitrates controller requests,
chooses the injected vector and performs EOI. The core neither decodes F0h nor
injects a vector automatically. A stalled instruction permits ordinary host
request, NMI and IRQ boundary arbitration, with the existing IF, shadow and
continuation rules. Assertions/waits are not deliveries and add no delivery-log
record; actual board interrupt entry records the selected vector as `Irq`.
Internal InGuest still records Fault/16; Outward does not record a delivery.

The appended CpuState route and external state preserve notification history
and the hold across cloned checkpoints. Direct edits are reconciled at positive
entry without rewriting SW/CW. Zero-budget calls change neither notification nor
hold. `reset_execution_state` preserves the hold; optional x87 initialization
clears the source and re-arms notification without acknowledging the board.
A whole-machine checkpoint additionally needs the embedder's board/PIC state,
queued events and the existing Vcpu continuation state. CPU restore does not
rewind the delivery log; use its explicit take/clear API.

**Scope limitation:** shared lifting folds some waiting control forms into their
no-wait twins, so those folded forms do not acquire a distinct recognition point.
Separating them belongs to xj-srh.34. Explicit WAIT instructions that the Machine
decoder retains still recognize the hold; protected 286 refusals and the early
8087-era pending-error refusal remain unchanged. No 486 model exists. A future
486 must derive routing from CR0.NE (NE=1 internal, NE=0 external), retain reset
NE=0 and use the explicit IGNNE# input and PRM control/non-control recognition
rules (Intel486 PRM §§4.1.3, 16.2.1.2, 25.1). `set_ignne` currently returns a
typed unsupported-model error; the 386 CR0 mask is unchanged.


### 17.10 CPU-model catalogue and Machine profile

Decision-23 §§1–4 and Migration M1 add `CpuModel::{I80286, I80386}` and a static
`ModelProfile` catalogue. `Vm::set_cpu_model` selects `ExecutionProfile::Machine`
before vCPU creation. Catalogue entries own the ISA/generation projection,
named quirks, system-register family, physical-address width, instruction-length
limit, available FPU pairings, optional timing table and stable semantics identity.
The existing generation-based execution checks remain in place; no lowering,
interpretation, address translation or timing changes in this migration step.
Timing entries are absent, and FPU pairings currently accept `Present` and `None`.
Later model and FPU variants extend non-exhaustive enums.

`CpuState::i386_reset(fpu_model)` supplies an explicit I80386 reset image;
`reset_execution_state` still abandons execution state without installing
registers. The image has EFLAGS=2, EIP=FFF0h, zero data selectors,
IDTR=0:03FFh and CR0's PG/TS/EM/MP/PE clear (80386 PRM §10.1
pp. 10-1–10-2, Figs. 10-1/10-2). DH=03h is documented; DL=03h is
**ASSUMED**, as are zero undefined registers/bits and treating `Present`
as an 80387 for CR0.ET. x87 keeps the engine's FNINIT image. CS=F000h
with hidden base FFFF0000h follows SDM Vol 3A §12.1.4 p. 12-5 (the
pinned PRM §10.1 prints CS=000h). Near transfers retain this base; the
first CS reload, including a far transfer or interrupt, replaces it with
selector×16 (PRM §10.2.3 p. 10-3; SDM §12.1.4 p. 12-5). Ordinary
FFFFh limits are **DERIVED** from PRM §14.1 p. 14-1; real-cache attribute
labels and retention of big-real limits keep the existing engine contract.

The embedder can use `Vm::with_backend_host_ram` with
`VmConfig::reserved(1 << 32)` and a demand-backed 4-GiB host virtual
reservation (`HostRam`, `guest_base=0`). It maps only the required regions:
for example, `vm.map(0xffff0000, 0x10000, Prot::RWX, RegionKind::Ram)`
and `write_bytes` install the high firmware copy, alongside low firmware
and RAM. This reserves virtual addresses, without allocating 4 GiB of
physical RAM; `Vm::with_backend` instead allocates the whole backing.
The ordinary fetch path applies the board A20 policy (decision-22): when
A20 is disabled, the embedder must also map/copy the firmware at
FFEF0000h..FFEFFFFFh, because only physical bit 20 is cleared (SDM
Vol 3A §11.7.13.4 p. 11-33); the reset fetch then uses FFEFFFF0h.
Applying that board gate at reset is **ASSUMED**. A missing window reports
the actual physical fetch fault and retains the linear address as evidence.

Decision-23 C1 makes `isa` an explicit encoding set, with a per-entry absent
encoding policy: #UD, a defined canonical alias, or typed refusal through
`Exit::UnknownInstruction`. I80286 and I80386 use #UD for absent reserved
encodings. Machine decoders and the lifter read this policy; Application and the
Real16 alias retain their existing decoder. Existing system helpers still use
`CpuGeneration` as a compatibility projection. The catalogue's AC/ID fixed-zero
attribute selects the existing early-processor flag helper shape at lift time;
POPF, IRET, interrupt images and task flags retain their fixed-zero behavior.
There is no per-op model branch. PUSH-SP, count-mask and real-FLAGS quirks
select the supported helper shape at lift time; each model supplies the shapes
its policies require.

Three `I80286` values from the 80286 PRM (210498-005) reach execution through
the catalogue (xj-50w.11); the other entries keep theirs, and their identities
are unchanged:
- **Address width.** The physical-bus gate forms addresses at the entry's
  `physical_bits`: a 24-bit address on `I80286`, so a part past FFFFFFh wraps
  to the first megabyte and a transfer splits there (§6.4 p. 6-6; App. C
  hardware item 2 p. C-3; App. D item 2 p. D-1). Data, descriptor-table,
  task-state, coprocessor and fetch transfers all pass this gate. `I80386`
  keeps 32 bits (80386 PRM §14.7 item 18 p. 14-7) and the Real16 alias its
  frozen 32-bit formation. A Direct context never forms an address past
  10FFEFh, so only Checked contexts can reach the wrap.
- **LOCK.** `ModelQuirks::lock_data_movement`: LOCK with MOV, MOVS, INS and
  OUTS raises no #UD ("LOCK should only be used with the XCHG, MOV, MOVS,
  INS, and OUTS instructions", App. C item 11 p. C-2; App. B LOCK p. B-68
  gives no real-mode exception). A locked MOV takes the engine's atomic path:
  its store is one `IrOp::AtomicRmw` exchange whose old value is discarded,
  which on a device is one `MmioWrite` and no read (§5.2); its load is one
  read, which the memory layer performs as one host access for an aligned
  operand, and on a device one `MmioRead`; a register-only MOV transfers
  nothing. A locked store keeps the rule of every
  locked operand: it lies in one region (§5.2, Data accesses across adjacent
  regions; xj-ir6.12), so a locked MOV whose operand spans two adjacent
  regions (RAM and RAM, or RAM and ROM) stops with `UnmappedMemory` before any
  byte is written, where the unprefixed MOV completes. An operand split by the
  A20 gate or at FFFFFFh takes the locked-operand fallback and completes. A
  locked MOVS, INS or OUTS (repeated or not) would hold the bus across its
  transfers, which the engine cannot do, so it returns `UnknownInstruction`
  with the instruction's bytes before anything commits, in Direct and Checked
  contexts alike, as the twenty-bit models' locked strings do (xj-50w.12).
  Every other LOCK form keeps the strict decoder's disposition: locked
  read-modify-write forms execute atomically as before, and LOCK on any other
  register or string form stays #UD. That #UD is the engine's rule (the
  maintainer's, as for xj-50w.12), not a manual value: App. B LOCK p. B-68
  lists no real-mode LOCK exception and the #UD list of p. B-12 has no LOCK
  case. `I80386` keeps #UD for these forms (80386 PRM §14.7 item 9 p. 14-6).
- **Coprocessor operand overrun.** `ModelQuirks::coprocessor_overrun_vector`
  is 9 on `I80286`. With an attached coprocessor, an operand larger than one
  word whose first byte passes the segment check and whose end does not raises
  interrupt 9, with no error code and the saved CS:IP on the instruction's
  first byte, prefixes included (App. B #MP p. B-9; §5.2 Table 5-2 p. 5-6 and
  p. 5-7; App. C Table C-1 and item 3, p. C-1). A one-word operand, or one
  whose start lies outside the limit, keeps the CPU's exception 13. The start
  test is on the operand's first byte, from Part II App. B item 7 p. B-2
  ("Interrupt 13 will occur if the starting address of a numeric operand
  falls outside a segment's size"). An operand of more than one word that
  starts at the limit, so that its first word straddles it, therefore raises
  9: its starting address is inside and its later words are outside (B-9).
  READ: no page states the straddling first word explicitly, and Table C-1
  gives 13 for "a word operation at offset 0FFFFH", which the engine applies
  to one-word operands only. A pending numeric error that the instruction
  tests raises interrupt 16 first: B-9 tests ERROR "at the beginning of most
  floating point instructions", while interrupt 9 "occurs during the
  execution of the numeric instruction by the 80287". Nothing
  of the operand is transferred and the coprocessor image is unchanged.
  ASSUMED, an engine choice the maintainer accepted: the manual describes the
  interrupt as an asynchronous demand of the coprocessor's data channel,
  raised "any time after the processor extension is started", in a task that
  need not be the one that executed the ESC (§9.6.3 pp. 9-10–11), and Table 2-4
  p. 2-26 answers "No" for its return address where Table 5-2 p. 5-6 and the
  p. 5-7 prose put it on the instruction; the engine delivers it as a fault of
  the ESC instruction. With `FpuModel::None` the CPU's whole-operand check
  stays (decision-25).
  Coprocessor instructions in protected mode remain a typed refusal on the
  Machine path, so the interrupt arises in real mode. `I80386` keeps 12/13
  (80386 PRM §14.6 Table 14-1 p. 14-5).

The instruction-length limit is the entry's `max_insn_len` in both segment
classes (xj-50w.13). A Direct block stops before an instruction longer than
the limit and lifts it from the bytes a fetch would hold, so an eleven-byte
`I80286` instruction raises #GP at its first byte in Direct as in Checked
(App. C item 10 p. C-2; App. B #GP p. B-9; decision-23, "The 80286 over-length
fault").

`I80286` executes LOADALL, 0F 05 (xj-oog; decision-23 amendment, LOADALL).
The 80286 PRM omits it; its source is Intel's "Undocumented iAPX 286 Test
Instruction", `oracles/intel/i286-loadall.pdf` (PROVENANCE.md §1), cited by
the page numbers printed on it. The document contradicts itself on the
opcode. p. 2 encodes LOADALL as 00001111 00000101, 0F 05, and pp. 1-3 and the
p. 15 erratum describe an instruction that loads the machine state in real
mode too. p. 13 calls it "opcode 0F04H", says that 0F05H in real mode or at
CPL 0 stops the 80286 until RESET, and ends by saying that 0F05H "may be
executed at any time in real address mode". The engine follows p. 2's bit
encoding and pp. 1-3 (DERIVED: the p. 13 paragraph contradicts itself, and no
pinned oracle has a 0F xx case). 0F 04 stays #UD (App. B p. B-12), a recorded
divergence from p. 13's exception 13 at CPL > 0. 0F 05 stays #UD on `I80386`
and SYSCALL in every Application mode; only the `I80286` identity moves.

In real mode, or in protected mode at CPL 0, LOADALL reads the 102-byte table
at physical 000800H as 51 words through the A20 gate and loads the MSW, TR,
FLAGS, IP, LDTR, the DS, SS, CS and ES selectors, the eight word registers,
the ES, CS, SS and DS descriptor caches, GDTR, the LDT cache, IDTR and the
TSS cache (p. 2). The bytes at 800h-805h and 808h-815h are read and ignored.
At CPL 1-3 it raises #GP(0) before reading (p. 1). PE stays set: LOADALL
"cannot switch the 80286 from protected mode to real mode" (p. 2; DERIVED).
A cache entry's present bit is a valid bit: a clear one marks the cache
`unusable`, and a reference through it raises #GP(0) in protected mode
(p. 3) and, in real mode, interrupt 13 without an error code, as the engine's
other real-mode segment faults (App. B MOV p. B-74). In protected mode the CS
DPL becomes the CPL (p. 3). The LDT cache's valid bit alone says whether an
LDT exists, so a valid LDT cache under a null LDTR selector serves LDT
references (pp. 2-3, with LLDT's "marked invalid", App. B p. B-66; DERIVED);
the 80386 keeps its null-selector test. A loaded cache, its base above 1 MiB
included, addresses memory until its register is loaded again. On the 80286
a real-mode load also resets the limit to FFFFh (p. 3; §5.1 p. 5-2) and the
rights to the engine's real-mode labels, 93h for DS, ES and SS and 9Bh for
CS, so a real-mode far transfer meets the new limit. 9Bh is ASSUMED: neither
source gives it, and after LMSW enters protected mode, before CS is loaded
again, the engine raises #GP for a CS-override write that p. 3's "a writable
segment" lets succeed.

The table policy. LOADALL checks nothing (pp. 1, 3), but p. 4 requires that
SS be valid, writable data (item 1), that CS be execute-only code,
execute/read code or writable expand-up data (item 2), and, "for proper
protected mode operation", that the CS and SS DPLs be equal (item 3). One
rule covers the document's wording. A table that breaks a requirement is the
typed refusal `I286LoadallTable`, with nothing loaded, because the document
gives no result for it. Advice is not checked, and the values load as given:
the DS and ES DPL "should be 3" (p. 3; p. 4 item 4), and GDTR and IDTR byte 3
"should be zeroes" (p. 3); that byte is ignored, as LGDT and LIDT ignore the
last byte of their operand (App. B p. B-65; DERIVED). p. 2's protected-mode
MSW "must have a one in bit position 0" states its own outcome, so it is not
checked either. Beyond the document, the engine refuses what it cannot
execute as the 80286 would: in protected mode a data CS, since it fetches only
through code, and a CS DPL of 1-3, which stays refused although CPL 1-3
now executes (xj-srh.9, decision O12); in
real mode an expand-down SS, DS or ES, since real-mode access checks every
limit as expand-up; and a valid DS or ES with a system type. An invalid CS or
SS stays refused (`UnusableCodeOrStack`). A conforming CS has one of item 2's
code capabilities and loads in either mode. Real-mode access checks no
rights, where the 80286 checks the loaded rights "in either operating mode"
(p. 1): a real-mode CS of any code form and a read-only or code DS or ES
load, and a real-mode write through such a cache, or a read through
execute-only code, succeeds where the 80286 raises exception 13. The LDT and
TSS caches load unchecked, as p. 3 says.

What the document leaves open, and the engine's choice, each ASSUMED. The 51
words are read in ascending order and every read precedes the first change,
so a fault or a device stop restarts LOADALL with nothing loaded (p. 2 gives
the count, not the order). FLAGS loads by the POPF and IRET rule: IOPL and NT
load only in protected mode (App. B IRET pp. B-51–52; §10.1 p. 10-1). The MSW
loads as LMSW does, bits 3:0 (App. B LMSW p. B-67). No interrupt or NMI hold
follows LOADALL, even when it loads SS, and LOADALL does not end NMI
blocking. A TF set before LOADALL traps after it, with the loaded CS:IP
saved; a TF that LOADALL loads traps after the next instruction, as after
IRET or a task switch (§9.7.1 p. 9-14). The model is a later stepping: the
A1/B1 erratum, a real-mode LOADALL that may enter protected mode (p. 15), is
not modelled. p. 2 gives both "195 clocks" and "190 clocks with no wait
states"; neither is modelled, with the rest of the timing model. Word
registers keep their bits above 15 as any 16-bit write does.

**Privilege levels on `I80286` and `I80386`** (xj-srh.9, first landing).
Protected contexts run at CPL 0-3; paging stays `LegacyEntryFacility`.
Sources: 80286 PRM 210498-005 ("286 PRM"), 80386 PRM 230985-001 ("386 PRM"),
SDM 325462-092US; printed page numbers. `interp/legacy/privilege.rs` holds the
helpers. Every fault of a transition precedes its commit point: the helpers
read the TSS, descriptors and frame first and probe the new stack, then set
the accessed bits of the new CS and SS (a descriptor on a device is refused as
`LegacyInstruction` before either changes), write the frame, and only then
the registers, so a device stop during the frame writes leaves `CpuState`
unchanged and the instruction or event replays.
- **Inward delivery.** An interrupt, trap or exception through a gate to a
  non-conforming handler with DPL < CPL takes SS:(E)SP for the target ring
  from the current TSS (SP_n at 2+4n and SS_n at 4+4n in a 286 TSS, 286 PRM
  §8.2 Fig. 8-1 p. 8-2; ESP_n at 4+8n and SS_n at 8+8n in a 386 TSS, 386 PRM
  §7.1 Fig. 7-1 p. 7-2) and pushes old SS, old (E)SP, FLAGS, CS, IP and the
  error code at the gate's width (286 PRM App. B INT p. B-49, §9.4 Fig. 9-4
  pp. 9-4-5; 386 PRM §17 INT pp. 17-70-71; SDM Vol 2A INT pp. 3-472-474). A
  field beyond the TSS limit, or an unusable TR, is #TS(TR + EXT) (SDM
  p. 3-472; DERIVED, the PRMs are silent). New-SS null #TS(EXT); index, RPL,
  DPL and type #TS(SS + EXT); not present #SS(SS + EXT); the gate offset
  beyond the new CS limit #GP(EXT), as on the same-level path.
- **Outward RETF, RETF n and IRET.** A popped CS.RPL above CPL checks the top
  8+n or 10 bytes (16+n or 20 with a 32-bit operand), #SS(0); the return CS
  (null #GP(0); index, type and DPL #GP(CS); not present #NP(CS)); the return
  SS (null #GP(0); index, RPL, type and DPL #GP(SS); not present #SS(SS));
  then the IP against the new CS limit, #GP(0) (286 PRM App. B RET
  pp. B-94-95, IRET pp. B-51-52; 386 PRM §17 RET pp. 17-142-144, IRET
  pp. 17-76-78, §6.3.4.2 Table 6-3 pp. 6-16-17; SDM Vol 2B RET pp. 4-571-574,
  Vol 2A IRET pp. 3-493-497). ESP := the popped (E)SP, a 16-bit pop
  zero-extended (SDM IRET pp. 3-493-494; RET p. 4-573, DERIVED). ES and DS,
  and FS and GS on the 386, become null when their cache is data or
  non-conforming code with DPL < the new CPL; the test reads the cache
  only, so a null selector over a cache that passes it (left by real mode or
  LOADALL) is kept as it is (SDM RET p. 4-574, IRET p. 3-495).
- **FLAGS at CPL > 0.** POPF and IRET load IOPL only at CPL 0 and IF only
  when CPL <= IOPL, from the CPL and IOPL before the instruction, with no
  fault (286 PRM App. B POPF p. B-86, IRET p. B-51, §10.1 p. 10-1; 386 PRM
  §8.3.1 p. 8-5, §17 POPF p. 17-130; SDM Vol 2B POPF pp. 4-407-408). A task
  switch loads the whole image, as before.
- **CPL and IOPL checks** run at run time, before any operand transfer or
  port exit, in the protected regime only (`LegacyOp::Privileged`). HLT,
  LGDT, LIDT, LLDT, LTR, LMSW, CLTS and MOV to or from CRn, DRn and TRn need
  CPL 0; CLI, STI, IN, OUT, INS and OUTS need CPL <= IOPL; else #GP(0) (286
  PRM App. B pp. B-30, B-31, B-42, B-45, B-47, B-65, B-66, B-67, B-72, B-81,
  B-82, B-106; 386 PRM §6.3.5.1 p. 6-17, §8.3.1 pp. 8-4-5; SDM Vol 3A §6.9
  p. 6-23). On `I80286` any instruction with a LOCK prefix needs CPL <= IOPL
  too, else #GP(0) (286 PRM App. B LOCK p. B-68, §3.11.2 p. 3-29); a locked
  MOVS, INS or OUTS that passes keeps its typed handoff (`UnknownInstruction`).
  The 386 has no such rule: its LOCK page lists #UD only (386 PRM §17 LOCK
  pp. 17-100-101). On `I80386` an I/O instruction with CPL > IOPL consults
  the I/O permission bitmap of a 386 TSS (386 PRM §8.3.2 p. 8-5), which is
  not modelled: the typed refusal `LegacyIoPermissionBitmap`, with nothing
  done (xj-srh.10). With a 286 TSS in TR the I/O permission function returns
  FALSE (386 PRM §17.2.2.5 p. 17-15; a 16-bit TSS has no I/O map, SDM Vol 3A
  §10.6 p. 10-18), so the access is #GP(0), as it is with a TR that holds no
  TSS (ASSUMED). REP INS and REP OUTS with a zero count (CX, or ECX with a
  32-bit address size) execute nothing and make no test (SDM Vol 2B REP
  Operation p. 4-564; the maintainer's decision, 2026-11-18). A reserved
  control register, and a LOCK prefix on a form the decode set does not
  lock, keep their #UD ahead of the check (ASSUMED).
- **Task switches** into a task whose CS.RPL is 1-3 run at that CPL; the
  existing segment loads check the incoming CS, SS and data DPLs against it
  (286 PRM App. B p. B-13; 386 PRM Table 7-1 p. 7-9).
- **Call gates** (xj-srh.9, second landing). A far CALL or JMP whose
  selector names a call gate in the GDT or the LDT, type 4 on both
  generations or type 12 on `I80386`, transfers through it; the gate type
  sets the offset width, the push width and the parameter unit, words for
  type 4 and doublewords for type 12, with the count from byte 4 bits 4:0
  (SDM Vol 3A §6.8.3 p. 6-14; 386 PRM §16.4.2.1 p. 16-6). Type 12 on
  `I80286` is #GP(selector) (286 PRM App. B CALL p. B-25, JMP p. B-58;
  DERIVED). The checks, in order (286 PRM App. B CALL pp. B-24-25, JMP
  p. B-57; 386 PRM §17 CALL pp. 17-39-41, JMP pp. 17-85-86; SDM Vol 2A CALL
  pp. 3-131-134; SDM Vol 3A Table 6-1 p. 6-16): gate DPL < CPL or < RPL
  #GP(gate); gate not present #NP(gate); target CS null #GP(0); index
  #GP(CS); not code, or DPL > CPL, #GP(CS); for JMP a non-conforming DPL !=
  CPL #GP(CS), so a JMP never changes privilege; CS not present #NP(CS).
  A CALL to non-conforming code with DPL < CPL takes the TSS stack of that
  ring with the inward checks above (null SS #TS(0)), needs room for the
  parameters plus 8 bytes (16 through type 12), then checks the offset
  against the new CS limit, #GP(0), then reads every parameter from the old
  SS:(E)SP before the first write, a parameter beyond the old SS limit
  being #SS(0); it pushes old SS, old (E)SP, the parameters in their order,
  CS and (E)IP and sets CPL to the CS DPL (SDM Vol 3A §6.8.5 p. 6-18,
  Fig. 6-13 p. 6-19). Every other gate CALL stays at CPL with no stack
  switch: room for 4 bytes (8 through type 12) #SS(0), the offset #GP(0),
  then CS:(E)IP pushed on the current stack. CS.RPL is the CPL after every
  gate transfer. A replay after a device stop in the frame copies the
  parameters it first read (the retry ledger, as for the far pointer). The
  ledger holds 64 entries: before the first parameter read, an inward CALL
  counts the entries it holds and those still to come (the far-pointer keys,
  two per kept descriptor, one per packed parameter group, one per device
  piece of a parameter read or frame write), and a count above 64 is the
  typed refusal `LegacyInstruction` with `CpuState` unchanged and no
  parameter or frame transfer made. With the tables in RAM and both stacks
  on devices, that is from 24 words through a type-4 gate or 21 doublewords
  through a type-12 gate
  (`a_call_gate_frame_beyond_the_retry_ledger_is_refused_before_any_device_transfer`).
  The descriptor and TSS reads that precede the count make at most 24
  entries. An interrupt or exception delivered inward starts with an empty
  ledger and makes at most six frame writes (12 bus pieces) and 22 other
  entries, so it cannot reach the bound. **Not modelled:** the
  TSS's SS_n:(E)SP_n is read again on a replay. A frame that overwrites
  those fields before a later slot stops on a device makes the replay build
  its frame from the new values, or fault; this holds for inward delivery
  and for an inward call-gate CALL alike (the maintainer's decision,
  2026-11-18; the PRMs call the fields "strictly read only values", 286 PRM
  §7.5.1.4 p. 7-20, 386 PRM §6.3.4.1 p. 6-14).
  Task gates and TSSs named by CALL and JMP switch tasks as before
  (decision-24); RETF and RETF n from the called procedure are the outward
  return above.
- **Still refused:** a VM image, paging, the 386 TSS T bit, LOADALL with a
  CS DPL of 1-3, the I/O bitmap of a 386 TSS and an inward call-gate CALL
  whose frame copy exceeds the retry ledger. `LegacyPrivilegeTransition`
  and `LegacyGateOrTaskTransfer` are no longer produced.

The source rows that disagree, and the engine's choice (decisions O1-O14):
- **Inward room (O2).** 286 PRM p. B-49 and 386 PRM p. 17-71 count 10 or 20
  bytes with #SS(0); 286 PRM Table 9-1 p. 9-6 ("5 words (or 6 if error
  code)", SS selector + EXT) and SDM p. 3-473 (10/12 or 20/24 bytes,
  #SS(NewSS + EXT)). The engine: #SS(new SS + EXT), counting the error code,
  on both generations.
- **Null new SS on an interrupt (O5b).** 386 PRM p. 17-70 gives #GP(EXT)
  (page image checked); 386 CALL p. 17-40, 286 PRM p. B-49 and SDM p. 3-473
  give #TS. The engine: #TS(EXT).
- **Inward gate offset.** 286 PRM p. B-49 gives #GP(0); the engine keeps the
  SDM's #GP(EXT) (p. 3-474) of the landed same-level path.
- **Return SS not present (O3).** 386 PRM pp. 17-77 and 17-144 give #NP(SS);
  386 Table 6-3 p. 6-17 and §9.8.12 p. 9-16, 286 PRM pp. B-52 and B-95 and
  SDM RET p. 4-573 give #SS(SS); SDM IRET p. 3-497 lists #SS(0). The engine:
  #SS(SS selector) on both generations.
- **Return SS selector, type and privilege (O4).** 286 PRM Table 7-4 p. 7-22
  lists SF; pp. B-52, B-95, 386 PRM and SDM give #GP. The engine: #GP(SS).
- **Conforming return CS on IRET (O5a).** 286 PRM p. B-52 and 386 PRM
  p. 17-77 print "DPL must be > CPL"; both RET pages and SDM IRET p. 3-497
  say DPL <= RPL. The engine: DPL <= RPL.
- **Data-segment nulling (O6).** The PRM prose nulls a segment whose DPL is
  "greater" than the new CPL (286 PRM §7.5.2 p. 7-21, page image checked;
  386 PRM §6.3.4.2 p. 6-16); the PRM pseudocode re-validates each register
  against its table and readability (286 pp. B-52, B-95; 386 pp. 17-78,
  17-144); SDM RET p. 4-574, IRET p. 3-495 and Vol 3A §6.8.6 p. 6-20 null
  from the hidden cache when DPL < CPL. The engine: the SDM rule, from the
  cache alone, whatever the selector; the re-validation is not modelled (it
  differs only if the table changed after the load). The pseudocode's test,
  "DPL ≥ CPL, or DPL ≥ RPL", read literally with RPL as the register's own
  selector RPL, never nulls a register that a protected-mode load checked:
  every register that passed the load rule DPL ≥ MAX(CPL, RPL) (286 PRM
  App. B MOV pp. B-73-74) satisfies the second clause. A register loaded
  without that check (by LOADALL, in real mode before PE was set, or as the
  embedder's initial state) can fail it.
- **RET n on the outer stack (O9).** The PRM pseudocode (pp. B-95, 17-144)
  loads SS:SP with no second adjustment; the prose (286 p. 7-21, 386 p. 6-16)
  and SDM p. 4-574 add n. The engine adds n, unchecked.
- **Interrupt CS presence order (O8).** 286 p. B-49 and 386 p. 17-70 test
  presence before the DPL dispatch; the engine keeps the SDM order (p. 3-472)
  that landed with xj-srh.6.
- **CPL or IOPL check against a memory operand (O10).** No source orders the
  two; the check comes first (ASSUMED), as LOADALL's does. This holds for
  LLDT and LTR as well, whose helper also checks CPL after the read.
- **VM = 1 in an IRET image at CPL > 0.** 386 PRM §17 IRET p. 17-74 goes to
  STACK-RETURN-TO-V86 whenever the image has VM = 1; SDM Vol 2A IRET
  p. 3-492 requires "tempEFLAGS(VM) = 1 and CPL = 0" and otherwise takes the
  protected-mode return, ignoring VM. The engine keeps the typed refusal
  `InvalidLegacyContext(Virtual8086Unsupported)` at any CPL (the
  maintainer's decision, 2026-11-18), as for a VM image in a task.
- **ESP after a 16-bit outward return.** The SDM pops tempESP with "16-bit
  pop; clear upper bits" and assigns ESP := tempESP (IRET pp. 3-493-494;
  RET p. 4-573), so the engine zero-extends SP into ESP. Processors that
  keep the upper bits of the old ESP after such a return are not modelled:
  no source pinned in `oracles/` describes that retention.
- **Inter-level CALL room (O1).** 286 PRM p. B-25 and p. 7-20 ("an error
  code of 0") give #SS(0); 386 PRM p. 17-40 and §6.3.4.1 p. 6-15 give
  #SS(0), its §9.8.12 p. 9-16 a selector for "overflow of the new stack
  during an interlevel CALL", and SDM p. 3-132 #SS(NewSS). The engine:
  #SS(0) on `I80286`, #SS(new SS) on `I80386`, whatever the gate type.
- **Call-gate target CS presence (O7).** 286 p. B-24 and 386 pp. 17-39-40
  omit the check; 286 Table 7-3 p. 7-18 and SDM p. 3-131 make it, #NP(CS).
  The engine checks it after the type and DPL tests, in the SDM order; 286
  Table 7-3 lists presence before the DPL test (and that test for
  non-conforming code only), and its order is not taken. 286 §7.5.1.1
  p. 7-18's "If EPL > CPL" should read "> DPL".
- **Parameter copy (O11).** No source orders the parameter reads against the
  offset check or the writes. The engine reads every parameter after the
  offset check and before the first write (ASSUMED), so a new stack that
  overlaps the old one copies the caller's values; a read beyond the old SS
  limit is #SS(0). The reads go in push order, highest address first
  (ASSUMED; visible to a device only).
- **Inward JMP through a gate.** 286 PRM §7.5.1.3 p. 7-19 says a JMP
  through a call gate that would change privilege causes a general
  protection fault "with an error code identifying the gate"; 286 PRM App. B
  JMP p. B-57, 386 PRM p. 17-85 and SDM JMP p. 3-509 give #GP(CS selector).
  The engine: #GP(CS selector), the pseudocode's and the SDM's.
- Recorded: 286 p. 7-20 calls an SS DPL mismatch a "task stack fault" and
  386 §6.3.4.1 p. 6-14 a "stack exception", where p. B-25 and the SDM give
  #TS; the PRM pseudocode's "OperandSize" for the gate push width reads as
  the gate type (386 PRM §16.4.2.1 p. 16-6).
- 286 PRM §3.11.1 p. 3-28 reads "CPL ≥ 0" for the IF rule in its OCR text;
  pp. B-86 and §10.1 p. 10-1 state the rule, and the engine cites those.

Retention (O13): a ring change interns a new context stamp and counts against
decision-20's 16 retained contexts. An INT and IRET round trip from ring 3,
and a call-gate CALL and RETF round trip, each retain two contexts and relift
nothing after the first trip
(`ring_trips_retain_two_contexts_and_relift_nothing`).

Machine decode selects the historical meanings of redundant prefixes: F3 90
is NOP (SDM Vol 2B §4.3 PAUSE p. 4-226); F3 0F BC/BD are BSF/BSR
without TZCNT/LZCNT (Vol 2B §4.3 TZCNT p. 4-730; Vol 2A §3.3 LZCNT
p. 3-578); F3 0F 09 is WBINVD on the 486 (Intel486 PRM §26 REP
p. 26-246). Application decode and generated code are unchanged.

Intel486 PRM §26 REP pp. 26-244–246 explicitly ignores REP/REPNE before
non-string instructions except LOOP, for which proper operation is not
guaranteed. The 8086 UM §2.7 pp. 2-42–43, 286 PRM App. B REP pp. B-92–93
and NEC V20/V30 UM §12 p. 12-15 describe string uses only; ignoring these
prefixes on ordinary non-strings is **ASSUMED** for those models. The 386
PRM §17 REP p. 17-141 specifies #UD on non-strings: for F3 90 the engine
follows the SDM's explicit backward-compatibility statement, as it does for
F3 BSF/BSR. F2 90 is an **ASSUMED** NOP compatibility case. Other 386
non-string REP forms raise #UD by default as specified by its PRM.

For the public `CpuModel::I80386` entry only,
`Vm::set_i386_ignore_rep_non_string(true)` selects the named
`ModelQuirks::i386_ignore_rep_non_string` compatibility option. It ignores
REP/REPNE on non-strings (including F3 C3 / F2 C3 RET and F3 40 INC),
executing the underlying instruction once in real and protected mode. This
behavior is **ASSUMED**: an embedder reports REP RET working on real 386
parts, but the 386 PRM §17 REP p. 17-141 specifies #UD, while SDM Vol 2A
§2.1.1 p. 2-2 reserves non-string repeat prefixes and warns of unpredictable
behavior. The SDM does not establish ignored prefixes as a 386 hardware fact.
F3 90 / F2 90 and F3 BSF/BSR keep the compatibility meanings above in both
settings. ESC, NEC extensions, string-repeat combinations and LOCK retain
their separate policies.

Select the option after `set_cpu_model` and before creating any vCPU;
`false` restores the default, and selecting a CPU model again resets the
option. Calls after any vCPU creation panic, including after those vCPUs
are dropped, just like CPU model selection. Both values return the typed
`I386RepQuirkError` on other models, Application profiles and the deprecated
legacy selector. New vCPUs inherit the selection; there is no per-vCPU
setter because decode policy is fixed with the VM's CPU model. Forks preserve
it. The enabled setting selects a static 386 profile variant and hashes its
named quirk into a distinct `semantics_id`; the VM and vCPUs carry that
identity in their existing `LegacyCpuModel`, so checked code-cache keys and
context stamps distinguish the settings. Direct translations remain scoped
to the immutable VM model. Default catalogue identities, other models,
Application profiles, CpuState and jit_abi are unchanged.

The witnesses live in `legacy_redundant_prefix`; the ignored
`legacy_census_386::redundant_prefix_census` reports the one-byte/0F CPU
forms under REP/REPNE, segment, operand-size and address-size prefixes.

Decision-23 C2 adds `I8088` and `I8086`. Both select the 8086 instruction set,
`system = None`, twenty physical address bits, no instruction-length limit,
`None`/`Present` FPU policies and no timing table. The 8086 Family User's Manual
(9800722-03) §4.2 Tables 4-12/4-13, pp. 4-22–36 defines the CPU encodings;
later additions, MOV CS, the 0F map and unused forms outside decision-23's
accepted compatibility amendment return `UnknownInstruction`, without an
invalid-opcode interrupt. That amendment permits 82h as 80h and F6h/F7h /1 as
TEST /0 (SDM Vol 3B §25.15 pp. 25-5–6, corroborated by the "alias" entries in
`oracles/ss8088/v2/metadata.json`); D0–D3 /6 remains refused because the metadata
marks it "undocumented". The amendment also permits ordinary 83h /1,/4,/6
(OR/AND/XOR r/m16 with sign-extended imm8): SDM Vol 2A §3.3 AND p. 3-60,
Vol 2B §4.3 OR p. 4-163, Vol 2D §6.1 XOR p. 6-40; the same corpus metadata
marks these forms "normal". AAM/AAD's
second byte is fixed to 0Ah by Table 4-12 p. 4-24. The decoder folds redundant
prefixes before using iced's bounded representation, while retaining the full
instruction length, real next IP, source spans and final-prefix restart IP.

These entries have a permanent Direct context; `CpuState::legacy` is inert.
Their dispatcher bypasses context validation, stamps and the retained cache.
Multi-instruction blocks use the shared cache and the same interpreter walk.
Blocks end after any instruction that crosses an IP or physical-bus wrap,
including a prefix run longer than 64 KiB, so a changed following instruction
is refetched under the shared self-modifying-code contract.
Every operand byte wraps its offset inside the segment, then its formed address
at 1 MiB; fetch and its cache key use the sixteen-bit IP offset, IP wraps
to zero and A20 settings have no effect (UM §2.3 p. 2-12,
§2.8 p. 2-68; 286 PRM App. C item 10 p. C-2). Lowering copies the FLAGS-image
constants and divide return/range policy into Machine operations and maps full
CL counts to equivalent counts for the shared arithmetic operations. New
`LegacyOp` forms carry those constants and wrapped transfers; Application IR,
CpuState and jit_abi layouts are unchanged. An interrupt taken inside a
repeated string saves the last prefix only (UM §2.7 p. 2-42); before its
first element completes, the frame names the first prefix. Budget and
exit-request yields keep the complete instruction as the owner; the dispatcher
retains the owner/restart pair once an element has completed and applies the
restart only at delivery, so an abandoned request resumes the whole instruction.
The pair is keyed on the instruction at CS:IP and on the inputs the string's
next element reads, at their read widths, as the U3 ENTER identity contract
keys an attempt (xj-50w.15): CX and DF; SI and the source segment for MOVS,
LODS and CMPS; DI and ES for MOVS, STOS, CMPS and SCAS; AL or AX for STOS and
SCAS (UM §2.7 Table 2-13 p. 2-40, pp. 2-42–43; ZF "does not need to be
initialized", p. 2-42). A host change of the instruction's bytes there, or
an edit of one of those inputs, between runs starts a new operation at that
address, which an interrupt names by its first prefix; an edit of an unread
register, or a write of the same bytes, keeps the pair. Only the host edits
between runs, so when a run returns with a pair pending the string's inputs,
the registers and the bytes (prefixes and opcode) its block was lifted from
are recorded, after pending code writes are applied to the cache. A string
no longer cached, or one longer than fifteen bytes, is not recorded and drops
the pair. The next run with a budget (a zero-budget call is no attempt
boundary) drops the pair unless those bytes still are in guest memory at
CS:IP and the inputs are unchanged. A code write whose notice comes after the
run returned therefore still drops it. The entry does not consult the
translation cache, so an eviction keeps the pair.
The NEC entries' three-prefix restart and REPC/REPNC share the rule.

**LOCK (xj-50w.12).** LOCK makes the processor "assert its bus LOCK signal
while the following instruction executes. LOCK does not affect any flags"
(UM §2.7 p. 2-48). The signal "remains active throughout execution of the
instruction that follows the LOCK prefix", and for a locked repeated string
"for the duration of the block operation"; it "provides information only"
(§2.5 pp. 2-17–18). §4.2 p. 4-14 adds that "greater latencies occur when a
'long' instruction is locked". No page restricts the instructions LOCK may
precede, and NEC UM p. 12-173 says the same of BUSLOCK, which p. 2-2 also
describes before HALT. The entries execute a locked form in one of two ways:
- a read-modify-write with an atomic operation (ADD, OR, AND, SUB, XOR, INC,
  DEC, NEG, NOT and XCHG with a memory destination, and the NEC CLR1, SET1 and
  NOT1 on memory, one AND, OR or XOR of the bit mask with no flag change,
  pp. 12-85–98, as a locked BTR, BTS or BTC on the 386 path) goes through the
  engine's atomic path, as `IrOp::AtomicRmw` does on the I80286 and I80386
  Machine paths;
- a form that transfers no data (register operands, LEA, flag and control
  instructions without a memory operand, HLT, WAIT, a register ESC or FPO2)
  runs as the unprefixed form: the bus lock has nothing to hold.

Every other locked form has no atomic path and returns `UnknownInstruction`
before it commits: a repeated string (REP, REPE, REPNE, REPC, REPNC), a single
string, ADC and SBB, shifts and rotates, MUL and DIV of memory, a memory read
or write without a read-modify-write (MOV, CMP, TEST, LDS, LES, XLAT, a memory
ESC or FPO2, the NEC TEST1), stack and frame transfers (PUSH, POP, CALL, RET,
INT, INTO, IRET, PUSHF, POPF and the NEC PUSH R, POP R, PREPARE, DISPOSE,
CHKIND), port transfers (IN, OUT, INM, OUTM), the NEC ROL4/ROR4 (which also
write AL) and BCD strings on memory. The lift decides per form, so INTO is
refused whatever OF is: with OF set it reads the interrupt pointer and pushes
as INT does (UM §2.7 pp. 2-46–47). A locked operand is one host atomic of its
width (`Memory::atomic_rmw`), the only data transfer of the twenty-bit walk
wider than one byte on RAM. A device operand inside one Trap region is one read,
then one write, of its bus transfers (§5.2, xj-ir6.33, xj-50w.20): a word
transfer each at an even address on the I8086 and NecV30, byte transfers
otherwise; a word over a device byte and a RAM byte is performed by pieces, its
RAM byte read once (§5.2, Locked operands partly in RAM; xj-ir6.34), a word
over two Trap regions is a typed refusal, and an operand over two adjacent RAM
regions keeps the containment miss of every locked operand (§5.2, Data accesses
across adjacent regions). A word at an odd address
takes the same atomic path: it is atomic inside one aligned 16-byte granule,
and a word whose first byte is at 15 mod 16 crosses the granule and has the
crossing operand's policy, a plain read and write, with the single-processor
result but not atomic against another vCPU (§8.2.3). A word whose bytes wrap
their segment (offset FFFFh) or the bus (FFFFFh) starts there and has no single
host address: as an A20-split operand on the 286 and 386, it is one read, then
one write, of its byte transfers (it starts at an odd address, so the 16-bit bus
makes it two byte cycles too). This policy (xj-0w3.8) replaced a typed
refusal of every odd word. It changes execution only, not the lifted
instruction, so no identity changes with it. Interrupts and the single-step trap see a locked
instruction as one instruction (UM §2.6 p. 2-24: prefixes are "part of" the
instruction they prefix). The four twenty-bit identities change with the LOCK
admission rule of xj-50w.12 (the early entries' prefix revision word); the
I80386 identity does not, and
the I80286 identity changes only with its own quirk fields (xj-50w.11). The witnesses are in `legacy_i8086` and `legacy_nec_v20_extensions`.

**Data-bus width (xj-50w.20).** Each entry records its external data bus,
`data_bus_bits`: 8 for the I8088 and NecV20, 16 for the I8086, NecV30 and
I80286, 32 for the I80386 (UM §2.3 p. 2-15; NEC UM §4 pp. 4-2–4-3; 80286 PRM
§2.5.1 p. 2-23; 80386 PRM §8.1.1 p. 8-1). The twenty-bit lift copies it into
every lowered transfer; external interrupt and exception delivery, INM's
completion, CALLN and RETEM read it from the vCPU's entry. It sets the width of
device transfers as §5.2 describes. RAM transfers do not read it, and the
I80286 and I80386 paths do not read it. Only a value that
departs from the canonical one (8 for the early entries, 16 for the 286, 32
for the 386) feeds the identity, so the I8086 and NecV30 identities change and
no other does. The witnesses are in `legacy_bus_width`, `mmio_rmw`, `legacy_i8086`
and `legacy_nec_v20_extensions`.
Port I/O does not follow the data bus: IN, OUT and the port side of the V20/V30
INM and OUTM are one `PortIo` exit per element, of the element's size,
on every model. That is an engine contract (ASSUMED), not the bus: the 8088
moves a word to a port "eight bits at a time in two bus cycles", and the 8086
moves a word at an even port in one bus cycle (UM §2.4 p. 2-16); the uPD70108
connects only 8-bit I/O devices, and the uPD70116 needs two bus cycles for a
word at an odd address (NEC UM §4 p. 4-3). The embedder that models a port of
another width splits or joins the cycles itself.
The cold event boundary reads the catalogue's priority policy: an external
interrupt enters its handler before the single-step trap (UM §2.6 pp. 2-26–28).
STI and IRET "reenable interrupts only after the end of the following
instruction" (UM §2.6 p. 2-24): either one delays external interrupts through
its successor when it sets IF from clear, and neither delays when IF was
already set. The lifted flag restore carries this policy; POPF does not.
When POPF or IRET loads TF, the following instruction is the first one
trapped (UM §2.6 p. 2-28).
Wrapped strings report acknowledged bus addresses and REP restart registers,
with conservative evidence of earlier RAM effects, matching ordinary strings.
A RAM-side fault that answered no device byte leaves no retry open.
The selected catalogue entry reaches lowering and delivery directly; the
generation projection serves the system models only. The deprecated selector
keeps refusing the I8086 generation at run entry.

The undefined status flags in UM §2.7 pp. 2-35–40 are not guest guarantees.
The deterministic engine policy preserves OF/SF/ZF/PF for AAA/AAS; clears OF
for DAA/DAS; preserves AF/CF/OF for AAM/AAD; preserves AF/PF/SF/ZF for
MUL/IMUL; preserves all six status flags for DIV/IDIV; clears AF for
AND/OR/XOR/TEST; and preserves AF for shifts. For a multi-bit shift or rotate,
undefined OF uses the shared engine's single-bit formula at the equivalent
count; a zero equivalent RCL/RCR count preserves it. LAHF's undefined image
bits 5, 3 and 1 are 0, 0 and 1 (UM p. 2-33). These are engine choices;
hardware-corpus comparison belongs to C3. AAM/AAD accept only second byte
0Ah, which Table 4-12 p. 4-24 explicitly prints; other immediates are absent.

For these models, `None` executes the CPU's ESC word read and discards the data,
including for coprocessor output encodings; register ESCs have no CPU data effect
(UM §2.7 p. 2-48; iAPX 86/88,186/188 PRM §6.3 p. 6-8). `Present` retains the
existing generation-neutral x87 engine; it makes no 8087 pairing claim. CPU
dummy-read acknowledgements and coprocessor operand acknowledgements remain
distinct through byte-granular MMIO retries. The adapter retains memory error
identity and compacts large-image acknowledgements inside the fixed retry ledger.
`set_fpu_emulation` requests a typed ESC handoff, with no fabricated interrupt 7.
Pending numeric-error delivery also remains a typed refusal, with no processor
interrupt 16. Generation-faithful `I8087` pairing is deferred to decision-23 §4.
FBLD/FBSTP remain `UnknownInstruction` with `Present` on I8086, I8088 and
NEC V20/V30. This is an engine availability policy: the pinned October 1979
8086 Family User's Manual §1.2 p. 1-4 describes future coprocessor hooks and
does not define the 8087 numeric instructions. FFREEP/FSTPNCE remain refused
on these models as well. The witnesses are in `legacy_x87_bcd`.
On I80286, packed-decimal transfers and alternate x87 register forms retain
the existing protected-mode coprocessor refusal; their real-mode execution
uses the 287 definitions (287 PRM Part II Ch. 2, Data Transfer Instructions,
pp. 2-3–4, App. A Table A-2 pp. A-2–5). This regime restriction is engine
policy, witnessed under both exception-delivery options in `legacy_i286_quirks`.
WAIT completes with an empty socket under the engine's ready-TEST policy.

UM Appendix B, "The 8088 Compared to the 8086", pp. B-31–32 states that
instruction results are identical and identifies the bus/prefetch/timing
differences. The entries therefore share ISA and quirks and have separate
catalogue identities; their cycle tables remain absent. The witness binary is
`legacy_i8086`; hardware-corpus differential testing remains decision-23 C3.

Decision-23 C4a adds `NecV20` and `NecV30` (NEC uPD70108/uPD70116 User's
Manual, Stock No. 500350, October 1986: "UM"). They run on the twenty-bit
Direct path of the 8086 entries with their own decode set and quirk values:
`system = None`, twenty physical address bits, no instruction-length limit,
`None`/`Present` FPU policies, no timing table. UM §1 p. 1-1 and §3 p. 3-1
separate the two by external bus width and prefetch queue (4 and 6 bytes); §12
gives one instruction set and per-processor clocks only, so the entries share
ISA and quirks and differ in identity.

The decode set is the 8086 set plus the 80186-class forms UM §12 lists for
native mode: PUSH imm8/imm16 (p. 12-149), PUSH R/POP R (pp. 12-148, 12-152),
MUL reg16,r/m16,imm8/imm16 (pp. 12-54/55), shift and rotate by imm8
(pp. 12-104–141), INM/OUTM (pp. 12-28/29), CHKIND (p. 12-170, memory only),
PREPARE/DISPOSE (pp. 12-152/153) and BRKEM imm8 (0F FF imm8, p. 12-169). The
compatibility amendment's aliases apply as on the 8088 (82h, F6h/F7h /1,
83h /1,/4,/6); shift /6, FFh /7, D6h, 63h, F1h and every other unlisted form
return `UnknownInstruction`. The native-mode extensions are decision-23 C4b,
below. REP (F3h) also
repeats INM/OUTM (p. 12-15); a segment prefix on INM, which p. 12-28 prohibits,
is refused, and so are PUSH SP and POP SP: the operation lists of pp. 12-147 and
12-150 do not say which SP value is stored or left. PREPARE uses the whole imm8
as its level; no modulus is stated. CHKIND raises BRK 5 with the next
instruction's PC (§3 p. 3-1); a bound pair on which a signed and an unsigned
comparison disagree is refused (`NecIndexComparison`), since the page does not
say which applies.

**Native-mode extensions (decision-23 C4b).** The twenty-bit fetch decodes
them itself, because the iced tables describe other instructions at these
bytes; the I8088/I8086 byte table still refuses 0Fh and 64h-67h. Each is one
step and one retired instruction (a repeat: one step per iteration), lowered
out of line from the shared 8086 lowering. Flags marked U on a page are kept,
as C2 keeps undefined flags: an engine choice, not a hardware claim.

| Encoding | Instruction | Forms | Flags | UM |
|---|---|---|---|---|
| 0F 10/11 /0 | TEST1 r/m8, r/m16, CL | reg, mem | Z = !bit; V, CY 0; S, AC, P U | pp. 12-81–83 |
| 0F 18/19 /0 ib | TEST1 r/m8,imm3; r/m16,imm4 | reg, mem | as above | pp. 12-83–85 |
| 0F 12/13, 1A/1B /0 | CLR1 r/m, CL or imm | reg, mem | none | pp. 12-90–93 |
| 0F 14/15, 1C/1D /0 | SET1 r/m, CL or imm | reg, mem | none | pp. 12-95–98 |
| 0F 16/17, 1E/1F /0 | NOT1 r/m, CL or imm | reg, mem | none | pp. 12-85–89 |
| 0F 28 /0, 0F 2A /0 | ROL4, ROR4 r/m8 | reg, mem | none | pp. 12-45–46 |
| 0F 31 11 r2 r1, 0F 39 11 000 r ib | INS reg8,reg8; reg8,imm4 | reg | all U | pp. 12-21–22 |
| 0F 33 11 r2 r1, 0F 3B 11 000 r ib | EXT reg8,reg8; reg8,imm4 | reg | all U | pp. 12-23–24 |
| 0F 20, 0F 22, 0F 26 | ADD4S, SUB4S, CMP4S | DS1:IY, seg:IX, CL digits | Z, CY; V, S, AC, P U | pp. 12-42–44 |
| 65h, 64h | REPC, REPNC | CMPBK, CMPM (CMPS, SCAS) | none | p. 12-14 |
| 66h/67h /r | FPO2 fp-op; fp-op,mem | reg, mem | none | pp. 12-174–175 |

The bit number is CL or the immediate, of which bits 2:0 (byte) or 3:0
(word) are used. ROL4/ROR4 rotate the operand's two digits through AL bits
3:0; AL bits 7:4 are "not assured", which the engine treats like a U flag and
keeps. INS stores the low `length` + 1 bits of AW at bit `offset` of DS1:IY,
EXT loads them from DS0:IX into AW and zeros the rest; the field may run into
the next word, which is accessed only then ("Transfers: 2 or 4", "1 or 2").
Afterwards the offset register holds (offset + length + 1) mod 16 and IY (IX)
advances by two when the field reached the word end. The pages' worked
examples show 3 + 5 bits -> offset 8 with IY unchanged and 8 + 13 bits ->
offset 5 with IY + 2. A field ending exactly at bit 16 is derived, not shown:
the registers are "updated to point to the next bit field" and the offset
register holds only 0-15, so the next field is bit 0 of the next word. The BCD
strings run from the low byte up; IX, IY and CL are unchanged, the source may
take a segment override and the destination is DS1 ("[DS1-spec:]dst-string",
pp. 12-42–44). "Transfers: 3n" gives no order of the reads and stores, so an
overlap that would show the order is refused (below). Disjoint strings,
identical ones (IX = IY in one segment) and an overlap in which every shared
byte is read as source no later than its step as destination give the same
result byte by byte or with all reads first, and run; CMP4S stores nothing.
REPC repeats while CY = 1
and REPNC while CY = 0, with CW tested first, as REPE/REPNE test Z; an
interrupted one keeps up to three prefixes, as other block instructions do.
FPO2 takes the ESC address and prefix policy: with `FpuModel::None` the CPU
computes the address and performs one word read, with no other effect. The
width is the page's: FPO2 fp-op,mem gives Clocks 15 (V20), 15 (V30, odd
address) and 11 (V30, even address) (p. 12-175, the same for FPO1 on
p. 12-174), and the manual prints an odd/even difference only for word
transfers (TEST1 mem16 p. 12-83: 16/16/12; TEST1 mem8 p. 12-82: 12).

Typed refusals, each before any store or device transfer (`NecBcdDigit` is
found after RAM reads of both strings):
- `UnknownInstruction`: 0Fh second bytes other than the listed ones; ModRM
  bits 5:3 not 000 in the bit and nibble forms and the INS/EXT immediate
  forms; INS/EXT with a memory operand or an imm4 above 15; ROL4/ROR4 AL (AL
  would be both the rotated byte and the digit source, and the page gives no
  result for that); EXT AL/AH; any repeat prefix on the 0Fh forms and FPO2;
  any prefix on INS/EXT and BRKEM; REPC/REPNC on anything but CMPS/SCAS, or
  with another repeat prefix, or in a prefix run longer than 15 bytes (the
  fetch keeps 15 bytes of evidence to check the run; REP/REPE/REPNE accept any
  count); FPO2 with `Present` (the engine defines no 66h/67h coprocessor
  instruction) or with the EM input set. EXT AL/AH: p. 12-23 orders the update
  ("AW <- 16-bit field ... After the transfer, the IX register and the 8-bit
  register specified by the first operand are updated"), so the page would
  give AL/AH the new offset, but the hardware corpus documentation
  (`oracles/ssv20/README.md`, "Using AL or AH as the first operand to EXT is
  invalid") marks the form invalid; the derived result is not claimed. INS
  AL/AH differs: AW is only read, before the transfer, and AL/AH is written
  once, after it, so no value is in question.
- `NecBitFieldOperand`: INS/EXT with nonzero bits 7:4 in the offset or length
  register, which the pages require to be 0 "for correct operation".
- `NecBcdStringLength`: CL = 0, 255 or odd (1-254 digits are given; an odd
  count leaves the upper digit, Z and CY undefined). `NecBcdStringOverlap`:
  ADD4S/SUB4S whose destination byte i is, at its physical address, the
  source byte j of a later step (j > i), so that a store before that read and
  a read before every store give different results. `NecBcdDigit`: a digit
  above 9 in either string. `NecBcdStringDevice`: an operand byte in a trapped
  region. This is an engine limit, not a gap in the page: the instruction is
  one unit of up to 381 transfers, and a replay after an MMIO stop would read
  RAM bytes it had already stored. An unmapped or unwritable byte faults
  before the first store.

Corpus disagreements (`oracles/ssv20/v1_native/metadata.json` and
`README.md`, harness classifications without captured bytes in the pin; the
manual is followed): the metadata marks Z undefined for TEST1 CL (0F10/0F11,
`"flags": "...szap."`, `"flags-mask": 65387`), where pp. 12-81–83 set it, and
marks no flag undefined for TEST1 imm (0F18/0F19), where pp. 12-83–85 mark S,
AC and P U; the README says "REPC/REPNC prefixes on INS and OUTS act as
regular REP prefixes" (6C/6D), which p. 12-14 does not describe, so the
engine refuses it.

BUSLOCK (F0h, p. 12-173) follows the C2 LOCK rule (xj-50w.12, above): CLR1,
SET1 and NOT1 on memory take the atomic path, a register-only bit or nibble
form and a register FPO2 run, and TEST1, ROL4 and ROR4 on memory, the BCD
strings, BRKEM and REPC/REPNC are typed refusals. The device-memory
BCD refusal and the REPC prefix-run limit above are engine limits.

The other entries' identities are unchanged; the NEC entries hash an
extension revision word. The witness binary is `legacy_nec_v20_extensions`.

Quirk values: the PSW image is MD,1,1,1 above V (§3 p. 3-3) and MD is 1 while
native code runs, so lifted images store F000h; vectors 0-5 are processor
assigned (§6 Table 6-1 p. 6-1); DIV/DIVU push the next PC and the signed
quotient stops at -127/-32767 (pp. 12-56–59). That bound is the pages' prose,
which also gives it in hexadecimal (81H/8001H). Their operation lines read
"0-7FH-1"/"0-7FFFH-1" (-128/-32768) with garbled comparison signs, and are
not taken. Offsets are sixteen-bit values
on a twenty-bit bus (§10 Fig. 10-1 p. 10-1; §4 p. 4-1) and PC is a sixteen-bit
counter (§3 p. 3-1); shift counts are used whole (pp. 12-102, 12-104); an
interrupted block instruction keeps up to three prefixes (§6 p. 6-4,
`rep_restart_prefixes = Some(3)`); INT is held after MOV r/m16,sreg as after
MOV/POP sreg, and not after RETI or POP PSW (§6 p. 6-4); EI holds INT
through the next instruction whatever IE was (§6 p. 6-4; p. 12-172 EI;
`ei_delay_unconditional`), where the other entries keep the rule of SDM Vol 2B
§4.3 STI. A single-step trap
and an acceptable INT at one boundary are refused (`NecStepWithInterrupt`):
Table 6-1 gives priorities, not the sequence.

**8080 emulation mode (UM §8).** The engine keeps PSW.MD and the hidden MD
write-enable latch in `CpuState::legacy.reserved` (bit 0 set while MD = 0,
bit 1 the latch, bit 2 the 8080 standby latch); the C2 entries keep their
twenty-bit state in the same places, and no layout changes. BRKEM pushes PSW,
PS, PC, clears IE and BRK (decision-23 rule R2, ASSUMED from p. 6-1), sets
MD = 0 and the latch, and loads the vector. p. 6-3 gives the general rule
that the single-step break follows the current instruction, but no page
describes how it combines with the switch to 8080 mode, so a BRKEM that begins
with BRK = 1 is refused before it commits (`NecEmulationSingleStep`). RETI and POP PSW load MD
from the stack only while the latch is set (pp. 12-168, 12-151). A POP PSW that
would clear MD there is refused (`NecPswModeWrite`). So is a RETI that would
clear it with BRK = 1 at its start (`NecEmulationSingleStep`), before it
commits: the break after it is not described for emulation mode. BRKEM with the
latch set is refused (`NecNestedEmulation`, p. 8-5).

At a boundary with MD = 0 the decision-16 order is unchanged: budget, host
request, delivery of a queued acceptable IRQ, the readiness stop; then the
model returns `Exit::I8080Mode { view, entry }` with zero steps and nothing in
flight. `I8080View` carries A = AL, the 8080 flags (PSW 7:0), BC = CX,
DE = DX, HL = BX, SP = BP, PC = IP, IE (that the 8080 EI and DI set and
clear it is ASSUMED, contract draft S4), the EI delay (the STI shadow,
verbatim), the standby latch and the code (PS) and data (DS0) segments
(Figs. 8-1/8-2 p. 8-1; Table 12-6 p. 12-3). The embedder runs an 8080 core on
the view and calls `Vcpu::hand_back_i8080(view, how)`. The hand-back writes
AL (not AH), BX, CX, DX, BP, IP, the five 8080 flags, IF, the delay and the
standby latch; it never writes a segment, SP, SI, DI, V, DIR, BRK, MD or the
latch, adds no step and changes nothing when it returns an error:
`NotInI8080Mode`, `SegmentChanged`, `HaltedStop`, `RefusedEscape` (`Refused`
naming EDh) or `InFlight` (an MMIO transfer of a frame access of RETEM, CALLN
or an interrupt taken in 8080 mode is in flight). `how` is:
- `EdEscape`: the 8080 core stopped on EDh without consuming it. At the next
  boundary, after delivery, the engine reads ED again at PS:PC with its own
  fetch (the second read of that byte) and executes RETEM (ED FD) or CALLN
  (ED ED imm8) as one step; any other ED form is `UnknownInstruction`. RETEM
  pops PC, PS and the whole PSW from SS:SP and write-disables MD (p. 12-177,
  whose "When the BRKEM instruction is executed" is taken as a misprint for
  RETEM, ASSUMED); a frame with MD = 0 is refused (`NecRetemToEmulation`).
  CALLN pushes the PSW with MD = 0, PS and the PC after its three bytes
  (ASSUMED), clears IE and BRK (rule R2), sets MD = 1 and loads the vector
  (p. 12-176, whose "SP <- PS - 6" is taken as a misprint for SP - 6,
  ASSUMED). Frames and vectors
  are read from memory when the unit executes.
- `Boundary`: the 8080 core stopped at a boundary. A queued IRQ that
  `irq_ready` accepts is delivered natively: the frame's PSW has MD = 0, entry
  clears IE and BRK, sets MD = 1 and ends standby (§6 p. 6-1, §8 p. 8-4). A
  delivery discards a pending `EdEscape`; RETI returns to the ED.
- `Refused { opcode }`: the 8080 core refused another byte. MD stays 0 and,
  until the next hand-back (or an interrupt taken there), every call at that
  PS:PC returns `UnknownInstruction` with the byte.
A boundary with MD = 0 and BRK = 1 is refused (`NecEmulationSingleStep`). The
embedder must not inject INT while NMI or HOLD is asserted (§6 p. 6-2). A
latched NMI is taken at an MD = 0 boundary before any of the above (see "NMI
input" below). The 8080 core's stop test ORs in
`ExitRequest::is_requested` and `NmiLine::is_raised`, non-consuming,
async-signal-safe loads. A host
request made while the core runs therefore stops it; the request stays pending,
and the next call reports it as `Interrupted` before any delivery. Writes by the 8080 core use
`Vm::write_bytes`, which invalidates overlapping translations. A V20/V30 reset
image writes `legacy.reserved = 0` (MD = 1; §7 p. 7-1). Steps: BRKEM, RETEM,
CALLN and RETI are one each; the exit, the hand-back and delivery are zero;
8080 instructions are counted by the embedder's core. No cycle table exists
yet, and the manual gives no 8080-mode clocks. UM §8 p. 8-1 names the
uPD8080AF instruction set without listing it; that it behaves as the Intel
8080A is an ASSUMPTION of the 8080 core (S16 of the contract draft), not of
this engine. The witness binary is
`legacy_nec_v20`, which also runs all fifteen of the sibling 8080-family
project's contract vectors on the native side, the three NMI cases included.

#### NMI input (xj-50w.19)

Every Machine model has an NMI input. `Vcpu::raise_nmi` (between runs) and
`NmiLine::raise` (from any thread, or from a signal handler on a handle obtained
beforehand) latch one rising edge; edges coalesce until one is accepted. The
latch is bit 33 (`NMI_EDGE`) of the vCPU's dispatch word, set with one lock-free
`fetch_or(Release)` like `ExitRequest::request_exit`; cloning or dropping a
handle is not async-signal-safe. An Application vCPU refuses with
`NmiUnsupported` and `nmi_line()` is `None`: its compiled code exits on any bit
of that word. There is no exit for an NMI, which has no acknowledge cycle
(8086 UM §2.6 p. 2-24; NEC UM p. 2-3, p. 6-2; 286 PRM §9.2 p. 9-3). A running
vCPU observes the edge at its next instruction or REP-iteration boundary.
`NmiLine::is_raised` is the raw latch; `Vcpu::has_pending_nmi` is the model's
view (false on an I80386 inside its NMI handler); `Vcpu::nmi_ready` says whether
a latched NMI would be accepted at this boundary, whatever IF and the STI/EI
shadow: not while blocked (I80286, I80386), after a segment load or store that
holds NMI (see the table), with an IN, INS or MMIO completion (answered or not) or
a task-switch continuation outstanding, or at an NEC 8080-mode boundary with
BRK = 1. Shutdown does not prevent it. HLT returns `Exit::Hlt` as before; the
embedder wakes on `has_pending_nmi() && nmi_ready()`. `irq_ready` is false
while that holds: NMI has priority and INT is not accepted (NEC UM p. 2-2;
8086 UM p. 2-24; 286 PRM p. 9-3). `inject_irq(2)` stays an ordinary maskable
vector (SDM Vol 3A §7.7.1 p. 7-6).

Acceptance is the consuming `fetch_and` of the latch after the model's
predicate holds, so no edge is lost or taken twice; delivery costs no step.
Boundary order (decision-16 §2 extended): budget; host request when no
completion is pending (the latch is kept); the service steps; an owned
continuation; in shutdown, only a latched NMI may proceed, otherwise
`Exit::Shutdown`; at an NEC 8080-mode boundary with BRK = 1 and an NMI latched,
`NecEmulationSingleStep` with the latch kept; the NMI; a queued eligible IRQ;
the readiness stop; the 8080-mode exit; the next unit. After any delivery the
order runs again before the handler's first instruction, so an NMI is taken
on top of a just-delivered frame, and an IRQ on top of an NMI entered through a
trap gate or a task whose image has IF = 1 (286 PRM §9.6 p. 9-9). A post-
instruction single-step trap is delivered from the unit's epilogue: on the
80286/80386 the NMI is taken before the step handler's first instruction; on
the 8086 family the NMI test runs inside that delivery (below).

An accepted NMI owns its delivery through bus handoffs, like an accepted IRQ.
A fault during delivery is delivered serially: NMI is benign (286 PRM §9.6.2
p. 9-10; 386 PRM Table 9-3 p. 9-13), the NMI is lost and blocking stays until
the exception handler's IRET (SDM Vol 3A §7.7.2 p. 7-6). An accepted NMI that
never enters the guest leaves no blocking: a real-address Outward delivery
fault, a nonresumable refusal and a continuation released because CS:IP moved
latch the edge again; a protected Outward delivery fault and an abandoned
(edited-state) continuation consume it. When an NMI's delivery faulted, the
exception delivered serially carries its blocking: if that delivery is released,
abandoned or refused in the same ways, blocking ends and the NMI stays lost.
An NMI taken out of shutdown and latched again also restores the shutdown.
`reset_execution_state` clears the latch, the hold and blocking (that reset
ends blocking is ASSUMED). The blocking state, the hold and the latch live
outside `CpuState`, `jit_abi` and the retained-context stamp; no getter exposes
them yet.

| Rule | I8088/I8086 | NecV20/NecV30 | I80286 | I80386 |
|---|---|---|---|---|
| Vector, frame | 2 through IVT 008H; FLAGS (15:12 set), CS, next IP; IF, TF cleared (UM p. 2-24, Fig. 2-30 p. 2-25, p. 2-26) | 2 through 008H; PSW (MD = 0 in 8080 mode), PS, PC; IE, BRK cleared, MD set, standby ended (UM Table 6-1, p. 6-1, p. 8-4, p. 9-1) | 2; real mode through the IVT (§5.2.2 p. 5-5); protected through IDT entry 2 as an external event: CPL and gate DPL unchecked, EXT on delivery faults, no error code (§9.4 p. 9-4, Table 9-3 p. 9-9) | as the 80286 with 386 gates (Table 9-1 p. 9-2; SDM Table 7-1 p. 7-2) |
| IF/IE | not masked (p. 2-24; CLI p. 2-48) | not masked (p. 2-3; DI p. 12-172) | not masked (§3.11.2 p. 3-28) | not masked (§4.1 p. 4-1) |
| NMI and INTR | NMI first (p. 2-24) | NMI first (p. 2-2) | NMI first (Table 5-1 p. 5-4) | NMI first (Table 9-2 p. 9-4) |
| NMI and the step trap | NMI frame, then the step frame: the step routine returns into the unstepped handler (p. 2-28; AP-67 Fig. 3E3a p. A-29). After INT n, INT 3, a taken INTO or a fault with TF: that frame, then the NMI frame, no step frame (AP-67 p. A-28, Fig. 3E3b p. A-29) | refused, `NecStepWithNmi`, with the trap and the latch kept: Table 6-1 gives priorities, not the sequence | step frame, then the NMI before the step handler (§9.7.1 p. 9-14) | as the 80286 (§12.3.1.4 p. 12-8) |
| NMI during NMI service | nested (p. 2-26; App. B p. B-6); the engine's acceptance is "the start of the NMI procedure" (ASSUMED) | nested, ASSUMED (the manual is silent) | blocked until the next IRET, one edge remembered and serviced after it (§9.2 p. 9-3) | blocked until the next IRET, edges ignored (§9.2.1 p. 9-3) |
| What ends blocking | — | — | any IRET: a nested handler's, a task return; a faulting IRET (ASSUMED); not a refused IRET or a bus handoff before completion | the next IRET; a faulting IRET (SDM §7.7.2 p. 7-6); not a refused IRET |
| Holds | MOV/POP of any segment register, each load (p. 2-24 footnote; held for NMI as for INTR, a choice against App. B p. B-6 and AP-67 p. A-27, which states the rule in its maskable-interrupt section) | segment loads and MOV reg16/mem16, sreg, each (p. 6-4; pp. 12-7, 12-8, 12-151); EI holds INT only (p. 12-172) | MOV SS and POP SS, each load (App. B pp. B-73, B-83); STI holds nothing for NMI (ASSUMED) | MOV SS and POP SS, each load (§9.2.4 pp. 9-3–4); not LSS (SDM §7.8.3 p. 7-9); STI holds nothing for NMI (ASSUMED) |
| HLT | wakes; saved IP after HLT (p. 2-48, p. 2-26) | wakes in both modes (p. 12-171, p. 9-1) | wakes unless blocked (App. B p. B-42) | wakes unless blocked (p. 17-60) |
| Inside REP | between elements; the last prefix (p. 2-42; App. B p. B-6) | between primitives; up to three prefixes (p. 6-4) | between repetitions (§3.7.2 p. 3-22) | between repetitions (§9.2 p. 9-2) |
| Shutdown | — | — | NMI ends it (§9.6.2 p. 9-10, §11.6 p. 11-7) | NMI ends it (SDM §7.15 p. 7-37, second witness) |

An edge raised while blocked that no boundary has seen yet is still latched
when the IRET ends blocking. The 80386 ignores it like the others (§9.2.1
p. 9-3). It also ignores an edge raised between that IRET's retirement and the
end of the IRET's unit: a window of the engine's granularity (ASSUMED). The
80286 keeps such an edge latched, as it would a remembered one, and services
it after the IRET (§9.2 p. 9-3).

Shutdown: the NMI is delivered through the current IVT or IDT and saves the
CS:IP held at shutdown (ASSUMED). If that delivery faults, the CPU re-enters
shutdown still blocked, and only reset ends it (286 PRM p. 9-10, reading
"processing the NMI" as its delivery, ASSUMED). In 8080 mode an NMI is taken
at every boundary with BRK = 1 refused first; the delivery ends the 8080 EI
delay (DERIVED from EI p. 12-172), drops a pending escape or refusal, and the
routine's RETI returns to 8080 mode (`I8080Entry::Reti`, also right after
BRKEM). The NMI hold is rearmed by every segment load as the model manuals say;
INT and the single-step trap keep decision-16 §4. `ModelQuirks::nmi_nesting`
(`NmiNesting`) carries the per-model policy; it is hashed into `semantics_id`
only where it departs from its canonical value, so no identity changes for it.
The NEC segment-register store holds NMI through a new IR operation, so
`NEC_EXTENSION_REVISION` is 2 and only the NecV20/NecV30 identities change.

`semantics_id` is a deterministic content digest of encoding membership and canonical alias pairs,
decoder/absence policy, quirks and system/lift/FPU attributes, plus an explicit
lifting-policy revision for changes to validation or alias rules. Changes to
lifting-relevant content require a new identity; rule changes bump the revision.
Checked cache keys and context stamps include it. Direct translations are scoped
to a VM whose model is immutable before execution; they are never shared between
catalogue revisions. There is no persisted translation cache to carry across a
library rebuild. CpuState and jit_abi layouts are unchanged.


`Vm::set_fpu_model` fixes the construction default (initially `Present`) before
any vCPU exists. New vCPUs inherit it, forks preserve it, and the per-vCPU setter
remains an override. `new_vcpu` validates the VM default against a selected
catalogue entry's FPU pairings.

With `None`, I286 Machine output ESC forms retire after the complete CPU operand
checks, preserving the x87 image. FNSTSW AX, FNSTSW m16 and FNSTCW m16 supply
FFFFh, the all-high board default (I286/287 PRM Part I Appendix A p. A-4;
Part II Appendix B item 10 p. B-2). Memory word stores use ordinary Machine
transfers. Other outputs preserve memory: their absent-extension outcome is
not documented (Part II Ch. 2 p. 2-49, Ch. 3 pp. 3-1–3-3; decision-25).
WAIT completes in real and protected mode unless MP and TS are both set;
the following ESC independently tests EM and TS. I386 retains output refusal because
its PRM §11.1.2 does not define absent output data. The MSW/CR0 exception gates,
input forms, Present and Application paths keep their existing behavior.

`ExecutionProfile::LegacySystem` is a deprecated associated constant for
`Machine`, including in patterns. `Vm::set_legacy_cpu_model` remains a deprecated
compatibility selector preserving the supplied generation and semantics identity.
State storage and its ABI keep their existing names and layout.

`Vm::set_cpu_model` defaults to InGuest exception delivery (decision-23 §5);
`Vm::set_legacy_cpu_model` keeps Outward for compatibility. Either default can be
overridden with `Vm::set_exception_delivery`. `set_cpu_mode(Real16)` still
selects the existing Application Real16 path at M1, with frozen IR snapshots.
Long64/Compat32 lowering, cache keys and generated code are unchanged.

#### Delivery log (xj-ir6.37)

With `ExceptionDelivery::InGuest` the host never sees a delivery. The delivery
log is an opt-in, per-vCPU ring of the last N exceptions and interrupts a
Machine vCPU delivered in the guest, which the embedder reads between runs.
`Vcpu::enable_delivery_log(capacity)` starts an empty ring of 1 to
`DeliveryLog::MAX_CAPACITY` (65536) records; an Application vCPU refuses with
`DeliveryLogError::NotMachine`, a capacity outside that range with
`DeliveryLogError::Capacity`. The ring's storage grows with the records it holds,
up to the capacity; enabling reserves nothing. `delivery_log()` and
`delivery_log_mut()` give the `DeliveryLog`. Its consuming reads are `take()`,
which returns a `DeliveryBatch` of the records (oldest first), `dropped` and
`total`, and `drain_into(&mut Vec<DeliveryRecord>)`, which appends the records to
the caller's buffer, so that a drain after every run reuses one allocation, and
returns the same two counts as a `DeliveryCounts`. `dropped` is the number of
records the full ring overwrote since the previous take or drain, and both reset
it; `total` is every record since the log was enabled, so a batch holds the
deliveries numbered `total - len` up to `total` and batches order themselves.
`iter()`, `len()`, `capacity()`, `dropped()` and `total()` only look. `clear()`
discards the records and the dropped count unread. `DeliveryRecord`,
`DeliveryBatch` and `DeliveryCounts` are `#[non_exhaustive]`: the engine builds
them and may add fields. `disable_delivery_log()` stops recording and returns the
log. `reset_execution_state` keeps it, with its records and counts, as it keeps
the other diagnostic accounting.

The log lives in the `Vcpu`, outside `CpuState`, `jit_abi` and the retained-context
stamp: it is host-side diagnostics that guest execution never reads. It is also
outside any vCPU checkpoint an embedder takes by saving and restoring `CpuState`:
a restore neither rewinds nor clears it, and the log keeps the records of the
abandoned timeline until the embedder takes or clears them. It is off by default.
While it is off, execution, retirement and generated code are unchanged: the
Machine delivery sites test the option before one cold recording call, and the
dispatcher's boundary derivation folds it into its existing cold-epilogue flag, so
the walk of a unit is unchanged. An 80286/80386 software interrupt marks a flag
that fills padding in the per-instruction outcome, which a unit test pins; a
twenty-bit one writes a temporary of its own op. Neither the per-unit progress nor
the walk's step state carries anything for the log, so a unit without a software
interrupt does no work for it. Application profiles, the Real16 alias among them, have no log.

A record is written when a delivery enters its handler, in delivery order, so the
same run gives the same records. `DeliveryRecord` holds:

- `vector`, the vector the delivery used;
- `kind` (`DeliveryKind`), which also says what the saved IP names:
  - `Fault`: a processor exception of the fault or abort class (SDM Vol 3A §7.5),
    #DF included, and BOUND's #BR, a fault that saves BOUND (SDM Vol 3A §7.15,
    Event 5 p. 7-31). The saved IP is the faulting instruction, which `retired`
    does not count. The one exception is the twenty-bit divide error, whose saved
    IP is the one its model saves (`ModelQuirks::divide_saves_next_ip`);
  - `Trap`: the single-step trap. The saved IP is the next instruction: after a
    twenty-bit software interrupt, the first instruction of its handler;
  - `Software`: INT n, INT 3, a taken INTO and INT1 on the 80386 (SDM Vol 3A
    §7.4.2), and on the NEC V20/V30 CHKIND's interrupt 5, BRKEM and the 8080-mode
    CALLN. The saved IP is the next instruction, and `retired` counts the
    interrupting one;
  - `Irq`, a vector queued with `inject_irq`, and `Nmi`. The saved IP is the
    instruction they interrupted, which has not run;
- `saved_cs` and `saved_ip`, the return address the delivery saved: in its frame,
  or in the outgoing TSS for a task gate; a twenty-bit IP is sixteen bits;
- `saved_linear`, the CS base in effect before the delivery plus `saved_ip`, before
  the A20 gate; on the twenty-bit models CS × 16 plus IP, wrapped to 20 bits
  (8086 Family UM §2.8 p. 2-68). The instruction bytes are the embedder's to read
  from its RAM;
- `error_code`, the code the delivery pushed: protected-mode exceptions that have
  one, through a gate or a task gate; `None` for every real-address and twenty-bit
  delivery, which push none (SDM Vol 3B §23.1.4);
- `cpl_before` and `cpl_after`: 3 and 0 for an inward delivery from ring 3; 0
  outside protected mode;
- `retired`, `Vcpu::retired_instructions` when the handler was entered. A software
  interrupt retires by delivering, so its own instruction is counted; a faulting
  instruction is not;
- `regs`, the interrupted code's registers (`DeliveryRegs`), or `None` unless
  the log was enabled with them (below).

Registers (xj-ir6.40). `Vcpu::enable_delivery_log_with(capacity, options)` takes
a `DeliveryLogOptions` (`#[non_exhaustive]`, `Default` records nothing extra;
`DeliveryLogOptions::default().registers(true)` sets the one option there is,
and `records_registers()` reads it); `enable_delivery_log(capacity)` is the same
call with the default options, and `DeliveryLog::options()` reports them. A log
keeps its options for its life: another `enable_delivery_log_with` is the only
way to change them, and it starts an empty ring. With the option set, each record
carries a boxed `DeliveryRegs` (`#[non_exhaustive]`) in `regs`, so that a record
without it is one pointer wider than before, not a whole register set (a test
pins the record at 40 bytes). `DeliveryRegs` holds `eax`, `ebx`, `ecx`, `edx`,
`esi`, `edi`, `ebp` and `esp` as `u32` (the twenty-bit models and the 80286 fill
the low 16 bits; I286 PRM §2.3.1), the selectors `ds`, `es` and `ss`, `fs` and
`gs` as `Some` on the 80386 only (80386 PRM §2.3.2), and `flags`. Every field is
its value when the delivery began: after the raising instruction's own effects
and before any of the delivery's, so before the frame push, a stack or task
switch and the FLAGS changes. For INT n, INT 3 and INTO they are the
instruction's inputs (a function number in AH arrives as `eax`); for a fault,
the state at the fault, the faulting instruction rolled back where the model
rolls it back; for an IRQ or NMI, the state after the last completed
instruction. `esp` with `ss` is the interrupted stack, before the frame: on an
inward delivery the outer stack, not the handler's. On the 80386 `esp` is the
whole register: on a sixteen-bit stack segment (always in real mode) the stack
pointer is its low 16 bits, and the high half is recorded as the register holds
it. `flags` is the FLAGS (EFLAGS) image as PUSHF forms it on the model, so RF and
VM are zero on the 80386 (SDM Vol 2B §4.3 PUSHF/PUSHFD/PUSHFQ, Operation)
whatever their live value, while a fault's frame can hold RF set; on the
twenty-bit models it is the image the frame holds, which an interrupt taken in
NEC 8080 mode saves with MD clear. A delivery that follows another without an
instruction between them (the NMI frame after an exception's, the step trap after
a twenty-bit INT, the exception a faulting delivery raises) records the state the
earlier one left; after a task switch has committed, a fault belongs to the new
task and records the new task's state, and a task switch handed off to the
embedder keeps the registers it began with until it completes. A delivery that
began before the option was enabled, which only an 80286/80386 task switch
handed off across the enable can be, records `regs: None`: no copy was taken.

The copy is taken off the walk. An 80286/80386 delivery begins in the
interpreter's one delivery routine, which, with the option on, copies the
registers into the task-switch state beside the walk from an out-of-line call;
with the option off it tests one flag. A twenty-bit delivery changes only SP,
FLAGS, CS and IP, so the recording call rebuilds the registers from the vCPU
state with SP before the six-byte frame and the FLAGS image the frame holds,
which the twenty-bit interrupt routine returns and the report of a twenty-bit
software interrupt carries in its high bits. Nothing else changes when the
option is off.

A fault during delivery. An attempt whose delivery raises an exception is not
recorded. The exception it raised is delivered next, serially or as #DF with
error code 0 (I286 PRM §9.6.2 p. 9-10; 386 PRM Tables 9-3/9-4 p. 9-13), or in the
new task after a committed task switch, and that delivery is recorded when it
enters its handler, with its own vector and error code and the return address its
frame holds: the interrupted instruction, the INT instruction for a software
interrupt whose delivery faulted, or the new task's CS:IP. A #DF whose delivery
faults enters shutdown: the log ends before it, the run returns `Exit::Shutdown`,
and an NMI that ends shutdown is recorded like any NMI. A delivery handed off to
the embedder (a device IDT entry or stack) is recorded once, when it completes; one
released because CS:IP moved, abandoned or refused is not recorded.

The record of a task-gate delivery keeps the outgoing task's CS:IP, CS base and CPL
through every bus handoff after the switch commits (an error-code push to a device
stack in the new task, for one); only a fault raised after the commit, which is
delivered in the new task, records the new task's.

Not recorded: an exception that `ExceptionDelivery::Outward` returns as an exit,
which the embedder sees as that exit (IRQs, NMIs and software interrupts are still
delivered in the guest and recorded), and a task-gate delivery whose incoming task
has the T bit set, which the engine refuses with `LegacyTaskDebugTrap` after the
switch.

Where each delivery is recorded. Exceptions, IRQs, NMIs and the step trap are
delivered by the dispatcher's event path, which records them after the delivery
call returns. A software interrupt is reported by the walk that takes it, and the
unit's cold epilogue records it. An 80286/80386 INT marks its instruction's outcome
and leaves the vector, the saved CS:IP and the CS base and CPL it read before
delivering in the task-switch state beside the walk; when its task switch is
handed off to the embedder, the switch remembers that it is a software
interrupt's, and the resumed instruction that completes it marks its outcome the
same way. A twenty-bit INT, INT 3, INTO,
CHKIND or BRKEM op has a temporary of its own: the walk writes 0 there when it
reaches the op, and the vector, the CS:IP and the FLAGS image the frame holds once the interrupt is
taken and the frame written; a not-taken INTO or CHKIND leaves 0. A twenty-bit
unit holds up to sixteen instructions and such an interrupt ends it (a debug
assertion checks that the op is the block's only one, in its last instruction), so
the epilogue reads the temporary after a walk that retired the whole block. Neither path infers the
interrupt from the stack pointer or the unit's ops, so the records do not depend on
how a run is sliced. CALLN is recorded by the 8080-mode boundary that decodes it.
