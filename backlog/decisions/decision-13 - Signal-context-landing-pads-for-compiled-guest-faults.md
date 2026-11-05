---
id: decision-13
title: Signal-context landing pads for compiled guest faults
date: '2026-11-05 01:10'
status: proposed
---

## Context

Decision-5 GP-2 used an embedder wrapper around glibc's `__sigsetjmp` to recover
guard faults. Rust has no returns-twice contract. Recovery must follow the
ordinary compiled-function return path, including on Mach-O hosts.

A per-call entry trampoline was measured and rejected: single-block eager JIT
fib32 rose from 67.8 to 124.9 ms, hotloop from 26.3 to 59.0 ms, memcpy from 4.2
to 6.7 ms, and indirect by 19–28%. Removing most restores still left fib32 at
108.8 ms and hotloop at 47.2 ms. The dispatcher calls compiled code on every
chain edge (about 21 million times in fib32). Per-call recovery stores do not fit
that path. Frame-pointer preservation alone measured within noise of direct calls.

## Proposed decision

**Supersede decision-5 GP-2 with compiler-described, one-frame recovery and a
static signal-context landing stub.** GP-1 protection callbacks and GP-3's
immutable codemap remain. This draft awaits maintainer review.

`call_block` calls the compiled function pointer directly. It performs no
recovery stores, entry trampoline, or register snapshots. Each generated block
and region preserves its frame pointer. At compile time the backend reads
Cranelift 0.115.1's finalized buffer `UnwindInst` records. `PushFrameRegs` and
`DefineNewFrame` describe the caller-SP and saved-FP anchors; `SaveReg` describes
each clobbered preserved GPR and, on AArch64, d8–d15 slot relative to FP.
Return-address rules follow Cranelift's own System V CIE and register mapper.
The recipe also records the end of the prologue; sites before it cannot recover.
Missing or unsupported descriptions yield no recipe and answer NOT OURS.
A recipe is accepted only for the ISA's default calling convention, saved FP at
FP+0 and caller SP at FP+16. Tail conventions and alternate frame shapes are
rejected before publication.

The recipe is immutable and retained alongside the exact guest-access trap sites
in the codemap entry. Both become visible with the same Release publication and
Acquire lookup as other function metadata. Recipes and old map roots have the
executable arena's unreclaimed lifetime, including after invalidation or backend
drop. Return addresses are tested against host ranges even when those ranges
have no guest source locations or recovery metadata.

Native AArch64 macOS Cranelift emits signed return addresses. The recipe records
the compiler's selected A/B key and signing state; the classifier authenticates
the saved address with that key and the caller SP before checking its range or
returning. This is required by the measured prologue, without changing codegen's
signing policy (Cranelift `aarch64/abi.rs::select_api_key` and `cranelift-native`;
Arm ARM DDI0487J.a §§C6.2.22–C6.2.23, AUTIA/AUTIB). Alternate SP rules remain
unsupported. Host ABI preservation follows System V AMD64 psABI §3.2.1 and
AAPCS64 release 2024Q3 §§6.1.1–6.1.2 and its “The Frame Pointer” section.

The real-prologue test witnesses PACIBSP (`0xD503237F`) and return_auth=2 on
macOS AArch64; PACIBZ (`0xD503235F`) is rejected by that witness. Other hosts
check the signing settings and emitted prologue. The runtime test first asks
whether PACIB changes a pointer, and exercises classifier authentication only
when it does. In the tested non-arm64e macOS process PACIB/AUTIB are identity,
so **real authentication remains runtime-unverified**. Linux AArch64 also
remains runtime-unverified until a PAuth-enabled process/host runs this path.
An unsigned Linux recipe does not exercise authentication.

## Embedder contract

Each vCPU owns a boxed per-run record. `Vcpu::fault_handle()` returns a Copy,
non-owning handle; `as_usize()` gives an integer for the embedder's per-thread
slot. Moving the vCPU preserves the handle; dropping it invalidates it. Clear
the slot before dropping the vCPU. Never fabricate a handle or use another
thread's active handle. An inactive live handle is safe to reject. The core
neither installs a signal handler nor owns a thread-local slot.

`run()` captures host FP control once, before dispatch or compilation, and
`run_inner` publishes pointers to CpuState and its local MemCtx once. The record
is active for that run. A drop guard deactivates it and clears both ABI pointers
on normal return and Rust unwinding, including a compiler panic. Standalone compiled callers may
use fixed-address `EntryRecord::activate` / `deactivate` around their whole run;
they keep the ABI pointers live and publish the handle themselves.

The embedder installs synchronous SIGSEGV and SIGBUS handling as appropriate.
It calls `jit_abi::fault::classify` with:

- The current thread's live handle, the interrupted host PC, and `si_addr`.
- A `HostFaultKind` derived from the OS fault reason as described below.
- PC, SP and FP copied from the interrupted ucontext.
- x86-64 RBX, R12–R15 in `callee_saved`, and R11 in `scratch`.
- AArch64 x19–x28 in `callee_saved`, x30 in `lr`, x9 in `scratch`, and the low
  64 bits of v8–v15 in `simd`.

`HostFaultKind::{Access, Alignment, Unsupported}` is plain core data. The
embedder owns decoding; ambiguous or unknown reasons fail safe as NOT OURS.

- Linux: SIGSEGV with SEGV_MAPERR/SEGV_ACCERR or SIGBUS with BUS_ADRERR is Access;
  SIGBUS with BUS_ADRALN is Alignment. Other causes, including si_code <= 0,
  must not enter recovery (or are passed as Unsupported).
- macOS AArch64: first exclude user-sent and unsupported signals. Derive the kind
  from the saved exception-state ESR, because protection failures can arrive as
  SIGBUS/BUS_ADRALN too. EC=0x24/0x25 identifies a Data Abort; DFSC=0x21 is
  Alignment, DFSC=0x04..0x07 translation, 0x08..0x0b access-flag and 0x0c..0x0f
  permission faults are Access. Other EC/DFSC combinations are Unsupported
  (Arm ARM DDI0487I.a §D17.2.37, ESR_EL1 Data Abort ISS/DFSC). A mapped PROT_NONE
  load was measured as SIGBUS/BUS_ADRALN with DFSC=0x07; a locked operand crossing
  a 16-byte granule was measured as Alignment and replays successfully.
- macOS x86-64: the test adapter rejects Alignment and ambiguous SIGBUS causes;
  it does not decode the saved trap/error state. This adapter is compile-unverified.
  Ordinary process startup has alignment checking disabled; this is an OS/ABI
  convention, not a processor guarantee. Intel SDM Vol 3A §6.10.5 requires both
  EFLAGS.AC and CR0.AM at CPL=3 to enable checking. Embedders that enable it must
  provide their own verified trap/error decoding before reporting Alignment.

On AArch64 the handler and classifier run under the interrupted FPCR; they must
not depend on its rounding or flush mode. More generally, handlers must not
assume any FP environment (Linux x86-64 starts its handler with default MXCSR).
The landing stub restores host FP control **before control returns to
call_block's caller**, after the signal handler has completed.

OURS requires an active handle, an exact guest transfer site in a compiled
function with a supported recipe, a return address outside **every** compiled
range, and a data address within the run's guest span. Nested compiled calls
are NOT OURS: this mechanism unwinds exactly one outermost compiled frame.
On OURS the classifier writes full-width guest RIP from the trap site and
rebased address, transfer width and access into CpuState/MemCtx. ReadWrite maps
to Write in the fallback exit. It restores the saved GPRs, FP and AArch64 SIMD
low halves using the recipe, retaining interrupted values of unsaved registers.
AArch64 SP becomes the caller's SP and LR its return address; x86-64 SP points
at the saved return address. PC becomes the static landing stub, and scratch
points at the record's saved FP-control value.

**Copy every returned field back into the ucontext**, including scratch and,
on AArch64, LR and the low halves of v8–v15. Preserve the other halves of the
signal's SIMD image. Returning normally from the handler enters the landing
stub, which restores FPCR or MXCSR, places RET_HOST_FAULT in the ABI result
register, and returns exactly as the compiled function would. No Rust frame is
skipped. MXCSR storage/restoration follows Intel SDM Vol 1 §10.2.3, and its
callee-saved control bits follow x86-64 psABI §3.2.1 (pinned in `oracles/psabi`).
FPCR fields FZ[24] and RMode[23:22] follow Arm ARM DDI0487J.a §C5.2.7.

NOT OURS leaves guest state and HostRegisters untouched. Chain to the prior
signal disposition, including default termination. Rust helpers, unsupported
hosts, PCs without exact guest-access sites, missing recipes, nested compiled
frames, and addresses outside the guest span remain NOT OURS. Helper recovery
is separate work (xj-obx.5).

## Dispatcher and signal safety

For an unmapped address, run_inner interprets exactly the faulting guest
instruction. Precise trap state is instruction-entry state (Intel SDM Vol 3A
§7.5); Memory's checks reproduce the interpreter's transfer base, splitting and
access direction, including a straddling access whose `si_addr` is an inner byte.
Alignment faults always replay in the interpreter, including mapped RAM. A
misaligned locked operation currently completes non-atomically there; the
cross-thread atomicity gap is tracked in **xj-0w3.5**. An Alignment fault is
never itself converted to UnmappedMemory; replay can still report an actual
unmapped operand. Tests compare lock add, xchg and lock cmpxchg final state with
the interpreter in both block and region modes and continue to HLT.

For an Access fault in mapped RAM independently protected by the embedder,
replay would fault in Rust. Instead return Exit::UnmappedMemory using rebased `si_addr` and the trap's
access; MemCtx retains the transfer width. This fallback cannot reconstruct a
straddling protected transfer's operand base. Exit::UnmappedMemory currently
exposes address and access, without a size field.

The classifier allocates nothing, locks nothing, calls no libc, uses no TLS,
and performs no stack walk or retries. Work is bounded: immutable AVL lookups,
one exact-site binary search, fixed-size register copies, and compiler-described
stack-slot reads. The existing codemap function/site capacity limits bound the
searches. The actual interrupted compiled stack must remain readable. Handlers
must not reenter execution or mutate guest state through another path; execution
and guest-state access belong exclusively to the owning thread. Keep the VM,
vCPU and record alive throughout classification and landing.

Supported/tested host: macOS AArch64. Linux x86-64/AArch64 adapters are workspace
cross-checked; macOS x86-64 is **compile-unverified**, with Alignment unsupported
by its test adapter. The landing stub is a mangled Rust naked function using
`naked_asm!`; each linked crate instance gets its own symbol on ELF/Mach-O.
A link test builds two crate instances to pin that property. Other targets keep
the direct compiled call and always classify NOT OURS. Test adapters cover Linux
x86-64 greg arrays, Linux AArch64's bounded FPSIMD extension records (Linux v6.12
`arch/arm64/include/uapi/asm/sigcontext.h`), and macOS AArch64 and x86-64 mcontexts.
Core dependencies remain **iced-x86 only**; no dependencies were added elsewhere.

## Validation and cost

Recovery adds **zero instructions or stores on compiled call edges**. Per-run
publication, cleanup and FP-control capture are outside the chain loop. The
following measurements precede the cold-path review fixes; compiled call edges
are unchanged. `cargo run -p x86jit-bench --release -- experiment` was measured first; final
validation built the base revision and change once, then alternated the binaries
for three A/B rounds. Each cell below is the experiment's minimum of three runs,
in milliseconds, **base → change**:

| Round | fib32 | hotloop | memcpy | indirect | hotloop (20M) |
|---|---:|---:|---:|---:|---:|
| 1 | 68.60 → 67.81 | 26.64 → 26.42 | 4.20 → 4.19 | 23.12 → 23.62 | 131.78 → 130.96 |
| 2 | 68.61 → 67.42 | 26.61 → 26.34 | 4.26 → 4.18 | 23.25 → 23.73 | 131.15 → 132.22 |
| 3 | 68.47 → 68.00 | 26.54 → 26.29 | 4.20 → 4.21 | 23.01 → 23.37 | 132.08 → 131.71 |

These eager-JIT pairs differ by roughly −1.7% to +2.2%, without the trampoline
regression. OS background indexing and backup activity affected earlier serial
measurements substantially. The maintainer will repeat measurements on an idle
host before merge; these numbers do not establish a strict zero-latency bound.

`fault_landing_pad.rs` covers block and region load/store, locked RMW, vector,
straddling and retry, RIP above 4 GiB, mapped protected RAM, default signal death,
and host FP-control restoration. Its assembly caller seeds every preserved GPR
and AArch64 d8–d15 with distinct patterns. Both a real busy region and a static
block-equivalent fault through the stub; the latter deliberately clobbers every
saved register. Classifier unit tests pin exact sites, absent recipes, inactive
runs, nested compiled return addresses, guest rebasing and untouched rejections.
A backend unit test generates a real function using many preserved GPRs and all
eight preserved AArch64 SIMD registers, reads its actual frame slots against
independently seeded values, and rejects deliberately shifted recipe offsets.

The reviewed tests also start from a non-default host FP mode, arm a watchdog
in every forked child, and assert protected load/store/RMW access direction.
A lifted region with a test-only CLIF pressure prelude faults after CPUID's Rust
helper call, requires eight preserved SIMD save slots on AArch64, observes that
the interrupted LR differs from the recovered LR, and checks every preserved
caller register. Mutation controls independently remove each restoration.

## Host-source provenance

The AArch64 host sources cited here are **not pinned in `oracles/` yet**; a
separate bead will pin them. This does not add a build-time/manual dependency.
The exact references used for human verification are:

- [AAPCS64 2024Q3 (5 September 2024)](https://github.com/ARM-software/abi-aa/blob/2024Q3/aapcs64/aapcs64.rst),
  §§6.1.1–6.1.2 (preserved registers), “The Frame Pointer”.
- Arm ARM DDI0487J.a (2023), §C5.2.7 (FPCR, FZ and RMode),
  §§C6.2.22–C6.2.23 (AUTIA/AUTIB), and §C6.2 (PACIA/PACIB instruction encodings).
- Arm ARM DDI0487I.a, §D17.2.37, ESR_EL1's EC and Data Abort ISS/DFSC fields;
  [Arm's I.a issue corrections](https://documentation-service.arm.com/static/644c18cbaa78c007af74fa2c)
  identify this register section explicitly.
- [Linux v6.12 arm64 uapi sigcontext.h](https://github.com/torvalds/linux/blob/v6.12/arch/arm64/include/uapi/asm/sigcontext.h),
  `sigcontext.__reserved`, `_aarch64_ctx`, FPSIMD_MAGIC and `fpsimd_context`.
  The adapter bounds its extension scan to that 4096-byte area.

Intel SDM Vol 3A §7.5 supplies guest fault-state precision and §9.1.1 the guest
locked-operation contract; the interpreter's present replay atomicity limitation
is stated above. The Intel manual and x86-64 psABI sources remain the existing
pinned authorities in `oracles/`.
