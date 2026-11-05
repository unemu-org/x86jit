---
id: decision-13
title: Signal-context landing pads for compiled guest faults
date: '2026-11-05 01:10'
status: accepted
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

## Decision

Accepted by the maintainer on 2026-11-05, together with its 2026-11-05 amendment
(opt-in recovery for Rust guest accesses) and the workspace MSRV of Rust 1.88
that the naked landing stub and fixup routines need.


**Supersede decision-5 GP-2 with compiler-described, one-frame recovery and a
static signal-context landing stub.** GP-1 protection callbacks and GP-3's
immutable codemap remain.

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
thread's active handle. An inactive live handle is rejected for compiled PCs but
can classify a fixup routine. The core neither installs a signal handler nor
owns the embedder's thread-local handle slot.

`run()` captures host FP control once, before dispatch or compilation, and
`run_inner` publishes pointers to CpuState and its local MemCtx once. The record
is active for that run. A drop guard deactivates it and clears both ABI pointers
on normal return and Rust unwinding, including a compiler panic. Standalone
compiled callers may use fixed-address `EntryRecord::activate` / `deactivate`
around their whole run; they keep the ABI pointers live and publish the handle
themselves.

The embedder installs synchronous SIGSEGV and SIGBUS handling as appropriate.
It calls `jit_abi::fault::classify` with:

- The current thread's live handle (or zero for Rust fixup routines), interrupted
  host PC, and `si_addr`.
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
- FreeBSD amd64/arm64: SIGSEGV with SEGV_MAPERR=1 or SEGV_ACCERR=2 is Access.
  SIGBUS with compatibility page-fault code UCODE_PAGEFLT=12 is also Access.
  FreeBSD 14.3 `vm_fault_trap` normally reports a mapped protection failure as
  SIGSEGV/SEGV_ACCERR for current FreeBSD binaries; an older or missing ELF ABI
  tag, or `machdep.prot_fault_translation=1`, selects SIGBUS/code 12 instead.
  The value is defined in `sys/vm/vm_fault.c`; `sys/sys/signal.h` has no
  BUS_PAGE_FAULT constant. SIGBUS/BUS_ADRALN=1 is Alignment. BUS_ADRERR=2,
  BUS_OBJERR=3, BUS_OOMERR=100 and all other causes are Unsupported. Explicitly
  allowlist the supported pairs: FreeBSD's user-sent SI_USER/SI_QUEUE/SI_LWP
  codes are positive, so excluding only si_code <= 0 is insufficient.
  The release sources below are unpinned in `oracles/`.
  On FreeBSD arm64 only, Alignment recovery ignores `si_addr`: `align_abort`
  passes the interrupted `tf_elr` to `call_trapsignal`, which sets `ksi_addr`.
  The classifier still requires the active owning handle, exact compiled guest
  trap site, supported recipe and outermost frame. It leaves `fault_addr`
  untouched and the interpreter recomputes the operand from guest RIP on replay.
  Access recovery still validates/rebases `si_addr`; Linux, macOS and FreeBSD
  amd64 require an in-span address for every kind. Fixup routines continue to
  reject Alignment on every host.
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

For compiled PCs, OURS requires an active handle, an exact guest transfer site
in a compiled function with a supported recipe, a return address outside
**every** compiled range, and a data address within the run's guest span.
The FreeBSD arm64 Alignment exception above drops only that address check.
Nested compiled calls are NOT OURS: this mechanism unwinds exactly one outermost
compiled frame. On OURS the classifier writes full-width guest RIP from the trap
site and rebased address, transfer width and access into CpuState/MemCtx.
ReadWrite maps to Write in the fallback exit. It restores the saved GPRs, FP and
AArch64 SIMD low halves using the recipe, retaining interrupted values of
unsaved registers. AArch64 SP becomes the caller's SP and LR its return address;
x86-64 SP points at the saved return address. PC becomes the static landing
stub, and scratch points at the record's saved FP-control value.

**Copy every returned field back into the ucontext**, including scratch and,
on AArch64, LR and the low halves of v8–v15. Preserve the other halves of the
signal's SIMD image. Returning normally from the handler enters the landing
stub, which restores FPCR or MXCSR, places RET_HOST_FAULT in the ABI result
register, and returns exactly as the compiled function would. No Rust frame is
skipped. MXCSR storage/restoration follows Intel SDM Vol 1 §10.2.3, and its
callee-saved control bits follow x86-64 psABI §3.2.1 (pinned in `oracles/psabi`).
FPCR fields FZ[24] and RMode[23:22] follow Arm ARM DDI0487J.a §C5.2.7.

NOT OURS leaves guest state and HostRegisters untouched. Chain to the prior
signal disposition, including default termination. Unsupported hosts, PCs outside
both the exact compiled guest sites and the Rust fixup table, missing compiled
recipes, nested compiled frames, and compiled data addresses outside the guest
span remain NOT OURS. Rust guest transfers recover through the fixup routines
specified in the amendment below; arbitrary Rust helper code remains NOT OURS.

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
the dispatcher uses a region-only check and returns Exit::UnmappedMemory in
both recovery modes, using rebased `si_addr` and the trap's
access; MemCtx retains the transfer width. Compiled split stores can already have
committed earlier host transfers (xj-obx.8); resuming rewrites those same bytes.
This fallback cannot reconstruct a
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
A link test builds two crate instances to pin that property. Hosts outside Linux,
macOS and FreeBSD keep the direct compiled call and always classify NOT OURS.
Test adapters cover Linux x86-64 greg arrays, Linux AArch64's bounded FPSIMD extension records (Linux v6.12
`arch/arm64/include/uapi/asm/sigcontext.h`), and macOS AArch64 and x86-64 mcontexts.
Core dependencies remain **iced-x86 only**; no dependencies were added elsewhere.

FreeBSD amd64 and arm64 use the same mangled naked ELF landing stubs and relative
fixup descriptors as Linux. Their test adapters copy the complete recovery
register set. The amd64 adapter uses `mc_rip`, `mc_rsp`, `mc_rbp`, `mc_rbx`,
`mc_r12`–`mc_r15` and `mc_r11`. The arm64 adapter uses `mc_gpregs.gp_x[]`,
`gp_lr`, `gp_sp`, `gp_elr` and the low halves of `mc_fpregs.fp_q[8..16]`, preserving
the upper halves and unrelated fields. It requires `_MC_FP_VALID`. libc 0.2.186
models `fp_q` as a single u128, truncating its arm64 mcontext; the test adapter
uses a local `repr(C)` signal-image layout from the release header instead.
FreeBSD amd64 execution of `fault_landing_pad` and `rust_memory_faults` remains
pending on an embedder's FreeBSD host; arm64 execution is also pending.

## Validation and cost

Recovery adds **zero instructions or stores on compiled call edges**. Per-run
publication, cleanup and FP-control capture are outside the chain loop. The
compiled-call measurements below compare the base revision and landing-pad
implementation in three interleaved A/B rounds. Each cell is the minimum of
three experiment runs in milliseconds, **base → change**:

| Round | fib32 | hotloop | memcpy | indirect | hotloop (20M) |
|---|---:|---:|---:|---:|---:|
| 1 | 68.60 → 67.81 | 26.64 → 26.42 | 4.20 → 4.19 | 23.12 → 23.62 | 131.78 → 130.96 |
| 2 | 68.61 → 67.42 | 26.61 → 26.34 | 4.26 → 4.18 | 23.25 → 23.73 | 131.15 → 132.22 |
| 3 | 68.47 → 68.00 | 26.54 → 26.29 | 4.20 → 4.21 | 23.01 → 23.37 | 132.08 → 131.71 |

These eager-JIT pairs differ by −1.88% to +2.16%, without the trampoline
regression. OS background load limits these timings; they do not establish a
strict zero-latency bound.

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

Fork-isolated tests also start from a non-default host FP mode, arm a watchdog
in every forked child, and assert protected load/store/RMW access direction.
A lifted region with a test-only CLIF pressure prelude faults after CPUID's Rust
helper call, requires eight preserved SIMD save slots on AArch64, observes that
the interrupted LR differs from the recovered LR, and checks every preserved
caller register. Mutation controls independently remove each restoration.

## Host-source provenance

The FreeBSD and AArch64 host references cited here are **not pinned in `oracles/`**.
No build-time manual dependency is required. The references and their verification
limits are:

- FreeBSD 14.3, tag `release/14.3.0`:
  [sys/sys/signal.h](https://github.com/freebsd/freebsd-src/blob/release/14.3.0/sys/sys/signal.h)
  defines the signal-specific and SI_* values;
  [sys/vm/vm_fault.c](https://github.com/freebsd/freebsd-src/blob/release/14.3.0/sys/vm/vm_fault.c),
  `vm_fault_trap`, defines UCODE_PAGEFLT=12 and protection-fault translation.
  [sys/amd64/amd64/trap.c](https://github.com/freebsd/freebsd-src/blob/release/14.3.0/sys/amd64/amd64/trap.c),
  `trap`/`trap_pfault`, uses BUS_ADRALN for alignment and `vm_fault_trap` for
  page faults.
  [sys/arm64/arm64/trap.c](https://github.com/freebsd/freebsd-src/blob/release/14.3.0/sys/arm64/arm64/trap.c),
  `align_abort`/`call_trapsignal`, supplies the host PC as Alignment's si_addr;
  `data_abort` delegates access-fault classification to `vm_fault_trap`.
- FreeBSD 14.3 signal images:
  [sys/x86/include/ucontext.h](https://github.com/freebsd/freebsd-src/blob/release/14.3.0/sys/x86/include/ucontext.h)
  and [sys/arm64/include/ucontext.h](https://github.com/freebsd/freebsd-src/blob/release/14.3.0/sys/arm64/include/ucontext.h)
  define the recovery fields; the arm64 image contains 32 u128 vectors and
  `_MC_FP_VALID=1`.
  [sys/sys/_ucontext.h](https://github.com/freebsd/freebsd-src/blob/release/14.3.0/sys/sys/_ucontext.h)
  puts mcontext after the signal mask, whose four u32 words are defined by
  [sys/sys/_sigset.h](https://github.com/freebsd/freebsd-src/blob/release/14.3.0/sys/sys/_sigset.h).
  Signal-code and arm64 layout/copy-back tests witness these source values;
  FreeBSD runtime delivery, landing and fixup recovery remain unverified.
- [AAPCS64 2024Q3 (5 September 2024)](https://github.com/ARM-software/abi-aa/blob/2024Q3/aapcs64/aapcs64.rst),
  §§6.1.1–6.1.2 (preserved registers), “The Frame Pointer”.
- Arm ARM DDI0487J.a (2023), §C5.2.7 (FPCR, FZ and RMode),
  §§C6.2.22–C6.2.23 (AUTIA/AUTIB), and §C6.2 (PACIA/PACIB instruction encodings).
- Arm ARM DDI0487I.a, §D17.2.37, ESR_EL1's EC and Data Abort ISS/DFSC fields;
  [Arm's I.a issue corrections](https://documentation-service.arm.com/static/644c18cbaa78c007af74fa2c)
  identify this register section explicitly.
- Arm ARM DDI0487J.a (2023), §B2.2.1: aligned single-register loads/stores
  are single-copy atomic; a successful exclusive-pair store makes the full
  128-bit update atomic, including writeback of an unchanged comparison value.
  Routed scalar contention and mixed-tier paired-CAS tests witness this.
  DDI0487I.a §B2.3.11 supplies the DMB memory-barrier rules.
- Arm ARM DDI0487I.a (2022), §D17.2.40 (FAR_EL1), corrected in
  [102105_I.a_06 §2.85](https://documentation-service.arm.com/static/644c18cbaa78c007af74fa2c):
  multiple faults from an unaligned crossing access have no prescribed priority.
  Wide copies retry bytes from the failing word. Distinct-byte snapshot/readback
  tests check the readable prefix, and synthetic fixup-entry tests witness both
  copy directions without depending on a particular reported FAR.
- [Arm DDI0602, 2023-06, BTI](https://support.arm.com/documentation/ddi0602/2023-06/Base-Instructions/BTI--Branch-Target-Identification-):
  BTI semantics are **source-unverified**; this reference is unpinned and has no
  independently validated subsection citation. Tests witness the `hint #34`
  encoding (`0xd503245f`) at every routine entry. Execution with guarded text is
  **runtime-unverified**.
- [Linux v6.12 arm64 uapi sigcontext.h](https://github.com/torvalds/linux/blob/v6.12/arch/arm64/include/uapi/asm/sigcontext.h),
  `sigcontext.__reserved`, `_aarch64_ctx`, FPSIMD_MAGIC and `fpsimd_context`.
  The adapter bounds its extension scan to that 4096-byte area.

Intel SDM Vol 3A §7.5 supplies guest fault-state precision and §11.1.2.2 the
guest locked-operation contract; the interpreter's present replay atomicity
limitation is stated above. The Intel manual and x86-64 psABI sources remain the
existing pinned authorities in `oracles/`.


## Amendment — 2026-11-05: Rust guest-transfer fixups

Rust guest transfers can use seventeen naked routines on x86-64 and AArch64:
copy-in/out, read/write probes, 1/2/4/8-byte load/store/CAS, and 16-byte paired
CAS. Six immutable relative offsets in each instruction image describe word and
byte-fallback ranges and fixup labels. The fixed table bounds classification to
seventeen descriptors. Loads and CAS return values in registers; locked RMW
uses a CAS loop. AArch64 entries contain `BTI c` (`hint #34`) before the branch
over their metadata, with the verification limits stated above. Misaligned
scalars use plain accesses; misaligned CAS retains its existing non-atomic
fallback without unaligned exclusive instructions. Expected CAS values are
width-masked on both hosts. Intel SDM Vol 3A §11.1.1 supplies plain scalar
atomicity, §11.1.2.2 locked-operation atomicity, §11.2.3.9 locked ordering, and
Vol 2A CMPXCHG8B/CMPXCHG16B, Operation, paired-CAS writeback. Host Arm sources
are listed above.

The embedder calls the same `jit_abi::fault::classify`. An Access fault in a
fixup range is OURS before any handle dereference, including handle zero or an
inactive live handle. The classifier changes PC and scratch (x9/r11), putting
`si_addr` in scratch. Returning from the handler reaches the fixup, which returns
normally to Rust with the fault address. No Rust frame is skipped; every Drop
runs. The routines do not change FP control. Alignment faults inside routines,
unknown fault reasons, and PCs outside both exact compiled sites and fixup
ranges are NOT OURS. Compiled recovery still requires the owning thread's
active live handle. Install the handler on every thread making routed accesses.
The classifier uses no allocation, locks, TLS or libc.

`Memory::set_host_fault_recovery(true)` opts into routed Rust transfers; the
interpreter chooses a const-generic specialization once per run and helpers
select the policy per call. Routed FX/x87/REP instances are out of line to keep
their frames out of default helper calls. Default mode performs no routing;
registered RAM must remain host-accessible to Rust interpreter, helper and
fetch accesses. Compiled inline faults recover in both modes: protected mapped
RAM returns `Exit::UnmappedMemory` directly, without a Rust replay. Other replay
paths use the run's own policy. Neither mode installs a handler.

The payload-free `MemTrap::HostFault` preserves the compact hot Result layouts.
Its exact address is captured after the routine returns, in an ordinary Rust
thread-local return slot; `fault_addr(transfer)` must be called on that
thread before another failing transfer replaces the slot. The classifier never
accesses it. Helpers return the captured byte and the instruction's access
direction as `Exit::UnmappedMemory`.

Fetch always decodes a private copy. Its first chunk is at most 128 bytes,
bounded by the host page, region and remaining window; later chunks extend only
when iced needs more bytes. Single-instruction fetch is capped at 15 bytes.
Default mode uses `ptr::copy_nonoverlapping`; routed mode copies words and
retries bytes from a faulting word. A failed copy retains the valid prefix and
an incomplete instruction reports the first unavailable byte. No cache lock is
held across the copy. `code_slice` returns `Result<Vec<u8>, MemTrap>`.

Interpreter and compiled vector emitters stage every guest load before any
result destination commits. Compiled VPSIGN stages all lane results; compiled
VPMADD loads the high operand before its first committing helper call. Routed
Rust copies and vector stores preflight before writing, including selected EVEX
opmask elements and narrowing stores. Interpreter selected-element preflights
check every guest region span and refine an unmapped span to its first unmapped
byte, separately from the host-page probes. VMASKMOV uses its local per-element
region preflight in default mode and the shared preflight in routed mode, once
per selected element. Routed stores then probe each touched host page;
guest region boundaries need not align to host pages. Inactive mask elements
cause no probes; each selected host page receives one write probe. A write probe
is an atomic same-value write of one byte: guest data is unchanged, but it
breaks copy-on-write and sets dirty/soft-dirty tracking even on earlier pages
when a later page faults. Preflight prevents partial Rust writes under stable
protections; it cannot exclude concurrent protection revocation between probe
and commit. An embedder requiring all-or-nothing writes must serialize
revocation with the whole transfer.

Selected MMIO elements before an unmapped element deliberately differ by mode
for EVEX k-masked stores, including byte-element stores, and VPMOVQD narrowing
stores. Default mode issues the preceding `Exit::MmioWrite` transfers and then
faults on the unmapped element. Routed mode checks every selected region first
and faults before issuing any device write. This commit-last behavior matches
VMASKMOV's region preflight in default mode.

Compiled split stores have a known gap, tracked in **xj-obx.8**. Their code is the
same in both modes. On x86-64, 16-byte vector transfers are two 8-byte stores;
32/64-byte vector and extract-lane stores contain multiple such transfers. On
AArch64 a 16-byte transfer lowers to STP, and 32/64-byte and extract-lane stores
contain multiple transfers. The AArch64 gap also covers misaligned scalar
stores that cross a host page. Completed transfers can remain visible when a
later transfer faults; no all-or-nothing promise is made for a faulting STP or
page-crossing scalar store. The scalar-straddle witness records the measured
host's behavior; it does not establish a guarantee for other implementations.
Register/flag recovery still names the guest instruction, and resuming rewrites
the same completed destination bytes before continuing. Intel SDM Vol 3A
§11.1.1 permits partial wide-store visibility. No split-store deferral or inline
atomic probe is emitted by this design.

An x87 memory fault restores the entry status word in both modes (Intel SDM
Vol 3A §7.5). REP MOVS/STOS remains per element in routed mode; completed
iterations, pointers and remaining count are retained, while the faulting
iteration commits nothing (Vol 2B REP, Description). Bulk REP chunking is
outside this design. Routed interpreter spinlock and CAS-increment tests
exercise scalar CAS; an exact-total LOCK ADD counter exercises the RMW CAS loop.
Plain routed scalar contention witnesses 1/2/4/8-byte single-copy atomicity, and
mixed-tier paired contention witnesses 16-byte CAS. Compiled atomics do not
depend on the recovery policy and are covered by the existing JIT tests.

Native runtime validation covers macOS AArch64. Linux x86-64/AArch64 are
cross-checked and x86-64 naked routines are object-emitted; real Linux runtime
validation remains **xj-jge.6**. The host Arm references remain unpinned as stated
above. Core retains its iced-x86-only dependency boundary.

### Measured cost and opt-in rationale

Release measurements on macOS AArch64 compare clean `88cfccf`, default mode
and routed mode. Three interleaved rounds use 15 record samples and two warmups;
all cells are minima over the three rounds, in milliseconds. Percentages use
these unrounded six-decimal millisecond values. Eager JIT values are the
recorded `jit_stat.min_ns` values.

| Workload | Main interp | Default interp | Routed interp | Main eager JIT | Default eager JIT | Routed eager JIT |
|---|---|---|---|---|---|---|
| fib32 | 934.944625 | 922.145625 | 931.527750 | 67.114084 | 67.029000 | 67.211333 |
| hotloop | 420.929833 | 427.667166 | 425.474042 | 26.125625 | 26.376333 | 26.451167 |
| simd | 43.340875 | 44.432542 | 42.789417 | 11.845542 | 11.860292 | 11.537791 |
| memcpy | 71.836708 | 71.685000 | 75.006083 | 4.122125 | 4.129000 | 4.093833 |
| indirect | 141.036042 | 141.739792 | 141.170250 | 23.247750 | 22.548542 | 22.715791 |

| Workload | Default interp vs main | Default JIT vs main | Routed interp vs main | Routed interp vs default |
|---|---|---|---|---|
| fib32 | -1.37% | -0.13% | -0.37% | +1.02% |
| hotloop | +1.60% | +0.96% | +1.08% | -0.51% |
| simd | +2.52% | +0.12% | -1.27% | -3.70% |
| memcpy | -0.21% | +0.17% | +4.41% | +4.63% |
| indirect | +0.50% | -3.01% | +0.10% | -0.40% |

Routing remains opt-in: interpreter memcpy costs more than 3% relative to both
main and the default path. Default mode has no per-access recovery dispatch in
the interpreter. In these three rounds the increases above 1% are hotloop
+1.60%, SIMD +2.52%, FX JIT +4.62%, three-instruction lift +6.61%,
twelve-instruction lift +4.71%, step_instruction +4.18%, crossing 16-byte stores
+2.14%, and aligned 16-byte stores +2.06%. The corresponding minima appear in
the tables.

A second measurement of default mode used five interleaved rounds of
`x86jit-bench record --iters 15`, `88cfccf` against this design, on macOS
AArch64 with background load. The table gives the minimum over the five
rounds, then the change of the minimum and of the median:

| Workload | Interp ms | Interp min / median | Eager JIT ms | JIT min / median | Region-bg ms | Region min / median |
|---|---|---|---|---|---|---|
| fib32 | 895.68 → 913.11 | +1.95% / +1.11% | 65.60 → 66.32 | +1.10% / +0.42% | 63.48 → 64.01 | +0.83% / +0.48% |
| hotloop | 407.65 → 417.88 | +2.51% / +1.23% | 25.30 → 26.01 | +2.81% / +1.20% | 7.38 → 7.47 | +1.22% / −1.18% |
| simd | 41.63 → 41.88 | +0.60% / +1.45% | 11.26 → 11.19 | −0.62% / −0.62% | 9.60 → 9.61 | +0.10% / +0.21% |
| memcpy | 86.86 → 66.69 | −23.22% / −19.97% | 4.01 → 4.07 | +1.50% / +0.00% | 2.37 → 2.32 | −2.11% / −0.42% |
| indirect | 134.02 → 138.95 | +3.68% / +3.16% | 22.05 → 21.59 | −2.09% / −2.70% | 21.07 → 20.46 | −2.90% / −3.18% |

Default mode adds no per-access work. Compiled scalar and store code is
unchanged; only the staged vector emitters differ. What remains has mixed signs.
By median:
- interpreter memcpy is 20% faster;
- interpreter indirect is about 3% slower;
- eager-JIT and region indirect are about 3% faster;
- everything else is within ±1.5%.

These are code-layout differences, assumed (not profiled) to come from the extra
const-generic instances of `walk_ops` and `run_inner`, which change inlining.
They are accepted as such.

Fetch uses a 128-byte first chunk, preserving page/region bounds and on-demand
extension. Timings below are nanoseconds per call (100,000 calls/sample,
minimum of 15, three rounds).

| Loop (ns/call) | Main | Default | Routed | Default vs main | Routed vs main |
|---|---|---|---|---|---|
| lift_3 | 129.93292 | 138.52708 | 138.10458 | +6.61% | +6.29% |
| lift_12 | 375.42708 | 393.12000 | 392.37167 | +4.71% | +4.51% |
| step_instruction | 58.17958 | 60.61000 | 61.14917 | +4.18% | +5.10% |

Helper loops below exclude JIT compilation and use 15 samples per round.
FX is 10,000 FXSAVE+FXRSTOR pairs; x87 is 100,000 FLD+FSTP pairs; REP is
200 repetitions of 4096 bytes (MOVSB) or 512 words (MOVSQ). Times are ms.
Routed REP remains per element; its costs below are part of the opt-in trade-off.

| Loop (ms) | Main | Default | Routed | Default vs main | Routed vs main | Routed vs default |
|---|---|---|---|---|---|---|
| interp fxstate | 0.588166 | 0.548000 | 0.832250 | -6.83% | +41.50% | +51.87% |
| interp x87 | 4.435167 | 4.450042 | 4.518709 | +0.34% | +1.88% | +1.54% |
| interp rep_movsb | 4.769834 | 4.365417 | 5.701459 | -8.48% | +19.53% | +30.61% |
| interp rep_movsq | 0.600000 | 0.576666 | 0.717916 | -3.89% | +19.65% | +24.49% |
| jit fxstate | 0.201166 | 0.210458 | 0.477458 | +4.62% | +137.35% | +126.87% |
| jit x87 | 1.412417 | 1.394459 | 1.982792 | -1.27% | +40.38% | +42.19% |
| jit rep_movsb | 2.567667 | 2.471542 | 3.036834 | -3.74% | +18.27% | +22.87% |
| jit rep_movsq | 0.385500 | 0.384542 | 0.381125 | -0.25% | -1.13% | -0.89% |

The compiled-store loops use 100,000 stores and 16 samples per round.
`cross32` crosses a 4 KiB boundary inside a host page; `cross32_16k` crosses
an actual 16 KiB host page; `slide32` advances by 33 bytes per store.
All retire zero interpreter instructions. The 16/32-byte store block dumps are
348/360 bytes and match `88cfccf` in both modes apart from four MOVZ/MOVK
words containing embedded pointers.

| Loop (ms) | Main | Default | Routed | Default vs main | Routed vs main |
|---|---|---|---|---|---|
| cross32 | 0.2803 | 0.2828 | 0.2828 | +0.89% | +0.89% |
| cross32_16k | 0.2803 | 0.2828 | 0.2828 | +0.89% | +0.89% |
| aligned32 | 0.2803 | 0.2828 | 0.2828 | +0.89% | +0.89% |
| slide32 | 0.3233 | 0.3233 | 0.3232 | +0.00% | -0.03% |
| cross16 | 0.2712 | 0.2770 | 0.2770 | +2.14% | +2.14% |
| aligned16 | 0.2714 | 0.2770 | 0.2770 | +2.06% | +2.06% |
| slide16 | 0.3179 | 0.3181 | 0.3166 | +0.06% | -0.41% |
