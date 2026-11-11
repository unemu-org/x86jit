# Dense JIT code memory — decision-17 phase 2

Implementation and validation on 2026-11-10. Phase 1 baseline: `c161085`.
**MEASURED** means executed locally; **READ** means checked in source or API
contracts; **ASSUMED** identifies platform behavior still requiring native coverage.

## Provider startup fallback and diagnostics (READ)

Provider selection is lazy with respect to `JitBackend` construction: the first
compiler module probes a real 128 MiB executable region before handing the
provider to Cranelift. A successful probe retains that region for allocations;
there is no disposable mapping probe. This avoids reserving memory for backends
that never compile, and selects the relocation policy and ownership before any
pointer can escape or code can be published.

Both MXCSR modules share a serialized, permanent backend-wide choice. If the
first alias region cannot be created, Linux/FreeBSD try anonymous W^X storage
before Cranelift's `SystemMemoryProvider`. The anonymous startup probe checks RW
commit and RX sealing on the first page of its retained 128 MiB reservation;
that empty sealed page is counted as used and never reopened. A failed probe
unmaps the complete reservation. On macOS the existing MAP_JIT-to-system fallback
is unchanged, including unsupported per-thread write protection. Partial alias
mappings and backing handles are cleaned up by the failed reservation path.
Once a startup probe succeeds, even in the other module and before publication,
later allocation/write/finalization errors retain
the existing sticky `CodeSpaceExhausted` behavior. There is no switch after a
successful first region and no mixed dense/system ownership within a backend.
`with_system_memory_provider()` explicitly selects the same system provider
without probing or counting an automatic fallback. The system provider's fork
inheritance policy and lack of dense-storage accounting apply to either route.

`MemoryStats.backend` retains numeric, Copy diagnostics:

| Field | Meaning |
|---|---|
| `code_provider_fallback` | Automatic switches: at most two on Linux/FreeBSD (aliases → anonymous W^X → system), at most one on macOS |
| `code_first_failure_op`, `code_last_failure_op` | Stable `CodeMemoryFailureOp` numeric IDs; zero means no recorded operation failure |
| `code_first_failure_errno`, `code_last_failure_errno` | Original raw errno, or zero if the API supplies none |

The JIT crate exports the non-exhaustive `CodeMemoryFailureOp` enum and Copy
`CodeMemoryFailure { operation, errno: Option<i32> }` value.
`JitBackend::code_first_failure()` and `code_last_failure()` return typed values
without taking a compiler lock. Operation/errno pairs are stored atomically;
MemoryStats fields remain independent snapshots. First failure is never
replaced. Both failures and the fallback count survive successful compilation.
`code_allocation_failures` includes the failed startup probe, even when fallback
recovers. System-provider allocation, protection, cache and pipeline failures also
record operation/errno diagnostics and increment this counter once per failed
allocation or finalization. All return `ModuleError::Allocation`, selecting sticky
`CodeSpaceExhausted` even for permission errors. The vendored system provider adds
`finalize_with` to wrap each publication operation; it preserves raw OS errors
(including those carried by `region::Error::SystemCall`) and removes cache/pipeline
`expect` calls from this path. The backend adapter supplies diagnostics and test
injection.
System-provider code storage counters stay zero while emitted bytes and finalized
unit counters continue to work.

Operation IDs distinguish object creation (`memfd_create`/`shm_open`), object
sizing (`ftruncate`), `PROT_NONE` address reservation, RW and RX/R mapping,
`madvise`/`minherit` exclusion, macOS support query, MAP_JIT mapping, Mach read-only
remapping and `mprotect`, write-protect disable/restore, cache maintenance and
pipeline synchronization. `CommitAnonymous` (15) and `SealAnonymous` (16)
distinguish anonymous page commitment and sealing. `AnonymousPersonality` (17)
identifies Linux personality query errors or incompatible `READ_IMPLIES_EXEC`;
`SystemAllocate` (18), `SystemProtectExecutable` (19) and
`SystemProtectReadOnly` (20) identify system-provider operations. Cache maintenance
and pipeline synchronization reuse their existing IDs; existing IDs remain stable.
Cleanup calls do not overwrite the originating error.
Mach return codes and unsupported-operation errors supply no errno; the void
macOS toggle/cache APIs cannot report native errors. Test-only injection at the
operation boundaries also witnesses their error paths without changing production
behavior. Linux arm64 registration still independently selects its existing
SYNC_CORE/receiver protocol; a failed optional registration is not an allocation
or publication failure.

**MEASURED** embedder report at `73ed5d6`: FreeBSD stable/15 arm64 processes with a
non-native syscall personality inside a jail each had one allocation failure,
zero regions/committed bytes, and background code-space failures. The failing OS
operation was unavailable in that revision. **ASSUMED:** the system provider can
allocate executable storage in that deployment; native confirmation with these
new diagnostics is still required. Anonymous storage now offers a path without
backing descriptors; native confirmation remains required. Cross-target checks do not establish native
FreeBSD execution.

Startup-fallback validation on macOS arm64 (**MEASURED**, 2026-11-10): backend
library 132 passed/four skipped, code-memory bound two passed (including 100,000
units), memory stats three passed, core library 242 passed, and Unicorn-enabled
`region_fault_state` 26 passed. The injected witnesses cover startup fallback,
execution in both modules, background publication, first/last errno retention,
over-aligned mapping errors, and later allocation/write/finalization exhaustion.
All-target/all-feature clippy with `-D warnings` passed on Rust 1.95. Linux
workspace/test checks passed for both ISAs on Rust 1.96.1. Both FreeBSD backend
checks passed on installed Rust 1.98.1, which reported four warnings about the
unchanged libc declarations in `libcalls.rs`. These are type checks, not native
Linux/FreeBSD execution. Cargo ran offline with two build/test threads, each
invocation under `timeout 180`, with no benchmarks.

## Descriptor-free storage and policy (READ)

```rust
use x86jit_cranelift::{CodeMemoryPolicy, JitBackend};

let backend = JitBackend::new()
    .with_no_fd_memory_provider(CodeMemoryPolicy::WriteXorExecute);
```

Configure before sharing or compiling, as with `with_system_memory_provider()`.
Linux/FreeBSD explicit selection never switches providers: any reservation,
commit, write or publication failure enters the existing sticky
`CodeSpaceExhausted` path. Automatic fallback selects only `WriteXorExecute`;
`ReadWriteExecute` always requires explicit embedder selection.

Linux/FreeBSD reserve one kernel-placed anonymous `PROT_NONE` mapping per region,
without descriptors or `MAP_FIXED`. Code, read-only data and writable data reuse
the existing per-kind bump allocator, allocation identities and region sizing.
Alignment is taken from the reservation's actual address, including alignment
above page size. Full regions grow by further reservations; live addresses stay
fixed. Only newly needed pages are committed using `mprotect`. The complete
reservation receives `MADV_DONTFORK` (Linux) or `minherit(INHERIT_NONE)` (FreeBSD)
once; protection changes preserve that exclusion.

The reservation uses `MAP_SHARED | MAP_ANON` with descriptor `-1`. One anonymous
shared backing avoids creating private COW identities for successive writable
frontiers, which can inhibit Linux VMA merging (READ:
[Linux `is_mergeable_anon_vma` and shared-anonymous setup](https://github.com/torvalds/linux/blob/master/mm/vma.c)).
FreeBSD's anonymous path passes a null object directly to `vm_mmap_object`,
without looking up or creating a process descriptor (READ:
[`kern_mmap`](https://github.com/freebsd/freebsd-src/blob/main/sys/vm/vm_mmap.c)).
Native Linux/FreeBSD map-entry counts remain ASSUMED until their bound test runs.

Before each strict anonymous page commitment (including the automatic startup
probe), Linux queries `personality(0xffffffff)` without changing the caller's
personality. A query failure or `READ_IMPLIES_EXEC` refuses commitment and records
`AnonymousPersonality`; policy rejection has no errno. Automatic selection proceeds
to the system provider, while explicit W^X selection returns sticky
`CodeSpaceExhausted` without switching. Rechecking each new commitment covers a
caller changing its personality between publication batches. Explicit RWX storage
does not impose this check. **READ:** [personality(2), DESCRIPTION](https://man7.org/linux/man-pages/man2/personality.2.html)
and [Linux v6.12 `do_mprotect_pkey`](https://github.com/torvalds/linux/blob/v6.12/mm/mprotect.c#L746)
specify the query and permission expansion. Production performs no map query or
other descriptor-creating protection verification. Test-only `region::query`
checks actual RW, RX and inaccessible frontier protections (Linux uses
`/proc/self/maps`); a thread-local hook forces the incompatible personality on
all test hosts without changing the test process's real personality.

Under `WriteXorExecute`, each region tracks a sealed prefix, committed frontier
and unused reservation. Functions in the same publication batch pack densely
into RW pages. After copying/relocation and cache maintenance, publication seals
only the new pages RX and advances allocation to the next page. A batch's partial
page tail remains used storage. No operation changes an executable page back to
RW, removes execute permission from published pages, or writes through an alias.
Read-only data follows the same RW-to-R sealing rule; writable data stays RW.
Protection failures block publication and retain their original operation/errno.
The existing cache, pipeline and receiver-generation protocol is shared unchanged.

`ReadWriteExecute` commits code pages RWX, retaining every partial-page tail
across publications, with no policy-imposed page waste. It permits writes to live
executable code and gives up W^X/code-integrity protection. An embedder must
explicitly accept that security trade-off, and the host must allow RWX pages.
Data retains its R/RW policies. Neither policy currently recycles allocations.

On macOS both public policy choices retain the existing descriptor-free MAP_JIT
provider, per-thread write protection, process reservation and startup fallback.
The anonymous frontier implementation is also exercised directly by internal
W^X witnesses on macOS, but is not the public macOS provider. The initial private
anonymous prototype left one map entry per publication batch on macOS, and its
anonymous RWX commit failed with `EACCES`; that experiment did not replace the
public MAP_JIT policy. The final anonymous allocator uses shared backing.

Anonymous code counters count one mapping view per reservation. Protection
boundaries can create multiple OS map entries, so `code_mapping_views` is not a
kernel map-entry count. Used bytes include alignment, veneer capacity, sealed
tails and the automatic startup probe page. Committed bytes count successfully
committed high-water pages once, including sealed tails; free/largest extents
exclude sealed tails. Region/offset/length allocation records and separate
sealed/committed boundaries remain available to reclamation. Future reuse must
prove that **every page in the reopened extent** is no longer executable by any
vCPU, remove stale code metadata, and pass both grace periods before making it
RW. No code-page reopening or reuse is implemented here.

### Review safeguards validation (2026-11-11)

**MEASURED — native macOS arm64, Rust 1.95:** backend library 142 passed/four
skipped; code-memory bound four passed, including a single 100,000-unit invocation
(31.91 seconds); memory stats three passed; core library 242 passed; Unicorn-enabled
`region_fault_state` 26 passed. The new injected witnesses cover incompatible
personality and query failure before strict commitment, automatic fallback,
explicit strict refusal, preservation of actual RX/inaccessible pages, system
allocation/RX/cache/pipeline failure after explicit or automatic selection, sticky
backend exhaustion, and system read-only protection failure retaining its errno.
Workspace all-target/all-feature clippy with `-D warnings` and workspace/vendor
format checks passed. Each test invocation selected one binary, two test threads,
and a 180-second timeout; Cargo ran offline with two build jobs. No benchmarks ran.

**MEASURED — compilation only:** workspace/test checks passed for x86-64 and
arm64 Linux on Rust 1.96.1. Backend-library checks passed for both FreeBSD targets
on Rust 1.98.1, with four existing warnings about libc declarations in
`libcalls.rs`. Nix devShell entry was sandbox-blocked; installed toolchains and
SDK were used. **ASSUMED:** native Linux/FreeBSD execution, including a real Linux
`READ_IMPLIES_EXEC` process; the forced-flag witness uses a test-only hook.

### Descriptor-free validation (2026-11-10)

**MEASURED — final native macOS arm64 checks:** backend library 138 passed/four
skipped; memory stats three passed; core library 242 passed; Unicorn-enabled
`region_fault_state` 26 passed. The native library run covers injected alias
creation failure selecting anonymous W^X storage, execution in both MXCSR
modules, both failed startup probes retaining first/last diagnostics, anonymous
reservation/commit/seal/publication failures, page protections and no RX-to-RW
transitions, alignment/growth, and exclusion/rebuild through child and grandchild.
Test-only alias selection exercises the Unix fallback on macOS; production macOS
still uses its original MAP_JIT-to-system route.

**MEASURED — mapping checks:** the corrected public no-fd option used one code
map entry for 512 foreground units and one for 512 batched units; each sample had
32,768 resident code bytes (16,384-byte pages). Those are MAP_JIT measurements on
macOS. Cross-thread publication of 1,000 units passed. These three smaller bound
binary tests passed after correcting the macOS routing.

The sole 100,000-unit invocation ran the initial private-anonymous macOS
prototype, finished in 47.71 seconds, and **failed** the mapping bound: 1,563 code
entries, 25,608,192 resident code bytes, 97,091,584 process RSS bytes. That result
motivated preserving macOS MAP_JIT. It is not a passing result for the final
implementation. The 100,000-unit test was not repeated under the requested
single-invocation load limit; its final version uses the same batched helper as
the passing 512-unit check. The test now accounts for one partial-page tail per
64-unit publication batch on anonymous W^X hosts.

**MEASURED — build checks:** Rust 1.95 all-target/all-feature clippy with
`-D warnings` and formatting passed. Rust 1.96.1 workspace/test checks passed for
both Linux ISAs. Rust 1.98.1 backend/test checks passed for aarch64 and x86-64
FreeBSD, with the same four pre-existing libc declaration warnings in
`libcalls.rs`. Cargo ran offline with two build/test threads and `timeout 180`
per invocation. No benchmarks ran.

**ASSUMED:** native Linux/FreeBSD execution, RWX execution/packing on those
hosts, their map-entry bounds and fork behavior, and the final 100,000-unit
result remain unmeasured. Cross-target type checks do not establish those facts.

## Storage and relocation design (READ)

- `x86jit-cranelift/src/code_memory.rs:115`: dense bump allocation by kind, retaining
  every unused tail across foreground and background finalize calls. Default
  logical regions are 128 MiB; oversized requests grow by a page-rounded larger
  region. Both writable and runtime addresses respect requested alignment, also
  above page size. Growth never moves live entries.
- `code_memory.rs:312`: Linux uses a close-on-exec memfd and separate shared RW/RX
  views. `code_memory.rs`: FreeBSD uses `shm_open(SHM_ANON)` and the same view
  machinery. Its `ftruncate` reserves swap for the entire backing object: normally
  **128 MiB per region**, including unused tails, separately for each allocation
  kind. FreeBSD's `shm_dotruncate_locked` calls `swap_reserve_by_cred` for
  the grown object size ([source](https://github.com/freebsd/freebsd-src/blob/main/sys/kern/uipc_shm.c)).
  The two aliases share that one swap reservation. Sparse physical residency
  does not remove this reservation; low swap can fail allocation before RAM is
  exhausted. RO data has RW/R views; writable data has one RW view. File handles
  close after mapping; partial mapping failures unmap their successful views.
- `code_memory.rs`: macOS first checks
  `pthread_jit_write_protect_supported_np()`. Unsupported hosts (including Intel
  Macs without per-thread protection) select Cranelift's default SystemMemoryProvider,
  which fills RW pages then seals them RX and starts fresh allocations. They do
  not create a W+X MAP_JIT mapping and do not promise dense memory bounds. Provider
  region/reservation counters are zero on that fallback; successful emitted-byte
  and unit counters remain available. The dense provider also refuses MAP_JIT
  allocation directly when protection is unsupported. Hardened Intel embedders
  must also permit the default allocator's non-MAP_JIT executable pages under
  Apple's [unsigned executable memory policy](https://developer.apple.com/documentation/BundleResources/Entitlements/com.apple.security.cs.allow-unsigned-executable-memory),
  or keep execution in the interpreter. That deployment remains unmeasured.
- On supported macOS hosts, x86jit reserves one lazy process-wide **1 TiB MAP_JIT**
  extent, shared by all backend instances and both MXCSR modules. Logical regions
  grow inside it. The large virtual reservation avoids exhausting repeated
  backend construction before phase 3; it is not committed or resident memory.
  Exhaustion, including reaching this fixed process capacity, returns an allocation
  error through the existing sticky `CodeSpaceExhausted` route. Reservations and
  their contents remain retained at ordinary backend drop.
- macOS writes use `pthread_jit_write_protect_np(0/1)` on the compiling thread.
  An RAII guard restores execute protection on unwind. The callback trampoline,
  arbitrary closure context and static callback allowlist are removed. Compiler
  write scopes do not nest or call generated code. Hardened-runtime embedders
  must enable `com.apple.security.cs.allow-jit` and **omit**
  `com.apple.security.cs.jit-write-allowlist`, which disables the direct toggle.
  An embedder requiring callback-only writes must use a separate compiler process
  with compatible entitlements; this provider does not support that policy in
  process. Entitled deployment remains unmeasured. RO data uses an anonymous
  Mach shared remap; writable data uses anonymous RW memory outside MAP_JIT.
- `vendor/cranelift-jit/src/memory/mod.rs:31`: narrow integration adds runtime-pointer
  translation, scoped copying/patching and a provider relocation allowlist. The
  provider retains the translation in its region/allocation state. The vendored
  package is explicitly named `x86jit-cranelift-jit` and referenced by path so an
  embedder's dependency graph also uses it; a root-only Cargo patch would disappear
  when this repository is consumed as a dependency. The upstream licence and
  checksum are recorded in `PROVENANCE.md`.
- `vendor/cranelift-jit/src/compiled_blob.rs:163`: relocation writes use RW;
  arithmetic uses RX sites and RX targets, including veneer branches and pointer
  literals. `CompiledBlob::ptr` exposes RX for finalized functions, data, symbols,
  perf entries and code ranges. Module relocation logic and the closed libcall
  resolver remain in use. Pending function/data lists are cleared only after
  write access, relocations and provider finalization all succeed. Either failure
  makes the module terminal: all function/data/address getters and later
  finalization attempts are blocked, including getters for earlier definitions.
  Relocation writes are never retried after partial progress. The provider accepts
  Abs4/Abs8, x86 PCRel4/call PCRel4,
  Arm64Call and the AArch64 page-high/add-low pair. Other kinds are rejected before
  allocation/copy/publication. Byte witnesses independently record reservation RW/RX pairs, check returned entries
  against RX (and against RW), and exercise absolute and cross-unit
  PC-relative targets, direct Arm64 calls and far veneers with actual nonzero
  RW/runtime offsets, including on this macOS host (the test aliases are RW/R and
  are inspected rather than executed).
- `x86jit-cranelift/src/lib.rs:3089`, `:3329`, `:3882` and `src/libcalls.rs:99`:
  both lazy MXCSR modules and libcall builders use this provider. Background
  definition batching, foreground finalization, saved compiler metadata,
  decision-13 recovery and the VFloatMov workaround are retained.

## Host fork contract (READ)

The dense arena and its compiled entries are unavailable in a host-fork child.
Embedders must follow `Vm::shutdown`: quiesce vCPUs, shut down all engines before
fork, then build independent engines in both processes. This contract also covers
code retained after shutdown; retaining addresses in the parent does not make
those addresses usable in the child.

On macOS the complete 1 TiB MAP_JIT reservation receives `VM_INHERIT_NONE` once,
at reservation time. Allocation and finalization never change inheritance or
split the reservation into per-page map entries. Fork neither copies nor shares
its code backing. The process arena checks its immutable owner PID before taking
an allocator mutex. A child installs a new process-local allocator and reserves
new backing, even if the inherited allocator mutex was locked. Grandchildren
follow the same rule; virtual addresses may be reused by the OS.

Linux sets `MADV_DONTFORK` on each complete memfd view; FreeBSD sets
`INHERIT_NONE` on each complete shm view. macOS data mappings also exclude both
aliases from inheritance. These settings happen only when a view is created,
with cleanup on failure, and preserve the mapping-count bound. Each dense
provider records its creator PID: inherited allocation, write-window entry and
finalization return errors before accessing mapped storage. Pointer translation
panics at the ownership check because that trait method cannot return an error;
inherited release is a no-op so it cannot unmap a child's new mapping at a
reused address. Previously saved raw code pointers must never be called in a
child; provider checks cannot intercept a raw indirect call.

The process-wide code map also records its creator PID. The first child code
registration atomically selects a fresh map before taking any writer mutex, so
reused virtual addresses cannot collide with inherited metadata and vanished
threads cannot leave the new map's locks or reader counters pinned. Signal
lookups only load the current retained map and never check the PID, allocate or
lock. Until that first child registration, inherited code from the system-provider
fixtures can use its inherited metadata. Parent maps remain allocated but become
inactive in a rebuilding child; translation reclamation remains separate.

`JitBackend::with_system_memory_provider()` explicitly selects Cranelift's
default allocation policy for tests and fork-heavy embedders. It applies to both
MXCSR modules, must be selected before compiling/sharing, and gives up the dense
storage counters and mapping bounds. The two `region_fault_state` fixtures use
it to preserve their parent-compiled-entry fault-state checks after fork. It
does not make inherited worker/VM state safe to reuse.

The bounded subprocess witness populates 128 MiB before fork, queries the child
for absent code/data views before rebuilding, rejects inherited provider
operations, compiles and executes a fresh child VM, then repeats in a grandchild.
A pipe forces parent-first allocation, followed by overlapping parent allocation
and child reconstruction. A process-group watchdog bounds the entire worker to
20 seconds, including fork itself and all descendants. On macOS the parent
allocator mutex is deliberately left locked in the child to verify that PID
selection happens before any inherited lock is taken. The separate 100,000-unit
mapping/residency witness covers dense allocation without inheritance splits.
Native macOS results are measured below; Linux/FreeBSD execution remains ASSUMED
pending native validation.

## Publication protocol (READ; locally exercised on arm64)

`code_memory.rs:435` cleans the writable AArch64 view and invalidates executable
cache lines. Linux/FreeBSD use the pinned Wasmtime cache helper. At provider initialization,
Linux arm64 registers `MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED_SYNC_CORE` (64).
On success it requires `MEMBARRIER_CMD_PRIVATE_EXPEDITED_SYNC_CORE` (32) at each
finalization; a later error blocks publication. Registration failure selects the
receiver protocol below. The provider never calls the pinned pipeline helper on
Linux arm64: its EINVAL-to-GLOBAL fallback orders memory without synchronizing
other cores' instruction streams. macOS uses Apple's
`sys_icache_invalidate`, as specified in its JIT porting guide.

The pinned Wasmtime helper's macOS/FreeBSD pipeline operation is a no-op.
`x86jit-core/src/jit_abi.rs` publishes a generation after cache maintenance on
macOS, FreeBSD and Linux arm64 (including Linux with working SYNC_CORE),
with a full store-ordering fence before later relaxed pointer stores. At every
shared compiled-entry boundary (`jit_abi.rs:747`), an acquire fence orders link/
return-predictor pointer loads before acquiring that generation. An executing
thread whose last synchronized generation differs issues ISB before entering the
code, then records its acknowledgment. Chain transfers return through this same
boundary. This is a receiver-side instruction visibility protocol, **not** an
execution or signal-reader grace period. A compiler-local ISB alone is not used
as evidence of multithread publication safety. Core gains no dependency.

## Counters (READ)

`x86jit-core/src/memory_stats.rs:8` exposes provider-owned counters. Each module
has separate relaxed atomics, aggregated by the backend without a compiler lock:

| Counter | Meaning |
|---|---|
| `code_regions` | Logical code regions owned by this backend |
| `code_mapping_views` | Two OS views per Linux/FreeBSD code region; one shared macOS process view |
| `code_bytes_reserved` | Exact logical region capacity, including unused tails and pending/failed definitions |
| `code_process_reservation_bytes` | Shared 1 TiB macOS virtual reservation, zero elsewhere; never sum across VMs |
| `code_bytes_committed` | Page-rounded used high-water extents counted once per backing; this is a provider accounting value, not measured physical commitment, RSS, PSS or FreeBSD swap reservation |
| `code_bytes_used` | Allocated extent bytes, including alignment and veneer capacity |
| `code_bytes_free` | Unused tails, with no reclaimed/reusable retired storage in this phase |
| `code_largest_free_extent` | Largest unused tail across the backend's modules |
| `code_allocation_failures` | Provider allocation/finalization failures |
| `code_allocation_metadata_bytes` | Region/allocation vector capacities, excluding allocator overhead |

Existing emitted-byte and unit counters still count successful finalizations.
Failed or unpublished allocations remain accounted in used/reserved storage.
Allocation records preserve region index, offset and length; their stable vector
index identifies a generation for this no-reuse phase (`code_memory.rs:34`).
Retirement events, reclaimable extents, reuse, metadata removal and module-table
retirement remain phase 3. No changes were made to `x86jit-core/src/codemap.rs`.

## Memory measurements (MEASURED)

Apple M5 Pro, macOS arm64, 16,384-byte host pages. Release measurements used the
same installed Rust toolchain (rustc 1.98.1) for baseline and implementation;
Rust 1.95 was used for the explicit clippy/MSRV gate. Baseline sources came from
an archive of `c161085`, without checkout/worktree changes. Added read-only probes
use `mach_vm_region_recurse` leaf entries and `task_info` RSS. Linux's probe counts
`/proc/self/maps` and reads smaps, matching backing object IDs so both alias views
and their PSS are included. Linux/FreeBSD numbers were not measured locally.

The direct `code_memory` example uses lifted `mov eax,7; hlt`, Speed optimization,
unique guest entries, and immediate per-unit finalization. The existing
`memory_stats` example uses background tiering, 102 vCPUs and
`mov eax,7; inc rax; hlt`; its batching was already reducing baseline page waste.
All byte values below are decimal bytes. RSS is the compiled checkpoint, before
invalidation; later checkpoints can differ because retained metadata/allocators
and the probe itself also consume memory.

| Workload | Units | Emitted bytes (both) | Before maps | Before resident code | Before process RSS | After maps | After resident code | After process RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | 10,000 | 424,640 | 10,000 | 163,840,000 | 202,457,088 | 1 | 425,984 | 35,864,576 |
| direct | 100,000 | 4,384,640 | 100,000 | 1,638,400,000 | 1,969,192,960 | 1 | 4,390,912 | 332,627,968 |
| background | 10,000 | 824,640 | 222 | 3,637,248 | 53,968,896 | 1 | 835,584 | 50,528,256 |
| background | 100,000 | 8,384,640 | 1,759 | 28,819,456 | 455,966,720 | 1 | 8,388,608 | 436,994,048 |

For the direct 100,000-unit run, provider counters are: one logical region,
134,217,728 reserved bytes, 4,384,640 used/emitted bytes, 4,390,912 committed bytes,
129,833,088 free bytes (also the largest extent), 3,145,920 bytes of allocation
metadata capacity, and zero failures. The whole-process MAP_JIT reservation is
1,099,511,627,776 virtual bytes with one mapping view. The background 100,000-unit
run commits 8,388,608 bytes for 8,384,640 emitted bytes. No alignment/veneer bytes
are hidden in these emitted counters; these small arm64 functions need neither
padding nor external call veneers.

**MEASURED negative baseline:** the direct 100,000-unit baseline exceeds the test's
four-entry bound (100,000 entries), resident-code bound (1,638,400,000 versus
6,001,024 allowed), and total-RSS bound (1,969,192,960 versus 476,708,864 allowed).
The new debug-profile 100,000-unit test passes these bounds. Linux permits up to
two resident views of the dense-byte bound while its summed PSS must remain below
the once-counted committed backing plus one page. FreeBSD uses the same two-view
allocator design but does not yet have the process-memory probe.

## Performance measurements (MEASURED)

The committed perf baseline belongs to a different Linux machine, so the normal
gate reports a host mismatch and skips. For a comparable local run, the existing
benchmark's unmodified `run_workloads(9, 2, false)` was invoked from a temporary
measurement command, first with baseline dependencies and then with current
dependencies. No benchmark workload or committed baseline was changed. Median
JIT wall time includes load, compilation and execution; each row used nine
samples and two warmups. These are two local invocations, not confidence intervals.

| Workload | Before JIT ms | After JIT ms | Raw JIT delta | Delta normalized by interpreter |
|---|---:|---:|---:|---:|
| fib32 | 98.322 | 100.727 | +2.45% | +4.11% |
| hotloop | 43.693 | 45.259 | +3.58% | +1.63% |
| mmx | 6.635 | 6.590 | -0.67% | +3.14% |
| simd | 11.601 | 10.820 | -6.73% | -0.80% |
| memcpy | 6.078 | 5.976 | -1.68% | +2.78% |
| indirect | 34.339 | 26.909 | -21.64% | -17.71% |

A subsequent invocation of the **existing perf gate**, using the measured main
record as a temporary host-local baseline and forcing the gate active, passed.
Its normalized deltas were fib32 +0.3%, hotloop +3.0%, mmx +3.5%, simd -0.0%,
memcpy +3.3%, indirect -7.3%; no metric exceeded 10%. The repository baseline and
benchmark sources remain unchanged. In particular the indirect timing improvement
is not claimed as a stable allocator speedup; its baseline dispersion was larger.

## Validation (MEASURED)

- `cargo +1.95 nextest run --offline -p x86jit-core -p x86jit-cranelift`: 391
  passed, five skipped, including the 100,000-unit map/resident/RSS bound (42.6 s).
  The subsequently added relaxed cross-thread handoff witness passes separately:
  an already-running execution thread consumes 1,000 newly compiled entries.
- Requested Unicorn-feature jit, bg_tier, guest_trap_sites, fault_metadata and smc
  suites: 364 passed, two skipped on the final run. No Unicorn SIGILL was observed
  in this requested suite; platform-specific native-x86 execution remains a CI job.
- Additional fault_landing_pad, fault_path, rust_memory_faults,
  exit_request_signal, code_exhaustion, fences and mxcsr_modes suites: 102 passed,
  two skipped; `mxcsr_modes_sse_matches_unicorn` aborted with SIGILL. This is a
  failed local oracle test, not a passed suite. The identical isolated test on
  unchanged baseline `c161085` also aborts with SIGILL (0.013 s). The failure
  therefore predates this provider; native oracle coverage remains incomplete in
  this sandbox. All 102 non-aborting tests in the additional run passed.
- Guest-agnostic dependency boundary witness: passed; core still depends only on
  iced-x86.
- `cargo +1.95 clippy --offline --all-targets --all-features -- -D warnings`: pass.
- `cargo +1.95 fmt --all -- --check`, `git diff --check`: pass.
- Offline workspace/tests checks for x86_64-unknown-linux-gnu and
  aarch64-unknown-linux-gnu using the Nix toolchain and its matching rustc/sysroot:
  pass. Initial cross attempts selected rustup's rustc and lacked those targets;
  explicitly selecting the matching Nix rustc corrected that environment issue.
- `cargo deny --offline --all-features check licenses bans sources`: all gates pass;
  the existing Unicorn GPL identifier deprecation warnings remain.

The Nix shell launcher and Backlog CLI could not create their shared cache/git
lock files in this sandbox. Existing Nix binaries/sysroots were used directly;
this guide was written at its repository-native design path after the doc CLI's
lock failure. No approval, beads, commit, push, merge or worktree operation was used.

## Remaining coverage and risks

**READ:** storage is intentionally retained. No per-unit release, codemap reuse,
execution/signal-reader grace period, empty-region release or decommit exists in
this phase. Per-backend logical reservations can exhaust the fixed shared macOS
extent after sufficiently many short-lived backends; exhaustion fails closed.
A contiguous 1 TiB reservation may fail in a fragmented/limited address space.
BTI requests currently fail closed; x86jit's current ISA does not request them.
OSes outside Linux, FreeBSD and macOS reject the dense probe and select the system provider.
Before a crates.io release, the integration must either land in an upstream pin
or the separately named vendored package must be made available by the maintainer;
path/git consumers already use the local integration.

**ASSUMED, requiring native confirmation:** Linux memfd aliases and cache/pipeline
publication on both Linux ISAs, and FreeBSD shm aliases/cache behavior. Linux
cross-checks type-check these paths but do not execute them. The alias relocation
witnesses on macOS inspect shared byte views; they do not prove native Linux code
execution. The 100,000-unit test is enabled for native Linux CI on either ISA.

**Unmeasured:** hardened-runtime entitlements and the macOS Intel fallback,
FreeBSD mapping/RSS/PSS bounds, and full execution/signal/publication stress over
many vCPUs (phase 4). The real-program ladder requires a Linux userland environment
and was not run on this macOS host. No upstream patch/PR was published.

Primary API references: [Apple's JIT porting guide](https://developer.apple.com/documentation/apple-silicon/porting-just-in-time-compilers-to-apple-silicon),
[Apple cache(3)](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/sys_icache_invalidate.3.html),
and the pinned Cranelift/Wasmtime 0.135.8/48.0.8 sources described in decision-17.


## Review corrections and platform behavior (2026-11-10)

**READ** API contracts and source; **MEASURED** tests on macOS arm64;
**ASSUMED** native behavior on the other hosts pending CI/native execution.

| Host | Storage / write protection | Instruction publication | Evidence |
|---|---|---|---|
| macOS arm64, protection supported | One MAP_JIT arena, same RW/RX address; scoped thread toggle, unwind guard | OS cache invalidation + receiver generation/ISB | MEASURED ordinary process; READ entitlement requirements |
| macOS x86-64, protection unsupported | Default Cranelift RW then RX, no MAP_JIT; dense counters unavailable | Default provider finalization | READ selection and provider; ASSUMED native execution |
| Linux arm64 with SYNC_CORE | Shared memfd RW/RX aliases | Register 64 at init, require 32 at finalize; receiver generation/ISB also enabled | READ implementation; MEASURED injected command-selection witness; ASSUMED native execution |
| Linux arm64 without SYNC_CORE | Same aliases | Registration failure selects receiver generation/ISB; no GLOBAL fallback | READ implementation; MEASURED injected EINVAL/ENOSYS/EPERM witnesses; ASSUMED native execution |
| Linux x86-64, with or without SYNC_CORE | Shared memfd RW/RX aliases | Existing coherent-instruction-cache path; no arm64 ISB protocol needed | READ pinned platform helper; ASSUMED native execution |
| FreeBSD | Shared shm RW/RX aliases; whole-region swap reservation | arm64 receiver generation/ISB; existing x86-64 coherent-cache path | READ implementation/API; ASSUMED native execution |

The byte relocation fixtures additionally force RW/R aliases on macOS to witness
nonzero alias offsets without executing those bytes. A separate production-code
allocation witness checks RW == RX on supported macOS and RW != RX on Linux and
FreeBSD. Expectations come from reservation records, independently of both
`executable_ptr` and `get_finalized_function`. The finalization failure witness
injects both pre-closure write denial and post-relocation finalization denial,
checks that function, data and address getters remain blocked (including prior
definitions), then removes the injected failure and verifies that finalization
still fails without another write scope or provider finalization.

Primary contracts: [Apple JIT porting guide](https://developer.apple.com/documentation/apple-silicon/porting-just-in-time-compilers-to-apple-silicon)
and [Linux uapi membarrier commands](https://github.com/torvalds/linux/blob/v6.18/include/uapi/linux/membarrier.h).


### Review validation (MEASURED, macOS arm64)

All Cargo commands used offline mode, an isolated temporary target directory and
an empty RUSTC_WRAPPER. The host is macOS arm64 with 16,384-byte pages. The Nix C
linker was selected explicitly (including PATH for a direct-rustc test); otherwise
the host linker and SDK combination failed to link build scripts and the landing
symbol fixture. No repository build configuration was changed for that issue.

- Core/backend nextest: **397 passed, five skipped**, including the 100,000-unit
  bound and relaxed cross-thread publication. A first attempt passed 396 and
  failed the direct-rustc landing-symbol fixture because of the linker environment;
  the complete corrected run passed in 40.953 seconds.
- Requested Unicorn-feature `jit`, `bg_tier`, `guest_trap_sites`, `fault_metadata`
  and `smc` suites: **364 passed, two skipped**. No SIGILL in this run.
- Rust 1.95 all-target/all-feature clippy with `-D warnings`: passed, including a
  final run after negative-control restoration. Format and diff whitespace checks
  passed. The guest-agnostic boundary witness also passed; core still depends
  only on iced-x86.
- Both offline workspace/tests Linux target checks passed with the existing Nix
  1.96.1 compiler and matching sysroot: x86_64 and aarch64. These type-check the
  Linux code, including its real syscall path; they do not execute it.
- Final focused provider witnesses: nine passed. The unsupported-macOS test
  returns early on this supported host; Intel fallback execution is still
  unmeasured. The Linux registration witness injects syscall results rather than
  running a Linux kernel here.
- **Negative controls:** restoring the original vendored pending-list code failed
  the terminal-finalization witness. Making the provider return RW as its executable pointer
  failed both relocation witnesses. Both controls were removed and all nine
  focused tests passed again. Only the intended corrected source remains.

| Repeated workload | Units / vCPUs | Code mappings | Resident code bytes | Process RSS bytes | Reserved bytes | Used / emitted bytes | Page-rounded used bytes | Provider failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Debug bound test, direct finalization | 100,000 / no execution vCPUs | 1 | 4,390,912 | 338,362,368 | 134,217,728 | Not printed | Not printed | 0 |
| Release memory_stats, background tiering | 100,000 / 102 | 1 | 8,388,608 | 434,880,512 | 134,217,728 | 8,384,640 | 8,388,608 | 0 |

The release example finished with 100,000 published entries and no queued or
in-flight compilation. The single process arena reserves 1,099,511,627,776 virtual
bytes; it is not resident memory. Allocation metadata capacity was 3,145,920 bytes.
Code storage stayed retained after invalidation, as required for this phase.
Hardened-runtime deployment, Intel macOS, native Linux and FreeBSD remain
**ASSUMED** where the platform table calls for native confirmation.


### Non-inherited arena validation (MEASURED, macOS arm64, 2026-11-10)

- The 128 MiB populated / 1 TiB reserved fork witness passed under a 20-second
  process-group watchdog, including a deliberately locked inherited allocator,
  parent-first allocation, absent code/data mappings, rejected inherited provider
  operations, and fresh child/grandchild VM compilation and execution. Fork
  returned in **0.807 ms** in the focused locked-allocator run and **2.133 ms**
  during the package run. These are individual local measurements under shared
  host load, not a timing guarantee. The complete worker finished within 0.11 s.
- The grandchild witness initially exposed stale inherited code-map entries when
  new storage reused virtual addresses. PID-aware code-map writer selection fixes
  that failure; the final child/grandchild witness and all 19 focused code-map
  tests pass. Signal lookups keep their lock-free allocation-free path.
- `region_fault_state`: **26 passed**, including the two fixture paths that
  deliberately execute parent-compiled code. Both were exercised individually
  before running the binary. Other `signal::child` callers compile inside the
  child; the native host-instruction oracle uses its own mappings.
- Backend package: **151 passed, five skipped**, with two nextest test threads.
- Existing `bg_tier::shutdown_fork_boundary_jit`: **passed**; shutdown joins the
  compiler before fork and both processes rebuild successfully.
- Rust 1.95 offline all-target/all-feature clippy with `-D warnings`,
  `cargo fmt --check`, and `git diff --check`: **passed**.
- Separate 100,000-unit bound: **passed in 39.161 s**, **one code mapping**,
  **4,390,912 resident code bytes**, **72,908,800 process RSS bytes**. The process
  had 225 VM entries in total; its code reservation was 1,099,511,627,776 virtual
  bytes. There were no per-page inheritance splits. The package run independently
  measured one code mapping and the same resident code bytes.

Every test command used `timeout 300`, offline Cargo, an isolated temporary target
and an empty compiler wrapper; build concurrency and nextest threads were capped
at two. Native Linux/FreeBSD inheritance and execution remain **ASSUMED**.
The Linux flag and FreeBSD inheritance constants were checked against their
primary headers: [Linux mman-common.h](https://github.com/torvalds/linux/blob/master/include/uapi/asm-generic/mman-common.h)
and [FreeBSD mman.h](https://github.com/freebsd/freebsd-src/blob/main/sys/sys/mman.h).
